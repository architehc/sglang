"""NVFP4 (e2m1 + fp8_e4m3 block scales + fp32 global scale) MLA KV cache codecs.

Enabled by env flag SGLANG_NSA_KV_NVFP4=1 on top of --kv-cache-dtype fp8_e4m3
(the CLI plumbing is untouched; the flag reinterprets the NSA MLA KV pool rows).

Per-token / per-layer byte layout (kv_lora_rank=512, qk_rope_head_dim=64,
576 dims total, 16-element quant blocks -> 36 blocks). ROW_BYTES = 328
(vs 656 for the fp8_e4m3 path, vs 1152 for bf16 => 3.51x compression):

    offset  size  contents
    ------  ----  -----------------------------------------------------------
    0       256   nope dims [0,512) packed e2m1, byte j = dim 2j (low nibble)
                  | dim 2j+1 (high nibble); quant block b covers dims
                  [16b, 16b+16), i.e. bytes [8b, 8b+8)
    256     32    fp8_e4m3 block scales for the 32 nope blocks
    288     32    rope dims [512,576) packed e2m1 (same nibble convention)
    320     4     fp8_e4m3 block scales for the 4 rope blocks
    324     4     zero padding (row size 328 = 8-byte aligned; reserved for a
                  possible future per-token fp32 scale)

The nope/rope regions are laid out contiguously as [nope_part(288) |
rope_part(40)] so that the existing two-tensor write kernel
(set_mla_kv_buffer_triton) can be reused unchanged.

Decoded value: x[d] = e2m1(nibble) * fp8_scale(block(d)) * GLOBAL_SCALE.

Global scale convention (matches the standalone validation from this
campaign): GLOBAL_SCALE = GLOBAL_AMAX / (448 * 6)  -- NOT the reciprocal.
Block-scale encode: stored_fp8 = block_amax / (6 * GLOBAL_SCALE), clamped to
[0, 448] so it always fits fp8_e4m3.

GLOBAL_AMAX choice: a per-tensor runtime amax would need a cross-token
reduction + host sync (.item()) and is NOT CUDA-graph capture-safe, and a
"running amax" would make previously written rows decode incorrectly when it
changes. We therefore use a FIXED calibrated constant, read once at import
time from env SGLANG_NSA_KV_NVFP4_GLOBAL_AMAX (default 64.0). Behavior when
miscalibrated is graceful:
  * actual block amax > GLOBAL_AMAX: the fp8 block scale saturates at 448 and
    values clamp to +-GLOBAL_AMAX (hard clip only above the constant);
  * actual block amax << GLOBAL_AMAX: fp8_e4m3 keeps ~2^-3 relative precision
    down to its smallest normal, so precision only degrades once
    block_amax < GLOBAL_AMAX / 28672 (then subnormal fp8 scales take over).
Use nvfp4_kv_bench.py to calibrate against real activations if needed.

This module deliberately has no sglang imports so the offline bench harness
can load it standalone (only torch + triton needed).
"""

import os

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Layout constants (kv_lora_rank=512, qk_rope_head_dim=64)
# ---------------------------------------------------------------------------
QUANT_BLOCK = 16

NOPE_DIM = 512
ROPE_DIM = 64
FULL_DIM = NOPE_DIM + ROPE_DIM  # 576

NOPE_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 32
ROPE_BLOCKS = ROPE_DIM // QUANT_BLOCK  # 4

NOPE_Q_BYTES = NOPE_DIM // 2  # 256
ROPE_Q_BYTES = ROPE_DIM // 2  # 32
PAD_BYTES = 4

NOPE_PART_BYTES = NOPE_Q_BYTES + NOPE_BLOCKS  # 288
ROPE_PART_BYTES = ROPE_Q_BYTES + ROPE_BLOCKS + PAD_BYTES  # 40
ROW_BYTES = NOPE_PART_BYTES + ROPE_PART_BYTES  # 328

# Byte offsets inside a pool row
OFF_NOPE_Q = 0
OFF_NOPE_S = NOPE_Q_BYTES  # 256
OFF_ROPE_Q = NOPE_PART_BYTES  # 288
OFF_ROPE_S = NOPE_PART_BYTES + ROPE_Q_BYTES  # 320

FP8_SCALE_MAX = 448.0
E2M1_MAX = 6.0

_fp8_dtype = torch.float8_e4m3fn


def nvfp4_row_bytes(kv_lora_rank: int, qk_rope_head_dim: int) -> int:
    """Pool row size (bytes/token/layer) for the NVFP4 KV layout."""
    assert kv_lora_rank % QUANT_BLOCK == 0
    assert qk_rope_head_dim % QUANT_BLOCK == 0
    return (
        kv_lora_rank // 2
        + kv_lora_rank // QUANT_BLOCK
        + qk_rope_head_dim // 2
        + qk_rope_head_dim // QUANT_BLOCK
        + PAD_BYTES
    )


def nvfp4_global_amax() -> float:
    # Read via os.environ (not sglang.srt.environ) so this file stays
    # importable standalone; the same variable is registered in environ.py
    # for discoverability. Fixed once per process => capture-safe.
    return float(os.environ.get("SGLANG_NSA_KV_NVFP4_GLOBAL_AMAX", "64.0"))


def nvfp4_global_scale() -> float:
    """GLOBAL_SCALE = GLOBAL_AMAX / (448 * 6). NOT the reciprocal."""
    return nvfp4_global_amax() / (FP8_SCALE_MAX * E2M1_MAX)


# ---------------------------------------------------------------------------
# Triton helpers
# ---------------------------------------------------------------------------


@triton.jit
def _e2m1_decode(c):
    """Decode a 4-bit e2m1 code (int tensor in [0,16)) to float32.

    codes 0..7 -> [0, 0.5, 1, 1.5, 2, 3, 4, 6]; bit 3 is the sign.
    """
    m = c & 7
    sign = tl.where((c & 8) != 0, -1.0, 1.0)
    mag = tl.where(
        m < 2,
        0.5 * m.to(tl.float32),
        (1.0 + 0.5 * (m & 1).to(tl.float32)) * tl.exp2(((m >> 1) - 1).to(tl.float32)),
    )
    return sign * mag


@triton.jit
def _e2m1_encode(t):
    """Encode float32 -> 4-bit e2m1 code, round-to-nearest (ties to even
    significand), input clamped to [-6, 6]."""
    a = tl.minimum(tl.abs(t), 6.0)
    m = tl.where(
        a <= 0.25,
        0,
        tl.where(
            a < 0.75,
            1,
            tl.where(
                a <= 1.25,
                2,
                tl.where(
                    a < 1.75,
                    3,
                    tl.where(a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7))),
                ),
            ),
        ),
    )
    return tl.where(t < 0, m + 8, m).to(tl.int32)


# ---------------------------------------------------------------------------
# Quantize-on-write (used by MLATokenToKVPool.set_mla_kv_buffer)
# ---------------------------------------------------------------------------


@triton.jit
def _quantize_k_cache_nvfp4_kernel(
    k_nope_ptr,  # (num_tokens, 512) bf16
    k_rope_ptr,  # (num_tokens, 64) bf16
    nope_q_ptr,  # uint8 view, row stride nope_part_stride
    nope_s_ptr,  # fp8 view, row stride nope_part_stride
    rope_q_ptr,  # uint8 view, row stride rope_part_stride
    rope_s_ptr,  # fp8 view, row stride rope_part_stride
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    nope_part_stride_0: int,
    rope_part_stride_0: int,
    global_scale,
    NOPE_NBLK: tl.constexpr,  # 32
    ROPE_NBLK: tl.constexpr,  # 4
):
    tok = tl.program_id(0)
    cols = tl.arange(0, 8)[None, :]  # 8 bytes per 16-elem block

    # ---- nope: 32 blocks x 16 dims ----
    rows = tl.arange(0, NOPE_NBLK)[:, None]
    e_off = rows * 16 + cols * 2
    xe = tl.load(k_nope_ptr + tok * k_nope_stride_0 + e_off).to(tl.float32)
    xo = tl.load(k_nope_ptr + tok * k_nope_stride_0 + e_off + 1).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))
    s_raw = tl.minimum(amax / (6.0 * global_scale), 448.0)
    s_fp8 = s_raw.to(nope_s_ptr.dtype.element_ty)
    tl.store(nope_s_ptr + tok * nope_part_stride_0 + tl.arange(0, NOPE_NBLK), s_fp8)

    # quantize against the DECODED scale so write/read are consistent
    s_dec = s_fp8.to(tl.float32) * global_scale
    inv = tl.where(s_dec > 0, 1.0 / s_dec, 0.0)[:, None]
    ce = _e2m1_encode(xe * inv)
    co = _e2m1_encode(xo * inv)
    packed = (ce | (co << 4)).to(tl.uint8)
    tl.store(nope_q_ptr + tok * nope_part_stride_0 + rows * 8 + cols, packed)

    # ---- rope: 4 blocks x 16 dims ----
    rrows = tl.arange(0, ROPE_NBLK)[:, None]
    r_off = rrows * 16 + cols * 2
    re = tl.load(k_rope_ptr + tok * k_rope_stride_0 + r_off).to(tl.float32)
    ro = tl.load(k_rope_ptr + tok * k_rope_stride_0 + r_off + 1).to(tl.float32)

    ramax = tl.maximum(tl.max(tl.abs(re), axis=1), tl.max(tl.abs(ro), axis=1))
    rs_raw = tl.minimum(ramax / (6.0 * global_scale), 448.0)
    rs_fp8 = rs_raw.to(rope_s_ptr.dtype.element_ty)
    tl.store(rope_s_ptr + tok * rope_part_stride_0 + tl.arange(0, ROPE_NBLK), rs_fp8)

    rs_dec = rs_fp8.to(tl.float32) * global_scale
    rinv = tl.where(rs_dec > 0, 1.0 / rs_dec, 0.0)[:, None]
    rce = _e2m1_encode(re * rinv)
    rco = _e2m1_encode(ro * rinv)
    rpacked = (rce | (rco << 4)).to(tl.uint8)
    tl.store(rope_q_ptr + tok * rope_part_stride_0 + rrows * 8 + cols, rpacked)


def quantize_k_cache_nvfp4_separate(k_nope: torch.Tensor, k_rope: torch.Tensor):
    """Quantize bf16 (k_nope, k_rope) into the NVFP4 row layout, returned as
    two uint8 tensors ready for set_mla_kv_buffer_triton:

      nope_part: (num_tokens, 1, 288) = [nope e2m1 packed (256) | nope fp8 scales (32)]
      rope_part: (num_tokens, 1, 40)  = [rope e2m1 packed (32) | rope fp8 scales (4) | pad (4)]

    Capture-safe: static shapes, no host sync; global scale is a process-wide
    constant baked in at kernel launch.
    """
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    num_tokens, dim_nope = k_nope_2d.shape
    _, dim_rope = k_rope_2d.shape
    assert dim_nope == NOPE_DIM, f"expected dim_nope={NOPE_DIM}, got {dim_nope}"
    assert dim_rope == ROPE_DIM, f"expected dim_rope={ROPE_DIM}, got {dim_rope}"
    assert k_rope_2d.shape[0] == num_tokens

    k_nope_2d = k_nope_2d.contiguous()
    k_rope_2d = k_rope_2d.contiguous()

    nope_part = torch.empty(
        (num_tokens, NOPE_PART_BYTES), dtype=torch.uint8, device=k_nope_2d.device
    )
    # zeros so the 4 pad bytes are deterministic
    rope_part = torch.zeros(
        (num_tokens, ROPE_PART_BYTES), dtype=torch.uint8, device=k_rope_2d.device
    )

    nope_q = nope_part[:, :NOPE_Q_BYTES]
    nope_s = nope_part[:, NOPE_Q_BYTES:].view(_fp8_dtype)
    rope_q = rope_part[:, :ROPE_Q_BYTES]
    rope_s = rope_part[:, ROPE_Q_BYTES : ROPE_Q_BYTES + ROPE_BLOCKS].view(_fp8_dtype)

    if num_tokens > 0:
        _quantize_k_cache_nvfp4_kernel[(num_tokens,)](
            k_nope_2d,
            k_rope_2d,
            nope_q,
            nope_s,
            rope_q,
            rope_s,
            k_nope_2d.stride(0),
            k_rope_2d.stride(0),
            nope_part.stride(0),
            rope_part.stride(0),
            nvfp4_global_scale(),
            NOPE_NBLK=NOPE_BLOCKS,
            ROPE_NBLK=ROPE_BLOCKS,
        )

    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)


# ---------------------------------------------------------------------------
# Gather + dequantize (used by nsa_backend._forward_tilelang)
# ---------------------------------------------------------------------------


@triton.jit
def _dequantize_k_cache_nvfp4_paged_kernel(
    output_ptr,  # (num_tokens, 576) bf16
    row_ptr,  # uint8 pool rows, stride row_stride
    page_table_1_ptr,
    output_stride_0: int,
    row_stride_0: int,
    global_scale,
    OFF_NOPE_S_C: tl.constexpr,
    OFF_ROPE_Q_C: tl.constexpr,
    OFF_ROPE_S_C: tl.constexpr,
    NOPE_NBLK: tl.constexpr,
    ROPE_NBLK: tl.constexpr,
    NOPE_DIM_C: tl.constexpr,
):
    tok = tl.program_id(0)
    src = tl.load(page_table_1_ptr + tok).to(tl.int64)
    base = row_ptr + src * row_stride_0
    cols = tl.arange(0, 8)[None, :]

    # ---- nope ----
    rows = tl.arange(0, NOPE_NBLK)[:, None]
    b = tl.load(base + rows * 8 + cols).to(tl.int32)
    s_u8 = tl.load(base + OFF_NOPE_S_C + rows)
    s = s_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * global_scale

    v_lo = (_e2m1_decode(b & 15) * s).to(output_ptr.dtype.element_ty)
    v_hi = (_e2m1_decode((b >> 4) & 15) * s).to(output_ptr.dtype.element_ty)

    dim = rows * 16 + cols * 2
    tl.store(output_ptr + tok * output_stride_0 + dim, v_lo)
    tl.store(output_ptr + tok * output_stride_0 + dim + 1, v_hi)

    # ---- rope ----
    rrows = tl.arange(0, ROPE_NBLK)[:, None]
    rb = tl.load(base + OFF_ROPE_Q_C + rrows * 8 + cols).to(tl.int32)
    rs_u8 = tl.load(base + OFF_ROPE_S_C + rrows)
    rs = rs_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * global_scale

    rv_lo = (_e2m1_decode(rb & 15) * rs).to(output_ptr.dtype.element_ty)
    rv_hi = (_e2m1_decode((rb >> 4) & 15) * rs).to(output_ptr.dtype.element_ty)

    rdim = NOPE_DIM_C + rrows * 16 + cols * 2
    tl.store(output_ptr + tok * output_stride_0 + rdim, rv_lo)
    tl.store(output_ptr + tok * output_stride_0 + rdim + 1, rv_hi)


def dequantize_k_cache_nvfp4_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
) -> torch.Tensor:
    """Fused gather + dequant of NVFP4 pool rows selected by
    page_table_1_flattened (int tensor of row indices; negative entries must
    already be clamped to 0 by the caller, matching the fp8 hop convention).

    quant_k_cache: (N, 1, 328) fp8/uint8 view of the pool (any layer buffer)
    Returns: (num_tokens, 1, 576) bf16

    Capture-safe: output shape depends only on page_table shape; single
    kernel, no host sync.
    """
    rows_u8 = quant_k_cache.view(torch.uint8).view(-1, quant_k_cache.shape[-1])
    assert (
        rows_u8.shape[-1] == ROW_BYTES
    ), f"expected NVFP4 row of {ROW_BYTES} bytes, got {rows_u8.shape[-1]}"

    num_tokens = page_table_1_flattened.shape[0]
    output = torch.empty(
        (num_tokens, 1, FULL_DIM), dtype=torch.bfloat16, device=rows_u8.device
    )

    if num_tokens > 0:
        _dequantize_k_cache_nvfp4_paged_kernel[(num_tokens,)](
            output,
            rows_u8,
            page_table_1_flattened,
            output.stride(0),
            rows_u8.stride(0),
            nvfp4_global_scale(),
            OFF_NOPE_S_C=OFF_NOPE_S,
            OFF_ROPE_Q_C=OFF_ROPE_Q,
            OFF_ROPE_S_C=OFF_ROPE_S,
            NOPE_NBLK=NOPE_BLOCKS,
            ROPE_NBLK=ROPE_BLOCKS,
            NOPE_DIM_C=NOPE_DIM,
        )

    return output


def dequantize_k_cache_nvfp4(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize ALL rows (prefill full-pool hop). Not used inside CUDA
    graphs (prefill is not captured), so the arange here is fine."""
    rows_u8 = quant_k_cache.view(torch.uint8).view(-1, quant_k_cache.shape[-1])
    n = rows_u8.shape[0]
    page_table = torch.arange(n, dtype=torch.int32, device=rows_u8.device)
    out = dequantize_k_cache_nvfp4_paged(rows_u8, page_table)
    if quant_k_cache.ndim == 3:
        return out.view(n, 1, FULL_DIM)
    return out.view(quant_k_cache.shape[:-1] + (FULL_DIM,))


# ---------------------------------------------------------------------------
# Torch reference implementation (numerics oracle for nvfp4_kv_bench.py)
# ---------------------------------------------------------------------------

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _e2m1_encode_torch(t: torch.Tensor) -> torch.Tensor:
    """t: float tensor of x / (block_scale * global_scale). Returns uint8 codes."""
    a = t.abs().clamp(max=6.0)
    m = torch.zeros_like(a, dtype=torch.uint8)
    m = torch.where(a > 0.25, torch.tensor(1, dtype=torch.uint8, device=a.device), m)
    m = torch.where(a >= 0.75, torch.tensor(2, dtype=torch.uint8, device=a.device), m)
    m = torch.where(a > 1.25, torch.tensor(3, dtype=torch.uint8, device=a.device), m)
    m = torch.where(a >= 1.75, torch.tensor(4, dtype=torch.uint8, device=a.device), m)
    m = torch.where(a > 2.5, torch.tensor(5, dtype=torch.uint8, device=a.device), m)
    m = torch.where(a >= 3.5, torch.tensor(6, dtype=torch.uint8, device=a.device), m)
    m = torch.where(a > 5.0, torch.tensor(7, dtype=torch.uint8, device=a.device), m)
    return torch.where(t < 0, m + 8, m)


def quantize_k_cache_nvfp4_ref(x: torch.Tensor) -> torch.Tensor:
    """x: (n, 576) float/bf16 -> (n, 328) uint8 rows in the pool layout."""
    n, d = x.shape
    assert d == FULL_DIM
    gs = nvfp4_global_scale()
    xf = x.float()

    rows = torch.zeros((n, ROW_BYTES), dtype=torch.uint8, device=x.device)

    def quant_section(vals, q_off, s_off, nblk):
        blocks = vals.view(n, nblk, QUANT_BLOCK)
        amax = blocks.abs().amax(dim=-1)
        s_fp8 = (amax / (6.0 * gs)).clamp(max=FP8_SCALE_MAX).to(_fp8_dtype)
        s_dec = s_fp8.float() * gs
        inv = torch.where(s_dec > 0, 1.0 / s_dec, torch.zeros_like(s_dec))
        codes = _e2m1_encode_torch(blocks * inv.unsqueeze(-1)).view(n, -1)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        rows[:, q_off : q_off + packed.shape[1]] = packed
        rows[:, s_off : s_off + nblk] = s_fp8.view(torch.uint8)

    quant_section(xf[:, :NOPE_DIM], OFF_NOPE_Q, OFF_NOPE_S, NOPE_BLOCKS)
    quant_section(xf[:, NOPE_DIM:], OFF_ROPE_Q, OFF_ROPE_S, ROPE_BLOCKS)
    return rows


def dequantize_k_cache_nvfp4_ref(rows: torch.Tensor) -> torch.Tensor:
    """rows: (n, 328) uint8 -> (n, 576) float32."""
    n = rows.shape[0]
    assert rows.shape[-1] == ROW_BYTES
    gs = nvfp4_global_scale()
    lut = E2M1_VALUES.to(rows.device)
    out = torch.empty((n, FULL_DIM), dtype=torch.float32, device=rows.device)

    def dequant_section(q_off, s_off, nblk, d_off, dim):
        packed = rows[:, q_off : q_off + dim // 2].to(torch.int32)
        lo, hi = packed & 15, (packed >> 4) & 15
        codes = torch.stack([lo, hi], dim=-1).view(n, dim)
        sign = torch.where(codes >= 8, -1.0, 1.0)
        vals = lut[(codes & 7).long()] * sign
        s = rows[:, s_off : s_off + nblk].view(_fp8_dtype).float() * gs
        s = s.unsqueeze(-1).expand(n, nblk, QUANT_BLOCK).reshape(n, dim)
        out[:, d_off : d_off + dim] = vals * s

    dequant_section(OFF_NOPE_Q, OFF_NOPE_S, NOPE_BLOCKS, 0, NOPE_DIM)
    dequant_section(OFF_ROPE_Q, OFF_ROPE_S, ROPE_BLOCKS, NOPE_DIM, ROPE_DIM)
    return out
