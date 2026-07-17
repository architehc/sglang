"""Triton implementation of the DeepSeek-V3.2 indexer MQA logits for GPUs
without deep_gemm support (e.g. sm_120).

    out[q, k] = k_scale[k] * sum_h( relu(q[q, h, :] . k[k, :]) * w[q, h] )
restricted to k in [ks[q], ke[q]); out-of-window entries are -inf.

Also includes a fused PAGED variant (``triton_fp8_paged_mqa_logits``) that
reads keys/scales straight from the page-blocked index-K buffer via the
block table — replacing the torch fallback in sm120_mqa_logits.py, which
gathers every page, expands the full context to bf16, and python-loops the
matmul. The paged kernel allocates only the [rows, max_seq_len] output.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mqa_logits_kernel(
    q_ptr,        # [n_q, H, D] fp8e4nv
    k_ptr,        # [n_k, D] fp8e4nv
    ksc_ptr,      # [n_k] fp32
    w_ptr,        # [n_q, H] fp32
    ks_ptr,       # [n_q] int32
    ke_ptr,       # [n_q] int32
    out_ptr,      # [n_q, n_k] fp32
    n_q, n_k,
    H: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    KSPAN: tl.constexpr,
    FP8_DOT: tl.constexpr,
):
    # One program owns a [BQ] query tile and sweeps KSPAN consecutive BK-key
    # blocks, so the query tile (and its weights/window bounds) is loaded once
    # per stripe instead of once per key block. All H heads fold into a single
    # [BQ*H, D] x [D, BK] tensor-core dot; rows of the flattened q are (q, h)
    # pairs, contiguous in h, so offs_m = q0*H + arange(BQ*H) is exact.
    pid_q = tl.program_id(0)
    pid_s = tl.program_id(1)
    q0 = pid_q * BQ
    offs_q = q0 + tl.arange(0, BQ)
    qm = offs_q < n_q
    offs_d = tl.arange(0, D)

    offs_m = q0 * H + tl.arange(0, BQ * H)
    mm = offs_m < n_q * H
    q_tile = tl.load(
        q_ptr + offs_m[:, None] * D + offs_d[None, :],
        mask=mm[:, None],
        other=0.0,
    )
    if not FP8_DOT:
        q_tile = q_tile.to(tl.bfloat16)
    w = tl.load(w_ptr + offs_m, mask=mm, other=0.0)  # [BQ*H]
    ks = tl.load(ks_ptr + offs_q, mask=qm, other=0)
    ke = tl.load(ke_ptr + offs_q, mask=qm, other=0)
    # n_q * n_k can exceed int32 (e.g. 3300 x 900k); index the output in int64.
    out_row = offs_q.to(tl.int64)[:, None] * n_k

    for s in tl.range(KSPAN):
        k0 = (pid_s * KSPAN + s) * BK
        offs_k = k0 + tl.arange(0, BK)
        km = offs_k < n_k
        k_tile = tl.load(
            k_ptr + offs_k[None, :] * D + offs_d[:, None],
            mask=km[None, :],
            other=0.0,
        )
        if not FP8_DOT:
            k_tile = k_tile.to(tl.bfloat16)
        s_mat = tl.dot(q_tile, k_tile)          # [BQ*H, BK] fp32
        s_mat = tl.maximum(s_mat, 0.0) * w[:, None]
        acc = tl.sum(tl.reshape(s_mat, (BQ, H, BK)), axis=1)  # [BQ, BK]
        ksc = tl.load(ksc_ptr + offs_k, mask=km, other=0.0)
        acc *= ksc[None, :]
        in_win = (offs_k[None, :] >= ks[:, None]) & (offs_k[None, :] < ke[:, None])
        acc = tl.where(in_win, acc, float("-inf"))
        tl.store(
            out_ptr + out_row + offs_k[None, :],
            acc,
            mask=qm[:, None] & km[None, :],
        )


_FP8_DOT_OK = None


def _fp8_dot_supported():
    """fp8e4nv tl.dot compiles on Blackwell Triton but probe once and cache."""
    global _FP8_DOT_OK
    if _FP8_DOT_OK is None:
        try:
            q = torch.zeros(16, 32, 128, dtype=torch.float8_e4m3fn, device="cuda")
            k = torch.zeros(128, 128, dtype=torch.float8_e4m3fn, device="cuda")
            _run_mqa_logits(q, k, torch.ones(128, device="cuda"),
                            torch.zeros(16, 32, device="cuda"),
                            torch.zeros(16, dtype=torch.int32, device="cuda"),
                            torch.full((16,), 128, dtype=torch.int32, device="cuda"),
                            fp8_dot=True)
            _FP8_DOT_OK = True
        except Exception:
            _FP8_DOT_OK = False
    return _FP8_DOT_OK


def _run_mqa_logits(q, k_fp8, k_scale, weights, ks, ke, fp8_dot):
    n_q, H, D = q.shape
    n_k = k_fp8.shape[0]
    out = torch.empty(n_q, n_k, dtype=torch.float32, device=q.device)
    BQ, BK, KSPAN = 4, 128, 64
    grid = (triton.cdiv(n_q, BQ), triton.cdiv(n_k, BK * KSPAN))
    _mqa_logits_kernel[grid](
        q, k_fp8, k_scale.to(torch.float32), weights.to(torch.float32),
        ks.to(torch.int32), ke.to(torch.int32), out,
        n_q, n_k, H=H, D=D, BQ=BQ, BK=BK, KSPAN=KSPAN, FP8_DOT=fp8_dot,
        num_warps=4, num_stages=2,
    )
    return out


def triton_fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=True):
    k_fp8, k_scale = kv
    return _run_mqa_logits(q, k_fp8, k_scale, weights, ks, ke,
                           fp8_dot=_fp8_dot_supported())


# ---------------------------------------------------------------------------
# Paged variant: keys/scales read directly from the page-blocked index-K
# buffer (memory_pool layout): per page of PAGE tokens,
#   buf[page, :PAGE*D]        = fp8 keys (token-major, D per token)
#   buf[page, PAGE*D:]        = PAGE fp32 scales
# so in the fp32 view of a page row, scale t sits at index PAGE*D/4 + t.
#
# One program per (query row, BK-key block): loads that row's [H, D] query,
# gathers the BK keys through the block table, and computes all H heads with
# a single [H, D] x [D, BK] dot — the decode-shaped counterpart of the
# prefill kernel above (which shares one K tile across BQ=16 queries).
# ---------------------------------------------------------------------------
@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,        # [rows, H, D] fp8e4nv
    kv_ptr,       # [num_pages, PAGE_BYTES] fp8e4nv (raw page rows)
    ksc_ptr,      # [num_pages, PAGE_BYTES // 4] fp32 (same bytes, f32 view)
    w_ptr,        # [rows, H] fp32
    seqlens_ptr,  # [rows] int32
    bt_ptr,       # [B, max_pages] int32
    out_ptr,      # [rows, max_seq_len] fp32
    max_seq_len, max_pages, next_n, num_pages,
    H: tl.constexpr,   # power of 2, >= 16
    D: tl.constexpr,   # index head dim (128)
    BK: tl.constexpr,  # keys per program
    PAGE: tl.constexpr,          # tokens per page (64)
    PAGE_BYTES: tl.constexpr,    # bytes per page row (PAGE*D + PAGE*4)
):
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = pid_r // next_n

    offs_k = pid_k * BK + tl.arange(0, BK)
    km = offs_k < max_seq_len
    seq = tl.load(seqlens_ptr + pid_r)
    # Blocks fully past the row's true seqlen only ever write -inf (see the
    # tl.where below); store it directly and skip the block-table gather and
    # dot. The grid is static (sized on max_seq_len), so CUDA-graph capture
    # is unaffected.
    if pid_k * BK >= seq:
        tl.store(out_ptr + pid_r * max_seq_len + offs_k, float("-inf"), mask=km)
        return

    page_slot = offs_k // PAGE
    in_page = offs_k % PAGE
    phys = tl.load(bt_ptr + b * max_pages + page_slot, mask=km, other=0)
    # Page-table entries beyond a sequence's real pages may be stale; clamp so
    # the gather can never read out of bounds (those lanes are masked below).
    phys = tl.minimum(tl.maximum(phys, 0), num_pages - 1)

    offs_d = tl.arange(0, D)
    k_tile = tl.load(
        kv_ptr + phys[None, :] * PAGE_BYTES + in_page[None, :] * D + offs_d[:, None],
        mask=km[None, :],
        other=0.0,
    ).to(tl.bfloat16)  # [D, BK]

    offs_h = tl.arange(0, H)
    q_row = tl.load(
        q_ptr + pid_r * (H * D) + offs_h[:, None] * D + offs_d[None, :]
    ).to(tl.bfloat16)  # [H, D]

    s = tl.dot(q_row, k_tile)  # [H, BK] fp32
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + pid_r * H + offs_h)  # [H]
    acc = tl.sum(s * w[:, None], axis=0)  # [BK]

    ksc = tl.load(
        ksc_ptr + phys * (PAGE_BYTES // 4) + (PAGE * D) // 4 + in_page,
        mask=km,
        other=0.0,
    )
    acc *= ksc

    acc = tl.where(offs_k < seq, acc, float("-inf"))
    tl.store(out_ptr + pid_r * max_seq_len + offs_k, acc, mask=km)


def triton_fp8_paged_mqa_logits(
    q, kv_cache, weights, seqlens, block_tables, schedule_metadata, max_seq_len, clean_logits=True
):
    """Fused paged MQA logits; same signature/semantics as
    sm120_mqa_logits.torch_fp8_paged_mqa_logits (schedule_metadata ignored,
    always clean). Raises on unsupported shapes so the caller can fall back
    to the torch path."""
    B, next_n, H, D = q.shape
    if H < 16 or (H & (H - 1)) != 0:
        raise ValueError(f"triton paged mqa: need power-of-2 H >= 16, got {H}")
    rows = B * next_n
    PAGE = 64
    num_pages = kv_cache.shape[0]
    flat = kv_cache.reshape(num_pages, -1)  # [P, PAGE_BYTES], view of the pool
    page_bytes = flat.shape[1]
    flat_fp8 = flat.view(torch.float8_e4m3fn)  # keys at [p, t*D + d]
    flat_f32 = flat.view(torch.float32)        # scale t at [p, PAGE*D/4 + t]

    out = torch.empty((rows, max_seq_len), dtype=torch.float32, device=q.device)
    q_flat = q.reshape(rows, H, D).contiguous()

    BK = 128
    grid = (rows, triton.cdiv(max_seq_len, BK))
    _paged_mqa_logits_kernel[grid](
        q_flat, flat_fp8, flat_f32, weights.to(torch.float32),
        seqlens.to(torch.int32), block_tables.to(torch.int32), out,
        max_seq_len, block_tables.shape[1], next_n, num_pages,
        H=H, D=D, BK=BK, PAGE=PAGE, PAGE_BYTES=page_bytes,
        num_warps=4, num_stages=2,
    )
    return out
