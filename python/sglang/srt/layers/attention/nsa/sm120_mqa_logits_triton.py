"""Triton implementation of the DeepSeek-V3.2 indexer MQA logits for GPUs
without deep_gemm support (e.g. sm_120).

    out[q, k] = k_scale[k] * sum_h( relu(q[q, h, :] . k[k, :]) * w[q, h] )
restricted to k in [ks[q], ke[q]); out-of-window entries are -inf.

Also includes a fused PAGED variant (``triton_fp8_paged_mqa_logits``) that
reads keys/scales straight from the page-blocked index-K buffer via the
block table — replacing the torch fallback in sm120_mqa_logits.py, which
gathers every page, expands the full context to bf16, and python-loops the
matmul. The paged kernel allocates only the [rows, max_seq_len] output.

And an exact two-pass (block-max) top-2048 selection for ragged prefill
(``two_pass_topk_ragged``; campaign-2 task 13 part 2, gated by
SGLANG_NSA_TWO_PASS_TOPK) that never materializes the [rows, L] fp32
logits: pass 1 is the same logits kernel with the store replaced by a
per-128-key block-max epilogue; pass 2 is the SAME kernel again (same
BQ/BK/KSPAN tiling, so the recomputed scores are bit-identical to the
full-logits path) with the store retargeted to a compacted candidate
buffer holding only the blocks whose max can reach the top-k. A
per-row BQ=1 recompute kernel was tried first and REJECTED: the
different dot/reduction layout shifts results by 1-4 ulps on ~31% of
lanes, which is not exact by construction at the selection boundary.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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
    bmax_ptr=None,  # [n_q, nblocks] fp32 block-max out (opt.)
    nblocks=0,
    sel_slot_ptr=None,  # [n_q, nblocks] int32 candidate slot (-1 = no)
    sel_flag_ptr=None,  # [cdiv(n_q, BQ), nblocks] uint8 tile-selected
    cand_ptr=None,  # [n_q, cand_stride] fp32 candidate scores
    cand_stride=0,
    EMIT_LOGITS: tl.constexpr = True,
    EMIT_BLOCK_MAX: tl.constexpr = False,
    EMIT_SEL: tl.constexpr = False,
):
    # One program owns a [BQ] query tile and sweeps KSPAN consecutive BK-key
    # blocks, so the query tile (and its weights/window bounds) is loaded once
    # per stripe instead of once per key block. All H heads fold into a single
    # [BQ*H, D] x [D, BK] tensor-core dot; rows of the flattened q are (q, h)
    # pairs, contiguous in h, so offs_m = q0*H + arange(BQ*H) is exact.
    pid_q = tl.program_id(0)
    pid_s = tl.program_id(1)
    q0 = pid_q * BQ
    if EMIT_SEL:
        # Whole-stripe skip: no block of this stripe is selected by any row
        # of this q-tile (short-window rows skip most stripes).
        soffs = pid_s * KSPAN + tl.arange(0, KSPAN)
        sf = tl.load(
            sel_flag_ptr + pid_q * nblocks + soffs, mask=soffs < nblocks, other=0
        )
        if tl.max(sf) == 0:
            return
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
        run = True
        if EMIT_SEL:
            # Per-block skip: not selected by any row of this q-tile.
            run = tl.load(sel_flag_ptr + pid_q * nblocks + (pid_s * KSPAN + s)) != 0
        if run:
            offs_k = k0 + tl.arange(0, BK)
            km = offs_k < n_k
            k_tile = tl.load(
                k_ptr + offs_k[None, :] * D + offs_d[:, None],
                mask=km[None, :],
                other=0.0,
            )
            if not FP8_DOT:
                k_tile = k_tile.to(tl.bfloat16)
            s_mat = tl.dot(q_tile, k_tile)  # [BQ*H, BK] fp32
            s_mat = tl.maximum(s_mat, 0.0) * w[:, None]
            acc = tl.sum(tl.reshape(s_mat, (BQ, H, BK)), axis=1)  # [BQ, BK]
            ksc = tl.load(ksc_ptr + offs_k, mask=km, other=0.0)
            acc *= ksc[None, :]
            in_win = (offs_k[None, :] >= ks[:, None]) & (offs_k[None, :] < ke[:, None])
            acc = tl.where(in_win, acc, float("-inf"))
            if EMIT_LOGITS:
                tl.store(
                    out_ptr + out_row + offs_k[None, :],
                    acc,
                    mask=qm[:, None] & km[None, :],
                )
            if EMIT_BLOCK_MAX:
                # One (pid_q, s) iteration covers exactly one BK-key block; the
                # max excludes lanes past n_k (partial tail block) via km, so a
                # block max is the max over real in-window scores only (-inf if
                # the block has none).
                bmax = tl.max(tl.where(km[None, :], acc, float("-inf")), axis=1)
                tl.store(
                    bmax_ptr + offs_q.to(tl.int64) * nblocks + (pid_s * KSPAN + s),
                    bmax,
                    mask=qm,
                )
            if EMIT_SEL:
                # Compacted store: block (pid_s*KSPAN+s) goes to candidate slot
                # sel_slot[q] (its rank among the row's selected blocks), which
                # keeps candidate keys in ascending order per row. The column
                # is slot*BK + within-block lane (NOT the global key offset).
                slots = tl.load(
                    sel_slot_ptr + offs_q * nblocks + (pid_s * KSPAN + s),
                    mask=qm,
                    other=-1,
                )
                tl.store(
                    cand_ptr
                    + offs_q.to(tl.int64)[:, None] * cand_stride
                    + slots[:, None] * BK
                    + tl.arange(0, BK)[None, :],
                    acc,
                    mask=qm[:, None] & (slots[:, None] >= 0) & km[None, :],
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
# Exact two-pass (block-max) top-k for ragged prefill (campaign-2 task 13
# part 2). Pass 1 is _mqa_logits_kernel with EMIT_BLOCK_MAX (no logits
# store); pass 2 is _mqa_logits_kernel again with EMIT_SEL — the SAME
# BQ/BK/KSPAN tiling, so recomputed scores are bit-identical to the
# full-logits path — storing only selected blocks, compacted per row.
# Exactness: at most topk-1 scores exceed the true topk-th score t, so at
# most topk-1 block maxes exceed t and the topk-th block max tau <= t;
# every key with score >= t therefore sits in a selected block (max >= tau,
# intersecting the window) and the candidate topk reproduces the full-row
# torch.topk set exactly. Ties only inflate the candidate count; rows whose
# selected-block count exceeds the cap report overflow (return None) so the
# caller can fall back to full materialization.
# ---------------------------------------------------------------------------

_TOPK_BLOCK = 128  # keys per selection block (== BK of the logits kernel)
_SEL_BQ = 4  # pass-2 q-tile rows; MUST match the full-logits kernel tiling
_SEL_KSPAN = 64  # pass-2 blocks swept per program (== full-logits KSPAN)


def triton_fp8_mqa_block_max(q, kv, weights, ks, ke):
    """Per-128-key block maxes of the MQA logits, [n_q, ceil(n_k/128)] fp32.

    Bit-exact vs the max of the full ``triton_fp8_mqa_logits`` output over
    each block (same kernel, same scores, logits store elided).
    """
    k_fp8, k_scale = kv
    n_q, H, D = q.shape
    n_k = k_fp8.shape[0]
    nblocks = triton.cdiv(n_k, _TOPK_BLOCK)
    bmax = torch.empty(n_q, nblocks, dtype=torch.float32, device=q.device)
    BQ, BK, KSPAN = 4, _TOPK_BLOCK, 64
    grid = (triton.cdiv(n_q, BQ), triton.cdiv(n_k, BK * KSPAN))
    _mqa_logits_kernel[grid](
        q,
        k_fp8,
        k_scale.to(torch.float32),
        weights.to(torch.float32),
        ks.to(torch.int32),
        ke.to(torch.int32),
        bmax,  # out_ptr unused
        n_q,
        n_k,
        H=H,
        D=D,
        BQ=BQ,
        BK=BK,
        KSPAN=KSPAN,
        FP8_DOT=_fp8_dot_supported(),
        bmax_ptr=bmax,
        nblocks=nblocks,
        EMIT_LOGITS=False,
        EMIT_BLOCK_MAX=True,
        num_warps=4,
        num_stages=2,
    )
    return bmax


def two_pass_topk_ragged(
    q, kv, weights, ks, ke, topk, cap_blocks=None, return_debug=False
):
    """Exact top-`topk` selection over the ragged MQA logits [n_q, n_k]
    without materializing the logits.

    Same contract as ``_torch_fast_topk_v2(..., row_starts=ks,
    prewindowed=True)``: int32 [n_q, topk] window-relative indices, -1 for
    slots with no valid element. Candidates are stored in ascending key
    order per row, preserving torch.topk's (value desc, index asc) tie
    order. Returns None when a row's selected-block count exceeds
    cap_blocks (pathological ties); the caller must fall back to full
    materialization then.
    """
    k_fp8, k_scale = kv
    n_q, H, D = q.shape
    n_k = k_fp8.shape[0]
    device = q.device
    nblocks = triton.cdiv(n_k, _TOPK_BLOCK)
    if cap_blocks is None:
        cap_blocks = 2 * topk
    cap_blocks = min(cap_blocks, nblocks)
    if nblocks == 0 or n_q == 0:
        return torch.full((n_q, topk), -1, dtype=torch.int32, device=device)

    block_max = triton_fp8_mqa_block_max(q, kv, weights, ks, ke)

    # tau[r] = topk-th largest block max of row r (its min block max when
    # nblocks < topk, which selects every window block).
    tau = torch.topk(block_max, min(topk, nblocks), dim=1).values[:, -1:]
    ks32 = ks.to(torch.int32)
    ke32 = ke.to(torch.int32)
    bstart = torch.arange(nblocks, device=device, dtype=torch.int32) * _TOPK_BLOCK
    in_win_blocks = (bstart[None, :] < ke32[:, None]) & (
        (bstart[None, :] + _TOPK_BLOCK) > ks32[:, None]
    )
    sel_mask = (block_max >= tau) & in_win_blocks
    # Host sync for the data-dependent candidate width; legal here because
    # ragged prefill is never CUDA-graph captured.
    max_sel = int(sel_mask.sum(dim=1).max().item())
    if max_sel > cap_blocks:
        logger.warning(
            "two_pass_topk_ragged: candidate overflow (%d selected blocks "
            "> cap %d; pathological block-max ties); returning None so the "
            "caller falls back to full logits materialization.",
            max_sel,
            cap_blocks,
        )
        return None
    width_blocks = max(max_sel, 1)

    # Per-row compacted candidate slot of each block (-1 = not selected);
    # the slot IS the block's rank among selected blocks, so candidate
    # columns stay in ascending key order per row.
    csum = sel_mask.to(torch.int32).cumsum(dim=1)
    sel_slot = torch.where(sel_mask, csum - 1, csum.new_full((), -1))
    # Per-q-tile any-selected flags let pass 2 skip blocks no row of the
    # tile selected (most blocks for short-window rows, ~topk/nblocks of
    # blocks at 1M).
    n_qt = triton.cdiv(n_q, _SEL_BQ)
    pad_q = n_qt * _SEL_BQ - n_q
    sel_padded = (
        torch.nn.functional.pad(sel_mask, (0, 0, 0, pad_q)) if pad_q else sel_mask
    )
    sel_flag = sel_padded.view(n_qt, _SEL_BQ, nblocks).any(dim=1).to(torch.uint8)
    # slot -> block-id inverse map (ascending per row; nonzero is row-major)
    # for translating candidate columns back to key indices after the topk.
    nz = sel_mask.nonzero()
    inv_slot = torch.full((n_q, width_blocks), -1, dtype=torch.int32, device=device)
    if nz.numel() > 0:
        row0 = nz[:, 0]
        row_start = torch.searchsorted(row0, torch.arange(n_q, device=device))
        within = torch.arange(nz.shape[0], device=device) - row_start[row0]
        inv_slot[row0, within] = nz[:, 1].to(torch.int32)

    # Pass 2 writes only selected slots; the rest must read as -inf so the
    # final topk maps them to -1 exactly like the prewindowed full path.
    cand = torch.full(
        (n_q, width_blocks * _TOPK_BLOCK),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    grid = (n_qt, triton.cdiv(n_k, _TOPK_BLOCK * _SEL_KSPAN))
    _mqa_logits_kernel[grid](
        q,
        k_fp8,
        k_scale.to(torch.float32),
        weights.to(torch.float32),
        ks32,
        ke32,
        cand,  # out_ptr unused in EMIT_SEL
        n_q,
        n_k,
        H=H,
        D=D,
        BQ=_SEL_BQ,
        BK=_TOPK_BLOCK,
        KSPAN=_SEL_KSPAN,
        FP8_DOT=_fp8_dot_supported(),
        nblocks=nblocks,
        sel_slot_ptr=sel_slot,
        sel_flag_ptr=sel_flag,
        cand_ptr=cand,
        cand_stride=width_blocks * _TOPK_BLOCK,
        EMIT_LOGITS=False,
        EMIT_SEL=True,
        num_warps=4,
        num_stages=2,
    )

    k = min(topk, cand.shape[1])
    vals, idx_c = torch.topk(cand, k, dim=1)
    bid = inv_slot.gather(1, idx_c // _TOPK_BLOCK)
    gidx = bid.to(torch.int64) * _TOPK_BLOCK + (idx_c % _TOPK_BLOCK)
    rel = gidx - ks.to(torch.int64)[:, None]
    out = torch.where(torch.isinf(vals) & (vals < 0), rel.new_full((), -1), rel).to(
        torch.int32
    )
    if k < topk:
        out = torch.nn.functional.pad(out, (0, topk - k), value=-1)
    if return_debug:
        debug = {
            "block_max": block_max,
            "sel_blocks": inv_slot,
            "cand": cand,
            "max_sel": max_sel,
            "tau": tau,
        }
        return out, debug
    return out


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
