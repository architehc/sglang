"""Exact, seqlen-scanning Triton top-k for the NSA indexer on GPUs without
sgl_kernel fast_topk support (e.g. sm_120).

Drop-in replacement for ``_torch_fast_topk_v2`` (nsa_backend.py), gated by
SGLANG_NSA_TRITON_TOPK (default OFF). The torch fallback runs ``torch.topk``
over the full static [B, L] logits buffer (~250 us at L = 1M) even though
only the first ``seqlen`` entries of each row are valid; this kernel reads
each row's window [row_starts[i], row_starts[i] + lengths[i]) from device
memory and grid blocks outside the window exit immediately, so the work
scales with the valid prefix instead of the static buffer width.

Algorithm: exact radix select over the sortable-uint32 image of fp32
(3 rounds of 2048-bin histograms, 11/11/10 bits) plus a deterministic
selection pass. Selection takes every element with key > tau, then the first
``k - count(> tau)`` elements with key == tau in ascending index order, i.e.
a total order of (value desc, index asc) — verified to reproduce torch.topk's
selected index set exactly, including duplicate-heavy inputs. Like the native
sgl_kernel fast_topk_v2, the output is NOT value-sorted: selected indices
occupy slots [0, n_selected) in ascending-index order and the rest is -1
(downstream consumers use it as an unordered gather set).

CUDA-graph capture safe: the grid depends only on the static buffer width L
(seqlen is read from a GPU tensor, never on the host), all intermediate state
lives in fixed-shape cached workspaces, and there are no host syncs or
data-dependent allocations. The output is freshly allocated per call (graph
pool); workspaces are reused across calls and fully rewritten each call.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096  # logits elements scanned per program
_NUM_WARPS = 4
_NBINS = tl.constexpr(2048)  # histogram bins per radix round (11 bits)
_TOPK_MAX = 8192
_L_MAX = 1 << 20

# state tensor layout per row (int32)
_S_PREFIX = tl.constexpr(0)  # accumulated high bits of tau (next compare)
_S_KPRIME = tl.constexpr(1)  # rank still to resolve within current bucket
_S_TAU = tl.constexpr(2)  # final 32-bit threshold key (bit pattern)
_S_TOTAL_GT = tl.constexpr(3)  # number of elements with key > tau
_S_NEEDED_EQ = tl.constexpr(4)  # number of key == tau elements still needed
_S_SHORT = tl.constexpr(5)  # row has fewer valid elements than topk
_STATE_LEN = tl.constexpr(8)


@triton.jit
def _sortable_u32(v):
    """Monotonic fp32 -> uint32 map: v1 > v2 iff s(v1) > s(v2)."""
    b = v.to(tl.int32, bitcast=True)
    # 0x80000000 as signed int32 is -2147483648; the bit pattern is what matters
    return (b ^ ((b >> 31) | (-2147483648))).to(tl.uint32, bitcast=True)


@triton.jit
def _load_window_chunk(
    score_ptr,
    lengths_ptr,
    row_starts_ptr,
    row,
    pid,
    L,
    BLOCK: tl.constexpr,
    PREWINDOWED: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
):
    """Returns (values, sortable keys, in-window mask, outside flag) for one
    BLOCK chunk; when outside flag is set the caller must not use the rest."""
    ln = tl.load(lengths_ptr + row)
    rs = 0
    if HAS_ROW_STARTS:
        rs = tl.load(row_starts_ptr + row)
    re = tl.minimum(rs + ln, L)
    bs = pid * BLOCK
    outside = (bs >= re) | (bs + BLOCK <= rs)
    offs = bs + tl.arange(0, BLOCK)
    inw = (offs >= rs) & (offs < re) & (outside == 0)
    v = tl.load(
        score_ptr + row.to(tl.int64) * L + offs, mask=inw & (offs < L), other=0.0
    )
    if not PREWINDOWED:
        # Match torch.nan_to_num(0.0) on the valid region: NaN -> 0 and
        # +-inf -> +-FLT_MAX (finite!), so real -inf values still outrank
        # invalid positions and keep their index.
        fmax = 3.4028234663852886e38
        v = tl.where(v != v, 0.0, v)
        v = tl.where(v == float("inf"), fmax, v)
        v = tl.where(v == float("-inf"), -fmax, v)
    return v, _sortable_u32(v), inw, outside


@triton.jit
def _topk_hist_kernel(
    score_ptr,
    lengths_ptr,
    row_starts_ptr,
    bins_ptr,
    state_ptr,
    L,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    PREWINDOWED: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
):
    """Per-block 2048-bin histogram of one radix digit over the window.

    ROUND 0: bits [31:21]; ROUND 1: bits [20:10] among elements whose top 11
    bits match the round-0 prefix; ROUND 2: bits [9:0] among elements whose
    top 22 bits match.
    """
    pid = tl.program_id(0)
    row = tl.program_id(1)
    if ROUND > 0:
        if tl.load(state_ptr + row * _STATE_LEN + _S_SHORT) != 0:
            return
    v, s, inw, outside = _load_window_chunk(
        score_ptr,
        lengths_ptr,
        row_starts_ptr,
        row,
        pid,
        L,
        BLOCK,
        PREWINDOWED,
        HAS_ROW_STARTS,
    )
    if outside:
        return
    if ROUND == 0:
        digit = s >> 21
        match = inw
    elif ROUND == 1:
        prefix = tl.load(state_ptr + row * _STATE_LEN + _S_PREFIX).to(
            tl.uint32, bitcast=True
        )
        digit = (s >> 10) & 0x7FF
        match = inw & ((s >> 21) == prefix)
    else:
        prefix = tl.load(state_ptr + row * _STATE_LEN + _S_PREFIX).to(
            tl.uint32, bitcast=True
        )
        digit = s & 0x3FF
        match = inw & ((s >> 10) == prefix)
    if tl.max(match.to(tl.int32)) == 0:
        return
    h = tl.histogram(digit.to(tl.int32, bitcast=True), _NBINS, mask=match)
    tl.atomic_add(bins_ptr + row * _NBINS + tl.arange(0, _NBINS), h)


@triton.jit
def _topk_scan_kernel(
    bins_ptr,
    state_ptr,
    topk,
    NBINS: tl.constexpr,
    ROUND: tl.constexpr,
):
    """Single program per row: fold the 2048-bin histogram into the radix
    prefix (or, on the last round, into tau/total_gt/needed_eq), then re-zero
    the bins for the next round/call."""
    row = tl.program_id(0)
    base = state_ptr + row * _STATE_LEN
    if ROUND == 0:
        tl.store(base + _S_SHORT, 0)
    else:
        if tl.load(base + _S_SHORT) != 0:
            return
    offs = tl.arange(0, NBINS)
    h = tl.load(bins_ptr + row * NBINS + offs)
    tl.store(bins_ptr + row * NBINS + offs, tl.zeros([NBINS], tl.int32))
    if ROUND == 0:
        kprime = topk
    else:
        kprime = tl.load(base + _S_KPRIME)
    total = tl.sum(h)
    # revcum[j] = count of elements with digit >= j
    revcum = total - tl.cumsum(h) + h
    bstar = tl.max(tl.where(revcum >= kprime, offs, -1))
    if bstar < 0:
        # Fewer valid elements than topk: everything valid is selected.
        tl.store(base + _S_SHORT, 1)
        tl.store(base + _S_TAU, 0)
        tl.store(base + _S_NEEDED_EQ, 0)
        tl.store(base + _S_TOTAL_GT, total)
        return
    gt_digit = tl.sum(tl.where(offs > bstar, h, 0))
    knew = kprime - gt_digit
    tl.store(base + _S_KPRIME, knew)
    if ROUND == 0:
        tl.store(base + _S_PREFIX, bstar)
    elif ROUND == 1:
        pold = tl.load(base + _S_PREFIX)
        tl.store(base + _S_PREFIX, (pold << 11) | bstar)
    else:
        pold = tl.load(base + _S_PREFIX)
        tl.store(base + _S_TAU, ((pold << 10) | bstar).to(tl.int32, bitcast=True))
        tl.store(base + _S_NEEDED_EQ, knew)
        tl.store(base + _S_TOTAL_GT, topk - knew)


@triton.jit
def _topk_sel_count_kernel(
    score_ptr,
    lengths_ptr,
    row_starts_ptr,
    state_ptr,
    counts_ptr,
    L,
    B,
    BLOCK: tl.constexpr,
    PREWINDOWED: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
):
    """Per-block counts of (key > tau) and (key == tau) within the window;
    blocks outside the window still store their zero counts."""
    pid = tl.program_id(0)
    row = tl.program_id(1)
    dst = counts_ptr + (pid * B + row) * 2
    v, s, inw, outside = _load_window_chunk(
        score_ptr,
        lengths_ptr,
        row_starts_ptr,
        row,
        pid,
        L,
        BLOCK,
        PREWINDOWED,
        HAS_ROW_STARTS,
    )
    if outside:
        tl.store(dst + 0, 0)
        tl.store(dst + 1, 0)
        return
    base = state_ptr + row * _STATE_LEN
    if tl.load(base + _S_SHORT) != 0:
        is_gt = inw
        is_eq = inw & (s != s)  # all-false, dtype-stable
    else:
        tau = tl.load(base + _S_TAU).to(tl.uint32, bitcast=True)
        is_gt = inw & (s > tau)
        is_eq = inw & (s == tau)
    tl.store(dst + 0, tl.sum(is_gt.to(tl.int32)))
    tl.store(dst + 1, tl.sum(is_eq.to(tl.int32)))


@triton.jit
def _topk_sel_scan_kernel(
    counts_ptr,
    offs_ptr,
    out_ptr,
    B,
    NBP: tl.constexpr,
    NB: tl.constexpr,
    K: tl.constexpr,
):
    """Exclusive prefix over per-block selection counts, and pre-fill the
    output with -1 (slots past n_selected stay -1)."""
    row = tl.program_id(0)
    bo = tl.arange(0, NBP)
    bm = bo < NB
    g = tl.load(counts_ptr + (bo * B + row) * 2 + 0, mask=bm, other=0)
    e = tl.load(counts_ptr + (bo * B + row) * 2 + 1, mask=bm, other=0)
    tl.store(offs_ptr + (bo * B + row) * 2 + 0, tl.cumsum(g) - g, mask=bm)
    tl.store(offs_ptr + (bo * B + row) * 2 + 1, tl.cumsum(e) - e, mask=bm)
    ko = tl.arange(0, K)
    tl.store(out_ptr + row * K + ko, tl.full([K], -1, tl.int32))


@triton.jit
def _topk_sel_write_kernel(
    score_ptr,
    lengths_ptr,
    row_starts_ptr,
    state_ptr,
    offs_ptr,
    out_ptr,
    L,
    B,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
    PREWINDOWED: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
):
    """Write the selected elements' indices into the output.

    (key > tau) elements all go, at deterministic per-block offsets; among
    (key == tau) only the first needed_eq in ascending index order go, which
    reproduces torch.topk's (value desc, index asc) tie-break exactly.
    """
    pid = tl.program_id(0)
    row = tl.program_id(1)
    v, s, inw, outside = _load_window_chunk(
        score_ptr,
        lengths_ptr,
        row_starts_ptr,
        row,
        pid,
        L,
        BLOCK,
        PREWINDOWED,
        HAS_ROW_STARTS,
    )
    if outside:
        return
    base = state_ptr + row * _STATE_LEN
    total_gt = tl.load(base + _S_TOTAL_GT)
    needed_eq = tl.load(base + _S_NEEDED_EQ)
    if tl.load(base + _S_SHORT) != 0:
        is_gt = inw
        is_eq = inw & (s != s)
    else:
        tau = tl.load(base + _S_TAU).to(tl.uint32, bitcast=True)
        is_gt = inw & (s > tau)
        is_eq = inw & (s == tau)
    rs = 0
    if HAS_ROW_STARTS:
        rs = tl.load(row_starts_ptr + row)
    src = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    if PREWINDOWED:
        # Out-of-window positions are -inf by contract; the torch fallback
        # maps every selected -inf to -1.
        sel_idx = tl.where(v == float("-inf"), -1, src - rs)
    else:
        sel_idx = src - rs
    goff = tl.load(offs_ptr + (pid * B + row) * 2 + 0)
    rank_g = tl.cumsum(is_gt.to(tl.int32)) - is_gt.to(tl.int32)
    dest = goff + rank_g
    tl.store(out_ptr + row * K + dest, sel_idx, mask=is_gt & (dest < K))
    eoff = tl.load(offs_ptr + (pid * B + row) * 2 + 1)
    grank = eoff + tl.cumsum(is_eq.to(tl.int32)) - is_eq.to(tl.int32)
    eq_take = is_eq & (grank < needed_eq)
    dest_e = total_gt + grank
    tl.store(out_ptr + row * K + dest_e, sel_idx, mask=eq_take & (dest_e < K))


def triton_topk_supports(score: torch.Tensor, topk: int) -> bool:
    return (
        score.is_cuda
        and score.dtype == torch.float32
        and score.dim() == 2
        and score.is_contiguous()
        and 16 <= topk <= _TOPK_MAX
        and (topk & (topk - 1)) == 0
        and score.shape[1] <= _L_MAX
    )


_WORKSPACES = {}


def _get_workspace(B, L, topk, device):
    nb = triton.cdiv(L, _BLOCK)
    nbp = triton.next_power_of_2(nb)
    key = (B, L, topk, device.index)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = {
            "bins": torch.zeros(B, _NBINS.value, dtype=torch.int32, device=device),
            "state": torch.empty(B, _STATE_LEN.value, dtype=torch.int32, device=device),
            "counts": torch.empty(nb, B, 2, dtype=torch.int32, device=device),
            "offs": torch.empty(nb, B, 2, dtype=torch.int32, device=device),
            "nb": nb,
            "nbp": nbp,
        }
        _WORKSPACES[key] = ws
    return ws


def triton_topk_v2(score, lengths, topk, row_starts=None, prewindowed=False):
    """Exact top-k indices over each row's valid window; see module docstring.

    Contract (identical to ``_torch_fast_topk_v2`` up to ordering of the
    selected slots): int32 [B, topk], window-relative when ``row_starts`` is
    given, -1 for slots with no valid element (prewindowed: for slots whose
    value is -inf).
    """
    B, L = score.shape
    device = score.device
    if lengths.dtype != torch.int32:
        lengths = lengths.to(torch.int32)
    has_rs = row_starts is not None
    if has_rs and row_starts.dtype != torch.int32:
        row_starts = row_starts.to(torch.int32)
    ws = _get_workspace(B, L, topk, device)
    rs_arg = row_starts if has_rs else lengths
    out = torch.empty(B, topk, dtype=torch.int32, device=device)
    grid = (ws["nb"], B)
    for rnd in (0, 1, 2):
        _topk_hist_kernel[grid](
            score,
            lengths,
            rs_arg,
            ws["bins"],
            ws["state"],
            L,
            BLOCK=_BLOCK,
            ROUND=rnd,
            PREWINDOWED=prewindowed,
            HAS_ROW_STARTS=has_rs,
            num_warps=_NUM_WARPS,
        )
        _topk_scan_kernel[(B,)](
            ws["bins"],
            ws["state"],
            topk,
            NBINS=_NBINS,
            ROUND=rnd,
            num_warps=4,
        )
    _topk_sel_count_kernel[grid](
        score,
        lengths,
        rs_arg,
        ws["state"],
        ws["counts"],
        L,
        B,
        BLOCK=_BLOCK,
        PREWINDOWED=prewindowed,
        HAS_ROW_STARTS=has_rs,
        num_warps=_NUM_WARPS,
    )
    _topk_sel_scan_kernel[(B,)](
        ws["counts"],
        ws["offs"],
        out,
        B,
        NBP=ws["nbp"],
        NB=ws["nb"],
        K=topk,
        num_warps=4,
    )
    _topk_sel_write_kernel[grid](
        score,
        lengths,
        rs_arg,
        ws["state"],
        ws["offs"],
        out,
        L,
        B,
        BLOCK=_BLOCK,
        K=topk,
        PREWINDOWED=prewindowed,
        HAS_ROW_STARTS=has_rs,
        num_warps=_NUM_WARPS,
    )
    return out
