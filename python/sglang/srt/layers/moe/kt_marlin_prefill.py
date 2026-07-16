# SPDX-License-Identifier: Apache-2.0
"""KT GPU-prefill runner backed by a pre-repacked Marlin expert cache.

Purpose
-------
The KT layerwise full-GPU prefill fallback in ``kt_ep_wrapper.py``
(``SharedFullContext``) sources expert bytes from the KT CPU wrapper via
``wrapper.submit_write_weight_scale_to_buffer``.  That API only exists for the
native AMX/AVX2 kt-kernel backends; the LLAMAFILE (GGUF) backend cannot export
GPU-format weights, so GGUF-backed deployments had to disable GPU prefill by
setting ``--kt-gpu-prefill-token-threshold`` above the context length and eat
CPU-bound prefill (~58 tok/s for GLM-5.2 on a 3995WX).

This module provides an alternative byte source that bypasses the KT wrapper
entirely: an offline-built, per-layer cache of expert weights already in the
*final* Marlin wNa16 layout (built from the same AWQ/compressed-tensors INT4
checkpoint that serves attention/dense and the resident GPU experts — see
``tools/build_marlin_prefill_cache.py`` in the ktransformers repo root).  At
prefill time, whole layers are streamed host->GPU with double-buffered
``copy_(non_blocking=True)`` from pinned host memory, and the MoE GEMM runs
through the proven ``fused_marlin_moe`` path (sm_120-validated in this stack
for resident GPU experts).

Because the cache is already Marlin-swizzled, the per-sweep GPU-side
transpose / ``gptq_marlin_repack`` / scale-permute pipeline of
``SharedFullContext._prepare_weight_int4`` is skipped completely; a layer load
is just a handful of large cudaMemcpyAsync calls (~5.6 GB/layer for GLM-5.2,
~223 ms on PCIe 4.0 x16).

The safetensors files remain the backing store. A bounded pinned-host ring
(three layers by default) stays ahead of the GPU: while the GPU computes layer
L and the copy stream transfers L+1, a loader thread fills a reusable host slot
for a later layer from the page cache. This avoids a ~380 GiB pinned duplicate
of the on-disk cache alongside the ~410 GiB GGUF expert mapping.

Measured on the target GLM-5.2 workstation
-------------------------------------------
An isolated pinned 5.20-GiB layer H2D takes 198-253 ms (20.6-26.3 GiB/s).
A warm-page-cache, JIT-warm synthetic sweep of all 75 real-weight MoE layers at
8192 tokens takes 34.40 s (238.1 MoE-equivalent tok/s). That microbenchmark
reuses random activations/routing and excludes attention, dense layers, router,
and server overhead; it is not an end-to-end prefill result. Host refill and
Marlin compute make the PCIe-only 16.8-s calculation a floor, so larger chunks
must be measured rather than extrapolated linearly.

Env knobs
---------
    SGLANG_KT_MARLIN_PREFILL_PIN=0        use pageable host-ring buffers
    SGLANG_KT_MARLIN_PREFILL_BUFFERS=1|2  GPU buffer sets (2 enables next-layer
                                          prefetch overlap; 5.6 GB VRAM each)
    SGLANG_KT_MARLIN_PREFILL_HOST_BUFFERS host slots (default 3; ~5.2 GiB each)
    SGLANG_KT_MARLIN_PREFILL_LOAD_WORKERS host-ring loader threads (default 1)
    SGLANG_KT_MARLIN_PREFILL_NUMA_NODE    bind host allocation/loaders to this
                                          NUMA node (unset by default)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Tensors stored per layer in the cache (already in final Marlin layout).
_STREAMED_NAMES = [
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
]
_ZP_NAMES = ["w13_weight_zero_point", "w2_weight_zero_point"]

_META_FILENAME = "meta.json"


def _layer_filename(layer_idx: int) -> str:
    return f"layer_{layer_idx:03d}.safetensors"


def _parse_cpu_list(value: str) -> set[int]:
    cpus = set()
    for part in value.strip().split(","):
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            cpus.update(range(start, end + 1))
        elif part:
            cpus.add(int(part))
    return cpus


class _HostBufferSlot:
    """One reusable host layer buffer and its H2D lifetime fence."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors
        self.layer_idx: Optional[int] = None
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.generation = 0
        self.h2d_done = torch.cuda.Event()


class MarlinPrefillCache:
    """Bounded host-ring cache of Marlin-format expert weights.

    Directory layout (produced by tools/build_marlin_prefill_cache.py):
        meta.json                     quant params + shapes + layer list
        layer_003.safetensors         per-layer tensors (keys above)
        layer_004.safetensors
        ...
    """

    def __init__(
        self, cache_dir: str, expected_model_path: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / _META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"KT marlin prefill cache meta not found: {meta_path}. "
                "Build it with tools/build_marlin_prefill_cache.py first."
            )
        with meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if expected_model_path is not None:
            cached_model_path = self.meta.get("model_path")
            if cached_model_path is None:
                raise ValueError(
                    "KT marlin prefill cache metadata has no model_path; rebuild "
                    "the cache before using it for serving"
                )
            expected = Path(expected_model_path).expanduser().resolve()
            cached = Path(cached_model_path).expanduser().resolve()
            if cached != expected:
                raise ValueError(
                    "KT marlin prefill cache checkpoint mismatch: cache was built "
                    f"for {cached}, server loaded {expected}"
                )

        if int(self.meta.get("format_version", -1)) != 1:
            raise ValueError(
                "Unsupported KT marlin prefill cache format_version="
                f"{self.meta.get('format_version')!r}; expected 1"
            )
        self.num_bits: int = int(self.meta["num_bits"])
        self.group_size: int = int(self.meta["group_size"])
        self.symmetric: bool = bool(self.meta["symmetric"])
        self.is_k_full: bool = bool(self.meta.get("is_k_full", True))
        self.num_experts: int = int(self.meta["num_experts"])
        self.layer_ids: List[int] = sorted(int(x) for x in self.meta["layers"])
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("KT marlin prefill cache layer list is empty or duplicated")
        self._layer_id_set = set(self.layer_ids)

        self.pin = os.environ.get("SGLANG_KT_MARLIN_PREFILL_PIN", "1") == "1"
        self._tensor_names = list(_STREAMED_NAMES)
        if not self.symmetric:
            self._tensor_names += _ZP_NAMES

        requested_slots = max(
            1, int(os.environ.get("SGLANG_KT_MARLIN_PREFILL_HOST_BUFFERS", "3"))
        )
        self.num_host_buffers = min(requested_slots, len(self.layer_ids))
        numa_node = os.environ.get("SGLANG_KT_MARLIN_PREFILL_NUMA_NODE")
        self.numa_node = int(numa_node) if numa_node is not None else None
        self._numa_cpus = None
        if self.numa_node is not None:
            cpulist = Path(
                f"/sys/devices/system/node/node{self.numa_node}/cpulist"
            )
            if not cpulist.is_file():
                raise ValueError(
                    "SGLANG_KT_MARLIN_PREFILL_NUMA_NODE="
                    f"{self.numa_node} does not exist"
                )
            self._numa_cpus = _parse_cpu_list(cpulist.read_text())
        self._slots: List[_HostBufferSlot] = []
        self._initialized = False
        self._state_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._loader_pool: Optional[ThreadPoolExecutor] = None
        self._tensor_specs: Dict[str, tuple] = {}

    # -- bounded host ring -------------------------------------------------

    def ensure_loaded(self) -> None:
        """Allocate the bounded ring and start its initial read-ahead."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return

            self._validate_files_and_get_specs()
            original_affinity = None
            if self._numa_cpus:
                original_affinity = os.sched_getaffinity(0)
                eligible = original_affinity.intersection(self._numa_cpus)
                if not eligible:
                    raise RuntimeError(
                        f"No allowed CPUs on requested NUMA node {self.numa_node}"
                    )
                os.sched_setaffinity(0, eligible)
            try:
                self._slots = [
                    _HostBufferSlot(self._allocate_slot_tensors())
                    for _ in range(self.num_host_buffers)
                ]
            finally:
                if original_affinity is not None:
                    os.sched_setaffinity(0, original_affinity)
            n_workers = max(
                1,
                min(
                    self.num_host_buffers,
                    int(
                        os.environ.get(
                            "SGLANG_KT_MARLIN_PREFILL_LOAD_WORKERS", "1"
                        )
                    ),
                ),
            )
            self._loader_pool = ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix="kt-marlin-host",
            )
            self._initialized = True

            slot_bytes = sum(
                t.numel() * t.element_size()
                for t in self._slots[0].tensors.values()
            )
            logger.info(
                "[KT-marlin-prefill] host ring: %d x %.2f GiB = %.2f GiB "
                "(layers=%d, pin=%s, workers=%d, numa=%s, backing=%s)",
                self.num_host_buffers,
                slot_bytes / 1024**3,
                slot_bytes * self.num_host_buffers / 1024**3,
                len(self.layer_ids),
                self.pin,
                n_workers,
                self.numa_node,
                self.cache_dir,
            )

            # Prime the first H layers. The runner waits only for the first;
            # the other reads continue in parallel.
            with self._state_lock:
                for slot, layer_idx in zip(self._slots, self.layer_ids):
                    self._schedule_load_locked(slot, layer_idx, wait_event=None)

    def _validate_files_and_get_specs(self) -> None:
        from safetensors import safe_open

        expected = None
        for layer_idx in self.layer_ids:
            path = self.cache_dir / _layer_filename(layer_idx)
            if not path.is_file():
                raise FileNotFoundError(
                    f"KT marlin prefill cache layer missing: {path}"
                )
            specs = {}
            with safe_open(str(path), framework="pt", device="cpu") as f:
                missing = set(self._tensor_names).difference(f.keys())
                if missing:
                    raise ValueError(
                        f"KT marlin prefill cache layer {layer_idx} misses "
                        f"tensors: {sorted(missing)}"
                    )
                for name in self._tensor_names:
                    tensor = f.get_tensor(name)
                    specs[name] = (tensor.dtype, tuple(tensor.shape))
            if expected is None:
                expected = specs
            elif specs != expected:
                raise ValueError(
                    f"KT marlin prefill cache layer {layer_idx} tensor specs "
                    "do not match the first cached layer"
                )
        assert expected is not None
        self._tensor_specs = expected

    def _allocate_slot_tensors(self) -> Dict[str, torch.Tensor]:
        out = {}
        for name, (dtype, shape) in self._tensor_specs.items():
            try:
                out[name] = torch.empty(
                    shape,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=self.pin,
                )
            except RuntimeError as exc:
                if not self.pin:
                    raise
                logger.warning(
                    "[KT-marlin-prefill] pinned allocation failed for %s %s "
                    "(%s); using a pageable host slot",
                    name,
                    shape,
                    exc,
                )
                out[name] = torch.empty(shape, dtype=dtype, device="cpu")
        return out

    def _schedule_load_locked(
        self,
        slot: _HostBufferSlot,
        layer_idx: int,
        wait_event: Optional[torch.cuda.Event],
    ) -> None:
        assert self._loader_pool is not None
        slot.generation += 1
        generation = slot.generation
        slot.layer_idx = layer_idx
        slot.error = None
        slot.ready.clear()
        self._loader_pool.submit(
            self._load_layer_into_slot,
            slot,
            layer_idx,
            generation,
            wait_event,
        )

    def _load_layer_into_slot(
        self,
        slot: _HostBufferSlot,
        layer_idx: int,
        generation: int,
        wait_event: Optional[torch.cuda.Event],
    ) -> None:
        from safetensors import safe_open

        started = time.perf_counter()
        error = None
        try:
            if self._numa_cpus:
                eligible = os.sched_getaffinity(0).intersection(self._numa_cpus)
                if not eligible:
                    raise RuntimeError(
                        f"Loader thread has no allowed CPUs on NUMA node "
                        f"{self.numa_node}"
                    )
                os.sched_setaffinity(0, eligible)
            if wait_event is not None:
                wait_event.synchronize()
            path = self.cache_dir / _layer_filename(layer_idx)
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for name in self._tensor_names:
                    src = f.get_tensor(name)
                    dst = slot.tensors[name]
                    if src.dtype != dst.dtype or src.shape != dst.shape:
                        raise ValueError(
                            f"KT marlin prefill layer {layer_idx} tensor {name} "
                            f"changed shape/dtype: got {src.dtype} {tuple(src.shape)}, "
                            f"expected {dst.dtype} {tuple(dst.shape)}"
                        )
                    dst.copy_(src)
        except BaseException as exc:
            error = exc

        with self._state_lock:
            # Do not publish bytes from a superseded load into a reused slot.
            if slot.generation != generation or slot.layer_idx != layer_idx:
                return
            slot.error = error
            slot.ready.set()
        if error is None:
            logger.debug(
                "[KT-marlin-prefill] host layer %d ready in %.1f ms",
                layer_idx,
                (time.perf_counter() - started) * 1000,
            )

    def _slot_for_layer(self, layer_idx: int) -> _HostBufferSlot:
        while True:
            with self._state_lock:
                slot = next(
                    (s for s in self._slots if s.layer_idx == layer_idx), None
                )
                if slot is None:
                    resident = sorted(
                        s.layer_idx for s in self._slots if s.layer_idx is not None
                    )
                    raise RuntimeError(
                        "KT marlin host ring lost sequential position: "
                        f"requested layer {layer_idx}, ring has {resident}"
                    )
                ready = slot.ready
            ready.wait()
            with self._state_lock:
                if slot.layer_idx != layer_idx:
                    continue
                if slot.error is not None:
                    raise RuntimeError(
                        f"Failed to stage KT marlin prefill layer {layer_idx}"
                    ) from slot.error
                return slot

    def release_after_copy(
        self, layer_idx: int, copy_stream: torch.cuda.Stream
    ) -> None:
        """Fence a consumed host slot and refill it H layers ahead."""
        slot = self._slot_for_layer(layer_idx)
        slot.h2d_done.record(copy_stream)
        if self.num_host_buffers >= len(self.layer_ids):
            return
        pos = self.layer_ids.index(layer_idx)
        replacement = self.layer_ids[
            (pos + self.num_host_buffers) % len(self.layer_ids)
        ]
        with self._state_lock:
            if slot.layer_idx != layer_idx:
                raise RuntimeError(
                    f"KT marlin host slot for layer {layer_idx} was reused early"
                )
            self._schedule_load_locked(slot, replacement, slot.h2d_done)

    # -- accessors ---------------------------------------------------------

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._layer_id_set

    def host_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        self.ensure_loaded()
        return self._slot_for_layer(layer_idx).tensors

    def next_layer(self, layer_idx: int, distance: int = 1) -> int:
        """A later MoE layer id, wrapping to the next chunk."""
        pos = self.layer_ids.index(layer_idx)
        return self.layer_ids[(pos + distance) % len(self.layer_ids)]


class _GpuBufferSet:
    """One GPU-resident set of full-layer Marlin tensors + its readiness event."""

    def __init__(self, host_proto: Dict[str, torch.Tensor], device: torch.device):
        self.tensors = {
            name: torch.empty(t.shape, dtype=t.dtype, device=device)
            for name, t in host_proto.items()
        }
        self.ready_event = torch.cuda.Event()
        self.layer_idx: Optional[int] = None
        self.copy_done = torch.cuda.Event()


class KTMarlinPrefillRunner:
    """Streams whole layers of Marlin experts H2D and runs fused_marlin_moe.

    One process-local runner per canonical cache path/device is shared by its
    MoE layers (which execute sequentially). With two GPU buffer sets, the H2D
    copy for MoE layer l+1 is issued on a side stream as soon as layer l's
    compute is enqueued; with one set, transfer and compute serialize (lower
    VRAM, slower prefill).
    """

    def __init__(self, cache: MarlinPrefillCache, device: torch.device):
        self.cache = cache
        self.device = device
        self.num_buffers = max(
            1, min(2, int(os.environ.get("SGLANG_KT_MARLIN_PREFILL_BUFFERS", "2")))
        )
        proto = cache.host_layer(cache.layer_ids[0])
        self.buffers = [
            _GpuBufferSet(proto, device) for _ in range(self.num_buffers)
        ]
        self.copy_stream = torch.cuda.Stream(device=device)
        # Empty g_idx / sort_indices (actorder=None checkpoints).
        E = cache.num_experts
        self._empty_g_idx = torch.empty(
            (E, 0), dtype=torch.int32, device=device
        )
        vram = sum(
            t.numel() * t.element_size()
            for b in self.buffers
            for t in b.tensors.values()
        )
        logger.info(
            "[KT-marlin-prefill] runner ready: %d GPU buffer set(s), "
            "%.2f GiB VRAM, layers %d..%d",
            self.num_buffers,
            vram / 1024**3,
            cache.layer_ids[0],
            cache.layer_ids[-1],
        )

    # -- streaming ---------------------------------------------------------

    def _slot_for(self, layer_idx: int) -> _GpuBufferSet:
        return self.buffers[
            self.cache.layer_ids.index(layer_idx) % self.num_buffers
        ]

    def _issue_copy(self, layer_idx: int) -> _GpuBufferSet:
        """Enqueue H2D copies for layer_idx on the copy stream (idempotent)."""
        buf = self._slot_for(layer_idx)
        if buf.layer_idx == layer_idx:
            return buf  # already resident / in flight
        host = self.cache.host_layer(layer_idx)
        with torch.cuda.stream(self.copy_stream):
            # Do not overwrite a buffer the compute stream may still be
            # consuming: wait until the previous user's compute finished.
            if buf.layer_idx is not None:
                self.copy_stream.wait_event(buf.ready_event)
            for name, host_t in host.items():
                buf.tensors[name].copy_(host_t, non_blocking=True)
            buf.copy_done.record(self.copy_stream)
            # Refill this host slot only after its H2D reads complete. The
            # loader worker waits on the event, never the model thread.
            self.cache.release_after_copy(layer_idx, self.copy_stream)
        buf.layer_idx = layer_idx
        return buf

    def prefetch(self, layer_idx: int) -> None:
        if self.num_buffers >= 2 and self.cache.has_layer(layer_idx):
            self._issue_copy(layer_idx)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """Run this layer's routed-expert MoE fully on GPU from streamed weights.

        Mirrors CompressedTensorsWNA16MoE.apply_weights (default resident
        path), with routed_scaling_factor=None by default to match
        SharedFullContext (the model's forward_normal applies it once for
        KTEPWrapperMethod).
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        buf = self._issue_copy(layer_idx)
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(buf.copy_done)

        t = buf.tensors
        zp13 = t.get("w13_weight_zero_point")
        zp2 = t.get("w2_weight_zero_point")
        output = fused_marlin_moe(
            hidden_states,
            t["w13_weight_packed"],
            t["w2_weight_packed"],
            t["w13_weight_scale"],
            t["w2_weight_scale"],
            router_logits,
            topk_weights,
            topk_ids,
            global_num_experts=self.cache.num_experts,
            expert_map=None,
            g_idx1=self._empty_g_idx,
            g_idx2=self._empty_g_idx,
            sort_indices1=self._empty_g_idx,
            sort_indices2=self._empty_g_idx,
            w1_zeros=zp13 if not self.cache.symmetric else None,
            w2_zeros=zp2 if not self.cache.symmetric else None,
            num_bits=self.cache.num_bits,
            is_k_full=self.cache.is_k_full,
            routed_scaling_factor=routed_scaling_factor,
        )
        # Mark the buffer free for reuse only after compute is enqueued.
        buf.ready_event.record(cur)

        # Kick the prefetch of the next MoE layer (wraps to the first layer so
        # the next chunk's first MoE layer is already in flight). This MUST run
        # after ready_event.record: with an odd MoE-layer count and two buffer
        # sets, next_layer(last) wraps to the first layer, which maps to the
        # same slot this layer's GEMM still occupies. _issue_copy only waits on
        # that slot's ready_event, so prefetching before the record would let
        # the wrap-around H2D overwrite weights the current GEMM is reading.
        self.prefetch(self.cache.next_layer(layer_idx))
        return output


# -- process-local runners ---------------------------------------------------

_RUNNERS: Dict[tuple, KTMarlinPrefillRunner] = {}
_RUNNER_FAILURES = set()
_RUNNER_LOCK = threading.Lock()


def get_marlin_prefill_runner(
    cache_dir: Optional[str],
    device: torch.device,
    expected_model_path: Optional[str] = None,
) -> Optional[KTMarlinPrefillRunner]:
    """Get or lazily create a runner scoped to cache path and CUDA device."""
    if cache_dir is None:
        return None
    normalized_device = torch.device(device)
    if normalized_device.type == "cuda" and normalized_device.index is None:
        normalized_device = torch.device("cuda", torch.cuda.current_device())
    expected = (
        str(Path(expected_model_path).expanduser().resolve())
        if expected_model_path is not None
        else None
    )
    key = (str(Path(cache_dir).resolve()), str(normalized_device), expected)
    if key in _RUNNER_FAILURES:
        return None
    if key in _RUNNERS:
        return _RUNNERS[key]
    with _RUNNER_LOCK:
        if key in _RUNNERS or key in _RUNNER_FAILURES:
            return _RUNNERS.get(key)
        try:
            cache = MarlinPrefillCache(key[0], expected_model_path=expected)
            cache.ensure_loaded()
            _RUNNERS[key] = KTMarlinPrefillRunner(cache, normalized_device)
        except Exception:
            logger.exception(
                "[KT-marlin-prefill] failed to initialize runner from %s; "
                "GPU prefill via marlin cache disabled",
                cache_dir,
            )
            _RUNNER_FAILURES.add(key)
            return None
    return _RUNNERS[key]
