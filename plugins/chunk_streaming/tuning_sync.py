# SPDX-License-Identifier: Apache-2.0
"""Kernel tuning shared between the main process and its decode workers, so that a split decode is bit-identical.

Several kernel libraries pick a launch configuration per shape by timing their candidates on first use, and keep the winners in a
per-process table that is read once from a cache file under ~/.triton/autotune and rewritten after each tuning:

- WGPKernels' INT8 and FP8 GEMMs (`wgpkernels.int8._tuned`, `wgpkernels.fp8._tuned`, plus the `_plans` built from them);
- WanGP's Triton kernels (`shared.kernels.triton_tuner.LaunchTuner` instances: the VAE's conv3d, group norm and INT8 convrot GEMM
  among them).

Candidates differ in tiling and split-K, so they round differently. Two processes that tune the same shape at the same time (the
main process and a decode worker at the first split decode of a new size) can pick different winners, and the worker's clips then
differ from the main GPU's in the last bits: in the H3 pipeline, 2.5-5% of their uint8 values. The main process therefore tunes first
(it decodes the first clip of a new clip shape itself) and sends its tables (`snapshot`), and the worker adopts them (`receive`)
before it decodes. Entries are keyed by GPU name, so they only carry over between GPUs of the same model.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading

_MODULES = ("wgpkernels.int8", "wgpkernels.fp8")
_LOCK = threading.Lock()
_TUNERS: dict = {"count": -1, "found": []}  # LaunchTuner instances, rescanned when the set of loaded modules changes
_PENDING: dict[str, dict] = {}  # received entries whose table is not loaded in this process yet


def enabled() -> bool:
    """Off with CHUNK_STREAMING_TUNING_SYNC=0 (the decode then splits from the first clip, and may differ in the last bits)."""
    return os.environ.get("CHUNK_STREAMING_TUNING_SYNC", "1") != "0"


def _launch_tuners() -> list:
    module = sys.modules.get("shared.kernels.triton_tuner")
    tuner_class = getattr(module, "LaunchTuner", None)
    if tuner_class is None:
        return []
    with _LOCK:
        if _TUNERS["count"] == len(sys.modules):
            return list(_TUNERS["found"])
        found, seen = [], set()
        for loaded in list(sys.modules.values()):
            try:
                values = list(vars(loaded).values())
            except TypeError:
                continue
            for value in values:
                if isinstance(value, tuner_class) and id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
        _TUNERS.update(count=len(sys.modules), found=found)
        return list(found)


def _plain(configs: dict) -> dict:
    return {str(key): list(value) if isinstance(value, (list, tuple)) else value for key, value in configs.items()}


def snapshot() -> dict:
    """{table: {key: config}} of every tuning table loaded in this process (tables never tuned here are left out)."""
    out = {}
    for name in _MODULES:
        tuned = getattr(sys.modules.get(name), "_tuned", None)
        if isinstance(tuned, dict) and tuned:
            out[f"module:{name}"] = _plain(tuned)
    for tuner in _launch_tuners():
        tuned = getattr(tuner, "_tuned", None)
        if isinstance(tuned, dict) and tuned:
            out[f"tuner:{tuner.path.name}"] = _plain(tuned)
    return out


def size(snap: dict) -> int:
    """Number of entries: tables only grow, so an unchanged size means nothing new to send."""
    return sum(len(configs) for configs in snap.values())


def _apply_module(name: str, configs: dict) -> bool:
    try:
        module = sys.modules.get(name) or importlib.import_module(name)
    except Exception:  # noqa: BLE001 - the library is not installed in this process: nothing to align
        return False
    loader = getattr(module, "_cache", None)
    tuned = loader() if callable(loader) else getattr(module, "_tuned", None)
    if not isinstance(tuned, dict):
        return False
    tuned.update(configs)
    plans = getattr(module, "_plans", None)
    if isinstance(plans, dict):
        plans.clear()  # plans are built from the tuned choice of their shape: rebuild them from the adopted one
    return True


def _apply_tuner(file_name: str, configs: dict) -> bool:
    for tuner in _launch_tuners():
        if tuner.path.name == file_name:
            tuner._cache().update(configs)
            return True
    return False


def flush() -> int:
    """Apply the received entries whose table is loaded now; the others wait for a later flush. Returns the entries applied."""
    applied = 0
    with _LOCK:
        pending = dict(_PENDING)
    for source, configs in pending.items():
        kind, _, name = source.partition(":")
        done = _apply_module(name, configs) if kind == "module" else _apply_tuner(name, configs) if kind == "tuner" else True
        if done:
            applied += len(configs)
            with _LOCK:
                if _PENDING.get(source) is configs:
                    del _PENDING[source]
    return applied


def receive(snap: dict) -> int:
    """Adopt another process's tables (its entries win over this process's own); returns the entries applied now."""
    with _LOCK:
        for source, configs in snap.items():
            merged = dict(_PENDING.get(source, {}))
            merged.update(configs)
            _PENDING[source] = merged
    return flush()


def pending() -> int:
    with _LOCK:
        return sum(len(configs) for configs in _PENDING.values())
