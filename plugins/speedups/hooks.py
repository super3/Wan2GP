# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The Speedups hook: keep the model loaded between queued tasks of the same model, installed from outside WanGP.

Nothing in WanGP is edited. install(wgp_globals) replaces names in wgp.py's globals, which its code looks up when it runs:

- generate_media: wrapped, to know the task running (its queue and model) and to decide at its end, with the next queued task in
  view, whether its models stay loaded.
- offload (mmgp.offload, as wgp.py sees it): profile() puts the transformers in the co-tenant map of a model loaded with "keep the
  model loaded" on, and wraps the unload_all() of what it returns; LoRAs loaded into a model kept in VRAM get their GPU copies
  (load_loras_into_model, sync_models_loras).
- gc (as wgp.py sees it): collect() skips the full garbage collections around a model kept in VRAM.

WanGP v20.12 does the rest of what this plugin used to: NVENC output (Hardware Video Encoding), background save and kept LoRAs.
It keeps a model loaded between queued tasks only for models marked no_model_swap; this hook does it for the others.

Every hook checks where it is called from (wgp.py's generate_media, the line or the locals it expects). When something does not
match this WanGP version, the hook does what WanGP does without it: the model is unloaded."""

from __future__ import annotations

import functools
import gc as _gc
import inspect
import linecache
import sys
import threading
from dataclasses import asdict, dataclass, fields


@dataclass
class Settings:
    keep_model_loaded: bool = False  # the models stay in VRAM when the next queued task uses the same model

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        names = {field.name for field in fields(cls)}
        return cls(**{key: bool(value) for key, value in (data or {}).items() if key in names})

    def to_dict(self) -> dict:
        return asdict(self)


STATE = {"installed": False, "globals": None, "originals": {}, "settings": Settings(), "kept_model_type": None}
_local = threading.local()


def settings() -> Settings:
    return STATE["settings"]


def configure(new_settings: Settings) -> None:
    STATE["settings"] = new_settings


# --- the running generation -----------------------------------------------------------------------------------------------------

class Call:
    """One generate_media call: its task and queue, and what the hooks deferred during it."""

    def __init__(self, task, gen, model_type):
        self.task, self.gen, self.model_type = task, gen, model_type
        self.thread = threading.current_thread()
        self.model_kept = False  # the previous task left this model in VRAM
        self.skip_next_gc = False  # the next full garbage collection of generate_media is skipped (a model kept in VRAM)
        self.unload_deferred = False  # its models were not unloaded after a generation: decided at the end of the call
        self.loras_reloaded = False  # LoRAs were loaded while the model was kept in VRAM
        self.failed = False


def current_call() -> Call | None:
    call = getattr(_local, "call", None)
    return call if call is not None and call.thread is threading.current_thread() else None


def next_task(gen: dict, task) -> dict | None:
    """The task queued after this one: the WebUI, API and CLI runners keep the running task in gen["queue"] until it ends."""
    queue = list((gen or {}).get("queue") or [])
    position = next((index for index, entry in enumerate(queue) if entry is task), None)
    return None if position is None or position + 1 >= len(queue) else queue[position + 1]


def _params(task) -> dict:
    return (task or {}).get("params") or {}


def _caller(depth: int = 2):
    """The frame of wgp.py's generate_media that called the hook, or None."""
    frame = sys._getframe(depth)
    return frame if frame.f_code.co_name == "generate_media" and frame.f_globals is STATE["globals"] else None


def _line(frame, offset: int = 0) -> str:
    return linecache.getline(frame.f_code.co_filename, frame.f_lineno + offset).strip()


def _wrap_generate_media(original):
    @functools.wraps(original)
    def generate_media(task, send_cmd, *args, **kwargs):
        G = STATE["globals"]
        state, model_type = kwargs.get("state"), kwargs.get("model_type")
        gen = G["get_gen_info"](state) if isinstance(state, dict) else {}
        call = Call(task, gen, model_type)
        call.model_kept = STATE["kept_model_type"] is not None and STATE["kept_model_type"] == model_type
        call.skip_next_gc = call.model_kept
        STATE["kept_model_type"] = None
        outer = getattr(_local, "call", None)
        _local.call = call
        result = None
        try:
            result = original(task, _wrap_send_cmd(send_cmd, call), *args, **kwargs)
            return result
        finally:
            _local.call = outer
            _finish(call, result)

    generate_media._speedups = True
    return generate_media


def _wrap_send_cmd(send_cmd, call: Call):
    def send(cmd, *args, **kwargs):
        if cmd == "error":
            call.failed = True
        return send_cmd(cmd, *args, **kwargs)

    return send


def _finish(call: Call, result) -> None:
    """End of a generate_media call: whether its models stay loaded for the next task."""
    if not call.unload_deferred:
        return
    following = next_task(call.gen, call.task)
    ok = result is not False and not call.failed and not call.gen.get("abort", False)
    if settings().keep_model_loaded and ok and following is not None and _params(following).get("model_type") == call.model_type:
        STATE["kept_model_type"] = call.model_type
        return
    offloadobj = STATE["globals"].get("offloadobj")
    if offloadobj is not None:
        STATE["originals"]["unload_all"](offloadobj)
    _gc.collect()
    _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- keep the model loaded ------------------------------------------------------------------------------------------------------

def _wrap_unload_all(offloadobj) -> None:
    original = type(offloadobj).unload_all
    STATE["originals"].setdefault("unload_all", original)

    def unload_all(*args, **kwargs):
        call = current_call()
        if (call is not None and settings().keep_model_loaded and sys.exc_info()[0] is None and not call.gen.get("abort", False)
                and STATE["globals"].get("offloadobj") is offloadobj):
            frame = _caller()
            if frame is not None and "offloadobj.park()" in _line(frame, -1) and _runs_again(call, frame.f_locals):
                call.unload_deferred = call.skip_next_gc = True  # after a generation: kept, the end of the task checks again
                return None
        return original(offloadobj, *args, **kwargs)

    offloadobj.unload_all = unload_all


def _runs_again(call: Call, local_vars: dict) -> bool:
    """Whether the model runs again right after this generation: another window or sample of the task (generate_media's loop
    counters; when they are not there, the end of the task decides), or the next queued task, with the same model."""
    following = next_task(call.gen, call.task)
    if following is not None and _params(following).get("model_type") == call.model_type:
        return True
    counters = ("window_no", "total_windows", "repeat_no", "total_generation")
    if not all(name in local_vars for name in counters):
        return True
    return local_vars["window_no"] < local_vars["total_windows"] or local_vars["repeat_no"] < local_vars["total_generation"]


class _GcProxy:
    """wgp.py's gc: collect() skips the full garbage collection after a generation whose models stay in VRAM, and the one before the
    next generation when the previous task left them there."""

    def __getattr__(self, name):
        return getattr(_gc, name)

    def collect(self, *args, **kwargs):
        call = current_call()
        if call is not None and call.skip_next_gc and sys.exc_info()[0] is None and _caller() is not None:
            call.skip_next_gc = False
            return 0
        return _gc.collect(*args, **kwargs)


class _OffloadProxy:
    """mmgp.offload as wgp.py sees it, with the Speedups hooks."""

    def __init__(self, offload):
        object.__setattr__(self, "_offload", offload)

    def __getattr__(self, name):
        return getattr(self._offload, name)

    def __setattr__(self, name, value):
        setattr(self._offload, name, value)

    def profile(self, pipe, *args, **kwargs):
        if settings().keep_model_loaded and isinstance(pipe, dict) and "transformer" in pipe:
            # the transformers stay in VRAM while the model's other parts (VAEs, encoders) run: mmgp otherwise unloads them for each
            # of those and loads them back for the next clip. Read at model load, for GPUs with room for them side by side.
            co_tenants = kwargs.get("coTenantsMap")
            if co_tenants is None:
                co_tenants = kwargs["coTenantsMap"] = {}
            for name in ("transformer", "transformer2"):
                if name in pipe:
                    co_tenants.setdefault(name, "*")
        offloadobj = self._offload.profile(pipe, *args, **kwargs)
        _wrap_unload_all(offloadobj)
        return offloadobj

    def load_loras_into_model(self, model, *args, **kwargs):
        result = self._offload.load_loras_into_model(model, *args, **kwargs)
        call = current_call()
        if call is not None and call.model_kept:
            options = inspect.signature(self._offload.load_loras_into_model).bind(model, *args, **kwargs).arguments
            if not options.get("check_only"):
                # its blocks stayed in VRAM: they get the GPU copies a load of them would have made (WanGP does the same for its
                # no_model_swap models; LoRAs it reuses as they are are activated, not loaded, and need none)
                call.loras_reloaded = True
                STATE["globals"]["offloadobj"].refresh_loras(model)
        return result

    def sync_models_loras(self, model, model2, *args, **kwargs):
        result = self._offload.sync_models_loras(model, model2, *args, **kwargs)
        call = current_call()
        if call is not None and call.loras_reloaded and model2 is not None:
            STATE["globals"]["offloadobj"].refresh_loras(model2)
        return result


# --- install --------------------------------------------------------------------------------------------------------------------

_WRAPPED = {"generate_media": _wrap_generate_media}


def install(wgp_globals: dict, new_settings: Settings | None = None) -> list[str]:
    """Install the hooks into wgp.py's globals (the WebUI's plugin does it at start; an API script passes
    `session._ensure_runtime().module.__dict__`). Returns what this WanGP lacks (the hooks then stay off)."""
    if new_settings is not None:
        configure(new_settings)
    if STATE["installed"]:
        return []
    missing = [name for name in [*_WRAPPED, "offload", "gc", "get_gen_info"] if name not in wgp_globals]
    if missing:
        return missing
    STATE["globals"] = wgp_globals
    for name, wrap in _WRAPPED.items():
        STATE["originals"][name] = wgp_globals[name]
        wgp_globals[name] = wrap(wgp_globals[name])
    STATE["originals"]["offload"], STATE["originals"]["gc"] = wgp_globals["offload"], wgp_globals["gc"]
    wgp_globals["offload"] = _OffloadProxy(wgp_globals["offload"])
    wgp_globals["gc"] = _GcProxy()
    if wgp_globals.get("offloadobj") is not None:  # a model loaded before the install
        _wrap_unload_all(wgp_globals["offloadobj"])
    STATE["installed"] = True
    return []


def uninstall() -> None:
    if not STATE["installed"]:
        return
    G, originals = STATE["globals"], STATE["originals"]
    for name in [*_WRAPPED, "offload", "gc"]:
        G[name] = originals[name]
    offloadobj = G.get("offloadobj")
    if offloadobj is not None:
        offloadobj.__dict__.pop("unload_all", None)
    STATE.update(installed=False, globals=None, originals={}, kept_model_type=None)
