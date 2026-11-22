# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The Speedups hooks: keep the model loaded, keep the LoRAs loaded and background save, installed from outside WanGP.

Nothing in WanGP is edited. install(wgp_globals) replaces names in wgp.py's globals, which its code looks up when it runs:

- generate_media: wrapped, to know the task running (its queue, model and LoRAs) and to decide at its end, with the next queued
  task in view, what stays loaded and whether to wait for the files being written.
- offload (mmgp.offload, as wgp.py sees it): profile() puts the transformers in the co-tenant map of a model loaded with "keep the
  model loaded" on, and wraps the unload_all() of what it returns; load_loras_into_model / unload_loras_from_model /
  sync_models_loras keep and reuse LoRAs.
- gc (as wgp.py sees it): collect() skips the full garbage collections around a model kept in VRAM.
- save path: ContinuedVideo.append, save_video, combine_and_concatenate_video_with_audio_tracks, record_file_metadata and
  store_api_output_artifact defer a plain generated video's save to the background saver; the generation's "output" event submits it.

Every hook checks where it is called from (wgp.py's generate_media, the line or the locals it expects). When something does not
match this WanGP version, the hook does what WanGP does without it: a file is saved in the foreground, a model is unloaded."""

from __future__ import annotations

import functools
import gc as _gc
import inspect
import linecache
import os
import shutil
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, fields

from .background_save import BackgroundSaver, remove_placeholder


@dataclass
class Settings:
    keep_model_loaded: bool = False  # the models stay in VRAM when the next queued task uses the same model
    keep_loras_loaded: bool = False  # the LoRAs stay loaded when the next queued task selects the same ones
    background_save: bool = False  # a plain generated video is written while the next queued task runs

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        names = {field.name for field in fields(cls)}
        return cls(**{key: bool(value) for key, value in (data or {}).items() if key in names})

    def to_dict(self) -> dict:
        return asdict(self)


STATE = {"installed": False, "globals": None, "originals": {}, "settings": Settings(), "kept_model_type": None, "kept_loras": None,
         "notes": set()}
saver = BackgroundSaver(max_waiting=1)
_local = threading.local()
_REQUIRED_SAVE_LOCALS = ("video_path", "container", "sample_is_hdr", "control_audio_tracks", "source_audio_tracks", "sliding_window", "side_files")


def settings() -> Settings:
    return STATE["settings"]


def configure(new_settings: Settings) -> None:
    STATE["settings"] = new_settings


def _note(text: str) -> None:
    """Print a message once per process."""
    if text not in STATE["notes"]:
        STATE["notes"].add(text)
        print(f"[Speedups] {text}")


# --- the running generation -----------------------------------------------------------------------------------------------------

class Call:
    """One generate_media call: its task and queue, and what the hooks deferred during it."""

    def __init__(self, task, gen, model_type, send_cmd):
        self.task, self.gen, self.model_type, self.send_cmd = task, gen, model_type, send_cmd
        self.thread = threading.current_thread()
        self.model_kept = False  # the previous task left this model in VRAM
        self.skip_next_gc = False  # the next full garbage collection of generate_media is skipped (a model kept in VRAM)
        self.unload_deferred = False  # its models were not unloaded after a generation: decided at the end of the call
        self.lora_models = []  # models whose LoRA unload at the end of the task was deferred: decided at the end of the call
        self.loras_reloaded = False  # LoRAs were loaded (not reused) while the model was kept in VRAM
        self.save = None  # the file being collected for the background saver (Save), until the "output" event submits it
        self.failed = False


class Save:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.label = os.path.basename(video_path)
        self.steps = []
        self.temp_files = []  # removed once the file is written (or its save failed)
        self.renamed = {}  # an intermediate file WanGP names, as the save writes it instead


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


def lora_selection(task) -> tuple:
    params = _params(task)
    return params.get("model_type"), tuple(params.get("activated_loras") or ()), str(params.get("loras_multipliers") or "")


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
        call = Call(task, gen, model_type, send_cmd)
        call.model_kept = STATE["kept_model_type"] is not None and STATE["kept_model_type"] == model_type
        call.skip_next_gc = call.model_kept
        STATE["kept_model_type"] = None
        kept = STATE["kept_loras"]
        if kept is not None and kept[0] != lora_selection(task):  # the queue changed since: the LoRAs kept for the next task are not its own
            _release_kept_loras()
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
        if cmd == "output" and call.save is not None and current_call() is call:
            _submit(call, functools.partial(send_cmd, cmd, *args, **kwargs))  # the event follows the file it announces
            return None
        return send_cmd(cmd, *args, **kwargs)

    return send


def _finish(call: Call, result) -> None:
    """End of a generate_media call: what stays loaded for the next task, and the files still being written."""
    G = STATE["globals"]
    if call.save is not None:  # stopped between writing a file and announcing it: written now, as WanGP would have
        save, call.save = call.save, None
        try:
            for step in save.steps:
                step()
        except Exception as error:  # noqa: BLE001 - reported like a failed background save
            print(f"[Speedups] Saving {save.label} failed: {error!r}")
        _remove_files(save.temp_files)
    following = next_task(call.gen, call.task)
    ok = result is not False and not call.failed and not call.gen.get("abort", False)
    same_model_next = ok and following is not None and _params(following).get("model_type") == call.model_type
    if call.lora_models:
        if settings().keep_loras_loaded and same_model_next and lora_selection(following) == lora_selection(call.task):
            STATE["kept_loras"] = (lora_selection(call.task), list(call.lora_models))
        else:
            for model in call.lora_models:
                _unload_loras(model)
    if call.unload_deferred:
        if settings().keep_model_loaded and same_model_next:
            STATE["kept_model_type"] = call.model_type
        else:
            offloadobj = G.get("offloadobj")
            if offloadobj is not None:
                STATE["originals"]["unload_all"](offloadobj)
            _gc.collect()
            _empty_cuda_cache()
    if following is None:  # the end of the queue (or a caller that keeps none): every file is written before it returns
        for label, error in saver.wait():
            call.send_cmd("error", f"Saving {label} failed: {error}")


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


# --- keep the LoRAs loaded ------------------------------------------------------------------------------------------------------

def lora_signature(lora_paths, lora_multipliers, pinned, max_reserved):
    """What makes a task's LoRAs the ones already loaded in a model: the same files (path, size, modification time), multipliers and
    pinning. Kept on the model object itself, so a reloaded model never matches."""
    try:
        files = tuple((os.path.abspath(path), os.path.getsize(path), os.path.getmtime(path)) for path in lora_paths)
    except (OSError, TypeError):
        return None
    return files, repr(lora_multipliers), bool(pinned), max_reserved


def _unload_loras(model) -> None:
    if model is not None:
        model.__dict__.pop("_speedups_lora_signature", None)
    STATE["originals"]["offload"].unload_loras_from_model(model)  # mmgp.offload, as wgp.py had it


def _release_kept_loras() -> None:
    kept, STATE["kept_loras"] = STATE["kept_loras"], None
    for model in (kept[1] if kept else []):
        _unload_loras(model)


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

    def load_loras_into_model(self, model, lora_path, lora_multi=None, *args, **kwargs):
        call = current_call()
        options = inspect.signature(self._offload.load_loras_into_model).bind(model, lora_path, lora_multi, *args, **kwargs).arguments
        if options.get("check_only") or call is None or not settings().keep_loras_loaded:
            if not options.get("check_only"):
                model.__dict__.pop("_speedups_lora_signature", None)
            return self._offload.load_loras_into_model(model, lora_path, lora_multi, *args, **kwargs)
        STATE["kept_loras"] = None  # this task takes them over
        signature = lora_signature(lora_path, lora_multi, options.get("pinnedLora"), options.get("maxReservedLoras"))
        adapters = getattr(model, "_loras_adapters", None)
        if signature is not None and adapters and getattr(model, "_speedups_lora_signature", None) == signature and not getattr(model, "_loras_errors", None):
            # the LoRAs the last task left loaded: activated again with this task's multipliers, as a load does at its end
            names = list(adapters)
            multipliers = [lora_multi[index] if lora_multi is not None and index < len(lora_multi) else 1.0 for index in range(len(names))]
            self._offload.activate_loras(model, names, multipliers)
            return None
        if getattr(model, "_speedups_lora_signature", None) is not None:  # other LoRAs kept from the last task
            _unload_loras(model)
        result = self._offload.load_loras_into_model(model, lora_path, lora_multi, *args, **kwargs)
        if signature is not None and not getattr(model, "_loras_errors", None):
            model._speedups_lora_signature = signature
        if call.model_kept:  # its blocks stayed in VRAM: they get the GPU copies a load of them would have made
            call.loras_reloaded = True
            STATE["globals"]["offloadobj"].refresh_loras(model)
        return result

    def sync_models_loras(self, model, model2):
        result = self._offload.sync_models_loras(model, model2)
        call = current_call()
        if call is not None and call.loras_reloaded and model2 is not None:
            STATE["globals"]["offloadobj"].refresh_loras(model2)
        return result

    def unload_loras_from_model(self, model):
        call = current_call()
        if (call is not None and settings().keep_loras_loaded and model is not None and sys.exc_info()[0] is None
                and getattr(model, "_speedups_lora_signature", None) is not None):
            call.lora_models.append(model)  # the end of the task decides: kept when the next task selects the same LoRAs
            return None
        if model is not None:
            model.__dict__.pop("_speedups_lora_signature", None)
        return self._offload.unload_loras_from_model(model)


# --- background save ------------------------------------------------------------------------------------------------------------

def save_blockers(local_vars: dict, continued_video=None) -> list[str]:
    """Why a generated output must be written before the next task starts (empty: it may be written in the background). The background
    path writes a plain generated video: its frames encoded with WanGP's codec, then muxed with the generated sound."""
    missing = [name for name in _REQUIRED_SAVE_LOCALS if name not in local_vars]
    if missing:
        return [f"this WanGP's save code is not the one the plugin knows ({', '.join(missing)})"]
    L = local_vars
    windows = L.get("total_windows", 1) if L.get("sliding_window") else 1
    reasons = [(L.get("video_file") is not None, "the model wrote the video itself"), (L.get("is_image") or L.get("audio_only"), "not a video"),
               (L.get("sample_is_hdr"), "HDR"), (continued_video is not None and (continued_video.video_source is not None or continued_video.file is not None), "it continues a video"),
               (windows != 1, "several windows"), (len(L["control_audio_tracks"] or []) + len(L["source_audio_tracks"] or []) > 0, "source or control audio"),
               (L.get("any_soundtrack_processor") or L.get("audio_source") is not None or bool(L.get("replace_voice_method") and L.get("replace_voice_sample") is not None), "soundtrack or voice processing"),
               (L.get("return_flashvsr_continue_cache"), "FlashVSR continuation"), (bool(L["side_files"]), "side files")]
    return [reason for blocked, reason in reasons if blocked]


def _start_save(call: Call, frame, continued_video=None) -> Save | None:
    """A Save for the file generate_media is writing, or None when it must be written now (and why is printed)."""
    if not settings().background_save or call.save is not None:
        return None
    blockers = save_blockers(frame.f_locals, continued_video)
    if blockers:
        print(f"[Speedups] {os.path.basename(str(frame.f_locals.get('video_path', 'the file')))} is saved before the next task starts: {', '.join(blockers)}")
        return None
    call.save = Save(frame.f_locals["video_path"])
    return call.save


def _work_file(video_path: str, suffix: str) -> str:
    base = video_path.rsplit(".", 1)[0]
    return f"{base}_speedups{uuid.uuid4().hex[:8]}{suffix}"


def _reserve(path: str) -> None:
    """An empty file reserves the name until the video is written (WanGP's get_available_filename skips existing files)."""
    if not os.path.exists(path):
        open(path, "ab").close()


def _patch_continued_video(continued_video_cls) -> None:
    original = continued_video_cls.append
    STATE["originals"]["ContinuedVideo.append"] = (continued_video_cls, original)

    @functools.wraps(original)
    def append(self, frames):
        call = current_call()
        frame = _caller() if call is not None else None
        if frame is None:
            return original(self, frames)
        line = _line(frame)
        with_sound = line.startswith("save_path_tmp = continued_video.append(")
        if not (with_sound or line.startswith("continued_video_path = continued_video.append(")):
            return original(self, frames)
        save = _start_save(call, frame, self)
        if save is None:
            return original(self, frames)
        # a fresh continued video (nothing before this window) would only encode the window to a work file of its own: the save
        # encodes it instead, and the continued video stays untouched
        container = frame.f_locals["container"]
        if with_sound:  # encoded alone, muxed with the sound below (combine_and_concatenate_video_with_audio_tracks)
            path = _work_file(save.video_path, f".{container}")
            save.steps.append(functools.partial(self.encode, frames, path))
            save.temp_files.append(path)
            _reserve(save.video_path)
            return path
        # no sound: WanGP copies the returned file to the video's path. It gets an empty one, which reserves the name; the save then
        # writes the video there
        empty = _work_file(save.video_path, f".{container}")
        open(empty, "wb").close()
        save.temp_files.append(empty)
        save.steps.append(functools.partial(self.encode, frames, save.video_path))
        return empty

    append._speedups = True
    continued_video_cls.append = append


def _wrap_save_video(original):
    """WanGP's save_video, for a video without a continued video (an API call that returns the frames): deferred to the save."""
    @functools.wraps(original)
    def save_video(*args, **kwargs):
        call = current_call()
        frame = _caller() if call is not None else None
        if frame is None or not _line(frame).startswith("save_video(") or "save_file" not in kwargs:
            return original(*args, **kwargs)
        target = kwargs["save_file"]
        save = _start_save(call, frame)
        if save is None:
            return original(*args, **kwargs)
        if target == save.video_path:  # no sound: written where it belongs
            _reserve(target)
            save.steps.append(functools.partial(original, *args, **kwargs))
            return target
        # with sound: WanGP removes this temporary file after the mux, so it gets an empty one; the save writes and muxes its own
        path = _work_file(save.video_path, os.path.splitext(target)[1])
        open(target, "wb").close()
        save.renamed[target] = path
        save.temp_files.append(path)
        _reserve(save.video_path)
        save.steps.append(functools.partial(original, *args, **dict(kwargs, save_file=path)))
        return target

    save_video._speedups = True
    return save_video


def _keep_link(path: str, save: Save) -> str:
    """A name of the save's own for a file WanGP removes once the mux is called (its temporary sound tracks)."""
    if not isinstance(path, str) or not os.path.isfile(path):
        return path
    base, extension = os.path.splitext(path)
    kept = f"{base}_speedups{uuid.uuid4().hex[:8]}{extension}"
    try:
        os.link(path, kept)
    except OSError:
        shutil.copyfile(path, kept)
    save.temp_files.append(kept)
    return kept


def _wrap_combine(original):
    @functools.wraps(original)
    def combine(*args, **kwargs):
        call = current_call()
        if call is None or call.save is None or _caller() is None:
            return original(*args, **kwargs)
        save = call.save
        bound = inspect.signature(original).bind(*args, **kwargs)
        names = list(bound.arguments)
        video_in = names[1]  # (output file, video file, source tracks, new tracks, ...)
        bound.arguments[video_in] = save.renamed.get(bound.arguments[video_in], bound.arguments[video_in])
        if "new_audio_tracks" in bound.arguments:
            bound.arguments["new_audio_tracks"] = [_keep_link(path, save) for path in bound.arguments["new_audio_tracks"] or []]
        save.steps.append(functools.partial(original, *bound.args, **bound.kwargs))
        return None

    combine._speedups = True
    return combine


def _wrap_deferred(original):
    """record_file_metadata, store_api_output_artifact: after the file is written."""
    @functools.wraps(original)
    def deferred(*args, **kwargs):
        call = current_call()
        if call is None or call.save is None or _caller() is None:
            return original(*args, **kwargs)
        call.save.steps.append(functools.partial(original, *args, **kwargs))
        return None

    deferred._speedups = True
    return deferred


def _remove_files(paths) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _submit(call: Call, announce) -> None:
    save, call.save = call.save, None
    steps = save.steps + [functools.partial(_remove_files, list(save.temp_files)), announce]

    def failed():
        _remove_files(save.temp_files)
        remove_placeholder(save.video_path)

    saver.submit(save.label, steps, on_failure=failed)  # blocks while an earlier save waits: never more than one clip ahead


# --- install --------------------------------------------------------------------------------------------------------------------

_WRAPPED = {"generate_media": _wrap_generate_media, "save_video": _wrap_save_video,
            "combine_and_concatenate_video_with_audio_tracks": _wrap_combine,
            "record_file_metadata": _wrap_deferred, "store_api_output_artifact": _wrap_deferred}


def install(wgp_globals: dict, new_settings: Settings | None = None) -> list[str]:
    """Install the hooks into wgp.py's globals (the WebUI's plugin does it at start; an API script passes
    `session._ensure_runtime().module.__dict__`). Returns what this WanGP lacks (the hooks that need it stay off)."""
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
    from shared.utils.video_continuation import ContinuedVideo
    _patch_continued_video(ContinuedVideo)
    STATE["installed"] = True
    return []


def uninstall() -> None:
    if not STATE["installed"]:
        return
    G, originals = STATE["globals"], STATE["originals"]
    for name in [*_WRAPPED, "offload", "gc"]:
        G[name] = originals[name]
    if "ContinuedVideo.append" in originals:
        cls, append = originals["ContinuedVideo.append"]
        cls.append = append
    offloadobj = G.get("offloadobj")
    if offloadobj is not None:
        offloadobj.__dict__.pop("unload_all", None)
    STATE.update(installed=False, globals=None, originals={}, kept_model_type=None, kept_loras=None)
