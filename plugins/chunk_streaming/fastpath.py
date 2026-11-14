# SPDX-License-Identifier: Apache-2.0
"""Opt-in cuts of the fixed per-job overhead of a MiniMax H3 generation (prototypes, off by default).

keep_models_resident: MMGP keeps one model of the pipeline on the GPU at a time: the first forward of another model (the DiT, the
audio VAE, the video VAE in turn) calls `unload_all` first, and WanGP unloads everything again at the end of each job. On a 2x RTX PRO
4500 run (H3, profile 1) every job reloaded the DiT (about 0.85 s) and paid three `unload_all` of 0.8-1.2 s each (mostly
`torch.cuda.empty_cache`). This marks every model of WanGP's offload object as a cotenant of all the others (`cotenants_map`, "*"),
so they stay loaded together, and skips the explicit `unload_all` calls; MMGP still evicts when an allocation needs the memory.
For GPUs with room for the DiT, the VAEs and the activations together (the text encoder is not needed again for a cached prompt).
It also keeps a job's LoRAs loaded for the next job: WanGP loads them at the start of every job (`offload.load_loras_into_model`,
1.5 s of the queue time, mostly pinning their weights again) and unloads them at the end. A load of the same LoRAs with the same
multipliers and options as the last one is skipped while they are still loaded; any other set unloads the kept one first.

Turn it on with CHUNK_STREAMING_KEEP_RESIDENT=1, or `keep_models_resident(True, wgp)`.

save_from_segments: with chunk streaming, every chunk of the video is already encoded once (the live fMP4 segments: libx264, CRF 18,
preset veryfast), and WanGP then encodes all the frames again for its saved file (3.5-8 s on a 2x RTX PRO 4500 run). This writes
the saved file's video track by concatenating the segments instead (`-c copy`, about 0.1-0.25 s); WanGP adds its own audio track
afterwards as usual. It only steps in when the segments cover exactly the frames being saved (same count and size, mp4 container,
every segment written); otherwise WanGP encodes as before. The video is the segments' encode, not WanGP's chosen codec.

Turn it on with CHUNK_STREAMING_SAVE_FROM_SEGMENTS=1, or `save_from_segments(True)`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

_STATE: dict = {"originals": None, "save_video": None, "last_save": None}


def _resident(offloadobj) -> None:
    """Every model of an MMGP offload object a cotenant of all the others, and its unload_all a no-op (undone by _restore)."""
    if offloadobj is None or "_fastpath_unload_all" in vars(offloadobj):
        return
    cotenants = getattr(offloadobj, "cotenants_map", None)
    if not isinstance(cotenants, dict):
        return
    vars(offloadobj)["_fastpath_cotenants"] = dict(cotenants)
    for model_id in list(getattr(offloadobj, "models", {})):
        cotenants[model_id] = "*"
    vars(offloadobj)["_fastpath_unload_all"] = offloadobj.unload_all
    offloadobj.unload_all = lambda *args, **kwargs: None
    release = offloadobj.release

    def release_all(*args, **kwargs):  # releasing the models (a model change) unloads them for real: MMGP's release calls unload_all
        _restore(offloadobj)
        return release(*args, **kwargs)

    offloadobj.release = release_all


def _lora_key(lora_path, lora_multi, args, kwargs) -> str:
    options = {key: value for key, value in kwargs.items() if key not in ("preprocess_sd", "split_linear_modules_map")}
    return repr((tuple(str(path) for path in (lora_path or ())), lora_multi, args, sorted(options.items())))


def _wrap_loras() -> tuple:
    """(load, unload): mmgp's LoRA calls, keeping an unchanged set loaded from one job to the next."""
    from mmgp import offload

    load, unload = offload.load_loras_into_model, offload.unload_loras_from_model

    def load_loras_into_model(model, lora_path, lora_multi=None, *args, **kwargs):
        if kwargs.get("check_only") or model is None:
            return load(model, lora_path, lora_multi, *args, **kwargs)
        key = _lora_key(lora_path, lora_multi, args, kwargs)
        state = vars(model).get("_fastpath_loras")
        if state is not None and state["kept"]:
            if state["key"] == key:
                state["kept"] = False  # the same LoRAs and multipliers, still loaded
                return state["result"]
            unload(model)  # the unload skipped at the end of the last job, now that another set comes
        result = load(model, lora_path, lora_multi, *args, **kwargs)
        vars(model)["_fastpath_loras"] = {"key": key, "result": result, "kept": False}
        # MMGP copies a model's LoRA weights to the GPU when it loads the model: a model kept resident would run without the LoRAs just
        # loaded (a 2x RTX PRO 4500 run lost its 4-step LoRA this way). Unload it once, so its next forward loads it with them.
        offloadobj = getattr(_STATE.get("wgp_resident"), "offloadobj", None)
        if offloadobj is not None and "_fastpath_unload_all" in vars(offloadobj):
            vars(offloadobj)["_fastpath_unload_all"]()
        return result

    def unload_loras_from_model(model):
        state = vars(model).get("_fastpath_loras") if model is not None else None
        if state is not None and not state["kept"]:
            state["kept"] = True  # kept loaded for the next job
            return None
        return unload(model)

    offload.load_loras_into_model, offload.unload_loras_from_model = load_loras_into_model, unload_loras_from_model
    return load, unload


def _restore(offloadobj) -> None:
    if offloadobj is None or "_fastpath_unload_all" not in vars(offloadobj):
        return
    offloadobj.cotenants_map.clear()
    offloadobj.cotenants_map.update(vars(offloadobj).pop("_fastpath_cotenants"))
    for name in ("unload_all", "release"):  # the class methods again
        vars(offloadobj).pop(name, None)
    vars(offloadobj).pop("_fastpath_unload_all")


def keep_models_resident(enable: bool = True, wgp=None) -> bool:
    """Keep the pipeline's models loaded together between its stages and jobs (enable=True), or restore MMGP's one-model-at-a-time
    swapping (False); returns whether it is on. Applied to WanGP's offload object at each H3 generation (a model change makes a new
    one). `wgp`: WanGP's module (default: the loaded `wgp`)."""
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline

    wgp = wgp if wgp is not None else _STATE.get("wgp") or sys.modules.get("wgp")
    if enable and _STATE["originals"] is None:
        original = MiniMaxH3Pipeline.generate

        def generate(self, *args, **kwargs):
            _resident(getattr(_STATE.get("wgp_resident"), "offloadobj", None))
            return original(self, *args, **kwargs)

        generate.__wrapped__ = original  # inspect.signature (the USP plugin reads generate's parameters) sees the original
        _STATE["originals"], _STATE["wgp_resident"] = original, wgp
        MiniMaxH3Pipeline.generate = generate
        try:
            _STATE["loras"] = _wrap_loras()
        except Exception:  # noqa: BLE001 - no mmgp here: the models still stay resident
            _STATE["loras"] = None
    elif not enable and _STATE["originals"] is not None:
        MiniMaxH3Pipeline.generate = _STATE["originals"]
        _restore(getattr(_STATE.get("wgp_resident"), "offloadobj", None))
        if _STATE.get("loras"):
            from mmgp import offload

            offload.load_loras_into_model, offload.unload_loras_from_model = _STATE["loras"]
        _STATE["originals"] = _STATE["wgp_resident"] = _STATE["loras"] = None
    return _STATE["originals"] is not None


def _segments_cover(records: list[dict], frames: int) -> bool:
    if not records or any(not r.get("path") or r.get("error") for r in records):
        return False
    position = 0
    for record in sorted(records, key=lambda r: r["frame_start"]):
        if record["frame_start"] != position:
            return False
        position = record["frame_end"]
    return position == frames


def _refuse(reason: str) -> None:
    _STATE["last_refusal"] = reason
    return None


def save_video_from_segments(live, tensor, save_file: str, container: str, fingerprint=None) -> str | None:
    """The video track of `tensor` (C, T, H, W) written to `save_file` from the live segments that hold the same frames, or None.
    `fingerprint` (streaming.output_fingerprint of the decode the segments came from) must match the frames being saved: a video
    changed after the decode (film grain, a filter) is encoded by WanGP as usual."""
    from .streaming import matches_fingerprint

    if live is None or container != "mp4" or save_file is None or not hasattr(tensor, "shape") or tensor.dim() not in (4, 5):
        return _refuse(f"no live segments, or not an mp4 video (container {container}, {type(tensor).__name__})")
    frames, height, width = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
    if not matches_fingerprint(fingerprint, tensor):
        return _refuse(f"frames differ from the decode's ({tuple(tensor.shape)} {tensor.dtype}, decode {fingerprint[:3] if fingerprint else None})")
    records = live.close()  # every chunk of this generation was finalized before WanGP saves: drain the muxing thread
    if getattr(live, "frame_size", None) != (height, width) or not _segments_cover(records, frames):
        return _refuse(f"segments do not cover the frames (size {getattr(live, 'frame_size', None)}, {len(records)} segments)")
    out = save_file if save_file.endswith(".mp4") else os.path.splitext(save_file)[0] + ".mp4"
    started = time.monotonic()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for record in sorted(records, key=lambda r: r["frame_start"]):
            listing.write(f"file '{os.path.abspath(record['path'])}'\n")
    try:
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing.name, "-map", "0:v:0",
                               "-c", "copy", "-movflags", "+faststart", out], capture_output=True, text=True)
    finally:
        os.unlink(listing.name)
    if proc.returncode != 0:
        return _refuse(f"ffmpeg concat failed: {proc.stderr.strip()[:200]}")
    _STATE["last_save"] = {"path": out, "seconds": round(time.monotonic() - started, 3), "segments": len(records)}
    return out


def save_from_segments(enable: bool = True, wgp=None) -> bool:
    """Write WanGP's saved video from the streamed segments when they hold the same frames (enable=True), or restore WanGP's
    encode (False); returns whether it is on. `wgp`: WanGP's module (default: the loaded `wgp`)."""
    wgp = wgp if wgp is not None else _STATE.get("wgp") or sys.modules.get("wgp")
    if wgp is None or not hasattr(wgp, "save_video"):
        return False
    if enable and _STATE["save_video"] is None:
        original = wgp.save_video

        def save_video(tensor, save_file=None, fps=30, codec_type="libx264_8", container="mp4", *args, **kwargs):
            from . import streaming

            rec = streaming.recorder()
            try:
                path = _refuse("chunk streaming is off") if rec is None else save_video_from_segments(rec.live, tensor, save_file,
                                                                                                         container, rec.fingerprint)
            except Exception as exc:  # noqa: BLE001 - the prototype must never cost a saved video
                path = _refuse(f"{type(exc).__name__}: {exc}"[:300])
            if path is not None:
                return path
            return original(tensor, save_file, fps, codec_type, container, *args, **kwargs)

        _STATE["save_video"], _STATE["wgp"] = original, wgp
        wgp.save_video = save_video
    elif not enable and _STATE["save_video"] is not None:
        wgp.save_video = _STATE["save_video"]
        _STATE["save_video"] = None
    return _STATE["save_video"] is not None


def last_save() -> dict | None:
    """Path, seconds and segment count of the last video saved from segments."""
    return _STATE["last_save"]


def last_refusal() -> str | None:
    """Why the last save did not use the segments (WanGP encoded it)."""
    return _STATE.get("last_refusal")


def apply_environment() -> dict:
    """Turn on the shortcuts their environment variables ask for; returns which are on."""
    return {"keep_models_resident": keep_models_resident(True) if os.environ.get("CHUNK_STREAMING_KEEP_RESIDENT") == "1" else False,
            "save_from_segments": save_from_segments(True) if os.environ.get("CHUNK_STREAMING_SAVE_FROM_SEGMENTS") == "1" else False}
