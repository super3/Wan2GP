# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""NVENC output codecs: "H.264 NVENC" and "HEVC NVENC" in WanGP's Configuration (Outputs, SDR Video Codec).

NVENC is the GPU's dedicated video encoder block, separate from the cores that run the models, so it encodes a clip in a fraction of
the time x264 takes on the CPU without slowing generation. Constant quality 14 (H.264) / 16 (HEVC) at preset p5, tune hq is close to
x264 CRF 10, WanGP's default. Used only where it works: the ffmpeg on PATH (or FFMPEG_BINARY) must encode a short test clip with it
here, checked once per process. A GPU without NVENC (A100, H100) or a container without the driver's video library fails that check;
videos are then saved with x264 CRF 10 and one warning gives ffmpeg's reason.

ImageIO writes WanGP's videos and otherwise runs its own ffmpeg build, made without NVENC: when the check passes, it is pointed at the
ffmpeg on PATH (IMAGEIO_FFMPEG_EXE), unless IMAGEIO_FFMPEG_EXE already names one, which must then pass the check too.

Without this plugin a saved "h264_nvenc" / "hevc_nvenc" setting falls back to x264 CRF 10 (WanGP's default for unknown codecs)."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import threading

CHOICES = [("H.264 NVENC (NVIDIA GPU encoder, fast)", "h264_nvenc"), ("HEVC NVENC (NVIDIA GPU encoder, fast)", "hevc_nvenc")]
NVENC_VIDEO_CODECS = {
    "h264_nvenc": ("h264_nvenc", "yuv420p", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "14", "-b:v", "0"]),
    "hevc_nvenc": ("hevc_nvenc", "yuv420p", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0"]),
}
FALLBACK = ("libx264", "yuv420p", ["-crf", "10"])
_ready: dict[str, bool] = {}
_lock = threading.Lock()


def ffmpeg_on_path() -> str | None:
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")


def probe(ffmpeg: str, encoder: str) -> str | None:
    """None when this ffmpeg encodes a short test clip with the encoder here, else why not."""
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=24:d=0.2",
                                 "-c:v", encoder, "-pix_fmt", "yuv420p", "-f", "null", "-"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{os.path.basename(ffmpeg)} could not run ({error})"
    if result.returncode != 0:
        lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
        own = [line for line in lines if line.startswith(f"[{encoder} @")]  # the encoder's own message says why, e.g. "Cannot load libcuda.so.1"
        return (own or lines or [f"{os.path.basename(ffmpeg)} exited with code {result.returncode}"])[0][:200]
    return None


def nvenc_ready(codec_key: str) -> bool:
    """Whether NVENC codec `codec_key` can encode here, checked once per process (see the module docstring)."""
    encoder = NVENC_VIDEO_CODECS[codec_key][0]
    with _lock:
        if encoder in _ready:
            return _ready[encoder]
        path_ffmpeg = ffmpeg_on_path()
        imageio_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
        if path_ffmpeg is None:
            reason = "no ffmpeg on PATH"
        else:
            reason = probe(path_ffmpeg, encoder)
            if reason is None and imageio_ffmpeg and os.path.realpath(imageio_ffmpeg) != os.path.realpath(path_ffmpeg):
                reason = probe(imageio_ffmpeg, encoder)
        ready = reason is None
        if ready and not imageio_ffmpeg:
            os.environ["IMAGEIO_FFMPEG_EXE"] = path_ffmpeg
        if not ready:
            print(f"[Speedups] {encoder} is not available here ({reason}): saving videos with x264 CRF 10 instead.")
        _ready[encoder] = ready
        return ready


def wrap_codec_spec(original):
    """WanGP's video_codecs._get_video_codec_spec, with the NVENC codecs."""
    @functools.wraps(original)
    def get_video_codec_spec(codec_key, container, *args, **kwargs):  # v20.12 adds hardware and ffmpeg_exe: passed on as they come
        key = str(codec_key or "").strip().lower()
        if key in NVENC_VIDEO_CODECS:
            codec, pixel_format, params = NVENC_VIDEO_CODECS[key] if nvenc_ready(key) else FALLBACK
            return codec, pixel_format, list(params)
        return original(codec_key, container, *args, **kwargs)

    get_video_codec_spec._speedups_nvenc = True
    return get_video_codec_spec


def install(video_codecs=None) -> None:
    """Add the NVENC choices to the SDR codec list (in place, before the editing codecs) and wrap the codec spec. Call it before the
    Configuration tab is built, so its dropdown lists them."""
    if video_codecs is None:
        from shared.utils import video_codecs
    choices = video_codecs.SDR_VIDEO_CODEC_CHOICES
    keys = [key for _, key in choices]
    position = keys.index("prores_422") if "prores_422" in keys else len(choices)
    for choice in reversed(CHOICES):
        if choice[1] not in keys:
            choices.insert(position, choice)
    if not getattr(video_codecs._get_video_codec_spec, "_speedups_nvenc", False):
        video_codecs._get_video_codec_spec = wrap_codec_spec(video_codecs._get_video_codec_spec)


def uninstall(video_codecs=None) -> None:
    if video_codecs is None:
        from shared.utils import video_codecs
    video_codecs.SDR_VIDEO_CODEC_CHOICES[:] = [choice for choice in video_codecs.SDR_VIDEO_CODEC_CHOICES if choice[1] not in NVENC_VIDEO_CODECS]
    if getattr(video_codecs._get_video_codec_spec, "_speedups_nvenc", False):
        video_codecs._get_video_codec_spec = video_codecs._get_video_codec_spec.__wrapped__
