import os
import shutil
import subprocess
import threading

SDR_VIDEO_CODEC_CHOICES = [
    ("x265 CRF 28 (Smaller Files)", "libx265_28"),
    ("x265 CRF 18 (Balanced)", "libx265_18"),
    ("x264 Level 8 (Balanced)", "libx264_8"),
    ("x265 CRF 8 (High Quality)", "libx265_8"),
    ("x264 Level 10 (High Quality)", "libx264_10"),
    ("x264 Lossless", "libx264_lossless"),
    ("H.264 NVENC (NVIDIA GPU encoder, fast)", "h264_nvenc"),
    ("HEVC NVENC (NVIDIA GPU encoder, fast)", "hevc_nvenc"),
    ("ProRes 422 (editing)", "prores_422"),
    ("DNxHR HQ (editing)", "dnxhr_hq"),
]

# NVENC: the GPU's dedicated video encoder block, separate from the cores that run the models, so it encodes a clip in a fraction
# of the time x264 takes on the CPU without slowing generation. Constant quality 14 (H.264) / 16 (HEVC) at the slow, high-quality
# preset is close to x264 CRF 10, WanGP's default. Used only where it works: the ffmpeg on PATH must encode with it here (a GPU
# without NVENC, such as the A100 or H100, or a container without the driver's video library cannot); otherwise videos are
# saved with x264 CRF 10, and a warning says why.
NVENC_VIDEO_CODECS = {
    "h264_nvenc": ("h264_nvenc", "yuv420p", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "14", "-b:v", "0"]),
    "hevc_nvenc": ("hevc_nvenc", "yuv420p", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0"]),
}
_NVENC_FALLBACK = ("libx264", "yuv420p", ["-crf", "10"])
_nvenc_ready = {}
_nvenc_lock = threading.Lock()


def _ffmpeg_on_path() -> str | None:
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")


def _nvenc_probe(ffmpeg: str, encoder: str) -> str | None:
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
    """Whether NVENC codec `codec_key` can encode here, probed once per process. When it can, ImageIO (which writes WanGP's videos
    and otherwise runs its own ffmpeg build, made without NVENC) is pointed at the ffmpeg on PATH, unless IMAGEIO_FFMPEG_EXE
    already names one, which must then pass the probe too."""
    encoder = NVENC_VIDEO_CODECS[codec_key][0]
    with _nvenc_lock:
        if encoder in _nvenc_ready:
            return _nvenc_ready[encoder]
        path_ffmpeg = _ffmpeg_on_path()
        imageio_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
        if path_ffmpeg is None:
            reason = "no ffmpeg on PATH"
        else:
            reason = _nvenc_probe(path_ffmpeg, encoder)
            if reason is None and imageio_ffmpeg and os.path.realpath(imageio_ffmpeg) != os.path.realpath(path_ffmpeg):
                reason = _nvenc_probe(imageio_ffmpeg, encoder)
        ready = reason is None
        if ready and not imageio_ffmpeg:
            os.environ["IMAGEIO_FFMPEG_EXE"] = path_ffmpeg
        if not ready:
            print(f"[Video] {encoder} is not available here ({reason}): saving videos with x264 CRF 10 instead.")
        _nvenc_ready[encoder] = ready
        return ready

VIDEO_CONTAINER_CHOICES = [
    ("MP4", "mp4"),
    ("MOV / QuickTime", "mov"),
    ("MKV / Matroska", "mkv"),
]

RGBA_VIDEO_OUTPUT_CHOICES = [
    ("RGBA PNG Frames (ZIP)", "png_zip"),
    ("ProRes 4444 (MOV)", "prores_4444"),
]

SUPPORTED_VIDEO_CONTAINERS = {"mkv", "mov", "mp4"}
CONFIG_VIDEO_CONTAINERS = {value for _, value in VIDEO_CONTAINER_CHOICES}
PROFESSIONAL_VIDEO_CODECS = {"prores_422", "dnxhr_hq"}
QUICKTIME_AUDIO_CODEC_KEYS = {"aac_128", "aac_192", "aac_256", "aac_320", "alac"}


def normalize_video_container(container: str | None) -> str:
    return str(container or "mp4").strip().lower() or "mp4"


def normalize_video_codec(codec_key: str | None) -> str:
    return str(codec_key or "libx264_8").strip().lower() or "libx264_8"


def normalize_video_audio_codec(codec_key: str | None) -> str:
    return str(codec_key or "aac_128").strip().lower() or "aac_128"


def get_video_container_extension(container: str | None) -> str:
    container = normalize_video_container(container)
    return f".{container}" if container in SUPPORTED_VIDEO_CONTAINERS else ".mp4"


def _get_video_codec_spec(codec_key: str | None, container: str | None) -> tuple[str, str, list[str]]:
    codec_key = normalize_video_codec(codec_key)
    container = normalize_video_container(container)
    if codec_key in NVENC_VIDEO_CODECS:
        codec, pixel_format, params = NVENC_VIDEO_CODECS[codec_key] if nvenc_ready(codec_key) else _NVENC_FALLBACK
        return codec, pixel_format, list(params)
    if codec_key == "libx264_8":
        return "libx264", "yuv420p", ["-crf", "10"]
    if codec_key == "libx264_10":
        return "libx264", "yuv420p", ["-crf", "0"]
    if codec_key == "libx265_28":
        return "libx265", "yuv420p", ["-crf", "28", "-x265-params", "log-level=none"]
    if codec_key == "libx265_18":
        return "libx265", "yuv420p", ["-crf", "18", "-x265-params", "log-level=none"]
    if codec_key == "libx265_8":
        return "libx265", "yuv420p", ["-crf", "8", "-x265-params", "log-level=none"]
    if codec_key == "libx264_lossless":
        if container == "mkv":
            return "ffv1", "rgb24", []
        return "libx264", "yuv444p", ["-crf", "0"]
    if codec_key == "prores_422":
        return "prores_ks", "yuv422p10le", ["-profile:v", "2"]
    if codec_key == "prores_4444":
        return "prores_ks", "yuva444p10le", ["-profile:v", "4", "-alpha_bits", "16"]
    if codec_key == "dnxhr_hq":
        return "dnxhd", "yuv422p", ["-profile:v", "dnxhr_hq"]
    return "libx264", "yuv420p", ["-crf", "10"]


def get_video_encode_args(codec_key: str | None, container: str | None) -> list[str]:
    codec, pixel_format, output_params = _get_video_codec_spec(codec_key, container)
    return ["-c:v", codec, *output_params, "-pix_fmt", pixel_format]


def get_imageio_codec_params(codec_key: str | None, container: str | None) -> dict:
    codec, pixel_format, output_params = _get_video_codec_spec(codec_key, container)
    # Alpha mattes must stay pixel-aligned with their source after VAE padding
    # is cropped away. ImageIO's default macroblock resizing breaks that alignment.
    return {"codec": codec, "quality": None, "pixelformat": pixel_format, "macro_block_size": 1, "output_params": [*output_params, "-hide_banner", "-nostats"]}


def validate_video_output_settings(video_codec: str | None, video_container: str | None, audio_codec: str | None = None, width: int | None = None, height: int | None = None, *, allowed_containers: set[str] | None = None) -> str | None:
    video_codec = normalize_video_codec(video_codec)
    video_container = normalize_video_container(video_container)
    audio_codec = normalize_video_audio_codec(audio_codec)
    allowed = CONFIG_VIDEO_CONTAINERS if allowed_containers is None else allowed_containers
    if video_container not in allowed:
        return f"Unsupported video container: {video_container}."
    if video_codec in PROFESSIONAL_VIDEO_CODECS and video_container not in {"mkv", "mov"}:
        return "ProRes 422 and DNxHR HQ require the MOV / QuickTime or MKV container."
    if video_container == "mov" and audio_codec == "flac":
        return "FLAC soundtracks require MP4 or MKV; choose AAC or ALAC for MOV."
    if video_container in {"mp4", "mov"} and audio_codec not in QUICKTIME_AUDIO_CODEC_KEYS and not (video_container == "mp4" and audio_codec == "flac"):
        return f"{video_container.upper()} output does not support audio codec setting '{audio_codec}'."
    if video_codec == "dnxhr_hq" and width is not None and height is not None and (int(width) < 256 or int(height) < 120):
        return "DNxHR HQ output requires a resolution of at least 256x120."
    return None
