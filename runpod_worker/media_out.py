"""Output transport chain + ``ffprobe`` metadata for the RunPod worker.

Imports nothing heavy at module scope: no torch, no wgp, no CUDA, and no
third-party package either — ``boto3`` (which ships as a ``runpod`` dependency)
and ``runpod``'s own ``rp_upload`` are imported *inside* the functions that need
them, so this module stays importable, and testable, on a bare CPU runner.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
**A container-local path is never a result.** PR #317 returned
``{"video_path": "/app/output/….mp4"}`` — a path inside an ephemeral container
that no client can ever read. Every successful return from :func:`deliver` is a
remote URL, a base64 blob, or a network-volume-relative path, and
``_assert_no_local_path`` re-checks that before returning.

THE TRANSPORT CHAIN
-------------------
``auto`` (the default) tries, in order:

1. ``presigned``  — a caller-supplied presigned **PUT** URL. Best for
   multi-tenant use: no storage credentials ever live on the worker.
2. ``rp_bucket``  — our own S3-compatible bucket, via ``rp_upload`` (or boto3
   directly with ``WANGP_S3_DIRECT=1``). Returns a presigned GET URL.
3. ``base64``     — only under ``WANGP_B64_OUT_MAX`` (6 MB). RunPod's ``/run``
   payload cap is 10 MB and base64 inflates by 4/3, so this is a debug
   affordance, not a transport.

...and then fails with ``output_too_large`` naming the env vars that fix it.
Never a truncated payload, never a dead local path.

``volume`` (copy to the network volume) is implemented and reachable with an
explicit ``output.mode`` or via ``WANGP_OUTPUT_CHAIN``, but is **not** in the
default chain: RunPod's volume S3 API cannot presign, so a volume "success"
hands a remote caller a path they have no way to read. Silent undeliverability
is worse than a loud error.

THE SILENT-DATA-LOSS BUG THIS GUARDS
------------------------------------
``runpod/serverless/utils/rp_upload.py`` (``:300-301``) does::

    if boto_client is None:
        return _save_to_local_fallback(file_name, source_path=...)

It does **not** raise. It writes to ``./local_upload/`` and returns a filesystem
path that dies with the worker. Every upload result is therefore checked for an
``http`` prefix before it is allowed to become a URL.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from . import config as C
from .errors import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    NO_OUTPUT,
    OUTPUT_TOO_LARGE,
    UPLOAD_FAILED,
    WorkerError,
)
from .media_in import check_url_target, open_pinned_connection
from .obs import LOG

__all__ = [
    "TRANSPORTS",
    "deliver",
    "ffprobe",
    "ffprobe_binary",
    "sha256_file",
    "guess_content_type",
    "object_key",
    "http_put",
    "find_existing",
    "bucket_configured",
    "default_chain",
]

#: Canonical transport names. ``output.mode`` also accepts the aliases in
#: ``_MODE_ALIASES``.
TRANSPORTS: tuple[str, ...] = ("presigned", "rp_bucket", "volume", "base64")

_MODE_ALIASES: dict[str, str] = {
    "auto": "auto",
    "presigned": "presigned",
    "presigned_url": "presigned",
    "url": "presigned",
    "put": "presigned",
    "rp_bucket": "rp_bucket",
    "bucket": "rp_bucket",
    "s3": "rp_bucket",
    "rp_upload": "rp_bucket",
    "volume": "volume",
    "network_volume": "volume",
    "base64": "base64",
    "b64": "base64",
    "inline": "base64",
}

_CONTENT_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".json": "application/json",
}

_STREAM_CHUNK = 1024 * 1024


# ===========================================================================
# Small helpers
# ===========================================================================

def guess_content_type(path: str | os.PathLike[str]) -> str:
    """MIME type from the extension we ourselves produced. Never from a caller."""
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_STREAM_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket_configured(cfg=None) -> bool:
    """Whether the bucket env vars ``rp_upload``/boto3 need are all present."""
    cfg = cfg or C.CONFIG
    getter = getattr(cfg, "bucket_configured", None)
    if isinstance(getter, bool):
        return getter
    return all(
        os.environ.get(key)
        for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
    )


def default_chain() -> list[str]:
    """The ``auto`` order. Override with ``WANGP_OUTPUT_CHAIN``."""
    raw = os.environ.get("WANGP_OUTPUT_CHAIN", "presigned,rp_bucket,base64")
    chain: list[str] = []
    for part in raw.split(","):
        name = _MODE_ALIASES.get(part.strip().lower())
        if name and name != "auto" and name not in chain:
            chain.append(name)
    return chain or ["presigned", "rp_bucket", "base64"]


def object_key(
    job_id: str,
    filename: str | os.PathLike[str],
    *,
    model_type: str | None = None,
    prefix: str | None = None,
) -> str:
    """The storage key for a job's output: ``<prefix>/<model_type>/<job_id><ext>``.

    Derived from the job id, so a client retry (or RunPod's own ``/retry``) maps
    to the same object and :func:`find_existing` can short-circuit the whole
    generation — failure mode 23, at a cost of zero GPU seconds.
    """
    prefix = (prefix if prefix is not None else os.environ.get("WANGP_S3_PREFIX", "wangp")).strip("/")
    model_type = (model_type or os.environ.get("WANGP_MODEL_TYPE", "")).strip("/")
    suffix = Path(str(filename)).suffix or ".mp4"
    safe_job = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(job_id))[:128]
    parts = [part for part in (prefix, model_type) if part]
    parts.append(f"{safe_job}{suffix}")
    return "/".join(parts)


def _expires_in() -> int:
    try:
        return max(60, int(os.environ.get("WANGP_S3_EXPIRES_S", str(7 * 24 * 3600))))
    except (TypeError, ValueError):
        return 7 * 24 * 3600


# ===========================================================================
# ffprobe
# ===========================================================================

def ffprobe_binary() -> str | None:
    """Locate ``ffprobe``: env, then PATH, then WanGP's own ``ffmpeg_bins``.

    ``shared/api.py:1088`` calls ``download_ffmpeg()`` on every runtime init,
    which drops the binaries in ``<repo root>/ffmpeg_bins`` and prepends that
    directory to ``PATH`` (``shared/ffmpeg_setup.py``). Before that has run — or
    in a CPU-only test — there may be no ffprobe at all, which is why every
    caller of :func:`ffprobe` must tolerate an empty result.
    """
    explicit = os.environ.get("WANGP_FFPROBE", "").strip()
    if explicit:
        return explicit if Path(explicit).is_file() or shutil.which(explicit) else None
    found = shutil.which("ffprobe")
    if found:
        return found
    for candidate in (
        Path(os.environ.get("WANGP_ROOT") or C.WANGP_ROOT) / "ffmpeg_bins" / "ffprobe",
        Path(__file__).resolve().parents[1] / "ffmpeg_bins" / "ffprobe",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _fraction(text: Any) -> float | None:
    try:
        raw = str(text or "").strip()
        if not raw or raw in ("0/0", "N/A"):
            return None
        if "/" in raw:
            num, _, den = raw.partition("/")
            denominator = float(den)
            if denominator == 0:
                return None
            return float(num) / denominator
        return float(raw)
    except (TypeError, ValueError):
        return None


def _tidy(value: float | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    rounded = round(float(value), digits)
    return int(rounded) if rounded == int(rounded) else rounded


def ffprobe(path: str | os.PathLike[str], *, timeout_s: float = 30.0) -> dict[str, Any]:
    """Container/stream metadata for ``path``.

    Returns ``{}``-plus-``probe_error`` rather than raising when ffprobe is
    missing, too old, times out, or cannot parse the file: metadata is a nicety,
    and a finished 8 MB video must not be thrown away because a probe failed.
    """
    binary = ffprobe_binary()
    if not binary:
        return {"probe_error": "ffprobe not found (set WANGP_FFPROBE or install ffmpeg)"}
    command = [
        binary, "-v", "error", "-hide_banner",
        "-print_format", "json", "-show_format", "-show_streams", str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, capture_output=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"probe_error": f"ffprobe failed: {type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        stderr = (completed.stderr or b"").decode("utf-8", "replace").strip()
        return {"probe_error": f"ffprobe exit {completed.returncode}: {stderr[:400]}"}
    try:
        payload = json.loads(completed.stdout or b"{}")
    except (ValueError, TypeError) as exc:
        return {"probe_error": f"ffprobe returned unparseable JSON: {exc}"}

    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    meta: dict[str, Any] = {
        "container": Path(str(path)).suffix.lstrip(".").lower() or None,
        "format_name": fmt.get("format_name"),
        "duration_s": _tidy(_fraction(fmt.get("duration"))),
        "has_audio": audio is not None,
        "has_video": video is not None,
    }
    bit_rate = _fraction(fmt.get("bit_rate"))
    if bit_rate:
        meta["bit_rate"] = int(bit_rate)
    if video is not None:
        fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
        meta.update({
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": _tidy(fps),
            "video_codec": video.get("codec_name"),
            "pix_fmt": video.get("pix_fmt"),
        })
        frames = video.get("nb_frames")
        if frames and str(frames).isdigit():
            meta["frame_count"] = int(frames)
        if meta.get("duration_s") is None:
            meta["duration_s"] = _tidy(_fraction(video.get("duration")))
    if audio is not None:
        sample_rate = _fraction(audio.get("sample_rate"))
        meta.update({
            "audio_codec": audio.get("codec_name"),
            "audio_sample_rate": int(sample_rate) if sample_rate else None,
            "audio_channels": audio.get("channels"),
        })
    return {key: value for key, value in meta.items() if value is not None}


# ===========================================================================
# Transport 1: caller-supplied presigned PUT
# ===========================================================================

def http_put(
    url: str,
    path: str | os.PathLike[str],
    content_type: str = "video/mp4",
    *,
    timeout_s: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    max_redirects: int = 2,
) -> dict[str, Any]:
    """Stream ``path`` to ``url`` with a PUT. Raises ``upload_failed`` on anything
    that is not a 2xx.

    The URL comes from the request, so it is a caller-steered outbound
    connection and gets the same SSRF treatment as a URL *input*: scheme and
    port allow-list, DNS resolved once here, and the socket pinned to the
    address that was checked (``media_in.check_url_target``). A private S3
    endpoint therefore needs ``ALLOW_URL_PRIVATE_HOSTS=1``, deliberately.
    """
    source = Path(path)
    size = source.stat().st_size
    timeout = float(timeout_s if timeout_s is not None else os.environ.get("WANGP_PUT_TIMEOUT_S", "600"))
    current = str(url)
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(size),
        "User-Agent": "wangp-runpod-worker/1",
        "Connection": "close",
    }
    for key, value in (extra_headers or {}).items():
        headers[str(key)] = str(value)

    started = time.monotonic()
    for hop in range(max_redirects + 1):
        target = check_url_target(current, purpose="output presigned")
        split = urlsplit(target.url)
        request_path = split.path or "/"
        if split.query:
            request_path = f"{request_path}?{split.query}"
        conn = open_pinned_connection(target, timeout=timeout)
        try:
            with open(source, "rb") as body:
                # http.client streams any object with .read() in blocks, as long
                # as Content-Length is supplied — which it is, above. Nothing is
                # ever slurped into memory.
                conn.request("PUT", request_path, body=body, headers=headers)
                response = conn.getresponse()
                status, reason = response.status, response.reason
                payload = response.read(2048)
        except WorkerError:
            raise
        except OSError as exc:
            raise WorkerError(
                UPLOAD_FAILED,
                f"presigned PUT to {target.host} failed: {type(exc).__name__}: {exc}",
                cause=exc,
            ) from exc
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

        if status in (301, 302, 307, 308) and hop < max_redirects:
            location = None
            try:
                location = response.getheader("Location")
            except Exception:  # noqa: BLE001 - response already consumed
                location = None
            if location:
                current = location if "://" in location else target.url.rsplit("/", 1)[0] + "/" + location.lstrip("/")
                continue
        if 200 <= status < 300:
            return {
                "status": status,
                "bytes": size,
                "duration_s": round(time.monotonic() - started, 3),
                "etag": None,
            }
        detail = payload.decode("utf-8", "replace").strip()[:400]
        raise WorkerError(
            UPLOAD_FAILED,
            f"presigned PUT rejected with HTTP {status} {reason}",
            details=[f"host: {target.host}", f"body: {detail}" if detail else "empty body",
                     "the URL must be a PUT signature, and its signed Content-Type "
                     "(if any) must match output.content_type"],
        )
    raise WorkerError(UPLOAD_FAILED,  # pragma: no cover - loop returns or raises
                      f"too many redirects following the presigned PUT for {url.split('?')[0]}")


# ===========================================================================
# Transport 2: our own S3-compatible bucket
# ===========================================================================

def _bucket_name() -> str | None:
    return (os.environ.get("BUCKET_NAME") or "").strip() or None


def _public_url(key: str) -> str | None:
    base = (os.environ.get("WANGP_S3_PUBLIC_BASE_URL") or "").strip()
    return f"{base.rstrip('/')}/{key}" if base else None


def _boto3_client():
    """An S3 client for the ``BUCKET_*`` env vars. Raises ``upload_failed``.

    boto3 arrives as a ``runpod`` dependency; it is imported here rather than at
    module scope so this file stays importable without it.
    """
    try:
        import boto3  # noqa: PLC0415 - deliberately lazy
        from botocore.config import Config  # noqa: PLC0415
    except ImportError as exc:
        raise WorkerError(
            UPLOAD_FAILED,
            "boto3 is not installed, so the bucket transport is unavailable",
            details=["it ships with runpod>=1.12; check requirements-worker.txt"],
            cause=exc,
        ) from exc
    endpoint = os.environ.get("BUCKET_ENDPOINT_URL")
    return boto3.session.Session().client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("BUCKET_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("BUCKET_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("BUCKET_REGION") or "us-east-1",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def _upload_via_rp_upload(path: Path, key: str, content_type: str) -> str:
    """``rp_upload.upload_file_to_bucket``, with the local-fallback trap defused."""
    try:
        from runpod.serverless.utils import rp_upload  # noqa: PLC0415 - lazy
    except ImportError as exc:
        raise WorkerError(
            UPLOAD_FAILED, "the runpod package is not installed", cause=exc
        ) from exc
    import inspect  # noqa: PLC0415

    candidate = {
        "file_name": os.path.basename(key),
        "file_location": str(path),
        "prefix": os.path.dirname(key),
        "bucket_name": _bucket_name(),
        "extra_args": {"ContentType": content_type},
    }
    parameters = inspect.signature(rp_upload.upload_file_to_bucket).parameters
    # Filter to the parameters this runpod version actually declares, so a
    # renamed/removed keyword upstream degrades instead of raising TypeError.
    accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD
                         for param in parameters.values())
    kwargs = {name: value for name, value in candidate.items()
              if (accepts_kwargs or name in parameters) and value not in (None, "")}
    out = rp_upload.upload_file_to_bucket(**kwargs)
    # -----------------------------------------------------------------------
    # THE SINGLE MOST LIKELY SILENT-DATA-LOSS BUG IN THIS DESIGN.
    # rp_upload.py:300-301 --
    #     if boto_client is None:
    #         return _save_to_local_fallback(file_name, source_path=...)
    # It does NOT raise. It writes to ./local_upload/ and returns a filesystem
    # path that dies with the worker. Never trust the return value without this.
    # -----------------------------------------------------------------------
    if not isinstance(out, str) or not out.lower().startswith("http"):
        raise WorkerError(
            UPLOAD_FAILED,
            "rp_upload fell back to local disk instead of uploading "
            f"(returned {out!r})",
            details=["boto3 or the BUCKET_* credentials are missing/invalid",
                     "rp_upload.py:300-301 returns a local path instead of raising"],
        )
    return out


def _upload_via_boto3(path: Path, key: str, content_type: str) -> str:
    bucket = _bucket_name()
    if not bucket:
        raise WorkerError(UPLOAD_FAILED, "BUCKET_NAME is not set")
    client = _boto3_client()
    try:
        client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
        raise WorkerError(
            UPLOAD_FAILED, f"S3 upload failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc
    public = _public_url(key)
    if public:
        return public
    try:
        url = client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=_expires_in()
        )
    except Exception as exc:  # noqa: BLE001
        raise WorkerError(
            UPLOAD_FAILED, f"could not presign the uploaded object: {exc}", cause=exc
        ) from exc
    if not isinstance(url, str) or not url.lower().startswith("http"):
        raise WorkerError(UPLOAD_FAILED, f"presign returned {url!r}, not a URL")
    return url


def find_existing(key: str, *, cfg=None) -> dict[str, Any] | None:
    """A previously uploaded object for ``key``, or ``None``.

    The idempotency probe (failure mode 23): a client retry or RunPod's own
    ``/retry`` re-derives the same key, so an already-delivered job can return
    in milliseconds having burned zero GPU seconds. Any error is swallowed into
    ``None`` — a probe must never be able to fail a job.
    """
    if not bucket_configured(cfg) or not _bucket_name():
        return None
    try:
        client = _boto3_client()
        head = client.head_object(Bucket=_bucket_name(), Key=key)
        url = _public_url(key) or client.generate_presigned_url(
            "get_object", Params={"Bucket": _bucket_name(), "Key": key},
            ExpiresIn=_expires_in(),
        )
    except WorkerError:
        return None
    except Exception:  # noqa: BLE001 - 404 is the common case and is not an error
        return None
    if not isinstance(url, str) or not url.lower().startswith("http"):
        return None
    size = head.get("ContentLength")
    result = {
        "transport": "rp_bucket",
        "kind": "url",
        "url": url,
        "expires_in_s": None if _public_url(key) else _expires_in(),
        "content_type": head.get("ContentType") or "video/mp4",
        "key": key,
        "cached": True,
    }
    if isinstance(size, int):
        result["size_bytes"] = size
        result["bytes"] = size
    return {name: value for name, value in result.items() if value is not None}


# ===========================================================================
# Transports 3 and 4: network volume, base64
# ===========================================================================

def _volume_root() -> Path:
    return Path(os.environ.get("WANGP_VOLUME_ROOT") or C.VOLUME_ROOT)


def _copy_to_volume(path: Path, key: str) -> dict[str, Any]:
    root = _volume_root()
    if not (root.is_dir() and os.access(root, os.W_OK)):
        raise WorkerError(
            UPLOAD_FAILED,
            f"the network volume at {root} is not mounted or not writable",
        )
    dest = root / "outputs" / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    # RunPod warns that concurrent writes from many workers to one volume can
    # corrupt it; the key is namespaced by job id, so two workers never collide.
    shutil.copy2(path, dest)
    return {"transport": "volume", "kind": "volume", "volume_path": f"outputs/{key}"}


def _encode_base64(path: Path, size: int, cap: int) -> dict[str, Any]:
    if size > cap:
        raise WorkerError(
            OUTPUT_TOO_LARGE,
            f"the output is {size} B, over the {cap} B base64 cap",
            details=[
                "base64 inflates by 4/3 and RunPod's /run payload cap is 10 MB",
                "set output.presigned_url, or the BUCKET_* env vars, or raise "
                "WANGP_B64_OUT_MAX if you know the envelope can carry it",
            ],
            retryable=False,
        )
    return {
        "transport": "base64",
        "kind": "base64",
        "encoding": "base64",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


# ===========================================================================
# The chain
# ===========================================================================

def _resolve_mode(raw: Any) -> str:
    text = str(raw or "auto").strip().lower()
    mode = _MODE_ALIASES.get(text)
    if mode is None:
        raise WorkerError(
            BAD_REQUEST,
            f"output.mode '{text}' is not a transport",
            details=[f"valid: {['auto', *TRANSPORTS]}"],
        )
    return mode


def _assert_no_local_path(result: Mapping[str, Any], path: Path) -> None:
    """The PR #317 guard: a container-local path is never a result.

    ``filename`` is fine (a name, not a path); anything that reproduces the
    on-disk location of the file we just generated is not, because that
    directory is deleted at the end of the job and the container after it.
    """
    forbidden = {str(path), str(path.resolve()), str(path.parent), str(path.parent.resolve())}
    for key, value in result.items():
        if key == "filename" or not isinstance(value, str):
            continue
        if value in forbidden or any(value.startswith(item + os.sep) for item in forbidden):
            raise WorkerError(
                INTERNAL_ERROR,
                f"refusing to return a container-local path in output.{key}",
                details=[f"value: {value}",
                         "this is the PR #317 bug: the path dies with the container"],
            )


def deliver(
    path: str | os.PathLike[str],
    *,
    key: str | None = None,
    request_opts: Mapping[str, Any] | None = None,
    req: Any = None,
    cfg=None,
    job_id: str | None = None,
    model_type: str | None = None,
    probe: bool = True,
) -> dict[str, Any]:
    """Get ``path`` to the caller, and describe how.

    ``request_opts`` is the request's ``input.output`` object
    (``{"mode": ..., "presigned_url": ..., "content_type": ...}``); passing
    ``req`` instead takes ``req.output``. ``key`` is the storage key — derive it
    with :func:`object_key` so a retry lands on the same object.

    Returns a dict carrying both the documented response fields
    (``transport`` / ``url`` / ``data`` / ``volume_path`` / ``filename`` /
    ``size_bytes`` / ``sha256`` + the ffprobe block) and the short aliases
    ``kind`` (``"url"`` | ``"base64"`` | ``"volume"``), ``bytes`` and
    ``content_type``.

    Raises ``output_too_large`` when nothing in the chain can carry the file —
    never a truncated payload, and never a local path.
    """
    cfg = cfg or C.CONFIG
    source = Path(path)
    opts: Mapping[str, Any] = request_opts if request_opts is not None else (
        getattr(req, "output", None) or {}
    )
    if not isinstance(opts, Mapping):
        raise WorkerError(BAD_REQUEST, "input.output must be an object")

    if not source.is_file():
        raise WorkerError(
            NO_OUTPUT, f"the generated file is missing: {source.name}",
            details=[f"expected at {source}"], retryable=False,
        )
    size = source.stat().st_size
    if size == 0:
        raise WorkerError(
            NO_OUTPUT, f"the generated file {source.name} is empty (0 bytes)",
            retryable=False,
        )

    storage_key = key or object_key(job_id or source.stem, source.name, model_type=model_type)
    content_type = str(opts.get("content_type") or guess_content_type(source))

    meta: dict[str, Any] = {
        "filename": source.name,
        "size_bytes": size,
        "bytes": size,
        "content_type": content_type,
        "key": storage_key,
        "sha256": sha256_file(source),
    }
    if probe:
        meta.update(ffprobe(source))

    mode = _resolve_mode(opts.get("mode", "auto"))
    order = default_chain() if mode == "auto" else [mode]
    tried: list[str] = []

    for transport in order:
        try:
            if transport == "presigned":
                url = opts.get("presigned_url") or opts.get("url")
                if not url:
                    reason = "no output.presigned_url given"
                    if mode != "auto":
                        raise WorkerError(
                            BAD_REQUEST,
                            "output.mode='presigned' requires output.presigned_url "
                            "(a presigned PUT URL)",
                        )
                    tried.append(f"presigned: {reason}")
                    continue
                put = http_put(str(url), source, content_type)
                # The signature is stripped: it is a PUT credential, and echoing
                # it into a response body (and every log line) leaks write access.
                outcome = {"transport": "presigned", "kind": "url",
                           "url": str(url).split("?")[0], "upload_s": put["duration_s"]}
            elif transport == "rp_bucket":
                if not bucket_configured(cfg):
                    reason = ("BUCKET_ENDPOINT_URL / BUCKET_ACCESS_KEY_ID / "
                              "BUCKET_SECRET_ACCESS_KEY not all set")
                    if mode != "auto":
                        raise WorkerError(
                            UPLOAD_FAILED,
                            f"output.mode='{transport}' but the bucket is not configured",
                            details=[reason, "BUCKET_NAME is needed too"],
                        )
                    tried.append(f"rp_bucket: {reason}")
                    continue
                direct = os.environ.get("WANGP_S3_DIRECT", "0") == "1"
                uploader = "boto3" if direct else "rp_upload"
                started = time.monotonic()
                url = (_upload_via_boto3 if direct else _upload_via_rp_upload)(
                    source, storage_key, content_type
                )
                outcome = {"transport": "rp_bucket", "kind": "url", "url": url,
                           "uploader": uploader,
                           "upload_s": round(time.monotonic() - started, 3)}
                if not _public_url(storage_key):
                    # rp_upload hardcodes ExpiresIn=604800 (rp_upload.py:321);
                    # the direct path uses WANGP_S3_EXPIRES_S.
                    outcome["expires_in_s"] = 604800 if uploader == "rp_upload" else _expires_in()
            elif transport == "volume":
                outcome = _copy_to_volume(source, storage_key)
            elif transport == "base64":
                outcome = _encode_base64(source, size, cfg.b64_out_max)
            else:  # pragma: no cover - _resolve_mode filters this
                tried.append(f"{transport}: unknown transport")
                continue
        except WorkerError as exc:
            if mode != "auto":
                raise
            tried.append(f"{transport}: {exc.message}")
            LOG.warn("transport_failed", transport=transport, error_code=exc.code,
                     error=exc.message, next="continuing the auto chain")
            continue

        result = {**meta, **outcome}
        _assert_no_local_path(result, source)
        LOG.info("output_delivered", transport=result["transport"], key=storage_key,
                 size_bytes=size, sha256=result["sha256"][:16], tried=tried or None)
        return result

    raise WorkerError(
        OUTPUT_TOO_LARGE,
        f"no transport could deliver the {size}-byte {source.name}",
        details=tried + [
            "fixes, cheapest first: pass input.output.presigned_url (a presigned "
            "PUT URL); or set BUCKET_ENDPOINT_URL + BUCKET_ACCESS_KEY_ID + "
            "BUCKET_SECRET_ACCESS_KEY + BUCKET_NAME on the endpoint; or raise "
            "WANGP_B64_OUT_MAX (currently "
            f"{cfg.b64_out_max} B) if the result really fits in RunPod's 10 MB "
            "response envelope",
        ],
        retryable=False,
    )
