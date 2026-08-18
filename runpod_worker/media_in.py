"""Materialize request attachments to absolute paths under a per-job temp dir.

Standard library only: no torch, no wgp, no CUDA, no third-party imports. This
module must stay importable on a plain CPU runner — ``tests/test_media.py``
depends on that.

WHY EVERY INPUT IS REWRITTEN INSTEAD OF PASSED THROUGH
------------------------------------------------------
1. WanGP validates media **by file extension only**
   (``shared/utils/utils.py:36-49``: ``has_video_file_extension`` /
   ``has_image_file_extension`` / ``has_audio_file_extension`` do
   ``os.path.splitext(...)[-1].lower() in [...]``). Nothing looks at the bytes.
   So a caller-supplied filename is a caller-supplied *decoder selection*. We
   sniff the magic bytes and name the file ourselves; the caller never picks the
   extension.
2. Paths must be **absolute**. ``WanGPSession._absolutize_setting_path``
   (``shared/api.py:1028-1043``) resolves relative attachment paths against
   ``Path.cwd()`` *at submit time*, and ``shared/api_cli.py:29`` ``chdir``s the
   whole process to the repo root for the duration of a job. A relative path
   would silently resolve somewhere else.
3. ``Path.resolve()`` in that same function **follows symlinks**, so a symlink
   in the job dir pointing at a volume file would hand WanGP the volume file's
   own (caller-controlled) extension. Symlinks are therefore never used here;
   we hardlink or copy.

The virtual-media suffix (``path|start_frame=..,end_frame=..[,audio_track_no=..]``,
``docs/API.md:452-485``) survives absolutization: ``_absolutize_setting_path``
parses it off, resolves the source path, and re-attaches the suffix via
``replace_virtual_media_source``. We build that suffix ourselves from the
``range`` object so a caller can never smuggle a path through it.

INPUT FORMS
-----------
    {"video_guide": {"b64": "<base64 or data: URI>"}}
    {"video_guide": {"volume": "clips/plate.mp4",
                     "range": {"start_frame": 0, "end_frame": 240}}}
    {"video_guide": {"url": "https://…"}}          # only when ALLOW_URL_INPUTS=1
    {"image_refs":  [{"b64": "…"}, {"b64": "…"}]}  # list-valued keys only
    {"video_guide": "volume://clips/plate.mp4"}    # string shorthand

SIZE ACCOUNTING
---------------
``b64`` and ``url`` bodies are *inline* bytes: they arrive inside (or are pulled
in because of) a 10 MB RunPod request envelope, and they are billed against
``WANGP_B64_IN_MAX`` per item and ``WANGP_MEDIA_TOTAL_MAX`` in total.
``volume`` inputs are already on the network volume — charging them against a
7 MB envelope budget would reject every real video guide — so they get their own
``WANGP_VOLUME_IN_MAX`` ceiling (2 GiB by default) and do not count toward the
inline total.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import ipaddress
import os
import re
import shutil
import socket
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin, urlsplit

from . import config as C
from .errors import (
    BAD_REQUEST,
    MEDIA_FETCH_FAILED,
    MEDIA_TOO_LARGE,
    MEDIA_UNSUPPORTED,
    SSRF_BLOCKED,
    WorkerError,
)
from .obs import LOG

__all__ = [
    "ATTACHMENT_KEYS",
    "MEDIA_KIND",
    "LIST_KEYS",
    "IMAGE_EXTS",
    "AUDIO_EXTS",
    "VIDEO_EXTS",
    "EXTS_BY_KIND",
    "SniffResult",
    "MediaItem",
    "MaterializedMedia",
    "sniff",
    "sniff_file",
    "job_dir_for",
    "resolve_volume_path",
    "materialize",
    "cleanup",
    "check_url_target",
    "open_pinned_connection",
    "URLTarget",
    "fetch_url",
    "url_inputs_enabled",
    "volume_item_max",
    "sweep",
]

# ---------------------------------------------------------------------------
# Keys and whitelists. Deliberately duplicated from schema.py rather than
# imported: schema.py reads models/_settings.json at import time, which would
# make this module unimportable wherever WANGP_ROOT is not populated.
# Verified against wgp.py:167-168 (ATTACHMENT_KEYS, 15 keys, same order) and
# shared/utils/utils.py:36-49 (the extension whitelists).
# ---------------------------------------------------------------------------
ATTACHMENT_KEYS: tuple[str, ...] = (
    "image_start", "image_end", "image_refs", "image_guide", "image_mask",
    "video_guide", "video_guide2", "video_mask", "video_source",
    "audio_guide", "audio_guide2", "audio_source",
    "replace_voice_sample", "replace_voice_sample2", "custom_guide",
)

#: NOTE: ``.webm`` is NOT accepted by WanGP; ``.avi`` is.
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
)
AUDIO_EXTS: frozenset[str] = frozenset({".wav", ".mp3", ".aac"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mkv", ".avi", ".mov"})
EXTS_BY_KIND: dict[str, frozenset[str]] = {
    "image": IMAGE_EXTS,
    "audio": AUDIO_EXTS,
    "video": VIDEO_EXTS,
}

#: Attachment key -> media kind. ``custom_guide`` is intentionally absent: it is
#: model-specific, no minimax_h3 variant consumes it, and schema.py's allow-list
#: does not admit it either. Keep the two tables identical.
MEDIA_KIND: dict[str, str] = {k: "image" for k in
                              ("image_start", "image_end", "image_refs",
                               "image_guide", "image_mask")}
MEDIA_KIND.update({k: "video" for k in
                   ("video_guide", "video_guide2", "video_mask", "video_source")})
MEDIA_KIND.update({k: "audio" for k in
                   ("audio_guide", "audio_guide2", "audio_source",
                    "replace_voice_sample", "replace_voice_sample2")})

#: Keys WanGP expects to hold a *list* of paths.
LIST_KEYS: frozenset[str] = frozenset({"image_refs"})

#: Only video slots accept a frame ``range`` (the virtual-media suffix targets a
#: decoded frame window; docs/API.md:452-467).
RANGE_KINDS: frozenset[str] = frozenset({"video"})

#: How much of the head we read for sniffing. Generous on purpose: Matroska's
#: DocType sits a little way into the EBML header, not at byte 0.
_HEAD_BYTES = 4096

_STREAM_CHUNK = 256 * 1024

_VOLUME_PREFIX_RE = re.compile(r"^volume://", re.IGNORECASE)


# ===========================================================================
# Magic-byte sniffing
# ===========================================================================

@dataclass(frozen=True)
class SniffResult:
    """What the bytes actually are, independent of any filename."""

    kind: str            # "image" | "video" | "audio"
    ext: str             # extension WanGP accepts for that kind, e.g. ".mp4"
    content_type: str    # best-effort MIME, for logs and upload headers
    format: str          # short human name, e.g. "iso-bmff/mp4"


#: Formats we can identify but WanGP will not accept for any slot. Recognising
#: them explicitly turns "unsupported media" into a message that says *why*.
_KNOWN_UNSUPPORTED: tuple[tuple[bytes, str, str], ...] = (
    (b"OggS", "ogg", "ogg/opus/vorbis is not in WanGP's whitelist (use .wav/.mp3/.aac)"),
    (b"fLaC", "flac", "flac is not in WanGP's whitelist (use .wav/.mp3/.aac)"),
    (b"FLV\x01", "flv", "flv is not in WanGP's whitelist (use .mp4/.mkv/.avi/.mov)"),
    (b"\x00\x00\x01\xba", "mpeg-ps", "mpeg program stream is not in WanGP's whitelist"),
    (b"\x1f\x8b", "gzip", "this is a gzip archive, not media"),
    (b"PK\x03\x04", "zip", "this is a zip archive, not media"),
    (b"%PDF", "pdf", "this is a PDF, not media"),
    (b"\x7fELF", "elf", "this is an executable, not media"),
)

_FTYP_MOV_BRANDS = frozenset({b"qt  "})
_FTYP_AUDIO_BRANDS = ("M4A", "M4B", "F4A")


def _sniff_riff(head: bytes) -> SniffResult | None:
    if head[:4] != b"RIFF" or len(head) < 12:
        return None
    form = head[8:12]
    if form == b"WAVE":
        return SniffResult("audio", ".wav", "audio/wav", "riff/wave")
    if form == b"AVI ":
        return SniffResult("video", ".avi", "video/x-msvideo", "riff/avi")
    if form == b"WEBP":
        return SniffResult("image", ".webp", "image/webp", "riff/webp")
    return None


def _sniff_isobmff(head: bytes) -> SniffResult | None:
    """ISO base media file format: a `ftyp` box at offset 4."""
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    brand = head[8:12]
    if brand in _FTYP_MOV_BRANDS:
        return SniffResult("video", ".mov", "video/quicktime", "iso-bmff/quicktime")
    text = brand.decode("ascii", "replace")
    if text.startswith(_FTYP_AUDIO_BRANDS):
        # m4a/m4b are audio in an MP4 container, and WanGP's audio whitelist has
        # no .m4a. Reported as unsupported rather than mislabelled .mp4.
        return None
    return SniffResult("video", ".mp4", "video/mp4", f"iso-bmff/{text.strip()}")


def _sniff_matroska(head: bytes) -> SniffResult | None:
    if head[:4] != b"\x1aE\xdf\xa3":
        return None
    # DocType lives in the EBML header, a few dozen bytes in. WebM is a Matroska
    # profile; WanGP has no .webm extension, but ffmpeg reads WebM out of a file
    # named .mkv without complaint, so we normalize both to .mkv.
    doctype = "matroska"
    window = head[:1024]
    if b"webm" in window:
        doctype = "webm"
    return SniffResult(
        "video", ".mkv",
        "video/webm" if doctype == "webm" else "video/x-matroska",
        f"ebml/{doctype}",
    )


def _sniff_mpeg_audio(head: bytes) -> SniffResult | None:
    """MPEG audio frame sync, plus ID3 and ADTS/ADIF AAC.

    An 11-bit frame sync (``0xFFE0`` masked) starts both MP3 and ADTS AAC. The
    two are told apart by the *layer* bits: ``00`` is "reserved" for MPEG audio
    and is exactly what ADTS AAC puts there.

        0xFFFA / 0xFFF3 -> layer 01 -> Layer III   -> .mp3
        0xFFF1 / 0xFFF9 -> layer 00 -> ADTS AAC    -> .aac

    A two-byte ``\\xff\\xfb``-only table rejects ordinary files; this does not.
    """
    if head[:3] == b"ID3":
        return SniffResult("audio", ".mp3", "audio/mpeg", "id3/mpeg-audio")
    if head[:4] == b"ADIF":
        return SniffResult("audio", ".aac", "audio/aac", "aac/adif")
    if len(head) < 2:
        return None
    b0, b1 = head[0], head[1]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    version = (b1 >> 3) & 0x03   # 0=MPEG2.5, 1=reserved, 2=MPEG2, 3=MPEG1
    layer = (b1 >> 1) & 0x03     # 0=reserved(ADTS AAC), 1=Layer III, 2=II, 3=I
    if version == 1:
        return None
    if layer == 0:
        return SniffResult("audio", ".aac", "audio/aac", "aac/adts")
    if layer == 1:
        return SniffResult("audio", ".mp3", "audio/mpeg", "mpeg-audio/layer3")
    # Layer I/II in the wild are still served as .mp3 and ffmpeg decodes them.
    return SniffResult("audio", ".mp3", "audio/mpeg", f"mpeg-audio/layer{4 - layer}")


def _sniff_image(head: bytes) -> SniffResult | None:
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return SniffResult("image", ".png", "image/png", "png")
    if head[:3] == b"\xff\xd8\xff":
        return SniffResult("image", ".jpg", "image/jpeg", "jpeg")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return SniffResult("image", ".gif", "image/gif", "gif")
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return SniffResult("image", ".tif", "image/tiff", "tiff")
    if head[:2] == b"BM" and len(head) >= 6:
        return SniffResult("image", ".bmp", "image/bmp", "bmp")
    return None


def sniff(head: bytes) -> SniffResult | None:
    """Identify ``head`` (the first bytes of a file). ``None`` if unrecognised.

    Order matters: container signatures that share a prefix are checked
    strongest-first, and the weak two-byte ones (BMP, MPEG audio sync) last.
    """
    if not head:
        return None
    for probe in (_sniff_isobmff, _sniff_matroska, _sniff_riff, _sniff_image):
        result = probe(head)
        if result is not None:
            return result
    return _sniff_mpeg_audio(head)


def _unsupported_reason(head: bytes) -> str | None:
    for magic, name, reason in _KNOWN_UNSUPPORTED:
        if head.startswith(magic):
            return f"{name}: {reason}"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", "replace").strip()
        return f"iso-bmff/{brand}: audio-only MP4 (.m4a) is not in WanGP's whitelist"
    return None


def sniff_file(path: str | os.PathLike[str]) -> SniffResult | None:
    """``sniff`` against the first bytes of a file on disk."""
    with open(path, "rb") as handle:
        return sniff(handle.read(_HEAD_BYTES))


def _require_kind(key: str, expected: str, head: bytes, *, source: str) -> SniffResult:
    result = sniff(head)
    if result is None:
        reason = _unsupported_reason(head) or (
            f"unrecognised magic bytes {head[:8].hex() or '(empty file)'}"
        )
        raise WorkerError(
            MEDIA_UNSUPPORTED,
            f"media.{key}: could not identify the content as {expected}",
            details=[reason, f"source: {source}",
                     f"accepted {expected} extensions: {sorted(EXTS_BY_KIND[expected])}"],
        )
    if result.kind != expected:
        raise WorkerError(
            MEDIA_UNSUPPORTED,
            f"media.{key} expects {expected} data but the bytes are "
            f"{result.kind} ({result.format})",
            details=[
                "the declared filename is never trusted; the extension is taken "
                "from the content (WanGP validates by extension only, "
                "shared/utils/utils.py:36-49)",
                f"source: {source}",
            ],
        )
    if result.ext not in EXTS_BY_KIND[expected]:  # pragma: no cover - table invariant
        raise WorkerError(
            MEDIA_UNSUPPORTED,
            f"media.{key}: {result.format} maps to '{result.ext}', which WanGP "
            f"does not accept for {expected}",
        )
    return result


# ===========================================================================
# URL inputs: SSRF guard + bounded streaming fetch
# ===========================================================================
#
# Everything below only runs when ALLOW_URL_INPUTS=1 (default "0"). It is off by
# default because a URL input turns the worker into a fetcher that a caller
# steers — the classic SSRF shape. When it is on:
#
#   * scheme allow-list (https only by default), no userinfo, port allow-list
#   * DNS is resolved *here*, every returned address is checked, and the
#     connection is pinned to the address we checked. Resolving twice (once to
#     validate, once inside urllib) is a DNS-rebinding hole; pinning closes it.
#   * every redirect hop is re-validated from scratch
#   * the body is capped while streaming, and a single deadline covers DNS,
#     connect, TLS, headers and body
# ---------------------------------------------------------------------------

#: RFC-special / infrastructure ranges that must never be reachable. Checked in
#: addition to ``is_global`` because ``is_global`` has drifted between CPython
#: releases (100.64.0.0/10 was mis-classified before 3.12.4 / CVE-2024-4032).
_BLOCKED_V4 = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32",
))
_BLOCKED_V6 = tuple(ipaddress.ip_network(n) for n in (
    "::/128", "::1/128", "100::/64", "2001:db8::/32", "fc00::/7", "fe80::/10",
    "ff00::/8",
))


def url_inputs_enabled(cfg=None) -> bool:
    cfg = cfg or C.CONFIG
    return bool(getattr(cfg, "allow_url_inputs", False))


def _allowed_schemes() -> tuple[str, ...]:
    raw = os.environ.get("WANGP_URL_SCHEMES", "https")
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _allowed_ports() -> tuple[int, ...]:
    raw = os.environ.get("WANGP_URL_PORTS", "80,443")
    ports: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ports.append(int(part))
    return tuple(ports)


def _allow_private_hosts() -> bool:
    """Test/debug escape hatch. Never set this in production.

    Needed only so an integration test can point a URL input at a loopback HTTP
    server; with it unset (the default) loopback is blocked like every other
    non-global address.
    """
    return os.environ.get("ALLOW_URL_PRIVATE_HOSTS", "0") == "1"


def _unwrap_v6(ip: ipaddress.IPv6Address) -> ipaddress._BaseAddress:
    """Follow the v4 address embedded in a mapped / 6to4 / Teredo v6 address."""
    for attr in ("ipv4_mapped", "sixtofour", "teredo"):
        value = getattr(ip, attr, None)
        if value is None:
            continue
        if attr == "teredo":
            value = value[1] if isinstance(value, tuple) else None
        if value is not None:
            return value
    return ip


def _ip_block_reason(raw: str) -> str | None:
    """``None`` when ``raw`` is a routable public address, else why it is not."""
    try:
        ip: ipaddress._BaseAddress = ipaddress.ip_address(raw)
    except ValueError:
        return f"{raw!r} is not an IP address"
    if isinstance(ip, ipaddress.IPv6Address):
        unwrapped = _unwrap_v6(ip)
        if unwrapped is not ip:
            inner = _ip_block_reason(str(unwrapped))
            if inner is not None:
                return f"{raw} embeds {unwrapped} ({inner})"
            ip = unwrapped
    networks = _BLOCKED_V4 if ip.version == 4 else _BLOCKED_V6
    for network in networks:
        if ip in network:
            return f"{ip} is in the blocked range {network}"
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return f"{ip} is loopback/link-local/multicast/unspecified"
    if ip.is_private or ip.is_reserved:
        return f"{ip} is private or reserved"
    if not ip.is_global:
        return f"{ip} is not globally routable"
    return None


@dataclass(frozen=True)
class URLTarget:
    """A URL that passed every check, plus the address the socket must use."""

    url: str
    scheme: str
    host: str
    port: int
    ip: str
    family: int


def check_url_target(url: str, *, purpose: str = "input") -> URLTarget:
    """Validate a URL and resolve it to one vetted IP. Raises on anything odd.

    Also used by ``media_out`` for caller-supplied presigned PUT targets, which
    are a caller-steered outbound request in exactly the same way.
    """
    parts = urlsplit(str(url).strip())
    scheme = (parts.scheme or "").lower()
    allowed = _allowed_schemes()
    if scheme not in allowed:
        raise WorkerError(
            SSRF_BLOCKED,
            f"{purpose} URL scheme '{scheme or '(none)'}' is not allowed",
            details=[f"allowed schemes: {list(allowed)} (WANGP_URL_SCHEMES)"],
        )
    if parts.username or parts.password:
        raise WorkerError(SSRF_BLOCKED, f"{purpose} URL must not carry credentials")
    host = parts.hostname
    if not host:
        raise WorkerError(SSRF_BLOCKED, f"{purpose} URL has no host")
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise WorkerError(SSRF_BLOCKED, f"{purpose} URL has an invalid port: {exc}") from exc
    ports = _allowed_ports()
    if ports and port not in ports:
        raise WorkerError(
            SSRF_BLOCKED,
            f"{purpose} URL port {port} is not allowed",
            details=[f"allowed ports: {list(ports)} (WANGP_URL_PORTS)"],
        )

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WorkerError(
            MEDIA_FETCH_FAILED, f"could not resolve {host!r}: {exc}", cause=exc
        ) from exc
    if not infos:
        raise WorkerError(MEDIA_FETCH_FAILED, f"{host!r} resolved to no addresses")

    permissive = _allow_private_hosts()
    reasons: list[str] = []
    chosen: tuple[int, str] | None = None
    for family, _type, _proto, _canon, sockaddr in infos:
        address = sockaddr[0]
        reason = _ip_block_reason(address)
        if reason is None or permissive:
            if chosen is None:
                chosen = (family, address)
            continue
        reasons.append(reason)
    if chosen is None:
        raise WorkerError(
            SSRF_BLOCKED,
            f"{purpose} URL host {host!r} resolves only to blocked addresses",
            details=reasons or [f"{host} resolved to nothing routable"],
        )
    if reasons:
        # Mixed answer: at least one address was private. Refuse rather than
        # race — this is what a rebinding record looks like.
        raise WorkerError(
            SSRF_BLOCKED,
            f"{purpose} URL host {host!r} resolves to a mix of public and "
            f"blocked addresses",
            details=reasons,
        )
    if permissive:
        LOG.warn("ssrf_guard_relaxed", host=host, ip=chosen[1],
                 note="ALLOW_URL_PRIVATE_HOSTS=1 is set; never use it in production")
    return URLTarget(url=parts.geturl(), scheme=scheme, host=host, port=port,
                     ip=chosen[1], family=chosen[0])


def open_pinned_connection(target: URLTarget, timeout: float) -> http.client.HTTPConnection:
    """A connection to ``target.ip`` that still presents ``target.host``.

    ``http.client`` looks up ``self.host`` for both the ``Host`` header and the
    TLS SNI / certificate check, and calls ``self._create_connection`` (an
    instance attribute assigned in ``HTTPConnection.__init__``) to open the
    socket. Replacing just that attribute pins the address without touching
    hostname verification.
    """
    if target.scheme == "https":
        context = ssl.create_default_context()
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            target.host, target.port, timeout=timeout, context=context
        )
    else:
        conn = http.client.HTTPConnection(target.host, target.port, timeout=timeout)
    pinned_ip = target.ip
    original = conn._create_connection  # type: ignore[attr-defined]

    def _connect(address, *args, **kwargs):
        return original((pinned_ip, address[1]), *args, **kwargs)

    conn._create_connection = _connect  # type: ignore[attr-defined]
    return conn


def fetch_url(
    url: str,
    dest: Path,
    *,
    max_bytes: int,
    timeout_s: float = 60.0,
    max_redirects: int = 3,
    head_out: bytearray | None = None,
) -> int:
    """Stream ``url`` into ``dest``, capped and deadlined. Returns bytes written.

    Every redirect hop is re-validated by ``check_url_target``; the byte cap is
    enforced while reading, not after; the deadline covers the whole exchange,
    so a server that trickles one byte per second cannot hold the worker open.
    """
    deadline = time.monotonic() + float(timeout_s)
    current = str(url)
    seen: list[str] = []

    for hop in range(max_redirects + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerError(MEDIA_FETCH_FAILED,
                              f"timed out after {timeout_s:.0f}s fetching {seen[0] if seen else current}")
        target = check_url_target(current)
        seen.append(target.url)
        conn = open_pinned_connection(target, timeout=min(remaining, 30.0))
        try:
            split = urlsplit(target.url)
            request_path = split.path or "/"
            if split.query:
                request_path = f"{request_path}?{split.query}"
            conn.request("GET", request_path, headers={
                "Host": target.host if target.port in (80, 443) else f"{target.host}:{target.port}",
                "User-Agent": "wangp-runpod-worker/1",
                "Accept": "*/*",
                "Accept-Encoding": "identity",
                "Connection": "close",
            })
            response = conn.getresponse()
            status = response.status
            if status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                response.read(0)
                if not location:
                    raise WorkerError(MEDIA_FETCH_FAILED,
                                      f"HTTP {status} with no Location header from {target.host}")
                if hop >= max_redirects:
                    raise WorkerError(MEDIA_FETCH_FAILED,
                                      f"too many redirects (>{max_redirects}) starting at {seen[0]}")
                current = urljoin(target.url, location)
                continue
            if status != 200:
                raise WorkerError(
                    MEDIA_FETCH_FAILED,
                    f"HTTP {status} {response.reason} fetching {target.host}",
                    details=[f"url: {target.url.split('?')[0]}"],
                )
            declared = response.getheader("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise WorkerError(
                    MEDIA_TOO_LARGE,
                    f"remote file is {int(declared)} B, over the {max_bytes} B cap",
                    details=[f"url: {target.url.split('?')[0]}"],
                )
            written = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as handle:
                while True:
                    if time.monotonic() > deadline:
                        raise WorkerError(
                            MEDIA_FETCH_FAILED,
                            f"timed out after {timeout_s:.0f}s while downloading "
                            f"{target.url.split('?')[0]} ({written} B read)",
                        )
                    chunk = response.read(_STREAM_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise WorkerError(
                            MEDIA_TOO_LARGE,
                            f"remote file exceeds the {max_bytes} B cap "
                            f"(stopped after {written} B)",
                            details=[f"url: {target.url.split('?')[0]}",
                                     "raise WANGP_B64_IN_MAX or send a smaller file"],
                        )
                    if head_out is not None and len(head_out) < _HEAD_BYTES:
                        head_out.extend(chunk[: _HEAD_BYTES - len(head_out)])
                    handle.write(chunk)
            return written
        except WorkerError:
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise WorkerError(
                MEDIA_FETCH_FAILED,
                f"could not fetch {target.host}: {type(exc).__name__}: {exc}",
                cause=exc,
            ) from exc
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - close must never mask the real error
                pass

    raise WorkerError(MEDIA_FETCH_FAILED,  # pragma: no cover - loop always returns/raises
                      f"too many redirects (>{max_redirects}) starting at {seen[0]}")


# ===========================================================================
# Job scratch space and volume paths
# ===========================================================================

def _job_root() -> Path:
    """Re-read the env every call so tests can point it at a tmp_path."""
    return Path(os.environ.get("WANGP_JOB_ROOT") or C.JOB_ROOT)


def _volume_root() -> Path:
    return Path(os.environ.get("WANGP_VOLUME_ROOT") or C.VOLUME_ROOT)


def volume_item_max() -> int:
    """Per-file ceiling for ``volume`` inputs (they are not inline bytes)."""
    try:
        return int(os.environ.get("WANGP_VOLUME_IN_MAX", str(2 * 1024 ** 3)))
    except (TypeError, ValueError):
        return 2 * 1024 ** 3


def _hash_max() -> int:
    try:
        return int(os.environ.get("WANGP_HASH_MAX", str(64 * 1024 * 1024)))
    except (TypeError, ValueError):
        return 64 * 1024 * 1024


def _safe_job_id(job_id: str) -> str:
    """A filesystem-safe directory name. RunPod ids look like ``60902e6c-…-u1``."""
    text = str(job_id or "job").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", text).lstrip(".")
    return (cleaned or "job")[:128]


def job_dir_for(job_id: str) -> Path:
    """``<WANGP_JOB_ROOT>/<job_id>`` — the per-job scratch dir. Not created."""
    return _job_root() / _safe_job_id(job_id)


def resolve_volume_path(relative: str) -> Path:
    """Resolve a caller-supplied ``volume`` path, refusing anything that escapes.

    The check is done on the **realpath**, so ``a/../../etc/passwd``, a symlink
    planted inside the volume that points at ``/etc``, and an absolute path are
    all rejected — the first two only show up after symlink resolution, which is
    why a lexical ``..`` scan is not enough.
    """
    raw = _VOLUME_PREFIX_RE.sub("", str(relative or "").strip())
    if not raw:
        raise WorkerError(BAD_REQUEST, "media volume path is empty")
    if raw.startswith(("/", "\\")) or os.path.isabs(raw) or re.match(r"^[A-Za-z]:", raw):
        raise WorkerError(
            BAD_REQUEST,
            f"media volume path must be relative to the network volume: {raw!r}",
            details=[f"volume root: {_volume_root()}"],
        )
    if "\x00" in raw:
        raise WorkerError(BAD_REQUEST, "media volume path contains a NUL byte")
    if "|" in raw:
        # WanGP splits on "|" to parse the virtual-media suffix
        # (shared/utils/virtual_media.py:36), so a pipe in a filename silently
        # truncates the path. Refuse it rather than hand over a mangled one.
        raise WorkerError(
            BAD_REQUEST,
            "media volume paths may not contain '|': the virtual-media syntax "
            "reserves it (docs/API.md:456)",
            details=['use the "range" object for start_frame/end_frame instead'],
        )

    root = Path(os.path.realpath(_volume_root()))
    candidate = Path(os.path.realpath(root / raw))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkerError(
            BAD_REQUEST,
            f"media volume path escapes the volume root: {raw!r}",
            details=[f"resolved to {candidate}", f"volume root: {root}"],
            cause=exc,
        ) from exc
    if not candidate.is_file():
        raise WorkerError(
            MEDIA_FETCH_FAILED,
            f"no such file on the network volume: {raw!r}",
            details=[f"looked in {root}"],
        )
    return candidate


# ===========================================================================
# Materialization
# ===========================================================================

#: Extensions that mean the same decoder as the sniffed one, so a volume file
#: that already carries one can be referenced in place instead of copied.
_EXT_ALIASES: dict[str, frozenset[str]] = {
    ".jpg": frozenset({".jpg", ".jpeg", ".jfif", ".pjpeg"}),
    ".tif": frozenset({".tif", ".tiff"}),
}


def _aliases(ext: str) -> frozenset[str]:
    return _EXT_ALIASES.get(ext, frozenset({ext}))


@dataclass
class MediaItem:
    """One materialized attachment."""

    key: str
    index: int | None
    kind: str
    source: str                 # "b64" | "volume" | "url"
    path: Path
    value: str                  # what goes into settings (path + virtual suffix)
    size_bytes: int
    content_type: str
    format: str
    inline: bool
    sha256: str | None = None
    copied: bool = True

    def to_dict(self) -> dict[str, Any]:
        body = {
            "key": self.key,
            "kind": self.kind,
            "source": self.source,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "format": self.format,
        }
        if self.index is not None:
            body["index"] = self.index
        if self.sha256:
            body["sha256"] = self.sha256
        return body


@dataclass
class MaterializedMedia:
    """The result of ``materialize``: what to merge, and what it cost."""

    settings: dict[str, Any] = field(default_factory=dict)
    items: list[MediaItem] = field(default_factory=list)
    job_dir: Path = field(default_factory=lambda: Path("."))
    input_dir: Path = field(default_factory=lambda: Path("."))
    inline_bytes: int = 0
    volume_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def paths(self) -> dict[str, Any]:
        """Alias for ``settings`` — the attachment-key -> path mapping."""
        return self.settings

    @property
    def total_bytes(self) -> int:
        return self.inline_bytes + self.volume_bytes

    def summary(self) -> dict[str, Any]:
        return {
            "count": len(self.items),
            "inline_bytes": self.inline_bytes,
            "volume_bytes": self.volume_bytes,
            "items": [item.to_dict() for item in self.items],
        }


class _Budget:
    """Running total for inline bytes, checked before every write."""

    __slots__ = ("item_max", "total_max", "used")

    def __init__(self, item_max: int, total_max: int) -> None:
        self.item_max = int(item_max)
        self.total_max = int(total_max)
        self.used = 0

    def check_item(self, key: str, size: int, *, what: str) -> None:
        if size > self.item_max:
            raise WorkerError(
                MEDIA_TOO_LARGE,
                f"media.{key} is {size} B, over the {self.item_max} B per-item cap",
                details=[f"{what}", "raise WANGP_B64_IN_MAX, or stage the file on the "
                                    "network volume and use {\"volume\": \"...\"}"],
            )
        if self.used + size > self.total_max:
            raise WorkerError(
                MEDIA_TOO_LARGE,
                f"media totals {self.used + size} B, over the {self.total_max} B cap "
                f"for all inline attachments",
                details=[f"media.{key} pushed it over ({size} B)",
                         "raise WANGP_MEDIA_TOTAL_MAX, or stage files on the network "
                         "volume — RunPod's own /run request cap is 10 MB"],
            )

    def spend(self, size: int) -> None:
        self.used += size


def _decode_b64(key: str, payload: Any, budget: _Budget) -> bytes:
    if not isinstance(payload, str):
        raise WorkerError(BAD_REQUEST, f"media.{key}.b64 must be a string")
    text = payload.strip()
    if text.startswith("data:"):
        header, _, remainder = text.partition(",")
        if not remainder or "base64" not in header.lower():
            raise WorkerError(BAD_REQUEST,
                              f"media.{key}: only base64 data: URIs are supported")
        text = remainder
    text = re.sub(r"\s+", "", text)
    if not text:
        raise WorkerError(BAD_REQUEST, f"media.{key}.b64 is empty")
    # Bound the work before allocating: 4 encoded chars carry 3 decoded bytes.
    budget.check_item(key, (len(text) * 3) // 4, what="base64 payload (estimated)")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WorkerError(
            MEDIA_FETCH_FAILED, f"media.{key}.b64 is not valid base64: {exc}", cause=exc
        ) from exc
    if not data:
        raise WorkerError(BAD_REQUEST, f"media.{key}.b64 decoded to zero bytes")
    budget.check_item(key, len(data), what="base64 payload (decoded)")
    return data


def _build_value(path: Path, spec: Mapping[str, Any] | None, key: str, kind: str) -> str:
    """Absolute path, plus the virtual-media suffix when a ``range`` was given.

    Mirrors ``shared/utils/virtual_media.build_virtual_media_path``; built here
    rather than imported so this module stays WanGP-free. The suffix survives
    ``_absolutize_setting_path`` (shared/api.py:1036-1043).
    """
    absolute = str(path)
    if not spec:
        return absolute
    if kind not in RANGE_KINDS:
        raise WorkerError(
            BAD_REQUEST,
            f"media.{key}: 'range' is only supported for video attachments",
            details=[f"'{key}' is a {kind} attachment"],
        )
    if not isinstance(spec, Mapping):
        raise WorkerError(BAD_REQUEST, f"media.{key}.range must be an object")
    unknown = sorted(set(spec) - {"start_frame", "end_frame", "audio_track_no"})
    if unknown:
        raise WorkerError(
            BAD_REQUEST, f"media.{key}.range has unknown fields {unknown}",
            details=["accepted: start_frame, end_frame, audio_track_no "
                     "(docs/API.md:452-467)"],
        )

    def _int(name: str) -> int | None:
        if spec.get(name) is None:
            return None
        try:
            value = int(spec[name])
        except (TypeError, ValueError) as exc:
            raise WorkerError(
                BAD_REQUEST, f"media.{key}.range.{name} must be an integer", cause=exc
            ) from exc
        if value < 0:
            raise WorkerError(BAD_REQUEST, f"media.{key}.range.{name} must be >= 0")
        return value

    start = _int("start_frame")
    end = _int("end_frame")
    track = _int("audio_track_no")
    if start is not None and end is not None and end < start:
        raise WorkerError(
            BAD_REQUEST,
            f"media.{key}.range.end_frame ({end}) is before start_frame ({start})",
            details=["start_frame is zero-based and end_frame is inclusive "
                     "(docs/API.md:460-461)"],
        )
    parts: list[str] = []
    if start is not None:
        parts.append(f"start_frame={start}")
    if end is not None:
        parts.append(f"end_frame={end}")
    if track is not None:
        parts.append(f"audio_track_no={track}")
    if not parts:
        return absolute
    if "|" in absolute:  # pragma: no cover - our own filenames never contain it
        raise WorkerError(MEDIA_FETCH_FAILED,
                          f"media.{key}: materialized path contains '|', which the "
                          f"virtual-media syntax reserves")
    return absolute + "|" + ",".join(parts)


def _normalize_item(key: str, raw: Any) -> dict[str, Any]:
    """Accept the object form, plus the ``scheme://`` string shorthand."""
    if isinstance(raw, str):
        text = raw.strip()
        lowered = text.lower()
        if lowered.startswith("volume://"):
            return {"volume": text[len("volume://"):]}
        if lowered.startswith(("http://", "https://")):
            return {"url": text}
        if lowered.startswith("data:"):
            return {"b64": text}
        raise WorkerError(
            BAD_REQUEST,
            f"media.{key} string form must start with volume://, https:// or data:",
            details=['use {"b64": "..."} / {"volume": "rel/path"} / {"url": "https://..."}'],
        )
    if not isinstance(raw, Mapping):
        raise WorkerError(
            BAD_REQUEST,
            f"media.{key} must be an object with one of b64 / volume / url",
        )
    item = dict(raw)
    if "base64" in item and "b64" not in item:
        item["b64"] = item.pop("base64")
    accepted = {"b64", "volume", "url", "range"}
    unknown = sorted(set(item) - accepted)
    if unknown:
        raise WorkerError(
            BAD_REQUEST, f"media.{key} has unknown fields {unknown}",
            details=[f"accepted: {sorted(accepted)}"],
        )
    sources = [name for name in ("b64", "volume", "url") if item.get(name) not in (None, "")]
    if not sources:
        raise WorkerError(
            BAD_REQUEST, f"media.{key} must carry exactly one of b64 / volume / url"
        )
    if len(sources) > 1:
        raise WorkerError(
            BAD_REQUEST,
            f"media.{key} carries more than one source ({sources}); pick one",
        )
    return item


def _write_bytes(dest: Path, data: bytes) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    with open(dest, "wb") as handle:
        handle.write(data)
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_STREAM_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_one(
    key: str,
    index: int | None,
    raw: Any,
    *,
    kind: str,
    input_dir: Path,
    budget: _Budget,
    cfg,
    warnings: list[str],
) -> MediaItem:
    item = _normalize_item(key, raw)
    stem = key if index is None else f"{key}_{index}"

    # ---- base64 ----------------------------------------------------------
    if item.get("b64") not in (None, ""):
        data = _decode_b64(key, item["b64"], budget)
        result = _require_kind(key, kind, data[:_HEAD_BYTES], source="b64 payload")
        dest = input_dir / f"{stem}{result.ext}"
        digest = _write_bytes(dest, data)
        budget.spend(len(data))
        return MediaItem(key=key, index=index, kind=kind, source="b64", path=dest,
                         value=_build_value(dest, item.get("range"), key, kind),
                         size_bytes=len(data), content_type=result.content_type,
                         format=result.format, inline=True, sha256=digest)

    # ---- network volume --------------------------------------------------
    if item.get("volume") not in (None, ""):
        source = resolve_volume_path(str(item["volume"]))
        size = source.stat().st_size
        ceiling = volume_item_max()
        if size > ceiling:
            raise WorkerError(
                MEDIA_TOO_LARGE,
                f"media.{key} is {size} B on the volume, over the {ceiling} B cap",
                details=[f"file: {source}", "raise WANGP_VOLUME_IN_MAX"],
            )
        if size == 0:
            raise WorkerError(MEDIA_FETCH_FAILED, f"media.{key}: {source} is empty")
        with open(source, "rb") as handle:
            head = handle.read(_HEAD_BYTES)
        result = _require_kind(key, kind, head, source=f"volume:{item['volume']}")

        copied = True
        if source.suffix.lower() in _aliases(result.ext):
            # The file already advertises what it actually is, so WanGP's
            # extension check will agree with the bytes. Reference it in place
            # and skip a potentially multi-GB copy.
            dest = source
            copied = False
        else:
            dest = input_dir / f"{stem}{result.ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            warnings.append(
                f"media.{key}: '{source.name}' is {result.format}; re-exposed as "
                f"'{dest.name}' (WanGP dispatches on the extension)"
            )
            try:
                # A hardlink keeps Path.resolve() (shared/api.py:1039) pointing at
                # OUR filename. A symlink would resolve back to the volume file
                # and hand WanGP the caller's extension again.
                os.link(source, dest)
            except OSError:
                shutil.copyfile(source, dest)
        digest = _sha256_file(dest) if size <= _hash_max() else None
        return MediaItem(key=key, index=index, kind=kind, source="volume", path=dest,
                         value=_build_value(dest, item.get("range"), key, kind),
                         size_bytes=size, content_type=result.content_type,
                         format=result.format, inline=False, sha256=digest,
                         copied=copied)

    # ---- URL -------------------------------------------------------------
    url = str(item["url"])
    if not url_inputs_enabled(cfg):
        raise WorkerError(
            BAD_REQUEST,
            f"media.{key}: URL inputs are disabled on this endpoint",
            details=["set ALLOW_URL_INPUTS=1 to enable them, or send "
                     '{"b64": "..."} / {"volume": "rel/path"}'],
        )
    remaining_total = max(0, budget.total_max - budget.used)
    cap = min(budget.item_max, remaining_total) if remaining_total else 0
    if cap <= 0:
        raise WorkerError(
            MEDIA_TOO_LARGE,
            f"media.{key}: the {budget.total_max} B inline budget is already spent",
        )
    input_dir.mkdir(parents=True, exist_ok=True)
    staging = input_dir / f"{stem}.download"
    head = bytearray()
    try:
        written = fetch_url(url, staging, max_bytes=cap, timeout_s=_url_timeout(),
                            max_redirects=_max_redirects(), head_out=head)
        if written == 0:
            raise WorkerError(MEDIA_FETCH_FAILED, f"media.{key}: {url} returned 0 bytes")
        result = _require_kind(key, kind, bytes(head), source=url.split("?")[0])
        dest = input_dir / f"{stem}{result.ext}"
        os.replace(staging, dest)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)
    budget.check_item(key, written, what=f"downloaded from {url.split('?')[0]}")
    budget.spend(written)
    return MediaItem(key=key, index=index, kind=kind, source="url", path=dest,
                     value=_build_value(dest, item.get("range"), key, kind),
                     size_bytes=written, content_type=result.content_type,
                     format=result.format, inline=True,
                     sha256=_sha256_file(dest) if written <= _hash_max() else None)


def _url_timeout() -> float:
    try:
        return float(os.environ.get("WANGP_URL_TIMEOUT_S", "60"))
    except (TypeError, ValueError):
        return 60.0


def _max_redirects() -> int:
    try:
        return max(0, int(os.environ.get("WANGP_URL_MAX_REDIRECTS", "3")))
    except (TypeError, ValueError):
        return 3


def materialize(media: Mapping[str, Any] | None, *, job_id: str, cfg=None) -> MaterializedMedia:
    """Turn ``input.media`` into absolute paths under ``<JOB_ROOT>/<job_id>/in``.

    Returns a :class:`MaterializedMedia` whose ``.settings`` is ready to merge
    into the WanGP settings dict (list-valued for ``image_refs``). Raises
    :class:`WorkerError` with a ``media_*`` / ``bad_request`` / ``ssrf_blocked``
    code for anything it will not accept.

    On failure the partially written job dir is left in place; the handler's
    ``finally`` calls :func:`cleanup` regardless of outcome.
    """
    cfg = cfg or C.CONFIG
    root = job_dir_for(job_id)
    input_dir = root / "in"
    result = MaterializedMedia(job_dir=root, input_dir=input_dir)
    if not media:
        return result
    if not isinstance(media, Mapping):
        raise WorkerError(BAD_REQUEST, "input.media must be an object")

    budget = _Budget(cfg.b64_in_max, cfg.media_total_max)
    started = time.monotonic()
    input_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(media):
        kind = MEDIA_KIND.get(key)
        if kind is None:
            raise WorkerError(
                BAD_REQUEST, f"'{key}' is not a WanGP attachment key",
                details=[f"valid: {sorted(MEDIA_KIND)}"],
            )
        value = media[key]
        if value is None:
            continue
        if key in LIST_KEYS:
            entries: Sequence[Any] = value if isinstance(value, (list, tuple)) else [value]
            if not entries:
                continue
            paths: list[str] = []
            for index, entry in enumerate(entries):
                item = _materialize_one(key, index, entry, kind=kind, input_dir=input_dir,
                                        budget=budget, cfg=cfg, warnings=result.warnings)
                result.items.append(item)
                paths.append(item.value)
            result.settings[key] = paths
        else:
            if isinstance(value, (list, tuple)):
                raise WorkerError(
                    BAD_REQUEST,
                    f"media.{key} takes a single attachment, not a list",
                    details=[f"list-valued keys: {sorted(LIST_KEYS)}"],
                )
            item = _materialize_one(key, None, value, kind=kind, input_dir=input_dir,
                                    budget=budget, cfg=cfg, warnings=result.warnings)
            result.items.append(item)
            result.settings[key] = item.value

    for item in result.items:
        if item.inline:
            result.inline_bytes += item.size_bytes
        else:
            result.volume_bytes += item.size_bytes

    LOG.info("media_materialized", job_dir=str(root), count=len(result.items),
             inline_bytes=result.inline_bytes, volume_bytes=result.volume_bytes,
             duration_ms=int((time.monotonic() - started) * 1000),
             keys=sorted(result.settings))
    return result


def cleanup(job_id_or_dir: str | os.PathLike[str]) -> bool:
    """Remove a job's scratch dir. Returns True when something was deleted.

    Refuses to delete anything outside ``WANGP_JOB_ROOT``: this runs in a
    ``finally`` block with a value that ultimately came from a request, and an
    ``rmtree`` there is not a place for optimism. Volume files referenced in
    place (never copied) live outside the job dir and are untouched.
    """
    root = Path(os.path.realpath(_job_root()))
    candidate = Path(job_id_or_dir)
    target = candidate if candidate.is_absolute() or os.sep in str(candidate) \
        else job_dir_for(str(job_id_or_dir))
    target = Path(os.path.realpath(target))
    if target == root or root not in target.parents:
        LOG.warn("media_cleanup_refused", target=str(target), job_root=str(root),
                 note="path is not inside WANGP_JOB_ROOT")
        return False
    if not target.exists():
        return False
    shutil.rmtree(target, ignore_errors=True)
    LOG.debug("media_cleanup", target=str(target))
    return not target.exists()


def sweep(max_age_s: float = 3600.0) -> int:
    """Delete job dirs older than ``max_age_s``. For boot, after a hard restart."""
    root = _job_root()
    if not root.is_dir():
        return 0
    removed = 0
    now = time.time()
    for child in root.iterdir():
        try:
            if not child.is_dir() or now - child.stat().st_mtime < max_age_s:
                continue
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        except OSError:  # pragma: no cover - racing with our own cleanup
            continue
    if removed:
        LOG.info("media_sweep", removed=removed, job_root=str(root))
    return removed
