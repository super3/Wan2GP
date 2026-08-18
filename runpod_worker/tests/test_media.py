"""CPU-only tests for media_in / media_out.

No torch, no wgp, no CUDA, no GPU, no weights, no network — the modules under
test import nothing heavy, which is the entire point of the split. ``boto3`` and
``runpod`` are faked where a transport needs them.

    pytest runpod_worker/tests/test_media.py -v
"""

from __future__ import annotations

import base64
import json
import os
import socketserver
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from runpod_worker import config as C
from runpod_worker import media_in, media_out
from runpod_worker.errors import WorkerError

# --------------------------------------------------------------------------
# Fixtures and byte fixtures
# --------------------------------------------------------------------------

PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WAV = b"RIFF" + (100).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 64
MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 64
AAC = b"\xff\xf1\x50\x80" + b"\x00" * 64
MP4 = (32).to_bytes(4, "big") + b"ftypisom" + b"\x00" * 64
MOV = (20).to_bytes(4, "big") + b"ftypqt  " + b"\x00" * 64
MKV = b"\x1aE\xdf\xa3" + b"\x00" * 64
AVI = b"RIFF" + (100).to_bytes(4, "little") + b"AVI LIST" + b"\x00" * 64
OGG = b"OggS" + b"\x00" * 64


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A private job root, volume root and worker config for one test."""
    jobs = tmp_path / "jobs"
    volume = tmp_path / "volume"
    jobs.mkdir()
    volume.mkdir()
    monkeypatch.setenv("WANGP_JOB_ROOT", str(jobs))
    monkeypatch.setenv("WANGP_VOLUME_ROOT", str(volume))
    monkeypatch.delenv("ALLOW_URL_INPUTS", raising=False)
    monkeypatch.delenv("ALLOW_URL_PRIVATE_HOSTS", raising=False)
    for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID",
                "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME",
                "WANGP_S3_DIRECT", "WANGP_S3_PUBLIC_BASE_URL", "WANGP_OUTPUT_CHAIN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANGP_FFPROBE", str(tmp_path / "no-such-ffprobe"))
    cfg = C.WorkerConfig()
    return types.SimpleNamespace(jobs=jobs, volume=volume, cfg=cfg, tmp=tmp_path)


# --------------------------------------------------------------------------
# Sniffing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("data", "kind", "ext"),
    [
        (PNG, "image", ".png"), (JPEG, "image", ".jpg"), (GIF, "image", ".gif"),
        (b"BM\x00\x00\x00\x00", "image", ".bmp"), (b"II*\x00rest", "image", ".tif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image", ".webp"),
        (WAV, "audio", ".wav"), (MP3, "audio", ".mp3"), (AAC, "audio", ".aac"),
        (b"ID3\x04\x00\x00", "audio", ".mp3"),
        (MP4, "video", ".mp4"), (MOV, "video", ".mov"),
        (MKV, "video", ".mkv"), (AVI, "video", ".avi"),
    ],
)
def test_sniff_table(data, kind, ext):
    result = media_in.sniff(data)
    assert result is not None
    assert (result.kind, result.ext) == (kind, ext)


@pytest.mark.parametrize(
    ("header", "expected_ext"),
    [
        # The cases a naive two-byte table gets wrong: layer bits, not the byte.
        (b"\xff\xfa", ".mp3"),   # MPEG1 Layer III
        (b"\xff\xf3", ".mp3"),   # MPEG2 Layer III
        (b"\xff\xfb", ".mp3"),   # MPEG1 Layer III, no CRC
        (b"\xff\xf9", ".aac"),   # ADTS, MPEG-2 AAC
        (b"\xff\xf1", ".aac"),   # ADTS, MPEG-4 AAC
    ],
)
def test_mpeg_sync_word_variants(header, expected_ext):
    result = media_in.sniff(header + b"\x00" * 32)
    assert result is not None and result.kind == "audio"
    assert result.ext == expected_ext


def test_sniff_rejects_unknown_and_unsupported():
    assert media_in.sniff(b"not media at all") is None
    assert media_in.sniff(OGG) is None
    assert media_in.sniff(b"") is None


def test_webm_is_named_mkv():
    """WanGP's whitelist has no .webm; a WebM file must not keep that name."""
    result = media_in.sniff(b"\x1aE\xdf\xa3" + b"\x42\x82\x84webm" + b"\x00" * 32)
    assert result is not None and result.ext == ".mkv"


# --------------------------------------------------------------------------
# media_in: base64
# --------------------------------------------------------------------------

def test_b64_materializes_with_sniffed_extension(sandbox):
    out = media_in.materialize({"image_start": {"b64": b64(PNG)}},
                               job_id="job-1", cfg=sandbox.cfg)
    path = Path(out.settings["image_start"])
    assert path.is_absolute() and path.suffix == ".png" and path.read_bytes() == PNG
    assert path.parent == sandbox.jobs / "job-1" / "in"
    assert out.inline_bytes == len(PNG)
    assert out.items[0].sha256


def test_media_magic_bytes_beats_the_slot(sandbox):
    """A PNG offered as audio is rejected; the caller never picks the decoder."""
    with pytest.raises(WorkerError) as excinfo:
        media_in.materialize({"audio_guide": {"b64": b64(PNG)}},
                             job_id="job-2", cfg=sandbox.cfg)
    assert excinfo.value.code == "media_unsupported"
    assert "image" in str(excinfo.value)


def test_data_uri_and_bad_base64(sandbox):
    out = media_in.materialize({"image_start": {"b64": "data:image/png;base64," + b64(PNG)}},
                               job_id="job-3", cfg=sandbox.cfg)
    assert Path(out.settings["image_start"]).suffix == ".png"
    with pytest.raises(WorkerError) as excinfo:
        media_in.materialize({"image_start": {"b64": "!!!not base64!!!"}},
                             job_id="job-4", cfg=sandbox.cfg)
    assert excinfo.value.code == "media_fetch_failed"


def test_per_item_and_total_caps(sandbox, monkeypatch):
    monkeypatch.setenv("WANGP_B64_IN_MAX", "512")
    monkeypatch.setenv("WANGP_MEDIA_TOTAL_MAX", "700")
    cfg = C.WorkerConfig()
    big = PNG + b"\x00" * 1024
    with pytest.raises(WorkerError) as excinfo:
        media_in.materialize({"image_start": {"b64": b64(big)}}, job_id="cap-1", cfg=cfg)
    assert excinfo.value.code == "media_too_large"

    medium = PNG + b"\x00" * 300              # 372 B, under the per-item cap
    with pytest.raises(WorkerError) as excinfo:
        media_in.materialize(
            {"image_refs": [{"b64": b64(medium)}, {"b64": b64(medium)}]},
            job_id="cap-2", cfg=cfg,
        )
    assert excinfo.value.code == "media_too_large"
    assert "all inline attachments" in str(excinfo.value)


def test_image_refs_is_a_list_and_others_are_not(sandbox):
    out = media_in.materialize({"image_refs": [{"b64": b64(PNG)}, {"b64": b64(JPEG)}]},
                               job_id="list-1", cfg=sandbox.cfg)
    values = out.settings["image_refs"]
    assert isinstance(values, list) and len(values) == 2
    assert Path(values[0]).name == "image_refs_0.png"
    assert Path(values[1]).name == "image_refs_1.jpg"
    with pytest.raises(WorkerError):
        media_in.materialize({"image_start": [{"b64": b64(PNG)}]},
                             job_id="list-2", cfg=sandbox.cfg)


def test_unknown_attachment_key_and_shapes(sandbox):
    for payload in ({"not_a_key": {"b64": b64(PNG)}},
                    {"image_start": {"b64": b64(PNG), "volume": "x.png"}},
                    {"image_start": {}},
                    {"image_start": {"nope": 1}},
                    {"image_start": "just a string"}):
        with pytest.raises(WorkerError) as excinfo:
            media_in.materialize(payload, job_id="shape", cfg=sandbox.cfg)
        assert excinfo.value.code == "bad_request"


# --------------------------------------------------------------------------
# media_in: the network volume, and the traversal guard
# --------------------------------------------------------------------------

def test_volume_input_referenced_in_place_when_extension_agrees(sandbox):
    clip = sandbox.volume / "clips" / "plate.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(MP4)
    out = media_in.materialize({"video_guide": {"volume": "clips/plate.mp4"}},
                               job_id="vol-1", cfg=sandbox.cfg)
    assert out.settings["video_guide"] == str(clip)
    assert out.volume_bytes == len(MP4) and out.inline_bytes == 0


def test_volume_input_relabelled_when_extension_lies(sandbox):
    """A WAV named .mp4 must not reach WanGP under that name."""
    liar = sandbox.volume / "liar.mp4"
    liar.write_bytes(WAV)
    out = media_in.materialize({"audio_guide": {"volume": "liar.mp4"}},
                               job_id="vol-2", cfg=sandbox.cfg)
    path = Path(out.settings["audio_guide"])
    assert path.suffix == ".wav"
    assert path.parent == sandbox.jobs / "vol-2" / "in"
    assert path.read_bytes() == WAV
    assert liar.exists(), "the volume file itself must never be moved or renamed"


@pytest.mark.parametrize(
    "attempt",
    ["../etc/passwd", "/etc/passwd", "clips/../../../etc/passwd",
     "volume://../outside.mp4", "./../../etc/hostname", ""],
)
def test_volume_path_traversal_is_rejected(sandbox, attempt):
    with pytest.raises(WorkerError) as excinfo:
        media_in.resolve_volume_path(attempt)
    assert excinfo.value.code in ("bad_request", "media_fetch_failed")


def test_volume_traversal_is_rejected_through_materialize(sandbox, tmp_path):
    """The end-to-end version of the guard above, on the path a caller reaches.

    ``resolve_volume_path`` is the unit; this is the request. A real
    ``/etc/passwd`` exists on every runner, so a lexical-only check that resolved
    before rejecting would happily hand WanGP the file — and WanGP's own
    ``has_*_file_extension`` gate would not stop a ``.png`` named copy of it.
    """
    for attempt in ("../../etc/passwd", "../../../../etc/passwd",
                    "clips/../../etc/passwd"):
        with pytest.raises(WorkerError) as excinfo:
            media_in.materialize({"image_start": {"volume": attempt}},
                                 job_id="trav-1", cfg=sandbox.cfg)
        assert excinfo.value.code in ("bad_request", "media_fetch_failed")

    # And no existence oracle: a target that really is there outside the root
    # must fail exactly like one that is not.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.png").write_bytes(PNG)
    errors = []
    for name in ("real.png", "absent.png"):
        with pytest.raises(WorkerError) as excinfo:
            media_in.materialize(
                {"image_start": {"volume": f"../{outside.name}/{name}"}},
                job_id="trav-2", cfg=sandbox.cfg,
            )
        errors.append((excinfo.value.code, excinfo.value.message))
    assert errors[0][0] == errors[1][0]

    assert not (media_in.job_dir_for("trav-1") / "in").exists() or not list(
        (media_in.job_dir_for("trav-1") / "in").iterdir()
    ), "a rejected input must not leave a materialized file behind"


def test_volume_symlink_escape_is_rejected(sandbox, tmp_path):
    """Lexical `..` checks miss this one; the realpath check does not."""
    secret = tmp_path / "outside"
    secret.mkdir()
    (secret / "passwd").write_bytes(b"root:x:0:0")
    (sandbox.volume / "escape").symlink_to(secret)
    with pytest.raises(WorkerError) as excinfo:
        media_in.resolve_volume_path("escape/passwd")
    assert excinfo.value.code == "bad_request"
    assert "escapes the volume root" in str(excinfo.value)


def test_volume_path_may_not_contain_a_pipe(sandbox):
    """WanGP splits on '|' to parse the virtual-media suffix."""
    with pytest.raises(WorkerError) as excinfo:
        media_in.resolve_volume_path("clips/we|rd.mp4")
    assert excinfo.value.code == "bad_request"


def test_volume_missing_file(sandbox):
    with pytest.raises(WorkerError) as excinfo:
        media_in.resolve_volume_path("nope/missing.mp4")
    assert excinfo.value.code == "media_fetch_failed"


# --------------------------------------------------------------------------
# media_in: the virtual-media suffix (docs/API.md:452-477)
# --------------------------------------------------------------------------

def test_range_builds_the_virtual_media_suffix(sandbox):
    clip = sandbox.volume / "plate.mp4"
    clip.write_bytes(MP4)
    out = media_in.materialize(
        {"video_guide": {"volume": "plate.mp4",
                         "range": {"start_frame": 57542, "end_frame": 57782,
                                   "audio_track_no": 2}}},
        job_id="range-1", cfg=sandbox.cfg,
    )
    value = out.settings["video_guide"]
    path, _, suffix = value.partition("|")
    assert Path(path).is_file()
    assert suffix == "start_frame=57542,end_frame=57782,audio_track_no=2"


def test_range_rejected_on_non_video_and_when_inverted(sandbox):
    with pytest.raises(WorkerError):
        media_in.materialize({"image_start": {"b64": b64(PNG), "range": {"end_frame": 3}}},
                             job_id="range-2", cfg=sandbox.cfg)
    clip = sandbox.volume / "plate.mp4"
    clip.write_bytes(MP4)
    with pytest.raises(WorkerError) as excinfo:
        media_in.materialize(
            {"video_guide": {"volume": "plate.mp4",
                             "range": {"start_frame": 90, "end_frame": 10}}},
            job_id="range-3", cfg=sandbox.cfg,
        )
    assert excinfo.value.code == "bad_request"


# --------------------------------------------------------------------------
# media_in: URL inputs and the SSRF guard
# --------------------------------------------------------------------------

def test_url_inputs_are_off_by_default(sandbox):
    with pytest.raises(WorkerError) as excinfo:
        media_in.materialize({"image_start": {"url": "https://example.com/a.png"}},
                             job_id="url-0", cfg=sandbox.cfg)
    assert excinfo.value.code == "bad_request"
    assert any("ALLOW_URL_INPUTS=1" in item for item in excinfo.value.details)


@pytest.mark.parametrize(
    "url",
    ["http://example.com/a.png",                 # scheme not allowed by default
     "file:///etc/passwd",
     "https://127.0.0.1/a.png",
     "https://[::1]/a.png",
     "https://10.1.2.3/a.png",
     "https://192.168.0.5/a.png",
     "https://169.254.169.254/latest/meta-data/",  # cloud metadata
     "https://100.100.100.200/latest/meta-data/",  # Alibaba metadata, CGNAT
     "https://[::ffff:127.0.0.1]/a.png",           # v4-mapped loopback
     "https://user:pass@example.com/a.png",
     "https://example.com:22/a.png"],
)
def test_ssrf_guard_blocks(url):
    with pytest.raises(WorkerError) as excinfo:
        media_in.check_url_target(url)
    assert excinfo.value.code in ("ssrf_blocked", "media_fetch_failed")


def _serve(handler_cls):
    """A loopback-only server. No outbound network: nothing leaves the host.

    A sandbox that forbids even a 127.0.0.1 bind skips the test rather than
    failing it — the URL path is opt-in (ALLOW_URL_INPUTS=0 by default) and the
    rest of the suite must stay runnable on a locked-down runner.
    """
    try:
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler_cls)
    except OSError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot bind a loopback socket in this sandbox: {exc}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_url_fetch_streams_redirects_and_caps(sandbox, monkeypatch):
    body = PNG + b"\x00" * 4096

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # noqa: D102 - silence the test server
            pass

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's contract
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/image.png")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = body if self.path != "/huge.png" else PNG + b"\x00" * 500_000
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = _serve(Handler)
    port = server.server_address[1]
    try:
        monkeypatch.setenv("ALLOW_URL_INPUTS", "1")
        monkeypatch.setenv("ALLOW_URL_PRIVATE_HOSTS", "1")
        monkeypatch.setenv("WANGP_URL_SCHEMES", "http,https")
        monkeypatch.setenv("WANGP_URL_PORTS", "")
        cfg = C.WorkerConfig()
        base = f"http://127.0.0.1:{port}"

        out = media_in.materialize({"image_start": {"url": f"{base}/redirect"}},
                                   job_id="url-1", cfg=cfg)
        assert Path(out.settings["image_start"]).read_bytes() == body

        monkeypatch.setenv("WANGP_B64_IN_MAX", "1024")
        capped = C.WorkerConfig()
        with pytest.raises(WorkerError) as excinfo:
            media_in.materialize({"image_start": {"url": f"{base}/huge.png"}},
                                 job_id="url-2", cfg=capped)
        assert excinfo.value.code == "media_too_large"
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------
# media_in: cleanup
# --------------------------------------------------------------------------

def test_cleanup_removes_the_job_dir_only(sandbox):
    clip = sandbox.volume / "plate.mp4"
    clip.write_bytes(MP4)
    media_in.materialize({"image_start": {"b64": b64(PNG)},
                          "video_guide": {"volume": "plate.mp4"}},
                         job_id="clean-1", cfg=sandbox.cfg)
    job_dir = media_in.job_dir_for("clean-1")
    assert job_dir.is_dir()
    assert media_in.cleanup("clean-1") is True
    assert not job_dir.exists()
    assert clip.exists(), "a volume file referenced in place must survive cleanup"
    assert media_in.cleanup("clean-1") is False


def test_cleanup_refuses_paths_outside_the_job_root(sandbox, tmp_path):
    victim = tmp_path / "not-a-job"
    victim.mkdir()
    (victim / "keep").write_text("data")
    assert media_in.cleanup(victim) is False
    assert (victim / "keep").exists()
    assert media_in.cleanup(sandbox.jobs) is False
    assert sandbox.jobs.exists()


# --------------------------------------------------------------------------
# media_out
# --------------------------------------------------------------------------

@pytest.fixture()
def video(sandbox):
    path = sandbox.tmp / "2026-08-18-14h22m01s_seed918273645.mp4"
    path.write_bytes(MP4 + b"\x00" * 4096)
    return path


def test_base64_boundary(sandbox, video, monkeypatch):
    size = video.stat().st_size
    monkeypatch.setenv("WANGP_B64_OUT_MAX", str(size))
    result = media_out.deliver(video, job_id="out-1", cfg=C.WorkerConfig())
    assert result["transport"] == "base64" and result["kind"] == "base64"
    assert base64.b64decode(result["data"]) == video.read_bytes()
    assert result["bytes"] == result["size_bytes"] == size
    assert result["content_type"] == "video/mp4"

    monkeypatch.setenv("WANGP_B64_OUT_MAX", str(size - 1))
    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(video, job_id="out-2", cfg=C.WorkerConfig())
    assert excinfo.value.code == "output_too_large"
    assert excinfo.value.retryable is False
    assert any("presigned" in item for item in excinfo.value.details)


def test_never_returns_a_container_local_path(sandbox, video):
    result = media_out.deliver(video, job_id="out-3", cfg=sandbox.cfg)
    blob = json.dumps(result)
    assert str(video) not in blob
    assert str(video.parent) not in blob
    assert result["filename"] == video.name


def _install_fake_rp_upload(monkeypatch, returns):
    calls: list[dict] = []

    def upload_file_to_bucket(file_name, file_location, prefix=None,
                              extra_args=None, bucket_name=None):
        calls.append({"file_name": file_name, "file_location": file_location,
                      "prefix": prefix, "extra_args": extra_args,
                      "bucket_name": bucket_name})
        return returns

    runpod = types.ModuleType("runpod")
    serverless = types.ModuleType("runpod.serverless")
    utils = types.ModuleType("runpod.serverless.utils")
    rp_upload = types.ModuleType("runpod.serverless.utils.rp_upload")
    rp_upload.upload_file_to_bucket = upload_file_to_bucket
    utils.rp_upload = rp_upload
    serverless.utils = utils
    runpod.serverless = serverless
    for name, module in (("runpod", runpod), ("runpod.serverless", serverless),
                         ("runpod.serverless.utils", utils),
                         ("runpod.serverless.utils.rp_upload", rp_upload)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_rp_upload_local_fallback_is_caught(sandbox, video, monkeypatch):
    """rp_upload.py:300-301 returns a local path instead of raising.

    The highest-value test in the suite: without the ``startswith("http")``
    guard the worker reports success and hands the client a path that dies with
    the container.
    """
    for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID",
                "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME"):
        monkeypatch.setenv(key, "set")
    _install_fake_rp_upload(monkeypatch, "local_upload/out.mp4")
    cfg = C.WorkerConfig()

    auto = media_out.deliver(video, job_id="out-4", cfg=cfg)
    assert auto["transport"] == "base64", "auto must fall through, not report success"

    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(video, job_id="out-5", request_opts={"mode": "s3"}, cfg=cfg)
    assert excinfo.value.code == "upload_failed"
    assert "local disk" in str(excinfo.value)


def test_bucket_upload_success(sandbox, video, monkeypatch):
    for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID",
                "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME"):
        monkeypatch.setenv(key, "set")
    calls = _install_fake_rp_upload(monkeypatch, "https://bucket.example.com/x.mp4?sig=1")
    result = media_out.deliver(video, job_id="60902e6c-u1",
                               model_type="minimax_h3_fl2va_pruned",
                               cfg=C.WorkerConfig())
    assert result["transport"] == "rp_bucket" and result["kind"] == "url"
    assert result["url"].startswith("https://")
    assert result["expires_in_s"] == 604800
    assert calls[0]["prefix"] == "wangp/minimax_h3_fl2va_pruned"
    assert calls[0]["file_name"] == "60902e6c-u1.mp4"


def test_explicit_mode_failures_are_specific(sandbox, video):
    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(video, job_id="out-6", request_opts={"mode": "presigned"},
                          cfg=sandbox.cfg)
    assert excinfo.value.code == "bad_request"
    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(video, job_id="out-7", request_opts={"mode": "s3"},
                          cfg=sandbox.cfg)
    assert excinfo.value.code == "upload_failed"
    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(video, job_id="out-8", request_opts={"mode": "carrier pigeon"},
                          cfg=sandbox.cfg)
    assert excinfo.value.code == "bad_request"


def test_missing_or_empty_output(sandbox):
    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(sandbox.tmp / "gone.mp4", job_id="out-9", cfg=sandbox.cfg)
    assert excinfo.value.code == "no_output"
    empty = sandbox.tmp / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(WorkerError) as excinfo:
        media_out.deliver(empty, job_id="out-10", cfg=sandbox.cfg)
    assert excinfo.value.code == "no_output"


def test_volume_transport_is_opt_in(sandbox, video):
    assert "volume" not in media_out.default_chain()
    result = media_out.deliver(video, job_id="out-11",
                               request_opts={"mode": "volume"}, cfg=sandbox.cfg)
    assert result["transport"] == "volume"
    assert result["volume_path"] == f"outputs/{result['key']}"
    assert (sandbox.volume / "outputs" / result["key"]).read_bytes() == video.read_bytes()


def test_object_key_is_derived_from_the_job_id(sandbox):
    key = media_out.object_key("60902e6c-…-u1", "whatever.mp4",
                               model_type="minimax_h3_fl2va_pruned")
    assert key.startswith("wangp/minimax_h3_fl2va_pruned/")
    assert key.endswith(".mp4")
    assert media_out.object_key("a/../b", "x.mp4", model_type="m", prefix="p") == \
        "p/m/a_.._b.mp4"


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------

FFPROBE_JSON = {
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 832, "height": 480,
         "avg_frame_rate": "24/1", "nb_frames": "124", "pix_fmt": "yuv420p"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "32000",
         "channels": 2},
    ],
    "format": {"format_name": "mov,mp4,m4a", "duration": "5.166667",
               "bit_rate": "1300000"},
}


def test_ffprobe_missing_is_not_fatal(sandbox, video):
    meta = media_out.ffprobe(video)
    assert "probe_error" in meta
    result = media_out.deliver(video, job_id="probe-1", cfg=sandbox.cfg)
    assert result["transport"] == "base64" and "probe_error" in result


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stub")
def test_ffprobe_parses_streams(sandbox, video, monkeypatch):
    stub = sandbox.tmp / "ffprobe-stub.sh"
    stub.write_text("#!/bin/sh\ncat <<'JSON'\n" + json.dumps(FFPROBE_JSON) + "\nJSON\n")
    stub.chmod(0o755)
    monkeypatch.setenv("WANGP_FFPROBE", str(stub))
    meta = media_out.ffprobe(video)
    assert meta["width"] == 832 and meta["height"] == 480
    assert meta["fps"] == 24 and meta["duration_s"] == 5.167
    assert meta["video_codec"] == "h264" and meta["audio_codec"] == "aac"
    assert meta["has_audio"] is True and meta["audio_sample_rate"] == 32000
    assert meta["audio_channels"] == 2 and meta["container"] == "mp4"


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stub")
def test_ffprobe_failure_is_reported_not_raised(sandbox, video, monkeypatch):
    stub = sandbox.tmp / "ffprobe-fail.sh"
    stub.write_text("#!/bin/sh\necho 'moov atom not found' >&2\nexit 1\n")
    stub.chmod(0o755)
    monkeypatch.setenv("WANGP_FFPROBE", str(stub))
    meta = media_out.ffprobe(video)
    assert "probe_error" in meta and "moov atom" in meta["probe_error"]


# --------------------------------------------------------------------------
# The bucket transport without boto3 installed
# --------------------------------------------------------------------------

def test_boto3_absence_is_a_typed_error_not_an_import_crash(sandbox, monkeypatch):
    """boto3 ships with runpod, but the CPU tier must not require it.

    Skipped when boto3 *is* installed; either way ``media_out`` itself imports
    without it (asserted by ``test_modules_stay_cpu_only``).
    """
    try:
        import boto3  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("boto3 is installed; the missing-dependency path cannot be exercised")
    with pytest.raises(WorkerError) as excinfo:
        media_out._boto3_client()
    assert excinfo.value.code == "upload_failed"
    assert "boto3" in excinfo.value.message


class _FakeS3Client:
    """Just enough S3 to drive the direct uploader. No boto3, no network."""

    def __init__(self):
        self.uploads: list[dict] = []
        self.heads: list[dict] = []
        self.head_result: dict | None = None

    def upload_file(self, filename, bucket, key, ExtraArgs=None):  # noqa: N803 - boto3's name
        self.uploads.append({"filename": filename, "bucket": bucket, "key": key,
                             "extra": ExtraArgs})

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):  # noqa: N803
        return f"https://bucket.example.com/{Params['Key']}?X-Amz-Expires={ExpiresIn}"

    def head_object(self, Bucket=None, Key=None):  # noqa: N803
        self.heads.append({"bucket": Bucket, "key": Key})
        if self.head_result is None:
            raise RuntimeError("404 Not Found")
        return self.head_result


def test_direct_s3_upload_sets_the_content_type(sandbox, video, monkeypatch):
    """WANGP_S3_DIRECT=1 exists because rp_upload hardcodes ExpiresIn=604800
    (rp_upload.py:321) and does not set ContentType — a browser then downloads
    the mp4 as application/octet-stream instead of playing it."""
    client = _FakeS3Client()
    monkeypatch.setattr(media_out, "_boto3_client", lambda: client)
    for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID",
                "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME"):
        monkeypatch.setenv(key, "set")
    monkeypatch.setenv("WANGP_S3_DIRECT", "1")
    monkeypatch.setenv("WANGP_S3_EXPIRES_S", "3600")

    result = media_out.deliver(video, job_id="direct-1",
                               model_type="minimax_h3_fl2va_pruned", cfg=C.WorkerConfig())
    assert result["transport"] == "rp_bucket" and result["uploader"] == "boto3"
    assert result["url"].startswith("https://")
    assert client.uploads[0]["extra"] == {"ContentType": "video/mp4"}
    assert client.uploads[0]["key"] == result["key"]
    assert result["expires_in_s"] == 3600


def test_find_existing_is_the_idempotency_probe(sandbox, monkeypatch):
    """Failure mode 23: a retry re-derives the key and returns 0 GPU seconds in."""
    client = _FakeS3Client()
    monkeypatch.setattr(media_out, "_boto3_client", lambda: client)
    for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID",
                "BUCKET_SECRET_ACCESS_KEY", "BUCKET_NAME"):
        monkeypatch.setenv(key, "set")

    assert media_out.find_existing("wangp/x/job.mp4", cfg=C.WorkerConfig()) is None
    client.head_result = {"ContentLength": 4096, "ContentType": "video/mp4"}
    found = media_out.find_existing("wangp/x/job.mp4", cfg=C.WorkerConfig())
    assert found and found["url"].startswith("https://")
    assert found["size_bytes"] == 4096
    # A probe must never be able to fail a job.
    monkeypatch.setattr(media_out, "_boto3_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert media_out.find_existing("wangp/x/job.mp4", cfg=C.WorkerConfig()) is None


def test_find_existing_is_inert_without_a_bucket(sandbox):
    assert media_out.find_existing("wangp/x/job.mp4", cfg=sandbox.cfg) is None


# --------------------------------------------------------------------------
# Cross-module invariants
# --------------------------------------------------------------------------

def test_modules_stay_cpu_only():
    """The split only pays off if these imports never drag in the heavy stack."""
    for module in ("torch", "wgp", "gradio", "numpy", "boto3"):
        assert module not in sys.modules or module == "numpy", (
            f"{module} was imported by media_in/media_out"
        )


def test_attachment_tables_match_wgp_source():
    """Re-derive ATTACHMENT_KEYS from wgp.py without importing it."""
    import ast
    import re

    source = Path(__file__).resolve().parents[2] / "wgp.py"
    text = source.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^ATTACHMENT_KEYS\s*=\s*(\[[^\]]*\])", text, re.MULTILINE)
    assert match, "ATTACHMENT_KEYS literal not found in wgp.py"
    keys = tuple(ast.literal_eval(match.group(1)))
    assert keys == media_in.ATTACHMENT_KEYS
    assert set(media_in.MEDIA_KIND) <= set(keys)


def test_extension_whitelists_match_wgp_source():
    """The three whitelists live in shared/utils/utils.py:36-49."""
    import ast
    import re

    source = Path(__file__).resolve().parents[2] / "shared" / "utils" / "utils.py"
    text = source.read_text(encoding="utf-8", errors="replace")
    found = {}
    for name, kind in (("has_video_file_extension", "video"),
                       ("has_image_file_extension", "image"),
                       ("has_audio_file_extension", "audio")):
        match = re.search(name + r"\(filename\):.*?extension in (\[[^\]]*\])", text, re.S)
        assert match, f"{name} not found"
        found[kind] = set(ast.literal_eval(match.group(1)))
    assert found["video"] == set(media_in.VIDEO_EXTS)
    assert found["image"] == set(media_in.IMAGE_EXTS)
    assert found["audio"] == set(media_in.AUDIO_EXTS)
