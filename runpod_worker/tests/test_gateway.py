"""Tests for the customer-facing gateway.

The security-relevant behaviours are the point of this file. A gateway exists so
a customer never holds a RunPod key (which is account-wide), so the tests that
matter are: unauthenticated calls are refused, one customer cannot read
another's job, the product parameters cannot be overridden by the caller, and a
failed submit does not consume quota.

RunPod itself is stubbed -- these run on CPU with no network and no GPU.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

KEY_A, KEY_B = "sk_test_aaa", "sk_test_bbb"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_stub")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep_stub")
    monkeypatch.setenv("GATEWAY_KEYS", json.dumps({KEY_A: "acme", KEY_B: "globex"}))
    monkeypatch.setenv("GATEWAY_CACHE", str(tmp_path))
    monkeypatch.setenv("GATEWAY_DAILY_LIMIT", "3")
    import importlib
    from runpod_worker.gateway import app as module
    importlib.reload(module)
    return TestClient(module.app), module


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


import base64 as _b64

MP4 = b"\x00\x00\x00\x18ftypisom-fake-bytes"
COMPLETED = {"status": "COMPLETED", "output": {
    "seed": 7,
    "metrics": {"generate_s": 55.8},
    "video": {"kind": "base64", "data": _b64.b64encode(MP4).decode(),
              "duration_s": 10.125, "width": 832, "height": 480, "has_audio": True}}}


def _backend(status=COMPLETED, run_id="job-1"):
    """Stub RunPod: /run returns an id, status/<id> returns `status`."""
    def fake(path, payload=None, timeout=60):
        return {"id": run_id} if path == "run" else status
    return fake


# --- auth ------------------------------------------------------------------

def test_rejects_missing_key(client):
    c, _ = client
    assert c.post("/v1/videos", json={"prompt": "x"}).status_code == 401


def test_rejects_unknown_key(client):
    c, _ = client
    r = c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr("sk_not_real"))
    assert r.status_code == 401


def test_one_customer_cannot_read_anothers_job(client):
    """404, not 403: a customer must not be able to probe which ids exist."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        r = c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A))
    job = r.headers["x-video-id"]      # the body is the mp4 now
    assert c.get(f"/v1/videos/{job}", headers=_hdr(KEY_B)).status_code == 404
    assert c.get(f"/v1/videos/{job}/content", headers=_hdr(KEY_B)).status_code == 404


# --- the product is fixed server-side --------------------------------------

def test_caller_cannot_choose_length_or_resolution(client):
    """The parameters that cost money are not caller-controlled."""
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"])
            return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos",
                   json={"prompt": "x", "video_length": 481, "resolution": "1920x1080"},
                   headers=_hdr(KEY_A))
    assert r.status_code == 200
    # a raw video_length in the body is ignored; only duration_s (5|10) is honoured
    assert seen["video_length"] == 124        # the 5 s default, not the requested 481
    assert seen["resolution"] == "832x480"


def test_seed_is_passed_through_and_omitted_when_random(client):
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.clear(); seen.update(payload["input"]["settings"])
            return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x", "seed": 42}, headers=_hdr(KEY_A))
        assert seen["seed"] == 42
        c.post("/v1/videos", json={"prompt": "x", "seed": -1}, headers=_hdr(KEY_A))
        assert "seed" not in seen


# --- quota -----------------------------------------------------------------

def test_daily_limit_is_enforced(client):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 200
        assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 429
    # the other customer has their own budget
    with mock.patch.object(m, "_rp", side_effect=_backend(run_id="job-2")):
        assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_B)).status_code == 200


def test_failed_submit_does_not_consume_quota(client):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=RuntimeError("backend down")):
        assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 503
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 200


# --- lifecycle -------------------------------------------------------------

def test_backend_errors_are_not_leaked(client):
    """A RunPod traceback or key must never reach a customer."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=RuntimeError("rpa_secretkey leaked!")):
        r = c.get("/v1/health")
    assert r.status_code == 503
    assert "rpa_" not in r.text and "secretkey" not in r.text


def test_docs_page_is_served_same_origin(client):
    """Serving the demo from the API itself is what removes the CORS problem:
    the page and the endpoints share an origin, so no grant is needed."""
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "/v1/videos" in body                 # documents the real route
    assert 'fetch("/v1/videos"' in body         # and calls it relatively
    assert "http://" not in body.replace("http://www.w3.org", "")   # no hardcoded host


def test_docs_page_needs_no_key(client):
    c, _ = client
    assert c.get("/").status_code == 200        # the page loads; the calls need a key


# --- duration -------------------------------------------------------------

def test_duration_defaults_to_five_seconds(client):
    """5 s is the default because it generates in ~22 s, inside a 60 s proxy
    timeout; 10 s at ~56 s is not."""
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A))
    assert seen["video_length"] == 124        # 17n+5, 5.17 s


def test_ten_seconds_is_selectable(client):
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x", "duration_s": 10}, headers=_hdr(KEY_A))
    assert seen["video_length"] == 243        # 10.125 s


@pytest.mark.parametrize("bad", [7, 15, 0, -5, 20])
def test_other_durations_are_refused(client, bad):
    """Not free-form: an arbitrary duration is both off the frame lattice and a
    way to queue a job more expensive than the caller thinks."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        r = c.post("/v1/videos", json={"prompt": "x", "duration_s": bad}, headers=_hdr(KEY_A))
    assert r.status_code == 422


def test_refused_duration_does_not_consume_quota(client):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        for _ in range(5):
            c.post("/v1/videos", json={"prompt": "x", "duration_s": 7}, headers=_hdr(KEY_A))
        # the daily limit is 3; none of the above should have counted
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 200


def test_every_file_the_app_serves_is_tracked_by_git():
    """.gitignore line 23 is a bare `*.html`. It has now silently dropped a
    committed file three times in this repo (.dockerignore and the CI workflow
    via the bare `.*` on line 1, the sage wheel via `*.whl`, and both the
    webdemo and this gateway page via `*.html`). Every time, `git add -A`
    reported success and CI failed on a file that existed locally.

    So: assert the asset actually exists in git's index, not just on disk."""
    import subprocess
    from runpod_worker.gateway import app as module

    for asset in module.STATIC.rglob("*"):
        if not asset.is_file():
            continue
        rel = asset.relative_to(REPO_ROOT)
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                                 cwd=REPO_ROOT, capture_output=True)
        assert tracked.returncode == 0, (
            f"{rel} is served by the app but is NOT tracked by git -- it will be "
            f"missing in CI and in any fresh clone. Check .gitignore."
        )


def test_health_needs_no_key_so_the_page_can_poll_it(client):
    """The capacity strip polls from page load, before a key is entered."""
    c, m = client
    with mock.patch.object(m, "_rp", return_value={"jobs": {}, "workers": {"ready": 1}}):
        r = c.get("/v1/health")          # no Authorization header
    assert r.status_code == 200
    assert r.json()["capacity"]["ready"] == 1


def test_docs_page_polls_capacity(client):
    c, _ = client
    body = c.get("/").text
    assert 'fetch("/v1/health")' in body
    assert "setInterval(poll" in body
    # the strip must not hijack the message box the generate flow writes to
    strip = body[body.index("function renderStatus"):body.index("async function poll")]
    assert "say(" not in strip
