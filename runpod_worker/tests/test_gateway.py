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
    with mock.patch.object(m, "_rp", return_value={"id": "job-1"}):
        r = c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A))
    job = r.json()["id"]
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

    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos",
                   json={"prompt": "x", "video_length": 481, "resolution": "1920x1080"},
                   headers=_hdr(KEY_A))
    assert r.status_code == 202
    assert seen["video_length"] == 243        # 17n+5, ~10 s
    assert seen["resolution"] == "832x480"


def test_seed_is_passed_through_and_omitted_when_random(client):
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.clear(); seen.update(payload["input"]["settings"])
        return {"id": "job-1"}

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x", "seed": 42}, headers=_hdr(KEY_A))
        assert seen["seed"] == 42
        c.post("/v1/videos", json={"prompt": "x", "seed": -1}, headers=_hdr(KEY_A))
        assert "seed" not in seen


# --- quota -----------------------------------------------------------------

def test_daily_limit_is_enforced(client):
    c, m = client
    with mock.patch.object(m, "_rp", return_value={"id": "job-1"}):
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 202
        assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 429
    # the other customer has their own budget
    with mock.patch.object(m, "_rp", return_value={"id": "job-2"}):
        assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_B)).status_code == 202


def test_failed_submit_does_not_consume_quota(client):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=RuntimeError("backend down")):
        assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 503
    with mock.patch.object(m, "_rp", return_value={"id": "job-1"}):
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 202


# --- lifecycle -------------------------------------------------------------

def test_status_and_content_round_trip(client):
    import base64
    c, m = client
    mp4 = b"\x00\x00\x00\x18ftypisom-fake-bytes"
    completed = {"status": "COMPLETED", "output": {
        "seed": 7, "video": {"kind": "base64", "data": base64.b64encode(mp4).decode(),
                             "duration_s": 10.125, "width": 832, "height": 480,
                             "has_audio": True}}}
    with mock.patch.object(m, "_rp", return_value={"id": "job-1"}):
        job = c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).json()["id"]
    with mock.patch.object(m, "_rp", return_value=completed):
        body = c.get(f"/v1/videos/{job}", headers=_hdr(KEY_A)).json()
    assert body["status"] == "completed" and body["seed"] == 7
    assert body["size_bytes"] == len(mp4)
    r = c.get(f"/v1/videos/{job}/content", headers=_hdr(KEY_A))
    assert r.status_code == 200 and r.content == mp4


def test_backend_errors_are_not_leaked(client):
    """A RunPod traceback or key must never reach a customer."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=RuntimeError("rpa_secretkey leaked!")):
        r = c.get("/v1/health")
    assert r.status_code == 503
    assert "rpa_" not in r.text and "secretkey" not in r.text
