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
pytest.importorskip("sqlalchemy", reason="gateway requirements not installed")
from fastapi.testclient import TestClient  # noqa: E402

KEY_A, KEY_B = "sk_test_aaa", "sk_test_bbb"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_stub")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep_stub")
    monkeypatch.setenv("GATEWAY_KEYS", json.dumps({KEY_A: "acme", KEY_B: "globex"}))
    monkeypatch.setenv("GATEWAY_CACHE", str(tmp_path))
    monkeypatch.setenv("GATEWAY_DAILY_LIMIT", "3")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/gw.db")
    import importlib
    from runpod_worker.gateway import db as dbmod
    dbmod.reset_for_tests()
    from runpod_worker.gateway import app as module
    importlib.reload(module)
    # TestClient's context manager fires the startup hook that seeds env keys.
    c = TestClient(module.app)
    c.__enter__()
    return c, module


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

    # A raw pixel resolution is REFUSED rather than silently ignored: a caller
    # who asks for 1080p should learn it is unavailable, not receive 480p.
    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos",
                   json={"prompt": "x", "video_length": 481, "resolution": "1920x1080"},
                   headers=_hdr(KEY_A))
    assert r.status_code == 422

    # And a raw video_length is ignored: only duration_s (5|10) is honoured.
    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos", json={"prompt": "x", "video_length": 481},
                   headers=_hdr(KEY_A))
    assert r.status_code == 200
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


@pytest.mark.parametrize("bad", [7, 12, 0, -5, 20])
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
    """/v1/health is keyless: the legacy page polls it from page load, and
    API callers use it to see whether the next request hits a warm worker."""
    c, m = client
    with mock.patch.object(m, "_rp", return_value={"jobs": {}, "workers": {"ready": 1}}):
        r = c.get("/v1/health")          # no Authorization header
    assert r.status_code == 200
    assert r.json()["capacity"]["ready"] == 1


def test_studio_v2_signup_markers(client):
    """Studio v2 swaps the header status strip for account buttons and adds
    the preview/Pro messaging. The page must no longer poll /v1/health."""
    c, _ = client
    body = c.get("/").text
    assert "Sign up free" in body
    assert "Pro feature — free during the preview" in body
    assert "no account needed for 5 s at 480p" in body
    assert "keeps your clips past the 1-hour limit" in body
    assert 'fetch("/v1/health")' not in body
    assert "setInterval(poll" not in body


def test_background_returns_immediately(client):
    """A browser cannot hold a 30 s fetch: phones abort it when the screen
    locks. background:true closes the connection at once and the caller polls."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend(status={"status": "IN_QUEUE"})):
        r = c.post("/v1/videos", json={"prompt": "x", "background": True}, headers=_hdr(KEY_A))
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["poll_url"] == "/v1/videos/job-1"
    assert r.headers["location"] == "/v1/videos/job-1"


def test_background_still_counts_against_quota(client):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend(status={"status": "IN_QUEUE"})):
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x", "background": True},
                          headers=_hdr(KEY_A)).status_code == 202
        assert c.post("/v1/videos", json={"prompt": "x", "background": True},
                      headers=_hdr(KEY_A)).status_code == 429


def test_page_uses_background_and_polls(client):
    c, _ = client
    body = c.get("/").text
    assert "background: true" in body
    assert "poll_url" in body


# --- resolution ------------------------------------------------------------

def test_resolution_defaults_to_480p(client):
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A))
    assert seen["resolution"] == "832x480"


def test_720p_is_forced_to_background(client):
    """720p takes minutes. Holding the connection until the proxy kills it loses
    the job id, which is the one thing the caller needs -- so the combination is
    not offered at all."""
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos",
                   json={"prompt": "x", "resolution": "720p", "background": False},
                   headers=_hdr(KEY_A))
    assert r.status_code == 202              # not 200, despite background:false
    assert seen["resolution"] == "1280x704"


@pytest.mark.parametrize("bad", ["1080p", "4k", "832x480", ""])
def test_unknown_resolution_refused(client, bad):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        r = c.post("/v1/videos", json={"prompt": "x", "resolution": bad}, headers=_hdr(KEY_A))
    assert r.status_code == 422


def test_720p_uses_a_resolution_the_model_actually_accepts():
    """1280x720 is REJECTED by the worker: the VAE's 16x spatial compression and
    the patch size put the height on a lattice that 720 misses, and schema.py
    answers "nearest valid: 1280x704". A live job failed in 63 ms on exactly
    this before it was corrected."""
    from runpod_worker.gateway import app as module
    assert module.DIMENSIONS[("720p", "horizontal")] == "1280x704"


# --- square + image input --------------------------------------------------

def test_square_resolution(client):
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos", json={"prompt": "x", "resolution": "square"}, headers=_hdr(KEY_A))
    assert r.status_code == 202                 # above 480p -> background
    assert seen["resolution"] == "960x960"


def test_every_offered_resolution_is_on_the_block_lattice():
    """block_size is 32 (schema.py:465). A resolution off the lattice fails the
    job instantly -- 1280x720 did exactly that -- so every cell of the
    aspect x tier matrix must be a multiple of 32 in BOTH dimensions."""
    from runpod_worker.gateway import app as module
    assert len(module.DIMENSIONS) == len(module.RESOLUTION_TIERS) * len(module.ASPECTS)
    for cell, value in module.DIMENSIONS.items():
        w, h = (int(x) for x in value.split("x"))
        assert w % 32 == 0 and h % 32 == 0, f"{cell}={value} is off the 32px lattice"


def test_image_sets_both_the_letter_and_the_attachment(client):
    """wgp.py:1409 reads image_start ONLY when "S" is in image_prompt_type. Set
    one without the other and the image is silently ignored -- the caller gets a
    video that simply does not use their frame, with no error."""
    c, m = client
    sent = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            sent["settings"] = payload["input"]["settings"]
            sent["media"] = payload["input"].get("media")
            return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x", "image": "aGVsbG8="}, headers=_hdr(KEY_A))
    assert sent["settings"]["image_prompt_type"] == "S"
    assert sent["media"]["image_start"] == {"b64": "aGVsbG8="}


def test_data_uri_prefix_is_stripped(client):
    """A browser FileReader yields 'data:image/png;base64,AAAA'; the worker wants
    the payload only."""
    c, m = client
    sent = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            sent.update(payload["input"].get("media") or {}); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x", "image": "data:image/png;base64,QUJD"},
               headers=_hdr(KEY_A))
    assert sent["image_start"] == {"b64": "QUJD"}


def test_no_image_means_no_attachment(client):
    c, m = client
    sent = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            sent["settings"] = payload["input"]["settings"]
            sent["media"] = payload["input"].get("media")
            return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A))
    assert sent["media"] == {}
    assert sent["settings"]["image_prompt_type"] == ""


def test_page_accepts_a_key_in_the_url_but_does_not_keep_it_there(client):
    """?key=... makes one link the whole demo, but the key must not linger in
    the address bar: it would leak into browser history, the Referer header on
    any outbound link, and every proxy log in between."""
    c, _ = client
    body = c.get("/").text
    assert 'searchParams.get("key")' in body
    assert 'searchParams.delete("key")' in body
    assert "history.replaceState" in body


def test_api_never_accepts_a_key_as_a_query_parameter(client):
    """Convenience on the docs page is one thing; query-string auth on the API
    would write the key into a server log line on every request."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        r = c.post("/v1/videos?key=" + KEY_A, json={"prompt": "x"})
    assert r.status_code == 401


# --- durability: the reason the database exists ------------------------------

def _restarted(module):
    """Simulate a gateway restart: new app object, same DATABASE_URL."""
    import importlib
    from runpod_worker.gateway import db as dbmod
    dbmod.reset_for_tests()
    fresh = importlib.reload(module)
    c = TestClient(fresh.app)
    c.__enter__()
    return c, fresh


def test_job_ownership_survives_a_restart(client):
    """The in-memory version orphaned every paid job on restart: the customer
    had an id, the gateway had no idea whose it was, and the 404-for-unowned
    rule turned their own job into 'no such job'."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend(status={"status": "IN_QUEUE"})):
        job = c.post("/v1/videos", json={"prompt": "x", "background": True},
                     headers=_hdr(KEY_A)).json()["id"]
    c2, m2 = _restarted(m)
    with mock.patch.object(m2, "_rp", side_effect=_backend()):
        r = c2.get(f"/v1/videos/{job}", headers=_hdr(KEY_A))
    assert r.status_code == 200 and r.json()["status"] == "completed"
    # and it is still invisible to the other key
    assert c2.get(f"/v1/videos/{job}", headers=_hdr(KEY_B)).status_code == 404


def test_quota_survives_a_restart(client):
    """A restart must not hand every key a fresh daily allowance."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend(status={"status": "IN_QUEUE"})):
        for _ in range(3):
            assert c.post("/v1/videos", json={"prompt": "x", "background": True},
                          headers=_hdr(KEY_A)).status_code == 202
    c2, m2 = _restarted(m)
    with mock.patch.object(m2, "_rp", side_effect=_backend(status={"status": "IN_QUEUE"})):
        assert c2.post("/v1/videos", json={"prompt": "x", "background": True},
                       headers=_hdr(KEY_A)).status_code == 429


def test_revocation_beats_the_env_var(client):
    """GATEWAY_KEYS seeds the database; it does not resurrect. A key revoked in
    the DB stays dead even though it is still sitting in the env, because env
    vars linger in deploy configs long after a credential should be gone."""
    c, m = client
    from runpod_worker.gateway import db as dbmod
    import sqlalchemy as sa, datetime
    with dbmod.engine().begin() as cx:
        cx.execute(dbmod.api_keys.update()
                   .where(dbmod.api_keys.c.key_hash == dbmod.hash_key(KEY_A))
                   .values(revoked_at=datetime.datetime.utcnow()))
    assert c.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 401
    # ...and a restart (which re-seeds from the env) must NOT un-revoke it
    c2, _ = _restarted(m)
    assert c2.post("/v1/videos", json={"prompt": "x"}, headers=_hdr(KEY_A)).status_code == 401


def test_generate_seconds_lands_in_the_jobs_table(client):
    """generate_s is what per-second billing computes from; losing it means
    billing from estimates."""
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        job = c.post("/v1/videos", json={"prompt": "x", "background": True},
                     headers=_hdr(KEY_A)).json()["id"]
        c.get(f"/v1/videos/{job}", headers=_hdr(KEY_A))
    from runpod_worker.gateway import db as dbmod
    import sqlalchemy as sa
    with dbmod.engine().connect() as cx:
        row = cx.execute(sa.select(dbmod.jobs).where(dbmod.jobs.c.id == job)).mappings().first()
    assert row["status"] == "completed"
    assert row["generate_s"] == 55.8
    # actual pixel dimensions, not the tier label -- billing wants what ran
    assert row["resolution"] == "832x480" and row["duration_s"] == 5


def test_raw_keys_never_touch_the_database(client):
    """A database dump must not be a credential dump."""
    c, m = client
    from runpod_worker.gateway import db as dbmod
    import sqlalchemy as sa
    with dbmod.engine().connect() as cx:
        rows = [dict(r) for r in cx.execute(sa.select(dbmod.api_keys)).mappings()]
    blob = json.dumps(rows, default=str)
    assert KEY_A not in blob and KEY_B not in blob
    assert dbmod.hash_key(KEY_A) in blob


# --- studio additions --------------------------------------------------------

def test_fifteen_seconds_is_selectable_and_background_only(client):
    """362 frames is on the 17n+5 lattice (the max single window) but takes
    ~98 s at 480p -- past SYNC_TIMEOUT, so it must come back as a job id."""
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos", json={"prompt": "x", "duration_s": 15}, headers=_hdr(KEY_A))
    assert r.status_code == 202
    assert seen["video_length"] == 362


@pytest.mark.parametrize("tier,aspect,expected", [
    ("480p", "portrait", "480x832"),
    ("480p", "square", "640x640"),
    ("720p", "portrait", "704x1280"),
    ("720p", "square", "960x960"),
])
def test_aspect_ratio_matrix(client, tier, aspect, expected):
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        c.post("/v1/videos", json={"prompt": "x", "resolution": tier,
                                   "aspect_ratio": aspect, "background": True},
               headers=_hdr(KEY_A))
    assert seen["resolution"] == expected


def test_legacy_square_alias_still_works(client):
    """The first customer integration was given resolution="square"; it must
    keep meaning 960x960 even though the new API spells it 720p + square."""
    c, m = client
    seen = {}

    def fake(path, payload=None, timeout=60):
        if payload:
            seen.update(payload["input"]["settings"]); return {"id": "job-1"}
        return COMPLETED

    with mock.patch.object(m, "_rp", side_effect=fake):
        r = c.post("/v1/videos", json={"prompt": "x", "resolution": "square"},
                   headers=_hdr(KEY_A))
    assert r.status_code == 202
    assert seen["resolution"] == "960x960"


def test_bad_aspect_refused(client):
    c, m = client
    with mock.patch.object(m, "_rp", side_effect=_backend()):
        r = c.post("/v1/videos", json={"prompt": "x", "aspect_ratio": "vertical"},
                   headers=_hdr(KEY_A))
    assert r.status_code == 422


def test_cache_purge_honours_retention(client, tmp_path, monkeypatch):
    """The Studio page says 'kept for 1 hour'; the purge is what makes that
    true. Billing rows must survive the bytes expiring."""
    import os, time as _t
    c, m = client
    old = m.CACHE / "ancient.mp4"; old.write_bytes(b"x")
    os.utime(old, (_t.time() - 7200, _t.time() - 7200))
    fresh = m.CACHE / "fresh.mp4"; fresh.write_bytes(b"y")
    with mock.patch.object(m, "_rp", side_effect=_backend(status={"status": "IN_QUEUE"})):
        c.post("/v1/videos", json={"prompt": "x", "background": True}, headers=_hdr(KEY_A))
    assert not old.exists(), "expired video should be purged on the next submit"
    assert fresh.exists(), "fresh video must survive the purge"


def test_studio_page_is_the_landing_page(client):
    c, _ = client
    body = c.get("/").text
    assert "Minimax H3 Studio" in body
    assert "aspect_ratio: state.aspect" in body       # the page sends the new field
    assert "kept for 1 hour" in body
    legacy = c.get("/legacy")
    assert legacy.status_code == 200 and "Video Generation API" in legacy.text
