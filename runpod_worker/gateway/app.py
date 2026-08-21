#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A minimal customer-facing API in front of one RunPod endpoint.

Why a gateway rather than handing out the endpoint: a RunPod API key is
ACCOUNT-WIDE. There is no per-endpoint key, so a customer given one could list,
modify and delete every pod and endpoint on the account, and run up spend on all
of them. The RunPod key stays here; customers get keys this service issues and
can revoke.

It also pins the product. Callers choose a prompt and (optionally) a seed;
everything that costs money -- clip length, resolution, model, accelerator
profile -- is fixed server-side, so a caller cannot ask for a 20-second 4K clip
and hand you the bill.

    POST /v1/videos   {"prompt": "...", "duration_s": 5}
                                          -> 200 video/mp4  (waits for it)
                                          -> 202 {"id"}      (only if it ran long)
    GET  /v1/videos/{id}                  -> status, for the 202 case
    GET  /v1/videos/{id}/content          -> the mp4 bytes
    GET  /v1/health

One call gives you the file. A warm generation measured ~56 s, well inside a
normal HTTP timeout; a COLD start adds 90-330 s of queue while a worker boots
and fits inside no sane timeout, so that case degrades to a job id instead of
failing.

    export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
    export GATEWAY_KEYS='{"sk_live_demo123":"acme corp"}'
    uvicorn runpod_worker.gateway.app:app --host 0.0.0.0 --port 8000

Single instance, in-memory job index, videos cached on local disk. That is
deliberate for a first customer; it is not multi-replica safe.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

RUNPOD_API = "https://api.runpod.ai/v2"

# ---- the fixed product ----------------------------------------------------
#: The frame lattice is 17n + 5, so a duration is not a free number: these are
#: the legal frame counts nearest 5 and 10 seconds at 24 fps. 5 s generates in
#: roughly 22 s warm and 10 s in roughly 56 s, so the short one is the default:
#: it fits inside a 60 s proxy timeout, which the long one does not.
DURATIONS = {5: 124, 10: 243}
DEFAULT_DURATION = 5
RESOLUTION = "832x480"
ACCEL_PROFILE = "Turbo Lightx2v FL2V 4 Steps v1.0 768p"
MODEL_TYPE = "minimax_h3_fl2va_pruned"

CACHE = Path(os.environ.get("GATEWAY_CACHE", "/tmp/gateway-videos"))
CACHE.mkdir(parents=True, exist_ok=True)


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _keys() -> dict[str, str]:
    """``{api_key: customer label}``. Revoke by removing one and restarting."""
    try:
        return json.loads(os.environ.get("GATEWAY_KEYS", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GATEWAY_KEYS is not valid JSON: {exc}") from exc


DAILY_LIMIT = int(os.environ.get("GATEWAY_DAILY_LIMIT", "100"))

#: How long POST /v1/videos will hold the connection before handing back a job
#: id instead. This MUST sit below whatever proxy fronts this service, or the
#: proxy cuts the connection first and the caller gets an opaque gateway error
#: instead of the clean 202 -- the 202 never gets a chance to be sent.
#:
#: 90 s is the default because the common ceilings are low: Cloudflare cuts at
#: 100 s, and RunPod's own *.proxy.runpod.net is Cloudflare-fronted, so a pod
#: deployed there is already behind that limit. Measured with a 240 s value on
#: RunPod's proxy, a slow generation returned HTTP 524 at 125 s.
#:
#: A warm 5 s clip generates in ~22 s and returns inline with room to spare.
SYNC_TIMEOUT = float(os.environ.get("GATEWAY_SYNC_TIMEOUT", "90"))
POLL_INTERVAL = float(os.environ.get("GATEWAY_POLL_INTERVAL", "2"))

app = FastAPI(title="Video Generation API", version="1.0.0",
              description="10-second 832x480 video with synchronized audio, from a text prompt.")

#: The docs page is served from this app deliberately: same origin as the API,
#: so the live demo needs no CORS grant and no key in a query string.
STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def docs_page():
    index = STATIC / "index.html"
    if not index.exists():
        raise HTTPException(404, "docs page not installed")
    return FileResponse(index, media_type="text/html")

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}          # our id -> {runpod_id, owner, ...}
_usage: dict[tuple[str, str], int] = defaultdict(int)   # (key, YYYY-MM-DD) -> count


def auth(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    owner = _keys().get(token)
    if not owner:
        raise HTTPException(401, "invalid or missing API key")
    return token


def _rp(path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    url = f"{RUNPOD_API}/{_env('RUNPOD_ENDPOINT_ID')}/{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_env('RUNPOD_API_KEY')}"},
        method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    #: 5 or 10 seconds. Constrained rather than free: the model's frame lattice
    #: only admits 17n + 5, and an open duration would let a caller queue a job
    #: far more expensive than the one they think they are buying.
    duration_s: int = Field(default=DEFAULT_DURATION,
                            description="clip length in seconds: 5 or 10")
    #: -1 (or omitted) picks a random seed; the resolved value is returned so a
    #: generation can be reproduced exactly.
    seed: int | None = Field(default=None, ge=-1, le=2**31 - 1)


@app.get("/v1/health")
def health() -> dict:
    try:
        h = _rp("health", timeout=15)
    except Exception:
        # Never leak RunPod's error surface to a customer.
        raise HTTPException(503, "generation backend unavailable") from None
    w, j = h.get("workers", {}), h.get("jobs", {})
    return {"status": "ok",
            "queue": {"waiting": j.get("inQueue", 0), "running": j.get("inProgress", 0)},
            "capacity": {"ready": w.get("ready", 0), "starting": w.get("initializing", 0)}}


@app.post("/v1/videos", status_code=202)
def create(body: VideoRequest, key: str = Depends(auth)) -> dict:
    if body.duration_s not in DURATIONS:
        raise HTTPException(422, f"duration_s must be one of {sorted(DURATIONS)}")
    today = date.today().isoformat()
    with _lock:
        if _usage[(key, today)] >= DAILY_LIMIT:
            raise HTTPException(429, f"daily limit of {DAILY_LIMIT} videos reached")
        _usage[(key, today)] += 1

    frames = DURATIONS[body.duration_s]
    settings: dict[str, Any] = {
        "prompt": body.prompt,
        "resolution": RESOLUTION,
        "video_length": frames,
        "sample_solver": "euler",
        "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": "",
    }
    if body.seed is not None and body.seed >= 0:
        settings["seed"] = body.seed

    try:
        created = _rp("run", {"input": {
            "model_type": MODEL_TYPE, "profile": ACCEL_PROFILE,
            "settings": settings, "output": {"mode": "auto"},
            "runtime": {"timeout_s": 1200},
        }})
    except Exception:
        with _lock:
            _usage[(key, today)] -= 1        # a failed submit must not bill the quota
        raise HTTPException(503, "could not queue the job") from None

    job_id = created.get("id")
    if not job_id:
        raise HTTPException(503, "could not queue the job")
    with _lock:
        _jobs[job_id] = {"owner": key, "created": time.time(), "seed": body.seed}

    # Hold the connection and hand back the mp4 itself. One call, one file, no
    # polling -- which is the whole point of this route.
    deadline = time.monotonic() + SYNC_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            st = _rp(f"status/{job_id}", timeout=30)
        except Exception:
            continue                      # a blip mid-generation is not a failure
        state = st.get("status")
        if state in ("IN_QUEUE", "IN_PROGRESS"):
            continue
        if state != "COMPLETED":
            message = (st.get("output") or {}).get("message", "generation failed")
            raise HTTPException(502, message)
        path = _materialise(job_id, st)
        if path is None:
            raise HTTPException(502, "generation produced no video")
        return _video_response(job_id, path, st)

    # Out of time -- almost always a cold start. The work is still running, so
    # return the id rather than throwing away a job the customer has paid for.
    return JSONResponse(
        status_code=202,
        headers={"Retry-After": "30", "Location": f"/v1/videos/{job_id}"},
        content={"id": job_id, "status": "processing",
                 "message": ("still generating past the synchronous limit "
                             f"({SYNC_TIMEOUT:.0f}s); poll the url in Location"),
                 "poll_url": f"/v1/videos/{job_id}"})


def _materialise(job_id: str, st: dict) -> Path | None:
    """Write the returned video to the cache and return its path."""
    video = ((st.get("output") or {}).get("video") or {})
    path = CACHE / f"{job_id}.mp4"
    if not path.exists():
        if video.get("kind") != "base64":
            return None
        path.write_bytes(base64.b64decode(video["data"]))
    return path


def _video_response(job_id: str, path: Path, st: dict) -> FileResponse:
    """The mp4, with the reproducibility metadata on headers.

    The body has to be the file for this to be a one-call API, so anything a
    caller needs alongside it goes in headers rather than a JSON envelope.
    """
    out = st.get("output") or {}
    video = out.get("video") or {}
    return FileResponse(
        path, media_type="video/mp4", filename=f"{job_id}.mp4",
        headers={
            "X-Video-Id": job_id,
            "X-Seed": str(out.get("seed", "")),
            "X-Duration-Seconds": str(video.get("duration_s", "")),
            "X-Width": str(video.get("width", "")),
            "X-Height": str(video.get("height", "")),
            "X-Has-Audio": str(bool(video.get("has_audio"))).lower(),
            "X-Generate-Seconds": str((out.get("metrics") or {}).get("generate_s", "")),
        })


def _owned(job_id: str, key: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    # 404 rather than 403 for someone else's job: a customer should not be able
    # to probe which ids exist.
    if not job or job["owner"] != key:
        raise HTTPException(404, "no such job")
    return job


@app.get("/v1/videos/{job_id}")
def status(job_id: str, key: str = Depends(auth)) -> dict:
    _owned(job_id, key)
    try:
        st = _rp(f"status/{job_id}", timeout=30)
    except Exception:
        raise HTTPException(503, "generation backend unavailable") from None

    state = st.get("status")
    if state in ("IN_QUEUE", "IN_PROGRESS"):
        return {"id": job_id, "status": "queued" if state == "IN_QUEUE" else "processing"}
    if state != "COMPLETED":
        return {"id": job_id, "status": "failed",
                "error": (st.get("output") or {}).get("message", "generation failed")}

    out = st.get("output") or {}
    video = out.get("video") or {}
    path = _materialise(job_id, st) or CACHE / f"{job_id}.mp4"
    return {"id": job_id, "status": "completed",
            "seed": out.get("seed"),
            "duration_s": video.get("duration_s"),
            "width": video.get("width"), "height": video.get("height"),
            "has_audio": video.get("has_audio"),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "content_url": f"/v1/videos/{job_id}/content"}


@app.get("/v1/videos/{job_id}/content")
def content(job_id: str, key: str = Depends(auth)):
    _owned(job_id, key)
    path = CACHE / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "video not ready; poll /v1/videos/{id} first")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{job_id}.mp4")
