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

    POST /v1/videos          {"prompt": "..."}        -> 202 {"id", "status"}
    GET  /v1/videos/{id}                              -> status, then metadata
    GET  /v1/videos/{id}/content                      -> the mp4 bytes
    GET  /v1/health

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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

RUNPOD_API = "https://api.runpod.ai/v2"

# ---- the fixed product ----------------------------------------------------
#: 243 frames = 10.125 s at 24 fps. The lattice is 17n + 5, so this is not a
#: free parameter: 243 is the legal value nearest 10 seconds.
VIDEO_LENGTH = 243
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

app = FastAPI(title="Video Generation API", version="1.0.0",
              description="10-second 832x480 video with synchronized audio, from a text prompt.")

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
    today = date.today().isoformat()
    with _lock:
        if _usage[(key, today)] >= DAILY_LIMIT:
            raise HTTPException(429, f"daily limit of {DAILY_LIMIT} videos reached")
        _usage[(key, today)] += 1

    settings: dict[str, Any] = {
        "prompt": body.prompt,
        "resolution": RESOLUTION,
        "video_length": VIDEO_LENGTH,
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
    return {"id": job_id, "status": "queued",
            "duration_s": round(VIDEO_LENGTH / 24, 2), "resolution": RESOLUTION}


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
    path = CACHE / f"{job_id}.mp4"
    if not path.exists() and video.get("kind") == "base64":
        path.write_bytes(base64.b64decode(video["data"]))
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
