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
import re
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

try:
    from . import db                     # imported as runpod_worker.gateway.app
    from . import story
except ImportError:                      # flat /app layout in the container
    import db                            # type: ignore[no-redef]
    import story                         # type: ignore[no-redef]

import jwt
from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

RUNPOD_API = "https://api.runpod.ai/v2"

# ---- the fixed product ----------------------------------------------------
#: The frame lattice is 17n + 5, so a duration is not a free number: these are
#: the legal frame counts nearest 5 and 10 seconds at 24 fps. 5 s generates in
#: roughly 22 s warm and 10 s in roughly 56 s, so the short one is the default:
#: it fits inside a 60 s proxy timeout, which the long one does not.
#: 17n + 5 frames at 24 fps. 362 (15.08 s) is the model's maximum single
#: window; beyond it WanGP starts chaining sliding windows internally.
DURATIONS = {5: 124, 10: 243, 15: 362}
DEFAULT_DURATION = 5
#: The Studio design's aspect x tier matrix. Every cell is a multiple of 32 in
#: both dimensions -- the model's block lattice. NOT 1280x720 / 720x1280: 720
#: is off the lattice and schema.py:1909 rejects it instantly ("nearest valid:
#: 1280x704"); 704 is the real 16:9-ish 720p. The turbo LoRA is 768p-trained,
#: so the 720p tier sits near its native size.
DIMENSIONS = {
    ("480p", "horizontal"): "832x480",     # every measurement in the README
    ("480p", "portrait"):  "480x832",
    ("480p", "square"):    "640x640",
    ("720p", "horizontal"): "1280x704",
    ("720p", "portrait"):  "704x1280",
    ("720p", "square"):    "960x960",      # 1.02x the pixels of 1280x704
}
RESOLUTION_TIERS = ("480p", "720p")
ASPECTS = ("horizontal", "portrait", "square")
DEFAULT_RESOLUTION = "480p"
DEFAULT_ASPECT = "horizontal"
ACCEL_PROFILE = "Turbo Lightx2v FL2V 4 Steps v1.1 768p"
MODEL_TYPE = "minimax_h3_fl2va_pruned"

# ---- prompt enhancement ----------------------------------------------------
#: RunPod's managed public LLM endpoint (create an API key, call it -- no
#: deployment). The SAME account key the video endpoint uses works here, so
#: enhancement adds no new credential. Token-priced, cents per thousand
#: enhancements; a failure of any kind falls back to the raw prompt rather
#: than failing the generation the caller actually paid for.
LLM_ENDPOINT = os.environ.get("GATEWAY_LLM_ENDPOINT", "qwen3-32b-awq")
LLM_MODEL = os.environ.get("GATEWAY_LLM_MODEL", "Qwen/Qwen3-32B-AWQ")
LLM_TIMEOUT_S = float(os.environ.get("GATEWAY_LLM_TIMEOUT_S", "30"))
#: A prompt already in the H3 structured format (the example chips, a reused
#: gallery prompt, a power user pasting the full format) must NOT go through
#: the enhancer again -- it IS the enhanced form.
_H3_FORMAT_PREFIX = "integrated_multimodal_description:"


def _load_prompt_guide() -> str | None:
    """MiniMax's official FL2VA prompt-writing guide, shipped with the repo as
    models/minimax_h3/prompt_enhancer.py (pure string definitions). The
    container build copies that file next to this one as prompt_guide.py; the
    repo layout finds it in the tree. No guide -> enhancement quietly becomes
    a pass-through rather than a crash at import."""
    try:
        try:
            from . import prompt_guide  # type: ignore[attr-defined]
        except ImportError:
            import prompt_guide  # type: ignore[no-redef]
        return prompt_guide.FL2VA_PROMPT_INFOS
    except Exception:  # noqa: BLE001
        pass
    try:
        source = (Path(__file__).resolve().parents[2]
                  / "models" / "minimax_h3" / "prompt_enhancer.py")
        namespace: dict[str, Any] = {}
        exec(source.read_text(), namespace)  # noqa: S102 - our own checked-in file
        return namespace["FL2VA_PROMPT_INFOS"]
    except Exception:  # noqa: BLE001
        return None


PROMPT_GUIDE = _load_prompt_guide()


def _llm_chat(messages: list[dict], max_tokens: int = 800) -> str:
    """One chat completion against the public endpoint; raises on any failure
    (the caller decides what a failure means)."""
    req = urllib.request.Request(
        f"{RUNPOD_API}/{LLM_ENDPOINT}/openai/v1/chat/completions",
        data=json.dumps({"model": LLM_MODEL, "messages": messages,
                         "max_tokens": max_tokens, "temperature": 0.7}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_env('RUNPOD_API_KEY')}"})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as r:
        payload = json.loads(r.read())
    return (payload["choices"][0]["message"].get("content") or "").strip()


def _enhance_prompt(prompt: str, duration_s: int) -> str:
    """Expand a plain-language idea into the full H3 structured prompt.

    Fail-open by design: enhancement is a quality upgrade, never a gate. Any
    problem -- guide missing, endpoint down, timeout, output that is not in
    the H3 format -- returns the original prompt unchanged.
    """
    if not PROMPT_GUIDE:
        return prompt
    if prompt.lstrip().lower().startswith(_H3_FORMAT_PREFIX):
        return prompt
    shots = ("a single continuous shot" if duration_s <= 5
             else "one to three shots with timed cuts")
    system = (
        "You are a prompt writer for the MiniMax H3 video model. Rewrite the "
        "user's idea into one H3 FL2VA prompt following this official guide "
        "exactly. Output ONLY the prompt text, no commentary, no markdown "
        f"fences. Target {shots} totalling about {duration_s} seconds. "
        "/no_think\n\n" + PROMPT_GUIDE)
    try:
        out = _llm_chat([{"role": "system", "content": system},
                         {"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001 - fail open, the generation must proceed
        return prompt
    if not out.lstrip().lower().startswith(_H3_FORMAT_PREFIX):
        return prompt
    return out[:4000]

CACHE = Path(os.environ.get("GATEWAY_CACHE", "/tmp/gateway-videos"))
CACHE.mkdir(parents=True, exist_ok=True)

#: How long a finished mp4 stays downloadable. Five minutes: the server is a
#: delivery buffer, not storage -- the Studio page saves each clip into the
#: viewer's browser (IndexedDB) the moment it downloads, and that copy is the
#: durable one. Short retention is also the privacy story: we do not keep
#: customers' videos. Billing rows in the jobs table are unaffected -- only
#: the bytes expire.
RETENTION_S = int(os.environ.get("GATEWAY_RETENTION_S", "300"))


def _purge_expired_cache() -> None:
    cutoff = time.time() - RETENTION_S
    try:
        for f in CACHE.glob("*.mp4"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except OSError:
        pass                                  # purging is best effort


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _seed_env_keys() -> None:
    """GATEWAY_KEYS stays supported as a bootstrap path: keys named there are
    upserted into the database at startup. The database is the authority --
    revoking a key there beats its presence in the env."""
    try:
        mapping = json.loads(os.environ.get("GATEWAY_KEYS", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GATEWAY_KEYS is not valid JSON: {exc}") from exc
    db.seed_keys(mapping)


DAILY_LIMIT = int(os.environ.get("GATEWAY_DAILY_LIMIT", "100"))
#: The Studio page's freemium ladder, enforced here rather than trusted to the
#: page: a visitor with no credential at all gets a 2-clip taste of the free
#: tier (5 s at 480p only), and a free Clerk account gets the beta allowance
#: the page advertises. sk_ API customers keep GATEWAY_DAILY_LIMIT.
ANON_DAILY_LIMIT = int(os.environ.get("GATEWAY_ANON_DAILY_LIMIT", "2"))
CLERK_DAILY_LIMIT = int(os.environ.get("GATEWAY_CLERK_DAILY_LIMIT", "20"))

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

# No auto-generated docs pages: the Studio page is the product surface, and
# API customers get reference material directly.
app = FastAPI(title="Video Generation API", version="1.0.0",
              docs_url=None, redoc_url=None, openapi_url=None)

#: The docs page is served from this app deliberately: same origin as the API,
#: so the live demo needs no CORS grant and no key in a query string.
STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def studio_page():
    index = STATIC / "index.html"
    if not index.exists():
        raise HTTPException(404, "studio page not installed")
    return FileResponse(index, media_type="text/html")


#: Real example generations the Studio gallery seeds itself with, so a
#: first-time visitor sees the product before spending a free generation.
_EXAMPLES = STATIC / "examples"
if _EXAMPLES.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/examples", StaticFiles(directory=_EXAMPLES), name="examples")


@app.on_event("startup")
def _startup() -> None:
    _seed_env_keys()
    for slug in story.STORIES:
        nodes = story.nodes_of(slug)
        db.adventure_seed(slug, [
            {"id": nid, "position": pos, "depth": story.depth_of(slug, nid),
             "title": nodes[nid]["title"], "prompt": nodes[nid]["prompt"],
             "parent_id": story.parent_of(slug, nid)}
            for pos, nid in enumerate(story.order_of(slug))])
    if os.environ.get("GATEWAY_ADVENTURE_AUTOGEN", "1").strip() != "0":
        _start_adventure_renderer()


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Process-liveness only, for the platform's health check. /v1/health
    reports the GPU backend and returns 503 when it is down -- pointing a
    restart-on-unhealthy probe at THAT turns a RunPod outage into a gateway
    restart loop."""
    return {"ok": True}


# ---- Clerk sign-in ---------------------------------------------------------
#: The Studio page signs people in with Clerk; API customers keep sk_ keys.
#: The publishable key is public by design -- it ships in the page source of
#: every Clerk site -- and encodes the instance domain, which is where ClerkJS
#: is served from and the issuer of session tokens. Override for another
#: instance (a production pk_live_) without a code change.
CLERK_PUBLISHABLE_KEY = os.environ.get(
    "CLERK_PUBLISHABLE_KEY",
    "pk_test_aW1wcm92ZWQtYnVjay00ODk3LmNsZXJrLmFjY291bnRzLmRldiQ").strip()


def _clerk_domain() -> str | None:
    """pk_test_<base64 of "domain$"> -> the instance's frontend API domain."""
    try:
        b64 = CLERK_PUBLISHABLE_KEY.split("_", 2)[2]
        domain = base64.b64decode(b64 + "=" * (-len(b64) % 4)).decode().rstrip("$")
        return domain or None
    except Exception:
        return None


_clerk_jwks: jwt.PyJWKClient | None = None


def _jwks_client() -> jwt.PyJWKClient:
    """Signing keys for Clerk session tokens, from the instance's public
    well-known URL. Deliberately NOT api.clerk.com with CLERK_SECRET_KEY:
    that fetch adds a credential that can be misconfigured (any 4xx from it
    surfaced to customers as a 503 on every signed-in call), while the
    public URL serves the same keys for free and pins itself to the same
    instance the issuer check verifies."""
    global _clerk_jwks
    if _clerk_jwks is None:
        _clerk_jwks = jwt.PyJWKClient(
            f"https://{_clerk_domain()}/.well-known/jwks.json", lifespan=3600)
    return _clerk_jwks


class Ident(NamedTuple):
    """Who is calling, reduced to what the handlers need. key_hash is what
    job ownership keys off and quota_hash is what the daily count is charged
    to -- the raw credential never sits in the job table or the logs. They
    differ only for visitors: ownership rides on the page's device token
    (stable across a wifi-to-cellular hop) while quota stays on the client
    address (not resettable by clearing browser storage). kind gates what
    the caller may buy."""
    key_hash: str
    kind: str            # "key" | "clerk" | "visitor"
    limit: int
    quota_hash: str


def _auth_clerk(token: str) -> Ident:
    """The signature is verified against Clerk's JWKS before ANY claim is
    trusted -- a forged token must not impersonate a user -- and only the
    verified sub becomes the identity that quota and jobs key off."""
    domain = _clerk_domain()
    if domain is None:
        raise HTTPException(401, "sign-in is not enabled on this deployment")
    try:
        key = _jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, key.key, algorithms=["RS256"],
            issuer=f"https://{domain}",
            options={"verify_aud": False},    # session tokens carry azp, not aud
            leeway=10)
    except jwt.exceptions.PyJWKClientConnectionError:
        raise HTTPException(503, "could not reach the sign-in service") from None
    except (jwt.exceptions.PyJWTError, jwt.exceptions.PyJWKClientError):
        raise HTTPException(401, "invalid or expired session") from None
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(401, "invalid or expired session")
    row = db.ensure_key(f"clerk:{sub}", label=claims.get("email") or sub,
                        daily_limit=CLERK_DAILY_LIMIT)
    if row is None:
        raise HTTPException(401, "this account has been disabled")
    kh = row["key_hash"]
    return Ident(kh, "clerk", row["daily_limit"] or CLERK_DAILY_LIMIT, kh)


_VISITOR_TOKEN = re.compile(r"^vt_[A-Za-z0-9-]{8,64}$")


def _auth_visitor(request: Request, token: str) -> Ident:
    """The free tier. Quota is charged to the client address: the LAST
    X-Forwarded-For entry is the one Railway's edge appended, while the first
    can be whatever the client wrote into the header themselves -- trusting
    it would let one machine mint fresh visitor quotas at will. Ownership
    keys off the page's random device token when it sends one, because a
    phone's address changes mid-generation (wifi to cellular) and neighbours
    behind one carrier-grade NAT must not be able to poll each other's jobs.
    A bare curl with no token falls back to the address for both."""
    fwd = request.headers.get("x-forwarded-for", "")
    ip = (fwd.rsplit(",", 1)[-1].strip()
          or (request.client.host if request.client else "unknown"))
    row = db.ensure_key(f"ip:{ip}", label=f"visitor {ip}",
                        daily_limit=ANON_DAILY_LIMIT)
    if row is None:
        raise HTTPException(401, "this address has been blocked")
    quota_hash = row["key_hash"]
    own_hash = quota_hash
    if token:
        own = db.ensure_key(token, label="visitor device")
        if own is None:
            raise HTTPException(401, "this device has been blocked")
        own_hash = own["key_hash"]
    return Ident(own_hash, "visitor",
                 row["daily_limit"] or ANON_DAILY_LIMIT, quota_hash)


def auth(request: Request, authorization: str = Header(default="")) -> Ident:
    """Four bearer shapes share the endpoint: sk_ keys are API customers,
    anything shaped like a JWT is a Clerk session from the Studio page, vt_
    is the page's anonymous device token, and NO credential at all is a bare
    visitor. All of them resolve to rows in api_keys, so quota and job
    ownership downstream do not care which ran."""
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return _auth_visitor(request, "")
    if token.count(".") == 2 and not token.startswith("sk_"):
        return _auth_clerk(token)
    if token.startswith("vt_"):
        if not _VISITOR_TOKEN.match(token):
            raise HTTPException(401, "invalid or missing API key")
        return _auth_visitor(request, token)
    row = db.lookup_key(token)
    if row is None:
        raise HTTPException(401, "invalid or missing API key")
    kh = db.hash_key(token)
    return Ident(kh, "key", row.get("daily_limit") or DAILY_LIMIT, kh)


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
    #: Return 202 + a job id immediately instead of holding the connection.
    #: Browsers -- phones especially -- abort a fetch when the screen locks or
    #: the tab is backgrounded, so a 30 s held connection is unreliable there
    #: even though it is perfectly fine for curl or a server-side caller.
    #: 720p is ~2.3x the pixels of 480p and takes minutes rather than seconds,
    #: which is why it is background-only below -- no synchronous route can
    #: outlast the proxy for it. "square" stays accepted as a legacy alias for
    #: resolution=720p tier + aspect_ratio=square (960x960), the shape the
    #: first customer integration was given.
    resolution: str = Field(default=DEFAULT_RESOLUTION,
                            description="480p (default) or 720p")
    aspect_ratio: str = Field(default=DEFAULT_ASPECT,
                              description="horizontal (default), portrait, or square")
    #: A start frame, base64 (raw or a data: URI). The model is first-last-to-
    #: video, so this conditions the opening frame and the motion follows from
    #: it. Its aspect should match `resolution` or the model letterboxes.
    image: str | None = Field(default=None, description="base64 start frame")
    background: bool = Field(default=False,
                             description="return a job id immediately instead of the file")
    #: -1 (or omitted) picks a random seed; the resolved value is returned so a
    #: generation can be reproduced exactly.
    seed: int | None = Field(default=None, ge=-1, le=2**31 - 1)
    #: Expand a plain-language idea into the full H3 structured prompt before
    #: generating. Prompts already in the H3 format (they start with
    #: "integrated_multimodal_description:") are never re-enhanced, so the
    #: Studio example chips and reused gallery prompts pass through verbatim.
    enhance_prompt: bool = Field(
        default=False, description="expand a short prompt into the H3 format")


#: Wall-clock seconds per (tier, duration) a warm request takes end to end
#: (dispatch + generate + decode + transfer), measured through the gateway on
#: the PRO 6000. These are only the floor: rolling MEDIANS of recorded wall_s
#: override each cell once it has MIN_SAMPLES real completions. Medians, not
#: means -- a cold-start outlier must not poison the warm number, but if
#: reload-on-first-job becomes the common case, the median follows it.
BASE_TIMES = {"480p": {5: 30, 10: 65, 15: 110},
              "720p": {5: 55, 10: 115, 15: 200}}
#: Observed full cold start (weight download, no network volume) ~150 s.
COLD_START_S = int(os.environ.get("GATEWAY_COLD_START_S", "150"))
MIN_SAMPLES = 3

_TIER_OF_DIMS = {dims: tier for (tier, _aspect), dims in DIMENSIONS.items()}


def _median(vals: list[float]) -> float:
    return sorted(vals)[len(vals) // 2]


def _estimate_block() -> dict:
    """What the page needs to compute an honest time estimate: warm per-combo
    wall seconds, the cold-start penalty, and a typical job time for
    queue-wait math."""
    times = {tier: dict(cells) for tier, cells in BASE_TIMES.items()}
    merged: dict[tuple[str, int], list[float]] = {}
    for (dims, dur), vals in db.recent_wall_times().items():
        tier = _TIER_OF_DIMS.get(dims)
        if tier:
            # < 5 s wall for a video generation is a recording glitch, not data
            merged.setdefault((tier, dur), []).extend(v for v in vals if v >= 5)
    all_vals: list[float] = []
    for (tier, dur), vals in merged.items():
        all_vals.extend(vals)
        if len(vals) >= MIN_SAMPLES and dur in times[tier]:
            times[tier][dur] = round(_median(vals))
    avg_job = round(_median(all_vals)) if all_vals else 60
    return {"times": {t: {str(d): s for d, s in cells.items()}
                      for t, cells in times.items()},
            "cold_start_s": COLD_START_S, "avg_job_s": avg_job}


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
            "capacity": {"ready": w.get("ready", 0), "starting": w.get("initializing", 0)},
            "estimate": _estimate_block()}


@app.get("/v1/quota")
def quota(ident: Ident = Depends(auth)) -> dict:
    """What the caller has left today. Identity resolution matches
    /v1/videos exactly -- sk_ key, Clerk session, device token, or bare
    address -- so the page can display the real count instead of keeping a
    client-side guess that drifts the moment quota outlives the tab."""
    used = db.usage_today(ident.quota_hash, date.today().isoformat())
    return {"kind": ident.kind, "limit": ident.limit,
            "used": used, "remaining": max(0, ident.limit - used)}


@app.post("/v1/videos", status_code=202)
def create(body: VideoRequest, ident: Ident = Depends(auth)) -> dict:
    key = ident.key_hash
    if body.duration_s not in DURATIONS:
        raise HTTPException(422, f"duration_s must be one of {sorted(DURATIONS)}")
    tier, aspect = body.resolution, body.aspect_ratio
    if tier == "square":                     # legacy alias, pre-aspect API
        tier, aspect = "720p", "square"
    if tier not in RESOLUTION_TIERS:
        raise HTTPException(422, f"resolution must be one of {sorted(RESOLUTION_TIERS)}")
    if aspect not in ASPECTS:
        raise HTTPException(422, f"aspect_ratio must be one of {sorted(ASPECTS)}")
    # The visitor tier is a taste, not the product: everything past 5 s at
    # 480p needs an account, exactly as the Studio page's lock icons promise.
    if ident.kind == "visitor" and (body.duration_s != 5 or tier != "480p"):
        raise HTTPException(
            401, "longer lengths and 720p need a free account -- "
                 "sign in or pass an API key")
    today = date.today().isoformat()
    if not db.try_consume_quota(ident.quota_hash, today, ident.limit):
        raise HTTPException(429, f"daily limit of {ident.limit} videos reached")

    prompt = body.prompt
    if body.enhance_prompt:
        prompt = _enhance_prompt(prompt, body.duration_s)

    frames = DURATIONS[body.duration_s]
    # Only 480p at 5 or 10 s fits inside SYNC_TIMEOUT (measured ~22 s and
    # ~56 s). Everything else takes minutes; a held connection cannot outlast
    # the proxy, and losing the job id is worse than waiting -- so those are
    # forced to background rather than offered as a combination that cannot
    # work.
    background = body.background or tier != "480p" or body.duration_s > 10
    _purge_expired_cache()
    settings: dict[str, Any] = {
        "prompt": prompt,
        "resolution": DIMENSIONS[(tier, aspect)],
        "video_length": frames,
        "sample_solver": "euler",
        "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": "",
        # prompt_parser.split_prompt_units turns a multi-line prompt into one
        # prompt PER LINE unless this says the lines are a single prompt
        # ("FG"). The H3 structured caption format is multi-line by design,
        # so without the pin a caller's prompt silently becomes several
        # generations' worth of fragments.
        "multi_prompts_gen_type": "FG",
    }
    media: dict[str, Any] = {}
    if body.image:
        # wgp.py:1409 reads image_start only when "S" is in image_prompt_type,
        # so the letter and the attachment must be set together or the image is
        # silently ignored.
        settings["image_prompt_type"] = "S"
        media["image_start"] = {"b64": body.image.split(",", 1)[-1]}
    if body.seed is not None and body.seed >= 0:
        settings["seed"] = body.seed

    try:
        created = _rp("run", {"input": {
            "model_type": MODEL_TYPE, "profile": ACCEL_PROFILE,
            "settings": settings, "media": media,
            "output": {"mode": "auto"},
            "runtime": {"timeout_s": 1200},
        }})
    except Exception:
        db.refund_quota(ident.quota_hash, today)   # a failed submit must not bill the quota
        raise HTTPException(503, "could not queue the job") from None

    job_id = created.get("id")
    if not job_id:
        raise HTTPException(503, "could not queue the job")
    db.record_job(job_id, key, resolution=DIMENSIONS[(tier, aspect)],
                  duration_s=body.duration_s, seed=body.seed)

    if background:
        return JSONResponse(
            status_code=202,
            headers={"Retry-After": "10", "Location": f"/v1/videos/{job_id}"},
            content={"id": job_id, "status": "queued",
                     "poll_url": f"/v1/videos/{job_id}"})

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
            db.mark_job(job_id, "failed")
            raise HTTPException(502, message)
        db.mark_job(job_id, "completed",
                    generate_s=((st.get("output") or {}).get("metrics") or {}).get("generate_s"))
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


# ---- the adventure ---------------------------------------------------------
# A fixed choose-your-path story (gateway/story.py). Scenes are generated
# ONCE, kept forever in Postgres, and shared by every player. Two render
# lanes work through encounter order concurrently -- the story forks into
# two paths, so both branches fill in together -- and each child scene
# starts from its parent's LAST frame (the model is first-last-to-video),
# so every transition is continuous.

ADVENTURE_LANES = int(os.environ.get("GATEWAY_ADVENTURE_LANES", "2"))

_adventure_started = False


def _start_adventure_renderer() -> None:
    global _adventure_started
    if _adventure_started:
        return
    _adventure_started = True
    threading.Thread(target=_adventure_loop, daemon=True,
                     name="adventure-renderer").start()


def _adventure_loop() -> None:
    # At process start no poller can be alive, so ANY 'rendering' row is an
    # orphan of a previous process: requeue immediately rather than waiting
    # out the steady-state staleness window. (A deploy's brief old/new
    # overlap can duplicate one render -- cents -- versus a 15-minute stall
    # after every deploy.)
    for slug in story.STORIES:
        db.adventure_requeue_stale(slug, older_than_s=0)
        db.adventure_reset_broken_chain(slug)
    lanes = [threading.Thread(target=_adventure_lane, daemon=True,
                              name=f"adventure-lane-{i}")
             for i in range(max(1, ADVENTURE_LANES))]
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()


def _adventure_lane() -> None:
    while True:
        scene, slug = None, None
        for candidate in story.STORIES:
            scene = db.adventure_claim(candidate)
            if scene is not None:
                slug = candidate
                break
        if scene is None:
            # Nothing claimable: children may be waiting on a parent another
            # lane is rendering, or a dead process left a stale claim.
            for candidate in story.STORIES:
                db.adventure_requeue_stale(candidate)
            if not any(db.adventure_any_rendering(s) for s in story.STORIES):
                return                 # every scene ready (or out of retries)
            time.sleep(30)
            continue
        try:
            _adventure_render_one(scene, slug)
        except Exception as exc:  # noqa: BLE001 - one bad scene must not end the run
            db.adventure_mark(scene["id"], "failed",
                              error=f"{type(exc).__name__}: {exc}")


#: 5 s of the parent's tail conditions each child scene (the D variant of the
#: continuity A/B): the child flows out of the parent instead of cutting.
#: 362 is the EXACT shape the A/B validated end to end: one default-size
#: window where 120 frames are context and 242 (10.1 s) are new story. A
#: 481-frame request looked like "context plus a full 15 s scene" but spills
#: past the default window and killed production workers mid-job; do not
#: raise this again without proving the longer shape on a scratch endpoint.
CONTINUITY_OVERLAP_FRAMES = 120
CONTINUITY_WINDOW_FRAMES = 362


def _video_duration_s(data: bytes) -> float | None:
    import subprocess
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "v.mp4"
            src.write_bytes(data)
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
                capture_output=True, timeout=60)
            return float(proc.stdout.decode().strip())
    except Exception:  # noqa: BLE001
        return None


def _trim_lead(data: bytes, lead_s: float) -> bytes:
    """A continuation's output carries the whole parent clip in front of the
    new content; the stored scene must be only the new part. Fail-open: an
    untrimmed clip is a worse UX but a working story."""
    import subprocess
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "full.mp4", Path(td) / "scene.mp4"
            src.write_bytes(data)
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{lead_s:.6f}",
                 "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "18", "-c:a", "aac", "-movflags", "+faststart",
                 str(out)],
                capture_output=True, timeout=300)
            if proc.returncode != 0 or not out.exists():
                return data
            return out.read_bytes()
    except Exception:  # noqa: BLE001
        return data


def _adventure_render_one(scene: dict, slug: str = story.STORY_ID) -> None:
    """Render one scene to completion and store the bytes. Raises on failure;
    the lane records it and moves on (with retries via adventure_claim)."""
    sid = scene["id"]
    settings: dict[str, Any] = {
        "prompt": scene["prompt"],
        "resolution": story.SCENE_RESOLUTION,
        "video_length": DURATIONS[story.SCENE_DURATION_S],
        "sample_solver": "euler",
        "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": "",
        "multi_prompts_gen_type": "FG",
    }
    media: dict[str, Any] = {}
    trim_lead_s: float | None = None
    parent_id = story.parent_of(slug, sid)
    parent = db.adventure_video(parent_id) if parent_id is not None else None
    if parent_id is not None and parent is None:
        # The claim gate should make this unreachable; a silent fallback
        # here would bake a cut into a story sold on continuity.
        raise RuntimeError(f"parent clip for {sid} missing; not rendering a cut")
    if parent is not None:
        # Continue the parent's video: its last 5 s condition the new scene.
        settings["image_prompt_type"] = "V"
        settings["sliding_window_overlap"] = CONTINUITY_OVERLAP_FRAMES
        settings["video_length"] = CONTINUITY_WINDOW_FRAMES
        media["video_source"] = {"b64": base64.b64encode(parent).decode()}
        trim_lead_s = _video_duration_s(parent)
    created = _rp("run", {"input": {
        "model_type": MODEL_TYPE, "profile": ACCEL_PROFILE,
        "settings": settings, "media": media,
        "output": {"mode": "auto"},
        "runtime": {"timeout_s": 1800},
    }})
    job_id = created.get("id")
    if not job_id:
        raise RuntimeError("submit returned no job id")
    db.adventure_mark(sid, "rendering", job_id=job_id)
    deadline = time.monotonic() + 2200      # cold start + a 481-frame window, with slack
    while time.monotonic() < deadline:
        time.sleep(5)
        try:
            st = _rp(f"status/{job_id}", timeout=30)
        except Exception:
            continue                        # a blip mid-render is not a failure
        state = st.get("status")
        if state in ("IN_QUEUE", "IN_PROGRESS"):
            continue
        if state != "COMPLETED":
            if "fps_mode" in json.dumps(st):
                # The worker image predates video_source support (its ffmpeg
                # rejects the decoder's flag). Not this scene's fault: requeue
                # without burning an attempt and back off until the endpoint
                # runs the fixed image.
                db.adventure_mark(sid, "queued", reset_attempts=True,
                                  error="waiting for a worker image with video_source support")
                time.sleep(120)
                return
            raise RuntimeError((st.get("output") or {}).get("message", f"job {state}"))
        out = st.get("output") or {}
        video = out.get("video") or {}
        if video.get("kind") != "base64":
            raise RuntimeError("no inline video in the job output")
        data = base64.b64decode(video["data"])
        if trim_lead_s:
            # A continuation returns parent + new; store only the new scene.
            data = _trim_lead(data, trim_lead_s)
        db.adventure_mark(
            sid, "ready", seed=out.get("seed"),
            generate_s=(out.get("metrics") or {}).get("generate_s"),
            video=data)
        return
    raise RuntimeError("timed out waiting for the scene")


@app.get("/adventures", include_in_schema=False)
def adventures_home():
    """The browse page: one featured story that is real (Biscuit) and a shelf
    of coming-soon cards. Static by design -- there is one story."""
    page = STATIC / "adventures.html"
    if not page.exists():
        raise HTTPException(404, "adventures page not installed")
    return FileResponse(page, media_type="text/html")


def _story_or_404(slug: str) -> str:
    if slug not in story.STORIES:
        raise HTTPException(404, "no such story")
    return slug


@app.get("/adventure", include_in_schema=False)
def adventure_legacy():
    """The pre-slug URL. Permanent redirect so old links and bookmarks land
    on the canonical story page."""
    return RedirectResponse(f"/adventures/{story.STORY_ID}", status_code=308)


@app.get("/adventures/{slug}", include_in_schema=False)
def adventure_page(slug: str):
    """One player page serves every story: its metadata rides on template
    tokens filled from the registry, and everything else the page needs it
    fetches from the story-scoped state route."""
    _story_or_404(slug)
    page = STATIC / "adventure.html"
    if not page.exists():
        raise HTTPException(404, "adventure page not installed")
    meta = story.STORIES[slug]
    html = page.read_text()
    for token, value in (("{{SLUG}}", slug),
                         ("{{PAGE_TITLE}}", meta["page_title"]),
                         ("{{STORY_TITLE}}", meta["title"].upper()),
                         ("{{BLURB}}", meta["blurb"])):
        html = html.replace(token, value)
    return Response(content=html, media_type="text/html")


@app.get("/adventures/{slug}/poster.jpg", include_in_schema=False)
@app.get("/adventure/poster.jpg", include_in_schema=False)   # og:image legacy
def adventure_poster(slug: str = story.STORY_ID):
    """The link-preview image (og:image) social scrapers fetch. Shipped art
    when the repo has it; otherwise a frame pulled from the story's opening
    scene once that has rendered."""
    _story_or_404(slug)
    for name in (f"adventure-poster-{slug}.jpg",
                 "adventure-poster.jpg" if slug == story.STORY_ID else ""):
        if name and (STATIC / name).exists():
            return FileResponse(STATIC / name, media_type="image/jpeg",
                                headers={"Cache-Control": "public, max-age=86400"})
    cached = CACHE / f"poster-{slug}.jpg"
    if not cached.exists():
        opening = db.adventure_video(story.order_of(slug)[0])
        if opening is None:
            raise HTTPException(404, "poster not available yet")
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "n0.mp4"
            src.write_bytes(opening)
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", "8", "-i", str(src),
                 "-frames:v", "1", "-update", "1", "-q:v", "2", str(cached)],
                capture_output=True, timeout=60)
            if proc.returncode != 0 or not cached.exists():
                raise HTTPException(404, "poster not available yet")
    return FileResponse(cached, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/adventures/{slug}/state")
@app.get("/adventure/state")                       # legacy alias
def adventure_state(slug: str = story.STORY_ID) -> dict:
    """The story tree (no prompts) with each scene's live render status --
    the page builds the branch map from this and polls it while scenes are
    still rendering. Public: the story is shared by everyone."""
    _story_or_404(slug)
    statuses = db.adventure_status(slug)
    nodes = []
    for node in story.public_tree(slug):
        row = statuses.get(node["id"], {})
        nodes.append({**node, "status": row.get("status", "queued"),
                      "seed": row.get("seed")})
    return {"story": slug,
            "scene_s": story.SCENE_DURATION_S, "nodes": nodes}


class WaitlistRequest(BaseModel):
    email: str = Field(..., max_length=254)
    #: Which surface collected it ("adventures-header", "adventure-end", ...)
    #: -- future stories will want to know which pitch converted.
    source: str = Field("adventures", max_length=40)


#: Best-effort spam brake for an unauthenticated endpoint: per address, per
#: process, resets on restart. The email primary key already bounds real
#: damage; this just keeps one script from hammering the table.
_waitlist_seen: dict[str, int] = {}
_WAITLIST_IP_CAP = 20


@app.post("/adventure/waitlist", status_code=204)
def adventure_waitlist(body: WaitlistRequest, request: Request) -> None:
    email = body.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]{2,}", email):
        raise HTTPException(422, "that does not look like an email address")
    fwd = request.headers.get("x-forwarded-for", "")
    ip = (fwd.rsplit(",", 1)[-1].strip()
          or (request.client.host if request.client else "unknown"))
    if _waitlist_seen.get(ip, 0) >= _WAITLIST_IP_CAP:
        return                            # quietly full: same 204 the page expects
    _waitlist_seen[ip] = _waitlist_seen.get(ip, 0) + 1
    db.waitlist_add(email, body.source)


@app.get("/adventures/{slug}/scene/{scene_id}")
@app.get("/adventure/scene/{scene_id}")            # legacy alias
def adventure_scene(scene_id: str, slug: str = story.STORY_ID):
    _story_or_404(slug)
    if scene_id not in story.nodes_of(slug):
        raise HTTPException(404, "no such scene")
    data = db.adventure_video(scene_id)
    if data is None:
        raise HTTPException(404, "scene not rendered yet")
    # Immutable once rendered: let the browser cache aggressively.
    return Response(content=data, media_type="video/mp4",
                    headers={"Cache-Control": "public, max-age=86400"})


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


def _owned(job_id: str, key: str) -> None:
    # 404 rather than 403 for someone else's job: a customer should not be able
    # to probe which ids exist. Ownership lives in the database, so it survives
    # a gateway restart -- with the in-memory dict, a restart mid-generation
    # orphaned every paid job.
    if not db.job_owned_by(job_id, key):
        raise HTTPException(404, "no such job")


@app.get("/v1/videos/{job_id}")
def status(job_id: str, ident: Ident = Depends(auth)) -> dict:
    _owned(job_id, ident.key_hash)
    try:
        st = _rp(f"status/{job_id}", timeout=30)
    except Exception:
        raise HTTPException(503, "generation backend unavailable") from None

    state = st.get("status")
    if state in ("IN_QUEUE", "IN_PROGRESS"):
        body: dict[str, Any] = {
            "id": job_id, "status": "queued" if state == "IN_QUEUE" else "processing"}
        # The worker pushes progress frames (phase, step, eta) through
        # RunPod's progress API; forward a sanitized copy so the page can say
        # what the generation is actually doing instead of a bare "processing".
        prog = st.get("output")
        if state == "IN_PROGRESS" and isinstance(prog, dict) and prog.get("phase"):
            def _num(v: Any) -> float | int | None:
                return v if isinstance(v, (int, float)) else None
            body["progress"] = {
                "phase": str(prog.get("phase", ""))[:60],
                "status": str(prog.get("status", ""))[:300],
                "pct": _num(prog.get("pct")),
                "step": _num(prog.get("step")),
                "total_steps": _num(prog.get("total_steps")),
                "eta_s": _num(prog.get("eta_s")),
            }
            # A small denoising preview JPEG the worker encodes from the
            # latents -- a partial image beats a blank stage. Size-capped so
            # a misbehaving worker cannot balloon the poll responses.
            preview = prog.get("preview_jpeg")
            if isinstance(preview, str) and 0 < len(preview) <= 300_000:
                body["progress"]["preview_jpeg"] = preview
        return body
    if state != "COMPLETED":
        db.mark_job(job_id, "failed")
        return {"id": job_id, "status": "failed",
                "error": (st.get("output") or {}).get("message", "generation failed")}

    out = st.get("output") or {}
    video = out.get("video") or {}
    # generate_s is the number per-second billing computes from; capture it at
    # the moment the completion is observed rather than hoping to re-fetch it.
    db.mark_job(job_id, "completed", generate_s=(out.get("metrics") or {}).get("generate_s"))
    path = _materialise(job_id, st) or CACHE / f"{job_id}.mp4"
    return {"id": job_id, "status": "completed",
            "seed": out.get("seed"),
            "duration_s": video.get("duration_s"),
            "width": video.get("width"), "height": video.get("height"),
            "has_audio": video.get("has_audio"),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "content_url": f"/v1/videos/{job_id}/content"}


@app.get("/v1/videos/{job_id}/content")
def content(job_id: str, ident: Ident = Depends(auth)):
    _owned(job_id, ident.key_hash)
    path = CACHE / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "video not ready; poll /v1/videos/{id} first")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{job_id}.mp4")
