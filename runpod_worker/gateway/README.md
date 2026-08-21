# Video Generation API — customer gateway

A thin HTTP service in front of one RunPod endpoint. Customers get keys this
service issues; the RunPod key never leaves the server.

## Why this exists

A RunPod API key is **account-wide**. There is no per-endpoint key, so a
customer holding one could list, modify and delete every pod and endpoint on the
account. The gateway is the smallest thing that prevents that, and it also pins
the product: the caller picks a prompt, and everything that costs money — clip
length, resolution, model, accelerator profile — is fixed server-side.

## Run it

```bash
export RUNPOD_API_KEY=rpa_...            # never returned to a caller
export RUNPOD_ENDPOINT_ID=59kne66vo58331
export GATEWAY_KEYS='{"sk_live_abc123":"acme corp"}'
export GATEWAY_DAILY_LIMIT=100           # per key, per day
export DATABASE_URL=postgresql://...     # omit locally -> SQLite in /tmp

uvicorn runpod_worker.gateway.app:app --host 0.0.0.0 --port 8000
# or: docker build -t video-api runpod_worker/gateway && docker run -p 8000:8000 --env-file .env video-api
```

No GPU. FastAPI + SQLAlchemy; runs on the cheapest CPU box you have.

## Deploy on Railway

The control plane belongs on a PaaS, not on GPU infra: the RunPod-pod era of
this gateway burned through seven URLs in one night (pods bake their boot
script in and cannot reliably restart after a patch), had no readable logs,
and kept every key, quota and job in process memory. Railway fixes all four:
stable domain, git deploys, visible logs, and a Postgres addon for the state.

1. New project -> **Deploy from GitHub repo** -> pick this repo, and set the
   deploy branch (any branch works; it does not have to be main).
2. Service settings -> **Root Directory** = `runpod_worker/gateway`. Railway
   finds the Dockerfile there.
3. Add the **Postgres** addon; Railway injects `DATABASE_URL` automatically.
   Without it the gateway falls back to SQLite in the container, which is
   EPHEMERAL on Railway -- fine for a smoke test, wrong for billing.
4. Variables: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `GATEWAY_KEYS` (seeds
   the DB at startup; the DB is the authority afterwards -- revoking a key in
   the DB beats its presence in the env), `GATEWAY_DAILY_LIMIT`.
5. Health check path: `/healthz`. NOT `/v1/health` -- that one reports the GPU
   backend and returns 503 when RunPod is down, which would turn a backend
   outage into a gateway restart loop.
6. Before telling customers the synchronous route works: measure Railway's
   edge timeout against `GATEWAY_SYNC_TIMEOUT` (90 s). RunPod's proxy cut at
   ~100 s; if Railway cuts sooner, lower the gateway value to match. The 202
   fallback makes this safe to get wrong, but the error message is uglier.

State lives in three tables: `api_keys` (SHA-256 hashes only -- a DB dump is
not a credential dump), `usage` (per key per day), and `jobs` (ownership plus
`generate_s`, the number per-second billing computes from).

## The API

One call in, an mp4 out. Interactive docs at `/docs`.

```bash
curl -X POST https://api.example.com/v1/videos \
  -H "Authorization: Bearer sk_live_abc123" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "integrated_multimodal_description: [Shot 1] ...\noverall_soundscape: ...\nnon_diegetic_music: None."}' \
  -o out.mp4 --max-time 300
```

That is the whole integration. The response body is the video; the metadata a
caller needs rides on headers, because the body has to be the file:

```
X-Video-Id  X-Seed  X-Duration-Seconds  X-Width  X-Height
X-Has-Audio  X-Generate-Seconds
```

`seed` is optional in the request; `X-Seed` returns the resolved value so a
generation can be reproduced exactly.

### The one case that is not synchronous

A warm generation is ~56 s, comfortably inside a normal HTTP timeout. A **cold
start** adds 90–330 s of queue while a worker boots, and fits inside no sane
timeout. Rather than failing and discarding work already paid for, the call
returns **202** with a job id:

```json
{"id": "...", "status": "processing", "poll_url": "/v1/videos/..."}
```

with `Location` and `Retry-After: 30`. Poll `GET /v1/videos/{id}` until
`completed`, then `GET /v1/videos/{id}/content`. Clients should handle 202;
in steady use they will rarely see it.

`GATEWAY_SYNC_TIMEOUT` (default **90 s**) is how long the connection is held.
**It must sit below whatever proxy fronts the service**, or the proxy cuts first
and the caller gets an opaque gateway error instead of the clean 202 — the 202
never gets sent at all.

Common ceilings are low: **Cloudflare cuts at 100 s**, AWS ALB and nginx default
to 60 s. RunPod's own `*.proxy.runpod.net` is Cloudflare-fronted, so a gateway
deployed on a RunPod pod is already behind that limit — measured with a 240 s
value there, a slow generation returned **HTTP 524 at 125 s**. Behind nginx or
an ALB at their defaults, drop this to ~50 s.

The GPU job keeps running when the connection is cut, so nothing is lost — but
the caller has no id to poll, which is precisely what the 202 exists to prevent.

## What a customer should expect

| | |
|---|---|
| output | 10.125 s, 832x480, H.264 + AAC, ~2 MB |
| warm generation | ~56 s (measured mean, n=5), returned inline |
| cold start | +90–330 s of queue; returns 202 + job id instead |
| audio | generated with the video, synchronized |

The endpoint scales to zero, so **the first request after an idle period pays a
cold start**. Tell customers to expect 2–6 minutes end to end for an occasional
request, and under 90 s when they are generating steadily. If that is not
acceptable, `workersMin: 1` keeps one warm — at full hourly rate, around
$1,500/month on a PRO 6000 Server Edition.

## Prompt format

MiniMax H3 wants three labelled blocks; see the model guide. Prompts that ignore
this still generate, but audio quality drops sharply.

## Limits and honest caveats

- **Single instance.** The job index is in memory and videos cache to local
  disk, so this is not multi-replica safe as written. Restarting loses the job
  index (videos already on disk survive).
- **Keys are env-configured.** Revoking one means editing `GATEWAY_KEYS` and
  restarting. A database is the obvious next step, not a rewrite.
- **The daily limit is per key, per calendar day, in memory** — it resets on
  restart. It is a spend guard, not billing.
- **No webhooks yet.** Customers poll. `runtime.webhook` is supported by the
  worker underneath and would be the next thing to expose.

## Cost to serve

~$0.032 per 10-second clip on a PRO 6000 at $2.09/hr, warm and batched
(~31 clips per dollar). A cold start costs about 7x that, so batching matters
more than any other tuning. Cheaper cards are proportionally cheaper — an A40 is
about 4.75x less per clip and gives up ~6% speed.
