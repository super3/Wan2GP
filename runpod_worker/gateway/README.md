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

uvicorn runpod_worker.gateway.app:app --host 0.0.0.0 --port 8000
# or: docker build -t video-api runpod_worker/gateway && docker run -p 8000:8000 --env-file .env video-api
```

No GPU. It is stdlib + FastAPI, so it runs on the cheapest CPU box you have.

## The API

Interactive docs at `/docs`.

```bash
# submit
curl -X POST https://api.example.com/v1/videos \
  -H "Authorization: Bearer sk_live_abc123" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "integrated_multimodal_description: [Shot 1] ...\noverall_soundscape: ...\nnon_diegetic_music: None."}'
# -> {"id":"...","status":"queued","duration_s":10.12,"resolution":"832x480"}

# poll
curl https://api.example.com/v1/videos/$ID -H "Authorization: Bearer sk_live_abc123"
# -> {"status":"queued"|"processing"|"completed"|"failed", ...}

# download
curl https://api.example.com/v1/videos/$ID/content \
  -H "Authorization: Bearer sk_live_abc123" -o out.mp4
```

`seed` is optional; the resolved value comes back on completion so a generation
can be reproduced exactly.

## What a customer should expect

| | |
|---|---|
| output | 10.125 s, 832x480, H.264 + AAC, ~2 MB |
| warm generation | ~56 s (measured mean, n=5) |
| cold start | +90–330 s of queue while a worker boots |
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
