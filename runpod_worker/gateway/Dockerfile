# The gateway is pure stdlib + FastAPI: no torch, no CUDA, no model weights. It
# runs on the cheapest CPU box you have; the GPU cost is entirely on the RunPod
# endpoint behind it.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# db.py and static/ ship with the app; the old boot-script deploys curl'd
# app.py alone, which is why the missing static/ copy never surfaced.
COPY app.py db.py story.py ./
COPY static ./static

# RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID stay server-side and are never returned
# to a caller. GATEWAY_KEYS maps customer keys to labels.
# No VOLUME instruction: Railway rejects it at Dockerfile validation ("docker
# VOLUME at Line 18 is not supported, use Railway Volumes") and the app mkdirs
# GATEWAY_CACHE at import anyway. For durable video cache on Railway, attach a
# Railway Volume mounted at /data; for docker run, pass -v as usual.
ENV GATEWAY_CACHE=/data/videos GATEWAY_DAILY_LIMIT=100
EXPOSE 8000

# /healthz is process-liveness only. /v1/health reports the GPU backend and
# returns 503 when RunPod is down; probing THAT turns a backend outage into a
# gateway restart loop.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'),timeout=4)" || exit 1

# Railway injects PORT; default keeps local `docker run -p 8000:8000` working.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
