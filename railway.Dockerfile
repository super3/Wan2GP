# Railway build for the customer gateway, written for a REPO-ROOT build
# context. This exists because Railway's zero-config path builds from the repo
# root: without it, Railway finds the root Dockerfile -- WanGP's multi-GB CUDA
# image -- and the deployment fails. runpod_worker/gateway/Dockerfile is the
# same image for a gateway-rooted context (RunPod, local docker); keep the two
# in sync when either changes.
#
# The gateway itself: FastAPI + SQLAlchemy, no GPU, no torch, no weights.
FROM python:3.12-slim

# ffmpeg: the adventure renderer extracts each parent clip's last frame
# as the child scene's start image (FL2V continuity).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY runpod_worker/gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runpod_worker/gateway/app.py runpod_worker/gateway/db.py runpod_worker/gateway/story.py ./
COPY runpod_worker/gateway/static ./static

# RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, GATEWAY_KEYS and DATABASE_URL come from
# Railway variables; none are baked here.
ENV GATEWAY_CACHE=/data/videos GATEWAY_DAILY_LIMIT=100
EXPOSE 8000

# /healthz is process liveness only. /v1/health reports the GPU backend and
# 503s when RunPod is down; probing that would turn a backend outage into a
# gateway restart loop.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
