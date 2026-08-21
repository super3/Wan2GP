# The gateway is pure stdlib + FastAPI: no torch, no CUDA, no model weights. It
# runs on the cheapest CPU box you have; the GPU cost is entirely on the RunPod
# endpoint behind it.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID stay server-side and are never returned
# to a caller. GATEWAY_KEYS maps customer keys to labels.
ENV GATEWAY_CACHE=/data/videos GATEWAY_DAILY_LIMIT=100
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/v1/health',timeout=4)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
