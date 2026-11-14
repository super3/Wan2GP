#!/bin/bash
# Run one MiniMax H3 benchmark on a Runpod pod, then delete the pod. See docs/H3_BENCHMARKS.md.
#
#   RUNPOD_API_KEY=... runpod_worker/scripts/h3_bench_runpod.sh [NAME=value ...]
#
# NAME=value pairs go to the pod's boot script (h3_bench_boot.sh) and the harness, e.g. H3_BENCH=usp H3_BENCH_MODES=mgd,usp
# H3_BENCH_COMBOS=3,1 H3_BENCH_ARGS="--attention sage2 --profile 1 --verbose 1". Only H3_BENCH* names are accepted.
#
# Launcher settings (environment):
#   H3_GPU_TYPE      Runpod GPU type id (default "NVIDIA RTX PRO 6000 Blackwell Server Edition")
#   H3_GPU_COUNT     GPUs per pod (default 2)
#   H3_DATA_CENTERS  comma-separated Runpod data center ids (default empty: any; e.g. US-KS-2)
#   H3_TRIES         pods to try until one passes the host checks (GPUs on one socket, P2P rate) (default 6)
#   H3_MAX_MIN       delete the pod after this many minutes whatever happens (default 150)
#   H3_DOWN_S        delete the pod when its status log has been unreachable this long (default 900)
#   H3_POLL_S        seconds between polls (default 60)
#   H3_LOG_DIR       where each pod's status log is saved (default outputs/h3_bench/pods)
#
# The API key is only read from RUNPOD_API_KEY and handed to curl on a file descriptor: it is never printed, written to a
# file or put on a command line. Every pod this script creates is deleted when the benchmark ends, fails, hits the time cap,
# goes silent, or when this script is interrupted.
set -uo pipefail
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (a Runpod API key with pod read/write)}"
HERE=$(cd "$(dirname "$0")" && pwd)
API=https://rest.runpod.io/v1
GPU_TYPE=${H3_GPU_TYPE:-NVIDIA RTX PRO 6000 Blackwell Server Edition}
GPU_COUNT=${H3_GPU_COUNT:-2}
TRIES=${H3_TRIES:-6}
MAX_MIN=${H3_MAX_MIN:-150}
DOWN_S=${H3_DOWN_S:-900}
POLL_S=${H3_POLL_S:-60}
LOG_DIR=${H3_LOG_DIR:-outputs/h3_bench/pods}
mkdir -p "$LOG_DIR"
for pair in "$@"; do
  case "$pair" in H3_BENCH*=*) ;; *) echo "not a H3_BENCH* setting: $pair" >&2; exit 2;; esac
done

auth() { printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY"; }
api() { curl -sS --max-time 60 -H @<(auth) -H "Content-Type: application/json" "$@"; }

# The pod spec: the boot script and the settings, packed into the container's start command (no image of our own).
SPEC=$(python3 - "$HERE/h3_bench_boot.sh" "$GPU_TYPE" "$GPU_COUNT" "${H3_DATA_CENTERS:-}" "$@" <<'PY'
import base64, io, json, shlex, sys, tarfile
boot, gpu_type, gpu_count, centers, *pairs = sys.argv[1:]
env = "".join(f"export {name}={shlex.quote(value)}\n" for name, _, value in (pair.partition("=") for pair in pairs))
payload = io.BytesIO()
with tarfile.open(fileobj=payload, mode="w:gz") as tar:
    for name, data in (("boot.sh", open(boot, "rb").read()), ("bench_env.sh", env.encode())):
        info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o755
        tar.addfile(info, io.BytesIO(data))
packed = base64.b64encode(payload.getvalue()).decode()
start = (f"mkdir -p /root/report && echo {packed} | base64 -d | tar xz -C /root && "
         "(set -a; . /root/bench_env.sh; set +a; nohup bash /root/boot.sh >> /root/report/status.log 2>&1 &); sleep infinity")
spec = {"name": "h3-bench", "imageName": "nvidia/cuda:12.8.1-base-ubuntu24.04", "gpuTypeIds": [gpu_type],
        "gpuCount": int(gpu_count), "cloudType": "SECURE", "minRAMPerGPU": 96, "minVCPUPerGPU": 4, "containerDiskInGb": 300,
        "ports": ["8888/http"], "interruptible": False, "dockerStartCmd": ["bash", "-c", start]}
if centers:
    spec["dataCenterIds"] = [c.strip() for c in centers.split(",") if c.strip()]
print(json.dumps(spec))
PY
) || exit 1

POD=""
delete_pod() {
  [ -n "$POD" ] || return 0
  for i in 1 2 3 4; do
    code=$(curl -sS --max-time 30 -o /dev/null -w "%{http_code}" -X DELETE -H @<(auth) "$API/pods/$POD")
    echo "DELETE $POD ($1): HTTP $code"
    case "$code" in 200|204|404) POD=""; return 0;; esac
    sleep 5
  done
}
trap 'delete_pod interrupted; exit 130' INT TERM

# Poll one pod to its end. Returns 0 when it finished, 3 when it failed a host check (try another pod), 1 otherwise.
watch_pod() {
  local log="$LOG_DIR/$POD.log" seen="$LOG_DIR/$POD.seen" url="https://$POD-8888.proxy.runpod.net/status.log"
  local t0 last_ok text elapsed state
  t0=$(date +%s); last_ok=$t0; : > "$seen"
  while true; do
    text=$(curl -sS --max-time 30 "$url" 2>/dev/null)
    if grep -q -E 'SERVE|BOOT' <<<"$text"; then last_ok=$(date +%s); printf '%s\n' "$text" > "$log"; fi
    grep -a -E 'BOOT|STAGE|SETUP|LINK|P2PBW|CODE |RESULT (cold|prewarm|usp_start|usp_broken|usp_not_started|FAILED|DONE)|BENCH_|Traceback|Error:|out of memory|Killed' <<<"$text" |
      while IFS= read -r line; do grep -qxF -- "$line" "$seen" || { echo "$line" >> "$seen"; echo "POD: ${line:0:300}"; }; done
    elapsed=$(( ($(date +%s) - t0) / 60 ))
    state=$(api "$API/pods/$POD" | python3 -I -c 'import json,sys
try: d=json.load(sys.stdin); print(d.get("desiredStatus"), "$%s/h" % d.get("costPerHr"))
except Exception: print("unknown")')
    echo "STATUS $(date -u +%H:%M:%S) +${elapsed}m $POD $state $(grep -a HEARTBEAT <<<"$text" | tail -1 | cut -c1-160)"
    if grep -a -q -E 'BENCH_FAILED (topology|GPU-to-GPU link)' <<<"$text"; then delete_pod "host check"; return 3; fi
    if grep -a -q -E 'BENCH_DONE|BENCH_FAILED|ALREADY_DONE' <<<"$text"; then
      delete_pod "$(grep -a -o -E 'BENCH_(DONE|FAILED).*' <<<"$text" | tail -1 | cut -c1-120)"
      grep -a -q BENCH_DONE <<<"$text"; return $?
    fi
    [ "$elapsed" -ge "$MAX_MIN" ] && { delete_pod "time cap ${MAX_MIN}m"; return 1; }
    [ $(( $(date +%s) - last_ok )) -ge "$DOWN_S" ] && { delete_pod "status log unreachable ${DOWN_S}s"; return 1; }
    sleep "$POLL_S"
  done
}

for attempt in $(seq 1 "$TRIES"); do
  POD=$(api -X POST "$API/pods" -d "$SPEC" | python3 -I -c 'import json,sys
d=json.load(sys.stdin)
print(d["id"] if isinstance(d, dict) and d.get("id") else "")' 2>/dev/null)
  if [ -z "$POD" ]; then echo "attempt $attempt: no pod created (no stock, or the request was refused); retrying in 60 s"; sleep 60; continue; fi
  echo "attempt $attempt: pod $POD"
  watch_pod; result=$?
  case $result in
    0) echo "DONE: log in $LOG_DIR/$POD.log; tables: python3 $HERE/h3_bench_tables.py $LOG_DIR/$POD.log"; exit 0;;
    3) echo "pod failed a host check (see $LOG_DIR/$POD.log): trying another";;
    *) echo "FAILED: log in $LOG_DIR/$POD.log"; exit 1;;
  esac
done
echo "no pod passed the host checks in $TRIES attempts"
exit 1
