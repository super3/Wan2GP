#!/bin/bash
# Boot script for the MiniMax H3 benchmarks on a rented GPU box (Ubuntu + NVIDIA driver >= 580, run as root): host checks,
# WanGP install, one benchmark, then idle. h3_bench_runpod.sh ships it to a Runpod pod; on another provider run it as the
# container's start command. See docs/H3_BENCHMARKS.md.
#
# Settings (environment):
#   H3_BENCH                 usp (default): h3_bench_usp.py, 2+ GPUs | models: h3_bench_models.py | decode: h3_bench_decode.py
#   H3_BENCH_REPO            git repository (default https://github.com/super3/Wan2GP)
#   H3_BENCH_REV             branch or commit with the harnesses and the USP plugin (default feature/usp-multi-gpu-rebased)
#   H3_BENCH_STREAMING_REV   branch or commit whose plugins/chunk_streaming is used (default feature/multi-gpu-decode;
#                            feature/chunk-streaming-rebased has streaming without the multi-GPU decode)
#   H3_BENCH_ARGS            WanGP flags (default "--attention sage2 --profile 1 --verbose 1")
#   H3_BENCH_REQUIRE_NODE    1 (default with 2+ GPUs): stop unless GPU 0 and 1 share a socket (NODE, PXB, PIX or NVLink)
#   H3_BENCH_MIN_P2P_GBS     stop when P2P is off or a GPU 0 -> 1 copy is slower than this (default 45; 0: no check)
#   plus the harnesses' own variables (H3_BENCH_MODES, H3_BENCH_COMBOS, H3_BENCH_SINGLE, ...), passed through.
#
# Everything goes to /root/report/status.log, served read-only on port 8888. The log ends with BENCH_DONE or BENCH_FAILED
# <reason>; the renter (h3_bench_runpod.sh) polls it and deletes the box then. Results: the RESULT lines, which
# h3_bench_tables.py turns into tables.
set -uo pipefail
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 PYTHONUNBUFFERED=1
export PATH="/root/venv/bin:$HOME/.local/bin:$PATH" VIRTUAL_ENV=/root/venv
H3_BENCH=${H3_BENCH:-usp}
REPO=${H3_BENCH_REPO:-https://github.com/super3/Wan2GP}
REV=${H3_BENCH_REV:-feature/usp-multi-gpu-rebased}
STREAMING_REV=${H3_BENCH_STREAMING_REV:-feature/multi-gpu-decode}
ARGS=${H3_BENCH_ARGS:---attention sage2 --profile 1 --verbose 1}
MIN_P2P=${H3_BENCH_MIN_P2P_GBS:-45}
W=/root/Wan2GP
REPORT=/root/report
idle() { exec sleep infinity; }
fail() { log "BENCH_FAILED $*"; touch /root/.bench_done; idle; }
serve() { pgrep -f "http.server 8888" >/dev/null || (cd $REPORT && nohup python3 -m http.server 8888 --bind 0.0.0.0 >/dev/null 2>&1 &); }
retry() { for i in 1 2 3 4 5; do "$@" && return 0; log "retry $i: $1 $2 $3"; sleep $((i*15)); done; return 1; }
if [ -f /root/.bench_done ]; then log "ALREADY_DONE"; serve; idle; fi

command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 >/dev/null; }
mkdir -p $REPORT; serve; log "SERVE $(command -v python3)"
GPUS=$(nvidia-smi -L | wc -l)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
log "BOOT $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader | tr '\n' ';') gpus=$GPUS bench=$H3_BENCH rev=$REV streaming=$STREAMING_REV"
nproc; free -g | head -2; df -h /root | tail -1
[ "${DRV%%.*}" -ge 580 ] 2>/dev/null || fail "driver $DRV < 580 (the stack is CUDA 13)"

# -- host checks, before any install: a cross-socket pair slows USP's per-layer exchange ---------------------------------
log "STAGE link"
nvidia-smi topo -m | head -4 | sed 's/^/TOPO /'
if [ "$GPUS" -ge 2 ]; then
  TOPO=$(nvidia-smi topo -m | awk '$1=="GPU0"{print $3}')
  log "LINK gpu0_gpu1=$TOPO"
  if [ "${H3_BENCH_REQUIRE_NODE:-1}" = 1 ]; then
    case "$TOPO" in NV*|PIX|PXB|NODE) ;; *) fail "topology $TOPO: GPUs not on one socket (want NODE or better)";; esac
  fi
fi
( while sleep 60; do
    echo "[$(date -u +%H:%M:%S)] HEARTBEAT ckpts=$(du -sh $W/ckpts 2>/dev/null | cut -f1) mem=$(free -g | awk '/Mem:/{print $3"/"$2"G"}') gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | tr '\n' ';')"
  done ) &

# -- install ---------------------------------------------------------------------------------------------------------------
log "STAGE setup"
# images such as nvidia/cuda ship a CUDA apt repository: not needed (torch brings its CUDA), and slow on some hosts
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list
APT="-o Acquire::http::Timeout=30 -o Acquire::Retries=3"
retry apt-get $APT update -qq && retry apt-get $APT install -y -qq --no-install-recommends git ffmpeg build-essential curl ca-certificates libgl1 libglib2.0-0 python3 >/dev/null || fail apt
log "SETUP apt done"
command -v uv >/dev/null || retry bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1" || fail uv
if [ ! -d $W/.git ]; then
  retry git clone -q "$REPO" $W || fail clone
  cd $W && retry git fetch -q origin "$REV" && git checkout -q FETCH_HEAD || fail "checkout $REV"
  # plugins/chunk_streaming from its own branch (streaming instrumentation, and the multi-GPU decode worker)
  retry git fetch -q origin "$STREAMING_REV" && git checkout -q FETCH_HEAD -- plugins/chunk_streaming || fail "chunk_streaming from $STREAMING_REV"
fi
cd $W && mkdir -p ckpts loras
log "CODE $(git rev-parse --short HEAD) chunk_streaming from $STREAMING_REV: $(ls plugins/chunk_streaming | tr '\n' ' ')"
[ -x /root/venv/bin/python ] || uv venv -q -p 3.11 /root/venv || fail venv
log "SETUP clone+venv done"
retry uv pip install -q torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 || fail torch
log "SETUP torch done"
retry uv pip install -q -r requirements.txt --index-strategy unsafe-best-match || fail requirements
log "SETUP requirements done"
retry uv pip install -q "triton>=3.6,<3.7" || fail triton
retry uv pip install -q https://github.com/deepbeepmeep/SageAttention/releases/download/v2.2.0-linux.cg1/sageattention-2.2.0+cu130torch2.10.0.cg1-cp311-cp311-linux_x86_64.whl || fail sage
retry uv pip install -q --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.3/wgpkernels-1.4.3%2Bcu130-py3-none-linux_x86_64.whl || fail wgpkernels
retry uv pip install -q https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl || fail light2xv
python -c "import torch; print('torch', torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))" || fail torch-check

# -- GPU-to-GPU copy rate ----------------------------------------------------------------------------------------------------
if [ "$GPUS" -ge 2 ] && [ "$MIN_P2P" != 0 ]; then
  log "STAGE p2p"
  MIN_P2P=$MIN_P2P python - <<'PY' || fail "GPU-to-GPU link: no P2P, or under $MIN_P2P GB/s"
import os, time, torch
ok = torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)
a = torch.empty(1 << 30, dtype=torch.uint8, device="cuda:0"); b = torch.empty_like(a, device="cuda:1")
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(0); torch.cuda.synchronize(1); t = time.perf_counter(); n = 20
for _ in range(n): b.copy_(a)
torch.cuda.synchronize(0); torch.cuda.synchronize(1); gbs = n * a.numel() / (time.perf_counter() - t) / 1e9
print(f"P2PBW p2p={'yes' if ok else 'no'} gpu0->gpu1 {gbs:.0f} GB/s", flush=True)
raise SystemExit(0 if ok and gbs >= float(os.environ["MIN_P2P"]) else 3)
PY
fi

# -- the benchmark -------------------------------------------------------------------------------------------------------------
FILTER="^RESULT|Traceback|Error:|out of memory|NCCL WARN|Chunk Streaming|decode worker|Killed"
log "STAGE bench-$H3_BENCH"
cd $W
case "$H3_BENCH" in
  usp)    python runpod_worker/scripts/h3_bench_usp.py $ARGS 2>&1 | grep --line-buffered -E "$FILTER" | stdbuf -oL cut -c1-6000 ;;
  models) python runpod_worker/scripts/h3_bench_models.py $ARGS 2>&1 | grep --line-buffered -E "$FILTER|^GPU[01]\|" | stdbuf -oL cut -c1-6000 ;;
  decode) retry python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('DeepBeepMeep/MiniMax-H3', 'minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors', local_dir='$W/ckpts'))" || fail vae-download
          python runpod_worker/scripts/h3_bench_decode.py $ARGS 2>&1 | grep --line-buffered -E "$FILTER|error:|multi-device" | stdbuf -oL cut -c1-6000 ;;
  *)      fail "unknown H3_BENCH=$H3_BENCH" ;;
esac
log "STAGE bench-done"
log "BENCH_DONE"
touch /root/.bench_done
idle
