# =============================================================================
#  WanGP / MiniMax H3 — RunPod Serverless worker image
#
#  BUILD FROM THE REPO ROOT (the context is the whole repo, not runpod_worker/):
#
#    docker build --platform linux/amd64 \
#      --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0" \
#      -f runpod_worker/Dockerfile -t you/wangp-h3:2026.08.18-1 .
#
#  * --platform linux/amd64 is MANDATORY. RunPod rejects ARM images.
#  * NEVER tag :latest. RunPod caches images per host and a mutable tag silently
#    serves stale code to warm workers. Use YYYY.MM.DD-N.
#  * First build is 40-90 min, nearly all of it SageAttention's nvcc passes.
#
#  Relationship to the repo's own Dockerfile: same base image and same torch
#  triple (Dockerfile:1, Dockerfile:40), and requirements.txt is consumed
#  UNMODIFIED. The three deliberate differences:
#    1. Two stages, so nvcc/ninja build output does not ride in the final image.
#    2. The application is COPYed IN. The repo's image ships no app code at all
#       -- it bind-mounts $(pwd):/workspace at run time (see run-docker-cuda-deb.sh),
#       which a serverless worker cannot do.
#    3. NO ENTRYPOINT. See the note above CMD.
#
#  WanGP disclosure (docs/API.md:9, LICENSE.txt:316): any product that
#  integrates WanGP must clearly disclose that it uses WanGP, in both its user
#  interface and its documentation. That obligation rides with this image.
# =============================================================================

# -----------------------------------------------------------------------------
# stage 1 — compile SageAttention (needs nvcc; there is NO GPU at build time)
# -----------------------------------------------------------------------------
# CUDA_ARCHITECTURES: the repo default "8.0;8.6" (Dockerfile:20) excludes L4 /
# L40S / RTX 4090 (SM89) and H100 (SM90) -- precisely the fleet a serverless
# endpoint schedules on. is_sage2_supported() (shared/sage2_core.py) checks the
# *device* capability at run time, not which arches the wheel was built for, so
# a wheel missing the host's SM fails at the first attention kernel launch,
# minutes into a billed job. Drop 8.0 only if A100 is out of your GPU priority
# list; every arch you add costs build minutes and wheel size, not run time.
ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0"
ARG SAGEATTENTION_REF="main"

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder
ARG CUDA_ARCHITECTURES
ARG SAGEATTENTION_REF
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES} \
    FORCE_CUDA=1 \
    MAX_JOBS=8 \
    CUDA_HOME=/usr/local/cuda \
    PIP_ROOT_USER_ACTION=ignore

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# setuptools/wheel/packaging are build-time imports of SageAttention's setup.py
# and must exist in the environment because we build with --no-build-isolation
# (torch cannot be seen from an isolated build env).
# hadolint ignore=DL3013,DL3059
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# The torch here must match the torch in the runtime stage exactly: a C++
# extension is compiled against a specific torch ABI.
# hadolint ignore=DL3059
RUN pip install --no-cache-dir torch==2.10.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

COPY runpod_worker/scripts/patch_sage_setup.py /tmp/patch_sage_setup.py

# The patch script pins the target architectures without a GPU present, and
# EXITS NONZERO if it cannot prove which arches will be compiled -- an empty
# SageAttention wheel installs perfectly and then dies at the first kernel
# launch. Current upstream honours TORCH_CUDA_ARCH_LIST natively, in which case
# the script is a verified no-op; see its docstring.
# hadolint ignore=DL3059
RUN git clone --depth 1 --branch "${SAGEATTENTION_REF}" \
      https://github.com/thu-ml/SageAttention.git /tmp/sage
WORKDIR /tmp/sage
# hadolint ignore=DL3059
RUN python3 /tmp/patch_sage_setup.py setup.py && \
    pip wheel --no-build-isolation --no-deps -w /wheels . && \
    ls -la /wheels

# -----------------------------------------------------------------------------
# stage 2 — runtime
# -----------------------------------------------------------------------------
# `devel`, NOT `runtime`. requirements.txt:64 pins insightface==0.7.3 for linux,
# which ships sdist-only on PyPI and compiles mesh_core_cython; pycocotools
# (requirements.txt:51) is the same. The repo's own Dockerfile gets away with
# `pip install -r requirements.txt` only because it builds in the devel base.
# Swapping to cudnn-runtime saves ~3 GB and breaks the pip install.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# --- environment -------------------------------------------------------------
# Split into three ENV blocks: platform, WanGP runtime layout, worker policy.
# Every one of these is overridable from the RunPod endpoint's env settings;
# nothing here is a hard-coded constant in the Python.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    SDL_AUDIODRIVER=dummy \
    PULSE_RUNTIME_PATH=/tmp/pulse-runtime \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    TORCH_ALLOW_TF32_CUDNN=1

# WANGP_ROOT is the repo root; shared/api.py chdir()s here to import wgp.
# WANGP_CONFIG_DIR holds wgp_config.json -- the FILE NAME is mandatory
# (shared/api.py:1071-1072 raises ValueError otherwise) and a non-default
# directory is passed to wgp as `--config <dir>` (shared/api.py:1073-1077),
# which is why WANGP_CLI_ARGS must never contain its own --config.
# PYTHONPATH lets `python3 /opt/wangp/runpod_worker/handler.py` import both
# `runpod_worker.*` and `shared.api`; handler.py also fixes sys.path itself.
ENV WANGP_ROOT=/opt/wangp \
    WANGP_CONFIG_DIR=/opt/wangp/config \
    WANGP_OUTPUT_DIR=/tmp/wangp-out \
    WANGP_JOB_ROOT=/tmp/wangp-jobs \
    WANGP_VOLUME_ROOT=/runpod-volume \
    PYTHONPATH=/opt/wangp

# One model_type per endpoint: wgp.py:6773 reloads the entire model whenever
# model_type, profile or config differs from the loaded one, so a per-request
# switch costs a full release_model()+reload. ALLOW_MODEL_SWITCH=1 opts out.
#
# --attention sdpa, not sage/sage2: resolve_attention_mode (shared/attention.py:294-302)
# RAISES on an unsupported mode, and wgp.py:6815-6818 turns an unsupported
# per-task override into send_cmd("exit"); return True -- a silent, file-less
# "success" that bills for the whole load. Flip to sage2 via WANGP_CLI_ARGS +
# WANGP_ATTENTION only after confirming the endpoint's GPU tier and measuring it.
# The whitelist wgp.py:3303 accepts is: auto sdpa sage sage2 flash xformers.
#
# --profile 4 (LowRAM_LowVRAM) is WanGP's own default and needs >=32 GB system
# RAM. Log psutil.virtual_memory().total from the first staging worker and fall
# back to --profile 5 if the tier is under ~64 GB.
#
# The turbo LoRA is deliberately NOT baked into the config: it is opt-in per
# request through input.profile (see runpod_worker/test_input.json).
ENV WANGP_MODEL_TYPE=minimax_h3_fl2va_pruned \
    WANGP_ATTENTION=sdpa \
    WANGP_PROFILE=4 \
    WANGP_CLI_ARGS="--attention sdpa --profile 4 --verbose 1" \
    WANGP_TRANSFORMER_QUANT=int8 \
    WANGP_TEXT_ENCODER_QUANT=int8 \
    WANGP_CONSOLE=1 \
    WANGP_OUTPUT_CHAIN=presigned,rp_bucket,base64 \
    ALLOW_URL_INPUTS=0 \
    ALLOW_MODEL_SWITCH=0

# --- system packages ---------------------------------------------------------
# ffmpeg is baked for two reasons: media_out shells out to `ffprobe` for the
# output metadata block, and having real binaries on PATH means a failed
# download_ffmpeg() is degraded rather than fatal.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git wget curl \
      libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# uid 1000 exists before the app is copied so COPY --chown can do the ownership
# work in the same layer instead of a second full-tree chown layer.
# hadolint ignore=DL3059
RUN useradd -u 1000 -ms /bin/bash user

WORKDIR /opt/wangp

# --- python stack ------------------------------------------------------------
# hadolint ignore=DL3013,DL3059
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# torch FIRST, from the cu128 index, so nothing in requirements.txt can pull the
# generic build in as a transitive dependency (Dockerfile:36-40 does the same).
# hadolint ignore=DL3059
RUN pip install --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
      torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# requirements.txt is consumed VERBATIM. Editing it is what made PR #317
# unmergeable. Its line 1 --extra-index-url is honoured from the file itself.
COPY requirements.txt /opt/wangp/requirements.txt
# hadolint ignore=DL3059
RUN pip install --no-cache-dir -r requirements.txt

# SageAttention wheel from stage 1. --no-deps: it declares torch, and we already
# have the exact cu128 build.
COPY --from=builder /wheels /wheels
# hadolint ignore=DL3013,DL3059
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels

# --- the runpod SDK, under constraints --------------------------------------
# runpod 1.12.0 requires fastapi[all]>=0.141.1, boto3>=1.43.66, cryptography>=50,
# paramiko>=5, aiohttp[speedups]>=3.14.3, colorama<0.4.7. fastapi[all] drags in
# pydantic-settings / pydantic-extra-types, which are free to bump pydantic and
# break the pydantic==2.10.6 / gradio==5.29.0 / mcp==1.10.1 pins in
# requirements.txt. Install under constraints and GATE THE LAYER ON `pip check`:
# a broken resolution must fail the build, not the first cold start.
# boto3 arrives here as a runpod dependency, which is why media_out.py may
# import it without a requirements line of its own.
COPY runpod_worker/requirements-worker.txt runpod_worker/constraints.txt /opt/wangp/runpod_worker/
# hadolint ignore=DL3059
RUN pip install --no-cache-dir -c runpod_worker/constraints.txt \
      -r runpod_worker/requirements-worker.txt && \
    pip check

# --- the application ---------------------------------------------------------
# The repo's own Dockerfile deliberately ships NO app code -- it bind-mounts
# $(pwd):/workspace at run time. A serverless image must carry it.
#
# --chown=user:user matters and is not cosmetic: the repo root MUST be writable
# by the runtime uid. `import wgp` does os.mkdir("settings") (wgp.py:2549),
# writes wgp_config.json (wgp.py:2618-2619), get_default_settings() json.dump()s
# settings/<model_type>_settings.json on first call (wgp.py:3174), and
# loras_url_cache_v2.json is a bare relative path. A read-only rootfs, or a
# root-owned tree under USER user, breaks all four.
COPY --chown=user:user . /opt/wangp

# Writable dirs. /opt/wangp itself is chowned explicitly: COPY --chown sets
# ownership on the files it writes, not on a destination directory WORKDIR
# already created as root -- and wgp creates new files directly in the root.
# hadolint ignore=DL3059
RUN mkdir -p /opt/wangp/config /opt/wangp/settings /opt/wangp/ckpts \
             /opt/wangp/loras /opt/wangp/ffmpeg_bins \
             /tmp/wangp-out /tmp/wangp-jobs /home/user/.cache && \
    chown user:user /opt/wangp && \
    chown -R user:user /opt/wangp/config /opt/wangp/settings /opt/wangp/ckpts \
                       /opt/wangp/loras /opt/wangp/ffmpeg_bins \
                       /tmp/wangp-out /tmp/wangp-jobs /home/user

USER user

# ffmpeg: shared/api.py:1090 calls module.download_ffmpeg() on EVERY runtime
# init. Running it here is a BUILD-TIME ASSERTION, not really a prefetch --
# on Linux download_ffmpeg() resolves ffmpeg/ffprobe through shutil.which
# (shared/ffmpeg_setup.py:116-131) and, when the hit is outside ffmpeg_bins,
# _libs_present() returns True for system FFmpeg (:137-145), so it takes the
# early return at :163-165 and downloads NOTHING. The apt ffmpeg above is what
# actually removes the network dependency from the cold path; this RUN proves it
# by failing over to a real download if that package ever disappears, and warms
# <repo root>/ffmpeg_bins (:27-29) if it does. Runs as `user` so the tree needs
# no second chown, and needs no GPU: shared/ffmpeg_setup imports only
# requests/tqdm. Non-fatal on purpose -- with system ffmpeg present a registry
# hiccup here costs nothing at all.
# hadolint ignore=DL3059
RUN python3 -c "import sys; sys.path.insert(0, '/opt/wangp'); \
from shared.ffmpeg_setup import download_ffmpeg; download_ffmpeg()" \
    || echo "[build] WARNING: ffmpeg_bins prefetch failed; the first cold start will retry it"

# NO ENTRYPOINT, on purpose. The repo's entrypoint.sh burns ~95 lines of
# nvidia-smi diagnostics per cold start, does an `su -p user -c` hop, and ends in
# `python3 wgp.py --listen` -- which enters wgp.py's __main__ (wgp.py:13623),
# launches Gradio, and can block on select.select([sys.stdin], ...) for 2 s when
# startup.lock exists (wgp.py:13787-13812). None of that is wanted here, and an
# inherited ENTRYPOINT would also swallow the SDK's own CLI flags.
#
# CMD (not ENTRYPOINT) so the flags the SDK parses still work by overriding it:
#   docker run ... python3 -u handler.py --rp_serve_api --rp_api_host 0.0.0.0
#   docker run ... python3 -u -m runpod_worker.scripts.verify_weights <model_type>
# For the local one-shot path the SDK reads test_input.json from the PROCESS CWD
# (runpod/serverless/modules/rp_local.py, hardcoded), hence `-w /opt/wangp/runpod_worker`
# in the Tier-3 recipe in docs/RUNPOD_SERVERLESS.md.
CMD ["python3", "-u", "/opt/wangp/runpod_worker/handler.py"]
