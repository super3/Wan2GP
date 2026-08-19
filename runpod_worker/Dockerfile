# =============================================================================
#  WanGP / MiniMax H3 — RunPod Serverless worker image
#
#  BUILD FROM THE REPO ROOT (the context is the whole repo, not runpod_worker/):
#
#    docker build --platform linux/amd64 \
#      -f runpod_worker/Dockerfile -t you/wangp-h3:2026.08.18-1 .
#
#  * --platform linux/amd64 is MANDATORY. RunPod rejects ARM images.
#  * NEVER tag :latest. RunPod caches images per host and a mutable tag silently
#    serves stale code to warm workers. Use YYYY.MM.DD-N.
#  * DEFAULT BUILD IS ~10 MIN AND SHIPS NO SageAttention. The image runs
#    `--attention sdpa` (see WANGP_CLI_ARGS below), so a compiled sage wheel
#    would be built, shipped and never loaded -- 40-90 min of nvcc for nothing,
#    and more than RunPod's own image builder allows per step (30 min), which the
#    phase-1 plan counts on. Opt in only when you have measured sage2 on the
#    endpoint's GPU tier AND flipped WANGP_CLI_ARGS/WANGP_ATTENTION to match:
#
#      docker build --platform linux/amd64 --build-arg WITH_SAGE=1 \
#        --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0;12.0" \
#        -f runpod_worker/Dockerfile -t you/wangp-h3:2026.08.18-1 .
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
# WITH_SAGE: 0 (default) skips the compile entirely -- the builder stage still
# runs but produces an empty /wheels, and the runtime stage's install is a no-op.
ARG WITH_SAGE=0
# CUDA_ARCHITECTURES: the repo default "8.0;8.6" (Dockerfile:20) excludes L4 /
# L40S / RTX 4090 (SM89), H100 (SM90) and B200 (SM120) -- precisely the fleet a
# serverless endpoint schedules on. is_sage2_supported() (shared/sage2_core.py:75-81)
# only checks that the *device* capability is >= 8.0; it never checks which arches
# the wheel carries, so a wheel missing the host's SM reports "supported" and then
# dies at the first attention kernel launch, minutes into a billed job. Every arch
# you add costs build minutes and wheel size, not run time.
ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0;12.0"
# Pin to a tag or branch for a reproducible build; `main` is a moving target.
# --depth 1 --branch only accepts a ref name, not a bare commit sha.
ARG SAGEATTENTION_REF="main"
# nvcc is memory-hungry: budget roughly 2 GB of builder RAM per job. Lower this
# (--build-arg MAX_JOBS=4) if the build dies with an OOM-killed cicc.
ARG MAX_JOBS="8"

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder
ARG WITH_SAGE
ARG CUDA_ARCHITECTURES
ARG SAGEATTENTION_REF
ARG MAX_JOBS
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}" \
    FORCE_CUDA=1 \
    MAX_JOBS="${MAX_JOBS}" \
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
RUN mkdir -p /wheels && \
    if [ "${WITH_SAGE}" = "1" ]; then \
      pip install --no-cache-dir --upgrade pip setuptools wheel packaging; \
    fi

# The torch here must match the torch in the runtime stage exactly: a C++
# extension is compiled against a specific torch ABI. The runtime stage no
# longer pins torch -- requirements.txt resolves it (measured: 2.13.0, CUDA 13
# wheels from PyPI) -- so this must track that same version, from the same
# index, or the wheel installs cleanly and dies at the first kernel launch.
# If requirements.txt moves torch, update BOTH this line and constraints.txt.
# hadolint ignore=DL3059
RUN if [ "${WITH_SAGE}" = "1" ]; then \
      pip install --no-cache-dir torch==2.13.0; \
    fi

COPY runpod_worker/scripts/patch_sage_setup.py /tmp/patch_sage_setup.py

# The patch script pins the target architectures without a GPU present, and
# EXITS NONZERO if it cannot prove which arches will be compiled -- an empty
# SageAttention wheel installs perfectly and then dies at the first kernel
# launch. Current upstream honours TORCH_CUDA_ARCH_LIST natively, in which case
# the script is a verified no-op; see its docstring.
#
# Skipped wholesale unless WITH_SAGE=1: this single step is 40-90 min, which is
# both the whole build's cost and more than RunPod's own image builder allows in
# one step, and the baked default (--attention sdpa) never loads the result.
# DL3003: `cd` rather than WORKDIR, because the whole step is inside one
# conditional -- a WORKDIR would have to exist even when the clone is skipped.
# hadolint ignore=DL3003,DL3059
RUN if [ "${WITH_SAGE}" = "1" ]; then \
      git clone --depth 1 --branch "${SAGEATTENTION_REF}" \
        https://github.com/thu-ml/SageAttention.git /tmp/sage && \
      cd /tmp/sage && \
      python3 /tmp/patch_sage_setup.py setup.py && \
      pip wheel --no-build-isolation --no-deps -w /wheels . ; \
    else \
      echo "[build] WITH_SAGE=0: SageAttention not compiled (--attention sdpa)"; \
    fi && \
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

LABEL org.opencontainers.image.title="WanGP MiniMax H3 RunPod Serverless worker" \
      org.opencontainers.image.description="RunPod Serverless handler driving WanGP (deepbeepmeep/Wan2GP) through shared/api.py. This image integrates WanGP; see docs/API.md:9 and LICENSE.txt:316 for the required disclosure." \
      org.opencontainers.image.source="https://github.com/deepbeepmeep/Wan2GP" \
      org.opencontainers.image.licenses="Apache-2.0"

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
# RAISES on an unsupported mode, and wgp.py:6816-6819 turns an unsupported
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
    ALLOW_MODEL_SWITCH=0 \
    WANGP_ALLOWED_LORAS="" \
    WANGP_VOLUME_INPUT_SUBDIR=inputs

# WANGP_ALLOWED_LORAS is EMPTY on purpose, and empty means "no caller-supplied
# LoRAs at all" -- not "any LoRA". An activated_loras entry is a filename WanGP
# resolves under the shared loras directory, and an http(s) entry used to be a
# caller-steered download primitive (update_loras_url_cache wgp.py:9980-9996 ->
# check_loras_exist wgp.py:3689-3706 -> urlretrieve). URLs are refused outright
# now; to let callers name staged files, list their BASENAMES here, comma
# separated. The turbo LoRA does not need this: it arrives through
# input.profile, whose fragment ships in the image.
#
# WANGP_VOLUME_INPUT_SUBDIR confines `{"volume": "..."}` inputs to
# /runpod-volume/inputs, keeping outputs/, ckpts/ and loras/ out of reach of a
# request. Set it to "" only on a single-tenant endpoint.

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

# NO torch pre-install here -- deliberately.
#
# This layer used to install torch==2.10.0+cu128 from the cu128 index "so
# nothing in requirements.txt can pull the generic build in". Measured on a real
# RunPod pod (2026-08-19): that does not work. `pip install -r requirements.txt`
# resolves torch to 2.13.0 and REPLACES the pinned build, pulling the CUDA 13
# wheel set (nvidia-cublas-13.x, nvidia-cudnn-cu13, ...) from PyPI. The pin cost
# ~2.5 GB of download per build and was discarded seconds later.
#
# The resolved stack -- torch 2.13.0 / torchvision 0.28.0 / torchaudio 2.11.0 on
# CUDA 13 wheels -- was verified end to end on that pod: `import wgp` succeeded
# and a full minimax_h3_fl2va_pruned generation completed on an RTX PRO 4500
# (Blackwell, sm_120, driver 580.173.02), 20 steps in 394.7 s. Note the cu13
# wheels run fine on this cu12.8.1 base image: torch ships its own CUDA runtime
# in those nvidia-* packages, so the base image's CUDA only has to satisfy the
# driver, not the wheels.
#
# requirements.txt therefore OWNS the torch version. constraints.txt pins the
# triple to the verified versions so the runpod layer cannot move them.

# requirements.txt is consumed VERBATIM. Editing it is what made PR #317
# unmergeable. Its line 1 --extra-index-url is honoured from the file itself.
COPY requirements.txt /opt/wangp/requirements.txt
# hadolint ignore=DL3059
RUN pip install --no-cache-dir -r requirements.txt

# SageAttention wheel from stage 1, if one was built. --no-deps: it declares
# torch, and we already have the exact cu128 build. With WITH_SAGE=0 the
# directory is empty and this is a no-op -- `pip install /wheels/*.whl` with no
# match would otherwise fail the build on a glob that never expanded.
COPY --from=builder /wheels /wheels
# hadolint ignore=DL3013,DL3059
RUN if ls /wheels/*.whl >/dev/null 2>&1; then \
      pip install --no-cache-dir --no-deps /wheels/*.whl; \
    else \
      echo "[build] no SageAttention wheel; running --attention sdpa"; \
    fi && \
    rm -rf /wheels

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
# The gate. `pip check` cannot be used raw: requirements.txt itself ships
# `decord 0.6.0`, whose wheel carries a platform tag this image does not match,
# so `pip check` reports it BEFORE this layer runs (verified by probe,
# 2026-08-19). Failing on it would make the gate unpassable; ignoring pip check
# entirely would have missed the real tomlkit/gradio break it caught on the same
# build. So: allow exactly that one known line, fail on anything else.
# `set -e` FIRST and the steps separated by `;`, not `&&`: an earlier version
# of this layer chained `pip install ... && {...} ; if [ -s ... ]`, so when the
# install failed the `&&` short-circuited, the `;` ran the `if` against a file
# that did not exist, and the RUN exited 0 -- producing an image with NO runpod
# SDK that looked like a clean build. A silently broken image is worse than a
# failed one. Never let this layer's first command be optional.
RUN set -e; \
    pip install --no-cache-dir -c runpod_worker/constraints.txt \
      -r runpod_worker/requirements-worker.txt; \
    pip check > /tmp/pc.raw 2>&1 || true; \
    grep -vE '^WARNING:|^decord 0\\.6\\.0 is not supported on this platform$' \
      /tmp/pc.raw > /tmp/pc.out || true; \
    if [ -s /tmp/pc.out ]; then \
      echo '[build] pip check found NEW dependency conflicts:'; \
      cat /tmp/pc.out; exit 1; \
    fi; \
    rm -f /tmp/pc.raw /tmp/pc.out; \
    python3 -c "import runpod, gradio, tomlkit; print('[build] runpod/gradio/tomlkit import OK')" 

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
# CMD (not ENTRYPOINT) so the flags the SDK parses still work by overriding it.
# NOTE the absolute path: WORKDIR is /opt/wangp, and handler.py lives one level
# down, so `python3 -u handler.py` here exits with "can't open file".
#   docker run ... python3 -u /opt/wangp/runpod_worker/handler.py \
#                    --rp_serve_api --rp_api_host 0.0.0.0
#   docker run ... python3 -u -m runpod_worker.scripts.verify_weights <model_type>
# For the local one-shot path the SDK reads test_input.json from the PROCESS CWD
# (runpod/serverless/modules/rp_local.py, hardcoded), hence `-w /opt/wangp/runpod_worker`
# in the Tier-3 recipe in docs/RUNPOD_SERVERLESS.md.
CMD ["python3", "-u", "/opt/wangp/runpod_worker/handler.py"]
