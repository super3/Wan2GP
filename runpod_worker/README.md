# WanGP MiniMax H3 — RunPod Serverless worker

A RunPod Serverless worker that generates MiniMax H3 video (with synchronized audio) by
driving WanGP in-process through `shared/api.py`. One `model_type` per endpoint, one
generation at a time per worker, weights on a network volume, output delivered as a URL.

**Callers must use `POST /run` + `GET /status/{id}` (or a webhook). Never `/runsync`.**
A generation takes minutes; `/runsync` waits 90 s by default and 300 s at most, and
retains its result for only 1–5 minutes. `/run` returns an id immediately and the result
survives for 30 minutes.

```bash
ID=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
      -d @runpod_worker/test_input.json | jq -r .id)

curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" | jq '.status, .output.video.url'
```

Full design rationale, line citations and the decision log live in
[`docs/RUNPOD_SERVERLESS.md`](../docs/RUNPOD_SERVERLESS.md). This file is the runbook:
how to stand it up, what every knob does, and what to do when it breaks.

---

## Built on WanGP — required disclosure

This worker integrates [WanGP](https://github.com/deepbeepmeep/Wan2GP) and calls its
Python API. `docs/API.md:9` states:

> **Please note that use of the WanGP API is subject to the WanGP Terms and Conditions.
> Any product that integrates WanGP should clearly disclose that it uses WanGP in both
> its user interface and its documentation.**

The WanGP Community License 2.0 (`LICENSE.txt`) repeats the requirement for wrappers and
integrations, in §7.3 (`LICENSE.txt:313-317`):

> **7.3 Free, non-monetized integrations and wrappers are permitted so long as they:
> (a) comply with this License; (b) clearly disclose WanGP use in reasonable
> documentation or an About section; and (c) preserve all notices required for the
> Software and for Third-Party Materials.**

**If you deploy this worker, the disclosure obligation is yours.** Put "Powered by
WanGP", with a link to the project, in the UI of whatever product calls this endpoint
*and* in that product's documentation. Documentation alone is not enough — `docs/API.md:9`
asks for both.

Read §7.2 before you charge anyone for access (`LICENSE.txt:307-311`):

> **7.2 Exposing the Software to third parties through an API, local service, hosted
> endpoint, plugin, Integration, wrapper, bridge, or Headless Usage in exchange for
> consideration is Restricted Commercialization and requires a separate written reseller
> or commercial license.**

A private endpoint you run for your own team or your own client work is Free Use
(§1.8, §5.3). A paid or metered API in front of this worker is not, and needs a written
commercial licence from the WanGP author first — `deepbeepmeep@yahoo.com`. Separately,
§6.2 asks for reasonable credit ("Made with WanGP") whenever you sell or license an
output as the thing being paid for.

Nothing here is legal advice; read `LICENSE.txt` yourself.

---

## Contents

- [What you get](#what-you-get)
- [One-time setup](#one-time-setup)
  - [1. Network volume](#1-network-volume)
  - [2. Warm the volume from a temporary Pod](#2-warm-the-volume-from-a-temporary-pod)
  - [3. Build and push the image](#3-build-and-push-the-image)
  - [4. Create the endpoint](#4-create-the-endpoint)
- [Environment variables](#environment-variables)
- [Request schema](#request-schema)
- [Response schema](#response-schema)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Operational chores](#operational-chores)
- [Cost notes (estimates)](#cost-notes-estimates)
- [File map](#file-map)

---

## What you get

| | |
|---|---|
| Model | `minimax_h3_fl2va_pruned` by default; `minimax_h3_fl2va`, `minimax_h3_ref2va`, `minimax_h3_ref2va_pruned` are supported, one per endpoint |
| Input | prompt + optional start/end frames, control video, reference images, audio references — as base64 or `volume://` paths |
| Output | a muxed `.mp4` with an AAC track, delivered by presigned PUT, your S3 bucket, or base64 under a cap |
| Concurrency | **Exactly one generation at a time per worker process.** WanGP holds a process-global generation lock and installs a process-global `redirect_stdout` for the whole job, so two jobs in one process is not a tuning question. Jobs are served sequentially by the same warm process; scale with `max_workers`, never with `concurrency_modifier`. |
| Validation | every request is pre-flighted on CPU in <50 ms — unknown keys, illegal flag-letter combinations, frame-count math, LoRA paths — so a bad request costs zero GPU seconds |
| Cancellation | `runtime.timeout_s` and `POST /cancel/{id}` both abort the generation cooperatively, landing at the model's next interrupt check (one denoising step). Neither is instantaneous — see [Cancellation, precisely](#cancellation-precisely). |

What it deliberately does **not** do: run more than one model per endpoint (a switch costs
a full `release_model()` + reload), accept arbitrary local paths in `settings`, return a
container-local file path, or download weights on the clock.

---

## One-time setup

### 1. Network volume

Weights are ~60 GB for the pruned 20B build at int8 (≈73 GB for the full 33B). They live
on a RunPod network volume, not in the image.

```bash
curl -X POST https://rest.runpod.io/v1/networkvolumes \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"wangp-h3-us-ca-2","size":200,"dataCenterId":"US-CA-2"}'
```

200 GB at ~$0.07/GB/month ≈ **$14/month**. Size can be increased, never decreased.
Pick a datacenter that actually carries your GPU tier: the volume pins every worker to
that datacenter, which shrinks the pool you can schedule on.

> **Keep `max_workers=1` until the volume is verified.** RunPod warns that simultaneous
> writes from several workers to one volume can corrupt it, and an empty volume means
> every cold worker starts downloading the same 60 GB.

### 2. Warm the volume from a temporary Pod

Launch a **GPU Pod** with the volume attached and this image. A GPU is mandatory even for
prefetching: `import wgp` calls `torch.cuda.get_device_capability()` at module scope
(`wgp.py:2508`, again at `shared/attention.py:14`).

> ### The mount-path asymmetry — the least obvious thing in this deployment
>
> | | Mount path |
> |---|---|
> | **Serverless worker** | `/runpod-volume` |
> | **Pod** | `/workspace` (it replaces the pod's default volume disk) |
>
> The image bakes `WANGP_VOLUME_ROOT=/runpod-volume`. On the Pod you must override it,
> or every file you prefetch lands on the pod's ephemeral disk and is destroyed with it:
>
> ```bash
> export WANGP_VOLUME_ROOT=/workspace     # or: mount --bind /workspace /runpod-volume
> ```
>
> Clone the repo **outside** the volume so ~5 GB of source does not ride along and get
> billed as storage forever.

```bash
# 1) Download every weight file the endpoint's model_type needs.
#    --transformer-quant MUST match what the workers run (int8 here). Warm as bf16 and
#    run as int8 and every cold start silently re-downloads 21 GB, billed.
python3 -m runpod_worker.scripts.prefetch_weights \
    --root /opt/wangp --config /opt/wangp/config/wgp_config.json \
    --transformer-quant int8 --text-encoder-quant int8 \
    minimax_h3_fl2va_pruned

# 2) Stage the accelerator LoRA. get_lora_local_path (wgp.py:3670-3677) maps an https
#    entry in activated_loras to os.path.join(lora_dir, basename(url)), so a file staged
#    under that basename resolves with zero network at request time.
python3 -m runpod_worker.scripts.prefetch_weights \
    --profile "Turbo Lightx2v FL2V 4 Steps v1.0 768p" minimax_h3_fl2va_pruned
#    (or by hand:)
mkdir -p "$WANGP_VOLUME_ROOT/loras/minimax_h3"
wget -P "$WANGP_VOLUME_ROOT/loras/minimax_h3" \
  https://huggingface.co/DeepBeepMeep/MiniMax-H3/resolve/main/loras/minimax_h3_lightx2v_fl2v_turbo_4step_alpha128_v1.0_768p_bf16.safetensors

# 3) Gate. This is the exact enumeration the worker's fitness check runs at boot.
python3 -m runpod_worker.scripts.verify_weights \
    --profile "Turbo Lightx2v FL2V 4 Steps v1.0 768p" minimax_h3_fl2va_pruned
echo "exit=$?"      # 0 = ready to deploy. 1 = do not point an endpoint at this volume.

du -sh "$WANGP_VOLUME_ROOT/ckpts"     # expect ≈55–60 GB for the pruned int8 build
```

`verify_weights.py` asserts the core file set is complete, reports the download status,
prints the resolved transformer and text-encoder filenames for eyeballing, and calls
`get_default_settings(model_type)` once so `settings/<model_type>_settings.json` exists —
that file is `json.dump()`ed on first call (`wgp.py:3174`) and you do not want that
happening on a billed request against a slow volume. `--json PATH` writes a machine-readable
report; `--strict` also fails on missing shared assets; `--allow-partial` tolerates a
secondary file (a VAE, a preload URL) that WanGP would fetch on first use.

`prefetch_weights.py` useful flags: `--dry-run` (report only), `--list-profiles`,
`--all-profiles`, `--config-id gguf_q4_k_m` (warm the GGUF text encoder instead of the
26.7 GB INT8 one), `--lora URL_OR_NAME`, `--json PATH`.

Then terminate the Pod and attach the volume to the endpoint:
**Serverless → endpoint → Manage → Edit Endpoint → Advanced → Network Volumes.**

Do not skip the shared assets. `download_models(file_type=0)` unconditionally pulls
DWPose, scribble, RAFT, Depth-Anything-V2, wav2vec ×2, BS-RoFormer, pyannote, det_align
and MatAnyone (`wgp.py:3545-3557`, `:3585-3587`) — several GB that are not optional and
*will* download on the first request if you skip the prefetch. `prefetch_weights.py`
handles them.

### 3. Build and push the image

```bash
docker build --platform linux/amd64 \
  -f runpod_worker/Dockerfile \
  -t you/wangp-h3:2026.08.18-1 .
docker push you/wangp-h3:2026.08.18-1
```

Build from the **repo root** — the Dockerfile copies `requirements.txt` and the whole
tree. The default build is ~10 minutes and ships **no SageAttention**: the image runs
`--attention sdpa`, so a compiled sage wheel would be built, shipped and never loaded —
40–90 minutes of nvcc for nothing, and more than the 30-minute-per-step limit RunPod's own
image builder allows, which the phase-1 plan counts on. Opt in only after measuring sage2
on the endpoint's GPU tier, and flip `WANGP_CLI_ARGS`/`WANGP_ATTENTION` to match:

```bash
docker build --platform linux/amd64 --build-arg WITH_SAGE=1 \
  --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0;12.0" \
  -f runpod_worker/Dockerfile -t you/wangp-h3-sage:2026.08.18-1 .
```

- **`--platform linux/amd64` is mandatory.** RunPod rejects ARM images. On an Apple
  Silicon machine, that flag makes it an emulated cross-build; a cloud builder is faster.
- **Never tag `:latest`.** RunPod caches images per host and a mutable tag silently
  serves stale code. Use `YYYY.MM.DD-N`. Rollback is "point the endpoint at the previous
  tag and save" — which only works if the tags are immutable.
- **With `WITH_SAGE=1`, `CUDA_ARCHITECTURES` must cover your GPU priority list.** The
  repo's own default `"8.0;8.6"` excludes L4/L40S/4090 (SM 8.9), H100 (SM 9.0) and B200
  (SM 12.0) — exactly the fleet a serverless endpoint schedules on. `is_sage2_supported()`
  (`shared/sage2_core.py:75-81`) only checks that the *device* capability is ≥ 8.0; it
  never checks what the wheel was built for, so a wheel missing your arch reports
  "supported" and then fails at the first attention kernel launch, minutes into a billed
  job. This worker's default is `"8.0;8.6;8.9;9.0;12.0"`.
- Other build args: `SAGEATTENTION_REF` (default `main`; pin it for reproducibility) and
  `MAX_JOBS` (default 8; nvcc needs ~2 GB of builder RAM per job — lower it if `cicc`
  gets OOM-killed).

> **Gotcha: `.gitignore:1` is `.*`, which matches every dotfile.** `.dockerignore` and
> `.github/workflows/worker-ci.yml` are tracked only because they were force-added
> (`git add -f`); ignore rules do not apply to files git already tracks, so they stay
> tracked. But **any new dotfile is silently skipped by a plain `git add`** — a second
> workflow, a `.env.example`, a `.github/dependabot.yml`. If `.dockerignore` ever goes
> missing from a clone, the build context grows by the whole `.git` directory plus any
> local `ckpts/`. The durable fix is two negation lines in `.gitignore`:
> `!.dockerignore` and `!.github/`.

Verify before pushing:

```bash
docker run --rm you/wangp-h3:2026.08.18-1 ls /opt/wangp/wgp.py
docker run --rm you/wangp-h3:2026.08.18-1 pip check
```

### 4. Create the endpoint

Serverless → New Endpoint → import from Docker Registry.

| Setting | Value | Why |
|---|---|---|
| **Execution timeout** | **3600 s** | Default 600 s kills the first request outright. Must exceed the handler budget so our cooperative cancel always wins the race against RunPod's hard kill. Budget arithmetic: 2600 + 150 grace + ~60 probe/upload ≈ 2810 < 3600 ✓ |
| **Idle timeout** | **180 s** (default 5 s) | 180 idle seconds on an L40S ≈ $0.095; reloading weights is ~200 s ≈ $0.106 *and* 3+ minutes of latency. Staying warm wins for any inter-arrival gap under ~3 minutes. |
| **Job TTL** | 24 h default | The timer starts at submission, so it covers queue wait. Lower it only once you have measured queue depth. |
| **Max workers** | **1** until the volume is verified, then ≥3 | Concurrent writes to one volume can corrupt it. |
| **Active workers** | 0 to start | One always-on L40S ≈ $1.91/h ≈ $1,374/month, but removes the weight load entirely. Sizing: `active = (req/min × duration_s) / 60`. |
| **GPU priority** | L40S / L40 / RTX 6000 Ada → A6000 / A40 → A100 80 GB | 48 GB tiers. Do **not** list 24 GB tiers: a 21 GB int8 transformer plus a 26.7 GB text encoder forces continuous PCIe block-swapping — cheap per second, expensive per generation. |
| **Autoscaling** | Request count, scaler 1 | Queue-delay scaling with a 4 s threshold reacts far too late for multi-minute jobs. |
| **FlashBoot** | on (default) | Free; helps most under steady traffic. |
| **Network volume** | the one you just warmed | |
| **Concurrency** | leave alone | The worker passes no `concurrency_modifier`; the SDK default of 1 is correct and required. |

Env worth setting on the endpoint. Nothing here is *required* — the image bakes working
defaults and the worker boots with an empty env — but these are the ones that matter:

```
WANGP_MODEL_TYPE=minimax_h3_fl2va_pruned    # baked; set it anyway, it is the endpoint's identity
WANGP_ALLOWED_LORAS=minimax_h3_lightx2v_fl2v_turbo_4step_alpha128_v1.0_768p_bf16.safetensors
BUCKET_ENDPOINT_URL=...          # or none of the four, and require callers to send
BUCKET_ACCESS_KEY_ID=...         # output.presigned_url instead
BUCKET_SECRET_ACCESS_KEY=...
BUCKET_NAME=...
REQUIRE_BUCKET=1                 # if outputs will exceed the 6 MB base64 cap
```

Leaving `WANGP_ALLOWED_LORAS` empty means **no caller-supplied LoRAs at all** — a request
that names one is rejected. Remote (`http://`/`https://`) entries are refused outright
whatever the allow-list says: WanGP would fetch them with `urlretrieve` from inside the
RunPod network and write the bytes to the shared volume, where the next job loads them as
weights. The one exception is a URL a shipped accelerator profile contributes verbatim
(`input.profile`), which resolves to a staged file by basename. To let callers name their
own staged LoRAs, list the basenames here.

VRAM is not the binding constraint (5–6 GB for 5 s at 832×480 with mmgp block-swapping);
**system RAM is**, because mmgp streams ~48–60 GB of weights through it. RunPod does not
publish per-tier host RAM for serverless workers, so log
`psutil.virtual_memory().total` from your first staging worker and fall back from
`--profile 4` to `--profile 5` (via `WANGP_CLI_ARGS` and `WANGP_PROFILE`) if it is under
~64 GB.

---

## Operational findings (measured on RunPod serverless, 2026-08-19)

Hard-won facts from the first production deployment; each was observed, not
inferred.

- **Set `minCudaVersion: "13.0"` on every endpoint.** The image's torch resolves
  to CUDA-13 wheels, and RunPod's fleet is mixed: hosts with pre-12.8 drivers
  refuse to start the container (`nvidia-container-cli: unsatisfied condition:
  cuda>=12.8`); hosts with exactly-12.8 drivers start it and then torch dies at
  init (`driver too old (found 12080)`), producing an unhealthy-worker restart
  loop that bills while going nowhere. Only CUDA >= 13.0 hosts work. The gate is
  the fix; rebuilding on cu12.8 wheels would widen the eligible fleet (A40/A6000
  Ampere hosts) at the cost of a rebuild.
- **mmgp profiles 1 and 2 are unusable in these containers -- CONFIRMED, not
  inferred.** Both pin the full model set (~45 GB) into page-locked RAM, and
  serverless containers get a ~46.6 GiB memory limit; the RunPod worker log
  shows the kill directly: `high memory utilization - 42.35GiB / 46.57GiB
  (90%)` -> `container is unhealthy: exit code 137 ... triggered memory limits
  (OOM)`, partway through pinning the 25 GB text encoder -- silent from inside
  the process (no traceback), followed by a crash loop. A 64 GB desktop clears
  the same load fine. Profile 4 pins only the ~20 GB
  transformer and works. Profile 3 works (VRAM-resident, ~24.9 GB peak) and a
  controlled A/B on identical hardware (two endpoints, both NVIDIA A40 @
  EU-SE-1, same image and payloads, seeds 12345/5555) measured it ~7% faster
  than profile 4: warm job 69.0 s vs 73.9 s, first job 108.5 s vs 116.6 s.
  Worth switching if workers are on >=32 GB cards; profile 4 remains the safe
  default (5.6 GB VRAM peak, runs on any tier). (Desktops without cgroup limits run profile 1 fine -- this is a
  container limit, not a model or VRAM limit.)
- **Profile 1 works on big-GPU tiers and roughly halves warm latency.** The
  ~46.6 GiB container limit is a 48 GB-class artifact; H100 80GB and full RTX
  PRO 6000 (96 GB) containers survive the ~45 GB pin. Measured warm (832x480,
  124f, 4 steps): H100 profile 1 = 22.9 s (vs 39.5 s profile 3 on the same
  card); full PRO 6000 profile 1 = 24.2 s. At pod-tier prices that is ~1.4
  cents/clip on the PRO 6000 -- production-MIG cost, half the latency. GPU
  value table (warm, profile 3 unless noted): H100 p1 22.9 s / PRO 6000 p1
  24.2 s / H100 p3 39.5 s / MIG 2g.48gb p3 46.7 s / A40 p3 67.0 s.
- **Multi-GPU sequence parallelism works and is worth 1.31x, no more.**
  `models/minimax_h3/usp.py` adds Ulysses sequence parallelism (an all-to-all
  around each block's attention so every rank holds the full sequence for
  `heads / world_size` heads) via plain `torch.distributed` -- no xfuser, no
  ray. Measured on ONE 2x A40 pod, same clip, warm second job:
  **profile 4: 70.86 s on 1 GPU -> 53.89 s on 2 GPUs (1.31x)**; profile 1:
  71.14 s -> 55.03 s (1.29x). Cold: 121.7 -> 92.3 s and 93.2 -> 101.5 s.
  That is Amdahl's law, not an implementation defect: of the 70.86 s, only
  ~33 s is denoise (the parallel part) and ~38 s is serial text encode + VAE
  decode + mux, so perfect scaling predicts 54.5 s and we measured 53.89 s --
  i.e. the denoise phase parallelized at ~100% efficiency and the all-to-all
  overhead is in the noise. Correctness is proven before timing: the gloo
  suite (`runpod_worker/scripts/test_usp_gloo.py`) shows USP attention is
  numerically identical to full-sequence SDPA, and it is re-run on the bench
  host before any GPU work. Sol sparse attention and the skip-steps caches
  raise under USP (both are global-sequence constructs).
  **Cost, however, moves the wrong way**: 2 GPUs bill 2x for 1.31x, so
  cents/clip goes 0.87 -> 1.32. Multi-GPU buys latency only. A single full
  RTX PRO 6000 beats 2x A40 on both axes (22.5 s, 1.06 cents/clip). Reach for
  USP only when one card cannot go fast enough, or for long/high-res clips
  where denoise dominates and the parallel fraction grows.
- **The memory profile does not affect speed on capable hosts; it decides
  whether you fit.** Same clip, warm, same host: full RTX PRO 6000 profile 1
  22.54 s vs profile 4 22.98 s; A40 profile 1 71.14 s vs profile 4 70.86 s --
  both inside the +/-0.4 s run-to-run variance measured across five identical
  SDPA legs. The earlier "profile 1 is ~2x faster" reading was the full card
  vs the MIG slice, not the profile. Worse, profile 1 is actively dangerous on
  a RAM-tight host: it pins every component to reserved RAM (~51 GB:
  transformer 20.1 + text encoder 24.7 + VAEs/encoders) and where that does
  not fit it does not fail -- one PRO 6000 host ran profile 1 legs at 113-117 s
  against 22.5 s on a roomier host of the same GPU model, a 5x silent
  degradation. Keep profile 4: same speed, ~5.6 GB VRAM, no RAM cliff.
- **Attention accelerators need verification, not configuration.** WanGP falls
  back to SDPA silently. A first sweep reported sdpa/sol/sage2 within 0.1 s of
  each other -- all three were SDPA (no sage wheel installed; sol inert).
  Two traps: `sage2` needs the compiled wheel actually present (check
  `get_attention_modes()` lists it), and **sol is not a global attention mode
  at all** -- `shared/attention.py:28-33` keeps it in
  `ATTENTION_MODE_AVAILABILITY`, "exposed only by per-generation overrides",
  so it must be passed as the per-job `override_attention` setting; set
  globally it is ignored without warning. `usp_bench.py` now probes
  `offload.shared_state["_attention"]` after a generation and reports
  `attention_ok` so a fallback can never be reported as a measurement.
- **Worker container logs have no read API** (the console view is the only
  reader; `/v2/{endpoint}/logs` is a worker-key ingest route), so the worker
  keeps its own history reachable through the job status API: every response to
  a FAILED job carries `worker_logs` (the tail of the worker's structured-log
  ring, boot events included; size via `WORKER_LOGS_TAIL`, default 60, ring via
  `WORKER_LOG_RING`, default 200), and any job can request the same on success
  with `"runtime": {"debug": true}`. For deaths *during boot* -- when no job
  ever runs -- set `LOG_SHIP_URL` on the endpoint: the worker POSTs JSON log
  batches there every `LOG_SHIP_INTERVAL_S` (default 2 s), best effort.
- **Read phase timings from tqdm, not `phase_marks_s`.** The "inference" mark
  fires when denoising *starts* on a warm model, so denoising time lands inside
  the "decoding" window. Warm-job reality at 832x480/124f/4 steps:
  encode ~2-7 s (first call ~30 s), denoise ~8.1 s/step, VAE decode + mux ~11 s.
- **Per-step denoise speed was identical (~8.1 s) on an A100-SXM4-80GB and a
  Blackwell PRO 6000 MIG 2g.48gb slice** -- memory-bandwidth-bound at these
  settings. Pay for the cheapest CUDA-13 tier, not the biggest chip.
  `vram_peak` was ~5.6 GB under profile 4.
- **Download-on-first-boot works well**: 54 GB of weights in ~150-300 s
  (hf_transfer), so a cold worker is serving in ~4-5 min with no network volume
  and no datacenter pin.

## Environment variables

Everything is env-overridable and everything has a default; the image bakes the values
marked **(baked)**. Nothing below is required for the worker to start.

### Paths and identity

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_ROOT` | `/opt/wangp` **(baked)** | Repo root. `shared/api.py` `chdir`s here to `import wgp`, so it must be **writable by uid 1000** — `import wgp` does `os.mkdir("settings")`, writes `wgp_config.json`, and `loras_url_cache_v2.json` is a bare relative path. |
| `WANGP_CONFIG_DIR` | `/opt/wangp/config` **(baked)** | Directory holding `wgp_config.json`. The **filename is mandatory** (`shared/api.py:1071-1072` raises `ValueError` otherwise); a non-default directory is passed to wgp as `--config <dir>`, which is why `WANGP_CLI_ARGS` must never contain its own `--config`. |
| `WANGP_VOLUME_ROOT` | `/runpod-volume` **(baked)** | Network volume mount. **`/workspace` on Pods** — see the asymmetry box above. When this directory exists, `checkpoints_paths[0]` and `loras_root` are derived from it. |
| `WANGP_OUTPUT_DIR` | `/tmp/wangp-out` **(baked)** | Where WanGP writes finished files. Deleted per job after delivery unless `WORKER_KEEP_OUTPUTS=1`. |
| `WANGP_JOB_ROOT` | `/tmp/wangp-jobs` **(baked)** | Per-job scratch for materialized input media. `<root>/<job_id>/in/`. |
| `WANGP_CHECKPOINTS_PATHS` | derived: `$WANGP_VOLUME_ROOT/ckpts`, `$WANGP_ROOT/ckpts`, `.` | Override the checkpoint search list. `,` or `:` separated, order = preference. |
| `WANGP_LORA_ROOT` | derived: `$WANGP_VOLUME_ROOT/loras`, else `$WANGP_ROOT/loras` | **Absolute** `loras_root`. `get_lora_dir()` deliberately returns a relative path (`wgp.py:2498-2499`), so an absolute root here is the only way volume-staged LoRAs are ever found. |
| `PYTHONPATH` | `/opt/wangp` **(baked)** | Lets `python3 /opt/wangp/runpod_worker/handler.py` import both `runpod_worker.*` and `shared.api`. |

### Model and engine

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_MODEL_TYPE` | `minimax_h3_fl2va_pruned` **(baked)** | The one model this endpoint serves. A request naming a different one is rejected with `bad_request` unless `ALLOW_MODEL_SWITCH=1`. |
| `WANGP_MODEL_CONFIG` | *(empty)* | WanGP `config` selection string (text encoder / VAE / DiT priority), e.g. `gguf_q4_k_m`. Trailing commas are stripped — they must be, or the reload test at `wgp.py:6773` never matches. |
| `ALLOW_MODEL_SWITCH` | `0` **(baked)** | `1` permits per-request `model_type`/`config` changes. Each one costs a full `release_model()` + reload (~200 s, billed). |
| `WANGP_ATTENTION` | `sdpa` **(baked)** | Baked into `wgp_config.json`. Validated against wgp's own whitelist (`wgp.py:3303`): `auto`, `sdpa`, `sage`, `sage2`, `flash`, `xformers`. An unlisted value is rejected here rather than crashing the import or silently producing zero files. |
| `WANGP_PROFILE` | `4` **(baked)** | mmgp profile. 4 = `LowRAM_LowVRAM`, WanGP's own default, needs ≥32 GB system RAM. |
| `WANGP_CLI_ARGS` | `--attention sdpa --profile 4 --verbose 1` **(baked)** | `shlex`-split and handed to `shared.api.init(cli_args=…)`. If `--attention` here disagrees with `WANGP_ATTENTION`, the CLI value wins (`wgp.py:3304-3305`) and the worker logs `attention_mode_conflict`. Never put `--config` or `--loras` in here. |
| `WANGP_TRANSFORMER_QUANT` | `int8` **(baked)** | Must match what you prefetched, or every cold start re-downloads the transformer. |
| `WANGP_TEXT_ENCODER_QUANT` | `int8` **(baked)** | Same. `gguf_q4_k_m` (14.58 GB) / `gguf_q2_k` (8.49 GB) are the smaller options; the handler warns they "can slightly affect prompt interpretation". |
| `WANGP_CONSOLE` | `1` **(baked)** | `console_output=True`. Set to `0` and WanGP's stdout is swallowed into the event queue instead of the container log. |
| `WANGP_WARM` | `0` | Load the model weights at boot instead of on the first request. Trades first-request latency for the risk of tripping RunPod's ~7-minute worker-start (unhealthy) threshold. |
| `WANGP_WARM_STRICT` | `0` | `1` makes a failed warm fatal instead of a logged warning. |
| `WANGP_WARM_MODEL` | `0` | **Deprecated alias for `WANGP_WARM`.** Still honoured (with a warning) so an existing endpoint config keeps working; use `WANGP_WARM`. |
| `WANGP_REQUIRE_FULL_WEIGHTS` | `0` | `1` makes a non-`EXPECTED` download status fatal at boot, not just a warning. Core files are always fatal. |
| `WORKER_SKIP_WEIGHT_CHECK` | `0` | `1` skips the boot weight gate. Debug only — a missing file then downloads on the clock. |
| `WANGP_STRICT_ATTACHMENT_KEYS` | `1` | Fail boot if upstream `wgp.ATTACHMENT_KEYS` grew keys our forbidden-key list does not cover (that gap is an arbitrary local-file read). Set `0` only to unblock an upgrade you are actively fixing. |
| `WANGP_RESULT_TIMEOUT_S` | `10` | How long the engine waits for the result object after the event queue closes. |
| `WANGP_LOG_TAIL` | `400` | Ring-buffer size of the WanGP log tail the engine keeps per job. |
| `WORKER_VRAM_LEAK_MB` | `4096` | Recycle the worker when the post-`empty_cache()` VRAM floor drifts this far above its baseline (after ≥3 jobs). `0` disables. |

### Budgets and limits

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_DEFAULT_BUDGET_S` | `1400` | Wall-clock budget when the request does not set `runtime.timeout_s`. |
| `WANGP_MAX_BUDGET_S` | `2600` | Ceiling on `runtime.timeout_s`. A larger request value is clamped and a warning is returned. The floor is 60 s. |
| `WANGP_CANCEL_GRACE_S` | `150` | After a budget overrun the worker calls `job.cancel()` and waits this long. Cancel lands at the model's next interrupt check — one denoising step. If it does not land, the process is permanently poisoned (`backend_fatal` + recycle). |
| `WANGP_PROGRESS_INTERVAL_S` | `5` | Minimum seconds between `progress_update` frames. |
| `WANGP_MAX_FRAMES` | `362` | Hard cap on `video_length` (362 frames ≈ 15.1 s at 24 fps). **This is worker-imposed and load-bearing:** `frames_maximum` exists only for Ref2VA (737); FL2VA has no cap anywhere in the headless path, so `video_length: 100000` would otherwise burn a GPU for the full execution timeout. There is **no "unlimited"**: `0`, a negative number or anything unparseable logs a warning and falls back to `362`. |
| `WANGP_MAX_MEDIA_ITEMS` | `16` | Hard cap on the number of attachments in one request, over every `media` key. The byte budget counts bytes, not entries, so thousands of tiny valid images fit inside it and would still cost thousands of files on the container disk. MiniMax H3 takes one reference image (FL2VA) or nine (Ref2VA). |
| `WANGP_MAX_STEPS` | `100` | Hard cap on `num_inference_steps`, same class of hole. |
| `WORKER_FAILURE_BUDGET` | `3` | Consecutive failures before the worker asks to be recycled. |

### Input media

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_B64_IN_MAX` | `6291456` (6 MB) | Per-item cap on inline (`b64`/`url`) input bytes. |
| `WANGP_MEDIA_TOTAL_MAX` | `7340032` (7 MB) | Total inline input bytes per request. RunPod's `/run` envelope is 10 MB and base64 inflates by 4/3. |
| `WANGP_VOLUME_IN_MAX` | `2147483648` (2 GiB) | Per-item cap for `{"volume": …}` inputs. These are already on the volume, so they are **not** charged against the inline budget. |
| `WANGP_VOLUME_TOTAL_MAX` | `8589934592` (8 GiB) | Cap on the **sum** of `volume` input bytes for one request. `image_refs` is list-valued, so the per-item cap alone let N entries copy N × 2 GiB into the container-local job dir. |
| `WANGP_VOLUME_INPUT_SUBDIR` | `inputs` **(baked)** | Sub-directory of the volume that `{"volume": …}` paths resolve against, so `outputs/`, `ckpts/` and `loras/` are out of a request's reach. `""` restores whole-volume access — single-tenant endpoints only. |
| `WANGP_HASH_MAX` | `67108864` (64 MB) | Largest input file that gets sha256'd for the log line. |
| `ALLOW_URL_INPUTS` | `0` **(baked)** | `1` lets requests pass `{"url": "https://…"}` **for input media**. It has never governed `activated_loras`, which is why remote LoRAs are now refused outright rather than being one env var away from a fetch primitive. Turning this on means owning an SSRF guard forever; the worker implements one (scheme/port allow-list, DNS resolved once and the socket pinned to that IP, per-hop redirect revalidation, streaming byte cap, magic-byte typing), but for a multi-tenant endpoint the right answer is to leave it off. |
| `WANGP_URL_SCHEMES` | `https` | Allowed URL input schemes. |
| `WANGP_URL_PORTS` | `80,443` | Allowed URL input ports. |
| `WANGP_URL_TIMEOUT_S` | `60` | Per-fetch timeout. |
| `WANGP_URL_MAX_REDIRECTS` | `3` | Redirect hops; every hop is re-validated. |
| `ALLOW_URL_PRIVATE_HOSTS` | `0` | Test-only escape hatch that permits private/loopback/link-local targets. Logs a warning every time. Also affects presigned PUT targets — a MinIO endpoint on a private address needs this. |
| `WANGP_JOB_SWEEP_AGE_S` | `3600` | Age above which stale job scratch directories are swept at boot. |

### Output delivery

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_OUTPUT_CHAIN` | `presigned,rp_bucket,base64` **(baked)** | The `auto` transport order. `volume` is implemented and can be inserted here or requested explicitly, but is not in the default chain: RunPod's volume S3 API cannot presign, so a volume "success" hands a remote caller a path they cannot fetch. |
| `WANGP_B64_OUT_MAX` | `6291456` (6 MB) | Largest file the base64 transport will inline. Over it, and with nothing else configured, the response is a structured `output_too_large` — never a truncated payload. |
| `BUCKET_ENDPOINT_URL` | *(unset)* | S3-compatible endpoint. All three of endpoint/key/secret must be set for the `rp_bucket` transport to be attempted at all. |
| `BUCKET_ACCESS_KEY_ID` | *(unset)* | |
| `BUCKET_SECRET_ACCESS_KEY` | *(unset)* | |
| `BUCKET_NAME` | *(unset)* | Required for the direct-boto3 uploader and for the bucket half of the idempotency probe (the volume half needs no credentials). Note `rp_upload`'s own default bucket name is the current `%m-%y`, which is never what you want. |
| `BUCKET_REGION` | `us-east-1` | |
| `WANGP_S3_DIRECT` | `0` | `1` uploads with boto3 directly instead of `runpod`'s `rp_upload`. Sets `ContentType` and honours `WANGP_S3_EXPIRES_S`. Used automatically when the `runpod` package is absent. (The idempotency probe always talks to boto3 directly, whatever this is set to.) |
| `WANGP_S3_PREFIX` | `wangp` | Key prefix. Object keys are `<prefix>/<model_type>/<job_id or idempotency_key>.mp4`. |
| `WANGP_S3_EXPIRES_S` | `604800` (7 d) | Presigned-GET lifetime for the direct uploader. `rp_upload` hardcodes 604800 and ignores this. |
| `WANGP_S3_PUBLIC_BASE_URL` | *(unset)* | If set, the response carries `<base>/<key>` instead of a presigned URL (for a public bucket or a CDN in front of it). |
| `WANGP_PUT_TIMEOUT_S` | `600` | Timeout for a caller-supplied presigned PUT. |
| `WANGP_FFPROBE` | *(auto)* | Explicit `ffprobe` path. Otherwise `PATH`, then `$WANGP_ROOT/ffmpeg_bins/ffprobe`. A missing probe degrades to `video.probe_error`; it never fails a job. |
| `REQUIRE_BUCKET` | `0` | `1` makes the worker fail its fitness check (and be replaced) when the `BUCKET_*` trio is incomplete. Use it on any endpoint whose outputs are too large for base64 — otherwise the failure surfaces per job as `output_too_large`. |

### Security

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_ALLOWED_LORAS` | *(empty, baked)* | Comma-separated **basenames** a request may put in `activated_loras`. **Empty means no caller-supplied LoRAs at all** (not "no allow-list"). Absolute paths, `..` and every `scheme://` entry are rejected regardless — the only exception is a URL a shipped accelerator profile contributes verbatim. |

### Handler and observability

| Variable | Default | Meaning |
|---|---|---|
| `WANGP_EAGER_BOOT` | `auto` | `auto` = boot (`import wgp`, weight gate) at import when running as `__main__` or when `RUNPOD_WEBHOOK_GET_JOB` is set, so a pytest import stays free. `1` forces it, `0` defers to the first job. |
| `WORKER_FITNESS` | `1` | Register the SDK fitness checks (boot, CUDA, weights, transport). A failing check exits the worker so the platform marks it unhealthy and replaces it. |
| `WORKER_SKIP_GPU_FITNESS` | `0` | `1` drops the `torch.cuda.is_available()` check — for a CPU smoke test of the container. It also defaults `RUNPOD_SKIP_GPU_CHECK=true`, because the SDK auto-registers its own GPU check (`rp_fitness.py:242` → `rp_gpu_fitness.auto_register_gpu_check`) that ours does not control. |
| `WORKER_DEBUG_DETAILS` | `0` | `1` puts the boot / unhandled-exception traceback tail in the client-facing `details`. Off by default: those lines name container paths and internal frames. They go to the structured log either way. |
| `WORKER_REQUIRE_DELIVERABLE` | `0` | `1` rejects a request in the first second when the only transport left is base64 (no `output.presigned_url`, no bucket, no `volume` in the chain). Default is a warning on the response instead, because a small output really does fit inline. |
| `WORKER_IDEMPOTENCY` | `1` | Probe the derived object key before generating. A retry (yours or RunPod's `/retry`) then returns the existing object at zero GPU seconds. The key is **scoped by a digest of the request**, not by `idempotency_key` alone, so a guessed or reused key can never return someone else's video. Only transports the current request can consume are probed: a HEAD against the bucket when the `BUCKET_*` trio **and** `BUCKET_NAME` are set and `rp_bucket` is in the chain, then `outputs/<key>` when `volume` is in the chain. `output.mode` of `presigned` or `base64` never hits. A hit is marked `video.cached: true` and carries no ffprobe block — nothing was generated to probe. |
| `WORKER_KEEP_OUTPUTS` | `0` | `1` keeps generated files in `WANGP_OUTPUT_DIR` after delivery. Debug only; the disk is not swept. |
| `WORKER_LOG_TAIL` | `30` | Lines of WanGP output attached to a failure response as `logs_tail`. |
| `WORKER_ERROR_OBJECT` | `0` | `1` replaces the plain `error` message with a **JSON-encoded string** `{"code","message","retryable","details"}`. Encoded, not a dict: `rp_job.run_job` assigns `run_result["error"] = error_msg` with no `str()` (`rp_job.py:266-273`), so a dict would reach the result endpoint as an object in a field that is a string everywhere else. Off by default. |
| `WORKER_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error`. |
| `WORKER_LOG_MAX_FIELD` | `8192` | Per-field truncation in the JSON log lines. |
| `RUNPOD_POD_ID`, `RUNPOD_ENDPOINT_ID` | set by the platform | Echoed into every log line as `worker_id` / `endpoint_id`. |

---

## Request schema

```jsonc
{
  "input": {
    "model_type": "minimax_h3_fl2va_pruned",  // optional; must equal WANGP_MODEL_TYPE
    "prompt": "…",                             // convenience alias for settings.prompt
    "profile": "Turbo Lightx2v FL2V 4 Steps v1.0 768p",   // optional accelerator profile
    "settings": { /* any subset of the model's settings universe */ },
    "media":    { /* attachment key -> {"b64"|"volume"|"url"}; list for image_refs */ },
    "output":   { "mode": "auto", "presigned_url": null, "content_type": "video/mp4" },
    "runtime":  { "timeout_s": 1400, "idempotency_key": null, "priority": 0 }
  },
  "webhook": "https://your.app/wangp-done",
  "policy":  { "executionTimeout": 3600000 }
}
```

**Merge order** (later wins): `get_default_settings(model_type)` → accelerator-profile
fragment → `input.settings` → `input.prompt` (only if `settings.prompt` was absent) →
materialized media paths → worker pins (`model_type`, `config`, resolved `seed`,
`batch_size=1`, `repeat_generation=1`, `image_mode=0`).

### `settings`

Any key in the model's settings universe — `models/_settings.json` (112 keys) ∪
`get_default_settings(model_type)`. Anything else is `unknown_setting`.

These are **forbidden** and raise `bad_request`: all 15 WanGP attachment keys
(`image_start`, `image_end`, `image_refs`, `image_guide`, `image_mask`, `video_guide`,
`video_guide2`, `video_mask`, `video_source`, `custom_guide`, `audio_guide`,
`audio_guide2`, `audio_source`, `replace_voice_sample`, `replace_voice_sample2`), plus
`mode`, `_api`, `client_id`, `state`, `type`, `base_model_type` and `priority`, plus the
five post-processing selectors `postprocess_audio`, `prompt_enhancer`,
`replace_voice_method`, `spatial_upsampling` and `temporal_upsampling`.
Media may come **only** from `input.media`; a path in `settings` would be handed straight
to WanGP as an arbitrary local file read, and `mode` flips WanGP's validator into an edit
branch that reads `video_source` off disk. The post-processing five are forbidden because
`download_requested_postprocessing_assets` (`wgp.py:3532`) runs *inside* `generate_media`
(`wgp.py:6786`) — on the clock, pulling assets the boot-time weight gate never proved were
present, and loading a second model into the same GPU.

Pre-flight rules the worker enforces on CPU, before anything loads:

| Rule | Behaviour |
|---|---|
| `video_length` | Floored onto the model's lattice (`107 + 17·n`: 107, 124, 141, 158, …) with a warning, then rejected if still above `WANGP_MAX_FRAMES`. `363 → 362`; `100000 → invalid_setting`. |
| `sliding_window_overlap` | **Rounded to nearest** legal value, matching `normalize_overlap` — `30 → 35`, not `18`. `0` is legal and left alone. |
| `sliding_window_size` | Floored onto the lattice and clamped to `124…481`. **`0` is rejected**: it does not disable sliding windows (`wgp.py:6930` only skips the flooring, and both variants declare `sliding_window: true`), it schedules nine zero-length windows with a negative overlap. To generate in one window, set `video_length ≤ sliding_window_size`. |
| `num_inference_steps` | Capped at `WANGP_MAX_STEPS`. |
| `resolution` | Must be on the model's block grid (32 for MiniMax H3); the error names the nearest valid size. |
| `sample_solver` | Must be one of the model's declared solvers. |
| `skip_steps_cache_type` / `skip_steps_multiplier` | `first_block` requires a multiplier in `0.06 / 0.08 / 0.10 / 0.12 / 0.14`. |
| `seed` | Any negative value (or absent) resolves to a random seed in `[1, 999999999]`, and the resolved value is echoed back so the job is reproducible. |
| `activated_loras` | Basename allow-list (`WANGP_ALLOWED_LORAS`, empty ⇒ none accepted); no absolute paths, no `..`, and **no URLs of any scheme** except one a shipped accelerator profile contributes verbatim. |
| flag letters | `image_prompt_type` ⊆ `TSEVL`; `video_prompt_type` and `audio_prompt_type` are checked against the variant's own whitelists — for FL2VA that is `guide_custom_choices` (`GVKFI`) **plus the mask group** (`A`, `N`: "Masked Area" / "Non Masked Area"), for Ref2VA `PDEV+-` plus `KI`. Every "you must provide a …" rule is pre-flighted (`S`→`image_start`, `E`→`image_end`, `I`→`image_refs`, `V`→`video_guide`, `V`+`A`→`video_mask`, `V`+`+`→`video_guide2`, `A`→`audio_guide`, `B`→`audio_guide2`), with the same nesting WanGP uses. |
| `media` count | At most `WANGP_MAX_MEDIA_ITEMS` (16) attachments per request, over every key — the byte budget alone cannot stop an entry-count attack. |
| Ref2VA counts | ≤9 image refs, ≤2 videos, ≤2 audio refs, `#audio ≤ #images + #videos`, ≤12 files total. |
| `image_mode` | Must be 0. A non-zero value makes WanGP emit images, which the video transport cannot deliver. |

Duration rules (each reference video ≥2 s and truncated to 15 s, each audio 2–15 s, totals
≤15 s) need ffprobe on the real files, so they stay with WanGP and come back as
`wangp_validation` within seconds — not minutes.

### `media`

```jsonc
"media": {
  "image_start": {"b64": "iVBORw0KGgo…"},              // raw base64 or a data: URI
  "video_guide": {"volume": "clips/plate.mp4",         // relative to $WANGP_VOLUME_ROOT/inputs
                  "range": {"start_frame": 0, "end_frame": 240, "audio_track_no": 1}},
  "audio_guide": {"url": "https://…"},                 // only when ALLOW_URL_INPUTS=1
  "image_refs":  [{"b64": "…"}, {"b64": "…"}]          // the only list-valued key
}
```

String shorthands are accepted: `"volume://clips/plate.mp4"`, `"https://…"`,
`"data:video/mp4;base64,…"`. A **bare** string (`"clips/plate.mp4"`) is refused on
purpose — it would be indistinguishable from a worker-filesystem path.

`volume` paths resolve under **`$WANGP_VOLUME_ROOT/$WANGP_VOLUME_INPUT_SUBDIR`**
(`/runpod-volume/inputs`), not the volume root: the volume is shared and also carries
`ckpts/`, `loras/` and — when `volume` is in the output chain — delivered `outputs/`. Stage
caller-readable material under `inputs/`. `..`, absolute paths, `|`, NUL and symlinks that
leave the directory are all rejected on the realpath, and the file is opened `O_NOFOLLOW`
so the check cannot be raced.

`range` applies to video slots only and becomes WanGP's virtual-media suffix
(`path|start_frame=…,end_frame=…`). `start_frame` is zero-based, `end_frame` is inclusive.

Every attachment's type is decided by **magic bytes**, never by a caller-supplied name,
and then written with the extension WanGP's own whitelist expects
(`shared/utils/utils.py:36-49`). Consequences worth knowing: `.webm` is not on that list,
so WebM is materialized as `.mkv` with a warning; audio-only MP4 (`.m4a`) is rejected
outright (accepted audio is wav/mp3/aac); `.avi` *is* accepted. A PNG sent as
`audio_guide` is `media_unsupported`.

### `output`

| Field | Values |
|---|---|
| `mode` | `auto` (default), `presigned`, `rp_bucket`, `volume`, `base64`. Aliases accepted: `s3`/`bucket`/`rp_upload` → `rp_bucket`, `b64`/`inline` → `base64`, `put`/`presigned_url` → `presigned`, `network_volume` → `volume`. |
| `presigned_url` | An http(s) **PUT** URL. Required when `mode` is `presigned`. |
| `content_type` | Default `video/mp4`. |

In `auto`, transports are tried in `WANGP_OUTPUT_CHAIN` order and a failure falls through
to the next with a logged reason. With an explicit `mode`, a failure is raised at the
point of failure instead of being silently downgraded.

### `runtime`

| Field | Meaning |
|---|---|
| `timeout_s` | Wall-clock budget, clamped to `[60, WANGP_MAX_BUDGET_S]`. Out-of-range values are clamped with a warning, not rejected. |
| `idempotency_key` | 1–128 chars of `[A-Za-z0-9._:-]` starting alphanumeric. Combined with a digest of the request it becomes the object key, so a retry of the **same** request returns the existing object at zero GPU seconds while the same key on a *different* request generates normally. |
| `priority` | `0`–`9`, accepted and **inert**: one generation per worker means there is no queue to order. A warning says so on the response. Scale with `max_workers`. |
| `priority` | 0–9. Accepted and echoed; the worker itself is FIFO at concurrency 1. |

### Worked example

```bash
cat > job.json <<'JSON'
{"input": {
  "model_type": "minimax_h3_fl2va_pruned",
  "profile": "Turbo Lightx2v FL2V 4 Steps v1.0 768p",
  "settings": {
    "prompt": "integrated_multimodal_description: [Shot 1] A five-second cinematic single take inside a rain-lashed glass observatory at midnight. A radio astronomer leans toward a brass receiver and says clearly (S1) <d>[English] If you can hear me, follow this signal.</d>\noverall_soundscape: Rain on the dome, a low electrical hum, three clean receiver tones, and her synchronized voice.\nnon_diegetic_music: One quiet bowed-glass chord rising and fading.",
    "resolution": "832x480",
    "video_length": 124,
    "sample_solver": "euler",
    "image_prompt_type": "",
    "video_prompt_type": "",
    "audio_prompt_type": "",
    "sliding_window_size": 362,
    "sliding_window_overlap": 18,
    "seed": 918273645
  },
  "output":  {"mode": "auto"},
  "runtime": {"timeout_s": 900, "idempotency_key": "demo-observatory-001"}
}}
JSON

ID=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H 'Content-Type: application/json' -d @job.json | jq -r .id)
echo "job $ID"

# poll (or set "webhook" in the request and skip this)
while :; do
  R=$(curl -s "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$ID" \
        -H "Authorization: Bearer $RUNPOD_API_KEY")
  S=$(jq -r .status <<<"$R")
  echo "$S $(jq -c '.output.phase // empty, .output.pct // empty' <<<"$R" | paste -sd' ')"
  case "$S" in COMPLETED|FAILED|CANCELLED|TIMED_OUT) echo "$R" | jq .; break;; esac
  sleep 5
done

# cancel -- see the caveat below: this ends the JOB, not the generation
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/cancel/$ID" \
     -H "Authorization: Bearer $RUNPOD_API_KEY"
```

#### Cancellation, precisely

`POST /cancel/{id}` reaches the worker: the SDK long-polls a stop channel and calls
`task.cancel()` on the asyncio task running the handler. The handler is `async` and hands
the body to `asyncio.to_thread` exactly so that poll loop keeps running — a synchronous
handler starves it for the whole multi-minute job and the signal is never seen.

`task.cancel()` cancels the *await*, not the thread, and Python cannot kill a running
thread. So the handler translates it into a cooperative signal: it shields the worker
thread's task, sets a `threading.Event`, and passes `Event.is_set` to `engine.run` as its
`cancel_check`, which the drain loop polls every 0.5 s. The engine then calls
`job.cancel()` — WanGP's abort flag plus `wan_model._interrupt` — and the model stops at
its next interrupt check, within one denoising step.

Two consequences worth knowing:

- **Cancellation is cooperative, so it is not instantaneous.** You are billed until the
  model reaches its next interrupt check. If it never does within `WANGP_CANCEL_GRACE_S`,
  the process is declared poisoned (`backend_fatal` + `refresh_worker`), because the WanGP
  worker thread is a daemon that cannot be killed and still holds the process-wide
  generation lock.
- **The result of a cancelled job is discarded by the platform**, so a `refresh_worker` in
  that response never reaches it. The budget path is what you should rely on for cost.

Practical consequence: **`runtime.timeout_s` is still your primary cost control.** Set it
to what you are actually willing to pay for; `/cancel` is the interactive escape hatch.

`124 = 5 + 17·7` is on the lattice, so nothing is floored. The profile supplies the turbo
LoRA, `num_inference_steps: 4`, `guidance_scale: 1` and `flow_shift: 6` — **the turbo
LoRA is opt-in per request**, not baked into the endpoint. Drop `"profile"` and the same
job runs at the model's default step count and costs roughly 3.5× as much.

More worked examples — first+last frame, control video with a new soundtrack, Ref2VA with
reference images and audio — are in `docs/RUNPOD_SERVERLESS.md` under "Worked examples".
Note that Ref2VA has **no shipped accelerator LoRA** in this repo: it runs at 20 steps.

---

## Response schema

RunPod wraps whatever the handler returns: `rp_job.run_job` **pops** `error` and
`refresh_worker`, puts the rest under `output`, sets `status: FAILED` when `error` is
truthy and `stopPod: True` when `refresh_worker` is.

### Success

```json
{"delayTime": 1842, "executionTime": 131940, "id": "60902e6c-…-u1", "status": "COMPLETED",
 "output": {
   "status": "completed",
   "model_type": "minimax_h3_fl2va_pruned",
   "model": {"model_type": "minimax_h3_fl2va_pruned", "name": "MiniMax H3 FL2VA Pruned 20B",
             "profile": "Turbo Lightx2v FL2V 4 Steps v1.0 768p", "config": ""},
   "seed": 918273645,
   "video": {
     "transport": "rp_bucket", "kind": "url",
     "url": "https://bucket.s3.…/wangp/minimax_h3_fl2va_pruned/demo-observatory-001.mp4?X-Amz-…",
     "expires_in_s": 604800, "uploader": "rp_upload", "upload_s": 2.14,
     "key": "wangp/minimax_h3_fl2va_pruned/demo-observatory-001.mp4",
     "filename": "2026-08-18-14h22m01s_seed918273645_….mp4",
     "size_bytes": 8412663, "bytes": 8412663, "content_type": "video/mp4",
     "sha256": "9f2c…",
     "container": "mp4", "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
     "duration_s": 5.167, "fps": 24, "width": 832, "height": 480,
     "video_codec": "h264", "has_video": true,
     "has_audio": true, "audio_codec": "aac", "audio_sample_rate": 32000, "audio_channels": 2
   },
   "resolved": {"model_type": "minimax_h3_fl2va_pruned", "seed": 918273645,
                "video_length": 124, "num_inference_steps": 4, "resolution": "832x480",
                "flow_shift": 6, "sample_solver": "euler",
                "sliding_window_size": 362, "sliding_window_overlap": 18},
   "warnings": ["accelerator profile 'Turbo Lightx2v FL2V 4 Steps v1.0 768p' applied before input.settings"],
   "metrics": {"validate_ms": 12, "inputs_ms": 61, "input_bytes": 0, "input_files": 0,
               "generate_s": 128.4, "upload_s": 2.14, "transport": "rp_bucket",
               "total_s": 131.9,
               "phase_marks_s": {"loading_model": 0.4, "encoding_text": 61.2,
                                 "inference_stage_1": 78.5, "decoding": 118.9},
               "jobs_served": 7, "boot_ms": 41002, "warm_ms": 0,
               "consecutive_failures": 0, "vram_floor_mb": 5312.0, "vram_peak_mb": 21874.5},
   "worker_id": "abc123"
 }}
```

Everything in `video` from `container` down comes from **ffprobe on the produced file**,
not from the request. That matters for Ref2VA `KI`, where the first reference image
defines the output dimensions — the `resolution` you sent may not be the one you got.
If ffprobe is unavailable the block is replaced by `"probe_error": "…"`; the job still
succeeds.

For `mode: "base64"` the transport block is `{"transport": "base64", "kind": "base64",
"encoding": "base64", "data": "<base64>"}` instead of a URL. For `presigned`, the returned
`url` has its query string stripped — the signature is a write credential and is never
echoed into a response or a log.

`refresh_worker: true` may appear on a **successful** response too: it means this worker
served you and is now retiring (failure budget, VRAM drift). It is not an error.

An idempotent hit (same `runtime.idempotency_key`, object already in the bucket or on the
network volume) returns
the same envelope with `metrics.idempotent_hit: true`, `video.cached: true`, no ffprobe
block, and an `executionTime` of a few hundred milliseconds.

### Progress

Readable mid-flight from `/status`. Each frame **overwrites** the previous one — it is a
status field, not an append-only log.

```json
{"status": "IN_PROGRESS",
 "output": {"phase": "inference_stage_1", "status": "Prompt 1/1 | Denoising | 7.2s",
            "pct": 44, "step": 2, "total_steps": 4, "elapsed_s": 63.4, "eta_s": 71.0}}
```

`phase` is one of `loading_model`, `encoding_text`, `inference_stage_1/2/3`, `decoding`,
`downloading_output`, `cancelled`, or the fallback `inference`. `pct` is an **estimate**,
banded per phase, and never reaches 100 — completion is signalled by the job status, not
by `pct`. `eta_s` appears once two steps of the same phase have been observed.

### Failure

```json
{"status": "FAILED", "id": "…",
 "error": "[validation] MiniMax H3 frame injection requires one position per Reference Image (found 0 positions and 2 images)",
 "output": {"status": "error",
            "error_message": "[validation] MiniMax H3 frame injection requires …",
            "error_code": "wangp_validation",
            "retryable": false,
            "details": ["[validation] MiniMax H3 frame injection requires …"],
            "logs_tail": ["status: Loading model MiniMax H3 FL2VA Pruned 20B…"],
            "model_type": "minimax_h3_fl2va_pruned",
            "seed": 4242,
            "metrics": {"validate_ms": 9},
            "worker_id": "abc123"}}
```

**Branch on `error_code`, never on message text.**

| `error_code` | Meaning | Retryable | Recycles the worker |
|---|---|---|---|
| `bad_request` | Malformed payload, forbidden key, wrong `model_type` for this endpoint | no | no |
| `unknown_setting` | A settings key outside this model's universe | no | no |
| `invalid_setting` | A known key with an out-of-range or cross-field-invalid value | no | no |
| `media_too_large` | An attachment, or the sum of them, exceeded the byte cap | no | no |
| `media_fetch_failed` | An attachment could not be materialized (bad base64, missing volume path, fetch failure) | **yes** | no |
| `media_unsupported` | Sniffed content type is not accepted for that slot | no | no |
| `ssrf_blocked` | A URL input resolved to a blocked network destination | no | no |
| `weights_missing` | Weights are incomplete on this worker (normally caught at boot) | no | no |
| `wangp_validation` | WanGP's own validator rejected the settings | no | no |
| `generation_failed` | The generation ran and failed | only when poisoned | when poisoned |
| `timeout` | Exceeded the request's wall-clock budget and was cancelled | **yes** | after the failure budget |
| `cancelled` | WanGP reported the generation aborted without a budget overrun (worker drain/shutdown, internal abort) | **yes** | no |
| `no_output` | WanGP reported success and produced no video file | no | **no** |
| `output_too_large` | No configured transport can carry the result | no | no |
| `upload_failed` | An upload was attempted and did not yield a real URL | **yes** | no |
| `worker_busy` | A generation was already in flight (unreachable at concurrency 1) | **yes** | no |
| `backend_fatal` | A cancel never landed; a daemon thread still holds WanGP's generation lock | **yes** (a fresh worker will serve it) | **yes** |
| `oom` | CUDA OOM or an equivalent poisoned-device condition | **yes** | **yes** |
| `internal_error` | Unhandled worker error — this is a bug, please report it | **yes** | no |

`no_output` deserves its own note: it is a *configuration* refusal, not a poisoned
process. `generate_media` returns success with no file on several paths — most commonly
an attention mode the device does not support, where WanGP emits an `info` message and
exits the task cleanly. **The explanation is in `logs_tail`.** Read it before retrying.

---

## Testing

### Tier 1 — CPU only. No GPU, no torch, no weights.

`schema.py`, `media_in.py`, `media_out.py`, `config.py`, `errors.py` and `obs.py` import
nothing from WanGP or torch, and `handler.py` defers every heavy import. That split is the
whole reason CI runs on a plain runner.

```bash
pip install pytest
pytest runpod_worker/tests -q          # 291 tests, ~3 s
```

This is `.github/workflows/worker-ci.yml`, plus a hadolint pass over the Dockerfile. It
needs no network, no `requirements.txt`, and no `runpod` package.

What it protects, in rough order of value:

- **`test_wgp_config_drift`** text-scans `wgp.py` for unguarded `server_config["…"]` reads
  and asserts our config covers them. This is the regression test for the one bug that
  stops the worker booting at all (see Troubleshooting #1).
- **`test_attachment_keys_match`** parses the `ATTACHMENT_KEYS` literal out of `wgp.py`
  with `ast.literal_eval` and compares it to ours — an upstream addition would otherwise
  open an arbitrary-local-file read.
- Frame math (`30 → 35` catches a floor-instead-of-round implementation), forbidden keys,
  LoRA guards, cross-variant flag rules, the `video_length` cap.
- Magic-byte typing including the real MP3/AAC sync-word variants (`0xFFFA`, `0xFFF3`,
  `0xFFF9`) — a naive two-byte table rejects ordinary files.
- **`test_rp_upload_local_fallback_is_caught`**: `rp_upload` returns a `local_upload/…`
  path instead of raising when it cannot build a client. Silently returning that to a
  caller is the single most likely data-loss bug in this design.
- **`tests/test_handler.py`** drives `handler.run_job` end to end with a stubbed engine
  and the *real* schema/media/config modules: the success envelope, every error code and
  its `retryable`/`refresh_worker` pairing, media materialization, scratch-dir and
  output-file cleanup, the idempotent replay, and `test_input.json` itself. This is what
  catches interface drift between the modules, which no single-module test can see.
- **`tests/test_engine.py`** drives the event drain loop with a fake `SessionJob`: the
  termination condition (`job.done` *and* `events.closed` *and* a drained queue, because
  the error text arrives after `_set_result`), the cooperative cancel on budget overrun,
  the `backend_fatal` latch when a cancel never lands, the poison-marker scan, and the
  between-jobs truncation of the lists WanGP appends to forever.

### Tier 2 — GPU, no RunPod

```bash
# One-shot. The SDK reads test_input.json from the PROCESS CWD, hardcoded.
cd /opt/wangp/runpod_worker && python3 handler.py

# Or explicitly, from anywhere:
python3 /opt/wangp/runpod_worker/handler.py \
  --test_input "$(cat /opt/wangp/runpod_worker/test_input.json)"

# Local FastAPI. /run does NOT execute the handler locally — only /runsync does.
python3 /opt/wangp/runpod_worker/handler.py --rp_serve_api --rp_api_host 0.0.0.0
curl -X POST localhost:8000/runsync -H 'Content-Type: application/json' \
     -d @runpod_worker/test_input.json | jq '.output.metrics'
```

Assert on the result: `output.video.has_audio == true`, `audio_sample_rate == 32000`,
`video_codec == "h264"`, `fps == 24`, and that the decoded file actually plays.

Note that fitness checks do **not** run on the local path — `worker._is_local`
short-circuits before them. A boot failure surfaces as `SystemExit(1)` from `main()` or as
a per-job `backend_fatal` instead.

### Tier 3 — container, before pushing

```bash
docker build --platform linux/amd64 \
  --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0" \
  -f runpod_worker/Dockerfile -t you/wangp-h3:2026.08.18-1 .

# The weight gate. This is the step that catches a filename mismatch between what you
# prefetched and what the handler will ask for.
docker run --rm --gpus all -v /path/to/ckpts:/runpod-volume/ckpts \
  you/wangp-h3:2026.08.18-1 \
  python3 -u -m runpod_worker.scripts.verify_weights minimax_h3_fl2va_pruned

# One real generation on the same image.
docker run --rm --gpus all --env-file .env.staging \
  -v /path/to/ckpts:/runpod-volume/ckpts \
  -v $PWD/runpod_worker/test_input.json:/opt/wangp/runpod_worker/test_input.json \
  -w /opt/wangp/runpod_worker you/wangp-h3:2026.08.18-1 python3 -u handler.py
```

### Tier 4 — on RunPod

1. Staging endpoint: `max_workers=1`, idle 180 s, execution timeout 3600 s, volume
   attached, GPU priority `[L40S, A6000, A100-80]`.
2. `POST /run` with `test_input.json`; poll `/status/{id}`; record `delayTime` and
   `executionTime`. **These are the numbers that replace every estimate in this file.**
3. Immediately fire a second, identical request. `executionTime` should drop by the
   weight-load time. If it does not, the model is reloading between jobs — check that
   `model_type`, `profile` and `config` are identical (`wgp.py:6773`), and look for
   `refresh_worker` in the first response.
4. `python3 -m runpod_worker.scripts.calibrate --endpoint $ENDPOINT_ID --matrix
   steps=4,8,20 frames=124,209,362 --repeat 3` → p50/p90 per cell, cold-start
   distribution, $/generation, and a recommended `WANGP_DEFAULT_BUDGET_S` (measured
   p99 × 1.3). `--json` / `--csv` write the raw data. It reads `$RUNPOD_API_KEY` (or
   `--api-key`) and honours `$RUNPOD_API_BASE` for a non-default API host.
5. Chaos pass. Each of these should produce a specific, fast, non-poisoning failure:
   `timeout_s: 60` on a 20-step job (expect `timeout`, not a platform kill); a `.png` in
   `audio_guide` (`media_unsupported`); `video_length: 700` (`invalid_setting`);
   `model_type: "t2v"` (`bad_request`); `audio_prompt_type: "2"` with no `video_guide`
   (`invalid_setting`); unset `BUCKET_*` with `REQUIRE_BUCKET=1` (the worker should die at
   its fitness check rather than return a dead path).
6. Watch the console Metrics tab: delay-time P70/P90/P98, cold-start count, throttled
   workers. If P90 delay exceeds ~400 s, the volume read is the bottleneck — add active
   workers or move to a baked-weights image.

---

## Troubleshooting

### 1. `KeyError: 'attention_mode'` during boot

The hardest-won fact in this deployment. If `wgp_config.json` is **absent**, wgp builds
its own defaults and writes them. If it **exists**, wgp does
`server_config = json.loads(text)` and *replaces* its defaults wholesale, then reads
`server_config["attention_mode"]` as a bare subscript at module scope (`wgp.py:3301`). A
hand-written config that omits one key kills `import wgp` with a stack trace that looks
nothing like a config problem.

`config.ensure_wgp_config()` always writes `attention_mode` (plus the three profile keys),
asserts `REQUIRED_WGP_KEYS`, and merges rather than overwrites so wgp's own migrations
survive a restart. `tests/test_wgp_config_drift.py` re-derives the unguarded-read set from
`wgp.py` on every CI run, so an upstream bump cannot reintroduce this quietly. **If you
hand-edit `wgp_config.json`, do not delete keys.**

### 2. Worker starts, immediately goes unhealthy, and is replaced — in a loop

The weight gate failed. Look for `boot_failed` or `weights_missing` in the container log;
it names the missing files. Run `verify_weights.py` against the same volume from a Pod
(remember `WANGP_VOLUME_ROOT=/workspace` there). Usual causes: the volume is not attached;
it is attached in a different datacenter than the worker; you prefetched with a different
`--transformer-quant` than `WANGP_TRANSFORMER_QUANT`; or the endpoint's `WANGP_MODEL_TYPE`
is not the one you warmed.

`REQUIRE_BUCKET=1` with an incomplete `BUCKET_*` trio also fails a fitness check — by
design.

**Not all fitness checks are ours.** The SDK auto-registers its own memory, disk and
network checks for every worker (`rp_system_fitness.py:29-30`): ≥4 GB free RAM
(`RUNPOD_MIN_MEMORY_GB`) and **≥10 % free disk** (`RUNPOD_MIN_DISK_PERCENT`), plus GPU,
CUDA-version, CUDA-init and GPU-benchmark checks. A failure is `os._exit(1)` with only the
SDK's own log line to explain it — none of this worker's structured logs. With a 20–30 GB
image on a small container disk the disk check is reachable, so if the loop shows no
`boot_failed` at all, raise the endpoint's container disk or lower
`RUNPOD_MIN_DISK_PERCENT`.

### 3. Jobs succeed but return `no_output`

WanGP reported success and wrote no file. **Read `logs_tail`** — the reason is in an
`info` line there. The classic cause is an attention mode the device does not support:
`wgp.py` emits an info message and exits the task cleanly, which counts as success. Set
`WANGP_ATTENTION=sdpa` and `WANGP_CLI_ARGS="--attention sdpa …"` and retry. This is not a
poisoned process, so the worker is deliberately *not* recycled.

### 4. `output_too_large`, or a URL that 404s

The response tells you exactly which env vars fix it. In order of preference: have the
caller pass `output.presigned_url` (no secrets on the worker, best for multi-tenant); or
set `BUCKET_ENDPOINT_URL` + `BUCKET_ACCESS_KEY_ID` + `BUCKET_SECRET_ACCESS_KEY` +
`BUCKET_NAME`; or raise `WANGP_B64_OUT_MAX` if the file genuinely fits in RunPod's 10 MB
envelope (it holds ~7.5 MB of binary after base64 inflation); or add `volume` to
`WANGP_OUTPUT_CHAIN`, which writes the file to the network volume and returns
`volume_path` instead of a URL — only useful when the caller can read that volume out of
band, which is why it is not in the default chain.

**You should not be finding this out after a generation.** On the shipped default
(`presigned,rp_bucket,base64`) with no bucket and no `output.presigned_url`, every
response already carries a warning saying the endpoint can only return outputs under
`WANGP_B64_OUT_MAX` — a 5 s 832×480 clip with audio is plausibly 2–8 MB, i.e. right on the
line. Set `WORKER_REQUIRE_DELIVERABLE=1` to turn that warning into a sub-second rejection,
or `REQUIRE_BUCKET=1` to fail the whole worker at fitness time.

If uploads "succeed" but the URL is unusable, you are probably hitting `rp_upload`'s
silent fallback: it returns a `local_upload/<name>` **path** rather than raising when it
cannot build an S3 client. The worker checks for that and treats it as a failure —
falling through the chain in `auto` mode, raising `upload_failed` when the mode was
explicit. A 403 on the returned URL usually means `BUCKET_REGION` or the endpoint URL is
wrong.

### 5. `backend_fatal` and the worker restarts

A cancel did not land within `WANGP_CANCEL_GRACE_S`. The WanGP generation runs on a daemon
thread that cannot be killed and still holds a process-global lock, so the process is
permanently unusable and the worker asks to be replaced. The client should retry — a
fresh worker will serve it. If this is frequent, your budget is too tight for the config
you are running: measure with `calibrate.py` and raise `WANGP_DEFAULT_BUDGET_S`.

### 6. Every job reloads the model (`executionTime` never drops)

`wgp.py:6773` reloads everything when `model_type`, `profile` or `config` differs from
what is loaded. Check that requests are not varying `model_type` or `settings.config`
(both are rejected by default — `ALLOW_MODEL_SWITCH=0`), that the endpoint's
`WANGP_MODEL_CONFIG` has no trailing comma, and that the previous job did not return
`refresh_worker: true`. Also check the idle timeout: at 5 s (the platform default) the
worker is torn down between requests and every job is a cold start.

### 7. LoRA is downloaded at request time, or "not staged on this endpoint"

`get_lora_local_path` maps an `https://` entry in `activated_loras` to
`lora_dir/basename(url)`. Stage the file under exactly that basename, in
`$WANGP_VOLUME_ROOT/loras/minimax_h3/`, and make sure `loras_root` is absolute (it is, by
default — `get_lora_dir()` returns a *relative* path, which is why the config must set an
absolute root). If you get `bad_request: LoRA '…' is not staged`, the basename is not in
`WANGP_ALLOWED_LORAS`; the error lists what is allowed.

A **caller** cannot name a URL at all (`bad_request: remote LoRA URLs are not accepted`).
Only a shipped accelerator profile may, and only verbatim — `check_loras_exist` downloads
whenever the local file is absent (`wgp.py:3697-3706`), so a caller-supplied URL is a
fetch-and-load primitive pointed at the shared volume. If the profile's own LoRA is not
staged you get a warning on the response saying WanGP will download it during the
generation; stage it with `prefetch_weights.py` and the warning goes away.

> **Privacy note.** The worker never logs the prompt or the settings dict. WanGP does:
> `WANGP_CONSOLE=1` (the baked default) routes its stdout to the container log, and
> `wgp.py:7276` prints the full enhanced prompt when prompt enhancement is on — which is
> one more reason `prompt_enhancer` is a forbidden key here. Container logs are visible to
> anyone with access to the RunPod account.

### 8. Blank container logs for the whole generation

`shared/api_cli.py` installs a process-global `redirect_stdout` for the length of a job.
The worker's own logger writes to `sys.__stdout__`, captured at import, precisely so it
survives that — and `WANGP_CONSOLE=1` (default) keeps WanGP's own output going to the
console rather than only into the event queue. If you set `WANGP_CONSOLE=0` you will see
nothing for 5–25 minutes at a stretch.

### 9. Nothing generates: `bad_request: this endpoint is pinned to '…'`

One `model_type` per endpoint, by design. Deploy a second endpoint, or set
`ALLOW_MODEL_SWITCH=1` and accept a ~200 s billed reload on every switch.

### 10. `/runsync` times out at 5 minutes

Expected. Use `/run` + `/status`, or a webhook. `?wait=` maxes out at 300000 ms.

### 11. A request that worked yesterday returns `unknown_setting` after an upgrade

The settings universe is derived from `models/_settings.json` ∪
`get_default_settings(model_type)` at runtime, so an upstream rename shows up here first.
Compare against `resolved` in a recent successful response, and check
`tests/test_schema.py` — it re-derives the key list from source on every CI run.

### 12. Frames or overlap came back different from what I sent

By design, and reported in `warnings`. `video_length` is floored onto the model's lattice
(`107 + 17·n`) and `sliding_window_overlap` is rounded to the *nearest* legal value — the
same arithmetic WanGP applies internally, done up front so the `resolved` block you get
back is the truth rather than your request.

### 13. Read-only rootfs / permission errors on boot

`import wgp` does `os.mkdir("settings")`, writes `wgp_config.json`, `json.dump()`s
`settings/<model_type>_settings.json` on first `get_default_settings`, and writes
`loras_url_cache_v2.json` — all relative to the repo root. The image chowns
`/opt/wangp` to uid 1000 for exactly this reason. A read-only root filesystem does not
work.

### 14. The endpoint stopped scaling

See "Operational chores" — an endpoint with no requests for 3 days has max workers
auto-reduced to 2, and to **0 after 7 days**, and stays reduced until raised by hand.

### 15. A new dotfile silently never lands in a commit

`.gitignore:1` is `.*`. `.dockerignore` and `.github/workflows/worker-ci.yml` survive only
because they were force-added; anything *new* under `.github/` (or any other dotfile) is
skipped by `git add` with no error. Symptoms: a workflow that never runs, or a build from
a fresh clone whose context is tens of GB larger than a local one because `.dockerignore`
did not come along. Add `!.dockerignore` and `!.github/` to `.gitignore`.

---

## Operational chores

**Rollback is by tag.** Endpoint → Manage → Edit Endpoint → change the image tag → save.
It rolls out as workers recycle. This only works because tags are immutable — never use
`:latest`, and never re-push an existing tag.

**Weekly synthetic job on any low-traffic endpoint.** RunPod auto-reduces max workers to 2
after 3 idle days and to 0 after 7, and does not raise them again by itself. One scheduled
`test_input.json` run per week prevents it and doubles as a canary.

**Alerts worth having** (all visible in the console Metrics tab): FAILED rate above 2 %,
a cold-start count spike, delay p90 above 60 s, sustained throttled workers, and a failed
weekly synthetic job.

**Recycling is normal.** `refresh_worker` fires on a poisoned generation (OOM, a cancel
that never landed), after `WORKER_FAILURE_BUDGET` consecutive failures, and on VRAM drift
past `WORKER_VRAM_LEAK_MB`. Each one costs a cold start, so a *rising* recycle rate is the
signal, not the presence of any at all.

**Log lines are single-line JSON** on stdout with `event`, `job_id`, `worker_id` and
`endpoint_id`. The useful ones: `boot_start` / `boot_complete` / `boot_failed`,
`request_validated`, `wangp_error`, `transport_failed`, `output_delivered`,
`job_completed` / `job_failed`, `recycling_after_success`.

**After a WanGP upgrade**, run Tier 1 before anything else: `test_wgp_config_drift` and
`test_attachment_keys_match` are specifically designed to fail loudly on the two upstream
changes that would otherwise break boot or open a file-read hole.

---

## Cost notes (estimates)

> **Everything in this section is an estimate, not a measurement.** Nothing in this repo
> or its README states a MiniMax H3 generation wall-clock; the README quotes VRAM only.
> The per-second prices are RunPod list prices at the time of writing and change.
> `scripts/calibrate.py` exists to replace every number here with a measured one.
> **Do not sign an SLA before running it.**

Cold-start budget on an L40S with weights on a volume, pruned 20B int8:

| Phase | Estimate | Billed? | Confidence |
|---|---|---|---|
| Image pull, ~20–30 GB compressed, uncached host | 90–240 s | see caveat | medium |
| Image pull, cached host / FlashBoot revival | 0–15 s | — | high |
| Container start + `import wgp` | **25–60 s** | yes | medium |
| Fitness checks (CUDA, weight enumeration) | 2–5 s | yes | high |
| **First job: 48–60 GB read from the volume @ 200–400 MB/s** | **150–250 s** | **yes** | medium |
| Generation, 124 frames @ 832×480, **4-step turbo** | **UNMEASURED — est. 90–260 s** | yes | **low** |
| Generation, 124 frames @ 832×480, **20 steps stock** | **UNMEASURED — est. 330–900 s** | yes | **low** |
| ffmpeg mux + ffprobe + upload of 5–20 MB | 5–20 s | yes | high |

> **Billing caveat, unresolved.** RunPod's worker-state table lists `Initializing` as not
> billed; the pricing page says charges cover "start time: initializing the container and
> loading models into GPU memory". Assume `import wgp` and the weight load **are** billed
> (the conservative reading) and treat the image pull as unbilled.

Cost per generation at the estimate midpoints (L40S @ $0.00053/s):

| Scenario | Billed seconds | Cost |
|---|---:|---:|
| 4-step turbo, warm worker | ~180 | **~$0.10** |
| 20-step stock, warm worker | ~620 | **~$0.33** |
| Cold-start adder (import + first-job volume read) | ~240 | **+~$0.13** |
| Idle-timeout tail per warm window (180 s) | 180 | **+~$0.10** |
| 4-step turbo on A6000/A40 @ $0.00034 | ~230 | **~$0.08** |

Plus ~$14/month for a 200 GB volume, and object storage for the outputs.

The two dominant controllable costs are the idle tail on bursty traffic and cold starts.
That is why the idle timeout is 180 s rather than the 5 s default, and why a
baked-weights image (which removes the 150–250 s volume read at the price of a ~70–90 GB
image, near RunPod's documented 80 GB cap) is the next lever once you have real traffic.

The single largest lever on the generation itself is the accelerator profile: 4 steps vs
20 is roughly 3.5× cheaper, and it is one field in the request. Note the quality
trade-off — `README.md:141` warns that a 1.0 LoRA multiplier can be too strong, and
suggests the 8-step profile if quality suffers. There is **no accelerator LoRA for Ref2VA
in this repo**; a Ref2VA endpoint runs at 20 steps.

---

## File map

| Path | What it is |
|---|---|
| `handler.py` | The RunPod entrypoint. `async def handler` → `asyncio.to_thread(run_job)`, fitness checks, `runpod.serverless.start`. |
| `engine.py` | The only module that imports WanGP, and only inside functions. Session singleton, event drain loop, cancel, weight gate, recycle policy. |
| `schema.py` | Request validation and settings assembly. No torch, no wgp. |
| `media_in.py` | Materializes `b64` / `volume://` / URL inputs to absolute temp paths with sniffed extensions. |
| `media_out.py` | Output transport chain + ffprobe metadata + idempotency probe. |
| `config.py` | Env-driven `WorkerConfig`; renders and repairs `wgp_config.json`. |
| `errors.py` | The stable error-code taxonomy. |
| `obs.py` | Single-line JSON logging to the stdout captured at import. |
| `wgp_config.json.tmpl` | Baked config template; `config.py` renders it with absolute paths. |
| `Dockerfile` | Two-stage build. Consumes the repo's `requirements.txt` unmodified. |
| `requirements-worker.txt`, `constraints.txt` | `runpod>=1.12.0,<2`, plus pins that stop the runpod install from moving `pydantic`/`gradio`/`mcp`. Additive only — the repo's `requirements.txt` is never edited. |
| `test_input.json` | Local one-shot job. The SDK reads it from the **process CWD**. |
| `scripts/prefetch_weights.py` | GPU-side volume warmer. |
| `scripts/verify_weights.py` | Pre-deploy gate; the same enumeration the boot fitness check runs. |
| `scripts/calibrate.py` | Timing matrix → measured timeout and cost numbers. |
| `scripts/patch_sage_setup.py` | Build-time: pins SageAttention's target architectures without a GPU present. |
| `tests/` | CPU-only test suite: `test_schema.py`, `test_media.py`, `test_wgp_config_drift.py`, `test_handler.py` (end-to-end `run_job` with a stubbed engine), `test_engine.py` (the drain loop with a fake job). No GPU, no torch, no weights, no network. |
| `../.github/workflows/worker-ci.yml` | The Tier-1 suite plus hadolint, on every PR touching this directory. |
