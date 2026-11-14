# MiniMax H3 benchmarks (one GPU, USP, multi-GPU decode)

The scripts that produced the H3 timing tables in the Multi-GPU (USP) plugin's PR: full render time and time to first frame with
chunk streaming, per model combination and size, on one GPU or with USP on two, and a per-phase breakdown. They live in
`runpod_worker/scripts/` (the `h3_bench_` files).

| Script | What it does |
|---|---|
| `h3_bench_models.py` | One model per GPU, no USP: the 8 FL2VA combinations, cold and warm. On two GPUs it runs a contention check first and shares the queue if running together costs <= 5%. |
| `h3_bench_usp.py` | Two GPUs with the USP plugin. Per combination: USP with the video decode split across both GPUs (`mgd`), then USP alone (`usp`), so both compare on one host. Reports per-phase times and compares the two videos. |
| `h3_bench_decode.py` | Decode only: WanGP's INT8 H3 video VAE on seeded latents, on GPU 0 alone and split across GPU 0 and a decode worker on GPU 1. No DiT, only the VAE is downloaded. |
| `h3_bench_tables.py` | Markdown tables (Table A cold start, Table B warm runs per mode, phase tables) from a log's `RESULT` lines. |
| `h3_bench_boot.sh` | Boot script for a rented box: host checks, WanGP install, one benchmark, status log served on port 8888. |
| `h3_bench_runpod.sh` | Runs one benchmark on a Runpod pod: creates pods until one passes the host checks, follows the log, saves it, deletes the pod. |

## What the numbers mean

Every run streams (the chunk-streaming plugin's instrumented decode, audio first), so each yields:

- **Full render**: from submitting the task to WanGP's saved `.mp4`.
- **First frame**: the earliest start at which playback never stalls, given when each streamed segment became ready
  (`streaming.no_buffer_start`). When the decode is slower than real time, this is later than the first segment.

Cold runs (the first run of a combination, 480p 5 s) include downloads and the model load; warm runs are the 4 sizes
(480p and 720p, 5 s and 10 s: 124 and 243 frames), seed 42, the default FL2VA prompt.

`h3_bench_usp.py` also reports, per run, from rank 0's clocks:

| Phase | From → to |
|---|---|
| queue | task submitted → pipeline `generate` (settings, model load on a cold run, the lockstep hand-off) |
| prompt | `generate` → first DiT forward (text encoder, references) |
| denoise | first DiT forward → end of the last one (with a CUDA sync after each forward, the same in both modes), and seconds per step |
| audio decode | the audio VAE (decoded first, so segments carry sound) |
| video decode | the video VAE's chunk loop, and its first clip |
| save + mux | `generate` returned → task result (WanGP's full `.mp4`: encode, audio mux, metadata) |

and checks the multi-GPU decode's video against USP alone's twice: the decoded streams (md5) and, per finalized chunk, the
raw uint8 frames (`chunk_compare`: equal chunks, and per differing chunk the largest and mean difference and the share of values
that differ). With two decode devices clip `i` decodes on device `i % 2`, so odd chunks come from the GPU 1 worker.

## Code needed

The harnesses run inside a WanGP checkout and import its plugins:

- `plugins/usp_multi_gpu` (branch `feature/usp-multi-gpu-rebased`, with these scripts) for `h3_bench_usp.py`;
- `plugins/chunk_streaming` for the streaming instrumentation: branch `feature/chunk-streaming-rebased`, or
  `feature/multi-gpu-decode` for the `mgd` mode and `h3_bench_decode.py`.

`h3_bench_boot.sh` builds that checkout: it clones `H3_BENCH_REV` (default `feature/usp-multi-gpu-rebased`) and takes
`plugins/chunk_streaming` from `H3_BENCH_STREAMING_REV` (default `feature/multi-gpu-decode`). Pass commit ids instead of branch
names to reproduce a run exactly.

The install is the stack the published tables used: Python 3.11, torch 2.10.0 (cu130), `requirements.txt`, triton 3.6,
SageAttention 2.2.0 cu130, WGPKernels 1.4.3 cu130, lightx2v kernels; NVIDIA driver 580 or newer.

## Running on a box you already have

From the checkout (with the plugins above), with WanGP's flags (USP helpers inherit them):

```bash
python runpod_worker/scripts/h3_bench_usp.py --attention sage2 --profile 1 --verbose 1 | tee usp.log
H3_BENCH_MODES=usp H3_BENCH_COMBOS=3 python runpod_worker/scripts/h3_bench_usp.py --attention sage2 --profile 1   # USP alone, one combination
H3_BENCH_SINGLE=1 python runpod_worker/scripts/h3_bench_models.py --attention sage2 --profile 2 --verbose 1        # one GPU
python runpod_worker/scripts/h3_bench_decode.py --attention sage2 --profile 1 --verbose 1
python runpod_worker/scripts/h3_bench_tables.py usp.log --title "2x RTX PRO 6000, USP"
```

Outputs (videos, streamed segments, WanGP's files) go to `outputs/h3_bench/` (`H3_BENCH_OUT`). `H3_BENCH_FAKE=1
python runpod_worker/scripts/h3_bench_models.py` dry-runs the model harness's scheduling with no GPU and no WanGP.

## Running on Runpod

```bash
export RUNPOD_API_KEY=...        # never commit it; the script only reads it from the environment
runpod_worker/scripts/h3_bench_runpod.sh H3_BENCH=usp H3_BENCH_MODES=mgd,usp
H3_GPU_COUNT=1 runpod_worker/scripts/h3_bench_runpod.sh H3_BENCH=models H3_BENCH_SINGLE=1 "H3_BENCH_ARGS=--attention sage2 --profile 1"
```

It prints each new stage and result line and a status line per poll, saves the pod's log to `outputs/h3_bench/pods/<pod>.log`,
and deletes the pod when the benchmark ends or fails, after `H3_MAX_MIN` minutes (default 150), when the log has been
unreachable for `H3_DOWN_S` seconds, or when interrupted (Ctrl-C). The pod holds no API key and cannot delete itself: if the
launcher is killed outright or loses its network, delete the pod in the Runpod console, or it keeps billing. Then:

```bash
python runpod_worker/scripts/h3_bench_tables.py outputs/h3_bench/pods/<pod>.log
```

Other launcher settings: `H3_GPU_TYPE`, `H3_DATA_CENTERS` (e.g. `US-KS-2`), `H3_TRIES`, `H3_POLL_S`, `H3_LOG_DIR`. On another
provider, run `h3_bench_boot.sh` as the container's start command with the same `H3_BENCH*` variables, expose port 8888 and read
`/status.log`.

### Host checks

Before installing anything, a two-GPU box stops (`BENCH_FAILED topology ...`) unless GPU 0 and GPU 1 share a CPU socket
(`nvidia-smi topo -m` shows `NODE`, `PXB`, `PIX` or NVLink), and after the install unless P2P is on and a GPU 0 → 1 copy reaches
`H3_BENCH_MIN_P2P_GBS` (default 45 GB/s). `h3_bench_runpod.sh` then deletes the pod and tries another, up to `H3_TRIES`.
On 2x RTX PRO 6000 Server pods, same-socket hosts copied at 52-53 GB/s and cross-socket (`SYS`) ones at 37-42 GB/s, and USP is
slower on the latter (its all-to-all exchange runs at every attention layer). Rejected hosts cost about 2-3 minutes each.
Set `H3_BENCH_REQUIRE_NODE=0` and `H3_BENCH_MIN_P2P_GBS=0` to measure any host.

### Cost

On 2x RTX PRO 6000 Server ($5.38/h on Runpod Secure Cloud in November 2026) the setup takes about 2 minutes, the first cold run
about 5 minutes (it downloads about 50 GB). The full `usp` benchmark (7 combinations, both modes) has not been run to the end yet; from the per-run times it should take roughly 80-90 minutes.

## Notes from the November 2026 runs

- With the multi-GPU decode, odd chunks (decoded by the worker on GPU 1) differed from USP alone's in 2.5-5% of their uint8
  values (largest difference about 10-20 of 255), the chunks blended with them in about 1%; chunk 0 and the audio were identical.
  The decode-only benchmark (both VAEs loaded the same way, outside the pipeline) is bit-identical, so the difference comes from
  how the pipeline's VAE runs in WanGP; the likely cause, separate INT8 kernel autotuning per process, is not confirmed.
- Model weights are downloaded by the cold run of each combination, inside the timed task: Table A's download columns are only
  measured by `h3_bench_models.py` (it watches the weight folders).
