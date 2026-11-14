# Multi-GPU (USP) for MiniMax H3

A WanGP plugin that splits each MiniMax H3 denoising step across 2, 4, 7 or 8 GPUs with Ulysses sequence parallelism
(USP). Turn it on from its tab and queue generations as usual. The result is identical to a single GPU.

## When it helps

USP trades attention heads between the GPUs twice in every transformer block, about 0.9 GB per GPU per block. That
only pays off when the GPUs reach each other directly:

| GPUs | Link | 5 s FastH3 clip, warm | vs 1 GPU |
|---|---|---|---|
| 1× RTX 5090 | — | 19.6 s | — |
| 2× RTX 5090 | PCIe 4.0 x16, **no P2P** (through system RAM) | 57.8 s | **2.95× slower** |
| 2× A40 (standard H3, profile 4) | PCIe | 53.9 s vs 70.9 s | 1.31× faster |

GeForce cards have no peer-to-peer access, so wide PCIe lanes are not enough. On them, run one WanGP per GPU instead,
one job each, for about N× the output. USP is for cutting the wait on a single clip on NVLink or P2P machines
(A100/H100/B200-class, some professional cards). The tab shows, for every GPU, whether it has direct access to the GPU
WanGP runs on.

## Using it

1. Enable **Multi-GPU (USP)** in the plugin manager, then open its tab.
2. Pick the number of GPUs, and optionally which ones (the default is the next GPUs after WanGP's own). Click **Start
   multi-GPU**.
3. Queue MiniMax H3 generations. The first one loads the model on every GPU; later ones reuse it.
4. **Stop** returns to one GPU. Start and Stop wait for the queue to be idle.

The world size must divide H3's 56 attention heads, hence 2, 4, 7 or 8.

## How it works

- **Helpers.** Start spawns one helper process per extra GPU (`helper.py`, `CUDA_VISIBLE_DEVICES` set to that GPU). Each
  is a headless WanGP session on the same folder, config and command-line options (profile, attention, config folder,
  …). All ranks join one NCCL group, and `usp.py` patches the H3 transformer classes in every process.
- **Same task everywhere.** A rank runs the whole pipeline (text encoders, VAE, sampling loop) and shards only the DiT
  sequence. When WanGP's queue reaches the model, the plugin sends the task's settings to the helpers. That happens only
  once rank 0 has loaded the model, so weights are never downloaded twice at once. Each helper adopts rank 0's resolved
  prompt (after the prompt enhancer), negative prompt and seed. Its own saved file is deleted, and it skips the
  final video decode of the last window, which only that file would use (earlier windows' decodes and H3's draft
  decode still run: their frames feed the next steps). The helper's GPU is then free for rank 0's decode, e.g. a
  multi-GPU decode worker on that GPU.
- **Lockstep** (`sync.py`). A localhost control connection per helper gates every `generate` call (sliding windows,
  dialogue segments) and every DiT forward:
  - At a `generate` call, the frame count, size, steps, solver and model must match, or the job stops with an error
    on every rank.
  - Before a forward's first collective, all ranks confirm they are entering it.
  - An abort on rank 0 is honored at the next denoising step: rank 0 stops before announcing the step, and the helpers
    stop at the same point.
  - A helper that fails answers instead of joining.
  - A cross-rank checksum of the assembled sequence catches any divergence at the first forward of each generation.
- **What still needs a restart.** If a rank fails *inside* a forward (for example out of memory mid-block), the others
  wait in NCCL until its 10-minute timeout, the job fails, and the tab reports the plugin as broken. Click **Stop**,
  then **Start**.

## Limits

- **Models:** MiniMax H3 only. Other models run on WanGP's GPU alone while the plugin is on.
- **Not supported under USP** (the job stops with an error): Sol, Veda and VDN sparse attention, step caches
  (Spectrum / FirstBlockCache), and ControlNet. sdpa and sage2 work, and so does FastH3's VSA.
- **Queue only:** only generations started from WanGP's queue (the UI, its API session) reach the helpers.
- **RAM:** each helper loads its own copy of the weights. Plan on about 40–50 GB of system RAM per GPU for H3.
- **CUDA graphs** are off for the DiT blocks while USP runs, because the blocks run collectives.
- **Abort latency:** an abort takes effect at the next denoising step, not mid-step.

## Files

| File | |
|---|---|
| `plugin.py` | The tab: GPU inventory with P2P, Start / Stop, and the queue hook around `generate_media` |
| `launcher.py` | Spawns the helpers, joins the NCCL group, and boots and stops them |
| `helper.py` | A helper process: a headless WanGP session serving rank 0's tasks |
| `sync.py` | The lockstep protocol (rank 0 `Leader`, helper `Follower`) |
| `usp.py` | The sequence-parallel H3 forward and attention hooks (also used by `torchrun` benchmarks) |
| `tests/test_sync.py` | CPU tests of the protocol: normal runs, aborts at each point, helper failure, mismatch, recovery, the helper's skipped decode |

`usp.py` also works without the plugin, under `torchrun`
(`runpod_worker/scripts/usp_bench.py`, `usp_fasth3_bench.py`). Its CPU checks are
`runpod_worker/scripts/test_usp_gloo.py` and `test_usp_vsa_cpu.py`. `runpod_worker/tests/test_usp_drift.py` fails when
upstream changes the H3 code that `usp.py` mirrors.

The plugin's timing tables come from `runpod_worker/scripts/h3_bench_usp.py` (USP alone, or with the multi-GPU decode, with a
per-phase breakdown) and `h3_bench_models.py` (one GPU); `h3_bench_runpod.sh` runs them on a Runpod pod. See
[docs/H3_BENCHMARKS.md](../../docs/H3_BENCHMARKS.md).
