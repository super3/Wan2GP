# Multi-GPU inference for MiniMax H3 via Raylight — evaluation & integration plan

Written 2026-08-19, researched against raylight v1.9.0 (komikndr/raylight),
ComfyUI >= 0.30.0, and the RunPod serverless API. All claims below were
verified against primary sources (repo code, live API schema, merged PRs).

## What Raylight is, in one paragraph

A ComfyUI custom-node extension that runs true multi-GPU diffusion inference:
one Ray actor per GPU (NCCL process group), xDiT/xFuser **USP** (Ulysses x
Ring sequence parallelism) splits the packed token sequence across GPUs each
denoise step, optional **FSDP2** shards weights, optional CFG-parallel
(unusable for H3 -- batch size 1). MiniMax H3 FL2VA/Ref2VA are officially
supported (USP yes, FSDP yes, CFG no) and Raylight ships example H3 workflows
(2 GPUs, ulysses=2). MiniMax themselves publicly credited Raylight's H3 port.

## Feasibility facts (verified)

- **ComfyUI runs our exact model natively** since 0.30.0 (Comfy-Org PR #15224,
  merged 2026-08-03, authored by kijai): the pruned `int8_convrot` transformer
  we already ship, the Qwen3-VL-32B text encoder, joint video+audio latents,
  and the lightx2v turbo LoRAs load with the *standard* LoRA loader.
- **Raylight supports H3** via the generic `Load Diffusion Model (Ray)` node +
  `XFuser SamplerCustomAdvanced`; the packed text+video+audio sequence is
  split per rank and all-gathered per step. Text encode and (by default) VAE
  decode stay serial on the host; an optional `Distributed VAE (Ray)` node
  can tile decode across workers.
- **RunPod serverless does multi-GPU workers**: `gpuCount` on the endpoint
  (verified in the live OpenAPI schema), all GPUs on one physical host,
  pricing linear per GPU. Current messaging: up to 8 GPUs/worker on modern
  tiers. Interconnect (NVLink vs PCIe) is undocumented -- assume PCIe.

## Blockers and risks (verified)

1. **Raylight issue #110 (OPEN): the H3 lightx2v Turbo LoRA applies ZERO
   weight patches** under `Load Lora Model (Ray)` -- keys load, nothing
   applies. Our production accelerator IS that LoRA. Until fixed, Raylight
   would silently run 4 "turbo" steps with no turbo weights (garbage) or
   force us to 20-step baseline sampling.
   Workaround available: **merge the LoRA into the transformer checkpoint
   offline** (alpha128 rank-8 merge, one-time script) and load the merged
   int8 checkpoint with no runtime LoRA at all.
2. **worker-comfyui (RunPod's stock ComfyUI worker) is not usable as-is**:
   it pins ComfyUI 0.29.0 (< 0.30.0 needed) and its handler only returns the
   `images` output key -- H3's SaveVideo mp4 may never be returned. We would
   keep OUR handler/schema/media/obs stack and swap only the engine.
3. Stack split: Raylight wants torch 2.8.1 (FSDP2 path) + ray + xfuser +
   NCCL 2.28.9; our image ships torch 2.13.0/cu13. A separate image is
   mandatory either way; USP-only mode may tolerate newer torch (untested).
4. Ring parallelism has a known VRAM leak upstream -- use pure Ulysses.
5. Two open H3 bugs upstream (#110 LoRA, #111 audio-continuation KeyError)
   suggest H3 support is young; expect sharp edges.

## The economics, against our own measurements

Warm 4-step clip (832x480, 124 frames): denoise ~32 s of a ~67 s A40 job;
already down to **24.2 s on one full RTX PRO 6000 (profile 1, ~1.4c/clip)**
and 22.9 s on one H100. USP parallelizes ONLY the denoise phase; encode
(~2-7 s warm) and VAE decode+mux (~11 s) stay serial. Raylight's own
benchmarks show ~1.7-1.8x at 2 GPUs for video DiTs.

| Setup (warm, 4-step turbo) | est. total | $/hr | c/clip |
|---|---|---|---|
| 1x PRO 6000, profile 1 (measured) | 24.2 s | 2.09 | 1.4 |
| 2x A40 USP (denoise 32->18 s) | ~53 s | 0.88 | 1.3 |
| 2x PRO 6000 USP (denoise ~13->7 s) | ~18-19 s | 4.18 | 2.2 |
| 4x A40 USP (denoise 32->11 s) | ~46 s | 1.76 | 2.2 |

**Conclusion: with the 4-step turbo LoRA, multi-GPU cannot beat a bigger
single GPU on cost, and beats it on latency only modestly (24 -> ~18 s)**
because Amdahl's law caps the win: denoise is already only half the job.
Where Raylight genuinely wins:
- **No-turbo / high-step workloads** (20 steps: denoise ~160 s dominates ->
  2 GPUs ~1.7x, 4 GPUs ~2.6x end-to-end).
- **Longer/higher-res clips** (sequence length grows quadratically in
  attention; USP splits it; single-card time and VRAM balloon).
- **A hard latency floor** below what any single card can do (~18 s).

## Plan (phased, each phase is a go/no-go gate)

**Phase 0 -- pick the target (decision, no code).** If the goal is cheaper
clips: stop; profile 1 on PRO 6000 already won. If the goal is a <20 s latency
floor or long-form/hi-res clips: proceed.

**Phase 1 -- prototype on a pod (0.5-1 day, ~$10).** Rent one 2x GPU pod
(2x A40 or 2x PRO 6000). Install ComfyUI >= 0.30, Raylight, torch 2.8.1,
ray, xfuser. Load our exact checkpoints (already on HF). Run Raylight's
shipped `Minimax_H3_I2V_Raylight.json` adapted to FL2VA + our clip settings.
Measure: 1-GPU vs 2-GPU s/step, end-to-end, VRAM. **Test issue #110
directly** (does the turbo LoRA apply?). Gate: >= 1.5x denoise speedup and a
working turbo path (native or merged-checkpoint workaround).

**Phase 2 -- turbo LoRA workaround (0.5 day).** If #110 unfixed: write a
one-time merge script (load bf16 transformer + alpha128 LoRA, merge, requant
int8_convrot, upload to HF). Validate output parity against WanGP frames on
the same seed.

**Phase 3 -- serverless engine swap (2-4 days).** Keep `runpod_worker`'s
handler/schema/media_in/media_out/obs/errors (engine-agnostic by design; see
engine.py docstring). Add `engine_comfy.py`: boot = launch headless ComfyUI +
Ray actors; run = submit API-format workflow JSON (template rendered from our
validated Request), poll history, map progress to our existing phase/progress
contract, return the same 5-tuple. New Dockerfile (ComfyUI + Raylight +
torch 2.8.1 stack, weights download-on-boot as today). Endpoint with
`gpuCount: 2`, `minCudaVersion` per torch build. CPU tests mirror the
existing suite (schema/media reused unchanged).

**Phase 4 -- benchmark + decide (0.5 day, ~$5).** Same job matrix as today
(seeds 12345/5555, warm pairs) on 2x A40 and 2x PRO 6000; publish the value
table next to the single-GPU one in runpod_worker/README.md; keep whichever
config wins the target metric. Watch upstream #110/#111 for fixes.

Total: roughly a week of effort, ~$20 of test spend, with a kill switch after
each phase.
