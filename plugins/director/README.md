# Director: multi-GPU shot rendering with consistent keyframes

`plugins/director/director.py` renders a planned film with MiniMax H3 (FastH3 Omni NVFP4 by default). It runs several GPUs in parallel and streams the finished clips in order while later ones are still rendering.

1. **Storyboard.** One short H3 generation holds one still shot per planned clip, joined by hard cuts at fixed times (`[Shot 2] At 00:01.000, cut to …`). A single generation keeps faces, wardrobe, light and places consistent across its shots. Long films use several boards. Each board after the first starts on the previous board's last keyframe, which carries the look forward.
2. **Keyframes.** The director takes the middle frame of every storyboard shot, at least 4 frames from any cut. It reads the decoded frames losslessly through the API's `return_media` (`keyframes/kf_NNN.png`).
3. **Clips.** There is one H3 generation per shot. Each starts exactly on its keyframe (`image_start`), uses the same subject reference images, and can optionally end on the next keyframe (`image_end`). Each GPU runs its own WanGP process, and the clips render in parallel.
4. **Stream.** Each clip is appended to `stream/playlist.m3u8` (HLS) as soon as it and every clip before it are done. At the end everything is stitched into `film.mp4`.

The [cut-away mode](#cut-away-mode-no-storyboard) skips steps 1 and 2: every clip starts on one source image and cuts away from it at once, so all clips start together.

It has run on RTX 5090s: see [Measured on RTX 5090](#measured-on-rtx-5090).

## In WanGP

This folder goes in WanGP's `plugins/` folder (as `plugins/director`). In WanGP, open **Plugins**, enable **Director** and restart WanGP. The **Director** tab then runs `director.py`:
- pick an example plan or give a plan file;
- list the GPUs;
- choose a mode: **Dry run**, **Rehearsal** (test patterns, no GPU) or **Render**.

It shows the run's log and where its stream and film are. Each GPU runs its own WanGP process, so unload the WebUI's model first or list other GPUs. The command line below does the same, with every option.

## Running it

```bash
# what will run: boards, frames, the first prompts, and a GPU-count estimate from measured RTX 5090 times
python plugins/director/director.py plugins/director/examples/plan.json --out runs/signal --dry-run --gpus 0,1,2,3

# render on four GPUs
python plugins/director/director.py plugins/director/examples/plan.json --out runs/signal --gpus 0,1,2,3

# rehearse without a GPU: ffmpeg test patterns, the same scheduling, keyframes and stream
python plugins/director/director.py my_plan.json --out runs/rehearsal --backend fake --gpus 0,0,0 --fake-seconds 2
```

| Option | Meaning |
|---|---|
| `--gpus 0,1,2,3` | One WanGP worker process per id. Each process sees only that GPU (`CUDA_VISIBLE_DEVICES`). |
| `--dry-run` | Writes `jobs.json` (every WanGP settings dict) and prints the prompts and the estimate. Nothing runs. |
| `--est-scale`, `--est-load-s` | Dry run only: scale the measured times for a slower host, and add a cold model load to each GPU's first job. |
| `--prebuffer N` | Playback starts once N clips are live (used in the stall estimate and `report.json`). |
| `--retries N` | Each failed job is retried N times, on any free GPU, before the run stops (default 1). |
| `--parallel-start` | Skips holding the other GPUs until the first job finishes. The hold exists because a first job may download the weights, and concurrent downloads into one checkpoint folder collide. Use this flag only when the weights are already present. With it, warmups run while the first storyboard renders. |
| `--no-warmup` | Turns off warmups. By default, a GPU with no model loaded and nothing to run (for example while a storyboard renders) runs the shortest text-only generation (107 frames) to load the model, so its first clip does not pay the load. |
| `--no-hls` | Skips the playlist and still writes `film.mp4`. |
| `--encoder x264\|nvenc\|wangp` | How workers encode clips (default: the plan's `encoder`, `x264`). See [Several workers on one machine](#several-workers-on-one-machine). |
| `--no-cpu-pinning` | Lets every worker use all CPU cores, as before. |
| `--backend fake` | No GPU and no WanGP import. Checks that every input a job names exists, then renders a test pattern of the right length. |

Watch the stream while it renders. `ffplay runs/signal/stream/playlist.m3u8` or VLC opens it directly. Alternatively, serve the folder (`python -m http.server -d runs/signal/stream`) and open it in Safari or an hls.js page.

## The plan

Paths are relative to the plan file. Three complete six-clip, one-minute examples are included:
- `plugins/director/examples/plan.json` uses reference images for a character and a place.
- `plugins/director/examples/textonly_plan.json` needs no assets. It describes its character the same way in every shot, and the single storyboard generation keeps the look consistent.
- `plugins/director/examples/cutaway_plan.json` is the same film in [cut-away mode](#cut-away-mode-no-storyboard). It needs one source image, `plugins/director/examples/refs/keeper_source.png` (not included).

```json
{
  "title": "The Signal",
  "model_type": "minimax_h3_fasth3_omni_nvfp4",
  "resolution": "832x480",
  "clip_seconds": 10,
  "seed": 42,
  "style": "a cinematic night drama shot on 35 mm film: cool teal moonlight, warm amber lamps",
  "subjects": [
    {"image": "refs/astronomer.png",
     "description": "the radio astronomer from {picture}, a woman in her sixties with round glasses",
     "retain": "face, hair, glasses and cardigan are retained"}
  ],
  "storyboard": {"shot_seconds": 1.0, "shots_per_board": 6, "link_boards": true, "seed": 7},
  "audio": {"mode": "generated", "cut_fade_ms": 40},
  "end_images": "none",
  "settings": {"config": ",lightvae_int8"},
  "wgp_config": {},
  "cli_args": [],
  "shots": [
    {"keyframe": "Close-up of <Subject 1>'s face in amber light.",
     "action": "<Subject 1> leans toward the microphone and speaks quietly.",
     "soundscape": "Rain on glass, her breathing.",
     "music": "N/A",
     "dialogue": [{"subject": 1, "text": "If you can hear me, answer.", "language": "English"}],
     "subjects": [1],
     "seconds": 10}
  ]
}
```

**Top level**
- `style`: one sentence, prepended to every storyboard and clip description. Write it once, so every generation sees the same look.
- `subjects`: up to 9 reference images. `<Subject N>` always means the same subject, while picture numbers are computed per generation. `{picture}` marks where the image label goes in the description. `retain` names what must stay identical.
- `clip_seconds`: rounded to H3's lengths (107 + 17n frames at 24 fps: 10 s → 243 frames, 15 s → 362). One clip is at most 362 frames, a single sliding window.
- `end_images`:
  - `"none"` (default): every clip boundary is a hard cut.
  - `"next"`: each clip also ends on the next clip's keyframe, for continuous action across a boundary. Over 10–15 s, the end can visibly snap onto that frame. `end_image_strength` (default 0.5) eases into it.
- `cutaway`: turns on [cut-away mode](#cut-away-mode-no-storyboard): `{"image": "refs/source.png", "seconds": 0.5}`.
- `encoder`: `x264` (default), `nvenc` or `wangp`. See [Several workers on one machine](#several-workers-on-one-machine).
- `settings`: merged into every WanGP job, for example `config`, `num_inference_steps` or `overrides`. The director's own keys (prompt, images, audio, length, seed) always win.
- `wgp_config`: overrides patched into each worker's copy of `wgp_config.json`, for example `{"int8_kernels": "kitchen", "compute_cache": true, "video_profile": 1, "profile": 1}`. This needs an existing `wgp_config.json` in the WanGP root, so start WanGP once first.
- `cli_args`: WanGP startup flags for every worker.

**Storyboard**
- `shot_seconds` (default 1.0, minimum 0.5) is how long each held storyboard shot lasts.
- `shots_per_board` (default 5) caps a board's shots. A board must fit one 362-frame window, anchor shot included.
- `link_boards` (default true) starts each board on the previous board's last keyframe.
- A shot with its own `keyframe_image` skips the storyboard.

**Shots**
- `keyframe` describes the still frame the clip starts on. Write it like a shot list entry: framing, who is where, light.
- `action` describes what happens from that frame. It may contain its own internal cuts (`[Shot 2] At 00:05.000, …`), and H3 keeps continuity inside a clip.
- `subjects` lists which subjects appear (default: all). Only those reference images are sent.
- `dialogue` lines become `(S1) <d>[English] …</d>` in generated and bed modes. In kept mode they become a lip-sync instruction instead.

## Cut-away mode (no storyboard)

With `"cutaway": {"image": "refs/source.png"}`, there is no storyboard. Every clip starts on that one source image, holds it for about half a second, and cuts away to its planned shot inside the same generation. H3 keeps the world, light and wardrobe of the first frame across a cut within one clip, and the subject reference images keep faces consistent. Since no clip waits for another job, all GPUs start at once and the first clip is live after one clip's render.

- **The cut** is placed on a latent boundary of the video VAE near `seconds` (default 0.5 s, so frame 13, 0.542 s), so no latent mixes the source image with the new shot.
- **The trim.** Each clip renders its lead too: `seconds` in a shot is the length it keeps in the film, and the rendered clip is about 0.8 s longer (a 10 s shot renders 260 frames). The worker finds the cut in the first frames, then re-encodes the clip from frame `cut + margin_frames` (default 6, so frame 19), frame-exact for video and sample-exact for audio. A clip's length in the film is therefore known before it renders.
  - If the cut comes later than planned, the clip starts `guard_frames` (default 2) after it and is a few frames shorter. The run log and `report.json` say `cut late`.
  - If no cut is found, the clip is trimmed at the planned frame and the log says `cut not found`.
- **Shots.** `keyframe` is optional and describes the first frame after the cut. `action` may contain its own cuts (`[Shot 2] At 00:05.000, …`): their numbers and times are shifted past the lead in the prompt. A shot with its own `keyframe_image` starts on that image instead, without a lead.
- **Audio.** All three modes work. In `kept` mode, each slice starts with as much silence as the lead, so the soundtrack stays continuous after the trim.
- `end_images` must be `"none"`.

```json
"cutaway": {"image": "refs/source.png", "seconds": 0.5, "margin_frames": 6, "guard_frames": 2}
```

The source image can come from H3-Image, any image tool, or a frame of an earlier render (`keyframes/kf_NNN.png`). The raw clips, lead included, stay in `clips/clip_NNN_raw.mp4`.

## Several workers on one machine

On the 4× RTX 5090 run below, the work after denoising (VAE decode, audio, saving) took about 3 s per clip with two clips at once but 8–12 s with four, while denoising itself did not slow down. The workers were competing for the CPU: each started a thread pool the size of the machine (256 threads there), and WanGP encodes its clips with x264 CRF 10 at the medium preset. The director now:

- **Pins each worker to its own CPU cores**, on the NUMA node of its GPU when `nvidia-smi` and the kernel report it, and sizes every CPU thread pool (OpenMP, MKL, torch, x264) to them. `--no-cpu-pinning` turns this off.
- **Encodes clips with x264 CRF 10 at the veryfast preset.** It keeps WanGP's quality setting for about a third of the CPU time (measured on a 243-frame 832×480 clip: 15.6 s of CPU instead of 48.3 s, similar file size).
  - `"encoder": "nvenc"` (or `--encoder nvenc`) encodes on the GPU instead. Containers without the driver's video capability cannot use NVENC; the worker then falls back to x264 and the log says so.
  - `"encoder": "wangp"` keeps WanGP's own settings.
- **Processes finished clips for the stream in a background thread**, so the director keeps handing jobs to free GPUs meanwhile.

`report.json` lists each worker's encoder and CPU thread count under `workers`.

## Audio

| `audio.mode` | What happens | Continuity |
|---|---|---|
| `generated` (default) | Each clip generates its own sound. The stream adds `cut_fade_ms` fades (default 40 ms) at every cut. `voices` (up to 3 `{"subject", "audio"}` samples) are passed as audio references, so a character keeps one voice across clips. | Ambience and music restart at each cut. Write `music: N/A`, or use `bed`. |
| `kept` | `audio.soundtrack` (dialogue, music and ambience prepared beforehand, for example with the H3 TTS model) is cut into one sample-exact slice per clip. Each clip uses H3's **Soundtrack Kept** mode (`AS`), so the picture is animated to that audio and lip-synced. `film.mp4` gets the original soundtrack back when it covers the whole film. | Continuous by construction. |
| `bed` | Like `generated`, plus `audio.bed` (music) mixed under every segment at `bed_gain_db` (default −12 dB), slice by slice. | Music is continuous. Effects restart at cuts. |

In kept mode, a clip that starts after the soundtrack ends generates its own audio, and the run prints a note. In bed mode, clips past the end of the bed get no bed.

## Outputs (`--out`)

| Path | Content |
|---|---|
| `jobs.json` | Every job: WanGP settings, dependencies, priority, outputs |
| `storyboards/board_NN.mp4`, `keyframes/kf_NNN.png` | Storyboards and the extracted keyframes |
| `clips/clip_NNN.mp4` | Clips as WanGP saved them; in cut-away mode, the trimmed clips |
| `clips/clip_NNN_raw.mp4` | Cut-away mode: each clip with its lead, before the trim |
| `audio/` | Soundtrack or bed as WAV, plus per-clip slices |
| `stream/playlist.m3u8`, `stream/seg_NNN.{mp4,ts}` | The live stream: processed segments in film order |
| `film.mp4` | The stitched film |
| `report.json` | Per-job GPU and timings (and, in cut-away mode, where each cut was found and the clip trimmed), when each clip went live, first-frame time, stalls with the chosen prebuffer, and each worker's encoder and CPU threads |
| `workers/workerN/` | Each worker's `wgp_config.json`, WanGP log and raw outputs |

## Sizing: how many GPUs

The dry-run estimate replays the real scheduler using warm RTX 5090 times for FastH3 Omni NVFP4 at 832×480: 5 s → 16.5 s, 10 s → 31.3 s, 15 s → 62.6 s. For the one-minute example (one 6.6 s storyboard, then six 10 s clips), it shows:
- 3 GPUs stall about 1 s.
- 4 GPUs stream without a stall, with the first clip live about 52 s after launch.

Add cold model loads with `--est-load-s`. Pass `--parallel-start` when the weights are present, so idle GPUs warm up while the storyboard renders. Without it, each GPU's first clip pays the load and playback stalls once.

Measured times are the best case: they reuse a cached prompt and the fastest host. Keep a spare GPU, or pass `--est-scale 1.2`, until fresh-prompt timings are measured.

## Known limits and open questions
- **Weights in RAM.** Each worker loads its own copy of the weights into system RAM. Plan on about 40–50 GB of RAM per worker on one machine.
- **Not yet measured:** Soundtrack Kept mode on FastH3, plans with reference images, the cut-away mode on GPUs, and how much CPU pinning and veryfast encoding recover on a real multi-GPU run.
- **No per-job timeout.** A WanGP process that hangs holds its GPU until you stop the run.

## Measured on RTX 5090

These runs predate the cut-away mode and the [CPU pinning and veryfast encoding](#several-workers-on-one-machine).

`plugins/director/examples/textonly_plan.json` (one 6-shot storyboard, then six 10 s clips: 60.8 s of film) on FastH3 Omni NVFP4 at 832×480, with `"wgp_config": {"int8_kernels": "kitchen", "compute_cache": true, "video_profile": 1, "profile": 1, "kernel_precision": "fast"}`, weights already downloaded, `--parallel-start`.

| Setup | Storyboard | 10 s clip | First clip live | Film rendered | Stalls |
|---|---|---|---|---|---|
| 4× RTX 5090, 450 W cap, one machine | 39 s (with model load) | 47–50 s with 4 at once, 39–41 s with 2 | 101 s | 146 s | **0** |
| 1× RTX 5090, 475 W cap | 45 s (with model load) | 37 s | 89 s | 277 s | 5 (135 s) |

- **Worker start:** 6–12 s once the weights are cached, about 55 s the first time. The three warmups (33–37 s) finished before the storyboard did.
- **A clip's phases:** a fresh prompt encodes in about 2 s, and the 8 denoising steps take about 31 s. VAE decode, audio and saving take 6 s with one or two clips running and about 14 s with four. That shared post-denoise work is the four-GPU slowdown.
- **Margin:** four 450 W cards run at about 0.9× realtime per clip. The minute streamed without a stall because the first four clips arrived together. Longer films need a fifth GPU on such cards. At the single-GPU 37 s per clip, four GPUs would sustain about 1.1×.
- **Consistency:** 1 s storyboard shots rendered as clean, distinct frames, and the keeper's face, beard, cap and coat stayed the same across all six clips.

## Tests

```bash
python -m unittest plugins/director/tests/test_director.py
```

The tests need no GPU and do not import WanGP. They cover:
- frame-length fitting;
- prompt layout and picture numbering;
- the job graph;
- scheduling and playback;
- audio slicing;
- keyframe extraction;
- full fake-backend runs, including in-order streaming and the first-job hold.

The media tests need ffmpeg.
