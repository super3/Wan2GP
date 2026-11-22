# H3 Live: real-time MiniMax H3 for WanGP

A [WanGP](https://github.com/deepbeepmeep/Wan2GP) plugin that turns one MiniMax H3 generation into a **live session**: a list of clips generated back to back, each one streamed to a player in the plugin's tab while the next renders, saved as an MP4, and joined into `film.mp4` at the end.

With **MiniMax H3 FL2VA Pruned pi-H3 DWD 2-Step** at 480p, a session keeps ahead of playback on a 32 GB GPU. On an RTX 5090 with WanGP v20.10, each 5.17 s clip is ready every 4.3 s (1.19× real time, measured with this plugin's session, Nov 22). On WanGP v20 with the earlier DWD port it was 4.6 s (1.12×), and 1.08× when the clips changed prompts, against 6.9 s per clip when the same clips were queued in WanGP.

## Install

1. Use WanGP v20.10 or later, which ships the pi-H3 DWD models, with `ffmpeg` on the PATH (WanGP's own ffmpeg works for x264; NVENC needs a system ffmpeg built with it).
2. This folder goes in WanGP's `plugins/` folder (as `plugins/h3_live`). In WanGP, open **Plugins**, enable **H3 Live** and restart WanGP.

## Use

1. Open the **H3 Live** tab and tick **Live mode**.
2. Write the clips' prompts, one line per clip (or leave it empty to repeat the generation's own prompt), and set **Clips** if you want more clips than prompts (the prompts are cycled; each clip's seed is the generation's seed plus the seed step).
3. Go to the Video Generator tab, pick a MiniMax H3 model (pi-H3 DWD 2-Step for real time), set the resolution and length of one clip (e.g. 832×480, 124 frames = 5.17 s), and press **Generate**.
4. Back in the H3 Live tab, press **Play the latest session** as soon as the first clip is listed. The player follows the session as clips arrive; the table shows each clip's generation time and the session's real-time factor.

The session folder (`outputs/h3_live/live_<date>/`) holds `clip_NNNNN.mp4` per clip, `film.mp4`, `live.m3u8` (open it in VLC or Safari while the session runs) and `session.json` with the timings. WanGP also saves the last clip in its gallery, as for any generation. **Stop after the current clip** ends a session early; WanGP's own Abort stops it at once. Only the newest sessions are kept (5 by default).

## What it needs for real time

- **GPU:** 32 GB or more, and about RTX 5090 compute or better (measured: RTX 5090, 1.19× on v20.10, 25 GB of VRAM used). The INT8 transformer (21 GB) and the DWD head have to stay in VRAM; below 32 GB their weights are streamed from RAM at every step, several times slower.
- **Prompts:** encoded once, before the first clip (the default). On a 32 GB GPU the text encoder does not fit beside the transformer, so a prompt encoded during the session would move the transformer out and back.
- **Model:** pi-H3 DWD 2-Step. The VAE-decoded models work too, about 1.0× on an RTX PRO 6000, because of their 1–1.5 s video decode per clip.

## How it works

Nothing in WanGP is edited. The plugin wraps `MiniMaxH3Pipeline.generate`: with live mode on, the one call WanGP makes for an H3 video runs the pipeline once per clip of the session and returns the last clip. Between clips nothing leaves the GPU, since it is still the same generation: no model unload, no LoRA reload, no garbage collection, no save in the generation thread. For the session, the transformer is made an mmgp co-tenant of the VAEs (of every model on GPUs with 60 GB or more), so the audio decode does not move it out. Each clip goes to a background thread, which encodes it with NVENC (or x264) as an MP4 and as an MPEG-TS segment of a rolling HLS playlist, without blocking the GPU.

The hooks it relies on are checked when live mode is turned on (`generate`'s keyword arguments `input_prompt`, `seed`, `frame_num`, `fps`; a result with `x`, `audio`, `audio_sampling_rate`; `_encode_prompt(prompt, presentation)`). If a WanGP update changes them, the tab says live mode is not available and H3 generations run as usual.

Sliding windows, continued videos and image generations are never turned into sessions.

## Demo: an endless real-time stream in the browser

`demo/h3live_demo.py` runs WanGP headless and turns one generation into an endless live session: the same prompt (by default a man running on a stadium track), a new seed for every clip, the prompt encoded once. It serves a page with the stream and the generation's numbers (speed, clips made, the clip and seed playing, GPU load) on port 8000:

```bash
cd Wan2GP   # with this plugin in plugins/h3_live
python plugins/h3_live/demo/h3live_demo.py --port 8000 -- --attention sage2 --profile 1
# then open http://<this machine>:8000
```

The session is paced: it keeps at most `--lead` seconds (12 by default) of video ahead of the stream's clock, so the GPU, faster than real time, waits instead of running away from the viewers, and the playlist holds the last `--window` clips (8), older clip files being deleted. `--prompt`, `--resolution`, `--frames` and `--model` change what it generates. `--dry-run` streams synthetic clips without WanGP or a GPU, to try the page. Ctrl+C stops after the current clip.

## Tests

```bash
python -m pytest plugins/h3_live/tests -q   # CPU only, needs torch, ffmpeg and ffprobe
```
