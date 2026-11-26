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
- **Start and end images:** a session encodes each picture with the VAE once and reuses it in every clip (`cache_image_encodes`, on by default). Without it the VAE encoder ran in every clip, which also moved the audio VAE out of the GPU and back. Measured on an RTX PRO 6000 (Nov 23), four clips of the same lines, prompts pre-encoded:

  | | Seconds per 5.17 s clip | Real time |
  |---|---|---|
  | Text only | 4.27 | 1.21× |
  | Start image | 5.38 | 0.96× |
  | Start and end image | 5.81 | 0.89× |
  | Start and end image, encodes cached | 4.66 | **1.11×** |

  What is left (0.4 s a clip) is the longer sequence: the two pictures' 780 conditioning rows, and the prompt as the text encoder sees it with them (1,019 tokens instead of 169).
- **Prompts of different lengths:** with profile 1, mmgp's automatic VRAM preload measures each new workload (the size of a block's largest input, which follows the prompt's length) and runs it with the transformer streamed from RAM until it is measured, then loads the transformer back. A session whose lines differ in length paid that in every clip. `steady_vram_preload` (on by default) rounds the workloads up (by at most 1.6%), so the clips share one measure. Measured on an RTX PRO 6000 Workstation (Clore, Nov 24, WanGP v20.12), sessions of four clips with a different line each, prompts pre-encoded, the image encodes cached:

  | | Weight copies from RAM per clip | Seconds per 5.17 s clip | Real time |
  |---|---|---|---|
  | Start and end image, without it | 0.70 s | 5.05 | 1.02× |
  | Start and end image, with it | 0.16 s | **4.83** | **1.07×** |
  | Start and end image, whole transformer preloaded (`--preload 40000`) | 0.15 s | 4.83 | 1.07× |
  | Text only, without it | | 4.22 | 1.23× |
  | Text only, with it | | **4.14** | **1.25×** |

  With it a session's clips run as fast as with the whole transformer preloaded by hand. Seconds per clip are the time between clips, which leaves out the first clip of a session: it measures its workload either way (and took about 0.6 s longer with it in this test).

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

## Prompt embeddings made elsewhere

H3 is conditioned on its text encoder's output for the prompt, not on the prompt itself. `embed_server.py` makes that output on one machine, and a video generation on another uses it instead of running its own text encoder: the prompts of a session can come from an LLM, and their encoding from a separate GPU.

**What an embedding is.** Two tensors: `prompt_embeds`, bfloat16 `[1, tokens, 5120]` (Qwen3-VL's hidden states: the 32B encoder truncated after layer 50, or the distilled 4B through its connector), and `text_token_tags`, int64 `[tokens]` (1 for a text token, 0 for an image token and its markers). That is 10,240 bytes per token. With a start or end image the encoder also sees the picture, which adds its image tokens (390 for an 832×480 picture, plus a few markers), so the same prompt has a different embedding with and without its pictures, and with other pictures. Measured on an RTX PRO 6000 (Nov 23): an FL2VA clip prompt of about 230 text tokens with a start and an end image at 832×480 is 1,008–1,020 tokens (10.3–10.5 MB); a short text-only prompt is 90 tokens (0.9 MB). Encoding took 1.8–2.4 s with the two pictures and 0.95 s as text only, and 0.06 s for a prompt the service had already encoded.

**Checked on a GPU** (RTX PRO 6000, WanGP v20.10, pi-H3 DWD 2-Step): the endpoint's embeddings were bit-identical to the ones the pipeline computed itself in a live session, with the two pictures and as text only, and the same session run from the endpoint's embeddings never ran its text encoder and made bit-identical clips.

**The endpoint.** WanGP runs headless and loads `--model` with one generation that does not end; H3 Live's hook turns it into a service that encodes each request with the pipeline's own text encoding, while the model stays loaded.

```bash
cd Wan2GP   # with this plugin in plugins/h3_live
python plugins/h3_live/embed_server.py --port 8890 -- --attention sage2 --profile 1
python plugins/h3_live/embed_server.py --dry-run   # no WanGP, no GPU: made-up embeddings of the right shape
```

- `POST /v1/embeddings` with JSON `{"prompt": "...", "images": ["<base64 PNG or JPEG>", ...], "width": 832, "height": 480, "format": "safetensors"}`. `images` are the start image, then the end image, as the video generation will get them (none for text only); `width` and `height` are the video's size, which the pictures are resized to, as the pipeline does. The answer is a safetensors file (`prompt_embeds`, `text_token_tags`, and the prompt, its picture kinds and their digest in the metadata), or with `"format": "json"` the same as JSON with `prompt_embeds` as base64 of the raw little-endian bfloat16 values. The headers `X-H3-Tokens`, `X-H3-Dim` and `X-H3-Encode-S` give the shape and the encode time.
- `GET /v1/health`: ready or not, the model, embeddings served, mean encode time.
- It listens on 127.0.0.1 unless `--host` says otherwise; `--token` (or `H3_EMBED_TOKEN`) then asks every request for `Authorization: Bearer <token>`.

```bash
curl -s localhost:8890/v1/embeddings -H 'Content-Type: application/json' \
  -d "{\"prompt\": \"$PROMPT\", \"images\": [\"$(base64 -w0 start.png)\", \"$(base64 -w0 start.png)\"]}" -o clip1.safetensors
```

From Python, `embeddings.fetch(url, prompt, images)` returns one embedding and `embeddings.save(embedding, path)` writes it.

**Using them.** Put the `.safetensors` files in a folder and give it to a session: `LiveSettings(embeddings_dir=...)`, or `--embeddings-dir` for the demo. A supplied embedding is used when the generation's prompt is the same text and it shows the encoder the same kinds of pictures (two images for a start and an end image); anything else is encoded as usual. `embeddings.supply(..., strict=True)` also requires the pictures to be the same ones (the pipeline's digest of them). From a script, `embeddings.supply(embedding)` works for any generation once the plugin's hooks are installed.

## Tests

```bash
python -m pytest plugins/h3_live/tests -q   # CPU only, needs torch, safetensors, ffmpeg and ffprobe
```
