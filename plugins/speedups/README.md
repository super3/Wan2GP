# Speedups

A WanGP plugin that makes queued generations faster. It changes nothing in WanGP's files: it hooks into WanGP while it runs, so removing the plugin (or WanGP adopting any of this) leaves WanGP exactly as it is.

| Setting | What it does | Where |
|---|---|---|
| **NVENC codecs** | *H.264 NVENC* and *HEVC NVENC* output codecs: the GPU's video encoder writes each video in a fraction of x264's time. | Configuration → Outputs → SDR Video Codec |
| **Background save** | A plain generated video is written (encode, sound, metadata, gallery entry) while the next queued task already runs. | Speedups tab |
| **Keep the model loaded** | When the next queued task uses the same model, its models stay in VRAM between the two: no unload, no reload, no full garbage collection. | Speedups tab |
| **Keep the LoRAs loaded** | When the next queued task selects the same LoRAs with the same multipliers, they are reused instead of being read, converted and pinned again. | Speedups tab |

All four are off by default.

## Install

This folder goes in WanGP's `plugins/` folder (as `plugins/speedups`). In WanGP, open **Plugins**, enable **Speedups** and restart WanGP.

## Details

- **NVENC** is used only where it works: the ffmpeg on PATH (or `FFMPEG_BINARY`) must encode a short test clip with it, checked once per process. On a GPU without NVENC (A100, H100) or an ffmpeg without it, videos are saved with x264 CRF 10 and the console says why. Quality: constant quality 14 (H.264) / 16 (HEVC), preset p5, tune hq, close to x264 CRF 10. If the plugin is removed, a saved NVENC setting falls back to x264 CRF 10.
- **Background save** applies to a plain generated video, with or without its generated sound. Anything else is written before the next task starts, and the console says why: HDR, several sliding windows, a continued Source Video, source or control audio, soundtrack or voice processing, side files, FlashVSR continuation. It holds one finished clip's frames in memory while they are written, and at most one more waits behind it. The last task of a queue returns only once every file is written. A failed save is reported then, and its file is not added to the gallery.
- **Keep the model loaded** also keeps the transformer in VRAM while the VAEs and text encoders run (mmgp's co-tenant map). That part applies to models loaded after the setting is turned on, and needs a GPU with room for both. After an abort, a failure, or when the next task uses another model, the models are unloaded as usual.
- **Keep the LoRAs loaded** compares the files (path, size, modification time), the multipliers and the pinning. A model kept in VRAM gets the GPU copies of newly loaded LoRAs refreshed, as WanGP does for its own single-model families.

## How it hooks in

`hooks.install()` replaces names in `wgp.py`'s globals, which WanGP's code looks up when it runs. It wraps `generate_media`, `save_video`, the audio mux, `record_file_metadata` and `store_api_output_artifact`, and puts proxies in place of `offload` (mmgp) and `gc`. It also patches `ContinuedVideo.append`.

Each hook checks that it is called from `generate_media` at the line it expects. When something does not match, it does what WanGP does without the plugin. `tests/test_wgp_compat.py` reads this checkout's `wgp.py`, mmgp and `ContinuedVideo` source and fails if what the hooks expect has moved.

WanGP loads plugins only in the WebUI. A script using WanGP's API installs the hooks itself:

```python
import sys; sys.path.insert(0, "plugins")
from speedups import hooks, nvenc
nvenc.install()
hooks.install(session._ensure_runtime().module.__dict__, hooks.Settings(background_save=True, keep_model_loaded=True, keep_loras_loaded=True))
```

## Measured

The plugin, on an RTX 5090 with WanGP v20.10 (Nov 22). MiniMax H3 FL2VA Pruned pi-H3 DWD 2-Step, 832×480, 124 frames, one API job of 4 queued clips after a warm-up job:

| | WanGP as is | With Speedups (all on, H.264 NVENC) |
|---|---|---|
| Time per queued clip | 6.95 s (0.74× real time) | **4.75 s (1.09×)** |
| Model unloads | 8 | 1, at the end of the queue |
| LoRA loads (the pi-H3 adapter) | 4 | 1 |
| Files | 4, x264 | 4, NVENC, about a quarter smaller |

- Each background save took about 1.0 s and never had to wait.
- Every clip reached the gallery with its sound.
- The first queued clip took 5.1 s: it reloaded the model, which the warm-up job had unloaded at its own end.
- The last took 5.4 s, including the wait for the last files.

Earlier, with the same changes built into `wgp.py`, before they became this plugin. MiniMax H3 pi-H3 2-step, 832×480 5 s clips, in queues on an RTX PRO 6000:

| Time per queued clip | |
|---|---|
| WanGP as before (x264 CRF 10, LoRAs reloaded) | 7.37 s |
| + NVENC, keep LoRAs loaded | 6.89 s |
| + background save | 6.19 s |
| keep the model loaded (another host, 124 frames) | 7.35 → 5.97 s |

## Tests

```bash
python -m pytest plugins/speedups/tests -q   # CPU only
```
