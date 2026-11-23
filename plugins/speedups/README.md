# Speedups

A WanGP plugin that keeps the model loaded between queued tasks of the same model. It changes nothing in WanGP's files: it hooks into WanGP while it runs, so removing the plugin leaves WanGP exactly as it is.

| Setting | What it does | Where |
|---|---|---|
| **Keep the model loaded** | When the next queued task uses the same model, its models stay in VRAM between the two: no unload, no reload, no full garbage collection. | Speedups tab |

It is off by default.

Earlier versions also had NVENC output codecs, background save and kept LoRAs. WanGP v20.12 does all three itself, so they were removed:

| Was in Speedups | In WanGP v20.12 |
|---|---|
| NVENC codecs | Configuration → Outputs → **Hardware Video Encoding** (Auto by default) encodes the x264 and x265 choices with NVENC. |
| Background save | Configuration → Outputs → **Save Videos in the Background** |
| Keep the LoRAs loaded | Configuration → RAM/VRAM Management → **Keep Loras Loaded Between Tasks** (Auto by default) |

A codec saved as *H.264 NVENC* or *HEVC NVENC* with an earlier version is now saved with x264 CRF 10 on the CPU (WanGP's fallback for codecs it does not know). Pick an x264 or x265 codec again in Configuration → Outputs, and Hardware Video Encoding moves it to NVENC.

WanGP v20.12 keeps a model loaded between queued tasks only for the models that never swap models (`no_model_swap`, such as HiDream and the TTS models). This plugin does it for the others, MiniMax H3 and Wan included.

## Install

This folder goes in WanGP's `plugins/` folder (as `plugins/speedups`). In WanGP, open **Plugins**, enable **Speedups** and restart WanGP.

## Details

- After an abort, a failure, the end of the queue, or when the next task uses another model, the models are unloaded as usual.
- It also keeps the transformer in VRAM while the VAEs and text encoders run (mmgp's co-tenant map). That part applies to models loaded after the setting is turned on, and needs a GPU with room for both.
- LoRAs loaded into a model kept in VRAM get their GPU copies refreshed (mmgp's `refresh_loras`), as WanGP does for its own `no_model_swap` models. LoRAs that WanGP reuses as they are (Keep Loras Loaded Between Tasks) are only activated again and need nothing.

## How it hooks in

`hooks.install()` replaces names in `wgp.py`'s globals, which WanGP's code looks up when it runs. It wraps `generate_media`, and puts proxies in place of `offload` (mmgp) and `gc`.

Each hook checks that it is called from `generate_media` at the line it expects. When something does not match, it does what WanGP does without the plugin. `tests/test_wgp_compat.py` reads this checkout's `wgp.py` and mmgp source and fails if what the hooks expect has moved.

WanGP loads plugins only in the WebUI. A script using WanGP's API installs the hooks itself:

```python
import sys; sys.path.insert(0, "plugins")
from speedups import hooks
hooks.install(session._ensure_runtime().module.__dict__, hooks.Settings(keep_model_loaded=True))
```

## Measured

Before v20.12. Keep the model loaded alone, built into `wgp.py` before it became this plugin, MiniMax H3 pi-H3 2-step, 832×480, 124-frame clips in a queue on an RTX PRO 6000: 7.35 → 5.97 s per clip (0.70× → 0.87× real time); the time before the first denoising step went from 0.81 to 0.17 s.

The plugin on WanGP v20.10, with every setting of its earlier version on (H.264 NVENC, background save, kept model and LoRAs), on an RTX 5090: MiniMax H3 FL2VA Pruned pi-H3 DWD 2-Step, 832×480, 124 frames, a queue of 4 clips, 6.95 s per clip without the plugin and 4.75 s with it (1.09× real time). The model was unloaded once instead of 8 times. On v20.12, WanGP's own NVENC, background save and kept LoRAs replace those three settings; that combination has not been measured.

## Tests

```bash
python -m pytest plugins/speedups/tests -q   # CPU only
```
