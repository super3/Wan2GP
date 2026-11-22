# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Speedups: NVENC output codecs, background save, keep the model and the LoRAs loaded between queued tasks (see README.md)."""

import json
from pathlib import Path

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

from . import hooks, nvenc

PlugIn_Name = "Speedups"
PlugIn_Id = "Speedups"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

INTRO = """### Speedups for queued generations
Nothing in WanGP is edited: the plugin hooks into it while it runs, and removing the plugin removes them.

- **NVENC codecs**: pick *H.264 NVENC* or *HEVC NVENC* in **Configuration → Outputs → SDR Video Codec**. The GPU's video encoder
  writes each video in a fraction of x264's time. Where NVENC cannot encode (no NVENC on the GPU, or an ffmpeg without it), videos are
  saved with x264 CRF 10 and the console says why.
- **Background save**: a plain generated video is written (encode, sound, metadata, gallery entry) while the next queued task already
  runs. It holds one finished clip's frames in memory meanwhile. The end of a queue still waits for every file.
- **Keep the model loaded**: when the next queued task uses the same model, its models stay in VRAM between the two (no unload, no
  reload, no full garbage collection). It also keeps the transformer in VRAM while the VAEs and encoders run; that part applies to
  models loaded after it is turned on, and needs a GPU with room for both.
- **Keep the LoRAs loaded**: when the next queued task selects the same LoRAs with the same multipliers, they are reused instead of
  being read, converted and pinned again.

Measured with the same changes built into wgp.py (before they became this plugin), MiniMax H3 pi-H3 2-step, 832×480 5 s clips, RTX
PRO 6000, time per queued clip: NVENC and kept LoRAs 7.37 → 6.89 s, with background save 6.19 s; keeping the model loaded (another
host) 7.35 → 5.97 s."""


def load_settings() -> hooks.Settings:
    try:
        return hooks.Settings.from_dict(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError, TypeError):
        return hooks.Settings()


def save_settings(settings: hooks.Settings) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings.to_dict(), indent=1))
    except OSError as error:
        print(f"[Speedups] could not save the settings: {error}")


def status_markdown(problems) -> str:
    if problems:
        return f"**Not active**: this WanGP lacks {', '.join(problems)}. NVENC codecs still work."
    saves = [record for record in hooks.saver.history[-5:] if record.get("end_t")]
    if not saves:
        return "Active. No background save yet."
    lines = ["Active. Last background saves:", "", "| File | Waited | Written in | Result |", "|---|---|---|---|"]
    for record in reversed(saves):
        waited, took = record["start_t"] - record["queued_t"], record["end_t"] - record["start_t"]
        lines.append(f"| {record['label'][-60:]} | {waited:.1f} s | {took:.1f} s | {record['error'] or 'OK'} |")
    return "\n".join(lines)


class SpeedupsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.1.0"
        self.description = "NVENC codecs, background save, and the model and LoRAs kept loaded between queued tasks."
        self.problems = []

    def setup_ui(self):
        nvenc.install()  # before the Configuration tab builds its codec list
        hooks.configure(load_settings())
        self.request_global("get_gen_info")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    def post_ui_setup(self, components):
        self.install()
        return {}

    def install(self):
        if hooks.STATE["installed"]:
            return
        wgp_function = self._set_wgp_global_func or getattr(self, "get_gen_info", None)
        if wgp_function is None:
            self.problems = ["access to its globals"]
            return
        self.problems = hooks.install(wgp_function.__globals__)
        if self.problems:
            print(f"[Speedups] not active: this WanGP lacks {', '.join(self.problems)}")

    def create_ui(self):
        self.install()
        settings = hooks.settings()
        gr.Markdown(INTRO)
        with gr.Row():
            background = gr.Checkbox(label="Background save", value=settings.background_save)
            keep_model = gr.Checkbox(label="Keep the model loaded", value=settings.keep_model_loaded)
            keep_loras = gr.Checkbox(label="Keep the LoRAs loaded", value=settings.keep_loras_loaded)
        status = gr.Markdown(status_markdown(self.problems))
        timer = gr.Timer(3.0)

        def changed(background_value, keep_model_value, keep_loras_value):
            updated = hooks.Settings(keep_model_loaded=bool(keep_model_value), keep_loras_loaded=bool(keep_loras_value), background_save=bool(background_value))
            hooks.configure(updated)
            save_settings(updated)
            return status_markdown(self.problems)

        inputs = [background, keep_model, keep_loras]
        for component in inputs:
            component.change(changed, inputs=inputs, outputs=[status])
        timer.tick(lambda: status_markdown(self.problems), outputs=[status])
