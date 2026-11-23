# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Speedups: keep the model loaded between queued tasks of the same model (see README.md)."""

import json
from pathlib import Path

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

from . import hooks

PlugIn_Name = "Speedups"
PlugIn_Id = "Speedups"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

INTRO = """### Speedups for queued generations
Nothing in WanGP is edited: the plugin hooks into it while it runs, and removing the plugin removes the hook.

- **Keep the model loaded**: when the next queued task uses the same model, its models stay in VRAM between the two (no unload, no
  reload, no full garbage collection). It also keeps the transformer in VRAM while the VAEs and encoders run; that part applies to
  models loaded after it is turned on, and needs a GPU with room for both.

WanGP itself (v20.12) now does what this plugin's other settings did: **Configuration → Outputs → Hardware Video Encoding** (NVENC)
and **Save Videos in the Background**, and **Configuration → RAM/VRAM Management → Keep Loras Loaded Between Tasks**."""


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
        return f"**Not active**: this WanGP lacks {', '.join(problems)}."
    return "Active."


class SpeedupsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.2.0"
        self.description = "Keeps the model loaded between queued tasks of the same model."
        self.problems = []

    def setup_ui(self):
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
        gr.Markdown(INTRO)
        keep_model = gr.Checkbox(label="Keep the model loaded", value=hooks.settings().keep_model_loaded)
        status = gr.Markdown(status_markdown(self.problems))

        def changed(keep_model_value):
            updated = hooks.Settings(keep_model_loaded=bool(keep_model_value))
            hooks.configure(updated)
            save_settings(updated)
            return status_markdown(self.problems)

        keep_model.change(changed, inputs=[keep_model], outputs=[status])
