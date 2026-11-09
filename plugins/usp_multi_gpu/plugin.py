# SPDX-License-Identifier: Apache-2.0
"""WanGP tab: run MiniMax H3's denoise on several GPUs with Ulysses sequence parallelism."""

import os
import sys

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Multi-GPU (USP)"
PlugIn_Id = "UspMultiGpu"

INTRO = """### Multi-GPU for MiniMax H3 (Ulysses sequence parallelism)
Splits each MiniMax H3 denoising step across several GPUs. Every GPU holds the whole model, works on its share of the
sequence and trades attention heads with the others twice per block. The output matches a single GPU.

**It only pays off when the GPUs talk to each other directly** (NVLink, or PCIe peer-to-peer). Without it, the exchange
goes through system RAM: on 2× RTX 5090 (PCIe 4.0 x16, no P2P) a 5 s FastH3 clip took **57.8 s instead of 19.6 s**.
GeForce cards have no P2P, so with them, run one WanGP per GPU instead (one job each).

- Supported: MiniMax H3 models with sdpa / sage2 attention, and FastH3 (VSA). Not supported: Sol, Veda and VDN
  sparse attention, step caches (Spectrum / FirstBlockCache), ControlNet. Those stop with an error under USP.
- Each extra GPU runs a helper WanGP process with its own copy of the weights in RAM (about 40–50 GB each for H3).
- Generations started from the queue use the GPUs; other models run on this GPU alone, as usual.
"""


def _launcher():
    from . import launcher

    return launcher


class UspMultiGpuPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self._orig_generate_media = None

    def setup_ui(self):
        self.request_global("args")
        self.request_global("generate_media")
        self.request_global("release_model")
        self.request_global("is_generation_in_progress")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    # -- UI -------------------------------------------------------------------
    def create_ui(self):
        gr.Markdown(INTRO)
        inventory = gr.Markdown(self._inventory_text())
        with gr.Row():
            world = gr.Dropdown(label="GPUs", choices=self._world_choices(), value=self._default_world(), interactive=True)
            helper_gpus = gr.Textbox(label="Helper GPUs (indices, empty = the next ones)", value="")
        with gr.Row():
            start = gr.Button("Start multi-GPU", variant="primary")
            stop = gr.Button("Stop")
            refresh = gr.Button("Refresh")
        status = gr.Markdown(self._status_text())
        start.click(self._start, inputs=[world, helper_gpus], outputs=[status])
        stop.click(self._stop, inputs=[], outputs=[status])
        refresh.click(lambda: (self._inventory_text(), self._status_text()), inputs=[], outputs=[inventory, status])

    def _rank0_gpu(self) -> int:
        import torch

        requested = str(getattr(getattr(self, "args", None), "gpu", "") or "").strip()
        if requested:
            return int(requested.split(":")[-1]) if requested.split(":")[-1].isdigit() else 0
        return torch.cuda.current_device() if torch.cuda.is_available() else 0

    def _inventory_text(self) -> str:
        try:
            gpus = _launcher().gpus()
        except Exception as exc:
            return f"Could not list GPUs: {exc}"
        if len(gpus) < 2:
            return f"**{len(gpus)} GPU visible to WanGP:** multi-GPU needs at least 2."
        rank0 = self._rank0_gpu()
        lines = ["| GPU | Name | Memory | Direct access to/from this WanGP's GPU |", "|---|---|---|---|"]
        for gpu in gpus:
            if gpu["index"] == rank0:
                link = "this WanGP's GPU"
            else:
                link = "yes (P2P)" if gpu["p2p"].get(rank0) else "**no**: traffic goes through system RAM"
            lines.append(f"| {gpu['index']} | {gpu['name']} | {gpu['memory_gb']} GB | {link} |")
        return "\n".join(lines)

    def _world_choices(self):
        try:
            count = len(_launcher().gpus())
        except Exception:
            count = 0
        return [size for size in _launcher().VALID_WORLD_SIZES if size <= count] or [2]

    def _default_world(self):
        return self._world_choices()[0]

    def _status_text(self) -> str:
        running = _launcher().running()
        if running is None:
            return "**Stopped.** MiniMax H3 runs on one GPU."
        gpus = ", ".join(f"GPU {gpu} ({name})" for gpu, name in zip([running.rank0_gpu] + running.helper_gpus, running.names, strict=True))
        broken = getattr(running.leader, "broken", None)
        state = f"**Broken:** {broken}. Stop, then start again." if broken else "**Running.**"
        return f"{state} MiniMax H3 queue generations use {running.world} GPUs: {gpus}.\n\nHelper logs: " + \
            ", ".join(f"`{log}`" for log in running.logs)

    # -- actions --------------------------------------------------------------
    def _busy(self) -> bool:
        check = getattr(self, "is_generation_in_progress", None)
        return bool(check()) if callable(check) else False

    def _start(self, world, helper_gpus):
        launcher = _launcher()
        if launcher.running() is not None:
            yield self._status_text()
            return
        if self._busy():
            yield "A generation is running. Start multi-GPU when the queue is idle."
            return
        world = int(world)
        rank0 = self._rank0_gpu()
        try:
            count = len(launcher.gpus())
            if helper_gpus.strip():
                helpers = [int(token) for token in helper_gpus.replace(",", " ").split()]
            else:
                helpers = [index for index in range(count) if index != rank0][: world - 1]
            if any(index < 0 or index >= count for index in helpers):
                raise ValueError(f"GPU indices must be between 0 and {count - 1}")
        except ValueError as exc:
            yield f"**Not started:** {exc}"
            return
        messages = []

        def status(text):
            messages.append(text)

        yield "Starting…"
        self._release_model()
        root = os.path.abspath(os.getcwd())
        cli_args = launcher.inherited_cli_args(sys.argv[1:])
        try:
            launcher.start(world, rank0, helpers, root, cli_args, status=status)
        except Exception as exc:
            yield f"**Could not start multi-GPU:** {exc}"
            return
        self._wrap_generate_media()
        yield self._status_text() + "\n\nThe next MiniMax H3 generation loads the model on every GPU."

    def _stop(self):
        if self._busy():
            return "A generation is running. Stop multi-GPU when the queue is idle."
        self._unwrap_generate_media()
        _launcher().stop()
        self._release_model()
        return self._status_text()

    def _release_model(self):
        release = getattr(self, "release_model", None)
        if callable(release):
            release()  # the next load builds the model under the current (patched or restored) classes

    # -- queue hook -----------------------------------------------------------
    def _wrap_generate_media(self):
        if self._orig_generate_media is not None:
            return
        original = self.generate_media
        launcher = _launcher()

        def generate_media(task, send_cmd, *args, **kwargs):
            lead = launcher.leader()
            if lead is None:
                return original(task, send_cmd, *args, **kwargs)
            lead.begin_task(task.get("params", {}) if isinstance(task, dict) else {})
            try:
                return original(task, send_cmd, *args, **kwargs)
            finally:
                lead.end_task()

        self._orig_generate_media = original
        self.set_global("generate_media", generate_media)

    def _unwrap_generate_media(self):
        if self._orig_generate_media is None:
            return
        self.set_global("generate_media", self._orig_generate_media)
        self._orig_generate_media = None
