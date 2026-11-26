# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Director: a tab that runs director.py (a planned multi-shot MiniMax H3 film on several GPUs, streamed in order; see README.md)."""

import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Director"
PlugIn_Id = "Director"
HERE = Path(__file__).resolve().parent
WANGP_ROOT = HERE.parents[1]
EXAMPLES = {"Text only (no assets)": HERE / "examples" / "textonly_plan.json", "With reference images": HERE / "examples" / "plan.json",
            "Cut-away (one source image)": HERE / "examples" / "cutaway_plan.json"}
MODES = {"Dry run: jobs, prompts and a GPU estimate": ["--dry-run"], "Rehearsal: test patterns, no GPU": ["--backend", "fake", "--fake-seconds", "2"],
         "Render": []}
RUN = {"process": None, "log": None, "out": None}

INTRO = """### Director: a planned MiniMax H3 film, rendered on several GPUs and streamed in order
A plan lists the film's shots. The director makes one storyboard generation (or, in cut-away mode, none), then one H3 clip per shot,
spread over the GPUs listed (one WanGP process each), and appends the clips in film order to an HLS stream while later ones still
render. `film.mp4` joins them at the end. The plan format, the modes and the measured runs are in `plugins/director/README.md`.

Each GPU runs its own WanGP process: unload this WebUI's model first, or list other GPUs. **Rehearsal** runs the whole pipeline with
ffmpeg test patterns (no GPU); **Dry run** prints the jobs and estimates the first-clip latency per GPU count."""


def command(plan_path, out, gpus, mode, encoder, parallel_start) -> list[str]:
    args = [sys.executable, str(HERE / "director.py"), str(plan_path), "--out", str(out), "--gpus", gpus.strip() or "0", *MODES[mode]]
    if encoder:
        args += ["--encoder", encoder]
    if parallel_start:
        args.append("--parallel-start")
    return args


def running() -> bool:
    return RUN["process"] is not None and RUN["process"].poll() is None


def start(plan_choice, custom_plan, out, gpus, mode, encoder, parallel_start):
    if running():
        return "A run is already going: stop it first.", read_log()
    plan_path = Path(custom_plan.strip()) if custom_plan and custom_plan.strip() else EXAMPLES[plan_choice]
    if not plan_path.is_file():
        return f"No plan at `{plan_path}`.", ""
    out = Path(out.strip() or WANGP_ROOT / "outputs" / "director" / plan_path.stem)
    if not out.is_absolute():
        out = WANGP_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    args = command(plan_path, out, gpus, mode, encoder, parallel_start)
    log_path = out / "director.log"
    log = open(log_path, "w", encoding="utf-8")
    log.write("$ " + " ".join(shlex.quote(arg) for arg in args) + "\n")
    log.flush()
    RUN.update(process=subprocess.Popen(args, cwd=WANGP_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True), log=log_path, out=out)
    threading.Thread(target=lambda process=RUN["process"], handle=log: (process.wait(), handle.close()), daemon=True).start()
    return status(), read_log()


def stop():
    if running():
        RUN["process"].terminate()
    return status(), read_log()


def read_log(lines: int = 60) -> str:
    if RUN["log"] is None or not Path(RUN["log"]).is_file():
        return ""
    return "\n".join(Path(RUN["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def status() -> str:
    if RUN["process"] is None:
        return "No run yet."
    out = RUN["out"]
    stream, film = out / "stream" / "playlist.m3u8", out / "film.mp4"
    state = "Running" if running() else f"Finished (exit code {RUN['process'].returncode})"
    parts = [f"**{state}**: `{out}`"]
    if stream.is_file():
        parts.append(f"stream: `{stream}` (open it in VLC, or `ffplay`)")
    if film.is_file():
        parts.append(f"film: `{film}`")
    return " · ".join(parts)


class DirectorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.1.0"
        self.description = "A planned multi-shot MiniMax H3 film, rendered on several GPUs and streamed in film order."

    def setup_ui(self):
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    def create_ui(self):
        gr.Markdown(INTRO)
        with gr.Row():
            plan_choice = gr.Dropdown(label="Plan", choices=list(EXAMPLES), value=next(iter(EXAMPLES)))
            custom_plan = gr.Textbox(label="Or a plan file (JSON)", placeholder="/path/to/plan.json")
        with gr.Row():
            out = gr.Textbox(label="Run folder (empty: outputs/director/<plan name>)")
            gpus = gr.Textbox(label="GPUs (e.g. 0,1,2,3)", value=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
        with gr.Row():
            mode = gr.Radio(label="Mode", choices=list(MODES), value=next(iter(MODES)))
            encoder = gr.Dropdown(label="Clip encoding", choices=[("x264 veryfast (default)", ""), ("NVENC", "nvenc"), ("WanGP's settings", "wangp")], value="")
            parallel_start = gr.Checkbox(label="Parallel start (weights already downloaded)", value=False)
        with gr.Row():
            start_button = gr.Button("Start", variant="primary")
            stop_button = gr.Button("Stop")
        state = gr.Markdown(status())
        log = gr.Textbox(label="Log", lines=18, max_lines=18, interactive=False, value=read_log())
        timer = gr.Timer(2.0)
        start_button.click(start, inputs=[plan_choice, custom_plan, out, gpus, mode, encoder, parallel_start], outputs=[state, log])
        stop_button.click(stop, outputs=[state, log])
        timer.tick(lambda: (status(), read_log()), outputs=[state, log])
