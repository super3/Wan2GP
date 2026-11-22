# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""H3 Live: real-time MiniMax H3 as a live stream, in a WanGP tab (see live.py and README.md)."""

import json
from pathlib import Path

import gradio as gr
from shared.utils.plugins import WAN2GPPlugin

from . import live

PlugIn_Name = "H3 Live"
PlugIn_Id = "H3Live"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
HLS_JS = "https://cdn.jsdelivr.net/npm/hls.js@1.5.20/dist/hls.min.js"

INTRO = """### H3 Live: MiniMax H3 clips as one live stream
With **Live mode** on, the next MiniMax H3 video you generate becomes a live session: the clips below are generated back to back
in that one generation, each streamed to the player here while the next renders, and saved as MP4s with `film.mp4` joining them.
Between clips the model and its LoRAs stay on the GPU and nothing is saved in the generation thread.

**Real time** (each clip ready before the previous one has played) was measured with *MiniMax H3 FL2VA Pruned pi-H3 DWD 2-Step*
at 480p on an RTX 5090 (1.19× on WanGP v20.10). It needs a GPU with 32 GB or more and the prompts encoded once (the default here); other H3
models and smaller GPUs work, slower than real time. Generate as usual from the Video Generator tab."""

PLAYER_HTML = ('<video id="h3live-player" controls playsinline style="width:100%;max-height:540px;background:#000;border-radius:8px"></video>'
               '<div style="opacity:.7;font-size:.9em;margin-top:4px">The player follows the latest session. It also opens in VLC or Safari from '
               '<code>live.m3u8</code> in the session folder.</div>')

PLAY_JS = """(url) => {
  const video = document.getElementById('h3live-player');
  if (!url || !video) { return; }
  const start = () => {
    if (window.__h3live_hls) { window.__h3live_hls.destroy(); window.__h3live_hls = null; }
    if (video.canPlayType('application/vnd.apple.mpegurl')) { video.src = url; video.play(); return; }
    const hls = new Hls({ startPosition: 0, xhrSetup: (xhr, address) => {
      if (address.indexOf('.m3u8') >= 0) { xhr.open('GET', address + (address.indexOf('?') >= 0 ? '&' : '?') + 't=' + Date.now(), true); } } });
    window.__h3live_hls = hls;
    hls.loadSource(url);
    hls.attachMedia(video);
    hls.on(Hls.Events.MANIFEST_PARSED, () => video.play());
  };
  if (window.Hls) { start(); } else { const script = document.createElement('script'); script.src = '__HLS_JS__'; script.onload = start; document.head.appendChild(script); }
}""".replace("__HLS_JS__", HLS_JS)


def load_settings() -> live.LiveSettings:
    try:
        return live.LiveSettings.from_dict(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError, TypeError):
        return live.LiveSettings()


def save_settings(settings: live.LiveSettings) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(live.settings_dict(settings), indent=1))
    except OSError as error:
        print(f"[H3 Live] could not save the settings: {error}")


def apply_settings(settings: live.LiveSettings) -> str:
    """Install the hook when live mode is on, and say what will happen."""
    if not settings.enabled:
        live.configure(settings)
        return "Live mode is **off**: H3 generations run as usual."
    try:
        problems = live.install()
    except Exception as error:  # noqa: BLE001 - a WanGP without MiniMax H3, or another change
        live.configure(None)
        return f"**Live mode is not available in this WanGP:** {error}"
    blocking = live.blocking_problems(problems)
    if blocking:
        live.configure(None)
        return "**Live mode is not available in this WanGP:** " + "; ".join(blocking)
    live.configure(settings)
    prompts = [prompt for prompt in settings.prompts if prompt.strip()]
    clips = settings.clips or max(1, len(prompts))
    text = (f"Live mode is **on**: the next MiniMax H3 video generation runs **{clips} clip{'s' if clips != 1 else ''}** "
            f"({'its own prompt' if not prompts else f'{len(prompts)} prompt' + ('s' if len(prompts) != 1 else '') + (', cycled' if clips > len(prompts) else '')}).")
    if problems:
        text += " Note: " + "; ".join(problems) + "."
    return text


def stats_markdown() -> tuple[str, str]:
    session = live.STATE["session"] or live.STATE["last_session"]
    if session is None:
        return "No live session yet.", ""
    summary = session.summary()
    url = live._web_uri()(session.dir / "live_web.m3u8") if live._web_uri() and session.web_playlist is not None else ""
    state = "running" if live.STATE["session"] is session else "done"
    lines = [f"**Session `{Path(summary['dir']).name}`** ({state}, encoder {summary['encoder']}): {summary['clips']} clips, "
             f"{summary['seconds_of_video']} s of video" + (f", **one clip every {summary['cycle_s']} s ({summary['realtime_x']}× real time)**" if summary["cycle_s"] else "")
             + (f", first clip ready at {summary['first_clip_ready_s']} s" if summary["first_clip_ready_s"] is not None else "")
             + (f", {summary['errors']} failed" if summary["errors"] else ""),
             f"Folder: `{summary['dir']}`" + (f" · film: `{summary['film']}`" if summary["film"] else ""), "",
             "| Clip | Seconds | Generated in | Ready at | Encode | Prompt |", "|---|---|---|---|---|---|"]
    for record in summary["records"][-12:]:
        if "error" in record:
            lines.append(f"| {record['clip'] + 1} | | {record['generate_s']} s | | failed | {record['error'][:80]} |")
        else:
            lines.append(f"| {record['clip'] + 1} | {record['seconds']} | {record['generate_s']} s | {record['ready_at_s']} s | {record['encode_s']} s | {record['prompt'][:60]} |")
    return "\n".join(lines), url


class H3LivePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.1.0"
        self.description = "MiniMax H3 clips generated back to back as one live stream (real time with pi-H3 DWD on 32 GB+ GPUs)."

    def setup_ui(self):
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    def create_ui(self):
        settings = load_settings()
        gr.Markdown(INTRO)
        with gr.Row():
            enabled = gr.Checkbox(label="Live mode", value=settings.enabled)
            encoder = gr.Dropdown(label="Encoder", choices=[("Auto (NVENC if available)", "auto"), ("NVENC", "nvenc"), ("x264", "x264")], value=settings.encoder)
            clips = gr.Number(label="Clips (0: one per prompt)", value=settings.clips, precision=0, minimum=0)
            seed_step = gr.Number(label="Seed step between clips", value=settings.seed_step, precision=0)
            keep = gr.Number(label="Sessions kept on disk", value=settings.keep_sessions, precision=0, minimum=1)
        prompts = gr.Textbox(label="Prompts, one line per clip (empty: the generation's own prompt for every clip)", lines=6, value="\n".join(settings.prompts))
        pre_encode = gr.Checkbox(label="Encode the prompts once before the first clip (needed for real time on 32 GB GPUs)", value=settings.pre_encode_prompts)
        status = gr.Markdown(apply_settings(settings))
        gr.HTML(PLAYER_HTML)
        with gr.Row():
            play = gr.Button("Play the latest session", variant="primary")
            stop = gr.Button("Stop after the current clip")
        url = gr.Textbox(label="Browser playlist of the latest session", interactive=False)
        stats = gr.Markdown("No live session yet.")
        timer = gr.Timer(2.0)

        def changed(enabled_value, prompts_value, clips_value, seed_step_value, encoder_value, keep_value, pre_encode_value):
            updated = live.LiveSettings(enabled=bool(enabled_value), prompts=[line.strip() for line in (prompts_value or "").splitlines() if line.strip()],
                                        clips=max(0, int(clips_value or 0)), seed_step=int(seed_step_value or 0), encoder=encoder_value or "auto",
                                        keep_sessions=max(1, int(keep_value or 1)), pre_encode_prompts=bool(pre_encode_value))
            save_settings(updated)
            return apply_settings(updated)

        inputs = [enabled, prompts, clips, seed_step, encoder, keep, pre_encode]
        for component in inputs:
            component.change(changed, inputs=inputs, outputs=[status])
        stop.click(lambda: "Stopping after the current clip." if live.request_stop() else "No live session is running.", outputs=[status])
        play.click(fn=None, inputs=[url], outputs=None, js=PLAY_JS)
        timer.tick(stats_markdown, outputs=[stats, url])
