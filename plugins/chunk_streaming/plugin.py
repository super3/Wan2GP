# SPDX-License-Identifier: Apache-2.0
"""Chunk Streaming plugin for WanGP.

Wraps streaming.py in this folder (an activation-style patch that touches
no core files): the MiniMax H3 video VAE finalizes 17 frames (~0.7 s of video)
per temporal decode chunk, and with the plugin enabled each finished chunk is
muxed into a standalone fragmented MP4 segment in a background thread while
the rest of the clip is still decoding. Audio is decoded before video so every
segment is playable the moment it lands. Segments appear under
`<output>/streaming/gen_NNN/` during generation; a player (MSE/HLS) can start
as soon as the estimated no-rebuffer start time passes instead of waiting for
the full clip.

Measured on one RTX 5090 with FastH3 Omni NVFP4 (8 steps, Nov 2026): the earliest
playback start that never stalls comes 14-31% sooner than the finished clip,
from 18-31% with the original VAE at 480p-720p, 5-10 s, down to 14-20% with
LightVAE INT8, whose faster decode leaves less to hide (5 s at 480p: 13.4 s
instead of 17.3 s). An earlier run with the 4-step turbo LoRA gave 1.2-1.31x at 480p.
"""

import os
import time

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Chunk Streaming"
PlugIn_Id = "ChunkStreaming"


def _streaming():
    from . import streaming

    return streaming


class ChunkStreamingPlugin(WAN2GPPlugin):
    def setup_ui(self):
        self.request_component("state")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    def create_ui(self):
        gr.Markdown(
            "### Chunk Streaming (MiniMax H3)\n"
            "Streams finished decode chunks as standalone fMP4 segments while the "
            "tail of the clip is still decoding. Audio is decoded first so every "
            "segment is playable when it lands. Only affects MiniMax H3 generations; "
            "the final saved video is unchanged (bit-identical decode)."
        )
        default_dir = os.path.join(os.path.abspath("outputs"), "streaming")
        with gr.Row():
            enabled = gr.Checkbox(label="Enable chunk streaming", value=False)
            out_dir = gr.Textbox(label="Segment output directory", value=default_dir)
        decode_devices = gr.Textbox(
            label="Multi-GPU decode devices (optional)", value="", placeholder="cuda:0, cuda:1",
            info="Decode the video chunks round-robin across these GPUs, so chunks land sooner; the output is unchanged. "
                 "Each GPU other than the one WanGP runs on keeps its own copy of the VAE decoder weights. Empty: one GPU.")
        status = gr.Markdown("Disabled.")
        with gr.Row():
            refresh = gr.Button("Refresh last generation stats")
        stats = gr.Dataframe(
            headers=["segment", "frames", "seconds of video", "ready at (s)", "mux (s)", "audio"],
            interactive=False,
        )
        summary = gr.Markdown("")
        enabled.change(self._toggle, inputs=[enabled, out_dir, decode_devices], outputs=[status])
        decode_devices.submit(self._toggle, inputs=[enabled, out_dir, decode_devices], outputs=[status])
        decode_devices.blur(self._toggle, inputs=[enabled, out_dir, decode_devices], outputs=[status])
        refresh.click(self._refresh, inputs=[], outputs=[stats, summary])

    def _toggle(self, enable: bool, out_dir: str, decode_devices: str = ""):
        streaming = _streaming()
        if enable:
            out_dir = (out_dir or "").strip() or os.path.join(os.path.abspath("outputs"), "streaming")
            os.makedirs(out_dir, exist_ok=True)
            try:
                streaming.activate(store_frames=False, live_dir=out_dir, audio_first=True, decode_devices=decode_devices)
            except (RuntimeError, ValueError) as exc:  # an unparsable device name
                return f"**Not enabled:** invalid decode devices `{decode_devices}`: {exc}"
            devices = streaming.decode_devices()
            split = f" Video chunks decode across {', '.join(devices)}." if devices else ""
            return (
                f"**Enabled.** Segments will appear under `{out_dir}/gen_NNN/` during the "
                f"next MiniMax H3 generation.{split}"
            )
        streaming.deactivate()
        return "Disabled."

    def _refresh(self):
        streaming = _streaming()
        rec = streaming.recorder()
        if rec is None:
            return [], "Streaming is not enabled."
        live = rec.live
        if live is None or not live.results:
            return [], "No completed generation recorded yet."
        t0 = rec.decode_start_t or 0.0
        rows = []
        for seg in live.results:
            rows.append([
                seg["segment"],
                seg["frame_end"] - seg["frame_start"],
                round(seg["duration_s"], 3),
                round(seg["t_done"] - t0, 3) if seg.get("t_done") else None,
                seg.get("mux_s"),
                "yes" if seg.get("has_audio") else "no",
            ])
        ready = [s["t_done"] - t0 for s in live.results if s.get("t_done")]
        durations = [s["duration_s"] for s in live.results]
        note = ""
        if ready and len(ready) == len(durations):
            est = streaming.linear_estimate_start(ready, durations, observe=2, pad=1.15)
            oracle = streaming.no_buffer_start(ready, durations)
            note = (
                f"Relative to decode start: playback could begin at **{est['start_s']:.2f}s** "
                f"(padded linear estimate; oracle {oracle:.2f}s) "
                f"{'with' if not est['would_rebuffer'] else 'WITH RISK OF'} zero rebuffering. "
                f"Audio-first decode {'active' if rec.audio_first_effective else 'fell back to default order'}; "
                f"audio decode took {rec.audio_s:.2f}s. Generated {time.strftime('%H:%M:%S')}."
            )
            plan = rec.decode_plan
            if plan:
                note += (f" Multi-GPU decode fell back to {plan['devices'][0]}: {plan['fallback']}." if plan.get("fallback")
                         else f" Video decoded across {', '.join(plan['devices'])}.")
        return rows, note
