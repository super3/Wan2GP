# SPDX-License-Identifier: Apache-2.0
"""Chunk Streaming plugin for WanGP.

Wraps models/minimax_h3/streaming.py (an activation-style patch that touches
no core files): the MiniMax H3 video VAE finalizes 17 frames (~0.7 s of video)
per temporal decode chunk, and with the plugin enabled each finished chunk is
muxed into a standalone fragmented MP4 segment in a background thread while
the rest of the clip is still decoding. Audio is decoded before video so every
segment is playable the moment it lands. Segments appear under
`<output>/streaming/gen_NNN/` during generation; a player (MSE/HLS) can start
as soon as the estimated no-rebuffer start time passes instead of waiting for
the full clip.

Measured on an RTX 5090 at 832x480 with the 4-step turbo LoRA: perceived time
to first frame drops 1.2x to 1.31x (6 to 14 seconds) with zero rebuffering.
"""

import os
import time

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Chunk Streaming"
PlugIn_Id = "ChunkStreaming"


def _streaming():
    from models.minimax_h3 import streaming

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
        status = gr.Markdown("Disabled.")
        with gr.Row():
            refresh = gr.Button("Refresh last generation stats")
        stats = gr.Dataframe(
            headers=["segment", "frames", "seconds of video", "ready at (s)", "mux (s)", "audio"],
            interactive=False,
        )
        summary = gr.Markdown("")
        enabled.change(self._toggle, inputs=[enabled, out_dir], outputs=[status])
        refresh.click(self._refresh, inputs=[], outputs=[stats, summary])

    def _toggle(self, enable: bool, out_dir: str):
        streaming = _streaming()
        if enable:
            out_dir = (out_dir or "").strip() or os.path.join(os.path.abspath("outputs"), "streaming")
            os.makedirs(out_dir, exist_ok=True)
            streaming.activate(store_frames=False, live_dir=out_dir, audio_first=True)
            return (
                f"**Enabled.** Segments will appear under `{out_dir}/gen_NNN/` during the "
                "next MiniMax H3 generation."
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
        return rows, note
