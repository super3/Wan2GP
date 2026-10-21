#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chunk-streaming bench: end-to-end wall vs perceived time-to-first-frame.

Runs the production H3 480p clip (sage2, profile 4, 4-step turbo LoRA) at
5/10/15 seconds (124/243/362 frames), each once as today's monolithic path and
once with models/minimax_h3/streaming.py instrumentation active. The streaming
leg records when every finished decode chunk (17 frames, ~0.7 s of video)
could have left the worker, muxes each chunk into a standalone fMP4 segment
(timed), and models the player two ways: the post-hoc oracle earliest
no-rebuffer start, and the deployable linear-rate estimator with padding.

Segment-ready times shift every video chunk by the measured audio decode
duration (a streaming pipeline decodes audio first; execution is serial on one
GPU so the shift is exact) and add each segment's own mux cost (conservative:
a real worker muxes segment i while chunk i+1 decodes).

Requires the worker image environment (runpod_worker + weights). Prints one
STREAM_LEG JSON line per leg and a final STREAM_TABLE summary, POSTed to
$LOG_SHIP_URL if set.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

_ROOT = os.environ.get("WANGP_ROOT") or "/opt/wangp"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PROMPT = (
    "integrated_multimodal_description: [Shot 1] A cinematic close-up of a "
    "brass desk lamp switching on in a dark study, dust drifting through the beam. The "
    "camera pushes in slowly.\n"
    "overall_soundscape: A single click, a low electrical hum, faint room tone.\n"
    "non_diegetic_music: One soft sustained cello note."
)
ACCEL_PROFILE = "Turbo Lightx2v FL2V 4 Steps v1.0 768p"
FPS = 24
SEED = 12345


def _legs() -> list[tuple[str, int, str]]:
    """(resolution, frames, tag) legs from $STREAM_LEGS ("WxH:frames,..").

    The frame lattice is 107 + 17k (124 = 5.2 s at 24 fps). The default is the
    resolution sweep at the 5 s production clip length.
    """
    spec = os.environ.get("STREAM_LEGS", "832x480:124,1280x720:124,1920x1088:124")
    legs = []
    for part in spec.split(","):
        resolution, frames = part.strip().split(":")
        legs.append((resolution, int(frames), f"{resolution}-{frames}f"))
    return legs


def _ship(payload: dict) -> None:
    url = os.environ.get("LOG_SHIP_URL", "").strip()
    if not url:
        return
    payload = dict(payload)
    payload.setdefault("run_id", os.environ.get("BENCH_RUN_ID", ""))
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"records": [payload]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"bench: shipping failed: {exc!r}", file=sys.stderr)


def _job(resolution: str, frames: int, tag: str) -> dict:
    return {
        "id": f"streambench-{tag}-{SEED}",
        "input": {
            "model_type": os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned"),
            "profile": ACCEL_PROFILE,
            "settings": {"prompt": PROMPT, "resolution": resolution,
                         "video_length": frames, "seed": SEED},
            "output": {"mode": "b64"},
            "runtime": {"timeout_s": 2400},
        },
    }


def main() -> int:
    # Pin the production knobs BEFORE runpod_worker.config freezes at import.
    os.environ["WANGP_ATTENTION"] = os.environ.get("BENCH_ATTENTION", "sage2")
    os.environ["WANGP_PROFILE"] = os.environ.get("BENCH_PROFILE", "4")
    os.environ["WANGP_CLI_ARGS"] = (
        f"--attention {os.environ['WANGP_ATTENTION']} --profile {os.environ['WANGP_PROFILE']} --verbose 1")
    os.environ.setdefault("WANGP_EAGER_BOOT", "0")
    # a 15 s 480p clip can exceed the default 6 MB base64 response cap
    os.environ.setdefault("WANGP_B64_OUT_MAX", str(64 * 1024 * 1024))

    from runpod_worker import config as _cfg  # noqa: PLC0415

    live = _cfg.reload_config()
    assert list(live.cli_args) == os.environ["WANGP_CLI_ARGS"].split(), (
        f"config froze early: {list(live.cli_args)}")

    from models.minimax_h3 import streaming  # noqa: PLC0415
    from runpod_worker import handler as H  # noqa: PLC0415

    H.bootstrap()

    # Dedicated cold run (model load + first-touch) so every table leg is warm.
    warm_started = time.monotonic()
    warm = H.run_job(_job("832x480", 124, "warmup"))
    print("STREAM_WARMUP " + json.dumps({
        "wall_s": round(time.monotonic() - warm_started, 2),
        "status": warm.get("status"),
        "error": warm.get("error_message")}), flush=True)
    del warm

    out_root = os.environ.get("STREAM_BENCH_OUT", "/tmp/stream_bench")
    legs = []
    for resolution, frames, tag in _legs():
        for mode in ("baseline", "streaming"):
            rec = None
            if mode == "streaming":
                rec = streaming.activate(store_frames=False,
                                         live_dir=f"{out_root}/{tag}", audio_first=True)
                rec.reset()
            started = time.monotonic()
            response = H.run_job(_job(resolution, frames, f"{tag}-{mode}"))
            wall_s = round(time.monotonic() - started, 2)
            metrics = dict(response.get("metrics") or {})
            leg = {"tag": tag, "mode": mode, "resolution": resolution, "frames": frames,
                   "wall_s": wall_s,
                   "status": response.get("status"), "error": response.get("error_message"),
                   "generate_s": metrics.get("generate_s"),
                   "phase_marks_s": metrics.get("phase_marks_s")}
            del response
            if rec is not None:
                try:
                    chunk_rel = rec.chunk_ready_rel(started)
                    audio_s = rec.audio_s
                    segments = rec.live.close() if rec.live is not None else []
                    # Live timings: t_done is the wall time the finished
                    # segment file (audio included, when audio-first held)
                    # landed on disk, with mux overlapping the next chunk's
                    # decode. In the fallback ordering segments are silent, so
                    # model the audio-first shift additively instead.
                    shift = 0.0 if rec.audio_first_effective else audio_s
                    ready = [s["t_done"] - started + shift for s in segments if s.get("t_done")]
                    durations = [s["duration_s"] for s in segments if s.get("t_done")]
                    leg.update({
                        "audio_first_effective": rec.audio_first_effective,
                        "audio_decode_s": round(audio_s, 3),
                        "chunk_ready_rel_s": [round(c, 3) for c in chunk_rel],
                        "segments": [{k: v for k, v in s.items() if k != "path"} for s in segments],
                        "segment_ready_s": [round(r, 3) for r in ready],
                    })
                    if ready and len(ready) == len(durations):
                        oracle = streaming.no_buffer_start(ready, durations)
                        est = streaming.linear_estimate_start(ready, durations, observe=2, pad=1.15)
                        leg.update({
                            "ttff_oracle_s": round(oracle, 3),
                            "ttff_est_s": round(est["start_s"], 3),
                            "est_rate_s_per_chunk": round(est["rate_est_s"], 3),
                            "would_rebuffer": est["would_rebuffer"],
                            "worst_margin_s": round(est["worst_margin_s"], 3),
                        })
                except Exception as exc:  # noqa: BLE001 - keep remaining legs alive
                    import traceback

                    traceback.print_exc()
                    leg["stream_error"] = repr(exc)[:300]
                    leg.setdefault("chunk_ready_rel_s",
                                   [round(c, 3) for c in rec.chunk_ready_rel(started)])
                    leg.setdefault("audio_decode_s", round(rec.audio_s, 3))
                finally:
                    streaming.deactivate()
            legs.append(leg)
            print("STREAM_LEG " + json.dumps(leg), flush=True)

    # final table: per leg, baseline e2e vs streaming e2e vs perceived
    table = []
    by = {}
    for leg in legs:
        by.setdefault(leg["tag"], {})[leg["mode"]] = leg
    for resolution, frames, tag in _legs():
        base, stream = by[tag].get("baseline") or {}, by[tag].get("streaming") or {}
        row = {"clip": tag, "resolution": resolution, "frames": frames,
               "e2e_baseline_s": base.get("wall_s"),
               "e2e_streaming_s": stream.get("wall_s"),
               "ttff_est_s": stream.get("ttff_est_s"),
               "ttff_oracle_s": stream.get("ttff_oracle_s"),
               "would_rebuffer": stream.get("would_rebuffer"),
               "audio_first": stream.get("audio_first_effective"),
               "audio_decode_s": stream.get("audio_decode_s")}
        if row["e2e_baseline_s"] and row["ttff_est_s"]:
            row["perceived_speedup"] = round(row["e2e_baseline_s"] / row["ttff_est_s"], 2)
        table.append(row)
    summary = {"event": "stream_bench_summary", "seed": SEED,
               "attention": os.environ["WANGP_ATTENTION"], "profile": os.environ["WANGP_PROFILE"],
               "table": table, "legs": legs}
    print("STREAM_TABLE " + json.dumps(table), flush=True)
    print("STREAM_SUMMARY " + json.dumps(summary), flush=True)
    _ship(summary)
    failed = any(leg["status"] != "completed" for leg in legs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
