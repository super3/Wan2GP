#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Storyboard-grid -> N parallel clips -> one film.

The idea this implements: an image model makes ONE image containing a 2x2 grid
of keyframes, which are therefore consistent in palette, lighting and lens
because they were denoised together. Each cell becomes the start (and optionally
end) frame of an INDEPENDENT video job, so the clips have no data dependency on
each other and run fully in parallel.

That is the whole point. Chaining clip N+1 off clip N's last frame is serial and
caps out around 0.34x realtime no matter how many GPUs you own; keyframes known
up front make the work embarrassingly parallel, so wall-clock is one clip rather
than N.

    # 1. crop a grid into cells
    python3 grid_to_film.py crop board.png --rows 2 --cols 2 --out cells/

    # 2. fire every shot at the endpoint at once, poll, download
    python3 grid_to_film.py run shots.json --endpoint <id> --out clips/

    # 3. concatenate in order
    python3 grid_to_film.py mux clips/ --out film.mp4

``shots.json`` is the director's output: a list of
``{"prompt": ..., "start": "cells/cell_0.png", "end": null, "frames": 362}``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = "https://api.runpod.ai/v2"
#: 17n + 5. 362 frames = 15.08 s at 24 fps, the model's default sliding window.
LEGAL_FRAMES = [5 + 17 * n for n in range(1, 29) if 5 + 17 * n >= 107]


def _post(url: str, payload: dict, key: str) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------

def cmd_crop(args) -> int:
    from PIL import Image  # noqa: PLC0415

    board = Image.open(args.image).convert("RGB")
    w, h = board.size
    cw, ch = w // args.cols, h // args.rows
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"board {w}x{h} -> {args.rows}x{args.cols} cells of {cw}x{ch}")
    if (cw, ch) != (832, 480):
        print(f"  note: cells are {cw}x{ch}, the model renders 832x480; "
              f"generate the board at {832*args.cols}x{480*args.rows} to avoid a resample")
    n = 0
    for r in range(args.rows):
        for c in range(args.cols):
            cell = board.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            # A few px of grid line or JPEG ringing at the seam becomes a hard
            # edge in frame 1 of the clip, so trim before any resize.
            if args.inset:
                cell = cell.crop((args.inset, args.inset, cw - args.inset, ch - args.inset))
            if cell.size != (832, 480):
                cell = cell.resize((832, 480), Image.LANCZOS)
            path = out / f"cell_{n}.png"
            cell.save(path)
            print(f"  {path}")
            n += 1
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def cmd_run(args) -> int:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        print("set RUNPOD_API_KEY", file=sys.stderr)
        return 2
    shots = json.loads(Path(args.shots).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, shot in enumerate(shots):
        frames = int(shot.get("frames", 362))
        if frames not in LEGAL_FRAMES:
            print(f"shot {i}: {frames} is off the 17n+5 lattice", file=sys.stderr)
            return 2
        settings = {
            "prompt": shot["prompt"],
            "resolution": "832x480",
            "video_length": frames,
            "sample_solver": "euler",
            "video_prompt_type": "",
            "audio_prompt_type": "",
        }
        media, letters = {}, ""
        if shot.get("start"):
            media["image_start"] = {"b64": _b64(shot["start"])}
            letters += "S"
        if shot.get("end"):
            media["image_end"] = {"b64": _b64(shot["end"])}
            letters += "E"
        # The letters and the attachments must agree: wgp.py:1409/1425 read
        # image_start/image_end only when S/E are present in image_prompt_type.
        settings["image_prompt_type"] = letters
        if shot.get("seed") is not None:
            settings["seed"] = int(shot["seed"])
        payload = {"input": {
            "model_type": "minimax_h3_fl2va_pruned",
            "profile": "Turbo Lightx2v FL2V 4 Steps v1.1 768p",
            "settings": settings, "media": media,
            "output": {"mode": "auto"},
            "runtime": {"timeout_s": 1800},
        }}
        body = _post(f"{API}/{args.endpoint}/run", payload, key)
        jobs.append(body.get("id"))
        print(f"shot {i}: {frames}f {letters or 'text-only':6s} -> {body.get('id')}")

    # All shots are already queued; this only waits. Wall-clock is one clip if
    # the endpoint has enough workers, N clips if it has one.
    pending = {i: j for i, j in enumerate(jobs) if j}
    while pending:
        time.sleep(10)
        for i, jid in list(pending.items()):
            st = _get(f"{API}/{args.endpoint}/status/{jid}", key)
            status = st.get("status")
            if status == "COMPLETED":
                v = (st.get("output") or {}).get("video") or {}
                if v.get("kind") == "base64":
                    p = out / f"shot_{i:02d}.mp4"
                    p.write_bytes(base64.b64decode(v["data"]))
                    print(f"shot {i}: saved {p} ({p.stat().st_size/1048576:.2f} MB)")
                del pending[i]
            elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                o = st.get("output") or {}
                print(f"shot {i}: {status} - {o.get('message')}", file=sys.stderr)
                del pending[i]
    return 0


# ---------------------------------------------------------------------------
# mux
# ---------------------------------------------------------------------------

def cmd_mux(args) -> int:
    import imageio_ffmpeg  # noqa: PLC0415

    clips = sorted(Path(args.clips).glob("shot_*.mp4"))
    if not clips:
        print("no shot_*.mp4 found", file=sys.stderr)
        return 2
    listing = Path(args.clips) / "_concat.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    # Every clip is the same codec/resolution out of one endpoint, so the
    # concat demuxer can stream-copy: no re-encode, no generation loss.
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
           "-i", str(listing), "-c", "copy", args.out]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        return 1
    print(f"wrote {args.out} from {len(clips)} clips")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crop"); c.set_defaults(fn=cmd_crop)
    c.add_argument("image"); c.add_argument("--rows", type=int, default=2)
    c.add_argument("--cols", type=int, default=2); c.add_argument("--out", default="cells")
    c.add_argument("--inset", type=int, default=0, help="px to trim off each cell edge")

    r = sub.add_parser("run"); r.set_defaults(fn=cmd_run)
    r.add_argument("shots"); r.add_argument("--endpoint", required=True)
    r.add_argument("--out", default="clips")

    m = sub.add_parser("mux"); m.set_defaults(fn=cmd_mux)
    m.add_argument("clips"); m.add_argument("--out", default="film.mp4")

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
