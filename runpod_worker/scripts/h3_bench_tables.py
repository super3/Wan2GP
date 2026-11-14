#!/usr/bin/env python3
"""Markdown tables from the RESULT lines of h3_bench_models.py or h3_bench_usp.py (a whole pod log is fine: other lines are skipped).

    python runpod_worker/scripts/h3_bench_tables.py status.log [--title "2x RTX PRO 6000, USP"] [--usp]

Table A: cold start (first run of each combination, 480p 5 s). Table B: warm runs, full render / first frame (the earliest
stall-free playback start with streaming), one table per mode (one GPU; USP + multi-GPU decode; USP alone). When the runs carry
phase timings (h3_bench_usp.py), one phase table per mode: prompt / denoise / audio decode / video decode / save + mux, seconds.
Cells not run yet stay blank; VDN (#4) reads "skipped" in USP logs (USP refuses its sparse attention).
"""

from __future__ import annotations

import argparse
import json
import re

ROWS = [("1", "FL2VA (33B)", "Larry v4 ckpt600", 6), ("1b", "FL2VA Pruned (20B)", "Larry v4 ckpt600", 6),
        ("2", "FL2VA (33B)", "Lightx2v 8 Steps v1.0", 8), ("2b", "FL2VA Pruned (20B)", "Lightx2v 8 Steps v1.0", 8),
        ("3", "FL2VA Pruned (20B)", "Lightx2v 4 Steps v0.1", 4), ("4", "FL2VA VDN 8-Step (33B)", "None (built in)", 8),
        ("5", "HyperFlow Flash FL2VA (14B)", "None (built in)", 6), ("6", "FL2VA (33B)", "None", 28)]
CELLS = [("832x480", 124), ("832x480", 243), ("1280x704", 124), ("1280x704", 243)]
HEADER = ("| # | Model | Profile | Steps | 480p, 5 s | 480p, 10 s | 720p, 5 s | 720p, 10 s |\n"
          "|---|---|---|---|---|---|---|---|")
MODE_TITLES = {None: "warm runs", "mgd": "warm runs, USP + multi-GPU decode", "usp": "warm runs, USP alone"}
PHASES = ("prompt_s", "denoise_s", "audio_decode_s", "video_decode_s", "save_mux_s")


def parse(path: str):
    cold, split, warm, notes = {}, {}, {}, []
    with open(path, errors="replace") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        line = line.strip()
        match = re.match(r"RESULT (\w+) (\{.*\})$", line)
        if not match:
            if line.startswith("RESULT DONE") or "BENCH_FAILED" in line or "RESULT FAILED" in line:
                notes.append(line)
            continue
        tag = match.group(1)
        try:
            data = json.loads(match.group(2))
        except json.JSONDecodeError:
            notes.append(f"unreadable RESULT {tag} line (truncated?)")
            continue
        if tag == "cold":
            cold[data["combo"]] = data
        elif tag == "cold_split":
            split[data["combo"]] = data
        elif tag in ("warm", "warm_usp"):
            mode = "usp" if tag == "warm_usp" else data.get("mode")
            warm.setdefault(mode, {})[(data["combo"], data["resolution"], data["frames"])] = data
        elif tag in ("worker_failed", "cold_failed_retrying", "usp_broken_stopping", "usp_not_started"):
            notes.append(f"{tag}: {json.dumps(data)[:300]}")
    return cold, split, warm, notes


def seconds(value) -> str:
    if value is None:
        return ""
    if value < 1:  # load + warm-up when the previous combination already loaded the base model
        return "~0 s"
    return f"{value:.0f} s" if value >= 100 else f"{value:.1f} s"


def row(cid, model, profile, steps, cells) -> str:
    return f"| {cid} | {model} | {profile} | {steps} | " + " | ".join(cells) + " |"


def table_a(cold, split, usp_log: bool) -> None:
    print("**Table A: cold start (first run of each combination, 480p, 5 s)**\n")
    print("| # | Model | Profile | Steps | Downloaded | Download time | Load + warm-up | First run total |")
    print("|---|---|---|---|---|---|---|---|")
    for cid, model, profile, steps in ROWS:
        c, sp = cold.get(cid), split.get(cid)
        if cid == "4" and usp_log:
            print(row(cid, model, profile, steps, ["skipped"] * 4))
        elif c is None:
            print(row(cid, model, profile, steps, [""] * 4))
        else:
            measured = "download_gb" in c and c["download_gb"] is not None
            failed = "" if c.get("ok") else " ⚠️ failed"
            print(row(cid, model, profile, steps, [
                f"{c['download_gb']:.1f} GB" if measured else "not measured",
                seconds(c.get("download_s")) if measured else "not measured",
                seconds(sp.get("load_and_warmup_s")) if sp else "not measured",
                seconds(c.get("wall_s")) + failed]))


def table_b(title, runs, usp_log: bool, cell) -> None:
    print(f"\n**{title}**\n")
    print(HEADER)
    for cid, model, profile, steps in ROWS:
        cells = []
        for resolution, frames in CELLS:
            run = runs.get((cid, resolution, frames))
            if cid == "4" and usp_log:
                cells.append("skipped")
            elif run is None:
                cells.append("")
            elif not run.get("ok"):
                cells.append("⚠️ failed")
            else:
                cells.append(cell(run))
        print(row(cid, model, profile, steps, cells))


def full_and_first(run) -> str:
    return f"{seconds(run.get('wall_s'))} / {seconds(run.get('ttff_s'))}"


def phase_cell(run) -> str:
    phases = run.get("phases") or {}
    if "denoise_s" not in phases:
        return "not measured"
    return " / ".join("–" if phases.get(key) is None else f"{phases[key]:.1f}" for key in PHASES)


def notes_for(warm, notes) -> list[str]:
    out = []
    for mode, runs in warm.items():
        silent = [k for k, run in runs.items() if run.get("ok") and run.get("segments") and run.get("segments_with_audio") != run.get("segments")]
        if silent:
            out.append(f"Segments missing audio ({MODE_TITLES.get(mode, mode)}): {silent}")
    mgd = warm.get("mgd", {})
    if mgd:
        split_runs = [k for k, run in mgd.items() if isinstance(run.get("decode_plan"), dict)
                      and len(run["decode_plan"].get("devices") or []) > 1 and not run["decode_plan"].get("fallback")]
        out.append(f"Decode split across the GPUs in {len(split_runs)}/{len(mgd)} multi-GPU decode runs.")
    compared = [(k, run) for k, run in warm.get("usp", {}).items() if "same_frames_as_other_mode" in run]
    if compared:
        same_v = sum(1 for _, run in compared if run["same_frames_as_other_mode"])
        same_a = sum(1 for _, run in compared if run.get("same_audio_as_other_mode"))
        out.append(f"USP alone vs multi-GPU decode: identical video {same_v}/{len(compared)}, identical audio {same_a}/{len(compared)}.")
        for key, run in compared:
            chunks = run.get("chunk_compare") or {}
            differ = [d for d in chunks.get("differ", []) if len(d) == 4]
            if differ:
                odd = [d[3] for d in differ if d[0] % 2]
                even = [d[3] for d in differ if not d[0] % 2]
                worst = max(d[1] for d in differ)
                out.append(f"  {key}: chunks equal {chunks.get('equal')}; share of differing values on odd chunks (GPU 1) "
                           f"{min(odd, default=0):.3f}-{max(odd, default=0):.3f}, even chunks {min(even, default=0):.3f}-"
                           f"{max(even, default=0):.3f}; largest difference {worst}/255.")
    return out + [f"Note: {n}" for n in notes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--title", default=None)
    parser.add_argument("--usp", action="store_true", help="a USP log from before runs carried their mode: VDN reads skipped")
    args = parser.parse_args()
    cold, split, warm, notes = parse(args.log)
    usp_log = args.usp or any(mode is not None for mode in warm) or any("mode" in c for c in cold.values())
    if args.title:
        print(f"## {args.title}\n")
    table_a(cold, split, usp_log)
    modes = [mode for mode in (None, "mgd", "usp") if mode in warm] or [None]
    for mode in modes:
        table_b(f"Table B: {MODE_TITLES[mode]} (full render / first frame with streaming)", warm.get(mode, {}), usp_log, full_and_first)
    for mode in modes:
        runs = warm.get(mode, {})
        if any((run.get("phases") or {}).get("denoise_s") is not None for run in runs.values()):
            table_b(f"Table B: phases, {MODE_TITLES[mode].removeprefix('warm runs, ')} "
                    "(prompt / denoise / audio decode / video decode / save + mux, seconds)", runs, usp_log, phase_cell)
    total = sum(len(runs) for runs in warm.values())
    print(f"\nProgress: {len(cold)} cold, {total} warm ({', '.join(f'{MODE_TITLES[m]}: {len(warm[m])}' for m in modes if m in warm)}).")
    for line in notes_for(warm, notes):
        print(line)


if __name__ == "__main__":
    main()
