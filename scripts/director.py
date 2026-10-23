#!/usr/bin/env python3
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Director: render a planned multi-shot film with MiniMax H3 on several GPUs and stream it in order.

1. Storyboard: one short H3 generation holds one still shot per planned clip ("[Shot 2] At 00:01.000, cut to ...").
   A single generation keeps identity, wardrobe and light consistent across its shots. Boards are chained: each board
   after the first starts on the previous board's last keyframe.
2. Keyframes: the middle frame of every storyboard shot, taken losslessly from the decoded frames.
3. Clips: one H3 generation per planned shot, starting on its keyframe (optionally ending on the next one) with the
   same subject references. Clips run in parallel, one WanGP process per GPU.
4. Stream: finished clips are appended in order to an HLS playlist (stream/playlist.m3u8), then stitched into film.mp4.

Audio modes: "generated" (each clip makes its own sound, with short fades at the cuts), "kept" (a master soundtrack is
sliced per clip and kept: H3 animates to it, so the audio is continuous by construction), "bed" (each clip makes its own
sound and a continuous music bed is mixed over it).

    python scripts/director.py plan.json --out runs/film --dry-run         # jobs, prompts and a GPU-count estimate
    python scripts/director.py plan.json --out runs/film --gpus 0,1,2,3    # render
    python scripts/director.py plan.json --out runs/rehearsal --backend fake --gpus 0,0   # no GPU: ffmpeg test patterns

See docs/DIRECTOR.md for the plan format.
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

WANGP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "minimax_h3_fasth3_omni_nvfp4"
H3_FRAMES = {"fps": 24, "minimum": 107, "step": 17, "offset": 5}  # minimax_h3_handler: frames_minimum / steps / offset
WINDOW_MAX_FRAMES = 362  # one H3 sliding window (15 s): a storyboard split across windows would lose its shot layout
MAX_REFERENCE_IMAGES = 9  # Ref2VA reference-image budget (start and end images are not counted)
MAX_VOICES = 3  # audio_guide, audio_guide2, audio_guide3
CUT_GUARD_FRAMES = 4  # the video VAE packs 4 frames per latent, so frames this close to a cut can mix both shots
MIN_STORYBOARD_SHOT_S = 0.5
AUDIO_MODES = ("generated", "kept", "bed")
END_IMAGE_MODES = ("none", "next")
# Warm generation time on one RTX 5090, FastH3 Omni NVFP4 at 832x480, clip seconds -> seconds (used by --dry-run)
MEASURED_GEN_S = ((5.17, 16.5), (10.13, 31.3), (15.08, 62.6))
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm")
FAKE_FPS = 24


class PlanError(ValueError):
    pass


class RunError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------- timing
def fit_frames(requested: float, rule: dict, mode: str = "nearest") -> int:
    """The H3 frame count (offset + step * n, at least minimum) nearest to, at least or at most `requested`."""
    minimum, step, offset = rule["minimum"], rule["step"], rule["offset"]
    requested = max(float(minimum), float(requested))
    lower = max(minimum, offset + step * math.floor((requested - offset) / step))
    upper = offset + step * math.ceil((requested - offset) / step)
    if mode == "ceil":
        return upper
    if mode == "floor":
        return lower
    return lower if requested - lower <= upper - requested else upper


def shot_time(seconds: float) -> str:
    """H3's cut timestamp, MM:SS.mmm."""
    millis = int(round(seconds * 1000))
    return f"{millis // 60000:02d}:{millis % 60000 / 1000:06.3f}"


def estimate_gen_s(clip_seconds: float, scale: float = 1.0) -> float:
    """Generation time for a clip, interpolated from MEASURED_GEN_S (extrapolated along the end segments)."""
    points = MEASURED_GEN_S
    if clip_seconds <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif clip_seconds >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
    else:
        (x0, y0), (x1, y1) = next((a, b) for a, b in zip(points, points[1:], strict=False) if a[0] <= clip_seconds <= b[0])
    seconds = y0 + (clip_seconds - x0) * (y1 - y0) / (x1 - x0)
    return max(1.0, seconds) * scale


# ------------------------------------------------------------------------------------------------------------ plan
def _resolve(base: Path, value) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if not text or text[-1] in ".!?" else text + "."


def _lower_first(text: str) -> str:
    """'Wide shot of' -> 'wide shot of', for text that follows 'cut to'; acronyms and labels stay as they are."""
    return text[0].lower() + text[1:] if len(text) > 1 and text[0].isupper() and text[1].islower() else text


def _upper_first(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def load_plan(path) -> dict:
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc
    return normalize_plan(raw, path.parent)


def normalize_plan(raw: dict, base: Path) -> dict:
    """Fill defaults, resolve paths against the plan's folder and validate. Missing files are listed, not fatal, so a
    plan can be dry-run before its assets exist."""
    if not isinstance(raw, dict):
        raise PlanError("the plan must be a JSON object")
    try:
        return _normalize_plan(raw, Path(base))
    except (TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, PlanError):
            raise
        raise PlanError(f"invalid plan value: {exc}") from exc


def _normalize_plan(raw: dict, base: Path) -> dict:
    errors, missing = [], []

    def need(condition, message):
        if not condition:
            errors.append(message)

    def file_field(value, label):
        path = _resolve(base, value)
        if path is not None and not Path(path).is_file():
            missing.append(f"{label}: {path}")
        return path

    rule = {**H3_FRAMES, **(raw.get("frames_rule") or {})}
    plan = {
        "title": str(raw.get("title") or "film"),
        "model_type": str(raw.get("model_type") or DEFAULT_MODEL),
        "resolution": str(raw.get("resolution") or "832x480"),
        "clip_seconds": float(raw.get("clip_seconds") or 10),
        "seed": int(raw.get("seed", 42)),
        "style": " ".join(str(raw.get("style") or "").split()),
        "frames_rule": rule,
        "end_images": str(raw.get("end_images") or "none"),
        "end_image_strength": float(raw.get("end_image_strength", 0.5)),
        "settings": dict(raw.get("settings") or {}),
        "wgp_config": dict(raw.get("wgp_config") or {}),
        "cli_args": [str(arg) for arg in (raw.get("cli_args") or [])],
    }
    width, _, height = plan["resolution"].partition("x")
    need(width.isdigit() and height.isdigit(), f"resolution must be WIDTHxHEIGHT, got {plan['resolution']!r}")
    need(plan["end_images"] in END_IMAGE_MODES, f"end_images must be one of {END_IMAGE_MODES}")

    subjects = []
    for number, item in enumerate(raw.get("subjects") or [], 1):
        item = {"image": item} if isinstance(item, str) else dict(item)
        need(item.get("image"), f"subject {number}: needs an image")
        subjects.append({
            "image": file_field(item.get("image"), f"subject {number} image"),
            "description": " ".join(str(item.get("description") or "the subject from {picture}").split()).rstrip("."),
            "retain": " ".join(str(item.get("retain") or "identity, appearance and wardrobe are retained").split()).rstrip("."),
        })
    need(len(subjects) <= MAX_REFERENCE_IMAGES, f"at most {MAX_REFERENCE_IMAGES} subjects (reference images)")
    plan["subjects"] = subjects

    board = raw.get("storyboard") or {}
    plan["storyboard"] = {
        "shot_seconds": float(board.get("shot_seconds", 1.0)),
        "shots_per_board": int(board.get("shots_per_board", 5)),
        "link_boards": bool(board.get("link_boards", True)),
        "seed": int(board.get("seed", plan["seed"])),
    }
    shot_s, per_board = plan["storyboard"]["shot_seconds"], plan["storyboard"]["shots_per_board"]
    need(shot_s >= MIN_STORYBOARD_SHOT_S, f"storyboard.shot_seconds must be at least {MIN_STORYBOARD_SHOT_S}")
    need(per_board >= 1, "storyboard.shots_per_board must be at least 1")
    slots = per_board + (1 if plan["storyboard"]["link_boards"] else 0)
    if shot_s >= MIN_STORYBOARD_SHOT_S and per_board >= 1:
        need(fit_frames(slots * shot_s * rule["fps"], rule, "ceil") <= WINDOW_MAX_FRAMES,
             f"a storyboard of {slots} shots of {shot_s} s exceeds one {WINDOW_MAX_FRAMES}-frame window: "
             "lower storyboard.shots_per_board or storyboard.shot_seconds")

    audio = raw.get("audio") or {}
    mode = str(audio.get("mode") or "generated")
    need(mode in AUDIO_MODES, f"audio.mode must be one of {AUDIO_MODES}")
    voices = []
    for number, item in enumerate(audio.get("voices") or [], 1):
        sid = int(item.get("subject", 0))
        need(1 <= sid <= len(subjects), f"audio.voices[{number}]: subject {sid} does not exist")
        need(item.get("audio"), f"audio.voices[{number}]: needs an audio file")
        voices.append({"subject": sid, "audio": file_field(item.get("audio"), f"voice {number}")})
    plan["audio"] = {
        "mode": mode,
        "soundtrack": file_field(audio.get("soundtrack"), "audio.soundtrack"),
        "bed": file_field(audio.get("bed"), "audio.bed"),
        "bed_gain_db": float(audio.get("bed_gain_db", -12)),
        "cut_fade_ms": float(audio.get("cut_fade_ms", 0 if mode == "kept" else 40)),
        "voices": voices,
    }
    need(mode != "kept" or plan["audio"]["soundtrack"], 'audio.mode "kept" needs audio.soundtrack')
    need(mode != "bed" or plan["audio"]["bed"], 'audio.mode "bed" needs audio.bed')

    shots = []
    all_subjects = list(range(1, len(subjects) + 1))
    for index, item in enumerate(raw.get("shots") or []):
        item = {"action": item} if isinstance(item, str) else dict(item)
        label = f"shot {index + 1}"
        ids = [int(sid) for sid in item.get("subjects", all_subjects)]
        need(all(1 <= sid <= len(subjects) for sid in ids), f"{label}: unknown subject in {ids}")
        dialogue = []
        for line in item.get("dialogue") or []:
            sid = int(line.get("subject", 1))
            need(1 <= sid <= len(subjects), f"{label}: dialogue subject {sid} does not exist")
            need(str(line.get("text") or "").strip(), f"{label}: dialogue line without text")
            dialogue.append({"subject": sid, "speaker": str(line.get("speaker") or f"S{sid}"),
                             "language": str(line.get("language") or "English"), "text": " ".join(str(line.get("text") or "").split())})
        seconds = float(item.get("seconds") or plan["clip_seconds"])
        frames = fit_frames(seconds * rule["fps"], rule)
        need(frames <= WINDOW_MAX_FRAMES, f"{label}: {seconds} s is longer than one {WINDOW_MAX_FRAMES}-frame window")
        shot = {
            "index": index,
            "keyframe": " ".join(str(item.get("keyframe") or "").split()),
            "keyframe_image": file_field(item.get("keyframe_image"), f"{label} keyframe_image"),
            "action": " ".join(str(item.get("action") or "").split()),
            "summary": " ".join(str(item.get("summary") or "").split()),
            "soundscape": " ".join(str(item.get("soundscape") or "").split()),
            "music": " ".join(str(item.get("music") or "N/A").split()),
            "dialogue": dialogue,
            "subjects": ids,
            "frames": frames,
            "duration": frames / rule["fps"],
        }
        need(shot["keyframe"] or shot["keyframe_image"], f"{label}: needs a keyframe description or a keyframe_image")
        text = " ".join([shot["keyframe"], shot["action"], shot["summary"]])
        named = sorted({int(sid) for sid in re.findall(r"<Subject (\d+)>", text)} - set(ids))
        need(not named, f"{label}: mentions <Subject {named[0] if named else ''}> but does not list it in \"subjects\"")
        need("<Picture" not in text, f"{label}: do not write <Picture N> labels, the director numbers pictures per generation")
        need(shot["action"], f"{label}: needs an action")
        shots.append(shot)
    need(shots, "the plan has no shots")
    plan["shots"] = shots
    if errors:
        raise PlanError("invalid plan:\n  " + "\n  ".join(errors))
    plan["missing_files"] = missing
    return plan


# ---------------------------------------------------------------------------------------------------------- layout
def keyframe_path(out: Path, index: int) -> Path:
    return out / "keyframes" / f"kf_{index:03d}.png"


def clip_path(out: Path, index: int) -> Path:
    return out / "clips" / f"clip_{index:03d}.mp4"


def board_path(out: Path, index: int) -> Path:
    return out / "storyboards" / f"board_{index:02d}.mp4"


def slice_path(out: Path, index: int, kind: str = "slice") -> Path:
    return out / "audio" / f"{kind}_{index:03d}.wav"


def shot_keyframe(plan: dict, out: Path, index: int) -> str:
    return plan["shots"][index]["keyframe_image"] or str(keyframe_path(out, index))


def plan_boards(plan: dict) -> list[dict]:
    """Group the shots without a supplied keyframe_image into storyboards; each board after the first holds the previous
    board's last keyframe as an extra leading shot (its start image) when link_boards is on."""
    board_def, fps, rule = plan["storyboard"], plan["frames_rule"]["fps"], plan["frames_rule"]
    shot_s, per_board = board_def["shot_seconds"], board_def["shots_per_board"]
    pending = [shot["index"] for shot in plan["shots"] if not shot["keyframe_image"]]
    boards = []
    for number, start in enumerate(range(0, len(pending), per_board)):
        shots = pending[start:start + per_board]
        anchor = boards[-1]["shots"][-1] if board_def["link_boards"] and boards else None
        lead = 1 if anchor is not None else 0
        frames = fit_frames((len(shots) + lead) * shot_s * fps, rule, "ceil")
        extract = []
        for slot, index in enumerate(shots, lead):
            frame = min(frames - 1, int(round((slot + 0.5) * shot_s * fps)))
            extract.append({"shot": index, "frame": frame})
        boards.append({"index": number, "shots": shots, "anchor_shot": anchor, "frames": frames, "extract": extract,
                       "cuts": [round((slot * shot_s) * fps) for slot in range(1, len(shots) + lead)]})
    return boards


# --------------------------------------------------------------------------------------------------------- prompts
def _subject_block(plan: dict, subject_ids: list[int], first_picture: int, appears: dict[int, str]):
    definitions, retention = [], []
    for offset, sid in enumerate(subject_ids):
        subject = plan["subjects"][sid - 1]
        picture = f"<Picture {first_picture + offset}>"
        text = subject["description"]
        text = text.replace("{picture}", picture) if "{picture}" in text else f"{text} from {picture}"
        definitions.append(f"<Subject {sid}> is {text}.")
        retention.append(f"<Subject {sid}> (appears in {appears[sid]}): fully_preserved - {subject['retain']}.")
    return definitions, retention


def _style_sentence(plan: dict, fallback: str) -> str:
    style = plan["style"]
    if not style:
        return fallback
    return _sentence(style if style.lower().startswith("the target video is") else f"The target video is {_lower_first(style)}")


def board_prompt(plan: dict, board: dict) -> str:
    """H3 Ref2VA prompt for one storyboard: one held shot per keyframe, joined by hard cuts at fixed times."""
    shot_s = plan["storyboard"]["shot_seconds"]
    shots = [plan["shots"][index] for index in board["shots"]]
    anchored = board["anchor_shot"] is not None
    lead = 1 if anchored else 0
    ids = sorted({sid for shot in shots for sid in shot["subjects"]})
    appears = {sid: ", ".join(f"[Shot {slot + 1}]" for slot, shot in enumerate(shots, lead) if sid in shot["subjects"]) for sid in ids}
    definitions, retention = _subject_block(plan, ids, 1 + lead, appears)
    lines = ["subject_definitions:"]
    if anchored:
        lines.append("<Picture 1> is the first frame of the target video at 0.00 seconds.")
    lines += definitions or ["N/A"]
    tags = (["keyframe completion"] if anchored else []) + (["reference generation"] if ids else [])
    tag = f"[{' + '.join(tags)}] " if tags else ""  # a text-only board has no reference relationship to declare
    lines += ["summary:", f"{tag}A storyboard of {len(shots)} separate moments from one story in one consistent world, "
              "joined by hard cuts. Each shot is a held, nearly still frame with a locked-off camera."]
    lines.append("retention_analysis:")
    if anchored:
        lines.append("<Picture 1> (from [Shot 1]): fully_preserved - the opening frame is held unchanged.")
    lines += retention or ["N/A"]
    description = [_style_sentence(plan, "The target video is a cinematic storyboard."),
                   "Every shot holds still for its whole duration with no camera movement and almost no motion, so each shot reads as one clean frame."]
    if anchored:
        description.append("[Shot 1] <Picture 1> holds still.")
    for slot, shot in enumerate(shots, lead):
        text = _sentence(shot["keyframe"])
        description.append(f"[Shot {slot + 1}] {text}" if slot == 0 else f"[Shot {slot + 1}] At {shot_time(slot * shot_s)}, cut to {_lower_first(text)}")
    lines += ["detailed_description:", " ".join(description),
              "overall_soundscape: Quiet, steady room tone.", "non_diegetic_music: N/A"]
    return "\n".join(lines)


def _dialogue(plan: dict, shot: dict) -> str:
    kept = plan["audio"]["mode"] == "kept"
    parts = []
    for line in shot["dialogue"]:
        who = f"<Subject {line['subject']}>"
        if kept:  # the kept soundtrack carries the words: H3 is only asked to move the lips (H3 Ref2VA prompt guide)
            parts.append(f"{who} is visibly speaking with natural lip movements synchronized.")
        else:
            parts.append(f"{who} says ({line['speaker']}) <d>[{line['language']}] {line['text']}</d>.")
    return " ".join(dict.fromkeys(parts))


def clip_prompt(plan: dict, shot: dict, has_end: bool, voices: list[dict]) -> str:
    """H3 Ref2VA prompt for one clip: <Picture 1> is its keyframe, <Picture 2> the next keyframe when it must end on it,
    then one reference image per subject."""
    ids = shot["subjects"]
    first_reference = 3 if has_end else 2
    definitions, retention = _subject_block(plan, ids, first_reference, {sid: "[Shot 1]" for sid in ids})
    last_shot = max(1, shot["action"].count("[Shot "))
    lines = ["subject_definitions:", "<Picture 1> is the first frame of the target video at 0.00 seconds."]
    if has_end:
        lines.append(f"<Picture 2> is the last frame of the target video at {shot['duration']:.2f} seconds.")
    lines += definitions
    lines += [f"<Audio {number}> is the voice of <Subject {voice['subject']}>." for number, voice in enumerate(voices, 1)]
    tags = ["keyframe completion"] + (["reference generation"] if ids else []) + (["audio reference"] if voices else [])
    summary = shot["summary"] or shot["action"].split(". ")[0].removeprefix("[Shot 1]").strip()
    lines += ["summary:", f"[{' + '.join(tags)}] {_upper_first(_sentence(summary))}", "retention_analysis:",
              "<Picture 1> (from [Shot 1]): fully_preserved - the opening composition, lighting and wardrobe."]
    if has_end:
        lines.append(f"<Picture 2> (from [Shot {last_shot}]): fully_preserved - the closing frame.")
    lines += retention
    lines += [f"<Audio {number}> (voice of <Subject {voice['subject']}>): reference - voice timbre and accent." for number, voice in enumerate(voices, 1)]
    action = shot["action"] if shot["action"].startswith("[Shot 1]") else f"[Shot 1] Starting exactly from <Picture 1>, {_lower_first(shot['action'])}"
    description = [_style_sentence(plan, ""), _sentence(action), _dialogue(plan, shot)]
    if has_end:
        description.append("The video ends exactly on <Picture 2>.")
    lines += ["detailed_description:", " ".join(part for part in description if part),
              f"overall_soundscape: {_sentence(shot['soundscape']) or 'Natural ambience that matches the scene.'}",
              f"non_diegetic_music: {shot['music'] or 'N/A'}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------ jobs
def _base_settings(plan: dict, prompt: str, frames: int, seed: int) -> dict:
    settings = dict(plan["settings"])  # passthrough first: the structural keys below always win
    settings.update({"model_type": plan["model_type"], "prompt": prompt, "resolution": plan["resolution"],
                     "video_length": int(frames), "seed": int(seed), "prompt_enhancer": "",
                     "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": ""})
    return settings


def timeline(plan: dict) -> list[dict]:
    """Start time and duration of every clip in the finished film."""
    start, entries = 0.0, []
    for shot in plan["shots"]:
        entries.append({"index": shot["index"], "start": start, "duration": shot["duration"]})
        start += shot["duration"]
    return entries


def build_jobs(plan: dict, out: Path, soundtrack_seconds: float | None = None) -> tuple[list[dict], list[dict]]:
    """Every generation as a job with its WanGP settings, the jobs it waits for and its dispatch priority.
    Paths are deterministic, so the whole graph is known before anything runs."""
    out = Path(out)
    audio_mode, boards, jobs = plan["audio"]["mode"], plan_boards(plan), []
    board_of = {}
    for board in boards:
        ids = sorted({sid for index in board["shots"] for sid in plan["shots"][index]["subjects"]})
        settings = _base_settings(plan, board_prompt(plan, board), board["frames"], plan["storyboard"]["seed"] + board["index"])
        if board["anchor_shot"] is not None:
            settings.update(image_prompt_type="S", image_start=shot_keyframe(plan, out, board["anchor_shot"]))
        if ids:
            settings.update(video_prompt_type="I", image_refs=[plan["subjects"][sid - 1]["image"] for sid in ids])
        job_id = f"board{board['index']}"
        jobs.append({"id": job_id, "kind": "board", "index": board["index"], "priority": [0, board["index"]],
                     "deps": [f"board{board['index'] - 1}"] if board["anchor_shot"] is not None else [],
                     "label": f"storyboard {board['index'] + 1}", "seconds": board["frames"] / plan["frames_rule"]["fps"],
                     "settings": settings, "output": str(board_path(out, board["index"])),
                     "extract": [{"shot": item["shot"], "frame": item["frame"], "path": str(keyframe_path(out, item["shot"]))} for item in board["extract"]]})
        board_of.update({index: job_id for index in board["shots"]})
    clips = timeline(plan)
    for shot in plan["shots"]:
        index = shot["index"]
        has_end = plan["end_images"] == "next" and index + 1 < len(plan["shots"])
        voices = [voice for voice in plan["audio"]["voices"] if voice["subject"] in shot["subjects"]][:MAX_VOICES] if audio_mode != "kept" else []
        settings = _base_settings(plan, clip_prompt(plan, shot, has_end, voices), shot["frames"], plan["seed"] + index)
        settings.update(image_prompt_type="SE" if has_end else "S", image_start=shot_keyframe(plan, out, index))
        deps = {board_of[index]} if index in board_of else set()
        if has_end:
            settings.update(image_end=shot_keyframe(plan, out, index + 1), input_video_strength=plan["end_image_strength"])
            if index + 1 in board_of:
                deps.add(board_of[index + 1])
        if shot["subjects"]:
            settings.update(video_prompt_type="I", image_refs=[plan["subjects"][sid - 1]["image"] for sid in shot["subjects"]])
        if audio_mode == "kept" and (soundtrack_seconds is None or clips[index]["start"] < soundtrack_seconds):
            settings.update(audio_prompt_type="AS", audio_guide=str(slice_path(out, index)))
        elif voices:
            settings["audio_prompt_type"] = ("A", "AB", "ABD")[len(voices) - 1]
            for key, voice in zip(("audio_guide", "audio_guide2", "audio_guide3"), voices, strict=False):
                settings[key] = voice["audio"]
        jobs.append({"id": f"clip{index}", "kind": "clip", "index": index, "priority": [1, index], "deps": sorted(deps),
                     "label": f"clip {index + 1}", "seconds": shot["duration"], "settings": settings,
                     "output": str(clip_path(out, index)), "extract": []})
    return jobs, boards


WARMUP_PROMPT = "\n".join([
    "subject_definitions:", "N/A", "summary:", "A short shot of a calm sea.", "retention_analysis:", "N/A", "detailed_description:",
    "The target video is a calm grey sea under an overcast sky, small waves rolling slowly toward the camera.",
    "overall_soundscape: Soft waves.", "non_diegetic_music: N/A"])


def warmup_job(plan: dict, out: Path, worker: int) -> dict:
    """The shortest text-only generation: loads the model on a GPU that would otherwise idle (while a storyboard
    renders), so its first clip does not pay the model load."""
    rule = plan["frames_rule"]
    return {"id": f"warmup{worker}", "kind": "warmup", "index": worker, "priority": [2, worker], "deps": [], "label": "warmup",
            "seconds": rule["minimum"] / rule["fps"], "settings": _base_settings(plan, WARMUP_PROMPT, rule["minimum"], plan["seed"]),
            "output": str(Path(out) / "workers" / f"worker{worker}" / "warmup.mp4"), "extract": []}


# ------------------------------------------------------------------------------------------------------- scheduling
class Scheduler:
    """Dependency-aware dispatch order shared by the live run and the --dry-run estimate: storyboards first (they gate
    everything), then clips in film order."""

    def __init__(self, jobs: list[dict]):
        self.jobs = {job["id"]: job for job in jobs}
        self.state = {job["id"]: "pending" for job in jobs}
        unknown = {dep for job in jobs for dep in job["deps"]} - set(self.jobs)
        if unknown:
            raise ValueError(f"jobs depend on unknown jobs: {sorted(unknown)}")

    def next_ready(self) -> dict | None:
        ready = [job for job_id, job in self.jobs.items()
                 if self.state[job_id] == "pending" and all(self.state[dep] == "done" for dep in job["deps"])]
        return min(ready, key=lambda job: job["priority"]) if ready else None

    def mark(self, job_id: str, state: str) -> None:
        self.state[job_id] = state

    def finished(self) -> bool:
        return all(state == "done" for state in self.state.values())

    def running(self) -> list[str]:
        return [job_id for job_id, state in self.state.items() if state == "running"]


def simulate(jobs: list[dict], workers: int, job_seconds, load_s: float = 0.0, serial_first: bool = True,
             warmup_s: float | None = None) -> dict[str, float]:
    """When every job would finish on `workers` GPUs, `job_seconds(job)` each, plus `load_s` on a GPU's first job.
    As in run(): `serial_first` holds the other GPUs until the first job finishes, and with `warmup_s` a cold GPU with
    nothing to run spends load_s + warmup_s loading the model."""
    scheduler, idle, warm, running, now, done_at = Scheduler(jobs), list(range(workers)), set(), [], 0.0, {}
    while True:
        while idle and not (serial_first and not done_at and running):
            job = scheduler.next_ready()
            if job is None:
                break
            idle.sort(key=lambda w: (w not in warm, w))  # a loaded model first, as run() does
            worker = idle.pop(0)
            scheduler.mark(job["id"], "running")
            cost = job_seconds(job) + (0.0 if worker in warm else load_s)
            warm.add(worker)
            heapq.heappush(running, (now + cost, job["id"], worker))
        if warmup_s is not None and not (serial_first and not done_at) and scheduler.next_ready() is None:
            for worker in [w for w in idle if w not in warm]:
                idle.remove(worker)
                warm.add(worker)
                heapq.heappush(running, (now + load_s + warmup_s, f"warmup{worker}", worker))
        if not running:
            return done_at
        now, job_id, worker = heapq.heappop(running)
        if not job_id.startswith("warmup"):
            scheduler.mark(job_id, "done")
            done_at[job_id] = now
        idle.append(worker)


def playback(ready_at: list[float], durations: list[float], prebuffer: int = 1) -> dict:
    """Play clips in order as soon as `prebuffer` of them are available, waiting (stalling) for any that are late."""
    available, latest = [], 0.0
    for ready in ready_at:  # a clip streams only once every earlier clip has
        latest = max(latest, ready)
        available.append(latest)
    if not available:
        return {"first_frame_s": None, "stall_s": 0.0, "stalls": 0, "end_s": None, "starts": []}
    clock = available[min(len(available), max(1, prebuffer)) - 1]
    first, stall, stalls, starts = clock, 0.0, 0, []
    for ready, duration in zip(available, durations, strict=True):
        if ready > clock + 1e-9:
            stall, stalls, clock = stall + ready - clock, stalls + 1, ready
        starts.append(clock)
        clock += duration
    return {"first_frame_s": first, "stall_s": stall, "stalls": stalls, "end_s": clock, "starts": starts}


def estimate(jobs: list[dict], gpu_counts, scale: float = 1.0, load_s: float = 0.0, prebuffer: int = 1,
             parallel_start: bool = False, warmup: bool = True, warmup_frames: int = H3_FRAMES["minimum"]) -> list[dict]:
    clips = [job for job in jobs if job["kind"] == "clip"]
    durations = [job["seconds"] for job in clips]
    warmup_s = estimate_gen_s(warmup_frames / H3_FRAMES["fps"], scale) if warmup else None
    rows = []
    for count in gpu_counts:
        done_at = simulate(jobs, count, lambda job: estimate_gen_s(job["seconds"], scale), load_s, not parallel_start, warmup_s)
        play = playback([done_at[job["id"]] for job in clips], durations, prebuffer)
        rows.append({"gpus": count, "first_frame_s": play["first_frame_s"], "stall_s": play["stall_s"],
                     "render_s": max(done_at.values()), "video_s": sum(durations)})
    return rows


# ------------------------------------------------------------------------------------------------------------ media
def find_ffmpeg() -> str:
    candidate = os.environ.get("DIRECTOR_FFMPEG") or shutil.which("ffmpeg")
    if candidate:
        return candidate
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - WanGP's requirements ship a static ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RunError("ffmpeg not found: install it or set DIRECTOR_FFMPEG") from exc


def run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *[str(arg) for arg in args]]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RunError(f"ffmpeg failed ({result.returncode}): {' '.join(command[4:])[:300]}\n{result.stderr[-1500:]}")


def has_audio(ffmpeg: str, path) -> bool:
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in probe.stderr


def extract_frame_file(ffmpeg: str, video, index: int, png) -> None:
    """Fallback when the decoded frames are not returned: one frame (0-based) decoded from the saved video."""
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(ffmpeg, ["-i", video, "-vf", f"select=eq(n\\,{int(index)})", "-frames:v", "1", png])
    if not Path(png).is_file():
        raise RunError(f"frame {index} not found in {video}")


def save_frame_tensor(frames, index: int, png) -> None:
    """Save frame `index` of a [C, F, H, W] uint8 tensor or array (WanGP's return_media layout) as a PNG."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    frame = frames[:, int(index)]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    array = np.ascontiguousarray(np.transpose(np.asarray(frame)[:3], (1, 2, 0)).astype(np.uint8))
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(png)


def prepare_wav(ffmpeg: str | None, source, target) -> float:
    """Copy (16-bit PCM WAV) or convert the audio to `target` and return its duration in seconds."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(source)) as handle:
            usable = handle.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        usable = False
    if usable:
        shutil.copyfile(source, target)
    else:
        if ffmpeg is None:
            raise RunError(f"{source} is not a 16-bit PCM WAV and ffmpeg is not available to convert it")
        run_ffmpeg(ffmpeg, ["-i", source, "-vn", "-c:a", "pcm_s16le", target])
    with wave.open(str(target)) as handle:
        return handle.getnframes() / handle.getframerate()


def write_slices(master, slices: list[tuple]) -> list[str]:
    """Cut a 16-bit PCM WAV into contiguous slices [(path, start_s, duration_s)], sample-exact. A slice that starts after
    the end of the audio is skipped; one that runs past it is shorter. Returns the paths written."""
    written = []
    with wave.open(str(master)) as source:
        rate, total = source.getframerate(), source.getnframes()
        for path, start, duration in slices:
            first, last = int(round(start * rate)), min(total, int(round((start + duration) * rate)))
            if first >= total or last <= first:
                continue
            source.setpos(first)
            data = source.readframes(last - first)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as target:
                target.setnchannels(source.getnchannels())
                target.setsampwidth(source.getsampwidth())
                target.setframerate(rate)
                target.writeframes(data)
            written.append(str(path))
    return written


class Streamer:
    """Appends finished clips to the stream strictly in film order: stream/seg_NNN.mp4 (fades or music bed applied),
    stream/seg_NNN.ts and an EVENT HLS playlist that any HLS player can open while the film is still rendering."""

    def __init__(self, out: Path, plan: dict, ffmpeg: str, beds: dict[int, str] | None = None, hls: bool = True, log=print):
        self.dir = Path(out) / "stream"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.out, self.plan, self.ffmpeg, self.beds, self.hls, self.log = Path(out), plan, ffmpeg, beds or {}, hls, log
        self.durations = {entry["index"]: entry["duration"] for entry in timeline(plan)}
        self.target = max(1, math.ceil(max(self.durations.values())))
        self.waiting, self.segments, self.next_index, self.ready_at = {}, [], 0, {}
        self.playlist = self.dir / "playlist.m3u8"
        if hls:
            self._write_playlist(final=False)

    def add(self, index: int, video, clock) -> list[int]:
        """Queue a finished clip; emit every clip now next in film order. `clock()` stamps when each one goes live."""
        self.waiting[index] = Path(video)
        emitted = []
        while self.next_index in self.waiting:
            self._emit(self.next_index, self.waiting.pop(self.next_index))
            self.ready_at[self.next_index] = clock()
            emitted.append(self.next_index)
            self.next_index += 1
        return emitted

    def _emit(self, index: int, video: Path) -> None:
        duration, audio = self.durations[index], self.plan["audio"]
        segment = self.dir / f"seg_{index:03d}.mp4"
        fade, bed = audio["cut_fade_ms"] / 1000.0, self.beds.get(index)
        if (fade > 0 or bed) and has_audio(self.ffmpeg, video):
            inputs, chain, source = ["-i", video], [], "[0:a]"
            if bed:
                inputs += ["-i", bed]
                chain.append(f"[1:a]volume={audio['bed_gain_db']}dB[bed];[0:a][bed]amix=inputs=2:duration=first:normalize=0[mix]")
                source = "[mix]"
            fades = f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}" if fade > 0 else "anull"
            chain.append(f"{source}{fades}[out]")
            run_ffmpeg(self.ffmpeg, [*inputs, "-filter_complex", ";".join(chain), "-map", "0:v", "-map", "[out]",
                                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", segment])
        else:
            shutil.copyfile(video, segment)
        self.segments.append((segment, duration))
        if self.hls:
            run_ffmpeg(self.ffmpeg, ["-i", segment, "-c", "copy", "-f", "mpegts", segment.with_suffix(".ts")])
            self._write_playlist(final=False)

    def _write_playlist(self, final: bool) -> None:
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{self.target}", "#EXT-X-MEDIA-SEQUENCE:0",
                 "#EXT-X-PLAYLIST-TYPE:EVENT"]
        for number, (segment, duration) in enumerate(self.segments):
            if number:
                lines.append("#EXT-X-DISCONTINUITY")  # every clip is its own encode: timestamps restart
            lines += [f"#EXTINF:{duration:.3f},", segment.with_suffix(".ts").name]
        if final:
            lines.append("#EXT-X-ENDLIST")
        temp = self.playlist.with_suffix(".tmp")
        temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temp, self.playlist)

    def finish(self, master_audio=None) -> Path:
        """Close the playlist and stitch film.mp4; in kept mode the master soundtrack replaces the clip audio, so the
        finished film has no seam at all."""
        if self.hls:
            self._write_playlist(final=True)
        listing = self.dir / "segments.txt"
        listing.write_text("".join(f"file '{segment.name}'\n" for segment, _ in self.segments), encoding="utf-8")
        film = self.out / "film.mp4"
        stitched = film if master_audio is None else self.out / "film_clip_audio.mp4"
        run_ffmpeg(self.ffmpeg, ["-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", stitched])
        if master_audio is not None:
            run_ffmpeg(self.ffmpeg, ["-i", stitched, "-i", master_audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                                     "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", film])
        return film


# ---------------------------------------------------------------------------------------------------------- workers
class WanGPBackend:
    """One WanGP session in this process (one GPU: CUDA_VISIBLE_DEVICES is set before torch loads)."""

    def __init__(self, spec: dict, worker: int):
        self.spec, self.worker = spec, worker
        self.root = Path(spec["wangp_root"]).resolve()
        self.work = Path(spec["out"]) / "workers" / f"worker{worker}"
        self.session = self.log = None

    def start(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        config = self.work / "wgp_config.json"  # one config per process: concurrent WanGP processes must not share one
        base = self.root / "wgp_config.json"
        if not config.is_file() and base.is_file():
            shutil.copyfile(base, config)
        if self.spec["wgp_config"]:
            if not config.is_file():
                raise RunError(f"wgp_config overrides need a full wgp_config.json: start WanGP once in {self.root} first")
            data = json.loads(config.read_text(encoding="utf-8"))
            data.update(self.spec["wgp_config"])
            config.write_text(json.dumps(data, indent=4), encoding="utf-8")
        os.chdir(self.root)  # wgp resolves settings/ and other folders against the working directory
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from shared.api import init  # noqa: PLC0415 - imports torch: only after CUDA_VISIBLE_DEVICES is set

        self.log = open(self.work / "wangp.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives with the worker
        self.session = init(root=self.root, config_path=config, output_dir=self.work / "outputs",
                            cli_args=self.spec["cli_args"], console_output=False)

    def run(self, job: dict, progress) -> dict:
        settings = dict(job["settings"])
        if job["extract"]:
            settings["_api"] = {"return_media": True}  # decoded frames in memory: keyframes skip the mp4 compression
        handle = self.session.submit_task(settings)
        tail, phase = collections.deque(maxlen=40), None
        for event in handle.events.iter(timeout=0.5):
            if event.kind == "progress" and event.data.phase != phase:
                phase = event.data.phase
                progress(phase)
            elif event.kind == "stream":
                text = getattr(event.data, "text", "")
                self.log.write(text if text.endswith("\n") else text + "\n")
                if text.strip():
                    tail.append(text.strip()[:300])
        result = handle.result()
        if not result.success:
            message = "; ".join(error.message for error in result.errors) or "generation failed"
            raise RunError(message + ("\n" + "\n".join(list(tail)[-15:]) if tail else ""))
        artifact = result.artifacts[0] if result.artifacts else None
        source = artifact.path if artifact is not None and artifact.path else next(
            (name for name in reversed(result.generated_files) if Path(name).suffix.lower() in VIDEO_SUFFIXES), None)
        if not source:
            raise RunError(f"no video among {result.generated_files}")
        target = Path(job["output"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        frames = artifact.video_tensor_uint8 if artifact is not None else None
        for item in job["extract"]:
            if frames is not None and item["frame"] < frames.shape[1]:
                save_frame_tensor(frames, item["frame"], item["path"])
            else:
                extract_frame_file(find_ffmpeg(), target, item["frame"], item["path"])
        return {"video": str(target), "lossless_keyframes": frames is not None}


class FakeBackend:
    """No GPU: renders an ffmpeg test pattern of the requested length (with the kept soundtrack slice when there is one)
    after checking that every input the job names exists. For rehearsing scheduling, keyframes and streaming."""

    def __init__(self, spec: dict, worker: int):
        self.spec, self.worker, self.ffmpeg = spec, worker, None

    def start(self) -> None:
        self.ffmpeg = find_ffmpeg()

    def run(self, job: dict, progress) -> dict:
        settings = job["settings"]
        image_flags, audio_flags = settings.get("image_prompt_type", ""), settings.get("audio_prompt_type", "")
        needed = [settings.get("image_start")] if "S" in image_flags else []
        needed += [settings.get("image_end")] if "E" in image_flags else []
        needed += list(settings.get("image_refs") or [])
        needed += [settings.get(key) for flag, key in (("A", "audio_guide"), ("B", "audio_guide2"), ("D", "audio_guide3")) if flag in audio_flags]
        absent = [str(path) for path in needed if not path or not Path(path).is_file()]
        if absent:
            raise RunError(f"missing inputs: {absent}")
        progress("inference")
        time.sleep(float(self.spec.get("fake_seconds") or 0.0))
        width, height = (int(value) for value in settings["resolution"].split("x"))
        frames = int(settings["video_length"])
        video = Path(job["output"])
        video.parent.mkdir(parents=True, exist_ok=True)
        inputs = ["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={FAKE_FPS}"]
        if "S" in audio_flags:
            inputs += ["-i", settings["audio_guide"]]
        else:
            inputs += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000"]
        run_ffmpeg(self.ffmpeg, [*inputs, "-frames:v", frames, "-t", f"{frames / FAKE_FPS:.4f}", "-map", "0:v", "-map", "1:a",
                                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", video])
        for item in job["extract"]:
            extract_frame_file(self.ffmpeg, video, item["frame"], item["path"])
        video.with_suffix(".json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
        return {"video": str(video), "lossless_keyframes": False}


def worker_main(worker: int, gpu: str, spec: dict, inbox, outbox) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before anything imports torch: WanGP then sees this GPU as cuda:0
    try:
        backend = (FakeBackend if spec["backend"] == "fake" else WanGPBackend)(spec, worker)
        backend.start()
    except BaseException as exc:  # noqa: BLE001 - reported to the director, which drops this worker
        outbox.put({"type": "dead", "worker": worker, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]})
        return
    outbox.put({"type": "ready", "worker": worker})
    while True:
        job = inbox.get()
        if job is None:
            return
        started = time.monotonic()

        def progress(phase, job_id=job["id"]):
            outbox.put({"type": "progress", "worker": worker, "job": job_id, "phase": str(phase)})

        try:
            info = backend.run(job, progress)
            outbox.put({"type": "done", "worker": worker, "job": job["id"], "seconds": time.monotonic() - started, "info": info})
        except Exception as exc:  # noqa: BLE001 - a failed job is retried elsewhere or ends the run
            outbox.put({"type": "failed", "worker": worker, "job": job["id"], "seconds": time.monotonic() - started,
                        "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]})


# ------------------------------------------------------------------------------------------------------------- run
def prepare_audio(plan: dict, out: Path, ffmpeg: str) -> tuple[float | None, str | None, dict[int, str]]:
    """Kept mode: the master soundtrack as WAV and one slice per clip. Bed mode: one bed slice per clip."""
    audio, clips = plan["audio"], timeline(plan)
    if audio["mode"] == "kept":
        master = out / "audio" / "soundtrack.wav"
        seconds = prepare_wav(ffmpeg, audio["soundtrack"], master)
        write_slices(master, [(slice_path(out, clip["index"]), clip["start"], clip["duration"]) for clip in clips])
        return seconds, str(master), {}
    if audio["mode"] == "bed":
        bed = out / "audio" / "bed.wav"
        prepare_wav(ffmpeg, audio["bed"], bed)
        written = write_slices(bed, [(slice_path(out, clip["index"], "bed"), clip["start"], clip["duration"]) for clip in clips])
        return None, None, {clip["index"]: str(slice_path(out, clip["index"], "bed")) for clip in clips
                            if str(slice_path(out, clip["index"], "bed")) in written}
    return None, None, {}


def run(plan: dict, out: Path, gpus: list[str], backend: str = "wangp", wangp_root: Path = WANGP_ROOT, retries: int = 1,
        hls: bool = True, prebuffer: int = 1, fake_seconds: float = 0.0, parallel_start: bool = False, warmup: bool = True,
        log=print) -> dict:
    out = Path(out).resolve()
    if plan["missing_files"] and backend != "fake":
        raise PlanError("missing files:\n  " + "\n  ".join(plan["missing_files"]))
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    soundtrack_seconds, master, beds = prepare_audio(plan, out, ffmpeg)
    jobs, boards = build_jobs(plan, out, soundtrack_seconds)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    clips = [job for job in jobs if job["kind"] == "clip"]
    skipped = [job["index"] + 1 for job in clips if plan["audio"]["mode"] == "kept" and "S" not in job["settings"]["audio_prompt_type"]]
    if skipped:
        log(f"note: the soundtrack ends before clip(s) {skipped}: they generate their own audio")
    log(f"{plan['title']}: {len(clips)} clips ({sum(job['seconds'] for job in clips):.1f} s), {len(boards)} storyboard(s), "
        f"{len(gpus)} GPU worker(s) {gpus}, audio {plan['audio']['mode']}")

    spec = {"backend": backend, "wangp_root": str(wangp_root), "out": str(out), "cli_args": plan["cli_args"],
            "wgp_config": plan["wgp_config"], "fake_seconds": fake_seconds}
    context = mp.get_context("spawn")
    outbox, workers = context.Queue(), {}
    for worker, gpu in enumerate(gpus):
        inbox = context.Queue()
        # not daemonic: WanGP may start child processes of its own; shutdown below stops every worker
        process = context.Process(target=worker_main, args=(worker, gpu, spec, inbox, outbox), name=f"director-{worker}")
        process.start()
        workers[worker] = {"process": process, "inbox": inbox, "gpu": gpu, "name": f"w{worker}/gpu{gpu}", "ready": False, "job": None}

    scheduler, attempts, records, streamer = Scheduler(jobs), collections.Counter(), {}, Streamer(out, plan, ffmpeg, beds, hls, log)
    started = time.monotonic()
    # The first WanGP job may download the weights: until one job has finished, run one at a time, since concurrent
    # downloads into the shared checkpoint folder would collide. Every clip waits for the first storyboard anyway.
    gated = not parallel_start

    known = dict(scheduler.jobs)  # graph jobs plus the warmups sent below

    def stamp() -> str:
        return f"[{time.monotonic() - started:7.1f}s]"

    def requeue(job_id: str, name: str, error: str) -> None:
        attempts[job_id] += 1
        log(f"{stamp()} {name} {scheduler.jobs[job_id]['label']} failed (attempt {attempts[job_id]}): {error}")
        if attempts[job_id] > retries:
            raise RunError(f"{scheduler.jobs[job_id]['label']} failed {attempts[job_id]} time(s): {error}")
        scheduler.mark(job_id, "pending")

    try:
        while not scheduler.finished():
            # warm workers (model already loaded) take the most urgent job first
            for info in sorted(workers.values(), key=lambda item: (not item.get("warm"), item["name"])):
                if gated and scheduler.running():
                    break
                if info["ready"] and info["job"] is None:
                    job = scheduler.next_ready()
                    if job is None:
                        break
                    scheduler.mark(job["id"], "running")
                    info["job"], info["started"] = job["id"], time.monotonic()
                    info["inbox"].put(job)
                    log(f"{stamp()} {info['name']} starts {job['label']} ({job['seconds']:.1f} s)")
            if warmup and not gated and scheduler.next_ready() is None:
                for worker, info in workers.items():
                    if info["ready"] and info["job"] is None and not info.get("warm"):
                        job = known.setdefault(f"warmup{worker}", warmup_job(plan, out, worker))
                        info["job"] = job["id"]
                        info["inbox"].put(job)
                        log(f"{stamp()} {info['name']} loads the model (warmup)")
            try:
                message = outbox.get(timeout=1.0)
            except queue.Empty:
                message = next(({"type": "dead", "worker": worker, "error": f"worker exited with code {info['process'].exitcode}"}
                                for worker, info in workers.items() if not info["process"].is_alive()), None)
            if message is not None:
                worker, kind = message["worker"], message["type"]
                info = workers.get(worker)
                if info is None:
                    continue
                if kind == "ready":
                    info["ready"] = True
                    log(f"{stamp()} {info['name']} worker ready")
                elif kind == "progress":
                    log(f"{stamp()} {info['name']} {known[message['job']]['label']}: {message['phase']}")
                elif kind in ("done", "failed") and known[message["job"]]["kind"] == "warmup":
                    info["job"], info["warm"] = None, True  # a failed warmup is not retried: the worker's first job will show it
                    end = time.monotonic() - started
                    records[message["job"]] = {"label": "warmup", "gpu": info["gpu"], "worker": worker, "start_s": round(end - message["seconds"], 2),
                                               "end_s": round(end, 2), "seconds": round(message["seconds"], 2), "status": kind}
                    log(f"{stamp()} {info['name']} warmup {kind} in {message['seconds']:.1f} s" + (f": {message['error']}" if kind == "failed" else ""))
                elif kind == "done":
                    job = scheduler.jobs[message["job"]]
                    scheduler.mark(job["id"], "done")
                    info["job"], info["warm"], gated = None, True, False
                    end = time.monotonic() - started
                    records[job["id"]] = {"label": job["label"], "gpu": info["gpu"], "worker": worker,
                                          "start_s": round(end - message["seconds"], 2), "end_s": round(end, 2),
                                          "seconds": round(message["seconds"], 2), **message.get("info", {})}
                    log(f"{stamp()} {info['name']} {job['label']} done in {message['seconds']:.1f} s")
                    if job["kind"] == "clip":
                        for index in streamer.add(job["index"], job["output"], lambda: time.monotonic() - started):
                            log(f"{stamp()} stream: clip {index + 1} is live ({streamer.durations[index]:.1f} s)")
                elif kind == "failed":
                    info["job"] = None
                    requeue(message["job"], info["name"], message["error"])
                elif kind == "dead":
                    workers.pop(worker)
                    log(f"{stamp()} {info['name']} worker lost: {message['error']}")
                    if message.get("trace"):
                        log(message["trace"])
                    if info["job"] is not None and info["job"] in scheduler.jobs:
                        requeue(info["job"], info["name"], message["error"])
                    if not workers:
                        raise RunError("no GPU worker left")
            if not scheduler.finished() and not scheduler.running() and scheduler.next_ready() is None:
                raise RunError("nothing left to run: a job's dependency never finished")
    finally:
        for info in workers.values():
            info["inbox"].put(None)
        for info in workers.values():
            info["process"].join(timeout=30)
            if info["process"].is_alive():
                info["process"].terminate()

    film_s = sum(job["seconds"] for job in clips)
    # the master soundtrack replaces the clip audio only when it covers the whole film; otherwise the clips past its end
    # keep the audio they generated
    film = streamer.finish(master if master is not None and soundtrack_seconds >= film_s - 0.05 else None)
    wall = time.monotonic() - started
    ready = [streamer.ready_at[job["index"]] for job in clips]
    durations = [job["seconds"] for job in clips]
    report = {"title": plan["title"], "gpus": gpus, "backend": backend, "audio_mode": plan["audio"]["mode"],
              "video_s": round(sum(durations), 3), "wall_s": round(wall, 2),
              "playback": {key: (round(value, 2) if isinstance(value, float) else value)
                           for key, value in playback(ready, durations, prebuffer).items() if key != "starts"},
              "clips": [{"index": job["index"], "duration_s": round(job["seconds"], 3),
                         "live_at_s": round(streamer.ready_at[job["index"]], 2)} for job in clips],
              "jobs": records, "film": str(film), "playlist": str(streamer.playlist) if hls else None}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    play = report["playback"]
    log(f"done in {wall:.1f} s: {report['video_s']:.1f} s of film, first clip live at {play['first_frame_s']} s, "
        f"{play['stalls']} stall(s) totalling {play['stall_s']} s with a {prebuffer}-clip buffer. Film: {film}")
    return report


def dry_run(plan: dict, out: Path, gpus: int, scale: float, load_s: float, prebuffer: int, parallel_start: bool = False,
            warmup: bool = True, log=print) -> list[dict]:
    jobs, boards = build_jobs(plan, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    fps = plan["frames_rule"]["fps"]
    log(f"{plan['title']}: {len(plan['shots'])} clips, {sum(shot['duration'] for shot in plan['shots']):.1f} s of film, "
        f"model {plan['model_type']} at {plan['resolution']}, audio {plan['audio']['mode']}, end images {plan['end_images']}")
    for board in boards:
        log(f"  storyboard {board['index'] + 1}: {board['frames']} frames ({board['frames'] / fps:.2f} s), shots "
            f"{[index + 1 for index in board['shots']]}, keyframes at frames {[item['frame'] for item in board['extract']]}"
            + (f", starts on clip {board['anchor_shot'] + 1}'s keyframe" if board["anchor_shot"] is not None else ""))
    for shot in plan["shots"]:
        log(f"  clip {shot['index'] + 1}: {shot['frames']} frames ({shot['duration']:.2f} s)")
    if plan["missing_files"]:
        log("missing files (needed before a real run):\n  " + "\n  ".join(plan["missing_files"]))
    for job in jobs[:1] + [job for job in jobs if job["kind"] == "clip"][:1]:
        log(f"\n--- {job['label']} prompt ---\n{job['settings']['prompt']}")
    counts = sorted({1, 2, 3, 4, 5, 6, 8, gpus})
    rows = estimate(jobs, counts, scale, load_s, prebuffer, parallel_start, warmup, plan["frames_rule"]["minimum"])
    log(f"\nestimate from warm RTX 5090 times x{scale} (cold load {load_s} s per GPU, warmups {'on' if warmup else 'off'}, "
        f"{'parallel start' if parallel_start else 'first job alone'}), playback starts after {prebuffer} clip(s):")
    log("  GPUs  first clip live  stalls  render done  (film length)")
    for row in rows:
        mark = "  <- realtime" if row["stall_s"] <= 0.0 else ""
        log(f"  {row['gpus']:4d}  {row['first_frame_s']:13.1f} s  {row['stall_s']:5.1f} s  {row['render_s']:9.1f} s  ({row['video_s']:.1f} s){mark}")
    log(f"\njobs written to {out / 'jobs.json'}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", help="film plan (JSON, see docs/DIRECTOR.md)")
    parser.add_argument("--out", required=True, help="run folder: storyboards/, keyframes/, clips/, stream/, film.mp4, report.json")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0",
                        help="comma-separated GPU ids, one WanGP worker process each (default: CUDA_VISIBLE_DEVICES or 0)")
    parser.add_argument("--wangp-root", default=str(WANGP_ROOT), help="WanGP checkout (default: this one)")
    parser.add_argument("--dry-run", action="store_true", help="print the jobs, prompts and a GPU-count estimate; run nothing")
    parser.add_argument("--backend", choices=("wangp", "fake"), default="wangp", help="fake: ffmpeg test patterns, no GPU")
    parser.add_argument("--fake-seconds", type=float, default=0.0, help="fake backend: seconds each job takes")
    parser.add_argument("--retries", type=int, default=1, help="retries per failed job before the run stops")
    parser.add_argument("--parallel-start", action="store_true",
                        help="do not hold the other GPUs until the first job finishes (only when the weights are already downloaded)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="do not load the model on idle GPUs with a short warmup generation (see --parallel-start)")
    parser.add_argument("--no-hls", action="store_true", help="skip the HLS playlist; still stitch film.mp4")
    parser.add_argument("--prebuffer", type=int, default=1, help="clips buffered before playback starts (report and estimate)")
    parser.add_argument("--est-scale", type=float, default=1.0, help="dry run: scale the measured RTX 5090 times (1.2 = 20%% slower host)")
    parser.add_argument("--est-load-s", type=float, default=0.0, help="dry run: cold model load added to each GPU's first job")
    args = parser.parse_args(argv)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    try:
        plan = load_plan(args.plan)
        if args.dry_run:
            dry_run(plan, Path(args.out).resolve(), len(gpus), args.est_scale, args.est_load_s, args.prebuffer,
                    args.parallel_start, not args.no_warmup)
            return 0
        if not gpus:
            raise PlanError("--gpus lists no GPU")
        run(plan, Path(args.out), gpus, args.backend, Path(args.wangp_root), args.retries, not args.no_hls, args.prebuffer,
            args.fake_seconds, args.parallel_start, not args.no_warmup)
        return 0
    except PlanError as exc:
        print(f"director: {exc}", file=sys.stderr)
        return 2
    except RunError as exc:
        print(f"director: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
