#!/usr/bin/env python3
"""assemble.py RENDERS_DIR OUT.mp4 [--deliv DIR]

Joins the "One Small Passenger" short: title card + the H3 shot renders (RENDERS_DIR/<shot id>.mp4, 1280x704, 24 fps, with H3's
audio) + end card, then lays the music bed under it.
- Order and lengths come from DIR/shots.json (DIR defaults to the deliverables folder).
- A "continues" shot loses its first frame (it repeats the previous shot's last frame).
- Each segment is conformed to its nominal length (trimmed, or its last frame held) so the music stays in sync.
- The music is ducked about 9 dB wherever the H3/card audio is loud, with a smooth attack and release.
- Loudness is normalised with two-pass ffmpeg loudnorm (I=-16 LUFS, TP=-1.5 dBTP).
- Output is scaled to 1280x720 and written as H.264/AAC mp4.
A render without an audio stream is treated as silence.  Needs only python3 + numpy + ffmpeg/ffprobe."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

FPS, SR = 24, 48000
DELIV = "/home/user/Wan2GP/outputs/shorts/geometric_lab"
TITLE_FRAMES, END_FRAMES = 72, 48
DUCK_DB, MUSIC_DB = 9.0, -3.0          # ducking depth; overall music level relative to the bed file


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd[:6])} ...\n{r.stderr[-2000:]}")
    return r


def has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", path])
    return bool(json.loads(r.stdout).get("streams"))


def audio(path, start_s, dur_s):
    """Decode a segment's audio as float32 stereo 48 kHz, from start_s, padded/trimmed to dur_s."""
    n = int(round(dur_s * SR))
    out = np.zeros((n, 2), np.float32)
    if path and has_audio(path):
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start_s:.6f}", "-i", path, "-vn", "-ac", "2", "-ar", str(SR),
                            "-f", "f32le", "-"], capture_output=True)
        if r.returncode != 0:
            sys.exit(f"audio decode failed for {path}:\n{r.stderr.decode()[-1500:]}")
        a = np.frombuffer(r.stdout, np.float32).reshape(-1, 2)[:n]
        out[:len(a)] = a
    return out


def smooth_env(x, attack_s=.03, release_s=.45):
    """One-pole attack/release follower on a per-sample level."""
    a_up, a_dn = np.exp(-1 / (attack_s * SR / 64)), np.exp(-1 / (release_s * SR / 64))
    blocks = x[: len(x) // 64 * 64].reshape(-1, 64).max(1)
    env = np.empty_like(blocks)
    v = 0.0
    for i, b in enumerate(blocks):
        v = a_up * v + (1 - a_up) * b if b > v else a_dn * v + (1 - a_dn) * b
        env[i] = v
    env = np.repeat(env, 64)
    return np.pad(env, (0, len(x) - len(env)), mode="edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("renders")
    ap.add_argument("out")
    ap.add_argument("--deliv", default=DELIV)
    args = ap.parse_args()
    deliv = args.deliv
    spec = json.load(open(os.path.join(deliv, "shots.json")))

    # ---- segment list: (path, first frame to use, frame count)
    segs = [(os.path.join(deliv, "title_card.mp4"), 0, TITLE_FRAMES)]
    missing = []
    for s in spec["shots"]:
        p = os.path.join(args.renders, s["id"] + ".mp4")
        if not os.path.exists(p):
            missing.append(p)
        skip = 1 if s.get("continues") else 0
        segs.append((p, skip, s["frames"] - skip))
    segs.append((os.path.join(deliv, "end_card.mp4"), 0, END_FRAMES))
    for p in (segs[0][0], segs[-1][0]):
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        sys.exit("missing inputs:\n  " + "\n  ".join(missing))
    total_frames = sum(n for _, _, n in segs)
    total_s = total_frames / FPS
    print(f"{len(segs)} segments, {total_frames} frames = {total_s:.2f} s")

    with tempfile.TemporaryDirectory() as tmp:
        # ---- audio: the H3/card track, the ducked music bed, the mix
        dia = np.concatenate([audio(p, k / FPS, n / FPS) for p, k, n in segs])
        n = len(dia)
        music = audio(os.path.join(deliv, "music_bed.wav"), 0, n / SR)
        lvl_db = 20 * np.log10(np.maximum(np.sqrt(smooth_env(dia.mean(1) ** 2, .02, .25)), 1e-6))
        depth = np.clip((lvl_db + 36) / 12, 0, 1) * DUCK_DB                    # no duck under quiet ambience (< -36 dB), full by -24 dB
        depth = smooth_env(depth, .04, .5)
        gain = 10 ** ((MUSIC_DB - depth) / 20)
        mix = dia + music * gain[:, None]
        wav = os.path.join(tmp, "mix.f32")
        mix.astype(np.float32).tofile(wav)
        raw = ["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", wav]
        # two-pass loudnorm
        r = run(["ffmpeg", "-nostdin", "-hide_banner", *raw, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
        m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
        ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
              f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
        aac_in = os.path.join(tmp, "mix.wav")
        run(["ffmpeg", "-nostdin", "-v", "error", *raw, "-af", ln + ",aresample=48000", "-c:a", "pcm_s24le", aac_in])

        # ---- video: conform every segment to its nominal frame count, scale to 1280x720, concatenate
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex_threads", "2"]
        chains = []
        for i, (p, k, nf) in enumerate(segs):
            cmd += ["-i", p]
            chains.append(f"[{i}:v]fps={FPS},scale=1280:720:flags=lanczos,setsar=1,trim=start_frame={k},setpts=PTS-STARTPTS,"
                          f"tpad=stop_mode=clone:stop={nf},trim=end_frame={nf},setpts=PTS-STARTPTS,format=yuv420p[v{i}]")
        cmd += ["-i", aac_in]
        graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]"
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", f"{len(segs)}:a", "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", "17",
                "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-t", f"{total_s:.4f}",
                "-movflags", "+faststart", args.out]
        run(cmd)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
