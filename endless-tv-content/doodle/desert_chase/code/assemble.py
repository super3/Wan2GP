#!/usr/bin/env python3
"""assemble.py RENDERS_DIR OUT.mp4 [--deliv DIR]

Builds the finished short "Clean Sweep":
  title_card.mp4 + <shot id>.mp4 for every shot in shots.json (screen order) + end_card.mp4
- every shot is forced to exactly its `frames` (24 fps; a short render is padded by holding its last frame, a long one is trimmed),
  and the first frame of a `"continues": true` shot is dropped (it repeats the previous shot's end frame);
- the H3 audio of each shot is kept (silence if a render has no audio track); the cards contribute no audio of their own because
  the music bed already contains their stings;
- music_bed.wav is mixed underneath and ducked about 9 dB wherever the H3 track is loud (envelope follower, 30 ms attack,
  350 ms release);
- loudness is normalised with ffmpeg loudnorm (I=-16, TP=-1.5, LRA=11), the picture is scaled to 1280x720,
  and the result is written as H.264 / AAC (yuv420p, +faststart).
DIR defaults to /home/user/Wan2GP/outputs/shorts/desert_chase (holds shots.json, title_card.mp4, end_card.mp4, music_bed.wav).
Needs python3 with numpy, and ffmpeg / ffprobe on PATH."""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

FPS, SR = 24, 48000
TITLE_FRAMES, END_FRAMES = 72, 48
DUCK_DB, LO_DB, HI_DB = 9.0, -46.0, -30.0   # full duck once the H3 track is above HI_DB (RMS), none below LO_DB


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def read_audio(path):
    """Decode a file's audio to float32 stereo at 48 kHz (empty array if it has none)."""
    if not has_audio(path):
        return np.zeros((0, 2), np.float32)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()


def fit(x, n):
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.zeros((n - len(x), 2), np.float32)])


def duck_gain(vox, block=480):
    """Per-sample gain (linear) for the music: -DUCK_DB when the H3 track is loud, smoothed with attack/release."""
    mono = vox.mean(1)
    nb = int(np.ceil(len(mono) / block))
    pad = np.pad(mono, (0, nb * block - len(mono)))
    rms = np.sqrt((pad.reshape(nb, block) ** 2).mean(1) + 1e-12)
    rms = np.convolve(rms, np.ones(3) / 3, mode="same")              # 30 ms window
    db = 20 * np.log10(rms + 1e-9)
    target = -DUCK_DB * np.clip((db - LO_DB) / (HI_DB - LO_DB), 0, 1)
    g = np.zeros(nb)
    cur = 0.0
    a_att, a_rel = 1 - np.exp(-1 / 3.0), 1 - np.exp(-1 / 35.0)      # 30 ms attack, 350 ms release (10 ms blocks)
    for i in range(nb):
        k = a_att if target[i] < cur else a_rel
        cur += (target[i] - cur) * k
        g[i] = cur
    gs = np.repeat(g, block)[:len(mono)]
    return (10 ** (gs / 20)).astype(np.float32)[:, None], g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("renders")
    ap.add_argument("out")
    ap.add_argument("--deliv", default="/home/user/Wan2GP/outputs/shorts/desert_chase")
    args = ap.parse_args()
    d = args.deliv
    shots = json.load(open(os.path.join(d, "shots.json")))["shots"]
    segs = [(os.path.join(d, "title_card.mp4"), 0, TITLE_FRAMES, False)]
    for s in shots:
        p = os.path.join(args.renders, f"{s['id']}.mp4")
        if not os.path.exists(p):
            sys.exit(f"missing render: {p}")
        skip = 1 if s.get("continues") else 0
        segs.append((p, skip, s["frames"] - skip, True))
    segs.append((os.path.join(d, "end_card.mp4"), 0, END_FRAMES, False))
    total = sum(n for _, _, n, _ in segs)

    with tempfile.TemporaryDirectory() as tmp:
        # ---- picture: exact frame counts, 1280x720, concatenated
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        for p, _, _, _ in segs:
            cmd += ["-i", p]
        fl = []
        for i, (p, skip, n, _) in enumerate(segs):
            fl.append(f"[{i}:v]fps={FPS},scale=1280:720:flags=lanczos,setsar=1,format=yuv420p,tpad=stop_mode=clone:stop_duration=5,"
                      f"trim=start_frame={skip}:end_frame={skip + n},setpts=PTS-STARTPTS[v{i}]")
        fl.append("".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[vout]")
        vid = os.path.join(tmp, "v.mp4")
        subprocess.run(cmd + ["-filter_complex", ";".join(fl), "-map", "[vout]", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                              "-pix_fmt", "yuv420p", "-r", str(FPS), vid], check=True)

        # ---- sound: H3 audio per shot on the same timeline, music bed ducked under it
        vox = []
        for p, skip, n, is_shot in segs:
            want = int(round(n / FPS * SR))
            if is_shot:
                a = read_audio(p)
                a = a[int(round(skip / FPS * SR)):]
                vox.append(fit(a, want))
            else:
                vox.append(np.zeros((want, 2), np.float32))
        vox = np.vstack(vox)
        bed = fit(read_audio(os.path.join(d, "music_bed.wav")), len(vox))
        g, gb = duck_gain(vox)
        mix = vox + bed * g
        raw = os.path.join(tmp, "mix.f32")
        mix.astype("<f4").tofile(raw)
        print(f"timeline {total} frames = {total / FPS:.2f} s; music ducked up to {-gb.min():.1f} dB "
              f"(>{DUCK_DB - 1:.0f} dB for {np.mean(gb < -(DUCK_DB - 1)) * 100:.0f}% of the time)")

        # ---- mux, loudness-normalise, encode
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", vid, "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", raw,
                        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SR), "-ac", "2",
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", args.out], check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
