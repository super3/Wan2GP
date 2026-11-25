#!/tmp/claude-0/-home-user-Wan2GP/4e1e6010-b786-5f93-b6fd-87ee44cba9e7/scratchpad/noroi-venv/bin/python
"""assemble.py RENDERS_DIR OUT.mp4 [ASSETS_DIR]

Joins title_card.mp4 + <shot id>.mp4 (from RENDERS_DIR, in shots.json order) + end_card.mp4, dropping the first frame of any
`continues` shot.  Mixes music_bed.wav under the H3 audio: the bed is cut per shot on its nominal timing and laid on the
real timeline, then ducked ~9 dB wherever the H3 track is loud.  Normalises loudness (two-pass ffmpeg loudnorm, I=-16),
scales to 1280x720 and writes H.264/AAC.  Needs only numpy, soundfile and ffmpeg/ffprobe.  Self-contained: imports nothing
from the drawing code."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

FPS, SR = 24, 48000
HERE_ASSETS = "/home/user/Wan2GP/outputs/shorts/wobble_suburb"
MUSIC_GAIN_DB = -4.0
DUCK_DB = -9.0


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def nframes(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
               "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return int(out.split(",")[0])


def has_audio(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
              capture_output=True, text=True).stdout.strip()
    return bool(out)


def read_audio(path, n_samples, skip_s=0.0):
    if not has_audio(path):
        return np.zeros((n_samples, 2), np.float32)
    raw = run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{skip_s:.4f}", "-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
              capture_output=True).stdout
    a = np.frombuffer(raw, np.float32).reshape(-1, 2)
    out = np.zeros((n_samples, 2), np.float32)
    m = min(n_samples, len(a))
    out[:m] = a[:m]
    return out


def fade_edges(x, ms=15):
    n = min(len(x) // 2, int(SR * ms / 1000))
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)[:, None]
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


def duck_gain(fg):
    """Music gain curve: 0 dB when the H3 track is quiet, DUCK_DB when it is loud; fast attack, slow release."""
    hop = SR // 100
    mono = fg.mean(1)
    n = len(mono) // hop + 1
    pad = np.zeros(n * hop, np.float32)
    pad[:len(mono)] = mono
    rms = np.sqrt((pad.reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    target = DUCK_DB * np.clip((db + 44) / 12, 0, 1)
    g = np.zeros(n)
    cur = 0.0
    a_att, a_rel = 1 - np.exp(-1 / 4), 1 - np.exp(-1 / 35)  # ~40 ms attack, ~350 ms release (per 10 ms hop)
    for i in range(n):
        cur += (target[i] - cur) * (a_att if target[i] < cur else a_rel)
        g[i] = cur
    gs = np.interp(np.arange(len(mono)), np.arange(n) * hop + hop / 2, g)
    return (10 ** (gs / 20)).astype(np.float32)[:, None]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    rdir, out = sys.argv[1], sys.argv[2]
    assets = sys.argv[3] if len(sys.argv) > 3 else HERE_ASSETS
    doc = json.load(open(os.path.join(assets, "shots.json")))
    segs = [dict(id="title", path=os.path.join(assets, "title_card.mp4"), continues=False, nominal=None)]
    for s in doc["shots"]:
        p = os.path.join(rdir, s["id"] + ".mp4")
        if not os.path.exists(p):
            sys.exit(f"missing render: {p}")
        segs.append(dict(id=s["id"], path=p, continues=bool(s.get("continues")), nominal=int(s["frames"])))
    segs.append(dict(id="end", path=os.path.join(assets, "end_card.mp4"), continues=False, nominal=None))

    # timings (actual and nominal) in frames
    t_act, t_nom = 0, 0
    for sg in segs:
        n = nframes(sg["path"])
        sg["skip"] = 1 if sg["continues"] else 0
        sg["frames"] = n - sg["skip"]
        sg["nom"] = (sg["nominal"] if sg["nominal"] is not None else n) - sg["skip"]
        sg["t_act"], sg["t_nom"] = t_act, t_nom
        t_act += sg["frames"]
        t_nom += sg["nom"]
        print(f"{sg['id']:>6}: {n} frames{' (drop first)' if sg['skip'] else ''}, at {sg['t_act'] / FPS:7.2f}s")
    total = t_act / FPS

    with tempfile.TemporaryDirectory() as tmp:
        # ---- picture: trim, conform, scale, concat
        args = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
        flt = []
        for k, sg in enumerate(segs):
            args += ["-i", sg["path"]]
            flt.append(f"[{k}:v]trim=start_frame={sg['skip']},setpts=PTS-STARTPTS,fps={FPS},scale=1280:720:flags=lanczos,setsar=1,"
                       f"format=yuv420p,trim=end_frame={sg['frames']}[v{k}]")
        flt.append("".join(f"[v{k}]" for k in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]")
        vid = os.path.join(tmp, "v.mp4")
        run(args + ["-filter_complex", ";".join(flt), "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-r", str(FPS), vid])

        # ---- sound: H3/card track on the real timeline, music bed cut per segment and re-laid, ducked under the H3 track
        N = int(round(total * SR))
        fg = np.zeros((N, 2), np.float32)
        mus = np.zeros((N, 2), np.float32)
        bed_path = os.path.join(assets, "music_bed.wav")
        bed, bsr = sf.read(bed_path, dtype="float32", always_2d=True) if os.path.exists(bed_path) else (np.zeros((1, 2), np.float32), SR)
        assert bsr == SR, "music bed must be 48 kHz"
        if bed.shape[1] == 1:
            bed = np.repeat(bed, 2, 1)
        for sg in segs:
            a0 = int(round(sg["t_act"] / FPS * SR))
            n = int(round(sg["frames"] / FPS * SR))
            n = min(n, N - a0)
            fg[a0:a0 + n] = read_audio(sg["path"], n, sg["skip"] / FPS)
            b0 = int(round(sg["t_nom"] / FPS * SR))
            piece = np.zeros((n, 2), np.float32)
            src = bed[b0:b0 + n]
            piece[:len(src)] = src
            mus[a0:a0 + n] += fade_edges(piece, 10) if sg["frames"] != sg["nom"] else piece
        mix = fg + mus * (10 ** (MUSIC_GAIN_DB / 20)) * duck_gain(fg)
        peak = np.max(np.abs(mix))
        if peak > .98:
            mix *= .98 / peak
        wav = os.path.join(tmp, "mix.wav")
        sf.write(wav, mix, SR, subtype="FLOAT")

        # ---- loudness: two-pass loudnorm to I=-16 LUFS, TP -1.5
        meas = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
        js = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
        ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}:"
              f"measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true,aresample={SR}")
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-af", ln, "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2", "-t", f"{total:.4f}", "-movflags", "+faststart", out])
    print(f"wrote {out}: {len(segs)} segments, {total:.2f}s")


if __name__ == "__main__":
    main()
