#!/usr/bin/env python3
"""assemble.py RENDERS_DIR OUT.mp4 [--deliv DIR] [--music-gain G]

Joins title card + the H3 shot renders (RENDERS_DIR/<shot id>.mp4, in shots.json order) + end card into the finished short.
  * every clip is conformed to its exact length from shots.json (trimmed, or its last frame held), scaled/padded to 1280x704 at
    24 fps, so the music bed stays in sync even if a render comes back a few frames off; a 'continues' shot drops its first frame
  * a clip with no audio stream gets silence
  * the music bed is mixed under the H3 audio and ducked by a sidechain compressor keyed on that audio (~8-10 dB under speech)
  * loudness is normalised with two-pass ffmpeg loudnorm (I=-16 LUFS, TP=-1.5 dBTP); output is 1280x720 H.264/AAC
Only the Python standard library and ffmpeg/ffprobe are needed."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

DELIV = "/home/user/Wan2GP/outputs/shorts/bold_shapes"
FPS = 24
TITLE_FRAMES, END_FRAMES = 72, 48


def has_audio(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"ffmpeg failed: {' '.join(cmd[:6])} ...")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("renders")
    ap.add_argument("out")
    ap.add_argument("--deliv", default=DELIV)
    ap.add_argument("--music-gain", type=float, default=0.55)
    a = ap.parse_args()
    with open(os.path.join(a.deliv, "shots.json")) as f:
        shots = json.load(f)["shots"]
    clips = [(os.path.join(a.deliv, "title_card.mp4"), TITLE_FRAMES, 0)]
    for s in shots:
        p = os.path.join(a.renders, f"{s['id']}.mp4")
        if not os.path.exists(p):
            raise SystemExit(f"missing render: {p}")
        clips.append((p, s["frames"], 1 if s.get("continues") else 0))
    clips.append((os.path.join(a.deliv, "end_card.mp4"), END_FRAMES, 0))
    total = sum(n - k for _, n, k in clips)
    music = os.path.join(a.deliv, "music_bed.wav")

    fc, ins = [], []
    for i, (path, n, skip) in enumerate(clips):
        ins += ["-i", path]
        keep = n - skip
        fc.append(f"[{i}:v]scale=1280:704:force_original_aspect_ratio=decrease:flags=lanczos,pad=1280:704:(ow-iw)/2:(oh-ih)/2,"
                  f"setsar=1,fps={FPS},format=yuv420p,trim=start_frame={skip},setpts=PTS-STARTPTS,"
                  f"tpad=stop_mode=clone:stop_duration=30,trim=end_frame={keep},setpts=PTS-STARTPTS[v{i}]")
        dur = keep / FPS
        if has_audio(path):
            fc.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=start={skip / FPS:.6f},"
                      f"asetpts=PTS-STARTPTS,apad,atrim=end={dur:.6f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            fc.append(f"anullsrc=r=48000:cl=stereo,atrim=end={dur:.6f},aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
    m = len(clips)
    ins += ["-i", music]
    fc.append("".join(f"[v{i}][a{i}]" for i in range(m)) + f"concat=n={m}:v=1:a=1[v][fg]")
    fc.append(f"[{m}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=end={total / FPS:.6f},"
              f"volume={a.music_gain}[mus]")
    fc.append("[fg]asplit=2[fg1][key]")
    fc.append("[mus][key]sidechaincompress=threshold=0.02:ratio=4:attack=15:release=450:knee=3:detection=rms[musd]")
    fc.append("[fg1][musd]amix=inputs=2:normalize=0:duration=first[mix]")
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "graph.txt")
        with open(script, "w") as f:
            f.write(";\n".join(fc))
        vid, wav = os.path.join(tmp, "v.mp4"), os.path.join(tmp, "mix.wav")
        run(["ffmpeg", "-nostdin", "-y", "-v", "error"] + ins + ["-filter_complex_script", script,
             "-map", "[v]", "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS), vid,
             "-map", "[mix]", "-c:a", "pcm_s24le", "-ar", "48000", wav])
        r = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
        meas = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S).group(0))
        ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:"
              f"measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}:linear=true")
        run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", vid, "-i", wav, "-filter_complex",
             f"[0:v]scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[vo];[1:a]{ln},aresample=48000[ao]",
             "-map", "[vo]", "-map", "[ao]", "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-r", str(FPS),
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", a.out])
    print(f"wrote {a.out}: {total} frames ({total / FPS:.2f} s), {m} clips, input loudness {meas['input_i']} LUFS -> -16")


if __name__ == "__main__":
    main()
