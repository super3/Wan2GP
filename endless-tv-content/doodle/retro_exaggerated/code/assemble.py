#!/usr/bin/env python3
"""assemble.py RENDERS_DIR OUT.mp4 [--deliv DIR] [--no-bed]

Joins the title card + every shot in shots.json order (RENDERS_DIR/<shot id>.mp4, 1280x704, 24 fps, with H3's audio) + the end
card into one short.  Per shot: forced to 24 fps / 1280x704, the first frame dropped when "continues" is true, then trimmed or
padded (last frame held, silence added) to exactly its listed frame count, so the music bed stays in sync even if a render comes
back a frame long or short.  A shot with no audio stream gets silence.  The cards' own audio is replaced by silence because their
stings are already inside music_bed.wav.

Audio: the music bed is ducked under the H3 track with a sidechain compressor (about 10 dB under loud speech), mixed,
then loudness-normalised with ffmpeg loudnorm (two passes, I=-16 LUFS, TP=-1.5 dBTP).  Video is scaled to 1280x720 and written as
H.264 (yuv420p) + AAC.  Needs only python3 + ffmpeg/ffprobe on PATH; no third-party Python modules.
"""
import argparse
import json
import os
import re
import subprocess
import sys

DELIV = "/home/user/Wan2GP/outputs/shorts/retro_exaggerated"
FPS = 24
TITLE_FRAMES, END_FRAMES = 72, 48


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def build_graph(segments, bed_index, total_s, bed_gain, loudnorm, video=True):
    """segments: list of (input index, frames to keep, skip first frame?, use audio?)"""
    parts, cat = [], []
    for k, (idx, n, skip, use_audio) in enumerate(segments):
        keep = n - skip
        dur = keep / FPS
        if video:
            parts.append(f"[{idx}:v]fps={FPS},scale=1280:704:flags=lanczos,setsar=1,format=yuv420p,trim=start_frame={skip},setpts=PTS-STARTPTS,"
                         f"tpad=stop_mode=clone:stop={keep},trim=end_frame={keep},setpts=PTS-STARTPTS[v{k}]")
        if use_audio:
            parts.append(f"[{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=start={skip / FPS:.6f},"
                         f"asetpts=PTS-STARTPTS,apad=whole_dur={dur:.6f},atrim=end={dur:.6f},asetpts=PTS-STARTPTS[a{k}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=end={dur:.6f},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                         f"asetpts=PTS-STARTPTS[a{k}]")
        cat.append(f"[v{k}][a{k}]" if video else f"[a{k}]")
    if video:
        parts.append("".join(cat) + f"concat=n={len(segments)}:v=1:a=1[vcat][dlg]")
    else:
        parts.append("".join(cat) + f"concat=n={len(segments)}:v=0:a=1[dlg]")
    parts.append("[dlg]asplit=2[dlg1][sc]")
    if bed_index is not None:
        parts.append(f"[{bed_index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={bed_gain},"
                     f"apad=whole_dur={total_s:.6f},atrim=end={total_s:.6f},asetpts=PTS-STARTPTS[bed]")
        # measured: speech at -18 dBFS RMS pulls the bed down ~10 dB, quieter speech (-24 dB) ~6 dB, silence leaves it alone
        parts.append("[bed][sc]sidechaincompress=threshold=0.05:ratio=3:attack=15:release=380:knee=3:makeup=1[bedduck]")
        parts.append("[dlg1][bedduck]amix=inputs=2:duration=first:normalize=0[mix]")
    else:
        parts.append("[sc]anullsink")
        parts.append("[dlg1]anull[mix]")
    parts.append(f"[mix]{loudnorm},aresample=48000[aout]")
    if video:
        parts.append("[vcat]scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[vout]")
    return ";".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders_dir")
    ap.add_argument("out")
    ap.add_argument("--deliv", default=DELIV, help="folder holding shots.json, the cards and music_bed.wav")
    ap.add_argument("--no-bed", action="store_true", help="leave the music bed out (for checking the H3 audio alone)")
    ap.add_argument("--bed-gain", type=float, default=1.0)
    a = ap.parse_args()

    shots = json.load(open(os.path.join(a.deliv, "shots.json")))["shots"]
    inputs, segments = [], []

    def add_input(path):
        if not os.path.exists(path):
            sys.exit(f"missing: {path}")
        inputs.append(path)
        return len(inputs) - 1

    t = add_input(os.path.join(a.deliv, "title_card.mp4"))
    segments.append((t, TITLE_FRAMES, 0, False))
    for s in shots:
        p = os.path.join(a.renders_dir, f"{s['id']}.mp4")
        i = add_input(p)
        segments.append((i, s["frames"], 1 if s.get("continues") else 0, has_audio(p)))
    e = add_input(os.path.join(a.deliv, "end_card.mp4"))
    segments.append((e, END_FRAMES, 0, False))
    total_frames = sum(n - skip for _, n, skip, _ in segments)
    total_s = total_frames / FPS
    bed_index = None
    if not a.no_bed:
        bed_index = add_input(os.path.join(a.deliv, "music_bed.wav"))

    cmd_in = []
    for p in inputs:
        cmd_in += ["-i", p]

    # pass 1: measure loudness of the final mix
    g1 = build_graph(segments, bed_index, total_s, a.bed_gain, "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", video=False)
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-y"] + cmd_in +
                       ["-filter_complex", g1, "-map", "[aout]", "-f", "null", "-"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("loudness pass failed:\n" + r.stderr[-3000:])
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if m:
        j = json.loads(m[-1])
        ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
              f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    else:
        ln = "loudnorm=I=-16:TP=-1.5:LRA=11"

    # pass 2: render
    g2 = build_graph(segments, bed_index, total_s, a.bed_gain, ln)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y"] + cmd_in +
                       ["-filter_complex", g2, "-map", "[vout]", "-map", "[aout]", "-r", str(FPS), "-c:v", "libx264", "-crf", "18",
                        "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        "-t", f"{total_s:.6f}", "-movflags", "+faststart", a.out], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("render failed:\n" + r.stderr[-3000:])
    print(f"wrote {a.out}: {total_frames} frames, {total_s:.2f} s, {len(shots)} shots")


if __name__ == "__main__":
    main()
