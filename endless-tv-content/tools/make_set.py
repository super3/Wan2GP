#!/usr/bin/env python3
"""make_set.py OUT.mp4 CLIP1.mp4 CLIP2.mp4 ...: join clips that each start and end on the same still into one set.

Every clip after the first drops its first frame (it repeats the previous clip's last frame) and the matching 1/24 s of sound,
so picture and sound stay in step. Video x264 CRF 16, audio AAC 256k."""
import json
import subprocess
import sys

out, clips = sys.argv[1], sys.argv[2:]
fps = 24
parts, labels = [], []
for i, clip in enumerate(clips):
    skip = 0 if i == 0 else 1
    frames = int(json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                            "stream=nb_read_frames", "-of", "json", clip], capture_output=True, text=True).stdout)["streams"][0]["nb_read_frames"])
    # H3's soundtrack runs ~0.1 s past the last frame: cut it to the picture, or the join pads the video with frozen frames
    parts.append(f"[{i}:v]trim=start_frame={skip}:end_frame={frames},setpts=PTS-STARTPTS[v{i}];"
                 f"[{i}:a]atrim=start={skip / fps:.6f}:end={frames / fps:.6f},asetpts=PTS-STARTPTS,aresample=48000,"
                 f"aformat=channel_layouts=stereo[a{i}];")
    labels.append(f"[v{i}][a{i}]")
graph = "".join(parts) + "".join(labels) + f"concat=n={len(clips)}:v=1:a=1[v][a]"
cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
for clip in clips:
    cmd += ["-i", clip]
cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out]
subprocess.run(cmd, check=True)
probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,nb_read_frames,duration",
                                   "-of", "json", out], capture_output=True, text=True).stdout)
print(out, json.dumps(probe["streams"]))
