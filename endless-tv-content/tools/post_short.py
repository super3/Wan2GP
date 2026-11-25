#!/usr/bin/env python3
"""post_short.py SLUG RUN_ID SEED: from outputs/shorts/<slug>/shots.json write key-frame packs (bake/pack_<slug>_<n>.json, run ids
pack-<slug>-<n>, each under ~900 KB) and one jobs command (bake/cmd_<slug>.json, run id RUN_ID).  Prints the clip names.
Job names are <slug>__<shot id>; the clips come back as <slug>__<shot id>.mp4 (renamed to <shot id>.mp4 for assemble.py)."""
import base64
import io
import json
import os
import sys

from PIL import Image

O = "/home/user/Wan2GP/outputs/shorts"
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bake")
slug, run_id, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = os.path.join(O, slug)
manifest = json.load(open(os.path.join(d, "shots.json")))
keys, jobs = {}, []
for k, shot in enumerate(manifest["shots"]):
    names = []
    for which in ("start", "end"):
        rel = shot.get(which) or shot["start"]
        key = f"{slug}/{os.path.splitext(rel)[0]}"
        if key not in keys:
            im = Image.open(os.path.join(d, rel)).convert("RGB")
            assert im.size == (1280, 704), (rel, im.size)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=93, subsampling=0)
            keys[key] = base64.b64encode(buf.getvalue()).decode()
        names.append(key)
    jobs.append({"name": f"{slug}__{shot['id']}", "start": names[0], "end": names[1], "frames": int(shot["frames"]),
                 "seed": seed + k, "prompt": shot["prompt"].strip()})
packs, cur, size = [], {}, 0
for key, b64 in keys.items():
    if cur and size + len(b64) > 900_000:
        packs.append(cur)
        cur, size = {}, 0
    cur[key] = b64
    size += len(b64)
if cur:
    packs.append(cur)
for n, p in enumerate(packs, 1):
    json.dump({"id": f"pack-{slug}-{n}", "kind": "keyframe-pack", "images": p}, open(os.path.join(B, f"pack_{slug}_{n}.json"), "w"))
json.dump({"id": run_id, "cmd": "jobs", "jobs": jobs}, open(os.path.join(B, f"cmd_{slug}.json"), "w"))
total = sum(int(s["frames"]) for s in manifest["shots"])
print(f"{slug}: {len(jobs)} shots, {total} frames = {total / 24:.1f} s, {len(keys)} key frames in {len(packs)} packs")
print("files:", ",".join(j["name"] + ".mp4" for j in jobs))
