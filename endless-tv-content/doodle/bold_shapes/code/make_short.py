#!/usr/bin/env python3
"""make_short.py: build the deliverables of the bold_shapes short 'One Little Smudge'.

  python make_short.py keys       render every key frame (1280x704) into OUT/keys
  python make_short.py json       write OUT/shots.json from story.py
  python make_short.py cards      title_card.mp4 + end_card.mp4
  python make_short.py music      music_bed.wav (timed to the shot list)
  python make_short.py animatic   animatic.mp4 (keys on the real timeline + music)
  python make_short.py script     script.md
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = "/home/user/Wan2GP/outputs/shorts/bold_shapes"
FPS = 24
TITLE_FRAMES, END_FRAMES = 72, 48


def timeline():
    """[(name, first_frame, n_frames)] for the assembled short: title card, shots (continues drop their first frame), end card."""
    import story
    t, out = 0, [("title", 0, TITLE_FRAMES)]
    t = TITLE_FRAMES
    for s in story.SHOTS:
        n = s["frames"] - (1 if s["continues"] else 0)
        out.append((s["id"], t, n))
        t += n
    out.append(("end", t, END_FRAMES))
    return out, t + END_FRAMES


def do_keys():
    import scenes
    scenes.render_all(os.path.join(OUT, "keys"))


def do_json():
    import story
    prompts = story.shot_prompts()
    shots = []
    for s, pr in zip(story.SHOTS, prompts):
        for k in (s["start"], s["end"]):
            assert os.path.exists(os.path.join(OUT, "keys", k + ".png")), k
        shots.append({"id": s["id"], "frames": s["frames"], "start": f"keys/{s['start']}.png", "end": f"keys/{s['end']}.png",
                      "continues": s["continues"], "prompt": pr})
    with open(os.path.join(OUT, "shots.json"), "w") as f:
        json.dump({"title": f"{story.SERIES}: {story.TITLE}", "shots": shots}, f, indent=2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for what in sys.argv[1:]:
        if what == "keys":
            do_keys()
        elif what == "json":
            do_json()
        elif what == "cards":
            import cards
            cards.make(OUT)
        elif what == "music":
            import score
            score.make(OUT, timeline())
        elif what == "animatic":
            import animatic
            animatic.make(OUT, timeline())
        elif what == "script":
            import scriptmd
            scriptmd.make(OUT, timeline())
        else:
            raise SystemExit(f"unknown step {what}")
