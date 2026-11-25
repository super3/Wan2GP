#!/usr/bin/env python3
"""make_short.py: rebuild every deliverable of "The Littlest Crossing" in order (keys, shots.json + script.md, music bed,
title/end cards, animatic).  WARNING: after READY the keys and shots.json must not change; pass --no-keys to keep them."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(*args):
    subprocess.run([PY, os.path.join(HERE, args[0])] + list(args[1:]), check=True)


if __name__ == "__main__":
    if "--no-keys" not in sys.argv:
        run("short_keys.py")
        run("short_shots.py")
    run("short_music.py")
    run("short_cards.py", "both")
    run("short_animatic.py", "animatic")
