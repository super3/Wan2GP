"""Score for "The Littlest Crossing": an original early-60s TV-cartoon cue sheet, timed shot by shot to the shot list.
Bertram gets a pompous oom-pah march (tuba, muted brass, snare), Tilly a tiptoe pizzicato/celesta figure, the hydrant a celesta
lullaby; every gag gets a hit.  Kept sparse and mid-low in level so it sits under dialogue (assemble.py ducks it further)."""
import math
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_shots as SH  # noqa: E402
from sound import (SR, bp, brass, brush, hp, hz, lp, noise, phase, pluck_bass, put, slide_whistle, tt, vibes,  # noqa: E402
                   whoosh, woodblock, xylo, env, reverb, pan)

TL, TOTAL_FRAMES = SH.timeline()
T0 = {sid: f0 / 24 for sid, f0, n in TL}
TLEN = {sid: n / 24 for sid, f0, n in TL}
DUR = TOTAL_FRAMES / 24
END0 = DUR - SH.END_FRAMES / 24


# ----------------------------------------------------------------------------------------------- extra instruments
def tuba(m, dur, g=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = hz(m)
    x = (2 * (phase(f, n) % 1) - 1) + .5 * np.sin(2 * np.pi * f * t)
    return lp(x, 520) * env(n, .025, .08, .7, .07) * g


def pah(notes, dur):
    """The off-beat 'pah' of the oom-pah: a short muted-brass chord."""
    n = int(dur * SR)
    x = np.zeros(n)
    for m in notes:
        f = hz(m)
        x += (2 * (phase(f, n) % 1) - 1) + .5 * (2 * (phase(f * 1.004, n) % 1) - 1)
    return bp(x, 400, 1800) * env(n, .01, .06, .5, .04)


def celesta(m, dur=1.2, g=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + .35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6) + .15 * np.sin(2 * np.pi * 4.1 * f * t) * np.exp(-t * 14)
    return x * np.exp(-t * 3.2) * np.clip(t / .002, 0, 1) * g


def pizz(m, dur=.3):
    n = int(dur * SR)
    t = tt(n)
    f = hz(m)
    x = (2 * (phase(f, n) % 1) - 1)
    return lp(x, 2400) * np.exp(-t * 16) * np.clip(t / .002, 0, 1)


def reed(m, dur, cutoff=1400, g=1.0, vib=.006):
    """Bassoon/clarinet-ish: a filtered square with a slow vibrato."""
    n = int(dur * SR)
    t = tt(n)
    f = hz(m) * (1 + vib * np.sin(2 * np.pi * 5 * t) * np.clip(t / .25, 0, 1))
    x = np.where(phase(f, n) % 1 < .5, 1.0, -1.0)
    return lp(x, cutoff) * env(n, .04, .1, .8, .08) * g


def snare(dur=.18, seed=3):
    n = int(dur * SR)
    t = tt(n)
    return bp(noise(n, seed), 1200, 8000) * np.exp(-t * 26) + .4 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 35)


def roll(dur, seed=5, g0=.3, g1=1.0):
    n = int(dur * SR)
    out = np.zeros(n)
    k = 0
    for t0 in np.arange(0, dur - .04, .045):
        put(out, snare(.08, seed + k), t0, g0 + (g1 - g0) * t0 / dur)
        k += 1
    return out


def cymbal(dur=1.6, seed=8):
    n = int(dur * SR)
    t = tt(n)
    return hp(noise(n, seed), 4500) * np.exp(-t * 2.6) * np.clip(t / .004, 0, 1)


def timp(m, dur=.8):
    n = int(dur * SR)
    t = tt(n)
    f = hz(m) * (1 + .05 * np.exp(-t * 20))
    return np.sin(2 * np.pi * phase(f, n)) * np.exp(-t * 4) + .2 * bp(noise(n, 4), 80, 400) * np.exp(-t * 30)


def harp_gliss(m0, m1, dur, seed=0):
    out = np.zeros(int((dur + 1.0) * SR))
    scale = [0, 2, 4, 5, 7, 9, 11]
    notes = [m for m in range(m0, m1 + 1) if (m - 5) % 12 in scale]
    for k, m in enumerate(notes):
        put(out, pizz(m, .9) * .8 + celesta(m, .9) * .3, k * dur / len(notes))
    return out


# ----------------------------------------------------------------------------------------------- composition helpers
class Score:
    def __init__(self):
        n = int(DUR * SR) + SR
        self.tr = {k: np.zeros(n) for k in ("bass", "brass", "perc", "keys", "reed", "fx")}

    def add(self, track, x, t, g=1.0):
        put(self.tr[track], x, t, g)


def march(S, t0, bars, beat, g=1.0, melody=True, snare_g=.5, prog=None):
    """Bertram's march: oom-pah in F, 2 beats per bar, with the pompous theme on muted brass."""
    prog = prog or [(41, (57, 60, 65)), (36, (58, 60, 64)), (41, (57, 60, 65)), (36, (55, 58, 64))]
    theme = [(65, 1), (69, 1), (72, 2), (72, .5), (74, .5), (72, .5), (69, .5), (70, 1), (67, 1), (65, 2)]
    for b in range(bars):
        root, ch = prog[b % len(prog)]
        tb = t0 + b * 2 * beat
        S.add("bass", tuba(root, beat * .8), tb, g)
        S.add("brass", pah(ch, beat * .45), tb + beat, .5 * g)
        S.add("perc", snare(.16, b), tb + beat, snare_g * g)
        S.add("perc", brush(.05, seed=300 + b, swish=False), tb + beat * .5, .25 * g)
    if melody:
        t = t0
        for m, d in theme:
            if t > t0 + bars * 2 * beat - .05:
                break
            S.add("brass", brass([m], d * beat * .95, wah=False, cutoff=2600), t, .55 * g)
            t += d * beat


def walk(S, t0, t1, beat, notes, g=1.0, brushes=True):
    k = 0
    t = t0
    while t < t1 - .05:
        S.add("bass", pluck_bass(notes[k % len(notes)], beat * .95, seed=k), t, g)
        if brushes and k % 2 == 1:
            S.add("perc", brush(.22, seed=k), t - .03, .35 * g)
        t += beat
        k += 1


def tiptoe(S, t0, beat, g=1.0, reps=1):
    fig = [(84, .5), (81, .5), (77, .5), (81, .5), (84, 1), (None, 1), (86, .5), (84, .5), (82, .5), (81, .5), (79, 1)]
    t = t0
    for _ in range(reps):
        for m, d in fig:
            if m:
                S.add("keys", pizz(m - 12, .3), t, .7 * g)
                S.add("keys", celesta(m, .5), t, .25 * g)
            t += d * beat
    return t


def lullaby(S, t0, beat, bars, g=1.0):
    mel = [(77, 2), (81, 1), (84, 2), (81, 1), (79, 2), (76, 1), (77, 3), (74, 2), (77, 1), (81, 2), (79, 1), (77, 3)]
    t = t0
    k = 0
    end = t0 + bars * 3 * beat
    for m, d in mel:
        if t >= end:
            break
        S.add("keys", celesta(m, d * beat + .6), t, .6 * g)
        t += d * beat
    for b in range(bars):           # waltz bass: oom-pa-pa
        tb = t0 + b * 3 * beat
        root = (41, 36, 41, 43)[b % 4]
        S.add("bass", pluck_bass(root, beat * .9, seed=b), tb, .6 * g)
        for q in (1, 2):
            S.add("keys", vibes(root + 28 + (4 if q == 2 else 0), .4, g=.5), tb + q * beat, .25 * g)


def tada(S, t, g=1.0, chord=(65, 69, 72, 77)):
    S.add("brass", brass(list(chord), .9, cutoff=3000), t, .9 * g)
    S.add("perc", cymbal(1.4), t, .45 * g)
    S.add("bass", tuba(41, .6), t, .9 * g)
    S.add("perc", timp(41, .8), t, .6 * g)


def wahwah(S, t, g=1.0):
    for k, m in enumerate((64, 63, 62)):
        S.add("brass", brass([m], .32, cutoff=1500), t + k * .34, .7 * g)
    S.add("brass", brass([61], .9, cutoff=1300), t + 1.02, .7 * g)


# ----------------------------------------------------------------------------------------------- the cue sheet
def title_sting(S, t0):
    beat = 60 / 132
    S.add("perc", roll(.7, g0=.15, g1=.8), t0 + .05, .5)
    for k, m in enumerate((65, 69, 72)):                     # "BER-TRAM BIGS-BY!"
        S.add("brass", brass([m, m - 4], .28, cutoff=2600), t0 + .75 + k * beat * .5, .8)
    S.add("brass", brass([77, 72, 69, 65], 1.1, cutoff=3200), t0 + .75 + 1.5 * beat, .9)
    S.add("perc", cymbal(1.8), t0 + .75 + 1.5 * beat, .5)
    S.add("bass", tuba(41, .9), t0 + .75 + 1.5 * beat, 1.0)
    for k, m in enumerate((72, 74, 76, 77, 79, 81, 84, 89)):
        S.add("keys", xylo(m), t0 + 2.05 + k * .035, .5)
    S.add("perc", woodblock(900, .15), t0 + 2.6, .4)


def end_sting(S, t0):
    for k, m in enumerate((84, 81, 77, 72)):
        S.add("keys", xylo(m), t0 + .05 + k * .07, .55)
    tada(S, t0 + .4, .9, chord=(65, 69, 72, 76))
    S.add("keys", vibes(77, 1.4), t0 + .4, .5)
    S.add("keys", vibes(81, 1.4), t0 + .4, .4)


def compose():
    S = Score()
    beat = 60 / 132
    title_sting(S, 0.0)
    # s01: Bertram's corner.  Walking bass + brushes under the brag, a swell on the chest puff
    t = T0["s01"]
    walk(S, t, t + TLEN["s01"], beat, [41, 45, 48, 50, 46, 50, 53, 50], g=.8)
    S.add("brass", brass([53, 57, 60], 1.2, cutoff=1800), t + 1.0, .45)       # chest puff swell
    S.add("keys", vibes(72, 1.2), t + .1, .35)
    S.add("keys", vibes(77, 1.2), t + 8.4, .35)
    # s02: Tilly's tiptoe figure, light
    t = T0["s02"]
    tiptoe(S, t + .2, beat * .9, g=1.0)
    S.add("keys", pizz(65, .4), t + 7.4, .7)
    S.add("fx", woodblock(1250, .1), t + 8.2, .25)
    S.add("fx", woodblock(1250, .1), t + 8.5, .25)
    tiptoe(S, t + 8.0, beat * .9, g=.7)
    # s03: strain: a slow rising reed + a deflating wah on "magnificence"
    t = T0["s03"]
    S.add("reed", reed(41, 1.0, 900), t + .3, .5)
    for k, m in enumerate((48, 49, 50, 51, 52, 53)):
        S.add("reed", reed(m, .55, 1000), t + 2.0 + k * .45, .45)
    S.add("brass", brass([69, 72, 77], .9, cutoff=2200), t + 6.3, .45)          # "magnificence" pomp chord
    # s04: the march across the empty road, a stab on "Halt, traffic!", ta-da on arrival
    t = T0["s04"]
    S.add("perc", roll(1.2, g0=.1, g1=.6), t + .2, .4)
    S.add("brass", brass([60, 64, 67, 70], .5), t + 2.6, .7)                     # "Halt, traffic!" stab
    march(S, t + 4.2, 9, beat, g=.75)
    tada(S, t + 12.6, .8)
    # s05: it didn't work: wah-wah and a puzzled vibes noodle
    t = T0["s05"]
    wahwah(S, t + 3.3, 1.0)
    walk(S, t, t + 3.2, beat, [48, 46, 45, 43], g=.5, brushes=True)
    for k, m in enumerate((72, 71, 69, 68, 67)):
        S.add("keys", vibes(m, .7), t + 5.4 + k * .28, .55)
    # s06: groping tiptoe pizzicato, then the lullaby as he cradles the hydrant
    t = T0["s06"]
    for k in range(10):
        S.add("keys", pizz((53, 52, 53, 55, 53, 52, 50, 48, 50, 52)[k], .25), t + .3 + k * beat * .75, .6)
    S.add("fx", woodblock(700, .12), t + 4.0, .35)                                # clunk: the hydrant comes loose
    S.add("keys", harp_gliss(65, 84, .6), t + 4.4, .45)
    lullaby(S, t + 5.4, beat * 1.15, 5, g=.7)
    # s07: lullaby continues... until the squirt
    t = T0["s07"]
    lullaby(S, t, beat * 1.15, 3, g=.6)
    hit = t + 5.4
    S.add("perc", cymbal(1.6), hit, .7)
    S.add("perc", timp(36, 1.0), hit, .8)
    S.add("fx", slide_whistle(1600, 300, .6, curve=1.3), hit + .05, .45)
    S.add("fx", whoosh(.6, 3000, 800, seed=21), hit, .4)
    for k in range(5):                                                            # dripping pizz
        S.add("keys", pizz(84 - (k % 2) * 3, .2), hit + 1.8 + k * .55, .45)
    # s08: glum descending reed line, then a cheeky climbing scale for Tilly
    t = T0["s08"]
    for k, m in enumerate((57, 55, 53, 52, 50)):
        S.add("reed", reed(m, .9, 900, vib=.012), t + .4 + k * .85, .55)
    for k, m in enumerate((65, 67, 69, 70, 72, 74, 76, 77, 79, 81)):
        S.add("keys", pizz(m, .3), t + 5.0 + k * .42, .55)
        S.add("keys", celesta(m + 12, .3), t + 5.0 + k * .42, .15)
    S.add("keys", celesta(89, 1.2), t + 9.3, .4)                                  # she sits: twinkle
    # s09: the lift: harp gliss up, eye-pop hit, bright brass burst on the reveal
    t = T0["s09"]
    S.add("keys", harp_gliss(53, 84, 1.0), t + 1.0, .8)
    walk(S, t + 3.0, t + TLEN["s09"], beat, [41, 45, 48, 45], g=.55)
    S.add("fx", slide_whistle(500, 1800, .35, curve=.7), t + 2.0, .35)
    S.add("brass", brass([65, 69, 72, 77], .7, cutoff=3200), t + 2.4, .7)
    S.add("perc", cymbal(1.2), t + 2.4, .35)
    for k, m in enumerate((77, 81, 84)):
        S.add("keys", vibes(m, 1.0), t + 5.8 + k * .18, .35)
    # s10: the triumphant march, full, with a ta-da at the far curb
    t = T0["s10"]
    S.add("perc", roll(.8, g0=.2, g1=.8), t + .1, .5)
    march(S, t + .9, 10, beat, g=1.0, snare_g=.6)
    tada(S, t + 10.2, 1.0)
    # s11: the theme, gentle, under the final brag; then an "uh-oh" for the ducklings
    t = T0["s11"]
    walk(S, t, t + 7.6, beat, [41, 45, 48, 50, 46, 50, 53, 50], g=.7)
    S.add("brass", brass([53, 57, 60], 1.2, cutoff=1800), t + 1.0, .4)
    for k, m in enumerate((79, 78, 77, 76)):
        S.add("keys", xylo(m), t + 7.7 + k * .22, .55)
    S.add("keys", xylo(72), t + 8.8, .6)
    end_sting(S, END0)
    return S


def mixdown(S):
    n = int(DUR * SR)
    stems = [("bass", .34, -.15, .12), ("brass", .30, .2, .3), ("perc", .20, .3, .15), ("keys", .26, -.25, .35), ("reed", .12, .1, .3),
             ("fx", .22, 0, .2)]
    dry, send = np.zeros((n, 2)), np.zeros(n)
    for name, g, p, r in stems:
        x = S.tr[name][:n]
        x = x / (np.max(np.abs(x)) + 1e-9) * g
        dry += pan(x, p)
        send += x * r
    mix = dry + reverb(send, 1.2) * .3
    t = tt(n)
    mix *= np.clip(t / .01, 0, 1)[:, None] * np.clip((n / SR - t) / .25, 0, 1)[:, None]
    mix = np.tanh(mix * 1.3) / np.tanh(1.3)
    return (mix * .5 / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)   # peak -6 dBFS: room under dialogue


def render(path):
    a = mixdown(compose())
    sf.write(path, a, SR, subtype="PCM_24")
    return a


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SH.OUT, "music_bed.wav")
    a = render(out)
    print(out, a.shape, len(a) / SR, "s")
