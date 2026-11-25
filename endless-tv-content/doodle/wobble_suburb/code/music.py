"""The score for "One Lousy Dandelion": a synthesised jump-blues / boogie combo cue sheet, one cue per shot, laid on the real
timeline (title card first: the bed is silent there because the card carries its own sting).  music.py OUT.wav"""
import math
import sys

import numpy as np
import soundfile as sf

import shots_def as SD
import snd
import tsound as S
from tsound import SR


def timeline():
    t = SD.TITLE_S
    out = {}
    for s in SD.SHOTS:
        d = s["frames"] / SD.FPS
        out[s["id"]] = (t, t + d)
        t += d
    return out, t + SD.END_S


def chord(st, root, t0, dur, vel=.35, kind="dom7", bright=1.0):
    iv = {"dom7": (0, 4, 7, 10), "maj6": (0, 4, 7, 9), "min": (0, 3, 7), "dim": (0, 3, 6, 9), "maj": (0, 4, 7)}[kind]
    for m in iv:
        S.put(st["piano"], snd.piano(root + m, dur, vel, bright), t0)


def wah_wah(st, t0, notes=(58, 57, 56, 51), vel=.9):
    for k, m in enumerate(notes):
        d = .42 if k < len(notes) - 1 else .9
        S.put(st["horn"], snd.trumpet(m, d, 1, vel, vib=.03 if k == len(notes) - 1 else .012), t0 + k * .42)


def boogie_at(st, t0, t1, root=48, bpm=150, seed=0, vel=1.0, rh=True):
    """Boogie between t0 and t1 at a tempo (re-uses snd.boogie by temporarily retuning BEAT)."""
    old = snd.BEAT
    snd.BEAT = 60 / bpm
    bars = max(1, int((t1 - t0) / (4 * snd.BEAT)))
    tmp = snd.new_stems(t1 + 2)
    snd.boogie(t0, bars, root=root, stems=tmp, seed=seed, rh=rh)
    for k in ("piano", "bass", "drums"):
        n = min(len(st[k]), len(tmp[k]))
        x = tmp[k][:n]
        e = int(t1 * SR)
        if e < n:
            x[e:] *= np.exp(-np.arange(n - e) / (SR * .05))
        st[k][:n] += x * vel
    snd.BEAT = old
    return t0 + bars * 4 * 60 / bpm


def tick(st, t0, t1, period=.5):
    t = t0
    k = 0
    while t < t1:
        n = int(.05 * SR)
        f = 1800 if k % 2 == 0 else 1400
        x = np.sin(2 * np.pi * f * S.tt(n)) * np.exp(-S.tt(n) * 90)
        S.put(st["drums"], x, t, .5)
        t += period
        k += 1


def build():
    T, total = timeline()
    st = snd.new_stems(total + .5)
    r = np.random.default_rng(1)

    # s01: jaunty boogie under the pride, a stop, then a brass stab + slide whistle on the double take
    a, b = T["s01"]
    end = boogie_at(st, a + .3, b - 3.6, root=48, bpm=150, seed=1, vel=.85)
    chord(st, 53, end, 1.2, .25, "maj6")
    S.put(st["fx"], snd.slide_whistle(400, 1500, .25), b - 2.6, .7)
    S.put(st["horn"], snd.trumpet(70, .5, 1, 1, bend=-.05) + snd.trumpet(74, .5, 1, .7, bend=-.05), b - 2.35)
    S.put(st["drums"], snd.crash_hit(1.2), b - 2.35, .7)

    # s02: western standoff: low drone, clip-clop woodblock, lonely muted trumpet, a "dun-dun"
    a, b = T["s02"]
    n = int((b - a - .5) * SR)
    drone = snd.ubass(28, b - a - .5, .6) * 0 + S.lp(S.osc("saw", S.hz(28), n), 200) * np.clip(S.tt(n) / 1.0, 0, 1) * np.clip((b - a - .5 - S.tt(n)) / .5, 0, 1)
    S.put(st["bass"], drone * .22, a + .2)
    for k in range(int((b - a - 1) / .5)):
        n2 = int(.04 * SR)
        S.put(st["drums"], S.bp(S.noise(n2, 50 + k), 900, 2500) * np.exp(-S.tt(n2) * 80), a + .4 + k * .5 + (.12 if k % 2 else 0), .5)
    for t0, m, d in ((a + .8, 64, .5), (a + 1.3, 67, .5), (a + 1.8, 69, 1.4), (a + 4.8, 71, .4), (a + 5.2, 69, .4), (a + 5.6, 64, 1.6)):
        S.put(st["horn"], snd.trumpet(m, d, 0, .6, vib=.03), t0)
    for t0 in (b - 1.6, b - 1.15):
        chord(st, 40, t0, .5, .5, "min", .8)

    # s03: strain build (rising tremolo), then snap: crash, slide whistle down, wah-wah
    a, b = T["s03"]
    for k in range(int(5.5 / .07)):
        m = 60 + int(k * .07 / .5)
        S.put(st["piano"], snd.piano(m + (12 if k % 2 else 0), .1, .12 + .2 * k / 78, 1.2), a + .5 + k * .07)
    S.put(st["drums"], snd.crash_hit(1.4), a + 6.1, .9)
    S.put(st["fx"], snd.slide_whistle(1600, 300, .6, 1.2), a + 6.1, .7)
    wah_wah(st, a + 7.0, vel=.8)

    # s04: Bea: lazy jazz piano tag (ii-V-I), brushes
    a, b = T["s04"]
    for k, (root, kind) in enumerate(((50, "min"), (55, "dom7"), (48, "maj6"))):
        chord(st, root, a + .6 + k * 1.6, 1.5, .22, kind)
        S.put(st["bass"], snd.ubass(root - 24 + 12, 1.4, .6), a + .6 + k * 1.6)
        S.put(st["drums"], snd.brush_swish(.3, 70 + k), a + 1.4 + k * 1.6, .4)
    S.put(st["piano"], snd.piano(84, .5, .25, 1.3), b - 1.2)

    # s05: frantic digging boogie at 184 bpm, then a "plunk ... hmm?" reveal
    a, b = T["s05"]
    end = boogie_at(st, a + 1.6, b - 2.4, root=50, bpm=184, seed=5, vel=.9)
    S.put(st["piano"], snd.piano(43, .6, .6), end + .1)
    S.put(st["horn"], snd.trumpet(67, .35, 1, .6), end + .6)
    S.put(st["horn"], snd.trumpet(70, .6, 1, .6, bend=.04), end + .95)

    # s06: vacuum: ostinato accelerating into the bang, then a crash and a wah-wah
    a, b = T["s06"]
    t = a + 2.6
    per = .36
    k = 0
    while t < a + 7.6:
        S.put(st["bass"], snd.ubass(38 + (0, 3, 5, 6)[k % 4], per * .9, .8, seed=k), t)
        S.put(st["horn"], snd.trumpet(62 + (k % 4), per * .8, 0, .25 + .3 * (t - a) / 7.6), t)
        t += per
        per = max(.12, per * .95)
        k += 1
    S.put(st["drums"], snd.crash_hit(2.0), a + 7.7, 1.0)
    S.put(st["drums"], snd.thud(50), a + 7.7, 1.0)
    chord(st, 36, a + 7.7, 1.5, .5, "dim", .8)
    wah_wah(st, a + 9.2, notes=(60, 59, 58, 53), vel=.7)

    # s07: the clock: woodblock tick-tock and a heartbeat bass, tension stabs
    a, b = T["s07"]
    tick(st, a + .3, b - .4, .5)
    for k in range(int((b - a - .8) / 1.0)):
        S.put(st["bass"], snd.ubass(31, .25, .9), a + .5 + k)
        S.put(st["bass"], snd.ubass(31, .25, .6), a + .75 + k)
    chord(st, 43, b - 2.6, 2.0, .25, "dim", .8)

    # s08: panic boogie at 200 bpm after the honk, then a fanfare on "I got it!"
    a, b = T["s08"]
    end = boogie_at(st, a + 2.6, b - 3.4, root=53, bpm=200, seed=8, vel=.85)
    for k, m in enumerate((60, 64, 67, 72)):
        S.put(st["horn"], snd.trumpet(m, .22 if k < 3 else 1.5, 0, .9), end + .1 + k * .16)
    chord(st, 48, end + .58, 1.6, .4, "maj6")
    S.put(st["drums"], snd.crash_hit(1.5), end + .58, .7)

    # s09: smug swing strut, then the blow: twinkling descending sparkle, and a two-note "uh-oh" at the end
    a, b = T["s09"]
    strut = [(60, .6), (64, .3), (67, .6), (69, .3), (67, .6), (64, .3), (60, .9)]
    tt = a + .4
    for m, d in strut:
        S.put(st["piano"], snd.piano(m + 12, d, .3, 1.2), tt)
        tt += d
    for k in range(5):
        S.put(st["bass"], snd.ubass((36, 40, 43, 45, 43)[k], .55, .6), a + .4 + k * .6)
    for k in range(28):
        m = 96 - k + (3 if k % 2 else 0)
        n2 = int(.4 * SR)
        x = np.sin(2 * np.pi * S.hz(m) * S.tt(n2)) * np.exp(-S.tt(n2) * 8)
        S.put(st["fx"], x, a + 4.3 + k * .09, .25)
    S.put(st["horn"], snd.trumpet(67, .35, 1, .55), b - 1.6)
    S.put(st["horn"], snd.trumpet(64, .8, 1, .55, bend=-.03), b - 1.2)

    # s10: morning pastoral (soft arpeggios and a whistled flute), then the brass "DUN!" and tremolo under the wail
    a, b = T["s10"]
    for k in range(20):
        m = (60, 64, 67, 72, 67, 64)[k % 6] + (5 if (k // 6) % 2 else 0)
        S.put(st["piano"], snd.piano(m, .5, .18, .8), a + .5 + k * .25)
    for t0, m, d in ((a + .8, 79, .6), (a + 1.5, 76, .4), (a + 2.0, 81, .9), (a + 3.1, 79, 1.0)):
        n2 = int(d * SR)
        f = S.hz(m) * (1 + .01 * np.sin(2 * np.pi * 5 * S.tt(n2)))
        x = S.osc("sine", f, n2) * S.adsr(n2, .05, .1, .8, .15) + .03 * S.bp(S.noise(n2, 9), 2000, 6000)
        S.put(st["fx"], x, t0, .35)
    hit = a + 6.0
    S.put(st["horn"], snd.trumpet(55, 1.0, 0, 1) + snd.trumpet(61, 1.0, 0, .8) + snd.trumpet(66, 1.0, 0, .7), hit)
    S.put(st["drums"], snd.crash_hit(2.5), hit, 1.0)
    S.put(st["drums"], snd.thud(45), hit, .9)
    for k in range(int(4.5 / .06)):
        S.put(st["piano"], snd.piano(36 + (6 if k % 2 else 0), .09, .25 * (1 - k / 80), .6), hit + .4 + k * .06)

    # s11: a lazy blues tag that resolves into the end card
    a, b = T["s11"]
    old = snd.BEAT
    snd.BEAT = 60 / 112
    tmp = snd.new_stems(total + 1)
    snd.boogie(a + .3, 3, root=48, stems=tmp, seed=11, rh=False, prog=[0, 5, 0])
    snd.BEAT = old
    for k in ("piano", "bass", "drums"):
        x = tmp[k][:len(st[k])]
        e = int((b - .3) * SR)
        x[e:] *= np.exp(-np.arange(len(x) - e) / (SR * .15))
        st[k] += x * .6
    for k, m in enumerate((76, 74, 72, 69, 67)):
        S.put(st["piano"], snd.piano(m, .3, .25, 1.2), a + 4.2 + k * .2)
    return st, total


def render():
    st, total = build()
    mix = S.master([(st["piano"], .30, -.15, .12), (st["bass"], .34, .05, .04), (st["drums"], .2, .2, .1), (st["horn"], .26, .25, .2),
                    (st["fx"], .2, 0, .15), (st["voice"], 0.0, 0, 0), (st["amb"], 0.0, 0, 0)], wet=.16, rev_s=1.1, drive=1.2)
    mix = mix[:int(total * SR)]
    mix[:int(SD.TITLE_S * SR)] = 0  # the title card carries its own sting
    return mix * (10 ** (-3 / 20)) / (np.max(np.abs(mix)) + 1e-9)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/user/Wan2GP/outputs/shorts/wobble_suburb/music_bed.wav"
    m = render()
    sf.write(out, m.astype(np.float32), SR, subtype="PCM_24")
    print(out, len(m) / SR, "s")
