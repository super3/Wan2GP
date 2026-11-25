"""music_bed.wav for 'One Little Smudge': an original 60s spy-lounge cartoon score, cue by cue on the shot list.

The bed is silent under the title and end cards (they carry their own stings); every cue change lands exactly on a cut;
mid-shot hits sit where the action most likely falls and are kept soft-edged, since the assembly ducks the bed under H3's audio."""
import math
import os

import numpy as np
import soundfile as sf

import tsound as S
from sound import bass_note, brass, twang
from cards import chime, vibes

SR = S.SR
BPM = 138
B = 60 / BPM


# ------------------------------------------------------------------------------------------------ instruments
def whistle(m, dur, vib=.012):
    n = int(dur * SR)
    t = S.tt(n)
    f = S.hz(m) * (1 + vib * np.sin(2 * np.pi * 5.8 * t) * np.clip(t / .12, 0, 1))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + .08 * S.bp(S.noise(n, m), 1500, 5000)
    return x * S.adsr(n, .025, .05, .85, .06)


def celesta(m, dur=.9):
    n = int(dur * SR)
    t = S.tt(n)
    f = S.hz(m)
    return (np.sin(2 * np.pi * f * t) + .35 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 12) + .15 * np.sin(2 * np.pi * f * 2 * t)) * \
        np.exp(-t * 4.5) * np.clip(t / .002, 0, 1)


def pizz(m, dur=.35, seed=0):
    x = S.pluck(S.hz(m), dur, seed=seed, damp=.993)
    return S.lp(x, 1800) * np.exp(-S.tt(len(x)) * 6)


def trombone(m0, m1, dur, wah=0.0):
    n = int(dur * SR)
    t = S.tt(n)
    f = S.hz(m0) * (S.hz(m1) / S.hz(m0)) ** (t / dur) * (1 + .02 * np.sin(2 * np.pi * 5 * t) * (t / dur))
    x = S.osc("saw", f, n)
    cut = 500 + 1500 * (.5 + .5 * np.sin(2 * np.pi * wah * t)) if wah else 1400
    y = np.zeros(n)
    for s0 in range(0, n, 1024):
        c = float(cut[s0]) if wah else cut
        seg = x[max(0, s0 - 1024):s0 + 1024]
        yy = S.lp(seg, c)[-min(1024, n - s0):]
        y[s0:s0 + len(yy)] = yy
    return y * S.adsr(n, .04, .1, .85, .12)


def tuba(m, dur):
    n = int(dur * SR)
    return S.lp(S.osc("saw", S.hz(m), n) + .5 * np.sin(2 * np.pi * S.hz(m) * S.tt(n)), 500) * S.adsr(n, .03, .15, .6, .15)


def organ(notes, dur, leslie=6.0):
    n = int(dur * SR)
    t = S.tt(n)
    out = np.zeros(n)
    for m in notes:
        f = S.hz(m)
        for h, g in ((1, 1), (2, .5), (3, .25), (4, .2)):
            out += np.sin(2 * np.pi * f * h * t) * g
    return out * (1 + .25 * np.sin(2 * np.pi * leslie * t)) * S.adsr(n, .01, .05, .9, .05)


def strings_trem(notes, dur, rate=12):
    n = int(dur * SR)
    t = S.tt(n)
    out = np.zeros(n)
    for m in notes:
        f = S.hz(m)
        out += S.osc("saw", f * 1.002, n) + S.osc("saw", f * .998, n)
    out = S.lp(out, 2200)
    return out * (.55 + .45 * np.abs(np.sin(np.pi * rate * t))) * S.adsr(n, .15, .1, .9, .2)


def swell(dur, seed=1):
    n = int(dur * SR)
    return S.hp(S.noise(n, seed), 4000) * (S.tt(n) / dur) ** 2.5


def crash(dur=1.6, seed=2):
    n = int(dur * SR)
    return S.hp(S.noise(n, seed), 3000) * np.exp(-S.tt(n) * 2.4)


def scratch(dur=.35):
    n = int(dur * SR)
    t = S.tt(n)
    f = 2500 * (1 - t / dur) ** 2 + 300
    y = np.zeros(n)
    x = S.noise(n, 5)
    for s0 in range(0, n, 512):
        seg = x[max(0, s0 - 1024):s0 + 512]
        yy = S.bp(seg, f[s0] * .7, f[s0] * 1.3)[-min(512, n - s0):]
        y[s0:s0 + len(yy)] = yy
    return y * np.clip((dur - t) / .05, 0, 1)


def slide(f0, f1, dur):
    n = int(dur * SR)
    return S.slide("sine", f0, f1, dur) * S.adsr(n, .01, .05, .9, .05)


def brushes(buf, t0, t1, g=.5):
    t = t0
    j = 0
    while t < t1:
        S.put(buf, S.hat(.05, seed=j), t, .25 * g if j % 2 else .4 * g)
        S.put(buf, S.hat(.04, seed=j + 99), t + B * .66, .18 * g)
        if j % 2:
            S.put(buf, S.snare(.3, seed=8, brush=True), t, .55 * g)
        t += B
        j += 1


def walk(buf, t0, notes, g=.8, step=B):
    for k, m in enumerate(notes):
        S.put(buf, bass_note(m, step * .95), t0 + k * step, g)


# ------------------------------------------------------------------------------------------------ the theme
THEME = [(0, .5, 72), (.5, .5, 74), (1, 1, 77), (2, .5, 76), (2.5, .5, 74), (3, 1, 72), (4, .5, 69), (4.5, .5, 70), (5, .5, 72),
         (5.5, .5, 74), (6, 1.5, 72), (8, .5, 77), (8.5, .5, 76), (9, .5, 74), (9.5, .5, 72), (10, 1.5, 77)]
THEME_B = [(0, .5, 74), (.5, .5, 72), (1, 1, 69), (2, .5, 70), (2.5, .5, 72), (3, 1, 74), (4, .5, 76), (4.5, .5, 77), (5, .5, 79),
           (5.5, .5, 77), (6, 2, 76), (8, .5, 72), (8.5, .5, 74), (9, .5, 76), (9.5, .5, 79), (10, 2, 77)]
CHORDS = [(0, 4, (53, 57, 60, 64)), (4, 2, (58, 62, 65, 69)), (6, 2, (48, 55, 58, 64)), (8, 4, (53, 57, 60, 64))]
WALK = [41, 45, 48, 50, 46, 50, 48, 45, 41, 45, 48, 51]


def theme(st, t0, beats=12, mel=THEME, lead_g=.8, full=True):
    for b, d, m in mel:
        if b < beats:
            S.put(st["lead"], whistle(m, d * B * .95), t0 + b * B, lead_g)
    for b, d, ch in CHORDS:
        if b < beats:
            S.put(st["keys"], vibes(ch, d * B + .4), t0 + b * B, .5)
    walk(st["bass"], t0, WALK[:int(beats)], .8)
    if full:
        brushes(st["kit"], t0, t0 + beats * B, .8)


# ------------------------------------------------------------------------------------------------ the cues
def score(tl, total):
    st = {k: S.buf(total) for k in ("lead", "keys", "bass", "kit", "brass", "fx", "strings")}
    T = {name: (f0 / 24, n / 24) for name, f0, n in tl}

    # s01: the theme, bright and full
    t0, d = T["s01"]
    theme(st, t0 + .05, 11.5)
    # s02: the theme's B half, softer under his line; a little 'ta-da' as he declares victory
    t0, d = T["s02"]
    theme(st, t0, 12, THEME_B, lead_g=.45)
    walk(st["bass"], t0 + 12 * B, [53, 57], .7)
    S.put(st["keys"], vibes((65, 69, 72, 77, 81), 2.6), t0 + d - 2.3, .7)
    S.put(st["fx"], chime(2794), t0 + d - 2.25, .35)
    # s03: happy chord rings out... then the double-take sting and a creeping spy ostinato
    t0, d = T["s03"]
    tn = t0 + 3.1
    for k, (m, g) in enumerate(((40, .9), (40, .9), (39, 1.0))):
        S.put(st["brass"], tuba(m, .3 if k < 2 else 1.1), tn + k * .32, g)
    brass((52, 55, 58), tn + .64, .9, st["brass"], .5)
    ost = [40, 40, 43, 40, 46, 45, 43, 40]
    tt = tn + 1.4
    k = 0
    while tt < t0 + d - .2:
        S.put(st["bass"], pizz(ost[k % 8] + 12, .3, seed=k), tt, .7)
        tt += B * .5
        k += 1
    S.put(st["strings"], strings_trem((64, 67, 70), d - 4.6, rate=10), tn + 1.4, .35)
    # s04: surf-spy action riff, two brass stabs for the swipes, the music cuts out, sad trombone
    t0, d = T["s04"]
    riff = [52, 55, 57, 58, 59, 58, 57, 55]
    for k in range(int(7.0 / (B * .5))):
        S.put(st["lead"], twang(riff[k % 8], .3, seed=k), t0 + .2 + k * B * .5, .5)
        if k % 2 == 0:
            S.put(st["bass"], bass_note(40, B * .45), t0 + .2 + k * B * .5, .7)
    for k in range(16):
        S.put(st["kit"], S.tom(170 + 40 * (k % 3), .2), t0 + .2 + k * B * .5, .35)
    brass((52, 59, 64, 67), t0 + 2.2, .5, st["brass"], .8)
    brass((53, 60, 65, 68), t0 + 4.6, .5, st["brass"], .9)
    S.put(st["fx"], crash(1.4), t0 + 4.6, .25)
    tw = t0 + 7.6
    for k, (m0, m1, dd) in enumerate(((62, 61, .45), (61, 60, .45), (60, 59, .45), (59, 57, 1.5))):
        S.put(st["brass"], trombone(m0 - 12, m1 - 12, dd, wah=4 if k == 3 else 0), tw + k * .5, .9)
    # s05: low chromatic creep; the boil (rising strings, swell, a kettle-whistle glide)
    t0, d = T["s05"]
    tt, k = t0 + .3, 0
    while tt < t0 + d - .4:
        S.put(st["bass"], bass_note(40 + (k % 2), B * .9), tt, .7)
        tt += B
        k += 1
    S.put(st["strings"], strings_trem((52, 55), 5.0, rate=8), t0 + .5, .25)
    for k in range(10):
        S.put(st["strings"], strings_trem((64 + k, 67 + k), .55, rate=14), t0 + 6.0 + k * .55, .22 + .03 * k)
    S.put(st["fx"], swell(3.2), t0 + d - 3.4, .5)
    S.put(st["fx"], slide(900, 2600, 1.5), t0 + d - 1.7, .25)
    # s06: heroic fanfare, timpani roll, SPLASH (slide whistle down), then soggy tuba bwomps
    t0, d = T["s06"]
    for k, (m, dd) in enumerate(((60, .2), (60, .2), (65, .2), (69, .6))):
        brass((m - 12, m - 5, m), t0 + .7 + k * .22, dd, st["brass"], .7)
    for k in range(22):
        S.put(st["kit"], S.tom(80, .4), t0 + 2.4 + k * .1, .2 + .5 * k / 22)
    S.put(st["fx"], crash(2.0, 9), t0 + 4.8, .6)
    brass((41, 48, 53), t0 + 4.8, .7, st["brass"], .7)
    S.put(st["fx"], slide(1800, 300, .9), t0 + 4.9, .35)
    S.put(st["brass"], tuba(33, .9), t0 + 7.2, .9)
    S.put(st["brass"], tuba(32, 1.4), t0 + 8.4, .9)
    for k, m in enumerate((84, 79, 76, 72)):
        S.put(st["keys"], celesta(m, .6), t0 + 9.8 + k * .45, .35)
    # s07: the gadget: organ + accelerating motor ostinato + surf twang, then dizzy wobbly vibes
    t0, d = T["s07"]
    S.put(st["keys"], organ((52, 59, 64), 2.0), t0 + .3, .25)
    tt, k = t0 + 2.4, 0
    while tt < t0 + 8.6:
        sp = max(.07, .22 - .02 * k / 3)
        S.put(st["lead"], twang(52 + (k % 4) * 3 + k // 8, sp * 1.2, seed=k), tt, .4)
        S.put(st["kit"], S.hat(.04, seed=k), tt, .3)
        tt += sp
        k += 1
    S.put(st["fx"], S.whoosh(2.4, 300, 3000, seed=5), t0 + 6.4, .6)
    for k in range(6):
        n = int(.9 * SR)
        x = vibes((76 - k * 2,), .9)
        x = x * (1 + .5 * np.sin(2 * np.pi * 3 * S.tt(len(x))))
        S.put(st["keys"], x, t0 + 8.8 + k * .5, .4)
    # s08: tea-time minuet in 3/4 (celesta + pizzicato), a sparkle glissando, then a record scratch and silence
    t0, d = T["s08"]
    mb = 60 / 104
    minuet = [76, 79, 77, 76, 74, 72, 74, 76, 74, 72, 71, 72, 74, 72, 71, 69, 71, 72]
    for k, m in enumerate(minuet):
        tt = t0 + .4 + k * mb
        if tt > t0 + 9.4:
            break
        S.put(st["keys"], celesta(m, .8), tt, .45)
        if k % 3 == 0:
            S.put(st["bass"], pizz(48 if (k // 3) % 2 == 0 else 43, .4, seed=k), tt, .6)
        else:
            S.put(st["bass"], pizz(55 if (k // 3) % 2 == 0 else 50, .3, seed=k + 50), tt, .3)
    for k, m in enumerate((72, 76, 79, 84, 88, 91)):
        S.put(st["keys"], celesta(m, .7), t0 + 9.0 + k * .06, .3)
    S.put(st["fx"], scratch(.4), t0 + 10.6, .45)
    # s09: awkward: plunk... plunk... and a sheepish muted 'wah-wah'
    t0, d = T["s09"]
    for k, tt in enumerate((1.4, 3.2, 4.6)):
        S.put(st["bass"], pizz(36 + (0, -1, -2)[k], .6, seed=k + 7), t0 + tt, .8)
    S.put(st["brass"], trombone(55, 54, .5, wah=3), t0 + 6.6, .5)
    S.put(st["brass"], trombone(54, 52, .9, wah=3), t0 + 7.2, .5)
    # s10: minuet reprise; the face-squish 'bloop'; an 'uh-oh'; the theme returns softly and resolves into the end card
    t0, d = T["s10"]
    for k, m in enumerate(minuet[:9]):
        S.put(st["keys"], celesta(m, .8), t0 + .4 + k * mb, .35)
        if k % 3 == 0:
            S.put(st["bass"], pizz(48 if (k // 3) % 2 == 0 else 43, .4, seed=k), t0 + .4 + k * mb, .5)
    S.put(st["fx"], slide(500, 160, .35), t0 + 4.3, .35)
    S.put(st["fx"], slide(200, 700, .18), t0 + 4.75, .3)
    for k, m in enumerate((47, 43)):
        S.put(st["brass"], tuba(m + 12, .45 if k == 0 else .8), t0 + 6.4 + k * .5, .55)
    theme(st, t0 + 8.6, 11, THEME, lead_g=.5, full=False)
    S.put(st["keys"], vibes((53, 57, 60, 64, 69), 2.0), t0 + d - 2.2, .45)
    S.put(st["bass"], bass_note(41, 1.5), t0 + d - 2.2, .7)
    return st


def make(out, timeline_total):
    tl, total_frames = timeline_total
    total = total_frames / 24
    st = score(tl, total)
    stems = [(st["lead"], .32, -.2, .3), (st["keys"], .26, .25, .4), (st["bass"], .34, -.05, .1), (st["kit"], .16, .15, .2),
             (st["brass"], .30, .05, .3), (st["fx"], .22, .1, .3), (st["strings"], .14, -.3, .4)]
    mix = S.master(stems, wet=.2, rev_s=1.5, drive=1.15, fade_out=.05)
    n = len(mix)
    t = np.arange(n) / SR
    title_end = tl[0][2] / 24
    end_start = tl[-1][1] / 24
    gate = np.clip((t - (title_end - .05)) / .05, 0, 1) * np.clip((end_start - t) / .12, 0, 1)
    mix = mix * gate[:, None]
    mix = mix * (10 ** (-3 / 20)) / (np.max(np.abs(mix)) + 1e-9)
    sf.write(os.path.join(out, "music_bed.wav"), mix.astype(np.float32), SR, subtype="PCM_24")
    return mix
