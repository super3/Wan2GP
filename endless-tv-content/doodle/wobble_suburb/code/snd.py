"""Sound for wobble_suburb: a synthesised jump-blues / boogie-woogie combo (barrelhouse piano, upright bass, brushes, muted
trumpet), a cartoon kid "voice" (formant gibberish), and slapstick SFX.  All maths, no samples."""
import math

import numpy as np

import tsound as S
from tsound import SR

BPM = 150
BEAT = 60 / BPM


# ------------------------------------------------------------------------------------------------ instruments
def piano(m, dur, vel=1.0, bright=1.0, seed=0):
    """Honky-tonk upright: two slightly detuned strings, inharmonic partials with their own decays, a hammer knock."""
    n = int(dur * SR)
    t = S.tt(n)
    f0 = S.hz(m)
    out = np.zeros(n)
    for det in (-.0045, .0045):
        f = f0 * (1 + det)
        for h in range(1, 9):
            fh = f * h * math.sqrt(1 + .0004 * h * h)
            if fh > 9000:
                break
            amp = (1 / h ** (1.25 - .35 * bright)) * (1 if h % 2 else .8)
            out += amp * np.sin(2 * np.pi * fh * t + h) * np.exp(-t * (1.6 + h * .9) * (1 + max(0, m - 60) / 40))
    knock = S.lp(S.noise(min(n, int(.03 * SR)), seed + m), 2500) * np.exp(-S.tt(min(n, int(.03 * SR))) * 120)
    out[:len(knock)] += knock * .3
    rel = np.clip((dur - t) / .05, 0, 1)
    return out * rel * vel * np.clip(t / .002, 0, 1)


def ubass(m, dur, vel=1.0, seed=1):
    f = S.hz(m)
    x = S.pluck(f, dur, seed=seed, damp=.993, bright=.5)
    x = S.lp(x, 900) + .5 * np.sin(2 * np.pi * f * S.tt(len(x))) * np.exp(-S.tt(len(x)) * 4)
    thump = S.lp(S.noise(int(.02 * SR), seed + 9), 400) * np.exp(-S.tt(int(.02 * SR)) * 150)
    x[:len(thump)] += thump * .6
    return x * vel * np.clip((dur - S.tt(len(x))) / .04, 0, 1)


def brush_swish(dur=.22, seed=3):
    return S.snare(dur, seed, brush=True)


def ride(seed=5):
    n = int(.25 * SR)
    t = S.tt(n)
    x = S.hp(S.noise(n, seed), 5000) * np.exp(-t * 18)
    ring = sum(np.sin(2 * np.pi * f * t) for f in (3150, 4270, 5490)) * np.exp(-t * 14) * .08
    return x * .6 + ring


def trumpet(m, dur, wah=1.0, vel=1.0, bend=0.0, vib=.012):
    """Harmon-muted trumpet: saw through a moving formant; wah sweeps the formant open-closed."""
    n = int(dur * SR)
    t = S.tt(n)
    f = S.hz(m) * (1 + bend * np.clip(1 - t / .08, 0, 1)) * (1 + vib * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / .2, 0, 1))
    x = S.osc("saw", f, n)
    y = np.zeros(n)
    blk = 480
    for s0 in range(0, n, blk):
        u = s0 / max(1, n)
        fc = 700 + 1500 * (math.sin(math.pi * u) ** 1.5 if wah else .6)
        seg = x[max(0, s0 - 2400):s0 + blk]
        y[s0:s0 + blk] = S.bp(seg, fc * .55, fc * 1.5)[-min(blk, n - s0):]
    return y * S.adsr(n, .025, .06, .85, .06) * vel


def swing8(beat_i, half):
    """Time of a swung eighth (half=0 on the beat, 1 the 'and')."""
    return beat_i * BEAT + (BEAT * 2 / 3 if half else 0)


# ------------------------------------------------------------------------------------------------ voice
FORM = {"a": (820, 1250, 2600), "e": (520, 1900, 2600), "i": (330, 2300, 3000), "o": (520, 880, 2500), "u": (360, 800, 2400),
        "ah": (900, 1350, 2700), "uh": (620, 1180, 2500)}


def vowel(v, f0, dur, rasp=.25, seed=0, f0_end=None, vib=.03):
    """A cartoon kid vowel: glottal buzz through three formants, a little nasal and scratchy."""
    n = int(dur * SR)
    t = S.tt(n)
    f1 = f0 if f0_end is None else f0_end
    f = np.linspace(f0, f1, n) * (1 + vib * np.sin(2 * np.pi * 6.5 * t) * np.clip(t / .15, 0, 1))
    src = S.osc("saw", f, n) + rasp * S.noise(n, seed) * (1 + S.osc("square", f, n)) * .5
    F = FORM[v]
    y = S.bp(src, F[0] * .8, F[0] * 1.25) * 1.0 + S.bp(src, F[1] * .85, F[1] * 1.18) * .7 + S.bp(src, F[2] * .9, F[2] * 1.1) * .35
    y += S.bp(src, 230, 300) * .3  # nasal hum
    return y * S.adsr(n, .015, .04, .9, .04)


def consonant(kind, seed=0):
    n = int(.045 * SR)
    t = S.tt(n)
    if kind in ("b", "d", "g"):
        return S.lp(S.noise(n, seed), 900) * np.exp(-t * 90) * .8
    if kind in ("s", "sh"):
        return S.bp(S.noise(int(.08 * SR), seed), 3500 if kind == "s" else 2000, 8000) * .5
    return S.bp(S.noise(n, seed), 1500, 5000) * np.exp(-t * 70) * .6  # t/k/p


def gibber(syls, f0=330, seed=0, rasp=.25):
    """syls: list of (consonant or '', vowel, duration, pitch factor).  Returns a mono buffer."""
    out = []
    for k, (c, v, d, pf) in enumerate(syls):
        seg = np.zeros(int(d * SR))
        if c:
            cs = consonant(c, seed + k)
            seg[:len(cs)] += cs[:len(seg)]
        vs = vowel(v, f0 * pf, d * .82, rasp, seed + k, f0_end=f0 * pf * .94)
        o = int(.03 * SR) if c else 0
        seg[o:o + len(vs)] += vs[:len(seg) - o]
        out.append(seg)
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ sfx
def boing(f=180, dur=.6, depth=.5):
    n = int(dur * SR)
    t = S.tt(n)
    ff = f * (1 + depth * np.exp(-t * 7)) * (1 + .08 * np.sin(2 * np.pi * 16 * t))
    return np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-t * 4.5)


def slide_whistle(f0, f1, dur, curve=.8):
    n = int(dur * SR)
    x = S.slide("sine", f0, f1, dur, curve) + .15 * S.slide("sine", f0 * 2, f1 * 2, dur, curve)
    return x * S.adsr(n, .02, .05, .9, .05)


def gulp():
    n = int(.28 * SR)
    t = S.tt(n)
    f = 420 * np.exp(-t * 9) + 90
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-((t - .08) / .07) ** 2)
    return x + .3 * S.lp(S.noise(n, 77), 600) * np.exp(-((t - .08) / .04) ** 2)


def chirp(seed=0):
    r = np.random.default_rng(seed)
    out = []
    for k in range(r.integers(2, 4)):
        n = int(r.uniform(.05, .09) * SR)
        t = S.tt(n)
        f = r.uniform(3200, 4200) + r.uniform(-900, 900) * t / t[-1]
        out.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) ** 2)
        out.append(np.zeros(int(r.uniform(.03, .07) * SR)))
    return np.concatenate(out)


def breeze(dur, seed=4):
    n = int(dur * SR)
    x = S.bp(S.noise(n, seed), 300, 1600)
    return x * (.6 + .4 * np.sin(2 * np.pi * .23 * S.tt(n) + 1)) * .4


def pop_sfx():
    n = int(.08 * SR)
    t = S.tt(n)
    return np.sin(2 * np.pi * np.cumsum(900 * np.exp(-t * 40) + 200) / SR) * np.exp(-t * 50)


def thud(f=70):
    k = S.kick(.4, f * 1.6, f)
    n = int(.2 * SR)
    k[:n] += .3 * S.lp(S.noise(n, 13), 400) * np.exp(-S.tt(n) * 25)
    return k


def crash_hit(dur=1.2, seed=21):
    n = int(dur * SR)
    t = S.tt(n)
    return S.hp(S.noise(n, seed), 3000) * np.exp(-t * 3.5) * .7


# ------------------------------------------------------------------------------------------------ boogie generator
LH = [0, 4, 7, 9, 10, 9, 7, 4]  # boogie walking figure in eighths (root, 3, 5, 6, b7, 6, 5, 3)
BLUES12 = [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7]


def boogie(t0, bars, root=48, stems=None, density=1.0, seed=0, rh=True, prog=None):
    """Write a swung boogie-woogie combo into the stem dict from time t0 for `bars` bars of 4/4."""
    prog = prog or BLUES12
    r = np.random.default_rng(seed)
    for b in range(bars):
        ch = root + prog[b % len(prog)]
        for beat in range(4):
            for half in (0, 1):
                k = beat * 2 + half
                tt = t0 + b * 4 * BEAT + swing8(beat, half)
                S.put(stems["piano"], piano(ch - 12 + LH[k], BEAT * .62, .55), tt)
            tb = t0 + b * 4 * BEAT + beat * BEAT
            S.put(stems["bass"], ubass(ch - 12 + (0, 7, 9, 7)[beat] - 12 * 0, BEAT * .95, .9, seed=b * 4 + beat), tb)
            if beat in (1, 3):
                S.put(stems["drums"], brush_swish(.2, b * 4 + beat), tb, .55)
            S.put(stems["drums"], ride(beat + 7), tb, .22)
            S.put(stems["drums"], ride(beat + 11), tb + BEAT * 2 / 3, .14)
        if rh and r.uniform() < density:
            # right hand: blues stabs and a tremolo now and then
            if b % 2 == 0:
                for kk, beat in enumerate((0, 1.667, 2.667)):
                    for m in (ch + 12 + 4, ch + 12 + 7, ch + 12 + 10):
                        S.put(stems["piano"], piano(m, BEAT * .5, .32, 1.2), t0 + b * 4 * BEAT + beat * BEAT)
            else:
                lick = [ch + 24, ch + 22, ch + 19, ch + 18, ch + 17, ch + 15, ch + 12]
                for kk, m in enumerate(lick):
                    S.put(stems["piano"], piano(m, BEAT * .45, .3, 1.2), t0 + b * 4 * BEAT + swing8(kk // 2, kk % 2))


def new_stems(dur):
    return {k: S.buf(dur) for k in ("piano", "bass", "drums", "horn", "voice", "fx", "amb")}


def mixdown(st, wet=.16):
    return S.master([(st["piano"], .30, -.15, .12), (st["bass"], .34, .05, .04), (st["drums"], .16, .2, .1),
                     (st["horn"], .24, .25, .2), (st["voice"], .34, 0, .08), (st["fx"], .30, 0, .12), (st["amb"], .05, .3, .2)],
                    wet=wet, rev_s=1.1, drive=1.25)


# ------------------------------------------------------------------------------------------------ the 5 s code shot
def sound_code_shot(dur=5.0):
    st = new_stems(dur)
    # music: boogie up to the boing, then a tag after the gulp
    tmp = new_stems(dur)
    boogie(0.0, 2, root=48, stems=tmp, seed=3)
    cut = int(2.9 * SR)
    for k in ("piano", "bass", "drums"):
        tmp[k][cut:] *= np.exp(-np.arange(len(tmp[k]) - cut) / (SR * .03))
        st[k] += tmp[k]
    # tag: descending piano lick and a two-note bass button
    for kk, m in enumerate((79, 76, 74, 72, 70, 67)):
        S.put(st["piano"], piano(m, .2, .45, 1.2), 4.12 + kk * .085)
    S.put(st["bass"], ubass(43, .3, 1.0), 4.66)
    S.put(st["bass"], ubass(36, .5, 1.1), 4.82)
    S.put(st["piano"], piano(48, .5, .5) + piano(55, .5, .4) + piano(64, .5, .35), 4.82)
    # voice: jabbering challenge, a yelp on the take
    syl = [("d", "i", .125, 1.1), ("s", "a", .125, 1.0), ("l", "o", .125, 1.15), ("", "a", .125, 1.0), ("b", "e", .125, 1.2),
           ("g", "a", .125, 1.1), ("", "o", .125, .95), ("t", "u", .125, 1.05), ("b", "a", .125, 1.2), ("s", "ah", .2, .9)]
    S.put(st["voice"], gibber(syl, 330, seed=4), 1.25)
    S.put(st["voice"], vowel("ah", 520, .55, .5, 9, f0_end=640, vib=.06), 3.06)
    # sfx
    S.put(st["fx"], boing(200, .7), 2.92, .9)
    S.put(st["fx"], slide_whistle(500, 1600, .16), 3.0, .6)
    S.put(st["horn"], trumpet(70, .45, 1, .9, bend=-.06) + trumpet(74, .45, 1, .6, bend=-.06), 3.06)
    S.put(st["fx"], crash_hit(.6), 3.04, .25)
    S.put(st["fx"], gulp(), 3.8, 1.0)
    S.put(st["horn"], trumpet(64, .25, 1, .5), 3.84)
    S.put(st["horn"], trumpet(62, .3, 1, .5, bend=-.03), 4.0)
    for t0, sd in ((.35, 1), (1.9, 2), (4.45, 3)):
        S.put(st["amb"], chirp(sd), t0, .6)
    st["amb"] += breeze(dur)[:len(st["amb"])] * .5
    return mixdown(st)
