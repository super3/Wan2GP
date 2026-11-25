"""Sound for the 5 s desert_chase shot: a small 1950s-theatrical-style orchestral cue that hits every action (bassoon strut,
pizzicato, xylophone, brass stab, string tremolo, clarinet run, slide whistle, cymbal), dry Foley (tumbleweed rustle, bounces,
whack, broom swishes, dust puffs) and Pip's nonverbal grunts, all synthesised from maths and fixed seeds."""
import math

import numpy as np

import tsound as S
from anim import EV, FPS

DUR = 5.0
SR = S.SR


def t(f):
    return f / FPS


def env(n, a=.005, d=.1, s=.6, r=.08):
    return S.adsr(n, a, d, s, r)


# ------------------------------------------------------------------------------------------------ instruments
def bassoon(m, dur, bend=0.0):
    n = int(dur * SR)
    f = S.hz(m) * (2 ** (bend * np.linspace(0, 1, n) / 12)) * (1 + .004 * np.sin(2 * np.pi * 5 * S.tt(n)))
    x = .6 * S.osc("saw", f, n) + .4 * S.osc("square", f, n, pw=.3)
    x = S.lp(x, 1500) + .6 * S.bp(x, 380, 620)
    return x * env(n, .012, .08, .7, .05)


def clarinet(m, dur):
    n = int(dur * SR)
    f = S.hz(m) * (1 + .003 * np.sin(2 * np.pi * 5.5 * S.tt(n)))
    x = S.lp(S.osc("square", f, n), 2600)
    return x * env(n, .01, .05, .8, .03)


def pizz(m, dur=.35, seed=0):
    x = S.pluck(S.hz(m), dur, seed=seed, damp=.994)
    return S.lp(x, 3500) * np.exp(-S.tt(len(x)) * 7)


def xylo(m, dur=.4):
    n = int(dur * SR)
    tt = S.tt(n)
    f = S.hz(m)
    x = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9) + .5 * np.sin(2 * np.pi * f * 3.93 * tt) * np.exp(-tt * 22) \
        + .2 * np.sin(2 * np.pi * f * 9.2 * tt) * np.exp(-tt * 40)
    return x * np.clip(tt / .001, 0, 1)


def woodblock(f=900, dur=.12):
    n = int(dur * SR)
    tt = S.tt(n)
    return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 45) + .3 * S.bp(S.noise(n, 3), 1500, 5000) * np.exp(-tt * 120)


def brass(ms, dur, cutoff=2400):
    n = int(dur * SR)
    x = np.zeros(n)
    for k, m in enumerate(ms):
        for dt in (-.004, .0, .005):
            x += S.osc("saw", S.hz(m) * (1 + dt), n, phase=.13 * k)
    tt = S.tt(n)
    fc = 600 + cutoff * np.exp(-tt * 5)
    y = np.zeros(n)
    blk = 512
    for s in range(0, n, blk):
        seg = x[max(0, s - 2048):s + blk]
        y[s:s + blk] = S.lp(seg, fc[s])[-min(blk, n - s):]
    return y * env(n, .01, .12, .55, .1)


def strings_trem(ms, dur, rate=13.0):
    n = int(dur * SR)
    tt = S.tt(n)
    x = np.zeros(n)
    for k, m in enumerate(ms):
        for dt in (-.006, .003, .008):
            x += S.osc("saw", S.hz(m) * (1 + dt) * (1 + .004 * np.sin(2 * np.pi * 5.3 * tt + k)), n, phase=.21 * k)
    x = S.lp(x, 3200)
    trem = .55 + .45 * np.abs(np.sin(np.pi * rate * tt))
    return x * trem * np.clip(tt / .05, 0, 1) * np.clip((dur - tt) / .06, 0, 1) * (.5 + .5 * tt / dur)


def cymbal(dur=1.2, seed=8):
    n = int(dur * SR)
    return S.hp(S.noise(n, seed), 5000) * np.exp(-S.tt(n) * 4)


def slide_whistle(m0, m1, dur, curve=1.0):
    n = int(dur * SR)
    x = np.linspace(0, 1, n) ** curve
    f = S.hz(m0) * (2 ** ((m1 - m0) * x / 12)) * (1 + .01 * np.sin(2 * np.pi * 6 * S.tt(n)))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) + .08 * S.bp(S.noise(n, 11), 1500, 6000)
    return y * env(n, .02, .05, .9, .05)


def rustle(dur, seed=21, density=260):
    """Dry tumbleweed crackle: sparse filtered impulses."""
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    idx = rng.integers(0, n, int(density * dur))
    x[idx] = rng.uniform(-1, 1, len(idx))
    x = S.bp(x, 1800, 7500) + .25 * S.bp(S.noise(n, seed + 1), 2500, 6000)
    return x


def puff(dur=.3, seed=31, f=900):
    n = int(dur * SR)
    tt = S.tt(n)
    return S.lp(S.noise(n, seed), f) * (tt / .02).clip(0, 1) * np.exp(-tt * 10)


def swish(dur=.16, seed=41, up=True):
    return S.whoosh(dur, 1500 if up else 5000, 5000 if up else 1500, seed=seed)


def ratchet(dur, r0=10, r1=40, seed=51):
    n = int(dur * SR)
    x = np.zeros(n)
    tcur = 0.0
    k = 0
    while tcur < dur:
        rate = r0 + (r1 - r0) * (tcur / dur)
        i = int(tcur * SR)
        c = S.bp(S.noise(400, seed + k), 2000, 6000) * np.exp(-np.arange(400) / 40)
        x[i:i + 400] += c[:max(0, min(400, n - i))]
        tcur += 1 / rate
        k += 1
    return x


def voice(contour, dur, vowel="hm", seed=61, growl=0.0):
    """A nonverbal cartoon grunt: a jittery glottal pulse through vowel formants. contour: list of (time 0..1, midi)."""
    n = int(dur * SR)
    tt = np.linspace(0, 1, n)
    cs = np.array(contour, float)
    m = np.interp(tt, cs[:, 0], cs[:, 1])
    rng = np.random.default_rng(seed)
    jit = 1 + .006 * np.interp(np.arange(n), np.linspace(0, n, 40), rng.normal(0, 1, 40))
    f = S.hz(m) * jit * (1 + growl * .05 * np.sin(2 * np.pi * 31 * S.tt(n)))
    src = S.osc("saw", f, n) * (1 + growl * .4 * np.sin(2 * np.pi * 37 * S.tt(n)))
    forms = {"hm": ((260, 1.0, 120), (2100, .25, 300)), "ah": ((750, 1.0, 140), (1250, .7, 160), (2600, .35, 300)),
             "uh": ((560, 1.0, 120), (1100, .6, 160), (2500, .25, 300)), "oo": ((330, 1.0, 100), (900, .45, 140), (2300, .15, 300)),
             "eh": ((560, 1.0, 120), (1800, .6, 200), (2600, .3, 300))}[vowel]
    y = np.zeros(n)
    for fc, g, bw in forms:
        y += g * S.bp(src, fc - bw, fc + bw)
    if vowel == "hm":
        y = S.lp(y, 1400) + .4 * S.lp(src, 400)
    return y * env(n, .025, .06, .85, .07)


# ------------------------------------------------------------------------------------------------ score
def sound():
    stems = {k: S.buf(DUR) for k in ("wind", "bass", "pizz", "xylo", "brass", "str", "perc", "cym", "whistle", "foley", "voice")}
    put = lambda k, x, tt, g=1.0: S.put(stems[k], x, tt, g)  # noqa: E731

    # 1. the strut (proud hold): a pompous little bassoon figure over pizzicato bass, 144 bpm
    e = 60 / 144 / 2
    for k, (m, d) in enumerate(((53, 1), (57, 1), (60, 1), (57, 1), (65, 2))):
        put("wind", bassoon(m, d * e * .85), .04 + sum((1, 1, 1, 1, 2)[:k]) * e, .9)
    for k, m in enumerate((41, 48, 41)):
        put("bass", pizz(m - 0, .45, seed=k), .04 + k * 2 * e, 1.0)
    # 2. the tumbleweed: dry rustle while it rolls, a descending xylophone note per bounce (mickey-moused), settle puff
    put("foley", rustle(t(EV["settle"]) - t(6) + .1, 21, 220) * np.linspace(.3, 1, int((t(EV["settle"]) - t(6) + .1) * SR)), t(6), .55)
    for k, (fb, m) in enumerate(zip(EV["bounce"], (84, 81, 77))):
        put("xylo", xylo(m, .35), t(fb), .8 - .1 * k)
        put("perc", woodblock(700 - 80 * k, .1), t(fb), .5)
        put("foley", puff(.2, 30 + k, 1200), t(fb), .35)
    put("xylo", xylo(96, .2), t(18), .35)            # ear twitch: a tiny tink
    put("pizz", pizz(72, .3, 5), t(EV["swivel"]), .8)  # eye swivel: plink-plonk
    put("pizz", pizz(79, .3, 6), t(EV["swivel"] + 2), .8)
    put("foley", puff(.35, 33, 800), t(EV["settle"]), .6)
    # 3. squash + TAKE: a low bassoon blurp, then a brass stab, cymbal, slide whistle up and a startled "Hm?!"
    put("wind", bassoon(43, .14), t(EV["squash"]), .9)
    put("brass", brass((62, 66, 69, 75), .5), t(EV["take"]), 1.0)
    put("cym", cymbal(1.0, 8), t(EV["take"]), .7)
    put("perc", S.snare(.2, 3), t(EV["take"]), .6)
    put("whistle", slide_whistle(67, 84, .22, .7), t(EV["take"]), .6)
    put("voice", voice([(0, 62), (.5, 66), (1, 74)], .3, "hm", 61), t(EV["take"]) + .02, 1.0)
    put("str", strings_trem((63, 66, 69, 72), t(48) - t(38) + .05), t(38), .55)  # vibrating hold: string tremolo
    # 4. whip-turn, land, grumble
    put("foley", swish(.14, 41, True), t(EV["turn1"]), .8)
    put("perc", S.tom(150, .25), t(EV["land1"]), .7)
    put("foley", puff(.25, 34, 700), t(EV["land1"]), .5)
    put("wind", bassoon(38, .22, bend=-1), t(52), .9)
    put("wind", bassoon(37, .3, bend=-1), t(52) + .24, .9)
    put("voice", voice([(0, 50), (.6, 48), (1, 46)], .42, "hm", 62, growl=1.0), t(52), .8)
    # 5. wind-up: ratchet and a rising chromatic clarinet run, then a held trill (anticipation hold)
    put("perc", ratchet(t(62) - t(EV["windup"]), 8, 30), t(EV["windup"]), .5)
    for k in range(10):
        put("wind", clarinet(62 + k, .06), t(EV["windup"]) + k * .025 * 1.0 + .02, .55)
    for k in range(int((t(66) - t(62)) / .05)):
        put("wind", clarinet(73 if k % 2 else 75, .05), t(62) + k * .05, .5)
    # 6. WHACK: snare + woodblock + low tom + cymbal + a grunt; then the flight: whoosh, slide whistle up and down, piccolo
    put("perc", S.snare(.25, 5), t(EV["strike"]), 1.0)
    put("perc", woodblock(1100, .12), t(EV["strike"]), .8)
    put("perc", S.tom(110, .4), t(EV["strike"]), .8)
    put("cym", cymbal(1.3, 9), t(EV["strike"]), .55)
    put("voice", voice([(0, 57), (1, 60)], .16, "uh", 63), t(EV["strike"]) - .03, .9)
    put("foley", S.whoosh(.35, 600, 3500, 7), t(EV["strike"]) - .08, .8)
    fl0, flp, fl1 = t(EV["launch"]), t(76), t(EV["spire"])
    put("whistle", slide_whistle(70, 91, flp - fl0, .6), fl0, .7)
    put("whistle", slide_whistle(91, 74, fl1 - flp, 1.4), flp, .7)
    # 7. landing on the spire: a far-off pizzicato plip + a tiny puff; satisfied nod "mm-hm"
    put("pizz", pizz(91, .3, 7), t(EV["spire"]), .9)
    put("xylo", xylo(89, .3), t(EV["spire"]) + .02, .5)
    put("foley", puff(.15, 35, 2000), t(EV["spire"]), .2)
    put("wind", bassoon(53, .12), t(EV["nod"]), .8)
    put("wind", bassoon(58, .2), t(EV["nod"]) + .14, .8)
    put("voice", voice([(0, 60), (.45, 60), (.55, 64), (1, 63)], .34, "hm", 64), t(EV["nod"]), .75)
    put("pizz", pizz(84, .25, 8), t(EV["notice"]), .7)
    # 8. sweeps: brush swishes with xylophone glissandi up / down / up
    for k, (fs, up) in enumerate(((100, True), (102, False), (104, True))):
        put("foley", swish(.13, 42 + k, up), t(fs) - .03, .9)
        put("foley", S.snare(.12, 9 + k, brush=True), t(fs) - .02, .5)
        notes = (72, 76, 79, 84) if up else (84, 79, 76, 72)
        for j, m in enumerate(notes):
            put("xylo", xylo(m, .18), t(fs) - .03 + j * .022, .45)
        put("foley", puff(.18, 36 + k, 1500), t(fs), .3)
    # 9. whip-turn back and the button: a "ta-DAA" brass + strings chord, light cymbal, a satisfied "hmph!"
    put("foley", swish(.14, 47, False), t(EV["turn2"]), .7)
    put("brass", brass((60, 64, 67), .12, 1800), t(EV["proud"]) - .13, .7)
    put("brass", brass((65, 69, 72, 77), .9, 2000), t(EV["proud"]), 1.0)
    put("str", strings_trem((53, 65, 69, 72), .85, rate=0.0001), t(EV["proud"]), .55)
    put("bass", pizz(41, .6, 9), t(EV["proud"]), 1.0)
    put("cym", cymbal(1.0, 10), t(EV["proud"]), .3)
    put("voice", voice([(0, 64), (.3, 61), (1, 58)], .3, "hm", 65), t(EV["proud"]) + .12, .9)

    g = dict(wind=.26, bass=.26, pizz=.18, xylo=.22, brass=.26, str=.15, perc=.32, cym=.10, whistle=.15, foley=.24, voice=.30)
    pans = dict(wind=-.2, bass=-.1, pizz=.2, xylo=.25, brass=0, str=-.15, perc=.05, cym=.15, whistle=.3, foley=.1, voice=0)
    rev = dict(wind=.15, bass=.05, pizz=.15, xylo=.2, brass=.2, str=.25, perc=.08, cym=.1, whistle=.2, foley=.05, voice=.1)
    return S.master([(stems[k], g[k], pans[k], rev[k]) for k in stems], wet=.22, rev_s=1.2, drive=1.25, fade_out=.25)


if __name__ == "__main__":
    import soundfile as sf
    import sys
    a = sound()
    sf.write(sys.argv[1], a, SR, subtype="PCM_24")
    print(a.shape, float(np.abs(a).max()))
