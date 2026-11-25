"""Sound for the Bertram shot, all synthesised: a jaunty early-60s TV-cartoon cue (pizzicato walking bass, brushes, vibraphone,
a muted-brass stab), a pompous baritone gibberish voice made by formant synthesis, and hand-timed cartoon effects."""
import math

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

import bertram as BT

SR = 48000
DUR = BT.DUR
FPS = BT.FPS


def T(f):
    return f / FPS


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def buf():
    return np.zeros(int(DUR * SR))


def put(dst, x, t0, g=1.0):
    i = int(t0 * SR)
    if i >= len(dst) or i + len(x) <= 0:
        return
    if i < 0:
        x, i = x[-i:], 0
    n = min(len(x), len(dst) - i)
    dst[i:i + n] += x[:n] * g


def lp(x, f, order=2):
    return sosfilt(butter(order, min(f, SR / 2 - 100) / (SR / 2), "low", output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f / (SR / 2), "high", output="sos"), x)


def bp(x, f0, f1, order=2):
    return sosfilt(butter(order, [f0 / (SR / 2), min(f1, SR / 2 - 100) / (SR / 2)], "band", output="sos"), x)


def noise(n, seed):
    return np.random.default_rng(seed).normal(0, 1, n)


def env(n, a=.005, d=.1, s=.6, r=.05):
    t = tt(n)
    dur = n / SR
    e = np.where(t < a, t / max(a, 1e-6), np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-6), s))
    return e * np.clip((dur - t) / max(r, 1e-6), 0, 1)


def phase(f, n):
    return np.cumsum(np.broadcast_to(f, (n,)) / SR)


def resonator(x, f, bw):
    r = math.exp(-math.pi * bw / SR)
    a1, a2 = -2 * r * math.cos(2 * math.pi * f / SR), r * r
    return lfilter([1 - r], [1, a1, a2], x)


# ----------------------------------------------------------------------------------------------- instruments
def pluck_bass(m, dur=.55, seed=0):
    """Pizzicato upright bass: Karplus-Strong string + a woody body thump."""
    f = hz(m)
    n, p = int(dur * SR), max(2, int(round(SR / f)))
    rng = np.random.default_rng(seed)
    b = lp(rng.uniform(-1, 1, p * 3), 900)[-p:]
    out = np.empty(n)
    for i in range(n):
        out[i] = b[i % p]
        b[i % p] = .994 * .5 * (b[i % p] + b[(i + 1) % p])
    t = tt(n)
    thump = np.sin(2 * np.pi * f * t) * np.exp(-t * 9) * .8
    x = lp(out, 1100) + thump
    return x * np.clip((dur - t) / .06, 0, 1)


def vibes(m, dur=1.3, trem=5.2, g=1.0):
    n = int(dur * SR)
    t = tt(n)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + .25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 9) + .08 * np.sin(2 * np.pi * f * 10 * t) * np.exp(-t * 25)
    x *= np.exp(-t * 2.2) * (1 - .3 * (.5 + .5 * np.sin(2 * np.pi * trem * t)))
    return x * np.clip(t / .002, 0, 1) * g


def xylo(m, dur=.35):
    n = int(dur * SR)
    t = tt(n)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + .4 * np.sin(2 * np.pi * f * 3.93 * t) * np.exp(-t * 30)) * np.exp(-t * 11)


def brass(notes, dur, wah=True, cutoff=2400, seed=0):
    """A muted-trumpet section stab: detuned saws through a 'wah' band-pass that opens then closes."""
    n = int(dur * SR)
    t = tt(n)
    x = np.zeros(n)
    for k, m in enumerate(notes):
        f = hz(m) * (1 + .004 * np.sin(2 * np.pi * 5.5 * t + k))
        x += (2 * (phase(f, n) % 1) - 1) + .6 * (2 * (phase(f * 1.003, n) % 1) - 1)
    out = np.zeros(n)
    blk = 512
    for s in range(0, n, blk):
        u = s / n
        fc = 500 + (cutoff - 500) * (math.sin(math.pi * min(u * 1.6, 1)) if wah else 1)
        seg = x[max(0, s - 2048):s + blk]
        y = bp(seg, fc * .5, fc * 1.6)[-min(blk, n - s):]
        out[s:s + len(y)] = y
    return out * env(n, .02, .12, .75, .08)


def brush(dur=.22, seed=3, swish=True):
    n = int(dur * SR)
    t = tt(n)
    x = bp(noise(n, seed), 2500, 11000)
    if swish:
        return x * np.clip(t / .05, 0, 1) * np.exp(-t * 10)
    return x * np.exp(-t * 60)


def woodblock(f=820, dur=.18):
    n = int(dur * SR)
    t = tt(n)
    return (np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * f * 1.58 * t)) * np.exp(-t * 38) + hp(noise(n, 9), 3000) * np.exp(-t * 200) * .3


def thud(f0=120, f1=48, dur=.4):
    n = int(dur * SR)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t * 30)
    return np.sin(2 * np.pi * phase(f, n)) * np.exp(-t * 11)


def slide_whistle(f0, f1, dur, curve=1.0, seed=0):
    n = int(dur * SR)
    u = np.linspace(0, 1, n) ** curve
    f = f0 * (f1 / f0) ** u * (1 + .012 * np.sin(2 * np.pi * 6 * tt(n)))
    x = np.sin(2 * np.pi * phase(f, n)) + .04 * bp(noise(n, seed), 1500, 6000)
    return x * env(n, .02, .05, .9, .04)


def whoosh(dur, f0, f1, seed=7):
    n = int(dur * SR)
    x = noise(n, seed)
    out = np.zeros(n)
    k = 512
    for s in range(0, n, k):
        fc = f0 * (f1 / f0) ** (s / n)
        seg = x[max(0, s - 2048):s + k]
        y = bp(seg, fc * .7, fc * 1.4)[-min(k, n - s):]
        out[s:s + len(y)] = y
    return out * np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5


def boing(f=110, dur=.55):
    """Jaw-harp style 'boi-oi-oing': a buzzy tone through a resonant filter that wobbles."""
    n = int(dur * SR)
    t = tt(n)
    src = 2 * (phase(f * (1 + .15 * np.exp(-t * 6)), n) % 1) - 1
    out = np.zeros(n)
    blk = 256
    for s in range(0, n, blk):
        tc = s / SR
        fc = 600 + 900 * (.5 + .5 * math.sin(2 * math.pi * 11 * tc)) * math.exp(-tc * 3)
        seg = src[max(0, s - 2048):s + blk]
        y = bp(seg, fc * .8, fc * 1.25)[-min(blk, n - s):]
        out[s:s + len(y)] = y
    return out * np.exp(-t * 4.5) * np.clip(t / .004, 0, 1)


def squeak(f0, f1, dur, seed=2):
    """Rubbery inflating squeak for the chest puff."""
    n = int(dur * SR)
    t = tt(n)
    u = np.linspace(0, 1, n)
    f = f0 * (f1 / f0) ** u * (1 + .05 * np.sin(2 * np.pi * 23 * t))
    x = np.sin(2 * np.pi * phase(f, n)) + .3 * np.sin(4 * np.pi * phase(f, n))
    return lp(x, 3000) * env(n, .03, .05, .9, .06)


def pop(dur=.12):
    n = int(dur * SR)
    t = tt(n)
    f = 900 * np.exp(-t * 30) + 250
    return np.sin(2 * np.pi * phase(f, n)) * np.exp(-t * 40) + hp(noise(n, 4), 2000) * np.exp(-t * 300) * .4


# ----------------------------------------------------------------------------------------------- the voice
FORM = {"ah": (760, 1180, 2500), "ee": (310, 2150, 2900), "oh": (520, 880, 2450), "mm": (260, 1000, 2300)}
ONSET = ["h", "tr", "f", "th", "k", "n", "z", "m"]


def glottal(f0, n, seed=0):
    """Band-limited buzzy source with a gentle spectral tilt and a little jitter."""
    rng = np.random.default_rng(seed)
    jit = 1 + .006 * lp(rng.normal(0, 1, n), 30) * 20
    ph = phase(f0 * jit, n)
    x = np.zeros(n)
    fmax = float(np.max(f0))
    for k in range(1, int(4200 / fmax)):
        x += np.sin(2 * np.pi * k * ph) / k ** 1.1
    return x


def voice():
    out = buf()
    base = 112.0
    for idx, (s0, nfr, mouth, jaw, semi) in enumerate(BT.SYL):
        t0 = T(s0) - .02
        dur = T(nfr) + .07
        n = int(dur * SR)
        t = tt(n)
        f0 = base * 2 ** (semi / 12)
        if idx == len(BT.SYL) - 1:      # the grand finish: up, then a vain sliding fall with a wide vibrato
            contour = f0 * (1 + .25 * np.sin(np.pi * np.clip(t / dur * 1.3, 0, 1)) - .18 * (t / dur) ** 2)
            vib = 1 + .035 * np.sin(2 * np.pi * 6.2 * t) * np.clip(t / .12, 0, 1)
        else:
            contour = f0 * (1.06 - .1 * t / dur)
            vib = 1 + .015 * np.sin(2 * np.pi * 5.5 * t)
        src = glottal(contour * vib, n, seed=idx)
        F1, F2, F3 = FORM[mouth]
        F1 *= 1 + .004 * (jaw - 10)
        y = resonator(src, F1, 90) * 1.0 + resonator(src, F2, 120) * .55 + resonator(src, F3, 170) * .25
        y = lp(y, 5000)
        y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9)
        e = env(n, .03, .08, .85, .06)
        emph = 1.25 if idx in (0, len(BT.SYL) - 1) else 1.0
        put(out, y * e, t0, emph)
        # consonant onset
        c = ONSET[idx % len(ONSET)]
        if c in ("h", "f", "th", "z"):
            m = int(.07 * SR)
            nz = bp(noise(m, 30 + idx), 1800 if c != "h" else 900, 7000 if c != "h" else 3000) * np.sin(np.pi * np.linspace(0, 1, m))
            put(out, nz, t0 - .05, .07 if c != "h" else .05)
        elif c in ("tr", "k"):
            m = int(.03 * SR)
            nz = bp(noise(m, 40 + idx), 1500, 6000) * np.exp(-tt(m) * 90)
            put(out, nz, t0 - .02, .16)
        else:   # nasal hum before the vowel
            m = int(.08 * SR)
            hm = resonator(glottal(np.full(m, f0), m, seed=50 + idx), 260, 60) * np.sin(np.pi * np.linspace(0, 1, m))
            put(out, lp(hm, 600), t0 - .06, .6)
    return out


# ----------------------------------------------------------------------------------------------- the cue
def music():
    bass, drums, vib, horn = buf(), buf(), buf(), buf()
    beat = 60 / 128
    t0 = .06
    walk = [41, 45, 48, 50, 46, 50, 53, 50, 48]
    for k, m in enumerate(walk):
        put(bass, pluck_bass(m, beat * .95, seed=k), t0 + k * beat, 1.0)
    put(bass, pluck_bass(41, 1.0, seed=20), T(EV_BUTTON), 1.1)
    for k in range(10):                    # brushes: swish on 2 and 4, light taps on the off-beats
        tb = t0 + k * beat
        if k % 2 == 1:
            put(drums, brush(.25, seed=k), tb - .03, .55)
        put(drums, brush(.05, seed=100 + k, swish=False), tb + beat * .5, .22)
        put(drums, brush(.04, seed=200 + k, swish=False), tb, .16)
    # vibes intro figure
    for m, tb in ((72, .06), (69, .29), (65, .53), (69, .70)):
        put(vib, vibes(m, 1.0), tb, .6)
    # the brass stab on the raise
    put(horn, brass([53, 57, 60, 62], .45), T(BT.EV["raise_"]) - .01, 1.0)
    put(horn, brass([57, 60, 64, 65], .3, wah=False, cutoff=1800), T(BT.EV["raise_"]) + .42, .5)
    # xylophone run up as the cap pops
    for k, m in enumerate((72, 74, 76, 77, 79, 81, 84)):
        put(vib, xylo(m), T(BT.EV["pop"]) + k * .03, .45)
    # button: vibes chord + muted brass doo-wah
    for m in (65, 69, 72, 76):
        put(vib, vibes(m, 1.0), T(EV_BUTTON), .45)
    put(horn, brass([60, 65], .5, cutoff=2000), T(EV_BUTTON) + .02, .55)
    return bass, drums, vib, horn


EV_BUTTON = 102


def sfx():
    fx = buf()
    put(fx, slide_whistle(900, 620, .16, seed=1), T(BT.EV["antic"]) + .02, .25)
    put(fx, slide_whistle(600, 2100, .12, curve=.7, seed=2), T(BT.EV["raise_"]) - .07, .35)
    put(fx, whoosh(.25, 600, 4000, seed=3), T(BT.EV["raise_"]) - .12, .5)
    put(fx, squeak(420, 900, T(BT.EV["pop"] - 62), seed=5), T(62), .2)
    put(fx, pop(), T(BT.EV["pop"]), .8)
    put(fx, slide_whistle(700, 1500, .16, seed=6), T(BT.EV["pop"]) + .02, .25)
    put(fx, slide_whistle(1500, 600, .2, curve=1.4, seed=7), T(BT.EV["land"]) - .22, .22)
    put(fx, woodblock(760), T(BT.EV["land"]), .8)
    put(fx, boing(105, .5), T(BT.EV["land"]) + .01, .45)
    put(fx, whoosh(.22, 3000, 500, seed=8), T(BT.EV["lower"]) + .02, .35)
    put(fx, thud(), T(BT.EV["thunk"]) - .01, .9)
    put(fx, woodblock(520, .2), T(BT.EV["thunk"]) - .01, .4)
    return fx


# ----------------------------------------------------------------------------------------------- mix
def reverb(x, seconds=1.0, seed=3, damp=4500):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-np.arange(n) / (SR * seconds / 6.9))[:, None]
    ir = lfilter(*butter(1, damp / (SR / 2)), ir, axis=0)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))
    return np.stack([fftconvolve(x, ir[:, c])[:len(x)] for c in range(2)], 1)


def pan(x, p):
    a = (p + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)], 1) * math.sqrt(2)


def duck(n):
    """Pull the band down while he talks (and while the cap gag plays), like a TV dub stage would."""
    t = tt(n)
    g = np.ones(n)
    a, b = T(BT.SYL[0][0]) - .1, T(BT.SYL[-1][0] + BT.SYL[-1][1]) + .1
    g -= .55 * np.clip(np.minimum((t - a) / .12, (b - t) / .2), 0, 1)
    return g


def sound():
    n = int(DUR * SR)
    bass, drums, vib, horn = music()
    v = voice()
    fx = sfx()
    d = duck(n)
    stems = [(bass * d, .30, -.15, .12), (drums * d, .13, .25, .15), (vib * d, .20, .3, .35), (horn * (.4 + .6 * d), .26, -.25, .3),
             (v, .62, 0, .14), (fx, .40, .08, .18)]
    dry, send = np.zeros((n, 2)), np.zeros(n)
    for x, g, p, r in stems:
        x = np.pad(x, (0, n - len(x)))[:n]
        x = x / (np.max(np.abs(x)) + 1e-9) * g
        dry += pan(x, p)
        send += x * r
    mix = dry + reverb(send, 1.0) * .28
    t = tt(n)
    mix *= np.clip(t / .02, 0, 1)[:, None] * np.clip((n / SR - t) / .3, 0, 1)[:, None]
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    return (mix * .89 / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)


if __name__ == "__main__":
    import soundfile as sf
    import sys
    a = sound()
    sf.write(sys.argv[1] if len(sys.argv) > 1 else "test.wav", a, SR, subtype="PCM_24")
