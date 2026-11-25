"""Sound for the geometric_lab shot: a prim little mid-century jazz cue (pizzicato bass, brushes, ride, vibraphone,
a muted-trumpet "ta-ta-TAAA"), crisp cartoon SFX synced to the animation, and a formant-synth gibberish voice for
Mr. Hoist.  Everything is computed from maths; no samples."""
import math

import numpy as np
from scipy.signal import lfilter

import synth as S
from geo import SYL, FPS, T_YANK, T_DING, T_POINT, T_SALUTE, T_BACK, T_BUTTON

SR = S.SR
DUR = 5.0
BEAT = .5                      # 120 bpm: 12 animation frames per beat
fr = lambda f: f / FPS


def env(n, a=.005, r=.05):
    t = np.arange(n) / SR
    return np.clip(t / max(a, 1e-6), 0, 1) * np.clip((n / SR - t) / max(r, 1e-6), 0, 1)


def bpf(x, f0, q):
    w0 = 2 * math.pi * f0 / SR
    al = math.sin(w0) / (2 * q)
    b = np.array([al, 0, -al])
    a = np.array([1 + al, -2 * math.cos(w0), 1 - al])
    return lfilter(b / a[0], a / a[0], x)


# ============================================================================================ voice
VOWELS = {"o": (560, 920, 2450), "u": (660, 1200, 2400), "i": (290, 2200, 2900), "e": (500, 1820, 2520), "a": (740, 1120, 2560),
          "l": (360, 1250, 2650), "r": (420, 1150, 1650), "n": (260, 1050, 2300)}
CONS = [("g", "", "o"), ("", "ng", "i"), ("", "p", "u"), ("", "k", "e"), ("st", "m", "i"), ("l", "", "i"), ("s", "", "i"),
        ("r", "", "i"), ("", "s", "u"), ("s", "ks", "a")]


def glottal(f0):
    """Differentiated Rosenberg pulse train: a buzzy, reedy voice source."""
    ph = np.cumsum(f0 / SR) % 1.0
    op = .58
    flow = np.where(ph < op, .5 * (1 - np.cos(np.pi * ph / op)), np.where(ph < op * 1.35, np.cos(np.pi * (ph - op) / (op * .7)), 0))
    src = np.gradient(flow)
    return src / (np.max(np.abs(src)) + 1e-9)


def formant(src, f, gains=(1.0, .55, .3)):
    out = np.zeros_like(src)
    for fc, g, q in zip(f, gains, (5.5, 9.0, 12.0)):
        y = bpf(src, fc, q)
        out += g * y / (np.max(np.abs(y)) + 1e-9)
    return out


def burst(n_ms, lo, hi, seed, decay=60.0):
    n = int(n_ms / 1000 * SR)
    return S.bp(S.noise(n, seed), lo, hi) * np.exp(-np.arange(n) / SR * decay)


def hiss(n_ms, seed, lo=4200, hi=9500):
    n = int(n_ms / 1000 * SR)
    return S.bp(S.noise(n, seed), lo, hi) * np.sin(np.pi * np.linspace(0, 1, n)) ** .7


def voice():
    out = S.buf(DUR)
    rng = np.random.default_rng(11)
    for k, ((t0, t1, op, wd, st, _v), (on, coda, vw)) in enumerate(zip(SYL, CONS)):
        dur = t1 - t0
        n = int(dur * SR)
        t = np.arange(n) / SR
        # pitch: a fussy old gent around 150 Hz; stressed syllables jump up and fall; a little elderly quaver
        p0 = 205 if st else 152 - 4 * (k % 3)
        p1 = 158 if st else p0 - 10
        if k == len(SYL) - 1:
            p0, p1 = 215, 140                                    # the final "SOCKS!" falls a long way
        f0 = (p0 + (p1 - p0) * (t / dur) ** .8) * (1 + .022 * np.sin(2 * np.pi * 5.8 * t + k)) * (1 + .004 * rng.normal(0, 1, n))
        src = glottal(f0) + .05 * S.noise(n, 30 + k)
        v = formant(src, VOWELS[vw])
        if on in ("l", "r"):                                     # glide in from the liquid's formants
            v0 = formant(src, VOWELS[on])
            w = np.clip(t / .05, 0, 1)
            v = v0 * (1 - w) + v * w
        if coda in ("m", "ng"):                                  # close into a nasal hum
            nas = S.lp(src, 380) * 2.2
            w = np.clip((t - (dur - .07)) / .03, 0, 1)
            v = v * (1 - w) + nas / (np.max(np.abs(nas)) + 1e-9) * w * .7
        amp = env(n, .012, .035 if coda in ("p", "k", "ks") else .05) * (1.0 if st else .78)
        if coda in ("p", "k", "ks"):                             # stop consonants cut the vowel short
            amp *= np.clip((dur - .02 - t) / .01, 0, 1)
        S.put(out, v * amp, t0)
        # consonants
        if on == "g":
            S.put(out, burst(14, 900, 2600, 50 + k, 120) * .5, t0 - .012)
        if on in ("s", "st"):
            S.put(out, hiss(70, 60 + k) * .35, t0 - .07)
        if on == "st":
            S.put(out, burst(12, 2500, 6000, 70 + k, 160) * .45, t0 - .012)
        if coda == "p":
            S.put(out, burst(10, 400, 1600, 80 + k, 200) * .4, t1 - .01)
        if coda in ("k", "ks"):
            S.put(out, burst(14, 1400, 3400, 90 + k, 150) * .45, t1 - .015)
        if coda in ("s", "ks"):
            S.put(out, hiss(90, 100 + k) * .32, t1 + (.0 if coda == "s" else .01))
    return S.hp(out, 90)


# ============================================================================================ music
def bass_note(m, dur, seed):
    f = S.hz(m)
    x = S.pluck(f, dur, seed=seed, damp=.993, bright=.6)
    n = len(x)
    thump = np.sin(2 * np.pi * f * np.arange(n) / SR) * np.exp(-np.arange(n) / SR * 9)
    y = S.lp(x * .8 + thump * .6, 950) * env(n, .003, .06)
    return y


def vibe(m, dur=1.2, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = S.hz(m)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.2) + .25 * np.sin(2 * np.pi * f * 4.0 * t) * np.exp(-t * 9)
         + .08 * np.sin(2 * np.pi * f * 9.9 * t) * np.exp(-t * 25))
    y *= 1 + .35 * np.sin(2 * np.pi * 5.2 * t)                 # motor tremolo
    y += S.bp(S.noise(n, m), 1500, 5000) * np.exp(-t * 300) * .2   # mallet tick
    return y * vel * env(n, .001, .08)


def ride(seed, dur=1.1, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for fq in rng.uniform(3200, 9000, 14):
        y += np.sin(2 * np.pi * fq * t + rng.uniform(0, 6)) * np.exp(-t * rng.uniform(3, 7))
    y = y / 14 + S.hp(S.noise(n, seed), 6000) * np.exp(-t * 16) * .2
    return y * vel * np.exp(-t * 2.5)


def brush(seed, dur=.3, vel=1.0):
    return S.lp(S.snare(dur, seed=seed, brush=True), 6500) * vel * np.exp(-np.arange(int(dur * SR)) / SR * 6)


def trumpet(m, dur, seed, fall=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = S.hz(m) * (1 + .012 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - .12) / .1, 0, 1)) * (1 - fall * np.clip((t - dur + .12) / .12, 0, 1) ** 2)
    x = S.osc("saw", f, n) * .7 + S.osc("square", f * 1.002, n) * .3
    # brassy filter blip + a harmon-mute nasal band
    out = np.zeros(n)
    k = 480
    for s in range(0, n, k):
        tt = s / SR
        fc = 900 + 2600 * math.exp(-tt * 9) + 300
        seg = x[max(0, s - 2048):s + k]
        y = S.lp(seg, fc)[-min(k, n - s):]
        out[s:s + len(y)] = y
    out = S.bp(out, 900, 3200) * 1.4 + out * .3
    return out * env(n, .02, .06)


def bell(f=880.0, dur=1.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for r, a, d in ((1, 1.0, 2.2), (2.0, .5, 3.5), (2.76, .45, 4.5), (5.40, .25, 7), (8.93, .12, 10)):
        y += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d)
    return y * env(n, .001, .1)


def music():
    bass, vib, rd, br, tp = (S.buf(DUR) for _ in range(5))
    walk = [(0, 41), (1, 45), (2, 48), (3, 50), (4, 46), (5, 45), (6, 43), (7, 48), (7.5, 41), (8, 48), (8.5, 40)]
    for b, m in walk:
        S.put(bass, bass_note(m, .48 if b < 7 else .25, int(b * 10)), b * BEAT)
    S.put(bass, bass_note(41, .55, 99), 9 * BEAT)
    S.put(bass, bass_note(29, .55, 98) * .8, 9 * BEAT)
    # vibes: a prim tiptoe figure, a held breath (silence) for the anticipation, then answers to the action
    for tb, m, d in ((0.0, 84, .3), (.34, 81, .3), (1.0, 77, .3), (1.34, 81, .5), (2.5, 84, .5), (3.0, 89, .9)):
        S.put(vib, vibe(m, d + .5, .9), tb * BEAT)
    for k, m in enumerate((65, 69, 74, 79, 84)):                 # final F6/9 roll
        S.put(vib, vibe(m, .9, .8), 9 * BEAT + k * .018)
    # ride: "ting, ting-a ting" swing until the speech, then just quarter pings; brushes on 2 and 4
    for b in range(10):
        S.put(rd, ride(200 + b, .9, .8 if b % 2 else .55), b * BEAT)
        if b % 2 == 1 and b < 4:
            S.put(rd, ride(300 + b, .6, .45), b * BEAT + BEAT * 2 / 3)
        if b % 2 == 1:
            S.put(br, brush(400 + b, .28, 1.0), b * BEAT)
    S.put(br, brush(499, .45, 1.2), 9 * BEAT)
    # muted trumpet fanfare for the salute: ta-ta-TAAA (+ a third underneath on the long note), then a button "bwap"
    ts = fr(T_SALUTE)
    S.put(tp, trumpet(72, .1, 1), ts - .25)
    S.put(tp, trumpet(72, .1, 2), ts - .125)
    S.put(tp, trumpet(77, .42, 3, fall=.0), ts)
    S.put(tp, trumpet(69, .42, 4) * .6, ts)
    S.put(tp, trumpet(77, .16, 5) * .8, 9 * BEAT)
    S.put(tp, trumpet(81, .16, 6) * .5, 9 * BEAT)
    # duck the band under the voice
    n = len(bass)
    t = np.arange(n) / SR
    s0, s1 = SYL[0][0] - .05, SYL[-1][1] + .05
    duck = 1 - .55 * np.clip(np.minimum((t - s0) / .08, (s1 - t) / .08), 0, 1)
    return [(bass * duck, .26, -.15, .06), (vib * duck, .17, .25, .25), (rd * duck, .06, .35, .15), (br * duck, .065, -.3, .1),
            (tp, .20, .1, .22)]


# ============================================================================================ SFX
def sfx():
    fx = S.buf(DUR)
    t_y, t_d, t_b = fr(T_YANK), fr(T_DING), fr(T_BACK)
    # anticipation: cloth rustle + a little ratchet as he winds the lever back
    S.put(fx, S.bp(S.noise(int(.22 * S.SR), 5), 900, 4000) * np.sin(np.pi * np.linspace(0, 1, int(.22 * S.SR))) * .25, fr(14))
    for k in range(4):
        S.put(fx, burst(6, 2200, 5200, 10 + k, 300) * .5, fr(15) + k * .055)
    # the yank: KA-CHUNK (thud + latch click + metal ring)
    n = int(.4 * SR)
    t = np.arange(n) / SR
    thud = np.sin(2 * np.pi * (60 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 14)
    ring = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((820, .3, 14), (1270, .25, 18), (2110, .18, 22)))
    S.put(fx, thud + ring * .6, t_y)
    S.put(fx, burst(8, 2500, 7000, 21, 250) * 1.2, t_y)
    S.put(fx, burst(6, 1800, 5000, 22, 300) * .8, t_y + .045)
    # the departure ding
    S.put(fx, bell(1046.5, 1.8) * .55, t_d)
    S.put(fx, bell(1318.5, 1.4) * .25, t_d + .004)
    # finger up: a tiny "tink"
    S.put(fx, bell(2637, .35) * .15, fr(T_POINT) - .02)
    # salute: whoosh into a crisp snap
    S.put(fx, S.whoosh(.16, 700, 3200, 31) * .55, fr(T_SALUTE) - .16)
    S.put(fx, burst(5, 1500, 7000, 32, 400) * 1.2, fr(T_SALUTE))
    # lever back: clack + springy boing
    S.put(fx, burst(7, 2000, 6000, 41, 300) * 1.0, t_b)
    nb = int(.45 * SR)
    tb = np.arange(nb) / SR
    boing = np.sin(2 * np.pi * np.cumsum(260 * (1 + .25 * np.exp(-tb * 6)) * (1 + .08 * np.sin(2 * np.pi * 22 * tb))) / SR) * np.exp(-tb * 7)
    S.put(fx, boing * .45, t_b + .01)
    S.put(fx, thud[:int(.25 * SR)] * .5, t_b)
    return fx


def room():
    n = int(DUR * SR)
    air = S.lp(S.hp(S.noise(n, 77), 120), 1800)
    return air * (1 + .2 * np.sin(2 * np.pi * .3 * np.arange(n) / SR))


def sound():
    stems = music()
    stems += [(voice(), .34, -.02, .12), (sfx(), .30, -.05, .14), (room(), .012, 0, 0)]
    return S.master(stems, wet=.16, rev_s=1.1, drive=1.25, fade_out=.25)


if __name__ == "__main__":
    import soundfile as sf
    import sys
    sf.write(sys.argv[1] if len(sys.argv) > 1 else "test.wav", sound(), SR)
