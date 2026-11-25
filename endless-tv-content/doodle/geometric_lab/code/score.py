"""score.py: the music bed for "One Small Passenger" (48 kHz stereo), timed to the shot list in shots.py.

A mid-century jazz/cartoon score: pizzicato bass, brushes and ride, vibraphone, a music-box celesta for Mrs. Dawdle, a plodding
bassoon for the waiting, muted trumpet fanfares for Mr. Hoist's ceremony, and cartoon hits (harp gliss, wah-wah trombone, slide
whistle, timpani bonk).  Silent under the title and end cards, which carry their own stings.  Everything is synthesised."""
import math
import os
import sys

import numpy as np
import soundfile as sf

import shots
import snd
import synth as S

SR = S.SR
OUT = "/home/user/Wan2GP/outputs/shorts/geometric_lab/music_bed.wav"


# ============================================================================================ extra instruments
def env(n, a=.005, r=.05):
    t = np.arange(n) / SR
    return np.clip(t / max(a, 1e-6), 0, 1) * np.clip((n / SR - t) / max(r, 1e-6), 0, 1)


def celesta(m, dur=1.0, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = S.hz(m)
    y = np.sin(2 * np.pi * f * t) * np.exp(-t * 3.2) + .3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6) + .12 * np.sin(2 * np.pi * 4.02 * f * t) * np.exp(-t * 12)
    return y * env(n, .002, .06) * vel


def reed(m, dur, kind="bassoon", vel=1.0, glide=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = S.hz(m) * (1 + glide * t / max(dur, 1e-3)) * (1 + .01 * np.sin(2 * np.pi * 5 * t) * np.clip((t - .15) / .2, 0, 1))
    if kind == "bassoon":
        x = S.osc("saw", f, n)
        y = S.bp(x, 300, 900) * 1.2 + S.lp(x, 500) * .6
    else:                                                    # clarinet: odd harmonics
        x = S.osc("square", f, n)
        y = S.lp(x, 2200)
    return y * env(n, .03, .08) * vel


def woodblock(hi=True):
    n = int(.12 * SR)
    t = np.arange(n) / SR
    f = 1250 if hi else 900
    return (np.sin(2 * np.pi * f * t) * np.exp(-t * 60) + S.bp(S.noise(n, 3 if hi else 4), 1500, 5000) * np.exp(-t * 200) * .3)


def harp_gliss(m0, m1, dur, vel=1.0):
    out = S.buf(dur + 1.2)
    notes = [m for m in range(m0, m1 + 1) if m % 12 in (0, 2, 4, 5, 7, 9, 11)]
    for k, m in enumerate(notes):
        x = S.pluck(S.hz(m), 1.0, seed=m, damp=.997)
        S.put(out, x * .5, k * dur / len(notes))
    return out * vel


def trombone(m0, m1, dur, vel=1.0):
    n = int(dur * SR)
    f = np.linspace(S.hz(m0), S.hz(m1), n)
    x = S.osc("saw", f, n)
    y = np.zeros(n)
    for s in range(0, n, 480):
        fc = 350 + 900 * math.sin(math.pi * s / n)
        y[s:s + 480] = S.bp(x[max(0, s - 2400):s + 480], fc * .6, fc * 1.6)[-min(480, n - s):]
    return y * env(n, .03, .1) * vel


def tremolo_pad(ms, dur, vel=1.0, rate=9.0, swell=True):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = sum(S.osc("saw", S.hz(m) * (1 + .003 * k), n) for k, m in enumerate(ms))
    y = S.lp(y, 1400) * (.6 + .4 * np.sin(2 * np.pi * rate * t))
    shape = np.clip(t / dur, 0, 1) ** 1.5 if swell else 1
    return y * env(n, .1, .2) * shape * vel


def brass_pad(ms, dur, vel=1.0):
    return sum(snd.trumpet(m, dur, k) for k, m in enumerate(ms)) * vel


def slide_whistle(f0, f1, dur, vel=1.0):
    n = int(dur * SR)
    x = S.slide("sine", f0, f1, dur, .8) * (1 + .03 * np.sin(2 * np.pi * 7 * np.arange(n) / SR))
    return x * env(n, .02, .06) * vel


def timp(f=70, dur=.8, vel=1.0):
    return S.tom(f, dur) * vel


# ============================================================================================ sequencing helpers
class Bus:
    def __init__(self, dur):
        self.tracks = {k: S.buf(dur) for k in ("bass", "drums", "vibes", "lead", "fx", "pad")}

    def put(self, trk, x, t, g=1.0):
        S.put(self.tracks[trk], x, t, g)


def walk(b, t0, t1, notes, beat=.5, g=1.0):
    """Pizzicato walking bass over [t0, t1) cycling through notes."""
    k, t = 0, t0
    while t < t1 - .1:
        b.put("bass", snd.bass_note(notes[k % len(notes)], beat * .95, k), t, g)
        k += 1
        t += beat


def brushes(b, t0, t1, beat=.5, g=1.0, ride=True):
    k, t = 0, t0
    while t < t1 - .1:
        if ride:
            b.put("drums", snd.ride(100 + k, .8, .55 if k % 2 == 0 else .8), t, g * .9)
            if k % 2 == 1:
                b.put("drums", snd.ride(300 + k, .5, .4), t + beat * 2 / 3, g * .9)
        if k % 2 == 1:
            b.put("drums", snd.brush(400 + k, .26, 1.0), t, g)
        k += 1
        t += beat


def melody(b, trk, t0, notes, inst, beat=.5, g=1.0):
    t = t0
    for m, d in notes:
        if m is not None:
            b.put(trk, inst(m, d * beat), t, g)
        t += d * beat
    return t


# ============================================================================================ the cues, one per shot
def cue(b, sid, t0, d):
    T = lambda x: t0 + x                                                       # noqa: E731
    vib = lambda m, dur: snd.vibe(m, dur + .5, .9)                             # noqa: E731
    cel = lambda m, dur: celesta(m, dur + .8)                                  # noqa: E731
    if sid == "s01":                                     # ceremony: prim walk, then a curious hush when he spots her
        walk(b, T(0), T(6.0), [41, 45, 48, 50, 46, 45, 43, 48], g=.9)
        brushes(b, T(0), T(6.0))
        melody(b, "vibes", T(0), [(84, .5), (81, .5), (77, 1), (None, 1), (81, .5), (84, .5), (89, 1)], vib, g=.8)
        b.put("pad", brass_pad([65, 69, 72], 1.6, .5), T(.0))
        for k, m in enumerate((72, 76, 79)):                                   # "hm?" — three tiptoe plucks rising
            b.put("vibes", snd.vibe(m, .6, .7), T(6.6 + k * .35))
        b.put("lead", reed(70, 1.4, "clarinet", .5, glide=.06), T(8.0))
    elif sid == "s02":                                   # Mrs. Dawdle's music-box theme
        melody(b, "vibes", T(.3), [(84, 1), (81, 1), (77, 1), (79, .5), (81, .5), (77, 2)], cel, beat=.55, g=.9)
        for k, m in enumerate((41, 48, 41, 48)):
            b.put("bass", snd.bass_note(m, .5, k) * .6, T(.3 + k * 1.1))
        b.put("fx", slide_whistle(900, 1500, .25, .45), T(2.6))
    elif sid == "s03":                                   # the creed: a pompous brass swell, then a noble little fanfare
        b.put("pad", brass_pad([53, 57, 60], 5.2, .35), T(.3))
        b.put("drums", timp(55, 1.5, .5), T(.3))
        t = melody(b, "lead", T(5.9), [(72, .5), (72, .5), (77, 1.5)], lambda m, dur: snd.trumpet(m, dur, m), g=.8)
        b.put("vibes", snd.vibe(81, 1.2, .6), T(6.9))
    elif sid == "s04":                                   # waiting: tick-tock and a plodding bassoon; the ticks get faster
        t, k = T(0), 0
        while t < T(d) - .2:
            b.put("drums", woodblock(k % 2 == 0), t, .55)
            k += 1
            t += 1.0 if t < T(7.5) else .5
        melody(b, "lead", T(.5), [(41, 2), (40, 2), (39, 2), (38, 2), (37, 2)], lambda m, dur: reed(m, dur * .9, "bassoon", .8), beat=1.0)
        b.put("vibes", snd.vibe(79, 1.2, .4), T(9.5))
        b.put("lead", reed(67, .5, "clarinet", .4, glide=.08), T(10.6))
    elif sid == "s05":                                   # the carpet: jaunty, a harp gliss as it unrolls, then a wah-wah deflate
        walk(b, T(0), T(7.0), [41, 45, 48, 50, 53, 50, 48, 45], beat=.45, g=.9)
        brushes(b, T(0), T(7.0), beat=.45)
        b.put("fx", harp_gliss(60, 96, .9, .9), T(2.8))
        b.put("vibes", snd.vibe(89, 1.0, .7), T(3.8))
        melody(b, "vibes", T(4.2), [(84, .5), (86, .5), (89, 1)], vib, beat=.45, g=.7)
        b.put("lead", trombone(58, 57, .45, .8), T(7.4))
        b.put("lead", trombone(57, 52, .9, .8), T(7.9))
        walk(b, T(9.0), T(d), [36, 38, 39, 40], beat=.6, g=.6)
    elif sid == "s06":                                   # lettuce: a coaxing waltz, munch-munch-munch, a yawn, jaw drop
        t = T(0)
        k = 0
        while t < T(5.2):
            b.put("bass", snd.bass_note(41 if (k // 3) % 2 == 0 else 48, .3, k) * (1 if k % 3 == 0 else 0), t)
            if k % 3:
                b.put("vibes", snd.vibe(69 if (k // 3) % 2 == 0 else 72, .4, .45), t)
            k += 1
            t += .4
        melody(b, "lead", T(.2), [(77, 3), (81, 3), (79, 3), (77, 3)], lambda m, dur: reed(m, dur * .95, "clarinet", .5), beat=.4)
        for j in range(3):
            b.put("fx", S.pluck(S.hz(48 + j * 2), .3, seed=j) * .8, T(5.4 + j * .3))
        melody(b, "vibes", T(6.6), [(84, 1), (81, 1), (77, 1), (72, 2)], cel, beat=.42, g=.8)
        b.put("lead", trombone(46, 34, 1.1, .9), T(9.0))
        b.put("drums", snd.ride(77, 1.4, .6), T(9.0))
    elif sid == "s07":                                   # strain (tremolo swell) ... then calm
        b.put("pad", tremolo_pad([41, 47, 53], 4.4, .9), T(.2))
        b.put("drums", timp(60, .8, .6), T(4.5))
        b.put("vibes", snd.vibe(65, 2.5, .6), T(4.7))
        b.put("vibes", snd.vibe(69, 2.5, .5), T(4.72))
        b.put("vibes", snd.vibe(72, 2.5, .5), T(4.74))
        b.put("pad", brass_pad([53, 57, 60], 2.8, .25), T(4.8))
    elif sid == "s08":                                   # night: music-box lullaby, then the wake-up BWAH and a happy run
        mel = [(77, 1), (81, 1), (84, 1), (82, 2), (79, 1), (77, 1), (76, 1), (77, 3), (72, 1), (74, 1), (76, 1), (77, 3)]
        melody(b, "vibes", T(.4), mel, cel, beat=.55, g=.75)
        t, k = T(.4), 0
        while t < T(10.2):
            b.put("bass", snd.bass_note([41, 36][k % 2], .6, k) * .5, t)
            k += 1
            t += .55 * 3
        b.put("lead", brass_pad([60, 64, 67], .5, .9), T(10.9))
        b.put("drums", snd.ride(5, 1.8, 1.0), T(10.9))
        for j, m in enumerate((72, 76, 79, 84, 88)):
            b.put("vibes", snd.vibe(m, .7, .7), T(11.6 + j * .08))
        walk(b, T(12.0), T(d), [41, 45, 48, 45], beat=.45, g=.7)
    elif sid == "s09":                                   # going up: march, lever hit + ding, salute chord, rising sweep to the cut
        walk(b, T(0), T(4.8), [41, 48, 41, 48], beat=.5)
        brushes(b, T(0), T(4.8), ride=False)
        b.put("pad", brass_pad([53, 57, 60], 1.2, .3), T(.2))
        b.put("drums", timp(55, .9, 1.0), T(5.0))
        b.put("lead", snd.trumpet(72, .12, 1), T(4.75))
        b.put("lead", snd.trumpet(77, 1.6, 2), T(5.0))
        b.put("lead", snd.trumpet(72, 1.6, 3) * .6, T(5.0))
        b.put("fx", snd.bell(1046.5, 1.5) * .5, T(5.3))
        b.put("drums", snd.ride(6, 2.0, 1.0), T(5.0))
        b.put("fx", harp_gliss(53, 89, 2.0, .7), T(d - 2.4))
        b.put("pad", tremolo_pad([65, 69, 72], 2.0, .5, rate=12), T(d - 2.2))
    elif sid == "s10":                                   # second floor: a bright pastoral moment, then the deflating bassoon
        melody(b, "vibes", T(.1), [(77, 1), (81, 1), (84, 1), (89, 2)], vib, beat=.45, g=.8)
        walk(b, T(0), T(3.8), [41, 45, 48, 45], beat=.45, g=.7)
        brushes(b, T(0), T(3.8), beat=.45, g=.7)
        melody(b, "lead", T(4.6), [(53, 1), (52, 1), (51, 1), (50, 1), (49, 2.5)], lambda m, dur: reed(m, dur * .9, "bassoon", .9), beat=.8)
    elif sid == "s11":                                   # babies: tiny pizzicato steps; the plank topple; wah-wah-waaah
        notes = [84, 86, 88, 89, 91, 89, 88, 86]
        for j in range(16):
            b.put("fx", S.pluck(S.hz(notes[j % 8]), .25, seed=j, damp=.99) * .6, T(.6 + j * .3))
            if j % 2 == 0:
                b.put("bass", snd.bass_note([48, 43][(j // 2) % 2], .25, j) * .5, T(.6 + j * .3))
        b.put("vibes", celesta(96, 1.0, .5), T(5.4))
        b.put("fx", slide_whistle(1400, 260, 1.0, .8), T(7.6))
        b.put("drums", timp(48, 1.0, 1.0), T(8.7))
        b.put("drums", woodblock(False), T(8.7), .8)
        for j, (a, bb, dd) in enumerate(((58, 57, .5), (57, 56, .5), (56, 50, 1.4))):
            b.put("lead", trombone(a, bb, dd, .8), T(9.3 + j * .55))


def build():
    tl, total = shots.timeline()
    b = Bus(total + 1.0)
    for s, t0, d in tl:
        cue(b, s["id"], t0, d)
    tr = b.tracks
    mix = S.master([(tr["bass"], .30, -.15, .06), (tr["drums"], .12, .3, .15), (tr["vibes"], .24, .25, .3), (tr["lead"], .26, -.05, .25),
                    (tr["fx"], .22, .1, .25), (tr["pad"], .16, 0, .3)], wet=.2, rev_s=1.4, drive=1.15, fade_out=.05)
    n = int(total * SR)
    mix = mix[:n]
    # silence under the cards (they have their own stings) with short fades at the joins
    t = np.arange(n) / SR
    t_end = total - shots.END_FRAMES / shots.FPS
    g = np.clip((t - shots.TITLE_FRAMES / shots.FPS) / .05, 0, 1) * np.clip((t_end - t) / .6, 0, 1)
    mix = mix * g[:, None]
    return (mix * (10 ** (-3 / 20)) / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, build(), SR, subtype="PCM_24")
    print(out)
