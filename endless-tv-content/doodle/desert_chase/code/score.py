"""The music bed for "Clean Sweep": an original 1950s-theatrical-style orchestral score synthesised in code, timed to the shot list.
Pip's theme (a pompous bassoon strut in F over pizzicato), Bramble's theme (a waddling tuba oompah in B-flat with a trombone tune),
plus cues for the dust cloud, the slow burn, the frantic sweeping, the mud, the storm, the reveal, the friendship and the button.
Mixed to sit under dialogue (the assembly ducks it further while H3 audio is loud)."""
import math

import numpy as np

import shotlist as SL
import tsound as S
from sound import bassoon, brass, clarinet, cymbal, pizz, slide_whistle, strings_trem, woodblock, xylo

SR = S.SR


def hz(m):
    return S.hz(m)


def env(n, a=.01, d=.1, s=.7, r=.08):
    return S.adsr(n, a, d, s, r)


def tuba(m, dur):
    n = int(dur * SR)
    f = hz(m) * (1 + .003 * np.sin(2 * np.pi * 4.5 * S.tt(n)))
    x = S.osc("saw", f, n) + .5 * S.osc("square", f * 1.002, n, pw=.4)
    x = S.lp(x, 700) + .5 * S.bp(x, 250, 450)
    return x * env(n, .03, .1, .75, .06)


def trombone(m, dur, slide=0.0, wah=0.0):
    n = int(dur * SR)
    tt = S.tt(n)
    f = hz(m) * 2 ** (slide * np.linspace(0, 1, n) ** 1.5 / 12) * (1 + .006 * np.sin(2 * np.pi * 5 * tt))
    x = S.osc("saw", f, n)
    y = np.zeros(n)
    blk = 480
    for s in range(0, n, blk):
        fc = 700 + 900 * (.5 + .5 * math.sin(2 * math.pi * 2.2 * s / SR)) * wah + 500 * (1 - wah)
        seg = x[max(0, s - 2400):s + blk]
        y[s:s + blk] = S.bp(seg, fc * .5, fc * 1.8)[-min(blk, n - s):]
    return y * env(n, .04, .08, .85, .08)


def strings(ms, dur, att=.15, cutoff=2600):
    n = int(dur * SR)
    tt = S.tt(n)
    x = np.zeros(n)
    for k, m in enumerate(ms):
        for dt in (-.005, .0, .006):
            x += S.osc("saw", hz(m) * (1 + dt) * (1 + .004 * np.sin(2 * np.pi * 5.2 * tt + k)), n, phase=.17 * k)
    return S.lp(x, cutoff) * env(n, att, .2, .85, min(.4, dur / 3))


def timp(m, dur=.8, roll=0.0):
    n = int(dur * SR)
    tt = S.tt(n)
    f = hz(m) * (1 + .15 * np.exp(-tt * 25))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 3.5) + .2 * S.lp(S.noise(n, int(m)), 400) * np.exp(-tt * 12)
    if roll > 0:  # a roll: dense retriggered strokes, crescendo
        x = np.sin(2 * np.pi * np.cumsum(f * 0 + hz(m)) / SR) * (.5 + .5 * np.abs(np.sin(2 * np.pi * 9 * tt))) * np.linspace(.2, 1, n) ** roll
        x += .3 * S.lp(S.noise(n, 5), 300) * np.linspace(.2, 1, n)
    return x


def celesta(m, dur=.6):
    n = int(dur * SR)
    tt = S.tt(n)
    f = hz(m)
    return (np.sin(2 * np.pi * f * tt) + .3 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt * 6)) * np.exp(-tt * 3.5) * np.clip(tt / .003, 0, 1)


def piccolo(m, dur):
    n = int(dur * SR)
    f = hz(m) * (1 + .006 * np.sin(2 * np.pi * 6 * S.tt(n)))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) + .05 * S.bp(S.noise(n, m), 2000, 8000)) * env(n, .02, .05, .8, .04)


def harp_gliss(m0, m1, dur, seed=0):
    out = np.zeros(int((dur + .8) * SR))
    scale = [0, 2, 4, 5, 7, 9, 11]
    notes = [m for m in range(min(m0, m1), max(m0, m1) + 1) if (m % 12) in scale]
    if m1 < m0:
        notes = notes[::-1]
    for k, m in enumerate(notes):
        x = pizz(m, .7, seed + k) * .7
        i = int(k / len(notes) * dur * SR)
        out[i:i + len(x)] += x[:len(out) - i]
    return out


class Bed:
    def __init__(self, seconds):
        self.n = int(seconds * SR)
        self.st = {}

    def put(self, stem, x, t, g=1.0):
        if stem not in self.st:
            self.st[stem] = np.zeros(self.n)
        S.put(self.st[stem], x, t, g)

    def seq(self, stem, inst, notes, t0, bpm, g=1.0, **kw):
        """notes: (beat, midi or tuple, beats)."""
        b = 60 / bpm
        for beat, m, d in notes:
            if m is None:
                continue
            if isinstance(m, tuple):
                x = inst(m, d * b, **kw)
            else:
                x = inst(m, d * b * .95, **kw)
            self.put(stem, x, t0 + beat * b, g)


# ------------------------------------------------------------------------------------------------ themes
PIP_MEL = [(0, 53, .5), (.5, 57, .5), (1, 60, .5), (1.5, 57, .5), (2, 58, .5), (2.5, 55, .5), (3, 52, .5), (3.5, 48, .5),
           (4, 53, .5), (4.5, 57, .5), (5, 60, .5), (5.5, 65, .5), (6, 64, .5), (6.5, 60, .5), (7, 53, 1.0)]
PIP_BASS = [(0, 41, .5), (1, 48, .5), (2, 36, .5), (3, 43, .5), (4, 41, .5), (5, 48, .5), (6, 36, .5), (7, 41, .5)]
BRAM_MEL = [(0, 50, 1), (1, 53, 1), (2, 58, 2), (4, 58, 1), (5, 60, 1), (6, 62, 1), (7, 60, 1), (8, 58, 2), (10, 53, 1), (11, 55, 1),
            (12, 53, 1), (13, 50, 2)]


def pip_theme(bed, t0, bpm=132, g=1.0, reps=1, wind=True):
    b = 60 / bpm
    for r in range(reps):
        t = t0 + r * 8 * b
        if wind:
            bed.seq("wind", bassoon, PIP_MEL, t, bpm, .9 * g)
        bed.seq("bass", lambda m, d: pizz(m, .45, m), PIP_BASS, t, bpm, 1.0 * g)
        for k in range(8):   # off-beat pizz chords
            ch = (65, 69, 72) if k % 4 in (0, 1) else (64, 67, 70)
            for j, m in enumerate(ch):
                bed.put("pizz", pizz(m, .25, 10 + j), t + (k + .5) * b, .35 * g)
    return t0 + reps * 8 * b


def bram_theme(bed, t0, bpm=100, g=1.0, bars=4, melody=True):
    b = 60 / bpm
    for k in range(bars * 3):   # 3/4 oompah
        beat = k % 3
        t = t0 + k * b
        if beat == 0:
            bed.put("bass", tuba(34 if (k // 3) % 2 == 0 else 41, b * .9), t, 1.0 * g)
        else:
            for m in (62, 65, 70):
                bed.put("brass", brass((m,), b * .45, 1200), t, .18 * g)
    if melody:
        bed.seq("trom", trombone, BRAM_MEL[:max(1, int(len(BRAM_MEL) * bars / 5))], t0, bpm, .8 * g, slide=0.0, wah=.3)
    return t0 + bars * 3 * b


def wah_wah(bed, t0, g=1.0):
    for k, (m, d) in enumerate(((58, .42), (57, .42), (56, .42), (55, 1.1))):
        bed.put("trom", trombone(m, d, slide=-1 if k == 3 else 0, wah=1.0), t0 + k * .44, g)


def sting(bed, t0, chord=(60, 64, 67, 72), g=1.0, cym=True):
    bed.put("brass", brass(chord, .6), t0, g)
    if cym:
        bed.put("cym", cymbal(1.2, int(t0 * 10)), t0, .7 * g)
    bed.put("perc", S.snare(.2, 3), t0, .5 * g)


# ------------------------------------------------------------------------------------------------ the score
def score():
    tl, total = SL.timeline()
    T = {sid: (f / SL.FPS, n / SL.FPS) for sid, f, n in tl}
    end0 = total / SL.FPS - SL.END_FRAMES / SL.FPS
    bed = Bed(total / SL.FPS + .5)

    # title: a bright fanfare and an upward xylophone run, landing on Pip's theme pickup
    title_fanfare(bed, 0.0)

    # s01: Pip's strut, thinned under his line, tidy pizz button at the end
    t, d = T["s01"]
    e = pip_theme(bed, t + .1, 132, 1.0, reps=1)
    bed.seq("wind", clarinet, [(0, 72, .5), (.5, 69, .5), (1, 65, 1)], e + .1, 132, .15)
    bed.put("pizz", pizz(77, .3, 3), t + d - .7, .8)
    bed.put("xylo", xylo(89, .3), t + d - .7, .5)

    # s02: Bramble's oompah, a big flop hit, a wriggly xylophone figure
    t, d = T["s02"]
    bram_theme(bed, t + .1, 100, 1.0, bars=3)
    flop = t + 5.6
    bed.put("perc", timp(38, 1.0), flop, 1.0)
    bed.put("cym", cymbal(1.4, 3), flop, .6)
    bed.put("bass", tuba(29, .5), flop, 1.0)
    for k in range(14):
        bed.put("xylo", xylo(77 + (k % 4) * 2 - (k % 2) * 5, .2), flop + .6 + k * .14, .45)
    bed.seq("trom", trombone, [(0, 58, 1.5), (2, 62, 1.5), (4, 65, 3)], flop + 1.2, 100, .6, wah=.4)

    # s03: suspense tremolo, a creeping woodwind line as the cloud rolls in, near-silence, plinks for the blinks, a muted wah
    t, d = T["s03"]
    bed.put("str", strings_trem((52, 55, 58, 61), 3.0), t + .2, .55)
    bed.seq("wind", clarinet, [(k * .5, 64 - k, .45) for k in range(8)], t + .5, 120, .16)
    bed.put("whistle", slide_whistle(76, 64, 1.4, 1.2) * .5, t + 2.0, .3)
    bed.put("pizz", pizz(84, .2, 4), t + 5.2, .5)
    bed.put("pizz", pizz(84, .2, 5), t + 5.7, .5)
    bed.put("trom", trombone(52, .5, wah=1.0), t + 6.6, .5)

    # s04: the slow burn: a low pedal, a heartbeat on timpani, a chromatic string climb, then a brass "That does it!" stab
    t, d = T["s04"]
    bed.put("bass", tuba(29, d - 1.4), t + .1, .3)
    for k in range(int((d - 2.0) / .75)):
        bed.put("perc", timp(33 + (k // 4), .5), t + .3 + k * .75, .35 + .05 * k)
    for k in range(7):
        bed.put("str", strings((52 + k, 58 + k), .95, att=.2), t + .6 + k * 1.0, .3 + .05 * k)
    sting(bed, t + d - 2.0, (49, 55, 58, 61, 67), 1.0)
    bed.put("perc", timp(37, 1.2), t + d - 2.0, 1.0)

    # s05: a frantic galop for the sweeping, Bramble's tuba barges in, then a wah-wah-wah-waaah as the dust blows back
    t, d = T["s05"]
    bpm = 168
    b = 60 / bpm
    for k in range(int(5.6 / b)):
        bed.put("xylo", xylo((72, 76, 79, 76, 74, 77, 81, 77)[k % 8], .15), t + .1 + k * b / 2, .75)
        bed.put("xylo", xylo((72, 76, 79, 76, 74, 77, 81, 77)[(k + 4) % 8] - 12, .15), t + .1 + k * b / 2 + b / 4, .25)
        if k % 2 == 0:
            bed.put("bass", pizz(41 if (k // 4) % 2 == 0 else 36, .25, k), t + .1 + k * b / 2, .8)
            bed.put("perc", S.snare(.08, k, brush=True), t + .1 + k * b / 2, .3)
    bram_theme(bed, t + 5.8, 100, .9, bars=2, melody=False)
    bed.seq("trom", trombone, [(0, 58, 1), (1, 62, 1), (2, 65, 2)], t + 5.9, 100, .7, wah=.3)
    wah_wah(bed, t + d - 2.6, .9)

    # s06: a pouring harp glissando, a mud "squelch" on tuba, a shaky woodwind tremolo, pizz splats, a deadpan low plunk
    t, d = T["s06"]
    bed.put("pizz", harp_gliss(84, 60, 1.6, 20), t + .5, .6)
    bed.put("wind", bassoon(41, .5, bend=-2), t + 3.0, .9)
    bed.put("bass", tuba(31, .5), t + 3.05, .8)
    for k in range(28):
        bed.put("wind", clarinet(67 + (k % 2) * 3, .05), t + 4.6 + k * .055, .18)
    for k in range(9):
        bed.put("pizz", pizz(60 + (k * 7) % 19, .2, 30 + k), t + 5.4 + k * .17, .55)
    bed.put("wind", bassoon(36, .6), t + d - 1.6, .9)

    # s07: the storm rises: a timpani roll crescendo under low brass climbing chromatically, a big chord on the take,
    # then Bramble's theme turned triumphant
    t, d = T["s07"]
    bed.put("perc", timp(31, 6.0, roll=1.5), t + .3, .9)
    for k in range(6):
        bed.put("brass", brass((36 + k, 43 + k), 1.0, 900), t + .6 + k * 1.0, .45 + .06 * k)
    sting(bed, t + 6.6, (46, 53, 58, 62, 65, 70), 1.1)
    bed.put("str", strings_trem((58, 62, 65, 70), 2.4), t + 6.8, .5)
    bram_theme(bed, t + 9.0, 112, .9, bars=1, melody=False)
    bed.seq("trom", trombone, [(0, 58, .5), (.5, 62, .5), (1, 65, 1), (2, 70, 1.5)], t + 9.0, 112, .8, wah=.2)

    # s08: the storm: whirling chromatic string runs, cymbal swells, a "clack" for the ball, timpani; fades into wind
    t, d = T["s08"]
    for k in range(12):
        up = k % 2 == 0
        run = [(j * .12, (60 + j) if up else (75 - j), .12) for j in range(15)]
        bed.seq("str", lambda m, dd: strings((m, m + 12), dd, att=.01, cutoff=3200), run, t + .2 + k * .62, 60, .6 * (1 - k / 16))
    bed.put("perc", woodblock(600, .12), t + 1.6, .9)
    bed.put("whistle", slide_whistle(84, 60, .5, .8), t + 1.5, .35)
    for k in range(3):
        bed.put("cym", cymbal(2.5, 40 + k)[::-1] * .8, t + .5 + k * 2.6, .45)
        bed.put("perc", timp(33, 1.0), t + 2.9 + k * 2.6, .8)
    bed.put("brass", brass((37, 44, 49, 52), 3.0, 700), t + 4.0, .9)
    bed.put("perc", timp(31, 5.0, roll=1.0), t + .2, .7)

    # s09: aftermath: celesta sparkle over soft strings, a harp glissando and a warm major chord for "spotless!"
    t, d = T["s09"]
    bed.put("str", strings((48, 55, 64), 5.0, att=1.2, cutoff=1800), t + .2, .45)
    for k in range(10):
        bed.put("xylo", celesta((72, 76, 79, 84, 79, 76, 74, 77, 81, 86)[k], .7), t + .6 + k * .42, .5)
    bed.put("pizz", harp_gliss(60, 91, 1.2, 50), t + d - 5.2, .7)
    bed.put("str", strings((53, 60, 65, 69, 72), 4.0, att=.3, cutoff=2400), t + d - 4.2, .55)
    bed.put("xylo", celesta(89, 1.2), t + d - 4.1, .5)

    # s10: friendship waltz: both themes together, the tuba waltzing under a sweet clarinet Pip theme, woodblock thumps
    t, d = T["s10"]
    bpm = 108
    b = 60 / bpm
    for k in range(int((d - .6) / b)):
        beat = k % 3
        tk = t + .2 + k * b
        if beat == 0:
            bed.put("bass", tuba(34 if (k // 3) % 2 == 0 else 41, b * .9), tk, .6)
        else:
            for m in (62, 65, 70):
                bed.put("pizz", pizz(m, .25, k + m), tk, .25)
    bed.seq("wind", clarinet, [(0, 65, 1), (1, 69, 1), (2, 72, 1), (3, 70, 2), (5, 67, 1), (6, 65, 1), (7, 69, 1), (8, 74, 2),
                               (10, 72, 1), (11, 70, 1), (12, 69, 1), (13, 67, 1), (14, 65, 2)], t + .2, bpm, .16)
    for k in range(5):
        bed.put("perc", woodblock(420, .1), t + 6.2 + k * .28, .6)

    # s11: the button: Pip's strut (stops for his line), a brass AH-CHOO stab, a deadpan wah-wah and a little tag
    t, d = T["s11"]
    pip_theme(bed, t + .1, 132, .9, reps=1)
    bed.put("whistle", slide_whistle(67, 79, .9, .7), t + 4.0, .3)
    sting(bed, t + 5.0, (58, 61, 64, 67, 70), 1.0)
    bed.put("perc", timp(34, .8), t + 5.0, .8)
    wah_wah(bed, t + 5.8, .7)

    # end card: an original three-chord tag
    end_tag(bed, end0)
    out = master(bed)
    return out[:int(round(total / SL.FPS * SR))]


def title_fanfare(bed, t0):
    bed.put("perc", timp(41, .6), t0 + .05, .9)
    for k, (ch, dur) in enumerate((((65, 69, 72), .18), ((65, 69, 72), .18), ((67, 70, 74, 77), .5), ((65, 69, 72, 77), 1.2))):
        bed.put("brass", brass(ch, dur, 2600), t0 + .1 + (0, .22, .44, 1.0)[k], 1.0)
    bed.put("cym", cymbal(1.6, 2), t0 + 1.0, .7)
    for k in range(12):
        bed.put("xylo", xylo(65 + (0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19)[k], .25), t0 + .1 + k * .07, .5)
    bed.put("pizz", harp_gliss(65, 96, .6, 70), t0 + 1.0, .5)
    bed.seq("wind", bassoon, [(0, 48, .5), (.5, 50, .5), (1, 52, .5)], t0 + 2.2, 132, .8)


def end_tag(bed, t0):
    for k, (ch, dur) in enumerate((((60, 64, 67), .25), ((62, 65, 69), .25), ((65, 69, 72, 77), 1.3))):
        bed.put("brass", brass(ch, dur, 2400), t0 + .2 + k * .3, .9)
    bed.put("cym", cymbal(1.4, 4), t0 + .8, .6)
    bed.put("perc", timp(41, .8), t0 + .8, .8)
    bed.put("bass", tuba(29, 1.0), t0 + .8, .8)
    bed.put("xylo", xylo(89, .5), t0 + .8, .5)


GAINS = dict(wind=.24, bass=.26, pizz=.16, xylo=.18, brass=.26, str=.18, perc=.26, cym=.09, whistle=.12, trom=.22)


def master(bed):
    stems = []
    for k, x in bed.st.items():
        stems.append((x, GAINS.get(k, .2), dict(wind=-.25, bass=-.1, pizz=.2, xylo=.3, brass=0, str=-.2, perc=.05, cym=.15, whistle=.25,
                                               trom=.15).get(k, 0), .18))
    # tsound.master normalises each stem to its own peak, so the gains are relative levels
    mix = S.master(stems, wet=.2, rev_s=1.4, drive=1.15, fade_out=.6)
    return (mix * (10 ** (-3 / 20)) / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)


if __name__ == "__main__":
    import sys
    import soundfile as sf
    a = score()
    sf.write(sys.argv[1], a, SR, subtype="PCM_24")
    print(a.shape, a.shape[0] / SR)
