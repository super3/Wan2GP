"""bold_shapes sound: a 5 s, all-synthesised cue for Squeaky's shot.

Music: a 60s spy-lounge vamp (walking upright bass, brushes, a twangy minor-major-7 guitar motif, vibraphone, a brass stab on
the swipe, a vibes-and-rimshot button ending).  SFX: high-altitude wind, rope creaks, slide-whistle wind-up, whoosh, squeegee
squeak, sparkle chimes, bubble plips, a 'boing' hop and a wooden landing.  Voice: formant-synthesised kid gibberish whose
syllables (draw.SYL) also drive the mouth flaps."""
import math

import numpy as np
from scipy.signal import lfilter

import tsound as S
from draw import SYL, DUR, FPS, T_IMPACT, T_SWIPE

SR = S.SR
BEAT = 60 / 138


def env_exp(n, k):
    return np.exp(-S.tt(n) * k)


def bass_note(m, dur, g=1.0):
    n = int(dur * SR)
    t = S.tt(n)
    f = S.hz(m)
    x = np.sin(2 * np.pi * f * t) + .35 * np.sin(4 * np.pi * f * t) + .12 * S.lp(S.osc("saw", f, n), 900)
    att = np.clip(t / .006, 0, 1)
    thump = S.lp(S.noise(n, int(m)), 400) * np.exp(-t * 60) * .3
    return (x * att * np.exp(-t * 3.2) + thump) * g


def vibes(notes, dur, seed=0):
    n = int(dur * SR)
    t = S.tt(n)
    out = np.zeros(n)
    trem = 1 + .35 * np.sin(2 * np.pi * 5.5 * t)
    for m in notes:
        f = S.hz(m)
        out += (np.sin(2 * np.pi * f * t) + .25 * np.sin(2 * np.pi * f * 4.0 * t) * np.exp(-t * 9)) * np.exp(-t * 1.6)
    return out * trem * np.clip(t / .003, 0, 1)


def twang(m, dur, seed):
    """A twangy surf-guitar note: plucked string, soft drive, quick vibrato from the bar."""
    x = S.pluck(S.hz(m), dur, seed=seed, damp=.997)
    n = len(x)
    x = np.tanh(x * 2.2) * .8
    x = S.bp(x, 180, 4200)
    tr = 1 + .25 * np.sin(2 * np.pi * 7 * S.tt(n))
    return x * tr


def brass(notes, t0, dur, buf, g=1.0):
    for m in notes:
        n = int(dur * SR)
        t = S.tt(n)
        f = S.hz(m) * (1 + .004 * np.sin(2 * np.pi * 5 * t))
        x = S.osc("saw", f * 1.003, n) + S.osc("saw", f * .997, n)
        cut = 900 + 3200 * np.exp(-t * 7)
        y = np.zeros(n)
        for s0 in range(0, n, 1024):  # a swept low-pass: the 'bwah'
            seg = x[max(0, s0 - 1024):s0 + 1024]
            yy = S.lp(seg, float(cut[s0]))[-min(1024, n - s0):]
            y[s0:s0 + len(yy)] = yy
        S.put(buf, y * S.adsr(n, .012, .12, .55, .18), t0, g)


def formant_voice(syl):
    """Kid gibberish: a glottal saw source through three vowel formants, with consonant noise onsets."""
    F = {"a": (1050, 1550, 3000), "i": (420, 2900, 3600), "o": (620, 1050, 2800), "u": (430, 1150, 2700), "e": (620, 2450, 3200)}
    out = S.buf(DUR)
    for k, (t0, dur, v, f0, cons) in enumerate(syl):
        n = int(dur * SR)
        t = S.tt(n)
        contour = f0 * (1 + .12 * np.sin(np.pi * t / dur) - .1 * (t / dur) ** 2) * (1 + .012 * np.sin(2 * np.pi * 6.5 * t))
        src = S.osc("saw", contour, n) + .3 * S.noise(n, 300 + k) * .15
        y = np.zeros(n)
        for fc, g in zip(F[v], (1.0, .55, .25)):
            bw = fc * .12
            y += S.bp(src, fc - bw, fc + bw, order=2) * g
        y += .15 * S.lp(src, 500)
        e = np.clip(t / .015, 0, 1) * np.clip((dur - t) / .04, 0, 1)
        y *= e
        if cons:  # consonant onset
            nc = int(.03 * SR)
            band = {"p": (300, 2500), "d": (500, 4000), "k": (1500, 5000), "sh": (2000, 7000), "w": (200, 800), "h": (1000, 6000)}[cons]
            c = S.bp(S.noise(nc, 500 + k), *band) * np.exp(-S.tt(nc) * (60 if cons != "sh" else 20))
            S.put(out, c, max(t0 - .025, 0), .5 if cons != "sh" else .8)
        S.put(out, y, t0, 1.0)
    return out


def sound():
    mus_bass, mus_kit, mus_gtr, mus_vib, mus_brass = (S.buf(DUR) for _ in range(5))
    sfx, amb, voice = S.buf(DUR), S.buf(DUR), S.buf(DUR)
    t_imp = T_IMPACT / FPS
    t_sw = T_SWIPE / FPS
    # --- music: intro vamp (0 - 0.6 s), wind-up, stab, light groove under the voice, bongo fill, button
    for b, m in enumerate((40, 43, 45, 46)):  # walking bass E G A Bb... resolves to the stab
        S.put(mus_bass, bass_note(m, BEAT * .95), .02 + b * BEAT * .5, .9)
    S.put(mus_bass, bass_note(28, 1.0), t_imp, 1.2)
    for b, m in enumerate((40, 47, 45, 43, 40, 38, 40)):  # groove resumes under the voice, softer
        S.put(mus_bass, bass_note(m, BEAT * .9), 1.85 + b * BEAT * .5, .5)
    S.put(mus_bass, bass_note(40, 1.2), 4.25, 1.0)
    swing = []  # brushes and ride: swung eighths
    t = .02
    while t < 4.2:
        swing += [t, t + BEAT * .33]
        t += BEAT * .5
    for j, tb in enumerate(swing):
        if t_sw - .35 < tb < t_imp + .45 or 3.3 < tb < 4.05:
            continue
        S.put(mus_kit, S.hat(.05, seed=j), tb, .25 if j % 2 else .4)
    for tb in (BEAT * .5, BEAT * 1.5, 1.85 + BEAT * .5, 1.85 + BEAT * 1.5, 1.85 + BEAT * 2.5):
        S.put(mus_kit, S.snare(.3, seed=8, brush=True), tb, .6)
    for j, m in enumerate((52, 55, 59, 63)):  # the spy motif: E minor-major-7 arpeggio on a twangy guitar
        S.put(mus_gtr, twang(m, .7, 20 + j), .04 + j * .11, .8)
    S.put(mus_gtr, twang(64, 1.0, 30), .04 + 4 * .11, .7)
    S.put(mus_vib, vibes((64, 67, 71, 75), 1.6), .0, .35)
    S.put(mus_vib, vibes((76, 79, 83, 86), 1.4), t_imp + .1, .5)  # sparkle shimmer
    S.put(mus_vib, vibes((64, 67, 71, 74, 78), 1.0), 4.25, .9)  # button
    brass((40, 47, 52, 55, 59), t_imp, .55, mus_brass, .9)  # BWAH on the impact
    brass((40, 47, 52, 55, 62), 4.25, .6, mus_brass, .55)
    for k, (tb, f) in enumerate(((3.36, 180), (3.44, 220), (3.52, 260), (3.6, 200), (3.68, 300), (3.8, 160))):  # bongo fill
        S.put(mus_kit, S.tom(f, .25), tb, .55)
    rim = S.bp(S.noise(int(.06 * SR), 77), 1500, 6000) * env_exp(int(.06 * SR), 70) + .6 * np.sin(2 * np.pi * 850 * S.tt(int(.06 * SR))) * env_exp(int(.06 * SR), 50)
    S.put(mus_kit, rim, 4.25, 1.0)
    # --- SFX
    n = int(DUR * SR)  # high-altitude wind: slow-moving band of noise, very low
    w = S.bp(S.noise(n, 2), 200, 1400)
    amb += w * (.6 + .4 * np.sin(2 * np.pi * .3 * S.tt(n) + 1))
    toot = S.lp(S.osc("square", 392, int(.18 * SR), pw=.4) + S.osc("square", 466, int(.18 * SR), pw=.4), 1200) * S.adsr(int(.18 * SR), .01, .05, .8, .05)
    S.put(amb, toot, .55, .08)  # a far-off car horn, 100 floors down
    S.put(sfx, S.slide("sine", 500, 1500, t_sw - .55, curve=1.6) * np.linspace(.2, 1, int((t_sw - .55) * SR)) ** 2, .55, .25)  # slide-whistle wind-up
    S.put(sfx, S.whoosh(.32, 600, 5000, seed=4), t_sw - .1, 1.0)
    nq = int(.34 * SR)  # the squeegee SQUEAK
    tq = S.tt(nq)
    fq = 1700 + 900 * np.sin(np.pi * tq / .34) + 80 * np.sin(2 * np.pi * 38 * tq)
    sq = (np.sin(2 * np.pi * np.cumsum(fq) / SR) + .4 * np.sin(4 * np.pi * np.cumsum(fq) / SR)) * np.clip(tq / .01, 0, 1) * np.clip((.34 - tq) / .06, 0, 1)
    S.put(sfx, sq * (1 + .5 * np.sign(np.sin(2 * np.pi * 45 * tq))), t_sw, .55)
    for k, tb in enumerate((t_imp + .08, t_imp + .2, t_imp + .34)):  # sparkle 'ting-a-ling'
        nb = int(.6 * SR)
        f = (2637, 3136, 3951)[k]
        tt_ = S.tt(nb)
        S.put(sfx, (np.sin(2 * np.pi * f * tt_) + .3 * np.sin(2 * np.pi * f * 2.76 * tt_)) * np.exp(-tt_ * 7), tb, .4)
    for k in range(6):  # bubble plips at each pop (same 5 s loop as the picture)
        tp = (2.7 - .35 - k * DUR / 6) % DUR
        tp = math.floor(tp * FPS / 2) * 2 / FPS + .02
        npl = int(.05 * SR)
        S.put(sfx, S.slide("sine", 900, 2200, .05) * env_exp(npl, 60), tp, .22)
    nb = int(.5 * SR)  # 'boing' on the hop
    tb_ = S.tt(nb)
    fb = 180 * (1 + .5 * np.exp(-tb_ * 5) * np.sin(2 * np.pi * 14 * tb_)) + 120 * tb_
    S.put(sfx, np.sin(2 * np.pi * np.cumsum(fb) / SR) * np.exp(-tb_ * 4.5), 3.42, .6)
    nl = int(.25 * SR)  # wooden landing knock + rope creak
    S.put(sfx, S.tom(140, .25) * .8 + S.bp(S.noise(nl, 9), 300, 2000) * env_exp(nl, 30), 3.83, .55)
    for tc, f in ((1.55, 340), (3.95, 300), (4.45, 280)):
        nc = int(.35 * SR)
        tcr = S.tt(nc)
        cr = S.bp(S.noise(nc, int(f)), f, f * 3) * (1 + np.sign(np.sin(2 * np.pi * (60 + 30 * tcr) * tcr))) * np.sin(np.pi * tcr / .35)
        S.put(sfx, cr, tc, .2)
    voice[:] = formant_voice(SYL)
    stems = [(mus_bass, .30, -.15, .15), (mus_kit, .17, .2, .2), (mus_gtr, .22, -.35, .35), (mus_vib, .20, .3, .45),
             (mus_brass, .30, 0, .3), (sfx, .40, .15, .2), (amb, .05, 0, 0), (voice, .46, 0, .12)]
    return S.master(stems, wet=.22, rev_s=1.3, drive=1.25, fade_out=.25)
