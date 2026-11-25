"""Shared synth toolkit for the cartoon prototypes: oscillators, envelopes, a plucked string, drums, effects and a master bus.
All sound is computed from maths and fixed seeds."""
import math

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 48000


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def buf(seconds):
    return np.zeros(int(seconds * SR))


def tt(n):
    return np.arange(n) / SR


def adsr(n, a=.01, d=.08, s=.7, r=.1):
    t = tt(n)
    dur = n / SR
    e = np.where(t < a, t / max(a, 1e-6), np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-6), s))
    rel = np.clip((dur - t) / max(r, 1e-6), 0, 1)
    return e * rel


def osc(kind, f, n, phase=0.0, pw=.5):
    """f: Hz (scalar or array per sample)."""
    ph = (np.cumsum(np.broadcast_to(f, (n,)) / SR) + phase) % 1.0
    if kind == "sine":
        return np.sin(2 * np.pi * ph)
    if kind == "square":
        return np.where(ph < pw, 1.0, -1.0)
    if kind == "saw":
        return 2 * ph - 1
    if kind == "tri":
        return 4 * np.abs(ph - .5) - 1
    raise ValueError(kind)


def lp(x, f, order=2):
    return sosfilt(butter(order, min(f, SR / 2 - 100) / (SR / 2), "low", output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f / (SR / 2), "high", output="sos"), x)


def bp(x, f0, f1, order=2):
    return sosfilt(butter(order, [f0 / (SR / 2), min(f1, SR / 2 - 100) / (SR / 2)], "band", output="sos"), x)


def put(dst, x, t0, g=1.0):
    i = int(t0 * SR)
    if i >= len(dst):
        return
    n = min(len(x), len(dst) - i)
    dst[i:i + n] += x[:n] * g


def noise(n, seed):
    return np.random.default_rng(seed).normal(0, 1, n)


def pluck(f, dur, seed=0, damp=.996, bright=1.0):
    n, p = int(dur * SR), max(2, int(round(SR / f)))
    rng = np.random.default_rng(seed)
    b = rng.uniform(-1, 1, p)
    if bright < 1:
        b = np.convolve(b, np.ones(3) / 3, mode="same")
    out = np.empty(n)
    for i in range(n):
        out[i] = b[i % p]
        b[i % p] = damp * .5 * (b[i % p] + b[(i + 1) % p])
    return out


def kick(dur=.35, f0=110, f1=42):
    n = int(dur * SR)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def snare(dur=.22, seed=3, tone=190, brush=False):
    n = int(dur * SR)
    t = tt(n)
    nz = bp(noise(n, seed), 1500, 9000) * np.exp(-t * (14 if brush else 22))
    if brush:  # a swish: slow attack, no tone
        return nz * np.clip(t / .03, 0, 1)
    return nz + .5 * np.sin(2 * np.pi * tone * t) * np.exp(-t * 30)


def hat(dur=.06, seed=5):
    n = int(dur * SR)
    return hp(noise(n, seed), 7000) * np.exp(-tt(n) * 60)


def tom(f=90, dur=.5):
    n = int(dur * SR)
    t = tt(n)
    ff = f * (1 + .6 * np.exp(-t * 20))
    return np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-t * 6)


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


def slide(kind, f0, f1, dur, curve=1.0):
    n = int(dur * SR)
    x = np.linspace(0, 1, n) ** curve
    return osc(kind, f0 * (f1 / f0) ** x, n)


def reverb(x, seconds=1.4, seed=3, damp=5000):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-np.arange(n) / (SR * seconds / 6.9))[:, None]
    ir = lfilter(*butter(1, damp / (SR / 2)), ir, axis=0)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))
    return np.stack([fftconvolve(x, ir[:, c])[:len(x)] for c in range(2)], 1)


def pan(x, p):
    a = (p + 1) * math.pi / 4
    return np.stack([x * math.cos(a), x * math.sin(a)], 1) * math.sqrt(2)


def master(stems, wet=.25, rev_s=1.4, drive=1.3, fade_out=.35):
    """stems: list of (mono signal, gain, pan, reverb send).  Returns stereo float32 normalised to -1 dBFS."""
    n = max(len(s[0]) for s in stems)
    dry, send = np.zeros((n, 2)), np.zeros(n)
    for x, g, p, r in stems:
        x = np.pad(x, (0, n - len(x)))
        x = x / (np.max(np.abs(x)) + 1e-9) * g
        dry += pan(x, p)
        send += x * r
    mix = dry + reverb(send, rev_s) * wet
    t = tt(n)
    mix *= np.clip(t / .03, 0, 1)[:, None] * np.clip((n / SR - t) / fade_out, 0, 1)[:, None]
    mix = np.tanh(mix * drive) / np.tanh(drive)
    return (mix * .89 / (np.max(np.abs(mix)) + 1e-9)).astype(np.float32)
