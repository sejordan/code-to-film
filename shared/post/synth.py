"""Shared tools for making a soundtrack from arithmetic: no recordings.

A film's own sound.py builds its layers out of these (filters, a room echo,
wooden knocks, a piano note) and hands back one stereo mix; finish() fades,
levels and saves it. shared/post/soundtrack.py is the command that runs it.

Sound is a numpy array: one row per ear, SR samples a second. Everything
random draws from the one generator here, rng, so a soundtrack comes out the
same every time it is made.
"""
import math
import wave

import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def db(x):
    return 10.0 ** (x / 20.0)


def seconds(film, tail=0.0):
    """Length of the film's picture, plus a tail of sound over black."""
    return film.total_frames() / film.FPS + tail


def blank(film, tail=0.0):
    """Silence as long as the film and its tail, to add layers to."""
    return np.zeros((2, int(seconds(film, tail) * SR)))


# ---------------------------------------------------------------- tools

def shape(x, lo=None, hi=None, soft=0.3):
    """Keep the band between lo and hi Hz, with soft shoulders."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1.0 / (1.0 + (lo / np.maximum(f, 1e-3)) ** (2.0 / soft))
    if hi:
        g *= 1.0 / (1.0 + (f / hi) ** (2.0 / soft))
    return np.fft.irfft(X * g, len(x))


def hann(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, math.tau, max(n, 2)))


def put(mix, sig, t, gain=1.0, pan=0.0):
    """Add a mono sound to the stereo mix at time t. pan: -1 left, 1 right."""
    n = mix.shape[1]
    i = int(t * SR)
    if i >= n or i + len(sig) <= 0:
        return
    sig = sig[max(0, -i):n - i]
    i = max(i, 0)
    a = (pan + 1.0) * math.pi / 4.0
    mix[0, i:i + len(sig)] += sig * gain * math.cos(a)
    mix[1, i:i + len(sig)] += sig * gain * math.sin(a)


def conv(x, ir):
    n = len(x) + len(ir) - 1
    m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]


def room_echo(stereo, seconds=1.6, wet=0.2, dark=3500.0, seed=3):
    """A soft room around the sound: each side through its own decaying hiss."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.empty_like(stereo)
    for c in range(2):
        ir = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir[:int(0.012 * SR)] *= 0.0                     # a short gap before the room answers
        ir = shape(ir, hi=dark)
        ir /= np.sqrt((ir ** 2).sum())
        out[c] = stereo[c] + wet * conv(stereo[c], ir)
    return out


def tok(f0=900.0, length=0.07, bright=1.0):
    """Wood knocking on wood: two damped tones and a tick of hiss at the start."""
    n = int(length * SR)
    t = np.arange(n) / SR
    y = np.sin(math.tau * f0 * t) * np.exp(-t / (length * 0.22))
    y += 0.45 * bright * np.sin(math.tau * f0 * 2.76 * t) * np.exp(-t / (length * 0.10))
    click = rng.standard_normal(n) * np.exp(-t / 0.0015)
    y += 0.35 * bright * shape(click, lo=1500.0, hi=7000.0)
    return y * np.minimum(1.0, t / 0.0006)


def thud(f0=110.0, length=0.22):
    n = int(length * SR)
    t = np.arange(n) / SR
    return np.sin(math.tau * f0 * t * (1.0 - 0.25 * t / length)) * np.exp(-t / (length * 0.25)) \
        * np.minimum(1.0, t / 0.002)


def creak(f0=520.0, f1=760.0, length=0.16):
    """A small wooden squeak, for the neck."""
    n = int(length * SR)
    t = np.arange(n) / SR
    f = f0 + (f1 - f0) * (t / length) ** 0.7
    ph = math.tau * np.cumsum(f) / SR
    y = np.sin(ph + 0.6 * np.sin(math.tau * 43.0 * t)) + 0.3 * np.sin(2.0 * ph)
    return shape(y * hann(n), lo=300.0, hi=3000.0)


def piano_note(freq, length, vel=0.7):
    """A soft piano note: a few partials, each dying away at its own pace,
    two strings a hair apart, and the knock of the hammer."""
    n = int(length * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    tau = 1.9 * (261.6 / freq) ** 0.45
    for det in (-0.0006, 0.0007):
        for k in range(1, 9):
            fk = freq * (1.0 + det) * k * math.sqrt(1.0 + 0.00035 * k * k)
            if fk > 9000.0:
                break
            amp = (1.0 / k ** (2.1 - 0.5 * vel))
            y += amp * np.sin(math.tau * fk * t + rng.uniform(0, math.tau)) * np.exp(-t / (tau / (1.0 + 0.45 * (k - 1))))
    knock = shape(rng.standard_normal(n) * np.exp(-t / 0.006), lo=200.0, hi=2500.0)
    y = 0.5 * y + 0.25 * knock / (np.abs(knock).max() + 1e-9)
    y *= np.minimum(1.0, t / 0.006) * np.minimum(1.0, (length - t) / 0.25)
    return shape(y, hi=4200.0) * vel


def _partials(freqs, n, detune, vib, weight, seed):
    """Sum of harmonic partials for each note, each note as several slightly
    detuned voices with their own slow vibrato. weight(k, fk) is a
    partial's level: an array over time, or a number."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f in freqs:
        for d in detune:
            inst = f * (1.0 + d) * (1.0 + vib * np.sin(math.tau * r.uniform(4.6, 5.6) * t + r.uniform(0, math.tau)))
            ph = math.tau * np.cumsum(inst) / SR
            for k in range(1, 40):
                fk = f * k
                if fk > 7000.0:
                    break
                y += weight(k, fk) * np.sin(k * ph + r.uniform(0, math.tau))
    return y


def pad(freqs, length, attack=0.8, release=1.0, seed=1):
    """A choir-like pad on an "ah": detuned voices, their partials lifted
    where the vowel's formants are. Peak level 1."""
    n = int(length * SR)
    formants = ((700.0, 130.0, 1.0), (1150.0, 160.0, 0.6), (2600.0, 260.0, 0.3))

    def weight(k, fk):
        g = 0.12 + sum(a * math.exp(-((fk - c) / w) ** 2) for c, w, a in formants)
        return g / k

    y = _partials(freqs, n, (-0.005, 0.0, 0.006), 0.004, weight, seed)
    t = np.arange(n) / SR
    y *= np.minimum(1.0, t / attack) * np.clip((length - t) / release, 0.0, 1.0)
    return y / (np.abs(y).max() + 1e-9)


def swell(freqs, length, rise=0.7, release=0.6, seed=2):
    """Brass-like chord that grows louder and brighter: the upper partials
    come in as it swells. rise is the fraction of the length spent rising.
    Peak level 1."""
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.clip(t / max(rise * length, 1e-3), 0.0, 1.0) ** 1.6
    env *= np.clip((length - t) / release, 0.0, 1.0)
    y = _partials(freqs, n, (-0.003, 0.003), 0.003, lambda k, fk: env ** (1.0 + 0.35 * (k - 1)) / k, seed)
    return y / (np.abs(y).max() + 1e-9)


def timpani(f0=73.4, length=2.0, vel=1.0):
    """A kettle drum: a low tone that sags in pitch, its overtone, and the
    slap of the stick."""
    n = int(length * SR)
    t = np.arange(n) / SR
    f = f0 * (1.0 + 0.06 * np.exp(-t / 0.08))
    ph = math.tau * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t / 0.7) + 0.4 * np.sin(1.5 * ph) * np.exp(-t / 0.25)
    y += 0.5 * shape(rng.standard_normal(n) * np.exp(-t / 0.012), lo=80.0, hi=1500.0)
    return y * vel * np.minimum(1.0, t / 0.002)


def shimmer(length=3.0, base=1046.5, seed=4):
    """A bright, bell-like cluster that rings and fades: for a moment of
    magic."""
    r = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio in (1.0, 1.26, 1.5, 2.0, 2.52, 3.0, 4.0):
        for _ in range(2):
            f = base * ratio * (1.0 + r.uniform(-0.004, 0.004))
            y += np.sin(math.tau * f * t + r.uniform(0, math.tau)) * np.exp(-t / r.uniform(0.6, 1.6)) / ratio
    y *= np.minimum(1.0, t / 0.01)
    return y / (np.abs(y).max() + 1e-9)


def hz(name):
    semis = dict(C=-9, D=-7, E=-5, F=-4, G=-2, A=0, B=2)
    n = semis[name[0]] + (1 if "#" in name else 0) + 12 * (int(name[-1]) - 4)
    return 440.0 * 2.0 ** (n / 12.0)


# ---------------------------------------------------------------- finish

def finish(mix, film, path, tail=0.0, fade_in=0.8, fade_out=2.2, peak_db=-3.0):
    """Fade the ends, bring the loudest moment to peak_db, save a 16-bit WAV
    and print how loud each shot is."""
    n = mix.shape[1]
    t = np.arange(n) / SR
    mix = mix * (np.minimum(1.0, t / fade_in) * np.clip((seconds(film, tail) - t) / fade_out, 0.0, 1.0))
    peak = np.abs(mix).max()
    mix *= db(peak_db) / peak
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(mix.T, -1, 1) * 32767.0).astype("<i2").tobytes())
    rms = 20 * np.log10(np.sqrt((mix ** 2).mean()) + 1e-12)
    print(f"{n / SR:.1f} s, peak moved from {20 * math.log10(peak):+.1f} to {peak_db:.1f} dBFS, overall {rms:.1f} dBFS")
    for sh in film.all_shots():
        a, b = int(film.at(sh["name"]) * SR), int(film.at(sh["name"], sh["seconds"]) * SR)
        seg = mix[:, a:b]
        print(f"  {sh['name']}  {20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12):6.1f} dBFS  "
              f"peak {20 * np.log10(np.abs(seg).max() + 1e-12):6.1f}")
