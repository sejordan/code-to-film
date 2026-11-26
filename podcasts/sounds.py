"""Story sounds for a narration, all made here from code. A trial.

Two kinds. A PLACE is a background that runs under a whole paragraph (night,
ducts, sea, city, hall, rain). A SPOT is one short sound heard just before a
paragraph starts and trailing off under its first words (siren, alarm, door,
camera, clang, suitcase, bell, rumble).

A script asks for them on a line of its own before the paragraph they
belong to, starting with ">":

    > night
    Es la madrugada del veinticinco de diciembre...

    > city, siren
    La policía estaba convencida...

Every place takes (seconds, seed) and every spot takes (seed), and each
returns stereo samples at SR, at a level of its own choosing around 1;
mix.py sets how loud they sit against the voice.

    uv run python podcasts/sounds.py          # writes one sample of each to podcasts/sounds/
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "shared", "post"), os.path.join(ROOT, "shared")]

from synth import SR, conv, db, hann, put, shape, thud  # noqa: E402


def norm(x):
    return x / (np.sqrt((x ** 2).mean()) + 1e-12)


def hiss(n, r, lo=None, hi=None):
    """Noise between lo and hi Hz, at a level of 1."""
    return norm(shape(r.standard_normal(n), lo=lo, hi=hi))


def stereo(n):
    return np.zeros((2, n))


def room(st, seconds, wet, dark, r):
    """The sound in a space that answers for `seconds`."""
    n = int(seconds * SR)
    out = np.empty_like(st)
    for c in range(2):
        ir = r.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir[:int(0.015 * SR)] = 0.0
        ir = shape(ir, lo=200.0, hi=dark)
        ir /= np.sqrt((ir ** 2).sum())
        out[c] = st[c] + wet * conv(st[c], ir)
    return out


def ring(freqs, length, r, decay=0.35, spread=0.6):
    """Struck metal: tones that are not in tune with each other, dying away."""
    n = int(length * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k, f in enumerate(freqs):
        y += np.sin(math.tau * f * t + r.uniform(0, math.tau)) * np.exp(-t / (decay * r.uniform(1 - spread, 1.0))) / (1 + 0.4 * k)
    return y * np.minimum(1.0, t / 0.001)


# ---------------------------------------------------------------- places

def night(seconds, seed=1):
    """Open air after midnight: a few crickets, unhurried, and a little wind."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    for c in range(2):
        gust = 0.6 + 0.4 * np.sin(math.tau * (0.07 * t + 0.3 * c)) * np.sin(math.tau * 0.023 * t + c)
        out[c] += hiss(n, r, 250.0, 1500.0) * gust * db(-14)
    k = int(0.016 * SR)
    tk = np.arange(k) / SR
    for who in range(3):
        f, pan, every = r.uniform(3900.0, 4700.0), r.uniform(-0.9, 0.9), r.uniform(1.7, 2.8)
        beep = np.sin(math.tau * f * tk) * hann(k) * (1.0 + 0.5 * np.sin(math.tau * 60.0 * tk))
        chirp = np.concatenate([beep, np.zeros(int(0.014 * SR))] * 3)
        at = r.uniform(0.0, every)
        while at < seconds - 0.2:
            put(out, chirp, at, db(-10 - 3 * who), pan)
            at += every * r.uniform(0.9, 1.15)
    return out


def ducts(seconds, seed=2):
    """Inside sheet-metal air ducts: moving air, and now and then the metal
    giving under someone's weight, a knee, a scrape."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    for c in range(2):
        flutter = 0.8 + 0.2 * np.sin(math.tau * 0.31 * t + 2.0 * c)
        out[c] += hiss(n, r, 300.0, 1100.0) * flutter * db(-16)
    at = r.uniform(0.6, 1.6)
    while at < seconds - 1.0:
        kind = r.random()
        pan = r.uniform(-0.6, 0.6)
        if kind < 0.45:                                   # the metal buckles: a dull boom with a ring on it
            f = r.uniform(150.0, 240.0)
            boom = ring([f, f * 1.47, f * 2.31, f * 3.9], 0.9, r, decay=0.22)
            put(out, boom + 0.6 * np.pad(thud(f * 0.9, 0.2), (0, len(boom) - int(0.2 * SR))), at, db(-9), pan)
        elif kind < 0.8:                                  # a knee or an elbow, twice
            for j in range(2):
                put(out, thud(r.uniform(170.0, 230.0), 0.16), at + 0.34 * j, db(-10), pan)
        else:                                             # cloth dragging over metal
            m = int(r.uniform(0.35, 0.6) * SR)
            put(out, hiss(m, r, 900.0, 3800.0) * hann(m) ** 2, at, db(-17), pan)
        at += r.uniform(1.6, 3.8)
    return room(out, 1.1, 0.5, 2600.0, r)


def sea(seconds, seed=3):
    """A beach: waves coming in and drawing back."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    for c in range(2):
        out[c] += hiss(n, r, 200.0, 900.0) * db(-20)                       # the sea further out
    at = r.uniform(0.0, 2.0)
    while at < seconds:
        length = r.uniform(5.0, 7.5)
        m = int(length * SR)
        u = np.arange(m) / m
        swell = np.where(u < 0.3, (u / 0.3) ** 2, np.exp(-(u - 0.3) / 0.22))    # rushes in, drains away
        pan = r.uniform(-0.5, 0.5)
        wave_ = hiss(m, r, 300.0, 2600.0) * swell + 0.5 * hiss(m, r, 1500.0, 6000.0) * swell ** 2
        put(out, wave_, at, db(-11), pan)
        at += length * r.uniform(0.75, 0.95)
    return out


def city(seconds, seed=4):
    """A big city from a little way off: traffic, and a horn now and then."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    for c in range(2):
        surge = 0.75 + 0.25 * np.sin(math.tau * 0.05 * t + 1.7 * c) * np.sin(math.tau * 0.017 * t)
        out[c] += hiss(n, r, 180.0, 700.0) * surge * db(-15)
    at = r.uniform(1.5, 4.0)
    while at < seconds - 1.0:
        m = int(r.uniform(0.25, 0.6) * SR)
        tm = np.arange(m) / SR
        f = r.choice([390.0, 440.0, 494.0])
        horn = (np.sin(math.tau * f * tm) + 0.6 * np.sin(math.tau * f * 1.26 * tm) + 0.3 * np.sin(math.tau * f * 2.0 * tm))
        horn *= np.minimum(1.0, tm / 0.02) * np.minimum(1.0, (m / SR - tm) / 0.06)
        put(out, shape(horn, lo=300.0, hi=1500.0), at, db(-22), r.uniform(-0.8, 0.8))
        if r.random() < 0.4:
            put(out, shape(horn, lo=300.0, hi=1500.0), at + m / SR + 0.12, db(-22), r.uniform(-0.8, 0.8))
        at += r.uniform(4.0, 9.0)
    return room(out, 1.4, 0.35, 1800.0, r)


def hall(seconds, seed=5):
    """A large empty hall with a stone floor: someone walking, unhurried."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = stereo(n)
    for c in range(2):
        out[c] += hiss(n, r, 250.0, 800.0) * db(-30)
    at, pan = r.uniform(0.3, 1.0), r.uniform(-0.5, 0.5)
    while at < seconds - 0.6:
        steps = int(r.integers(4, 9))
        drift = r.uniform(-0.06, 0.06)
        for j in range(steps):
            if at >= seconds - 0.6:
                break
            m = int(0.09 * SR)
            tm = np.arange(m) / SR
            heel = shape(r.standard_normal(m) * np.exp(-tm / 0.006), lo=700.0, hi=3800.0)
            heel = heel / (np.abs(heel).max() + 1e-9) + 0.5 * thud(r.uniform(130.0, 160.0), 0.09)
            put(out, heel, at, db(-10) * r.uniform(0.7, 1.0), pan)
            pan = float(np.clip(pan + drift, -0.8, 0.8))
            at += r.uniform(0.58, 0.66)
        at += r.uniform(1.5, 3.5)
    return room(out, 2.6, 1.1, 3000.0, r)


def rain(seconds, seed=6):
    """Steady rain on a roof and a street."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    for c in range(2):
        out[c] += hiss(n, r, 900.0, 7000.0) * (0.85 + 0.15 * np.sin(math.tau * 0.09 * t + c)) * db(-16)
        out[c] += hiss(n, r, 250.0, 900.0) * db(-22)
    for _ in range(int(seconds * 14)):                     # the nearer drops
        m = int(0.012 * SR)
        drop = np.sin(math.tau * r.uniform(1800.0, 4200.0) * np.arange(m) / SR) * np.exp(-np.arange(m) / (0.003 * SR))
        put(out, drop, r.uniform(0.0, seconds - 0.05), db(-20) * r.uniform(0.3, 1.0), r.uniform(-1.0, 1.0))
    return out


# ---------------------------------------------------------------- spots

def siren(seed=1, seconds=7.0):
    """A police car some streets away, passing."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = 650.0 + 420.0 * (0.5 - 0.5 * np.cos(math.tau * t / 3.4))           # the wail, up and down
    ph = math.tau * np.cumsum(f) / SR
    y = np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    y = shape(y, lo=400.0, hi=2000.0)
    near = np.exp(-((t - seconds * 0.4) / (seconds * 0.3)) ** 2)            # comes nearer, goes
    out = stereo(n)
    pan = np.linspace(-0.6, 0.5, n)
    a = (pan + 1.0) * math.pi / 4.0
    out[0], out[1] = y * near * np.cos(a), y * near * np.sin(a)
    return room(out, 1.8, 0.9, 1800.0, r) * db(-12)


def alarm(seed=1, seconds=3.2):
    """An alarm bell ringing on a wall."""
    r = np.random.default_rng(seed)
    out = stereo(int(seconds * SR))
    at = 0.0
    while at < seconds - 1.0:
        put(out, ring([2050.0, 2790.0, 3390.0, 4530.0], 0.25, r, decay=0.07), at, db(-12), 0.1)
        at += 1.0 / 17.0
    return room(out, 1.0, 0.4, 5000.0, r)


def door(seed=1):
    """A wooden door: the hinge, then the latch."""
    r = np.random.default_rng(seed)
    out = stereo(int(2.2 * SR))
    m = int(0.9 * SR)
    tm = np.arange(m) / SR
    f = 310.0 + 170.0 * (tm / 0.9) ** 0.6 + 14.0 * np.sin(math.tau * 9.0 * tm)
    ph = math.tau * np.cumsum(f) / SR
    grain = 0.55 + 0.45 * np.sign(np.sin(math.tau * 31.0 * tm + 3.0 * np.sin(math.tau * 2.3 * tm)))   # it catches and slips
    squeak = shape((np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)) * grain * hann(m), lo=250.0, hi=3200.0)
    put(out, squeak / (np.abs(squeak).max() + 1e-9), 0.0, db(-12), -0.1)
    put(out, thud(140.0, 0.18), 1.05, db(-6), 0.0)
    put(out, ring([900.0, 1730.0, 2600.0], 0.12, r, decay=0.03), 1.06, db(-16), 0.0)
    return room(out, 0.7, 0.3, 3000.0, r)


def camera(seed=1):
    """Press photographers: shutters, and the pop of flashbulbs."""
    r = np.random.default_rng(seed)
    out = stereo(int(3.0 * SR))
    at = 0.05
    for _ in range(7):
        m = int(0.05 * SR)
        tm = np.arange(m) / SR
        click = shape(r.standard_normal(m) * (np.exp(-tm / 0.004) + 0.7 * np.exp(-np.abs(tm - 0.028) / 0.003)), lo=1200.0, hi=7000.0)
        pan = r.uniform(-0.8, 0.8)
        put(out, click / (np.abs(click).max() + 1e-9), at, db(-10) * r.uniform(0.5, 1.0), pan)
        if r.random() < 0.6:
            k = int(0.25 * SR)
            pop = hiss(k, r, 500.0, 5000.0) * np.exp(-np.arange(k) / (0.035 * SR))
            put(out, pop, at + 0.01, db(-22), pan)
        at += r.uniform(0.18, 0.5)
    return room(out, 0.9, 0.35, 4000.0, r)


def clang(seed=1):
    """A barred iron door shut hard, in a corridor."""
    r = np.random.default_rng(seed)
    out = stereo(int(3.2 * SR))
    put(out, ring([207.0, 311.0, 523.0, 811.0, 1370.0, 2140.0], 1.6, r, decay=0.45), 0.0, db(-12), 0.0)
    put(out, thud(85.0, 0.3), 0.0, db(-4), 0.0)
    put(out, ring([1900.0, 2950.0], 0.2, r, decay=0.04), 0.33, db(-22), 0.15)      # the bolt
    return room(out, 2.4, 1.0, 3200.0, r)


def suitcase(seed=1):
    """A suitcase: the zip, then two catches."""
    r = np.random.default_rng(seed)
    out = stereo(int(2.0 * SR))
    m = int(0.8 * SR)
    tm = np.arange(m) / SR
    teeth = 0.5 + 0.5 * np.sign(np.sin(math.tau * np.cumsum(140.0 + 260.0 * tm / 0.8) / SR))
    zip_ = shape(r.standard_normal(m) * teeth * hann(m) ** 0.5, lo=1500.0 , hi=7500.0)
    put(out, zip_ / (np.abs(zip_).max() + 1e-9), 0.0, db(-14), -0.1)
    for at in (1.05, 1.32):
        put(out, ring([1250.0, 2380.0, 3900.0], 0.1, r, decay=0.02) + 0.6 * np.pad(thud(190.0, 0.08), (0, int(0.1 * SR) - int(0.08 * SR))), at, db(-10), 0.1)
    return room(out, 0.5, 0.25, 4000.0, r)


def bell(seed=1, strokes=3):
    """A church bell across a square: slow strokes."""
    r = np.random.default_rng(seed)
    every, length = 1.7, 4.5
    out = stereo(int((every * (strokes - 1) + length) * SR))
    n = int(length * SR)
    t = np.arange(n) / SR
    f0 = 294.0
    # a bell's own tones: the hum below, the strike note, the minor third above it, and so on up
    tones = ((0.5, 0.55, 3.2), (1.0, 1.0, 2.0), (1.19, 0.7, 1.3), (1.5, 0.45, 0.9), (2.0, 0.6, 0.7),
             (2.51, 0.3, 0.4), (3.01, 0.22, 0.3), (4.1, 0.15, 0.18))
    for k in range(strokes):
        y = sum(a * np.sin(math.tau * f0 * m * t + r.uniform(0, math.tau)) * np.exp(-t / d) for m, a, d in tones)
        y = y * np.minimum(1.0, t / 0.002)
        put(out, y / np.abs(y).max(), k * every, db(-12) * r.uniform(0.8, 1.0), -0.15)
    return room(out, 2.6, 0.7, 2600.0, r)


def rumble(seed=1, seconds=6.0):
    """The ground shaking: a low roar that swells, and things rattling in it."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    swell = np.exp(-((t - seconds * 0.42) / (seconds * 0.26)) ** 2)
    out = stereo(n)
    for c in range(2):
        shake = 0.7 + 0.3 * np.sin(math.tau * 5.5 * t + c + 2.0 * np.sin(math.tau * 0.7 * t))
        out[c] += (hiss(n, r, 45.0, 140.0) * db(-8) + hiss(n, r, 140.0, 420.0) * db(-17)) * swell * shake
    for _ in range(int(seconds * 9)):                       # tiles, crockery, shutters
        at = r.uniform(0.6, seconds - 1.2)
        loud = math.exp(-((at - seconds * 0.42) / (seconds * 0.24)) ** 2)
        put(out, ring([r.uniform(700.0, 2600.0), r.uniform(2600.0, 5200.0)], 0.08, r, decay=0.02), at,
            db(-24) * loud * r.uniform(0.4, 1.0), r.uniform(-0.8, 0.8))
    for at in (seconds * 0.38, seconds * 0.55):             # something heavy comes down
        put(out, thud(70.0, 0.35), at, db(-10), r.uniform(-0.3, 0.3))
    return room(out, 1.2, 0.35, 1800.0, r)


PLACES = dict(night=night, ducts=ducts, sea=sea, city=city, hall=hall, rain=rain)
SPOTS = dict(siren=siren, alarm=alarm, door=door, camera=camera, clang=clang, suitcase=suitcase, bell=bell,
             rumble=rumble)
# how long a spot is heard on its own before the voice comes in over it
LEAD = dict(siren=2.2, alarm=1.6, door=1.6, camera=1.4, clang=1.5, suitcase=1.6, bell=2.6, rumble=2.6)


def main():
    out = os.path.join(ROOT, "podcasts", "sounds")
    os.makedirs(out, exist_ok=True)
    for name, fn in list(PLACES.items()) + list(SPOTS.items()):
        y = fn(12.0) if name in PLACES else fn()
        y = y / max(1.0, np.abs(y).max() / 0.9)
        wav = os.path.join(out, name + ".wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(y.T, -1, 1) * 32767.0).astype("<i2").tobytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "libmp3lame", "-b:a", "128k",
                        wav[:-4] + ".mp3"], check=True)
        os.remove(wav)
        rms = 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)
        print(f"{name:9s} {y.shape[1] / SR:5.1f} s  rms {rms:6.1f} dBFS  peak {20 * np.log10(np.abs(y).max() + 1e-9):6.1f}", flush=True)


if __name__ == "__main__":
    main()
