"""The soundtrack of The Marble, made from arithmetic: no recordings but the voice.

    uv run python shared/post/soundtrack.py --film example     # writes out/sound.wav

It reads out/motion.json (from shared/blender/motion.py) for where the
marble is on every frame, so the rolling follows the picture, and the
narrator's recording from out/voice/ (from shared/post/voice.py).

Layers: room tone, the marble rolling, the click and clatter of the domino,
a quiet pad under it all, and the voice. The tools they are built from are
shared: shared/post/synth.py.
"""
import wave

import numpy as np

import dialogue
from synth import SR, blank, db, hz, pad, piano_note, put, rng, room_echo, shape, thud, tok

TAIL = 2.0                      # seconds of sound after the last frame, over black


def room_tone(mix):
    for c in range(2):
        x = np.cumsum(rng.standard_normal(mix.shape[1]))
        x = shape(x - x.mean(), lo=40.0, hi=300.0)
        mix[c] += x / np.abs(x).max() * db(-50)


def rolling(mix, film, motion):
    """A low glassy rumble whose level follows the marble's speed."""
    n = mix.shape[1]
    level = np.zeros(n)
    for name, m in motion.items():
        x = np.array(m["x"])
        t0 = film.at(name)
        ft = t0 + np.arange(len(x)) / film.FPS
        i0, i1 = int(t0 * SR), min(n, int((t0 + len(x) / film.FPS) * SR))
        v = np.abs(np.gradient(x)) * film.FPS
        level[i0:i1] = np.interp(np.arange(i0, i1) / SR, ft, np.minimum(1.0, v / 0.1))
    k = int(0.05 * SR)
    level = np.convolve(level, np.ones(k) / k, mode="same")
    noise = shape(rng.standard_normal(n), lo=120.0, hi=900.0)
    noise /= np.abs(noise).max()
    dry = np.zeros_like(mix)
    dry += noise * level * db(-26)
    mix += room_echo(dry, 0.6, 0.15, 5000.0, seed=3)


def domino(mix, film, motion):
    """A click when the marble hits it, and a clatter when it lands."""
    dry = np.zeros_like(mix)
    tip = np.array(motion["S02"]["tip"])
    hit = np.argmax(tip > 0.0) - 1
    land = np.argmax(tip > 1.5)
    put(dry, tok(2600.0, 0.03, bright=1.0), film.at("S02", hit / film.FPS), db(-14), pan=0.1)
    t = film.at("S02", land / film.FPS)
    put(dry, thud(180.0, 0.18), t, db(-12), pan=0.2)
    put(dry, tok(1400.0, 0.05), t, db(-16), pan=0.2)
    mix += room_echo(dry, 0.8, 0.2, 4500.0, seed=5)


def music(mix, film):
    chord = [hz(n) for n in ("D3", "A3", "F#4")]
    put(mix, pad(chord, film.at("S02", 1.2) + 0.5, attack=1.5, release=1.0), 0.0, db(-30), pan=-0.2)
    put(mix, pad(chord, film.at("S02", 1.2) + 0.5, attack=1.5, release=1.0, seed=2), 0.0, db(-30), pan=0.2)
    put(mix, piano_note(hz("D5"), 3.0, vel=0.5), film.at("S02", 2.0), db(-22), pan=0.15)


def voices(mix, film):
    """Every line at its moment. voice.py saves at SR, mono, 16-bit."""
    for ln in dialogue.LINES:
        with wave.open(film.path("out", "voice", ln["id"] + ".wav"), "rb") as w:
            assert w.getframerate() == SR, "voice.py saves at synth.SR"
            y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0
        put(mix, y / (np.abs(y).max() + 1e-9), film.at(ln["shot"], ln["at"]), db(-8))


def build(film, motion):
    """Every layer, balanced. The voice sets the level; the rest sits under it."""
    if not motion:
        raise SystemExit("no out/motion.json: run shared/blender/motion.py --film example first")
    mix = blank(film, TAIL)
    room_tone(mix)
    rolling(mix, film, motion)
    domino(mix, film, motion)
    music(mix, film)
    voices(mix, film)
    return mix
