"""Music and story sounds under a narration. A trial.

    uv run python podcasts/mix.py podcasts/example/01_lighthouse/out/drama3_seed1.wav
    uv run python podcasts/mix.py podcasts/example/01_lighthouse/out/drama3_seed1.wav --under -34 --no-sounds
    uv run python podcasts/mix.py NARRATION.wav --script SCRIPT.txt --place -20 --seed 8

Takes a narration (the .wav narrate.py writes) and writes <name>_mix.mp3
beside it.

music   a slow, sparse piano, made here from the project's own piano
        (shared/post/synth.py): it plays alone for a few seconds, drops far
        under the voice when it starts, and comes back up at the end.
        --under and --alone are its level against the voice, in dB, while
        the voice speaks and while it is silent. --seed changes the notes.
sounds  what the script asks for on its lines starting with ">" (see
        sounds.py): a place sets the scene as its paragraph begins and then
        fades away (--stay seconds under the voice, then a slow fade; a
        sound that ran on under the whole paragraph was found distracting;
        --stay 0 keeps it for the whole paragraph), a spot is heard in a
        pause made for it just before its paragraph. --place and --spot are
        their levels against the voice, in dB.

The script is the script.txt in the folder above the narration's, unless
--script names another. Nothing is downloaded.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "shared", "post"), os.path.join(ROOT, "shared"), os.path.dirname(os.path.abspath(__file__))]

from synth import SR, db, hz, piano_note, put, room_echo, shape  # noqa: E402
import sounds  # noqa: E402

BPM = 66.0
BEAT = 60.0 / BPM
# one bar each, round and round: the note underneath, then the chord
ROUND = [
    ("F3", ["A3", "C4", "E4"]),
    ("E3", ["G3", "B3", "D4"]),
    ("D3", ["F3", "A3", "C4"]),
    ("C3", ["E3", "G3", "B3"]),
    ("F3", ["A3", "C4", "E4"]),
    ("G3", ["B3", "D4", "E4"]),
    ("A3", ["G3", "C4", "E4"]),
    ("A3", ["G3", "B3", "E4"]),
]
TUNE = ["G4", "A4", "C5", "D5", "E5", "G5"]      # the notes picked out on top
VOICE_DB = -20.0        # how loud the voice sits in the mix
LEAD, TAIL = 4.0, 7.0   # seconds of music alone before the voice and after it
GAP = 0.75              # the pause between paragraphs
PLACE_FADE = 5.0        # seconds a place takes to fade away once it has set the scene


def bed(seconds, seed=5):
    """The music, in stereo. Sparse: a chord spread over each bar, a few
    notes after it, and nothing low enough or long enough to drone."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    dry = np.zeros((2, n + int(6.0 * SR)))
    made = {}

    def note(name, beats, vel):
        key = (name, beats, vel)
        if key not in made:
            made[key] = piano_note(hz(name), beats * BEAT, vel)
        return made[key]

    t, bar = 0.0, 0
    while t < seconds:
        root, chord = ROUND[bar % len(ROUND)]
        put(dry, note(root, 2.5, 0.3), t, db(-19), pan=-0.2)
        for k, name in enumerate(chord):
            put(dry, note(name, 2.0, 0.3), t + (0.5 + 0.5 * k) * BEAT + r.uniform(-0.01, 0.01), db(-19), pan=-0.1 + 0.1 * k)
        for beat in (2.5, 3.0, 3.5):
            if r.random() < 0.3:
                name = chord[r.integers(len(chord))]
                name = name[:-1] + str(int(name[-1]) + 1)
                put(dry, note(name, 1.5, 0.3), t + beat * BEAT + r.uniform(-0.012, 0.012), db(-21),
                    pan=r.uniform(-0.3, 0.3))
        if bar % 2 == 1 and r.random() < 0.7:
            put(dry, note(TUNE[r.integers(len(TUNE))], 2.5, 0.4), t + r.choice([2.0, 3.0]) * BEAT, db(-18), pan=0.25)
        t += 4 * BEAT
        bar += 1
    wet = room_echo(dry, seconds=1.4, wet=0.12)
    return np.stack([shape(wet[c], lo=170.0) for c in range(2)])[:, :n]


def loudness(y, window=0.05):
    """RMS over the parts that are not silence."""
    k = int(window * SR)
    y = np.atleast_2d(y)
    frames = y[:, :y.shape[1] // k * k].reshape(y.shape[0], -1, k)
    rms = np.sqrt((frames ** 2).mean(axis=(0, 2)))
    loud = rms[rms > rms.max() * 0.1]
    return float(np.sqrt((loud ** 2).mean()))


def ducking(speech, n, under, alone, ahead=0.9, down=0.30, up=1.6):
    """The music's gain at every sample: `alone` dB (against the voice) when
    nobody is speaking, `under` dB while they are. It starts down a little
    before each word, and comes back up slowly."""
    rate = 100
    k = SR // rate
    m = n // k + 1
    rms = np.zeros(m)
    frames = speech[:len(speech) // k * k].reshape(-1, k)
    rms[:len(frames)] = np.sqrt((frames ** 2).mean(axis=1))
    speaking = rms > rms.max() * db(-35.0)
    look = int(ahead * rate)
    soon = np.array([speaking[i:i + look + 1].any() for i in range(m)])
    g = np.empty(m)
    cur = alone
    for i in range(m):
        target = under if soon[i] else alone
        cur += (target - cur) * (1.0 - np.exp(-1.0 / (rate * (down if target < cur else up))))
        g[i] = cur
    return db(np.interp(np.arange(n) / k, np.arange(m), g))


# ---------------------------------------------------------------- the script and the narration

def directions(path):
    """For each spoken paragraph of a script, the sounds asked for on the
    ">" lines above it."""
    out, waiting = [], []
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8").read()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        for l in lines:
            if l.startswith(">"):
                waiting += [w.strip() for w in l[1:].split(",") if w.strip()]
        if any(not l.startswith(("#", ">")) for l in lines):
            out.append(waiting)
            waiting = []
    for name in (w for ws in out for w in ws):
        if name not in sounds.PLACES and name not in sounds.SPOTS:
            raise SystemExit(f"{path}: no sound called {name!r}; there are {', '.join(list(sounds.PLACES) + list(sounds.SPOTS))}")
    return out


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def paragraphs(path, y, count):
    """Where each paragraph starts and ends in a narration, in samples:
    from the .json narrate.py wrote beside it, or else from the stretches of
    perfect silence it left between them."""
    side = path[:-4] + ".json"
    if os.path.exists(side):
        return [(int(a * SR), int(b * SR)) for a, b in json.load(open(side))["paragraphs"]]
    quiet = np.concatenate([[False], y == 0.0, [False]])
    edges = np.flatnonzero(quiet[1:] != quiet[:-1]).reshape(-1, 2)
    gaps = [(a, b) for a, b in edges if b - a >= int((GAP - 0.01) * SR) and a > 0 and b < len(y)]
    first = int(np.flatnonzero(y != 0.0)[0])
    last = int(np.flatnonzero(y != 0.0)[-1]) + 1
    cuts = [first] + [x for g in gaps for x in g] + [last]
    found = list(zip(cuts[0::2], cuts[1::2]))
    if len(found) != count:
        raise SystemExit(f"{path}: found {len(found)} paragraphs in the sound but the script has {count}")
    return found


# ---------------------------------------------------------------- the mix

def faded(st, fade_in, fade_out):
    n = st.shape[1]
    t = np.arange(n) / SR
    return st * np.clip(t / fade_in, 0.0, 1.0) * np.clip((n / SR - t) / fade_out, 0.0, 1.0)


def mix(narration, parts, asked, music=True, under=-30.0, alone=-12.0, place=-24.0, spot=-6.0, seed=5, stay=3.5):
    """parts: (start, end) of each paragraph in the narration, in samples.
    asked: the sounds each paragraph asks for. Returns stereo."""
    gain = db(VOICE_DB) / loudness(narration)
    # when each paragraph starts in the mix: spots get a pause long enough to be heard in
    starts, t = [], 0.0
    for i, (a, b) in enumerate(parts):
        lead = max([sounds.LEAD[s] for s in asked[i] if s in sounds.SPOTS], default=0.0)
        pause = (LEAD if music else 0.4) if i == 0 else GAP
        t += max(pause, lead + 0.35 if lead else 0.0)
        starts.append(t)
        t += (b - a) / SR
    n = int((t + (TAIL if music else 1.0)) * SR)
    speech = np.zeros(n)
    for (a, b), at in zip(parts, starts):
        speech[int(at * SR):int(at * SR) + b - a] = narration[a:b] * gain
    ends = [at + (b - a) / SR for (a, b), at in zip(parts, starts)]
    out = np.zeros((2, n))

    def add(st, at, level):
        i = int(at * SR)
        st = st[:, max(0, -i):]
        i = max(i, 0)
        st = st[:, :n - i]
        out[:, i:i + st.shape[1]] += st * level

    # places: one stretch of sound for each run of paragraphs that share it
    for name, fn in sounds.PLACES.items():
        i = 0
        while i < len(parts):
            if name not in asked[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(parts) and name in asked[j + 1]:
                j += 1
            a, b = starts[i] - 1.2, ends[j] + 1.8
            brief = stay > 0 and starts[i] + stay + PLACE_FADE < b
            if brief:                                   # set the scene, then leave the voice alone
                b = starts[i] + stay + PLACE_FADE
            st = fn(b - a, seed + 10 * i)
            add(faded(st / loudness(st), 1.2, PLACE_FADE if brief else 1.8), a, db(VOICE_DB + place))
            i = j + 1
    # spots: heard alone for their lead, then pulled down as the voice comes in
    for i, names in enumerate(asked):
        for name in names:
            if name in sounds.SPOTS:
                st = sounds.SPOTS[name](seed + i)
                lead = sounds.LEAD[name]
                t_ = np.arange(st.shape[1]) / SR
                st = st / np.abs(st).max() * db(-9.0 * np.clip((t_ - lead + 0.2) / 0.6, 0.0, 1.0))
                add(faded(st, 0.01, 0.6), starts[i] - lead, db(VOICE_DB + spot + 8.0))
    if music:
        m = bed(n / SR + 0.5, seed)[:, :n]
        m *= db(VOICE_DB) / loudness(m)
        m *= ducking(speech, n, under, alone)
        add(faded(m, 1.0, 4.5), 0.0, 1.0)
    out += speech
    # no limiter: bending the peaks (a tanh here until 2026-11-20) put distortion 26 to 31 dB under
    # the voice in a loud greeting. If the sum would go over, the whole mix is turned down instead.
    out *= 1.3
    return out * min(1.0, 0.84 / np.abs(out).max())       # -1.5 dBFS: room for what mp3 encoding adds


def write_mp3(path, stereo):
    wav = path[:-4] + ".tmp.wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(stereo.T, -1, 1) * 32767.0).astype("<i2").tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ar", "44100", "-c:a", "libmp3lame",
                    "-b:a", "128k", path], check=True)
    os.remove(wav)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("narration")
    p.add_argument("--script")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--no-sounds", action="store_true")
    p.add_argument("--under", type=float, default=-30.0, help="music against the voice while it speaks, dB")
    p.add_argument("--alone", type=float, default=-12.0, help="music against the voice when it is silent, dB")
    p.add_argument("--place", type=float, default=-24.0, help="a place's sound against the voice, dB")
    p.add_argument("--spot", type=float, default=-6.0, help="a spot sound against the voice, dB")
    p.add_argument("--stay", type=float, default=3.5,
                   help="seconds a place is heard under the voice before it fades away; 0 for the whole paragraph")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--name", help="what to call the result (default: <narration>_mix)")
    a = p.parse_args()

    script = a.script or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.narration))), "script.txt")
    asked = directions(script)
    y = read_wav(a.narration)
    parts = paragraphs(a.narration, y, len(asked))
    asked = asked[:len(parts)]
    if a.no_sounds:
        asked = [[] for _ in parts]
    out = mix(y, parts, asked, not a.no_music, a.under, a.alone, a.place, a.spot, a.seed, a.stay)
    path = os.path.join(os.path.dirname(a.narration), (a.name or os.path.basename(a.narration)[:-4] + "_mix") + ".mp3")
    write_mp3(path, out)
    used = ", ".join(sorted({s for ws in asked for s in ws})) or "none"
    print(f"{os.path.relpath(path, ROOT)}: {out.shape[1] / SR / 60:.1f} min, sounds: {used}, "
          f"peak {20 * np.log10(np.abs(out).max()):.1f} dBFS", flush=True)


if __name__ == "__main__":
    main()
