"""A text read aloud from start to finish by one voice, on this Mac. A trial.

    uv run python podcasts/narrate.py podcasts/example/01_lighthouse/script.txt --engine drama3 --seed 1
    uv run python podcasts/narrate.py podcasts/example/01_lighthouse/script.txt --engine voxtral --voice neutral_male

The script is plain text: paragraphs separated by blank lines. Lines
starting with # are notes and lines starting with > ask for story sounds
(see sounds.py; mix.py adds them); neither is spoken. Each paragraph is spoken on its own, so one
that goes wrong is tried again without redoing the rest, and the paragraphs
are joined with a pause between them.

drama3   invents a voice, a different one for every --seed. The first
         paragraph (keep it short: a greeting) is spoken first and then
         serves as the reference recording for all the others, so the voice
         stays the same. --ref gives a reference recording instead, with
         what it says as --ref-text or as the first paragraph of
         --ref-script. With --reref the first paragraph is spoken in the
         reference's voice and then becomes the reference itself, so the
         rest takes its manner from this script's greeting.
         --temperature (1.0 unless given) and --top-p (0.9) set how freely
         it reads: higher is more varied and more likely to slip.
         Delivery cues go in the text in square brackets: [laughs], [sighs],
         [excited].
         --direction puts one cue before every paragraph. An accent cue
         such as "[Scottish accent]" can shift the delivery, though the
         model documents no accent control: judge it by ear.
voxtral  one of its fixed voices (--voice); its Spanish ones are from Spain.
         It is given a sentence at a time.

Writes <name>.wav and <name>.mp3 beside the script, in out/. Both models are
for non-commercial use only; never give drama3 a real person's voice without
their consent.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "shared", "post"), os.path.join(ROOT, "shared")]

import voice  # noqa: E402

SR = voice.SR
BETWEEN = 0.75          # seconds of silence between paragraphs
SENTENCE = 0.30         # and between sentences, for an engine given one at a time
LONGEST_PAUSE = 0.9     # silences inside a paragraph are capped at this


def paragraphs(path):
    text = open(path, encoding="utf-8").read()
    blocks = [" ".join(l.strip() for l in b.splitlines() if l.strip() and not l.lstrip().startswith(("#", ">")))
              for b in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if b]


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?…])\s+(?=[¿¡A-ZÁÉÍÓÚÑ])", text) if s]


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def cut_short(y, last=0.02, floor_db=-38.0):
    """Whether a take, as the model gave it, stops while still sounding: it
    broke off in the middle of a word instead of dying away. (Heard as a
    glitch where two paragraphs meet; a whole take ends below -45 dB.)"""
    k = int(last * SR)
    return len(y) > k and float(np.sqrt(np.mean(y[-k:].astype(np.float64) ** 2))) > 10.0 ** (floor_db / 20.0)


def usable(speak, text, seed, tries=4):
    """A take of text whose length is plausible for its words, and which
    does not break off."""
    for attempt in range(tries):
        s = seed + 100 * attempt
        y = speak(text, s)
        if cut_short(y):
            print(f"    seed {s}: breaks off at the end, trying another", flush=True)
            continue
        y = voice.trim(y)
        if voice.plausible(spoken(text), len(y) / SR):
            return voice.tighten(y, LONGEST_PAUSE)
        print(f"    seed {s}: {len(y) / SR:.1f} s is not a plausible length for {len(text)} characters, trying another",
              flush=True)
    raise SystemExit(f"no usable take of {text[:60]!r}")


def drama(text, ref, ref_text, seed, temperature, top_p):
    """voice.say_drama, with the sampling settings exposed."""
    import mlx.core as mx
    model = voice.load(voice.DRAMA)
    mx.random.seed(seed)
    kw = dict(ref_audio=ref, ref_text=ref_text) if ref else {}
    parts = [np.array(r.audio, dtype=np.float32).reshape(-1)
             for r in model.generate(text=text, temperature=temperature, top_p=top_p, **kw)]
    return voice.to_sr(np.concatenate(parts), model.sample_rate) if parts else np.zeros(0, dtype=np.float32)


def spoken(text):
    """The text without its delivery cues, for judging how long it should run."""
    return re.sub(r"\s*\[[^\[\]]*\]\s*", " ", text).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("script")
    p.add_argument("--engine", choices=("drama3", "voxtral"), default="drama3")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--voice", default="es_female")
    p.add_argument("--ref")
    p.add_argument("--ref-text")
    p.add_argument("--ref-script", help="the reference says this script's first paragraph")
    p.add_argument("--reref", action="store_true")
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--out", help="folder for the result (default: out/ beside the script)")
    p.add_argument("--only", type=int, help="speak just the first N paragraphs")
    p.add_argument("--direction", help='a delivery cue put before every paragraph, such as "[calm]" or "[Scottish accent]"')
    p.add_argument("--name")
    a = p.parse_args()

    paras = paragraphs(a.script)[:a.only]
    if a.direction:
        paras = [a.direction + " " + x for x in paras]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.script)), "out")
    name = a.name or (f"drama3_seed{a.seed}" if a.engine == "drama3" else f"voxtral_{a.voice}")
    words = sum(len(spoken(x).split()) for x in paras)
    print(f"{len(paras)} paragraphs, {words} words, by {name}", flush=True)

    t0 = time.time()
    clips = []
    if a.engine == "drama3":
        ref, ref_text = a.ref, a.ref_text
        if a.ref_script:
            ref_text = paragraphs(a.ref_script)[0]
        for i, text in enumerate(paras):
            t1 = time.time()
            y = usable(lambda t, s: drama(t, ref, ref_text, s, a.temperature, a.top_p), text, a.seed)
            if ref is None or (a.reref and i == 0):     # the first paragraph becomes the voice
                ref, ref_text = os.path.join(out, f"{name}_voice.wav"), text
                voice.save(ref, y)
            clips.append(y)
            print(f"  {i + 1}/{len(paras)}: {len(y) / SR:.1f} s, made in {time.time() - t1:.0f} s", flush=True)
    else:
        for i, text in enumerate(paras):
            t1 = time.time()
            parts = [usable(lambda t, s: voice.say(t, a.voice, seed=s), s_, a.seed) for s_ in sentences(text)]
            y = np.concatenate([c for part in parts for c in (part, silence(SENTENCE))][:-1])
            clips.append(y)
            print(f"  {i + 1}/{len(paras)}: {len(y) / SR:.1f} s, made in {time.time() - t1:.0f} s", flush=True)

    y = np.concatenate([c for clip in clips for c in (clip, silence(BETWEEN))][:-1])
    y = np.concatenate([silence(0.3), y, silence(0.6)])
    wav = os.path.join(out, name + ".wav")
    voice.save(wav, y)
    at, where = 0.3, []                   # where each paragraph lies, for mix.py
    for clip in clips:
        where.append([round(at, 4), round(at + len(clip) / SR, 4)])
        at += len(clip) / SR + BETWEEN
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(dict(paragraphs=where), f)
    mp3 = os.path.join(out, name + ".mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k", mp3], check=True)
    secs = len(y) / SR
    print(f"{os.path.relpath(mp3, ROOT)}: {secs / 60:.1f} min, {words / secs * 60:.0f} words a minute, "
          f"made in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
