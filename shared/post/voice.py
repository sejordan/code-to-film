"""Speech: lines of dialogue spoken by a text-to-speech model, on this Mac.

    uv run python shared/post/voice.py lines --film example    # a film's dialogue
    uv run python shared/post/voice.py audition                     # every English and Spanish voice
    uv run python shared/post/voice.py audition --voices casual_male,es_female --takes 3
    uv run python shared/post/voice.py audition --engine drama3 --lang es --takes 4
    uv run python shared/post/voice.py audition --engine drama3 --lang es --ref clip.wav --ref-text "what it says"

Two models can do the speaking. Both run through mlx-audio from mlx-community
conversions of the weights, and both are free for non-commercial use only.

voxtral  Voxtral TTS (mistralai/Voxtral-4B-TTS-2603, CC BY-NC 4.0). Twenty
         fixed voices: five English, two each for most other languages. No
         new or copied voices, and no dial for speed or feeling.
drama3   Fish Audio Drama 3 (fishaudio/drama-3, Fish Audio Community License).
         No fixed voices: it speaks in the voice of a short reference
         recording (given with its transcript) and takes on much of that
         recording's accent, or invents a voice when given none. Delivery can
         be steered with cues in the text, such as [sighs] or [excited].
         Never give it a real person's voice without their consent.

With either, delivery also comes from the words and punctuation, and from the
seed: the same line with another seed is another take. Whatever the model's
own rate, a line comes back at SR, mono.

lines   speaks every line in a film's dialogue.py (CAST says who has which
        voice, LINES what is said and when). The recordings go to the film's
        out/voice/; how long each runs and how loud it is on every frame of
        film goes to the film's voice/<variant>.json, which the picture reads
        to move mouths. That file is small and travels to rented machines;
        the recordings stay home. A line already spoken is not spoken again
        unless its words, voice or seed change. Ends with a table of how the
        lines fit their shots. In CAST, "pitch" plays a voice higher and
        "pause" caps the silences inside a line; in LINES, "seed" picks the
        take, for every language or as {"es": 2} for one.
audition writes one clip per voice and take into auditions/<engine>/, so
        the voices can be chosen by ear. For drama3 each take without a
        reference is a different invented voice.
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import films  # noqa: E402

MODEL = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
DRAMA = "mlx-community/fish-audio-drama-3-3bit"
SR = 48000                      # every line is brought to this, the soundtrack's own rate
VOICES = {
    "en": ["casual_male", "casual_female", "cheerful_female", "neutral_male", "neutral_female"],
    "es": ["es_male", "es_female"],
    "fr": ["fr_male", "fr_female"],
    "de": ["de_male", "de_female"],
    "it": ["it_male", "it_female"],
    "pt": ["pt_male", "pt_female"],
    "nl": ["nl_male", "nl_female"],
    "ar": ["ar_male"],
    "hi": ["hi_male", "hi_female"],
}
_loaded = (None, None)          # (which model, the model): one at a time, they are large


def load(name=MODEL):
    """A model, loaded once. Voxtral is about 8 GB in memory and Drama 3
    about 23 GB, so loading one lets go of the other."""
    global _loaded
    if _loaded[0] != name:
        import mlx.core as mx
        from mlx_audio.tts.utils import load as mlx_load
        _loaded = (None, None)
        mx.clear_cache()
        _loaded = (name, mlx_load(name))
    return _loaded[1]


def to_sr(y, rate):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if rate == SR or len(y) == 0:
        return y
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(SR, int(rate))
    return resample_poly(y, SR // g, int(rate) // g).astype(np.float32)


def say(text, voice, seed=0, temperature=0.8):
    """One take of a line by Voxtral: mono samples at SR, between -1 and 1."""
    import mlx.core as mx
    model = load(MODEL)
    mx.random.seed(seed)
    parts = [np.array(r.audio, dtype=np.float32).reshape(-1)
             for r in model.generate(text=text, voice=voice, temperature=temperature)]
    return to_sr(np.concatenate(parts), 24000) if parts else np.zeros(0, dtype=np.float32)


def say_drama(text, ref=None, ref_text=None, seed=0):
    """One take of a line by Drama 3. ref is a recording of the voice to
    speak in and ref_text what that recording says; with neither, the model
    invents a voice, a different one for every seed."""
    import mlx.core as mx
    model = load(DRAMA)
    mx.random.seed(seed)
    kw = dict(ref_audio=ref, ref_text=ref_text) if ref else {}
    parts = [np.array(r.audio, dtype=np.float32).reshape(-1) for r in model.generate(text=text, **kw)]
    return to_sr(np.concatenate(parts), model.sample_rate) if parts else np.zeros(0, dtype=np.float32)


def trim(y, floor_db=-45.0, keep=0.04):
    """Drop the silence before the first sound and after the last, keeping a
    short margin so nothing is clipped."""
    if len(y) == 0:
        return y
    k = int(0.01 * SR)
    level = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(k) / k, mode="same"))
    loud = np.nonzero(level > 10.0 ** (floor_db / 20.0))[0]
    if len(loud) == 0:
        return y[:0]
    m = int(keep * SR)
    return y[max(0, loud[0] - m):loud[-1] + m]


def envelope(y, fps):
    """How loud the voice is on each frame of film, 0 to 1: what a mouth, a
    light or a nod follows."""
    n = int(np.ceil(len(y) / SR * fps))
    step = SR / fps
    rms = np.array([np.sqrt(np.mean(y[int(i * step):int((i + 1) * step)].astype(np.float64) ** 2) + 1e-12)
                    for i in range(n)])
    return np.clip(rms / (rms.max() + 1e-12), 0.0, 1.0)


def pitched(y, semitones):
    """Played faster to sound higher, like a small toy: pitch and pace rise
    together."""
    if not semitones:
        return y
    rate = 2.0 ** (semitones / 12.0)
    return np.interp(np.arange(0, len(y) - 1, rate), np.arange(len(y)), y).astype(np.float32)


def run_down(y, start=0.35, to=0.25, fade=True):
    """A spring giving out: from `start` (a fraction of the way through) the
    voice is played slower and slower, down to `to` times its speed at the
    last sample, so it drops in pitch as it drags. With fade it also dies
    away to nothing; without, it only sags. Every word is still there; the
    line just takes longer."""
    i = int(start * len(y))
    left = len(y) - 1 - i
    if left < 2:
        return y
    n = int(left / (1.0 - (1.0 - to) * 0.4))        # the mean speed of the slide below is 1 - 0.4 (1 - to)
    u = np.linspace(0.0, 1.0, n)
    pos = i + np.cumsum(1.0 - (1.0 - to) * u ** 1.5)
    pos = np.minimum(pos, len(y) - 1)
    tail = np.interp(pos, np.arange(len(y)), y)
    if fade:
        tail = tail * (1.0 - u ** 3)
    return np.concatenate([y[:i], tail]).astype(np.float32)


def longest_pause(y, floor_db=-38.0, least=0.12):
    """The longest quiet stretch in the middle of a take, as (start, end)
    in samples, or None if nothing there is quiet for `least` seconds."""
    k = int(0.02 * SR)
    level = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(k) / k, mode="same"))
    quiet = level < 10.0 ** (floor_db / 20.0)
    lo, hi = int(0.15 * len(y)), int(0.88 * len(y))
    best, start = None, None
    for i in range(lo, hi + 1):
        if i < hi and quiet[i]:
            start = i if start is None else start
        elif start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best if best and best[1] - best[0] >= least * SR else None


def tighten(y, most, floor_db=-38.0):
    """No silence inside a take longer than `most` seconds: the middle of any
    longer one is cut out, with a short crossfade over the join. Drama 3 can
    leave a second or more after a question mark or an ellipsis."""
    k, x = int(0.02 * SR), int(0.01 * SR)
    level = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(k) / k, mode="same"))
    quiet = np.concatenate([level < 10.0 ** (floor_db / 20.0), [False]])
    keep, out, start = int(most * SR / 2), [], None
    last = 0
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if start > 0 and i < len(y) and i - start > most * SR:
                a, b = start + keep, i - keep           # drop y[a:b], fading across x samples
                ramp = np.linspace(0.0, 1.0, x)
                out.append(y[last:a - x])
                out.append(y[a - x:a] * (1.0 - ramp) + y[b - x:b] * ramp)
                last = b
            start = None
    out.append(y[last:])
    return np.concatenate(out).astype(np.float32)


def save(path, y, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767.0).astype("<i2").tobytes())


# ---------------------------------------------------------------- audition

AUDITION = {
    "en": ["Hello? Is anyone there? I can't see a thing.",
           "Well, that went better than expected. Shall we try it again?"],
    "es": ["¿Hola? ¿Hay alguien ahí? No veo nada.",
           "Bueno, salió mejor de lo que esperaba. ¿Lo intentamos otra vez?"],
}


def audition_drama(a):
    """Drama 3 has no list of voices. With --ref, each take is that voice
    again; without, each take is a newly invented voice. Both audition lines
    are spoken in one go, so a take is one voice throughout."""
    out = os.path.join(films.ROOT, "auditions", "drama3")
    text = " ".join(AUDITION.get(a.lang, AUDITION["en"]))
    label = a.label or ("ref" if a.ref else "invented")
    for take in range(a.takes):
        t0 = time.time()
        y = trim(say_drama(text, a.ref, a.ref_text, seed=a.seed + take))
        path = os.path.join(out, f"{a.lang}_{label}_take{take + 1}.wav")
        save(path, y)
        ok = "" if plausible(text, len(y) / SR) else "  << an odd length for these words"
        print(f"{os.path.relpath(path, films.ROOT)}: {len(y) / SR:.1f} s, peak "
              f"{20 * np.log10(np.abs(y).max() + 1e-9):.1f} dBFS, made in {time.time() - t0:.0f} s{ok}", flush=True)


def audition(a):
    if a.engine == "drama3":
        return audition_drama(a)
    out = os.path.join(films.ROOT, "auditions", "voxtral")
    wanted = a.voices.split(",") if a.voices else VOICES["en"] + VOICES["es"]
    gap = np.zeros(int(0.6 * SR), dtype=np.float32)
    for voice in wanted:
        lang = next(k for k, v in VOICES.items() if voice in v)
        lines = AUDITION.get(lang, AUDITION["en"])
        for take in range(a.takes):
            t0 = time.time()
            clips = [trim(say(text, voice, seed=a.seed + take)) for text in lines]
            y = np.concatenate([c for clip in clips for c in (clip, gap)][:-1])
            path = os.path.join(out, f"{voice}_take{take + 1}.wav")
            save(path, y)
            peak = 20 * np.log10(np.abs(y).max() + 1e-9)
            print(f"{os.path.relpath(path, films.ROOT)}: {len(y) / SR:.1f} s "
                  f"(lines {', '.join(f'{len(c) / SR:.1f}' for c in clips)} s), peak {peak:.1f} dBFS, "
                  f"made in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- a film's lines

def _drama_line(text, spec, seed):
    ref = spec.get("ref")
    return say_drama(text, os.path.join(films.ROOT, ref) if ref else None, spec.get("ref_text"), seed=seed)


# engine name -> (text, the speaker's entry in a film's CAST, seed) -> samples at SR
ENGINES = {"voxtral": lambda text, spec, seed: say(text, spec["voice"], seed=seed),
           "drama3": _drama_line}


def plausible(text, seconds):
    """Speech runs at very roughly 15 characters a second. A take far off
    that has gone wrong (cut short, or wandered off into silence)."""
    n = len(text)
    return 0.03 * n + 0.2 <= seconds <= 0.14 * n + 1.2


def take(text, spec, seed, split, sag=None):
    """One usable take: (seed used, first part, second part or None). Tries
    further seeds if a take's length is implausible or, for a line that is
    to be cut in two, it has no pause to cut at. sag is a line's runs_down:
    the first part (or the whole line) is slowed as run_down describes."""
    def finish(y, first=True):
        if spec.get("pause"):
            y = tighten(y, spec["pause"])
        y = pitched(y, spec.get("pitch", 0.0))
        return run_down(y, **sag) if sag and first else y

    for attempt in range(6):
        s = seed + 100 * attempt
        y = trim(ENGINES[spec.get("engine", "voxtral")](text, spec, s))
        if not plausible(text, len(y) / SR):
            print(f"    seed {s}: {len(y) / SR:.1f} s is not a plausible length, trying another", flush=True)
            continue
        if not split:
            return s, finish(y), None
        gap = longest_pause(y)
        if gap is None:
            print(f"    seed {s}: no pause to cut at, trying another", flush=True)
            continue
        a, b = trim(y[:gap[0]]), trim(y[gap[1]:])
        return s, finish(a), finish(b, first=False)
    raise SystemExit(f"no usable take of {text!r} in six tries")


def lines(a):
    film = films.load(a.film, a.variant).use()
    import dialogue                                 # the film's own
    lang = film.variant
    data_path = film.path("voice", f"{lang or 'lines'}.json")
    data = {}
    if os.path.exists(data_path) and not a.again:
        with open(data_path) as f:
            data = json.load(f)
    out = {}

    def entry(line_id, y, made):
        path = film.path("out", "voice", line_id + ".wav")
        if y is not None:
            save(path, y)
            out[line_id] = dict(made=made, seconds=round(len(y) / SR, 3),
                                env=[round(float(v), 3) for v in envelope(y, film.FPS)])
        else:
            out[line_id] = data[line_id]

    for ln in dialogue.LINES:
        spec = dialogue.CAST[ln["who"]]
        spec = spec[lang] if lang else spec
        text = ln[lang] if lang else ln["text"]
        seed = ln.get("seed", 1)                        # one for every language, or one per language
        seed = seed.get(lang, 1) if isinstance(seed, dict) else seed
        made = dict(text=text, seed=seed, runs_down=ln.get("runs_down"), **spec)
        ids = [ln["id"]] + ([ln["rest"]["id"]] if "rest" in ln else [])
        fresh = all(data.get(i, {}).get("made") == made and os.path.exists(film.path("out", "voice", i + ".wav"))
                    for i in ids)
        if fresh:
            for i in ids:
                entry(i, None, made)
            continue
        t0 = time.time()
        used, first, second = take(text, spec, made["seed"], "rest" in ln, ln.get("runs_down"))
        entry(ids[0], first, made)
        if second is not None:
            entry(ids[1], second, made)
        print(f"  {ln['id']} {ln['who']:5s} seed {used}: " + " + ".join(f"{out[i]['seconds']:.2f} s" for i in ids)
              + f"  ({time.time() - t0:.0f} s to make)  {text}", flush=True)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(out, f, indent=0)

    # how the lines sit in their shots
    cues = []
    for ln in dialogue.LINES:
        cues.append((ln["shot"], ln["at"], ln["id"], ln["who"]))
        if "rest" in ln:
            cues.append((ln["rest"]["shot"], ln["rest"]["at"], ln["rest"]["id"], ln["who"]))
    print(f"{film.name} {lang or ''}: {len(out)} lines, {sum(v['seconds'] for v in out.values()):.1f} s of speech")
    for sh in film.all_shots():
        mine = sorted(c for c in cues if c[0] == sh["name"])
        note, prev_end = [], 0.0
        for _, at, line_id, who in mine:
            end = at + out[line_id]["seconds"]
            flag = ""
            if end > sh["seconds"] - 0.25:
                flag = "  << RUNS PAST THE SHOT"
            elif at < prev_end + 0.15:
                flag = "  << ON TOP OF THE LINE BEFORE"
            note.append(f"{line_id} {who} {at:.1f}-{end:.1f}{flag}")
            prev_end = end
        print(f"  {sh['name']} {sh['seconds']:4.1f} s  " + ("; ".join(note) if note else "-"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lines", "audition"])
    ap.add_argument("--film", default=None, help="lines: the film's folder name under films/")
    ap.add_argument("--variant", default=None, help="lines: for a film with variants (say en, es): which one")
    ap.add_argument("--again", action="store_true", help="lines: speak every line afresh")
    ap.add_argument("--engine", default="voxtral", choices=sorted(ENGINES), help="audition: which model")
    ap.add_argument("--lang", default="en", help="audition, drama3: which language's lines to speak")
    ap.add_argument("--ref", default=None, help="audition, drama3: a recording of the voice to speak in")
    ap.add_argument("--ref-text", default=None, help="audition, drama3: what that recording says")
    ap.add_argument("--label", default=None, help="audition, drama3: a name for these takes in the file names")
    ap.add_argument("--voices", default=None, help="comma-separated; left out, every English and Spanish voice")
    ap.add_argument("--takes", type=int, default=1, help="takes per voice, each with its own seed")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.what == "lines":
        if not a.film:
            ap.error("lines needs --film")
        lines(a)
    else:
        audition(a)


if __name__ == "__main__":
    main()
