"""An episode read aloud by Drama 3 on a rented GPU. A trial.

    uv run python podcasts/gpu_narrate.py URL podcasts/example/01_lighthouse/script.txt --name episode
    uv run python podcasts/gpu_narrate.py URL SCRIPT.txt --voice narrator_long --temperature 0.8 --top-p 0.8 --how apart

URL is the speech server on the machine (see rent.py and gpu_bench.py). The
voice is a recording in podcasts/voices/ with what it says beside it
(<name>.wav and <name>.txt).

--how   chain  (the default) one after another, each also given the
               paragraph before it to follow on from. Of six versions
               compared on 2026-11-19 this one, with a short reference voice and the
               server's own temperature, sounded most natural
        apart  every paragraph on its own and all at once: quickest, but
               each starts cold, knowing only the voice recording
        whole  the whole script as one request. Does not work as the
               server is started in rent.py: one request stops at about 47
               seconds of speech, so only short texts come back complete

Writes <name>.wav, .mp3 and (except for whole) .json into out/ beside the
script, as narrate.py does. A chain also keeps every take in
out/<name>_takes/ as it arrives, and a second run of the same command goes
on from the first paragraph that is missing or whose text has changed.

One fault no check here catches: a take can come out sounding overdriven,
like a voice too close to a cheap microphone, without clipping or measuring
any differently, and a chain carries that tone into the next few paragraphs
(the first four paragraphs of one episode, 2026-11-20, after an [excited]
greeting). It has to be heard. The cure was to remake those paragraphs
(narrate.py --only 4 on the Mac) and put them in place of the bad ones.

RunPod's proxy gives up on a request that takes 100 seconds to start
answering, and on 2026-11-19 requests through it stalled at random (the
same request made inside the machine took two seconds). So a request is
given up on after 25 seconds and asked again, and the paragraph before goes
up as an mp3, a tenth the size of a wav.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "shared", "post"), os.path.join(ROOT, "shared"), os.path.dirname(os.path.abspath(__file__))]

import gpu_bench  # noqa: E402
import narrate  # noqa: E402
import voice  # noqa: E402

SR = voice.SR
VOICES = os.path.join(ROOT, "podcasts", "voices")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("script")
    p.add_argument("--voice", default="narrator")
    p.add_argument("--how", choices=("chain", "apart", "whole"), default="chain")
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--direction", help='a delivery cue put before every paragraph, such as "[calm]" or "[Scottish accent]"')
    p.add_argument("--name", required=True)
    a = p.parse_args()

    paras = narrate.paragraphs(a.script)
    if a.direction:
        paras = [a.direction + " " + x for x in paras]
    ref = gpu_bench.as_reference(os.path.join(VOICES, a.voice + ".wav"),
                                 open(os.path.join(VOICES, a.voice + ".txt"), encoding="utf-8").read().strip())

    def one(text, refs, tries=10):
        """A take that does not break off in the middle of a word. The server
        takes no seed, so asking again gives a different take."""
        for attempt in range(tries):
            try:
                y = gpu_bench.speak(a.url, text, refs, a.temperature, a.top_p, timeout=25)[0]     # a stalled request is given up on quickly
            except (SystemExit, OSError) as e:      # the proxy gives up on a request that is slow to start (524)
                print(f"  a take of {text[:40]!r} failed ({str(e)[:80]})" + (", asking again" if attempt + 1 < tries else ""), flush=True)
                if attempt + 1 == tries:
                    raise
                continue
            if not narrate.cut_short(y):
                return y
            print(f"  a take of {text[:40]!r} breaks off at the end" + (", asking again" if attempt + 1 < tries else ", kept"), flush=True)
        return y

    t0 = time.time()
    if a.how == "apart":
        with ThreadPoolExecutor(len(paras)) as pool:
            raw = list(pool.map(lambda t: one(t, ref), paras))
    elif a.how == "chain":
        # each take is kept as it arrives, so a run that is cut off (the machine's
        # time ran out at paragraph 18 of 26 on 2026-11-19) goes on from there
        takes = os.path.join(os.path.dirname(os.path.abspath(a.script)), "out", a.name + "_takes")
        os.makedirs(takes, exist_ok=True)
        raw = []
        for text in paras:
            kept = os.path.join(takes, f"{len(raw) + 1:02d}")
            if os.path.exists(kept + ".npy") and open(kept + ".txt", encoding="utf-8").read() == text:
                raw.append(np.load(kept + ".npy"))
                continue
            for later in range(len(raw) + 1, len(paras) + 1):       # what followed a changed paragraph is redone
                for ext in (".npy", ".txt"):
                    if os.path.exists(os.path.join(takes, f"{later:02d}{ext}")):
                        os.remove(os.path.join(takes, f"{later:02d}{ext}"))
            print(f"  {len(raw) + 1}/{len(paras)} at {time.time() - t0:.0f} s", flush=True)
            before = [gpu_bench.as_reference(voice.trim(raw[-1]), narrate.spoken(paras[len(raw) - 1]), small=True)] if raw else []
            raw.append(one(text, [ref] + before))
            np.save(kept + ".npy", raw[-1])
            open(kept + ".txt", "w", encoding="utf-8").write(text)
    else:
        raw = [one(" ".join(paras), ref)]
    made = time.time() - t0

    clips = [voice.tighten(voice.trim(y), narrate.LONGEST_PAUSE) for y in raw]
    texts = paras if a.how != "whole" else [" ".join(paras)]
    for i, (text, clip) in enumerate(zip(texts, clips)):
        n = len(narrate.spoken(text))
        if not 0.03 * n <= len(clip) / SR <= 0.14 * n + 1.2:
            print(f"  part {i + 1}: {len(clip) / SR:.1f} s is not a plausible length for its {n} characters", flush=True)
    whole = np.concatenate([c for clip in clips for c in (clip, narrate.silence(narrate.BETWEEN))][:-1])
    whole = np.concatenate([narrate.silence(0.3), whole, narrate.silence(0.6)])
    out = os.path.join(os.path.dirname(os.path.abspath(a.script)), "out")
    wav = os.path.join(out, a.name + ".wav")
    voice.save(wav, whole)
    if a.how != "whole":
        at, where = 0.3, []
        for clip in clips:
            where.append([round(at, 4), round(at + len(clip) / SR, 4)])
            at += len(clip) / SR + narrate.BETWEEN
        json.dump(dict(paragraphs=where), open(wav[:-4] + ".json", "w"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k", wav[:-4] + ".mp3"], check=True)
    words = sum(len(narrate.spoken(x).split()) for x in paras)
    secs = len(whole) / SR
    print(f"{os.path.relpath(wav[:-4] + '.mp3', ROOT)}: {secs / 60:.1f} min, {words / secs * 60:.0f} words a minute, "
          f"made in {made:.0f} s ({a.how})", flush=True)


if __name__ == "__main__":
    main()
