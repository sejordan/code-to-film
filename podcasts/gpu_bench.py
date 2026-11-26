"""Time Drama 3 on a rented GPU, and bring back an episode it spoke. A trial.

    uv run python podcasts/gpu_bench.py https://PODID-8000.proxy.runpod.net podcasts/example/01_lighthouse/script.txt \
        --ref podcasts/example/01_lighthouse/out/drama3_seed1_voice.wav --ref-script podcasts/example/01_lighthouse/script.txt --name a40

The machine runs the model's own server (SGLang-Omni); this asks it for
speech over the web, as a stream, the way any program would. Three timings:
one paragraph on its own, all the script's paragraphs at once, and all of
them twice over at once. "Times real time" is seconds of speech made per
second of waiting.

The paragraphs from the all-at-once run are joined into an episode beside
the script, in out/gpu_<name>.wav (and .json and .mp3), which mix.py can
put music and sounds under. The reference recording is sent with each
request; never send a real person's voice without their consent.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "shared", "post"), os.path.join(ROOT, "shared"), os.path.dirname(os.path.abspath(__file__))]

import narrate  # noqa: E402
import voice  # noqa: E402

SR = voice.SR


def as_reference(samples_or_path, text, small=False):
    """A reference for a request: a recording (a .wav file, or samples at SR)
    packed into the request itself, and what it says."""
    if isinstance(samples_or_path, str):
        raw = open(samples_or_path, "rb").read()
    else:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(samples_or_path, -1, 1) * 32767.0).astype("<i2").tobytes())
        raw = buf.getvalue()
        if small:       # as mp3, a tenth the size: RunPod's proxy has hung on requests of several MB
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "-b:a", "128k", "pipe:1"],
                                 input=raw, capture_output=True, check=True).stdout
            return dict(audio_path="data:audio/mpeg;base64," + base64.b64encode(raw).decode(), text=text)
    return dict(audio_path="data:audio/wav;base64," + base64.b64encode(raw).decode(), text=text)


def speak(url, text, ref=None, temperature=None, top_p=None, timeout=900):
    """One request. ref is one reference or a list of them. Returns (samples
    at SR, seconds waited, seconds until the first sound arrived)."""
    body = dict(input=text, stream=True, response_format="pcm")
    if ref:
        body["references"] = ref if isinstance(ref, list) else [ref]
    if temperature:
        body["temperature"] = temperature
    if top_p:
        body["top_p"] = top_p
    req = urllib.request.Request(url.rstrip("/") + "/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "tts-bench/0.1"})
    t0 = time.time()
    first = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rate = int(r.headers.get("X-Sample-Rate") or 44100)
            chans = int(r.headers.get("X-Channels") or 1)
            bits = int(r.headers.get("X-Bit-Depth") or 16)
            raw = bytearray()
            while True:
                piece = r.read(65536)
                if not piece:
                    break
                if first is None:
                    first = time.time() - t0
                raw += piece
    except urllib.error.HTTPError as e:
        raise SystemExit(f"the server said {e.code}: {e.read().decode(errors='replace')[:600]}")
    waited = time.time() - t0
    assert bits == 16, bits
    y = np.frombuffer(bytes(raw[:len(raw) // 2 * 2]), dtype="<i2").astype(np.float32) / 32768.0
    if chans > 1:
        y = y[:len(y) // chans * chans].reshape(-1, chans).mean(axis=1)
    return voice.to_sr(y, rate), waited, first or waited


def together(url, texts, ref, temperature):
    t0 = time.time()
    with ThreadPoolExecutor(len(texts)) as pool:
        got = list(pool.map(lambda t: speak(url, t, ref, temperature), texts))
    wall = time.time() - t0
    return got, wall, sum(len(y) for y, _, _ in got) / SR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("script")
    p.add_argument("--ref")
    p.add_argument("--ref-script", help="the reference says this script's first paragraph")
    p.add_argument("--temperature", type=float)
    p.add_argument("--name", default="gpu")
    p.add_argument("--only", choices=("one", "all", "twice"), help="run just one of the three timings")
    a = p.parse_args()

    paras = narrate.paragraphs(a.script)
    ref = None
    if a.ref:
        ref = as_reference(a.ref, narrate.paragraphs(a.ref_script)[0])

    if a.only in (None, "one"):
        y, waited, first = speak(a.url, paras[1], ref, a.temperature)
        print(f"one paragraph alone: {len(y) / SR:.1f} s of speech in {waited:.1f} s = {len(y) / SR / waited:.2f} times real time "
              f"(first sound after {first:.1f} s)", flush=True)
    if a.only in (None, "all"):
        got, wall, spoken = together(a.url, paras, ref, a.temperature)
        print(f"{len(paras)} paragraphs at once: {spoken:.1f} s of speech in {wall:.1f} s = {spoken / wall:.2f} times real time", flush=True)
        out = os.path.join(os.path.dirname(os.path.abspath(a.script)), "out")
        clips = [voice.tighten(voice.trim(y), narrate.LONGEST_PAUSE) for y, _, _ in got]
        for i, (text, clip) in enumerate(zip(paras, clips)):
            if not voice.plausible(narrate.spoken(text), len(clip) / SR):
                print(f"  paragraph {i + 1}: {len(clip) / SR:.1f} s is not a plausible length for its words", flush=True)
        whole = np.concatenate([c for clip in clips for c in (clip, narrate.silence(narrate.BETWEEN))][:-1])
        whole = np.concatenate([narrate.silence(0.3), whole, narrate.silence(0.6)])
        wav = os.path.join(out, f"gpu_{a.name}.wav")
        voice.save(wav, whole)
        at, where = 0.3, []
        for clip in clips:
            where.append([round(at, 4), round(at + len(clip) / SR, 4)])
            at += len(clip) / SR + narrate.BETWEEN
        json.dump(dict(paragraphs=where), open(wav[:-4] + ".json", "w"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                        "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k", wav[:-4] + ".mp3"], check=True)
        print(f"  saved {os.path.relpath(wav[:-4] + '.mp3', ROOT)}: {len(whole) / SR / 60:.1f} min", flush=True)
    if a.only in (None, "twice"):
        got, wall, spoken = together(a.url, paras + paras, ref, a.temperature)
        print(f"{2 * len(paras)} paragraphs at once: {spoken:.1f} s of speech in {wall:.1f} s = {spoken / wall:.2f} times real time", flush=True)


if __name__ == "__main__":
    main()
