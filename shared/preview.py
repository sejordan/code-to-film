"""Quick looks at shots on this machine, before anything is rented.

    uv run python shared/preview.py --film example stills --shots S01,S02 --at 0.1,0.5,0.9
    uv run python shared/preview.py --film example film --step 2

stills  renders a few moments of each shot (fractions of its length) at low
        resolution and tiles them into one sheet per shot in the film's
        out/stills.
film    renders every Nth frame of every shot at low resolution into the
        film's out/preview. Join them with shared/post/compose.py
        --frames out/preview --step N, which holds each frame N times so the
        timing is true.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import films  # noqa: E402

BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")


def blender(film, shot, frames, out, scale, samples, nth=1):
    cmd = [BLENDER, "-b", "--factory-startup", "-P", os.path.join(HERE, "blender", "render.py"), "--",
           "--film", film.name, *(["--variant", film.variant] if film.variant else []), "--shot", shot, "--frames", frames, "--scale", str(scale), "--samples", str(samples), "--out", out,
           "--nth", str(nth)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("BUILT", "DONE", "LIGHT")) or "Error" in ln
             or "Traceback" in ln]
    if r.returncode != 0 or not any(ln.startswith("DONE") for ln in lines):
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{shot}: Blender failed")
    return lines


def stills(a, film):
    names = [s["name"] for s in film.SHOTS] if a.shots == "all" else a.shots.split(",")
    fracs = [float(x) for x in a.at.split(",")]
    for name in names:
        shot = film.shot(name)
        idx = sorted({min(shot["frames"] - 1, max(0, round(fr * (shot["frames"] - 1)))) for fr in fracs})
        out = os.path.join("out", "stills", name)
        shutil.rmtree(film.path(out), ignore_errors=True)
        t0 = time.time()
        blender(film, name, ",".join(str(i) for i in idx), out, a.scale, a.samples)
        files = [film.path(out, f"{name}_{i:04d}.jpg") for i in idx]
        sheet = film.path("out", "stills", f"{name}.jpg")
        inputs = []
        for p in files:
            inputs += ["-i", p]
        cols = min(len(files), a.cols)
        layout = "|".join(("+".join(["w0"] * (i % cols)) or "0") + "_" + ("+".join(["h0"] * (i // cols)) or "0")
                          for i in range(len(files)))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex",
                        f"xstack=inputs={len(files)}:layout={layout}:fill=black" if len(files) > 1 else "null",
                        "-q:v", "3", sheet], check=True)
        print(f"{name}: frames {idx} in {time.time() - t0:.0f}s -> {os.path.relpath(sheet, films.ROOT)}", flush=True)


def frames(a, film):
    names = [s["name"] for s in film.SHOTS] if a.shots == "all" else a.shots.split(",")
    for name in names:
        t0 = time.time()
        blender(film, name, "all", os.path.join("out", "preview", name), a.scale, a.samples, nth=a.step)
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    films.add_argument(ap)
    ap.add_argument("what", choices=["stills", "film"])
    ap.add_argument("--shots", default="all")
    ap.add_argument("--at", default="0.08,0.5,0.92")
    ap.add_argument("--scale", type=int, default=25)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3, help="stills: frames per row of the sheet")
    a = ap.parse_args()
    film = films.load(a.film, a.variant)
    stills(a, film) if a.what == "stills" else frames(a, film)


if __name__ == "__main__":
    main()
