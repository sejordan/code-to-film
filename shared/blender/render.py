"""Build a shot from code and render frames of it.

    blender -b --factory-startup -P shared/blender/render.py -- \
        --film example --shot S01 --frames 0,24,48,71 --scale 25 --samples 48 --out out/stills/S01

--film    the film's folder under films/. Its shot list says which of its
          sets ("places") a shot uses; that file is imported from the film's
          folder and asked to build() the set and animate() the shot.
--out     where the frames go, inside the film's folder. Left out, it is
          out/frames/<shot>.

--frames  all | 0,24,48 | 10-20   (film frame numbers within the shot, from 0)
--nth N --offset K   render only every Nth frame, starting at K. This is how
                     several machines share a shot: each takes one offset, so
                     a machine that dies loses only its own share.
--scale   percent of 3840 x 2160. Test stills use 25, final frames 100.
Frames already on disk are skipped, so a stopped run can be started again.
"""
import argparse
import importlib
import json
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import core    # noqa: E402
import films   # noqa: E402


def parse_frames(spec, n):
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    films.add_argument(ap)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--frames", default="all")
    ap.add_argument("--nth", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--save", default=None, help="also save the built scene as a .blend")
    ap.add_argument("--denoise", default="gpu", choices=["gpu", "cpu", "off"],
                    help="where the denoiser runs; cpu and off exist for benchmarking")
    a = ap.parse_args(argv)

    film = films.load(a.film, a.variant).use()
    shot = film.shot(a.shot)
    core.FPS = film.FPS
    place = importlib.import_module(shot["place"])
    samples = a.samples or getattr(place, "SAMPLES", 160)
    out = film.path(a.out or os.path.join("out", "frames", a.shot))
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    sc = core.reset()
    device = core.setup_render(sc, samples=samples, scale=a.scale)
    if a.device != "AUTO":
        device = core.pick_device(sc, a.device)
    if a.denoise == "off":
        sc.cycles.use_denoising = False
    elif a.denoise == "cpu":
        sc.cycles.denoising_use_gpu = False
    st = place.build(sc)
    place.animate(sc, st, shot)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=film.path(a.save))
    w = sc.render.resolution_x * a.scale // 100
    h = sc.render.resolution_y * a.scale // 100
    print(f"BUILT {a.shot} in {time.time() - t0:.1f}s  device {device}  {w}x{h}  {samples} samples  denoise {a.denoise}", flush=True)

    todo = [i for i in parse_frames(a.frames, shot["frames"]) if i % a.nth == a.offset % a.nth]
    log = open(os.path.join(out, "render_log.jsonl"), "a")
    for i in todo:
        path = os.path.join(out, f"{a.shot}_{i:04d}.jpg")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"HAVE {os.path.basename(path)}")
            continue
        t1 = time.time()
        sc.frame_set(shot["block_start"] + i)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        secs = time.time() - t1
        print(f"FRAME {os.path.basename(path)} {secs:.1f}s", flush=True)
        log.write(json.dumps(dict(film=film.name, variant=film.variant, shot=a.shot, frame=i, seconds=round(secs, 2), device=device,
                                  scale=a.scale, samples=samples, host=os.uname().nodename)) + "\n")
        log.flush()
    print(f"DONE {len(todo)} frames in {time.time() - t0:.1f}s", flush=True)


main()
