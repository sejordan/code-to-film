"""Make a film's soundtrack.

    uv run python shared/post/soundtrack.py --film example

Runs the film's own sound.py, which builds the layers and their balance
from the shared tools in synth.py, then fades the ends, sets the level and
saves the film's out/sound.wav. If the film has an out/motion.json (from
shared/blender/motion.py) the sound is given it, so it can follow the
picture.

A film's sound.py offers build(film, motion), which returns the stereo mix, and
TAIL, the seconds of sound it wants after the last frame.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import films   # noqa: E402
import synth   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    films.add_argument(ap)
    ap.add_argument("--out", default="out/sound.wav", help="inside the film's folder")
    a = ap.parse_args()
    film = films.load(a.film, a.variant).use()
    import sound                                   # the film's own
    motion = {}
    if os.path.exists(film.path("out", "motion.json")):
        with open(film.path("out", "motion.json")) as f:
            motion = json.load(f)
    path = film.path(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(os.path.relpath(path, films.ROOT) + ":", end=" ")
    synth.finish(sound.build(film, motion), film, path, tail=getattr(sound, "TAIL", 0.0))


if __name__ == "__main__":
    main()
