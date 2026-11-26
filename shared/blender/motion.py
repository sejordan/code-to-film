"""Write out how things move in every shot, for the sound to follow.

    blender -b --factory-startup -P shared/blender/motion.py -- --film example

Builds each of the film's sets once, runs every shot's animation without
rendering, and saves the film's out/motion.json: for each shot, whatever its
set's trace() reports per frame (for the example film, how far the marble
has rolled and whether it has hit the block). A set without a trace() is skipped.
"""
import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import core    # noqa: E402
import films   # noqa: E402

ap = argparse.ArgumentParser()
films.add_argument(ap)
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
film = films.load(a.film, a.variant).use()
core.FPS = film.FPS

out, built = {}, None
for sh in film.all_shots():
    place = importlib.import_module(sh["place"])
    if not hasattr(place, "trace"):
        continue
    if built is None or built[0] is not place:      # one build per set, not per shot
        sc = core.reset()
        core.setup_render(sc, scale=10)
        built = (place, place.build(sc))
    place.animate(sc, built[1], sh)
    out[sh["name"]] = place.trace(sh)
os.makedirs(film.out, exist_ok=True)
path = os.path.join(film.out, "motion.json")
with open(path, "w") as f:
    json.dump(out, f)
print("MOTION", path, {k: {n: len(v) for n, v in row.items()} for k, row in out.items()})
