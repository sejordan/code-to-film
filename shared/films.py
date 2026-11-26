"""Find a film and read its shot list.

Every film lives in its own folder under films/, with a timeline.py that
names it and lists its shots. Every shared tool takes --film NAME and calls
load(NAME) here, so nothing in shared/ knows about any one film.

A film can come in variants, say one per language: its timeline.py lists
them as VARIANTS = ("en", "es") and the tools take --variant. Each variant
has its own outputs, in out/<variant>/, and every shot is told which variant
it is being made for. TITLE can then be a dict with one title per variant.

Standard library only, so Blender's Python and the project's own Python can
both use it.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILMS = os.path.join(ROOT, "films")

# Inside Blender each shot owns a private block of frame numbers, a thousand
# apart, and is animated one frame past both ends. Motion blur looks at the
# frames either side of the one being rendered, so this keeps blur from ever
# reaching across a cut.
BLOCK = 1000


def names():
    """Every film there is."""
    if not os.path.isdir(FILMS):
        return []
    return sorted(n for n in os.listdir(FILMS) if os.path.exists(os.path.join(FILMS, n, "timeline.py")))


class Film:
    """One film: where it lives and what its timeline.py says."""

    def __init__(self, name, variant=None):
        self.name = name
        self.dir = os.path.join(FILMS, name)
        path = os.path.join(self.dir, "timeline.py")
        if not os.path.exists(path):
            raise SystemExit(f"no film called {name!r} in films/ (there is: {', '.join(names()) or 'none yet'})")
        spec = importlib.util.spec_from_file_location(f"timeline_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.VARIANTS = tuple(getattr(mod, "VARIANTS", ()))
        if self.VARIANTS:
            variant = variant or self.VARIANTS[0]
            if variant not in self.VARIANTS:
                raise SystemExit(f"{name} has no variant {variant!r} (it has: {', '.join(self.VARIANTS)})")
        elif variant:
            raise SystemExit(f"{name} has no variants; leave --variant out")
        self.variant = variant
        self.out = os.path.join(self.dir, "out", variant) if variant else os.path.join(self.dir, "out")
        self.FPS, self.SHOTS = mod.FPS, mod.SHOTS
        self.TITLE = mod.TITLE[variant] if isinstance(mod.TITLE, dict) else mod.TITLE
        self.TITLE_SHOT = getattr(mod, "TITLE_SHOT", None)
        self.TEXTURES = getattr(mod, "TEXTURES", {})
        self.SKIES = getattr(mod, "SKIES", {})

    def path(self, *parts):
        """A path inside the film's folder. Paths given to a tool on the
        command line are read this way, so out/frames means this film's,
        and for a film with variants, this variant's: out/<variant>/frames."""
        p = os.path.join(*parts)
        if os.path.isabs(p):
            return p
        first, _, rest = p.partition(os.sep)
        return os.path.join(self.out, rest) if first == "out" else os.path.join(self.dir, p)

    def use(self):
        """Let Python import the film's own files (its sets, its sound)."""
        if self.dir not in sys.path:
            sys.path.insert(0, self.dir)
        return self

    def n_frames(self, shot):
        return round(shot["seconds"] * self.FPS)

    def shot(self, name):
        for i, s in enumerate(self.SHOTS):
            if s["name"] == name:
                return dict(s, index=i, frames=self.n_frames(s), block_start=(i + 1) * BLOCK, variant=self.variant,
                            film_start=sum(self.n_frames(x) for x in self.SHOTS[:i]))
        raise KeyError(name)

    def all_shots(self):
        return [self.shot(s["name"]) for s in self.SHOTS]

    def total_frames(self):
        return sum(self.n_frames(s) for s in self.SHOTS)

    def at(self, shot, t=0.0):
        """Film time, in seconds, of a moment t seconds into a shot."""
        return self.shot(shot)["film_start"] / self.FPS + t


def load(name, variant=None):
    return Film(name, variant)


def here(file):
    """The film whose folder holds this file: here(__file__)."""
    return Film(os.path.basename(os.path.dirname(os.path.abspath(file))))


def add_argument(ap):
    ap.add_argument("--film", required=True, help="the film's folder name under films/: " + ", ".join(names()))
    ap.add_argument("--variant", default=None, help="for a film with variants (say en, es): which one")
