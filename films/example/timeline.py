"""The Marble: a small example film, to show how a film is laid out.

A glass marble rolls across a table and knocks over a domino.

Everything else reads this file: the set, the renderer, the voice, the sound
and the edit. Change a length here and the whole film moves with it. Plain
Python with no imports, so Blender's Python and the project's own Python can
both load it (shared/films.py does the loading).
"""

FPS = 24
TITLE = "The Marble"
TITLE_SHOT = "S01"          # the title sits over this shot

# name, place (the file in this folder that builds the set), length in seconds, what happens
SHOTS = [
    dict(name="S01", place="table", seconds=6.0,
         what="Wide. A marble rolls in from the left across a wooden table. The camera follows it."),
    dict(name="S02", place="table", seconds=3.5,
         what="Low and close. The marble reaches a domino, knocks it flat, and rolls back a little."),
]

# Poly Haven scans, id -> resolution. shared/fetch_assets.py downloads them and
# only these go to rented machines. This film is all plain materials, so none.
TEXTURES = {}
SKIES = {}
