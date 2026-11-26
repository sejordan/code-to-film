"""What is said in The Marble, and when.

shared/post/voice.py reads this, has each line spoken, and writes how long
it runs and how loud it is on every frame to voice/lines.json. A film with
characters reads that file to open and close their mouths; here the sound
reads the recording and places it in the mix.

A line's "at" is seconds into its shot; voice.py prints how the lines fit.
A line's "seed" picks its take: another seed is another reading.

Plain Python with no imports.
"""

CAST = {
    "narrator": dict(engine="voxtral", voice="neutral_male"),
}

LINES = [
    dict(id="L01", who="narrator", shot="S01", at=0.5,
         text="Every film here starts as a few lines of Python."),
]
