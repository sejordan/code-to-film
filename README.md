# code-to-film

Short films and podcasts made entirely from code, on a Mac:

- **Picture:** Blender's Cycles renderer, driven by Python. Every set, prop and camera move is a script.
- **Sound:** numpy arithmetic. Every knock, rumble, pad and piano note is synthesised; there are no sample files.
- **Speech:** two text-to-speech models running locally through [mlx-audio](https://github.com/Blaizzy/mlx-audio):
  Voxtral (fixed voices) and Fish Audio Drama 3 (voices from a reference recording, with delivery cues).
- **Edit:** ffmpeg, with a light film finish (glow, vignette, grain), title and fades.
- **Rented GPUs:** RunPod or Vast.ai, by the minute, when a render or a long narration would tie up the Mac.

`films/example` is a 9.5-second film (a marble knocks over a domino, with one narrated line) that runs through
every step, and `podcasts/example` is a one-minute episode that does the same for the podcast tools.

## What you need

- A Mac with Apple Silicon (mlx-audio runs the speech models on it). Picture and sound work anywhere Blender runs.
- [uv](https://docs.astral.sh/uv/) for Python, then `uv sync`.
- [Blender](https://www.blender.org/download/) 5.3 (the tools were written against it).
- ffmpeg (`brew install ffmpeg`).
- Optional: a RunPod API key in `.env`, or the `vastai` CLI signed in, to rent GPUs. See `.env.example`.

The speech models download from Hugging Face the first time they run, several GB each.

## Layout

```
films/<name>/        one film, and only that film
    timeline.py      title, shot list, the scans it needs
    <set>.py         a set ("place"): build() and animate(), and optionally trace() for the sound
    sound.py         the film's soundtrack: its layers and their balance
    dialogue.py      for a film with speech: who has which voice, the lines, and when
    out/             everything made for it: stills, preview, frames, voice, sound, the film
shared/              tools every film uses; none of them knows any one film
    films.py         finds a film and reads its timeline
    preview.py       low-res stills and preview frames on this machine
    fetch_assets.py  downloads the Poly Haven scans a film lists
    blender/         core (scene, materials, shapes), anim (timing, camera),
                     render and motion (the two Blender entry points)
    post/            synth (sound tools), voice (speech), soundtrack (runs a film's sound.py),
                     compose (the edit)
    cloud/           render one shot, or a whole film, on rented GPUs; hosts.py knows
                     how RunPod and Vast.ai each rent and remove a machine
podcasts/            a script read aloud as an episode, with music and story sounds; no picture
    narrate.py       text to one voice, on the Mac
    rent.py          rent one large-memory GPU and start Drama 3's speech server on it
    gpu_narrate.py   narrate on that server, a paragraph at a time
    gpu_bench.py     time the server
    sounds.py, mix.py  story sounds and a sparse piano under the voice
    publish.py       a static page and podcast feeds, sent to an S3-compatible bucket
```

Every film tool takes `--film NAME`. Paths given to a tool (`--out`, `--frames`, `--sound`) are inside that
film's folder. If it is about one film's story it lives in that film's folder; if a second film would want
it unchanged it lives in `shared/`. A film never imports from another film.

The film's folder comes first on Python's path, so don't name a file there after a shared one (`core`,
`anim`, `films`, `synth`, `voice`, `render`, `motion`).

## Making a film

```sh
B=/Applications/Blender.app/Contents/MacOS/Blender
uv run python shared/fetch_assets.py --film example                       # scans it lists (none here)
uv run python shared/post/voice.py lines --film example                   # speak dialogue.py; prints how lines fit
$B -b --factory-startup -P shared/blender/motion.py -- --film example     # motion the sound will follow
uv run python shared/preview.py --film example stills --shots S01,S02 --at 0.1,0.5,0.9
uv run python shared/preview.py --film example film --step 2              # every 2nd frame, low res
uv run python shared/post/soundtrack.py --film example
uv run python shared/post/compose.py --film example --frames out/preview --step 2 --out out/preview/preview.mp4
```

On an M-series Mac the example's stills take about 10 seconds and its whole preview about 2 minutes.

### How the parts talk to each other

- `voice.py lines` writes each line's recording to `out/voice/` and, per frame of film, how loud it is to
  `voice/lines.json`. A set can read that to open and close a character's mouth; `sound.py` reads the
  recordings into the mix. It ends with a table flagging any line that **runs past its shot** or lands
  **on top of the line before**. A line is only re-spoken when its words, voice or seed change.
- A set's `trace(shot)` reports what the sound should follow, per frame (in the example, where the marble
  is and how far the domino has fallen). `motion.py` runs every shot's animation without rendering and
  saves that to `out/motion.json`, which `soundtrack.py` hands to the film's `sound.py`.
- `compose.py` refuses to edit a film with any frame missing, so a half-rendered film is never delivered
  by mistake. `--step N` holds each preview frame N times so the timing stays true. The title fades in
  about 3 seconds into `TITLE_SHOT`, so give that shot 6 seconds or more.

### Starting a new film

Make `films/<name>/` with:

- `timeline.py`: `FPS`, `TITLE`, `TITLE_SHOT` (optional), `SHOTS` (each with `name`, `place`, `seconds`,
  `what`), and `TEXTURES` / `SKIES` (Poly Haven id → resolution). `VARIANTS = ("en", "es")` makes one
  version per language; the tools then take `--variant`.
- one file per `place` named in the shots, with `build(sc)` returning the set's parts (including `"cam"`),
  `animate(sc, st, shot)`, optionally `SAMPLES`, and optionally `trace(shot)`.
- `sound.py` (optional) with `build(film, motion)` returning the stereo mix, and `TAIL`.
- `dialogue.py` (optional) with `CAST` and `LINES`.

## Voices

```sh
uv run python shared/post/voice.py audition                                   # every English and Spanish Voxtral voice
uv run python shared/post/voice.py audition --voices casual_male --takes 4    # four takes of one voice
uv run python shared/post/voice.py audition --engine drama3 --takes 4         # four voices Drama 3 invents
```

|              | Voxtral (mistralai/Voxtral-4B-TTS-2603) | Fish Audio Drama 3 |
|--------------|------------------------------------------|--------------------|
| Voices       | 20 fixed: 5 English, 2 for most other languages | None fixed: speaks in the voice of a short reference recording, or invents one per seed |
| Steering     | Only words, punctuation and seed         | Cues in the text: `[sighs]`, `[excited]`, `[laughs]` |
| Licence      | CC BY-NC 4.0                             | Fish Audio Community License |

Both are for **non-commercial use only**. Never give Drama 3 a real person's voice as a reference without
their consent.

What has worked:

- The seed is the take. A line that sounds robotic is usually one bad seed: try several and choose by ear.
  A take much slower than about 14 characters a second is a warning sign.
- For English adult characters, Voxtral's fixed voices were more convincing than Drama 3's invented ones.
- In `CAST`, `pitch` (semitones) plays a voice higher, like a small toy, and `pause` caps long silences
  inside a line. In `LINES`, `seed` picks the take, for every language or per language (`{"es": 2}`).

## Rented GPUs

```sh
uv run python shared/cloud/film.py --film example --on vast --pods 2 --gpu "RTX 4090" --max-price 0.60 \
    --max-minutes 30 --step 2 --scale 25 --samples 40 --out out/preview --dry-run
uv run python shared/cloud/film.py --film example --pods 4 --gpu "NVIDIA GeForce RTX 4090" \
    --cloud SECURE --max-price 0.90 --max-minutes 120 --dry-run           # full 4K frames on RunPod
uv run python shared/cloud/render.py --cleanup                            # remove anything a crashed run left
```

Drop `--dry-run` to rent. Every missing frame is dealt out to the machines like cards, so each gets a fair
share of cheap and expensive shots. Each machine gets `shared/`, that film's folder (without `out/`) and
the scans it lists, installs Blender and renders its share. Finished frames come home every 30 seconds and
each machine is removed the moment its share is home. Running the same command again renders only what
is still missing.

Every machine also removes itself at `--max-minutes` whatever happens, and every run is recorded in
`.cloud/run-*.json` so `--cleanup` can remove what a crash left behind. A good habit: set the limit to
about twice the expected run, and remove machines as soon as the output is checked.

- **RunPod**: containers in data centres (`--cloud SECURE`) or on vetted hosts (`COMMUNITY`), one price per
  GPU model. The community price is much lower than secure; check which you will get.
- **Vast.ai** (`--on vast`): a marketplace of individual hosts, cheaper and with more stock. The tools
  take the cheapest verified, reliable, well-connected single-GPU machines in `--regions` (default `US,CA`).
  Traffic is billed by the gigabyte on top. Needs `vastai set api-key`; the project's SSH key is added
  to your Vast account the first time.

For scale: a 9.5-second 960×540 preview is a few minutes on the Mac. A 68-second one took 11 minutes on two
Vast RTX 4090s for about $0.13; the same run locally would have taken about 40 minutes.

The glTF models in `assets/sketchfab/` are not yet sent to rented machines, so a film using them renders
locally only.

## Podcasts

A script is plain text, paragraphs separated by blank lines. Lines starting with `#` are notes, lines
starting with `>` ask for story sounds, and bracketed cues steer Drama 3's delivery. See
`podcasts/example/01_lighthouse/script.txt`.

```sh
# on the Mac
uv run python podcasts/narrate.py podcasts/example/01_lighthouse/script.txt --engine voxtral --voice neutral_male --name voxtral
uv run python podcasts/narrate.py podcasts/example/01_lighthouse/script.txt --engine drama3 --seed 1
uv run python podcasts/mix.py podcasts/example/01_lighthouse/out/voxtral.wav
uv run python podcasts/sounds.py                              # one sample of every story sound

# on a rented GPU, for whole episodes in Drama 3
uv run python podcasts/rent.py                                # the price; rents nothing
uv run python podcasts/rent.py --go --minutes 90              # rent one, start the server, say how to reach it
uv run python podcasts/gpu_narrate.py URL podcasts/example/01_lighthouse/script.txt --voice narrator
uv run python podcasts/rent.py --remove

# the site
uv run python podcasts/publish.py --dry-run                   # build podcasts/site/out/; upload nothing
uv run python podcasts/publish.py
uv run python -m unittest podcasts/site/test_publish.py
```

- **Story sounds** are all synthesised. A *place* (`night`, `ducts`, `sea`, `city`, `hall`, `rain`) sets the
  scene as its paragraph begins, then fades after a few seconds; a sound running under a whole paragraph
  was distracting. A *spot* (`siren`, `alarm`, `door`, `camera`, `clang`, `suitcase`, `bell`, `rumble`)
  plays in a pause made for it just before its paragraph. Music sits about 30 dB under the voice.
- **A narrator voice for Drama 3**: run `narrate.py --engine drama3` with a few seeds. Each run saves its
  first paragraph as `out/<name>_voice.wav`; copy the one you like to `podcasts/voices/narrator.wav`, with
  the words it says in `podcasts/voices/narrator.txt`. `gpu_narrate.py --voice narrator` then reads every paragraph in that voice.
- **On the GPU**: Drama 3's 4-bit release needs about 21 GB for the model alone, so `rent.py` asks for a
  48 GB card (or larger). It runs the model's own server, SGLang-Omni; `rent.py`'s docstring has the exact
  setup commands and timings. One request makes at most about 47 seconds of speech, so a script goes a
  paragraph at a time. `--how chain` (the default) gives each paragraph the one before it as a second
  reference, which sounded most natural. Eight episodes (83 minutes of speech) took 37 minutes on one
  A100 80 GB, about $1.28.
- **Two faults to listen for.** A take can come out sounding overdriven without measuring any differently,
  and a chain carries that tone into the next few paragraphs; remake those paragraphs. And RunPod's proxy
  can stall requests at random, so a stalled request is retried after 25 seconds and every take is kept,
  so a second run carries on where the first stopped.
- **The site** (`publish.py`) is one page plus a feed for everything and one per tag, built from each
  `podcasts/*/series.json`. It is unlisted rather than private: search engines and podcast directories
  are told to stay out. Set `SITE_TITLE`, `LANGUAGE` and `COLOURS` at the top of `publish.py`, and the
  bucket and address in `.env`. The docstring covers hosting; `podcasts/site/app/` is a small Caddy app
  for Fly.io that serves a bucket at a bare address.

## Secrets

Keys live in `.env` (git-ignored) or in the `vastai` tool's own store, and no script prints them. The SSH key
for rented machines is made on first use in `.cloud/`, also git-ignored.

## Licence

The code is MIT. The speech models have their own non-commercial licences (above), as do any scans or
models you download (Poly Haven's are CC0; check each Sketchfab model's `license.txt`).
