"""Download the Poly Haven scans the films use (CC0, free to use).

    uv run python shared/fetch_assets.py                      # every film's
    uv run python shared/fetch_assets.py --film example  # one film's

Each film lists what it needs in its timeline.py (TEXTURES and SKIES). The
scans themselves are shared: each lands once in assets/polyhaven/<id>/ with
an info.json holding its real-world tile size, so materials can be mapped at
true scale, and any film can use it.
"""
import argparse, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import films  # noqa: E402

ROOT = films.ROOT
DEST = os.path.join(ROOT, "assets", "polyhaven")
API = "https://api.polyhaven.com"
UA = {"User-Agent": "blender-render-pipeline/0.1"}

# our file name -> the names Poly Haven uses for that map (first one present wins)
MAPS = {"diff": ("Diffuse", "col1"), "rough": ("Rough",), "nor_gl": ("nor_gl",)}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return False
    tmp = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default=None, help="only this film's scans; left out, every film's")
    a = ap.parse_args()
    TEXTURES, SKIES = {}, {}
    for name in [a.film] if a.film else films.names():
        film = films.load(name)
        TEXTURES.update(film.TEXTURES)
        SKIES.update(film.SKIES)
    total = 0
    for asset, res in TEXTURES.items():
        d = os.path.join(DEST, asset)
        os.makedirs(d, exist_ok=True)
        files = get_json(f"{API}/files/{asset}")
        info = get_json(f"{API}/info/{asset}")
        dims = info.get("dimensions") or [1000, 1000]
        for short, keys in MAPS.items():
            key = next(k for k in keys if k in files)
            url = files[key][res]["jpg"]["url"]
            path = os.path.join(d, f"{short}.jpg")
            new = download(url, path)
            total += os.path.getsize(path)
            print(("got  " if new else "have ") + os.path.relpath(path, ROOT))
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump({"id": asset, "res": res, "size_m": dims[0] / 1000.0,
                       "name": info.get("name"), "license": "CC0, polyhaven.com"}, f, indent=1)
    for asset, res in SKIES.items():
        d = os.path.join(DEST, asset)
        os.makedirs(d, exist_ok=True)
        files = get_json(f"{API}/files/{asset}")
        path = os.path.join(d, "sky.hdr")
        new = download(files["hdri"][res]["hdr"]["url"], path)
        total += os.path.getsize(path)
        print(("got  " if new else "have ") + os.path.relpath(path, ROOT))
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump({"id": asset, "res": res, "license": "CC0, polyhaven.com"}, f, indent=1)
    print(f"{len(TEXTURES)} surfaces, {len(SKIES)} sky, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
