"""Put the finished episodes on the web: one page, and podcast feeds.

    uv run python podcasts/publish.py --dry-run     # build the site in podcasts/site/out/; upload nothing
    uv run python podcasts/publish.py               # build it, and send what is new or changed
    uv run python podcasts/publish.py --prune       # ...and remove what the site no longer has

    https://podcasts.example.com                    every episode, with tags to filter them by
    https://podcasts.example.com/feed.xml           the feed of everything, to add in a podcast player
    https://podcasts.example.com/history/feed.xml   the feed of one tag

The site is not about any one topic. Episodes carry tags; the page lists
them all, newest series first, and its tag buttons narrow the list (and the
feed address shown) to one tag. There is a feed for everything and one per
tag; a feed that holds a single series is marked to be heard in order.

Unlisted, not private: anyone with the address can listen. Search engines
are told to stay out (robots.txt, a noindex tag and header) and the feeds
are marked not to be listed in podcast directories. The page and the feeds
say the scripts and the voice are AI-made, and carry no name or email.
SITE_TITLE, LANGUAGE, MADE_BY and COLOURS below set the site's words and look.

To add an episode, add it to its series' series.json and run this. To add a
series, give its folder a series.json: every podcasts/*/series.json is
published (podcasts/example/series.json shows the shape).

    title        the series' name; shown with each episode ("A Short History, episode 3")
    tags         the tags of all its episodes, such as ["history"]: a-z, 0-9 and -; shown capitalised
    published    YYYY-MM-DD; episode N is dated that day at 12:00 UTC plus N minutes
    episodes     number, slug, audio (path inside the series folder), title, summary;
                 optionally its own "published" (an episode added later) and more "tags"

An episode's audio is stored under a name that carries the file's
fingerprint (history/05-river-7fcecc87.mp3). Remake an episode and it
gets a new address, so players fetch it again; its old file is named as
left over, and --prune removes it. Audio goes up first and the page last,
so the page never names audio that is not there yet. Nothing in the built
site depends on when it was built: run it twice and the second sends nothing.

Where it goes: any S3-compatible bucket that is public to read (Tigris on
Fly.io, Cloudflare R2, AWS S3, ...). Put its details in .env, never in code:

    PODCAST_S3_ENDPOINT   PODCAST_S3_BUCKET   PODCAST_S3_KEY_ID   PODCAST_S3_SECRET
    PODCAST_BASE_URL      the address the site is served at (or pass --base-url)

A bucket shows index.html only at /index.html, so something must serve it
at the bare address. podcasts/site/app/ is one way: a stock Caddy on one
small Fly.io machine that passes every request to the bucket, holds no key
and sleeps when idle (set your app name and bucket host in fly.toml, then
fly deploy podcasts/site/app --ha=false), with a CNAME from your domain to
the app. A bucket on Cloudflare R2 with a custom domain needs no app at all.

Tests: uv run python -m unittest podcasts/site/test_publish.py
"""
import argparse
import datetime
import email.utils
import glob
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(HERE, "site")
# Set these for your own site. The address can also come from PODCAST_BASE_URL in .env or --base-url.
BASE_URL = "https://podcasts.example.com"
SITE_TITLE = "Podcasts"
LANGUAGE = "en"                                  # the feeds' language code, such as en or es
AUTHOR = "Made with AI"
ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ATOM = "http://www.w3.org/2005/Atom"
COLOURS = ["#1F3B57", "#E0A030", "#5FA8D3"]      # every cover: its ground, and the two colours of its bands
MADE_BY = "The scripts and the voice are made with AI."
LONG, SHORT = "public, max-age=31536000, immutable", "public, max-age=300"
KEYS = ("PODCAST_S3_KEY_ID", "PODCAST_S3_SECRET", "PODCAST_S3_ENDPOINT", "PODCAST_S3_BUCKET")
FONTS = ["/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]

ET.register_namespace("itunes", ITUNES)
ET.register_namespace("atom", ATOM)


def load_series(root):
    """Every <root>/*/series.json, by folder name, with each episode's audio
    measured. An episode carries all the site needs: its series' title, its
    tags (the series' and its own), its date, its guid and its audio's address."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*", "series.json"))):
        folder = os.path.dirname(path)
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        s["slug"] = os.path.basename(folder)
        need(path, s, ("title", "published", "episodes"))
        day(path, s["published"])
        seen = set()
        for ep in s["episodes"]:
            need(path, ep, ("number", "slug", "audio", "title", "summary"))
            ep["published"] = ep.get("published", s["published"])
            day(path, ep["published"])
            ep["tags"] = list(dict.fromkeys(list(s.get("tags", [])) + list(ep.get("tags", []))))
            for tag in ep["tags"]:
                if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", tag):
                    raise SystemExit(f"{path}: the tag {tag!r} goes into an address, so only a-z, 0-9 and -")
            if ep["number"] in seen:
                raise SystemExit(f"{path}: two episodes are numbered {ep['number']}")
            seen.add(ep["number"])
            ep["path"] = os.path.join(folder, ep["audio"])
            if not os.path.isfile(ep["path"]):
                raise SystemExit(f"{path}: the audio of episode {ep['number']} is missing: {ep['path']}")
            ep["bytes"] = os.path.getsize(ep["path"])
            with open(ep["path"], "rb") as f:
                ep["sha"] = hashlib.sha256(f.read()).hexdigest()[:8]
            ep["seconds"] = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ep["path"]],
                capture_output=True, text=True, check=True).stdout)
        for slug in [s["slug"]] + [ep["slug"] for ep in s["episodes"]]:
            if not re.fullmatch(r"[a-z0-9_-]+", slug):
                raise SystemExit(f"{path}: \"{slug}\" goes into an address, so only a-z, 0-9, - and _")
        for ep in s["episodes"]:
            ep["series"], ep["folder"] = s["title"], s["slug"]
            ep["guid"] = f"{s['slug']}-{ep['number']:02d}"
            # The name carries the file's fingerprint, so a remade episode gets
            # a new address and players fetch it again.
            ep["key"] = f"{s['slug']}/{ep['number']:02d}-{ep['slug']}-{ep['sha']}.mp3"
        s["episodes"].sort(key=lambda ep: ep["number"])
        out.append(s)
    return out


def day(path, text):
    try:
        return datetime.date.fromisoformat(text)
    except (TypeError, ValueError):
        raise SystemExit(f"{path}: \"published\" must be a date like 2026-11-20, not {text!r}")


def need(path, d, fields):
    for field in fields:
        if field not in d:
            raise SystemExit(f"{path}: \"{field}\" is missing")


def clock(seconds):
    h, rest = divmod(round(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def listing(all_series):
    """Every episode in the order the page shows them: the newest series
    first, each series in its own order."""
    newest_first = sorted(all_series, key=lambda s: s["published"], reverse=True)
    return [ep for s in newest_first for ep in s["episodes"]]


def label(tag):
    return tag.replace("-", " ").capitalize()


def tags_of(episodes):
    return sorted({tag for ep in episodes for tag in ep["tags"]})


def feeds(all_series):
    """The feed for everything, then one for each tag."""
    eps = listing(all_series)
    out = [dict(path="feed.xml", cover="cover.jpg", title=SITE_TITLE, heading=SITE_TITLE, small="", episodes=eps,
                description=f"Every episode, on every topic. {MADE_BY}")]
    for tag in tags_of(eps):
        out.append(dict(path=f"{tag}/feed.xml", cover=f"{tag}/cover.jpg", title=f"{SITE_TITLE}: {label(tag)}",
                        heading=label(tag), small=SITE_TITLE, episodes=[ep for ep in eps if tag in ep["tags"]],
                        description=f"The episodes tagged {label(tag)}. {MADE_BY}"))
    return out


def feed_xml(feed, base_url):
    """One of feeds() as a podcast feed. Nothing in it depends on when it is
    made. A feed holding a single series is marked to be heard in order."""
    rss = ET.Element("rss", version="2.0")
    ch = ET.SubElement(rss, "channel")

    def add(parent, tag, text=None, **attrs):
        e = ET.SubElement(parent, tag, attrs)
        e.text = text
        return e
    serial = len({ep["folder"] for ep in feed["episodes"]}) == 1
    add(ch, "title", feed["title"])
    add(ch, "link", base_url + "/")
    add(ch, f"{{{ATOM}}}link", href=f"{base_url}/{feed['path']}", rel="self", type="application/rss+xml")
    add(ch, "description", feed["description"])
    add(ch, "language", LANGUAGE)
    add(ch, f"{{{ITUNES}}}author", AUTHOR)
    add(ch, f"{{{ITUNES}}}image", href=f"{base_url}/{feed['cover']}")
    add(ch, f"{{{ITUNES}}}block", "Yes")
    add(ch, f"{{{ITUNES}}}explicit", "false")
    add(ch, f"{{{ITUNES}}}type", "serial" if serial else "episodic")
    for ep in feed["episodes"]:
        noon = datetime.datetime.combine(datetime.date.fromisoformat(ep["published"]), datetime.time(12),
                                         datetime.timezone.utc)
        it = add(ch, "item")
        add(it, "title", f"{ep['series']} {ep['number']}: {ep['title']}")
        add(it, "description", ep["summary"])
        add(it, "link", f"{base_url}/#{ep['guid']}")
        add(it, "guid", ep["guid"], isPermaLink="false")
        add(it, "pubDate", email.utils.format_datetime(noon + datetime.timedelta(minutes=ep["number"]), usegmt=True))
        add(it, "enclosure", url=f"{base_url}/{ep['key']}", length=str(ep["bytes"]), type="audio/mpeg")
        add(it, f"{{{ITUNES}}}duration", str(round(ep["seconds"])))
        if serial:
            add(it, f"{{{ITUNES}}}episode", str(ep["number"]))
    ET.indent(rss)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(rss, encoding="unicode") + "\n"


def count(episodes):
    """As the page's script also writes it: '8 episodes, 85 minutes'."""
    n, minutes = len(episodes), round(sum(round(ep["seconds"]) for ep in episodes) / 60)
    return f"{n} episode{'' if n == 1 else 's'}, {minutes} minute{'' if minutes == 1 else 's'}"


def page_html(all_series, base_url):
    """The one page: every episode, the tags to filter them by, and the feed's
    address. Audio is linked relatively, so the page also plays from a folder.
    The filtering itself is the script in page.html; without it, all show."""
    t = html.escape
    eps = listing(all_series)
    chips = [f'<a class="tema" href="#" data-tema="">All <span>{len(eps)}</span></a>']
    for tag in tags_of(eps):
        n = sum(tag in ep["tags"] for ep in eps)
        chips.append(f'<a class="tema" href="#tema={tag}" data-tema="{tag}">{t(label(tag))} <span>{n}</span></a>')
    rows = []
    for ep in eps:
        key = t(ep["key"])
        links = "".join(f'<a href="#tema={tag}">{t(label(tag))}</a>' for tag in ep["tags"])
        rows.append(f"""  <li id="{ep['guid']}" data-temas="{' '.join(ep['tags'])}" data-segundos="{round(ep['seconds'])}">
    <p class="serie">{t(ep['series'])}, episode {ep['number']}</p>
    <div class="titulo"><h2>{t(ep['title'])}</h2><span class="duracion">{clock(ep['seconds'])}</span></div>
    <p class="resumen">{t(ep['summary'])}</p>
    <audio controls preload="none" src="{key}"></audio>
    <p class="pie">{links}<a class="descargar" href="{key}" download>Download</a></p>
  </li>""")
    feed = t(f"{base_url}/feed.xml")
    content = f"""<nav class="temas" aria-label="Tags">
  {chr(10).join("  " + c for c in chips).strip()}
</nav>
<p class="cuenta" id="cuenta">{count(eps)}</p>
<div class="suscribir" id="suscribir" data-base="{t(base_url)}/">
  <p>To listen in a podcast app, add this address:</p>
  <p class="direccion"><a href="{feed}">{feed}</a></p>
  <button type="button">Copy address</button>
</div>
<ol class="episodios">
{chr(10).join(rows)}
</ol>"""
    with open(os.path.join(SITE, "page.html"), encoding="utf-8") as f:
        return (f.read().replace("<!--CONTENIDO-->", content).replace("<!--TITLE-->", t(SITE_TITLE))
                .replace("<!--LANG-->", LANGUAGE).replace("<!--MADE_BY-->", t(MADE_BY)))


def cover(feed, path):
    """A plain square cover for a feed: its heading between two bands."""
    size, step = 1400, 100
    c1, c2, c3 = COLOURS
    img = Image.new("RGB", (size, size), c1)
    d = ImageDraw.Draw(img)
    for top, up in ((0, False), (size - step, True)):
        for i, x in enumerate(range(-step, size + step, step)):
            tip, base = (top, top + step) if up else (top + step, top)
            d.polygon([(x, base), (x + step, base), (x + step // 2, tip)], fill=c2 if i % 2 else c3)

    def font(px):
        for name in FONTS:
            if os.path.exists(name):
                return ImageFont.truetype(name, px)
        return ImageFont.load_default(px)
    big, lines = font(170), [""]
    for word in feed["heading"].split():
        trial = (lines[-1] + " " + word).strip()
        if lines[-1] and d.textlength(trial, font=big) > size - 240:
            lines.append(word)
        else:
            lines[-1] = trial
    y = size - step - 150 - 190 * len(lines)
    for line in lines:
        d.text((120, y), line, font=big, fill="white")
        y += 190
    if feed["small"]:
        d.text((124, step + 90), feed["small"], font=font(64), fill="white")
    img.save(path, "JPEG", quality=90)


def build(all_series, out_dir, base_url):
    """Write the whole site into out_dir. Returns its files in the order they
    should go up: audio and covers first, the feeds and the page last, so the
    page never names audio that is not there yet."""
    audio, covers, feed_files = [], [], []

    def entry(key, kind, cache, text=None):
        path = os.path.join(out_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if text is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return dict(key=key, path=path, type=kind, cache=cache)
    for s in all_series:
        for ep in s["episodes"]:
            e = entry(ep["key"], "audio/mpeg", LONG)
            if not os.path.exists(e["path"]) or os.path.getsize(e["path"]) != ep["bytes"]:
                shutil.copyfile(ep["path"], e["path"])
            audio.append(e)
    for feed in feeds(all_series):
        e = entry(feed["cover"], "image/jpeg", SHORT)
        cover(feed, e["path"])
        covers.append(e)
        feed_files.append(entry(feed["path"], "application/rss+xml; charset=utf-8", SHORT, feed_xml(feed, base_url)))
    robots = entry("robots.txt", "text/plain; charset=utf-8", SHORT, "User-agent: *\nDisallow: /\n")
    page = entry("index.html", "text/html; charset=utf-8", SHORT, page_html(all_series, base_url))
    return audio + covers + [robots] + feed_files + [page]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def plan(files, remote):
    """What to send, given what the bucket holds ({key: ETag}): the files it
    lacks or holds differently, in the order given, and the keys it holds
    that the site no longer has."""
    send = [f for f in files if remote.get(f["key"]) != md5(f["path"])]
    return send, sorted(set(remote) - {f["key"] for f in files})


def settings(env=None):
    """The bucket's address and key from .env. Never printed."""
    found = {}
    with open(env or os.path.join(ROOT, ".env")) as f:
        for line in f:
            name, _, value = line.strip().partition("=")
            if name in KEYS:
                found[name] = value.strip().strip('"').strip("'")
    for name in KEYS:
        if not found.get(name):
            raise SystemExit(f"{name} not found in .env")
    return found


def env_value(name):
    """One setting from .env, or from the environment; None if neither has it."""
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == name and value.strip():
                    return value.strip().strip('"').strip("'")
    return os.environ.get(name)


def publish(files, prune):
    """Send what the bucket lacks, the page last. Files the site no longer has
    are named, and removed only with prune."""
    import boto3
    from botocore.config import Config
    cfg = settings()
    bucket = cfg["PODCAST_S3_BUCKET"]
    s3 = boto3.client("s3", endpoint_url=cfg["PODCAST_S3_ENDPOINT"], region_name="auto",
                      aws_access_key_id=cfg["PODCAST_S3_KEY_ID"], aws_secret_access_key=cfg["PODCAST_S3_SECRET"],
                      config=Config(request_checksum_calculation="when_required",
                                    response_checksum_validation="when_required"))
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for o in page.get("Contents", []):
            remote[o["Key"]] = o["ETag"].strip('"')
    send, left = plan(files, remote)
    for f in send:
        with open(f["path"], "rb") as body:
            s3.put_object(Bucket=bucket, Key=f["key"], Body=body, ContentType=f["type"], CacheControl=f["cache"])
        print(f"  sent {f['key']}  ({os.path.getsize(f['path']) / 1e6:.1f} MB)")
    print(f"{len(send)} sent, {len(files) - len(send)} already there" if send else "nothing to upload")
    for key in left:
        if prune:
            s3.delete_object(Bucket=bucket, Key=key)
        print(f"  {'removed' if prune else 'left over (remove with --prune)'}: {key}")


def failure(e):
    """A failed request in a few words: what was asked and the service's error
    code. Never the service's own message, which can repeat the key's name."""
    code = getattr(e, "response", {}).get("Error", {}).get("Code") or type(e).__name__
    return f"{code} during {getattr(e, 'operation_name', 'the upload')}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="build the site locally and upload nothing")
    p.add_argument("--prune", action="store_true", help="remove files from the bucket that the site no longer has")
    p.add_argument("--base-url", default=None, help="the site's address; else PODCAST_BASE_URL from .env or the environment")
    a = p.parse_args()
    a.base_url = a.base_url or env_value("PODCAST_BASE_URL") or BASE_URL
    all_series = load_series(HERE)
    if not all_series:
        raise SystemExit("no series.json found under " + HERE)
    out = os.path.join(SITE, "out")
    files = build(all_series, out, a.base_url.rstrip("/"))
    for s in all_series:
        print(f"{s['title']}: {len(s['episodes'])} episodes, {clock(sum(ep['seconds'] for ep in s['episodes']))}")
    total = sum(os.path.getsize(f["path"]) for f in files)
    print(f"{len(files)} files, {total / 1e6:.0f} MB, built in {out}")
    if a.dry_run:
        print("nothing uploaded; open " + os.path.join(out, "index.html"))
        return
    try:
        publish(files, a.prune)
    except Exception as e:
        if type(e).__module__.split(".")[0] not in ("botocore", "boto3"):
            raise
        raise SystemExit(f"upload failed: {failure(e)}\nnothing is half-published: the page goes up last. Run it again.")
    print(a.base_url.rstrip("/"))


if __name__ == "__main__":
    main()
