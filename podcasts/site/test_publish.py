"""Tests for podcasts/publish.py.

    uv run python -m unittest podcasts/site/test_publish.py -v
"""
import email.utils
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import publish  # noqa: E402

ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
BASE = "https://example.test"


def read(path, mode="rb"):
    with open(path, mode) as f:
        return f.read()


def silent_mp3(path, seconds=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", str(seconds), "-b:a", "64k", path], check=True)


class Case(unittest.TestCase):
    """Two series: a_first (tag alpha; its second episode is also tagged
    shared) published first, and b_second (tags beta and shared) later."""

    def setUp(self):
        self.top = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.top)
        self.root = os.path.join(self.top, "podcasts")

        def also_shared(series):
            series["episodes"][1]["tags"] = ["shared"]
        self.write_series("a_first", ["One", "Two"], tags=["alpha"], change=also_shared)
        self.write_series("b_second", ["One", "Two"], tags=["beta", "shared"], published="2026-12-01")

    def write_series(self, slug, titles, tags=(), published="2026-11-20", change=None):
        eps = []
        for n, title in enumerate(titles, 1):
            silent_mp3(self.audio(slug, n))
            eps.append(dict(number=n, slug="part%d" % n, audio="ep%d.mp3" % n, title=title, summary="Summary %d." % n))
        series = dict(title="Series " + slug, tags=list(tags), published=published, episodes=eps)
        if change:
            change(series)
        with open(os.path.join(self.root, slug, "series.json"), "w") as f:
            json.dump(series, f)

    def audio(self, slug, n):
        return os.path.join(self.root, slug, "ep%d.mp3" % n)

    def load(self):
        return publish.load_series(self.root)

    def feed(self, path):
        return next(f for f in publish.feeds(self.load()) if f["path"] == path)

    def channel(self, path):
        return ET.fromstring(publish.feed_xml(self.feed(path), BASE)).find("channel")

    def tmp(self, name):
        return os.path.join(self.top, name)

    def file(self, data):
        path = self.tmp(hashlib.md5(data).hexdigest())
        with open(path, "wb") as f:
            f.write(data)
        return path


class Catalogue(Case):
    def test_catalogue_order_and_measures(self):
        s = self.load()
        self.assertEqual([x["slug"] for x in s], ["a_first", "b_second"])
        self.assertEqual([e["number"] for e in s[0]["episodes"]], [1, 2])
        first = s[0]["episodes"][0]
        self.assertEqual(first["bytes"], os.path.getsize(first["path"]))
        self.assertAlmostEqual(first["seconds"], 1.0, delta=0.2)
        self.assertEqual(first["key"], "a_first/01-part1-%s.mp3" % first["sha"])
        self.assertEqual(first["guid"], "a_first-01")
        self.assertEqual(first["series"], "Series a_first")

    def test_episodes_carry_their_series_tags_and_their_own(self):
        a, b = self.load()
        self.assertEqual([e["tags"] for e in a["episodes"]], [["alpha"], ["alpha", "shared"]])
        self.assertEqual([e["tags"] for e in b["episodes"]], [["beta", "shared"], ["beta", "shared"]])

    def test_bad_tag_names_the_file(self):
        self.write_series("a_first", ["One", "Two"], tags=["Not A Tag"])
        with self.assertRaises(SystemExit) as c:
            self.load()
        self.assertIn(os.path.join("a_first", "series.json"), str(c.exception))
        self.assertIn("Not A Tag", str(c.exception))

    def test_listing_puts_the_newest_series_first_and_keeps_episode_order(self):
        self.assertEqual([e["guid"] for e in publish.listing(self.load())],
                         ["b_second-01", "b_second-02", "a_first-01", "a_first-02"])

    def test_missing_audio_stops(self):
        os.remove(self.audio("a_first", 2))
        with self.assertRaises(SystemExit) as c:
            self.load()
        self.assertIn("ep2.mp3", str(c.exception))

    def test_repeated_episode_number_stops(self):
        def twice(series):
            series["episodes"][1]["number"] = 1
        self.write_series("a_first", ["One", "Two"], change=twice)
        with self.assertRaises(SystemExit) as c:
            self.load()
        self.assertIn("a_first", str(c.exception))

    def test_clock(self):
        self.assertEqual(publish.clock(835.76), "13:56")
        self.assertEqual(publish.clock(3605), "1:00:05")

    def test_patched_audio_gets_new_address(self):
        before = self.load()[0]["episodes"][0]
        silent_mp3(self.audio("a_first", 1), seconds=2)
        after = self.load()[0]["episodes"][0]
        self.assertNotEqual(before["key"], after["key"])
        self.assertEqual(before["guid"], after["guid"])


class Feed(Case):
    def test_there_is_one_feed_for_everything_and_one_per_tag(self):
        feeds = publish.feeds(self.load())
        self.assertEqual([f["path"] for f in feeds], ["feed.xml", "alpha/feed.xml", "beta/feed.xml", "shared/feed.xml"])
        self.assertEqual([f["cover"] for f in feeds], ["cover.jpg", "alpha/cover.jpg", "beta/cover.jpg", "shared/cover.jpg"])
        guids = {f["path"]: [e["guid"] for e in f["episodes"]] for f in feeds}
        self.assertEqual(guids["feed.xml"], ["b_second-01", "b_second-02", "a_first-01", "a_first-02"])
        self.assertEqual(guids["alpha/feed.xml"], ["a_first-01", "a_first-02"])
        self.assertEqual(guids["shared/feed.xml"], ["b_second-01", "b_second-02", "a_first-02"])

    def test_feed_is_complete(self):
        feed = self.feed("feed.xml")
        text = publish.feed_xml(feed, BASE)
        ch = ET.fromstring(text).find("channel")
        self.assertEqual(ch.find("title").text, publish.SITE_TITLE)
        self.assertEqual(ch.find(ITUNES + "block").text, "Yes")
        self.assertEqual(ch.find(ITUNES + "image").get("href"), BASE + "/cover.jpg")
        self.assertEqual(ch.find("{http://www.w3.org/2005/Atom}link").get("href"), BASE + "/feed.xml")
        self.assertEqual(ch.find("language").text, publish.LANGUAGE)
        self.assertIn("AI", ch.find("description").text)
        items = ch.findall("item")
        self.assertEqual(len(items), 4)
        for ep, it in zip(feed["episodes"], items):
            enc = it.find("enclosure")
            self.assertEqual(enc.get("url"), BASE + "/" + ep["key"])
            self.assertEqual(int(enc.get("length")), ep["bytes"])
            self.assertEqual(enc.get("type"), "audio/mpeg")
            self.assertEqual(it.find("guid").text, ep["guid"])
            self.assertEqual(it.find("title").text, "%s %d: %s" % (ep["series"], ep["number"], ep["title"]))
        self.assertNotIn("@", text)

    def test_a_tag_feed_is_named_for_its_tag(self):
        ch = self.channel("alpha/feed.xml")
        self.assertEqual(ch.find("title").text, publish.SITE_TITLE + ": Alpha")
        self.assertEqual(ch.find(ITUNES + "image").get("href"), BASE + "/alpha/cover.jpg")
        self.assertEqual(ch.find("{http://www.w3.org/2005/Atom}link").get("href"), BASE + "/alpha/feed.xml")
        self.assertEqual(len(ch.findall("item")), 2)

    def test_one_series_is_told_in_order_and_a_mix_is_not(self):
        self.assertEqual(self.channel("alpha/feed.xml").find(ITUNES + "type").text, "serial")
        self.assertEqual(self.channel("shared/feed.xml").find(ITUNES + "type").text, "episodic")
        self.assertEqual(self.channel("feed.xml").find(ITUNES + "type").text, "episodic")

    def dates(self, path):
        return {i.find("guid").text: email.utils.parsedate_to_datetime(i.find("pubDate").text)
                for i in self.channel(path).findall("item")}

    def test_episodes_are_dated_in_order(self):
        d = self.dates("feed.xml")
        self.assertLess(d["a_first-01"], d["a_first-02"])
        self.assertLess(d["a_first-02"], d["b_second-01"])
        self.assertEqual((d["b_second-01"].year, d["b_second-01"].month, d["b_second-01"].day), (2026, 12, 1))

    def test_later_episode_carries_its_own_date(self):
        def later(series):
            series["episodes"][1]["published"] = "2027-01-15"
        self.write_series("a_first", ["One", "Two"], tags=["alpha"], change=later)
        d = self.dates("alpha/feed.xml")
        self.assertEqual((d["a_first-01"].year, d["a_first-01"].month, d["a_first-01"].day), (2026, 11, 20))
        self.assertEqual((d["a_first-02"].year, d["a_first-02"].month, d["a_first-02"].day), (2027, 1, 15))

    def test_bad_date_names_the_file(self):
        for where in ("series", "episode"):
            def bad(series):
                (series if where == "series" else series["episodes"][0])["published"] = "20 Nov 2026"
            self.write_series("a_first", ["One", "Two"], change=bad)
            with self.assertRaises(SystemExit) as c:
                self.load()
            self.assertIn(os.path.join("a_first", "series.json"), str(c.exception))
            self.assertIn("20 Nov 2026", str(c.exception))

    def test_special_characters_survive(self):
        odd = 'Tom & "Jerry" <niño>'

        def everywhere(series):
            series["title"] = "T " + odd
            series["episodes"][0]["summary"] = "S " + odd
        self.write_series("a_first", ["E " + odd, "Two"], tags=["alpha"], change=everywhere)
        item = self.channel("alpha/feed.xml").find("item")
        self.assertEqual(item.find("title").text, "T %s 1: E %s" % (odd, odd))
        self.assertEqual(item.find("description").text, "S " + odd)
        page = publish.page_html(self.load(), BASE)
        safe = "Tom &amp; &quot;Jerry&quot; &lt;niño&gt;"
        for mark in "TES":
            self.assertIn(mark + " " + safe, page)
        self.assertNotIn("<niño>", page)


class Page(Case):
    def test_page_lists_every_episode_newest_series_first(self):
        all_s = self.load()
        page = publish.page_html(all_s, BASE)
        self.assertIn('content="noindex, nofollow"', page)
        keys = [e["key"] for e in publish.listing(all_s)]
        self.assertEqual(sorted(keys, key=page.index), keys)
        self.assertIn("AI", page)
        self.assertNotIn("@", page.split("</style>")[-1])

    def test_each_episode_says_its_series_and_carries_its_tags(self):
        page = publish.page_html(self.load(), BASE)
        self.assertIn('<li id="a_first-02" data-temas="alpha shared" data-segundos="1">', page)
        self.assertIn("Series a_first, episode 2", page)

    def test_page_offers_every_tag_as_a_filter_with_its_count(self):
        page = publish.page_html(self.load(), BASE)
        self.assertIn('<a class="tema" href="#" data-tema="">All <span>4</span></a>', page)
        self.assertIn('<a class="tema" href="#tema=alpha" data-tema="alpha">Alpha <span>2</span></a>', page)
        self.assertIn('<a class="tema" href="#tema=shared" data-tema="shared">Shared <span>3</span></a>', page)
        self.assertLess(page.index('data-tema="alpha"'), page.index('data-tema="beta"'))

    def test_page_names_the_feed_for_everything(self):
        page = publish.page_html(self.load(), BASE)
        self.assertIn('data-base="https://example.test/"', page)
        self.assertIn('<a href="https://example.test/feed.xml">https://example.test/feed.xml</a>', page)

    def test_nothing_in_the_site_itself_is_about_one_topic(self):
        page = publish.page_html(self.load(), BASE)
        feeds = "".join(publish.feed_xml(f, BASE) for f in publish.feeds(self.load()))
        for text in (page, feeds):
            self.assertNotIn("uatemal", text)
            self.assertNotIn("uetzal", text)

    def test_build_is_repeatable(self):
        a = publish.build(self.load(), self.tmp("o1"), BASE)
        b = publish.build(self.load(), self.tmp("o2"), BASE)
        self.assertEqual([f["key"] for f in a], [f["key"] for f in b])
        for x, y in zip(a, b):
            self.assertEqual(read(x["path"]), read(y["path"]), x["key"])
        self.assertEqual(a[-1]["key"], "index.html")
        self.assertTrue(a[0]["key"].endswith(".mp3"))
        self.assertEqual(sorted(f["key"] for f in a if not f["key"].endswith(".mp3")),
                         ["alpha/cover.jpg", "alpha/feed.xml", "beta/cover.jpg", "beta/feed.xml", "cover.jpg",
                          "feed.xml", "index.html", "robots.txt", "shared/cover.jpg", "shared/feed.xml"])

    def test_build_marks_types_and_caching(self):
        files = {f["key"]: f for f in publish.build(self.load(), self.tmp("o"), BASE)}
        self.assertEqual(files["index.html"]["type"], "text/html; charset=utf-8")
        self.assertEqual(files["feed.xml"]["type"], "application/rss+xml; charset=utf-8")
        self.assertEqual(files["alpha/feed.xml"]["type"], "application/rss+xml; charset=utf-8")
        self.assertEqual(files["cover.jpg"]["type"], "image/jpeg")
        self.assertEqual(files["index.html"]["cache"], "public, max-age=300")
        mp3 = next(f for k, f in files.items() if k.endswith(".mp3"))
        self.assertEqual(mp3["type"], "audio/mpeg")
        self.assertEqual(mp3["cache"], "public, max-age=31536000, immutable")
        self.assertIn("Disallow: /", read(files["robots.txt"]["path"], "r"))


class Upload(Case):
    def test_failure_text_leaves_out_what_the_service_said(self):
        from botocore.exceptions import ClientError
        e = ClientError({"Error": {"Code": "InvalidAccessKeyId", "Message": "the key tid_ABCDEF123456 is not known"}},
                        "PutObject")
        text = publish.failure(e)
        self.assertIn("InvalidAccessKeyId", text)
        self.assertIn("PutObject", text)
        self.assertNotIn("tid_ABCDEF123456", text)

    def test_plan_reports_changed_and_left_over(self):
        files = [dict(key="a.mp3", path=self.file(b"one")), dict(key="index.html", path=self.file(b"two"))]
        remote = {"a.mp3": hashlib.md5(b"one").hexdigest(), "index.html": "stale", "old.mp3": "x"}
        up, left = publish.plan(files, remote)
        self.assertEqual([f["key"] for f in up], ["index.html"])
        self.assertEqual(left, ["old.mp3"])

    def test_upload_order_puts_pages_last(self):
        files = publish.build(self.load(), self.tmp("o"), BASE)
        up, _ = publish.plan(files, {})
        kinds = [f["key"].rsplit(".", 1)[-1] for f in up]
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "mp3"),
                        min(i for i, k in enumerate(kinds) if k in ("xml", "html")))

    def test_settings_names_the_missing_value(self):
        env = self.tmp("env")
        with open(env, "w") as f:
            f.write("PODCAST_S3_KEY_ID=a\nPODCAST_S3_SECRET='b'\nPODCAST_S3_ENDPOINT=https://s3.example.test\n")
        with self.assertRaises(SystemExit) as c:
            publish.settings(env)
        self.assertIn("PODCAST_S3_BUCKET", str(c.exception))
        with open(env, "a") as f:
            f.write("PODCAST_S3_BUCKET=shelf\n")
        self.assertEqual(publish.settings(env), dict(PODCAST_S3_KEY_ID="a", PODCAST_S3_SECRET="b",
                                                     PODCAST_S3_ENDPOINT="https://s3.example.test",
                                                     PODCAST_S3_BUCKET="shelf"))


if __name__ == "__main__":
    unittest.main()
