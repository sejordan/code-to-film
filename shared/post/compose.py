"""The edit: join rendered frames into the film and give them a film finish.

    uv run python shared/post/compose.py --film example --frames out/frames --out out/film.mp4
    uv run python shared/post/compose.py --film example --frames out/preview --step 2 --out out/preview/preview.mp4

--frames, --out and --sound are paths inside the film's folder.

Frames from Blender are clean. This adds the small faults of a real camera:
a glow around the brightest areas (warm by default; --glow neutral keeps it
the colour of what glows), slightly darker corners and fine grain that
changes every frame. It fades up from black, lays the title over
its shot, fades out at the end, and adds the soundtrack. Then H.264.

Shots are read from the film's timeline.py, in order. A shot with frames missing
stops the edit, so a half-rendered film is never delivered by mistake.
--step N is for previews rendered on every Nth frame: each frame is held N
times, so the timing stays true.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import films  # noqa: E402

ROOT = films.ROOT

FONTS = [("/System/Library/Fonts/Supplemental/Baskerville.ttc", "Italic"),
         ("/System/Library/Fonts/Supplemental/Georgia Italic.ttf", None),
         ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", None)]


def collect(frames_dir, tmp, step, shots):
    """Link every frame of every shot into one numbered run, in film order."""
    n, missing = 0, []
    for shot in shots:
        for i in range(0, shot["frames"], step):
            src = os.path.join(frames_dir, shot["name"], f"{shot['name']}_{i:04d}.jpg")
            if not os.path.exists(src) or os.path.getsize(src) == 0:
                missing.append(os.path.relpath(src, ROOT))
                continue
            os.symlink(src, os.path.join(tmp, f"f_{n:06d}.jpg"))
            n += 1
    return n, missing


def title_card(path, w, h, text):
    """The title as a clear picture the size of the frame, lettered top left."""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for file, style in FONTS:
        if not os.path.exists(file):
            continue
        for index in range(8 if file.endswith(".ttc") else 1):
            try:
                f = ImageFont.truetype(file, round(h * 0.078), index=index)
            except Exception:
                break
            if style is None or f.getname()[1] == style:
                font = f
                break
        if font:
            break
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((round(w * 0.065), round(h * 0.085)), text, font=font or ImageFont.load_default(),
              fill=(250, 236, 214, 235))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    films.add_argument(ap)
    ap.add_argument("--frames", default="out/frames", help="folder holding one folder of frames per shot")
    ap.add_argument("--out", default="out/film.mp4")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--fade", type=float, default=0.8, help="seconds to fade up from black")
    ap.add_argument("--fade-out", type=float, default=1.4, help="seconds to fade to black at the end")
    ap.add_argument("--tail", type=float, default=2.5, help="seconds of black after the last frame")
    ap.add_argument("--grain", type=int, default=3,
                    help="grain strength; it is the main cost in file size (6 made a 3 s shot 168 MB)")
    ap.add_argument("--step", type=int, default=1, help="the frames were rendered every Nth frame")
    ap.add_argument("--sound", default="out/sound.wav", help="soundtrack, or 'none'")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--shots", default="all", help="join only these shots, for example S01,S02,S03")
    ap.add_argument("--clean", action="store_true", help="no film finish, just join the frames")
    ap.add_argument("--glow", default="warm", choices=["warm", "neutral"],
                    help="the glow's tint: warm, or neutral for a film whose brightest things are cold (white eyes, a screen)")
    a = ap.parse_args()

    film = films.load(a.film, a.variant)
    frames_dir = film.path(a.frames)
    out = film.path(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="compose_")
    try:
        names = [s["name"] for s in film.SHOTS]
        if a.shots != "all":
            names = a.shots.split(",")
        shots = [film.shot(x) for x in names]
        whole = names == [s["name"] for s in film.SHOTS]
        prefix = names == [s["name"] for s in film.SHOTS][:len(names)]
        n, missing = collect(frames_dir, tmp, a.step, shots)
        if missing:
            print(f"{len(missing)} frames missing, first: {missing[0]}")
            return 1
        size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=width,height", "-of", "csv=p=0", os.path.join(tmp, "f_000000.jpg")],
                              capture_output=True, text=True, check=True).stdout.strip().split(",")
        w, h = int(size[0]), int(size[1])
        glow = min(w, h) / 60.0                            # the glow's reach scales with the picture
        secs = n * a.step / film.FPS
        fps_in = f"{film.FPS}/{a.step}"
        inputs = ["-framerate", fps_in, "-i", os.path.join(tmp, "f_%06d.jpg")]
        finish = "scale=out_range=tv:out_color_matrix=bt709,format=yuv420p"
        if a.clean:
            graph = f"[0:v]fps={film.FPS},{finish}[v]"
        else:
            graph = (
                f"[0:v]fps={film.FPS},"
                # glow: the highlights, blurred, tinted (warm by default) and screened back on at 34%
                "format=gbrp,split=2[base][hl];"
                "[hl]lutrgb=r='clip((val-185)*255/70,0,255)':g='clip((val-185)*255/70,0,255)':"
                f"b='clip((val-185)*255/70,0,255)',gblur=sigma={glow:.1f},"
                + ("colorchannelmixer=rr=1.0:gg=0.55:bb=0.30[glow];" if a.glow == "warm" else "null[glow];") +
                "[base][glow]blend=all_mode=screen:all_opacity=0.34,"
                # a gentle S-curve and darker corners
                "curves=all='0/0 0.25/0.245 0.75/0.77 1/1',"
                "vignette=angle=PI/9[graded];"
            )
            last = "[graded]"
            if not a.no_title and film.TITLE_SHOT in names:
                card = os.path.join(tmp, "title.png")
                title_card(card, w, h, film.TITLE)
                inputs += ["-loop", "1", "-framerate", str(film.FPS), "-i", card]
                shot = film.shot(film.TITLE_SHOT)
                start = sum(x["seconds"] for x in shots[:names.index(film.TITLE_SHOT)])
                t0 = start + 2.9
                t1 = start + shot["seconds"] - 0.9
                graph += (f"[1:v]format=rgba,fade=t=in:st={t0:.3f}:d=0.9:alpha=1,"
                          f"fade=t=out:st={t1:.3f}:d=0.7:alpha=1[title];"
                          "[graded][title]overlay=shortest=1:format=auto[titled];")
                last = "[titled]"
            # grain that changes every frame, the fades, and black to finish on
            graph += f"{last}noise=alls={a.grain}:allf=t,fade=t=in:st=0:d={a.fade},"
            if whole:
                graph += (f"fade=t=out:st={secs - a.fade_out:.3f}:d={a.fade_out},"
                          f"tpad=stop_mode=add:stop_duration={a.tail}:color=black,")
            # frames are full-range JPEGs; video players expect limited range
            graph += f"{finish}[v]"
        sound = None if a.sound == "none" else film.path(a.sound)
        if sound and not os.path.exists(sound):
            print(f"no soundtrack at {a.sound}: the film will be silent")
            sound = None
        if sound and not prefix:
            print("these shots do not start at the beginning of the film, so the sound would not line up: silent")
            sound = None
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs]
        if sound:
            cmd += ["-i", sound]
        cmd += ["-filter_complex", graph, "-map", "[v]"]
        if sound:
            cmd += ["-map", f"{inputs.count('-i')}:a", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-c:v", "libx264", "-crf", str(a.crf), "-preset", "slow", "-r", str(film.FPS),
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-color_range", "tv", "-t", f"{secs + (a.tail if whole and not a.clean else 0):.3f}",
                "-movflags", "+faststart", out]
        subprocess.run(cmd, check=True)
        print(f"{os.path.relpath(out, ROOT)}: {n} frames, {secs:.1f} s of picture, "
              f"{'with sound' if sound else 'silent'}, {os.path.getsize(out) / 1e6:.1f} MB")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
