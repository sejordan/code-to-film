"""Render the whole film on rented GPUs.

    uv run python shared/cloud/film.py --film example --pods 10 --gpu "NVIDIA RTX A5000" \
        --cloud SECURE --max-price 0.30 --max-minutes 120
    uv run python shared/cloud/film.py ... --dry-run      # the plan and the price, nothing rented
    uv run python shared/cloud/film.py --film example --on vast --pods 8 --gpu "RTX 4090" \
        --max-price 0.60 --max-minutes 120
    uv run python shared/cloud/film.py ... --step 2 --scale 25 --samples 40 --out out/preview    # a preview

Every frame of every shot that is not already in the film's out/frames is dealt out to
the pods like cards, one at a time, so each pod gets an even share of the
cheap shots and the dear ones. A pod works through the shots in film order.
Finished frames come home every half minute, and each pod is removed the
moment it finishes, or at --max-minutes whatever happens.

Running the same command again renders only what is still missing, so a pod
that died costs only its unfinished share.

The single-shot driver (render.py, next to this file) does the SSH and the
setup, and hosts.py the renting (RunPod, or Vast.ai with --on vast); this
file only decides who renders what.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import films           # noqa: E402
import hosts           # noqa: E402  (shared/cloud/hosts.py)
import render as one   # noqa: E402  (shared/cloud/render.py)

ROOT, CLOUD, say = one.ROOT, one.CLOUD, one.say


def pull(addr, film, dest):
    """Copy home any finished frames this Mac does not have yet."""
    host, port = addr
    there = f"{one.remote_out(film)}/frames"
    r = one.ssh(addr, f"cd {there} 2>/dev/null && ls -t */*.jpg 2>/dev/null; "
                      "test -f /root/job/DONE && echo DONE")
    words = r.stdout.split()
    names = [n for n in words if n.endswith(".jpg")]
    if "DONE" not in words:
        names = names[1:]                    # the newest may still be being written: take it next time
    want = [n for n in names if not os.path.exists(os.path.join(dest, n))]
    if not want:
        return 0
    for d in {os.path.dirname(n) for n in want}:
        os.makedirs(os.path.join(dest, d), exist_ok=True)
    tar = subprocess.Popen(["ssh", *one.SSH_OPTS, "-p", str(port), f"root@{host}",
                            f"cd {there} && tar cf - " + " ".join(want)], stdout=subprocess.PIPE)
    subprocess.run(["tar", "xf", "-", "-C", dest], stdin=tar.stdout, check=False)
    tar.wait()
    return sum(1 for n in want if os.path.exists(os.path.join(dest, n)))


def run_block(share, a):
    """The pod's own script: one Blender run per shot, then a DONE marker."""
    by_shot = {}
    for shot, i in share:
        by_shot.setdefault(shot, []).append(i)
    specs = " ".join(f"{shot}:{','.join(str(i) for i in frames)}" for shot, frames in by_shot.items())
    samples = f"--samples {a.samples}" if a.samples else ""
    return ("cd /root/job\n"
            f"for spec in {specs}; do\n"
            "  shot=${spec%%:*}; frames=${spec#*:}\n"
            "  /opt/blender/blender -b --factory-startup -P shared/blender/render.py -- "
            f"--film {one.film_args(a.film)} --shot $shot --frames $frames --scale {a.scale} {samples} >> render.log 2>&1\n"
            "done\n"
            "touch DONE")


def run_pod(k, host, share, a, pub, packfile, dest, state, results):
    name = f"toyfilm-film-{k}"
    info = dict(name=name, k=k, share=len(share), frames=0, id=None, note="")
    results[k] = info
    pid = one.rent(host, name, a, pub, info, state, extra=f"  {len(share)} frames to do")
    if pid is None:
        return
    t_create = time.time()
    deadline = t_create + a.max_minutes * 60
    try:
        addr = one.wait_for_ssh(host, pid, t_create, deadline, info)
        if addr is None:
            return
        gpu = one.gpu_name(addr)
        say(name, f"up after {time.time() - t_create:.0f}s on {gpu or 'unknown GPU'}, sending project")
        r = one.scp_to(addr, packfile, "/root/pack.tgz")
        if r.returncode != 0:
            info["note"] = "upload failed: " + r.stderr.strip()[:200]
            return
        r = one.ssh(addr, one.setup_script(host, a, deadline, run_block(share, a)), timeout=900)
        if "STARTED" not in r.stdout:
            info["note"] = "setup failed at: " + " | ".join(
                (r.stdout.strip() + " " + r.stderr.strip()).split("\n")[-3:])[-400:]
            return
        t_render = time.time()
        info["setup_seconds"] = round(t_render - t_create)
        say(name, f"Blender installed, rendering (setup took {info['setup_seconds']}s in all)"
            + ("" if "SAFETY ok" in r.stdout else "  NOTE: it cannot remove itself; this script will"))
        last_say = 0.0
        while time.time() < deadline:
            time.sleep(30)
            try:
                info["frames"] += pull(addr, a.film, dest)
                st = one.ssh(addr, "test -f /root/job/DONE && echo DONE; "
                                   "grep -E 'Error|Traceback' /root/job/render.log | tail -2")
            except subprocess.TimeoutExpired:
                continue
            if time.time() - last_say > 300:
                say(name, f"{info['frames']} of {len(share)} frames home")
                last_say = time.time()
            if "DONE" in st.stdout:
                info["frames"] += pull(addr, a.film, dest)
                log = one.ssh(addr, "cat /root/job/render.log", timeout=300).stdout
                with open(os.path.join(dest, f"pod_{k}_{pid}.log"), "w") as f:
                    f.write(log)
                times = sorted(float(ln.split()[-1].rstrip("s")) for ln in log.splitlines() if ln.startswith("FRAME"))
                info["device"] = next((ln.split("device ")[1].split("  ")[0] for ln in log.splitlines()
                                       if ln.startswith("BUILT")), "")
                info["sec_per_frame"] = round(times[len(times) // 2], 1) if times else None
                info["render_seconds"] = round(time.time() - t_render)
                if len(times) < len(share):
                    info["note"] = f"finished with {len(times)} of {len(share)} frames: " + log.strip()[-300:]
                break
        else:
            info["note"] = "hit the time limit"
            info["frames"] += pull(addr, a.film, dest)
    except Exception as e:                          # whatever went wrong, the pod still gets removed below
        info["note"] = f"{type(e).__name__}: {e}"
    finally:
        info["seconds"] = round(time.time() - t_create)
        info["removed"] = host.remove(pid)
        say(name, f"removed after {info['seconds']}s, {info['frames']} frames home" if info["removed"]
            else "COULD NOT CONFIRM REMOVAL of " + pid, info["note"])


def main():
    ap = argparse.ArgumentParser()
    films.add_argument(ap)
    ap.add_argument("--shots", default="all")
    ap.add_argument("--pods", type=int, default=8)
    ap.add_argument("--gpu", default="NVIDIA RTX A5000")
    hosts.add_arguments(ap)
    ap.add_argument("--max-price", type=float, required=True,
                    help="refuse to rent if a machine costs more than this per hour")
    ap.add_argument("--max-minutes", type=int, default=120, help="every pod is removed by this time")
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--step", type=int, default=1, help="render every Nth frame (2 for a 12 fps preview)")
    ap.add_argument("--out", default="out/frames", help="inside the film's folder")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and the price; rent nothing")
    a = ap.parse_args()

    a.film = film = films.load(a.film, a.variant)
    names = [s["name"] for s in film.SHOTS] if a.shots == "all" else a.shots.split(",")
    dest = film.path(a.out)
    todo = [(n, i) for n in names for i in range(0, film.shot(n)["frames"], a.step)
            if not os.path.exists(os.path.join(dest, n, f"{n}_{i:04d}.jpg"))]
    total = sum(len(range(0, film.shot(n)["frames"], a.step)) for n in names)
    if not todo:
        print(f"all {total} frames are already home")
        return 0
    a.pods = min(a.pods, len(todo))
    shares = [todo[k::a.pods] for k in range(a.pods)]

    host = hosts.HOSTS[a.on](a)
    quote = host.quote(a.pods)
    prices = quote["prices"]
    each = one.price_range(prices)
    print(f"{film.name}{' ' + film.variant if film.variant else ''}: {len(todo)} of {total} frames to render at "
          f"{a.scale}% on {a.pods} x {host.describe()} at {each}/hr each (${sum(prices):.2f}/hr in all), "
          f"{quote['stock']}; every pod removed within {a.max_minutes} min, "
          f"so at most ${sum(prices) * a.max_minutes / 60:.2f}" + (f"; {quote['note']}" if quote["note"] else ""))
    if len(prices) < a.pods or max(prices) > a.max_price:
        print(f"only {len(prices)} of {a.pods} machines are to be had within the ${a.max_price}/hr limit: nothing rented")
        return 1
    if a.dry_run:
        return 0

    pub = one.ensure_key()
    packfile = one.pack(film)
    os.makedirs(dest, exist_ok=True)
    state = dict(file=os.path.join(CLOUD, time.strftime(f"run-%Y%m%d-%H%M%S-{os.getpid()}.json")), film=film.name, shot="film",
                 pods=[])
    say(f"pack {os.path.getsize(packfile) / 1e6:.0f} MB")
    results, threads = {}, []
    t0 = time.time()
    try:
        for k in range(a.pods):
            t = threading.Thread(target=run_pod, args=(k, host, shares[k], a, pub, packfile, dest, state, results),
                                 daemon=True)
            t.start()
            threads.append(t)
            time.sleep(2)                           # not all at the API in the same instant
        for t in threads:
            t.join()
    finally:
        one.still_there(state)                      # nothing this run made is left running
        if os.path.exists(packfile):
            os.remove(packfile)
    left = [(n, i) for n, i in todo if not os.path.exists(os.path.join(dest, n, f"{n}_{i:04d}.jpg"))]
    cost = sum((r.get("cost") or 0) * r.get("seconds", 0) / 3600 for r in results.values())
    summary = dict(film=film.name, frames_asked=len(todo), frames_missing=len(left), pods=list(results.values()),
                   wall_seconds=round(time.time() - t0), est_cost_usd=round(cost, 3), gpu=a.gpu, on=host.describe())
    json.dump(summary, open(os.path.join(dest, time.strftime("film_summary-%Y%m%d-%H%M%S.json")), "w"), indent=1)
    say(f"{len(todo) - len(left)} of {len(todo)} frames home in {summary['wall_seconds']}s, about ${cost:.2f}")
    for r in results.values():
        say(f"  {r['name']}: {r.get('frames')} of {r.get('share')} frames, {r.get('sec_per_frame')} s/frame, "
            f"{r.get('seconds', 0)}s, {r.get('device', '')} {r['note']}")
    if left:
        say(f"{len(left)} frames still missing; run the same command again to render just those")
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
