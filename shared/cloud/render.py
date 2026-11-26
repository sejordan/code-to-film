"""Render a shot on rented GPUs and bring the frames home.

    uv run python shared/cloud/render.py --film example --shot S01 --pods 2
    uv run python shared/cloud/render.py --film example --shot S10 --pods 1 --on vast --max-price 0.60

The machines come from RunPod or, with --on vast, from Vast.ai (hosts.py,
next to this file, knows how each is rented and removed).

A pod is one rented computer with one graphics card. Each pod gets a copy of
what the film needs (shared/, the film's own folder without its out/, and
the scans its timeline.py lists), installs Blender, and renders every
Nth frame of the shot, so a pod that dies loses only its own share. This Mac
collects finished frames every 30 seconds and removes each pod the moment its
share is home. Whatever happens, every pod also removes itself after
--max-minutes, and this script removes its pods when it exits.

No API key is ever printed. Pods this run created are written to .cloud/run-*.json, so

    uv run python shared/cloud/render.py --cleanup

can remove anything a crashed run left behind. It only touches those pods.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import films  # noqa: E402
import hosts  # noqa: E402  (shared/cloud/hosts.py)

ROOT = films.ROOT

CLOUD = os.path.join(ROOT, ".cloud")
KEY = os.path.join(CLOUD, "id_ed25519")
BLENDER_URL = "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"
BLENDER_SHA = "dacb8264197a3f7fda6001285a5a506fc97ee943881261c2c3bd63c882f22e05"
SSH_OPTS = ["-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", "-o", "IdentitiesOnly=yes",
            "-o", "ServerAliveInterval=15"]
LOCK = threading.Lock()


def say(*a):
    with LOCK:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_frames(spec, n):
    """all | 0,24,48 | 10-20, as film frame numbers within the shot."""
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += range(int(lo), int(hi) + 1)
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < n))


# ---------------------------------------------------------------- local helpers

def ensure_key():
    """A key pair used only for this project's render pods. On RunPod the
    public half is handed to each pod when it is created; on Vast.ai it is
    kept on the account, which gives it to each new machine."""
    os.makedirs(CLOUD, exist_ok=True)
    if not os.path.exists(KEY):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "toy-film-render", "-f", KEY],
                       check=True)
    with open(KEY + ".pub") as f:
        return f.read().strip()


def pack(film):
    """One film as the pods need it: the shared tools, the film's own folder
    (not its out/) and the scans its timeline.py lists. Other films, and
    scans this one does not use, stay home."""
    os.makedirs(CLOUD, exist_ok=True)
    path = os.path.join(CLOUD, f"pack-{os.getpid()}.tgz")
    items = ["shared"]
    items += [os.path.join("films", film.name, n) for n in sorted(os.listdir(film.dir))
              if n not in ("out", "__pycache__", ".DS_Store")]
    for asset in sorted({**film.TEXTURES, **film.SKIES}):
        if not os.path.isdir(os.path.join(ROOT, "assets", "polyhaven", asset)):
            raise SystemExit(f"{asset} is not downloaded: uv run python shared/fetch_assets.py --film {film.name}")
        items.append(os.path.join("assets", "polyhaven", asset))
    subprocess.run(["tar", "--exclude", "__pycache__", "--exclude", ".DS_Store", "--no-xattrs",
                    "-czf", path, "-C", ROOT, *items], check=True,
                   env=dict(os.environ, COPYFILE_DISABLE="1"))
    return path


def remote_out(film):
    """The film's out folder on a pod."""
    return "/root/job/" + os.path.relpath(film.out, ROOT)


def film_args(film):
    """How a pod's Blender is told which film, and which variant of it."""
    return film.name + (f" --variant {film.variant}" if film.variant else "")


def ssh(addr, command, timeout=120):
    host, port = addr
    return subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", command],
                          capture_output=True, text=True, timeout=timeout)


def scp_to(addr, src, dst, timeout=600):
    host, port = addr
    return subprocess.run(["scp", *SSH_OPTS, "-P", str(port), src, f"root@{host}:{dst}"],
                          capture_output=True, text=True, timeout=timeout)


def pull_frames(addr, film, shot, dest):
    """Copy home any finished frames this Mac does not have yet."""
    host, port = addr
    there = f"{remote_out(film)}/frames/{shot}"
    r = ssh(addr, f"cd {there} 2>/dev/null && ls *.jpg render_log.jsonl 2>/dev/null")
    names = [n for n in r.stdout.split() if n.endswith(".jpg")]
    want = [n for n in names if not os.path.exists(os.path.join(dest, n))]
    if not want:
        return 0
    # a frame is written whole before the next begins, but the newest may still be open: take it next time
    r2 = ssh(addr, f"cd {there} && ls -t *.jpg | head -1; test -f /root/job/DONE && echo DONE")
    newest = next((w for w in r2.stdout.split() if w.endswith(".jpg")), "")
    done = "DONE" in r2.stdout
    if not done:
        want = [n for n in want if n != newest]
    if not want:
        return 0
    tar = subprocess.Popen(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
                            f"cd {there} && tar cf - " + " ".join(want)],
                           stdout=subprocess.PIPE)
    subprocess.run(["tar", "xf", "-", "-C", dest], stdin=tar.stdout, check=False)
    tar.wait()
    return sum(1 for n in want if os.path.exists(os.path.join(dest, n)))


REMOTE_SETUP = r"""
set -e
mkdir -p /root/job && cd /root/job
touch /root/.no_auto_tmux
# safety net: this pod removes itself after {max_minutes} minutes, whatever happens
{self_remove}
echo STEP unpack; tar xzf /root/pack.tgz
export DEBIAN_FRONTEND=noninteractive
echo STEP packages          # a host's package mirror can be briefly unreachable: three tries
ok=0
for try in 1 2 3; do
  if (apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libxkbcommon0 libsm6 libice6 libx11-6 xz-utils \
      curl ca-certificates \
      ) > /root/job/apt.log 2>&1; then ok=1; break; fi
  sleep 8
done
if [ $ok = 0 ]; then echo "packages failed: $(tail -c 300 /root/job/apt.log)"; exit 1; fi
echo STEP download; curl -sSL --retry 3 -o /root/blender.tar.xz {blender_url}
echo STEP checksum; echo "{blender_sha}  /root/blender.tar.xz" | sha256sum -c - > /root/job/sha.log 2>&1
echo STEP install; mkdir -p /opt/blender && tar xJf /root/blender.tar.xz -C /opt/blender --strip-components 1
echo STEP start
cat > /root/job/run.sh <<'EOS'
{run_block}
EOS
nohup sh /root/job/run.sh > /dev/null 2>&1 &
echo STARTED
"""

RUN_NORMAL = """cd /root/job
/opt/blender/blender -b --factory-startup -P shared/blender/render.py -- \
    --film {film} --shot {shot} --frames {frames} --scale {scale} {samples} > render.log 2>&1
touch DONE"""

# Benchmark: the same frames three ways, with the GPU's activity sampled four
# times a second, so tracing, denoising and host overhead can be told apart.
RUN_BENCH = """cd /root/job
for mode in {passes}; do
  if [ $mode = gpu ]; then out=out/frames; else out=out/bench_$mode; fi
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 250 > util_$mode.log 2>/dev/null &
  SP=$!
  /opt/blender/blender -b --factory-startup -P shared/blender/render.py -- \
      --film {film} --shot {shot} --frames {frames} --scale {scale} {samples} --denoise $mode --out $out/{shot} \
      > render_$mode.log 2>&1
  kill $SP
done
cat render_*.log > render.log
touch DONE"""

BENCH_STATS = """cd /root/job
for m in {passes}; do
  echo "##$m"; grep -E '^FRAME' render_$m.log
  echo "#util $(awk '$1>=50{{b++}} END{{print b+0, NR}}' util_$m.log)"
done
echo "##host $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) | $(nproc) threads"
"""


def bench_stats(text):
    """Median seconds per frame for each pass, and how long the GPU was busy."""
    out, mode = {}, None
    for line in text.splitlines():
        if line.startswith("##host"):
            out["host"] = line[7:].strip()
        elif line.startswith("##"):
            mode = line[2:].strip()
            out[mode] = dict(times=[])
        elif line.startswith("FRAME") and mode:
            out[mode]["times"].append(float(line.split()[-1].rstrip("s")))
        elif line.startswith("#util") and mode:
            busy, total = (int(x) for x in line.split()[1:3])
            out[mode]["gpu_busy_s"] = busy * 0.25
    for mode in ("gpu", "off", "cpu"):
        m = out.get(mode) or {}
        t = sorted(m.get("times", []))
        if t:
            m["median_s"] = t[len(t) // 2]
            m["gpu_busy_per_frame_s"] = round(m.get("gpu_busy_s", 0) / len(t), 2)
    return out



# ---------------------------------------------------------------- renting a pod

def rent(host, name, a, pub, info, state, extra=""):
    """Rent one machine and write it down in this run's record, so that a
    crashed run can still be cleaned up. Returns its id, or None with the
    reason in info["note"]."""
    try:
        m = host.create(name, pub)
    except RuntimeError as e:
        info["note"] = f"create failed: {e}"
        say(name, info["note"])
        return None
    mid = info["id"] = m["id"]
    info["cost"] = m.get("cost")
    with LOCK:
        state["pods"].append(dict(id=mid, name=name, on=host.name))
        json.dump(state, open(state["file"], "w"), indent=1)
    say(name, f"created {mid}  ${m.get('cost')}/hr{extra}" + (f"  ({m['where']})" if m.get("where") else ""))
    if (m.get("cost") or 0) > a.max_price:
        info["note"] = f"priced at ${m.get('cost')}/hr, above the limit: removed at once"
        info["seconds"], info["removed"] = 0, host.remove(mid)
        say(name, info["note"])
        return None
    return mid


def wait_for_ssh(host, mid, t_create, deadline, info):
    """The machine's SSH address once it answers, or None with the reason in
    info["note"]."""
    addr = None
    while time.time() < deadline and addr is None:
        time.sleep(8)
        addr = host.address(mid)
        if addr is None and time.time() - t_create > host.patience:
            info["note"] = f"no SSH address after {host.patience // 60} minutes (bad machine), removed"
            return None
    if addr is None:
        info["note"] = "ran out of time before start"
        return None
    for _ in range(30):                          # the address appears before sshd is up
        try:
            if "ok" in ssh(addr, "echo ok", timeout=30).stdout.split():
                return addr
        except subprocess.TimeoutExpired:
            pass
        time.sleep(6)
    info["note"] = "SSH never answered, removed"
    return None


def price_range(prices):
    """"$0.89" when every machine costs the same, "$0.42 to $0.53" when they differ."""
    if not prices:
        return "$?"
    lo, hi = min(prices), max(prices)
    return f"${lo:.2f}" if hi - lo < 0.005 else f"${lo:.2f} to ${hi:.2f}"


def setup_script(host, a, deadline, run_block):
    return REMOTE_SETUP.format(
        max_minutes=a.max_minutes, self_remove=host.self_remove(int(max(60, deadline - time.time()))),
        blender_url=BLENDER_URL, blender_sha=BLENDER_SHA, run_block=run_block)


def gpu_name(addr):
    out = ssh(addr, "nvidia-smi --query-gpu=name --format=csv,noheader").stdout.strip().splitlines()
    return out[-1].strip() if out else "unknown GPU"


def still_there(state):
    """Belt and braces at the end of a run: remove anything it rented that is
    somehow still present."""
    for p in state["pods"]:
        host = a_host(p)
        if host.alive(p["id"]):
            say("still present, removing", p["id"], "->", "gone" if host.remove(p["id"]) else "FAILED")


class _Recorded:
    """What a run record needs a host for: asking after a machine and removing it."""
    gpu, cloud, create_tries, max_price, regions, min_reliability = "", "SECURE", 1, 0.0, "any", 0.0


def a_host(p):
    return hosts.HOSTS[p.get("on", "runpod")](_Recorded)


# ---------------------------------------------------------------- one pod's life

def run_pod(k, host, a, pub, packfile, dest, state, results):
    name = f"toyfilm-{a.shot}-{k}"
    info = dict(name=name, k=k, frames=0, id=None, note="")
    results[k] = info
    pid = rent(host, name, a, pub, info, state)
    if pid is None:
        return
    t_create = time.time()
    deadline = t_create + a.max_minutes * 60
    try:
        addr = wait_for_ssh(host, pid, t_create, deadline, info)
        if addr is None:
            return
        gpu = gpu_name(addr)
        say(name, f"up after {time.time() - t_create:.0f}s on {gpu or 'unknown GPU'}, sending project")
        r = scp_to(addr, packfile, "/root/pack.tgz")
        if r.returncode != 0:
            info["note"] = "upload failed: " + r.stderr.strip()[:200]
            return
        passes = " ".join(a.bench_passes.split(","))
        run_block = (RUN_BENCH if a.bench else RUN_NORMAL).format(
            passes=passes, film=film_args(a.film), shot=a.shot, frames=",".join(str(i) for i in a.todo[k::a.pods]),
            scale=a.scale, samples=f"--samples {a.samples}" if a.samples else "")
        r = ssh(addr, setup_script(host, a, deadline, run_block), timeout=900)
        if "STARTED" not in r.stdout:
            info["note"] = "setup failed at: " + " | ".join((r.stdout.strip() + " " + r.stderr.strip()).split("\n")[-3:])[-400:]
            return
        say(name, f"Blender installed, rendering (setup took {time.time() - t_create:.0f}s in all)"
            + ("" if "SAFETY ok" in r.stdout else "  NOTE: it cannot remove itself; this script will"))
        t_render = time.time()
        while time.time() < deadline:
            time.sleep(30)
            try:
                got = pull_frames(addr, a.film, a.shot, dest)
                info["frames"] += got
                st = ssh(addr, "test -f /root/job/DONE && echo DONE; grep -c '^FRAME' /root/job/render.log; "
                               "grep -E '^BUILT|Error|Traceback' /root/job/render.log | tail -3")
            except subprocess.TimeoutExpired:
                continue
            if got:
                say(name, f"+{got} frames ({info['frames']} home)")
            if "DONE" in st.stdout:
                info["frames"] += pull_frames(addr, a.film, a.shot, dest)
                log = ssh(addr, "cat /root/job/render.log").stdout
                with open(os.path.join(dest, f"pod_{k}_{pid}.log"), "w") as f:
                    f.write(log)
                times = [float(l.split()[-1].rstrip("s")) for l in log.splitlines() if l.startswith("FRAME")]
                info["device"] = next((l for l in log.splitlines() if l.startswith("BUILT")), "")
                info["sec_per_frame"] = round(sorted(times)[len(times) // 2], 1) if times else None
                info["render_seconds"] = round(time.time() - t_render)
                info["setup_seconds"] = round(t_render - t_create)
                if a.bench:
                    info["bench"] = bench_stats(ssh(addr, BENCH_STATS.format(passes=passes)).stdout)
                    info["gpu"], info["price"] = a.gpu, info.get("cost")
                if not times:
                    info["note"] = "rendered nothing: " + log.strip()[-300:]
                break
        else:
            info["note"] = "hit the time limit"
    except Exception as e:                        # whatever went wrong, the pod still gets removed below
        info["note"] = f"{type(e).__name__}: {e}"
    finally:
        info["seconds"] = round(time.time() - t_create)
        gone = host.remove(pid)
        info["removed"] = gone
        say(name, f"removed after {info['seconds']}s" if gone else "COULD NOT CONFIRM REMOVAL of " + pid,
            info["note"])


# ---------------------------------------------------------------- main

def cleanup():
    n = 0
    for path in sorted(glob.glob(os.path.join(CLOUD, "run-*.json"))):
        for p in json.load(open(path)).get("pods", []):
            host = a_host(p)
            if host.alive(p["id"]):
                print("removing", p["id"], p["name"], "->", "gone" if host.remove(p["id"]) else "FAILED")
                n += 1
    print(f"{n} pods needed removing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", help="the film's folder name under films/: " + ", ".join(films.names()))
    ap.add_argument("--variant", default=None, help="for a film with variants (say en, es): which one")
    ap.add_argument("--shot")
    ap.add_argument("--pods", type=int, default=2)
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    hosts.add_arguments(ap)
    ap.add_argument("--max-price", type=float, default=0.40,
                    help="refuse to rent if a machine costs more than this per hour")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--max-minutes", type=int, default=45, help="every pod is removed by this time")
    ap.add_argument("--out", default="out/frames", help="inside the film's folder")
    ap.add_argument("--bench", default=None, metavar="LABEL",
                    help="benchmark this GPU: render the frames three ways into the film's out/bench/LABEL")
    ap.add_argument("--bench-passes", default="gpu,off,cpu",
                    help="which denoiser settings a benchmark renders; gpu must be one of them")
    ap.add_argument("--cleanup", action="store_true")
    a = ap.parse_args()
    if a.cleanup:
        return cleanup()
    if not a.film or not a.shot:
        ap.error("--film and --shot are needed")
    a.film = film = films.load(a.film, a.variant)
    shot = film.shot(a.shot)
    if a.bench:
        a.out = os.path.join("out", "bench", a.bench)
    host = hosts.HOSTS[a.on](a)
    dest = film.path(a.out, a.shot)
    os.makedirs(dest, exist_ok=True)
    # only frames that are not home yet are rendered, so a run that lost a pod
    # (or was stopped) is finished by simply running the same command again
    a.todo = [i for i in parse_frames(a.frames, shot["frames"])
              if not os.path.exists(os.path.join(dest, f"{a.shot}_{i:04d}.jpg"))]
    if not a.todo:
        print(f"{a.shot}: every requested frame is already home")
        return 0
    a.pods = min(a.pods, len(a.todo))
    quote = host.quote(a.pods)
    if len(quote["prices"]) < a.pods or max(quote["prices"]) > a.max_price:
        print(f"{host.describe()}: {len(quote['prices'])} of {a.pods} machines to be had within "
              f"${a.max_price}/hr ({quote['stock']}): nothing rented")
        return 1
    pub = ensure_key()
    packfile = pack(film)
    state = dict(file=os.path.join(CLOUD, time.strftime(f"run-%Y%m%d-%H%M%S-{os.getpid()}.json")), film=film.name,
                 shot=a.shot, pods=[])
    say(f"{a.shot}: {len(a.todo)} of {shot['frames']} frames to render at {a.scale}% on {a.pods} x {host.describe()} "
        f"at {price_range(quote['prices'])}/hr each, {quote['stock']}, "
        f"pack {os.path.getsize(packfile) / 1e6:.0f} MB, limit {a.max_minutes} min"
        + (f"; {quote['note']}" if quote["note"] else ""))
    results, threads = {}, []
    t0 = time.time()
    try:
        for k in range(a.pods):
            t = threading.Thread(target=run_pod, args=(k, host, a, pub, packfile, dest, state, results), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        still_there(state)                         # nothing this run made is left running
    if os.path.exists(packfile):
        os.remove(packfile)
    have = len(glob.glob(os.path.join(dest, f"{a.shot}_*.jpg")))
    cost = sum((r.get("cost") or 0) * r.get("seconds", 0) / 3600 for r in results.values())
    summary = dict(film=film.name, shot=a.shot, frames_expected=shot["frames"], frames_home=have, pods=list(results.values()),
                   wall_seconds=round(time.time() - t0), est_cost_usd=round(cost, 3))
    json.dump(summary, open(os.path.join(dest, time.strftime("cloud_summary-%H%M%S.json")), "w"), indent=1)
    say(f"{have} of {shot['frames']} frames home in {summary['wall_seconds']}s, about ${cost:.2f}")
    for r in results.values():
        say(f"  {r['name']}: {r.get('frames')} frames, {r.get('sec_per_frame')} s/frame, {r.get('device', '')} {r['note']}")
    for r in results.values():
        b = r.get("bench")
        if b:
            say("  BENCH " + json.dumps({k: (v if k == "host" else {x: v.get(x) for x in ("median_s", "gpu_busy_per_frame_s")})
                                         for k, v in b.items()}))
    asked = set(parse_frames(a.frames, shot["frames"]))
    return 0 if all(os.path.exists(os.path.join(dest, f"{a.shot}_{i:04d}.jpg")) for i in asked) else 1


if __name__ == "__main__":
    sys.exit(main())
