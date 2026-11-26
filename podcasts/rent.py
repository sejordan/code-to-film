"""Rent one large-memory GPU to try the speech model there. A trial.

    uv run python podcasts/rent.py                                    # what it would cost; rents nothing
    uv run python podcasts/rent.py --on vast --gpu "RTX 4090" --max-price 0.75
    uv run python podcasts/rent.py --go --minutes 90                  # rent one, and say how to reach it
    uv run python podcasts/rent.py --remove                           # remove it

Drama 3's official release does not fit a 24 GB card: its 4-bit version
needs about 21 GB for the model alone, plus room for the rest. So this asks
for a 48 GB card.

runpod  (the default) a data-centre card by its RunPod name, such as
        "NVIDIA A40" or "NVIDIA L40S", in --cloud SECURE or COMMUNITY. One
        price per card, and traffic is not billed.
vast    someone's own machine: an RTX 4090 rebuilt with 48 GB (Vast lists
        them under the same name), x86, with a fast line that is cheap to
        download over, since the model is about 24 GB.

Give --minutes about twice what the job should take, and --remove the
machine as soon as it is done: the limit is a safety net, not the plan.
The machine is told to remove itself after --minutes whatever happens, and
is written down in .cloud/ like a render machine, so
`uv run python shared/cloud/render.py --cleanup` removes it too. A machine
that comes up dearer than --max-price is removed at once. RunPod's key is
read from .env and Vast's is held by the `vastai` tool; neither is printed.
Nothing is rented without --go.

What was run on the RunPod machine on 2026-11-19 (an A40; it has no SSH, so
each line went through RunPod's exec), and what it took:

    cd /root && git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
    uv venv .venv -p 3.12 && source .venv/bin/activate
    uv pip install --prerelease=allow -e ".[fishaudio]"          # under a minute
    hf download fishaudio/drama-3-int4                            # 24 GB, under a minute
    sgl-omni serve --model-path fishaudio/drama-3-int4 --port 8000 --host 0.0.0.0 \
        --tts_engine.factory.max_new_tokens 1024

The image took 11 minutes to arrive and the server's first start 14 more
(it builds its kernels; the second start took 80 seconds). Without the
last option the server stops at the end of its start-up on a 48 GB card:
its defaults are sized for a 141 GB one. gpu_bench.py then measured one
paragraph at 3.7 times real time and 16 to 64 at once at 21 to 25 times.

A second rental the same day landed on a busier machine, where the first
start gave up after ten minutes without progress; `export
SGLANG_OMNI_STARTUP_TIMEOUT=1800` before `sgl-omni serve` and a second start
(six minutes) got it going. With max_new_tokens at 1024 one request makes
at most about 47 seconds of speech, so a script is sent a paragraph at a
time (gpu_narrate.py), never whole.

A fourth rental that evening lost its hour: requests through RunPod's proxy
stalled at random (the same request made inside the machine took two
seconds), the narration was at paragraph 18 of 26 when the machine removed
itself, and nothing had been saved. gpu_narrate.py now gives up on a
stalled request after 25 seconds, sends smaller requests, and keeps every
take so a later run carries on. Allow more than 60 minutes if the proxy is
slow, or run the narration from inside the machine.

On 2026-11-20 no 48 GB card was in stock, so an A100 80GB PCIe ($2.09/hr,
secure) ran the same commands unchanged, max_new_tokens 1024 included. The
image took 11 minutes and the server's first start 18; then eight chained
episodes (83 minutes of speech) narrated side by side in under 5 minutes,
with one stalled request in all. 37 minutes, $1.28.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "shared", "cloud"), os.path.join(ROOT, "shared")]

import hosts  # noqa: E402
import render  # noqa: E402

hosts.DISK = 80                 # GB: the model, the server's Python packages, and room to spare
NOTE = os.path.join(render.CLOUD, "podcast-machine.json")


class RunPodServer(hosts.RunPod):
    """As hosts.RunPod, but running the speech server's own image, which has
    no SSH: it is kept alive with `sleep`, driven with RunPod's exec, and its
    port 8000 is reached at https://<pod id>-8000.proxy.runpod.net."""
    image = "hongccc/sglang-omni:dev"

    def create(self, name, pub):
        code, pod = 0, {}
        for attempt in range(self.tries):
            code, pod = hosts.runpod("POST", "/pods", dict(
                name=name, image=self.image, cloud=self.cloud, gpu=dict(id=self.gpu, count=1), disk=hosts.DISK,
                ports=["8000/http"], args=json.dumps(dict(entrypoint=["/bin/bash", "-c"], cmd=["sleep infinity"]))))
            if code < 300 and "id" in pod:
                return dict(id=pod["id"], cost=pod.get("cost"))
            time.sleep(20)
        raise RuntimeError(f"HTTP {code} {pod.get('title', '')} {pod.get('detail', '')}")

    def running(self, mid):
        code, pod = hosts.runpod("GET", f"/pods/{mid}")
        return code == 200 and pod.get("status") == "RUNNING", (pod.get("status") if code == 200 else code)


class Vast48(hosts.Vast):
    """As hosts.Vast, but only cards with 46 GB or more, and a fast line that
    is cheap to download over."""
    MAX_PER_GB = 0.01

    def search(self):
        q = (f"gpu_name={self.gpu.replace(' ', '_')} gpu_ram>=46 num_gpus=1 rentable=true verified=true "
             f"reliability>={self.reliability} inet_up>=100 inet_down>=500 cpu_cores_effective>=8 "
             f"cpu_arch=amd64 disk_space>={hosts.DISK + 10} direct_port_count>=2 dph<={self.max_price}")
        if self.regions.lower() != "any":
            q += f" geolocation in [{self.regions}]"
        found, text = hosts.vast("search", "offers", q, "--storage", str(hosts.DISK), "--limit", "300", "--raw")
        if not isinstance(found, list):
            raise SystemExit("Vast.ai search failed: " + text)
        seen, offers = set(), []
        for o in sorted(found, key=lambda o: o["dph_total"]):
            dear = (o.get("inet_down_cost") or 0) > self.MAX_PER_GB      # little is sent back: only audio
            if o["dph_total"] > self.max_price or dear or o["machine_id"] in seen:
                continue
            seen.add(o["machine_id"])
            offers.append(o)
        self.offers = offers
        return offers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--on", default="runpod", choices=("runpod", "vast"))
    p.add_argument("--gpu", default="NVIDIA A40")
    p.add_argument("--cloud", default="SECURE", choices=("SECURE", "COMMUNITY"))
    p.add_argument("--max-price", type=float, default=0.65, help="dollars an hour")
    p.add_argument("--minutes", type=int, default=90,
                   help="the machine removes itself after this long: give about twice what the job should take, and --remove it when done")
    p.add_argument("--regions", default="US,CA")
    p.add_argument("--min-reliability", type=float, default=0.98)
    p.add_argument("--create-tries", type=int, default=3)
    p.add_argument("--go", action="store_true", help="really rent")
    p.add_argument("--remove", action="store_true")
    a = p.parse_args()
    host = Vast48(a) if a.on == "vast" else RunPodServer(a)

    if a.remove:
        if not os.path.exists(NOTE):
            raise SystemExit("no podcast machine on record")
        m = json.load(open(NOTE))
        host = render.a_host(m)
        gone = host.remove(m["id"])
        hours = (time.time() - m["created"]) / 3600.0
        print(f"machine {m['id']}: {'gone' if gone else 'REMOVAL FAILED'} after {hours * 60:.0f} minutes, "
              f"about ${hours * m['cost']:.2f} at ${m['cost']}/hr before traffic")
        if gone:
            os.remove(NOTE)
        return

    if a.on == "runpod":
        q = host.quote(1)
        if not q["prices"]:
            raise SystemExit(f"RunPod has no price for {a.gpu} in {a.cloud}")
        price = q["prices"][0]
        print(f"{host.describe()}: ${price}/hr, {q['stock']}; traffic is not billed")
        print(f"{a.minutes} minutes: ${price * a.minutes / 60:.2f}")
        if price > a.max_price:
            raise SystemExit(f"that is above --max-price ${a.max_price}")
    else:
        quote_vast(host, a)
    if not a.go:
        print("nothing rented (add --go)")
        return
    if os.path.exists(NOTE):
        raise SystemExit(f"a podcast machine is already on record ({NOTE}); remove it first")

    pub = render.ensure_key()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    state = dict(file=os.path.join(render.CLOUD, f"run-{stamp}-podcast.json"), film="podcast", shot="", pods=[])
    info = {}
    t0 = time.time()
    deadline = t0 + a.minutes * 60
    mid = render.rent(host, "podcast-tts", a, pub, info, state)
    if mid is None:
        raise SystemExit(info.get("note", "could not rent"))
    note = dict(id=mid, on=host.name, cost=info["cost"], created=t0, minutes=a.minutes)
    json.dump(note, open(NOTE, "w"))
    if a.on == "runpod":
        status = None
        while time.time() - t0 < 900:
            up, status = host.running(mid)
            if up:
                break
            time.sleep(10)
        else:
            print(f"not running after 15 minutes (status {status}) ->", "gone" if host.remove(mid) else "REMOVAL FAILED")
            os.remove(NOTE)
            raise SystemExit(1)
        render.say(f"running after {time.time() - t0:.0f} s at ${info['cost']}/hr; pod {mid}")
        render.say(f"its port 8000: https://{mid}-8000.proxy.runpod.net")
        render.say("it has no timer of its own yet: remove it by " + time.strftime("%H:%M", time.localtime(deadline)))
        return
    addr = render.wait_for_ssh(host, mid, t0, deadline, info)
    if addr is None:
        print(info.get("note"), "->", "gone" if host.remove(mid) else "REMOVAL FAILED")
        os.remove(NOTE)
        raise SystemExit(1)
    r = render.ssh(addr, "mkdir -p /root/job && touch /root/.no_auto_tmux\n"
                   + host.self_remove(int(deadline - time.time())) + "\nnvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    safety = "SAFETY ok" in r.stdout
    json.dump(dict(note, host=addr[0], port=addr[1]), open(NOTE, "w"))
    render.say(f"up after {time.time() - t0:.0f} s at ${info['cost']}/hr: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else '?'}")
    render.say("removes itself at " + time.strftime("%H:%M", time.localtime(deadline))
               + ("" if safety else "  << THE SELF-REMOVAL TIMER DID NOT START: remove it by hand when done"))
    render.say(f"ssh -i {render.KEY} -p {addr[1]} root@{addr[0]}")


def quote_vast(host, a):
    offers = host.search()
    print(f"{len(offers)} machines match: {a.gpu} with 46 GB or more, x86, {a.regions}, at most ${a.max_price}/hr")
    for o in offers[:4]:
        print(f"  ${o['dph_total']:.3f}/hr  {o['gpu_ram'] / 1024:.0f} GB  {o.get('geolocation')}  "
              f"{o.get('reliability', 0) * 100:.1f}% reliable  down {o.get('inet_down', 0):.0f} Mb/s  "
              f"traffic ${o.get('inet_down_cost') or 0:.4f}/GB down, ${o.get('inet_up_cost') or 0:.4f}/GB up")
    if not offers:
        raise SystemExit(1)
    top = offers[0]
    traffic = 30.0 * (top.get("inet_down_cost") or 0)
    print(f"the cheapest for {a.minutes} minutes: ${top['dph_total'] * a.minutes / 60:.2f}, plus about "
          f"${traffic:.2f} to download 30 GB; never more than ${a.max_price * a.minutes / 60:.2f} plus traffic")


if __name__ == "__main__":
    main()
