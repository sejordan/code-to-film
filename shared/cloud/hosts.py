"""Who rents us the machines: RunPod or Vast.ai.

The render drivers (render.py and film.py, next to this file) ask one of the
two classes here for the same few things: what it would cost, a machine, the
machine's SSH address, and the machine's removal. Everything after that
(sending the project, installing Blender, rendering, fetching frames) is
plain SSH and is the same wherever the machine is.

RunPod   a pod is a container in one of RunPod's own data centres (SECURE)
         or on a vetted host's machine (COMMUNITY). One price per GPU model.
         Its API key is read from .env (RUNPOD_API_KEY) and is never printed.
Vast.ai  a marketplace: each offer is one host's machine at that host's
         price, so the plan is a list of particular offers, cheapest first,
         filtered for reliability, connection and region. Most hosts are
         individuals, not data centres. Download and upload are billed by
         the gigabyte on top of the hourly price. Renting goes through the
         `vastai` command-line tool, which keeps its own copy of the API key
         (`vastai set api-key`); this script never reads the key.

Either way a machine is told to remove itself after the time limit, using a
key that the service gives each machine for that one purpose.
"""
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DISK = 20                       # GB of disk on each machine: Blender, the project and a share of the frames


# ---------------------------------------------------------------- RunPod

RUNPOD_API = "https://api.runpod.io/v2"


def runpod_key():
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.strip().startswith("RUNPOD_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("RUNPOD_API_KEY not found in .env")


def runpod(method, path, body=None):
    req = urllib.request.Request(RUNPOD_API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + runpod_key(),
                                          "Content-Type": "application/json",
                                          "User-Agent": "toy-film-render/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"detail": raw.decode(errors="replace")[:400]}


class RunPod:
    name = "runpod"
    image = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    patience = 420              # seconds to wait for a new machine's SSH address before calling it a bad one

    def __init__(self, a):
        self.gpu, self.cloud, self.tries = a.gpu, a.cloud, a.create_tries

    def describe(self):
        return f"{self.gpu} (RunPod {self.cloud})"

    def quote(self, n):
        """What n machines would cost: a price per hour for each, and the stock."""
        code, cat = runpod("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=" + self.cloud)
        entry = next((g for g in cat.get("gpus", []) if g["id"] == self.gpu), None) if code == 200 else None
        price = (entry or {}).get("price", {}).get(self.cloud.lower())
        stock = (entry or {}).get("availability") or (entry or {}).get("stockStatus")
        return dict(prices=[] if price is None else [price] * n, stock=f"stock: {stock}", note="")

    def create(self, name, pub):
        """Rent one machine: dict(id, cost per hour). Stock comes and goes, so a few tries."""
        code, pod = 0, {}
        for attempt in range(self.tries):
            code, pod = runpod("POST", "/pods", dict(
                name=name, image=self.image, cloud=self.cloud, gpu=dict(id=self.gpu, count=1), disk=DISK,
                ports=["22/tcp"], env={"PUBLIC_KEY": pub}))
            if code < 300 and "id" in pod:
                return dict(id=pod["id"], cost=pod.get("cost"))
            time.sleep(20)
        raise RuntimeError(f"HTTP {code} {pod.get('title', '')} {pod.get('detail', '')}")

    def address(self, mid):
        """Public IP and outside port for the machine's SSH, once the mapping exists."""
        code, pod = runpod("GET", f"/pods/{mid}")
        if code != 200:
            return None
        direct = (pod.get("ssh") or {}).get("direct") or {}
        host = direct.get("host") or direct.get("ip")
        if host and direct.get("port"):
            return host, int(direct["port"])
        for p in (pod.get("runtime") or {}).get("ports") or []:
            inside = p.get("private") or p.get("privatePort") or p.get("container")
            outside = p.get("public") or p.get("publicPort") or p.get("external")
            if inside == 22 and outside and p.get("ip"):
                return p["ip"], int(outside)
        return None

    def alive(self, mid):
        code, body = runpod("GET", f"/pods/{mid}")
        return code == 200 and body.get("status") not in ("TERMINATED", "DELETED")

    def remove(self, mid):
        """Remove a machine and confirm it is gone."""
        for _ in range(4):
            code, _body = runpod("DELETE", f"/pods/{mid}")
            if code == 404:
                return True
            time.sleep(3)
            code, body = runpod("GET", f"/pods/{mid}")
            if code == 404 or body.get("status") in ("TERMINATED", "DELETED"):
                return True
        return False

    def self_remove(self, seconds):
        """Shell for the machine itself: remove this machine after `seconds`, whatever happens."""
        return (
            "v=$(cat /proc/1/environ | tr '\\0' '\\n' | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs) || true\n"
            '[ -n "$v" ] && export $v\n'
            'if [ -n "$RUNPOD_POD_ID" ]; then echo SAFETY ok; else echo SAFETY missing; fi\n'
            f'nohup sh -c "sleep {seconds}; runpodctl pod remove $RUNPOD_POD_ID || runpodctl remove pod $RUNPOD_POD_ID" '
            "> /root/job/selfremove.log 2>&1 &")


# ---------------------------------------------------------------- Vast.ai

def vast(*args, timeout=90):
    """Run the vastai tool and return what it prints, read as JSON (None if
    it is not JSON), and the raw text. The tool has its own copy of the key."""
    try:
        r = subprocess.run(["vastai", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise SystemExit("the vastai tool is not installed: https://cloud.vast.ai/cli/")
    except subprocess.TimeoutExpired:
        return None, "vastai did not answer"
    text = (r.stdout or "").strip()
    try:
        return json.loads(text), text
    except ValueError:
        return None, (text + " " + (r.stderr or "").strip()).strip()[:400]


class Vast:
    name = "vast"
    image = "nvidia/cuda:12.4.1-base-ubuntu22.04"       # small, so a host that has not got it fetches it quickly
    # The least CUDA a host's driver must support: the image's own. On 2026-11-20 a host whose driver
    # stopped at CUDA 12.2 rendered at 13 s a frame instead of 1 (Blender, most likely, falling back to
    # the CPU) and ran out the clock with half its share undone.
    CUDA = 12.4
    patience = 600              # a host may have to download the image first
    MAX_PER_GB = 0.02           # dollars: hosts that charge more than this for traffic are passed over

    def __init__(self, a):
        self.gpu = a.gpu.replace("NVIDIA ", "").replace("GeForce ", "")     # "NVIDIA GeForce RTX 4090" -> "RTX 4090"
        self.max_price, self.regions, self.reliability = a.max_price, a.regions, a.min_reliability
        self.tries = a.create_tries
        self.offers = None      # the machines that could be rented, cheapest first
        self.lock = threading.Lock()
        self.key_ok = False

    def describe(self):
        return f"{self.gpu} (Vast.ai, {self.regions})"

    def search(self):
        """Single-GPU machines of this model that are verified, reliable, well
        connected, in the wanted countries and within the price limit, one
        offer per machine, cheapest first."""
        q = (f"gpu_name={self.gpu.replace(' ', '_')} num_gpus=1 rentable=true verified=true "
             f"reliability>={self.reliability} inet_up>=100 inet_down>=100 cpu_cores_effective>=8 "
             f"disk_space>={DISK + 10} direct_port_count>=2 dph<={self.max_price} "
             f"cuda_max_good>={self.CUDA}")
        if self.regions.lower() != "any":
            q += f" geolocation in [{self.regions}]"
        found, text = vast("search", "offers", q, "--storage", str(DISK), "--limit", "300", "--raw")
        if not isinstance(found, list):
            raise SystemExit("Vast.ai search failed: " + text)
        seen, offers = set(), []
        for o in sorted(found, key=lambda o: o["dph_total"]):
            dear = max(o.get("inet_up_cost") or 0, o.get("inet_down_cost") or 0) > self.MAX_PER_GB
            if o["dph_total"] > self.max_price or dear or o["machine_id"] in seen:
                continue
            seen.add(o["machine_id"])
            offers.append(o)
        self.offers = offers
        return offers

    def quote(self, n):
        offers = self.search()
        pick = offers[:n]
        per_gb = max([max(o.get("inet_up_cost") or 0, o.get("inet_down_cost") or 0) for o in pick] or [0])
        where = ", ".join(sorted({(o.get("geolocation") or "?").split(", ")[-1] for o in pick}))
        return dict(prices=[round(o["dph_total"], 3) for o in pick],
                    stock=f"{len(offers)} machines match",
                    note=f"hosts in {where}; traffic billed at up to ${per_gb:.3f}/GB on top")

    def account_key(self, pub):
        """Vast gives a new machine the SSH keys on the account, so this
        project's public key has to be one of them."""
        with self.lock:
            if self.key_ok:
                return
            keys, text = vast("show", "ssh-keys", "--raw")
            if pub.split()[1] not in json.dumps(keys if keys is not None else text):
                vast("create", "ssh-key", pub, "-y")
                keys, text = vast("show", "ssh-keys", "--raw")
                if pub.split()[1] not in json.dumps(keys if keys is not None else text):
                    raise RuntimeError("could not add this project's SSH key to the Vast.ai account: " + text[:200])
            self.key_ok = True

    def create(self, name, pub):
        """Rent the cheapest offer not yet taken. Another renter may get
        there first, so on failure the next offer is tried."""
        self.account_key(pub)
        if self.offers is None:
            self.search()
        text = "no machine on offer matches"
        for attempt in range(self.tries):
            with self.lock:
                if not self.offers:
                    break
                o = self.offers.pop(0)
            got, text = vast("create", "instance", str(o["id"]), "--image", self.image, "--disk", str(DISK),
                             "--ssh", "--direct", "--label", name, "--env", "-e NVIDIA_DRIVER_CAPABILITIES=all",
                             "--cancel-unavail", "--raw", timeout=120)
            if isinstance(got, dict) and got.get("success") and got.get("new_contract"):
                return dict(id=str(got["new_contract"]), cost=round(o["dph_total"], 3),
                            where=f"{o.get('geolocation')}, {o.get('reliability', 0) * 100:.1f}% reliable, "
                                  f"{o.get('inet_up', 0):.0f} Mb/s up")
        raise RuntimeError(text[:300])

    def address(self, mid):
        inst, _ = vast("show", "instance", str(mid), "--raw")
        if not isinstance(inst, dict) or inst.get("actual_status") != "running":
            return None
        mapped = (inst.get("ports") or {}).get("22/tcp")
        if mapped and inst.get("public_ipaddr"):          # straight to the host
            return inst["public_ipaddr"].strip(), int(mapped[0]["HostPort"])
        if inst.get("ssh_host") and inst.get("ssh_port"):  # through Vast's relay
            return inst["ssh_host"], int(inst["ssh_port"])
        return None

    def alive(self, mid):
        """Whether the account still has this machine. If Vast cannot be
        asked, it is taken to be there: better a second look than a machine
        left running."""
        mine, _ = vast("show", "instances", "--raw")
        if not isinstance(mine, list):
            return True
        return any(str(i.get("id")) == str(mid) for i in mine)

    def remove(self, mid):
        for _ in range(4):
            vast("destroy", "instance", str(mid), "-y", "--raw")
            time.sleep(3)
            if not self.alive(mid):
                return True
        return False

    def self_remove(self, seconds):
        url = "https://console.vast.ai/api/v0/instances/$CONTAINER_ID"
        return (
            "v=$( (cat /proc/1/environ | tr '\\0' '\\n'; cat /etc/environment 2>/dev/null) "
            "| grep -E '^CONTAINER_(API_KEY|ID)=' | xargs) || true\n"
            '[ -n "$v" ] && export $v\n'
            'if [ -n "$CONTAINER_ID" ] && [ -n "$CONTAINER_API_KEY" ]; then echo SAFETY ok; else echo SAFETY missing; fi\n'
            f"nohup sh -c \"sleep {seconds}; for u in {url}/ {url}; do "
            "curl -s -X DELETE -H 'Authorization: Bearer '$CONTAINER_API_KEY \\$u; done\" "
            "> /root/job/selfremove.log 2>&1 &")


HOSTS = {"runpod": RunPod, "vast": Vast}


def add_arguments(ap):
    ap.add_argument("--on", default="runpod", choices=sorted(HOSTS), help="where to rent")
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"],
                    help="runpod: its own data centres, or the cheaper machines of vetted hosts")
    ap.add_argument("--regions", default="US,CA",
                    help="vast: two-letter countries the machines may be in, or 'any'")
    ap.add_argument("--min-reliability", type=float, default=0.98,
                    help="vast: the least reliability score a host may have, 0 to 1")
    ap.add_argument("--create-tries", type=int, default=4,
                    help="attempts to rent each machine (runpod: 20 s apart; vast: the next offer each time)")
