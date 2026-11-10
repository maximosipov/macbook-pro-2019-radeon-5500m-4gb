#!/usr/bin/env python3
"""Count full prefills per OpenCode task through LocalAI (Gemma-4-E2B, opencode.md#verified).

Runs tasks A, B, A, B in fresh temporary projects, then reads the
backend log for the prompt tokens each model call processed. A call that processes
more than 5,000 tokens is a full prefill. Writes cachetest-<label>.json to the current
directory.

Usage: opencode-cachetest.py <label>
"""
import json, os, re, subprocess, sys, tempfile, time, pathlib, urllib.request

LABEL = sys.argv[1]
MODEL = "gemma-4-e2b-coding"
BASE = "http://127.0.0.1:8085"
DB = pathlib.Path.home() / ".local/share/opencode/opencode.db"

def http(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    return urllib.request.urlopen(req, timeout=60).read().decode()

def vram():
    o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    m = re.findall(r'"inUseVidMemoryBytes"=(\d+)', o)
    return max(int(x) for x in m) if m else None

def oc(prompt, d):
    d = os.path.realpath(str(d))
    t0 = time.time()
    r = subprocess.run(["opencode", "run", "--dir", d, "-m", f"localai/{MODEL}", prompt],
                       cwd=d, env=dict(os.environ, PWD=d), capture_output=True, text=True, timeout=1800)
    got = subprocess.run(["sqlite3", str(DB), "select directory from session order by time_created desc limit 1"],
                         capture_output=True, text=True).stdout.strip()
    if os.path.realpath(got) != d:
        sys.exit(f"ABORT: session in {got}, expected {d}")
    return round(time.time() - t0, 1)

def prefills():
    raw = http(f"/api/backend-logs/{MODEL}")
    evals = [int(n) for n in re.findall(r"prompt eval time = +[0-9.]+ ms / +(\d+) tokens", raw)]
    return evals

CALC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
STATS = ("def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
         "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n")
TEST = ("from stats import mean, median\n\nassert mean([1, 2, 3]) == 2\n"
        "assert median([3, 1, 2]) == 2\nassert median([4, 1, 3, 2]) == 2.5, median([4, 1, 3, 2])\nprint('ok')\n")

out = {"label": LABEL, "tasks": []}
for kind in ["A", "B", "A", "B"]:
    try:
        http(f"/api/backend-logs/{MODEL}/clear", {})
    except Exception:
        pass
    d = pathlib.Path(tempfile.mkdtemp(prefix="opencode-cachetest-"))
    if kind == "A":
        (d / "calc.py").write_text(CALC)
        s = oc("The add function in calc.py returns the wrong result. Fix it. Reply DONE when finished.", d)
        ok = "return a + b" in (d / "calc.py").read_text()
    else:
        (d / "stats.py").write_text(STATS); (d / "test_stats.py").write_text(TEST)
        s = oc("The test in test_stats.py fails. Run it with `python3 test_stats.py`, find and fix the bug in "
               "stats.py (do not change the test), then run the test again to confirm it passes. Reply DONE when it passes.", d)
        ok = subprocess.run(["python3", "-s", "-E", "test_stats.py"], cwd=d, capture_output=True).returncode == 0 \
             and (d / "test_stats.py").read_text() == TEST
    ev = prefills()
    rec = {"kind": kind, "pass": ok, "seconds": s, "steps": len(ev), "full_prefills": sum(1 for n in ev if n > 5000),
           "prompt_tokens": ev, "vram": vram()}
    out["tasks"].append(rec); print(rec, flush=True)
pathlib.Path(f"cachetest-{LABEL}.json").write_text(json.dumps(out, indent=2))
print("total full prefills:", sum(t["full_prefills"] for t in out["tasks"]), "over", len(out["tasks"]), "tasks")
