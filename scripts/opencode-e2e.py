#!/usr/bin/env python3
"""OpenCode end to end through LocalAI, one model.

Latency: cold first turn (model unloaded), warm one-word, warm three-sentence.
Task A (x3): planted bug, add() returns a - b. Pass = file fixed.
Task B (x3): run a failing test, fix median() for even-length lists, re-run.
             Pass = test exits 0 and the test file is unchanged.
Each run is a fresh temporary project, so each pays the full prefill of OpenCode's
~11.5K-token opening prompt. Real use in one project reuses it after the first turn.
The agent edits files: OpenCode takes its project from $PWD, so this passes --dir,
cwd and PWD, and aborts if the session lands anywhere else. See opencode.md.
Usage: opencode-e2e.py <localai-model-name> <out.json>
"""
import json, os, subprocess, sys, tempfile, time, pathlib, urllib.request, hashlib

MODEL, OUT = sys.argv[1], sys.argv[2]
BASE = "http://127.0.0.1:8085"
TIMEOUT = 1200

DB = pathlib.Path.home() / ".local/share/opencode/opencode.db"

def last_session_dir():
    r = subprocess.run(["sqlite3", str(DB), "select directory from session order by time_created desc limit 1"],
                       capture_output=True, text=True)
    return r.stdout.strip()

def oc(prompt, cwd):
    # OpenCode resolves its project from PWD, not the process cwd: pass all three,
    # and refuse to continue if the session landed anywhere else.
    cwd = os.path.realpath(str(cwd))
    env = dict(os.environ, PWD=cwd)
    t0 = time.time()
    try:
        r = subprocess.run(["opencode", "run", "--dir", cwd, "-m", f"localai/{MODEL}", prompt],
                           cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = "TIMEOUT", -1
    got = os.path.realpath(last_session_dir())
    if got != cwd:
        sys.exit(f"ABORT: OpenCode session ran in {got}, expected {cwd}")
    return round(time.time() - t0, 1), rc, out

def unload():
    for path, body in (("/backend/shutdown", {"model": MODEL}),):
        try:
            urllib.request.urlopen(urllib.request.Request(BASE + path,
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}),
                timeout=60).read()
        except Exception as e:
            print("  unload:", e)
    time.sleep(5)

CALC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
STATS = ("def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
         "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]\n")
TEST = ("from stats import mean, median\n\n"
        "assert mean([1, 2, 3]) == 2\n"
        "assert median([3, 1, 2]) == 2\n"
        "assert median([4, 1, 3, 2]) == 2.5, median([4, 1, 3, 2])\n"
        "print('ok')\n")

res = {"model": MODEL, "latency": {}, "task_a": [], "task_b": []}
empty = tempfile.mkdtemp()

unload()
s, rc, out = oc("Reply with exactly the word: ready", empty)
res["latency"]["cold_one_word"] = s; print(f"  cold one-word {s}s rc={rc} {out.strip()[-60:]!r}", flush=True)
for i in range(2):
    s, rc, out = oc("Reply with exactly the word: ready", empty)
    res["latency"].setdefault("warm_one_word", []).append(s); print(f"  warm one-word {s}s {out.strip()[-40:]!r}", flush=True)
s, rc, out = oc("In three sentences, explain what a hash map is. Do not use any tools.", empty)
res["latency"]["warm_three_sentences"] = s; print(f"  warm three sentences {s}s", flush=True)

for i in range(3):
    d = pathlib.Path(tempfile.mkdtemp()); (d / "calc.py").write_text(CALC)
    s, rc, out = oc("The add function in calc.py returns the wrong result. Fix it. "
                    "Reply DONE when finished.", d)
    ok = "return a + b" in (d / "calc.py").read_text().replace("b + a", "a + b")
    res["task_a"].append({"pass": ok, "seconds": s, "rc": rc, "tail": out.strip()[-300:]})
    print(f"  task A #{i+1} {'PASS' if ok else 'FAIL'} {s}s", flush=True)

for i in range(3):
    d = pathlib.Path(tempfile.mkdtemp()); (d / "stats.py").write_text(STATS); (d / "test_stats.py").write_text(TEST)
    h = hashlib.sha256(TEST.encode()).hexdigest()
    s, rc, out = oc("The test in test_stats.py fails. Run it with `python3 test_stats.py`, find and fix "
                    "the bug in stats.py (do not change the test), then run the test again to confirm "
                    "it passes. Reply DONE when it passes.", d)
    t = subprocess.run(["python3", "-s", "-E", "test_stats.py"], cwd=d, capture_output=True, text=True)
    unchanged = hashlib.sha256((d / "test_stats.py").read_bytes()).hexdigest() == h
    ok = t.returncode == 0 and unchanged
    res["task_b"].append({"pass": ok, "seconds": s, "rc": rc, "test_unchanged": unchanged,
                          "stats_py": (d / "stats.py").read_text(), "tail": out.strip()[-400:]})
    print(f"  task B #{i+1} {'PASS' if ok else 'FAIL'} {s}s unchanged={unchanged}", flush=True)

pathlib.Path(OUT).write_text(json.dumps(res, indent=2))
print("  A %d/3  B %d/3" % (sum(x["pass"] for x in res["task_a"]), sum(x["pass"] for x in res["task_b"])))
