#!/usr/bin/env python3
"""HumanEval+: write a Python function from its signature and docstring.

From the companion repository maximosipov/macbook-pro-2023-m2-pro-32gb
(scripts/humaneval.py); only this paragraph and the dataset note differ, so
scores compare across the two machines. HumanEval+ is HumanEval with EvalPlus's extended
tests -- tens to thousands of inputs per problem instead of a handful -- so a
solution that only handles the docstring's examples fails.

Sample: every eighth problem, 20 of 164 after one with a broken test. The set runs roughly easy to hard, so
a stride covers the range where a prefix would take only the easy end.

Scoring: pass@1 at temperature 0, the fraction of problems whose code passes
every test. A reply with no ```python block defining the entry point is
UNPARSED, counted separately from code that runs and fails.

The model's code is executed. Each program runs under sandbox-exec with network
denied and writes confined to its own temporary directory, in a separate
interpreter (tools/eval-venv, which provides the numpy the tests import), with
a timeout. There is no memory cap: macOS does not enforce RLIMIT_AS.

Dataset: data/datasets/humaneval-plus.json, the evalplus/humanevalplus test split
(data/test-00000-of-00001-5973903632b82d40.parquet on Hugging Face) as a JSON list
of records, e.g. pandas.read_parquet(f).to_json(out, orient="records").
tools/eval-venv is any Python 3 venv with numpy.

Usage: humaneval.py <name> <base_url> [stride]
"""
import json, os, re, subprocess, sys, tempfile, time, pathlib, urllib.request

NAME, BASE = sys.argv[1], sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 8
D = pathlib.Path(__file__).resolve().parent.parent
# HumanEval/32's test in this mirror unpacks a float (`_poly(*candidate(*inp), inp)`),
# so even the reference solution fails it. Excluding it leaves 20 problems.
BROKEN = {"HumanEval/32"}
ITEMS = [it for it in json.load(open(D / "data" / "datasets" / "humaneval-plus.json"))[::STRIDE]
         if it["task_id"] not in BROKEN]
PY = D / "tools" / "eval-venv" / "bin" / "python"
BUDGET = int(os.environ.get("BUDGET", "2048"))
RUN_TIMEOUT = 120   # the largest test files carry ~1000 inputs

def chat(prompt, max_tokens=BUDGET, timeout=int(os.environ.get("TIMEOUT", "7200"))):
    body = {"model": "local", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    # Only content counts: code left inside a think block was never answered.
    return d["choices"][0]["message"].get("content") or ""

def extract(txt, entry):
    """The last fenced block that defines the entry point."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", txt, re.S)
    for b in reversed(blocks):
        if re.search(rf"^\s*def {re.escape(entry)}\s*\(", b, re.M):
            return b
    return None

PROFILE = """(version 1)(allow default)(deny network*)
(deny file-write*)(allow file-write* (subpath "{tmp}") (literal "/dev/null"))"""

def run(code, it):
    prog = f"{code}\n\n{it['test']}\n\ncheck({it['entry_point']})\n"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)
        f = pathlib.Path(tmp) / "prog.py"; f.write_text(prog)
        try:
            r = subprocess.run(
                ["sandbox-exec", "-p", PROFILE.format(tmp=tmp), str(PY), "-I", "-B", str(f)],
                cwd=tmp, capture_output=True, text=True, timeout=RUN_TIMEOUT,
                env={"TMPDIR": tmp, "PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return "timeout", ""
    if r.returncode == 0: return "pass", ""
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    return "fail", tail[:120]

if __name__ == "__main__":
    ok = unparsed = errors = 0
    results = []
    for it in ITEMS:
        p = (f"Complete this Python function.\n\n```python\n{it['prompt']}```\n\n"
             "Reply with the complete function, including any imports it needs, "
             "in a single ```python code block.")
        t0 = time.time()
        try:
            txt = chat(p)
        except Exception as e:
            errors += 1
            results.append({"task": it["task_id"], "status": "error", "error": str(e)[:70]})
            print(f"  {it['task_id']:<14} ERROR {str(e)[:40]}", flush=True)
            continue
        secs = time.time() - t0
        code = extract(txt, it["entry_point"])
        if code is None:
            unparsed += 1; status, why = "unparsed", ""
        else:
            status, why = run(code, it)
            if status == "pass": ok += 1
        results.append({"task": it["task_id"], "status": status, "why": why,
                        "seconds": round(secs, 1)})
        print(f"  {it['task_id']:<14} {status:<8} {secs:7.1f}s {why}", flush=True)

    out = {"model": NAME, "n": len(ITEMS), "correct": ok, "unparsed": unparsed,
           "errors": errors, "score": round(ok / len(ITEMS), 3) if ITEMS else None,
           "budget": BUDGET, "results": results}
    (D / "data" / "results" / (NAME + ".humaneval.json")).write_text(json.dumps(out, indent=2))
    print(f"  HumanEval+ {out['score']} ({ok}/{len(ITEMS)}, {unparsed} unparsed, {errors} errors)", flush=True)
