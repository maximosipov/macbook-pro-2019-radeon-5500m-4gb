#!/usr/bin/env python3
"""Multi-turn tool use: state across turns, and recovery after a failed call.

From the companion repository maximosipov/macbook-pro-2023-m2-pro-32gb
(scripts/multiturn.py), unchanged except this paragraph. A single-turn tool
test ranks little: most models here pass it. An agent loop fails in ways a single turn cannot show: losing the
earlier turn's context, giving up (or retrying blindly) after a tool error,
never feeding one call's result into the next, or inventing a tool when none
fits.

Each scenario is a short conversation against scripted tools whose replies are
fixed, including the errors. The model may make up to ROUNDS tool rounds per
user turn. A scenario passes or fails as a whole; it also fails on any call to
a tool that does not exist or with arguments that are not valid JSON.

Scoring: fraction of scenarios passed. A scenario where the model ran out of
budget before replying counts as failed and is also tallied as unparsed.

Usage: multiturn.py <name> <base_url>
"""
import json, os, re, sys, time, pathlib, urllib.request

NAME, BASE = sys.argv[1], sys.argv[2]
D = pathlib.Path(__file__).resolve().parent.parent
BUDGET = int(os.environ.get("BUDGET", "2048"))
ROUNDS = 6

def chat(messages, tools, timeout=int(os.environ.get("TIMEOUT", "7200"))):
    body = {"model": "local", "messages": messages, "tools": tools, "temperature": 0,
            "max_tokens": BUDGET, "stream": False}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    return c["message"], c.get("finish_reason")

TOOLS = [
 {"type":"function","function":{"name":"get_weather","description":"Current weather for a city",
  "parameters":{"type":"object","properties":{"city":{"type":"string"},
   "unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}},
 {"type":"function","function":{"name":"send_email","description":"Send an email",
  "parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},
   "body":{"type":"string"}},"required":["to","subject","body"]}}},
 {"type":"function","function":{"name":"add","description":"Add two numbers",
  "parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},
   "required":["a","b"]}}},
]
NAMES = {t["function"]["name"] for t in TOOLS}
WEATHER = {"paris": 18, "berlin": 12, "tokyo": 22, "saint petersburg": 9}   # Celsius
# The tool knows only the full name. Models silently fix a misspelling like
# "Sanfrancisco" before calling, which tests nothing; "St. Petersburg" is a
# correct spelling they pass through, so the error actually fires.
ALIAS = {"st. petersburg": "Saint Petersburg", "st petersburg": "Saint Petersburg"}

class Tools:
    """Scripted tool backend. `fail_first` makes the first call to that tool
    return a transient error, as a flaky service would."""
    def __init__(self, fail_first=None):
        self.fail_first = fail_first
    def __call__(self, name, args):
        if name == self.fail_first:
            self.fail_first = None
            return {"error": "503 Service Unavailable. Temporary failure, please retry."}
        if name == "get_weather":
            city = str(args.get("city", "")).strip().lower()
            if city not in WEATHER:
                hint = f" Did you mean '{ALIAS[city]}'?" if city in ALIAS else ""
                return {"error": f"Unknown city '{args.get('city')}'.{hint}"}
            c = WEATHER[city]
            if args.get("unit") == "f":
                return {"city": city.title(), "temperature": round(c * 9 / 5 + 32), "unit": "F"}
            return {"city": city.title(), "temperature": c, "unit": "C"}
        if name == "send_email":
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(args.get("to", ""))):
                return {"error": f"Invalid recipient address '{args.get('to')}'."}
            return {"status": "sent", "id": "msg-0001"}
        if name == "add":
            return {"result": args["a"] + args["b"]}

def has(text, *words):
    return all(re.search(rf"(?<![\d.]){re.escape(w)}(?![\d.])" if w[0].isdigit() else re.escape(w),
                         text, re.I) for w in words)

def city(c): return str(c[1].get("city", "")).strip().lower()

def clean(reply):
    """Drop empty think tags: with thinking forced off, one model emitted
    nothing but `<think></think>` repeated, which is not a reply."""
    return re.sub(r"</?think>", "", reply).strip()

def temp(reply, c):
    """The model picks the unit unless told, so accept either."""
    return has(reply, str(WEATHER[c])) or has(reply, str(round(WEATHER[c] * 9 / 5 + 32)))

# Each scenario: user turns, the tool backend, and a check over the trace.
# trace[i] = {"calls": [(name, args), ...], "reply": final text of user turn i}
SCENARIOS = [
 ("follow_up", ["What's the weather in Paris?", "And in Berlin?"], Tools(),
  lambda t: any(n == "get_weather" and city((n, a)) == "berlin" for n, a in t[1]["calls"])
            and temp(t[1]["reply"], "berlin")),
 ("correct_after_error", ["What's the weather in St. Petersburg right now?"], Tools(),
  lambda t: any(city(c) == "saint petersburg" for c in t[0]["calls"]) and temp(t[0]["reply"], "saint petersburg")),
 ("retry_transient", ["What's the weather in Tokyo, in celsius?"], Tools(fail_first="get_weather"),
  lambda t: sum(n == "get_weather" for n, _ in t[0]["calls"]) >= 2 and has(t[0]["reply"], "22")),
 ("chain_result", ["Get the current weather in Tokyo in fahrenheit, then email it to "
                   "alice@example.com with the subject 'Tokyo weather'."], Tools(),
  lambda t: [n for n, _ in t[0]["calls"]][:1] == ["get_weather"]
            and any(n == "send_email" and a.get("to") == "alice@example.com" and has(str(a.get("body")), "72")
                    for n, a in t[0]["calls"])),
 ("two_step_arith", ["Use the add tool to compute 17 + 25 + 8."], Tools(),
  lambda t: sum(n == "add" for n, _ in t[0]["calls"]) >= 2 and has(t[0]["reply"], "50")),
 # The address is invalid. Resending it, or guessing a "fixed" one, are both
 # failures: the right move is to report the problem and ask.
 ("invalid_no_guess", ["Email bob@example with the subject 'Hi' and the body 'Hello'."], Tools(),
  lambda t: sum(n == "send_email" for n, _ in t[0]["calls"]) <= 1 and t[0]["reply"] != ""),
 ("no_call_after_tools", ["What's the weather in Paris?", "Thanks! Who wrote Hamlet?"], Tools(),
  lambda t: not t[1]["calls"] and has(t[1]["reply"], "Shakespeare")),
 ("no_tool_fits", ["Book me a flight to Rome for tomorrow morning."], Tools(),
  lambda t: not t[0]["calls"] and t[0]["reply"] != ""),
]

def play(turns, backend):
    msgs, trace = [], []
    for u in turns:
        msgs.append({"role": "user", "content": u})
        calls, reply, cut = [], "", False
        for r in range(ROUNDS):
            m, finish = chat(msgs, TOOLS)
            tc = m.get("tool_calls") or []
            if finish == "length": cut = True
            if not tc:
                msgs.append({"role": "assistant", "content": m.get("content") or ""})
                reply = clean(m.get("content") or "")
                break
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tc})
            for i, c in enumerate(tc):
                name = c["function"]["name"]
                try: args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError: return trace, f"bad arguments for {name}", cut
                if name not in NAMES: return trace, f"invented tool {name}", cut
                calls.append((name, args))
                msgs.append({"role": "tool", "tool_call_id": c.get("id") or f"call_{r}_{i}",
                             "content": json.dumps(backend(name, args))})
        trace.append({"calls": calls, "reply": reply})
        if cut and not reply: return trace, "out of budget", cut
    return trace, None, False

if __name__ == "__main__":
    ok = unparsed = errors = 0
    results = []
    for name, turns, backend, check in SCENARIOS:
        t0 = time.time()
        try:
            trace, fault, cut = play(turns, backend)
        except Exception as e:
            errors += 1
            results.append({"scenario": name, "status": "error", "error": str(e)[:70]})
            print(f"  {name:<20} ERROR {str(e)[:40]}", flush=True)
            continue
        if fault == "out of budget": unparsed += 1
        passed = fault is None and check(trace)
        ok += passed
        status = "pass" if passed else "fail"
        results.append({"scenario": name, "status": status, "fault": fault,
                        "calls": [[n, a] for t in trace for n, a in t["calls"]],
                        "reply": (trace[-1]["reply"] if trace else "")[:200],
                        "seconds": round(time.time() - t0, 1)})
        print(f"  {name:<20} {status:<5} {time.time()-t0:6.1f}s {fault or ''}", flush=True)

    out = {"model": NAME, "n": len(SCENARIOS), "correct": ok, "unparsed": unparsed,
           "errors": errors, "score": round(ok / len(SCENARIOS), 3), "budget": BUDGET,
           "results": results}
    (D / "data" / "results" / (NAME + ".multiturn.json")).write_text(json.dumps(out, indent=2))
    print(f"  Multi-turn tools {out['score']} ({ok}/{len(SCENARIOS)}, {unparsed} out of budget, {errors} errors)", flush=True)
