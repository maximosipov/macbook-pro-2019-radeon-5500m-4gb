#!/usr/bin/env bash
# Coding-agent bake-off on the Radeon 5500M, one model at a time.
#   1. llama-bench pp512/tg128 at depth 0 and 12288 (OpenCode's opening prompt is ~11K
#      tokens, so the depth-12K row is the speed an agent actually sees), q4_0 KV, FA on
#   2. HumanEval+ (20 problems) and multi-turn tools (8 scenarios), thinking off
# Results land in data/results/ (gitignored). See models.md#coding-agents.
# Usage: coding-eval.sh name=/path/model.gguf ...
set -uo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)"
# BIN: a llama.cpp build from the fork (llama.cpp.md). PY: a venv with numpy, which
# humaneval.py also expects at tools/eval-venv.
BIN="${BIN:-$W/llama.cpp/build/bin}"
PY="${PY:-$W/tools/eval-venv/bin/python}"
export GGML_VK_VISIBLE_DEVICES=0
PORT=8099; U="http://127.0.0.1:$PORT"
R="$W/data/results"; mkdir -p "$R"

serve() { # model extra...
  local m="$1"; shift
  pkill -f "llama-server.*--port $PORT" 2>/dev/null; sleep 2
  "$BIN/llama-server" -m "$m" -ngl 99 -c 8192 --host 127.0.0.1 --port $PORT --no-webui \
    -np 1 --temp 0 --jinja --reasoning-format deepseek --reasoning off --reasoning-budget 0 "$@" \
    > "$R/serve.log" 2>&1 &
  echo $! > "$R/.pid"
  for i in $(seq 1 240); do
    curl -s "$U/health" | grep -q ok && return 0
    kill -0 "$(cat "$R/.pid")" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

for spec in "$@"; do
  name="${spec%%=*}"; m="${spec#*=}"
  echo "=== $name $(date +%H:%M:%S)"
  "$BIN/llama-bench" -m "$m" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128 -d 0,12288 -r 2 \
    -o json > "$R/$name.bench.json" 2> "$R/$name.bench.log" \
    || echo "  bench FAILED (see $name.bench.log)"
  "$PY" -I -c 'import json,sys
for r in json.load(open(sys.argv[1])):
    t="pp%d"%r["n_prompt"] if r["n_prompt"] else "tg%d"%r["n_gen"]
    print("  %-6s depth %5d  %7.1f +- %.1f tok/s"%(t,r.get("n_depth",0),r["avg_ts"],r["stddev_ts"]))' \
    "$R/$name.bench.json" 2>/dev/null
  sleep 20
  if ! serve "$m" --chat-template-kwargs '{"enable_thinking":false}'; then
    echo "  serve with enable_thinking kwarg failed; retrying without"
    serve "$m" || { echo "  SERVE FAILED"; tail -5 "$R/serve.log"; continue; }
  fi
  "$PY" -I "$W/scripts/humaneval.py" "$name" "$U" | tail -1
  "$PY" -I "$W/scripts/multiturn.py" "$name" "$U" | tail -1
  kill "$(cat "$R/.pid")" 2>/dev/null; sleep 20
done
echo "=== done $(date +%H:%M:%S)"
