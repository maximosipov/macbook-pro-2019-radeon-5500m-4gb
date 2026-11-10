# OpenCode with Gemma-4-E2B, through LocalAI

A local coding agent on the Radeon: OpenCode driving Gemma-4-E2B through the LocalAI menu-bar app.

| | |
|---|---|
| **Role in the stack** | Coding-agent client of [LocalAI](localai.md) |
| **Model** | Gemma-4-E2B Q4_K_M, 64K context, f16 KV cache, thinking off, served from [`localai-models/gemma-4-e2b-coding.yaml`](localai-models/gemma-4-e2b-coding.yaml). Chosen in [Coding agents](models.md#coding-agents) |
| **Versions measured here** | OpenCode CLI 1.19.5 · LocalAI `v4.12.0-552-g98a562d33` in `LocalAI M.app`, llama-cpp backend on llama.cpp `4c4ca5b18` plus the fixes (`785322d17` for the end-to-end task results) · MoltenVK 1.4.2 · macOS 26.7.2 · 2026-11-10 |
| **Config** | `~/.config/opencode/opencode.json`, user level, [below](#the-config) |

## Result

- **Gemma-4-E2B works as an OpenCode agent through LocalAI.** It passed 15 of 16 end-to-end tasks, using OpenCode's own Read, Edit, Write and Bash tools. It fixed a one-line bug 9 of 9 times. 6 of 7 times it ran a failing test, fixed the code without touching the test, and reran the test to `ok`. Granite-4.0-H-Micro, the other finalist, passed 2 of 6.
- **Generation meets the bar: 34–39 tok/s** with OpenCode's 11.5K-token prompt in context, on a cool machine. Sustained back-to-back work heats the laptop until the CPU is held at 36–48% of its speed, and then generation drops to 24–27 tok/s and prefill to 165–175 tok/s. [Heat](#heat).
- **The first turn of a session takes about a minute.** LocalAI's llama.cpp backend prefills OpenCode's 11.5K-token prompt at 226–229 tok/s, the same as a standalone `llama-server`. Until the backend was rebuilt on llama.cpp `4c4ca5b18` it prefilled at about 75 tok/s, and the first turn took about 2.5 minutes. [The backend pin](#the-backend-pin).
- **Later turns reuse that prefill and take seconds, but not always.** Most measured tasks paid a second full prefill partway through: 5 of 8 on the old backend, 4 of 4 on the new one. On the new backend a miss costs about 55 s instead of 2.5 minutes. The cause is not established. llama.cpp's full-size sliding-window cache (`swa_full`) did not change it and costs 0.86 GB of VRAM, so the config does not set it. [Details](#verified).
- **One setting decides whether warm turns stay warm:** `agent.title.disable`. With one slot, OpenCode's title request would evict the agent's cached prompt. [The config](#the-config).

## The config

`~/.config/opencode/opencode.json`. The previous file is kept beside it as `opencode.json.bak-2026-11-10`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "enabled_providers": ["localai"],
  "model": "localai/gemma-4-e2b-coding",
  "small_model": "localai/gemma-4-e2b-coding",
  "agent": { "title": { "disable": true } },
  "provider": {
    "localai": {
      "name": "LocalAI (Radeon 5500M)",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8085/v1" },
      "models": {
        "gemma-4-e2b-coding": {
          "name": "Gemma-4-E2B (LocalAI, 64K)",
          "tool_call": true,
          "limit": { "context": 65536, "output": 8192 }
        }
      }
    }
  }
}
```

| Setting | Why |
|---|---|
| `agent.title.disable: true` | OpenCode sends a separate request to title each session. The model config has a single slot (`parallel:1`), so the title prompt would displace the agent's cached prompt, and the next turn would pay the full prefill again: about 2.5 minutes here. Sessions keep their default names. |
| `enabled_providers: ["localai"]` | Keeps everything local, so no cloud model is ever picked as a fallback. |
| `model`, `small_model` both Gemma | `small_model` handles summaries and compaction. A second model would evict Gemma: only one fits in 4 GB, and a switch with requests in flight loses one ([localai.md](localai.md#troubleshooting)). |
| Model key = LocalAI `name:` | LocalAI matches the request's `model` against the YAML's `name:`. |
| `baseURL` port 8085 | The menu-bar app's server. |
| `limit.context: 65536` | Must equal `context_size` in the YAML. 64K is verified with OpenCode's real prompt plus tool turns, at 2.17 GB resident. |
| `limit.output: 8192` | Thinking is off, so answers are short; 8K leaves 56K for the conversation. |
| `tool_call: true` | Gemma emits OpenAI-style `tool_calls` through LocalAI. Checked directly before the end-to-end runs: a correct call with valid JSON arguments, `finish_reason: tool_calls`. |
| No `reasoning: true` | The model config turns thinking off (`enable_thinking: false`, `reasoning_budget:0`). A 2B model thinking at length on every tool turn costs more time than it saves. |

On the model side, [the YAML](localai-models/gemma-4-e2b-coding.yaml) explains each setting. The one that matters for speed is the **f16 KV cache**: with 12K tokens in context, q4_0 KV decodes at 22.6 tok/s and f16 at 39.6.

## Verified

All on 2026-11-10, LocalAI resident on `127.0.0.1:8085`. Each `opencode run` used a fresh temporary project. [`scripts/opencode-e2e.py`](scripts/opencode-e2e.py) drove the runs. The task results below were measured on the old backend (`785322d17`); the [new backend](#the-backend-pin) changes speed, not the model's output.

| Task | Gemma-4-E2B, head-to-head | Gemma-4-E2B, later runs | Granite-4.0-H-Micro |
|---|---|---|---|
| A. `add()` returns `a - b`; fix it | **3/3**, 166–326 s | **6/6**, 170–364 s | 2/3, 149–161 s. One run read `/path/to/calc.py`, which does not exist |
| B. Run the failing test, fix `median()` for even lengths, rerun; the test must stay byte-identical | **3/3**, 380–530 s | 3/4, 225–530 s. The failure fixed even lengths, broke odd ones, and stopped | 0/3, 155–156 s. Ran the test, ran `stats.py`, replied DONE, edited nothing |

Granite's runs are faster because it stops early. The later runs are two smoke tests and the cache test below.

Turn latency, Gemma-4-E2B through LocalAI:

| Turn | Old backend (`785322d17`) | New backend (`4c4ca5b18`) |
|---|---|---|
| First turn, model unloaded: load plus prefill of OpenCode's ~11.5K-token prompt | **169 s** | about 60 s |
| Warm turn, one-word answer | **3.2 s** | not remeasured |
| Warm turn, three-sentence answer | **5.6 s** | not remeasured |
| Prefill, measured in the backend, ~11.5K-token prompt | 74.5 tok/s | **226–229 tok/s** cool, 165–208 hot |
| Generation, measured in the backend at 11.5K context | 36.5–38.5 tok/s | **34–39 tok/s** cool, 24–27 hot |

The new first-turn figure is the measured prefill time (50 s) plus the model load; the warm-turn rows depend on generation and the small prefill of each new message, which did not change.

Where a task's time goes. Prompt tokens processed per model call, from the backend log. [`scripts/opencode-cachetest.py`](scripts/opencode-cachetest.py) produced the rows:

| Backend | Task | Model calls | Prompt tokens processed per call | Wall clock |
|---|---|---|---|---|
| Old | A, cache reused | 3 | 11,528 · 170 · 15 | 172 s |
| Old | A, one miss | 6 | 11,529 · **11,833** · 1 · 311 · 264 · 132 | 364 s |
| New | A, one miss | 3 | 11,527 · **11,894** · 15 | 145 s |
| New | A, one miss | 3 | 11,527 · 170 · **11,764** | 147 s |
| New | B, one miss, passed | 15 | 11,562 · 93 · 79 · 190 · **12,229** · 238 · … · 840 | 243 s |
| New | B, one miss, failed | 6 | 11,561 · **11,825** · 79 · 189 · 595 · 171 | 168 s |

A miss reprocesses the whole conversation: about 2.5 minutes on the old backend, about 55 s on the new one. On the new backend all four tasks had exactly one miss. On the old one, across eight tasks, four with `swa_full:true` and four without, five had one such miss: 6 full prefills with `swa_full` against 7 without, so it made no measurable difference. VRAM was 3.04 GB with it and 2.18 GB without. In the head-to-head, Granite-4.0-H-Micro never missed, and nor did one Gemma task against a standalone `llama-server`. Gemma's interleaved sliding-window attention can roll back only to saved checkpoints, which makes it the likely suspect, but `swa_full` should have removed that limit and did not. The cause is open.

## The backend pin

LocalAI's llama-cpp backend used to be built on llama.cpp `785322d17` plus the fixes. It prefilled three times slower than a standalone `llama-server` from the [llama.cpp fork](llama.cpp.md)'s branch tip, whose upstream base is `4c4ca5b18`, three days and 72 commits later. Every default where the backend differs from `llama-server` was tried on the standalone server first, and none slowed it ([localai.md](localai.md#performance)).

Rebuilding the backend on `4c4ca5b18` with the same ten fixes closed the gap. `grpc-server.cpp` compiled against it unchanged. Same ~11.5K-token prompt, same card, each run started cool:

| Server | Prefill | Generation |
|---|---|---|
| LocalAI, backend on `785322d17` plus the fixes (before) | 74.5 tok/s | 36.5–38.5 tok/s |
| LocalAI, backend on `4c4ca5b18` plus the fixes (now) | **226–229 tok/s** | 34–39 tok/s |
| `llama-server`, fork branch tip `821a0ad27`, OpenMP | 224–229 tok/s | 36–39 tok/s |
| `llama-server`, the new backend's tree, no OpenMP | 217–229 tok/s | 34–39 tok/s |

The documented 247 tok/s was at a 10K-token prompt; prefill slows as the context grows, and at 9K the new backend measured 258. The build is in [localai.md](localai.md#the-llamacpp-backend). Which of the 72 upstream commits made the difference was not bisected.

The cost of a first turn is per session, not per turn. The terminal UI (`opencode`) keeps one session across prompts; each `opencode run` starts a new one, and OpenCode puts the project directory early in its system prompt, so a new project reuses only the first ~1,900 cached tokens.

## Heat

The numbers above start from a cool machine. Back-to-back prefills heat the laptop within one or two requests: `pmset -g therm` then shows `CPU_Speed_Limit` at 36–48, and the GPU slows with it. Hot, prefill measured 164–208 tok/s and generation 16–27 tok/s. The machine recovers to full speed within a few minutes of idle. Measure with `pmset -g therm` beside the numbers, and compare runs only at a similar thermal level (`sysctl machdep.xcpm.cpu_thermal_level`).

## Troubleshooting

**Every turn takes a minute.** The cached prompt is being thrown away. Check `agent.title.disable`. Check that nothing else is using another LocalAI model, because loading one evicts Gemma. Even with both right, expect most tasks to pay one extra full prefill ([Verified](#verified)). If every first turn takes 2.5 minutes, the backend is still on the old pin ([The backend pin](#the-backend-pin)).

**Generation drops below 30 tok/s during long sessions.** The laptop is hot ([Heat](#heat)). Check `pmset -g therm`.

**A scripted `opencode run` edits files in the wrong directory.** OpenCode 1.19.5 takes its project from `$PWD`, not from the process's working directory. A script that sets only `cwd` runs the agent in whatever directory launched the script, and the agent treats that directory as writable. Pass `--dir <dir>` and set `PWD=<dir>` as well, as [`opencode-e2e.py`](scripts/opencode-e2e.py) does. That script also checks each session's directory in `~/.local/share/opencode/opencode.db` and aborts on a mismatch. Never launch an agent from `/tmp` or `$HOME`.

**The first request after starting a session fails with `rpc error: code = Unavailable`.** Another model was loaded, and OpenCode's opening requests arrived during the switch. Warm Gemma first with one request: `curl -s 127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"gemma-4-e2b-coding","messages":[{"role":"user","content":"ok"}],"max_tokens":1}'`.

**The model edits code but never runs anything, or stops after one tool call.** Expect this from a 2B model on long or open-ended tasks. Gemma-4-E2B failed the multi-turn scenarios about recovering from a failed tool call ([Coding agents](models.md#coding-agents)). Give it short, concrete tasks, and say which command proves the task is done.

**No response at all.** `curl -s http://127.0.0.1:8085/readyz` should return 200, and `curl -s http://127.0.0.1:8085/v1/models` should list `gemma-4-e2b-coding`.
