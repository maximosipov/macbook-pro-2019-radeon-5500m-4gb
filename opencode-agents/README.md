# OpenCode agents

Four agent definitions that route work to whichever local model measured best at it. Model choice comes from [models.md](../models.md); server setup from [localai.md](../localai.md).

> **Not the setup in use here.** These agents predate the [coding-agent bake-off](../models.md#coding-agents) and need five models installed. The LocalAI install on this machine now keeps one coding model, Gemma-4-E2B ([opencode.md](../opencode.md)), alongside one decision model and one image model. In OpenCode end to end, Granite-4.0-H-Micro, the primary below, passed 2 of 6 tasks against Gemma-4-E2B's 6 of 6. To use these agents, install the other configs and weights from [localai-models/](../localai-models/README.md) first.

| Agent | Model | Why this model |
|---|---|---|
| *(primary)* | Granite-4.0-H-Micro | Most reliable tool calling measured here (BFCL 0.86, 24/24 on the custom set) and the widest window |
| `plan` | Qwen3.5-4B | Best reasoning (MATH-500 0.81, GPQA-Diamond 0.56) |
| `longread` | Gemma-4-E4B | The only model here that retrieves across a long context (MRCR 1.000 against 0.014 for the next best) |
| `review` | Gemma-4-E4B | Best broad knowledge (MMLU-Pro 0.54); shares the primary's need to actually read the diff |
| `scribe` | LFM2.5-2.6B | Fastest and smallest, so the cheapest model to swap in for trivial text |

Phi-4-mini and SmolLM3-3B are deliberately absent: neither emits parseable `tool_calls` on this stack, so an agent built on either silently never calls a tool.

## The constraint that shapes all of this

Only one model fits in 4 GB. LocalAI evicts the loaded model to serve a request for a different one, and **anything in flight during that window fails**. Measured over the API:

| Request pattern spanning a switch | Result |
|---|---|
| Sequential | Reliable — three models in one session, every answer correct |
| Three concurrent | Exactly one fails with `rpc error: code = Unavailable`, in 3 of 3 trials |

Two rules follow, and they are not optional:

1. **Never run two agents on different models at the same time.** Delegate to one subagent, wait for it, then delegate to the next. A parallel fan-out across models will lose requests.
2. **Keep the primary agent on one model for the whole session.** OpenCode opens a session with three parallel requests, so a mid-session switch of the *primary* model is the exact case that breaks.

Each switch also costs a model load — roughly 15–30 s on this card, and nothing stays warm. Delegating a one-line question to a different model can easily cost more than answering it in the model already loaded. Route by phase, not by sentence.

**Warm the primary model before starting a session.** If another model is resident when OpenCode starts, its three opening requests arrive during the eviction and one of them dies. One sequential request first makes the switch safely:

```bash
curl -s http://127.0.0.1:8085/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"granite-4.0-h-micro","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' >/dev/null
opencode
```

## Install

```bash
mkdir -p ~/.config/opencode/agents
cp *.md ~/.config/opencode/agents/
```

Per-project instead: copy into `.opencode/agents/` at the repo root.

Then register the models in `~/.config/opencode/opencode.json`. The keys must match the `name:` field in each [LocalAI model YAML](../localai-models/README.md), and `limit.context` must match that file's `context_size`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "localai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LocalAI (Radeon 5500M)",
      "options": { "baseURL": "http://127.0.0.1:8085/v1" },
      "models": {
        "granite-4.0-h-micro": {
          "name": "Granite-4.0-H-Micro (128K, q4_0 KV)",
          "limit": { "context": 131072, "output": 4096 }
        },
        "qwen3.5-4b": {
          "name": "Qwen3.5-4B (96K, q4_0 KV)",
          "limit": { "context": 98304, "output": 8192 }
        },
        "gemma-4-e4b": {
          "name": "Gemma-4-E4B (64K, q4_0 KV)",
          "limit": { "context": 65536, "output": 8192 }
        },
        "lfm2.5-2.6b": {
          "name": "LFM2.5-2.6B (128K, q4_0 KV)",
          "limit": { "context": 131072, "output": 4096 }
        }
      }
    }
  },
  "model": "localai/granite-4.0-h-micro"
}
```

Only Gemma-4-E4B's context has been verified under a real prompt load; the other three come from the allocation sweep and may not hold when full. See [the caveat](../localai-models/README.md#maximum-context-measured) before trusting the wide ones.

The generous `output` budgets are not padding. Qwen3.5-4B, Gemma-4-E4B and LFM2.5 all emit a reasoning preamble, and a tight budget returns an empty `content` with the answer stranded in `reasoning_content`.

## What to expect

These are 2.6–4B models. Their best instruction-following score here is IFBench strict 0.26, which is the number that predicts how well an agent loop holds together over many steps. Expect them to work on narrow, well-scoped tasks — write this function, review this diff, summarise this file — and to drift on long autonomous chains. That is a property of the models, not of the harness or these prompts.

The `plan` → implement → `review` sequence is worth more here than on a frontier model, because it keeps each step short enough that the model does not have to hold much at once.
