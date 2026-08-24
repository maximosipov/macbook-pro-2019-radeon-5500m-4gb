# AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)

Build-and-run recipes for **ggml**, **llama.cpp**, **stable-diffusion.cpp**, **ollama** and **LocalAI** on a 2019 Intel MacBook Pro, with the model resident in the discrete AMD GPU's VRAM. Every recipe here was built and measured on one machine:

> **MacBook Pro 16" 2019** · Intel Core i9 · **AMD Radeon Pro 5500M, 4 GB (RDNA1)** · Intel UHD 630 (present, deliberately excluded) · macOS 14.8.2 · MoltenVK 1.4.2

## RDNA generations, 1 to 4

Three rows in that table are marked **Silicon** rather than driver bug — `bf16: 0`, `matrix cores: none` and `uma: 0` — which means "no fix exists *for this chip*", not "no fix exists". A later generation supplies the first two outright. This is where this GPU sits in the line, and what each step up the line would actually have bought. AMD splits its GPUs into two families: **RDNA** for graphics parts (this card) and **CDNA** for datacentre compute (MI100 onward); the matrix and bf16 features arrive in the two lines at different times, which is why `bf16` shows up in this document as "CDNA1 or RDNA3".

| Generation | Shipped | Representative parts | `fp16` | `bf16` | `int dot` | `matrix cores` | Native wave |
|---|---|---|---|---|---|---|---|
| **RDNA 1** (`gfx101x`) | 2019 | RX 5700 XT, **Radeon Pro 5500M / 5600M** (Navi 14/12) | ✅ 2× packed | ❌ | ⚠️ Navi 12/14 only | ❌ | 32 |
| **RDNA 2** (`gfx103x`) | 2020 | RX 6000, Radeon Pro W6800X, PS5, Steam Deck | ✅ | ❌ | ✅ | ❌ | 32 |
| **RDNA 3** (`gfx110x`) | 2022 | RX 7900 XTX, Radeon Pro W7900 | ✅ dual-issue | ✅ | ✅ | ✅ WMMA | 32 |
| **RDNA 3.5** (`gfx115x`) | 2024 | Radeon 890M (Strix Point APUs) | ✅ | ✅ | ✅ | ✅ WMMA | 32 |
| **RDNA 4** (`gfx120x`) | 2025 | RX 9070 XT, RX 9060 XT | ✅ | ✅ | ✅ 2× rate | ✅ WMMA + FP8 | 32 |

- **RDNA 1** — the clean break from GCN: work-group processors, and **wave32 as the native compute mode** rather than GCN's wave64 (the architecture is dual-mode, which is exactly the ambiguity MoltenVK got wrong — see [`warp size`](#g-warp)). No ray accelerators, no matrix units, no bf16. Integer dot product is the one split within the generation: Navi 12 and Navi 14 have `V_DOT4_I32_I8`, Navi 10 (RX 5700) does not, so an RX 5700 is *less* capable here than this laptop chip.
- **RDNA 2** — Infinity Cache, much higher clocks, one ray accelerator per CU, and **8-bit/4-bit integer dot across the whole line**, properly reported as accelerated by a native driver. For LLM work this is the generation where quantized matmul stops needing the emulation this repo carries. It is also **the last AMD architecture macOS ever ran**: the Radeon Pro W6800X/W6900X MPX modules for the 2019 Mac Pro are Navi 21. Everything below this line is Linux or Windows only.
- **RDNA 3** — the generation that erases most of this document's "not fixable" column. **WMMA** (wave matrix multiply-accumulate) for fp16/bf16/int8/int4 lands `VK_KHR_cooperative_matrix`, so ggml's coopmat kernels light up and [flash attention](#g-flash-attn) finally has a fast path; **bf16 arithmetic** arrives, so bf16 GGUFs stop producing [NaN](#g-nan); and the vector SIMD dual-issues fp32. Chiplet packaging (one GCD + memory chiplets) is a manufacturing change, not a capability one.
- **RDNA 3.5** — an integrated-graphics refresh only, in the Ryzen AI 300 APUs. Same feature set as RDNA 3 with memory-subsystem work for shared DRAM; note that these report **`uma: 1`**, which puts them in the opposite regime from this card — no PCIe copy, but no dedicated VRAM either.
- **RDNA 4** — monolithic again, third-generation ray tracing, and roughly **double the matrix throughput per CU** versus RDNA 3, with **FP8** (E4M3/E5M2) added to the matrix path and structured sparsity. This is the current consumer generation. FP4/FP6 formats remain a datacentre feature (CDNA 4 / MI350) rather than an RDNA one, which is why the `fp4` field in the banner is `0` on everything in this table.

AMD has said the generation after this one folds RDNA and CDNA back into a single **UDNA** architecture; nothing under that name has shipped as of this writing.

**What this means if you are shopping.** Two of the six landmines on this page are architectural rather than driver bugs, and both are answered by RDNA 3: no matrix cores (landmines 3 and 5 fall back badly) and no bf16. The third architectural limit — 4 GB — is answered by any card with more of it. But none of that is reachable *on a Mac*: macOS tops out at RDNA 2, and MoltenVK would still be translating to Metal, which has no DP4a-style intrinsic and no cooperative-matrix extension regardless of what the silicon underneath supports. **On this hardware the ceiling is the API, not only the chip** — which is the reason the fixes on these branches are keyed to the MoltenVK driver rather than to the GPU.

> GPU capability as ggml reports it: `uma: 0 · fp16: 1 · bf16: 0 · fp4: 0 · int dot: 0 · matrix cores: none · warp size: 32` — [what each field means](#reading-the-ggml-capability-line)

## Reading the ggml capability line

The banner quoted above is the GPU's feature report; each field changes which code paths ggml takes. It is worth reading carefully, because a `0` in it means one of two very different things: *the silicon does not have this*, or *Metal cannot express what the silicon has*. Only the second kind is fixable.

Validated against the hardware and against what MoltenVK reports (device `0x7340` = Navi 14 / gfx1012, PCIe x16, 4 GB dedicated):

| Field | What the driver reports | Verdict |
|---|---|---|
| `uma: 0` | Bus PCIe x16, VRAM (Total) 4 GB, dedicated | **Silicon.** Genuinely not unified memory |
| `fp16: 1` | `shaderFloat16 yes` | **Silicon, present and used.** RDNA1 does packed FP16 at 2× rate |
| `bf16: 0` | `VK_KHR_shader_bfloat16` absent | **Silicon.** AMD added bf16 with CDNA1 and RDNA3; Navi 14 predates both |
| `int dot: 0` | extension **present**, every `*Accelerated` flag **false** | **Software ceiling — now reclaimed, but the line still says 0.** Metal cannot ask for the instruction, but the emulation still wins for batched matmul: enabling it is worth **+68% prefill** |
| `matrix cores: none` | `VK_KHR_cooperative_matrix` absent | **Silicon.** WMMA arrives with RDNA3, MFMA with CDNA |
| `warp size: 32` | 1.4.2 reports `subgroupSize 32 (min 32, max 32)`; **≤ 1.4.1 said 64** | **Driver bug, now fixed.** It was never a ceiling — the driver was misreporting a 32-wide SIMD group as 64, which is what broke every subgroup shader |

Four of the six are honest reports of an older GPU. The other two were not: `int dot: 0` under-reports what the silicon can do, and `warp size: 64` was simply **wrong** on MoltenVK ≤ 1.4.1. Both have since been turned into performance — see landmines 2 and 3.

One trap in reading that line at all: **the banner is printed from the driver's raw report, not from what the backend then decided.** It applies the `integerDotProduct4x8BitPackedSignedAccelerated` gate itself, so a build with the integer-dot fix active still prints `int dot: 0`. `warp size` is the field that does move — it is your check that you are on MoltenVK ≥ 1.4.2 — and prefill throughput (≈59 pp512 on a 4B Q4_K_M rather than ≈35) is how you confirm the integer-dot path is live.

Field-by-field definitions are in the glossary: [`uma`](#g-uma), [`fp16`/`bf16`](#g-fp16), [`int dot`](#g-intdot), [`matrix cores`](#g-coopmat), [`warp size`](#g-warp).

Jargon is unavoidable in a document like this. Every term used below — [`pp512`](#g-pp512), [KV](#g-kv), [flash attention](#g-flash-attn), [`vk::DeviceLostError`](#g-device-lost), [conv-direct](#g-conv-direct), [`matrix cores: none`](#g-coopmat) — is defined, with a reference, in the [**Glossary**](#12-glossary) at the end.

Three of the five needed source patches; those live on a `macbook-pro-2019-radeon-5500m-4gb` branch of the corresponding fork, alongside a self-contained runbook. The other two need only the right build and run flags, which are below.

This page carries what each platform is, why it is here, what had to change and what it measures. **The build, run and reproduce steps for each one live in its own file:**

| Platform | What it gives you | Changes needed |
|---|---|---|
| **[ggml](ggml.md)** · [branch](https://github.com/maximosipov/ggml/tree/macbook-pro-2019-radeon-5500m-4gb) | the compute kernels + `test-backend-ops`, where the GPU bugs get found and fixed | 4 correctness + 2 performance fixes + runbook |
| **[llama.cpp](llama.cpp.md)** · [branch](https://github.com/maximosipov/llama.cpp/tree/macbook-pro-2019-radeon-5500m-4gb) | CLI, OpenAI-compatible server, `llama-bench` | the same 6 + diagnostic knobs + runbook |
| **[ollama](ollama.md)** · [branch](https://github.com/maximosipov/ollama/tree/macbook-pro-2019-radeon-5500m-4gb) | `ollama run`, model library, desktop app + DMG | 5 patches + the llama.cpp fixes + runbook |
| **[stable-diffusion.cpp](stable-diffusion.cpp.md)** · [branch](https://github.com/maximosipov/stable-diffusion.cpp/tree/macbook-pro-2019-radeon-5500m-4gb) | text-to-image | no source changes needed — build/run flags + runbook |
| **[LocalAI](localai.md)** · [branch](https://github.com/maximosipov/LocalAI/tree/macbook-pro-2019-radeon-5500m-4gb) | drop-in OpenAI API server + model gallery, text **and** images, packaged as a menu-bar app | no source changes needed — build/run flags + runbook |

**The five forks are submodules of this repo**, each pinned to its `macbook-pro-2019-radeon-5500m-4gb` branch, so a recursive clone gives you this page *and* the exact trees every measurement on it was taken from:

```bash
git clone --recurse-submodules https://github.com/maximosipov/macbook-pro-2019-radeon-5500m-4gb.git
git submodule update --init            # if you already cloned without --recurse-submodules
git submodule update --init ggml       # or just the one you need — llama.cpp is ~540 MB
git submodule update --remote ggml     # advance a submodule to the current branch tip
```

Each tool page still opens with a standalone `git clone` of its fork, which is what you want if you are building only one of them. If you cloned recursively, that tree is already here on the right branch and you can skip to the cmake.

## Installing the LocalAI app, and pointing a coding agent at it

The rest of this page is how the pieces were built and what they measure. This section is the shortest path from "nothing installed" to "an editor talking to my own GPU": install the menu-bar app, then point [OpenCode](https://opencode.ai) at it. Read [the expectations](#what-to-expect) at the end before you plan your day around it.

### Install the menu-bar app

There is no download — you build the bundle from the [LocalAI branch](#8-localai), which produces `LocalAI M.app` (367 MB) and `LocalAI-M.dmg` (142 MB). See [the LocalAI setup](localai.md#setup) for the server and backends and [the packaging step](localai.md#a-menu-bar-app-and-a-dmg) for what it has to get right. Then:

```bash
open dist/LocalAI-M.dmg                                     # drag "LocalAI M.app" to /Applications
xattr -dr com.apple.quarantine "/Applications/LocalAI M.app"
open "/Applications/LocalAI M.app"
```

The `xattr` line is required, not optional: the bundle is **ad-hoc signed** (no Apple Developer identity was involved, so it is not notarized) and Gatekeeper blocks the first launch otherwise.

It is an [`LSUIElement`](https://developer.apple.com/documentation/bundleresources/information-property-list/lsuielement) app — a menu-bar icon and **no Dock icon**. It owns the server process: starting it with the MoltenVK environment and the [device pin](#g-visible-devices) already set, polling `/readyz` to drive its status line, and shutting it down on Quit. The menu offers Open WebUI, Copy API Base URL, Open Models Folder, Show Log, Restart and Quit.

| What | Where |
|---|---|
| API + WebUI | `http://127.0.0.1:8085` |
| Models and their YAML configs | `~/Library/Application Support/LocalAI/models` |
| Log | `~/Library/Application Support/LocalAI/logs/local-ai.log` |

**Weights are deliberately not bundled** — they are gigabytes. On first run the app seeds the model YAMLs and, if a development checkout is present, symlinks the weights next to them. Otherwise put the GGUF in the models folder yourself; the YAML's `parameters.model` is resolved relative to that directory, and LocalAI rejects an absolute path outside it as an invalid file path, so a **symlink is the way to keep weights elsewhere**:

```bash
ln -s /path/to/granite-4.0-h-micro-Q4_K_M.gguf ~/Library/Application\ Support/LocalAI/models/
```

Check it came up, and that the model actually reached the Radeon rather than the CPU ([how to verify](localai.md#verifying-it-is-actually-on-the-gpu) — LocalAI's own logs will not tell you):

```bash
curl -s http://127.0.0.1:8085/readyz -o /dev/null -w '%{http_code}\n'   # 200
curl -s http://127.0.0.1:8085/v1/models | python3 -m json.tool
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

### Give it a model profile sized for an agent

The default `context_size: 4096` in [the LocalAI run recipe](localai.md#run) is fine for chat and **useless for a coding agent**: OpenCode's system prompt and tool schemas fill most of that before you have typed anything. Add a profile with a bigger window and [quantized KV](#g-kv-quant) to pay for it.

Either Qwen3-4B-Instruct-2507 or Granite-4.0-H-Micro works. **Granite is the better pick**: it scored best on tool calling on this box, and its [Mamba-2](#g-ssm) recurrent state means a wider window costs almost no VRAM. Whichever you choose, configure **only that one** — see [one model per session](#one-model-per-session).

`context_size: 16384` is comfortably above the 11,132 tokens OpenCode's opening prompt needs, and leaves room for a conversation:

```yaml
# ~/Library/Application Support/LocalAI/models/granite-coder.yaml
name: granite-coder
backend: llama-cpp
parameters:
  model: granite-4.0-h-micro-Q4_K_M.gguf
context_size: 16384
f16: true
gpu_layers: 99
flash_attention: "true"     # required for quantized KV — and it runs on the GPU here (landmine 3)
cache_type_k: q4_0
cache_type_v: q4_0          # q4_0 is the best size/quality trade here — see 4.4
```

Restart from the menu-bar icon and the model appears in `/v1/models`. Weights plus 16K of q4_0 cache sit comfortably inside the 4 GB card.

### Configure OpenCode

OpenCode reaches any OpenAI-compatible endpoint through the AI SDK's `openai-compatible` provider. Add a `localai` block to `~/.config/opencode/opencode.json` (global) or `opencode.json` (per project); an existing `provider` map just gains another key:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "localai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LocalAI (Radeon 5500M)",
      "options": { "baseURL": "http://127.0.0.1:8085/v1" },
      "models": {
        "granite-coder": {
          "name": "Granite-4.0-H-Micro (16K, q4_0 KV)",
          "limit": { "context": 16384, "output": 4096 }
        }
      }
    }
  }
}
```

The keys under `models` must match the `name:` in the model YAML — that is the id LocalAI publishes in `/v1/models`. No API key is needed; the endpoint is unauthenticated on loopback. `limit.context` is what stops OpenCode from packing a prompt the server will reject, so keep it equal to `context_size`.

```bash
opencode models | grep localai          # localai/granite-coder
opencode run -m localai/granite-coder "Reply with exactly the word: ready"
```

In the TUI, `/models` switches provider mid-session; `opencode run -m provider/model` sets it for one-shot runs.

<a name="what-to-expect"></a>
### What to expect

Measured on this machine, `opencode run` against `localai/granite-coder`:

| Turn | Wall clock |
|---|---|
| First turn (cold — model load + full prefill of the agent prompt) | **~3 min** |
| Warm turn, one-word answer | **10.1 s** |
| Warm turn, a real question with a few sentences of answer | **13.8 s** / **23.6 s** |

The shape of that is the whole story. **The first turn is expensive and the rest are not**, because the agent's system prompt and tool schemas — thousands of tokens — only have to be [prefilled](#g-prefill) once, and this card prefills at roughly 24–31 tok/s at long context ([llama.cpp run recipes](llama.cpp.md#run)). Keep the server resident and the session alive and it is usable; restart it between every question and you pay three minutes each time.

Where the ~3 minutes goes is measurable, not mysterious. OpenCode's opening request carries **12 tool schemas — 38 KB of JSON** — and renders to **11,132 prompt tokens**, which the server reports itself:

```bash
# ask a 4096-context profile to process it and it tells you the size
$ curl ... -d @opencode-request.json
{"error":{"message":"request (11132 tokens) exceeds the available context size (4096 tokens), try increasing it"}}
```

11,132 tokens at ~25 tok/s is ~7 minutes of [prefill](#g-prefill), once. After that the prefix is cached and turns cost seconds.

<a name="one-model-per-session"></a>
**Load one model per server session.** This is the one rule that matters on a 4 GB card, and getting it wrong produces a failure with no error message:

> A model loaded **after another model has been loaded in the same LocalAI session** returns an *empty completion* on a large prompt — `finish_reason: "stop"`, zero completion tokens, after paying the full prefill. LocalAI logs only `Backend returned empty response, retrying` five times; the client hangs on the retries and prints nothing.

Reproduced four times, and it is none of the things it looks like. It is **not** context overflow (the same 11,132-token prompt succeeds at `context_size: 16384`), **not** streaming (the same request succeeds streamed), **not** the model (Qwen3-4B-2507 and Granite both do it, and both work when loaded first), and **not** a cancelled request left behind. The variable that predicts it is whether another model was loaded first: **3 of 3 clean sessions succeeded, 4 of 4 sessions with a prior model load failed.**

The mechanism is a 4 GB card being asked for more than it has. Two resident models measure **3.47 GB** of the 4.28 GB usable *before* the 11K-token KV cache is allocated:

```bash
$ ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
2295623680   # Granite alone
3467935744   # Granite + Qwen both resident
```

So the *hardware* limit is real — 4 GB genuinely cannot hold two 4B-class models plus a large cache. The *software* defect is that nothing says so: the allocation shortfall surfaces as an empty completion rather than an out-of-memory error, which is what makes it cost hours instead of seconds. That is worth reporting upstream to LocalAI; `--max-active-backends=1` is not a fix (it evicts the previous model, but the eviction then races with the three requests OpenCode opens a session with, and you get `connection refused` instead).

**What to actually do:** point OpenCode at one model and leave it there. If you change models, Quit and relaunch from the menu-bar icon first. A clean session is reliable; a mixed one is not.

Finally, **it is a 3B-class model and it shows.** Asked for a one-liner counting lines across `.md` files it produced `grep -rl --include='*.md' '' . | wc -l` and described it as counting lines — that counts *files*. Asked what llama.cpp's `-ngl` flag does, it answered that it "disables the NVIDIA GPU library", which is the opposite of true. Both came back in under 25 seconds. It is fast enough to be pleasant and wrong often enough that you must read everything it hands you.

One failure mode that looks alarming and isn't: an empty `content` with a non-empty `reasoning` is a [thinking model](#g-thinking) behaving normally over an OpenAI-compatible API, not MoltenVK corruption ([verifying GPU residency](localai.md#verifying-it-is-actually-on-the-gpu)).

## Contents

Above this line: [Reading the ggml capability line](#reading-the-ggml-capability-line), [RDNA generations, 1 to 4](#rdna-generations-1-to-4), and [installing the LocalAI app](#installing-the-localai-app-and-pointing-a-coding-agent-at-it). Below it, the build-and-measure account:

- **1.** [Why any of this is necessary](#1-why-any-of-this-is-necessary)
- **2.** [The six landmines](#2-the-six-landmines)
- **3.** [Shared setup (do this once)](#3-shared-setup-do-this-once)
- **4.** [ggml](#4-ggml) — build and test recipe: [**ggml.md**](ggml.md)
    - **4.1** [What it is](#41-what-it-is)
    - **4.2** [Why](#42-why)
    - **4.3** [What was modified](#43-what-was-modified)
    - **4.4** [Benchmarks](#44-benchmarks)
- **5.** [llama.cpp](#5-llamacpp) — build and run recipe: [**llama.cpp.md**](llama.cpp.md)
    - **5.1** [What it is](#51-what-it-is)
    - **5.2** [Why](#52-why)
    - **5.3** [What was modified](#53-what-was-modified)
    - **5.4** [Models that fit 4 GB](#54-models-that-fit-4-gb)
    - **5.5** [KV cache](#55-kv-cache)
    - **5.6** [A warning about benchmarking this machine](#56-a-warning-about-benchmarking-this-machine)
    - **5.7** [Capability benches](#57-capability-benches)
- **6.** [stable-diffusion.cpp](#6-stable-diffusioncpp) — build and run recipe: [**stable-diffusion.cpp.md**](stable-diffusion.cpp.md)
    - **6.1** [What it is](#61-what-it-is)
    - **6.2** [Why](#62-why)
    - **6.3** [What was modified](#63-what-was-modified)
    - **6.4** [Backend comparison](#64-backend-comparison)
    - **6.5** [Model fit and throughput](#65-model-fit-and-throughput)
- **7.** [ollama](#7-ollama) — build and run recipe: [**ollama.md**](ollama.md)
    - **7.1** [What it is](#71-what-it-is)
    - **7.2** [Why](#72-why)
    - **7.3** [What was modified](#73-what-was-modified)
    - **7.4** [Benchmarks](#74-benchmarks)
- **8.** [LocalAI](#8-localai) — build, run and packaging recipe: [**localai.md**](localai.md)
    - **8.1** [What it is](#81-what-it-is)
    - **8.2** [Why](#82-why)
    - **8.3** [What was modified](#83-what-was-modified)
    - **8.4** [Benchmarks](#84-benchmarks)
- **9.** [Benchmarks and how they were run](#9-benchmarks-and-how-they-were-run)
    - **9.1** [Correctness instruments](#91-correctness-instruments)
    - **9.2** [Speed and fit instruments](#92-speed-and-fit-instruments)
    - **9.3** [Capability benches](#93-capability-benches)
- **10.** [Running unattended: picking for accuracy, not speed](#10-running-unattended-picking-for-accuracy-not-speed)
- **11.** [Negative results worth knowing](#11-negative-results-worth-knowing)
- **12.** [Glossary](#12-glossary)
    - **12.1** [The GPU stack](#121-the-gpu-stack)
    - **12.2** [Capability-line fields](#122-capability-line-fields)
    - **12.3** [Failure modes](#123-failure-modes)
    - **12.4** [LLM inference and serving](#124-llm-inference-and-serving)
    - **12.5** [Benchmarks and metrics](#125-benchmarks-and-metrics)
    - **12.6** [Diffusion](#126-diffusion)

## 1. Why any of this is necessary

macOS on an Intel Mac gives you [Metal](#g-metal). On this machine **Metal is the wrong backend**: the GPU is [discrete, not unified-memory](#g-uma), and ggml's Metal path re-reads model weights across PCIe on every token (~0.8 tok/s for an LLM; for diffusion it doesn't finish at all — the [macOS watchdog](#g-watchdog) kills the command buffer with `kIOAccelCommandBufferCallbackErrorTimeout`).

The path that works is **[Vulkan](#g-vulkan), translated to Metal by [MoltenVK](#g-moltenvk)**. Weights stay resident in the 4 GB of VRAM and generation runs roughly 40× faster (0.8 → ~34 tok/s on a 4B Q4_K_M). But MoltenVK is a translation layer, and it does not honour every guarantee a shader expects from a native Vulkan driver — which is where the bugs below come from. All of them produce **wrong output, not crashes** (see [NaN](#g-nan)), which is why they need to be written down.

## 2. The six landmines

Everything else in this repo is setup detail. These are the findings.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | ~0.8 [tok/s](#g-toks), or a [GPU watchdog timeout](#g-watchdog) | Metal re-reads weights over PCIe on a discrete GPU | `-DGGML_METAL=OFF`, use Vulkan — **and pass the flag explicitly**, ggml auto-enables Metal on macOS even when the wrapper's own `SD_METAL=OFF`/`BUILD_TYPE=vulkan` says otherwise, and Metal then wins device 0 at runtime |
| 2 | Gibberish tokens, or a runner that 500s on allocation | Two causes. The machine's [integrated GPU](#g-dgpu) advertises **32 GiB** of shared host RAM against the Radeon's real 4 GiB, so every "pick the biggest GPU" heuristic picks it. And **MoltenVK up to 1.4.1 reported `subgroupSize` 64 while Metal runs 32-wide SIMD groups on AMD**, so every subgroup reduction spanned the wrong lane count — the real root cause of this *and* landmine 4, and of upstream issue 15846 | [`GGML_VK_VISIBLE_DEVICES=0`](#g-visible-devices) to pin the Radeon, and **MoltenVK ≥ 1.4.2**, which corrects the subgroup size. The branches here gate their workarounds on `driverVersion >= 10402` and keep the safe path below it |
| 3 | `-fa on` looks **~3.5× slower** for generation | Flash attention was **never running on the GPU**. `supports_op` rejects scalar FA unless subgroup shuffle+vote are available, and those are disabled on AMD Macs — so ggml silently scheduled every attention op on the **CPU**. `test-backend-ops support -o FLASH_ATTN_EXT` reported 0 of 5097 cases supported | Use the subgroup-free FA path (on these branches). Flash attention then runs on the GPU and becomes the **faster** option for decode: tg128 **10.2 → 40.4**. It is also what makes [quantized KV](#g-kv-quant) usable at all — `-ctk q4_0 -ctv q4_0` costs ~4% and takes the cache from 32 to 9 KiB/token. All three quantized KV precisions are correct on this branch; q8_0 needed [a further fix](#44-benchmarks) |
| 4 | [Hybrid/state-space](#g-arch) models ([Mamba-2](#g-ssm), [Gated DeltaNet](#g-gdn)) emit token salad on GPU, coherent on CPU | `SSM_SCAN` and `GATED_DELTA_NET` drove shared-memory reductions from `gl_SubgroupInvocationID`, assuming a workgroup is one contiguous [subgroup](#g-subgroup). MoltenVK doesn't guarantee that → **[NaN](#g-nan)**. Same root cause as landmine 2: the driver's wrong subgroup size | Index by `gl_LocalInvocationID.x` instead (two shader files) — on the ggml and llama.cpp branches. The correct path also appeared faster than the broken one, though the two measurements predate the [regime controls](#56-a-warning-about-benchmarking-this-machine) below, so treat that as directional |
| 5 | Diffusion output is full-frame colourful noise | The generic Vulkan [UNet](#g-unet) convolution ([im2col + matmul](#g-conv-direct)) is numerically broken on this RDNA1/MoltenVK stack. Not root-caused; it is a different defect from landmines 2–4 and it lives in a different vendored ggml ([leejet's](#63-what-was-modified)) | [`--diffusion-conv-direct`](#g-conv-direct) — one flag, and it's also **~3× faster** than the broken path |
| 6 | Model "fits" but is unusably slow, or OOMs at the last step | 4 GB is the real constraint. For diffusion, **peak VRAM is at [VAE decode](#g-vae)**, not sampling | [Quantize](#g-quant) (`--type q8_0`), keep the VAE on CPU for SDXL, and check fit against **4278 MB** usable, not 4096 |

Landmines 2, 3 and 4 all trace to one driver bug: **MoltenVK reported a 64-lane subgroup where Metal runs 32**. That single misreport produced the wrong-matmul gibberish, the NaN in the hybrid shaders, and — because ggml's flash-attention support check keys off the subgroup flags that were disabled to work around it — the total absence of GPU flash attention. [MoltenVK 1.4.2](https://github.com/KhronosGroup/MoltenVK) fixes the report; upgrading is the single highest-value thing you can do on this hardware.

## 3. Shared setup (do this once)

```bash
# Homebrew toolchain. vulkan-loader (not libMoltenVK directly) so the driver stays
# swappable at runtime; glslang/shaderc/spirv-headers compile ggml's compute shaders.
brew install cmake libomp \
  vulkan-headers vulkan-loader molten-vk glslang shaderc spirv-headers

# molten-vk must be >= 1.4.2. Up to 1.4.1 it reported a 64-lane subgroup where Metal runs
# 32 on AMD, which silently corrupts every subgroup shader (landmines 2, 3 and 4).
brew list --versions molten-vk

# Runtime: point the Vulkan loader at MoltenVK, and pin the discrete GPU.
export VK_ICD_FILENAMES="$(brew --prefix)/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/molten-vk/lib:$(brew --prefix)/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export GGML_VK_VISIBLE_DEVICES=0
```

Building against the loader rather than linking `libMoltenVK` directly is what makes `VK_ICD_FILENAMES` work as a runtime driver switch — see [ICD](#g-icd). Confirm the enumeration before anything else — `Vulkan0` must be the Radeon:

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Pro 5500M (MoltenVK) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 0 | matrix cores: none
ggml_vulkan: 1 = Intel(R) UHD Graphics 630 (MoltenVK) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 0 | matrix cores: none
```

**`warp size: 32` is your check that MoltenVK is >= 1.4.2.** On 1.4.1 and below this said `64`, which was the driver misreporting Metal's 32-wide SIMD group and is the root cause of landmines 2, 3 and 4. `int dot: 0` stays `0` even with the fix active — see [Reading the ggml capability line](#reading-the-ggml-capability-line).

Unset [`GGML_VK_VISIBLE_DEVICES`](#g-visible-devices) once to confirm the ordering on your machine, then pin whichever index is the Radeon and leave it pinned. Nothing on this page ever runs on the integrated GPU.

---

# 4. [ggml](ggml.md)

> **Build it, test it, read the results:** [**ggml.md**](ggml.md) — [setup](ggml.md#setup), [run / evaluate](ggml.md#run--evaluate), and the full commit-by-commit account of [what was modified](ggml.md#what-was-modified).

### 4.1 What it is

The tensor library everything else on this page is built on — plain C, no dependencies, and *not* an application: there is nothing here to chat with. It provides the pieces a runtime needs:

- **Tensor ops and the compute graph** — matmul, attention (`FLASH_ATTN_EXT`), convolution, normalization, RoPE, and the recurrent ops (`SSM_SCAN`, `GATED_DELTA_NET`) that hybrid models need. A model is a graph of these; a runtime builds it, allocates it, and calls compute.
- **The backend system** — CPU (hand-written SIMD), [Vulkan](#g-vulkan), Metal, CUDA, HIP, SYCL, OpenCL, BLAS, and an RPC backend. A scheduler splits the graph across available devices and falls back to CPU per-op when a backend doesn't implement something, which is why a broken shader degrades quietly instead of failing loudly.
- **[GGUF](#g-ggml) and the quantization types** — the file format plus the k-quant/i-quant implementations ([`Q4_K_M`](#g-quant) and friends), including the dequantize-on-the-fly kernels each backend needs.
- **[`test-backend-ops`](#g-test-backend-ops)** — the differential tester (`test` mode for correctness against CPU, `perf` mode for per-op timing), plus the graph allocator and memory planner.

### 4.2 Why

[ggml](#g-ggml) is where the GPU actually is. Both correctness fixes live in its Vulkan backend, and [`test-backend-ops`](#g-test-backend-ops) — its per-operator differential test against a CPU reference — is the only tool that tells you *which* shader is lying to you. Build this first when something produces wrong output; it turns "the model is broken" into "[`SSM_SCAN`](#g-ssm) returns [NaN](#g-nan)" in about a minute.

### 4.3 What was modified

On the [ggml branch](https://github.com/maximosipov/ggml/tree/macbook-pro-2019-radeon-5500m-4gb), six changes to `ggml-vulkan.cpp` and three `.comp` shaders. Four fix wrong output; two lift a software ceiling.

- **Correctness** — the subgroup matmul path on RDNA1 and older Intel iGPUs; [NaN](#g-nan) from `SSM_SCAN`/`GATED_DELTA_NET` (landmine 4); subgroup *clustered* ops in the `quantize_q8_1` activation quantizer; and two guards keeping q8_0 off paths MoltenVK gets wrong — the multi-column `mul_mat_vecq` variant and the flash-attention MMQ loader. The last two are the same defect, `pack32(i16vec2(...))` repacking from a 16-bit view.
- **Performance** — emulated integer dot for matmul (**+68% [prefill](#g-prefill)**, decode unchanged; see [Reading the ggml capability line](#reading-the-ggml-capability-line) for why `int dot: 0` was never the whole story), flash attention on the GPU (landmine 3), and trusting MoltenVK subgroups from 1.4.2 onwards so three of the workarounds switch themselves off.

All are no-ops on GPUs where [subgroup](#g-subgroup) arithmetic behaves (native Vulkan AMD/NVIDIA, Mesa) — they are keyed on the MoltenVK driver. **Per-commit detail: [ggml.md](ggml.md#what-was-modified).**

### 4.4 Benchmarks

This backend's "benchmark" is a correctness table ([how it's run](#b-tbo)), which is the point. Measured on the current branch tip, MoltenVK 1.4.2:

| Op | Before | After |
|---|---|---|
| `SSM_SCAN` (Vulkan0, AMD) | FAIL — NaN | **OK, 3/3** |
| `GATED_DELTA_NET` (Vulkan0, AMD) | FAIL — ERR ≈ 1.0 | **OK, 36/36** |
| `MUL_MAT` (Vulkan0, AMD) | OK | **OK, 982/982** — including the integer-dot path |
| `MUL_MAT` (integrated GPU, excluded device) | **every case FAIL**, ERR 2.4–11.4 (tol 5e-4) | OK |
| `FLASH_ATTN_EXT` (Vulkan0, AMD) | **0 of 5097 cases supported** — silently on the CPU | **4757 / 4757** |
| full sweep (Vulkan0) | — | **15426 / 15426 — zero failures** |

**The card now passes every case it runs.** It did not until recently: an earlier revision of this page recorded 337 failures, all of them flash attention with `type_K=q8_0` at ERR 0.041–0.092. That is worth keeping, because the cause turned out to be **self-inflicted and instructive**.

q8_0 was correct everywhere *except* flash attention — `GET_ROWS`, `CPY`, `SET_ROWS`, `MUL_MAT`, `MUL_MAT_ID` and `MUL_MAT_VEC_FUSION` all passed it — so the signed 8-bit load was fine and only one loader was wrong. That loader is `k_block_to_shmem` in `flash_attn_mmq_funcs.glsl`, the **only** q8_0 path that repacks through `pack32(i16vec2(...))` from a 16-bit view; every other type packs from `u16vec2`. And it was only reachable because *this branch* enables integer dot and trusts 1.4.2 subgroups — `ggml_vk_fa_scalar_uses_mmq` requires both. Upstream, q8_0 KV took the dequantize path and was correct; the performance patch silently turned it **from correct-but-slower into fast-and-wrong**.

Routing q8_0 back to the dequantize path on MoltenVK restores it, at no cost to the other quantized KV types, which keep the faster MMQ path:

| | before | after |
|---|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | 4420 / 4757 | **4757 / 4757** |
| full sweep | 15089 / 15426 | **15426 / 15426** |

That also identifies the **same construct** behind the other q8_0 defect this branch carries — the multi-column `mul_mat_vecq` case — which was previously logged as "root cause not found". Both are `pack32(i16vec2(...))` under MoltenVK; both are handled by keeping q8_0 off the affected path.

(A further ~3276 cases report `not supported` and are skipped rather than run, which is normal: they are ops or type combinations the Vulkan backend does not implement and hands to the CPU.)

The one thing to know: a [shader](#g-shader) bug here shows up downstream as "this model is broken", three layers away. [`test-backend-ops -b Vulkan0`](ggml.md#run--evaluate) before blaming a model.

---

# 5. [llama.cpp](llama.cpp.md)

> **Build it and run it:** [**llama.cpp.md**](llama.cpp.md) — [setup](llama.cpp.md#setup), [run](llama.cpp.md#run), the [nine commit SHAs](llama.cpp.md#what-was-modified) the other recipes cherry-pick, and [how the benchmarks were reproduced](llama.cpp.md#reproducing-the-benchmarks).

### 5.1 What it is

The reference LLM runtime on top of ggml, and a toolbox rather than a single binary. What ships in `tools/`:

- **`llama-cli`** — one-shot or interactive generation, full sampler control, [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) grammar-constrained output.
- **`llama-server`** — an [OpenAI-compatible](#g-openai-api) HTTP server (`/v1/chat/completions`, `/completion`, `/embedding`, `/tokenize`, `/apply-template`) with a built-in web UI, parallel request slots, context shift, speculative decoding, tool calling and multimodal support.
- **`llama-bench`** — the [pp/tg](#g-pp512) microbenchmark used throughout this page; `llama-batched-bench` for batch-size sweeps.
- **`llama-quantize`** + **`llama-imatrix`** — convert an f16 GGUF to [Q4_K_M](#g-quant) etc., optionally guided by an importance matrix computed over a calibration corpus.
- **`llama-perplexity`** — [perplexity](#g-ppl), KL-divergence against a reference, and multiple-choice evaluation.
- **`llama-mtmd-cli`** — multimodal inference: an LLM plus an [`mmproj`](#g-vlm) vision/audio encoder.
- **Smaller tools** — `llama-tokenize`, `llama-gguf-split`, `llama-export-lora`, `llama-cvector-generator`, `llama-tts`, and `llama-rpc-server` for spreading layers across machines.
- **Runtime features that matter on 4 GB** — [layer offload](#g-ngl), [KV cache quantization](#g-kv-quant), LoRA adapters at load time, and the [flash-attention](#g-flash-attn) toggle this page keeps warning you about.

### 5.2 Why

Everything else here (ollama, LocalAI) is a wrapper around this code, so measure here first — and [`llama-bench`](#g-pp512) is the only honest way to compare models on this box.

### 5.3 What was modified

On the [llama.cpp branch](https://github.com/maximosipov/llama.cpp/tree/macbook-pro-2019-radeon-5500m-4gb): the same six Vulkan changes as [4.3](#43-what-was-modified) (llama.cpp vendors a synced copy of ggml), plus `vulkan: add diagnostic env knobs` — `GGML_VK_FORCE_ARCH`, `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ`, `GGML_VK_FORCE_INTEGER_DOT`, opt-in, used to isolate them.

Nine commits in total, and the SHAs matter: the [ollama](ollama.md#setup) and [LocalAI](localai.md#setup) recipes cherry-pick exactly this list onto their own pinned llama.cpp commits, because neither can build from this branch's tip. **The SHAs are in [llama.cpp.md](llama.cpp.md#what-was-modified).**

### 5.4 Models that fit 4 GB

All [Q4_K_M](#g-quant), `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3` ([layer offload](#g-ngl), [flash attention](#g-flash-attn)), post-fix, **thermally gated before each model**:

| Model | [Architecture](#g-arch) | [pp512](#g-pp512) tok/s | [tg128](#g-pp512) tok/s | Fit |
|---|---|---|---|---|
| Gemma-4-E2B | gemma4 [PLE](#g-arch) | **142.6 ± 0.3** | **50.7 ± 0.2** | 1.45 GB (PLE stays in host RAM) |
| Granite-4.1-3B | dense | 78.9 ± 15.2 | 46.1 ± 0.1 | 2.24 GB, KV-bound |
| Qwen3-4B-Instruct-2507 | dense [GQA](#g-arch) | 59.4 ± 0.6 | 40.5 ± 0.1 | 2.82 GB, worst [KV](#g-kv) (144 KiB/tok) |
| Granite-4.0-H-Micro | [Mamba-2](#g-ssm) hybrid | 83.6 ± 17.1 | 34.9 ± 11.2 | 1.95 GB, ~2.1 GB headroom |
| Qwen3.5-4B | [Gated DeltaNet](#g-gdn) | 59.5 ± 0.6 | 33.6 ± 0.1 | 3.05 GB, ~1.0 GB headroom |
| Gemma-4-E4B | gemma4 [PLE](#g-arch) | 55.1 ± 0.5 | 30.9 ± 0.0 | 3.3 GB, runs fine |

Measured with `-fa on` on MoltenVK 1.4.2 with the fixes on the [ggml/llama.cpp branches](#4-ggml). **Every one of these numbers roughly doubled or better** against the previous edition of this table — prefill by 2.2-2.5x, decode by 2.6-3.9x — for three cumulative reasons: MoltenVK 1.4.2 corrected the AMD subgroup size (landmine 2), flash attention now runs on the GPU instead of falling back to the CPU (landmine 3), and integer dot is enabled for batched matmul. Gemma-4-E4B, previously recorded here as losing the device and elsewhere as "impractically slow" at 3.9 tg128, now runs at **30.9**. Two rows carry large error bars (Granite-4.0-H-Micro tg128, Granite-4.1-3B pp512); treat their exact values as indicative and see [5.6](#56-a-warning-about-benchmarking-this-machine).

> **Measured under the regime in [5.6](#56-a-warning-about-benchmarking-this-machine)**, one `llama-bench` process per model with a cooldown between runs. The GPU idles at 10 MHz, so a cold `tg128` still partly measures the clock ramp — read these as a floor, treat gaps under ~20% as noise, and prefer a sustained serving measurement for "what will I actually get".

The commands behind this table are in [llama.cpp.md](llama.cpp.md#reproducing-the-benchmarks).

### 5.5 KV cache

Measured on Qwen3.5-4B — the number that decides your [context length](#g-ctx):

| [KV precision](#g-kv-quant) | bytes/token | Max context with zero [CPU spill](#g-spill) | Numerically safe here? |
|---|---|---|---|
| f16 | 32.0 KiB | ~41K | ✅ |
| q8_0 | 17.0 KiB | ~82K | ✅ — but it failed 337 op-test cases until the [flash-attention MMQ fix](#44-benchmarks); check your build |
| q4_0 | 9.0 KiB | **128K** | ✅ |

Quantized KV requires [flash attention](#g-flash-attn). That used to be the catch, because FA fell back to the CPU (landmine 3) and cost ~3.5× on decode. With FA running on the GPU it costs about **4%** — `-ctk q4_0 -ctv q4_0` measures **36.7 tg128** against 38.1 for f16 — so the long-context choice is no longer a trade at all.

**On q8_0:** it was the one precision the op tester rejected on this driver, and it is fixed on this branch rather than merely avoided — see [4.4](#44-benchmarks). If you are on an older build of these forks, or upstream with integer dot forced on, prefer f16 or q4_0.

### 5.6 A warning about benchmarking this machine

This is a 2019 laptop and it will lie to you if you let it. The same binary, model and flags produced **10.5, 24.9 and 33.9 tok/s** across three sessions. None of those runs was buggy; they were different regimes. Three effects, in order of size:

- **GPU clock state — dominant, ~2.4×.** `ioreg` reports the Radeon idling at **10 MHz** core clock, and `llama-bench`'s single warmup pass doesn't spin it up. The decisive test, same binary and flags back-to-back:

  | GPU | CPU | tg128 |
  |---|---|---|
  | warm (straight off a serving workload) | **throttled to 30** | **24.92 ± 8.10** |
  | cold (after cooling to 100) | unthrottled | 10.49 ± 0.11 |

  The run on the *throttled* CPU is 2.4× faster. Cooling the machine to avoid CPU throttling **parks the GPU**, so the intuitive benchmarking hygiene is backwards here.
- **Position within a multi-config run.** `llama-bench -fa 0,1` gives the first config a different machine than the second. That alone produced an apparent "3.9× flash-attention penalty" that a [counterbalanced](#g-counterbalance) experiment puts at **2.27×**.
- **[CPU thermal throttling](#g-throttle).** `CPU_Speed_Limit` falls to **20–36** within ~2 minutes of load. Real — but smaller than the GPU effect and pointing the other way.

So: **warm the GPU with a discarded run, use one process per config, and alternate the order of what you're comparing.** Report the per-round values, not just the mean — the spread is what tells you whether a difference is real. Never difference two tables produced on different days; answer config questions with an A/B.

Two caveats to that advice, both learned the hard way. A warm-up pass is **not free**: adding one triggered [`vk::DeviceLostError`](#g-device-lost) on Qwen3.5-4B, so a more representative number also means a less likely finish. And watch for **orphaned benchmark processes** — one outlived a `pkill` of its parent script and held the machine at `CPU_Speed_Limit=36` while the next run sat in a cooldown loop, waiting for a machine that another process was busy heating.

Honestly, for "what will I actually get?", **measure sustained serving, not micro-benchmarks.** LocalAI serving Qwen3-4B-2507 returned 27.1 / 27.9 / 28.4 tok/s across three runs — tighter than any `llama-bench` figure on this box, and it's the number you actually feel.

### 5.7 Capability benches

Sampled, over `llama-server`'s OpenAI endpoint. These are **relative rankings on this hardware under one protocol — not leaderboard-comparable**; sample sizes are small and quantization/subset/protocol all differ from the published runs. What each one measures and exactly how it was run: [Benchmarks and how they were run](#9-benchmarks-and-how-they-were-run).

| Bench | Measures | Qwen3.5-4B | Granite-4.0-H-Micro | Qwen3-4B-2507 | Frontier reference |
|---|---|---|---|---|---|
| [BFCL-V4 AST](#g-bfcl) (n=80) | tool/function calling | 0.750 | 0.863 | **0.875** | 0.775 ᵃ |
| [MMLU-Pro](#g-mmlu) (n=28, 0-shot [CoT](#g-thinking)) | 10-choice reasoning | **0.464** | 0.393 | 0.429 | 0.896 ᵇ |
| [LongBench-v2](#g-longbench) (n=3, ~12K tok) | long-document QA | timeout | 0/3 | timeout | 0.644 ᶜ |
| [MRCR](#g-mrcr) v2 (n=2, ~19K tok) | multi-round coref recall | 0.00 | 0.014 | timeout | 0.915 ᵈ |
| custom tool set (n=24) | tool calls incl. negatives | 23/24 | **24/24** | 22/24 | — (local set) |

**How to read the scores.** BFCL AST, MMLU-Pro and LongBench-v2 are **accuracy in 0–1** (fraction of cases answered correctly; MMLU-Pro's chance floor is 0.1 with ten options, LongBench-v2's is 0.25 with four). MRCR is **mean similarity in 0–1** against the reference message, so partial credit is possible and ~0 means the model never found the right message. **`timeout` is not a score** — the request exceeded the 400 s client limit and returned nothing, so the model was never graded. It is a throughput failure of this hardware at that context length, not evidence that the model would answer wrongly. `0/3` means it did finish and got none right.

**The frontier column** is the best *published, full-benchmark* score at the time of writing, for scale only — those runs use unquantized weights, the complete test set, and each vendor's own harness. It is not comparable with the columns to its left, which are tiny samples of Q4_K_M models under one local protocol; treat it as the ceiling of the field, not as a target this hardware was measured against. Leaderboards move — re-check before quoting.

ᵃ Claude Opus 4.5 (FC), **overall BFCL-V4** — [gorilla leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), July 2026. Metric mismatch worth naming: overall V4 includes multi-turn, agentic and memory categories that drag scores down, while our column is the single-turn AST subset only. The frontier number on AST alone would be higher.
ᵇ Qwen3.7 Max, full MMLU-Pro — [llm-stats](https://llm-stats.com/benchmarks/mmlu-pro), updated 2 Aug 2026.
ᶜ Claude Opus 4.5, full LongBench-v2 (8K–2M token contexts) — [llm-stats](https://llm-stats.com/benchmarks/longbench-v2), June 2026. For calibration: human experts score **0.537** on this set under a 15-minute limit.
ᵈ GPT-5.6 Sol, MRCR v2 **8-needle at 128K** — [llm-stats](https://llm-stats.com/benchmarks/mrcr-v2-(8-needle)), July 2026. Ours is ~19K context with fewer needles, i.e. a much easier setting that these models still could not complete.

**These capability scores are stale.** They were taken before flash attention ran on the GPU and before the MoltenVK 1.4.2 subgroup fix, so the throughput underlying them was roughly half of today's. In particular the timeouts are **not** evidence of a hardware wall: they came from a 400 s client limit in the bench scripts. Re-measured since, an 8192-token prefill completes in 4-6 minutes (pp8192 31.5 with f16 KV, 24.3 with q4_0 KV) and neither configuration times out or loses the device. The relative ranking between the three models is probably still indicative; the absolute numbers are not.

---

# 6. [stable-diffusion.cpp](stable-diffusion.cpp.md)

> **Build it and run it:** [**stable-diffusion.cpp.md**](stable-diffusion.cpp.md) — [setup](stable-diffusion.cpp.md#setup), [txt2img command lines](stable-diffusion.cpp.md#run) for SD-Turbo / SDXL-Turbo / SD 1.5, and [how to reproduce](stable-diffusion.cpp.md#reproducing) the numbers below.

### 6.1 What it is

The diffusion counterpart to llama.cpp — image and video generation on the same [ggml](#g-ggml) backends, one static binary, no Python:

- **Modes** — txt2img, img2img, inpainting, instruction-based image editing, ESRGAN upscaling, and a convert mode that rewrites weights to GGUF or safetensors.
- **Model families** — SD 1.x/2.x, SDXL and their [Turbo](#g-turbo) distillations, SD3/3.5, Flux.1/2, Qwen-Image, Chroma, Z-Image; edit models (Flux Kontext, Qwen-Image-Edit); video (Wan 2.1/2.2, LTX, HunyuanVideo). On 4 GB the practical subset is the first group — see the fit table below.
- **Conditioning and adapters** — [LoRA](#g-lora), ControlNet (SD 1.5), IP-Adapter, PhotoMaker, LCM/LCM-LoRA, ADetailer.
- **Memory levers** — on-the-fly quantization with `--type`, [TAESD](https://github.com/madebyollin/taesd) or CPU-side [VAE decode](#g-vae), and VAE tiling.
- **Two binaries** — `sd-cli` for one-shot generation, `sd-server` for an HTTP service that keeps the model resident (which matters here: model load and quantization dominate a 4-step turbo run). The same models are also servable over an OpenAI-compatible images API — see [LocalAI](#8-localai).

### 6.2 Why

Image generation on the same ggml/Vulkan stack. It's the one workload where the Metal path doesn't merely underperform — it doesn't complete at all.

### 6.3 What was modified

**No source changes.** stable-diffusion.cpp builds clean for this hardware. Everything needed is build flags and run flags — which is exactly why they need writing down, because two of them are non-obvious and both produce silently wrong output. The branch carries [a runbook](https://github.com/maximosipov/stable-diffusion.cpp/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) and nothing else.

Worth knowing: this project vendors **[leejet/ggml](https://github.com/leejet/ggml)**, a separate lineage from the [ggml fork](#4-ggml) above, so the Vulkan fixes from chapters 4 and 5 are *not* in this build. They have not been needed — on MoltenVK 1.4.2 the subgroup misreport behind most of them is corrected in the driver, and the diffusion-specific defect is handled by `--diffusion-conv-direct`. Whether the integer-dot enablement would also speed up diffusion here is **untested**.

The two flags that matter, and why each is load-bearing: [stable-diffusion.cpp.md](stable-diffusion.cpp.md#setup).

### 6.4 Backend comparison

By [CPU-reference diff](#b-imgdiff); SD 1.5 fp16, 512×512, seed 42, identical prompt:

| Backend | Fits? | Correct? | Speed |
|---|---|---|---|
| Metal | targets the AMD, 4278 MB set | ❌ **[GPU watchdog timeout](#g-watchdog)** | — (shader library alone takes 40–1000 s to load) |
| Vulkan, no extra flags | ✅ ~2.6 GB | ❌ **colourful noise** | ~11 s/step |
| **Vulkan + `--diffusion-conv-direct`** | ✅ ~2.6 GB | ✅ **matches CPU** | **~3.4 s/step** |
| CPU (ground truth) | n/a | ✅ | ~24 s/step |

"Matches CPU" means verified against the CPU reference image, not "looks like an image". GPU is ~7× CPU, and the correct GPU path is ~3× faster than the broken one.

### 6.5 Model fit and throughput

| Model | Load config | VRAM | s/step | Notes |
|---|---|---|---|---|
| SD-Turbo | `--type q8_0` | **2049 MB** | **~1.7** | comfortable; 4-step image in ~7 s compute |
| SD 1.5 | fp16 | 2035 MB | ~3.4 | baseline, 20 steps |
| SDXL-Turbo | `--type q8_0 --vae-on-cpu` | **3836 MB** | ~9.3 | fits, but tight against 4278 MB |

Recommendation: **[SD-Turbo](#g-turbo) q8_0** for iteration, **SDXL-Turbo q8_0 + CPU VAE** for hero shots (clearly more photoreal, ~5× slower per step, and at 4-step/[cfg-1](#g-diffusion) it drops prompt detail more often), **SD 1.5** when you need the [LoRA/ControlNet](#g-lora) ecosystem.

Other landmines on this card: **[bf16](#g-fp16) weights produce [NaN](#g-nan)** (this GPU is `bf16: 0`) — use fp16 or GGUF quants; **`--diffusion-fa` only works on NVIDIA [coopmat2](#g-coopmat)** — leave it off; peak VRAM is at **[VAE decode](#g-vae)**, so `--vae-on-cpu` is both the SDXL fp16-NaN fix and the VRAM lever.

Re-verified on a fresh build of the branch: SD-Turbo q8_0, 4 steps, 512×512, seed 42, `--diffusion-conv-direct` — a recognisable apple, **2049.08 MB** of parameters resident (text encoders 500.5, UNet 1388.9, VAE 159.7), `generate_image` complete in **44.7 s**, of which **27.1 s was [VAE decode](#g-vae)**. That last split is worth internalising: on a 4-step turbo run more time goes into decoding the latent than into sampling it, and it is also where peak VRAM lands.

---

# 7. [ollama](ollama.md)

> **Build it and run it:** [**ollama.md**](ollama.md) — the [5 patches in detail](ollama.md#what-was-modified), the [fake Vulkan SDK setup](ollama.md#setup) that makes its nested CMake find MoltenVK, and [how to run and reproduce](ollama.md#run).

### 7.1 What it is

A model manager and always-on local server wrapping llama.cpp — closer to Docker for models than to a chat program:

- **Model lifecycle** — `ollama pull/run/create/list/ps/rm/push`, a public registry, and content-addressed blob storage shared between models.
- **`Modelfile`** — a small declarative recipe (`FROM` a base model or a local GGUF, plus `SYSTEM`, `TEMPLATE`, `PARAMETER`, adapters) that turns weights into a named, reusable model.
- **A server that manages memory for you** — models load on demand, unload after an idle `keep_alive`, and the scheduler decides the CPU/GPU split ([`num_gpu`](#g-ngl)); `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` bound concurrency.
- **Two APIs** — its own REST API (`/api/generate`, `/api/chat`, `/api/embed`, `/api/create`, `/api/ps`, …) and an [OpenAI-compatible](#g-openai-api) `/v1` surface, with tool calling, JSON-schema structured outputs, embeddings and [vision](#g-vlm) models.
- **A desktop app** (macOS/Windows) with a chat UI and settings, which is what the DMG and the extra patches on this branch are for.

### 7.2 Why

It is the friendliest of the five, and the one most likely to be someone's first install. Official macOS Ollama ships **no Vulkan runner**, so on this machine the Radeon shows up as `id=cpu` ([ollama#13127](https://github.com/ollama/ollama/issues/13127)) and you get CPU speeds. This branch builds the runner that upstream doesn't ship.

### 7.3 What was modified

On the [ollama branch](https://github.com/maximosipov/ollama/tree/macbook-pro-2019-radeon-5500m-4gb), 5 patches, on top of the [nine llama.cpp fix commits](llama.cpp.md#what-was-modified) cherry-picked onto ollama's pinned llama.cpp:

- **`vulkan-visible-device-index`** — the actual cause of the gibberish (landmine 2): ollama's device list counts CPU as index 0, so it passed a CPU-inclusive index where ggml expects a **Vulkan-relative** one, asked for the Radeon and got the integrated GPU.
- **`image-min-tokens-env`** — `OLLAMA_IMAGE_MIN_TOKENS`, against a hardcoded 1024 [image tokens](#g-vlm) that collapse sustained throughput on a thermally limited laptop.
- **`gpu-percent-slider`**, **`bundle-vulkan-env`**, **`raw-protocol-tab`** — a global CPU/GPU split default plus a desktop slider, a packaged `.app` that sets the MoltenVK environment itself, and a debug tab showing the literal `/api/chat` traffic.

**What each one fixes, in full: [ollama.md](ollama.md#what-was-modified).**

### 7.4 Benchmarks

Via the [text battery](#b-battery) and the [vision eval](#b-vision):

| Workload | Result |
|---|---|
| Qwen3-1.7B text generation | **64–67 tok/s**, 100% GPU |
| Qwen3-VL-4B vision, `OLLAMA_IMAGE_MIN_TOKENS=512` | **32.0 tok/s**, 4/4 grounded |
| Qwen3-VL-4B vision, upstream default (1024) | 8.3–15.3 tok/s, decays across requests |
| VRAM, Qwen3-VL-4B | ~3.2–3.6 GiB of 4080 MiB, no CPU spill |

The 64 tok/s is a single run on the rebuilt binary with `OLLAMA_FLASH_ATTENTION=1`; the 67 is the earlier figure from the battery. The vision rows predate the rebuild and were taken with flash attention off.

Device choice, same model (Qwen3-1.7B), same build — the reason for the pin:

| Device | Generation |
|---|---|
| AMD Radeon Pro 5500M | **71 tok/s** |
| CPU | 24 tok/s |

CPU is the only fallback worth having. The integrated GPU is excluded by the pin and is not a third option. [How to reproduce, and the per-request CPU/GPU split](ollama.md#reproducing).

---

# 8. [LocalAI](localai.md)

> **Build it, run it, package it:** [**localai.md**](localai.md) — [setup](localai.md#setup) (composing a Vulkan `grpc-server`), [run](localai.md#run), [verifying it is actually on the GPU](localai.md#verifying-it-is-actually-on-the-gpu), [text-to-image on the same GPU](localai.md#text-to-image-on-the-same-gpu) and [the menu-bar app and DMG](localai.md#a-menu-bar-app-and-a-dmg).

### 8.1 What it is

An OpenAI-API-compatible server that federates *many* inference backends behind one endpoint, plus the management layer around them:

- **API surface well beyond chat** — `/v1/chat/completions` and completions, embeddings, image generation, audio transcription and text-to-speech, a realtime speech-to-speech API, vision, reranking, and object detection.
- **Backends as separate processes** — llama.cpp, vLLM, transformers, diffusers, whisper, piper and others, each a [gRPC server](#g-grpc) LocalAI spawns and supervises. That indirection is the whole story of the build.
- **Galleries** — a model gallery (pull from Hugging Face, or a curated index) and a backend gallery that installs backends on the fly as [OCI images](#g-oci), plus a web UI over both.
- **Agent-side features** — constrained grammars, tool/function calling, MCP support and built-in agents.
- **Scale-out** — P2P and distributed mode, irrelevant on one laptop but it explains the amount of machinery in the repo.

On this machine only the llama.cpp backend is worth building: the rest are Python/CUDA-oriented and won't find an accelerator here.

### 8.2 Why

It is the only one of the five that gives you a drop-in OpenAI endpoint *plus* model management without being a desktop app — point any OpenAI-SDK client at it and it works.

### 8.3 What was modified

No source changes are needed, but LocalAI's macOS build path targets Apple Silicon + Metal, so getting a Vulkan backend requires composing two pieces yourself: its llama backend is a **[separate gRPC server process](#g-grpc)** built from a pinned llama.cpp, and you have to substitute a fixed one — that pin with the [nine Vulkan fix commits](llama.cpp.md#what-was-modified) cherry-picked onto it, built with `-DGGML_METAL=OFF` forced through the environment, and spawned by a wrapper script that carries the MoltenVK environment into the child process.

*Verified with LocalAI `ecdb321`, its pinned llama.cpp `1cbfd1988`, Homebrew gRPC 1.83.0 / protobuf 35.1 / abseil 20260107.1, Go 1.26.5, Node 24, CMake 3.27.0, MoltenVK 1.4.2, macOS 14.8.2.*

Every step, including the two Makefile traps that silently give you a Metal build: **[localai.md](localai.md#setup)**.

### 8.4 Benchmarks

Qwen3-4B-Instruct-2507 Q4_K_M, ctx 4096, f16 KV, `gpu_layers: 99` ([battery](#b-battery), [sustained serving](#b-serving), [VRAM probe](#b-vram)):

| Measure | FA off (original) | FA on (rebuilt backend) |
|---|---|---|
| Correctness battery over [`/v1/chat/completions`](#g-openai-api) | **4/4** (arithmetic, factual, sequence, string reversal) | `20+34` → `54` |
| Generation, 128 tokens, 3 runs | **27.1 / 27.9 / 28.4 tok/s** | 21.3 / **30.9 / 30.7** tok/s |
| VRAM resident (IOKit, during generation) | 3.35–3.50 GiB of 4080 MiB | **3.06 GiB** |
| GPU utilisation during generation | 82% | — |
| Model load → first token | ~0.9 s for a short prompt | — |

The FA-off column is the original careful measurement and is the one to trust for spread: those three runs are the **tightest numbers in this whole README**, which is why the benchmarking section above recommends sustained serving over micro-benchmarks. The FA-on column comes from the rebuilt backend but was taken **while a compile was saturating all 16 cores** — the first run is cold and contended, and the pair that follows should be read as "at least as fast as before", not as a measured +10%.

Granite-4.0-H-Micro, the [accuracy pick](#10-running-unattended-picking-for-accuracy-not-speed), served from the same stack at ctx 8192:

| Measure | Result |
|---|---|
| Generation, 128 tokens, 3 warm runs | **27.8 / 32.5 / 32.3 tok/s** |
| VRAM resident during generation | **2.77 GB** of 4080 MiB, no CPU spill |

It is both the most reliable model measured here *and* the fastest one served — the reliability cost nothing.

**Images from the same server:** SD-Turbo q8_0, 4 steps, 512×512 — ~2 GB resident, **3.04 GB peak** (peak is [VAE decode](#g-vae), not sampling), ~50 s for the first call including model load and on-the-fly quantization, ~44 s after; SD 1.5 fp16 at 20 steps also works (88 s). Build and model config: [text-to-image on the same GPU](localai.md#text-to-image-on-the-same-gpu). [How to reproduce these numbers](localai.md#reproduce).

---

## 9. Benchmarks and how they were run

Every number on this page comes from one of the instruments below. They fall into three groups: instruments that check the GPU is *correct*, instruments that measure *speed and fit*, and sampled *capability* benches that rank models against each other. Read the [benchmarking warning](#56-a-warning-about-benchmarking-this-machine) before comparing any two numbers.

### 9.1 Correctness instruments

- <a name="b-tbo"></a>**[`test-backend-ops`](#g-test-backend-ops)** — runs every ggml operator on a backend with random inputs and diffs the result against the CPU implementation, per shape and per data type, against a tolerance (5e-4 for matmul). `test` mode reports OK/FAIL per case, `perf` mode reports µs per case. **Score:** a count of passing cases (`36/36`, or `15089/15426` over the current full sweep); a FAIL is a real bug, since the comparison is against the same computation on the CPU. Two reading rules, both learned the hard way here: a case reported `not supported` is **skipped, not passed** — a backend where every case is skipped still prints a green `OK`, which is how the flash-attention CPU fallback hid for months — and the case totals move with upstream, so compare failures against a same-day baseline rather than against a count quoted in an older document. This is the only instrument here that localizes a bug to a specific shader; everything else just tells you something is wrong.
- <a name="b-battery"></a>**Text correctness battery** — four deterministic prompts (arithmetic, a factual recall question, a sequence continuation, and string reversal) at `temperature 0`, scored on the final answer. It also computes a **4-gram repetition ratio** over the output, which is what actually catches [MoltenVK garbage](#g-nan): broken GPU output is usually fluent-looking loops rather than wrong answers. **Score:** `n/4` correct answers, with the repetition ratio as a separate pass/fail gate. It measures *coherence, not capability* — 4/4 means the model is running correctly, not that it is any good.
- <a name="b-vision"></a>**Vision eval** — four images with known ground truth (a sign with readable text, a counting scene, a colour/shape question, a chart) sent as base64 over the chat API, scored on the final answer after stripping any reasoning block. **Score:** `n/4` correct final answers. Confirms the [`mmproj`](#g-vlm) path works end to end, not just that the LLM loads.
- <a name="b-imgdiff"></a>**Diffusion CPU-reference diff** — the same prompt, seed, sampler, step count and resolution generated on CPU and on the GPU, then compared visually. **Score:** pass/fail by eye against the reference. "Matches CPU" in the diffusion tables means this comparison passed, which is a much stronger claim than "produced an image" — the [colourful-noise failure](#g-conv-direct) also produces an image.

### 9.2 Speed and fit instruments

- <a name="b-llamabench"></a>**[`llama-bench`](#g-pp512) (pp512 / tg128)** — the standard llama.cpp microbenchmark: `pp512` times [prefill](#g-prefill) of a 512-token prompt, `tg128` times generation of 128 tokens, both in tokens/second, repeated `-r 3` and reported as mean ± [sd](#g-sd). **Score:** tokens/second, higher is better; the ± is what tells you whether two numbers actually differ. Run here **one config per process**, thermally gated before each model. Treat its absolute numbers as cold-GPU lower bounds and its cross-config comparisons as suspect unless they came from a [counterbalanced A/B](#g-counterbalance).
- <a name="b-serving"></a>**Sustained serving throughput** — a 128-token completion requested over the [OpenAI endpoint](#g-openai-api) against an already-warm server, timed end to end from the client, repeated three times. **Score:** tokens/second including server-side queueing and detokenization, so it is slightly pessimistic versus `llama-bench` in principle — and in practice higher, because the GPU is warm. This is the number a user actually experiences, and on this box it is by far the most reproducible (spread under 5%).
- <a name="b-vram"></a>**VRAM residency and fit probes** — three complementary checks: the llama.cpp/sd.cpp **load log** (which reports per-buffer allocation and reveals [CPU spill](#g-spill) as a CPU buffer appearing), **[`ioreg`](#g-ioreg)** `inUseVidMemoryBytes` on the AMD accelerator node during generation (the only cross-runtime ground truth, and the one that works when a runtime lies), and a **context sweep** that raises `--ctx-size` until spill appears, which is what produced the KV table. **Score:** megabytes resident and a yes/no on spill — there is no partial credit, a model either fits or quietly halves its speed.
- <a name="b-sdstep"></a>**Diffusion s/step and peak VRAM** — sd.cpp's own per-step timing from its log, plus peak VRAM sampled during [VAE decode](#g-vae) rather than at load, because that is where a 4 GB run dies. **Score:** seconds per sampling step (lower is better) and peak megabytes. Wall-clock is reported separately since model load and on-the-fly quantization dominate a 4-step turbo generation.

### 9.3 Capability benches

All four were run over `llama-server`'s OpenAI endpoint at `temperature 0`, on Q4_K_M weights, with small samples chosen to fit this machine's patience. **They rank these models against each other under one protocol; they are not comparable to published leaderboard numbers** — different subset, quantization, sample size and prompt format.

One convention applies to all of them: a request that exceeds the **400 s client timeout** is recorded as `timeout`, which means *no answer was produced and nothing was graded*. It is a statement about this hardware at that context length, not about the model's ability — a timeout and a wrong answer are different failures and are never averaged together.

- <a name="b-bfcl"></a>**[BFCL-V4](#g-bfcl) AST (n=80)** — Berkeley Function-Calling Leaderboard. Each case is a user request plus one or more function schemas; the model is asked for a tool call through the OpenAI tools API and the emitted call is parsed and compared **structurally** (function name, argument names, argument types and values) against the reference answers — the AST check, no execution involved. **Score:** fraction of cases whose call matches, 0–1; a wrong function name, a missing required argument or a malformed JSON body all score 0 for that case. Run over the single-turn categories only (`simple`, `multiple`, `parallel`, `parallel_multiple`), so it says nothing about multi-turn agent behaviour or error recovery.
- <a name="b-mmlu"></a>**[MMLU-Pro](#g-mmlu) (n=28, 0-shot CoT)** — ten-option multiple choice across academic and professional subjects, harder and less guessable than MMLU. The model is told to reason step by step and finish with `The answer is (X)`; grading extracts that letter with the official-style regex. **Score:** fraction correct, 0–1, with a **0.1 chance floor** (ten options) — so 0.393 is meaningfully above guessing, and a model that rambles past its token budget without emitting `The answer is (X)` scores 0 for that case regardless of whether its reasoning was right. Sampled stratified across categories; at n=28 the confidence interval is wide, so treat gaps under ~0.1 as a tie.
- <a name="b-longbench"></a>**[LongBench-v2](#g-longbench) (n=3, ~12K tokens)** — four-choice questions over long documents, answered with a bare letter. **Score:** fraction correct, 0–1, chance floor **0.25**; `0/3` means it answered all three and got none right. Only the smallest samples in the set were used, so this is a *reduced-context* variant of the bench, chosen so the card could prefill it at all.
- <a name="b-mrcr"></a>**[MRCR](#g-mrcr) v2 (n=2, ~19K tokens)** — Multi-Round Co-reference Resolution: a long synthetic conversation contains several near-identical requests, and the model must reproduce one specific earlier reply, prefixed with a required random string. **Score:** mean `difflib` sequence-similarity ratio against the reference answer, 0–1, gated on the required prefix being present — so it is *not* accuracy: partial credit is possible, 1.0 means a verbatim reproduction, and the 0.014 in the table means the model emitted something with almost nothing in common with the target message. There is no chance floor; a wrong-but-fluent answer scores near 0.
- <a name="b-toolcall"></a>**Custom tool set (n=24)** — 24 hand-written cases against local function schemas (weather, stock price, and friends), graded on the API's *parsed* `tool_calls` rather than raw text, so chat-template differences don't skew it: correct function name, all required arguments present, valid JSON, and correct values where checkable. **Score:** cases passed out of 24. Crucially it includes **negative cases** — requests where the right behaviour is to answer directly and call nothing — which is where small models usually fail, and which a bench that only counts successful calls would reward them for getting wrong.
- <a name="b-clip"></a>**[CLIP score](#g-clip) — not run.** The intended prompt-adherence metric for the diffusion bake-off. Skipped because the visual gap between the candidates was decisive without it; noted so the absence isn't mistaken for a passing grade.

## 10. Running unattended: picking for accuracy, not speed

If the machine is meant to work on its own — an agent loop, a batch job, a scheduled task — the ranking above is the wrong one. Speed stops mattering once a task finishes in minutes rather than hours, and three other properties start to:

1. **Does it call tools correctly, including declining to?** An unattended loop fails the moment it invents a function name or fabricates an argument. This is the [custom tool set](#b-toolcall) and [BFCL AST](#b-bfcl), and the negative cases matter more than the positive ones.
2. **Does the run survive?** A [`vk::DeviceLostError`](#g-device-lost) or a [timeout](#b-longbench) ends the job with no output at all — a much worse outcome than a slow but complete answer, and the failure mode this hardware actually produces.
3. **Does context growth kill it?** Agent loops accumulate history. Long context is no longer the wall it was — an 8192-token prefill now completes in 4-6 minutes and [quantized KV](#g-kv-quant) costs ~4% instead of 3.5x — but architecture still decides how fast the cache grows, and a 4 GB card still runs out.

**The pick is still Granite-4.0-H-Micro, but for narrower reasons than before.** It has the best tool-calling behaviour measured here (**24/24** on the custom set including negatives, 0.863 BFCL AST), and its [Mamba-2](#g-ssm) fixed-size recurrent state means context growth costs almost no VRAM — the property that matters most for a loop that accumulates history. What is no longer true is the speed argument: it was the fastest model in the old table, and it is now **fourth** at 34.9 tg128, behind Gemma-4-E2B (50.7), Granite-4.1-3B (46.1) and Qwen3-4B-2507 (40.5). It also carries the widest error bars in the set. Pick it for the recurrent state and the tool calling, not for throughput.

Qwen3-4B-Instruct-2507 scores marginally higher on BFCL AST (0.875 vs 0.863) and is now also **faster** (40.5 vs 34.9 tg128) with tighter variance. It is dense, so its KV cache grows fastest in the set (144 KiB/token) — but with quantized KV effectively free that matters less than it did. It is a reasonable default if your contexts are bounded. Avoid nothing in this set on stability grounds any more: Gemma-4-E4B, previously the one model that lost the device, now runs.

**Configuration for unattended work**, all of it costing throughput you don't need:

- **Serve, don't re-launch.** A resident [LocalAI](#8-localai) or `llama-server` process avoids paying model load per task, and keeps the GPU warm — which on this box makes it *faster*, not slower.
- **Cap the context deliberately** and let the client truncate, rather than discovering the ceiling as a device loss mid-job. Size it from the [KV table](#55-kv-cache) with headroom, and use `-fa on -ctk q4_0 -ctv q4_0` if you need more context than f16 allows — that is now a ~4% cost rather than the 3.5× it used to be. q8_0 KV is also correct on this branch now, though q4_0 remains the better trade for context length ([4.4](#44-benchmarks)).
- **Supervise the process.** Device loss is unrecoverable in-process: the runtime must be restarted, so run it under `launchd`/a supervisor with a restart policy, and make the client retry idempotently.
- **Don't stack GPU work back-to-back.** The one reproducible trigger for device loss here was continuous sweeps with no gap. Serialize jobs and leave the GPU a moment between them.
- **Log the [`ioreg`](#g-ioreg) residency** alongside your job output. If a run silently falls back to CPU it will still produce correct answers, just ~3× slower, and you want that in the log rather than as a mystery.

**What this recommendation does not rest on:** no multi-turn agent benchmark was run here (τ²-bench was out of scope), so "calls tools correctly" is measured single-turn only. Multi-turn state tracking, recovery after a failed call, and long-horizon planning are exactly where 4B-class models are weakest, and nothing on this page measures them. Treat the pick as "the most reliable thing that fits in 4 GB", not as a claim that it is reliable in absolute terms.

## 11. Negative results worth knowing

Everything on this page that "doesn't work" is one of two things, and the distinction is the whole point of the section: a **silicon limit**, which you can only design around, or a **software limit**, which someone can fix. Sorting them took most of the effort behind this document, because on this stack they present identically — as wrong output, not as an error.

| What fails | Which kind | Status |
|---|---|---|
| Metal on a discrete GPU: weights re-read over PCIe every token; watchdog kill for diffusion | **Silicon/architecture** — Metal assumes [unified memory](#g-uma) | Not fixable. Use Vulkan (landmine 1) |
| bf16 weights → [NaN](#g-nan) | **Silicon** — `bf16: 0`, AMD added bf16 with CDNA1/RDNA3 | Not fixable. Use fp16 or a quantized GGUF |
| `--diffusion-fa` / cooperative-matrix attention has no fast kernel | **Silicon** — `matrix cores: none`, WMMA arrives with RDNA3 | Not fixable. Leave it off |
| A 4B model plus a large KV cache, or two resident models, exceeds VRAM | **Silicon** — 4 GB is 4 GB | Not fixable. Budget against **4278 MB** and load [one model per session](#one-model-per-session) |
| Integrated GPU slower than the CPU at every size | **Silicon** | Not fixable. Pinned out (landmine 2) |
| Gibberish on multi-GPU Macs | **Software** — ggml picked the device by reported VRAM; ollama passed a CPU-inclusive index | **Fixed** (device pin + `vulkan-visible-device-index`) |
| `warp size: 64`, and every subgroup shader computing over the wrong lane count | **Software** — MoltenVK misreported Metal's 32-wide SIMD group | **Fixed upstream** in MoltenVK 1.4.2 |
| `SSM_SCAN` / `GATED_DELTA_NET` → NaN on hybrid models | **Software** — shaders assumed a workgroup is one contiguous subgroup | **Fixed** on these branches |
| `int dot: 0`, quantized matmul on a slow generic path | **Software ceiling** — Metal exposes no DP4a, but the emulation still wins | **Fixed**: +68% prefill |
| Flash attention "3.5× slower" | **Software** — `FLASH_ATTN_EXT` was unsupported, so attention ran on the CPU | **Fixed**: now the faster option for decode |
| q8_0 KV wrong under flash attention; q8_0 multi-column `mul_mat_vecq` wrong | **Software** — `pack32(i16vec2(...))` mistranslated by MoltenVK | **Fixed**: full sweep clean ([4.4](#44-benchmarks)) |
| ollama's Vulkan runner fails to build on current headers | **Software** — missing `vk_video` in the fake SDK | **Fixed** ([ollama.md](ollama.md#setup)) |
| A model loaded second returns an empty completion | **Silicon cause, software symptom** — VRAM exhausted, reported as `finish_reason: stop` with zero tokens instead of an allocation error | **Open upstream.** Mitigation: one model per session |
| LocalAI reports `GPU vendor=""` and `VRAM 0` on macOS | **Software** — its probe is Linux-oriented | **Open upstream.** Use [`ioreg`](#g-ioreg) instead ([localai.md](localai.md#verifying-it-is-actually-on-the-gpu)) |
| Diffusion im2col convolution → colourful noise | **Software**, not root-caused — and in a *different* vendored ggml ([leejet's](#63-what-was-modified)), so the fixes above do not apply | Worked around by `--diffusion-conv-direct`, which is also ~3× faster |
| [`vk::DeviceLostError`](#g-device-lost) under hours of continuous load | Undetermined | Mitigation: leave the GPU a gap between jobs |

- **[llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104)** (Intel-Mac AMD Vulkan gibberish) **does not reproduce** on a current checkout: the correctness battery passes and GPU [perplexity](#g-ppl) matches CPU within 0.05%. The gibberish people hit on this hardware today is landmine 2 (wrong device selected) or landmine 4 (hybrid shaders), both fixed on these branches.
- **The integrated GPU was evaluated once and permanently excluded.** Slower than the CPU at every size tested, and numerically wrong under MoltenVK before the matmul fix. It is pinned out in every recipe here; the fix on the ggml/llama.cpp branches exists to stop it being *silently wrong*, not to make it usable.
- **Metal was measured, not assumed.** It is broken differently for LLMs (PCIe re-reads, ~0.8 tok/s) than for diffusion (watchdog timeout, no output at all).
- **[CLIP prompt-adherence scoring](#g-clip) for the diffusion bake-off was not run** — the visual gap between SD-Turbo and SDXL-Turbo was decisive without it.
- **The old flash-attention findings were all measuring a CPU fallback.** Every `-fa on` number in this project's history — the "2.3x penalty", the counterbalanced A/Bs, the claim that the effect reverses for state-space models — was taken while `FLASH_ATTN_EXT` was unsupported on Vulkan and running on the CPU. They are void. With FA on the GPU it is the faster option for decode.
- **The Radeon can lose the device under sustained load.** The one reproducible trigger left is hours of back-to-back GPU work with no gap — during one such session even a plain baseline run died with [`vk::DeviceLostError`](#g-device-lost) with the patch reverted, and ~40 minutes of idle restored normal behaviour. It is machine state, not a correctness bug, but budget for it in any automated sweep. The *other* documented trigger no longer reproduces: Gemma-4-E4B ran out of VRAM at `pp512` with FA off, and now completes in both FA configurations (52.6 / 55.2 pp512).
- **`int dot` and `warp size` in the capability banner do not tell you what the build is doing.** Both are printed from the driver's own report, before the branch's decisions are applied. `int dot: 0` shows on a build where the integer-dot path is live, and the only way to see the difference is prefill throughput. Read [Reading the ggml capability line](#reading-the-ggml-capability-line) before drawing conclusions from that line.
- **q8_0 was the problem quantization on this driver, and it is now root-caused.** Two failures that looked independent — the multi-column `mul_mat_vecq` path and q8_0 KV under flash attention — are the same defect: `pack32(i16vec2(...))` repacking from a 16-bit view, which MoltenVK mistranslates. Every other type packs from `u16vec2` and is fine. Both paths now route q8_0 elsewhere, and the full sweep passes. The lesson generalises: **a performance patch can convert a correct-but-slow path into a fast-and-wrong one**, and only a per-operator differential test catches it.

---

## 12. Glossary

Every term this README leans on, with a reference. Grouped by where it bites you.

### 12.1 The GPU stack

- <a name="g-vulkan"></a>**Vulkan** — Khronos' low-level GPU API, and the reason a 2019 AMD dGPU is usable on macOS at all: ggml ships a Vulkan compute backend, so the same kernels that run on Linux/AMD run here. ([spec home](https://www.vulkan.org/))
- <a name="g-moltenvk"></a>**MoltenVK** — an implementation of Vulkan on top of [Metal](#g-metal): every Vulkan call is translated at runtime. Faithful enough to run llama.cpp, *not* faithful enough to preserve every guarantee a compute shader assumes — which is the origin of landmines 2 and 4. ([repo](https://github.com/KhronosGroup/MoltenVK))
- <a name="g-metal"></a>**Metal** — Apple's native GPU API, and the wrong choice on this machine (landmine 1). It assumes unified memory; on a discrete GPU ggml's Metal path re-reads weights across PCIe every token. ([docs](https://developer.apple.com/metal/))
- <a name="g-icd"></a>**ICD / `VK_ICD_FILENAMES`** — an *Installable Client Driver* is the actual Vulkan driver the loader `dlopen`s. Building against Homebrew's `vulkan-loader` rather than linking `libMoltenVK` directly means this env var picks the driver at runtime, so you can swap MoltenVK for another implementation without rebuilding. ([loader/driver interface](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md))
- <a name="g-rdna1"></a>**RDNA1** — the architecture of the Radeon Pro 5500M (Navi 14). First-generation RDNA: no matrix cores, and no integer dot product the API can reach, so several ggml fast paths that are well-tested on RDNA2+/NVIDIA are cold code here. Where it sits in the line, and what each later generation would have changed: [RDNA generations, 1 to 4](#rdna-generations-1-to-4). ([overview](https://en.wikipedia.org/wiki/RDNA_(microarchitecture)))
- <a name="g-dgpu"></a>**dGPU / iGPU** — discrete (the Radeon: 4 GB of its own VRAM behind PCIe) versus integrated (a slice of system DRAM). Both enumerate under Vulkan, which is exactly the problem — see landmine 2. Everything on this page targets the discrete GPU only; the integrated one is pinned out and never used.
- <a name="g-ggml"></a>**ggml / GGUF** — the C tensor library underneath all five platforms here, and its single-file model format (weights + metadata + quantization type). One shader fix in ggml therefore moves llama.cpp, ollama, LocalAI and stable-diffusion.cpp at once. ([ggml](https://github.com/ggml-org/ggml), [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md))
- <a name="g-shader"></a>**Compute shader / SPIR-V / `.comp`** — ggml's GPU kernels are GLSL compute shaders (`ggml/src/ggml-vulkan/vulkan-shaders/*.comp`) compiled to SPIR-V bytecode by `glslc`. Both correctness fixes on these branches are edits to `.comp` files. ([SPIR-V](https://www.khronos.org/spir/))
- <a name="g-subgroup"></a>**Subgroup vs workgroup (`gl_SubgroupInvocationID` vs `gl_LocalInvocationID`)** — a *workgroup* is the block of threads that share memory and can barrier together; a *subgroup* is the hardware-lockstep slice inside it (AMD calls it a wave, Metal a SIMD-group). A shader that assumes "the workgroup is one contiguous, in-order subgroup" is relying on something MoltenVK does not guarantee — the shared root cause of landmines 2 and 4. ([subgroup tutorial](https://www.khronos.org/blog/vulkan-subgroup-tutorial))
- <a name="g-test-backend-ops"></a>**`test-backend-ops`** — ggml's per-operator differential test: run each op on a backend, diff against the CPU reference, report max error against a tolerance. The tool that converts "this model is broken" into "`SSM_SCAN` returns NaN". ([source](https://github.com/ggml-org/ggml/tree/master/tests))
- <a name="g-visible-devices"></a>**`GGML_VK_VISIBLE_DEVICES`** — ggml env var restricting which Vulkan devices are enumerated at all; `0` here means the Radeon is the only device that exists as far as ggml is concerned. The Vulkan-only analogue of `CUDA_VISIBLE_DEVICES`.

### 12.2 Capability-line fields

The line these define, and what a `0` in it does or doesn't mean, is at the top of this page: [Reading the ggml capability line](#reading-the-ggml-capability-line).

- <a name="g-uma"></a>**`uma`** — unified memory architecture. `0` here: GPU memory is separate from host RAM, so every weight must be copied across PCIe and *stay* there. On Apple Silicon this is `1`, which is why Metal advice written for M-series does not transfer.
- <a name="g-fp16"></a>**`fp16` / `bf16`** — half-precision support. This card does fp16 (`1`) but not bfloat16 (`0`), so bf16 weights produce [NaN](#g-nan); use fp16 or a quantized GGUF. ([bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format))
- <a name="g-intdot"></a>**`int dot`** — hardware 8-bit integer dot-product (`DP4a`-style), the instruction that makes quantized matmul fast. `0` here, so quantized kernels take a slower generic path — **but not because the chip lacks it**. ggml gates this on a single flag, `integerDotProduct4x8BitPackedSignedAccelerated`, and MoltenVK exposes `VK_KHR_shader_integer_dot_product` while reporting every `Accelerated` flag as false. That is honest: Metal Shading Language has no DP4a-style intrinsic (nothing like HLSL's `dot4add_i8packed` or CUDA's `__dp4a`), so MoltenVK can only emulate it in generic integer math. Navi 14 is one of the RDNA1 parts that *does* have the native `V_DOT4_I32_I8` instruction — Navi 12 and Navi 14 got AMD's dot-product instructions, Navi 10 did not. Silicon has it; the API cannot request it. (Cross-check on Linux under RADV, where the same chip should report that flag `true`.)
- <a name="g-coopmat"></a>**`matrix cores` / cooperative matrix (coopmat, coopmat2)** — tensor-core-class matrix instructions exposed through `VK_KHR_cooperative_matrix`. `none` here — which is why [flash attention](#g-flash-attn) and `--diffusion-fa` have no fast kernel and fall back badly (landmines 3 and 5). ([extension](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html))
- <a name="g-warp"></a>**`warp size`** — threads per [subgroup](#g-subgroup). AMD's GCN/RDNA1 can run wave64, versus 32 on NVIDIA and typically 32 on Apple GPUs, so shaders tuned for 32 behave differently on a wave64 device. RDNA1 is a dual-mode wave32/wave64 architecture and **wave32 is its native compute mode**. MoltenVK up to 1.4.1 reported the subgroup as 64 while Metal actually ran 32 — not a ceiling but a **wrong number**, and the reason every subgroup reduction in every shader computed over the wrong lane count. 1.4.2 reports 32 and the mistranslations disappear; this is the field to check first on any new machine. This is the same refusal that makes upstream disable subgroup enforcement on Intel ("performance issues when enforcing subgroup sizes"), and a 64-lane subgroup is the environment in which `SSM_SCAN`/`GATED_DELTA_NET` made the contiguous-subgroup assumption that produced [NaN](#g-nan).

The `VK_AMD_shader_core_properties` extension is also absent, which is why ggml's AMD architecture detection falls through to `OTHER` on macOS and every RDNA-specific tuning path is skipped — the reason the fork carries a `GGML_VK_FORCE_ARCH` override.

### 12.3 Failure modes

- <a name="g-nan"></a>**NaN** — IEEE-754 "not a number". A single NaN propagates through every subsequent matmul, so a broken shader surfaces as confident token salad or full-frame image noise, never as a crash. That is what makes these bugs worth writing down. ([IEEE 754](https://en.wikipedia.org/wiki/IEEE_754))
- <a name="g-device-lost"></a>**`vk::DeviceLostError` (`VK_ERROR_DEVICE_LOST`)** — the GPU stopped responding; every Vulkan object belonging to it is now invalid and the process cannot recover. On this card it means either a VRAM blow-up (Gemma-4-E4B at `pp512`) or sustained back-to-back GPU load (Qwen3.5-4B under a warm-up sweep). ([VkResult](https://registry.khronos.org/vulkan/specs/latest/man/html/VkResult.html))
- <a name="g-watchdog"></a>**GPU watchdog timeout (`kIOAccelCommandBufferCallbackErrorTimeout`)** — macOS kills a command buffer that runs too long, so the desktop stays responsive. Metal diffusion trips it on every run here (landmine 1). ([MTLCommandBufferError](https://developer.apple.com/documentation/metal/mtlcommandbuffererror))
- <a name="g-spill"></a>**CPU spill** — when a model or its [KV cache](#g-kv) doesn't fit in VRAM, ggml keeps the overflow in host RAM and pays a PCIe round-trip for it. It shows up in llama.cpp's load log as a CPU buffer appearing, and in throughput as a cliff, not an error.
- <a name="g-throttle"></a>**Thermal throttling / `CPU_Speed_Limit`** — the SMC's current CPU clock cap in percent, readable with `pmset -g therm`; 100 is unthrottled, this laptop reaches 20–36 under two minutes of load. Counter-intuitively, *cooling* the machine makes the GPU numbers worse — see [the benchmarking warning](#56-a-warning-about-benchmarking-this-machine).
- <a name="g-ioreg"></a>**`ioreg` / `IOAccelerator` / `inUseVidMemoryBytes`** — macOS's IOKit registry, and the per-accelerator property that reports bytes of VRAM actually in use. The ground truth for "is the model really on the Radeon?" when a runtime's own reporting is unreliable (LocalAI's is). ([ioreg(8)](https://keith.github.io/xcode-man-pages/ioreg.8.html))

### 12.4 LLM inference and serving

- <a name="g-kv"></a>**KV cache ("KV")** — attention's cached per-token key/value tensors, so generating token *n* doesn't recompute tokens *1…n-1*. It grows linearly with context, it is the second consumer of VRAM after the weights, and on a 4 GB card it — not the model size — usually sets your context ceiling. ([explainer](https://huggingface.co/docs/transformers/en/kv_cache))
- <a name="g-kv-quant"></a>**KV precision (`f16` / `q8_0` / `q4_0`)** — the cache can itself be quantized, trading recall for context length (the table in [5.5](#55-kv-cache): 32 → 9 KiB/token, ~41K → 128K context). In llama.cpp quantized KV requires [flash attention](#g-flash-attn). That used to make it unreachable here, because FA fell back to the CPU; with FA on the GPU it costs ~4%. On this driver use **q4_0, not q8_0** — the q8_0 KV path fails 337 `FLASH_ATTN_EXT` cases at ERR 0.04–0.11.
- <a name="g-ctx"></a>**Context size (`--ctx-size`, `n_ctx`, `context_size`)** — how many tokens the model can attend to at once, and the direct multiplier on [KV](#g-kv) size.
- <a name="g-prefill"></a>**Prefill vs decode** — prefill (prompt processing) runs the whole prompt in parallel and is compute-bound; decode (token generation) emits one token at a time and is memory-bandwidth-bound. Different bottlenecks, which is why they are benchmarked separately as [pp512 and tg128](#g-pp512).
- <a name="g-flash-attn"></a>**Flash attention (`--flash-attn`, `-fa`, `OLLAMA_FLASH_ATTENTION`, `flash_attention`)** — an attention kernel that tiles the computation so the N×N attention matrix is never materialized. A large win where cooperative-matrix kernels exist. On this stack there are none, and for a long time it was worse than that: `FLASH_ATTN_EXT` was unsupported on Vulkan entirely, so ggml ran attention on the **CPU** and `-fa on` looked like a 3.5× penalty (landmine 3). With the subgroup-free path on these branches it runs on the GPU and is the faster option for decode. ([paper](https://arxiv.org/abs/2205.14135))
- <a name="g-quant"></a>**Quantization (`Q4_K_M`, `q8_0`, `--type q8_0`)** — storing weights below fp16. `Q4_K_M` is llama.cpp's k-quant "4-bit, medium" mix at ~4.5 bits/weight — the default sweet spot, and what makes a 4B model fit in 4 GB. sd.cpp's `--type` quantizes a safetensors checkpoint on load instead. ([k-quants](https://github.com/ggml-org/llama.cpp/pull/1684), [type table](https://huggingface.co/docs/hub/en/gguf#quantization-types))
- <a name="g-ngl"></a>**`-ngl` / `gpu_layers` / `num_gpu` (layer offload)** — how many transformer layers live on the GPU; `99` means "all of them", `0` is CPU-only, anything between splits the model and pays a PCIe hop per token. Same knob, three spellings: llama.cpp, LocalAI, ollama.
- <a name="g-arch"></a>**Dense / GQA / hybrid / PLE** — the architecture column of the fit table. *Dense* = attention in every layer, [KV](#g-kv) grows with context. *[GQA](https://arxiv.org/abs/2305.13245)* (grouped-query attention) = several query heads share one key/value head, shrinking that cache. *Hybrid* = most layers are recurrent state-space blocks with fixed-size state, so long context costs almost no memory. *[PLE](https://ai.google.dev/gemma/docs/gemma-3n)* (per-layer embeddings, Gemma) = a large embedding table stays in host RAM while the compute core sits on the GPU, so "E4B" fits in far less VRAM than its parameter count suggests.
- <a name="g-ssm"></a>**SSM / Mamba-2 / `SSM_SCAN`** — the state-space model family and its selective-scan operator; `SSM_SCAN` is the ggml op (and Vulkan shader) that implements it. Returned NaN on this stack until the fix in landmine 4. ([paper](https://arxiv.org/abs/2405.21060))
- <a name="g-gdn"></a>**Gated DeltaNet / `GATED_DELTA_NET`** — a gated linear-attention variant with delta-rule state updates, used by Qwen3.5; the second shader hit by landmine 4. Its prefill is a sequential scan, i.e. thousands of tiny dispatches — cheap on a native driver, expensive through MoltenVK, which is why it times out on long context. ([paper](https://arxiv.org/abs/2412.06464))
- <a name="g-vlm"></a>**VLM / `mmproj` (vision projector) / image tokens** — a vision-language model pairs an LLM with an image encoder; in GGUF form the encoder+projector is a separate `mmproj-*.gguf`, and an image is turned into N embedding slots ("image tokens"). More tokens = better grounding and proportionally more [prefill](#g-prefill), which is what `OLLAMA_IMAGE_MIN_TOKENS` trades off. ([llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md))
- <a name="g-thinking"></a>**Thinking models / reasoning field / CoT** — models trained to emit a chain of thought before answering. Over an OpenAI-compatible API the thinking text may arrive in `message.reasoning` while `message.content` stays empty until the think block closes — a failure-looking behaviour that isn't one. ([CoT paper](https://arxiv.org/abs/2201.11903))
- <a name="g-openai-api"></a>**OpenAI-compatible API / `/v1/chat/completions`** — the de-facto REST shape (`model`, `messages`, `max_tokens`, streamed chunks) that llama-server, ollama, and LocalAI all speak, so any OpenAI SDK client works unchanged. ([reference](https://platform.openai.com/docs/api-reference/chat))
- <a name="g-grpc"></a>**gRPC backend (LocalAI)** — LocalAI doesn't link llama.cpp; it spawns a `grpc-server` child process and talks [gRPC](https://grpc.io/) to it. That indirection is why the MoltenVK environment has to be carried by a wrapper script, and why the backend's ggml banner never reaches LocalAI's log.
- <a name="g-oci"></a>**OCI image** — Open Container Initiative: the standardised format and distribution protocol behind container images, split from Docker in 2015 so that images, registries and runtimes interoperate. An OCI image is just content-addressed layers plus a manifest, and any OCI registry can serve them — which is why it gets used to ship things that are not containers at all. LocalAI's backend gallery does exactly that: a backend is pulled as an OCI image from a registry rather than compiled locally. Nothing on this page uses that path — every backend here is built from source for Vulkan, because no published image targets an Intel Mac with a discrete AMD GPU. ([spec](https://github.com/opencontainers/image-spec))
- <a name="g-ppl"></a>**Perplexity** — exp of the mean negative log-likelihood over held-out text; the standard check that a backend is numerically sane, since a broken GPU path diverges from CPU immediately. ([definition](https://huggingface.co/docs/transformers/en/perplexity))

### 12.5 Benchmarks and metrics

One-line definitions; the full protocol for each is in [Benchmarks and how they were run](#9-benchmarks-and-how-they-were-run).

- <a name="g-pp512"></a>**pp512 / tg128** — `llama-bench`'s two workloads, both reported in tokens/second: **pp** = [prefill](#g-prefill) of an N-token prompt, **tg** = generation of N tokens. So `pp512` is "how fast it digests a 512-token prompt", `tg128` is "how fast it types". ([llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench))
- <a name="g-toks"></a>**tok/s** — tokens per second; unqualified here it means generation. A token averages ~¾ of an English word.
- <a name="g-counterbalance"></a>**Counterbalanced A/B** — running A-then-B and B-then-A and reporting both rounds, so drift (here: GPU clock ramp and thermal state) cancels instead of accumulating against one arm. Every performance claim on these branches was taken this way, and it is what caught a "3–14% regression" that turned out to be pure drift: re-measuring after reverting the change reproduced the same slow numbers. ([counterbalancing](https://en.wikipedia.org/wiki/Counterbalanced_measures_design))
- <a name="g-sd"></a>**± and "sd"** — standard deviation across repetitions. It is the load-bearing number here: `10.88 ± 0.13` is a measurement, `14.30 ± 6.25` is noise wearing a mean's clothing.
- <a name="g-bfcl"></a>**BFCL-V4 / AST** — Berkeley Function-Calling Leaderboard: does the model emit the right tool call with the right arguments? *AST* scoring parses the emitted call and compares it structurally against the reference rather than executing it. ([leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html))
- <a name="g-mmlu"></a>**MMLU-Pro** — a harder MMLU: 10 options instead of 4, reasoning-heavy, run 0-shot with [CoT](#g-thinking). ([dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro))
- <a name="g-longbench"></a>**LongBench-v2** — multiple-choice QA over long documents. ([dataset](https://huggingface.co/datasets/THUDM/LongBench-v2))
- <a name="g-mrcr"></a>**MRCR** — OpenAI's Multi-Round Co-reference Resolution: retrieve one specific earlier message from a long synthetic conversation full of near-identical distractors. ([dataset](https://huggingface.co/datasets/openai/mrcr))

### 12.6 Diffusion

- <a name="g-diffusion"></a>**Latent diffusion / steps / cfg-scale / sampler** — generation denoises a small latent image over N *steps*, then decodes it to pixels. *cfg-scale* is how hard the prompt is enforced (turbo models want ~1, SD 1.5 ~7); the *sampler* (`euler`, `euler_a`, …) is the integration rule. ([Stable Diffusion paper](https://arxiv.org/abs/2112.10752), [sampler design](https://arxiv.org/abs/2206.00364))
- <a name="g-unet"></a>**UNet** — the convolutional encoder/decoder that performs the denoising, and where nearly all the GPU time — and landmine 5 — lives. ([paper](https://arxiv.org/abs/1505.04597))
- <a name="g-conv-direct"></a>**im2col convolution vs `--diffusion-conv-direct`** — the default GPU path turns each convolution into one large matmul over an unfolded ("im2col") copy of the input, which is numerically broken on this [RDNA1](#g-rdna1)/MoltenVK stack. `--diffusion-conv-direct` convolves directly instead: correct *and* ~3× faster. ([im2col background](https://arxiv.org/abs/1410.0759), [sd.cpp #563](https://github.com/leejet/stable-diffusion.cpp/issues/563))
- <a name="g-vae"></a>**VAE / VAE decode / `--vae-on-cpu`** — the variational autoencoder that turns the final latent into pixels. It is the *peak* VRAM moment of a run, so a model can load comfortably and still die at the last step; moving it to the CPU is the standard 4 GB lever, and also dodges SDXL's fp16 NaN. ([paper](https://arxiv.org/abs/1312.6114))
- <a name="g-turbo"></a>**SD-Turbo / SDXL-Turbo** — Stable Diffusion variants distilled (adversarial diffusion distillation) to produce usable images in 1–4 steps rather than 20–50. That 5–10× step reduction is what makes image generation practical on this card. ([paper](https://arxiv.org/abs/2311.17042))
- <a name="g-safetensors"></a>**safetensors** — Hugging Face's memory-mappable tensor format with no code execution on load; sd.cpp reads it directly, quantizing on the fly with `--type`. ([repo](https://github.com/huggingface/safetensors))
- <a name="g-lora"></a>**LoRA / ControlNet** — the two ecosystems that make an older base model worth keeping: small low-rank fine-tune adapters ([LoRA](https://arxiv.org/abs/2106.09685)) and structural conditioning on an edge map, pose or depth map ([ControlNet](https://arxiv.org/abs/2302.05543)). SD 1.5 has by far the largest library of both.
- <a name="g-clip"></a>**CLIP score** — cosine similarity between CLIP's embedding of the prompt and of the generated image; a reference-free proxy for prompt adherence. Listed in the negative results because the visual gap didn't need it. ([paper](https://arxiv.org/abs/2104.08718))
