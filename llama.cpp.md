# llama.cpp

The reference LLM runtime, and the tool to measure on. Everything else here wraps this code.

| | |
|---|---|
| **Role in the stack** | LLM runtime on top of ggml |
| **Use it for** | Comparing models, tuning flags, benchmarking, and serving an OpenAI-compatible API directly |
| **Fork and branch** | [`maximosipov/llama.cpp`, `macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/llama.cpp/tree/macbook-pro-2019-radeon-5500m-4gb) |
| **Upstream** | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **Source changes** | 10 commits: the [ggml fixes](ggml.md#changes-on-this-branch) plus diagnostic switches |
| **Used by** | [ollama](ollama.md) and [LocalAI](localai.md), which cherry-pick these commits onto their own pinned versions |
| **Runbook on the branch** | [RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md](https://github.com/maximosipov/llama.cpp/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) |

Complete the [shared setup](README.md#shared-setup) first. Terms are defined in the [glossary](glossary.md).

## Overview

llama.cpp is a toolbox rather than a single binary. What ships in `tools/`:

- **`llama-cli`** — one-shot or interactive generation, full control of sampling, and [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) grammar-constrained output.
- **`llama-server`** — an [OpenAI-compatible](glossary.md#g-openai-api) HTTP server (`/v1/chat/completions`, `/completion`, `/embedding`, `/tokenize`) with a built-in web UI, parallel request slots, context shifting, speculative decoding, tool calling and multimodal support.
- **`llama-bench`** — the [pp512 and tg128](glossary.md#g-pp512) microbenchmark used throughout this repository. `llama-batched-bench` sweeps batch sizes.
- **`llama-quantize`** and **`llama-imatrix`** — convert an f16 GGUF to [Q4_K_M](glossary.md#g-quant) and similar, optionally guided by an importance matrix.
- **`llama-perplexity`** — [perplexity](glossary.md#g-ppl), KL divergence against a reference, and multiple-choice evaluation.
- **`llama-mtmd-cli`** — multimodal inference: an LLM plus an [`mmproj`](glossary.md#g-vlm) vision or audio encoder.
- **Smaller tools** — `llama-tokenize`, `llama-gguf-split`, `llama-export-lora`, `llama-tts`, and `llama-rpc-server`.

**Why it matters here:** ollama and LocalAI are both wrappers around this code, so measure here first. `llama-bench` is the only reliable way to compare models on this machine, and the features that make a 4 GB card work — [layer offload](glossary.md#g-ngl), [KV cache quantization](glossary.md#g-kv-quant) and [flash attention](glossary.md#g-flash-attn) — are all controlled by its flags.

## Changes on this branch

The [same nine Vulkan changes as ggml](ggml.md#changes-on-this-branch) (llama.cpp vendors a synced copy of ggml), plus one commit adding diagnostic environment switches.

**These SHAs matter.** The [ollama](ollama.md#build) and [LocalAI](localai.md#build) recipes cherry-pick exactly this list onto their own pinned llama.cpp commits, because neither can build from this branch's tip:

```
7847fca13  matmul subgroup correctness (RDNA1 and older Intel GPUs)
595936bcd  NaN from SSM_SCAN and GATED_DELTA_NET
a91f29c5f  diagnostic environment switches
acb45b872  disable subgroup clustered operations (the q8_1 activation quantizer)
8cbd19056  emulated integer dot for matmul: +68% prefill
8e8d7fb26  GGML_VK_FA_NO_SUBGROUPS diagnostic switch
16ea5d8e7  run flash attention on the GPU
e629266a1  guard the broken q8_0 mul_mat_vecq path
2707498b9  trust MoltenVK subgroups from 1.4.2 onwards
268aff1aa  keep q8_0 off the flash-attention MMQ path
```

All ten apply cleanly to ollama's pinned `b10091` and to LocalAI's pinned `1cbfd1988`, verified while writing this guide.

## Build

The fork is a submodule of this repository, already on the right branch:

```bash
git submodule update --init llama.cpp     # about 540 MB
cd llama.cpp
```

To build it on its own instead:

```bash
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/llama.cpp.git
cd llama.cpp
```

Then, in either case:

```bash
P="$(brew --prefix)"
cmake -B build -DGGML_VULKAN=1 -DGGML_METAL=OFF -DLLAMA_CURL=1 \
  -DVulkan_INCLUDE_DIR="$P/opt/vulkan-headers/include" \
  -DVulkan_LIBRARY="$P/opt/vulkan-loader/lib/libvulkan.dylib" \
  -DVulkan_GLSLC_EXECUTABLE="$P/opt/shaderc/bin/glslc" \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE="$P/opt/glslang/bin/glslangValidator" \
  -DOpenMP_ROOT="$P/opt/libomp" \
  -DCMAKE_CXX_FLAGS="-I$P/opt/spirv-headers/include" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"
```

`-DGGML_METAL=OFF` is required, not optional ([known issue](README.md#ki-metal)).

## Run

Use `--flash-attn on` everywhere. It runs on the GPU on this branch, which reverses this project's earlier advice ([known issue](README.md#ki-flash-attention)). `-st` makes `llama-cli` answer once and exit; without it, newer builds open a chat UI.

```bash
# One-shot generation
./build/bin/llama-cli -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 --flash-attn on --ctx-size 4096 -st --temp 0 \
  -p "How much is 20+34? Answer with just the number."

# OpenAI-compatible server
./build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --flash-attn on --ctx-size 8192 --host 127.0.0.1 --port 8080

# Long context on 4 GB: a quantized KV cache needs flash attention, and costs about 4%
./build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --flash-attn on -ctk q4_0 -ctv q4_0 --ctx-size 32768

# Vision: an LLM plus its projector
./build/bin/llama-mtmd-cli -m models/Qwen3VL-4B-Thinking-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3VL-4B-Thinking-Q8_0.gguf \
  -ngl 99 --flash-attn on --ctx-size 2048 --image photo.png -p "What does the sign say?"
```

Which model to load, and how much context each one can hold: [models.md](models.md).

## Verify

Check the device banner in the startup log: `Vulkan0` must be the Radeon, with `warp size: 32` ([shared setup](README.md#shared-setup)).

Confirm the model is really on the GPU. The load log should show the weights in a Vulkan buffer, and no CPU buffer for the offloaded layers:

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on --ctx-size 8192 2>&1 | grep -E "buffer size|KV"
```

For an independent check, read VRAM use from IOKit while generating. A 4B Q4_K_M model resident on the Radeon shows about 3.4 GiB:

```bash
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

## Performance

### Reproducing the measurements

```bash
# Throughput and prompt processing: one configuration per process
./build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 3

# GPU against CPU
./build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -n 128 -r 3
./build/bin/llama-bench -m model.gguf -ngl 0  -fa 1 -n 128 -r 3

# Context ceiling: raise --ctx-size until the load log shows a CPU buffer appearing
./build/bin/llama-server -m model.gguf -ngl 99 -fa on --ctx-size 40960 2>&1 | grep -E "KV|buffer size"
```

Reproduced on a fresh build of the branch tip — Qwen3-4B-Instruct-2507 Q4_K_M, `-ngl 99 -fa 1 -r 3`, CPU unthrottled: **pp512 59.65 ± 0.51, tg128 40.44 ± 0.11**, against 59.4 and 40.8 when the fixes landed.

**Read [benchmarking.md](benchmarking.md) before trusting any number these commands print.** The same binary, model and flags produced 10.5, 24.9 and 33.9 tok/s across three sessions.

### KV cache precision

Measured on Qwen3.5-4B. This is the setting that decides your [context length](glossary.md#g-ctx):

| [KV precision](glossary.md#g-kv-quant) | Bytes per token | Max context with no [CPU spill](glossary.md#g-spill) | Correct on this stack? |
|---|---|---|---|
| f16 | 32.0 KiB | ~41K | Yes |
| q8_0 | 17.0 KiB | ~82K | Yes on this branch; wrong on builds without the [q8_0 flash-attention guard](README.md#ki-q8-0) |
| q4_0 | 9.0 KiB | **128K** | Yes |

A quantized KV cache requires flash attention. Now that flash attention runs on the GPU, it costs about **4%**: `-ctk q4_0 -ctv q4_0` measures 36.7 tg128 against 38.1 for f16. Prefer q4_0.

### Flash attention

`-fa on` is a clear win for generation, and at long context a *loss* for prefill: pp8192 measures 24.3 with a q4_0 KV cache, against 31.5 with f16 and flash attention off. So the rule isn't a blanket setting: **turn it on for generation-heavy work, and whenever you need a quantized KV cache**, which it is a prerequisite for.

Per-model speed, VRAM and context ceilings are in [models.md](models.md).

## Troubleshooting

**Output is gibberish.** Check the device banner: if `warp size` reads 64, upgrade MoltenVK ([known issue](README.md#ki-subgroup-size)). If the model is a hybrid (Mamba-2, Gated DeltaNet), check that you're on this branch rather than upstream. Then run [`test-backend-ops`](ggml.md#run).

**Generation is far slower than the numbers here.** Check that Metal wasn't compiled in (`otool -L build/bin/llama-cli | grep -i metal` should print nothing), that `-ngl 99` was accepted, and that no CPU buffer appears in the load log.

**`vk::DeviceLostError` part-way through a run.** Unrecoverable in-process; the runtime must be restarted. On this card it follows hours of continuous GPU work. Leave gaps between jobs and supervise long-running servers.

**A benchmark warm-up pass triggers a device loss.** Adding one did exactly that for Qwen3.5-4B. A more representative number also means a less likely finish.

**Watch for orphaned benchmark processes.** One outlived a `pkill` of its parent script and held the machine at `CPU_Speed_Limit=36` while the next run sat waiting for a machine that something else was busy heating.

## Reference

Key flags on this hardware:

| Flag | Use |
|---|---|
| `-ngl 99` | Put every layer on the GPU. `0` is CPU-only. |
| `--flash-attn on` | Flash attention. Required for a quantized KV cache. |
| `-ctk q4_0 -ctv q4_0` | Quantize the KV cache: about 4% slower, and 3.5× more context |
| `--ctx-size N` | Context window. Size it from [models.md](models.md). |
| `-st`, `--temp 0` | Single-turn output, deterministic sampling |

Diagnostic environment switches added by this branch (all opt-in, used to isolate the fixes):

| Variable | Effect |
|---|---|
| `GGML_VK_FORCE_ARCH` | Override ggml's AMD architecture detection, which falls through to `OTHER` on macOS |
| `GGML_VK_FORCE_INTEGER_DOT` | Force the integer-dot matmul path on |
| `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ` | Select matmul shader variants |
| `GGML_VK_FA_NO_SUBGROUPS` | Force the subgroup-free flash-attention path |

The ggml-level variables listed in [ggml.md](ggml.md#reference) work here too.
