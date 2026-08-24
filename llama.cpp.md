# llama.cpp — build and run recipe

Part of [AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)](README.md). Do the [shared setup](README.md#3-shared-setup-do-this-once) first; every term used here is defined in the [glossary](README.md#12-glossary). What llama.cpp is and why it is the thing to measure on: [chapter 5](README.md#5-llamacpp). The measurements themselves — [models that fit 4 GB](README.md#54-models-that-fit-4-gb), the [KV cache table](README.md#55-kv-cache), the [benchmarking warning](README.md#56-a-warning-about-benchmarking-this-machine) and the [capability benches](README.md#57-capability-benches) — stay on the main page.

Branch: [`macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/llama.cpp/tree/macbook-pro-2019-radeon-5500m-4gb).

## What was modified

The same six Vulkan changes as [ggml](ggml.md#what-was-modified) (llama.cpp vendors a synced copy of ggml), plus `vulkan: add diagnostic env knobs` — `GGML_VK_FORCE_ARCH`, `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ`, `GGML_VK_FORCE_INTEGER_DOT`, opt-in, used to isolate them.

Nine commits in total, and the SHAs matter: the [ollama](ollama.md#setup) and [LocalAI](localai.md#setup) recipes cherry-pick exactly this list onto their own pinned llama.cpp commits, because neither can build from this branch's tip.

```
7847fca13  matmul subgroup correctness (RDNA1 + older Intel iGPUs)
595936bcd  NaN SSM_SCAN / GATED_DELTA_NET
a91f29c5f  diagnostic env knobs
acb45b872  disable subgroup clustered ops (the q8_1 activation quantizer)
8cbd19056  emulated integer dot for matmul — +68% prefill
8e8d7fb26  flash-attention subgroup knob (superseded by the next one)
16ea5d8e7  run flash attention on the GPU
e629266a1  guard the broken q8_0 mul_mat_vecq path
2707498b9  trust MoltenVK subgroups from 1.4.2 onwards
```

## Setup

This fork is a submodule of this repo at `llama.cpp/`, already on the right branch — `git submodule update --init llama.cpp` and `cd llama.cpp` is enough, and it carries the full history the nine SHAs above resolve against. The standalone clone is for building it on its own:

```bash
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/llama.cpp.git
cd llama.cpp

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

## Run

Note `--flash-attn on` everywhere — the reverse of this project's earlier advice, because FA now runs on the GPU (landmine 3). `-st` is what makes `llama-cli` answer once and exit; newer builds open a chat UI otherwise.

```bash
# text
./build/bin/llama-cli -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 --flash-attn on --ctx-size 4096 -st --temp 0 \
  -p "How much is 20+34? Answer with just the number."

# OpenAI-compatible server
./build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --flash-attn on --ctx-size 8192 --host 127.0.0.1 --port 8080

# long context on 4 GB — quantized KV needs FA, and now costs ~4%
./build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --flash-attn on -ctk q4_0 -ctv q4_0 --ctx-size 32768

# vision (Qwen3-VL-4B-Thinking, LLM + projector)
./build/bin/llama-mtmd-cli -m models/Qwen3VL-4B-Thinking-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3VL-4B-Thinking-Q8_0.gguf \
  -ngl 99 --flash-attn on --ctx-size 2048 --image photo.png -p "What does the sign say?"
```

`-fa on` is a decode win and, at long context, a prefill *loss* (pp8192 24.3 with q4_0 KV against 31.5 with f16 KV and FA off). The rule is not a blanket on/off: **on for generation-heavy work, and it is the only way to get quantized KV at all.**

Which model to load, and how much context each precision buys you: [5.4 Models that fit 4 GB](README.md#54-models-that-fit-4-gb) and [5.5 KV cache](README.md#55-kv-cache).

## Reproducing the benchmarks

```bash
# throughput + prompt processing (one config per process — see the warning below)
./build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 3

# GPU vs CPU
./build/bin/llama-bench -m model.gguf -ngl 99 -fa 1 -n 128 -r 3
./build/bin/llama-bench -m model.gguf -ngl 0  -fa 1 -n 128 -r 3

# context ceiling: raise --ctx-size until the load log shows a CPU buffer appearing
./build/bin/llama-server -m model.gguf -ngl 99 -fa on --ctx-size 40960 2>&1 | grep -E "KV|buffer size"
```

Reproduced on a fresh build of the branch tip while writing this revision — Qwen3-4B-Instruct-2507 Q4_K_M, `-ngl 99 -fa 1 -r 3`, unthrottled CPU: **pp512 59.65 ± 0.51, tg128 40.44 ± 0.11**, against the 59.4 / 40.8 recorded when the fixes landed.

**Read [the warning about benchmarking this machine](README.md#56-a-warning-about-benchmarking-this-machine) before you trust any number these commands print.** The same binary, model and flags produced 10.5, 24.9 and 33.9 tok/s across three sessions; warm the GPU with a discarded run, use one process per config, and alternate the order of what you are comparing.
