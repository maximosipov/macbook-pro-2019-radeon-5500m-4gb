# AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)

Build-and-run recipes for **ggml**, **llama.cpp**, **stable-diffusion.cpp**, **ollama**
and **LocalAI** on a 2019 Intel MacBook Pro, with the model resident in the discrete
AMD GPU's VRAM.

Every recipe here was built and measured on one machine:

> **MacBook Pro 16" 2019** · Intel Core i9 · **AMD Radeon Pro 5500M, 4 GB (RDNA1)** ·
> Intel UHD 630 (present, deliberately excluded) · macOS 14.8.2
> GPU capability as ggml reports it: `uma: 0 · fp16: 1 · bf16: 0 · int dot: 0 ·
> matrix cores: none · warp size: 64`

Three of the five needed source patches; those live on a `macbook-pro-2019-radeon-5500m-4gb`
branch of the corresponding fork, alongside a self-contained runbook. The other two need
only the right build and run flags, which are below.

| Platform | What it gives you | Changes needed |
|---|---|---|
| [ggml](https://github.com/maximosipov/ggml/tree/macbook-pro-2019-radeon-5500m-4gb) | the compute kernels + `test-backend-ops`, where the GPU bugs get found and fixed | 2 shader fixes + runbook |
| [llama.cpp](https://github.com/maximosipov/llama.cpp/tree/macbook-pro-2019-radeon-5500m-4gb) | CLI, OpenAI-compatible server, `llama-bench` | same 2 fixes + diagnostic knobs + runbook |
| [ollama](https://github.com/maximosipov/ollama/tree/macbook-pro-2019-radeon-5500m-4gb) | `ollama run`, model library, desktop app + DMG | 5 patches + runbook |
| [stable-diffusion.cpp](https://github.com/maximosipov/stable-diffusion.cpp) | text-to-image | no source changes needed — the recipe is the build/run flags below |
| [LocalAI](https://github.com/maximosipov/LocalAI) | drop-in OpenAI API server + model gallery | no source changes needed — the recipe is the build/run flags below |

## Why any of this is necessary

macOS on an Intel Mac gives you Metal. On this machine **Metal is the wrong backend**:
the GPU is discrete, not unified-memory, and ggml's Metal path re-reads model weights
across PCIe on every token (~0.8 tok/s for an LLM; for diffusion it doesn't finish at all
— the macOS watchdog kills the command buffer with
`kIOAccelCommandBufferCallbackErrorTimeout`).

The path that works is **Vulkan, translated to Metal by [MoltenVK](https://github.com/KhronosGroup/MoltenVK)**.
Weights stay resident in the 4 GB of VRAM and generation runs roughly 40× faster (0.8 →
~34 tok/s on a 4B Q4_K_M). But MoltenVK
is a translation layer, and it does not honour every guarantee a shader expects from a
native Vulkan driver — which is where the bugs below come from. All of them produce
**wrong output, not crashes**, which is why they need to be written down.

## The six landmines

Everything else in this repo is setup detail. These are the findings.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | ~0.8 tok/s, or a GPU watchdog timeout | Metal re-reads weights over PCIe on a discrete GPU | `-DGGML_METAL=OFF`, use Vulkan — **and pass the flag explicitly**, ggml auto-enables Metal on macOS even when the wrapper's own `SD_METAL=OFF`/`BUILD_TYPE=vulkan` says otherwise, and Metal then wins device 0 at runtime |
| 2 | Gibberish tokens, or a runner that 500s on allocation | The Intel UHD 630 advertises **32 GiB** of shared host RAM against the Radeon's real 4 GiB, so every "pick the biggest GPU" heuristic picks the iGPU — where MoltenVK mistranslates subgroup arithmetic and **`MUL_MAT` fails every case** (error 2.4–11.4 against a 5e-4 tolerance) | `GGML_VK_VISIBLE_DEVICES=0` to pin the Radeon; the matmul fix in the ggml/llama.cpp branches makes the UHD merely slow instead of silently wrong |
| 3 | **Generation ~2.3× slower** than it should be | `--flash-attn` — RDNA1-over-MoltenVK has no cooperative-matrix kernels, so FA falls back to a very slow path. The official run commands for most models turn it on | Turn it **off**. Qwen3-4B-2507 tg128: 4.80 → **10.88 tok/s**, measured order-counterbalanced (sd 0.01/0.13). Caveat: FA-off forces f16 KV, which costs VRAM — and for one model (Gemma-4-E4B) that's enough to lose the device |
| 4 | Hybrid/state-space models (Mamba-2, Gated DeltaNet) emit token salad on GPU, coherent on CPU | `SSM_SCAN` and `GATED_DELTA_NET` drove shared-memory reductions from `gl_SubgroupInvocationID`, assuming a workgroup is one contiguous subgroup. MoltenVK doesn't guarantee that → **NaN** | Index by `gl_LocalInvocationID.x` instead (26 lines, 2 files) — on the ggml and llama.cpp branches. The correct path also appeared faster than the broken one, though the two measurements predate the [regime controls](#a-warning-about-benchmarking-this-machine) below, so treat that as directional |
| 5 | Diffusion output is full-frame colourful noise | The generic Vulkan UNet convolution (im2col + matmul) is numerically broken on RDNA1/MoltenVK (no matrix cores, no int-dot) | `--diffusion-conv-direct` — one flag, and it's also **~3× faster** than the broken path |
| 6 | Model "fits" but is unusably slow, or OOMs at the last step | 4 GB is the real constraint. For diffusion, **peak VRAM is at VAE decode**, not sampling | Quantize (`--type q8_0`), keep the VAE on CPU for SDXL, and check fit against **4278 MB** usable, not 4096 |

Landmines 2 and 4 have the same root cause — MoltenVK's subgroup mapping — and account
for every "the GPU produces garbage" report on this hardware class.

## Shared setup (do this once)

```bash
# Homebrew toolchain. vulkan-loader (not libMoltenVK directly) so the driver stays
# swappable at runtime; glslang/shaderc/spirv-headers compile ggml's compute shaders.
brew install cmake libomp \
  vulkan-headers vulkan-loader molten-vk glslang shaderc spirv-headers

# Runtime: point the Vulkan loader at MoltenVK, and pin the discrete GPU.
export VK_ICD_FILENAMES="$(brew --prefix)/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/molten-vk/lib:$(brew --prefix)/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export GGML_VK_VISIBLE_DEVICES=0
```

Confirm the enumeration before anything else — `Vulkan0` must be the Radeon:

```
ggml_vulkan: 0 = AMD Radeon Pro 5500M (MoltenVK) | uma: 0 | fp16: 1 | bf16: 0 |
             warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: none
```

Unset `GGML_VK_VISIBLE_DEVICES` once to check the ordering on your machine; if the UHD 630
enumerates first, pin its index's counterpart instead.

---

# ggml

**Why:** ggml is where the GPU actually is. Both correctness fixes live in its Vulkan
backend, and `test-backend-ops` — its per-operator differential test against a CPU
reference — is the only tool that tells you *which* shader is lying to you. Build this
first when something produces wrong output; it turns "the model is broken" into "`SSM_SCAN`
returns NaN" in about a minute.

**What was modified** ([branch](https://github.com/maximosipov/ggml/tree/macbook-pro-2019-radeon-5500m-4gb)):
- `vulkan: fix MoltenVK/Intel subgroup matmul correctness on RDNA1 and older Intel iGPUs`
  — upstream already disables the subgroup matmul path on Apple for AMD; this extends it
  to Intel, so the UHD 630 falls back to correct non-subgroup shaders (landmine 2).
- `vulkan: fix NaN SSM_SCAN/GATED_DELTA_NET output on MoltenVK/RDNA1` (landmine 4).

Both are no-ops on GPUs where subgroup arithmetic behaves (native Vulkan AMD/NVIDIA, Mesa).

**Setup:**

```bash
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/ggml.git
cd ggml

P="$(brew --prefix)"
cmake -B build -DGGML_VULKAN=1 -DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR="$P/opt/vulkan-headers/include" \
  -DVulkan_LIBRARY="$P/opt/vulkan-loader/lib/libvulkan.dylib" \
  -DVulkan_GLSLC_EXECUTABLE="$P/opt/shaderc/bin/glslc" \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE="$P/opt/glslang/bin/glslangValidator" \
  -DCMAKE_CXX_FLAGS="-I$P/opt/spirv-headers/include" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"
```

Built as the top-level project (not vendored inside llama.cpp), ggml auto-enables
`GGML_BUILD_TESTS` → you get `build/bin/test-backend-ops`.

**Run / evaluate:**

```bash
./build/bin/test-backend-ops -o SSM_SCAN        -b Vulkan0   # the regression test for fix 2
./build/bin/test-backend-ops -o GATED_DELTA_NET -b Vulkan0
./build/bin/test-backend-ops -o MUL_MAT         -b Vulkan0   # the regression test for fix 1
./build/bin/test-backend-ops                    -b Vulkan0   # full sweep, ~10k cases
```

**Benchmarks** — this backend's "benchmark" is a correctness table, which is the point:

| Op | Before | After |
|---|---|---|
| `SSM_SCAN` (Vulkan0, AMD) | FAIL — NaN | **OK, 3/3** |
| `GATED_DELTA_NET` (Vulkan0, AMD) | FAIL — ERR ≈ 1.0 | **OK, 36/36** |
| `MUL_MAT` (Vulkan1, Intel UHD 630) | **every case FAIL**, ERR 2.4–11.4 (tol 5e-4) | OK |
| `MUL_MAT` (Vulkan0, AMD) | OK | OK (unaffected) |
| full suite (Vulkan0) | — | **0 failures** (~10,270 cases) |

The one thing to know: a shader bug here shows up downstream as "this model is broken",
three layers away. `test-backend-ops -b Vulkan0` before blaming a model.

---

# llama.cpp

**Why:** the reference implementation — `llama-cli`, an OpenAI-compatible `llama-server`,
and `llama-bench`, which is the only honest way to compare models on this box. Everything
else here (ollama, LocalAI) is a wrapper around this code, so measure here first.

**What was modified** ([branch](https://github.com/maximosipov/llama.cpp/tree/macbook-pro-2019-radeon-5500m-4gb)):
the same two Vulkan fixes (llama.cpp vendors a synced copy of ggml), plus
`vulkan: add diagnostic env knobs` — `GGML_VK_FORCE_ARCH`, `GGML_VK_RM_KQ`,
`GGML_VK_RM_STDQ`, `GGML_VK_FORCE_INTEGER_DOT`, opt-in, used to isolate the two fixes.

**Setup:**

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

**Run** — note `--flash-attn off` everywhere (landmine 3):

```bash
# text
./build/bin/llama-cli -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -ngl 99 --flash-attn off --ctx-size 4096 -no-cnv --temp 0 \
  -p "How much is 20+34? Answer with just the number."

# OpenAI-compatible server
./build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf \
  -ngl 99 --flash-attn off --ctx-size 8192 --host 127.0.0.1 --port 8080

# vision (Qwen3-VL-4B-Thinking, LLM + projector)
./build/bin/llama-mtmd-cli -m models/Qwen3VL-4B-Thinking-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3VL-4B-Thinking-Q8_0.gguf \
  -ngl 99 --flash-attn off --ctx-size 2048 --image photo.png -p "What does the sign say?"
```

**Models that fit 4 GB**, all Q4_K_M, `llama-bench -ngl 99 -fa off -p 512 -n 128 -r 3`,
post-fix, **thermally gated before each model**:

| Model | Architecture | pp512 tok/s | tg128 tok/s | Fit |
|---|---|---|---|---|
| Gemma-4-E2B | gemma4 PLE | **56.2 ± 0.5** | **41.3 ± 5.2** | 1.45 GB (PLE stays in host RAM) |
| Granite-4.0-H-Micro | Mamba-2 hybrid | 37.1 ± 0.5 | 13.2 ± 1.1 | 1.95 GB, ~2.1 GB headroom |
| Granite-4.1-3B | dense | 36.5 ± 1.1 | 11.7 ± 0.1 | 2.24 GB, KV-bound |
| Qwen3-4B-Instruct-2507 | dense GQA | 34.5 ± 1.0 | 10.5 ± 0.1 | 2.82 GB, worst KV (144 KiB/tok) |
| Qwen3.5-4B | Gated DeltaNet | 27.5 ± 5.9 | 8.9 ± 1.4 | 3.05 GB, ~1.0 GB headroom |
| Gemma-4-E4B | gemma4 PLE | **device lost** | — | 3.3 GB — with FA off the f16 KV pushes pp512 past 4 GB |

The two hybrids in the middle are the models that produced garbage before the shader fix —
they now run correctly. Gemma-4-E4B is the one model that does *not* survive this config:
with flash attention off the KV cache is f16, and a 512-token prefill on top of 3.3 GB of
weights takes the GPU out (`vk::DeviceLostError`). It only runs with FA on, and slowly.

> **These are cold-GPU lower bounds.** Each model was measured from a cooled machine, and
> the Radeon idles at 10 MHz — so a short `tg128` partly measures the clock ramp. The same
> Qwen3-4B-2507 reads **24.9 tok/s** when measured warm, and **27–28 tok/s** under sustained
> serving. Multiply by roughly 2–2.5× for what you'll actually experience, and read the
> [benchmarking warning](#a-warning-about-benchmarking-this-machine) before trusting any
> number here — including these. Treat gaps under ~20% as noise.

**KV cache** (measured, Qwen3.5-4B) — the number that decides your context length:

| KV precision | bytes/token | Max context with zero CPU spill |
|---|---|---|
| f16 | 32.0 KiB | ~41K |
| q8_0 | 17.0 KiB | ~82K |
| q4_0 | 9.0 KiB | **128K** |

Quantized KV requires flash-attn, which costs ~4× throughput — so f16 and a short context
is the fast choice, q4_0 and 128K is the long-context choice, and you can't have both.

**Reproducing the benchmarks:**

```bash
# throughput + prompt processing (one config per process — see the warning below)
./build/bin/llama-bench -m model.gguf -ngl 99 -fa off -p 512 -n 128 -r 3

# GPU vs CPU
./build/bin/llama-bench -m model.gguf -ngl 99 -fa off -n 128 -r 3
./build/bin/llama-bench -m model.gguf -ngl 0  -fa off -n 128 -r 3

# context ceiling: raise --ctx-size until the load log shows a CPU buffer appearing
./build/bin/llama-server -m model.gguf -ngl 99 -fa off --ctx-size 40960 2>&1 | grep -E "KV|buffer size"
```

### A warning about benchmarking this machine

This is a 2019 laptop and it will lie to you if you let it. The same binary, model and
flags produced **10.5, 24.9 and 33.9 tok/s** across three sessions. None of those runs was
buggy; they were different regimes. Three effects, in order of size:

- **GPU clock state — dominant, ~2.4×.** `ioreg` reports the Radeon idling at **10 MHz**
  core clock, and `llama-bench`'s single warmup pass doesn't spin it up. The decisive test,
  same binary and flags back-to-back:

  | GPU | CPU | tg128 |
  |---|---|---|
  | warm (straight off a serving workload) | **throttled to 30** | **24.92 ± 8.10** |
  | cold (after cooling to 100) | unthrottled | 10.49 ± 0.11 |

  The run on the *throttled* CPU is 2.4× faster. Cooling the machine to avoid CPU
  throttling **parks the GPU**, so the intuitive benchmarking hygiene is backwards here.
- **Position within a multi-config run.** `llama-bench -fa 0,1` gives the first config a
  different machine than the second. That alone produced an apparent "3.9× flash-attention
  penalty" that a counterbalanced experiment puts at **2.27×**.
- **CPU thermal throttling.** `CPU_Speed_Limit` falls to **20–36** within ~2 minutes of
  load. Real — but smaller than the GPU effect and pointing the other way.

So: **warm the GPU with a discarded run, use one process per config, and alternate the
order of what you're comparing.** Report the per-round values, not just the mean — the
spread is what tells you whether a difference is real. Never difference two tables produced
on different days; answer config questions with an A/B.

Two caveats to that advice, both learned the hard way. A warm-up pass is **not free**:
adding one triggered `vk::DeviceLostError` on Qwen3.5-4B, so a more representative number
also means a less likely finish. And watch for **orphaned benchmark processes** — one
outlived a `pkill` of its parent script and held the machine at `CPU_Speed_Limit=36` while
the next run sat in a cooldown loop, waiting for a machine that another process was busy
heating.

Honestly, for "what will I actually get?", **measure sustained serving, not micro-benchmarks.**
LocalAI serving Qwen3-4B-2507 returned 27.1 / 27.9 / 28.4 tok/s across three runs — tighter
than any `llama-bench` figure on this box, and it's the number you actually feel.

**Capability benches** (sampled, over `llama-server`'s OpenAI endpoint). These are
**relative rankings on this hardware under one protocol — not leaderboard-comparable**;
sample sizes are small and quantization/subset/protocol all differ from the published
runs:

| Bench | Measures | Qwen3.5-4B | Granite-4.0-H-Micro | Qwen3-4B-2507 |
|---|---|---|---|---|
| BFCL-V4 AST (n=80) | tool/function calling | 0.750 | 0.863 | **0.875** |
| MMLU-Pro (n=28, 0-shot CoT) | 10-choice reasoning | **0.464** | 0.393 | 0.429 |
| LongBench-v2 (n=3, ~12K tok) | long-document QA | timeout | 0/3 | timeout |
| MRCR v2 (n=2, ~19K tok) | multi-round coref recall | 0.00 | 0.014 | timeout |
| custom tool set (n=24) | tool calls incl. negatives | 23/24 | **24/24** | 22/24 |

The useful signal isn't the scores, it's the timeouts: **long context is a hardware wall
here**. Only the Mamba-2 hybrid finishes a 12–19K-token sample. Gated DeltaNet's
sequential-scan prefill becomes thousands of tiny MoltenVK dispatches, and the dense model
needs quantized KV, which forces the ~2.3×-slower flash-attn path.

---

# ollama

**Why:** the friendly wrapper — `ollama pull`/`ollama run`, a model library, a desktop app.
Official macOS Ollama ships **no Vulkan runner**, so on this machine the Radeon shows up as
`id=cpu` ([ollama#13127](https://github.com/ollama/ollama/issues/13127)) and you get CPU
speeds. This branch builds the runner that upstream doesn't ship.

**What was modified** ([branch](https://github.com/maximosipov/ollama/tree/macbook-pro-2019-radeon-5500m-4gb), 5 patches):

| Patch | What it fixes |
|---|---|
| `vulkan-visible-device-index` | ollama's device list counts CPU as index 0, so it passed a CPU-inclusive index where ggml expects a **Vulkan-relative** one — it asked for the Radeon and got the UHD 630 (landmine 2, the actual cause of the gibberish) |
| `image-min-tokens-env` | upstream hardcodes 1024 image tokens for Qwen-VL; on a thermally limited laptop that inflates prefill until sustained throughput collapses. `OLLAMA_IMAGE_MIN_TOKENS=512` holds ~32 tok/s and still scores 4/4 |
| `gpu-percent-slider` | `OLLAMA_GPU_PERCENT` — a global CPU/GPU split default (percentage of layers), plus a "GPU offload" slider in the desktop app's Settings |
| `bundle-vulkan-env` | a packaged `.app` sets the MoltenVK environment itself, so no shell wrapper is needed |
| `raw-protocol-tab` | a debug tab in the desktop chat UI showing the literal `/api/chat` request, the rendered prompt, and the streamed response chunks |

**Setup** — ollama vendors a *pinned* llama.cpp commit and its compat patches are written
against that exact commit, so you don't build against the fork's branch tip. Check out the
pinned commit from the same fork and cherry-pick the three fix commits onto it:

```bash
brew install go node   # go >= 1.26

git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/llama.cpp.git llama.cpp-radeon
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/ollama.git ollama-radeon

cd llama.cpp-radeon
LLAMA_TAG="$(cat ../ollama-radeon/LLAMA_CPP_VERSION)"
LLAMA_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git "refs/tags/$LLAMA_TAG" | cut -f1)"
git fetch https://github.com/ggml-org/llama.cpp.git "$LLAMA_SHA"
git checkout -b radeon-for-ollama "$LLAMA_SHA"
git cherry-pick 7847fca13 595936bcd a91f29c5f     # the 3 Vulkan fixes, oldest first
cd ..

cp -R llama.cpp-radeon llama.cpp-radeon-ollama
( cd llama.cpp-radeon-ollama && \
  git apply ../ollama-radeon/llama/compat/*.patch ../ollama-radeon/llama/compat/models/*.patch )
```

Then build. ollama's CMake doesn't forward `-DVulkan_*` into its nested llama.cpp build, so
a fake "Vulkan SDK" of symlinks is how you point it at MoltenVK (`FindVulkan` honours
`$VULKAN_SDK`):

```bash
cd ollama-radeon
P="$(brew --prefix)"
mkdir -p vulkan-sdk/include vulkan-sdk/lib vulkan-sdk/bin
ln -sfn "$P/opt/vulkan-headers/include/vulkan"       vulkan-sdk/include/vulkan
ln -sfn "$P/opt/spirv-headers/include/spirv"         vulkan-sdk/include/spirv
ln -sfn "$P/opt/vulkan-loader/lib/libvulkan.dylib"   vulkan-sdk/lib/libvulkan.dylib
ln -sfn "$P/opt/vulkan-loader/lib/libvulkan.1.dylib" vulkan-sdk/lib/libvulkan.1.dylib
ln -sfn "$P/opt/shaderc/bin/glslc"                   vulkan-sdk/bin/glslc
ln -sfn "$P/opt/glslang/bin/glslangValidator"        vulkan-sdk/bin/glslangValidator
export VULKAN_SDK="$PWD/vulkan-sdk"
export OLLAMA_LLAMA_CPP_SOURCE="$PWD/../llama.cpp-radeon-ollama"

cmake -B build -DOLLAMA_LLAMA_BACKENDS="vulkan" -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel "$(sysctl -n hw.ncpu)"
```

The desktop app + DMG (needed to see the GPU slider and the raw-protocol tab) is a further
half-page of `npm run build`, bundle assembly, Vulkan self-containment relinking and ad-hoc
codesigning — see
[the branch runbook](https://github.com/maximosipov/ollama/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md#part-2--desktop-app--dmg-needed-to-see-the-gpu-slider-and-raw-protocol-tab).

**Run:**

```bash
export VK_ICD_FILENAMES="$(brew --prefix)/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/molten-vk/lib:$(brew --prefix)/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export OLLAMA_VULKAN=1
export GGML_VK_VISIBLE_DEVICES=0     # load-bearing — see landmine 2
export OLLAMA_FLASH_ATTENTION=0      # load-bearing — see landmine 3
export OLLAMA_IMAGE_MIN_TOKENS=512   # vision only
export OLLAMA_HOST=127.0.0.1:11435

./build/ollama serve &
./build/ollama run qwen3:1.7b "How much is 20+34?"
./build/ollama ps                    # PROCESSOR must read 100% GPU
```

**Benchmarks:**

| Workload | Result |
|---|---|
| Qwen3-1.7B text generation | **67.0 tok/s**, 100% GPU |
| Qwen3-VL-4B vision, `OLLAMA_IMAGE_MIN_TOKENS=512` | **32.0 tok/s**, 4/4 grounded |
| Qwen3-VL-4B vision, upstream default (1024) | 8.3–15.3 tok/s, decays across requests |
| VRAM, Qwen3-VL-4B | ~3.2–3.6 GiB of 4080 MiB, no CPU spill |

Device choice, same model (Qwen3-1.7B), same build — the reason for the pin:

| Device | Generation |
|---|---|
| AMD Radeon Pro 5500M | **71 tok/s** |
| CPU | 24 tok/s |
| Intel UHD 630 | 7 tok/s (and numerically wrong before the matmul fix) |

**Never run inference on the UHD 630.** It is slower than the CPU — it shares system DRAM
*and* pays MoltenVK overhead. Order of preference: AMD, then CPU, never the iGPU.

Reproduce with `ollama ps` for placement and the timing fields the API returns:

```bash
curl -s http://127.0.0.1:11435/api/generate -d '{"model":"qwen3:1.7b","prompt":"...","stream":false}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["eval_count"]/(d["eval_duration"]/1e9), "tok/s")'
```

Per-request override of the CPU/GPU split: `"options": {"num_gpu": 0}` (CPU) / `999` (all
layers), or globally with `OLLAMA_GPU_PERCENT=0|100` / the desktop slider.

---

# stable-diffusion.cpp

**Why:** image generation on the same ggml/Vulkan stack. It's the one workload where the
Metal path doesn't merely underperform — it doesn't complete at all.

**What was modified:** **nothing.** stable-diffusion.cpp builds clean for this hardware.
Everything needed is build flags and run flags — which is exactly why they need writing
down, because two of them are non-obvious and both produce silently wrong output.

**Setup:**

```bash
git clone --recursive https://github.com/maximosipov/stable-diffusion.cpp.git
cd stable-diffusion.cpp

P="$(brew --prefix)"
cmake -B build-vulkan \
  -DSD_VULKAN=ON -DSD_METAL=OFF -DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR="$P/opt/vulkan-headers/include" \
  -DVulkan_LIBRARY="$P/opt/vulkan-loader/lib/libvulkan.dylib" \
  -DVulkan_GLSLC_EXECUTABLE="$P/opt/shaderc/bin/glslc" \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE="$P/opt/glslang/bin/glslangValidator" \
  -DOpenMP_ROOT="$P/opt/libomp" \
  -DCMAKE_CXX_FLAGS="-I$P/opt/spirv-headers/include" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j"$(sysctl -n hw.ncpu)"
```

`-DGGML_METAL=OFF` alongside `-DSD_METAL=OFF` is the load-bearing part: ggml auto-enables
Metal on macOS, so `-DSD_METAL=OFF` alone still compiles Metal in, and it wins device 0 at
runtime. Your "Vulkan" build silently runs Metal and times out.

**Run:**

```bash
# SD-Turbo — the daily driver. 4 steps, cfg 1.
./build-vulkan/bin/sd-cli --diffusion-conv-direct \
  -m models/sd_turbo.safetensors --type q8_0 \
  -p "a red apple on a wooden table, studio lighting" \
  --steps 4 --cfg-scale 1 --sampling-method euler -W 512 -H 512 --seed 42 -o out.png

# SDXL-Turbo — better photorealism, at the edge of 4 GB.
./build-vulkan/bin/sd-cli --diffusion-conv-direct --vae-on-cpu \
  -m models/sd_xl_turbo_1.0_fp16.safetensors --type q8_0 \
  -p "..." --steps 4 --cfg-scale 1 --sampling-method euler -W 512 -H 512 -o out.png

# SD 1.5 — baseline, best LoRA/ControlNet ecosystem. 20 steps, cfg 7.
./build-vulkan/bin/sd-cli --diffusion-conv-direct \
  -m models/v1-5-pruned-emaonly-fp16.safetensors \
  -p "..." --steps 20 --cfg-scale 7 -W 512 -H 512 -o out.png
```

**Backend comparison** (SD 1.5 fp16, 512×512, seed 42, identical prompt):

| Backend | Fits? | Correct? | Speed |
|---|---|---|---|
| Metal | targets the AMD, 4278 MB set | ❌ **GPU watchdog timeout** | — (shader library alone takes 40–1000 s to load) |
| Vulkan, no extra flags | ✅ ~2.6 GB | ❌ **colourful noise** | ~11 s/step |
| **Vulkan + `--diffusion-conv-direct`** | ✅ ~2.6 GB | ✅ **matches CPU** | **~3.4 s/step** |
| CPU (ground truth) | n/a | ✅ | ~24 s/step |

"Matches CPU" means verified against the CPU reference image, not "looks like an image".
GPU is ~7× CPU, and the correct GPU path is ~3× faster than the broken one.

**Model fit and throughput:**

| Model | Load config | VRAM | s/step | Notes |
|---|---|---|---|---|
| SD-Turbo | `--type q8_0` | **2049 MB** | **~1.7** | comfortable; 4-step image in ~7 s compute |
| SD 1.5 | fp16 | 2035 MB | ~3.4 | baseline, 20 steps |
| SDXL-Turbo | `--type q8_0 --vae-on-cpu` | **3836 MB** | ~9.3 | fits, but tight against 4278 MB |

Recommendation: **SD-Turbo q8_0** for iteration, **SDXL-Turbo q8_0 + CPU VAE** for hero
shots (clearly more photoreal, ~5× slower per step, and at 4-step/cfg-1 it drops prompt
detail more often), **SD 1.5** when you need the LoRA/ControlNet ecosystem.

Other landmines on this card: **bf16 weights produce NaN** (this GPU is `bf16: 0`) — use
fp16 or GGUF quants; **`--diffusion-fa` only works on NVIDIA coopmat2** — leave it off;
peak VRAM is at **VAE decode**, so `--vae-on-cpu` is both the SDXL fp16-NaN fix and the
VRAM lever.

**Reproducing:** run the same prompt+seed on `--backend cpu` and diff the images; time with
sd.cpp's own per-step log. Wall clock includes model load and on-the-fly quantization
(~30 s for SD-Turbo), which dominates a 4-step generation — use `sd-server` to amortize it.

---

# LocalAI

**Why:** the drop-in **OpenAI API server** — point any OpenAI-SDK client at it and it
works, with a model gallery and a web UI on top. It's the only one of the five that gives
you `/v1/chat/completions` plus model management without being an app.

**What was modified:** no source changes are needed, but LocalAI's macOS build path targets
Apple Silicon + Metal, so getting a Vulkan backend requires composing two pieces yourself:
its llama backend is a **separate gRPC server process** built from a pinned llama.cpp, and
you have to substitute a fixed one.

*Verified with LocalAI `ecdb321`, its pinned llama.cpp `1cbfd1988`, Homebrew gRPC 1.83 /
protobuf 35.1, Go 1.26.5, Node 24.*

**Setup:**

```bash
brew install go node cmake grpc protobuf abseil

git clone https://github.com/maximosipov/LocalAI.git && cd LocalAI
make build                          # Go server + React UI -> ./local-ai
```

The backend's llama.cpp source must be **LocalAI's pinned commit with the three Vulkan
fixes cherry-picked onto it** — not the fork's branch tip. LocalAI's `grpc-server.cpp` is
written against that exact commit and llama.cpp's master drifts within days:

```bash
PIN="$(sed -n 's/^LLAMA_VERSION?=//p' backend/cpp/llama-cpp/Makefile)"
git clone https://github.com/maximosipov/llama.cpp.git backend/cpp/llama-cpp/llama.cpp
cd backend/cpp/llama-cpp/llama.cpp
git fetch https://github.com/ggml-org/llama.cpp.git master --tags
git checkout -b build "$PIN"
git cherry-pick 7847fca13 595936bcd a91f29c5f     # oldest first
git submodule update --init --recursive --depth 1
cd ../../../..

# graft LocalAI's grpc-server into llama.cpp/tools/ (run once; it is not idempotent)
( cd backend/cpp/llama-cpp && mkdir -p llama.cpp/tools/grpc-server && bash prepare.sh )
```

Then build the backend. **`CMAKE_ARGS` must go through the environment**, not the make
command line — the Makefile does `CMAKE_ARGS?=` then `CMAKE_ARGS+=…`, and a command-line
variable would override those appends and drop `-DGGML_VULKAN=1` itself:

```bash
P="$(brew --prefix)"
export BUILD_TYPE=vulkan
export CMAKE_ARGS="-DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR=$P/opt/vulkan-headers/include \
  -DVulkan_LIBRARY=$P/opt/vulkan-loader/lib/libvulkan.dylib \
  -DVulkan_GLSLC_EXECUTABLE=$P/opt/shaderc/bin/glslc \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE=$P/opt/glslang/bin/glslangValidator \
  -DOpenMP_ROOT=$P/opt/libomp \
  -DCMAKE_CXX_FLAGS=-I$P/opt/spirv-headers/include \
  -DCMAKE_BUILD_TYPE=Release"
make -C backend/cpp/llama-cpp grpc-server
```

`-DGGML_METAL=OFF` is required here for a subtle reason: in that Makefile `BUILD_TYPE=vulkan`
and the Darwin branch are arms of **one if/else chain**, so choosing vulkan on macOS means
the Darwin arm — the one that would have set `GGML_METAL=OFF` — never runs. Gate the build
with `otool -L backend/cpp/llama-cpp/grpc-server | grep -i metal` returning nothing.

**Run.** Register the Vulkan `grpc-server` as an external backend. LocalAI spawns it as a
child process, so wrap it in a script that carries the MoltenVK environment:

```bash
cat > backend/cpp/llama-cpp/run-vulkan.sh <<'EOF'
#!/usr/bin/env bash
P="$(brew --prefix)"
export VK_ICD_FILENAMES="$P/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$P/opt/molten-vk/lib:$P/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export GGML_VK_VISIBLE_DEVICES=0
exec "$(dirname "$0")/grpc-server" "$@"
EOF
chmod +x backend/cpp/llama-cpp/run-vulkan.sh

# The GGUF must live INSIDE --models-path. LocalAI rejects a path outside it as
# "invalid file path" during config validation, and the model then just silently
# doesn't appear in /v1/models. A symlink satisfies it.
mkdir -p models && ln -s /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf models/

cat > models/qwen3-4b.yaml <<'EOF'
name: qwen3-4b
backend: llama-cpp
parameters:
  model: Qwen3-4B-Instruct-2507-Q4_K_M.gguf
context_size: 4096
f16: true
gpu_layers: 99
flash_attention: "false"      # landmine 3
EOF

./local-ai run --address 127.0.0.1:8085 --models-path ./models \
  --external-grpc-backends "llama-cpp:$PWD/backend/cpp/llama-cpp/run-vulkan.sh"
```

```bash
curl -s http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b",
  "messages": [{"role":"user","content":"How much is 20+34? Answer with just the number."}]
}' | python3 -m json.tool
```

**Verifying it's actually on the GPU is the hard part here.** LocalAI's own hardware probe
reports `GPU vendor=""` and `Total available VRAM 0` on macOS, and it does not forward the
backend's ggml init banner into its log — so nothing LocalAI prints tells you where the
model landed. Ask IOKit instead:

```bash
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

A 4B Q4_K_M model resident on the Radeon shows **~3.4 GiB** in use on the AMD accelerator
node. If that stays near idle while generation runs, the model is on the CPU — check that
`gpu_layers` survived the load (`LOCALAI_DISABLE_HARDWARE_DEFAULTS=true` disables LocalAI's
auto-tuning if it's overriding you).

One more thing that looks like a failure and isn't: with a **thinking** model (Qwen3 and
friends), `choices[].message.content` comes back **empty** and the chain of thought is in
`choices[].message.reasoning`. `content` only fills once the model closes the think block,
so an empty `content` with `finish_reason: "length"` is a token budget problem — not the
MoltenVK garbage-output failure it resembles.

**Benchmarks** — Qwen3-4B-Instruct-2507 Q4_K_M, ctx 4096, f16 KV, FA off, `gpu_layers: 99`:

| Measure | Result |
|---|---|
| Correctness battery over `/v1/chat/completions` | **4/4** (arithmetic, factual, sequence, string reversal) |
| Generation, 128 tokens, 3 runs | **27.1 / 27.9 / 28.4 tok/s** |
| VRAM resident (IOKit, during generation) | **3.35–3.50 GiB** of 4080 MiB |
| GPU utilisation during generation | 82% |
| Model load → first token | ~0.9 s for a short prompt |

Those three throughput runs are the **tightest numbers in this whole README** (spread under
5%), which is why the benchmarking section above recommends sustained serving over
micro-benchmarks on this hardware.

Reproduce:

```bash
# correctness + throughput over the OpenAI endpoint
python3 - <<'PY'
import json, time, urllib.request
body = json.dumps({"model": "qwen3-4b",
                   "messages": [{"role":"user","content":"Write a detailed paragraph about how a GPU renders a triangle."}],
                   "max_tokens": 128, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8085/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(req))
dt = time.perf_counter() - t0
print(f'{r["usage"]["completion_tokens"]/dt:.1f} tok/s')
PY

# GPU residency, in another shell while the above runs
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

---

## Negative results worth knowing

- **[llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104)** (Intel-Mac AMD
  Vulkan gibberish) **does not reproduce** on a current checkout: the correctness battery
  passes and GPU perplexity matches CPU within 0.05%. The gibberish people hit on this
  hardware today is landmine 2 (wrong device selected) or landmine 4 (hybrid shaders), both
  fixed on these branches.
- **The Intel UHD 630 is never the right answer.** Slower than the CPU at every size tested.
  The matmul fix on the ggml/llama.cpp branches exists to stop it being *silently wrong*,
  not to make it useful.
- **Metal was measured, not assumed.** It is broken differently for LLMs (PCIe re-reads,
  ~0.8 tok/s) than for diffusion (watchdog timeout, no output at all).
- **CLIP prompt-adherence scoring for the diffusion bake-off was not run** — the visual gap
  between SD-Turbo and SDXL-Turbo was decisive without it.
- **Flash attention does *not* reverse for state-space models.** A cold-GPU run made
  Qwen3.5-4B (Gated DeltaNet) look like it preferred FA on, which would have been a neat
  architecture-specific finding. A counterbalanced A/B says otherwise: `fa on` 18.72/9.88,
  `fa off` 21.82/15.09 — the rounds disagree on direction and the standard deviations are a
  third of the means. For that model on this stack the effect is **inside the noise**. The
  dense model, measured identically, gave sd 0.01. Report the spread, not just the mean.
- **The Radeon can lose the device**, in two distinct ways: Gemma-4-E4B runs out of VRAM at
  `pp512` with FA off (f16 KV on top of 3.3 GB of weights), and Qwen3.5-4B dies with
  `vk::DeviceLostError` under back-to-back benchmark load. Neither is a correctness bug —
  but budget for it in any automated sweep.
