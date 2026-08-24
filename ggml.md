# ggml — build, patch and test recipe

Part of [AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)](README.md). Do the [shared setup](README.md#3-shared-setup-do-this-once) first; every term used here is defined in the [glossary](README.md#12-glossary). What ggml is and why it is the first thing to build: [chapter 4](README.md#4-ggml). The correctness results this recipe produces: [4.4 Benchmarks](README.md#44-benchmarks).

Branch: [`macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/ggml/tree/macbook-pro-2019-radeon-5500m-4gb).

## What was modified

Six changes to `ggml-vulkan.cpp` and three `.comp` shaders. Four fix wrong output; two lift a software ceiling.

**Correctness:**
- `vulkan: fix MoltenVK/Intel subgroup matmul correctness on RDNA1 and older Intel iGPUs` — upstream already disables the subgroup matmul path on Apple for AMD; this extends it to Intel. Purely defensive: it makes a mis-targeted integrated GPU produce correct results instead of silent garbage. The device is excluded here anyway (landmine 2) — the patch exists so that *if* something slips past the pin, it fails visibly rather than quietly.
- `vulkan: fix NaN SSM_SCAN/GATED_DELTA_NET output on MoltenVK/RDNA1` (landmine 4).
- `vulkan: disable subgroup clustered ops on MoltenVK (AMD/Intel)` — `quantize_q8_1` reduced 8-lane blocks with `subgroupClusteredMax`/`Add`, which MoltenVK does not map to the intended lanes, corrupting the activations both integer-dot matmul paths consume. Same family as the two above.
- `vulkan: do not use mul_mat_vecq for q8_0 on MoltenVK` — the multi-column variant is wrong for q8_0 (n=2..9, err ≈ 1.0) while `n=1` and the `mul_mmq` path are correct. 1.4.2 does not fix it, so the guard is unconditional. Root cause since identified: `pack32(i16vec2(...))` from a 16-bit view, the same construct behind the flash-attention defect below.
- `vulkan: keep q8_0 off the flash-attention MMQ path on MoltenVK` — the last correctness failure on this card, and one this branch caused itself by enabling integer dot ([4.4](README.md#44-benchmarks)).

**Performance:**
- `vulkan: enable emulated integer dot for matmul on MoltenVK/AMD` — **+68% [prefill](README.md#g-prefill)**, decode unchanged. See [Reading the ggml capability line](README.md#reading-the-ggml-capability-line) for why `int dot: 0` was never the whole story.
- `vulkan: run flash attention on the GPU under MoltenVK` (landmine 3).
- `vulkan: trust MoltenVK subgroups from 1.4.2 onwards` — gates the three subgroup workarounds on `driverVersion >= 10402`, with `GGML_VK_FORCE_MOLTENVK_WORKAROUNDS` / `GGML_VK_NO_MOLTENVK_WORKAROUNDS` overriding in either direction.

All are no-ops on GPUs where [subgroup](README.md#g-subgroup) arithmetic behaves (native Vulkan AMD/NVIDIA, Mesa) — they are keyed on the MoltenVK driver, and on 1.4.2 three of them switch themselves off.

## Setup

This fork is a submodule of this repo at `ggml/`, already on the right branch — `git submodule update --init ggml` and `cd ggml` is enough. The standalone clone is for building it on its own:

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

Built as the top-level project (not vendored inside llama.cpp), ggml auto-enables `GGML_BUILD_TESTS` → you get `build/bin/test-backend-ops`.

## Run / evaluate

```bash
./build/bin/test-backend-ops -o SSM_SCAN        -b Vulkan0   # regression test for the NaN fix
./build/bin/test-backend-ops -o GATED_DELTA_NET -b Vulkan0   # same
./build/bin/test-backend-ops -o MUL_MAT         -b Vulkan0   # matmul + clustered + integer dot
./build/bin/test-backend-ops -o FLASH_ATTN_EXT  -b Vulkan0   # landmine 3
./build/bin/test-backend-ops                    -b Vulkan0   # full sweep, ~15k cases (~35 min)
```

**Read the case counts, not the `OK`.** An op that is *unsupported* is skipped, and a backend with every case skipped still prints a green `OK`. That is exactly how the flash-attention CPU fallback stayed hidden for months: `-o FLASH_ATTN_EXT` looked like it passed while `test-backend-ops support -o FLASH_ATTN_EXT` reported **0 of 5097 cases supported**.

What the numbers should come back as — and the story behind the 337 failures that are now gone — is in [4.4 Benchmarks](README.md#44-benchmarks).

The one thing to know: a [shader](README.md#g-shader) bug here shows up downstream as "this model is broken", three layers away. `test-backend-ops -b Vulkan0` before blaming a model.
