# ggml

The tensor library underneath every other tool here, and the place where GPU bugs are found and fixed.

| | |
|---|---|
| **Role in the stack** | Tensor library and Vulkan backend |
| **Use it for** | Diagnosing wrong output, and developing fixes |
| **Fork and branch** | [`maximosipov/ggml`, `macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/ggml/tree/macbook-pro-2019-radeon-5500m-4gb) |
| **Upstream** | [ggml-org/ggml](https://github.com/ggml-org/ggml) |
| **Source changes** | 9 commits in `ggml-vulkan.cpp` and 3 shaders |
| **Also included in** | llama.cpp (vendored copy, [same fixes](llama.cpp.md)) |
| **Runbook on the branch** | [RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md](https://github.com/maximosipov/ggml/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) |

Complete the [shared setup](README.md#shared-setup) first. Terms are defined in the [glossary](glossary.md).

## Overview

ggml is a tensor library in plain C with no dependencies. It is not an application; it provides the parts a runtime is built from:

- **Tensor operations and the compute graph** — matrix multiplication, attention (`FLASH_ATTN_EXT`), convolution, normalization, RoPE, and the recurrent operations (`SSM_SCAN`, `GATED_DELTA_NET`) that hybrid models need. A model is a graph of these operations.
- **The backend system** — CPU, [Vulkan](glossary.md#g-vulkan), Metal, CUDA, HIP, SYCL, OpenCL, BLAS and RPC. A scheduler splits the graph across the available devices, and falls back to the CPU for any operation a backend doesn't implement. This is why a broken shader degrades quietly instead of failing loudly.
- **[GGUF](glossary.md#g-ggml) and quantization** — the model file format, the k-quant and i-quant implementations such as [`Q4_K_M`](glossary.md#g-quant), and the dequantization kernels each backend needs.
- **[`test-backend-ops`](glossary.md#g-test-backend-ops)** — a differential tester that runs each operation on a backend and compares the result against the CPU.

**Why it matters here:** the GPU code lives in ggml, so all the correctness fixes are made in its Vulkan backend. `test-backend-ops` is the only tool that will tell you *which* operation is returning wrong values. It turns "this model is broken" into "`SSM_SCAN` returns NaN" in about a minute. Fixes made here propagate to llama.cpp, ollama and LocalAI.

## Changes on this branch

Nine commits, touching `ggml-vulkan.cpp` and three `.comp` shaders. All are conditional on the MoltenVK driver, so they do nothing on GPUs where subgroup operations behave as specified (native Vulkan on AMD or NVIDIA, or Mesa).

**Correctness** — each of these produced wrong output rather than a crash:

| Commit | What it fixes |
|---|---|
| `fix MoltenVK/Intel subgroup matmul correctness…` | Upstream already disables the subgroup matmul path on Apple for AMD; this extends it to Intel GPUs. Defensive: it makes a mis-targeted Intel GPU produce correct results instead of silent garbage. |
| `fix NaN SSM_SCAN/GATED_DELTA_NET output…` | Both shaders drove a shared-memory reduction from `gl_SubgroupInvocationID`, assuming a workgroup is one contiguous [subgroup](glossary.md#g-subgroup). MoltenVK doesn't guarantee that, so hybrid models emitted [NaN](glossary.md#g-nan). Now indexed by `gl_LocalInvocationID.x`. |
| `disable subgroup clustered ops on MoltenVK…` | `quantize_q8_1` reduced 8-lane blocks with `subgroupClusteredMax` and `subgroupClusteredAdd`, which MoltenVK doesn't map to the intended lanes. That corrupted the quantized activations both integer-dot matmul paths consume. |
| `do not use mul_mat_vecq for q8_0 on MoltenVK` | The multi-column variant is wrong for q8_0 (error near 1.0), while the single-column and `mul_mmq` paths are correct. MoltenVK 1.4.2 doesn't fix this, so the guard is unconditional. |
| `keep q8_0 off the flash-attention MMQ path…` | The same defect in a second place. See [Troubleshooting](#troubleshooting). |

Both q8_0 defects have the same root cause: the affected shaders repack through `pack32(i16vec2(...))` from a 16-bit view, which MoltenVK mistranslates. Every other quantization type packs from `u16vec2` and is unaffected.

**Performance:**

| Commit | Effect |
|---|---|
| `enable emulated integer dot for matmul…` | **+68% [prefill](glossary.md#g-prefill)**, generation unchanged. Metal has no DP4a-style instruction, but emulating it still beats the generic path. |
| `run flash attention on the GPU under MoltenVK` | Uses a flash-attention path that needs no subgroup operations, so ggml stops scheduling attention on the CPU |
| `trust MoltenVK subgroups from 1.4.2 onwards` | Switches three of the workarounds off when `driverVersion >= 10402`, since the driver bug they compensate for is fixed |

A ninth commit adds `GGML_VK_FA_NO_SUBGROUPS`, a diagnostic switch used while isolating the flash-attention fallback.

## Build

The fork is a submodule of this repository, already on the right branch:

```bash
git submodule update --init ggml
cd ggml
```

To build it on its own instead:

```bash
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/ggml.git
cd ggml
```

Then, in either case:

```bash
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

`-DGGML_METAL=OFF` is required, not optional ([known issue](README.md#ki-metal)). Built as the top-level project, ggml enables its tests automatically, which produces `build/bin/test-backend-ops`.

## Run

`test-backend-ops` is the tool this build exists for:

```bash
./build/bin/test-backend-ops -o SSM_SCAN        -b Vulkan0   # hybrid-model NaN regression test
./build/bin/test-backend-ops -o GATED_DELTA_NET -b Vulkan0   # same
./build/bin/test-backend-ops -o MUL_MAT         -b Vulkan0   # matmul, including the integer-dot path
./build/bin/test-backend-ops -o FLASH_ATTN_EXT  -b Vulkan0   # attention
./build/bin/test-backend-ops                    -b Vulkan0   # everything: ~15k cases, about 35 minutes

./build/bin/test-backend-ops support -o FLASH_ATTN_EXT -b Vulkan0   # is the operation run on the GPU at all?
./build/bin/test-backend-ops perf    -o MUL_MAT        -b Vulkan0   # per-operation timing
```

## Verify

**Read the case counts, not the `OK`.** An unsupported operation is *skipped*, and a backend where every case was skipped still prints a green `OK`. That is exactly how the flash-attention CPU fallback stayed hidden: `-o FLASH_ATTN_EXT` looked like it passed, while `support -o FLASH_ATTN_EXT` reported 0 of 5097 cases supported.

On the branch tip with MoltenVK 1.4.2, the Radeon passes everything it runs:

| Operation | Result |
|---|---|
| `SSM_SCAN` | 3 / 3 |
| `GATED_DELTA_NET` | 36 / 36 |
| `MUL_MAT` | 982 / 982 |
| `FLASH_ATTN_EXT` | 4757 / 4757 |
| Full sweep | **15426 / 15426, zero failures** |

A further ~3276 cases report `not supported` and are skipped. That's normal: they are operations or type combinations the Vulkan backend doesn't implement and hands to the CPU.

Case totals move as upstream changes, so compare any failure count against a baseline you produced the same day, not against a number quoted in a document.

## Performance

`test-backend-ops perf` gives per-operation timings, but the results that matter are visible at the model level. Measured through llama.cpp on a 4B Q4_K_M model:

| Change | Effect |
|---|---|
| Emulated integer dot | Prefill **+68%** (about 35 → 59 tok/s for pp512); generation unchanged |
| Flash attention on the GPU | Generation **10.2 → 40.4 tok/s** (tg128) |
| Quantized KV cache, now usable | About 4% slower, and the cache shrinks from 32 to 9 KiB per token |

See [benchmarking.md](benchmarking.md) before comparing any two numbers from this machine.

## Troubleshooting

**A model produces gibberish or NaN.** Run `test-backend-ops -b Vulkan0` before blaming the model. A shader bug surfaces three layers away as "this model is broken".

**An operation reports `OK` but is slow.** Check `test-backend-ops support -o <OP> -b Vulkan0`. If no case is supported, ggml is running that operation on the CPU.

**The Intel UHD 630 still fails on MoltenVK 1.4.2.** `mvk_subgroups_trustworthy()` switches the subgroup workarounds off from driver 1.4.2 onward for every device. That conclusion was reached on the Radeon and does not hold for the Intel GPU, which still fails `SSM_SCAN` and `GATED_DELTA_NET` there. Keep the Intel GPU pinned out with `GGML_VK_VISIBLE_DEVICES`; if anything must run on it, force the workarounds back on with `GGML_VK_FORCE_MOLTENVK_WORKAROUNDS=1`.

**A performance patch can turn a correct path into a fast, wrong one.** This branch did exactly that. Enabling integer dot made `ggml_vk_fa_scalar_uses_mmq` accept q8_0, which routed the q8_0 KV cache onto a block loader MoltenVK mistranslates. Upstream, that path was never reached and q8_0 was correct but slower. The result was 337 `FLASH_ATTN_EXT` failures with an error of 0.04–0.09, all with `type_K=q8_0`, while q8_0 passed every other operation. Only a per-operation differential test catches this class of bug.

## Reference

Environment variables added or used by this branch:

| Variable | Effect |
|---|---|
| `GGML_VK_VISIBLE_DEVICES` | Restricts which Vulkan devices are enumerated. `0` exposes only the Radeon. |
| `GGML_VK_FORCE_MOLTENVK_WORKAROUNDS` | Keeps the subgroup workarounds active regardless of driver version |
| `GGML_VK_NO_MOLTENVK_WORKAROUNDS` | Disables them regardless of driver version |
| `GGML_VK_FA_NO_SUBGROUPS` | Forces the subgroup-free flash-attention path |

llama.cpp's copy of ggml adds more diagnostic switches; see [llama.cpp.md](llama.cpp.md#reference).

- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [ggml capability line, field by field](hardware.md#reading-the-capability-line)
