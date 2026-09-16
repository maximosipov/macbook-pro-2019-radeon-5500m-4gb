# stable-diffusion.cpp

Image generation on the same GPU stack. No source changes are needed, but two flags decide whether it works at all.

| | |
|---|---|
| **Role in the stack** | Diffusion runtime on top of ggml |
| **Use it for** | Text-to-image generation from the command line |
| **Fork and branch** | [`maximosipov/stable-diffusion.cpp`, `macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/stable-diffusion.cpp/tree/macbook-pro-2019-radeon-5500m-4gb) |
| **Upstream** | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) |
| **Source changes** | None. Build and run flags only. |
| **ggml copy** | Vendors [leejet/ggml](https://github.com/leejet/ggml), a separate lineage that carries none of the [ggml fixes](ggml.md) |
| **Runbook on the branch** | [RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md](https://github.com/maximosipov/stable-diffusion.cpp/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) |

Complete the [shared setup](README.md#shared-setup) first. Terms are defined in the [glossary](glossary.md).

## Overview

The diffusion counterpart to llama.cpp: image and video generation on the same ggml backends, as one static binary with no Python.

- **Modes** — text-to-image, image-to-image, inpainting, instruction-based editing, ESRGAN upscaling, and conversion of weights to GGUF or safetensors.
- **Model families** — SD 1.x and 2.x, SDXL and their [Turbo](glossary.md#g-turbo) distillations, SD3 and 3.5, Flux, Qwen-Image, Chroma, Z-Image; editing models; video models. On 4 GB, the practical subset is the first group.
- **Conditioning and adapters** — [LoRA](glossary.md#g-lora), ControlNet (SD 1.5), IP-Adapter, PhotoMaker, LCM.
- **Memory controls** — on-the-fly quantization with `--type`, [TAESD](https://github.com/madebyollin/taesd) or CPU-side [VAE decoding](glossary.md#g-vae), and VAE tiling.
- **Two binaries** — `sd-cli` for one-shot generation, and `sd-server` for an HTTP service that keeps the model loaded. The second matters here, because model loading and quantization dominate a 4-step generation.

**Why it matters here:** this is the one workload where Metal doesn't merely underperform but fails outright — the macOS watchdog kills the command buffer before an image is finished. The same models are also reachable over an OpenAI-compatible images API through [LocalAI](localai.md#images).

## Changes on this branch

**None.** stable-diffusion.cpp builds cleanly for this hardware. The branch carries only a runbook.

Everything needed is in the build and run flags, and two of them are load-bearing. Both produce *silently wrong output* when missing, which is why they're worth documenting:

| Flag | Why |
|---|---|
| `-DGGML_METAL=OFF` at build time | `-DSD_METAL=OFF` alone still compiles Metal in, because ggml enables it on macOS by default. Metal then wins device 0 at runtime, and your "Vulkan" build silently runs Metal and times out ([known issue](README.md#ki-metal)). |
| `--diffusion-conv-direct` at run time | Without it, the [UNet](glossary.md#g-unet) convolution is numerically broken on this stack and every image is colourful noise ([known issue](README.md#ki-diffusion-noise)). It is also about 3× faster. |

Because this project vendors its own copy of ggml, the fixes from [ggml.md](ggml.md) aren't present in this build. They haven't been needed: MoltenVK 1.4.2 corrects the driver bug behind most of them, and the diffusion-specific defect is handled by `--diffusion-conv-direct`. Whether the integer-dot speed-up would also help diffusion here is untested.

## Build

The fork is a submodule of this repository, already on the right branch:

```bash
git submodule update --init --recursive stable-diffusion.cpp    # it has nested submodules
cd stable-diffusion.cpp
```

To build it on its own instead:

```bash
git clone --recursive --branch macbook-pro-2019-radeon-5500m-4gb \
  https://github.com/maximosipov/stable-diffusion.cpp.git
cd stable-diffusion.cpp
```

Then, in either case:

```bash
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

Check that Metal didn't come along: `otool -L build-vulkan/bin/sd-cli | grep -i metal` must print nothing.

## Run

Every command needs `--diffusion-conv-direct`.

```bash
# SD-Turbo: the everyday choice. 4 steps, cfg 1.
./build-vulkan/bin/sd-cli --diffusion-conv-direct \
  -m models/sd_turbo.safetensors --type q8_0 \
  -p "a red apple on a wooden table, studio lighting" \
  --steps 4 --cfg-scale 1 --sampling-method euler -W 512 -H 512 --seed 42 -o apple-sdturbo.png

# SDXL-Turbo: more photorealistic, at the edge of 4 GB. The VAE must run on the CPU.
./build-vulkan/bin/sd-cli --diffusion-conv-direct --vae-on-cpu \
  -m models/sd_xl_turbo_1.0_fp16.safetensors --type q8_0 \
  -p "a photograph of an astronaut riding a horse on the moon, detailed" \
  --steps 4 --cfg-scale 1 --sampling-method euler -W 512 -H 512 --seed 42 -o astronaut-sdxlturbo.png

# SD 1.5: the baseline, and the best LoRA and ControlNet ecosystem. 20 steps, cfg 7.
./build-vulkan/bin/sd-cli --diffusion-conv-direct \
  -m models/v1-5-pruned-emaonly-fp16.safetensors \
  -p "a photograph of an astronaut riding a horse on the moon, detailed" \
  --steps 20 --cfg-scale 7 --sampling-method euler_a -W 512 -H 512 --seed 42 -o astronaut-sd15.png
```

Model loading and on-the-fly quantization take about 30 seconds for SD-Turbo, which dominates a 4-step generation. Use `sd-server` to pay that cost once.

## Verify

**Looking like an image is not enough.** Generate the same prompt and seed on `--backend cpu` and compare the two images. The [failure mode](README.md#ki-diffusion-noise) this guards against also produces an image, just one made of noise.

Verified on a fresh build: SD-Turbo q8_0, 4 steps, 512×512, seed 42, with `--diffusion-conv-direct` — a recognisable apple, **2049.08 MB** of parameters resident (text encoders 500.5, UNet 1388.9, VAE 159.7), and `generate_image` complete in **44.7 s**, of which **27.1 s was [VAE decoding](glossary.md#g-vae)**.

That last split is worth remembering: in a 4-step turbo run, more time goes into decoding the latent image than into sampling it, and decoding is also where VRAM peaks.

## Performance

Backend comparison, by CPU-reference diff. SD 1.5 fp16, 512×512, seed 42, identical prompt:

| Backend | Fits? | Correct? | Speed |
|---|---|---|---|
| Metal | Targets the Radeon, 4278 MB | No: [GPU watchdog timeout](glossary.md#g-watchdog) | Never finishes; loading the shader library alone takes 40–1000 s |
| Vulkan, no extra flags | Yes, about 2.6 GB | No: colourful noise | ~11 s/step |
| **Vulkan with `--diffusion-conv-direct`** | Yes, about 2.6 GB | **Yes, matches CPU** | **~3.4 s/step** |
| CPU (reference) | n/a | Yes | ~24 s/step |

Model fit and throughput:

| Model | Load options | VRAM | s/step | Notes |
|---|---|---|---|---|
| SD-Turbo | `--type q8_0` | **2049 MB** | **~1.7** | Comfortable. A 4-step image takes about 7 s of compute. |
| SD 1.5 | fp16 | 2035 MB | ~3.4 | Baseline, 20 steps |
| SDXL-Turbo | `--type q8_0 --vae-on-cpu` | **3836 MB** | ~9.3 | Fits, but tight against the 4278 MB usable |

**Recommendation:** SD-Turbo at q8_0 for iterating, SDXL-Turbo at q8_0 with the VAE on the CPU for final images (clearly more photorealistic, about 5× slower per step, and at 4 steps it drops prompt details more often), and SD 1.5 when you need the [LoRA and ControlNet](glossary.md#g-lora) ecosystem.

## Troubleshooting

**Every image is colourful noise.** `--diffusion-conv-direct` is missing.

**Generation never finishes, or the process is killed.** Metal was compiled in. Rebuild with `-DGGML_METAL=OFF` and check with `otool`.

**The run dies at the last step, after sampling looked fine.** VRAM peaks during [VAE decoding](glossary.md#g-vae), not sampling. Add `--vae-on-cpu`.

**bf16 weights produce NaN.** This GPU has no bf16 support. Use fp16 or a quantized GGUF.

**`--diffusion-fa` does nothing useful.** It needs cooperative-matrix hardware, which this card lacks. Leave it off.

## Reference

| Flag | Use |
|---|---|
| `--diffusion-conv-direct` | Required. Correct convolution, and about 3× faster. |
| `--type q8_0` | Quantize the checkpoint while loading. The main VRAM control. |
| `--vae-on-cpu` | Decode on the CPU. Lowers peak VRAM, and avoids SDXL's fp16 NaN. |
| `--steps`, `--cfg-scale` | 4 steps at cfg 1 for Turbo models; 20 steps at cfg 7 for SD 1.5 |
| `--backend cpu` | The correctness reference |

The same models can be served over an OpenAI-compatible images API: [localai.md](localai.md#images).
