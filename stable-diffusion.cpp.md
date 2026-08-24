# stable-diffusion.cpp — build and run recipe

Part of [AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)](README.md). Do the [shared setup](README.md#3-shared-setup-do-this-once) first; every term used here is defined in the [glossary](README.md#12-glossary). What it is and why it is here: [chapter 6](README.md#6-stable-diffusioncpp). The measurements — [backend comparison](README.md#64-backend-comparison) and [model fit and throughput](README.md#65-model-fit-and-throughput) — stay on the main page.

Branch: [`macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/stable-diffusion.cpp/tree/macbook-pro-2019-radeon-5500m-4gb) — it carries [a runbook](https://github.com/maximosipov/stable-diffusion.cpp/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) and **no source changes**. Everything needed is build flags and run flags, and two of them are non-obvious and both produce silently wrong output.

## Setup

This fork is a submodule of this repo at `stable-diffusion.cpp/`, already on the right branch — `git submodule update --init --recursive stable-diffusion.cpp` (it has nested submodules of its own) and `cd stable-diffusion.cpp` is enough. The standalone clone is for building it on its own:

```bash
git clone --recursive --branch macbook-pro-2019-radeon-5500m-4gb \
  https://github.com/maximosipov/stable-diffusion.cpp.git
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

**The first load-bearing flag** is `-DGGML_METAL=OFF` alongside `-DSD_METAL=OFF`: ggml auto-enables Metal on macOS, so `-DSD_METAL=OFF` alone still compiles Metal in, and it wins device 0 at runtime. Your "Vulkan" build silently runs Metal and times out (landmine 1).

**The second is [`--diffusion-conv-direct`](README.md#g-conv-direct) at run time** (landmine 5) — without it the [UNet](README.md#g-unet) convolution is numerically broken on this stack and you get full-frame colourful noise. It is also ~3× faster than the path it replaces.

## Run

```bash
# SD-Turbo — the daily driver. 4 steps, cfg 1.
./build-vulkan/bin/sd-cli --diffusion-conv-direct \
  -m models/sd_turbo.safetensors --type q8_0 \
  -p "a red apple on a wooden table, studio lighting" \
  --steps 4 --cfg-scale 1 --sampling-method euler -W 512 -H 512 --seed 42 -o apple-sdturbo.png

# SDXL-Turbo — better photorealism, at the edge of 4 GB.
./build-vulkan/bin/sd-cli --diffusion-conv-direct --vae-on-cpu \
  -m models/sd_xl_turbo_1.0_fp16.safetensors --type q8_0 \
  -p "a photograph of an astronaut riding a horse on the moon, detailed" \
  --steps 4 --cfg-scale 1 --sampling-method euler -W 512 -H 512 --seed 42 -o astronaut-sdxlturbo.png

# SD 1.5 — baseline, best LoRA/ControlNet ecosystem. 20 steps, cfg 7.
./build-vulkan/bin/sd-cli --diffusion-conv-direct \
  -m models/v1-5-pruned-emaonly-fp16.safetensors \
  -p "a photograph of an astronaut riding a horse on the moon, detailed" \
  --steps 20 --cfg-scale 7 --sampling-method euler_a -W 512 -H 512 --seed 42 -o astronaut-sd15.png
```

Which of the three to use, and what each costs in VRAM and seconds per step: [6.5 Model fit and throughput](README.md#65-model-fit-and-throughput). The same models are servable over an OpenAI-compatible images API — see [LocalAI](localai.md#text-to-image-on-the-same-gpu).

## Reproducing

Run the same prompt+seed on `--backend cpu` and diff the images ([CPU-reference diff](README.md#b-imgdiff)); time with sd.cpp's own per-step log. Wall clock includes model load and on-the-fly [quantization](README.md#g-quant) of the [safetensors](README.md#g-safetensors) checkpoint (~30 s for SD-Turbo), which dominates a 4-step generation — use `sd-server` to amortize it.

Re-verified on a fresh build of the branch: SD-Turbo q8_0, 4 steps, 512×512, seed 42, `--diffusion-conv-direct` — a recognisable apple, **2049.08 MB** of parameters resident (text encoders 500.5, UNet 1388.9, VAE 159.7), `generate_image` complete in **44.7 s**, of which **27.1 s was [VAE decode](README.md#g-vae)**. That last split is worth internalising: on a 4-step turbo run more time goes into decoding the latent than into sampling it, and it is also where peak VRAM lands.
