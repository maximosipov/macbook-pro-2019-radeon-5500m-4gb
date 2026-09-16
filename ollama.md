# ollama

The friendliest way to run models locally, and the one most likely to be someone's first install. Upstream ships no Vulkan runner for macOS; this fork builds one.

| | |
|---|---|
| **Role in the stack** | Model manager and always-on server, wrapping llama.cpp |
| **Use it for** | `pull` and `run` simplicity, the model library, vision models, the desktop app |
| **Fork and branch** | [`maximosipov/ollama`, `macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/ollama/tree/macbook-pro-2019-radeon-5500m-4gb) |
| **Upstream** | [ollama/ollama](https://github.com/ollama/ollama) |
| **Source changes** | 5 patches, plus the [ten llama.cpp commits](llama.cpp.md#changes-on-this-branch) cherry-picked onto ollama's pinned llama.cpp |
| **Pinned llama.cpp** | `b10091` (see `LLAMA_CPP_VERSION`) |
| **Runbook on the branch** | [RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md](https://github.com/maximosipov/ollama/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) |

Complete the [shared setup](README.md#shared-setup) first. Terms are defined in the [glossary](glossary.md).

## Overview

ollama is closer to Docker for models than to a chat program:

- **Model lifecycle** — `ollama pull`, `run`, `create`, `list`, `ps`, `rm`, `push`, a public registry, and content-addressed blob storage shared between models.
- **`Modelfile`** — a small declarative recipe (`FROM` a base model or a local GGUF, plus `SYSTEM`, `TEMPLATE`, `PARAMETER` and adapters) that turns weights into a named, reusable model.
- **A server that manages memory for you** — models load on demand, unload after an idle `keep_alive`, and the scheduler decides the CPU/GPU split ([`num_gpu`](glossary.md#g-ngl)).
- **Two APIs** — its own REST API (`/api/generate`, `/api/chat`, `/api/embed`, …) and an [OpenAI-compatible](glossary.md#g-openai-api) `/v1` surface, with tool calling, structured output, embeddings and [vision](glossary.md#g-vlm).
- **A desktop app** for macOS and Windows, with a chat UI and settings.

**Why it matters here:** the official macOS build ships no Vulkan runner, so this machine's Radeon appears as `id=cpu` ([ollama#13127](https://github.com/ollama/ollama/issues/13127)) and you get CPU speed. This branch builds the runner upstream doesn't ship, at 71 tok/s on the GPU against 24 on the CPU.

## Changes on this branch

Five patches, on top of the cherry-picked llama.cpp fixes:

| Patch | What it fixes |
|---|---|
| `vulkan-visible-device-index` | ollama's device list counts the CPU as index 0, so it passed a CPU-inclusive index where ggml expects a Vulkan-relative one. It asked for the Radeon and got the Intel GPU — the direct cause of gibberish output ([known issue](README.md#ki-wrong-device)). |
| `image-min-tokens-env` | Upstream hardcodes 1024 [image tokens](glossary.md#g-vlm) for Qwen-VL. On a thermally limited laptop, that inflates prefill until sustained throughput collapses. `OLLAMA_IMAGE_MIN_TOKENS=512` holds about 32 tok/s and still scores 4/4 on the vision check. |
| `gpu-percent-slider` | Adds `OLLAMA_GPU_PERCENT`, a global default for the CPU/GPU split ([share of layers](glossary.md#g-ngl)), plus a slider in the desktop app's settings |
| `bundle-vulkan-env` | A packaged `.app` sets the MoltenVK environment itself, so no shell wrapper is needed |
| `raw-protocol-tab` | A debug tab in the desktop chat UI showing the literal `/api/chat` request, the rendered prompt, and the streamed response chunks |

## Build

ollama vendors a *pinned* llama.cpp commit, and its compatibility patches are written against that exact commit, so you can't build against the fork's branch tip. Check out the pinned commit and cherry-pick the [ten fix commits](llama.cpp.md#changes-on-this-branch) onto it.

> If you cloned this repository recursively, the ollama fork is already at `ollama/`, and you can use it in place of the `ollama-radeon` clone below. The `llama.cpp/` submodule is **not** a substitute for the `llama.cpp-radeon` checkout: it sits at the branch tip, and this build needs a tree at ollama's pin with the ten commits cherry-picked onto it.

```bash
brew install go node   # go >= 1.26

git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/llama.cpp.git llama.cpp-radeon
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/ollama.git ollama-radeon

cd llama.cpp-radeon
LLAMA_TAG="$(cat ../ollama-radeon/LLAMA_CPP_VERSION)"
LLAMA_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git "refs/tags/$LLAMA_TAG" | cut -f1)"
git fetch https://github.com/ggml-org/llama.cpp.git "$LLAMA_SHA"
git checkout -b radeon-for-ollama "$LLAMA_SHA"
git cherry-pick 7847fca13 595936bcd a91f29c5f acb45b872 8cbd19056 \
                8e8d7fb26 16ea5d8e7 e629266a1 2707498b9 268aff1aa
git submodule update --init --recursive --depth 1
cd ..

cp -R llama.cpp-radeon llama.cpp-radeon-ollama
( cd llama.cpp-radeon-ollama && \
  git apply ../ollama-radeon/llama/compat/*.patch ../ollama-radeon/llama/compat/models/*.patch )
```

All ten commits cherry-pick cleanly onto ollama's current pin, and the compatibility patches still apply on top.

Then build. ollama's CMake doesn't pass `-DVulkan_*` into its nested llama.cpp build, so the way to point it at MoltenVK is a fake "Vulkan SDK" made of symlinks, which `FindVulkan` picks up through `$VULKAN_SDK`:

```bash
cd ollama-radeon
P="$(brew --prefix)"
mkdir -p vulkan-sdk/include vulkan-sdk/lib vulkan-sdk/bin
ln -sfn "$P/opt/vulkan-headers/include/vulkan"       vulkan-sdk/include/vulkan
ln -sfn "$P/opt/vulkan-headers/include/vk_video"     vulkan-sdk/include/vk_video
ln -sfn "$P/opt/spirv-headers/include/spirv"         vulkan-sdk/include/spirv
ln -sfn "$P/opt/vulkan-loader/lib/libvulkan.dylib"   vulkan-sdk/lib/libvulkan.dylib
ln -sfn "$P/opt/vulkan-loader/lib/libvulkan.1.dylib" vulkan-sdk/lib/libvulkan.1.dylib
ln -sfn "$P/opt/shaderc/bin/glslc"                   vulkan-sdk/bin/glslc
ln -sfn "$P/opt/glslang/bin/glslangValidator"        vulkan-sdk/bin/glslangValidator
export VULKAN_SDK="$PWD/vulkan-sdk"
export OLLAMA_LLAMA_CPP_SOURCE="$PWD/../llama.cpp-radeon-ollama"

cmake -B build -DOLLAMA_LLAMA_BACKENDS="vulkan" -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel "$(sysctl -n hw.ncpu)"

# CMake produces the runner payload under build/lib/ollama; the CLI is a plain go build
CGO_ENABLED=1 go build -o build/ollama .
```

The `vk_video` symlink is not optional: current `vulkan_core.h` includes `vk_video/vulkan_video_codec_h264std.h`, and without it the nested build fails with a header-not-found error several hundred lines into the log.

**Desktop app and DMG.** Needed to see the GPU slider and the raw-protocol tab, and a further half-page of `npm run build`, bundle assembly, relinking for Vulkan self-containment and ad-hoc code signing. See [part 2 of the branch runbook](https://github.com/maximosipov/ollama/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md#part-2--desktop-app--dmg-needed-to-see-the-gpu-slider-and-raw-protocol-tab).

## Run

```bash
export VK_ICD_FILENAMES="$(brew --prefix)/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/molten-vk/lib:$(brew --prefix)/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export OLLAMA_VULKAN=1
export GGML_VK_VISIBLE_DEVICES=0     # required: see known issues
export OLLAMA_FLASH_ATTENTION=1      # now the fast path
export OLLAMA_IMAGE_MIN_TOKENS=512   # vision models only
export OLLAMA_HOST=127.0.0.1:11435

./build/ollama serve &
./build/ollama run qwen3:1.7b "How much is 20+34?"
```

## Verify

```bash
./build/ollama ps         # PROCESSOR must read 100% GPU
```

If it reads any share of CPU, the model didn't fit or the device index is wrong. Confirm VRAM use independently:

```bash
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

Throughput comes from the timing fields the API returns:

```bash
curl -s http://127.0.0.1:11435/api/generate -d '{
  "model": "qwen3:1.7b",
  "prompt": "Write a detailed paragraph about how a GPU renders a triangle.",
  "stream": false,
  "options": {"num_predict": 128, "temperature": 0, "num_gpu": 999}
}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(round(d["eval_count"]/(d["eval_duration"]/1e9), 1), "tok/s")'
```

## Performance

| Workload | Result |
|---|---|
| Qwen3-1.7B, text generation | **64–67 tok/s**, 100% GPU |
| Qwen3-VL-4B vision, `OLLAMA_IMAGE_MIN_TOKENS=512` | **32.0 tok/s**, 4/4 correct |
| Qwen3-VL-4B vision, upstream default of 1024 | 8.3–15.3 tok/s, decaying across requests |
| Qwen3-VL-4B, VRAM | 3.2–3.6 GiB of 4080 MiB, no CPU spill |

The 64 tok/s figure is a single run on the rebuilt binary with `OLLAMA_FLASH_ATTENTION=1`; 67 is the earlier figure. The vision rows predate the rebuild and were taken with flash attention off.

Device choice, same model and build — the reason for pinning the device:

| Device | Generation |
|---|---|
| AMD Radeon Pro 5500M | **71 tok/s** |
| CPU | 24 tok/s |

The CPU is the only fallback worth having. The Intel GPU is excluded by the pin and is not a third option.

## Troubleshooting

**The Radeon shows up as `id=cpu`.** You're on an official build with no Vulkan runner. Build this fork.

**Output is gibberish.** The device index is wrong. Confirm `GGML_VK_VISIBLE_DEVICES=0`, and that you're running a binary built with the `vulkan-visible-device-index` patch.

**Vision throughput collapses over several requests.** Set `OLLAMA_IMAGE_MIN_TOKENS=512`.

**The nested build fails on a missing `vulkan_video_codec_h264std.h`.** The `vk_video` symlink is missing from the fake SDK.

**A model runs partly on the CPU.** Force full offload per request with `"options": {"num_gpu": 999}`, or globally with `OLLAMA_GPU_PERCENT=100` or the desktop slider. If it still spills, the model plus its context doesn't fit in 4 GB.

## Reference

| Variable | Use |
|---|---|
| `OLLAMA_VULKAN=1` | Enable the Vulkan runner |
| `GGML_VK_VISIBLE_DEVICES=0` | Expose only the Radeon. Required. |
| `OLLAMA_FLASH_ATTENTION=1` | Flash attention, which is the fast path on this build |
| `OLLAMA_IMAGE_MIN_TOKENS` | Image tokens per image. 512 on this machine. |
| `OLLAMA_GPU_PERCENT` | Default share of layers on the GPU |
| `OLLAMA_HOST` | Listen address |

Per-request overrides go in `"options"`: `{"num_gpu": 0}` for CPU-only, `{"num_gpu": 999}` for full offload.
