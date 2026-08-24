# ollama — build and run recipe

Part of [AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)](README.md). Do the [shared setup](README.md#3-shared-setup-do-this-once) first; every term used here is defined in the [glossary](README.md#12-glossary). What ollama is and why it needs a custom build at all: [chapter 7](README.md#7-ollama). The measurements stay on the main page: [7.4 Benchmarks](README.md#74-benchmarks).

Branch: [`macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/ollama/tree/macbook-pro-2019-radeon-5500m-4gb).

## What was modified

5 patches:

| Patch | What it fixes |
|---|---|
| `vulkan-visible-device-index` | ollama's device list counts CPU as index 0, so it passed a CPU-inclusive index where ggml expects a **Vulkan-relative** one — it asked for the Radeon and got the integrated GPU (landmine 2, the actual cause of the gibberish) |
| `image-min-tokens-env` | upstream hardcodes 1024 [image tokens](README.md#g-vlm) for Qwen-VL; on a thermally limited laptop that inflates prefill until sustained throughput collapses. `OLLAMA_IMAGE_MIN_TOKENS=512` holds ~32 tok/s and still scores 4/4 |
| `gpu-percent-slider` | `OLLAMA_GPU_PERCENT` — a global CPU/GPU split default ([percentage of layers](README.md#g-ngl)), plus a "GPU offload" slider in the desktop app's Settings |
| `bundle-vulkan-env` | a packaged `.app` sets the MoltenVK environment itself, so no shell wrapper is needed |
| `raw-protocol-tab` | a debug tab in the desktop chat UI showing the literal `/api/chat` request, the rendered prompt, and the streamed response chunks |

## Setup

Ollama vendors a *pinned* llama.cpp commit and its compat patches are written against that exact commit, so you don't build against the fork's branch tip. Check out the pinned commit from the same fork and cherry-pick the [nine fix commits](llama.cpp.md#what-was-modified) onto it:

> **If you cloned this repo recursively**, the ollama fork is already here as a submodule at `ollama/` — use it in place of the `ollama-radeon` clone below. The `llama.cpp/` submodule is **not** a substitute for the `llama.cpp-radeon` checkout: it sits at the branch tip, and this build needs a tree at ollama's pin with the nine commits cherry-picked onto it.

```bash
brew install go node   # go >= 1.26

git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/llama.cpp.git llama.cpp-radeon
git clone --branch macbook-pro-2019-radeon-5500m-4gb https://github.com/maximosipov/ollama.git ollama-radeon

cd llama.cpp-radeon
LLAMA_TAG="$(cat ../ollama-radeon/LLAMA_CPP_VERSION)"
LLAMA_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git "refs/tags/$LLAMA_TAG" | cut -f1)"
git fetch https://github.com/ggml-org/llama.cpp.git "$LLAMA_SHA"
git checkout -b radeon-for-ollama "$LLAMA_SHA"
git cherry-pick 7847fca13 595936bcd a91f29c5f \
                acb45b872 8cbd19056 8e8d7fb26 16ea5d8e7 e629266a1 2707498b9
git submodule update --init --recursive --depth 1
cd ..

cp -R llama.cpp-radeon llama.cpp-radeon-ollama
( cd llama.cpp-radeon-ollama && \
  git apply ../ollama-radeon/llama/compat/*.patch ../ollama-radeon/llama/compat/models/*.patch )
```

All nine cherry-pick cleanly onto ollama's current pin (`b10091`), and the compat patches still apply on top — verified while writing this revision.

Then build. ollama's CMake doesn't forward `-DVulkan_*` into its nested llama.cpp build, so a fake "Vulkan SDK" of symlinks is how you point it at MoltenVK (`FindVulkan` honours `$VULKAN_SDK`):

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

# CMake produces the runner payload under build/lib/ollama; the CLI binary is a plain go build
CGO_ENABLED=1 go build -o build/ollama .
```

The `vk_video` symlink is not optional — current `vulkan_core.h` includes `vk_video/vulkan_video_codec_h264std.h`, and without it the nested ggml-vulkan build fails with a header-not-found error several hundred lines into the log.

The desktop app + DMG (needed to see the GPU slider and the raw-protocol tab) is a further half-page of `npm run build`, bundle assembly, Vulkan self-containment relinking and ad-hoc codesigning — see [the branch runbook](https://github.com/maximosipov/ollama/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md#part-2--desktop-app--dmg-needed-to-see-the-gpu-slider-and-raw-protocol-tab).

## Run

```bash
export VK_ICD_FILENAMES="$(brew --prefix)/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/molten-vk/lib:$(brew --prefix)/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export OLLAMA_VULKAN=1
export GGML_VK_VISIBLE_DEVICES=0     # load-bearing — see landmine 2
export OLLAMA_FLASH_ATTENTION=1      # now the fast path — see landmine 3
export OLLAMA_IMAGE_MIN_TOKENS=512   # vision only
export OLLAMA_HOST=127.0.0.1:11435

./build/ollama serve &
./build/ollama run qwen3:1.7b "How much is 20+34?"
./build/ollama ps                    # PROCESSOR must read 100% GPU
```

## Reproducing

`ollama ps` gives you placement; the timing fields the API returns give you throughput:

```bash
curl -s http://127.0.0.1:11435/api/generate -d '{
  "model": "qwen3:1.7b",
  "prompt": "Write a detailed paragraph about how a GPU renders a triangle.",
  "stream": false,
  "options": {"num_predict": 128, "temperature": 0, "num_gpu": 999}
}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(round(d["eval_count"]/(d["eval_duration"]/1e9), 1), "tok/s")'
```

Per-request override of the CPU/GPU split: [`"options": {"num_gpu": 0}`](README.md#g-ngl) (CPU) / `999` (all layers), or globally with `OLLAMA_GPU_PERCENT=0|100` / the desktop slider. What the numbers come back as: [7.4 Benchmarks](README.md#74-benchmarks).
