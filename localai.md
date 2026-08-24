# LocalAI — build, run and packaging recipe

Part of [AI on a 2019 MacBook Pro — Radeon Pro 5500M (4 GB)](README.md). Do the [shared setup](README.md#3-shared-setup-do-this-once) first; every term used here is defined in the [glossary](README.md#12-glossary). What LocalAI is and why it is worth the build: [chapter 8](README.md#8-localai). The measurements stay on the main page: [8.4 Benchmarks](README.md#84-benchmarks). For the shortest path from nothing installed to a coding agent talking to this server, see [the quick start](README.md#installing-the-localai-app-and-pointing-a-coding-agent-at-it).

Branch: [`macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/LocalAI/tree/macbook-pro-2019-radeon-5500m-4gb).

**No source changes are needed**, but LocalAI's macOS build path targets Apple Silicon + Metal, so getting a Vulkan backend requires composing two pieces yourself: its llama backend is a **[separate gRPC server process](README.md#g-grpc)** built from a pinned llama.cpp, and you have to substitute a fixed one.

*Verified with LocalAI `ecdb321`, its pinned llama.cpp `1cbfd1988`, Homebrew gRPC 1.83.0 / protobuf 35.1 / abseil 20260107.1, Go 1.26.5, Node 24, CMake 3.27.0, MoltenVK 1.4.2, macOS 14.8.2.*

## Setup

```bash
brew install go node cmake grpc protobuf abseil

git clone https://github.com/maximosipov/LocalAI.git && cd LocalAI
make build                          # Go server + React UI -> ./local-ai
```

> **If you cloned this repo recursively**, the LocalAI fork is already here as a submodule at `LocalAI/` on the right branch — `cd LocalAI && make build` and skip the clone. The `llama.cpp/` submodule does **not** replace the backend checkout below: it sits at the branch tip, and the backend must be built from LocalAI's pinned commit with the nine fixes cherry-picked onto it.

The backend's llama.cpp source must be **LocalAI's pinned commit with the [nine Vulkan fix commits](llama.cpp.md#what-was-modified) cherry-picked onto it** — not the fork's branch tip. LocalAI's `grpc-server.cpp` is written against that exact commit and llama.cpp's master drifts within days:

```bash
PIN="$(sed -n 's/^LLAMA_VERSION?=//p' backend/cpp/llama-cpp/Makefile)"
git clone https://github.com/maximosipov/llama.cpp.git backend/cpp/llama-cpp/llama.cpp
cd backend/cpp/llama-cpp/llama.cpp
# Fetch the pin by SHA. It is a plain master commit, not a tag, and not necessarily
# an ancestor of the fork's branch — a fork-only clone will not have it.
git fetch https://github.com/ggml-org/llama.cpp.git "$PIN"
git checkout -b build "$PIN"
git cherry-pick 7847fca13 595936bcd a91f29c5f \
                acb45b872 8cbd19056 8e8d7fb26 16ea5d8e7 e629266a1 2707498b9
git submodule update --init --recursive --depth 1
cd ../../../..

# graft LocalAI's grpc-server into llama.cpp/tools/ (run once; it is not idempotent)
( cd backend/cpp/llama-cpp && mkdir -p llama.cpp/tools/grpc-server && bash prepare.sh )
```

Then build the backend. **`CMAKE_ARGS` must go through the environment**, not the make command line — the Makefile does `CMAKE_ARGS?=` then `CMAKE_ARGS+=…`, and a command-line variable would override those appends and drop `-DGGML_VULKAN=1` itself:

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

`-DGGML_METAL=OFF` is required here for a subtle reason: in that Makefile `BUILD_TYPE=vulkan` and the Darwin branch are arms of **one if/else chain**, so choosing vulkan on macOS means the Darwin arm — the one that would have set `GGML_METAL=OFF` — never runs. Gate the build with `otool -L backend/cpp/llama-cpp/grpc-server | grep -i metal` returning nothing.

## Run

Register the Vulkan `grpc-server` as an external backend. LocalAI spawns it as a child process, so wrap it in a script that carries the MoltenVK environment:

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
flash_attention: "true"       # landmine 3 — now the fast path, not the slow one
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

`context_size: 4096` is fine for chat and useless for a coding agent — for that profile, see [the quick start](README.md#give-it-a-model-profile-sized-for-an-agent), and load [one model per session](README.md#one-model-per-session).

## Verifying it is actually on the GPU

This is the hard part here. LocalAI's own hardware probe reports `GPU vendor=""` and `Total available VRAM 0` on macOS, and it does not forward the backend's ggml init banner into its log — so nothing LocalAI prints tells you where the model landed. Ask IOKit instead:

```bash
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

A 4B Q4_K_M model resident on the Radeon shows **~3.4 GiB** in use on the [AMD accelerator node](README.md#g-ioreg). If that stays near idle while generation runs, the model is on the CPU — check that [`gpu_layers`](README.md#g-ngl) survived the load (`LOCALAI_DISABLE_HARDWARE_DEFAULTS=true` disables LocalAI's auto-tuning if it's overriding you).

One more thing that looks like a failure and isn't: with a **[thinking](README.md#g-thinking)** model (Qwen3 and friends), `choices[].message.content` comes back **empty** and the chain of thought is in `choices[].message.reasoning`. `content` only fills once the model closes the think block, so an empty `content` with `finish_reason: "length"` is a token budget problem — not the MoltenVK garbage-output failure it resembles.

## Text-to-image on the same GPU

LocalAI's `stablediffusion-ggml` backend wraps stable-diffusion.cpp, so [chapter 6](README.md#6-stable-diffusioncpp)'s image models are reachable over the OpenAI images API without standing up a second server. **No source changes are needed here either** — the flag this card cannot do without is a per-model option.

The build trap from [Setup](#setup) repeats one directory over: in `backend/go/stablediffusion-ggml/Makefile`, `BUILD_TYPE=vulkan` and the `OS=Darwin` case are arms of one if/else chain, so selecting vulkan on macOS skips the Darwin arm that sets `GGML_METAL=OFF`, and ggml auto-enables Metal. Pass it explicitly, through the environment:

```bash
export BUILD_TYPE=vulkan
export CMAKE_ARGS="-DGGML_METAL=OFF \
  -DVulkan_INCLUDE_DIR=$(brew --prefix)/opt/vulkan-headers/include \
  -DVulkan_LIBRARY=$(brew --prefix)/opt/vulkan-loader/lib/libvulkan.dylib \
  -DVulkan_GLSLC_EXECUTABLE=$(brew --prefix)/opt/shaderc/bin/glslc \
  -DVulkan_GLSLANG_VALIDATOR_EXECUTABLE=$(brew --prefix)/opt/glslang/bin/glslangValidator \
  -DOpenMP_ROOT=$(brew --prefix)/opt/libomp \
  -DCMAKE_CXX_FLAGS=-I$(brew --prefix)/opt/spirv-headers/include \
  -DCMAKE_BUILD_TYPE=Release"
make -C backend/go/stablediffusion-ggml JOBS="$(sysctl -n hw.ncpu)" stablediffusion-ggml
```

Gate: `otool -L backend/go/stablediffusion-ggml/libgosd-fallback.so` lists libvulkan and **no** `Metal.framework`.

The backend is a child process, so it needs its own wrapper carrying the MoltenVK environment, and it resolves the library it dlopens through `SD_LIBRARY`:

```bash
cat > backend/go/stablediffusion-ggml/run-vulkan.sh <<'WRAP'
#!/usr/bin/env bash
P="$(brew --prefix)"
D="$(cd "$(dirname "$0")" && pwd)"
export VK_ICD_FILENAMES="$P/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$D/lib:$P/opt/molten-vk/lib:$P/opt/vulkan-loader/lib:$DYLD_LIBRARY_PATH"
export GGML_VK_VISIBLE_DEVICES=0
export SD_LIBRARY="$D/libgosd-fallback.so"
exec "$D/stablediffusion-ggml" "$@"
WRAP
chmod +x backend/go/stablediffusion-ggml/run-vulkan.sh
```

[`diffusion_conv_direct`](README.md#g-conv-direct) is **mandatory** (landmine 5) and is set per model:

```bash
cat > models/sd-turbo.yaml <<'EOF'
name: sd-turbo
backend: stablediffusion-ggml
parameters:
  model: sd_turbo.safetensors
step: 4
cfg_scale: 1
options:
- "sampler:euler"
- "diffusion_conv_direct:true"
- "wtype:q8_0"
EOF

./local-ai run --address 127.0.0.1:8080 --models-path ./models \
  --external-grpc-backends "llama-cpp:$PWD/backend/cpp/llama-cpp/run-vulkan.sh,stablediffusion-ggml:$PWD/backend/go/stablediffusion-ggml/run-vulkan.sh"

curl -s http://127.0.0.1:8080/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "sd-turbo",
  "prompt": "a red apple on a wooden table, studio lighting",
  "size": "512x512",
  "n": 1
}'
```

Open the returned URL. A recognisable apple means the whole path is correct; colourful noise means `diffusion_conv_direct` never reached the backend. Measured here: SD-Turbo q8_0, 4 steps, 512×512 — ~2 GB resident, **3.04 GB peak** (peak is [VAE decode](README.md#g-vae), not sampling), ~50 s for the first call including model load and on-the-fly quantization, ~44 s after. SD 1.5 fp16 at 20 steps also works (88 s); SDXL-Turbo needs `"keep_vae_on_cpu:true"` and sits at 3.8 GB, the edge of the card.

Re-verified on freshly rebuilt backends: the same request returned a recognisable apple in **44.5 s** end to end over the HTTP API.

## A menu-bar app and a DMG

LocalAI ships no desktop app, so this is a small one: an `NSStatusItem` accessory app (menu bar, no Dock icon) that owns the server process — starts it with the MoltenVK environment and the device pin, polls `/readyz` to drive its status line, and offers Open WebUI / Copy API Base URL / Open Models Folder / Show Log / Restart / Quit.

Two things it has to get right, both of which fail silently otherwise:

- **Pass the storage paths explicitly.** LocalAI resolves its data, backends and configuration directories *relative to the working directory*. An app bundle launches with `cwd=/`, so the data path becomes `//data` and the server exits at startup with `read-only file system` — with nothing in the UI to say why. The app passes `--backends-path`, `--localai-config-dir`, `--generated-content-path` and `LOCALAI_DATA_PATH` under `~/Library/Application Support/LocalAI/`.
- **Make the bundle self-contained.** `grpc-server` links **107 Homebrew dylibs** (grpc, protobuf, abseil, re2, c-ares, openssl@3, libomp, vulkan-loader) — against two for the Ollama DMG. The packaging script walks that closure with `otool`, copies each into `Contents/Frameworks/`, rewrites every install name to `@rpath`, and adds the matching rpaths; the bundled ICD json points at `../../Frameworks/libMoltenVK.dylib` so the app stays relocatable. The acceptance test is that no Homebrew path survives anywhere in the bundle.

Weights are deliberately **not** bundled — they are gigabytes. On first run the app seeds the model YAMLs into `~/Library/Application Support/LocalAI/models` and symlinks weights from a development checkout if one is present.

Result on this machine: a 367 MB `.app`, a **142 MB** DMG, ad-hoc signed (no Apple identity here, so not notarized — clear the quarantine bit on first launch with `xattr -dr com.apple.quarantine "/Applications/LocalAI M.app"`). Verified by launching the bundle and serving both a chat completion (`20+34` → `54`, 8.2 s) and a 512×512 image (43.5 s), with Homebrew irrelevant to the run.

Installing it and pointing an editor at it: [the quick start](README.md#install-the-menu-bar-app).

## Reproduce

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

What these come back as, and why the LocalAI numbers are the most trustworthy on this machine: [8.4 Benchmarks](README.md#84-benchmarks).
