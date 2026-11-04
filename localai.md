# LocalAI

One OpenAI-compatible endpoint for text and images, packaged as a menu-bar app. One patch, but most of the work is composing the Vulkan build by hand.

| | |
|---|---|
| **Role in the stack** | OpenAI-compatible server that supervises backend processes |
| **Use it for** | Pointing a coding agent or any OpenAI SDK at this machine |
| **Fork and branch** | [`maximosipov/LocalAI`, `macbook-pro-2019-radeon-5500m-4gb`](https://github.com/maximosipov/LocalAI/tree/macbook-pro-2019-radeon-5500m-4gb) |
| **Upstream** | [mudler/LocalAI](https://github.com/mudler/LocalAI) |
| **Source changes** | 1 commit (VRAM detection). Plus build composition, wrapper scripts and an app bundle. |
| **Pinned llama.cpp** | `1cbfd1988` (see `backend/cpp/llama-cpp/Makefile`) |
| **Verified with** | LocalAI `ecdb321`, gRPC 1.83.0, protobuf 35.1, abseil 20260107.1, Go 1.26.5, Node 24, CMake 3.27.0, MoltenVK 1.4.2, macOS 14.8.2 |
| **Decision models** | Kev-4B over `/v1/systemone` on the Radeon, with LocalAI v4.12 and a rebuilt backend; tested on a local rebase, see [Decision models](#decision-models) |
| **Runbook on the branch** | [RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md](https://github.com/maximosipov/LocalAI/blob/macbook-pro-2019-radeon-5500m-4gb/RUNBOOK-macbook-pro-2019-radeon-5500m-4gb.md) |

Complete the [shared setup](README.md#shared-setup) first. Terms are defined in the [glossary](glossary.md).

## Overview

Many inference backends behind one OpenAI-compatible endpoint:

- **An API well beyond chat** — chat and text completions, embeddings, image generation, audio transcription and text-to-speech, a realtime speech API, vision, reranking and object detection. From v4.11, typed [decisions](decision-models.md) too, and on the `llama-cpp` backend from v4.12.
- **Backends as separate processes** — llama.cpp, vLLM, transformers, diffusers, whisper, piper and others, each a [gRPC server](glossary.md#g-grpc) that LocalAI starts and supervises. That indirection is the whole story of this build.
- **Galleries** — a model gallery and a backend gallery that installs backends as [OCI images](glossary.md#g-oci), with a web UI over both.
- **Agent-side features** — constrained grammars, tool calling, MCP support and built-in agents.

Only the llama.cpp and stable-diffusion backends are worth building here; the rest target Python and CUDA and find no accelerator. It is the only tool here that gives a drop-in OpenAI endpoint plus model management without being a desktop chat app.

## Changes on this branch

**One patch: VRAM detection.** `GetGPUMemoryUsage` tries `nvidia-smi`, `rocm-smi`, `clinfo` and an Apple probe that filters on a vendor string containing `apple`. On an Intel Mac with a discrete AMD card every path misses, so LocalAI logs `GPU vendor=""` and `Total available VRAM 0` and sizes its defaults as if the machine had no GPU. The patch adds a `system_profiler` fallback that reads `spdisplays_vram`, so the Radeon reports its 4 GB. Proposed upstream; see [Upstream submissions](#upstream-submissions).

The rest is not source but composition. LocalAI's macOS build targets Apple Silicon and Metal, so a Vulkan build is assembled from two pieces:

1. The llama backend is a separate gRPC server process built from a pinned llama.cpp; substitute one built from that pin with the [ten Vulkan fix commits](llama.cpp.md#changes-on-this-branch) cherry-picked on.
2. The backend is a child process, so a wrapper script must carry the MoltenVK environment into it.

Two Makefile traps silently produce a Metal build instead; both are covered in [Build](#build). The branch also adds the menu-bar app and its packaging script.

## Build

```bash
brew install go node cmake grpc protobuf abseil

git clone https://github.com/maximosipov/LocalAI.git && cd LocalAI
make build                          # Go server + React UI -> ./local-ai
```

> If you cloned this repository recursively, the LocalAI fork is already at `LocalAI/` on the right branch: `cd LocalAI && make build`, and skip the clone. The `llama.cpp/` submodule does **not** replace the backend checkout below.

### The llama.cpp backend

The backend's llama.cpp source must be LocalAI's pinned commit with the ten fix commits cherry-picked on, not the fork's branch tip: `grpc-server.cpp` is written against that exact commit, and master drifts within days.

```bash
PIN="$(sed -n 's/^LLAMA_VERSION?=//p' backend/cpp/llama-cpp/Makefile)"
git clone https://github.com/maximosipov/llama.cpp.git backend/cpp/llama-cpp/llama.cpp
cd backend/cpp/llama-cpp/llama.cpp
# Fetch the pin by SHA: it is a plain master commit, not a tag, and not necessarily an
# ancestor of the fork's branch, so a fork-only clone will not have it.
git fetch https://github.com/ggml-org/llama.cpp.git "$PIN"
git checkout -b build "$PIN"
git cherry-pick 7847fca13 595936bcd a91f29c5f acb45b872 8cbd19056 \
                8e8d7fb26 16ea5d8e7 e629266a1 2707498b9 268aff1aa
git submodule update --init --recursive --depth 1
cd ../../../..

# Graft LocalAI's grpc-server into llama.cpp/tools/. Run once: it is not idempotent.
( cd backend/cpp/llama-cpp && mkdir -p llama.cpp/tools/grpc-server && bash prepare.sh )
```

`CMAKE_ARGS` must be passed through the environment, not the make command line: the Makefile does `CMAKE_ARGS?=` then `CMAKE_ARGS+=…`, and a command-line variable overrides those appends and drops `-DGGML_VULKAN=1`.

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

`-DGGML_METAL=OFF` is required because `BUILD_TYPE=vulkan` and the Darwin branch are arms of one if/else chain: choosing vulkan on macOS skips the Darwin arm that would have set `GGML_METAL=OFF`.

Check the result: `otool -L backend/cpp/llama-cpp/grpc-server | grep -i metal` must print nothing.

### The image backend

The `stablediffusion-ggml` backend wraps [stable-diffusion.cpp](stable-diffusion.cpp.md), so image models are reachable over the OpenAI images API without a second server. The same Makefile trap repeats, with the same fix:

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

Check: `otool -L backend/go/stablediffusion-ggml/libgosd-fallback.so` must list libvulkan and no `Metal.framework`.

### Package the app

The branch adds a menu-bar (`NSStatusItem`) app with no Dock icon that owns the server process: it starts the server with the MoltenVK environment and device pin set, polls `/readyz` for its status line, and offers Open WebUI, Copy API Base URL, Open Models Folder, Show Log, Restart and Quit.

Two things fail silently if wrong:

- **Pass the storage paths explicitly.** LocalAI resolves its data, backends and configuration directories *relative to the working directory*. An app bundle launches with `cwd=/`, so the data path becomes `//data` and the server exits with `read-only file system`, with nothing in the UI to say why. The app passes `--backends-path`, `--localai-config-dir`, `--generated-content-path` and `LOCALAI_DATA_PATH` under `~/Library/Application Support/LocalAI/`.
- **Make the bundle self-contained.** `grpc-server` links **107 Homebrew dylibs**, against two for the ollama DMG. The packaging script walks that dependency closure with `otool`, copies each library into `Contents/Frameworks/`, rewrites every install name to `@rpath`, and adds the matching rpaths. The bundled ICD JSON points at `../../Frameworks/libMoltenVK.dylib` so the app stays relocatable. The acceptance test is that no Homebrew path survives anywhere in the bundle.

The result is a 367 MB `.app` and a 142 MB DMG, ad-hoc signed — not notarized, so the quarantine bit must be cleared on first launch. Weights are not bundled; on first run the app seeds model YAML files into `~/Library/Application Support/LocalAI/models` and symlinks weights from a development checkout if present.

Installing it: [quick start](README.md#quick-start-localai-and-opencode).

## Run

Ready-made model configs for the recommended models are in [localai-models/](localai-models/) — copy them into your models folder instead of writing the YAML by hand.

Register the Vulkan `grpc-server` as an external backend, wrapped in a script that carries the MoltenVK environment into the child process:

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
```

The GGUF must live **inside** `--models-path`. LocalAI rejects a path outside it as an invalid file path, and the model then silently doesn't appear in `/v1/models`. A symlink satisfies it:

```bash
mkdir -p models && ln -s /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf models/

cat > models/qwen3-4b.yaml <<'EOF'
name: qwen3-4b
backend: llama-cpp
parameters:
  model: Qwen3-4B-Instruct-2507-Q4_K_M.gguf
context_size: 4096
f16: true
gpu_layers: 99
flash_attention: "true"
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

### Images

[`diffusion_conv_direct`](README.md#ki-diffusion-noise) is mandatory, and is set per model. The image backend is also a child process, so it needs its own wrapper, and it resolves the library it loads through `SD_LIBRARY`:

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
  "model": "sd-turbo", "prompt": "a red apple on a wooden table, studio lighting",
  "size": "512x512", "n": 1
}'
```

Open the returned URL. A recognisable apple means the whole path is correct; colourful noise means `diffusion_conv_direct` never reached the backend.

SDXL-Turbo additionally needs `"keep_vae_on_cpu:true"` and sits at 3.8 GB, the edge of the card.

### Use with a coding agent

The default `context_size: 4096` is useless for a coding agent: OpenCode's opening request carries 12 tool schemas, 38 KB of JSON, rendering to 11,132 prompt tokens. The server reports it:

```
{"error":{"message":"request (11132 tokens) exceeds the available context size (4096 tokens), try increasing it"}}
```

Use a larger window with a [quantized KV cache](glossary.md#g-kv-quant). Granite-4.0-H-Micro is the better pick: best tool calling measured here, and its [Mamba-2](glossary.md#g-ssm) recurrent state makes a wide window nearly free in VRAM.

```yaml
# ~/Library/Application Support/LocalAI/models/granite-coder.yaml
name: granite-coder
backend: llama-cpp
parameters:
  model: granite-4.0-h-micro-Q4_K_M.gguf
context_size: 16384
f16: true
gpu_layers: 99
flash_attention: "true"     # required for a quantized KV cache, and it runs on the GPU here
cache_type_k: q4_0
cache_type_v: q4_0
```

The OpenCode side of the configuration is in the [quick start](README.md#quick-start-localai-and-opencode). Measured with `opencode run` against this profile:

| Turn | Wall clock |
|---|---|
| First turn (model load plus prefill of the whole agent prompt) | **~3 min** |
| Warm turn, one-word answer | **10.1 s** |
| Warm turn, a real question with a few sentences of answer | **13.8 s** and **23.6 s** |

The first turn is expensive and the rest are not: the agent's system prompt is [prefilled](glossary.md#g-prefill) once. Keep the server resident and the session alive.

It is a 3B-class model. Asked for a one-liner counting lines across `.md` files it produced a command that counts files; asked what `-ngl` does it answered that it "disables the NVIDIA GPU library". Read what it produces.

## Decision models

LocalAI serves [decision models](decision-models.md) through `/v1/systemone`, the request shape Kev and Ollama use, from v4.11. v4.12 adds an OpenAI-style `/v1/decisions` and decisions in the `llama-cpp` backend. The decision code itself is llama.cpp's (`server-decision.cpp`, upstream since 2 October 2026); LocalAI's backend forwards to it. The fork branch above predates both, so this needs a newer server **and** a newer backend.

> **Tested 4 November 2026** on a local rebase onto upstream LocalAI `v4.12.0-548` (`81a56608d`). The fork branch has not been moved yet. Until it is, build from upstream master with the one patch on top, as below.

### Build the server

```bash
cd LocalAI
git fetch https://github.com/mudler/LocalAI.git master
git checkout -b decisions FETCH_HEAD
git cherry-pick 36540448d     # VRAM detection: upstream still probes Apple Silicon GPUs only
make build
```

The patch applies without conflicts.

### Build the backend

Follow [The llama.cpp backend](#the-llamacpp-backend), with three differences.

**The pin is newer, and one cherry-pick conflicts.** The Makefile now pins `785322d17`. Of the ten fix commits, `a91f29c5f` (the diagnostic env knobs) conflicts in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, because upstream rewrote the RDNA3/4 rows-per-workgroup code beside it. Keep upstream's side of the hunk, add the commit's two overrides above it, then `git add` and `git cherry-pick --continue`:

```cpp
    // mac-radeon-ai: rows-per-workgroup overrides for tuning. rm_kq drives the
    // K-quant matvec (q4_K/q6_K), which is ~54% of decode time on this stack.
    if (const char * e = getenv("GGML_VK_RM_KQ"))   { rm_kq   = atoi(e); }
    if (const char * e = getenv("GGML_VK_RM_STDQ")) { rm_stdq = atoi(e); }
    // RDNA3/4: above four columns, static 4 rows ...   (upstream's code continues unchanged)
```

The other nine apply cleanly. The result is identical to the tree the measurements below were taken on.

**Two more libraries.** The decision bridge validates image input and links zlib and libjpeg even for text-only models:

```bash
brew install zlib jpeg-turbo
```

**`prepare.sh` stages more.** Besides applying LocalAI's llama.cpp patches as before, it copies the decision bridge (`decision_compat.h`, `decision_images.h`) beside `grpc-server.cpp`. It is still run once.

Build with the same `BUILD_TYPE` and `CMAKE_ARGS` as before; it takes about 4 minutes. Check: `otool -L backend/cpp/llama-cpp/grpc-server` lists `libvulkan`, `libjpeg` and `libz`, and nothing matching `metal`.

### Run

The upstream server exits at startup with `addon runtime unavailable or invalid protocol` unless add-ons are turned off; `make build` doesn't appear to produce the add-on runtime. Decisions don't use it.

```bash
ln -s /path/to/Kev-4B-Q4_K_M.gguf models/     # ggml-org/Kev-4B-GGUF

cat > models/kev-4b.yaml <<'EOF'
name: kev-4b
backend: llama-cpp
context_size: 8192
gpu_layers: 99
flash_attention: "true"
known_usecases:
  - decisions
parameters:
  model: Kev-4B-Q4_K_M.gguf
EOF

./local-ai run --address 127.0.0.1:8086 --models-path ./models --addons-enabled=false \
  --external-grpc-backends "llama-cpp:$PWD/backend/cpp/llama-cpp/run-vulkan.sh"
```

Declare `known_usecases: [decisions]`. LocalAI never guesses it, and a model that declares other usecases without it gets a `400` from the decision endpoints.

```bash
curl -s http://127.0.0.1:8086/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "kev-4b",
  "state": "My order arrived broken and I want my money back. This is the second time.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments, invoices and refunds",
                          "shipping": "Delivery and damaged goods",
                          "product": "Questions about how the product works"}},
    "refund": {"type": "noul", "instructions": "The customer explicitly asks for a refund"},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]}
  }
}' | python3 -m json.tool
```

Measured:

| Check | Result |
|---|---|
| `/v1/systemone`, three questions (choice, noul, score) | shipping 0.67, refund 0.96, urgency 2.11; 128 input tokens, 0 output |
| `/v1/decisions`, the OpenAI-style contract | Typed answers in question order |
| First request, including model load | 10.6 s |
| GPU placement | `n_gpu_layers=99` applied. All 33 layers on the Radeon, about 3.7 GB of 4080 MiB, and the probabilities match a direct `llama-server` run with every layer offloaded, digit for digit |
| VRAM detection | `/api/resources` reports the Radeon Pro 5500M with 4 GB; the patch still works on v4.12 |

Kev-4B's accuracy and calibration on this path are in [decision-models.md](decision-models.md#results): 0.745 on JevBench, with the best calibration measured.

What doesn't work here:

- **Jeva-4B.** Neither LocalAI nor llama.cpp supports it.
- **The `vllm-cpp` backend.** On macOS it is packaged only for Apple Silicon, and its Vulkan engine keeps weights in system RAM on a discrete card. See [Runtime paths](decision-models.md#runtime-paths).
- **The menu-bar app** has not been rebuilt on v4.12. Its launch arguments would need `--addons-enabled=false` too.

## Verify

LocalAI's hardware probe reports `GPU vendor=""` and `Total available VRAM 0` on macOS and does not forward the backend's ggml banner, so its logs cannot confirm placement. Ask IOKit:

```bash
curl -s http://127.0.0.1:8085/readyz -o /dev/null -w '%{http_code}\n'   # 200
curl -s http://127.0.0.1:8085/v1/models | python3 -m json.tool
ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
```

A 4B Q4_K_M model resident on the Radeon shows about 3.4 GiB. If that stays near idle during generation the model is on the CPU: check [`gpu_layers`](glossary.md#g-ngl) survived the load, and set `LOCALAI_DISABLE_HARDWARE_DEFAULTS=true` if auto-tuning overrides it.

## Performance

Qwen3-4B-Instruct-2507 Q4_K_M, context 4096, f16 KV cache, `gpu_layers: 99`:

| Measure | Flash attention off | Flash attention on |
|---|---|---|
| Correctness battery over `/v1/chat/completions` | **4/4** | `20+34` → `54` |
| Generation, 128 tokens, 3 runs | **27.1 / 27.9 / 28.4 tok/s** | 21.3 / **30.9 / 30.7** tok/s |
| VRAM resident during generation | 3.35–3.50 GiB of 4080 MiB | **3.06 GiB** |
| GPU utilisation during generation | 82% | — |
| Model load to first token | ~0.9 s for a short prompt | — |

The flash-attention-off column is the more careful measurement and the tightest spread in this repository, which is why [benchmarking.md](benchmarking.md) prefers sustained serving to micro-benchmarks. The flash-attention-on column was taken while a compile saturated all 16 cores; read it as a lower bound.

Granite-4.0-H-Micro, served from the same stack at context 8192:

| Measure | Result |
|---|---|
| Generation, 128 tokens, 3 warm runs | **27.8 / 32.5 / 32.3 tok/s** |
| VRAM resident during generation | **2.77 GB** of 4080 MiB, no CPU spill |

The most reliable model measured here is also among the fastest served.

**Images from the same server:** SD-Turbo q8_0, 4 steps, 512×512 — about 2 GB resident, **3.04 GB peak** (the peak is [VAE decoding](glossary.md#g-vae), not sampling), about 50 s for the first call including model load and quantization, and about 44 s after. SD 1.5 fp16 at 20 steps also works, at 88 s.

## Troubleshooting

**Load one model per server session.** This is the one rule that matters on a 4 GB card, and getting it wrong produces a failure with no error message:

> A model loaded **after another model has been loaded in the same session** returns an *empty completion* for a large prompt: `finish_reason: "stop"`, zero completion tokens, after paying the full prefill cost. LocalAI logs only `Backend returned empty response, retrying`, five times. The client hangs through the retries and prints nothing.

Reproduced four times, and it is none of the things it resembles. Not context overflow (the same 11,132-token prompt succeeds at `context_size: 16384`), not streaming (the same request succeeds when streamed), not the model (two models both do it, and both work when loaded first), and not a cancelled request. The predictor is whether another model was loaded first: 3 of 3 clean sessions succeeded, 4 of 4 sessions with a prior load failed.

The cause is a 4 GB card being asked for more than it has. Two resident models measure **3.47 GB** of the 4.28 GB usable, *before* the 11K-token KV cache is allocated:

```bash
$ ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"inUseVidMemoryBytes"=[0-9]*'
2295623680   # Granite alone
3467935744   # Granite and Qwen both resident
```

The hardware limit is real. The software defect is that nothing says so: the shortfall surfaces as an empty completion instead of an out-of-memory error.

`--max-active-backends=1` (this machine's `runtime_settings.json` already sets it) prevents two models being resident at once, so it does rule out the empty-completion path above. What it does not do is make switching safe under load. Measured over the OpenAI API:

| Request pattern spanning a model switch | Result |
|---|---|
| Sequential — one request at a time | Reliable. Three models in one session, every answer correct. |
| Concurrent — three requests at once | **Exactly one of the three fails** with `rpc error: code = Unavailable`, reproduced in 3 of 3 trials, both switch directions. |

The eviction window is the whole story: the first request tears down one backend and starts another, and anything arriving during that gap hits a dead socket. So a **single-threaded** client can route across models freely, while a client that opens parallel requests — OpenCode starts a session with three — will lose one per switch. **Point a parallel client at one model and leave it there; to change models, quit and relaunch first.** A serialised caller can switch at will.

**`content` is empty and `reasoning` is full.** That's a [thinking model](glossary.md#g-thinking) behaving normally over an OpenAI-compatible API. `content` only fills once the model closes its think block, so an empty `content` with `finish_reason: "length"` is a token-budget problem, not the MoltenVK corruption it resembles.

**The model doesn't appear in `/v1/models`.** Its weights are outside `--models-path`. Symlink them in.

**The server exits at startup with `read-only file system`.** Storage paths were resolved relative to a working directory of `/`. Pass them explicitly, as the app bundle does.

**The server exits at startup with `addon runtime unavailable or invalid protocol`.** That's v4.12's add-ons, whose runtime a plain `make build` doesn't seem to produce. Pass `--addons-enabled=false`.

**Images come back as noise.** `diffusion_conv_direct:true` didn't reach the backend. Check the model YAML's `options` list.

## Upstream submissions

Three changes are prepared for [mudler/LocalAI](https://github.com/mudler/LocalAI). Only the first is part of the build described above; the other two sit on their own branches off `upstream/master` and are not compiled into the binary this guide produces.

| Change | Branch | In this build |
|---|---|---|
| Detect non-Apple GPUs on macOS via `system_profiler`, so a discrete AMD card reports its VRAM instead of 0. Still needed on v4.12, where it applies without conflicts | `macbook-pro-2019-radeon-5500m-4gb` | Yes |
| Make img2img `strength` configurable on the stable-diffusion backend, instead of hardcoding 0.75 | `feat/sd-img2img-strength` | No |
| Honour cloud-proxy passthrough on `/v1/images/generations`, which already works for chat | `feat/images-cloud-proxy` | No |

## Reference

| Setting | Where | Use |
|---|---|---|
| `gpu_layers: 99` | Model YAML | Put every layer on the GPU |
| `flash_attention: "true"` | Model YAML | Required for a quantized KV cache |
| `cache_type_k`, `cache_type_v` | Model YAML | KV cache precision. `q4_0` is the best trade here. |
| `context_size` | Model YAML | Context window. 16384 for agent use. |
| `diffusion_conv_direct:true` | Model YAML `options` | Required for image models |
| `keep_vae_on_cpu:true` | Model YAML `options` | Required for SDXL |
| `known_usecases: [decisions]` | Model YAML | Serve a decision model on `/v1/systemone` and `/v1/decisions` |
| `--external-grpc-backends` | Command line | Register the wrapper scripts |
| `--addons-enabled=false` | Command line | Required to start v4.12 built with `make build` |
| `--models-path` | Command line | Weights and YAML files must live inside it |
| `LOCALAI_DISABLE_HARDWARE_DEFAULTS=true` | Environment | Stop LocalAI auto-tuning over your settings |

| Path | What |
|---|---|
| `http://127.0.0.1:8085` | API and web UI |
| `~/Library/Application Support/LocalAI/models` | Models and YAML configs (app install) |
| `~/Library/Application Support/LocalAI/logs/local-ai.log` | Log (app install) |
