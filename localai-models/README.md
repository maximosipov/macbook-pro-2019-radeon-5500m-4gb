# LocalAI model configs

LocalAI model YAML files for every model analysed in [models.md](../models.md), with context settings described below — read [the caveat on allocation ceilings](#maximum-context-measured) before trusting a wide one. Server setup: [localai.md](../localai.md).

| File | Model | Context | Notes |
|---|---|---|---|
| `gemma-4-e2b.yaml` | Gemma-4-E2B | 128K | Fast prefill, strong capability per gigabyte. Good default. |
| `gemma-4-e4b.yaml` | Gemma-4-E4B | 64K | Best broad knowledge and the only 1.000 on MRCR. Slowest. Verified under load. |
| `granite-4.0-h-micro.yaml` | Granite-4.0-H-Micro | 128K | Best tool calling. The pick for agents and unattended work. |
| `granite-4.2-3b.yaml` | Granite-4.2-3B | 64K | Dense sibling of the 4.0 hybrid. Scores lower on everything here. |
| `lfm3-1.7b.yaml` | LFM3-1.7B | 8K | Text model; conservative initial context, not a measured ceiling. |
| `d1-2b.yaml` | d1-2B | 8K | Decision model; tested for text questions. Single-pass requests must fit the physical batch. |
| `d1-vision-cpu.yaml` | d1-2B + F16 projector | 2K | CPU image-decision profile; one chart fixture tested. Batch 1024, so larger images can exceed the limit. Do not send video via LocalAI. |
| `lfm2.5-2.6b.yaml` | LFM2.5-2.6B | 128K | Fastest decoder; small footprint. Scores 0.000 on MRCR. |
| `phi-4-mini.yaml` | Phi-4-mini | 56K | Only model above zero on LongBench. **No tool calls.** |
| `qwen3-4b-instruct-2507.yaml` | Qwen3-4B-Instruct-2507 | 24K | Tool-calling leader; heaviest KV cache, so the narrowest window. |
| `qwen3.5-4b.yaml` | Qwen3.5-4B | 96K | Best maths and reasoning. A thinking model. |
| `smollm3-3b.yaml` | SmolLM3-3B | 64K | Fully open training recipe. **No tool calls.** |
| `kev-4b.yaml` | Kev-4B | 8K | **A decision model, not a chat model**: typed answers on `/v1/systemone`, calibrated. Needs LocalAI v4.12 and a rebuilt backend ([localai.md](../localai.md#decision-models)). |
| `sd-turbo.yaml` | SD-Turbo | — | Everyday image generation, 4 steps |
| `sd-1.5.yaml` | SD 1.5 | — | Baseline, best LoRA and ControlNet ecosystem |
| `sdxl-turbo.yaml` | SDXL-Turbo | — | Most photorealistic; at the edge of 4 GB |

## Install

Copy into LocalAI's models folder — with the menu-bar app:

```bash
cp *.yaml ~/Library/Application\ Support/LocalAI/models/
```

Restart the server from the menu-bar icon; the models appear in `/v1/models`.

## Weights

Weights are referenced by filename and must live inside the models folder: LocalAI rejects a path outside it and the model then silently does not appear. A symlink satisfies this.

| Config expects | Get it from |
|---|---|
| `gemma-4-E2B-it-Q4_K_M.gguf` | [bartowski/google_gemma-4-E2B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF) |
| `gemma-4-E4B-it-Q4_K_M.gguf` | [bartowski/google_gemma-4-E4B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF) |
| `granite-4.0-h-micro-Q4_K_M.gguf` | [ibm-granite/granite-4.0-h-micro-GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF) |
| `granite-4.2-3b-Q4_K_M.gguf` | [ibm-granite/granite-4.2-3b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF) |
| `LFM3-1.7B-Q4_K_M.gguf` | [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) |
| `d1-2B-Q4_K_M.gguf` | [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) |
| `mmproj-d1-2B-F16.gguf` | [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) (CPU image profile) |
| `LFM2.5-2.6B-Q4_K_M.gguf` | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) |
| `Phi-4-mini-instruct-Q4_K_M.gguf` | [bartowski/microsoft_Phi-4-mini-instruct-GGUF](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF) |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | [Qwen/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF) |
| `Qwen3.5-4B-Q4_K_M.gguf` | [bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF) |
| `SmolLM3-Q4_K_M.gguf` | [bartowski/HuggingFaceTB_SmolLM3-3B-GGUF](https://huggingface.co/bartowski/HuggingFaceTB_SmolLM3-3B-GGUF) |
| `Kev-4B-Q4_K_M.gguf` | [ggml-org/Kev-4B-GGUF](https://huggingface.co/ggml-org/Kev-4B-GGUF) |
| `sd_turbo.safetensors` | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) |
| `v1-5-pruned-emaonly-fp16.safetensors` | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| `sd_xl_turbo_1.0_fp16.safetensors` | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) |

```bash
ln -s /path/to/weights.gguf ~/Library/Application\ Support/LocalAI/models/
```

## Maximum context, measured

Except for the LFM3 and d1 starting profiles, each `context_size` here is the largest window the model holds without spilling its KV cache to host RAM, found by loading each model at descending context sizes and comparing generation throughput against its own rate at 4K. A spill is unmistakable — throughput collapses by a factor of 30 or more — so the ceiling is sharp rather than a judgement call.

| Model | Context | tok/s there | tok/s at 4K | First size that spilled |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 24576 | 37.9 | 37.2 | 32768 → 0.9 |
| Qwen3.5-4B | 98304 | 31.0 | 31.1 | 131072 → 1.2 |
| Gemma-4-E2B | 131072 | 44.8 | 45.1 | — (trained ceiling) |
| Gemma-4-E4B | 131072 *(allocates; 64K usable — see below)* | 28.8 | 28.5 | — (trained ceiling) |
| Granite-4.0-H-Micro | 131072 | 39.2 | 41.4 | — (trained ceiling) |
| Granite-4.2-3B | 65536 | 41.3 | 40.9 | 81920 → 0.8 |
| LFM2.5-2.6B | 131072 | 56.4 | 53.3 | — (trained ceiling) |
| LFM3-1.7B | 262144 allocated, profile remains 8192 | 66.0 | 64.3 | — (trained ceiling; q4_0 KV) |
| Phi-4-mini | 57344 | 38.5 | 39.1 | 65536 → 1.3 |
| SmolLM3-3B | 65536 | 46.4 | 46.3 | — (trained ceiling) |

Six models, including LFM3, reach their full trained window in this short-prompt sweep and stop there because the model runs out of context, not because the card runs out of VRAM. The rest are VRAM-bound, and the binding constraint is weight size plus cache geometry rather than parameter count: Qwen3-4B-2507 is the strongest tool caller here but has the heaviest cache, so it gets the narrowest window of any model in the set.

**These are allocation ceilings, and an allocation ceiling is not a usable context.** The sweep above loads each model and generates from a one-token prompt, which proves the KV cache fits and nothing more. Fill that window with real tokens and a model can degrade badly while still "fitting".

Gemma-4-E4B is the worked example. It passes the sweep at 131072, but with a 9,650-token prompt:

| Context | Prefill | Generation | Outcome |
|---|---|---|---|
| 65536 | 46.1 tok/s | 22.1 tok/s | correct needle retrieval |
| 131072 | — | 4.5 tok/s | 5× slower, empty content |
| 131072, ~43K-token prompt | — | — | backend crashed after 105 min |

So its config is set to **64K**, the largest context verified under a real prompt, not the 128K the sweep reports. **The other original eight models have not had this second test.** Their configs use sweep figures, so treat any of them as provisional until exercised with a full prompt — the wider the window, the more likely it overstates. LFM3 retains a conservative 8K setting rather than its 256K allocation result.

Two further cautions:

- **The cliff is one step away.** The sweep used `llama-cli` defaults; a larger `n_batch` needs a bigger compute buffer and will push a model at its ceiling over the edge. If you raise batch size, lower the context.
- **A wide window is not recall.** LFM2.5 holds 128K and scores 0.000 on MRCR. Size the context to what the model can actually use — see [long-context retrieval](../models.md#long-context-retrieval).

## The settings, and why

The original chat profiles use the following settings. The LFM3 and decision profiles use conservative starting contexts and only the settings shown in their YAML; the CPU image profile explicitly disables GPU offload:

| Setting | Why |
|---|---|
| `gpu_layers: 99` | Put every layer on the Radeon. Anything less splits the model and pays a PCIe hop per token. |
| `flash_attention: "true"` | It runs on the GPU on these branches and is the faster option, and a quantized KV cache is impossible without it. |
| `cache_type_k`/`cache_type_v: q4_0` | About 4% slower than f16, and 3.5× more context. Avoid q8_0 unless your backend includes the q8_0 flash-attention guard. |
| `context_size` | See the measured ceilings and starting-profile exceptions above. |
| `f16: true` | Standard for GGUF weights here. |

The image-generation configs set `diffusion_conv_direct:true`, required for correctness rather than speed: without it every image is [colourful noise](../README.md#ki-diffusion-noise). SDXL also needs `keep_vae_on_cpu:true`, since VRAM peaks during VAE decoding.

## Verified

Two different levels of testing sit behind these files.

**Run through LocalAI end to end**, over the OpenAI-compatible API — an arithmetic question checked for the right answer, then three 128-token generations for throughput:

| Config | Result | VRAM | Warm throughput |
|---|---|---|---|
| `granite-4.0-h-micro` | Correct answer | 2.14 GiB | 25.9 / 31.8 / 32.2 tok/s |
| `lfm2.5-2.6b` | Correct answer | 1.90 GiB | 27.2 / 40.8 / 39.8 tok/s |
| `qwen3-4b-instruct-2507` | Correct answer | 3.14 GiB | 26.2 / 31.2 / 30.6 tok/s |
| `qwen3.5-4b` | Correct answer | 3.07 GiB | 23.1 / 26.5 / 27.0 tok/s |
| `sd-turbo` | Recognisable apple, not noise | — | 50.2 s for a 512×512 image, first call |
| `sd-1.5` | Recognisable apple, not noise | — | 80.7 s at 20 steps, first call |
| `sdxl-turbo` | Recognisable apple, not noise | — | 221.9 s, first call |

That run predates the context changes above, so its throughput figures were taken at the older, narrower settings. LFM3 also answered 20 + 34 correctly through LocalAI with a larger output budget; VRAM and sustained throughput were not measured for that request.

**Decisions, run through LocalAI v4.12 end to end:** d1-2B answered a short three-question text request, and the CPU image profile answered one chart question correctly. The model-card video request is not reliable on this LocalAI build; see [the media limitation](../localai.md#lfm3-and-d1-2b). On 4 November 2026, `kev-4b` answered a three-question `/v1/systemone` request (choice, noul, score) and a `/v1/decisions` request, with every layer on the Radeon and probabilities identical to a direct `llama-server` run. Its JevBench accuracy and calibration are in [decision-models.md](../decision-models.md#results). It uses only two of the five text settings below, `gpu_layers` and `flash_attention`: no KV quantization was tested, and its context is the one it was run at, not a swept ceiling.

**Context fit measured with `llama-cli`, not yet run through LocalAI**: `gemma-4-e2b`, `gemma-4-e4b`, `granite-4.2-3b`, `phi-4-mini`, `smollm3-3b`. The backend and flags are identical to the four verified configs, so these are expected to work, but nothing has exercised them over the API.

Notes:

- Image VRAM is not listed: the probe sampled after generation, once the backend had released the memory. Use the figures in [models.md](../models.md#image-models).
- Image timings are cold first-call, including model load and quantization, which dominate a 4-step generation.
- The image backends need `SD_LIBRARY` and the MoltenVK environment set. The menu-bar app does this; a hand-rolled server needs the [wrapper script](../localai.md#images).

## Three rules for a 4 GB card

1. **One model at a time, and switch only when nothing else is in flight.** Only one fits in 4 GB, and LocalAI evicts the previous model to load the next. A sequential caller can switch freely — measured reliable across three models in one session. A caller with requests in flight loses one per switch to the eviction window: exactly one of three concurrent requests failed with `rpc error: code = Unavailable`, in 3 of 3 trials. So a parallel client like OpenCode should stay on one model per session. See [localai.md](../localai.md#troubleshooting).
2. **Give thinking models room.** Qwen3.5 and LFM2.5 emit a reasoning preamble, so `content` stays empty until the block closes and the text arrives in `message.reasoning`. A small `max_tokens` yields an empty answer. Both configs carry a commented-out `chat_template_kwargs` block to disable thinking.
3. **Do not point an agent loop at Phi-4-mini or SmolLM3-3B.** Neither emits parseable `tool_calls` on this stack, so a loop waiting for a function call will silently never get one.
