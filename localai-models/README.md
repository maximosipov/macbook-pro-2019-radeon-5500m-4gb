# LocalAI model configs

Ready-to-use LocalAI model YAML files for the models this repository recommends, with the settings this hardware needs. Background: [localai.md](../localai.md) for the server, [models.md](../models.md) for why these models.

| File | Model | Context | Notes |
|---|---|---|---|
| `granite-4.0-h-micro.yaml` | Granite-4.0-H-Micro | 32K, q4_0 KV | Best tool calling. The pick for agents and unattended work. |
| `lfm2.5-2.6b.yaml` | LFM2.5-2.6B | 32K, q4_0 KV | Fastest and smallest. Most headroom. |
| `qwen3-4b-instruct-2507.yaml` | Qwen3-4B-Instruct-2507 | 16K, q4_0 KV | Tool-calling leader; heaviest KV cache, so context is capped lower. |
| `qwen3.5-4b.yaml` | Qwen3.5-4B | 32K, q4_0 KV | Best maths and reasoning. A thinking model. |
| `sd-turbo.yaml` | SD-Turbo | — | Everyday image generation, 4 steps |
| `sd-1.5.yaml` | SD 1.5 | — | Baseline, best LoRA and ControlNet ecosystem |
| `sdxl-turbo.yaml` | SDXL-Turbo | — | Most photorealistic; at the edge of 4 GB |

## Install

Copy the files into LocalAI's models folder. With the menu-bar app that is:

```bash
cp *.yaml ~/Library/Application\ Support/LocalAI/models/
```

Then restart the server (Restart in the menu-bar icon), and the models appear in `/v1/models`.

## Weights

The YAML files reference weights by filename, and **the weights must live inside the models folder**: LocalAI rejects a path outside it, and the model then silently doesn't appear. A symlink satisfies it.

| Config expects | Get it from |
|---|---|
| `granite-4.0-h-micro-Q4_K_M.gguf` | [ibm-granite/granite-4.0-h-micro-GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF) |
| `LFM2.5-2.6B-Q4_K_M.gguf` | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | [Qwen/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF) |
| `Qwen3.5-4B-Q4_K_M.gguf` | Qwen3.5-4B, Q4_K_M quantization |
| `sd_turbo.safetensors` | [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo) |
| `v1-5-pruned-emaonly-fp16.safetensors` | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| `sd_xl_turbo_1.0_fp16.safetensors` | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) |

```bash
ln -s /path/to/weights.gguf ~/Library/Application\ Support/LocalAI/models/
```

## The settings, and why

Every text config carries the same five, and none of them is a tuning preference:

| Setting | Why |
|---|---|
| `gpu_layers: 99` | Put every layer on the Radeon. Anything less splits the model and pays a PCIe hop per token. |
| `flash_attention: "true"` | It runs on the GPU on these branches and is the faster option, and a quantized KV cache is impossible without it. |
| `cache_type_k`/`cache_type_v: q4_0` | About 4% slower than f16, and 3.5× more context. Avoid q8_0 unless your backend includes the q8_0 flash-attention guard. |
| `context_size` | Sized per model from its measured [context ceiling](../models.md#context-ceilings), leaving room for the weights in 4 GB. |
| `f16: true` | Standard for GGUF weights here. |

The image configs all set `diffusion_conv_direct:true`, which is **required for correctness**, not speed: without it every image is [colourful noise](../README.md#ki-diffusion-noise). SDXL additionally needs `keep_vae_on_cpu:true`, because VRAM peaks during VAE decoding and it won't fit otherwise.

## Verified

Every config in this folder was run through LocalAI on the target machine, each in its **own** server session, over the OpenAI-compatible API. Text models were asked an arithmetic question (checked for the right answer), then three 128-token generations for throughput.

| Config | Result | VRAM | Warm throughput |
|---|---|---|---|
| `granite-4.0-h-micro` | Correct answer | 2.14 GiB | 25.9 / 31.8 / 32.2 tok/s |
| `lfm2.5-2.6b` | Correct answer | 1.90 GiB | 27.2 / 40.8 / 39.8 tok/s |
| `qwen3-4b-instruct-2507` | Correct answer | 3.14 GiB | 26.2 / 31.2 / 30.6 tok/s |
| `qwen3.5-4b` | Correct answer | 3.07 GiB | 23.1 / 26.5 / 27.0 tok/s |
| `sd-turbo` | Recognisable apple, not noise | — | 50.2 s for a 512×512 image, first call |
| `sd-1.5` | Recognisable apple, not noise | — | 80.7 s at 20 steps, first call |
| `sdxl-turbo` | Recognisable apple, not noise | — | 221.9 s, first call |

Notes on what these numbers are and aren't:

- **Image VRAM was not measured.** The probe sampled after generation finished, by which point the backend had released the memory. The figures in [models.md](../models.md#image-models) come from a proper measurement during the run.
- **Image timings are first-call, cold**, and include model load and on-the-fly quantization, which dominate a 4-step generation. A warm server is much faster per image.
- **Text throughput is a warm, sustained-serving figure,** which is the most reproducible measurement on this machine. The first of each three is still partly cold.
- The image models each needed their backend started with `SD_LIBRARY` and the MoltenVK environment set. The menu-bar app does this itself; a hand-rolled server needs the [wrapper script](../localai.md#images).

## Two rules for a 4 GB card

1. **One model per server session.** A model loaded after another returns empty completions with no error. To switch models, restart the server. See [localai.md](../localai.md#troubleshooting).
2. **Give thinking models room.** Qwen3.5 and LFM2.5 emit a reasoning preamble; `content` stays empty until that block closes, and the text arrives in `message.reasoning`. With a small `max_tokens` you get an empty answer that looks like a failure and isn't. Both configs carry a commented-out `chat_template_kwargs` block to turn thinking off.
