# Models for a 4 GB GPU

Which models fit, how fast they run, how much context they hold, and how capable they are. All measurements come from this machine; see [benchmarking.md](benchmarking.md) for the methods and their caveats.

- [Speed and fit](#speed-and-fit)
- [Context ceilings](#context-ceilings)
- [Capability scores](#capability-scores)
- [Choosing a model for unattended work](#choosing-a-model-for-unattended-work)
- [Image models](#image-models)

## Speed and fit

All [Q4_K_M](glossary.md#g-quant), measured with `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3`, one process per model, with the machine allowed to cool between models.

| Model | [Architecture](glossary.md#g-arch) | [pp512](glossary.md#g-pp512) tok/s | [tg128](glossary.md#g-pp512) tok/s | VRAM |
|---|---|---|---|---|
| LFM2.5-2.6B | [LFM2 hybrid](glossary.md#g-lfm2) (short convolution) | 122.3 ± 0.1 | **58.3 ± 0.1** | 1.84 GB — smallest, most room for context |
| SmolLM3-3B | dense | 93.6 ± 14.6 | 51.2 ± 0.1 | 2.28 GB |
| Gemma-4-E2B | gemma4 [PLE](glossary.md#g-arch) | **142.6 ± 0.3** | 50.7 ± 0.2 | 1.45 GB (the embedding table stays in host RAM) |
| Granite-4.1-3B | dense | 78.9 ± 15.2 | 46.1 ± 0.1 | 2.24 GB, KV-bound |
| Phi-4-mini | dense [GQA](glossary.md#g-arch) | 75.2 ± 13.1 | 42.0 ± 0.1 | 3.04 GB — tightest fit here |
| Qwen3-4B-Instruct-2507 | dense [GQA](glossary.md#g-arch) | 59.4 ± 0.6 | 40.5 ± 0.1 | 2.82 GB, heaviest [KV cache](glossary.md#g-kv) (144 KiB/token) |
| Granite-4.2-3B | dense [GQA](glossary.md#g-arch) | 76.1 ± 12.2 | 40.2 ± 10.4 | 2.48 GB |
| Granite-4.0-H-Micro | [Mamba-2](glossary.md#g-ssm) hybrid | 83.6 ± 17.1 | 34.9 ± 11.2 | 1.95 GB |
| Qwen3.5-4B | [Gated DeltaNet](glossary.md#g-gdn) | 59.5 ± 0.6 | 33.6 ± 0.1 | 3.05 GB |
| Gemma-4-E4B | gemma4 [PLE](glossary.md#g-arch) | 55.1 ± 0.5 | 30.9 ± 0.0 | 3.3 GB |

Every model here loads and runs correctly on the Vulkan build with no source changes.

**Read these as a floor.** The GPU idles at 10 MHz, so a cold `tg128` partly measures the clock ramping up. Treat gaps under about 20% as noise, especially on the rows with large error bars (the Granite models), and prefer a [sustained serving measurement](benchmarking.md#sustained-serving-throughput) for "what will I actually get".

**LFM2.5-2.6B is the standout:** the fastest decoder in the set, and the smallest resident, which leaves the most room for a KV cache. Its short-convolution state-space operation (`SSM_CONV`) is correct on this MoltenVK stack, which matters because that operation family is what [produced token salad](README.md#ki-subgroup-size) on earlier hybrid models.

## Context ceilings

**Architecture, not parameter count, decides how much context fits.** The maximum below is the largest window that fits before [spilling to the CPU](glossary.md#g-spill), computed from each model's [KV cache](glossary.md#g-kv) geometry against the **4278 MB** usable, and cross-checked against measured [`ioreg`](glossary.md#g-ioreg) residency.

| Model | [Architecture](glossary.md#g-arch) (attention layers) | KV f16 per token | Max context, f16 KV | Max context, q4_0 KV | Trained for |
|---|---|---|---|---|---|
| Granite-4.0-H-Micro | [Mamba-2](glossary.md#g-ssm) hybrid (4/40) | 8 KiB | **~292K** | ~1.0M (capped) | 1.0M |
| LFM2.5-2.6B | [short convolution](glossary.md#g-lfm2) hybrid (8/30) | 16 KiB | **~131K (capped)** | ~131K (capped) | 131K |
| Qwen3.5-4B | [Gated DeltaNet](glossary.md#g-gdn) hybrid (~8/32) | 32 KiB | ~37K | ~131K | 262K |
| SmolLM3-3B | dense [GQA](glossary.md#g-arch) (36/36) | 72 KiB | ~32K | ~66K (capped) | 66K |
| Granite-4.2-3B | dense GQA (40/40) | 80 KiB | ~27K | ~95K | 131K |
| Phi-4-mini | dense GQA (32/32) | 128 KiB | ~14K | ~49K | 131K |
| Qwen3-4B-2507 | dense GQA (36/36) | 144 KiB | ~10K | ~35K | 262K |

"Capped" means the model's *trained* context runs out before the VRAM does.

The split is stark, and it's the practical payoff of hybrid architectures. Granite-4.0-H-Micro keeps only 4 of its 40 layers as attention — the rest hold a fixed-size recurrent state — so its cache grows at 8 KiB per token and it reaches about 292K tokens even at full f16 precision. LFM2.5 fits its entire 131K trained window with no KV quantization at all. The dense models are VRAM-bound and modest: Qwen3-4B-2507 manages about 10K at f16, and a [quantized cache](llama.cpp.md#kv-cache-precision) is what lifts it to about 35K.

## Capability scores

Measured over `llama-server`'s OpenAI endpoint at temperature 0, with thinking disabled and any reasoning output excluded from grading. All Q4_K_M, on this GPU.

> **These are relative rankings on this hardware under one protocol. They are not comparable to published leaderboard numbers** — the sample sizes are small, and the quantization, subset and prompt format all differ from published runs. Protocols are in [benchmarking.md](benchmarking.md#capability-benchmarks).

| Model | [MATH-500](glossary.md#g-math500) | [IFBench](glossary.md#g-ifbench) strict / loose | [MMLU-Pro](glossary.md#g-mmlu) | [GPQA-D](glossary.md#g-gpqa) | [BFCL AST](glossary.md#g-bfcl) |
|---|---|---|---|---|---|
| Qwen3.5-4B | **0.81** | **0.26** / 0.27 | **0.46** | **0.56** | 0.75 |
| Qwen3-4B-2507 | 0.75 | 0.21 / 0.23 | 0.43 | 0.37 § | **0.88** |
| SmolLM3-3B | 0.74 | 0.08 / 0.11 | 0.29 | 0.29 | n/a ‡ |
| LFM2.5-2.6B | 0.68 | 0.23 / 0.23 | 0.14 † | 0.22 § | 0.69 |
| Phi-4-mini | 0.67 | 0.10 / 0.13 | 0.39 | 0.30 | n/a ‡ |
| Granite-4.0-H-Micro | 0.63 | 0.19 / 0.20 | 0.39 | 0.26 | 0.86 |
| Granite-4.2-3B | 0.62 | 0.19 / 0.24 | 0.29 | 0.22 § | 0.65 |
| *frontier reference* | *~0.98* ᵍ | *~0.83* ᵉ | *~0.90* ᵇ | *~0.955* ᶠ | *~0.78* ᵃ |

Sample sizes: n=100 for MATH-500, IFBench and GPQA-Diamond; n=28 (stratified) for MMLU-Pro; n=80 for BFCL AST.

**How to read it.** Every column is accuracy from 0 to 1. MATH-500 is graded by symbolic equivalence. IFBench strict is the fraction of prompts where *every* verifiable instruction was met; loose tolerates markdown and stray boundary lines. MMLU-Pro is 10-option reasoning (chance floor 0.1); GPQA-Diamond is 4-option graduate science (chance floor 0.25); BFCL AST is single-turn tool-call correctness. The frontier reference row gives scale only — those are published full-benchmark scores from each vendor's own harness, not a target this hardware was measured against.

**Three caveats are measurement artifacts, not model verdicts:**

- **† LFM2.5's MMLU-Pro score of 0.14** is depressed by the answer-token budget. With its reasoning preamble, it often doesn't reach the required `The answer is (X)` before the cap, and scores 0 on those items. Read it as *not measured well here*; its MATH and IFBench scores are mid-pack.
- **‡ Phi-4-mini and SmolLM3 show "n/a" for BFCL.** Both answer tool prompts *in prose* and emit no parseable `tool_calls` through llama.cpp's `--jinja` path on this build. That's a chat-template gap on this stack, likely fixable with a `--chat-template` override or a newer build, not an absence of tool-calling ability. Recorded as n/a rather than 0 so it isn't averaged in.
- **§ GPQA is understated for these rows** by a high unparsed rate: the answer letter had to be recovered from the reasoning channel, and these models often emitted no graded A–D (LFM2.5 57%, Granite-4.2 60%, Qwen3-2507 40% unparsed), scoring 0 there. Their true scores are above the printed value.

**What the numbers say.** Qwen3.5-4B leads on maths, with SmolLM3-3B and Qwen3-4B-2507 close behind. **IFBench is brutal for everything in this class** — the best here is 0.26 against a frontier of about 0.83, making strict instruction-following the clearest gap. Tool calling splits cleanly into models that emit OpenAI `tool_calls` on this stack (Qwen3-4B-2507 and Granite-4.0-H-Micro lead) and models that don't yet. On GPQA-Diamond, only Qwen3.5-4B and Qwen3-4B-2507 sit clearly above the chance floor. Turning thinking on would raise the maths, MMLU and GPQA columns for the reasoning-capable models, at a large cost in throughput; it was disabled here for a like-for-like comparison.

**Coherence.** Every model clears the [correctness battery](benchmarking.md#text-correctness-battery)'s repetition check (repetition ratio ≈ 0.01), including the hybrids. A MATH-500 score of 0.62–0.81 is itself proof that the output is coherent rather than [NaN](glossary.md#g-nan) garbage.

**Long context was not usable for the benchmarks.** [LongBench-v2](glossary.md#g-longbench) and [MRCR](glossary.md#g-mrcr) were run on three models and largely timed out at 12–19K tokens. That's a runtime wall on this card at long context, not a capability verdict.

ᵃ Claude Opus 4.5 (FC), overall BFCL-V4 — [Gorilla leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), 2026. Our column is the single-turn AST subset only.
ᵇ Qwen3.7 Max, full MMLU-Pro — [llm-stats](https://llm-stats.com/benchmarks/mmlu-pro), 2026.
ᵉ Grok 4.3 (medium), IFBench — [Artificial Analysis](https://artificialanalysis.ai/evaluations/ifbench).
ᶠ Gemini 3.1 Pro tier, GPQA Diamond — [Artificial Analysis](https://artificialanalysis.ai/evaluations/gpqa-diamond), 2026.
ᵍ Frontier tier, MATH-500 — near-saturated at about 0.98; no single canonical leaderboard.

Leaderboards move. Re-check these before quoting them.

## Choosing a model for unattended work

If the machine is meant to work on its own — an agent loop, a batch job, a scheduled task — the rankings above are the wrong ones. Speed stops mattering once a task finishes in minutes rather than hours, and three other properties start to:

1. **Does it call tools correctly, including declining to?** An unattended loop fails the moment it invents a function name or fabricates an argument. Negative cases matter more than positive ones.
2. **Does the run survive?** A [`vk::DeviceLostError`](glossary.md#g-device-lost) or a timeout ends the job with no output at all, which is much worse than a slow but complete answer.
3. **Does context growth kill it?** Agent loops accumulate history, and architecture decides how fast the cache grows.

**Granite-4.0-H-Micro is the pick**, with real competition. It has the best tool-calling behaviour measured here (24/24 on the custom set including negative cases, and 0.86 BFCL AST, statistically tied with Qwen3-4B-2507's 0.88), and its fixed-size recurrent state means context growth costs almost no VRAM. It's mid-pack on speed.

**LFM2.5-2.6B is the strongest all-rounder** if footprint and throughput matter: the fastest decoder, the smallest resident, cheap context growth, proper `tool_calls`, and coherent on Vulkan. Choose it over Granite for speed and headroom; choose Granite when tool-call reliability is paramount. Qwen3-4B-Instruct-2507 remains the tool-calling leader and a good default for bounded context.

**Avoid Phi-4-mini and SmolLM3-3B for tool-driven loops.** They emit no parseable `tool_calls` on this stack, so they will silently never call a function. Nothing in the set needs avoiding on stability grounds: all of them are coherent and survive.

**Configuration for unattended work**, all of it trading throughput you don't need:

- **Serve, don't relaunch.** A resident [LocalAI](localai.md) or `llama-server` process avoids paying model load per task, and keeps the GPU warm — which on this machine makes it *faster*.
- **Cap the context deliberately** and let the client truncate, rather than discovering the ceiling as a device loss mid-job. Size it from the [table above](#context-ceilings), with headroom, and use `-fa on -ctk q4_0 -ctv q4_0` if you need more than f16 allows.
- **Supervise the process.** Device loss can't be recovered in-process, so run under `launchd` or another supervisor with a restart policy, and make the client retry idempotently.
- **Don't stack GPU work back to back.** The one reproducible trigger for device loss here was continuous sweeps with no gap.
- **Log [`ioreg`](glossary.md#g-ioreg) residency** alongside your job output. A silent fall back to the CPU still produces correct answers, just about 3× slower, and you want that in the log rather than as a mystery.

**What this doesn't rest on:** no multi-turn agent benchmark was run, so "calls tools correctly" is measured single-turn only. Multi-turn state tracking, recovery after a failed call and long-horizon planning are exactly where 4B-class models are weakest, and nothing here measures them. Treat the recommendation as "the most reliable thing that fits in 4 GB", not as a claim of reliability in absolute terms.

## Image models

| Model | Load options | VRAM | s/step | Use for |
|---|---|---|---|---|
| SD-Turbo | `--type q8_0` | 2049 MB | ~1.7 | Iterating. A 4-step image takes about 7 s of compute. |
| SD 1.5 | fp16 | 2035 MB | ~3.4 | The [LoRA and ControlNet](glossary.md#g-lora) ecosystem, at 20 steps |
| SDXL-Turbo | `--type q8_0 --vae-on-cpu` | 3836 MB | ~9.3 | Final images. Tight against the 4278 MB usable. |

Details and command lines: [stable-diffusion.cpp.md](stable-diffusion.cpp.md#performance).

**Vision models** run through [ollama](ollama.md#performance): Qwen3-VL-4B holds about 32 tok/s with `OLLAMA_IMAGE_MIN_TOKENS=512`, against 8–15 tok/s and decaying at the upstream default of 1024.
