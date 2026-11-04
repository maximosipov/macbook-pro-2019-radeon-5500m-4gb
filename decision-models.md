# Decision models (System One) on a 4 GB GPU

Which [System One](glossary.md#g-system-one) decision models — Jev, Jeva, Kev and their kin — run on this card, how well they decide, and what it costs.

> **Measured on the test machine, 2–3 November 2026,** on the JevBench public split (231 decisions). Jev itself is not measured: it is API-only, and its terms reportedly forbid publishing benchmarks. Model details not measured here come from model cards and project READMEs, and most published scores are self-reported by the people who made the model.

- [What a decision model is](#what-a-decision-model-is)
- [Results](#results)
- [Candidates](#candidates)
- [Prefill and the cached state](#prefill-and-the-cached-state)
- [Runtime paths](#runtime-paths)
- [Evaluations](#evaluations)
- [What's left](#whats-left)

## What a decision model is

An LLM answers a classification question by generating text, which code then parses. A decision model skips the generation step. It reads a block of *state* once, takes one or more typed questions, and returns a probability over the declared options in a single forward pass:

| Question type | Returns | Code equivalent |
|---|---|---|
| **choice** | A probability for each listed option, optionally including "none of these" | `switch` |
| **noul** | P(yes) | `if` |
| **score** | A value on a declared ordered scale | threshold or sort |

Because the state is prefilled once and every question runs against it, ten questions about one ticket cost roughly what one costs. And because no tokens are generated, there is nothing to parse: no malformed JSON, no prose answer where a tool call was wanted. That is the failure that rules [Phi-4-mini and SmolLM3](models.md#choosing-a-model-for-unattended-work) out of agent loops here.

TypeSafe AI introduced the category with **Jev** on 15 September 2026 ([coverage](https://simonwillison.net/2026/Sep/21/jev/)). Jev is closed and API-only. Open models with the same interface appeared within weeks, and that is what this page evaluates.

## Results

[JevBench](https://benchlm.ai/benchmarks/jevbench) public split as shipped: 231 decisions in three tiers (48 easy, 72 standard, 111 hard), scored with JevBench's own `score_task`. The general models are read SemIf-style, from the option letters' logprobs in `llama-server` on the Radeon at Q4_K_M. Kev and Jeva run their own heads in PyTorch.

| Model | Path | Easy | Standard | Hard | **All** | Intelligence | ECE | Median s / decision | Hard median s |
|---|---|---|---|---|---|---|---|---|---|
| **Jeva-4B** | CPU, fp32 | 1.00 | 0.96 | **0.75** | **0.87** | **81.8** | 0.098 | 3.7 | 13.4 |
| Qwen3.5-4B | Radeon, Q4_K_M logprobs | 0.98 | 0.96 | 0.60 | 0.79 | 72.2 | 0.060 | 3.6 | 11.6 |
| Gemma-4-E4B | Radeon, Q4_K_M logprobs | 1.00 | 0.97 | 0.54 | 0.77 | 69.6 | 0.173 | 3.9 | 12.4 |
| **Kev-4B** | CPU, fp32 | 1.00 | 0.93 | 0.54 | 0.76 | 67.2 | **0.038** | 3.1 | 11.9 |
| Qwen3-4B-Instruct-2507 | Radeon, Q4_K_M logprobs | 1.00 | 0.82 | 0.44 | 0.68 | 54.7 | 0.304 | 3.4 | 11.5 |
| Gemma-4-E2B | Radeon, Q4_K_M logprobs | 1.00 | 0.81 | 0.40 | 0.65 | 51.1 | 0.319 | 1.9 | 5.0 |
| **Kev-0.8B** | CPU, fp32 | 1.00 | 0.81 | 0.37 | 0.64 | 49.4 | 0.044 | **0.6** | **2.6** |
| Granite-4.0-H-Micro | Radeon, Q4_K_M logprobs | 1.00 | 0.61 | 0.43 | 0.61 | 42.4 | 0.314 | 3.0 | 8.2 |
| LFM2.5-2.6B | Radeon, Q4_K_M logprobs | — | — | — | *invalid* | — | — | — | — |
| Kev-4B | Radeon via MPS, fp16 | — | — | — | *does not fit* | — | — | — | — |

*Intelligence* is JevBench v1.3's chance-corrected index over the public tiers only: the judge tier is held out and its weight renormalised away. So it is not comparable with leaderboard figures, such as SemIf over Qwen3.5-4B's 68.66. *ECE* is top-label, 10 bins. Latencies are wall time per decision, median prompt 600–700 tokens, each state new to the model.

What the results say:

- **Jeva-4B is the best decision model on this machine, by a wide margin on hard items** (0.75 against 0.60 for the next best). It leads on choice and score questions (0.92 and 0.89). On noul it trails Qwen3.5-4B (0.76 against 0.81).
- **Kev is the calibrated one.** Kev-4B's ECE of 0.038 sits inside its published 0.017–0.101, and Kev-0.8B's is 0.044. Every logprob-read general model except Qwen3.5-4B is badly overconfident (raw ECE 0.17–0.32, best-fit temperature 3–10). A fitted temperature brings them to 0.07–0.13, but that needs labelled data per model.
- **Qwen3.5-4B read through logprobs is the best GPU option.** It is the most accurate thing that runs on the Radeon (0.79) and the only general model that is calibrated out of the box (temperature 1.35). Its outcomes agree with SemIf's published Qwen3.5-4B BF16 run on 96.5% of items (0.79 against 0.81), so the Q4_K_M Vulkan readout costs little.
- **The CPU is as fast as the Radeon here.** Kev-4B and Jeva-4B in fp32 PyTorch take about the same time per decision as Qwen3.5-4B in llama.cpp on the GPU (3.1–3.7 s median). Each full pass took about 50 minutes, against 45 for Qwen3.5-4B on the GPU. Kev-0.8B is the fastest model measured at 0.6 s, and it leaves the GPU free for the model doing the work.
- **LFM2.5-2.6B's result is invalid, not low.** The option letters carry none of its next-token probability (letter mass 0.000), so the readout is reading noise. It needs a different answer format before it can be scored.

## Candidates

| Model | Maker | Built on | Weights | On this machine | Verdict |
|---|---|---|---|---|---|
| [Jev](https://www.dsebastien.net/jev-and-system-one-models-a-semantic-if-statement-for-your-code/) | TypeSafe AI | Undisclosed | **None**, API only | — | A reference to compare against, not a candidate. Its terms reportedly forbid publishing benchmark results, so its numbers stay off this site. |
| [Jeva-4B](https://huggingface.co/LIQIIIII/Jeva-4B) | Qi Li, Xinchao Wang (NUS) | Qwen3.5-4B + adapter + decision head (13.4M trainable) | Safetensors, Apache-2.0; no GGUF | CPU fp32 only, on its reference path without the compiled kernels. Peaks near 19 GB of RAM while loading. | **Use when accuracy matters most.** The most accurate model measured (0.87; 0.75 hard), and exactly permutation-invariant as claimed. Paper: arXiv 2610.18773. |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Jared Palmer | Qwen3.5-4B-Base + LoRA r16 (33.8M) + pointer head | Safetensors, Apache-2.0; community GGUF 2.5 GB (`turintech/Kev-4B-Artemis-GGUF`), untested | CPU fp32 works. MPS fp16 runs out of memory. | **Use when the probability itself matters.** Best calibration measured (ECE 0.038), at Qwen3.5-4B-level accuracy. Implements TypeSafe's `POST /v1/systemone` API. |
| Kev-0.8B | Jared Palmer | Qwen3.5-0.8B + LoRA + pointer head | Apache-2.0 | CPU fp32, and MPS fp16 with the same answers | **Use as the router.** Fastest measured (0.6 s median), well calibrated (0.044), and small enough to sit in RAM beside a GPU worker. Accuracy about matches Gemma-4-E2B's (0.64 against 0.65). |
| Kev-9B | Jared Palmer | Qwen3.5-9B | Community GGUF 4.8 GB | **Does not fit** (4278 MB usable) | Out. |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev) | Theo Lee | **Any** open model: reads option-label logits, no trained head | MIT code; uses the base model's weights | Its readout reproduced over the fork's `llama-server` on every bake-off model | **Use with Qwen3.5-4B** if the decision must run on the GPU. Gemma-4-E4B is close on accuracy but overconfident. |
| vLLM-SR d3-mini | vLLM Semantic Router | 4.5B | Apache-2.0, GGUF | Not run | A routing specialist; listed as GGUF-loadable. |
| Every 25–35B decision model (Blink, Sieve, Decider, JEV-35B, …) | various | — | — | Does not fit | Out. These lead the open leaderboards, and none is within reach of 4 GB. |

## Prefill and the cached state

A decision model does no decoding: its whole cost is [prefill](glossary.md#g-prefill). pp512 matters here, not generation speed. Every dedicated decision model that fits is built on Qwen3.5, the backbone with the slowest prefill on this card (59.5 pp512).

The cache test took the 6 longest hard states (median 2,750 tokens) and asked about each 5 times through `llama-server`, changing only the option order. The first request ran with the prompt cache off, the other four with it on:

| Backbone | Layers | Cold, median | Cached, median | Tokens recomputed per cached request |
|---|---|---|---|---|
| Gemma-4-E2B | attention | 22.2 s | **1.3 s** | ~105 (the question only) |
| Qwen3-4B-Instruct-2507 | attention | 58.6 s | 2.8 s | ~105 (the question only) |
| LFM2.5-2.6B | [conv hybrid](glossary.md#g-lfm2) | 25.2 s | 4.4 s | 516 |
| Granite-4.0-H-Micro | [Mamba-2 hybrid](glossary.md#g-ssm) | 32.9 s | 5.7 s | 516 |
| Qwen3.5-4B | [Gated DeltaNet](glossary.md#g-gdn) hybrid | 44.4 s | 7.1 s | 516 |

- **The cache works, and it is the whole game.** A new state costs 22–59 s; a further question about it costs 1–7 s. Ask every question about a state in one request, or keep its slot warm.
- **Hybrid backbones recompute a fixed 516 tokens per cached request**, against about 105 for attention-only ones. That fits a recurrent state that llama.cpp can only restore from a saved checkpoint, not rewind token by token. It hits Qwen3.5 hardest, the backbone every decision model here uses: 7.1 s cached against Gemma-4-E2B's 1.3 s.
- **States longer than about 2.7K tokens are unmeasured.** Qwen3.5's prefill gets worse with length on MoltenVK. It took 1032 s per 12K-token [LongBench](models.md#long-context-retrieval) item, so expect minutes per decision on an 8K-token state.

The cache was tested through `llama-server` only. Kev's and Jeva's own state caches were not measured.

## Runtime paths

| Path | Models | Status on this machine |
|---|---|---|
| **Option logprobs from `llama-server`**: one prefill, read the probability of each option's label token | SemIf; any GGUF in [localai-models](localai-models/README.md) | **Works** on the Vulkan fork. SemIf's exact prompt and payload, rendered with each GGUF's chat template (thinking off). Check the letter mass per model: LFM2.5 puts none of its probability on the letters. |
| **PyTorch from conda-forge** | Kev, Jeva | **Works.** PyPI's last x86_64 macOS PyTorch is 2.2, but conda-forge ships osx-64 builds of 2.8 and 2.9.1, recent enough for Qwen3.5 in `transformers`. Kev runs through its own `kev.serve`; Jeva runs in process, on the slow reference fallbacks for its fla and causal-conv1d kernels. CPU fp32 is both the fastest path and the correctness reference. |
| **MPS on the Radeon** | Kev | Not worth it. PyTorch caps MPS allocations on this card at 6.77 GB, so Kev-4B in fp16 (about 8 GB) runs out of memory while loading. Kev-0.8B fits and gives the CPU's answers (largest probability difference 0.004), but at 2.5 s median it is 4× slower than the CPU. |
| **Merged backbone in llama.cpp + head outside it** | Kev, Jeva | Not attempted. The GGUF carries the backbone, and the head would read llama.cpp's per-token embeddings (`--embeddings`, pooling `none`). It is the only way to get a trained head onto the GPU at Q4_K_M. |

Two hardware facts carry over from the rest of this repository. There is **no bfloat16**, and Kev serves in bf16 by default, so load fp16, fp32 or a GGUF quant. And **one model is resident at a time** on the GPU. That makes the CPU paths more useful than they look: Kev or Jeva on the CPU never evicts the model doing the work.

## Evaluations

| Eval | What it establishes | Result here |
|---|---|---|
| **GPU against CPU probabilities** | The Vulkan path computes the option probabilities the CPU does on the same GGUF | 120 easy and standard decisions. Gemma-4-E2B: largest difference 0.027, mean 0.0005, no answer changed. Qwen3.5-4B: largest 0.125, mean 0.008, 2 answers changed. **Neither meets the planned pass mark of 0.01.** Small on average, but one probability moved by 0.125, so the GPU path is not exact. Kev-0.8B on MPS against CPU: largest 0.004, passes. |
| **JevBench public split** | Accuracy across choice, noul and score decisions | See [Results](#results). |
| **[Calibration](glossary.md#g-ece)**: ECE and Brier | Whether "80%" means right 80% of the time | See [Results](#results). Temperature fitted on one half (split by paraphrase group), scored on the other. On 231 items this is noisy: it made Kev's ECE worse (0.038 to 0.082), which says the raw figure was already about as good as this sample can show. Kev was measured at fp32, not quantized. |
| **Permutation flip rate** | Whether the answer changes when the options are shuffled | One shuffle per item. **Jeva-4B: no answer changed, largest probability change 1.3 × 10⁻⁶,** invariant by construction as claimed. Qwen3.5-4B: 18 of 231 answers changed (8%), largest change 0.58. Gemma-4-E2B: 32 of 231 (14%), largest 1.0. Accuracy barely moved in either case, so label-logit reading is position-biased item by item rather than overall. |
| **Cold and cached latency** | Whether a second question about a state reuses its prefill | See [Prefill and the cached state](#prefill-and-the-cached-state). Measured at about 2.7K tokens only, not at 256 or 8K. |
| **Null-option cases** | Declining correctly when no listed option fits | Not run. |
| **Agreement with Jev** | How close a local model gets to the hosted original | Not run, and its result would stay local. |

How the [benchmarking rules](benchmarking.md#the-rules-that-follow) applied:

- **One protocol, every model:** the same items and option-order seeds. Prompt formats differ by path, because Kev and Jeva take structured requests rather than SemIf's chat prompt.
- **Check the letter mass** for logprob readouts: it is how LFM2.5's invalid result showed up. Every other logprob-read model put more than 99.9% of its next-token probability on the letters.
- **Failures are not wrong answers.** No run recorded one, apart from the Kev-4B MPS load, which is reported as not fitting.

## What's left

1. **Null-option cases**, recast from the 24-case [custom tool set](benchmarking.md#capability-benchmarks) with negatives included. Jeva is the only candidate that claims to handle "none of these".
2. **Kev-4B's GGUF plus its head outside llama.cpp**: the one route to a calibrated trained head on the GPU, and a check on what Q4_K_M does to Kev's calibration.
3. **Latency at 8K-token states**, on Qwen3.5-4B and Gemma-4-E2B.
4. **A different answer format for LFM2.5**, then a rerun.
5. **Kev-0.8B as the router** for the [OpenCode agents](opencode-agents/README.md), on the CPU beside a GPU worker.
