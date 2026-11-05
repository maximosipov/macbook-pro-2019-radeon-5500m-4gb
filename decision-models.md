# Decision models (System One) on a 4 GB GPU

Which [System One](glossary.md#g-system-one) decision models — Jev, Jeva, Kev and their kin — run on this card, how well they decide, and what it costs.

> **Measured on the test machine, 2–5 November 2026,** on the JevBench public split (231 decisions). Native decision support in llama.cpp and LocalAI was tested on 4 November, on the fork branches as rebased onto upstream that day. Jev itself is not measured: it is API-only, and its terms reportedly forbid publishing benchmarks. Model details not measured here come from model cards and project READMEs, and most published scores are self-reported by the people who made the model.

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

[JevBench](https://benchlm.ai/benchmarks/jevbench) public split as shipped: 231 decisions in three tiers (48 easy, 72 standard, 111 hard), scored with JevBench's own `score_task`. The general models are read SemIf-style, from the option letters' logprobs in `llama-server` on the Radeon at Q4_K_M. Kev and Jeva run their own heads in PyTorch; Kev-4B and d1-2B also run natively in llama.cpp on the Radeon (see [Runtime paths](#runtime-paths)).

| Model | Path | Easy | Standard | Hard | **All** | Intelligence | ECE | Median s / decision | Hard median s |
|---|---|---|---|---|---|---|---|---|---|
| **Jeva-4B** | CPU, fp32 | 1.00 | 0.96 | **0.75** | **0.87** | **81.8** | 0.098 | 3.7 | 13.4 |
| **d1-2B** | CPU + Radeon (5 GPU layers), Q4_K_M native llama.cpp | 1.00 | 0.96 | 0.60 | 0.80 | 72.8 | 0.056 | 3.1 | 6.8 |
| Qwen3.5-4B | Radeon, Q4_K_M logprobs | 0.98 | 0.96 | 0.60 | 0.79 | 72.2 | 0.060 | 3.6 | 11.6 |
| Gemma-4-E4B | Radeon, Q4_K_M logprobs | 1.00 | 0.97 | 0.54 | 0.77 | 69.6 | 0.173 | 3.9 | 12.4 |
| **Kev-4B** | CPU, fp32 | 1.00 | 0.93 | 0.54 | 0.76 | 67.2 | **0.038** | 3.1 | 11.9 |
| **Kev-4B** | Radeon, Q4_K_M, native llama.cpp | 1.00 | 0.93 | 0.51 | 0.75 | 65.5 | 0.043 | 2.3 | 15.2 |
| Qwen3-4B-Instruct-2507 | Radeon, Q4_K_M logprobs | 1.00 | 0.82 | 0.44 | 0.68 | 54.7 | 0.304 | 3.4 | 11.5 |
| Gemma-4-E2B | Radeon, Q4_K_M logprobs | 1.00 | 0.81 | 0.40 | 0.65 | 51.1 | 0.319 | 1.9 | 5.0 |
| **Kev-0.8B** | CPU, fp32 | 1.00 | 0.81 | 0.37 | 0.64 | 49.4 | 0.044 | **0.6** | **2.6** |
| Granite-4.0-H-Micro | Radeon, Q4_K_M logprobs | 1.00 | 0.61 | 0.43 | 0.61 | 42.4 | 0.314 | 3.0 | 8.2 |
| **d1-2B** † | Radeon, Q4_K_M, native llama.cpp | 1.00 (48/48) | 0.96 (69/72) | *incomplete* (1/111) | — | — | — | — | — |
| LFM2.5-2.6B | Radeon, Q4_K_M logprobs | — | — | — | *invalid* | — | — | — | — |
| Kev-4B | Radeon via MPS, fp16 | — | — | — | *does not fit* | — | — | — | — |

**† Full-GPU d1-2B:** Easy and standard are complete; only one hard item completed before device loss. The separate five-GPU-layer run completed all 231 cases and is listed above. Its timings describe mixed CPU/Radeon execution. See [d1-2B results and batch limit](#d1-2b-results-and-batch-limit).

*Intelligence* is JevBench v1.3's chance-corrected index over the public tiers only: the judge tier is held out and its weight renormalised away. So it is not comparable with leaderboard figures, such as SemIf over Qwen3.5-4B's 68.66. *ECE* is top-label, 10 bins. Latencies are wall time per decision, median prompt 600–700 tokens, each state new to the model.

What the results say:

- **Jeva-4B is the best decision model among the complete runs, by a wide margin on hard items** (0.75 against 0.60 for the next best). It leads on choice and score questions (0.92 and 0.89). On noul it trails Qwen3.5-4B (0.76 against 0.81).
- **Kev has the lowest measured calibration error.** Its probabilities already match its hit rate: Kev-4B's ECE of 0.038 sits inside its published 0.017–0.101, Kev-0.8B's is 0.044, and the best-fit temperature for both is 0.93–1.05, so fitting changes nothing. Qwen3.5-4B is slightly overconfident (temperature 1.35, ECE 0.060). d1-2B has ECE 0.056 on the complete mixed-offload run; this small sample does not establish calibration for another workload. The other logprob-read general models are badly overconfident (raw ECE 0.17–0.32, temperature 3–10). A fitted temperature brings them to 0.07–0.13, but that needs labelled data per model.
- **Jeva-4B is underconfident.** Its best-fit temperature is 0.55: it is right more often than its probabilities say. A raw threshold ("act above 0.8") makes it hold back more than it should; it needs sharpening, not flattening.
- **Kev-4B survives Q4_K_M on the Radeon.** Through llama.cpp's native decision support it scores 0.745 against 0.758 for the fp32 PyTorch reference, keeps its calibration (ECE 0.043), and changes 3 of 231 answers. It needs about 3.7 GB of VRAM instead of 16 GB or more of RAM, and it is the calibrated option on the GPU.
- **Qwen3.5-4B read through logprobs is the most accurate fully GPU-offloaded option among complete runs** (0.79). Its outcomes agree with SemIf's published Qwen3.5-4B BF16 run on 96.5% of items (0.79 against 0.81), so the Q4_K_M Vulkan readout costs little.
- **The CPU is about as fast as the Radeon here.** Kev-4B and Jeva-4B in fp32 PyTorch take about the same time per decision as Qwen3.5-4B in llama.cpp on the GPU (3.1–3.7 s median), and a full pass took about 50 minutes against 45. Kev-4B on the Radeon is faster on short states (2.3 s median) but slower on long hard ones (15.2 s against 11.9 s), so its full pass took 56 minutes. Kev-0.8B is the fastest model measured at 0.6 s, and it leaves the GPU free for the model doing the work.
- **LFM2.5-2.6B's result is invalid, not low.** The option letters carry none of its next-token probability (letter mass 0.000), so the readout is reading noise. It needs a different answer format before it can be scored.

### d1-2B: results and batch limit

[Liquid AI d1-2B](https://huggingface.co/LiquidAI/d1-2B) is a 1.92B-parameter text and vision decision model, built on LFM3-VL-2B and released under LFM Open License v1.0. The official [Q4_K_M GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) was tested on 4 November with the forked llama.cpp `821a0ad27`, Vulkan on the Radeon, 8K context and flash attention. The full-GPU result is a **partial** JevBench run:

| Tier | Completed | Correct | Median per decision |
|---|---:|---:|---:|
| Easy | 48 / 48 | 48 / 48 | 0.73 s |
| Standard | 72 / 72 | 69 / 72 | 1.06 s |
| Hard | 1 / 111 | 1 / 1 | 34.7 s |

**Complete mixed-offload run (5 November):** using `-ngl 5 -b 4096 -ub 4096`, d1 completed all 231 decisions with no request failures: 48/48 easy, 69/72 standard and 67/111 hard, or **184/231 (0.797)** overall. JevBench intelligence is 72.8 and top-label ECE is 0.056; median latency is 3.1 s overall and 6.8 s on hard items. These timings include CPU work and are not full-GPU timings. Across the 121 items completed on both configurations, the selected answer did not change; maximum probability difference was 0.0204 (mean maximum per item 0.0010).

The model reads each question in a single physical batch. With llama-server's default 512-token physical batch, longer questions returned HTTP 500. A rerun at `-b 4096 -ub 4096` completed the easy and standard tiers and one 3,687-token hard question, then lost the Vulkan device on the next hard question. The remaining 110 hard requests failed after device loss; these are failures, not wrong answers, and do not support an overall score, calibration estimate or ranking. Startup GPU allocation was about 1.78 GB; fitting the weights and buffers does not establish reliability under a long prompt. The failure's cause is unconfirmed.

The [LocalAI profile](localai-models/d1-2b.yaml) was checked on a short text request with the default 512-token batch. It returns typed `choice`, `noul` and `score` answers; image/video input and longer LocalAI batches were not tested. No projector is configured in that profile.

**Media checks:** On 5 November, a separate CPU-only llama-server with the official F16 projector selected the correct answer on all four existing image fixtures (chart, object count, price and sign). The [model card's example video](https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/4cba700bd771f44d72b549253da025c32e944d42/sample_demo_1.mp4) also returned “reading” correctly at batch/ubatch 4096; it was rejected at 2048 because its rendered input was 2320 tokens. These are five smoke checks, not a vision/video benchmark; A separate CPU-only LocalAI profile also returned the correct chart answer. The media path was not tested on the Radeon, and LocalAI video was not tested. The card documents additional model capabilities, but llama.cpp currently supports only a subset. See [LocalAI](localai.md#lfm3-and-d1-2b) for the batch limit.

## Candidates

| Model | Maker | Built on | Weights | On this machine | Verdict |
|---|---|---|---|---|---|
| [Jev](https://www.dsebastien.net/jev-and-system-one-models-a-semantic-if-statement-for-your-code/) | TypeSafe AI | Undisclosed | **None**, API only | — | A reference to compare against, not a candidate. Its terms reportedly forbid publishing benchmark results, so its numbers stay off this site. |
| [Jeva-4B](https://huggingface.co/LIQIIIII/Jeva-4B) | Qi Li, Xinchao Wang (NUS) | Qwen3.5-4B + adapter + decision head (13.4M trainable) | Safetensors, Apache-2.0; no GGUF | CPU fp32 only, on its reference path without the compiled kernels. Peaks near 19 GB of RAM while loading. Neither llama.cpp nor LocalAI supports it. | **Use when accuracy matters most.** The most accurate model measured (0.87; 0.75 hard), and exactly permutation-invariant as claimed. Underconfident, so don't threshold its raw output. Paper: arXiv 2610.18773. |
| [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | Liquid AI | LFM3-VL-2B decision model | [Official GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF), Q4_K_M; LFM Open License v1.0 | **Radeon**, through native llama.cpp; short text decisions also tested through LocalAI. Default 512-token physical batch limits question length; a 4096-token batch lost the device during a hard case. | **Complete mixed-offload run:** 0.797 overall, 0.604 hard, ECE 0.056; full-GPU hard run incomplete. Four image fixtures and one video smoke test pass on a separate CPU path; the CPU LocalAI chart check passed. Radeon media remains untested. |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Jared Palmer | Qwen3.5-4B-Base + LoRA r16 (33.8M) + pointer head | Safetensors, Apache-2.0; official GGUF `ggml-org/Kev-4B-GGUF` (Q4_K_M, 3.03 GB) | **Radeon**, through llama.cpp's native decision support: about 3.7 GB of VRAM, also served by LocalAI's `/v1/systemone`. CPU fp32 in PyTorch. MPS fp16 runs out of memory. | **Use when the probability itself matters.** Best calibration measured (ECE 0.038 fp32, 0.043 at Q4_K_M), at Qwen3.5-4B-level accuracy, and the calibrated option on the GPU. Implements TypeSafe's `POST /v1/systemone` API. |
| Kev-0.8B | Jared Palmer | Qwen3.5-0.8B + LoRA + pointer head | Apache-2.0 | CPU fp32, and MPS fp16 with the same answers | **Use as the router.** Fastest measured (0.6 s median), well calibrated (0.044), and small enough to sit in RAM beside a GPU worker. Accuracy about matches Gemma-4-E2B's (0.64 against 0.65). |
| Kev-9B | Jared Palmer | Qwen3.5-9B | Community GGUF 4.8 GB | **Does not fit** (4278 MB usable) | Out. |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev) | Theo Lee | **Any** open model: reads option-label logits, no trained head | MIT code; uses the base model's weights | Its readout reproduced over the fork's `llama-server` on every bake-off model | **Use with Qwen3.5-4B** if the decision must run on the GPU. Gemma-4-E4B is close on accuracy but overconfident. |
| vLLM-SR d3-mini | vLLM Semantic Router | 4.5B | Apache-2.0, GGUF | Not run | A routing specialist; listed as GGUF-loadable. |
| Every 25–35B decision model (Blink, Sieve, Decider, JEV-35B, …) | various | — | — | Does not fit | Out. These lead the open leaderboards, and none is within reach of 4 GB. |

## Prefill and the cached state

A decision model does no decoding: its whole cost is [prefill](glossary.md#g-prefill). pp512 matters here, not generation speed. Kev and Jeva use Qwen3.5, whose prefill is 59.5 pp512 on this card. d1-2B instead uses an LFM3 backbone; its cache behavior was not included in the test below.

The cache test took the 6 longest hard states (median 2,750 tokens) and asked about each 5 times through `llama-server`, changing only the option order. The first request ran with the prompt cache off, the other four with it on:

| Backbone | Layers | Cold, median | Cached, median | Tokens recomputed per cached request |
|---|---|---|---|---|
| Gemma-4-E2B | attention | 22.2 s | **1.3 s** | ~105 (the question only) |
| Qwen3-4B-Instruct-2507 | attention | 58.6 s | 2.8 s | ~105 (the question only) |
| LFM2.5-2.6B | [conv hybrid](glossary.md#g-lfm2) | 25.2 s | 4.4 s | 516 |
| Granite-4.0-H-Micro | [Mamba-2 hybrid](glossary.md#g-ssm) | 32.9 s | 5.7 s | 516 |
| Qwen3.5-4B | [Gated DeltaNet](glossary.md#g-gdn) hybrid | 44.4 s | 7.1 s | 516 |

- **The cache works, and it is the whole game.** A new state costs 22–59 s; a further question about it costs 1–7 s. Ask every question about a state in one request, or keep its slot warm.
- **Hybrid backbones recompute a fixed 516 tokens per cached request**, against about 105 for attention-only ones. That fits a recurrent state that llama.cpp can only restore from a saved checkpoint, not rewind token by token. It hits Qwen3.5 hardest, the backbone Kev and Jeva use: 7.1 s cached against Gemma-4-E2B's 1.3 s.
- **The cache test did not measure states longer than about 2.7K tokens.** Qwen3.5's prefill gets worse with length on MoltenVK. It took 1032 s per 12K-token [LongBench](models.md#long-context-retrieval) item, so expect minutes per decision on an 8K-token state.

The cache was tested through `llama-server`'s logprob readout only. Kev's and Jeva's own state caches were not measured, and llama.cpp turns its prompt cache off for native decision models by default (`--cache-ram N` turns it back on). Jeva cannot share a state across questions at all: its input puts the question before the state, so every question re-encodes the whole state.

## Runtime paths

| Path | Models | Status on this machine |
|---|---|---|
| **Option logprobs from `llama-server`**: one prefill, read the probability of each option's label token | SemIf; any GGUF in [localai-models](localai-models/README.md) | **Works** on the Vulkan fork. SemIf's exact prompt and payload, rendered with each GGUF's chat template (thinking off). Check the letter mass per model: LFM2.5 puts none of its probability on the letters. |
| **PyTorch from conda-forge** | Kev, Jeva | **Works.** PyPI's last x86_64 macOS PyTorch is 2.2, but conda-forge ships osx-64 builds of 2.8 and 2.9.1, recent enough for Qwen3.5 in `transformers`. Kev runs through its own `kev.serve`; Jeva runs in process, on the slow reference fallbacks for its fla and causal-conv1d kernels. CPU fp32 is both the fastest path and the correctness reference. |
| **MPS on the Radeon** | Kev | Not worth it. PyTorch caps MPS allocations on this card at 6.77 GB, so Kev-4B in fp16 (about 8 GB) runs out of memory while loading. Kev-0.8B fits and gives the CPU's answers (largest probability difference 0.004), but at 2.5 s median it is 4× slower than the CPU. |
| **Native decisions in llama.cpp**: `llama-server` `/v1/systemone` | Kev-4B and d1-2B measured (d1 complete with five GPU layers; full GPU partial). Upstream also supports Lev, Laya, Julia-1, OpenJev, Nimble, Clef, Granite Switch, LFM2-d1 and others | **Works on the Radeon.** Upstream added it on 2 October 2026 ([#29818](https://github.com/ggml-org/llama.cpp/pull/29818)), after the fork first branched. The fork was [rebased onto upstream](llama.cpp.md#changes-on-this-branch) on 4 November to get it, with one hand merge in `ggml-vulkan.cpp`. Kev's head is a single matrix multiply on the backbone's output, so no new GPU code is needed; all 33 Kev layers sit on the GPU in about 3.7 GB. d1's long-batch run lost the Vulkan device; see [its partial results](#d1-2b-results-and-batch-limit). |
| **LocalAI's Decisions API**: `/v1/systemone` and `/v1/decisions` | Same as llama.cpp | **Works end to end** with the LocalAI fork, now on v4.12, and its `llama-cpp` backend. See [localai.md](localai.md#decision-models). |
| **vllm.cpp**, LocalAI's other decision backend | Kev-0.8B, Laya, Tev1, Nimble, CLM | **Does not use this card.** On macOS LocalAI packages only a Metal/MLX build, for Apple Silicon. vllm.cpp's Vulkan engine assumes an integrated GPU: it allocates every tensor from host-visible, coherent memory, and MoltenVK offers the Radeon that only in system RAM. Its quantized-weight tier and Qwen3.5's recurrences also run as CPU kernels. Not built or run here. |
| **Jeva in llama.cpp** | Jeva-4B | Not supported. Its backbone maps onto existing embedding mode (`--embeddings --pooling last` after merging the LoRA), and its head is about 13M parameters of set attention, cheap on the CPU. Each option needs its own backbone pass, so native support means a new kind of family in `server-decision.cpp`. |

Two hardware facts carry over from the rest of this repository. There is **no bfloat16**, and Kev serves in bf16 by default, so load fp16, fp32 or a GGUF quant. And **one model is resident at a time** on the GPU. That makes the CPU paths more useful than they look: Kev or Jeva on the CPU never evicts the model doing the work.

## Evaluations

| Eval | What it establishes | Result here |
|---|---|---|
| **GPU against CPU probabilities** | The Vulkan path computes the option probabilities the CPU does on the same GGUF | 120 easy and standard decisions. Gemma-4-E2B: largest difference 0.027, mean 0.0005, no answer changed. Qwen3.5-4B: largest 0.125, mean 0.008, 2 answers changed. **Neither meets the planned pass mark of 0.01.** Small on average, but one probability moved by 0.125, so the GPU path is not exact. Native Kev-4B Q4_K_M: largest 0.040, mean 0.004, no answer changed; closer, but still above 0.01. d1-2B Q4_K_M, 120 easy and standard items, full GPU against CPU: largest 0.0208, mean maximum per item 0.00175, no answers changed; above the 0.01 pass mark. CPU ran with two threads while LFM3 accuracy tests were active, so its latency is not compared. Kev-0.8B on MPS against CPU: largest 0.004, passes. |
| **Quantization** | What Q4_K_M in llama.cpp costs a trained head | Kev-4B Q4_K_M on the Radeon against fp32 PyTorch, all 231 decisions: 3 answers changed, largest probability difference 0.195, mean 0.021, outcome the same on 98.7% of items. Accuracy 0.745 against 0.758, ECE 0.043 against 0.038. d1-2B: official F16 against Q4_K_M on the same two-thread CPU path, 120 easy and standard cases: one answer changed, largest probability difference 0.295, mean maximum per item 0.0104; accuracy 0.967 F16 and 0.975 Q4. Hard cases and a full-run ECE comparison were not tested at F16. |
| **JevBench public split** | Accuracy across choice, noul and score decisions | See [Results](#results). d1-2B completed all 231 cases with five GPU layers (0.797); its full-GPU run completed only 1 of 111 hard cases before device loss. |
| **[Calibration](glossary.md#g-ece)**: ECE and Brier | Whether "80%" means right 80% of the time | See [Results](#results). Temperature fitted on one half (split by paraphrase group), scored on the other. On 231 items this is noisy: it made Kev's ECE worse (0.038 to 0.082), which says the raw figure was already about as good as this sample can show. |
| **Permutation flip rate** | Whether the answer changes when the options are shuffled | One shuffle per item. **Jeva-4B: no answer changed, largest probability change 1.3 × 10⁻⁶,** invariant by construction as claimed. Qwen3.5-4B: 18 of 231 answers changed (8%), largest change 0.58. Gemma-4-E2B: 32 of 231 (14%), largest 1.0. d1-2B on the same two-thread CPU path before and after one shuffle: no answers changed on the 72 easy and standard choice cases; the maximum probability change was 0.104 (mean per-item maximum 0.0099). This d1 subset excludes hard and score cases and does not establish invariance. Accuracy barely moved in either logprob-read case, so label-logit reading is position-biased item by item rather than overall. |
| **Cold and cached latency** | Whether a second question about a state reuses its prefill | See [Prefill and the cached state](#prefill-and-the-cached-state). Measured at about 2.7K tokens only, not at 256 or 8K. |
| **Tool routing with an explicit none option** | Choose a named tool or decline when no tool is needed | d1-2B Q4_K_M on CPU: 24/24 existing custom prompts, including 4/4 no-tool cases, with no request errors. Each request listed available tools plus an explicit `none` choice. This checks selection only, not tool-call argument generation or a native null-option feature. Other decision models were not run on this adaptation. |
| **Agreement with Jev** | How close a local model gets to the hosted original | Not run, and its result would stay local. |

How the [benchmarking rules](benchmarking.md#the-rules-that-follow) applied:

- **One protocol, every model:** the same items and option-order seeds. Prompt formats differ by path, because Kev and Jeva take structured requests rather than SemIf's chat prompt.
- **Check the letter mass** for logprob readouts: it is how LFM2.5's invalid result showed up. Every other logprob-read model put more than 99.9% of its next-token probability on the letters.
- **Failures are not wrong answers.** Kev-4B failed to load on MPS. d1-2B rejected questions over the default physical batch, and its larger-batch run lost the device during the hard tier; failed requests are not scored as wrong answers.

## What's left

1. **Null-option cases for the other decision models**, recast from the 24-case [custom tool set](benchmarking.md#capability-benchmarks) with negatives included. d1 was tested with an explicit `none` option; native null-option handling remains untested.
2. **Jeva-4B on the Radeon**, without new llama.cpp code: merge its LoRA, convert the backbone to GGUF, serve it with `llama-server --embeddings --pooling last`, and run its head in a small `/v1/systemone` service. About a day, and it measures whether Jeva's 0.87 survives Q4. Native llama.cpp support is about 4–6 days more, and worth it only if Q4 holds up.
3. **Kev-0.8B as a GGUF** for the native path, converted with upstream's Kev support; no GGUF is published.
4. **Latency at 8K-token states**, on Qwen3.5-4B, Gemma-4-E2B and native Kev-4B.
5. **A different answer format for LFM2.5**, then a rerun.
6. **Kev as the router** for the [OpenCode agents](opencode-agents/README.md), through LocalAI: Kev-0.8B on the CPU beside a GPU worker, or Kev-4B on the GPU where only one model runs.