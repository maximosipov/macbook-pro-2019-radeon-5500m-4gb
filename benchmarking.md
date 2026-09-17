# Benchmarking this machine

How to get numbers from this laptop that mean something, and how every measurement in this repository was produced.

- [Why this machine lies to you](#why-this-machine-lies-to-you)
- [Correctness instruments](#correctness-instruments)
- [Speed and fit instruments](#speed-and-fit-instruments)
- [Capability benchmarks](#capability-benchmarks)
- [Long-context benchmarks](#long-context-benchmarks)
- [Superseded results](#superseded-results)

## Why this machine lies to you

This is a 2019 laptop, and it will mislead you if you let it. The same binary, model and flags produced **10.5, 24.9 and 33.9 tok/s** across three sessions. None of those runs was faulty; they were different machine states. Three effects, in order of size:

**1. GPU clock state — dominant, about 2.4×.** [`ioreg`](glossary.md#g-ioreg) reports the Radeon idling at **10 MHz**, and `llama-bench`'s single warm-up pass doesn't spin it up. Back to back, with the same binary and flags:

| GPU | CPU | tg128 |
|---|---|---|
| Warm (straight off a serving workload) | **throttled to 30** | **24.92 ± 8.10** |
| Cold (after cooling to 100) | unthrottled | 10.49 ± 0.11 |

The run on the *throttled* CPU is 2.4× faster. Cooling the machine to avoid CPU throttling parks the GPU, so the intuitive benchmarking hygiene is backwards here.

**2. Position within a multi-configuration run.** `llama-bench -fa 0,1` gives the first configuration a different machine than the second. That alone produced an apparent 3.9× flash-attention penalty that a [counterbalanced](glossary.md#g-counterbalance) experiment put at 2.27×.

**3. [CPU thermal throttling](glossary.md#g-throttle).** `CPU_Speed_Limit` falls to 20–36 within about two minutes of load. Real, but smaller than the GPU effect, and it points the other way.

### The rules that follow

- **Warm the GPU** with a discarded run before measuring.
- **Use one process per configuration.** Never compare two configurations from one multi-config invocation.
- **Alternate the order** of what you're comparing, and report both rounds.
- **Report the per-round values,** not just the mean. The spread is what tells you whether a difference is real.
- **Never difference two tables produced on different days.** Answer configuration questions with an A/B run.
- **For "what will I actually get", measure sustained serving,** not micro-benchmarks. LocalAI serving Qwen3-4B-2507 returned 27.1 / 27.9 / 28.4 tok/s across three runs — tighter than any `llama-bench` figure on this machine.

Two caveats, both learned the hard way. A warm-up pass **is not free**: adding one triggered [`vk::DeviceLostError`](glossary.md#g-device-lost) on Qwen3.5-4B, so a more representative number also means a less likely finish. And watch for **orphaned benchmark processes**: one outlived a `pkill` of its parent script and held the machine at `CPU_Speed_Limit=36` while the next run sat in a cooldown loop, waiting for a machine that another process was busy heating.

## Correctness instruments

### `test-backend-ops`

Runs every ggml operation on a backend with random inputs and compares the result against the CPU implementation, per shape and per data type, against a tolerance (5e-4 for matmul). `test` mode reports OK or FAIL per case; `perf` mode reports microseconds per case.

**Score:** a count of passing cases. A FAIL is a real bug, since the comparison is against the same computation on the CPU.

Two reading rules, both learned here. A case reported `not supported` is **skipped, not passed** — and a backend where every case is skipped still prints a green `OK`, which is how the flash-attention CPU fallback hid for months. And the case totals move with upstream, so compare failures against a same-day baseline rather than a count quoted in a document.

This is the only instrument here that localizes a bug to a specific shader. Everything else just tells you something is wrong. Usage: [ggml.md](ggml.md#run).

### Text correctness battery

Four deterministic prompts (arithmetic, factual recall, sequence continuation and string reversal) at temperature 0, scored on the final answer. It also computes a **4-gram repetition ratio** over the output, which is what actually catches [MoltenVK garbage](glossary.md#g-nan): broken GPU output is usually fluent-looking loops rather than wrong answers.

**Score:** *n*/4 correct, with the repetition ratio as a separate pass/fail gate. It measures *coherence, not capability*: 4/4 means the model is running correctly, not that it's any good.

One artifact to know: the exact-match factual scorer keys off the `[Start thinking]` or `<think>` preamble that `llama-cli` leaves in the output, so reasoning models can report `OVERALL: FAIL` while the repetition gate still passes. Coherence is real there; only the matcher is fooled.

### Vision evaluation

Four images with known ground truth (a sign with readable text, a counting scene, a colour and shape question, and a chart) sent as base64 over the chat API, scored on the final answer after any reasoning block is stripped.

**Score:** *n*/4. Confirms the [`mmproj`](glossary.md#g-vlm) path works end to end, not just that the LLM loads.

### Diffusion CPU-reference diff

The same prompt, seed, sampler, step count and resolution generated on the CPU and on the GPU, then compared.

**Score:** pass/fail by eye against the reference. "Matches CPU" in the diffusion tables means this comparison passed, which is a much stronger claim than "produced an image" — the [noise failure](README.md#ki-diffusion-noise) also produces an image.

## Speed and fit instruments

### `llama-bench` (pp512 / tg128)

The standard llama.cpp microbenchmark: `pp512` times [prefill](glossary.md#g-prefill) of a 512-token prompt, `tg128` times generation of 128 tokens, both in tokens per second, repeated with `-r 3` and reported as mean ± [standard deviation](glossary.md#g-sd).

**Score:** tokens per second, higher is better. The ± is what tells you whether two numbers actually differ.

Run one configuration per process, with the machine cooled before each model. Treat the absolute numbers as cold-GPU lower bounds, and cross-configuration comparisons as suspect unless they came from a [counterbalanced A/B](glossary.md#g-counterbalance).

### Sustained serving throughput

A 128-token completion requested over the [OpenAI endpoint](glossary.md#g-openai-api) against an already-warm server, timed end to end from the client, repeated three times.

**Score:** tokens per second including server-side queueing and detokenization. In principle that's slightly pessimistic against `llama-bench`; in practice it's higher, because the GPU is warm. This is the number a user actually experiences, and on this machine it's by far the most reproducible (spread under 5%).

### VRAM residency and fit probes

> **The loader accounting is hidden by default on current builds.** `llama-cli` and `llama-server` print a short banner instead of the per-buffer allocation log unless you pass `-v`. Any fit check that greps for a CPU buffer line will therefore see nothing and report a pass at every context size. Pass `-v`, or measure [`ioreg`](glossary.md#g-ioreg) residency instead.

Three complementary checks:

- the llama.cpp or sd.cpp **load log**, which reports per-buffer allocation and reveals [CPU spill](glossary.md#g-spill) as a CPU buffer appearing;
- **[`ioreg`](glossary.md#g-ioreg)** `inUseVidMemoryBytes` on the AMD accelerator node during generation, which is the only cross-runtime ground truth and the one that works when a runtime's own reporting is wrong;
- a **context sweep** that raises `--ctx-size` until spill appears, which is what produced the [context ceilings](models.md#context-ceilings).

**Score:** megabytes resident, and yes or no on spill. There's no partial credit: a model either fits or quietly halves its speed.

### Diffusion seconds per step and peak VRAM

sd.cpp's own per-step timing from its log, plus peak VRAM sampled during [VAE decoding](glossary.md#g-vae) rather than at load, because that's where a 4 GB run dies.

**Score:** seconds per sampling step, and peak megabytes. Wall-clock time is reported separately, since model load and on-the-fly quantization dominate a 4-step turbo generation.

### Pipette on-device performance

Liquid AI's own client ([pipette-clients](https://github.com/Liquid4All/pipette-clients)) run against its 34 standard local definitions: `prefill_throughput`, `decode_throughput`, `end_to_end_latency` and `max_memory_usage` at context depths from 100 to 8192, each with a discarded warm-up, a readiness gate and five measured repetitions.

The catch on this machine is provenance, not method. The client installs the **stock** upstream llama.cpp release for the platform, and the only x86 macOS build has **no Vulkan** — so every Pipette number here measures the CPU floor, not the Vulkan path the rest of this repository documents. That makes it a clean floor-versus-ceiling contrast, and nothing more.

Results are **staged locally and not published**. A submission publishes device and contact details, so it's gated on an explicit decision to do so.

## Capability benchmarks

All were run over `llama-server`'s OpenAI endpoint at temperature 0, on Q4_K_M weights, with small samples chosen to fit this machine's patience. **They rank these models against each other under one protocol; they are not comparable to published leaderboard numbers.** Results: [models.md](models.md#capability-scores).

**Thinking is disabled** for the comparison, for two reasons. Several of these models default to emitting a `<think>` block, which (a) makes every generation many times longer, which is untenable across a set of models on a thermally limited card, and (b) *breaks* IFBench, whose verifiers grade the raw response — a reasoning preamble violates "respond with exactly N bullets". Models are served with `--reasoning-format deepseek`, which moves any think block into a separate `reasoning_content` field and leaves the graded `content` clean, plus `--chat-template-kwargs '{"enable_thinking":false}'` where the template accepts it.

One convention applies throughout: a request that exceeds the **400 s client timeout** is recorded as `timeout`, meaning no answer was produced and nothing was graded. A timeout and a wrong answer are different failures and are never averaged together.

| Benchmark | What it measures | Scoring |
|---|---|---|
| **[BFCL-V4](glossary.md#g-bfcl) AST** (n=80) | A user request plus function schemas; the model is asked for a tool call through the OpenAI tools API, and the emitted call is compared structurally against the reference (function name, argument names, types and values). No execution. | Fraction matching, 0–1. A wrong name, a missing required argument or malformed JSON all score 0. Single-turn categories only, so it says nothing about multi-turn behaviour. |
| **[MMLU-Pro](glossary.md#g-mmlu)** (n=28, 0-shot CoT) | Ten-option multiple choice across academic and professional subjects. The model reasons step by step and finishes with `The answer is (X)`. | Fraction correct, **chance floor 0.1**. A model that rambles past its token budget without emitting the answer line scores 0 regardless of its reasoning. At n=28 the confidence interval is wide: treat gaps under 0.1 as a tie. |
| **[MATH-500](glossary.md#g-math500)** (n=100) | Competition mathematics, answered step by step with the final answer in `\boxed{}`. | Fraction correct by **symbolic equivalence** (`math_verify`), so `1/2`, `0.5` and `\frac{1}{2}` all match. No chance floor. |
| **[IFBench](glossary.md#g-ifbench)** (n=100) | Prompts carrying programmatically checkable constraints ("exactly three bullet points"), graded by the dataset's own 83 verifier functions rather than an LLM judge. | **Prompt-level strict accuracy**: the fraction of prompts where *every* constraint is met. Loose tolerates markdown and a stray boundary line. Deliberately hard. |
| **[GPQA Diamond](glossary.md#g-gpqa)** (n=100) | Graduate-level "Google-proof" science, four options, 0-shot CoT. Options are shuffled with a fixed per-item seed so the correct letter isn't predictable. | Fraction correct, **chance floor 0.25**. The letter is extracted from either `content` or `reasoning_content`; the *unparsed rate* is tracked separately, and a high one means the score understates the model. Gated dataset: needs an accepted licence and `HF_TOKEN`. |
| **Custom tool set** (n=24) | Hand-written cases against local function schemas, graded on the API's *parsed* `tool_calls` rather than raw text, so chat-template differences don't skew it. | Cases passed out of 24. Includes **negative cases**, where the right behaviour is to answer directly and call nothing — which is where small models usually fail, and which a benchmark that only counts successful calls would reward them for getting wrong. |
| **[LongBench-v2](glossary.md#g-longbench)** (n=3, ~12K tokens) | Four-choice questions over long documents. Only the smallest samples were used, so this is a reduced-context variant chosen so the card could prefill it at all. | Fraction correct, chance floor 0.25. |
| **[MRCR](glossary.md#g-mrcr) v2** (n=2, ~19K tokens) | A long synthetic conversation contains several near-identical requests; the model must reproduce one specific earlier reply, with a required prefix. | Mean sequence-similarity ratio against the reference, 0–1, gated on the prefix being present. Not accuracy: partial credit is possible and a fluent wrong answer scores near 0. |

### Long-context benchmarks

LongBench-v2 and MRCR are run at ctx 32768 with the KV precision each model needs to fit, and **time to
response is recorded per item** — on this card it ranges from 114 s to over 20 minutes, so it is a result
in its own right rather than overhead.

Three things had to be fixed before these benchmarks measured anything. Every earlier long-context number
on this page is affected by at least one of them:

- **The 400-second client timeout was shorter than the work.** A 19K-token MRCR item takes 200–1300 s here.
  One item that the old limit recorded as `timeout` in fact scores **0.896** when allowed to finish. The
  limits are now 1800 s (LongBench) and 2400 s (MRCR).
- **The answer budget was 24 tokens.** A reasoning model spends that inside its think block and returns
  empty content, scoring 0 by construction. Raised to 1024.
- **Only `content` was graded.** Models served with `--reasoning-format deepseek` put their answer in
  `reasoning_content` while `content` stays empty — the same root cause as the GPQA unparsed-rate caveat.
  Both benchmarks now read either channel, and LongBench reports an explicit `unparsed` count so this
  failure can never again be mistaken for a wrong answer.

The lesson generalizes beyond this project: **a benchmark harness that reports a timeout or a zero is
making a claim about itself as much as about the model.** Check the unparsed rate and the wall-clock
distribution before believing either.

### CLIP score (prompt adherence)

Cosine similarity between CLIP's embedding of the prompt and of the generated image (`openai/clip-vit-base-patch32`, reported as 100 x cosine). Reference-free, so it needs no ground-truth image — it measures whether the image matches what was asked for, not whether it is attractive.

**Score:** roughly 20-40 in practice; higher is better. Judge it per prompt rather than by the mean: across six prompts the three image models here land within 0.4 points of each other on average while differing by up to 10 points on individual prompts. It does detect a missing subject — SDXL-Turbo omitting the horse from "astronaut riding a horse" cost it 10 points against SD-Turbo. Results: [stable-diffusion.cpp.md](stable-diffusion.cpp.md#image-quality).

Earlier editions of this page listed CLIP score as "not run" on the grounds that the visual gap between the candidates was decisive without it. That turned out to be wrong in an interesting way: scored, the models are tied on adherence, and the assumed gap does not exist.

## Superseded results

Findings that were true at some point in this project and are not true now. They're recorded because each one is easy to rediscover from an old note or an upstream issue, and re-deriving them costs hours.

- **Every old flash-attention measurement is void.** The "2.3× penalty", the counterbalanced A/B runs, and the claim that the effect reverses for state-space models were all taken while `FLASH_ATTN_EXT` was unsupported on Vulkan and attention was running on the **CPU**. With it on the GPU, flash attention is the faster option for generation.
- **q8_0 was the problem quantization on this driver, and it is now root-caused.** Two failures that looked independent — the multi-column `mul_mat_vecq` path and a q8_0 KV cache under flash attention — are the same defect, and both paths now route q8_0 elsewhere. The general lesson: **a performance patch can convert a correct-but-slow path into a fast-and-wrong one**, and only a per-operation differential test catches it.
- **`int dot: 0` and `warp size` in the capability banner don't tell you what a build is doing.** Both are printed from the driver's raw report, before the backend applies its own decisions. See [hardware.md](hardware.md#reading-the-capability-line).
- **[llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104)** (Intel-Mac AMD Vulkan gibberish) **does not reproduce** on a current checkout: the correctness battery passes, and GPU [perplexity](glossary.md#g-ppl) matches the CPU within 0.05%. The gibberish people hit on this hardware today is a [wrong device](README.md#ki-wrong-device) or the [subgroup misreport](README.md#ki-subgroup-size).
- **Metal was measured, not assumed.** It's broken differently for LLMs (PCIe re-reads, about 0.8 tok/s) than for image generation (watchdog timeout, no output at all).
- **The Intel GPU was evaluated once and permanently excluded.** It was slower than the CPU at every size tested, and numerically wrong under MoltenVK. The defensive fix on these branches exists to stop it being *silently* wrong, not to make it usable.
- **Gemma-4-E4B no longer triggers device loss.** It once ran out of VRAM at `pp512` with flash attention off, and now completes in both configurations. The remaining reproducible trigger is hours of back-to-back GPU work with no gap; about 40 minutes of idle restored normal behaviour.
