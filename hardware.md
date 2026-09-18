# The GPU: capabilities and limits

What ggml reports about the Radeon Pro 5500M, what each field changes, and where this card sits among AMD's generations.

- [Reading the capability line](#reading-the-capability-line)
- [RDNA generations](#rdna-generations)

## Reading the capability line

Every ggml tool prints this banner for each device it finds:

```
uma: 0 · fp16: 1 · bf16: 0 · fp4: 0 · int dot: 0 · matrix cores: none · warp size: 32
```

A `0` means one of two things: the silicon lacks the feature, or Metal cannot express what the silicon has. Only the second is reclaimable in software.

Validated against the hardware and MoltenVK's own report (device `0x7340`, Navi 14 / gfx1012, PCIe x16, 4 GB dedicated):

| Field | What the driver reports | Verdict |
|---|---|---|
| `uma: 0` | PCIe x16, 4 GB dedicated VRAM | **Silicon.** Genuinely not unified memory. |
| `fp16: 1` | `shaderFloat16 yes` | **Silicon, present and used.** |
| `bf16: 0` | `VK_KHR_shader_bfloat16` absent | **Silicon.** Navi 14 predates AMD's bf16 support, so bf16 weights produce [NaN](glossary.md#g-nan). |
| `int dot: 0` | Extension **present**, every `*Accelerated` flag **false** | **Software ceiling, reclaimed — but the line still reads 0.** Metal can't ask for the instruction, but emulating it is still worth **+68% prefill**. |
| `matrix cores: none` | `VK_KHR_cooperative_matrix` absent | **Silicon.** No tensor-core-class instructions, so there's no fast kernel for `--diffusion-fa`. |
| `warp size: 32` | 1.4.2 reports `subgroupSize 32`; **1.4.1 and earlier said 64** | **Driver bug, fixed.** Never a ceiling: MoltenVK was misreporting a 32-wide SIMD group as 64, which broke every subgroup shader. |

The banner is the driver's raw report, not the backend's decisions: a build with the integer-dot fix active still prints `int dot: 0`, and only prefill throughput shows it is live (about 59 pp512 on a 4B Q4_K_M against about 35). `warp size` does move, and is the check for MoltenVK 1.4.2 or later.

`VK_AMD_shader_core_properties` is absent, so ggml's AMD architecture detection falls through to `OTHER` on macOS and skips every RDNA-specific tuning path — the reason for the `GGML_VK_FORCE_ARCH` override on the [llama.cpp fork](llama.cpp.md#reference).

## RDNA generations

Three rows above are marked **Silicon**, meaning no fix exists for this chip; a later generation supplies two of them outright.

> This section comes from vendor documentation, not from measurements on this machine.

AMD splits its GPUs into **RDNA** (graphics, this card) and **CDNA** (datacentre compute). Matrix and bf16 features arrive in the two lines at different times, which is why bf16 is described as arriving with either CDNA1 or RDNA3.

| Generation | Shipped | Representative parts | `fp16` | `bf16` | `int dot` | `matrix cores` |
|---|---|---|---|---|---|---|
| **RDNA 1** (`gfx101x`) | 2019 | RX 5700 XT, **Radeon Pro 5500M / 5600M** (Navi 14/12) | Yes, 2× packed | No | Navi 12/14 only | No |
| **RDNA 2** (`gfx103x`) | 2020 | RX 6000, Radeon Pro W6800X, PS5, Steam Deck | Yes | No | Yes | No |
| **RDNA 3** (`gfx110x`) | 2022 | RX 7900 XTX, Radeon Pro W7900 | Yes | Yes | Yes | Yes (WMMA) |
| **RDNA 3.5** (`gfx115x`) | 2024 | Radeon 890M (Strix Point APUs) | Yes | Yes | Yes | Yes (WMMA) |
| **RDNA 4** (`gfx120x`) | 2025 | RX 9070 XT, RX 9060 XT | Yes | Yes | Yes, 2× rate | Yes (WMMA + FP8) |

- **RDNA 1** is the clean break from GCN, with wave32 as the native compute mode rather than GCN's wave64. The architecture is dual-mode, which is exactly the ambiguity MoltenVK got wrong. No matrix units and no bf16. Integer dot product splits the generation: Navi 12 and Navi 14 have it, Navi 10 (RX 5700) doesn't — so an RX 5700 is *less* capable here than this laptop chip.
- **RDNA 2** adds 8-bit and 4-bit integer dot across the whole line, properly reported as accelerated by a native driver. It's also **the last AMD architecture macOS ever ran**: the Radeon Pro W6800X modules for the 2019 Mac Pro are RDNA 2. Everything below this line is Linux or Windows only.
- **RDNA 3** erases most of this repository's "not fixable" column. Matrix instructions land `VK_KHR_cooperative_matrix`, so ggml's cooperative-matrix kernels light up and [flash attention](glossary.md#g-flash-attn) finally has a fast path, and bf16 arithmetic arrives, so bf16 GGUFs stop producing NaN.
- **RDNA 3.5** is an integrated-graphics refresh with the same feature set. Note that these report `uma: 1`, putting them in the opposite regime from this card: no PCIe copies, but no dedicated VRAM either.
- **RDNA 4** roughly doubles matrix throughput per compute unit against RDNA 3 and adds FP8 to the matrix path. FP4 and FP6 remain datacentre features, which is why the `fp4` field reads `0` on everything in this table.

Two of this repository's [known issues](README.md#known-issues) are architectural rather than driver bugs, and RDNA 3 answers both: matrix cores and bf16. The third, 4 GB, is answered by any larger card.

None of that is reachable on a Mac: macOS tops out at RDNA 2, and MoltenVK still translates to Metal, which has no DP4a-style instruction and no cooperative-matrix extension regardless of the silicon. The ceiling here is the API as much as the chip, which is why the fixes are keyed to the MoltenVK driver rather than to the GPU.
