#!/usr/bin/env python3
"""Grouped bar chart of every model across every benchmark, for models.md.

Mermaid's xychart-beta renders multiple bar series but has no legend, so a multi-model
grouped chart is not readable there. This draws it with matplotlib instead and writes a PNG.

Colours are grouped by maker (one hue family per vendor) so the chart matches the way the
tables are ordered. Missing bars are genuinely missing data, annotated on the chart.

Usage: 69-benchmark-chart.py <out.png>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# model -> (maker, colour), in the vendor order used by models.md
MODELS = [
    ("Qwen3-4B-2507",       "Alibaba",      "#1f4e79"),
    ("Qwen3.5-4B",          "Alibaba",      "#5b9bd5"),
    ("Gemma-4-E2B",         "Google",       "#a61c00"),
    ("Gemma-4-E4B",         "Google",       "#e8734a"),
    ("Granite-4.0-H-Micro", "IBM",          "#1e6b52"),
    ("Granite-4.2-3B",      "IBM",          "#6cc4a1"),
    ("LFM2.5-2.6B",         "Liquid AI",    "#6a4c93"),
    ("Phi-4-mini",          "Microsoft",    "#d98c00"),
    ("SmolLM3-3B",          "Hugging Face", "#8c8c8c"),
]

# benchmark -> {model: score}; None = not measured / not applicable
BENCH = {
    "MATH-500\n(n=100)": {
        "Qwen3-4B-2507": 0.75, "Qwen3.5-4B": 0.81, "Gemma-4-E2B": 0.29, "Gemma-4-E4B": 0.15,
        "Granite-4.0-H-Micro": 0.63, "Granite-4.2-3B": 0.62, "LFM2.5-2.6B": 0.68,
        "Phi-4-mini": 0.67, "SmolLM3-3B": 0.74},
    "IFBench strict\n(n=100)": {
        "Qwen3-4B-2507": 0.21, "Qwen3.5-4B": 0.26, "Gemma-4-E2B": 0.24, "Gemma-4-E4B": 0.24,
        "Granite-4.0-H-Micro": 0.19, "Granite-4.2-3B": 0.19, "LFM2.5-2.6B": 0.23,
        "Phi-4-mini": 0.10, "SmolLM3-3B": 0.08},
    "MMLU-Pro\n(n=28)": {
        "Qwen3-4B-2507": 0.43, "Qwen3.5-4B": 0.46, "Gemma-4-E2B": 0.43, "Gemma-4-E4B": 0.54,
        "Granite-4.0-H-Micro": 0.39, "Granite-4.2-3B": 0.29, "LFM2.5-2.6B": 0.14,
        "Phi-4-mini": 0.39, "SmolLM3-3B": 0.29},
    "GPQA-Diamond\n(n=100)": {
        "Qwen3-4B-2507": 0.37, "Qwen3.5-4B": 0.56, "Gemma-4-E2B": 0.44, "Gemma-4-E4B": 0.49,
        "Granite-4.0-H-Micro": 0.26, "Granite-4.2-3B": 0.22, "LFM2.5-2.6B": 0.22,
        "Phi-4-mini": 0.30, "SmolLM3-3B": 0.29},
    "BFCL AST\n(n=80)": {
        "Qwen3-4B-2507": 0.88, "Qwen3.5-4B": 0.75, "Gemma-4-E2B": 0.84, "Gemma-4-E4B": 0.85,
        "Granite-4.0-H-Micro": 0.86, "Granite-4.2-3B": 0.65, "LFM2.5-2.6B": 0.69,
        "Phi-4-mini": None, "SmolLM3-3B": None},
    "MRCR\n(n=2, ~19K tok)": {
        "Qwen3-4B-2507": 0.468, "Qwen3.5-4B": 0.0, "Gemma-4-E2B": 0.113, "Gemma-4-E4B": 1.0,
        "Granite-4.0-H-Micro": 0.014, "Granite-4.2-3B": 0.0, "LFM2.5-2.6B": 0.0,
        "Phi-4-mini": 0.103, "SmolLM3-3B": 0.034},
}

def main(out):
    benches = list(BENCH)
    n_models = len(MODELS)
    x = np.arange(len(benches))
    width = 0.092

    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=110)
    for i, (name, maker, colour) in enumerate(MODELS):
        offs = (i - (n_models - 1) / 2) * width
        vals = [BENCH[b].get(name) for b in benches]
        heights = [0 if v is None else v for v in vals]
        ax.bar(x + offs, heights, width, label=f"{name}  ({maker})",
               color=colour, edgecolor="white", linewidth=0.4)
        for xi, v in zip(x + offs, vals):
            if v is None:
                ax.text(xi, 0.015, "n/a", ha="center", va="bottom",
                        fontsize=6.5, color="#666", rotation=90)
            elif v == 0:                          # a measured zero, not missing data
                ax.text(xi, 0.015, "0", ha="center", va="bottom", fontsize=6.2, color="#333")
            elif v >= 0.40:                       # label only the taller bars, to stay readable
                ax.text(xi, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center", va="bottom",
                        fontsize=6.2, color="#333")

    ax.set_xticks(x); ax.set_xticklabels(benches, fontsize=10)
    ax.set_ylabel("score (0–1, higher is better)", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("Every model on every benchmark — Radeon Pro 5500M, 4 GB, Q4_K_M",
                 fontsize=13, pad=12)
    ax.legend(ncol=3, fontsize=8.5, loc="upper left", framealpha=0.95,
              title="model (maker)", title_fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005,
             "Chance floors: MMLU-Pro 0.10, GPQA-Diamond 0.25. "
             "n/a = the model emits no parseable tool_calls on this stack. "
             "Small samples: treat gaps under ~0.1 as ties.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks.png")
