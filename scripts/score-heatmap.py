#!/usr/bin/env python3
"""White-to-blue heatmap of the capability table, for models.md.

GitHub strips inline CSS from Markdown, so a real coloured table cell is not possible there;
this renders the same numbers as an image instead. The Markdown table stays the authoritative,
copyable version.

Colour is normalised PER COLUMN (palest = worst on that benchmark, deepest = best), because the
benchmarks have very different ceilings here — IFBench tops out at 0.26 while BFCL starts at 0.65,
so a single 0-1 scale would render most of the grid uniformly pale. The printed numbers are the
raw scores.

Rows are ordered by maker, matching the tables in models.md.

Usage: score-heatmap.py <out.png>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [  # (model, maker) in vendor order
    ("Qwen3-4B-2507", "Alibaba"), ("Qwen3.5-4B", "Alibaba"),
    ("Gemma-4-E2B", "Google"), ("Gemma-4-E4B", "Google"),
    ("Granite-4.0-H-Micro", "IBM"), ("Granite-4.2-3B", "IBM"),
    ("LFM2.5-2.6B", "Liquid AI"), ("Phi-4-mini", "Microsoft"), ("SmolLM3-3B", "Hugging Face"),
]
COLS = ["MATH-500", "IFBench\nstrict", "MMLU-Pro", "GPQA-D", "BFCL AST", "MRCR"]
SCORES = {  # model -> row of raw scores, None = not measurable on this stack
    "Qwen3-4B-2507":       [0.75, 0.21, 0.43, 0.37, 0.88, 0.468],
    "Qwen3.5-4B":          [0.81, 0.26, 0.46, 0.56, 0.75, 0.000],
    "Gemma-4-E2B":         [0.29, 0.24, 0.43, 0.44, 0.84, 0.113],
    "Gemma-4-E4B":         [0.15, 0.24, 0.54, 0.49, 0.85, 1.000],
    "Granite-4.0-H-Micro": [0.63, 0.19, 0.39, 0.26, 0.86, 0.014],
    "Granite-4.2-3B":      [0.62, 0.19, 0.29, 0.22, 0.65, 0.000],
    "LFM2.5-2.6B":         [0.68, 0.23, 0.14, 0.22, 0.69, 0.000],
    "Phi-4-mini":          [0.67, 0.10, 0.39, 0.30, None, 0.103],
    "SmolLM3-3B":          [0.74, 0.08, 0.29, 0.29, None, 0.034],
}

def main(out):
    raw = np.array([[np.nan if v is None else v for v in SCORES[m]] for m, _ in MODELS])
    # per-column min-max normalisation, ignoring missing cells
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm[:, j] = 0.5 if hi == lo else (col - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=120)
    ax.imshow(np.ma.masked_invalid(norm), cmap="Blues", vmin=0, vmax=1, aspect="auto")

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f2f2f2",
                                           edgecolor="white", linewidth=2, zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9,
                        color="#999", zorder=3)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5, zorder=3,
                        color="white" if norm[i, j] > 0.6 else "#1a1a1a",
                        fontweight="bold" if norm[i, j] == 1.0 else "normal")

    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=10)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([f"{m}  ·  {mk}" for m, mk in MODELS], fontsize=10)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # separator lines between makers
    prev = None
    for i, (_, mk) in enumerate(MODELS):
        if prev is not None and mk != prev:
            ax.axhline(i - .5, color="#4a4a4a", linewidth=1.4, zorder=4)
        prev = mk

    ax.set_title("Capability scores — colour is rank within each benchmark, numbers are raw scores",
                 fontsize=12, pad=34)
    fig.text(0.5, 0.012,
             "Deepest blue = best on that benchmark, palest = worst. Columns are scaled "
             "independently because their ranges differ (IFBench tops out at 0.26, BFCL starts at 0.65).\n"
             "Chance floors: MMLU-Pro 0.10, GPQA-Diamond 0.25. n/a = emits no parseable tool_calls "
             "on this stack. Horizontal rules separate makers.",
             ha="center", fontsize=8.2, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "score-heatmap.png")
