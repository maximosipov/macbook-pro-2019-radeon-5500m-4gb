#!/usr/bin/env python3
"""Emit the capability heatmap in models.md as a Mermaid `block-beta` grid.

Mermaid has no heatmap diagram, but block-beta lays out a fixed-column grid and accepts a
per-block `style` with a fill colour, which is enough to build one natively — no image, and
the numbers stay selectable text in the document.

Colour is normalised PER COLUMN (palest = worst on that benchmark, deepest = best), because
the benchmarks have very different ranges here: IFBench tops out at 0.26 while BFCL starts at
0.65, so one scale across the whole grid would render most cells uniformly pale.

Usage: score-heatmap.py            # prints the mermaid block to stdout
"""
MAKER = {
    "Qwen3-4B-2507": "Alibaba", "Qwen3.5-4B": "Alibaba",
    "Gemma-4-E2B": "Google", "Gemma-4-E4B": "Google",
    "Granite-4.0-H-Micro": "IBM", "Granite-4.2-3B": "IBM",
    "LFM2.5-2.6B": "Liquid AI", "Phi-4-mini": "Microsoft", "SmolLM3-3B": "Hugging Face",
}
# model -> [MATH-500, IFBench strict, MMLU-Pro, GPQA-D, BFCL AST, MRCR]; None = not measurable
MODELS = [
    ("Qwen3-4B-2507",       [0.75, 0.21, 0.43, 0.37, 0.88, 0.468]),
    ("Qwen3.5-4B",          [0.81, 0.26, 0.46, 0.56, 0.75, 0.000]),
    ("Gemma-4-E2B",         [0.29, 0.24, 0.43, 0.44, 0.84, 0.113]),
    ("Gemma-4-E4B",         [0.15, 0.24, 0.54, 0.49, 0.85, 1.000]),
    ("Granite-4.0-H-Micro", [0.63, 0.19, 0.39, 0.26, 0.86, 0.014]),
    ("Granite-4.2-3B",      [0.62, 0.19, 0.29, 0.22, 0.65, 0.000]),
    ("LFM2.5-2.6B",         [0.68, 0.23, 0.14, 0.22, 0.69, 0.000]),
    ("Phi-4-mini",          [0.67, 0.10, 0.39, 0.30, None,  0.103]),
    ("SmolLM3-3B",          [0.74, 0.08, 0.29, 0.29, None,  0.034]),
]
COLS = ["MATH-500", "IFBench", "MMLU-Pro", "GPQA-D", "BFCL", "MRCR"]
RAMP = ["#eff3ff", "#c6dbef", "#9ecae1", "#4292c6", "#08519c"]   # pale -> deep

def main():
    rows = ["block-beta", "    columns 7",
            '    h0["model"] ' + " ".join(f'h{j+1}["{c}"]' for j, c in enumerate(COLS))]
    styles = [f"    style h{j} fill:#ffffff,stroke:#ffffff,color:#333333,font-weight:bold"
              for j in range(7)]
    for i, (name, vals) in enumerate(MODELS):
        cells = [f'm{i}["{name} · {MAKER[name]}"]']
        for j, v in enumerate(vals):
            cells.append(f'c{i}_{j}["{"n/a" if v is None else f"{v:.2f}"}"]')
        rows.append("    " + " ".join(cells))
        styles.append(f"    style m{i} fill:#ffffff,stroke:#ffffff,color:#333333")
    for j in range(len(COLS)):
        col = [(i, m[1][j]) for i, m in enumerate(MODELS) if m[1][j] is not None]
        lo, hi = min(v for _, v in col), max(v for _, v in col)
        for i, v in col:
            k = 0 if hi == lo else int(round((v - lo) / (hi - lo) * (len(RAMP) - 1)))
            fg = "#ffffff" if k >= 3 else "#1a1a1a"
            styles.append(f"    style c{i}_{j} fill:{RAMP[k]},stroke:#ffffff,color:{fg}")
        for i, m in enumerate(MODELS):
            if m[1][j] is None:
                styles.append(f"    style c{i}_{j} fill:#f2f2f2,stroke:#ffffff,color:#999999")
    print("\n".join(rows + styles))

if __name__ == "__main__":
    main()
