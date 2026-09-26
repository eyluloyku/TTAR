"""Figure 3 — DepthLM robustification staircase. Fixed label overlap + clipping."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROWS = [
    ("A0 identity",        0.0000,  0.0000,  "#bbbbbb"),
    ("A1 raw VLM",        +0.2131, +0.1752,  "#c7391b"),
    ("A2 +scale align",   +0.0849, +0.0589,  "#e07a3b"),
    ("A3 +MAD screen",    +0.0083, +0.0035,  "#c9a010"),
    ("A4 +agree gate",    +0.0061, +0.0017,  "#8fb04a"),
    ("A5 GT oracle",      -0.0021, -0.0029,  "#2a7a3b"),
]
LABELS    = [r[0] for r in ROWS]
FULL_MEAN = [r[1] for r in ROWS]
TAIL10    = [r[2] for r in ROWS]
COLORS    = [r[3] for r in ROWS]
WIN_RATE  = [0, 0, 13, 29, 22, 142]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    x = np.arange(len(ROWS))
    w = 0.38

    # ── LEFT: full-mean + tail-10 bars ───────────────────────────────────────
    ax = axes[0]
    # y limit: leave 18% headroom above max value for labels
    ymax = max(FULL_MEAN) * 1.22
    ymin = min(min(FULL_MEAN), min(TAIL10)) - 0.008

    bars_full = ax.bar(x - w/2, FULL_MEAN, width=w, color=COLORS,
                       edgecolor="#555", linewidth=0.7, label="full-mean Δ")
    bars_tail = ax.bar(x + w/2, TAIL10,    width=w, color=COLORS,
                       edgecolor="#555", linewidth=0.7, alpha=0.55,
                       label="tail-10 Δ", hatch="///")

    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5, rotation=30, ha="right")
    ax.set_ylabel("Δ-siRMSE  (neg = improvement)", fontsize=9)
    ax.set_title("DepthLM robustification: Δ-siRMSE", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    # value labels on full-mean bars — inside bar if large, above if small
    for bar, val in zip(bars_full, FULL_MEAN):
        if val == 0.0:
            continue
        bar_top = bar.get_height()
        if abs(val) > 0.03:
            # put label inside the bar
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar_top * 0.55,
                    f"{val:+.4f}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", rotation=90)
        else:
            offset = 0.003 if val >= 0 else -0.006
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                    f"{val:+.4f}", ha="center", va=va, fontsize=7.5)

    # 35× reduction bracket
    ax.annotate("", xy=(4 - w/2, FULL_MEAN[4] + 0.003),
                xytext=(1 - w/2, FULL_MEAN[1] * 0.98),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.0))
    ax.text(2.5, 0.17,
            "35× reduction\n(+0.2131 → +0.0061)",
            ha="center", fontsize=7.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaa", lw=0.7))

    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")

    # ── RIGHT: win-rate ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(x, WIN_RATE, color=COLORS, edgecolor="#555", linewidth=0.7)
    ax2.axhline(100, color="#888", lw=0.8, ls=":", label="100 of 200")
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS, fontsize=8.5, rotation=30, ha="right")
    ax2.set_ylabel("Images improved  (of 200)", fontsize=9)
    ax2.set_title("Win-rate (N=200)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 165)
    ax2.tick_params(axis="y", labelsize=8)
    for xi, val in enumerate(WIN_RATE):
        if val == 0:
            continue
        ax2.text(xi, val + 4, str(val), ha="center", fontsize=8.5)

    fig.tight_layout(pad=1.2)
    fig.savefig(out / "fig3_vlm_staircase.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out / "fig3_vlm_staircase.png", bbox_inches="tight", dpi=180)
    print(f"saved {out}/fig3_vlm_staircase.{{pdf,png}}")
    plt.close(fig)

if __name__ == "__main__":
    main()
