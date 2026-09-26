"""Figure 2 — Error mass vs distance to depth discontinuity. Cleaned up."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BIN_LABELS  = ["[0,2)", "[2,4)", "[4,8)", "[8,16)", "[16,∞)"]
EDGE_ORDER  = ["gt", "d0", "rgb"]
EDGE_NAMES  = {"gt": "GT edges  (3.8× conc.)",
               "d0": r"$D_0$ edges  (3.0× conc.)",
               "rgb": "RGB edges  (1.0×)"}
EDGE_COLORS = {"gt": "#2a7a3b", "d0": "#2255aa", "rgb": "#888888"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/diagnostics/boundary_phase0/results.json")
    parser.add_argument("--out-dir", default="outputs/figures/")
    args = parser.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        d = json.load(f)
    agg  = d["aggregate"]
    full = agg["full"]
    gross_pct = agg["gross_frac"]["mean"] * 100
    mod_pct   = 100.0 - gross_pct
    n_images  = agg["n_images"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ── LEFT: grouped bar chart ───────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(BIN_LABELS))
    w = 0.26

    for k, et in enumerate(EDGE_ORDER):
        mf = np.array(full["mass_frac"][et]["e2c"]) * 100
        ax.bar(x + (k-1)*w, mf, width=w,
               color=EDGE_COLORS[et], edgecolor="#555", linewidth=0.55,
               label=EDGE_NAMES[et], alpha=0.88)
        # value on the [0,2) bar only
        ax.text(x[0] + (k-1)*w, mf[0] + 1.2,
                f"{mf[0]:.0f}%", ha="center", fontsize=8.0,
                color=EDGE_COLORS[et], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=9)
    ax.set_xlabel("Distance to nearest depth discontinuity (px)", fontsize=9)
    ax.set_ylabel("Moderate error² mass (%)", fontsize=9)
    ax.set_title(f"Moderate error concentration by boundary type\n"
                 f"(N={n_images} val images; |log err| ≤ 1.0)",
                 fontsize=9.5, fontweight="bold")
    ax.tick_params(labelsize=8.5)
    ax.set_ylim(0, 92)
    ax.legend(fontsize=8.5, framealpha=0.9)

    # ── RIGHT: gross vs moderate pie ─────────────────────────────────────────
    ax2 = axes[1]
    wedges, texts = ax2.pie(
        [gross_pct, mod_pct],
        labels=[f"Gross\n(|log err| > 1.0)\n{gross_pct:.1f}%",
                f"Moderate\n(|log err| ≤ 1.0)\n{mod_pct:.1f}%"],
        colors=["#c7391b", "#5588cc"],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2.0),
        textprops=dict(fontsize=8.5),
        radius=0.92,
    )
    ax2.set_title("Error² mass decomposition\n(all val images, all valid pixels)",
                  fontsize=9.5, fontweight="bold")
    ax2.text(0, -1.30,
             "Gross: 71% unfixable within ±8 px\n"
             "Moderate: 80% within ±2 px of GT discontinuity",
             ha="center", fontsize=8.0, color="#444",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#ccc", lw=0.7))

    fig.tight_layout(pad=1.2)
    fig.savefig(out / "fig2_boundary_hist.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out / "fig2_boundary_hist.png", bbox_inches="tight", dpi=180)
    print(f"saved {out}/fig2_boundary_hist.{{pdf,png}}")
    plt.close(fig)

if __name__ == "__main__":
    main()
