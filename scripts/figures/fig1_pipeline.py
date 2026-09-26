"""Figure 1 — TTAR pipeline diagram. Clean redesign."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_INPUT  = "#cde8f4"
C_BASE   = "#c8dfc8"
C_HEAT   = "#fde8c4"
C_ANCHOR = "#f5d0d0"
C_SOLVER = "#e4d4f4"
C_OUT    = "#c8dfc8"
C_EDGE   = "#555555"


def box(ax, cx, cy, w, h, title, sub="", color="#ddd", fs=8.5, sub_fs=6.5):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.05",
                       facecolor=color, edgecolor=C_EDGE, linewidth=0.9)
    ax.add_patch(r)
    if sub:
        ax.text(cx, cy + 0.12, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", clip_on=False)
        ax.text(cx, cy - 0.15, sub, ha="center", va="center",
                fontsize=sub_fs, color="#444", clip_on=False,
                multialignment="center")
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", clip_on=False)


def arr(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444",
                                lw=1.1, mutation_scale=11))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    FW, FH = 13.5, 4.2
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

    # row heights
    R1 = 2.90   # top row centre
    R2 = 1.20   # bottom row centre
    BH = 0.70   # box height

    # ── TOP ROW: RGB → DA-V2 → D₀ → heat-map → select K anchors → anchor source
    x_rgb   = 0.82;  BW_rgb  = 0.85
    x_da    = 2.28;  BW_da   = 1.10
    x_d0    = 3.62;  BW_d0   = 0.88
    x_heat  = 5.18;  BW_heat = 1.35
    x_asel  = 7.15;  BW_asel = 1.85   # wide enough for 2-line sub
    x_asrc  = 10.15; BW_asrc = 1.85

    box(ax, x_rgb,  R1, BW_rgb,  BH, "RGB",            color=C_INPUT)
    box(ax, x_da,   R1, BW_da,   BH, "DA-V2",          "frozen backbone",  color=C_BASE)
    box(ax, x_d0,   R1, BW_d0,   BH, r"$D_0$",         "baseline depth",   color=C_BASE)
    box(ax, x_heat, R1, BW_heat, BH, "heat-map $H$",
        r"$z(|\nabla\log D_0|)$",                                           color=C_HEAT)
    box(ax, x_asel, R1, BW_asel, BH, "select $K$ anchors",
        "heat-greedy,\nspatial suppression",                                color=C_HEAT)
    box(ax, x_asrc, R1, BW_asrc, BH, "anchor source",
        "query_depth(rgb, anchors)",                                        color=C_ANCHOR)

    # top-row arrows
    arr(ax, x_rgb  + BW_rgb/2,  R1, x_da   - BW_da/2,  R1)
    arr(ax, x_da   + BW_da/2,   R1, x_d0   - BW_d0/2,  R1)
    arr(ax, x_d0   + BW_d0/2,   R1, x_heat - BW_heat/2, R1)
    arr(ax, x_heat + BW_heat/2, R1, x_asel - BW_asel/2, R1)
    arr(ax, x_asel + BW_asel/2, R1, x_asrc - BW_asrc/2, R1)

    # red dashed outline around anchor source
    pw, ph = BW_asrc + 0.18, BH + 0.22
    plug = FancyBboxPatch((x_asrc - pw/2, R1 - ph/2),
                          pw, ph,
                          boxstyle="round,pad=0.08", facecolor="none",
                          edgecolor="#cc3333", linewidth=1.0, linestyle="--")
    ax.add_patch(plug)
    ax.text(x_asrc, R1 + ph/2 + 0.10, "pluggable backend",
            fontsize=7.2, color="#cc3333", ha="center", style="italic")

    # ── BOTTOM ROW: dense solver + D_refined ─────────────────────────────────
    x_solver  = 6.20;  BW_solver  = 1.70
    x_refined = 2.75;  BW_refined = 1.60

    box(ax, x_solver,  R2, BW_solver,  BH, "dense solver",
        "heat-modulated CG Laplacian",                        color=C_SOLVER)
    box(ax, x_refined, R2, BW_refined, BH, r"$D_\mathrm{refined}$",
        r"$D_0 \cdot \exp(\delta - \overline{\delta})$",      color=C_OUT)

    # D₀ → down to mid-gap, right to above solver, down-arrow into solver top
    # (avoids colliding with the bottom-row arrows)
    mid_y = (R1 - BH/2 + R2 + BH/2) / 2   # halfway between row edges ≈ 2.05
    ax.plot([x_d0, x_d0],         [R1 - BH/2, mid_y], color="#444", lw=1.1,
            solid_capstyle="round")
    ax.plot([x_d0, x_solver],     [mid_y, mid_y],      color="#444", lw=1.1,
            solid_capstyle="round")
    arr(ax,  x_solver, mid_y,  x_solver, R2 + BH/2)   # arrow down into solver top

    # anchor source → down (line) then left-arrow into solver right side
    ax.plot([x_asrc, x_asrc], [R1 - BH/2, R2], color="#444", lw=1.1,
            solid_capstyle="round")
    arr(ax, x_asrc, R2, x_solver + BW_solver/2, R2)

    # solver → refined  (left-arrow at bottom-row level, no conflicts)
    arr(ax, x_solver - BW_solver/2, R2, x_refined + BW_refined/2, R2)

    # ── Annotations below bottom row ─────────────────────────────────────────
    ax.text(x_solver, R2 - 0.55,
            r"$\min_\delta\;\sum_a w_a(\delta_a - \Delta_a)^2"
            r"+ \lambda\!\!\sum_{p,q}\!\!e^{-\alpha H}(\delta_p - \delta_q)^2$",
            fontsize=8.2, ha="center", va="top", color="#333")

    ax.text(x_refined, R2 - 0.55,
            "zero-mean $\\delta$ strips\nglobal log-shift\n(siRMSE-aware)",
            fontsize=7.2, ha="center", va="top", color="#555", style="italic")

    fig.savefig(out / "fig1_pipeline.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out / "fig1_pipeline.png", bbox_inches="tight", dpi=180)
    print(f"saved {out}/fig1_pipeline.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
