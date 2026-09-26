"""Figure 4 — Qualitative panel: RGB / D₀ / error map / TTAR-refined (oracle).

Runs the TTAR dense solver with GT oracle anchors on one example image.
No GPU required — uses precomputed D₀ (.npy) and the pure-numpy solver.

Usage:
    python scripts/figures/fig4_qualitative.py \\
        --stem train_021903 \\
        --preds-dir outputs/val_preds/vits_lora_r8/ \\
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \\
        --out-dir outputs/figures/ \\
        --K 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# make project imports work
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.common.data import load_rgb                                             # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse                             # noqa: E402
from src.refine.heatmap import compute_heatmap, select_anchors_heatmap          # noqa: E402
from src.refine.correct_dense import solve_dense_correction                      # noqa: E402

EPS = 1e-6


def depth_colorize(d: np.ndarray, mask: np.ndarray,
                   vmin: float | None = None, vmax: float | None = None):
    """Map depth to [0,1] with shared scale, masking invalid pixels."""
    vis = d.copy().astype(np.float32)
    vis[~mask] = np.nan
    vmin = vmin or float(np.nanpercentile(vis, 2))
    vmax = vmax or float(np.nanpercentile(vis, 98))
    return vis, vmin, vmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem",       default="train_013302")
    parser.add_argument("--preds-dir",  default="outputs/val_preds/vits_lora_r8/")
    parser.add_argument("--train-root", default="/cluster/courses/cil/monocular-depth-estimation/train")
    parser.add_argument("--out-dir",    default="outputs/figures/")
    parser.add_argument("--K",          type=int, default=3000)
    parser.add_argument("--rand-anchors", action="store_true", default=True,
                        help="Use random anchor selection (matching README §7.1 oracle config)")
    parser.add_argument("--alpha-smooth", type=float, default=0.0,
                        help="Heat-map modulation strength (0=uniform, matching §7.1 oracle)")
    args = parser.parse_args()

    out_dir    = Path(args.out_dir)
    train_root = Path(args.train_root)
    preds_dir  = Path(args.preds_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    stem = args.stem
    pred = np.load(preds_dir / f"{stem}.npy").astype(np.float32)
    gt   = np.load(train_root / f"{stem}_depth.npy").astype(np.float32)
    rgb  = load_rgb(train_root / f"{stem}_rgb.png")
    valid = (gt > 0) & (pred > 0) & np.isfinite(pred)

    s0 = si_rmse(pred, gt)
    print(f"stem: {stem}   baseline siRMSE: {s0:.4f}")

    # ── TTAR with GT oracle anchors ───────────────────────────────────────────
    # Random anchors over valid pixels (matches README §7.1 oracle K=3000 config)
    rng = np.random.default_rng(42)
    if args.rand_anchors:
        H_w, W_w = pred.shape
        valid_flat = np.where(valid.ravel())[0]
        idx = rng.choice(valid_flat, min(args.K, len(valid_flat)), replace=False)
        anchors = [(int(i // W_w), int(i % W_w)) for i in idx]
        # For display: also compute the heat-map (not used for selection)
        H = compute_heatmap(pred, alpha=1.0, beta=1.0, gamma=0.0)
    else:
        H = compute_heatmap(pred, alpha=1.0, beta=1.0, gamma=0.0)
        anchors = select_anchors_heatmap(H, K=args.K, min_dist_px=24)

    # GT anchor targets: log-shift = log(gt) - log(d0)
    anchor_pys = [a[0] for a in anchors]
    anchor_pxs = [a[1] for a in anchors]
    gt_vals = gt[anchor_pys, anchor_pxs]
    d0_vals = pred[anchor_pys, anchor_pxs]
    valid_a = (gt_vals > 0) & (d0_vals > 0)
    delta_targets = np.zeros(len(anchors), np.float32)
    delta_targets[valid_a] = (
        np.log(np.maximum(gt_vals[valid_a], EPS))
        - np.log(np.maximum(d0_vals[valid_a], EPS))
    )
    confidences = np.ones(len(anchors), np.float32)
    confidences[~valid_a] = 0.0

    # Uniform smoothness (alpha_smooth=0) matching oracle eval config
    H_solver = np.zeros_like(pred) if args.alpha_smooth == 0.0 else H
    pred_refined, info = solve_dense_correction(
        D0=pred,
        H=H_solver,
        anchors_yx=anchors,
        anchor_dlog=delta_targets,
        anchor_weights=confidences,
        lam_s=0.1,
        alpha_smooth=args.alpha_smooth,
        cg_max_iter=500,
        cg_tol=1e-5,
    )
    print(f"solver: n_anchors={info['n_anchors']} cg_info={info['cg_info']}")

    s_ref = si_rmse(pred_refined, gt)
    print(f"refined siRMSE: {s_ref:.4f}  (Δ = {s_ref - s0:+.4f})")

    # ── figure ────────────────────────────────────────────────────────────────
    # Shared depth scale over valid GT + pred for panels 2-4
    combined = np.concatenate([gt[valid], pred[valid], pred_refined[valid]])
    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 98))

    # Error map clipped at ±1.5 log units
    err_d0  = np.full(pred.shape, np.nan, np.float32)
    err_ref = np.full(pred.shape, np.nan, np.float32)
    err_d0[valid]  = np.log(np.maximum(pred[valid], EPS)) - np.log(np.maximum(gt[valid], EPS))
    err_ref[valid] = np.log(np.maximum(pred_refined[valid], EPS)) - np.log(np.maximum(gt[valid], EPS))
    vlim = 1.5

    # 1×4 layout — wider figure, colorbars below each panel to avoid crowding
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    fig.subplots_adjust(wspace=0.28)

    def add_colorbar(ax, im, label=""):
        """Colorbar placed below the axes so it doesn't steal horizontal space."""
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.046, pad=0.04, aspect=25)
        cb.ax.tick_params(labelsize=7.5)
        if label:
            cb.set_label(label, fontsize=8)

    ax0, ax1, ax2, ax3 = axes

    # Panel 1: RGB
    ax0.imshow(rgb)
    ax0.set_title("RGB input", fontsize=10, fontweight="bold", pad=6)
    ax0.axis("off")

    # Panel 2: D₀ depth
    im1 = ax1.imshow(pred, cmap="plasma", vmin=vmin, vmax=vmax)
    ax1.set_title(f"$D_0$ (baseline)\nsiRMSE = {s0:.3f}", fontsize=10,
                  fontweight="bold", pad=6)
    ax1.axis("off")
    add_colorbar(ax1, im1, "depth (m)")

    # Panel 3: D₀ log-error map
    im2 = ax2.imshow(err_d0, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax2.set_title("$D_0$ log-error\nlog(pred) − log(GT)", fontsize=10,
                  fontweight="bold", pad=6)
    ax2.axis("off")
    add_colorbar(ax2, im2, "log ratio  (blue=too near, red=too far)")

    # Panel 4: refined log-error map
    im3 = ax3.imshow(err_ref, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax3.set_title(f"Refined log-error\n"
                  f"siRMSE {s0:.3f} → {s_ref:.3f}  (Δ = {s_ref-s0:+.3f})",
                  fontsize=10, fontweight="bold", pad=6)
    ax3.axis("off")
    add_colorbar(ax3, im3, "log ratio  (blue=too near, red=too far)")

    anchor_config = f"rand+unif K={args.K}" if args.rand_anchors else f"heat+heat K={args.K}"
    fig.suptitle(
        f"Qualitative example — {stem}    GT oracle anchors ({anchor_config})",
        fontsize=9.5, y=1.02)

    fig.savefig(out_dir / f"fig4_qualitative_{stem}.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_dir / f"fig4_qualitative_{stem}.png", bbox_inches="tight", dpi=180)
    print(f"saved {out_dir}/fig4_qualitative_{stem}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
