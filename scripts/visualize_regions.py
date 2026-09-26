"""Per-image diagnostic for the SAM region-smoothing refiner.

For each image stem, writes a 3x3 PNG:

  Row 1: RGB, RGB + SAM region boundaries, blank
  Row 2: D0, refined depth, GT depth
  Row 3: |log D0 - log GT|, |log D_ref - log GT|, per-pixel delta
         (red = refiner hurt, blue = refiner helped)

Usage:
    python scripts/visualize_regions.py \\
        --preds-dir outputs/val_preds/no_gf/ \\
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \\
        --stems train_001414 train_001221 train_000767 train_002556 \\
        --alpha 0.4 \\
        --out-dir outputs/regressor_viz/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import load_rgb  # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402
from src.refine.region_smooth import SAMRegionSmoothRefiner, _load_regions  # noqa: E402

EPS = 1e-6


def hash_rgb(rgb: np.ndarray) -> str:
    return hashlib.blake2b(rgb.tobytes(), digest_size=12).hexdigest()


def make_boundary_overlay(rgb: np.ndarray, regions) -> np.ndarray:
    out = rgb.copy()
    for region in regions:
        edge = region.mask & ~binary_erosion(region.mask, iterations=2)
        out[edge] = [255, 255, 0]
    return out


def visualize(
    stem: str,
    refiner: SAMRegionSmoothRefiner,
    preds_dir: Path,
    train_root: Path,
    out_dir: Path,
    alpha: float,
) -> None:
    rgb = load_rgb(train_root / f"{stem}_rgb.png")
    D0 = np.load(preds_dir / f"{stem}.npy").astype(np.float32)
    gt = np.load(train_root / f"{stem}_depth.npy").astype(np.float32)

    cache_path = refiner.cache_root / f"{hash_rgb(rgb)}.npz"
    if not cache_path.exists():
        print(f"[viz] no cached regions for {stem} at {cache_path}, skipping")
        return
    regions = _load_regions(cache_path)
    d_ref = refiner(rgb, D0)

    s0 = si_rmse(D0, gt)
    s1 = si_rmse(d_ref, gt)

    log_gt = np.log(np.maximum(gt, EPS))
    err_d0 = np.where(gt > 0, np.abs(np.log(np.maximum(D0, EPS)) - log_gt), 0)
    err_dref = np.where(gt > 0, np.abs(np.log(np.maximum(d_ref, EPS)) - log_gt), 0)
    delta = np.where(gt > 0, err_dref - err_d0, 0)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(make_boundary_overlay(rgb, regions))
    axes[0, 1].set_title(f"SAM regions ({len(regions)})")
    axes[0, 2].axis("off")

    vmin = float(min(D0.min(), gt[gt > 0].min(), d_ref.min())) if (gt > 0).any() else 0.0
    vmax = float(max(D0.max(), gt.max(), d_ref.max()))
    axes[1, 0].imshow(D0, vmin=vmin, vmax=vmax, cmap="inferno")
    axes[1, 0].set_title(f"D0  siRMSE={s0:.4f}")
    axes[1, 1].imshow(d_ref, vmin=vmin, vmax=vmax, cmap="inferno")
    axes[1, 1].set_title(f"D_refined (α={alpha})  siRMSE={s1:.4f}  Δ{s1 - s0:+.4f}")
    axes[1, 2].imshow(gt, vmin=vmin, vmax=vmax, cmap="inferno")
    axes[1, 2].set_title("GT")

    err_vmax = float(max(err_d0.max(), err_dref.max(), 1e-3))
    axes[2, 0].imshow(err_d0, vmin=0, vmax=err_vmax, cmap="hot")
    axes[2, 0].set_title("|log D0 − log GT|")
    axes[2, 1].imshow(err_dref, vmin=0, vmax=err_vmax, cmap="hot")
    axes[2, 1].set_title("|log D_ref − log GT|")

    dmax = float(max(abs(delta.min()), abs(delta.max()), 1e-3))
    axes[2, 2].imshow(delta, vmin=-dmax, vmax=dmax, cmap="RdBu_r")
    axes[2, 2].set_title("Δerror  (red = hurt, blue = helped)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{stem}    baseline {s0:.4f} → refined {s1:.4f}    Δ {s1 - s0:+.4f}",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = out_dir / f"{stem}_alpha{alpha}_{refiner.aggregator}.png"
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--aggregator", default="plane", choices=["median", "plane"])
    parser.add_argument("--cache-dir", default="outputs/stage_b_cache/sam_regions")
    parser.add_argument("--out-dir", default="outputs/regressor_viz/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # No sam_kwargs means same config_hash as the sweep, so the cached masks
    # are picked up automatically.
    refiner = SAMRegionSmoothRefiner(
        alpha=args.alpha,
        aggregator=args.aggregator,
        cache_dir=args.cache_dir,
    )

    for stem in args.stems:
        visualize(
            stem,
            refiner,
            Path(args.preds_dir),
            Path(args.train_root),
            out_dir,
            args.alpha,
        )


if __name__ == "__main__":
    main()
