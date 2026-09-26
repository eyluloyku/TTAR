"""TTAR-refined Kaggle test submission.

Applies the dense correction solver to Stage A test predictions using a second
model (e.g. Depth Pro) as the anchor source, then writes a submission CSV.

Usage:
    python scripts/ttar_test_submission.py \
        --d0-dir      submissions/stage_a_test/preds/ \
        --anchor-dir  submissions/depth_pro/preds/ \
        --out-dir     submissions/ttar_dp_k3000/preds/ \
        --K 3000

The script also writes the submission CSV automatically at:
    submissions/ttar_dp_k3000/submission.csv
"""
from __future__ import annotations

import argparse
import base64
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.refine.correct_dense import solve_dense_correction    # noqa: E402
from src.refine.heatmap import compute_heatmap, select_anchors_heatmap  # noqa: E402

EPS = 1e-6


def process_one(
    stem: str,
    d0_dir: Path,
    anchor_dir: Path,
    out_dir: Path,
    K: int,
    rand_anchors: bool,
    alpha_smooth: float,
    lam_s: float,
    rng: np.random.Generator,
) -> dict | None:
    d0_path     = d0_dir     / f"{stem}.npy"
    anchor_path = anchor_dir / f"{stem}.npy"

    if not d0_path.exists():
        print(f"  skip {stem}: D0 not found")
        return None
    if not anchor_path.exists():
        print(f"  skip {stem}: anchor pred not found")
        return None

    d0          = np.load(d0_path).astype(np.float32)
    anchor_pred = np.load(anchor_path).astype(np.float32)

    valid = (d0 > 0) & (anchor_pred > 0) & np.isfinite(d0) & np.isfinite(anchor_pred)

    if rand_anchors:
        H_w, W_w    = d0.shape
        valid_flat  = np.where(valid.ravel())[0]
        idx         = rng.choice(valid_flat, min(K, len(valid_flat)), replace=False)
        anchors     = [(int(i // W_w), int(i % W_w)) for i in idx]
        H           = np.zeros_like(d0)        # uniform smoothness
    else:
        H       = compute_heatmap(d0, alpha=1.0, beta=1.0, gamma=0.0)
        anchors = select_anchors_heatmap(H, K=K, min_dist_px=4)

    anchor_pys = [a[0] for a in anchors]
    anchor_pxs = [a[1] for a in anchors]

    d0_vals     = d0[anchor_pys, anchor_pxs]
    anchor_vals = anchor_pred[anchor_pys, anchor_pxs]
    valid_a     = (anchor_vals > 0) & (d0_vals > 0)

    delta_targets              = np.zeros(len(anchors), np.float32)
    delta_targets[valid_a]     = (
        np.log(np.maximum(anchor_vals[valid_a], EPS))
        - np.log(np.maximum(d0_vals[valid_a], EPS))
    )
    confidences                = np.ones(len(anchors), np.float32)
    confidences[~valid_a]      = 0.0

    H_solver = np.zeros_like(d0) if alpha_smooth == 0.0 else H

    d_refined, info = solve_dense_correction(
        D0=d0,
        H=H_solver,
        anchors_yx=anchors,
        anchor_dlog=delta_targets,
        anchor_weights=confidences,
        lam_s=lam_s,
        alpha_smooth=alpha_smooth,
        cg_max_iter=500,
        cg_tol=1e-5,
    )

    np.save(out_dir / f"{stem}.npy", d_refined.astype(np.float32))
    return info


def create_submission_csv(pred_dir: Path, out_csv: Path) -> None:
    rows = []
    for pred_path in sorted(pred_dir.glob("test_*.npy")):
        depth       = np.load(pred_path)
        depth_f16   = np.asarray(depth, dtype=np.float16)
        compressed  = zlib.compress(depth_f16.tobytes(), level=9)
        encoded     = base64.b64encode(compressed).decode("utf-8")
        idx         = pred_path.stem.split("_")[-1]
        rows.append({"id": f"test_{idx}_depth", "Depths": encoded})
    df = pd.DataFrame(rows, columns=["id", "Depths"])
    df.to_csv(out_csv, index=False)
    print(f"submission CSV saved: {out_csv}  ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-dir",       required=True,
                        help="Stage A test predictions (one .npy per image)")
    parser.add_argument("--anchor-dir",   required=True,
                        help="Second-model test predictions used as anchors")
    parser.add_argument("--out-dir",      required=True,
                        help="Output directory (refined .npy + submission.csv)")
    parser.add_argument("--K",            type=int, default=3000)
    parser.add_argument("--rand-anchors", action="store_true",
                        help="Random anchor selection (matches oracle config)")
    parser.add_argument("--alpha-smooth", type=float, default=0.0,
                        help="0 = uniform smoothness (default; safest with imperfect anchors)")
    parser.add_argument("--lam-s",        type=float, default=0.1)
    args = parser.parse_args()

    d0_dir     = Path(args.d0_dir)
    anchor_dir = Path(args.anchor_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_str = f"{'rand' if args.rand_anchors else 'heat'}_{'unif' if args.alpha_smooth == 0 else 'heat'}"
    print(f"TTAR test submission: K={args.K}  config={config_str}  lam_s={args.lam_s}")
    print(f"D0:     {d0_dir}")
    print(f"anchor: {anchor_dir}")
    print(f"out:    {out_dir}")

    stems = sorted(p.stem for p in d0_dir.glob("test_*.npy"))
    if not stems:
        print("ERROR: no test_*.npy files found in --d0-dir")
        sys.exit(1)
    print(f"found {len(stems)} test images")

    rng = np.random.default_rng(42)

    for i, stem in enumerate(stems):
        process_one(stem, d0_dir, anchor_dir, out_dir,
                    args.K, args.rand_anchors, args.alpha_smooth, args.lam_s, rng)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(stems)}]")

    print("TTAR done. creating submission CSV...")
    create_submission_csv(out_dir, out_dir.parent / "submission.csv")


if __name__ == "__main__":
    main()
