"""Compute siRMSE on a folder of saved .npy predictions vs. ground truth.

Usage:
    python scripts/eval_preds.py \
        --preds-dir outputs/val_preds/with_gf/ \
        --gt-dir /cluster/courses/cil/monocular-depth-estimation/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", required=True, help="Folder of <stem>.npy predictions.")
    parser.add_argument("--gt-dir", required=True, help="Folder containing *_depth.npy ground truth.")
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    gt_dir = Path(args.gt_dir)

    pred_files = sorted(preds_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files in {preds_dir}")

    scores = []
    missing = 0
    for pred_path in pred_files:
        gt_path = gt_dir / f"{pred_path.stem}_depth.npy"
        if not gt_path.exists():
            missing += 1
            continue
        pred = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        s = si_rmse(pred, gt)
        if not np.isnan(s):
            scores.append((pred_path.name, s))

    if not scores:
        print("No matching prediction/GT pairs found.")
        return

    vals = np.array([s for _, s in scores])
    print(f"N={len(scores)}  (skipped {missing} missing GT)")
    print(f"mean={vals.mean():.4f}  median={np.median(vals):.4f}")
    print(f"p10={np.percentile(vals, 10):.4f}  p90={np.percentile(vals, 90):.4f}")
    print(f"worst 5: " + ", ".join(f"{n}:{v:.3f}" for n, v in sorted(scores, key=lambda x: -x[1])[:5]))
    print(f"best  5: " + ", ".join(f"{n}:{v:.3f}" for n, v in sorted(scores, key=lambda x: x[1])[:5]))


if __name__ == "__main__":
    main()
