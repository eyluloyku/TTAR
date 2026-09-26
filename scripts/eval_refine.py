"""Score a Stage B refiner against a directory of Stage A predictions.

Reports per-image siRMSE for D0 (baseline) and D_refined, plus the delta
distribution and win/loss rate. Compares pixel-aligned files between
``--preds-dir`` (Stage A outputs from ``src.finetune.predict --val``) and the
matching RGB+GT in the train root.

Usage:
    python scripts/eval_refine.py \
        --preds-dir outputs/val_preds/vits_lora_r8/ \
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \
        --refiner identity

The identity refiner is a sanity check: mean delta must be 0.0 exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import load_rgb  # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402
from src.refine.pipeline import build_refiner  # noqa: E402


def _parse_kwargs(items: list[str]) -> dict:
    """Parse a list of 'key=value' strings into a dict, auto-typing values."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--refiner-kwargs entry must be key=value, got '{item}'")
        k, v = item.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", required=True,
                        help="Directory of Stage A <stem>.npy depth predictions on val.")
    parser.add_argument("--train-root", required=True,
                        help="Train data root containing <stem>_rgb.png and <stem>_depth.npy.")
    parser.add_argument("--refiner", default="identity",
                        help="Refiner name registered in src.refine.pipeline.REFINERS.")
    parser.add_argument("--refiner-kwargs", nargs="*", default=[],
                        help="Extra refiner kwargs as key=value pairs (e.g. alpha=0.4). "
                             "Values are auto-parsed as int/float/bool/str.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of images (for quick smoke tests).")
    parser.add_argument("--save-refined", default=None,
                        help="Optional output dir for refined .npy files.")
    args = parser.parse_args()

    refiner_kwargs = _parse_kwargs(args.refiner_kwargs)

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    pred_files = sorted(preds_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files in {preds_dir}")
    if args.limit is not None:
        pred_files = pred_files[: args.limit]

    refiner = build_refiner(args.refiner, **refiner_kwargs)
    save_dir = Path(args.save_refined) if args.save_refined else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = 0
    for pred_path in pred_files:
        stem = pred_path.stem
        rgb_path = train_root / f"{stem}_rgb.png"
        gt_path = train_root / f"{stem}_depth.npy"
        if not rgb_path.exists() or not gt_path.exists():
            missing += 1
            continue

        D0 = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        rgb = load_rgb(rgb_path)

        D_ref = refiner(rgb, D0)
        if D_ref.shape != D0.shape:
            raise ValueError(f"{refiner.name} returned shape {D_ref.shape} != {D0.shape}")

        s0 = si_rmse(D0, gt)
        s1 = si_rmse(D_ref, gt)
        if np.isnan(s0) or np.isnan(s1):
            continue
        rows.append((stem, s0, s1))

        if save_dir is not None:
            np.save(save_dir / f"{stem}.npy", D_ref.astype(np.float32))

    if not rows:
        print("No matching pred/GT/RGB triples found.")
        return

    s0 = np.array([r[1] for r in rows])
    s1 = np.array([r[2] for r in rows])
    delta = s1 - s0  # negative = refiner improved siRMSE

    kwargs_str = "  " + ", ".join(f"{k}={v}" for k, v in refiner_kwargs.items()) if refiner_kwargs else ""
    print(f"refiner: {refiner.name}{kwargs_str}")
    if hasattr(refiner, "cache_hits") and hasattr(refiner, "cache_misses"):
        print(f"cache: {refiner.cache_hits} hits, {refiner.cache_misses} misses")
    if hasattr(refiner, "regions_smoothed") and hasattr(refiner, "regions_skipped"):
        total = refiner.regions_smoothed + refiner.regions_skipped
        if total > 0:
            print(f"regions: {refiner.regions_smoothed} smoothed, {refiner.regions_skipped} skipped "
                  f"({100 * refiner.regions_skipped / total:.1f}% gated)")
    print(f"N={len(rows)}  (skipped {missing} missing files)")
    print(f"baseline   siRMSE  mean={s0.mean():.4f}  median={np.median(s0):.4f}")
    print(f"refined    siRMSE  mean={s1.mean():.4f}  median={np.median(s1):.4f}")
    print(f"delta             mean={delta.mean():+.4f}  median={np.median(delta):+.4f}")
    print(f"   p10={np.percentile(delta,10):+.4f}  p90={np.percentile(delta,90):+.4f}")
    print(f"   improved: {(delta < -1e-6).sum()}  unchanged: {(np.abs(delta)<=1e-6).sum()}  worsened: {(delta > 1e-6).sum()}")

    worst = sorted(rows, key=lambda r: r[2] - r[1])[-5:]
    best = sorted(rows, key=lambda r: r[2] - r[1])[:5]
    print("biggest improvements (delta):")
    for n, a, b in best:
        print(f"  {n}: {a:.4f} -> {b:.4f}  ({b-a:+.4f})")
    print("biggest regressions (delta):")
    for n, a, b in worst:
        print(f"  {n}: {a:.4f} -> {b:.4f}  ({b-a:+.4f})")


if __name__ == "__main__":
    main()
