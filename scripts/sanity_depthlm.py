"""Quick sanity test for DepthLMBackend.

Loads the model in int8, queries 5 random anchors on one val image, prints
the parsed depths alongside D0 and GT for the same pixels.

A success looks like:
  - the model loads in <2 min (8-bit BnB)
  - each anchor returns a positive finite depth
  - depths are roughly in the right ballpark (within ~2-5x of D0 / GT)

Usage (login or compute node with a 16+ GB GPU):
  python scripts/sanity_depthlm.py \
      --preds-dir outputs/val_preds/vits_lora_r8/ \
      --train-root /cluster/courses/cil/monocular-depth-estimation/train \
      --stem train_007817
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.refine.anchor_backends import DepthLMBackend  # noqa: E402

EPS = 1e-6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--stem", required=True, help="Image stem, e.g. train_007817.")
    p.add_argument("--n-anchors", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-int8", action="store_true",
                   help="Disable int8 quantization (>=24 GB VRAM needed).")
    args = p.parse_args()

    train_root = Path(args.train_root)
    preds_dir = Path(args.preds_dir)
    rgb_path = train_root / f"{args.stem}_rgb.png"
    gt_path = train_root / f"{args.stem}_depth.npy"
    d0_path = preds_dir / f"{args.stem}.npy"
    for q in (rgb_path, gt_path, d0_path):
        if not q.exists():
            raise SystemExit(f"missing: {q}")

    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    gt = np.load(gt_path).astype(np.float32)
    D0 = np.load(d0_path).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    valid = (gt > 0) & (D0 > 0) & np.isfinite(gt) & np.isfinite(D0)
    # Ensure anchors are far enough from the border that the arrow fits.
    Hh, Ww = D0.shape
    margin = 8  # > marker_size=5
    border = np.zeros_like(valid)
    border[margin:Hh - margin, margin:Ww - margin] = True
    valid &= border
    valid_idx = np.flatnonzero(valid)
    sel = rng.choice(valid_idx, size=args.n_anchors, replace=False)
    anchors = [(int(i // Ww), int(i % Ww)) for i in sel]

    print(f"[sanity] image={args.stem}  shape={rgb.shape}  anchors={anchors}")

    # Dump a debug PNG of the image with all anchor arrows overlaid — so we
    # can verify the marker looks right before burning GPU time on the model.
    debug = rgb.copy()
    s = 5  # matches DepthLM marker_size
    for (y, x) in anchors:
        if not (s <= x < Ww - s and s <= y < Hh - s):
            continue
        col = np.array([255, 0, 0], dtype=np.uint8)
        for dx in range(1, s + 1):
            debug[y, x - dx] = col
        for dy in range(1, s // 2 + 1):
            debug[y + dy, x - dy - 1] = col
            debug[y - dy, x - dy - 1] = col
    debug_path = Path("outputs/diagnostics") / f"depthlm_marker_debug_{args.stem}.png"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(debug).save(debug_path)
    print(f"[sanity] wrote marker preview: {debug_path}")

    # Batch=1 keeps activation memory minimal — the int8 12B model already
    # uses ~15.4 GB on the 16 GB 5060 Ti; batching multiple images
    # simultaneously OOMs the activation buffer.
    backend = DepthLMBackend(load_in_8bit=not args.no_int8, batch_size=1)
    print(f"[sanity] backend ready: {backend.name}")

    log_depths, conf = backend.query_depth(rgb, anchors, stem=args.stem)
    print(f"\n  {'anchor':>15s}   D0 (m)   GT (m)   VLM (m)   log diff vs D0")
    for (y, x), ld in zip(anchors, log_depths):
        d0_m = float(D0[y, x])
        gt_m = float(gt[y, x])
        if not np.isfinite(ld):
            vlm_m, diff = float("nan"), float("nan")
        else:
            vlm_m = float(np.exp(ld))
            diff = ld - float(np.log(max(d0_m, EPS)))
        print(f"  ({y:>3d}, {x:>3d})    {d0_m:6.3f}   {gt_m:6.3f}   "
              f"{vlm_m:6.3f}   {diff:+.3f}")


if __name__ == "__main__":
    main()
