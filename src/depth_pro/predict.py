"""Depth Pro zero-shot predictions for val-set analysis and test-set submission.

Apple's Depth Pro is a metric monocular depth foundation model that produces
depth maps in metres at the original image resolution without camera intrinsics.
This script mirrors src/finetune/predict.py in interface so the same downstream
scripts (eval_preds.py, eval_refine.py, analyzer-app.py) accept its outputs.

Prerequisites:
    pip install --user git+https://github.com/apple/ml-depth-pro
    # Download the checkpoint once (from within the ml-depth-pro clone):
    #   source get_pretrained_models.sh
    # Then copy or symlink checkpoints/depth_pro.pt into this project's
    # checkpoints/ directory.

Val-set mode (error analysis / comparison with Stage A):
    python -m src.depth_pro.predict \\
        --checkpoint checkpoints/depth_pro.pt \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --val \\
        --out-dir outputs/val_preds/depth_pro/

Test-set mode (Kaggle submission):
    python -m src.depth_pro.predict \\
        --checkpoint checkpoints/depth_pro.pt \\
        --test-root /cluster/courses/cil/monocular-depth-estimation/test \\
        --out-dir submissions/depth_pro/preds/

    Then: python /cluster/courses/cil/monocular-depth-estimation/create_submission.py
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

from src.common.data import make_splits


def _load_model(checkpoint: Path, device: torch.device):
    config = dataclasses.replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(checkpoint.resolve()),
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
    )
    model.eval()
    return model, transform


def _infer_one(model, transform, rgb_path: Path) -> np.ndarray:
    image, _, f_px = depth_pro.load_rgb(rgb_path)
    image = transform(image)
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].squeeze().cpu().numpy().astype(np.float32)
    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the depth_pro.pt model weights.")
    parser.add_argument("--config", default=None,
                        help="YAML config path; required for --val to read the split parameters.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N images (for smoke tests).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose .npy output already exists (enables safe resume).")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-root",
                      help="Path to test images (Kaggle submission mode).")
    mode.add_argument("--val", action="store_true",
                      help="Run on the validation split defined by --config.")

    args = parser.parse_args()

    if args.val and args.config is None:
        parser.error("--val requires --config to determine the validation split.")

    cfg = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Depth Pro checkpoint not found: {checkpoint}")

    print(f"[predict] loading Depth Pro from {checkpoint}")
    model, transform = _load_model(checkpoint, device)

    if args.val:
        _, val_files = make_splits(
            cfg["data"]["root"],
            val_frac=cfg["data"]["val_frac"],
            seed=cfg["data"]["seed"],
        )
        files = val_files
        print(f"[predict] val split: {len(files)} images")
    else:
        files = sorted(Path(args.test_root).glob("*_rgb.png"))
        if not files:
            raise FileNotFoundError(f"No *_rgb.png under {args.test_root}")
        print(f"[predict] test set: {len(files)} images")

    if args.limit is not None:
        files = files[: args.limit]
        print(f"[predict] limited to {len(files)} images")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        before = len(files)
        files = [f for f in files
                 if not (out_dir / f.name.replace("_rgb.png", ".npy")).exists()]
        print(f"[predict] skip-existing: {before - len(files)} already done, {len(files)} remaining")

    n_written = 0
    for i, rgb_path in enumerate(files):
        depth = _infer_one(model, transform, rgb_path)
        stem = rgb_path.name.replace("_rgb.png", "")
        np.save(out_dir / f"{stem}.npy", depth)
        n_written += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"[predict] {i + 1}/{len(files)}")

    print(f"[predict] wrote {n_written} .npy files to {out_dir}")
    if not args.val:
        print(f"[predict] next: run create_submission.py pointing pred_dir at {out_dir}")


if __name__ == "__main__":
    main()
