"""Local prediction on Mac (Apple Silicon / CPU).

Runs the same inference as src/finetune/predict.py, but:
  - auto-picks MPS > CUDA > CPU
  - lets you limit to the first N images
  - num_workers=0 to play nice with macOS

Run from the project root (so `src` is importable):
    python predict_local.py \
        --config configs/finetune/da_v2_vits_lora.yaml \
        --checkpoint best.pt \
        --test-root /Users/halperk/Documents/eth-spring-26/CIL/segmentation/train \
        --out-dir outputs/local_preds \
        --limit 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow MPS to fall back to CPU for any op that isn't implemented yet.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.common.data import DepthDataset
from src.common.utils import load_ckpt
from src.finetune.model import LoRAConfig, build_model


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=10,
                        help="Predict on only the first N images. Use 0 for all.")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    device = pick_device()
    print(f"[predict] device: {device}")

    # Rebuild the same architecture used during training.
    lora_dict = cfg["model"].get("lora")
    lora_cfg = LoRAConfig(**lora_dict) if lora_dict else None
    model = build_model(
        encoder=cfg["model"]["encoder"],
        checkpoint=None,
        lora_cfg=lora_cfg,
    ).to(device)
    load_ckpt(args.checkpoint, model, map_location=device, strict=True)
    model.eval()

    test_files = sorted(Path(args.test_root).glob("*_rgb.png"))
    if not test_files:
        raise FileNotFoundError(f"No *_rgb.png under {args.test_root}")
    if args.limit and args.limit > 0:
        test_files = test_files[: args.limit]
    print(f"[predict] running on {len(test_files)} images")

    ds = DepthDataset(test_files, has_depth=False, augment=None)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,        # safer on macOS
        pin_memory=False,     # pin_memory + MPS don't mix
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred = model(images)                 # [B, 1, H, W]
            pred = pred.squeeze(1).cpu().numpy() # [B, H, W]
            for arr, name in zip(pred, batch["name"]):
                stem = name.replace("_rgb.png", "")
                out_path = out_dir / f"{stem}.npy"
                np.save(out_path, arr.astype(np.float32))
                print(f"[predict] wrote {out_path.name}")

    print(f"[predict] done — {len(test_files)} .npy files in {out_dir}")


if __name__ == "__main__":
    main()