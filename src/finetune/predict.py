"""Generate depth predictions for Kaggle submission or val-set analysis.

Writes one ``<stem>.npy`` float array per image into an output directory.

Test-set mode (for Kaggle submission):
    python -m src.finetune.predict \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt \\
        --test-root /cluster/courses/cil/monocular-depth-estimation/test \\
        --out-dir submissions/vits_lora_r8/preds/

    Then: python /cluster/courses/cil/monocular-depth-estimation/create_submission.py

Val-set mode (for error analysis with the analyzer app):
    python -m src.finetune.predict \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt \\
        --val \\
        --out-dir outputs/val_preds/vits_lora_r8/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.common.data import DepthDataset, make_splits
from src.common.utils import load_ckpt
from src.finetune.model import LoRAConfig, build_model
from src.refine.guided_filter import guided_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--guided-filter", action="store_true",
                        help="Apply guided image filter post-processing to sharpen depth boundaries.")
    parser.add_argument("--gf-radius", type=int, default=8,
                        help="Guided filter window radius (default 8).")
    parser.add_argument("--gf-eps", type=float, default=1e-4,
                        help="Guided filter regularization (default 1e-4).")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-root", help="Path to test images (Kaggle submission mode).")
    mode.add_argument("--val", action="store_true", help="Run on the val split (analysis mode).")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    lora_dict = cfg["model"].get("lora")
    lora_cfg = LoRAConfig(**lora_dict) if lora_dict else None
    model = build_model(
        encoder=cfg["model"]["encoder"],
        checkpoint=None,
        lora_cfg=lora_cfg,
    ).to(device)
    load_ckpt(args.checkpoint, model, map_location=device, strict=True)
    model.eval()

    if args.val:
        _, val_files = make_splits(
            cfg["data"]["root"],
            val_frac=cfg["data"]["val_frac"],
            seed=cfg["data"]["seed"],
        )
        files = val_files
        has_depth = False
        print(f"[predict] val split: {len(files)} images")
    else:
        files = sorted(Path(args.test_root).glob("*_rgb.png"))
        if not files:
            raise FileNotFoundError(f"No *_rgb.png under {args.test_root}")
        has_depth = False
        print(f"[predict] test set: {len(files)} images")

    ds = DepthDataset(files, has_depth=has_depth, augment=None)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.guided_filter:
        print(f"[predict] guided filter on  (radius={args.gf_radius}, eps={args.gf_eps})")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            pred = model(images)                  # [B, 1, H, W]
            pred = pred.squeeze(1).cpu().numpy()  # [B, H, W]
            raw_rgbs = batch["raw_rgb"].permute(0, 2, 3, 1).numpy()  # [B, H, W, 3]
            for arr, name, raw_rgb in zip(pred, batch["name"], raw_rgbs):
                if args.guided_filter:
                    arr = guided_filter(
                        raw_rgb,
                        arr,
                        radius=args.gf_radius,
                        eps=args.gf_eps,
                    )
                stem = name.replace("_rgb.png", "")
                np.save(out_dir / f"{stem}.npy", arr.astype(np.float32))

    print(f"[predict] wrote {len(files)} .npy files to {out_dir}")
    if not args.val:
        print(f"[predict] next: run create_submission.py pointing pred_dir at {out_dir}")


if __name__ == "__main__":
    main()
