"""Precompute per-image TTA log-depth variance for the val (or test) split.

Writes one ``<stem>.npy`` float32 array of shape [H, W] (var of log D across
TTA augmentations) per image into ``--out-dir``. Downstream consumers
(``compute_heatmap``'s γ channel, ``diagnose_error_signals.py``) load
these arrays directly — no model needed at eval time.

Usage:
    python scripts/precompute_tta_variance.py \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt \\
        --val \\
        --out-dir outputs/val_preds/vits_lora_r8/tta_var/

5 forward passes per image (see src/refine/tta.py for the augmentation set).
Expect ~5× the runtime of plain `src.finetune.predict --val`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import DepthDataset, make_splits  # noqa: E402
from src.common.utils import load_ckpt  # noqa: E402
from src.finetune.model import LoRAConfig, build_model  # noqa: E402
from src.refine.tta import predict_tta  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-root")
    mode.add_argument("--val", action="store_true")
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
        _, files = make_splits(
            cfg["data"]["root"],
            val_frac=cfg["data"]["val_frac"],
            seed=cfg["data"]["seed"],
        )
        print(f"[tta] val split: {len(files)} images")
    else:
        files = sorted(Path(args.test_root).glob("*_rgb.png"))
        if not files:
            raise FileNotFoundError(f"No *_rgb.png under {args.test_root}")
        print(f"[tta] test set: {len(files)} images")

    # batch_size=1 because TTA forms its own internal batch of N augmentations
    # per image; doubling the outer dimension would blow up activation memory
    # for marginal speed-up on a 5060-ti.
    ds = DepthDataset(files, has_depth=False, augment=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, batch in enumerate(loader):
        raw_rgb = batch["raw_rgb"][0]  # [3, H, W] in [0, 1]
        name = batch["name"][0]
        stem = name.replace("_rgb.png", "")
        out = predict_tta(model, raw_rgb, device=device)
        np.save(out_dir / f"{stem}.npy", out["var_logD"])
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(files) - (i + 1)) / max(rate, 1e-6)
            print(f"  {i+1}/{len(files)}  ({rate:.2f} img/s)  ETA {eta/60:.1f} min")

    print(f"[tta] wrote {len(files)} variance .npy files to {out_dir}")


if __name__ == "__main__":
    main()
