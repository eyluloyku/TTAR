"""Evaluate a model on the validation split and report siRMSE.

Examples:
    # Zero-shot baseline (pretrained DA-V2, no LoRA, no training):
    python -m src.finetune.evaluate \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --no-lora

    # A trained checkpoint (includes LoRA weights + DPT head):
    python -m src.finetune.evaluate \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt

The val split is the same deterministic hold-out that train.py uses (seeded in
the config), so zero-shot and fine-tuned scores from this script are directly
comparable apples-to-apples.
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml
from torch.utils.data import DataLoader

from src.common.data import DepthDataset, make_splits
from src.common.metrics import si_rmse_per_image
from src.common.utils import load_ckpt, set_seed
from src.finetune.model import LoRAConfig, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Training checkpoint to load on top of the base model. "
                             "Omit for zero-shot (pretrained only).")
    parser.add_argument("--no-lora", action="store_true",
                        help="Skip LoRA wrapping even if the config specifies it. "
                             "Use for the raw pretrained zero-shot baseline.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    set_seed(cfg["data"]["seed"])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    lora_dict = cfg["model"].get("lora")
    lora_cfg = None if (args.no_lora or lora_dict is None) else LoRAConfig(**lora_dict)

    model = build_model(
        encoder=cfg["model"]["encoder"],
        checkpoint=cfg["model"].get("checkpoint"),
        lora_cfg=lora_cfg,
    ).to(device)

    if args.checkpoint:
        load_ckpt(args.checkpoint, model, map_location=device, strict=True)
        print(f"[eval] loaded training checkpoint: {args.checkpoint}")
    else:
        print(f"[eval] no --checkpoint (zero-shot); lora={'off' if lora_cfg is None else 'on (random init)'}")

    model.eval()

    _, val_files = make_splits(
        cfg["data"]["root"], val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"],
    )
    print(f"[eval] val split: {len(val_files)} images   (seed={cfg['data']['seed']})")

    ds = DepthDataset(val_files, has_depth=True, augment=None)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    per_image = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            pred = model(images)
            s = si_rmse_per_image(pred, depth, mask)
            for v, n in zip(s.cpu().tolist(), batch["name"]):
                if v == v:   # drop NaN (image with no valid pixels, should be rare)
                    per_image.append((n, v))

    elapsed = time.time() - t0
    scores = torch.tensor([v for _, v in per_image])
    mean = scores.mean().item()
    median = scores.median().item()
    q10 = torch.quantile(scores, 0.10).item()
    q90 = torch.quantile(scores, 0.90).item()

    print()
    print(f"[eval] N={len(per_image)}   elapsed={elapsed:.1f}s")
    print(f"[eval] val siRMSE   mean={mean:.4f}   median={median:.4f}")
    print(f"[eval]              p10={q10:.4f}    p90={q90:.4f}")
    print(f"[eval] worst 5: " + ", ".join(f"{n}:{v:.3f}"
          for n, v in sorted(per_image, key=lambda x: -x[1])[:5]))
    print(f"[eval] best  5: " + ", ".join(f"{n}:{v:.3f}"
          for n, v in sorted(per_image, key=lambda x: x[1])[:5]))


if __name__ == "__main__":
    main()
