"""Render RGB + GT depth + predicted depth + error map side-by-side for named images.

Usage (from repo root):
    python scripts/visualize_cases.py \\
        --config configs/finetune/da_v2_vits_lora.yaml \\
        --no-lora \\
        --names train_012119_rgb.png train_002809_rgb.png \\
        --out-dir outputs/inspections/zero_shot_worst/

The output is one PNG per input image with a 2x2 panel:
    RGB   |   GT depth (viridis)
    Pred  |   log-error heatmap (seismic)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.common.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.common.metrics import si_rmse_per_image  # noqa: E402
from src.common.utils import load_ckpt, set_seed  # noqa: E402
from src.finetune.model import LoRAConfig, build_model  # noqa: E402


def load_sample(train_root: Path, name: str):
    rgb_path = train_root / name
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    depth = np.load(str(rgb_path).replace("_rgb.png", "_depth.npy")).astype(np.float32)
    mask = depth > 0
    return rgb, depth, mask


def make_figure(rgb, gt, mask, pred, sirmse, out_path, title):
    # Hide invalid GT pixels.
    gt_vis = gt.copy()
    gt_vis[~mask] = np.nan

    # log-error, also masked.
    eps = 1e-6
    err = np.log(np.clip(pred, eps, None)) - np.log(np.clip(gt, eps, None))
    err_vis = err.copy()
    err_vis[~mask] = np.nan
    vlim = float(np.nanpercentile(np.abs(err_vis), 98))

    # Shared depth colour scale over valid GT + pred, robust to outliers.
    combined = np.concatenate([gt[mask].flatten(), pred[mask].flatten()])
    vmin = float(np.nanpercentile(combined, 2))
    vmax = float(np.nanpercentile(combined, 98))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB"); axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(gt_vis, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"GT depth (valid {mask.mean()*100:.1f}%)"); axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Predicted depth"); axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(err_vis, cmap="seismic", vmin=-vlim, vmax=vlim)
    axes[1, 1].set_title(f"log(pred) - log(gt)   (±p98 = {vlim:.2f})")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}   siRMSE = {sirmse:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--names", nargs="+", required=True,
                        help="Image filenames like train_012119_rgb.png")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-lora", action="store_true")
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
    model.eval()

    train_root = Path(cfg["data"]["root"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    for name in args.names:
        rgb, gt, mask = load_sample(train_root, name)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x - mean) / std

        with torch.no_grad():
            pred = model(x)                       # [1, 1, H, W]

        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        sirmse = si_rmse_per_image(pred, gt_t, mask_t).item()

        pred_np = pred.squeeze().cpu().numpy()
        out_path = out_dir / f"{Path(name).stem}.png"
        make_figure(rgb, gt, mask, pred_np, sirmse, out_path, title=name)
        print(f"  {name:30s}  siRMSE={sirmse:.3f}  ->  {out_path}")

    print(f"[done] wrote {len(args.names)} figures to {out_dir}")


if __name__ == "__main__":
    main()
