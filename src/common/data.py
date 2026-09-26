"""Depth dataset + augmentations.

Dataset layout on the cluster:
    /cluster/courses/cil/monocular-depth-estimation/
        train/ train_NNNNNN_rgb.png  +  train_NNNNNN_depth.npy   (22605 pairs)
        test/  test_NNNNNN_rgb.png                               ( 591 images, no GT)

All images are native 560x560. Depths are float32 meters with many invalid
pixels stored as 0 (mask = depth > 0). We keep depth in meters throughout —
the siRMSE metric and the DA-V2 decoder are both happy with that range.

RGB is normalised with ImageNet statistics (what DA-V2's DINOv2 encoder was
trained with). Depth is returned raw; only the loss / metric touches log-space.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

NATIVE_SIZE = 560  # 14 * 40, multiple of the DA-V2 patch size


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image as a [H, W, 3] uint8 array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def make_splits(
    train_root: str | Path,
    val_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Deterministic train/val split by hash of filename.

    Hashing the stem gives every teammate the same split without shipping a
    manifest, and keeps new files landing in a consistent bucket if the dataset
    is ever extended.
    """
    train_root = Path(train_root)
    rgb_files = sorted(train_root.glob("*_rgb.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png under {train_root}")

    threshold = int(val_frac * 2**32)
    salt = str(seed).encode()

    train, val = [], []
    for p in rgb_files:
        h = hashlib.md5(salt + p.stem.encode()).digest()
        bucket = int.from_bytes(h[:4], "big")
        (val if bucket < threshold else train).append(p)
    return train, val


@dataclass
class AugConfig:
    crop_size: int | None = None           # e.g. 518 for random crops; None = no crop
    hflip_prob: float = 0.5
    color_jitter: float = 0.2              # brightness/contrast/saturation amplitude; 0 disables


class DepthDataset(Dataset):
    """RGB + depth pairs (or RGB-only for test).

    Args:
        rgb_files:   explicit list of RGB paths. For the train/val split use
                     ``make_splits``; for test, pass ``sorted(test_root.glob("*_rgb.png"))``.
        has_depth:   whether to load *_depth.npy alongside each RGB. False for
                     the test set.
        augment:     augmentation config; None = no augmentation (val/test).
    """

    def __init__(
        self,
        rgb_files: Sequence[Path],
        has_depth: bool = True,
        augment: AugConfig | None = None,
    ):
        self.rgb_files = list(rgb_files)
        self.has_depth = has_depth
        self.aug = augment

    def __len__(self) -> int:
        return len(self.rgb_files)

    def __getitem__(self, idx: int) -> dict:
        rgb_path = self.rgb_files[idx]
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # [3, H, W]

        if self.has_depth:
            depth_path = rgb_path.with_name(rgb_path.name.replace("_rgb.png", "_depth.npy"))
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
            mask = (depth > 0).float()
        else:
            depth = torch.zeros(1, rgb.shape[1], rgb.shape[2])
            mask = torch.zeros_like(depth)

        if self.aug is not None:
            rgb, depth, mask = self._apply_aug(rgb, depth, mask)

        raw_rgb = rgb.clone()  # [3, H, W] in [0, 1] before normalization
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "image": rgb,
            "raw_rgb": raw_rgb,
            "depth": depth,
            "mask": mask,
            "name": rgb_path.name,
        }

    def _apply_aug(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aug = self.aug
        assert aug is not None

        # Horizontal flip (applied to rgb, depth, mask together).
        if aug.hflip_prob > 0 and random.random() < aug.hflip_prob:
            rgb = torch.flip(rgb, dims=[2])
            depth = torch.flip(depth, dims=[2])
            mask = torch.flip(mask, dims=[2])

        # Random crop.
        if aug.crop_size is not None:
            h, w = rgb.shape[1], rgb.shape[2]
            ch = cw = aug.crop_size
            if ch > h or cw > w:
                raise ValueError(f"crop_size {aug.crop_size} > image {(h, w)}")
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            rgb = rgb[:, top:top + ch, left:left + cw]
            depth = depth[:, top:top + ch, left:left + cw]
            mask = mask[:, top:top + ch, left:left + cw]

        # Color jitter on RGB only.
        if aug.color_jitter > 0:
            a = aug.color_jitter
            # brightness
            rgb = rgb * (1.0 + random.uniform(-a, a))
            # contrast
            mean = rgb.mean(dim=(1, 2), keepdim=True)
            rgb = (rgb - mean) * (1.0 + random.uniform(-a, a)) + mean
            # saturation
            gray = rgb.mean(dim=0, keepdim=True)
            rgb = (rgb - gray) * (1.0 + random.uniform(-a, a)) + gray
            rgb = rgb.clamp(0.0, 1.0)

        return rgb, depth, mask