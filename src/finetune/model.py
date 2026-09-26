"""Depth Anything V2 wrapper + LoRA adapters.

The DA-V2 class is imported from the submodule at ``third_party/Depth-Anything-V2``.
This module adds:
  * a channel dim to the model output ([B, H, W] -> [B, 1, H, W]) so downstream
    code has a consistent shape with the dataset tensors;
  * a ``build_model`` factory that loads a pretrained checkpoint and, when a
    LoRA config is passed, wraps the encoder's attention Linear layers with
    peft adapters while keeping the DPT head fully trainable.

Training parameter-efficient LoRA + full DPT head is our default Stage A recipe.
Full fine-tuning (``lora_cfg=None``) is supported for ablations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DA_V2_PATH = _REPO_ROOT / "third_party" / "Depth-Anything-V2"
if str(_DA_V2_PATH) not in sys.path:
    sys.path.insert(0, str(_DA_V2_PATH))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


# DPT-head feature widths per encoder size (from DA-V2's run.py).
_ENCODER_CONFIGS = {
    "vits": {"features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    # Regex over fully-qualified module names. Default scopes LoRA to encoder
    # attention Linears only (avoids wrapping DPT head layers that happen to
    # share leaf names like "proj").
    target_modules_regex: str = r"pretrained\.blocks\.\d+\.attn\.(qkv|proj)"


class DepthAnythingWrapper(nn.Module):
    """Thin wrapper: forwards to DA-V2 and unsqueezes a channel dim."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)               # [B, H, W]
        return y.unsqueeze(1)          # [B, 1, H, W]


def build_model(
    encoder: str = "vits",
    checkpoint: str | Path | None = None,
    lora_cfg: LoRAConfig | None = None,
) -> nn.Module:
    """Instantiate DA-V2, optionally load a checkpoint, optionally attach LoRA.

    Args:
        encoder: one of 'vits', 'vitb', 'vitl', 'vitg'.
        checkpoint: path to a DA-V2 .pth (state_dict). Skipped if None.
        lora_cfg:   if set, wrap encoder attention Linears with LoRA and leave
                    the DPT head fully trainable; encoder-base weights are
                    frozen. If None, the whole model is trainable.

    Returns:
        A module whose forward(x) returns [B, 1, H, W] float depth in meters.
    """
    if encoder not in _ENCODER_CONFIGS:
        raise ValueError(f"encoder must be one of {list(_ENCODER_CONFIGS)}")

    base = DepthAnythingV2(encoder=encoder, **_ENCODER_CONFIGS[encoder])

    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        # DA-V2 checkpoints are raw state_dicts.
        missing, unexpected = base.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[build_model] missing={len(missing)} unexpected={len(unexpected)} keys")
            if missing: print(f"  e.g. missing: {missing[:3]}")
            if unexpected: print(f"  e.g. unexpected: {unexpected[:3]}")

    if lora_cfg is None:
        return DepthAnythingWrapper(base)

    # Lazy import — peft is an optional dep at the module level.
    from peft import LoraConfig, get_peft_model

    # Freeze encoder; LoRA will reintroduce trainable low-rank params on the
    # target modules. DPT head stays trainable (listed below in modules_to_save).
    for p in base.pretrained.parameters():
        p.requires_grad = False

    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules_regex,
        modules_to_save=["depth_head"],
        bias="none",
    )
    base = get_peft_model(base, peft_cfg)
    return DepthAnythingWrapper(base)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
