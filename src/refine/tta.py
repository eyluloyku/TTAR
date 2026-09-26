"""Test-time-augmentation variance as an epistemic-uncertainty signal.

The heat-map uses `|grad log D0|` and `−log D0` as suspect-pixel
proxies — both validated as *correlated* with error in §5.2, but neither is
a true model-uncertainty signal: a confident sharp edge has high gradient,
yet may also be perfectly localised, and near-field pixels are harder *on
average* but the signal is per-image-stable rather than per-pixel-informative.

This module produces a per-pixel **variance of log-depth across a small set
of label-preserving augmentations**. Pixels where the model is genuinely
uncertain (texture-ambiguous regions, semi-transparent surfaces, distant
sky, repeated patterns) move under augmentation; pixels where the model is
locked in (textured planes with strong geometric cues) stay still. This is
what we plug into `compute_heatmap` as a third channel γ·z(TTA_var).

The augmentations are deliberately *small* and *grid-preserving* so that
per-pixel comparison is meaningful:

  - identity                                 (anchors the variance)
  - horizontal flip                          (geometry symmetry probe)
  - brightness ±delta                        (photometric perturbation)
  - hflip + brightness +delta                (combined)

Five passes total — ~5× DA-V2 inference. Multi-scale crops are deliberately
excluded: they break pixel correspondence and require resampling that
introduces its own variance, swamping the model-uncertainty signal we
actually care about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.common.data import IMAGENET_MEAN, IMAGENET_STD

EPS = 1e-6


@dataclass(frozen=True)
class TTAOp:
    """One augmentation = a forward transform on raw_rgb in [0,1] and the
    inverse transform on the predicted depth so all predictions live on the
    same pixel grid."""
    name: str
    hflip: bool = False
    brightness: float = 0.0  # additive in [0,1] space, clipped to [0,1]


DEFAULT_OPS: tuple[TTAOp, ...] = (
    TTAOp("id"),
    TTAOp("hflip", hflip=True),
    TTAOp("bright_p", brightness=+0.05),
    TTAOp("bright_n", brightness=-0.05),
    TTAOp("hflip_bright_p", hflip=True, brightness=+0.05),
)


def _apply(raw_rgb: torch.Tensor, op: TTAOp) -> torch.Tensor:
    """raw_rgb: [3, H, W] in [0, 1]. Returns transformed [3, H, W]."""
    x = raw_rgb
    if op.brightness != 0.0:
        x = (x + op.brightness).clamp(0.0, 1.0)
    if op.hflip:
        x = torch.flip(x, dims=[2])
    return x


def _unapply_depth(depth: torch.Tensor, op: TTAOp) -> torch.Tensor:
    """depth: [H, W]. Inverts the spatial part of `op` (brightness is RGB-only)."""
    if op.hflip:
        depth = torch.flip(depth, dims=[1])
    return depth


@torch.no_grad()
def predict_tta(
    model: torch.nn.Module,
    raw_rgb: torch.Tensor,
    device: torch.device,
    ops: tuple[TTAOp, ...] = DEFAULT_OPS,
    batch: bool = True,
) -> dict:
    """Run TTA and aggregate.

    Args:
        model:    callable [B,3,H,W] (ImageNet-normalised) -> [B,1,H,W] depth in m.
        raw_rgb:  [3, H, W] float tensor in [0, 1] (un-normalised). Single image.
        device:   torch device the model lives on.
        ops:      tuple of TTAOp; defaults to DEFAULT_OPS (5 passes).
        batch:    if True, stack all augmentations into a single forward pass.

    Returns:
        dict with keys:
          'mean_D'   : [H, W] float32, mean depth in meters across ops
          'var_logD' : [H, W] float32, variance of log-depth across ops
          'n'        : int, number of ops used
    """
    if raw_rgb.dim() != 3 or raw_rgb.shape[0] != 3:
        raise ValueError(f"raw_rgb must be [3,H,W], got {tuple(raw_rgb.shape)}")
    H, W = raw_rgb.shape[1], raw_rgb.shape[2]

    mean_t = IMAGENET_MEAN.to(raw_rgb.device)
    std_t = IMAGENET_STD.to(raw_rgb.device)

    augmented = [_apply(raw_rgb, op) for op in ops]
    normalised = [(a - mean_t) / std_t for a in augmented]

    if batch:
        x = torch.stack(normalised, dim=0).to(device, non_blocking=True)  # [N,3,H,W]
        y = model(x).squeeze(1).cpu()  # [N, H, W]
        preds = [y[i] for i in range(len(ops))]
    else:
        preds = []
        for a in normalised:
            yi = model(a.unsqueeze(0).to(device, non_blocking=True)).squeeze(0).squeeze(0).cpu()
            preds.append(yi)

    aligned = [_unapply_depth(p, op) for p, op in zip(preds, ops)]
    stack = torch.stack(aligned, dim=0).clamp(min=EPS).float()  # [N, H, W]
    log_stack = torch.log(stack)
    mean_D = stack.mean(dim=0).numpy().astype(np.float32)
    var_logD = log_stack.var(dim=0, unbiased=False).numpy().astype(np.float32)
    return {"mean_D": mean_D, "var_logD": var_logD, "n": len(ops)}
