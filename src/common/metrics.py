"""Scale-invariant RMSE (siRMSE) — the Kaggle scoring metric.

The competition defines siRMSE (Eigen et al. 2014): log-transform both depths,
take the per-pixel difference d_i = log(pred_i) - log(gt_i), remove the optimal
scalar log-scale bias alpha = mean_i(d_i) (this is what makes it
scale-invariant — the dataset's depth scales are only meaningful *within* an
image), then take the RMSE of the residual. Substituting the optimal alpha gives
the closed form used here, over a valid-depth mask M:

    siRMSE = sqrt( mean_i(d_i^2) - (mean_i(d_i))^2 ),     i in M

We compute this per image and average across images, matching the dataset's
per-image scale convention. The SILog training loss with lambda<1 is a softer
proxy; this metric uses lambda=1 and takes the sqrt.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6


def si_rmse_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image scale-invariant RMSE for numpy depth maps (meters).

    Numpy counterpart to :func:`si_rmse`, used by the evaluation and
    diagnostic scripts that score saved ``.npy`` predictions against GT.
    Pixels are valid where both depths are positive and finite; images with
    fewer than two valid pixels return NaN.
    """
    valid = (gt > 0) & (pred > 0) & np.isfinite(pred) & np.isfinite(gt)
    if valid.sum() < 2:
        return float("nan")
    d = np.log(np.maximum(pred[valid], EPS)) - np.log(np.maximum(gt[valid], EPS))
    return float(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2))


def _log_diff(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.log(pred.clamp_min(EPS)) - torch.log(gt.clamp_min(EPS))


def si_rmse_per_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-image scale-invariant RMSE.

    Args:
        pred, gt: float tensors [B, 1, H, W] or [B, H, W]. Units: meters.
        mask:     float/bool tensor with the same shape. Nonzero = valid pixel.

    Returns:
        [B] tensor of per-image siRMSE. Images with zero valid pixels yield NaN;
        the caller should filter or ignore them.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        gt = gt.unsqueeze(1)
        mask = mask.unsqueeze(1)

    d = _log_diff(pred, gt)  # [B, 1, H, W]
    m = (mask > 0).float()
    n = m.flatten(1).sum(dim=1).clamp_min(1.0)  # [B]

    d_flat = (d * m).flatten(1)
    sum_d = d_flat.sum(dim=1)
    sum_d2 = (d_flat * d_flat).sum(dim=1)

    mean_d = sum_d / n
    mean_d2 = sum_d2 / n
    var = (mean_d2 - mean_d * mean_d).clamp_min(0.0)

    sirmse = torch.sqrt(var)

    # Images with no valid pixels: return NaN so the caller can drop them.
    no_valid = mask.flatten(1).sum(dim=1) == 0
    sirmse = sirmse.masked_fill(no_valid, float("nan"))
    return sirmse


def si_rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-image siRMSE across the batch, ignoring images with no valid pixels."""
    per_image = si_rmse_per_image(pred, gt, mask)
    valid = ~torch.isnan(per_image)
    if not valid.any():
        return torch.tensor(float("nan"), device=pred.device)
    return per_image[valid].mean()
