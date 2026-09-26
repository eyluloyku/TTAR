"""Training losses for Stage A.

The Kaggle metric is siRMSE (scale-invariant RMSE in log-depth; see
``src.common.metrics``). Training directly on siRMSE is possible but the sqrt
has a singular gradient at zero and the metric uses lambda=1 (the maximally
scale-invariant regime), which empirically trains slowly.

The standard fix is the SILog loss (Eigen et al. 2014, used by BinsFormer,
AdaBins, DA-V2 supervised fine-tuning): a lambda-weighted log-variance term
with lambda in [0.5, 0.85]. Gradient is well-behaved and scale bias is still
allowed to leak through, which gives faster convergence.

    SILog(pred, gt) = alpha * sqrt( mean(d^2) - lambda * (mean d)^2 )
    where d = log(pred) - log(gt) over valid pixels.

Set lambda=1.0 to train directly on the metric.
"""

from __future__ import annotations

import torch

EPS = 1e-6


def silog_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    lambda_: float = 0.85,
    alpha: float = 10.0,
) -> torch.Tensor:
    """Per-image SILog, averaged across the batch (images with no valid pixels skipped).

    Args:
        pred, gt: [B, 1, H, W] positive depths in meters.
        mask:     [B, 1, H, W] nonzero = valid pixel.
        lambda_:  weight on the squared-mean term. 1.0 = fully scale-invariant.
        alpha:    output scaling used in most depth papers (~10); keeps the loss
                  in a range that plays well with standard LRs.
    """
    pred = pred.clamp_min(EPS)
    gt = gt.clamp_min(EPS)

    d = torch.log(pred) - torch.log(gt)             # [B, 1, H, W]
    m = (mask > 0).float()
    n = m.flatten(1).sum(dim=1).clamp_min(1.0)      # [B]

    d_flat = (d * m).flatten(1)
    mean_d = d_flat.sum(dim=1) / n
    mean_d2 = (d_flat * d_flat).sum(dim=1) / n

    var = (mean_d2 - lambda_ * mean_d.pow(2)).clamp_min(EPS)
    per_image = alpha * torch.sqrt(var)

    valid = mask.flatten(1).sum(dim=1) > 0
    if not valid.any():
        return per_image.sum() * 0.0
    return per_image[valid].mean()
