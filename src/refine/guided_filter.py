"""Guided image filter for depth boundary sharpening.

Uses the RGB image as a guide to transfer sharp edges from the photo
onto the (blurry) depth map. Operates in log-depth space to match siRMSE.

Reference: He et al., "Guided Image Filtering", TPAMI 2013.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def guided_filter(
    rgb: np.ndarray,
    depth: np.ndarray,
    radius: int = 8,
    eps: float = 1e-4,
) -> np.ndarray:
    """Sharpen depth boundaries using the RGB image as a guide.

    Args:
        rgb:    [H, W, 3] float32 in [0, 1] or uint8.
        depth:  [H, W] float32 depth in meters (invalid pixels = 0).
        radius: filter window half-size in pixels. Larger = more smoothing.
        eps:    regularization. Smaller = sharper edges but more noise.

    Returns:
        [H, W] float32 filtered depth, invalid pixels preserved as 0.
    """
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0

    # Grayscale guide: standard luminance weights.
    guide = (0.299 * rgb[:, :, 0]
             + 0.587 * rgb[:, :, 1]
             + 0.114 * rgb[:, :, 2]).astype(np.float32)

    valid = (depth > 0).astype(np.float32)

    # Work in log-depth space to match siRMSE.
    # Invalid pixels are excluded via validity weighting so they don't
    # corrupt the filter output near boundaries.
    log_depth = np.where(depth > 0, np.log(np.maximum(depth, 1e-6)), 0.0)

    filtered_log = _guided_filter_gray_valid(guide, log_depth, valid, radius, eps)

    result = np.where(depth > 0, np.exp(filtered_log), 0.0)
    return result.astype(np.float32)


def _guided_filter_gray_valid(
    guide: np.ndarray,
    src: np.ndarray,
    valid: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Validity-weighted guided filter — invalid pixels don't contribute to averages."""
    size = 2 * radius + 1

    # Weighted sums: only valid pixels contribute.
    w_sum    = uniform_filter(valid,              size)
    wI_sum   = uniform_filter(valid * guide,      size)
    wp_sum   = uniform_filter(valid * src,        size)
    wIp_sum  = uniform_filter(valid * guide * src, size)
    wII_sum  = uniform_filter(valid * guide ** 2, size)

    safe_w = np.maximum(w_sum, 1e-6)

    mean_I  = wI_sum  / safe_w
    mean_p  = wp_sum  / safe_w
    mean_Ip = wIp_sum / safe_w
    mean_II = wII_sum / safe_w

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I ** 2

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size)
    mean_b = uniform_filter(b, size)

    return mean_a * guide + mean_b
