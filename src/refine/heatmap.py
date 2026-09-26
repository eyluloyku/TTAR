"""Heat-map of where D0 is suspected to be wrong, and anchor selection on it.

The heat-map combines up to three signals:

    H = alpha * z(|grad log D0|)         (depth-edge proxy; §5.2)
      + beta  * z(-log D0)               (near-field proxy; §5.2)
      + gamma * z(tta_var)               (TTA log-depth variance; optional)

All three channels are standardised per-image to mean 0 / std 1 so the
weights are dimensionless and comparable. High H ↔ pixel is suspected to
carry high siRMSE error.

The first two are *correlation-based* proxies (validated in §5.2 /
`outputs/diagnostics/error_localizability/`). The third is the only
epistemic-uncertainty signal — it requires a precomputed array from
`scripts/precompute_tta_variance.py` and adds nothing under ablation (§8).
Pass `tta_var=None` and `gamma=0` (the default) to use only the two
validated proxies.

`select_anchors_heatmap` does greedy heat-weighted top-K with spatial
suppression (a Gaussian-disk non-max trick) so the K anchors don't all sit
on the same edge.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def compute_heatmap(
    D0: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    tta_var: np.ndarray | None = None,
) -> np.ndarray:
    """Per-image standardised error-likelihood heat-map. Same shape as D0.

    Args:
        D0:      [H, W] baseline depth (positive, finite).
        alpha:   weight on |grad log D0|.
        beta:    weight on -log D0 (near-field bias).
        gamma:   weight on per-pixel TTA log-depth variance. Only used when
                 `tta_var` is provided; ignored otherwise.
        tta_var: optional [H, W] precomputed variance from
                 `scripts/precompute_tta_variance.py`. Must match D0's shape.
    """
    log_d = np.log(np.maximum(D0, EPS)).astype(np.float32)
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    def z(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / (s + EPS)

    H = alpha * z(grad) + beta * z(-log_d)
    if gamma != 0.0 and tta_var is not None:
        if tta_var.shape != D0.shape:
            raise ValueError(
                f"tta_var shape {tta_var.shape} does not match D0 shape {D0.shape}"
            )
        H = H + gamma * z(tta_var.astype(np.float32))
    return H.astype(np.float32)


def select_anchors_heatmap(
    H: np.ndarray,
    K: int,
    min_dist_px: int = 24,
    top_frac: float = 0.20,
    valid: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Greedy heat-weighted top-K with spatial suppression.

    Args:
        H:           [H, W] heat-map.
        K:           number of anchors to return.
        min_dist_px: each chosen pixel suppresses a disk of this radius.
        top_frac:    only consider the top fraction of pixels by H to avoid
                     drifting into low-heat junk if we run out.
        valid:       optional [H, W] bool mask; only pixels with valid=True
                     are eligible.

    Returns:
        list of (y, x) integer pixel coordinates, len <= K.
    """
    Hh, Ww = H.shape
    score = H.astype(np.float32).copy()
    if valid is not None:
        score = np.where(valid, score, -np.inf)

    finite = np.isfinite(score)
    if not finite.any():
        return []
    threshold = float(np.quantile(score[finite], 1 - top_frac))
    eligible = score >= threshold

    suppress = np.zeros((Hh, Ww), dtype=bool)
    chosen: list[tuple[int, int]] = []
    masked = np.where(eligible, score, -np.inf)

    while len(chosen) < K:
        idx = int(np.argmax(masked))
        if not np.isfinite(masked.flat[idx]):
            break
        y, x = divmod(idx, Ww)
        chosen.append((y, x))
        y0, y1 = max(0, y - min_dist_px), min(Hh, y + min_dist_px + 1)
        x0, x1 = max(0, x - min_dist_px), min(Ww, x + min_dist_px + 1)
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        disk = (ys - y) ** 2 + (xs - x) ** 2 <= min_dist_px ** 2
        suppress[y0:y1, x0:x1] |= disk
        masked[suppress] = -np.inf
    return chosen


def select_anchors_random(
    K: int,
    shape: tuple[int, int],
    valid: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Uniform random valid-pixel sampling. Baseline / control."""
    Hh, Ww = shape
    rng = rng or np.random.default_rng()
    if valid is None:
        valid = np.ones((Hh, Ww), dtype=bool)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []
    take = min(K, len(idx))
    sel = rng.choice(idx, size=take, replace=False)
    return [(int(i // Ww), int(i % Ww)) for i in sel]
