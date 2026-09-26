"""SAM-region-based depth smoothing.

For each SAM region, blend the per-pixel Stage A depth toward a per-region
target in log-depth space:

    D_refined[p] = exp( alpha * log_target[p] + (1 - alpha) * log D0[p] )

The target is set by ``aggregator``:

* ``"median"``: region-wide constant (median of valid log-depths).
* ``"plane"``: 2D linear fit a*x + b*y + c to the valid pixels'
  (xs, ys, log D0), evaluated per-pixel.

Pixels outside every region keep D0 unchanged.

Region masks are cached so sweeps reuse the SAM pass. The cache directory
is keyed by SAM-relevant kwargs only, so swapping aggregators hits the
same cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from .segment import Region, RegionExtractor

# Kwargs whose values change SAM's output. Folded into the cache subdir
# name so changing any threshold invalidates only the affected masks.
_CACHE_RELEVANT_KEYS = (
    "model_type",
    "min_area_frac", "max_area_frac", "n_splits", "max_regions",
    "points_per_side", "pred_iou_thresh", "stability_score_thresh",
    "crop_n_layers", "min_mask_region_area",
)


class SAMRegionSmoothRefiner:
    """Smooth Stage A depth by pulling each pixel toward a per-region target.

    Region masks are cached to ``<cache_dir>/<config_hash>/<image_hash>.npz``.
    SAM is loaded lazily, so a run whose images are fully cached never imports
    segment_anything.

    Args:
        alpha: blend weight toward the per-region target in log space.
        aggregator: "median" or "plane".
        boundary_only: if True, apply the blend only to a band_width-pixel rim
            around each region. The aggregator is still fit on the full region.
        band_width: rim width in pixels when boundary_only=True.
        min_valid_pixels: regions with fewer valid pixels are skipped.
        max_plane_residual: with aggregator="plane", regions whose per-pixel
            RMSE exceeds this threshold are left unsmoothed. Default disables
            the gate.
        sam_checkpoint: SAM weights path.
        cache_dir: directory for cached region masks.
        sam_kwargs: forwarded to RegionExtractor.
    """

    name = "region_smooth"

    def __init__(
        self,
        alpha: float = 0.5,
        aggregator: str = "median",
        boundary_only: bool = False,
        band_width: int = 3,
        min_valid_pixels: int = 10,
        max_plane_residual: float = float("inf"),
        sam_checkpoint: str = "checkpoints/sam_vit_b.pth",
        cache_dir: str = "outputs/stage_b_cache/sam_regions",
        **sam_kwargs,
    ):
        if aggregator not in ("median", "plane"):
            raise ValueError(f"aggregator must be 'median' or 'plane', got {aggregator!r}")
        self.alpha = float(alpha)
        self.aggregator = aggregator
        self.boundary_only = bool(boundary_only)
        self.band_width = int(band_width)
        self.min_valid_pixels = int(min_valid_pixels)
        self.max_plane_residual = float(max_plane_residual)
        self.sam_checkpoint = sam_checkpoint
        self.sam_kwargs = sam_kwargs
        self.regions_smoothed = 0
        self.regions_skipped = 0

        config = {k: sam_kwargs[k] for k in _CACHE_RELEVANT_KEYS if k in sam_kwargs}
        cfg_hash = hashlib.blake2b(
            json.dumps(config, sort_keys=True).encode(),
            digest_size=6,
        ).hexdigest()
        self.cache_root = Path(cache_dir) / cfg_hash
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self._extractor: RegionExtractor | None = None  # lazy
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_extractor(self) -> RegionExtractor:
        if self._extractor is None:
            print(f"[region_smooth] cache miss → loading SAM from {self.sam_checkpoint}")
            self._extractor = RegionExtractor(
                sam_checkpoint=self.sam_checkpoint,
                **self.sam_kwargs,
            )
        return self._extractor

    def __call__(self, rgb: np.ndarray, D0: np.ndarray) -> np.ndarray:
        regions = self._load_or_extract(rgb)
        if not regions:
            return D0.astype(np.float32, copy=False)

        out = D0.astype(np.float32, copy=True)
        log_D0 = np.log(np.maximum(D0, 1e-6))

        for region in regions:
            valid = region.mask & (D0 > 0)
            n_valid = int(valid.sum())
            if n_valid < self.min_valid_pixels:
                continue
            log_vals_fit = log_D0[valid]

            # In boundary_only mode the aggregator is fit on the full region
            # for stability, then applied only to the rim.
            if self.boundary_only and self.band_width > 0:
                eroded = binary_erosion(region.mask, iterations=self.band_width)
                apply_mask = region.mask & ~eroded & (D0 > 0)
            else:
                apply_mask = valid
            if not apply_mask.any():
                continue

            if self.aggregator == "median":
                target_val = float(np.median(log_vals_fit))
                log_target = np.full(int(apply_mask.sum()), target_val, dtype=np.float32)
            else:  # "plane"
                ys_fit, xs_fit = np.where(valid)
                A_fit = np.column_stack([
                    xs_fit.astype(np.float32),
                    ys_fit.astype(np.float32),
                    np.ones(n_valid, dtype=np.float32),
                ])
                coeffs, *_ = np.linalg.lstsq(A_fit, log_vals_fit.astype(np.float32), rcond=None)
                fit_pred = (A_fit @ coeffs).astype(np.float32)
                residual = float(np.sqrt(np.mean((log_vals_fit - fit_pred) ** 2)))
                if residual > self.max_plane_residual:
                    self.regions_skipped += 1
                    continue

                if apply_mask is valid:
                    log_target = fit_pred
                else:
                    ys_a, xs_a = np.where(apply_mask)
                    A_apply = np.column_stack([
                        xs_a.astype(np.float32),
                        ys_a.astype(np.float32),
                        np.ones(len(ys_a), dtype=np.float32),
                    ])
                    log_target = (A_apply @ coeffs).astype(np.float32)

            self.regions_smoothed += 1
            log_vals_apply = log_D0[apply_mask]
            blended = self.alpha * log_target + (1.0 - self.alpha) * log_vals_apply
            out[apply_mask] = np.exp(blended).astype(np.float32)

        return out

    def _load_or_extract(self, rgb: np.ndarray) -> list[Region]:
        key = hashlib.blake2b(rgb.tobytes(), digest_size=12).hexdigest()
        path = self.cache_root / f"{key}.npz"
        if path.exists():
            self.cache_hits += 1
            return _load_regions(path)
        self.cache_misses += 1
        regions = self._get_extractor().extract(rgb)
        _save_regions(path, regions)
        return regions


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _save_regions(path: Path, regions: list[Region]) -> None:
    """Atomically save regions to a compressed .npz file."""
    if regions:
        masks = np.stack([r.mask for r in regions])                       # [N, H, W] bool
        areas = np.array([r.area_frac for r in regions], dtype=np.float32)
        bboxes = np.array([list(r.bbox) for r in regions], dtype=np.int32)
        parents = np.array(
            [r.parent_idx if r.parent_idx is not None else -1 for r in regions],
            dtype=np.int32,
        )
    else:
        masks = np.zeros((0, 1, 1), dtype=bool)
        areas = np.zeros((0,), dtype=np.float32)
        bboxes = np.zeros((0, 4), dtype=np.int32)
        parents = np.zeros((0,), dtype=np.int32)

    # Pass a file handle so numpy doesn't auto-append ".npz" to the tmp path
    # (which it does for string/Path targets that don't already end in .npz).
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, masks=masks, areas=areas, bboxes=bboxes, parents=parents)
    tmp.rename(path)


def _load_regions(path: Path) -> list[Region]:
    data = np.load(path)
    n = len(data["areas"])
    regions: list[Region] = []
    for i in range(n):
        p = int(data["parents"][i])
        regions.append(Region(
            mask=data["masks"][i],
            area_frac=float(data["areas"][i]),
            bbox=tuple(int(x) for x in data["bboxes"][i]),
            parent_idx=p if p >= 0 else None,
        ))
    return regions
