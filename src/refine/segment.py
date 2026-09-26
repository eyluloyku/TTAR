"""Stage B — SAM-based region extraction with spatial subdivision.

Usage:
    from src.refine.segment import RegionExtractor, Region

    extractor = RegionExtractor(sam_checkpoint="checkpoints/sam_vit_b.pth")
    regions = extractor.extract(rgb_image)  # rgb_image: HxWx3 uint8 numpy array
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class Region:
    mask: np.ndarray        # [H, W] bool
    area_frac: float        # fraction of total image pixels
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixels
    parent_idx: int | None = None    # if set, all regions sharing this id were
                                     # subdivided from the same SAM mask
                                     # (used to label intra- vs inter-object pairs)


class RegionExtractor:
    """Runs SAM on an image and returns filtered, optionally subdivided regions.

    SAM's automatic mask generator is tuned by default to favour high-confidence
    masks of large, sharp objects (walls, sky, vehicles). Foreground people and
    other small/cluttered objects get filtered out by the IoU and stability
    thresholds. We loosen those thresholds, raise the sampling density, and
    enable a second cropped pass so SAM emits more candidate masks.

    The candidates are then turned into a disjoint partition via **bottom-up
    pixel claiming**: walk masks small→large and restrict each to pixels not
    yet claimed by an earlier (smaller) mask. This naturally suppresses SAM's
    common failure mode where it emits both a small object mask (e.g. a
    pedestrian) and a large containing mask (the wall behind them) — the
    pedestrian claims their pixels first; the wall is left with the rest.

    Args:
        sam_checkpoint: path to SAM .pth weights file.
        model_type:     SAM model variant matching the checkpoint ('vit_b', 'vit_l', 'vit_h').
        min_area_frac:  regions whose post-claim area is below this fraction are dropped.
        max_area_frac:  regions larger than this are spatially subdivided.
        n_splits:       number of sub-regions to split large regions into.
        max_regions:    maximum number of regions returned (largest post-claim area first).
        points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers,
        min_mask_region_area: passed through to SamAutomaticMaskGenerator.
        device:         'cuda' or 'cpu'.
    """

    def __init__(
        self,
        sam_checkpoint: str = "checkpoints/sam_vit_b.pth",
        model_type: str = "vit_b",
        min_area_frac: float = 0.005,
        max_area_frac: float = 0.10,
        n_splits: int = 2,
        max_regions: int = 12,
        points_per_side: int = 64,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.90,
        crop_n_layers: int = 1,
        min_mask_region_area: int = 200,
        device: str = "cuda",
    ):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            min_mask_region_area=min_mask_region_area,
        )
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.n_splits = n_splits
        self.max_regions = max_regions

    def extract(self, rgb_image: np.ndarray) -> list[Region]:
        """Run SAM, claim pixels bottom-up, subdivide large regions, cap to max_regions."""
        h, w = rgb_image.shape[:2]
        total_pixels = h * w
        min_area_pix = int(self.min_area_frac * total_pixels)

        raw_masks = self.generator.generate(rgb_image)

        # Drop noise upfront, then walk smallest→largest so small specific
        # objects get first claim on their pixels.
        candidates = [m["segmentation"] for m in raw_masks if m["area"] >= min_area_pix]
        candidates.sort(key=lambda seg: int(seg.sum()))

        claimed = np.zeros((h, w), dtype=bool)
        regions: list[Region] = []
        next_parent = 0

        for seg in candidates:
            owned = seg & ~claimed
            owned_pix = int(owned.sum())
            if owned_pix < min_area_pix:
                continue
            owned_frac = owned_pix / total_pixels

            if owned_frac > self.max_area_frac:
                sub = _subdivide(owned, self.n_splits, total_pixels, parent_idx=next_parent)
                kept_sub = [s for s in sub if s.area_frac * total_pixels >= min_area_pix]
                if len(kept_sub) >= 2:
                    next_parent += 1
                    for s in kept_sub:
                        regions.append(s)
                        claimed |= s.mask
                elif len(kept_sub) == 1:
                    s = kept_sub[0]
                    regions.append(Region(mask=s.mask, area_frac=s.area_frac, bbox=s.bbox))
                    claimed |= s.mask
            else:
                bbox = _mask_bbox(owned)
                regions.append(Region(mask=owned, area_frac=owned_frac, bbox=bbox))
                claimed |= owned

        # Cap to the most prominent regions by post-claim area.
        regions.sort(key=lambda r: -r.area_frac)
        return regions[: self.max_regions]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) bounding box of a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1))


def _subdivide(mask: np.ndarray, n_splits: int, total_pixels: int, parent_idx: int | None = None) -> list[Region]:
    """Split a large mask into n_splits parts along its principal axis."""
    coords = np.argwhere(mask).astype(np.float32)  # [N, 2] (row, col)

    pca = PCA(n_components=1)
    projections = pca.fit_transform(coords).flatten()  # [N]

    percentiles = np.linspace(0, 100, n_splits + 1)
    sub_regions = []
    for i in range(n_splits):
        lo = np.percentile(projections, percentiles[i])
        hi = np.percentile(projections, percentiles[i + 1])
        # Include the upper boundary only on the last split to avoid gaps.
        if i < n_splits - 1:
            in_split = (projections >= lo) & (projections < hi)
        else:
            in_split = projections >= lo

        if not in_split.any():
            continue

        sub_mask = np.zeros_like(mask)
        split_coords = coords[in_split].astype(int)
        sub_mask[split_coords[:, 0], split_coords[:, 1]] = True

        area_frac = int(in_split.sum()) / total_pixels
        bbox = _mask_bbox(sub_mask)
        sub_regions.append(Region(mask=sub_mask, area_frac=area_frac, bbox=bbox, parent_idx=parent_idx))

    return sub_regions
