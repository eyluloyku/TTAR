"""Stage B refinement pipeline — Refiner interface + identity baseline.

A ``Refiner`` takes a fine-tuned Stage A depth prediction ``D0`` (and the RGB
image it came from) and returns a refined depth map of the same shape. This is
the whole-image refiner interface used by ``scripts/eval_refine.py`` (identity
and the SAM-region smoothing control).

The sparse-anchor corrector (TTAR) does not go through this protocol: it takes
explicit anchor coordinates, so it is driven directly by
``scripts/eval_anchor_sources.py`` over ``src.refine.correct_dense``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .region_smooth import SAMRegionSmoothRefiner


class Refiner(Protocol):
    """Maps (RGB, initial depth) to a refined depth map.

    Implementations must:
    - return a float32 array with the same shape as ``D0``
    - never reorder or resize pixels (the eval harness compares pixel-wise)
    - be deterministic given the same inputs (for reproducible ablations)
    """

    name: str

    def __call__(self, rgb: np.ndarray, D0: np.ndarray) -> np.ndarray: ...


class IdentityRefiner:
    """Returns ``D0`` unchanged. Sanity-checks the eval harness."""

    name = "identity"

    def __call__(self, rgb: np.ndarray, D0: np.ndarray) -> np.ndarray:
        return D0.astype(np.float32, copy=False)


REFINERS: dict[str, type] = {
    "identity": IdentityRefiner,
    "region_smooth": SAMRegionSmoothRefiner,
}


def build_refiner(name: str, **kwargs) -> Refiner:
    if name not in REFINERS:
        raise KeyError(f"Unknown refiner '{name}'. Available: {sorted(REFINERS)}")
    return REFINERS[name](**kwargs)
