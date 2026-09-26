"""Stage B — Test-Time Anchor Refinement (TTAR).

Correct a frozen Stage A depth map ``D0`` at test time by propagating a small
set of high-quality depth *anchors* through a confidence-modulated, siRMSE-aware
dense solver. The anchor source is pluggable (GT oracle / second depth model /
depth-from-language VLM) behind one ``AnchorBackend`` protocol.

Public API:

    from src.refine import (
        Refiner, IdentityRefiner, REFINERS, build_refiner,   # eval harness
        compute_heatmap, select_anchors_heatmap, select_anchors_random,  # anchors
        solve_dense_correction, smooth_only_correction,      # dense corrector
        AnchorBackend, OracleBackend, SecondModelBackend,  # anchor sources
        DepthLMBackend, make_backend,
        RegionExtractor, Region,                             # SAM (diagnostics)
        SAMRegionSmoothRefiner,                              # SAM-region refiner
        predict_tta, TTAOp,                                  # TTA uncertainty
        guided_filter,                                       # Stage A post-process
    )

Module map:
    heatmap.py        — error heat-map + anchor selection
    correct_dense.py  — confidence-modulated CG graph-Laplacian solver
    anchor_backends.py— anchor-source protocol + oracle/second-model/DepthLM
    pipeline.py       — Refiner Protocol + IdentityRefiner + REFINERS registry
    segment.py        — SAM region extractor (boundary diagnostics)
    region_smooth.py  — SAM-region depth-smoothing refiner
    tta.py            — test-time-augmentation prediction variance
    guided_filter.py  — RGB-guided post-process (used by Stage A predict)

See the project ``README.md`` for the method, the anchor-source study, and
reproduction commands.
"""

from .anchor_backends import (
    AnchorBackend,
    DepthLMBackend,
    OracleBackend,
    SecondModelBackend,
    make_backend,
)
from .correct_dense import smooth_only_correction, solve_dense_correction
from .guided_filter import guided_filter
from .heatmap import compute_heatmap, select_anchors_heatmap, select_anchors_random
from .pipeline import REFINERS, IdentityRefiner, Refiner, build_refiner
from .region_smooth import SAMRegionSmoothRefiner
from .segment import Region, RegionExtractor
from .tta import TTAOp, predict_tta

__all__ = [
    # eval harness
    "Refiner", "IdentityRefiner", "REFINERS", "build_refiner",
    # anchor selection
    "compute_heatmap", "select_anchors_heatmap", "select_anchors_random",
    # dense corrector
    "solve_dense_correction", "smooth_only_correction",
    # anchor sources
    "AnchorBackend", "OracleBackend", "SecondModelBackend",
    "DepthLMBackend", "make_backend",
    # SAM segmentation + region refiner
    "Region", "RegionExtractor", "SAMRegionSmoothRefiner",
    # TTA uncertainty
    "predict_tta", "TTAOp",
    # Stage A post-process
    "guided_filter",
]
