"""Stage B v5 — SAM boundary quality.

Phase 0a showed D0's moderate error is concentrated at depth discontinuities
and the relocalization ceiling is -0.045. Q0.2 asks whether SAM's region
partition is well enough localized to *realize* that ceiling.

Runs SAM on val (via the existing RegionExtractor), caches the disjoint region
partition as an int label map, and measures three edge maps against true GT
depth discontinuities:

  recall@d     frac of GT-discontinuity pixels within d px of the edge map.
               Under-segmentation check — does SAM cover the depth edges at all?
  precision@d  frac of edge-map pixels within d px of a GT discontinuity.
               SAM precision is *expected low* — many SAM boundaries are
               co-planar (texture / attached objects); separating those is the
               occlusion classifier's job, not SAM's.
  displacement median distance from edge-map pixels to the nearest GT
               discontinuity. The localization quality that matters.

Compared head-to-head: `sam` (region partition), `d0` (|grad log D0|), `rgb`
(grayscale gradient). v5 needs SAM recall clearly above D0's and SAM
displacement clearly below D0's.

Usage:
    python scripts/diagnose_sam_boundary_quality.py \
        --preds-dir outputs/val_preds/vits_lora_r8/ \
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \
        --sam-checkpoint checkpoints/sam_vit_b.pth \
        --cache-dir outputs/stage_b_cache/sam_partitions/ \
        --out-dir outputs/diagnostics/boundary_phase0b/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import load_rgb  # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402
from src.refine.segment import RegionExtractor  # noqa: E402

EPS = 1e-6
DISTS = [1, 2, 4]   # px tolerances for precision/recall


def stats(arr) -> dict:
    arr = np.asarray(arr, np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {"n": int(len(arr)), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75))}


# --- edge maps (same definitions as diagnose_boundary_errormode.py) --------

def grad_mag_masked(field, valid):
    H, W = field.shape
    mag = np.zeros((H, W), np.float32)
    defined = np.zeros((H, W), bool)
    dh = np.abs(field[:, 1:] - field[:, :-1]); vh = valid[:, 1:] & valid[:, :-1]
    dhv = np.where(vh, dh, 0.0).astype(np.float32)
    mag[:, :-1] = np.maximum(mag[:, :-1], dhv); defined[:, :-1] |= vh
    mag[:, 1:] = np.maximum(mag[:, 1:], dhv);   defined[:, 1:] |= vh
    dv = np.abs(field[1:, :] - field[:-1, :]); vv = valid[1:, :] & valid[:-1, :]
    dvv = np.where(vv, dv, 0.0).astype(np.float32)
    mag[:-1, :] = np.maximum(mag[:-1, :], dvv); defined[:-1, :] |= vv
    mag[1:, :] = np.maximum(mag[1:, :], dvv);   defined[1:, :] |= vv
    return mag, defined


def grad_mag_dense(field):
    gy, gx = np.gradient(field.astype(np.float32))
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def mask_border(valid):
    inv = ~valid
    nb = np.zeros_like(valid)
    nb[:, :-1] |= inv[:, 1:]; nb[:, 1:] |= inv[:, :-1]
    nb[:-1, :] |= inv[1:, :]; nb[1:, :] |= inv[:-1, :]
    return valid & nb


def edge_set(mag, defined, pct):
    vals = mag[defined]
    if vals.size == 0:
        return np.zeros_like(defined)
    return defined & (mag >= float(np.percentile(vals, pct)))


def partition_boundary(lab):
    """Pixels where the region label differs from a 4-neighbour."""
    b = np.zeros(lab.shape, bool)
    dh = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= dh; b[:, 1:] |= dh
    dv = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= dv; b[1:, :] |= dv
    return b


def label_map(regions, shape):
    """Disjoint int label map from a RegionExtractor region list (0 = unclaimed)."""
    lab = np.zeros(shape, np.int32)
    for i, r in enumerate(regions):
        lab[r.mask] = i + 1
    return lab


def pr_metrics(edge, gt_disc):
    """precision@d, recall@d, median displacement of `edge` vs `gt_disc`."""
    out = {"precision": {}, "recall": {}, "displacement": float("nan")}
    if not edge.any() or not gt_disc.any():
        for d in DISTS:
            out["precision"][str(d)] = float("nan")
            out["recall"][str(d)] = float("nan")
        return out
    d_to_gt = distance_transform_edt(~gt_disc)     # dist from any pixel to GT disc
    d_to_edge = distance_transform_edt(~edge)      # dist from any pixel to edge
    pe = d_to_gt[edge]        # edge pixel -> nearest GT disc
    rg = d_to_edge[gt_disc]   # GT disc pixel -> nearest edge
    for d in DISTS:
        out["precision"][str(d)] = float((pe <= d).mean())
        out["recall"][str(d)] = float((rg <= d).mean())
    out["displacement"] = float(np.median(pe))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", required=True,
                   help="Where to save SAM partition label maps (.npy) for reuse "
                        "in Phases 1-2.")
    p.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_b.pth")
    p.add_argument("--sam-model-type", default="vit_b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stems-file", default=None)
    p.add_argument("--edge-pct", type=float, default=88.0)
    # The boundary diagnostic wants a fine partition covering every depth edge;
    # the occlusion classifier removes co-planar boundaries later. Two deliberate
    # departures from the SAM RegionExtractor defaults:
    #   max_area_frac=1.0 -> disable PCA subdivision. Subdividing a large region
    #     (e.g. a wall) injects ARTIFICIAL internal boundaries that are not
    #     depth discontinuities — they would corrupt the precision metric here
    #     and cause spurious depth jumps in the Phase 1/2 solver.
    #   max_regions=256 -> keep essentially all regions. A low cap (e.g. 12)
    #     drops the smallest regions, i.e. exactly the small foreground objects
    #     whose silhouettes the boundary method targets.
    p.add_argument("--max-regions", type=int, default=256)
    p.add_argument("--min-area-frac", type=float, default=0.001)
    p.add_argument("--max-area-frac", type=float, default=1.0,
                   help="Regions above this area fraction are PCA-subdivided. "
                        "1.0 = no subdivision (recommended for v5).")
    # SAM cost knobs. The RegionExtractor v1 default (points_per_side=64) is
    # ~44 s/image on a 5060 Ti — untenable for full val. 32 (SAM's own default
    # density) is ~4x cheaper; Q0.2 measures whether it still has the recall v5
    # needs. If not, that is itself the finding (-> finer SAM / SAM2).
    p.add_argument("--points-per-side", type=int, default=32)
    p.add_argument("--crop-n-layers", type=int, default=1)
    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(preds_dir.glob("*.npy"))
    if args.stems_file:
        wanted = {ln.strip() for ln in Path(args.stems_file).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")}
        pred_files = [f for f in pred_files if f.stem in wanted]
    if args.limit:
        pred_files = pred_files[: args.limit]

    extractor = RegionExtractor(
        sam_checkpoint=args.sam_checkpoint, model_type=args.sam_model_type,
        min_area_frac=args.min_area_frac, max_area_frac=args.max_area_frac,
        max_regions=args.max_regions,
        points_per_side=args.points_per_side, crop_n_layers=args.crop_n_layers,
        device=args.device,
    )

    edge_types = ["sam", "d0", "rgb"]
    records: list[dict] = []
    skipped = 0
    t0 = time.time()

    for i, pred_path in enumerate(pred_files):
        stem = pred_path.stem
        gt_path = train_root / f"{stem}_depth.npy"
        rgb_path = train_root / f"{stem}_rgb.png"
        if not gt_path.exists() or not rgb_path.exists():
            skipped += 1
            continue

        D0 = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        rgb = load_rgb(rgb_path)
        if D0.shape != gt.shape or rgb.shape[:2] != D0.shape:
            skipped += 1
            continue
        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        if valid.sum() < 1000:
            skipped += 1
            continue

        # --- SAM partition (cached) -------------------------------------
        regions = extractor.extract(rgb)
        lab = label_map(regions, D0.shape)
        np.save(cache_dir / f"{stem}.npy", lab)
        # coverage: fraction of valid pixels claimed by some region (label > 0).
        # Unclaimed pixels become label 0 and must be handled in Phase 1.
        coverage = float(((lab > 0) & valid).sum()) / max(int(valid.sum()), 1)

        # --- edge maps --------------------------------------------------
        log_d0 = np.log(np.maximum(D0, EPS)).astype(np.float32)
        log_gt = np.log(np.maximum(gt, EPS)).astype(np.float32)
        gt_mag, gt_def = grad_mag_masked(log_gt, valid)
        gt_disc = edge_set(gt_mag, gt_def, args.edge_pct) | mask_border(valid)
        d0_mag, d0_def = grad_mag_masked(log_d0, valid)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]).astype(np.float32)
        edges = {
            "sam": partition_boundary(lab),
            "d0": edge_set(d0_mag, d0_def, args.edge_pct),
            "rgb": edge_set(grad_mag_dense(gray), np.ones_like(valid), args.edge_pct),
        }

        rec = {"stem": stem, "s0": si_rmse(D0, gt), "n_regions": len(regions),
               "coverage": coverage,
               "metrics": {et: pr_metrics(edges[et], gt_disc) for et in edge_types}}
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pred_files)}  kept={len(records)}  "
                  f"({(i+1)/(time.time()-t0):.2f} img/s)", flush=True)

    if not records:
        raise SystemExit("No images evaluated.")

    # ---- aggregate ------------------------------------------------------
    agg = {"n_images": len(records), "skipped": skipped,
           "n_regions": stats([r["n_regions"] for r in records]),
           "coverage": stats([r["coverage"] for r in records]),
           "by_edge": {}}
    for et in edge_types:
        agg["by_edge"][et] = {
            "displacement": stats([r["metrics"][et]["displacement"] for r in records]),
            "precision": {str(d): stats([r["metrics"][et]["precision"][str(d)]
                                         for r in records]) for d in DISTS},
            "recall": {str(d): stats([r["metrics"][et]["recall"][str(d)]
                                      for r in records]) for d in DISTS},
        }
    (out_dir / "results.json").write_text(
        json.dumps({"aggregate": agg, "per_image": records}, indent=2))

    # ---- summary.md -----------------------------------------------------
    L = ["# Stage B v5 — Phase 0b (Q0.2): SAM boundary quality\n"]
    L.append(f"**N images:** {agg['n_images']}  (skipped {skipped})  ")
    L.append(f"**SAM regions/image:** mean={agg['n_regions']['mean']:.1f}  "
             f"median={agg['n_regions']['median']:.0f}  ")
    L.append(f"**Partition coverage of valid pixels:** "
             f"mean={agg['coverage']['mean']*100:.1f}%  median="
             f"{agg['coverage']['median']*100:.1f}%  "
             f"(unclaimed pixels = label 0 — handled in Phase 1)  ")
    L.append(f"**SAM:** {args.sam_model_type}, "
             f"points_per_side={args.points_per_side}, "
             f"crop_n_layers={args.crop_n_layers}, "
             f"max_regions={args.max_regions}, "
             f"min_area_frac={args.min_area_frac}, "
             f"max_area_frac={args.max_area_frac} (subdivision "
             f"{'off' if args.max_area_frac >= 1.0 else 'on'})\n")
    L.append("Edge maps vs true GT depth discontinuities. `recall` = depth edges "
             "covered by the map (under-segmentation check). `displacement` = "
             "median px from map to nearest GT discontinuity.\n")

    L.append("## Recall@d — fraction of GT discontinuities covered\n")
    L.append("| Edge | d=1 | d=2 | d=4 |")
    L.append("|---|---:|---:|---:|")
    for et in edge_types:
        r = agg["by_edge"][et]["recall"]
        L.append(f"| `{et}` | " + " | ".join(f"{r[str(d)]['mean']*100:.1f}%"
                                             for d in DISTS) + " |")

    L.append("\n## Precision@d — fraction of edge pixels on a real depth edge\n")
    L.append("(SAM precision is expected low — co-planar boundaries are handled "
             "later by the occlusion classifier.)\n")
    L.append("| Edge | d=1 | d=2 | d=4 |")
    L.append("|---|---:|---:|---:|")
    for et in edge_types:
        pr = agg["by_edge"][et]["precision"]
        L.append(f"| `{et}` | " + " | ".join(f"{pr[str(d)]['mean']*100:.1f}%"
                                             for d in DISTS) + " |")

    L.append("\n## Displacement — median px to nearest GT discontinuity\n")
    L.append("| Edge | displacement |")
    L.append("|---|---:|")
    for et in edge_types:
        L.append(f"| `{et}` | {agg['by_edge'][et]['displacement']['mean']:.2f} |")

    # ---- verdict --------------------------------------------------------
    sam_rec = agg["by_edge"]["sam"]["recall"]["2"]["mean"]
    d0_rec = agg["by_edge"]["d0"]["recall"]["2"]["mean"]
    sam_disp = agg["by_edge"]["sam"]["displacement"]["mean"]
    d0_disp = agg["by_edge"]["d0"]["displacement"]["mean"]
    L.append("\n## Verdict\n")
    L.append(f"- SAM recall@2 of GT discontinuities: **{sam_rec*100:.1f}%** "
             f"(D0 edges: {d0_rec*100:.1f}%).")
    L.append(f"- SAM boundary displacement: **{sam_disp:.2f} px** "
             f"(D0 edges: {d0_disp:.2f} px).")
    L.append(f"- Partition coverage of valid pixels: "
             f"**{agg['coverage']['mean']*100:.1f}%** "
             f"({agg['n_regions']['mean']:.0f} regions/image).")
    q02 = "PASS" if (sam_rec > 0.70 and sam_disp < d0_disp) else "REVIEW"
    L.append(f"- **Q0.2 (SAM recall@2 > 70% and displacement < D0's): {q02}**")
    if q02 != "PASS":
        L.append("- If REVIEW: SAM's partition is the bottleneck — upgrade the "
                 "segmenter (SAM2 / vit-h) or loosen region thresholds before "
                 "Phase 1.")
    (out_dir / "summary.md").write_text("\n".join(L))

    print(f"\n[q0.2] N={agg['n_images']}  SAM regions/img={agg['n_regions']['mean']:.1f}"
          f"  coverage={agg['coverage']['mean']*100:.1f}%")
    print(f"[q0.2] SAM recall@2={sam_rec*100:.1f}% (D0 {d0_rec*100:.1f}%)  "
          f"disp SAM={sam_disp:.2f}px D0={d0_disp:.2f}px  -> {q02}")
    print(f"[q0.2] partitions cached to {cache_dir}")
    print(f"[q0.2] wrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
