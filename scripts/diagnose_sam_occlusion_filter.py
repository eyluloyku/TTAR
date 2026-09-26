"""Stage B v5 — Phase 0b follow-up: are SAM's *occlusion* boundaries well placed?

Q0.2 found raw SAM boundaries sit a median 6.5 px from true depth
discontinuities — but precision is only 29%, i.e. most SAM boundaries are
co-planar (texture / object parts). The planned occlusion classifier removes
exactly those. This de-risking script asks: if we keep only the SAM boundaries
with a D0 depth-step across them (a geometric occlusion proxy), are *those*
well localized?

For each cached SAM partition, every region gets a median log-D0; a boundary
pixel is an 'occlusion candidate' if the two regions it separates differ in
median log-D0 by more than tau. recall / precision / displacement vs GT depth
discontinuities are recomputed on the filtered boundary, swept over tau.

Decisive:
  - occlusion-filtered displacement -> ~1-2 px  => SAM's real depth boundaries
    ARE well placed; the v5 SAM design survives (recall is the remaining gap).
  - displacement stays ~5-6 px  => SAM is the wrong localizer; pivot the guide.

Pure CPU; re-aggregates the partitions cached by diagnose_sam_boundary_quality.

Usage:
    python scripts/diagnose_sam_occlusion_filter.py \
        --preds-dir outputs/val_preds/vits_lora_r8/ \
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \
        --cache-dir outputs/stage_b_cache/sam_partitions/ \
        --out-dir outputs/diagnostics/boundary_phase0b_occ/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_sam_boundary_quality import (  # noqa: E402
    DISTS, EPS, edge_set, grad_mag_masked, load_rgb, mask_border,
    partition_boundary, pr_metrics, si_rmse, stats,
)


def region_medians(log_d0: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Median log-D0 per region label (index = label id)."""
    n = int(lab.max()) + 1
    med = np.zeros(n, np.float32)
    for r in range(n):
        m = lab == r
        if m.any():
            med[r] = float(np.median(log_d0[m]))
    return med


def cross_region_gap(med_map: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Per-pixel max |Δ region-median log-D0| over 4-neighbours in other regions.

    Large gap ⇒ the boundary here separates two depth-different regions ⇒ a
    geometric occlusion-boundary candidate.
    """
    gap = np.zeros(lab.shape, np.float32)
    dh = np.abs(med_map[:, 1:] - med_map[:, :-1])
    diffh = lab[:, 1:] != lab[:, :-1]
    gh = np.where(diffh, dh, 0.0).astype(np.float32)
    gap[:, :-1] = np.maximum(gap[:, :-1], gh)
    gap[:, 1:] = np.maximum(gap[:, 1:], gh)
    dv = np.abs(med_map[1:, :] - med_map[:-1, :])
    diffv = lab[1:, :] != lab[:-1, :]
    gv = np.where(diffv, dv, 0.0).astype(np.float32)
    gap[:-1, :] = np.maximum(gap[:-1, :], gv)
    gap[1:, :] = np.maximum(gap[1:, :], gv)
    return gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--cache-dir", required=True,
                   help="SAM partitions cached by diagnose_sam_boundary_quality.py.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--edge-pct", type=float, default=88.0)
    p.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.2, 0.4],
                   help="Occlusion thresholds on |Δ region-median log-D0|.")
    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cache_files = sorted(cache_dir.glob("*.npy"))
    if not cache_files:
        raise SystemExit(f"No cached partitions in {cache_dir}")

    variants = ["full"] + [f"occ@{t}" for t in args.taus]
    records: list[dict] = []
    skipped = 0
    t0 = time.time()

    for i, lab_path in enumerate(cache_files):
        stem = lab_path.stem
        pred_path = preds_dir / f"{stem}.npy"
        gt_path = train_root / f"{stem}_depth.npy"
        rgb_path = train_root / f"{stem}_rgb.png"
        if not (pred_path.exists() and gt_path.exists() and rgb_path.exists()):
            skipped += 1
            continue

        D0 = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        lab = np.load(lab_path).astype(np.int32)
        if D0.shape != gt.shape or lab.shape != D0.shape:
            skipped += 1
            continue
        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        if valid.sum() < 1000:
            skipped += 1
            continue

        log_d0 = np.log(np.maximum(D0, EPS)).astype(np.float32)
        log_gt = np.log(np.maximum(gt, EPS)).astype(np.float32)
        gt_mag, gt_def = grad_mag_masked(log_gt, valid)
        gt_disc = edge_set(gt_mag, gt_def, args.edge_pct) | mask_border(valid)

        sam_b = partition_boundary(lab)
        med = region_medians(log_d0, lab)
        gap = cross_region_gap(med[lab], lab)

        rec = {"stem": stem, "n_sam_boundary": int(sam_b.sum()), "metrics": {}}
        rec["metrics"]["full"] = pr_metrics(sam_b, gt_disc)
        rec["metrics"]["full"]["survive"] = 1.0
        for t in args.taus:
            occ = sam_b & (gap > t)
            m = pr_metrics(occ, gt_disc)
            m["survive"] = float(occ.sum()) / max(int(sam_b.sum()), 1)
            rec["metrics"][f"occ@{t}"] = m
        records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cache_files)}  kept={len(records)}  "
                  f"({(i+1)/(time.time()-t0):.1f} img/s)", flush=True)

    if not records:
        raise SystemExit("No images evaluated.")

    # ---- aggregate ------------------------------------------------------
    agg = {"n_images": len(records), "skipped": skipped, "by_variant": {}}
    for v in variants:
        agg["by_variant"][v] = {
            "displacement": stats([r["metrics"][v]["displacement"] for r in records]),
            "survive": stats([r["metrics"][v]["survive"] for r in records]),
            "precision": {str(d): stats([r["metrics"][v]["precision"][str(d)]
                                         for r in records]) for d in DISTS},
            "recall": {str(d): stats([r["metrics"][v]["recall"][str(d)]
                                      for r in records]) for d in DISTS},
        }
    (out_dir / "results.json").write_text(
        json.dumps({"aggregate": agg, "per_image": records}, indent=2))

    # ---- summary.md -----------------------------------------------------
    L = ["# Stage B v5 — Phase 0b follow-up: SAM occlusion-boundary quality\n"]
    L.append(f"**N images:** {agg['n_images']}  (skipped {skipped})  ")
    L.append(f"**Occlusion filter:** keep SAM boundary pixels where the two "
             f"regions differ in median log-D0 by > tau.\n")
    L.append("`survive` = fraction of SAM boundary kept. `recall@2` = depth "
             "edges still covered. `displacement` = median px to nearest GT "
             "discontinuity (the localization quality that matters).\n")

    L.append("| Variant | survive | recall@2 | precision@2 | displacement |")
    L.append("|---|---:|---:|---:|---:|")
    for v in variants:
        e = agg["by_variant"][v]
        L.append(f"| `{v}` | {e['survive']['mean']*100:.0f}% | "
                 f"{e['recall']['2']['mean']*100:.1f}% | "
                 f"{e['precision']['2']['mean']*100:.1f}% | "
                 f"{e['displacement']['mean']:.2f} |")

    L.append("\n## Recall / precision at d ∈ {1,2,4}\n")
    L.append("| Variant | rec@1 | rec@2 | rec@4 | prec@1 | prec@2 | prec@4 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for v in variants:
        e = agg["by_variant"][v]
        L.append(f"| `{v}` | "
                 + " | ".join(f"{e['recall'][str(d)]['mean']*100:.1f}%" for d in DISTS)
                 + " | "
                 + " | ".join(f"{e['precision'][str(d)]['mean']*100:.1f}%" for d in DISTS)
                 + " |")

    # ---- verdict --------------------------------------------------------
    best_t = min(args.taus, key=lambda t: agg["by_variant"][f"occ@{t}"]["displacement"]["mean"])
    occ_disp = agg["by_variant"][f"occ@{best_t}"]["displacement"]["mean"]
    occ_rec = agg["by_variant"][f"occ@{best_t}"]["recall"]["2"]["mean"]
    full_disp = agg["by_variant"]["full"]["displacement"]["mean"]
    L.append("\n## Verdict\n")
    L.append(f"- Best occlusion filter: tau={best_t}. Displacement "
             f"**{occ_disp:.2f} px** (raw SAM: {full_disp:.2f} px; "
             f"D0 edges from Q0.2: 1.36 px).")
    L.append(f"- Recall@2 of the occlusion-filtered boundary: {occ_rec*100:.1f}%.")
    if occ_disp < 2.5:
        L.append("- **SAM occlusion boundaries ARE well localized** — the v5 SAM "
                 "design survives; recall/coverage is the remaining gap "
                 "(→ option B: improve the segmenter).")
    else:
        L.append("- **SAM occlusion boundaries are still poorly localized** — "
                 "the displacement is not a co-planar-contamination artefact; "
                 "SAM is the wrong localizer (→ option C/D: pivot the guide).")
    (out_dir / "summary.md").write_text("\n".join(L))

    print(f"\n[occ] N={agg['n_images']}  best tau={best_t}")
    print(f"[occ] displacement: raw SAM={full_disp:.2f}px  "
          f"occ-filtered={occ_disp:.2f}px  (D0 edges 1.36px)")
    print(f"[occ] occ-filtered recall@2={occ_rec*100:.1f}%")
    print(f"[occ] wrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
