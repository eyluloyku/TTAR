"""Tail-10% deep dive: what's wrong with the hardest images, and what kind
of correction could actually help.

Given the GT-oracle run's per-image deltas + the underlying D0 and GT arrays, we
characterise the tail-10 subset along the axes that determine whether a
sparse-anchor corrector can ever bridge the gap:

  1. Where does the per-pixel siRMSE-aligned error live? (mean, P95,
     fraction of pixels above thresholds — concentrated or diffuse?)
  2. Are errors *spatially clustered* into a few large regions, or
     scattered across the image? (connected-component count + size
     distribution above an error threshold.)
  3. How much of the achievable headroom did K=100 oracle anchors use?
     (gap = baseline_siRMSE - best_refined_siRMSE; ceiling = baseline,
     since siRMSE→0 if we replace D0 with GT everywhere.)
  4. Per-image: which corrector cell helped most? Are the tail-10 images
     uniformly responsive to one config, or do they split by sub-type?

Outputs:
  tail10_per_image.csv         — one row per tail-10 image, all features.
  tail10_summary.md            — aggregate verdicts + strategy candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EPS = 1e-6


def per_image_error(D0: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    log_d0 = np.log(np.maximum(D0, EPS))
    log_gt = np.log(np.maximum(gt, EPS))
    d = log_d0 - log_gt
    mu = d[valid].mean()
    return np.abs(d - mu)


def cluster_stats(error_map: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    """Connected-component analysis of high-error pixels.

    A tail-10 image whose error lives in 1-2 large blobs is *correctable* in
    principle by sparse anchors landing in those blobs. One whose error is
    scattered across many small blobs is not — sparse correction can never
    reach all of them.
    """
    high = (error_map > threshold) & valid
    if not high.any():
        return {"n_cc": 0, "largest_cc_frac": 0.0, "top3_cc_frac": 0.0,
                "frac_high": 0.0}
    labeled, n = ndimage.label(high)
    sizes = np.bincount(labeled.ravel())[1:]  # skip background
    sizes_sorted = np.sort(sizes)[::-1]
    total = sizes.sum()
    return {
        "n_cc": int(n),
        "largest_cc_frac": float(sizes_sorted[0] / total) if total > 0 else 0.0,
        "top3_cc_frac": float(sizes_sorted[:3].sum() / total) if total > 0 else 0.0,
        "frac_high": float(total / valid.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", required=True,
                   help="GT-oracle results.json with per-image s0 and per-config s_ref/delta.")
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--gt-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--error-thresholds", type=float, nargs="+",
                   default=[0.5, 1.0, 1.5],
                   help="Log-error thresholds for high-error CC analysis.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = Path(args.preds_dir)
    gt_root = Path(args.gt_root)

    raw = json.loads(Path(args.eval_json).read_text())
    per_image = raw["per_image"]
    if not per_image:
        raise SystemExit("Empty per_image.")

    cfg_keys = sorted(per_image[0]["results"].keys())

    rows = []
    for rec in per_image:
        rows.append({
            "stem": rec["stem"],
            "s0": float(rec["s0"]),
            **{k: float(rec["results"][k]["s_ref"])
               if np.isfinite(rec["results"][k]["s_ref"]) else float("nan")
               for k in cfg_keys},
        })
    rows.sort(key=lambda r: -r["s0"])
    n = len(rows)
    tail_n = max(1, int(np.ceil(n * 0.10)))
    tail_rows = rows[:tail_n]
    rest_rows = rows[tail_n:]

    # Tail vs rest baseline summary.
    tail_s0 = np.array([r["s0"] for r in tail_rows])
    rest_s0 = np.array([r["s0"] for r in rest_rows])
    print(f"[tail10] N tail={len(tail_rows)}  N rest={len(rest_rows)}")
    print(f"[tail10] baseline siRMSE  tail mean={tail_s0.mean():.4f}  "
          f"rest mean={rest_s0.mean():.4f}")

    # Per-image error/CC analysis — only on tail-10 since rest is the larger pool.
    enriched = []
    for r in tail_rows:
        stem = r["stem"]
        D0_path = preds_dir / f"{stem}.npy"
        gt_path = gt_root / f"{stem}_depth.npy"
        if not D0_path.exists() or not gt_path.exists():
            continue
        D0 = np.load(D0_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        if D0.shape != gt.shape:
            continue
        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        if valid.sum() < 1000:
            continue
        err = per_image_error(D0, gt, valid)
        err_v = err[valid]

        row = dict(r)
        row.update({
            "n_valid": int(valid.sum()),
            "err_mean": float(err_v.mean()),
            "err_p50":  float(np.percentile(err_v, 50)),
            "err_p90":  float(np.percentile(err_v, 90)),
            "err_p95":  float(np.percentile(err_v, 95)),
            "err_p99":  float(np.percentile(err_v, 99)),
        })
        for thr in args.error_thresholds:
            cs = cluster_stats(err, valid, thr)
            row[f"frac_high_{thr}"] = cs["frac_high"]
            row[f"n_cc_{thr}"] = cs["n_cc"]
            row[f"largest_cc_frac_{thr}"] = cs["largest_cc_frac"]
            row[f"top3_cc_frac_{thr}"] = cs["top3_cc_frac"]

        # Best refined cell and the implied gap to the ceiling (siRMSE=0).
        ref_vals = [(k, row[k]) for k in cfg_keys if np.isfinite(row[k])]
        if ref_vals:
            best_k, best_s = min(ref_vals, key=lambda kv: kv[1])
            row["best_cfg"] = best_k
            row["best_s_ref"] = float(best_s)
            row["best_delta"] = float(best_s - row["s0"])
            row["gap_to_ceiling"] = float(best_s)  # ceiling is 0
            row["frac_headroom_used"] = (
                float((row["s0"] - best_s) / row["s0"]) if row["s0"] > 0 else 0.0
            )
        enriched.append(row)

    if not enriched:
        raise SystemExit("Could not load any tail-10 image.")

    # ----- Aggregate verdicts -----
    err_mean = np.array([r["err_mean"] for r in enriched])
    err_p95  = np.array([r["err_p95"] for r in enriched])
    err_p99  = np.array([r["err_p99"] for r in enriched])
    frac_05 = np.array([r["frac_high_0.5"] for r in enriched])
    frac_10 = np.array([r["frac_high_1.0"] for r in enriched])
    n_cc_05 = np.array([r["n_cc_0.5"] for r in enriched])
    top3_05 = np.array([r["top3_cc_frac_0.5"] for r in enriched])
    headroom_used = np.array([r["frac_headroom_used"] for r in enriched])

    # CSV
    import csv
    with (out_dir / "tail10_per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        w.writeheader()
        for r in enriched:
            w.writerow(r)

    # Best-cfg distribution
    from collections import Counter
    best_cfg_dist = Counter(r["best_cfg"] for r in enriched)

    # Summary md
    lines = ["# Tail-10% deep dive\n"]
    lines.append(f"**N tail-10:** {len(enriched)}  ")
    lines.append(f"**Baseline siRMSE:** mean={tail_s0.mean():.4f}  "
                 f"median={np.median(tail_s0):.4f}  "
                 f"max={tail_s0.max():.4f}\n")
    lines.append("## Per-pixel error structure (tail-10 only)\n")
    lines.append("Higher = harder. P95 ≈ "
                 "the 5%-of-pixels at which the depth ratio is exp(P95) off.\n")
    lines.append("| Stat | mean | median | p25 | p75 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, arr in [("err_mean", err_mean), ("err_p95", err_p95),
                        ("err_p99", err_p99),
                        ("frac_pixels |log err|>0.5", frac_05),
                        ("frac_pixels |log err|>1.0", frac_10),
                        ("# high-error CC (>0.5)", n_cc_05),
                        ("top-3 CC frac of high-error pixels", top3_05),
                        ("frac of headroom captured by K=100 oracle",
                         headroom_used),
                       ]:
        lines.append(f"| {label} | {arr.mean():.3f} | {np.median(arr):.3f} | "
                     f"{np.percentile(arr,25):.3f} | "
                     f"{np.percentile(arr,75):.3f} |")
    lines.append("")

    lines.append("## Best-cell distribution (per image)\n")
    lines.append("Which (config × K) cell achieved the lowest refined siRMSE "
                 "for each tail-10 image?\n")
    lines.append("| Cell | # images | % of tail-10 |")
    lines.append("|---|---:|---:|")
    for cfg, count in best_cfg_dist.most_common():
        lines.append(f"| `{cfg}` | {count} | {count/len(enriched)*100:.1f}% |")
    lines.append("")

    # Diagnostic + recommendation
    lines.append("## Diagnostic\n")
    f10_med = float(np.median(frac_10))
    cc_med = float(np.median(n_cc_05))
    top3_med = float(np.median(top3_05))
    headroom_med = float(np.median(headroom_used))
    if f10_med > 0.10:
        kind = "**many** pixels are catastrophically wrong"
    elif f10_med > 0.03:
        kind = "moderate fraction of pixels carry the error"
    else:
        kind = "errors concentrated in a small fraction of pixels"
    if top3_med > 0.6:
        spatial = "errors are **clustered** — top-3 connected components cover most high-error mass"
    elif top3_med > 0.3:
        spatial = "errors are partly clustered"
    else:
        spatial = "errors are **scattered** across many small regions"
    lines.append(f"- Pixel-level: {kind} (median frac >|log err|>1.0 = {f10_med*100:.1f}%).")
    lines.append(f"- Spatial: {spatial} (median top-3 CC frac = {top3_med:.2f}, "
                 f"median # CC = {cc_med:.0f}).")
    lines.append(f"- Headroom: K=100 oracle captures **{headroom_med*100:.1f}%** of the "
                 f"absolute siRMSE budget (median). Remaining gap to siRMSE=0 = "
                 f"baseline × {1-headroom_med:.2f}.")

    lines.append("\n## Strategy implications\n")
    if top3_med > 0.5 and headroom_med < 0.05:
        lines.append("- Errors are **clustered** but only a small fraction of headroom is captured. "
                     "Either: (a) anchors are not landing in the bad clusters, or "
                     "(b) the smoothness term is not propagating far enough. "
                     "**Action**: try heat-anchor selection guided by the *actual* error map "
                     "(sanity check) — if that fixes it, the issue is anchor placement; "
                     "if it doesn't, the smoothness propagator is too local.")
    if cc_med >= 10 and top3_med < 0.5:
        lines.append("- Many scattered components: sparse anchors fundamentally cannot reach all of them. "
                     "**Action**: this subset is *uncorrectable* under the sparse-anchor framework; "
                     "report as a finding.")
    if headroom_med > 0.3:
        lines.append("- High headroom captured already — close to the K=100 oracle limit. "
                     "**Action**: scale K. Run a K=200/500/1000 oracle sweep.")
    lines.append("")

    (out_dir / "tail10_summary.md").write_text("\n".join(lines))

    print(f"\n[tail10] wrote {out_dir/'tail10_per_image.csv'}")
    print(f"[tail10] wrote {out_dir/'tail10_summary.md'}")
    print(f"[tail10] tail-10 best-cfg distribution: {dict(best_cfg_dist.most_common())}")
    print(f"[tail10] median frac headroom captured: {headroom_med*100:.1f}%")


if __name__ == "__main__":
    main()
