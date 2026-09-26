"""Robust VLM-anchored refinement, take 2.

What changed vs `eval_anchor_sources.py --backend vlm`:

  (1) Per-image scale alignment. Anchor delta becomes
          Δ_a = (log d_VLM(p) − bias) − log D0(p)
      where bias = median(log d_VLM(q) − log D0(q)) over all calibration
      anchors in this image. The diagnostic on N=25 val images showed
      this 3×s the mean error in anchor Δ vs the corrector's truth-Δ
      (1.60 → 0.56). Without this step the corrector treats DepthLM's
      per-image scale miscalibration (|bias| median ≈ 1.5 log units!) as
      structure and blows up siRMSE by +0.3 in the worst cases.

  (2) Outlier screening. After calibration, drop anchors where |Δ_a|
      exceeds a threshold (in log units). These are points where DepthLM
      and D0 are still wildly inconsistent — most likely DepthLM error,
      since D0 is already strongly correlated with GT (ρ ≈ 0.89 vs
      DepthLM's ρ ≈ 0.16 in the diagnostic).

  (5) Per-image selective gate. Compute ρ(log d_VLM, log D0) across
      calibration anchors as a free agreement signal. If ρ < τ_gate,
      DepthLM is anti-correlated or uninformative on this image — skip
      refinement and return D0 unchanged. By construction this caps the
      downside risk at "no worse than D0" on the gated-out subset.

Anchors are queried ONCE per image and reused for both calibration and
correction (no double VLM cost).

Output schema matches `eval_anchor_sources.py` so downstream analysis
scripts (`fit_selective_gate.py`, FINDINGS templates) work unchanged.

Usage:
  sbatch slurm/eval_robust_vlm_refine.sbatch \\
      --out outputs/diagnostics/vlm_robust_tail10/ \\
      --stems-file outputs/diagnostics/one_tail_stem.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import load_rgb  # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402
from src.refine.anchor_backends import DepthLMBackend  # noqa: E402
from src.refine.correct_dense import solve_dense_correction  # noqa: E402
from src.refine.heatmap import (  # noqa: E402
    compute_heatmap, select_anchors_random,
)

EPS = 1e-6


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    am, bm = a.mean(), b.mean()
    da, db = a - am, b - bm
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / denom) if denom > 0 else float("nan")


def mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be ~σ for Gaussian data."""
    m = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - m)))


def calibrate_and_screen(
    anchors_yx: list[tuple[int, int]],
    log_d_vlm: list[float],
    log_d0_arr: np.ndarray,
    *,
    outlier_max: float,
    use_mad_threshold: bool,
    min_calib_anchors: int,
) -> tuple[list[tuple[int, int]], list[float], dict]:
    """Per-image median-bias calibration + |Δ_a| outlier screening.

    Returns (anchors_kept, deltas_kept, info).
    """
    pairs = [
        (a, ld) for a, ld in zip(anchors_yx, log_d_vlm)
        if ld is not None and np.isfinite(ld)
    ]
    if len(pairs) < min_calib_anchors:
        return [], [], {"bias": float("nan"), "rho": float("nan"),
                        "sigma_after": float("nan"),
                        "n_valid_vlm": len(pairs), "n_dropped_outlier": 0}

    ld_vlm = np.array([ld for (_, ld) in pairs], dtype=np.float64)
    ld_d0 = np.array([float(log_d0_arr[y, x]) for ((y, x), _) in pairs],
                     dtype=np.float64)
    # Per-image scale alignment (β=1 fixed; see docstring).
    bias = float(np.median(ld_vlm - ld_d0))
    delta = (ld_vlm - bias) - ld_d0
    rho = pearson(ld_vlm, ld_d0)

    # Outlier screening threshold: max(outlier_max, K * MAD(delta)).
    sigma_post = mad(delta) if use_mad_threshold else float("nan")
    if use_mad_threshold and np.isfinite(sigma_post):
        # 2.5 MAD ≈ 2.5σ for Gaussian; combine with the hard cap.
        thr = min(outlier_max, max(2.5 * sigma_post, 0.15))
    else:
        thr = outlier_max
    keep_mask = np.abs(delta) < thr

    kept_anchors = [pairs[i][0] for i in range(len(pairs)) if keep_mask[i]]
    kept_deltas = [float(delta[i]) for i in range(len(pairs)) if keep_mask[i]]

    return kept_anchors, kept_deltas, {
        "bias": bias,
        "rho": rho,
        "sigma_after": sigma_post,
        "thr": thr,
        "n_valid_vlm": len(pairs),
        "n_dropped_outlier": int((~keep_mask).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stems-file", default=None)
    p.add_argument("--seed", type=int, default=42)

    # Anchor / corrector knobs (defaults inherited from the GT-oracle best cell, §7.1).
    p.add_argument("--K", type=int, default=100,
                   help="Total VLM queries per image (calibration ∪ refinement).")
    p.add_argument("--alpha-heat", type=float, default=1.0)
    p.add_argument("--beta-heat",  type=float, default=0.0)
    p.add_argument("--lam-s", type=float, default=0.1)
    p.add_argument("--alpha-smooth", type=float, default=2.0)
    p.add_argument("--cg-max-iter", type=int, default=200)
    p.add_argument("--cg-tol", type=float, default=1e-5)

    # Robustification knobs.
    p.add_argument("--outlier-max", type=float, default=0.6,
                   help="Hard cap on |Δ_a| (log units) after calibration.")
    p.add_argument("--use-mad-threshold", action="store_true", default=True,
                   help="Combine outlier_max with 2.5*MAD adaptive threshold.")
    p.add_argument("--no-mad-threshold", dest="use_mad_threshold",
                   action="store_false")
    p.add_argument("--min-anchors-after-screen", type=int, default=10,
                   help="If fewer survive screening, return D0 unchanged.")
    p.add_argument("--min-calib-anchors", type=int, default=20,
                   help="If fewer VLM queries succeed, return D0 unchanged.")

    # Per-image gate.
    p.add_argument("--tau-gate", type=float, default=0.0,
                   help="ρ(log d_VLM, log D0) threshold. Below = skip refinement.")
    p.add_argument("--disable-gate", action="store_true",
                   help="Bypass the per-image gate (apply refinement everywhere).")

    # VLM backend.
    p.add_argument("--vlm-model-id", default="facebook/DepthLM")
    p.add_argument("--vlm-batch-size", type=int, default=4)
    p.add_argument("--vlm-quant", choices=["nf4", "int8", "bf16"], default="nf4")

    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = DepthLMBackend(
        model_id=args.vlm_model_id,
        load_in_4bit=(args.vlm_quant == "nf4"),
        load_in_8bit=(args.vlm_quant == "int8"),
        batch_size=args.vlm_batch_size,
    )
    print(f"[robust] backend ready: {backend.name}  K={args.K}  "
          f"τ_gate={args.tau_gate}  outlier_max={args.outlier_max}")

    pred_files = sorted(preds_dir.glob("*.npy"))
    if args.stems_file:
        wanted = set(
            line.strip() for line in Path(args.stems_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        pred_files = [f for f in pred_files if f.stem in wanted]
    if args.limit:
        pred_files = pred_files[:args.limit]
    print(f"[robust] images to process: {len(pred_files)}")

    rng = np.random.default_rng(args.seed)
    per_image: list[dict] = []
    t0 = time.time()

    for i, pred_path in enumerate(pred_files):
        stem = pred_path.stem
        rgb_path = train_root / f"{stem}_rgb.png"
        gt_path = train_root / f"{stem}_depth.npy"
        if not rgb_path.exists() or not gt_path.exists():
            continue

        D0 = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        rgb = load_rgb(rgb_path)
        if D0.shape != gt.shape or rgb.shape[:2] != D0.shape:
            continue
        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        # Exclude border (DepthLM arrow doesn't fit there).
        Hh, Ww = D0.shape
        margin = 8
        border_ok = np.zeros_like(valid)
        border_ok[margin:Hh - margin, margin:Ww - margin] = True
        valid &= border_ok
        if valid.sum() < 1000:
            continue

        s0 = si_rmse(D0, gt)
        if not np.isfinite(s0):
            continue

        # Anchor pool: K random valid pixels. Used for both calibration
        # and refinement — no separate calibration query.
        anchors = select_anchors_random(args.K, D0.shape, valid=valid, rng=rng)
        log_d0 = np.log(np.maximum(D0, EPS)).astype(np.float64)

        # One VLM forward pass per image (batched internally).
        t_q0 = time.time()
        log_d_vlm, _ = backend.query_depth(rgb, anchors, stem=stem)
        t_q = time.time() - t_q0

        kept_anchors, kept_deltas, calib_info = calibrate_and_screen(
            anchors, log_d_vlm, log_d0,
            outlier_max=args.outlier_max,
            use_mad_threshold=args.use_mad_threshold,
            min_calib_anchors=args.min_calib_anchors,
        )

        # Default: refinement skipped (gate path).
        gate_open = (
            (args.disable_gate)
            or (np.isfinite(calib_info["rho"])
                and calib_info["rho"] >= args.tau_gate)
        )
        n_kept = len(kept_anchors)

        if not gate_open or n_kept < args.min_anchors_after_screen:
            D_ref = D0
            s_ref = s0
            mode = "gate_skip" if not gate_open else "too_few_anchors"
        else:
            H = compute_heatmap(
                D0, alpha=args.alpha_heat, beta=args.beta_heat, gamma=0.0,
            )
            D_ref, _ = solve_dense_correction(
                D0, H, kept_anchors, kept_deltas,
                anchor_weights=None,
                lam_s=args.lam_s,
                alpha_smooth=args.alpha_smooth,
                cg_max_iter=args.cg_max_iter,
                cg_tol=args.cg_tol,
            )
            s_ref = si_rmse(D_ref, gt)
            mode = "refined"

        per_image.append({
            "stem": stem,
            "s0": s0,
            "s_ref": s_ref,
            "delta": float(s_ref - s0),
            "mode": mode,
            "n_anchors_requested": args.K,
            "n_anchors_vlm_valid": calib_info["n_valid_vlm"],
            "n_anchors_kept": n_kept,
            "n_anchors_dropped_outlier": calib_info["n_dropped_outlier"],
            "bias": calib_info["bias"],
            "rho": calib_info["rho"],
            "sigma_after": calib_info["sigma_after"],
            "thr": calib_info.get("thr", float("nan")),
            "vlm_seconds": t_q,
        })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(pred_files)}] {stem}  "
                  f"s0={s0:.3f}→{s_ref:.3f}  Δ={s_ref-s0:+.4f}  "
                  f"mode={mode}  ρ={calib_info['rho']:+.2f}  "
                  f"kept={n_kept}/{calib_info['n_valid_vlm']}/{args.K}  "
                  f"({t_q:.1f}s VLM)")

    if not per_image:
        raise SystemExit("No images processed.")

    deltas = np.array([r["delta"] for r in per_image])
    s0s = np.array([r["s0"] for r in per_image])
    s_refs = np.array([r["s_ref"] for r in per_image])
    refined_mask = np.array([r["mode"] == "refined" for r in per_image])
    n = len(per_image)

    agg = {
        "n_images": n,
        "n_refined": int(refined_mask.sum()),
        "n_skipped_gate": int(np.sum([r["mode"] == "gate_skip" for r in per_image])),
        "n_skipped_anchors": int(np.sum([r["mode"] == "too_few_anchors" for r in per_image])),
        "baseline_sirmse_mean": float(s0s.mean()),
        "refined_sirmse_mean": float(s_refs.mean()),
        "delta_mean":   float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "n_improved":   int(np.sum(deltas < -1e-6)),
        "n_worsened":   int(np.sum(deltas >  1e-6)),
        "n_unchanged":  int(np.sum(np.abs(deltas) <= 1e-6)),
        # On the gated-IN subset only:
        "delta_mean_refined":    float(deltas[refined_mask].mean()) if refined_mask.any() else float("nan"),
        "n_improved_refined":    int(np.sum(deltas[refined_mask] < -1e-6)),
        "n_worsened_refined":    int(np.sum(deltas[refined_mask] >  1e-6)),
        "rho_distribution":      {
            "mean":   float(np.mean([r["rho"] for r in per_image if np.isfinite(r["rho"])])),
            "median": float(np.median([r["rho"] for r in per_image if np.isfinite(r["rho"])])),
        },
        "bias_distribution":     {
            "abs_mean":   float(np.mean([abs(r["bias"]) for r in per_image if np.isfinite(r["bias"])])),
            "abs_median": float(np.median([abs(r["bias"]) for r in per_image if np.isfinite(r["bias"])])),
        },
        "args": vars(args),
    }

    (out_dir / "results.json").write_text(json.dumps(
        {"aggregate": agg, "per_image": per_image}, indent=2))

    # Human-readable summary.
    L = []
    L.append("# Robust VLM-anchored refinement (per-image calibrated + gated)\n")
    L.append(f"**N images:** {n}  |  refined: {agg['n_refined']}  "
             f"|  gate-skipped: {agg['n_skipped_gate']}  "
             f"|  anchor-skipped: {agg['n_skipped_anchors']}")
    L.append("")
    L.append(f"**Δ-siRMSE (all images, gate-applied):** "
             f"mean={agg['delta_mean']:+.4f}  median={agg['delta_median']:+.4f}  "
             f"({agg['n_improved']}↑ / {agg['n_worsened']}↓ / {agg['n_unchanged']}=)")
    if agg['n_refined'] > 0:
        L.append(f"**On refined subset only:** "
                 f"mean={agg['delta_mean_refined']:+.4f}  "
                 f"({agg['n_improved_refined']}↑ / {agg['n_worsened_refined']}↓)")
    L.append(f"\n**ρ distribution** (per-image agreement, all images): "
             f"mean={agg['rho_distribution']['mean']:+.3f}  "
             f"median={agg['rho_distribution']['median']:+.3f}")
    L.append(f"**|bias| distribution** (per-image scale miscal): "
             f"mean={agg['bias_distribution']['abs_mean']:.3f}  "
             f"median={agg['bias_distribution']['abs_median']:.3f}\n")

    L.append("## Per-image\n")
    L.append("| stem | s0 | s_ref | Δ | mode | ρ | bias | kept/valid/K |")
    L.append("|---|---:|---:|---:|---|---:|---:|---|")
    for r in per_image:
        L.append(
            f"| {r['stem']} | {r['s0']:.3f} | {r['s_ref']:.3f} | "
            f"{r['delta']:+.4f} | {r['mode']} | "
            f"{r['rho']:+.2f} | {r['bias']:+.2f} | "
            f"{r['n_anchors_kept']}/{r['n_anchors_vlm_valid']}/{r['n_anchors_requested']} |"
        )

    (out_dir / "summary.md").write_text("\n".join(L))
    print(f"\n[robust] wrote {out_dir/'results.json'}, {out_dir/'summary.md'}")
    print(f"[robust] Δ-mean (all) = {agg['delta_mean']:+.4f}  "
          f"refined: {agg['n_refined']}/{n}  "
          f"({agg['n_improved']}↑ / {agg['n_worsened']}↓ on all)")
    print(f"[robust] total time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
