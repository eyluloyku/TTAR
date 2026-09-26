"""Phase 3: fit + evaluate a per-image selective-application gate.

Question: given the per-image deltas from a Phase 1c oracle run, can a
small set of cheap features (computable from D0 and TTA variance at test
time, no GT required) predict whether the corrector helps this image — so
we skip the corrector on images it would regress?

Inputs (all already on disk after Phase 1c + Phase 2a-b):
  - results.json from `eval_anchor_sources.py` (per-image s0, s_ref, delta).
  - val D0 prediction directory (one .npy per image).
  - val TTA-variance directory (one .npy per image; Phase 2 output).

Outputs:
  - features.json  — per-image features + labels for reproducibility.
  - gate.json      — fold-averaged metrics, coefficients, decision rule.
  - summary.md     — human-readable report (full + tail-10 numbers).

The gate is a simple logistic regression on standardised features. We use
5-fold CV; the gate decision for each image comes from the fold in which
it was held out. This keeps the per-image gating decisions honest.

Decision rule at evaluation: refine if P(improve) >= 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

EPS = 1e-6


def per_image_features(D0: np.ndarray, tta_var: np.ndarray | None) -> dict[str, float]:
    """Cheap test-time-computable features. No GT, no model re-inference."""
    log_d = np.log(np.maximum(D0, EPS)).astype(np.float32)
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    feats = {
        "mean_log_d0":  float(log_d.mean()),
        "std_log_d0":   float(log_d.std()),
        "mean_grad":    float(grad.mean()),
        "p95_grad":     float(np.percentile(grad, 95)),
    }
    if tta_var is not None:
        # Compress the heavy tail with log1p so the linear gate isn't
        # dominated by a few sky/transparent-surface outliers.
        ltv = np.log1p(np.maximum(tta_var, 0.0))
        feats["mean_log_tta_var"] = float(ltv.mean())
        feats["p95_log_tta_var"]  = float(np.percentile(ltv, 95))
    return feats


def evaluate_gate(
    s0: np.ndarray, s_ref: np.ndarray, gate: np.ndarray,
    tail_idx: np.ndarray,
) -> dict:
    """Compute gated mean siRMSE = gate * s_ref + (1 - gate) * s0,
    on full set and on the catastrophic-tail subset."""
    chosen = np.where(gate.astype(bool), s_ref, s0)
    return {
        "full_mean":    float(chosen.mean()),
        "full_median":  float(np.median(chosen)),
        "tail10_mean":  float(chosen[tail_idx].mean()),
        "frac_refined": float(gate.mean()),
        "n_refined":    int(gate.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", required=True,
                   help="Phase 1c results.json (per-image s0, s_ref, delta).")
    p.add_argument("--config-key", default="rand_unif_K100",
                   help="Which config × K cell to gate on (default rand_unif_K100 — "
                        "best tail-10 cell from Phase 1c).")
    p.add_argument("--preds-dir", required=True,
                   help="Dir of D0 .npy files (val preds).")
    p.add_argument("--tta-var-dir", default=None,
                   help="Optional dir of per-image TTA variance .npy.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    eval_path = Path(args.eval_json)
    preds_dir = Path(args.preds_dir)
    tta_var_dir = Path(args.tta_var_dir) if args.tta_var_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(eval_path.read_text())
    per_image = raw["per_image"]
    cfg = args.config_key
    if not per_image or cfg not in per_image[0]["results"]:
        raise SystemExit(f"Config key {cfg!r} not found in {eval_path}.")

    rows: list[dict] = []
    t0 = time.time()
    skipped = 0
    for i, rec in enumerate(per_image):
        stem = rec["stem"]
        s0 = float(rec["s0"])
        cfg_rec = rec["results"][cfg]
        delta = cfg_rec["delta"]
        s_ref = cfg_rec["s_ref"]
        if not np.isfinite(delta) or not np.isfinite(s_ref):
            skipped += 1
            continue

        d0_path = preds_dir / f"{stem}.npy"
        if not d0_path.exists():
            skipped += 1
            continue
        D0 = np.load(d0_path).astype(np.float32)
        tta_var = None
        if tta_var_dir is not None:
            tv = tta_var_dir / f"{stem}.npy"
            if tv.exists():
                tta_var = np.load(tv).astype(np.float32)

        feats = per_image_features(D0, tta_var)
        rows.append({
            "stem": stem, "s0": s0, "s_ref": s_ref, "delta": float(delta),
            "improved": int(delta < -1e-6),
            **feats,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(per_image)}  ({(i+1)/(time.time()-t0):.1f}/s)")

    if not rows:
        raise SystemExit("No usable rows. Check paths.")
    print(f"[gate] {len(rows)} images, skipped {skipped}")

    feat_names = sorted(rows[0].keys() - {"stem", "s0", "s_ref", "delta", "improved"})
    X = np.array([[r[k] for k in feat_names] for r in rows], dtype=np.float64)
    y = np.array([r["improved"] for r in rows], dtype=np.int64)
    s0_arr = np.array([r["s0"] for r in rows], dtype=np.float64)
    s_ref_arr = np.array([r["s_ref"] for r in rows], dtype=np.float64)

    n = len(rows)
    order = np.argsort(-s0_arr)
    tail10_idx = order[: max(1, int(np.ceil(n * 0.10)))]

    # Out-of-fold predictions: each image gets a gate decision from a model
    # that didn't see it during training. This is what "honest gated mean"
    # requires.
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_proba = np.zeros(n, dtype=np.float64)
    fold_coefs: list[list[float]] = []
    for tr, te in kf.split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
        ])
        pipe.fit(X[tr], y[tr])
        oof_proba[te] = pipe.predict_proba(X[te])[:, 1]
        fold_coefs.append(pipe.named_steps["lr"].coef_[0].tolist())

    gate_decision = (oof_proba >= 0.5).astype(np.int64)

    # Baselines + gated result.
    baseline_always_d0 = {
        "full_mean":   float(s0_arr.mean()),
        "full_median": float(np.median(s0_arr)),
        "tail10_mean": float(s0_arr[tail10_idx].mean()),
    }
    baseline_always_refine = {
        "full_mean":   float(s_ref_arr.mean()),
        "full_median": float(np.median(s_ref_arr)),
        "tail10_mean": float(s_ref_arr[tail10_idx].mean()),
    }
    gated = evaluate_gate(s0_arr, s_ref_arr, gate_decision, tail10_idx)

    # An oracle gate (label-leaking; upper bound on how good gating could ever be).
    oracle_gate = (s_ref_arr < s0_arr).astype(np.int64)
    oracle = evaluate_gate(s0_arr, s_ref_arr, oracle_gate, tail10_idx)

    # Fit on all data once for reporting + saving the gate.
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
    ]).fit(X, y)
    final_coef = final.named_steps["lr"].coef_[0].tolist()
    final_intercept = float(final.named_steps["lr"].intercept_[0])
    scaler = final.named_steps["scaler"]
    feat_mean = scaler.mean_.tolist()
    feat_std = scaler.scale_.tolist()

    summary = {
        "n_images": n,
        "config_key": cfg,
        "features": feat_names,
        "improved_rate": float(y.mean()),
        "baseline_always_d0": baseline_always_d0,
        "baseline_always_refine": baseline_always_refine,
        "gated_cv": gated,
        "oracle_gate_upper_bound": oracle,
        "final_model": {
            "coef": final_coef,
            "intercept": final_intercept,
            "feature_mean": feat_mean,
            "feature_std": feat_std,
        },
        "fold_coefs": fold_coefs,
        "tail10_indices_size": int(len(tail10_idx)),
    }

    (out_dir / "gate.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "features.json").write_text(json.dumps(rows, indent=2))

    # summary.md
    lines = [f"# Phase 3 — Selective-application gate ({cfg})\n"]
    lines.append(f"**N images:** {n}  **Improved-by-refine rate (label):** {y.mean()*100:.1f}%\n")
    lines.append("## Mean siRMSE — full val\n")
    lines.append("| Strategy | Mean | Median | Tail-10 mean |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Always D0 (Stage A) | {baseline_always_d0['full_mean']:.4f} | "
                 f"{baseline_always_d0['full_median']:.4f} | "
                 f"{baseline_always_d0['tail10_mean']:.4f} |")
    lines.append(f"| Always refine | {baseline_always_refine['full_mean']:.4f} | "
                 f"{baseline_always_refine['full_median']:.4f} | "
                 f"{baseline_always_refine['tail10_mean']:.4f} |")
    lines.append(f"| **Gated (5-fold OOF)** | **{gated['full_mean']:.4f}** | "
                 f"{gated['full_median']:.4f} | "
                 f"**{gated['tail10_mean']:.4f}** |")
    lines.append(f"| Oracle gate (upper bound) | {oracle['full_mean']:.4f} | "
                 f"{oracle['full_median']:.4f} | "
                 f"{oracle['tail10_mean']:.4f} |")
    lines.append("")
    lines.append(f"Gated refines {gated['n_refined']}/{n} images "
                 f"({gated['frac_refined']*100:.1f}%).\n")
    lines.append("## Feature coefficients (standardised, fit on all data)\n")
    lines.append("| Feature | Coefficient | Direction |")
    lines.append("|---|---:|---|")
    for name, c in zip(feat_names, final_coef):
        direction = "→ refine" if c > 0 else "→ skip"
        lines.append(f"| `{name}` | {c:+.3f} | {direction} |")
    lines.append(f"\nIntercept: {final_intercept:+.3f}")
    lines.append("")
    lines.append("Decision rule: refine if "
                 "`P(improve | features) >= 0.5`, where P is from a logistic "
                 "regression on the features above.")
    (out_dir / "summary.md").write_text("\n".join(lines))

    print(f"\n[gate] always-D0:     full={baseline_always_d0['full_mean']:.4f}  "
          f"tail10={baseline_always_d0['tail10_mean']:.4f}")
    print(f"[gate] always-refine: full={baseline_always_refine['full_mean']:.4f}  "
          f"tail10={baseline_always_refine['tail10_mean']:.4f}")
    print(f"[gate] gated (OOF):   full={gated['full_mean']:.4f}  "
          f"tail10={gated['tail10_mean']:.4f}  "
          f"({gated['n_refined']}/{n} refined)")
    print(f"[gate] oracle gate:   full={oracle['full_mean']:.4f}  "
          f"tail10={oracle['tail10_mean']:.4f}")
    print(f"[gate] wrote {out_dir/'gate.json'}, {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
