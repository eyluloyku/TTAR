"""Diagnose whether D0 carries a signal that predicts its own per-pixel error.

For each val image we have D0 (Stage A prediction), GT, and RGB. We compute a
per-pixel error map E aligned to siRMSE convention (centred per-image log-diff,
absolute value), then test several cheap, no-extra-inference signals S that
might predict E:

    S1: |grad log D0|           (depth-edge magnitude)
    S2: RGB local std            (texture proxy; signed - sign tells direction)
    S3: distance to RGB Canny edge
    S4: log D0                   (depth value itself)

For each signal, per image, on a random subsample of valid pixels we report:
    - Spearman rho(S, E)         (signed; magnitude tells strength)
    - AUROC for top-k% pixels of E (k=5,10,20) using S as score
                                  (if rho<0 we use -S; we report sign)

Aggregates (mean / median / IQR) are written to JSON; a markdown summary is
written next to it for the report.

Run:
    conda activate monodepth-5060
    python scripts/diagnose_error_signals.py \\
        --preds-dir outputs/val_preds/vits_lora_r8/ \\
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \\
        --out-dir outputs/diagnostics/error_localizability/ \\
        --subsample 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import load_rgb  # noqa: E402

EPS = 1e-6


def per_image_error(D0: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """siRMSE-aligned per-pixel absolute error in log space."""
    log_d0 = np.log(np.maximum(D0, EPS))
    log_gt = np.log(np.maximum(gt, EPS))
    d = log_d0 - log_gt
    mu = d[valid].mean()
    return np.abs(d - mu)


def signal_grad_logD0(D0: np.ndarray) -> np.ndarray:
    log_d = np.log(np.maximum(D0, EPS)).astype(np.float32)
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def signal_rgb_local_std(rgb: np.ndarray, ksize: int = 9) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=(ksize, ksize))
    sq_mean = cv2.boxFilter(gray * gray, ddepth=cv2.CV_32F, ksize=(ksize, ksize))
    var = np.clip(sq_mean - mean * mean, 0.0, None)
    return np.sqrt(var)


def signal_edge_distance(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    inv = (edges == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)


def signal_log_D0(D0: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(D0, EPS)).astype(np.float32)


def signal_tta_var(tta_var: np.ndarray | None) -> np.ndarray | None:
    """Per-pixel TTA log-depth variance (precomputed). True epistemic signal."""
    if tta_var is None:
        return None
    return tta_var.astype(np.float32)


def _z(x: np.ndarray) -> np.ndarray:
    """Per-image standardisation: same convention as compute_heatmap()."""
    s = x.std()
    return ((x - x.mean()) / (s + EPS)).astype(np.float32)


def signal_h_ab(D0: np.ndarray) -> np.ndarray:
    """Reproduces the default heat-map (α=β=1, γ=0): z(grad) + z(-log D0).

    Both component signs are aligned positive with error (§5.2): `grad` is +,
    `log_D0` is −, so we use z(-log). Including this as a stand-alone signal
    lets us compare the default heat-map directly against single signals and
    against the proposed full ensemble.
    """
    grad = signal_grad_logD0(D0)
    log_d = signal_log_D0(D0)
    return _z(grad) + _z(-log_d)


def signal_h_full(D0: np.ndarray, tta_var: np.ndarray | None) -> np.ndarray | None:
    """Proposed full ensemble: z(grad) + z(-log D0) + z(tta_var).

    The TTA-variance question (§8): does it add independent signal on top
    of the two correlation-based proxies? Score this composite against each
    component and against `h_ab` to answer it.
    """
    if tta_var is None:
        return None
    grad = signal_grad_logD0(D0)
    log_d = signal_log_D0(D0)
    return _z(grad) + _z(-log_d) + _z(tta_var)


# Base signals derive from D0/RGB only. `tta_var` is included via a side-loaded
# precomputed array (see scripts/precompute_tta_variance.py) and only when
# --tta-var-dir is supplied. Ensemble signals follow.
SIGNALS = {
    "grad_logD0":   lambda D0, rgb, tta_var: signal_grad_logD0(D0),
    "rgb_localstd": lambda D0, rgb, tta_var: signal_rgb_local_std(rgb),
    "edge_distance": lambda D0, rgb, tta_var: signal_edge_distance(rgb),
    "log_D0":       lambda D0, rgb, tta_var: signal_log_D0(D0),
    "tta_var":      lambda D0, rgb, tta_var: signal_tta_var(tta_var),
    "h_ab":         lambda D0, rgb, tta_var: signal_h_ab(D0),
    "h_full":       lambda D0, rgb, tta_var: signal_h_full(D0, tta_var),
}


# Pairs we report pixel-wise Pearson correlation for. Tests whether the
# component signals carry *independent* information (low |corr|) or are
# redundant proxies for the same underlying error mechanism (high |corr|).
PAIRWISE_BASE = ("grad_logD0", "log_D0", "tta_var")


def topk_auroc(signal: np.ndarray, error: np.ndarray, k_pct: float) -> float:
    n = len(error)
    k = max(1, int(n * k_pct / 100))
    thresh = np.partition(error, -k)[-k]
    labels = (error >= thresh).astype(np.int8)
    if labels.sum() == 0 or labels.sum() == n:
        return float("nan")
    try:
        return float(roc_auc_score(labels, signal))
    except ValueError:
        return float("nan")


def evaluate_image(D0: np.ndarray, gt: np.ndarray, rgb: np.ndarray,
                   subsample: int, rng: np.random.Generator,
                   tta_var: np.ndarray | None = None) -> dict | None:
    valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
    if valid.sum() < 1000:
        return None

    error_full = per_image_error(D0, gt, valid)

    # Subsample valid pixels for correlation/AUROC.
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) > subsample:
        sel = rng.choice(valid_idx, size=subsample, replace=False)
    else:
        sel = valid_idx
    error = error_full.ravel()[sel]

    out = {
        "n_valid": int(valid.sum()),
        "n_sampled": int(len(sel)),
        "error_mean": float(error_full[valid].mean()),
        "signals": {},
    }

    # Cache the subsampled per-pixel values per signal so we can compute
    # pairwise correlations cheaply at the end (one matmul, not k² calls).
    sampled_values: dict[str, np.ndarray] = {}

    for name, fn in SIGNALS.items():
        S_full = fn(D0, rgb, tta_var)
        if S_full is None:
            continue  # tta_var not provided this run
        S = S_full.ravel()[sel]
        sampled_values[name] = S
        # Spearman
        rho, _ = spearmanr(S, error)
        rho = float(rho) if np.isfinite(rho) else float("nan")
        # AUROC: orient signal so that higher = more error.
        oriented = S if (np.isnan(rho) or rho >= 0) else -S
        sign = "+" if (np.isnan(rho) or rho >= 0) else "-"
        out["signals"][name] = {
            "spearman_rho": rho,
            "auroc_top5":  topk_auroc(oriented, error, 5),
            "auroc_top10": topk_auroc(oriented, error, 10),
            "auroc_top20": topk_auroc(oriented, error, 20),
            "orientation": sign,  # sign applied to S to align with error
        }

    # Pairwise Pearson correlations between the *base* signals on the same
    # subsampled pixels. Tests independence: a high |ρ(grad, tta_var)| would
    # mean TTA-variance is mostly proxying the depth-gradient signal already
    # in the heat map; a low value means it carries independent information.
    out["pairwise"] = {}
    bases = [b for b in PAIRWISE_BASE if b in sampled_values]
    for i, a in enumerate(bases):
        for b in bases[i + 1:]:
            va, vb = sampled_values[a], sampled_values[b]
            if va.std() < EPS or vb.std() < EPS:
                r = float("nan")
            else:
                r = float(np.corrcoef(va, vb)[0, 1])
            out["pairwise"][f"{a}__vs__{b}"] = r

    return out


def aggregate(per_image: list[dict]) -> dict:
    # Union of signal names — robust to heterogeneous coverage (e.g. tta_var
    # only present when --tta-var-dir was set and the file existed).
    signals: list[str] = []
    seen: set[str] = set()
    for img in per_image:
        for s in img["signals"]:
            if s not in seen:
                seen.add(s)
                signals.append(s)
    agg = {"n_images": len(per_image), "signals": {}}

    def stats(arr: np.ndarray) -> dict:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {"mean": float("nan"), "median": float("nan"),
                    "p25": float("nan"), "p75": float("nan")}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    for s in signals:
        rhos = np.array([img["signals"][s]["spearman_rho"]
                         for img in per_image if s in img["signals"]])
        a5 = np.array([img["signals"][s]["auroc_top5"]
                       for img in per_image if s in img["signals"]])
        a10 = np.array([img["signals"][s]["auroc_top10"]
                        for img in per_image if s in img["signals"]])
        a20 = np.array([img["signals"][s]["auroc_top20"]
                        for img in per_image if s in img["signals"]])
        orientations = [img["signals"][s]["orientation"]
                        for img in per_image if s in img["signals"]]
        frac_pos = float(np.mean([o == "+" for o in orientations])) if orientations else float("nan")
        agg["signals"][s] = {
            "n_images_with_signal": int(len(rhos)),
            "spearman_rho": stats(rhos),
            "auroc_top5":  stats(a5),
            "auroc_top10": stats(a10),
            "auroc_top20": stats(a20),
            "frac_positive_orientation": frac_pos,
            "abs_spearman_mean": float(np.nanmean(np.abs(rhos))),
        }

    # Aggregate pairwise pixel-correlations across images.
    pair_keys: set[str] = set()
    for img in per_image:
        pair_keys.update(img.get("pairwise", {}).keys())
    agg["pairwise"] = {}
    for key in sorted(pair_keys):
        vals = np.array([
            img["pairwise"][key] for img in per_image
            if key in img.get("pairwise", {})
        ])
        agg["pairwise"][key] = stats(vals) | {"abs_mean": float(np.nanmean(np.abs(vals)))}
    return agg


def write_summary_md(out_dir: Path, agg: dict, args: argparse.Namespace) -> None:
    lines = []
    lines.append("# Error-localizability diagnostic — Phase 1\n")
    lines.append(f"**N images:** {agg['n_images']}  ")
    lines.append(f"**Subsample / image:** {args.subsample} pixels  ")
    lines.append(f"**Source preds:** `{args.preds_dir}`  ")
    lines.append(f"**GT root:** `{args.train_root}`\n")
    lines.append("## Question")
    lines.append("Does D0 (or RGB) carry any signal that predicts where D0 will be wrong?")
    lines.append("Per-pixel error is `|log D0 − log GT − mean_per_image|` (siRMSE-aligned).")
    lines.append("Each signal is scored against this error on a random subset of valid pixels per image.\n")
    lines.append("## Aggregate results\n")
    lines.append("| Signal | Spearman ρ (mean) | Spearman ρ (median) | |ρ| mean | AUROC top-5% | AUROC top-10% | AUROC top-20% | sign stable? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for s, v in agg["signals"].items():
        rho_m = v["spearman_rho"]["mean"]
        rho_med = v["spearman_rho"]["median"]
        abs_rho = v["abs_spearman_mean"]
        a5 = v["auroc_top5"]["mean"]
        a10 = v["auroc_top10"]["mean"]
        a20 = v["auroc_top20"]["mean"]
        fpos = v["frac_positive_orientation"]
        sign_note = f"{fpos*100:.0f}% +" if fpos >= 0.5 else f"{(1-fpos)*100:.0f}% −"
        lines.append(f"| `{s}` | {rho_m:+.3f} | {rho_med:+.3f} | {abs_rho:.3f} | "
                     f"{a5:.3f} | {a10:.3f} | {a20:.3f} | {sign_note} |")
    lines.append("\n## Decision rule (from the plan)")
    lines.append("- **Strong signal:** mean |ρ| > 0.30 *or* top-10% AUROC > 0.70 → use as active query selector.")
    lines.append("- **Weak signal:** none of the above → fall back to random / grid sampling; record as a finding.\n")

    # Auto verdict per signal.
    lines.append("## Per-signal verdict")
    for s, v in agg["signals"].items():
        abs_rho = v["abs_spearman_mean"]
        a10 = v["auroc_top10"]["mean"]
        if abs_rho > 0.30 or a10 > 0.70:
            verdict = "**STRONG** — usable as query selector."
        elif abs_rho > 0.15 or a10 > 0.60:
            verdict = "moderate — combine with others or use as tiebreaker."
        else:
            verdict = "weak — not informative on its own."
        sign = "+ (high signal → high error)" if v["frac_positive_orientation"] >= 0.5 \
               else "− (low signal → high error)"
        lines.append(f"- `{s}`: {verdict}  Direction: {sign}")
    lines.append("")

    # Pairwise pixel-correlations between base signals (independence test).
    if agg.get("pairwise"):
        lines.append("## Pairwise pixel-correlation between base signals\n")
        lines.append("Per-image Pearson ρ on the subsampled pixels, aggregated across images. "
                     "**Low |ρ|** ⇒ signals carry *independent* information (good for ensembling). "
                     "**High |ρ|** ⇒ one signal is mostly a proxy for the other (redundant).\n")
        lines.append("| Pair | mean ρ | median ρ | |ρ| mean |")
        lines.append("|---|---:|---:|---:|")
        for key, v in agg["pairwise"].items():
            a, b = key.split("__vs__")
            lines.append(f"| `{a}` vs `{b}` | {v['mean']:+.3f} | {v['median']:+.3f} | {v['abs_mean']:.3f} |")
        lines.append("")

    # Ensemble comparison verdict.
    s_dict = agg["signals"]
    if all(k in s_dict for k in ("grad_logD0", "log_D0", "h_ab")):
        best_single = max(s_dict["grad_logD0"]["auroc_top10"]["mean"],
                          s_dict["log_D0"]["auroc_top10"]["mean"])
        h_ab_auroc = s_dict["h_ab"]["auroc_top10"]["mean"]
        h_full_auroc = (s_dict["h_full"]["auroc_top10"]["mean"]
                        if "h_full" in s_dict else None)
        lines.append("## Ensemble comparison (AUROC @ top-10%)\n")
        lines.append(f"- Best single signal: **{best_single:.3f}**")
        lines.append(f"- v2 heat-map `h_ab` = z(grad) + z(−log D0): **{h_ab_auroc:.3f}** "
                     f"(*Δ vs best single: {h_ab_auroc - best_single:+.3f}*)")
        if h_full_auroc is not None:
            lines.append(f"- Full ensemble `h_full` = `h_ab` + z(tta_var): "
                         f"**{h_full_auroc:.3f}** "
                         f"(*Δ vs `h_ab`: {h_full_auroc - h_ab_auroc:+.3f}*)")
            verdict = (
                "TTA-var earns its place in the heat map."
                if (h_full_auroc - h_ab_auroc) > 0.01
                else "TTA-var does **not** add useful independent signal on top of "
                     "`grad_logD0` + `log_D0`; demote from the heat map.")
            lines.append(f"\n**Verdict:** {verdict}")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--subsample", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta-var-dir", default=None,
                   help="Optional dir of precomputed per-image TTA variance "
                        ".npy files (matching stems). Adds 'tta_var' to SIGNALS.")
    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(preds_dir.glob("*.npy"))
    if args.limit is not None:
        pred_files = pred_files[: args.limit]

    rng = np.random.default_rng(args.seed)
    per_image = []
    skipped_missing = 0
    skipped_invalid = 0
    tta_var_dir = Path(args.tta_var_dir) if args.tta_var_dir else None

    for i, pred_path in enumerate(pred_files):
        stem = pred_path.stem
        rgb_path = train_root / f"{stem}_rgb.png"
        gt_path = train_root / f"{stem}_depth.npy"
        if not rgb_path.exists() or not gt_path.exists():
            skipped_missing += 1
            continue

        D0 = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        rgb = load_rgb(rgb_path)

        if D0.shape != gt.shape:
            skipped_invalid += 1
            continue
        if rgb.shape[:2] != D0.shape:
            rgb = cv2.resize(rgb, (D0.shape[1], D0.shape[0]),
                             interpolation=cv2.INTER_AREA)

        tta_var = None
        if tta_var_dir is not None:
            tta_path = tta_var_dir / f"{stem}.npy"
            if tta_path.exists():
                tta_var = np.load(tta_path).astype(np.float32)
                if tta_var.shape != D0.shape:
                    tta_var = None  # silent skip — diagnostic is robust

        rec = evaluate_image(D0, gt, rgb, subsample=args.subsample, rng=rng,
                             tta_var=tta_var)
        if rec is None:
            skipped_invalid += 1
            continue
        rec["stem"] = stem
        per_image.append(rec)

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(pred_files)} (kept {len(per_image)})")

    if not per_image:
        raise SystemExit("No images evaluated.")

    agg = aggregate(per_image)
    agg["skipped_missing"] = skipped_missing
    agg["skipped_invalid"] = skipped_invalid
    agg["args"] = vars(args)

    (out_dir / "results.json").write_text(json.dumps(
        {"aggregate": agg, "per_image": per_image}, indent=2))
    write_summary_md(out_dir, agg, args)

    print(f"\n[diagnose] N={len(per_image)}  skipped_missing={skipped_missing}  "
          f"skipped_invalid={skipped_invalid}")
    for s, v in agg["signals"].items():
        print(f"  {s:14s}  |rho|_mean={v['abs_spearman_mean']:.3f}  "
              f"AUROC@10%={v['auroc_top10']['mean']:.3f}  "
              f"sign={'+' if v['frac_positive_orientation']>=0.5 else '-'}")
    print(f"\n[diagnose] wrote: {out_dir/'results.json'}")
    print(f"[diagnose] wrote: {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
