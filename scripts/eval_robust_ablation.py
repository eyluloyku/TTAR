"""Single-pass ablation: identity / raw VLM / + calib / + screen / + gate / oracle.

One VLM forward pass per image (cached on disk under <out-dir>/vlm_cache/)
feeds five DepthLM rows; the oracle row reads GT directly. Same K random
anchors and same corrector knobs across all rows so the comparison is
clean.

Output: results.json (per-image deltas for every row + aggregates),
summary.md (the report table).

Stratified sampling by baseline-siRMSE quartile keeps the sample fair to
both easy and hard images. Pass --stems-file to override with an explicit
list (used for the tail-10 sanity).

Usage:
  sbatch slurm/eval_robust_ablation.sbatch \\
      --out-dir outputs/diagnostics/ablation_v1/ \\
      --n-images 200
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


# ---------------------------------------------------------------------------
# Metrics

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    am, bm = a.mean(), b.mean()
    da, db = a - am, b - bm
    denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / denom) if denom > 0 else float("nan")


def mad(x: np.ndarray) -> float:
    m = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - m)))


# ---------------------------------------------------------------------------
# Per-image ablation logic

def oracle_patch_logd(gt: np.ndarray, anchors_yx, r: int = 7) -> list[float]:
    """Patch-median log GT around each anchor — matches the GT-oracle backend (§7.1).
    patch = 2r+1 = 15 by default."""
    H, W = gt.shape
    log_gt = np.log(np.maximum(gt, EPS))
    out = []
    for (y, x) in anchors_yx:
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        patch_gt = gt[y0:y1, x0:x1]
        mask = patch_gt > 0
        if mask.any():
            out.append(float(np.median(log_gt[y0:y1, x0:x1][mask])))
        else:
            out.append(float(log_gt[y, x]))
    return out


def derive_deltas(
    anchors_yx, log_d_vlm, log_d_gt_patch, log_d0_arr,
    *, outlier_max: float, min_anchors_after_screen: int,
):
    """Returns dict of (mode_name -> (anchors, deltas, info)) ready for the solver.

    Modes: 'raw', 'calib', 'calib_screen', 'oracle'.
    """
    # Filter to VLM-valid anchors first
    pairs = [
        (a, ld) for a, ld in zip(anchors_yx, log_d_vlm)
        if ld is not None and np.isfinite(ld)
    ]
    ld_vlm = np.array([ld for _, ld in pairs], dtype=np.float64)
    yx = [a for a, _ in pairs]
    ld_d0 = np.array([float(log_d0_arr[y, x]) for (y, x) in yx], dtype=np.float64)

    bias = float(np.median(ld_vlm - ld_d0)) if len(ld_vlm) else float("nan")
    rho = pearson(ld_vlm, ld_d0) if len(ld_vlm) >= 2 else float("nan")

    # raw: Δ = log_vlm - log_d0
    delta_raw = (ld_vlm - ld_d0).tolist()
    # calib: subtract per-image median bias
    delta_calib = ((ld_vlm - bias) - ld_d0).tolist()

    # calib + screen
    sigma_post = mad(np.asarray(delta_calib)) if len(delta_calib) else float("nan")
    thr = min(outlier_max, max(2.5 * sigma_post, 0.15)) if np.isfinite(sigma_post) else outlier_max
    keep = [i for i, d in enumerate(delta_calib) if abs(d) < thr]
    cs_anchors = [yx[i] for i in keep]
    cs_deltas = [delta_calib[i] for i in keep]
    n_dropped = len(yx) - len(keep)

    # oracle: read patch-median log GT at the original anchor list (same K).
    # Includes anchors VLM dropped, since oracle doesn't depend on VLM.
    oracle_anchors = list(anchors_yx)
    delta_oracle = [
        float(g - float(log_d0_arr[y, x]))
        for (y, x), g in zip(anchors_yx, log_d_gt_patch)
    ]

    return {
        "raw":          {"anchors": yx,         "deltas": delta_raw,
                         "n_anchors": len(yx)},
        "calib":        {"anchors": yx,         "deltas": delta_calib,
                         "n_anchors": len(yx)},
        "calib_screen": {"anchors": cs_anchors, "deltas": cs_deltas,
                         "n_anchors": len(cs_anchors),
                         "n_dropped": n_dropped},
        "oracle":       {"anchors": oracle_anchors, "deltas": delta_oracle,
                         "n_anchors": len(oracle_anchors)},
        "_meta": {
            "bias": bias, "rho": rho,
            "sigma_after": sigma_post, "thr": thr,
            "n_vlm_valid": len(yx),
        }
    }


def solve_or_identity(
    D0, H, anchors, deltas, args, min_after_screen: int,
):
    """Run corrector if enough anchors, else return D0."""
    if len(anchors) < min_after_screen:
        return D0, "no_anchors"
    D_ref, _ = solve_dense_correction(
        D0, H, anchors, deltas,
        anchor_weights=None,
        lam_s=args.lam_s, alpha_smooth=args.alpha_smooth,
        cg_max_iter=args.cg_max_iter, cg_tol=args.cg_tol,
    )
    return D_ref, "refined"


# ---------------------------------------------------------------------------
# Sampling

def compute_baseline_sirmses(preds_dir: Path, train_root: Path, max_n: int | None):
    out = []
    pred_files = sorted(preds_dir.glob("*.npy"))
    if max_n is not None:
        # use evenly-spaced subset for speed if max_n provided
        pass  # use all by default
    for pp in pred_files:
        stem = pp.stem
        gp = train_root / f"{stem}_depth.npy"
        if not gp.exists(): continue
        try:
            D0 = np.load(pp).astype(np.float32)
            gt = np.load(gp).astype(np.float32)
        except Exception:
            continue
        s = si_rmse(D0, gt)
        if np.isfinite(s):
            out.append((stem, s))
    return out


def stratified_sample(stem_sirmses, n_total, seed):
    """Equal sample from baseline-siRMSE quartiles."""
    rng = np.random.default_rng(seed)
    arr = sorted(stem_sirmses, key=lambda x: x[1])
    n = len(arr)
    per_q = n_total // 4
    q_bounds = [0, n // 4, n // 2, 3 * n // 4, n]
    picked = []
    for k in range(4):
        block = arr[q_bounds[k]:q_bounds[k + 1]]
        idx = rng.choice(len(block), size=min(per_q, len(block)), replace=False)
        picked.extend([block[i][0] for i in idx])
    rng.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-images", type=int, default=200,
                   help="Stratified sample size (50/quartile if N=200).")
    p.add_argument("--stems-file", default=None,
                   help="Explicit stems (skips stratified sampling).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--K", type=int, default=50)
    p.add_argument("--alpha-heat", type=float, default=1.0)
    p.add_argument("--beta-heat",  type=float, default=0.0)
    p.add_argument("--lam-s", type=float, default=0.1)
    p.add_argument("--alpha-smooth", type=float, default=2.0)
    p.add_argument("--cg-max-iter", type=int, default=200)
    p.add_argument("--cg-tol", type=float, default=1e-5)

    p.add_argument("--outlier-max", type=float, default=0.6)
    p.add_argument("--min-anchors-after-screen", type=int, default=10)
    p.add_argument("--tau-gate", type=float, default=0.0)
    p.add_argument("--oracle-patch", type=int, default=15)

    p.add_argument("--vlm-model-id", default="facebook/DepthLM")
    p.add_argument("--vlm-batch-size", type=int, default=4)
    p.add_argument("--vlm-quant", choices=["nf4", "int8", "bf16"], default="nf4")

    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "vlm_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build stem list.
    if args.stems_file:
        stems = [ln.strip() for ln in Path(args.stems_file).read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        print(f"[ablate] stems from file: {len(stems)}")
    else:
        print("[ablate] computing baseline siRMSEs for stratified sampling ...")
        sl = compute_baseline_sirmses(preds_dir, train_root, max_n=None)
        print(f"[ablate] {len(sl)} val images with valid baseline")
        stems = stratified_sample(sl, args.n_images, args.seed)
        (out_dir / "stems_sampled.txt").write_text("\n".join(stems) + "\n")
        print(f"[ablate] stratified sample: {len(stems)} stems")

    # Lazy VLM backend init — only if any cache miss.
    backend = None

    rng_anchors = np.random.default_rng(args.seed)
    per_image: list[dict] = []
    t0 = time.time()

    oracle_r = args.oracle_patch // 2

    for i, stem in enumerate(stems):
        rgb_path = train_root / f"{stem}_rgb.png"
        gt_path = train_root / f"{stem}_depth.npy"
        d0_path = preds_dir / f"{stem}.npy"
        if not (rgb_path.exists() and gt_path.exists() and d0_path.exists()):
            continue

        D0 = np.load(d0_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        rgb = load_rgb(rgb_path)
        if D0.shape != gt.shape or rgb.shape[:2] != D0.shape:
            continue

        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        Hh, Ww = D0.shape
        margin = 8
        b = np.zeros_like(valid)
        b[margin:Hh - margin, margin:Ww - margin] = True
        valid &= b
        if valid.sum() < 1000:
            continue
        s0 = si_rmse(D0, gt)
        if not np.isfinite(s0):
            continue

        # ----- anchors (deterministic per-stem via seed XOR hash, for cache compatibility)
        seed_i = args.seed ^ (hash(stem) & 0xFFFFFFFF)
        rng_im = np.random.default_rng(seed_i)
        anchors = select_anchors_random(args.K, D0.shape, valid=valid, rng=rng_im)
        if len(anchors) == 0:
            continue

        # ----- VLM cache
        cache_path = cache_dir / f"{stem}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            cached_anchors = [tuple(int(v) for v in row)
                              for row in data["anchors"]]
            log_d_vlm = data["log_d_vlm"].tolist()
            # If anchor set changed (e.g. K changed), invalidate.
            if cached_anchors != [tuple(a) for a in anchors]:
                cache_path.unlink()
        if not cache_path.exists():
            if backend is None:
                print("[ablate] no cache for this stem — loading VLM ...")
                backend = DepthLMBackend(
                    model_id=args.vlm_model_id,
                    load_in_4bit=(args.vlm_quant == "nf4"),
                    load_in_8bit=(args.vlm_quant == "int8"),
                    batch_size=args.vlm_batch_size,
                )
                print(f"[ablate] backend ready: {backend.name}")
            t_q = time.time()
            log_d_vlm, _ = backend.query_depth(rgb, anchors, stem=stem)
            t_q = time.time() - t_q
            np.savez(cache_path,
                     anchors=np.array(anchors, dtype=np.int32),
                     log_d_vlm=np.array(log_d_vlm, dtype=np.float64))
        else:
            t_q = 0.0

        # ----- oracle anchor log-depths (patch median)
        log_d_gt_patch = oracle_patch_logd(gt, anchors, r=oracle_r)

        # ----- build all variants' anchor sets
        log_d0_arr = np.log(np.maximum(D0, EPS)).astype(np.float64)
        variants = derive_deltas(
            anchors, log_d_vlm, log_d_gt_patch, log_d0_arr,
            outlier_max=args.outlier_max,
            min_anchors_after_screen=args.min_anchors_after_screen,
        )
        meta = variants["_meta"]

        # ----- gate (A4 only)
        gate_open = (np.isfinite(meta["rho"]) and meta["rho"] >= args.tau_gate)

        # ----- heatmap (same for all rows)
        H = compute_heatmap(D0, alpha=args.alpha_heat, beta=args.beta_heat, gamma=0.0)

        # ----- solve each row
        row_results: dict[str, dict] = {}
        # A0: identity
        row_results["A0_identity"] = {"s_ref": s0, "delta": 0.0, "mode": "identity"}
        # A1: raw, A2: calib, A3: calib+screen, A5: oracle  — no gate
        for mode_name, row_label in [
            ("raw",          "A1_raw"),
            ("calib",        "A2_calib"),
            ("calib_screen", "A3_calib_screen"),
            ("oracle",       "A5_oracle"),
        ]:
            v = variants[mode_name]
            D_ref, m = solve_or_identity(D0, H, v["anchors"], v["deltas"], args,
                                         args.min_anchors_after_screen)
            s_ref = si_rmse(D_ref, gt) if m != "no_anchors" else s0
            row_results[row_label] = {
                "s_ref": s_ref, "delta": float(s_ref - s0),
                "mode": m, "n_anchors_used": v["n_anchors"],
            }
        # A4: same as calib_screen but with gate
        v = variants["calib_screen"]
        if gate_open:
            D_ref, m = solve_or_identity(D0, H, v["anchors"], v["deltas"], args,
                                         args.min_anchors_after_screen)
            s_ref = si_rmse(D_ref, gt) if m != "no_anchors" else s0
            row_results["A4_robust"] = {
                "s_ref": s_ref, "delta": float(s_ref - s0),
                "mode": m, "n_anchors_used": v["n_anchors"],
            }
        else:
            row_results["A4_robust"] = {
                "s_ref": s0, "delta": 0.0, "mode": "gate_skip",
                "n_anchors_used": v["n_anchors"],
            }

        per_image.append({
            "stem": stem, "s0": s0,
            "rho": meta["rho"], "bias": meta["bias"],
            "sigma_after": meta["sigma_after"],
            "n_vlm_valid": meta["n_vlm_valid"],
            "vlm_seconds": t_q,
            "rows": row_results,
        })

        if (i + 1) % 5 == 0 or i + 1 == len(stems):
            elapsed = (time.time() - t0) / 60
            cache_hits = sum(1 for r in per_image if r["vlm_seconds"] == 0.0)
            print(f"  [{i+1}/{len(stems)}] {stem}  s0={s0:.3f}  "
                  f"ρ={meta['rho']:+.2f}  bias={meta['bias']:+.2f}  "
                  f"A1={row_results['A1_raw']['delta']:+.3f}  "
                  f"A4={row_results['A4_robust']['delta']:+.3f}  "
                  f"A5={row_results['A5_oracle']['delta']:+.3f}  "
                  f"({t_q:.0f}s)  [{elapsed:.1f}min, cache_hits={cache_hits}]")
            # Incremental save
            _write_results(per_image, args, out_dir)

    _write_results(per_image, args, out_dir)
    _write_summary(per_image, args, out_dir)
    print(f"\n[ablate] total time: {(time.time() - t0)/60:.1f} min")


def _write_results(per_image, args, out_dir):
    (out_dir / "results.json").write_text(json.dumps(
        {"per_image": per_image, "args": vars(args)}, indent=2))


def _write_summary(per_image, args, out_dir):
    if not per_image:
        print("[ablate] no records, skip summary")
        return
    rows = ["A0_identity", "A1_raw", "A2_calib", "A3_calib_screen",
            "A4_robust", "A5_oracle"]
    n = len(per_image)
    s0s = np.array([r["s0"] for r in per_image])
    tail_idx = np.argsort(-s0s)[: max(1, n // 10)]

    L = ["# Ablation: identity → raw → calib → screen → gate → oracle\n"]
    L.append(f"**N images:** {n}  (stratified by baseline siRMSE quartile)\n")
    L.append(f"**Baseline siRMSE:** mean={s0s.mean():.4f}  "
             f"median={np.median(s0s):.4f}  "
             f"tail-10 mean={s0s[tail_idx].mean():.4f}\n")
    L.append("\n## Δ-siRMSE relative to D0 (lower = better)\n")
    L.append("| Row | full mean | full median | tail-10 mean | win-rate | n_refined |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for key in rows:
        deltas = np.array([r["rows"][key]["delta"] for r in per_image])
        n_ref = sum(1 for r in per_image
                    if r["rows"][key]["mode"] not in ("identity", "gate_skip", "no_anchors"))
        improved = int(np.sum(deltas < -1e-6))
        L.append(
            f"| `{key}` | {deltas.mean():+.4f} | {np.median(deltas):+.4f} "
            f"| {deltas[tail_idx].mean():+.4f} "
            f"| {improved}/{n} ({100*improved/n:.0f}%) | {n_ref} |"
        )

    # Per-quartile breakdown (rows A1 / A4 / A5 only — keep table small)
    qs = np.argsort(s0s)
    L.append("\n## By baseline-siRMSE quartile  (A1 vs A4 vs A5)\n")
    L.append("| Quartile | n | s0 mean | A1 raw Δ | A4 robust Δ | A5 oracle Δ |")
    L.append("|---|---:|---:|---:|---:|---:|")
    quartiles = [qs[k * n // 4: (k + 1) * n // 4] for k in range(4)]
    for k, idx in enumerate(quartiles):
        if len(idx) == 0:
            continue
        s0m = s0s[idx].mean()
        d1 = np.array([per_image[i]["rows"]["A1_raw"]["delta"] for i in idx]).mean()
        d4 = np.array([per_image[i]["rows"]["A4_robust"]["delta"] for i in idx]).mean()
        d5 = np.array([per_image[i]["rows"]["A5_oracle"]["delta"] for i in idx]).mean()
        L.append(f"| Q{k+1} | {len(idx)} | {s0m:.3f} | {d1:+.4f} | {d4:+.4f} | {d5:+.4f} |")

    # ρ histogram
    rhos = np.array([r["rho"] for r in per_image
                     if np.isfinite(r["rho"])])
    L.append(f"\n## VLM–D0 agreement (across {len(rhos)} images)\n")
    L.append(f"ρ mean = {rhos.mean():+.3f}  median = {np.median(rhos):+.3f}  "
             f"frac(ρ ≥ 0) = {100*np.mean(rhos >= 0):.1f}%  "
             f"frac(ρ ≥ 0.5) = {100*np.mean(rhos >= 0.5):.1f}%")

    (out_dir / "summary.md").write_text("\n".join(L))
    print(f"[ablate] wrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
