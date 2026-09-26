"""Backend-agnostic anchor-source evaluation harness.

The single evaluation entry point for the TTAR corrector. It runs the full
config × K matrix (tail metrics, win-rate) for a pluggable anchor source from
`src.refine.anchor_backends`:

  --backend oracle           (uses GT — the corrector ceiling)
  --backend second           (anchors from a precomputed second model)
  --backend vlm              (DepthLM Pixtral-12B marker queries)

Swapping sources never touches the corrector or the eval logic.

Per image it:
  - builds the heat-map (α, β, optional γ on TTA-variance);
  - selects K anchors via heat-greedy or random;
  - queries the chosen backend for log-depths at those anchors;
  - solves the dense correction (optionally joint-bilateral via --alpha-rgb);
  - scores against GT and against D0.

The config × K ablation matrix in the README is produced in one invocation.
The `fit_selective_gate.py` analysis and the FINDINGS docs read its
`results.json` directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import load_rgb  # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402
from src.refine.anchor_backends import (  # noqa: E402
    AnchorBackend, DepthLMBackend, OracleBackend, SecondModelBackend,
)
from src.refine.correct_dense import (  # noqa: E402
    smooth_only_correction, solve_dense_correction,
)
from src.refine.heatmap import (  # noqa: E402
    compute_heatmap, select_anchors_heatmap, select_anchors_random,
)

EPS = 1e-6


def stats(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def build_backend(args: argparse.Namespace) -> AnchorBackend:
    if args.backend == "oracle":
        return OracleBackend(gt_root=args.train_root, patch=args.anchor_patch)
    if args.backend == "second":
        if not args.second_preds_dir:
            raise SystemExit("--backend second requires --second-preds-dir")
        return SecondModelBackend(
            preds_dir=args.second_preds_dir,
            name=args.second_name,
            in_log=args.second_in_log,
        )
    if args.backend == "vlm":
        return DepthLMBackend(
            model_id=args.vlm_model_id,
            load_in_8bit=args.vlm_int8,
            load_in_4bit=args.vlm_nf4,
            batch_size=args.vlm_batch_size,
        )
    raise SystemExit(f"unknown backend {args.backend!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True,
                   help="GT root (used by oracle backend, also for si_rmse scoring).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stems-file", default=None,
                   help="Optional newline-delimited file of stems to restrict "
                        "evaluation to (e.g. tail-10 subset).")

    p.add_argument("--Ks", type=int, nargs="+", default=[0, 10, 20, 50, 100])
    p.add_argument("--alpha-heat", type=float, default=1.0)
    p.add_argument("--beta-heat",  type=float, default=1.0)
    p.add_argument("--gamma-heat", type=float, default=0.0)
    p.add_argument("--tta-var-dir", default=None)
    p.add_argument("--lam-s", type=float, default=0.1)
    p.add_argument("--alpha-smooth", type=float, default=2.0)
    p.add_argument("--data-alpha", type=float, default=2.0)
    p.add_argument("--alpha-rgb", type=float, default=0.0,
                   help="Joint-bilateral smoothness weight: edge weight gets an extra "
                        "exp(-alpha_rgb * ||RGB(p)-RGB(q)||_1/(3*255)) factor. "
                        "0 = heat-only (default).")
    p.add_argument("--min-dist-px", type=int, default=24)
    p.add_argument("--anchor-top-frac", type=float, default=0.20)
    p.add_argument("--cg-max-iter", type=int, default=200)
    p.add_argument("--cg-tol", type=float, default=1e-5)
    p.add_argument("--anchor-patch", type=int, default=15,
                   help="Used by OracleBackend; ignored by other backends.")

    # Backend selection.
    p.add_argument("--backend", choices=["oracle", "second", "vlm"], required=True)
    p.add_argument("--second-preds-dir", default=None,
                   help="Required if --backend second. Dir of <stem>.npy depth (or log-depth).")
    p.add_argument("--second-name", default="second_model")
    p.add_argument("--second-in-log", action="store_true",
                   help="Set if second-model .npy files store log-depth, not raw depth.")
    p.add_argument("--vlm-model-id", default="facebook/DepthLM",
                   help="HuggingFace repo or local snapshot for the VLM backend.")
    p.add_argument("--vlm-nf4", action="store_true", default=True,
                   help="Load VLM in 4-bit NF4 (default on 16 GB GPUs).")
    p.add_argument("--vlm-no-nf4", dest="vlm_nf4", action="store_false",
                   help="Disable NF4 quantization.")
    p.add_argument("--vlm-int8", action="store_true", default=False,
                   help="Load in int8 instead of NF4 (uses more VRAM).")
    p.add_argument("--vlm-batch-size", type=int, default=4)
    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    tta_var_dir = Path(args.tta_var_dir) if args.tta_var_dir else None
    if args.gamma_heat != 0.0 and tta_var_dir is None:
        raise SystemExit("--gamma-heat != 0 requires --tta-var-dir.")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    print(f"[eval] backend: {backend.name}")

    pred_files = sorted(preds_dir.glob("*.npy"))
    if args.stems_file:
        wanted = set(
            line.strip() for line in Path(args.stems_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        pred_files = [p for p in pred_files if p.stem in wanted]
        print(f"[eval] filtered to {len(pred_files)} stems from {args.stems_file}")
    if args.limit:
        pred_files = pred_files[: args.limit]

    Ks = args.Ks
    configs = [
        ("heat_heat", "heat", True),
        ("rand_heat", "rand", True),
        ("heat_unif", "heat", False),
        ("rand_unif", "rand", False),
    ]

    rng = np.random.default_rng(args.seed)
    per_image_records: list[dict] = []
    skipped_missing = 0
    skipped_invalid = 0
    t_start = time.time()

    for i, pred_path in enumerate(pred_files):
        stem = pred_path.stem
        rgb_path = train_root / f"{stem}_rgb.png"
        gt_path = train_root / f"{stem}_depth.npy"
        if not rgb_path.exists() or not gt_path.exists():
            skipped_missing += 1
            continue

        D0 = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        if D0.shape != gt.shape:
            skipped_invalid += 1
            continue
        rgb = load_rgb(rgb_path)
        if rgb.shape[:2] != D0.shape:
            skipped_invalid += 1
            continue

        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        if valid.sum() < 1000:
            skipped_invalid += 1
            continue

        # Exclude a border zone so the DepthLM arrow marker (5 px shaft +
        # 2 px arrowhead) can always be rendered: anchors landing within
        # this margin would yield NaN log_depths → NaN siRMSE. Oracle and
        # second-model backends don't strictly need this, but the cost (a
        # 7-pixel border strip of unusable anchors) is negligible.
        if args.backend == "vlm":
            margin = 8
            Hh, Ww = valid.shape
            border_ok = np.zeros_like(valid)
            border_ok[margin:Hh - margin, margin:Ww - margin] = True
            valid &= border_ok

        s0 = si_rmse(D0, gt)
        if not np.isfinite(s0):
            skipped_invalid += 1
            continue

        tta_var = None
        if tta_var_dir is not None:
            tv_path = tta_var_dir / f"{stem}.npy"
            if tv_path.exists():
                tv = np.load(tv_path).astype(np.float32)
                if tv.shape == D0.shape:
                    tta_var = tv

        H = compute_heatmap(
            D0,
            alpha=args.alpha_heat, beta=args.beta_heat,
            gamma=args.gamma_heat, tta_var=tta_var,
        )
        H_for_anchor = np.where(valid, H, -np.inf)

        log_d0 = np.log(np.maximum(D0, EPS))
        rec = {"stem": stem, "s0": s0, "results": {}}

        # Per-image anchor + query cache, keyed by (anchor_mode, K). Heat
        # anchors are deterministic; random anchors are drawn once per (mode,
        # K) and reused so {rand_heat, rand_unif} share queries. Smoothness
        # mode is independent of anchors, so reuse across configs is safe.
        query_cache: dict[tuple[str, int], dict] = {}

        for cfg_name, anchor_mode, use_heat_smooth in configs:
            for K in Ks:
                H_smooth = H if use_heat_smooth else np.zeros_like(H)

                if K == 0:
                    if anchor_mode == "heat":
                        D_ref, info = smooth_only_correction(
                            D0, H_smooth,
                            lam_s=args.lam_s,
                            alpha_smooth=args.alpha_smooth,
                            data_alpha=args.data_alpha,
                            cg_max_iter=args.cg_max_iter,
                            cg_tol=args.cg_tol,
                            rgb=rgb,
                            alpha_rgb=args.alpha_rgb,
                        )
                        s_ref = si_rmse(D_ref, gt)
                        rec["results"][f"{cfg_name}_K{K}"] = {
                            "s_ref": s_ref, "delta": s_ref - s0,
                            "n_anchors": 0, "delta_std": info["delta_std"],
                        }
                    else:
                        mirror_key = f"heat_{'heat' if use_heat_smooth else 'unif'}_K0"
                        rec["results"][f"{cfg_name}_K{K}"] = dict(
                            rec["results"].get(mirror_key, {
                                "s_ref": float("nan"), "delta": float("nan"),
                                "n_anchors": 0, "delta_std": float("nan"),
                            })
                        )
                    continue

                cache_key = (anchor_mode, K)
                if cache_key not in query_cache:
                    if anchor_mode == "heat":
                        anchors = select_anchors_heatmap(
                            H_for_anchor, K=K, min_dist_px=args.min_dist_px,
                            top_frac=args.anchor_top_frac, valid=valid,
                        )
                    else:
                        anchors = select_anchors_random(K, D0.shape, valid=valid, rng=rng)
                    if len(anchors) == 0:
                        query_cache[cache_key] = {
                            "anchors_kept": [], "anchor_dlog": [], "anchor_w": [],
                            "n_anchors_requested": 0,
                        }
                    else:
                        anchor_logd, conf = backend.query_depth(rgb, anchors, stem=stem)
                        # Safety net: drop anchors where the backend couldn't
                        # produce a finite log-depth (border-skip, parse fail,
                        # etc.).
                        triples = [
                            (a, ld, c if conf is not None else 1.0)
                            for a, ld, c in zip(
                                anchors,
                                anchor_logd,
                                conf if conf is not None else [1.0] * len(anchor_logd),
                            )
                            if ld is not None and np.isfinite(ld)
                        ]
                        anchors_kept = [t[0] for t in triples]
                        anchor_dlog = [
                            ld - float(log_d0[y, x])
                            for (y, x), ld, _ in triples
                        ]
                        anchor_w = [t[2] for t in triples]
                        query_cache[cache_key] = {
                            "anchors_kept": anchors_kept,
                            "anchor_dlog": anchor_dlog,
                            "anchor_w": anchor_w,
                            "n_anchors_requested": len(anchors),
                        }

                cached = query_cache[cache_key]
                anchors_kept = cached["anchors_kept"]
                anchor_dlog = cached["anchor_dlog"]
                anchor_w = cached["anchor_w"]
                if not anchors_kept:
                    rec["results"][f"{cfg_name}_K{K}"] = {
                        "s_ref": float("nan"), "delta": float("nan"),
                        "n_anchors": 0, "delta_std": float("nan"),
                        "n_anchors_requested": cached["n_anchors_requested"],
                    }
                    continue

                D_ref, info = solve_dense_correction(
                    D0, H_smooth, anchors_kept, anchor_dlog, anchor_w,
                    lam_s=args.lam_s,
                    alpha_smooth=args.alpha_smooth,
                    cg_max_iter=args.cg_max_iter,
                    cg_tol=args.cg_tol,
                    rgb=rgb,
                    alpha_rgb=args.alpha_rgb,
                )
                s_ref = si_rmse(D_ref, gt)
                rec["results"][f"{cfg_name}_K{K}"] = {
                    "s_ref": s_ref, "delta": s_ref - s0,
                    "n_anchors": info["n_anchors"],
                    "n_anchors_requested": cached["n_anchors_requested"],
                    "delta_std": info["delta_std"],
                }

        per_image_records.append(rec)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(pred_files)}  ({rate:.1f} img/s)  "
                  f"kept={len(per_image_records)}")

    if not per_image_records:
        raise SystemExit("No images evaluated.")

    s0_arr = np.array([r["s0"] for r in per_image_records])
    order = np.argsort(-s0_arr)
    n = len(s0_arr)
    tail_10_idx = order[: max(1, int(np.ceil(n * 0.10)))]
    tail_25_idx = order[: max(1, int(np.ceil(n * 0.25)))]

    aggregate = {
        "backend": backend.name,
        "n_images": n,
        "skipped_missing": skipped_missing,
        "skipped_invalid": skipped_invalid,
        "baseline_sirmse": stats(s0_arr),
        "baseline_sirmse_tail10": stats(s0_arr[tail_10_idx]),
        "baseline_sirmse_tail25": stats(s0_arr[tail_25_idx]),
        "args": vars(args),
        "by_config": {},
    }

    keys = sorted(per_image_records[0]["results"].keys())
    for key in keys:
        deltas = np.array([r["results"][key]["delta"] for r in per_image_records])
        s_refs = np.array([r["results"][key]["s_ref"] for r in per_image_records])
        good = np.isfinite(deltas)
        n_improved = int(np.sum(deltas[good] < -1e-6))
        n_worsened = int(np.sum(deltas[good] > 1e-6))

        def tail_winrate(idx: np.ndarray) -> float:
            d = deltas[idx]
            d = d[np.isfinite(d)]
            if len(d) == 0:
                return float("nan")
            return float((d < -1e-6).mean())

        aggregate["by_config"][key] = {
            "delta_sirmse":         stats(deltas),
            "delta_sirmse_tail10":  stats(deltas[tail_10_idx]),
            "delta_sirmse_tail25":  stats(deltas[tail_25_idx]),
            "refined_sirmse":       stats(s_refs),
            "n_improved": n_improved,
            "n_worsened": n_worsened,
            "n_unchanged": int(np.sum(np.abs(deltas[good]) <= 1e-6)),
            "winrate_tail10": tail_winrate(tail_10_idx),
            "winrate_tail25": tail_winrate(tail_25_idx),
        }

    (out_dir / "results.json").write_text(json.dumps(
        {"aggregate": aggregate, "per_image": per_image_records}, indent=2))

    lines = [f"# Anchor-source eval — backend `{backend.name}`\n"]
    lines.append(f"**N images:** {n}  ")
    lines.append(f"**Baseline siRMSE:** mean={aggregate['baseline_sirmse']['mean']:.4f}  "
                 f"median={aggregate['baseline_sirmse']['median']:.4f}  "
                 f"tail-10 mean={aggregate['baseline_sirmse_tail10']['mean']:.4f}\n")
    for label, key_suffix in [
        ("Δ-siRMSE (full mean)", "delta_sirmse"),
        ("Δ-siRMSE (tail-10 mean)", "delta_sirmse_tail10"),
        ("Δ-siRMSE (tail-25 mean)", "delta_sirmse_tail25"),
    ]:
        lines.append(f"\n## {label}\n")
        lines.append("| Config | " + " | ".join(f"K={k}" for k in Ks) + " |")
        lines.append("|---" + "|---:" * len(Ks) + "|")
        for cfg_name, _, _ in configs:
            row = [f"`{cfg_name}`"]
            for K in Ks:
                v = aggregate["by_config"][f"{cfg_name}_K{K}"][key_suffix]["mean"]
                row.append(f"{v:+.4f}")
            lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Win-rate on tail-10%\n")
    lines.append("| Config | " + " | ".join(f"K={k}" for k in Ks) + " |")
    lines.append("|---" + "|---:" * len(Ks) + "|")
    for cfg_name, _, _ in configs:
        row = [f"`{cfg_name}`"]
        for K in Ks:
            wr = aggregate["by_config"][f"{cfg_name}_K{K}"]["winrate_tail10"]
            row.append(f"{wr*100:.1f}%" if np.isfinite(wr) else "—")
        lines.append("| " + " | ".join(row) + " |")

    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"\n[eval] backend={backend.name}  N={n}")
    print(f"[eval] wrote {out_dir/'results.json'}, {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
