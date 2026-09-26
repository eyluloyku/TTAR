"""Counterfactual sky ablation

Phase 0 found 74% of D0's error is *gross* — thin foreground silhouetted
against sky, with D0 bleeding the far sky depth onto it (91% of gross pixels,
~1 px from the sky border, "D0 says farther"). Everything to date treated that
74% as un-fixable. This probe tests the opposite: is the failure fixable by
removing its *cause*?

Intervention: inpaint the invalid (sky / far-context) region of the RGB with
nearest valid content, re-run the Stage A model, and check whether the
foreground pixels are predicted better. The decisive metrics:

  abs_e on gross pixels   orig vs counterfactual — does removing the sky
                          context recover the bled-out foreground depth?
  spliced siRMSE          D0 with the counterfactual values spliced in on the
                          gross / near-sky pixels (scale-aligned) — the siRMSE
                          a real merge would deliver.

DATASET- AND MODEL-SPECIFIC BY DESIGN. This is a headroom probe to decide
whether the gross-error bucket is worth a method, not a method itself.

Usage:
    python scripts/diagnose_counterfactual_sky.py \
        --config configs/finetune/da_v2_vits_lora.yaml \
        --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt \
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \
        --preds-dir outputs/val_preds/vits_lora_r8/ \
        --out-dir outputs/diagnostics/counterfactual_sky/ --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data import IMAGENET_MEAN, IMAGENET_STD, make_splits  # noqa: E402
from src.common.metrics import si_rmse_np as si_rmse  # noqa: E402
from src.common.utils import load_ckpt  # noqa: E402
from src.finetune.model import LoRAConfig, build_model  # noqa: E402

EPS = 1e-6


def stats(arr) -> dict:
    arr = np.asarray(arr, np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {"n": int(len(arr)), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75))}


def nn_inpaint(rgb: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Fill `fill`-True pixels of rgb [H,W,3] with the nearest non-fill pixel."""
    if not fill.any():
        return rgb.copy()
    idx = distance_transform_edt(fill, return_distances=False, return_indices=True)
    return rgb[idx[0], idx[1]]


@torch.no_grad()
def predict(model, rgb01: np.ndarray, device) -> np.ndarray:
    """rgb01 [H,W,3] in [0,1] -> depth [H,W] in metres (same path as predict.py)."""
    t = torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1).float()
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    out = model(t.unsqueeze(0).to(device))      # [1,1,H,W]
    return out.squeeze().cpu().numpy().astype(np.float32)


def zero_mean_logerr(pred, gt, valid):
    e = np.log(np.maximum(pred, EPS)) - np.log(np.maximum(gt, EPS))
    e = e - e[valid].mean()
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--preds-dir", default=None,
                   help="Cached D0 dir — used only to sanity-check the "
                        "re-prediction path matches.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--near-sky-px", type=int, default=8,
                   help="A valid pixel is 'near sky' if within this many px of "
                        "the invalid region.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lora_dict = cfg["model"].get("lora")
    lora_cfg = LoRAConfig(**lora_dict) if lora_dict else None
    model = build_model(encoder=cfg["model"]["encoder"], checkpoint=None,
                        lora_cfg=lora_cfg).to(device)
    load_ckpt(args.checkpoint, model, map_location=device, strict=True)
    model.eval()

    _, val_files = make_splits(cfg["data"]["root"], val_frac=cfg["data"]["val_frac"],
                               seed=cfg["data"]["seed"])
    val_files = val_files[: args.limit]
    train_root = Path(args.train_root)
    preds_dir = Path(args.preds_dir) if args.preds_dir else None
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    sanity_gap = []
    t0 = time.time()

    for i, rgb_path in enumerate(val_files):
        stem = rgb_path.name.replace("_rgb.png", "")
        gt_path = train_root / f"{stem}_depth.npy"
        if not gt_path.exists():
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"), np.float32) / 255.0
        gt = np.load(gt_path).astype(np.float32)
        if rgb.shape[:2] != gt.shape:
            continue
        valid = (gt > 0) & np.isfinite(gt)
        sky = ~valid                              # invalid = sky / far context proxy
        if valid.sum() < 1000 or sky.sum() < 100:
            continue

        D0 = predict(model, rgb, device)          # re-predict original
        rgb_edit = nn_inpaint(rgb, sky)           # ablate the sky context
        De = predict(model, rgb_edit, device)     # counterfactual prediction
        valid = valid & (D0 > 0) & (De > 0) & np.isfinite(D0) & np.isfinite(De)
        if valid.sum() < 1000:
            continue

        if preds_dir is not None:
            cached = preds_dir / f"{stem}.npy"
            if cached.exists():
                cd = np.load(cached).astype(np.float32)
                if cd.shape == D0.shape:
                    sanity_gap.append(float(np.abs(np.log(np.maximum(cd, EPS))
                                     - np.log(np.maximum(D0, EPS)))[valid].mean()))

        s0 = si_rmse(D0, gt)
        e0 = zero_mean_logerr(D0, gt, valid)
        ee = zero_mean_logerr(De, gt, valid)

        gross = valid & (np.abs(e0) > 1.0)
        d_sky = distance_transform_edt(valid)      # dist of each pixel to invalid
        near = valid & (d_sky <= args.near_sky_px)

        # Splice the counterfactual values into D0 on a target region, after
        # scale-aligning De to D0 on the complement (siRMSE strips global scale).
        def spliced_sirmse(target):
            base = valid & ~target
            if base.sum() < 100 or target.sum() < 1:
                return float("nan")
            shift = float(np.median((np.log(np.maximum(D0, EPS))
                          - np.log(np.maximum(De, EPS)))[base]))
            De_al = De * np.exp(shift)
            D_spl = D0.copy()
            D_spl[target] = De_al[target]
            return si_rmse(D_spl, gt)

        rec = {
            "stem": stem, "s0": s0,
            "gross_frac": float(gross.sum()) / max(int(valid.sum()), 1),
            "abs_e_gross_orig": float(np.abs(e0[gross]).mean()) if gross.any() else float("nan"),
            "abs_e_gross_edit": float(np.abs(ee[gross]).mean()) if gross.any() else float("nan"),
            "abs_e_near_orig": float(np.abs(e0[near]).mean()) if near.any() else float("nan"),
            "abs_e_near_edit": float(np.abs(ee[near]).mean()) if near.any() else float("nan"),
            "s_splice_gross": spliced_sirmse(gross),
            "s_splice_near": spliced_sirmse(near),
        }
        rec["delta_splice_gross"] = rec["s_splice_gross"] - s0
        rec["delta_splice_near"] = rec["s_splice_near"] - s0
        records.append(rec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(val_files)}  kept={len(records)}  "
                  f"({(i+1)/(time.time()-t0):.2f} img/s)", flush=True)

    if not records:
        raise SystemExit("No images evaluated.")

    def delta(idx_key):
        return stats([r[idx_key] for r in records])

    agg = {
        "n_images": len(records),
        "sanity_logdepth_gap_vs_cached": stats(sanity_gap) if sanity_gap else None,
        "baseline_sirmse": stats([r["s0"] for r in records]),
        "gross_frac": stats([r["gross_frac"] for r in records]),
        "abs_e_gross_orig": delta("abs_e_gross_orig"),
        "abs_e_gross_edit": delta("abs_e_gross_edit"),
        "abs_e_near_orig": delta("abs_e_near_orig"),
        "abs_e_near_edit": delta("abs_e_near_edit"),
        "delta_splice_gross": delta("delta_splice_gross"),
        "delta_splice_near": delta("delta_splice_near"),
        "winrate_splice_gross": float(np.mean(
            [r["delta_splice_gross"] < -1e-4 for r in records])),
        "winrate_splice_near": float(np.mean(
            [r["delta_splice_near"] < -1e-4 for r in records])),
    }
    (out_dir / "results.json").write_text(
        json.dumps({"aggregate": agg, "per_image": records}, indent=2))

    geo = agg["abs_e_gross_orig"]["mean"]
    gee = agg["abs_e_gross_edit"]["mean"]
    L = ["# Stage B v5 — probe: counterfactual sky ablation\n"]
    L.append(f"**N images:** {agg['n_images']}  ")
    L.append(f"**Baseline siRMSE:** {agg['baseline_sirmse']['mean']:.4f}  ")
    if agg["sanity_logdepth_gap_vs_cached"]:
        L.append(f"**Sanity** (re-pred vs cached D0, mean |Δlog|): "
                 f"{agg['sanity_logdepth_gap_vs_cached']['mean']:.2e} "
                 f"(should be ~0)  ")
    L.append("\n## Does ablating the sky recover the gross foreground pixels?\n")
    L.append("Mean |zero-meaned log-error| on the relevant pixel sets.\n")
    L.append("| Pixel set | original D0 | counterfactual | change |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| gross (|log err| > 1) | {geo:.4f} | {gee:.4f} | "
             f"{(gee-geo):+.4f} ({(gee/geo-1)*100:+.0f}%) |")
    no = agg["abs_e_near_orig"]["mean"]; ne = agg["abs_e_near_edit"]["mean"]
    L.append(f"| near-sky valid | {no:.4f} | {ne:.4f} | "
             f"{(ne-no):+.4f} ({(ne/no-1)*100:+.0f}%) |")

    L.append("\n## Spliced siRMSE — what a real merge would deliver\n")
    L.append("Counterfactual values spliced into D0 on the target region "
             "(scale-aligned on the complement). Negative = improvement.\n")
    L.append("| Splice target | Δ-siRMSE (mean) | Δ (median) | win-rate |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| gross pixels | {agg['delta_splice_gross']['mean']:+.4f} | "
             f"{agg['delta_splice_gross']['median']:+.4f} | "
             f"{agg['winrate_splice_gross']*100:.0f}% |")
    L.append(f"| near-sky pixels | {agg['delta_splice_near']['mean']:+.4f} | "
             f"{agg['delta_splice_near']['median']:+.4f} | "
             f"{agg['winrate_splice_near']*100:.0f}% |")

    L.append("\n## Verdict\n")
    recovered = (geo - gee) / max(geo, EPS)
    dg = agg["delta_splice_gross"]["mean"]
    L.append(f"- Counterfactual ablation cuts gross-pixel error by "
             f"**{recovered*100:.0f}%** ({geo:.3f} → {gee:.3f}).")
    L.append(f"- Best spliced Δ-siRMSE: **{min(dg, agg['delta_splice_near']['mean']):+.4f}** "
             f"full-mean.")
    go = recovered > 0.25 and dg < -0.005
    L.append(f"- **Probe verdict: {'GO' if go else 'WEAK'}** — "
             + ("the gross bucket is partially recoverable; worth a method."
                if go else "ablation does not recover enough; the gross bucket "
                "stays out of scope."))
    (out_dir / "summary.md").write_text("\n".join(L))

    print(f"\n[cf-sky] N={agg['n_images']}  baseline siRMSE="
          f"{agg['baseline_sirmse']['mean']:.4f}")
    if agg["sanity_logdepth_gap_vs_cached"]:
        print(f"[cf-sky] sanity |Δlog| vs cached D0: "
              f"{agg['sanity_logdepth_gap_vs_cached']['mean']:.2e}")
    print(f"[cf-sky] gross-pixel |err|: {geo:.4f} -> {gee:.4f} "
          f"({recovered*100:+.0f}% recovered)")
    print(f"[cf-sky] spliced Δ-siRMSE: gross={agg['delta_splice_gross']['mean']:+.4f}  "
          f"near={agg['delta_splice_near']['mean']:+.4f}")
    print(f"[cf-sky] wrote {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
