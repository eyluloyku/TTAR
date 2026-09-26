"""Is D0's error boundary-relocalizable?
  Q0.1  Where does D0's siRMSE error live, and is it *moderate boundary-band*
        error or *catastrophic* error?  Bin per-pixel error mass by distance to
        the nearest depth discontinuity, under two error measures:
          - `e2`   raw squared log-error — dominated by a few catastrophic
                   pixels (depth off by 10-45x; see tail10 deep-dive).
          - `e2c`  log-error clipped to +-`err_clip` then squared — the
                   *moderate* error a relocalization method could plausibly
                   fix. This is the honest target measure.

  Q0.3  Relocalization ceiling.  For each pixel, re-sample D0 from the best-
        matching value within a +-w-px window (chosen against GT). siRMSE of
        that warped map upper-bounds boundary relocalization with search radius
        w: it can move a misplaced edge a few pixels, but it CANNOT fix a gross
        error (no good value exists nearby), so catastrophic pixels stay. This
        is the realistic ceiling; the older "set band := GT" number assumes
        perfect knowledge and is reported only as a loose reference.

Three discontinuity definitions are compared on the same footing:
  - `gt`   |grad log GT| over both-valid 4-neighbour edges, plus the valid/
           invalid mask border (sky/far silhouettes are real occlusion edges).
  - `d0`   |grad log D0| — D0's own (possibly mislocalized) edges.
  - `rgb`  grayscale RGB gradient — the photometric edges the existing guided
           filter already uses; the baseline v5 must beat.


Usage:
    python scripts/diagnose_boundary_errormode.py \
        --preds-dir outputs/val_preds/vits_lora_r8/ \
        --train-root /cluster/courses/cil/monocular-depth-estimation/train \
        --out-dir outputs/diagnostics/boundary_phase0/
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

EPS = 1e-6
DIST_BINS = [0.0, 2.0, 4.0, 8.0, 16.0, np.inf]   # px; bin i = [DIST_BINS[i], DIST_BINS[i+1])
WARP_RADII = [2, 4, 8]                            # px; relocalization search radius


def stats(arr) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


# --------------------------------------------------------------------------
# Edge maps
# --------------------------------------------------------------------------

def grad_mag_masked(field: np.ndarray, valid: np.ndarray):
    """Per-pixel gradient magnitude over both-valid 4-neighbour edges.

    mag[p] = max |dfield| over incident edges with both endpoints valid;
    defined[p] = whether any such edge exists.
    """
    H, W = field.shape
    mag = np.zeros((H, W), np.float32)
    defined = np.zeros((H, W), bool)

    dh = np.abs(field[:, 1:] - field[:, :-1])
    vh = valid[:, 1:] & valid[:, :-1]
    dhv = np.where(vh, dh, 0.0).astype(np.float32)
    mag[:, :-1] = np.maximum(mag[:, :-1], dhv); defined[:, :-1] |= vh
    mag[:, 1:] = np.maximum(mag[:, 1:], dhv);   defined[:, 1:] |= vh

    dv = np.abs(field[1:, :] - field[:-1, :])
    vv = valid[1:, :] & valid[:-1, :]
    dvv = np.where(vv, dv, 0.0).astype(np.float32)
    mag[:-1, :] = np.maximum(mag[:-1, :], dvv); defined[:-1, :] |= vv
    mag[1:, :] = np.maximum(mag[1:, :], dvv);   defined[1:, :] |= vv

    return mag, defined


def grad_mag_dense(field: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(field.astype(np.float32))
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def mask_border(valid: np.ndarray) -> np.ndarray:
    """Valid pixels that 4-touch an invalid pixel — sky/far silhouettes."""
    inv = ~valid
    nb = np.zeros_like(valid)
    nb[:, :-1] |= inv[:, 1:]; nb[:, 1:] |= inv[:, :-1]
    nb[:-1, :] |= inv[1:, :]; nb[1:, :] |= inv[:-1, :]
    return valid & nb


def edge_set(mag: np.ndarray, defined: np.ndarray, pct: float) -> np.ndarray:
    vals = mag[defined]
    if vals.size == 0:
        return np.zeros_like(defined)
    thr = float(np.percentile(vals, pct))
    return defined & (mag >= thr)


def dist_to_edge(edge: np.ndarray, shape) -> np.ndarray:
    if not edge.any():
        return np.full(shape, np.inf, np.float32)
    return distance_transform_edt(~edge).astype(np.float32)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift array so that out[p] = a[p + (dy,dx)], edge-replicating the border."""
    pad = max(abs(dy), abs(dx))
    if pad == 0:
        return a
    H, W = a.shape
    ap = np.pad(a, pad, mode="edge")
    return ap[pad + dy: pad + dy + H, pad + dx: pad + dx + W]


def warp_ceiling(log_d0: np.ndarray, log_gt: np.ndarray, gt: np.ndarray,
                 w: int) -> float:
    """siRMSE if each pixel is re-sampled from the best D0 value within +-w px.

    out[p] = log_d0(q*) with q* in the (2w+1)^2 window minimising
    |log_d0(q) - log_gt(p)|. Upper-bounds boundary relocalization with search
    radius w; gross errors survive because no good value exists nearby.
    """
    best_err = np.full(log_d0.shape, np.inf, np.float32)
    best_val = log_d0.astype(np.float32).copy()
    for dy in range(-w, w + 1):
        for dx in range(-w, w + 1):
            cand = _shift(log_d0, dy, dx)
            err = np.abs(cand - log_gt)
            upd = err < best_err
            best_err = np.where(upd, err, best_err)
            best_val = np.where(upd, cand, best_val)
    return si_rmse(np.exp(best_val).astype(np.float32), gt)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stems-file", default=None)
    p.add_argument("--edge-pct", type=float, default=88.0,
                   help="Percentile threshold for edge sets (88 -> top 12%%).")
    p.add_argument("--err-clip", type=float, default=1.0,
                   help="Log-error clip for the 'moderate error' measure e2c. "
                        "1.0 -> depth ratio e^1=2.7x; errors beyond are 'gross'.")
    args = p.parse_args()

    preds_dir = Path(args.preds_dir)
    train_root = Path(args.train_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(preds_dir.glob("*.npy"))
    if args.stems_file:
        wanted = {ln.strip() for ln in Path(args.stems_file).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")}
        pred_files = [f for f in pred_files if f.stem in wanted]
    if args.limit:
        pred_files = pred_files[: args.limit]

    edge_types = ["gt", "d0", "rgb"]
    err_measures = ["e2", "e2c"]
    n_bins = len(DIST_BINS) - 1
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
        if D0.shape != gt.shape:
            skipped += 1
            continue
        rgb = load_rgb(rgb_path)
        if rgb.shape[:2] != D0.shape:
            skipped += 1
            continue

        valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
        if valid.sum() < 1000:
            skipped += 1
            continue

        log_d0 = np.log(np.maximum(D0, EPS)).astype(np.float32)
        log_gt = np.log(np.maximum(gt, EPS)).astype(np.float32)

        # Zero-meaned log error. siRMSE^2 = mean(e'^2) over valid.
        e = log_d0 - log_gt
        e = e - e[valid].mean()
        s0 = float(np.sqrt((e[valid] ** 2).mean()))
        if not np.isfinite(s0) or s0 <= 0:
            skipped += 1
            continue

        # Two error measures: raw e2 (outlier-dominated) and clipped e2c (moderate).
        ec = np.clip(e, -args.err_clip, args.err_clip)
        err = {
            "e2": np.where(valid, e ** 2, 0.0).astype(np.float64),
            "e2c": np.where(valid, ec ** 2, 0.0).astype(np.float64),
        }
        total = {m: float(err[m].sum()) for m in err_measures}
        # frac of error mass that is 'gross' (beyond the clip)
        gross_frac = 1.0 - (total["e2c"] / max(total["e2"], EPS)) \
            if total["e2"] > 0 else 0.0

        # --- edge maps ---------------------------------------------------
        gt_mag, gt_def = grad_mag_masked(log_gt, valid)
        gt_edge = edge_set(gt_mag, gt_def, args.edge_pct) | mask_border(valid)
        d0_mag, d0_def = grad_mag_masked(log_d0, valid)
        d0_edge = edge_set(d0_mag, d0_def, args.edge_pct)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]).astype(np.float32)
        rgb_edge = edge_set(grad_mag_dense(gray), np.ones_like(valid), args.edge_pct)
        edges = {"gt": gt_edge, "d0": d0_edge, "rgb": rgb_edge}

        rec = {"stem": stem, "s0": s0, "n_valid": int(valid.sum()),
               "invalid_frac": float((~valid).mean()), "gross_frac": gross_frac,
               "mass_frac": {}, "area_frac": {}, "warp_ceiling": {},
               "perfect_ceiling": {}}

        for et in edge_types:
            dist = dist_to_edge(edges[et], D0.shape)
            rec["area_frac"][et] = []
            rec["mass_frac"][et] = {m: [] for m in err_measures}
            for b in range(n_bins):
                in_bin = (dist >= DIST_BINS[b]) & (dist < DIST_BINS[b + 1])
                rec["area_frac"][et].append(
                    float((in_bin & valid).sum()) / max(valid.sum(), 1))
                for m in err_measures:
                    rec["mass_frac"][et][m].append(
                        float(err[m][in_bin].sum()) / max(total[m], EPS))
            # loose reference ceiling: set D0 := GT inside the +-8px band
            band = dist <= 8.0
            rec["perfect_ceiling"][et] = si_rmse(
                np.where(band & valid, gt, D0), gt) - s0

        # --- Q0.3 honest relocalization ceiling --------------------------
        for w in WARP_RADII:
            rec["warp_ceiling"][str(w)] = warp_ceiling(log_d0, log_gt, gt, w) - s0

        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pred_files)}  kept={len(records)}  "
                  f"({(i+1)/(time.time()-t0):.1f} img/s)")

    if not records:
        raise SystemExit("No images evaluated.")

    # ---- aggregate ------------------------------------------------------
    s0_arr = np.array([r["s0"] for r in records])
    order = np.argsort(-s0_arr)
    tail10 = list(order[: max(1, int(np.ceil(len(s0_arr) * 0.10)))])
    all_idx = list(range(len(records)))

    def agg(idx):
        out = {"mass_frac": {}, "area_frac": {}, "warp_ceiling": {},
               "perfect_ceiling": {}}
        for et in edge_types:
            out["area_frac"][et] = np.mean(
                [records[j]["area_frac"][et] for j in idx], axis=0).tolist()
            out["mass_frac"][et] = {
                m: np.mean([records[j]["mass_frac"][et][m] for j in idx],
                           axis=0).tolist()
                for m in err_measures}
            out["perfect_ceiling"][et] = stats(
                [records[j]["perfect_ceiling"][et] for j in idx])
        for w in WARP_RADII:
            out["warp_ceiling"][str(w)] = stats(
                [records[j]["warp_ceiling"][str(w)] for j in idx])
        return out

    aggregate = {
        "n_images": len(records), "skipped": skipped,
        "edge_pct": args.edge_pct, "err_clip": args.err_clip,
        "baseline_sirmse": stats(s0_arr),
        "baseline_sirmse_tail10": stats(s0_arr[tail10]),
        "invalid_frac": stats([r["invalid_frac"] for r in records]),
        "gross_frac": stats([r["gross_frac"] for r in records]),
        "full": agg(all_idx), "tail10": agg(tail10),
    }
    (out_dir / "results.json").write_text(
        json.dumps({"aggregate": aggregate, "per_image": records}, indent=2))

    # ---- summary.md -----------------------------------------------------
    bl = [f"[{int(DIST_BINS[b])},{int(DIST_BINS[b+1])})"
          if np.isfinite(DIST_BINS[b + 1]) else f"[{int(DIST_BINS[b])},inf)"
          for b in range(n_bins)]
    L = ["# Stage B v5 — Phase 0: boundary error-mode diagnostic\n"]
    L.append(f"**N images:** {aggregate['n_images']}  (skipped {skipped})  ")
    L.append(f"**Baseline siRMSE:** mean={aggregate['baseline_sirmse']['mean']:.4f}  "
             f"median={aggregate['baseline_sirmse']['median']:.4f}  ")
    L.append(f"**GT invalid fraction:** mean={aggregate['invalid_frac']['mean']:.2f}  ")
    L.append(f"**Gross-error fraction** (|log err| > {args.err_clip}, share of raw "
             f"e2 mass): mean={aggregate['gross_frac']['mean']*100:.1f}%  ")
    L.append(f"**Edge percentile:** top {100-args.edge_pct:.0f}% of gradient "
             f"magnitude.\n")

    def mass_table(scope):
        sub = aggregate[scope]
        out = ["| Edge | measure | " + " | ".join(bl) + " | <=8px |",
               "|---|---" + "|---:" * (n_bins + 1) + "|"]
        for et in edge_types:
            af = sub["area_frac"][et]
            out.append(f"| `{et}` | area% | "
                       + " | ".join(f"{x*100:.1f}" for x in af)
                       + f" | {sum(af[:3])*100:.1f} |")
            for m in err_measures:
                mf = sub["mass_frac"][et][m]
                out.append(f"| `{et}` | {m} mass% | "
                           + " | ".join(f"{x*100:.1f}" for x in mf)
                           + f" | **{sum(mf[:3])*100:.1f}** |")
        return out

    L.append("## Q0.1 — Error mass vs distance to discontinuity\n")
    L.append("`area%` = share of valid pixels in the bin. `e2` = raw squared "
             "log-error (outlier-dominated). `e2c` = clipped/moderate error — "
             "the honest relocalization target. mass% >> area% near the edge "
             "=> error is boundary-concentrated.\n")
    L.append("### Full val\n")
    L += mass_table("full")
    L.append("\n### Tail-10% (worst baseline)\n")
    L += mass_table("tail10")

    L.append("\n## Q0.3 — Relocalization ceiling\n")
    L.append("`warp w` = Delta-siRMSE if every pixel is re-sampled from the best "
             "D0 value within +-w px (honest bound on relocalization — gross "
             "errors survive). `perfect band` = Delta if D0 := GT inside +-8px "
             "(loose; assumes perfect knowledge). Negative = improvement.\n")
    L.append("| Quantity | full | tail-10% |")
    L.append("|---|---:|---:|")
    for w in WARP_RADII:
        L.append(f"| warp w={w} | "
                 f"{aggregate['full']['warp_ceiling'][str(w)]['mean']:+.4f} | "
                 f"{aggregate['tail10']['warp_ceiling'][str(w)]['mean']:+.4f} |")
    for et in ["gt", "d0"]:
        L.append(f"| perfect band (`{et}`, d=8) | "
                 f"{aggregate['full']['perfect_ceiling'][et]['mean']:+.4f} | "
                 f"{aggregate['tail10']['perfect_ceiling'][et]['mean']:+.4f} |")

    # ---- verdict --------------------------------------------------------
    gt_e2c8 = sum(aggregate["full"]["mass_frac"]["gt"]["e2c"][:3])
    gt_area8 = sum(aggregate["full"]["area_frac"]["gt"][:3])
    conc = gt_e2c8 / max(gt_area8, 1e-6)
    warp4 = aggregate["full"]["warp_ceiling"]["4"]["mean"]
    gross = aggregate["gross_frac"]["mean"]
    L.append("\n## Verdict\n")
    L.append(f"- Moderate-error (`e2c`) mass within <=8px of a true "
             f"discontinuity: **{gt_e2c8*100:.1f}%** on **{gt_area8*100:.1f}%** "
             f"of valid pixels (concentration **{conc:.2f}x**).")
    L.append(f"- Gross error (|log err| > {args.err_clip}) is "
             f"**{gross*100:.1f}%** of raw e2 mass — relocalization cannot "
             f"touch this part.")
    L.append(f"- Honest relocalization ceiling (warp w=4): "
             f"**{warp4:+.4f}** full-mean siRMSE "
             f"(v3 anchor-corrector oracle was -0.002).")
    q01 = "PASS" if (gt_e2c8 > 0.35 and conc > 1.3) else "FAIL"
    q03 = "PASS" if warp4 <= -0.006 else "FAIL"
    L.append(f"- **Q0.1 (e2c mass <=8px > 35% and concentration > 1.3x): {q01}**")
    L.append(f"- **Q0.3 (warp w=4 ceiling <= -0.006): {q03}**")
    gate = "GO" if q01 == "PASS" and q03 == "PASS" else "REVIEW"
    L.append(f"- **Phase 0a gate: {gate}** — Q0.2 (SAM boundary quality) still "
             "required before committing.")
    (out_dir / "summary.md").write_text("\n".join(L))

    print(f"\n[phase0] N={aggregate['n_images']}  "
          f"siRMSE={aggregate['baseline_sirmse']['mean']:.4f}  "
          f"gross={gross*100:.0f}%")
    print(f"[phase0] gt-edge e2c mass <=8px = {gt_e2c8*100:.1f}%  "
          f"(area {gt_area8*100:.1f}%, conc {conc:.2f}x)")
    print(f"[phase0] warp ceilings: "
          + "  ".join(f"w={w}:{aggregate['full']['warp_ceiling'][str(w)]['mean']:+.4f}"
                      for w in WARP_RADII))
    print(f"[phase0] Q0.1={q01}  Q0.3={q03}  -> {gate}")
    print(f"[phase0] wrote {out_dir/'summary.md'}  and  results.json")


if __name__ == "__main__":
    main()
