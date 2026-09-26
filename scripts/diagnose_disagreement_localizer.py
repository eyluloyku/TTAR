"""Does VLM↔D0 disagreement localize D0's error?

This is the gating question for "use DepthLM as a structural-error
localizer instead of an oracle". The corrector framing doesn't need
DepthLM to be more accurate than D0 at each pixel — only that *where*
DepthLM disagrees with D0 correlates with *where* D0 is actually wrong.

Inputs: the existing per-anchor records from
  outputs/diagnostics/depthlm_quality/records.json
(25 val images × 15 random anchors, condition `current`).

For each anchor we compute:
  bias_image  = median over image of (log d_VLM − log D0)
  disagreement = | (log d_VLM − bias_image) − log D0(p) |
                                                  ↑ scale-aligned per image
  d0_error_si  = | (log D0(p) − log GT(p))
                  − mean_image(log D0 − log GT) |
                                                  ↑ siRMSE-aligned (strips
                                                    the per-image log-shift
                                                    siRMSE itself strips)

We then test, both pooled and per-image:
  - Pearson + Spearman ρ between disagreement and d0_error_si
  - AUROC for "is this pixel in the top quantile of D0 error?"
    using disagreement as the score

Decision rule for whether to build angle 2:
  - if pooled |ρ| > 0.20 AND AUROC@top30% > 0.60: GO
  - if marginal (0.10–0.20): worth a small experiment
  - if < 0.10: drop angle 2, fall through to negative-result paper
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def log_(x): return math.log(max(x, 1e-6))


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    sx = math.sqrt(sum((x-mx)**2 for x in xs))
    sy = math.sqrt(sum((y-my)**2 for y in ys))
    if sx*sy == 0: return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / (sx*sy)


def spearman(xs, ys):
    """Pearson on ranks."""
    n = len(xs)
    if n < 2: return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def auroc(scores, labels):
    """AUROC where labels ∈ {0,1} and higher score → predicted positive."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return float("nan")
    wins = 0; ties = 0
    for p in pos:
        for n in neg:
            if p > n: wins += 1
            elif p == n: ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/diagnostics/depthlm_quality/records.json")
    ap.add_argument("--condition", default="current",
                    help="Which condition's anchors to use.")
    ap.add_argument("--out", default="outputs/diagnostics/disagreement_localizer.md")
    ap.add_argument("--top-quantile", type=float, default=0.30,
                    help="Top fraction of D0-error pixels to label as 'positive' for AUROC.")
    args = ap.parse_args()

    records = json.load(open(args.records))
    rows = [r for r in records if r["cond"] == args.condition
            and r["d_vlm_m"] == r["d_vlm_m"] and r["d_vlm_m"] > 0]
    print(f"loaded {len(rows)} anchors  (condition={args.condition})")

    # Group by image for per-image scale alignment.
    by_img = defaultdict(list)
    for r in rows: by_img[r["stem"]].append(r)

    pooled_disagree = []   # |calibrated VLM − D0|, per anchor
    pooled_d0_err_si = []  # |D0 − GT| after per-image shift, per anchor
    pooled_image = []      # image id (for grouping later)

    per_image_corrs = []   # one (ρ_pearson, ρ_spearman) per image
    per_image_n_anchors = []

    for stem, grp in by_img.items():
        ld_v = [log_(r["d_vlm_m"]) for r in grp]
        ld_0 = [log_(r["d0_m"])    for r in grp]
        ld_g = [log_(r["gt_m"])    for r in grp]

        bias_v = st.median([v - d for v, d in zip(ld_v, ld_0)])
        # Per-image siRMSE-aligned shift for D0-vs-GT
        e_d0 = [d - g for d, g in zip(ld_0, ld_g)]
        shift_d0 = sum(e_d0) / len(e_d0)

        disagree = [abs((v - bias_v) - d) for v, d in zip(ld_v, ld_0)]
        d0_err_si = [abs(e - shift_d0) for e in e_d0]

        # Pool
        pooled_disagree += disagree
        pooled_d0_err_si += d0_err_si
        pooled_image += [stem] * len(grp)

        if len(grp) >= 4:
            rp = pearson(disagree, d0_err_si)
            rs = spearman(disagree, d0_err_si)
            per_image_corrs.append((rp, rs))
            per_image_n_anchors.append(len(grp))

    # Pooled correlations
    rp_pool = pearson(pooled_disagree, pooled_d0_err_si)
    rs_pool = spearman(pooled_disagree, pooled_d0_err_si)

    # Per-image aggregate
    rp_per = [c[0] for c in per_image_corrs if c[0] == c[0]]
    rs_per = [c[1] for c in per_image_corrs if c[1] == c[1]]

    # AUROC: top-q D0-error anchors as positives, disagreement as score
    n = len(pooled_d0_err_si)
    sorted_d0 = sorted(pooled_d0_err_si, reverse=True)
    q_cut = sorted_d0[int(n * args.top_quantile)] if n else 0
    labels = [1 if e >= q_cut else 0 for e in pooled_d0_err_si]
    pooled_auroc = auroc(pooled_disagree, labels)

    # Random baseline for sanity
    # (AUROC of a constant score = 0.5)

    # Per-image AUROC (only when image has enough anchors with both classes)
    per_image_auroc = []
    for stem, grp in by_img.items():
        if len(grp) < 5: continue
        ld_v = [log_(r["d_vlm_m"]) for r in grp]
        ld_0 = [log_(r["d0_m"])    for r in grp]
        ld_g = [log_(r["gt_m"])    for r in grp]
        bias_v = st.median([v - d for v, d in zip(ld_v, ld_0)])
        e_d0 = [d - g for d, g in zip(ld_0, ld_g)]
        shift_d0 = sum(e_d0) / len(e_d0)
        disagree = [abs((v - bias_v) - d) for v, d in zip(ld_v, ld_0)]
        d0_err_si = [abs(e - shift_d0) for e in e_d0]
        cut = sorted(d0_err_si, reverse=True)[max(1, int(len(d0_err_si)*args.top_quantile))]
        lab = [1 if e >= cut else 0 for e in d0_err_si]
        if sum(lab) and sum(1-l for l in lab):
            per_image_auroc.append(auroc(disagree, lab))

    # Save markdown
    L = ["# Disagreement-as-error-localizer diagnostic\n"]
    L.append(f"**Input:** `{args.records}`, condition `{args.condition}`")
    L.append(f"**Pooled anchors:** {n}  across {len(by_img)} images\n")

    L.append("## Pooled correlation: disagreement vs. D0 error (scale-stripped)\n")
    L.append(f"- Pearson ρ:    **{rp_pool:+.3f}**")
    L.append(f"- Spearman ρ:   **{rs_pool:+.3f}**")
    L.append(f"- AUROC@top-{int(args.top_quantile*100)}%:    **{pooled_auroc:.3f}**  "
             f"(0.50 = random, ≥0.60 = useful, ≥0.70 = strong)\n")

    L.append("## Per-image correlation (across N images with ≥4 anchors)\n")
    if rp_per:
        L.append(f"- Pearson ρ mean = {sum(rp_per)/len(rp_per):+.3f}   "
                 f"median = {st.median(rp_per):+.3f}   "
                 f"frac(ρ > 0) = {100*sum(1 for r in rp_per if r > 0)/len(rp_per):.0f}%")
        L.append(f"- Spearman ρ mean = {sum(rs_per)/len(rs_per):+.3f}   "
                 f"median = {st.median(rs_per):+.3f}   "
                 f"frac(ρ > 0) = {100*sum(1 for r in rs_per if r > 0)/len(rs_per):.0f}%")
    if per_image_auroc:
        a = per_image_auroc
        L.append(f"- AUROC@top-{int(args.top_quantile*100)}% per image: mean = {sum(a)/len(a):.3f}, "
                 f"median = {st.median(a):.3f}, frac(>0.5) = {100*sum(1 for v in a if v > 0.5)/len(a):.0f}%")

    L.append("\n## Decision\n")
    if rp_pool == rp_pool and abs(rp_pool) >= 0.20 and pooled_auroc >= 0.60:
        verdict = "**GO** — disagreement is a useful error-localizer signal. Build angle 2."
    elif rp_pool == rp_pool and abs(rp_pool) >= 0.10:
        verdict = ("**MARGINAL** — signal is weak but non-trivial. A cheap "
                   "angle-2 prototype is worth one day; if it doesn't move "
                   "the mean by 0.001+ on val, abandon.")
    else:
        verdict = ("**NO-GO** — disagreement does not localize D0's error. "
                   "Drop angle 2; commit to the negative-result framing.")
    L.append(verdict)

    # As a sanity benchmark, also report how well |grad log D0| at each anchor
    # would have localized D0 error — that's the §5.2 signal we already know
    # works (AUROC@10% = 0.794 on full val). Here it gives the floor for "any
    # cheap from-D0 signal" so we know whether disagreement adds anything
    # beyond what's already free.
    L.append("\n## Reference floor: |∇log D0| at anchors\n")
    L.append("(C0 found |∇log D0| has AUROC@10% = 0.794 on full val. "
             "On this 25×15 anchor sample we don't have local gradient, "
             "but log_D0 magnitude — also a C0-strong signal — can stand in.)")
    # Use |log D0| as a rough stand-in: §5.2 found z(−log D0) is one of the
    # two strong signals.
    log_d0_anchors = [-log_(r["d0_m"]) for r in rows]  # higher = nearer → more error per §5.2
    rp_ref = pearson(log_d0_anchors, pooled_d0_err_si)
    sorted_d0e = sorted(pooled_d0_err_si, reverse=True)
    cut = sorted_d0e[int(n * args.top_quantile)] if n else 0
    lab = [1 if e >= cut else 0 for e in pooled_d0_err_si]
    auroc_ref = auroc(log_d0_anchors, lab)
    L.append(f"- ρ(−log D0, D0 error) = {rp_ref:+.3f}")
    L.append(f"- AUROC@top-{int(args.top_quantile*100)}% with −log D0 score: {auroc_ref:.3f}")
    L.append("\nIf the disagreement signal AUROC is below this reference, "
             "it adds nothing over signals you already have from D0 alone.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L))
    # Echo to stdout too
    print("\n".join(L))
    print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
