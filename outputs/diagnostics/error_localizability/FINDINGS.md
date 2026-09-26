# Error Localizability — Phase 1 Findings

**Date:** 2026-05-11
**Setup:** N=1159 val images (deterministic split), 8000 random valid pixels per image, no extra inference (signals computed from existing Stage A predictions).
**Source:** [`scripts/diagnose_error_signals.py`](../../../scripts/diagnose_error_signals.py) → [`outputs/diagnostics/error_localizability/`](.)

---

## Question

Before designing an active-query strategy for the VLM-anchor refiner, we need to know: **does D0 (or the RGB image) carry a signal that predicts *where* D0 is wrong**, or is the error effectively unpredictable from inputs alone?

If yes → spend the VLM query budget on those high-error pixels.
If no → fall back to random / grid sampling and report that as a finding.

Per-pixel error is defined siRMSE-consistently:
```
E(x,y) = | log D0(x,y) − log GT(x,y) − μ_image |
```
where μ_image is the per-image mean log-difference (the free direction siRMSE strips out).

We tested four signals — all derivable in milliseconds from D0 and the RGB:

| Signal | What it captures |
|---|---|
| `grad_logD0` | Sobel magnitude on `log D0`. High = depth discontinuity in the prediction. |
| `log_D0` | The log-depth value itself. Tests whether error scales with predicted depth. |
| `rgb_localstd` | Local std of greyscale RGB in a 9×9 window. Texture proxy. |
| `edge_distance` | Distance transform from RGB Canny edges. |

---

## Results

| Signal | ρ (mean) | ρ (median) | \|ρ\| (mean) | AUROC@5% | AUROC@10% | AUROC@20% | sign stable |
|---|---:|---:|---:|---:|---:|---:|---|
| **`grad_logD0`** | **+0.369** | +0.370 | **0.369** | 0.780 | **0.794** | 0.767 | **100% +** |
| **`log_D0`**     | −0.313    | −0.320 | **0.358** | 0.696 | **0.716** | 0.718 | 86% −   |
| `edge_distance`  | +0.066    | +0.065 | 0.129     | 0.548 | 0.559 | 0.569 | 69% + |
| `rgb_localstd`   | −0.031    | −0.022 | 0.130     | 0.554 | 0.564 | 0.572 | 57% − (unstable) |

Decision threshold from the plan: |ρ| > 0.30 *or* AUROC@10% > 0.70 → **STRONG**, usable as active-query selector.

---

## Interpretation

### Two strong, independently meaningful signals

**1. Depth-gradient magnitude (`grad_logD0`).**
The single best predictor: every one of 1159 images shows the same sign (ρ > 0 in 100% of images), mean ρ ≈ 0.37, AUROC@10% ≈ 0.79. **Errors concentrate at depth discontinuities** in the prediction. Two reasons this is unsurprising but useful:

- DA-V2's DPT decoder smooths across boundaries; ground truth is sharp. A few-pixel misalignment at an object edge generates a large log-depth error in those pixels.
- Errors at *true* discontinuities are exactly the structural errors that hurt siRMSE most (siRMSE penalises shape, not scale) — so this signal localises the error class we most want to fix.

**2. Predicted depth value (`log_D0`).**
Comparable strength (|ρ| ≈ 0.36, AUROC@10% ≈ 0.72), with a **negative** sign in 86% of images: lower predicted depth → higher error. Plain reading: **the model is most uncertain on near-field pixels**. Two non-exclusive explanations:

- Near-field structure (foreground objects) has more high-frequency content and small calibration mistakes show up as large log errors there.
- The fine-tuning data is dominated by mid/far range; the near-field tail is undertrained.

This is an additional, partially independent axis from depth-edges: a smooth foreground patch (low gradient, low depth) is suspicious for *value*-reasons even though `grad_logD0` would skip it.

### Two signals that don't work

`rgb_localstd` and `edge_distance` both have |ρ| ≈ 0.13 with unstable signs across images. Plausible explanation: DA-V2 already integrates RGB texture and RGB edges into its prediction, so any residual error is approximately *uncorrelated* with the same low-level RGB features — these signals carry information the model already used.

This is itself a small but reportable finding: **RGB-derived "easy" uncertainty proxies are uninformative** for a strong monocular depth model. Future work that wants free uncertainty signals should look at the depth output, not the RGB input.

---

## Implications for the VLM-anchor pipeline

1. **Active query selection is justified.** Two cheap signals beat random sampling by a wide margin. The VLM query budget should not be spent uniformly.
2. **Compose, don't pick.** Use a heat-map `H = α · standardise(grad_logD0) + β · standardise(−log_D0)`; treat it as a probability surface and sample queries from it. Quick starting point: α = β = 1, normalised per-image. Tune α, β as a hyperparameter in the ablation.
3. **Diversify spatially.** The hottest pixels cluster along edges. Without spatial spread, K queries will all sit on one boundary. Mitigations: (a) farthest-point sampling within the heat-map's top decile; (b) stratify by image quadrant or by SAM region; (c) suppress already-queried pixels with a Gaussian penalty.
4. **No need for Phase 2 (TTA variance) yet.** Phase 1 already produced a strong signal at zero extra GPU cost. TTA variance is worth running later as a comparison (and it's the closest analog to the "uncertainty proxy" baseline in the related work) but it is not blocking.

## Caveats

- Subsample is 8000 random valid pixels per image. The numbers are stable at this size (smoke test on 10 images gave very similar aggregates) but per-image ρ for individual difficult images may be noisier.
- The diagnostic was run on the validation split only. We assume similar localisability on test; we cannot verify until we have labels.
- These signals are correlation-based. A signal with high ρ tells us where to look, not how big the error is — that's a separate calibration problem the VLM is meant to solve.

## Reportable angle

> "We evaluated four input-derived error signals on 1159 val images. The depth-gradient magnitude and the predicted log-depth value are both strongly predictive of per-pixel siRMSE error (|ρ|=0.37, AUROC@10%=0.79 and |ρ|=0.36, AUROC@10%=0.72 respectively, with sign stability >86% across images). RGB texture and edge distance are uninformative (|ρ|≈0.13, unstable sign), consistent with the depth backbone already integrating those features. This justifies an active query strategy for the VLM-anchor refiner and provides the heat-map components used for query selection."

## Reproducing

```bash
sbatch slurm/diagnose_error_signals.sbatch \
    --out-dir outputs/diagnostics/error_localizability/
```
Outputs: `summary.md` (auto-generated), `results.json` (per-image), `FINDINGS.md` (this file).
