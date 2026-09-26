# Stage B v3 — Phase 3 findings (selective-application gate)

**Run:** [`gate_v3/`](gate_v3/), N=1159 val images. Config: `rand_unif_K100` (best tail-10 cell from Phase 1c). Logistic regression on 6 per-image features (`mean_log_d0`, `std_log_d0`, `mean_grad`, `p95_grad`, `mean_log_tta_var`, `p95_log_tta_var`), 5-fold CV, default decision threshold 0.5.

## Result

| Strategy | Full mean siRMSE | Median | Tail-10 mean |
|---|---:|---:|---:|
| Always D0 (Stage A) | 0.4612 | 0.4461 | 0.7765 |
| Always refine | 0.4594 | 0.4428 | 0.7704 |
| **Gated (5-fold OOF)** | **0.4594** | 0.4428 | **0.7708** |
| Oracle gate (label-leaking upper bound) | 0.4579 | 0.4419 | 0.7697 |

The gate's full-mean matches "always refine" to four decimals. Even the *oracle* gate (which knows the true per-image delta) saves only 0.0015 over always-refine on full-mean, and 0.0007 on tail-10. The gate refines 83.5% of images; the "improved" label holds for 59.6% of images.

## Why the gate doesn't help

Phase 1c showed `rand_unif_K100` improves 59.6% of images and worsens ~40%. For the gate to help, **the magnitude of worsening on the regressing 40% must be large enough** for skipping them to outweigh the harm of not refining ambiguous cases. The numbers say otherwise:

- Mean Δ on all images: −0.0017.
- The 40% regressing images cannot be losing more than roughly 0.0015 on average — otherwise always-refine wouldn't be a net win.
- Therefore the upside of correctly skipping them is at most ~0.0006 (the oracle-gate margin), and any imperfect gate captures even less.

This is a fundamental property of the corrector at this anchor density and patch size, not a gate-design failure. The corrector is *per-image near-neutral*; gating individual decisions has nothing meaningful to recover.

## Feature coefficients (informational)

Standardised logistic coefficients (positive → push toward "refine"):

| Feature | Coefficient | Direction |
|---|---:|---|
| `mean_log_tta_var` | +0.418 | → refine |
| `mean_log_d0` | +0.275 | → refine |
| `mean_grad` | +0.178 | → refine |
| `std_log_d0` | −0.176 | → skip |
| `p95_log_tta_var` | −0.265 | → skip |
| `p95_grad` | −0.343 | → skip |

The signs are interpretable: images with diffuse uncertainty (high mean, low extreme) tend to benefit; images dominated by a few extreme-gradient or extreme-TTA-var pixels (high P95, low mean) tend to be regressed by refinement. Useful as a diagnostic; not strong enough to overcome the near-neutral per-image margin.

## Implications

This is the third successive Phase that **rules out** a design axis the v2 plan flagged as promising:

- Phase 1c — K=0 smoothness-only baseline → ruled out as "the story".
- Phase 2c — per-pixel TTA-variance as primary uncertainty signal → moderate, not strong.
- Phase 3 — selective application of refinement → no meaningful gain over always-refine.

Each is a clean negative result; together they sharpen what the contribution actually *is*. After these three phases, the surviving claim is narrow but well-supported:

> **A standard Tikhonov-regularised sparse-to-dense corrector, applied unconditionally at test time with K=100 random anchors and uniform smoothness, improves DA-V2-LoRA's mean siRMSE by 0.4% and tail-10 mean siRMSE by 0.8%.**

Whether this remains the project's headline now depends entirely on **Phase 4** — the anchor-source comparison. If oracle anchors meaningfully beat realistic anchor sources (second monocular model, fine-tuned VLM), the corrector framework has headroom and the question "which sparse-depth sensor is best?" carries the contribution. If realistic anchor sources approach the oracle, the project's interest shifts to "test-time refinement works with any reasonable anchor source" — smaller but still defensible.
