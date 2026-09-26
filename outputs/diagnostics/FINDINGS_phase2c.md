# Stage B v3 — Phase 2c findings (TTA-variance as per-pixel uncertainty)

**Run:** [`error_localizability_v3/`](error_localizability_v3/), N=1159 val images, 8000 px subsample per image. TTA = 5 augmentations (identity, hflip, ±5% brightness, hflip+5%) implemented in [`src/refine/tta.py`](../../src/refine/tta.py).

The question: *can per-pixel TTA log-depth variance predict where D0 is wrong, better than the correlation-based proxies validated in C0?*

---

## Result

| Signal | \|ρ\| mean | AUROC@10% | Sign | Verdict |
|---|---:|---:|---|---|
| `grad_logD0` | **0.369** | **0.794** | 100% + | STRONG |
| `log_D0`     | **0.358** | **0.716** | 86% −  | STRONG |
| **`tta_var`** | **0.217** | **0.639** | 80% + | **moderate** |
| `edge_distance`  | 0.129 | 0.559 | 69% + | weak |
| `rgb_localstd`   | 0.130 | 0.564 | 57% − | weak |

Decision rule from the plan: strong = `|ρ| > 0.30` *or* `AUROC@10% > 0.70`. TTA-var clears neither.

**Verdict: TTA-var has measurable but moderate signal. It is dominated by the simpler correlation-based proxies already in the heat map.**

---

## Reading the result

TTA-var is 80% sign-stable (high variance → high error in 80% of images). Its AUROC@10% of 0.639 is meaningfully above the random baseline of 0.50 and clearly above the "weak" RGB-based signals — so the augmentations *do* carry real epistemic information. But that information is weaker than what `grad_logD0` already encodes.

Three plausible explanations, in decreasing order of likelihood:

1. **Augmentations are too mild.** Identity + hflip + ±5% brightness perturb the input only slightly; a model trained with augmentation will be near-invariant to them. The variance we measure is closer to "noise floor" than to genuine model disagreement.
2. **Photometric perturbations don't probe the model's actual failure modes.** DA-V2-LoRA's residual error is dominated by *geometric* mistakes (boundary localization, near-field magnitude). These are not the kind of errors that brightness changes expose; they would need scale or perspective probes that we deliberately excluded to keep pixel correspondence intact.
3. **The model is well-calibrated under photometric perturbations.** Its mistakes are systematic / bias-driven, not stochastic.

We could re-test (1) with stronger augmentations (larger color jitter, ±15% brightness, small tilts) at the cost of going further out-of-distribution. We could not cheaply fix (2) without breaking the per-pixel comparison the variance computation relies on. For the level of investment this project warrants, the right call is **accept the moderate verdict and move on**.

---

## Implications for downstream phases

| Phase | Change after Phase 2c |
|---|---|
| **Phase 2e** (γ-sweep in oracle eval) | **Demoted.** Per-pixel TTA-var is weaker than `grad_logD0`; adding it as a third heat-map channel is unlikely to revive heat-anchor selection (which Phase 1c showed is destructive at full val regardless). Worth one γ=1 sanity-check, not a sweep. |
| **Phase 3** (selective application gate) | **Promoted.** Per-pixel TTA-var failed as a query selector, but *image-level* TTA-var statistics (mean, P95, fraction-of-high-variance-pixels) may still be a useful feature for predicting "is this image hard?" — the gate input. The per-image variance distribution is exactly the kind of integrated signal that gets washed out at the per-pixel level. |
| **Phase 4** (anchor sources) | **Headline contribution.** With per-pixel uncertainty failing to revive heat-anchor selection, and the smoothness term contributing ~0 on its own (Phase 1c K=0), the project's central question is now systematically: *which anchor source produces the best test-time-refinement gain?* Oracle / second monocular model / fine-tuned VLM. This is a clean, defensible comparative study irrespective of which source wins. |

---

## Honest project framing post-Phase-2c

The original v2 hypothesis — "an error-likelihood heat map computed from D0 itself drives both anchor selection and smoothness modulation" — survives in qualified form:

- Heat-map signals **correlate with error** (C0: `grad_logD0`, `log_D0` both strong) but do **not translate into corrector gain** when used to *select anchors* (Phase 1c: heat-anchor selection is destructive).
- **Heat-modulated smoothness** is necessary when anchors *are* heat-driven but adds little when anchors are random and uniform smoothness is used (Phase 1c).
- **TTA-variance**, the only genuine epistemic-uncertainty signal we tried, is moderately informative per-pixel but does not beat the correlation-based proxies (this phase).

The two remaining novelty claims, both defensible:

1. **Test-time sparse-anchor refinement of monocular depth, with a systematic comparison of anchor sources.** Oracle (upper bound), second monocular model (cheap, decorrelated), VLM (semantic). The corrector formulation is classical Tikhonov; the *anchor source as design axis* is the contribution. (Phase 4.)
2. **Selective application** — learn when to refine vs when to skip, using cheap per-image features. Improves mean by leaving good predictions alone. (Phase 3.)

A third optional contribution if Phase 4 produces a multi-source pipeline:

3. **Cross-source self-verification** — independent-sensor agreement as a per-anchor reliability score. (Phase 6.)

These are the three things the report leads with.
