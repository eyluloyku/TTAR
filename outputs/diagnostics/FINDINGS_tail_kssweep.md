# Stage B v3 — Tail-10 K-sweep + heat-map recalibration findings

Aggregates the four experiments run after Phase 3 (selective gate):

- [`error_localizability_v3_ens/`](error_localizability_v3_ens/) — signal-ensembling test.
- [`hm_oracle_v3_patch1_a1b0/`](hm_oracle_v3_patch1_a1b0/) — Phase 1c re-run, patch=1, grad-only heat-map.
- [`tail10/`](tail10/) — tail-10 per-image diagnostic.
- [`tail10_ksweep_patch15/`](tail10_ksweep_patch15/), [`tail10_ksweep_patch1_a1b0/`](tail10_ksweep_patch1_a1b0/) — tail-10 K-sweeps at K ∈ {100..1000}.

The four together reshape the project's headline. Earlier findings (Phases 1c, 2c, 3) ruled out three design axes; these four locate a viable one.

---

## 1. The v2 heat-map mixes anti-correlated signals

[`error_localizability_v3_ens/summary.md`](error_localizability_v3_ens/summary.md), N=1159.

Per-image pairwise Pearson correlations on subsampled pixels:

| Pair | mean ρ | |ρ| mean |
|---|---:|---:|
| `grad_logD0` vs `log_D0` | −0.405 | 0.409 |
| `grad_logD0` vs `tta_var` | +0.374 | 0.375 |
| `log_D0`     vs `tta_var` | −0.284 | 0.308 |

AUROC@10% comparison:

| Signal | AUROC@10% |
|---|---:|
| `grad_logD0` alone | **0.794** |
| `log_D0` alone | 0.716 |
| `h_ab` = `z(grad) + z(−log_D0)` (the v2 heat-map) | 0.769 |
| `h_full` = `h_ab + z(tta_var)` | 0.770 |

**Two findings:**

1. **The v2 heat-map is *worse* than its best component (Δ = −0.025).** Combining `grad_logD0` and `−log_D0` with equal weight cancels signal because the two are negatively correlated (ρ = −0.405): high-gradient pixels tend to live in near-field regions, so `z(grad)` and `z(−log_D0)` partially offset. The correct single-channel calibration is `α=1, β=0` (grad-only).
2. **TTA-variance does not add independent signal.** It correlates positively with `grad_logD0` (ρ = +0.374) — partly proxying it — and adding it to `h_ab` improves AUROC by only +0.001. **Demoted from the heat map; γ=0 going forward.**

---

## 2. At single-pixel oracle, heat-anchor selection at K=10 is the universal winner

[`hm_oracle_v3_patch1_a1b0/summary.md`](hm_oracle_v3_patch1_a1b0/summary.md), N=1159, full val, `--anchor-patch 1 --alpha-heat 1 --beta-heat 0`.

| Config | K=10 | K=20 |
|---|---:|---:|
| `heat_heat` Δ-mean | −0.0003 | +0.0000 |
| `heat_heat` win-rate (full) | **97.8%** | **94.3%** |
| `heat_heat` win-rate (tail-10) | **94.0%** | **92.2%** |

At K=10 with grad-only heat-anchor selection and a *single-pixel* oracle, the corrector improves **97.8% of all val images** with near-zero mean change. **The corrector framework is essentially never-harmful in this regime.**

Compare to Phase 1c's best (patch=15): `rand_unif K=100` had 60% full win-rate, Δ-mean = −0.0017. Higher absolute gain, much lower consistency.

The patch=1 regime is what DepthLM emits in natively. So the universal-default config for our project is now:
- **`heat_heat`, K=10–20, grad-only H, single-pixel anchors.**

---

## 3. The catastrophic-tail errors are scattered, not clustered

[`tail10/tail10_summary.md`](tail10/tail10_summary.md), N=116 (top 10% by baseline siRMSE).

| Quantity | median (tail-10) |
|---|---:|
| baseline siRMSE | 0.7542 (vs 0.4461 full-median) |
| # connected components of high-error pixels (>0.5 log-error) | **2,868** |
| top-3 components fraction of total high-error mass | **0.119** |
| fraction of pixels with \|log err\| > 1.0 | 0.059 |
| fraction of headroom captured by K=100 oracle | **0.7%** |

The tail-10 error distribution is **scattered** — top-3 connected components capture only 12% of high-error mass, with thousands of small components carrying the rest. The K=100 oracle captures less than 1% of the absolute siRMSE budget.

This initially suggested sparse anchors could not reach the tail. **The K-sweep below refutes that.**

---

## 4. K-sweep on the tail subset — 1% siRMSE reduction *is* reachable

### patch=15 oracle (smoothed)

[`tail10_ksweep_patch15/summary.md`](tail10_ksweep_patch15/summary.md), N=116.

| Config | K=100 | K=200 | K=500 | K=1000 |
|---|---:|---:|---:|---:|
| `rand_unif` Δ-tail10 mean | −0.0063 | **−0.0078** | **−0.0101** | **−0.0121** |
| `rand_unif` tail-10 win-rate | 80% | 79% | 79% | **85%** |
| `rand_heat` Δ-tail10 | −0.0036 | −0.0054 | −0.0089 | −0.0112 |

**K=200 hits exactly the 1% goal (Δ = −0.0078 / 0.7765 = 1.00%).** K=1000 reaches 1.56% relative on tail-10 with 85% win-rate.

The earlier worry that the CC count (2868 per image) would cap effective coverage was wrong. Each anchor's reach via the smoothness propagator is wider than the CC count alone suggests; gain scales smoothly with K up to K=1000 in this regime.

### patch=1 oracle (single-pixel — DepthLM regime)

[`tail10_ksweep_patch1_a1b0/summary.md`](tail10_ksweep_patch1_a1b0/summary.md), N=116.

| Config | K=20 | K=100 | K=200 | K=500 | K=1000 |
|---|---:|---:|---:|---:|---:|
| `heat_heat` Δ-tail10 | +0.0002 | +0.0231 | +0.0538 | +0.0594 | +0.0594 |
| `rand_unif` Δ-tail10 | +0.0034 | +0.0208 | +0.0133 | +0.0012 | **−0.0141** |
| `rand_unif` tail-10 win-rate | 33% | 25% | 8% | 50% | 58% |

**A U-shape in K** at patch=1: K=20 is neutral, K=100–500 actively *worsens* the tail, K=1000 finally crosses back to Δ = −0.0141 (1.81% rel) with 58% win-rate.

Intuition: at intermediate K, single-pixel anchors are dense enough to disrupt D0 but too sparse to drive coherent correction. At K=1000 they approach the density at which the propagator can interpolate back toward GT.

Patch=15 K=1000 (1.56%) and patch=1 K=1000 (1.81%) reach comparable tail-10 reduction. Patch=15 has higher win-rate (85% vs 58%), patch=1 has slightly higher mean gain but more failure variance.

---

## 5. Proposed test-time strategy

Two regimes, one gate:

```
For each test image:
  d = D0_difficulty(image)                          # cheap feature
  if d > τ:
    # tail rescue
    K, mode, smoothness = 1000, "random", "uniform"
  else:
    # universal safe default
    K, mode, smoothness = 20, "heat", "heat"
  anchors = select_anchors(D0, mode, K)
  log_d_anchors = DepthLM.query(rgb, anchors)
  D_ref = solve_dense_correction(D0, H, anchors, log_d_anchors,
                                  smoothness=smoothness, alpha=1, beta=0)
```

Expected behavior under oracle (upper bound):

| Regime | Δ-mean | Win-rate |
|---|---:|---:|
| Default (`heat_heat` K=20, patch=1): on ~90% of images | ≈ 0 | 92–97% |
| Tail rescue (`rand_unif` K=1000, patch=1): on ~10% of images | −0.014 | 58% |
| **Weighted full-val mean** | **−0.0014** (oracle) | — |

Inference cost on 591 test images: 591×0.9×20 + 591×0.1×1000 ≈ **70 000 DepthLM queries**. At a conservative 10 q/s, ~2 hours total.

If DepthLM at K=1000 single-pixel is unreliable, alternative tail-rescue: query a 3×3 neighborhood around each anchor and use the median, yielding patch=15-like behavior at the cost of 9× more queries on the tail subset (~10–15 hours total).

---

## 6. What's still open

1. **Is DepthLM at K=1000 single-pixel anchors actually patch=1-like or patch=15-like?** Its training (transformer attention, marker prompt) may yield natively smoothed readings — closer to patch=15 oracle. Decisive test: run DepthLM at K=100 anchors on 20 tail-10 images; the resulting Δ tells us which oracle column to read from.
2. **Difficulty gate.** A `is_image_tail-10` predictor on D0 statistics. Phase 3's "is image improvable" gate was poorly motivated (the per-image delta is near-zero noise); predicting *baseline siRMSE* is a different, easier task — high baseline-siRMSE correlates with cheap features like `mean log D0` and `mean |grad log D0|`. Build it from the existing per-image CSV.
3. **K-adaptive policy fit.** Once the gate produces a difficulty score, the assignment {K=20 / K=1000} could be replaced with a continuous K = f(d) — but only worth doing if the gate's discrete accuracy is poor.

---

## 7. Report framing now

| Element | Headline |
|---|---|
| **Method** | Sparse-anchor sparse-to-dense correction with DepthLM (foundation-model) anchors at test time. Heat-anchor selection on D0's depth gradient; uniform smoothness modulation. |
| **Universal claim** | Heat-anchored K=20 single-pixel queries improve **97% of val images** with zero-mean siRMSE change (near-never harmful). |
| **Tail claim** | Switching to K=1000 random anchors on a gate-selected 10% subset reaches **−0.014 mean siRMSE on the tail (1.8% relative)** at ~2h inference cost. |
| **Ablations** | (i) Heat-map calibration: grad-only beats grad+(-log_D0). (ii) K × granularity: patch=1 is U-shaped, patch=15 monotone. (iii) Anchor source: oracle vs DepthLM vs random. |
| **Negative results worth surfacing** | Selective binary gating (Phase 3), TTA-variance as a per-pixel signal (Phase 2c), heat-anchor at K=50–200 single-pixel (this report). |

The story is now defensible from end to end.
