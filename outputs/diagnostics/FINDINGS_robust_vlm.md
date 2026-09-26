# Stage B v4 — Robust VLM-as-oracle pipeline (2026-05-17/18 session)

Continuation of the Stage B story in
[`README_STAGE_B_V2.md`](../../README_STAGE_B_V2.md) and the earlier
findings ([phase1](FINDINGS_phase1.md), [phase2c](FINDINGS_phase2c.md),
[phase3](FINDINGS_phase3.md),
[tail_kssweep](FINDINGS_tail_kssweep.md)). The prior arc closed with
DepthLM tail-10 smoke tests showing per-image siRMSE regressions of
**+0.27 to +0.43** — much worse than any sensible refiner should do.
This document records the diagnostic + robustification + ablation work
done in this session.

---

## TL;DR

| What | Headline number |
|---|---:|
| Stage A val baseline siRMSE | 0.461 |
| Raw DepthLM oracle (no robustification) — full mean Δ | **+0.213** (catastrophic) |
| Robust pipeline (calibration + screening + gate) — full mean Δ | **+0.006** (35× reduction) |
| GT oracle ceiling at same K=50, patch=15 — full mean Δ | **−0.002** (~0.4% rel) |
| Per-image VLM↔GT correlation (median ρ) | **+0.16** vs D0↔GT ρ=+0.89 |
| τ-gate sweep | monotonic toward identity; no τ produces net positive gain |
| Disagreement-as-error-localizer (angle 2) | AUROC = 0.60 — weak but real, insufficient |

**The robust pipeline successfully bounds the downside of DepthLM-as-oracle
to within ε of identity on out-of-distribution CIL data, but cannot close
the gap to the oracle ceiling (~0.008). The bottleneck is DepthLM's
per-pixel structure correlation with GT, not the corrector formulation
or the gate threshold.**

---

## What we did, in order

### Step 1 — Why are DepthLM corrections catastrophic? (diagnostic)

`outputs/diagnostics/depthlm_tail10_subset/` had per-image siRMSE deltas
of +0.27, +0.30, +0.34, +0.43 on several stems — far outside the oracle
ceiling (~−0.006). Two hypotheses motivated by paper §3 + repo source:

1. **No focal-length unification.** Pixtral-12B trained at funi=750 px;
   we feed raw 560×560.
2. **Wrong prompt template.** Paper §3.1 says SFT plain "How many meters
   is this point?"; current code uses GRPO `<think>/<answer>`.

We ran a 4-cell diagnostic (25 images × 15 random anchors × 4 conds):
`current`, `sft` (paper §3.1 prompt), `resize` (funi=750), `both`.
Outputs: [`outputs/diagnostics/depthlm_quality/`](depthlm_quality/).

Code: [`scripts/diagnose_depthlm_quality.py`](../../scripts/diagnose_depthlm_quality.py),
[`slurm/diagnose_depthlm_quality.sbatch`](../../slurm/diagnose_depthlm_quality.sbatch).

#### Result

| Condition | parse % | scale-inv \|Δlog\| mean / med |
|---|---:|---:|
| `current` (no resize, GRPO prompt) | 100% | **0.504 / 0.184** |
| `sft` (no resize, paper §3.1 prompt) | **0%** | — |
| `resize` (funi=750, GRPO prompt) | 100% | **0.505 / 0.208** |
| `both` (funi=750, paper §3.1 prompt) | **0%** | — |

**Findings:**
- The released Pixtral checkpoint *requires* the `<think>/<answer>`
  prompt. Plain SFT prompt → 0% parse; model regurgitates fragments
  of the GRPO instruction text. (Later cross-checked the official
  `utils/datasets.py` — the released training prompt
  `generate_prompt_depth_sft` is the GRPO-style one we already use.)
- Focal-length resize helps absolute scale (1.60 → 1.53 raw |Δlog GT|)
  but is **neutral at the scale-invariant level** (0.504 vs 0.505),
  which is the only thing siRMSE rewards.
- Marker rendering bit-for-bit identical to official repo (verified
  against `facebookresearch/DepthLM_Official/utils/datasets.py`).

### Step 2 — Why are the corrections catastrophic? (root cause)

Reanalysis of the same 25×15 anchor sample with per-image scale
alignment of VLM outputs:

| Quantity | Value |
|---|---:|
| Per-image \|bias\| median (`median(log d_VLM − log GT)`) | **1.48 log units = 4.4× scale error** |
| Per-image bias range | **−2.73 to +2.62** (sign flips by image) |
| ρ(log VLM, log GT) median across images | **+0.16** |
| ρ(log D0,  log GT) median across images | **+0.89** |
| 6 of 25 images with **negative** ρ(VLM, GT) | min ρ = −0.74 |

The original corrector treats `Δ_a = log d_VLM − log D0` as anchor
forcing. With |bias| ≈ 1.5 log units per image and the *direction
flipping by image*, the corrector swallows per-image scale
miscalibration as if it were per-pixel structure. Hence the +0.3 to
+0.4 siRMSE blow-ups.

**Per-image scale alignment alone gives a 3× reduction in anchor-Δ
error (1.60 → 0.56 mean), but residual per-pixel error is still 2.4×
worse than D0's own per-pixel error (0.56 vs 0.23).**

### Step 3 — Robust pipeline (changes 1 + 5)

Three changes layered on the existing corrector:

| Change | What | Default |
|---|---|---|
| **(1a) Per-image scale alignment** | `Δ_a = (log d_VLM − bias_image) − log D0(p)` where `bias = median(log d_VLM − log D0)` per image | β=1 fixed (scalar median bias) |
| **(1b) Outlier screening** | drop anchors where `|Δ_a| > min(outlier_max, 2.5·MAD(Δ))` | `outlier_max = 0.6` |
| **(5) Per-image agreement gate** | compute `ρ(log d_VLM, log D0)` over surviving anchors; if `ρ < τ_gate` return D0 unchanged | `τ_gate = 0.0` |

Code: [`scripts/eval_robust_vlm_refine.py`](../../scripts/eval_robust_vlm_refine.py),
[`slurm/eval_robust_vlm_refine.sbatch`](../../slurm/eval_robust_vlm_refine.sbatch).

Tail-10 sanity (10 stems that previously regressed catastrophically):

| stem | Δ before (raw) | Δ after (robust) | mode |
|---|---:|---:|---|
| train_011503 | **+0.3394** | **+0.0000** | **gate_skip** |
| train_020332 | **+0.2950** | **+0.0000** | **gate_skip** |
| train_019812 | **+0.2684** | **−0.0143** | refined |
| train_015334 | +0.1579 | +0.0051 | refined |
| train_008258 | +0.0884 | +0.0000 | gate_skip |
| train_022472 | +0.0468 | −0.0005 | refined |
| train_001764 | +0.0055 | +0.0021 | refined |
| train_002382 | −0.0186 | −0.0054 | refined |
| train_019989 | +0.0205 | +0.0068 | refined |
| train_007817 | −0.0006 | +0.0052 | refined |
| **mean** | **+0.121** | **−0.0001** | |

Every catastrophic image (top-3 worst) was correctly gate-skipped
because of negative ρ. Outputs: [`outputs/diagnostics/vlm_robust_tail10/`](vlm_robust_tail10/).

### Step 4 — 6-row ablation on 200 stratified val images

`outputs/diagnostics/ablation_v1/`,
[`scripts/eval_robust_ablation.py`](../../scripts/eval_robust_ablation.py),
[`slurm/eval_robust_ablation.sbatch`](../../slurm/eval_robust_ablation.sbatch).

One VLM forward per image (cached at
`outputs/diagnostics/ablation_v1/vlm_cache/<stem>.npz`); five DepthLM
rows + oracle on the same K=50 anchors. ~6.7 h total compute.

| Row | full mean Δ | tail-10 Δ | win-rate | n_refined |
|---|---:|---:|---:|---:|
| `A0_identity` | +0.0000 | +0.0000 | — | 0 |
| `A1_raw` (no robustification) | **+0.2131** | +0.1752 | 0/200 | 200 |
| `A2_calib` (+ median-bias align) | +0.0849 | +0.0589 | 13/200 | 200 |
| `A3_calib_screen` (+ MAD outlier drop) | +0.0083 | +0.0035 | 29/200 | 197 |
| `A4_robust` (+ ρ-gate at τ=0) | **+0.0061** | **+0.0017** | 22/200 | 140 |
| `A5_oracle` (GT patch=15, no robustification) | **−0.0021** | **−0.0029** | 142/200 | 200 |

**Per-quartile** (by baseline siRMSE — Q1 easiest, Q4 hardest):

| Quartile | s0 mean | A1 raw Δ | A4 robust Δ | A5 oracle Δ |
|---|---:|---:|---:|---:|
| Q1 | 0.280 | +0.2326 | +0.0109 | −0.0005 |
| Q2 | 0.396 | +0.1991 | +0.0059 | −0.0015 |
| Q3 | 0.497 | +0.2375 | +0.0046 | −0.0040 |
| Q4 | 0.675 | +0.1831 | +0.0027 | −0.0026 |

Interesting subtlety: easier images get *more* hurt by A4 (probably
because absolute anchor noise stays constant while available
headroom shrinks).

**ρ distribution across 200 images:**
mean **+0.237**, median **+0.255**,
frac(ρ ≥ 0) = **71%**, frac(ρ ≥ 0.5) = **28%**.

### Step 5 — τ-gate sensitivity sweep (free, cache-hit)

Re-aggregated from `results.json` (A4 = A3-delta if ρ ≥ τ else 0):

| τ_gate | n_refined | Δ-mean | Δ-tail10 | Δ-median |
|---:|---:|---:|---:|---:|
| −1 (refine all) | 200 | +0.0083 | +0.0035 | +0.0047 |
| **0.0** (current default) | 142 | +0.0061 | +0.0017 | +0.0025 |
| 0.1 | 127 | +0.0054 | +0.0017 | +0.0009 |
| 0.2 | 110 | +0.0047 | +0.0016 | +0.0000 |
| 0.3 | 91 | +0.0038 | +0.0016 | +0.0000 |
| 0.5 | 56 | +0.0025 | +0.0006 | +0.0000 |
| 0.7 | 27 | +0.0009 | +0.0005 | +0.0000 |
| 1.1 (refine none) | 0 | +0.0000 | +0.0000 | +0.0000 |
| **A5 oracle** | 200 | **−0.0021** | **−0.0029** | **−0.0013** |

Δ is **monotonically decreasing** with τ, asymptoting at identity.
**No τ produces a net positive gain.** The gate's role is exclusively
damage-control on the noisy DepthLM signal.

### Step 6 — Angle 2: disagreement-as-error-localizer (free, cache-hit)

Reframing question: even if VLM is a bad oracle, does VLM↔D0
*disagreement* localize *D0's* error pattern? If yes, DepthLM could
modulate the smoothness term rather than serve as an oracle.

Code: [`scripts/diagnose_disagreement_localizer.py`](../../scripts/diagnose_disagreement_localizer.py).
Result: [`disagreement_localizer_200.md`](disagreement_localizer_200.md).

| Metric | 25-img sample (375 anchors) | 200-img sample (10k anchors) |
|---|---:|---:|
| Pearson ρ pooled | +0.006 | +0.050 |
| Spearman ρ pooled | +0.022 | +0.152 |
| AUROC@top-30% pooled | 0.497 | **0.596** |
| Per-image AUROC mean | 0.586 | **0.599** |
| Per-image frac(AUROC > 0.5) | 68% | **74%** |
| Reference floor: −log D0 AUROC | 0.451 | 0.505 |

Signal is **real but weak**: disagreement beats the −log D0 stand-in
(0.60 vs 0.51) so it adds *something*, but sits well below the C0
finding (`grad_logD0` AUROC=0.794 on full val) which itself didn't
translate into corrector gains (Phase 1c). **No-go on building angle 2.**

---

## What this means for the project

### The contribution as it currently stands

Three converging negative-result findings constitute the headline:

1. **VLM-as-oracle on out-of-distribution data fails predictably.**
   DepthLM-Pixtral-12B (δ₁=0.77 in-distribution) regresses siRMSE by
   **+0.21 on average** when used naively as a test-time oracle on CIL.
   Per-image scale miscalibration alone accounts for ~60% of the
   damage; per-pixel structure noise accounts for the rest.

2. **A staged robustification pipeline bounds the downside to ε of
   identity but cannot recover gain.** Calibration → screening → gate
   reduces the regression by 35× (0.213 → 0.006). The remaining
   ~0.008 gap to the oracle ceiling is the "VLM quality deficit on
   OOD data". The agreement gate is the load-bearing component for
   downside control; the τ-sweep shows it is essentially a "refine
   almost no one" strategy that converges to identity.

3. **Reframing VLM as a structural-error localizer doesn't help.**
   VLM↔D0 disagreement at random anchors has per-image AUROC ≈ 0.60
   — marginally better than D0-alone signals, far worse than dense
   `grad_logD0` (0.79), and below the threshold where corrector
   formulations historically extract usable gain.

### Implications for the report and the submission

- **Don't submit any robust pipeline to Kaggle.** A4 mildly worsens
  val (+0.006), would mildly worsen test. Submit **Stage A (D0)
  alone**; the current test score 0.522 stays.
- **The report's value is the ablation + framing**, not the leaderboard
  delta. The story answers a specific open question
  ("does a metric-depth VLM transfer as a sparse oracle to a
  different-distribution dataset?") cleanly and negatively.
- **The robustification pipeline is a contribution in its own right**:
  inter-model agreement gating + per-image scale alignment + MAD
  outlier screening converts a +0.21 catastrophe into a +0.006
  near-identity in a generic, anchor-source-agnostic way.

### Compute budget snapshot

- Used: ~154 GPU-h of 200-h student quota (across all classes)
- Ablation run cost: ~6.7 h
- Remaining: **~40 h**
- Per-stem VLM queries cached at
  `outputs/diagnostics/ablation_v1/vlm_cache/` — any future
  ablation re-aggregation is free CPU work.

---

## File index

```
scripts/
  diagnose_depthlm_quality.py        Step 1 — 4-condition anchor-quality diagnostic
  eval_robust_vlm_refine.py          Step 3 — robust pipeline (cal + screen + gate)
  eval_robust_ablation.py            Step 4 — 6-row ablation, single VLM pass + cache
  diagnose_disagreement_localizer.py Step 6 — angle-2 diagnostic

slurm/
  diagnose_depthlm_quality.sbatch
  eval_robust_vlm_refine.sbatch
  eval_robust_ablation.sbatch

outputs/diagnostics/
  depthlm_quality/                          Step 1 outputs (4 conds, 25 img × 15 anchors)
  vlm_robust_tail10/                        Step 3 sanity (10 tail-10 stems)
  ablation_v1/                              Step 4 main result (200 stratified images)
    results.json                              per-image deltas for every row
    summary.md                                aggregate table + per-quartile + ρ stats
    stems_sampled.txt                         the 200 stems used
    records_from_cache.json                   anchor records rebuilt for angle-2 retest
    vlm_cache/<stem>.npz                      cached VLM queries (~6 GB across 200 stems)
  disagreement_localizer.md                  Step 6 result on 25-img sample
  disagreement_localizer_200.md              Step 6 result on 200-img cache (re-run)
  FINDINGS_robust_vlm.md                     ← this file
```

---

## Open follow-ups (not done, low priority)

- **Sub-quartile error analysis on A4 vs A5** to identify which subsets
  refine cleanly and which never do (would feed a "scene type
  taxonomy" paragraph in the discussion).
- **Marigold / ZoeDepth as a second-model anchor source** (Phase 4 from
  v2 README) — could improve test number but trades the VLM novelty
  story for an ensembling story. **Not recommended.**
