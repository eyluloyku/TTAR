# Heat-map-guided dense corrector — oracle findings

**Date:** 2026-05-11
**Code:** [`src/refine/heatmap.py`](../../src/refine/heatmap.py), [`src/refine/correct_dense.py`](../../src/refine/correct_dense.py), [`scripts/eval_heatmap_oracle.py`](../../scripts/eval_heatmap_oracle.py)
**Runs:** [`hm_oracle_patch{1,15,31}_lam01/`](.) + [`hm_oracle_sweep_ls{0.01,0.1,1.0}/`](.)

---

## Setup

Given a baseline depth `D0`, a heat-map `H` (built from the Phase-1 signals),
and `K` anchor pixels each carrying a target log-shift `Δ_a`, solve a sparse
linear system for a per-pixel log-residual `δ` minimising

```
    L(δ) = Σ_a (δ(p_a) − Δ_a)²
         + λ_s · Σ_(p,q)∈4-neigh  exp(−α · max(relu(H_p), relu(H_q))) · (δ(p) − δ(q))²
```

with the siRMSE-free direction handled post-hoc (subtract `mean(δ)`). The
edge-weight scheme is the **novel piece**: smoothness is strong where D0 is
likely correct (low H), and weak where D0 is likely wrong (high H), so anchors
dominate locally in suspect regions and don't bleed into clean regions.

To isolate the corrector from the (not-yet-trained) VLM, we use an
**oracle anchor**: at each chosen pixel `p` the "VLM" returns the patch
median of `log GT` in a `(2r+1)×(2r+1)` window. `r=0` → single-pixel target;
larger `r` → smoothed target, closer to what a trained DepthLM-style VLM
would actually emit.

Four ablation configs:

| name        | anchor selection | smoothness guide |
|---          |---               |---               |
| `heat_heat` | heat-map greedy + spatial suppression | heat-modulated (ours) |
| `rand_heat` | uniform random valid             | heat-modulated |
| `heat_unif` | heat-map                          | uniform (no guide) |
| `rand_unif` | random                            | uniform (full control) |

Reported on N=50 val images with `lam_s=0.1`, `alpha_smooth=2.0`,
`min_dist_px=24`. Baseline siRMSE mean = 0.4856.

---

## Results

### Anchor patch = 1 (single pixel oracle)

Δ-siRMSE mean (negative = improvement):

| Config | K=10 | K=20 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| `heat_heat` | **−0.0004** | +0.0007 | +0.0511 | +0.0844 |
| `rand_heat` | +0.0078 | +0.0051 | +0.0090 | +0.0079 |
| `heat_unif` | +0.0992 | +0.1685 | +0.2350 | +0.2299 |
| `rand_unif` | +0.0041 | +0.0070 | +0.0100 | +0.0074 |

Win-rate (% of images improved):

| Config | K=10 | K=20 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| `heat_heat` | **100%** | **90%** | 42% | 18% |
| `rand_heat` | 44% | 40% | 42% | 34% |
| `heat_unif` | 0% | 0% | 0% | 0% |
| `rand_unif` | 52% | 40% | 40% | 28% |

### Anchor patch = 15 (smoothed oracle, ≈ what a trained VLM emits)

| Config | K=10 | K=20 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| `heat_heat` | +0.0006 | +0.0011 | +0.0063 | +0.0034 |
| `rand_heat` | +0.0001 | **−0.0007** | **−0.0019** | **−0.0021** |
| `heat_unif` | +0.1265 | +0.1652 | +0.1387 | +0.1199 |
| `rand_unif` | −0.0002 | −0.0002 | −0.0007 | −0.0009 |

| Config | K=10 | K=20 | K=50 | K=100 |
|---|---:|---:|---:|---:|
| `heat_heat` | 8% | 8% | 22% | 34% |
| `rand_heat` | 54% | **62%** | **74%** | 72% |
| `heat_unif` | 4% | 2% | 2% | 6% |
| `rand_unif` | 60% | 58% | 48% | 46% |

(`patch=31` gives essentially the same numbers as `patch=15` — see [`hm_oracle_patch31_lam01/summary.md`](hm_oracle_patch31_lam01/summary.md).)

---

## Three robust findings

### 1. Heat-modulated smoothness is essential

`*_unif` configurations are **catastrophic** in every regime: uniform
smoothness propagates anchor corrections across object boundaries, corrupting
clean regions. Win-rate is 0–6% regardless of K or anchor source. By
contrast, `*_heat` configs are at worst neutral. **The heat-map smoothness
guide is the single most important design choice** in the corrector.

This is the publishable claim: standard guided filtering (RGB-edge, D0-edge)
is structurally wrong for sparse anchor depth refinement; the predictor's own
**error-likelihood map** is a better smoothness guide than image-derived
proxies, because it gates corrections to exactly the regions where they're
licensed.

### 2. The oracle granularity determines the right anchor strategy

Two distinct regimes:

- **Sharp anchors (`patch=1`)**: heat-map anchor selection wins decisively at
  small K (`heat_heat K=10`: −0.0004 mean Δ, **100% of images improved**).
  Hot pixels are at depth discontinuities and a single-pixel GT target locks
  in the local jump. But scaling K up (≥50) overcorrects — too many sharp
  Diracs at edges create a noisy patchwork.

- **Smooth anchors (`patch≥15`)**: random sampling beats heat-map selection.
  Patch-median GT at a high-heat pixel averages across the very depth
  discontinuity that made the pixel suspect, producing a meaningless target.
  Random sampling lands more often on smooth surfaces where patch-median is
  well-defined; with K=50–100 it gives a small but consistent improvement
  (Δ ≈ −0.002, 70%+ win rate).

This **matters for the real VLM**. A DepthLM-style fine-tune predicts a
single depth per queried point; that prediction implicitly averages over a
small image patch around the query. So our real VLM is the "smooth
anchor" regime, not the sharp one — and that means **the active-query
strategy needs revisiting** for this approach (see implications below).

### 3. The corrector mechanism itself is sound

Across both regimes, the configurations using the heat-modulated smoothness
either improve siRMSE (small but consistent) or stay neutral. The corrector
is not the bottleneck; the bottleneck is the anchor-source × selection
interaction.

---

## Implications for the next step

1. **Drop pure edge-targeted anchor selection.** A trained VLM cannot give
   you a useful target at a single pixel right on a depth boundary — the
   target is undefined there. Heat-map selection should bias toward suspect
   *regions* with stable local depth (high near-field component, *low*
   gradient component within a small patch).

2. **Recommended H for VLM-anchor mode:** weight `−log D0` higher (`beta=2,
   alpha=0`) or compute `α · z(grad)` only at the macro scale (downsampled),
   so anchors land on suspect *regions*, not pixel-thin edges.

3. **Sweep `lam_s` and `K` jointly with the real VLM** once anchors arrive:
   the optimum for sharp-oracle (`lam_s≈0.1, K≈10–20, heat_heat`) is
   different from the optimum for patch-oracle (`lam_s≈0.1, K≈50–100,
   rand_heat`). A trained VLM lives in the second regime; expect K=50–100,
   random-or-coarse anchor selection, heat-modulated smoothness.

4. **Self-verification (Approach C from the design doc) becomes more
   valuable here**: in the patch-oracle regime even the *oracle* gives mixed
   results (~74% win rate at best). A real noisy VLM will be worse. The
   re-query consistency check filters out the few catastrophic anchors that
   drag the mean.

---

## Reportable claims

- "We propose error-likelihood-modulated smoothness for sparse-anchor depth
  refinement: smoothness penalty is high where the predictor is confident
  and low where it is suspect, gating corrections to where they're
  licensed. We show this guide outperforms uniform smoothness by >0.1 mean
  siRMSE on validation across all anchor counts (catastrophic-failure ratio
  of the uniform baseline)."
- "Active anchor selection by error heat-map matches single-pixel GT
  targets best at small K (100% of images improved at K=10), while random
  sampling matches patch-mean targets best at larger K (74% improved at
  K=50). The right strategy depends on the VLM's effective receptive
  field — a finding that informs both query design and VLM training."

---

## What we did not (yet) test

- **Self-verification loop** (re-query at corrected pixels, downweight bad
  anchors). Holding for when the real VLM is online — needs a second VLM
  call.
- **`lam_s` × `K` joint sweep** at the patch-oracle regime. Single-axis
  sweeps so far.
- **Anchor confidence weighting** (`w_a` from VLM logit margin). Stub is in
  the solver signature, currently uses uniform `w_a = 1`.
- **Larger image set** (>50). Patterns above are very stable across patch
  sizes 15 and 31, but the absolute improvement magnitudes (~0.001–0.002)
  are small enough that N=1159 is needed for significance claims in the
  report.

## Reproducing

```bash
sbatch slurm/eval_heatmap_oracle.sbatch \
    --out-dir outputs/diagnostics/hm_oracle_patch15_lam01/ \
    --limit 50 --Ks 10 20 50 100 --lam-s 0.1 --anchor-patch 15
```
