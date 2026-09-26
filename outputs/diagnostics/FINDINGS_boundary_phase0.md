# Stage B v5 — Phase 0a findings (boundary error-mode diagnostic)

First phase of the VLM-Guided Boundary Relocalization plan
([`README_STAGE_B_V5.md`](../../README_STAGE_B_V5.md)). Pure-measurement,
no new model. Answers two of the three Phase-0 go/no-go questions; Q0.2
(SAM boundary quality) is still outstanding.

**Run:** [`boundary_phase0/`](boundary_phase0/), N=1159 val images.
Code: [`scripts/diagnose_boundary_errormode.py`](../../scripts/diagnose_boundary_errormode.py),
[`scripts/probe_gross_errors.py`](../../scripts/probe_gross_errors.py),
[`slurm/diagnose_boundary_errormode.sbatch`](../../slurm/diagnose_boundary_errormode.sbatch).

---

## TL;DR

| What | Number |
|---|---:|
| Baseline siRMSE (Stage A, full val) | 0.4612 |
| Error² mass that is **gross** (\|log err\| > 1.0 — depth off > 2.7×) | **74%** |
| Moderate-error mass within ≤8 px of a true depth discontinuity | **93%** (1.69× concentration) |
| Error concentration in the immediate ≤2 px band (`gt` edges) | **3.8×** |
| Same for **RGB** edges (what the existing guided filter keys on) | **1.0× — none** |
| Honest relocalization ceiling (warp ±2 / ±4 / ±8 px) | **−0.032 / −0.045 / −0.061** |
| v3 sparse-anchor corrector oracle ceiling (for comparison) | −0.002 |

**Phase 0a verdict: GO.** D0's *moderate* error is genuinely concentrated at
depth discontinuities, the relocalization ceiling is ~20× the v3 anchor-corrector
oracle, and RGB edges are confirmed useless as a localizer — which both supports
the v5 thesis and explains the May-19 `rgb_n50` null result. Q0.2 (are SAM
boundaries well enough localized to *realize* this ceiling) is still required
before committing.

---

## The error splits in two

D0's per-pixel siRMSE error decomposes into two populations with completely
different character. Squared log-error mass:

- **74% is "gross"** — \|log err\| > 1.0, i.e. depth wrong by more than 2.7×.
- **26% is "moderate"** — the rest.

These need separate treatment, so the diagnostic measures them separately
(`e2` = raw squared error, `e2c` = error clipped to ±1.0 then squared).

### The gross 74% — sky-depth bleed onto thin foreground

120-image probe ([`probe_gross_errors.py`](../../scripts/probe_gross_errors.py)):

| Property of gross pixels | Value |
|---|---:|
| Share of valid pixels | 2.8% |
| Direction: "D0 predicts farther than GT" | **91%** |
| GT-depth rank of those (0 = nearest, 1 = farthest) | **0.01** |
| Median distance to the nearest invalid (sky) pixel | **1.0 px** |
| In isolated ≤2 px components | 38% |
| Not recoverable by a ±8 px local re-sample | **71%** |

The dominant pattern: the **nearest objects in the scene, silhouetted against
sky, which D0 predicts as far.** Two mechanisms:

1. **Sky-depth bleed.** Sky has no GT returns → it is masked out of the Stage A
   loss → D0's sky depth is unconstrained (predicted far). DA-V2 **ViT-S** has a
   low-resolution DPT decoder; where a thin near object (pole, wire, branch,
   building edge) is silhouetted against sky, the decoder bleeds the far sky
   depth across the thin object. The object is thinner than the decoder's
   effective resolution, so its true near depth is **never represented anywhere**.
2. **GT label noise.** 38% of gross pixels are isolated 1–2 px specks — depth-
   sensor *mixed pixels* at silhouette boundaries, where the label itself is
   unreliable.

**Why relocalization cannot touch this.** Relocalization re-arranges D0's
*existing* values — it slides a blurry edge onto the true boundary. For
mechanism 1 the correct near value was never predicted (71% have nothing usable
within ±8 px) — recovering it needs a stronger backbone or an external depth
source. For mechanism 2 there is no correct answer to recover; "fixing" it would
be fitting label noise. The gross 74% is therefore **out of scope for v5 by
construction** — and the Q0.3 ceiling below already excludes it.

### The moderate 26% — boundary-concentrated, as hypothesised

`e2c` mass vs distance to a true (`gt`) depth discontinuity, full val:

| Distance band | [0,2) | [2,4) | [4,8) | [8,16) | [16,∞) |
|---|---:|---:|---:|---:|---:|
| error mass % | **80.2** | 7.0 | 6.2 | 4.3 | 2.4 |
| valid-pixel area % | 21.1 | 13.6 | 20.5 | 23.4 | 21.4 |
| concentration (mass/area) | **3.8×** | 0.5× | 0.3× | 0.2× | 0.1× |

The moderate error collapses onto the immediate ±2 px band — 80% of it on 21%
of pixels. Beyond 8 px the error is *below* uniform density. This is the
signature v5 targets.

---

## Q0.1 — which edge map localizes the error?

Concentration in the immediate [0,2) band, by discontinuity definition:

| Edge definition | mass % | area % | concentration |
|---|---:|---:|---:|
| `gt`  — true \|∇log GT\| + sky border | 80.2 | 21.1 | **3.8×** |
| `d0`  — D0's own \|∇log D0\| | 51.0 | 17.0 | **3.0×** |
| `rgb` — grayscale RGB gradient | 44.0 | 43.3 | **1.0×** |

**Two findings:**

1. **RGB edges carry no error-localization signal** (mass% ≈ area% in every
   bin). This is the hard-number explanation of why the May-19 `rgb_n50`
   `alpha_rgb` sweep was negligible: the photometric guide the existing
   [`guided_filter.py`](../../src/refine/guided_filter.py) keys on simply does
   not coincide with where D0 is wrong. Texture edges ≠ depth edges.
2. **D0's own edges are a usable test-time proxy (3.0×).** They are blurry and
   mislocalized — that is the thing v5 fixes — but as a *region to apply
   correction in*, `|∇log D0|` is 3× concentrated and needs no GT. The method
   can bootstrap from D0's own edges and snap them to SAM boundaries.

---

## Q0.3 — relocalization ceiling

`warp w` = Δ-siRMSE if every pixel is re-sampled from the best-matching D0 value
within ±w px (chosen against GT). This honestly bounds boundary relocalization:
it can slide a misplaced edge a few pixels, but gross errors survive because no
good value exists nearby.

| Ceiling | full | tail-10% |
|---|---:|---:|
| warp ±2 px | **−0.0324** | −0.0251 |
| warp ±4 px | **−0.0452** | −0.0405 |
| warp ±8 px | **−0.0614** | −0.0629 |
| *perfect band-fix (`gt`, d=8) — loose ref. only* | −0.380 | −0.665 |

- The ceiling is **−0.045 to −0.061** — roughly **10–13% relative** on the 0.461
  baseline, and **~20–30× the v3 sparse-anchor corrector's oracle ceiling**
  (−0.002). The headroom v3 never had is real and it is here.
- **~70% of the headroom lies within ±2 px** (−0.032 of −0.045). The
  mislocalization is small-displacement — the method needs a *narrow* boundary
  band (r ≈ 2–4 px), not large warps.
- The ceiling holds up on the catastrophic tail (−0.063 at ±8 px), unlike the
  v3 corrector which was near-neutral there.

A realistic segment-guided solver will recover only a fraction of −0.045 — but
even a third would be −0.015, an order of magnitude past anything v3/v4
delivered. And segment-guided propagation reaches across a whole region, not
just ±w px, so it may exceed the local-warp ceiling on thin connected objects
(hypothesis for Phase 1).

---

## Decision

| Gate | Rule | Result |
|---|---|---|
| Q0.1 | moderate error concentrated at depth edges (>35% ≤8 px, conc >1.3×) | **PASS** (93%, 1.69×) |
| Q0.3 | relocalization ceiling ≤ −0.006 full-mean | **PASS** (−0.045) |
| Q0.2 | SAM boundaries localize true discontinuities — *not yet measured* | **PENDING** |

**Phase 0a is a GO.** Proceed to Q0.2: run SAM on val and measure whether SAM
region boundaries are better localized to true depth discontinuities than D0's
own edges (displacement, precision/recall, under-segmentation). Q0.2 decides
whether SAM's partition is good enough to *realize* the −0.045 ceiling, or
whether the segmenter must be upgraded (SAM2 / vit-h) first.

## Implications for the report

- The error budget is now honestly decomposed: **~74% gross** (≈irreducible —
  GT noise + ViT-S capacity on thin foreground) **+ ~26% moderate**, of which the
  boundary-concentrated part is v5's target. Any siRMSE method on this dataset
  is capped by the gross 74%; the full-val headline number will be modest by
  construction, and that is a finding, not a failure.
- "RGB-edge guidance does not localize depth error (1.0×)" is a clean,
  quotable negative result that motivates the semantic-boundary design.
