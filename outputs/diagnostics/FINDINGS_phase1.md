# Stage B v3 — Phase 1 findings (oracle corrector, N=1159 val)

**Run:** [`hm_oracle_v3_phase1/`](hm_oracle_v3_phase1/) — `--Ks 0 10 20 50 100 --lam-s 0.1 --anchor-patch 15`, 3h32m on the 5060-ti env.

This file extends [`FINDINGS_corrector_oracle.md`](FINDINGS_corrector_oracle.md) (C1, N=50) with three additions that the v3 plan introduced: (i) K=0 smoothness-only baseline, (ii) tail-10% / tail-25% subset metrics, (iii) full-val scale.

---

## Headline numbers

Baseline siRMSE: mean **0.4612**, median 0.4461. Tail-10 baseline mean **0.7765** (1.68× the full-mean — a real catastrophic-failure subset exists).

| Cell | Full-mean Δ | Tail-10 Δ | Tail-10 win-rate | Full win-rate |
|---|---:|---:|---:|---:|
| `rand_unif K=100` (random anchors, **uniform** smoothness) | −0.0017 | **−0.0061** | **77.6%** | 59.6% |
| `rand_heat K=100` (random anchors, heat-modulated smoothness) | **−0.0019** | −0.0038 | 70.7% | 67.8% |
| `heat_heat K=100` (heat anchors, heat smoothness — v2 "ours") | +0.0056 | −0.0012 | 63.8% | 33.6% |
| `heat_heat K=0` (smoothness-only with heat) | +0.0086 | +0.0096 | 12.1% | 10.7% |
| `*_unif K=0` (smoothness-only with uniform) | +0.0000 | +0.0000 | 31.0% | 46.6% |

Numbers are Δ-siRMSE relative to D0 (negative = improvement). `*_unif K=0` is mathematically a near-no-op (CG converges to δ≈0 since the smoothness term has no source).

---

## Three findings

### 1. K=0 is dead — the anchors do real work

Smoothness-only correction *without external anchors* fails decisively:

- `heat_heat K=0` actively *worsens* D0 by Δ = +0.0086 mean (+0.0096 on tail-10). Heat-modulated smoothing pulls suspect-region pixels toward their neighbours *across true depth discontinuities*, destroying genuine boundaries.
- `*_unif K=0` is a no-op (Δ ≈ 0): with uniform smoothness and no anchor source there's nothing to drive the solution away from δ = 0.

The stress-test hypothesis that "smoothness-only captures most of the gain" — Phase 1b's reason to exist — is **decisively refuted**. The whole signal in the corrector comes from the anchors, not from the smoothness prior alone. This is a clean negative result that justifies the entire Phase 4 onward (anchor sources matter; smoothing alone doesn't).

### 2. Heat-anchor selection is worse than random at full val

Every `heat_*` row underperforms its `rand_*` counterpart for K ≥ 10:

| K | `heat_heat` Δ | `rand_heat` Δ | Difference |
|---:|---:|---:|---:|
| 10 | +0.0007 | +0.0003 | rand wins |
| 20 | +0.0013 | +0.0002 | rand wins |
| 50 | +0.0059 | −0.0006 | rand wins |
| 100 | +0.0056 | −0.0019 | rand wins |

Heat-anchor selection (greedy top-K on `H = α·z(|∇log D0|) + β·z(−log D0)`) lands disproportionately on depth edges, where the patch-median oracle (the smooth-VLM proxy at `--anchor-patch 15`) is ill-defined: the median across a depth boundary mixes near and far surfaces. With random anchor placement, anchors land in informative *interiors* of regions and the corrector has well-posed local data.

This was flagged in [`FINDINGS_corrector_oracle.md`](FINDINGS_corrector_oracle.md) at N=50 and is now confirmed at N=1159 with much larger gap. **The "active query selection" idea from C0 was correct only in the sharp single-pixel-oracle regime, which is not the regime a real VLM operates in.**

### 3. Heat-modulated smoothness loses to uniform smoothness on the tail

This is the most uncomfortable finding. The C1 doc's central claim — "heat-modulated smoothness is essential, uniform is catastrophic" — only holds *when the anchors are also heat-driven*:

- `heat_unif K=100`: +0.1221 (catastrophic, as C1 claimed).
- `rand_unif K=100`: −0.0017 full / **−0.0061 tail-10**.
- `rand_heat K=100`: −0.0019 full / −0.0038 tail-10.

When anchors are scattered randomly, **uniform smoothness wins on the tail by ~60% in magnitude** (and ties on full-mean). The "catastrophic uniform" story from C1 is an artefact of pairing uniform smoothness with heat-anchor selection: in that pairing, both the anchor selection and the smoothness prior concentrate at edges, and they fight each other across the boundary. Decouple them (random anchors + uniform smoothness) and uniform smoothness is fine.

**Implication for the report.** The v2 README framed heat-modulated smoothness as the "single most important design choice and the cleanest publishable claim." Phase 1c says that claim survives only as a *qualified* one: heat-modulated smoothness gives marginally better full-mean (−0.0019 vs −0.0017) but loses on the tail. The novelty has to move to either (i) the *anchor source* axis (Phase 4 onward), (ii) the *selective-application* axis (Phase 3), or (iii) an *epistemic* version of the heat map (Phase 2 TTA-variance — TBD).

---

## What this means for the remaining phases

| Phase | Status change |
|---|---|
| **Phase 1c** | Done. Decision rule answered: K=0 does not capture gain; anchors are central. |
| **Phase 2 (TTA-var)** | More important than before. The heat-map's role is in question; a *true* epistemic signal might revive it. If TTA-var is weak, the heat map becomes a minor component, not the headline. |
| **Phase 3 (selective gate)** | Still high-ROI. The best cell (`rand_unif K=100`) only improves 60% of full-val images. Gating out the 40% that regress should boost full-mean materially. |
| **Phase 4 (anchor sources)** | Now the *headline* contribution. Phase 1c shows the gain lives entirely in the anchors; comparing oracle / second-model / VLM is the central research question. |
| **Phase 5 (VLM)** | Depends on C2. Unchanged. |
| **Phase 6 (verifier)** | Still optional polish. Cross-source verification only meaningful once Phase 4 produces multi-source anchors. |

---

## What the report leads with (post-Phase-1c, pre-Phase-2)

The honest framing of the project at this point:

1. **The Stage B v3 corrector improves DA-V2-LoRA on the catastrophic-tail subset by ~0.8% siRMSE relative (Δ = −0.006 on tail-10, win-rate 78%).** Full-mean gain is smaller (~0.4%).
2. **The corrector's mechanism is sparse-anchor sparse-to-dense interpolation; the smoothness prior contributes ~0 on its own.** Anchors are the load-bearing component.
3. **Heat-map-based anchor *selection* fails in the smooth-VLM regime** because greedy top-K on D0's gradient concentrates anchors at depth boundaries where any patch-aggregated oracle is ill-defined. Random anchor placement is the working baseline.
4. **The "heat-modulated smoothness" claim from v2 is qualified**: it is essential when paired with heat-anchor selection (otherwise the two concentrate at the same edges and the system breaks); decoupled from heat-anchor selection it is comparable to or slightly worse than uniform smoothness.

The remaining open question — answered by Phase 2 — is whether a *true* epistemic uncertainty signal (TTA log-depth variance) can revive heat-anchor selection. If yes, the v2 method is largely vindicated with one corrected component. If no, the project's novelty consolidates around Phase 4 (anchor sources) + Phase 3 (selective application).
