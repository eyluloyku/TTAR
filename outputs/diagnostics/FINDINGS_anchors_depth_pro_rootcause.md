# DepthPro anchor-source — per-pixel root-cause

Tests README §7.2's claim that DepthPro fails *on the same pixels* and *in the
same direction* as Stage A, especially at heat-map-selected anchor pixels.

**Sample:** N=1159 val images. K=3000 anchors per image,
heat-greedy with min_dist=24px (TTAR's anchor selector); same-K random control.

## Per-image medians (aggregated over images)

| Statistic | At heat pixels | At random pixels |
|---|---:|---:|
| Stage A median \|log err\| | 0.372 | 0.110 |
| DepthPro median \|log err\| | 1.067 | 0.693 |
| Stage A mean \|log err\|   | 0.975 | 0.203 |
| DepthPro mean \|log err\|   | 1.749 | 0.753 |
| Pearson(signed err D0, err DP) | 0.941 | 0.903 |
| Same-sign rate | 0.621 | 0.556 |

## Reading

- **Heat-map pixels are hard for both models.** Stage A median |log err| is
  0.372 at heat vs 0.110
  at random (×3.4);
  DepthPro median |log err| is 1.067 vs
  0.693 (×1.5).
- **Same-sign rate at heat pixels = 0.62** — the two models err in
  the same direction on the majority of heat-selected pixels.
- **Pearson(D0 err, DP err) at heat pixels = 0.94** — signed errors
  are correlated, consistent with shared structural causes.

This quantifies §7.2's "root cause": at the pixels TTAR queries, DepthPro
inherits Stage A's failure modes, so its anchor values cannot pull the prediction
toward the GT.
