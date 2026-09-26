"""One-off probe: characterize D0's gross/catastrophic errors."""
import glob, os
import numpy as np
from scipy.ndimage import label, distance_transform_edt

EPS = 1e-6
root = '/cluster/courses/cil/monocular-depth-estimation/train'
preds = sorted(glob.glob('outputs/val_preds/vits_lora_r8/*.npy'))[:120]

agg = dict(n=0, gross_pix=[], over=[], under=[], rank_over=[], rank_under=[],
           iso_frac=[], d_sky=[], warp_unfix=[], big_cc_frac=[], maxd=[])
for pf in preds:
    stem = os.path.basename(pf)[:-4]
    gp = f'{root}/{stem}_depth.npy'
    if not os.path.exists(gp):
        continue
    D0 = np.load(pf).astype(np.float32)
    gt = np.load(gp).astype(np.float32)
    if D0.shape != gt.shape:
        continue
    valid = (gt > 0) & (D0 > 0) & np.isfinite(D0) & np.isfinite(gt)
    if valid.sum() < 1000:
        continue
    ld0 = np.log(np.maximum(D0, EPS)); lgt = np.log(np.maximum(gt, EPS))
    e = ld0 - lgt
    e = e - e[valid].mean()
    gross = valid & (np.abs(e) > 1.0)
    ng = int(gross.sum())
    if ng == 0:
        continue
    agg['n'] += 1
    agg['gross_pix'].append(ng / valid.sum())
    over = gross & (e > 0)        # D0 says relatively FARTHER than GT
    under = gross & (e < 0)       # D0 says relatively NEARER than GT
    agg['over'].append(over.sum() / ng)
    agg['under'].append(under.sum() / ng)
    agg['maxd'].append(float(gt[valid].max()))

    # GT-depth rank (0=nearest, 1=farthest) of gross pixels, by sign
    gvs = np.sort(gt[valid])
    rank = lambda m: (np.searchsorted(gvs, gt[m]) / len(gvs)) if m.sum() else np.array([])
    ro, ru = rank(over), rank(under)
    if len(ro): agg['rank_over'].append(float(np.median(ro)))
    if len(ru): agg['rank_under'].append(float(np.median(ru)))

    # connected components of the gross mask: isolated (<=2 px) vs big (>=50)
    lab, nlab = label(gross)
    if nlab:
        sizes = np.bincount(lab.ravel())[1:]
        per_pix = sizes[lab[gross] - 1]
        agg['iso_frac'].append(float((per_pix <= 2).mean()))
        agg['big_cc_frac'].append(float((per_pix >= 50).mean()))

    # distance from gross pixels to the nearest invalid (sky/far) pixel
    d_inv = distance_transform_edt(valid)
    agg['d_sky'].append(float(np.median(d_inv[gross])))

    # can a +-8px local re-sample fix it? min |ld0(q)-lgt(p)| over window
    best = np.full(D0.shape, np.inf, np.float32)
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            pad = 8
            ap = np.pad(ld0, pad, mode='edge')
            cand = ap[pad+dy:pad+dy+D0.shape[0], pad+dx:pad+dx+D0.shape[1]]
            best = np.minimum(best, np.abs(cand - lgt))
    agg['warp_unfix'].append(float((best[gross] > 1.0).mean()))

m = lambda k: float(np.mean(agg[k])) if agg[k] else float('nan')
print(f"images with gross errors: {agg['n']}/{len(preds)}")
print(f"gross pixel fraction (of valid):     {m('gross_pix')*100:.1f}%")
print(f"  of gross: D0 says FARTHER (over):   {m('over')*100:.0f}%")
print(f"            D0 says NEARER  (under):  {m('under')*100:.0f}%")
print(f"GT-depth rank of OVER  gross pixels:  {m('rank_over'):.2f}  (0=near,1=far)")
print(f"GT-depth rank of UNDER gross pixels:  {m('rank_under'):.2f}")
print(f"gross in isolated CCs (<=2 px):       {m('iso_frac')*100:.0f}%   <- GT noise if high")
print(f"gross in big CCs (>=50 px):           {m('big_cc_frac')*100:.0f}%   <- region error if high")
print(f"median dist of gross pixel to sky:    {m('d_sky'):.1f} px")
print(f"gross unfixable by +-8px re-sample:   {m('warp_unfix')*100:.0f}%")
print(f"mean image max GT depth:              {m('maxd'):.1f} m")
