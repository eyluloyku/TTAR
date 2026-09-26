"""Heat-map-guided dense log-residual correction.

Given a baseline depth D0, a heat-map H of suspected error, and K sparse
anchor pixels each carrying a target log-shift Δ_a (where the corrected
depth should be exp(Δ_a) × D0(p_a)), solve for a per-pixel log-residual
δ(x,y) that minimises:

    L(δ) =   Σ_anchors w_a · (δ(p_a) − Δ_a)²                  ← data
           + λ_s · Σ_(p,q)∈4-neigh  w_smooth(p,q) · (δ(p) − δ(q))²
                                                              ← smoothness

with edge weights modulated by H:

    w_smooth(p,q) = exp( -alpha_smooth · max(relu(H(p)), relu(H(q))) )

→ smoothness is **strong** in confident regions (H ≤ 0, weight = 1, δ stays
flat) and **weak** in suspect regions (H > 0, weight ≪ 1, anchors dominate).

The siRMSE-free direction (per-image log shift) is handled post-hoc by
subtracting mean(δ); penalising it inside the loss would waste regularisation
budget on a dimension the metric strips.

Returns the corrected depth D_refined = D0 · exp(δ − mean δ); the per-image
mean of δ is stripped post-hoc (see above) so the correction is siRMSE-invariant.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS = 1e-6


def _cg(A, b, tol: float, maxiter: int):
    # scipy >=1.12 renamed cg's `tol` kwarg to `rtol`; support both.
    try:
        return spla.cg(A, b, rtol=tol, maxiter=maxiter)
    except TypeError:
        return spla.cg(A, b, tol=tol, maxiter=maxiter)


def _build_graph_laplacian(
    H: np.ndarray,
    alpha_smooth: float,
    rgb: np.ndarray | None = None,
    alpha_rgb: float = 0.0,
) -> sp.csr_matrix:
    """4-connected weighted graph Laplacian from heat-map H.

    Edge weight:
        w(p,q) = exp( -alpha_smooth * max(relu(H_p), relu(H_q))
                      -alpha_rgb    * ||RGB(p) - RGB(q)||_1 / (3*255) )
    The RGB term (joint-bilateral) preserves smoothness across image-flat
    neighbours and drops it across real image edges, independent of the heat
    map. With `alpha_rgb=0` (default) the behaviour is identical to the
    heat-only formulation.
    """
    Hh, Ww = H.shape
    N = Hh * Ww
    Hf = np.maximum(H.astype(np.float32), 0.0)  # relu(H)

    if rgb is not None and alpha_rgb > 0.0:
        rgb_f = rgb.astype(np.float32)
        if rgb_f.ndim != 3 or rgb_f.shape[:2] != H.shape or rgb_f.shape[2] != 3:
            raise ValueError(
                f"rgb must be [H, W, 3] matching H.shape; got {rgb_f.shape}"
            )
        rgb_h = np.abs(rgb_f[:, :-1, :] - rgb_f[:, 1:, :]).sum(axis=2) / (3.0 * 255.0)
        rgb_v = np.abs(rgb_f[:-1, :, :] - rgb_f[1:, :, :]).sum(axis=2) / (3.0 * 255.0)
    else:
        rgb_h = 0.0
        rgb_v = 0.0

    # Horizontal edges (i, j) — (i, j+1)
    wh = np.exp(
        -alpha_smooth * np.maximum(Hf[:, :-1], Hf[:, 1:]) - alpha_rgb * rgb_h
    )
    # Vertical edges (i, j) — (i+1, j)
    wv = np.exp(
        -alpha_smooth * np.maximum(Hf[:-1, :], Hf[1:, :]) - alpha_rgb * rgb_v
    )

    rows_list = []
    cols_list = []
    data_list = []
    diag = np.zeros(N, dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(Hh), np.arange(Ww - 1), indexing="ij")
    p = (yy * Ww + xx).ravel()
    q = (yy * Ww + xx + 1).ravel()
    w = wh.ravel()
    rows_list += [p, q]
    cols_list += [q, p]
    data_list += [-w, -w]
    np.add.at(diag, p, w)
    np.add.at(diag, q, w)

    yy, xx = np.meshgrid(np.arange(Hh - 1), np.arange(Ww), indexing="ij")
    p = (yy * Ww + xx).ravel()
    q = ((yy + 1) * Ww + xx).ravel()
    w = wv.ravel()
    rows_list += [p, q]
    cols_list += [q, p]
    data_list += [-w, -w]
    np.add.at(diag, p, w)
    np.add.at(diag, q, w)

    L = sp.coo_matrix(
        (np.concatenate(data_list).astype(np.float32),
         (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N, N),
    ).tocsr()
    L = L + sp.diags(diag.astype(np.float32))
    return L


def solve_dense_correction(
    D0: np.ndarray,
    H: np.ndarray,
    anchors_yx: Sequence[tuple[int, int]],
    anchor_dlog: Sequence[float],
    anchor_weights: Sequence[float] | None = None,
    lam_s: float = 1.0,
    alpha_smooth: float = 2.0,
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
    ridge: float = 1e-6,
    rgb: np.ndarray | None = None,
    alpha_rgb: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Solve the heat-map-guided dense correction system.

    Returns (D_refined, info) where info has keys:
        - 'cg_info': scipy CG flag (0 = converged)
        - 'delta_mean': mean of δ before zero-mean strip
        - 'delta_std': std of δ
        - 'n_anchors': number of anchors used
    """
    if not isinstance(D0, np.ndarray) or D0.ndim != 2:
        raise ValueError("D0 must be a 2-D numpy array")
    Hh, Ww = D0.shape
    N = Hh * Ww

    if len(anchors_yx) == 0:
        return D0.astype(np.float32).copy(), {
            "cg_info": 0, "delta_mean": 0.0, "delta_std": 0.0, "n_anchors": 0,
        }

    L = _build_graph_laplacian(H, alpha_smooth, rgb=rgb, alpha_rgb=alpha_rgb)

    if anchor_weights is None:
        anchor_weights = [1.0] * len(anchors_yx)

    anchor_idx = np.array([y * Ww + x for y, x in anchors_yx], dtype=np.int64)
    aw = np.asarray(anchor_weights, dtype=np.float32)
    ad = np.asarray(anchor_dlog, dtype=np.float32)

    D_diag = np.zeros(N, dtype=np.float32)
    np.add.at(D_diag, anchor_idx, aw)

    A = sp.diags(D_diag.astype(np.float32)) + (lam_s * L)
    if ridge > 0:
        A = A + sp.eye(N, dtype=np.float32) * ridge

    b = np.zeros(N, dtype=np.float32)
    np.add.at(b, anchor_idx, aw * ad)

    delta_flat, info = _cg(A, b, cg_tol, cg_max_iter)
    delta = delta_flat.reshape(Hh, Ww).astype(np.float32)
    delta_mean = float(delta.mean())
    delta_std = float(delta.std())
    delta -= delta_mean  # siRMSE strips per-image log shift

    D_refined = (D0.astype(np.float32) * np.exp(delta)).astype(np.float32)

    return D_refined, {
        "cg_info": int(info),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "n_anchors": int(len(anchors_yx)),
    }


def smooth_only_correction(
    D0: np.ndarray,
    H: np.ndarray,
    lam_s: float = 1.0,
    alpha_smooth: float = 2.0,
    data_alpha: float = 2.0,
    cg_max_iter: int = 200,
    cg_tol: float = 1e-5,
    ridge: float = 1e-6,
    rgb: np.ndarray | None = None,
    alpha_rgb: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """K=0 baseline: heat-modulated edge-aware smoothing of D0 itself.

    No external anchors. Confident pixels (H ≤ 0) stay near log D0; suspect
    pixels (H > 0) get pulled toward their neighbours through the same
    heat-modulated smoothness term used by `solve_dense_correction`.

    Solves in log space:
        min_u  Σ_p w_data(p) · (u(p) − log D0(p))²
             + λ_s · Σ_(p,q) w_smooth(p,q) · (u(p) − u(q))²
    with
        w_data(p)      = exp(−data_alpha · relu(H(p)))
        w_smooth(p,q)  = exp(−alpha_smooth · max(relu(H_p), relu(H_q)))

    Tests whether the smoothness prior alone — without any external "VLM" or
    oracle sensor — captures most of the corrector's headroom. If it does,
    the contribution shifts from "VLM-anchored refinement" to
    "self-uncertainty-aware smoothing".
    """
    if not isinstance(D0, np.ndarray) or D0.ndim != 2:
        raise ValueError("D0 must be a 2-D numpy array")
    Hh, Ww = D0.shape
    N = Hh * Ww

    log_d0 = np.log(np.maximum(D0, EPS)).astype(np.float32)
    w_data = np.exp(-data_alpha * np.maximum(H.astype(np.float32), 0.0)).ravel()

    L = _build_graph_laplacian(H, alpha_smooth, rgb=rgb, alpha_rgb=alpha_rgb)
    A = sp.diags(w_data) + (lam_s * L)
    if ridge > 0:
        A = A + sp.eye(N, dtype=np.float32) * ridge
    b = (w_data * log_d0.ravel()).astype(np.float32)

    u_flat, info = _cg(A, b, cg_tol, cg_max_iter)
    u = u_flat.reshape(Hh, Ww).astype(np.float32)
    delta = u - log_d0
    delta_mean = float(delta.mean())
    delta_std = float(delta.std())
    delta -= delta_mean

    D_refined = (D0.astype(np.float32) * np.exp(delta)).astype(np.float32)

    return D_refined, {
        "cg_info": int(info),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "n_anchors": 0,
    }
