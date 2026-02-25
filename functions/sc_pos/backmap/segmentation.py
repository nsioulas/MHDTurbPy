from __future__ import annotations

"""sc_pos.backmap.segmentation

Vectorized, physics-motivated source-regime segmentation.

Why the old approach fails (first principles)
--------------------------------------------
A same-source interval is NOT defined by "low turbulence". It is defined by a
*stationary multivariate physical state* on a chosen window. A single scalar
"variability score" inevitably confuses:
  - large-amplitude Alfv\'enic fluctuations (high variance but *same source*)
  - genuine state changes (new stream / HCS crossing / compressive structure)

This module implements a regime-change detector in a multivariate feature
space built from physically interpretable diagnostics, and scores changes using
an *explicit ridge-regularized Mahalanobis metric*.

Core method
-----------
1) Build a feature vector z(t) from multiple physical diagnostics.
   For each diagnostic x(t), include:
     - baseline: rolling median  \tilde{x}(t)
     - constancy: rolling MAD scaled in a way that does not blow up at median~0

2) Robustly standardize each feature using median/MAD over the full interval.

3) Compute a two-sided mean-shift vector in feature space:
       d(i) = mean[z(i:i+w)] - mean[z(i-w:i)]

4) Convert d(i) into a scalar change score using a ridge-regularized
   Mahalanobis norm:
       S(i) = sqrt( d^T (C + \lambda I)^{-1} d ) / sqrt(p)
   where C is the covariance of standardized features and
       \lambda = ridge_alpha * tr(C)/p.

5) Stable points satisfy S(i) <= threshold (threshold is robustly estimated
   from the score distribution unless the user provides one).

Optional ML regularization
--------------------------
If mode="gmm_cpd", a GaussianMixture is fit in feature space to suppress label
flicker and optionally split stable spans into distinct regimes.

Public function
---------------
`segment_sources(...)` returns:
  - score: multivariate change score time series (low = stable)
  - segment: integer labels for stable segments (unstable = -1)
  - score_components: (kept for backward compatibility; usually empty)
  - meta: bookkeeping, including which diagnostics were used.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SegmentationResult:
    score: pd.Series
    segment: np.ndarray
    score_components: Dict[str, pd.Series]
    meta: Dict[str, Any]


# ---------------------------------------------------------------------
# Robust helpers
# ---------------------------------------------------------------------

def _nan_mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median absolute deviation, NaN-safe."""
    med = np.nanmedian(x, axis=axis)
    return np.nanmedian(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)


def _global_mad(x: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float)
    xx = xx[np.isfinite(xx)]
    if xx.size == 0:
        return float("nan")
    m = float(np.nanmedian(xx))
    return float(np.nanmedian(np.abs(xx - m)))


def _estimate_dt_seconds(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return float("nan")
    t = pd.to_datetime(index)
    dt = (t[1:] - t[:-1]).total_seconds().to_numpy(dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return float("nan")
    return float(np.nanmedian(dt))


def _window_npoints(index: pd.DatetimeIndex, window: str, *, floor: int = 3) -> int:
    dt = _estimate_dt_seconds(index)
    try:
        wsec = float(pd.Timedelta(str(window)).total_seconds())
    except Exception:
        return int(floor)
    if (not np.isfinite(dt)) or (dt <= 0.0) or (not np.isfinite(wsec)) or (wsec <= 0.0):
        return int(floor)
    return int(max(int(floor), int(round(wsec / dt))))


def _rolling_med_mad(series: pd.Series, window: str, min_periods: int) -> Tuple[pd.Series, pd.Series]:
    x = pd.to_numeric(series, errors="coerce")
    med = x.rolling(str(window), center=True, min_periods=int(min_periods)).median()
    mad = (x - med).abs().rolling(str(window), center=True, min_periods=int(min_periods)).median()
    return med, mad


def _robust_standardize(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    mad = _nan_mad(X, axis=0)
    scale = mad + float(eps)
    Z = (X - med[None, :]) / scale[None, :]
    return Z, med, scale


# ---------------------------------------------------------------------
# Physics-driven feature handling
# ---------------------------------------------------------------------

def _infer_kind(varname: str) -> str:
    """Classify variable scaling to avoid pathological normalization."""
    v = str(varname).strip().lower()
    if v in {"sigma_c", "sigma_r"}:
        return "bounded"
    if (v == "br_r2") or ("br_r2" in v) or (v == "br"):
        return "signed_abs"
    return "positive"


def _ensure_basic_derived(data: pd.DataFrame) -> None:
    """Best-effort derived diagnostics if the pipeline did not compute them.

    This is deliberately minimal: constants (m_p) do not matter for segmentation
    after robust standardization, so we use proportional proxies.
    """
    # Bmag
    if ("Bmag" not in data.columns) and all(c in data.columns for c in ("Br", "Bt", "Bn")):
        Br = pd.to_numeric(data["Br"], errors="coerce")
        Bt = pd.to_numeric(data["Bt"], errors="coerce")
        Bn = pd.to_numeric(data["Bn"], errors="coerce")
        data["Bmag"] = np.sqrt(Br * Br + Bt * Bt + Bn * Bn)

    # Choose a speed column
    vcol = "Vr_bg" if "Vr_bg" in data.columns else ("Vr" if "Vr" in data.columns else None)

    # Choose a radius column for r^2-scaled fluxes. Units do not matter here:
    # segmentation later robust-standardizes every feature.
    rcol = "r_sc" if "r_sc" in data.columns else ("Dist_au" if "Dist_au" in data.columns else None)

    # mass_flux ~ Np * Vr
    if ("mass_flux" not in data.columns) and ("Np" in data.columns) and (vcol is not None):
        Np = pd.to_numeric(data["Np"], errors="coerce")
        Vr = pd.to_numeric(data[vcol], errors="coerce")
        data["mass_flux"] = Np * Vr

    # mass_flux_r2 ~ (Np * Vr) * r^2 (spherical expansion proxy)
    if ("mass_flux_r2" not in data.columns) and ("mass_flux" in data.columns) and (rcol is not None):
        mf = pd.to_numeric(data["mass_flux"], errors="coerce")
        r = pd.to_numeric(data[rcol], errors="coerce")
        data["mass_flux_r2"] = mf * (r * r)

    # P_ram ~ Np * Vr^2
    if ("P_ram" not in data.columns) and ("Np" in data.columns) and (vcol is not None):
        Np = pd.to_numeric(data["Np"], errors="coerce")
        Vr = pd.to_numeric(data[vcol], errors="coerce")
        data["P_ram"] = Np * (Vr * Vr)

    # thermal pressure proxy: P_th ~ Np * Tp
    if ("P_th" not in data.columns) and ("Np" in data.columns) and ("Tp" in data.columns):
        Np = pd.to_numeric(data["Np"], errors="coerce")
        Tp = pd.to_numeric(data["Tp"], errors="coerce")
        data["P_th"] = Np * Tp

    # Br_r2 ~ Br * r_sc^2
    if ("Br_r2" not in data.columns) and ("Br" in data.columns) and ("r_sc" in data.columns):
        Br = pd.to_numeric(data["Br"], errors="coerce")
        r = pd.to_numeric(data["r_sc"], errors="coerce")
        data["Br_r2"] = Br * (r * r)

    # mag_mass_flux ~ mass_flux / |Br|
    if ("mag_mass_flux" not in data.columns) and ("mass_flux" in data.columns) and ("Br" in data.columns):
        mf = pd.to_numeric(data["mass_flux"], errors="coerce")
        Br = pd.to_numeric(data["Br"], errors="coerce")
        data["mag_mass_flux"] = mf / np.maximum(np.abs(Br), 1e-30)


def _auto_select_vars(data: pd.DataFrame) -> Sequence[str]:
    """Select a balanced, physically interpretable diagnostic set.

    The goal is *not* to throw every column into the model. It is to include
    complementary constraints on:
      (i) expansion-scaled flux proxies,
     (ii) bulk flow/thermodynamic state,
    (iii) pressure/flux transport,
     (iv) optional turbulence state.
    """
    # ensure minimal derived diagnostics exist (proxies are fine here)
    _ensure_basic_derived(data)

    cols = set(map(str, data.columns))

    # Priority-ordered groups (first principles)
    # (i) Expansion / transport proxies
    invariants = [
        "Br_r2",
        "mass_flux_r2",
        "mass_flux",
        "P_ram",
        "mag_mass_flux",
    ]

    # (ii) Bulk / thermodynamic state
    bulk_state = [
        "Vr_bg", "Vr",
        "Np",
        "Bmag",
        "beta",
        "Tp", "P_th",
    ]

    # (iii) Optional turbulence state (bounded)
    turbulence = ["sigma_c", "sigma_r"]

    out: list[str] = []

    def add_first_present(cands: Sequence[str], *, max_add: Optional[int] = None) -> None:
        nonlocal out
        k = 0
        for c in cands:
            if c in cols and c not in out:
                out.append(c)
                k += 1
                if (max_add is not None) and (k >= int(max_add)):
                    break

    # invariants: take up to 4 to avoid domination by correlated proxies
    add_first_present(invariants, max_add=4)

    # bulk_state: require at least one speed; prefer Vr_bg
    add_first_present(["Vr_bg", "Vr"], max_add=1)
    add_first_present(["Np", "Bmag", "beta"])

    # turbulence state if present
    add_first_present(turbulence)

    return tuple(out)


def _constancy_component(
    x: pd.Series,
    *,
    window: str,
    min_periods: int,
    kind: str,
    eps: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (baseline, constancy, aux_sector)."""
    xs = pd.to_numeric(x, errors="coerce")

    if kind == "signed_abs":
        xa = xs.abs()
        med_a, mad_a = _rolling_med_mad(xa, window, min_periods)
        const = mad_a / (med_a.abs() + float(eps))
        # sector indicator based on signed rolling median
        med_s, _ = _rolling_med_mad(xs, window, min_periods)
        sector = np.sign(med_s.to_numpy(dtype=float))
        sector[~np.isfinite(med_s.to_numpy(dtype=float))] = np.nan
        sector = pd.Series(sector.astype(float), index=xs.index)
        return med_a, const, sector

    med, mad = _rolling_med_mad(xs, window, min_periods)

    if kind == "bounded":
        s0 = _global_mad(xs.to_numpy(dtype=float))
        if (not np.isfinite(s0)) or (s0 <= float(eps)):
            s0 = 1.0
        const = mad / (float(s0) + float(eps))
        return med, const, pd.Series(np.nan, index=xs.index, dtype=float)

    const = mad / (med.abs() + float(eps))
    return med, const, pd.Series(np.nan, index=xs.index, dtype=float)


# ---------------------------------------------------------------------
# Change-score computation (vectorized)
# ---------------------------------------------------------------------

def _two_sided_mean_shift(Z: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (score_sub_placeholder, delta_sub, start_index).

    delta_sub has shape (n-2w, p) and corresponds to i = w..n-w-1.
    """
    n, p = Z.shape
    if (w <= 0) or (n < 2 * w + 1) or (p <= 0):
        return np.asarray([], dtype=float), np.asarray([], dtype=float), int(w)

    Z0 = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    pref = np.vstack([np.zeros((1, p), dtype=float), np.cumsum(Z0, axis=0)])

    left_sum = pref[w:n - w] - pref[0:n - 2 * w]
    right_sum = pref[2 * w:n] - pref[w:n - w]

    left_mean = left_sum / float(w)
    right_mean = right_sum / float(w)
    delta = right_mean - left_mean
    return np.asarray([], dtype=float), delta, int(w)


def _ridge_inv_cov(Zfit: np.ndarray, ridge_alpha: float) -> Tuple[np.ndarray, float]:
    """Ridge-regularized inverse covariance in feature space."""
    Zf = np.asarray(Zfit, dtype=float)
    p = int(Zf.shape[1])
    if (Zf.shape[0] < max(5, p + 2)) or (p <= 0):
        return np.eye(max(1, p), dtype=float), float("nan")

    C = np.cov(Zf, rowvar=False, bias=False)
    if C.ndim == 0:
        C = np.array([[float(C)]], dtype=float)

    tr = float(np.trace(C)) if p > 0 else 1.0
    lam = float(max(0.0, float(ridge_alpha))) * (tr / float(max(1, p)))
    Creg = C + lam * np.eye(p, dtype=float)

    try:
        inv = np.linalg.inv(Creg)
    except Exception:
        inv = np.linalg.pinv(Creg)

    return inv, float(lam)


def _mahalanobis_scores(delta: np.ndarray, invC: np.ndarray) -> np.ndarray:
    """Vectorized sqrt(delta^T invC delta) for all rows."""
    d = np.asarray(delta, dtype=float)
    if d.size == 0:
        return np.asarray([], dtype=float)
    # einsum computes row-wise quadratic form
    q = np.einsum("ij,jk,ik->i", d, invC, d)
    q = np.maximum(q, 0.0)
    return np.sqrt(q)


def _majority_filter_labels(labels: np.ndarray, *, k: int, half_width: int) -> np.ndarray:
    lab = np.asarray(labels, dtype=int)
    n = lab.size
    if (half_width <= 0) or (k <= 1) or (n == 0):
        return lab.copy()

    counts = np.empty((n, int(k)), dtype=int)
    for j in range(int(k)):
        m = (lab == j).astype(int)
        cs = np.concatenate([[0], np.cumsum(m)])
        lo = np.maximum(0, np.arange(n) - half_width)
        hi = np.minimum(n, np.arange(n) + half_width + 1)
        counts[:, j] = cs[hi] - cs[lo]

    return np.argmax(counts, axis=1).astype(int)


def _segments_from_mask_and_labels(
    stable: np.ndarray,
    labels: Optional[np.ndarray],
    *,
    min_points: int,
) -> np.ndarray:
    stable = np.asarray(stable, dtype=bool)
    n = stable.size
    if n == 0:
        return np.asarray([], dtype=int)

    if labels is None:
        lab = np.zeros(n, dtype=int)
    else:
        lab = np.asarray(labels, dtype=int)

    prev_stable = np.r_[False, stable[:-1]]
    prev_lab = np.r_[-999999, lab[:-1]]
    start = stable & ((~prev_stable) | (lab != prev_lab))

    seg = np.full(n, -1, dtype=int)
    if not np.any(start):
        return seg

    seg_id = np.cumsum(start.astype(int)) - 1
    seg[stable] = seg_id[stable]

    if int(min_points) > 1:
        good_ids = seg[seg >= 0]
        if good_ids.size:
            counts = np.bincount(good_ids)
            too_short = np.zeros_like(counts, dtype=bool)
            too_short[counts < int(min_points)] = True
            bad = (seg >= 0) & too_short[seg]
            seg[bad] = -1

            kept = np.unique(seg[seg >= 0])
            if kept.size:
                remap = -np.ones(int(kept.max()) + 1, dtype=int)
                remap[kept] = np.arange(kept.size, dtype=int)
                seg2 = seg.copy()
                ok = seg2 >= 0
                seg2[ok] = remap[seg2[ok]]
                seg = seg2
            else:
                seg[:] = -1

    return seg


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def segment_sources(
    data: pd.DataFrame,
    *,
    vars: Optional[Sequence[str]] = None,
    window: str,
    weights: Optional[Mapping[str, float]] = None,
    mode: str = "gmm_cpd",
    threshold: Optional[float] = None,
    threshold_k: float = 4.0,
    min_points: int = 3,
    transition_pad: int = 1,
    ridge_alpha: float = 0.2,
    gmm_kmax: int = 6,
    gmm_reg_covar: float = 1e-3,
    gmm_smooth_halfwidth: int = 1,
    random_state: int = 0,
) -> SegmentationResult:
    """Segment time ranges into stable, physically-similar regimes.

    Parameters
    ----------
    vars:
        Diagnostics to include. If None, a balanced set is selected automatically
        from the columns available in `data` (including best-effort derived
        diagnostics).

    ridge_alpha:
        Shrinkage strength for the Mahalanobis metric. This is a *ridge* on the
        feature-space covariance: inv(C + lam I). Increasing ridge_alpha makes
        the score less sensitive to poorly conditioned correlations.

    mode:
        - "mv_cpd"  : ridge-Mahalanobis change score only
        - "gmm_cpd" : same score + optional GMM regime labels

    Notes
    -----
    The returned `segment` labels stable runs only (unstable = -1).
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    mode_lc = str(mode).strip().lower()
    if mode_lc not in {"mv_cpd", "gmm_cpd"}:
        raise ValueError(f"mode must be 'mv_cpd' or 'gmm_cpd', got {mode!r}")

    win = str(window)
    n_win = _window_npoints(data.index, win, floor=3)
    mp = int(max(3, int(0.5 * n_win)))

    # pick variables
    if vars is None:
        used_vars = list(_auto_select_vars(data))
    else:
        used_vars = [str(v) for v in vars if str(v) in data.columns]
        if len(used_vars) == 0:
            _ensure_basic_derived(data)
            used_vars = [str(v) for v in (vars or []) if str(v) in data.columns]

    if len(used_vars) == 0:
        score = pd.Series(np.nan, index=data.index, dtype=float)
        seg = np.full(len(data), -1, dtype=int)
        return SegmentationResult(score=score, segment=seg, score_components={}, meta={"used_features": []})

    weights_d = dict(weights or {})
    eps = 1e-12

    feats: list[np.ndarray] = []
    feat_names: list[str] = []
    var_blocks: Dict[str, Tuple[int, int]] = {}

    for v in used_vars:
        v_lc = str(v).strip().lower()
        kind = _infer_kind(v_lc)

        med, const, sector = _constancy_component(data[v], window=win, min_periods=mp, kind=kind, eps=eps)

        j0 = len(feat_names)

        # IMPORTANT: do NOT apply user weights before robust standardization.
        # Pre-scaling cancels out under standardization and is therefore meaningless.
        # We apply weights AFTER standardization below, at the level of feature space.
        feats.append(med.to_numpy(dtype=float))
        feat_names.append(f"{v}:med")

        feats.append(const.to_numpy(dtype=float))
        feat_names.append(f"{v}:const")

        if kind == "signed_abs":
            feats.append(sector.to_numpy(dtype=float))
            feat_names.append(f"{v}:sector")

        j1 = len(feat_names)
        var_blocks[v] = (j0, j1)

    X = np.column_stack(feats)
    Z, Z_med, Z_scale = _robust_standardize(X, eps=1e-12)

    # Apply per-variable weights in standardized feature space (meaningful).
    if weights_d:
        for v, (j0, j1) in var_blocks.items():
            wv = float(weights_d.get(v, weights_d.get(str(v), 1.0)))
            if (not np.isfinite(wv)) or (wv <= 0.0):
                continue
            Z[:, int(j0):int(j1)] *= wv

    # rows with too few finite features are invalid
    finite = np.isfinite(Z)
    min_feat = int(max(2, int(0.85 * Z.shape[1])))
    row_ok = (np.sum(finite, axis=1) >= min_feat)

    # covariance on fully finite rows only
    ok_cov = row_ok & np.all(np.isfinite(Z), axis=1)
    Zfit = Z[ok_cov]

    invC, lam = _ridge_inv_cov(Zfit, float(ridge_alpha))

    # mean-shift vectors (vectorized)
    _, delta_sub, start = _two_sided_mean_shift(Z, int(n_win))

    score_arr = np.full(len(data), np.nan, dtype=float)
    if delta_sub.size:
        p = int(Z.shape[1])
        m = _mahalanobis_scores(delta_sub, invC) / float(np.sqrt(max(1, p)))
        score_arr[int(start):int(start) + m.size] = m

    score_arr[~row_ok] = np.nan

    # threshold: robust median + k*MAD unless user supplied
    if threshold is None:
        sgood = score_arr[np.isfinite(score_arr)]
        if sgood.size:
            smed = float(np.nanmedian(sgood))
            smad = float(np.nanmedian(np.abs(sgood - smed)))
            thr = smed + float(threshold_k) * (smad + 1e-12)
        else:
            thr = float("nan")
    else:
        thr = float(threshold)

    stable = row_ok & np.isfinite(score_arr)
    if np.isfinite(thr):
        stable &= (score_arr <= thr)

    # transition padding: dilate unstable neighborhoods around score excursions
    if int(transition_pad) > 0 and np.isfinite(thr):
        trans = np.isfinite(score_arr) & (score_arr > thr)
        if np.any(trans):
            k = 2 * int(transition_pad) + 1
            ker = np.ones(int(k), dtype=int)
            dil = np.convolve(trans.astype(int), ker, mode="same") > 0
            stable[dil] = False

    # Optional ML regularization
    labels = None
    gmm_meta: Dict[str, Any] = {}
    if mode_lc == "gmm_cpd":
        ok = row_ok & np.all(np.isfinite(Z), axis=1)
        Zg = Z[ok]
        if Zg.shape[0] >= max(30, 6 * Zg.shape[1]):
            try:
                from sklearn.mixture import GaussianMixture

                kmax = int(max(1, gmm_kmax))
                best = None
                best_bic = float("inf")
                bics = []
                for k in range(1, kmax + 1):
                    gm = GaussianMixture(
                        n_components=int(k),
                        covariance_type="full",
                        reg_covar=float(gmm_reg_covar),
                        random_state=int(random_state),
                        max_iter=300,
                        n_init=2,
                    )
                    gm.fit(Zg)
                    bic = float(gm.bic(Zg))
                    bics.append((int(k), bic))
                    if bic < best_bic:
                        best_bic = bic
                        best = gm

                if best is not None:
                    lab = np.full(len(data), -1, dtype=int)
                    lab_ok = best.predict(Zg).astype(int)
                    lab[ok] = lab_ok

                    if int(gmm_smooth_halfwidth) > 0:
                        lab2 = lab.copy()
                        ok2 = lab2 >= 0
                        kbest = int(best.n_components)
                        lab2[ok2] = _majority_filter_labels(lab2[ok2], k=kbest, half_width=int(gmm_smooth_halfwidth))
                        lab = lab2

                    labels = lab
                    gmm_meta = {
                        "enabled": True,
                        "k_best": int(best.n_components),
                        "bic_best": float(best_bic),
                        "bics": [(int(kk), float(bb)) for kk, bb in bics],
                        "reg_covar": float(gmm_reg_covar),
                        "smooth_halfwidth": int(gmm_smooth_halfwidth),
                    }
            except Exception as e:
                gmm_meta = {"enabled": False, "error": repr(e)}

    seg = _segments_from_mask_and_labels(stable, labels, min_points=int(min_points))

    score = pd.Series(score_arr.astype(float), index=data.index, name="source_score")

    meta = {
        "mode": mode_lc,
        "metric": "ridge_mahalanobis",
        "ridge_alpha": float(ridge_alpha),
        "ridge_lambda": float(lam) if np.isfinite(lam) else float("nan"),
        "window": win,
        "n_win": int(n_win),
        "min_periods": int(mp),
        "threshold": float(thr) if np.isfinite(thr) else float("nan"),
        "threshold_k": float(threshold_k),
        "min_points": int(min_points),
        "transition_pad": int(transition_pad),
        "used_features": [str(v) for v in used_vars],
        "weights": {str(k): float(v) for k, v in (weights_d or {}).items()} if weights_d else {},
        "feature_blocks": {str(k): [int(a), int(b)] for k, (a, b) in var_blocks.items()},
        "feature_matrix": {
            "n_features": int(Z.shape[1]),
            "names": feat_names,
            "robust_center": np.asarray(Z_med, dtype=float).tolist(),
            "robust_scale": np.asarray(Z_scale, dtype=float).tolist(),
        },
        "gmm": gmm_meta,
        "score_stats": {
            "p16": float(np.nanpercentile(score_arr, 16)) if np.isfinite(score_arr).any() else float("nan"),
            "p50": float(np.nanpercentile(score_arr, 50)) if np.isfinite(score_arr).any() else float("nan"),
            "p84": float(np.nanpercentile(score_arr, 84)) if np.isfinite(score_arr).any() else float("nan"),
            "max": float(np.nanmax(score_arr)) if np.isfinite(score_arr).any() else float("nan"),
            "frac_above_threshold": float(np.nanmean(score_arr > float(thr)))
            if (np.isfinite(thr) and np.isfinite(score_arr).any())
            else float("nan"),
        },
        "n_segments": int(len(np.unique(seg[seg >= 0]))),
    }

    # Backward-compat: components dict retained but no longer interpreted as
    # "contributions". Keep empty to avoid misleading plots.
    comps: Dict[str, pd.Series] = {}

    return SegmentationResult(score=score, segment=seg, score_components=comps, meta=meta)
