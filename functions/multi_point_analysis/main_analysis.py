# -*- coding: utf-8 -*-
"""Main analysis entry point.

Design rules (per LOG.md):
- Single source-of-truth settings: dict.
- No downloads; consume only MHDTurbPy interval pickle folders.
- Conservative, auditable outputs; store full diagnostic time series and full
  correlation/cross-correlation curves (not just peaks).
- Minimal .py file count: keep the full pipeline in this module.

Stages implemented here:
  Stage-0: load_intervals
  Stage-1: event detection (Theta_PS / Theta_med) + event-level MVA + boundaries
  Stage-2: scored candidate matching
  Stage-3: alignment (lag estimation) + full correlation curves
  Stage-4: geometry (when position products are available)
  Stage-5: boundary timing / MVAB normals / speed / thickness proxies
  Stage-6: ensemble-ready per-pair table + separation/aspect-ratio summary stats
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os

import numpy as np
import pandas as pd

from .load import load_intervals


# ----------------------------
# Optional imports (MHDTurbPy)
# ----------------------------

try:
    import TurbPy as turb  # type: ignore
except Exception:
    turb = None


# ----------------------------
# Small utilities
# ----------------------------

def _ensure_dtindex(obj: Any) -> Any:
    """Ensure a DatetimeIndex for DataFrame or Series (preserving the input type)."""
    if isinstance(obj, pd.Series):
        s = obj
        if isinstance(s.index, pd.DatetimeIndex):
            out = s if s.index.is_monotonic_increasing else s.sort_index()
            if out.index.hasnans:
                out = out.loc[~out.index.isna()].copy()
            return out
        out = s.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[~out.index.isna()].sort_index()
        return out
    if isinstance(obj, pd.DataFrame):
        df = obj
        if isinstance(df.index, pd.DatetimeIndex):
            out = df if df.index.is_monotonic_increasing else df.sort_index()
            if out.index.hasnans:
                out = out.loc[~out.index.isna()].copy()
            return out
        out = df.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[~out.index.isna()].sort_index()
        return out
    raise TypeError(f"expected DataFrame/Series, got {type(obj)}")




def _shift_to_epoch(obj: Any, t_center: Any, epoch: Any = "2000-01-01T00:00:00") -> Any:
    """Shift a Series/DataFrame time axis so that t_center maps to a common epoch.

    Used for event-centered cross-correlations. Without this, cross-correlation between
    two windows that are separated in clock time will only have overlap at lags comparable
    to the absolute window separation, producing misleading τ* and unreadable curves.
    """
    if obj is None:
        return None
    if hasattr(obj, "empty") and obj.empty:
        return obj
    obj = _ensure_dtindex(obj)
    tc = pd.Timestamp(t_center)
    ep = pd.Timestamp(epoch)
    rel = obj.index - tc
    out = obj.copy()
    out.index = ep + rel
    return out


def _as_path(p: Any) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _mkdir(p: Path) -> None:
    os.makedirs(str(p), exist_ok=True)


def _stable_sort_df(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    for c in by:
        if c not in df.columns:
            return df
    return df.sort_values(list(by), kind="mergesort").reset_index(drop=True)


def _infer_vec_cols(df: pd.DataFrame, candidates: Sequence[Sequence[str]]) -> List[str]:
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return list(cols)
    raise KeyError(f"Could not infer vector columns from candidates={candidates}; df.columns={list(df.columns)}")


def _unitvec(arr: np.ndarray, axis: int = -1, eps: float = 1e-15) -> np.ndarray:
    nrm = np.linalg.norm(arr, axis=axis, keepdims=True)
    nrm = np.where(nrm > eps, nrm, np.nan)
    return arr / nrm


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu = _unitvec(u)
    vv = _unitvec(v)
    dot = np.nansum(uu * vv, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _rle_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start_idx, end_idx) segments where mask is True."""
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    diff = np.diff(m.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e >= s]


def _apply_hysteresis(mask_on: np.ndarray, mask_off: np.ndarray) -> np.ndarray:
    """Hysteresis without per-sample loops.

    The "on" condition starts events; the "off" condition defines the extent.
    We keep any contiguous True-run of mask_off that contains at least one True
    sample of mask_on.
    """
    if mask_on.size == 0:
        return mask_on.astype(bool)
    segs_off = _rle_true_segments(mask_off.astype(bool))
    keep = np.zeros(mask_on.size, dtype=bool)
    for i0, i1 in segs_off:
        if bool(np.any(mask_on[i0 : i1 + 1])):
            keep[i0 : i1 + 1] = True
    return keep




def _speed_to_mps(v: np.ndarray) -> float:
    """Best-effort convert a 3-vector speed to m/s.

    Convention in MHDTurbPy products is typically km/s. We detect units by scale.
    """
    if v is None:
        return float('nan')
    vv = np.asarray(v, dtype=float).reshape(-1)
    if vv.size == 0 or (not np.isfinite(vv).any()):
        return float('nan')
    mag = float(np.linalg.norm(vv[np.isfinite(vv)]))
    if not np.isfinite(mag):
        return float('nan')
    # heuristic: solar-wind speeds ~200-800 km/s -> 200-800 if km/s, 2e5-8e5 if m/s
    return mag * 1e3 if mag < 2e4 else mag
def _median_dt_seconds(index: pd.DatetimeIndex) -> Optional[float]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return None
    diffs = np.diff(index.view("i8"))  # ns
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs)) * 1e-9


def _resample_series_regular(s: pd.Series, cadence_s: float) -> pd.Series:
    """Resample to a regular time grid using time interpolation."""
    if not isinstance(s.index, pd.DatetimeIndex) or cadence_s <= 0:
        return s
    step_ms = int(round(float(cadence_s) * 1000.0))
    step_ms = max(step_ms, 1)
    rule = f"{step_ms}ms"
    s2 = s.sort_index()
    t0, t1 = s2.index[0], s2.index[-1]
    grid = pd.date_range(t0, t1, freq=rule)
    s3 = s2.reindex(s2.index.union(grid)).interpolate(method="time").reindex(grid)
    return s3


def _resample_df_regular(df: pd.DataFrame, cadence_s: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    out = pd.DataFrame(index=_resample_series_regular(df.iloc[:, 0], cadence_s).index)
    for c in cols:
        out[c] = _resample_series_regular(df[c], cadence_s).to_numpy(dtype=float)
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    xx = x[m]
    yy = y[m]
    sx = np.std(xx)
    sy = np.std(yy)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _nearest_finite_row(diag: pd.DataFrame, t: pd.Timestamp, cols: Sequence[str], *, max_dt: pd.Timedelta) -> Optional[np.ndarray]:
    if diag is None or diag.empty or any(c not in diag.columns for c in cols):
        return None
    t = pd.Timestamp(t)
    # nearest by index, then expand search to nearest finite sample within +/- max_dt
    i0 = diag.index.get_indexer([t], method="nearest")
    if i0.size != 1 or i0[0] < 0:
        return None
    i0 = int(i0[0])
    # candidate indices within window
    t0 = t - max_dt
    t1 = t + max_dt
    sub = diag.loc[(diag.index >= t0) & (diag.index <= t1), list(cols)]
    if sub.empty:
        v = diag.iloc[i0][list(cols)].to_numpy(dtype=float)
        return v if np.isfinite(v).all() else None
    m = np.isfinite(sub.to_numpy(dtype=float)).all(axis=1)
    if not np.any(m):
        return None
    sub2 = sub.loc[m]
    # choose nearest finite time to t
    dt_ns = np.abs((sub2.index.to_numpy(dtype="datetime64[ns]") - np.datetime64(t)).astype("timedelta64[ns]").astype("int64"))
    j = int(np.argmin(dt_ns))
    # NOTE: .to_numpy(...) already returns an ndarray; do not call it.
    v = sub2.iloc[j].to_numpy(dtype=float)
    return v if np.isfinite(v).all() else None


# ----------------------------
# Parker spiral + rolling-median background
# ----------------------------

def _compute_parker_deflection(
    B: pd.DataFrame,
    V: pd.DataFrame,
    density: Optional[pd.Series],
    *,
    r0: float,
    av_window: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    if turb is None:
        raise ImportError("TurbPy could not be imported; Parker-spiral diagnostics require TurbPy")
    if density is None:
        raise KeyError("Parker-spiral diagnostics require a density Series (settings['analysis']['density_col'])")

    B_parker, alpha_p, Vr_filtered, parker_def_angle = turb.local_parker_spiral(
        B.copy(),
        V.copy(),
        density.interpolate(),
        r0=r0,
        av_window=av_window,
    )
    theta_ps = pd.Series(np.asarray(parker_def_angle, dtype=float), index=B.index, name="Theta_PS_deg")
    Bp = _ensure_dtindex(B_parker) if isinstance(B_parker, pd.DataFrame) else None
    return Bp, theta_ps


def _rolling_median_direction(B: pd.DataFrame, window: str) -> pd.DataFrame:
    # Centered window to avoid phase-lagging the background direction.
    return B.rolling(window, min_periods=1, center=True).median()


# ----------------------------
# MVA / MVAB
# ----------------------------

@dataclass
class MVAResult:
    evals: np.ndarray  # ascending
    evecs: np.ndarray  # columns correspond to evals (ascending)

    @property
    def n_hat(self) -> np.ndarray:
        return self.evecs[:, 0]

    @property
    def m_hat(self) -> np.ndarray:
        return self.evecs[:, 1]

    @property
    def l_hat(self) -> np.ndarray:
        return self.evecs[:, 2]

    @property
    def ratio_lm(self) -> float:
        return float(self.evals[2] / self.evals[1]) if self.evals[1] > 0 else float("nan")

    @property
    def ratio_mn(self) -> float:
        return float(self.evals[1] / self.evals[0]) if self.evals[0] > 0 else float("nan")


def _compute_mva(B: pd.DataFrame) -> Optional[MVAResult]:
    if B is None or not isinstance(B, pd.DataFrame) or len(B) < 10:
        return None
    X = B.to_numpy(dtype=float)
    m = np.isfinite(X).all(axis=1)
    X = X[m]
    if X.shape[0] < 10:
        return None
    X = X - np.mean(X, axis=0, keepdims=True)
    C = (X.T @ X) / float(X.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)  # ascending
    # Enforce right-handed (L,M,N): flip N if needed.
    L = evecs[:, 2]
    M = evecs[:, 1]
    N = evecs[:, 0]
    det = np.linalg.det(np.column_stack([L, M, N]))
    if det < 0:
        evecs[:, 0] = -evecs[:, 0]
    return MVAResult(evals=np.asarray(evals, dtype=float), evecs=np.asarray(evecs, dtype=float))


# ----------------------------
# Spherical polarization diagnostics (Dunn et al. 2023)
# ----------------------------

def _magnetic_compressibility(B: pd.DataFrame) -> float:
    """Magnetic compressibility C_B.

    Implements the definition used in Dunn et al. (2023) and Chen et al. (2020):
      C_B = Var(|B|) / tr(K_B)
    where K_B is the 3x3 covariance matrix of the magnetic-field components.
    """
    if B is None or (not isinstance(B, pd.DataFrame)) or B.empty:
        return float("nan")
    X = B.to_numpy(dtype=float)
    m = np.isfinite(X).all(axis=1)
    X = X[m]
    if X.shape[0] < 10:
        return float("nan")
    bmag = np.linalg.norm(X, axis=1)
    if not np.isfinite(bmag).any():
        return float("nan")
    var_bmag = float(np.nanvar(bmag, ddof=1))
    X0 = X - np.mean(X, axis=0, keepdims=True)
    K = (X0.T @ X0) / float(max(X0.shape[0] - 1, 1))
    trK = float(np.trace(K))
    if (not np.isfinite(var_bmag)) or (not np.isfinite(trK)) or trK <= 0:
        return float("nan")
    return float(var_bmag / trK)


def _psi_perp_isotropy(B: pd.DataFrame) -> Tuple[float, float, float]:
    """Perpendicular variance isotropy Ψ.

    Following Dunn et al. (2023): project B onto the plane perpendicular to the
    mean field, compute the covariance eigenvalues within that plane, and return

      Ψ = λ_2 / (λ_1 + λ_2),

    where λ_1 ≥ λ_2 are the eigenvalues (powers) along the directions of maximum
    and minimum perpendicular variance.

    Returns: (psi, lambda1, lambda2) with λ_1 ≥ λ_2.
    """
    if B is None or (not isinstance(B, pd.DataFrame)) or B.empty:
        return float("nan"), float("nan"), float("nan")
    X = B.to_numpy(dtype=float)
    m = np.isfinite(X).all(axis=1)
    X = X[m]
    if X.shape[0] < 10:
        return float("nan"), float("nan"), float("nan")

    B0 = np.mean(X, axis=0)
    nb = float(np.linalg.norm(B0))
    if (not np.isfinite(nb)) or nb <= 0:
        return float("nan"), float("nan"), float("nan")
    bh = B0 / nb

    # Project onto plane perpendicular to the mean field.
    # P = I - b b^T
    P = np.eye(3) - np.outer(bh, bh)
    Xp = (P @ X.T).T
    Xp = Xp - np.mean(Xp, axis=0, keepdims=True)

    C = (Xp.T @ Xp) / float(max(Xp.shape[0] - 1, 1))
    evals, _ = np.linalg.eigh(C)  # ascending
    evals = np.sort(np.asarray(evals, dtype=float))[::-1]  # descending

    # Two largest eigenvalues correspond to the perpendicular plane.
    if evals.size < 2:
        return float("nan"), float("nan"), float("nan")
    lam1 = float(evals[0])
    lam2 = float(evals[1])
    denom = lam1 + lam2
    if (not np.isfinite(denom)) or denom <= 0:
        return float("nan"), float("nan"), float("nan")
    psi = float(lam2 / denom)
    return psi, lam1, lam2


# ----------------------------
# Boundary finding (robust step-like sharp drop in |B|)
# ----------------------------

def _boundary_times_sharp_bmag_drop(
    B: pd.DataFrame,
    *,
    t_core_start: pd.Timestamp,
    t_core_stop: pd.Timestamp,
    lead_search: pd.Timedelta,
    trail_search: pd.Timedelta,
    pre_window: pd.Timedelta,
    post_window: pd.Timedelta,
    smooth_window: str,
    inside_max: pd.Timedelta,
    min_drop_frac: float,
    min_drop_abs: float,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Dict[str, Any]]:
    """Return (t_lead, t_trail, meta) using a step/drop score on smoothed |B|.

    Idea:
      - boundaries are step-like, sharp transitions with appreciable amplitude,
        not tiny gradients. We detect them by comparing robust medians on each side
        of a candidate time t, over configurable pre/post windows, and weighting by
        steepness (time derivative magnitude with the correct sign).

    Conservative behavior:
      - require a minimum drop amplitude (absolute and fractional) relative to the
        local baseline; otherwise return NaT and fall back upstream.
    """
    meta: Dict[str, Any] = {"method": "sharp_bmag_drop", "ok_lead": False, "ok_trail": False}

    if B is None or B.empty or not isinstance(B.index, pd.DatetimeIndex):
        return None, None, meta

    t_core_start = pd.Timestamp(t_core_start)
    t_core_stop = pd.Timestamp(t_core_stop)

    # Build a working window around the core.
    t0 = t_core_start - lead_search
    t1 = t_core_stop + trail_search
    Bb = B.loc[(B.index >= t0) & (B.index <= t1)]
    if Bb.empty:
        return None, None, meta

    bmag = pd.Series(np.linalg.norm(Bb.to_numpy(dtype=float), axis=1), index=Bb.index, name="bmag")
    bmag = bmag.rolling(smooth_window, min_periods=1).median()

    cad = _median_dt_seconds(bmag.index) or 1.0
    bmag_r = _resample_series_regular(bmag, cad)
    if bmag_r.empty or len(bmag_r) < 30:
        return None, None, meta

    # helper: time-based windows converted to sample counts (robust on regular grid)
    def _nwin(dt: pd.Timedelta) -> int:
        return max(3, int(round(float(dt.total_seconds()) / float(cad))))

    n_pre = _nwin(pre_window)
    n_post = _nwin(post_window)

    s = bmag_r.to_numpy(dtype=float)
    t = bmag_r.index

    # pre median (ending at i)
    pre_med = pd.Series(s, index=t).rolling(n_pre, min_periods=max(3, n_pre // 2)).median().to_numpy(dtype=float)
    # post median (starting at i): compute on reversed series
    post_med = pd.Series(s[::-1], index=t[::-1]).rolling(n_post, min_periods=max(3, n_post // 2)).median().to_numpy(dtype=float)[::-1]

    # derivative (central differences)
    dsdt = np.gradient(s, float(cad))

    # scores
    drop = pre_med - post_med
    rise = post_med - pre_med
    drop_steep = np.maximum(0.0, -dsdt)
    rise_steep = np.maximum(0.0, dsdt)

    # fractional amplitude against baseline
    frac_drop = drop / (np.abs(pre_med) + 1e-12)
    frac_rise = rise / (np.abs(post_med) + 1e-12)

    score_drop = drop * drop_steep
    score_rise = rise * rise_steep

    # constrain "inside" part of the core to avoid grabbing interior wiggles:
    inside = min(inside_max, max(pd.Timedelta(seconds=0), (t_core_stop - t_core_start) / 5))
    lead_win = (t >= (t_core_start - lead_search)) & (t <= (t_core_start + inside))
    trail_win = (t >= (t_core_stop - inside)) & (t <= (t_core_stop + trail_search))

    def _pick(win_mask: np.ndarray, which: str) -> Tuple[Optional[pd.Timestamp], Dict[str, Any]]:
        if not np.any(win_mask):
            return None, {"ok": False}
        iwin = np.where(win_mask)[0]

        # primary: sharp drop
        sc = score_drop[iwin]
        imax = int(iwin[int(np.nanargmax(sc))]) if np.isfinite(sc).any() else -1
        if imax >= 0:
            amp = float(drop[imax])
            frac = float(frac_drop[imax]) if np.isfinite(frac_drop[imax]) else float("nan")
            ok = (np.isfinite(amp) and np.isfinite(frac) and (amp >= min_drop_abs) and (frac >= min_drop_frac))
            if ok:
                return pd.Timestamp(t[imax]), {"ok": True, "mode": "drop", "amp": amp, "frac": frac}

        # fallback: sharp rise (exit boundary can be a rise)
        sc2 = score_rise[iwin]
        imax2 = int(iwin[int(np.nanargmax(sc2))]) if np.isfinite(sc2).any() else -1
        if imax2 >= 0:
            amp2 = float(rise[imax2])
            frac2 = float(frac_rise[imax2]) if np.isfinite(frac_rise[imax2]) else float("nan")
            ok2 = (np.isfinite(amp2) and np.isfinite(frac2) and (amp2 >= min_drop_abs) and (frac2 >= min_drop_frac))
            if ok2:
                return pd.Timestamp(t[imax2]), {"ok": True, "mode": "rise", "amp": amp2, "frac": frac2}

        return None, {"ok": False}

    tL, metaL = _pick(lead_win, "lead")
    tT, metaT = _pick(trail_win, "trail")
    meta.update({"lead": metaL, "trail": metaT, "cadence_s": float(cad), "n_pre": int(n_pre), "n_post": int(n_post)})

    meta["ok_lead"] = bool(metaL.get("ok", False))
    meta["ok_trail"] = bool(metaT.get("ok", False))
    return tL, tT, meta


def _boundary_times_nonparam_bmag_extrema(
    B: pd.DataFrame,
    *,
    t_core_start: pd.Timestamp,
    t_core_stop: pd.Timestamp,
    lead_search: pd.Timedelta,
    trail_search: pd.Timedelta,
    smooth_window: str,
    inside_max: pd.Timedelta,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Dict[str, Any]]:
    """Non-parametric boundary estimate from extrema of d|B|/dt.

    This is intentionally *threshold-free*: within configurable search windows,
    pick the steepest negative gradient near the leading edge and the steepest
    positive gradient near the trailing edge of the deflection-defined core.

    The method is designed to provide a reproducible boundary guess without
    amplitude tuning. Downstream diagnostics should still audit |B| and the
    resulting spans.
    """
    meta: Dict[str, Any] = {"method": "nonparam_bmag_extrema", "ok_lead": False, "ok_trail": False}
    if B is None or B.empty or not isinstance(B.index, pd.DatetimeIndex):
        return None, None, meta

    t_core_start = pd.Timestamp(t_core_start)
    t_core_stop = pd.Timestamp(t_core_stop)

    t0 = t_core_start - lead_search
    t1 = t_core_stop + trail_search
    Bb = B.loc[(B.index >= t0) & (B.index <= t1)]
    if Bb.empty:
        return None, None, meta

    bmag = pd.Series(np.linalg.norm(Bb.to_numpy(dtype=float), axis=1), index=Bb.index, name="bmag")
    bmag = bmag.rolling(smooth_window, min_periods=1).median()

    cad = _median_dt_seconds(bmag.index) or 1.0
    bmag_r = _resample_series_regular(bmag, cad)
    if bmag_r.empty or len(bmag_r) < 30:
        return None, None, meta

    s = bmag_r.to_numpy(dtype=float)
    t = bmag_r.index
    dsdt = np.gradient(s, float(cad))

    inside = min(inside_max, max(pd.Timedelta(seconds=0), (t_core_stop - t_core_start) / 5))
    lead_win = (t >= (t_core_start - lead_search)) & (t <= (t_core_start + inside))
    trail_win = (t >= (t_core_stop - inside)) & (t <= (t_core_stop + trail_search))

    tL = None
    tT = None
    if np.any(lead_win):
        iwin = np.where(lead_win)[0]
        # leading boundary: strongest negative gradient
        j = int(iwin[int(np.nanargmin(dsdt[iwin]))]) if np.isfinite(dsdt[iwin]).any() else -1
        if j >= 0 and np.isfinite(dsdt[j]):
            tL = pd.Timestamp(t[j])
            meta["ok_lead"] = True
            meta["lead_dsdt"] = float(dsdt[j])

    if np.any(trail_win):
        iwin = np.where(trail_win)[0]
        # trailing boundary: strongest positive gradient
        j = int(iwin[int(np.nanargmax(dsdt[iwin]))]) if np.isfinite(dsdt[iwin]).any() else -1
        if j >= 0 and np.isfinite(dsdt[j]):
            tT = pd.Timestamp(t[j])
            meta["ok_trail"] = True
            meta["trail_dsdt"] = float(dsdt[j])

    meta["cadence_s"] = float(cad)
    return tL, tT, meta


# ----------------------------
# Correlation functions (vector and scalar)
# ----------------------------

def _overlap_slices(na: int, nb: int, kshift: int) -> Tuple[int, int, int, int]:
    """Return overlap slices (i0,i1,j0,j1) for mapping j = i + kshift."""
    if na <= 0 or nb <= 0:
        return 0, 0, 0, 0
    if kshift >= 0:
        i0 = 0
        i1 = min(na, nb - kshift)
        j0 = kshift
        j1 = j0 + (i1 - i0)
    else:
        kk = -kshift
        i0 = kk
        i1 = min(na, nb + kk)
        j0 = 0
        j1 = j0 + (i1 - i0)
    if i1 < i0 or j1 < j0:
        return 0, 0, 0, 0
    return int(i0), int(i1), int(j0), int(j1)


def _vector_corr_curve(
    A: pd.DataFrame,
    B: Optional[pd.DataFrame],
    *,
    tau_max_s: float,
    cadence_s: Optional[float],
) -> pd.DataFrame:
    """Normalized correlation curve of vector fluctuations on a regular grid.

    If B is None -> autocorrelation of A.
    If B provided -> cross-correlation between A and B.

    Discrete definition (Matthaeus-style):
      R(τ) = < δA(t) · δB(t+τ) > / sqrt( <|δA|^2> <|δB|^2> )

    Important implementation detail:
      We do NOT intersect timestamps between A and B (that biases toward near-zero lag
      when the grids are not identical). Instead, we resample each window to a regular
      cadence, then correlate by index shifting using the relative start-time offset.
    """
    if A is None or A.empty:
        return pd.DataFrame(columns=["lag_s", "R"])
    A = _ensure_dtindex(A)
    if B is not None:
        B = _ensure_dtindex(B)
    dtA = _median_dt_seconds(A.index)
    dtB = _median_dt_seconds(B.index) if B is not None else dtA
    dt_ref = max([d for d in (dtA, dtB) if d is not None], default=1.0)

    if cadence_s is None:
        cadence_s = dt_ref
    else:
        cadence_s = max(float(cadence_s), dt_ref)
    if cadence_s <= 0:
        cadence_s = 1.0



    Ar = _resample_df_regular(A, cadence_s)
    if B is None:
        Br = Ar
    else:
        Br = _resample_df_regular(B, cadence_s)

    if Ar.empty or Br.empty or len(Ar) < 20 or len(Br) < 20:
        return pd.DataFrame(columns=["lag_s", "R"])

    Xa = Ar.to_numpy(dtype=float)
    Yb = Br.to_numpy(dtype=float)

    # global mean subtraction over finite rows (stable across lags)
    mA = np.isfinite(Xa).all(axis=1)
    mB = np.isfinite(Yb).all(axis=1)
    if mA.sum() < 20 or mB.sum() < 20:
        return pd.DataFrame(columns=["lag_s", "R"])

    muA = np.mean(Xa[mA], axis=0, keepdims=True)
    muB = np.mean(Yb[mB], axis=0, keepdims=True)
    Xa0 = Xa - muA
    Yb0 = Yb - muB

    eA = float(np.mean(np.sum(Xa0[mA] * Xa0[mA], axis=1)))
    eB = float(np.mean(np.sum(Yb0[mB] * Yb0[mB], axis=1)))
    if (not np.isfinite(eA)) or (not np.isfinite(eB)) or eA <= 0 or eB <= 0:
        return pd.DataFrame(columns=["lag_s", "R"])
    denom = math.sqrt(eA * eB)

    # start-time offset (in samples) between regular grids
    t0A = Ar.index[0]
    t0B = Br.index[0]
    d0 = int(round(float((t0B - t0A).total_seconds()) / cadence_s))

    kmax = int(round(float(tau_max_s) / cadence_s))
    lags = np.arange(-kmax, kmax + 1, dtype=int)
    out = np.full(lags.shape, np.nan, dtype=float)

    na = Xa0.shape[0]
    nb = Yb0.shape[0]

    for i, k in enumerate(lags):
        kshift = int(k - d0)  # j = i + (k - d0)
        i0, i1, j0, j1 = _overlap_slices(na, nb, kshift)
        n = i1 - i0
        if n < 10:
            continue
        X = Xa0[i0:i1]
        Y = Yb0[j0:j1]
        mm = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        if mm.sum() < 10:
            continue
        out[i] = float(np.mean(np.sum(X[mm] * Y[mm], axis=1)) / denom)

    return pd.DataFrame({"lag_s": lags.astype(float) * cadence_s, "R": out})


def _vector_dot_curve(
    A: pd.DataFrame,
    B: Optional[pd.DataFrame],
    *,
    tau_max_s: float,
    cadence_s: Optional[float],
) -> pd.DataFrame:
    r"""Mean dot-product curve for unit directions \hat{b}(t).

    Computes:
      D(τ) = < \hat{b}_A(t) · \hat{b}_B(t+τ) >

    This is often a more direct morphology/alignment metric than a mean-subtracted
    δB correlation when the dominant signal is a coherent rotation.
    """
    if A is None or A.empty:
        return pd.DataFrame(columns=["lag_s", "dot"])
    A = _ensure_dtindex(A)
    if B is not None:
        B = _ensure_dtindex(B)

    dtA = _median_dt_seconds(A.index)
    dtB = _median_dt_seconds(B.index) if B is not None else dtA
    dt_ref = max([d for d in (dtA, dtB) if d is not None], default=1.0)

    if cadence_s is None:
        cadence_s = dt_ref
    else:
        cadence_s = max(float(cadence_s), dt_ref)
    if cadence_s <= 0:
        cadence_s = 1.0



    Ar = _resample_df_regular(A, cadence_s)
    if B is None:
        Br = Ar
    else:
        Br = _resample_df_regular(B, cadence_s)

    if Ar.empty or Br.empty or len(Ar) < 20 or len(Br) < 20:
        return pd.DataFrame(columns=["lag_s", "dot"])

    Xa = Ar.to_numpy(dtype=float)
    Yb = Br.to_numpy(dtype=float)

    # unit vectors
    nA = np.linalg.norm(Xa, axis=1)
    nB = np.linalg.norm(Yb, axis=1)
    mA = np.isfinite(Xa).all(axis=1) & np.isfinite(nA) & (nA > 0)
    mB = np.isfinite(Yb).all(axis=1) & np.isfinite(nB) & (nB > 0)
    if mA.sum() < 20 or mB.sum() < 20:
        return pd.DataFrame(columns=["lag_s", "dot"])

    bhA = Xa / (nA[:, None] + 1e-15)
    bhB = Yb / (nB[:, None] + 1e-15)

    # start-time offset (in samples)
    t0A = Ar.index[0]
    t0B = Br.index[0]
    d0 = int(round(float((t0B - t0A).total_seconds()) / cadence_s))

    kmax = int(round(float(tau_max_s) / cadence_s))
    lags = np.arange(-kmax, kmax + 1, dtype=int)
    out = np.full(lags.shape, np.nan, dtype=float)

    na = bhA.shape[0]
    nb = bhB.shape[0]

    for i, k in enumerate(lags):
        kshift = int(k - d0)
        i0, i1, j0, j1 = _overlap_slices(na, nb, kshift)
        n = i1 - i0
        if n < 10:
            continue
        X = bhA[i0:i1]
        Y = bhB[j0:j1]
        mm = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        if mm.sum() < 10:
            continue
        out[i] = float(np.mean(np.sum(X[mm] * Y[mm], axis=1)))

    return pd.DataFrame({"lag_s": lags.astype(float) * cadence_s, "dot": out})


def _scalar_corr_curve(x: pd.Series, y: pd.Series, *, tau_max_s: float, cadence_s: Optional[float]) -> pd.DataFrame:
    """Pearson correlation curve for scalar series.

    As with `_vector_corr_curve`, this does NOT intersect timestamps; it correlates
    by index shifting after regular resampling, accounting for the relative start-time
    offset between the two resampled grids.
    """
    x = _ensure_dtindex(x.dropna())
    y = _ensure_dtindex(y.dropna())
    if x.empty or y.empty:
        return pd.DataFrame(columns=["lag_s", "corr"])
    dtX = _median_dt_seconds(x.index)
    dtY = _median_dt_seconds(y.index)
    dt_ref = max([d for d in (dtX, dtY) if d is not None], default=1.0)

    if cadence_s is None:
        cadence_s = dt_ref
    else:
        cadence_s = max(float(cadence_s), dt_ref)
    if cadence_s <= 0:
        cadence_s = 1.0


    xg = _resample_series_regular(x, cadence_s)
    yg = _resample_series_regular(y, cadence_s)
    if xg.empty or yg.empty or len(xg) < 20 or len(yg) < 20:
        return pd.DataFrame(columns=["lag_s", "corr"])

    xa = xg.to_numpy(dtype=float)
    ya = yg.to_numpy(dtype=float)

    # start-time offset (in samples)
    t0x = xg.index[0]
    t0y = yg.index[0]
    d0 = int(round(float((t0y - t0x).total_seconds()) / cadence_s))

    kmax = int(round(float(tau_max_s) / cadence_s))
    lags = np.arange(-kmax, kmax + 1, dtype=int)
    out = np.full(lags.shape, np.nan, dtype=float)

    na = xa.shape[0]
    nb = ya.shape[0]

    for i, k in enumerate(lags):
        kshift = int(k - d0)  # j = i + (k - d0)
        i0, i1, j0, j1 = _overlap_slices(na, nb, kshift)
        n = i1 - i0
        if n < 10:
            continue
        xs = xa[i0:i1]
        ys = ya[j0:j1]
        out[i] = _pearson_corr(xs, ys)

    return pd.DataFrame({"lag_s": lags.astype(float) * cadence_s, "corr": out})

def _peak_and_width(curve: pd.DataFrame, *, col: str) -> Tuple[float, float, float]:
    """Return (tau_star, r_star, width_proxy_s) for curve[col]."""
    if curve is None or curve.empty or col not in curve.columns:
        return float("nan"), float("nan"), float("nan")
    c = curve.dropna(subset=[col])
    if c.empty:
        return float("nan"), float("nan"), float("nan")
    idx = int(c[col].idxmax())
    tau = float(curve.loc[idx, "lag_s"])
    r = float(curve.loc[idx, col])
    half = 0.5 * r
    left = c[c["lag_s"] <= tau]
    right = c[c["lag_s"] >= tau]
    try:
        tl = float(left[left[col] >= half]["lag_s"].min())
        tr = float(right[right[col] >= half]["lag_s"].max())
        width = float(tr - tl)
    except Exception:
        width = float("nan")
    return tau, r, width


def _corr_time_proxies(curve: pd.DataFrame, *, col: str) -> Tuple[float, float]:
    """Return (tau_1e, tau_int) from an autocorrelation-like curve.

    tau_1e: first positive lag where R falls below exp(-1).
    tau_int: integral of positive-lag R until first zero crossing (or max lag).
    """
    if curve is None or curve.empty or col not in curve.columns:
        return float("nan"), float("nan")
    c = curve.dropna(subset=[col]).copy()
    if c.empty:
        return float("nan"), float("nan")
    c = c.sort_values("lag_s")
    c = c[c["lag_s"] >= 0.0]
    if c.empty:
        return float("nan"), float("nan")

    y = c[col].to_numpy(dtype=float)
    x = c["lag_s"].to_numpy(dtype=float)

    # tau_1e
    thr = math.exp(-1.0)
    m = np.isfinite(y)
    tau_1e = float("nan")
    if np.any(m):
        yy = y[m]
        xx = x[m]
        below = np.where(yy <= thr)[0]
        if below.size > 0:
            tau_1e = float(xx[int(below[0])])

    # tau_int: integrate until first zero crossing
    tau_int = float("nan")
    if np.isfinite(y).any():
        idx0 = np.where(y <= 0.0)[0]
        iend = int(idx0[0]) if idx0.size > 0 else len(y) - 1
        if iend >= 1:
            tau_int = float(np.trapz(y[: iend + 1], x[: iend + 1]))
    return tau_1e, tau_int


# ----------------------------
# Position extraction (best-effort)
# ----------------------------

def _pos_cols(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    """Best-effort identification of position columns.

    Accepts common conventions:
      - x/y/z or X/Y/Z
      - x_gse/y_gse/z_gse, x_rtn/y_rtn/z_rtn, etc.
      - r/t/n or R/T/N
    """
    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
        return None
    cols = list(map(str, df.columns))
    lcols = [c.lower() for c in cols]

    def pick(axis: str) -> Optional[str]:
        # Prefer exact matches, then common suffixed forms.
        for pat in (axis, f"{axis}_gse", f"{axis}_rtn", f"{axis}_sc", f"{axis}_heli", f"{axis}_hci"):
            if pat in lcols:
                return cols[lcols.index(pat)]
        # Then: any column that starts with axis + delimiter or ends with delimiter + axis
        for i, lc in enumerate(lcols):
            if lc == axis:
                return cols[i]
            if lc.startswith(axis + "_") or lc.endswith("_" + axis):
                return cols[i]
        # Finally: startswith axis (covers e.g. 'X_GSE')
        for i, lc in enumerate(lcols):
            if lc.startswith(axis):
                return cols[i]
        return None

    x = pick("x")
    y = pick("y")
    z = pick("z")
    if x and y and z:
        return x, y, z

    r = pick("r")
    t = pick("t")
    n = pick("n")
    if r and t and n:
        return r, t, n

    return None

def _extract_position_series(gen_obj: Any) -> Optional[pd.DataFrame]:
    if gen_obj is None:
        return None

    if isinstance(gen_obj, dict):
        for k in ("sc_pos", "sc_pos_rtn", "SC_pos", "pos", "position", "rtn_pos", "R"):
            v = gen_obj.get(k, None)
            if isinstance(v, pd.DataFrame):
                return _ensure_dtindex(v)
        for v in gen_obj.values():
            if isinstance(v, pd.DataFrame):
                vv = _ensure_dtindex(v)
                if _pos_cols(vv) is not None:
                    return vv

    if isinstance(gen_obj, pd.DataFrame):
        gg = _ensure_dtindex(gen_obj)
        if _pos_cols(gg) is not None:
            return gg

    return None



AU_KM = 1.495978707e8  # km

def _infer_pos_scale_to_km(pos: Optional[pd.DataFrame]) -> float:
    """Infer a scale factor that converts the position vectors in `pos` to km.

    We cannot assume upstream units. We infer from the typical magnitude:
      - ~O(1) .. O(10): likely AU -> multiply by AU_KM
      - ~1e9 or larger: likely meters -> multiply by 1e-3
      - otherwise: assume km
    """
    if pos is None or (not isinstance(pos, pd.DataFrame)) or pos.empty:
        return 1.0
    cc = _pos_cols(pos)
    if cc is None:
        return 1.0
    arr = pos[list(cc)].to_numpy(dtype=float)
    m = np.isfinite(arr).all(axis=1)
    if m.sum() < 10:
        return 1.0
    # sample to avoid huge memory
    samp = arr[m]
    if samp.shape[0] > 5000:
        samp = samp[:5000]
    norms = np.linalg.norm(samp, axis=1)
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return 1.0
    med = float(np.median(norms))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    if med < 1e3:
        # most heliospheric positions are around 1 AU -> treat as AU-like
        return AU_KM
    if med > 1e9:
        # meters -> km
        return 1e-3
    return 1.0


def _speed_vec_to_kmps(v: np.ndarray) -> np.ndarray:
    """Best-effort convert a 3-vector speed to km/s (vector, not magnitude)."""
    vv = np.asarray(v, dtype=float).reshape(-1)
    if vv.size != 3 or (not np.isfinite(vv).any()):
        return vv
    mag = float(np.linalg.norm(vv[np.isfinite(vv)]))
    if not np.isfinite(mag):
        return vv
    # heuristic: km/s ~ O(10^2-10^3); m/s ~ O(10^5-10^6)
    return vv * 1e-3 if mag > 2e4 else vv


def _pos_vec_at(pos: pd.DataFrame, t: pd.Timestamp, *, scale_to_km: float = 1.0) -> Optional[np.ndarray]:
    if pos is None or pos.empty:
        return None
    t = pd.Timestamp(t)
    idx = pos.index.get_indexer([t], method="nearest")
    if idx.size != 1 or idx[0] < 0:
        return None
    row = pos.iloc[int(idx[0])]
    cc = _pos_cols(pos)
    if cc is None:
        return None
    return row[list(cc)].to_numpy(dtype=float) * float(scale_to_km)


# ----------------------------
# Plotting
# ----------------------------

def _plot_pair_hodogram_3d(
    *,
    out_png: Path,
    scA: str,
    scB: str,
    B_cols: Sequence[str],
    B_A: pd.DataFrame,
    B_B: pd.DataFrame,
    mva_A: Optional[MVAResult],
    mva_B: Optional[MVAResult],
    b_med_hat_A: Optional[np.ndarray],
    b_med_hat_B: Optional[np.ndarray],
    b_ps_hat_A: Optional[np.ndarray],
    b_ps_hat_B: Optional[np.ndarray],
    title: str,
    annot_text: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Do not depend on an external LaTeX installation for figure text.
    # (User environments may set matplotlibrc: text.usetex=True.)
    matplotlib.rcParams["text.usetex"] = False

    out_png = _as_path(out_png)
    _mkdir(out_png.parent)

    # Hodogram in mean-subtracted coordinates (keeps the rotation geometry clear).
    XA = B_A.to_numpy(dtype=float)
    XB = B_B.to_numpy(dtype=float)
    dA = XA - np.nanmean(XA, axis=0, keepdims=True)
    dB = XB - np.nanmean(XB, axis=0, keepdims=True)

    scale = float(np.nanpercentile(np.r_[np.linalg.norm(dA, axis=1), np.linalg.norm(dB, axis=1)], 95))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    colA = "k"
    colB = "0.6"

    ax.plot(dA[:, 0], dA[:, 1], dA[:, 2], lw=1.0, color=colA, alpha=0.35)
    ax.plot(dB[:, 0], dB[:, 1], dB[:, 2], lw=1.0, color=colB, alpha=0.35)

    def _draw_axes(mva: MVAResult, color: str, prefix: str) -> None:
        # By construction: l_hat = max variance, m_hat = intermediate, n_hat = min variance.
        vecs = {
            "max": (mva.l_hat, "-"),
            "int": (mva.m_hat, "--"),
            "min": (mva.n_hat, ":"),
        }
        for name, (v, ls) in vecs.items():
            if v is None or (not np.isfinite(v).all()):
                continue
            vv = v / (np.linalg.norm(v) + 1e-15)
            ax.plot([0, scale * vv[0]], [0, scale * vv[1]], [0, scale * vv[2]], lw=2.0, ls=ls, color=color)
            ax.text(scale * 1.02 * vv[0], scale * 1.02 * vv[1], scale * 1.02 * vv[2], f"{prefix} {name}", color=color, fontsize=8)

    if mva_A is not None:
        _draw_axes(mva_A, colA, scA)
    if mva_B is not None:
        _draw_axes(mva_B, colB, scB)

    def _draw_dir(v: Optional[np.ndarray], color: str, label: str) -> None:
        if v is None or (not np.isfinite(v).all()):
            return
        vv = v / (np.linalg.norm(v) + 1e-15)
        ax.plot([0, scale * vv[0]], [0, scale * vv[1]], [0, scale * vv[2]], lw=2.5, color=color)
        ax.text(scale * 1.05 * vv[0], scale * 1.05 * vv[1], scale * 1.05 * vv[2], label, color=color, fontsize=8)

    _draw_dir(b_med_hat_A, colA, "b_med")
    _draw_dir(b_med_hat_B, colB, "b_med")
    _draw_dir(b_ps_hat_A, colA, "b_PS")
    _draw_dir(b_ps_hat_B, colB, "b_PS")

    ax.set_xlabel(str(B_cols[0]))
    ax.set_ylabel(str(B_cols[1]))
    ax.set_zlabel(str(B_cols[2]))
    ax.set_title(title)

    if annot_text:
        ax.text2D(0.02, 0.98, annot_text, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


def _plot_pair_timeseries(
    *,
    out_png: Path,
    scA: str,
    scB: str,
    B_cols: Sequence[str],
    B_A: pd.DataFrame,
    B_B: pd.DataFrame,
    theta_A: pd.Series,
    theta_B: pd.Series,
    t_start_A: pd.Timestamp,
    t_stop_A: pd.Timestamp,
    t_start_B: pd.Timestamp,
    t_stop_B: pd.Timestamp,
    t_center_A: pd.Timestamp,
    t_center_B: pd.Timestamp,
    t_lead_A: Optional[pd.Timestamp],
    t_trail_A: Optional[pd.Timestamp],
    t_lead_B: Optional[pd.Timestamp],
    t_trail_B: Optional[pd.Timestamp],
    tau_star_s: float,
    title: str,
    tau_star_metric: str = "",
    dr_par: float = float("nan"),
    dr_perp: float = float("nan"),
    dr_par_ps: float = float("nan"),
    dr_perp_ps: float = float("nan"),
    aspect_med: float = float("nan"),
    aspect_ps: float = float("nan"),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Do not depend on an external LaTeX installation for figure text.
    matplotlib.rcParams["text.usetex"] = False

    out_png = _as_path(out_png)
    _mkdir(out_png.parent)

    # plotting window: center on the identified A-event (show ONE event per figure)
    durA = float((pd.Timestamp(t_stop_A) - pd.Timestamp(t_start_A)).total_seconds())
    durB = float((pd.Timestamp(t_stop_B) - pd.Timestamp(t_start_B)).total_seconds())
    pad_s = 1.5 * max(durA, durB, 1.0)
    t0 = pd.Timestamp(t_center_A) - pd.to_timedelta(pad_s, unit="s")
    t1 = pd.Timestamp(t_center_A) + pd.to_timedelta(pad_s, unit="s")

    B_A2 = _ensure_dtindex(B_A)[list(B_cols)]
    BA = B_A2.loc[(B_A2.index >= t0) & (B_A2.index <= t1)].copy()
    B_B2 = _ensure_dtindex(B_B)[list(B_cols)]
    BB = B_B2.loc[(B_B2.index >= t0) & (B_B2.index <= t1)].copy()
    thA2 = _ensure_dtindex(theta_A)
    thA = thA2.loc[(thA2.index >= t0) & (thA2.index <= t1)].copy()
    thB2 = _ensure_dtindex(theta_B)
    thB = thB2.loc[(thB2.index >= t0) & (thB2.index <= t1)].copy()

    # shift B series by -tau_star (for aligned overlay on A-time)
    BBs = BB.copy()
    thBs = thB.copy()
    dt_shift = None
    if np.isfinite(tau_star_s):
        dt_shift = pd.to_timedelta(float(tau_star_s), unit="s")
        BBs.index = BBs.index - dt_shift
        thBs.index = thBs.index - dt_shift

    colA = "k"
    colB = "0.6"

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    def _bmag(df: pd.DataFrame) -> pd.Series:
        arr = df[list(B_cols)].to_numpy(dtype=float)
        return pd.Series(np.sqrt(np.nansum(arr * arr, axis=1)), index=df.index)

    # |B| panel
    bmagA = _bmag(BA)
    bmagBs = _bmag(BBs)

    ax0 = axes[0]
    ax0.plot(bmagA.index, bmagA.to_numpy(dtype=float), lw=1.1, color=colA, label=f"{scA} |B|")
    ax0.plot(bmagBs.index, bmagBs.to_numpy(dtype=float), lw=1.1, color=colB, ls="--", label=f"{scB} |B| (shifted)")

    vv = np.r_[bmagA.to_numpy(dtype=float), bmagBs.to_numpy(dtype=float)]
    vv = vv[np.isfinite(vv)]
    if vv.size:
        p2, p98 = np.percentile(vv, [2, 98])
        if np.isfinite(p2) and np.isfinite(p98) and p98 > p2:
            pad = 0.05 * (p98 - p2)
            ax0.set_ylim(p2 - pad, p98 + pad)

    # event spans (aligned B-span when τ* is finite)
    ax0.axvspan(pd.Timestamp(t_start_A), pd.Timestamp(t_stop_A), color=colA, alpha=0.08)
    if dt_shift is not None:
        ax0.axvspan(pd.Timestamp(t_start_B) - dt_shift, pd.Timestamp(t_stop_B) - dt_shift, color=colB, alpha=0.08)
    else:
        ax0.axvspan(pd.Timestamp(t_start_B), pd.Timestamp(t_stop_B), color=colB, alpha=0.08)

    ax0.set_ylabel("|B|")
    ax0.legend(loc="upper right", fontsize=8)

    # component panels
    for i, comp in enumerate(B_cols):
        ax = axes[i + 1]

        if comp in BA.columns:
            ax.plot(BA.index, BA[comp].to_numpy(dtype=float), lw=1.0, color=colA, label=f"{scA}")
        if comp in BB.columns:
            ax.plot(BB.index, BB[comp].to_numpy(dtype=float), lw=1.0, color=colB, alpha=0.35, label=f"{scB} (raw)")
        if comp in BBs.columns:
            ax.plot(BBs.index, BBs[comp].to_numpy(dtype=float), lw=1.2, color=colB, ls="--", label=f"{scB} (shifted)")

        # robust y-limits
        vals = []
        for ser in (BA.get(comp, None), BB.get(comp, None), BBs.get(comp, None)):
            if ser is None:
                continue
            v = ser.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(v)
        if vals:
            vv = np.concatenate(vals)
            p2, p98 = np.percentile(vv, [2, 98])
            if np.isfinite(p2) and np.isfinite(p98) and p98 > p2:
                pad = 0.05 * (p98 - p2)
                ax.set_ylim(p2 - pad, p98 + pad)

        # event spans (aligned B-span when τ* is finite)
        ax.axvspan(pd.Timestamp(t_start_A), pd.Timestamp(t_stop_A), color=colA, alpha=0.04)
        if dt_shift is not None:
            ax.axvspan(pd.Timestamp(t_start_B) - dt_shift, pd.Timestamp(t_stop_B) - dt_shift, color=colB, alpha=0.04)
        else:
            ax.axvspan(pd.Timestamp(t_start_B), pd.Timestamp(t_stop_B), color=colB, alpha=0.04)

        ax.set_ylabel(str(comp))
        ax.legend(loc="upper right", fontsize=8)

    # Θ panel
    ax = axes[4]
    ax.plot(thA.index, thA.to_numpy(dtype=float), lw=1.0, color=colA, label=f"{scA}")
    ax.plot(thB.index, thB.to_numpy(dtype=float), lw=1.0, color=colB, alpha=0.35, label=f"{scB} (raw)")
    ax.plot(thBs.index, thBs.to_numpy(dtype=float), lw=1.2, color=colB, ls="--", label=f"{scB} (shifted)")

    ax.axvspan(pd.Timestamp(t_start_A), pd.Timestamp(t_stop_A), color=colA, alpha=0.04)
    if dt_shift is not None:
        ax.axvspan(pd.Timestamp(t_start_B) - dt_shift, pd.Timestamp(t_stop_B) - dt_shift, color=colB, alpha=0.04)
    else:
        ax.axvspan(pd.Timestamp(t_start_B), pd.Timestamp(t_stop_B), color=colB, alpha=0.04)

    ax.set_ylabel(r"$\Theta$ [deg]")
    ax.legend(loc="upper right", fontsize=8)

    # annotate shift and key times
    for ax in axes:
        ax.axvline(pd.Timestamp(t_center_A), lw=1.0, color=colA)
        ax.axvline(pd.Timestamp(t_center_B), lw=1.0, color=colB, ls=":")
        if dt_shift is not None:
            ax.axvline(pd.Timestamp(t_center_B) - dt_shift, lw=1.0, color=colB, ls="--")

        if t_lead_A is not None and pd.notna(t_lead_A):
            ax.axvline(pd.Timestamp(t_lead_A), lw=0.9, color=colA, ls="--")
        if t_trail_A is not None and pd.notna(t_trail_A):
            ax.axvline(pd.Timestamp(t_trail_A), lw=0.9, color=colA, ls="--")

        if t_lead_B is not None and pd.notna(t_lead_B):
            tb = pd.Timestamp(t_lead_B) - dt_shift if dt_shift is not None else pd.Timestamp(t_lead_B)
            ax.axvline(tb, lw=0.9, color=colB, ls=":")
        if t_trail_B is not None and pd.notna(t_trail_B):
            tb = pd.Timestamp(t_trail_B) - dt_shift if dt_shift is not None else pd.Timestamp(t_trail_B)
            ax.axvline(tb, lw=0.9, color=colB, ls=":")

    head = title
    if tau_star_metric:
        head += f" | {tau_star_metric}"
    head += (f"  tau*={tau_star_s:.2f}s" if np.isfinite(tau_star_s) else "  tau*=nan")
    axes[0].set_title(head)

    # separation annotation
    sep_txt = []
    if np.isfinite(dr_par) and np.isfinite(dr_perp):
        sep_txt.append(f"dr_par(med)={dr_par:.3g}, dr_perp(med)={dr_perp:.3g}")
    if np.isfinite(dr_par_ps) and np.isfinite(dr_perp_ps):
        sep_txt.append(f"dr_par(PS)={dr_par_ps:.3g}, dr_perp(PS)={dr_perp_ps:.3g}")
    if np.isfinite(aspect_med):
        sep_txt.append(f"aspect(med)={aspect_med:.3g}")
    if np.isfinite(aspect_ps):
        sep_txt.append(f"aspect(PS)={aspect_ps:.3g}")
    if sep_txt:
        axes[0].text(0.01, 0.88, " | ".join(sep_txt), transform=axes[0].transAxes, fontsize=9, va="top")

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


def _plot_pair_corr_curves(
    *,
    out_png: Path,
    scA: str,
    scB: str,
    autoA: pd.DataFrame,
    autoB: pd.DataFrame,
    curves: Dict[str, pd.DataFrame],
    tau_star_s: float,
    tau_star_phys_s: float = float('nan'),
    tau_star_metric: str,
    dr_par: float,
    dr_perp: float,
    dr_par_ps: float,
    dr_perp_ps: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Do not depend on an external LaTeX installation for figure text.
    # (User environments may set matplotlibrc: text.usetex=True.)
    matplotlib.rcParams["text.usetex"] = False

    out_png = _as_path(out_png)
    _mkdir(out_png.parent)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Prefer d_i-normalized lag when available
    xcol = 'lag_s'
    xlabel = 'lag [s]'
    if autoA is not None and ('lag_di' in autoA.columns):
        xcol = 'lag_di'
        xlabel = 'lag [d_i]'


    colA = "k"
    colB = "0.6"

    if autoA is not None and (not autoA.empty) and ("lag_s" in autoA.columns) and ("R" in autoA.columns):
        sub = autoA.copy()
        sub = sub[sub.get("lag_s", 0.0) >= 0]
        axes[0].plot(sub[xcol].to_numpy(dtype=float), sub["R"].to_numpy(dtype=float), color=colA, lw=1.2)
    axes[0].set_ylabel(f"{scA} auto R")

    if autoB is not None and (not autoB.empty) and ("lag_s" in autoB.columns) and ("R" in autoB.columns):
        sub = autoB.copy()
        sub = sub[sub.get("lag_s", 0.0) >= 0]
        axes[1].plot(sub[xcol].to_numpy(dtype=float), sub["R"].to_numpy(dtype=float), color=colB, lw=1.2)
    axes[1].set_ylabel(f"{scB} auto R")

    # cross-correlation curves (minimal palette: black primary, gray secondary)
    for met, curve in curves.items():
        if curve is None or curve.empty or ("lag_s" not in curve.columns):
            continue
        if met == "bhat" and ("dot" in curve.columns):
            y = curve["dot"].to_numpy(dtype=float)
            lab = "bhat dot"
        elif met == "dbvec" and ("R" in curve.columns):
            y = curve["R"].to_numpy(dtype=float)
            lab = "dbvec R"
        elif met.startswith("theta") and ("corr" in curve.columns):
            y = curve["corr"].to_numpy(dtype=float)
            lab = f"{met} corr"
        else:
            continue
        color = colA if met == tau_star_metric else colB
        ls = "-" if met == tau_star_metric else "--"
        xx = curve[xcol].to_numpy(dtype=float) if (xcol in curve.columns) else curve["lag_s"].to_numpy(dtype=float)
        axes[2].plot(xx, y, color=color, lw=1.2, ls=ls, label=lab)

    if np.isfinite(tau_star_s):
        xmark = float(tau_star_s)
        if xcol == "lag_di":
            c0 = curves.get(tau_star_metric, None)
            if c0 is not None and (not c0.empty) and ("lag_s" in c0.columns) and ("lag_di" in c0.columns):
                import numpy as _np
                xs = c0["lag_s"].to_numpy(dtype=float)
                xd = c0["lag_di"].to_numpy(dtype=float)
                m = _np.isfinite(xs) & _np.isfinite(xd)
                if m.sum() >= 2:
                    xmark = float(_np.interp(float(tau_star_s), xs[m], xd[m]))
        axes[2].axvline(xmark, color="k", lw=1.0, ls=":")

    axes[2].set_ylabel("cross")
    axes[2].set_xlabel(xlabel)
    axes[2].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)

    head = title
    if tau_star_metric:
        head += f" | {tau_star_metric}"
    head += (f"  tau_rel={tau_star_s:.2f}s" if np.isfinite(tau_star_s) else "  tau_rel=nan")
    if np.isfinite(tau_star_phys_s):
        head += f"  tau_phys={tau_star_phys_s:.2f}s"
    axes[0].set_title(head)

    sep_txt = []
    if np.isfinite(dr_par) and np.isfinite(dr_perp):
        sep_txt.append(f"dr_par(med)={dr_par:.3g}, dr_perp(med)={dr_perp:.3g}")
    if np.isfinite(dr_par_ps) and np.isfinite(dr_perp_ps):
        sep_txt.append(f"dr_par(PS)={dr_par_ps:.3g}, dr_perp(PS)={dr_perp_ps:.3g}")
    if sep_txt:
        axes[0].text(0.01, 0.88, " | ".join(sep_txt), transform=axes[0].transAxes, fontsize=9, va="top")

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


def _plot_pair_geom_stats(
    *,
    out_png: Path,
    key: str,
    pairs: pd.DataFrame,
) -> None:
    """Compact ensemble geometry summary for a spacecraft pair.

    Produces a single figure containing:
      - |drperp| vs |dr||| scatter (rolling-median decomposition)
      - aspect-ratio histograms (rolling-median, and Parker-spiral when available)
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Do not depend on an external LaTeX installation for figure text.
    # (User environments may set matplotlibrc: text.usetex=True.)
    matplotlib.rcParams["text.usetex"] = False

    out_png = _as_path(out_png)
    _mkdir(out_png.parent)

    if pairs is None or pairs.empty:
        return

    drp = pairs.get("dr_perp", pd.Series(dtype=float)).to_numpy(dtype=float)
    drl = pairs.get("dr_par", pd.Series(dtype=float)).to_numpy(dtype=float)
    asp_med = pairs.get("aspect_ratio_med", pd.Series(dtype=float)).to_numpy(dtype=float)
    asp_ps = pairs.get("aspect_ratio_ps", pd.Series(dtype=float)).to_numpy(dtype=float)

    m = np.isfinite(drp) & np.isfinite(drl)
    x = np.abs(drp[m])
    y = np.abs(drl[m])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    if x.size:
        axes[0].scatter(x, y, s=14, color="k", alpha=0.55)
        axes[0].set_xlabel(r"$|\Delta r_\perp|$ (med)")
        axes[0].set_ylabel(r"$|\Delta r_\parallel|$ (med)")
        axes[0].set_title(f"{key}: separations (N={int(x.size)})")
    else:
        axes[0].text(0.5, 0.5, "No finite separations available", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()

    a1 = asp_med[np.isfinite(asp_med)]
    if a1.size:
        axes[1].hist(a1, bins=40, histtype="step", color="k", lw=1.4, label="aspect (med)")
    a2 = asp_ps[np.isfinite(asp_ps)]
    if a2.size:
        axes[1].hist(a2, bins=40, histtype="step", color="0.6", lw=1.4, label="aspect (PS)")
    axes[1].set_xlabel(r"$|\Delta r_\parallel|/|\Delta r_\perp|$")
    axes[1].set_ylabel("count")
    axes[1].legend(loc="upper right", fontsize=8)

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


# ----------------------------
# Bootstrap correlation statistics (events vs full interval)
# ----------------------------

def _interp_df_to_unit_grid(
    df: pd.DataFrame,
    *,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    cols: Sequence[str],
    n_grid: int,
) -> Optional[np.ndarray]:
    """Interpolate a multicomponent series onto a uniform grid on s∈[0,1]."""
    if df is None or df.empty or n_grid < 20:
        return None
    df = _ensure_dtindex(df)
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    if t1 <= t0:
        return None
    sub = df.loc[(df.index >= t0) & (df.index <= t1), list(cols)]
    if sub.empty:
        return None

    dur = float((t1 - t0).total_seconds())
    if dur <= 0:
        return None

    s = (sub.index - t0).total_seconds() / dur
    s = np.asarray(s, dtype=float)
    if s.size < 10 or (not np.isfinite(s).any()):
        return None
    s_grid = np.linspace(0.0, 1.0, int(n_grid), dtype=float)

    out = np.full((int(n_grid), len(cols)), np.nan, dtype=float)
    for j, c in enumerate(cols):
        y = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(s) & np.isfinite(y)
        if m.sum() < 5:
            continue
        ss = s[m]
        yy = y[m]
        o = np.argsort(ss)
        ss = ss[o]
        yy = yy[o]
        # collapse duplicate ss by mean
        su, inv = np.unique(ss, return_inverse=True)
        if su.size < 5:
            continue
        ysum = np.zeros_like(su)
        cnt = np.zeros_like(su)
        np.add.at(ysum, inv, yy)
        np.add.at(cnt, inv, 1.0)
        yu = ysum / np.where(cnt > 0, cnt, np.nan)
        out[:, j] = np.interp(s_grid, su, yu)
    if not np.isfinite(out).any():
        return None
    return out


def _acf_curve_unit(arr: np.ndarray) -> np.ndarray:
    """Component-wise autocorrelation on a unit grid.

    arr: shape (N, C). Returns shape (N, C) for lags k=0..N-1.
    """
    X = np.asarray(arr, dtype=float)
    if X.ndim != 2:
        raise ValueError("arr must be 2D")
    N, C = X.shape
    out = np.full((N, C), np.nan, dtype=float)
    # mean-subtract per component
    X0 = X - np.nanmean(X, axis=0, keepdims=True)
    for k in range(N):
        a = X0[: N - k, :]
        b = X0[k:, :]
        for c in range(C):
            out[k, c] = _pearson_corr(a[:, c], b[:, c])
    return out


def _ccf_curve_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component-wise cross-correlation on a unit grid (positive lags).

    a,b: shape (N, C) on the same unit grid. Returns shape (N, C) for lags k=0..N-1
    defined as corr(a(t), b(t+τ)).
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError("a and b must be 2D with the same shape")
    N, C = A.shape
    out = np.full((N, C), np.nan, dtype=float)
    A0 = A - np.nanmean(A, axis=0, keepdims=True)
    B0 = B - np.nanmean(B, axis=0, keepdims=True)
    for k in range(N):
        aa = A0[: N - k, :]
        bb = B0[k:, :]
        for c in range(C):
            out[k, c] = _pearson_corr(aa[:, c], bb[:, c])
    return out


def _bootstrap_mean_ci(
    curves: np.ndarray,
    *,
    n_boot: int,
    seed: int = 0,
    lo: float = 16.0,
    hi: float = 84.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap mean curve (resample events with replacement)."""
    X = np.asarray(curves, dtype=float)
    if X.ndim != 3:
        raise ValueError("curves must be (n_events, n_lags, n_comp)")
    n, nlag, ncomp = X.shape
    if n == 0:
        return (
            np.full((nlag, ncomp), np.nan),
            np.full((nlag, ncomp), np.nan),
            np.full((nlag, ncomp), np.nan),
        )
    rng = np.random.default_rng(int(seed))
    boots = np.full((int(n_boot), nlag, ncomp), np.nan, dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[b] = np.nanmean(X[idx, :, :], axis=0)
    mean = np.nanmean(X, axis=0)
    loq = np.nanpercentile(boots, float(lo), axis=0)
    hiq = np.nanpercentile(boots, float(hi), axis=0)
    return mean, loq, hiq


def _plot_corr_ensemble(
    *,
    out_png: Path,
    u: np.ndarray,
    curves: Optional[np.ndarray],
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    comps: Sequence[str],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matplotlib.rcParams["text.usetex"] = False
    out_png = _as_path(out_png)
    _mkdir(out_png.parent)

    C = len(comps)
    fig, axes = plt.subplots(C, 1, figsize=(10, 3.2 * C), sharex=True)
    if C == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        if curves is not None and curves.size:
            # thin per-event curves
            for k in range(curves.shape[0]):
                ax.plot(u, curves[k, :, j], color="0.75", lw=0.7, alpha=0.6)
        ax.fill_between(u, lo[:, j], hi[:, j], alpha=0.25)
        ax.plot(u, mean[:, j], lw=1.5, color="k")
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"R ({comps[j]})")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("normalized lag (0..1)")
    axes[0].set_title(title)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


def _bootstrap_corr_curve_full(
    x: pd.Series,
    y: Optional[pd.Series],
    *,
    tau_max_s: float,
    cadence_s: Optional[float],
    n_boot: int,
    n_pairs: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap correlation curve for full-interval series with controlled sample size."""
    x = _ensure_dtindex(pd.Series(x)).dropna()
    if y is not None:
        y = _ensure_dtindex(pd.Series(y)).dropna()

    if x.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Regular grid: use the coarser (max) cadence.
    dtX = _median_dt_seconds(x.index)
    dtY = _median_dt_seconds(y.index) if y is not None else dtX
    dt_ref = max([d for d in (dtX, dtY) if d is not None], default=1.0)
    if cadence_s is None:
        cadence_s = dt_ref
    else:
        cadence_s = max(float(cadence_s), dt_ref)
    cadence_s = max(float(cadence_s), 1e-6)

    if y is None:
        # autocorr only needs x on a regular grid
        xg = _resample_series_regular(x, cadence_s)
        yg = xg
    else:
        # build a common grid on the overlap
        y = _ensure_dtindex(y)
        t0 = max(x.index[0], y.index[0])
        t1 = min(x.index[-1], y.index[-1])
        if t1 <= t0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        step_ms = int(round(float(cadence_s) * 1000.0))
        step_ms = max(step_ms, 1)
        grid = pd.date_range(t0, t1, freq=f"{step_ms}ms")
        xg = x.reindex(x.index.union(grid)).interpolate(method="time").reindex(grid)
        yg = y.reindex(y.index.union(grid)).interpolate(method="time").reindex(grid)

    xa = pd.to_numeric(xg, errors="coerce").to_numpy(dtype=float)
    ya = pd.to_numeric(yg, errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[m]
    ya = ya[m]
    if xa.size < 50:
        return np.array([]), np.array([]), np.array([]), np.array([])

    xa = xa - np.mean(xa)
    ya = ya - np.mean(ya)

    kmax = int(round(float(tau_max_s) / float(cadence_s)))
    kmax = max(kmax, 1)
    lags = np.arange(0, kmax + 1, dtype=int)
    lag_s = lags.astype(float) * float(cadence_s)
    u = lag_s / float(max(tau_max_s, 1e-9))

    rng = np.random.default_rng(int(seed))
    mean = np.full(lags.shape, np.nan, dtype=float)
    lo = np.full(lags.shape, np.nan, dtype=float)
    hi = np.full(lags.shape, np.nan, dtype=float)

    N = int(xa.size)
    for i, k in enumerate(lags):
        n = N - int(k)
        if n < 20:
            continue
        x0 = xa[:n]
        y0 = ya[int(k): int(k) + n]
        # controlled sample size at this lag
        n_use = int(min(n_pairs, n))
        if n_use < 20:
            continue
        boots = np.full(int(n_boot), np.nan, dtype=float)
        for b in range(int(n_boot)):
            idx = rng.integers(0, n, size=n_use)
            boots[b] = _pearson_corr(x0[idx], y0[idx])
        mean[i] = float(np.nanmean(boots))
        lo[i] = float(np.nanpercentile(boots, 16.0))
        hi[i] = float(np.nanpercentile(boots, 84.0))
    return u, mean, lo, hi


def run_analysis(settings: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    loaded = load_intervals(settings)

    # Particle/sig sourcing controls (used for density lookup tolerance when needed).
    particles_cfg = settings.get("particles", {}) or {}
    if not isinstance(particles_cfg, dict):
        raise TypeError("settings['particles'] must be a dict if provided")
    sig_max_dt = pd.to_timedelta(str(particles_cfg.get("sig_max_dt", "60s")))

    io_cfg = settings.get("io", {}) or {}
    if not isinstance(io_cfg, dict):
        raise TypeError("settings['io'] must be a dict")

    out_dir = _as_path(io_cfg.get("out_dir", Path("./out")))
    overwrite = bool(io_cfg.get("overwrite", False))
    _mkdir(out_dir)

    analysis_cfg = settings.get("analysis", {}) or {}
    if not isinstance(analysis_cfg, dict):
        raise TypeError("settings['analysis'] must be a dict if provided")

    enabled = bool(analysis_cfg.get("enabled", True))
    if not enabled:
        return {"loaded": loaded}

    theta_on = float(analysis_cfg.get("theta_thresh_on_deg", 90.0))
    theta_off = analysis_cfg.get("theta_thresh_off_deg", None)
    theta_off = float(theta_off) if theta_off is not None else None

    roll_window = str(analysis_cfg.get("rolling_median_window", "6H"))

    # Event extraction parameters.
    min_duration_s = float(analysis_cfg.get("min_event_duration_s", 10.0))  # requested: start with 10s
    pad_s = float(analysis_cfg.get("event_pad_s", 0.0))

    # Boundary parameters.
    # `boundaries.method` controls the algorithm. Supported:
    #   - "nonparam_bmag_extrema" (default): threshold-free extrema of d|B|/dt
    #   - "sharp_bmag_drop": step-score with optional amplitude criteria
    bnd_cfg = analysis_cfg.get("boundaries", {}) or {}
    bnd_method = str(bnd_cfg.get("method", "nonparam_bmag_extrema"))
    lead_window = pd.to_timedelta(str(bnd_cfg.get("lead_search_window", "2min")))
    trail_window = pd.to_timedelta(str(bnd_cfg.get("trail_search_window", "2min")))
    bmag_smooth = str(bnd_cfg.get("bmag_smooth_window", "2s"))
    bnd_interval = pd.to_timedelta(str(bnd_cfg.get("boundary_interval", "6s")))

    pre_window = pd.to_timedelta(str(bnd_cfg.get("pre_window", "8s")))
    post_window = pd.to_timedelta(str(bnd_cfg.get("post_window", "8s")))
    inside_max = pd.to_timedelta(str(bnd_cfg.get("inside_max", "20s")))
    min_drop_frac = float(bnd_cfg.get("min_drop_frac", 0.03))
    min_drop_abs = float(bnd_cfg.get("min_drop_abs", 0.0))

    # NOTE: previous revisions contained an additional SB-quality filter with
    # ad hoc thresholds. That filter is intentionally removed.
    # SB “quality” is quantified instead via spherical-polarization diagnostics
    # (Ψ, C_B) stored per event.


    # Parker parameters.
    parker_cfg = analysis_cfg.get("parker", {}) or {}
    parker_enabled = bool(parker_cfg.get("enabled", False))
    parker_r0 = float(parker_cfg.get("r0", 10.0))
    parker_av_window = str(parker_cfg.get("av_window", "90min"))
    density_col = analysis_cfg.get("density_col", None)

    # Matching parameters.
    match_cfg = analysis_cfg.get("matching", {}) or {}
    match_window = pd.to_timedelta(str(match_cfg.get("time_window", "30min")))
    top_n = int(match_cfg.get("top_n", 3))

    # Alignment parameters.
    align_cfg = analysis_cfg.get("alignment", {}) or {}
    tau_max_s = float(align_cfg.get("tau_max_s", 600.0))
    cadence_s = align_cfg.get("cadence_s", None)
    cadence_s = float(cadence_s) if cadence_s is not None else None
    metrics = align_cfg.get("metrics", ["dbvec", "theta_med", "theta_ps"])
    if not isinstance(metrics, (list, tuple)):
        raise TypeError("alignment.metrics must be a list")

    # Correlation-function outputs.
    corr_cfg = analysis_cfg.get("correlations", {}) or {}
    compute_event_autocorr = bool(corr_cfg.get("event_autocorr", True))
    autocorr_tau_max_s = float(corr_cfg.get("autocorr_tau_max_s", 120.0))

    # Plot parameters.
    plot_cfg = analysis_cfg.get("plots", {}) or {}
    make_plots = bool(plot_cfg.get("make", True))
    plots_top_k = int(plot_cfg.get("top_k_per_A", top_n))
    make_timeseries = bool(plot_cfg.get("timeseries", True))
    make_corr_plots = bool(plot_cfg.get("correlations", True))
    dir_max_dt = pd.to_timedelta(str(plot_cfg.get("dir_max_dt", "5min")))

    # Bootstrap correlation statistics (events vs full interval).
    corrstats_cfg = analysis_cfg.get("corr_stats", {}) or {}
    corrstats_enabled = bool(corrstats_cfg.get("enabled", True))
    corrstats_n_grid = int(corrstats_cfg.get("n_grid", 256))
    corrstats_n_boot = int(corrstats_cfg.get("n_boot", 200))
    corrstats_full_tau_max_s = float(corrstats_cfg.get("full_tau_max_s", 600.0))
    corrstats_full_cadence_s = corrstats_cfg.get("full_cadence_s", None)
    corrstats_full_cadence_s = float(corrstats_full_cadence_s) if corrstats_full_cadence_s is not None else None
    corrstats_full_n_pairs = int(corrstats_cfg.get("full_n_pairs", 5000))

    by_sc = loaded["by_sc"]
    sc_list = list(loaded["meta"]["sc_list"])

    # Enforce a single, consistent vector-component convention across spacecraft.
    # If not provided explicitly, infer from the first spacecraft and require it for all others.
    if analysis_cfg.get("B_cols", None) is None:
        B0 = _ensure_dtindex(by_sc[sc_list[0]]["Mag"])
        analysis_cfg["B_cols"] = _infer_vec_cols(B0, [("Br", "Bt", "Bn"), ("Bx", "By", "Bz")])
    if analysis_cfg.get("V_cols", None) is None:
        V0 = _ensure_dtindex(by_sc[sc_list[0]]["Par"])
        analysis_cfg["V_cols"] = _infer_vec_cols(V0, [("Vr", "Vt", "Vn"), ("Vx", "Vy", "Vz"), ("vx", "vy", "vz")])

    for _sc in sc_list:
        _B = _ensure_dtindex(by_sc[_sc]["Mag"])
        _V = _ensure_dtindex(by_sc[_sc]["Par"])
        for _c in list(analysis_cfg["B_cols"]):
            if _c not in _B.columns:
                raise KeyError(f"analysis.B_cols={analysis_cfg['B_cols']!r} but {_sc} Mag is missing column {_c!r}. Set analysis.B_cols explicitly to a common basis.")
        for _c in list(analysis_cfg["V_cols"]):
            if _c not in _V.columns:
                raise KeyError(f"analysis.V_cols={analysis_cfg['V_cols']!r} but {_sc} Par is missing column {_c!r}. Set analysis.V_cols explicitly to a common basis.")

    # --------------------
    # Stage-1: diagnostics + events (per spacecraft)
    # --------------------
    diagnostics_by_sc: Dict[str, pd.DataFrame] = {}
    events_by_sc: Dict[str, pd.DataFrame] = {}

    for sc in sc_list:
        scd = by_sc[sc]
        B = _ensure_dtindex(scd["Mag"])
        V = _ensure_dtindex(scd["Par"])
        Sig = scd.get("Sig", None)

        B_cols = analysis_cfg.get("B_cols", None)
        if B_cols is None:
            B_cols = _infer_vec_cols(B, [("Br", "Bt", "Bn"), ("Bx", "By", "Bz")])
        else:
            B_cols = list(B_cols)

        V_cols = analysis_cfg.get("V_cols", None)
        if V_cols is None:
            V_cols = _infer_vec_cols(V, [("Vr", "Vt", "Vn"), ("Vx", "Vy", "Vz"), ("vx", "vy", "vz")])
        else:
            V_cols = list(V_cols)

        Bv = B[B_cols].copy()
        Vv = V[V_cols].copy()

        # Rolling-median background direction + deflection.
        B_med = _rolling_median_direction(Bv, roll_window)
        theta_med = pd.Series(
            _angle_deg(Bv.to_numpy(dtype=float), B_med.to_numpy(dtype=float)),
            index=Bv.index,
            name="Theta_med_deg",
        )

        # Parker-spiral diagnostics (optional).
        B_ps = None
        theta_ps = None
        if parker_enabled:
            if density_col is None:
                raise KeyError("analysis.parker.enabled=True but analysis.density_col was not provided")
            have_sig = isinstance(Sig, pd.DataFrame) and (density_col in Sig.columns)
            if density_col not in V.columns and density_col not in B.columns and (not have_sig):
                raise KeyError(f"density_col={density_col!r} not found in Par, Mag, or Sig DataFrame columns")

            if density_col in V.columns:
                dens = V[density_col]
            elif density_col in B.columns:
                dens = B[density_col]
            else:
                # When Sig is provider-sourced from another spacecraft, the loader may have already reindexed
                # to the target Par grid. We still reindex to Bv for safety.
                dens = Sig[density_col]  # type: ignore[index]

            if not isinstance(dens, pd.Series):
                dens = pd.Series(dens, index=getattr(dens, "index", Bv.index))
            dens = dens.reindex(Bv.index, method="nearest", tolerance=sig_max_dt)
            B_ps, theta_ps = _compute_parker_deflection(
                Bv,
                Vv,
                dens,
                r0=parker_r0,
                av_window=parker_av_window,
            )

        gate = str(analysis_cfg.get("sb_gate", "OR")).upper()
        is_ps = (theta_ps > theta_on) if theta_ps is not None else pd.Series(False, index=Bv.index)
        is_med = theta_med > theta_on
        if gate == "AND":
            is_sb = (is_ps & is_med).astype(bool)
        else:
            is_sb = (is_ps | is_med).astype(bool)

        if theta_off is not None:
            mask_on = is_sb.to_numpy(dtype=bool)
            mask_off = (theta_med.to_numpy(dtype=float) >= float(theta_off))
            is_sb = pd.Series(_apply_hysteresis(mask_on, mask_off), index=is_sb.index, name="is_sb")
        else:
            is_sb = is_sb.rename("is_sb")

        diag = pd.DataFrame(index=Bv.index)
        diag["Theta_med_deg"] = theta_med
        diag["is_sb"] = is_sb.astype(bool)
        Bm_u = _unitvec(B_med.to_numpy(dtype=float))
        diag["b_med_1"] = Bm_u[:, 0]
        diag["b_med_2"] = Bm_u[:, 1]
        diag["b_med_3"] = Bm_u[:, 2]

        # Ion inertial length d_i from Sig (preferred) when density is available.
        # Stored for correlation-vs-separation in units of d_i.
        if isinstance(Sig, pd.DataFrame):
            cand_cols = []
            if density_col is not None:
                cand_cols.append(str(density_col))
            cand_cols += ["np", "n_p", "Np", "n", "density", "N"]
            dens_col = next((c for c in cand_cols if c in Sig.columns), None)
            if dens_col is not None:
                dens = Sig[dens_col]
                if not isinstance(dens, pd.Series):
                    dens = pd.Series(dens, index=getattr(dens, "index", Bv.index))
                dens = dens.reindex(Bv.index, method="nearest", tolerance=sig_max_dt)
                n_cm3 = pd.to_numeric(dens, errors="coerce").to_numpy(dtype=float)
                diag["n_cm3"] = n_cm3
                diag["di_m"] = np.where((n_cm3 > 0) & np.isfinite(n_cm3), 2.28e5 / np.sqrt(n_cm3), np.nan)

        if theta_ps is not None and B_ps is not None:
            diag["Theta_PS_deg"] = theta_ps
            Bps_u = _unitvec(B_ps.to_numpy(dtype=float))
            diag["b_ps_1"] = Bps_u[:, 0]
            diag["b_ps_2"] = Bps_u[:, 1]
            diag["b_ps_3"] = Bps_u[:, 2]

        diagnostics_by_sc[sc] = diag

        # Event extraction: core segments from is_sb.
        segs = _rle_true_segments(is_sb.to_numpy(dtype=bool))
        rows: List[Dict[str, Any]] = []
        corr_dir = out_dir / "corr_curves" / sc
        _mkdir(corr_dir)

        for j, (i0, i1) in enumerate(segs):
            t0_core = Bv.index[i0]
            t1_core = Bv.index[i1]

            # padded window for scoring
            t0p = t0_core - pd.to_timedelta(pad_s, unit="s")
            t1p = t1_core + pd.to_timedelta(pad_s, unit="s")

            seg_theta = theta_med.iloc[i0 : i1 + 1]
            t_center = pd.Timestamp(seg_theta.idxmax()) if not seg_theta.empty else pd.Timestamp(t0_core + (t1_core - t0_core) / 2)

            # Boundaries around the core.
            # Default: non-parametric extrema of d|B|/dt (threshold-free).
            _bm = str(bnd_method).strip().lower()
            if _bm in ("sharp_bmag_drop", "sharp"):
                tL, tT, bmeta = _boundary_times_sharp_bmag_drop(
                    Bv,
                    t_core_start=pd.Timestamp(t0_core),
                    t_core_stop=pd.Timestamp(t1_core),
                    lead_search=lead_window,
                    trail_search=trail_window,
                    pre_window=pre_window,
                    post_window=post_window,
                    smooth_window=bmag_smooth,
                    inside_max=inside_max,
                    min_drop_frac=min_drop_frac,
                    min_drop_abs=min_drop_abs,
                )
            else:
                tL, tT, bmeta = _boundary_times_nonparam_bmag_extrema(
                    Bv,
                    t_core_start=pd.Timestamp(t0_core),
                    t_core_stop=pd.Timestamp(t1_core),
                    lead_search=lead_window,
                    trail_search=trail_window,
                    smooth_window=bmag_smooth,
                    inside_max=inside_max,
                )

            # Define physical event extent: prefer [tL,tT] when both are valid and ordered.
            t_start = pd.Timestamp(t0_core)
            t_stop = pd.Timestamp(t1_core)
            if tL is not None and tT is not None and (pd.Timestamp(tL) < pd.Timestamp(tT)):
                t_start = pd.Timestamp(tL)
                t_stop = pd.Timestamp(tT)
            elif tL is not None:
                t_start = pd.Timestamp(tL)
            elif tT is not None:
                t_stop = pd.Timestamp(tT)

            dur_s = float((t_stop - t_start).total_seconds())
            if dur_s < min_duration_s:
                continue

            # Boundary intervals for MVAB diagnostics.
            IL0 = pd.Timestamp(tL) - bnd_interval / 2 if tL is not None else None
            IL1 = pd.Timestamp(tL) + bnd_interval / 2 if tL is not None else None
            IT0 = pd.Timestamp(tT) - bnd_interval / 2 if tT is not None else None
            IT1 = pd.Timestamp(tT) + bnd_interval / 2 if tT is not None else None

            # Event-level MVA (required) on the PHYSICAL extent.
            B_evt = Bv.loc[(Bv.index >= t_start) & (Bv.index <= t_stop)]
            mva_evt = _compute_mva(B_evt)

            # Boundary MVAB (optional, for propagation products).
            mva_L = _compute_mva(Bv.loc[(Bv.index >= IL0) & (Bv.index <= IL1)]) if (IL0 is not None and IL1 is not None) else None
            mva_T = _compute_mva(Bv.loc[(Bv.index >= IT0) & (Bv.index <= IT1)]) if (IT0 is not None and IT1 is not None) else None


            # Spherical polarization diagnostics (Ψ, C_B) computed on the physical event extent.
            psi = float('nan')
            psi_lambda1 = float('nan')
            psi_lambda2 = float('nan')
            C_B = float('nan')
            if B_evt is not None and (not B_evt.empty):
                psi, psi_lambda1, psi_lambda2 = _psi_perp_isotropy(B_evt)
                C_B = _magnetic_compressibility(B_evt)

            # Event autocorrelation (vector δB) for auditing / correlation-time proxies.
            autocorr_path = ""
            tau1e = float("nan")
            tauint = float("nan")
            if compute_event_autocorr and (B_evt is not None) and (not B_evt.empty):
                ac = _vector_corr_curve(B_evt, None, tau_max_s=autocorr_tau_max_s, cadence_s=cadence_s)
                # keep only non-negative lags for autocorr outputs
                ac2 = ac.sort_values("lag_s").reset_index(drop=True)
                # Add convective-lag mapping in units of d_i when available.
                # Use event-mean flow speed magnitude (m/s) and event-median d_i from diagnostics.
                di_ref = float('nan')
                if 'di_m' in diag.columns:
                    di_vals = diag.loc[(diag.index >= t_start) & (diag.index <= t_stop), 'di_m'].to_numpy(dtype=float)
                    di_vals = di_vals[np.isfinite(di_vals)]
                    if di_vals.size:
                        di_ref = float(np.median(di_vals))
                uref = Vv.loc[(Vv.index >= t_start) & (Vv.index <= t_stop)].mean().to_numpy(dtype=float)
                uref_mps = _speed_to_mps(uref)
                if np.isfinite(di_ref) and di_ref > 0 and np.isfinite(uref_mps):
                    ac2['lag_di'] = (uref_mps * ac2['lag_s'].to_numpy(dtype=float)) / di_ref
                out_csv = corr_dir / f"event_{j:04d}_autocorr_dbvec.csv"
                ac2.to_csv(str(out_csv), index=False)
                autocorr_path = str(out_csv)
                tau1e, tauint = _corr_time_proxies(ac2, col="R")

            row: Dict[str, Any] = {
                "sc": sc,
                "event_id": int(j),
                "t_start": pd.Timestamp(t_start),
                "t_stop": pd.Timestamp(t_stop),
                "t_center": pd.Timestamp(t_center),
                "duration_s": float(dur_s),
                "t_core_start": pd.Timestamp(t0_core),
                "t_core_stop": pd.Timestamp(t1_core),
                "max_theta_med_deg": float(np.nanmax(seg_theta.to_numpy(dtype=float))) if len(seg_theta) else float("nan"),
                "psi_perp": float(psi),
                "psi_lambda1": float(psi_lambda1),
                "psi_lambda2": float(psi_lambda2),
                "C_B": float(C_B),
                "log10_C_B": float(np.log10(C_B)) if (np.isfinite(C_B) and C_B > 0) else float("nan"),
                "t_lead": pd.Timestamp(tL) if tL is not None else pd.NaT,
                "t_trail": pd.Timestamp(tT) if tT is not None else pd.NaT,
                "lead_i0": pd.Timestamp(IL0) if IL0 is not None else pd.NaT,
                "lead_i1": pd.Timestamp(IL1) if IL1 is not None else pd.NaT,
                "trail_i0": pd.Timestamp(IT0) if IT0 is not None else pd.NaT,
                "trail_i1": pd.Timestamp(IT1) if IT1 is not None else pd.NaT,
                "bnd_method": str(bmeta.get("method", "")),
                "bnd_ok_lead": bool(bmeta.get("ok_lead", False)),
                "bnd_ok_trail": bool(bmeta.get("ok_trail", False)),
                "autocorr_path": autocorr_path,
                "tau_1e_s": float(tau1e),
                "tau_int_s": float(tauint),
            }

            def _store_mva(prefix: str, mva: Optional[MVAResult]) -> None:
                if mva is None:
                    row[f"{prefix}_lambda1"] = float("nan")
                    row[f"{prefix}_lambda2"] = float("nan")
                    row[f"{prefix}_lambda3"] = float("nan")
                    row[f"{prefix}_ratio_lm"] = float("nan")
                    row[f"{prefix}_ratio_mn"] = float("nan")
                    for name in ("L", "M", "N"):
                        row[f"{prefix}_{name}1"] = float("nan")
                        row[f"{prefix}_{name}2"] = float("nan")
                        row[f"{prefix}_{name}3"] = float("nan")
                    return
                e = mva.evals
                row[f"{prefix}_lambda1"] = float(e[0])
                row[f"{prefix}_lambda2"] = float(e[1])
                row[f"{prefix}_lambda3"] = float(e[2])
                row[f"{prefix}_ratio_lm"] = float(mva.ratio_lm)
                row[f"{prefix}_ratio_mn"] = float(mva.ratio_mn)
                for name, vec in ("L", mva.l_hat), ("M", mva.m_hat), ("N", mva.n_hat):
                    row[f"{prefix}_{name}1"] = float(vec[0])
                    row[f"{prefix}_{name}2"] = float(vec[1])
                    row[f"{prefix}_{name}3"] = float(vec[2])

            _store_mva("mva_evt", mva_evt)
            _store_mva("mva_lead", mva_L)
            _store_mva("mva_trail", mva_T)

            rows.append(row)

        ev = pd.DataFrame(rows)
        ev = _stable_sort_df(ev, by=["t_start"]) if not ev.empty else ev
        events_by_sc[sc] = ev

        # Stage-1 outputs
        diag_path = out_dir / f"diagnostics_{sc}.pkl"
        ev_path = out_dir / f"events_{sc}.pkl"
        if overwrite or (not diag_path.exists()):
            diag.to_pickle(str(diag_path))
        if overwrite or (not ev_path.exists()):
            ev.to_pickle(str(ev_path))

    # --------------------
    # Stage-2/3/4/5/6: pairwise analysis
    # --------------------

    candidates_by_pair: Dict[str, pd.DataFrame] = {}
    pairs_by_pair: Dict[str, pd.DataFrame] = {}
    stats_by_pair: Dict[str, pd.DataFrame] = {}

    for scA, scB in combinations(sc_list, 2):
        key = f"{scA}__{scB}"
        evA = events_by_sc.get(scA, pd.DataFrame())
        evB = events_by_sc.get(scB, pd.DataFrame())
        if evA.empty or evB.empty:
            continue

        cand_rows: List[Dict[str, Any]] = []
        diagA = diagnostics_by_sc[scA]
        diagB = diagnostics_by_sc[scB]
        # Morphology similarity must be lag-aware; do not intersect timestamps (biases toward τ≈0).
        match_tau_max_s = float(match_cfg.get("tau_max_s", min(600.0, float(match_window.total_seconds()))))
        match_cadence_s = match_cfg.get("cadence_s", None)
        match_cadence_s = float(match_cadence_s) if match_cadence_s is not None else None

        for _, a in evA.iterrows():
            tA = pd.Timestamp(a["t_center"])
            dt = (evB["t_center"] - tA).abs()
            pool = evB.loc[dt <= match_window].copy()
            if pool.empty:
                continue

            t0A = pd.Timestamp(a["t_start"]) - pd.to_timedelta(pad_s, unit="s")
            t1A = pd.Timestamp(a["t_stop"]) + pd.to_timedelta(pad_s, unit="s")
            sA = diagA.loc[(diagA.index >= t0A) & (diagA.index <= t1A), "Theta_med_deg"]

            for _, b in pool.iterrows():
                tB = pd.Timestamp(b["t_center"])
                t0B = pd.Timestamp(b["t_start"]) - pd.to_timedelta(pad_s, unit="s")
                t1B = pd.Timestamp(b["t_stop"]) + pd.to_timedelta(pad_s, unit="s")
                sB = diagB.loc[(diagB.index >= t0B) & (diagB.index <= t1B), "Theta_med_deg"]

                curve_m = _scalar_corr_curve(sA, sB, tau_max_s=match_tau_max_s, cadence_s=match_cadence_s)
                tau_m, morph_r, morph_w = _peak_and_width(curve_m, col="corr")

                dt_s = float((tB - tA).total_seconds())
                dur_ratio = float(b["duration_s"] / a["duration_s"]) if float(a["duration_s"]) > 0 else float("nan")

                dt0 = float(match_cfg.get("dt_scale_s", 600.0))
                w_time = float(match_cfg.get("w_time", 1.0))
                w_morph = float(match_cfg.get("w_morph", 1.0))
                w_dur = float(match_cfg.get("w_dur", 0.5))

                time_score = math.exp(-0.5 * (dt_s / dt0) ** 2)
                morph_score = max(0.0, float(morph_r)) if np.isfinite(morph_r) else 0.0
                dur_pen = abs(math.log(dur_ratio)) if np.isfinite(dur_ratio) else 0.0
                score = w_time * time_score + w_morph * morph_score - w_dur * dur_pen

                cand_rows.append(
                    {
                        "scA": scA,
                        "scB": scB,
                        "event_id_A": int(a["event_id"]),
                        "event_id_B": int(b["event_id"]),
                        "t_center_A": tA,
                        "t_center_B": tB,
                        "dt_center_s": float(dt_s),
                        "morph_r_star_theta_med": float(morph_r) if np.isfinite(morph_r) else float("nan"),
                        "morph_tau_star_s": float(tau_m) if np.isfinite(tau_m) else float("nan"),
                        "morph_width_s": float(morph_w) if np.isfinite(morph_w) else float("nan"),
                        "dur_ratio": float(dur_ratio),
                        "score": float(score),
                        "time_score": float(time_score),
                        "morph_score": float(morph_score),
                        "dur_pen": float(dur_pen),
                    }
                )

        cand = pd.DataFrame(cand_rows)
        if cand.empty:
            continue

        cand = _stable_sort_df(cand, by=["event_id_A", "score"])
        cand = cand.groupby("event_id_A", as_index=False, group_keys=False).apply(lambda g: g.sort_values("score", ascending=False).head(top_n))
        cand = cand.reset_index(drop=True)
        candidates_by_pair[key] = cand

        cand_path = out_dir / f"candidates_{key}.pkl"
        if overwrite or (not cand_path.exists()):
            cand.to_pickle(str(cand_path))

        # Stage-3+: alignment + pair products
        pair_rows: List[Dict[str, Any]] = []
        curves_dir = out_dir / "xcorr_curves" / key
        figs_dir = out_dir / "figures" / key
        _mkdir(curves_dir)
        _mkdir(figs_dir)

        B_A_full = _ensure_dtindex(by_sc[scA]["Mag"])
        B_B_full = _ensure_dtindex(by_sc[scB]["Mag"])

        B_cols = analysis_cfg.get("B_cols", None)
        if B_cols is None:
            B_cols_A = _infer_vec_cols(B_A_full, [("Br", "Bt", "Bn"), ("Bx", "By", "Bz")])
            B_cols_B = _infer_vec_cols(B_B_full, [("Br", "Bt", "Bn"), ("Bx", "By", "Bz")])
            if list(B_cols_A) != list(B_cols_B):
                # enforce a consistent ordering but keep physical meaning; prefer A's cols
                B_cols = list(B_cols_A)
            else:
                B_cols = list(B_cols_A)
        else:
            B_cols = list(B_cols)

        B_A_full = B_A_full[list(B_cols)]
        B_B_full = B_B_full[list(B_cols)]

        # Position products are best-effort. Prefer Gen, but fall back to Par/Mag if positions were stored there.
        posA = _extract_position_series(by_sc[scA].get("Gen", None))
        if posA is None:
            posA = _extract_position_series(by_sc[scA].get("Par", None))
        if posA is None:
            posA = _extract_position_series(by_sc[scA].get("Mag", None))

        posB = _extract_position_series(by_sc[scB].get("Gen", None))
        if posB is None:
            posB = _extract_position_series(by_sc[scB].get("Par", None))
        if posB is None:
            posB = _extract_position_series(by_sc[scB].get("Mag", None))
        # Infer position units and convert separations to km for consistent advection correction.
        posA_scale_km = _infer_pos_scale_to_km(posA)
        posB_scale_km = _infer_pos_scale_to_km(posB)

        # Velocity for advection correction (A spacecraft, event-mean)
        V_A = _ensure_dtindex(by_sc[scA]["Par"])
        V_cols = analysis_cfg.get("V_cols", None)
        if V_cols is None:
            V_cols = _infer_vec_cols(V_A, [("Vr", "Vt", "Vn"), ("Vx", "Vy", "Vz"), ("vx", "vy", "vz")])
        V_A = V_A[list(V_cols)]

        for _, c in cand.iterrows():
            a_id = int(c["event_id_A"])
            b_id = int(c["event_id_B"])
            a = evA.loc[evA["event_id"] == a_id].iloc[0]
            b = evB.loc[evB["event_id"] == b_id].iloc[0]

            t0A = pd.Timestamp(a["t_start"]) - pd.to_timedelta(pad_s, unit="s")
            t1A = pd.Timestamp(a["t_stop"]) + pd.to_timedelta(pad_s, unit="s")
            t0B = pd.Timestamp(b["t_start"]) - pd.to_timedelta(pad_s, unit="s")
            t1B = pd.Timestamp(b["t_stop"]) + pd.to_timedelta(pad_s, unit="s")

            t0 = min(t0A, t0B)
            t1 = max(t1A, t1B)

            # Prepare per-spacecraft windows for correlation/alignment (do not form a union window here)
            BA_win = B_A_full.loc[(B_A_full.index >= t0A) & (B_A_full.index <= t1A)]
            BB_win = B_B_full.loc[(B_B_full.index >= t0B) & (B_B_full.index <= t1B)]

            # Event-centered relative-time copies for cross-correlation.
            # Correlation is computed in a relative-time coordinate where each event center maps to a shared epoch.
            # This makes lag=0 interpretable and avoids spurious τ* driven by absolute timestamp separation.
            t_center_A = pd.Timestamp(a['t_center'])
            t_center_B = pd.Timestamp(b['t_center'])
            dt_centers_s = float((t_center_B - t_center_A).total_seconds())
            _epoch = pd.Timestamp('2000-01-01T00:00:00')
            BA_rel = _shift_to_epoch(BA_win, t_center_A, _epoch)
            BB_rel = _shift_to_epoch(BB_win, t_center_B, _epoch)


            curves: Dict[str, pd.DataFrame] = {}
            peaks: Dict[str, Tuple[float, float, float]] = {}
            curve_paths: Dict[str, str] = {}

            for met in metrics:
                met = str(met)

                if met == "bhat":
                    curve = _vector_dot_curve(BA_rel, BB_rel, tau_max_s=tau_max_s, cadence_s=cadence_s)
                    col = "dot"
                elif met == "dbvec":
                    curve = _vector_corr_curve(BA_rel, BB_rel, tau_max_s=tau_max_s, cadence_s=cadence_s)
                    col = "R"
                elif met == "theta_med":
                    x = diagA.loc[(diagA.index >= t0A) & (diagA.index <= t1A), "Theta_med_deg"]
                    y = diagB.loc[(diagB.index >= t0B) & (diagB.index <= t1B), "Theta_med_deg"]
                    x = _shift_to_epoch(x, t_center_A, _epoch)
                    y = _shift_to_epoch(y, t_center_B, _epoch)
                    curve = _scalar_corr_curve(x, y, tau_max_s=tau_max_s, cadence_s=cadence_s)
                    col = "corr"
                elif met == "theta_ps" and ("Theta_PS_deg" in diagA.columns) and ("Theta_PS_deg" in diagB.columns):
                    x = diagA.loc[(diagA.index >= t0A) & (diagA.index <= t1A), "Theta_PS_deg"]
                    y = diagB.loc[(diagB.index >= t0B) & (diagB.index <= t1B), "Theta_PS_deg"]
                    x = _shift_to_epoch(x, t_center_A, _epoch)
                    y = _shift_to_epoch(y, t_center_B, _epoch)
                    curve = _scalar_corr_curve(x, y, tau_max_s=tau_max_s, cadence_s=cadence_s)
                    col = "corr"
                else:
                    continue

                out_csv = curves_dir / f"A{a_id:04d}_B{b_id:04d}_{met}.csv"
                if 'lag_s' in curve.columns:
                    curve = curve.copy()
                    curve['lag_phys_s'] = curve['lag_s'].to_numpy(dtype=float) + dt_centers_s
                curve.to_csv(str(out_csv), index=False)
                curve_paths[met] = str(out_csv)
                curves[met] = curve
                peaks[met] = _peak_and_width(curve, col=col)


            # Choose τ* from a declared primary metric, with a robust fallback.
            # NOTE: use the validated alignment configuration parsed near the top of run_analysis.
            primary_metric = str(align_cfg.get("primary_metric", "bhat"))
            tau_star = float("nan")
            r_star = float("nan")
            width_star = float("nan")
            tau_star_metric = ""

            def _is_good(p: Tuple[float, float, float]) -> bool:
                return np.isfinite(p[0]) and np.isfinite(p[1])

            if primary_metric in peaks and _is_good(peaks[primary_metric]):
                tau_star, r_star, width_star = peaks[primary_metric]
                tau_star_metric = primary_metric
            else:
                best_met = None
                best_r = -np.inf
                for met, p in peaks.items():
                    if _is_good(p) and float(p[1]) > best_r:
                        best_r = float(p[1])
                        best_met = met
                if best_met is not None:
                    tau_star, r_star, width_star = peaks[best_met]
                    tau_star_metric = str(best_met)


            # Convert the *relative* lag (event-centered) into a physical clock-time shift.
            tau_star_rel = float(tau_star)
            tau_star_phys = float('nan')
            if np.isfinite(tau_star_rel):
                tau_star_phys = dt_centers_s + tau_star_rel

            # Physics-based goodness score (used to decide which candidate pairs are worth plotting).
            # Components:
            #   - peak correlation (primary metric)
            #   - peak sharpness via width proxy (narrower peak preferred)
            #   - duration ratio penalty
            #   - boundary consistency penalty (aligned leading/trailing times)
            dur_ratio = float(b.get("duration_s", float('nan'))) / float(a.get("duration_s", float('nan'))) if (float(a.get("duration_s", 0.0)) > 0) else float('nan')
            pen_dur = abs(math.log(dur_ratio)) if (np.isfinite(dur_ratio) and dur_ratio > 0) else 1.0
            # Normalize width by a characteristic event duration (A)
            Tref = float(a.get("duration_s", float('nan')))
            width_norm = float(width_star) / Tref if (np.isfinite(width_star) and np.isfinite(Tref) and Tref > 0) else float('nan')
            sharp = (max(0.0, float(r_star)) / (1.0 + (width_norm if np.isfinite(width_norm) else 1.0)))
            # Boundary alignment penalty: evaluate after tau* is known (use core start/stop if boundaries are NaT)
            pen_bnd = 0.0
            if np.isfinite(tau_star_phys):
                dt = float(tau_star_phys)
                tLa = a.get('t_lead', pd.NaT)
                tTa = a.get('t_trail', pd.NaT)
                tLb = b.get('t_lead', pd.NaT)
                tTb = b.get('t_trail', pd.NaT)
                if pd.notna(tLa) and pd.notna(tLb):
                    pen_bnd += abs((pd.Timestamp(tLb) - pd.to_timedelta(dt, unit='s') - pd.Timestamp(tLa)).total_seconds()) / max(Tref, 1.0)
                if pd.notna(tTa) and pd.notna(tTb):
                    pen_bnd += abs((pd.Timestamp(tTb) - pd.to_timedelta(dt, unit='s') - pd.Timestamp(tTa)).total_seconds()) / max(Tref, 1.0)
            phys_score = float(sharp * math.exp(-pen_dur) * math.exp(-pen_bnd)) if np.isfinite(sharp) else float('nan')
            # Geometry products (best-effort)
            t_ref = pd.Timestamp(a["t_center"])
            rA = _pos_vec_at(posA, t_ref, scale_to_km=posA_scale_km) if posA is not None else None
            rB = _pos_vec_at(posB, t_ref, scale_to_km=posB_scale_km) if posB is not None else None
            dr = (rB - rA) if (rA is not None and rB is not None) else None

            # Reference directions at t_ref (nearest finite)
            b_med_hat_A = _nearest_finite_row(diagA, t_ref, ["b_med_1", "b_med_2", "b_med_3"], max_dt=dir_max_dt)
            b_med_hat_B = _nearest_finite_row(diagB, t_ref, ["b_med_1", "b_med_2", "b_med_3"], max_dt=dir_max_dt)
            b_ps_hat_A = _nearest_finite_row(diagA, t_ref, ["b_ps_1", "b_ps_2", "b_ps_3"], max_dt=dir_max_dt) if "b_ps_1" in diagA.columns else None
            b_ps_hat_B = _nearest_finite_row(diagB, t_ref, ["b_ps_1", "b_ps_2", "b_ps_3"], max_dt=dir_max_dt) if "b_ps_1" in diagB.columns else None

            # U_ref: mean plasma velocity for A over A-event
            U_ref = V_A.loc[(V_A.index >= pd.Timestamp(a["t_start"])) & (V_A.index <= pd.Timestamp(a["t_stop"]))].mean().to_numpy(dtype=float)
            U_ref_kmps = _speed_vec_to_kmps(U_ref)

            dr_eff = None
            if dr is not None and np.isfinite(tau_star_phys):
                dr_eff = dr - U_ref_kmps * float(tau_star_phys)

            def _decomp(dv: Optional[np.ndarray], b0: Optional[np.ndarray]) -> Tuple[float, float]:
                if dv is None or b0 is None or (not np.isfinite(dv).all()) or (not np.isfinite(b0).all()):
                    return float("nan"), float("nan")
                b0u = b0 / (np.linalg.norm(b0) + 1e-15)
                dr_par = float(np.dot(dv, b0u))
                dr_perp = float(np.linalg.norm(dv - dr_par * b0u))
                return dr_par, dr_perp

            dr_par, dr_perp = _decomp(dr, b_med_hat_A)
            dr_eff_par, dr_eff_perp = _decomp(dr_eff, b_med_hat_A)
            dr_par_ps, dr_perp_ps = _decomp(dr, b_ps_hat_A)
            dr_eff_par_ps, dr_eff_perp_ps = _decomp(dr_eff, b_ps_hat_A)

            # Boundary timing offsets + normal speed/thickness (best-effort)
            dt_lead = float("nan")
            dt_trail = float("nan")
            if pd.notna(a.get("t_lead", pd.NaT)) and pd.notna(b.get("t_lead", pd.NaT)):
                dt_lead = float((pd.Timestamp(b["t_lead"]) - pd.Timestamp(a["t_lead"])).total_seconds())
            if pd.notna(a.get("t_trail", pd.NaT)) and pd.notna(b.get("t_trail", pd.NaT)):
                dt_trail = float((pd.Timestamp(b["t_trail"]) - pd.Timestamp(a["t_trail"])).total_seconds())

            def _vec_from_row(row: pd.Series, prefix: str, which: str) -> Optional[np.ndarray]:
                try:
                    v = np.array([float(row[f"{prefix}_{which}1"]), float(row[f"{prefix}_{which}2"]), float(row[f"{prefix}_{which}3"])])
                    return v if np.isfinite(v).all() else None
                except Exception:
                    return None

            n_lead = _vec_from_row(a, "mva_lead", "N") or _vec_from_row(a, "mva_evt", "N")
            n_trail = _vec_from_row(a, "mva_trail", "N") or _vec_from_row(a, "mva_evt", "N")

            def _vn_and_delta(dt_bnd: float, n_hat: Optional[np.ndarray]) -> Tuple[float, float]:
                if dr is None or n_hat is None or (not np.isfinite(dt_bnd)) or abs(dt_bnd) <= 0:
                    return float("nan"), float("nan")
                nn = n_hat / (np.linalg.norm(n_hat) + 1e-15)
                vn = float(np.dot(dr, nn) / dt_bnd)
                dt_int = float(bnd_interval.total_seconds())
                u_n = float(np.dot(U_ref_kmps, nn))
                delta = abs(u_n - vn) * dt_int
                return vn, delta

            vn_lead, delta_lead = _vn_and_delta(dt_lead, n_lead)
            vn_trail, delta_trail = _vn_and_delta(dt_trail, n_trail)

            # Reconstruct event-level MVA objects for plotting.
            def _mva_from_event_row(row: pd.Series, prefix: str) -> Optional[MVAResult]:
                try:
                    evals = np.array([float(row[f"{prefix}_lambda1"]), float(row[f"{prefix}_lambda2"]), float(row[f"{prefix}_lambda3"])])
                    if not np.isfinite(evals).all():
                        return None
                    L = np.array([float(row[f"{prefix}_L1"]), float(row[f"{prefix}_L2"]), float(row[f"{prefix}_L3"])])
                    M = np.array([float(row[f"{prefix}_M1"]), float(row[f"{prefix}_M2"]), float(row[f"{prefix}_M3"])])
                    N = np.array([float(row[f"{prefix}_N1"]), float(row[f"{prefix}_N2"]), float(row[f"{prefix}_N3"])])
                    evecs = np.column_stack([N, M, L])
                    return MVAResult(evals=evals, evecs=evecs)
                except Exception:
                    return None

            mvaA_evt = _mva_from_event_row(a, "mva_evt")
            mvaB_evt = _mva_from_event_row(b, "mva_evt")

            # Event windows for plots
            B_A_evt = B_A_full.loc[(B_A_full.index >= pd.Timestamp(a["t_start"])) & (B_A_full.index <= pd.Timestamp(a["t_stop"]))]
            B_B_evt = B_B_full.loc[(B_B_full.index >= pd.Timestamp(b["t_start"])) & (B_B_full.index <= pd.Timestamp(b["t_stop"]))]

            thetaA_evt = diagA["Theta_med_deg"]
            thetaB_evt = diagB["Theta_med_deg"]

            annot_lines: List[str] = []
            if np.isfinite(dr_par) and np.isfinite(dr_perp):
                annot_lines.append(f"dr_par(med)={dr_par:.3g}  dr_perp(med)={dr_perp:.3g}")
            if np.isfinite(dr_par_ps) and np.isfinite(dr_perp_ps):
                annot_lines.append(f"dr_par(PS)={dr_par_ps:.3g}  dr_perp(PS)={dr_perp_ps:.3g}")

            # Elongation/aspect proxies (geometry-only; do not interpret as intrinsic SB aspect without ensemble context)
            aspect_med = float(abs(dr_par) / abs(dr_perp)) if (np.isfinite(dr_par) and np.isfinite(dr_perp) and abs(dr_perp) > 0) else float("nan")
            aspect_ps = float(abs(dr_par_ps) / abs(dr_perp_ps)) if (np.isfinite(dr_par_ps) and np.isfinite(dr_perp_ps) and abs(dr_perp_ps) > 0) else float("nan")
            if np.isfinite(aspect_med):
                annot_lines.append(f"aspect(med)={aspect_med:.3g}")
            if np.isfinite(aspect_ps):
                annot_lines.append(f"aspect(PS)={aspect_ps:.3g}")
            annot_text = "\n".join(annot_lines) if annot_lines else None

            # Plotting deferred: we only render top-N candidate pairs by phys_score in a second pass.

            pair_rows.append(
                {
                    "scA": scA,
                    "scB": scB,
                    "event_id_A": a_id,
                    "event_id_B": b_id,
                    "score": float(c["score"]),
                    "tau_star_rel_s": float(tau_star_rel),
                    "tau_star_s": float(tau_star_phys),
                    "tau_star_metric": str(tau_star_metric),
                    "r_star": float(r_star),
                    "width_proxy_s": float(width_star),
                    "curve_paths": {},
                    "phys_score": float(phys_score),
                    "t_ref": t_ref,
                    "dr1": float(dr[0]) if dr is not None else float("nan"),
                    "dr2": float(dr[1]) if dr is not None else float("nan"),
                    "dr3": float(dr[2]) if dr is not None else float("nan"),
                    "dr_mag": float(np.linalg.norm(dr)) if dr is not None else float("nan"),
                    "dr_eff1": float(dr_eff[0]) if dr_eff is not None else float("nan"),
                    "dr_eff2": float(dr_eff[1]) if dr_eff is not None else float("nan"),
                    "dr_eff3": float(dr_eff[2]) if dr_eff is not None else float("nan"),
                    "dr_eff_mag": float(np.linalg.norm(dr_eff)) if dr_eff is not None else float("nan"),
                    "dr_par": float(dr_par),
                    "dr_perp": float(dr_perp),
                    "dr_eff_par": float(dr_eff_par),
                    "dr_eff_perp": float(dr_eff_perp),
                    "dr_par_ps": float(dr_par_ps),
                    "dr_perp_ps": float(dr_perp_ps),
                    "dr_eff_par_ps": float(dr_eff_par_ps),
                    "dr_eff_perp_ps": float(dr_eff_perp_ps),
                    "aspect_ratio_med": float(aspect_med),
                    "aspect_ratio_ps": float(aspect_ps),
                    "dt_lead_s": float(dt_lead),
                    "dt_trail_s": float(dt_trail),
                    "vn_lead": float(vn_lead),
                    "vn_trail": float(vn_trail),
                    "delta_lead": float(delta_lead),
                    "delta_trail": float(delta_trail),
                }
            )

        pairs = pd.DataFrame(pair_rows)
        if pairs.empty:
            continue
        pairs = _stable_sort_df(pairs, by=["event_id_A", "score"])
        pairs_by_pair[key] = pairs

        pairs_path = out_dir / f"pairs_{key}.pkl"
        if overwrite or (not pairs_path.exists()):
            pairs.to_pickle(str(pairs_path))

        # Simple per-pair summary stats (separations + aspect proxies)
        stat_rows: List[Dict[str, Any]] = []
        for col in ("dr_par", "dr_perp", "dr_par_ps", "dr_perp_ps", "aspect_ratio_med", "aspect_ratio_ps"):
            x = pairs[col].to_numpy(dtype=float) if col in pairs.columns else np.array([], dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            stat_rows.append(
                {
                    "pair": key,
                    "metric": col,
                    "n": int(x.size),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "p16": float(np.percentile(x, 16)),
                    "p84": float(np.percentile(x, 84)),
                }
            )
        stats = pd.DataFrame(stat_rows)
        stats_by_pair[key] = stats
        stats_path = out_dir / f"stats_{key}.pkl"
        if overwrite or (not stats_path.exists()):
            stats.to_pickle(str(stats_path))

        # Ensemble geometry/aspect diagnostics figure (pair-level summary)
        if make_plots:
            out_png = (out_dir / "figures" / key) / "geom_stats.png"
            _plot_pair_geom_stats(out_png=out_png, key=key, pairs=pairs)

            # Second pass: plot ONLY top-N candidate pairs by a physics-based goodness score.
            top_n_pairs = int(plot_cfg.get("top_n_pairs", 20))
            if "phys_score" in pairs.columns:
                sel = pairs.sort_values(["phys_score", "r_star"], ascending=[False, False]).head(top_n_pairs)
            else:
                sel = pairs.sort_values(["r_star", "score"], ascending=[False, False]).head(top_n_pairs)

            figs_dir = out_dir / "figures" / key
            _mkdir(figs_dir)

            for _, pr in sel.iterrows():
                a_id = int(pr["event_id_A"])
                b_id = int(pr["event_id_B"])
                tau_star = float(pr.get("tau_star_s", float('nan')))
                tau_star_rel = float(pr.get('tau_star_rel_s', float('nan')))
                tau_star_metric = str(pr.get("tau_star_metric", ""))
                r_star = float(pr.get("r_star", float('nan')))

                a = evA.loc[evA["event_id"] == a_id].iloc[0]
                b = evB.loc[evB["event_id"] == b_id].iloc[0]

                # Event windows
                B_A_evt = B_A_full.loc[(B_A_full.index >= pd.Timestamp(a["t_start"])) & (B_A_full.index <= pd.Timestamp(a["t_stop"]))]
                B_B_evt = B_B_full.loc[(B_B_full.index >= pd.Timestamp(b["t_start"])) & (B_B_full.index <= pd.Timestamp(b["t_stop"]))]

                # Reference time, directions
                t_ref = pd.Timestamp(a["t_center"])
                b_med_hat_A = _nearest_finite_row(diagA, t_ref, ["b_med_1", "b_med_2", "b_med_3"], max_dt=dir_max_dt)
                b_med_hat_B = _nearest_finite_row(diagB, t_ref, ["b_med_1", "b_med_2", "b_med_3"], max_dt=dir_max_dt)
                b_ps_hat_A = _nearest_finite_row(diagA, t_ref, ["b_ps_1", "b_ps_2", "b_ps_3"], max_dt=dir_max_dt) if "b_ps_1" in diagA.columns else None
                b_ps_hat_B = _nearest_finite_row(diagB, t_ref, ["b_ps_1", "b_ps_2", "b_ps_3"], max_dt=dir_max_dt) if "b_ps_1" in diagB.columns else None

                # MVA objects
                mvaA_evt = _mva_from_event_row(a, "mva_evt")
                mvaB_evt = _mva_from_event_row(b, "mva_evt")

                # Separations (already computed/stored in pairs)
                dr_par = float(pr.get("dr_par", float('nan')))
                dr_perp = float(pr.get("dr_perp", float('nan')))
                dr_par_ps = float(pr.get("dr_par_ps", float('nan')))
                dr_perp_ps = float(pr.get("dr_perp_ps", float('nan')))
                aspect_med = float(pr.get("aspect_ratio_med", float('nan')))
                aspect_ps = float(pr.get("aspect_ratio_ps", float('nan')))

                annot_lines = []
                if np.isfinite(dr_par) and np.isfinite(dr_perp):
                    annot_lines.append(f"dr_par(med)={dr_par:.3g}  dr_perp(med)={dr_perp:.3g}")
                if np.isfinite(dr_par_ps) and np.isfinite(dr_perp_ps):
                    annot_lines.append(f"dr_par(PS)={dr_par_ps:.3g}  dr_perp(PS)={dr_perp_ps:.3g}")
                if np.isfinite(aspect_med):
                    annot_lines.append(f"aspect(med)={aspect_med:.3g}")
                if np.isfinite(aspect_ps):
                    annot_lines.append(f"aspect(PS)={aspect_ps:.3g}")
                annot_text = "\n".join(annot_lines) if annot_lines else None

                # Hodogram
                _plot_pair_hodogram_3d(
                    out_png=figs_dir / f"hodogram_A{a_id:04d}_B{b_id:04d}.png",
                    scA=scA,
                    scB=scB,
                    B_cols=B_cols,
                    B_A=B_A_evt,
                    B_B=B_B_evt,
                    mva_A=mvaA_evt,
                    mva_B=mvaB_evt,
                    b_med_hat_A=b_med_hat_A,
                    b_med_hat_B=b_med_hat_B,
                    b_ps_hat_A=b_ps_hat_A,
                    b_ps_hat_B=b_ps_hat_B,
                    title=f"{scA} A{a_id} vs {scB} B{b_id} | {tau_star_metric} tau*={tau_star:.1f}s, R*={r_star:.2f}",
                    annot_text=annot_text,
                )

                # Timeseries (one event, centered) + spans
                if make_timeseries:
                    _plot_pair_timeseries(
                        out_png=figs_dir / f"timeseries_A{a_id:04d}_B{b_id:04d}.png",
                        scA=scA,
                        scB=scB,
                        B_cols=B_cols,
                        B_A=B_A_full,
                        B_B=B_B_full,
                        theta_A=diagA["Theta_med_deg"],
                        theta_B=diagB["Theta_med_deg"],
                        t_start_A=pd.Timestamp(a["t_start"]),
                        t_stop_A=pd.Timestamp(a["t_stop"]),
                        t_start_B=pd.Timestamp(b["t_start"]),
                        t_stop_B=pd.Timestamp(b["t_stop"]),
                        t_center_A=pd.Timestamp(a["t_center"]),
                        t_center_B=pd.Timestamp(b["t_center"]),
                        t_lead_A=pd.Timestamp(a["t_lead"]) if pd.notna(a.get("t_lead", pd.NaT)) else None,
                        t_trail_A=pd.Timestamp(a["t_trail"]) if pd.notna(a.get("t_trail", pd.NaT)) else None,
                        t_lead_B=pd.Timestamp(b["t_lead"]) if pd.notna(b.get("t_lead", pd.NaT)) else None,
                        t_trail_B=pd.Timestamp(b["t_trail"]) if pd.notna(b.get("t_trail", pd.NaT)) else None,
                        tau_star_s=float(tau_star),
                        tau_star_metric=str(tau_star_metric),
                        dr_par=float(dr_par),
                        dr_perp=float(dr_perp),
                        dr_par_ps=float(dr_par_ps),
                        dr_perp_ps=float(dr_perp_ps),
                        aspect_med=float(aspect_med),
                        aspect_ps=float(aspect_ps),
                        title=f"{scA} A{a_id} vs {scB} B{b_id}",
                    )

                # Correlation curves + figure (save CSVs only for selected pairs)
                if make_corr_plots:
                    # Compute curves on event windows.
                    # NOTE: use the same metric list parsed from settings ("metrics"), not an undefined alias.
                    curves = {}
                    t_center_A = pd.Timestamp(a['t_center'])
                    t_center_B = pd.Timestamp(b['t_center'])
                    dt_centers_s = float((t_center_B - t_center_A).total_seconds())
                    _epoch = pd.Timestamp('2000-01-01T00:00:00')
                    BA_rel = _shift_to_epoch(B_A_evt, t_center_A, _epoch)
                    BB_rel = _shift_to_epoch(B_B_evt, t_center_B, _epoch)
                    for met in metrics:
                        met = str(met)
                        if met == 'dbvec':
                            curves[met] = _vector_corr_curve(BA_rel, BB_rel, tau_max_s=tau_max_s, cadence_s=cadence_s)
                        elif met == 'bhat':
                            curves[met] = _vector_dot_curve(BA_rel, BB_rel, tau_max_s=tau_max_s, cadence_s=cadence_s)
                        elif met == 'theta_med':
                            sA = diagA.loc[(diagA.index >= pd.Timestamp(a['t_start'])) & (diagA.index <= pd.Timestamp(a['t_stop'])), 'Theta_med_deg']
                            sB = diagB.loc[(diagB.index >= pd.Timestamp(b['t_start'])) & (diagB.index <= pd.Timestamp(b['t_stop'])), 'Theta_med_deg']
                            sA = _shift_to_epoch(sA, t_center_A, _epoch)
                            sB = _shift_to_epoch(sB, t_center_B, _epoch)
                            curves[met] = _scalar_corr_curve(sA, sB, tau_max_s=tau_max_s, cadence_s=cadence_s)
                        elif met == 'theta_ps' and ('Theta_PS_deg' in diagA.columns) and ('Theta_PS_deg' in diagB.columns):
                            sA = diagA.loc[(diagA.index >= pd.Timestamp(a['t_start'])) & (diagA.index <= pd.Timestamp(a['t_stop'])), 'Theta_PS_deg']
                            sB = diagB.loc[(diagB.index >= pd.Timestamp(b['t_start'])) & (diagB.index <= pd.Timestamp(b['t_stop'])), 'Theta_PS_deg']
                            sA = _shift_to_epoch(sA, t_center_A, _epoch)
                            sB = _shift_to_epoch(sB, t_center_B, _epoch)
                            curves[met] = _scalar_corr_curve(sA, sB, tau_max_s=tau_max_s, cadence_s=cadence_s)
                        else:
                            continue

                    # Add lag_di based on event-mean U_ref and event-median d_i (from A)
                    di_ref = float('nan')
                    if 'di_m' in diagA.columns:
                        di_vals = diagA.loc[(diagA.index >= pd.Timestamp(a['t_start'])) & (diagA.index <= pd.Timestamp(a['t_stop'])), 'di_m'].to_numpy(dtype=float)
                        di_vals = di_vals[np.isfinite(di_vals)]
                        if di_vals.size:
                            di_ref = float(np.median(di_vals))
                    uref = V_A.loc[(V_A.index >= pd.Timestamp(a['t_start'])) & (V_A.index <= pd.Timestamp(a['t_stop']))].mean().to_numpy(dtype=float)
                    uref_mps = _speed_to_mps(uref)
                    if np.isfinite(di_ref) and di_ref > 0 and np.isfinite(uref_mps):
                        for met, dfc in curves.items():
                            if dfc is None or dfc.empty or ('lag_s' not in dfc.columns):
                                continue
                            dfc = dfc.copy()
                            dfc['lag_phys_s'] = dfc['lag_s'].to_numpy(dtype=float) + dt_centers_s
                            dfc['lag_di'] = (uref_mps * dfc['lag_s'].to_numpy(dtype=float)) / di_ref
                            curves[met] = dfc

                    # Save cross-corr CSVs
                    xdir = out_dir / 'xcorr_curves' / key
                    _mkdir(xdir)
                    for met, dfc in curves.items():
                        if dfc is None or dfc.empty:
                            continue
                        dfc.to_csv(str(xdir / f"A{a_id:04d}_B{b_id:04d}_{met}.csv"), index=False)

                    # Autocorr CSVs (selected pairs; include lag_di when possible)
                    autoA = _vector_corr_curve(B_A_evt, None, tau_max_s=autocorr_tau_max_s, cadence_s=cadence_s)
                    autoB = _vector_corr_curve(B_B_evt, None, tau_max_s=autocorr_tau_max_s, cadence_s=cadence_s)
                    if np.isfinite(di_ref) and di_ref > 0 and np.isfinite(uref_mps):
                        autoA = autoA.copy(); autoA['lag_di'] = (uref_mps * autoA['lag_s'].to_numpy(dtype=float)) / di_ref
                        autoB = autoB.copy(); autoB['lag_di'] = (uref_mps * autoB['lag_s'].to_numpy(dtype=float)) / di_ref

                    _plot_pair_corr_curves(
                        out_png=figs_dir / f"corr_A{a_id:04d}_B{b_id:04d}.png",
                        scA=scA,
                        scB=scB,
                        autoA=autoA,
                        autoB=autoB,
                        curves=curves,
                        tau_star_s=float(tau_star_rel),
                        tau_star_phys_s=float(tau_star),
                        tau_star_metric=str(tau_star_metric),
                        dr_par=float(dr_par),
                        dr_perp=float(dr_perp),
                        dr_par_ps=float(dr_par_ps),
                        dr_perp_ps=float(dr_perp_ps),
                        title=f"{scA} A{a_id} vs {scB} B{b_id}",
                    )

            out_png = (out_dir / "figures" / key) / "geom_stats.png"
            _plot_pair_geom_stats(out_png=out_png, key=key, pairs=pairs)

    # --------------------
    # Stage: bootstrap correlation statistics
    # --------------------
    if corrstats_enabled:
        cs_dir = out_dir / "corr_stats"
        _mkdir(cs_dir)

        comps = list(analysis_cfg.get("B_cols", []))
        if not comps:
            # fall back to a conservative default
            comps = ["Br", "Bt", "Bn"]

        # (1) Events: autocorrelation ensembles per spacecraft
        for sc in sc_list:
            ev = events_by_sc.get(sc, pd.DataFrame())
            if ev is None or ev.empty:
                continue
            Bfull = _ensure_dtindex(by_sc[sc]["Mag"])[list(comps)]
            curves_evt: List[np.ndarray] = []
            for _, r in ev.iterrows():
                t0 = pd.Timestamp(r["t_start"])
                t1 = pd.Timestamp(r["t_stop"])
                arr = _interp_df_to_unit_grid(Bfull, t0=t0, t1=t1, cols=comps, n_grid=corrstats_n_grid)
                if arr is None:
                    continue
                curves_evt.append(_acf_curve_unit(arr))
            if not curves_evt:
                continue
            X = np.stack(curves_evt, axis=0)
            mean, lo, hi = _bootstrap_mean_ci(X, n_boot=corrstats_n_boot, seed=abs(hash(("acf", sc))) % (2**31))
            u = np.linspace(0.0, 1.0, int(corrstats_n_grid), dtype=float)

            # save CSV
            out_csv = cs_dir / f"events_autocorr_{sc}.csv"
            tab = {"u": u}
            for j, c in enumerate(comps):
                tab[f"mean_{c}"] = mean[:, j]
                tab[f"lo_{c}"] = lo[:, j]
                tab[f"hi_{c}"] = hi[:, j]
            pd.DataFrame(tab).to_csv(str(out_csv), index=False)

            # save PNG
            _plot_corr_ensemble(
                out_png=cs_dir / f"events_autocorr_{sc}.png",
                u=u,
                curves=X,
                mean=mean,
                lo=lo,
                hi=hi,
                comps=comps,
                title=f"{sc}: event autocorrelation (unit-normalized time; N={X.shape[0]})",
            )

        # (1) Events: cross-correlation ensembles per spacecraft pair (best match per A-event)
        for key, pairs in pairs_by_pair.items():
            if pairs is None or pairs.empty:
                continue
            try:
                scA, scB = str(key).split("__", 1)
            except Exception:
                continue
            evA = events_by_sc.get(scA, pd.DataFrame())
            evB = events_by_sc.get(scB, pd.DataFrame())
            if evA.empty or evB.empty:
                continue

            # best match per A event
            p = pairs.copy()
            sort_col = "phys_score" if "phys_score" in p.columns else "score"
            p = p.sort_values(["event_id_A", sort_col], ascending=[True, False], kind="mergesort")
            p = p.groupby("event_id_A", as_index=False, group_keys=False).head(1).reset_index(drop=True)

            mapA = {int(r["event_id"]): r for _, r in evA.iterrows()}
            mapB = {int(r["event_id"]): r for _, r in evB.iterrows()}

            B_A_full = _ensure_dtindex(by_sc[scA]["Mag"])[list(comps)]
            B_B_full = _ensure_dtindex(by_sc[scB]["Mag"])[list(comps)]

            curves_pair: List[np.ndarray] = []
            for _, r in p.iterrows():
                a_id = int(r["event_id_A"])
                b_id = int(r["event_id_B"])
                tau = float(r.get("tau_star_s", float("nan")))
                if not np.isfinite(tau):
                    continue
                if a_id not in mapA or b_id not in mapB:
                    continue

                a = mapA[a_id]
                b = mapB[b_id]
                tA0 = pd.Timestamp(a["t_start"])
                tA1 = pd.Timestamp(a["t_stop"])
                tB0 = pd.Timestamp(b["t_start"])
                tB1 = pd.Timestamp(b["t_stop"])

                BA = B_A_full.loc[(B_A_full.index >= tA0) & (B_A_full.index <= tA1)]
                BB = B_B_full.loc[(B_B_full.index >= tB0) & (B_B_full.index <= tB1)]
                if BA.empty or BB.empty:
                    continue
                BBs = BB.copy()
                BBs.index = BBs.index - pd.to_timedelta(float(tau), unit="s")

                t0 = max(BA.index[0], BBs.index[0])
                t1 = min(BA.index[-1], BBs.index[-1])
                if t1 <= t0:
                    continue

                arrA = _interp_df_to_unit_grid(BA, t0=t0, t1=t1, cols=comps, n_grid=corrstats_n_grid)
                arrB = _interp_df_to_unit_grid(BBs, t0=t0, t1=t1, cols=comps, n_grid=corrstats_n_grid)
                if arrA is None or arrB is None:
                    continue
                curves_pair.append(_ccf_curve_unit(arrA, arrB))

            if not curves_pair:
                continue
            X = np.stack(curves_pair, axis=0)
            mean, lo, hi = _bootstrap_mean_ci(X, n_boot=corrstats_n_boot, seed=abs(hash(("ccf", key))) % (2**31))
            u = np.linspace(0.0, 1.0, int(corrstats_n_grid), dtype=float)

            out_csv = cs_dir / f"events_crosscorr_{key}.csv"
            tab = {"u": u}
            for j, c in enumerate(comps):
                tab[f"mean_{c}"] = mean[:, j]
                tab[f"lo_{c}"] = lo[:, j]
                tab[f"hi_{c}"] = hi[:, j]
            pd.DataFrame(tab).to_csv(str(out_csv), index=False)

            _plot_corr_ensemble(
                out_png=cs_dir / f"events_crosscorr_{key}.png",
                u=u,
                curves=X,
                mean=mean,
                lo=lo,
                hi=hi,
                comps=comps,
                title=f"{key}: event cross-correlation (tau* aligned; unit-normalized; N={X.shape[0]})",
            )

        # (2) Full interval: autocorrelation ensembles per spacecraft (bootstrap pairs)
        for sc in sc_list:
            Bfull = _ensure_dtindex(by_sc[sc]["Mag"])[list(comps)]
            u_ref: Optional[np.ndarray] = None
            mean_mat: List[np.ndarray] = []
            lo_mat: List[np.ndarray] = []
            hi_mat: List[np.ndarray] = []
            for c in comps:
                u, m, l, h = _bootstrap_corr_curve_full(
                    Bfull[c],
                    None,
                    tau_max_s=corrstats_full_tau_max_s,
                    cadence_s=corrstats_full_cadence_s,
                    n_boot=corrstats_n_boot,
                    n_pairs=corrstats_full_n_pairs,
                    seed=abs(hash(("full_acf", sc, c))) % (2**31),
                )
                if u.size == 0:
                    continue
                if u_ref is None:
                    u_ref = u
                mean_mat.append(m)
                lo_mat.append(l)
                hi_mat.append(h)
            if u_ref is None or (not mean_mat):
                continue
            mean = np.column_stack(mean_mat)
            lo = np.column_stack(lo_mat)
            hi = np.column_stack(hi_mat)

            out_csv = cs_dir / f"full_autocorr_{sc}.csv"
            tab = {"u": u_ref}
            for j, c in enumerate(comps[: mean.shape[1]]):
                tab[f"mean_{c}"] = mean[:, j]
                tab[f"lo_{c}"] = lo[:, j]
                tab[f"hi_{c}"] = hi[:, j]
            pd.DataFrame(tab).to_csv(str(out_csv), index=False)

            _plot_corr_ensemble(
                out_png=cs_dir / f"full_autocorr_{sc}.png",
                u=u_ref,
                curves=None,
                mean=mean,
                lo=lo,
                hi=hi,
                comps=comps[: mean.shape[1]],
                title=f"{sc}: full-interval autocorrelation (bootstrap pairs; Npairs={corrstats_full_n_pairs})",
            )

        # (2) Full interval: cross-correlation ensembles per spacecraft pair (bootstrap pairs)
        for key in pairs_by_pair.keys():
            try:
                scA, scB = str(key).split("__", 1)
            except Exception:
                continue
            B_A_full = _ensure_dtindex(by_sc[scA]["Mag"])[list(comps)]
            B_B_full = _ensure_dtindex(by_sc[scB]["Mag"])[list(comps)]
            u_ref: Optional[np.ndarray] = None
            mean_mat = []
            lo_mat = []
            hi_mat = []
            for c in comps:
                u, m, l, h = _bootstrap_corr_curve_full(
                    B_A_full[c],
                    B_B_full[c],
                    tau_max_s=corrstats_full_tau_max_s,
                    cadence_s=corrstats_full_cadence_s,
                    n_boot=corrstats_n_boot,
                    n_pairs=corrstats_full_n_pairs,
                    seed=abs(hash(("full_ccf", key, c))) % (2**31),
                )
                if u.size == 0:
                    continue
                if u_ref is None:
                    u_ref = u
                mean_mat.append(m)
                lo_mat.append(l)
                hi_mat.append(h)
            if u_ref is None or (not mean_mat):
                continue
            mean = np.column_stack(mean_mat)
            lo = np.column_stack(lo_mat)
            hi = np.column_stack(hi_mat)

            out_csv = cs_dir / f"full_crosscorr_{key}.csv"
            tab = {"u": u_ref}
            for j, c in enumerate(comps[: mean.shape[1]]):
                tab[f"mean_{c}"] = mean[:, j]
                tab[f"lo_{c}"] = lo[:, j]
                tab[f"hi_{c}"] = hi[:, j]
            pd.DataFrame(tab).to_csv(str(out_csv), index=False)

            _plot_corr_ensemble(
                out_png=cs_dir / f"full_crosscorr_{key}.png",
                u=u_ref,
                curves=None,
                mean=mean,
                lo=lo,
                hi=hi,
                comps=comps[: mean.shape[1]],
                title=f"{key}: full-interval cross-correlation (bootstrap pairs; Npairs={corrstats_full_n_pairs})",
            )

    return {
        "loaded": loaded,
        "diagnostics_by_sc": diagnostics_by_sc,
        "events_by_sc": events_by_sc,
        "candidates_by_pair": candidates_by_pair,
        "pairs_by_pair": pairs_by_pair,
        "stats_by_pair": stats_by_pair,
        "out_dir": str(out_dir),
    }
