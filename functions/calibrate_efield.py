"""
calibrate_efield.py
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import traceback

from joblib import Parallel, delayed, effective_n_jobs
from scipy import signal

import os
import sys
from pathlib import Path

_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent

sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'pyspedas'))
sys.path.insert(0, str(_FUNCTIONS_DIR))
import general_functions as func


# =============================================================================
# 1) Hygiene + cadence helpers
# =============================================================================

def _ensure_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex.")

    out = df.copy()
    out.index = pd.to_datetime(out.index)

    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _infer_dt_seconds(df: pd.DataFrame) -> float:
    """
    Infer cadence in seconds. Uses func.find_cadence if available,
    otherwise uses median timestamp difference.
    """
    df = _ensure_df(df, "df")

    dt = np.nan
    try:
        dt = float(func.find_cadence(df))
    except Exception:
        dt = np.nan

    if np.isfinite(dt) and dt > 0:
        return float(dt)

    t_ns = df.index.view("int64")
    if t_ns.size < 3:
        raise ValueError("Cannot infer cadence from <3 samples.")

    dts = np.diff(t_ns).astype(np.float64) * 1e-9
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        raise ValueError("Invalid cadence inferred.")
    return float(np.median(dts))


def _overlap_range(dfs: List[pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0 = max(df.index.min() for df in dfs)
    t1 = min(df.index.max() for df in dfs)
    return pd.to_datetime(t0), pd.to_datetime(t1)


# =============================================================================
# 2) Robust binning + anti-alias filtering helpers
# =============================================================================

def _bin_aggregate(
    df: pd.DataFrame,
    dt_s: float,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    how: str = "mean",
) -> pd.DataFrame:
    """
    Downsample by time-binning WITHOUT interpolation.

    NOTE:
    Use this AFTER proper low-pass filtering to prevent aliasing.
    """
    df = _ensure_df(df, "df")

    if how not in ("mean", "median"):
        raise ValueError("how must be 'mean' or 'median'")

    if t0 >= t1:
        return df.iloc[0:0].copy()

    dt_ns = int(np.round(float(dt_s) * 1e9))
    if dt_ns <= 0:
        raise ValueError("Invalid dt_s")

    x = df.loc[t0:t1].copy()
    if len(x) == 0:
        return x

    t0_ns = int(pd.to_datetime(t0).value)
    t_ns = x.index.view("int64")
    bin_id = ((t_ns - t0_ns) // dt_ns).astype(np.int64)

    x["__bin__"] = bin_id
    g = x.groupby("__bin__", sort=True)

    if how == "mean":
        y = g.mean(numeric_only=True)
    else:
        y = g.median(numeric_only=True)

    if "__bin__" in y.columns:
        y = y.drop(columns=["__bin__"], errors="ignore")

    bins = y.index.to_numpy(dtype=np.int64)
    idx_ns = t0_ns + bins * dt_ns
    y.index = pd.to_datetime(idx_ns)

    return y.sort_index()


def _interp_to_grid_bounded(
    t_src_ns: np.ndarray,
    y_src: np.ndarray,
    t_tgt_ns: np.ndarray,
    tol_s: float,
    max_bracket_gap_s: float,
) -> np.ndarray:
    """
    SAFE bounded interpolation y(t_src)->y(t_tgt) using int64 arithmetic.

    Two guards (both needed):
      (i) Nearest source sample must be within tol_s
      (ii) The BRACKET gap (tr - tl) must be <= max_bracket_gap_s
           (this enforces "do not cross gaps")

    Returns NaN when either condition fails.
    """
    t_src_ns = np.asarray(t_src_ns, np.int64)
    t_tgt_ns = np.asarray(t_tgt_ns, np.int64)
    y_src = np.asarray(y_src, float)

    out = np.full(t_tgt_ns.size, np.nan, dtype=float)
    if t_src_ns.size < 2:
        return out

    good = np.isfinite(y_src)
    t_src_ns = t_src_ns[good]
    y_src = y_src[good]

    if t_src_ns.size < 2:
        return out

    order = np.argsort(t_src_ns)
    t_src_ns = t_src_ns[order]
    y_src = y_src[order]

    tol_ns = int(np.round(float(tol_s) * 1e9))
    max_gap_ns = int(np.round(float(max_bracket_gap_s) * 1e9))

    # bracket indices
    jr = np.searchsorted(t_src_ns, t_tgt_ns, side="left")
    jl = jr - 1

    ok = (jl >= 0) & (jr < t_src_ns.size)
    if not np.any(ok):
        return out

    tl = t_src_ns[jl[ok]]
    tr = t_src_ns[jr[ok]]
    yl = y_src[jl[ok]]
    yr = y_src[jr[ok]]

    # (ii) bracket-gap gating: do not interpolate across large holes
    gap = tr - tl
    ok_gap = gap <= max_gap_ns
    if not np.any(ok_gap):
        return out

    # keep only those that pass bracket-gap gating
    idx_ok = np.where(ok)[0]
    idx_ok = idx_ok[ok_gap]

    tl = tl[ok_gap]
    tr = tr[ok_gap]
    yl = yl[ok_gap]
    yr = yr[ok_gap]
    tt = t_tgt_ns[idx_ok]

    # (i) nearest-distance gating: target must be near one of the endpoints
    dl = tt - tl
    dr = tr - tt
    dmin = np.minimum(np.abs(dl), np.abs(dr))
    ok_near = dmin <= tol_ns
    if not np.any(ok_near):
        return out

    idx_ok2 = idx_ok[ok_near]

    tl = tl[ok_near]
    tr = tr[ok_near]
    yl = yl[ok_near]
    yr = yr[ok_near]
    tt = tt[ok_near]

    denom = (tr - tl).astype(np.float64)
    denom = np.maximum(denom, 1.0)

    alpha = (tt - tl).astype(np.float64) / denom
    out[idx_ok2] = yl + alpha * (yr - yl)

    return out


def _lowpass_filtfilt_segments(
    df: pd.DataFrame,
    fs: float,
    fc_hz: float,
    order: int = 4,
    gap_factor: float = 3.0,
) -> pd.DataFrame:
    """
    Zero-phase Butterworth low-pass filter per contiguous finite run.

    FIX:
    - filter ONLY on slices with NO NaNs inside (contiguous finite runs)
    - do NOT filter across large time gaps (gap_factor * dt_native)
    """
    df = _ensure_df(df, "df")
    if len(df) < 8:
        return df

    if not np.isfinite(fc_hz) or fc_hz <= 0:
        return df

    nyq = 0.5 * float(fs)
    if fc_hz >= 0.98 * nyq:
        return df

    Wn = float(fc_hz / nyq)
    b, a = signal.butter(int(order), Wn, btype="low", analog=False)

    dt_native = 1.0 / float(fs)
    max_gap_s = float(gap_factor * dt_native)
    max_gap_ns = int(np.round(max_gap_s * 1e9))

    t_ns = df.index.view("int64")

    out = df.copy()
    for col in df.columns:
        y = df[col].to_numpy(float)
        ok = np.isfinite(y)
        if np.sum(ok) < 8:
            continue

        idx = np.where(ok)[0]
        if idx.size < 8:
            continue

        # contiguous finite runs + no large time gaps
        breaks = [0]
        for k in range(1, idx.size):
            if (idx[k] != idx[k - 1] + 1) or ((t_ns[idx[k]] - t_ns[idx[k - 1]]) > max_gap_ns):
                breaks.append(k)
        breaks.append(idx.size)

        y_f = y.copy()
        for a0, a1 in zip(breaks[:-1], breaks[1:]):
            i0 = int(idx[a0])
            i1 = int(idx[a1 - 1]) + 1  # contiguous slice end
            seg = y[i0:i1]
            if seg.size < 8:
                continue
            # seg is guaranteed finite here (contiguous finite run)
            try:
                y_f[i0:i1] = signal.filtfilt(b, a, seg, method="pad")
            except Exception:
                continue

        out[col] = y_f

    return out


def _compute_Eref_xy(V_kms: np.ndarray, B_nT: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    V (km/s), B (nT) -> Eref (V/m)
    """
    V = V_kms * 1e3
    B = B_nT * 1e-9
    E = -np.cross(V, B)
    return E[:, 0], E[:, 1]


def make_fit_df(
    bdf: pd.DataFrame,
    vdf: pd.DataFrame,
    edf: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct calibration dataset at dt_ref = max(dt_V, dt_B):

    - Interpolate V onto B grid with strict "do not cross gaps" logic
    - Build E_inst = -(V x B) on B grid
    - Low-pass E_inst and dv segment-wise (no gap contamination)
    - Bin-mean downsample both to dt_ref
    - Join -> fit_df = [dvx,dvy,Ex_ref,Ey_ref] on dt_ref grid
    """
    bdf = _ensure_df(bdf, "bdf")
    vdf = _ensure_df(vdf, "vdf")
    edf = _ensure_df(edf, "edf")

    if not all(c in bdf.columns for c in ("Bx", "By", "Bz")):
        if all(c in bdf.columns for c in ("Br", "Bt", "Bn")):
            bdf = bdf.rename(columns={"Br": "Bx", "Bt": "By", "Bn": "Bz"})
        else:
            raise KeyError("bdf must contain (Bx,By,Bz) or (Br,Bt,Bn)")

    if not all(c in vdf.columns for c in ("Vx", "Vy", "Vz")):
        if all(c in vdf.columns for c in ("Vr", "Vt", "Vn")):
            vdf = vdf.rename(columns={"Vr": "Vx", "Vt": "Vy", "Vn": "Vz"})
        else:
            raise KeyError("vdf must contain (Vx,Vy,Vz) or (Vr,Vt,Vn)")

    if not all(c in edf.columns for c in ("dvx", "dvy")):
        raise KeyError("edf must contain (dvx,dvy)")

    dtB = _infer_dt_seconds(bdf[["Bx", "By", "Bz"]])
    dtV = _infer_dt_seconds(vdf[["Vx", "Vy", "Vz"]])
    dtDV = _infer_dt_seconds(edf[["dvx", "dvy"]])

    dt_ref = float(max(dtB, dtV))
    t0, t1 = _overlap_range([bdf, vdf, edf])

    meta = {
        "dt_B_seconds": float(dtB),
        "dt_V_seconds": float(dtV),
        "dt_dV_seconds": float(dtDV),
        "dt_ref_seconds": float(dt_ref),
        "t0": t0,
        "t1": t1,
    }

    if t0 >= t1:
        return pd.DataFrame(), meta

    nyq_ref = 0.5 / dt_ref
    fc_hz = 0.40 * nyq_ref
    filt_order = 4
    gap_factor = 3.0

    meta["fc_hz"] = float(fc_hz)
    meta["filt_order"] = int(filt_order)
    meta["gap_factor"] = float(gap_factor)

    B_seg = bdf.loc[t0:t1, ["Bx", "By", "Bz"]].dropna()
    V_seg = vdf.loc[t0:t1, ["Vx", "Vy", "Vz"]].dropna()

    if len(B_seg) < 8 or len(V_seg) < 2:
        return pd.DataFrame(), meta

    tB_ns = B_seg.index.view("int64")
    tV_ns = V_seg.index.view("int64")

    # FIX: enforce "do not cross gaps" with bracket-gap gating
    tol_s = 0.90 * float(dtV)
    max_bracket_gap_s = 3.0 * float(dtV)
    meta["V_interp_tol_s"] = float(tol_s)
    meta["V_interp_max_bracket_gap_s"] = float(max_bracket_gap_s)

    Vx = _interp_to_grid_bounded(
        tV_ns, V_seg["Vx"].to_numpy(float), tB_ns,
        tol_s=tol_s, max_bracket_gap_s=max_bracket_gap_s
    )
    Vy = _interp_to_grid_bounded(
        tV_ns, V_seg["Vy"].to_numpy(float), tB_ns,
        tol_s=tol_s, max_bracket_gap_s=max_bracket_gap_s
    )
    Vz = _interp_to_grid_bounded(
        tV_ns, V_seg["Vz"].to_numpy(float), tB_ns,
        tol_s=tol_s, max_bracket_gap_s=max_bracket_gap_s
    )

    V_on_B = np.vstack([Vx, Vy, Vz]).T
    B_on_B = B_seg.to_numpy(float)

    good = np.all(np.isfinite(V_on_B), axis=1) & np.all(np.isfinite(B_on_B), axis=1)

    meta["n_B_seg"] = int(len(B_seg))
    meta["n_V_seg"] = int(len(V_seg))
    meta["frac_V_on_B_valid"] = float(np.mean(np.all(np.isfinite(V_on_B), axis=1))) if len(V_on_B) else 0.0
    meta["frac_E_inst_valid"] = float(np.mean(good)) if len(good) else 0.0

    Ex_inst = np.full(tB_ns.size, np.nan, dtype=float)
    Ey_inst = np.full(tB_ns.size, np.nan, dtype=float)

    if np.any(good):
        Ex_tmp, Ey_tmp = _compute_Eref_xy(V_on_B[good], B_on_B[good])
        Ex_inst[good] = Ex_tmp
        Ey_inst[good] = Ey_tmp

    E_inst = pd.DataFrame(index=B_seg.index, data={"Ex_ref": Ex_inst, "Ey_ref": Ey_inst})

    DV_seg = edf.loc[t0:t1, ["dvx", "dvy"]]

    fs_dv = 1.0 / float(dtDV)
    fs_B = 1.0 / float(dtB)

    DV_f = _lowpass_filtfilt_segments(
        DV_seg, fs=fs_dv, fc_hz=fc_hz, order=filt_order, gap_factor=gap_factor
    )
    E_f = _lowpass_filtfilt_segments(
        E_inst, fs=fs_B, fc_hz=fc_hz, order=filt_order, gap_factor=gap_factor
    )

    dV_ref = _bin_aggregate(DV_f, dt_ref, t0, t1, how="mean")
    E_ref = _bin_aggregate(E_f, dt_ref, t0, t1, how="mean")

    fit_df = dV_ref.join(E_ref, how="inner").dropna()
    meta["n_fit"] = int(len(fit_df))

    return fit_df, meta


# =============================================================================
# 3) Robust window fit (FULL 2x2 model)
# =============================================================================

def _robust_fit_full_matrix(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    n_iter: int = 2,
    huber_k: float = 1.5,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Fit:
      dvx = g11*Ex + g12*Ey + c
      dvy = g21*Ex + g22*Ey + d

    Using IRLS with Huber weights on the 2D residual magnitude.
    """
    Ex = np.asarray(Ex, float)
    Ey = np.asarray(Ey, float)
    dvx = np.asarray(dvx, float)
    dvy = np.asarray(dvy, float)

    N = Ex.size
    if N < 16:
        raise ValueError("Too few points for robust fit.")

    A = np.zeros((2 * N, 6), dtype=float)
    A[:N, 0] = Ex
    A[:N, 1] = Ey
    A[:N, 4] = 1.0
    A[N:, 2] = Ex
    A[N:, 3] = Ey
    A[N:, 5] = 1.0

    y = np.hstack([dvx, dvy]).astype(float)

    w = np.ones(N, dtype=float)
    p = np.full(6, np.nan, dtype=float)

    for _ in range(max(1, int(n_iter))):
        W = np.repeat(w, 2)
        Aw = A * W[:, None]
        M = A.T @ Aw
        b = A.T @ (W * y)

        cond = float(np.linalg.cond(M))
        if not np.isfinite(cond) or cond > 1e18:
            raise ValueError("Ill-conditioned window.")

        p = np.linalg.solve(M, b)

        yhat = A @ p
        r1 = y[:N] - yhat[:N]
        r2 = y[N:] - yhat[N:]
        rmag = np.sqrt(r1 * r1 + r2 * r2)

        med = float(np.median(rmag))
        mad = float(np.median(np.abs(rmag - med)))
        scale = 1.4826 * mad if mad > 0 else (med if med > 0 else 1e-6)
        scale = float(max(scale, 1e-6))

        delta = float(huber_k * scale)
        w = np.minimum(1.0, delta / (rmag + eps))

    g11, g12, g21, g22, c, d = [float(x) for x in p]

    dvx_hat = g11 * Ex + g12 * Ey + c
    dvy_hat = g21 * Ex + g22 * Ey + d
    r1 = dvx - dvx_hat
    r2 = dvy - dvy_hat

    rss = float(np.sum(r1 * r1) + np.sum(r2 * r2))
    dof = max(2 * N - 6, 1)
    sigma2 = rss / dof

    W = np.repeat(w, 2)
    Aw = A * W[:, None]
    M = A.T @ Aw
    Minv = np.linalg.solve(M, np.eye(6, dtype=float))
    cov = sigma2 * Minv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    Cx = float(np.corrcoef(dvx, dvx_hat)[0, 1]) if np.nanstd(dvx) > 0 and np.nanstd(dvx_hat) > 0 else np.nan
    Cy = float(np.corrcoef(dvy, dvy_hat)[0, 1]) if np.nanstd(dvy) > 0 and np.nanstd(dvy_hat) > 0 else np.nan
    nx = float(np.sqrt(np.nanmean(r1 * r1)) / (np.nanstd(dvx) + 1e-12))
    ny = float(np.sqrt(np.nanmean(r2 * r2)) / (np.nanstd(dvy) + 1e-12))

    det = float(g11 * g22 - g12 * g21)
    inv_ok = np.isfinite(det) and abs(det) > 1e-12

    return {
        "g11": g11, "g12": g12, "g21": g21, "g22": g22,
        "c": c, "d": d,
        "sigma_g11": float(se[0]), "sigma_g12": float(se[1]),
        "sigma_g21": float(se[2]), "sigma_g22": float(se[3]),
        "sigma_c": float(se[4]), "sigma_d": float(se[5]),
        "cond": float(np.linalg.cond(M)),
        "detG": det,
        "invertible": bool(inv_ok),
        "C_dVX": Cx, "C_dVY": Cy,
        "C_min": float(np.nanmin([Cx, Cy])),
        "NRMSE_sum": float(nx + ny),
        "NRMSE_dVX": float(nx),
        "NRMSE_dVY": float(ny),
        "std_Ex": float(np.nanstd(Ex)),
        "std_Ey": float(np.nanstd(Ey)),
    }


# =============================================================================
# 4) Windows + coefficient fitting
# =============================================================================

def _make_windows(t_ns: np.ndarray, win_s: float, step_s: float) -> Tuple[np.ndarray, np.ndarray]:
    if t_ns.size < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    win_ns = int(max(float(win_s), 1e-6) * 1e9)
    step_ns = int(max(float(step_s), 1e-6) * 1e9)

    span_ns = int(t_ns[-1] - t_ns[0])
    if span_ns <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if span_ns < win_ns:
        win_ns = span_ns
        step_ns = span_ns

    stop = t_ns[-1] - win_ns
    if stop < t_ns[0]:
        starts = np.array([t_ns[0]], dtype=np.int64)
    else:
        starts = np.arange(t_ns[0], stop + 1, step_ns, dtype=np.int64)

    ends = starts + win_ns
    return starts, ends


def _fit_one_window(
    t_ns: np.ndarray,
    Ex_ref: np.ndarray,
    Ey_ref: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    s_ns: int,
    e_ns: int,
    robust: bool,
    max_internal_gap_s: float,
    min_pred_std: float,
) -> Dict[str, Any]:
    i0 = int(np.searchsorted(t_ns, s_ns, side="left"))
    i1 = int(np.searchsorted(t_ns, e_ns, side="right"))

    if (i1 - i0) < 16:
        raise ValueError("Too few points in window.")

    sl = slice(i0, i1)

    if (i1 - i0) >= 2:
        gaps_s = np.diff(t_ns[sl]).astype(np.float64) * 1e-9
        if np.nanmax(gaps_s) > float(max_internal_gap_s):
            raise ValueError("Internal gap too large in window.")

    if float(np.nanstd(Ex_ref[sl])) < float(min_pred_std) or float(np.nanstd(Ey_ref[sl])) < float(min_pred_std):
        raise ValueError("Degenerate predictor (Ex/Ey nearly constant).")

    center_time = pd.to_datetime(int(0.5 * (t_ns[sl][0] + t_ns[sl][-1])))

    out = _robust_fit_full_matrix(
        Ex_ref[sl], Ey_ref[sl], dvx[sl], dvy[sl],
        n_iter=2 if robust else 1,
        huber_k=1.5,
    )

    out["interval_start"] = pd.to_datetime(s_ns)
    out["interval_end"] = pd.to_datetime(e_ns)
    out["center_time"] = center_time
    return out


def _qc_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Conservative QC rules.

    NOTE:
    - use abs(C_min) so sign flips are not wrongly discarded
    """
    cond = df["cond"].to_numpy(float)
    cmin = df["C_min"].to_numpy(float)
    nrmse = df["NRMSE_sum"].to_numpy(float)
    detG = df["detG"].to_numpy(float)
    inv_ok = df["invertible"].to_numpy(bool)

    bad = (
        ~np.isfinite(cond) | (cond > 1e12) |
        ~np.isfinite(cmin) | (np.abs(cmin) < 0.5) |
        ~np.isfinite(nrmse) | (nrmse > 2.0) |
        ~np.isfinite(detG) | (~inv_ok)
    )
    return bad


def _choose_window_length_simple(
    t_ns: np.ndarray,
    Ex_ref: np.ndarray,
    Ey_ref: np.ndarray,
    dvx: np.ndarray,
    dvy: np.ndarray,
    dt: float,
    span_s: float,
    robust: bool,
    n_jobs: int,
    verbose: bool,
) -> Tuple[float, float]:
    """
    Simple auto-tuning:
      try a few window sizes and pick the one with best median NRMSE after QC,
      while penalizing discard fraction.
    """
    candidates = np.array([200.0, 300.0, 600.0], dtype=float)
    candidates = candidates[candidates < max(0.8 * span_s, 121.0)]
    if candidates.size == 0:
        win_s = float(min(600.0, max(60.0, 0.2 * span_s)))
        return win_s, float(max(win_s / 2.0, dt))

    bestJ = np.inf
    best = (float(candidates[0]), float(max(candidates[0] / 2.0, dt)))

    for win_s in candidates:
        step_s = float(max(win_s / 2.0, dt))
        starts, ends = _make_windows(t_ns, win_s, step_s)
        nW = len(starts)
        if nW < 3:
            continue

        idx = np.linspace(0, nW - 1, num=min(nW, 12), dtype=int)

        max_internal_gap_s = float(3.0 * dt)
        min_pred_std = 1e-6

        def _job(j):
            return _fit_one_window(
                t_ns, Ex_ref, Ey_ref, dvx, dvy,
                int(starts[j]), int(ends[j]),
                robust=robust,
                max_internal_gap_s=max_internal_gap_s,
                min_pred_std=min_pred_std,
            )

        rows = Parallel(n_jobs=n_jobs)(delayed(_job)(int(j)) for j in idx)
        df = pd.DataFrame(rows).set_index("center_time").sort_index()
        bad = _qc_mask(df)
        keep = df[~bad]

        disc = float(np.mean(bad)) if len(df) else 1.0
        if len(keep) < 3:
            continue

        J = float(np.nanmedian(keep["NRMSE_sum"].to_numpy(float)) + 2.0 * disc)

        if verbose:
            print(f"[auto] win={win_s:.0f}s step={step_s:.0f}s disc={disc:.2f} J={J:.3g}")

        if J < bestJ:
            bestJ = J
            best = (float(win_s), float(step_s))

    if verbose:
        print(f"[auto] selected win={best[0]:.0f}s step={best[1]:.0f}s")

    return best


def process_data(
    bdf: pd.DataFrame,
    vdf: pd.DataFrame,
    edf: pd.DataFrame,
    auto: bool = True,
    robust: bool = True,
    win_s: Optional[float] = None,
    step_s: Optional[float] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """
    Fit coefficient windows.

    Returns:
      coeffs_df, discard_fraction, meta
    """
    try:
        fit_df, meta0 = make_fit_df(bdf=bdf, vdf=vdf, edf=edf)
        if fit_df.empty or len(fit_df) < 80:
            raise RuntimeError("Not enough valid calibration points after filtering/binning/joining.")

        dt = _infer_dt_seconds(fit_df)
        span_s = (fit_df.index[-1] - fit_df.index[0]).total_seconds()
        if span_s <= 0:
            raise RuntimeError("Invalid fit span.")

        dvx = fit_df["dvx"].to_numpy(float)
        dvy = fit_df["dvy"].to_numpy(float)
        t_ns = fit_df.index.view("int64")

        if not all(c in fit_df.columns for c in ("Ex_ref", "Ey_ref")):
            raise RuntimeError("fit_df is missing Ex_ref/Ey_ref predictor columns.")

        Ex_ref = fit_df["Ex_ref"].to_numpy(float)
        Ey_ref = fit_df["Ey_ref"].to_numpy(float)

        if auto or (win_s is None) or (step_s is None):
            win_s_use, step_s_use = _choose_window_length_simple(
                t_ns, Ex_ref, Ey_ref, dvx, dvy,
                dt=dt, span_s=span_s,
                robust=robust,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        else:
            win_s_use = float(win_s)
            step_s_use = float(step_s)

        starts, ends = _make_windows(t_ns, win_s_use, step_s_use)
        if len(starts) == 0:
            raise RuntimeError("No windows created.")

        nj = max(1, int(effective_n_jobs(n_jobs)))
        idx = np.arange(len(starts), dtype=int)
        chunk = max(32, int(np.ceil(len(idx) / (4 * nj))))
        chunks = [idx[i:i + chunk] for i in range(0, len(idx), chunk)]

        max_internal_gap_s = float(max(3.0 * dt, 1.5 * meta0.get("dt_ref_seconds", dt)))
        min_pred_std = 1e-6

        def _job_block(block_idx: np.ndarray) -> List[Dict[str, Any]]:
            out = []
            for j in block_idx:
                try:
                    out.append(_fit_one_window(
                        t_ns, Ex_ref, Ey_ref, dvx, dvy,
                        int(starts[j]), int(ends[j]),
                        robust=robust,
                        max_internal_gap_s=max_internal_gap_s,
                        min_pred_std=min_pred_std,
                    ))
                except Exception:
                    continue
            return out

        blocks = Parallel(n_jobs=n_jobs)(delayed(_job_block)(ch) for ch in chunks)
        rows = [r for blk in blocks for r in blk]
        if not rows:
            raise RuntimeError("No window fits succeeded.")

        coeffs = pd.DataFrame(rows).set_index("center_time").sort_index()

        bad = _qc_mask(coeffs)
        coeffs["discarded"] = bad.astype(bool)
        disc_frac = float(np.mean(bad))

        meta = dict(meta0)
        meta.update({
            "fit_dt_seconds": float(dt),
            "fit_span_seconds": float(span_s),
            "win_seconds": float(win_s_use),
            "step_seconds": float(step_s_use),
            "discard_fraction": float(disc_frac),
            "n_windows": int(len(coeffs)),
            "robust": bool(robust),
            "max_internal_gap_s": float(max_internal_gap_s),
            "min_pred_std": float(min_pred_std),
        })

        if verbose:
            print(f"[fit] win={win_s_use:.0f}s step={step_s_use:.0f}s disc={disc_frac:.3f} nW={len(coeffs)}")

        return coeffs, disc_frac, meta

    except Exception:
        traceback.print_exc()
        return pd.DataFrame(), float("nan"), {}


# =============================================================================
# 5) Coefficient interpolation + high-rate application
# =============================================================================

def _interp_coeffs_bounded(
    t_ns: np.ndarray,
    tc_ns: np.ndarray,
    yc: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    """
    Linear interpolation of coefficients from (tc_ns, yc) onto t_ns,
    but ONLY if the bracket gap is <= max_gap_s.
    """
    t_ns = np.asarray(t_ns, np.int64)
    tc_ns = np.asarray(tc_ns, np.int64)
    yc = np.asarray(yc, float)

    out = np.full(t_ns.size, np.nan, dtype=float)
    if tc_ns.size < 2:
        return out

    max_gap_ns = int(np.round(float(max_gap_s) * 1e9))

    j = np.searchsorted(tc_ns, t_ns, side="right")
    jl = j - 1
    jr = j

    ok = (jl >= 0) & (jr < tc_ns.size)
    if not np.any(ok):
        return out

    tl = tc_ns[jl[ok]]
    tr = tc_ns[jr[ok]]
    gap = tr - tl
    ok2 = gap <= max_gap_ns

    idx_ok = np.where(ok)[0]
    idx_ok2 = idx_ok[ok2]
    if idx_ok2.size == 0:
        return out

    tl = tc_ns[jl[idx_ok2]]
    tr = tc_ns[jr[idx_ok2]]
    yl = yc[jl[idx_ok2]]
    yr = yc[jr[idx_ok2]]

    alpha = (t_ns[idx_ok2] - tl) / np.maximum((tr - tl).astype(np.float64), 1.0)
    out[idx_ok2] = yl + alpha * (yr - yl)

    return out


def calibrate_data(
    edf: pd.DataFrame,
    coeffs_df: pd.DataFrame,
    max_gap_coeff_s: float = 3_000.0,
) -> pd.DataFrame:
    """
    Apply fitted coefficients to the high-rate dv grid.

    We interpolate coefficients in time to avoid step artifacts, but we do NOT
    interpolate across huge gaps (max_gap_coeff_s).
    """
    edf = _ensure_df(edf, "edf")

    if coeffs_df is None or len(coeffs_df) == 0:
        return pd.DataFrame(index=edf.index, data={"Ex": np.nan, "Ey": np.nan})

    coeffs = coeffs_df.copy()
    if "discarded" in coeffs.columns:
        coeffs = coeffs[~coeffs["discarded"]].copy()

    if len(coeffs) < 2:
        return pd.DataFrame(index=edf.index, data={"Ex": np.nan, "Ey": np.nan})

    tc_ns = pd.to_datetime(coeffs.index).values.astype("datetime64[ns]").astype("int64")
    t_ns = edf.index.values.astype("datetime64[ns]").astype("int64")

    g11 = _interp_coeffs_bounded(t_ns, tc_ns, coeffs["g11"].to_numpy(float), max_gap_coeff_s)
    g12 = _interp_coeffs_bounded(t_ns, tc_ns, coeffs["g12"].to_numpy(float), max_gap_coeff_s)
    g21 = _interp_coeffs_bounded(t_ns, tc_ns, coeffs["g21"].to_numpy(float), max_gap_coeff_s)
    g22 = _interp_coeffs_bounded(t_ns, tc_ns, coeffs["g22"].to_numpy(float), max_gap_coeff_s)
    c = _interp_coeffs_bounded(t_ns, tc_ns, coeffs["c"].to_numpy(float), max_gap_coeff_s)
    d = _interp_coeffs_bounded(t_ns, tc_ns, coeffs["d"].to_numpy(float), max_gap_coeff_s)

    dvx = edf["dvx"].to_numpy(float)
    dvy = edf["dvy"].to_numpy(float)

    x1 = dvx - c
    x2 = dvy - d

    det = g11 * g22 - g12 * g21
    good = np.isfinite(det) & (np.abs(det) > 1e-12)

    Ex = np.full_like(dvx, np.nan, dtype=float)
    Ey = np.full_like(dvy, np.nan, dtype=float)

    Ex[good] = ( g22[good] * x1[good] - g12[good] * x2[good]) / det[good]
    Ey[good] = (-g21[good] * x1[good] + g11[good] * x2[good]) / det[good]

    # keep your original convention (often desired output is mV/m)
    Ex *= 1e3
    Ey *= 1e3

    return pd.DataFrame({"Ex": Ex, "Ey": Ey}, index=edf.index)


from typing import Tuple, Dict, Any, Optional


def _enforce_EdotB0_weighted(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    wz: float = 0.05,
    eps: float = 1e-30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stable perpendicular projection: enforce E·B=0 without dividing by Bz.

    We start from E0=(Ex,Ey,0) and find the closest E_perp (in a weighted sense)
    such that E_perp·B=0.

    wz << 1 makes the correction preferentially go into Ez, keeping Ex/Ey close
    to the calibrated values, while avoiding blow-ups when Bz->0.
    """
    Ex = np.asarray(Ex, float)
    Ey = np.asarray(Ey, float)
    Bx = np.asarray(Bx, float)
    By = np.asarray(By, float)
    Bz = np.asarray(Bz, float)

    wx = 1.0
    wy = 1.0
    wz = float(max(wz, 1e-12))

    S = Ex * Bx + Ey * By  # E0·B (Ez0=0)

    D = (Bx * Bx) / wx + (By * By) / wy + (Bz * Bz) / wz
    D = np.maximum(D, eps)

    Exp = Ex - (S / D) * (Bx / wx)
    Eyp = Ey - (S / D) * (By / wy)
    Ezp = 0.0 - (S / D) * (Bz / wz)

    return Exp, Eyp, Ezp


def calibrate_electric_field(
    bdf: pd.DataFrame,
    vdf: pd.DataFrame,
    edf: pd.DataFrame,
    auto: bool = True,
    robust: bool = True,
    win_s: Optional[float] = None,
    step_s: Optional[float] = None,
    n_jobs: int = -1,
    verbose: bool = True,
    wz_Ez: float = 0.05,
    max_gap_coeff_s: float = 3_000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Full pipeline:
      1) fit coefficients using anti-aliased E_ref = -(V x B) on dt_ref
      2) apply to high-rate dv -> (Ex_sc, Ey_sc)
      3) infer Ez_sc by enforcing E·B=0 (stable weighted projection)
      4) compute Esw = Esc + Vsw x B (Vsw = window-mean velocity)

    Returns:
      E_out_df : high-rate dataframe with columns
                 [Ex_sc, Ey_sc, Ez_sc, Ex_sw, Ey_sw, Ez_sw]
      coeffs_df: fitted window coefficients (+ window mean Vsw columns)
      meta     : diagnostics
    """
    # ------------------------------------------------------------
    # 1) Fit window coefficients (unchanged)
    # ------------------------------------------------------------
    coeffs_df, disc_frac, meta = process_data(
        bdf=bdf,
        vdf=vdf,
        edf=edf,
        auto=auto,
        robust=robust,
        win_s=win_s,
        step_s=step_s,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 2) Apply coefficients -> Esc_xy on high-rate dv grid (unchanged)
    # ------------------------------------------------------------
    Esc_xy = calibrate_data(edf=edf, coeffs_df=coeffs_df, max_gap_coeff_s=max_gap_coeff_s)
    # Esc_xy has columns ["Ex","Ey"] in mV/m by your convention
    Esc_xy = _ensure_df(Esc_xy, "Esc_xy")

    # ------------------------------------------------------------
    # 3) Interpolate B onto dv timestamps (bounded, do not cross gaps)
    # ------------------------------------------------------------
    bdf = _ensure_df(bdf, "bdf")
    if not all(c in bdf.columns for c in ("Bx", "By", "Bz")):
        if all(c in bdf.columns for c in ("Br", "Bt", "Bn")):
            bdf = bdf.rename(columns={"Br": "Bx", "Bt": "By", "Bn": "Bz"})
        else:
            raise KeyError("bdf must contain (Bx,By,Bz) or (Br,Bt,Bn)")

    B_seg = bdf.loc[edf.index.min():edf.index.max(), ["Bx", "By", "Bz"]].dropna()
    if len(B_seg) < 2:
        E_out = pd.DataFrame(
            index=edf.index,
            data={
                "Ex_sc": np.nan, "Ey_sc": np.nan, "Ez_sc": np.nan,
                "Ex_sw": np.nan, "Ey_sw": np.nan, "Ez_sw": np.nan,
            },
        )
        meta = dict(meta)
        meta["discard_fraction"] = float(disc_frac)
        meta["Ez_method"] = "E·B=0 weighted projection (failed: insufficient B)"
        return E_out, coeffs_df, meta

    dtB = _infer_dt_seconds(B_seg)
    tol_s_B = 0.90 * float(dtB)
    max_gap_B = 3.0 * float(dtB)

    tB_ns = B_seg.index.view("int64")
    tedf_ns = edf.index.view("int64")

    Bx_i = _interp_to_grid_bounded(
        t_src_ns=tB_ns,
        y_src=B_seg["Bx"].to_numpy(float),
        t_tgt_ns=tedf_ns,
        tol_s=tol_s_B,
        max_bracket_gap_s=max_gap_B,
    )
    By_i = _interp_to_grid_bounded(
        t_src_ns=tB_ns,
        y_src=B_seg["By"].to_numpy(float),
        t_tgt_ns=tedf_ns,
        tol_s=tol_s_B,
        max_bracket_gap_s=max_gap_B,
    )
    Bz_i = _interp_to_grid_bounded(
        t_src_ns=tB_ns,
        y_src=B_seg["Bz"].to_numpy(float),
        t_tgt_ns=tedf_ns,
        tol_s=tol_s_B,
        max_bracket_gap_s=max_gap_B,
    )

    # ------------------------------------------------------------
    # 4) Estimate Ez_sc robustly by enforcing E·B=0
    # ------------------------------------------------------------
    Ex_sc = Esc_xy["Ex"].to_numpy(float)
    Ey_sc = Esc_xy["Ey"].to_numpy(float)

    goodB = np.isfinite(Bx_i) & np.isfinite(By_i) & np.isfinite(Bz_i)
    goodE = np.isfinite(Ex_sc) & np.isfinite(Ey_sc)
    good = goodB & goodE

    Ex_sc_p = np.full_like(Ex_sc, np.nan, dtype=float)
    Ey_sc_p = np.full_like(Ey_sc, np.nan, dtype=float)
    Ez_sc_p = np.full_like(Ey_sc, np.nan, dtype=float)

    if np.any(good):
        Ex_sc_p[good], Ey_sc_p[good], Ez_sc_p[good] = _enforce_EdotB0_weighted(
            Ex=Ex_sc[good],
            Ey=Ey_sc[good],
            Bx=Bx_i[good],
            By=By_i[good],
            Bz=Bz_i[good],
            wz=wz_Ez,
        )

    # ------------------------------------------------------------
    # 5) Compute Vsw per fit window and interpolate it to dv timestamps
    # ------------------------------------------------------------
    vdf = _ensure_df(vdf, "vdf")
    if not all(c in vdf.columns for c in ("Vx", "Vy", "Vz")):
        if all(c in vdf.columns for c in ("Vr", "Vt", "Vn")):
            vdf = vdf.rename(columns={"Vr": "Vx", "Vt": "Vy", "Vn": "Vz"})
        else:
            raise KeyError("vdf must contain (Vx,Vy,Vz) or (Vr,Vt,Vn)")

    coeffs_df = coeffs_df.copy()
    if len(coeffs_df) > 0:
        # compute window-mean Vsw for each fitted window row
        Vsw_x = np.full(len(coeffs_df), np.nan, dtype=float)
        Vsw_y = np.full(len(coeffs_df), np.nan, dtype=float)
        Vsw_z = np.full(len(coeffs_df), np.nan, dtype=float)

        for k, row in enumerate(coeffs_df.itertuples()):
            t0 = getattr(row, "interval_start", None)
            t1 = getattr(row, "interval_end", None)
            if t0 is None or t1 is None:
                continue
            seg = vdf.loc[pd.to_datetime(t0):pd.to_datetime(t1), ["Vx", "Vy", "Vz"]]
            if len(seg) == 0:
                continue
            m = seg.mean(numeric_only=True)
            Vsw_x[k] = float(m["Vx"])
            Vsw_y[k] = float(m["Vy"])
            Vsw_z[k] = float(m["Vz"])

        coeffs_df["Vsw_x"] = Vsw_x
        coeffs_df["Vsw_y"] = Vsw_y
        coeffs_df["Vsw_z"] = Vsw_z

    # use non-discarded windows only
    coeffs_use = coeffs_df.copy()
    if "discarded" in coeffs_use.columns:
        coeffs_use = coeffs_use[~coeffs_use["discarded"]].copy()

    Ex_sw = np.full_like(Ex_sc_p, np.nan, dtype=float)
    Ey_sw = np.full_like(Ey_sc_p, np.nan, dtype=float)
    Ez_sw = np.full_like(Ez_sc_p, np.nan, dtype=float)

    if len(coeffs_use) >= 2 and all(c in coeffs_use.columns for c in ("Vsw_x", "Vsw_y", "Vsw_z")):
        tc_ns = pd.to_datetime(coeffs_use.index).values.astype("datetime64[ns]").astype("int64")

        Vswx_i = _interp_coeffs_bounded(tedf_ns, tc_ns, coeffs_use["Vsw_x"].to_numpy(float), max_gap_coeff_s)
        Vswy_i = _interp_coeffs_bounded(tedf_ns, tc_ns, coeffs_use["Vsw_y"].to_numpy(float), max_gap_coeff_s)
        Vswz_i = _interp_coeffs_bounded(tedf_ns, tc_ns, coeffs_use["Vsw_z"].to_numpy(float), max_gap_coeff_s)

        goodV = np.isfinite(Vswx_i) & np.isfinite(Vswy_i) & np.isfinite(Vswz_i) & goodB
        if np.any(goodV):
            V_ms = np.vstack([Vswx_i[goodV], Vswy_i[goodV], Vswz_i[goodV]]).T * 1e3
            B_T = np.vstack([Bx_i[goodV], By_i[goodV], Bz_i[goodV]]).T * 1e-9

            motional_mVm = 1e3 * np.cross(V_ms, B_T)  # mV/m

            Ex_sw[goodV] = Ex_sc_p[goodV] + motional_mVm[:, 0]
            Ey_sw[goodV] = Ey_sc_p[goodV] + motional_mVm[:, 1]
            Ez_sw[goodV] = Ez_sc_p[goodV] + motional_mVm[:, 2]

    # ------------------------------------------------------------
    # 6) Package output
    # ------------------------------------------------------------
    E_out = pd.DataFrame(
        index=edf.index,
        data={
            "Ex_sc": Ex_sc_p,
            "Ey_sc": Ey_sc_p,
            "Ez_sc": Ez_sc_p,
            "Ex_sw": Ex_sw,
            "Ey_sw": Ey_sw,
            "Ez_sw": Ez_sw,
        },
    )

    meta = dict(meta)
    meta["discard_fraction"] = float(disc_frac)
    meta["Ez_method"] = "E·B=0 weighted projection"
    meta["Ez_wz"] = float(wz_Ez)
    meta["B_interp_tol_s"] = float(tol_s_B)
    meta["B_interp_max_gap_s"] = float(max_gap_B)
    meta["Esw_method"] = "Esc + Vsw(window-mean) x B"

    return E_out, coeffs_df, meta


__all__ = [
    "make_fit_df",
    "process_data",
    "calibrate_data",
    "calibrate_electric_field",
]

