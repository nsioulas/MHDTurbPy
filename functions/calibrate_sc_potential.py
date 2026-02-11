import numpy as np
import pandas as pd
import scipy
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(REPO_ROOT / 'functions'))
import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func

from scipy.optimize import least_squares
from scipy.signal   import butter, filtfilt
from joblib         import Parallel, delayed



def process_sc_pot(df, voltage_columns=None):
    """
    Combine or negate the voltage columns to yield a single 'v_sc' potential.
    """
    if voltage_columns is None:
        voltage_columns = [col for col in df.columns if 'V' in col]
    v_sc = -np.nanmean(df[voltage_columns], axis=1)
    return pd.DataFrame(v_sc, index=df.index, columns=['V'])


def process_window_2param(start, end, t_low, V_low, logn):
    """
    Fit log(n) = log(A) - B * V over [start,end].
    Returns (start, end, A, B, q), with q=0 if fit fails.
    """
    mask   = (t_low >= start) & (t_low <= end)
    v_win  = V_low[mask]
    ln_win = logn[mask]
    if len(v_win) < 2:
        return (start, end, np.nan, np.nan, 0.0)

    # 1) initial linear LS: ln n ≈ β0 + β1·V  ⇒  β0=ln A,  β1≈−B
    X      = np.vstack([np.ones_like(v_win), v_win]).T
    β, *_  = np.linalg.lstsq(X, ln_win, rcond=None)
    A0     = np.exp(β[0])
    B0     = -β[1]
    # ensure a non-negative initial slope
    if B0 < 0:
        B0 = abs(B0)

    # 2) robust refinement, bounding B≥0
    def resid(p):
        return ln_win - (np.log(p[0]) - p[1]*v_win)

    bounds = ([1e-12, 0.0], [np.inf, np.inf])
    res    = least_squares(resid, [A0, B0],
                           bounds=bounds,
                           loss='huber',
                           x_scale='jac')
    A1, B1 = res.x
    q      = 1.0 / (1.0 + np.sqrt(res.cost))

    return (start, end, A1, B1, q)

def calibrate_density(
    df_V, df_n,
    mode          = 'local',                # 'local' or 'global'
    window_str    = '30s',
    overlap_ratio = 0.5,
    n_jobs        = -1,
    lower_pct     = 0.0,
    upper_pct     = 100
):
    """
    If mode=='global': one single fit over the whole series, no windows.
    If mode=='local' (default): rolling-window fits + Hanning-parameter blend.
    """
    # 1) synchronize
    df_V_low, df_n_sync = func.synchronize_dfs(df_V, df_n.rolling('15s').median(), 0)

    # 2) low-pass the density only, then log-transform
    dt = (df_n_sync.index[1] - df_n_sync.index[0]).total_seconds()
    fs = 1.0 / dt
    df_n_sync['np_filt'] =df_n_sync['np'].values
    df_n_sync['logn'] = np.log(df_n_sync['np_filt'])

    # 3) pull arrays
    t_low   = df_n_sync.index.values.astype('int64')
    V_low   = df_V_low['V'].values
    logn    = df_n_sync['logn'].values
    t_high  = df_V.index.values.astype('int64')
    V_high  = df_V['V'].values
    t_index = pd.to_datetime(t_high)

    # --- global mode: one fit on full interval ---
    if mode == 'global':
        s, e, A, B, q = process_window_2param(
            t_low[0], t_low[-1], t_low, V_low, logn
        )
        ts = pd.Series(A * np.exp(-B * V_high), index=t_index)
        # clip outliers + fill
        lb, ub = np.percentile(ts.dropna(), [lower_pct, upper_pct])
        ts[(ts < lb) | (ts > ub)] = np.nan
        return ts.ffill().bfill(), pd.DataFrame(
            [(s, e, A, B, q)],
            columns=['start','end','A','B','q']
        )

    # --- local mode: rolling windows + param-level blend ---
    # window boundaries in ns
    w_ns    = pd.to_timedelta(window_str).value
    step_ns = int(w_ns * (1 - overlap_ratio))
    starts  = np.arange(t_high[0], t_high[-1], step_ns)
    ends    = starts + w_ns

    # parallel per-window fits
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_window_2param)(s, e, t_low, V_low, logn)
        for s, e in zip(starts, ends)
    )
    wins = pd.DataFrame(results, columns=['start','end','A','B','q'])
    wins = wins.sort_values('start')

    # build weighted sums of A and B
    num_A = np.zeros_like(V_high, dtype=float)
    num_B = np.zeros_like(V_high, dtype=float)
    denom = np.zeros_like(V_high, dtype=float)

    for _, row in wins.iterrows():
        m    = (t_high >= row.start) & (t_high <= row.end)
        idxs = np.nonzero(m)[0]
        if len(idxs) < 2:
            continue
        hwin      = np.hanning(len(idxs))
        weights   = row.q * hwin               # incorporate fit quality
        num_A[idxs] += weights * row.A
        num_B[idxs] += weights * row.B
        denom[idxs] += weights

    # avoid divide-by-zero, then reconstruct n(t)
    valid = denom > 1e-6
    A_t   = np.full_like(V_high, np.nan, dtype=float)
    B_t   = np.full_like(V_high, np.nan, dtype=float)
    A_t[valid] = num_A[valid] / denom[valid]
    B_t[valid] = num_B[valid] / denom[valid]
    ts = pd.Series(A_t * np.exp(-B_t * V_high), index=t_index)

    # clip outliers + fill gaps
    lb, ub = np.percentile(ts.dropna(), [lower_pct, upper_pct])
    ts[(ts < lb) | (ts > ub)] = np.nan
    return ts.ffill().bfill(), wins,df_n_sync
