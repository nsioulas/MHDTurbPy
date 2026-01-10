import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from datetime import datetime
from pathlib import Path
import pickle
from scipy import stats
import numba
from numba import jit, njit, prange, objmode
from scipy.optimize import curve_fit
import joblib
from joblib import Parallel, delayed
import statistics
from statistics import mode
#import orderedstructs
import sys
import pytplot

import warnings
warnings.filterwarnings('ignore')
import polars as pl


# Import urbPy
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))


from plasma_params import*
import signal_processing  #import synchronize_dfs



def _finalize_alignment(df1, df2, *, interp_method='time'):
    """
    Internal helper: clamp two DataFrames to their overlapping time range,
    time-interpolate small gaps, and trim leading/trailing NaNs jointly.

    Assumes both have a DatetimeIndex.
    """
    if not isinstance(df1.index, pd.DatetimeIndex):
        raise TypeError("df1 must have a DatetimeIndex.")
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise TypeError("df2 must have a DatetimeIndex.")

    # Ensure monotonic indices for time interpolation
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    # 1) Overlapping time window
    overlap_start = max(df1.index.min(), df2.index.min())
    overlap_end   = min(df1.index.max(), df2.index.max())

    if overlap_start >= overlap_end:
        # No overlap -> return empties with the right columns/dtypes
        return df1.iloc[0:0].copy(), df2.iloc[0:0].copy()

    df1 = df1.loc[overlap_start:overlap_end].copy()
    df2 = df2.loc[overlap_start:overlap_end].copy()

    # 2) Interpolate small internal gaps in time
    df1 = df1.interpolate(method=interp_method)
    df2 = df2.interpolate(method=interp_method)

    # 3) Trim leading/trailing NaNs jointly
    fvi1 = df1.first_valid_index()
    lvi1 = df1.last_valid_index()
    fvi2 = df2.first_valid_index()
    lvi2 = df2.last_valid_index()

    if fvi1 is None or fvi2 is None:
        # One (or both) are all-NaN in this window
        return df1.iloc[0:0].copy(), df2.iloc[0:0].copy()

    new_start = max(fvi1, fvi2)
    new_end   = min(lvi1, lvi2)

    if new_start > new_end:
        # No joint interval where both have finite values
        return df1.iloc[0:0].copy(), df2.iloc[0:0].copy()

    df1 = df1.loc[new_start:new_end]
    df2 = df2.loc[new_start:new_end]

    return df1, df2


def synchronize_dfs(df_higher_freq, df_lower_freq, upsample):
    """
    Synchronize two time-indexed DataFrames with different cadences.

    Parameters
    ----------
    df_higher_freq : pd.DataFrame
        Higher-cadence time series.
    df_lower_freq : pd.DataFrame
        Lower-cadence time series.
    upsample : bool
        If True  -> upsample the lower-frequency DF to the higher cadence
                    using signal_processing.upsample_dataframe.
        If False -> downsample the higher-frequency DF to the lower cadence
                    using signal_processing.downsample_and_filter (with
                    anti-aliasing low-pass filtering).

    Returns
    -------
    df_high_sync, df_low_sync : pd.DataFrame
        Two DataFrames that:
          1. Live strictly on the common overlapping time range,
          2. Have small internal gaps interpolated in time,
          3. Have leading/trailing NaNs removed based on the joint valid window.
    """
    if not isinstance(df_higher_freq.index, pd.DatetimeIndex):
        raise TypeError("df_higher_freq must have a DatetimeIndex.")
    if not isinstance(df_lower_freq.index, pd.DatetimeIndex):
        raise TypeError("df_lower_freq must have a DatetimeIndex.")

    # Work on sorted copies; do not modify the caller's frames in place
    df_higher_freq = df_higher_freq.sort_index()
    df_lower_freq  = df_lower_freq.sort_index()

    if upsample:
        # ===== UPSAMPLE: low cadence -> high cadence =====
        # Clean low-cadence series before FIR + filtfilt to avoid NaNs in the filter
        low_clean = df_lower_freq.interpolate(method='time').dropna(how='all')
        if low_clean.empty:
            # Nothing usable in low series
            return df_higher_freq.iloc[0:0].copy(), df_lower_freq.iloc[0:0].copy()

        high_sorted = df_higher_freq  # already sorted

        # Anti-imaging FIR low-pass + interpolation onto the high-cadence grid
        aligned_low = signal_processing.upsample_dataframe(
            low_clean,
            high_sorted,
        )

        # Final overlap clamp + interpolation + joint trim
        high_sync, low_sync = _finalize_alignment(high_sorted, aligned_low)

    else:
        # ===== DOWNSAMPLE: high cadence -> low cadence =====
        # Clean both before low-pass filtering; filtfilt cannot handle NaNs
        high_clean = df_higher_freq.interpolate(method='time').dropna(how='all')
        low_clean  = df_lower_freq.interpolate(method='time').dropna(how='all')

        if high_clean.empty or low_clean.empty:
            # Nothing usable in one of the series
            return df_higher_freq.iloc[0:0].copy(), df_lower_freq.iloc[0:0].copy()

        # Anti-aliasing low-pass filter at ~Nyquist(low_fs) and resample onto low_clean.index
        aligned_high = signal_processing.downsample_and_filter(
            high_clean,
            low_clean,
        )

        # Final overlap clamp + interpolation + joint trim
        high_sync, low_sync = _finalize_alignment(aligned_high, low_clean)

    return high_sync, low_sync





def read_pickle(path):
    class FixUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # intercept either numpy._core.multiarray._frombuffer
            # or numpy.core.numeric._frombuffer
            if name == '_frombuffer' and module.startswith('numpy'):
                def _fixed(buf, dtype=None, shape=None, order=None):
                    arr = np.frombuffer(buf, dtype=dtype)
                    if shape is not None:
                        return arr.reshape(shape, order=order)
                    return arr
                return _fixed
            return super().find_class(module, name)
    with open(path, 'rb') as f:
        return FixUnpickler(f).load()



def create_gap_mask(master_index: pd.DatetimeIndex,
                    gap_dfs: list[pd.DataFrame],
                    buffer: str = '10s',
                    min_gap: str = '1s'
                   ) -> pd.Series:
    """
    Build a 0/1 mask over master_index:
      • 0 if timestamp lies in any gap [start-buffer, end+buffer],
      • 1 otherwise.
    Then remove any gap-run shorter than min_gap by setting it back to 1.

    Parameters
    ----------
    master_index : pd.DatetimeIndex
        Full timeline (e.g. coh['sig_c'].index)
    gap_dfs : list of pd.DataFrame
        Each DF must have 'Start'/'End' columns (case-insensitive).
    buffer : str
        Pandas offset string to extend each interval on both sides.
    min_gap : str
        Minimum gap duration; shorter zero-runs are reset to 1.
    """

    # Convert to Timedelta
    buff       = pd.to_timedelta(buffer)
    min_gap_td = pd.to_timedelta(min_gap)

    # Initialize all valid (1)
    mask_arr = np.ones(len(master_index), dtype=int)

    # 1) Mark gaps with buffer
    for gdf in gap_dfs:
        if gdf.empty:
            continue

        cols_lc = {c.lower(): c for c in gdf.columns}
        if 'start' not in cols_lc or 'end' not in cols_lc:
            continue  # skip if no interval columns

        starts = pd.to_datetime(gdf[cols_lc['start']], errors='coerce') - buff
        ends   = pd.to_datetime(gdf[cols_lc['end']],   errors='coerce') + buff

        for s, e in zip(starts, ends):
            if pd.isna(s) or pd.isna(e):
                continue
            if s > e:
                s, e = e, s
            i0 = master_index.searchsorted(s, side='left')
            i1 = master_index.searchsorted(e, side='right')
            mask_arr[i0:i1] = 0

    mask = pd.Series(mask_arr, index=master_index, name='data_valid')

    # 2) Heal tiny gaps: any zero-run < min_gap → set back to 1
    df = pd.DataFrame({'mask': mask})
    # group number increments on mask-change
    df['grp'] = (df['mask'] != df['mask'].shift()).cumsum()

    for _, sub in df.groupby('grp'):
        if sub['mask'].iat[0] == 0:
            duration = sub.index[-1] - sub.index[0]
            if duration < min_gap_td:
                mask.loc[sub.index] = 1

    return mask



import numpy as np
from joblib import Parallel, delayed

# def compute_quantile_edges_function(
#     x, y,
#     Nx_bins,
#     Ny_bins,
#     log_x           = True,
#     poly_order      = None,
#     auto_poly       = True,               # NEW: automatic degree selector
#     max_poly_order  = 5,              # NEW: search cap for auto selector
#     criterion       = "bic",               # NEW: {"bic"} (others reserved)
#     n_jobs          = -1,
#     return_counts   = False,
#     return_avg_y    = False,
#     low_pct         = 0.05,
#     high_pct        = 99.8,
# ):
#     """
#     Compute quantile edges of y as a function of x.

#     Extensions over the previous version
#     ------------------------------------
#       • y-values outside the [low_pct, high_pct] range are discarded;
#       • polynomial fits are weighted by the # of points in each x-bin;
#       • *optional* automatic selection of the best polynomial degree
#         using a conservative weighted-BIC score (set auto_poly=True).
#     """
#     # ---------- input checks & percentile clipping -------------------
#     x = np.asarray(x, float).ravel()
#     y = np.asarray(y, float).ravel()
#     if x.shape != y.shape:
#         raise ValueError("x and y must have the same length")
#     if not (0 <= low_pct < high_pct <= 100):
#         raise ValueError("0 ≤ low_pct < high_pct ≤ 100")

#     lo, hi = np.percentile(y, [low_pct, high_pct])
#     keep   = (y >= lo) & (y <= hi)
#     x, y   = x[keep], y[keep]

#     # ---------- x-bin edges ------------------------------------------
#     x_edges = (np.logspace(np.log10(x.min()), np.log10(x.max()), Nx_bins + 1)
#                if log_x else
#                np.linspace(x.min(), x.max(), Nx_bins + 1))

#     order                = np.argsort(x)
#     x_sorted, y_sorted   = x[order], y[order]
#     bin_idx              = np.searchsorted(x_sorted, x_edges)
#     q_levels             = np.linspace(0.0, 1.0, Ny_bins + 1)

#     # ---------- quantiles per x-bin ----------------------------------
#     def _bin_q(i):
#         s, e = bin_idx[i], bin_idx[i + 1]
#         return (np.quantile(y_sorted[s:e], q_levels)
#                 if e - s >= 2 else np.full_like(q_levels, np.nan))

#     y_edges_per_bin = np.vstack(
#         Parallel(n_jobs=n_jobs)(delayed(_bin_q)(i) for i in range(Nx_bins))
#     )
#     x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
#     bin_sizes = np.diff(bin_idx)                            # <-- counts for weighting

#     # ---------- counts / averages if requested -----------------------
#     counts = av_y = None
#     if return_counts or return_avg_y:
#         def _stats(i):
#             s, e = bin_idx[i], bin_idx[i + 1]
#             if e - s < 1:
#                 return np.zeros(Ny_bins, int), np.full(Ny_bins, np.nan)
#             yy = y_sorted[s:e]
#             c, _ = np.histogram(yy, bins=y_edges_per_bin[i])
#             av  = np.array([
#                 yy[(yy >= y_edges_per_bin[i,j]) & (yy < y_edges_per_bin[i,j+1])].mean()
#                 if c[j] else np.nan for j in range(Ny_bins)
#             ])
#             return c, av
#         tmp = Parallel(n_jobs=n_jobs)(delayed(_stats)(i) for i in range(Nx_bins))
#         if return_counts: counts = np.vstack([t[0] for t in tmp])
#         if return_avg_y : av_y   = np.vstack([t[1] for t in tmp])

#     # ---------- (auto-)select polynomial degree ----------------------
#     # If auto_poly is requested, ignore any user-supplied poly_order
#     selected_poly_order = None
#     selection_table     = None
#     if auto_poly:
#         if max_poly_order < 0:
#             raise ValueError("max_poly_order must be non-negative")
#         degrees   = range(max_poly_order + 1)
#         bic_score = []

#         for p in degrees:
#             wrss = 0.0
#             npts = 0
#             for j in range(Ny_bins + 1):
#                 ok = (~np.isnan(y_edges_per_bin[:, j])) & (bin_sizes > 0)
#                 if ok.sum() < p + 1:       # not enough points to fit p-th order
#                     continue
#                 coeffs = np.polyfit(
#                     x_centres[ok],
#                     y_edges_per_bin[ok, j],
#                     p,
#                     w=np.sqrt(bin_sizes[ok]),
#                 )
#                 resid  = (np.polyval(coeffs, x_centres[ok])
#                           - y_edges_per_bin[ok, j])
#                 wrss  += np.sum(bin_sizes[ok] * resid**2)
#                 npts  += ok.sum()
#             if npts == 0 or wrss <= 0:
#                 bic = np.inf
#             else:
#                 k   = p + 1
#                 bic = npts * np.log(wrss / npts) + k * np.log(npts)
#             bic_score.append(bic)

#         selected_poly_order = int(np.argmin(bic_score))
#         poly_order          = selected_poly_order
#         selection_table     = {"degree": list(degrees), "score": bic_score}

#     # ---------- weighted polynomial fits -----------------------------
#     edge_functions      = None
#     quantile_classifier = None
#     if isinstance(poly_order, int) and poly_order >= 0:
#         edge_functions = []
#         for j in range(Ny_bins + 1):
#             ok = (~np.isnan(y_edges_per_bin[:, j])) & (bin_sizes > 0)
#             if ok.sum() < poly_order + 1:
#                 edge_functions.append(None)
#                 continue
#             coeffs = np.polyfit(
#                 x_centres[ok],
#                 y_edges_per_bin[ok, j],
#                 poly_order,
#                 w=np.sqrt(bin_sizes[ok])
#             )
#             edge_functions.append(np.poly1d(coeffs))

#         def _classifier(x0, y0):
#             edges = np.array([
#                 f(x0) if f is not None else np.nan for f in edge_functions
#             ])
#             return np.searchsorted(edges, y0, side="right") - 1
#         quantile_classifier = _classifier

#     # ---------- return ------------------------------------------------
#     out = dict(
#         x_edges=x_edges,
#         x_centres=x_centres,
#         y_edges_per_bin=y_edges_per_bin,
#     )
#     if edge_functions is not None:
#         out["edge_functions"]      = edge_functions
#         out["quantile_classifier"] = quantile_classifier
#     if counts is not None:  out["counts"]      = counts
#     if av_y   is not None:  out["av_y_values"] = av_y
#     if auto_poly:
#         out["selected_poly_order"] = selected_poly_order
#         out["selection_table"]     = selection_table
#     return out


import numpy as np

def masked_nanmean(arr, axis=1):
    arr = np.asarray(arr)

    # Mask: valid entries are non-nan and non-zero
    mask = (~np.isnan(arr)) & (arr != 0)

    # Replace nans with 0 for summation
    arr_filled = np.nan_to_num(arr, nan=0.0)

    # Sum over valid entries
    sums = np.sum(arr_filled * mask, axis=axis)

    # Count valid entries
    counts = np.sum(mask, axis=axis)

    # Compute mean safely
    out = sums / counts
    out[counts == 0] = np.nan  # if no valid entries, return nan
    return out


import numpy as np
from joblib import Parallel, delayed

# optional – only needed for model="parker" or "empirical"
from scipy.special import lambertw
from scipy.optimize import least_squares


def compute_quantile_edges_function(
    x, y,
    Nx_bins,
    Ny_bins,
    *,
    log_x           = True,
    poly_order      = None,
    auto_poly       = True,
    max_poly_order  = 5,
    criterion       = "bic",
    model           = "parker",          # now: {"poly", "parker", "empirical"}
    n_jobs          = -1,
    return_counts   = False,
    return_avg_y    = False,
    low_pct         = 0.05,
    high_pct        = 99.8,
):
    # --- input checks & percentile clipping -------------------
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError("0 ≤ low_pct < high_pct ≤ 100")
    if model not in {"poly", "parker", "empirical"}:
        raise ValueError("model must be 'poly', 'parker' or 'empirical'")

    lo, hi = np.percentile(y, [low_pct, high_pct])
    keep   = (y >= lo) & (y <= hi)
    x, y   = x[keep], y[keep]

    # --- x‑bin edges --------------------------------------------
    x_edges = (np.logspace(np.log10(x.min()), np.log10(x.max()), Nx_bins + 1)
               if log_x else
               np.linspace(x.min(), x.max(), Nx_bins + 1))

    order                = np.argsort(x)
    x_sorted, y_sorted   = x[order], y[order]
    bin_idx              = np.searchsorted(x_sorted, x_edges)
    q_levels             = np.linspace(0.0, 1.0, Ny_bins + 1)

    # --- quantiles per x‑bin ------------------------------------
    def _bin_q(i):
        s, e = bin_idx[i], bin_idx[i + 1]
        if e - s < 2:
            return np.full_like(q_levels, np.nan)
        return np.quantile(y_sorted[s:e], q_levels)

    y_edges_per_bin = np.vstack(
        Parallel(n_jobs=n_jobs)(delayed(_bin_q)(i) for i in range(Nx_bins))
    )
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    bin_sizes = np.diff(bin_idx)

    # --- optional counts / averages -----------------------------
    counts = av_y = None
    if return_counts or return_avg_y:
        def _stats(i):
            s, e = bin_idx[i], bin_idx[i + 1]
            if e - s < 1:
                return np.zeros(Ny_bins, int), np.full(Ny_bins, np.nan)
            yy = y_sorted[s:e]
            c, _ = np.histogram(yy, bins=y_edges_per_bin[i])
            av  = np.array([
                yy[(yy >= y_edges_per_bin[i,j]) & (yy < y_edges_per_bin[i,j+1])].mean()
                if c[j] else np.nan
                for j in range(Ny_bins)
            ])
            return c, av

        tmp = Parallel(n_jobs=n_jobs)(delayed(_stats)(i) for i in range(Nx_bins))
        if return_counts: counts = np.vstack([t[0] for t in tmp])
        if return_avg_y: av_y  = np.vstack([t[1] for t in tmp])

    # ========== MODEL‑SPECIFIC EDGE FUNCTIONS ======================
    edge_functions      = []
    fit_parameters      = []
    selected_poly_order = None
    selection_table     = None

    # ----------------------------------------------------------------
    # (A) Polynomial branch
    # ----------------------------------------------------------------
    if model == "poly":
        # optional automatic degree selection
        if auto_poly:
            degrees   = range(max_poly_order + 1)
            bic_score = []
            for p in degrees:
                wrss = 0.0
                npts = 0
                for j in range(Ny_bins + 1):
                    ok = (~np.isnan(y_edges_per_bin[:, j])) & (bin_sizes > 0)
                    if ok.sum() < p + 1:
                        continue
                    coeffs = np.polyfit(x_centres[ok], y_edges_per_bin[ok, j],
                                        p, w=np.sqrt(bin_sizes[ok]))
                    resid  = (np.polyval(coeffs, x_centres[ok])
                              - y_edges_per_bin[ok, j])
                    wrss  += np.sum(bin_sizes[ok] * resid**2)
                    npts  += ok.sum()
                bic = (np.inf if (npts == 0 or wrss <= 0)
                       else npts*np.log(wrss/npts) + (p+1)*np.log(npts))
                bic_score.append(bic)
            selected_poly_order = int(np.argmin(bic_score))
            poly_order          = selected_poly_order
            selection_table     = {"degree": list(degrees), "score": bic_score}

        if not isinstance(poly_order, int) or poly_order < 0:
            raise ValueError("poly_order must be a non‑negative integer")

        for j in range(Ny_bins + 1):
            ok = (~np.isnan(y_edges_per_bin[:, j])) & (bin_sizes > 0)
            if ok.sum() < poly_order + 1:
                edge_functions.append(None)
                fit_parameters.append(None)
                continue
            coeffs = np.polyfit(x_centres[ok], y_edges_per_bin[ok, j],
                                poly_order, w=np.sqrt(bin_sizes[ok]))
            fit_parameters.append(coeffs)

            p = np.poly1d(coeffs)
            edge_functions.append(p)

    # ----------------------------------------------------------------
    # (B) Parker branch
    # ----------------------------------------------------------------
    elif model == "parker":
        def parker_speed(r, C, rc):
            r  = np.asarray(r, float)
            R  = 4.0 * (np.log(r/rc) + rc/r - 1.0)
            z  = -np.exp(-(R + 1.0))
            w0  = lambertw(z, k=0).real
            w_1 = lambertw(z, k=-1).real
            w   = np.where(r <= rc, w0, w_1)
            return C * np.sqrt(-w)

        for j in range(Ny_bins + 1):
            ok = (~np.isnan(y_edges_per_bin[:, j])) & (bin_sizes > 0)
            if ok.sum() < 3:
                edge_functions.append(None)
                fit_parameters.append(None)
                continue

            r_data = x_centres[ok]
            v_data = y_edges_per_bin[ok, j]
            wts    = np.sqrt(bin_sizes[ok])

            C0  = np.nanmedian(v_data) / np.sqrt(2.0)
            rc0 = np.nanmedian(r_data)

            def resid_parker(p):
                return (parker_speed(r_data, *p) - v_data) * wts

            sol = least_squares(resid_parker, x0=[C0, rc0],
                                bounds=([0.0, 0.0], np.inf))
            C_fit, rc_fit = sol.x

            fit_parameters.append((C_fit, rc_fit))
            edge_functions.append(
                lambda r, C=C_fit, rc=rc_fit: parker_speed(r, C, rc)
            )

    # ----------------------------------------------------------------
    # (C) Empirical exponential branch
    # ----------------------------------------------------------------
    else:  # model == "empirical"
        def empirical_speed(r, u_inf, r1, a):
            return u_inf * (1 - np.exp(- (r / r1)**a))

        for j in range(Ny_bins + 1):
            ok = (~np.isnan(y_edges_per_bin[:, j])) & (bin_sizes > 0)
            if ok.sum() < 3:
                edge_functions.append(None)
                fit_parameters.append(None)
                continue

            r_data = x_centres[ok]
            v_data = y_edges_per_bin[ok, j]
            wts    = np.sqrt(bin_sizes[ok])

            # initial guesses
            u_inf0 = np.nanmax(v_data)
            r10    = np.nanmedian(r_data)
            a0     = 1.0

            def resid_empirical(p):
                return (empirical_speed(r_data, *p) - v_data) * wts

            sol = least_squares(
                resid_empirical,
                x0=[u_inf0, r10, a0],
                bounds=([0.0, 0.0, 0.0], np.inf)
            )
            u_inf_fit, r1_fit, a_fit = sol.x

            fit_parameters.append((u_inf_fit, r1_fit, a_fit))
            edge_functions.append(
                lambda r, ui=u_inf_fit, r1=r1_fit, a=a_fit: empirical_speed(r, ui, r1, a)
            )

    # --- common classifier -----------------------------------------
    def _classifier(x0, y0):
        edges = np.array([
            f(x0) if f is not None else np.nan
            for f in edge_functions
        ])
        return np.searchsorted(edges, y0, side="right") - 1

    # --- package output --------------------------------------------
    out = dict(
        x_edges             = x_edges,
        x_centres           = x_centres,
        y_edges_per_bin     = y_edges_per_bin,
        edge_functions      = edge_functions,
        quantile_classifier = _classifier,
        fit_parameters      = fit_parameters,
        model               = model,
    )
    if counts is not None:    out["counts"]      = counts
    if av_y is not None:      out["av_y_values"] = av_y
    if model == "poly" and auto_poly:
        out["selected_poly_order"] = selected_poly_order
        out["selection_table"]     = selection_table

    return out


from scipy.special import lambertw

# Synthetic demonstration of full vs. surrogate Parker‐wind curves.
# Replace fit_params, d_all, and v_all with your own quantile_functions and data_dict.

# --- Define helper functions ---
def eval_lambertw(r, fit_parameters):
    r = np.asarray(r, float)
    params = np.asarray(fit_parameters, float)
    C  = params[:, 0][:, None]
    rc = params[:, 1][:, None]
    R  = r[None, :]
    X  = 4.0 * (np.log(R/rc) + rc/R - 1.0)
    Z  = -np.exp(-(X + 1.0))
    W0 = lambertw(Z, k=0).real
    W1 = lambertw(Z, k=-1).real
    mask = (R <= rc)
    W = np.where(mask, W0, W1)
    return C * np.sqrt(-W)

def build_parker_surrogates(fit_parameters, r_min, r_max, n_samples=100, poly_degree=5):
    r_samp = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_samples))
    V_mat = eval_lambertw(r_samp, fit_parameters)
    ln_r = np.log(r_samp)
    coefs = np.zeros((len(fit_parameters), poly_degree+1))
    for j in range(len(fit_parameters)):
        ln_V = np.log(V_mat[j])
        coefs[j] = np.polyfit(ln_r, ln_V, poly_degree)
    return coefs

def eval_parker_surrogates(r, coefs):
    ln_r = np.log(r)
    ln_V = np.vstack([np.polyval(coefs[j], ln_r) for j in range(len(coefs))])
    return np.exp(ln_V)



def compute_optimal_y_bins(x,
                           y,
                           Nx_bins,
                           Ny_bins,
                           log_x         = True,
                           n_jobs        = -1,
                           return_counts = False,
                           return_avg_y  = False):
    """
    Given arrays x and y, this function:
      1. Creates Nx_bins x bins (using linear or logarithmic spacing).
      2. Within each x bin, splits the corresponding y values into Ny_bins bins (via quantiles).
      3. Computes the optimal y bin edges as the median of the y bin edges computed across x bins.
      4. Optionally returns:
         - counts: a (Nx_bins, Ny_bins) counts matrix (number of points per x bin for each global y bin).
         - avg_y: overall average y per x bin.
         - av_y_values: a (Nx_bins, Ny_bins) matrix of the cell‐averaged y values.
    
    Returns a dictionary with keys:
        'x_edges': x bin edges,
        'y_edges': optimal y bin edges,
        'x': x bin centers,
        and optionally 'counts', 'avg_y', 'av_y_values'.
    """
    def _compute_quantile_for_bin(start, end, y_sorted, q):
        if start == end:
            return None
        return np.quantile(y_sorted[start:end], q)
    
    def _compute_counts_for_xbin(i, bin_idx, y_sorted, optimal_y_edges, Ny_bins):
        start, end = bin_idx[i], bin_idx[i+1]
        if start == end:
            return np.zeros(Ny_bins, dtype=int)
        return np.histogram(y_sorted[start:end], bins=optimal_y_edges)[0]
    
    def _compute_cell_avg_for_xbin(i, bin_idx, y_sorted, optimal_y_edges, Ny_bins):
        start, end = bin_idx[i], bin_idx[i+1]
        if start == end:
            return np.full(Ny_bins, np.nan)
        y_cell = y_sorted[start:end]
        avg_values = np.empty(Ny_bins, dtype=float)
        for j in range(Ny_bins):
            lower = optimal_y_edges[j]
            upper = optimal_y_edges[j+1]
            # Use a half-open interval except for the last bin
            if j < Ny_bins - 1:
                mask = (y_cell >= lower) & (y_cell < upper)
            else:
                mask = (y_cell >= lower) & (y_cell <= upper)
            avg_values[j] = np.mean(y_cell[mask]) if np.any(mask) else np.nan
        return avg_values
    
    def _compute_avg_for_xbin(i, bin_idx, y_sorted):
        start, end = bin_idx[i], bin_idx[i+1]
        return np.mean(y_sorted[start:end]) if start != end else np.nan
    


    
    results = {}

    # Ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute x bin edges (linear or logarithmic)
    if log_x:
        x_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), Nx_bins+1)
    else:
        x_edges = np.linspace(x.min(), x.max(), Nx_bins+1)
    results['x_edges'] = x_edges

    # Sort x and y for fast slicing
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # Find indices corresponding to x bin boundaries
    bin_idx = np.searchsorted(x_sorted, x_edges)
    
    # Precompute quantile levels for Ny_bins bins
    q = np.linspace(0, 1, Ny_bins+1)
    
    # Parallel computation: compute quantile edges for each x bin
    quantiles = Parallel(n_jobs=n_jobs)(
        delayed(_compute_quantile_for_bin)(bin_idx[i], bin_idx[i+1], y_sorted, q)
        for i in range(Nx_bins)
    )
    
    # Compute x bin centers
    xb = x_edges[:-1] + 0.5 * (x_edges[1:] - x_edges[:-1])
    results['x'] = xb

    # Filter out empty x bins
    valid_quantiles = [q_arr for q_arr in quantiles if q_arr is not None]
    if not valid_quantiles:
        raise ValueError("No valid x bins with data found.")
    
    # Stack per-bin quantile arrays and compute median across x bins to get global y edges
    y_quantiles = np.vstack(valid_quantiles)
    optimal_y_edges = np.median(y_quantiles, axis=0)
    results['y_edges'] = optimal_y_edges
    
    if return_counts:
        counts_list = Parallel(n_jobs=n_jobs)(
            delayed(_compute_counts_for_xbin)(i, bin_idx, y_sorted, optimal_y_edges, Ny_bins)
            for i in range(Nx_bins)
        )
        counts = np.vstack(counts_list)
        results['counts'] = counts
    

    if return_avg_y:
        cell_avg_list = Parallel(n_jobs=n_jobs)(
            delayed(_compute_cell_avg_for_xbin)(i, bin_idx, y_sorted, optimal_y_edges, Ny_bins)
            for i in range(Nx_bins)
        )
        cell_avg_y = np.vstack(cell_avg_list)
        results['av_y_values'] = cell_avg_y
    
    return results



def create_folder_if_not_exists(path0, overwrite_files=False):
    folder_path = Path(path0)
    
    # Check if folder exists or overwrite_files is True
    if not folder_path.exists() or overwrite_files:
        folder_path.mkdir( exist_ok=True)
        print(f"Folder created or overwritten: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

from collections.abc import Iterable

def ensure_iterable(obj):
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return obj
    else:
        return [obj]


def savepickle_dill(df_2_save, save_path, filename):
    file_path = Path(save_path).joinpath(filename)
    with open(file_path, 'wb') as file:
        pickle.dump(df_2_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
import dill

def load_and_construct_lambdas(save_path, fname):
    """
    Load the coefficients and construct lambda functions.
    
    Parameters:
    save_path (str): Directory path where the file is saved.
    fname (str): Filename of the saved file.
    
    Returns:
    dict: Dictionary containing the reconstructed lambda functions.
    """
    # Full path to the file
    full_path = os.path.join(save_path, fname)

    # Load the coefficients
    with open(full_path, 'rb') as file:
        coefficients_dict = dill.load(file)

    # Reconstruct the lambda functions
    f_dict = {}
    for key, coeffs in coefficients_dict.items():
        f_dict[key] = lambda x, c=coeffs: sum(c_i * (x ** i) for i, c_i in enumerate(c[::-1]))

    return f_dict


def format_datetime_to_string(numpy_datetime):
    """
    Converts a numpy.datetime64 object to a string in 'YYYY-MM-DD HH:MM' format.
    
    Parameters:
    numpy_datetime (numpy.datetime64): The input datetime object.
    
    Returns:
    str: Formatted datetime string.
    """
    # Convert to datetime object
    datetime_obj = numpy_datetime.astype('datetime64[s]').tolist()
    
    # Convert to the desired string format: 'YYYY-MM-DD HH:MM'
    formatted_time = datetime_obj.strftime('%Y-%m-%d %H:%M')
    
    return formatted_time


def estimate_derivatives(x, y):
    """
    Estimate the first and second derivatives of a function using central differences.
    
    :param x: An array of x values.
    :param y: An array of y values.
    :return: Arrays of the first and second derivatives of y.
    """
    # First derivative (dy/dx)
    dy = np.gradient(y, x, edge_order=2)
    
    # Second derivative (d^2y/dx^2)
    d2y = np.gradient(dy, x, edge_order=2)
    
    return dy, d2y

def compute_curvature(x, y):
    """
    Compute the curvature of a function.
    
    :param x: An array of x values.
    :param y: An array of y values.
    :return: An array of curvature values.
    """
    dy, d2y = estimate_derivatives(x, y)
    
    # Curvature formula
    curvature = np.abs(d2y) / (1 + dy**2)**1.5
    
    return curvature

import pytplot
def tplot_to_dataframe(file_path, 
                       var_name                 = None, 
                       convert_time_to_datetime = True,
                       time_unit                = 's'):
    """
    Restore a TPlot file (IDL .tplot or .sav with TPlot variables) using pytplot
    and convert a specified TPlot variable into a pandas DataFrame with a DateTimeIndex.
    
    Parameters
    ----------
    file_path : str
        Full path to the TPlot save file (e.g. '.tplot' or '.sav').
    var_name : str, optional
        Name of the TPlot variable inside the file. If None, and the file
        contains exactly one variable, we use that one. Otherwise, this must be specified.
    convert_time_to_datetime : bool, optional
        If True, converts numeric time (often seconds since 1970) to a DateTimeIndex.
        If False, leaves time as numeric values.
    time_unit : str, optional
        If converting time to DateTimeIndex, the unit of the time array. Typically 's' 
        for seconds. Options include 'ms', 'ns', etc., depending on your data.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame indexed by time (either numeric or a DateTimeIndex),
        with one or more columns for the TPlot variable data.

    Raises
    ------
    FileNotFoundError
        If the file_path does not exist on disk.
    ValueError
        If the file contains no TPlot variables, or if var_name is specified but not found,
        or if multiple variables exist and var_name is not specified.
    """

    # 1. Check if the file exists on disk
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 2. Attempt to restore the TPlot file
    #    If this fails due to the file not being a valid TPlot file,
    #    pytplot might print a warning or raise an error, and no variables will load.
    pytplot.tplot_restore(file_path)

    # 3. See which TPlot variables were loaded
    all_vars = pytplot.tplot_names()

    if len(all_vars) == 0:
        # Means no TPlot variables recognized from this file
        raise ValueError(
            f"No TPlot variables found in file: {file_path}\n"
            f"Check if it's really a TPlot save file created with IDL tplot_save()."
        )

    # If user didn't specify var_name, auto-pick if there's exactly one
    if var_name is None:
        if len(all_vars) == 1:
            var_name = all_vars[0]
            print(f"Auto-selected single TPlot variable: {var_name}")
        else:
            raise ValueError(
                f"Multiple TPlot variables found. Please specify var_name.\n"
                f"Available variables: {all_vars}"
            )
    else:
        if var_name not in all_vars:
            raise ValueError(
                f"Requested var_name='{var_name}' not in loaded TPlot variables: {all_vars}"
            )

    # 4. Retrieve data for the chosen variable
    result = pytplot.get_data(var_name)
    if result is None:
        raise ValueError(
            f"Could not retrieve data for TPlot variable '{var_name}'. "
            "Possibly no valid data in the file."
        )

    # get_data can return either (time, data) or (time, data, metadata)
    if len(result) == 2:
        time_vals, data_vals = result
    elif len(result) == 3:
        time_vals, data_vals, _ = result
    else:
        raise ValueError("Unexpected format from pytplot.get_data().")

    # 5. Convert time array if requested
    if convert_time_to_datetime:
        # By default, TPlot times are often in seconds since 1970.
        time_index = pd.to_datetime(time_vals, unit=time_unit, origin='unix')
    else:
        # Leave time as numeric
        time_index = pd.Index(time_vals, name='time')

    # 6. Wrap in a DataFrame
    if data_vals.ndim == 1:
        # 1D data => single column named after var_name
        df = pd.DataFrame(data_vals, index=time_index, columns=[var_name])
    else:
        # 2D or more => multiple columns
        # Name each column var_name_0, var_name_1, etc.
        col_count = data_vals.shape[1]
        col_names = [f"{var_name}_{i}" for i in range(col_count)]
        df        = pd.DataFrame(data_vals, index=time_index, columns=col_names)

    return df







def clean_data(x, y):
    """Remove non-finite values from the data."""
    finite_mask = np.isfinite(x) & np.isfinite(y)
    return x[finite_mask], y[finite_mask]



def symlogspace(start, end, num=50, linthresh=1):
    """
    Generate a symmetric logarithmic scale array.

    Parameters:
    - start, end: The starting and ending values of the sequence.
    - num: Number of samples to generate.
    - linthresh: The range within which the plot is linear (to avoid having a zero value).

    Returns:
    - ndarray
    """

    if start * end > 0:
        raise ValueError("Start and end values must have different signs for symlogspace to be meaningful.")

    # Divide the number of bins to account for both negative and positive values.
    num_half = num // 2

    # Create the logarithmic spaces for negative and positive values.
    log_neg = np.logspace(np.log10(linthresh), np.log10(abs(start)), num_half)
    log_pos = np.logspace(np.log10(linthresh), np.log10(end), num_half)

    # Combine the negative and positive logarithmic spaces.
    return np.concatenate((-log_neg[::-1], log_pos))

def most_common(List):
    return(mode(List))

def load_files(load_path, filenames, conect_2= '', sort= True):
    import glob
    
    pattern = Path(load_path, '*', conect_2, filenames)
    print(pattern)
    if sort:
        fnames = np.sort(glob.glob(str(pattern)))
    else:
        
        fnames = glob.glob(str(pattern))      
    
    return fnames


@njit(parallel=True)
def custom_nansum_product(xvec, yvec, axis):
    result = np.zeros(xvec.shape[1-axis], dtype=xvec.dtype)
    # Parallelizing the outer loop
    for j in prange(xvec.shape[1-axis]):
        for i in range(xvec.shape[axis]):
            if axis == 0:
                if not np.isnan(xvec[i, j]) and not np.isnan(yvec[i, j]):
                    result[j] += xvec[i, j] * yvec[i, j]
            else:
                if not np.isnan(xvec[j, i]) and not np.isnan(yvec[j, i]):
                    result[j] += xvec[j, i] * yvec[j, i]
    return result




import sqlite3
import os

def print_last_100_commands():
    # Locate the default IPython history file.
    # If you use a different profile, adjust the path accordingly.
    history_path = os.path.expanduser("~/.ipython/profile_default/history.sqlite")
    
    if not os.path.exists(history_path):
        print("IPython history file not found. Check your IPython profile settings.")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(history_path)
    cursor = conn.cursor()
    
    # The history is stored in a table named 'history' where each row represents a command.
    # This query orders by the session and line number in descending order, then limits to 30 entries.
    query = """
        SELECT source
        FROM history
        ORDER BY session DESC, line DESC
        LIMIT 100
    """
    cursor.execute(query)
    commands = cursor.fetchall()

    conn.close()

    # Print each command cell
    for idx, (command,) in enumerate(commands, start=1):
        print(f"--- Command {idx} ---")
        print(command)
        print()



def find_matching_files_with_common_parent(f_names,
                                           f_file_name,
                                           gen_names,
                                           gen_file_name,
                                           num_parents_f=1,
                                           num_parents_g=1):
    
    gen_parent = [Path(gen_name).parents[num_parents_g-1] for gen_name in gen_names]
    f_parents  = [Path(f_name).parents[num_parents_f-1] for f_name in f_names]
    
    
    parents    = list(set(gen_parent).intersection(f_parents))
    
    f_names    = [Path(parent).joinpath( f_file_name) for parent in parents]
    gen_names  = [Path(parent).joinpath( gen_file_name) for parent in parents]

    return list(np.sort(np.array(f_names).astype(str))),  list(np.sort(np.array(gen_names).astype(str)))



def delete_files_and_folders(file_and_folder_list):
    import shutil
    from pathlib import Path
    for fname in file_and_folder_list:
        try:
            if Path(fname).is_file():
                Path(fname).unlink()
                print(f"Deleted file: {fname}")
            elif Path(fname).is_dir():
                shutil.rmtree(fname)
                print(f"Deleted directory: {fname}")
        except Exception as e:
            print(f"Error deleting {fname}: {e}")


def generate_date_range_df(Start_date, 
                           End_date, 
                           step,
                           step2):
    """
    Generate a DataFrame with a date range.

    Args:
        Start_date (str): The starting date in the format 'YYYY-MM-DD'.
        End_date (str): The ending date in the format 'YYYY-MM-DD'.
        step (int): The number of days in each interval.
        step2 (int): The number of days to subtract from the previous 'End_date'.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'Starting_date' and 'Ending_date'.
                      Both columns contain Timestamp objects representing date intervals.

    Example:
        >>> start_date = '2023-08-01'
        >>> end_date = '2023-08-15'
        >>> step = 3
        >>> step2 = 1
        >>> result_df = generate_date_range_df(start_date, end_date, step, step2)
        >>> print(result_df)
    """
    from datetime import datetime, timedelta
    start_datetime  = datetime.strptime(Start_date, '%Y-%m-%d')
    end_datetime    = datetime.strptime(End_date, '%Y-%m-%d')
    step_timedelta  = timedelta(days=step)
    step2_timedelta = timedelta(days=step2)

    dates = []
    while start_datetime < end_datetime:
        end_of_range = start_datetime + step_timedelta
        dates.append((start_datetime, end_of_range))
        start_datetime = end_of_range - step2_timedelta  # Subtract step2
        
    df = pd.DataFrame(dates, columns=['Start', 'End'])
    df['Start'] = pd.to_datetime(df['Start'])  # Convert to Timestamp
    df['End'] = pd.to_datetime(df['End'])      # Convert to Timestamp
    return df



import numpy as np
from numba import njit, prange





# Original function
def angle_between_vectors(V,
                          B,
                          return_denom  = False,
                          restrict_2_90 = False):
                    
    """
    Calculate the angle between two vectors.

    Args:
        V (np.ndarray)                : A 2D numpy array representing the first vector.
        B (np.ndarray)                : A 2D numpy array representing the second vector.
        return_denom (bool, optional) : Whether to return the denominator components.
        restrict_2_90(bool, optional) : Restrict angles to 0-90
            Defaults to False.
            
    Returns:
        np.ndarray                    : A 1D numpy array representing the angles in degrees between the two input vectors.
        tuple                         : A tuple containing the angle, dot product, V_norm, and B_norm (if denom is True).
    """
    
    V_norm      = estimate_vec_magnitude(V)
    B_norm      = estimate_vec_magnitude(B)
    dot_prod    = dot_product(V, B)

    if restrict_2_90:
        angle       = np.arccos(np.abs(dot_prod) / (V_norm * B_norm)) / np.pi * 180
    else:
        angle       = np.arccos(dot_prod / (V_norm * B_norm)) / np.pi * 180       
        
    if return_denom:
        return angle, dot_prod, V_norm, B_norm
    else:
        return angle
    
    
def dot_product(xvec, yvec ):
    # Determine which axis is shorter
    axis_to_sum = 1 if xvec.shape[0] > xvec.shape[1] else 0

    # Calculate the product of the two arrays, handling NaNs effectively

    # Sum along the determined shorter axis, skipping NaNs
    result = np.nansum(xvec * yvec, axis=axis_to_sum)
    
    return result

def estimate_vec_magnitude(a):
    """
    Estimate the magnitude of each vector in the input array `a`.

    Parameters:
    ----------
    a : numpy.ndarray
        The input array containing vectors. The shape of `a` should be (N, M) or (M, N), where N is the number of vectors,
        and M is the dimensionality of each vector.

    Returns:
    -------
    numpy.ndarray
        An array containing the magnitude of each vector in `a`. The output array will have shape (N,) if the input
        array `a` has shape (N, M), or shape (M,) if the input array `a` has shape (M, N).
    """

    shortest_axis = 0 if a.shape[0] <= a.shape[1] else 1

    return  np.sqrt(np.nansum(a**2, axis=shortest_axis))


def perp_vector(a, b, return_paral_comp = False):
    """
    This function calculates the component of a vector perpendicular to another vector.

    Parameters:
    a (ndarray) : A 2D numpy array representing the first vector.
    b (ndarray) : A 2D numpy array representing the second vector.

    Returns:
    ndarray     : A 2D numpy array representing the component of the first input vector that is perpendicular to the second input vector.
    """
    b_unit = b / estimate_vec_magnitude(b)[:, np.newaxis]
    proj   = dot_product(a, b_unit)[:, np.newaxis]* b_unit
    perp   = a - proj
    if return_paral_comp:
        
        return perp, proj
    else:
        return perp
        

def update_dates_strings(t0, t1, addit_time):
    """
    Update the given datetime strings by adding or subtracting a specific amount of time.

    This function takes two datetime strings `t0` and `t1`, and an `addit_time` (time duration in seconds) and
    returns updated datetime strings by subtracting `addit_time` seconds from the first datetime (`t0`) and adding
    `addit_time` seconds to the second datetime (`t1`).

    Parameters:
    ----------
    t0 : str
        The first datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
    t1 : str
        The second datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
    addit_time : int or float
        The time duration in seconds to be added to `t1` and subtracted from `t0`.

    Returns:
    -------
    tuple of str
        A tuple containing two updated datetime strings. The first element of the tuple is the updated `t0` datetime
        string, and the second element is the updated `t1` datetime string.

    Example:
    --------
    >>> t0 = '2023-08-05 10:30:00'
    >>> t1 = '2023-08-06 12:45:00'
    >>> addit_time = 20

    >>> updated_t0, updated_t1 = update_dates_strings(t0, t1, addit_time)
    >>> print(updated_t0)
    '2023-08-05 10:29:40'
    >>> print(updated_t1)
    '2023-08-06 12:45:20'
    """
    from datetime import datetime, timedelta

    # Convert strings to datetime objects
    format_str = '%Y-%m-%d %H:%M:%S'
    dt0 = datetime.strptime(t0, format_str)
    dt1 = datetime.strptime(t1, format_str)

    # Subtract `addit_time` seconds from the first date
    new_dt0 = dt0 - timedelta(seconds=addit_time)

    # Add `addit_time` seconds to the second date
    new_dt1 = dt1 + timedelta(seconds=addit_time)

    # Convert datetime objects back to strings
    new_t0 = new_dt0.strftime(format_str)
    new_t1 = new_dt1.strftime(format_str)

    return new_t0, new_t1


def filter_dict(d, keys_to_keep):
    return dict(filter(lambda item: item[0] in keys_to_keep, d.items()))


from dateutil import parser

def format_date_to_str(date_input):
    """
    Takes a date input (string or object that can be converted to a string) and attempts to parse and format it
    into a 'YYYY-MM-DD HH:MM' format.

    Parameters:
    - date_input: The date input to be formatted. Can be a string or an object that can be converted to a string.

    Returns:
    - A string representing the formatted date in 'YYYY-MM-DD HH:MM' format if successful.
    - None if parsing fails.
    """
    try:
        # Convert input to string in case it's not already a string
        date_str = str(date_input)
        # Try to parse the date string
        date_obj = parser.parse(date_str)
        # Format the datetime object to the desired format
        formatted_date_str = date_obj.strftime('%Y-%m-%d %H:%M')
        return formatted_date_str
    except ValueError as e:
        print(f"Error parsing date: {e}")
        # Return None or consider a default value or re-raise the exception based on your use case
        return None
    
def replace_negative_with_nan(df):
    """
    Replace negative values with NaN in a DataFrame.
    """
    return df.where(df >= -1e5, np.nan)


def string_to_datetime_index(datetime_string, datetime_format='%Y-%m-%d %H:%M:%S'):
    return pd.to_datetime(datetime_string, format=datetime_format)

def string_to_timestamp(datetime_string, datetime_format='%Y-%m-%d %H:%M:%S'):
    """
    Converts a string representation of a date and time to a timestamp in the format 'Timestamp('YYYY-MM-DD HH:MM:SS.ssssss')'.

    Parameters:
    datetime_string (str): The string representation of the date and time.
    datetime_format (str, optional): The format of the input string. Defaults to '%Y-%m-%d %H:%M:%S'.

    Returns:
    str: The timestamp representation of the date and time in the format 'Timestamp('YYYY-MM-DD HH:MM:SS.ssssss')'.

    Raises:
    ValueError: If the input string does not match the specified format.
    """
    datetime_object = datetime.datetime.strptime(datetime_string, datetime_format)
    timestamp = datetime_object.timestamp()
    return f"Timestamp('{datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')}')"

def add_time_to_datetime_string(start_time, time_amount, time_unit):
    """
    Adds a specified amount of time to a datetime string and returns the result as a string.

    Parameters:
    start_time (str): The original datetime string, which may or may not include fractional seconds.
    time_amount (int): The amount of time to add.
    time_unit (str): The unit of the added time, either 's' (seconds), 'm' (minutes), 'h' (hours), or 'd' (days).

    Returns:
    str: The datetime string after the specified time has been added, in the format '%Y-%m-%d %H:%M:%S'.

    Raises:
    ValueError: If an invalid time unit is specified.
    """
    import datetime

    # Define the mapping of time units to their corresponding attributes in timedelta
    units = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days'}
    unit = units.get(time_unit)

    # Raise an error if the time unit is invalid
    if unit is None:
        raise ValueError("Invalid time unit")

    # Create a timedelta object with the specified time amount and unit
    delta = datetime.timedelta(**{unit: time_amount})

    # Try parsing the datetime string with and without fractional seconds
    formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']
    for fmt in formats:
        try:
            # Attempt to parse the datetime string
            start_datetime = datetime.datetime.strptime(start_time, fmt)
            break
        except ValueError:
            continue
    else:
        # If both parsing attempts fail, raise an exception
        raise ValueError("start_time does not match expected formats")

    # Add the time delta to the parsed datetime
    end_datetime = start_datetime + delta

    # Return the resulting datetime string in the specified format, without fractional seconds
    return end_datetime.strftime('%Y-%m-%d %H:%M:%S')
    

from datetime import timedelta
import re

def parse_time_duration(duration_str):
    # Define regex to capture value and unit
    time_regex = re.compile(r'(\d+)([a-z]+)')
    
    # Define supported time units and their equivalent in timedelta
    unit_mapping = {
        'ms': 'milliseconds',
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days'
    }
    
    matches = time_regex.findall(duration_str)
    if not matches:
        raise ValueError(f"Invalid time duration format: {duration_str}")
    
    kwargs = {}
    for value, unit in matches:
        if unit in unit_mapping:
            kwargs[unit_mapping[unit]] = int(value)
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
    
    return timedelta(**kwargs)


def count_fits_in_duration(df, wind_size):
    """
    This function takes a pandas dataframe with a datetime index and an integer wind_size as input.
    It calculates the total duration of the dataframe in hours and then finds how many times wind_size can fit in the duration.
    """
    
    # Calculate the total duration of the dataframe in hours
    total_duration = (df.index[-1] - df.index[0]).total_seconds() / 3600
    
    #print(total_duration)
    
    # Find how many times wind_size can fit into the duration
    fits_in_duration = int(total_duration // wind_size)
    
    return fits_in_duration


def smooth_filter(xv, arr, window):
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    # Convolve with sobel filter
    xv, arr = clean_data(xv, arr)
    grad    = signal.convolve(arr, [1,-1,0])[:-1]
    # Smooth gradient
    smooth_grad = smooth_grad = gaussian_filter1d(grad, window)
    
    return smooth_grad


import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt

@njit
def custom_median(arr):
    """Compute the median of a 1D array."""
    sarr = np.sort(arr)
    n = sarr.shape[0]
    mid = n // 2
    if n % 2 == 0:
        return 0.5 * (sarr[mid - 1] + sarr[mid])
    else:
        return sarr[mid]

@njit
def hampel_filter_core(arr, window_size, n_sigmas):
    """
    Core Hampel filter that processes every element.
    At the boundaries the window is reduced to the available points.
    """
    half_window = window_size // 2
    n = arr.shape[0]
    filtered = arr.copy()
    for i in range(n):
        # Determine window boundaries (using available points at edges)
        start = i - half_window if i - half_window >= 0 else 0
        end = i + half_window + 1 if i + half_window + 1 <= n else n
        window = arr[start:end]
        med = custom_median(window)
        mad = custom_median(np.abs(window - med))
        threshold = n_sigmas * 1.4826 * mad
        if np.abs(arr[i] - med) > threshold:
            filtered[i] = med
    return filtered

def hampel(arr, window_size=5, n=3):
    """
    Apply the Hampel filter to a 1D time series to replace outliers.

    Parameters
    ----------
    arr : numpy.ndarray, pandas.Series, or pandas.DataFrame
        1D input time series.
    window_size : int, optional
        Size of the sliding window (must be odd). If even, it will be adjusted.
    n : int or float, optional
        The number of MADs (after scaling) used to define outliers.

    Returns
    -------
    filtered_arr : numpy.ndarray
        The filtered time series with outliers replaced by the window median.
    outlier_indices : numpy.ndarray
        Indices where outliers were detected and replaced.
    """
    # Convert input to a 1D numpy array
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.values.flatten()
    else:
        arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional!")
        
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"window_size adjusted to {window_size} for symmetry.")
    
    # Apply the core Hampel filter
    filtered_arr = hampel_filter_core(arr, window_size, n)
    
    # Determine outlier indices (where the value was replaced)
    outlier_indices = np.where(filtered_arr != arr)[0]
    
    return filtered_arr, outlier_indices

# solve for a and b
def best_fit(X, Y):
    """
    Function to calculate the best linear fit for a given set of data.

    Parameters
    ----------
    X: list or numpy array
        The x-values of the data set
    Y: list or numpy array
        The y-values of the data set

    Returns
    -------
    a: float
        The y-intercept of the best fit line
    b: float
        The slope of the best fit line
    """

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum(xi*yi for xi,yi in zip(X, Y)) - n * xbar * ybar
    denum = sum(xi**2 for xi in X) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

def powlaw(x, a, b) : 
    return a * np.power(x, b)
def expo(x, a, b) :
    return a*np.exp(-b*x)
def linlaw(x, a, b) : 
    return a + x * b

def curve_fit_log(xdata, ydata) : 
    """
    Function to fit data to a power law with weights according to a log scale.

    Parameters
    ----------
    xdata: numpy array
        The x-values of the data set to fit
    ydata: numpy array
        The y-values of the data set to fit

    Returns
    -------
    popt_log: tuple
        The parameters of the best fit line
    pcov_log: numpy array
        The covariance of the parameters
    ydatafit_log: numpy array
        The y-values of the best fit line

    """
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# plaw_fit_est_plot
# ----------------------------------------------------------------------
def plaw_fit_est_plot(
    x, y,
    x0, xf,
    ax=None,
    n_plot: int = 200,
    var_symbol: str = "R",
    **plot_kwargs,
):
    """
    Fit and (optionally) plot a power law :math:`y = A\,x^{m}` on
    the interval :math:`[x_0,\,x_f]`.

    Parameters
    ----------
    x, y : array‑like
        Data vectors.
    x0, xf : float
        Lower and upper bounds of the fitting window.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the fitted curve.
    n_plot : int, default = 200
        Number of points in the smooth curve plotted for visualisation.
    var_symbol : str, default = ``"R"``
        LaTeX variable to display as the base of the power‑law in the legend.
        For example, ``var_symbol="k"`` will label the fit as
        :math:`k^{m\\,\\pm\\,\\Delta m}`.
    **plot_kwargs
        Passed directly to :py:meth:`matplotlib.axes.Axes.loglog`.

    Returns
    -------
    x_fit : ndarray
        Geometric grid spanning :math:`[x_0,\,x_f]` (length = *n_plot*).
    y_fit : ndarray
        Best‑fit curve :math:`A\,x^{m}`.
    plaw_exp : float
        Slope :math:`m` of the power law.
    err_plaw_exp : float
        1‑σ uncertainty on :math:`m` (from covariance of the log–log fit).
    """
    # --- 1. clean & select data ------------------------------------------
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = (x > 0) & (y > 0)      # enforce domain (logarithms below)
    x, y = x[good], y[good]

    sel = (x >= x0) & (x <= xf)
    if sel.sum() < 2:
        raise ValueError("Not enough points in fitting window")

    x_win, y_win = x[sel], y[sel]

    # --- 2. log–log linear regression ------------------------------------
    logx, logy = np.log(x_win), np.log(y_win)
    (m, logA), cov = np.polyfit(logx, logy, 1, cov=True)
    err_m = float(np.sqrt(cov[0, 0]))
    A     = float(np.exp(logA))

    # --- 3. smooth curve for plotting ------------------------------------
    x_fit = np.geomspace(x0, xf, n_plot)
    y_fit = A * x_fit**m

    # --- 4. optional plotting --------------------------------------------
    if ax is None:
        ax = plt.gca()
    lbl = rf"${var_symbol}^{{{m:.2f}\,\pm\,{err_m:.2f}}}$"
    ax.loglog(x_fit, y_fit, label=lbl, **plot_kwargs)

    return x_fit, y_fit, m, err_m


def find_fit(x, y, x0, xf, return_fit_values=False):
    """
    Perform a log–log (power‐law) fit of y(x) over [x0, xf], using natural logs.

    Returns fit parameters and, optionally, the fit curve values.
    """
    x = np.array(x)
    y = np.array(y)
    # Sort and clean positive values
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    mask = y > 0
    x = x[mask]
    y = y[mask]

    # Determine slice indices for the fitting range
    s = np.searchsorted(x, x0, 'left')
    e = np.searchsorted(x, xf, 'right')
    if e - s < 2:
        raise ValueError("Not enough points in fitting range")

    # Fit log–log: log y = a + m log x
    logx = np.log(x[s:e])
    logy = np.log(y[s:e])
    m, a = np.polyfit(logx, logy, 1)

    if return_fit_values:
        x_fit = x[s:e]
        y_fit = np.exp(a) * x_fit**m
        return (a, m), s, e, x_fit, y_fit

    return (a, m), s, e, x, y

def curve_fit_log_wrap(x, y, x0, xf):  
    
    from scipy.optimize import curve_fit
    
    def linlaw(x, a, b) : 
        return a + x * b


    def curve_fit_log(xdata, ydata) : 

        """Fit data to a power law with weights according to a log scale"""
        # Weights according to a log scale
        # Apply fscalex
        xdata_log = np.log10(xdata)
        # Apply fscaley
        ydata_log = np.log10(ydata)
        # Fit linear
        popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
        #print(popt_log, pcov_log)
        # Apply fscaley^-1 to fitted data
        ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
        # There is no need to apply fscalex^-1 as original data is already available
        return (popt_log, pcov_log, ydatafit_log)

            
   # Apply fit on specified range #
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
        # s = np.min([s,e])
        # e = np.max([s,e])
        if s>e:
            s,e = e,s
        else:
            pass

        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = curve_fit_log(x[s:e],y[s:e])
            #print(fit)

            return fit, s, e, False
        else:
            return [np.nan, np.nan, np.nan],np.nan,np.nan, True




def find_fit_expo(x, y, x0, xf):  
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>-0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
        
        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = curve_fit_log_expo(x[s:e],y[s:e])
            #print(fit)
            return fit, s, e, x, y
        else:
            return [0],0,0,0,[0]

def curve_fit_log_expo(xdata, ydata) : 
    """Fit data to an exponential law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    #ydatafit_log = np.power(10, linlaw(xdata, *popt_log)
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log)



def histogram(quant, bins2, logx):

    """
    Function to create a histogram of a given data set.

    Parameters
    ----------
    quant: numpy array
        The data set to create the histogram from
    bins2: int
        The number of bins to use in the histogram
    logx: boolean
        Indicates whether the x-axis should be in log scale

    Returns
    -------
    nout: list
        The frequency counts of the data in each bin
    bout: list
        The center of each bin
    errout: list
        The error on the frequency count in each bin

    """
    nout = []
    bout = []
    errout=[]
    if logx == True:
        binsa = np.logspace(np.log10(min(quant)),np.log10(max(quant)),bins2)
    else:
        binsa = np.linspace((min(quant)),(max(quant)),bins2)

    histoout,binsout = np.histogram(quant,binsa,density=True)
    erroutt = histoout/np.float64(np.size(quant))
    erroutt = np.sqrt(erroutt*(1.0-erroutt)/np.float64(np.size(quant)))
    erroutt[: np.size(erroutt)] = erroutt[: np.size(erroutt)] / (
        binsout[1 : np.size(binsout)] - binsout[: np.size(binsout) - 1]
    )

    bin_centersout   = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])

    for k in range(len(bin_centersout)):
        if (histoout[k]!=0.):
            nout.append(histoout[k])
            bout.append(bin_centersout[k])
            errout.append(erroutt[k])
    return nout, bout,errout


import numpy as np

def scotts_rule_PDF(x):
    """
    Estimate bin edges using Scott's rule for histogram binning.

    Scott’s rule minimizes the integrated mean squared error in the bin approximation
    under the assumption that the data is approximately Gaussian, increasing the number
    of bins for smaller scales.

    Parameters
    ----------
    x : array_like
        Input data array.

    Returns
    -------
    array
        An array containing the estimated bin edges.

    Notes
    -----
    This function estimates the bin edges for histogram binning using Scott's rule, which
    is based on the standard deviation of the data. The number of bins is adjusted for
    smaller scales, assuming the data follows a Gaussian distribution.
    """
    x = np.real(x)
    N = len(x)
    sigma = np.nanstd(x)

    # Scott's rule for bin width
    dui = 3.5 * sigma / N ** (1 / 3)

    # create bins
    return np.arange(np.nanmin(x), np.nanmax(x), dui)




def pdf(val,
        bins, 
        loglog      = False,
        density     = False, 
        scott_rule  = False):
    """
    Calculate the Probability Density Function (PDF) from a given dataset.

    Parameters
    ----------
    val : array_like
        Input data array.
    bins : int or array_like
        Number of bins or bin edges for the histogram.
    loglog : bool, optional
        If True, use logarithmic bins. Default is False.
    density : bool, optional
        If True, compute a probability density function. Default is False.
    scott_rule : bool, optional
        If True, use Scott's rule for estimating the number of bins. Default is False.

    Returns
    -------
    tuple
        A tuple containing the bin centers, PDF values, PDF errors, and raw counts.

    Notes
    -----
    This function calculates the Probability Density Function (PDF) from a given dataset using a histogram.
    The bins for the histogram can be specified as an integer (number of bins) or an array (bin edges).
    The function supports logarithmic bins if `loglog` is set to True.
    If `density` is True, the function computes a normalized probability density function.
    If `scott_rule` is True, Scott's rule is used for estimating the number of bins.

    """
    nout = []
    bout = []
    errout = []
    countsout = []

    val = np.array(val)
    val = val[np.abs(val) < 1e15]

    if loglog:
        binsa = np.logspace(np.log10(min(val)), np.log10(max(val)), bins)
    else:
        if scott_rule:
            binsa = scotts_rule_PDF(val)
        else:
            binsa = np.linspace(min(val), max(val), bins)

    if density:
        numout, binsout, patchesout = plt.hist(val, density=True, bins=binsa, alpha=0)
    else:
        numout, binsout, patchesout = plt.hist(val, density=False, bins=binsa, alpha=0)

    counts, _, _ = plt.hist(val, density=False, bins=binsa, alpha=0)

    if loglog:
        bin_centers = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])
    else:
        bin_centers = binsout[:-1] + 0.5 * (binsout[1:] - binsout[:-1])

    if density:
        histoout, edgeout = np.histogram(val, binsa, density=True)
    else:
        histoout, edgeout = np.histogram(val, binsa, density=False)

    erroutt = histoout / np.float64(np.size(val))
    erroutt = np.sqrt(erroutt * (1.0 - erroutt) / np.float64(np.size(val)))
    erroutt[: np.size(erroutt)] = erroutt[: np.size(erroutt)] / (
        edgeout[1 : np.size(edgeout)] - edgeout[: np.size(edgeout) - 1]
    )

    for i in range(len(numout)):
        if numout[i] != 0.0:
            nout.append(numout[i])
            bout.append(bin_centers[i])
            errout.append(erroutt[i])
            countsout.append(counts[i])

    return np.array(bout), np.array(nout), np.array(errout), np.array(countsout)


def moving_average(xvals, yvals, window_size):
    """
    Calculate the moving average of the data.

    Parameters
    ----------
    xvals : array_like
        Input array representing the independent variable (x).
    yvals : array_like
        Input array representing the dependent variable (y).
    window_size : int
        Size of the moving average window.

    Returns
    -------
    tuple
        A tuple containing the smoothed `xvals` and the corresponding smoothed `yvals`.

    Notes
    -----
    This function calculates the moving average of the data using a specified window size.
    The `xvals` and `yvals` inputs are sorted based on `xvals` before calculating the moving average.

    """
    # Turn input into np.arrays
    xvals, yvals = np.array(xvals), np.array(yvals)

    # Now sort them
    index = np.argsort(xvals).astype(int)
    xvals = xvals[index]
    yvals = yvals[index]

    window = np.ones(int(window_size)) / float(window_size)
    y_new = np.convolve(yvals, window, 'same')
    return xvals, y_new





def plot_plaw(start, end, exponent, c):
    """
    Calculate points on a power-law line within a specified range.

    Parameters
    ----------
    start : float
        Starting value for the x range.
    end : float
        Ending value for the x range.
    exponent : float
        Exponent of the power-law function.
    c : float
        Scaling constant for the power-law function.

    Returns
    -------
    tuple
        A tuple containing the `x` values and the corresponding `y` values representing the points on the power-law line.

    Notes
    -----
    This function calculates the points on a power-law line given by the equation f(x) = c * x ** exponent.
    The points are calculated within the specified range from `start` to `end`.
    The function returns the `x` values and the corresponding `y` values representing the points on the power-law line.

    """
    # Calculating the points on the line
    x = np.logspace(np.log10(start), np.log10(end), 10000)
    
    # Power-law function f(x) = c * x ** exponent
    f = lambda x: c * x ** exponent
    
    return x, f(x)


import numpy as np
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D


class LineAnnotation(Annotation):
    def __init__(
        self, text, line, x, xytext=(0, 5), textcoords="offset points", font_size=None, **kwargs
    ):
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])

        self.neighbours = n1, n2 = neighbours(x, xs, ys)

        # Calculate y by interpolating neighboring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            "fontsize": font_size,  # Set the font size using the font_size parameter
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


# def line_annotate(text, line, x, font_size=None, *args, **kwargs):
#     ax = line.axes
#     a = LineAnnotation(text, line, x, font_size=font_size, *args, **kwargs)
#     if "clip_on" in kwargs:
#         a.set_clip_path(ax.patch)
#     ax.add_artist(a)
#     return a
def line_annotate(ax, text, line, x, font_size=None, *args, **kwargs):
    a = LineAnnotation(text, line, x, font_size=font_size, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a




@jit(nopython=True, parallel=True)
def smoothing_function(x, y, mean=True, window=2):
    """
    Optimized smoothing function for time series data.
    [Description same as before...]
    """
    
    def optimized_bisection(array, value):
        """
        Optimized bisection search function.
        [Description same as before...]
        """
        n = len(array)
        if value < array[0]:
            return -1
        elif value > array[n-1]:
            return n
        jl, ju = 0, n-1
        while ju - jl > 1:
            jm = (ju + jl) >> 1
            if value >= array[jm]:
                jl = jm
            else:
                ju = jm
        return jl if value != array[n-1] else n-1

    len_x = len(x)
    max_x = np.max(x)
    xoutmid,  yout = np.full(len_x, np.nan), np.full(len_x, np.nan)

    for i in prange(len_x):
        x0 = x[i]
        xf = window * x0

        if xf < max_x:
            e = optimized_bisection(x, xf)
            if e < len_x:
                x_range = x[i:e]
                y_range = y[i:e]
                if mean:
                    yout[i] = np.nanmean(y_range)
                else:
                    yout[i] = np.nanmedian(y_range)
                xoutmid[i] = x0 + np.log10(0.4) * (x0 - x[e])
                #xoutmid[i] = np.nanmedian(x_range)
               

    return xoutmid, yout


def calculate_parker_spiral(B):
    """
    This function estimates φ_{rB} = arctan(Bt / Br) for arrays of Br and Bt.
    It also returns the rolling mean of the computed angles over a 24-hour window.
    
    Parameters:
    B : pd.DataFrame
        A DataFrame where:
        - B.iloc[:, 0] corresponds to Br (Radial component of the magnetic field).
        - B.iloc[:, 1] corresponds to Bt (Tangential component of the magnetic field).
        - The index is the datetime for each measurement.
    
    Returns:
    pd.DataFrame:
        A DataFrame with the rolling mean of φ_{rB} (in degrees) over 24-hour windows.
    """
    # Extract Br and Bt using positional indexing (0 for Br, 1 for Bt)
    Br = B.iloc[:, 0].to_numpy()
    Bt = B.iloc[:, 1].to_numpy()
    
    # Calculate the Parker Spiral angle (phi_rB) in degrees using arctan2
    phi_rB = np.degrees(np.arctan2(Bt, Br))
    
    # Create a DataFrame for phi_rB with datetime index
    df_phi = pd.DataFrame({'phi_rB': phi_rB}, index=B.index)
    
    # Apply the rolling mean over a 24-hour window
    df_phi_rolling = df_phi.rolling('24H', center=True).mean()
    
    return df_phi_rolling


def interp(df, new_index):
    """
    Interpolate a DataFrame's columns values to new index values and return a new DataFrame.

    This function takes a DataFrame `df` and a new index `new_index`. It performs linear interpolation on each column
    of the DataFrame to calculate the corresponding values at the new index points. The resulting interpolated values
    are returned as a new DataFrame.

    Parameters:
    ----------
    df : pandas DataFrame
        The input DataFrame to be interpolated. The DataFrame should have a valid index.
    new_index : array-like
        The new index values to which the columns of `df` will be interpolated.

    Returns:
    -------
    pandas DataFrame
        A new DataFrame containing the interpolated values of the columns from `df` at the new index points.

    Notes:
    -----
    The function uses NumPy's `np.interp` function to perform linear interpolation for each column of the DataFrame.
    If the new index values lie outside the range of the original DataFrame's index, the function will extrapolate
    based on the closest available values.
    
    """
    
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out




# def simple_python_rolling_median(vector: np.ndarray,
#                                  window_length: int) -> np.ndarray:
#     """Computes a rolling median of a numpy vector returning a new numpy
#     vector of the same length.
#     NaNs in the input are not handled but a ValueError will be raised."""
#     if vector.ndim != 1:
#         raise ValueError(
#             f'vector must be one dimensional not shape {vector.shape}'
#         )
#     skip_list = orderedstructs.SkipList(float)
#     ret = np.empty_like(vector)
#     for i in range(len(vector)):
#         value = vector[i]
#         skip_list.insert(value)
#         if i >= window_length - 1:
#             # // 4 for lower quartile
#             # * 3 // 4 for upper quartile etc.
#             median = skip_list.at(window_length // 2)
#             skip_list.remove(vector[i - window_length + 1])
#         else:
#             median = np.nan
#         ret[i] = median
#     return ret




# def  use_dates_return_elements_of_df_inbetween(t0, t1, df):
#     """
#     Return the rows of df between the nearest indices to t0 and t1 using iloc.

#     Parameters:
#     -----------
#     t0 : datetime-like or str
#         Start date (if str, converted to datetime).
#     t1 : datetime-like or str
#         End date (if str, converted to datetime).
#     df : pd.DataFrame
#         DataFrame with a sorted datetime-like index.

#     Returns:
#     --------
#     pd.DataFrame
#         A DataFrame slice from the nearest index to t0 up to the nearest index to t1.
#     """
#     df = df.sort_index()

#     # Convert to datetime if necessary
#     if isinstance(t0, str):
#         t0 = pd.to_datetime(t0)
#     if isinstance(t1, str):
#         t1 = pd.to_datetime(t1)

#     # Find nearest indices
#     unique_idx = df.index.unique()
#     start_idx = unique_idx.get_indexer([t0], method="nearest")[0]
#     end_idx = unique_idx.get_indexer([t1], method="nearest")[0]

#     # Slice using iloc
#     return df.iloc[start_idx:end_idx]



def use_dates_return_elements_of_df_inbetween(start_date, end_date, df):
    """
    Returns a subset of the DataFrame `df` between `start_date` and `end_date` (inclusive).

    Parameters:
    -----------
    start_date : pd.Timestamp
        Lower bound for filtering.
    end_date   : pd.Timestamp
        Upper bound for filtering.
    df         : pd.DataFrame
        The DataFrame to filter. Its index must be datetime-like.

    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame containing rows where the index is between
        `start_date` and `end_date`.
    """

    df = df.sort_index()

    # Convert to datetime if necessary
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    return df.loc[(df.index >= start_date) & (df.index <= end_date)]




# def find_big_gaps(df, gap_time_threshold):
#     """
#     Filter a data set by the values of its first column and identify gaps in time that are greater than a specified threshold.

#     Parameters:
#     df (pandas DataFrame): The data set to be filtered and analyzed.
#     gap_time_threshold (float): The threshold for identifying gaps in time, in seconds.

#     Returns:
#     big_gaps (pandas Series): The time differences between consecutive records in df that are greater than gap_time_threshold.
#     """
#     keys = df.keys()

#     filtered_data = df[df[keys[1]] > -1e10]
#     time_diff     = (filtered_data.index.to_series().diff() / np.timedelta64(1, 's'))
#     big_gaps      = time_diff[time_diff > gap_time_threshold]

#     return big_gaps


def find_big_gaps(
    df, 
    gap_time_threshold = 10.0, 
    expected_start     = None, 
    expected_end       = None
):
    """
    Identifies "big gaps" where:
      1) The gap between consecutive *filtered* data points exceeds `gap_time_threshold`.
      2) The gap from an `expected_start` (if given) to the first valid row 
         exceeds `gap_time_threshold`.
      3) The gap from the last valid row to an `expected_end` (if given) 
         exceeds `gap_time_threshold`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DateTimeIndex and at least one column.
        Rows whose first column <= -1e10 will be excluded from gap checks.
    gap_time_threshold : float, optional
        The gap size threshold in seconds. Default=60.0 seconds.
    expected_start : None or str or pd.Timestamp, optional
        If provided (e.g. "2025-01-01 00:00:00"), we check if there's a large
        gap between this time and the first valid row.
    expected_end : None or str or pd.Timestamp, optional
        If provided, we check if there's a large gap between the last valid row 
        and this time.

    Returns
    -------
    gaps_df : pandas.DataFrame
        A DataFrame with columns ["Start", "End"] listing each detected gap.
    """
    # Convert expected_start / expected_end to Timestamps if needed
    if expected_start is not None and not isinstance(expected_start, pd.Timestamp):
        expected_start = pd.Timestamp(expected_start)
    if expected_end is not None and not isinstance(expected_end, pd.Timestamp):
        expected_end = pd.Timestamp(expected_end)

    # Ensure the DataFrame is sorted by its DateTimeIndex
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Filter out rows where the first column <= -1e10 (and drop NaNs)
    filtered_df = df[df.iloc[:, 0] > -1e10].dropna(subset=[df.columns[0]])
    gap_rows = []

    # CASE 1: No valid data after filtering
    if filtered_df.empty:
        if (expected_start is not None) and (expected_end is not None):
            diff_sec = (expected_end - expected_start).total_seconds()
            if diff_sec > gap_time_threshold:
                gap_rows.append({"Start": expected_start, "End": expected_end})
        return pd.DataFrame(gap_rows, columns=["Start", "End"])

    # CASE 2: Check gap from expected_start to first valid row
    first_time = filtered_df.index[0]
    if expected_start is not None:
        start_diff_sec = (first_time - expected_start).total_seconds()
        if start_diff_sec > gap_time_threshold:
            gap_rows.append({"Start": expected_start, "End": first_time})

    # If only one valid row, we only check the gap to expected_end
    if len(filtered_df) == 1:
        if expected_end is not None:
            end_diff_sec = (expected_end - first_time).total_seconds()
            if end_diff_sec > gap_time_threshold:
                gap_rows.append({"Start": first_time, "End": expected_end})
        return pd.DataFrame(gap_rows, columns=["Start", "End"])

    # CASE 3: Check consecutive row gaps in the filtered data
    time_diffs = filtered_df.index.to_series().diff().dt.total_seconds()
    gap_mask = time_diffs > gap_time_threshold
    gap_indices = np.where(gap_mask.values)[0]
    f_idx = filtered_df.index

    for i in gap_indices:
        gap_rows.append({"Start": f_idx[i - 1], "End": f_idx[i]})

    # CASE 4: Check gap from last valid row to expected_end
    last_time = filtered_df.index[-1]
    if expected_end is not None:
        end_diff_sec = (expected_end - last_time).total_seconds()
        if end_diff_sec > gap_time_threshold:
            gap_rows.append({"Start": last_time, "End": expected_end})

    return pd.DataFrame(gap_rows, columns=["Start", "End"])


# def find_big_gaps(df, gap_time_threshold=10):
#     """
#     Identifies gaps where the time difference between consecutive filtered entries
#     exceeds the gap_time_threshold, in a vectorized manner.
    
#     Parameters:
#     - df: pandas DataFrame with a datetime index and at least one column.
#     - gap_time_threshold: float, the gap size threshold in seconds.
    
#     Returns:
#     - A DataFrame with the start and end times of the gaps.
#     """
#     # Filter rows based on the condition for the first column
#     filtered_df = df[df.iloc[:, 0] > -1e10]
    
#     # Calculate time differences in seconds between consecutive rows
#     time_diffs = filtered_df.index.to_series().diff().dt.total_seconds()
    
#     # Identify indices where time differences exceed the threshold
#     gap_mask = time_diffs > gap_time_threshold
    
#     # Using the mask, find the end times of the gaps
#     gap_ends = filtered_df.index[gap_mask]
    
#     # The start times are just before the ends, adjust indices accordingly
#     gap_starts = filtered_df.index[gap_mask.shift(-1, fill_value=False)]
    
#     # Remove the last element from starts and the first element from ends to align
#     if len(gap_starts) > 0 and len(gap_ends) > 0:  # Ensure there are gaps
#         gap_starts = gap_starts[:-1]
#         gap_ends = gap_ends[1:]
    
#     # Create a DataFrame to return the start and end times of gaps
#     gaps_df = pd.DataFrame({'Start': gap_starts, 'End': gap_ends})
    
#     return gaps_df

def percentile(y,percentile):
    return(np.percentile(y,percentile))


import numpy as np


def binned_statistics_exclude(x, values, bins, statistic='mean', N=2, log_binning=True, n_jobs=1):
    """
    Compute binned statistics with exclusion of outliers greater than N standard deviations from the bin-specific mean.

    Parameters:
    - x : (N,) array_like
        Input values to be binned.
    - values : (N,) array_like
        Data values to compute the statistics on.
    - bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins. If bins is a sequence, it defines the bin edges.
    - statistic : string in ['mean', 'sum', 'std', 'count'] or callable
        The statistic to compute (default is 'mean').
    - N : float
        Number of standard deviations to exclude values from the mean within each bin.
    - log_binning : bool
        If True, use logarithmic bins.
    - n_jobs : int, default=1
        Number of CPU cores to use when parallelizing. Use -1 for all cores.
    
    Returns:
    - result : (nbins,) array
        The computed statistic for each bin.
    """
    
    
    from joblib import Parallel, delayed

    def compute_bin_statistic(bin_values, statistic, N):
        # Function to compute statistics for a single bin
        mean_val = np.mean(bin_values)
        std_val  = np.std(bin_values)

        # Exclude values more than N std dev from the bin-specific mean
        mask_std = np.abs(bin_values - mean_val) <= N * std_val
        bin_values = bin_values[mask_std]

        if statistic == 'mean':
            return np.nanmean(bin_values)
        elif statistic == 'median':
            return np.nanmedian(bin_values)
        elif statistic == 'sum':
            return np.nansum(bin_values)
        elif statistic == 'std':
            return np.nanstd(bin_values)
        elif statistic == 'count':
            return len(bin_values)
        elif callable(statistic):
            return statistic(bin_values)
        else:
            return np.nan

    # Remove nan and inf values
    mask_valid = np.isfinite(x) & np.isfinite(values)
    x          = x[mask_valid]
    values     = values[mask_valid]
    
    # Determine bins 
    if log_binning:
        if isinstance(bins, int):
            bin_edges = np.logspace(np.log10(min(x)), np.log10(max(x)), bins+1)
        else:
            bin_edges = np.logspace(np.log10(min(bins)), np.log10(max(bins)), len(bins))
    else:
        if isinstance(bins, int):
            bin_edges = np.linspace(min(x), max(x), bins+1)
        else:
            bin_edges = bins
        
    bin_indices = np.digitize(x, bin_edges)

    # Compute statistic for each bin using parallel processing
    results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], statistic, N) 
                                      for i in range(1, len(bin_edges)))
    
    std_results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], 'std', N) 
                                      for i in range(1, len(bin_edges)))
    count_results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], 'count', N) 
                                      for i in range(1, len(bin_edges)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return bin_centers, np.array(results), np.array(std_results)/np.sqrt(count_results)

# Example usage


def binned_statistics_high_percentile(x, values, bins, statistic='mean', percentile=50, log_binning=True, n_jobs=1):
    """
    Compute binned statistics considering only values above the given percentile for each bin.

    Parameters:
    - x : (N,) array_like
        Input values to be binned.
    - values : (N,) array_like
        Data values to compute the statistics on.
    - bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins. If bins is a sequence, it defines the bin edges.
    - statistic : string in ['mean', 'sum', 'std', 'count'] or callable
        The statistic to compute (default is 'mean').
    - percentile : float, default=50
        Percentile below which data will be excluded from each bin. 
        Values should be between 0 and 100.
    - log_binning : bool
        If True, use logarithmic bins.
    - n_jobs : int, default=1
        Number of CPU cores to use when parallelizing. Use -1 for all cores.
    
    Returns:
    - bin_centers : (nbins,) array
        The center of each bin.
    - result : (nbins,) array
        The computed statistic for each bin.
    """
    
    from joblib import Parallel, delayed

    def compute_bin_statistic(bin_values, statistic, percentile):
        # Check if bin_values is empty
        if len(bin_values) == 0:
            return np.nan

        # Filter values that are below the provided percentile
        threshold = np.percentile(bin_values, percentile)
        bin_values = bin_values[bin_values >= threshold]

        if statistic == 'mean':
            return np.mean(bin_values)
        elif statistic == 'sum':
            return np.sum(bin_values)
        elif statistic == 'std':
            return np.std(bin_values)
        elif statistic == 'count':
            return len(bin_values)
        elif callable(statistic):
            return statistic(bin_values)
        else:
            return np.nan


    # Remove nan and inf values
    mask_valid = np.isfinite(x) & np.isfinite(values)
    x          = x[mask_valid]
    values     = values[mask_valid]
    
    # Determine bins 
    if log_binning:
        if isinstance(bins, int):
            bin_edges = np.logspace(np.log10(min(x)), np.log10(max(x)), bins+1)
        else:
            bin_edges = np.logspace(np.log10(min(bins)), np.log10(max(bins)), len(bins))
    else:
        if isinstance(bins, int):
            bin_edges = np.linspace(min(x), max(x), bins+1)
        else:
            bin_edges = bins
        
    bin_indices = np.digitize(x, bin_edges)

    # Compute statistic for each bin using parallel processing
    results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], statistic, percentile) 
                                      for i in range(1, len(bin_edges)))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return bin_centers, np.array(results)



import numpy as np
from scipy import stats

def binned_quantity_percentile(x, y, what, std_or_error_of_mean, bins, loglog, percentile):
    # Ensure x and y are float type arrays
    x = x.astype(float)
    y = y.astype(float)
    
    # Filter out invalid y values
    ind = y > -1e15
    x = x[ind]
    y = y[ind]
    
    # Define bins in logarithmic scale if specified
    if loglog:
        bins = np.logspace(np.log10(min(x)), np.log10(max(x)), bins)
    
    # Calculate the binned percentiles
    percentiles, x_b, binnumber = stats.binned_statistic(x, y, lambda y: np.percentile(y, percentile), bins=bins)
    
    # Initialize arrays to store binned quantities for values above the percentile
    y_b = np.zeros(len(bins) - 1)
    z_b = np.zeros(len(bins) - 1)
    points = np.zeros(len(bins) - 1)
    
    # Iterate through each bin and calculate the statistics for values above the percentile
    for i in range(len(bins) - 1):
        bin_indices = np.where((x >= bins[i]) & (x < bins[i+1]))[0]
        if len(bin_indices) > 0:
            bin_y = y[bin_indices]
            bin_percentile_value = np.percentile(bin_y, percentile)
            bin_y_above_percentile = bin_y[bin_y >= bin_percentile_value]
            
            if len(bin_y_above_percentile) > 0:
                if what == 'mean':
                    y_b[i] = np.nanmean(bin_y_above_percentile)
                elif what == 'median':
                    y_b[i] = np.nanmedian(bin_y_above_percentile)
                elif what == 'sum':
                    y_b[i] = np.nansum(bin_y_above_percentile)
                elif what == 'std':
                    y_b[i] = np.nanstd(bin_y_above_percentile)
                elif what == 'var':
                    y_b[i] = np.nanvar(bin_y_above_percentile)
                
                z_b[i] = np.nanstd(bin_y_above_percentile)
                points[i] = len(bin_y_above_percentile)
    
    # Calculate the standard error of the mean if specified
    if std_or_error_of_mean == 0:
        z_b = z_b / np.sqrt(points)
    
    # Calculate the bin centers
    x_b = x_b[:-1] + 0.5 * (x_b[1:] - x_b[:-1])
    
    return x_b, y_b, z_b, percentiles


def ensure_time_format(start_time, end_time):
    
    """
    Ensure that the input start and end times are in the desired format and return them as formatted strings.

    This function takes `start_time` and `end_time` as inputs. It ensures that both `start_time` and `end_time`
    are in the desired format "%Y-%m-%d %H:%M:%S" and returns them as formatted strings.

    Parameters:
    ----------
    start_time : str or datetime-like object
        The start time of the desired time period. If provided as a datetime-like object, it will be converted
        to a string in the format "%Y-%m-%d %H:%M:%S".
    end_time : str or datetime-like object
        The end time of the desired time period. If provided as a datetime-like object, it will be converted
        to a string in the format "%Y-%m-%d %H:%M:%S".

    Returns:
    -------
    tuple of str
        A tuple containing two elements:
        1. The formatted start time in the format "%Y-%m-%d %H:%M:%S".
        2. The formatted end time in the format "%Y-%m-%d %H:%M:%S".

    Notes:
    -----
    The function uses the `datetime` module to handle datetime-like objects. If the input times are not provided
    as strings, the function converts them to the desired format. If the time is provided without a specific time
    (only date), the function appends "00:00:00" to the time before converting it to the desired format.

    """

    desired_format = "%Y-%m-%d %H:%M:%S"
    if not isinstance(start_time, str):
        start_time = datetime.strftime(start_time, desired_format)
    if not isinstance(end_time, str):
        end_time = datetime.strftime(end_time, desired_format)
    
    try:
        t0 = datetime.strptime(start_time, desired_format)
    except ValueError:
        t0 = datetime.strptime(start_time + " 00:00:00", desired_format)
    
    try:
        t1 = datetime.strptime(end_time, desired_format)
    except ValueError:
        t1 = datetime.strptime(end_time + " 00:00:00", desired_format)
        
    return t0.strftime(desired_format), t1.strftime(desired_format)





def binned_quantity(
    x, y,
    what                             = "mean",
    std_or_error_of_mean: bool | int = True,
    bins: int | np.ndarray = 100,
    loglog: bool = True,
    return_counts: bool = False,
    return_percentiles: bool = False,
    lower_percentile: float = 25,
    higher_percentile: float = 75,
    low_per: float | None = None,
    high_perc: float | None = None,
    *,                       # NEW: keyword-only opt-out flags
    return_edges: bool = False,
):
    """
    Identical call signature; two improvements:

    • For log-bins the centre x_b is now the geometric mean √(x_L x_R).
    • Set return_edges=True to also receive (x_lo, x_hi).
    """
    # 1 · sanitise
    x, y = map(np.asarray, (x, y))
    mask = (x > 0) if loglog else np.ones_like(x, bool)
    mask &= np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # 2 · bin edges
    if np.ndim(bins) == 0 or isinstance(bins, (int, np.integer)):
        bins = (np.logspace if loglog else np.linspace)(
            np.log10(x.min()) if loglog else x.min(),
            np.log10(x.max()) if loglog else x.max(),
            int(bins),
        )
        # if loglog:
        #     bins = 10 ** bins
    bins = np.asarray(bins, float)
    n_bins = bins.size - 1
    x_lo, x_hi = bins[:-1], bins[1:]
    x_ctr = np.sqrt(x_lo * x_hi) if loglog else 0.5 * (x_lo + x_hi)  # FIX 1

    # 3 · fast grouping
    idx = np.digitize(x, bins) - 1
    keep = (idx >= 0) & (idx < n_bins)
    idx, y = idx[keep], y[keep]

    count = np.bincount(idx, minlength=n_bins).astype(float)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    mean  = sum_y / np.where(count, count, np.nan)

    # robust variance
    sum_y2 = np.bincount(idx, weights=y ** 2, minlength=n_bins)
    var = (sum_y2 / np.where(count, count, np.nan)) - mean ** 2
    var[var < 0] = 0.0
    std = np.sqrt(var)

    # optional: per-bin custom statistic
    if callable(what):
        y_stat = np.array([what(y[idx == i]) if c else np.nan
                           for i, c in enumerate(count)])
    else:
        mapping = {"mean": mean, "std": std,
                   "median": np.array([np.nan if c == 0 else
                                       np.median(y[idx == i])
                                       for i, c in enumerate(count)]),
                   "count": count}
        y_stat = mapping.get(what, mean)

    z_stat = std if std_or_error_of_mean else std / np.sqrt(count)

    out = (x_ctr, y_stat, z_stat)
    if return_counts:
        out += (count,)
    if return_percentiles:
        pct_lo  = np.full_like(mean, np.nan)
        pct_hi  = np.full_like(mean, np.nan)
        for i, c in enumerate(count):
            if c:
                yy = y[idx == i]
                pct_lo[i], pct_hi[i] = np.percentile(yy,
                                                     [lower_percentile,
                                                      higher_percentile])
        out += ((pct_lo, pct_hi),)
    if return_edges:                         # NEW
        out = (x_lo, x_hi) + out
    return out


# def binned_quantity(x, y, what='mean', std_or_error_of_mean=True, bins=100, loglog=True, return_counts=False, return_percentiles=False, lower_percentile =25, higher_percentile = 75):
#     """
#     Vectorised alternative to `binned_quantity` that avoids multiple passes
#     through the data.  Median and percentile estimates are ∼10‑100× faster
#     (depending on sample size) because the data are sorted only once.
#     """
#     # 1. sanitise & mask
#     x, y = np.asarray(x), np.asarray(y)
#     if loglog:
#         mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
#     else:
#         mask = np.isfinite(x) & np.isfinite(y)
#     x, y = x[mask].astype(float), y[mask].astype(float)

#     # 2. bin edges
#     if np.ndim(bins)==0 or isinstance(bins, (int, np.integer)):
#         if loglog:
#             bins = np.logspace(np.log10(x.min()), np.log10(x.max()), int(bins))
#         else:
#             bins = np.linspace(x.min(), x.max(), int(bins))
#     bins = np.asarray(bins, dtype=float)
#     n_bins = bins.size - 1

#     # 3. assign each point to a bin (−1 for out‑of‑range)
#     bin_idx = np.digitize(x, bins) - 1
#     valid = (bin_idx >= 0) & (bin_idx < n_bins)
#     bin_idx = bin_idx[valid]
#     x, y = x[valid], y[valid]

#     # 4. fast aggregated sums
#     count = np.bincount(bin_idx, minlength=n_bins).astype(float)
#     sum_y  = np.bincount(bin_idx, weights=y, minlength=n_bins)
#     sum_y2 = np.bincount(bin_idx, weights=y*y, minlength=n_bins)

#     with np.errstate(invalid='ignore', divide='ignore'):
#         mean = sum_y / count
#         var  = (sum_y2 / count) - mean**2
#         std  = np.sqrt(var)

#     # 5. optional median/percentiles (single pass sorting)
#     y_median = None
#     pct_low = pct_high = None
#     if (what == 'median') or return_percentiles:
#         order = np.argsort(bin_idx, kind='mergesort')  # stable
#         sorted_bins = bin_idx[order]
#         sorted_y    = y[order]
#         # cumulative counts to split
#         split_idx = np.cumsum(np.bincount(sorted_bins, minlength=n_bins))[:-1]
#         groups = np.split(sorted_y, split_idx)
#         # list comprehension is fine: n_bins is small (∼10‑100)
#         if what == 'median':
#             y_median = np.array([np.median(g) if g.size else np.nan for g in groups])
#         if return_percentiles:
#             pct_low  = np.array([np.percentile(g, lower_percentile) if g.size else np.nan for g in groups])
#             pct_high = np.array([np.percentile(g, higher_percentile) if g.size else np.nan for g in groups])

#     # 6. choose requested statistic
#     stats_map = {
#         'mean': mean,
#         'std': std,
#         'count': count,
#         'median': y_median,
#     }
#     if callable(what):
#         y_b = np.full(n_bins, np.nan)
#         for idx, grp in enumerate(np.split(sorted_y, split_idx) if (what!='mean' and what!='median') else []):
#             y_b[idx] = what(grp) if grp.size else np.nan
#     else:
#         y_b = stats_map.get(what, mean)  # default mean

#     # 7. error of mean or std
#     z_b = std.copy()
#     if std_or_error_of_mean == 0:
#         with np.errstate(divide='ignore', invalid='ignore'):
#             z_b = std/np.sqrt(count)

#     # 8. bin centres
#     x_b = 0.5*(bins[1:] + bins[:-1])

#     # 9. assemble output
#     out = (x_b, y_b, z_b)
#     if return_counts:
#         out += (count,)
#     if return_percentiles:
#         out += ((pct_low, pct_high),)
#     return out

# import numpy as np
# from scipy import stats

# def binned_quantity(x, y, what='mean', std_or_error_of_mean=True, bins=100, loglog=True, return_counts=False, return_percentiles=False, lower_percentile =25, higher_percentile = 75):
#     """
#     Calculate binned statistics of one variable (y) with respect to another variable (x).

#     Parameters
#     ----------
#     x : array_like
#         Input array. This represents the independent variable.
#     y : array_like
#         Input array. This represents the dependent variable.
#     what : str or callable, optional
#         The type of binned statistic to compute. This can be any of the options supported by `scipy.stats.binned_statistic()`.
#         The default is 'mean'.
#     std_or_error_of_mean : bool, optional
#         Indicates whether to return the standard deviation (True) or the error of the mean (False) of the binned statistic.
#         The default is True.
#     bins : int or array_like, optional
#         The number of bins to use for the histogram. If `loglog` is True, this value is used to generate logarithmic bins.
#         The default is 100.
#     loglog : bool, optional
#         If True, logarithmic bins are used instead of linear bins. The default is True.
#     return_counts : bool, optional
#         If True, also return the number of points in each bin. The default is False.
#     return_percentiles : bool, optional
#         If True, also return the 25th and 75th percentiles for each bin. The default is False.

#     Returns
#     -------
#     x_b : ndarray
#         The centers of the bins.
#     y_b : ndarray
#         The value of the binned statistic.
#     z_b : ndarray
#         The standard deviation or error of the mean of the binned statistic.
#     points : ndarray, optional
#         The number of points in each bin. This is only returned if `return_counts` is True.
#     percentiles : tuple of ndarrays, optional
#         The 25th and 75th percentiles for each bin. This is only returned if `return_percentiles` is True.
#     """
    
#     if loglog:
#         mask = np.where((y > -1e10) & (x > 0) )[0]        
#     else:
#         mask = np.where((y > -1e10) & (x > -1e10) )[0]
#     x = np.asarray(x[mask], dtype=float)
#     y = np.asarray(y[mask], dtype=float)

#     if loglog:
#         bins = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), bins)

#     # Binned statistic calculation
#     y_b, x_b, _ = stats.binned_statistic(x, y, statistic=what, bins=bins)
#     z_b, _, _ = stats.binned_statistic(x, y, statistic='std', bins=bins)
#     points, _, _ = stats.binned_statistic(x, y, statistic='count', bins=bins)

#     if std_or_error_of_mean == 0:
#         z_b /= np.sqrt(points)

#     x_b = x_b[:-1] + 0.5 * (x_b[1:] - x_b[:-1])

#     result = (x_b, y_b, z_b, points) if return_counts else (x_b, y_b, z_b)

#     if return_percentiles:
#         percentile_25, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, lower_percentile), bins=bins)
#         percentile_75, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, higher_percentile), bins=bins)
#         percentiles = (percentile_25, percentile_75)
#         result += (percentiles,)

#     return result




# from concurrent.futures import ThreadPoolExecutor  # Ensure this import is present
# import numpy as np
# from scipy import stats

# def binned_quantity(x, y, what='mean', std_or_error_of_mean=True, bins=100, 
#                     loglog=True, return_counts=False, return_percentiles=False, 
#                     lower_percentile=25, higher_percentile=75):
#     """
#     Optimized version that computes bin centers using vectorized operations.
#     """
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
#     if loglog:
#         mask = (x > 0) & (y > -1e10)
#     else:
#         mask = (x > -1e10) & (y > -1e10)
#     x = x[mask]
#     y = y[mask]
    
#     if np.isscalar(bins):
#         nbins = int(bins)
#         if loglog:
#             bin_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
#         else:
#             bin_edges = np.linspace(x.min(), x.max(), nbins + 1)
#     else:
#         bin_edges = np.asarray(bins, dtype=float)
#         nbins = len(bin_edges) - 1
        
#     # Compute bin centers
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
#     # Determine bin indices (vectorized)
#     bin_indices = np.digitize(x, bin_edges) - 1
#     valid = (bin_indices >= 0) & (bin_indices < nbins)
#     x = x[valid]
#     y = y[valid]
#     bin_indices = bin_indices[valid]
    
#     counts = np.bincount(bin_indices, minlength=nbins)
    
#     # Initialize arrays with float type explicitly
#     stat_vals = np.full(nbins, np.nan, dtype=float)
#     err_vals  = np.full(nbins, np.nan, dtype=float)
    
#     if isinstance(what, str) and what in ['mean', 'sum']:
#         if what == 'mean':
#             sum_y = np.bincount(bin_indices, weights=y, minlength=nbins)
#             stat_vals = np.divide(
#                 sum_y, 
#                 counts, 
#                 out=np.full(sum_y.shape, np.nan, dtype=float), 
#                 where=counts > 0
#             )
#         elif what == 'sum':
#             stat_vals = np.bincount(bin_indices, weights=y, minlength=nbins)
#         sum_y2 = np.bincount(bin_indices, weights=y*y, minlength=nbins)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mean_y = stat_vals
#             std = np.sqrt(np.divide(
#                 sum_y2, 
#                 counts, 
#                 out=np.full(sum_y2.shape, np.nan, dtype=float), 
#                 where=counts > 0
#             ) - mean_y**2)
#         err_vals = std.copy()
#     else:
#         order = np.argsort(bin_indices)
#         sorted_bins = bin_indices[order]
#         sorted_y = y[order]
#         bin_start = np.searchsorted(sorted_bins, np.arange(nbins))
#         bin_end   = np.searchsorted(sorted_bins, np.arange(nbins + 1))
        
#         def compute_stat(i):
#             if bin_end[i] > bin_start[i]:
#                 y_slice = sorted_y[bin_start[i]:bin_end[i]]
#                 if what == 'median':
#                     return np.median(y_slice)
#                 else:
#                     return what(y_slice)
#             else:
#                 return np.nan
        
#         with ThreadPoolExecutor() as executor:
#             stat_list = list(executor.map(compute_stat, range(nbins)))
#         stat_vals = np.array(stat_list, dtype=float)
        
#         def compute_std(i):
#             if bin_end[i] > bin_start[i]:
#                 y_slice = sorted_y[bin_start[i]:bin_end[i]]
#                 return np.std(y_slice, ddof=1) if len(y_slice) > 1 else 0.0
#             else:
#                 return np.nan
        
#         with ThreadPoolExecutor() as executor:
#             std_list = list(executor.map(compute_std, range(nbins)))
#         err_vals = np.array(std_list, dtype=float)
    
#     if std_or_error_of_mean == False:
#         with np.errstate(divide='ignore', invalid='ignore'):
#             err_vals = np.divide(
#                 err_vals, 
#                 np.sqrt(counts), 
#                 out=np.full(err_vals.shape, np.nan, dtype=float), 
#                 where=counts > 0
#             )
    
#     percentiles = None
#     if return_percentiles:
#         percentile_25, _, _ = stats.binned_statistic(
#             x, y, 
#             statistic=lambda arr: np.percentile(arr, lower_percentile), 
#             bins=bin_edges
#         )
#         percentile_75, _, _ = stats.binned_statistic(
#             x, y, 
#             statistic=lambda arr: np.percentile(arr, higher_percentile), 
#             bins=bin_edges
#         )
#         percentiles = (percentile_25, percentile_75)
    
#     result = (bin_centers, stat_vals, err_vals)
#     if return_counts:
#         result += (counts,)
#     if return_percentiles:
#         result += (percentiles,)
    
#     return result




def find_fit_semilogy(x, y, x0, xf): 
    def line(x, a, b):
        return a*x+b
    # Apply fit on specified range #
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]

        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = fun.curve_fit(line, x[s:e],np.log10(y[s:e]))
            y = 10**line(x[s:e], fit[0][0], fit[0][1]) 
            return fit, s, e, x[s:e], y
        else:
            return [0],0,0,0,[0]
        
        
        
# import numpy as np
# from scipy.optimize import minimize

# def three_plaw_fit(x: np.ndarray, y: np.ndarray, num_segments: int = 3, max_iter: int = 10000,
#                    middle_weight: float = 1.0, initial_breakpoints: np.ndarray = None,
#                    breakpoint_bounds: list = None):
#     """
#     Optimize the breakpoints for a piecewise power-law fit with continuity constraints,
#     and weight the middle segment more heavily in the optimization.

#     Parameters:
#     -----------
#     x : np.ndarray
#         Independent variable data.
#     y : np.ndarray
#         Dependent variable data.
#     num_segments : int, optional
#         Number of segments for the piecewise power-law fit. Default is 3.
#     max_iter : int, optional
#         Maximum number of iterations for the optimizer. Default is 10000.
#     middle_weight : float, optional
#         Weight applied to the residuals of the middle segment. Default is 1.0.
#     initial_breakpoints : np.ndarray, optional
#         Initial guesses for the breakpoint x-values. Should be of length num_segments - 1.
#     breakpoint_bounds : list of tuples, optional
#         Bounds for the breakpoint x-values. Should be a list of tuples with length num_segments - 1.

#     Returns:
#     --------
#     fits_dict : dict
#         Dictionary containing the fit results for each segment.
#     """
#     # Ensure x and y are sorted by x
#     sort_idx = np.argsort(x)
#     x = x[sort_idx]
#     y = y[sort_idx]

#     # Remove any NaN or infinite values
#     finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
#     x = x[finite_mask]
#     y = y[finite_mask]

#     n = len(x)
#     logx = np.log(x)
#     logy = np.log(y)

#     # Initial guess for breakpoints (x-values)
#     if initial_breakpoints is None:
#         initial_breakpoints = np.linspace(
#             x[0],
#             x[-1],
#             num_segments + 1
#         )[1:-1]  # Exclude the first and last point

#     # Ensure initial breakpoints satisfy constraints
#     initial_breakpoints = np.array(initial_breakpoints)
#     initial_breakpoints += np.arange(num_segments - 1) * 1e-5

#     # Initial guesses for slopes and intercepts
#     initial_a = np.full(num_segments, -1.0)  # Initial slope guesses
#     initial_logc = np.full(num_segments, np.mean(logy))  # Initial intercept guesses

#     # Combine all variables into a single array
#     x0 = np.concatenate([initial_breakpoints, initial_a, initial_logc])

#     # Define bounds for variables
#     if breakpoint_bounds is None:
#         breakpoint_bounds = [(x[1], x[-2])] * (num_segments - 1)
#     else:
#         breakpoint_bounds = [(max(b[0], x[1]), min(b[1], x[-2])) for b in breakpoint_bounds]

#     bounds = breakpoint_bounds  # Bounds for breakpoints (x-values)
#     bounds += [(-np.inf, np.inf)] * (2 * num_segments)  # Bounds for slopes and intercepts

#     # Constraints: Ordering of breakpoints
#     constraints = []
#     for i in range(num_segments - 2):
#         def breakpoint_order_constraint(x_vars, i=i):
#             return x_vars[i + 1] - x_vars[i] - 1e-5
#         constraints.append({
#             'type': 'ineq',
#             'fun': breakpoint_order_constraint
#         })

#     # Continuity constraints at breakpoints
#     for i in range(num_segments - 1):
#         def continuity_constraint(x_vars, i=i):
#             # Breakpoint x-value
#             x_b = x_vars[i]
#             if x_b <= x[0] or x_b >= x[-1]:
#                 return 0  # Return zero to avoid errors

#             # Slopes and intercepts
#             a_i = x_vars[num_segments - 1 + i]
#             logc_i = x_vars[2 * num_segments - 1 + i]
#             a_next = x_vars[num_segments - 1 + i + 1]
#             logc_next = x_vars[2 * num_segments - 1 + i + 1]

#             # Continuity equation
#             y_i = logc_i + a_i * np.log(x_b)
#             y_next = logc_next + a_next * np.log(x_b)
#             return y_i - y_next  # Should be zero for continuity

#         constraints.append({
#             'type': 'eq',
#             'fun': continuity_constraint
#         })

#     # Objective function with weighted middle segment
#     def objective(x_vars):
#         # Extract variables
#         breakpoints = x_vars[:num_segments - 1]
#         a_i = x_vars[num_segments - 1:2 * num_segments - 1]
#         logc_i = x_vars[2 * num_segments - 1:]

#         # Clip and sort breakpoints
#         breakpoints = np.clip(breakpoints, x[1], x[-2])
#         breakpoints = np.sort(breakpoints)

#         # Determine the indices where the breakpoints occur
#         indices = np.searchsorted(x, breakpoints)
#         start_idx = np.concatenate(([0], indices))
#         end_idx = np.concatenate((indices, [n]))

#         residuals = []
#         for i in range(num_segments):
#             idx = slice(start_idx[i], end_idx[i])
#             x_seg = logx[idx]
#             y_seg = logy[idx]

#             y_fit = logc_i[i] + a_i[i] * x_seg
#             res = y_seg - y_fit

#             # Apply weighting to the middle segment
#             if num_segments == 3 and i == 1:
#                 weight = middle_weight
#             else:
#                 weight = 1.0

#             residuals.extend(weight * res)

#         residuals = np.array(residuals)
#         return np.sum(residuals ** 2)

#     # Minimize the total residuals with tighter tolerances
#     res = minimize(
#         objective,
#         x0,
#         method='SLSQP',
#         bounds=bounds,
#         constraints=constraints,
#         options={
#             'maxiter': int(max_iter),
#             'disp': True,
#             'ftol': 1e-12,   # Decrease function tolerance
#             'eps': 1e-12     # Decrease step size for numerical gradient
#         }
#     )

#     if not res.success:
#         print("Optimizer did not converge:", res.message)

#     # Extract optimized variables
#     x_vars = res.x
#     breakpoints = x_vars[:num_segments - 1]
#     a_i = x_vars[num_segments - 1:2 * num_segments - 1]
#     logc_i = x_vars[2 * num_segments - 1:]

#     # Clip and sort breakpoints
#     breakpoints = np.clip(breakpoints, x[1], x[-2])
#     breakpoints = np.sort(breakpoints)

#     # Determine the indices where the breakpoints occur
#     indices = np.searchsorted(x, breakpoints)
#     start_idx = np.concatenate(([0], indices))
#     end_idx = np.concatenate((indices, [n]))

#     # Get the x-values of the breakpoints
#     breakpoints_values = breakpoints  # These are already x-values

#     # Build fits_dict
#     fits_dict = {}
#     segment_labels = ['p{}'.format(i + 1) for i in range(num_segments)]
#     for i, label in enumerate(segment_labels):
#         idx_range = slice(start_idx[i], end_idx[i])
#         x_seg = x[idx_range]
#         y_seg = y[idx_range]

#         # Compute predicted y values
#         y_fit = np.exp(logc_i[i] + a_i[i] * np.log(x_seg))

#         # Compute residuals
#         residuals = np.log(y_seg) - (logc_i[i] + a_i[i] * np.log(x_seg))
#         residual_sum = np.sum(residuals ** 2)

#         # Compute standard error of the slope
#         n_seg = len(x_seg)
#         if n_seg > 2:
#             s_squared = residual_sum / (n_seg - 2)
#             Sxx = np.sum((np.log(x_seg) - np.mean(np.log(x_seg))) ** 2)
#             if Sxx > 0:
#                 s_a = np.sqrt(s_squared / Sxx)
#             else:
#                 s_a = np.nan
#         else:
#             s_a = np.nan

#         fits_dict[label] = {
#             'plaw-index': a_i[i],
#             'plaw-index-err': s_a,
#             'err': residual_sum,
#             'xv': x_seg,
#             'yv': y_fit,
#             'x_break': breakpoints_values[i] if i < num_segments - 1 else np.nan,
#             'n_iter': res.nit
#         }

#     return fits_dict



import numpy as np
from scipy.ndimage import gaussian_filter1d

def local_slope(x, y, bin_size=1, smoothing_sigma=3, return_max_diff_points=False):
    """
    Compute the local slope of y with respect to x in log-log space, and estimate y at midpoints.

    Parameters:
    -----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    bin_size : int, optional
        The number of data points to include in each bin. Default is 1 (no binning).
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel used in smoothing. Default is 3.
    return_max_diff_points : bool, optional
        If True, the function returns the x-values where the absolute differences
        of the slopes are maximum.

    Returns:
    --------
    midpoints : np.ndarray
        The x-values at the middle of the bins where the slopes are estimated.
    slopes_smooth : np.ndarray
        The smoothed local slopes computed in log-log space.
    y_midpoints : np.ndarray
        The y-values estimated at the midpoints.
    max_diff_points : np.ndarray (optional)
        The x-values where the absolute differences of the slopes are maximum.
        Only returned if `return_max_diff_points` is True.
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Ensure x and y are sorted by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove any NaN or infinite values and non-positive values
    finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[finite_mask]
    y = y[finite_mask]

    # Log-transform x and y
    logx = np.log(x)
    logy = np.log(y)

    # Binning
    if bin_size > 1:
        num_complete_bins = len(logx) // bin_size
        logx_binned = np.array([
            np.mean(logx[i * bin_size:(i + 1) * bin_size]) for i in range(num_complete_bins)
        ])
        logy_binned = np.array([
            np.mean(logy[i * bin_size:(i + 1) * bin_size]) for i in range(num_complete_bins)
        ])
    else:
        logx_binned = logx
        logy_binned = logy

    # Compute the differences in log-log space
    dlogx = np.diff(logx_binned)
    dlogy = np.diff(logy_binned)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.true_divide(dlogy, dlogx)
        slopes[~np.isfinite(slopes)] = 0  # Replace infinities and NaNs with zero

    # Compute midpoints of x and y in log space
    logx_mid = (logx_binned[:-1] + logx_binned[1:]) / 2
    midpoints = np.exp(logx_mid)

    logy_mid = (logy_binned[:-1] + logy_binned[1:]) / 2
    y_midpoints = np.exp(logy_mid)

    # Smooth the slopes to reduce noise
    slopes_smooth = gaussian_filter1d(slopes, sigma=smoothing_sigma)

    if return_max_diff_points:
        # Compute differences of slopes
        slope_diffs = np.diff(slopes_smooth)
        # Find indices where the absolute differences are maximum
        max_diff_indices = np.where(np.abs(slope_diffs) == np.max(np.abs(slope_diffs)))[0]
        # Corresponding x-values (midpoints between midpoints)
        x_max_diff = (midpoints[max_diff_indices] + midpoints[max_diff_indices + 1]) / 2
        return midpoints, slopes_smooth, y_midpoints, x_max_diff
    else:
        return midpoints, slopes_smooth, y_midpoints




from scipy.optimize import differential_evolution
import numpy as np

def three_plaw_fit(x: np.ndarray, y: np.ndarray, num_breaks: int = 2):
    """
    Fit a piecewise power-law (3 segments) to the data (x, y) by optimizing the breakpoints.
    Parameters:
    -----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    num_breaks : int, optional
        Number of breakpoints (Default is 2, resulting in 3 segments)
    Returns:
    --------
    fits_dict : dict
        Dictionary containing the fit results for each segment, including standard errors.
    """

    # Ensure x and y are sorted by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove any NaN or infinite values
    finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[finite_mask]
    y = y[finite_mask]

    logx = np.log(x)
    logy = np.log(y)

    # Define the objective function
    def objective(breakpoints):
        # Ensure breakpoints are sorted and within x range
        breakpoints = np.sort(breakpoints)
        if np.any(breakpoints <= x[0]) or np.any(breakpoints >= x[-1]):
            return np.inf  # Penalty for invalid breakpoints

        # Split data into segments
        residuals = []
        previous_idx = 0

        for bp in breakpoints:
            idx = np.searchsorted(x, bp, side='right')
            x_seg = logx[previous_idx:idx]
            y_seg = logy[previous_idx:idx]

            # Linear regression in log-log space
            if len(x_seg) > 1:
                A = np.vstack([x_seg, np.ones(len(x_seg))]).T
                slope, intercept = np.linalg.lstsq(A, y_seg, rcond=None)[0]
                y_fit = intercept + slope * x_seg
                residuals.extend(y_seg - y_fit)
            else:
                return np.inf  # Penalty for too few points in segment

            previous_idx = idx

        # Last segment
        x_seg = logx[previous_idx:]
        y_seg = logy[previous_idx:]
        if len(x_seg) > 1:
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            slope, intercept = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            y_fit = intercept + slope * x_seg
            residuals.extend(y_seg - y_fit)
        else:
            return np.inf  # Penalty for too few points in segment

        residuals = np.array(residuals)
        return np.sum(residuals ** 2)

    # Define bounds for breakpoints
    bounds = [(x[1], x[-2])] * num_breaks  # Avoid the very first and last points

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=False
    )

    if not result.success:
        print("Optimization failed.")
        return None

    # Get the best breakpoints
    best_breakpoints = np.sort(result.x)

    # Now, compute the final fit parameters
    fits_dict = {}
    segment_labels = ['p{}'.format(i + 1) for i in range(num_breaks + 1)]
    previous_idx = 0
    residuals_total = []

    for i, bp in enumerate(np.append(best_breakpoints, x[-1])):
        idx = np.searchsorted(x, bp, side='right')
        x_seg = x[previous_idx:idx]
        y_seg = y[previous_idx:idx]
        logx_seg = logx[previous_idx:idx]
        logy_seg = logy[previous_idx:idx]

        if len(x_seg) > 1:
            A = np.vstack([logx_seg, np.ones(len(logx_seg))]).T
            # Solve for parameters
            beta = np.linalg.lstsq(A, logy_seg, rcond=None)[0]
            slope, intercept = beta
            y_fit = np.exp(intercept + slope * logx_seg)
            residuals = logy_seg - (intercept + slope * logx_seg)
            residual_sum = np.sum(residuals ** 2)

            # Calculate standard errors
            n = len(logx_seg)
            p = 2  # Number of parameters (slope and intercept)
            dof = n - p  # Degrees of freedom
            if dof > 0:
                sigma_squared = np.sum(residuals ** 2) / dof
                # Compute covariance matrix
                cov_beta = sigma_squared * np.linalg.inv(np.dot(A.T, A))
                # Standard errors are square roots of diagonal elements
                standard_errors = np.sqrt(np.diag(cov_beta))
                slope_error, intercept_error = standard_errors
            else:
                slope_error = np.nan
                intercept_error = np.nan
        else:
            slope = np.nan
            intercept = np.nan
            slope_error = np.nan
            intercept_error = np.nan
            y_fit = np.full_like(y_seg, np.nan)
            residual_sum = np.nan

        fits_dict[segment_labels[i]] = {
            'plaw-index': slope,
            'plaw-index-error': slope_error,
            'plaw-intercept': intercept,
            'plaw-intercept-error': intercept_error,
            'err': residual_sum,
            'xv': x_seg,
            'yv': y_fit,
            'x_break': bp if i < num_breaks else np.nan
        }

        residuals_total.extend(residuals)
        previous_idx = idx

    fits_dict['breakpoints'] = best_breakpoints
    fits_dict['total_error'] = np.sum(np.array(residuals_total) ** 2)

    return fits_dict

import numpy as np
from joblib import Parallel, delayed

def mov_fit_func_joblib(xx,
                        yy,
                        w_size,
                        xmin,
                        xmax,
                        keep_plot=0,
                        pad=1,
                        n_jobs=-1):
    """
    Perform moving fits on the data within a specified range.
    Optimized with Joblib for parallel processing.
    
    Parameters
    ----------
    xx : ndarray
        Input array representing the independent variable (x).
    yy : ndarray
        Input array representing the dependent variable (y).
    w_size : float
        Window size used to perform the fits.
    xmin : float
        Minimum value of x for the fitting range.
    xmax : float
        Maximum value of x for the fitting range.
    keep_plot : bool
        If True, additional data for plotting fits is returned.
    pad : int
        Step size to reduce the number of points for fitting.
    n_jobs : int
        Number of parallel jobs to run (-1 uses all available CPUs).
    
    Returns
    -------
    dict
        A dictionary containing information about the fits.
    """
    
    # Convert inputs to arrays and filter based on valid ranges
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    
    mask = (xx > -1e10) & (yy > -1e10)
    xx, yy = xx[mask], yy[mask]

    # Find indices in the range of interest
    index1 = np.searchsorted(xx, xmin, side='left')
    index2 = np.searchsorted(xx, xmax, side='right') - 1
    where_fit = np.arange(index1, index2 + 1, step=int(pad))  # Skip with stride of `pad`

    # Function to perform fit (to be run in parallel)
    def perform_fit(i):
        x0 = xx[i]
        xf = x0 * w_size

        if xf < 0.95 * xmax:
            fit, s, e, x1, y1 = find_fit(xx, yy, x0, xf)
            if len(np.shape(x1)) > 0:
                err = np.sqrt(fit[1][1][1])
                ind = fit[0][1]
                x_val = x1[s]

                result = {
                    'err': err,
                    'ind': ind,
                    'x_val': x_val,
                }

                if keep_plot:
                    result['plot_x'] = x1[s:e]
                    result['plot_y'] = 2 * fit[2]

                return result
        return None

    # Run fits in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(perform_fit)(i) for i in where_fit
    )

    # Extract valid results
    keep_err, keep_ind, keep_x = [], [], []
    xvals, yvals = [], []

    for result in results:
        if result is not None:
            keep_err.append(result['err'])
            keep_ind.append(result['ind'])
            keep_x.append(result['x_val'])
            if keep_plot:
                xvals.append(result['plot_x'])
                yvals.append(result['plot_y'])

    # Prepare the result dictionary
    result_dict = {
        'xvals': np.array(keep_x),
        'plaw': np.array(keep_ind),
        'fit_err': np.array(keep_err),
    }

    if keep_plot:
        result_dict['plot_x'] = xvals
        result_dict['plot_y'] = yvals

    return result_dict


import numpy as np
from scipy.optimize import curve_fit



def moving_fit(x, 
               y,
               fwin,
               df, 
               make_df_adapt_2_scale = True,
               df_multiplier         = 0.005
              ):
    """
    Process data by fitting curves within specified window and step sizes.

    Parameters:
    x (np.ndarray): Array of x values.
    y (np.ndarray): Array of y values.
    fwin (float): Window size for the logarithmic fitting.
    df (float): Step size for shifting the window in logarithmic scale.

    Returns:
    list: List of fit values.
    list: List of x midpoints of each fitting window.
    """
    # Delete NaNs
    ind = np.isnan(y)
    x = x[np.invert(ind)]
    y = y[np.invert(ind)]

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)


    x1 = xmin
    x2 = fwin*xmin
    
    if make_df_adapt_2_scale:
        df = x1*df_multiplier
        print('Using adaptive df. Init Value:',df)
    
    
    fit_vals  = []
    xmids     = []

    while x2 < xmax:
        try:
            fit, _, _, flag = curve_fit_log_wrap(x, y, x1, x2)

            fit_vals.append(fit[0][1])   
            xmids.append(x1)
                         

        except:
            fit_vals.append(np.nan)
            xmids.append(x1)
                            
        x1 = x1 + df
        x2 = fwin*x1
        
        if make_df_adapt_2_scale:
            df = x1*df_multiplier
        

    return  xmids, fit_vals

def freq2wavenum(freq, P, Vtot, di):
    """ Takes the frequency, the PSD, the SW velocity and the di.
        Gives the k* and the E(k*), normalised with di"""
    
    # xvals          =  xvals/Vtotal*(2*np.pi*di)
    # yvals          =  yvals*(2*np.pi*di)/Vtotal

    
    k_star = freq/Vtot*(2*np.pi*di)
    
    eps_of_k_star = P*Vtot/(2*np.pi*di)
    
    return k_star, eps_of_k_star

def freq2wavenum_only_kdi(freq, Vtot, di):
    """ Takes the frequency, the PSD, the SW velocity and the di.
        Gives the k* and the E(k*), normalised with di"""
    
    # xvals          =  xvals/Vtotal*(2*np.pi*di)
    # yvals          =  yvals*(2*np.pi*di)/Vtotal

    
    k_star = freq/Vtot*(2*np.pi*di)
    
    
    return k_star

import numpy as np

def freq2wavenum_only_kdi_arrays(freq, Vtot, di):
    """
    Returns a 2D array k_star of shape (len(Vtot), len(freq)),
    i.e. (14401, 184).
    """
    freq = np.asarray(freq).reshape(1, -1)     # (1, 184)
    Vtot = np.asarray(Vtot).reshape(-1, 1)     # (14401, 1)
    di   = np.asarray(di).reshape(-1, 1)       # (14401, 1)

    # Elementwise: (1,184) / (14401,1) * 2*pi*(14401,1) 
    # => shape (14401, 184)
    k_star = freq / Vtot * (2 * np.pi * di)  

    return k_star.T



import numpy as np

def integrate_psd_in_k_range_2d(kdi_2d, psd_2d, kmin, kmax):
    """
    Integrate the PSD in each column of `psd_2d` over the k-range
    [kmin, kmax], using the corresponding k-values in `kdi_2d`.

    Parameters
    ----------
    kdi_2d : (M, N) array
        2D array of wavenumbers. Each column can have its own k-values.
    psd_2d : (M, N) array
        2D array of PSD values matching the shape of kdi_2d.
    kmin : float
        Lower limit of k-range to integrate over.
    kmax : float
        Upper limit of k-range to integrate over.

    Returns
    -------
    integrated : (N,) ndarray
        The integrated PSD for each of the N columns.
        If no valid data in a column, returns np.nan in that position.
    """
    kdi_2d = np.asarray(kdi_2d)
    psd_2d = np.asarray(psd_2d)
    M, N   = kdi_2d.shape

    result = np.full(N, np.nan)  # Default to NaN

    for i in range(N):
        kvals = kdi_2d[:, i]
        pvals = psd_2d[:, i]

        # 1) Remove NaNs
        valid_mask = ~np.isnan(kvals) & ~np.isnan(pvals)
        kvals = kvals[valid_mask]
        pvals = pvals[valid_mask]
        
        # 2) Restrict to k in [kmin, kmax]
        range_mask = (kvals >= kmin) & (kvals <= kmax)
        kvals = kvals[range_mask]
        pvals = pvals[range_mask]
        
        # If nothing remains, leave result[i] = np.nan
        if kvals.size == 0:
            continue

        # 3) Sort k in ascending order (so integration is positive if pvals >= 0)
        sort_inds = np.argsort(kvals)
        kvals = kvals[sort_inds]
        pvals = pvals[sort_inds]

        # 4) Integrate with trapezoidal rule
        result[i] = np.trapz(pvals, x=kvals)

    return result




import numpy as np
from scipy.interpolate import griddata

def smooth_2d_data(X, Y, Z, Ntimes):
    """
    Interpolate 2D data on a new grid.

    Parameters:
    X (2D array): X-coordinates of the data.
    Y (2D array): Y-coordinates of the data.
    Z (2D array): Values at each (X, Y) point.
    Ntimes (int): Factor to scale the new grid size.

    Returns:
    Xn, Yn (2D arrays): New meshgrid for X and Y.
    data1 (2D array): Interpolated data on the new grid.
    """

    # Calculate differences and midpoints
    X_diff = np.diff(X, axis=1)
    Y_diff = np.diff(Y.T, axis=1)
    X_mid = X[:, :-1] + X_diff / 2
    Y_mid = (Y.T)[:,:-1] + Y_diff / 2

    # Flatten the arrays
    x = X_mid[1:, :].flatten()
    y = (Y_mid[1:,:].T).flatten()
    z = Z.flatten()

    # Filter out NaN values
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]

    # Define the new grid for interpolation
    xnew = np.logspace(np.log10(np.nanmin(X)), np.log10(np.nanmax(X)), int(Ntimes*len(X[0])))
    ynew = np.logspace(np.log10(np.nanmin(Y)), np.log10(np.nanmax(Y)), int(Ntimes*len(Y[0])))

    # Create a meshgrid for the new grid
    Xn, Yn = np.meshgrid(xnew, ynew)

    # Perform the interpolation
    data1 = griddata((x_filtered, y_filtered), z_filtered, (Xn, Yn), method='linear')

    return Xn, Yn, data1

# Example usage
# Xn, Yn, interpolated_data = interpolate_2d_data(X, Y, Z, Ntimes)


def smooth(x, n=5):
    """
    Apply a running mean smoothing to the input signal.

    Parameters
    ----------
    x : ndarray
        The signal to be smoothed.
    n : int, optional
        Window width for the running mean. The default is 5.

    Returns
    -------
    ndarray
        The smoothed signal of the same length as *x*.

    Notes
    -----
    This function applies a running mean smoothing to the input signal using a convolution operation.
    The running mean is calculated using a window of width `n`, and the smoothed signal is returned.
    The convolution operation is performed in 'same' mode to ensure that the output has the same length as the input.

    """
    return np.convolve(x, np.ones(n) / n, mode='same')




def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


def resample_find_equal_elements(keep_unique, interpolate, xarr1, yarr1, xarr2, yarr2,choose_min_max, interp_method, npoints,parx_min, parx_max, perx_min, perx_max ):
    
    if interpolate:
        df_par = pd.DataFrame({'x': xarr1, 'y': yarr1}).set_index('x')
        df_per = pd.DataFrame({'x': xarr2, 'y':yarr2}).set_index('x')



        if choose_min_max:
            parx_min, parx_max = np.nanmin(df_par.index.values),np.nanmax(df_par.index.values)
            perx_min, perx_max = np.nanmin(df_per.index.values),np.nanmax(df_per.index.values)   

        new_index_par  = np.logspace(np.log10(parx_min), np.log10(parx_max), npoints)
        new_index_per  = np.logspace(np.log10(perx_min), np.log10(perx_max), npoints)

        # new_index_par  = np.linspace((parx_min), (parx_max), npoints)
        # new_index_per  = np.linspace((perx_min), (perx_max), npoints)

        df_par         =    newindex(df_par, new_index_par, interp_method)
        df_per         =    newindex(df_per, new_index_per, interp_method)
        # df_par         = func.interp(df_par, new_index_par)
        # df_per         = func.interp(df_per, new_index_per)


        x_para, y_para   = np.real(df_par.index.values),np.real( df_par.values.T[0])
        x_pera, y_pera = np.real(df_per.index.values),np.real( df_per.values.T[0])
    else:

        x_para, y_para   = xarr1, yarr1
        x_pera, y_pera   = xarr2, yarr2  
    
    
    res = closest_argmin(y_para, y_pera)


    xparnew, yparnew = x_para,y_para
    xpernew, ypernew = x_pera[res], y_pera[res]

    if keep_unique:
        unq, unq_inv, unq_cnt = np.unique(np.sort(res), return_inverse=True, return_counts=True)

        xparnew1, yparnew1  = xparnew[unq], yparnew[unq]
        xpernew1, ypernew1 = xpernew[unq], ypernew[unq]
    else:
        xparnew1, yparnew1  = xparnew, yparnew
        xpernew1, ypernew1 = xpernew, ypernew     
    
    index1 =  np.argsort(xpernew1)
    
    
    return  xparnew1[index1], yparnew1[index1], xpernew1[index1], ypernew1[index1]


def fit(x, y, deg=1, fullyes=False):
    """
    Fit function wrapper that calls `nupmpy.polyfit`. Returns the fit parameters as well as the standard deviation
    of the fit.

    Args:
        x: [ndarray] X-coordinates of the data points
        y: [ndarray] Y-coordinates of the data points (same shape as *x*)
        deg: [int] Degree of the polynomial
        fullyes: [boolean] Whether to return the full set of arguments from the fit function or not.
        Argument forwarded to `numpy.polyfit()`

    Returns:
        fitpars: [list] List of fit parameters containing at least the polynomial coefficients, and also residuals,
        rank, etc. See numpy.polyfit for full details.
        fitpars_std: [numpy.ndarray] The standard deviation of each fit parameter estimate.

    """

    try:
        if np.any(np.isnan(x + y)):
            raise ValueError('Input argument *x* or *y* contains NAN.')
    except ValueError as err:
        err_fitpars = env.ERRORVAL*np.ones(deg+1)
        err_cov = env.ERRORVAL*np.ones(deg+1)
        return err_fitpars, err_cov

    if fullyes:
        fitpars, cov = np.polyfit(x, y, deg=deg, cov=True)
        fitpars_std = np.sqrt(np.diag(cov))
    else:
        fitpars = np.polyfit(x, y, deg=deg)
        fitpars_std = env.ERRORVAL

    return fitpars, fitpars_std


def savepickle(df_2_save, save_path, filename):
    """
    Save a list of variables into a single pickle file.

    Parameters
    ----------
    df_2_save : object
        The data or variables to be saved in the pickle file.
    save_path : str
        The path to the folder where the file will be saved.
    filename : str
        The name of the file to save (including the extension).

    Returns
    -------
    None

    Notes
    -----
    This function creates the specified directory (`save_path`) if it doesn't exist and saves the data or variables (`df_2_save`)
    into a single pickle file with the provided filename.

    """
    
    # Ensure the directory exists
    os.makedirs(str(save_path), exist_ok=True)
    
    # Use the highest protocol available for more efficient serialization
    # Open the file using a context manager to ensure it's properly closed after writing
    file_path = Path(save_path).joinpath(filename)
    with open(file_path, 'wb') as file:
        pickle.dump(df_2_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        
      

# def savefeather(df, path_to_save, filename, include_index= True):
#     """
#     Saves a DataFrame to a Feather file.

#     Parameters:
#     - df: pandas.DataFrame, the DataFrame to save.
#     - path_to_save: str, the directory path where the Feather file will be saved.
#     - filename: str, the name of the Feather file to be saved.

#     Returns:
#     - None, saves the file to the specified path.
#     """
#     if include_index:
#         df.reset_index(inplace=True)
        
#     # Construct the full file path
#     full_file_path = f"{path_to_save}/{filename}"
    
#     # Save the DataFrame to a Feather file
#     df.to_feather(full_file_path)
    
def saveparquet(df, path_to_save, filename, column_names=None):
    """
    Saves a DataFrame to a Parquet file, with an option to save only specified columns.
    Checks if the save path exists, and creates it if it doesn't.

    Parameters:
    - df            : pandas.DataFrame, the DataFrame to save.
    - path_to_save  : str, the directory path where the Parquet file will be saved.
    - filename      : str, the name of the Parquet file to be saved.
    - column_names  : list (optional), a list of column names to save from the DataFrame. If None, all columns are saved.

    Returns:
    - None, saves the file to the specified path.
    """
    # Check if the path exists, create it if it doesn't
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)

    # Construct the full file path
    full_file_path = os.path.join(path_to_save, filename)
    
    # If column_names is specified, select only those columns
    if column_names is not None:
        df_to_save = df[column_names]
    else:
        df_to_save = df
    
    # Save the DataFrame (or the subset) to a Parquet file
    df_to_save.to_parquet(full_file_path)
    
# import pandas as pd

def load_parquet(path_to_save, filename= None, column_names= None, engine='pyarrow'):
    """
    Reads specific columns from a Parquet file using the specified engine.

    Parameters:
    - path_to_save: str, the directory path where the Parquet file is saved.
    - filename: str, the name of the Parquet file.
    - column_names: list, a list of column names to read from the Parquet file.
    - engine: str, the engine to use for reading the Parquet file ('pyarrow' or 'fastparquet').

    Returns:
    - A pandas DataFrame containing only the specified columns.
    """
    # Construct the full file path
    if filename== None:
        full_file_path = f"{path_to_save}"    
    else:
        full_file_path = f"{path_to_save}/{filename}"

    # Read specific columns from the Parquet file using the specified engine
    df = pd.read_parquet(full_file_path, columns=column_names, engine=engine)
    
    return df



# def load_parquet(path: str,
#                       *,
#                       columns: list[str] | None = None,
#                       memory_map: bool          = True,
#                       threads: bool             = True):
#     md  = pq.read_metadata(path).metadata
#     idx = []
#     if b"pandas" in md:
#         meta = json.loads(md[b"pandas"].decode())
#         idx_descr = meta.get("index_columns", [])
#         if idx_descr and isinstance(idx_descr[0], str):
#             idx = idx_descr

#     proj_cols = None
#     if columns is not None:
#         proj_cols = list(columns)
#         for c in idx:
#             if c not in proj_cols:
#                 proj_cols.append(c)

#     table = pq.read_table(
#         path,
#         columns=proj_cols,
#         memory_map=memory_map,
#         use_threads=threads
#     )

#     df = table.to_pandas(
#         types_mapper=pd.ArrowDtype,
#         use_threads=threads
#     )

#     if columns is not None:
#         df = df[columns]

#     return df

def replace_filename_extension(oldfilename, newextension, addon=False):
    """
    Replace the extension of the file name with *newextension*

    Args:
        oldfilename: [str] file name to be changed
        newextension: [str] the new extension
        addon: [boolean] whether or not to add on the new extension, or replace old extension with new

    Returns:
        newfilename: [str] filename with the new extension
    """

    # extension is the part after the last period in the filename
    dot_ix = oldfilename.rfind('.')

    # if oldfilename doesn't have extension, then just add the new extension
    if dot_ix == -1:
        addon = True

    # if desired, just add the new extension forming double extension file like `filename.old.new`
    if addon:
        dot_ix = len(oldfilename)

    return oldfilename[:dot_ix] + '.' + newextension.strip('.')




# def newindex(df, ix_new, interp_method='linear'):
#     """
#     Reindex a DataFrame according to the new index *ix_new* supplied, ensuring no duplicate labels in the index.

#     Args:
#         df: [pandas DataFrame] The dataframe to be reindexed.
#         ix_new: [np.array or pandas Index] The new index.
#         interp_method: [str] Interpolation method to be used; forwarded to `pandas.DataFrame.interpolate`.

#     Returns:
#         df_reindexed: [pandas DataFrame] DataFrame interpolated and reindexed to *ix_new*.
#     """
#     # Remove duplicate indices
#     df = df[~df.index.duplicated(keep='first')].sort_index()

#     # Remove duplicates in new index and sort
#     ix_new = np.unique(ix_new)

#     # Trim the new index to the overlapping range to ensure synchronization constraints
#     start, end = max(df.index.min(), ix_new.min()), min(df.index.max(), ix_new.max())
#     ix_new     = ix_new[(ix_new >= start) & (ix_new <= end)]

#     # Reindex and interpolate
#     df_reindexed = df.reindex(ix_new).interpolate(method=interp_method).dropna()

#     return df_reindexed

def newindex(df, ix_new, interp_method='linear'):
    """
    Reindex a DataFrame according to the new index *ix_new* supplied, ensuring no duplicate labels in the index.

    Args:
        df: [pandas DataFrame] The dataframe to be reindexed.
        ix_new: [np.array] The new index.
        interp_method: [str] Interpolation method to be used; forwarded to `pandas.DataFrame.reindex.interpolate`.

    Returns:
        df3: [pandas DataFrame] DataFrame interpolated and reindexed to *ix_new*.
    """
    # Ensure df.index and ix_new do not contain duplicates
    df     = df[~df.index.duplicated(keep='first')].interpolate().dropna()
    ix_new = np.unique(ix_new)

    # Verify that reindexing is necessary and feasible
    if not np.array_equal(df.index.sort_values(), ix_new.sort()):
        # Sort the DataFrame index in increasing order
        df = df.sort_index(ascending=True)

        # Create combined index from old and new index arrays, ensuring no duplicates
        ix_com = np.unique(np.concatenate([df.index.values, ix_new]))

        # Re-index and interpolate over the non-matching points
        df2 = df.reindex(ix_com).interpolate(method=interp_method)

        # Reindex to the new index, ix_new
        return df2.reindex(ix_new)
    else:
        # If the current index and new index are effectively the same, no reindexing is needed
        print("No reindexing necessary; DataFrame index matches the new index.")
        return df



def listsearch(search_string, input_list):
    """
    Return matching items from a list

    Args:
        search_string: [str] String to search for (starting only)
        input_list: [list] List to search in

    Returns:
        found_list: [list] List of matching items

    """

    return [si for si in input_list if si.startswith(search_string)]


def window_selector(N, win_name='Hanning'):
    """
    Simply a wrapper for *get_window* from *scipy.signal*. Return the window coefficients.

    Args:
        N: [int] Window length
        win_name: [str] Name of the window

    Returns:
        w: [ndarray] Window coefficients
    """
    import scipy.signal as signal

    return signal.windows.get_window(win_name, N)


def chunkify(ts_in, chunk_duration):
    """
    Divide a given timeseries in to chunks of *chunk_duration*

    Args:
        ts_in: [pd.Timeseries] Input timeseries
        chunk_duration: [float] Duration of the chunks in seconds

    Returns:

    """
    #print('converting to chunks of len %.2f sec' % chunk_duration)

    dchunk_str = f'{str(chunk_duration)}S'

    return pd.date_range(
        ts_in[0].ceil('1s'), ts_in[-1].floor('1s'), freq=dchunk_str
    )



def progress_bar(jj, length):
    """
    Display a progress bar showing the completion percentage.

    Parameters
    ----------
    jj : int
        The current progress value.
    length : int
        The total length or maximum value for the progress bar.

    Returns
    -------
    None

    Notes
    -----
    This function displays a simple progress bar indicating the percentage of completion for a task.

    """
    percentage = round(100 * (jj / length), 2)
    print('Completed', percentage)


# def find_ind_of_closest_dates(df, dates):
#     """
#     Find the indices of the closest dates in a DataFrame to a list of input dates.

#     Parameters
#     ----------
#     df : pandas DataFrame
#         Input DataFrame containing time series data with a unique index.
#     dates : list-like
#         List of input dates for which the closest indices need to be found.

#     Returns
#     -------
#     list
#         A list containing the indices of the closest dates in the DataFrame `df` to each element in the `dates` list.

#     Notes
#     -----
#     This function calculates the indices of the closest dates in the DataFrame `df` to each date in the input `dates`.
#     It uses the pandas DataFrame `index.unique().get_loc()` method with the 'nearest' method to find the indices.

#     """
#     return [df.index.unique().get_loc(date, method='nearest') for date in dates]





def find_ind_of_closest_dates(df, dates):
    """
    Find the indices of the closest dates in a DataFrame to a list of input dates in a vectorized manner.

    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame containing time series data with a DateTime index.
    dates : list-like
        List of dates (as pandas Timestamps or compatible types) for which the closest indices need to be found.

    Returns
    -------
    list
        A list containing the indices of the closest dates in the DataFrame `df` to each date in the `dates` list.
    """
    # Ensure the DataFrame index is in datetime64[ns] format
    df_timestamps = df.index.values.astype('datetime64[ns]')
    # Convert input dates to numpy array in datetime64[ns] format
    input_dates = np.array(pd.to_datetime(dates).values.astype('datetime64[ns]'))
    # Calculate the absolute differences between all dates
    abs_diff = np.abs(df_timestamps[:, np.newaxis] - input_dates)
    # Find the index of the minimum difference for each input date
    closest_indices = np.argmin(abs_diff, axis=0)
    return closest_indices.tolist()



def find_closest_values_of_2_arrays(a, b):
    """
    Find the closest values of two arrays and return their indices.

    Parameters
    ----------
    a : array_like
        The first input array.
    b : array_like
        The second input array.

    Returns
    -------
    ndarray
        An array containing pairs of indices where the values in arrays `a` and `b` are closest to each other.

    Notes
    -----
    This function finds the closest values of two arrays `a` and `b` and returns their corresponding indices.
    It searches for the closest values of `b` in `a`, and for each unique index in `a`, it finds the index in `b`
    where the values are closest.

    Example
    --------
    >>> import numpy as np

    >>> def find_closest_values_of_2_arrays(a, b):
    ...     # (Your function implementation here)

    >>> a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    >>> b = np.array([2.0, 4.0, 6.0, 8.0])
    >>> closest_indices = find_closest_values_of_2_arrays(a, b)
    >>> print(closest_indices)
    """
    dup = np.searchsorted(a, b)
    uni = np.unique(dup)
    uni = uni[uni < a.shape[0]]
    ret_b = np.zeros(uni.shape[0], dtype=int)
    for idx, val in enumerate(uni):
        bw = np.argmin(np.abs(a[val] - b[dup == val]))
        tt = dup == val
        ret_b[idx] = np.where(tt)[0][bw]
    return np.column_stack((uni, ret_b))


def find_cadence(df, mean_or_median_cadence='median'):
    """
    Find the cadence (time interval) between successive timestamps in a DataFrame's index.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame.
    mean_or_median_cadence : str, optional
        The type of cadence to compute. It can be 'Mean' or 'Median'. The default is 'Mean'.

    Returns
    -------
    float
        The mean or median cadence in seconds between successive timestamps in the DataFrame's index.

    Notes
    -----
    This function calculates the cadence (time interval) between successive timestamps in the DataFrame's index.
    It drops any rows with missing values and computes either the mean or median cadence based on the `mean_or_median_cadence` parameter.
    """
    keys = list(df.keys())
    if mean_or_median_cadence == 'mean':
        return np.nanmean((df[keys[0]].dropna().index.to_series().diff() / np.timedelta64(1, 's')))
    else:
        return np.nanmedian((df[keys[0]].dropna().index.to_series().diff() / np.timedelta64(1, 's')))





#def resample_timeseries_estimate_gaps(df, resolution, large_gaps=5)



# def resample_timeseries_estimate_gaps(
#     df,
#     resolution_ms=1000,
#     large_gaps=10.0,
#     aggregator="mean",
#     do_interpolation=True,
#     interpolation_method="time",
#     handle_infs_as_nans=True,
#     enforce_res_not_smaller=True,
#     gap_mode="median"
# ):
#     """
#     Resample a time series and estimate gaps, returning a dictionary with 
#     the same keys as originally specified.
#     """
#     # Prepare the return dictionary with default (in case of error).
#     results = {
#         "Init_dt"      : None,
#         "resampled_df" : None,
#         "Frac_miss"    : 100.0,  # 100% if something fails
#         "Large_gaps"   : None,
#         "Tot_gaps"     : None,
#         "resol"        : np.nan
#     }

#     try:
#         # 1) Ensure DataFrame is sorted by its DateTimeIndex
#         if not df.index.is_monotonic_increasing:
#             df = df.sort_index()

#         # 2) Optionally replace inf/-inf with NaN
#         if handle_infs_as_nans:
#             df = df.replace([np.inf, -np.inf], np.nan)

#         # 3) Compute original cadence (Init_dt) in seconds from consecutive diffs
#         time_diffs = df.index.to_series().diff().dt.total_seconds().dropna()
#         if len(time_diffs) < 1:
#             # if we cannot measure dt
#             init_dt = 0.0
#         else:
#             if gap_mode == "mean":
#                 init_dt = time_diffs.mean()
#             else:
#                 init_dt = time_diffs.median()
#         results["Init_dt"] = init_dt

#         # 4) Compute the total interval duration
#         if len(df.index) < 2:
#             interval_dur_s = 0.0
#         else:
#             interval_dur_s = (df.index[-1] - df.index[0]).total_seconds()

#         # 5) If there's a positive total duration, measure large & total gaps
#         if interval_dur_s > 0:
#             # Large gaps fraction
#             large_gap_mask = time_diffs > large_gaps
#             sum_large_gaps = time_diffs[large_gap_mask].sum()
#             total_large_gaps = 100.0 * sum_large_gaps / interval_dur_s
#             results["Large_gaps"] = total_large_gaps

#             # Possibly enforce final resolution not < init_dt
#             desired_res_s = resolution_ms / 1000.0
#             if enforce_res_not_smaller and init_dt > 0 and (desired_res_s < init_dt):
#                 desired_res_s = init_dt

#             final_res_ms = desired_res_s * 1000.0
#             results["resol"] = final_res_ms

#             # measure total gaps fraction above that final interval
#             tot_gap_mask = time_diffs > desired_res_s
#             sum_tot_gaps = time_diffs[tot_gap_mask].sum()
#             total_gaps = 100.0 * sum_tot_gaps / interval_dur_s
#             results["Tot_gaps"] = total_gaps

#             # 6) Resample with aggregator to get uniform time steps
#             resample_rule = f"{int(round(final_res_ms))}ms"
#             df_resampled_raw = getattr(df.resample(resample_rule), aggregator)()

#             # 7) OPTIONAL: measure fraction missing before interpolation
#             #    (If you only want the fraction in final, skip or keep for debugging)
#             n_vals_raw = df_resampled_raw.size
#             if n_vals_raw > 0:
#                 n_missing_raw = df_resampled_raw.isna().sum().sum()
#                 fraction_missing_raw = 100.0 * n_missing_raw / n_vals_raw
#             else:
#                 fraction_missing_raw = 0.0
#             # print("Fraction missing (pre‐interpolation):", fraction_missing_raw, "%")

#             # 8) Interpolate if requested
#             if do_interpolation:
#                 df_resampled_filled = df_resampled_raw.interpolate(method=interpolation_method)
#             else:
#                 df_resampled_filled = df_resampled_raw

#             # 9) Now measure final fraction of missing
#             n_vals_res = df_resampled_filled.size
#             if n_vals_res > 0:
#                 n_missing_res = df_resampled_filled.isna().sum().sum()
#                 fraction_missing = 100.0 * n_missing_res / n_vals_res
#             else:
#                 fraction_missing = 0.0

#             # 10) If you want absolutely no missing data in final (no NaNs),
#             #     you could do a second fill method:
#             #       df_resampled_filled = df_resampled_filled.fillna(method="ffill").fillna(method="bfill")
#             #
#             #     Or if there's an unbounded region, you might choose a constant fill:
#             #       df_resampled_filled = df_resampled_filled.fillna(0)

#             # Crucially: DO NOT dropna() if you want to keep a continuous time axis
#             #   Because dropna() would remove entire timestamps (rows), reintroducing time gaps

#             results["Frac_miss"]    = fraction_missing
#             results["resampled_df"] = df_resampled_filled

#         else:
#             # If there's no real duration, we can't measure these
#             results["Large_gaps"] = 0.0
#             results["Tot_gaps"]   = 0.0
#             results["Frac_miss"]  = 100.0
#             # We'll leave the rest as is
#     except Exception as e:
#         # If something goes wrong, results dict stays with safe defaults
#         print(f"ERROR in resample_timeseries_estimate_gaps: {e}")

#     return results



    
def resample_timeseries_estimate_gaps(df, resolution, large_gaps=5):
    """
    Resample a time series and estimate gaps.

    Parameters
    ----------
    df : pandas DataFrame
        Input time series data as a pandas DataFrame.
    resolution : int
        Resolution in milliseconds to resample the time series.
    large_gaps : int, optional
        Large gaps threshold in seconds. Gaps greater than this threshold are considered large.
        The default is 10.

    Returns
    -------
    dict
        A dictionary containing the following information:
        - 'Init_dt': Initial resolution of the input time series.
        - 'resampled_df': Resampled DataFrame with interpolated missing values.
        - 'Frac_miss': Fraction of missing values in the resampled interval.
        - 'Large_gaps': Fraction of large gaps (greater than `large_gaps` seconds) in the resampled interval.
        - 'Tot_gaps': Total fraction of gaps (greater than the resampled resolution) in the resampled interval.
        - 'resol': The actual resolution used for resampling.

    Notes
    -----
    This function resamples the input time series `df` to the specified `resolution` using the mean of data points within each resampled interval.
    If the initial resolution of `df` is greater than the desired `resolution`, the function increases the `resolution` slightly until it is lower than the initial resolution.
    The function estimates the fraction of missing values and gaps in the resampled data and provides the information in the returned dictionary.


    """
    try:
        keys    = list(df.keys())
        init_dt = find_cadence(df)


        # Estimate fraction of missing values within interval
        fraction_missing = 100 * len(df[(np.abs(df[keys[0]])>1e10) | (np.isnan(df[keys[0]])) |  (np.isinf(df[keys[0]]))  ])/ len(df)
        
        # Make sure that you resample to a resolution that is lower than the initial df's resolution
        while init_dt > resolution * 1e-3:
            resolution = 1.005 * resolution

        # Estimate duration of interval selected in seconds
        interval_dur = (df.index[-1] - df.index[0]).total_seconds()
        
        # Estimate sum of gaps greater than large_gaps seconds
        res = (df.dropna().index.to_series().diff() / np.timedelta64(1, 's'))

        # Gives you the fraction of large gaps in the time series
        total_large_gaps = 100 * (res[res > large_gaps].sum() / interval_dur)
        
        # Gives you the total fraction of gaps in the time series
        total_gaps = 100 * (res[res > resolution * 1e-3].sum() / interval_dur)

        # Resample time-series to desired resolution
        df_resampled = df.resample(f"{int(resolution)}ms").median().interpolate()


    except:
        init_dt           = None
        df_resampled      = None
        fraction_missing  = 100
        total_gaps        = None
        total_large_gaps  = None
        resolution        = np.nan

    return {
        "Init_dt"         : init_dt,
        "resampled_df"    : df_resampled,
        "Frac_miss"       : fraction_missing,
        "Large_gaps"      : total_large_gaps,
        "Tot_gaps"        : total_gaps,
        "resol"           : resolution
    }




