import numpy as np
import pandas as pd
import traceback
from astropy import units as u
import ssqueezepy
import scipy
import os
import matplotlib as mpl
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
from scipy.optimize import curve_fit, least_squares
import joblib
from joblib import Parallel, delayed
import statistics
from statistics import mode
import orderedstructs
import sys

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func




import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process_sc_pot(df, voltage_columns=None):
    """
    Combine or negate the voltage columns to yield a single 'v_sc' potential.
    """
    if voltage_columns is None:
        voltage_columns = [col for col in df.columns if 'V' in col]
    v_sc = -np.nanmean(df[voltage_columns], axis=1)
    return pd.DataFrame(v_sc, index=df.index, columns=['V'])

def calibrate_density(df_V, df_n, window_str='30s', overlap_ratio=0.9,
                       model_type='2param', n_jobs=-1,
                       lower_pct=0, upper_pct=99.9,
                       cut_off_freq=1e-2):
    """
    Revised calibration function that computes window-specific density estimates
    and blends them via a weighted (Hann) overlap-add. In this version, after
    synchronizing the high-resolution voltage (df_V) and the low-resolution density (df_n)
    data, a low-pass filter is applied to both time series using the specified cutoff
    frequency (default 1e-2 Hz). The filtered components are then used for the fitting.
    
    Parameters:
      df_V, df_n: DataFrames containing the high- and low-resolution data.
      window_str: Window duration as a string (e.g., '30s').
      overlap_ratio: Fractional overlap between windows.
      model_type: '2param' or '3param' calibration model.
      n_jobs: Number of parallel jobs.
      lower_pct, upper_pct: Percentile thresholds for filtering the final density.
      cut_off_freq: Cutoff frequency (in Hz) for the low-pass filter (default 1e-2 Hz).
    
    Returns:
      A pandas Series of the calibrated density estimates indexed by high-resolution timestamps.
    """
    # Step 1: Synchronize the high-cadence voltage (df_V) and density (df_n) data.
    df_V_low, df_n = func.synchronize_dfs(df_V, df_n, 0)
    
    # Compute sampling frequency from the synchronized low-resolution index.
    if len(df_n.index) > 1:
        dt = (df_n.index[1] - df_n.index[0]).total_seconds()
        fs = 1 / dt
    else:
        raise ValueError("df_n must contain at least two timestamps to compute sampling frequency.")
    
    # Step 2: Apply low-pass filter to both time series.
    # Create copies to hold filtered data.
    df_V_low_filtered = df_V_low.copy()
    df_n_filtered = df_n.copy()
    
    # Filter the voltage signal.
    df_V_low_filtered['V'] = apply_lowpass_filter(df_V_low['V'].values, cut_off_freq, fs)
    # Filter the density signal.
    df_n_filtered['np'] = apply_lowpass_filter(df_n['np'].values, cut_off_freq, fs)
    
    # Use the filtered signals for further processing.
    df_n_filtered['logn'] = np.log(df_n_filtered['np'].clip(lower=1e-10))
    df_n_filtered['V_lp'] = df_V_low_filtered['V']
    
    # Convert time indices to nanosecond integers.
    t_low = df_n_filtered.index.values.astype(np.int64)
    V_low = df_n_filtered['V_lp'].values
    logn = df_n_filtered['logn'].values
    t_high = df_V.index.values.astype(np.int64)
    V_high = df_V['V'].values
    
    # Step 3: Define window parameters.
    window_ns = pd.to_timedelta(window_str).value  # window length in nanoseconds
    step_ns = int(window_ns * (1 - overlap_ratio))   # step between windows
    starts = np.arange(t_high[0], t_high[-1], step_ns)
    ends = starts + window_ns

    # Step 4: Process each window in parallel to obtain calibration parameters and quality.
    if model_type == '3param':
        expected_cols = ['start', 'end', 'A', 'B', 'C', 'q']
    else:
        expected_cols = ['start', 'end', 'A', 'B', 'q']
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_window)(start, end, t_low, V_low, logn, model_type)
        for start, end in zip(starts, ends)
    )
    # Only keep valid results that have the expected number of columns.
    results_list = [r for r in results if r is not None and len(r) == len(expected_cols)]
    windows = pd.DataFrame(results_list, columns=expected_cols).sort_values('start')
    
    # Step 5: Compute weighted window-specific density estimates.
    n_samples = len(t_high)
    density_sum = np.zeros(n_samples)
    weight_sum = np.zeros(n_samples)
    
    # Loop over each window and compute its contribution.
    for _, row in windows.iterrows():
        start = row['start']
        end = row['end']
        # Identify high-res time points within the window.
        mask = (t_high >= start) & (t_high <= end)
        if not np.any(mask):
            continue
        indices = np.where(mask)[0]
        # Normalize time within the window to [0, 1]
        t_norm = (t_high[indices].astype(np.float64) - start) / window_ns
        # Compute Hann window weights.
        weights = 0.5 * (1 - np.cos(2 * np.pi * t_norm))
        # Incorporate the quality factor from the calibration fit.
        weights *= row['q']
        
        # Compute the window-specific density estimate.
        if model_type == '3param':
            n_window = row['A'] * np.exp(-row['B'] * (V_high[indices] + row['C']))
        else:
            n_window = row['A'] * np.exp(-row['B'] * V_high[indices])
        
        # Accumulate the weighted densities.
        density_sum[indices] += weights * n_window
        weight_sum[indices] += weights

    # Step 6: Compute the final density as the weighted average.
    valid = weight_sum > 0
    n_est = np.full(n_samples, np.nan)
    n_est[valid] = density_sum[valid] / weight_sum[valid]
    
    # Convert the result to a pandas Series indexed by high-res timestamps.
    time_index = pd.to_datetime(t_high)
    n_est_series = pd.Series(n_est, index=time_index)
    
    # Step 7: Apply percentile-based filtering and fill any remaining gaps.
    n_est_series = percentile_filter_interpolate_ts(n_est_series, lower_pct, upper_pct)
    n_est_series = n_est_series.ffill().bfill()
    
    return n_est_series, df_n_filtered

def process_window(start, end, t_low, V_low, logn, model_type):
    """
    Process a single window: initialize via linear regression and refine parameters using 
    least_squares. Also compute a quality factor based on the fit residual (cost).
    
    For '2param', returns: (start, end, A, B, q)
    For '3param', returns: (start, end, A, B, C, q)
    """
    mask = (t_low >= start) & (t_low <= end)
    window_V = V_low[mask]
    window_logn = logn[mask]
    
    try:
        if len(window_V) > 1:
            # Linear regression initialization.
            X = np.vstack([np.ones(len(window_V)), window_V]).T
            beta = np.linalg.lstsq(X, window_logn, rcond=None)[0]
            A0, B0 = np.exp(beta[0]), -beta[1]
        else:
            # Fallback for a single point.
            A0 = np.exp(window_logn[0]) if len(window_logn) > 0 else 1.0
            B0 = 0.1
        
        if model_type == '3param':
            C0 = np.median(-window_V + (window_logn - np.log(A0)) / B0) if B0 != 0 else 0.0
            p0 = [A0, B0, C0]
            bounds = ([1e-10, 1e-10, -10], [np.inf, np.inf, 10])
            res = least_squares(
                lambda p: window_logn - (np.log(p[0]) - p[1] * (window_V + p[2])),
                p0, bounds=bounds, loss='huber', x_scale='jac'
            )
            quality = 1.0 / (1.0 + np.sqrt(res.cost))
            return (start, end, res.x[0], res.x[1], res.x[2], quality)
        else:
            res = least_squares(
                lambda p: window_logn - (np.log(p[0]) - p[1] * window_V),
                [A0, B0], bounds=([1e-10, 1e-10], [np.inf, np.inf]),
                loss='huber', x_scale='jac'
            )
            quality = 1.0 / (1.0 + np.sqrt(res.cost))
            return (start, end, res.x[0], res.x[1], quality)
    except Exception as e:
        print(f"Window error ({start}-{end}): {e}")
        return None

def percentile_filter_interpolate_ts(ts, lower_pct, upper_pct):
    """
    Filter outlier values in the time series based on specified percentiles.
    Values outside [lower_pct, upper_pct] are set to NaN, then gaps are filled via linear interpolation.
    """
    valid = ts.dropna()
    if valid.empty:
        return ts
    lower_bound = np.percentile(valid, lower_pct)
    upper_bound = np.percentile(valid, upper_pct)
    ts_filtered = ts.copy()
    ts_filtered[(ts_filtered < lower_bound) | (ts_filtered > upper_bound)] = np.nan
    ts_filtered = ts_filtered.interpolate(method='linear', limit_direction='both')
    return ts_filtered




# ##########################################
# # 1. Process Spacecraft Potential Function
# ##########################################
# def process_sc_pot(df, voltage_columns=None):
#     """
#     Processes a DataFrame to compute an averaged signal from voltage columns 
#     and return the result as a DataFrame with the same datetime index.
    
#     Parameters:
#         df (pd.DataFrame): Input DataFrame with a datetime index and voltage columns.
#         voltage_columns (list of str, optional): List of voltage column names. 
#             If None, all columns containing 'V' in their name are used.
    
#     Returns:
#         pd.DataFrame: DataFrame containing the computed v_sc signal with the original datetime index.
#     """
#     if voltage_columns is None:
#         voltage_columns = [col for col in df.columns if 'V' in col]
#     v_sc = -np.nanmean(df[voltage_columns], axis=1)
#     return pd.DataFrame(v_sc, index=df.index, columns=['v_sc'])

# ##########################################
# # 3. Low-Pass Filter Function
# ##########################################
# from scipy.signal import butter, filtfilt

# def lowpass_filter(data, cutoff, fs, order=4):
#     """
#     Apply a low-pass Butterworth filter to the data.
    
#     Parameters:
#         data (array_like): Input signal.
#         cutoff (float): Cutoff frequency in Hz.
#         fs (float): Sampling frequency of the data in Hz.
#         order (int): Filter order.
    
#     Returns:
#         np.ndarray: The filtered data, or the original data if too short for filtering.
#     """
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     padlen = 3 * (max(len(b), len(a)) - 1)
#     if len(data) <= padlen:
#         return data
#     filtered = filtfilt(b, a, data)
#     return filtered





# from scipy import stats
# from scipy.optimize import least_squares

# def fit_exponential_two_param(x, y, weights=None, n_sigma=3, max_iter=5, A_bound_factor=10.0):
#     """
#     Fit the model: n = A * exp(-B * V)
    
#     In log-space the model is:
#          log(n) = log(A) - B * V.
#     This function uses robust (iterative sigma-clipping) weighted least squares
#     (if weights are provided) to fit the model and returns the fitted parameters
#     along with uncertainties.
    
#     Parameters
#     ----------
#     x : np.ndarray
#         Independent variable array (V).
#     y : np.ndarray
#         Dependent variable array (n); must be positive.
#     weights : np.ndarray or None, optional
#         Weights for each data point (typically 1/sigma for log(n)).
#         If None, all points are equally weighted.
#     n_sigma : float, optional
#         Sigma-clipping threshold in log-space (default: 3).
#     max_iter : int, optional
#         Maximum number of sigma-clipping iterations (default: 5).
#     A_bound_factor : float, optional
#         Multiplicative factor used to set the upper bound on A based on the robust initial guess.
#         (Default is 10.)
    
#     Returns
#     -------
#     A : float
#         Fitted amplitude.
#     B : float
#         Fitted exponential decay coefficient.
#     perr : np.ndarray
#         Estimated 1-sigma uncertainties [err_A, err_B].
#     """
#     # --- Data Cleaning ---
#     x = np.asarray(x).ravel()
#     y = np.asarray(y).ravel()
#     valid = (y > 0) & np.isfinite(x) & np.isfinite(y)
#     V = x[valid]
#     n = y[valid]
#     if V.size < 3:
#         raise ValueError("Not enough valid points to fit the model.")
#     if weights is not None:
#         weights = np.asarray(weights).ravel()[valid]
#     else:
#         weights = np.ones_like(V)
    
#     # --- Initial Estimates in Log-Space ---
#     log_n = np.log(n)
#     # Robust estimate for A from median(log_n)
#     A_robust = np.exp(np.median(log_n))
#     # Linear regression provides an intercept; however, it can be influenced by outliers.
#     slope, intercept, _, _, _ = stats.linregress(V, log_n)
#     A_reg = np.exp(intercept)
#     # Combine the two estimates (e.g., average them)
#     A0 = 0.5*(A_robust + A_reg)
#     # In the model log(n)=log(A)-B*V, the slope is -B.
#     B0 = -slope
#     initial_guess = [A0, B0]
    
#     # Set bounds:
#     # A must be positive, and we now impose an upper bound based on the robust estimate.
#     lower_bounds = [1e-12, 0]
#     upper_bounds = [A0 * A_bound_factor, 10]  # B is bounded between 0 and 10 (adjustable if needed)
    
#     # --- Define the Weighted Residual Function in Log-Space ---
#     def residuals(params, V_vals, n_obs, w):
#         A, B = params
#         # Compute the residuals in log-space.
#         res = np.log(n_obs) - (np.log(A) - B * V_vals)
#         return res / w
    
#     # --- Iterative Sigma Clipping ---
#     inliers = np.ones(len(V), dtype=bool)
#     popt = None
#     for ii in range(max_iter):
#         result = least_squares(
#             lambda p, V_sub, n_sub: residuals(p, V_sub, n_sub, weights[inliers]),
#             x0=initial_guess if popt is None else popt,
#             args=(V[inliers], n[inliers]),
#             loss='soft_l1',
#             bounds=(lower_bounds, upper_bounds)
#         )
#         if not result.success:
#             raise ValueError(f"Robust fitting did not converge in iteration {ii}")
#         popt = result.x
#         r_all = residuals(popt, V, n, weights)
#         mad = np.median(np.abs(r_all - np.median(r_all)))
#         robust_std = 1.4826 * mad if mad > 0 else np.std(r_all)
#         new_inliers = np.abs(r_all) < (n_sigma * robust_std)
#         if np.array_equal(new_inliers, inliers):
#             break
#         inliers = new_inliers
#         if np.sum(inliers) < 3:
#             raise ValueError("Too many outliers removed; not enough points remain for a robust fit.")
    
#     # --- Final Fit with Linear Loss for Covariance Estimation ---
#     result_final = least_squares(
#         lambda p, V_sub, n_sub: residuals(p, V_sub, n_sub, weights[inliers]),
#         x0=popt,
#         args=(V[inliers], n[inliers]),
#         loss='linear',
#         bounds=(lower_bounds, upper_bounds)
#     )
#     if not result_final.success:
#         raise ValueError("Final fitting did not converge.")
#     popt = result_final.x
#     A_opt, B_opt = popt
    
#     # --- Covariance Estimation ---
#     dof = max(len(result_final.fun) - len(popt), 1)
#     s_sq = 2 * result_final.cost / dof  # cost is 1/2 * sum of squares.
#     J = result_final.jac
#     cov = np.linalg.pinv(J.T.dot(J)) * s_sq
#     perr = np.sqrt(np.diag(cov))
    
#     return A_opt, B_opt, perr



# #########################################
# # 4. Revised Calibration Function
# #########################################
# def calibrate_highfreq_in_intervals(
#     df_highfreq,        # Original high-frequency DataFrame
#     df_qtn,             # QTN DataFrame (lower-frequency reference)
#     interval_size='4min',
#     col_sc_pot='v_sc',  # Column in df_highfreq with spacecraft potential
#     rol_med_wind='30s',
#     est_roll_med=True,
#     n_sigma=3,
#     clip_coeffs=[0.9, 1.1],
#     max_iter=1000,
#     fs=256,             # Sampling frequency of high-frequency data (Hz)
#     cutoff=None         # Calibration cutoff frequency (Hz); if None, computed from synchronized cadence.
# ):
#     """
#     Calibrates high-frequency spacecraft potential data using lower-frequency QTN density
#     measurements. In each interval the calibration mapping is assumed to follow the model:
    
#          n = A * exp(-B * V)
    
#     where A and B are determined using synchronized (low-frequency) data.
    
#     If est_roll_med is True, the QTN density is averaged using a rolling median, and a
#     corresponding rolling standard deviation is computed to form weights (w = 1/(std+eps))
#     for the fit. This weighted fit allows a more accurate estimate of the uncertainties in A and B.
    
#     The full-resolution potential is decomposed into a low-frequency component (V_low, via low-pass filtering)
#     and a high-frequency residual (delta_V). The calibrated density is computed as:
    
#          n_slow = A * exp(-B * V_low)
#          F = exp(-B * delta_V)
#          n_cal = n_slow * clip(F, clip_coeffs[0], clip_coeffs[1])
    
#     Parameters
#     ----------
#     df_highfreq : pd.DataFrame
#         High-frequency data with a DateTime index containing col_sc_pot.
#     df_qtn : pd.DataFrame
#         DataFrame with a DateTime index containing QTN density data.
#     interval_size : str or pd.Timedelta
#         Non-overlapping chunk size, e.g., "4min".
#     col_sc_pot : str
#         Column name for the spacecraft potential in df_highfreq.
#     rol_med_wind : str
#         Window length for rolling median (if est_roll_med is True).
#     est_roll_med : bool
#         Whether to apply a rolling median (and std) on the QTN data prior to synchronization.
#     n_sigma : float
#         Sigma-clipping threshold (default=3).
#     max_iter : int
#         Maximum outlier-removal iterations (default=1000).
#     clip_coeffs : list
#         Lower and upper clipping bounds for the high-frequency correction factor.
#     fs : float
#         Sampling frequency of the high-frequency data in Hz.
#     cutoff : float or None
#         Cutoff frequency for the low-pass filter in Hz. If None, computed as 1.15*(fs_sync/2),
#         where fs_sync is the sampling frequency of the synchronized data.
    
#     Returns
#     -------
#     df_out : pd.DataFrame
#         High-frequency data with an added column "sc_pot_dens" containing the calibrated density.
#     df_qtn_sync : pd.DataFrame
#         The synchronized QTN DataFrame.
#     save_A : list
#         List of fitted amplitude coefficients (A) for each interval.
#     save_B : list
#         List of fitted exponential coefficients (B) for each interval.
#     save_err_A : list
#         List of error estimates for coefficient A for each interval.
#     save_err_B : list
#         List of error estimates for coefficient B for each interval.
#     df_high_sync : pd.DataFrame
#         The synchronized high-frequency DataFrame.
#     Fs : np.ndarray
#         Concatenated high-frequency correction factors (before clipping).
#     Fs_cor : np.ndarray
#         Concatenated correction factors (after clipping).
#     V_lows : np.ndarray
#         Concatenated low-frequency potential values.
#     delta_Vs : np.ndarray
#         Concatenated high-frequency residuals.
#     """
#     # Process the spacecraft potential at full resolution.
#     df_highfreq_processed = process_sc_pot(df_highfreq)

#     print('GOT HERE')
#     # Synchronize the two DataFrames.
#     df_high_sync, df_qtn_sync = func.synchronize_dfs(
#         pd.DataFrame(df_highfreq_processed),
#         pd.DataFrame(df_qtn),
#         False)

#     print('GOT HERE 2')
    
#     # Prepare output DataFrame.
#     df_out = df_highfreq_processed.copy()
#     df_out["sc_pot_dens"] = np.nan
    
#     interval_size = pd.Timedelta(interval_size)
#     if len(df_high_sync) < 2:
#         return (df_out, df_qtn_sync, [], [], [], [], df_high_sync,
#                 np.array([]), np.array([]), np.array([]), np.array([]))
    
#     t_min = df_high_sync.index[0]
#     t_max = df_high_sync.index[-1]
#     current_start = t_min
    
#     save_A = []
#     save_B = []
#     save_err_A = []
#     save_err_B = []
#     Fs = []
#     Fs_cor = []
#     V_lows = []
#     delta_Vs = []
    
#     # Determine sampling frequencies.
#     fs_full = 1 / func.find_cadence(df_highfreq_processed)
#     fs_sync = 1 / func.find_cadence(df_high_sync)
#     if cutoff is None:
#         cutoff = 1.15 * (fs_sync / 2)
    
#     while current_start < t_max:
#         current_end = current_start + interval_size
        
#         # Obtain synchronized low-frequency data for the interval.
#         if est_roll_med:
#             chunk_sync_hf = df_high_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).median()
#             chunk_sync_qtn = df_qtn_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).median()
#             # Also compute the rolling standard deviation to form weights.
#             chunk_qtn_std = df_qtn_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).std()
#             eps = 1e-6
#             weights = 1.0 / (chunk_qtn_std.values.ravel() + eps)
#         else:
#             chunk_sync_hf = df_high_sync.loc[current_start:current_end]
#             chunk_sync_qtn = df_qtn_sync.loc[current_start:current_end]
#             weights = None
        
#         if len(chunk_sync_hf) < 2 or len(chunk_sync_qtn) < 2:
#             current_start = current_end
#             continue
        
#         # Prepare the fitting data.
#         x = chunk_sync_hf[col_sc_pot].values
#         y = chunk_sync_qtn.values.ravel()
        
#         try:
#             A, B, perr = fit_exponential_two_param(x, y, weights=weights, n_sigma=n_sigma, max_iter=max_iter)
#         except ValueError:
#             current_start = current_end
#             continue
        
#         # Use the full-resolution potential for calibration.
#         hf_chunk = df_highfreq_processed.loc[current_start:current_end, col_sc_pot].values
#         V_low = lowpass_filter(hf_chunk, cutoff, fs_full)
#         delta_V = hf_chunk - V_low
        
#         # Compute calibrated density.
#         n_slow = A * np.exp(-B * V_low)
#         F = np.exp(-B * delta_V)
#         F_corr = np.clip(F, clip_coeffs[0], clip_coeffs[1])
#         n_cal = n_slow * F_corr
        
#         df_out.loc[current_start:current_end, "sc_pot_dens"] = pd.Series(n_cal, index=df_out.loc[current_start:current_end].index)
        
#         save_A.append(A)
#         save_B.append(B)
#         save_err_A.append(perr[0])
#         save_err_B.append(perr[1])
#         V_lows.append(V_low)
#         delta_Vs.append(delta_V)
#         Fs.append(F)
#         Fs_cor.append(F_corr)
        
#         current_start = current_end

#     print('Final', df_out)
    
#     return (df_out, df_qtn_sync, save_A, save_B, save_err_A, save_err_B,
#             df_high_sync, np.hstack(Fs), np.hstack(Fs_cor), np.hstack(V_lows), np.hstack(delta_Vs))
