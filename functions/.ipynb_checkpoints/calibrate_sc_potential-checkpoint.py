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

##########################################
# 1. Process Spacecraft Potential Function
##########################################
def process_sc_pot(df, voltage_columns=None):
    """
    Processes a DataFrame to compute an averaged signal from voltage columns 
    and return the result as a DataFrame with the same datetime index.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a datetime index and voltage columns.
        voltage_columns (list of str, optional): List of voltage column names. 
            If None, all columns containing 'V' in their name are used.
    
    Returns:
        pd.DataFrame: DataFrame containing the computed v_sc signal with the original datetime index.
    """
    if voltage_columns is None:
        voltage_columns = [col for col in df.columns if 'V' in col]
    v_sc = -np.nanmean(df[voltage_columns], axis=1)
    return pd.DataFrame(v_sc, index=df.index, columns=['v_sc'])

##########################################
# 3. Low-Pass Filter Function
##########################################
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
        data (array_like): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency of the data in Hz.
        order (int): Filter order.
    
    Returns:
        np.ndarray: The filtered data, or the original data if too short for filtering.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    padlen = 3 * (max(len(b), len(a)) - 1)
    if len(data) <= padlen:
        return data
    filtered = filtfilt(b, a, data)
    return filtered



##########################################
# 2. Robust Fitting Function
##########################################

# import numpy as np
# from scipy import stats
# from scipy.optimize import least_squares


# def fit_single_exponential_with_outliers(x, y, n_sigma=3, max_iter=5,
#                                           a_bound_factor=10.0,
#                                           b_bound_factor=5.0,
#                                           c_range_factor=0.5,
#                                           reg_factor=1.0):
#     """
#     Fit the model: y = a * exp(b*(x+c))
#     using a reparameterization and a regularization penalty on c to reduce the
#     degeneracy between a and c.
    
#     We set u = log(a) so that the model becomes:
#          log(y) = u + b*(x+c)
#     and we add a penalty term sqrt(lambda_c)*(c - c0) to the residuals,
#     where c0 = -median(x) (a natural centering choice) and
#          lambda_c = reg_factor / (x_range^2).
#     This acts as a weak prior on c.
    
#     Parameters
#     ----------
#     x : np.ndarray
#         Independent variable array.
#     y : np.ndarray
#         Dependent variable array (must be positive).
#     n_sigma : float, optional
#         Sigma-clipping threshold in log-space (default is 3).
#     max_iter : int, optional
#         Maximum number of sigma clipping iterations (default is 5).
#     a_bound_factor : float, optional
#         Factor by which the initial estimate for a (via u=log(a)) is allowed to vary (default: 10).
#     b_bound_factor : float, optional
#         Factor for setting the bounds on b based on the regression standard error (default: 5).
#     c_range_factor : float, optional
#         Fraction of the x-range (divided by 2) allowed for variation in c (default: 0.5).
#     reg_factor : float, optional
#         Factor for the regularization weight on c. The effective penalty weight is
#             lambda_c = reg_factor / (x_range^2).
#         Default is 1.0.
    
#     Returns
#     -------
#     a : float
#         Fitted amplitude.
#     b : float
#         Fitted exponential coefficient.
#     c : float
#         Fitted horizontal offset.
#     perr : np.ndarray
#         Estimated 1-sigma uncertainties [err_a, err_b, err_c].
#     """
#     # --- Data Cleaning ---
#     x = np.asarray(x).ravel()
#     y = np.asarray(y).ravel()
#     valid = (y > 0) & np.isfinite(x) & np.isfinite(y)
#     x_clean = x[valid]
#     y_clean = y[valid]
#     if x_clean.size < 3:
#         raise ValueError("Not enough valid points to fit the model.")
    
#     # --- Initial Estimates via Linear Regression in Log-Space ---
#     log_y = np.log(y_clean)
#     slope, intercept, _, _, stderr = stats.linregress(x_clean, log_y)
#     # Center x for decoupling; choose c0 = -median(x)
#     c0 = -np.median(x_clean)
#     # Compute initial u (log(a)): u0 = intercept - slope*c0, so that a0 = exp(u0)
#     u0 = intercept - slope * c0
#     a_initial = np.exp(u0)
    
#     # For b: if the slope is nearly zero, use default bounds.
#     if np.abs(slope) < 1e-6:
#         b_lower, b_upper = -10, 10
#     else:
#         eps = 1e-3  # small floor for stderr
#         b_scale = stderr if (stderr and stderr > eps) else eps
#         b_lower = slope - b_bound_factor * np.abs(b_scale)
#         b_upper = slope + b_bound_factor * np.abs(b_scale)
#         b_lower = max(b_lower, -10)
#         b_upper = min(b_upper, 10)
#     b_initial = np.clip(slope, b_lower, b_upper)
    
#     # --- Dynamic Bounds ---
#     # For u = log(a): allow ±log(a_bound_factor) variation around u0.
#     u_lower = u0 - np.log(a_bound_factor)
#     u_upper = u0 + np.log(a_bound_factor)
    
#     # For c: tie bounds to the range of x.
#     x_range = np.max(x_clean) - np.min(x_clean)
#     c_lower = c0 - c_range_factor * (x_range / 2.0)
#     c_upper = c0 + c_range_factor * (x_range / 2.0)
    
#     # --- Regularization Weight for c ---
#     lambda_c = reg_factor / (x_range**2) if x_range > 0 else reg_factor
#     sqrt_lambda = np.sqrt(lambda_c)
    
#     # Compose the initial guess and bounds for parameters [u, b, c]
#     initial_guess = [u0, b_initial, c0]
#     lower_bounds = [u_lower, b_lower, c_lower]
#     upper_bounds = [u_upper, b_upper, c_upper]
    
#     # --- Define the Residual Function with Regularization ---
#     def residuals(params, x_vals, log_y_obs):
#         u, b, c = params
#         # Data residuals (in log-space)
#         r_data = (u + b * (x_vals + c) - log_y_obs)
#         # Regularization penalty on c (pulling toward c0)
#         r_penalty = sqrt_lambda * (c - c0)
#         # Concatenate the data residuals with the penalty term.
#         return np.concatenate([r_data, [r_penalty]])
    
#     # --- For Sigma Clipping, use only the data residuals ---
#     def data_residual(params, x_vals, log_y_obs):
#         u, b, c = params
#         return u + b * (x_vals + c) - log_y_obs
    
#     # --- Iterative Sigma Clipping ---
#     inliers = np.ones(len(x_clean), dtype=bool)
#     popt = None
#     for ii in range(max_iter):
#         result = least_squares(
#             residuals,
#             x0=initial_guess if popt is None else popt,
#             args=(x_clean[inliers], log_y[inliers]),
#             loss='soft_l1',
#             bounds=(lower_bounds, upper_bounds)
#         )
#         if not result.success:
#             raise ValueError(f"Robust fitting did not converge in iteration {ii}")
#         popt = result.x
#         r_data_all = data_residual(popt, x_clean, log_y)
#         mad = np.median(np.abs(r_data_all - np.median(r_data_all)))
#         robust_std = 1.4826 * mad if mad > 0 else np.std(r_data_all)
#         new_inliers = np.abs(r_data_all) < (n_sigma * robust_std)
#         if np.array_equal(new_inliers, inliers):
#             break
#         inliers = new_inliers
#         if np.sum(inliers) < 3:
#             raise ValueError("Too many outliers removed; not enough points remain for a robust fit.")
    
#     # --- Final Fit with Linear Loss for Covariance Estimation ---
#     result_final = least_squares(
#         residuals,
#         x0=popt,
#         args=(x_clean[inliers], log_y[inliers]),
#         loss='linear',
#         bounds=(lower_bounds, upper_bounds)
#     )
#     if not result_final.success:
#         raise ValueError("Final fitting did not converge.")
    
#     popt = result_final.x
#     u_opt, b_opt, c_opt = popt
#     a_opt = np.exp(u_opt)
    
#     # --- Covariance Estimation ---
#     dof = max(len(result_final.fun) - len(popt), 1)
#     s_sq = 2 * result_final.cost / dof  # cost = 1/2 * sum of squares
#     J = result_final.jac
#     JTJ = J.T.dot(J)
#     cov = np.linalg.pinv(JTJ) * s_sq
#     perr_params = np.sqrt(np.diag(cov))
#     # Transform error in u to error in a: Δa = a * Δu.
#     a_err = a_opt * perr_params[0]
#     b_err = perr_params[1]
#     c_err = perr_params[2]
#     perr = np.array([a_err, b_err, c_err])
    
#     return a_opt, b_opt, c_opt, perr



# #########################################
# #4. Revised Calibration Function
# #########################################
# def calibrate_highfreq_in_intervals(
#     df_highfreq,        # Original high-frequency DataFrame
#     df_qtn,             # QTN DataFrame (lower-frequency reference)
#     interval_size='4min',
#     col_sc_pot='v_sc',  # Column in df_highfreq with spacecraft potential
#     rol_med_wind='30s',
#     est_roll_med=True,
#     n_sigma=3,
#     clip_coeffs = [0.6, 1.2],
#     max_iter=1000,
#     fs=256,             # Sampling frequency of high-frequency data (Hz)
#     cutoff=None         # Calibration cutoff frequency (Hz); if None, it is computed from synchronized cadence.
# ):
#     """
#     Calibrates high-frequency spacecraft potential data using lower-frequency QTN density
#     measurements. In each interval, the calibration mapping
#          n = a * exp(b*(V + c))
#     is determined using the synchronized data. To avoid aliasing when applying the calibration
#     to high-frequency data, the full-resolution potential is decomposed into a low-frequency (calibrated)
#     component and a high-frequency residual. The low-frequency calibrated density is computed as
#          n_slow = a * exp(b*(V_low + c))
#     and then a nonlinear correction is applied:
#          n_cal = n_slow * exp(b*delta_V)
#     To avoid over-amplification of high-frequency noise (which may lead to aliasing), the correction factor
#     is clipped to a narrow range (e.g. [0.9, 1.1]). This yields:
#          n_cal = n_slow * clip(exp(b*delta_V), 0.9, 1.1)
    
#     This approach uses the low-frequency calibration coefficients while limiting the impact of high-frequency fluctuations.
    
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
#         Whether to apply a rolling median on the QTN data prior to synchronization.
#     n_sigma : float
#         Sigma-clipping threshold (default=3).
#     max_iter : int
#         Maximum outlier-removal iterations (default=10000).
#     fs : float
#         Sampling frequency of the high-frequency data in Hz.
#     cutoff : float or None
#         Cutoff frequency for the low-pass filter in Hz. If None, it is computed as 
#         1.15*(fs_sync/2), where fs_sync is the sampling frequency of the synchronized data.
    
#     Returns
#     -------
#     df_out : pd.DataFrame
#         High-frequency data with an added column "sc_pot_dens" containing the calibrated density.
#     df_qtn_sync : pd.DataFrame
#         The synchronized QTN DataFrame.
#     save_a : list
#         List of fitted amplitude coefficients for each interval.
#     save_b : list
#         List of fitted exponential coefficients for each interval.
#     save_c : list
#         List of fitted offset coefficients for each interval.
#     save_err_a : list
#         List of error estimates for coefficient a for each interval.
#     save_err_b : list
#         List of error estimates for coefficient b for each interval.
#     save_err_c : list
#         List of error estimates for coefficient c for each interval.
#     df_highfreq_processed : pd.DataFrame
#         The processed high-frequency data (with column 'v_sc').
#     """
#     # Process the spacecraft potential at full resolution.
#     df_highfreq_processed = process_sc_pot(df_highfreq)
    
#     # Synchronize using the (potentially downsampled) data
#     df_high_sync, df_qtn_sync = func.synchronize_dfs(
#         pd.DataFrame(df_highfreq_processed),
#         pd.DataFrame(df_qtn),
#         False)


    
#     # Prepare output DataFrame (full resolution).
#     df_out = df_highfreq_processed.copy()
#     df_out["sc_pot_dens"] = np.nan
    
#     interval_size = pd.Timedelta(interval_size)
#     if len(df_high_sync) < 2:
#         return df_out, df_qtn_sync, [], [], [], [], [], [], df_highfreq_processed
    
#     t_min = df_high_sync.index[0]
#     t_max = df_high_sync.index[-1]
#     current_start = t_min
    
#     save_a = []
#     save_b = []
#     save_c = []
#     save_err_a = []
#     save_err_b = []
#     save_err_c = []
#     Fs         = []
#     Fs_cor     = []
#     V_lows, delta_Vs = [], []
    
    
#     # Determine sampling frequencies.
#     fs_full = 1 / func.find_cadence(df_highfreq_processed)
#     fs_sync = 1 / func.find_cadence(df_high_sync)
#     if cutoff is None:
#         cutoff = 1.15* (fs_sync / 2)  # Use a margin on the Nyquist frequency of the synchronized data.
    
#     while current_start < t_max:
#         current_end = current_start + interval_size

#         # Use the synchronized (low-frequency) data for fitting.
#         if est_roll_med:
#             # Estimate rolling median
#             chunk_sync_hf =  df_high_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).median()
#             chunk_sync_qtn  = df_qtn_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).median()
#         else:
            
#             chunk_sync_hf = df_high_sync.loc[current_start:current_end]
#             chunk_sync_qtn = df_qtn_sync.loc[current_start:current_end]
#         if len(chunk_sync_hf) < 2 or len(chunk_sync_qtn) < 2:
#             current_start = current_end
#             continue
        
#         x = chunk_sync_hf[col_sc_pot].values
#         y = chunk_sync_qtn.values.ravel()
        
#         try:
#             a, b, c, err = fit_single_exponential_with_outliers(x, y, n_sigma=n_sigma, max_iter=max_iter)
#         except ValueError:
#             current_start = current_end
#             continue
        
#         # Use the full-resolution potential from df_highfreq_processed for calibration.
#         hf_chunk = df_highfreq_processed.loc[current_start:current_end, col_sc_pot].values
        
#         # Decompose the full-resolution potential:
#         V_low   = lowpass_filter(hf_chunk, cutoff, fs_full)
#         delta_V = hf_chunk - V_low
        
#         # Compute the calibrated low-frequency density.
#         n_slow = a * np.exp(b * (V_low + c))
#         # Compute full correction factor F = exp(b * delta_V)
#         F = np.exp(b * delta_V)
#         # To avoid over–amplification of high-frequency noise, clip F to a reasonable range.

#         #F_corr = np.clip(F, clip_coeffs[0], F_outliers)
#         F_corr = np.clip(F, clip_coeffs[0],clip_coeffs[1])
#         n_cal = n_slow * F_corr
        
#         # Assign the calibrated density to the output.
#         df_out.loc[current_start:current_end, "sc_pot_dens"] = pd.Series(n_cal, index=df_out.loc[current_start:current_end].index)
        
#         save_a.append(a)
#         save_b.append(b)
#         save_c.append(c)
#         save_err_a.append(err[0])
#         save_err_b.append(err[1])
#         save_err_c.append(err[2])
#         V_lows.append(V_low)
#         delta_Vs.append(delta_V)
#         Fs.append(F)
#         Fs_cor.append(F_corr)
        
#         current_start = current_end
    
#    # if "v_sc" in df_out.columns:
#     #    del df_out["v_sc"]
    
#     return (df_out, df_qtn_sync, save_a, save_b, save_c,
#             save_err_a, save_err_b, save_err_c, df_high_sync, np.hstack(Fs), np.hstack(Fs_cor), np.hstack(V_lows), np.hstack(delta_Vs))

# import numpy as np
# from scipy import stats
# from scipy.optimize import least_squares

# def fit_exponential_two_param(x, y, weights=None, n_sigma=3, max_iter=5, A_bound_factor=100.0):
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
#         (Default is 100.)
    
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
    
#     # Robust estimate for log(A) from the median of log(n)
#     log_A_robust = np.median(log_n)
#     A_robust = np.exp(log_A_robust)
    
#     # Weighted linear regression in log-space
#     def weighted_linregress(x, y, w):
#         sum_w = np.sum(w)
#         x_wavg = np.sum(w * x) / sum_w
#         y_wavg = np.sum(w * y) / sum_w
#         cov_xy = np.sum(w * (x - x_wavg) * (y - y_wavg)) / sum_w
#         var_x = np.sum(w * (x - x_wavg)**2) / sum_w
#         slope = cov_xy / var_x
#         intercept = y_wavg - slope * x_wavg
#         return slope, intercept
    
#     if weights is not None:
#         slope, intercept = weighted_linregress(V, log_n, weights)
#     else:
#         slope, intercept, _, _, _ = stats.linregress(V, log_n)
    
#     A_reg = np.exp(intercept)
#     B0 = -slope  # Model: log(n) = log(A) - B*V => slope = -B
    
#     # Combine robust and regression-based estimates for A
#     A0 = 0.5 * (A_robust + A_reg)
#     initial_guess = [A0, B0]
    
#     # --- Parameter Bounds ---
#     lower_bounds = [1e-12, 0]  # A > 0, B >= 0
#     upper_bounds = [
#         A0 * A_bound_factor,  # Lenient upper bound for A
#         1e5  # Physically reasonable upper bound for B
#     ]
    
#     # --- Residual Function (Log-Space) ---
#     def residuals(params, V_vals, n_obs, w):
#         A, B = params
#         res = np.log(n_obs) - (np.log(A) - B * V_vals)
#         return res / w  # Weighted residuals
    
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
#             raise ValueError(f"Fit did not converge in iteration {ii}")
#         popt = result.x
        
#         # Update inliers using robust std
#         r_all = residuals(popt, V, n, weights)
#         mad = np.median(np.abs(r_all - np.median(r_all)))
#         robust_std = 1.4826 * mad if mad > 0 else np.std(r_all)
#         new_inliers = np.abs(r_all) < (n_sigma * robust_std)
        
#         if np.array_equal(new_inliers, inliers):
#             break
#         inliers = new_inliers
#         if np.sum(inliers) < 3:
#             raise ValueError("Too many outliers removed; insufficient data for fit.")
    
#     # --- Final Fit with Linear Loss (for Covariance) ---
#     result_final = least_squares(
#         lambda p, V_sub, n_sub: residuals(p, V_sub, n_sub, weights[inliers]),
#         x0=popt,
#         args=(V[inliers], n[inliers]),
#         loss='linear',
#         bounds=(lower_bounds, upper_bounds))
#     if not result_final.success:
#         raise ValueError("Final fitting did not converge.")
#     A_opt, B_opt = result_final.x
    
#     # --- Uncertainty Estimation ---
#     dof = max(len(result_final.fun) - len(result_final.x), 1)
#     s_sq = 2 * result_final.cost / dof  # Residual variance
#     J = result_final.jac
#     cov = np.linalg.pinv(J.T.dot(J)) * s_sq
#     perr = np.sqrt(np.diag(cov))
    
#     return A_opt, B_opt, perr




import numpy as np
from scipy import stats
from scipy.optimize import least_squares

def fit_exponential_two_param(x, y, weights=None, n_sigma=3, max_iter=5, A_bound_factor=10.0):
    """
    Fit the model: n = A * exp(-B * V)
    
    In log-space the model is:
         log(n) = log(A) - B * V.
    This function uses robust (iterative sigma-clipping) weighted least squares
    (if weights are provided) to fit the model and returns the fitted parameters
    along with uncertainties.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable array (V).
    y : np.ndarray
        Dependent variable array (n); must be positive.
    weights : np.ndarray or None, optional
        Weights for each data point (typically 1/sigma for log(n)).
        If None, all points are equally weighted.
    n_sigma : float, optional
        Sigma-clipping threshold in log-space (default: 3).
    max_iter : int, optional
        Maximum number of sigma-clipping iterations (default: 5).
    A_bound_factor : float, optional
        Multiplicative factor used to set the upper bound on A based on the robust initial guess.
        (Default is 10.)
    
    Returns
    -------
    A : float
        Fitted amplitude.
    B : float
        Fitted exponential decay coefficient.
    perr : np.ndarray
        Estimated 1-sigma uncertainties [err_A, err_B].
    """
    # --- Data Cleaning ---
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = (y > 0) & np.isfinite(x) & np.isfinite(y)
    V = x[valid]
    n = y[valid]
    if V.size < 3:
        raise ValueError("Not enough valid points to fit the model.")
    if weights is not None:
        weights = np.asarray(weights).ravel()[valid]
    else:
        weights = np.ones_like(V)
    
    # --- Initial Estimates in Log-Space ---
    log_n = np.log(n)
    # Robust estimate for A from median(log_n)
    A_robust = np.exp(np.median(log_n))
    # Linear regression provides an intercept; however, it can be influenced by outliers.
    slope, intercept, _, _, _ = stats.linregress(V, log_n)
    A_reg = np.exp(intercept)
    # Combine the two estimates (e.g., average them)
    A0 = 0.5*(A_robust + A_reg)
    # In the model log(n)=log(A)-B*V, the slope is -B.
    B0 = -slope
    initial_guess = [A0, B0]
    
    # Set bounds:
    # A must be positive, and we now impose an upper bound based on the robust estimate.
    lower_bounds = [1e-12, 0]
    upper_bounds = [A0 * A_bound_factor, 10]  # B is bounded between 0 and 10 (adjustable if needed)
    
    # --- Define the Weighted Residual Function in Log-Space ---
    def residuals(params, V_vals, n_obs, w):
        A, B = params
        # Compute the residuals in log-space.
        res = np.log(n_obs) - (np.log(A) - B * V_vals)
        return res / w
    
    # --- Iterative Sigma Clipping ---
    inliers = np.ones(len(V), dtype=bool)
    popt = None
    for ii in range(max_iter):
        result = least_squares(
            lambda p, V_sub, n_sub: residuals(p, V_sub, n_sub, weights[inliers]),
            x0=initial_guess if popt is None else popt,
            args=(V[inliers], n[inliers]),
            loss='soft_l1',
            bounds=(lower_bounds, upper_bounds)
        )
        if not result.success:
            raise ValueError(f"Robust fitting did not converge in iteration {ii}")
        popt = result.x
        r_all = residuals(popt, V, n, weights)
        mad = np.median(np.abs(r_all - np.median(r_all)))
        robust_std = 1.4826 * mad if mad > 0 else np.std(r_all)
        new_inliers = np.abs(r_all) < (n_sigma * robust_std)
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
        if np.sum(inliers) < 3:
            raise ValueError("Too many outliers removed; not enough points remain for a robust fit.")
    
    # --- Final Fit with Linear Loss for Covariance Estimation ---
    result_final = least_squares(
        lambda p, V_sub, n_sub: residuals(p, V_sub, n_sub, weights[inliers]),
        x0=popt,
        args=(V[inliers], n[inliers]),
        loss='linear',
        bounds=(lower_bounds, upper_bounds)
    )
    if not result_final.success:
        raise ValueError("Final fitting did not converge.")
    popt = result_final.x
    A_opt, B_opt = popt
    
    # --- Covariance Estimation ---
    dof = max(len(result_final.fun) - len(popt), 1)
    s_sq = 2 * result_final.cost / dof  # cost is 1/2 * sum of squares.
    J = result_final.jac
    cov = np.linalg.pinv(J.T.dot(J)) * s_sq
    perr = np.sqrt(np.diag(cov))
    
    return A_opt, B_opt, perr



#########################################
# 4. Revised Calibration Function
#########################################
def calibrate_highfreq_in_intervals(
    df_highfreq,        # Original high-frequency DataFrame
    df_qtn,             # QTN DataFrame (lower-frequency reference)
    interval_size='4min',
    col_sc_pot='v_sc',  # Column in df_highfreq with spacecraft potential
    rol_med_wind='30s',
    est_roll_med=True,
    n_sigma=3,
    clip_coeffs=[0.9, 1.1],
    max_iter=1000,
    fs=256,             # Sampling frequency of high-frequency data (Hz)
    cutoff=None         # Calibration cutoff frequency (Hz); if None, computed from synchronized cadence.
):
    """
    Calibrates high-frequency spacecraft potential data using lower-frequency QTN density
    measurements. In each interval the calibration mapping is assumed to follow the model:
    
         n = A * exp(-B * V)
    
    where A and B are determined using synchronized (low-frequency) data.
    
    If est_roll_med is True, the QTN density is averaged using a rolling median, and a
    corresponding rolling standard deviation is computed to form weights (w = 1/(std+eps))
    for the fit. This weighted fit allows a more accurate estimate of the uncertainties in A and B.
    
    The full-resolution potential is decomposed into a low-frequency component (V_low, via low-pass filtering)
    and a high-frequency residual (delta_V). The calibrated density is computed as:
    
         n_slow = A * exp(-B * V_low)
         F = exp(-B * delta_V)
         n_cal = n_slow * clip(F, clip_coeffs[0], clip_coeffs[1])
    
    Parameters
    ----------
    df_highfreq : pd.DataFrame
        High-frequency data with a DateTime index containing col_sc_pot.
    df_qtn : pd.DataFrame
        DataFrame with a DateTime index containing QTN density data.
    interval_size : str or pd.Timedelta
        Non-overlapping chunk size, e.g., "4min".
    col_sc_pot : str
        Column name for the spacecraft potential in df_highfreq.
    rol_med_wind : str
        Window length for rolling median (if est_roll_med is True).
    est_roll_med : bool
        Whether to apply a rolling median (and std) on the QTN data prior to synchronization.
    n_sigma : float
        Sigma-clipping threshold (default=3).
    max_iter : int
        Maximum outlier-removal iterations (default=1000).
    clip_coeffs : list
        Lower and upper clipping bounds for the high-frequency correction factor.
    fs : float
        Sampling frequency of the high-frequency data in Hz.
    cutoff : float or None
        Cutoff frequency for the low-pass filter in Hz. If None, computed as 1.15*(fs_sync/2),
        where fs_sync is the sampling frequency of the synchronized data.
    
    Returns
    -------
    df_out : pd.DataFrame
        High-frequency data with an added column "sc_pot_dens" containing the calibrated density.
    df_qtn_sync : pd.DataFrame
        The synchronized QTN DataFrame.
    save_A : list
        List of fitted amplitude coefficients (A) for each interval.
    save_B : list
        List of fitted exponential coefficients (B) for each interval.
    save_err_A : list
        List of error estimates for coefficient A for each interval.
    save_err_B : list
        List of error estimates for coefficient B for each interval.
    df_high_sync : pd.DataFrame
        The synchronized high-frequency DataFrame.
    Fs : np.ndarray
        Concatenated high-frequency correction factors (before clipping).
    Fs_cor : np.ndarray
        Concatenated correction factors (after clipping).
    V_lows : np.ndarray
        Concatenated low-frequency potential values.
    delta_Vs : np.ndarray
        Concatenated high-frequency residuals.
    """
    # Process the spacecraft potential at full resolution.
    df_highfreq_processed = process_sc_pot(df_highfreq)
    
    # Synchronize the two DataFrames.
    df_high_sync, df_qtn_sync = func.synchronize_dfs(
        pd.DataFrame(df_highfreq_processed),
        pd.DataFrame(df_qtn),
        False)
    
    # Prepare output DataFrame.
    df_out = df_highfreq_processed.copy()
    df_out["sc_pot_dens"] = np.nan
    
    interval_size = pd.Timedelta(interval_size)
    if len(df_high_sync) < 2:
        return (df_out, df_qtn_sync, [], [], [], [], df_high_sync,
                np.array([]), np.array([]), np.array([]), np.array([]))
    
    t_min = df_high_sync.index[0]
    t_max = df_high_sync.index[-1]
    current_start = t_min
    
    save_A = []
    save_B = []
    save_err_A = []
    save_err_B = []
    Fs = []
    Fs_cor = []
    V_lows = []
    delta_Vs = []
    
    # Determine sampling frequencies.
    fs_full = 1 / func.find_cadence(df_highfreq_processed)
    fs_sync = 1 / func.find_cadence(df_high_sync)
    if cutoff is None:
        cutoff = 1.15 * (fs_sync / 2)
    
    while current_start < t_max:
        current_end = current_start + interval_size
        
        # Obtain synchronized low-frequency data for the interval.
        if est_roll_med:
            chunk_sync_hf = df_high_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).median()
            chunk_sync_qtn = df_qtn_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).median()
            # Also compute the rolling standard deviation to form weights.
            chunk_qtn_std = df_qtn_sync.loc[current_start:current_end].rolling(rol_med_wind, center=True).std()
            eps = 1e-6
            weights = 1.0 / (chunk_qtn_std.values.ravel() + eps)
        else:
            chunk_sync_hf = df_high_sync.loc[current_start:current_end]
            chunk_sync_qtn = df_qtn_sync.loc[current_start:current_end]
            weights = None
        
        if len(chunk_sync_hf) < 2 or len(chunk_sync_qtn) < 2:
            current_start = current_end
            continue
        
        # Prepare the fitting data.
        x = chunk_sync_hf[col_sc_pot].values
        y = chunk_sync_qtn.values.ravel()
        
        try:
            A, B, perr = fit_exponential_two_param(x, y, weights=weights, n_sigma=n_sigma, max_iter=max_iter)
        except ValueError:
            current_start = current_end
            continue
        
        # Use the full-resolution potential for calibration.
        hf_chunk = df_highfreq_processed.loc[current_start:current_end, col_sc_pot].values
        V_low = lowpass_filter(hf_chunk, cutoff, fs_full)
        delta_V = hf_chunk - V_low
        
        # Compute calibrated density.
        n_slow = A * np.exp(-B * V_low)
        F = np.exp(-B * delta_V)
        F_corr = np.clip(F, clip_coeffs[0], clip_coeffs[1])
        n_cal = n_slow * F_corr
        
        df_out.loc[current_start:current_end, "sc_pot_dens"] = pd.Series(n_cal, index=df_out.loc[current_start:current_end].index)
        
        save_A.append(A)
        save_B.append(B)
        save_err_A.append(perr[0])
        save_err_B.append(perr[1])
        V_lows.append(V_low)
        delta_Vs.append(delta_V)
        Fs.append(F)
        Fs_cor.append(F_corr)
        
        current_start = current_end
    
    return (df_out, df_qtn_sync, save_A, save_B, save_err_A, save_err_B,
            df_high_sync, np.hstack(Fs), np.hstack(Fs_cor), np.hstack(V_lows), np.hstack(delta_Vs))
