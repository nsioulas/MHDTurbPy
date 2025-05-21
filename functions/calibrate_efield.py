"""e_field_calibration.py
========================
Module providing a modern, vectorised pipeline to project probe voltages into spacecraft
coordinates, fit a four‑parameter electric‑field calibration model on overlapping time
windows, and return smoothly calibrated Ex/Ey time series together with window‑level
fit diagnostics.

Main entry point
----------------
calibrate_electric_field(...) – blends Hann‑tapered, quality‑weighted window fits in
parallel, optional low‑pass pre‑filtering, and percentile‑based outlier rejection.

Utility helpers
---------------
* project_dV               – rotate probe‐pair voltages into S/C coordinates.
* apply_lowpass_filter     – zero‑phase Butterworth low‑pass.
* percentile_filter_interpolate_ts – robust percentile clipping + gap interpolation.
* find_longest_intervals   – pick the M longest |Bz|>threshold intervals.
* estimate_Ez              – reconstruct Ez from E·B=0 plus optional Hampel filter.

Notes
-----
* Synchronisation of data frames relies on ``general_functions.synchronize_dfs``
  (imported as ``func``).  Replace with your own routine if needed.
* All physical unit conversions are explicit and NumPy‑vectorised.
* No external state is stored; everything is functional and testable.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import least_squares
from joblib import Parallel, delayed



import numpy as np
import pandas as pd
import traceback
from astropy import units as u

import traceback
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
from scipy.optimize import curve_fit
import joblib
from joblib import Parallel, delayed
import statistics
from statistics import mode
import orderedstructs
import sys
from scipy.optimize import curve_fit
import astropy.units as u


import numpy as np
import pandas as pd
import traceback
from astropy import units as u
import traceback
from scipy.signal import butter, filtfilt

from scipy.signal import firwin, filtfilt
from scipy.odr import ODR, Model, Data
from astropy import units as u

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func



def project_dV(dV12, dV34):
    """
    Project dV12 and dV34 from whip coordinates to spacecraft (SC) coordinates.

    Parameters:
    - dV12: numpy array of differential voltages from probes 1 and 2 (V).
    - dV34: numpy array of differential voltages from probes 3 and 4 (V).

    Returns:
    - dVX, dVY: numpy arrays of differential voltages in SC coordinates.
    """
    R_V_to_SC = np.array([[0.64524, -0.82228],
                          [0.76897,  0.57577]])
    dV_whip = np.vstack((dV12, dV34))
    dV_SC   = np.dot(R_V_to_SC, dV_whip)
    dVX     = dV_SC[0, :]
    dVY     = dV_SC[1, :]
    return dVX, dVY





def fit_coupled_linear_model(Vp, B, dVX, dVY, robust=False, rejection_threshold=3):
    """
    Fit the projected differential voltages using a coupled four-parameter linear model
    via Orthogonal Distance Regression (ODR) to obtain both the best-fit parameters
    and their uncertainties.
    
    The two equations are:
      Equation 1: dVX = a * VpxBx + b * VpxBy + c
      Equation 2: dVY = a * VpxBy - b * VpxBx + d
    
    We build an independent variable array for ODR of shape (2, 2N) where the first N columns 
    correspond to Equation 1 and the next N to Equation 2.
    
    Parameters:
      - Vp: numpy array of proton velocities (in km/s; converted internally to m/s).
      - B: numpy array of magnetic field measurements (in nT; converted internally to T).
      - dVX, dVY: numpy arrays of differential voltages (V).
      - robust (bool): If True, perform one iteration of robust reweighting based on residuals.
      - rejection_threshold (float): Multiplier (in MAD units) for rejecting outliers.
      
    Returns:
      - a, b, c, d: fitted parameters (a and b in meters, c and d in volts).
      - sigma_a, sigma_b, sigma_c, sigma_d: estimated standard errors for the fitted parameters.
    """
    # Convert Vp (km/s) to m/s and B (nT) to T.
    Vp_m_per_s = Vp * u.m / u.s
    B_tesla = B * u.T

    # Compute -Vp x B (units: V/m)
    VxB = -np.cross(Vp_m_per_s, B_tesla)
    VpxBx = VxB[:, 0].to(u.V / u.m).value
    VpxBy = VxB[:, 1].to(u.V / u.m).value

    N = Vp.shape[0]
    
    # Build independent variable array for ODR:
    # For Equation 1: dVX = a * VpxBx + b * VpxBy + c
    # For Equation 2: dVY = a * VpxBy - b * VpxBx + d
    # We form x_odr as a 2 x (2N) array.
    x_odr = np.empty((2, 2 * N))
    # First N columns: Equation 1
    x_odr[0, :N] = VpxBx
    x_odr[1, :N] = VpxBy
    # Next N columns: Equation 2 (using the same independent variables)
    x_odr[0, N:] = VpxBx
    x_odr[1, N:] = VpxBy
    
    # Construct the dependent variable vector.
    y_odr = np.empty(2 * N)
    y_odr[:N] = dVX
    y_odr[N:] = dVY

    # Define the model function. It splits the 2N data points into two halves.
    def coupled_model(p, x):
        # p = [a, b, c, d]
        n_pts = x.shape[1] // 2  # number of data points per equation
        y_fit = np.empty(x.shape[1])
        # Equation 1 for the first n_pts points:
        y_fit[:n_pts] = p[0] * x[0, :n_pts] + p[1] * x[1, :n_pts] + p[2]
        # Equation 2 for the next n_pts points:
        y_fit[n_pts:] = p[0] * x[1, n_pts:] - p[1] * x[0, n_pts:] + p[3]
        return y_fit

    model = Model(coupled_model)

    # Obtain an initial guess using TLS (SVD) on the original design matrix.
    A = np.empty((2 * N, 4))
    A[0::2, 0] = VpxBx
    A[0::2, 1] = VpxBy
    A[0::2, 2] = 1
    A[0::2, 3] = 0
    A[1::2, 0] = VpxBy
    A[1::2, 1] = -VpxBx
    A[1::2, 2] = 0
    A[1::2, 3] = 1
    y_vec = np.empty(2 * N)
    y_vec[0::2] = dVX
    y_vec[1::2] = dVY
    augmented_matrix = np.hstack((A, y_vec.reshape(-1, 1)))
    U, S, Vt = np.linalg.svd(augmented_matrix, full_matrices=False)
    v = Vt.T[:, -1]
    if np.isclose(v[-1], 0):
        raise ValueError("The TLS solution is undefined (small singular value).")
    p0 = -v[0:4] / v[4]
    
    # Set up and run ODR.
    data = Data(x_odr, y_odr)
    odr_obj = ODR(data, model, beta0=p0)
    odr_output = odr_obj.run()

    # Optional robust reweighting.
    if robust:
        residuals = coupled_model(odr_output.beta, x_odr) - y_odr
        abs_res = np.abs(residuals)
        median_res = np.median(abs_res)
        mad = np.median(np.abs(abs_res - median_res))
        threshold = rejection_threshold * (mad if mad > 0 else 1e-6)
        mask = abs_res < threshold
        if np.sum(mask) >= 4:  # require a minimum number of points
            x_in = x_odr[:, mask]
            y_in = y_odr[mask]
            data_in = Data(x_in, y_in)
            odr_obj_in = ODR(data_in, model, beta0=odr_output.beta)
            odr_output = odr_obj_in.run()

    a, b, c, d = odr_output.beta
    sigma_a, sigma_b, sigma_c, sigma_d = odr_output.sd_beta

    return a, b, c, d, sigma_a, sigma_b, sigma_c, sigma_d

def invert_parameters_to_calibration_coefficients(a, b, c, d):
    """
    Invert the model parameters to obtain calibration coefficients.

    Parameters:
      - a, b: effective dipole components (meters).
      - c, d: offset voltages (volts).

    Returns:
      - Leff: effective dipole length (meters).
      - theta: rotation angle (degrees), computed robustly with arctan2.
      - c, d: offset voltages (volts).
    """
    Leff = np.sqrt(a**2 + b**2)
    theta = np.degrees(np.arctan2(b, a))
    return Leff, theta, c, d

def compute_cross_correlation(Ex, Ey, VxB_x, VxB_y):
    """
    Compute the cross-correlation between calibrated E-fields and -V x B.

    Parameters:
      - Ex, Ey: calibrated electric field components (V/m).
      - VxB_x, VxB_y: components of -V x B (V/m).

    Returns:
      - Cxx, Cyy: cross-correlation coefficients.
    """
    Ex_zero_mean = Ex - np.mean(Ex)
    Ey_zero_mean = Ey - np.mean(Ey)
    VxB_x_zero_mean = VxB_x - np.mean(VxB_x)
    VxB_y_zero_mean = VxB_y - np.mean(VxB_y)
    Cxx = np.corrcoef(Ex_zero_mean, VxB_x_zero_mean)[0, 1]
    Cyy = np.corrcoef(Ey_zero_mean, VxB_y_zero_mean)[0, 1]
    return Cxx, Cyy

def synchronize_merge_dfs(bdf, vdf, edf):
    """
    Synchronize and merge magnetic field, velocity, and electric field DataFrames.
    Assumes that external functions (e.g., func.synchronize_dfs) are defined.
    """
    edf, _ = func.synchronize_dfs(edf, bdf, False)
    edf, _ = func.synchronize_dfs(edf, vdf, False)
    bdf, vdf = func.synchronize_dfs(bdf, vdf, False)
    fin_data = edf.copy()
    fin_data[['Vx', 'Vy', 'Vz']] = vdf
    fin_data[['Bx', 'By', 'Bz']] = bdf
    return fin_data.interpolate().dropna()

def process_data(bdf, vdf, edf,
                 cadence_seconds      = None,
                 fit_interval_minutes = 4,
                 stride_minutes       = 4,   # non-overlapping intervals for piecewise calibration
                 min_correlation      = 0.5,
                 apply_hampel         = True,
                 window_size          = 501,
                 n                    = 3,
                 robust_fit           = False,
                 apply_lowpass        = True,
                 cutoff_frequency     = None,    # in Hz
                 fir_numtaps          = 101,     # number of filter coefficients (taps) for FIR design
                 fir_window           = 'hamming',
                 rel_uncertainty_thresh = 0.5):  # relative uncertainty threshold (e.g., 50%)
    """
    Process the data to compute calibration coefficients over short, non-overlapping intervals.
    
    Improvements include:
      (1) Optionally applying a zero-phase FIR low-pass filter to the velocity data.
      (2) Downsampling to a fixed cadence if cadence_seconds is provided.
      (3) Optional robust TLS/ODR fitting (with iterative reweighting) and Hampel filtering on E-field data.
      (4) A quality check based on the relative uncertainties of the fitted parameters.
          If the maximum relative uncertainty (sigma/|parameter|) exceeds rel_uncertainty_thresh,
          the calibration for that interval is discarded (set to NaN). The fraction of intervals discarded is returned.
    
    Parameters:
      bdf, vdf, edf       : DataFrames containing magnetic, velocity, and electric field data.
      cadence_seconds     : Desired cadence (in seconds) for downsampling.
      fit_interval_minutes: Interval length (in minutes) for computing calibration coefficients.
      stride_minutes      : Time stride (in minutes) for non-overlapping calibration intervals.
      min_correlation     : Minimum cross-correlation threshold required to validate calibration coefficients.
      apply_hampel        : Flag to apply Hampel filtering on the E-field data.
      window_size, n      : Parameters for the Hampel filter.
      robust_fit          : Flag to run the TLS/ODR fitting function in robust mode.
      apply_lowpass       : Flag to apply the FIR low-pass filter on velocity data.
      cutoff_frequency    : Cutoff frequency (in Hz) for the FIR low-pass filter.
      fir_numtaps         : Number of taps for the FIR filter.
      fir_window          : Window type used for FIR design.
      rel_uncertainty_thresh : Relative uncertainty threshold above which a fit is considered unreliable.
      
    Returns:
      A tuple:
        - DataFrame containing calibration coefficients with interval boundaries and quality flags.
          Additional columns include sigma_a, sigma_b, sigma_c, sigma_d (fit uncertainties) and a 'discarded' flag.
        - The fraction of intervals that were discarded due to high relative uncertainty.
    """
    
    # --- Helper: Zero-Phase FIR Low-Pass Filter ---
    def fir_lowpass_filter(data, cutoff, fs, numtaps, window):
        nyq = 0.5 * fs  # Nyquist Frequency
        normalized_cutoff = cutoff / nyq
        taps = firwin(numtaps, normalized_cutoff, window=window)
        filtered_data = filtfilt(taps, [1.0], data)
        return filtered_data

    # --- Step 1: Merge and Synchronize Data ---
    averaged_data = synchronize_merge_dfs(bdf, vdf, edf)
    
    # Determine cadence_seconds if not provided.
    if cadence_seconds is None:
        cadence_seconds = func.find_cadence(vdf)
    
    fs = 1.0 / cadence_seconds  # sampling frequency in Hz
    
    # --- Step 2: Apply FIR Low-Pass Filter (if requested) ---
    if apply_lowpass and (cutoff_frequency is not None):
        print("Applying FIR low-pass filter to velocity data with cutoff frequency {} Hz".format(cutoff_frequency))
        for col in ['Vx', 'Vy', 'Vz']:
            averaged_data[col] = fir_lowpass_filter(averaged_data[col].values,
                                                    cutoff_frequency, fs,
                                                    numtaps=fir_numtaps,
                                                    window=fir_window)
    
    # --- Step 3: Downsample Data ---
    if cadence_seconds is not None:
        print("Downsampling data to calibrate E-field at {} seconds cadence".format(cadence_seconds))
        averaged_data = averaged_data.resample(f"{int(cadence_seconds)}S").mean().dropna()
    
    # --- Step 4: Optionally Apply Hampel Filter to E-field Data ---
    if apply_hampel:
        for column in edf.columns:
            try:
                filtered_arr, outliers_indices = func.hampel(edf[column], window_size, n)
                edf[column] = filtered_arr
            except Exception as e:
                pass
    
    # --- Step 5: Convert Units and Prepare Data Arrays ---
    Vp = averaged_data[['Vx', 'Vy', 'Vz']].values * 1e3  # Convert km/s to m/s.
    B  = averaged_data[['Bx', 'By', 'Bz']].values * 1e-9    # Convert nT to T.
    dVX = averaged_data['dvx'].values  # Electric field component (V)
    dVY = averaged_data['dvy'].values  # Electric field component (V)
    times = averaged_data.index.values
    N = len(averaged_data)
    
    points_per_interval = max(int((fit_interval_minutes * 60) / cadence_seconds), 1)
    points_per_stride   = max(int((stride_minutes * 60) / cadence_seconds), 1)
    
    results = []
    num_intervals = int((N - points_per_interval) / points_per_stride) + 1
    discard_flags = []  # to track which intervals are discarded

    for i in range(num_intervals):
        start_idx = i * points_per_stride
        end_idx = start_idx + points_per_interval
        if end_idx > N:
            break

        dVX_interval = dVX[start_idx:end_idx]
        dVY_interval = dVY[start_idx:end_idx]
        Vp_interval  = Vp[start_idx:end_idx]
        B_interval   = B[start_idx:end_idx]
        time_interval = times[start_idx:end_idx]
        
        try:
            # Fit using the revised coupled linear model with ODR.
            a, b, c, d, sigma_a, sigma_b, sigma_c, sigma_d = fit_coupled_linear_model(
                Vp_interval, B_interval, dVX_interval, dVY_interval, robust=robust_fit)
            
            Leff, theta, _, _ = invert_parameters_to_calibration_coefficients(a, b, c, d)

            # Compute the calibrated E-field components (V/m).
            Ex = ((-a * c + a * dVX_interval + b * d - b * dVY_interval) /
                  (a**2 + b**2))
            Ey = ((-a * d + a * dVY_interval - b * c + b * dVX_interval) /
                  (a**2 + b**2))

            # Compute -V x B for cross-correlation.
            VxB = -np.cross(Vp_interval, B_interval)
            VxB_x = VxB[:, 0]
            VxB_y = VxB[:, 1]
            Cxx, Cyy = compute_cross_correlation(Ex, Ey, VxB_x, VxB_y)

            # Initialize quality flag.
            discarded = False

            # Quality check: compute relative uncertainties.
            eps = 1e-10
            r_a = sigma_a / (abs(a) + eps)
            r_b = sigma_b / (abs(b) + eps)
            r_c = sigma_c / (abs(c) + eps)
            r_d = sigma_d / (abs(d) + eps)
            max_rel_unc = max(r_a, r_b, r_c, r_d)

            if max_rel_unc > rel_uncertainty_thresh:
                # Mark the interval as unreliable.
                a_fit = b_fit = c_fit = d_fit = np.nan
                sigma_a_fit = sigma_b_fit = sigma_c_fit = sigma_d_fit = np.nan
                Leff = theta = np.nan
                Cxx = Cyy = np.nan
                discarded = True
            else:
                a_fit, b_fit, c_fit, d_fit = a, b, c, d
                sigma_a_fit, sigma_b_fit, sigma_c_fit, sigma_d_fit = sigma_a, sigma_b, sigma_c, sigma_d

            discard_flags.append(discarded)

            results.append({
                'interval_start': pd.to_datetime(time_interval[0]),
                'interval_end': pd.to_datetime(time_interval[-1]),
                'center_time': pd.to_datetime(time_interval[len(time_interval)//2]),
                'Leff': Leff,
                'theta': theta,
                'a': a_fit,
                'b': b_fit,
                'c': c_fit,
                'd': d_fit,
                'sigma_a': sigma_a_fit,
                'sigma_b': sigma_b_fit,
                'sigma_c': sigma_c_fit,
                'sigma_d': sigma_d_fit,
                'Cxx': Cxx,
                'Cyy': Cyy,
                'discarded': discarded
            })
        except Exception as ex:
            traceback.print_exc()
            discard_flags.append(True)
            results.append({
                'interval_start': pd.to_datetime(time_interval[0]),
                'interval_end': pd.to_datetime(time_interval[-1]),
                'center_time': pd.to_datetime(time_interval[len(time_interval)//2]),
                'Leff': np.nan,
                'theta': np.nan,
                'a': np.nan,
                'b': np.nan,
                'c': np.nan,
                'd': np.nan,
                'sigma_a': np.nan,
                'sigma_b': np.nan,
                'sigma_c': np.nan,
                'sigma_d': np.nan,
                'Cxx': np.nan,
                'Cyy': np.nan,
                'discarded': True
            })
    
    results_df = pd.DataFrame(results)
    results_df.set_index('center_time', inplace=True)
    # Calculate the fraction of intervals discarded.
    discard_fraction = np.mean(results_df['discarded'])
    
    # Interpolate over NaN values in the final DataFrame (excluding the 'discarded' flag)
    interp_cols = ['Leff', 'theta', 'a', 'b', 'c', 'd', 'sigma_a', 'sigma_b', 'sigma_c', 'sigma_d', 'Cxx', 'Cyy']
    results_df[interp_cols] = results_df[interp_cols].interpolate()

    return results_df, discard_fraction

def calibrate_data(edf, coeffs):
    """
    Calibrate high-cadence electric field data (edf) using piecewise calibration coefficients in a vectorized manner.
    
    For each high-frequency timestamp in edf, determine the corresponding calibration interval (using
    'interval_start' and 'interval_end' from coeffs) and apply the calibration formula:
    
        Ex = ((-a*c + a*dvx + b*d - b*dvy) / (a^2 + b^2)) * 1e3   [mV/m]
        Ey = ((-a*d + a*dvy - b*c + b*dvx) / (a^2 + b^2)) * 1e3   [mV/m]
    
    Parameters:
      edf: DataFrame with columns 'dvx' and 'dvy' and a datetime index.
      coeffs: DataFrame or tuple (DataFrame, discard_fraction) with calibration coefficients and interval boundaries.
              Must include columns: 'interval_start', 'interval_end', 'a', 'b', 'c', and 'd'.
              Assumes non-overlapping intervals.
              
    Returns:
      DataFrame with calibrated electric field components 'Ex' and 'Ey' (in mV/m) on the same index as edf.
      Timestamps outside valid calibration intervals are assigned NaN.
    """
    edf = edf.copy()
    edf.index = pd.to_datetime(edf.index)
    
    # If coeffs is a tuple, extract the DataFrame
    if isinstance(coeffs, tuple):
        coeffs = coeffs[0]
    
    coeffs = coeffs.sort_values('interval_start').reset_index(drop=True)
    
    for col in ['a', 'b', 'c', 'd']:
        coeffs[col] = coeffs[col].apply(lambda x: x.value if hasattr(x, 'value') else x)
    
    interval_start = coeffs['interval_start'].values.astype('datetime64[ns]')
    interval_end = coeffs['interval_end'].values.astype('datetime64[ns]')
    t_edf = edf.index.values.astype('datetime64[ns]')
    
    idx = np.searchsorted(interval_start, t_edf, side='right') - 1
    valid = (idx >= 0) & (t_edf <= interval_end[idx])
    
    n_points = len(t_edf)
    a_arr = np.full(n_points, np.nan, dtype=float)
    b_arr = np.full(n_points, np.nan, dtype=float)
    c_arr = np.full(n_points, np.nan, dtype=float)
    d_arr = np.full(n_points, np.nan, dtype=float)
    
    valid_idx = np.where(valid)[0]
    selected_idx = idx[valid_idx]
    a_arr[valid_idx] = coeffs['a'].values[selected_idx]
    b_arr[valid_idx] = coeffs['b'].values[selected_idx]
    c_arr[valid_idx] = coeffs['c'].values[selected_idx]
    d_arr[valid_idx] = coeffs['d'].values[selected_idx]
    
    dvx = edf['dvx'].values
    dvy = edf['dvy'].values
    denom = a_arr**2 + b_arr**2
    invalid_denom = (denom == 0)
    
    Ex = np.full(n_points, np.nan, dtype=float)
    Ey = np.full(n_points, np.nan, dtype=float)
    
    valid_calc = ~np.isnan(a_arr) & ~invalid_denom
    Ex[valid_calc] = ((-a_arr[valid_calc]*c_arr[valid_calc] +
                       a_arr[valid_calc]*dvx[valid_calc] +
                       b_arr[valid_calc]*d_arr[valid_calc] -
                       b_arr[valid_calc]*dvy[valid_calc]) / denom[valid_calc]) * 1e3
    Ey[valid_calc] = ((-a_arr[valid_calc]*d_arr[valid_calc] +
                       a_arr[valid_calc]*dvy[valid_calc] -
                       b_arr[valid_calc]*c_arr[valid_calc] +
                       b_arr[valid_calc]*dvx[valid_calc]) / denom[valid_calc]) * 1e3
    
    Edf = pd.DataFrame({'Ex': Ex, 'Ey': Ey}, index=edf.index)
    return Edf.dropna().interpolate()



def find_longest_intervals(df, thresh, M, buffer_seconds= 120):
    # Create a boolean mask where the absolute value of Bz is greater than thresh
    mask = df['Bz'].abs() > thresh

    # Identify the start of new intervals
    df['start_interval'] = (mask & ~mask.shift(1, fill_value=False))

    # Group contiguous True values together
    df['group'] = df['start_interval'].cumsum() * mask

    # Get the start and end of each interval
    grouped = df[df['group'] != 0].groupby('group')
    intervals = pd.DataFrame({
        'start_date': grouped.apply(lambda x: x.index.min()),
        'end_date': grouped.apply(lambda x: x.index.max())
    })

    # Adjust start_date and end_date by adding buffer_seconds after the start and before the end
    intervals['adjusted_start_date'] = intervals['start_date'] + pd.Timedelta(seconds=buffer_seconds)
    intervals['adjusted_end_date'] = intervals['end_date'] - pd.Timedelta(seconds=buffer_seconds)

    # Ensure that adjusted_start_date <= adjusted_end_date
    intervals = intervals[intervals['adjusted_start_date'] <= intervals['adjusted_end_date']]

    # Ensure that adjusted dates are within the DataFrame's index range
    min_index_date = df.index.min()
    max_index_date = df.index.max()
    intervals['adjusted_start_date'] = intervals['adjusted_start_date'].apply(lambda x: max(x, min_index_date))
    intervals['adjusted_end_date'] = intervals['adjusted_end_date'].apply(lambda x: min(x, max_index_date))

    # Get corresponding indices from the original DataFrame
    intervals['adjusted_start_index'] = df.index.get_indexer(intervals['adjusted_start_date'], method='nearest')
    intervals['adjusted_end_index'] = df.index.get_indexer(intervals['adjusted_end_date'], method='nearest')

    # Calculate the length and duration in seconds for each adjusted interval
    intervals['length'] = intervals['adjusted_end_index'] - intervals['adjusted_start_index'] + 1
    intervals['duration_seconds'] = (intervals['adjusted_end_date'] - intervals['adjusted_start_date']).dt.total_seconds()

    # Sort intervals by length in descending order
    intervals_sorted = intervals.sort_values('length', ascending=False)

    # Return the top M intervals
    return intervals_sorted.head(M)


import numpy as np
import pandas as pd
import traceback

def estimate_Ez(B_df, E_df, min_bz=1, window_size=51, n=2, apply_hampel=True):
    """
    Estimate the missing Ez component of the electric field using the condition E · B = 0.
    
    Assumptions on units:
      - B_df: DataFrame with magnetic field components (Bx, By, Bz) in nanotesla (nT).
      - E_df: DataFrame with electric field components (Ex, Ey) in millivolts per meter (mV/m).
    The computed Ez will be in mV/m.
    
    The relation used is:
        Ez = (-Bx * Ex - By * Ey) / Bz
    This ensures that the total electric field is perpendicular to the magnetic field.
    
    Procedure:
      1. For numerical stability, any Bz values with absolute magnitude less than min_bz are set to NaN.
      2. Ez is computed with the above formula.
      3. The computed Ez is assigned to E_df (in a new column 'Ez'), after which missing values are interpolated and dropped.
      4. Optionally, a Hampel filter is applied to each column of E_df to remove outliers.
    
    Parameters:
      B_df (DataFrame): Magnetic field DataFrame with columns representing Bx, By, Bz in nT.
      E_df (DataFrame): Electric field DataFrame with columns representing Ex, Ey in mV/m.
      min_bz (float): Minimum absolute value of Bz considered valid (default 1 nT).
      window_size (int): Window size for the Hampel filter (default 51).
      n (int): Number of standard deviations for outlier detection in the Hampel filter (default 2).
      apply_hampel (bool): If True, apply the Hampel filter to each column of E_df.
    
    Returns:
      DataFrame: The updated E_df with an additional column 'Ez', with all field components in mV/m.
    """
    # --- Step 1: In-place modification of Bz ---
    # Identify the column corresponding to Bz (assumed to be the third column)
    Bz_col = B_df.columns[2]
    Bz = B_df[Bz_col]
    # Set values where |Bz| is less than min_bz to NaN (to avoid unstable divisions)
    mask = np.abs(Bz) < min_bz
    B_df.loc[mask, Bz_col] = np.nan

    # --- Step 2: Extract field components without making unnecessary copies ---
    Bx = B_df.iloc[:, 0]
    By = B_df.iloc[:, 1]
    Bz = B_df.iloc[:, 2]
    Ex = E_df.iloc[:, 0]
    Ey = E_df.iloc[:, 1]

    # --- Step 3: Compute Ez ---
    # Use the relation: E_z = (-B_x * E_x - B_y * E_y) / B_z.
    # With B in nT and E in mV/m, the resulting Ez is in mV/m.
    Ez = (-Bx * Ex - By * Ey) / Bz

    # Assign the computed Ez to E_df (in-place) and interpolate missing values.
    E_df['Ez'] = Ez
    E_df = E_df.interpolate().dropna()
    
    # --- Step 4: Optionally apply Hampel filter to each column for outlier removal ---
    if apply_hampel:
        for column in E_df.columns:
            try:
                print(f"Processing column: {column}")
                filtered_arr, outliers_indices = func.hampel(E_df[column], window_size, n)
                print(f"Identified {len(outliers_indices)} outliers in column: {column}")
                E_df[column] = filtered_arr
            except Exception:
                traceback.print_exc()
                
    return E_df






# # -----------------------------------------------------------------------------
# # Coordinate helpers
# # -----------------------------------------------------------------------------

# def project_dV(dV12: np.ndarray, dV34: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Rotate differential voltages from *whip* coordinates into spacecraft (SC).

#     Parameters
#     ----------
#     dV12, dV34 : ndarray, shape (N,)
#         Differential voltages from probe pairs (V).

#     Returns
#     -------
#     dVx, dVy : ndarray, shape (N,)
#         Voltages in SC X and Y directions (V).
#     """
#     R_V_TO_SC = np.array([[0.64524, -0.82228],
#                           [0.76897,  0.57577]])
#     dV_sc = R_V_TO_SC @ np.vstack((dV12, dV34))
#     return dV_sc[0], dV_sc[1]

# # -----------------------------------------------------------------------------
# # Filtering helpers
# # -----------------------------------------------------------------------------

# def _butter_lowpass(cutoff: float, fs: float, order: int = 5):
#     nyq = 0.5 * fs
#     return butter(order, cutoff / nyq, btype="low", analog=False)


# def apply_lowpass_filter(arr: np.ndarray, cutoff: float, fs: float,
#                           order: int = 5) -> np.ndarray:
#     """Zero‑phase Butterworth low‑pass using *filtfilt*."""
#     b, a = _butter_lowpass(cutoff, fs, order)
#     return filtfilt(b, a, arr)

# # -----------------------------------------------------------------------------
# # Robust percentile clip + interpolate
# # -----------------------------------------------------------------------------

# def percentile_filter_interpolate_ts(df: pd.DataFrame | pd.Series,
#                                      low_pct: float = 0,
#                                      hi_pct: float = 99.9) -> pd.DataFrame:
#     """Column‑wise percentile clipping followed by linear interpolation.

#     Any value outside the [`low_pct`, `hi_pct`] range is set to *NaN* then filled.
#     """
#     out = df.copy()
#     if isinstance(out, pd.Series):
#         out = out.to_frame()
#     for col in out.columns:
#         s = out[col]
#         valid = s.dropna()
#         if valid.empty:
#             continue
#         lo, hi = np.percentile(valid, [low_pct, hi_pct])
#         s[(s < lo) | (s > hi)] = np.nan
#         out[col] = s.interpolate(limit_direction="both")
#     return out if isinstance(df, pd.DataFrame) else out.iloc[:, 0]

# # -----------------------------------------------------------------------------
# # Window‑level fit (private)
# # -----------------------------------------------------------------------------

# def _fit_window(start_ns: int, end_ns: int,
#                 t: np.ndarray,
#                 VxB_x: np.ndarray, VxB_y: np.ndarray,
#                 dVx: np.ndarray, dVy: np.ndarray,
#                 robust: bool = True):
#     """Internal routine: least‑squares fit of the 4‑parameter model on one window."""
#     mask = (t >= start_ns) & (t <= end_ns)
#     if mask.sum() < 10:
#         return None  # not enough points
#     X1, X2 = VxB_x[mask], VxB_y[mask]
#     Y1, Y2 = dVx[mask], dVy[mask]

#     def residual(p):
#         a, b, c, d = p
#         r1 = Y1 - (a * X1 + b * X2 + c)
#         r2 = Y2 - (a * X2 - b * X1 + d)
#         return np.hstack([r1, r2])

#     # OLS initial guess on first equation
#     A = np.vstack([X1, X2, np.ones_like(X1)]).T
#     beta, *_ = np.linalg.lstsq(A, Y1, rcond=None)
#     p0 = [beta[0], beta[1], beta[2], 0.0]

#     res = least_squares(residual, p0,
#                         loss="huber" if robust else "linear")
#     a, b, c, d = res.x
#     q = 1.0 / (1.0 + np.sqrt(res.cost / res.fun.size))  # quality 0–1
#     return start_ns, end_ns, a, b, c, d, q

# # -----------------------------------------------------------------------------
# # Main calibration pipeline
# # -----------------------------------------------------------------------------




# def calibrate_electric_field(edf: pd.DataFrame,
#                              vdf: pd.DataFrame,
#                              bdf: pd.DataFrame,
#                              window: str                   = "30s",
#                              overlap: float                = 0.9,
#                              lowpass_hz: float | None      = None,
#                              lowpass_order: int            = 5,
#                              pct_clip: Tuple[float, float] = (0, 99.9),
#                              robust_ls: bool               = True,
#                              n_jobs: int                   = -1,
#                              func_module                   = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Calibrate *dvx,dvy* to *Ex,Ey* using overlapping Hann‑weighted windows.

#     Parameters
#     ----------
#     edf : DataFrame
#         High‑rate differential voltages with columns ``dvx`` and ``dvy`` (V).
#     vdf : DataFrame
#         Spacecraft velocity ``Vx,Vy,Vz`` (km/s).
#     bdf : DataFrame
#         Magnetic field ``Bx,By,Bz`` (nT).
#     window : str, default "30s"
#         Window length – any pandas offset alias.
#     overlap : float, default 0.9
#         Fractional overlap between successive windows (0 ≤ overlap < 1).
#     lowpass_hz : float or None
#         Apply zero‑phase Butterworth low‑pass to both *V* and *B* before the fit.
#     lowpass_order : int, default 5
#         Filter order for ``lowpass_hz``.
#     pct_clip : (low, high)
#         Percentile bounds for final Ex/Ey clipping.
#     robust_ls : bool, default True
#         Use Huber loss in ``scipy.optimize.least_squares``.
#     n_jobs : int, default ‑1
#         Parallel workers (joblib).  *‑1* → all cores.
#     func_module : module or None
#         Module providing ``synchronize_dfs``.  If ``None`` we try to import
#         ``general_functions as func``.

#     Returns
#     -------
#     E_cal : DataFrame
#         Calibrated ``Ex,Ey`` (mV/m) on the edf index.
#     coeffs : DataFrame
#         Window‑level coefficients ``a,b,c,d`` plus quality weights.
#     """

#     try:
#         # --- 0. housekeeping -----------------------------------------------------
#         if func_module is None:
#             import general_functions as func_module  # type: ignore
    
#         edf, _ = func_module.synchronize_dfs(edf, vdf, False)
#         edf, _ = func_module.synchronize_dfs(edf, bdf, False)
#         vdf, bdf = func_module.synchronize_dfs(vdf, bdf, False)
    
#         df = edf.copy()
#         df[["Vx", "Vy", "Vz"]] = vdf
#         df[["Bx", "By", "Bz"]] = bdf
#         df = df.dropna()
    
#         # --- 1. optional low‑pass -------------------------------------------------
#         if lowpass_hz is not None:
#             dt = np.median(np.diff(df.index.view("int64"))) * 1e-9  # seconds
#             fs = 1.0 / dt
#             for col in ("Vx", "Vy", "Vz", "Bx", "By", "Bz"):
#                 df[col] = apply_lowpass_filter(df[col].values, lowpass_hz, fs,
#                                                order=lowpass_order)
    
#         # --- 2. pre‑compute arrays ----------------------------------------------
#         V = df[["Vx", "Vy", "Vz"]].values * 1e3  # km/s → m/s
#         B = df[["Bx", "By", "Bz"]].values * 1e-9  # nT  → T
#         VxB = -np.cross(V, B)
#         VxB_x, VxB_y = VxB[:, 0], VxB[:, 1]
#         dVx = df["dvx"].values
#         dVy = df["dvy"].values
#         t_ns = df.index.view("int64")
    
#         # --- 3. window list ------------------------------------------------------
#         win_ns = pd.to_timedelta(window).value
#         step_ns = int(win_ns * (1 - overlap))
#         starts = np.arange(t_ns[0], t_ns[-1] - win_ns + 1, step_ns)
#         ends = starts + win_ns
    
#         # --- 4. fit all windows in parallel -------------------------------------
#         coeff_cols = ["start", "end", "a", "b", "c", "d", "q"]
#         results: List[Tuple] = Parallel(n_jobs=n_jobs)(
#             delayed(_fit_window)(s, e, t_ns, VxB_x, VxB_y, dVx, dVy, robust_ls)
#             for s, e in zip(starts, ends)
#         )
#         coeffs = pd.DataFrame([r for r in results if r is not None],
#                               columns=coeff_cols)
#         if coeffs.empty:
#             raise RuntimeError("No successful window fits – check input data.")
    
#         # --- 5. overlap‑add synthesis -------------------------------------------
#         n_pts = len(df)
#         Ex_sum = np.zeros(n_pts)
#         Ey_sum = np.zeros(n_pts)
#         W_sum = np.zeros(n_pts)
    
#         for row in coeffs.itertuples(index=False):  # type: ignore
#             s, e, a, b, c, d, q = row
#             mask = (t_ns >= s) & (t_ns <= e)
#             if not mask.any():
#                 continue
#             idx = np.where(mask)[0]
#             tau = (t_ns[idx].astype("float64") - s) / win_ns  # 0–1
#             w = 0.5 * (1 - np.cos(2 * np.pi * tau)) * q        # Hann × quality
#             denom = a * a + b * b
#             Ex_win = ((-a * c + a * dVx[idx] + b * d - b * dVy[idx]) / denom) * 1e3
#             Ey_win = ((-a * d + a * dVy[idx] - b * c + b * dVx[idx]) / denom) * 1e3
#             Ex_sum[idx] += w * Ex_win
#             Ey_sum[idx] += w * Ey_win
#             W_sum[idx] += w
    
#         valid = W_sum > 0
#         Ex = np.full(n_pts, np.nan)
#         Ey = np.full(n_pts, np.nan)
#         Ex[valid] = Ex_sum[valid] / W_sum[valid]
#         Ey[valid] = Ey_sum[valid] / W_sum[valid]
    
#         E_cal = pd.DataFrame({"Ex": Ex, "Ey": Ey}, index=df.index)
    
#         # --- 6. final percentile clip + fill ------------------------------------
#         E_cal = percentile_filter_interpolate_ts(E_cal, *pct_clip)
#         E_cal = E_cal.ffill().bfill()
#     except:
#         traceback.print_exc()

#     return E_cal, coeffs

# # -----------------------------------------------------------------------------
# # Interval utilities
# # -----------------------------------------------------------------------------

# def find_longest_intervals(df: pd.DataFrame, thresh: float, M: int,
#                            buffer_s: int = 120) -> pd.DataFrame:
#     """Return the *M* longest contiguous intervals with |Bz| > *thresh* nT.

#     A ±buffer is trimmed off each end, and intervals are clipped to the data span.
#     """
#     mask = df["Bz"].abs() > thresh
#     df = df.copy()
#     df["interval_start"] = mask & ~mask.shift(1, fill_value=False)
#     df["group"] = (df["interval_start"].cumsum() * mask)
#     grouped = df[df["group"] != 0].groupby("group")

#     intervals = pd.DataFrame({
#         "start": grouped.apply(lambda x: x.index.min()),
#         "end": grouped.apply(lambda x: x.index.max())
#     })
#     intervals["start"] += pd.Timedelta(seconds=buffer_s)
#     intervals["end"] -= pd.Timedelta(seconds=buffer_s)
#     intervals = intervals[intervals["start"] <= intervals["end"]]

#     # indices + duration
#     intervals["duration_s"] = (intervals["end"] - intervals["start"]).dt.total_seconds()
#     return intervals.sort_values("duration_s", ascending=False).head(M)

# # -----------------------------------------------------------------------------
# # Ez reconstruction
# # -----------------------------------------------------------------------------

# def estimate_Ez(B_df: pd.DataFrame, E_df: pd.DataFrame,
#                 min_bz: float = 1.0,
#                 hampel_window: int = 51,
#                 hampel_n: int = 2,
#                 func_module=None) -> pd.DataFrame:
#     """Compute missing Ez under the assumption *E·B = 0* and optional Hampel filter.

#     All inputs/outputs are assumed in units:
#     * B – nT;  E – mV/m;  Ez returned in mV/m.
#     """
#     if func_module is None:
#         import general_functions as func_module  # type: ignore

#     B_df = B_df.copy()
#     E_df = E_df.copy()
#     Bz = B_df.iloc[:, 2]
#     B_df.loc[Bz.abs() < min_bz, B_df.columns[2]] = np.nan

#     Bx, By, Bz = (B_df.iloc[:, i] for i in range(3))
#     Ex, Ey = (E_df.iloc[:, i] for i in range(2))
#     Ez = (-Bx * Ex - By * Ey) / Bz
#     E_df["Ez"] = Ez
#     E_df = E_df.interpolate().dropna()

#     # optional Hampel outlier removal
#     for col in E_df.columns:
#         try:
#             clean, _ = func_module.hampel(E_df[col], hampel_window, hampel_n)
#             E_df[col] = clean
#         except Exception:
#             pass

#     return E_df

# __all__: List[str] = [
#     "project_dV",
#     "apply_lowpass_filter",
#     "percentile_filter_interpolate_ts",
#     "calibrate_electric_field",
#     "find_longest_intervals",
#     "estimate_Ez",
# ]  # for * import cleanliness

