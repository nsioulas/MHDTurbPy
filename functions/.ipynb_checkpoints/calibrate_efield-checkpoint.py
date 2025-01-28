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

from scipy.optimize import curve_fit
import astropy.units as u


# def fit_coupled_linear_model(Vp, B, dVX, dVY):
#     """
#     Fit the projected differential voltages using the coupled four-parameter linear model.

#     Parameters:
#     - Vp: numpy array of proton velocities (km/s), shape (N, 3).
#     - B: numpy array of magnetic field measurements (nT), shape (N, 3).
#     - dVX, dVY: numpy arrays of differential voltages in SC coordinates (V), shape (N,).

#     Returns:
#     - a, b, c, d: fitted parameters of the model (a and b in meters, c and d in volts).
#     """
#     # Convert Vp from km/s to m/s and B from nT to T
#     Vp_m_per_s = Vp * u.m / u.s  # Convert to Quantity with units
#     B_tesla    = B * u.T  # Convert to Quantity with units

#     # Compute -Vp x B (units: V/m)
#     VxB = -np.cross(Vp_m_per_s, B_tesla)  # Units: (m/s) x (T) = V/m

#     # Extract x and y components and convert to numeric values in V/m
#     VpxBx = VxB[:, 0].to(u.V / u.m).value
#     VpxBy = VxB[:, 1].to(u.V / u.m).value

#     # Define the model for dVX and dVY
#     def model(xdata, a, b, c, d):
#         VpxBx, VpxBy = xdata
#         dVX = a * VpxBx + b * VpxBy + c
#         dVY = -b * VpxBx + a * VpxBy + d
#         return np.concatenate([dVX, dVY])

#     # Stack the independent variables into one array
#     xdata = np.vstack((VpxBx, VpxBy))

#     # Concatenate the dependent variables (observed dVX and dVY)
#     ydata = np.concatenate([dVX, dVY])

#     # Initial guess for parameters (a, b, c, d)
#     initial_params = [1, 1, 0, 0]

#     # Perform the curve fitting
#     params_opt, _ = curve_fit(model, xdata, ydata, p0=initial_params)

#     # Unpack the optimized parameters
#     a_opt, b_opt, c_opt, d_opt = params_opt

#     return a_opt, b_opt, c_opt, d_opt


def fit_coupled_linear_model(Vp, B, dVX, dVY):
    """
    Fit the projected differential voltages using the coupled four-parameter linear model
    via Total Least Squares (TLS).

    Parameters:
    - Vp: numpy array of proton velocities (km/s), shape (N, 3).
    - B: numpy array of magnetic field measurements (nT), shape (N, 3).
    - dVX, dVY: numpy arrays of differential voltages in SC coordinates (V), shape (N,).

    Returns:
    - a, b, c, d: fitted parameters of the model (a and b in meters, c and d in volts).
    """
    # Convert Vp from km/s to m/s and B from nT to T
    Vp_m_per_s = Vp * u.m / u.s  # Convert to Quantity with units
    B_tesla    = B * u.T  # Convert to Quantity with units

    # Compute -Vp x B (units: V/m)
    VxB = -np.cross(Vp_m_per_s, B_tesla)  # Units: (m/s) x (T) = V/m

    # Extract x and y components and convert to numeric values in V/m
    VpxBx = VxB[:, 0].to(u.V / u.m).value
    VpxBy = VxB[:, 1].to(u.V / u.m).value
    
    
    # Number of data points
    N = Vp.shape[0]

    # Construct the design matrix A and observation vector y
    # For TLS, we need to consider errors in both A and y
    # Formulate the augmented matrix [A | y]
    # Each data point contributes two rows to A and y

    # Initialize A (2N x 4) and y (2N)
    A = np.zeros((2 * N, 4))
    y = np.zeros(2 * N)

    # Populate A and y
    for i in range(N):
        # First equation: dVX = a * VpxBx + b * VpxBy + c
        A[2 * i] = [VpxBx[i], VpxBy[i], 1, 0]
        y[2 * i] = dVX[i]
        
        # Second equation: dVY = -b * VpxBx + a * VpxBy + d
        A[2 * i + 1] = [VpxBy[i], -VpxBx[i], 0, 1]
        y[2 * i + 1] = dVY[i]

    # Form the augmented matrix [A | y]
    augmented_matrix = np.hstack((A, y.reshape(-1, 1)))  # Shape: (2N x 5)

    # Perform Singular Value Decomposition (SVD) on the augmented matrix
    U, S, Vt = np.linalg.svd(augmented_matrix, full_matrices=False)
    V = Vt.T  # Transpose to get V

    # The solution is the last column of V corresponding to the smallest singular value
    v = V[:, -1]

    # Check if the last element is non-zero to avoid division by zero
    if np.isclose(v[-1], 0):
        raise ValueError("The TLS solution is undefined (small singular value).")

    # Extract the parameters: p = [a, b, c, d] = -v[0:4] / v[4]
    p = -v[0:4] / v[4]

    a_opt, b_opt, c_opt, d_opt = p

    return a_opt, b_opt, c_opt, d_opt



def invert_parameters_to_calibration_coefficients(a, b, c, d):
    """
    Invert the model parameters to obtain calibration coefficients.

    Parameters:
    - a, b: effective dipole components (meters).
    - c, d: offset voltages (volts).

    Returns:
    - Leff: effective dipole length (meters).
    - theta: rotation angle (degrees).
    - c, d: offset voltages (volts).
    """
    Leff   = np.sqrt(a**2 + b**2)        # meters
    theta  = np.degrees(np.arctan(b/a))  # degrees
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
    
    # Synchronize
    edf, _                        = func.synchronize_dfs(edf, bdf, False)
    edf, _                        = func.synchronize_dfs(edf, vdf, False)
    bdf, vdf                      = func.synchronize_dfs(bdf, vdf, False)

    # Merge
    fin_data                      = edf
    fin_data[['Vx', 'Vy', 'Vz']]  = vdf
    fin_data[['Bx', 'By', 'Bz']]  = bdf

    # Interpolate dropna
    return fin_data.interpolate().dropna()


def process_data(bdf,
                 vdf,
                 edf,
                 cadence_seconds      = 12,
                 fit_interval_minutes = 4,
                 stride_minutes       = 1,
                 min_correlation      = 0.5,
                 apply_hampel         = True,
                 window_size          = 501,
                 n                    = 3):
    """
    Process the data to compute calibration coefficients over sliding intervals.

    Parameters:
    - data: pandas DataFrame containing the data with a datetime index.
    - cadence_seconds: block averaging cadence in seconds (e.g., 12).
    - fit_interval_minutes: length of each fitting interval in minutes (e.g., 4).
    - stride_minutes: stride length between intervals in minutes (e.g., 1).
    - min_correlation: minimum acceptable cross-correlation value (e.g., 0.5).

    Returns:
    - DataFrame containing calibration coefficients and correlation metrics.
    """
    
    # Synchronize dfs
    
    # Apply Hampel filter if required
    if apply_hampel:
        # Determine which velocity components to filter based on data presence
        columns_for_hampel = edf.columns

        for column in columns_for_hampel:
            print(column)
            try:
                filtered_arr, outliers_indices = func.hampel(edf[column], window_size, n)
                print('Identified', len(outliers_indices),' outliers')
                #Edf.loc[Edf.index[outliers_indices], column] = np.nan
                edf[column] = filtered_arr
            except Exception as e:
                pass

    averaged_data   = synchronize_merge_dfs(bdf, vdf, edf)
    cadence_seconds = func.find_cadence(vdf)
    
    # # Find mov averages
    # averaged_data = block_average(data, cadence_seconds)
    
    #print(averaged_data)

    # Extract variables
    B       = (averaged_data[['Bx', 'By', 'Bz']].values  * 1e-9 * u.T).value    # T
    Vp      = (averaged_data[['Vx', 'Vy', 'Vz']].values  * 1e3 * u.m / u.s ).value # m/s

    # Project to SC coordinates
    dVX     = (averaged_data['dvx'].values.T * u.V).value # Volt
    dVY     = (averaged_data['dvy'].values.T * u.V).value # Volt

    times                = averaged_data.index.values
    N                    = len(averaged_data)
    points_per_interval  = int((fit_interval_minutes * 60) / cadence_seconds)
    points_per_stride    = int((stride_minutes * 60) / cadence_seconds)
    
    if points_per_interval < 1:
        points_per_interval = 1
    if points_per_stride < 1:
        points_per_stride = 1
    results = []
    num_intervals = int((N - points_per_interval) / points_per_stride) + 1

    for i in range(num_intervals):
        start_idx = i * points_per_stride
        end_idx   = start_idx + points_per_interval
        if end_idx > N:
            break
        dVX_interval  = dVX[start_idx:end_idx]
        dVY_interval  = dVY[start_idx:end_idx]
        Vp_interval   = Vp[start_idx:end_idx]
        B_interval    = B[start_idx:end_idx]
        time_interval = times[start_idx:end_idx]
        try:
            a, b, c, d = fit_coupled_linear_model(Vp_interval, B_interval, dVX_interval, dVY_interval)

            # Invert parameters to calibration coefficients
            Leff, theta, _, _ = invert_parameters_to_calibration_coefficients(a, b, c, d)


            # Convert to astropy quantities
            Ex = ((-a*c + a*dVX_interval + b*d - b*dVY_interval)/(a**2 + b**2)) * u.V / u.m
            Ey = ((-a*d + a*dVY_interval - b*c + b*dVX_interval)/(a**2 + b**2)) * u.V / u.m

            # Compute -V x B in V/m
            Vp_interval_m_per_s = Vp_interval *  u.m / u.s  # km/s to m/s
            B_interval_tesla    = B_interval * u.T           # nT to T
            VxB_interval        = -np.cross(Vp_interval_m_per_s, B_interval_tesla)  # V/m

            VxB_x = VxB_interval[:, 0].to(u.V / u.m).value  # V/m
            VxB_y = VxB_interval[:, 1].to(u.V / u.m).value  # V/m

            # Compute cross-correlation
            Ex_value = Ex.value
            Ey_value = Ey.value
            Cxx, Cyy = compute_cross_correlation(Ex_value, Ey_value, VxB_x, VxB_y)

            # Check correlation threshold
            if abs(Cxx) < min_correlation or abs(Cyy) < min_correlation:
                Leff = np.nan
                theta = np.nan
                c_offset = np.nan
                d_offset = np.nan
                Cxx = np.nan
                Cyy = np.nan

            # Time tag at the center of the interval
            time_tag = time_interval[len(time_interval)//2]
            results.append({
                'datetime' : pd.to_datetime(time_tag),
                'Leff'     : Leff,           # meters
                'theta'    : theta,          # degrees
                'a'        : a,    
                'b'        : b,   
                'c'        : c,   # volts
                'd'        : d,   # volts
                'Cxx'      : Cxx,
                'Cyy'      : Cyy
            })
            #print('worked')
        except:
            #Print traceback for debugging
            traceback.print_exc()
            # Handle errors
            time_tag = time_interval[len(time_interval)//2]
            results.append({
                'datetime': pd.to_datetime(time_tag),
                'Leff': np.nan,
                'theta': np.nan,
                'offset_c': np.nan,
                'offset_d': np.nan,
                'Cxx': np.nan,
                'Cyy': np.nan
            })
    results_df = pd.DataFrame(results)
    results_df.set_index('datetime', inplace=True)
    return results_df



def calibrate_data(edf,
                   coeffs):
                                                 
    
    # Upsample the low freq estimates of the coefficients
    #edf, coeffs_hf = func.synchronize_dfs(edf, coeffs.interpolate().dropna(), True)
    coeffs_hf = func.newindex( coeffs, edf.index)
    
    
    
    # Convert to astropy quantities
    dVx = edf['dvx'].values # u.V
    dVy = edf['dvy'].values # u.V
    a   = coeffs_hf['a'].values      # u.m
    b   = coeffs_hf['b'].values      # u.m
    c   = coeffs_hf['c'].values      # u.V
    d   = coeffs_hf['d'].values      # u.V

    # Calibrate and overt to mV/v
    Ex = ((-a*c + a*dVx + b*d - b*dVy)/(a**2 + b**2)) *1e3
    Ey = ((-a*d + a*dVy - b*c + b*dVx)/(a**2 + b**2)) *1e3
    
    Edf = pd.DataFrame({'datetime': edf.index.values, 'Ex': Ex, 'Ey':Ey}).set_index('datetime')
    
    # Interpolate missing values and drop any remaining NaNs
    return Edf.interpolate().dropna()



# def estimate_Ez(
#                 B_df,
#                 E_df, min_bz = 0.1):
    
#     #B_df_fin.values.T[2][np.abs(B_df_fin.values.T[2])<0.1]=np.nan
#     B          = B_df.values
#     E          = E_df[E_df.columns[0:2]].values
    
    
#     Ez         = (-B.T[0] * E.T[0] - B.T[1] * E.T[1])/B.T[2]
    
    
#     E_df[E_df.columns[0:2][0][0]+'z'] = Ez
#     return E_df





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




def estimate_Ez(B_df, E_df, min_bz=1, window_size = 51, n=2,  apply_hampel = True):
    # Modify Bz in place where abs(Bz) < min_bz
    Bz_col = B_df.columns[2]
    Bz = B_df[Bz_col]
    mask = np.abs(Bz) < min_bz
    B_df.loc[mask, Bz_col] = np.nan

    # Extract Bx, By, Bz, Ex, Ey without making copies
    Bx = B_df.iloc[:, 0]
    By = B_df.iloc[:, 1]
    Bz = B_df.iloc[:, 2]
    Ex = E_df.iloc[:, 0]
    Ey = E_df.iloc[:, 1]

    # Compute Ez directly; NaNs in Bz will propagate
    Ez = (-Bx * Ex - By * Ey) / Bz

    # Assign Ez to E_df in place
    E_df['Ez'] = Ez
    E_df = E_df.interpolate().dropna()
    
    # Apply Hampel filter if required
    if apply_hampel:
        # Determine which velocity components to filter based on data presence
        columns_for_hampel = E_df.columns

        for column in columns_for_hampel:
            print(column)
            try:
                filtered_arr, outliers_indices = func.hampel(E_df[column], window_size, n)
                print('Identified', len(outliers_indices),' outliers')
                #Edf.loc[Edf.index[outliers_indices], column] = np.nan
                E_df[column] = filtered_arr
            except:
                traceback.print_exc()
                

    return E_df



# from astropy import units as u

# def block_average(data, cadence_seconds):
#     """
#     Block-average data to common times at specified cadence.

#     Parameters:
#     - data: pandas DataFrame containing the data with a datetime index.
#     - cadence_seconds: block averaging cadence in seconds (e.g., 12).

#     Returns:
#     - DataFrame with averaged data at the new cadence.
#     """
#     data = data.copy()
#     data.index = pd.to_datetime(data.index)
#     # Resample data at the specified cadence
#     averaged_data = data#.resample(f'{cadence_seconds}S').mean()
#     return averaged_data

# def project_dV(dV12, dV34):
#     """
#     Project dV12 and dV34 from whip coordinates to spacecraft (SC) coordinates.

#     Parameters:
#     - dV12: numpy array of differential voltages from probes 1 and 2 (V).
#     - dV34: numpy array of differential voltages from probes 3 and 4 (V).

#     Returns:
#     - dVX, dVY: numpy arrays of differential voltages in SC coordinates.
#     """
#     R_V_to_SC = np.array([[0.64524, -0.82228],
#                           [0.76897,  0.57577]])
#     dV_whip = np.vstack((dV12, dV34))
#     dV_SC = np.dot(R_V_to_SC, dV_whip)
#     dVX = dV_SC[0, :]
#     dVY = dV_SC[1, :]
#     return dVX, dVY

# from scipy.optimize import curve_fit
# import astropy.units as u

# def fit_coupled_linear_model(Vp, B, dVX, dVY):
#     """
#     Fit the projected differential voltages using the coupled four-parameter linear model.

#     Parameters:
#     - Vp: numpy array of proton velocities (km/s), shape (N, 3).
#     - B: numpy array of magnetic field measurements (nT), shape (N, 3).
#     - dVX, dVY: numpy arrays of differential voltages in SC coordinates (V), shape (N,).

#     Returns:
#     - a, b, c, d: fitted parameters of the model (a and b in meters, c and d in volts).
#     """
#     # Convert Vp from km/s to m/s and B from nT to T
#     Vp_m_per_s = Vp * u.m / u.s  # Convert to Quantity with units
#     B_tesla    = B * u.T  # Convert to Quantity with units

#     # Compute -Vp x B (units: V/m)
#     VxB = -np.cross(Vp_m_per_s, B_tesla)  # Units: (m/s) x (T) = V/m

#     # Extract x and y components and convert to numeric values in V/m
#     VpxBx = VxB[:, 0].to(u.V / u.m).value
#     VpxBy = VxB[:, 1].to(u.V / u.m).value

#     # Define the model for dVX and dVY
#     def model(xdata, a, b, c, d):
#         VpxBx, VpxBy = xdata
#         dVX = a * VpxBx + b * VpxBy + c
#         dVY = -b * VpxBx + a * VpxBy + d
#         return np.concatenate([dVX, dVY])

#     # Stack the independent variables into one array
#     xdata = np.vstack((VpxBx, VpxBy))

#     # Concatenate the dependent variables (observed dVX and dVY)
#     ydata = np.concatenate([dVX, dVY])

#     # Initial guess for parameters (a, b, c, d)
#     initial_params = [1, 1, 0, 0]

#     # Perform the curve fitting
#     params_opt, _ = curve_fit(model, xdata, ydata, p0=initial_params)

#     # Unpack the optimized parameters
#     a_opt, b_opt, c_opt, d_opt = params_opt

#     return a_opt, b_opt, c_opt, d_opt


# def invert_parameters_to_calibration_coefficients(a, b, c, d):
#     """
#     Invert the model parameters to obtain calibration coefficients.

#     Parameters:
#     - a, b: effective dipole components (meters).
#     - c, d: offset voltages (volts).

#     Returns:
#     - Leff: effective dipole length (meters).
#     - theta: rotation angle (degrees).
#     - c, d: offset voltages (volts).
#     """
#     Leff   = np.sqrt(a**2 + b**2)        # meters
#     theta  = np.degrees(np.arctan(b/a))  # degrees
#     return Leff, theta, c, d

# def compute_cross_correlation(Ex, Ey, VxB_x, VxB_y):
#     """
#     Compute the cross-correlation between calibrated E-fields and -V x B.

#     Parameters:
#     - Ex, Ey: calibrated electric field components (V/m).
#     - VxB_x, VxB_y: components of -V x B (V/m).

#     Returns:
#     - Cxx, Cyy: cross-correlation coefficients.
#     """
#     Ex_zero_mean = Ex - np.mean(Ex)
#     Ey_zero_mean = Ey - np.mean(Ey)
#     VxB_x_zero_mean = VxB_x - np.mean(VxB_x)
#     VxB_y_zero_mean = VxB_y - np.mean(VxB_y)

#     Cxx = np.corrcoef(Ex_zero_mean, VxB_x_zero_mean)[0, 1]
#     Cyy = np.corrcoef(Ey_zero_mean, VxB_y_zero_mean)[0, 1]

#     return Cxx, Cyy


# def synchronize_merge_dfs(bdf, vdf, edf):
    
#     # Synchronize
#     edf, vdf                      = func.synchronize_dfs(edf, vdf, False)
#     bdf, vdf                      = func.synchronize_dfs(bdf, vdf, False)

#     # Merge
#     fin_data                      = edf
#     fin_data[['Vx', 'Vy', 'Vz']]  = vdf
#     fin_data[['Bx', 'By', 'Bz']]  = bdf

#     # Interpolate dropna
#     return fin_data.interpolate().dropna()


# def process_data(bdf, vdf, edf, cadence_seconds=12, fit_interval_minutes=4, stride_minutes=1, min_correlation=0.5):
#     """
#     Process the data to compute calibration coefficients over sliding intervals.

#     Parameters:
#     - data: pandas DataFrame containing the data with a datetime index.
#     - cadence_seconds: block averaging cadence in seconds (e.g., 12).
#     - fit_interval_minutes: length of each fitting interval in minutes (e.g., 4).
#     - stride_minutes: stride length between intervals in minutes (e.g., 1).
#     - min_correlation: minimum acceptable cross-correlation value (e.g., 0.5).

#     Returns:
#     - DataFrame containing calibration coefficients and correlation metrics.
#     """
    
#     # Synchronize dfs
#     data           = synchronize_merge_dfs(bdf, vdf, edf)
#     cadence_seconds= func.find_cadence(vdf)
    
#     # Find mov averages
#     averaged_data = block_average(data, cadence_seconds)

#     # Extract variables
#     #print(averaged_data)

#     B       = (averaged_data[['Bx', 'By', 'Bz']].values  * 1e-9 * u.T).value    # T
#     Vp      = (averaged_data[['Vx', 'Vy', 'Vz']].values  * 1e3 * u.m / u.s ).value # m/s

#     # Project to SC coordinates
#     dVX     = (averaged_data['dvx'].values.T * u.V).value # Volt
#     dVY     = (averaged_data['dvy'].values.T * u.V).value # Volt

#     times = averaged_data.index.values
#     N     = len(averaged_data)
#     points_per_interval  = int((fit_interval_minutes * 60) / cadence_seconds)
#     points_per_stride    = int((stride_minutes * 60) / cadence_seconds)
    
#     if points_per_interval < 1:
#         points_per_interval = 1
#     if points_per_stride < 1:
#         points_per_stride = 1
#     results = []
#     num_intervals = int((N - points_per_interval) / points_per_stride) + 1

#     for i in range(num_intervals):
#         start_idx = i * points_per_stride
#         end_idx   = start_idx + points_per_interval
#         if end_idx > N:
#             break
#         dVX_interval  = dVX[start_idx:end_idx]
#         dVY_interval  = dVY[start_idx:end_idx]
#         Vp_interval   = Vp[start_idx:end_idx]
#         B_interval    = B[start_idx:end_idx]
#         time_interval = times[start_idx:end_idx]
#         try:
#             a, b, c, d = fit_coupled_linear_model(Vp_interval, B_interval, dVX_interval, dVY_interval)

#             # Invert parameters to calibration coefficients
#             Leff, theta, _, _ = invert_parameters_to_calibration_coefficients(a, b, c, d)


#             # Convert to astropy quantities
#             Ex = ((-a*c + a*dVX_interval + b*d - b*dVY_interval)/(a**2 + b**2)) * u.V / u.m
#             Ey = ((-a*d + a*dVY_interval - b*c + b*dVX_interval)/(a**2 + b**2)) * u.V / u.m

#             # Compute -V x B in V/m
#             Vp_interval_m_per_s = Vp_interval *  u.m / u.s  # km/s to m/s
#             B_interval_tesla    = B_interval * u.T           # nT to T
#             VxB_interval        = -np.cross(Vp_interval_m_per_s, B_interval_tesla)  # V/m

#             VxB_x = VxB_interval[:, 0].to(u.V / u.m).value  # V/m
#             VxB_y = VxB_interval[:, 1].to(u.V / u.m).value  # V/m

#             # Compute cross-correlation
#             Ex_value = Ex.value
#             Ey_value = Ey.value
#             Cxx, Cyy = compute_cross_correlation(Ex_value, Ey_value, VxB_x, VxB_y)

#             # Check correlation threshold
#             if abs(Cxx) < min_correlation or abs(Cyy) < min_correlation:
#                 Leff = np.nan
#                 theta = np.nan
#                 c_offset = np.nan
#                 d_offset = np.nan
#                 Cxx = np.nan
#                 Cyy = np.nan

#             # Time tag at the center of the interval
#             time_tag = time_interval[len(time_interval)//2]
#             results.append({
#                 'datetime' : pd.to_datetime(time_tag),
#                 'Leff'     : Leff,           # meters
#                 'theta'    : theta,         # degrees
#                 'a'        : a,    
#                 'b'        : b,   
#                 'c'        : c,   # volts
#                 'd'        : d,   # volts
#                 'Cxx'      : Cxx,
#                 'Cyy'      : Cyy
#             })
#             #print('worked')
#         except Exception as e:
#             #Print traceback for debugging
#             traceback.print_exc()
#             # Handle errors
#             time_tag = time_interval[len(time_interval)//2]
#             results.append({
#                 'datetime': pd.to_datetime(time_tag),
#                 'Leff': np.nan,
#                 'theta': np.nan,
#                 'offset_c': np.nan,
#                 'offset_d': np.nan,
#                 'Cxx': np.nan,
#                 'Cyy': np.nan
#             })
#     results_df = pd.DataFrame(results)
#     results_df.set_index('datetime', inplace=True)
#     return results_df



# def calibrate_data(edf, coeffs, apply_hampel= True, window_size=200, n=3):
                                                 
    
#     # Upsample the low freq estimates of the coefficients
#     #edf, coeffs_hf = func.synchronize_dfs(edf, coeffs.interpolate().dropna(), True)
#     coeffs_hf = func.newindex( coeffs, edf.index)
    
    
    
#     # Convert to astropy quantities
#     dVx = edf['dvx'].values # u.V
#     dVy = edf['dvy'].values # u.V
#     a   = coeffs_hf['a'].values      # u.m
#     b   = coeffs_hf['b'].values      # u.m
#     c   = coeffs_hf['c'].values      # u.V
#     d   = coeffs_hf['d'].values      # u.V

#     # Calibrate and overt to mV/v
#     Ex = ((-a*c + a*dVx + b*d - b*dVy)/(a**2 + b**2)) *1e3
#     Ey = ((-a*d + a*dVy - b*c + b*dVx)/(a**2 + b**2)) *1e3
    
    
#     Edf = pd.DataFrame({'datetime': edf.index.values, 'Ex': Ex, 'Ey':Ey}).set_index('datetime')
    



# #     # Replace 'Ex' and 'Ey' with NaN where the condition is True
# #     Edf.loc[(np.abs(Edf['Ex']) > 4e1) | (np.abs(Edf['Ey']) > 4e1), ['Ex', 'Ey']] = np.nan
    



#     # Interpolate missing values and drop any remaining NaNs
#     return Edf.interpolate().dropna()

