import warnings
warnings.filterwarnings('ignore')


import traceback
import ssqueezepy

import scipy
import os
import numpy as np
import pandas as pd
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
from scipy.signal import stft


sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))

import TurbPy as turb
import general_functions as func


import traceback
from numba import njit, prange
from joblib import Parallel, delayed
from tqdm import tqdm
from CWTPy import cwt_module


def estimate_cwt_old(signal,
                 dt,
                 nv            = 32,
                 omega0        = 6,
                 scale_type    = 'log-piecewise',
                 vectorized    = True,
                 l1_norm       = False,
                 min_freq = None):
    """
    Estimate continuous wavelet transform of the signal.

    Parameters:
    - signal (pd.DataFrame or np.ndarray): Input signal(s).
    - dt (float): Sampling interval.
    - nv (int): Number of voices per octave.
    - omega0 (int): Morlet wavelet parameter.
    - min_freq (float, optional): Minimum frequency to retain.

    Returns:
    - w_df (dict or np.ndarray): Wavelet coefficients per column or array.
    - scales (np.ndarray): Scales used.
    - freqs (np.ndarray): Frequencies corresponding to scales.
    - coi (None): Cone of influence (not computed here).
    """
    fs = 1 / dt
    wavelet = ssqueezepy.Wavelet(('morlet', {'mu': omega0}))

    if isinstance(signal, pd.DataFrame):
        w_df = {}
        for col in signal.columns:
            W, scales = ssqueezepy.cwt(signal[col].values,
                                       wavelet    = wavelet, 
                                       scales     = scale_type,
                                       l1_norm    = l1_norm,
                                       fs         = fs,
                                       nv         = nv,
                                       vectorized = vectorized)
            # Compute frequencies corresponding to scales
            freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal[col]), fs)
            scales = (omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * omega0**2))

            # Remove the first five scales and corresponding coefficients

            fs = 1 / dt
            nyquist = fs / 2.0
            cutoff = 0.95 * nyquist
            
            # Create a boolean mask to keep frequencies below the cutoff.
            mask = freqs < cutoff
            
            W      = W[mask, :]
            scales = scales[mask]
            freqs  = freqs[mask]

            # Remove frequencies lower than min_freq
            if min_freq is not None:
                indices = np.where(freqs >= min_freq)[0]
                W       = W[indices, :]
                scales  = scales[indices]
                freqs   = freqs[indices]

            w_df[col] = W
    else:
        W, scales = ssqueezepy.cwt(signal,
                                   wavelet    = wavelet, 
                                   scales     = scale_type,
                                   l1_norm    = l1_norm,
                                   fs         = fs,
                                   nv         = nv,
                                   vectorized = vectorized)
        # Compute frequencies corresponding to scales
        freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal), fs)
        scales = (omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * omega0**2))*fs 

        # Remove the first five scales and corresponding coefficients
        W       = W[5:, :]
        scales  = scales[5:]
        freqs   = freqs[5:]

        # Remove frequencies lower than min_freq
        if min_freq is not None:
            indices = np.where(freqs >= min_freq)[0]
            W       = W[indices, :]
            scales  = scales[indices]
            freqs   = freqs[indices]

        w_df = W

    coi = None

    return w_df, scales, freqs, 2*dt, coi


def estimate_cwt    (signal,
                     dt,
                     nv                = 16,
                     omega0            = 6.0,
                     min_freq          = None,
                     max_freq          = None,
                     use_omp           = False,
                     consider_coi      = False,
                     compute_trace_psd = False,
                     scale_type       = 'log'):
    """
    Estimate the CWT of the given signal using cwt_module.cwt_morlet_full.
    
    This wrapper accepts a 1D numpy array or a pandas DataFrame (processed column-wise) 
    and returns a dictionary with:
      - 'W': wavelet coefficients (2D array, num_scales x time_points)
      - 'scales': 1D array of scales
      - 'freqs': 1D array of wavelet frequencies (Hz)
      - 'psd_norm': normalization factor for converting power to a PSD
      - 'fft_freqs': 1D array of FFT frequencies (Hz)
      - (optionally) 'trace_psd': the trace PSD (if compute_trace_psd is True)
      - (optionally) 'coi': the cone-of-influence (if consider_coi is True)
    
    Parameters
    ----------
    signal : np.ndarray or pd.DataFrame
        Input time-series data.
    dt : float
        Sampling interval.
    nv : int, optional
        Voices per octave (default 16).
    omega0 : float, optional
        Morlet wavelet parameter (default 6.0).
    min_freq : float or None, optional
        Lowest frequency of interest (Hz). If None, C++ uses ~1/(N*dt).
    max_freq : float or None, optional
        Highest frequency of interest (Hz). If None, defaults to fs/2.
    use_omp : bool, optional
        If True, parallelize using OpenMP.
    consider_coi : bool, optional
        If True, return the cone of influence (COI) and use it in PSD masking.
    compute_trace_psd : bool, optional
        If True, compute the trace PSD from the wavelet coefficients.
    
    Returns
    -------
    dict or dict of dicts
        For a 1D input, returns a dictionary with keys:
          'W', 'scales', 'freqs', 'psd_norm', 'fft_freqs', and optionally 'trace_psd', 'coi'.
        For a DataFrame, returns a dictionary mapping column names to such dictionaries.
    """
    # Convert None to 0.0 so C++ defaults are used.
    if min_freq is None:
        min_freq = 0.0
    if max_freq is None:
        max_freq = 0.0

    def _process_1d(sig_1d):
        sig_1d = np.asarray(sig_1d, dtype=np.float64)
        # Call the C++ function, with always returning FFT frequencies.
        ret = cwt_module.cwt_morlet_full(
            sig_1d,
            float(dt),
            int(nv),
            float(omega0),
            float(min_freq),
            float(max_freq),
            bool(use_omp),
            1.0,              # norm_mult, can be adjusted if needed4
            str(scale_type),
            bool(consider_coi),
            True              # Always return FFT frequencies.
        )
        # Unpack outputs.
        W         = ret[0]
        scales    = ret[1]
        freqs     = ret[2]
        psd_norm  = ret[3]
        fft_freqs = ret[4]

        return W, scales, freqs, psd_norm

    w_df = {}
    if isinstance(signal, pd.DataFrame):
        result_dict = {}
        trace_psd   = 0
        for col in signal.columns:
            W, scales, freqs, psd_norm = _process_1d(signal[col].values)
            w_df[col]       = W   
    else:
        W, scales,freqs,psd_norm = _process_1d(signal)
        w_df            = W

    return w_df, scales, freqs, psd_norm, None




def local_gaussian_averaging(signal, dt, scale, alpha=1.0, num_efoldings=3):
    """
    Smooth `signal` with a Gaussian of standard deviation sigma = alpha * scale (seconds).
    We convert that to sample units and do a 'same' convolution.

    Parameters
    ----------
    signal : 1D array
        The input signal, sampled at uniform dt.
    dt : float
        The sampling interval in seconds.
    scale : float
        The wavelet scale in seconds.
    alpha : float
        The dimensionless parameter in eqn(22).
    num_efoldings : float
        The half-width in standard deviations for the kernel. Default=3 => ±3σ.

    Returns
    -------
    smoothed_signal : 1D array, same length as input
    """
    # 1) The "sigma" in time
    sigma_time = alpha * scale
    
    # 2) Convert to sample units
    sigma_samples = sigma_time / dt
    
    # 3) Build the kernel in sample units over ±num_efoldings*sigma_samples
    half_width = int(np.ceil(num_efoldings * sigma_samples))
    t_samples  = np.arange(-half_width, half_width+1)
    kernel     = np.exp(-(t_samples**2)/(2*sigma_samples**2))

    # Normalize the Gaussian window to ensure it sums to one, maintaining the total signal energy after convolution
    kernel /= kernel.sum()
    
    # 4) Convolve
    return scipy.signal.convolve(signal, kernel, mode='same')



def unit_vectors(df, prefix, sufix ='_hat', vector_cols=None):
    """
    Normalize specified numeric columns of the DataFrame and add unit vector columns with the specified prefix.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the columns to normalize.
    - prefix (str): The prefix for the new unit vector columns.
    - columns (list of str, optional): List of column names to normalize. 
      If None, all numeric columns will be normalized.

    Returns:
    - pd.DataFrame: The DataFrame with new unit vector columns added.
    """

    
    # Create new column names with the specified prefix
    unit_col_names = [f"{prefix}{col}{sufix}" for col in vector_cols]
    
    
    #print('Worked on', vector_cols)
    # Assign unit vectors back to the DataFrame
    df[unit_col_names] =  df[vector_cols].values / np.linalg.norm(df[vector_cols].values, axis=1)[:, np.newaxis]
    
    #print('Created', unit_col_names)
    return df



    
def coherence_analysis(B0_f_o,
                       V0_f_o,
                       df_w,
                       method,
                       func_params = None
                      ):
    


    
    def compute_first_eigenvectors(RRe, RTe, RNe, TTe, TNe, NNe):
        n = RRe.shape[0]
        # Stack matrices into a 3D array
        M = np.zeros((n, 3, 3))
        M[:, 0, 0] = RRe
        M[:, 0, 1] = RTe
        M[:, 0, 2] = RNe
        M[:, 1, 0] = RTe
        M[:, 1, 1] = TTe
        M[:, 1, 2] = TNe
        M[:, 2, 0] = RNe
        M[:, 2, 1] = TNe
        M[:, 2, 2] = NNe

        # Compute eigenvalues and eigenvectors for all matrices
        eigvals, eigvecs = np.linalg.eigh(M)

        # Extract the eigenvector corresponding to the largest eigenvalue
        largest_eigvecs = eigvecs[:, :, -1]

        return largest_eigvecs

    
    def unit_eigenvector_computation(df, prefix='eigen'):
        RRe = df['RRe'].values
        RTe = df['RTe'].values
        RNe = df['RNe'].values
        TTe = df['TTe'].values
        TNe = df['TNe'].values
        NNe = df['NNe'].values

        eigen_vectors = compute_first_eigenvectors(RRe, RTe, RNe, TTe, TNe, NNe)

        df[[f"{prefix}_1_hat", f"{prefix}_2_hat", f"{prefix}_3_hat"]] = eigen_vectors
        
        # Estimate unit vector
        return unit_vectors(df,
                           prefix      = '',
                           sufix       = '', 
                           vector_cols =['eigen_1_hat', 'eigen_2_hat', 'eigen_3_hat'])


    # Estimate the unit vectors
    B0_f_o = unit_vectors(B0_f_o, prefix = 'B_0_', vector_cols= ['R', 'T', 'N'])
    V0_f_o = unit_vectors(V0_f_o, prefix = 'V_0_', vector_cols= ['R', 'T', 'N'])
    

  
    # Estimate angle between local backgrounds
    VBangles = np.degrees(np.arccos(np.einsum('ij,ij->i',
                                              B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].values, 
                                              V0_f_o[['V_0_R_hat', 'V_0_T_hat', 'V_0_N_hat']].values)))
    
    if method== 'min_var':
        
  
        # Calculate matrix elements
        B0_f_o['RRe'] = B0_f_o['RR'] - np.square(B0_f_o['R'])
        B0_f_o['TTe'] = B0_f_o['TT'] - np.square(B0_f_o['T'])
        B0_f_o['NNe'] = B0_f_o['NN'] - np.square(B0_f_o['N'])
        B0_f_o['RTe'] = B0_f_o['RT'] - B0_f_o['R'] * B0_f_o['T']
        B0_f_o['RNe'] = B0_f_o['RN'] - B0_f_o['R'] * B0_f_o['N']
        B0_f_o['TNe'] = B0_f_o['TN'] - B0_f_o['T'] * B0_f_o['N']

        # Find eigenvectors
        B0_f_o = unit_eigenvector_computation(B0_f_o, prefix='eigen')
        

        # Calculate the first perpendicular unit vector
        B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']] = np.cross(B0_f_o[['eigen_1_hat', 'eigen_2_hat', 'eigen_3_hat']], 
                                                                   B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']])
  

        # Calculate second perpendicular unit vector
        B0_f_o[['B_2_R_hat', 'B_2_T_hat', 'B_2_N_hat']] = np.cross(B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']], 
                                                                   B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']])

        # Memory cleanup by dropping intermediate columns
        columns_to_drop = [
                           # 'B_1_R', 'B_1_T', 'B_1_N', 
                            'RR', 'TT', 'NN', 'RT', 'RN', 'TN', 
                            'RRe', 'TTe', 'NNe', 'RTe', 'RNe', 'TNe'
        ]
        B0_f_o.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        
        # Extract necessary arrays without copying
        B0    = B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].to_numpy(copy=False)
        B1    = B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']].to_numpy(copy=False)
        B2    = B0_f_o[['B_2_R_hat', 'B_2_T_hat', 'B_2_N_hat']].to_numpy(copy=False)
        r_t_n = df_w[['R', 'T', 'N']].to_numpy(copy=False)

        # Compute Wz, Wy, and Wx
        df_w['W0'] = np.einsum('ij,ij->i', B0, r_t_n)
        df_w['W1'] = np.einsum('ij,ij->i', B1, r_t_n)
        df_w['W2'] = np.einsum('ij,ij->i', B2, r_t_n)

        # Drop original 'R', 'T', 'N'
        df_w.drop(columns = ['R', 'T', 'N'], inplace=True, errors='ignore')

        PL, PR            = calculate_polarization_spectra(df_w['W1'].values, df_w['W2'].values)

        return df_w,  VBangles, PR - PL, PR + PL, -2*np.imag((np.conj(df_w['W0'])*df_w['W1'])), np.abs((np.conj(df_w['W1'])*df_w['W1'])) + np.abs((np.conj(df_w['W0'])*df_w['W0'])) + np.abs((np.conj(df_w['W2'])*df_w['W2']))
    
    elif method =='TN_only':
        
       # PL, PR            = calculate_polarization_spectra(df_w['T'].values, df_w['N'].values)

        #return df_w,  VBangles, PR - PL, PR + PL, None   
        return df_w,  VBangles, 2*np.imag((np.conj(df_w['N'])*df_w['T'])), np.abs(np.conj(df_w['N'])*df_w['N']) + np.abs(np.conj(df_w['T'])*df_w['T']), np.abs(np.conj(df_w['N'])*df_w['N']) + np.abs(np.conj(df_w['T'])*df_w['T']), np.abs(np.conj(df_w['N'])*df_w['N']) #-> Last one does not matter
    else :

        # Calculate second perpendicular unit vector
        B0_f_o[['B_y_R_hat', 'B_y_T_hat', 'B_y_N_hat']] =  np.cross( B0_f_o[['R', 'T', 'N']],
                                                                     V0_f_o[['R', 'T', 'N']])
          
        B0_f_o                                          =  unit_vectors(B0_f_o,
                                                                         prefix       = '', 
                                                                         sufix        = '', 
                                                                         vector_cols = ['B_y_R_hat', 'B_y_T_hat', 'B_y_N_hat'])

        
        Bz    = B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].to_numpy(copy=False)
        By    = B0_f_o[['B_y_R_hat', 'B_y_T_hat', 'B_y_N_hat']].to_numpy(copy=False)
        Bx    = np.cross(By, Bz) 
        r_t_n = df_w[['R', 'T', 'N']].to_numpy(copy=False)


        # Compute Wz, Wy, and Wx
        df_w['Wx'] = np.einsum('ij,ij->i', Bx, r_t_n)
        df_w['Wy'] = np.einsum('ij,ij->i', By, r_t_n)
        df_w['W0'] = np.einsum('ij,ij->i', Bz, r_t_n)
        
    
        # Drop original 'R', 'T', 'N'
        df_w.drop(columns=['R', 'T', 'N'], inplace=True, errors='ignore')
        
        PL, PR = calculate_polarization_spectra(df_w['Wx'].values, df_w['Wy'].values)


        return df_w,  VBangles, PR - PL, PR + PL, 2*np.imag((np.conj(df_w['Wy'])*df_w['W0'])),  np.abs((np.conj(df_w['Wy'])*df_w['Wy'])) + np.abs((np.conj(df_w['Wx'])*df_w['Wx'])) + np.abs((np.conj(df_w['W0'])*df_w['W0']))
    
def calculate_polarization_spectra(Bx, By):
    # Left-handed polarization
    PL = np.abs(Bx - 1j * By)**2

    # Right-handed polarization
    PR = np.abs(Bx + 1j * By)**2

    return PL, PR



def calculate_wavelet_flatness(df_ws_needed, rolling_params, step):
    """
    Calculate  Wavelet flatness.

    Parameters:
    - df_ws_needed (pd.DataFrame or np.ndarray): Input wavelet spectrogram data.
    - rolling_params (dict): Parameters for rolling window operations.
    - step (str): Resampling step size.

    Returns:
    - pd.Series: wavelet flatness values.
    """
    # Ensure DataFrame structure
    df_ws_needed = df_ws_needed if isinstance(df_ws_needed, pd.DataFrame) else df_ws_needed.to_frame()


    flatness = (
        (
            ((df_ws_needed * np.conj(df_ws_needed)).sum(axis=1).pow(2))
            .rolling(**rolling_params)
            .mean()
        )
        /
        (
            (df_ws_needed * np.conj(df_ws_needed)).sum(axis=1)
            .rolling(**rolling_params)
            .mean().pow(2)
        )
    ).resample(step).mean()

    return flatness




def compute_norm_psd(df_ws_needed, rolling_params, dt, counts, window_size, step, psd_norm):
    """
    Compute Power Spectral Density (PSD).

    Parameters:
    - df_ws_needed (pd.DataFrame or np.ndarray): Input wavelet spectrogram data.
    - rolling_params (dict): Parameters for rolling window operations.
    - dt (float): Time step interval.
    - counts (pd.Series or np.ndarray): Normalization counts.
    - window_size (int): Window size for normalization.
    - step (str): Resampling step size.

    Returns:
    - pd.Series: Power Spectral Density (PSD).
    """
    # Ensure DataFrame structure
    df_ws_needed =  df_ws_needed if isinstance(df_ws_needed, pd.DataFrame) else df_ws_needed.to_frame()

    
    psd_sum = psd_norm*(df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean().sum(axis=1)
    PSD    = ( psd_sum * (counts / window_size)).resample(step).mean()
    
    return PSD


def compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm):
    """
    Compute Power Spectral Density (PSD) and return associated counts.

    Parameters:
    - df_ws_needed (pd.DataFrame or np.ndarray): Input wavelet spectrogram data.
    - rolling_params (dict): Parameters for rolling window operations.
    - dt (float): Time step interval.
    - index_mask (pd.Series or np.ndarray): Boolean mask indicating relevant indices.
    - step (str): Resampling step size.

    Returns:
    - PSD (pd.Series): Computed Power Spectral Density.
    - counts (pd.Series): Number of relevant events in the rolling window.
    """

    # Ensure DataFrame structure
    df_ws_needed = df_ws_needed if isinstance(df_ws_needed, pd.DataFrame) else df_ws_needed.to_frame()

    psd_sum = psd_norm * (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean().sum(axis=1)
    if index_mask is not None:
        counts  = index_mask.rolling(**rolling_params).sum().resample(step).mean()
    else:
        counts  = np.nan
    PSD     = (psd_sum).resample(step).mean()

    return PSD, counts



    
    
def est_anisotropic_PSDs(df_w,
                         df_mod,
                         index_par,
                         index_per,
                         dt,
                         psd_norm,
                         func_params=None):

    use_rolling_mean = func_params.get('use_rolling_mean', False)

    if func_params["estimate_PSDs"]:
        if use_rolling_mean:
            averaging_window = func_params['averaging_window']  # e.g., '60s'
            step = func_params['step']  # e.g., '10s'
            rolling_params = {'window': averaging_window, 'center': True, 'min_periods': 10}

            """ Estimate Parallel PSD and SDK """
            index_mask   = pd.Series(index_par, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_par, _   = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_par      = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            """ Estimate Perpendicular PSD and SDK """
            index_mask = pd.Series(index_per, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_per, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            """ Estimate PSDs for the modulus of the fields """
            if func_params['est_mod']:
                index_mask = pd.Series(index_par, index=df_w.index)
                df_mod_needed = pd.Series(df_mod, index=df_w.index).where(index_mask)
                PSD_par_mod, _ = compute_psd_with_counts(df_mod_needed, rolling_params, dt, index_mask, step, psd_norm)

                index_mask = pd.Series(index_per, index=df_w.index)
                df_mod_needed = pd.Series(df_mod, index=df_w.index).where(index_mask)
                PSD_per_mod, _ = compute_psd_with_counts(df_mod_needed, rolling_params, dt, index_mask, step, psd_norm)
            else:
                PSD_par_mod = np.nan
                PSD_per_mod = np.nan

        else:
            """ Estimate anisotropic PSDs without rolling means """
            PSD_par = psd_norm*np.nanmean(np.real(df_w.iloc[index_par].values * np.conj(df_w.iloc[index_par].values)), axis=0).sum() 
            PSD_per = psd_norm*np.nanmean(np.real(df_w.iloc[index_per].values * np.conj(df_w.iloc[index_per].values)), axis=0).sum() 

            """ Estimate anisotropic PSDs for modulus of fields """
            if func_params['est_mod']:
                PSD_par_mod = psd_norm*np.nanmean(np.real(df_mod[index_par] * np.conj(df_mod[index_par]))) 
                PSD_per_mod = psd_norm*np.nanmean(np.real(df_mod[index_per] * np.conj(df_mod[index_per]))) 
            else:
                PSD_par_mod = np.nan
                PSD_per_mod = np.nan

    else:
        PSD_par = PSD_per = PSD_par_mod = PSD_per_mod = SDK_par = SDK_per = np.nan

    return {
        'PSD_par': PSD_par,
        'PSD_per': PSD_per,
        'SDK_par': SDK_par,
        'SDK_per': SDK_per,
        'PSD_par_mod': PSD_par_mod,
        'PSD_per_mod': PSD_per_mod,
    }





def est_sfuncs(df_w,
               df_mod,
               index_par,
               index_per,
               scale, 
               dts,
               func_params = None):
    
    def compute_SF(db, m, tau, dts):
        """
        Compute S^m(τ, θ_VB) based on the inputs delta B, m, and tau.

        Parameters:
        db  : array-like
              Delta B values, assumed to be a list or numpy array of the B fluctuations over time.
        m   : int
              The exponent in the equation.
        tau : float
              The characteristic timescale τ.

        Returns:
        S_m : float
              The result of the equation S^m(τ, θ_VB).
        """
        return np.nanmean(np.abs(db / np.sqrt(tau)) ** m)

    # Define types and initialize dictionary for structure functions
    types  = ['ov', 'par', 'per', 'mod']
    sf_dict = {f'SF_{t}_{m}': [] for t in types for m in range(func_params['max_qorder'])}

    # Compute delta B vector
    db_vec = np.sqrt(np.nansum(np.abs(df_w.values * np.conj(df_w.values)), axis=1))
    db_mods = np.sqrt(np.abs(df_mod * np.conj(df_mod)))



    if func_params.get("est_sfuncs", False):
        for t in types:
            db = {
                'par': db_vec[index_par],
                'per': db_vec[index_per],
                'ov' : db_vec,
                'mod': db_mods
            }.get(t)

            for m in range(func_params['max_qorder']):

                sf_dict[f'SF_{t}_{m}'].append(compute_SF(db, m, scale, dts))

    return sf_dict










def est_compress(df_mod,
                 df_w,
                 index_par,
                 index_per,
                 dt,
                 psd_norm,
                 func_params=None):

    use_rolling_mean = func_params.get('use_rolling_mean', False)
    
    if func_params['estimate_comp']:
        try:
            if use_rolling_mean:
                averaging_window = func_params['averaging_window']  # e.g., '60s'
                step = func_params['step']  # e.g., '10s'
                rolling_params = {'window': averaging_window, 'center': True, 'min_periods': 10}

                """ Calculate PSDs for W0 component (compressive fluctuations) """
                df_ws_needed = df_w['W0']
                PSD_W0, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, None, step, psd_norm)

                """ Calculate PSD for modulus of fields """
                df_ws_needed = pd.Series(df_mod, index=df_w.index)
                PSD_mod, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, None, step, psd_norm)

                """ Calculate total PSDs (trace) for all components """
                df_ws_needed = df_w
                Trace_PSD, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, None, step, psd_norm)

                """ Compute compressibility """
                compress_MOD = (PSD_mod / Trace_PSD).resample(step).mean()
                compress     = (PSD_W0 / Trace_PSD).resample(step).mean()

                """ Calculate PSDs for W0 in parallel modes """
                index_mask = pd.Series(index_par, index=df_w.index)
                df_ws_needed = df_w['W0'].where(index_mask)
                PSD_W0_par, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)

                """ Calculate total PSDs for parallel modes """
                df_ws_needed = df_w.where(index_mask)
                PSD_par, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)

                """ Compute parallel compressibility """
                compress_par = (PSD_W0_par / PSD_par).resample(step).mean()

                """ Calculate PSDs for W0 in perpendicular modes """
                index_mask = pd.Series(index_per, index=df_w.index)
                df_ws_needed = df_w['W0'].where(index_mask)
                PSD_W0_per, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)

                """ Calculate total PSDs for perpendicular modes """
                df_ws_needed = df_w.where(index_mask)
                PSD_per, _ = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)

                """ Compute perpendicular compressibility """
                compress_per = (PSD_W0_per / PSD_per).resample(step).mean()

            else:
                """ Compute compressibility without rolling means """
                compress_par = (np.nanmean(np.real(df_w['W0'][index_par].values * 
                                                   np.conj(df_w['W0'][index_par].values))) /
                                np.nanmean(np.real(df_w.iloc[index_par].values * 
                                                   np.conj(df_w.iloc[index_par].values)), axis=0).sum())
                compress_per = (np.nanmean(np.real(df_w['W0'][index_per].values * 
                                                   np.conj(df_w['W0'][index_per].values))) /
                                np.nanmean(np.real(df_w.iloc[index_per].values * 
                                                   np.conj(df_w.iloc[index_per].values)), axis=0).sum())
                compress = (np.nanmean(np.real(df_w['W0'].values * 
                                               np.conj(df_w['W0'].values))) /
                            np.nanmean(np.real(df_w.values * 
                                               np.conj(df_w.values)), axis=0).sum())
                compress_MOD = (np.nanmean(np.real(df_mod * 
                                                   np.conj(df_mod))) /
                                np.nanmean(np.real(df_w.values * 
                                                   np.conj(df_w.values)), axis=0).sum())
        except:
            # traceback.print_exc()
            compress_par = compress_per = compress = compress_MOD = np.nan
    else:
        compress_par = compress_per = compress = compress_MOD = np.nan

    return {
        'compress_MOD'   : compress_MOD,
        'compress'       : compress,
        'compress_par'   : compress_par,
        'compress_per'   : compress_per,
        'PSD_mod'        : PSD_mod,
        'PSD_Trace_v2'   : Trace_PSD,
    }


def do_coh_analysis(df_w,
                    S0,
                    S3,
                    S0_full,
                    Syz,
                    index_par,
                    index_per,
                    dt,
                    scale,
                    psd_norm,
                    func_params=None):
    """
    Calculate coherent and non-coherent sums for wave components.

    Parameters:
    - df_w (DataFrame): DataFrame representing different wave components (real, tangential, normal).
    - S0, S3, Syz (array-like): Arrays or Series for computation.
    - index_par, index_per (array-like): Boolean arrays for parallel and perpendicular components.
    - dt (float): Time step used in the local Gaussian averaging.
    - scale (float): Scale parameter.
    - func_params (dict): Dictionary containing various function parameters.

    Returns:
    - dict: Dictionary containing calculated values.
    """

    if not func_params.get('estimate_coh_coeffs', True):
        return None

    use_rolling_mean = func_params.get('use_rolling_mean', False)
    estimate_KAW_psd = func_params.get('est_sig_yz_cond_moms', False)

    if use_rolling_mean:
        # Ensure df_w has a DateTimeIndex
        if not isinstance(df_w.index, pd.DatetimeIndex):
            raise ValueError("df_w must have a DateTimeIndex when use_rolling_mean is True.")

        averaging_window = func_params['averaging_window']  # e.g., '60s'
        step             = func_params['step']  # e.g., '10s'
        coh_th           = func_params['coh_th']
        rolling_params   = {'window': averaging_window, 'center': True, 'min_periods': 10}
        
        

        # Estmate sigma_yz for heatmaps!
        sigma_av_yz  = np.nan#(pd.Series(num_value_yz / den_value_yz, index=df_w.index)).resample(step).mean()

        #del num_value_yz, den_value_yz
        
        
        # Use gaussian averaging for sigma_xy
        num_value = local_gaussian_averaging(S3, dt, scale,
                                             alpha=func_params.get('alpha_sigma', 3),
                                             num_efoldings=func_params.get('num_efoldings', 3))
        den_value = local_gaussian_averaging(S0, dt, scale,
                                             alpha=func_params.get('alpha_sigma', 3),
                                             num_efoldings=func_params.get('num_efoldings', 3))
        

        # Don't downsample at first to find coh indices
        sigma = num_value / den_value
        
        
        # Boolean indices for coherent and non-coherent conditions based on the threshold
        index_coh         = np.abs(sigma) > func_params['coh_th']
        index_non_coh     = ~index_coh  # Logical negation of index_coh
        
        # Estmate sigma_xy for heatmaps!
        sigma_av_xy       = (pd.Series(sigma, index=df_w.index)).resample(step).mean()

        del sigma, den_value, num_value, 


        # Estimate counts
        coh_counts       = (pd.Series(index_coh,  index=df_w.index)).rolling(**rolling_params).sum()
        non_coh_counts   = (pd.Series(~index_coh, index=df_w.index)).rolling(**rolling_params).sum()
        window_size      = (coh_counts + non_coh_counts)
        


        if estimate_KAW_psd:

            # Use gaussian averaging for sigma_xy
            num_value = local_gaussian_averaging(Syz, dt, scale,
                                                 alpha=1,
                                                 num_efoldings=func_params.get('num_efoldings', 3))
            den_value = local_gaussian_averaging(S0_full, dt, scale,
                                                 alpha=1,
                                                 num_efoldings=func_params.get('num_efoldings', 3))
            

            # Don't downsample at first to find coh indices
            sigma_yz_ind = num_value/den_value
    
            # Boolean indices for coherent and non-coherent conditions based on the threshold
            index_p           = sigma_yz_ind > func_params['sigma_yz_thresh']['pos']
            index_n           = sigma_yz_ind < func_params['sigma_yz_thresh']['neg']
            index_z           = np.abs(sigma_yz_ind) < func_params['sigma_yz_thresh']['zer']
            
       
            del sigma_yz_ind, den_value, num_value#, sigma_yz_ind 
            
                 

        # Compute sigma_xy, sigma_yz
        sigma_xy     = (pd.Series(S3, index=df_w.index).rolling(**rolling_params).mean()  / pd.Series(S0, index=df_w.index).rolling(**rolling_params).mean()).resample(step).mean()
        sigma_yz     = (pd.Series(Syz, index=df_w.index).rolling(**rolling_params).mean()  / pd.Series(S0_full, index=df_w.index).rolling(**rolling_params).mean()).resample(step).mean()
    

        # Per
        num_mean_per_yz = pd.Series(Syz, index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
        den_mean_per    = pd.Series(S0_full,  index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
        sigma_yz_per    = (num_mean_per_yz / den_mean_per).resample(step).mean()
        
        
        del den_mean_per, num_mean_per_yz
        
        
        # Par
        num_mean_par    = pd.Series(S3,  index=df_w.index).where(pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).mean()
        den_mean_par    = pd.Series(S0,  index=df_w.index).where(pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).mean()
        sigma_xy_par    = (num_mean_par / den_mean_par).resample(step).mean()

        del num_mean_par, den_mean_par
        
        num_mean_per    = pd.Series(S3,  index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
        den_mean_per    = pd.Series(S0,  index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
        sigma_xy_per    = (num_mean_per / den_mean_per).resample(step).mean()
        
        del num_mean_per, den_mean_per


        """Coherent"""
        index_mask   = pd.Series(index_coh, index=df_w.index)
        df_ws_needed = df_w.where(index_mask)
        PSD_coh      = compute_norm_psd(df_ws_needed, rolling_params, dt, coh_counts, window_size, step, psd_norm)

        """Non coherent"""
        index_mask   =  pd.Series(~index_coh, index=df_w.index)
        df_ws_needed = df_w.where(index_mask)
        PSD_non_coh  = compute_norm_psd(df_ws_needed, rolling_params, dt, non_coh_counts, window_size, step, psd_norm)
        SDK_non_coh  = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
        """Trace"""
        Trace_PSD = np.nansum([PSD_coh, PSD_non_coh], axis=0)
        SDK       = calculate_wavelet_flatness(df_w, rolling_params, step)

        
        # Resample if they are pandas objects
        if isinstance(window_size, (pd.Series, pd.DataFrame)):
            window_size = window_size.resample(step).sum()
        if isinstance(coh_counts, (pd.Series, pd.DataFrame)):
            coh_counts = coh_counts.resample(step).sum()
        if isinstance(non_coh_counts, (pd.Series, pd.DataFrame)):
            non_coh_counts = non_coh_counts.resample(step).sum()



        
        """Non coherent_perp"""
        index_mask                          =  pd.Series(~index_coh & index_per, index=df_w.index)
        df_ws_needed                        = df_w.where(index_mask)
        PSD_non_coh_per, non_coh_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
        SDK_non_coh_per                     = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)


        """Non coherent_par"""
        index_mask                          = pd.Series(~index_coh & index_par, index=df_w.index)
        df_ws_needed                        = df_w.where(index_mask)
        PSD_non_coh_par, non_coh_par_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
        SDK_non_coh_par                     = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)



        if estimate_KAW_psd:
            """KAW: Positive Component"""
            index_mask                              = pd.Series(index_p, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_p, sig_yz_p_counts           = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_p                            = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Positive Perpendicular Component"""
            index_mask                              = pd.Series(index_p & index_per, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_p_per, sig_yz_p_per_counts   = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_p_per                        = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Negative Component"""
            index_mask                              = pd.Series(index_n, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_n, sig_yz_n_counts           = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_n                            = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Negative Perpendicular Component"""
            index_mask                              = pd.Series(index_n & index_per, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_n_per, sig_yz_n_per_counts   = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_n_per                        = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Positive + Negative Component"""
            index_mask                              = pd.Series(index_n | index_p, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_pn, sig_yz_pn_counts         = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_pn                           = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Positive + Negative Perpendicular Component"""
            index_mask                              = pd.Series((index_n | index_p) & index_per, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_pn_per, sig_yz_pn_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_pn_per                       = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Zero Component"""
            index_mask                              = pd.Series(index_z, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_z, sig_yz_z_counts           = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_z                            = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
            """KAW: Zero Perpendicular Component"""
            index_mask                              = pd.Series(index_z & index_per, index=df_w.index)
            df_ws_needed                            = df_w.where(index_mask)
            PSD_sig_yz_z_per, sig_yz_z_per_counts   = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_z_per                        = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        
        else:
            PSD_sig_yz_n = PSD_sig_yz_p = PSD_sig_yz_z = PSD_sig_yz_pn = np.nan
            PSD_sig_yz_z_per = PSD_sig_yz_p_per = PSD_sig_yz_n_per = PSD_sig_yz_pn_per = np.nan
            SDK_sig_yz_n = SDK_sig_yz_p = SDK_sig_yz_z = SDK_sig_yz_pn = np.nan
            SDK_sig_yz_z_per = SDK_sig_yz_p_per = SDK_sig_yz_n_per = SDK_sig_yz_pn_per = np.nan
            sig_yz_p_counts = sig_yz_n_counts = sig_yz_z_counts = sig_yz_pn_counts = np.nan
            sig_yz_p_per_counts = sig_yz_n_per_counts = sig_yz_z_per_counts = sig_yz_pn_per_counts = np.nan
        
                

        return {
                    'sigma_xy_av'        : sigma_av_xy,
                    'sigma_yz_av'        : sigma_av_yz,
            
                    'sigma_xy'           : sigma_xy,
                    'sigma_xy_par'       : sigma_xy_par,
                    'sigma_xy_per'       : sigma_xy_per,
            
                    'sigma_yz'           : sigma_yz,
                    'sigma_yz_per'       : sigma_yz_per,
            
                    'PSD_Trace'          : Trace_PSD,
                    'PSD_coh'            : PSD_coh,
                    'PSD_non_coh'        : PSD_non_coh,
            
                    'SDK'                : SDK,
                    'SDK_non_coh'        : SDK_non_coh,
                    'SDK_non_coh_par'    : SDK_non_coh_par,
                    'SDK_non_coh_per'    : SDK_non_coh_per,
                    'SDK_sig_yz_n'       : SDK_sig_yz_n,
                    'SDK_sig_yz_p'       : SDK_sig_yz_p,
                    'SDK_sig_yz_z'       : SDK_sig_yz_z,
                    'SDK_sig_yz_pn'      : SDK_sig_yz_pn,
                    'SDK_sig_yz_z_per'   : SDK_sig_yz_z_per,
                    'SDK_sig_yz_p_per'   : SDK_sig_yz_p_per,
                    'SDK_sig_yz_n_per'   : SDK_sig_yz_n_per,
                    'SDK_sig_yz_pn_per'  : SDK_sig_yz_pn_per,
                    
            
                    'PSD_non_coh_per'    : PSD_non_coh_per,
                    'counts_non_coh_per' : non_coh_per_counts,
                    
                    'PSD_non_coh_par'    : PSD_non_coh_par,
                    'counts_non_coh_par' : non_coh_par_counts,
                    
                    'PSD_sig_yz_n'       : PSD_sig_yz_n,
                    'PSD_sig_yz_p'       : PSD_sig_yz_p,
                    'PSD_sig_yz_z'       : PSD_sig_yz_z,
                    'PSD_sig_yz_pn'      : PSD_sig_yz_pn,
                
                    'PSD_sig_yz_z_per'   : PSD_sig_yz_z_per,
                    'PSD_sig_yz_p_per'   : PSD_sig_yz_p_per,        
                    'PSD_sig_yz_n_per'   : PSD_sig_yz_n_per,              
                    'PSD_sig_yz_pn_per'  : PSD_sig_yz_pn_per,
                

                
                    'counts_sig_yz_p'    : sig_yz_p_counts,
                    'counts_sig_yz_n'    : sig_yz_n_counts,
                    'counts_sig_yz_z'    : sig_yz_z_counts,
                    'counts_sig_yz_pn'   : sig_yz_pn_counts,
                
                    'counts_sig_yz_p_per' : sig_yz_p_per_counts,
                    'counts_sig_yz_n_per' : sig_yz_n_per_counts,
                    'counts_sig_yz_z_per' : sig_yz_z_per_counts,
                    'counts_sig_yz_pn_per': sig_yz_pn_per_counts,
            
                    'counts_par'          : ((pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).sum()).resample(step).sum(),
                    'counts_per'          : ((pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).sum()).resample(step).sum(),
                    'counts_coh'          : coh_counts,
                    'counts_non_coh'      : non_coh_counts,
            
                    'counts'             : window_size,
                    'coh_thresh'         : func_params['coh_th']
        }
        

    else:

        print('WRONG NORMALIZATION HERE FIX with psd_norm!!!!')
        num_value = local_gaussian_averaging(S3, dt, scale,  alpha = func_params['alpha'], num_efoldings = func_params['num_efoldings'])
        den_value = local_gaussian_averaging(S0, dt, scale,  alpha = func_params['alpha'], num_efoldings = func_params['num_efoldings'])

        # Polarization parameter
        sigma     =  num_value / den_value

        # Estimate it specifically in the par, perp direction and fin the mean for the specific scale
        sigma_xy             = np.nanmean(S3)/ np.nanmean(S0)
        sigma_xy_par         = np.nanmean(S3[index_par])/np.nanmean(S0[index_par])
        sigma_xy_per         = np.nanmean(S3[index_per])/np.nanmean(S0[index_per])   

        sigma_yz             = np.nanmean(Syz)/ np.nanmean(S0)
        sigma_yz_par         = np.nanmean(Syz[index_par])/np.nanmean(S0[index_par])
        sigma_yz_per         = np.nanmean(Syz[index_per])/np.nanmean(S0[index_per])   

        # Boolean indices for coherent and non-coherent conditions based on the threshold
        index_coh         = np.abs(sigma) > func_params['coh_th']
        index_non_coh     = ~index_coh  # Logical negation of index_coh

        # Calculate the coherent component sum
        coherent_sum     = np.nanmean(np.real(df_w.iloc[index_coh].values*np.conj(df_w.iloc[index_coh].values)), axis=0).sum() 

        # Calculate the non-coherent component sum
        non_coherent_sum = np.nanmean(np.real(df_w.iloc[index_non_coh].values*np.conj(df_w.iloc[index_non_coh].values)), axis=0).sum() 

        # Estimae PSDs for  coh and non_coh coefficients 
        PSD_coh          =  np.sum(index_coh) / len(index_coh)  * coherent_sum
        PSD_non_coh      =  np.sum(index_non_coh) / len(index_coh) * non_coherent_sum
        Trace_PSD        = PSD_coh + PSD_non_coh

        return {
                    'sigma_xy'        : sigma_xy,
                    'sigma_xy_par'    : sigma_xy_par,
                    'sigma_xy_per'    : sigma_xy_per,
                    'sigma_yz'        : sigma_yz,
                    'sigma_yz_par'    : sigma_yz_par,
                    'sigma_yz_per'    : sigma_yz_per,
                    'PSD_coh'         : PSD_coh,
                    'PSD_non_coh'     : PSD_non_coh,
                    'counts_par'      : np.nansum(index_par),
                    'counts_per'      : np.nansum(index_per),

                    'counts_coh'      : np.nansum(index_coh),
                    'counts_non_coh'  : np.nansum(index_non_coh),
                    'counts_Trace'    : len(df_w.dropna()),
                    'coh_thresh'      : func_params['coh_th']
        }


                                            
def return_desired_quants( df_w,
                           df_mod,
                           S0,
                           S3,
                           S0_full,
                           Syz,
                           VBangles,
                           dt,
                           scale,
                           psd_norm,
                           func_params = None):
    

    
    if func_params["estimate_PSDs"] or func_params['estimate_coh_coeffs']:
        # Find times where sampling is quasi-par(perp)
        
        if func_params["use_rolling_mean"]:
            index_per   = VBangles > func_params['per_thresh']
            index_par   = VBangles < func_params['par_thresh']            
            
        else:
            index_per   = (np.where(VBangles > func_params['per_thresh'])[0]).astype(np.int64)
            index_par   = (np.where(VBangles < func_params['par_thresh'])[0]).astype(np.int64)
        

    else:
        index_per   = None
        index_par   = None
    
    
    # Do polarization analysis
    coh_res = do_coh_analysis(df_w, S0, S3, S0_full, Syz, index_par, index_per, dt, scale, psd_norm, func_params=func_params)
    
    # Estimate Anisotropic PSD
    anis_res = est_anisotropic_PSDs(df_w, df_mod, index_par, index_per, dt, psd_norm, func_params=func_params)
    
    # Estimate Structure functions
    sf_res = est_sfuncs(df_w,
                        df_mod,
                        index_par,
                        index_per,
                        scale,
                        dt,
                        func_params = func_params )
    
    # Estimate compressibility diagnostics
    comp_res = est_compress(df_mod, df_w, index_par, index_per, dt,psd_norm, func_params=func_params)

    # Merge all dictionaries into one and return
    return {**coh_res, **anis_res, **comp_res, **sf_res}


def anisotropy_coherence2(  
                           B_df,
                           V_df, 
                           field_flag,
                           E_df                  =  None,
                           method                =  'min_var',
                           func_params           =  None,
                           f_dict                =  None
                          ):
    """
    Method to calculate the 1) wavelet coefficients in RTN 2) The scale dependent angle between Vsw and Β.

    Parameters:
        B_df (pandas.DataFrame): Magnetic field timeseries dataframe.
        V_df (pandas.DataFrame): Velocity timeseries dataframe.
        dj (float): The time resolution.
        alpha (float, optional): Gaussian parameter. Default is 3.
        pycwt (bool, optional): Use the PyCWT library for wavelet transform. Default is False.

    Returns:
        tuple: A tuple containing the following elements:
            np.ndarray: Frequencies in the x-direction.
            np.ndarray: Frequencies in the y-direction.
            np.ndarray: Frequencies in the z-direction.
            pandas.DataFrame: Angles between magnetic field and scale dependent background in degrees.
            pandas.DataFrame: Angles between velocity and scale dependent background in degrees.
            np.ndarray: Frequencies in Hz.
            np.ndarray: Power spectral density.
            np.ndarray: Physical space scales in seconds.
            np.ndarray: Wavelet scales.
    """
    

    def define_B_df(B_df):
    
        B_df['RR'] = B_df.R* B_df.R
        B_df['TT'] = B_df['T']* B_df['T']
        B_df['NN'] = B_df.N* B_df.N

        B_df['RT'] = B_df.R* B_df['T']
        B_df['RN'] = B_df.R* B_df.N
        B_df['TN'] = B_df['T']* B_df.N
        return B_df
              

    def define_W_df(B_index, R, T, N):
        return      pd.DataFrame({ 'DateTime' : B_index,
                                    'R'       : R,
                                    'T'       : T,
                                    'N'       : N}).set_index('DateTime')

    def parallel_oper(ii, 
                      scale,
                      dt,
                      B_df,
                      V_df, 
                      df_w,
                      df_mod,
                      psd_norm,
                      method      = 'min_var',
                      func_params = None):
        try:

            if func_params['do_coherence_analysis']:

                if method =='min_var':
                    B_df = define_B_df(B_df)
                          
                # Do coherence analysis
                df_w, VBangles, S3, S0, Syz, S0_full       = coherence_analysis(
                                                                        B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0),
                                                                        V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0),
                                                                        df_w,
                                                                        method,
                                                                        func_params       = func_params)
            else:

                B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0)
                

                if func_params['estimate_local_V']:
                    V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0)
                    

                # Estimate the unit vectors
                B_df = unit_vectors(B_df, prefix = 'B_0_', vector_cols= ['R', 'T', 'N'])
                V_df = unit_vectors(V_df, prefix = 'V_0_', vector_cols= ['R', 'T', 'N'])
                    
                # Estimate angle between local backgrounds
                VBangles = np.degrees(np.arccos(np.einsum('ij,ij->i',
                                                          B_df[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].values, 
                                                          V_df[['V_0_R_hat', 'V_0_T_hat', 'V_0_N_hat']].values)))
                
                
                S3, S0, Syz = None, None, None
                
                
            # Restrict VB angles
            VBangles[VBangles > 90] = 180 - VBangles[VBangles > 90]
            
            #Estimate Anistropic Power Spectra
            est_quants = return_desired_quants(df_w,
                                               df_mod,
                                               S0,
                                               S3,
                                               S0_full, 
                                               Syz,
                                               VBangles,
                                               dt,
                                               scale,
                                               psd_norm,
                                               func_params = func_params)

            if func_params['return_coeffs'] is False:
                return est_quants, None, None, None, None, None
                

            return est_quants, VBangles, df_w.values.T, S3, S0, Syz
        except Exception as e:
            traceback.print_exc()
            return np.nan, np.nan
                      
                      
    print('Using', func_params['njobs'], 'cores')      
        
    # Rename the columns
    if B_df.columns[0] =='Bx':
        B_df  = B_df.rename(columns={'Bx': 'R', 'By': 'T', 'Bz': 'N'}) 
        V_df  = V_df.rename(columns={'Vx': 'R', 'Vy': 'T', 'Vz': 'N'})
        
        
        
        if (method =='TN_only') & (E_df is not None):
            
            if field_flag =='N_p':
                
                E_df['R'] =   E_df['np']
                E_df['T'] =   0*E_df['R']
                E_df['N'] =   0*E_df['R']
                
            else:

                E_df['R'] = 0*E_df['Ey']
                E_df['T'] =   E_df['Ex']
                E_df['N'] = - E_df['Ey']

            
            #Drop original 'R', 'T', 'N'
            E_df.drop(columns=['Ex', 'Ey'], inplace=True, errors='ignore')
        else:
            E_df  = None if E_df is None else E_df.rename(columns={'Ex': 'R', 'Ey': 'T', 'Ez': 'N'})
    else:
        B_df  = B_df.rename(columns={'Br': 'R', 'Bt': 'T', 'Bn': 'N'})
        V_df  = V_df.rename(columns={'Vr': 'R', 'Vt': 'T', 'Vn': 'N'})
        E_df  = None if E_df is None else E_df.rename(columns={'Er': 'R', 'Et': 'T', 'En': 'N'})


    # Estimate sampling times of time series
    dt_B, dt_V = func.find_cadence(B_df), func.find_cadence(V_df)
    dt_E       = func.find_cadence(E_df) if E_df is not None else None
    
    

    # Synchronize E_df and B_df if necessary
    #if E_df is not None and dt_E != dt_B:
       # print('Got here', len(B_df), len(E_df))
        
    if E_df is not None :
        E_df, B_df = func.synchronize_dfs(E_df, B_df,True)#(B_df, E_df.index)
       


    # Synchronize B_df and V_df if necessary
    if dt_V != dt_B:
        B_df, V_df = func.synchronize_dfs(B_df, V_df,True)#(B_df, E_df.index)

    # Determine the common dt
    dt = dt_E if dt_E is not None else dt_B
    
    # Estimate wavelet coefficients
    if func_params['use_custom_cwt']:
        Wvec, scales, freqs, psd_norm, coi= estimate_cwt(E_df if E_df is not None else  B_df, 
                                                      dt,
                                                      nv       = func_params['nv'],
                                                      min_freq = func_params['min_freq'],
                                                      use_omp  = func_params['open_mp'])
    else:

        Wvec, scales, freqs, psd_norm, coi= estimate_cwt_old(E_df if E_df is not None else  B_df, 
                                                      dt,
                                                      nv       = func_params['nv'],
                                                      min_freq = func_params['min_freq'])



    # Estimate magnitude of magnetic field
    if func_params['est_mod']:
        if func_params['use_custom_cwt']:
            Wmod, _, _, _ ,_= estimate_cwt(np.sqrt(B_df.values.T[0]**2 + B_df.values.T[1]**2 + B_df.values.T[2]**2),
                                            dt, 
                                            nv       = func_params['nv'],
                                            min_freq = func_params['min_freq'],
                                            use_omp  = func_params['open_mp'])
        else:
            Wmod, _, _, _ ,_= estimate_cwt(np.sqrt(B_df.values.T[0]**2 + B_df.values.T[1]**2 + B_df.values.T[2]**2),
                                            dt, 
                                            nv       = func_params['nv'],
                                            min_freq = func_params['min_freq'])            
        
        # Wmod, _, _, _ = estimate_cwt(np_df.values.T[0],
        #                         dt, 
        #                         nv       = func_params['nv'],
        #                         min_freq = func_params['min_freq'],
        #                          use_omp  = func_params['open_mp'])
    else:
        Wmod             = None
        
    
    # Initialize arrays
    PSD_par = np.zeros(len(freqs))
    PSD_per = np.zeros(len(freqs)) 
 
    PSD_par_mod = np.zeros(len(freqs))
    PSD_per_mod = np.zeros(len(freqs))

    # Use joblib for parallel processing
    results = Parallel(n_jobs=func_params['njobs'])(
        delayed(parallel_oper)(
                                ii, 
                                scale,
                                dt,
                                B_df.copy(),
                                V_df.copy(), 
                                define_W_df(B_df.index, Wvec['R'][ii], Wvec['T'][ii], Wvec['N'][ii]),
                                Wmod[ii],
                                psd_norm,
                                method                = method,
                                func_params           = func_params
        ) for ii, scale in tqdm(enumerate(scales), total=len(scales))
    )


    
    # Unpack results
    est_quants, VBangles, df_w, S3, S0, Syz = zip(*results)
    
    # Initialize the dictionary and populate it in one step
    
    #field_flag            = 'E' if E_df is not None else  'B'
    if f_dict             == None:
        
        if func_params['use_rolling_mean']:
            f_dict                = {field_flag : {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}}
        else:
            f_dict                = {field_flag : {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}}           
    else:
        if func_params['use_rolling_mean']:
            f_dict[field_flag ]    = {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}
        else:
            f_dict[field_flag ]   =  {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}       
    f_dict['freqs']       = freqs
    f_dict['scales']      = scales
    f_dict['Wave_coeffs'] = df_w
    f_dict['S3_ts']       = S3
    f_dict['S0_ts']       = S0
    f_dict['VB_ts']       = VBangles
    f_dict['flag']        = method
    
    return f_dict  




from scipy.signal import stft

def TN_polarization_stft(E_df, B_df, V_df, sig, fs, 
                         window_duration=1.0, overlap_fraction=0.5):
    """
    Process electric and magnetic field data using STFT and compute the average angle between
    B and V vectors over the same windows used in the STFT.

    Parameters:
    E_df (DataFrame): Electric field data with columns ['Ex', 'Ey', 'Ez']
    B_df (DataFrame): Magnetic field data with columns ['Bx', 'By', 'Bz']
    V_df (DataFrame): Velocity data with columns ['Vx', 'Vy', 'Vz']
    sig (DataFrame): Signal data with columns ['sigma_c', 'd_i', 'rho_ci', 'Vsw']
    fs (float): Sampling frequency in Hz
    window_duration (float): Duration of each window in seconds
    overlap_fraction (float): Fraction of window overlap (0 to 1)

    Returns:
    f (ndarray): Array of sample frequencies.
    Et (ndarray): STFT of the transverse electric field component.
    En (ndarray): STFT of the normal electric field component.
    avg_angles (ndarray): Average angles over each window.
    sig_c (ndarray): Averaged sigma_c over each window.
    di (ndarray): Averaged d_i over each window.
    rhoi (ndarray): Averaged rho_ci over each window.
    Vsw (ndarray): Averaged Vsw over each window.
    """

    import numpy as np

    # Calculate window size and overlap in samples
    window_size = int(window_duration * fs)
    noverlap = int(overlap_fraction * window_size)
    step = window_size - noverlap

    # Compute the angle between B and V vectors
    angles = func.angle_between_vectors(B_df.values, V_df.values)

    try:
        # Process E_df
        E_df['T'] = E_df['Ex']
        E_df['N'] = -E_df['Ey']
        # Drop original 'Ex', 'Ey'
        E_df.drop(columns=['Ex', 'Ey'], inplace=True, errors='ignore')
    except:
        # Process E_df
        E_df['T'] = E_df['Bx']
        E_df['N'] = -E_df['By']
        # Drop original 'Bx', 'By'
        E_df.drop(columns=['Bx', 'By'], inplace=True, errors='ignore')

    # Compute STFT of 'T' and 'N' components
    f, t_stft, Et = stft(
        E_df['T'].values, fs=fs, window='hann',
        nperseg=window_size, noverlap=noverlap,
    )
    _, _, En = stft(
        E_df['N'].values, fs=fs, window='hann',
        nperseg=window_size, noverlap=noverlap
    )

    # Compute average angles and other parameters over the same windows used in the STFT
    n_segments    = len(t_stft)
    signal_length = len(angles)
    avg_angles    = np.empty(n_segments)
    sig_c         = np.empty(n_segments)
    di            = np.empty(n_segments)
    rhoi          = np.empty(n_segments)
    Vsw           = np.empty(n_segments)
    counts        = np.empty(n_segments)
    for i in range(n_segments):
        # Compute start and end indices
        start = int(np.round(t_stft[i] * fs - window_size / 2))
        end = start + window_size
        # Ensure indices are within bounds
        start = max(start, 0)
        end = min(end, signal_length)
        avg_angles[i] = np.nanmean(angles[start:end])
        sig_c[i]      = np.nanmean(np.abs(sig['sigma_c'].values[start:end]))
        di[i]         = np.nanmean(sig['d_i'].values[start:end])
        rhoi[i]       = np.nanmean(sig['rho_ci'].values[start:end])
        Vsw[i]        = np.nanmean(sig['Vsw'].values[start:end])
        counts[i]     = len(sig['Vsw'].values[start:end])

    return f, Et, En, avg_angles, sig_c, di, rhoi, Vsw, counts




# def coherence_analysis(df_w,
#                        B0_f_o,
#                        V0_f_o,
#                        freq,
#                        dt, 
#                        min_var       = True
#                       ):
    


#     @njit(parallel=True)
#     def compute_first_eigenvectors(RRe, RTe, RNe, TTe, TNe, NNe):
#         n = RRe.shape[0]
#         eigen_vectors = np.empty((n, 3), dtype=np.float64)

#         for i in prange(n):
#             # Construct the symmetric 3x3 matrix
#             M = np.array([
#                 [RRe[i], RTe[i], RNe[i]],
#                 [RTe[i], TTe[i], TNe[i]],
#                 [RNe[i], TNe[i], NNe[i]]
#             ])

#             # Compute eigenvalues and eigenvectors
#             eigvals, eigvecs = np.linalg.eigh(M)

#             # Store the eigenvector corresponding to the largest eigenvalue (maximum variance)
#             eigen_vectors[i, :] = eigvecs[:, -1]

#         return eigen_vectors
    

    
#     def unit_eigenvector_computation(df, prefix='eigen'):
#         RRe = df['RRe'].values
#         RTe = df['RTe'].values
#         RNe = df['RNe'].values
#         TTe = df['TTe'].values
#         TNe = df['TNe'].values
#         NNe = df['NNe'].values

#         eigen_vectors = compute_first_eigenvectors(RRe, RTe, RNe, TTe, TNe, NNe)

#         df[[f"{prefix}_1_hat", f"{prefix}_2_hat", f"{prefix}_3_hat"]] = eigen_vectors
        
#         # Estimate unit vector
#         return unit_vectors(df,
#                            prefix      = '',
#                            sufix       = '', 
#                            vector_cols =['eigen_1_hat', 'eigen_2_hat', 'eigen_3_hat'])

#     # Estimate the unit vectors
#     B0_f_o = unit_vectors(B0_f_o, prefix = 'B_0_', vector_cols= ['R', 'T', 'N'])
#     V0_f_o = unit_vectors(V0_f_o, prefix = 'V_0_', vector_cols= ['R', 'T', 'N'])
  
#     # Estimate angle between local backgrounds
#     VBangles = np.degrees(np.arccos(np.einsum('ij,ij->i',
#                                               B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].values, 
#                                               V0_f_o[['V_0_R_hat', 'V_0_T_hat', 'V_0_N_hat']].values)))
    
#     if min_var:
  
#         # Calculate matrix elements
#         B0_f_o['RRe'] = B0_f_o['RR'] - np.square(B0_f_o['R'])
#         B0_f_o['TTe'] = B0_f_o['TT'] - np.square(B0_f_o['T'])
#         B0_f_o['NNe'] = B0_f_o['NN'] - np.square(B0_f_o['N'])
#         B0_f_o['RTe'] = B0_f_o['RT'] - B0_f_o['R'] * B0_f_o['T']
#         B0_f_o['RNe'] = B0_f_o['RN'] - B0_f_o['R'] * B0_f_o['N']
#         B0_f_o['TNe'] = B0_f_o['TN'] - B0_f_o['T'] * B0_f_o['N']

#         # Find eigenvectors
#         B0_f_o = unit_eigenvector_computation(B0_f_o, prefix='eigen')
        

#         # Calculate the first perpendicular unit vector
#         B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']] = np.cross(B0_f_o[['eigen_1_hat', 'eigen_2_hat', 'eigen_3_hat']], 
#                                                                    B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']])
  

#         # Calculate second perpendicular unit vector
#         B0_f_o[['B_2_R_hat', 'B_2_T_hat', 'B_2_N_hat']] = np.cross(B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']], 
#                                                                    B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']])

#         # Memory cleanup by dropping intermediate columns
#         columns_to_drop = [
#                            # 'B_1_R', 'B_1_T', 'B_1_N', 
#                             'RR', 'TT', 'NN', 'RT', 'RN', 'TN', 
#                             'RRe', 'TTe', 'NNe', 'RTe', 'RNe', 'TNe'
#         ]
#         B0_f_o.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        
#         # Extract necessary arrays without copying
#         B0    = B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].to_numpy(copy=False)
#         B1    = B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']].to_numpy(copy=False)
#         B2    = B0_f_o[['B_2_R_hat', 'B_2_T_hat', 'B_2_N_hat']].to_numpy(copy=False)
#         r_t_n = df_w[['R', 'T', 'N']].to_numpy(copy=False)

#         # Compute Wz, Wy, and Wx
#         df_w['W0'] = np.einsum('ij,ij->i', B0, r_t_n)
#         df_w['W1'] = np.einsum('ij,ij->i', B1, r_t_n)
#         df_w['W2'] = np.einsum('ij,ij->i', B2, r_t_n)
        
#         df_w, _, _, _ = estimate_cwt(df_w[['W0', 'W1', 'W2']], dt, freqs= np.array([freq]), return_df =True,  col_names = ['W0', 'W1', 'W2'])

        
#         return df_w,  VBangles, 2*np.imag( (np.conj(df_w['W2'])*df_w['W1'])), (np.abs(df_w['W1'])**2+np.abs(df_w['W2'])**2), -2*np.imag((np.conj(df_w['W0'])*df_w['W1']))

        
#     else:
#         #print('Using Loyds method')
        

#         # Calculate second perpendicular unit vector
#         B0_f_o[['B_y_R', 'B_y_T', 'B_y_N']]             =  np.cross( B0_f_o[['R', 'T', 'N']],
#                                                                      V0_f_o[['R', 'T', 'N']])
          
#         B0_f_o                                          =   unit_vectors(B0_f_o,
#                                                                          prefix       = '', 
#                                                                          #sufix        = '', 
#                                                                          vector_cols  = ['B_y_R', 'B_y_T', 'B_y_N'])
        
 
#         # Extract necessary arrays without copying
#         Bz    = B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].to_numpy(copy=False)
#         By    = B0_f_o[['B_y_R_hat', 'B_y_T_hat', 'B_y_N_hat']].to_numpy(copy=False)
#         Bx    = np.cross(By, Bz) 

        
#         r_t_n = df_w[['R', 'T', 'N']].to_numpy(copy=False)

#         # Project the data  (X,Y,Z)
#         df_w['Wx'] = np.einsum('ij,ij->i', Bx, r_t_n)
#         df_w['Wy'] = np.einsum('ij,ij->i', By, r_t_n)
#         df_w['Wz'] = np.einsum('ij,ij->i', Bz, r_t_n)

#         # Estimate wavelet coefficints
#         #df_w, _, _, _ = estimate_cwt(df_w[['Wx', 'Wy', 'Wz']], dt, freqs= np.array([freq]), return_df =True,  col_names = ['Wx', 'Wy', 'Wz'])
#         df_w, _, _, _ = estimate_cwt(df_w[['R', 'T', 'N']], dt, freqs= np.array([freq]), return_df =True,  col_names = ['R', 'T', 'N'])
    

#         #return df_w,  VBangles, 2*np.imag( (np.conj(df_w['Wy'])*df_w['Wx'])), (np.abs(df_w['Wy'])**2 + np.abs(df_w['Wx'])**2 + np.abs(df_w['Wz'])**2), 2*np.imag((np.conj(df_w['Wy'])*df_w['Wz']))
#         return df_w,  VBangles, 2*np.imag( (np.conj(df_w['N'])*df_w['T'])), (np.abs(df_w['R'])**2 + np.abs(df_w['T'])**2 + np.abs(df_w['N'])**2), 2*np.imag((np.conj(df_w['T'])*df_w['R']))

    
# import pycwt
# def estimate_cwt(signal, dt, freqs=None, omega0 =6.0, return_df =False, col_names = ['R', 'T', 'N']):

#     """
#     Estimate continuous wavelet transform of the signal using PyWavelets.

#     Parameters:
#     - signal (pd.DataFrame or np.ndarray): Input signal(s).
#     - dt (float): Sampling interval.
#     - freqs (np.ndarray or None): Frequencies at which to compute the CWT. If None, frequencies are computed automatically.
#     - dj: determines how many scales are used to estimate wavelet coeff

#         (e.g., for dj=1 -> 2**numb_scales 

#     Returns:
#     - w_df (dict or np.ndarray): Wavelet coefficients per column or array.
#     - scales_used (np.ndarray): Scales used.
#     - freqs_used (np.ndarray): Frequencies corresponding to scales.
#     - coi (None): Cone of influence (not computed here).
#     """

#     # Now, compute scales and frequencies
#     if freqs is not None:
#         # Ensure freqs is an array
#         freqs = np.array(np.asarray(freqs).astype(float))

#         # Compute corresponding scales
#         scales = (omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * omega0**2))


#     # Perform the CWT
#     if isinstance(signal, pd.DataFrame):
#         if return_df:
#             w_df             = pd.DataFrame()
#             w_df['datetime'] = signal.index.values
#         else:
#             w_df = {}
            
#         for jj, (col, col_name) in enumerate(zip(signal.columns, col_names)):

#             coeffs, _, freqs, _, _, _ = pycwt.cwt(signal[col].values, dt, wavelet=pycwt.Morlet(), freqs= freqs)

#             print(col, col_name)
#             if len(scales) == 1:
#                 w_df[col_name] = coeffs[0, :]
#             else:
#                 w_df[col_name] = coeffs

#     else:
#         coeffs, _, freqs, _, _, _ = pycwt.cwt(signal, dt, wavelet=pycwt.Morlet(), freqs= freqs)
#         if len(scales) == 1:
#             w_df = coeffs[0, :]
#         else:
#             w_df = coeffs

#     coi = None  # Cone of influence not computed

#     if return_df:
#         w_df = w_df.set_index('datetime')

#     return w_df, scales, freqs, coi


# def anisotropy_coherence(
#                                B_df,
#                                V_df, 
#                                E_df                  = None,
#                                dt                    = 0, 
#                                nv                    = 32,
#                                alpha                 = 1, 
#                                per_thresh            = 80,
#                                par_thresh            = 10,
#                                coh_th                = 0.7,
#                                njobs                 = -1,
#                                est_mod               = True,
#                                estimate_local_V      = False,
#                                min_var               = False,
#                                do_coherence_analysis = False,
#                                estimate_PSDs         = False,
#                                estimate_coh_coeffs   = False,
#                                return_coeffs         = True 
#                               ):
#     """
#     Method to calculate the 1) wavelet coefficients in RTN 2) The scale dependent angle between Vsw and Β.

#     Parameters:
#         B_df (pandas.DataFrame): Magnetic field timeseries dataframe.
#         V_df (pandas.DataFrame): Velocity timeseries dataframe.
#         dj (float): The time resolution.
#         alpha (float, optional): Gaussian parameter. Default is 3.
#         pycwt (bool, optional): Use the PyCWT library for wavelet transform. Default is False.

#     Returns:
#         tuple: A tuple containing the following elements:
#             np.ndarray: Frequencies in the x-direction.
#             np.ndarray: Frequencies in the y-direction.
#             np.ndarray: Frequencies in the z-direction.
#             pandas.DataFrame: Angles between magnetic field and scale dependent background in degrees.
#             pandas.DataFrame: Angles between velocity and scale dependent background in degrees.
#             np.ndarray: Frequencies in Hz.
#             np.ndarray: Power spectral density.
#             np.ndarray: Physical space scales in seconds.
#             np.ndarray: Wavelet scales.
#     """
    
    


#     def generate_scales_2_use(N, dt, nv=32, omega0=6):
#         fs           = 1 / dt  # Sampling frequency
#         T            = N * dt
#         f_min        = 1 / T
#         f_max        = fs / 2
#         num_octaves  = np.log2(f_max / f_min)
#         num_freqs    = nv * int(np.ceil(num_octaves))
#         freqs_used   = np.logspace(np.log10(f_min), np.log10(f_max), num=num_freqs, base=10)
#         scales_used  = omega0 / (2 * np.pi * freqs_used) * (1 + 1 / (2 * omega0**2))

#         return freqs_used, scales_used

    

#     def define_B_df(B_df):
    
#         B_df['RR'] = B_df.R* B_df.R
#         B_df['TT'] = B_df['T']* B_df['T']
#         B_df['NN'] = B_df.N* B_df.N

#         B_df['RT'] = B_df.R* B_df['T']
#         B_df['RN'] = B_df.R* B_df.N
#         B_df['TN'] = B_df['T']* B_df.N
#         return B_df
              

#     def define_W_df(B_index, R, T, N):
#         return      pd.DataFrame({ 'DateTime' : B_index,
#                                     'R'       : R,
#                                     'T'       : T,
#                                     'N'       : N}).set_index('DateTime')
    
    
                    


#     def parallel_oper(ii, 
#                       scale,
#                       freq,
#                       dt,
#                       CWT_df,
#                       B_df,
#                       V_df, 
#                       # df_w,
#                       # df_mod,
#                       alpha,
#                       per_thresh            = 80,
#                       par_thresh            = 10,
#                       coh_th                = 0.7,
#                       njobs                 = -1,
#                       est_mod               = False,
#                       estimate_local_V      = False,
#                       min_var               = False,
#                       do_coherence_analysis = False,
#                       estimate_PSDs         = False,
#                       estimate_coh_coeffs   = False,
#                       return_coeffs         = True):
#         try:

#             if do_coherence_analysis:

#                 if min_var:
#                     B_df = define_B_df(B_df)
                          
                        

#                 # Do coherence analysis
#                 df_w, VBangles, S3, S0, Syz       = coherence_analysis(CWT_df,
#                                                                        B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =alpha), axis=0),
#                                                                        V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =alpha), axis=0),
#                                                                        freq,
#                                                                        dt,
#                                                                        min_var       = min_var)
#             else:

#                 df_w, _, _, _ = estimate_cwt(CWT_df, dt, freqs= np.array([freq]), return_df =True)
                

                    
                    
                
                
#                 B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =alpha), axis=0)
                

#                 if estimate_local_V:
#                     V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =alpha), axis=0)
                    

#                 # Estimate the unit vectors
#                 B_df = unit_vectors(B_df, prefix = 'B_0_', vector_cols= ['R', 'T', 'N'])
#                 V_df = unit_vectors(V_df, prefix = 'V_0_', vector_cols= ['R', 'T', 'N'])
                    
#                 # Estimate angle between local backgrounds
#                 VBangles = np.degrees(np.arccos(np.einsum('ij,ij->i',
#                                                           B_df[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].values, 
#                                                           V_df[['V_0_R_hat', 'V_0_T_hat', 'V_0_N_hat']].values)))
                
                
#                 S3, S0, Syz = None, None, None
                
                
#             # Restrict VB angles
#             VBangles[VBangles > 90] = 180 - VBangles[VBangles > 90]
            
            
#             if est_mod:
#                 df_mod, _, _, _ = estimate_cwt(np.sqrt(CWT_df.values.T[0]**2 + CWT_df.values.T[1].values**2 + CWT_df.values.T[2].values**2),
#                                                  dt,
#                                                  freqs     = np.array([freq]),
#                                                  return_df = False)
#             else:
#                 df_mod          = None
            
#             #Estimate Anistropic Power Spectra
#             est_quants = return_desired_quants(df_w,
#                                                df_mod,
#                                                S0,
#                                                S3,
#                                                Syz,
#                                                VBangles,
#                                                dt,
#                                                scale,
#                                                alpha                 = alpha,
#                                                num_efoldings         = 3, 
#                                                coh_th                = coh_th,
#                                                par_thresh            = par_thresh,
#                                                per_thresh            = per_thresh,
#                                                estimate_PSDs         = estimate_PSDs,
#                                                estimate_coh_coeffs   = estimate_coh_coeffs,
#                                                est_mod               = est_mod)
            
            
            
       
#             if return_coeffs is False:
#                 return est_quants, None, None, None, None, None
                

#             return est_quants, VBangles, df_w.values.T, S3, S0, Syz
#         except Exception as e:
#             traceback.print_exc()
#             return np.nan, np.nan
        
        
#     # Rename the columns
#     if B_df.columns[0] =='Bx':
#         B_df  = B_df.rename(columns={'Bx': 'R', 'By': 'T', 'Bz': 'N'}) 
#         V_df  = V_df.rename(columns={'Vx': 'R', 'Vy': 'T', 'Vz': 'N'})
#         E_df  = None if E_df is None else E_df.rename(columns={'Ex': 'R', 'Ey': 'T', 'Ez': 'N'})
#     else:
#         B_df  = B_df.rename(columns={'Br': 'R', 'Bt': 'T', 'Bn': 'N'})
#         V_df  = V_df.rename(columns={'Vr': 'R', 'Vt': 'T', 'Vn': 'N'})
#         E_df  = None if E_df is None else E_df.rename(columns={'Er': 'R', 'Et': 'T', 'En': 'N'})

 
#     print(B_df.columns)

#     print('Using', njobs, 'cores')
#     print('C1')
#     # Estimate sampling times of time series
#     dt_B, dt_V = func.find_cadence(B_df), func.find_cadence(V_df)
#     dt_E       = func.find_cadence(E_df) if E_df is not None else None

#     # Synchronize E_df and B_df if necessary
#     if E_df is not None and dt_E != dt_B:
#         B_df = func.newindex(B_df, E_df.index)

#     print('C2')
#     # Synchronize B_df and V_df if necessary
#     if dt_V != dt_B:
#         V_df = func.newindex(V_df, B_df.index)

#     print('C3')
#     # Determine the common dt
#     dt = dt_E if dt_E is not None else dt_B

#     # print('C4')
#     # # Estimate wavelet coefficients
#     # Wvec, scales, freqs, coi       = estimate_cwt(E_df if E_df is not None else  B_df, 
#     #                                               dt,
#     #                                               nv = nv)
#     # print('C5')
#     # # Estimate magnitude of magnetic field
#     # if est_mod:
#     #     Wmod, _, _, _, _ = estimate_cwt(np.sqrt(B_df.values.T[0]**2 + B_df.values.T[1].values**2 + B_df.values.T[2].values**2), dt, nv=nv)
#     # else:
#     #     Wmod             = None

    
#     # Define scales and frequencies to estimate the CWT
#     freqs, scales  = generate_scales_2_use(len(E_df)if E_df is not None else  len(B_df), dt, nv=nv)
    
#     # Define the dataframe to estimate the CWT
#     CWT_df = E_df.copy() if E_df is not None else  B_df.copy()
 
    
#     PSD_par = np.zeros(len(freqs))
#     PSD_per = np.zeros(len(freqs)) 
 
#     PSD_par_mod = np.zeros(len(freqs))
#     PSD_per_mod = np.zeros(len(freqs))

#     # Use joblib for parallel processing
#     print('C6')
#     print('Using', njobs, 'cores')

#     # Assuming the scales and other parameters are already defined
#     results = Parallel(n_jobs=njobs)(
#         delayed(parallel_oper)(
#             ii, 
#             scale,
#             freq,
#             dt,
#             CWT_df.copy(),
#             B_df.copy(),
#             V_df.copy(), 
#             alpha                 = alpha,
#             per_thresh            = per_thresh,
#             par_thresh            = par_thresh,
#             coh_th                = coh_th, 
#             njobs                 = njobs,
#             est_mod               = est_mod,
#             estimate_local_V      = estimate_local_V,
#             min_var               = min_var,
#             do_coherence_analysis = do_coherence_analysis,
#             estimate_PSDs         = estimate_PSDs,
#             return_coeffs         = return_coeffs,
#             estimate_coh_coeffs    = estimate_coh_coeffs
#         ) for ii, (freq, scale) in tqdm(enumerate(zip(freqs, scales)), total=len(scales))
#     )


    
#     # Unpack results
#     #PSD_par, PSD_per,PSD_par_mod, PSD_per_mod, sigma_xy, sigma_xy_par, sigma_xy_per, PSD_coh, PSD_non_coh, overall_PSD,  VBangles, df_w, S3, S0 = zip(*results)
#     est_quants, VBangles, df_w, S3, S0, Syz = zip(*results)
    
    

#     return est_quants, freqs, scales,  VBangles, df_w, S3, S0, Syz





# Outdated functions

def estimate_polarization(num_coh, den_coh, scales, dt, alpha=1, num_efoldings=3, n_jobs=-1):
    """
    Estimates the polar values based on the ratio of local Gaussian averaged numerator
    to the local Gaussian averaged denominator across different scales.

    Parameters:
    num_coh (list): Numerator values for computation.
    den_coh (list): Denominator values for computation.
    scales (list): List of scales at which the local averaging is done.
    dt (float): Time step used in the local Gaussian averaging.
    alpha (int, optional): Alpha parameter for the Gaussian averaging. Default is 1.
    num_efoldings (int, optional): Number of e-foldings in the Gaussian averaging. Default is 1.
    n_jobs (int, optional): The number of parallel jobs to run. Default is -1 (use all processors).

    Returns:
    list: A list of sigma values computed as the ratio of averaged values.
    """
    

    def compute_ratio(i, num_coh, den_coh, scales, dt, alpha, num_efoldings):
        num_value = local_gaussian_averaging(num_coh[i], dt,  scales[i],  alpha =alpha, num_efoldings = num_efoldings)
        den_value = local_gaussian_averaging(den_coh[i], dt, scales[i],  alpha = alpha, num_efoldings = num_efoldings)
        return num_value / den_value

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_ratio)(i, num_coh, den_coh, scales, dt, alpha, num_efoldings)
        for i in range(len(num_coh))
    )
    return results





def choose_dates_heatmap(freqs, inds,  data, original, target):
    fe = []
    dt = []
    increment = original // target
    for i in range (0, len(freqs), increment):
        fe.append(freqs[i])
        dt.append(data[i][inds[0]: inds[1]])
    return np.array(fe), np.array(dt)



