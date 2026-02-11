import warnings
warnings.filterwarnings('ignore')

import os
import sys
import traceback
import pickle
from pathlib import Path
from datetime import datetime
import statistics
from statistics import mode

# Scientific computing
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import stft


# Performance
import numba
from numba import jit, njit, prange, objmode
import joblib
from joblib import Parallel, delayed

# Progress bar
from tqdm import tqdm

# Third-party signal processing
import ssqueezepy
#from CWTPy import cwt_module

# Local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(REPO_ROOT / 'functions'))
import TurbPy as turb
import general_functions as func


def estimate_cwt(signal,
                 dt,
                 nv            = 32,
                 omega0        = 6,
                 scale_type    = 'log-piecewise',
                 vectorized    = True,
                 l1_norm       = False,
                 min_freq = None,max_freq = None):
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
                indices = np.where((freqs >= min_freq) & (freqs<= max_freq) )[0]
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
        freqs  = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal), fs)
        scales = (omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * omega0**2))

        # Remove the first five scales and corresponding coefficients
        W       = W[5:, :]
        scales  = scales[5:]
        freqs   = freqs[5:]

        # Remove frequencies lower than min_freq
        if min_freq is not None:
            indices = np.where((freqs >= min_freq) & (freqs<= max_freq) )[0]
            W       = W[indices, :]
            scales  = scales[indices]
            freqs   = freqs[indices]

        w_df = W

    coi = None

    return w_df, scales, freqs, 2*dt, coi


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



# ───────────────────────────────────────────────────────────────
#  helper: row-wise normalisation of arbitrary vector columns
# ───────────────────────────────────────────────────────────────
def unit_vectors(df, prefix='', sufix='_hat', vector_cols=None):
    if vector_cols is None:
        raise ValueError("`vector_cols` must be supplied.")
    arr  = df[vector_cols].to_numpy(float, copy=False)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    df[[f"{prefix}{c}{sufix}" for c in vector_cols]] = arr
    return df


# ───────────────────────────────────────────────────────────────
#  coherence_analysis (now k̂ is *always* estimated on-the-fly)
# ───────────────────────────────────────────────────────────────


def coherence_analysis(
    B0_f_o,
    V0_f_o,
    df_w,
    method,
    func_params=None,
    return_geom: bool = False,
):
    import numpy as np

    if func_params is None:
        func_params = {}
    rotate_to_k = bool(func_params.get("rotate_to_k", False))

    eps = 1e-12

    def _safe_unit(arr):
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        n = np.where(n < eps, np.nan, n)
        return arr / n, n

    # -----------------------------------------------------------
    # 0) unit vectors of background B0 and V0 (adds *_hat columns)
    # -----------------------------------------------------------
    B0_f_o = unit_vectors(B0_f_o, prefix="B_0_", vector_cols=["R", "T", "N"])
    V0_f_o = unit_vectors(V0_f_o, prefix="V_0_", vector_cols=["R", "T", "N"])

    B_hat = B0_f_o[["B_0_R_hat", "B_0_T_hat", "B_0_N_hat"]].to_numpy(float, copy=False)
    V_hat = V0_f_o[["V_0_R_hat", "V_0_T_hat", "V_0_N_hat"]].to_numpy(float, copy=False)

    VBangles = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", B_hat, V_hat), -1.0, 1.0))
    )

    # -----------------------------------------------------------
    # 1) BV basis:
    #     e_z = B_hat
    #     e_y = (B × V) / |B × V|
    #     e_x = e_y × e_z
    # -----------------------------------------------------------
    B_vec = B0_f_o[["R", "T", "N"]].to_numpy(float, copy=False)
    V_vec = V0_f_o[["R", "T", "N"]].to_numpy(float, copy=False)

    By_vec = np.cross(B_vec, V_vec)
    By_hat, By_n = _safe_unit(By_vec)

    bad = np.isnan(By_hat).any(axis=1)
    if np.any(bad):
        absB = np.abs(B_hat)
        idx = np.argmin(absB, axis=1)
        ref = np.zeros_like(B_hat)
        ref[np.arange(len(B_hat)), idx] = 1.0
        By_fallback = np.cross(B_hat, ref)
        By_fallback, _ = _safe_unit(By_fallback)
        By_hat[bad] = By_fallback[bad]

    Bx_vec = np.cross(By_hat, B_hat)
    Bx_hat, _ = _safe_unit(Bx_vec)

    # -----------------------------------------------------------
    # 2) Project complex wavelet coeffs into BV basis
    # -----------------------------------------------------------
    rtn = df_w[["R", "T", "N"]].to_numpy(copy=False)
    Wx = np.einsum("ij,ij->i", Bx_hat, rtn).astype(np.complex128)
    Wy = np.einsum("ij,ij->i", By_hat, rtn).astype(np.complex128)
    W0 = np.einsum("ij,ij->i", B_hat, rtn).astype(np.complex128)

    df_w["Wx"] = Wx
    df_w["Wy"] = Wy
    df_w["W0"] = W0

    phi = None
    e_Bxk_hat = None
    k_perp_hat = None
    lam1 = None
    lam2 = None

    # -----------------------------------------------------------
    # 3) OPTIONAL φ-rotation in perp plane that diagonalizes P
    #     Pxx = |Wx|^2, Pyy = |Wy|^2, Pxy = Re(Wx Wy*)
    #     φ = 1/2 atan2(2Pxy, Pyy-Pxx)
    # -----------------------------------------------------------
    if rotate_to_k:
        Pxx = (Wx.real * Wx.real + Wx.imag * Wx.imag)
        Pyy = (Wy.real * Wy.real + Wy.imag * Wy.imag)
        Pxy = (Wx.real * Wy.real + Wx.imag * Wy.imag)

        phi = 0.5 * np.arctan2(2.0 * Pxy, (Pyy - Pxx))

        sinφ = np.sin(phi)
        cosφ = np.cos(phi)

        W_perp2 = Wy * cosφ + Wx * sinφ
        W_perp1 = -Wy * sinφ + Wx * cosφ

        df_w["Wy"] = W_perp2
        df_w["Wx"] = W_perp1

        # Eigenvalues of the 2x2 power tensor (useful to diagnose degeneracy)
        tr = Pxx + Pyy
        disc = np.sqrt((Pyy - Pxx) * (Pyy - Pxx) + 4.0 * Pxy * Pxy)
        lam1 = 0.5 * (tr + disc)
        lam2 = 0.5 * (tr - disc)

        # Physical-space unit vector corresponding to rotated Wy axis
        e_Bxk = By_hat * cosφ[:, None] + Bx_hat * sinφ[:, None]
        e_Bxk_hat, _ = _safe_unit(e_Bxk)

        # Infer k_perp_hat from e_Bxk_hat:  (B × k)_hat ~ e_Bxk_hat  => k_perp_hat ∝ B_hat × e_Bxk_hat
        kph = np.cross(B_hat, e_Bxk_hat)
        k_perp_hat, _ = _safe_unit(kph)

    # -----------------------------------------------------------
    # 4) Diagnostics (same as before)
    # -----------------------------------------------------------
    PL, PR = calculate_polarization_spectra(df_w["Wx"].values, df_w["Wy"].values)

    helicity = PR - PL
    trace_PSD = PR + PL
    cross_term = 2.0 * np.imag(np.conj(df_w["Wy"]) * df_w["W0"])
    power_sum = (np.abs(df_w["Wx"]) ** 2 + np.abs(df_w["Wy"]) ** 2 + np.abs(df_w["W0"]) ** 2)

    df_w.drop(columns=["R", "T", "N"], inplace=True, errors="ignore")

    if not return_geom:
        return df_w, VBangles, helicity, trace_PSD, cross_term, power_sum

    geom = {
        "B_hat": B_hat,
        "V_hat": V_hat,
        "Bx_hat": Bx_hat,
        "By_hat": By_hat,
        "phi": phi,
        "e_Bxk_hat": e_Bxk_hat,
        "k_perp_hat": k_perp_hat,
        "lam1": lam1,
        "lam2": lam2,
    }
    return df_w, VBangles, helicity, trace_PSD, cross_term, power_sum, geom









# def coherence_analysis(B0_f_o,
#                        V0_f_o,
#                        df_w,
#                        method,
#                        func_params=None):


#     """
#     Rotate magnetic-field wavelet coefficients from RTN into the field-aligned
#     (BV) basis **and** – optionally – into the (B×k) basis that maximises the
#     perpendicular power, exactly as described in the manuscript.

#     Parameters
#     ----------
#     B0_f_o, V0_f_o : pandas.DataFrame
#         Background magnetic-field and bulk-flow 3-vectors in columns ``'R'``,
#         ``'T'``, ``'N'``.
#     df_w : pandas.DataFrame
#         Wavelet (or FFT) coefficients of the magnetic-field perturbations in
#         the same RTN columns.
#     method : str
#         Unused placeholder (kept for API compatibility with legacy code).
#     func_params : dict, optional
#         If ``{'rotate_to_k': True}`` a second, in-plane rotation is performed
#         to align $\hat e_{\perp2}$ with the principal direction of the local
#         power tensor.

#     Returns
#     -------
#     df_w : pandas.DataFrame
#         *df_w* augmented with rotated coefficients:

#         * ``Wx_BV``, ``Wy_BV``, ``W0`` – BV basis (always),
#         * ``Wx_Bk``, ``Wy_Bk`` & ``k_*_hat`` – (B×k) basis (only when requested).

#     VBangles : ndarray
#         Angle (degrees) between $\boldsymbol B_0$ and $\boldsymbol V_0$.
#     helicity, trace_PSD, cross_term, power_sum : ndarray
#         Polarisation diagnostics $S_3$, $S_0$, $S_{yz}$ and total power.

#     Notes
#     -----
#     • All heavy operations are vectorised NumPy – no Python loops.  
#     • The “maximise |W⊥2|” prescription is solved analytically (not sampled).  
#     • Original RTN columns are dropped to save memory.
#     """

#     if func_params is None:
#         func_params = {}
#     rotate_to_k = bool(func_params.get('rotate_to_k', False))

#     # -----------------------------------------------------------
#     # 0.  unit B0 & V0
#     # -----------------------------------------------------------
#     B0_f_o = unit_vectors(B0_f_o, prefix='B_0_', vector_cols=['R', 'T', 'N'])
#     V0_f_o = unit_vectors(V0_f_o, prefix='V_0_', vector_cols=['R', 'T', 'N'])

#     Bz_vec = B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].values        # ≡ e‖

#     # angle between B0 & V0 (diagnostic only)
#     VBangles = np.degrees(np.arccos(np.einsum(
#         'ij,ij->i',
#         Bz_vec,
#         V0_f_o[['V_0_R_hat', 'V_0_T_hat', 'V_0_N_hat']].values
#     )))

#     # -----------------------------------------------------------
#     # 1.  FIRST basis   (e_x,e_y,e_z) = ((B×V)×B , B×V , B)
#     # -----------------------------------------------------------
#     By_vec = np.cross(B0_f_o[['R', 'T', 'N']].values,
#                       V0_f_o[['R', 'T', 'N']].values)
#     By_vec /= np.linalg.norm(By_vec, axis=1, keepdims=True)

#     Bx_vec = np.cross(By_vec, Bz_vec)
#     Bx_vec /= np.linalg.norm(Bx_vec, axis=1, keepdims=True)

#     rtn = df_w[['R', 'T', 'N']].values
#     Wx  = np.einsum('ij,ij->i', Bx_vec, rtn).astype(np.complex128)  # complex coeffs
#     Wy  = np.einsum('ij,ij->i', By_vec, rtn).astype(np.complex128)

#     df_w['Wx'] = Wx
#     df_w['Wy'] = Wy
#     df_w['W0'] = np.einsum('ij,ij->i', Bz_vec, rtn).astype(np.complex128)

#     # -----------------------------------------------------------
#     # 2.  OPTIONAL φ-rotation into B×k system
#     # -----------------------------------------------------------
#     if rotate_to_k:

#         # (i) principal-direction angle  φ  (vectorised, one per row)
#         Pxx = (Wx.real**2 + Wx.imag**2)
#         Pyy = (Wy.real**2 + Wy.imag**2)
#         Pxy = (Wx.real * Wy.real + Wx.imag * Wy.imag)          # ≡ Re(Wx Wy*)
#         phi = 0.5 * np.arctan2(2.0*Pxy, Pyy - Pxx)             # radians
#         sinφ = np.sin(phi)
#         cosφ = np.cos(phi)

#         # (ii) rotated wavelet amplitudes  (still complex)
#         W_perp2 = Wy * cosφ + Wx * sinφ
#         W_perp1 = -Wy * sinφ + Wx * cosφ

#         df_w['Wy'] = W_perp2
#         df_w['Wx'] = W_perp1


#     # -----------------------------------------------------------
#     # 3.  diagnostics  (exactly as before)
#     # -----------------------------------------------------------
#     PL, PR = calculate_polarization_spectra(df_w['Wx'].values,
#                                             df_w['Wy'].values)

#     helicity   = PR - PL
#     trace_PSD  = PR + PL
#     cross_term = 2.0 * np.imag(np.conj(df_w['Wy']) * df_w['W0'])
#     power_sum  = (np.abs(df_w['Wx'])**2 +
#                   np.abs(df_w['Wy'])**2 +
#                   np.abs(df_w['W0']   )**2)

#     df_w.drop(columns=['R', 'T', 'N'], inplace=True, errors='ignore')

#     return df_w, VBangles, helicity, trace_PSD, cross_term, power_sum
#              #df_w, VBangles, S3, S0, Syz, S0_full

    

def calculate_polarization_spectra(Bx, By):
    # Left-handed polarization
    PL = np.abs(Bx - 1j * By)**2

    # Right-handed polarization
    PR = np.abs(Bx + 1j * By)**2

    return PL, PR



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
    # Ensure DataFrame structure: try to call .to_frame() if available; otherwise, use the constructor.
    if not isinstance(df_ws_needed, pd.DataFrame):
        if hasattr(df_ws_needed, 'to_frame'):
            df_ws_needed = df_ws_needed.to_frame()
        else:
            df_ws_needed = pd.DataFrame(df_ws_needed)
    
    psd_sum = psd_norm * (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean().sum(axis=1)
    PSD     = (psd_sum * (counts / window_size)).resample(step).mean()
    
    return PSD


def calculate_wavelet_flatness(df_ws, rolling_params, step):
    """
    Compute the flatness of the vector wavelet coefficients,
    for a DataFrame `df_ws` with N columns (e.g. 1,2,3,... components).

    1) Form the amplitude series A(t) = sqrt(sum_i |W_i(t)|^2)
    2) Compute rolling 2nd and 4th moments of A
    3) Flatness F = <A^4> / <A^2>^2
    4) Resample F at the given step (e.g. '10s')

    Returns
    -------
    pd.Series
        Flatness time series of A, indexed at the resampled times.
    """
    # ensure DataFrame
    if not isinstance(df_ws, pd.DataFrame):
        df_ws = pd.DataFrame(df_ws)

    # 1) compute power per component, then sum to get A^2
    power = np.abs(df_ws)**2            # shape (time, N)
    A2    = power.sum(axis=1)           # series of A^2
    A     = np.sqrt(A2)                 # series of A

    # 2) rolling moments of A
    m2 = A.pow(2).rolling(**rolling_params).mean()   # ⟨A^2⟩
    m4 = A.pow(4).rolling(**rolling_params).mean()   # ⟨A^4⟩

    # 3) flatness and 4) resample
    flatness = (m4 / m2.pow(2)).resample(step).mean()
    return flatness






def compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm):
    """
    Compute Power Spectral Density (PSD) from wavelet data and return associated counts.

    Parameters:
    -----------
    df_ws_needed : pd.DataFrame or np.ndarray
        Input wavelet spectrogram data (complex).
    rolling_params : dict
        Rolling window parameters (e.g. {'window': '60s', 'center': True, 'min_periods': 10}).
    dt : float
        Time step interval for normalization, if needed.
    index_mask : pd.Series or np.ndarray
        Boolean mask indicating relevant indices. (Currently unused in calculation.)
    step : str
        Resampling step size (e.g. '10s').
    psd_norm : float
        Normalizing factor for the PSD.

    Returns:
    --------
    PSD : pd.Series
        Computed Power Spectral Density, after rolling and resampling.
    counts : pd.Series or float
        Number of events in the rolling window (also resampled), or NaN if no mask provided.
    """
    if not isinstance(df_ws_needed, pd.DataFrame):
        df_ws_needed = pd.DataFrame(df_ws_needed)

    # Mean of power in each column, sum across columns, multiply by normalization
    power   = (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean()
    psd_sum = psd_norm * power.sum(axis=1)

    # If a mask is given, compute the rolling sum of True entries
    if index_mask is not None:
        counts = index_mask.rolling(**rolling_params).sum().resample(step).mean()
    else:
        counts = np.nan

    # Final PSD is resampled
    PSD       = psd_sum.resample(step).mean()
    return PSD, counts






def calculate_wavelet_flatness_from_power(power, rolling_params, step):
    """
    Calculate Wavelet flatness (SDK) from a precomputed power DataFrame
    (i.e., real(power) = |df_ws|^2), then apply the same rolling & resample.

    Flatness = ( < sum(power)^2 > ) / ( < sum(power) >^2 ), where
    each < ... > is a rolling mean in time.

    Parameters:
    -----------
    power : pd.DataFrame (real)
        Precomputed power: for each column col, power[col] = |df_ws[col]|^2.
    rolling_params : dict
        e.g., {'window': '60s', 'center': True, 'min_periods': 10}
    step : str
        Resampling step, e.g., '10s'

    Returns:
    --------
    pd.Series
        The wavelet flatness over time.
    """
    # sum of powers across columns for each row/time
    sum_power = power.sum(axis=1)

    # numerator = rolling mean of (sum_power^2)
    numerator   = sum_power.pow(2).rolling(**rolling_params).mean()
    # denominator = rolling mean of sum_power, then squared
    denominator = sum_power.rolling(**rolling_params).mean().pow(2)

    flatness = (numerator / denominator).resample(step).mean()
    return flatness


def compute_all_metrics(df_num, 
                        df_den, 
                        rolling_params, 
                        step, 
                        psd_norm, 
                        dt, 
                        compute_sdk=True, 
                        mask=None):
    """
    Compute PSD, wavelet flatness (SDK), and compressibility (if df_den is given)
    in one pass, with minimal repeated calculations of (df * conj(df)).

    Parameters:
    -----------
    df_num : pd.DataFrame (complex)
        The numerator data for which we want PSD and SDK. E.g., W0, or Mod.
    df_den : pd.DataFrame (complex) or None
        If provided, we also compute compressibility as ratio of
        sum(|df_num|^2) / sum(|df_den|^2), with the same rolling.
    rolling_params : dict
        e.g., {'window': '60s', 'center': True, 'min_periods': 10}
    step : str
        e.g., '10s' (resample final step).
    psd_norm : float
        Normalization factor for PSD.
    dt : float
        Time step (kept for signature consistency, not used inside).
    compute_sdk : bool
        If True, compute wavelet flatness (SDK) for df_num.
    mask : pd.Series(bool) or None
        Boolean mask for e.g. parallel/perpendicular subsets.

    Returns:
    --------
    dict with:
      'PSD'      -> PSD of df_num
      'SDK'      -> wavelet flatness of df_num (or None if compute_sdk=False)
      'compress' -> ratio-based compressibility (or None if df_den=None)
    """
    # 1) Ensure DataFrame structure
    if not isinstance(df_num, pd.DataFrame):
        df_num = pd.DataFrame(df_num)
    if df_den is not None and not isinstance(df_den, pd.DataFrame):
        df_den = pd.DataFrame(df_den)

    # 2) Apply mask if needed
    if mask is not None:
        df_num = df_num.where(mask)
        if df_den is not None:
            df_den = df_den.where(mask)

    # 3) Precompute power once for df_num
    #    We'll do the real part in case df is complex
    power_num = (df_num * np.conj(df_num)).apply(np.real)

    # PSD for df_num = rolling mean of sum of powers, multiplied by psd_norm, then resampled
    rolling_power_num = power_num.rolling(**rolling_params).mean()
    PSD_num           = psd_norm * rolling_power_num.sum(axis=1)
    PSD_num           = PSD_num.resample(step).mean()

    # Wavelet flatness (SDK), if requested
    if compute_sdk:
        SDK_num = calculate_wavelet_flatness_from_power(power_num, rolling_params, step)
    else:
        SDK_num = None

    # 4) If df_den is given, compute compressibility
    compress = None
    if df_den is not None:
        power_den = (df_den * np.conj(df_den)).apply(np.real)
        # ratio(t) = sum(power_num(t)) / sum(power_den(t)) -> then rolling -> then resample
        ratio = power_num.sum(axis=1) / power_den.sum(axis=1)
        compress = ratio.rolling(**rolling_params).mean().resample(step).mean()

    return {
        'PSD'      : PSD_num,
        'SDK'      : SDK_num,
        'compress' : compress
    }


def est_compress(df_mod,
                 df_w,
                 index_par,
                 index_per,
                 dt,
                 psd_norm,
                 func_params=None):
    """
    Estimate compressibility and PSD/SDK measures over parallel/perpendicular modes.

    Parameters:
    -----------
    df_mod : pd.Series or np.ndarray
        Modulus of the fields (magnitude) in wavelet space.
    df_w : pd.DataFrame
        Wavelet components. Expected columns: ['W0','Wx','Wy'] (or more).
    index_par : array-like
        Boolean mask indicating "parallel" wave vectors/times.
    index_per : array-like
        Boolean mask indicating "perpendicular" wave vectors/times.
    dt : float
        Time step interval.
    psd_norm : float
        Multiplicative normalization factor for PSD.
    func_params : dict, optional
        Additional parameters:
        - estimate_comp (bool): whether to estimate compressibility or not
        - use_rolling_mean (bool): whether to use rolling-window PSD/SDK or simple nanmean
        - averaging_window (str): rolling window size, e.g. '60s'
        - step (str): resampling step, e.g. '10s'
        - any other needed arguments

    Returns:
    --------
    dict
        A dictionary containing PSDs, SDKs, and compressibility measures for
        total/parallel/perpendicular data.
    """
    if func_params is None:
        func_params = {}
    estimate_comp     = func_params.get('estimate_comp', False)
    use_rolling_mean  = func_params.get('use_rolling_mean', False)

    # Initialize placeholders for output
    PSD_par      = PSD_per = PSD_mod = PSD_mod_per = PSD_mod_par = np.nan
    PSD_W0       = PSD_W0_per = PSD_W0_par = np.nan
    PSD_Wxy      = PSD_Wxy_per = PSD_Wxy_par = np.nan
    compress_mod = compress_mod_par = compress_mod_per = np.nan
    compress     = compress_par     = compress_per     = np.nan
    SDK_par      = SDK_per = SDK_W0 = SDK_Wxy = SDK_mod_par = SDK_mod_per = SDK_mod = np.nan
    
    if estimate_comp:
        try:
            if use_rolling_mean:
                # Rolling / resampling params
                averaging_window = func_params.get('averaging_window', '60s')
                step             = func_params.get('step', '10s')
                rolling_params   = {'window': averaging_window, 'center': True, 'min_periods': 10}

                # Convert df_mod to DataFrame
                df_mod_df = pd.DataFrame({'Mod': df_mod}, index=df_w.index)

                ########## All data (no mask) ##########
                # (1) W0 vs total
                out_w0 = compute_all_metrics(
                    df_num         = df_w[['W0']],   # e.g. compress numerator
                    df_den         = df_w,           # e.g. compress denominator
                    rolling_params = rolling_params,
                    step           = step,
                    psd_norm       = psd_norm,
                    dt             = dt,
                    compute_sdk    = True,
                    mask           = None
                )
                
                PSD_W0    = out_w0['PSD']
                SDK_W0    = out_w0['SDK']
                compress  = out_w0['compress']  # ratio: W0^2 / total

                # (2) Wxy (just PSD & SDK, no compress)
                out_wxy = compute_all_metrics(
                    df_num         = df_w[['Wx','Wy']],
                    df_den         = None,
                    rolling_params = rolling_params,
                    step           = step,
                    psd_norm       = psd_norm,
                    dt             = dt,
                    compute_sdk    = True,
                    mask           = None
                )
                PSD_Wxy = out_wxy['PSD']
                SDK_Wxy = out_wxy['SDK']

                # (3) Mod vs total
                out_mod = compute_all_metrics(
                    df_num           = df_mod_df,
                    df_den           = df_w,
                    rolling_params   = rolling_params,
                    step             = step,
                    psd_norm         = psd_norm,
                    dt               = dt,
                    compute_sdk      = True,
                    mask             = None
                )
                
                PSD_mod              = out_mod['PSD']
                SDK_mod              = out_mod['SDK']
                compress_mod         = out_mod['compress']

                # (4) The entire trace itself (PSD + optional SDK)
                # For consistency with your old code, we only keep PSD, not the wavelet flatness
                out_total = compute_all_metrics(
                    df_num            = df_w,
                    df_den            = None,
                    rolling_params    = rolling_params,
                    step              = step,
                    psd_norm          = psd_norm,
                    dt                = dt,
                    compute_sdk       = False,
                    mask              = None
                )
                PSD_total  = out_total['PSD']  # we don't store it in the final dictionary

                ########## Parallel subset ##########
                mask_par = pd.Series(index_par, index=df_w.index)

                # W0 vs total (parallel)
                out_w0_par = compute_all_metrics(
                    df_num            = df_w[['W0']],
                    df_den            = df_w,
                    rolling_params    = rolling_params,
                    step              = step,
                    psd_norm          = psd_norm,
                    dt                = dt,
                    compute_sdk       = True,
                    mask              = mask_par
                )
                PSD_W0_par        = out_w0_par['PSD']
                SDK_Bpar_par      = out_w0_par['SDK']
                compress_par      = out_w0_par['compress']

                # Wxy (parallel)
                out_wxy_par        = compute_all_metrics(
                    df_num         = df_w[['Wx','Wy']],
                    df_den         = None,
                    rolling_params = rolling_params,
                    step           = step,
                    psd_norm       = psd_norm,
                    dt             = dt,
                    compute_sdk    = True,
                    mask           = mask_par
                )
                PSD_Wxy_par       = out_wxy_par['PSD']
                SDK_Bper_par      = out_wxy_par['SDK']

                # entire wavelet (parallel)
                out_par           = compute_all_metrics(
                    df_num        = df_w,
                    df_den        = None,
                    rolling_params= rolling_params,
                    step          = step,
                    psd_norm      = psd_norm,
                    dt            = dt,
                    compute_sdk   = True,
                    mask          = mask_par
                )
                PSD_par   = out_par['PSD']
                SDK_par   = out_par['SDK']

                # mod (parallel)
                out_mod_par = compute_all_metrics(
                    df_num=df_mod_df,
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_par
                )
                PSD_mod_par      = out_mod_par['PSD']
                SDK_mod_par      = out_mod_par['SDK']
                compress_mod_par = out_mod_par['compress']

                ########## Perp subset ##########
                mask_per = pd.Series(index_per, index=df_w.index)

                # W0 vs total (perp)
                out_w0_per = compute_all_metrics(
                    df_num=df_w[['W0']],
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per
                )
                PSD_W0_per        = out_w0_per['PSD']
                SDK_Bpar_per      = out_w0_per['SDK']
                compress_per      = out_w0_per['compress']

                # Wxy (perp)
                out_wxy_per = compute_all_metrics(
                    df_num=df_w[['Wx','Wy']],
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per
                )
                PSD_Wxy_per       = out_wxy_per['PSD']
                SDK_Bper_per      = out_wxy_per['SDK']

                # entire wavelet (perp)
                out_per = compute_all_metrics(
                    df_num=df_w,
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per
                )
                PSD_per   = out_per['PSD']
                SDK_per   = out_per['SDK']

                # mod (perp)
                out_mod_per = compute_all_metrics(
                    df_num=df_mod_df,
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per
                )
                PSD_mod_per      = out_mod_per['PSD']
                SDK_mod_per      = out_mod_per['SDK']
                compress_mod_per = out_mod_per['compress']

            else:
                # --- Compute compressibility WITHOUT rolling means (simple nanmeans) ---
                # parallel
                #w0_par_power     = np.real(df_w['W0'][index_par] * np.conj(df_w['W0'][index_par]))
                tot_par_power    = np.real(df_w.iloc[index_par] * np.conj(df_w.iloc[index_par]))
                # compress_par     = (np.nanmean(w0_par_power) /
                #                     np.nanmean(tot_par_power.sum(axis=1)))
                
                mod_par_power    = np.real(df_mod[index_par] * np.conj(df_mod[index_par]))
                compress_mod_par = (np.nanmean(mod_par_power) /
                                    np.nanmean(tot_par_power.sum(axis=1)))
                
                # perp
                # w0_per_power     = np.real(df_w['W0'][index_per] * np.conj(df_w['W0'][index_per]))
                tot_per_power    = np.real(df_w.iloc[index_per] * np.conj(df_w.iloc[index_per]))
                # compress_per     = (np.nanmean(w0_per_power) /
                #                     np.nanmean(tot_per_power.sum(axis=1)))
                
                mod_per_power    = np.real(df_mod[index_per] * np.conj(df_mod[index_per]))
                compress_mod_per = (np.nanmean(mod_per_power) /
                                    np.nanmean(tot_per_power.sum(axis=1)))

                # all
                # w0_all_power   = np.real(df_w['W0'] * np.conj(df_w['W0']))
                tot_all_power  = np.real(df_w * np.conj(df_w))
                # compress       = (np.nanmean(w0_all_power) /
                #                   np.nanmean(tot_all_power.sum(axis=1)))
                
                mod_all_power  = np.real(df_mod * np.conj(df_mod))
                compress_mod   = (np.nanmean(mod_all_power) /
                                  np.nanmean(tot_all_power.sum(axis=1)))

        except Exception:
            traceback.print_exc()
            compress_par = compress_per = compress = compress_mod = np.nan

    # Return dictionary of all relevant results
    return {
        'PSD_par'            : PSD_par,
        'PSD_per'            : PSD_per,

        'PSD_mod'            : PSD_mod,
        'PSD_mod_par'        : PSD_mod_par,
        'PSD_mod_per'        : PSD_mod_per,

        'PSD_Bpar'           : PSD_W0,     # "Bpar" used for the W0 component
        'PSD_Bpar_par'       : PSD_W0_par,
        'PSD_Bpar_per'       : PSD_W0_per,

        'PSD_Bper'           : PSD_Wxy,    # "Bper" used for the Wx,Wy components
        'PSD_Bper_par'       : PSD_Wxy_par,
        'PSD_Bper_per'       : PSD_Wxy_per,

        'compress_mod'       : compress_mod,
        'compress_mod_par'   : compress_mod_par,
        'compress_mod_per'   : compress_mod_per,

        # 'compress'           : compress,
        # 'compress_par'       : compress_par,
        # 'compress_per'       : compress_per,

        'SDK_par'            : SDK_par,
        'SDK_per'            : SDK_per,

        'SDK_Bpar'           : SDK_W0,
        'SDK_Bper'           : SDK_Wxy,

        'SDK_mod'            : SDK_mod,
        'SDK_mod_par'        : SDK_mod_par,
        'SDK_mod_per'        : SDK_mod_per
    }






def do_coh_analysis(k_df, 
                    df_w,
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
            

        # Compute ks
        k_df =  k_df.rolling(**rolling_params).mean().resample(step).mean()
        
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
        Trace_PSD    = np.nansum([PSD_coh, PSD_non_coh], axis=0)
        SDK          = calculate_wavelet_flatness(df_w, rolling_params, step)

        
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

                    'k'                  : k_df['k'].values,
                    #'k_par'              : k_df['k_par'].values,
            
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
                           k_df, 
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
    coh_res = do_coh_analysis(k_df, df_w, S0, S3, S0_full, Syz, index_par, index_per, dt, scale, psd_norm, func_params=func_params)
    
    # Estimate Anisotropic PSD
    #anis_res = est_anisotropic_PSDs(df_w, df_mod, index_par, index_per, dt, psd_norm, func_params=func_params)
    
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
    return {**coh_res,  **comp_res, **sf_res}


def define_W_df(B_index, R, T, N):
    return      pd.DataFrame({ 'DateTime' : B_index,
                                'R'       : R,
                                'T'       : T,
                                'N'       : N}).set_index('DateTime')



# def anisotropy_coherence(  
#                            B_df,
#                            V_df, 
#                            mag_V_mod_TH,
                           
#                            field_flag,
#                            E_df                  =  None,
#                            Np_df                 =  None,
#                            method                =  'min_var',
#                            func_params           =  None,
#                            f_dict                =  None,
#                            frame_flag            =  None
#                           ):
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
    

#     def define_B_df(B_df):
    
#         B_df['RR'] = B_df.R* B_df.R
#         B_df['TT'] = B_df['T']* B_df['T']
#         B_df['NN'] = B_df.N* B_df.N

#         B_df['RT'] = B_df.R* B_df['T']
#         B_df['RN'] = B_df.R* B_df.N
#         B_df['TN'] = B_df['T']* B_df.N
#         return B_df
              


#     def parallel_oper(mag_V_mod_TH,
#                       ii, 
#                       scale,
#                       dt,
#                       B_df,
#                       V_df, 
#                       df_w,
#                       df_mod,
#                       frequencies,
#                       psd_norm,
#                       method      = 'min_var',
#                       func_params = None,
#                       frame_flag  =  None ):
#         try:

#             if func_params['do_coherence_analysis']:

#                 if method =='min_var':
#                     B_df = define_B_df(B_df)

#                 # Estimate local B
#                 B_df = B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0)

#                 if func_params['estimate_local_V']:
#                     # Estimate local V
#                     V_df = V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0)

#                 # Do coherence analysis
#                 df_w, VBangles, S3, S0, Syz, S0_full       = coherence_analysis(
#                                                                         B_df,
#                                                                         V_df,
#                                                                         df_w,
#                                                                         method,
#                                                                         func_params       = func_params)
#             else:

#                 B_df = B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0)
                

#                 if func_params['estimate_local_V']:
#                     V_df = V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale, alpha =func_params['alpha']), axis=0)
                    

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

#             est_quants = return_desired_quants(
#                 df_w, 
#                 df_mod,
#                 S0,
#                 S3,
#                 S0_full, 
#                 Syz,
#                 VBangles,
#                 pd.DataFrame({
#                     'k' : 2 * np.pi * frequencies / np.hstack(mag_V_mod_TH.values)
#                     #'k_perp': 2 * np.pi * frequencies / np.linalg.norm(V_df.values, axis=1)* np.sin(np.deg2rad(VBangles))
#                 }, index=V_df.index),
#                 dt          = dt, 
#                 scale       = scale, 
#                 psd_norm    = psd_norm, 
#                 func_params = func_params
#             )

#             if func_params['return_coeffs'] is False:
#                 return est_quants, None, None, None, None, None
                

#             return est_quants, VBangles, df_w.values.T, S3, S0, Syz
#         except Exception as e:
#             traceback.print_exc()
#             return np.nan, np.nan
                      
                      
#     print('Using', func_params['njobs'], 'cores')      
        
#     # Rename the columns
#     if B_df.columns[0] =='Bx':
#         B_df  = B_df.rename(columns={'Bx': 'R', 'By': 'T', 'Bz': 'N'}) 
#         V_df  = V_df.rename(columns={'Vx': 'R', 'Vy': 'T', 'Vz': 'N'})
        
        
        
#         if (method =='TN_only') & (E_df is not None):

#             E_df['R'] = 0*E_df['Ey']
#             E_df['T'] =   E_df['Ex']
#             E_df['N'] = - E_df['Ey']

            
#             #Drop original 'R', 'T', 'N'
#             E_df.drop(columns=['Ex', 'Ey'], inplace=True, errors='ignore')
#         else:
#             E_df  = None if E_df is None else E_df.rename(columns={'Ex': 'R', 'Ey': 'T', 'Ez': 'N'})
#     else:
#         B_df  = B_df.rename(columns={'Br': 'R', 'Bt': 'T', 'Bn': 'N'})
#         V_df  = V_df.rename(columns={'Vr': 'R', 'Vt': 'T', 'Vn': 'N'})
#         E_df  = None if E_df is None else E_df.rename(columns={'Er': 'R', 'Et': 'T', 'En': 'N'})


#     # Estimate sampling times of time series
#     dt_B, dt_V = func.find_cadence(B_df), func.find_cadence(V_df)
#     dt_E       = func.find_cadence(E_df) if E_df is not None else None
    
    

#     # Synchronize E_df and B_df if necessary
#     if E_df is not None and dt_E != dt_B:
#         try:
#             E_df, B_df = func.synchronize_dfs(E_df, B_df,True)#(B_df, E_df.index)
#         except:
#            E_df        =func.newindex(E_df, B_df.index)
       
        
#     else:
#         print('Got here', len(B_df))



#     # Synchronize B_df and V_df if necessary
#     if dt_V != dt_B:
#         B_df, V_df = func.synchronize_dfs(B_df, V_df,True)#(B_df, E_df.index)

#     print('passed')

#     # Determine the common dt
#     dt = dt_E if dt_E is not None else dt_B
    
#     # Estimate wavelet coefficients
#     Wvec, scales, freqs, psd_norm, coi= estimate_cwt(E_df if E_df is not None else  B_df, 
#                                                   dt,
#                                                   nv       = func_params['nv'],
#                                                   min_freq = func_params['min_freq'],
#                                                   max_freq = func_params['max_freq'])



#     # Estimate magnitude of magnetic field
#     if func_params['est_mod']:

#         if Np_df is None:
#             Wmod, _, _, _ ,_= estimate_cwt( pd.DataFrame({'Mod': np.linalg.norm(B_df.iloc[:, :3], axis=1)},index=B_df.index),
#                                             dt, 
#                                             nv       = func_params['nv'],
#                                             min_freq = func_params['min_freq'], max_freq = func_params['max_freq'],)     
#         else:
            
#             Wmod, _, _, _ ,_= estimate_cwt( pd.DataFrame({'Mod': Np_df.values.T[0]},index= Np_df.index),
#                                             dt, 
#                                             nv       = func_params['nv'],
#                                             min_freq = func_params['min_freq'], max_freq = func_params['max_freq'])  
#     else:
#         Wmod             = None
        
    
#     # Initialize arrays
#     PSD_par = np.zeros(len(freqs))
#     PSD_per = np.zeros(len(freqs)) 
 
#     PSD_par_mod = np.zeros(len(freqs))
#     PSD_per_mod = np.zeros(len(freqs))

#     # Use joblib for parallel processing
#     results = Parallel(n_jobs=func_params['njobs'])(
#         delayed(parallel_oper)(
#                                 mag_V_mod_TH,
#                                 ii, 
#                                 scale,
#                                 dt,
#                                 B_df.copy(),
#                                 V_df.copy(), 
#                                 define_W_df(B_df.index, Wvec['R'][ii], Wvec['T'][ii], Wvec['N'][ii]),
#                                 Wmod['Mod'][ii],
#                                 freqs[ii],
#                                 psd_norm,
#                                 method                = method,
#                                 func_params           = func_params,
#                                 frame_flag            =  frame_flag 
#         ) for ii, scale in tqdm(enumerate(scales), total=len(scales))
#     )


    
#     # Unpack results
#     est_quants, VBangles, df_w, S3, S0, Syz = zip(*results)
    
#     # Initialize the dictionary and populate it in one step
    
#     #field_flag            = 'E' if E_df is not None else  'B'
#     if f_dict is None:
        
#         if func_params['use_rolling_mean']:
#             f_dict                = {field_flag : {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}}

#             f_dict['freqs']       = freqs
#             f_dict['scales']      = scales
#             f_dict['Wave_coeffs'] = df_w
#             f_dict['S3_ts']       = S3
#             f_dict['S0_ts']       = S0
#             f_dict['VB_ts']       = VBangles
#             f_dict['flag']        = method
#         else:
#             f_dict                = {field_flag : {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}}           
#     else:
#         if func_params['use_rolling_mean']:
#             f_dict[field_flag ]   = {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}
#         else:

#             f_dict[field_flag ]   =  {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}       


    
#     return f_dict  


def anisotropy_coherence(
    B_df,
    V_df,
    mag_V_mod_TH,
    field_flag,
    E_df=None,
    Np_df=None,
    method="min_var",
    func_params=None,
    f_dict=None,
    frame_flag=None,
    mag_Va_ts=None,
    sigma_Va_ts=None,
):
    import numpy as np
    import pandas as pd
    import traceback
    from tqdm import tqdm
    from joblib import Parallel, delayed

    if func_params is None:
        func_params = {}

    def define_B_df(B_df):
        B_df["RR"] = B_df.R * B_df.R
        B_df["TT"] = B_df["T"] * B_df["T"]
        B_df["NN"] = B_df.N * B_df.N
        B_df["RT"] = B_df.R * B_df["T"]
        B_df["RN"] = B_df.R * B_df.N
        B_df["TN"] = B_df["T"] * B_df.N
        return B_df

    def parallel_oper(
        mag_V_mod_TH,
        mag_Va_ts,
        sigma_Va_ts,
        ii,
        scale,
        dt,
        B_df,
        V_df,
        df_w,
        df_mod,
        frequencies,
        psd_norm,
        method="min_var",
        func_params=None,
        frame_flag=None,
    ):
        try:
            import numpy as np
            import pandas as pd

            if func_params is None:
                func_params = {}

            kper_mode = func_params.get("kper_mode", "simple")
            kpar_mode = func_params.get("kpar_mode", "mth")
            use_cphi = bool(func_params.get("use_cphi", True))
            cphi_floor = float(func_params.get("cphi_floor", 1e-3))

            if func_params.get("do_coherence_analysis", False):

                if method == "min_var":
                    B_df = define_B_df(B_df)

                B_df = B_df.apply(
                    lambda col: local_gaussian_averaging(
                        col.values, dt, scale, alpha=func_params["alpha"]
                    ),
                    axis=0,
                )

                if func_params.get("estimate_local_V", False):
                    V_df = V_df.apply(
                        lambda col: local_gaussian_averaging(
                            col.values, dt, scale, alpha=func_params["alpha"]
                        ),
                        axis=0,
                    )

                out = coherence_analysis(
                    B_df,
                    V_df,
                    df_w,
                    method,
                    func_params=func_params,
                    return_geom=True,
                )
                df_w, VBangles, S3, S0, Syz, S0_full, geom = out

            else:
                B_df = B_df.apply(
                    lambda col: local_gaussian_averaging(
                        col.values, dt, scale, alpha=func_params["alpha"]
                    ),
                    axis=0,
                )

                if func_params.get("estimate_local_V", False):
                    V_df = V_df.apply(
                        lambda col: local_gaussian_averaging(
                            col.values, dt, scale, alpha=func_params["alpha"]
                        ),
                        axis=0,
                    )

                B_df = unit_vectors(B_df, prefix="B_0_", vector_cols=["R", "T", "N"])
                V_df = unit_vectors(V_df, prefix="V_0_", vector_cols=["R", "T", "N"])

                B_hat = B_df[["B_0_R_hat", "B_0_T_hat", "B_0_N_hat"]].values
                V_hat = V_df[["V_0_R_hat", "V_0_T_hat", "V_0_N_hat"]].values
                VBangles = np.degrees(
                    np.arccos(np.clip(np.einsum("ij,ij->i", B_hat, V_hat), -1.0, 1.0))
                )

                S3 = None
                S0 = None
                Syz = None
                S0_full = None
                geom = {"B_hat": B_hat, "k_perp_hat": None}

            VBangles[VBangles > 90] = 180 - VBangles[VBangles > 90]

            idx = V_df.index
            omega_sc = 2.0 * np.pi * float(frequencies)

            if mag_Va_ts is not None:
                Va = (
                    pd.Series(np.asarray(mag_Va_ts).squeeze(), index=mag_Va_ts.index)
                    .reindex(idx)
                    .interpolate()
                    .to_numpy()
                )
            else:
                Va = np.full(len(idx), np.nan)

            if sigma_Va_ts is not None:
                sigVa = (
                    pd.Series(np.asarray(sigma_Va_ts).squeeze(), index=sigma_Va_ts.index)
                    .reindex(idx)
                    .interpolate()
                    .to_numpy()
                )
            else:
                sigVa = np.full(len(idx), np.nan)

            B_hat = geom.get("B_hat", None)
            if B_hat is None:
                B_hat = unit_vectors(
                    B_df.copy(), prefix="B_0_", vector_cols=["R", "T", "N"]
                )[["B_0_R_hat", "B_0_T_hat", "B_0_N_hat"]].values

            V_rel = V_df[["R", "T", "N"]].to_numpy(float, copy=False)

            Vpar = np.einsum("ij,ij->i", V_rel, B_hat)
            Vpar_abs = np.abs(Vpar)

            Vperp_vec = V_rel - Vpar[:, None] * B_hat
            Vperp = np.linalg.norm(Vperp_vec, axis=1)

            c_phi = np.ones(len(idx), dtype=float)
            k_perp_hat = geom.get("k_perp_hat", None)

            if use_cphi and (k_perp_hat is not None):
                Vperp_hat = Vperp_vec / np.where(Vperp[:, None] == 0.0, np.nan, Vperp[:, None])
                c_phi = np.abs(np.einsum("ij,ij->i", Vperp_hat, k_perp_hat))

            c_phi = np.clip(c_phi, cphi_floor, 1.0)

            if str(kpar_mode).lower() == "mth":
                denom_par = np.abs(Vpar + sigVa * Va)
            else:
                denom_par = Vpar_abs

            k_par = omega_sc / np.where(denom_par == 0.0, np.nan, denom_par)
            df_dk_par = denom_par / (2.0 * np.pi)

            denom_perp = Vperp * c_phi
            k_perp_simple = omega_sc / np.where(denom_perp == 0.0, np.nan, denom_perp)

            if str(kper_mode).lower() == "mth_exact":
                numer = omega_sc - sigVa * Va * k_par
                k_perp = np.abs(numer) / np.where(denom_perp == 0.0, np.nan, denom_perp)
            else:
                k_perp = k_perp_simple

            df_dk_perp = denom_perp / (2.0 * np.pi)

            Vmod = (
                pd.Series(np.hstack(mag_V_mod_TH.values), index=mag_V_mod_TH.index)
                .reindex(idx)
                .interpolate()
                .to_numpy()
            )
            k_iso = omega_sc / np.where(Vmod == 0.0, np.nan, Vmod)

            k_df = pd.DataFrame(
                {
                    "k": k_iso,
                    "k_perp": k_perp,
                    "k_par": k_par,
                    "c_phi": c_phi,
                    "Vrel_par": Vpar_abs,
                    "Vrel_perp": Vperp,
                    "Va": Va,
                    "sigma_Va": sigVa,
                    "df_dk_perp": df_dk_perp,
                    "df_dk_par": df_dk_par,
                },
                index=idx,
            )

            est_quants = return_desired_quants(
                df_w,
                df_mod,
                S0,
                S3,
                S0_full,
                Syz,
                VBangles,
                k_df,
                dt=dt,
                scale=scale,
                psd_norm=psd_norm,
                func_params=func_params,
            )

            if func_params.get("return_coeffs", False) is False:
                return est_quants, None, None, None, None, None

            return est_quants, VBangles, df_w.values.T, S3, S0, Syz

        except Exception:
            traceback.print_exc()
            return np.nan, np.nan

    print("Using", func_params["njobs"], "cores")

    if B_df.columns[0] == "Bx":
        B_df = B_df.rename(columns={"Bx": "R", "By": "T", "Bz": "N"})
        V_df = V_df.rename(columns={"Vx": "R", "Vy": "T", "Vz": "N"})

        if (method == "TN_only") & (E_df is not None):
            E_df["R"] = 0 * E_df["Ey"]
            E_df["T"] = E_df["Ex"]
            E_df["N"] = -E_df["Ey"]
            E_df.drop(columns=["Ex", "Ey"], inplace=True, errors="ignore")
        else:
            E_df = None if E_df is None else E_df.rename(columns={"Ex": "R", "Ey": "T", "Ez": "N"})
    else:
        B_df = B_df.rename(columns={"Br": "R", "Bt": "T", "Bn": "N"})
        V_df = V_df.rename(columns={"Vr": "R", "Vt": "T", "Vn": "N"})
        E_df = None if E_df is None else E_df.rename(columns={"Er": "R", "Et": "T", "En": "N"})

    dt_B, dt_V = func.find_cadence(B_df), func.find_cadence(V_df)
    dt_E = func.find_cadence(E_df) if E_df is not None else None

    if E_df is not None and dt_E != dt_B:
        try:
            E_df, B_df = func.synchronize_dfs(E_df, B_df, True)
        except Exception:
            E_df = func.newindex(E_df, B_df.index)
    else:
        print("Got here", len(B_df))

    if dt_V != dt_B:
        B_df, V_df = func.synchronize_dfs(B_df, V_df, True)

    print("passed")

    dt = dt_E if dt_E is not None else dt_B

    Wvec, scales, freqs, psd_norm, coi = estimate_cwt(
        E_df if E_df is not None else B_df,
        dt,
        nv=func_params["nv"],
        min_freq=func_params["min_freq"],
        max_freq=func_params["max_freq"],
    )

    if func_params["est_mod"]:
        if Np_df is None:
            Wmod, _, _, _, _ = estimate_cwt(
                pd.DataFrame({"Mod": np.linalg.norm(B_df.iloc[:, :3], axis=1)}, index=B_df.index),
                dt,
                nv=func_params["nv"],
                min_freq=func_params["min_freq"],
                max_freq=func_params["max_freq"],
            )
        else:
            Wmod, _, _, _, _ = estimate_cwt(
                pd.DataFrame({"Mod": Np_df.values.T[0]}, index=Np_df.index),
                dt,
                nv=func_params["nv"],
                min_freq=func_params["min_freq"],
                max_freq=func_params["max_freq"],
            )
    else:
        Wmod = None

    results = Parallel(n_jobs=func_params["njobs"])(
        delayed(parallel_oper)(
            mag_V_mod_TH,
            mag_Va_ts,
            sigma_Va_ts,
            ii,
            scale,
            dt,
            B_df.copy(),
            V_df.copy(),
            define_W_df(B_df.index, Wvec["R"][ii], Wvec["T"][ii], Wvec["N"][ii]),
            Wmod["Mod"][ii] if Wmod is not None else None,
            freqs[ii],
            psd_norm,
            method=method,
            func_params=func_params,
            frame_flag=frame_flag,
        )
        for ii, scale in tqdm(enumerate(scales), total=len(scales))
    )

    est_quants, VBangles, df_w, S3, S0, Syz = zip(*results)

    if f_dict is None:
        if func_params["use_rolling_mean"]:
            f_dict = {field_flag: {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}}
            f_dict["freqs"] = freqs
            f_dict["scales"] = scales
            f_dict["Wave_coeffs"] = df_w
            f_dict["S3_ts"] = S3
            f_dict["S0_ts"] = S0
            f_dict["VB_ts"] = VBangles
            f_dict["flag"] = method
        else:
            f_dict = {field_flag: {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}}
    else:
        if func_params["use_rolling_mean"]:
            f_dict[field_flag] = {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}
        else:
            f_dict[field_flag] = {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}

    return f_dict





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





