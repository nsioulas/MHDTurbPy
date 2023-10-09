import os
import time
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.linalg import solve
from scipy import constants
from distutils.log import warn

from numba import jit, njit, prange, objmode
import numba
from scipy.fft import fft, fftfreq
from joblib import Parallel, delayed

import pycwt as wavelet
import ssqueezepy

# SPEDAS API
# make sure to use the local spedas
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data


from general_functions import *
from three_D_funcs import *


def shifted_df_calcs(B, lag_coefs, coefs, return_df=False):
    """
    Calculate the shifted dataframe.

    Parameters:
        B (pandas.DataFrame): The input dataframe.
        lag_coefs (list): A list of integers representing the lags.
        coefs (list): A list of coefficients for the calculation.
        return_df (bool, optional): If True, return the result as a DataFrame. Otherwise, return a 2D numpy array.

    Returns:
        pandas.DataFrame or numpy.ndarray: The result of the calculation, either as a DataFrame or a 2D numpy array.
    """
    result = np.add.reduce([x * B.shift(y) for x, y in zip(coefs, lag_coefs)])
    if return_df:
        return pd.DataFrame(result, index=B.index, columns=B.columns)
    else:
        return result

def flucts(tau,
           B,
           five_points_sfunc  = True,
           return_dataframe   = False,
           estimate_mod_flucts= False ):
    """
    Calculate fluctuations for structure functions.

    Args:
        tau (int): Time lag.
        B (pd.Series or np.ndarray): Input field.
        five_points_sfunc (bool, optional): Estimate 5-point structure functions if True. Defaults to True.

    Returns:
        dB (np.ndarray): Fluctuations of the input field.
    """

    # Estimate 5-point Structure functions
    if five_points_sfunc:
        
        # Define coefs for fluctuations
        coefs_db      = np.array([1, -4, +6, -4, 1]) / np.sqrt(35)
        lag_coefs_db  = np.array([-2 * tau, -tau, 0, tau, 2 * tau]).astype(int)
        
        # Compute the fluctuation
        if estimate_mod_flucts:
            # Create B mod df
            df_keys    = list(B.keys())
            B_mod      = pd.DataFrame({'DateTime': B.index, 
                                       'B_mod'   : np.sqrt(B[df_keys[0]]**2 + B[df_keys[1]]**2 + B[df_keys[2]]**2)}).set_index('DateTime')

            if return_dataframe:
                dB            = shifted_df_calcs(B_mod,
                                                 lag_coefs_db,
                                                 coefs_db,
                                                 return_df = True)
            else:
                dB            = shifted_df_calcs(B_mod,
                                                 lag_coefs_db,
                                                 coefs_db)                

        else:
            if return_dataframe:
                dB            = shifted_df_calcs(B,
                                                 lag_coefs_db,
                                                 coefs_db,
                                                 return_df = True)
            else:
                dB            = shifted_df_calcs(B,
                                                 lag_coefs_db,
                                                 coefs_db)                

    # Estimate regular 2-point Structure functions
    else:
        if estimate_mod_flucts:
            
            # Create B mod df
            df_keys    = list(B.keys())
            B_mod      = pd.DataFrame({'DateTime': B.index, 
                                       'B_mod'   : np.sqrt(B[df_keys[0]]**2 + B[df_keys[1]]**2 + B[df_keys[2]]**2)}).set_index('DateTime')
            
            if return_dataframe:
                dB                      = (B_mod.iloc[:-tau].values - B_mod.iloc[tau:].values)
                dB_shape                = B_mod.shape
                dB_filled               = pd.DataFrame(np.nan, index=B_mod.index, columns=B_mod.columns)
                dB_filled.iloc[:-tau,:] = dB
                dB                      = dB_filled
            else:
                dB                      = (B_mod.iloc[:-tau].values - B_mod.iloc[tau:].values)
            
        else:
            if return_dataframe:
                dB                      = (B.iloc[:-tau].values - B.iloc[tau:].values)
                dB_shape                = B.shape
                dB_filled               = pd.DataFrame(np.nan, index=B.index, columns=B.columns)
                dB_filled.iloc[:-tau,:] = dB
                dB                      = dB_filled#.iloc[tau:,:]
            else:
                dB                      = (B.iloc[:-tau].values - B.iloc[tau:].values)

    return dB


def structure_functions_parallel(B,
                                 scales,
                                 max_qorder, 
                                 five_points_sfunc=False, 
                                 keep_sdk=False,
                                 n_jobs=-1):
    """
    Estimate the structure functions of a field in parallel.

    Args:
        B (pd.Series or np.ndarray): Input field.
        scales (list or np.ndarray): Scales at which to calculate the structure functions.
        max_qorder (int): Maximum order of the structure functions to be calculated.
        five_points_sfunc (bool, optional): Estimate 5-point structure functions if True. Defaults to False.
        keep_sdk (bool, optional): Keep the SDK if True. Defaults to False.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (use all available cores).

    Returns:
        sfn (np.ndarray): Estimated structure functions.
        sdk (np.ndarray): Structure functions' SDK.
    """
    #
    qorders = np.arange(1, max_qorder + 1)

    def _calc_sfn(dB, qorder):
        return np.sum(np.nanmean(dB ** qorder, axis=0))

    def process_scale(tau):
        dB  = np.abs(flucts(tau, B, five_points_sfunc=five_points_sfunc))
        sfn = np.array([_calc_sfn(dB, qorder) for qorder in qorders])

        sdk = sfn.T[3] / np.sum(np.nanmean(dB ** 2, axis=0) ** 2)

        return sfn, sdk

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_scale)(tau) for tau in scales
    )

    sfn, sdk = zip(*results)

    return np.array(sfn), np.array(sdk)


def trace_PSD_wavelet(x,
                      y,
                      z, 
                      dt, 
                      dj,
                      mother_wave='morlet'):
    """
    Method to calculate the  power spectral density using wavelet method.
    Parameters
    ----------
    x,y,z: array-like
        the components of the field to apply wavelet tranform
    dt: float
        the sampling time of the timeseries
    dj: determines how many scales are used to estimate wavelet coeff
    
        (e.g., for dj=1 -> 2**numb_scales 
    mother_wave: str
        The main waveform to transform data.
        Available waves are:
        'gaussian':
        'paul': apply lomb method to compute PSD
        'mexican_hat':
    Returns
    -------
    db_x,db_y,db_zz: array-like
        component coeficients of th wavelet tranform
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    scales : list
        The scales at which wavelet was estimated
    """
    
    mother_wave_dict = {
    'gaussian': wavelet.DOG(),
    'paul': wavelet.Paul(),
    'mexican_hat': wavelet.MexicanHat()}
    

    if mother_wave in mother_wave_dict.keys():
        mother_morlet = mother_wave_dict[mother_wave]
    else:
        mother_morlet = wavelet.Morlet()
        
    N                                       = len(x)


    db_x, sj, freqs, coi, signal_ft, ftfreqs = wavelet.cwt(x, dt, dj, wavelet=mother_morlet)
    db_y, _, freqs, _, _, _                  = wavelet.cwt(y, dt, dj, wavelet=mother_morlet)
    db_z, _, freqs, _, _, _                  = wavelet.cwt(z, dt, dj, wavelet=mother_morlet)
     
    # Estimate trace powerspectral density
    PSD = (np.nanmean(np.abs(db_x)**2, axis=1) + np.nanmean(np.abs(db_y)**2, axis=1) + np.nanmean(np.abs(db_z)**2, axis=1)   )*( 2*dt)
    
    # Remember!
    scales = (1/freqs)/dt
    
    
    return db_x, db_y, db_z, freqs, PSD, scales



def trace_PSD_cwt_ssqueezepy(x, 
                             y,
                             z, 
                             dt,
                             nv =16,
                             scales='log-piecewise',
                             wavelet=None,
                             wname=None,
                             l1_norm=False,
                             est_PSD=True):
    """
    Method to calculate the wavelet coefficients and  power spectral density using wavelet method.
    Parameters
    ----------
    x,y,z: array-like
        the components of the field to apply wavelet tranform
    dt: float
        the sampling time of the timeseries
        
    scales: str['log', 'log-piecewise', 'linear', 'log:maximal', ...]
                / np.ndarray
            CWT scales.
    Returns
    -------
    W_x, W_y, W_zz: array-like
        component coeficients of th wavelet tranform
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    scales : list
        The scales at which wavelet was estimated
    """
    
    if wavelet is None:
        wavelet    = ssqueezepy.Wavelet(('morlet', {'mu': 13.4}))
    else:
        wavelet    = ssqueezepy.Wavelet((wname, {'mu': 13.4}))  
        
    if  scales is  None:
        scales = 'log'

    # Estimate sampling frequency
    fs         = 1/dt
    # Estimate wavelet coefficients
    Wx, scales = ssqueezepy.cwt(x, wavelet, scales, fs, l1_norm=l1_norm, nv=nv)
    Wy, _      = ssqueezepy.cwt(y, wavelet, scales, fs, l1_norm=l1_norm, nv=nv)
    Wz, _      = ssqueezepy.cwt(z, wavelet, scales, fs, l1_norm=l1_norm, nv=nv)
    
    # Estimate corresponding frequencies
    freqs      = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(x), fs)
    
    if est_PSD:
        # Estimate trace powers pectral density
        PSD = (np.nanmean(np.abs(Wx)**2, axis=1) + np.nanmean(np.abs(Wy)**2, axis=1) + np.nanmean(np.abs(Wz)**2, axis=1)   )*( 2*dt)
    else:
        PSD = None
    
    return Wx, Wy, Wz, freqs, PSD, scales


def TracePSD(x, 
             y,
             z,
             dt,
             remove_mean=False):
    """ 
    Estimate Power Spectral Density (PSD).

    Parameters:
        x, y, z (np.ndarray or pandas.Series): Timeseries data for the three components.
        dt (float): Time step (1/sampling frequency).
        remove_mean (bool, optional): If True, remove the mean from the input timeseries. Default is False.

    Returns:
        tuple: A tuple containing:
            freqs (np.ndarray): Array of frequencies.
            B_pow (np.ndarray): Power spectral density estimates.
    """
    if not isinstance(x, np.ndarray):
        x = x.values
        y = y.values
        z = z.values

    if remove_mean:
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)
        z = z - np.nanmean(z)

    N = len(x)
    xf = np.fft.rfft(x)
    yf = np.fft.rfft(y)
    zf = np.fft.rfft(z)

    B_pow = 2 * (np.abs(xf) ** 2 + np.abs(yf) ** 2 + np.abs(zf) ** 2) / N * dt

    freqs = np.fft.fftfreq(len(x), dt)
    freqs = freqs[freqs > 0]
    idx = np.argsort(freqs)

    return freqs[idx], B_pow[idx]




def estimated_windowed_PSD(mag, magvars,  w_size, chuncktime, windowStr='boxcar', chunk_plot=-1):
    """
    Args:
        fn:
        mag: Dataframe with magnetic field data
        windowStr: Type of window to use
        chunk_plot: [int] Plot the steps in calc f_break for a specific chunk. If *chunk_plot* is integer, that number
        will be used. If *chunk_plot* = 'r', choose a random chunk to plot. Pick an impossible number like
        *chunk_plot=-1* to ensure the steps are not plot.


    Returns:
        freq_log_lst: List of log-spaced frequency arrays
        P_log_lst: List of power as estimated at the frequencies *freq_log_lst*
        f_break_lst: List of break frequency estimates, one per element of *freq_log_lst*
    """

    # log10 frequency ratio
    freqratio = 1.05


    # build chunks
    # time per chunk in seconds
    chunk_duration_sec = chuncktime
    chunktime_str = f'{int(chuncktime)}s'
    ts_chunk = chunkify(mag.index, chunk_duration_sec)

    # get timeseries for the break freq (in-between ts_chunk)
    ts_spec = pd.Series(ts_chunk[:-1]) + pd.Timedelta(
        f'{int(chunk_duration_sec / 2)}s'
    )

    Nchunks = len(ts_chunk)


    # if random plot chunk is selected
    if chunk_plot == 'r':
        chunk_plot = np.random.randint(Nchunks)

    # sampling period
    Ts = (mag.dropna().index.to_series().diff()/np.timedelta64(1, 's')).median()#np.round(mag.index.freq.nanos * 1e-9, decimals=6)

    Fs = 1 / Ts

    P_log_lst = []
    freq_log_lst = []
    spectral_ts_lst = []

    # output raw data to ease plotting later on
    btrace_lst = []
    freq_lst = []

    fb_arr = np.zeros(len(ts_spec))
    fb_ep_arr = np.zeros_like(fb_arr)
    fb_em_arr = np.zeros_like(fb_arr)

    # check if outside freq range
    fb_er_arr = np.zeros_like(fb_arr)

    # check if outside interval
    fb_ei_arr = np.zeros_like(fb_arr)

    # errorbar interval error check
    fb_ebr_arr = np.zeros_like(fb_arr)



    for ti in range(Nchunks - 1):
        # for ti in chunk_plot:

        plotsteps = ti == chunk_plot
        # print('ti = %d' % ti)

        t0str = ts_chunk[ti]
        tNstr = ts_chunk[ti + 1]

        # use strings to get chunk data
        dat = mag[t0str:tNstr][magvars]


        # get chunk size
        N = dat.index.size

        # get the frequencies
        freq0 = fftfreq(N, d=Ts)

        # first half of the vector (for positive frequencies)
        k = np.arange(0, N)
        freq0[k > N / 2] = freq0[k > N / 2] - np.max(freq0)

        # i_half = range(0, int(N / 2))
        # freq = freq0[i_half]
        freq = freq0[freq0 > 0]
        freq_nyq = Fs / 2

        # set up trace matrix
        Bf_tr = np.zeros_like(dat, dtype=complex)

        # for each component of the B field
        for i in range(np.min(dat.shape)):
            # set window
            # ft_window = window_selector(N, win_name=windowStr)
            ft_window = window_selector(N, win_name=windowStr)

            # get the current component
            Bi = dat[dat.columns[i]].values

            # detrend and apply window
            ft_input_signal = mpl.mlab.detrend(Bi) * ft_window

            # get the FFT of the detrended and windowed B-field component, scale by freq
            Bf = fft(ft_input_signal, N) / np.sqrt(N / Ts)

            # get the transpose
            Bf_tr[:, i] = Bf.transpose()

        # take sum along the diagonal
        Btr = np.sum(np.squeeze(Bf_tr * np.conj(Bf_tr)), axis=1)
        # only use positive freq
        Btr = Btr[freq0 > 0]

        # smooth the trace
        Btr_smooth = smooth(np.real(Btr), w_size)

        # number of frequencies to use in logspace
        numfreqs = np.floor((np.log10(np.max(freq) / np.min(freq))) / np.log10(freqratio))

        # set up log-spaced frequency array
        freq_log = np.logspace(np.log(np.min(freq)) / np.log(freqratio),
                               np.log(freq_nyq) / np.log(freqratio),
                               base=freqratio, num=int(numfreqs))

        # interpolate smoothed trace to log-spaced freqs
        Plog = np.interp(freq_log, freq, Btr_smooth)

        return freq_log, Plog

def power_spec(signal,npoints):
    """Computes FFT for the signal, discards the zero freq and the
    above-Nyquist freqs. Auto-pads signals nonmultple of npoints, auto-averages results from streams longer than npoints.
    Thus, npoints results in npoints/2 bands.

    Returns a numpy array, each element represents the raw amplitude of a frequency band.
     """

    signal = signal.copy()
    if divmod(len(signal),npoints)[1] != 0:
        round_up = len(signal) / npoints * npoints + npoints
        signal.resize( round_up )

    window = scipy.signal.hanning(npoints)
   # print(int(len(signal) / npoints))
   # print(signal)
    window_blocks = scipy.vstack(
        [window for _ in range(int(len(signal) / npoints))]
    )

    signal_blocks = signal.reshape((-1,npoints))

    windowed_signals = signal_blocks * window_blocks

    ffts = np.fft.rfft(windowed_signals)[:,1:]

    result = pow(abs(ffts),2) / npoints
    result = result.mean(0)

    return result


@njit(nogil=True, parallel=True)
def norm_factor_Gauss_window(phys_space_scales, scales, dt, lambdaa=3):
    """
    Calculate normalization factor and Gaussian window.

    Parameters:
        phys_space_scales (float): Physical space scales.
        scales (np.ndarray): Array of scales.
        dt (float): Time step.
        lambdaa (float, optional): Gaussian parameter. Default is 3.

    Returns:
        int: Length of the window.
        np.ndarray: Array of the multiplication factors.
        float: Normalization factor.
    """
    s = scales
    numer = np.arange(-3 * phys_space_scales, 3 * phys_space_scales, dt)
    multiplic_fac = np.exp(-(numer) ** 2 / (2 * (lambdaa ** 2) * (s ** 2)))
    norm_factor = np.sum(multiplic_fac)
    window = len(multiplic_fac)

    return window, multiplic_fac, norm_factor


def estimate_wavelet_coeff(B_df, V_df, dj, lambdaa=3, pycwt=False):
    """
    Method to calculate the 1) wavelet coefficients in RTN 2) The scale dependent angle between Vsw and Β.

    Parameters:
        B_df (pandas.DataFrame): Magnetic field timeseries dataframe.
        V_df (pandas.DataFrame): Velocity timeseries dataframe.
        dj (float): The time resolution.
        lambdaa (float, optional): Gaussian parameter. Default is 3.
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
    # Estimate sampling time of timeseries
    dt_B = find_cadence(B_df)
    dt_V = find_cadence(V_df)

    if dt_V != dt_B:
        V_df = newindex(V_df, B_df.index, interp_method='linear')

    # Common dt
    dt = dt_B

    # Turn columns of df into arrays
    Br, Bt, Bn = B_df.Br.values, B_df.Bt.values, B_df.Bn.values
    Vr, Vt, Vn = V_df.Vr.values, V_df.Vt.values, V_df.Vn.values

    # Estimate magnitude of magnetic field
    mag_orig = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    # Estimate the magnitude of V vector
    mag_v = np.sqrt(Vr ** 2 + Vt ** 2 + Vn ** 2)

    angles = pd.DataFrame()
    VBangles = pd.DataFrame()

    # Estimate PSD and scale dependent fluctuations
    if pycwt:
        db_x, db_y, db_z, freqs, PSD, scales = trace_PSD_wavelet(Br, Bt, Bn, dt, dj, mother_wave='morlet')
    else:
        # To avoid rewriting the code (2/dj) is a good compromise (and it doesn't really matter)
        db_x, db_y, db_z, freqs, PSD, scales = trace_PSD_cwt_ssqueezepy(Br, Bt, Bn, dt, nv=int(2/dj))

    # Calculate the scales in physical space and units of seconds
    phys_space_scales = 1 / (freqs)

    for ii in range(len(scales)):
        try:
            window, multiplic_fac, norm_factor = norm_factor_Gauss_window(phys_space_scales[ii], scales[ii], dt, lambdaa)

            # Estimate scale dependent background magnetic field using a Gaussian averaging window
            res2_Br = (1 / norm_factor) * signal.convolve(Br, multiplic_fac, 'same')
            res2_Bt = (1 / norm_factor) * signal.convolve(Bt, multiplic_fac, 'same')
            res2_Bn = (1 / norm_factor) * signal.convolve(Bn, multiplic_fac, 'same')

            # Estimate magnitude of scale dependent background
            mag_bac = np.sqrt(res2_Br ** 2 + res2_Bt ** 2 + res2_Bn ** 2)

            # Estimate angle
            angles[str(ii + 1)] = np.arccos(res2_Br / mag_bac) * 180 / np.pi

            # Estimate VB angle
            VBangles[str(ii + 1)] = np.arccos((Vr * res2_Br + Vt * res2_Bt + Vn * res2_Bn) / (mag_bac * mag_v)) * 180 / np.pi

            # Restrict to 0 < Θvb < 90
            VBangles[str(ii + 1)][VBangles[str(ii + 1)] > 90] = 180 - VBangles[str(ii + 1)][VBangles[str(ii + 1)] > 90]

            # Restrict to 0 < Θvb < 90
            angles[str(ii + 1)][angles[str(ii + 1)] > 90] = 180 - angles[str(ii + 1)][angles[str(ii + 1)] > 90]
        except:
            pass

    return db_x, db_y, db_z, angles, VBangles, freqs, PSD, phys_space_scales, scales

@jit( parallel =True)
def estimate_PSD_wavelets(dbx_par, dby_par, dbz_par, unique_par, freq_par, dt):

    PSD_par = []

    for i in prange(len(unique_par)):
        index_par = np.where(freq_par== unique_par[i])[0].astype(np.int64)

        dbx_par_sq = np.nanmean(dbx_par[index_par]**2)
        dby_par_sq = np.nanmean(dby_par[index_par]**2)
        dbz_par_sq = np.nanmean(dbz_par[index_par]**2)

        PSD_par.append((dbx_par_sq  + dby_par_sq  + dbz_par_sq)*( 2*dt))
    return unique_par, np.array(PSD_par)




@jit( parallel =True, nopython=True)
def structure_functions_wavelets(db_x, db_y, db_z, angles,  scales, dt, max_moment, per_thresh, par_thresh):
    
    tau = scales*dt
    m_vals = np.arange(1, max_moment+1)
    
    sfunc_par  = np.zeros((len(tau), len(m_vals))) 
    sfunc_per  = np.zeros((len(tau), len(m_vals))) 
    counts_par = np.zeros((len(tau), len(m_vals))) 
    counts_per = np.zeros((len(tau), len(m_vals))) 
   # print(sfunc_per)
    

    for j in prange(len(tau)):
        
        dbtot     = (db_x[j]*np.conjugate(db_x[j]) + db_y[j]*np.conjugate(db_y[j])  +db_z[j]*np.conjugate(db_z[j]) )**(1/2)
        index_per = (np.where(angles[j]>per_thresh)[0])
        index_par = (np.where(angles[j]<par_thresh)[0])

        for m in prange(len( m_vals)):
            
            sfunc_par[j, m]  = np.nanmean(np.abs(dbtot[index_par.astype(np.int64)]/np.sqrt(tau[j]))**m_vals[m])
            sfunc_per[j, m]  = np.nanmean(np.abs(dbtot[index_per.astype(np.int64)]/np.sqrt(tau[j]))**m_vals[m])
            counts_par[j, m] = len(index_par)#.astype('float')
            counts_per[j, m] = len(index_per)#.astype('float')
    return tau, sfunc_par, sfunc_per, counts_par, counts_per



def estimate_anisotropic_PSD_wavelets(db_x, db_y, db_z, angles, freqs,   dt,  per_thresh, par_thresh):
    """
    Method to calculate the par and perp Power Spectral Density (PSD) of a signal using wavelets 

    Parameters
    ----------
    db_x: numpy array
        The x component of the magnetic field timeseries data
    db_y: numpy array
        The y component of the magnetic field timeseries data
    db_z: numpy array
        The z component of the magnetic field timeseries data
    angles: numpy array
        The scale dependent angle between Vsw and Β
    freqs: list
        The frequency of the corresponding psd points
    dt: float
        The time step of the signal
    per_thresh: float
        The threshold for perpendicular intervals
    par_thresh: float
        The threshold for parallel intervals

    Returns
    -------
    PSD_par: numpy array
        The Power Spectral Density for parallel intervals
    PSD_per: numpy array
        The Power Spectral Density for perpendicular intervals
    """

    PSD_par = np.zeros(len(freqs))
    PSD_per = np.zeros(len(freqs)) 

    for i in range(np.shape(angles)[1]):

        index_per = (np.where(angles.T[i]>per_thresh)[0]).astype(np.int64)
        index_par = (np.where(angles.T[i]<par_thresh)[0]).astype(np.int64)


        PSD_par[i]  = (np.nanmean(np.abs(np.array(db_x[i])[index_par])**2) + np.nanmean(np.abs(np.array(db_y[i])[index_par])**2) + np.nanmean(np.abs(np.array(db_z[i])[index_par])**2) ) * ( 2*dt)
        PSD_per[i]  = (np.nanmean(np.abs(np.array(db_x[i])[index_per])**2) + np.nanmean(np.abs(np.array(db_y[i])[index_per])**2) + np.nanmean(np.abs(np.array(db_z[i])[index_per])**2) ) * ( 2*dt)

    return PSD_par, PSD_per



# First method to find the deHoffmann-Teller frame velocity
def HoffmannTellerizer(v, B):
    '''
    Finds the ideal deHoffmann-Teller frame velocity using the linear solution
    described in Paschmann1998 using the measured plasma velocity and magnetic 
    field vectors to minimize  E' = -v x B.
    This analysis must be performed over a discrete timerange, since a single-point 
    solution would just reduce to v_HT = v.
    
    Inputs:
        v: rank2 [n,3] vector of plasma velocity in km/s
        B: rank2 [n,3] vector magnetic field
    Outputs:
        v_HT: rank1 [3] vector describing deHoffmann-T
    '''
    
    
    def KBuilderHoffmannTeller(v, B):
        K = np.zeros((len(v), 3, 3))
        TEMP_K_0 = np.zeros((3, 3))
        Bmag_squared = np.linalg.norm(B, axis=1)**2

        K[:, 0, 0] = Bmag_squared * (1 - (B[:, 0]*B[:, 0]) / Bmag_squared)
        K[:, 1, 1] = Bmag_squared * (1 - (B[:, 1]*B[:, 1]) / Bmag_squared)
        K[:, 2, 2] = Bmag_squared * (1 - (B[:, 2]*B[:, 2]) / Bmag_squared)

        K[:, 0, 1] = Bmag_squared * (0 - (B[:, 0]*B[:, 1]) / Bmag_squared)
        K[:, 0, 2] = Bmag_squared * (0 - (B[:, 0]*B[:, 2]) / Bmag_squared)
        K[:, 1, 2] = Bmag_squared * (0 - (B[:, 1]*B[:, 2]) / Bmag_squared)

        K[:, 1, 0] = K[:, 0, 1]  
        K[:, 2, 0] = K[:, 0, 2]  
        K[:, 2, 1] = K[:, 1, 2]  

        TEMP_K_0[0, 0] = np.nanmean(K[:, 0, 0])
        TEMP_K_0[1, 1] = np.nanmean(K[:, 1, 1])
        TEMP_K_0[2, 2] = np.nanmean(K[:, 2, 2])

        TEMP_K_0[0, 1] = np.nanmean(K[:, 0, 1])
        TEMP_K_0[0, 2] = np.nanmean(K[:, 0, 2])
        TEMP_K_0[1, 2] = np.nanmean(K[:, 1, 2])

        TEMP_K_0[1, 0] = TEMP_K_0[0, 1]  # Woohoo symmetry
        TEMP_K_0[2, 0] = TEMP_K_0[0, 2]
        TEMP_K_0[2, 1] = TEMP_K_0[1, 2]

        return K, TEMP_K_0

    K, TEMP_K_0 = KBuilderHoffmannTeller(v, B)

    # Explicitly build matrix from TEMP_K_0 outputs
    K_0 = np.array([[TEMP_K_0[0, 0], TEMP_K_0[0, 1], TEMP_K_0[0, 2]],
                    [TEMP_K_0[0, 1], TEMP_K_0[1, 1], TEMP_K_0[1, 2]],
                    [TEMP_K_0[0, 2], TEMP_K_0[1, 2], TEMP_K_0[2, 2]]])

    K_0_inverse = np.linalg.inv(K_0)

    # K*v
    Kdotv = np.zeros((len(v), 3))
    Kdotv[:, 0] = K[:, 0, 0]*v[:, 0] + K[:, 0, 1]*v[:, 1] + K[:, 0, 2]*v[:, 2]
    Kdotv[:, 1] = K[:, 1, 0]*v[:, 0] + K[:, 1, 1]*v[:, 1] + K[:, 1, 2]*v[:, 2]
    Kdotv[:, 2] = K[:, 2, 0]*v[:, 0] + K[:, 2, 1]*v[:, 1] + K[:, 2, 2]*v[:, 2]

    # <K*v>
    Kdotv_average = np.nanmean(Kdotv, axis=0)

    # K_0^-1 * <K*v>
    v_HT = solve(K_0, Kdotv_average)

    return v_HT

# Second Method:Provided by Trevor Bowen
def calculate_dhtf(v, b):
    """
    Calculate dhtf vector using the given v and b arrays.

    Parameters:
        v (ndarray): Input array v.
        b (ndarray): Input array b.

    Returns:
        ndarray: The calculated dhtf vector.
    """

    # Calculate dv by subtracting the mean of each column of v from v
    dvx = v[:, 0] - np.nanmean(v[:, 0])
    dvy = v[:, 1] - np.nanmean(v[:, 1])
    dvz = v[:, 2] - np.nanmean(v[:, 2])
    dv = np.column_stack((dvx, dvy, dvz))

    # Compute cross products of dv and b
    #cp = np.cross(dv, b)
    cp        = np.cross(v, b)

    # Compute the dot products of each component of b with itself
    bx_bx = np.nansum(b[:, 0] * b[:, 0])
    bx_by = np.nansum(b[:, 0] * b[:, 1])
    bx_bz = np.nansum(b[:, 0] * b[:, 2])
    by_by = np.nansum(b[:, 1] * b[:, 1])
    by_bz = np.nansum(b[:, 1] * b[:, 2])
    bz_bz = np.nansum(b[:, 2] * b[:, 2])

    # Construct the matrix mat
    mat = np.array([[by_by + bz_bz, -bx_by, -bx_bz],
                    [-bx_by, bx_bx + bz_bz, -by_bz],
                    [-bx_bz, -by_bz, bx_bx + by_by]])

    # Perform singular value decomposition
    U, S, VT = np.linalg.svd(mat)

    # Calculate result using the singular value decomposition
    result = np.diag(S) @ VT.T
    # The @ operator performs matrix multiplication in numpy

    # Calculate the inverse matrix
    inverse = VT.T @ np.diag(1. / S) @ U.T

    # Calculate the components of vec
    vecx = np.nansum(cp[:, 2] * b[:, 1]) - np.nansum(cp[:, 1] * b[:, 2])
    vecy = np.nansum(cp[:, 0] * b[:, 2]) - np.nansum(cp[:, 2] * b[:, 0])
    vecz = np.nansum(cp[:, 1] * b[:, 0]) - np.nansum(cp[:, 0] * b[:, 1])
    vec = np.array([vecx, vecy, vecz])

    # Calculate dhtf using the inverse matrix and vec
    dhtf = inverse @ vec

    return dhtf


# def select_intervals_WIND_analysis(E, thresh_value, hours_needed,  min_toler =60):

#     dt_df = E[E.values > thresh_value].dropna().index.to_series().diff() / np.timedelta64(1, 's')


#     bad_indices               = dt_df[np.array(dt_df) <= min_toler].index.to_numpy()
#     indices_in_original_df    = np.where(E.index.isin(bad_indices))[0]-1

#     fix_array                        = E.values.T[0]#
#     fix_array[indices_in_original_df]= 0.5
#     E['E']                           = fix_array#[indices_in_original_df]
    
    
#     dt_df = (E[E.values > thresh_value].dropna().index.to_series().diff() / np.timedelta64(1, 's'))

#     init_dates, intervals= dt_df[np.array(dt_df) >= hours_needed*3600].index, dt_df[np.array(dt_df) >= hours_needed*3600].values
    
#     selected_dates = {}
#     for index, (init_date, interval) in enumerate(zip(init_dates, intervals)):
        
#         # Convert string to datetime
#         fin_dt                  =  pd.to_datetime(init_date)
#         init_dt                 =  fin_dt- pd.Timedelta(seconds=interval)
#         selected_dates[str(index)] =  {'Start':init_dt, 'End': fin_dt}
#     return pd.DataFrame(selected_dates).T


def select_intervals_WIND_analysis(E, thresh_value, hours_needed,  min_toler =60):

    dt_df = E[E.values > thresh_value].dropna().index.to_series().diff() / np.timedelta64(1, 's')


    bad_indices               = dt_df[np.array(dt_df) <= min_toler].index.to_numpy()
    indices_in_original_df    = np.where(E.index.isin(bad_indices))[0]-1


    E_old                            = E.copy()
    fix_array                        = E.values.T[0]#
    fix_array[indices_in_original_df]= 3.5
    E['E']                           = fix_array#[indices_in_original_df]
    
    
    dt_df = (E[E.values > thresh_value].dropna().index.to_series().diff() / np.timedelta64(1, 's'))

    init_dates, intervals= dt_df[np.array(dt_df) >= hours_needed*3600].index, dt_df[np.array(dt_df) >= hours_needed*3600].values
    
    selected_dates = {}
    for index, (init_date, interval) in enumerate(zip(init_dates, intervals)):
        
        # Convert string to datetime
        fin_dt                     =  pd.to_datetime(init_date)
        init_dt                    =  fin_dt- pd.Timedelta(seconds=interval)
        
        ind                        = func.find_ind_of_closest_dates(E, [init_dt, fin_dt])
        
        vals_selected              = E_old[ind[0]:ind[1]]
        selected_dates[str(index)] =  {'Start'         :init_dt,
                                       'End'           : fin_dt,
                                       'Perc_exc_thres': 100*len(vals_selected[vals_selected.values> thresh_value])/len(vals_selected)}
    return pd.DataFrame(selected_dates).T


def variance_anisotropy_verdini(av_window,
                                B,
                                av_hours=1,
                                return_df =False):
    """
    Calculate variance anisotropy as defined by Verdini et al. (2018).

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.Series: The variance anisotropy values.
    """
    lag = func.find_cadence(B)
    av_window1 = int(av_hours * 3600 / lag)

    # Calculate variance of components after applying moving average
    b = np.sqrt(((B- B.rolling(av_window, center=True).mean()) ** 2)
                      .rolling(av_window1, center=True).mean())
                     

    # Calculate variance anisotropy
    quant = (b['Bt'] ** 2 + b['Bn'] ** 2) / b['Br'] ** 2
    if return_df:
        return pd.DataFrame({'E': quant.values}, index=quant.index)
    else:
        return quant
    
    
def variance_anisotropy_verdini_spec(av_window,
                                B,
                                av_hours=1,
                                return_df =False):
    """
    Calculate variance anisotropy as defined by Verdini et al. (2018).

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.Series: The variance anisotropy values.
    """
    lag = func.find_cadence(B)
    av_window1 = int(av_hours * 3600 / lag)

    # Calculate variance of components after applying moving average
    b = np.sqrt(((B- B.rolling('2h', center=True).mean()) ** 2).rolling('2h', center=True).mean())
                     

    # Calculate variance anisotropy
    quant = ((b['Bt'] ** 2 + b['Bn'] ** 2) / b['Br'] ** 2).rolling('2H', center=True).mean()
    if return_df:
        return pd.DataFrame({'E': quant.values}, index=quant.index)
    else:
        return quant



def compressibility_complex_squire(av_window,
                                   B, 
                                   keys     = ['Br', 'Bt', 'Bn'],
                                   av_hours = 1 ):
    """
    Calculate compressibility as defined by Squire et al. (2021).

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.DataFrame: DataFrame with 'DateTime' and 'Values' columns representing the compressibility values.
    """
    lag                    = find_cadence(B)
    av_window1             = int(av_hours * 3600 / lag)
    

    B['mod_sqrd']        = B[keys[0]] ** 2 + B[keys[1]] ** 2 + B[keys[2]] ** 2

    diff                   = (B - B.rolling(av_window, center=True).mean()) 
    rms                    =  np.sqrt((diff**2).rolling(av_window1, center=True).mean())

    return pd.DataFrame( rms['mod_sqrd'] /(rms[keys[0]]**2  +  rms[keys[1]]**2 +  rms[keys[2]]**2 ))
    #denom                  = np.sqrt((np.sqrt(diff[keys[0]]**2 + diff[keys[1]]**2 + diff[keys[2]]**2  )**4).rolling(av_window1, center=True).mean())
    
    
    
    return pd.DataFrame( rms /denom)


def compressibility_complex_chen(av_window,
                                   B,
                                   keys    = ['Br', 'Bt', 'Bn'],
                                   av_hours=1,
                                   if_use_same_window= False):
    """
    Calculate compressibility as defined by Chen et al. (2020).

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.DataFrame: DataFrame with 'DateTime' and 'Values' columns representing the compressibility values.
    """
    lag             = find_cadence(B)
    av_window1      = int(av_hours * 3600 / lag)

    B['mod']      = np.sqrt(B[keys[0]] ** 2 + B[keys[1]] ** 2 + B[keys[2]] ** 2)
    
    diff            = (B - B.rolling(av_window, center=True).mean()) 
    if if_use_same_window:
        rms             =  np.sqrt((diff**2).rolling(av_window, center=True).mean())        
    else:
        rms             =  np.sqrt((diff**2).rolling(av_window1, center=True).mean())

    return pd.DataFrame( rms['mod']**2 /(rms[keys[0]]**2 +  rms[keys[1]]**2  +  rms[keys[2]]**2  ))

def calculate_compressibility( 
                               window,
                               B,
                               keys    = ['Br', 'Bt', 'Bn'],
                               five_points_sfunc=True):
    

    B['compress'] = np.sqrt(B[keys[0]]**2 + B[keys[1]]**2 + B[keys[2]]**2)
    dB            =  flucts(
                                 window,
                                 B,
                                 five_points_sfunc = five_points_sfunc,
                                 return_dataframe  = True)


    return pd.DataFrame(np.abs(dB['compress'])/np.sqrt((dB[keys[0]].values)**2 + (dB[keys[1]].values)**2 + (dB[keys[2]].values)**2))


def norm_fluct_amplitude(window,
                            B,
                            keys             = ['Br', 'Bt', 'Bn'],
                            av_hours         = 2,
                            denom_av_hours   = 2,
                            five_points_sfunc= True):
    
    """
    Calculate normalize fluctuation amplitude

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.Series: The variance anisotropy values.
    """
    lag        = func.find_cadence(B)
    if type(denom_av_hours)==str:
        av_window1 =denom_av_hours 
    else:
        av_window1 = int(denom_av_hours * 3600 / lag)
        
        
    av_window2 = int(av_hours * 3600 / lag)
    # Calculate rms of components after applying moving average
    rms = np.sqrt(((B - B.rolling(window, center=True).mean()) ** 2)
                       .rolling(av_window2, center=True).mean())
    
#     dB            =  turb.flucts(
#                                  window,
#                                  B,
#                                  five_points_sfunc = five_points_sfunc,
#                                  return_dataframe  = True)

    return pd.DataFrame((rms[keys[0]]+ rms[keys[1]] +  rms[keys[2]])/np.sqrt(B[keys[0]]**2 + B[keys[1]]**2 + B[keys[2]]**2).rolling(av_window1, center=True).mean())
    #return  pd.DataFrame((dB[keys[0]].values)**2 + (dB[keys[1]].values)**2 + (dB[keys[2]].values)**2)/np.sqrt(B[keys[0]]**2 + B[keys[1]]**2 + B[keys[2]]**2).rolling(av_window1, center=True).mean())





def estimate_PVI(B_df,
                 hmany,
                 taus,
                 di,
                 Vsw,
                 hours,
                 keys              = ['Br', 'Bt', 'Bn'],
                 five_points_sfunc = True,
                 PVI_vec_or_mod    = 'vec',
                 use_taus          = False,
                 return_only_PVI   = False,
                 n_jobs            =-1,
                 input_flucts      = False,
                 dbs               = None):
    
    B_resampled = B_df.copy()
    av_hours    = hours * 3600
    lag         = (B_resampled.index[1] - B_resampled.index[0])/ np.timedelta64(1, 's')
    av_window   = int(av_hours / lag)


    results = Parallel(n_jobs=n_jobs)(delayed(estimate_PVI_single_iteration)(kk,
                                                                             B_resampled.copy(),
                                                                             hmany,
                                                                             taus,
                                                                             di,
                                                                             Vsw,
                                                                             lag,
                                                                             av_window,
                                                                             keys               =  keys,
                                                                             five_points_sfunc  =  five_points_sfunc,
                                                                             PVI_vec_or_mod     =  PVI_vec_or_mod,
                                                                             use_taus           =  use_taus,
                                                                             return_only_PVI    =  return_only_PVI,                 
                                                                             input_flucts       =  input_flucts,
                                                                             dbs                =  dbs) for kk in range(len(hmany)))


    for kk in range(len(hmany)):
        if PVI_vec_or_mod == 'vec':
            B_resampled[f'PVI_{str(kk)}'] = results[kk][f'PVI_{str(kk)}']
        else:
            B_resampled[f'PVI_mod_{str(kk)}'] = results[kk][f'PVI_mod_{str(kk)}']
   # del  B_resampled[keys[0]], B_resampled[keys[1]], B_resampled[keys[2]]
    
    # Now delete for memory
    keys_to_delete = keys

    for key in keys_to_delete:
        if key in B_resampled:
            del B_resampled[key]

    return B_resampled



def estimate_PVI_single_iteration(kk,
                                  B_resampled,
                                  hmany,
                                  taus,
                                  di,
                                  Vsw,
                                  lag,
                                  av_window,
                                  keys              = ['Br', 'Bt', 'Bn'],
                                  five_points_sfunc = True,
                                  PVI_vec_or_mod    = 'vec',
                                  use_taus          = False,
                                  return_only_PVI   = False,
                                  input_flucts      = False,
                                  dbs               = None):
    if use_taus:
        tau       = taus[kk]
        hmany[kk] =  taus[kk]*lag*Vsw/di
    else:
        tau = round((hmany[kk] * di) / (Vsw * lag))

        if tau < 1:
            print('The value of hmany you chose is too low. You will have to use higher resol mag data!')
            while tau < 1:
                hmany[kk] = hmany[kk] + 0.01 * hmany[kk]
                tau = round((hmany[kk] * di) / (Vsw * lag))
                print('The value was set to the minimum possible, hmany=', hmany[kk])
    
    ### Estimate PVI ###
    if tau > 0:
        if PVI_vec_or_mod =='vec':
            
            if input_flucts:
                db = dbs
            else:
            
                # Estimate increments
                db =  flucts(tau,
                             B_resampled,
                             five_points_sfunc = five_points_sfunc,
                             return_dataframe  = True)

            B_resampled['DBtotal']         = result = np.sqrt(sum((db[key])**2 for key in keys))
            B_resampled['DBtotal_squared'] = B_resampled['DBtotal']**2
            denominator = np.sqrt(B_resampled['DBtotal_squared'].rolling(av_window, center=True).mean())


            PVI_dB = pd.DataFrame({'DateTime' : B_resampled.index,
                                    'PVI'     : B_resampled['DBtotal'] / denominator})
            PVI_dB = PVI_dB.set_index('DateTime')
            B_resampled[f'PVI_{str(kk)}'] = PVI_dB.values
            del B_resampled['DBtotal_squared'], B_resampled['DBtotal']
        else:
            B_resampled['B_modulus']       = np.sqrt(sum((B_resampled[key])**2 for key in keys))
            
            # Estimate increments
            db =  flucts(tau,
                         pd.DataFrame(B_resampled['B_modulus']),
                         five_points_sfunc = five_points_sfunc,
                         return_dataframe  = True)

            B_resampled['DBtotal']         = db['B_modulus']
            B_resampled['DBtotal_squared'] = B_resampled['DBtotal']**2
            denominator                    = np.sqrt(B_resampled['DBtotal_squared'].rolling(av_window, center=True).mean())

            PVI_dB = pd.DataFrame({'DateTime': B_resampled.index,
                                    'PVI': B_resampled['DBtotal'] / denominator})
            PVI_dB = PVI_dB.set_index('DateTime')
            B_resampled[f'PVI_mod_{str(kk)}'] = PVI_dB.values
            
            if return_only_PVI:
                keys_to_delete = ['DBtotal_squared', 'DBtotal', 'B_modulus'] + keys
                
                for key in keys_to_delete:
                    if key in B_resampled:
                        del B_resampled[key]

            else:
                del B_resampled['DBtotal_squared'], B_resampled['DBtotal'], B_resampled['B_modulus']
    elif PVI_vec_or_mod:
        B_resampled[f'PVI_{str(kk)}'] = np.nan * B_resampled.Br.values
    else:
        B_resampled[f'PVI_mod_{str(kk)}'] = np.nan * B_resampled.Br.values

    return B_resampled





def remove_big_gaps(big_gaps, B_resampled):
    """ Removes big gaps identified earlier """ 
    if len(big_gaps) <= 0:
        return B_resampled

    for o in range(len(big_gaps)):
        if o%50==0:
            print(f"Completed = {str(100 * o / len(big_gaps))}")
        dt2 = big_gaps.index[o]
        dt1 = big_gaps.index[o]-datetime.timedelta(seconds=big_gaps[o])
        B_resampled1 = (
            B_resampled[(B_resampled.index < dt1) | (B_resampled.index > dt2)]
            if o == 0
            else B_resampled1[
                (B_resampled1.index < dt1) | (B_resampled1.index > dt2)
            ]
        )
    nindex = pd.date_range( B_resampled1.index[0], periods=len( B_resampled1.index), freq=str(1e3*(B_resampled1.index[1]-B_resampled1.index[0])/np.timedelta64(1,'s'))+"ms")
    return B_resampled1.reindex(nindex)

def estimate_WT_distribution(big_gaps, B_resampled, PVI_thresholds, hmany):
    """ ESTIMATE WT DISTRIBUTIONS, remove the gaps indentified earlier """ 
    if len(big_gaps)>0:
        for o in range(len(big_gaps)):
            if o%50==0:
                print(f"Completed = {str(100 * o / len(big_gaps))}")
            dt2 = big_gaps.index[o]
            dt1 = big_gaps.index[o]-datetime.timedelta(seconds=big_gaps[o])
            if o==0:
                B_resampled1   = B_resampled[(B_resampled.index<dt1) | (B_resampled.index>dt2) ]
            else:
                B_resampled1   = B_resampled1[(B_resampled1.index<dt1) | (B_resampled1.index>dt2) ]   

        nindex = pd.date_range( B_resampled1.index[0], periods=len( B_resampled1.index), freq=str(1e3*(B_resampled1.index[1]-B_resampled1.index[0])/np.timedelta64(1,'s'))+"ms")
        B_resampled1 = B_resampled1.reindex(nindex)
    else:
        B_resampled1 = B_resampled


    WT     = {}
    for k in hmany:
        thresh = {}
        for i in PVI_thresholds:
            f2 = B_resampled1[f'PVI_{str(k)}'][B_resampled1[f'PVI_{str(k)}'] > i]
            time        = (f2.index.to_series().diff()/np.timedelta64(1, 's'))
            #res2        = pdf(time.values[1:], hmany_bins_PDF_WT, 1,1)
            thresh[f'PVI>{str(i)}'] = time.values[1:]

        WT[f'PVI_{str(k)}'] = thresh

    return WT


import random
@jit(parallel =True)
def estimate_kurtosis_with_rand_samples(hmany_stds, di, vsw, xvals, yvals, nxbins, nrounds, sample_size):
    """" 
     Estimate the kurtosis of a field, by drawing random samples from the distribution. 
    """
    # convert to di units
    fxvals = xvals *vsw/di

    bins = np.logspace(np.log10(np.nanmin(fxvals)), np.log10(np.nanmax(fxvals)), nxbins)

    gfg         = np.digitize(fxvals, bins)
    unique_vals = np.unique(gfg)

    kurt       = np.empty((len(unique_vals),nrounds))*np.nan
    xvalues    = np.empty((len(unique_vals),nrounds))*np.nan
    counts     = np.empty((len(unique_vals),nrounds))*np.nan
    Sf1_f      = np.empty((len(unique_vals),nrounds))*np.nan
    Sf2_f      = np.empty((len(unique_vals),nrounds))*np.nan
    Sf2_f      = np.empty((len(unique_vals),nrounds))*np.nan
    Sf3_f      = np.empty((len(unique_vals),nrounds))*np.nan
    Sf4_f      = np.empty((len(unique_vals),nrounds))*np.nan
    Sf5_f      = np.empty((len(unique_vals),nrounds))*np.nan
    Sf6_f      = np.empty((len(unique_vals),nrounds))*np.nan

    for i in prange(len(unique_vals)):
        if np.mod(i,10)==0:
            print('Unique values completed', round(100*i/len(unique_vals),2))
        ynew     = yvals[gfg==unique_vals[i]]
        xnew     = xvals[gfg==unique_vals[i]]
        xnew_f   = fxvals[gfg==unique_vals[i]]
        di_new   = di[gfg==unique_vals[i]]
        Vsw_new  = vsw[gfg==unique_vals[i]]        


        #percentile   = np.percentile(ynew, remove_percntile)
        nanstd       = np.nanstd(ynew)
        init_length  = len(ynew)
        remove_ind   = ~(ynew>hmany_stds*nanstd);



        ynew         = ynew[remove_ind]
        xnew         = xnew[remove_ind];
        di_new       = di_new[remove_ind];
        xnew_f       = xnew_f[remove_ind];
        Vsw_new      = Vsw_new[remove_ind];
        len_xnew_f   = len(xnew_f)
        print('Removed (%)',100*(1-len_xnew_f/init_length))


        sample_size1 = len(xnew_f) if len_xnew_f<sample_size else sample_size
        nrounds1 = 1 if sample_size1<sample_size else nrounds
        index_array = np.arange(0, len_xnew_f,1)
        if sample_size1>0:
            for k in prange(nrounds1):
                if k==0:
                    print('No points',sample_size1)
                if np.mod(k,10)==0:
                    print('Rounds completed',k)

                rand_indices = np.array(random.choices(index_array,k=int(sample_size1)))
                terma        = di_new[rand_indices]/Vsw_new[rand_indices]
                termb        = ynew[rand_indices]/np.sqrt(xnew[rand_indices] )

                Sf1          = np.nanmean(((terma)**(1/2))*np.abs(termb)**1)
                Sf2 = np.nanmean(terma**1 * np.abs(termb)**2)
                Sf3          = np.nanmean(((terma)**(3/2))*np.abs(termb)**3)
                Sf4          = np.nanmean(((terma)**(4/2))*np.abs(termb)**4)
                Sf5          = np.nanmean(((terma)**(5/2))*np.abs(termb)**5)
                Sf6          = np.nanmean(((terma)**(6/2))*np.abs(termb)**6)  

                kurt[i, k]       = Sf4/Sf2**2
                xvalues[i, k]    = np.nanmean(xnew_f[rand_indices])
                counts[i,k]      = sample_size1
                Sf1_f[i,k]       = Sf1
                Sf2_f[i,k]       = Sf2
                Sf3_f[i,k]       = Sf3
                Sf4_f[i,k]       = Sf4
                Sf5_f[i,k]       = Sf5
                Sf6_f[i,k]       = Sf6

    return xvalues, kurt, counts, Sf1_f, Sf2_f, Sf3_f, Sf4_f, Sf5_f, Sf6_f





def K41_linear_scaling(max_qorder):
    f              = lambda x: x/3
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)


def IK_linear_scaling(max_qorder):
    f              = lambda x: x/4
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)

def Chandran_scaling(max_qorder):
    f              = lambda x: (1-(0.691)**(x))
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)

def HB_K41_scaling(max_qorder):
    f              = lambda x: x/9+1-(1/3)**(x/3)
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)

def GPP_IK_scaling(max_qorder):
    f              = lambda x: x/8+1-(1/2)**(x/4)
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)

def MS17_perp(max_qorder):
    f              = lambda x: 1-(1/np.sqrt(2))**x
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)

def MS17_flucs(max_qorder):
    f              = lambda n: n*(1-1/2**(n/2))/(n/2 + 1 - 1/2**(n/2))
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)

def MS17_par(max_qorder):
    f              = lambda n: 2*(1- 1/2**(n/2))
    xvals          = np.arange(1, max_qorder+1,1)
    return xvals, f(xvals)


