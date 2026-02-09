###########################################################################
#                                                                         #
#    Copyright 2024 Nikos Sioulas                                         #
#    UCLA                                                                 #
#                                                                         #
#    nsioulas@g.uca.edu                                                   #
#                                                                         #
#    This file is part of MHDTurbPy toolbox.                              #
#                                                                         #
#    MHDTurbPy toolbox is free software: you can redistribute it          #H
#    and/or modify it under the terms of the GNU General Public           #
#    License as published by the Free Software Foundation, either         #
#    version 3 of the License, or (at your option) any later version.     #
#                                                                         #
#    MHDTurbPy toolbox is distributed in the hope that it will be         #
#    useful, but WITHOUT ANY WARRANTY; without even the implied warranty  #
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.              #
###########################################################################


# Basic librariesCH

import pandas 
import numpy as np
import sys


# Scipy
import scipy
from scipy import signal
from scipy.linalg import solve
from scipy import constants
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq

# Locate files
import os
from pathlib import Path
from glob import glob

# Wavelets
import ssqueezepy
import pycwt
import pywt

# parallelize functions
import numba
from joblib import Parallel, delayed
from numba import jit, njit, prange, objmode

# others
import time
import random

# Print errors
import traceback
from distutils.log import warn

#Import custom functions
from general_functions import *
from three_D_funcs import *

sys.path.insert(1, os.path.join(os.getcwd(), 'functions/modwt/wmtsa'))
import  modwt

import astropy.units as u
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

from psd_estimators import (
    FFTTracePSD,
    HaarWaveletPSD,
    MODWTTracePSD,
    PyCWTWaveletPSD,
    SSqueezepyWaveletPSD,
    get_psd_estimator,
)




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
    B_np               = B.to_numpy()
    num_rows, num_cols = B_np.shape
    result = np.zeros_like(B_np)

    for coef, lag in zip(coefs, lag_coefs):
        if lag == 0:
            result += coef * B_np
        else:
            shifted_B = np.roll(B_np, lag, axis=0)
            if lag > 0:
                shifted_B[:lag, :] = np.nan
            else:
                shifted_B[lag:, :] = np.nan
            result += coef * shifted_B

    if return_df:
        return pd.DataFrame(result, index=B.index, columns=B.columns)
    else:
        return result
    
    


def flucts(tau,
           B,
           five_points_sfunc   = True,
           return_dataframe    = False,
           estimate_mod_flucts = False,):
    """
    Calculate increments for structure functions.

    Args:
        tau (int): Time lag.
        B (pd.Series or np.ndarray): Input field.
        five_points_sfunc (bool, optional): Estimate 5-point structure functions if True. Defaults to True.

    Returns:
        dB (np.ndarray): Increments of the input field.
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
                B_values = B.values if isinstance(B, pd.DataFrame) else B
                dB = (B_values[:-tau] - B_values[tau:])
                dB_filled = np.full(B_values.shape, np.nan)
                dB_filled[:-tau, :] = dB
                dB = dB_filled


    return dB



def structure_functions_parallel(B,
                                 scales,
                                 max_qorder, 
                                 five_points_sfunc = False, 
                                 keep_sdk          = False,
                                 return_components = False,
                                 return_Bmod       = False, 
                                 return_compress   = False,
                                 return_flucts     = False,
                                 n_jobs            = -1):
    """
    Estimate the structure functions of a vector field B in parallel.

    Args:
        B (pd.Series or np.ndarray):       Input field (shape (N,) if 1D or (N,3) if 3D).
        scales (list or np.ndarray):       Scales (lags) at which to calculate the structure functions.
        max_qorder (int):                 Maximum order of the structure functions to be calculated.
        five_points_sfunc (bool):         Whether to estimate 5-point increments instead of 2-point.
        keep_sdk (bool):                  (Currently unused) Option to store or skip certain diagnostics.
        return_components (bool):         If True, also return separate components of the SF.
        return_Bmod (bool):               If True, also compute magnitude increments dBmod and return the 
                                          corresponding structure functions in parallel to the “trace.”
        return_compress (bool):           If True, also compute a “compressibility” measure from the fluctuations.
        return_flucts (bool):             If True, return the raw increments dB and dBmod for each scale, 
                                          rather than the structure functions.
        n_jobs (int):                     Number of parallel jobs. Defaults to -1 (all cores).

    Returns:
        If return_flucts is True:
            dB_all_scales   (np.ndarray): shape (len(scales), ...) of dB increments
            dBmod_all_scales(np.ndarray): shape (len(scales), ...) of |dB| increments (if return_Bmod=True)

        Else if return_components is True:
            sfn     (np.ndarray): shape (len(scales), max_qorder) of the trace SF
            sdk     (np.ndarray): shape (len(scales),)  normalizing factor from 4th order (if max_qorder>=4)
            sfn_cmp (np.ndarray): shape (len(scales), max_qorder, n_components) of each component's SF
            SF_dBmod(np.ndarray): shape (len(scales), max_qorder) of the modulus SF (if return_Bmod=True)
            compress(np.ndarray): shape (len(scales),) compressibility measure (if return_compress=True)
            counts  (np.ndarray): shape (len(scales),) number of non‐NaN points in dBmod

        Else:
            sfn (np.ndarray): shape (len(scales), max_qorder)
            sdk (np.ndarray): shape (len(scales),)
    """

    # Define the qorders
    qorders = np.arange(1, max_qorder + 1)

    # -------------------------------------------------------------------------
    # A small helper that calculates the SF at a given qorder for dB and dBmod
    def calc_sfn(dB, dBmod, qorder, return_components=False, return_Bmod=False):
        """
        Computes the structure function of order qorder:
            SF(dB)   = mean( |dB|^qorder ) across the chosen dimension,
            SF(dBmod)= mean( |dBmod|^qorder ) if return_Bmod is True.

        Args:
            dB (np.ndarray): shape (N, 3) if 3D, or (N,) if 1D.
            dBmod (np.ndarray or float): shape (N,) if returning magnitude, else np.nan.
            qorder (int)
            return_components (bool): if True, also return the separate comp SF.
            return_Bmod (bool): if True, also compute and return SF(dBmod).

        Returns:
            If return_components == False:
                (sfn_sum, sfn_mod)
            Else:
                (sfn_sum, comps_array, sfn_mod)
            where
                sfn_sum   = sum of component-wise means of |dB|^qorder
                comps_arr = mean of each component in |dB|^qorder if returning comps
                sfn_mod   = mean of |dBmod|^qorder, or np.nan if not return_Bmod
        """
        # Mean of each component^qorder
        comps = np.nanmean(dB ** qorder, axis=0)  # shape (#components,)
        # If returning the magnitude's SF
        if return_Bmod and isinstance(dBmod, np.ndarray):
            SF_dBmod = np.nanmean(dBmod ** qorder)
        else:
            SF_dBmod = np.nan

        if return_components:
            # return (trace, [comp1, comp2, comp3, ...], magnitude)
            return np.sum(comps), comps, SF_dBmod
        else:
            # return (trace, magnitude)
            return np.sum(comps), SF_dBmod


    # -------------------------------------------------------------------------
    # The actual worker for each scale
    def process_scale(tau,
                      return_components=False,
                      return_Bmod=False,
                      return_compress=False,
                      return_flucts=False):
        """
        Computes increments dB, dBmod (if needed), and from there either:
         - returns them directly if return_flucts=True,
         - or computes the SF across qorders.
        """
        # -- First, get the fluctuations
        dB = np.abs(flucts(tau, B, five_points_sfunc=five_points_sfunc))
        # shape of dB is typically (N,3) for a vector B

        compress = np.nan
        # If we need the magnitude increments:
        if return_Bmod:
            dBmod = np.abs(
                flucts(tau, B,
                       five_points_sfunc=five_points_sfunc,
                       estimate_mod_flucts=return_Bmod)
            )
            # If return_flucts is True, we *only* return the raw increments
            if return_flucts:
                # Force no compress if returning raw increments
                return_compress = False
            if return_compress:
                # Example compressibility measure
                #   compress = mean(|delta B_parallel|^2 / (|delta B|^2))
                #   but code below does something like dBmod.T[0] ...
                #   If we truly want parallel component, define it carefully.
                #   For demonstration let's just do a ratio:
                #       compress = mean( (dBmod[:,0])^2 / sum of squares of dB )
                #   *But watch shape carefully. If dBmod is a single column
                #   (the magnitude), we can't do dBmod[:,0].
                #   Possibly the user meant the projection of dB along something.
                #   We'll keep the line but ensure shapes are correct:
                #
                #   compress = mean( dBmod[:,0]^2 / (dB[:,0]^2 + dB[:,1]^2 + dB[:,2]^2 ) )
                #   or if "dBmod" is just a single column, you might do dBmod**2 / sum(dB**2).
                #
                if dBmod.ndim == 2 and dBmod.shape[1] == 3:
                    # Then we can do .T[0] etc. if that's the parallel part
                    compress = np.nanmean(
                        np.abs(dBmod[:, 0])**2 /
                        (dB[:, 0]**2 + dB[:, 1]**2 + dB[:, 2]**2)
                    )
                else:
                    # If "dBmod" is purely the magnitude:
                    #   compress doesn't have a straightforward meaning here
                    compress = np.nan
        else:
            dBmod = np.nan

        # If the user only wants the raw increments:
        if return_flucts:
            return dB, dBmod

        # Otherwise, compute structure functions over qorders
        if return_components:
            # We want: sfn, sfn_comps, SF_dBmod, plus sdk and possibly compress
            # We'll gather (trace, comps, mod) for each qorder
            tmp = [calc_sfn(dB, dBmod, q, 
                            return_components=True, 
                            return_Bmod=return_Bmod)
                   for q in qorders]
            # tmp is list of length max_qorder, each element is ( trace_val, comps_vec, mod_val )
            trace_vals, comps_list, mod_vals = zip(*tmp)  # each is length max_qorder
            sfn       = np.array(trace_vals)             # shape (max_qorder,)
            sfn_comps = np.array(comps_list)             # shape (max_qorder, n_components?)
            SF_dBmod  = np.array(mod_vals)               # shape (max_qorder,)

            # Compute sdk if we have at least q=4
            if max_qorder >= 4:
                # The code uses the 4th order / sum of squares of the 2nd moment, etc.
                # We'll match the original usage:
                #  sdk = sfn[3] / np.sum(np.nanmean(dB ** 2, axis=0) ** 2)
                sdk = sfn[3] / np.sum(np.nanmean(dB**2, axis=0)**2)
            else:
                sdk = np.nan

            counts = np.count_nonzero(~np.isnan(dBmod)) if isinstance(dBmod, np.ndarray) else 0

            return sfn, sdk, sfn_comps, SF_dBmod, compress, counts

        else:
            # return_components=False -> simpler: just return trace + sdk
            tmp = [calc_sfn(dB, dBmod, q, 
                            return_components=False, 
                            return_Bmod=return_Bmod)
                   for q in qorders]
            # tmp is list of length max_qorder, each (trace, mod)
            trace_vals, mod_vals = zip(*tmp)   # each is length max_qorder
            sfn      = np.array(trace_vals)    # shape (max_qorder,)
            SF_dBmod = np.array(mod_vals)      # shape (max_qorder,) but not used here

            if max_qorder >= 4:
                sdk = sfn[3] / np.sum(np.nanmean(dB**2, axis=0)**2)
            else:
                sdk = np.nan

            return sfn, sdk

    # -------------------------------------------------------------------------
    # Now run the above worker in parallel over each scale
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_scale)(tau,
                               return_components=return_components,
                               return_Bmod=return_Bmod,
                               return_compress=return_compress,
                               return_flucts=return_flucts)
        for tau in scales
    )

    # Finally, assemble outputs
    if return_flucts:
        # results is a list of (dB, dBmod) for each scale
        dB_all, dBmod_all = zip(*results)
        return np.array(dB_all), np.array(dBmod_all)

    else:
        if return_components:
            # results is a list of 6-tuples: (sfn, sdk, sfn_comps, SF_dBmod, compress, counts)
            sfn, sdk, sfn_comps, SF_dBmod, compress, counts = zip(*results)
            return (np.array(sfn),
                    np.array(sdk),
                    np.array(sfn_comps),
                    np.array(SF_dBmod),
                    np.array(compress),
                    np.array(counts))
        else:
            # results is a list of 2-tuples: (sfn, sdk)
            sfn, sdk = zip(*results)
            return np.array(sfn), np.array(sdk)



def est_5_pt_sfuncs(B_df,
                    dt,
                    func_params = None):

    max_hours   = round(len(B_df)*dt/60)+1
    dt_step     = func_params['dt_step']
    max_lag     = int((max_hours*3600)/dt)
    tau_values  = 2**np.arange(0, 1000, dt_step)
    max_ind     = (tau_values<max_lag) & (tau_values>0)
    lags        = np.unique(tau_values[max_ind].astype(int))

    # estimate sfuncs
    res = turb.structure_functions_parallel(B_df, 
                                            lags,
                                            func_params['max_qorder'], 
                                            five_points_sfunc  = func_params['five_points_sfunc'],
                                            return_Bmod        = func_params['return_Bmod'],
                                            return_compress    = 0,
                                            return_flucts      = False,
                                            return_components  = 1,
                                            n_jobs             =-1)
    
    # Assign results
    sfn, sdk, sfn_comps, SF_dBmod, compress, counts = res
            
    return {'dt'       : dt,
            'lags'     : lags, 
            'counts'   : counts,
            
            'SF_trace' : sfn.T,
            'SF_mod'   : SF_dBmod.T,
            
            'SDK_vec'  : sfn.T[3]/sfn.T[1]**2,
            'SDK_mod'  : SF_dBmod.T[3]/SF_dBmod.T[1]**2}




        
def MODWT_wave_coeffs(x, wname ='la20'):
    
    return modwt.modwt(x, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    
    
def estimate_coeffs_background_flucs(x, wname ='la20'):

    # Estimate length of timeseries
    sample_length = len(x)
    
    # Estimate MODWT coefficients and weights
    Wj, Vj   = modwt.modwt(x, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    
    # Perform forwards multiresolution analysis obtain 
    # fluctuations (details) and background (approximations) at each level
    Det, Appr  = modwt.imodwt_mra(Wj, Vj)
    
    # It returns a timeseries with length 2x sample_length
    Det, Appr  = Det[:, 0: sample_length],  Appr[ 0: sample_length]
    
    # Reconstruct the approximations at each level using the details
    Approx  = []
    for i in range(len(Det)):
        if i==0:
            Approx.append(Appr)
        else:
            Approx.append(Approx[i-1] + Det[i-1])
    
    # Remove the phase shift in the detail coefficients at each levels 
    Swd, Vjd       = modwt.cir_shift(Wj, Vj, subtract_mean_VJ0t=True)

 
    return Approx, Det, Swd


# def Trace_PSD_MODWT(R, T, N, dt, wname='la8'):
    
    
#     # Function to compute MODWT coefficients
#     def compute_modwt(data):
#         W, Vj = modwt.modwt(data, wtf=wname, nlevels='conservative',
#                             boundary='reflection', RetainVJ=True)
#         return W

#     # Function to compute PSD
#     def compute_psd(W):
#         PSD = modwt.wspec(W, dt)
#         return PSD[0]

#     # Parallel computation of MODWT coefficients
#     modwt_results = Parallel(n_jobs=3)(
#         delayed(compute_modwt)(data) for data in [R, T, N]
#     )
#     Wr, Wt, Wn = modwt_results

#     # Return freqs and scales
#     scale = 2 ** np.arange(1, Wr.shape[0] + 1)
#     freqs = pywt.scale2frequency(wname, scale) / dt

#     # Parallel computation of PSDs

#     psd_results = Parallel(n_jobs=3)(
#         delayed(compute_psd)(W) for W in [Wr, Wt, Wn])
    
#     PSD_R, PSD_T, PSD_N = psd_results

#     # Calculate total PSD
#     total_PSD = 2 * (PSD_R + PSD_T + PSD_N)

#     return freqs, total_PSD, scale



        
def Trace_PSD_MODWT(R, T, N, dt, wname ='sym8'):
    
#     if (dt< 0.3):
        
#         wname ='d20'
#         print(f'Cadence is high.Switch  wname ={wname} to capture steep scalings!')
        
    
    estimator = MODWTTracePSD(wname=wname)
    return estimator.estimate(R, T, N, dt)


def Trace_haar_wavelet_psd(x, y, z, dt, wavelet='haar'):
    estimator = HaarWaveletPSD(wavelet=wavelet)
    return estimator.estimate(x, y, z, dt)


def trace_PSD_wavelet(x,
                      y,
                      z, 
                      dt, 
                      dj         =1/2,
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
    
    estimator = PyCWTWaveletPSD(dj=dj, mother_wave=mother_wave)
    return estimator.estimate(x, y, z, dt)







def trace_PSD_cwt_ssqueezepy(x, 
                             y,
                             z, 
                             dt,
                             nv            = 16,
                             scales_type   = 'log',
                             wavelet       = None,
                             wname         = None,
                             l1_norm       = False,
                             est_PSD       = True,
                             est_mod       = False,
                             omega0        = 6.0):
    """
    Method to calculate the wavelet coefficients and  power spectral density using the Morlet wavelet method.
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
    
    estimator = SSqueezepyWaveletPSD(
        nv=nv,
        scales_type=scales_type,
        wavelet=wavelet,
        wname=wname,
        l1_norm=l1_norm,
        est_psd=est_PSD,
        est_mod=est_mod,
        omega0=omega0,
    )
    return estimator.estimate(x, y, z, dt)



def local_gaussian_averaging(signal, scale, dt, num_efoldings=3,alpha=1):
    
    # Effective width of the Gaussian
    sigma_b          = alpha * scale
    
    # Calculate window size to include the desired number of e-foldings of the Gaussian distribution
    sigma_b_samples  = sigma_b / dt
    N                = int(np.ceil(num_efoldings * sigma_b_samples))
    N                = max(1, N)
    t_samples        = np.arange(-N, N + 1)
    
    # Define the Gaussian window
    gaussian_kernel  = np.exp(- (t_samples ** 2) / (2 * sigma_b_samples ** 2))
    
    # Normalize the Gaussian window to ensure it sums to one, maintaining the total signal energy after convolution
    gaussian_kernel /= gaussian_kernel.sum()
    
    # Convolve the input signal with the Gaussian window using 'same' mode 
    return scipy.signal.convolve(signal, gaussian_kernel, mode='same')



def coherence_analysis(B0_f_o, df_w):
    def get_intensity(df_):
        return np.sqrt(np.square(df_).sum(axis=1))

    def eigen_outer_freq(ser):
        M = np.array([[ser['RRe'], ser['RTe'], ser['RNe']],
                      [ser['RTe'], ser['TTe'], ser['TNe']],
                      [ser['RNe'], ser['TNe'], ser['NNe']]])
        Eig_value, Eig_vector = np.linalg.eigh(M)
        return Eig_vector[:, 0]

    # Calculate matrix elements
    B0_f_o['RRe'] = B0_f_o['RR'] - np.square(B0_f_o['R'])
    B0_f_o['TTe'] = B0_f_o['TT'] - np.square(B0_f_o['T'])
    B0_f_o['NNe'] = B0_f_o['NN'] - np.square(B0_f_o['N'])
    B0_f_o['RTe'] = B0_f_o['RT'] - B0_f_o['R'] * B0_f_o['T']
    B0_f_o['RNe'] = B0_f_o['RN'] - B0_f_o['R'] * B0_f_o['N']
    B0_f_o['TNe'] = B0_f_o['TN'] - B0_f_o['T'] * B0_f_o['N']

    # Find eigenvectors
    B0_f_o[['eigen_1', 'eigen_2', 'eigen_3']] = np.vstack(B0_f_o.apply(eigen_outer_freq, axis=1))

    # Compute unit vectors
    intensity = get_intensity(B0_f_o[['R', 'T', 'N']])
    B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']] = B0_f_o[['R', 'T', 'N']].div(intensity, axis='index')
    
    # Calculate and normalize perpendicular vectors
    B0_f_o[['B_1_R', 'B_1_T', 'B_1_N']] = np.cross(B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']], 
                                                   B0_f_o[['eigen_1', 'eigen_2', 'eigen_3']])
    
    intensity_1 = get_intensity(B0_f_o[['B_1_R', 'B_1_T', 'B_1_N']])
    B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']] = B0_f_o[['B_1_R', 'B_1_T', 'B_1_N']].div(intensity_1, axis='index')
    
    # Calculate second perpendicular unit vector
    B0_f_o[['B_2_R_hat', 'B_2_T_hat', 'B_2_N_hat']] = np.cross(B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']], 
                                                              B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']])
    
    # Memory cleanup
    B0_f_o.drop(columns=['B_1_R', 'B_1_T', 'B_1_N', 'RR', 'TT', 'NN', 'RT', 'RN', 'TN', 'RRe', 'TTe', 'NNe', 'RTe', 'RNe', 'TNe'], inplace=True)

    # Calculate magnetic field components in the wave tensor
    df_w['MWT_0'] = (B0_f_o[['B_0_R_hat', 'B_0_T_hat', 'B_0_N_hat']].values * df_w[['R', 'T', 'N']].values).sum(axis=1)
    df_w['MWT_1'] = (B0_f_o[['B_1_R_hat', 'B_1_T_hat', 'B_1_N_hat']].values * df_w[['R', 'T', 'N']].values).sum(axis=1)
    df_w['MWT_2'] = (B0_f_o[['B_2_R_hat', 'B_2_T_hat', 'B_2_N_hat']].values * df_w[['R', 'T', 'N']].values).sum(axis=1)

    return B0_f_o['R'].values, B0_f_o['T'].values, B0_f_o['N'].values, -2 * (np.imag( df_w['MWT_1']*np.conj(df_w['MWT_2'])) ) , (np.abs(df_w['MWT_1'])**2+np.abs(df_w['MWT_2'])**2)


def anisotropy_coherence(
                           B_df,
                           V_df, 
                           dt,  
                           nv                    = 32,
                           alpha                 = 1, 
                           per_thresh            = 80,
                           par_thresh            = 10,
                           njobs                 = -1,
                           est_mod               = True,
                           estimate_local_V      = False,
                           do_coherence_analysis = False
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
    
    def estimate_cwt(signal, dt, nv=32, omega0=6):
        fs         = 1 / dt
        wavelet    = ssqueezepy.Wavelet(('morlet', {'mu': omega0}))
        W, scales  = ssqueezepy.cwt(signal, wavelet=wavelet, scales='log', fs=fs, nv=nv)
        freqs      = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal), fs)
        scales     = (omega0)/(2*np.pi*freqs)*(1  + 1/(2*omega0**2))*fs
        coi        = None
        return W, scales, freqs, coi


    

    def define_B_df(B_index, Br, Bt, Bn):
        return      pd.DataFrame({ 'DateTime' : B_index,
                                    'R'       : Br,
                                    'T'       : Bt,
                                    'N'       : Bn,
                                    'RR'      : Br **2.,
                                    'TT'      : Bt **2.,
                                    'NN'      : Bn **2.,
                                    'RT'      : Br * Bt,
                                    'RN'      : Br * Bn,
                                    'TN'      : Bt * Bn}).set_index('DateTime')

    def define_W_df(B_index, R, T, N):
        return      pd.DataFrame({ 'DateTime' : B_index,
                                    'R'       : R,
                                    'T'       : T,
                                    'N'       : N}).set_index('DateTime')

    def parallel_oper(ii, 
                      scale,
                      dt,
                      time_index,
                      w_R, w_T, w_N,
                      Br, Bt, Bn,
                      Vr, Vt, Vn, 
                      mag_b, mag_v,
                      db_x, db_y, db_z, 
                      db_mod,
                      alpha,
                      per_thresh,
                      par_thresh,
                      njobs                 = -1,
                      est_mod               = False,
                      estimate_local_V      = False,
                      do_coherence_analysis = False):
        try:
            
            if do_coherence_analysis:
                B_df = define_B_df(time_index, Br, Bt, Bn)
                df_w = define_W_df(B_index, w_R, w_T, w_N)
                
                del Br, Bt, Bn, time_index

                # Apply the function to each column
                B_df_smoothed = B_df.apply(lambda col: local_gaussian_averaging(col.values, scale, dt, alpha), axis=0)
                
     
                # Do coherence analysis
                Br_0, Bt_0, Bn_0, num_coh, den_coh     = coherence_analysis(B_df_smoothed, df_w)
                
                
            else:

                Br_0    = local_gaussian_averaging(Br, scale, dt, alpha=alpha)
                Bt_0    = local_gaussian_averaging(Bt, scale, dt, alpha=alpha)
                Bn_0    = local_gaussian_averaging(Bn, scale, dt, alpha=alpha)

                if estimate_local_V:
                    Vr      = local_gaussian_averaging(Vr, scale, dt, alpha=alpha)
                    Vt      = local_gaussian_averaging(Vt, scale, dt, alpha=alpha)
                    Vn      = local_gaussian_averaging(Vn, scale, dt, alpha=alpha)

                mag_V_0 = np.sqrt( Vr**2  +  Vt**2 +  Vn**2  )
                num_coh, den_coh = np.nan, np.nan

          
            mag_b_0 = np.sqrt( Br_0**2  +  Bt_0**2 +  Bn_0**2  )
            
            VBangles                = np.arccos((Vr * Br_0 + Vt * Bt_0 + Vn * Bn_0) / (mag_b_0 * mag_v)) * 180 / np.pi
            VBangles[VBangles > 90] = 180 - VBangles[VBangles > 90]

            index_per   = (np.where(VBangles > per_thresh)[0]).astype(np.int64)
            index_par   = (np.where(VBangles < par_thresh)[0]).astype(np.int64)

            PSD_par_val = (np.nanmean(np.abs(np.array(db_x[ii])[index_par])**2) + 
                          np.nanmean(np.abs(np.array(db_y[ii])[index_par])**2) + 
                          np.nanmean(np.abs(np.array(db_z[ii])[index_par])**2) ) * ( 2* dt)

            PSD_per_val = (np.nanmean(np.abs(np.array(db_x[ii])[index_per])**2) + 
                          np.nanmean(np.abs(np.array(db_y[ii])[index_per])**2) + 
                          np.nanmean(np.abs(np.array(db_z[ii])[index_per])**2) ) * ( 2* dt)
            
            if est_mod:
                PSD_par_mod_val = (np.nanmean(np.abs(np.array(db_mod[ii])[index_par])**2) ) * ( 2* dt)

                PSD_per_mod_val = (np.nanmean(np.abs(np.array(db_mod[ii])[index_per])**2) ) * ( 2* dt)
            else:
                PSD_par_mod_val = np.nan

                PSD_per_mod_val = np.nan             

            return PSD_par_val, PSD_per_val, PSD_par_mod_val, PSD_per_mod_val, num_coh, den_coh, VBangles
        except Exception as e:
            traceback.print_exc()
            return np.nan, np.nan

    # Estimate sampling time of timeseries
    dt_B = func.find_cadence(B_df)
    dt_V = func.find_cadence(V_df)

    if dt_V != dt_B:
        V_df = func.newindex(V_df, B_df.index, interp_method='linear')
        print(len(V_df.Vr.values))
    # Common dt
    dt = dt_B

    # Turn columns of df into arrays
    Br, Bt, Bn = B_df.Br.values, B_df.Bt.values, B_df.Bn.values
    Vr, Vt, Vn = V_df.Vr.values, V_df.Vt.values, V_df.Vn.values
    B_index    =  B_df.index.values
    del B_df, V_df
    
    # Estimate magnitude of magnetic field
    mag_b = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    # Estimate the magnitude of V vector
    mag_v = np.sqrt(Vr ** 2 + Vt ** 2 + Vn ** 2)

    # Estimate wavelet coefficients
    Wr, scales, freqs, coi       = estimate_cwt(Br, dt, nv = nv)
    Wt, scales, freqs, coi       = estimate_cwt(Bt, dt, nv = nv)
    Wn, scales, freqs, coi       = estimate_cwt(Bn, dt, nv = nv)
    

    if est_mod:

        Wmod, scales, freqs, coi = estimate_cwt(mag_b, dt, nv = nv)
    else:
        Wmod = np.nan
        
    
    PSD_par = np.zeros(len(freqs))
    PSD_per = np.zeros(len(freqs)) 
 
    PSD_par_mod = np.zeros(len(freqs))
    PSD_per_mod = np.zeros(len(freqs))

    # Use joblib for parallel processing
    results = Parallel(n_jobs=njobs)(delayed(parallel_oper)(
                                                              ii, 
                                                              scale,
                                                              dt,
                                                              B_index,
                                                              Wr[ii], Wt[ii], Wn[ii],
                                                              Br, Bt, Bn,
                                                              Vr, Vt, Vn, 
                                                              mag_b, mag_v,
                                                              Wr, Wt, Wn, 
                                                              Wmod,
                                                              alpha,
                                                              per_thresh,
                                                              par_thresh,
                                                              njobs                 = njobs,
                                                              est_mod               = est_mod,
                                                              estimate_local_V      = estimate_local_V,
                                                              do_coherence_analysis = do_coherence_analysis
    ) for ii, scale in enumerate(scales))
 
    
    # Unpack results
    PSD_par, PSD_per, PSD_par_mod, PSD_per_mod,  num_coh, den_coh, VBangles = zip(*results)

    return freqs, PSD_par, PSD_per, PSD_par_mod, PSD_per_mod, scales,  Wr, Wt, Wn, num_coh, den_coh, VBangles  





def estimate_polarization(num_coh, den_coh, scales, dt, alpha=1, num_efoldings=1, n_jobs=-1):
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
    
    def local_gaussian_average(value, scale, dt, alpha, num_efoldings):
    # Assuming turb.local_gaussian_averaging is defined elsewhere and available here
        return turb.local_gaussian_averaging(value, scale, dt, alpha=alpha, num_efoldings=num_efoldings)

    def compute_ratio(i, num_coh, den_coh, scales, dt, alpha, num_efoldings):
        num_value = local_gaussian_average(num_coh[i], scales[i], dt, alpha, num_efoldings)
        den_value = local_gaussian_average(den_coh[i], scales[i], dt, alpha, num_efoldings)
        return num_value / den_value

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_ratio)(i, num_coh, den_coh, scales, dt, alpha, num_efoldings)
        for i in range(len(num_coh))
    )
    return results




def calculate_coherence_PSDs(num_coh, sigma, Wr, Wt, Wn, dt, coh_th=0.7):
    """
    Calculate coherent and non-coherent sums for wave components.

    Parameters:
    - num_coh: List or array of indices or lengths corresponding to the number of elements to process.
    - sigma: List of arrays containing the sigma values for threshold comparison.
    - Wr, Wt, Wn: Lists of arrays representing different wave components (real, tangential, normal).
    - dt: Time step or similar scale factor for the calculations.
    - coh_th: Threshold value for determining coherence (default is 0.7).

    Returns:
    - coh: List of calculated coherent sums.
    - non_coh: List of calculated non-coherent sums.
    """

    # Lists to store coherent and non-coherent values
    coh = []
    non_coh = []

    # Iterate through each index in num_coh
    for i in range(len(num_coh)):
        # Boolean indices for coherent and non-coherent conditions based on the threshold
        index_coh = np.abs(sigma[i]) > coh_th
        index_non_coh = ~index_coh  # Logical negation of index_coh

        # Calculate the coherent component sum
        coherent_sum = (np.nanmean(Wr[i][index_coh] * np.conj(Wr[i][index_coh])) +
                        np.nanmean(Wt[i][index_coh] * np.conj(Wt[i][index_coh])) +
                        np.nanmean(Wn[i][index_coh] * np.conj(Wn[i][index_coh])))

        # Calculate the non-coherent component sum
        non_coherent_sum = (np.nanmean(Wr[i][index_non_coh] * np.conj(Wr[i][index_non_coh])) +
                            np.nanmean(Wt[i][index_non_coh] * np.conj(Wt[i][index_non_coh])) +
                            np.nanmean(Wn[i][index_non_coh] * np.conj(Wn[i][index_non_coh])))

        # Append computed values to coh and non_coh lists
        coh.append(2 * np.sum(index_coh) / len(index_coh) * dt * coherent_sum)
        non_coh.append(2 * np.sum(index_non_coh) / len(index_coh) * dt * non_coherent_sum)
        
    overall_psd = np.real(np.nansum([np.array(coh), np.array(non_coh)], axis=0))

    return coh, non_coh, overall_psd


def choose_dates_heatmap(freqs, inds,  data, original, target):
    fe = []
    dt = []
    increment = original // target
    for i in range (0, len(freqs), increment):
        fe.append(freqs[i])
        dt.append(data[i][inds[0]: inds[1]])
    return np.array(fe), np.array(dt)

def TracePSD(x, y, z, dt, 
             remove_mean       = False,
             return_components = False,
             return_mod        = False):
    """
    Estimate Fourier Power Spectral Density (PSD) for a trace composed of three orthogonal components.

    Parameters:
        x, y, z (np.ndarray or pandas.Series): Timeseries data for the three components.
        dt (float): Time step (1/sampling frequency).
        remove_mean (bool, optional): If True, remove the mean from the input timeseries. Default is False.
        return_components (bool, optional): If True, return the PSD for individual components. Default is False.
        return_mod (bool, optional): If True, return the modulus PSD. Default is False.

    Returns:
        tuple: Depending on `return_components` and `return_mod`, a tuple containing:
               - freqs (np.ndarray): Array of frequencies.
               - B_pow (np.ndarray): Power spectral density estimates of the trace or individual components and/or modulus.
    """
    estimator = FFTTracePSD(
        remove_mean=remove_mean,
        return_components=return_components,
        return_mod=return_mod,
    )
    return estimator.estimate(x, y, z, dt)


def estimate_trace_psd(
    x,
    y,
    z,
    dt,
    method="traceFFT",
    **kwargs,
):
    """
    Dispatch trace PSD estimation across supported methods.

    Parameters
    ----------
    x, y, z : array-like
        Components of the field.
    dt : float
        Sampling time step.
    method : str, optional
        PSD estimation method. Default is "traceFFT".
        Supported: "traceFFT" (alias for "fft"), "fft", "modwt", "haar", "pycwt", "ssqueezepy".
    **kwargs
        Method-specific keyword arguments forwarded to the underlying estimator
        constructor (not its ``estimate`` method).

    Returns
    -------
    tuple
        Method-specific outputs from the selected PSD estimator.
    """

    method_key = method.lower()
    aliases = {
        "tracefft": "fft",
    }
    estimator_key = aliases.get(method_key, method_key)
    estimator_kwargs = dict(kwargs)
    if estimator_key == "ssqueezepy":
        if "est_PSD" in estimator_kwargs and "est_psd" not in estimator_kwargs:
            estimator_kwargs["est_psd"] = estimator_kwargs.pop("est_PSD")
        else:
            estimator_kwargs.pop("est_PSD", None)
    estimator = get_psd_estimator(estimator_key, **estimator_kwargs)
    return estimator.estimate(x, y, z, dt)

def Trace_psd_Hann(B,  dt, nperseg=2**14, noverlap=2**13):
    from scipy.signal import welch

    keys = list(B.keys())
    
    x    = B[keys[0]].values
    y    = B[keys[1]].values
    z    = B[keys[2]].values
    
    N  = len(x)
    fs = 1/dt
    
    f, Px = welch(x, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    f, Py = welch(y, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    f, Pz = welch(z, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    return f, (Px + Py +Pz)/(N*fs)


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




@jit( parallel =True, nopython=True)
def structure_functions_wavelets_per_par(db_x, db_y, db_z, angles,  scales, dt, max_moment, per_thresh, par_thresh):
    
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



@jit( parallel =True, nopython=True)
def structure_functions_wavelets(db_x, db_y, db_z,   scales, dt, max_moment):
    
    tau = scales*dt
    m_vals = np.arange(1, max_moment+1)

    
    sfunc  = np.zeros((len(tau), len(m_vals))) 
    counts = np.zeros((len(tau), len(m_vals))) 


    for j in prange(len(tau)):
        
        dbtot     = (db_x[j]*np.conjugate(db_x[j]) + db_y[j]*np.conjugate(db_y[j])  +db_z[j]*np.conjugate(db_z[j]) )**(1/2)

        for m in prange(len( m_vals)):
            
            #sfunc[j, m]  = np.nanmean(np.abs(dbtot/np.sqrt(tau[j]))**m_vals[m])
            sfunc[j, m]  = np.nanmean(np.abs(dbtot)**m_vals[m])
         
            counts[j, m] = len(dbtot)#.astype('float')
    return scales, sfunc,counts


# -*- coding: utf-8 -*-
"""
Cr09_cascade_rate — end-to-end revision (binning decoupled, raw/fit product control, LaTeX)
==========================================================================================

This version addresses your three concrete requirements:

(1) Raw values in Q-terms regardless of derivative fitting mode
    ----------------------------------------------------------------
    • New argument `product_value_source` controls what VALUES are used in the
      product terms of Qp/Qe/Qe_qpar independently of how derivatives are fit.
        - "fit"       → products use fitted profiles
        - "raw"       → products use raw measurements (NaNs kept as NaNs)
        - "raw_fill"  → products use raw, but NaNs are replaced by fitted values
                         only for r < R_nan_fill_max (same behavior you wanted)
    • Backward compatibility: if `use_raw_in_products=True` is passed, we map it
      to `product_value_source="raw_fill"` unless you explicitly set the new arg.

(2) Binned averages are ALWAYS computed and returned
    -------------------------------------------------
    • Even when `use_binning=False` (i.e., fits are done on raw points), we *still*
      compute log-spaced bin averages and counts for every variable and return them
      in `fits[var]["avg_x"]`, `fits[var]["avg_y"]`, and `fits[var]["avg_n"]`.

(3) Proper LaTeX equations
    -----------------------
    • For polynomials: we emit both an inline ($…$) and a display equation using
      `\begin{equation} ... \end{equation}` with `x = \ln(r/\mathrm{AU})`.
    • For piecewise power-laws: we emit a clean cases block inside an
      `equation` environment:
          Y(r) = { A1 (r/AU)^{p1}, r<rb ; A2 (r/AU)^{p2}, r>=rb }
      with continuity enforced via A2 = A1 (rb/AU)^{p1-p2}.

Other guardrails & fixes retained from prior pass
-------------------------------------------------
• `allow_piecewise` is a hard switch: when False, no PWL search/eval is performed.
• Breakpoint caps: `min_break` / `max_break` (Quantities) intersect with the data
  support and robust quantile window to construct candidate `x_b = ln(r/AU)`.
• BIC includes variance-model parameters; hinge slope uses right-branch at the hinge.
"""



from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from astropy import constants as const
from astropy import units as u

# ------------------------------ constants -------------------------------------

OMEGA = 2.7e-6 / u.s     # solar rotation rate (rad/s)
NU_0  = 8.4e-9  / u.s    # Coulomb collision prefactor (1/s)

# ------------------------------ Parker angle ----------------------------------

def parker_spiral_angle(
    r: u.Quantity,
    U: u.Quantity,
    *,
    theta_deg: float = 0.0,
    r0: u.Quantity = 0.045 * u.au,
) -> u.Quantity:
    r"""
    Analytic Parker-spiral angle relative to radial direction.

    Parameters
    ----------
    r : Quantity [length]
    U : Quantity [speed]
    theta_deg : float, optional
        Colatitude in degrees used in the analytic expression.
    r0 : Quantity [length], optional
        Source surface.

    Returns
    -------
    phi : Quantity [rad]

    Notes
    -----
    \[
      \phi(r) \equiv -\arctan\!\left(
        \frac{\Omega\,(r-r_0)\,\sin\theta}{U}
      \right).
    \]
    """
    theta_rad = np.deg2rad(theta_deg)
    arg = (OMEGA * (r - r0) * np.sin(theta_rad) / U).to(u.dimensionless_unscaled)
    return -np.arctan(arg.value) * u.rad

# ------------------------------ math helpers ----------------------------------

def _poly_deriv(coef: np.ndarray, x_ln: np.ndarray) -> np.ndarray:
    r"""
    Derivative of a polynomial \(p(x)\) evaluated at \(x=\ln(r/\text{AU})\).

    Parameters
    ----------
    coef : ndarray
        np.polyval-style coefficients for \(p(x)\).
    x_ln : ndarray
        Locations where to evaluate.

    Returns
    -------
    deriv : ndarray
        \(p'(x)\).

    Notes
    -----
    If \(y=\ln Y\) is modeled by \(p(x)\), then
    \[
      \frac{d\ln Y}{d\ln r} = p'(x),\qquad x=\ln(r/\text{AU}).
    \]
    For a non-log variable \(z\) modeled directly by \(p(x)\),
    \[
      \frac{dz}{dr} = \frac{1}{r}\,p'(x).
    \]
    """
    return np.zeros_like(x_ln) if coef is None else np.polyval(np.polyder(coef), x_ln)

def _design(x_ln: np.ndarray, deg: int) -> np.ndarray:
    r"""
    Vandermonde design \(V_{ij}=x^{d-j}\) for polynomial degree `deg` in \(x=\ln(r/\text{AU})\).
    """
    return np.vander(x_ln, deg + 1)

# ------------------------------ LaTeX helpers ---------------------------------

def _latex_sci_number(x: float) -> str:
    """Format a float to LaTeX with ~3 sig figs, m×10^{e} for |e|>=3."""
    if not np.isfinite(x):
        return r"\mathrm{nan}"
    if x == 0.0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if abs(e) >= 3:
        m = x / (10.0**e)
        return f"{m:.3g}" + r"\times 10^{" + str(e) + "}"
    return f"{x:.3g}"

def _latex_poly_inline(coef: np.ndarray, var: str, expr: str, logy: bool) -> str:
    r"""
    Inline LaTeX for polynomial \(p(x)\) with \(x=\ln(r/\text{AU})\).
    Prints either \(\ln(\text{expr})=...\) or \(\text{expr}=...\) depending on `logy`.
    """
    if coef is None or len(coef) == 0:
        return r"$ $"
    d = len(coef) - 1
    terms = []
    for i, c in enumerate(coef):
        p = d - i
        mag = _latex_sci_number(abs(float(c)))
        if p > 1:
            mon = rf"{mag}\,{var}^{{{p}}}"
        elif p == 1:
            mon = rf"{mag}\,{var}"
        else:
            mon = rf"{mag}"
        s = (mon if (i == 0 and c >= 0) else ("+" + mon if c >= 0 else "-" + mon))
        terms.append(s)
    lhs = rf"\ln({expr})" if logy else expr
    return r"$" + lhs + "=" + " ".join(terms) + r"$"

def _latex_poly_equation(coef: np.ndarray, expr: str, logy: bool) -> str:
    r"""
    Display equation for polynomial \(p(x)\), \(x=\ln(r/\text{AU})\).
    """
    if coef is None or len(coef) == 0:
        return r"\begin{equation}\end{equation}"
    d = len(coef) - 1
    parts = []
    for i, c in enumerate(coef):
        p = d - i
        mag = _latex_sci_number(abs(float(c)))
        if p > 1:
            term = f"{mag}\\,x^{{{p}}}"
        elif p == 1:
            term = f"{mag}\\,x"
        else:
            term = f"{mag}"
        parts.append(("+" if (i > 0 and c >= 0) else "-" if (i > 0 and c < 0) else "") + term)
    rhs = " ".join(parts).lstrip("+")
    lhs = (r"\ln(" + expr + r")") if logy else expr
    return (
        r"\begin{equation}" "\n"
        + lhs + r" = " + rhs + r",\quad x=\ln\!\left(\frac{r}{\mathrm{AU}}\right)" "\n"
        r"\end{equation}"
    )

def _latex_piecewise_inline(b: float, k1: float, dk: float, xb: float, expr: str) -> str:
    r"""
    Inline LaTeX for 2-part power law in natural scale:

    \[
      Y(r)=
      \begin{cases}
        A_1 \left(\frac{r}{\text{AU}}\right)^{p_1}, & r < r_b\\
        A_2 \left(\frac{r}{\text{AU}}\right)^{p_2}, & r \ge r_b
      \end{cases}
    \]
    with \(p_2=p_1+\Delta k\) and continuity \(A_2=A_1 (r_b/\text{AU})^{p_1-p_2}\).
    """
    A1   = float(np.exp(b))
    p1   = float(k1)
    p2   = float(k1 + dk)
    rbAU = float(np.exp(xb))
    A2   = A1 * (rbAU ** (p1 - p2))
    return (
        r"$" + expr + r"(r)="
        + _latex_sci_number(A1) + r"\left(\frac{r}{\mathrm{AU}}\right)^{" + f"{p1:.3g}" + r"},\ r<"
        + _latex_sci_number(rbAU) + r"\,\mathrm{AU};\ "
        + _latex_sci_number(A2) + r"\left(\frac{r}{\mathrm{AU}}\right)^{" + f"{p2:.3g}" + r"},\ r\ge "
        + _latex_sci_number(rbAU) + r"\,\mathrm{AU}"
        + r"$"
    )

# ------------------------------ binning ---------------------------------------

def _bin_mean_and_count(
    x_AU: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    require_pos: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Log-spaced binning in \(x=r/\text{AU}\), arithmetic mean of \(y\), and counts.

    Parameters
    ----------
    x_AU : array-like
        Positive radii in AU units.
    y : array-like
        Values to average.
    bins : int
        Number of bin edges (→ bins-1 bins).
    require_pos : bool
        If True, discard non-positive \(y\) before averaging (needed for log fits).

    Returns
    -------
    centers : ndarray
        Geometric centers of occupied bins.
    means : ndarray
        Means of \(y\) per occupied bin.
    counts : ndarray
        Counts per occupied bin.
    """
    x = np.asarray(x_AU, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    if require_pos:
        m &= (y > 0.0)
    if not np.any(m):
        return np.array([]), np.array([]), np.array([])

    x = x[m]
    y = y[m]
    xmin, xmax = x.min(), x.max()
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0.0:
        return np.array([]), np.array([]), np.array([])

    if xmin == xmax:
        edges = np.array([xmin * 0.999, xmin * 1.001])
    else:
        edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)

    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    B = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    counts  = np.bincount(idx, minlength=B).astype(float)
    sums    = np.bincount(idx, weights=y, minlength=B)
    have = counts > 0
    means = sums[have] / counts[have]
    return centers[have], means, counts[have]

# ------------------------------ likelihoods/GLS -------------------------------

def _loglike_wls(y: np.ndarray, V: np.ndarray, params: np.ndarray, w: np.ndarray) -> float:
    r"""
    Gaussian WLS log-likelihood under heteroskedastic weights \(w_i\).

    Model:
    \[
      y = V\theta + \epsilon,\quad \epsilon_i \sim \mathcal{N}(0,\sigma_i^2),\quad
      w_i \propto \sigma_i^{-2}.
    \]
    We use \( \hat\sigma^2 = \frac{1}{n}\sum_i w_i (y_i - V_i\theta)^2\) as scale.

    Returns
    -------
    ll : float
        Log-likelihood up to an additive constant (consistent across models).
    """
    resid  = y - V.dot(params)
    n      = len(y)
    sigma2 = np.sum(w * resid**2) / n
    return -0.5 * (n * (np.log(2*np.pi) + 1.0) + n * np.log(sigma2 + 1e-30) - np.sum(np.log(w + 1e-30)))

def _fgls_fit(
    X_AU: np.ndarray,
    Y: np.ndarray,
    N: Optional[np.ndarray],
    deg: int,
    logY: bool,
    R_min: float,
    R_max: float,
    *,
    var_deg: int = 1,
    weight_by_counts: bool = True,
):
    r"""
    One-iteration Feasible GLS (FGLS) for a polynomial in \(x=\ln(r/\text{AU})\).

    Targets
    -------
    If `logY` is True:
    \[
      y \equiv \ln Y,\quad y = p(x) + \epsilon.
    \]
    Else:
    \[
      y \equiv Z,\quad y = p(x) + \epsilon.
    \]
    Variance model:
    \[
      \ln \epsilon^2 = q(x),\ \deg q = \text{var\_deg}.
    \]

    Returns
    -------
    theta, cov, V, y, w, x
    """
    if X_AU is None or Y is None:
        return None

    X_AU = np.asarray(X_AU, float)
    Y    = np.asarray(Y, float)
    N    = np.ones_like(X_AU, float) if N is None else np.asarray(N, float)

    m = np.isfinite(X_AU) & np.isfinite(Y) & (X_AU > 0.0) & (X_AU >= R_min) & (X_AU <= R_max)
    if logY:
        m &= (Y > 0.0)
    if np.count_nonzero(m) < (deg + 1):
        return None

    X = X_AU[m]
    Y = Y[m]
    Nw = N[m]
    x = np.log(X)
    y = np.log(Y) if logY else Y

    V    = _design(x, deg)
    base = Nw if weight_by_counts else np.ones_like(Nw)

    # Stage 1 (WLS with base weights)
    res0   = sm.WLS(y, V, weights=base).fit()
    theta0 = res0.params
    eps0   = y - V.dot(theta0)

    # Variance model in log-residual-squared
    Vm    = _design(x, var_deg)
    eps2  = np.log(eps0**2 + 1e-30)
    res_v = sm.WLS(eps2, Vm, weights=base).fit()
    c     = res_v.params
    sigma2 = np.exp(Vm.dot(c))

    # Stage 2 (FGLS with heteroskedastic weights)
    w     = base / (sigma2 + 1e-30)
    res   = sm.WLS(y, V, weights=w).fit()
    theta = res.params

    resid      = y - V.dot(theta)
    n          = len(y)
    sigma_hat2 = float(np.sum(w * resid**2) / n)
    VW   = V.T * w
    XtWX = VW.dot(V)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
    cov = sigma_hat2 * XtWX_inv
    return theta, cov, V, y, w, x

def _blocked_cv_lppd_fgls(
    X: np.ndarray,
    Y: np.ndarray,
    N: Optional[np.ndarray],
    deg: int,
    logY: bool,
    R_min: float,
    R_max: float,
    *,
    var_deg: int = 1,
    K: int = 5,
    weight_by_counts: bool = True,
) -> float:
    r"""
    Contiguous K-fold CV of per-point log posterior predictive density (LPPD).

    For test fold with design \(V_{\text{te}}\) and variance model \(\ln\sigma^2= q(x)\):
    \[
      \mu_{\text{te}} = V_{\text{te}}\hat\theta,\quad
      s^2_{\text{te}} =
        \begin{cases}
          \exp(q(x_{\text{te}}))/N_{\text{te}}, & \text{if weight\_by\_counts}\\
          \exp(q(x_{\text{te}})), & \text{otherwise}
        \end{cases}
    \]
    LPPD per point:
    \[
      \ell = -\tfrac{1}{2}\left[\ln(2\pi s^2) + \frac{(y-\mu)^2}{s^2}\right].
    \]
    """
    if X is None or Y is None:
        return -np.inf

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N = np.ones_like(X, float) if N is None else np.asarray(N, float)

    m = np.isfinite(X) & np.isfinite(Y) & (X > 0.0) & (X >= R_min) & (X <= R_max)
    if logY:
        m &= (Y > 0.0)
    if np.count_nonzero(m) < (deg + 1):
        return -np.inf

    Xw, Yw, Nw = X[m], Y[m], N[m]
    x = np.log(Xw)
    y = np.log(Yw) if logY else Yw

    order = np.argsort(x)
    x, y, Nw = x[order], y[order], Nw[order]
    n = len(x)
    if n < K + (deg + 1):
        return -np.inf

    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[: n % K] += 1
    idx    = np.cumsum(fold_sizes)
    starts = np.concatenate(([0], idx[:-1]))
    ends   = idx

    total_ll = 0.0
    total_n  = 0

    for s, e in zip(starts, ends):
        te = np.zeros(n, dtype=bool); te[s:e] = True
        tr = ~te

        x_tr, y_tr, N_tr = x[tr], y[tr], Nw[tr]
        x_te, y_te, N_te = x[te], y[te], Nw[te]

        base_tr = N_tr if weight_by_counts else np.ones_like(N_tr)

        # Stage 1
        V_tr  = _design(x_tr, deg)
        res0  = sm.WLS(y_tr, V_tr, weights=base_tr).fit()
        theta0 = res0.params
        eps0   = y_tr - V_tr.dot(theta0)

        # Variance model
        Vm_tr = _design(x_tr, var_deg)
        eps2  = np.log(eps0**2 + 1e-30)
        res_v = sm.WLS(eps2, Vm_tr, weights=base_tr).fit()
        c     = res_v.params
        sigma2_tr = np.exp(Vm_tr.dot(c))
        w_tr      = base_tr / (sigma2_tr + 1e-30)

        # Final
        res   = sm.WLS(y_tr, V_tr, weights=w_tr).fit()
        theta = res.params

        # Predict on test
        V_te   = _design(x_te, deg)
        Vm_te  = _design(x_te, var_deg)
        mu_te  = V_te.dot(theta)
        s2_eps = np.exp(Vm_te.dot(c))
        s2     = s2_eps / (N_te + 1e-30) if weight_by_counts else s2_eps

        ll = -0.5 * (np.log(2*np.pi * s2) + (y_te - mu_te)**2 / s2)
        total_ll += float(np.sum(ll))
        total_n  += len(y_te)

    return total_ll / max(total_n, 1)

# ------------------------------ piecewise helpers -----------------------------

def _hinge_design(x: np.ndarray, xb: float) -> np.ndarray:
    r"""
    Design for continuous broken line:
    \[
      \ln Y = b + k_1 x + \Delta k\, (x-x_b)_+,\quad (z)_+ \equiv \max(0,z).
    \]
    Columns: [1, x, (x-x_b)_+].
    """
    h = np.maximum(0.0, x - xb)
    return np.column_stack([np.ones_like(x), x, h])

def _ln_AU_cap(q: Optional[u.Quantity]) -> Optional[float]:
    """Convert length Quantity to ln(AU); return None if invalid/missing."""
    if q is None:
        return None
    try:
        rb_au = u.Quantity(q).to(u.au).value
    except Exception:
        return None
    if not np.isfinite(rb_au) or rb_au <= 0:
        return None
    return float(np.log(rb_au))

def _grid_breaks(
    x: np.ndarray,
    *,
    n_grid: int = 40,
    quantiles: Tuple[float, float] = (0.2, 0.8),
    x_lo_cap: Optional[float] = None,
    x_hi_cap: Optional[float] = None,
) -> np.ndarray:
    r"""
    Candidate hinge grid within the intersection:
    \[
      [x_{\min},x_{\max}] \cap [x_{q\_lo},x_{q\_hi}] \cap [x_{\text{lo\_cap}},x_{\text{hi\_cap}}].
    \]
    """
    qlo, qhi = quantiles
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_q_lo = float(np.quantile(x, qlo))
    x_q_hi = float(np.quantile(x, qhi))

    lo = max(x_min, x_q_lo)
    hi = min(x_max, x_q_hi)

    if x_lo_cap is not None:
        lo = max(lo, x_lo_cap)
    if x_hi_cap is not None:
        hi = min(hi, x_hi_cap)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = x_min if x_lo_cap is None else max(x_min, x_lo_cap)
        hi = x_max if x_hi_cap is None else min(x_max, x_hi_cap)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([])

    eps = 1e-6 * max(1.0, abs(hi - lo))
    lo_i = lo + eps
    hi_i = hi - eps
    if hi_i <= lo_i:
        lo_i, hi_i = lo, hi

    return np.linspace(lo_i, hi_i, n_grid)

def _bic_from_ll(ll: float, n: int, k: int) -> float:
    r"""
    Bayesian Information Criterion (BIC):
    \[
      \mathrm{BIC} = k\ln n - 2\,\ell.
    \]
    `k` must count all mean parameters, variance-model parameters, and one scale.
    """
    return max(int(k), 1) * np.log(max(int(n), 1)) - 2.0 * float(ll)

def _fit_piecewise_given_w(x: np.ndarray, y: np.ndarray, w: np.ndarray, xb: float):
    r"""
    Weighted LS for fixed breakpoint \(x_b\) with design \([1,\ x,\ (x-x_b)_+]\).

    Returns
    -------
    theta=[b,k1,dk], ll, cov, V
    """
    V = _hinge_design(x, xb)
    res = sm.WLS(y, V, weights=w).fit()
    theta = res.params
    resid      = y - V.dot(theta)
    n          = len(y)
    sigma_hat2 = float(np.sum(w * resid**2) / n)
    VW   = V.T * w
    XtWX = VW.dot(V)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)
    cov = sigma_hat2 * XtWX_inv
    ll  = _loglike_wls(y, V, theta, w)
    return theta, ll, cov, V

def _fit_piecewise_powerlaw(
    X_AU: np.ndarray,
    Y: np.ndarray,
    N: Optional[np.ndarray],
    logY: bool,
    R_min: float,
    R_max: float,
    *,
    var_deg: int = 1,
    n_grid: int = 40,
    min_break: Optional[u.Quantity] = None,
    max_break: Optional[u.Quantity] = None,
    weight_by_counts: bool = True,
    grid_quantiles: Tuple[float, float] = (0.2, 0.8),
):
    r"""
    Two-part power law on \(\ln Y\) vs \(x=\ln(r/\text{AU})\) with hinge at \(x_b\):

    \[
      \ln Y = b + k_1 x + \Delta k (x-x_b)_+.
    \]

    Returns
    -------
    params, cov, V, y, w, x, ll, bic

    Notes
    -----
    BIC parameter count includes the breakpoint:
    \[
      k = 3\ (\text{mean}) + 1\ (x_b) + (\text{var\_deg}+1)\ (\text{variance}) + 1\ (\sigma^2)
      = \text{var\_deg} + 6.
    \]
    """
    if (X_AU is None) or (Y is None) or (not logY):
        return None

    X_AU = np.asarray(X_AU, float)
    Y    = np.asarray(Y, float)
    N    = np.ones_like(X_AU, float) if N is None else np.asarray(N, float)

    m = np.isfinite(X_AU) & np.isfinite(Y) & (X_AU > 0.0) & (X_AU >= R_min) & (X_AU <= R_max) & (Y > 0.0)
    if np.count_nonzero(m) < 6:
        return None

    X, Y, Nw = X_AU[m], Y[m], N[m]
    x = np.log(X)
    y = np.log(Y)
    base = Nw if weight_by_counts else np.ones_like(Nw)

    # Cap domain in ln(AU)
    x_lo_cap = _ln_AU_cap(min_break)
    x_hi_cap = _ln_AU_cap(max_break)

    # Candidate grid
    xb_candidates = _grid_breaks(
        x, n_grid=n_grid, quantiles=grid_quantiles, x_lo_cap=x_lo_cap, x_hi_cap=x_hi_cap
    )
    if xb_candidates.size == 0:
        return None

    # Initial search with base weights
    best = None
    for xb in xb_candidates:
        theta, ll, cov, V = _fit_piecewise_given_w(x, y, base, xb)
        left  = np.sum(x <  xb)
        right = np.sum(x >= xb)
        if (left >= 2) and (right >= 2) and (best is None or ll > best[0]):
            best = (ll, xb, theta, cov, V)
    if best is None:
        return None

    ll1, xb1, th1, cov1, V1 = best
    eps0 = y - V1.dot(th1)

    # Variance model
    Vm    = _design(x, var_deg)
    eps2  = np.log(eps0**2 + 1e-30)
    res_v = sm.WLS(eps2, Vm, weights=base).fit()
    c     = res_v.params
    sigma2 = np.exp(Vm.dot(c))
    w = base / (sigma2 + 1e-30)

    # Re-search with heteroskedastic weights
    best2 = None
    for xb in xb_candidates:
        theta, ll, cov, V = _fit_piecewise_given_w(x, y, w, xb)
        left  = np.sum(x <  xb)
        right = np.sum(x >= xb)
        if (left >= 2) and (right >= 2) and (best2 is None or ll > best2[0]):
            best2 = (ll, xb, theta, cov, V)
    if best2 is None:
        return None

    llf, xbf, thf, covf, Vf = best2
    b, k1, dk = float(thf[0]), float(thf[1]), float(thf[2])

    params = {
        "b": b,
        "k1": k1,
        "dk": dk,
        "k_right": k1 + dk,
        "k2": dk,            # alias for back-compat
        "xb": float(xbf),
    }

    n = len(y)
    k_eff = (var_deg + 6)  # <-- corrected BIC parameter count (includes breakpoint)
    bic = _bic_from_ll(llf, n, k_eff)
    return params, covf, Vf, y, w, x, llf, bic

def _blocked_cv_lppd_piecewise(
    X: np.ndarray,
    Y: np.ndarray,
    N: Optional[np.ndarray],
    logY: bool,
    R_min: float,
    R_max: float,
    *,
    var_deg: int = 1,
    K: int = 5,
    n_grid: int = 30,
    min_break: Optional[u.Quantity] = None,
    max_break: Optional[u.Quantity] = None,
    weight_by_counts: bool = True,
    grid_quantiles: Tuple[float, float] = (0.2, 0.8),
) -> float:
    r"""
    Contiguous K-fold CV LPPD for the hinge model in \(\ln Y\).

    Returns
    -------
    mean per-point LPPD over all folds.
    """
    if (X is None) or (Y is None) or (not logY):
        return -np.inf

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N = np.ones_like(X, float) if N is None else np.asarray(N, float)

    m = np.isfinite(X) & np.isfinite(Y) & (X > 0.0) & (X >= R_min) & (X <= R_max) & (Y > 0.0)
    if np.count_nonzero(m) < 6:
        return -np.inf

    X, Y, N = X[m], Y[m], N[m]
    x = np.log(X)
    y = np.log(Y)
    order = np.argsort(x)
    x, y, N = x[order], y[order], N[order]
    n = len(x)
    if n < K + 4:
        return -np.inf

    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[: n % K] += 1
    idx    = np.cumsum(fold_sizes)
    starts = np.concatenate(([0], idx[:-1]))
    ends   = idx

    x_lo_cap = _ln_AU_cap(min_break)
    x_hi_cap = _ln_AU_cap(max_break)

    total_ll = 0.0
    total_n  = 0

    for s, e in zip(starts, ends):
        te = np.zeros(n, dtype=bool); te[s:e] = True
        tr = ~te
        x_tr, y_tr, N_tr = x[tr], y[tr], N[tr]
        x_te, y_te, N_te = x[te], y[te], N[te]

        base_tr = N_tr if weight_by_counts else np.ones_like(N_tr)

        xb_candidates = _grid_breaks(
            x_tr, n_grid=n_grid, quantiles=grid_quantiles,
            x_lo_cap=x_lo_cap, x_hi_cap=x_hi_cap
        )
        if xb_candidates.size == 0:
            return -np.inf

        # Initial search
        best = None
        for xb in xb_candidates:
            V = _hinge_design(x_tr, xb)
            res = sm.WLS(y_tr, V, weights=base_tr).fit()
            theta = res.params
            resid = y_tr - V.dot(theta)
            ll = _loglike_wls(y_tr, V, theta, base_tr)
            left  = np.sum(x_tr <  xb)
            right = np.sum(x_tr >= xb)
            if (left >= 2) and (right >= 2) and (best is None or ll > best[0]):
                best = (ll, xb, theta, V, resid)

        if best is None:
            return -np.inf

        _, xb1, th1, V1, eps0 = best

        # Variance model
        Vm   = _design(x_tr, var_deg)
        eps2 = np.log(eps0**2 + 1e-30)
        resv = sm.WLS(eps2, Vm, weights=base_tr).fit()
        c    = resv.params
        sigma2_tr = np.exp(Vm.dot(c))
        w_tr      = base_tr / (sigma2_tr + 1e-30)

        # Re-search under hetero weights
        best2 = None
        for xb in xb_candidates:
            V = _hinge_design(x_tr, xb)
            res = sm.WLS(y_tr, V, weights=w_tr).fit()
            theta = res.params
            ll = _loglike_wls(y_tr, V, theta, w_tr)
            left  = np.sum(x_tr <  xb)
            right = np.sum(x_tr >= xb)
            if (left >= 2) and (right >= 2) and (best2 is None or ll > best2[0]):
                best2 = (ll, xb, theta)

        if best2 is None:
            return -np.inf

        _, xb_final, theta_final = best2

        # Predict on test
        V_te  = _hinge_design(x_te, xb_final)
        mu_te = V_te.dot(theta_final)
        Vm_te = _design(x_te, var_deg)
        s2_eps = np.exp(Vm_te.dot(c))
        s2 = s2_eps / (N_te + 1e-30) if weight_by_counts else s2_eps

        ll = -0.5 * (np.log(2*np.pi * s2) + (y_te - mu_te)**2 / s2)
        total_ll += float(np.sum(ll))
        total_n  += len(y_te)

    return total_ll / max(total_n, 1)

# ------------------------------ selection & eval -------------------------------

def _choose_among_degrees_fgls(
    X: np.ndarray,
    Y: np.ndarray,
    N: Optional[np.ndarray],
    *,
    min_deg: int,
    max_deg: int,
    logY: bool,
    R_min: float,
    R_max: float,
    var_deg: int,
    bic_threshold: float,
    cv_k_folds: int,
    cv_gain_frac: float,
    weight_by_counts: bool,
):
    r"""
    Step-up degree selection (polynomial) using BIC + CV gates.

    Accept a higher degree \(d'\) over incumbent \(d\) only if
    \[
      \Delta\mathrm{BIC}=\mathrm{BIC}(d)-\mathrm{BIC}(d') \ge \text{bic\_threshold}
      \quad\text{and}\quad
      \Delta\mathrm{CV}=\mathrm{CV}(d')-\mathrm{CV}(d) \ge \text{cv\_gain\_frac},
    \]
    where CV is mean per-point LPPD.
    """
    max_deg = int(max_deg)
    min_deg = int(max(1, min_deg))

    # Baseline: try min_deg down to 1, keep the first that fits
    base = None
    for d in range(min_deg, 0, -1):
        fit = _fgls_fit(X, Y, N, d, logY, R_min, R_max, var_deg=var_deg, weight_by_counts=weight_by_counts)
        if fit is not None:
            base = (d, *fit)
            break
    if base is None:
        return None, None, None

    d_best, p_best, cov_best, Vb, yb, wb, xb = base
    ll_best  = _loglike_wls(yb, Vb, p_best, wb)
    n_best   = len(yb)
    k_best = len(p_best) + (var_deg + 1) + 1   # +1 for residual scale
    bic_best = _bic_from_ll(ll_best, n_best, k_best)
    cv_best  = _blocked_cv_lppd_fgls(X, Y, N, d_best, logY, R_min, R_max,
                                     var_deg=var_deg, K=cv_k_folds, weight_by_counts=weight_by_counts)

    # Step-up search
    for d in range(d_best + 1, max_deg + 1):
        fit = _fgls_fit(X, Y, N, d, logY, R_min, R_max,
                        var_deg=var_deg, weight_by_counts=weight_by_counts)
        if fit is None:
            continue  # try higher degree; don't stop the search

        p, cov, V, y, w, x = fit
        ll   = _loglike_wls(y, V, p, w)
        n    = len(y)
        k    = len(p) + (var_deg + 1) + 1
        bic  = _bic_from_ll(ll, n, k)
        cv   = _blocked_cv_lppd_fgls(X, Y, N, d, logY, R_min, R_max,
                                     var_deg=var_deg, K=cv_k_folds,
                                     weight_by_counts=weight_by_counts)

        Dbic = bic_best - bic   # positive if new degree is better
        Dcv  = cv - cv_best     # positive if new degree is better

        if (Dbic >= bic_threshold) and (Dcv >= cv_gain_frac):
            d_best, p_best, cov_best = d, p, cov
            bic_best, cv_best = bic, cv
        else:
            continue

    metrics = {"model": "poly", "deg": d_best, "bic": bic_best, "cv": cv_best}
    return p_best, cov_best, metrics

def _choose_poly_vs_piecewise(
    X: np.ndarray,
    Y: np.ndarray,
    N: Optional[np.ndarray],
    *,
    logY: bool,
    R_min: float,
    R_max: float,
    var_deg: int,
    min_deg: int,
    max_deg: int,
    bic_threshold: float,
    cv_k_folds: int,
    cv_gain_frac: float,
    min_break: Optional[u.Quantity],
    max_break: Optional[u.Quantity],
    weight_by_counts: bool,
    allow_piecewise: bool,
    grid_quantiles: Tuple[float, float] = (0.2, 0.8),
):
    r"""
    Compare best polynomial vs. PWL (hinge) model, honoring `allow_piecewise`.

    Returns
    -------
    dict with keys: {'type', 'params', 'cov', 'metrics', 'logY'}
    """
    # Polynomial branch
    p_poly, cov_poly, met_poly = _choose_among_degrees_fgls(
        X, Y, N,
        min_deg=min_deg, max_deg=max_deg,
        logY=logY, R_min=R_min, R_max=R_max, var_deg=var_deg,
        bic_threshold=bic_threshold, cv_k_folds=cv_k_folds, cv_gain_frac=cv_gain_frac,
        weight_by_counts=weight_by_counts,
    )
    if (p_poly is None) or (met_poly is None):
        return None

    poly_out = {'type': 'poly', 'params': p_poly, 'cov': cov_poly, 'metrics': met_poly, 'logY': logY}

    # Early return if PWL not allowed or not applicable (requires logY True)
    if (not allow_piecewise) or (not logY):
        return poly_out

    # Piecewise branch
    pwl = _fit_piecewise_powerlaw(
        X, Y, N, logY, R_min, R_max,
        var_deg=var_deg, n_grid=40,
        min_break=min_break, max_break=max_break,
        weight_by_counts=weight_by_counts,
        grid_quantiles=grid_quantiles,
    )
    if pwl is None:
        return poly_out

    params_pwl, cov_pwl, Vp, yp, wp, xp, ll_pwl, bic_pwl = pwl
    cv_pwl = _blocked_cv_lppd_piecewise(
        X, Y, N, logY, R_min, R_max,
        var_deg=var_deg, K=cv_k_folds, n_grid=30,
        min_break=min_break, max_break=max_break,
        weight_by_counts=weight_by_counts,
        grid_quantiles=grid_quantiles,
    )

    Dbic = met_poly["bic"] - bic_pwl
    Dcv  = cv_pwl - met_poly["cv"]
    if (Dbic >= bic_threshold) and (Dcv >= cv_gain_frac):
        metrics = {"model": "pwl", "bic": bic_pwl, "cv": cv_pwl, "Dbic_vs_poly": Dbic, "Dcv_vs_poly": Dcv}
        return {'type': 'pwl', 'params': params_pwl, 'cov': cov_pwl, 'metrics': metrics, 'logY': True}
    else:
        return poly_out


def _eval_log_model(model_dict: Dict, ln_r: np.ndarray):
    """
    Evaluate model (poly or PWL) on ln_r.

    Returns
    -------
    y_nat : ndarray
    slope_ln : ndarray   (d ln y / d ln r)
    """
    if (model_dict is None) or (model_dict.get('params') is None):
        return None, None
    if model_dict['type'] == 'poly':
        c = model_dict['params']
        ln_y  = np.polyval(c, ln_r)
        slope = _poly_deriv(c, ln_r)
        return np.exp(ln_y), slope
    elif model_dict['type'] == 'pwl':
        p = model_dict['params']
        b, k1, dk, xb = p['b'], p['k1'], p['dk'], p['xb']
        h = np.maximum(0.0, ln_r - xb)
        ln_y  = b + k1*ln_r + dk*h
        slope = k1 + dk*(ln_r >= xb)  # right-branch at hinge
        return np.exp(ln_y), slope
    else:
        return None, None


def Cr09_cascade_rate(
    df_in: pd.DataFrame,
    *,
    # ---------------- data window / binning for FITTING ONLY ----------------
    R_min: float      = 0.05,
    R_max: float      = 0.30,
    n_bins: int       = 100,
    use_binning: bool = True,   # (1) bin before fitting? (independent of outputs)

    # ---------------- degree controls & selection ---------------------------
    deg_Tp: int = 2,
    deg_Te: int = 2,
    deg_n : int = 2,
    deg_q : int = 2,
    deg_phi: int = 4,
    min_deg_phi: int = 3,
    bic_threshold: float = 6.0,
    cv_k_folds: int = 5,
    cv_gain_frac: float = 0.01,
    var_deg: int = 1,

    # ---------------- columns / Parker setup --------------------------------
    which_Te: str = "Te_spane",
    which_Tp: str = "T_p_Davin",
    theta_deg: float = 0.0,
    r0: u.Quantity = 0.045 * u.au,

    # ---------------- options ------------------------------------------------
    weight_by_counts: bool = True,

    # ---------------- PWL breakpoint caps & switch --------------------------
    min_break: Optional[u.Quantity] = None,
    max_break: Optional[u.Quantity] = 30*u.R_sun,
    allow_piecewise: bool = False,  # hard gate for PWL

    # ---------------- (2) product values policy -----------------------------
    # What VALUES go into Q-terms (independent of how derivatives are fit):
    #   "fit"      → use fitted profiles
    #   "raw"      → use raw measurements (NaNs remain)
    #   "raw_fill" → use raw, but replace NaNs with fitted values for r < fill_R_max
    product_values: str = "fit",
    fill_R_max: u.Quantity = 200*u.R_sun,  # only used if product_values == "raw_fill"

    # ---------------- outputs ------------------------------------------------
    return_raw_values: bool = False,
    return_std: bool = False,
    n_jobs: int = -1,  # reserved
):
    r"""
    Compute Cr09-based heating-rate terms.

    **Contract enforced here (only change vs. your previous version):**
    - All **derivatives** are computed from **fitted** profiles only.
    - All **non-derivative factors** in products follow `product_values`:
        'fit'      → use fitted values (error if fit missing),
        'raw'      → use raw values (unit matched),
        'raw_fill' → elementwise: raw if finite; else (fit & r<fill_R_max) else NaN.

    Conduction derivative is constructed consistently from fitted φ:
      (d/dr) cos^2 φ |_fit = -sin(2 φ_fit) * (dφ/dr)_fit
    while the multiplicative non-derivative factors (q_∥, cos^2 φ) follow `product_values`.
    """
    # ---- validate simple product-values policy ----
    pv = str(product_values).lower()
    if pv not in {"fit", "raw", "raw_fill"}:
        raise ValueError("product_values must be one of {'fit','raw','raw_fill'}")

    # ---- column checks and initial mask ----
    needed = ['d', 'V0', which_Tp, which_Te, 'Np', 'Ne']
    missing = [c for c in needed if c not in df_in.columns]
    if missing:
        out = df_in.copy()
        for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
            out[col] = np.nan
        return out, {"Fail": f"Missing column(s): {missing!r}"}

    mask = df_in[['d', 'V0']].notna().all(axis=1)
    if mask.sum() < 3:
        out = df_in.copy()
        for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
            out[col] = np.nan
        if return_raw_values:
            for c in ['Tp_raw','Te_raw','Np_raw','Ne_raw','qpar_raw','Phi_raw']:
                out[c] = np.nan
        return out, {"Fail": "≤2 valid rows"}

    dfc = df_in.loc[mask].copy()

    # ---- convert to SI quantities ----
    r    = (dfc['d'].astype(float).values * u.au).to(u.m)
    U    = (dfc['V0'].astype(float).values * u.km/u.s).to(u.m/u.s)
    Tp   = (dfc[which_Tp].astype(float).values * u.eV).to(u.K, equivalencies=u.temperature_energy())
    Te   = (dfc[which_Te].astype(float).values * u.eV).to(u.K, equivalencies=u.temperature_energy())
    n_p  = (dfc['Np'].astype(float).values * u.cm**-3).to(u.m**-3)
    n_e  = (dfc['Ne'].astype(float).values * u.cm**-3).to(u.m**-3)
    phi  = (dfc['Phi'].astype(float).values * u.rad) if 'Phi' in dfc.columns else (np.nan * dfc['d'].values * u.rad)
    q_e  = (dfc['qpar'].astype(float).values * (u.W/u.m**2)) if 'qpar' in dfc.columns else (np.full(r.size, np.nan) * (u.W/u.m**2))

    # Independent variable
    r_AU = r.to(u.au).value
    ln_r = np.log(r_AU)

    # ---- raw arrays (unitless for fitting) ----
    Tp_v = Tp.value
    Te_v = Te.value
    np_v = n_p.value
    ne_v = n_e.value
    qe_v = q_e.to_value(u.W/u.m**2)
    ph_v = phi.to_value(u.rad)

    # ================= ALWAYS compute binned stats for reporting ==============
    def _report_bins(require_pos: bool, arr: np.ndarray):
        return _bin_mean_and_count(r_AU, arr, bins=n_bins + 1, require_pos=require_pos)

    Bc_Tp, Bm_Tp, BN_Tp = _report_bins(True, Tp_v)
    Bc_Te, Bm_Te, BN_Te = _report_bins(True, Te_v)
    Bc_np, Bm_np, BN_np = _report_bins(True, np_v)
    Bc_ne, Bm_ne, BN_ne = _report_bins(True, ne_v)
    qpos = np.isfinite(qe_v) & (qe_v > 0.0)
    if np.any(qpos):
        Bc_q, Bm_q, BN_q = _bin_mean_and_count(r_AU[qpos], qe_v[qpos], bins=n_bins + 1, require_pos=True)
    else:
        Bc_q = Bm_q = BN_q = np.array([])
    Bc_ph, Bm_ph, BN_ph = _report_bins(False, ph_v)

    # ================= Prepare data for FITTING (bin or raw) ==================
    if use_binning:
        X_Tp, Y_Tp, N_Tp = Bc_Tp, Bm_Tp, BN_Tp
        X_Te, Y_Te, N_Te = Bc_Te, Bm_Te, BN_Te
        X_np, Y_np, N_np = Bc_np, Bm_np, BN_np
        X_ne, Y_ne, N_ne = Bc_ne, Bm_ne, BN_ne
        X_q,  Y_q,  N_q  = (Bc_q, Bm_q, BN_q) if Bc_q.size else (None, None, None)
        X_ph, Y_ph, N_ph = Bc_ph, Bm_ph, BN_ph
    else:
        X_Tp, Y_Tp, N_Tp = r_AU, Tp_v, None
        X_Te, Y_Te, N_Te = r_AU, Te_v, None
        X_np, Y_np, N_np = r_AU, np_v, None
        X_ne, Y_ne, N_ne = r_AU, ne_v, None
        X_q,  Y_q,  N_q  = (r_AU[qpos], qe_v[qpos], None) if np.any(qpos) else (None, None, None)
        X_ph, Y_ph, N_ph = r_AU, ph_v, None

    # ================= Model selection per variable ===========================
    def _select_task(name, X, Y, N, logY, mind, maxd):
        if X is None or Y is None or (isinstance(X, np.ndarray) and X.size == 0):
            return name, None
        chosen = _choose_poly_vs_piecewise(
            X, Y, N,
            logY=logY, R_min=R_min, R_max=R_max, var_deg=var_deg,
            min_deg=mind, max_deg=maxd,
            bic_threshold=bic_threshold, cv_k_folds=cv_k_folds, cv_gain_frac=cv_gain_frac,
            min_break=min_break, max_break=max_break,
            weight_by_counts=weight_by_counts,
            allow_piecewise=allow_piecewise,
            grid_quantiles=(0.2, 0.8),
        )
        return name, chosen

    tasks = [
        ("Tp",  X_Tp,  Y_Tp,  N_Tp,  True,  1, min(deg_Tp, 2)),
        ("Te",  X_Te,  Y_Te,  N_Te,  True,  1, min(deg_Te, 2)),
        ("np",  X_np,  Y_np,  N_np,  True,  1, min(deg_n , 2)),
        ("ne",  X_ne,  Y_ne,  N_ne,  True,  1, min(deg_n , 2)),
        ("q",   X_q,   Y_q,   N_q,   True,  1, min(deg_q , 2)),
        ("phi", X_ph,  Y_ph,  N_ph,  False, min_deg_phi, min(deg_phi, 4)),
    ]
    sel_dict = dict(_select_task(*t) for t in tasks)

    # Require successful fits for core variables
    if any((sel_dict[k] is None) or (sel_dict[k].get('params') is None) for k in ["Tp","Te","np","ne"]):
        out = df_in.copy()
        for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
            out[col] = np.nan
        if return_raw_values:
            for c in ['Tp_raw','Te_raw','Np_raw','Ne_raw','qpar_raw','Phi_raw']:
                out[c] = np.nan
        return out, {"Fail": "model-fit failure"}

    # ================= Evaluate fitted profiles on original grid ==============
    mod_Tp = sel_dict["Tp"];  Tp_fit_vals, slope_Tp = _eval_log_model(mod_Tp, ln_r)
    mod_Te = sel_dict["Te"];  Te_fit_vals, slope_Te = _eval_log_model(mod_Te, ln_r)
    mod_np = sel_dict["np"];  np_fit_vals, slope_np = _eval_log_model(mod_np, ln_r)
    mod_ne = sel_dict["ne"];  ne_fit_vals, slope_ne = _eval_log_model(mod_ne, ln_r)

    mod_q  = sel_dict["q"]
    if (mod_q is not None) and (mod_q.get('params') is not None):
        q_fit_vals, slope_q = _eval_log_model(mod_q, ln_r)
        have_q_fit = True
    else:
        # keep structure: derivative from fit only → set slope to 0; values handled by policy below
        q_fit_vals = np.full_like(r.value, np.nan)
        slope_q    = np.zeros_like(r.value)
        have_q_fit = False

    Tp_fit = Tp_fit_vals * u.K
    Te_fit = Te_fit_vals * u.K
    np_fit = np_fit_vals * (u.m**-3)
    ne_fit = ne_fit_vals * (u.m**-3)
    q_fit  = q_fit_vals  * (u.W/u.m**2)

    # Phi: polynomial if available; otherwise Parker (derivative must be from the fitted curve)
    mod_ph = sel_dict["phi"]
    if (mod_ph is not None) and (mod_ph.get('type') == 'poly') and (mod_ph.get('params') is not None):
        ph_c         = mod_ph['params']
        phi_fit      = (np.polyval(ph_c, ln_r) * u.rad)
        dphi_dr_fit  = (_poly_deriv(ph_c, ln_r) / r).to(1/u.m)
        have_phi_fit = True
    else:
        phi_fit = parker_spiral_angle(r, U, theta_deg=theta_deg, r0=r0)
        dphi_dr_fit = (np.gradient(phi_fit.to_value(u.rad), r.to_value(u.m)) * (1/u.m)).to(1/u.m)
        have_phi_fit = True  # parker fallback counts as a valid fitted curve for derivatives

    # ================= Derivatives from fits ONLY =============================
    dTp_dr  = (Tp_fit * slope_Tp / r).to(u.K/u.m)
    dTe_dr  = (Te_fit * slope_Te / r).to(u.K/u.m)
    dnp_dr  = (np_fit * slope_np / r).to(u.m**-4)
    dne_dr  = (ne_fit * slope_ne / r).to(u.m**-4)
    dq_dr   = (q_fit  * slope_q  / r).to(u.W/u.m**3)

    # ========== Product VALUES used in Q-terms (policy applies here) ==========
    r_m   = r.to_value(u.m)
    cap_m = u.Quantity(fill_R_max).to_value(u.m)

    def _apply_product_values(policy: str, raw_q: u.Quantity, fit_q: Optional[u.Quantity], *, fit_ok: bool) -> u.Quantity:
        """
        Implement EXACT policy for non-derivative values:
          'fit'      → return fit_q (error if not provided/valid)
          'raw'      → return raw_q (unit coerced to fit unit if given)
          'raw_fill' → out[i] = raw[i] if finite; else if (fit_ok and r[i] < fill_R_max and fit[i] finite) then fit[i]; else NaN
        """
        p = policy.lower()
        if p not in {"fit", "raw", "raw_fill"}:
            raise ValueError("policy must be one of {'fit','raw','raw_fill'}")
        # choose output unit: prefer fit's unit if supplied
        if fit_q is not None:
            unit_out = u.Quantity(fit_q).unit
        else:
            unit_out = u.Quantity(raw_q).unit
        rawv = u.Quantity(raw_q).to_value(unit_out)
        fitv = np.full_like(rawv, np.nan, dtype=float) if fit_q is None else u.Quantity(fit_q).to_value(unit_out)

        if p == "fit":
            if (fit_q is None) or (not fit_ok):
                raise RuntimeError("Requested product_values='fit' but no valid fit is available for this variable.")
            return fitv * unit_out
        if p == "raw":
            return rawv * unit_out

        # raw_fill: fill NaN raws with fit inside radius cap when a valid fit exists
        if not fit_ok:
            return rawv * unit_out
        need_nan   = ~np.isfinite(rawv)
        inside_cap = r_m < cap_m
        can_fill   = np.isfinite(fitv)
        use_fit    = need_nan & inside_cap & can_fill
        outv = np.where(use_fit, fitv, rawv)
        return outv * unit_out

    Tp_use  = _apply_product_values(pv, Tp,  Tp_fit,  fit_ok=True)
    Te_use  = _apply_product_values(pv, Te,  Te_fit,  fit_ok=True)
    np_use  = _apply_product_values(pv, n_p, np_fit, fit_ok=True)
    ne_use  = _apply_product_values(pv, n_e, ne_fit, fit_ok=True)
    q_use   = _apply_product_values(pv, q_e, q_fit, fit_ok=have_q_fit)
    phi_use = _apply_product_values(pv, phi, phi_fit, fit_ok=have_phi_fit)

    # ================= Collisional & conduction terms =========================
    # (all derivatives already from fits; multiplicative values from *_use)

    # Coulomb frequencies (SI)
    nu_pe = (NU_0 * (ne_use/(2.5*u.cm**-3)) * (Te_use/(1e5*u.K))**(-1.5)).to(1/u.s)
    nu_ep = (NU_0 * (np_use/(2.5*u.cm**-3)) * (Tp_use/(1e5*u.K))**(-1.5)).to(1/u.s)

    dT = (Tp_use - Te_use).to(u.K)

    # Q_p
    Qp = (1.5*np_use*U*const.k_B*dTp_dr
          - U*const.k_B*Tp_use*dnp_dr
          + 1.5*np_use*const.k_B*nu_pe*dT).to(u.W/u.m**3)

    # Q_e (collisional part)
    Qe_coll = (1.5*ne_use*U*const.k_B*dTe_dr
               - U*const.k_B*Te_use*dne_dr
               - 1.5*ne_use*const.k_B*nu_ep*dT).to(u.W/u.m**3)

    # Conduction: (1/A) d/dr (A q_∥ cos^2 φ), A ∝ r^2
    A         = r**2
    dA_dr     = (2*r).to(u.m)
    cos2_use  = np.cos(phi_use.to_value(u.rad))**2               # VALUE term: policy
    dC_dr_fit = (-np.sin(2.0*phi_fit.to_value(u.rad)) * dphi_dr_fit).to(1/u.m)   # DERIVATIVE: fitted only

    conduction = ((dA_dr*q_use*cos2_use + A*dq_dr*cos2_use + A*q_use*dC_dr_fit) / A).to(u.W/u.m**3)
    Qe_qpar = (Qe_coll + conduction).to(u.W/u.m**3)

    # ================= Assemble outputs =======================================
    out = df_in.copy()
    for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
        out[col] = np.nan
    out.loc[mask, 'Qp']      = Qp.value
    out.loc[mask, 'Qe']      = Qe_coll.value
    out.loc[mask, 'Qe_qpar'] = Qe_qpar.value
    out.loc[mask, 'Phi']     = phi_use.to_value(u.rad)

    if return_std:
        for c in ['dQp', 'dQe', 'dQe_qpar']:
            out[c] = np.nan  # not propagated here

    if return_raw_values:
        out.loc[mask, 'Tp_raw']   = Tp.to_value(u.K)
        out.loc[mask, 'Te_raw']   = Te.to_value(u.K)
        out.loc[mask, 'Np_raw']   = n_p.to_value(u.m**-3)
        out.loc[mask, 'Ne_raw']   = n_e.to_value(u.m**-3)
        out.loc[mask, 'qpar_raw'] = q_e.to_value(u.W/u.m**2)
        out.loc[mask, 'Phi_raw']  = phi.to_value(u.rad)

    # ================= Diagnostics (fits + ALWAYS binned stats) ===============
    def _entry(model_dict, expr, logy, avg_x, avg_y, avg_n):
        """
        Build diagnostics dict. Always returns avg_* even if model is None.
        """
        diag = dict(
            latex=None, latex_equation=None,
            fit_x=None, fit_y=None,
            avg_x=avg_x, avg_y=avg_y, avg_n=avg_n,
            cov=None, model=None, metrics=None
        )
        if (model_dict is None) or (model_dict.get('params') is None):
            return diag
    
        fit_x = np.logspace(np.log10(r_AU.min()), np.log10(r_AU.max()), 200)
        ln_fx = np.log(fit_x)
        if logy:
            y_model, _ = _eval_log_model(model_dict, ln_fx)
            y_plot = y_model
        else:
            c = model_dict['params']
            y_plot = np.polyval(c, ln_fx)
    
        if model_dict['type'] == 'poly':
            diag['latex']          = _latex_poly_inline(model_dict['params'], 'x', expr, logy)
            diag['latex_equation'] = _latex_poly_equation(model_dict['params'], expr, logy)
        else:
            p = model_dict['params']
            diag['latex']          = _latex_piecewise_inline(p['b'], p['k1'], p['dk'], p['xb'], expr)
            diag['latex_equation'] = _latex_piecewise_inline(p['b'], p['k1'], p['dk'], p['xb'], expr)
    
        diag['fit_x']  = fit_x
        diag['fit_y']  = y_plot
        diag['cov']    = model_dict.get('cov')
        diag['model']  = model_dict['type']
        diag['metrics']= model_dict.get('metrics')
        return diag
    
    fits = {
        'Tp' : _entry(mod_Tp,  r'T_p',           True,  Bc_Tp, Bm_Tp, BN_Tp),
        'Te' : _entry(mod_Te,  r'T_e',           True,  Bc_Te, Bm_Te, BN_Te),
        'np' : _entry(mod_np,  r'n_p',           True,  Bc_np, Bm_np, BN_np),
        'ne' : _entry(mod_ne,  r'n_e',           True,  Bc_ne, Bm_ne, BN_ne),
        'q'  : _entry(mod_q,   r'q_{\parallel}', True,  Bc_q,  Bm_q,  BN_q),
        'phi': _entry(mod_ph,  r'\Phi',          False, Bc_ph, Bm_ph, BN_ph),
    }
    return out, fits




# def Cr09_cascade_rate(
#     df_in: pd.DataFrame,
#     # data window / log-binning
#     R_min: float = 0.05,
#     R_max: float = 0.30,
#     n_bins: int = 100,
#     use_binning: bool = True,

#     # degree controls & selection (log-variables search 1..2; Phi up to 4)
#     deg_Tp: int = 2,
#     deg_Te: int = 2,
#     deg_n : int = 2,
#     deg_q : int = 2,
#     deg_phi: int = 4,
#     min_deg_phi: int = 3,
#     bic_threshold: float = 6.0,
#     cv_k_folds: int = 5,
#     cv_gain_frac: float = 0.01,
#     var_deg: int = 1,

#     # columns / Parker setup
#     which_Te: str = "Te_spane",
#     which_Tp: str = "T_p_Davin",
#     theta_deg: float = 0.0,
#     r0: u.Quantity = 0.045 * u.au,

#     # options
#     weight_by_counts: bool = True,
#     max_break: u.Quantity | None = 25*const.R_sun,

#     # NEW options (raw values & NaN fill below a radial cap; always return averages)
#     use_raw_in_products: bool = True,
#     R_nan_fill_max: u.Quantity = 60*const.R_sun,
#     return_raw_values: bool = False,

#     return_std: bool = False,
#     n_jobs: int = -1
# ):
#     """
#     Additions vs. original:
#       • Product terms use RAW series (optionally), with any non-finite values
#         replaced by fitted values only for r < R_nan_fill_max.
#       • Derivatives still from fits.
#       • Always returns bin-averaged diagnostics (avg_x, avg_y) in fits[…].
#       • *_raw SI columns optionally returned.
#     """

#     # --- tight helpers (handle NaN/±∞ robustly) ---
#     def _fill_under_cap(raw_q, fit_q, r_q, cap_q):
#         unit = u.Quantity(fit_q).unit
#         raw = u.Quantity(raw_q).to_value(unit)
#         fit = u.Quantity(fit_q).to_value(unit)
#         r_m = u.Quantity(r_q).to_value(u.m)
#         cap_m = u.Quantity(cap_q).to_value(u.m)
#         need = (~np.isfinite(raw)) & (r_m < cap_m)
#         out = np.where(need, fit, raw)
#         return out * unit

#     def _diag_avgs(r_AU, Tp_v, Te_v, np_v, ne_v, qe_v, ph_v, n_bins):
#         A_Tp = _bin_mean_and_count(r_AU, Tp_v, bins=n_bins + 1, require_pos=True)
#         A_Te = _bin_mean_and_count(r_AU, Te_v, bins=n_bins + 1, require_pos=True)
#         A_np = _bin_mean_and_count(r_AU, np_v, bins=n_bins + 1, require_pos=True)
#         A_ne = _bin_mean_and_count(r_AU, ne_v, bins=n_bins + 1, require_pos=True)
#         qpos = np.isfinite(qe_v) & (qe_v > 0.0)
#         A_q  = _bin_mean_and_count(r_AU[qpos], qe_v[qpos], bins=n_bins + 1, require_pos=True) if np.any(qpos) else (np.array([]), np.array([]), np.array([]))
#         A_ph = _bin_mean_and_count(r_AU, ph_v, bins=n_bins + 1, require_pos=False)
#         return dict(Tp=A_Tp, Te=A_Te, np=A_np, ne=A_ne, q=A_q, ph=A_ph)

#     # ---- column checks / mask ----
#     needed = ['d', 'V0', which_Tp, which_Te, 'Np', 'Ne']
#     missing = [c for c in needed if c not in df_in.columns]
#     if missing:
#         raise KeyError(f"Missing column(s): {missing!r}")

#     mask = df_in[['d', 'V0']].notna().all(axis=1)
#     if mask.sum() < 3:
#         out = df_in.copy()
#         for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
#             out[col] = np.nan
#         if return_raw_values:
#             for c in ['Tp_raw','Te_raw','Np_raw','Ne_raw','qpar_raw','Phi_raw']:
#                 out[c] = np.nan
#         return out, {"Fail": "≤2 valid rows"}

#     dfc = df_in.loc[mask].copy()

#     # ---- RAW quantities (non-binned) in SI ----
#     r    = (dfc['d'].astype(float).values * u.au).to(u.m)
#     U    = (dfc['V0'].astype(float).values * u.km/u.s).to(u.m/u.s)
#     Tp   = (dfc[which_Tp].astype(float).values * u.eV).to(u.K, equivalencies=u.temperature_energy())
#     Te   = (dfc[which_Te].astype(float).values * u.eV).to(u.K, equivalencies=u.temperature_energy())
#     n_p  = (dfc['Np'].astype(float).values * u.cm**-3).to(u.m**-3)
#     n_e  = (dfc['Ne'].astype(float).values * u.cm**-3).to(u.m**-3)
#     phi  = (dfc['Phi'].astype(float).values * u.rad) if 'Phi' in dfc.columns else (np.nan * dfc['d'].values * u.rad)
#     q_e  = (dfc['qpar'].astype(float).values * (u.W/u.m**2)) if 'qpar' in dfc.columns else (np.full(r.size, np.nan) * (u.W/u.m**2))

#     # independent var for fitting
#     r_AU = r.to(u.au).value
#     ln_r = np.log(r_AU)

#     # float views for fitting
#     Tp_v = Tp.value; Te_v = Te.value
#     np_v = n_p.value; ne_v = n_e.value
#     qe_v = q_e.to_value(u.W/u.m**2)
#     ph_v = phi.to_value(u.rad)

#     # ---- fitting data: binning OR raw ----
#     if use_binning:
#         Bc_Tp, Bm_Tp, BN_Tp = _bin_mean_and_count(r_AU, Tp_v, bins=n_bins + 1, require_pos=True)
#         Bc_Te, Bm_Te, BN_Te = _bin_mean_and_count(r_AU, Te_v, bins=n_bins + 1, require_pos=True)
#         Bc_np, Bm_np, BN_np = _bin_mean_and_count(r_AU, np_v, bins=n_bins + 1, require_pos=True)
#         Bc_ne, Bm_ne, BN_ne = _bin_mean_and_count(r_AU, ne_v, bins=n_bins + 1, require_pos=True)
#         qpos = np.isfinite(qe_v) & (qe_v > 0.0)
#         if np.any(qpos):
#             Bc_q, Bm_q, BN_q = _bin_mean_and_count(r_AU[qpos], qe_v[qpos], bins=n_bins + 1, require_pos=True)
#         else:
#             Bc_q = Bm_q = BN_q = np.array([])
#         Bc_ph, Bm_ph, BN_ph = _bin_mean_and_count(r_AU, ph_v, bins=n_bins + 1, require_pos=False)

#         X_Tp, Y_Tp, N_Tp = Bc_Tp, Bm_Tp, BN_Tp
#         X_Te, Y_Te, N_Te = Bc_Te, Bm_Te, BN_Te
#         X_np, Y_np, N_np = Bc_np, Bm_np, BN_np
#         X_ne, Y_ne, N_ne = Bc_ne, Bm_ne, BN_ne
#         X_q,  Y_q,  N_q  = (Bc_q, Bm_q, BN_q) if Bc_q.size else (None, None, None)
#         X_ph, Y_ph, N_ph = Bc_ph, Bm_ph, BN_ph
#     else:
#         X_Tp, Y_Tp, N_Tp = r_AU, Tp_v, None
#         X_Te, Y_Te, N_Te = r_AU, Te_v, None
#         X_np, Y_np, N_np = r_AU, np_v, None
#         X_ne, Y_ne, N_ne = r_AU, ne_v, None
#         qpos = np.isfinite(qe_v) & (qe_v > 0.0)
#         X_q,  Y_q,  N_q  = (r_AU[qpos], qe_v[qpos], None) if np.any(qpos) else (None, None, None)
#         X_ph, Y_ph, N_ph = r_AU, ph_v, None

#     # ---- ALWAYS compute diagnostic averages (independent of fitting path) ----
#     _diag = _diag_avgs(r_AU, Tp_v, Te_v, np_v, ne_v, qe_v, ph_v, n_bins)
#     Bc_Tp_avg, Bm_Tp_avg = _diag['Tp'][0], _diag['Tp'][1]
#     Bc_Te_avg, Bm_Te_avg = _diag['Te'][0], _diag['Te'][1]
#     Bc_np_avg, Bm_np_avg = _diag['np'][0], _diag['np'][1]
#     Bc_ne_avg, Bm_ne_avg = _diag['ne'][0], _diag['ne'][1]
#     Bc_q_avg,  Bm_q_avg  = _diag['q'][0],  _diag['q'][1]
#     Bc_ph_avg, Bm_ph_avg = _diag['ph'][0], _diag['ph'][1]

#     # ---- Model selection ----
#     def _select_task(name, X, Y, N, logY, mind, maxd):
#         if X is None or Y is None or (isinstance(X, np.ndarray) and X.size == 0):
#             return name, None
#         return name, _choose_poly_vs_piecewise(
#             X, Y, N,
#             logY=logY, R_min=R_min, R_max=R_max, var_deg=var_deg,
#             min_deg=mind, max_deg=maxd,
#             bic_threshold=bic_threshold, cv_k_folds=cv_k_folds, cv_gain_frac=cv_gain_frac,
#             max_break=max_break, weight_by_counts=bool(weight_by_counts)
#         )

#     tasks = [
#         ("Tp",  X_Tp,  Y_Tp,  N_Tp,  True,  1, min(deg_Tp, 2)),
#         ("Te",  X_Te,  Y_Te,  N_Te,  True,  1, min(deg_Te, 2)),
#         ("np",  X_np,  Y_np,  N_np,  True,  1, min(deg_n , 2)),
#         ("ne",  X_ne,  Y_ne,  N_ne,  True,  1, min(deg_n , 2)),
#         ("q",   X_q,   Y_q,   N_q,   True,  1, min(deg_q , 2)),
#         ("phi", X_ph,  Y_ph,  N_ph,  False, min_deg_phi, min(deg_phi, 4)),
#     ]
#     sel = dict(_select_task(*t) for t in tasks)

#     mod_Tp, mod_Te, mod_np, mod_ne = sel["Tp"], sel["Te"], sel["np"], sel["ne"]
#     mod_q,  mod_ph = sel["q"], sel["phi"]

#     if any((m is None) or (m.get('params') is None) for m in (mod_Tp, mod_Te, mod_np, mod_ne)):
#         out = df_in.copy()
#         for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
#             out[col] = np.nan
#         if return_raw_values:
#             for c in ['Tp_raw','Te_raw','Np_raw','Ne_raw','qpar_raw','Phi_raw']:
#                 out[c] = np.nan
#         return out, {"Fail": "model-fit failure"}

#     # ---- Evaluate fitted profiles on the original grid (for derivatives) ----
#     Tp_fit_vals, slope_Tp = _eval_log_model(mod_Tp, ln_r)
#     Te_fit_vals, slope_Te = _eval_log_model(mod_Te, ln_r)
#     np_fit_vals, slope_np = _eval_log_model(mod_np, ln_r)
#     ne_fit_vals, slope_ne = _eval_log_model(mod_ne, ln_r)
#     if (mod_q is not None) and (mod_q.get('params') is not None):
#         q_fit_vals, slope_q = _eval_log_model(mod_q, ln_r)
#     else:
#         q_fit_vals = np.zeros_like(r.value); slope_q = np.zeros_like(r.value)

#     Tp_fit = Tp_fit_vals * u.K
#     Te_fit = Te_fit_vals * u.K
#     np_fit = np_fit_vals * (u.m**-3)
#     ne_fit = ne_fit_vals * (u.m**-3)
#     q_fit  = q_fit_vals  * (u.W/u.m**2)

#     # Phi (value for products; derivative for conduction)
#     if (mod_ph is not None) and (mod_ph.get('type') == 'poly') and (mod_ph.get('params') is not None):
#         ph_c   = mod_ph['params']
#         phi_fit = (np.polyval(ph_c, ln_r) * u.rad)
#         dphi_dr = (_poly_deriv(ph_c, ln_r) / r).to(1/u.m)
#     else:
#         phi_fit = parker_spiral_angle(r, U, theta_deg=theta_deg, r0=r0)
#         dphi_dr = (np.gradient(phi_fit.to_value(u.rad), r.to_value(u.m)) * (1/u.m)).to(1/u.m)

#     # ---- Derivatives (from fits only) ----
#     inv_r = (1.0 / r).to(1/u.m)
#     dTp_dr = (Tp_fit * slope_Tp * inv_r).to(u.K/u.m)
#     dTe_dr = (Te_fit * slope_Te * inv_r).to(u.K/u.m)
#     dnp_dr = (np_fit * slope_np * inv_r).to(u.m**-4)
#     dne_dr = (ne_fit * slope_ne * inv_r).to(u.m**-4)
#     dq_dr  = (q_fit  * slope_q  * inv_r).to(u.W/u.m**3)

#     # ---- Values for products: RAW with non-finite fill under cap (if requested) ----
#     if use_raw_in_products:
#         Tp_use  = _fill_under_cap(Tp,  Tp_fit,  r, R_nan_fill_max)
#         Te_use  = _fill_under_cap(Te,  Te_fit,  r, R_nan_fill_max)
#         np_use  = _fill_under_cap(n_p, np_fit,  r, R_nan_fill_max)
#         ne_use  = _fill_under_cap(n_e, ne_fit,  r, R_nan_fill_max)
#         q_use   = _fill_under_cap(q_e, q_fit,   r, R_nan_fill_max)
#         phi_use = _fill_under_cap(phi, phi_fit, r, R_nan_fill_max)
#     else:
#         Tp_use, Te_use, np_use, ne_use = Tp_fit, Te_fit, np_fit, ne_fit
#         q_use,  phi_use = q_fit, phi_fit

#     # ---- Collisional terms ----
#     nu_pe = (NU_0 * (ne_use/(2.5*u.cm**-3)) * (Te_use/(1e5*u.K))**(-1.5)).to(1/u.s)
#     nu_ep = (NU_0 * (np_use/(2.5*u.cm**-3)) * (Tp_use/(1e5*u.K))**(-1.5)).to(1/u.s)
#     dT    = (Tp_use - Te_use).to(u.K)

#     Qp = (1.5*np_use*U*const.k_B*dTp_dr
#           - U*const.k_B*Tp_use*dnp_dr
#           + 1.5*np_use*const.k_B*nu_pe*dT).to(u.W/u.m**3)

#     Qe_coll = (1.5*ne_use*U*const.k_B*dTe_dr
#                - U*const.k_B*Te_use*dne_dr
#                - 1.5*ne_use*const.k_B*nu_ep*dT).to(u.W/u.m**3)

#     # ---- Conduction term ----
#     A         = r**2
#     dA        = (2*r).to(u.m)
#     cos_phi   = np.cos(phi_use.to_value(u.rad))
#     C_factor  = cos_phi**2
#     dC_factor = (-np.sin(2*phi_use.to_value(u.rad)) * dphi_dr).to(1/u.m)
#     conduction = ((dA*q_use*C_factor + A*dq_dr*C_factor + A*q_use*dC_factor) / A).to(u.W/u.m**3)

#     Qe_qpar = (Qe_coll + conduction).to(u.W/u.m**3)

#     # ---- Output frame ----
#     out = df_in.copy()
#     for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
#         out[col] = np.nan
#     out.loc[mask, 'Qp']      = Qp.value
#     out.loc[mask, 'Qe']      = Qe_coll.value
#     out.loc[mask, 'Qe_qpar'] = Qe_qpar.value
#     out.loc[mask, 'Phi']     = u.Quantity(phi_use).to_value(u.rad)

#     if return_std:
#         for c in ['dQp', 'dQe', 'dQe_qpar']:
#             out[c] = np.nan

#     if return_raw_values:
#         out.loc[mask, 'Tp_raw']   = Tp.to_value(u.K)
#         out.loc[mask, 'Te_raw']   = Te.to_value(u.K)
#         out.loc[mask, 'Np_raw']   = n_p.to_value(u.m**-3)
#         out.loc[mask, 'Ne_raw']   = n_e.to_value(u.m**-3)
#         out.loc[mask, 'qpar_raw'] = q_e.to_value(u.W/u.m**2)
#         out.loc[mask, 'Phi_raw']  = phi.to_value(u.rad)

#     # ---- Diagnostics payload (always carries averages) ----
#     def _entry(model_dict, expr, logy, avg_x, avg_y):
#         entry = dict(
#             latex=None, latex_display=None, fit_x=None, fit_y=None,
#             avg_x=avg_x, avg_y=avg_y, cov=None, model=None, metrics=None
#         )
#         if (model_dict is None) or (model_dict.get('params') is None):
#             return entry
#         fx = np.logspace(np.log10(r_AU.min()), np.log10(r_AU.max()), 200)
#         lfx = np.log(fx)
#         if logy:
#             y_model, _ = _eval_log_model(model_dict, lfx)
#             y_plot = y_model
#         else:
#             c = model_dict['params']
#             y_plot = np.polyval(c, lfx)
#         if model_dict['type'] == 'poly':
#             latex_inline  = _latex_poly_inline(model_dict['params'], 'x', expr, logy)
#             latex_display = None
#         else:
#             p = model_dict['params']
#             latex_inline, latex_display = _latex_piecewise_powerlaw_forms(p['b'], p['k1'], p['k2'], p['xb'], expr)
#         entry.update(dict(
#             latex=latex_inline, latex_display=latex_display,
#             fit_x=fx, fit_y=y_plot, cov=model_dict.get('cov'),
#             model=model_dict['type'], metrics=model_dict.get('metrics')
#         ))
#         return entry

#     fits = {
#         'Tp' : _entry(mod_Tp,  r'T_p',           True,  Bc_Tp_avg, Bm_Tp_avg),
#         'Te' : _entry(mod_Te,  r'T_e',           True,  Bc_Te_avg, Bm_Te_avg),
#         'np' : _entry(mod_np,  r'n_p',           True,  Bc_np_avg, Bm_np_avg),
#         'ne' : _entry(mod_ne,  r'n_e',           True,  Bc_ne_avg, Bm_ne_avg),
#         'q'  : _entry(mod_q,   r'q_{\parallel}', True,  Bc_q_avg,  Bm_q_avg),
#         'phi': _entry(mod_ph,  r'\Phi',          False, Bc_ph_avg, Bm_ph_avg),
#     }
#     return out, fits




# # -*- coding: utf-8 -*-
# # FGLS-based fitting and heating-rate computation with explicit heteroskedasticity handling
# # Dependencies: numpy, pandas, astropy, statsmodels

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from astropy import units as u
# from astropy import constants as const

# # ------------------ constants ------------------
# OMEGA = 2.7e-6 / u.s     # solar rotation rate
# NU_0  = 8.4e-9  / u.s    # Coulomb collision prefactor

# # ------------------ Parker angle (kept EXACT) ------------------
# def parker_spiral_angle(
#     r: u.Quantity,
#     U: u.Quantity,
#     theta_deg: float = 0.0,
#     r0: u.Quantity = 0.045 * u.au
# ) -> u.Quantity:
#     """
#     Analytic Parker-spiral angle relative to radial direction.
#     """
#     theta_rad = np.deg2rad(theta_deg)
#     arg = (OMEGA * (r - r0) * np.sin(theta_rad) / U).to(u.dimensionless_unscaled)
#     return -np.arctan(arg.value) * u.rad

# # ------------------ math helpers ------------------
# def _poly_deriv(coef: np.ndarray, ln_r: np.ndarray) -> np.ndarray:
#     """Evaluate derivative of polynomial p(ln r) with respect to ln r at the points ln r."""
#     return np.zeros_like(ln_r) if coef is None else np.polyval(np.polyder(coef), ln_r)

# def _design(x_ln: np.ndarray, deg: int) -> np.ndarray:
#     """Vandermonde design matrix for polynomial of degree 'deg' in x = ln(r/AU)."""
#     return np.vander(x_ln, deg + 1)

# def _latex_poly(coef: np.ndarray, var: str, expr: str, logy: bool) -> str:
#     """Human-readable LaTeX of fitted polynomial; for diagnostics/export."""
#     if coef is None:
#         return ""
#     d = len(coef) - 1
#     parts = []
#     for i, c in enumerate(coef):
#         p = d - i
#         mon = f"{abs(c):.3g}{var}^{p}" if p > 1 else (f"{abs(c):.3g}{var}" if p == 1 else f"{abs(c):.3g}")
#         parts.append(("+" if (c >= 0 and i) else "-") + mon if i else mon)
#     return (rf"$\ln({expr})={' '.join(parts)}$") if logy else (rf"${expr}={' '.join(parts)}$")

# # ------------------ binning (mean & count on log-r grid) ------------------
# def _bin_mean_and_count(x_AU, y, *, bins: int, require_pos: bool):
#     """
#     Bin x on a logarithmic radial grid.
#     For each non-empty bin, return:
#       centers (geometric), arithmetic mean of y (natural scale), and count N.
#     If require_pos=True, discard non-positive y prior to binning.
#     """
#     x = np.asarray(x_AU, float)
#     y = np.asarray(y, float)
#     m = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
#     if require_pos:
#         m &= (y > 0.0)
#     if not np.any(m):
#         return np.array([]), np.array([]), np.array([])

#     x = x[m]
#     y = y[m]
#     xmin, xmax = x.min(), x.max()
#     if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin <= 0.0:
#         return np.array([]), np.array([]), np.array([])
#     if xmin == xmax:
#         edges = np.array([xmin * 0.999, xmin * 1.001])
#     else:
#         edges = np.logspace(np.log10(xmin), np.log10(xmax), bins)

#     # Assign each x to a bin index in [0, B-1]
#     idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)

#     B = len(edges) - 1
#     centers = np.sqrt(edges[:-1] * edges[1:])  # geometric centers
#     counts  = np.bincount(idx, minlength=B).astype(float)
#     sums    = np.bincount(idx, weights=y, minlength=B)

#     have = counts > 0
#     means = sums[have] / counts[have]
#     return centers[have], means, counts[have]

# # ------------------ FGLS: initial WLS -> variance model -> final WLS ------------------
# def _fgls_fit(X_AU, Y, N, deg: int, logY: bool, R_min: float, R_max: float, var_deg: int = 1):
#     """
#     1-step Feasible GLS:
#       1) Initial WLS with count-weights W0 = diag(N_b)
#       2) Regress log(residual^2) on polynomial in x (degree var_deg) to model variance
#       3) Final WLS with weights w_b = N_b / sigma_b^2(x)
#     Returns:
#       params, cov, V, y, w, x
#     """
#     if X_AU is None or Y is None:
#         return None

#     X_AU = np.asarray(X_AU, float)
#     Y    = np.asarray(Y, float)
#     if N is None:
#         N = np.ones_like(X_AU, dtype=float)
#     else:
#         N = np.asarray(N, float)

#     # Windowing and validity
#     m = np.isfinite(X_AU) & np.isfinite(Y) & (X_AU > 0.0) & (X_AU >= R_min) & (X_AU <= R_max)
#     if logY:
#         m &= (Y > 0.0)
#     if np.count_nonzero(m) < (deg + 1):
#         return None

#     X = X_AU[m]
#     Y = Y[m]
#     Nw = N[m]
#     x = np.log(X)
#     y = np.log(Y) if logY else Y

#     # Design matrices
#     V  = _design(x, deg)
#     W0 = Nw  # initial weights: counts

#     # Stage 1: initial WLS fit
#     res0   = sm.WLS(y, V, weights=W0).fit()
#     theta0 = res0.params
#     eps0   = y - V.dot(theta0)

#     # Stage 2: variance model on log residual^2
#     eps2  = np.log(eps0**2 + 1e-30)
#     Vm    = _design(x, var_deg)
#     res_v = sm.WLS(eps2, Vm, weights=W0).fit()
#     c     = res_v.params
#     log_sigma2 = Vm.dot(c)
#     sigma2     = np.exp(log_sigma2)

#     # Stage 3: final WLS fit with heteroskedastic weights
#     w    = Nw / (sigma2 + 1e-30)
#     res  = sm.WLS(y, V, weights=w).fit()
#     theta = res.params

#     # Sigma-hat^2 (WLS) and covariance of coefficients
#     resid      = y - V.dot(theta)
#     n          = len(y)
#     sigma_hat2 = float(np.sum(w * resid**2) / n)

#     VW      = V.T * w
#     XtWX    = VW.dot(V)
#     try:
#         XtWX_inv = np.linalg.inv(XtWX)
#     except np.linalg.LinAlgError:
#         XtWX_inv = np.linalg.pinv(XtWX)
#     cov = sigma_hat2 * XtWX_inv

#     return theta, cov, V, y, w, x

# # ------------------ Gaussian WLS log-likelihood ------------------
# def _loglike_wls(y, V, params, w):
#     """WLS Gaussian log-likelihood using sigma^2 estimated from weighted residuals."""
#     resid = y - V.dot(params)
#     n     = len(y)
#     sigma2 = np.sum(w * resid**2) / n
#     return -0.5 * (n * (np.log(2*np.pi) + 1.0) + n * np.log(sigma2 + 1e-30) - np.sum(np.log(w + 1e-30)))

# # ------------------ blocked CV with FGLS weighting ------------------
# def _blocked_cv_lppd_fgls(X, Y, N, deg, logY, R_min, R_max, var_deg=1, K=5):
#     """
#     Contiguous K-fold CV in x = ln(r/AU).
#     Train FGLS on K-1 folds; score held-out fold with predictive variance from trained variance model.
#     Returns mean test log-likelihood across all folds.
#     """
#     if X is None or Y is None:
#         return -np.inf

#     X = np.asarray(X, float)
#     Y = np.asarray(Y, float)
#     N = np.ones_like(X, float) if N is None else np.asarray(N, float)

#     m = np.isfinite(X) & np.isfinite(Y) & (X > 0.0) & (X >= R_min) & (X <= R_max)
#     if logY:
#         m &= (Y > 0.0)
#     if np.count_nonzero(m) < (deg + 1):
#         return -np.inf

#     Xw, Yw, Nw = X[m], Y[m], N[m]
#     x = np.log(Xw)
#     y = np.log(Yw) if logY else Yw

#     order = np.argsort(x)
#     x, y, Nw = x[order], y[order], Nw[order]
#     n = len(x)
#     if n < K + (deg + 1):
#         return -np.inf

#     fold_sizes = np.full(K, n // K, dtype=int)
#     fold_sizes[: n % K] += 1
#     idx = np.cumsum(fold_sizes)
#     starts = np.concatenate(([0], idx[:-1]))
#     ends   = idx

#     total_ll = 0.0
#     total_n  = 0

#     for s, e in zip(starts, ends):
#         te = np.zeros(n, dtype=bool); te[s:e] = True
#         tr = ~te

#         x_tr, y_tr, N_tr = x[tr], y[tr], Nw[tr]
#         x_te, y_te, N_te = x[te], y[te], Nw[te]

#         # Stage 1 (train): initial WLS with counts
#         V_tr  = _design(x_tr, deg)
#         res0  = sm.WLS(y_tr, V_tr, weights=N_tr).fit()
#         theta0 = res0.params
#         eps0   = y_tr - V_tr.dot(theta0)

#         # Variance model on training
#         Vm_tr = _design(x_tr, var_deg)
#         eps2  = np.log(eps0**2 + 1e-30)
#         res_v = sm.WLS(eps2, Vm_tr, weights=N_tr).fit()
#         c     = res_v.params

#         # Final WLS on training with w_tr = N_tr / sigma2_tr
#         log_sigma2_tr = Vm_tr.dot(c)
#         sigma2_tr     = np.exp(log_sigma2_tr)
#         w_tr          = N_tr / (sigma2_tr + 1e-30)
#         res           = sm.WLS(y_tr, V_tr, weights=w_tr).fit()
#         theta         = res.params

#         # Predict on test with variance from trained variance model
#         V_te  = _design(x_te, deg)
#         Vm_te = _design(x_te, var_deg)
#         mu_te  = V_te.dot(theta)
#         sigma2_te = np.exp(Vm_te.dot(c))   # variance of residual in regression space
#         s2 = sigma2_te / (N_te + 1e-30)    # variance of bin-mean target

#         # Gaussian predictive log-likelihood on test fold
#         ll = -0.5 * (np.log(2*np.pi * s2) + (y_te - mu_te)**2 / s2)
#         total_ll += float(np.sum(ll))
#         total_n  += len(y_te)

#     return total_ll / max(total_n, 1)

# # ------------------ degree selection (step-up with BIC + CV) ------------------
# def _choose_among_degrees_fgls(
#     X, Y, N, *, min_deg, max_deg, cap,
#     logY, R_min, R_max, var_deg,
#     bic_threshold, cv_k_folds, cv_gain_frac
# ):
#     """
#     Step-up selection from baseline degree in [1..min_deg] to at most max_deg.
#     Accept d+1 only if ΔBIC >= bic_threshold and ΔCV >= cv_gain_frac.
#     """
#     max_deg = int(min(max_deg, cap))
#     min_deg = int(max(1, min_deg))

#     # Baseline: try min_deg, then lower if infeasible (enforces your minimum preference)
#     base = None
#     for d in range(min_deg, 0, -1):  # min_deg, min_deg-1, ..., 1
#         fit = _fgls_fit(X, Y, N, d, logY, R_min, R_max, var_deg)
#         if fit is not None:
#             base = (d, *fit)
#             break
#     if base is None:
#         return None, None

#     d_best, p_best, cov_best, Vb, yb, wb, xb = base
#     ll_best  = _loglike_wls(yb, Vb, p_best, wb)
#     n_best   = len(yb)
#     k_best   = len(p_best)
#     bic_best = k_best * np.log(n_best) - 2.0 * ll_best
#     cv_best  = _blocked_cv_lppd_fgls(X, Y, N, d_best, logY, R_min, R_max, var_deg=var_deg, K=cv_k_folds)

#     # Step-up search
#     for d in range(d_best + 1, max_deg + 1):
#         fit = _fgls_fit(X, Y, N, d, logY, R_min, R_max, var_deg)
#         if fit is None:
#             break
#         p, cov, V, y, w, x = fit
#         ll   = _loglike_wls(y, V, p, w)
#         n    = len(y)
#         k    = len(p)
#         bic  = k * np.log(n) - 2.0 * ll
#         cv   = _blocked_cv_lppd_fgls(X, Y, N, d, logY, R_min, R_max, var_deg=var_deg, K=cv_k_folds)

#         Dbic = bic_best - bic
#         Dcv  = cv - cv_best
#         accept = (Dbic >= bic_threshold) and (Dcv >= cv_gain_frac)
#         if accept:
#             d_best, p_best, cov_best = d, p, cov
#             bic_best, cv_best = bic, cv
#         else:
#             break

#     return p_best, cov_best

# # ------------------ main: compute fits, derivatives, and heating rates ------------------
# def Cr09_cascade_rate(
#     df_in: pd.DataFrame,
#     # data window / log-binning
#     R_min: float = 0.05,
#     R_max: float = 0.30,
#     n_bins: int = 100,
#     use_binning: bool = True,

#     # degree controls & selection
#     deg_Tp: int = 2,
#     deg_Te: int = 2,
#     deg_n : int = 2,
#     deg_q : int = 2,
#     deg_phi: int = 4,
#     min_deg_phi: int = 3,       # prefer cubic for Phi; fallback allowed
#     bic_threshold: float = 6.0,
#     cv_k_folds: int = 5,
#     cv_gain_frac: float = 0.01,
#     var_deg: int = 1,           # degree of log-variance model

#     # columns / Parker setup
#     which_Te: str = "Te_spane",
#     which_Tp: str = "T_p_Davin",
#     theta_deg: float = 0.0,
#     r0: u.Quantity = 0.045 * u.au,

#     return_std: bool = False,
#     n_jobs: int = -1
# ):
#     """
#     FGLS-based fitting for Tp, Te, np, ne, q_parallel, and Phi with heteroskedasticity handling.
#     Produces smoothed profiles, analytic derivatives via chain rule, and heating-rate budgets.
#     """

#     # ---- column checks and initial mask ----
#     needed = ['d', 'V0', which_Tp, which_Te, 'Np', 'Ne']
#     missing = [c for c in needed if c not in df_in.columns]
#     if missing:
#         raise KeyError(f"Missing column(s): {missing!r}")

#     mask = df_in[['d', 'V0']].notna().all(axis=1)
#     if mask.sum() < 3:
#         out = df_in.copy()
#         for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
#             out[col] = np.nan
#         return out, {"Fail": "≤2 valid rows"}

#     dfc = df_in.loc[mask].copy()

#     # ---- convert to SI quantities ----
#     r    = (dfc['d'].astype(float).values * u.au).to(u.m)
#     U    = (dfc['V0'].astype(float).values * u.km/u.s).to(u.m/u.s)
#     Tp   = (dfc[which_Tp].astype(float).values * u.eV).to(u.K, equivalencies=u.temperature_energy())
#     Te   = (dfc[which_Te].astype(float).values * u.eV).to(u.K, equivalencies=u.temperature_energy())
#     n_p  = (dfc['Np'].astype(float).values * u.cm**-3).to(u.m**-3)
#     n_e  = (dfc['Ne'].astype(float).values * u.cm**-3).to(u.m**-3)
#     phi  = (dfc['Phi'].astype(float).values * u.rad) if 'Phi' in dfc.columns else (np.nan * dfc['d'].values * u.rad)
#     q_e  = (dfc['qpar'].astype(float).values * (u.W/u.m**2)) if 'qpar' in dfc.columns else (np.full(r.size, np.nan) * (u.W/u.m**2))

#     # Independent variable
#     r_AU = r.to(u.au).value
#     ln_r = np.log(r_AU)

#     # raw arrays (unitless for fitting)
#     Tp_v = Tp.value
#     Te_v = Te.value
#     np_v = n_p.value
#     ne_v = n_e.value
#     qe_v = q_e.to_value(u.W/u.m**2)
#     ph_v = phi.to_value(u.rad)

#     # ---- binning (means & counts) OR raw points ----
#     if use_binning:
#         Bc_Tp, Bm_Tp, BN_Tp = _bin_mean_and_count(r_AU, Tp_v, bins=n_bins + 1, require_pos=True)
#         Bc_Te, Bm_Te, BN_Te = _bin_mean_and_count(r_AU, Te_v, bins=n_bins + 1, require_pos=True)
#         Bc_np, Bm_np, BN_np = _bin_mean_and_count(r_AU, np_v, bins=n_bins + 1, require_pos=True)
#         Bc_ne, Bm_ne, BN_ne = _bin_mean_and_count(r_AU, ne_v, bins=n_bins + 1, require_pos=True)
#         qpos = np.isfinite(qe_v) & (qe_v > 0.0)
#         if np.any(qpos):
#             Bc_q,  Bm_q,  BN_q  = _bin_mean_and_count(r_AU[qpos], qe_v[qpos], bins=n_bins + 1, require_pos=True)
#         else:
#             Bc_q = Bm_q = BN_q = np.array([])
#         Bc_ph, Bm_ph, BN_ph = _bin_mean_and_count(r_AU, ph_v, bins=n_bins + 1, require_pos=False)

#         X_Tp, Y_Tp, N_Tp = Bc_Tp, Bm_Tp, BN_Tp
#         X_Te, Y_Te, N_Te = Bc_Te, Bm_Te, BN_Te
#         X_np, Y_np, N_np = Bc_np, Bm_np, BN_np
#         X_ne, Y_ne, N_ne = Bc_ne, Bm_ne, BN_ne
#         X_q,  Y_q,  N_q  = (Bc_q, Bm_q, BN_q) if Bc_q.size else (None, None, None)
#         X_ph, Y_ph, N_ph = Bc_ph, Bm_ph, BN_ph
#     else:
#         X_Tp, Y_Tp, N_Tp = r_AU, Tp_v, None
#         X_Te, Y_Te, N_Te = r_AU, Te_v, None
#         X_np, Y_np, N_np = r_AU, np_v, None
#         X_ne, Y_ne, N_ne = r_AU, ne_v, None
#         qpos = np.isfinite(qe_v) & (qe_v > 0.0)
#         X_q,  Y_q,  N_q  = (r_AU[qpos], qe_v[qpos], None) if np.any(qpos) else (None, None, None)
#         X_ph, Y_ph, N_ph = r_AU, ph_v, None

#         # Ensure any references to bin summaries are safe later
#         Bc_Tp = Bm_Tp = Bc_Te = Bm_Te = Bc_np = Bm_np = Bc_ne = Bm_ne = None
#         Bc_q  = Bm_q  = Bc_ph = Bm_ph = None

#     # ---- FGLS fitting with degree constraints ----
#     def _select_task(name, X, Y, N, logY, mind, maxd):
#         if X is None or Y is None or (isinstance(X, np.ndarray) and X.size == 0):
#             return name, (None, None)
#         p, cov = _choose_among_degrees_fgls(
#             X, Y, N,
#             min_deg=mind, max_deg=maxd, cap=maxd,
#             logY=logY, R_min=R_min, R_max=R_max, var_deg=var_deg,
#             bic_threshold=bic_threshold, cv_k_folds=cv_k_folds, cv_gain_frac=cv_gain_frac
#         )
#         return name, (p, cov)

#     tasks = [
#         ("Tp",  X_Tp,  Y_Tp,  N_Tp,  True,  1, min(deg_Tp, 2)),
#         ("Te",  X_Te,  Y_Te,  N_Te,  True,  1, min(deg_Te, 2)),
#         ("np",  X_np,  Y_np,  N_np,  True,  1, min(deg_n , 2)),
#         ("ne",  X_ne,  Y_ne,  N_ne,  True,  1, min(deg_n , 2)),
#         ("q",   X_q,   Y_q,   N_q,   True,  1, min(deg_q , 2)),
#         ("phi", X_ph,  Y_ph,  N_ph,  False, min_deg_phi, min(deg_phi, 4)),
#     ]
#     sel = [_select_task(*t) for t in tasks]
#     sel_dict = dict(sel)

#     Tp_c,  Tp_cov  = sel_dict["Tp"]
#     Te_c,  Te_cov  = sel_dict["Te"]
#     np_c,  np_cov  = sel_dict["np"]
#     ne_c,  ne_cov  = sel_dict["ne"]
#     q_c,   q_cov   = sel_dict["q"]
#     ph_c,  ph_cov  = sel_dict["phi"]

#     if any(c is None for c in (Tp_c, Te_c, np_c, ne_c)):
#         out = df_in.copy()
#         for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
#             out[col] = np.nan
#         return out, {"Fail": "poly-fit failure"}

#     # ---- Evaluate fitted profiles on the original grid ----
#     Tp_fit  = np.exp(np.polyval(Tp_c, ln_r)) * u.K
#     Te_fit  = np.exp(np.polyval(Te_c, ln_r)) * u.K
#     np_fit  = np.exp(np.polyval(np_c, ln_r)) * (u.m**-3)
#     ne_fit  = np.exp(np.polyval(ne_c, ln_r)) * (u.m**-3)
#     q_fit   = (np.exp(np.polyval(q_c, ln_r)) * (u.W/u.m**2)) if q_c is not None else (np.zeros_like(r.value) * (u.W/u.m**2))

#     # Phi: use fitted polynomial if available; otherwise fall back to analytic Parker angle
#     if ph_c is not None:
#         phi_fit = (np.polyval(ph_c, ln_r) * u.rad)
#         dphi_dr = (_poly_deriv(ph_c, ln_r) / r).to(1/u.m)
#     else:
#         phi_fit = parker_spiral_angle(r, U, theta_deg=theta_deg, r0=r0)
#         dphi_vals = phi_fit.to_value(u.rad)
#         dr_vals   = r.to_value(u.m)
#         dphi_num  = np.gradient(dphi_vals, dr_vals)
#         dphi_dr   = (dphi_num * (1.0/u.m)).to(1/u.m)

#     # Chain-rule derivatives for log-fitted variables: d/dr = (1/r) d/d ln r
#     dTp_dr  = (Tp_fit * _poly_deriv(Tp_c, ln_r) / r).to(u.K/u.m)
#     dTe_dr  = (Te_fit * _poly_deriv(Te_c, ln_r) / r).to(u.K/u.m)
#     dnp_dr  = (np_fit * _poly_deriv(np_c, ln_r) / r).to(u.m**-4)
#     dne_dr  = (ne_fit * _poly_deriv(ne_c, ln_r) / r).to(u.m**-4)
#     dq_dr   = (q_fit  * _poly_deriv(q_c,  ln_r) / r).to(u.W/u.m**3) if q_c is not None else (0.0 * (u.W/u.m**3))

#     # Use smoothed values for products
#     Tp_use, Te_use, np_use, ne_use = Tp_fit, Te_fit, np_fit, ne_fit
#     q_use   = q_fit
#     phi_use = phi_fit

#     # ---- collisional terms ----
#     nu_pe = (NU_0 * (ne_use/(2.5*u.cm**-3)) * (Te_use/(1e5*u.K))**(-1.5)).to(1/u.s)
#     nu_ep = (NU_0 * (np_use/(2.5*u.cm**-3)) * (Tp_use/(1e5*u.K))**(-1.5)).to(1/u.s)
#     dT    = (Tp_use - Te_use).to(u.K)

#     Qp = (1.5*np_use*U*const.k_B*dTp_dr
#           - U*const.k_B*Tp_use*dnp_dr
#           + 1.5*np_use*const.k_B*nu_pe*dT).to(u.W/u.m**3)

#     Qe_coll = (1.5*ne_use*U*const.k_B*dTe_dr
#                - U*const.k_B*Te_use*dne_dr
#                - 1.5*ne_use*const.k_B*nu_ep*dT).to(u.W/u.m**3)

#     # ---- conduction term (parallel) ----
#     A         = r**2
#     dA        = (2*r).to(u.m)               # dA/dr
#     cos_phi   = np.cos(phi_use.to_value(u.rad))
#     C_factor  = cos_phi**2
#     dC_factor = (-np.sin(2*phi_use.to_value(u.rad)) * dphi_dr).to(1/u.m)
#     conduction = ((dA*q_use*C_factor + A*dq_dr*C_factor + A*q_use*dC_factor) / A).to(u.W/u.m**3)

#     Qe_qpar = (Qe_coll + conduction).to(u.W/u.m**3)

#     # ---- assemble output ----
#     out = df_in.copy()
#     for col in ['Qp', 'Qe', 'Qe_qpar', 'dQp', 'dQe', 'dQe_qpar', 'Phi']:
#         out[col] = np.nan
#     out.loc[mask, 'Qp']      = Qp.value
#     out.loc[mask, 'Qe']      = Qe_coll.value
#     out.loc[mask, 'Qe_qpar'] = Qe_qpar.value
#     out.loc[mask, 'Phi']     = phi_use.to_value(u.rad)

#     if return_std:
#         # Placeholder for future uncertainty propagation
#         for c in ['dQp', 'dQe', 'dQe_qpar']:
#             out[c] = np.nan

#     # Diagnostics payload: fitted curves and binned summaries
#     def _entry(c, cov, expr, logy, avg_x, avg_y):
#         if c is None:
#             return None
#         fit_x = np.logspace(np.log10(r_AU.min()), np.log10(r_AU.max()), 200)
#         ln_fx = np.log(fit_x)
#         y_model = (np.exp(np.polyval(c, ln_fx)) if logy else np.polyval(c, ln_fx))
#         return dict(
#             latex=_latex_poly(c, 'x', expr, logy),
#             fit_x=fit_x, fit_y=y_model,
#             avg_x=avg_x, avg_y=avg_y,
#             cov=cov
#         )

#     fits = {
#         'Tp' : _entry(Tp_c,  Tp_cov,  r'T_p',           True,  Bc_Tp, Bm_Tp),
#         'Te' : _entry(Te_c,  Te_cov,  r'T_e',           True,  Bc_Te, Bm_Te),
#         'np' : _entry(np_c,  np_cov,  r'n_p',           True,  Bc_np, Bm_np),
#         'ne' : _entry(ne_c,  ne_cov,  r'n_e',           True,  Bc_ne, Bm_ne),
#         'q'  : _entry(q_c,   q_cov,   r'q_{\parallel}', True,  Bc_q,  Bm_q),
#         'phi': _entry(ph_c,  ph_cov,  r'\Phi',          False, Bc_ph, Bm_ph),
#     }
#     return out, fits




import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d
import statsmodels.api as sm

def CH_09_cascade_rate(
    df,
    col_Ma   = "Ma_r",
    col_r_au = "d",       # AU
    col_V0   = "Vsw",     # km/s
    col_VA0  = "Va",      # km/s
    col_Zp2  = "Zp_amp",  # (km/s)^2 if quant_is_squared=True, else km/s
    col_rho  = "rho",     # interpreted as kg/cm^3 if unitless (legacy behavior)
    units: str    = "SI",
    n_bins: int   = 100,
    poly_deg: int = 1,
    R_min: float  = 0.05,
    R_max: float  = 0.3,
    use_binning: bool = True,
    hetero_fit: str = "fgls",   # 'ols'|'fgls' (used only if use_binning=False)
    quant_is_squared: bool = True,
):
    # ── map columns (keep your names/logic) ───────────────────────────────────
    keep_Ma  = df[col_Ma].values
    keep_d   = df[col_r_au].values
    keep_V0  = df[col_V0].values
    keep_VA0 = df[col_VA0].values
    keep_rho = df[col_rho].values
    if quant_is_squared:
        keep_quant = df[col_Zp2].values
    else:
        keep_quant = (df[col_Zp2].values)**2

    # ── units (preserve legacy rho unless Quantity given) ─────────────────────
    if not isinstance(keep_d, u.Quantity):
        keep_d = keep_d * u.au
    if not isinstance(keep_V0, u.Quantity):
        keep_V0 = keep_V0 * (u.km / u.s)
    if not isinstance(keep_VA0, u.Quantity):
        keep_VA0 = keep_VA0 * (u.km / u.s)
    if not isinstance(keep_quant, u.Quantity):
        keep_quant = keep_quant * (u.km ** 2 / u.s ** 2)
    if not isinstance(keep_rho, u.Quantity):
        keep_rho = keep_rho * (u.kg / u.cm ** 3)   # legacy assumption

    AU_SI    = u.au.to(u.m)
    R_SUN_SI = (1.0 * u.au / 215.032).to(u.m)

    r_si    = keep_d.to_value(u.m)
    V0_si   = keep_V0.to_value(u.m / u.s)
    Va0_si  = keep_VA0.to_value(u.m / u.s)
    Zp_si   = np.sqrt(keep_quant).to_value(u.m / u.s)
    rho_si  = keep_rho.to_value(u.kg / u.m ** 3)

    # ── physics transforms ────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        eta  = (Va0_si / V0_si) ** 2
    gp2 = (Zp_si * (1.0 + np.sqrt(eta)) / np.power(eta, 0.25)) ** 2

    # validity before logs
    valid = (
        (r_si > 0) & np.isfinite(r_si) &
        (Va0_si > 0) & np.isfinite(Va0_si) &
        (gp2 > 0) & np.isfinite(gp2) &
        np.isfinite(rho_si)
    )
    if not np.any(valid):
        raise ValueError("No valid samples after sanity checks (r>0, Va>0, gp2>0, finite).")

    sorter   = np.argsort(r_si[valid])
    r_sorted = r_si[valid][sorter]
    d_sorted = (r_sorted * u.m).to_value(u.au)
    Va_sorted  = Va0_si[valid][sorter]
    gp2_sorted = gp2[valid][sorter]
    rho_sorted = rho_si[valid][sorter]
    eta_sorted = eta[valid][sorter]

    ln_r_all  = np.log(r_sorted)
    ln_Va_all = np.log(Va_sorted)

    # fit selection
    fit_sel = (d_sorted >= R_min) & (d_sorted <= R_max)
    if fit_sel.sum() < max(2, poly_deg + 1):
        fit_sel = np.isfinite(ln_r_all) & np.isfinite(ln_Va_all)

    ln_r_fit  = ln_r_all[fit_sel]
    ln_Va_fit = ln_Va_all[fit_sel]
    deg_eff   = int(np.clip(poly_deg, 1, max(1, len(ln_r_fit) - 1)))

    # ── ALWAYS build log-binned diagnostics via func.binned_quantity ──────────
    # (do not replace this with any custom binning)
    avg_r, avg_Va, _, avg_counts = func.binned_quantity(
        r_sorted, Va_sorted, bins=n_bins + 1, return_counts=True
    )
    avg_d = avg_r / AU_SI

    # ── fit ln(Va) vs ln(r) ───────────────────────────────────────────────────
    cov_desc = None
    if use_binning:
        bin_sel = (avg_counts > 0) & np.isfinite(avg_Va) & np.isfinite(avg_r)
        x_b = np.log(avg_r[bin_sel])
        y_b = np.log(avg_Va[bin_sel])
        if len(x_b) < deg_eff + 1:
            use_binning = False
        else:
            # keep your original weights by counts
            w_b = avg_counts[bin_sel].astype(float)
            coef_desc, cov_desc = np.polyfit(x_b, y_b, deg=deg_eff, w=w_b, cov=True)
    if not use_binning:
        # polynomial design matrix for ln(r): columns [1, x, x^2, ..., x^deg]
        X = np.column_stack([ln_r_fit**k for k in range(deg_eff + 1)])
        if hetero_fit.lower() == "ols":
            mod = sm.OLS(ln_Va_fit, X).fit()
            params_asc = mod.params                  # ndarray
            cov_asc    = np.asarray(mod.cov_params())  # <-- FIX: no .values
        elif hetero_fit.lower() == "fgls":
            ols   = sm.OLS(ln_Va_fit, X).fit()
            resid = ln_Va_fit - ols.fittedvalues
            eps   = 1e-12
            var_m = sm.OLS(np.log(resid**2 + eps), X).fit()
            weights = 1.0 / np.exp(var_m.fittedvalues)  # 1/Var
            wls = sm.WLS(ln_Va_fit, X, weights=weights).fit()
            params_asc = wls.params
            cov_asc    = np.asarray(wls.cov_params())   # <-- FIX: no .values
        else:
            raise ValueError("hetero_fit must be 'ols' or 'fgls'.")

        # reorder to descending for np.polyval / polyder
        coef_desc = params_asc[::-1]
        cov_desc  = cov_asc[::-1, ::-1]

    # model & derivative on sorted grid
    ln_Va_model = np.polyval(coef_desc, ln_r_all)
    Va_model    = np.exp(ln_Va_model)
    dcoeff      = np.polyder(coef_desc)
    dlnV        = np.polyval(dcoeff, ln_r_all)
    dVa_dr      = Va_model * dlnV / r_sorted

    # interpolate back to full original r-grid
    deriv_all = np.full_like(r_si, np.nan, dtype=float)
    fac_all   = np.full_like(r_si, np.nan, dtype=float)
    ln_to_der = interp1d(ln_r_all, dVa_dr, kind="linear", fill_value="extrapolate", assume_sorted=True)
    ln_to_fac = interp1d(
        ln_r_all,
        (-(rho_sorted / 4.0) * gp2_sorted / (1.0 + np.sqrt(eta_sorted))),
        kind="linear", fill_value="extrapolate", assume_sorted=True
    )
    deriv_all[valid] = ln_to_der(np.log(r_si[valid]))
    fac_all[valid]   = ln_to_fac(np.log(r_si[valid]))
    Q_full = fac_all * deriv_all  # SI: W/m^3

    # scale height H = Va / (dVa/dr), in R_sun
    with np.errstate(divide="ignore", invalid="ignore"):
        H_full_m = keep_VA0.to_value(u.m/u.s) / deriv_all
    H_Rsun_full = H_full_m / R_SUN_SI

    # ── output units ──────────────────────────────────────────────────────────
    units_l = units.lower()
    if units_l == "si":
        Q_out, out_unit = Q_full, "W / m^3"
    elif units_l == "cgs":
        rho_val = keep_rho.to_value(u.kg / u.m ** 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q_mass = np.divide(Q_full, rho_val, out=np.full_like(Q_full, np.nan), where=np.isfinite(rho_val))
        Q_out = (Q_mass * (u.W / u.kg)).to(u.erg / (u.g * u.s)).value
        out_unit = "erg / (g s)"
    elif units_l == "cgs_vol":
        Q_out = (Q_full * (u.W / u.m ** 3)).to(u.erg / (u.cm ** 3 * u.s)).value
        out_unit = "erg / (cm^3 s)"
    else:
        raise ValueError("units must be 'SI', 'cgs', or 'cgs_vol'.")

    # LaTeX for ln(V_A) polynomial (descending coef order)
    def _latex_poly(coef: np.ndarray, var: str, expr: str, logy: bool) -> str:
        if coef is None:
            return ""
        d = len(coef) - 1
        parts = []
        for i, c in enumerate(coef):
            p = d - i
            mon = f"{abs(c):.3g}{var}^{p}" if p>1 else (f"{abs(c):.3g}{var}" if p==1 else f"{abs(c):.3g}")
            parts.append(("+" if (c>=0 and i) else "-")+mon if i else (f"-{mon}" if c<0 else mon))
        return (rf"$\ln({expr})={' '.join(parts)}$") if logy else (rf"${expr}={' '.join(parts)}$")

    latex = _latex_poly(coef_desc, 'x', r'V_A', True)

    # smooth fit curve across observed range in AU
    fit_x = np.logspace(np.log10(max(d_sorted.min(), 1e-12)), np.log10(d_sorted.max()), 200)
    r_fit = fit_x * AU_SI
    fit_y = np.exp(np.polyval(coef_desc, np.log(r_fit)))  # Va model [m/s]

    # ALWAYS include log-binned data from func.binned_quantity
    return {
        "Ma": keep_Ma,
        "x":  df[col_r_au].values,    # AU (raw)
        "y":  Q_out,                  # heating rate
        "y_err": np.zeros_like(Q_out),
        "units": out_unit,
        "deriv": deriv_all,           # dVa/dr [1/s]
        "scale_height": H_Rsun_full,  # H in R_sun
        "fits": {
            "latex":   latex,
            "coef":    coef_desc,     # descending order
            "cov":     cov_desc,      # descending order (or None if no cov)
            "order":   deg_eff,
            "fit_x":   fit_x,         # AU
            "fit_y":   fit_y,         # m/s
            "avg_x":   avg_d,         # AU
            "avg_y":   avg_Va,        # m/s
            "weights": avg_counts
        },
    }


# def CH_09_cascade_rate(
#     df,
#     *,
#     col_Ma   = "Ma_r",
#     col_r_au = "d",     # AU
#     col_V0   = "Vsw",    # km/s
#     col_VA0  = "Va",    # km/s
#     col_Zp2  = "Zp_amp",# (km/s)^2 if quant_is_squared=True, else km/s
#     col_rho  = "rho",   # kg/m^3
#     units: str    = "SI",
#     n_bins: int   = 100,
#     poly_deg: int = 2,
#     R_min: float  = 0.05,   # AU
#     R_max: float  = 0.30,   # AU
#     use_binning: bool   = True,
#     hetero_fit: str     = "fgls",
#     quant_is_squared: bool = True,
# ):
#     keep_Ma  = df[col_Ma].values
#     keep_d   = (df[col_r_au].values * u.au)
#     keep_V0  = (df[col_V0].values   * u.km / u.s)
#     keep_VA0 = (df[col_VA0].values  * u.km / u.s)

#     if quant_is_squared:
#         keep_quant = df[col_Zp2].values * (u.km/u.s)**2
#     else:
#         keep_quant = (df[col_Zp2].values * (u.km/u.s))**2

#     keep_rho = (df[col_rho].values * (u.kg/u.cm**3)).to(u.kg/u.m**3)

#     AU_m    = u.au.to(u.m)              # [m per AU]
#     R_sun_m = (1.0 * u.au / 215.032).to(u.m)  # ← FIX: Rsun from AU (no import)

#     r_si   = keep_d.to(u.m).value
#     V_si   = keep_V0.to(u.m/u.s).value
#     VA_si  = keep_VA0.to(u.m/u.s).value
#     Zp2_q  = keep_quant.to(u.m**2/u.s**2)
#     rho_SI = keep_rho

#     eta    = (VA_si / V_si)**2
#     valid  = (r_si > 0) & np.isfinite(r_si) & np.isfinite(VA_si) & np.isfinite(V_si) & (eta > 0)
#     if not np.any(valid):
#         raise RuntimeError("CH_09_cascade_rate: no valid samples after basic checks.")

#     sorter = np.argsort(r_si[valid])
#     r_s   = r_si[valid][sorter]
#     VA_s  = VA_si[valid][sorter]

#     avg_r, avg_VA, _, counts = func.binned_quantity(r_s, VA_s, bins=n_bins + 1, return_counts=True)
#     avg_x   = avg_r / AU_m
#     avg_y   = avg_VA
#     weights = counts

#     ln_r_all  = np.log(r_s)
#     ln_VA_all = np.log(VA_s)
#     d_s_au    = r_s / AU_m
#     sel       = (d_s_au >= R_min) & (d_s_au <= R_max)

#     if use_binning:
#         mask      = (avg_x >= R_min) & (avg_x <= R_max) & (weights > 0) & np.isfinite(avg_y)
#         ln_r_fit  = np.log(avg_r[mask])
#         ln_VA_fit = np.log(avg_VA[mask])
#         w_fit     = weights[mask]
#         deg       = poly_deg
#         coeff     = np.polyfit(ln_r_fit, ln_VA_fit, deg, w=w_fit)
#     else:
#         ln_r_fit  = ln_r_all[sel]
#         ln_VA_fit = ln_VA_all[sel]
#         X = sm.add_constant(ln_r_fit)
#         if hetero_fit.lower() == "ols":
#             mod = sm.OLS(ln_VA_fit, X).fit()
#             slope, intercept = mod.params[1], mod.params[0]
#         elif hetero_fit.lower() == "fgls":
#             ols   = sm.OLS(ln_VA_fit, X).fit()
#             resid = ln_VA_fit - ols.fittedvalues
#             varmd = sm.OLS(np.log(resid**2 + 1e-300), X).fit()
#             wts   = 1.0/np.exp(varmd.fittedvalues)
#             wls   = sm.WLS(ln_VA_fit, X, weights=wts).fit()
#             slope, intercept = wls.params[1], wls.params[0]
#         else:
#             raise ValueError("hetero_fit must be 'ols' or 'fgls'")
#         coeff = np.array([slope, intercept])
#         deg   = len(coeff) - 1

#     dpoly     = np.polyder(coeff)
#     dlnVA_dr  = np.polyval(dpoly, ln_r_all) / r_s
#     dVA_dr_s  = np.exp(np.polyval(coeff, ln_r_all)) * dlnVA_dr

#     deriv_all        = np.full_like(r_si, np.nan, dtype=float)
#     interp_d         = interp1d(ln_r_all, dVA_dr_s, fill_value="extrapolate")
#     deriv_all[valid] = interp_d(np.log(r_si[valid]))
#     dVA_dr_q         = deriv_all * (1/u.s)

#     Q_q = (
#         -0.25
#         * (keep_V0.to(u.m/u.s) + keep_VA0.to(u.m/u.s)) / keep_VA0.to(u.m/u.s)
#         * dVA_dr_q * rho_SI * Zp2_q
#     ).to(u.W/u.m**3)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         H_q = (keep_VA0.to(u.m/u.s) / dVA_dr_q).to(u.m) / R_sun_m  # dimensionless in R_sun

#     fit_x = np.logspace(np.log10(max(R_min, 1e-6)), np.log10(max(R_max, R_min + 1e-6)), 100)
#     r_fit = fit_x * AU_m
#     lnVA_fit_curve = np.polyval(coeff, np.log(r_fit))
#     fit_y = np.exp(lnVA_fit_curve)

#     if units.lower() == "si":
#         Q_out, out_u = Q_q.value, "W / m3"
#     elif units.lower() == "cgs":
#         Qm    = (Q_q / rho_SI).to(u.erg/(u.g*u.s))
#         Q_out, out_u = Qm.value, "erg / (g s)"
#     elif units.lower() == "cgs_vol":
#         Qv    = Q_q.to(u.erg/(u.cm**3*u.s))
#         Q_out, out_u = Qv.value, "erg / (cm3 s)"
#     else:
#         raise ValueError("units must be 'SI','cgs', or 'cgs_vol'")

#     return {
#         "Ma": df[col_Ma].values,
#         "x":    df[col_r_au].values,     # AU
#         "y":    Q_out,
#         "y_err": np.zeros_like(Q_out),
#         "units": out_u,
#         "deriv": dVA_dr_q.to_value(1/u.s),    # 1/s
#         "scale_height": H_q.to_value(u.one),  # in R_sun
#         "fits": {
#             "fit_x":   fit_x,    # AU
#             "fit_y":   fit_y,    # m/s
#             "avg_x":   avg_x,    # AU
#             "avg_y":   avg_y,    # m/s
#             "weights": weights,
#             "order":   deg,
#         },
#     }


# def CH_09_cascade_rate(
#     df,
#     *,
#     col_Ma   = "Ma_r",
#     col_r_au = "d",       # AU
#     col_V0   = "Vsw",     # km/s
#     col_VA0  = "Va",      # km/s
#     col_Zp2  = "Zp_amp",  # (km/s)^2 if quant_is_squared=True, else km/s
#     col_rho  = "rho",     # NOTE: original function assumed kg/cm^3 when no units are provided
#     units: str    = "SI",
#     n_bins: int   = 100,
#     poly_deg: int = 1,
#     R_min: float  = 0.05,
#     R_max: float  = 0.3,
#     use_binning: bool = True,
#     hetero_fit: str = "fgls",
#     quant_is_squared: bool = True,
# ):
#     # --- Map DataFrame columns to the original inputs (no other changes below) ---
#     keep_Ma  = df[col_Ma].values
#     keep_d   = df[col_r_au].values
#     keep_V0  = df[col_V0].values
#     keep_VA0 = df[col_VA0].values
#     keep_rho = df[col_rho].values
#     if quant_is_squared:
#         keep_quant = df[col_Zp2].values
#     else:
#         keep_quant = (df[col_Zp2].values)**2

#     # ─────────────────────────────────────────────────────────────────────────────
#     # ORIGINAL BODY (unchanged)
#     # ─────────────────────────────────────────────────────────────────────────────
#     if not isinstance(keep_d, u.Quantity):
#         keep_d = keep_d * u.AU
#     if not isinstance(keep_V0, u.Quantity):
#         keep_V0 = keep_V0 * (u.km / u.s)
#     if not isinstance(keep_VA0, u.Quantity):
#         keep_VA0 = keep_VA0 * (u.km / u.s)
#     if not isinstance(keep_quant, u.Quantity):
#         keep_quant = keep_quant * (u.km ** 2 / u.s ** 2)
#     if not isinstance(keep_rho, u.Quantity):
#         keep_rho = keep_rho * (u.kg / u.cm ** 3)

#     # AU_SI = 1.496e11
#     # R_SUN_SI = R_sun.to_value(u.m)

#     AU_SI    = u.au.to(u.m)              # [m per AU]
#     R_SUN_SI = (1.0 * u.au / 215.032).to(u.m)  # ← FIX: Rsun from AU (no import)


#     r_si    = keep_d.to_value(u.m)
#     V0_si   = keep_V0.to_value(u.m / u.s)
#     Va0_si  = keep_VA0.to_value(u.m / u.s)
#     Zp_si   = np.sqrt(keep_quant).to_value(u.m / u.s)
#     rho_si  = keep_rho.to_value(u.kg / u.m ** 3)

#     eta   = (Va0_si / V0_si) ** 2
#     gp2   = (Zp_si * (1 + np.sqrt(eta)) / eta ** 0.25) ** 2
#     valid = (r_si > 0) & np.isfinite(r_si) & (gp2 > 0) & np.isfinite(gp2)

#     sorter     = np.argsort(r_si[valid])
#     r_sorted   = r_si[valid][sorter]
#     d_sorted   = keep_d.to_value(u.AU)[valid][sorter]
#     Va_sorted  = Va0_si[valid][sorter]
#     gp2_sorted = gp2[valid][sorter]
#     rho_sorted = rho_si[valid][sorter]
#     eta_sorted = eta[valid][sorter]

#     ln_r_all  = np.log(r_sorted)
#     ln_Va_all = np.log(Va_sorted)
#     fit_sel   = (d_sorted >= R_min) & (d_sorted <= R_max)
#     ln_r_fit  = ln_r_all[fit_sel]
#     ln_Va_fit = ln_Va_all[fit_sel]

#     if use_binning:
#         avg_r, avg_Va, _, avg_counts = func.binned_quantity(
#             r_sorted, Va_sorted, bins=n_bins + 1, return_counts=True
#         )
#         avg_d = avg_r / AU_SI
#         bin_sel = (avg_counts > 0) & np.isfinite(avg_Va) & (avg_d >= R_min) & (
#             avg_d <= R_max
#         )
#         x_b = np.log(avg_r[bin_sel])
#         y_b = np.log(avg_Va[bin_sel])
#         w_b = avg_counts[bin_sel]
#         deg = poly_deg
#         coeff = np.polyfit(x_b, y_b, deg, w=w_b)
#     else:

#         avg_r, avg_Va, _, avg_counts = func.binned_quantity(
#             r_sorted, Va_sorted, bins=n_bins + 1, return_counts=True
#         )
#         avg_d = avg_r / AU_SI

#         deg = poly_deg
#         X = sm.add_constant(ln_r_fit)
#         if hetero_fit.lower() == "ols":
#             mod = sm.OLS(ln_Va_fit, X).fit()
#             coeff = [mod.params[1], mod.params[0]]
#         elif hetero_fit.lower() == "fgls":
#             ols = sm.OLS(ln_Va_fit, X).fit()
#             resid = ln_Va_fit - ols.fittedvalues
#             var_m = sm.OLS(np.log(resid ** 2), X).fit()
#             weights = 1 / np.exp(var_m.fittedvalues)
#             wls = sm.WLS(ln_Va_fit, X, weights=weights).fit()
#             coeff = [wls.params[1], wls.params[0]]
#         else:
#             raise ValueError("hetero_fit must be 'ols' or 'fgls'")

#     ln_Va_model = np.polyval(coeff, ln_r_all)
#     Va_model    = np.exp(ln_Va_model)
#     dcoeff      = np.polyder(coeff)
#     dlnV        = np.polyval(dcoeff, ln_r_all)
#     dVa_dr      = Va_model * dlnV / r_sorted

#     deriv_all   = np.full_like(r_si, np.nan)
#     fac_all     = np.full_like(r_si, np.nan)
#     ln_to_der   = interp1d(ln_r_all, dVa_dr, fill_value="extrapolate")
#     ln_to_fac   = interp1d(
#         ln_r_all,
#         (-(rho_sorted / 4) * gp2_sorted / (1 + np.sqrt(eta_sorted))),
#         fill_value="extrapolate",
#     )
#     deriv_all[valid] = ln_to_der(np.log(r_si[valid]))
#     fac_all[valid] = ln_to_fac(np.log(r_si[valid]))

#     Q_full = fac_all * deriv_all

#     with np.errstate(divide="ignore", invalid="ignore"):
#         H_full = Va0_si / deriv_all
#     Hinv_Rsun_full = H_full / R_SUN_SI

#     # The original function takes a 'fit_method' arg; since the new signature
#     # matches the second function, we default to the original behavior ("poly").
#     fit_method = "poly"

#     fit_x = np.logspace(np.log10(d_sorted.min()), np.log10(d_sorted.max()), 100)
#     r_fit = fit_x * AU_SI
#     if fit_method.lower() == "rolling":
#         ln_i = interp1d(ln_r_all, ln_Va_model, fill_value="extrapolate")
#         fit_y = np.exp(ln_i(np.log(r_fit)))
#     else:
#         fit_y = np.exp(np.polyval(coeff, np.log(r_fit)))

#     if units.lower() == "si":
#         Q_out, out_unit = Q_full, "W / m3"
#     elif units.lower() == "cgs":
#         rho_val = keep_rho.to_value(u.kg / u.m ** 3)
#         Q_mass = Q_full / rho_val
#         Q_out = (Q_mass * (u.W / u.kg)).to(u.erg / (u.g * u.s)).value
#         out_unit = "erg / (g s)"
#     elif units.lower() == "cgs_vol":
#         Q_out = (Q_full * (u.W / u.m ** 3)).to(u.erg / (u.cm ** 3 * u.s)).value
#         out_unit = "erg / (cm3 s)"
#     else:
#         raise ValueError("units must be 'SI', 'cgs', or 'cgs_vol'.")

#     return {
#         "Ma": df[col_Ma].values,
#         "x": df[col_r_au].values,
#         "y": Q_out,
#         "y_err": np.zeros_like(Q_out),
#         "units": out_unit,
#         "deriv": deriv_all,
#         "scale_height": Hinv_Rsun_full,
#         "fits": {
#             "fit_x": fit_x,
#             "fit_y": fit_y,
#             "avg_x": avg_d,
#             "avg_y": avg_Va,
#             "weights": avg_counts,
#             "order": deg,
#         },
#     }


# ─────────────────────────────────────────────────────────────────────────────
# 2) f_w_gradient — same math/outputs, now reads directly from a DataFrame
#    No writes to the DataFrame; arrays only.
# ─────────────────────────────────────────────────────────────────────────────
def f_w_gradient(
    df,
    *,
    # column mapping
    col_Ma   = "Ma_r",
    col_r_au = "d",
    col_du2  = "v_amp",     # interpreted exactly as before: numbers × (km/s)^2
    col_dva2 = "va_amp",    # interpreted exactly as before: numbers × (km/s)^2
    col_rho  = "rho",
    # options identical to the original function
    units: str    = "SI",
    n_bins: int   = 100,
    poly_deg: int = 1,
    statistic: str= "mean",
    R_min: float  = 0.05,
    R_max: float  = 0.30,
    # preserve previous behavior: treat col_du2/col_dva2 numbers as already
    # in km^2/s^2 (no squaring).
    values_are_squared: bool = True,
):
    """
    CH09-style evaluation of the wave-pressure-gradient force density:

        f_w = ρ/r·(⟨δu²⟩−⟨δv_A²⟩) − d/dr[½ρ⟨δv_A²⟩] .

    Returns structure identical to the original `f_w_gradient`.
    """

    r_m_q = (df[col_r_au].values * u.au).to(u.m)
    if values_are_squared:
        du2_q  = df[col_du2].values  * (u.km/u.s)**2
        va2_q  = df[col_dva2].values * (u.km/u.s)**2
    else:
        du2_q  = (df[col_du2].values  * (u.km/u.s))**2
        va2_q  = (df[col_dva2].values * (u.km/u.s))**2

    # match original: interpret numerics as kg/cm^3 before converting
    rho_q  = df[col_rho].values * (u.kg/u.cm**3)

    var_q  = 0.5 * rho_q * va2_q   # N m^-2

    msk = np.isfinite(var_q)
    if not np.any(msk):
        raise RuntimeError("f_w_gradient_df: no finite data points.")

    order = np.argsort(r_m_q[msk].value)
    r_sorted_m   = r_m_q[msk].value[order]
    var_sorted_v = var_q[msk].value[order]

    mids_m, avg_var, _, counts = func.binned_quantity(
        r_sorted_m, var_sorted_v, bins=n_bins + 1, return_counts=True
    )

    AU_SI   = (1 * u.au).to_value(u.m)
    avg_x_au= mids_m / AU_SI

    sel = (counts > 0) & np.isfinite(avg_var) & (avg_x_au >= R_min) & (avg_x_au <= R_max)
    if not np.any(sel):
        sel = (counts > 0) & np.isfinite(avg_var)

    ln_r_fit = np.log(mids_m[sel])
    ln_v_fit = np.log(avg_var[sel])
    deg      = min(poly_deg, max(len(ln_r_fit) - 1, 1))
    coeff    = np.polyfit(ln_r_fit, ln_v_fit, deg, w=counts[sel])

    ln_r_s    = np.log(r_sorted_m)
    var_mod_v = np.exp(np.polyval(coeff, ln_r_s))
    dvar_dr_v = var_mod_v * np.polyval(np.polyder(coeff), ln_r_s) / r_sorted_m
    dvar_dr_q = dvar_dr_v * (var_q.unit / r_m_q.unit)  # N m^-3

    dvar_full_q = interp1d(ln_r_s, dvar_dr_q.value, fill_value="extrapolate")(
        np.log(r_m_q.value)
    ) * dvar_dr_q.unit

    first_term_q = (rho_q / r_m_q) * (du2_q - va2_q)  # N m^-3
    fw_q         = first_term_q - dvar_full_q         # N m^-3

    if units.lower() == "si":
        fw_out, out_unit = fw_q.to(u.N/u.m**3).value, "N / m3"
    elif units.lower() == "cgs":
        fw_out, out_unit = fw_q.to(u.dyne/u.cm**3).value, "dyn / cm3"
    else:
        raise ValueError("units must be 'SI' or 'cgs'.")

    fit_x_au = np.logspace(np.log10(avg_x_au[avg_x_au > 0].min()),
                           np.log10(avg_x_au.max()), 100)
    fit_y    = (np.exp(np.polyval(coeff, np.log(fit_x_au * AU_SI))) * var_q.unit).value

    return {
        "Ma": df[col_Ma].values,
        "x":    df[col_r_au].values,           # AU
        "y":    fw_out,
        "y_err": np.zeros_like(fw_out),
        "units": out_unit,
        "deriv": dvar_full_q.to(u.N/u.m**3).value,
        "fits": {
            "fit_x":   fit_x_au,               # AU
            "fit_y":   fit_y,                  # N m^-2
            "avg_x":   avg_x_au,               # AU
            "avg_y":   avg_var,                # N m^-2
            "weights": counts,
            "order":   deg,
        },
    }


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

##########################################

import numpy as np
from scipy.signal import stft, istft, medfilt


def _interp_nans_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return x
    ok = np.isfinite(x)
    if ok.all():
        return x
    if ok.sum() < 2:
        return np.zeros_like(x)
    idx = np.arange(n)
    y = x.copy()
    y[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return y


def _find_runs(idx: np.ndarray):
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)


def remove_wheel_noise(
    x: np.ndarray,
    fs: float,
    *,
    # ---- STFT geometry
    freq_min: float = 8.0,
    stft_nperseg: int = 2048,
    stft_overlap: float = 0.5,

    # ---- candidate detection (global in time)
    percentile_q: float = 99.5,
    kernel: int = 301,
    thresh_db: float = 3.0,
    merge_hz: float = 0.20,
    max_lines: int = 2000,

    # ---- OPTIONAL: slope flattening (helps “dense forest” high-f spectra)
    whiten_exp: float = 0.0,        # try 8/3 if needed

    # ---- drift tracking + removal
    track_half_width_hz: float = 4.0,
    remove_half_width_hz: float = 1.5,
    atten_db: float = 100.0,

    # ---- robustness
    fallback_top_k: int = 80,       # if detection yields nCand=0, force top-k candidates
    return_debug: bool = False,

    # ---- compatibility (ignore unknown keys safely)
    mad_mult=None,
    **_ignored,
):
    """
    Drift-safe coherent-line removal for SCM wheel/electronics tones.

    Simple description:
      1) STFT -> S(f,t)
      2) Build a 1D candidate spectrum Sq(f) = percentile over time
      3) Score(f) = dB above a smooth baseline (median filter)
      4) Pick candidate line bands
      5) For each candidate, track the ridge in time (local argmax in S)
      6) Attenuate a frequency band around that ridge
      7) iSTFT
    """

    x = _interp_nans_1d(np.asarray(x, float))
    n = x.size
    if n < 64 or not np.isfinite(fs) or fs <= 0:
        return (x, {"candidates_hz": np.array([]), "mask_frac": 0.0}) if return_debug else x

    nperseg = int(min(max(256, stft_nperseg), n))
    noverlap = int(nperseg * float(stft_overlap))
    noverlap = min(max(0, noverlap), nperseg - 1)

    # IMPORTANT: boundary/padded -> avoids NOLA/invertibility issues
    f, tt, Z = stft(
        x,
        fs=float(fs),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )

    if f.size < 2 or Z.size == 0:
        return (x, {"candidates_hz": np.array([]), "mask_frac": 0.0}) if return_debug else x

    df = float(f[1] - f[0])
    nyq = 0.5 * fs

    S = np.abs(Z) ** 2  # (nf, nt)

    # ---------------------------
    # (1) GLOBAL candidate spectrum Sq(f)
    # ---------------------------
    Sq = np.percentile(S, float(percentile_q), axis=1)

    eps = 1e-30
    logSq = np.log10(np.maximum(Sq, eps))

    # Optional whitening (flatten power-law slopes)
    if whiten_exp != 0.0:
        ff = np.maximum(f, 1e-6)
        logSq = logSq + float(whiten_exp) * np.log10(ff)

    # Smooth baseline across frequency
    k = int(kernel)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    if k > logSq.size:
        k = logSq.size if (logSq.size % 2 == 1) else max(3, logSq.size - 1)

    base = medfilt(logSq, kernel_size=k)
    score = 10.0 * (logSq - base)  # dB above baseline

    valid = (f >= float(freq_min)) & (f < nyq)

    if not np.any(valid):
        return (x, {"candidates_hz": np.array([]), "mask_frac": 0.0}) if return_debug else x

    score_valid = score[valid]
    max_score_db = float(np.max(score_valid))

    # Primary thresholding
    cand = np.where(valid & (score > float(thresh_db)))[0]

    # If nothing passes threshold, force candidates from the strongest bins
    if cand.size == 0 and fallback_top_k is not None and int(fallback_top_k) > 0:
        idx_valid = np.where(valid)[0]
        ktop = min(int(fallback_top_k), idx_valid.size)
        top_idx = idx_valid[np.argpartition(score[idx_valid], -ktop)[-ktop:]]
        cand = np.sort(top_idx)

    if cand.size == 0:
        # absolutely nothing usable -> do nothing
        dbg = {
            "candidates_hz": np.array([]),
            "mask_frac": 0.0,
            "nCand": 0,
            "fs": float(fs),
            "df_hz": df,
            "max_score_db": max_score_db,
            "thresh_db": float(thresh_db),
        }
        return (x, dbg) if return_debug else x

    # Group contiguous bins and pick max per run
    runs = _find_runs(cand)

    cand_bins = []
    cand_heights = []
    for run in runs:
        j = run[np.argmax(score[run])]
        cand_bins.append(j)
        cand_heights.append(score[j])

    cand_bins = np.array(cand_bins, dtype=int)
    cand_heights = np.array(cand_heights, dtype=float)

    order = np.argsort(cand_heights)[::-1]
    cand_bins = cand_bins[order]
    cand_heights = cand_heights[order]

    cand_freqs = f[cand_bins]

    # Merge close candidates
    keep = []
    for i, fi in enumerate(cand_freqs):
        if len(keep) == 0:
            keep.append(i)
        else:
            if np.min(np.abs(fi - cand_freqs[keep])) > float(merge_hz):
                keep.append(i)

    cand_bins = cand_bins[keep]
    cand_freqs = f[cand_bins]

    if cand_bins.size > int(max_lines):
        cand_bins = cand_bins[: int(max_lines)]
        cand_freqs = cand_freqs[: int(max_lines)]

    # ---------------------------
    # (2) TRACK ridges in time + build mask
    # ---------------------------
    track_bins = int(np.ceil(float(track_half_width_hz) / df))
    rm_bins = int(np.ceil(float(remove_half_width_hz) / df))

    track_bins = max(1, track_bins)
    rm_bins = max(1, rm_bins)

    nf, nt = S.shape
    mask = np.zeros((nf, nt), dtype=bool)
    cols = np.arange(nt)

    for k0 in cand_bins:
        a = max(0, k0 - track_bins)
        b = min(nf - 1, k0 + track_bins)

        band = S[a:b + 1, :]  # (band_nf, nt)
        local_argmax = np.argmax(band, axis=0) + a  # (nt,)

        # Vectorized band marking (no inner time-loop)
        for off in range(-rm_bins, rm_bins + 1):
            rr = np.clip(local_argmax + off, 0, nf - 1)
            mask[rr, cols] = True

    # Attenuate coefficients in mask
    atten = 10.0 ** (-float(atten_db) / 20.0)
    Zc = Z.copy()
    Zc[mask] *= atten

    # Invert STFT
    _, y = istft(
        Zc,
        fs=float(fs),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=True,
    )

    # Match length
    if y.size > n:
        y = y[:n]
    elif y.size < n:
        y = np.pad(y, (0, n - y.size))

    if return_debug:
        dbg = {
            "candidates_hz": cand_freqs,
            "mask_frac": float(mask.mean()),
            "nCand": int(cand_freqs.size),
            "fs": float(fs),
            "df_hz": df,
            "max_score_db": max_score_db,
            "thresh_db": float(thresh_db),
        }
        return y, dbg

    return y


from scipy.signal import stft, istft
def remove_wheel_noise_scam_PSP(data,
                       fs, 
                       window_size      = 2**15, 
                       avg_length       = 1,
                       power_threshold  = 6.0,
                       freq_min         = 10.0, 
                       hampel_wind      = 51,
                       hampel_thresh    = 3.5):
    """
    Remove noise tones from a time series of magnetic field measurements.

    Parameters:
    data : numpy.ndarray
        Input time series data.
    fs : float
        Sampling frequency in Hz.
    window_size : int, optional
        Window size for STFT (number of data points). Default is 2**15.
    avg_length : int, optional
        Number of consecutive STFT windows to average. Default is 5.
    power_threshold : float, optional
        Power threshold factor to detect noise frequencies. Default is 3.0.
    freq_min : float, optional
        Minimum frequency (Hz) to consider for noise detection. Default is 1.0 Hz.

    Returns:
    cleaned_data : numpy.ndarray
        Time series data with noise tones removed.
    extracted_noise : numpy.ndarray
        Time series of the extracted noise.
    """
    # Compute STFT (removed boundary=None to use default padding)
    f, t_stft, Zxx = stft(data, fs=fs, window='hann', nperseg=window_size,
                          noverlap=window_size//2)

    print(np.shape(f), np.shape(Zxx))
    # Compute power spectrum (magnitude squared)
    power_spectrum = (np.abs(Zxx).T**2*f**(8/3)).T
    #power_spectrum = np.abs(Zxx)**2

    # Average spectra: For every avg_length consecutive windows, average the spectra
    n_segments = Zxx.shape[1]
    n_groups = n_segments // avg_length
    if n_groups == 0:
        raise ValueError("Not enough data to form at least one group for averaging spectra. "
                         "Reduce avg_length or provide more data.")

    averaged_power_spectra = []
    for i in range(n_groups):
        start_idx = i * avg_length
        end_idx   = start_idx + avg_length
        avg_power = np.mean(power_spectrum[:, start_idx:end_idx], axis=1)
        averaged_power_spectra.append(avg_power)

    averaged_power_spectra = np.array(averaged_power_spectra)  # Shape: (n_groups, n_frequencies)

    # Compute background spectrum by averaging over all averaged spectra
   # background_spectrum = np.mean(averaged_power_spectra, axis=0)  # Shape: (n_frequencies,)

    # Identify noise frequencies where the power exceeds threshold times the background spectrum
    noise_mask = np.zeros_like(Zxx, dtype=bool)
    for i in range(n_groups):
        start_idx   = i * avg_length
        end_idx     = start_idx + avg_length
        group_power = power_spectrum[:, start_idx:end_idx]  # Shape: (n_frequencies, avg_length)

        # Thresholding
        hamp     = func.hampel(averaged_power_spectra[i], hampel_wind, hampel_thresh)
        threshold = power_threshold *hamp[0][:, np.newaxis]  # Shape: (n_frequencies, 1)
        # Identify noise frequencies
        noise_frequencies = (group_power > threshold) & (f[:, np.newaxis] >= freq_min)  # Shape: (n_frequencies, avg_length)

        # Assign to noise_mask
        noise_mask[:, start_idx:end_idx] = noise_frequencies

    # Use the inverse STFT on the identified noise coefficients to create a noise time series
    noise_Zxx = np.zeros_like(Zxx, dtype=complex)
    noise_Zxx[noise_mask] = Zxx[noise_mask]

    # Generate noise time series
    _, noise_time_series = istft(noise_Zxx, fs=fs, window='hann', nperseg=window_size,
                                 noverlap=window_size//2)

    # Ensure the noise time series has the same length as original data
    if len(noise_time_series) > len(data):
        # Trim the reconstructed noise to match the original data length
        noise_time_series = noise_time_series[:len(data)]
    elif len(noise_time_series) < len(data):
        # Pad the reconstructed noise with zeros to match the original data length
        noise_time_series = np.pad(noise_time_series, (0, len(data) - len(noise_time_series)), 'constant')

    # Subtract the noise time series from the original data
    cleaned_data = data - noise_time_series

    return cleaned_data






def build_V_mod_TH(
    B, V,
    axes: list[str] | None = None,
    sc: str            = "PSP",
    window: str        = "30min",
    correct_sign: bool = True,
    consider_Va:  bool = True,
    consider_Vsc: bool = True,
    return_addit: bool = False
):
    import numpy as np
    import pandas as pd

    _C_ALFVEN = 21.82  # Va[km/s] = 21.82 * B[nT] / sqrt(n_p[cm^-3])

    if axes is None:
        axes = ["r","t","n"] if "Br" in B.columns else ["x","y","z"]

    is_rtn      = axes[0] == "r"
    base_cols   = [f"V{a}" for a in axes]
    B_cols      = [f"B{a}" for a in axes]
    sc_vel_cols = [f"sc_vel_{a}" for a in ("r","t","n")]

    n_p_raw = V["np"].astype(float)
    Np_background = n_p_raw.rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    Np_background[Np_background <= 0] = np.nan

    B_arr = B[B_cols].to_numpy(dtype=float)
    B_inst_mag = np.linalg.norm(B_arr, axis=1)
    B0_scalar = pd.Series(B_inst_mag, index=B.index).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)

    B0_vec = B[B_cols].rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    B0_vec_mag = np.linalg.norm(B0_vec, axis=1)
    B0_vec_mag[B0_vec_mag == 0] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        e_par = B0_vec / B0_vec_mag[:, None]

    V_arr = V[base_cols].to_numpy(dtype=float)

    if consider_Vsc and is_rtn and sc.upper() == "PSP":
        if all(c in V.columns for c in sc_vel_cols):
            V_sc_arr = V[sc_vel_cols].to_numpy(dtype=float)
            V_arr = V_arr - V_sc_arr

    V_rel_df = pd.DataFrame(V_arr, index=V.index, columns=base_cols)

    def _stable_perp_basis(epar: np.ndarray):
        ref1 = np.array([1.0, 0.0, 0.0])
        ref2 = np.array([0.0, 1.0, 0.0])
        dot1 = np.abs(np.einsum("ij,j->i", epar, ref1))
        ref  = np.where(dot1[:, None] < 0.9, ref1[None, :], ref2[None, :])
        e_p1 = np.cross(epar, ref)
        n1   = np.linalg.norm(e_p1, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            e_p1 = e_p1 / n1[:, None]
        e_p2 = np.cross(epar, e_p1)
        return e_p1, e_p2

    e_p1, e_p2 = _stable_perp_basis(e_par)

    v_par = np.einsum("ij,ij->i", V_arr, e_par)
    v_p1  = np.einsum("ij,ij->i", V_arr, e_p1)
    v_p2  = np.einsum("ij,ij->i", V_arr, e_p2)

    Vrel_perp_mag = np.sqrt(v_p1**2 + v_p2**2)
    Vrel_mag      = np.sqrt(v_par**2 + Vrel_perp_mag**2)

    Va_bulk = _C_ALFVEN * B0_scalar / np.sqrt(Np_background)
    Va_guide = _C_ALFVEN * B0_vec_mag / np.sqrt(Np_background)

    if is_rtn:
        Br_like = B0_vec[:, 0]
    else:
        Br_like = -B0_vec[:, 2]

    sigma = np.sign(Br_like)
    sigma[sigma == 0] = 1.0
    if not correct_sign:
        sigma = np.abs(sigma)

    Va_used = Va_guide if consider_Va else 0.0
    v_par_eff = v_par + sigma * Va_used
    V_mod_mag = np.sqrt(v_par_eff**2 + Vrel_perp_mag**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = v_par / Vrel_mag
    theta_VB = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    summary_df = pd.DataFrame(
        {
            "θ_VB": theta_VB,
            "|Va|": Va_bulk,
            "|V_rel|": Vrel_mag,
            "|V_mod_TH|": V_mod_mag,
            "B0_scalar": B0_scalar,
            "B0_vec_mag": B0_vec_mag,
            "Np_background": Np_background,
        },
        index=V.index,
    )

    if return_addit:
        V_MTH_extra = pd.DataFrame(
            {
                "Vrel_par": v_par,
                "Vrel_perp": Vrel_perp_mag,
                "sigma_Va": sigma * (Va_guide if consider_Va else 0.0),
            },
            index=V.index,
        )
        return summary_df, V_rel_df, V_MTH_extra

    return summary_df, V_rel_df


def build_V_mod_TH(
    B, V,
    axes: list[str] | None = None,
    sc: str            = "PSP",
    window: str        = "30min",
    correct_sign: bool = True,
    consider_Va:  bool = True,
    consider_Vsc: bool = True,
    return_addit: bool = False  # <--- This flag was present but unused
):
    """
    Computes Effective Advection Speeds (MTH) and returns vector components.
    """
    import numpy as np
    import pandas as pd

    # --- Constants ---
    _C_ALFVEN = 21.82 

    # --- Axes Setup ---
    if axes is None:
        axes = ["r","t","n"] if "Br" in B.columns else ["x","y","z"]
    is_rtn        = axes[0] == "r"
    base_cols     = [f"V{a}" for a in axes]
    sc_vel_cols   = [f"sc_vel_{a}" for a in ("r","t","n")]
    B_cols        = [f"B{a}" for a in axes]

    # --- 1. Background Fields Calculation ---
    # Rolling mean for Density and B-field to get consistent background
    n_p_raw = V["np"] 
    Np_mean = n_p_raw.rolling(window, center=True, min_periods=1).mean()

    # Vector Background (Direction)
    B0_vec_df  = B[B_cols].rolling(window, center=True, min_periods=1).median()
    B0_vec_arr = B0_vec_df.values.astype(float)
    
    # Scalar Background (Energy)
    B_inst_mag = np.linalg.norm(B[B_cols].values, axis=1)
    B0_scalar_series = pd.Series(B_inst_mag, index=B.index).rolling(window, center=True, min_periods=1).mean()
    B0_scalar_vals   = B0_scalar_series.values

    # --- 2. Frame Transformation (Get V_rel) ---
    V_arr  = V[base_cols].values.astype(float)
    
    if consider_Vsc and is_rtn and sc.upper() == "PSP":
        if all(c in V.columns for c in sc_vel_cols):
            V_sc_arr = V[sc_vel_cols].values.astype(float)
            V_arr    = V_arr - V_sc_arr  # V_rel = V_sw - V_sc
        else:
            print("Warning: PSP SC velocity columns missing. Using V_sw as V_rel.")
            
    # Save V_rel (V_sc_rem) for output
    V_rel_df = pd.DataFrame(V_arr, index=V.index, columns=base_cols)

    # --- 3. Basis Vectors (b_hat) ---
    B0_vec_mag = np.linalg.norm(B0_vec_arr, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        e_par = B0_vec_arr / B0_vec_mag[:, None]
    e_par[np.isnan(e_par)] = 0.0

    # Perp Basis
    ref = np.array([1.0, 0.0, 0.0])
    e_p1 = np.cross(e_par, ref)
    idx_deg = np.linalg.norm(e_p1, axis=1) < 1e-6
    if np.any(idx_deg):
        e_p1[idx_deg] = np.cross(e_par[idx_deg], np.array([0.0, 1.0, 0.0]))
    
    e_p1 /= np.linalg.norm(e_p1, axis=1)[:, None]
    e_p2 = np.cross(e_par, e_p1)

    # --- 4. Projections (The Core Physics) ---
    v_par = np.einsum("ij,ij->i", V_arr, e_par)  # V_rel_parallel
    v_p1  = np.einsum("ij,ij->i", V_arr, e_p1)
    v_p2  = np.einsum("ij,ij->i", V_arr, e_p2)
    
    Vrel_perp_mag = np.sqrt(v_p1**2 + v_p2**2)   # V_rel_perp
    Vrel_mag      = np.sqrt(v_par**2 + Vrel_perp_mag**2) 

    # --- 5. Alfvén Speed & Polarity ---
    Np_vals = Np_mean.values
    Np_vals[Np_vals <= 0] = np.nan
    
    Va_scalar_mag = _C_ALFVEN * B0_scalar_vals / np.sqrt(Np_vals)
    Va_scalar_mag[np.isnan(Va_scalar_mag)] = 0.0

    # Polarity (sigma)
    Br_like = B0_vec_arr[:, 0] if is_rtn else -B0_vec_arr[:, 2]
    sigma = np.sign(Br_like)
    sigma[sigma == 0] = 1.0
    if not correct_sign: sigma = np.abs(sigma)

    # MTH Velocity Magnitude
    # This accounts for the Doppler shift: V_eff = V_rel +/- V_A
    v_par_eff = v_par + (sigma * Va_scalar_mag if consider_Va else 0.0)
    V_mod_mag = np.sqrt(v_par_eff**2 + Vrel_perp_mag**2) 

    # --- 6. Theta_VB ---
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = v_par / Vrel_mag
    theta_VB = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # --- 7. Output ---
    summary_df = pd.DataFrame({
        "θ_VB"        : theta_VB,
        "|Va|"        : Va_scalar_mag, 
        "|V_rel|"     : Vrel_mag,  
        "|V_mod_TH|"  : V_mod_mag,
        "B0_scalar"   : B0_scalar_vals,
        "B0_vec_mag"  : B0_vec_mag,
        "Np_background": Np_vals
    }, index=V.index)

    # [FIX] Logic to return the 3rd DataFrame containing the components
    if return_addit:
        V_MTH_extra = pd.DataFrame({
            "Vrel_par":  v_par,          # Use this for V_rel,||
            "Vrel_perp": Vrel_perp_mag,  # Use this for V_rel,perp
            "sigma_Va":  sigma * Va_scalar_mag # The Alfvenic term
        }, index=V.index)
        return summary_df, V_rel_df, V_MTH_extra
    
    return summary_df, V_rel_df

def calculate_non_linearity_parameter(d_zp_lambda,
                                      d_zp_xi,
                                      d_zp_ell,
                                      d_zm_lambda,
                                      d_zm_xi,
                                      d_zm_ell,
                                      zp_lambda,
                                      zp_xi,
                                      zp_ell,
                                      zm_lambda,
                                      zm_xi,
                                      zm_ell,
                                      align_angle,
                                      Va,
                                      method = 'slinear'):
    # Interpolation functions
    interp_d_zm_lambda = interp1d(zm_lambda, d_zm_lambda, kind=method, bounds_error=False)
    interp_d_zm_ell    = interp1d(zm_ell, d_zm_ell, kind=method, bounds_error=False)
    interp_d_zm_xi     = interp1d(zm_xi, d_zm_xi, kind=method, bounds_error=False)
    
    interp_d_zp_lambda = interp1d(zp_lambda, d_zp_lambda, kind=method, bounds_error=False)
    interp_d_zp_ell    = interp1d(zp_ell, d_zp_ell, kind=method, bounds_error=False)
    interp_d_zp_xi     = interp1d(zp_xi, d_zp_xi, kind=method, bounds_error=False)

    interp_zm_lambda   = interp1d(d_zm_lambda, zm_lambda, kind=method, bounds_error=False)
    interp_zm_ell      = interp1d(d_zm_ell, zm_ell, kind=method, bounds_error=False)
    interp_zm_xi       = interp1d(d_zm_xi, zm_xi, kind=method, bounds_error=False)
    
    interp_zp_lambda  = interp1d(d_zp_lambda, zp_lambda, kind=method, bounds_error=False)
    interp_zp_ell     = interp1d(d_zp_ell, zp_ell, kind=method, bounds_error=False)
    interp_zp_xi      = interp1d(d_zp_xi, zp_xi, kind=method, bounds_error=False)
    
    
    # Calculating chi_m_lambda_ast
    chi_m_lambda_ast_results = []
    chi_m_xi_ast_results = []
    for jj, zm_lambda_ast in enumerate(zm_lambda):
        d_zm_lambda_ast = interp_d_zm_lambda(zm_lambda_ast)
 
        chi_m_lambda_ast = (interp_zm_ell(d_zm_lambda_ast) / interp_zm_lambda(d_zm_lambda_ast)) * (interp_d_zp_lambda(zm_lambda_ast) / Va)
        chi_m_lambda_ast_results.append(chi_m_lambda_ast)

        chi_m_xi_ast_results.append(chi_m_lambda_ast* align_angle[jj])
        
        
    # Calculating chi_p_lambda_ast
    chi_p_lambda_ast_results = []
    chi_p_xi_ast_results = []
    lambdas = []
    for jj, zp_lambda_ast in enumerate(zp_lambda):
        d_zp_lambda_ast = interp_d_zp_lambda(zp_lambda_ast)
 
        chi_p_lambda_ast = (interp_zp_ell(d_zp_lambda_ast) / interp_zp_lambda(d_zp_lambda_ast)) *(interp_d_zm_lambda(zp_lambda_ast) / Va)
        chi_p_lambda_ast_results.append(chi_p_lambda_ast)

        chi_p_xi_ast_results.append(chi_p_lambda_ast* align_angle[jj])
        
        lambdas.append(zp_lambda_ast)
    return np.array(lambdas), np.array(chi_m_lambda_ast_results), np.array(chi_m_xi_ast_results),  np.array(chi_p_lambda_ast_results), np.array(chi_p_xi_ast_results)



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
                                av_hours  ='2H',
                                return_df = False):
    """
    Calculate variance anisotropy as defined by Verdini et al. (2018).

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.Series: The variance anisotropy values.
    """
    lag       = func.find_cadence(B)
   

    # Calculate variance of components after applying moving average
    
    dB_sq = (B- B.rolling(av_window, center = True).mean())**2
    b     = np.sqrt( dB_sq.rolling(av_hours, center = True).mean())
                     

    # Calculate variance anisotropy
    quant = (b['Bt'] ** 2 + b['Bn'] ** 2) / b['Br'] ** 2
    if return_df:
        return pd.DataFrame({'E': quant.values}, index=quant.index)
    else:
        return quant
    
    
    
def exp_verdini_correct_scale_dependent(
                B,
                fluct_window, 
                av_hours      = 2,
                use_av_hours  = True,
                h_many_stds   = 3.5,
                return_df     = False):
    """
    Calculate variance anisotropy as defined by Verdini et al. (2018).

    Parameters:
        av_window (int): Size of the moving average window in data points.
        B (pandas.DataFrame): The input magnetic field DataFrame with columns 'Br', 'Bt', and 'Bn'.
        av_hours (int, optional): Size of the averaging window in hours. Default is 1.

    Returns:
        pandas.Series: The variance anisotropy values.
    """
    lag          = func.find_cadence(B)
    #av_window    = int(av_hours * 3600 / lag)
    #fluct_window = int(fluct_hours * 3600 / lag)

    # Calculate variance of components after applying moving average
    
    try:

        
        dbs      = B - B.rolling(fluct_window, center =True).mean()
        if use_av_hours:
            rms_db   = dbs.pow(2).rolling(str(av_hours)+"H",  center=True).mean().apply(np.sqrt, raw=True)
        else:
            rms_db   = dbs.pow(2).rolling(2*fluct_window,  center=True).mean().apply(np.sqrt, raw=True)



        # Calculate variance anisotropy
        
        val   = (rms_db['Bt']**2  + rms_db['Bn'] ** 2) / rms_db['Br'] ** 2
        
        stds  = np.nanstd(val)
        quant = np.nanmean(val[val<h_many_stds*stds])#.rolling(av_window, center=True).mean()
        if return_df:
            return pd.DataFrame({'E': quant.values}, index=quant.index)
        else:
            return quant
    except:
        return np.nan
    
    
def exp_verdini_correct(
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
    lag       = func.find_cadence(B)
    av_window = int(av_hours * 3600 / lag)

    # Calculate variance of components after applying moving average
    #b = (B- B.rolling(av_window, center=True).mean()) 
    dbs      = B- B.rolling(av_window, center=True).mean()
    rms_db   = np.sqrt((dbs ** 2).rolling(av_window, center=True).mean())
                    

    # Calculate variance anisotropy
    quant = ((rms_db['Bt']**2  + rms_db['Bn'] ** 2) / rms_db['Br'] ** 2).rolling(av_window, center=True).mean()
    if return_df:
        return pd.DataFrame({'E': quant.values}, index=quant.index)
    else:
        return quant
    
def mag_rotations_zhdankin_single_iter(tau,
                  B,
                  keys = ['Br', 'Bt', 'Bn'],
                  return_dataframe=False):
    """
    Calculate magnetic rotations using the Zhdankin formula, optimized for speed and memory.

    Args:
        tau (int): Time lag.
        B (pd.Series or np.ndarray): Input field.

    Returns:
        alpha (np.ndarray): Magnetic rotations of the input field in degrees.
    """
    # Only keep what you need from df
    B = B[keys]
 
    # Estimate Mod B
    Bmod     = np.sqrt(B[keys[0]]**2 + B[keys[1]]**2 + B[keys[2]]**2)
    Bmod     = Bmod.values if isinstance(Bmod, pd.DataFrame) else np.array(Bmod)
    
    # Convert B to a numpy array for faster operations
    B_values = B.values if isinstance(B, pd.DataFrame) else np.array(B)

    # Calculate dot product and norms directly
    dot_product = np.sum(B_values[:-tau]* B_values[tau:], axis=1)
    norms       = Bmod[:-tau] * Bmod[tau:]

    # Calculate alpha in radians and then convert to degrees
    alpha_degrees = np.arccos(dot_product / norms) * (180/np.pi)

    # If return_dataframe is True, convert the array to a DataFrame with NaN padding
    if return_dataframe:
        alpha_df                  = pd.DataFrame(np.nan, index=B.index if isinstance(B, pd.DataFrame) else range(len(B)), columns=['rotations_deg'])
        alpha_df.iloc[:-tau, 0]   = alpha_degrees 
        return alpha_df

    return alpha_degrees

    
def variance_anisotropy_verdini_spec(av_window,
                                B,
                                av_hours  = None,
                                return_df = False):
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
#     #denom                  = np.sqrt((np.sqrt(diff[keys[0]]**2 + diff[keys[1]]**2 + diff[keys[2]]**2  )**4).rolling(av_window1, center=True).mean())
    
    
    
#     return pd.DataFrame( rms /denom)


def compressibility_complex_chen(  av_window,
                                   B,
                                   keys              = ['Br', 'Bt', 'Bn'],
                                   av_hours          = 1,
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
    lag                 = find_cadence(B)
    av_window1          = int(av_hours * 3600 / lag)

    B['mod']            = np.sqrt(B[keys[0]] ** 2 + B[keys[1]] ** 2 + B[keys[2]] ** 2)
    
    diff                = (B - B.rolling(av_window, center=True).mean()) 
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


    return pd.DataFrame((np.abs(dB['compress'])/np.sqrt((dB[keys[0]].values)**2 + (dB[keys[1]].values)**2 + (dB[keys[2]].values)**2))**2)



def local_parker_spiral(B,
                        V,
                        r, 
                        r0                 =  10, 
                        av_window          = '120min',
                        filter_type        = 'median',
                        estimate_def_angle =  True):
    """
    Transform the magnetic field coordinates from RTN to Parker spiral coordinates.
    
    Parameters:
    B (pd.DataFrame): DataFrame containing the magnetic field data with a datetime index.
    V (pd.DataFrame): DataFrame containing the solar wind velocity data with a datetime index.
    r (float): The distance of the spacecraft from the center of the Sun in AU.
    r0 (float): The source distance in solar radii where the Parker spiral angle is measured.
    av_window (str): Rolling window size for the low-pass filter.
    filter_type (str): Type of the filter to apply ('mean' for rolling mean, 'butter' for Butterworth).
    
    Returns:
    pd.DataFrame: The transformed magnetic field coordinates.
    pd.Series: The Parker spiral angle for each timestamp.
    pd.Series: The filtered timesries of Vr   
    pd.Series: The deflection angle with respect to the  PS  
    """
    
    if len(B)!= len(V):
        V = func.newindex(V, B.index)
    if len(B)!= len(r):
        r = func.newindex(r, B.index)
        
    au_to_km           = 1.496e8  # Conversion from AU to kilometers.
    solar_radius_to_km = 695700  # Conversion from solar radii to kilometers.
    omega              = 2.9e-6  # The Sun's rotational frequency in rad/s.
    
    # Convert r0 from solar radii to kilometers
    r0_km              = r0 * solar_radius_to_km
    
    # Convert r from AU to kilometers
    r_km               = np.hstack(r.values)* au_to_km
    
    # Apply low-pass filter to the radial component of the solar wind velocity
    if filter_type == 'median':
        # Simple rolling mean
        Vr_filtered = V['Vr'].rolling(window=av_window, center=True).median()
    elif filter_type == 'mean':
        Vr_filtered = V['Vr'].rolling(window=av_window, center=True).mean()
    else:
        raise ValueError("Invalid filter type. Choose 'mean' or 'median'.")

    # Calculate the Parker spiral angle in radians using arctan2
    alpha_p = np.arctan(-omega * (r_km - r0_km)/ Vr_filtered)
    
    
    # Estimate magnitude of magnetic field timeseries
    mag_B             = np.sqrt(
                                B.Br**2 +
                                B.Bt**2 +
                                B.Bn**2 
                               ).values

    sign = np.sign(B['Br'].rolling(window='2h', center=True).mean()).values
    # Transform the magnetic field to Parker spiral coordinates
    B_parker       =  B.copy()
    B_parker['Br'] =  sign*mag_B * np.cos(alpha_p)
    B_parker['Bt'] =  sign*mag_B * np.sin(alpha_p)
    B_parker['Bn'] =  0*B['Bn'] # # Bn remains unchanged
    
    
    if estimate_def_angle:
    
        mag_B0            = np.sqrt(
                                    B_parker['Br']**2 +
                                    B_parker['Bt']**2 +
                                    B_parker['Bn']**2 
                                   ).values
        

        
        def_angles   = np.arccos((B['Br'] * B_parker['Br']   + 
                                  B['Bt'] * B_parker['Bt']   +
                                  B['Bn'] * B_parker['Bn'])  / (mag_B0      * mag_B)) * 180 / np.pi
    else:
            def_angles =None
    
    return B_parker, alpha_p, Vr_filtered, def_angles


def parallel_compress(lag,
                      Bdf,
                      keys              = ['Br', 'Bt', 'Bn'],
                      five_points_sfunc = True):
    
    comp = calculate_compressibility(
                         lag,
                         Bdf,
                         keys=keys,
                         five_points_sfunc=five_points_sfunc).values
    ind = np.isinf(comp) | (comp>2.)
    return np.nanmean(comp[~ind])







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
    
    return pd.DataFrame((rms[keys[0]]+ rms[keys[1]] +  rms[keys[2]])/np.sqrt(B[keys[0]]**2 + B[keys[1]]**2 + B[keys[2]]**2).rolling(av_window1, center=True).mean())



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
            B_resampled[f'PVI_{str(hmany[kk])}'] = results[kk][f'PVI_{str(kk)}']
        else:
            B_resampled[f'PVI_mod_{str(hmany[kk])}'] = results[kk][f'PVI_mod_{str(kk)}']
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

        print(tau)
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

def estimate_WT_distribution(big_gaps, 
                             B_resampled,
                             PVI_thresholds,
                             hmany,
                             remove_gaps= False):
    """ ESTIMATE WT DISTRIBUTIONS, remove the gaps indentified earlier """ 
    
    if remove_gaps:
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


@jit(nopython=True, parallel=True)
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
    xvals          = np.arange(0, max_qorder+1,1)
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


def sc_sampling_angle(dfpar, window):

    # Extracting spacecraft and solar wind velocities
    Vsc = dfpar[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].values
    Vsw = dfpar[['Vr', 'Vt', 'Vn']].values

    dv = Vsc - Vsw
    dot_product = dv[:, 0]
    dv_magnitudes = np.linalg.norm(dv, axis=1)
    dv_magnitudes[dv_magnitudes == 0] = np.nan
    cos_theta = np.clip(dot_product / dv_magnitudes, -1, 1)
    angles    = np.degrees(np.arccos(cos_theta))


     # Creating a DataFrame for angles
    angles_df = pd.DataFrame(angles, index=dfpar.index, columns=['Angle'])

    return angles_df.resample(f'{window}s').mean()

def calculate_angle(which_perihelion,
                    days_around,
                    window, 
                    credentials,
                    save_path,
                    vars_2_downnload,
                    use_span   =True):
    # Function to calculate the angle
    
    sys.path.insert(1, os.path.join(os.getcwd(), 'functions/downloading_helpers'))
    import   PSP #$import  LoadTimeSeriesPSP
    au_to_km       = 1.496e8  # Conversion factor
    
    #Important!! Make sure your current directory is the MHDTurbPy folder!
    os.chdir("/Users/nokni/work/MHDTurbPy/")


    # Make sure to use the local spedas
    sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))


    
    print(f'Loading data for E{which_perihelion}')
    which_perihelion = which_perihelion- 1
    
    peri_dates = [pd.Timestamp(x) for x in [
        '2018-11-06/03:27',
        '2019-04-04/22:39',
        '2019-09-01/17:50',
        '2020-01-29/09:37',
        '2020-06-07/08:23',
        '2020-09-27/09:16',
        '2021-01-17/17:40',
        '2021-04-29/08:48',
        '2021-08-09/19:11',
        '2021-11-21/08:23',
        '2022-02-25/15:38',
        '2022-06-01/22:51',
        '2022-09-06/06:04',
        '2022-12-11/13:16',
        '2023-03-17/20:30',
        '2023-06-22/03:46',
        '2023-09-27/23:28',
        '2023-12-29/00:54',
        '2024-03-30/02:20',
        '2024-06-30/03:46',
        '2024-09-30/05:13',
        '2024-12-24/11:41',
        '2025-03-22/22:25',
        '2025-06-19/09:09'
    ]]

    wing = pd.Timedelta(str(int(days_around))+'d')



    encounters = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

    # Change to to specified working dir
    #os.chdir(choose_working_dir)


    dictionary ={}
    for jj, encounter in enumerate(encounters):
       # collect()
        peri_date = peri_dates[encounter-1]
        wing = pd.Timedelta('7d')
        t00 = (peri_date - wing).floor('1d')-pd.Timedelta('60min')
        t10 = (peri_date + wing).ceil('1d')+pd.Timedelta('60min')


             
        # Define final path
        final_path              =  Path(save_path)

        dictionary[str(jj)] = {'Start': t00, 'End': t10}
    # Create a DataFrame
    df = pd.DataFrame(dictionary).T

    
    start, end     = df['Start'][which_perihelion], df['End'][which_perihelion]
    t0i, t1i       = func.ensure_time_format(start, end)


    varnames_MAG, varnames_QTN, varnames_SPAN, varnames_SPC,  varnames_SPAN_alpha, varnames_EPHEM = PSP.default_variables_to_download_PSP(vars_2_downnload)

    settings ={ }
    settings['use_local_data'] = False
    if use_span:
        dfpar =  PSP.download_SPAN_PSP(t0i, t1i, credentials, varnames_SPAN, varnames_SPAN_alpha, settings)
        
    else:
        dfephem     = PSP.download_ephemeris_PSP(t0i, t1i, credentials, ['position', 'velocity'], settings)
        dfpar       =  PSP.download_SPC_PSP(t0i, t1i, credentials, varnames_SPC, settings)
   
        dfephem     = func.newindex(dfephem, dfpar.index)
        dfpar[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']] = dfephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']]



    # Calculate angles for each row
    angles = sc_sampling_angle(dfpar, window)
    
    func.savepickle(angles, str(Path(save_path).joinpath(f'E_{which_perihelion+1}')), 'angles.pkl')

    return angles, dfpar
