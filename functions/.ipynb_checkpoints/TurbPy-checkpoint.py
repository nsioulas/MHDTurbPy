###########################################################################
#                                                                         #
#    Copyright 2024 Nikos Sioulas                                         #
#    UCLA                                                                 #
#                                                                         #
#    nsioulas@g.uca.edu                                                   #
#                                                                         #
#    This file is part of MHDTurbPy toolbox.                              #
#                                                                         #
#    MHDTurbPy toolbox is free software: you can redistribute it          #
#    and/or modify it under the terms of the GNU General Public           #
#    License as published by the Free Software Foundation, either         #
#    version 3 of the License, or (at your option) any later version.     #
#                                                                         #
#    MHDTurbPy toolbox is distributed in the hope that it will be         #
#    useful, but WITHOUT ANY WARRANTY; without even the implied warranty  #
#    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.              #
###########################################################################


# Basic libraries
import pandas as pd
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
    B_np = B.to_numpy()
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
                                 n_jobs            =-1):
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

    def calc_sfn( dB,
                  dBmod,
                  qorder,
                  return_components = False,
                  return_Bmod       = False):
        
        # Estimate SF for each component
        comps    = np.nanmean(dB ** qorder, axis=0)
        if return_Bmod:
            SF_dBmod = np.nanmean(dBmod ** qorder)
        else:
            SF_dBmod = np.nan
        
        if return_components:
            return np.sum(comps), comps, SF_dBmod
        else:
            return np.sum(comps), SF_dBmod

    def process_scale(tau,
                      return_components = False,
                      return_Bmod       = False,
                      return_compress   = False,
                      return_flucts     = False):
        
        
        dB  = np.abs(flucts(tau,
                            B, 
                            five_points_sfunc = five_points_sfunc))
        
        if return_Bmod:
            dBmod  = np.abs(flucts(tau,
                                   B,
                                   five_points_sfunc   = five_points_sfunc,
                                   estimate_mod_flucts = return_Bmod ))
        
            # make sure to prevent!
            if  return_flucts:
                return_compress = False
                
            if return_compress:
                compress = np.nanmean(np.abs(dBmod.T[0])**2/((dB.T[0])**2 + (dB.T[1])**2 + (dB.T[2])**2))
        else:
            dBmod    = np.nan
            compress = np.nan
            
            
        if  return_flucts:
            return dB,  dBmod
        else:
        
            if return_components:
                # Unpack results directly in a list comprehension
                result1, result2, SF_dBmod = zip(*[calc_sfn(  dB,
                                                              dBmod.T[0],
                                                              qorder,
                                                              return_components = return_components,
                                                              return_Bmod       = return_Bmod) for qorder in qorders])

                # Convert the tuples to numpy arrays
                sfn       = np.array(result1)
                SF_dBmod  = np.array(SF_dBmod)
                sfn_comps = np.array(result2)
                sdk       = sfn.T[3] / np.sum(np.nanmean(dB ** 2, axis=0) ** 2)
                counts    =  np.count_nonzero(~np.isnan(dBmod))
                compress  = np.nan

                return sfn, sdk, sfn_comps, SF_dBmod, compress, counts
            else:
                sfn = np.array([calc_sfn(dB,
                                         dBmod,
                                         qorder) for qorder in qorders])

                sdk = sfn.T[3] / np.sum(np.nanmean(dB ** 2, axis=0) ** 2)

                return sfn, sdk


    results = Parallel(n_jobs=n_jobs)(
        delayed(process_scale)(tau, 
                               return_components = return_components,
                               return_compress   = return_compress,
                               return_Bmod       = return_Bmod,
                               return_flucts     = return_flucts) for tau in scales)
    
    # Now for the final return
    if return_flucts:
        dbs, dbs_mod = zip(*results)
      
        return np.array(dbs), np.array(dbs_mod)
    else:
        if return_components:
            sfns, sdk, sfn_comps, SF_dBmod, compress,  counts = zip(*results)

            return np.array(sfns), np.array(sdk), np.array(sfn_comps), np.array(SF_dBmod), np.array(compress), np.array(counts)  #The last one is the components!!      
        else:
            sfn, sdk = zip(*results)

            return np.array(sfn), np.array(sdk)

        
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
        
    
    # Estimate MODWT coefficients and weights
    Wr, Vj   = modwt.modwt(R, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    Wt, Vj   = modwt.modwt(T, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    Wn, Vj   = modwt.modwt(N, wtf=wname, nlevels='conservative', boundary='reflection', RetainVJ=True)
    
    # Return freqs and scales too
    scale = 2**np.arange(1,np.shape(Wr)[0]+1);
    freqs = pywt.scale2frequency('coif6', scale)/dt
   
    # Estimate Fsc_{ii} and PSD = Σ Fsc_{ii}
    PSD_R = modwt.wspec(Wr, dt)
    PSD_T = modwt.wspec(Wt, dt)
    PSD_N = modwt.wspec(Wn, dt)
    
    return freqs, 2*(PSD_R[0] + PSD_T[0] + PSD_N[0]), scale


def Trace_haar_wavelet_psd(x, y, z, dt, wavelet='haar'):

    # Perform the wavelet decomposition on the padded data
    x = pywt.wavedec(x, wavelet)
    y = pywt.wavedec(y, wavelet)
    z = pywt.wavedec(z, wavelet)
    
    px = []
    py = []
    pz = []
    for i in range(1, len(x)):

        px.append(np.nanmean(x[i]**2))
        py.append(np.nanmean(y[i]**2))
        pz.append(np.nanmean(z[i]**2))
        
    px       = dt*(np.array(px[::-1]))/np.log2(2)
    py       = dt*(np.array(py[::-1]))/np.log2(2)
    pz       = dt*(np.array(pz[::-1]))/np.log2(2)
    p_trace  = px + py + pz
    
    freqs = 2.0**(-np.arange(1, len(px)+1))/dt

    return x, y, z, freqs, p_trace, px, py, pz


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
    
    mother_wave_dict = {
    'gaussian': pycwt.DOG(),
    'paul': pycwt.Paul(),
    'mexican_hat': pycwt.MexicanHat()}
    

    if mother_wave in mother_wave_dict.keys():
        mother_morlet = mother_wave_dict[mother_wave]
    else:
        mother_morlet = pycwt.Morlet()
        
    N                                       = len(x)


    db_x, sj, freqs, coi, signal_ft, ftfreqs = pycwt.cwt(x, dt, dj, wavelet=mother_morlet)
    db_y, _, freqs, _, _, _                  = pycwt.cwt(y, dt, dj, wavelet=mother_morlet)
    db_z, _, freqs, _, _, _                  = pycwt.cwt(z, dt, dj, wavelet=mother_morlet)
     
    # Estimate trace powerspectral density
    PSD = (np.nanmean(np.abs(db_x)**2, axis=1) + np.nanmean(np.abs(db_y)**2, axis=1) + np.nanmean(np.abs(db_z)**2, axis=1)   )*( 2*dt)
    
    # Remember!
    scales = (1/(freqs))/dt
    
    
    return db_x, db_y, db_z, freqs, PSD, scales







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
    
    if wavelet is None:
        wavelet    = ssqueezepy.Wavelet(('morlet', {'mu': 13.4}))
    else:
        wavelet    = ssqueezepy.Wavelet((wname, {'mu': 13.4}))  
        
    if  scales_type  is  None:
        scales_type  = 'log'

    # Estimate sampling frequency
    fs          = 1/dt
    
    # Estimate wavelet coefficients
    Wx, scales  = ssqueezepy.cwt(x, wavelet,  scales_type , fs, l1_norm=l1_norm, nv=nv)
    Wy, _       = ssqueezepy.cwt(y, wavelet,  scales_type , fs, l1_norm=l1_norm, nv=nv)
    Wz, _       = ssqueezepy.cwt(z, wavelet,  scales_type , fs, l1_norm=l1_norm, nv=nv)
    
    
     
    if est_mod:
        Wmod , _  = ssqueezepy.cwt(np.sqrt(x**2 + y**2 + z**2), wavelet,  scales_type , fs, l1_norm=l1_norm, nv=nv)
    else:
        Wmod      = None
    
    # Estimate corresponding frequencies
    freqs       = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(x), fs)
    
    # This is the correct one!
    scales     = (omega0)/(2*np.pi*freqs)*(1  + 1/(2*omega0**2))*fs
    
    if est_PSD:
        # Estimate trace powers pectral density
        PSD        = (np.nanmean(np.abs(Wx)**2, axis=1) + np.nanmean(np.abs(Wy)**2, axis=1) + np.nanmean(np.abs(Wz)**2, axis=1)   )*( 2*dt)
        
        if est_mod:
            PSD_mod = (np.nanmean(np.abs(Wmod)**2, axis=1)  )*( 2*dt)
        else:
            PSD_mod     = None
    else:
        PSD        = None
        PSD_mod     = None
    

    return Wx, Wy, Wz, Wmod,  freqs, PSD, PSD_mod, scales 



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
    if not isinstance(x, np.ndarray):
        x = x.values
        y = y.values
        z = z.values

    if remove_mean:
        x -= np.nanmean(x)
        y -= np.nanmean(y)
        z -= np.nanmean(z)

    N = len(x)

    xf = np.fft.rfft(x)
    yf = np.fft.rfft(y)
    zf = np.fft.rfft(z)

    p_X     = 2 * (np.abs(xf) ** 2) / N * dt
    p_Y     = 2 * (np.abs(yf) ** 2) / N * dt
    p_Z     = 2 * (np.abs(zf) ** 2) / N * dt
    p_Trace = p_X + p_Y + p_Z

    freqs = np.fft.rfftfreq(N, dt)

    if return_mod:
        mod = np.sqrt(x**2 + y**2 + z**2)
        p_Mod = 2 * (np.abs(np.fft.rfft(mod)) ** 2) / N * dt
        return freqs, p_Trace, p_X, p_Y, p_Z, p_Mod

    if return_components:
        return freqs, p_Trace, p_X, p_Y, p_Z

    return freqs, p_Trace




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


# from scipy.signal import stft, istft
# def remove_wheel_noise(data,
#                        fs, 
#                        window_size      = 2**15, 
#                        avg_length       = 2,
#                        power_threshold  = 3.0,
#                        freq_min         = 1.0):
#     """
#     Remove noise tones from a time series of magnetic field measurements.

#     Parameters:
#     data : numpy.ndarray
#         Input time series data.
#     fs : float
#         Sampling frequency in Hz.
#     window_size : int, optional
#         Window size for STFT (number of data points). Default is 2**15.
#     avg_length : int, optional
#         Number of consecutive STFT windows to average. Default is 5.
#     power_threshold : float, optional
#         Power threshold factor to detect noise frequencies. Default is 3.0.
#     freq_min : float, optional
#         Minimum frequency (Hz) to consider for noise detection. Default is 1.0 Hz.

#     Returns:
#     cleaned_data : numpy.ndarray
#         Time series data with noise tones removed.
#     extracted_noise : numpy.ndarray
#         Time series of the extracted noise.
#     """
#     # Compute STFT (removed boundary=None to use default padding)
#     f, t_stft, Zxx = stft(data, fs=fs, window='hann', nperseg=window_size,
#                           noverlap=window_size//2)

#     # Compute power spectrum (magnitude squared)
#     power_spectrum = np.abs(Zxx)**2

#     # Average spectra: For every avg_length consecutive windows, average the spectra
#     n_segments = Zxx.shape[1]
#     n_groups = n_segments // avg_length
#     if n_groups == 0:
#         raise ValueError("Not enough data to form at least one group for averaging spectra. "
#                          "Reduce avg_length or provide more data.")

#     averaged_power_spectra = []
#     for i in range(n_groups):
#         start_idx = i * avg_length
#         end_idx   = start_idx + avg_length
#         avg_power = np.mean(power_spectrum[:, start_idx:end_idx], axis=1)
#         averaged_power_spectra.append(avg_power)

#     averaged_power_spectra = np.array(averaged_power_spectra)  # Shape: (n_groups, n_frequencies)

#     # Compute background spectrum by averaging over all averaged spectra
#    # background_spectrum = np.mean(averaged_power_spectra, axis=0)  # Shape: (n_frequencies,)

#     # Identify noise frequencies where the power exceeds threshold times the background spectrum
#     noise_mask = np.zeros_like(Zxx, dtype=bool)
#     for i in range(n_groups):
#         start_idx   = i * avg_length
#         end_idx     = start_idx + avg_length
#         group_power = power_spectrum[:, start_idx:end_idx]  # Shape: (n_frequencies, avg_length)

#         # Thresholding
#         hamp     = func.hampel(averaged_power_spectra[i], 1201, 2.5)
#         threshold = power_threshold *hamp[0][:, np.newaxis]  # Shape: (n_frequencies, 1)
#         # Identify noise frequencies
#         noise_frequencies = (group_power > threshold) & (f[:, np.newaxis] >= freq_min)  # Shape: (n_frequencies, avg_length)

#         # Assign to noise_mask
#         noise_mask[:, start_idx:end_idx] = noise_frequencies

#     # Use the inverse STFT on the identified noise coefficients to create a noise time series
#     noise_Zxx = np.zeros_like(Zxx, dtype=complex)
#     noise_Zxx[noise_mask] = Zxx[noise_mask]

#     # Generate noise time series
#     _, noise_time_series = istft(noise_Zxx, fs=fs, window='hann', nperseg=window_size,
#                                  noverlap=window_size//2)

#     # Ensure the noise time series has the same length as original data
#     if len(noise_time_series) > len(data):
#         # Trim the reconstructed noise to match the original data length
#         noise_time_series = noise_time_series[:len(data)]
#     elif len(noise_time_series) < len(data):
#         # Pad the reconstructed noise with zeros to match the original data length
#         noise_time_series = np.pad(noise_time_series, (0, len(data) - len(noise_time_series)), 'constant')

#     # Subtract the noise time series from the original data
#     cleaned_data = data - noise_time_series

#     return cleaned_data


# def remove_wheel_noise(data, 
#                        fs,
#                        f_wj_start, dfw_dt, bandwidth_Hz=1.0, attenuation_db=-80):
#     """
#     Remove narrowband noise from data at frequencies f_wj.

#     Parameters:
#     data : numpy.ndarray
#         Input time series data.
#     fs : float
#         Sampling frequency in Hz.
#     f_wj_start : list or numpy.ndarray
#         List of contaminated frequencies (Hz) at the start of data.
#     dfw_dt : float
#         Maximum rate of change of the wheel frequencies (Hz per second).
#     bandwidth_Hz : float, optional
#         Bandwidth around each contaminated frequency to attenuate (Hz).
#     attenuation_db : float, optional
#         Attenuation in decibels (-80 dB by default).

#     Returns:
#     cleaned_data : numpy.ndarray
#         The data with narrowband noise removed.
#     """
#     data = np.asarray(data)
#     data_length = len(data)
#     attenuation_factor = 10 ** (attenuation_db / 20)  # Convert dB to amplitude attenuation factor

#     # Compute N based on the given algorithm
#     N = int(np.sqrt(fs ** 2 / abs(dfw_dt)))
#     if N % 2 != 0:
#         N += 1  # Ensure N is even for FFT symmetry
#     N = min(N, data_length)  # Ensure N does not exceed data length

#     # Set overlap parameters for overlap-add method
#     overlap = N // 2
#     step_size = N - overlap
#     num_steps = (data_length - overlap) // step_size + 1

#     # Initialize arrays for overlap-add
#     cleaned_data = np.zeros(data_length)
#     window_sum = np.zeros(data_length)

#     # Define window function
#     window = np.hanning(N)

#     for i in range(num_steps):
#         start_idx = i * step_size
#         end_idx = start_idx + N
#         if end_idx > data_length:
#             end_idx = data_length
#             start_idx = end_idx - N
#             if start_idx < 0:
#                 start_idx = 0
#                 end_idx = data_length
#                 N_chunk = end_idx - start_idx
#                 window = np.hanning(N_chunk)
#             else:
#                 N_chunk = N
#         else:
#             N_chunk = N

#         data_chunk = data[start_idx:end_idx] * window

#         # Time at this chunk (we can take the midpoint time)
#         t_chunk = (start_idx + N_chunk / 2) / fs

#         # Compute wheel frequencies at this time
#         f_wj_t = [f_wj0 + dfw_dt * t_chunk for f_wj0 in f_wj_start]

#         # Compute FFT
#         freqs = np.fft.rfftfreq(N_chunk, d=1/fs)
#         fft_chunk = np.fft.rfft(data_chunk, n=N_chunk)

#         # Attenuate contaminated frequencies
#         for f_wj in f_wj_t:
#             # Find indices where frequency is within bandwidth_Hz / 2 of f_wj
#             indices_to_attenuate = np.where(np.abs(freqs - f_wj) <= bandwidth_Hz / 2)[0]
#             # Attenuate the coefficients
#             fft_chunk[indices_to_attenuate] *= attenuation_factor

#         # Inverse FFT to reconstruct the cleaned chunk
#         cleaned_chunk = np.fft.irfft(fft_chunk, n=N_chunk)
#         cleaned_chunk *= window  # Apply window again to match overlap-add

#         # Overlap-add the cleaned chunk to the output
#         cleaned_data[start_idx:end_idx] += cleaned_chunk
#         window_sum[start_idx:end_idx] += window ** 2

#     # Normalize to account for window overlap
#     nonzero_indices = window_sum > 1e-10  # Avoid division by zero
#     cleaned_data[nonzero_indices] /= window_sum[nonzero_indices]

#     return cleaned_data



def remove_wheel_noise(signal,
                   fs, 
                   freq_threshold         =  1.5,
                   windows_width_hz       = [0.1, 2],
                   empirical_threshold    = 1.3
                  ):
    for window_hz in func.ensure_iterable(windows_width_hz):
    #for window_hz in (windows_width_hz if len(windows_width_hz) > 1 else [windows_width_hz]):

        # Calculate the Fourier transform and power spectrum of the signal
        N = len(signal)
        signal_fft = np.fft.rfft(signal)
        power_spec = np.abs(signal_fft)**2
        frequencies = np.fft.rfftfreq(N, d=1/fs)

        # Filter frequencies higher than the threshold
        valid_indices        = frequencies > freq_threshold
        frequencies_filtered = frequencies[valid_indices]
        power_spec_filtered  = power_spec[valid_indices]

        # Calculate moving-window mean and standard deviation
        window_size_samples  = int(np.ceil(window_hz / (frequencies[1] - frequencies[0])))*2 + 1

        moving_mean          = np.convolve(power_spec_filtered, np.ones(window_size_samples)/window_size_samples, mode='same')
        
        #_, moving_mean       = func.smoothing_function(frequencies_filtered, power_spec_filtered)
        moving_std           = np.sqrt(np.convolve((power_spec_filtered - moving_mean)**2, np.ones(window_size_samples)/window_size_samples, mode='same'))


        # Define z(f)
        z_f  =  moving_std / moving_mean


        # Calculate empirical threshold based on the mean value of z(f)
        z_cutoff =  empirical_threshold* np.mean(z_f)



        roling_median        = func.simple_python_rolling_median(z_f, 5*window_size_samples)
        quant                = empirical_threshold* roling_median
        #noise_mask           = (z_f > quant) | ( np.isnan(roling_median) & (frequencies_filtered > 2*freq_threshold)) | ((z_f >empirical_threshold) & (frequencies_filtered > 3*freq_threshold))
        noise_mask           = (z_f > quant)
        f_noise              = frequencies_filtered[noise_mask]


        # Mask for frequencies without noise

        # Calculate moving-window mean for no-noise frequencies     
        power_spec_no_noise = power_spec_filtered[~noise_mask]
        frequencies_no_noise = frequencies_filtered[~noise_mask]
        moving_mean_no_noise = np.convolve(power_spec_no_noise, np.ones(window_size_samples)/window_size_samples, mode='same')


        #frequencies_no_noise, _, moving_mean_no_noise = func.smoothing_function(frequencies_no_noise, power_spec_no_noise, 2)

        # Interpolate moving-window mean for the noise frequencies
        moving_mean_interpolated = np.interp(frequencies_filtered, frequencies_no_noise, moving_mean_no_noise)

        # Replace power spectrum values at noise frequencies with interpolated moving mean
        power_spec_noise_removed                                     = power_spec.copy()

        # First, find the indices in the full frequency array where noise is present
        noise_indices = np.where(frequencies > freq_threshold)[0][noise_mask]

        # Now update the power_spec_noise_removed at those indices with the interpolated values
        power_spec_noise_removed[noise_indices] = moving_mean_interpolated[noise_mask]

        # Recalculate magnitude for noise-removed Fourier transform
        magnitude_noise_removed = np.sqrt(power_spec_noise_removed)

        # Retain phases for no-noise frequencies, randomize for noise frequencies
        phases = np.angle(signal_fft)
        # Generate random phases only for the noise frequencies
        random_phases = np.random.uniform(-np.pi, np.pi, len(f_noise))

        # Assign random phases only to the noise frequencies
        phases_noise_indices = np.where(np.isin(frequencies, f_noise))[0]
        phases[phases_noise_indices] = random_phases


        # Construct the noise-removed Fourier transform
        noise_removed_fft = magnitude_noise_removed * np.exp(1j * phases)

        # Perform the inverse Fourier transform to get the noise-removed signal
        signal = np.fft.irfft(noise_removed_fft, n=N)



    return signal






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
                        av_window          = '60min',
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
                                    B_parker['Bn']**2 +
                                    B_parker['Bn']**2 
                                   ).values
        

        
        def_angles   = np.arccos((B['Br'] * B_parker['Br']   + 
                                  B['Bt'] * B_parker['Bt']   +
                                  B['Bn'] * B_parker['Bn'])  / (mag_B0     * mag_B)) * 180 / np.pi
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
