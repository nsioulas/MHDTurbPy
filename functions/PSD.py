import pandas as pd
import numpy as np
import scipy
from scipy import signal
import os
from pathlib import Path
from glob import glob
from numba import jit,njit,prange
import pycwt as wavelet
from scipy import signal
from distutils.log import warn
import time
from numba import jit,njit,prange,objmode 
import datetime
import traceback

mother_wave_dict = {
    'gaussian': wavelet.DOG(),
    'paul': wavelet.Paul(),
    'mexican_hat': wavelet.MexicanHat()
}

def trace_PSD_wavelet(x, y, z, dt, dj,  mother_wave='morlet'):
    """
    Method to calculate the  power spectral density using wavelet method.
    Parameters
    ----------
    x,y,z: array-like
        the components of the field to apply wavelet tranform
    dt: int
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
    psd : list
        The scales at which wavelet was estimated
    """

    if mother_wave in mother_wave_dict.keys():
        mother_morlet = mother_wave_dict[mother_wave]
    else:
        mother_morlet = wavelet.Morlet()
        
    N                                       = len(x)

    db_x, _, freqs, _, _, _  = wavelet.cwt(x, dt,  dj, wavelet=mother_morlet)
    db_y, _, freqs, _, _, _  = wavelet.cwt(y, dt,  dj, wavelet=mother_morlet)
    db_z, _, freqs, _, _, _  = wavelet.cwt(z, dt, dj, wavelet=mother_morlet)
     
    # Estimate trace powerspectral density
    PSD = (np.nanmean(np.abs(db_x)**2, axis=1) + np.nanmean(np.abs(db_y)**2, axis=1) + np.nanmean(np.abs(db_z)**2, axis=1)   )*( 2*dt)
    
    # Also estimate the scales to use later
    scales = ((1/freqs)/dt)#.astype(int)
    
    return db_x, db_y, db_z, freqs, PSD, scales



def TracePSD_FFT(x, y, z , remove_mean,dt):
    """ 
    Estimate Power spectral density:
    Inputs:
    u : timeseries, np.array
    dt: 1/sampling frequency
    """
    if isinstance(x, np.ndarray):
        pass
    else:
        x = x.values; y = y.values; z =z.values
    
    if remove_mean:
        x = x  - np.nanmean(x)
        y = y  - np.nanmean(y)
        z = z  - np.nanmean(z)

    N  = len(x)
    xf = np.fft.rfft(x);  yf = np.fft.rfft(y); zf = np.fft.rfft(z);
    
    B_pow = (np.abs(xf) ** 2 + np.abs(yf) ** 2 + np.abs(zf) ** 2    ) / N * dt

    freqs = np.fft.fftfreq(len(x), dt)
    freqs = freqs[freqs>0]
    idx   = np.argsort(freqs)
    
    return freqs[idx], B_pow[idx]