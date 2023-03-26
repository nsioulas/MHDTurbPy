
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
import traceback
import ssqueezepy

# SPEDAS API
# make sure to use the local spedas
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data


sys.path.insert(0,'/Users/nokni/work/MHDTurbPy/functions')
import general_functions as func




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
    scales = sj
    
    return db_x, db_y, db_z, freqs, PSD, scales


def trace_PSD_cwt_ssqueezepy(x, y, z, dt, nv =16, scales='log-piecewise', wavelet=None, wname=None, l1_norm=False, est_PSD=True):
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
        scales = 'log-piecewise'

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



def TracePSD(x, y, z , dt, remove_mean=False):
    """ 
    Estimate Power spectral density:
    Inputs:
    u : timeseries, np.array
    dt: 1/sampling frequency
    """
    if not isinstance(x, np.ndarray):
        x = x.values
        y = y.values
        z =z.values

    if remove_mean:
        x = x  - np.nanmean(x)
        y = y  - np.nanmean(y)
        z = z  - np.nanmean(z)

    N  = len(x)
    xf = np.fft.rfft(x)
    yf = np.fft.rfft(y)
    zf = np.fft.rfft(z);


    B_pow = 2 * (np.abs(xf) ** 2 + np.abs(yf) ** 2 + np.abs(zf) ** 2    ) / N * dt

    freqs = np.fft.fftfreq(len(x), dt)
    freqs = freqs[freqs>0]
    idx   = np.argsort(freqs)

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

    from scipy.fft import fft, fftfreq

    # log10 frequency ratio
    freqratio = 1.05


    # build chunks
    # time per chunk in seconds
    chunk_duration_sec = chuncktime
    chunktime_str = f'{int(chuncktime)}s'
    ts_chunk = func.chunkify(mag.index, chunk_duration_sec)

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
            ft_window = func.window_selector(N, win_name=windowStr)

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
        Btr_smooth = func.smooth(np.real(Btr), w_size)

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


@jit(nopython=True, parallel=True)
def hampel_filter(input_series, window_size, n_sigmas=3):
    """
      Hampel filter function for despiking a timeseries (i.e., remove spurious datapoints) 
    """

    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    indices = []
    
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices


@njit(nogil=True)
def norm_factor_Gauss_window(phys_space_scales, scales, dt, lambdaa=3):
    
    s             = scales
    numer         = np.arange(-3*phys_space_scales, 3*phys_space_scales, dt)
    multiplic_fac = np.exp(-(numer)**2/(2*(lambdaa**2)*(s**2)))
    norm_factor   = np.sum(multiplic_fac)
    window        = len(multiplic_fac)
    
    return window,  multiplic_fac, norm_factor


def estimate_wavelet_coeff(B_df, V_df,  dj , lambdaa=3, pycwt=False):

    """
    Method to calculate the  1) wavelet coefficients in RTN 2) The scale dependent angle between Vsw and Β
    
    Parameters
    ----------
    B_df: dataframe
        Magnetic field timeseries dataframe

    mother_wave: str
        The main waveform to transform data.
        Available waves are:
        'gaussian':
        'paul': apply lomb method to compute PSD
        'mexican_hat':
    Returns
    -------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """
    #Estimate sampling time of timeseries
    dt_B                                 =  func.find_cadence(B_df)
    dt_V                                 =  func.find_cadence(V_df)
    
    if dt_V !=dt_B:
        V_df = func.newindex(V_df, B_df.index, interp_method='linear')
    
    # Common dt
    dt        =  dt_B
    
    # Turn columns of df into arrays   
    Br, Bt, Bn                           =  B_df.Br.values, B_df.Bt.values, B_df.Bn.values
    Vr, Vt, Vn                           =  V_df.Vr.values, V_df.Vt.values, V_df.Vn.values    

    # Estimate magnitude of magnetic field
    mag_orig                             =  np.sqrt(Br**2 + Bt**2 +  Bn**2 )
    
    # Estimate the magnitude of V vector
    mag_v                                = np.sqrt(Vr**2 + Vt**2 + Vn**2)


    angles   = pd.DataFrame()
    VBangles = pd.DataFrame()
    from scipy import signal
    # Estimate PSDand scale dependent fluctuations
    if pycwt:
        db_x, db_y, db_z, freqs, PSD,  scales = trace_PSD_wavelet(Br, Bt, Bn, dt, dj,  mother_wave='morlet')
    else:
        # To avoid rewrititing the code (2/dj) is a good compromise (and it doesnt really matter)
        db_x, db_y, db_z, freqs, PSD, scales = trace_PSD_cwt_ssqueezepy(Br, Bt, Bn, dt, nv=int(2/dj))   

    # Calculate the scales in physical space and units of seconds
    k                 = 6.0 / (2 * np.pi)
    phys_space_scales = k / (freqs * dt)

    for ii in range(len(scales)):
        try:
            window, multiplic_fac, norm_factor= norm_factor_Gauss_window(phys_space_scales[ii], scales[ii], dt, lambdaa)


            # Estimate scale dependent background magnetic field using a Gaussian averaging window

            res2_Br = (1/norm_factor)*signal.convolve(Br, multiplic_fac, 'same')
            res2_Bt = (1/norm_factor)*signal.convolve(Bt, multiplic_fac, 'same')
            res2_Bn = (1/norm_factor)*signal.convolve(Bn, multiplic_fac, 'same')


            # Estimate magnitude of scale dependent background
            mag_bac = np.sqrt(res2_Br**2 + res2_Bt**2 + res2_Bn**2 )

            # Estimate angle
            angles[str(ii+1)] = np.arccos(res2_Br/mag_bac) * 180 / np.pi

            # Estimate VB angle
            VBangles[str(ii+1)] = np.arccos((Vr*res2_Br + Vt*res2_Bt + Vn*res2_Bn)/(mag_bac*mag_v)) * 180 / np.pi

            # Restric to 0< Θvb <90
            VBangles[str(ii+1)][VBangles[str(ii+1)]>90] = 180 - VBangles[str(ii+1)][VBangles[str(ii+1)]>90]

            # Restric to 0< Θvb <90
            angles[str(ii+1)][angles[str(ii+1)]>90] = 180 - angles[str(ii+1)][angles[str(ii+1)]>90]
        except:
             pass

    return db_x, db_y, db_z, angles, VBangles, freqs, PSD, phys_space_scales,  scales



"""" Define function to estimate PVI timeseries"""
def estimate_PVI(B_resampled, hmany, di, Vsw,  hours, PVI_vec_or_mod='vec'):

    av_hours                           = hours*3600
    lag                                = (B_resampled.index[1]-B_resampled.index[0])/np.timedelta64(1,'s')
    av_window                          = int(av_hours/lag)



    for kk in range(len(hmany)):
        tau                             = round((hmany[kk]*di)/(Vsw*lag))

        if tau<1:
            print('The value of hmany you chose is too low. You will have to use higher resol mag data!')
            while tau<1:
                hmany[kk] = hmany[kk] + 0.01*hmany[kk]
                tau       = round((hmany[kk]*di)/(Vsw*lag))

            print('The value was set to the minimum possible, hmany=',hmany[kk]) 

        if tau>0:
            if PVI_vec_or_mod:
                ### Estimate PVI ###
                B_resampled['DBtotal']         = np.sqrt((B_resampled.Br.diff(tau))**2 + (B_resampled.Bt.diff(tau))**2 + (B_resampled.Bn.diff(tau))**2)
                B_resampled['DBtotal_squared'] = B_resampled['DBtotal']**2
                denominator = np.sqrt(
                    B_resampled['DBtotal_squared']
                    .rolling(av_window, center=True)
                    .mean()
                )

                PVI_dB                         = pd.DataFrame({     'DateTime' : B_resampled.index,
                                                                    'PVI'      : B_resampled['DBtotal']/denominator})
                PVI_dB                         = PVI_dB.set_index('DateTime')
                B_resampled[f'PVI_{str(hmany[kk])}'] = PVI_dB.values

                # Save RAM
                del  B_resampled['DBtotal_squared'], B_resampled['DBtotal']
            else:
                B_resampled['B_modulus']           = np.sqrt((B_resampled.Br)**2 + (B_resampled.Bt)**2 + (B_resampled.Bn)**2)
                B_resampled['DBtotal']             = B_resampled['B_modulus'].diff(tau)
                B_resampled['DBtotal_squared']     = B_resampled['DBtotal']**2

                denominator = np.sqrt(
                    B_resampled['DBtotal_squared']
                    .rolling(av_window, center=True)
                    .mean()
                )

                PVI_dB                             = pd.DataFrame({'DateTime': B_resampled.index,
                                                                        'PVI': B_resampled['DBtotal']/denominator})
                PVI_dB                             = PVI_dB.set_index('DateTime')
                B_resampled[f'PVI_mod_{str(hmany[kk])}'] = PVI_dB.values
                del  B_resampled['DBtotal_squared'], B_resampled['DBtotal'] , B_resampled['B_modulus']
        elif PVI_vec_or_mod:
            B_resampled[f'PVI_{str(hmany[kk])}'] = np.nan*B_resampled.Br.values
        else:
            B_resampled[f'PVI_mod_{str(hmany[kk])}'] = np.nan*B_resampled.Br.values

    return B_resampled



"""" Define function to estimate the structure functions of a field """
@jit( parallel =True,  nopython=True)
def structure_functions(scales, qorder,mat):
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]
    
    # initiate arrays
    length = np.zeros(len(scales))
    result = np.zeros((len(scales),len(qorder)))
    
    # Estimate sfuncs
    for k in prange(len(scales)):
        dB = np.sqrt((ar[scales[k]:]-ar[:-scales[k]])**2 + 
                     (at[scales[k]:]-at[:-scales[k]])**2 + 
                     (an[scales[k]:]-an[:-scales[k]])**2)
        
        for i in prange(len(qorder)):   
            result[k,i] = np.nanmean(np.abs(dB**qorder[i]))
    return result


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



from numba import prange
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




#@jit( parallel =True)
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




@jit(parallel =True)
def estimate_PSD_wavelets_all_intervals_E1(db_x, db_y, db_z, angles, freqs,   dt,  per_thresh, par_thresh):

    PSD_par = np.zeros(len(freqs))
    PSD_per = np.zeros(len(freqs)) 

    for i in range(np.shape(angles)[0]):

        index_per = (np.where(angles[i]>per_thresh)[0]).astype(np.int64)
        index_par = (np.where(angles[i]<par_thresh)[0]).astype(np.int64)
        #print(len(index_par), len(index_per))

        PSD_par[i]  = (np.nanmean(np.abs(np.array(db_x[i])[index_par])**2) + np.nanmean(np.abs(np.array(db_y[i])[index_par])**2) + np.nanmean(np.abs(np.array(db_z[i])[index_par])**2) ) * ( 2*dt)
        PSD_per[i]  = (np.nanmean(np.abs(np.array(db_x[i])[index_per])**2) + np.nanmean(np.abs(np.array(db_y[i])[index_per])**2) + np.nanmean(np.abs(np.array(db_z[i])[index_per])**2) ) * ( 2*dt)

    return PSD_par, PSD_per


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
            #res2        = func.pdf(time.values[1:], hmany_bins_PDF_WT, 1,1)
            thresh[f'PVI>{str(i)}'] = time.values[1:]

        WT[f'PVI_{str(k)}'] = thresh

    return WT


from numba import prange
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


