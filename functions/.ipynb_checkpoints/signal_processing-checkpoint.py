"""Module providing Multirate signal processing functionality.
Largely based on MATLAB's Multirate signal processing toolbox with consultation 
of Octave m-file source code.
"""


import os
import sys
import fractions
import numpy
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, firwin
from numpy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d



""" Import manual functions """
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def downsample_and_filter(high_df,
                          low_df,
                          order      = 5,
                          #percentage = 1.1):
                          percentage = 1.15):
    
    
    # Calculate sampling rates for both datasets
    high_fs = 1 / func.find_cadence(high_df)  # Adjusted for high_df
    low_fs = 1 /  func.find_cadence(low_df)   # assuming index is datetime
    

    cutoff = percentage * low_fs / 2  # Define cutoff frequency
    
    # Ensure the cutoff frequency is within valid range for digital filter design
    if cutoff >= high_fs / 2:
        raise ValueError("Cutoff frequency must be less than half the high_df sampling rate.")

    filtered_df = high_df.copy()
    
    print( high_df.columns)
    for column in high_df.columns:
        # Apply low-pass filter to each column
        filtered_data = apply_lowpass_filter(high_df[column].values, cutoff, high_fs, order=order)
        filtered_df[column] = filtered_data

    # Interpolate or reindex the filtered data to match the lower sample rate DataFrame's index
    resampled_df = func.newindex(filtered_df, low_df.index)
    return resampled_df





def upsample_dataframe(original_df, target_df, numtaps=101, cutoff_ratio=0.85, interp_kind='cubic'):
    """
    Upsamples the original_df to match the sampling rate of target_df, applying a linear-phase FIR filter.

    Parameters:
        original_df (pd.DataFrame): DataFrame containing the original time series.
                                    The index should be datetime-like.
        target_df (pd.DataFrame): DataFrame containing the target time series with higher sampling rate.
                                  The index should be datetime-like.
        numtaps (int, optional): Number of filter coefficients (taps) for the FIR filter. Default is 101.
        cutoff_ratio (float, optional): Ratio of Nyquist frequency to set the cutoff frequency (default 0.89).
        interp_kind (str, optional): Kind of interpolation ('linear', 'cubic', etc.). Default is 'cubic'.

    Returns:
        pd.DataFrame: Upsampled original_df with time matching target_df.
    """
    
    # Ensure the indices are datetime and sorted
    if not isinstance(original_df.index, pd.DatetimeIndex):
        raise TypeError("original_df must have a DatetimeIndex.")
    if not isinstance(target_df.index, pd.DatetimeIndex):
        raise TypeError("target_df must have a DatetimeIndex.")
    
    original_df = original_df.sort_index()
    target_df = target_df.sort_index()
    
    # Determine data columns to process
    data_cols = original_df.columns.tolist()
    
    # Convert datetime indices to numerical values (e.g., seconds since start)
    t0 = original_df.index[0]
    original_time = (original_df.index - t0).total_seconds().values
    target_time = (target_df.index - t0).total_seconds().values
    
    # Compute sampling rates
    delta_t_original = np.median(np.diff(original_time))
    delta_t_target = np.median(np.diff(target_time))
    
    Fs_original = 1.0 / delta_t_original
    Fs_target = 1.0 / delta_t_target
    
    if Fs_target <= Fs_original:
        raise ValueError("Target sampling rate must be higher than original sampling rate for upsampling.")
    
    # Design a linear-phase FIR low-pass filter
    nyq_original = 0.5 * Fs_original
    cutoff_freq = cutoff_ratio * nyq_original
    normalized_cutoff = cutoff_freq / nyq_original  # Normalize with original Nyquist
    
    # Ensure the cutoff is below the Nyquist frequency
    if cutoff_freq >= nyq_original:
        raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyq_original} Hz)")
    
    # Use a Hamming window for the FIR filter to minimize ripple
    fir_coeff = firwin(numtaps, cutoff=normalized_cutoff, window='hamming', pass_zero='lowpass')
    
    # Initialize a dictionary to hold upsampled data
    upsampled_data = {'datetime': target_df.index}
    
    # Process each data column
    for col in data_cols:
        data_original = original_df[col].values
        
        # Step 1: Apply FIR low-pass filter using filtfilt for zero-phase
        data_filtered = filtfilt(fir_coeff, [1.0], data_original)
        
        # Step 2: Interpolate to target_time using higher-order interpolation
        interp_func = interp1d(original_time, data_filtered, kind=interp_kind, fill_value='extrapolate')
        data_upsampled = interp_func(target_time)
        
        upsampled_data[col] = data_upsampled
    
    # Create the upsampled DataFrame
    upsampled_df = pd.DataFrame(upsampled_data).set_index('datetime')
    
    return upsampled_df





# Function to apply FFT-based low-pass filter
def fft_low_pass_filter(signal, cutoff_hz, fs):
    """
    Apply a low-pass FFT filter to a signal.
    
    Parameters:
    - signal: The input signal (time domain).
    - cutoff_hz: The cutoff frequency in Hz.
    - fs: The sampling rate of the signal in Hz.
    
    Returns:
    - The filtered signal (time domain).
    """
    # FFT of the signal
    signal_fft = fft(signal)
    
    # Generate the frequency axis
    frequencies = fftfreq(len(signal), 1/fs)
    
    # Zero out frequencies beyond the cutoff
    signal_fft[(np.abs(frequencies) > cutoff_hz)] = 0
    
    # Inverse FFT to get the filtered signal back in time domain
    filtered_signal = ifft(signal_fft)
    
    return filtered_signal.real  # Return the real part of the inverse FFT




def downsample(s, n, phase=0):
    """Decrease sampling rate by integer factor n with included offset phase.
    """
    return s[phase::n]


def upsample(s, n, phase=0):
    """Increase sampling rate by integer factor n  with included offset phase.
    """
    return numpy.roll(numpy.kron(s, numpy.r_[1, numpy.zeros(n-1)]), phase)


def decimate(s, r, n=None, fir=False):
    """Decimation - decrease sampling rate by r. The decimation process filters 
    the input data s with an order n lowpass filter and then resamples the 
    resulting smoothed signal at a lower rate. By default, decimate employs an 
    eighth-order lowpass Chebyshev Type I filter with a cutoff frequency of 
    0.8/r. It filters the input sequence in both the forward and reverse 
    directions to remove all phase distortion, effectively doubling the filter 
    order. If 'fir' is set to True decimate uses an order 30 FIR filter (by 
    default otherwise n), instead of the Chebyshev IIR filter. Here decimate 
    filters the input sequence in only one direction. This technique conserves 
    memory and is useful for working with long sequences.
    """
    if fir:
        if n is None:
            n = 30
        b = signal.firwin(n, 1.0/r)
        a = 1
        f = signal.lfilter(b, a, s)
    else: #iir
        if n is None:
            n = 8
        b, a = signal.cheby1(n, 0.05, 0.8/r)
        f = signal.filtfilt(b, a, s)
    return downsample(f, r)


def interp(s, r, l=4, alpha=0.5):
    """Interpolation - increase sampling rate by integer factor r. Interpolation 
    increases the original sampling rate for a sequence to a higher rate. interp
    performs lowpass interpolation by inserting zeros into the original sequence
    and then applying a special lowpass filter. l specifies the filter length 
    and alpha the cut-off frequency. The length of the FIR lowpass interpolating
    filter is 2*l*r+1. The number of original sample values used for 
    interpolation is 2*l. Ordinarily, l should be less than or equal to 10. The 
    original signal is assumed to be band limited with normalized cutoff 
    frequency 0=alpha=1, where 1 is half the original sampling frequency (the 
    Nyquist frequency). The default value for l is 4 and the default value for 
    alpha is 0.5.
    """
    b = signal.firwin(2*l*r+1, alpha/r);
    a = 1
    return r*signal.lfilter(b, a, upsample(s, r))[r*l+1:-1]


def resample(s, p, q, h=None):
    """Change sampling rate by rational factor. This implementation is based on
    the Octave implementation of the resample function. It designs the 
    anti-aliasing filter using the window approach applying a Kaiser window with
    the beta term calculated as specified by [2].
    
    Ref [1] J. G. Proakis and D. G. Manolakis,
    Digital Signal Processing: Principles, Algorithms, and Applications,
    4th ed., Prentice Hall, 2007. Chap. 6
    Ref [2] A. V. Oppenheim, R. W. Schafer and J. R. Buck, 
    Discrete-time signal processing, Signal processing series,
    Prentice-Hall, 1999
    """
    gcd = fractions.gcd(p,q)
    if gcd>1:
        p=p/gcd
        q=q/gcd
    
    if h is None: #design filter
        #properties of the antialiasing filter
        log10_rejection = -3.0
        stopband_cutoff_f = 1.0/(2.0 * max(p,q))
        roll_off_width = stopband_cutoff_f / 10.0
    
        #determine filter length
        #use empirical formula from [2] Chap 7, Eq. (7.63) p 476
        rejection_db = -20.0*log10_rejection;
        l = numpy.ceil((rejection_db-8.0) / (28.714 * roll_off_width))
  
        #ideal sinc filter
        t = numpy.arange(-l, l + 1)
        ideal_filter=2*p*stopband_cutoff_f*numpy.sinc(2*stopband_cutoff_f*t)  
  
        #determine parameter of Kaiser window
        #use empirical formula from [2] Chap 7, Eq. (7.62) p 474
        beta = signal.kaiser_beta(rejection_db)
          
        #apodize ideal filter response
        h = numpy.kaiser(2*l+1, beta)*ideal_filter

    ls = len(s)
    lh = len(h)

    l = (lh - 1)/2.0
    ly = numpy.ceil(ls*p/float(q))

    #pre and postpad filter response
    nz_pre = numpy.floor(q - numpy.mod(l,q))
    hpad = h[-lh+nz_pre:]

    offset = numpy.floor((l+nz_pre)/q)
    nz_post = 0;
    while numpy.ceil(((ls-1)*p + nz_pre + lh + nz_post )/q ) - offset < ly:
        nz_post += 1
    hpad = hpad[:lh + nz_pre + nz_post]

    #filtering
    xfilt = upfirdn(s, hpad, p, q)

    return xfilt[offset-1:offset-1+ly]


def upfirdn(s, h, p, q):
    """Upsample signal s by p, apply FIR filter as specified by h, and 
    downsample by q. Using fftconvolve as opposed to lfilter as it does not seem
    to do a full convolution operation (and its much faster than convolve).
    """
    return downsample(signal.fftconvolve(h, upsample(s, p)), q)
