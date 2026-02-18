"""Module providing Multirate signal processing functionality.
Largely based on MATLAB's Multirate signal processing toolbox with consultation 
of Octave m-file source code.
"""

import os
import sys
from pathlib import Path

try:
    from .path_setup import ensure_project_paths
except ImportError:
    from path_setup import ensure_project_paths

_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
import fractions
import numpy
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, firwin
from numpy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d



""" Import manual functions """
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
                          order=5,
                          # percentage=1.1,
                          percentage=0.999):
    """
    Downsample (and low-pass filter) high_df to match low_df's sampling rate.
    Keeps the same function signature, but trims to the overlapping time range first.
    """
    if not isinstance(high_df.index, pd.DatetimeIndex):
        raise TypeError("high_df must have a DatetimeIndex.")
    if not isinstance(low_df.index, pd.DatetimeIndex):
        raise TypeError("low_df must have a DatetimeIndex.")

    # 1) Find overlapping portion
    overlap_start = max(high_df.index.min(), low_df.index.min())
    overlap_end   = min(high_df.index.max(), low_df.index.max())

    if overlap_start >= overlap_end:
        raise ValueError("No overlapping time range between high_df and low_df.")
    
    # 2) Subset both DataFrames to the overlap
    high_df = high_df.loc[overlap_start:overlap_end].copy()
    low_df  = low_df.loc[overlap_start:overlap_end].copy()

    # 3) Calculate sampling rates
    high_fs = 1 / func.find_cadence(high_df)  
    low_fs  = 1 / func.find_cadence(low_df)

    # 4) Define cutoff frequency
    cutoff = percentage * low_fs / 2.0  
    if cutoff >= (high_fs / 2.0):
        raise ValueError("Cutoff frequency must be less than half the high_df sampling rate.")

    # 5) Filter high_df columns
    filtered_df = high_df.copy()
    for column in high_df.columns:
        filtered_data = apply_lowpass_filter(high_df[column].values, cutoff, high_fs, order=order)
        filtered_df[column] = filtered_data

    # 6) Reindex/Interpolate to low_df's index
    #    (Assuming you have func.newindex(...) for reindexing)
    resampled_df = func.newindex(filtered_df, low_df.index)

    return resampled_df


def upsample_dataframe(
    original_df, 
    target_df, 
    numtaps=201, 
    cutoff_ratio=0.99,     
    interp_kind='linear',
    edge_margin=None
):
    """
    Upsamples the original_df to match the sampling rate of target_df,
    applying a linear-phase FIR filter with filtfilt.

    Keeps the same signature, but applies overlap trimming first 
    (and optional edge_margin).
    """
    # 1) Basic checks
    if not isinstance(original_df.index, pd.DatetimeIndex):
        raise TypeError("original_df must have a DatetimeIndex.")
    if not isinstance(target_df.index, pd.DatetimeIndex):
        raise TypeError("target_df must have a DatetimeIndex.")
    
    original_df = original_df.sort_index()
    target_df   = target_df.sort_index()
    
    # 2) Find overlapping time range
    overlap_start = max(original_df.index.min(), target_df.index.min())
    overlap_end   = min(original_df.index.max(), target_df.index.max())
    
    if overlap_start >= overlap_end:
        raise ValueError("No overlapping time range between original_df and target_df.")
    
    # 3) Optionally expand edges with edge_margin
    if edge_margin is not None and edge_margin > 0:
        extended_start = overlap_start - pd.Timedelta(seconds=edge_margin)
        extended_start = max(extended_start, original_df.index.min())
        
        extended_end = overlap_end + pd.Timedelta(seconds=edge_margin)
        extended_end = min(extended_end, original_df.index.max())
    else:
        extended_start = overlap_start
        extended_end   = overlap_end
    
    # 4) Subset to the overlap (plus margin if specified)
    original_subset = original_df.loc[extended_start:extended_end].copy()
    target_subset   = target_df.loc[overlap_start:overlap_end].copy()

    # 5) Convert datetime indices to numeric
    t0 = original_subset.index[0]
    original_time = (original_subset.index - t0).total_seconds().values
    target_time   = (target_subset.index   - t0).total_seconds().values

    # 6) Compute sampling rate & design filter
    delta_t_original = np.median(np.diff(original_time))
    Fs_original      = 1.0 / delta_t_original
    
    nyq_original = 0.5 * Fs_original
    cutoff_freq  = cutoff_ratio * nyq_original
    
    if cutoff_freq >= nyq_original:
        raise ValueError("Cutoff frequency must be less than the original Nyquist.")

    fir_coeff = firwin(
        numtaps,
        cutoff=cutoff_freq / nyq_original,
        window='hamming',
        pass_zero='lowpass'
    )
    
    # 7) Filter & Interpolate
    data_cols = original_subset.columns.tolist()
    upsampled_data = {'datetime': target_subset.index}
    
    for col in data_cols:
        data_original = original_subset[col].values
        
        # -- Low-pass filter
        data_filtered = filtfilt(fir_coeff, [1.0], data_original)
        
        # -- Interpolate to target_time
        interp_func    = interp1d(original_time, data_filtered, kind=interp_kind,
                                  bounds_error=False, fill_value='extrapolate')
        data_upsampled = interp_func(target_time)
        
        upsampled_data[col] = data_upsampled

    # 8) Construct final result
    upsampled_df = pd.DataFrame(upsampled_data).set_index('datetime')
    
    # 9) (Optional) final trim to remove any margin region if you wish
    #    upsampled_df = upsampled_df.loc[overlap_start:overlap_end]

    return upsampled_df



# def upsample_dataframe(original_df, target_df, numtaps=101, cutoff_ratio=0.85, interp_kind='cubic'):
#     """
#     Upsamples the original_df to match the sampling rate of target_df, applying a linear-phase FIR filter.

#     Parameters:
#         original_df (pd.DataFrame): DataFrame containing the original time series.
#                                     The index should be datetime-like.
#         target_df (pd.DataFrame): DataFrame containing the target time series with higher sampling rate.
#                                   The index should be datetime-like.
#         numtaps (int, optional): Number of filter coefficients (taps) for the FIR filter. Default is 101.
#         cutoff_ratio (float, optional): Ratio of Nyquist frequency to set the cutoff frequency (default 0.89).
#         interp_kind (str, optional): Kind of interpolation ('linear', 'cubic', etc.). Default is 'cubic'.

#     Returns:
#         pd.DataFrame: Upsampled original_df with time matching target_df.
#     """
    
#     # Ensure the indices are datetime and sorted
#     if not isinstance(original_df.index, pd.DatetimeIndex):
#         raise TypeError("original_df must have a DatetimeIndex.")
#     if not isinstance(target_df.index, pd.DatetimeIndex):
#         raise TypeError("target_df must have a DatetimeIndex.")
    
#     original_df = original_df.sort_index()
#     target_df = target_df.sort_index()
    
#     # Determine data columns to process
#     data_cols = original_df.columns.tolist()
    
#     # Convert datetime indices to numerical values (e.g., seconds since start)
#     t0 = original_df.index[0]
#     original_time = (original_df.index - t0).total_seconds().values
#     target_time = (target_df.index - t0).total_seconds().values
    
#     # Compute sampling rates
#     delta_t_original = np.median(np.diff(original_time))
#     delta_t_target = np.median(np.diff(target_time))
    
#     Fs_original = 1.0 / delta_t_original
#     Fs_target = 1.0 / delta_t_target
    
#     if Fs_target <= Fs_original:
#         raise ValueError("Target sampling rate must be higher than original sampling rate for upsampling.")
    
#     # Design a linear-phase FIR low-pass filter
#     nyq_original = 0.5 * Fs_original
#     cutoff_freq = cutoff_ratio * nyq_original
#     normalized_cutoff = cutoff_freq / nyq_original  # Normalize with original Nyquist
    
#     # Ensure the cutoff is below the Nyquist frequency
#     if cutoff_freq >= nyq_original:
#         raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyq_original} Hz)")
    
#     # Use a Hamming window for the FIR filter to minimize ripple
#     fir_coeff = firwin(numtaps, cutoff=normalized_cutoff, window='hamming', pass_zero='lowpass')
    
#     # Initialize a dictionary to hold upsampled data
#     upsampled_data = {'datetime': target_df.index}
    
#     # Process each data column
#     for col in data_cols:
#         data_original = original_df[col].values
        
#         # Step 1: Apply FIR low-pass filter using filtfilt for zero-phase
#         data_filtered = filtfilt(fir_coeff, [1.0], data_original)
        
#         # Step 2: Interpolate to target_time using higher-order interpolation
#         interp_func = interp1d(original_time, data_filtered, kind=interp_kind, fill_value='extrapolate')
#         data_upsampled = interp_func(target_time)
        
#         upsampled_data[col] = data_upsampled
    
#     # Create the upsampled DataFrame
#     upsampled_df = pd.DataFrame(upsampled_data).set_index('datetime')
    
#     return upsampled_df



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








"""
Solar Orbiter merged MAG tone removal (Solar-Orbiter-specific).

The merged MAG product is typically distributed at 256 Hz (daily) and 4096 Hz (burst), in SRF or RTN.
Power spectra often contain narrowband "tones" (sharp lines with harmonics) that are instrumental /
spacecraft contamination (not physical turbulence). Two prominent families reported for the merged product:
- Clock family: 8 Hz, 16 Hz, and harmonics (time-varying amplitude).
- ~1.33 Hz family (and harmonics) that appears in the SCM X_SRF channel (component-restricted).

Operational goal:
- After cleaning, the PSD at tone frequencies matches a smooth baseline inferred from neighboring
  frequencies (no residual peaks, no notches), while preserving broadband turbulence and diagnostics
  away from edited bands.
- Avoid PSP-specific assumptions.

Implementation sketch:
- Stage 1 (optional): joint lock-in / quadrature demodulation (rank-1) to estimate and subtract a
  coherent contaminant (recommended for the clock family).
- Stage 2: PSD-consistent STFT-domain equalization inside tone bands using shoulder-derived baselines.
  A single scalar gain per (f,t) is applied jointly to all selected components to preserve cross-phase.

Multivariate safety:
- Stage 2 equalization uses a *single scalar gain* per (f,t) applied to all components inside a tone band.
  This preserves cross-phase outside the edited bands.
- SRF (~1.33 Hz) is disabled by default unless the user explicitly provides srf_apply_to=[idx] that maps
  to the SCM X_SRF channel in their input array.

This file is intentionally single-module and copy/paste friendly.
"""



from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np
from scipy import signal


# --------------------------
# Configuration
# --------------------------

@dataclass(frozen=True)
class LockInConfig:
    enabled: bool = True

    # bandpass half-width (Hz) used to isolate the tone neighborhood
    bw_hz: float = 0.55

    # envelope LP cutoff in the baseband (Hz). For low-f tones you usually need higher.
    env_lp_hz: float = 0.35
    lowf_env_lp_hz: float = 0.9
    lowf_thresh_hz: float = 2.0

    # center-frequency refinement around nominal f0 (Hz)
    center_search_hz: float = 0.35
    peak_prom_db: float = 3.0  # required peak prominence vs shoulders to accept f_center update

    # SNR gate
    snr_gate_db: float = 4.0
    snr_win_s: float = 4.0

    # filters
    bp_order: int = 4
    lp_order: int = 4

    # segmentation
    min_seg_s: float = 4.0

    # do-no-harm / joint-polarization checks
    rank1_min: float = 0.75      # require dominant eigenvalue fraction in baseband covariance
    pol_stability_deg: float = 25.0  # reject if polarization direction varies too much across subwindows
    min_gate_cycles: float = 6.0     # require at least this many cycles of contiguous gate to subtract


@dataclass(frozen=True)
class EqualizeConfig:
    enabled: bool = True

    # STFT params (must satisfy COLA for the chosen window/hop)
    nperseg: int = 8192
    noverlap: int = 4096
    window: str = "hann"

    # shoulders: [f0 - s2*bw, f0 - s1*bw] and [f0 + s1*bw, f0 + s2*bw]
    shoulder_mult: Tuple[float, float] = (2.0, 3.5)

    # center-frequency refinement in the STFT mean spectrum (Hz)
    center_search_hz: float = 0.35
    peak_prom_db: float = 2.5


    # time-aggregation statistic for intermittent tones (median can miss them)
    center_time_quantile: float = 0.90
    dilate_time_quantile: float = 0.90

    # projection equalization (rank-1 subspace): avoids notches by not attenuating orthogonal content
    projection_equalize: bool = True
    rank1_min: float = 0.65
    # optional band dilation (captures drift/leakage/sidebands)
    dilate_enabled: bool = True
    dilate_db: float = 2.5
    dilate_max_mult: float = 3.0

    # gain policy
    attenuate_only: bool = True
    g_floor: float = 0.05
    g_ceil: float = 1.50

    # segmentation
    min_seg_s: float = 10.0


@dataclass(frozen=True)
class ToneBand:
    f0_hz: float
    bw_hz: float
    family: str = "generic"  # "clock" | "srf" | "generic"


@dataclass(frozen=True)
class DetectConfig:
    """
    Optional peak-based tone detection to catch lines not exactly at the nominal families.
    This is the main refinement when "the first tone is not captured" due to slight offsets,
    unexpected additional lines, or aliasing.

    Strategy:
    - Build a trace PSD on finite-only data (vector magnitude across components).
    - Smooth log-PSD and find narrow peaks by prominence in dB.
    - Optionally classify peaks into clock/SRF families by proximity to harmonics.
    - Return extra ToneBand entries centered on detected peaks, with bw inferred from peak width.
    """
    enabled: bool = False
    fmin_hz: float = 0.2
    fmax_hz: float = 45.0
    smooth_bins: int = 9
    prom_db: float = 6.0
    max_width_bins: int = 25
    bw_floor_hz: float = 0.20
    bw_mult: float = 2.0
    family_tol_hz: float = 0.20   # tolerance to assign to 8 Hz*k or 1.33*k
    max_peaks: int = 50


# --------------------------
# Utilities
# --------------------------

def _finite_segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, bool)
    if mask.sum() == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    out = []
    for s, e in zip(starts, ends):
        s = int(s); e = int(e)
        if e - s >= min_len:
            out.append((s, e))
    return out


def _butter_bandpass_sos(fs: float, f1: float, f2: float, order: int) -> Optional[np.ndarray]:
    ny = 0.5 * fs
    if f2 <= 0 or f1 >= ny:
        return None
    f1c = max(1e-6, f1 / ny)
    f2c = min(0.999999, f2 / ny)
    if f2c <= f1c:
        return None
    return signal.butter(order, [f1c, f2c], btype="bandpass", output="sos")


def _butter_lowpass_sos(fs: float, fc: float, order: int) -> np.ndarray:
    ny = 0.5 * fs
    fc = min(0.999999, max(1e-6, fc / ny))
    return signal.butter(order, fc, btype="lowpass", output="sos")


def _welch_trace_psd(X: np.ndarray, fs: float, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trace PSD estimate: sum of Welch PSDs over components (finite-only segment assumed).
    X: (n, m)
    """
    X = np.asarray(X, float)
    m = X.shape[1]
    Psum = None
    for j in range(m):
        f, P = signal.welch(
            X[:, j],
            fs=fs,
            window="hann",
            nperseg=min(nperseg, X.shape[0]),
            noverlap=min(nperseg//2, (min(nperseg, X.shape[0])//2)),
            detrend="constant",
            scaling="density",
        )
        if Psum is None:
            Psum = P.copy()
        else:
            Psum += P
    return f, Psum


def _peak_center_from_psd(f: np.ndarray, P: np.ndarray, f0: float, search: float, shoulder_bw: float) -> Tuple[float, float]:
    """
    Return (f_center, prom_db) where prom_db is peak prominence vs shoulders.
    """
    f = np.asarray(f); P = np.asarray(P)
    m = (f >= f0 - search) & (f <= f0 + search)
    if not np.any(m):
        return f0, 0.0
    ff = f[m]; PP = P[m]
    ip = int(np.argmax(PP))
    fpk = float(ff[ip])
    Ppk = float(PP[ip])

    left = (f >= fpk - 3*shoulder_bw) & (f <= fpk - 2*shoulder_bw)
    right = (f >= fpk + 2*shoulder_bw) & (f <= fpk + 3*shoulder_bw)
    if not left.any() and not right.any():
        return fpk, 0.0
    vals = []
    if left.any():
        vals.append(np.median(P[left]))
    if right.any():
        vals.append(np.median(P[right]))
    base = float(np.median(vals))
    prom_db = 10.0 * np.log10((Ppk + 1e-30) / (base + 1e-30))
    return fpk, float(prom_db)



def detect_tone_bands_from_psd(
    B: np.ndarray,
    fs: float,
    cfg: DetectConfig,
) -> List[ToneBand]:
    """
    Detect narrowband spectral lines in the trace PSD and return ToneBand entries.
    Intended as a *supplement* to nominal family lists when a tone is missed.

    Notes:
    - Uses only finite samples.
    - Conservative by default: requires strong prominence in dB.
    """
    if not cfg.enabled:
        return []

    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    # trace time series (vector magnitude)
    tr = np.sqrt(np.nansum(B*B, axis=1))
    m = np.isfinite(tr)
    if m.sum() < 1024:
        return []

    # Welch PSD (avoid zero-fill)
    x = tr[m]
    nper = min(8192, x.size)
    f, P = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nper,
        noverlap=min(nper//2, nper-1),
        detrend="constant",
        scaling="density",
    )

    sel = (f >= cfg.fmin_hz) & (f <= min(cfg.fmax_hz, 0.5*fs-1e-6))
    f = f[sel]; P = P[sel]
    if f.size < 10:
        return []

    logP = 10.0*np.log10(P + 1e-30)

    # smooth with a median filter to suppress random spikes but keep narrow lines
    k = int(cfg.smooth_bins)
    if k % 2 == 0:
        k += 1
    if k >= 3:
        logPs = signal.medfilt(logP, kernel_size=k)
    else:
        logPs = logP

    # residual above smooth baseline
    resid = logP - logPs
    peaks, props = signal.find_peaks(resid, prominence=cfg.prom_db)
    if peaks.size == 0:
        return []

    # peak widths in bins at half-prominence
    widths, w_heights, left_ips, right_ips = signal.peak_widths(resid, peaks, rel_height=0.5)

    # sort by prominence
    prom = props.get("prominences", np.zeros_like(peaks, float))
    order = np.argsort(prom)[::-1][: cfg.max_peaks]

    df = float(np.median(np.diff(f))) if f.size > 1 else 0.0
    out: List[ToneBand] = []

    for ii in order:
        pidx = int(peaks[ii])
        fpk = float(f[pidx])
        wbins = float(widths[ii])
        wbins = min(wbins, float(cfg.max_width_bins))
        bw = max(cfg.bw_floor_hz, cfg.bw_mult * 0.5 * wbins * df) if df > 0 else cfg.bw_floor_hz

        # classify into families if close to harmonics
        fam = "generic"
        # clock 8 Hz harmonics
        k8 = int(round(fpk/8.0)) if fpk > 0 else 0
        if k8 >= 1 and abs(fpk - 8.0*k8) <= cfg.family_tol_hz:
            fam = "clock"
        # SRF 1.33 harmonics
        k1 = int(round(fpk/1.33)) if fpk > 0 else 0
        if k1 >= 1 and abs(fpk - 1.33*k1) <= cfg.family_tol_hz:
            fam = "srf"

        out.append(ToneBand(f0_hz=fpk, bw_hz=bw, family=fam))

    # de-duplicate near-identical peaks
    out_sorted = sorted(out, key=lambda b: b.f0_hz)
    dedup: List[ToneBand] = []
    for b in out_sorted:
        if not dedup or abs(b.f0_hz - dedup[-1].f0_hz) > max(0.5*b.bw_hz, 0.08):
            dedup.append(b)

    return dedup
def default_clock_tones(fs: float, max_hz: Optional[float] = 45.0) -> List[float]:
    nyq = 0.5 * fs
    fmax = min(nyq - 1e-6, max_hz) if max_hz is not None else (nyq - 1e-6)
    out = []
    k = 1
    while 8.0 * k <= fmax:
        out.append(8.0 * k)
        k += 1
    return out


def default_srf_tones(fs: float, max_hz: Optional[float] = 45.0) -> List[float]:
    nyq = 0.5 * fs
    fmax = min(nyq - 1e-6, max_hz) if max_hz is not None else (nyq - 1e-6)
    out = []
    k = 1
    while 1.33 * k <= fmax:
        out.append(1.33 * k)
        k += 1
    return out


def make_default_bands(
    fs: float,
    clock_bw_hz: float = 0.35,
    srf_bw_hz: float = 0.25,
    *,
    max_hz_clock: Optional[float] = None,
    max_hz_srf: float = 45.0,
) -> List[ToneBand]:
    """
    Default tone-band list.

    Notes
    -----
    - Clock family uses 8 Hz harmonics, so it automatically includes 16 Hz and higher harmonics.
    - By default we include clock harmonics up to min(Nyquist, 256 Hz) to cover typical science
      ranges without exploding the number of bands at fs=4096.
    - SRF (~1.33 Hz) harmonics can be numerous; we keep their default max modest (45 Hz) and they
      are *disabled unless srf_apply_to is provided* in clean_solo_merged_mag_tones().
    """
    nyq = 0.5 * fs
    if max_hz_clock is None:
        max_hz_clock = min(nyq - 1e-6, 256.0)
    max_hz_clock = float(min(nyq - 1e-6, max_hz_clock))
    max_hz_srf = float(min(nyq - 1e-6, max_hz_srf))

    bands: List[ToneBand] = []
    for f0 in default_clock_tones(fs, max_hz=max_hz_clock):
        bands.append(ToneBand(f0_hz=f0, bw_hz=clock_bw_hz, family="clock"))
    for f0 in default_srf_tones(fs, max_hz=max_hz_srf):
        bands.append(ToneBand(f0_hz=f0, bw_hz=srf_bw_hz, family="srf"))
    return bands


# --------------------------
# Stage 1: joint lock-in (rank-1) subtraction
# --------------------------

def _principal_vector(E: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    E: (n, m) complex baseband envelopes
    Return (u, rank1_frac) where u is unit-norm principal eigenvector of covariance.
    """
    # covariance m x m
    C = (E.conj().T @ E) / max(E.shape[0], 1)
    # hermitian eig
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    lam1 = float(np.real(w[0])) if w.size else 0.0
    tr = float(np.real(w).sum()) if w.size else 0.0
    u = V[:, 0] if V.size else np.zeros((E.shape[1],), dtype=complex)
    # normalize
    nu = np.linalg.norm(u) + 1e-30
    u = u / nu
    rank1 = lam1 / (tr + 1e-30)
    return u, float(rank1)


def _pol_angle_deg(u1: np.ndarray, u2: np.ndarray) -> float:
    """
    Angle between complex vectors modulo global phase (subspace angle).
    """
    # align phase via inner product
    ip = np.vdot(u1, u2)
    if np.abs(ip) < 1e-30:
        return 90.0
    u2a = u2 * np.exp(-1j * np.angle(ip))
    c = np.clip(np.abs(np.vdot(u1, u2a)) / ((np.linalg.norm(u1)+1e-30)*(np.linalg.norm(u2a)+1e-30)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def remove_tones_lockin_joint(
    X: np.ndarray,
    fs: float,
    tone_freqs_hz: Sequence[float],
    cfg: LockInConfig,
    apply_to: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Joint (multi-component) lock-in subtraction using a rank-1 coherent model in a narrow band.

    Key refinement vs a naive implementation:
    - Build an SNR gate *before* estimating polarization/envelopes (based on bandpassed energy vs sidebands).
      This makes the estimator intermittent-safe and prevents the principal-vector estimate from being diluted
      by long intervals where the tone is absent.
    - Estimate the rank-1 polarization direction using only gated samples.

    X: (n, m) real
    apply_to: optional subset of component indices to be cleaned (others left untouched)
    """
    X = np.asarray(X, float)
    n, m = X.shape
    Y = X.copy()

    if not cfg.enabled:
        return Y, {"enabled": False, "segments": 0}

    if apply_to is None:
        apply_to = list(range(m))
    apply_to = list(apply_to)

    finite = np.all(np.isfinite(Y[:, apply_to]), axis=1)
    min_seg = int(round(cfg.min_seg_s * fs))
    segs = _finite_segments(finite, min_len=min_seg)
    if not segs:
        return Y, {"enabled": True, "segments": 0}

    snr_win = max(64, int(round(cfg.snr_win_s * fs)))
    ker = np.ones(snr_win) / snr_win

    edits: List[Dict] = []

    for f0_nom in tone_freqs_hz:
        if f0_nom <= 0.0 or f0_nom >= 0.5 * fs:
            continue

        for i0, i1 in segs:
            seg = Y[i0:i1, :]

            # --- optional center refinement on segment trace PSD ---
            f_psd, P_psd = _welch_trace_psd(seg[:, apply_to], fs=fs, nperseg=min(4096, i1 - i0))
            f0, prom_db = _peak_center_from_psd(
                f_psd, P_psd,
                f0=float(f0_nom),
                search=cfg.center_search_hz,
                shoulder_bw=max(cfg.bw_hz, 0.15),
            )
            if prom_db < cfg.peak_prom_db:
                f0 = float(f0_nom)

            bw = float(cfg.bw_hz)
            sos_bp = _butter_bandpass_sos(fs, f0 - bw, f0 + bw, order=cfg.bp_order)
            if sos_bp is None:
                continue

            env_lp = cfg.lowf_env_lp_hz if f0 < cfg.lowf_thresh_hz else cfg.env_lp_hz
            sos_lp = _butter_lowpass_sos(fs, env_lp, order=cfg.lp_order)

            # sidebands for gating (same bw scale)
            sb1 = (f0 - 3 * bw, f0 - 2 * bw)
            sb2 = (f0 + 2 * bw, f0 + 3 * bw)
            sos_sb1 = _butter_bandpass_sos(fs, sb1[0], sb1[1], order=cfg.bp_order)
            sos_sb2 = _butter_bandpass_sos(fs, sb2[0], sb2[1], order=cfg.bp_order)

            # --- bandpass each component in apply_to ---
            bp = np.zeros((i1 - i0, len(apply_to)))
            for jj, j in enumerate(apply_to):
                bp[:, jj] = signal.sosfiltfilt(sos_bp, seg[:, j])

            # --- SNR gate using summed in-band energy vs sidebands (pre-envelope) ---
            p_bp = signal.fftconvolve(np.sum(bp * bp, axis=1), ker, mode="same")

            p_sb = np.zeros_like(p_bp)
            cnt = 0
            if sos_sb1 is not None:
                sb = np.zeros_like(bp)
                for jj, j in enumerate(apply_to):
                    sb[:, jj] = signal.sosfiltfilt(sos_sb1, seg[:, j])
                p_sb += signal.fftconvolve(np.sum(sb * sb, axis=1), ker, mode="same")
                cnt += 1
            if sos_sb2 is not None:
                sb = np.zeros_like(bp)
                for jj, j in enumerate(apply_to):
                    sb[:, jj] = signal.sosfiltfilt(sos_sb2, seg[:, j])
                p_sb += signal.fftconvolve(np.sum(sb * sb, axis=1), ker, mode="same")
                cnt += 1
            if cnt > 0:
                p_sb /= cnt
            else:
                p_sb[:] = np.median(p_bp)

            snr_db = 10.0 * np.log10((p_bp + 1e-30) / (p_sb + 1e-30))
            gate = snr_db >= cfg.snr_gate_db

            # enforce minimum contiguous duration (cycles)
            min_len = int(round(cfg.min_gate_cycles * fs / max(f0, 1e-6)))
            if min_len > 1 and gate.any():
                d = np.diff(gate.astype(np.int8))
                s = np.where(d == 1)[0] + 1
                e = np.where(d == -1)[0] + 1
                if gate[0]:
                    s = np.r_[0, s]
                if gate[-1]:
                    e = np.r_[e, gate.size]
                gate2 = np.zeros_like(gate)
                for a, b in zip(s, e):
                    if b - a >= min_len:
                        gate2[a:b] = True
                gate = gate2

            if gate.sum() == 0:
                continue

            # --- baseband complex envelope per component (compute only after gate is nonempty) ---
            analytic = signal.hilbert(bp, axis=0)  # complex
            t = np.arange(i1 - i0) / fs
            mix = np.exp(-1j * 2 * np.pi * f0 * t)[:, None]
            bb = analytic * mix

            # lowpass in I/Q separately for stability
            env_r = signal.sosfiltfilt(sos_lp, np.real(bb), axis=0)
            env_i = signal.sosfiltfilt(sos_lp, np.imag(bb), axis=0)
            E = env_r + 1j * env_i  # (nseg, m_apply)

            Eg = E[gate, :]
            if Eg.shape[0] < max(64, int(round(0.5 * fs))):
                continue

            # --- polarization stability check across subwindows (on gated samples) ---
            sub_win = max(256, int(round(2.0 * fs)))
            us = []
            for a in range(0, Eg.shape[0], sub_win):
                b = min(Eg.shape[0], a + sub_win)
                if b - a < max(64, int(round(0.2 * fs))):
                    continue
                u_sub, _ = _principal_vector(Eg[a:b, :])
                if np.linalg.norm(u_sub) > 0:
                    us.append(u_sub)

            ok_pol = True
            if len(us) >= 2:
                for u_prev, u_cur in zip(us[:-1], us[1:]):
                    if _pol_angle_deg(u_prev, u_cur) > cfg.pol_stability_deg:
                        ok_pol = False
                        break
            if not ok_pol:
                continue

            u, rank1 = _principal_vector(Eg)
            if rank1 < cfg.rank1_min:
                continue

            # scalar envelope on the principal direction
            c = E @ np.conj(u)  # (nseg,)

            # reconstruct vector tone in apply_to channels
            carrier = np.exp(1j * 2 * np.pi * f0 * t)
            tone_apply = np.real((c[:, None] * carrier[:, None]) * (u[None, :]))

            # apply subtraction only where gate is true
            seg2 = seg.copy()
            for jj, j in enumerate(apply_to):
                # shrinkage weight: subtract only the coherent fraction implied by the SNR gate
                w = gate.astype(float) * np.clip(1.0 - 10.0**(-0.1*snr_db), 0.0, 1.0)
                seg2[:, j] = seg2[:, j] - tone_apply[:, jj] * w

            Y[i0:i1, :] = seg2

            edits.append({
                "seg": (int(i0), int(i1)),
                "f0_nom": float(f0_nom),
                "f0_used": float(f0),
                "bw_hz": float(bw),
                "prom_db": float(prom_db),
                "rank1": float(rank1),
                "gate_frac": float(gate.mean()),
            })

    return Y, {"enabled": True, "segments": len(segs), "edits": edits}

def _stft_multi(seg: np.ndarray, fs: float, cfg: EqualizeConfig):
    """
    seg: (n, m)
    Return f, t, Z: (nf, nt, m)
    """
    seg = np.asarray(seg, float)
    n, m = seg.shape
    Zs = []
    f = t = None
    for j in range(m):
        fj, tj, Zj = signal.stft(
            seg[:, j],
            fs=fs,
            window=cfg.window,
            nperseg=min(cfg.nperseg, n),
            noverlap=min(cfg.noverlap, min(cfg.nperseg, n)-1),
            detrend=False,
            boundary="zeros",
            padded=True,
        )
        if f is None:
            f, t = fj, tj
        Zs.append(Zj)
    Z = np.stack(Zs, axis=2)
    return f, t, Z


def _istft_multi(Z: np.ndarray, fs: float, cfg: EqualizeConfig, n_out: int) -> np.ndarray:
    """
    Z: (nf, nt, m)
    """
    nf, nt, m = Z.shape
    out = np.zeros((n_out, m), float)
    for j in range(m):
        _, x = signal.istft(
            Z[:, :, j],
            fs=fs,
            window=cfg.window,
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            input_onesided=True,
            boundary=True,
        )
        out[:, j] = x[:n_out]
    return out


def equalize_tone_bands_stft_multicomponent(
    X: np.ndarray,
    fs: float,
    bands: Sequence[ToneBand],
    cfg: EqualizeConfig,
    apply_to: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Multivariate STFT magnitude equalization:
    - Compute vector magnitude A(f,t) across components.
    - Build a smooth baseline A_base(f,t) from shoulders.
    - Inside tone bands, apply a single scalar gain g(f,t) to *all* components:
        Z_new(f,t,j) = g(f,t) * Z(f,t,j).

    This preserves cross-phase and minimizes bias to coherence/helicity outside edited bands.
    """
    X = np.asarray(X, float)
    n, m = X.shape
    Y = X.copy()

    if not cfg.enabled:
        return Y, {"enabled": False, "segments": 0}

    if apply_to is None:
        apply_to = list(range(m))
    apply_to = list(apply_to)

    finite = np.all(np.isfinite(Y[:, apply_to]), axis=1)
    min_seg = int(round(cfg.min_seg_s * fs))
    min_seg = max(min_seg, int(cfg.nperseg))
    segs = _finite_segments(finite, min_len=min_seg)
    if not segs:
        return Y, {"enabled": True, "segments": 0}

    edits = []

    for i0, i1 in segs:
        seg = Y[i0:i1, :].copy()

        f, tt, Z = _stft_multi(seg[:, apply_to], fs=fs, cfg=cfg)  # Z: (nf,nt,m_apply)
        nf, nt, ma = Z.shape
        A = np.sqrt(np.sum(np.abs(Z)**2, axis=2))  # vector magnitude

        # precompute robust "high-activity" spectra for center refinement and dilation
        # (median can miss strongly intermittent tones)
        logA = np.log(A + 1e-30)
        q_center = float(np.clip(cfg.center_time_quantile, 0.5, 0.99))
        q_dilate = float(np.clip(cfg.dilate_time_quantile, 0.5, 0.99))
        mean_logA = np.quantile(logA, q_center, axis=1)  # (nf,)
        mean_logA_d = np.quantile(logA, q_dilate, axis=1)  # (nf,)

        df = float(np.median(np.diff(f))) if f.size > 1 else 0.0

        for band in bands:
            f0_nom = float(band.f0_hz)
            if f0_nom <= 0.0 or f0_nom >= 0.5 * fs:
                continue

            # --- center refinement in mean spectrum ---
            msearch = (f >= f0_nom - cfg.center_search_hz) & (f <= f0_nom + cfg.center_search_hz)
            f0 = f0_nom
            prom_db = 0.0
            if msearch.any():
                ip = int(np.argmax(mean_logA[msearch]))
                fpk = float(f[msearch][ip])
                # prominence vs shoulders in logA
                bw0 = float(band.bw_hz)
                # shoulders around fpk
                left = (f >= fpk - 3*bw0) & (f <= fpk - 2*bw0)
                right = (f >= fpk + 2*bw0) & (f <= fpk + 3*bw0)
                bases = []
                if left.any(): bases.append(np.median(mean_logA[left]))
                if right.any(): bases.append(np.median(mean_logA[right]))
                if bases:
                    base = float(np.median(bases))
                    prom_db = 10.0/np.log(10.0) * (float(mean_logA[msearch][ip]) - base)
                    if prom_db >= cfg.peak_prom_db:
                        f0 = fpk

            bw = float(band.bw_hz)
            # ensure at least a few bins
            if df > 0:
                bw = max(bw, 4.0*df)

            # core band mask
            band_mask = (f >= f0 - bw) & (f <= f0 + bw)
            if not band_mask.any():
                continue

            # optional dilation based on excess vs baseline in the *mean* spectrum
            if cfg.dilate_enabled and df > 0:
                bw_max = cfg.dilate_max_mult * bw
                neigh = (f >= f0 - bw_max) & (f <= f0 + bw_max)
                if neigh.any():
                    # build a mean-spectrum baseline from shoulders
                    s1, s2 = cfg.shoulder_mult
                    left = (f >= f0 - s2*bw) & (f <= f0 - s1*bw)
                    right = (f >= f0 + s1*bw) & (f <= f0 + s2*bw)
                    if (not left.any()) or (not right.any()):
                        # broaden fallback shoulders
                        left = (f >= max(0.0, f0 - 6*bw)) & (f < f0 - bw)
                        right = (f > f0 + bw) & (f <= min(0.5*fs, f0 + 6*bw))
                    if left.any() and right.any():
                        base_left = np.median(mean_logA[left])
                        base_right = np.median(mean_logA[right])
                        # linear baseline across neighborhood
                        ff = f[neigh]
                        u = (ff - ff.min())/(ff.max()-ff.min() + 1e-12)
                        base = base_left*(1-u) + base_right*u
                        excess_db = 10.0/np.log(10.0) * (mean_logA_d[neigh] - base)
                        cand = excess_db >= cfg.dilate_db
                        if cand.any():
                            pk = int(np.argmin(np.abs(f - f0)))
                            neigh_idx = np.where(neigh)[0]
                            cand_idx = neigh_idx[cand]

                            # candidate bins + core band; enforce a single contiguous region around the peak
                            mask = np.zeros_like(band_mask)
                            mask[cand_idx] = True
                            mask[band_mask] = True
                            mask &= neigh

                            idx = np.where(mask)[0]
                            if idx.size > 0:
                                d = np.diff(idx)
                                split = np.where(d > 1)[0]
                                starts = np.r_[0, split + 1]
                                ends = np.r_[split + 1, idx.size]

                                chosen = None
                                for a, b in zip(starts, ends):
                                    run = idx[a:b]
                                    if pk >= run[0] and pk <= run[-1]:
                                        chosen = run
                                        break

                                if chosen is None:
                                    scores = [np.max(mean_logA_d[idx[a:b]]) for a, b in zip(starts, ends)]
                                    jbest = int(np.argmax(scores))
                                    chosen = idx[starts[jbest]:ends[jbest]]

                                band_mask2 = np.zeros_like(band_mask)
                                band_mask2[chosen] = True
                                band_mask = band_mask2

            # shoulders (based on original bw around f0)
            s1, s2 = cfg.shoulder_mult
            left = (f >= f0 - s2*bw) & (f <= f0 - s1*bw)
            right = (f >= f0 + s1*bw) & (f <= f0 + s2*bw)
            if not left.any():
                left = (f >= max(0.0, f0 - 6*bw)) & (f < f0 - bw)
            if not right.any():
                right = (f > f0 + bw) & (f <= min(0.5*fs, f0 + 6*bw))

            if not left.any() and not right.any():
                continue

            # baseline in log domain per time frame
            if left.any():
                ml = np.median(np.log(A[left, :] + 1e-30), axis=0)
            else:
                ml = np.median(np.log(A + 1e-30), axis=0)
            if right.any():
                mr = np.median(np.log(A[right, :] + 1e-30), axis=0)
            else:
                mr = np.median(np.log(A + 1e-30), axis=0)

            fb = f[band_mask]
            if fb.size == 1:
                targ_log = 0.5*(ml + mr)
                Abase = np.exp(targ_log)[None, :]
            else:
                denom = (fb.max() - fb.min() + 1e-12)
                u = ((fb - fb.min())/denom)[:, None]
                targ = ml[None, :]*(1-u) + mr[None, :]*u
                Abase = np.exp(targ)

            Zb = Z[band_mask, :, :]  # (nb,nt,ma)
            if cfg.projection_equalize and ma >= 2:
                # estimate dominant complex direction in this band using high-activity samples
                Ab = A[band_mask, :]  # (nb,nt)
                thr = np.quantile(Ab, 0.85)  # focuses on tone-present frames (intermittency-safe)
                sel = Ab >= thr
                W = Zb.reshape(-1, ma)
                if sel.any():
                    Wsel = W[sel.reshape(-1)]
                else:
                    Wsel = W
                rank1 = 0.0
                uvec = None
                if Wsel.shape[0] >= max(32, 16*ma):
                    C = (Wsel.conj().T @ Wsel) / max(Wsel.shape[0], 1)
                    w, V = np.linalg.eigh(C)
                    ii = np.argsort(w)[::-1]
                    w = w[ii]; V = V[:, ii]
                    lam1 = float(np.real(w[0]))
                    tr = float(np.real(w).sum())
                    rank1 = lam1 / (tr + 1e-30)
                    uvec = V[:, 0]
                    uvec = uvec / (np.linalg.norm(uvec) + 1e-30)
            
                if (uvec is not None) and (rank1 >= cfg.rank1_min):
                    # project, then only attenuate the rank-1 (tone-like) subspace to match the baseline
                    c = np.tensordot(Zb, np.conj(uvec), axes=([2], [0]))  # (nb,nt)
                    Zpar = c[:, :, None] * uvec[None, None, :]
                    Zperp = Zb - Zpar
                    perp = np.sqrt(np.sum(np.abs(Zperp)**2, axis=2))
                    cabs = np.abs(c) + 1e-30
                    # target parallel magnitude to achieve total magnitude ~= Abase without suppressing orthogonal content
                    ctar = np.sqrt(np.clip(Abase**2 - perp**2, 0.0, np.inf))
                    if cfg.attenuate_only:
                        ctar = np.minimum(ctar, cabs)
                    ratio = ctar / cabs
                    ratio = np.clip(ratio, cfg.g_floor, cfg.g_ceil)
                    cnew = (ratio * cabs) * (c / cabs)
                    Znew = Zperp + cnew[:, :, None] * uvec[None, None, :]
                    Z[band_mask, :, :] = Znew
                else:
                    # fallback: scalar gain on all components
                    Acur = A[band_mask, :] + 1e-30
                    g = Abase / Acur
                    if cfg.attenuate_only:
                        g = np.minimum(g, 1.0)
                    g = np.clip(g, cfg.g_floor, cfg.g_ceil)
                    Z[band_mask, :, :] = Zb * g[:, :, None]
            else:
                # scalar gain on all components (single-component case or projection disabled)
                Acur = A[band_mask, :] + 1e-30
                g = Abase / Acur
                if cfg.attenuate_only:
                    g = np.minimum(g, 1.0)
                g = np.clip(g, cfg.g_floor, cfg.g_ceil)
                Z[band_mask, :, :] = Zb * g[:, :, None]

            edits.append({
                "seg": (int(i0), int(i1)),
                "family": str(band.family),
                "f0_nom": float(f0_nom),
                "f0_used": float(f0),
                "bw_eff_hz": float(bw),
                "prom_db": float(prom_db),
            })

        # reconstruct apply_to, insert back
        rec_apply = _istft_multi(Z, fs=fs, cfg=cfg, n_out=i1-i0)
        seg[:, apply_to] = rec_apply
        Y[i0:i1, :] = seg

    return Y, {"enabled": True, "segments": len(segs), "edits": edits}




def equalize_tone_bands_fft_multicomponent(
    X: np.ndarray,
    fs: float,
    bands: Sequence[ToneBand],
    cfg: EqualizeConfig,
    apply_to: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Fast, deterministic frequency-domain equalization for *narrow* tone bands.

    Rationale (SOAR merged SCM clock-family):
      - Clock tones are coherent narrowband lines (fundamental + harmonics).
      - For these, a per-segment FFT edit is sufficient and substantially faster than STFT.
      - We enforce: |F(f)| inside the edited band matches a smooth baseline inferred from shoulders,
        with no residual peaks and no broad notches.

    Method:
      1) Split into contiguous finite segments (no NaNs) on apply_to channels.
      2) For each segment, compute rFFT across time for all components jointly.
      3) For each band, refine center frequency in the 1D mean spectrum (optional),
         build shoulder baseline in log-domain, and compute a scalar gain g(f) applied to all
         components: F_new(f,j) = g(f) * F(f,j).
      4) Inverse rFFT back to time domain and write into output.

    Notes:
      - This preserves cross-phase across components within the edited bins (scalar gain),
        while removing the line amplitude.
      - Dilation logic reuses the STFT equalizer heuristic in a 1D setting to capture minor
        drift/sidebands without broad suppression.
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n, m)")
    n_all, m_all = X.shape
    Y = X.copy()

    if not cfg.enabled:
        return Y, {"enabled": False, "segments": 0, "edits": []}

    if apply_to is None:
        apply_to = list(range(m_all))
    apply_to = list(apply_to)

    finite = np.all(np.isfinite(Y[:, apply_to]), axis=1)
    # For FFT edits we can operate on shorter segments than cfg.nperseg; do not enforce STFT window length.
    min_len = int(max(128, round(float(cfg.min_seg_s) * float(fs))))
    segs = _finite_segments(finite, min_len=min_len)
    if not segs:
        return Y, {"enabled": True, "segments": 0, "edits": []}

    edits = []
    fs = float(fs)
    eps = 1e-30
    db_fac = 10.0 / np.log(10.0)

    for i0, i1 in segs:
        seg = Y[i0:i1, :].copy()
        Xseg = seg[:, apply_to].astype(float, copy=False)

        # Remove component means to avoid DC leakage into nearby bins
        mu = np.mean(Xseg, axis=0)
        X0 = Xseg - mu[None, :]

        n = X0.shape[0]
        if n < 64:
            continue

        F = np.fft.rfft(X0, axis=0)  # (nf, ma)
        f = np.fft.rfftfreq(n, d=1.0 / fs)  # (nf,)
        nf, ma = F.shape
        df = float(f[1] - f[0]) if f.size > 1 else 0.0

        A = np.sqrt(np.sum(np.abs(F) ** 2, axis=1))  # (nf,)
        logA = np.log(A + eps)

        for band in bands:
            f0_nom = float(band.f0_hz)
            if f0_nom <= 0.0 or f0_nom >= 0.5 * fs:
                continue

            # --- center refinement in mean spectrum ---
            f0 = f0_nom
            prom_db = 0.0
            msearch = (f >= f0_nom - cfg.center_search_hz) & (f <= f0_nom + cfg.center_search_hz)
            if msearch.any():
                ip = int(np.argmax(logA[msearch]))
                fpk = float(f[msearch][ip])

                bw0 = float(band.bw_hz)
                # shoulders around fpk
                left0 = (f >= fpk - 3*bw0) & (f <= fpk - 2*bw0)
                right0 = (f >= fpk + 2*bw0) & (f <= fpk + 3*bw0)
                bases0 = []
                if left0.any(): bases0.append(np.median(logA[left0]))
                if right0.any(): bases0.append(np.median(logA[right0]))
                if bases0:
                    base0 = float(np.median(bases0))
                    prom_db = db_fac * (float(logA[msearch][ip]) - base0)
                    if prom_db >= cfg.peak_prom_db:
                        f0 = fpk

            bw = float(band.bw_hz)
            if df > 0:
                bw = max(bw, 4.0 * df)

            band_mask = (f >= f0 - bw) & (f <= f0 + bw)
            if not band_mask.any():
                continue

            # optional dilation based on excess vs baseline in the mean spectrum
            if cfg.dilate_enabled and df > 0:
                bw_max = float(cfg.dilate_max_mult) * bw
                neigh = (f >= f0 - bw_max) & (f <= f0 + bw_max)
                if neigh.any():
                    s1, s2 = cfg.shoulder_mult
                    left = (f >= f0 - s2*bw) & (f <= f0 - s1*bw)
                    right = (f >= f0 + s1*bw) & (f <= f0 + s2*bw)
                    if (not left.any()) or (not right.any()):
                        left = (f >= max(0.0, f0 - 6*bw)) & (f < f0 - bw)
                        right = (f > f0 + bw) & (f <= min(0.5*fs, f0 + 6*bw))

                    if left.any() and right.any():
                        base_left = float(np.median(logA[left]))
                        base_right = float(np.median(logA[right]))
                        ff = f[neigh]
                        u = (ff - ff.min()) / (ff.max() - ff.min() + 1e-12)
                        base = base_left * (1 - u) + base_right * u
                        excess_db = db_fac * (logA[neigh] - base)
                        cand = excess_db >= float(cfg.dilate_db)
                        if cand.any():
                            pk = int(np.argmin(np.abs(f - f0)))
                            neigh_idx = np.where(neigh)[0]
                            cand_idx = neigh_idx[cand]

                            mask = np.zeros_like(band_mask)
                            mask[cand_idx] = True
                            mask[band_mask] = True
                            mask &= neigh

                            idx = np.where(mask)[0]
                            if idx.size > 0:
                                d = np.diff(idx)
                                split = np.where(d > 1)[0]
                                starts = np.r_[0, split + 1]
                                ends = np.r_[split + 1, idx.size]

                                chosen = None
                                for a, b in zip(starts, ends):
                                    run = idx[a:b]
                                    if pk >= run[0] and pk <= run[-1]:
                                        chosen = run
                                        break
                                if chosen is None:
                                    scores = [np.max(logA[idx[a:b]]) for a, b in zip(starts, ends)]
                                    jbest = int(np.argmax(scores))
                                    chosen = idx[starts[jbest]:ends[jbest]]

                                band_mask2 = np.zeros_like(band_mask)
                                band_mask2[chosen] = True
                                band_mask = band_mask2

            # shoulders baseline in log domain
            s1, s2 = cfg.shoulder_mult
            left = (f >= f0 - s2*bw) & (f <= f0 - s1*bw)
            right = (f >= f0 + s1*bw) & (f <= f0 + s2*bw)
            if not left.any():
                left = (f >= max(0.0, f0 - 6*bw)) & (f < f0 - bw)
            if not right.any():
                right = (f > f0 + bw) & (f <= min(0.5*fs, f0 + 6*bw))
            if (not left.any()) and (not right.any()):
                continue

            ml = float(np.median(logA[left])) if left.any() else float(np.median(logA))
            mr = float(np.median(logA[right])) if right.any() else float(np.median(logA))

            fb = f[band_mask]
            if fb.size == 1:
                targ_log = 0.5 * (ml + mr)
                Abase = np.array([np.exp(targ_log)], dtype=float)
            else:
                denom = (fb.max() - fb.min() + 1e-12)
                u = (fb - fb.min()) / denom
                targ = ml * (1 - u) + mr * u
                Abase = np.exp(targ)

            Acur = A[band_mask] + eps
            g = Abase / Acur
            if cfg.attenuate_only:
                g = np.minimum(g, 1.0)
            g = np.clip(g, float(cfg.g_floor), float(cfg.g_ceil))

            F[band_mask, :] = F[band_mask, :] * g[:, None]

            # update A/logA for subsequent bands (prevents reusing stale amplitudes)
            A[band_mask] = Acur * g
            logA[band_mask] = np.log(A[band_mask] + eps)

            edits.append({
                "seg": (int(i0), int(i1)),
                "family": str(band.family),
                "f0_nom": float(f0_nom),
                "f0_used": float(f0),
                "bw_eff_hz": float(bw),
                "prom_db": float(prom_db),
                "mode": "fft",
            })

        rec = np.fft.irfft(F, n=n, axis=0)
        rec = rec + mu[None, :]
        seg[:, apply_to] = rec
        Y[i0:i1, :] = seg

    return Y, {"enabled": True, "segments": len(segs), "edits": edits, "mode": "fft"}

# --------------------------
# High-level API
# --------------------------

def clean_solo_merged_mag_tones(
    B: np.ndarray,
    fs: float,
    *,
    bands: Optional[Sequence[ToneBand]] = None,
    lockin: LockInConfig = LockInConfig(),
    equalize: EqualizeConfig = EqualizeConfig(),
    detect: DetectConfig = DetectConfig(),
    max_hz_clock: Optional[float] = None,
    max_hz_srf: float = 45.0,
    clock_apply_to: Optional[Sequence[int]] = None,
    srf_apply_to: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Clean Solar Orbiter merged MAG tones on a multi-component time series.

    B: (n, m) real array (e.g., merged MAG components)
    - NaNs are preserved: operations are applied on finite-only segments.
    - clock_apply_to: indices of channels to apply clock-family lock-in / equalization.
      Default: all components.
    - srf_apply_to: indices of channels to apply SRF-family edits (recommended: ONLY SCM X_srf).
      Default: None -> SRF family is NOT edited (safer than blanket removal).
    - max_hz_clock/max_hz_srf: only used when bands is None. Controls the default harmonic
      coverage for the clock (8 Hz*k) and SRF (~1.33 Hz*k) families.

    Returns:
        B_clean, info
    """
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    n, m = B.shape

    if bands is None:
        bands = make_default_bands(fs, max_hz_clock=max_hz_clock, max_hz_srf=max_hz_srf)

    # optional peak-based detection (supplemental)
    extra_bands = detect_tone_bands_from_psd(B, fs, detect)
    if extra_bands:
        # merge, preferring user/nominal bands if overlapping
        merged = list(bands)
        for eb in extra_bands:
            keep = True
            for b in merged:
                if abs(eb.f0_hz - b.f0_hz) <= max(0.5*max(eb.bw_hz, b.bw_hz), 0.08):
                    keep = False
                    break
            if keep:
                merged.append(eb)
        bands = merged

    # split by family
    clock_freqs = sorted({b.f0_hz for b in bands if b.family == "clock"})
    srf_freqs = sorted({b.f0_hz for b in bands if b.family == "srf"})

    clock_bands = [b for b in bands if b.family == "clock"]
    srf_bands = [b for b in bands if b.family == "srf"]
    other_bands = [b for b in bands if b.family not in ("clock", "srf")]

    # Default policies
    if clock_apply_to is None:
        clock_apply_to = list(range(m))
    if srf_apply_to is None:
        # safest default: do not edit SRF unless user explicitly maps SCM X_srf channel
        srf_bands = []
        srf_freqs = []

    Y = B.copy()
    info: Dict = {"fs": float(fs), "n": int(n), "m": int(m), "families": {},
                 "detected_bands": [{"f0_hz": float(b.f0_hz), "bw_hz": float(b.bw_hz), "family": str(b.family)} for b in extra_bands] if "extra_bands" in locals() else []}

    # Stage 1: lock-in only on clock-family by default
    if lockin.enabled and clock_freqs:
        Y, inf1 = remove_tones_lockin_joint(
            Y, fs=fs, tone_freqs_hz=[clock_freqs[0]], cfg=lockin, apply_to=clock_apply_to
        )
        info["families"]["clock_lockin"] = inf1
    else:
        info["families"]["clock_lockin"] = {"enabled": False}

    # Stage 2: multivariate equalization (clock + SRF + other)
    if equalize.enabled:
        # clock bands apply to clock_apply_to
        if clock_bands:
            Y, inf2c = equalize_tone_bands_fft_multicomponent(
                Y, fs=fs, bands=clock_bands, cfg=equalize, apply_to=clock_apply_to
            )
        else:
            inf2c = {"enabled": False}

        # SRF bands apply to srf_apply_to
        if srf_bands and srf_apply_to is not None:
            Y, inf2s = equalize_tone_bands_stft_multicomponent(
                Y, fs=fs, bands=srf_bands, cfg=equalize, apply_to=srf_apply_to
            )
        else:
            inf2s = {"enabled": False, "note": "SRF disabled unless srf_apply_to is provided."}

        # other bands (generic) apply to all by default
        if other_bands:
            Y, inf2o = equalize_tone_bands_stft_multicomponent(
                Y, fs=fs, bands=other_bands, cfg=equalize, apply_to=list(range(m))
            )
        else:
            inf2o = {"enabled": False}

        info["families"]["equalize_clock"] = inf2c
        info["families"]["equalize_srf"] = inf2s
        info["families"]["equalize_other"] = inf2o
    else:
        info["families"]["equalize_clock"] = {"enabled": False}
        info["families"]["equalize_srf"] = {"enabled": False}
        info["families"]["equalize_other"] = {"enabled": False}

    return Y, info


# --------------------------
# Single-call wrapper (optional plotting)
# --------------------------


def remove_wheel_noise_SOLO(
    B: np.ndarray,
    fs: float,
    *,
    # ---- core cleaning (method is in clean_solo_merged_mag_tones)
    bands: Optional[Sequence[ToneBand]] = None,
    lockin: LockInConfig = LockInConfig(),
    equalize: EqualizeConfig = EqualizeConfig(),
    detect: DetectConfig = DetectConfig(),
    max_hz_clock: Optional[float] = None,
    max_hz_srf: float = 45.0,
    clock_apply_to: Optional[Sequence[int]] = None,
    srf_apply_to: Optional[Sequence[int]] = None,

    # ---- plotting
    plot: bool = False,
    plot_backend: str = "plotly",     # "plotly" | "matplotlib"
    plot_time_s: float = 5.0,         # first seconds of channel-0 trace (0 disables)
    plot_max_hz: float = 60.0,
    psd_nperseg: int = 8192,
    psd_overlap: float = 0.5,
    plot_outfile: Optional[str] = None,  # e.g. "report.html" (plotly) or "report.png" (mpl)

    # ---- outputs
    return_debug: bool = False,

    # ---- compatibility (ignore unknown keys safely)
    **_ignored,
):
    """
    One-call entry point: clean Solar Orbiter merged MAG tones and (optionally) plot before/after.

    This is a thin wrapper around `clean_solo_merged_mag_tones` that keeps the cleaning method intact.
    The plotting is meant as a quick diagnostic (PSD + optional short time-series preview).

    Inputs
    ------
    B : array-like
        Time series, shape (n,) or (n, m). Real-valued.
    fs : float
        Sampling rate [Hz].

    Returns
    -------
    If return_debug is False:
        B_clean
    If return_debug is True:
        (B_clean, info)  where `info` also contains an optional `plot` entry.
    """
    B = np.asarray(B, float)
    if B.ndim == 1:
        B_in_shape = "1d"
        B2 = B[:, None]
    elif B.ndim == 2:
        B_in_shape = "2d"
        B2 = B
    else:
        raise ValueError("B must be 1D or 2D (n,) or (n, m).")

    n = B2.shape[0]
    if n < 64 or not np.isfinite(fs) or fs <= 0:
        info = {"fs": float(fs), "n": int(n), "note": "Input too short or fs invalid; returned input unchanged."}
        Bout = B2.copy()
        if B_in_shape == "1d":
            Bout = Bout[:, 0]
        return (Bout, info) if return_debug else Bout

    # --------------------
    # Run cleaning (method unchanged)
    # --------------------
    B_clean, info = clean_solo_merged_mag_tones(
        B2,
        fs=float(fs),
        bands=bands,
        lockin=lockin,
        equalize=equalize,
        detect=detect,
        max_hz_clock=max_hz_clock,
        max_hz_srf=max_hz_srf,
        clock_apply_to=clock_apply_to,
        srf_apply_to=srf_apply_to,
    )

    # --------------------
    # Optional plotting
    # --------------------
    plot_info = None
    if bool(plot):
        # PSD (trace + components) on finite-only data
        Bf = B2.copy()
        Cf = B_clean.copy()
        ok = np.all(np.isfinite(Bf), axis=1) & np.all(np.isfinite(Cf), axis=1)
        Bf = Bf[ok]
        Cf = Cf[ok]

        def _welch(y):
            y = np.asarray(y, float)
            nn = y.size
            if nn < 16:
                return np.array([0.0, 1.0]), np.array([np.nan, np.nan])
            nper = int(min(max(256, int(psd_nperseg)), nn))
            nov = int(min(max(0, int(nper * float(psd_overlap))), nper - 1))
            fw, Pw = signal.welch(y, fs=float(fs), window="hann", nperseg=nper, noverlap=nov, detrend="constant")
            return fw, Pw

        # Trace PSD = sum over component PSDs
        f0 = None
        P0_tr = None
        P1_tr = None
        P0_comp = []
        P1_comp = []
        m = B2.shape[1]
        for j in range(m):
            fj, Pbj = _welch(Bf[:, j])
            fk, Pcj = _welch(Cf[:, j])
            if f0 is None:
                f0 = fj
                P0_tr = np.zeros_like(Pbj)
                P1_tr = np.zeros_like(Pcj)
            # (Welch grids match for same args)
            P0_tr = P0_tr + Pbj
            P1_tr = P1_tr + Pcj
            P0_comp.append(Pbj)
            P1_comp.append(Pcj)

        # Band list for shading (best-effort reconstruction)
        if bands is None:
            bands_used = make_default_bands(float(fs), max_hz_clock=max_hz_clock, max_hz_srf=max_hz_srf)
        else:
            bands_used = list(bands)

        # Add detected bands if present (non-overlapping merge; same as clean_solo_merged_mag_tones)
        extra = []
        for d in info.get("detected_bands", []) or []:
            try:
                extra.append(ToneBand(f0_hz=float(d["f0_hz"]), bw_hz=float(d["bw_hz"]), family=str(d.get("family", "extra"))))
            except Exception:
                pass
        if extra:
            merged = list(bands_used)
            for eb in extra:
                keep = True
                for b in merged:
                    if abs(eb.f0_hz - b.f0_hz) <= max(0.5 * max(eb.bw_hz, b.bw_hz), 0.08):
                        keep = False
                        break
                if keep:
                    merged.append(eb)
            bands_used = merged

        # Match cleaning policy: SRF bands are edited only if srf_apply_to is provided.
        if srf_apply_to is None:
            bands_used = [b for b in bands_used if getattr(b, "family", None) != "srf"]

        plot_backend_l = str(plot_backend).lower().strip()

        if plot_backend_l == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=False,
                vertical_spacing=0.12,
                row_heights=[0.62, 0.38],
                subplot_titles=("PSD (trace + components): before vs after", f"Channel-0 preview (first {float(plot_time_s):g} s)" if plot_time_s and plot_time_s > 0 else "Channel-0 preview (disabled)"),
            )

            # PSD
            m = B2.shape[1]
            # Trace
            fig.add_trace(go.Scatter(x=f0, y=P0_tr, mode="lines", name="trace (before)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=f0, y=P1_tr, mode="lines", name="trace (after)"), row=1, col=1)

            # Components (thin)
            for j in range(m):
                fig.add_trace(go.Scatter(x=f0, y=P0_comp[j], mode="lines", name=f"c{j} (before)", opacity=0.35), row=1, col=1)
                fig.add_trace(go.Scatter(x=f0, y=P1_comp[j], mode="lines", name=f"c{j} (after)", opacity=0.35), row=1, col=1)

            # Shaded tone bands
            for b in bands_used:
                fL = float(b.f0_hz) - 0.5 * float(b.bw_hz)
                fR = float(b.f0_hz) + 0.5 * float(b.bw_hz)
                if fR <= 0 or fL >= float(plot_max_hz):
                    continue
                fig.add_vrect(x0=max(0.0, fL), x1=min(float(plot_max_hz), fR), line_width=0, opacity=0.12, row=1, col=1)

            fig.update_xaxes(type="log", title_text="f [Hz]", range=[np.log10(max(1e-3, 0.2)), np.log10(float(plot_max_hz))], row=1, col=1)
            fig.update_yaxes(type="log", title_text="PSD", row=1, col=1)

            # Time series preview
            if plot_time_s and float(plot_time_s) > 0:
                nshow = int(min(n, max(1, round(float(plot_time_s) * float(fs)))))
                t = np.arange(nshow) / float(fs)
                fig.add_trace(go.Scatter(x=t, y=B2[:nshow, 0], mode="lines", name="c0 (before)"), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=B_clean[:nshow, 0], mode="lines", name="c0 (after)"), row=2, col=1)
                fig.update_xaxes(title_text="t [s]", row=2, col=1)
                fig.update_yaxes(title_text="B", row=2, col=1)

            fig.update_layout(height=780, legend=dict(orientation="h"))

            # Save / show
            if plot_outfile:
                if str(plot_outfile).lower().endswith(".html"):
                    fig.write_html(str(plot_outfile), include_plotlyjs="cdn")
                else:
                    # requires kaleido for static images; keep as best-effort
                    fig.write_image(str(plot_outfile))
            else:
                fig.show()

            plot_info = {"backend": "plotly", "outfile": plot_outfile, "max_hz": float(plot_max_hz)}

        elif plot_backend_l == "matplotlib":
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.loglog(f0, P0_tr, label="trace (before)")
            ax1.loglog(f0, P1_tr, label="trace (after)")
            for j in range(B2.shape[1]):
                ax1.loglog(f0, P0_comp[j], alpha=0.35, label=f"c{j} (before)")
                ax1.loglog(f0, P1_comp[j], alpha=0.35, label=f"c{j} (after)")
            for b in bands_used:
                fL = float(b.f0_hz) - 0.5 * float(b.bw_hz)
                fR = float(b.f0_hz) + 0.5 * float(b.bw_hz)
                if fR <= 0 or fL >= float(plot_max_hz):
                    continue
                ax1.axvspan(max(0.0, fL), min(float(plot_max_hz), fR), alpha=0.12)
            ax1.set_xlim(0.2, float(plot_max_hz))
            ax1.set_xlabel("f [Hz]")
            ax1.set_ylabel("PSD")
            ax1.legend(ncol=2, fontsize=8)

            ax2 = fig.add_subplot(2, 1, 2)
            if plot_time_s and float(plot_time_s) > 0:
                nshow = int(min(n, max(1, round(float(plot_time_s) * float(fs)))))
                t = np.arange(nshow) / float(fs)
                ax2.plot(t, B2[:nshow, 0], label="c0 (before)")
                ax2.plot(t, B_clean[:nshow, 0], label="c0 (after)")
                ax2.set_xlabel("t [s]")
                ax2.set_ylabel("B")
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, "time preview disabled", ha="center", va="center")
                ax2.set_axis_off()

            fig.tight_layout()

            if plot_outfile:
                fig.savefig(str(plot_outfile), dpi=150, bbox_inches="tight")
            else:
                plt.show()

            plot_info = {"backend": "matplotlib", "outfile": plot_outfile, "max_hz": float(plot_max_hz)}
        else:
            raise ValueError("plot_backend must be 'plotly' or 'matplotlib'.")

        if plot_info is not None:
            info = dict(info)  # shallow copy for safety
            info["plot"] = plot_info

    # Restore original shape
    if B_in_shape == "1d":
        B_clean_out = B_clean[:, 0]
    else:
        B_clean_out = B_clean

    return (B_clean_out, info) if return_debug else B_clean_out