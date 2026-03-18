from __future__ import annotations

r"""Core implementation of the self-consistent wavelet-frame anisotropy pipeline.

The central design choice is that the background field, the fluctuation, the local
basis, and the conditioned statistics are all defined from the same centered
multiscale decomposition. The present revision keeps the outward notebook-facing
entry points, but tightens the core estimator in six ways:

1. the FFT-based background/fluctuation estimate is treated explicitly as a
   symmetric, zero-phase, centered estimator in the interior;
2. the validity mask is an explicit cone-of-influence mask, with raw data gaps
   treated as additional local boundaries at each scale;
3. row-wise fluctuation exports are normalized to a strict samplewise contract so
   scalar level summaries cannot crash the table writer;
4. the default coefficient export always includes magnetic fluctuations in the
   user-selected units, while explicit access to both nT and velocity-unit forms
   is preserved;
5. the raw conditioned wavelet moments are kept separate from the increment-
   equivalent structure-function surrogate obtained after removing the L2-wavelet
   \sqrt{s} normalization;
6. the reduced spectral estimate is normalized by the exact filter-energy
   integral \int_0^\infty |H_s(f)|^2\,df rather than by a half-power bandwidth.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from functools import lru_cache
import math
import os
import re
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy import constants
from scipy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import maximum_filter1d


# --------------------------------------------------------------------------------------
# External project imports
# --------------------------------------------------------------------------------------
from functions import general_functions as func
from functions import TurbPy as turb


# --------------------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------------------
mu0 = constants.mu_0
m_p = constants.m_p
_EPS = 1.0e-30
_FFT_LEVEL_BLOCK = 8
_PACKAGE_NAME = Path(__file__).resolve().parent.name
_METHOD_NAME = 'self_consistent_wavelet_frame'
_METHOD_TOKEN = 'scwf'
_DEFAULT_OUTPUT_SUBDIR = 'anisotropy_scwf_v2'
_DEFAULT_COEF_EXPORT_KEYS = frozenset({'dB', 'l_mag', 'l_lambda', 'l_xi', 'l_ell', 'thetas', 'phis', 'polarity', 'local_polarity', 'polarity_used'})


@dataclass(frozen=True)
class WaveletFrameSpec:
    voices_per_octave: int
    order: int
    pad_mode: str
    support_sigma: float
    energy_fraction: float


@dataclass
class LevelData:
    level: int
    tau_equiv_samples: float
    tau_equiv_samples_int: int
    scale_samples: float
    scale_seconds: float
    band_low_samples: float
    band_high_samples: float
    boundary_width_samples: int
    period_s: float
    frequency_hz: float
    bandwidth_hz: float
    response_energy_integral: float
    coi_radius_samples: int
    is_effective: bool
    needed_index: pd.Index
    valid_mask: np.ndarray
    coi_mask: np.ndarray
    keep_turb_amp: Dict[str, Any]
    B_l: np.ndarray
    V_l: np.ndarray
    N_l: np.ndarray
    dB: np.ndarray
    dB_nT: np.ndarray
    dV: np.ndarray
    dVa: np.ndarray
    dN: np.ndarray
    dB_perp: np.ndarray
    dB_parallel: np.ndarray
    dB_perp_nT: np.ndarray
    dB_parallel_nT: np.ndarray
    dVa_perp: np.ndarray
    dVa_parallel: np.ndarray
    dZp: np.ndarray
    dZm: np.ndarray
    leader_B: np.ndarray
    l_mag: np.ndarray
    l_ell: np.ndarray
    l_xi: np.ndarray
    l_lambda: np.ndarray
    VBangle: np.ndarray
    Phiangle: np.ndarray
    polarity: np.ndarray
    local_polarity: np.ndarray
    polarity_used: np.ndarray
    kinet_normal: np.ndarray
    align_angles_vb: Dict[str, Any]
    align_angles_zpm: Dict[str, Any]


def _to_frame(x: Union[pd.Series, pd.DataFrame], name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name if x.name is None else x.name)
    if isinstance(x, pd.DataFrame):
        return x.copy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        return pd.DataFrame({name: arr})
    raise TypeError('Expected Series/DataFrame or 1D array-like.')


def _interpolate_frame_finite(df: pd.DataFrame) -> pd.DataFrame:
    """Return a finite-valued copy of *df* suitable for FFT-based transforms.

    The wavelet backend uses FFTs, so even a single NaN would contaminate an entire
    transformed component. The strategy here is therefore:

    - keep track of the original finite-sample mask separately;
    - interpolate/forward-fill/back-fill only for the purpose of evaluating the
      transform;
    - later exclude coefficients whose support overlaps missing raw samples.
    """
    out = df.copy()
    if out.empty:
        return out
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.interpolate(method='time', limit_direction='both')
    else:
        out = out.interpolate(method='linear', limit_direction='both')
    out = out.ffill().bfill()
    return out


def _build_scale_valid_masks(base_valid: np.ndarray, coi_half_width_samples: np.ndarray) -> np.ndarray:
    """Construct per-scale cone-of-influence masks.

    The estimator is centered in time, so the relevant support at each sample is
    symmetric about that sample. A coefficient is therefore retained only if its
    full local support lies inside the interval and does not overlap any raw gap.
    Internal data gaps are treated as additional boundaries with the same support
    radius as the interval edges.
    """
    base_valid = np.asarray(base_valid, dtype=bool)
    invalid = (~base_valid).astype(np.uint8)
    n_samples = base_valid.size
    masks = []
    for radius in np.asarray(coi_half_width_samples, dtype=int):
        boundary_mask = _valid_mask_from_boundary(n_samples, int(radius))
        if invalid.any():
            contaminated = maximum_filter1d(
                invalid,
                size=(2 * int(max(1, radius)) + 1),
                mode='constant',
                cval=0,
            ).astype(bool)
            masks.append(boundary_mask & (~contaminated))
        else:
            masks.append(boundary_mask)
    return np.vstack(masks)


def _get_component_keys(df: pd.DataFrame, candidate_sets: Sequence[Sequence[str]]) -> List[str]:
    cols = set(df.columns)
    for keys in candidate_sets:
        if all(k in cols for k in keys):
            return list(keys)
    raise KeyError('Could not infer vector component keys.')


def _infer_frame_from_data(B: pd.DataFrame, sc: Optional[str] = None, frame: Optional[str] = None) -> str:
    if frame is not None:
        return str(frame)
    cols = set(B.columns)
    if {'Br', 'Bt', 'Bn'}.issubset(cols):
        return 'RTN'
    if {'Bx', 'By', 'Bz'}.issubset(cols):
        return 'GSE'
    return 'UNKNOWN'


def _estimate_vec_magnitude_stacked(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(np.asarray(arr, dtype=float) ** 2, axis=-1))


def _fast_unit_vec_stacked(arr: np.ndarray) -> np.ndarray:
    mag = _estimate_vec_magnitude_stacked(arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = arr / mag[..., None]
    out[~np.isfinite(out)] = np.nan
    return out


def _perp_vector_stacked(a: np.ndarray, b: np.ndarray, return_paral_comp: bool = False):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    bmag2 = np.nansum(b * b, axis=-1)
    basis_valid = np.isfinite(bmag2) & (bmag2 > 1.0e-24) & np.all(np.isfinite(b), axis=-1)
    par = np.full_like(a, np.nan, dtype=float)
    if np.any(basis_valid):
        with np.errstate(divide='ignore', invalid='ignore'):
            par_valid = (np.nansum(a[basis_valid] * b[basis_valid], axis=-1) / bmag2[basis_valid])[:, None] * b[basis_valid]
        par[basis_valid] = par_valid
    perp = np.full_like(a, np.nan, dtype=float)
    perp[basis_valid] = a[basis_valid] - par[basis_valid]
    if return_paral_comp:
        return perp, par
    return perp


def _angle_between_vectors_stacked(a: np.ndarray, b: np.ndarray, restrict_2_90: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    amag = _estimate_vec_magnitude_stacked(a)
    bmag = _estimate_vec_magnitude_stacked(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.nansum(a * b, axis=-1) / (amag * bmag)
    c = np.clip(c, -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    if restrict_2_90:
        ang = np.where(ang > 90.0, 180.0 - ang, ang)
    return ang


def _mag_of_ell_projections_and_angles_stacked(
    l_vector: np.ndarray,
    B_l_vector: np.ndarray,
    db_perp_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e_l = _fast_unit_vec_stacked(B_l_vector)
    e_xi = _fast_unit_vec_stacked(db_perp_vector)
    e_lambda = _fast_unit_vec_stacked(np.cross(e_l, e_xi, axis=-1))
    l_ell = np.abs(np.nansum(l_vector * e_l, axis=-1))
    l_xi = np.abs(np.nansum(l_vector * e_xi, axis=-1))
    l_lambda = np.abs(np.nansum(l_vector * e_lambda, axis=-1))
    l_perp = _perp_vector_stacked(l_vector, e_l)
    theta = _angle_between_vectors_stacked(l_vector, e_l, restrict_2_90=True)
    phi = _angle_between_vectors_stacked(l_perp, e_xi, restrict_2_90=True)

    db_perp_mag = _estimate_vec_magnitude_stacked(db_perp_vector)
    orient_valid = db_perp_mag > 1.0e-12
    l_xi = np.where(orient_valid, l_xi, np.nan)
    l_lambda = np.where(orient_valid, l_lambda, np.nan)
    phi = np.where(orient_valid, phi, np.nan)
    return l_ell, l_xi, l_lambda, theta, phi


def _nanmean_no_warn(arr: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    count = np.sum(np.isfinite(x), axis=axis)
    total = np.nansum(x, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = total / count
    out = np.asarray(out, dtype=float)
    out[count == 0] = np.nan
    return out


def _nanmedian_no_warn(arr: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if axis != 1:
        return np.nanmedian(x, axis=axis)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        finite = row[np.isfinite(row)]
        if finite.size:
            out[i] = float(np.median(finite))
    return out


# --------------------------------------------------------------------------------------
# Continuous wavelet-frame backend (Gaussian scaling + even DoG wavelet)
# --------------------------------------------------------------------------------------

def parse_wavelet_frame_spec(wname: str) -> WaveletFrameSpec:
    s = str(wname).strip().lower()
    digits = re.findall(r'(\d+)', s)
    voices = int(digits[-1]) if digits else 8
    voices = max(2, min(32, voices))
    order = 4
    match = re.search(r'(?:m|ord|order)(\d+)', s)
    if match is not None:
        order = int(match.group(1))
    elif 'mexh' in s or 'dog2' in s:
        order = 2
    elif 'dog6' in s:
        order = 6
    elif 'dog8' in s:
        order = 8
    if order < 2:
        order = 2
    if order % 2 != 0:
        order += 1
    pad_mode = 'reflect' if 'mirror' not in s else 'symmetric'
    support_sigma = 4.0
    if 'wide' in s or 'broad' in s:
        support_sigma = 5.0
    elif 'sharp' in s:
        support_sigma = 3.5
    return WaveletFrameSpec(
        voices_per_octave=voices,
        order=order,
        pad_mode=pad_mode,
        support_sigma=support_sigma,
        energy_fraction=0.999,
    )


def recommended_nlevels(n_samples: int, spec: WaveletFrameSpec, level: Optional[int], level_mode: str = 'recommended') -> int:
    if level is not None:
        return max(1, int(level))
    mode = str(level_mode).strip().lower()
    if mode in ('recommended', 'safe', 'conservative'):
        max_tau_samples = max(8.0, float(n_samples) / 8.0)
    elif mode in ('legacy', 'max', 'aggressive', 'all'):
        max_tau_samples = max(8.0, float(n_samples) / 4.0)
    else:
        raise ValueError(f'Unknown level_mode={level_mode!r}.')
    s0 = max(1.0, math.sqrt(float(spec.order)) / (2.0 * math.pi))
    jmax = int(np.floor(spec.voices_per_octave * max(0.0, np.log2(max_tau_samples / s0)))) + 1
    return max(1, jmax)


@lru_cache(maxsize=32)
def _mother_wavelet_sq_norm(order: int) -> float:
    eta = np.logspace(-6, 3, 40000)
    vals = (eta ** order) * np.exp(-0.5 * eta * eta)
    return float(np.trapz(vals * vals, eta))


def _wavelet_response_energy_integral(order: int) -> float:
    r"""Return \int_0^\infty |H_s(f)|^2 df for the implemented L2-normalized filter.

    For H_s(f) \propto sqrt(s) (2 pi s f)^M exp[-(2 pi s f)^2 / 2] / ||psi||_2,
    the change of variables x = 2 pi s f gives a scale-independent integral equal
    to (2 pi)^{-1}. The numerical cache above keeps the definition explicit, but
    the ratio reduces exactly to a constant for every even DoG order used here.
    """
    norm2 = max(_mother_wavelet_sq_norm(order), _EPS)
    return float(norm2 / (2.0 * math.pi * norm2))


def build_wavelet_frame(n_samples: int, dt: float, wname: str = 'mw8', level: Optional[int] = None, level_mode: str = 'recommended') -> Dict[str, Any]:
    spec = parse_wavelet_frame_spec(wname)
    J0 = recommended_nlevels(n_samples, spec, level, level_mode=level_mode)
    levels = np.arange(1, J0 + 1, dtype=int)
    s0_samples = max(1.0, math.sqrt(float(spec.order)) / (2.0 * math.pi))
    scale_samples = s0_samples * (2.0 ** ((levels - 1.0) / float(spec.voices_per_octave)))
    scale_seconds = scale_samples * float(dt)
    f_peak = np.sqrt(float(spec.order)) / (2.0 * np.pi * np.maximum(scale_seconds, _EPS))
    tau_equiv_samples = 1.0 / np.maximum(f_peak * float(dt), _EPS)
    tau_equiv_samples_int = np.maximum(1, np.rint(tau_equiv_samples).astype(int))
    period_s = 1.0 / np.maximum(f_peak, _EPS)

    # Approximate half-power frequencies from the squared response envelope
    # |x^M exp(-x^2/2)|^2 = x^{2M} exp(-x^2), so the target is half the peak power.
    xs = np.logspace(-4, 3, 30000)
    mother_amp = (xs ** spec.order) * np.exp(-0.5 * xs * xs)
    mother_power = mother_amp * mother_amp
    x_peak = xs[int(np.nanargmax(mother_power))]
    target = 0.5 * float(np.nanmax(mother_power))
    below = xs[xs < x_peak]
    above = xs[xs > x_peak]
    m_below = mother_power[xs < x_peak]
    m_above = mother_power[xs > x_peak]
    x_low = float(below[np.argmin(np.abs(m_below - target))]) if below.size else x_peak / math.sqrt(2.0)
    x_high = float(above[np.argmin(np.abs(m_above - target))]) if above.size else x_peak * math.sqrt(2.0)
    f_low = x_low / (2.0 * np.pi * np.maximum(scale_seconds, _EPS))
    f_high = x_high / (2.0 * np.pi * np.maximum(scale_seconds, _EPS))
    band_high_samples = 1.0 / np.maximum(f_low * dt, _EPS)
    band_low_samples = 1.0 / np.maximum(f_high * dt, _EPS)
    boundary_width_samples = np.maximum(1, np.ceil(spec.support_sigma * scale_samples).astype(int))
    coi_half_width_samples = boundary_width_samples.copy()
    # This placeholder is replaced by the exact positive-frequency discrete integral
    # of the implemented FFT response once the actual frequency grid is known.
    response_energy_integral = np.full(scale_seconds.shape, np.nan, dtype=float)
    coeff_unit_scale = np.sqrt(np.maximum(scale_seconds, _EPS))

    return {
        'method': _METHOD_NAME,
        'decomposition': 'gaussian_scaling_plus_even_dog_wavelet',
        'spec': spec,
        'J0': int(J0),
        'levels': levels,
        'scale_samples': scale_samples,
        'scale_seconds': scale_seconds,
        'tau_equiv_samples': tau_equiv_samples,
        'tau_equiv_samples_int': tau_equiv_samples_int,
        'period_s': period_s,
        'frequency_hz': f_peak,
        'band_low_samples': band_low_samples,
        'band_high_samples': band_high_samples,
        'bandwidth_hz': np.maximum(f_high - f_low, _EPS),
        'boundary_width_samples': boundary_width_samples,
        'coi_half_width_samples': coi_half_width_samples,
        'response_energy_integral': response_energy_integral,
        'coeff_unit_scale': coeff_unit_scale,
        'estimator_centering': 'symmetric_zero_phase',
        'lag_calibration': 'tau_equiv_from_peak_frequency',
        'wname_parse_note': 'The wname token controls voices_per_octave via trailing digits; the derivative order defaults to 4 unless an explicit order token is present.',
    }


def _pad_for_wavelets(arr: np.ndarray, pad: int, mode: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if pad <= 0:
        return arr.copy()
    mode_used = 'reflect' if mode == 'reflect' else 'symmetric'
    return np.pad(arr, ((pad, pad), (0, 0)), mode=mode_used)


def _build_frequency_responses(freq_hz: np.ndarray, meta: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    spec = meta['spec']
    order = int(spec.order)
    scale_seconds = np.asarray(meta['scale_seconds'], dtype=float)
    omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)[None, :]
    eta = scale_seconds[:, None] * omega
    env = np.exp(-0.5 * eta * eta)
    lows = env
    mother_norm = math.sqrt(max(_mother_wavelet_sq_norm(order), _EPS))
    highs = (np.sqrt(np.maximum(scale_seconds, _EPS))[:, None] * (eta ** order) * env) / mother_norm
    return lows.astype(float, copy=False), highs.astype(float, copy=False)


def estimate_wavelet_backgrounds_and_fluctuations(
    x: Union[np.ndarray, pd.DataFrame],
    wname: str = 'mw8',
    level: Optional[int] = None,
    level_mode: str = 'recommended',
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_samples, n_features = arr.shape
    meta = build_wavelet_frame(n_samples=n_samples, dt=dt, wname=wname, level=level, level_mode=level_mode)
    pad = int(np.max(meta['boundary_width_samples']))
    x_pad = _pad_for_wavelets(arr, pad, meta['spec'].pad_mode)
    freq = rfftfreq(x_pad.shape[0], d=float(dt))
    low_resp, high_resp = _build_frequency_responses(freq, meta)
    if freq.size > 1:
        df_hz = float(freq[1] - freq[0])
        response_energy_integral = np.sum(np.abs(high_resp) ** 2, axis=1) * df_hz
    else:
        response_energy_integral = np.zeros(int(meta['J0']), dtype=float)
    xhat = rfft(x_pad, axis=0)
    J0 = int(meta['J0'])
    backgrounds = np.empty((J0, n_samples, n_features), dtype=float)
    details = np.empty((J0, n_samples, n_features), dtype=float)

    block = int(min(max(1, _FFT_LEVEL_BLOCK), J0))
    for j0 in range(0, J0, block):
        j1 = min(J0, j0 + block)
        low_spec = low_resp[j0:j1, :, None] * xhat[None, :, :]
        high_spec = high_resp[j0:j1, :, None] * xhat[None, :, :]
        low_time = irfft(low_spec, axis=1)
        high_time = irfft(high_spec, axis=1)
        backgrounds[j0:j1] = low_time[:, pad: pad + n_samples, :]
        details[j0:j1] = high_time[:, pad: pad + n_samples, :]

    meta = dict(meta)
    meta['method'] = _METHOD_NAME
    meta['background_definition'] = 'gaussian_low_pass'
    meta['fluctuation_definition'] = 'even_dog_band_pass_coefficient'
    meta['background_plus_fluctuation_is_exact_reconstruction'] = False
    meta['response_energy_integral'] = np.asarray(response_energy_integral, dtype=float)
    meta['response_energy_integral_definition'] = 'positive_frequency_discrete_integral_of_implemented_fft_response'
    return backgrounds, details, meta


def _valid_mask_from_boundary(n_samples: int, boundary_width: int) -> np.ndarray:
    mask = np.ones(int(n_samples), dtype=bool)
    bw = int(max(0, boundary_width))
    if bw == 0:
        return mask
    if 2 * bw >= n_samples:
        mask[:] = False
        return mask
    mask[:bw] = False
    mask[-bw:] = False
    return mask


def _mask_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if out.ndim == 1:
        out[~mask] = np.nan
    else:
        out[~mask, ...] = np.nan
    return out


def _background_polarity(B_full: pd.DataFrame, B_l_all: np.ndarray, index: pd.Index, sc: Optional[str], frame: str) -> Tuple[np.ndarray, np.ndarray]:
    if frame == 'RTN' and 'Br' in B_full.columns:
        base = func.newindex(B_full['Br'], index).to_numpy(dtype=float)
        polarity = np.sign(base)
        polarity[polarity == 0.0] = 1.0
        polarity[~np.isfinite(polarity)] = np.nan
        local = np.sign(B_l_all[:, :, 0])
        local[local == 0.0] = 1.0
        local[~np.isfinite(local)] = np.nan
        return np.broadcast_to(polarity, local.shape), local
    if frame == 'GSE' and 'Bx' in B_full.columns:
        base = func.newindex(B_full['Bx'], index).to_numpy(dtype=float)
        polarity = np.sign(base)
        polarity[polarity == 0.0] = 1.0
        polarity[~np.isfinite(polarity)] = np.nan
        local = np.sign(B_l_all[:, :, 0])
        local[local == 0.0] = 1.0
        local[~np.isfinite(local)] = np.nan
        return np.broadcast_to(polarity, local.shape), local
    polarity = np.ones_like(B_l_all[:, :, 0], dtype=float)
    local = np.ones_like(B_l_all[:, :, 0], dtype=float)
    return polarity, local


def _alignment_stats_by_level(xvec_levels: np.ndarray, yvec_levels: np.ndarray) -> Dict[str, Any]:
    xmag = _estimate_vec_magnitude_stacked(xvec_levels)
    ymag = _estimate_vec_magnitude_stacked(yvec_levels)
    x2 = xmag * xmag
    y2 = ymag * ymag
    cross_mag = _estimate_vec_magnitude_stacked(np.cross(xvec_levels, yvec_levels, axis=-1))
    dot = np.nansum(xvec_levels * yvec_levels, axis=-1)
    denom = xmag * ymag
    with np.errstate(divide='ignore', invalid='ignore'):
        sins = cross_mag / denom
    sins[~np.isfinite(sins)] = np.nan
    reg = np.degrees(np.arctan2(_nanmean_no_warn(cross_mag, axis=1), _nanmean_no_warn(np.abs(dot), axis=1)))
    polar = np.degrees(np.arcsin(np.clip(_nanmean_no_warn(sins, axis=1), -1.0, 1.0)))
    num = x2 - y2
    den = x2 + y2
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_ts = num / den
    sigma_ts[~np.isfinite(sigma_ts)] = np.nan
    x2_mean = _nanmean_no_warn(x2, axis=1)
    y2_mean = _nanmean_no_warn(y2, axis=1)
    den_scale = x2_mean + y2_mean
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_scale = (x2_mean - y2_mean) / den_scale
    sigma_scale[~np.isfinite(sigma_scale)] = np.nan
    return {
        'sigma_ts': sigma_ts,
        'sigma_scale': sigma_scale,
        'sigma_local_mean': _nanmean_no_warn(sigma_ts, axis=1),
        'sigma_local_median': _nanmedian_no_warn(sigma_ts, axis=1),
        'sigma_energy_x_mean': x2_mean,
        'sigma_energy_y_mean': y2_mean,
        'sins_num': _nanmean_no_warn(cross_mag, axis=1),
        'cos_num': _nanmean_no_warn(np.abs(dot), axis=1),
        'sins_den': _nanmean_no_warn(denom, axis=1),
        'x_mag': xmag,
        'y_mag': ymag,
        'reg': reg,
        'polar': polar,
        'weighted': np.full(xvec_levels.shape[0], np.nan),
        'counts': np.sum(np.isfinite(cross_mag), axis=1).astype(int),
    }


def _compute_wavelet_leaders(coeff_levels: np.ndarray, scale_samples: np.ndarray) -> np.ndarray:
    """Approximate 1D wavelet leaders for the vector amplitude.

    Fast implementation using a nearest-edge maximum filter instead of
    a Python loop over every sample.
    """
    amp = _estimate_vec_magnitude_stacked(coeff_levels)
    J0, n = amp.shape
    out = np.empty_like(amp)
    for j in range(J0):
        merged = amp[j].copy()
        if j > 0:
            merged = np.fmax(merged, amp[j - 1])
        if j + 1 < J0:
            merged = np.fmax(merged, amp[j + 1])
        win = int(max(1, math.ceil(scale_samples[j])))
        out[j] = maximum_filter1d(merged, size=(2 * win + 1), mode='nearest')
    return out


def _build_leader_valid_masks(base_valid: np.ndarray, coi_half_width_samples: np.ndarray, scale_samples: np.ndarray) -> np.ndarray:
    r"""Construct conservative validity masks for leader estimators.

    Leaders merge neighboring scales and apply an additional max filter across a
    local window of width \pm ceil(scale_samples[j]). Their effective support is
    therefore larger than the support of a single coefficient and must be masked
    separately from the coefficient cone of influence.
    """
    coi = np.asarray(coi_half_width_samples, dtype=int)
    scale_samples = np.asarray(scale_samples, dtype=float)
    leader_radii = np.empty_like(coi)
    n_levels = coi.size
    for j in range(n_levels):
        j0 = max(0, j - 1)
        j1 = min(n_levels, j + 2)
        neighbor_radius = int(np.max(coi[j0:j1]))
        leader_window = int(max(1, math.ceil(scale_samples[j])))
        leader_radii[j] = neighbor_radius + leader_window
    return _build_scale_valid_masks(base_valid, leader_radii)


def estimate_local_wavelet_geometry(
    B: pd.DataFrame,
    V: pd.DataFrame,
    V_sc_vel_removed: Optional[pd.DataFrame],
    Np: Union[pd.Series, pd.DataFrame],
    dt: Optional[float],
    wname: str = 'mw8',
    level: Optional[int] = None,
    level_mode: str = 'recommended',
    estimate_alignment_angle: bool = False,
    return_B_in_vel_units: bool = True,
    use_local_polarity: bool = True,
    sc: Optional[str] = None,
    frame: Optional[str] = None,
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
) -> Dict[str, Any]:
    """Estimate scale-dependent local geometry and fluctuation quantities.

    The FFT-based filterbank is evaluated on gap-filled inputs because a single NaN
    would otherwise contaminate an entire transformed component. The resulting
    coefficients are then restricted to an explicit cone of influence built from the
    original finite-support mask. Because the responses are even in frequency and the
    padding is symmetric, the interior estimator is centered and zero-phase.
    """
    if V_sc_vel_removed is None:
        V_sc_vel_removed = V

    B = B.copy()
    interp_method = 'time' if isinstance(B.index, pd.DatetimeIndex) else 'linear'
    V = func.newindex(V.copy(), B.index).interpolate(method=interp_method)
    V_sc_vel_removed = func.newindex(V_sc_vel_removed.copy(), B.index).interpolate(method=interp_method)
    Np_df = _to_frame(func.newindex(_to_frame(Np, 'np'), B.index), 'np').interpolate(method=interp_method)

    dt = float(func.find_cadence(B)) if dt is None else float(dt)
    frame_used = _infer_frame_from_data(B, sc=sc, frame=frame)
    b_keys = _get_component_keys(B, (('Br', 'Bt', 'Bn'), ('Bx', 'By', 'Bz')))
    v_keys = _get_component_keys(V, (('Vr', 'Vt', 'Vn'), ('Vx', 'Vy', 'Vz')))

    B_comp_raw = B.loc[:, b_keys].copy()
    V_raw = V.loc[:, v_keys].copy()
    V_used_raw = V_sc_vel_removed.loc[:, v_keys].copy()
    Np_raw = Np_df.iloc[:, [0]].copy()

    base_valid = (
        np.all(np.isfinite(B_comp_raw.to_numpy(dtype=float)), axis=1)
        & np.all(np.isfinite(V_used_raw.to_numpy(dtype=float)), axis=1)
        & np.isfinite(Np_raw.to_numpy(dtype=float).ravel())
    )

    B_comp = _interpolate_frame_finite(B_comp_raw)
    V_raw_filled = _interpolate_frame_finite(V_raw)
    V_used = _interpolate_frame_finite(V_used_raw)
    Np_df = _interpolate_frame_finite(Np_raw)

    stacked = np.concatenate(
        [
            B_comp.to_numpy(dtype=float),
            V_used.to_numpy(dtype=float),
            V_used.to_numpy(dtype=float),
            Np_df.to_numpy(dtype=float),
        ],
        axis=1,
    )

    backgrounds_all, details_all, meta = estimate_wavelet_backgrounds_and_fluctuations(
        stacked,
        wname=wname,
        level=level,
        level_mode=level_mode,
        dt=dt,
    )
    n_levels = int(meta['J0'])
    n_samples = len(B.index)

    B_l_all = backgrounds_all[:, :, 0:3]
    V_l_all = backgrounds_all[:, :, 3:6]
    dB_nT_all = details_all[:, :, 0:3]
    dV_all = details_all[:, :, 6:9]
    N_l_all = np.clip(backgrounds_all[:, :, 9:10], 1.0e-12, None)
    dN_all = details_all[:, :, 9:10]

    kinet_normal_all = (1.0e-15 / np.sqrt(mu0 * N_l_all[:, :, 0] * m_p)).astype(float)
    dVa_all = dB_nT_all * kinet_normal_all[:, :, None]
    dB_perp_nT_all, dB_parallel_nT_all = _perp_vector_stacked(dB_nT_all, B_l_all, return_paral_comp=True)
    dVa_perp_all, dVa_parallel_all = _perp_vector_stacked(dVa_all, B_l_all, return_paral_comp=True)

    tau_s = np.asarray(meta['tau_equiv_samples'], dtype=float)[:, None, None] * dt
    l_vec_all = V_l_all * tau_s
    l_ell_all, l_xi_all, l_lambda_all, theta_all, phi_all = _mag_of_ell_projections_and_angles_stacked(
        l_vec_all,
        B_l_all,
        dB_perp_nT_all,
    )
    di_arr = (228.0 / np.sqrt(np.squeeze(N_l_all, axis=-1))).astype(float)
    l_mag_all = _estimate_vec_magnitude_stacked(l_vec_all) / di_arr
    l_ell_all = l_ell_all / di_arr
    l_xi_all = l_xi_all / di_arr
    l_lambda_all = l_lambda_all / di_arr

    polarity_all, local_polarity_all = _background_polarity(B_comp, B_l_all, B.index, sc=sc, frame=frame_used)
    sign_used = local_polarity_all if use_local_polarity else polarity_all
    dZp_all = dV_all + sign_used[:, :, None] * dVa_all
    dZm_all = dV_all - sign_used[:, :, None] * dVa_all

    du_perp_all = _perp_vector_stacked(dV_all, B_l_all)
    dzp_perp_all = _perp_vector_stacked(dZp_all, B_l_all)
    dzm_perp_all = _perp_vector_stacked(dZm_all, B_l_all)

    align_vb = _alignment_stats_by_level(du_perp_all, dVa_perp_all)
    align_zpm = _alignment_stats_by_level(dzp_perp_all, dzm_perp_all)

    leader_B_all = _compute_wavelet_leaders(dB_nT_all, np.asarray(meta['scale_samples'], dtype=float))

    valid_masks = _build_scale_valid_masks(base_valid, np.asarray(meta['coi_half_width_samples'], dtype=int))
    leader_valid_masks = _build_leader_valid_masks(
        base_valid,
        np.asarray(meta['coi_half_width_samples'], dtype=int),
        np.asarray(meta['scale_samples'], dtype=float),
    )
    n_valid = np.sum(valid_masks, axis=1).astype(int)
    valid_fraction = n_valid.astype(float) / float(max(1, n_samples))
    effective_level_mask = (n_valid >= int(max(1, min_valid_count))) & (valid_fraction >= float(min_valid_fraction))
    good_levels = np.where(effective_level_mask)[0]
    J_effective = int(good_levels[-1] + 1) if good_levels.size else 0

    level_results: List[LevelData] = []
    for j in range(n_levels):
        mask = valid_masks[j]
        B_l = _mask_array(B_l_all[j], mask)
        V_l = _mask_array(V_l_all[j], mask)
        N_l = _mask_array(N_l_all[j, :, 0], mask)
        dB_nT = _mask_array(dB_nT_all[j], mask)
        dV = _mask_array(dV_all[j], mask)
        dVa = _mask_array(dVa_all[j], mask)
        dN = _mask_array(dN_all[j], mask)
        dB_perp_nT = _mask_array(dB_perp_nT_all[j], mask)
        dB_parallel_nT = _mask_array(dB_parallel_nT_all[j], mask)
        dVa_perp = _mask_array(dVa_perp_all[j], mask)
        dVa_parallel = _mask_array(dVa_parallel_all[j], mask)
        dZp = _mask_array(dZp_all[j], mask)
        dZm = _mask_array(dZm_all[j], mask)
        leader_B = _mask_array(leader_B_all[j], leader_valid_masks[j])
        l_mag = _mask_array(l_mag_all[j], mask)
        l_ell = _mask_array(l_ell_all[j], mask)
        l_xi = _mask_array(l_xi_all[j], mask)
        l_lambda = _mask_array(l_lambda_all[j], mask)
        theta = _mask_array(theta_all[j], mask)
        phi = _mask_array(phi_all[j], mask)
        polarity = _mask_array(polarity_all[j], mask)
        local_polarity = _mask_array(local_polarity_all[j], mask)
        polarity_used = _mask_array(sign_used[j], mask)
        kinet_level = _mask_array(kinet_normal_all[j], mask)

        if return_B_in_vel_units:
            dB_out = dVa
            dB_perp_out = dVa_perp
            dB_parallel_out = dVa_parallel
        else:
            dB_out = dB_nT
            dB_perp_out = dB_perp_nT
            dB_parallel_out = dB_parallel_nT

        keep_turb_amp = {
            'dB_perp_amp_nT': func.estimate_vec_magnitude(dB_perp_nT),
            'dB_parallel_amp_nT': func.estimate_vec_magnitude(dB_parallel_nT),
            'B_l': B_l,
        }

        align_angles_vb: Dict[str, Any] = {}
        align_angles_zpm: Dict[str, Any] = {}
        if align_vb is not None and align_zpm is not None:
            align_angles_vb = {
                'sig_r_ts': _mask_array(align_vb['sigma_ts'][j], mask),
                'sig_r_scale': align_vb['sigma_scale'][j],
                'sig_r_local_mean': align_vb['sigma_local_mean'][j],
                'sig_r_local_median': align_vb['sigma_local_median'][j],
                'sig_r_mean': align_vb['sigma_scale'][j],
                'sig_r_median': align_vb['sigma_local_median'][j],
                'u_perp2_mean': align_vb['sigma_energy_x_mean'][j],
                'va_perp2_mean': align_vb['sigma_energy_y_mean'][j],
                'sins_ub_num': align_vb['sins_num'][j],
                'cos_ub_num': align_vb['cos_num'][j],
                'sins_ub_den': align_vb['sins_den'][j],
                'v_mag': _mask_array(align_vb['x_mag'][j], mask),
                'va_mag': _mask_array(align_vb['y_mag'][j], mask),
                'reg_angle': align_vb['reg'][j],
                'polar_inter_angle': align_vb['polar'][j],
                'weighted_angle': align_vb['weighted'][j],
                'counts': int(align_vb['counts'][j]),
            }
            align_angles_zpm = {
                'sig_c_ts': _mask_array(align_zpm['sigma_ts'][j], mask),
                'sig_c_scale': align_zpm['sigma_scale'][j],
                'sig_c_local_mean': align_zpm['sigma_local_mean'][j],
                'sig_c_local_median': align_zpm['sigma_local_median'][j],
                'sig_c_mean': align_zpm['sigma_scale'][j],
                'sig_c_median': align_zpm['sigma_local_median'][j],
                'zp_perp2_mean': align_zpm['sigma_energy_x_mean'][j],
                'zm_perp2_mean': align_zpm['sigma_energy_y_mean'][j],
                'sins_zp_num': align_zpm['sins_num'][j],
                'cos_zp_num': align_zpm['cos_num'][j],
                'sins_zp_den': align_zpm['sins_den'][j],
                'zp_mag': _mask_array(align_zpm['x_mag'][j], mask),
                'zm_mag': _mask_array(align_zpm['y_mag'][j], mask),
                'reg_angle': align_zpm['reg'][j],
                'polar_inter_angle': align_zpm['polar'][j],
                'weighted_angle': align_zpm['weighted'][j],
                'counts': int(align_zpm['counts'][j]),
            }

        level_results.append(
            LevelData(
                level=int(meta['levels'][j]),
                tau_equiv_samples=float(meta['tau_equiv_samples'][j]),
                tau_equiv_samples_int=int(meta['tau_equiv_samples_int'][j]),
                scale_samples=float(meta['scale_samples'][j]),
                scale_seconds=float(meta['scale_seconds'][j]),
                band_low_samples=float(meta['band_low_samples'][j]),
                band_high_samples=float(meta['band_high_samples'][j]),
                boundary_width_samples=int(meta['boundary_width_samples'][j]),
                period_s=float(meta['period_s'][j]),
                frequency_hz=float(meta['frequency_hz'][j]),
                bandwidth_hz=float(meta['bandwidth_hz'][j]),
                response_energy_integral=float(meta['response_energy_integral'][j]),
                coi_radius_samples=int(meta['coi_half_width_samples'][j]),
                is_effective=bool(effective_level_mask[j]),
                needed_index=B.index,
                valid_mask=mask.copy(),
                coi_mask=mask.copy(),
                keep_turb_amp=keep_turb_amp,
                B_l=B_l,
                V_l=V_l,
                N_l=N_l,
                dB=dB_out,
                dB_nT=dB_nT,
                dV=dV,
                dVa=dVa,
                dN=dN,
                dB_perp=dB_perp_out,
                dB_parallel=dB_parallel_out,
                dB_perp_nT=dB_perp_nT,
                dB_parallel_nT=dB_parallel_nT,
                dVa_perp=dVa_perp,
                dVa_parallel=dVa_parallel,
                dZp=dZp,
                dZm=dZm,
                leader_B=leader_B,
                l_mag=l_mag,
                l_ell=l_ell,
                l_xi=l_xi,
                l_lambda=l_lambda,
                VBangle=theta,
                Phiangle=phi,
                polarity=polarity,
                local_polarity=local_polarity,
                polarity_used=polarity_used,
                kinet_normal=kinet_level,
                align_angles_vb=align_angles_vb,
                align_angles_zpm=align_angles_zpm,
            )
        )

    return {
        'B': B_comp_raw,
        'V': V_used_raw,
        'V_raw': V_raw,
        'B_filled': B_comp,
        'V_filled': V_used,
        'V_raw_filled': V_raw_filled,
        'Np': Np_raw,
        'Np_filled': Np_df,
        'b_keys': b_keys,
        'v_keys': v_keys,
        'di_mean': float(np.nanmedian(228.0 / np.sqrt(np.clip(Np_raw.to_numpy(dtype=float).ravel(), 1.0e-12, None)))),
        'Vsw_mean': float(np.nanmedian(func.estimate_vec_magnitude(V_used_raw.to_numpy(dtype=float)))),
        'velocity_used_for_analysis': 'spacecraft_corrected_if_provided',
        'frame': frame_used,
        'polarity_definition': 'Br_sign' if frame_used == 'RTN' and 'Br' in B_comp_raw.columns else ('Bx_sign' if frame_used == 'GSE' and 'Bx' in B_comp_raw.columns else 'unity_fallback_for_unknown_frame'),
        'density_units_assumed': 'cm^-3',
        'return_B_in_vel_units': bool(return_B_in_vel_units),
        'B_export_units': 'km/s_equivalent' if return_B_in_vel_units else 'nT',
        'meta': meta,
        'levels': level_results,
        'effective_level_mask': effective_level_mask,
        'J_effective': J_effective,
        'base_valid_samples': base_valid,
        'leader_valid_masks': leader_valid_masks,
    }


# --------------------------------------------------------------------------------------
# Requested variables: keep the same outward contract as the previous pipeline
# --------------------------------------------------------------------------------------

def _estimate_vec_pvi(df: pd.DataFrame, keys: Sequence[str], tau_value: int, di: float, Vsw: float, five_points_sfunc: bool = False) -> np.ndarray:
    if hasattr(turb, 'estimate_PVI'):
        try:
            out = turb.estimate_PVI(df.copy(), [1], [tau_value], di, Vsw, hours=1, keys=list(keys), five_points_sfunc=five_points_sfunc, PVI_vec_or_mod='vec', use_taus=True, return_only_PVI=True, n_jobs=-1, input_flucts=True, dbs=df)
            return func.newindex(out, df.index).values.T[0]
        except Exception:
            pass
    amp2 = np.nansum(df.loc[:, list(keys)].to_numpy(dtype=float) ** 2, axis=1)
    win = int(max(5, 2 * int(max(1, tau_value)) + 1))
    denom = pd.Series(amp2, index=df.index).rolling(win, center=True, min_periods=max(5, win // 5)).mean().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(amp2 / denom)


def _estimate_mod_pvi(series_df: pd.DataFrame, tau_value: int, di: float, Vsw: float, five_points_sfunc: bool = False) -> np.ndarray:
    if hasattr(turb, 'estimate_PVI'):
        try:
            out = turb.estimate_PVI(series_df.copy(), [1], [tau_value], di, Vsw, hours=1, keys=list(series_df.columns), five_points_sfunc=five_points_sfunc, PVI_vec_or_mod='mod', use_taus=True, return_only_PVI=True, n_jobs=-1)
            return func.newindex(out, series_df.index).values.T[0]
        except Exception:
            pass
    x = series_df.to_numpy(dtype=float).ravel()
    win = int(max(5, 2 * int(max(1, tau_value)) + 1))
    denom = pd.Series(x ** 2, index=series_df.index).rolling(win, center=True, min_periods=max(5, win // 5)).mean().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(x) / np.sqrt(denom)


def _parallel_energy_fraction_from_flucts(dpar: np.ndarray, dvec: np.ndarray) -> np.ndarray:
    num = np.nansum(np.asarray(dpar, dtype=float) ** 2, axis=-1)
    den = np.nansum(np.asarray(dvec, dtype=float) ** 2, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return num / den


def _normalize_requested(ts_list: Optional[Union[str, Sequence[str]]]) -> set:
    if ts_list is None:
        return set(_DEFAULT_COEF_EXPORT_KEYS)
    if isinstance(ts_list, str):
        requested = {ts_list}
    else:
        requested = set(ts_list)
    return requested | set(_DEFAULT_COEF_EXPORT_KEYS)


def _coerce_samplewise_output(value: Any, n_samples: int) -> np.ndarray:
    """Normalize an output to one samplewise 1-D array."""
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            arr = value.iloc[:, 0].to_numpy()
        else:
            return np.full(n_samples, np.nan)
    elif isinstance(value, pd.Series):
        arr = value.to_numpy()
    else:
        arr = np.asarray(value)

    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full(n_samples, float(arr.item()))
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.full(n_samples, float(arr.item()))
    if arr.ndim != 1:
        return np.full(n_samples, np.nan)
    if arr.shape[0] == n_samples:
        return arr.astype(float, copy=False) if np.issubdtype(arr.dtype, np.number) else arr
    if arr.size == 1:
        item = arr.reshape(-1)[0]
        try:
            item = float(item)
        except Exception:
            pass
        return np.full(n_samples, item)
    return np.full(n_samples, np.nan)


def _finalize_requested_variables(variables: Mapping[str, Any], n_samples: int) -> Dict[str, np.ndarray]:
    return {key: _coerce_samplewise_output(val, n_samples) for key, val in variables.items()}


def _store_component_family(variables: MutableMapping[str, Any], requested: set, arr: np.ndarray, canonical_prefix: str, family_aliases: Sequence[str], output_alias_prefixes: Sequence[str] = (), bare_components: bool = False) -> None:
    comp_map = {'R': 0, 'T': 1, 'N': 2}
    family_requested = any(alias in requested for alias in family_aliases)
    for comp, idx in comp_map.items():
        canonical_key = f'{canonical_prefix}{comp}'
        if family_requested or canonical_key in requested:
            variables[canonical_key] = arr[:, idx]
        if bare_components and comp in requested:
            variables[comp] = arr[:, idx]
        for prefix in output_alias_prefixes:
            alias_key = f'{prefix}{comp}'
            if alias_key in requested:
                variables[alias_key] = arr[:, idx]


def estimate_requested_quantities(level_data: LevelData, raw_context: Mapping[str, Any], ts_list: Optional[Union[str, Sequence[str]]] = None, av_hours: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Build row-wise exported variables for one level.

    Every returned entry is normalized to one 1-D samplewise array with the same
    cadence and length as the level. This prevents the fluctuation-table writer from
    mixing per-sample quantities with levelwise scalars. Raw-field contextual
    diagnostics are evaluated on the aligned raw series and then masked by the
    current level support so they are not contaminated by the FFT gap-filling used
    internally by the wavelet estimator.
    """
    requested = _normalize_requested(ts_list)
    variables: Dict[str, Any] = {}
    n_samples = len(level_data.needed_index)

    if av_hours is None:
        av_hours = 1.0 / 60.0

    B = raw_context['B']
    V = raw_context['V']
    valid_mask = level_data.valid_mask
    b_keys = raw_context['b_keys']
    v_keys = raw_context['v_keys']
    tau_value = int(level_data.tau_equiv_samples_int)
    di = float(raw_context['di_mean'])
    Vsw = float(raw_context['Vsw_mean'])
    idx = level_data.needed_index

    _store_component_family(variables, requested, level_data.dB, 'dB_', family_aliases=('dB', 'W_B'), output_alias_prefixes=('W_B_',), bare_components=True)
    _store_component_family(variables, requested, level_data.dB_nT, 'dB_nT_', family_aliases=('dB_nT', 'W_B_nT'), output_alias_prefixes=('W_B_nT_',))
    _store_component_family(variables, requested, level_data.dVa, 'dVa_', family_aliases=('dVa', 'dB_vel', 'W_B_vel'), output_alias_prefixes=('W_B_vel_',))
    _store_component_family(variables, requested, level_data.dV, 'dV_', family_aliases=('dV', 'W_V'), output_alias_prefixes=('V_', 'W_V_'))
    _store_component_family(variables, requested, level_data.dZp, 'zp_', family_aliases=('dzp', 'zp', 'W_Zp'), output_alias_prefixes=('dzp_', 'W_Zp_'))
    _store_component_family(variables, requested, level_data.dZm, 'zm_', family_aliases=('dzm', 'zm', 'W_Zm'), output_alias_prefixes=('dzm_', 'W_Zm_'))
    _store_component_family(variables, requested, level_data.B_l, 'B_l_', family_aliases=('B_l',))

    direct_keys = {
        'l_mag': level_data.l_mag,
        'l_ell': level_data.l_ell,
        'l_lambda': level_data.l_lambda,
        'l_xi': level_data.l_xi,
        'phis': level_data.Phiangle,
        'thetas': level_data.VBangle,
        'local_polarity': level_data.local_polarity,
        'polarity': level_data.polarity,
        'polarity_used': level_data.polarity_used,
        'kinet_normal': level_data.kinet_normal,
        'leader_B': level_data.leader_B,
        'W_leader_B': level_data.leader_B,
        'coi_mask': level_data.coi_mask.astype(float),
    }
    for key, arr in direct_keys.items():
        if key in requested:
            variables[key] = arr
    if 'sign_Bx' in requested:
        variables['sign_Bx'] = level_data.polarity

    if 'N_p' in requested:
        variables['N_p'] = level_data.N_l
    if 'Vsw' in requested:
        variables['Vsw'] = _mask_array(func.newindex(pd.Series(func.estimate_vec_magnitude(V.loc[:, v_keys].to_numpy(dtype=float)), index=V.index), idx).to_numpy(), valid_mask)
    if 'Bmod' in requested:
        variables['Bmod'] = _mask_array(func.newindex(pd.Series(func.estimate_vec_magnitude(B.loc[:, b_keys].to_numpy(dtype=float)), index=B.index), idx).to_numpy(), valid_mask)
    if 'VBangle_big' in requested:
        big = pd.Series(func.angle_between_vectors(B.loc[:, b_keys].to_numpy(dtype=float), V.loc[:, v_keys].to_numpy(dtype=float)), index=B.index)
        variables['VBangle_big'] = _mask_array(func.newindex(big, idx).to_numpy(), valid_mask)

    if 'dB_mag' in requested or 'W_B_mag' in requested:
        dB_mag = _estimate_vec_magnitude_stacked(level_data.dB)
        if 'dB_mag' in requested:
            variables['dB_mag'] = dB_mag
        if 'W_B_mag' in requested:
            variables['W_B_mag'] = dB_mag
    if 'dB_nT_mag' in requested or 'W_B_nT_mag' in requested:
        dB_nT_mag = _estimate_vec_magnitude_stacked(level_data.dB_nT)
        if 'dB_nT_mag' in requested:
            variables['dB_nT_mag'] = dB_nT_mag
        if 'W_B_nT_mag' in requested:
            variables['W_B_nT_mag'] = dB_nT_mag
    if 'dVa_mag' in requested or 'W_B_vel_mag' in requested:
        dVa_mag = _estimate_vec_magnitude_stacked(level_data.dVa)
        if 'dVa_mag' in requested:
            variables['dVa_mag'] = dVa_mag
        if 'W_B_vel_mag' in requested:
            variables['W_B_vel_mag'] = dVa_mag

    if 'sig_c_ts' in requested and 'sig_c_ts' in level_data.align_angles_zpm:
        variables['sig_c_ts'] = level_data.align_angles_zpm['sig_c_ts']
    if 'sig_r_ts' in requested and 'sig_r_ts' in level_data.align_angles_vb:
        variables['sig_r_ts'] = level_data.align_angles_vb['sig_r_ts']
    if 'sig_c_scale' in requested and 'sig_c_scale' in level_data.align_angles_zpm:
        variables['sig_c_scale'] = level_data.align_angles_zpm['sig_c_scale']
    if 'sig_r_scale' in requested and 'sig_r_scale' in level_data.align_angles_vb:
        variables['sig_r_scale'] = level_data.align_angles_vb['sig_r_scale']
    if 'sig_c' in requested and 'sig_c_scale' in level_data.align_angles_zpm:
        variables['sig_c'] = level_data.align_angles_zpm['sig_c_scale']
    if 'sig_r' in requested and 'sig_r_scale' in level_data.align_angles_vb:
        variables['sig_r'] = level_data.align_angles_vb['sig_r_scale']

    scalar_aliases = {
        'sins_ub_num': ('align_angles_vb', 'sins_ub_num'),
        'cos_ub_num': ('align_angles_vb', 'cos_ub_num'),
        'sins_ub_den': ('align_angles_vb', 'sins_ub_den'),
        'sins_zp_num': ('align_angles_zpm', 'sins_zp_num'),
        'cos_zp_num': ('align_angles_zpm', 'cos_zp_num'),
        'sins_zp_den': ('align_angles_zpm', 'sins_zp_den'),
        'sins_zpm_num': ('align_angles_zpm', 'sins_zp_num'),
        'cos_zpm_num': ('align_angles_zpm', 'cos_zp_num'),
        'sins_zpm_den': ('align_angles_zpm', 'sins_zp_den'),
        'zp_mag': ('align_angles_zpm', 'zp_mag'),
        'zm_mag': ('align_angles_zpm', 'zm_mag'),
        'v_mag': ('align_angles_vb', 'v_mag'),
        'va_mag': ('align_angles_vb', 'va_mag'),
    }
    for out_key, (src_name, src_key) in scalar_aliases.items():
        if out_key in requested:
            src = level_data.align_angles_vb if src_name == 'align_angles_vb' else level_data.align_angles_zpm
            if src_key in src:
                variables[out_key] = src[src_key]

    if 'sins_zp' in requested and 'sins_zp_num' in level_data.align_angles_zpm and 'sins_zp_den' in level_data.align_angles_zpm:
        variables['sins_zp'] = np.divide(level_data.align_angles_zpm['sins_zp_num'], level_data.align_angles_zpm['sins_zp_den'])
    if 'sins_zpm' in requested and 'sins_zp_num' in level_data.align_angles_zpm and 'sins_zp_den' in level_data.align_angles_zpm:
        variables['sins_zpm'] = np.divide(level_data.align_angles_zpm['sins_zp_num'], level_data.align_angles_zpm['sins_zp_den'])
    if 'sins_ub' in requested and 'sins_ub_num' in level_data.align_angles_vb and 'sins_ub_den' in level_data.align_angles_vb:
        variables['sins_ub'] = np.divide(level_data.align_angles_vb['sins_ub_num'], level_data.align_angles_vb['sins_ub_den'])

    if 'db_perp_amp_nT' in requested:
        variables['db_perp_amp_nT'] = level_data.keep_turb_amp['dB_perp_amp_nT']
    if 'db_par_amp_nT' in requested:
        variables['db_par_amp_nT'] = level_data.keep_turb_amp['dB_parallel_amp_nT']

    if 'PVI_vec_zp' in requested:
        dzp_df = pd.DataFrame(level_data.dZp, index=idx, columns=['Zpr', 'Zpt', 'Zpn'])
        variables['PVI_vec_zp'] = _estimate_vec_pvi(dzp_df, ['Zpr', 'Zpt', 'Zpn'], tau_value, di, Vsw)
    if 'PVI_vec_zm' in requested:
        dzm_df = pd.DataFrame(level_data.dZm, index=idx, columns=['Zmr', 'Zmt', 'Zmn'])
        variables['PVI_vec_zm'] = _estimate_vec_pvi(dzm_df, ['Zmr', 'Zmt', 'Zmn'], tau_value, di, Vsw)
    if 'PVI_vec' in requested:
        db_df = pd.DataFrame(level_data.dB_nT, index=idx, columns=b_keys)
        variables['PVI_vec'] = _estimate_vec_pvi(db_df, b_keys, tau_value, di, Vsw)
    if 'PVI_vec_V' in requested:
        dv_df = pd.DataFrame(level_data.dV, index=idx, columns=v_keys)
        variables['PVI_vec_V'] = _estimate_vec_pvi(dv_df, v_keys, tau_value, di, Vsw)
    if 'PVI_Np' in requested:
        np_df = pd.DataFrame(level_data.dN, index=idx, columns=['np'])
        variables['PVI_Np'] = _estimate_mod_pvi(np_df, tau_value, di, Vsw)

    if 'compress_simple' in requested or 'parallel_energy_fraction_B' in requested:
        compress_B = _parallel_energy_fraction_from_flucts(level_data.dB_parallel_nT, level_data.dB_nT)
        if 'compress_simple' in requested:
            variables['compress_simple'] = compress_B
        if 'parallel_energy_fraction_B' in requested:
            variables['parallel_energy_fraction_B'] = compress_B
    if 'compress_simple_V' in requested or 'parallel_energy_fraction_V' in requested:
        compress_V = _parallel_energy_fraction_from_flucts(_perp_vector_stacked(level_data.dV, level_data.B_l, return_paral_comp=True)[1], level_data.dV)
        if 'compress_simple_V' in requested:
            variables['compress_simple_V'] = compress_V
        if 'parallel_energy_fraction_V' in requested:
            variables['parallel_energy_fraction_V'] = compress_V

    complex_calls = {
        'compress_squire': ('compressibility_complex_squire', B.copy(), b_keys),
        'compress_squire_V': ('compressibility_complex_squire', V.copy(), v_keys),
        'compress_chen': ('compressibility_complex_chen', B.copy(), b_keys),
        'compress_chen_V': ('compressibility_complex_chen', V.copy(), v_keys),
        'variance': ('variance_anisotropy_verdini', B.copy(), b_keys),
        'norm_turb_amplitude': ('norm_fluct_amplitude', B.copy(), b_keys),
    }
    for req_key, (fn_name, df_obj, keys) in complex_calls.items():
        if req_key not in requested:
            continue
        if hasattr(turb, fn_name):
            try:
                fn = getattr(turb, fn_name)
                if req_key == 'variance':
                    out = fn(tau_value, df_obj, av_hours=av_hours)
                elif req_key == 'norm_turb_amplitude':
                    out = fn(tau_value, df_obj, av_hours=av_hours, denom_av_hours='4H')
                else:
                    out = fn(tau_value, df_obj, keys=keys, av_hours=av_hours) if 'V' in req_key else fn(tau_value, df_obj, av_hours=av_hours)
                masked_out = _mask_array(_coerce_samplewise_output(func.newindex(out, idx), n_samples), valid_mask)
                variables[req_key] = masked_out
                variables[f'external_raw__{req_key}'] = masked_out
            except Exception:
                variables[req_key] = np.full(n_samples, np.nan)
                variables[f'external_raw__{req_key}'] = np.full(n_samples, np.nan)
        else:
            variables[req_key] = np.full(n_samples, np.nan)
            variables[f'external_raw__{req_key}'] = np.full(n_samples, np.nan)

    variables['B_export_units_flag'] = np.full(n_samples, 1.0 if raw_context.get('return_B_in_vel_units', False) else 0.0)
    variables['B_analysis_units_flag'] = variables['B_export_units_flag']
    return _finalize_requested_variables(variables, n_samples)


# --------------------------------------------------------------------------------------
# Conditional moments, spectra, and public interval analysis
# --------------------------------------------------------------------------------------

def _save_flucs(indices: np.ndarray, final_variables: Mapping[str, np.ndarray], ells: np.ndarray, ell_identifier: str) -> Dict[str, Any]:
    indices = np.asarray(indices, dtype=int)
    ell_arr = np.asarray(ells)
    if indices.size == 0:
        out = {key: np.array([np.nan]) for key in final_variables.keys()}
        out[ell_identifier] = np.array([np.nan])
        return out
    out: Dict[str, Any] = {}
    full_length = ell_arr.shape[0]
    for key, val in final_variables.items():
        arr = _coerce_samplewise_output(val, full_length)
        out[key] = arr[indices]
    out[ell_identifier] = ell_arr[indices]
    return out


def _moments_from_precomputed_powers(power_matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Average precomputed |delta X|^q values over an index selection."""
    if indices.size == 0:
        return np.full(power_matrix.shape[1], np.nan)
    subset = np.asarray(power_matrix[indices], dtype=float)
    count = np.sum(np.isfinite(subset), axis=0)
    total = np.nansum(subset, axis=0)
    out = np.full(subset.shape[1], np.nan, dtype=float)
    valid = count > 0
    out[valid] = total[valid] / count[valid]
    return out


def _increment_equivalent_moments_from_wavelet_moments(
    wavelet_moments: np.ndarray,
    qorder: Sequence[float],
    tau_seconds: float,
) -> np.ndarray:
    r"""Convert raw L2-normalized wavelet moments to increment-equivalent surrogates.

    For the implemented even-DoG frame, the coefficient scales as
    W_X(s, t) \sim s^{1/2} \Delta_s X(t) up to a wavelet-shape constant.  The exact
    shape-dependent constant is not removed here, because the pipeline only needs a
    scale-stable moment with the correct physical dimensions.  The implemented
    conversion therefore removes the universal s^{q/2} factor implied by the L2
    normalization, but should still be interpreted as an increment-equivalent
    surrogate rather than as a strict finite-difference structure function.
    """
    q = np.asarray(qorder, dtype=float)
    tau = float(tau_seconds)
    if not np.isfinite(tau) or tau <= 0.0:
        return np.full_like(np.asarray(wavelet_moments, dtype=float), np.nan)
    return np.asarray(wavelet_moments, dtype=float) / np.power(tau, 0.5 * q)


def estimate_wavelet_interval(
    B: pd.DataFrame,
    V: pd.DataFrame,
    V_sc_vel_removed: Optional[pd.DataFrame],
    Np: Union[pd.Series, pd.DataFrame],
    dt: Optional[float],
    di: Optional[float],
    conditions: Mapping[str, Mapping[str, float]],
    qorder: Optional[Sequence[float]] = None,
    wname: str = 'mw8',
    level: Optional[int] = None,
    level_mode: str = 'recommended',
    estimate_alignment_angle: bool = False,
    return_coefs: bool = False,
    ts_list: Optional[Union[str, Sequence[str]]] = None,
    return_B_in_vel_units: bool = True,
    use_local_polarity: bool = True,
    sc: Optional[str] = None,
    frame: Optional[str] = None,
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
    respect_effective_levels: bool = False,
) -> Dict[str, Any]:
    """Estimate conditional multiscale moments for one interval."""
    q = np.asarray([2.0] if qorder is None else qorder, dtype=float)
    results = estimate_local_wavelet_geometry(
        B=B,
        V=V,
        V_sc_vel_removed=V_sc_vel_removed,
        Np=Np,
        dt=dt,
        wname=wname,
        level=level,
        level_mode=level_mode,
        estimate_alignment_angle=estimate_alignment_angle,
        return_B_in_vel_units=return_B_in_vel_units,
        use_local_polarity=use_local_polarity,
        sc=sc,
        frame=frame,
        min_valid_fraction=min_valid_fraction,
        min_valid_count=min_valid_count,
    )

    if di is None:
        di = float(results['di_mean'])

    n_levels = len(results['levels'])
    nbins = ('ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall')
    sf_shape = (n_levels, len(q))

    def _empty_moment_container() -> Dict[str, np.ndarray]:
        return {key: np.full(sf_shape, np.nan) for key in nbins}

    family_keys = ('B', 'B_nT', 'B_vel', 'V', 'Zp', 'Zm', 'LeaderB')
    wavelet_moments = {key: _empty_moment_container() for key in family_keys}
    wavelet_moments['counts'] = {key: np.zeros(n_levels, dtype=int) for key in nbins}
    wavelet_moments['l_di'] = {key: np.full(n_levels, np.nan) for key in nbins}
    sf = {key: _empty_moment_container() for key in family_keys}
    sf['counts'] = wavelet_moments['counts']
    sf['l_di'] = wavelet_moments['l_di']
    spectra = {key: {bucket: np.full(n_levels, np.nan) for bucket in nbins} for key in family_keys}
    spectra['counts'] = sf['counts']
    spectra['l_di'] = sf['l_di']
    spectra['frequency_hz'] = np.full(n_levels, np.nan)
    spectra['bandwidth_hz'] = np.full(n_levels, np.nan)
    spectra['response_energy_integral'] = np.full(n_levels, np.nan)
    legacy_spectra = {key: {bucket: np.full(n_levels, np.nan) for bucket in nbins} for key in family_keys}
    legacy_spectra['counts'] = sf['counts']
    legacy_spectra['l_di'] = sf['l_di']
    legacy_spectra['frequency_hz'] = np.full(n_levels, np.nan)
    legacy_spectra['bandwidth_hz'] = np.full(n_levels, np.nan)

    fluct_tables: Dict[str, Dict[str, pd.DataFrame]] = {
        'ell_perp': {},
        'Ell_perp': {},
        'ell_par': {},
        'ell_par_rest': {},
        'ell_all': {},
        'by_level': {},
    }
    all_thetas: Dict[str, np.ndarray] = {}
    all_phis: Dict[str, np.ndarray] = {}

    align_summary = {
        'VB': {'reg': [], 'polar': [], 'weighted': [], 'sig_r_scale': [], 'sig_r_local_mean': [], 'sig_r_local_median': [], 'u_perp2_mean': [], 'va_perp2_mean': [], 'counts': []},
        'Zpm': {'reg': [], 'polar': [], 'weighted': [], 'sig_c_scale': [], 'sig_c_local_mean': [], 'sig_c_local_median': [], 'zp_perp2_mean': [], 'zm_perp2_mean': [], 'counts': []},
    }

    cond_perp = conditions['ell_perp']
    cond_disp = conditions['Ell_perp']
    cond_par = conditions['ell_par']
    cond_par_rest = conditions.get('ell_par_rest', cond_par)
    q2_idx = np.where(np.isclose(q, 2.0))[0]

    for j, lvl in enumerate(results['levels']):
        all_thetas[str(j)] = lvl.VBangle
        all_phis[str(j)] = lvl.Phiangle
        spectra['frequency_hz'][j] = lvl.frequency_hz
        spectra['bandwidth_hz'][j] = lvl.bandwidth_hz
        spectra['response_energy_integral'][j] = lvl.response_energy_integral
        legacy_spectra['frequency_hz'][j] = lvl.frequency_hz
        legacy_spectra['bandwidth_hz'][j] = lvl.bandwidth_hz

        if estimate_alignment_angle and lvl.align_angles_vb and lvl.align_angles_zpm:
            align_summary['VB']['reg'].append(lvl.align_angles_vb['reg_angle'])
            align_summary['VB']['polar'].append(lvl.align_angles_vb['polar_inter_angle'])
            align_summary['VB']['weighted'].append(lvl.align_angles_vb['weighted_angle'])
            align_summary['VB']['sig_r_scale'].append(lvl.align_angles_vb['sig_r_scale'])
            align_summary['VB']['sig_r_local_mean'].append(lvl.align_angles_vb['sig_r_local_mean'])
            align_summary['VB']['sig_r_local_median'].append(lvl.align_angles_vb['sig_r_local_median'])
            align_summary['VB']['u_perp2_mean'].append(lvl.align_angles_vb['u_perp2_mean'])
            align_summary['VB']['va_perp2_mean'].append(lvl.align_angles_vb['va_perp2_mean'])
            align_summary['VB']['counts'].append(lvl.align_angles_vb['counts'])
            align_summary['Zpm']['reg'].append(lvl.align_angles_zpm['reg_angle'])
            align_summary['Zpm']['polar'].append(lvl.align_angles_zpm['polar_inter_angle'])
            align_summary['Zpm']['weighted'].append(lvl.align_angles_zpm['weighted_angle'])
            align_summary['Zpm']['sig_c_scale'].append(lvl.align_angles_zpm['sig_c_scale'])
            align_summary['Zpm']['sig_c_local_mean'].append(lvl.align_angles_zpm['sig_c_local_mean'])
            align_summary['Zpm']['sig_c_local_median'].append(lvl.align_angles_zpm['sig_c_local_median'])
            align_summary['Zpm']['zp_perp2_mean'].append(lvl.align_angles_zpm['zp_perp2_mean'])
            align_summary['Zpm']['zm_perp2_mean'].append(lvl.align_angles_zpm['zm_perp2_mean'])
            align_summary['Zpm']['counts'].append(lvl.align_angles_zpm['counts'])

        if respect_effective_levels and not lvl.is_effective:
            continue

        finite_theta = np.isfinite(lvl.VBangle) & np.isfinite(lvl.l_mag) & np.isfinite(lvl.l_ell)
        finite_phi_lambda = finite_theta & np.isfinite(lvl.Phiangle) & np.isfinite(lvl.l_lambda)
        finite_phi_xi = finite_theta & np.isfinite(lvl.Phiangle) & np.isfinite(lvl.l_xi)
        mask_perp = finite_phi_lambda & (lvl.VBangle > cond_perp['theta']) & (lvl.Phiangle > cond_perp['phi'])
        mask_disp = finite_phi_xi & (lvl.VBangle > cond_disp['theta']) & (lvl.Phiangle < cond_disp['phi'])
        mask_par = finite_theta & (lvl.VBangle < cond_par['theta'])
        mask_par_rest = finite_theta & (lvl.VBangle < cond_par_rest['theta'])
        mask_all = np.isfinite(lvl.l_mag)

        idx_perp = np.flatnonzero(mask_perp)
        idx_disp = np.flatnonzero(mask_disp)
        idx_par = np.flatnonzero(mask_par)
        idx_par_rest = np.flatnonzero(mask_par_rest)
        idx_all = np.flatnonzero(mask_all)

        for name, idx_sel in [('ell_perp', idx_perp), ('Ell_perp', idx_disp), ('ell_par', idx_par), ('ell_par_rest', idx_par_rest), ('ell_overall', idx_all)]:
            sf['counts'][name][j] = idx_sel.size

        # Precompute magnitudes and powers once per family at this level.
        mag_B = _estimate_vec_magnitude_stacked(lvl.dB)
        mag_B_nT = _estimate_vec_magnitude_stacked(lvl.dB_nT)
        mag_B_vel = _estimate_vec_magnitude_stacked(lvl.dVa)
        mag_V = _estimate_vec_magnitude_stacked(lvl.dV)
        mag_Zp = _estimate_vec_magnitude_stacked(lvl.dZp)
        mag_Zm = _estimate_vec_magnitude_stacked(lvl.dZm)
        mag_Leader = np.abs(np.asarray(lvl.leader_B, dtype=float))

        power_lookup = {
            'B': np.power(mag_B[:, None], q[None, :]),
            'B_nT': np.power(mag_B_nT[:, None], q[None, :]),
            'B_vel': np.power(mag_B_vel[:, None], q[None, :]),
            'V': np.power(mag_V[:, None], q[None, :]),
            'Zp': np.power(mag_Zp[:, None], q[None, :]),
            'Zm': np.power(mag_Zm[:, None], q[None, :]),
            'LeaderB': np.power(mag_Leader[:, None], q[None, :]),
        }

        mapping = [
            ('ell_perp', idx_perp, lvl.l_lambda),
            ('Ell_perp', idx_disp, lvl.l_xi),
            ('ell_par', idx_par, lvl.l_ell),
            ('ell_par_rest', idx_par_rest, lvl.l_ell),
            ('ell_overall', idx_all, lvl.l_mag),
        ]
        for bucket, idx_sel, ell_arr in mapping:
            if idx_sel.size:
                for family in ('B', 'B_nT', 'B_vel', 'V', 'Zp', 'Zm', 'LeaderB'):
                    wavelet_moments[family][bucket][j] = _moments_from_precomputed_powers(power_lookup[family], idx_sel)
                    sf[family][bucket][j] = _increment_equivalent_moments_from_wavelet_moments(
                        wavelet_moments[family][bucket][j],
                        q,
                        (lvl.tau_equiv_samples * dt),
                    )
                sf['l_di'][bucket][j] = np.nanmean(ell_arr[idx_sel])
                if q2_idx.size:
                    q2 = q2_idx[0]
                    for family in ('B', 'B_nT', 'B_vel', 'V', 'Zp', 'Zm', 'LeaderB'):
                        spectra[family][bucket][j] = wavelet_moments[family][bucket][j, q2] / max(lvl.response_energy_integral, _EPS)
                        legacy_spectra[family][bucket][j] = wavelet_moments[family][bucket][j, q2] / max(lvl.bandwidth_hz, _EPS)

        if return_coefs:
            variables = estimate_requested_quantities(lvl, results, ts_list=ts_list)
            fluct_tables['by_level'][str(j)] = variables
            if idx_perp.size:
                fluct_tables['ell_perp'][str(j)] = pd.DataFrame(_save_flucs(idx_perp, variables, lvl.l_lambda, 'l_lambda'))
            if idx_disp.size:
                fluct_tables['Ell_perp'][str(j)] = pd.DataFrame(_save_flucs(idx_disp, variables, lvl.l_xi, 'l_xi'))
            if idx_par.size:
                fluct_tables['ell_par'][str(j)] = pd.DataFrame(_save_flucs(idx_par, variables, lvl.l_ell, 'l_ell'))
            if idx_par_rest.size:
                fluct_tables['ell_par_rest'][str(j)] = pd.DataFrame(_save_flucs(idx_par_rest, variables, lvl.l_ell, 'l_ell'))
            if idx_all.size:
                fluct_tables['ell_all'][str(j)] = pd.DataFrame(_save_flucs(idx_all, variables, lvl.l_mag, 'l_mag'))

    flucts = None
    if return_coefs:
        flucts = {
            'ell_perp': pd.concat(fluct_tables['ell_perp'], axis=0) if fluct_tables['ell_perp'] else pd.DataFrame(),
            'Ell_perp': pd.concat(fluct_tables['Ell_perp'], axis=0) if fluct_tables['Ell_perp'] else pd.DataFrame(),
            'ell_par': pd.concat(fluct_tables['ell_par'], axis=0) if fluct_tables['ell_par'] else pd.DataFrame(),
            'ell_par_rest': pd.concat(fluct_tables['ell_par_rest'], axis=0) if fluct_tables['ell_par_rest'] else pd.DataFrame(),
            'ell_all': pd.concat(fluct_tables['ell_all'], axis=0) if fluct_tables['ell_all'] else pd.DataFrame(),
            'by_level': fluct_tables['by_level'],
        }

    overall_align_angles = {
        'VB': {k: np.asarray(v) for k, v in align_summary['VB'].items()},
        'Zpm': {k: np.asarray(v) for k, v in align_summary['Zpm'].items()},
    }

    wavelet_moments_out = {
        family: {k: (v.T if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in wavelet_moments[family].items()}
        for family in ('B', 'B_nT', 'B_vel', 'V', 'Zp', 'Zm', 'LeaderB')
    }
    wavelet_moments_out['counts'] = wavelet_moments['counts']
    wavelet_moments_out['l_di'] = wavelet_moments['l_di']
    sfuncs = {
        family: {k: (v.T if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in sf[family].items()}
        for family in ('B', 'B_nT', 'B_vel', 'V', 'Zp', 'Zm', 'LeaderB')
    }
    sfuncs['counts'] = sf['counts']
    sfuncs['l_di'] = sf['l_di']

    l_overall = sf['l_di']['ell_overall']
    return {
        'thetas': all_thetas,
        'phis': all_phis,
        'flucts': flucts,
        'l_di': l_overall,
        'ell_di': l_overall,
        'Sfunctions': sfuncs,
        'Sfuncs': sfuncs,
        'StructureFunctions': sfuncs,
        'IncrementEquivalentWaveletMoments': sfuncs,
        'IncrementEquivalentMoments': sfuncs,
        'ScaleNormalizedWaveletMoments': sfuncs,
        'WaveletMoments': wavelet_moments_out,
        'RawWaveletMoments': wavelet_moments_out,
        'Spectra': spectra,
        'ConditionalBandAveragedPSD': spectra,
        'ConditionalBandPower': spectra,
        'LegacySpectraBandwidth': legacy_spectra,
        'PDFs': None,
        'overall_align_angles': overall_align_angles,
        'meta': results['meta'],
        'B_export_units': results['B_export_units'],
        'B_analysis_units': results['B_export_units'],
        'raw': results,
    }


def estimate_3D_sfuncs_same_format(
    B: pd.DataFrame,
    V: pd.DataFrame,
    V_sc_vel_removed: Optional[pd.DataFrame],
    Np: Union[pd.Series, pd.DataFrame],
    dt: float,
    di: float,
    conditions: Mapping[str, Mapping[str, float]],
    qorder: Optional[Sequence[float]],
    tau_values: Optional[Sequence[float]] = None,
    five_points_sfuncs: bool = False,
    estimate_alignment_angle: bool = False,
    return_mag_align_correl: bool = False,
    return_coefs: bool = False,
    only_general: bool = False,
    theta_thresh_gen: float = 0.0,
    phi_thresh_gen: float = 0.0,
    extra_conditions: bool = False,
    ts_list: Optional[Union[str, Sequence[str]]] = None,
    thetas_phis_step: int = 10,
    return_B_in_vel_units: bool = True,
    turb_amp_analysis: bool = True,
    estimate_dzp_dzm: bool = False,
    also_return_db_nT: bool = False,
    use_local_polarity: bool = True,
    sc: Optional[str] = None,
    wname: str = 'mw8',
    level: Optional[int] = None,
    level_mode: str = 'recommended',
    analysis_mode: str = 'full',
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
    respect_effective_levels: bool = False,
):
    del tau_values, five_points_sfuncs, return_mag_align_correl, extra_conditions, thetas_phis_step, turb_amp_analysis, estimate_dzp_dzm, also_return_db_nT, analysis_mode
    out = estimate_wavelet_interval(
        B=B,
        V=V,
        V_sc_vel_removed=V_sc_vel_removed,
        Np=Np,
        dt=dt,
        di=di,
        conditions=conditions,
        qorder=qorder,
        wname=wname,
        level=level,
        level_mode=level_mode,
        estimate_alignment_angle=estimate_alignment_angle,
        return_coefs=return_coefs,
        ts_list=ts_list,
        return_B_in_vel_units=return_B_in_vel_units,
        use_local_polarity=use_local_polarity,
        sc=sc,
        min_valid_fraction=min_valid_fraction,
        min_valid_count=min_valid_count,
        respect_effective_levels=respect_effective_levels,
    )

    if return_coefs and only_general:
        flucts = out['flucts']
        if flucts is not None and 'ell_all' in flucts and not flucts['ell_all'].empty:
            mask = (flucts['ell_all']['thetas'] > theta_thresh_gen) & (flucts['ell_all']['phis'] > phi_thresh_gen)
            flucts = dict(flucts)
            flucts['ell_all'] = flucts['ell_all'].loc[mask].reset_index(drop=True)
        out['flucts'] = flucts

    last = out['raw']['levels'][-1]
    return last.l_mag, last.l_lambda, last.l_xi, last.l_ell, last.VBangle, last.Phiangle, out['flucts'], out['l_di'], out['Sfunctions'], out['PDFs'], out['overall_align_angles']


estimate_3D_sfuncs = estimate_3D_sfuncs_same_format


# --------------------------------------------------------------------------------------
# Batch driver: preserve the notebook-facing orchestration
# --------------------------------------------------------------------------------------

def _pick_existing_columns(df: pd.DataFrame, candidates: Sequence[Sequence[str]]) -> List[str]:
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return list(cols)
    return list(df.columns[: min(3, len(df.columns))])


def _get_B_dataframe(res: Mapping[str, Any], use_low_resol_data: bool = False) -> pd.DataFrame:
    mag = res['Mag']
    B_src = mag.get('B_resampled_part_res', mag['B_resampled']) if use_low_resol_data else mag['B_resampled']
    return B_src[_pick_existing_columns(B_src, [('Br', 'Bt', 'Bn'), ('Bx', 'By', 'Bz')])]


def _get_V_dataframe(res: Mapping[str, Any]) -> pd.DataFrame:
    V_src = res['Par']['V_resampled']
    return V_src[_pick_existing_columns(V_src, [('Vr', 'Vt', 'Vn'), ('Vx', 'Vy', 'Vz')])]


def _get_Np_dataframe(res: Mapping[str, Any]) -> pd.DataFrame:
    V_src = res['Par']['V_resampled']
    if 'np' in V_src.columns:
        return V_src[['np']]
    if 'Np' in V_src.columns:
        return V_src[['Np']].rename(columns={'Np': 'np'})
    raise KeyError("Could not find density column 'np' or 'Np'.")




def _sanitize_token(value: Optional[str]) -> str:
    if value is None:
        return ''
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
    token = re.sub(r'_+', '_', token).strip('_.-')
    return token


def _compose_output_subdir(output_subdir: Optional[str], method_token: str) -> str:
    chosen = _sanitize_token(output_subdir)
    if chosen:
        return chosen
    return f'anisotropy_{method_token}'


def _prefix_filename(base: str, file_name_root: Optional[str]) -> str:
    root = _sanitize_token(file_name_root)
    if not root:
        return base
    return f'{root}__{base}'

def build_output_names(
    consider_Vsc: bool,
    strict_thresh: int,
    return_flucs: bool,
    only_general: int,
    extra_conditions: bool,
    theta_thresh_gen: float,
    phi_thresh_gen: float,
    thetas_phis_step: int,
    wname: str,
    file_name_root: Optional[str] = None,
    method_token: str = _METHOD_TOKEN,
) -> Tuple[str, Optional[str]]:
    strict_suffix = '5deg_' if strict_thresh == 1 else ('2deg_' if strict_thresh == 2 else '')
    conditions_suffix = 'extra_conditions_' if extra_conditions else ''
    sfuncs_suffix = '' if return_flucs else 'sfuncs_estimated_'
    general_suffix = 'general_SF_' if only_general == 1 else ''
    vsc_suffix = 'Vsc_removed_' if consider_Vsc else ''
    method_suffix = f'{method_token}_{wname}_'
    align_name = None
    if only_general == 1:
        fname = f"{general_suffix}{method_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}theta_{theta_thresh_gen}_phi_{phi_thresh_gen}.pkl"
        align_name = f'alignment_angles_{method_suffix}{vsc_suffix}.pkl'
    elif only_general == 2:
        fname = f"_all_bins_{general_suffix}{method_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}step_{thetas_phis_step}.pkl"
    else:
        fname = f"{general_suffix}{method_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}final.pkl"
        align_name = f'alignment_angles_{method_suffix}{vsc_suffix}.pkl'
    fname = _prefix_filename(fname, file_name_root)
    if align_name is not None:
        align_name = _prefix_filename(align_name, file_name_root)
    return fname, align_name


def run_filterbank_interval_analysis(
    i: int,
    fnames: Sequence[str],
    credentials: Any,
    conditions: Mapping[str, Mapping[str, float]],
    return_flucs: bool,
    consider_Vsc: bool,
    Estimate_5point: bool,
    keep_wave_coeefs: bool,
    strict_thresh: int,
    max_hours: float,
    qorder: Sequence[float],
    estimate_alignment_angle: bool,
    return_mag_align_correl: bool,
    only_general: int,
    phi_thresh_gen: float,
    theta_thresh_gen: float,
    sc: str = 'PSP',
    extra_conditions: bool = False,
    ts_list: Optional[Union[str, Sequence[str]]] = None,
    overwrite_existing_files: bool = False,
    thetas_phis_step: int = 10,
    return_B_in_vel_units: bool = True,
    max_interval_dur: float = 240,
    estimate_dzp_dzm: bool = False,
    use_low_resol_data: bool = False,
    use_local_polarity: bool = False,
    dt_step: float = 0.25,
    wname: str = 'mw8',
    level: Optional[int] = None,
    level_mode: str = 'recommended',
    analysis_mode: str = 'full',
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
    respect_effective_levels: bool = True,
    output_subdir: Optional[str] = None,
    file_name_root: Optional[str] = None,
    method_token: str = _METHOD_TOKEN,
):
    del credentials, Estimate_5point, keep_wave_coeefs, max_hours, estimate_dzp_dzm, dt_step, return_mag_align_correl, analysis_mode
    warnings.filterwarnings('ignore')
    try:
        func.progress_bar(i, len(fnames))
        res = pd.read_pickle(fnames[i])
        gen_name = str(fnames[i]).replace('final.pkl', 'general.pkl')
        gen = pd.read_pickle(gen_name)
        dts = (gen['End_Time'] - gen['Start_Time']).total_seconds() / 3600.0
        if dts >= max_interval_dur:
            return None

        fname, align_name = build_output_names(consider_Vsc, strict_thresh, return_flucs, only_general, extra_conditions, theta_thresh_gen, phi_thresh_gen, thetas_phis_step, wname, file_name_root=file_name_root, method_token=method_token)
        resolved_output_subdir = _compose_output_subdir(output_subdir, method_token)
        outdir = str(Path(gen_name.replace('general.pkl', '')).joinpath(resolved_output_subdir))
        check_file = str(Path(outdir).joinpath(fname))
        if os.path.exists(check_file) and not overwrite_existing_files:
            print('Skipping existing file:', check_file)
            return check_file

        B = _get_B_dataframe(res, use_low_resol_data=use_low_resol_data)
        V = _get_V_dataframe(res)
        Np = _get_Np_dataframe(res)
        B = B[~B.index.duplicated()]
        V = V[~V.index.duplicated()]
        Np = Np[~Np.index.duplicated()]

        try:
            ephem = res.get('Ephem', {}) if isinstance(res, Mapping) else {}
            V = func.newindex(V, B.index)
            if consider_Vsc:
                if isinstance(ephem, pd.DataFrame) and all(c in ephem.columns for c in ['sc_vel_r', 'sc_vel_t', 'sc_vel_n']):
                    V_sc = func.newindex(ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].interpolate(), B.index)
                    V_sc_rem = V - V_sc.values
                elif isinstance(ephem, Mapping) and all(c in ephem for c in ['sc_vel_r', 'sc_vel_t', 'sc_vel_n']):
                    V_sc_df = pd.DataFrame(
                        {c: np.asarray(ephem[c]) for c in ['sc_vel_r', 'sc_vel_t', 'sc_vel_n']},
                        index=B.index,
                    )
                    V_sc = func.newindex(V_sc_df.interpolate(), B.index)
                    V_sc_rem = V - V_sc.values
                else:
                    V_sc_rem = V
            else:
                V_sc_rem = V
        except Exception:
            V_sc_rem = V

        Np = func.newindex(Np, B.index)
        di = float(res['Par']['di_mean'])
        Vsw = float(res['Par']['Vsw_mean'])
        Vsw_norm = float(np.nanmean(np.linalg.norm(np.asarray(V_sc_rem, dtype=float), axis=1)))
        dt = func.find_cadence(B)

        interval_out = estimate_wavelet_interval(
            B=B,
            V=V,
            V_sc_vel_removed=V_sc_rem,
            Np=Np,
            dt=dt,
            di=di,
            conditions=conditions,
            qorder=qorder,
            wname=wname,
            level=level,
            level_mode=level_mode,
            estimate_alignment_angle=estimate_alignment_angle,
            return_coefs=return_flucs,
            ts_list=ts_list,
            return_B_in_vel_units=return_B_in_vel_units,
            use_local_polarity=use_local_polarity,
            sc=sc,
            min_valid_fraction=min_valid_fraction,
            min_valid_count=min_valid_count,
            respect_effective_levels=respect_effective_levels,
        )
        flucts = interval_out['flucts']
        ell_di = interval_out['ell_di']
        Sfunctions = interval_out['Sfunctions']
        PDFs = interval_out['PDFs']
        overall_align_angles = interval_out['overall_align_angles']
        keep_sfuncs_final = {
            'di': di,
            'Vsw': Vsw,
            'Vsw_norm': Vsw_norm,
            'ell_di': ell_di,
            'Sfuncs': Sfunctions,
            'StructureFunctions': interval_out['StructureFunctions'],
            'WaveletMoments': interval_out['WaveletMoments'],
            'Spectra': interval_out['Spectra'],
            'B_analysis_units': interval_out['B_analysis_units'],
            'LegacySpectraBandwidth': interval_out['LegacySpectraBandwidth'],
            'flucts': flucts,
            'PDFs': PDFs,
            'meta': {
                'analysis_chain': [
                    f'{_PACKAGE_NAME}/notebooks/3D_LOGFILTERBANK_sfuncs_cleaned.ipynb',
                    f'{_PACKAGE_NAME}.data_analysis.run_logscale_filterbank_analysis',
                    f'{_PACKAGE_NAME}.three_D_funcs.run_filterbank_interval_analysis',
                    f'{_PACKAGE_NAME}.three_D_funcs.estimate_wavelet_interval',
                    f'{_PACKAGE_NAME}.three_D_funcs.estimate_local_wavelet_geometry',
                    f'{_PACKAGE_NAME}.three_D_funcs.estimate_wavelet_backgrounds_and_fluctuations',
                ],
                'method': _METHOD_NAME,
                'estimator': 'gaussian_scaling_plus_even_dog_wavelet',
                'note': 'Sfuncs/StructureFunctions/ScaleNormalizedWaveletMoments store scale-normalized wavelet-moment surrogates obtained by removing the universal L2-wavelet tau^(q/2) factor at the same tau_equiv used in Taylor mapping. They are not strict finite-difference structure functions. WaveletMoments stores the raw conditional wavelet-coefficient moments. Spectra/ConditionalBandAveragedPSD stores second-order raw wavelet moments of the full vector coefficient magnitude divided by the positive-frequency discrete response-energy integral of the implemented FFT filter; the local basis is used to define conditional angle bins, not to project the fluctuation power itself onto basis components. LegacySpectraBandwidth preserves the older bandwidth-based normalization only for comparison. The Gaussian background and even-DoG fluctuation are paired filters, not an exact additive reconstruction. When spacecraft-corrected velocity is supplied, both the Taylor mapping and the velocity/Elsasser fluctuations use that corrected velocity consistently. Raw contextual diagnostics are evaluated on aligned raw series rather than on the gap-filled transform inputs. If the frame does not identify a radial-like component, polarity correction falls back to unity rather than guessing from the first component. The compatibility B family still follows the user-selected magnetic unit system; B_nT and B_vel are exported explicitly to avoid unit ambiguity.',
                'package_name': _PACKAGE_NAME,
                'method_token': method_token,
                'output_subdir': resolved_output_subdir,
                'file_name_root': file_name_root,
                'wavelet_meta': interval_out['meta'],
            },
        }
        func.savepickle(keep_sfuncs_final, outdir, fname)
        if estimate_alignment_angle and align_name is not None:
            func.savepickle(overall_align_angles, outdir, align_name)
        return str(Path(outdir) / fname)
    except Exception:
        traceback.print_exc()
        return None


# Notebook-facing compatibility alias
run_logscale_filterbank_analysis = run_filterbank_interval_analysis
