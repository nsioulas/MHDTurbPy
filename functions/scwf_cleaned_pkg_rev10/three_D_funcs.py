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
_DEFAULT_OUTPUT_SUBDIR = 'anisotropy_scwf'
_DEFAULT_COEF_EXPORT_KEYS = frozenset({'dB', 'l_mag', 'l_lambda', 'l_xi', 'l_ell', 'thetas', 'phis', 'polarity', 'local_polarity', 'polarity_used', 'B_par', 'B_perp', 'V_par', 'V_perp', 'Zp_par', 'Zp_perp', 'Zp_xi', 'Zp_lambda'})


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


def _local_basis_from_background_and_perp(
    B_l_vector: np.ndarray,
    db_perp_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the local (ell, xi, lambda) basis and its validity mask.

    The basis is defined by the local mean-field direction and the local magnetic
    perpendicular-coefficient direction used by the geometry estimator. When the
    latter is undefined, xi/lambda are undefined as well.
    """
    e_l = _fast_unit_vec_stacked(B_l_vector)
    e_xi = _fast_unit_vec_stacked(db_perp_vector)
    e_lambda = _fast_unit_vec_stacked(np.cross(e_l, e_xi, axis=-1))
    valid = (
        np.all(np.isfinite(e_l), axis=-1)
        & np.all(np.isfinite(e_xi), axis=-1)
        & np.all(np.isfinite(e_lambda), axis=-1)
    )
    e_l = np.asarray(e_l, dtype=float)
    e_xi = np.asarray(e_xi, dtype=float)
    e_lambda = np.asarray(e_lambda, dtype=float)
    e_l[~valid] = np.nan
    e_xi[~valid] = np.nan
    e_lambda[~valid] = np.nan
    return e_l, e_xi, e_lambda, valid


def _abs_scalar_projection(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return |vec . basis| for a unit-vector basis with NaN-safe masking."""
    vec = np.asarray(vec, dtype=float)
    basis = np.asarray(basis, dtype=float)
    out = np.full(vec.shape[:-1], np.nan, dtype=float)
    valid = np.all(np.isfinite(vec), axis=-1) & np.all(np.isfinite(basis), axis=-1)
    if np.any(valid):
        out[valid] = np.abs(np.nansum(vec[valid] * basis[valid], axis=-1))
    return out


def _bucket_scale_stat_summary(values: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'p16': np.nan,
            'p84': np.nan,
            'p05': np.nan,
            'p95': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    return {
        'mean': float(np.nanmean(vals)),
        'median': float(np.nanmedian(vals)),
        'p16': float(np.nanpercentile(vals, 16.0)),
        'p84': float(np.nanpercentile(vals, 84.0)),
        'p05': float(np.nanpercentile(vals, 5.0)),
        'p95': float(np.nanpercentile(vals, 95.0)),
        'min': float(np.nanmin(vals)),
        'max': float(np.nanmax(vals)),
    }


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

    e_l, e_xi, e_lambda, _basis_valid = _local_basis_from_background_and_perp(level_data.B_l, level_data.dB_perp_nT)
    projection_aliases = {
        'B_par': _abs_scalar_projection(level_data.dB, e_l),
        'B_perp': _estimate_vec_magnitude_stacked(level_data.dB_perp),
        'B_xi': _abs_scalar_projection(level_data.dB, e_xi),
        'B_lambda': _abs_scalar_projection(level_data.dB, e_lambda),
        'B_nT_par': _abs_scalar_projection(level_data.dB_nT, e_l),
        'B_nT_perp': _estimate_vec_magnitude_stacked(level_data.dB_perp_nT),
        'B_nT_xi': _abs_scalar_projection(level_data.dB_nT, e_xi),
        'B_nT_lambda': _abs_scalar_projection(level_data.dB_nT, e_lambda),
        'B_vel_par': _abs_scalar_projection(level_data.dVa, e_l),
        'B_vel_perp': _estimate_vec_magnitude_stacked(level_data.dVa_perp),
        'B_vel_xi': _abs_scalar_projection(level_data.dVa, e_xi),
        'B_vel_lambda': _abs_scalar_projection(level_data.dVa, e_lambda),
        'V_par': _abs_scalar_projection(level_data.dV, e_l),
        'V_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(level_data.dV, level_data.B_l)),
        'V_xi': _abs_scalar_projection(level_data.dV, e_xi),
        'V_lambda': _abs_scalar_projection(level_data.dV, e_lambda),
        'Zp_par': _abs_scalar_projection(level_data.dZp, e_l),
        'Zp_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(level_data.dZp, level_data.B_l)),
        'Zp_xi': _abs_scalar_projection(level_data.dZp, e_xi),
        'Zp_lambda': _abs_scalar_projection(level_data.dZp, e_lambda),
        'Zm_par': _abs_scalar_projection(level_data.dZm, e_l),
        'Zm_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(level_data.dZm, level_data.B_l)),
        'Zm_xi': _abs_scalar_projection(level_data.dZm, e_xi),
        'Zm_lambda': _abs_scalar_projection(level_data.dZm, e_lambda),
        'e_ell_r': e_l[:, 0],
        'e_ell_t': e_l[:, 1],
        'e_ell_n': e_l[:, 2],
        'e_xi_r': e_xi[:, 0],
        'e_xi_t': e_xi[:, 1],
        'e_xi_n': e_xi[:, 2],
        'e_lambda_r': e_lambda[:, 0],
        'e_lambda_t': e_lambda[:, 1],
        'e_lambda_n': e_lambda[:, 2],
    }
    for key, arr in projection_aliases.items():
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


def _column_units_hint(columns: Sequence[str], b_export_units: str) -> Dict[str, str]:
    units: Dict[str, str] = {}
    b_units = 'km/s * s^0.5' if str(b_export_units).lower().startswith('vel') else 'nT * s^0.5'
    for key in columns:
        if key in {'level_index', 'bucket_code', 'polarity', 'local_polarity', 'polarity_used', 'coi_mask', 'B_export_units_flag', 'B_analysis_units_flag', 'is_effective_level'}:
            units[key] = 'dimensionless'
        elif key in {'thetas', 'phis', 'VBangle_big'}:
            units[key] = 'deg'
        elif key.endswith('_seconds') or key == 'period_s':
            units[key] = 's'
        elif key.endswith('_hz') or key == 'frequency_hz' or key == 'bandwidth_hz':
            units[key] = 'Hz'
        elif key.endswith('_di') or key.startswith('l_') or key == 'selected_scale_di':
            units[key] = 'd_i'
        elif key.startswith('W_B_nT') or key.startswith('dB_nT') or key.startswith('B_nT_') or key in {'db_perp_amp_nT', 'db_par_amp_nT'}:
            units[key] = 'nT * s^0.5'
        elif key.startswith('W_B_vel') or key.startswith('dVa') or key.startswith('B_vel_'):
            units[key] = 'km/s * s^0.5'
        elif key.startswith('W_B') or key.startswith('dB') or key.startswith('B_'):
            units[key] = b_units
        elif key.startswith('W_V') or key.startswith('dV') or key.startswith('V_') or key.startswith('W_Zp') or key.startswith('W_Zm') or key.startswith('zp_') or key.startswith('zm_') or key.startswith('Zp_') or key.startswith('Zm_'):
            units[key] = 'km/s * s^0.5'
        elif key in {'response_energy_integral'}:
            units[key] = 'Hz'
        elif key in {'N_p'}:
            units[key] = 'cm^-3'
        elif key in {'Vsw', 'Bmod'}:
            units[key] = 'native'
        else:
            units[key] = 'dimensionless'
    return units


def _augment_saved_flucs_for_compact_store(
    saved: MutableMapping[str, np.ndarray],
    level_data: LevelData,
    level_index: int,
    ell_identifier: str,
    b_export_units: str,
) -> Dict[str, np.ndarray]:
    out = {key: np.asarray(val) for key, val in saved.items()}
    n_rows = int(next(iter(out.values())).shape[0]) if out else 0

    def _ensure(key: str, value: Any) -> None:
        if key in out:
            return
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = np.full(n_rows, arr.item(), dtype=float)
        elif arr.shape[0] == n_rows:
            out[key] = arr
        else:
            out[key] = np.full(n_rows, np.nan, dtype=float)

    e_l, e_xi, e_lambda, _basis_valid = _local_basis_from_background_and_perp(level_data.B_l, level_data.dB_perp_nT)
    derived = {
        'l_mag': level_data.l_mag,
        'l_ell': level_data.l_ell,
        'l_xi': level_data.l_xi,
        'l_lambda': level_data.l_lambda,
        'selected_scale_di': out.get(ell_identifier, np.full(n_rows, np.nan, dtype=float)),
        'thetas': level_data.VBangle,
        'phis': level_data.Phiangle,
        'polarity': level_data.polarity,
        'local_polarity': level_data.local_polarity,
        'polarity_used': level_data.polarity_used,
        'coi_mask': level_data.coi_mask.astype(float),
        'W_B_mag': _estimate_vec_magnitude_stacked(level_data.dB),
        'W_B_nT_mag': _estimate_vec_magnitude_stacked(level_data.dB_nT),
        'W_B_vel_mag': _estimate_vec_magnitude_stacked(level_data.dVa),
        'W_V_mag': _estimate_vec_magnitude_stacked(level_data.dV),
        'W_Zp_mag': _estimate_vec_magnitude_stacked(level_data.dZp),
        'W_Zm_mag': _estimate_vec_magnitude_stacked(level_data.dZm),
        'B_par': _abs_scalar_projection(level_data.dB, e_l),
        'B_perp': _estimate_vec_magnitude_stacked(level_data.dB_perp),
        'B_xi': _abs_scalar_projection(level_data.dB, e_xi),
        'B_lambda': _abs_scalar_projection(level_data.dB, e_lambda),
        'B_nT_par': _abs_scalar_projection(level_data.dB_nT, e_l),
        'B_nT_perp': _estimate_vec_magnitude_stacked(level_data.dB_perp_nT),
        'B_nT_xi': _abs_scalar_projection(level_data.dB_nT, e_xi),
        'B_nT_lambda': _abs_scalar_projection(level_data.dB_nT, e_lambda),
        'B_vel_par': _abs_scalar_projection(level_data.dVa, e_l),
        'B_vel_perp': _estimate_vec_magnitude_stacked(level_data.dVa_perp),
        'B_vel_xi': _abs_scalar_projection(level_data.dVa, e_xi),
        'B_vel_lambda': _abs_scalar_projection(level_data.dVa, e_lambda),
        'V_par': _abs_scalar_projection(level_data.dV, e_l),
        'V_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(level_data.dV, level_data.B_l)),
        'V_xi': _abs_scalar_projection(level_data.dV, e_xi),
        'V_lambda': _abs_scalar_projection(level_data.dV, e_lambda),
        'Zp_par': _abs_scalar_projection(level_data.dZp, e_l),
        'Zp_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(level_data.dZp, level_data.B_l)),
        'Zp_xi': _abs_scalar_projection(level_data.dZp, e_xi),
        'Zp_lambda': _abs_scalar_projection(level_data.dZp, e_lambda),
        'Zm_par': _abs_scalar_projection(level_data.dZm, e_l),
        'Zm_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(level_data.dZm, level_data.B_l)),
        'Zm_xi': _abs_scalar_projection(level_data.dZm, e_xi),
        'Zm_lambda': _abs_scalar_projection(level_data.dZm, e_lambda),
        'tau_equiv_seconds': np.full(level_data.l_mag.shape[0], float(level_data.tau_equiv_samples * level_data.scale_seconds / max(level_data.scale_samples, _EPS))),
        'tau_equiv_samples': np.full(level_data.l_mag.shape[0], float(level_data.tau_equiv_samples)),
        'scale_seconds': np.full(level_data.l_mag.shape[0], float(level_data.scale_seconds)),
        'scale_samples': np.full(level_data.l_mag.shape[0], float(level_data.scale_samples)),
        'frequency_hz': np.full(level_data.l_mag.shape[0], float(level_data.frequency_hz)),
        'period_s': np.full(level_data.l_mag.shape[0], float(level_data.period_s)),
        'bandwidth_hz': np.full(level_data.l_mag.shape[0], float(level_data.bandwidth_hz)),
        'response_energy_integral': np.full(level_data.l_mag.shape[0], float(level_data.response_energy_integral)),
        'level_index': np.full(level_data.l_mag.shape[0], int(level_index), dtype=float),
        'is_effective_level': np.full(level_data.l_mag.shape[0], 1.0 if level_data.is_effective else 0.0, dtype=float),
    }
    for key, val in derived.items():
        _ensure(key, val)
    return {key: np.asarray(val)[:n_rows] for key, val in out.items()}


def _finalize_compact_coefficient_store(
    bucket_rows: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    bucket_ell_identifier: Mapping[str, str],
    b_export_units: str,
    qorder: Sequence[float],
) -> Dict[str, Any]:
    global_columns: List[str] = []
    for rows in bucket_rows.values():
        for row in rows:
            for key in row.keys():
                if key not in global_columns:
                    global_columns.append(key)
    buckets_out: Dict[str, Any] = {}
    for bucket, rows in bucket_rows.items():
        if not rows:
            buckets_out[bucket] = {'data': np.empty((0, len(global_columns)), dtype=np.float64), 'n_rows': 0}
            continue
        matrices = []
        for row in rows:
            n_rows = int(next(iter(row.values())).shape[0]) if row else 0
            cols = []
            for key in global_columns:
                arr = np.asarray(row.get(key, np.full(n_rows, np.nan, dtype=float)))
                arr = np.squeeze(arr)
                if arr.ndim == 0:
                    arr = np.full(n_rows, float(arr.item()), dtype=float)
                cols.append(np.asarray(arr, dtype=np.float64))
            matrices.append(np.column_stack(cols))
        data = np.concatenate(matrices, axis=0) if matrices else np.empty((0, len(global_columns)), dtype=np.float64)
        buckets_out[bucket] = {'data': data, 'n_rows': int(data.shape[0])}
    return {
        'version': 'scwf_compact_coefficients_v1',
        'column_order': list(global_columns),
        'column_units': _column_units_hint(global_columns, b_export_units=b_export_units),
        'bucket_ell_identifier': dict(bucket_ell_identifier),
        'qorder': np.asarray(qorder, dtype=float),
        'b_export_units': str(b_export_units),
        'buckets': buckets_out,
    }


def coefficient_store_to_dataframe(
    store: Mapping[str, Any],
    bucket: str = 'ell_all',
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    payload = store['buckets'][bucket]
    column_order = list(store['column_order'])
    data = np.asarray(payload['data'], dtype=float)
    if columns is None:
        use_cols = column_order
        use_data = data
    else:
        idx = [column_order.index(col) for col in columns]
        use_cols = list(columns)
        use_data = data[:, idx]
    return pd.DataFrame(use_data, columns=use_cols)


def merge_compact_coefficient_stores(stores: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [store for store in stores if store is not None]
    if not valid:
        raise ValueError('No compact coefficient stores were provided.')
    all_columns: List[str] = []
    all_buckets: List[str] = []
    for store in valid:
        for key in store.get('column_order', []):
            if key not in all_columns:
                all_columns.append(key)
        for bucket in store.get('buckets', {}).keys():
            if bucket not in all_buckets:
                all_buckets.append(bucket)
    bucket_ell_identifier: Dict[str, str] = {}
    for store in valid:
        bucket_ell_identifier.update(store.get('bucket_ell_identifier', {}))

    def _aligned_matrix(store: Mapping[str, Any], bucket: str) -> np.ndarray:
        payload = store.get('buckets', {}).get(bucket)
        if payload is None:
            return np.empty((0, len(all_columns)), dtype=np.float64)
        src_cols = list(store.get('column_order', []))
        src_data = np.asarray(payload.get('data', np.empty((0, len(src_cols)))), dtype=np.float64)
        if src_data.size == 0:
            return np.empty((0, len(all_columns)), dtype=np.float64)
        out = np.full((src_data.shape[0], len(all_columns)), np.nan, dtype=np.float64)
        for i, key in enumerate(src_cols):
            out[:, all_columns.index(key)] = src_data[:, i]
        return out

    buckets = {}
    for bucket in all_buckets:
        pieces = [_aligned_matrix(store, bucket) for store in valid]
        data = np.concatenate([p for p in pieces if p.size], axis=0) if any(p.size for p in pieces) else np.empty((0, len(all_columns)), dtype=np.float64)
        buckets[bucket] = {'data': data, 'n_rows': int(data.shape[0])}
    return {
        'version': 'scwf_compact_coefficients_v1',
        'column_order': all_columns,
        'column_units': _column_units_hint(all_columns, b_export_units=str(valid[0].get('b_export_units', 'native'))),
        'bucket_ell_identifier': bucket_ell_identifier,
        'qorder': np.asarray(valid[0].get('qorder', np.array([2.0])), dtype=float),
        'b_export_units': str(valid[0].get('b_export_units', 'native')),
        'buckets': buckets,
    }


def reduce_compact_coefficient_store(
    store: Mapping[str, Any],
    bucket: str,
    value_key: str,
    qorder: Optional[Sequence[float]] = None,
    scale_bin_edges_di: Optional[Sequence[float]] = None,
    scale_key: Optional[str] = None,
    normalization: str = 'scale_normalized',
    constraints: Optional[Mapping[str, Union[float, Sequence[Optional[float]]]]] = None,
    absolute_value: bool = True,
    min_count: int = 1,
) -> Dict[str, Any]:
    payload = store['buckets'][bucket]
    columns = list(store['column_order'])
    col_index = {key: i for i, key in enumerate(columns)}
    data = np.asarray(payload['data'], dtype=float)
    if data.ndim != 2:
        raise ValueError('Compact coefficient payload must be a 2-D array.')
    if value_key not in col_index:
        raise KeyError(f'Column {value_key!r} is not present in the compact coefficient store.')
    if scale_key is None:
        scale_key = store.get('bucket_ell_identifier', {}).get(bucket, 'selected_scale_di')
        if scale_key not in col_index:
            scale_key = 'selected_scale_di'
    if scale_key not in col_index:
        raise KeyError(f'Scale column {scale_key!r} is not present in the compact coefficient store.')
    edges = _resolve_scale_bin_edges_di(scale_bin_edges_di)
    q = np.asarray(store.get('qorder') if qorder is None else qorder, dtype=float)
    values = np.asarray(data[:, col_index[value_key]], dtype=float)
    scales = np.asarray(data[:, col_index[scale_key]], dtype=float)
    mask = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
    if constraints:
        for key, spec in constraints.items():
            if key not in col_index:
                raise KeyError(f'Constraint column {key!r} is not present in the compact coefficient store.')
            arr = np.asarray(data[:, col_index[key]], dtype=float)
            mask &= np.isfinite(arr)
            if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
                seq = list(spec)
                if len(seq) != 2:
                    raise ValueError(f'Constraint for {key!r} must be a scalar or a 2-element sequence.')
                lo, hi = seq
                if lo is not None:
                    mask &= arr >= float(lo)
                if hi is not None:
                    mask &= arr <= float(hi)
            else:
                mask &= arr == float(spec)
    if not np.any(mask):
        centers = np.sqrt(edges[:-1] * edges[1:])
        empty = np.full((q.size, centers.size), np.nan, dtype=float)
        return {'qorder': q, 'scale_bin_edges_di': edges, 'scale_bin_centers_di': centers, 'counts': np.zeros(centers.size, dtype=np.int64), 'mean_scale_di': np.full(centers.size, np.nan), 'moments': empty, 'normalization': normalization, 'value_key': value_key, 'bucket': bucket}
    values = values[mask]
    scales = scales[mask]
    if absolute_value:
        values = np.abs(values)
    bin_idx = np.digitize(scales, edges, right=False) - 1
    keep = (bin_idx >= 0) & (bin_idx < edges.size - 1)
    values = values[keep]
    scales = scales[keep]
    bin_idx = bin_idx[keep]
    counts = np.bincount(bin_idx, minlength=edges.size - 1).astype(np.int64)
    sum_scale = np.bincount(bin_idx, weights=scales, minlength=edges.size - 1)
    mean_scale = np.full(edges.size - 1, np.nan, dtype=float)
    valid_counts = counts >= int(min_count)
    mean_scale[valid_counts] = sum_scale[valid_counts] / counts[valid_counts]
    out = np.full((q.size, edges.size - 1), np.nan, dtype=float)
    if normalization.lower() in {'psd', 'spectrum', 'spectra'}:
        if 'response_energy_integral' not in col_index:
            raise KeyError('response_energy_integral is required for PSD reduction.')
        rei = np.asarray(data[:, col_index['response_energy_integral']], dtype=float)[mask][keep]
        contrib = np.square(values) / np.maximum(rei, _EPS)
        sums = np.bincount(bin_idx, weights=contrib, minlength=edges.size - 1)
        out = np.full((1, edges.size - 1), np.nan, dtype=float)
        out[0, valid_counts] = sums[valid_counts] / counts[valid_counts]
        q_used = np.asarray([2.0], dtype=float)
    else:
        if normalization.lower() in {'scale_normalized', 'structure_function_surrogate', 'increment_equivalent'}:
            if 'tau_equiv_seconds' not in col_index:
                raise KeyError('tau_equiv_seconds is required for scale-normalized moment reduction.')
            tau = np.asarray(data[:, col_index['tau_equiv_seconds']], dtype=float)[mask][keep]
        else:
            tau = None
        for iq, qv in enumerate(q):
            contrib = np.power(values, qv)
            if tau is not None:
                contrib = contrib / np.power(np.maximum(tau, _EPS), 0.5 * qv)
            sums = np.bincount(bin_idx, weights=contrib, minlength=edges.size - 1)
            out[iq, valid_counts] = sums[valid_counts] / counts[valid_counts]
        q_used = q
    return {
        'qorder': q_used,
        'scale_bin_edges_di': edges,
        'scale_bin_centers_di': np.sqrt(edges[:-1] * edges[1:]),
        'counts': counts,
        'mean_scale_di': mean_scale,
        'moments': out,
        'normalization': normalization,
        'value_key': value_key,
        'bucket': bucket,
        'scale_key': scale_key,
        'constraints': dict(constraints) if constraints else None,
    }


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



_PROJECTED_FAMILY_KEYS = (
    'B_par', 'B_perp', 'B_xi', 'B_lambda',
    'B_nT_par', 'B_nT_perp', 'B_nT_xi', 'B_nT_lambda',
    'B_vel_par', 'B_vel_perp', 'B_vel_xi', 'B_vel_lambda',
    'V_par', 'V_perp', 'V_xi', 'V_lambda',
    'Zp_par', 'Zp_perp', 'Zp_xi', 'Zp_lambda',
    'Zm_par', 'Zm_perp', 'Zm_xi', 'Zm_lambda',
)
_DEFAULT_SCALE_BIN_EDGES_DI = np.logspace(np.log10(0.5), np.log10(1.0e5), 97)


def _mean_from_indices(values: np.ndarray, indices: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return np.nan
    subset = np.asarray(values, dtype=float)[indices]
    finite = np.isfinite(subset)
    if not np.any(finite):
        return np.nan
    return float(np.nansum(subset[finite]) / np.sum(finite))


def _resolve_scale_bin_edges_di(scale_bin_edges_di: Optional[Sequence[float]]) -> np.ndarray:
    if scale_bin_edges_di is None:
        edges = np.asarray(_DEFAULT_SCALE_BIN_EDGES_DI, dtype=float)
    else:
        edges = np.asarray(scale_bin_edges_di, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('scale_bin_edges_di must be a one-dimensional array with at least two edges.')
    if not np.all(np.isfinite(edges)):
        raise ValueError('scale_bin_edges_di must contain only finite values.')
    if not np.all(edges > 0.0):
        raise ValueError('scale_bin_edges_di must be strictly positive.')
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError('scale_bin_edges_di must be strictly increasing.')
    return edges


def _empty_scale_binned_accumulator(
    family_keys: Sequence[str],
    buckets: Sequence[str],
    scale_edges_di: np.ndarray,
    qorder: np.ndarray,
) -> Dict[str, Any]:
    n_bins = int(scale_edges_di.size - 1)
    nq = int(qorder.size)

    def _bucket_container() -> Dict[str, np.ndarray]:
        return {
            'sum_raw': np.zeros((n_bins, nq), dtype=float),
            'sum_scale_normalized': np.zeros((n_bins, nq), dtype=float),
            'sum_psd': np.zeros(n_bins, dtype=float),
            'count': np.zeros(n_bins, dtype=np.int64),
            'sum_scale_di': np.zeros(n_bins, dtype=float),
            'sum_scale_di2': np.zeros(n_bins, dtype=float),
        }

    return {
        'scale_bin_edges_di': np.asarray(scale_edges_di, dtype=float),
        'scale_bin_centers_di': np.sqrt(np.asarray(scale_edges_di[:-1], dtype=float) * np.asarray(scale_edges_di[1:], dtype=float)),
        'qorder': np.asarray(qorder, dtype=float),
        'trace': {family: {bucket: _bucket_container() for bucket in buckets} for family in family_keys},
        'projected': {family: {bucket: _bucket_container() for bucket in buckets} for family in _PROJECTED_FAMILY_KEYS},
    }


def _accumulate_scale_binned_family(
    entry: MutableMapping[str, np.ndarray],
    power_matrix: np.ndarray,
    amp_sq: np.ndarray,
    indices: np.ndarray,
    scale_values_di: np.ndarray,
    scale_bin_edges_di: np.ndarray,
    tau_seconds: float,
    response_energy_integral: float,
    qorder: np.ndarray,
) -> None:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return
    scales = np.asarray(scale_values_di, dtype=float)[indices]
    valid = np.isfinite(scales) & (scales > 0.0)
    if not np.any(valid):
        return
    scales = scales[valid]
    power_subset = np.asarray(power_matrix, dtype=float)[indices][valid]
    amp_sq_subset = np.asarray(amp_sq, dtype=float)[indices][valid]
    bin_idx = np.digitize(scales, scale_bin_edges_di, right=False) - 1
    keep = (bin_idx >= 0) & (bin_idx < scale_bin_edges_di.size - 1)
    if not np.any(keep):
        return
    scales = scales[keep]
    power_subset = power_subset[keep]
    amp_sq_subset = amp_sq_subset[keep]
    bin_idx = bin_idx[keep]
    n_bins = int(scale_bin_edges_di.size - 1)
    entry['count'] += np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    entry['sum_scale_di'] += np.bincount(bin_idx, weights=scales, minlength=n_bins)
    entry['sum_scale_di2'] += np.bincount(bin_idx, weights=np.square(scales), minlength=n_bins)
    tau_factor = np.power(max(float(tau_seconds), _EPS), 0.5 * np.asarray(qorder, dtype=float))
    for iq in range(power_subset.shape[1]):
        entry['sum_raw'][:, iq] += np.bincount(bin_idx, weights=power_subset[:, iq], minlength=n_bins)
        entry['sum_scale_normalized'][:, iq] += np.bincount(bin_idx, weights=power_subset[:, iq] / max(float(tau_factor[iq]), _EPS), minlength=n_bins)
    entry['sum_psd'] += np.bincount(bin_idx, weights=amp_sq_subset / max(float(response_energy_integral), _EPS), minlength=n_bins)


def _finalize_scale_binned_accumulator(accumulator: Mapping[str, Any]) -> Dict[str, Any]:
    edges = np.asarray(accumulator['scale_bin_edges_di'], dtype=float)
    centers = np.asarray(accumulator['scale_bin_centers_di'], dtype=float)
    qorder = np.asarray(accumulator['qorder'], dtype=float)

    def _safe_mean(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
        out = np.full_like(sum_arr, np.nan, dtype=float)
        valid = count_arr > 0
        if sum_arr.ndim == 1:
            out[valid] = sum_arr[valid] / count_arr[valid]
        else:
            out[valid, :] = sum_arr[valid, :] / count_arr[valid, None]
        return out

    def _family_output(source: Mapping[str, Any]) -> Dict[str, Any]:
        moments = {}
        normalized = {}
        spectra = {}
        counts = {}
        mean_scale = {}
        rms_scale = {}
        for family, bucket_map in source.items():
            moments[family] = {}
            normalized[family] = {}
            spectra[family] = {}
            counts[family] = {}
            mean_scale[family] = {}
            rms_scale[family] = {}
            for bucket, entry in bucket_map.items():
                count = np.asarray(entry['count'], dtype=np.int64)
                raw_mean = _safe_mean(np.asarray(entry['sum_raw'], dtype=float), count)
                norm_mean = _safe_mean(np.asarray(entry['sum_scale_normalized'], dtype=float), count)
                psd_mean = _safe_mean(np.asarray(entry['sum_psd'], dtype=float), count)
                scale_mean = _safe_mean(np.asarray(entry['sum_scale_di'], dtype=float), count)
                scale2_mean = _safe_mean(np.asarray(entry['sum_scale_di2'], dtype=float), count)
                rms = np.full_like(scale_mean, np.nan, dtype=float)
                valid = np.isfinite(scale2_mean)
                rms[valid] = np.sqrt(np.maximum(scale2_mean[valid], 0.0))
                moments[family][bucket] = raw_mean.T
                normalized[family][bucket] = norm_mean.T
                spectra[family][bucket] = psd_mean
                counts[family][bucket] = count
                mean_scale[family][bucket] = scale_mean
                rms_scale[family][bucket] = rms
        return {
            'WaveletMoments': moments,
            'ScaleNormalizedWaveletMoments': normalized,
            'Spectra': spectra,
            'counts': counts,
            'mean_scale_di': mean_scale,
            'rms_scale_di': rms_scale,
        }

    return {
        'scale_bin_edges_di': edges,
        'scale_bin_centers_di': centers,
        'qorder': qorder,
        'trace': _family_output(accumulator['trace']),
        'projected': _family_output(accumulator['projected']),
    }


def merge_scale_binned_accumulators(accumulators: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid_accs = [acc for acc in accumulators if acc is not None]
    if not valid_accs:
        raise ValueError('No accumulators were provided.')

    def _clone_entry(entry: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: np.array(v, copy=True) for k, v in entry.items()}

    base = {
        'scale_bin_edges_di': np.array(valid_accs[0]['scale_bin_edges_di'], copy=True),
        'scale_bin_centers_di': np.array(valid_accs[0]['scale_bin_centers_di'], copy=True),
        'qorder': np.array(valid_accs[0]['qorder'], copy=True),
        'trace': {family: {bucket: _clone_entry(entry) for bucket, entry in bucket_map.items()} for family, bucket_map in valid_accs[0]['trace'].items()},
        'projected': {family: {bucket: _clone_entry(entry) for bucket, entry in bucket_map.items()} for family, bucket_map in valid_accs[0]['projected'].items()},
    }
    for acc in valid_accs[1:]:
        if not np.allclose(np.asarray(acc['scale_bin_edges_di'], dtype=float), base['scale_bin_edges_di'], equal_nan=True):
            raise ValueError('All scale-binned accumulators must use the same scale_bin_edges_di.')
        if not np.allclose(np.asarray(acc['qorder'], dtype=float), base['qorder'], equal_nan=True):
            raise ValueError('All scale-binned accumulators must use the same qorder.')
        for domain in ('trace', 'projected'):
            for family, bucket_map in acc[domain].items():
                for bucket, entry in bucket_map.items():
                    for key, arr in entry.items():
                        base[domain][family][bucket][key] += np.asarray(arr, dtype=base[domain][family][bucket][key].dtype)
    return base


def finalize_scale_binned_accumulator(accumulator: Mapping[str, Any]) -> Dict[str, Any]:
    return _finalize_scale_binned_accumulator(accumulator)


def reduce_saved_interval_accumulators(paths: Sequence[Union[str, os.PathLike]]) -> Dict[str, Any]:
    accumulators = []
    for path in paths:
        payload = pd.read_pickle(path)
        acc = payload.get('ScaleBinnedAccumulator') if isinstance(payload, Mapping) else None
        if acc is None:
            raise KeyError(f'File {path!s} does not contain a ScaleBinnedAccumulator entry.')
        accumulators.append(acc)
    merged = merge_scale_binned_accumulators(accumulators)
    return finalize_scale_binned_accumulator(merged)


def reduce_saved_interval_coefficient_stores(paths: Sequence[Union[str, os.PathLike]]) -> Dict[str, Any]:
    stores = []
    for path in paths:
        payload = pd.read_pickle(path)
        store = None
        if isinstance(payload, Mapping):
            store = payload.get('CoefficientStore', payload.get('CompactCoefficients'))
        if store is None:
            raise KeyError(f'File {path!s} does not contain a CoefficientStore/CompactCoefficients entry.')
        stores.append(store)
    return merge_compact_coefficient_stores(stores)


def reduce_saved_interval_coefficients(
    paths: Sequence[Union[str, os.PathLike]],
    bucket: str,
    value_key: str,
    qorder: Optional[Sequence[float]] = None,
    scale_bin_edges_di: Optional[Sequence[float]] = None,
    scale_key: Optional[str] = None,
    normalization: str = 'scale_normalized',
    constraints: Optional[Mapping[str, Union[float, Sequence[Optional[float]]]]] = None,
    absolute_value: bool = True,
    min_count: int = 1,
) -> Dict[str, Any]:
    merged_store = reduce_saved_interval_coefficient_stores(paths)
    return reduce_compact_coefficient_store(
        merged_store,
        bucket=bucket,
        value_key=value_key,
        qorder=qorder,
        scale_bin_edges_di=scale_bin_edges_di,
        scale_key=scale_key,
        normalization=normalization,
        constraints=constraints,
        absolute_value=absolute_value,
        min_count=min_count,
    )


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
    scale_bin_edges_di: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    q = np.asarray([2.0] if qorder is None else qorder, dtype=float)
    scale_bin_edges_di_arr = _resolve_scale_bin_edges_di(scale_bin_edges_di)
    results = estimate_local_wavelet_geometry(
        B=B, V=V, V_sc_vel_removed=V_sc_vel_removed, Np=Np, dt=dt, wname=wname, level=level, level_mode=level_mode,
        estimate_alignment_angle=estimate_alignment_angle, return_B_in_vel_units=return_B_in_vel_units,
        use_local_polarity=use_local_polarity, sc=sc, frame=frame,
        min_valid_fraction=min_valid_fraction, min_valid_count=min_valid_count,
    )
    if di is None:
        di = float(results['di_mean'])
    n_levels = len(results['levels'])
    nbins = ('ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall')
    sf_shape = (n_levels, len(q))

    def _empty_moment_container() -> Dict[str, np.ndarray]:
        return {key: np.full(sf_shape, np.nan) for key in nbins}

    family_keys = ('B', 'B_nT', 'B_vel', 'V', 'Zp', 'Zm', 'LeaderB')
    projected_family_keys = tuple(_PROJECTED_FAMILY_KEYS)
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
    projected_wavelet_moments = {key: _empty_moment_container() for key in projected_family_keys}
    projected_wavelet_moments['counts'] = wavelet_moments['counts']
    projected_wavelet_moments['l_di'] = wavelet_moments['l_di']
    projected_sf = {key: _empty_moment_container() for key in projected_family_keys}
    projected_sf['counts'] = sf['counts']
    projected_sf['l_di'] = sf['l_di']
    projected_spectra = {key: {bucket: np.full(n_levels, np.nan) for bucket in nbins} for key in projected_family_keys}
    projected_spectra['counts'] = sf['counts']
    projected_spectra['l_di'] = sf['l_di']
    projected_spectra['frequency_hz'] = np.full(n_levels, np.nan)
    projected_spectra['bandwidth_hz'] = np.full(n_levels, np.nan)
    projected_spectra['response_energy_integral'] = np.full(n_levels, np.nan)
    bucket_scale_stats = {bucket: {name: np.full(n_levels, np.nan) for name in ('mean','median','p16','p84','p05','p95','min','max')} for bucket in nbins}
    scale_binned_accumulator = _empty_scale_binned_accumulator(family_keys, nbins, scale_bin_edges_di_arr, q)
    fluct_tables = {'ell_perp': {}, 'Ell_perp': {}, 'ell_par': {}, 'ell_par_rest': {}, 'ell_all': {}, 'by_level': {}}
    compact_bucket_rows: Dict[str, List[Dict[str, np.ndarray]]] = {'ell_perp': [], 'Ell_perp': [], 'ell_par': [], 'ell_par_rest': [], 'ell_all': []}
    bucket_ell_identifier = {'ell_perp': 'l_lambda', 'Ell_perp': 'l_xi', 'ell_par': 'l_ell', 'ell_par_rest': 'l_ell', 'ell_all': 'l_mag'}
    all_thetas, all_phis = {}, {}
    align_summary = {
        'VB': {'reg': [], 'polar': [], 'weighted': [], 'sig_r_scale': [], 'sig_r_local_mean': [], 'sig_r_local_median': [], 'u_perp2_mean': [], 'va_perp2_mean': [], 'counts': []},
        'Zpm': {'reg': [], 'polar': [], 'weighted': [], 'sig_c_scale': [], 'sig_c_local_mean': [], 'sig_c_local_median': [], 'zp_perp2_mean': [], 'zm_perp2_mean': [], 'counts': []},
    }
    cond_perp, cond_disp, cond_par = conditions['ell_perp'], conditions['Ell_perp'], conditions['ell_par']
    cond_par_rest = conditions.get('ell_par_rest', cond_par)

    for j, lvl in enumerate(results['levels']):
        all_thetas[str(j)] = lvl.VBangle
        all_phis[str(j)] = lvl.Phiangle
        for container in (spectra, legacy_spectra, projected_spectra):
            container['frequency_hz'][j] = lvl.frequency_hz
            container['bandwidth_hz'][j] = lvl.bandwidth_hz
        spectra['response_energy_integral'][j] = lvl.response_energy_integral
        projected_spectra['response_energy_integral'][j] = lvl.response_energy_integral
        if estimate_alignment_angle and lvl.align_angles_vb and lvl.align_angles_zpm:
            for k in align_summary['VB']:
                align_summary['VB'][k].append(lvl.align_angles_vb[k])
            for k in align_summary['Zpm']:
                align_summary['Zpm'][k].append(lvl.align_angles_zpm[k])
        if respect_effective_levels and not lvl.is_effective:
            continue
        finite_theta = np.isfinite(lvl.VBangle) & np.isfinite(lvl.l_mag) & np.isfinite(lvl.l_ell)
        finite_phi_lambda = finite_theta & np.isfinite(lvl.Phiangle) & np.isfinite(lvl.l_lambda)
        finite_phi_xi = finite_theta & np.isfinite(lvl.Phiangle) & np.isfinite(lvl.l_xi)
        idx_perp = np.flatnonzero(finite_phi_lambda & (lvl.VBangle > cond_perp['theta']) & (lvl.Phiangle > cond_perp['phi']))
        idx_disp = np.flatnonzero(finite_phi_xi & (lvl.VBangle > cond_disp['theta']) & (lvl.Phiangle < cond_disp['phi']))
        idx_par = np.flatnonzero(finite_theta & (lvl.VBangle < cond_par['theta']))
        idx_par_rest = np.flatnonzero(finite_theta & (lvl.VBangle < cond_par_rest['theta']))
        idx_all = np.flatnonzero(np.isfinite(lvl.l_mag))
        for name, idx_sel in [('ell_perp', idx_perp), ('Ell_perp', idx_disp), ('ell_par', idx_par), ('ell_par_rest', idx_par_rest), ('ell_overall', idx_all)]:
            sf['counts'][name][j] = idx_sel.size
        trace_amp_lookup = {
            'B': _estimate_vec_magnitude_stacked(lvl.dB),
            'B_nT': _estimate_vec_magnitude_stacked(lvl.dB_nT),
            'B_vel': _estimate_vec_magnitude_stacked(lvl.dVa),
            'V': _estimate_vec_magnitude_stacked(lvl.dV),
            'Zp': _estimate_vec_magnitude_stacked(lvl.dZp),
            'Zm': _estimate_vec_magnitude_stacked(lvl.dZm),
            'LeaderB': np.abs(np.asarray(lvl.leader_B, dtype=float)),
        }
        trace_power_lookup = {k: np.power(np.asarray(v, dtype=float)[:, None], q[None, :]) for k, v in trace_amp_lookup.items()}
        trace_amp_sq_lookup = {k: np.square(np.asarray(v, dtype=float)) for k, v in trace_amp_lookup.items()}
        e_l, e_xi, e_lambda, _basis_valid = _local_basis_from_background_and_perp(lvl.B_l, lvl.dB_perp_nT)
        projected_amp_lookup = {
            'B_par': _abs_scalar_projection(lvl.dB, e_l), 'B_perp': _estimate_vec_magnitude_stacked(lvl.dB_perp), 'B_xi': _abs_scalar_projection(lvl.dB, e_xi), 'B_lambda': _abs_scalar_projection(lvl.dB, e_lambda),
            'B_nT_par': _abs_scalar_projection(lvl.dB_nT, e_l), 'B_nT_perp': _estimate_vec_magnitude_stacked(lvl.dB_perp_nT), 'B_nT_xi': _abs_scalar_projection(lvl.dB_nT, e_xi), 'B_nT_lambda': _abs_scalar_projection(lvl.dB_nT, e_lambda),
            'B_vel_par': _abs_scalar_projection(lvl.dVa, e_l), 'B_vel_perp': _estimate_vec_magnitude_stacked(lvl.dVa_perp), 'B_vel_xi': _abs_scalar_projection(lvl.dVa, e_xi), 'B_vel_lambda': _abs_scalar_projection(lvl.dVa, e_lambda),
            'V_par': _abs_scalar_projection(lvl.dV, e_l), 'V_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(lvl.dV, lvl.B_l)), 'V_xi': _abs_scalar_projection(lvl.dV, e_xi), 'V_lambda': _abs_scalar_projection(lvl.dV, e_lambda),
            'Zp_par': _abs_scalar_projection(lvl.dZp, e_l), 'Zp_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(lvl.dZp, lvl.B_l)), 'Zp_xi': _abs_scalar_projection(lvl.dZp, e_xi), 'Zp_lambda': _abs_scalar_projection(lvl.dZp, e_lambda),
            'Zm_par': _abs_scalar_projection(lvl.dZm, e_l), 'Zm_perp': _estimate_vec_magnitude_stacked(_perp_vector_stacked(lvl.dZm, lvl.B_l)), 'Zm_xi': _abs_scalar_projection(lvl.dZm, e_xi), 'Zm_lambda': _abs_scalar_projection(lvl.dZm, e_lambda),
        }
        projected_power_lookup = {k: np.power(np.asarray(v, dtype=float)[:, None], q[None, :]) for k, v in projected_amp_lookup.items()}
        projected_amp_sq_lookup = {k: np.square(np.asarray(v, dtype=float)) for k, v in projected_amp_lookup.items()}
        for bucket, idx_sel, ell_arr in [('ell_perp', idx_perp, lvl.l_lambda), ('Ell_perp', idx_disp, lvl.l_xi), ('ell_par', idx_par, lvl.l_ell), ('ell_par_rest', idx_par_rest, lvl.l_ell), ('ell_overall', idx_all, lvl.l_mag)]:
            if idx_sel.size:
                for family in family_keys:
                    wavelet_moments[family][bucket][j] = _moments_from_precomputed_powers(trace_power_lookup[family], idx_sel)
                    sf[family][bucket][j] = _increment_equivalent_moments_from_wavelet_moments(wavelet_moments[family][bucket][j], q, lvl.tau_equiv_samples * dt)
                    spectra[family][bucket][j] = _mean_from_indices(trace_amp_sq_lookup[family], idx_sel) / max(lvl.response_energy_integral, _EPS)
                    legacy_spectra[family][bucket][j] = _mean_from_indices(trace_amp_sq_lookup[family], idx_sel) / max(lvl.bandwidth_hz, _EPS)
                    _accumulate_scale_binned_family(scale_binned_accumulator['trace'][family][bucket], trace_power_lookup[family], trace_amp_sq_lookup[family], idx_sel, ell_arr, scale_bin_edges_di_arr, lvl.tau_equiv_samples * dt, lvl.response_energy_integral, q)
                for family in projected_family_keys:
                    projected_wavelet_moments[family][bucket][j] = _moments_from_precomputed_powers(projected_power_lookup[family], idx_sel)
                    projected_sf[family][bucket][j] = _increment_equivalent_moments_from_wavelet_moments(projected_wavelet_moments[family][bucket][j], q, lvl.tau_equiv_samples * dt)
                    projected_spectra[family][bucket][j] = _mean_from_indices(projected_amp_sq_lookup[family], idx_sel) / max(lvl.response_energy_integral, _EPS)
                    _accumulate_scale_binned_family(scale_binned_accumulator['projected'][family][bucket], projected_power_lookup[family], projected_amp_sq_lookup[family], idx_sel, ell_arr, scale_bin_edges_di_arr, lvl.tau_equiv_samples * dt, lvl.response_energy_integral, q)
                stat_summary = _bucket_scale_stat_summary(ell_arr[idx_sel])
                for stat_name, stat_val in stat_summary.items():
                    bucket_scale_stats[bucket][stat_name][j] = stat_val
                sf['l_di'][bucket][j] = stat_summary['mean']
                wavelet_moments['l_di'][bucket][j] = stat_summary['mean']
        if return_coefs:
            variables = estimate_requested_quantities(lvl, results, ts_list=ts_list)
            fluct_tables['by_level'][str(j)] = variables
            if idx_perp.size:
                saved = _save_flucs(idx_perp, variables, lvl.l_lambda, 'l_lambda')
                fluct_tables['ell_perp'][str(j)] = pd.DataFrame(saved)
                compact_bucket_rows['ell_perp'].append(_augment_saved_flucs_for_compact_store(saved, lvl, j, 'l_lambda', results['B_export_units']))
            if idx_disp.size:
                saved = _save_flucs(idx_disp, variables, lvl.l_xi, 'l_xi')
                fluct_tables['Ell_perp'][str(j)] = pd.DataFrame(saved)
                compact_bucket_rows['Ell_perp'].append(_augment_saved_flucs_for_compact_store(saved, lvl, j, 'l_xi', results['B_export_units']))
            if idx_par.size:
                saved = _save_flucs(idx_par, variables, lvl.l_ell, 'l_ell')
                fluct_tables['ell_par'][str(j)] = pd.DataFrame(saved)
                compact_bucket_rows['ell_par'].append(_augment_saved_flucs_for_compact_store(saved, lvl, j, 'l_ell', results['B_export_units']))
            if idx_par_rest.size:
                saved = _save_flucs(idx_par_rest, variables, lvl.l_ell, 'l_ell')
                fluct_tables['ell_par_rest'][str(j)] = pd.DataFrame(saved)
                compact_bucket_rows['ell_par_rest'].append(_augment_saved_flucs_for_compact_store(saved, lvl, j, 'l_ell', results['B_export_units']))
            if idx_all.size:
                saved = _save_flucs(idx_all, variables, lvl.l_mag, 'l_mag')
                fluct_tables['ell_all'][str(j)] = pd.DataFrame(saved)
                compact_bucket_rows['ell_all'].append(_augment_saved_flucs_for_compact_store(saved, lvl, j, 'l_mag', results['B_export_units']))

    flucts = None
    compact_coefficients = None
    if return_coefs:
        flucts = {k: (pd.concat(v, axis=0) if v else pd.DataFrame()) for k, v in [('ell_perp', fluct_tables['ell_perp']), ('Ell_perp', fluct_tables['Ell_perp']), ('ell_par', fluct_tables['ell_par']), ('ell_par_rest', fluct_tables['ell_par_rest']), ('ell_all', fluct_tables['ell_all'])]}
        flucts['by_level'] = fluct_tables['by_level']
        compact_coefficients = _finalize_compact_coefficient_store(compact_bucket_rows, bucket_ell_identifier, results['B_export_units'], q)
    overall_align_angles = {'VB': {k: np.asarray(v) for k, v in align_summary['VB'].items()}, 'Zpm': {k: np.asarray(v) for k, v in align_summary['Zpm'].items()}}
    wavelet_moments_out = {family: {k: (v.T if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in wavelet_moments[family].items()} for family in family_keys}
    wavelet_moments_out['counts'] = wavelet_moments['counts']; wavelet_moments_out['l_di'] = wavelet_moments['l_di']
    sfuncs = {family: {k: (v.T if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in sf[family].items()} for family in family_keys}
    sfuncs['counts'] = sf['counts']; sfuncs['l_di'] = sf['l_di']
    projected_wavelet_moments_out = {family: {k: (v.T if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in projected_wavelet_moments[family].items()} for family in projected_family_keys}
    projected_wavelet_moments_out['counts'] = wavelet_moments['counts']; projected_wavelet_moments_out['l_di'] = wavelet_moments['l_di']
    projected_sfuncs = {family: {k: (v.T if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in projected_sf[family].items()} for family in projected_family_keys}
    projected_sfuncs['counts'] = sf['counts']; projected_sfuncs['l_di'] = sf['l_di']
    scale_axis = {
        'levels': np.asarray(results['meta']['levels'], dtype=int),
        'tau_equiv_samples': np.asarray(results['meta']['tau_equiv_samples'], dtype=float),
        'tau_equiv_seconds': np.asarray(results['meta']['tau_equiv_samples'], dtype=float) * dt,
        'scale_samples': np.asarray(results['meta']['scale_samples'], dtype=float),
        'scale_seconds': np.asarray(results['meta']['scale_seconds'], dtype=float),
        'frequency_hz': np.asarray(results['meta']['frequency_hz'], dtype=float),
        'period_s': np.asarray(results['meta']['period_s'], dtype=float),
        'bandwidth_hz': np.asarray(results['meta']['bandwidth_hz'], dtype=float),
        'response_energy_integral': np.asarray(results['meta']['response_energy_integral'], dtype=float),
        'effective_level_mask': np.asarray(results['meta'].get('effective_level_mask', np.ones(n_levels, dtype=bool)), dtype=bool),
        'n_valid': np.asarray(results['meta'].get('n_valid', np.full(n_levels, np.nan)), dtype=float),
        'valid_fraction': np.asarray(results['meta'].get('valid_fraction', np.full(n_levels, np.nan)), dtype=float),
        'scale_bin_edges_di': np.asarray(scale_bin_edges_di_arr, dtype=float),
        'scale_bin_centers_di': np.sqrt(np.asarray(scale_bin_edges_di_arr[:-1], dtype=float) * np.asarray(scale_bin_edges_di_arr[1:], dtype=float)),
    }
    scale_binned_products = _finalize_scale_binned_accumulator(scale_binned_accumulator)
    l_overall = sf['l_di']['ell_overall']
    return {
        'thetas': all_thetas, 'phis': all_phis, 'flucts': flucts, 'l_di': l_overall, 'ell_di': l_overall,
        'Sfunctions': sfuncs, 'Sfuncs': sfuncs, 'StructureFunctions': sfuncs, 'IncrementEquivalentWaveletMoments': sfuncs, 'IncrementEquivalentMoments': sfuncs, 'ScaleNormalizedWaveletMoments': sfuncs,
        'WaveletMoments': wavelet_moments_out, 'RawWaveletMoments': wavelet_moments_out,
        'ProjectedWaveletMoments': projected_wavelet_moments_out, 'ComponentWaveletMoments': projected_wavelet_moments_out,
        'ProjectedScaleNormalizedWaveletMoments': projected_sfuncs, 'ComponentScaleNormalizedWaveletMoments': projected_sfuncs, 'ProjectedSfuncs': projected_sfuncs,
        'ProjectedSpectra': projected_spectra, 'ComponentSpectra': projected_spectra,
        'Spectra': spectra, 'ConditionalBandAveragedPSD': spectra, 'ConditionalBandPower': spectra,
        'ScaleBinnedAccumulator': scale_binned_accumulator, 'ScaleBinnedProducts': scale_binned_products,
        'ScaleBinnedWaveletMoments': scale_binned_products['trace']['WaveletMoments'],
        'ScaleBinnedScaleNormalizedWaveletMoments': scale_binned_products['trace']['ScaleNormalizedWaveletMoments'],
        'ScaleBinnedSpectra': scale_binned_products['trace']['Spectra'],
        'ScaleBinnedProjectedWaveletMoments': scale_binned_products['projected']['WaveletMoments'],
        'ScaleBinnedProjectedScaleNormalizedWaveletMoments': scale_binned_products['projected']['ScaleNormalizedWaveletMoments'],
        'ScaleBinnedProjectedSpectra': scale_binned_products['projected']['Spectra'],
        'ScaleBinnedCounts': scale_binned_products['trace']['counts'],
        'ScaleBinnedProjectedCounts': scale_binned_products['projected']['counts'],
        'ScaleBinnedMeanScaleDi': scale_binned_products['trace']['mean_scale_di'],
        'ScaleBinnedProjectedMeanScaleDi': scale_binned_products['projected']['mean_scale_di'],
        'CompactCoefficients': compact_coefficients, 'CoefficientStore': compact_coefficients,
        'BucketScaleStats': bucket_scale_stats, 'ScaleAxis': scale_axis, 'LegacySpectraBandwidth': legacy_spectra,
        'PDFs': None, 'overall_align_angles': overall_align_angles, 'meta': results['meta'],
        'B_export_units': results['B_export_units'], 'B_analysis_units': results['B_export_units'], 'raw': results,
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
    scale_bin_edges_di: Optional[Sequence[float]] = None,
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
            scale_bin_edges_di=scale_bin_edges_di,
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
            'ProjectedWaveletMoments': interval_out['ProjectedWaveletMoments'],
            'ProjectedScaleNormalizedWaveletMoments': interval_out['ProjectedScaleNormalizedWaveletMoments'],
            'ProjectedSpectra': interval_out['ProjectedSpectra'],
            'ScaleBinnedAccumulator': interval_out['ScaleBinnedAccumulator'],
            'ScaleBinnedProducts': interval_out['ScaleBinnedProducts'],
            'ScaleBinnedWaveletMoments': interval_out['ScaleBinnedWaveletMoments'],
            'ScaleBinnedScaleNormalizedWaveletMoments': interval_out['ScaleBinnedScaleNormalizedWaveletMoments'],
            'ScaleBinnedSpectra': interval_out['ScaleBinnedSpectra'],
            'ScaleBinnedProjectedWaveletMoments': interval_out['ScaleBinnedProjectedWaveletMoments'],
            'ScaleBinnedProjectedScaleNormalizedWaveletMoments': interval_out['ScaleBinnedProjectedScaleNormalizedWaveletMoments'],
            'ScaleBinnedProjectedSpectra': interval_out['ScaleBinnedProjectedSpectra'],
            'BucketScaleStats': interval_out['BucketScaleStats'],
            'ScaleAxis': interval_out['ScaleAxis'],
            'Spectra': interval_out['Spectra'],
            'CompactCoefficients': interval_out['CompactCoefficients'],
            'CoefficientStore': interval_out['CoefficientStore'],
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
                'note': 'Sfuncs/StructureFunctions/ScaleNormalizedWaveletMoments store scale-normalized wavelet-moment surrogates obtained by removing the universal L2-wavelet tau^(q/2) factor at the same tau_equiv used in Taylor mapping. They are not strict finite-difference structure functions. WaveletMoments stores the raw conditional wavelet-coefficient moments. Spectra/ConditionalBandAveragedPSD stores second-order raw wavelet moments of the full vector coefficient magnitude divided by the positive-frequency discrete response-energy integral of the implemented FFT filter; the local basis is used to define conditional angle bins for the trace/vector observables. Separate projected-component products are now also exported. For B itself, however, the xi direction is defined from the same perpendicular magnetic coefficient, so B_xi equals |B_perp| and B_lambda is identically zero up to numerical error; those two magnetic projected components must not be interpreted as independent polarization diagnostics. ScaleBinnedAccumulator stores pointwise-accumulated raw moments, scale-normalized moments, PSD-normalized second-order contributions, and counts on a common physical d_i grid so multiple intervals can be merged exactly before finalization. CompactCoefficients/CoefficientStore stores the selected coefficient rows as dense NumPy matrices with explicit column_order, column_units, and bucket labels so intervals can be merged quickly without pandas overhead and reduced later with reduce_compact_coefficient_store. LegacySpectraBandwidth preserves the older bandwidth-based normalization only for comparison. The Gaussian background and even-DoG fluctuation are paired filters, not an exact additive reconstruction. When spacecraft-corrected velocity is supplied, both the Taylor mapping and the velocity/Elsasser fluctuations use that corrected velocity consistently. Raw contextual diagnostics are evaluated on aligned raw series rather than on the gap-filled transform inputs. If the frame does not identify a radial-like component, polarity correction falls back to unity rather than guessing from the first component. The compatibility B family still follows the user-selected magnetic unit system; B_nT and B_vel are exported explicitly to avoid unit ambiguity.',
                'package_name': _PACKAGE_NAME,
                'method_token': method_token,
                'output_subdir': resolved_output_subdir,
                'file_name_root': file_name_root,
                'wavelet_meta': interval_out['meta'],
                'scale_bin_edges_di': None if scale_bin_edges_di is None else list(np.asarray(scale_bin_edges_di, dtype=float)),
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


# ======================================================================================
# Rev9 overrides: compact coefficient-only export, no duplicated directional buckets,
# and a trimmed file payload for the first-stage fluctuation pass.
# ======================================================================================

_estimate_wavelet_interval_legacy = estimate_wavelet_interval
_run_filterbank_interval_analysis_legacy = run_filterbank_interval_analysis
_coefficient_store_to_dataframe_legacy = coefficient_store_to_dataframe
_reduce_compact_coefficient_store_legacy = reduce_compact_coefficient_store


def _normalize_requested(ts_list: Optional[Union[str, Sequence[str]]]) -> set:
    """Return only explicitly requested auxiliary coefficient fields.

    Rev9 keeps the coefficient export lean by default. Auxiliary row-level diagnostics
    are added only when they are requested explicitly.
    """
    if ts_list is None:
        return set()
    if isinstance(ts_list, str):
        return {ts_list}
    return set(ts_list)

# Only the physically required row-level quantities are exported by default.
# The downstream conditional-analysis stage can derive spectra and scale-normalized
# higher-order moments from these rows without saving bulky first-pass summaries.
_DEFAULT_COMPACT_COEFF_COLUMNS = (
    'l_mag', 'l_ell', 'l_xi', 'l_lambda',
    'thetas', 'phis',
    'coi_mask', 'level_index', 'is_effective_level',
    'tau_equiv_seconds', 'frequency_hz', 'response_energy_integral',
    'polarity', 'local_polarity', 'polarity_used',
    'sig_c', 'sig_r', 'sig_c_ts', 'sig_r_ts',
    'compress_simple', 'compress_simple_V',
    'W_B_nT_mag', 'W_B_vel_mag', 'W_V_mag', 'W_Zp_mag', 'W_Zm_mag',
    'B_nT_par', 'B_nT_perp',
    'B_vel_par', 'B_vel_perp',
    'V_par', 'V_perp', 'V_xi', 'V_lambda',
    'Zp_par', 'Zp_perp', 'Zp_xi', 'Zp_lambda',
    'Zm_par', 'Zm_perp', 'Zm_xi', 'Zm_lambda',
)
_DEFAULT_BUCKET_ELL_IDENTIFIER = {
    'ell_all': 'l_mag',
    'ell_perp': 'l_lambda',
    'Ell_perp': 'l_xi',
    'ell_par': 'l_ell',
    'ell_par_rest': 'l_ell',
}
_DEFAULT_BUCKET_FLAG_COLUMNS: Dict[str, str] = {}


def _explicit_requested_keys(ts_list: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Return only explicitly requested extra coefficient columns.

    The coefficient-only export should stay lean by default. Extras are added only if
    the caller explicitly asks for them through ``ts_list``.
    """
    if ts_list is None:
        return []
    raw = [ts_list] if isinstance(ts_list, str) else list(ts_list)
    out: List[str] = []
    seen = set(_DEFAULT_COMPACT_COEFF_COLUMNS)
    for key in raw:
        key_s = str(key)
        if key_s not in seen:
            out.append(key_s)
            seen.add(key_s)
    return out


def _column_or_nan(column_data: Mapping[str, np.ndarray], key: str, n_rows: int) -> np.ndarray:
    arr = column_data.get(key)
    if arr is None:
        return np.full(n_rows, np.nan, dtype=np.float64)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full(n_rows, float(arr.item()), dtype=np.float64)
    if arr.shape[0] != n_rows:
        return np.full(n_rows, np.nan, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)

def _bucket_mask_from_angles(theta: np.ndarray, phi: np.ndarray, bucket: str, bucket_conditions: Optional[Mapping[str, Mapping[str, float]]] = None) -> np.ndarray:
    """Reconstruct directional bucket membership from saved local angles."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    finite_theta = np.isfinite(theta)
    finite_phi = finite_theta & np.isfinite(phi)
    conds = {} if bucket_conditions is None else bucket_conditions
    if bucket in ('ell_all', 'ell_overall'):
        return finite_theta
    cond = conds.get(bucket, {})
    if bucket == 'ell_perp':
        return finite_phi & (theta > float(cond.get('theta', np.nan))) & (phi > float(cond.get('phi', np.nan)))
    if bucket == 'Ell_perp':
        return finite_phi & (theta > float(cond.get('theta', np.nan))) & (phi < float(cond.get('phi', np.nan)))
    if bucket == 'ell_par':
        return finite_theta & (theta < float(cond.get('theta', np.nan)))
    if bucket == 'ell_par_rest':
        return finite_theta & (theta < float(cond.get('theta', np.nan)))
    raise KeyError(f'Unknown bucket {bucket!r}.')


def _build_minimal_compact_level_columns(
    level_data: LevelData,
    raw_context: Mapping[str, Any],
    level_index: int,
    theta_mask_perp: np.ndarray,
    theta_mask_disp: np.ndarray,
    theta_mask_par: np.ndarray,
    theta_mask_par_rest: np.ndarray,
    dt: float,
    ts_list: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, np.ndarray]:
    """Build the compact row-level coefficient payload for one level.

    This is the first-pass product that the downstream conditional-analysis notebook
    should consume. It deliberately stores only what is needed to reconstruct
    conditional spectra and scale-normalized higher-order moments:

    * the coefficient amplitudes or projected amplitudes themselves;
    * the local geometry needed to define directional buckets later;
    * the exact per-row normalization quantities ``tau_equiv_seconds`` and
      ``response_energy_integral``.

    The table is stored once, on the ``ell_all`` support. Directional buckets are not
    duplicated. They are reconstructed later from the saved local angles ``thetas`` and
    ``phis`` together with the stored bucket conditions.
    """
    n_rows = int(level_data.l_mag.shape[0])
    if n_rows == 0:
        return {key: np.empty((0,), dtype=np.float64) for key in _DEFAULT_COMPACT_COEFF_COLUMNS}

    e_l, e_xi, e_lambda, _basis_valid = _local_basis_from_background_and_perp(level_data.B_l, level_data.dB_perp_nT)

    v_perp_vec = _perp_vector_stacked(level_data.dV, level_data.B_l)
    zp_perp_vec = _perp_vector_stacked(level_data.dZp, level_data.B_l)
    zm_perp_vec = _perp_vector_stacked(level_data.dZm, level_data.B_l)

    sig_c_scale = level_data.align_angles_zpm.get('sig_c_scale') if level_data.align_angles_zpm else np.nan
    sig_r_scale = level_data.align_angles_vb.get('sig_r_scale') if level_data.align_angles_vb else np.nan
    sig_c_ts = level_data.align_angles_zpm.get('sig_c_ts') if level_data.align_angles_zpm else np.full(n_rows, np.nan)
    sig_r_ts = level_data.align_angles_vb.get('sig_r_ts') if level_data.align_angles_vb else np.full(n_rows, np.nan)

    tau_equiv_seconds = float(level_data.tau_equiv_samples) * float(dt)

    column_data: Dict[str, np.ndarray] = {
        'l_mag': np.asarray(level_data.l_mag, dtype=np.float64),
        'l_ell': np.asarray(level_data.l_ell, dtype=np.float64),
        'l_xi': np.asarray(level_data.l_xi, dtype=np.float64),
        'l_lambda': np.asarray(level_data.l_lambda, dtype=np.float64),
        'thetas': np.asarray(level_data.VBangle, dtype=np.float64),
        'phis': np.asarray(level_data.Phiangle, dtype=np.float64),
        'coi_mask': np.asarray(level_data.coi_mask, dtype=np.float64),
        'level_index': np.full(n_rows, float(level_index), dtype=np.float64),
        'is_effective_level': np.full(n_rows, 1.0 if level_data.is_effective else 0.0, dtype=np.float64),
        'tau_equiv_seconds': np.full(n_rows, tau_equiv_seconds, dtype=np.float64),
        'frequency_hz': np.full(n_rows, float(level_data.frequency_hz), dtype=np.float64),
        'response_energy_integral': np.full(n_rows, float(level_data.response_energy_integral), dtype=np.float64),
        'polarity': np.asarray(level_data.polarity, dtype=np.float64),
        'local_polarity': np.asarray(level_data.local_polarity, dtype=np.float64),
        'polarity_used': np.asarray(level_data.polarity_used, dtype=np.float64),
        'sig_c': np.full(n_rows, float(sig_c_scale) if np.isscalar(sig_c_scale) else np.nan, dtype=np.float64),
        'sig_r': np.full(n_rows, float(sig_r_scale) if np.isscalar(sig_r_scale) else np.nan, dtype=np.float64),
        'sig_c_ts': _coerce_samplewise_output(sig_c_ts, n_rows).astype(np.float64),
        'sig_r_ts': _coerce_samplewise_output(sig_r_ts, n_rows).astype(np.float64),
        'compress_simple': np.asarray(_parallel_energy_fraction_from_flucts(level_data.dB_parallel_nT, level_data.dB_nT), dtype=np.float64),
        'compress_simple_V': np.asarray(_parallel_energy_fraction_from_flucts(_perp_vector_stacked(level_data.dV, level_data.B_l, return_paral_comp=True)[1], level_data.dV), dtype=np.float64),
        'W_B_nT_mag': np.asarray(_estimate_vec_magnitude_stacked(level_data.dB_nT), dtype=np.float64),
        'W_B_vel_mag': np.asarray(_estimate_vec_magnitude_stacked(level_data.dVa), dtype=np.float64),
        'W_V_mag': np.asarray(_estimate_vec_magnitude_stacked(level_data.dV), dtype=np.float64),
        'W_Zp_mag': np.asarray(_estimate_vec_magnitude_stacked(level_data.dZp), dtype=np.float64),
        'W_Zm_mag': np.asarray(_estimate_vec_magnitude_stacked(level_data.dZm), dtype=np.float64),
        'B_nT_par': np.asarray(_abs_scalar_projection(level_data.dB_nT, e_l), dtype=np.float64),
        'B_nT_perp': np.asarray(_estimate_vec_magnitude_stacked(level_data.dB_perp_nT), dtype=np.float64),
        'B_vel_par': np.asarray(_abs_scalar_projection(level_data.dVa, e_l), dtype=np.float64),
        'B_vel_perp': np.asarray(_estimate_vec_magnitude_stacked(level_data.dVa_perp), dtype=np.float64),
        'V_par': np.asarray(_abs_scalar_projection(level_data.dV, e_l), dtype=np.float64),
        'V_perp': np.asarray(_estimate_vec_magnitude_stacked(v_perp_vec), dtype=np.float64),
        'V_xi': np.asarray(_abs_scalar_projection(level_data.dV, e_xi), dtype=np.float64),
        'V_lambda': np.asarray(_abs_scalar_projection(level_data.dV, e_lambda), dtype=np.float64),
        'Zp_par': np.asarray(_abs_scalar_projection(level_data.dZp, e_l), dtype=np.float64),
        'Zp_perp': np.asarray(_estimate_vec_magnitude_stacked(zp_perp_vec), dtype=np.float64),
        'Zp_xi': np.asarray(_abs_scalar_projection(level_data.dZp, e_xi), dtype=np.float64),
        'Zp_lambda': np.asarray(_abs_scalar_projection(level_data.dZp, e_lambda), dtype=np.float64),
        'Zm_par': np.asarray(_abs_scalar_projection(level_data.dZm, e_l), dtype=np.float64),
        'Zm_perp': np.asarray(_estimate_vec_magnitude_stacked(zm_perp_vec), dtype=np.float64),
        'Zm_xi': np.asarray(_abs_scalar_projection(level_data.dZm, e_xi), dtype=np.float64),
        'Zm_lambda': np.asarray(_abs_scalar_projection(level_data.dZm, e_lambda), dtype=np.float64),
    }

    extra_keys = _explicit_requested_keys(ts_list)
    if extra_keys:
        extra_payload = estimate_requested_quantities(level_data, raw_context, ts_list=extra_keys)
        for key in extra_keys:
            if key not in column_data and key in extra_payload:
                column_data[key] = np.asarray(_coerce_samplewise_output(extra_payload[key], n_rows), dtype=np.float64)

    return column_data


def _finalize_compact_store_from_matrices(
    bucket_matrices: Mapping[str, Sequence[np.ndarray]],
    column_order: Sequence[str],
    b_export_units: str,
    qorder: Sequence[float],
    bucket_ell_identifier: Optional[Mapping[str, str]] = None,
    bucket_selection_flag: Optional[Mapping[str, str]] = None,
    bucket_conditions: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Finalize the lean coefficient store from already assembled dense matrices."""
    buckets_out: Dict[str, Any] = {}
    for bucket, parts in bucket_matrices.items():
        valid_parts = [np.asarray(part, dtype=np.float64) for part in parts if np.asarray(part).size]
        if valid_parts:
            data = np.concatenate(valid_parts, axis=0)
        else:
            data = np.empty((0, len(column_order)), dtype=np.float64)
        buckets_out[bucket] = {'data': data, 'n_rows': int(data.shape[0])}
    return {
        'version': 'scwf_compact_coefficients_v3',
        'column_order': list(column_order),
        'column_units': _column_units_hint(column_order, b_export_units=b_export_units),
        'bucket_ell_identifier': dict(_DEFAULT_BUCKET_ELL_IDENTIFIER if bucket_ell_identifier is None else bucket_ell_identifier),
        'bucket_selection_flag': dict(_DEFAULT_BUCKET_FLAG_COLUMNS if bucket_selection_flag is None else bucket_selection_flag),
        'bucket_conditions': {k: dict(v) for k, v in ({} if bucket_conditions is None else bucket_conditions).items()},
        'qorder': np.asarray(qorder, dtype=float),
        'b_export_units': str(b_export_units),
        'buckets': buckets_out,
    }


def coefficient_store_to_dataframe(
    store: Mapping[str, Any],
    bucket: str = 'ell_all',
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a dataframe view of a compact coefficient store.

    Rev9 stores a single ``ell_all`` table by default. If a directional bucket is
    requested, the function automatically filters ``ell_all`` using the stored bucket
    membership flag rather than expecting a duplicated table.
    """
    buckets = store.get('buckets', {})
    if bucket in buckets:
        payload = buckets[bucket]
        column_order = list(store['column_order'])
        data = np.asarray(payload['data'], dtype=float)
    else:
        payload = buckets.get('ell_all')
        if payload is None:
            raise KeyError(f'Bucket {bucket!r} is not present in the compact coefficient store.')
        column_order = list(store['column_order'])
        data = np.asarray(payload['data'], dtype=float)
        if bucket not in ('ell_all', 'ell_overall'):
            flag_col = store.get('bucket_selection_flag', {}).get(bucket)
            if flag_col is not None and flag_col in column_order:
                mask = data[:, column_order.index(flag_col)] >= 0.5
            else:
                theta = data[:, column_order.index('thetas')]
                phi = data[:, column_order.index('phis')]
                mask = _bucket_mask_from_angles(theta, phi, bucket, store.get('bucket_conditions'))
            data = data[mask]
    if columns is None:
        return pd.DataFrame(data, columns=column_order)
    idx = [column_order.index(col) for col in columns]
    return pd.DataFrame(data[:, idx], columns=list(columns))


def reduce_compact_coefficient_store(
    store: Mapping[str, Any],
    bucket: str,
    value_key: str,
    qorder: Optional[Sequence[float]] = None,
    scale_bin_edges_di: Optional[Sequence[float]] = None,
    scale_key: Optional[str] = None,
    normalization: str = 'scale_normalized',
    constraints: Optional[Mapping[str, Union[float, Sequence[Optional[float]]]]] = None,
    absolute_value: bool = True,
    min_count: int = 1,
) -> Dict[str, Any]:
    """Reduce a compact coefficient store to pooled conditional moments.

    The function now understands the lean rev9 store layout, where rows are stored
    once on ``ell_all`` and directional buckets are recovered from per-row boolean
    membership flags.
    """
    buckets = store.get('buckets', {})
    effective_constraints: Dict[str, Union[float, Sequence[Optional[float]]]] = {} if constraints is None else dict(constraints)
    if bucket in buckets:
        payload = buckets[bucket]
    else:
        payload = buckets.get('ell_all')
        if payload is None:
            raise KeyError(f'Bucket {bucket!r} is not present in the compact coefficient store.')
        if bucket not in ('ell_all', 'ell_overall'):
            flag_col = store.get('bucket_selection_flag', {}).get(bucket)
            if flag_col is not None:
                effective_constraints.setdefault(flag_col, (0.5, None))
            else:
                bucket_conditions = store.get('bucket_conditions', {})
                cond = bucket_conditions.get(bucket, {})
                if bucket == 'ell_perp':
                    effective_constraints.setdefault('thetas', (float(cond.get('theta', np.nan)), None))
                    effective_constraints.setdefault('phis', (float(cond.get('phi', np.nan)), None))
                elif bucket == 'Ell_perp':
                    effective_constraints.setdefault('thetas', (float(cond.get('theta', np.nan)), None))
                    effective_constraints.setdefault('phis', (None, float(cond.get('phi', np.nan))))
                elif bucket in ('ell_par', 'ell_par_rest'):
                    effective_constraints.setdefault('thetas', (None, float(cond.get('theta', np.nan))))
                else:
                    raise KeyError(f'Bucket {bucket!r} is not present and cannot be reconstructed.')
    return _reduce_compact_coefficient_store_legacy(
        store={**store, 'buckets': {'__active__': payload}},
        bucket='__active__',
        value_key=value_key,
        qorder=qorder,
        scale_bin_edges_di=scale_bin_edges_di,
        scale_key=(store.get('bucket_ell_identifier', {}) or _DEFAULT_BUCKET_ELL_IDENTIFIER).get(bucket, scale_key) if scale_key is None else scale_key,
        normalization=normalization,
        constraints=effective_constraints,
        absolute_value=absolute_value,
        min_count=min_count,
    )


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
    scale_bin_edges_di: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Estimate the interval products for the SCWF pipeline.

    Rev9 keeps the full legacy path for reduced products, but introduces a strict,
    fast coefficient-only mode. When ``return_coefs`` is true, the function does *not*
    compute or return interval-level spectra or moments. It returns only the compact
    row store needed by the second-stage conditional analysis.
    """
    if not return_coefs:
        return _estimate_wavelet_interval_legacy(
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
            return_coefs=False,
            ts_list=ts_list,
            return_B_in_vel_units=return_B_in_vel_units,
            use_local_polarity=use_local_polarity,
            sc=sc,
            frame=frame,
            min_valid_fraction=min_valid_fraction,
            min_valid_count=min_valid_count,
            respect_effective_levels=respect_effective_levels,
            scale_bin_edges_di=scale_bin_edges_di,
        )

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

    bucket_matrices: Dict[str, List[np.ndarray]] = {'ell_all': []}
    column_order = list(_DEFAULT_COMPACT_COEFF_COLUMNS) + _explicit_requested_keys(ts_list)
    cond_perp = conditions['ell_perp']
    cond_disp = conditions['Ell_perp']
    cond_par = conditions['ell_par']
    cond_par_rest = conditions.get('ell_par_rest', cond_par)

    n_levels = len(results['levels'])
    all_thetas: Dict[str, np.ndarray] = {}
    all_phis: Dict[str, np.ndarray] = {}
    scale_axis = {
        'levels': np.asarray(results['meta']['levels'], dtype=int),
        'tau_equiv_samples': np.asarray(results['meta']['tau_equiv_samples'], dtype=float),
        'tau_equiv_seconds': np.asarray(results['meta']['tau_equiv_samples'], dtype=float) * float(dt),
        'scale_samples': np.asarray(results['meta']['scale_samples'], dtype=float),
        'scale_seconds': np.asarray(results['meta']['scale_seconds'], dtype=float),
        'frequency_hz': np.asarray(results['meta']['frequency_hz'], dtype=float),
        'period_s': np.asarray(results['meta']['period_s'], dtype=float),
        'bandwidth_hz': np.asarray(results['meta']['bandwidth_hz'], dtype=float),
        'response_energy_integral': np.asarray(results['meta']['response_energy_integral'], dtype=float),
        'effective_level_mask': np.asarray(results['meta'].get('effective_level_mask', np.ones(n_levels, dtype=bool)), dtype=bool),
        'n_valid': np.asarray(results['meta'].get('n_valid', np.full(n_levels, np.nan)), dtype=float),
        'valid_fraction': np.asarray(results['meta'].get('valid_fraction', np.full(n_levels, np.nan)), dtype=float),
        'di_mean': float(di),
    }

    for j, lvl in enumerate(results['levels']):
        all_thetas[str(j)] = lvl.VBangle
        all_phis[str(j)] = lvl.Phiangle
        if respect_effective_levels and not lvl.is_effective:
            continue

        finite_theta = np.isfinite(lvl.VBangle) & np.isfinite(lvl.l_mag) & np.isfinite(lvl.l_ell)
        finite_phi_lambda = finite_theta & np.isfinite(lvl.Phiangle) & np.isfinite(lvl.l_lambda)
        finite_phi_xi = finite_theta & np.isfinite(lvl.Phiangle) & np.isfinite(lvl.l_xi)
        mask_ell_perp = finite_phi_lambda & (lvl.VBangle > cond_perp['theta']) & (lvl.Phiangle > cond_perp['phi'])
        mask_Ell_perp = finite_phi_xi & (lvl.VBangle > cond_disp['theta']) & (lvl.Phiangle < cond_disp['phi'])
        mask_ell_par = finite_theta & (lvl.VBangle < cond_par['theta'])
        mask_ell_par_rest = finite_theta & (lvl.VBangle < cond_par_rest['theta'])
        idx_all = np.flatnonzero(np.isfinite(lvl.l_mag))
        if idx_all.size == 0:
            continue

        level_columns = _build_minimal_compact_level_columns(
            lvl,
            raw_context=results,
            level_index=j,
            theta_mask_perp=mask_ell_perp,
            theta_mask_disp=mask_Ell_perp,
            theta_mask_par=mask_ell_par,
            theta_mask_par_rest=mask_ell_par_rest,
            dt=float(dt),
            ts_list=ts_list,
        )
        mat = np.column_stack([_column_or_nan(level_columns, key, level_columns['l_mag'].shape[0])[idx_all] for key in column_order])
        bucket_matrices['ell_all'].append(mat)

    compact_coefficients = _finalize_compact_store_from_matrices(
        bucket_matrices=bucket_matrices,
        column_order=column_order,
        b_export_units=results['B_export_units'],
        qorder=q,
        bucket_conditions=conditions,
    )

    return {
        'flucts': None,
        'CompactCoefficients': compact_coefficients,
        'CoefficientStore': compact_coefficients,
        'ScaleAxis': scale_axis,
        'meta': results['meta'],
        'B_export_units': results['B_export_units'],
        'B_analysis_units': results['B_export_units'],
        'raw': None,
    }


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
    scale_bin_edges_di: Optional[Sequence[float]] = None,
):
    """Run the interval pipeline and write a trimmed output payload.

    Rev9 changes the contract of the first pass:

    * if ``return_flucs`` is true, the function saves only the compact coefficient
      store and the small amount of metadata needed by the downstream conditional
      reducer;
    * if ``return_flucs`` is false, the function saves the reduced products only,
      without bulky raw coefficient tables or legacy duplicates.
    """
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

        fname, align_name = build_output_names(
            consider_Vsc,
            strict_thresh,
            return_flucs,
            only_general,
            extra_conditions,
            theta_thresh_gen,
            phi_thresh_gen,
            thetas_phis_step,
            wname,
            file_name_root=file_name_root,
            method_token=method_token,
        )
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
                    V_sc_df = pd.DataFrame({c: np.asarray(ephem[c]) for c in ['sc_vel_r', 'sc_vel_t', 'sc_vel_n']}, index=B.index)
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
            scale_bin_edges_di=scale_bin_edges_di,
        )

        common_meta = {
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
            'package_name': _PACKAGE_NAME,
            'method_token': method_token,
            'output_subdir': resolved_output_subdir,
            'file_name_root': file_name_root,
            'wavelet_meta': interval_out['meta'],
            'scale_bin_edges_di': None if scale_bin_edges_di is None else list(np.asarray(scale_bin_edges_di, dtype=float)),
            'note': 'If return_flucs/return_coefs is true, only the compact coefficient store is saved. Conditional spectra and scale-normalized higher-order moments must then be computed in a second pass from CoefficientStore. The row store is written once on ell_all, with all required local scales, local angles, and normalization columns; directional buckets are reconstructed later from the saved angles and the stored bucket conditions.',
        }

        if return_flucs:
            keep_payload = {
                'di': di,
                'Vsw': Vsw,
                'Vsw_norm': Vsw_norm,
                'CoefficientStore': interval_out['CoefficientStore'],
                'CompactCoefficients': interval_out['CompactCoefficients'],
                'ScaleAxis': interval_out['ScaleAxis'],
                'B_analysis_units': interval_out['B_analysis_units'],
                'meta': common_meta,
            }
        else:
            keep_payload = {
                'di': di,
                'Vsw': Vsw,
                'Vsw_norm': Vsw_norm,
                'ell_di': interval_out['ell_di'],
                'Sfuncs': interval_out['Sfuncs'],
                'WaveletMoments': interval_out['WaveletMoments'],
                'ProjectedWaveletMoments': interval_out['ProjectedWaveletMoments'],
                'ProjectedScaleNormalizedWaveletMoments': interval_out['ProjectedScaleNormalizedWaveletMoments'],
                'Spectra': interval_out['Spectra'],
                'ProjectedSpectra': interval_out['ProjectedSpectra'],
                'BucketScaleStats': interval_out['BucketScaleStats'],
                'ScaleAxis': interval_out['ScaleAxis'],
                'B_analysis_units': interval_out['B_analysis_units'],
                'meta': common_meta,
            }
        func.savepickle(keep_payload, outdir, fname)
        if estimate_alignment_angle and align_name is not None and not return_flucs:
            func.savepickle(interval_out.get('overall_align_angles'), outdir, align_name)
        return str(Path(outdir) / fname)
    except Exception:
        traceback.print_exc()
        return None


# Notebook-facing compatibility alias
run_logscale_filterbank_analysis = run_filterbank_interval_analysis
