from __future__ import annotations

"""
MODWT pipeline rewritten to follow the 5pt structure-function logic while
calling ``general_functions`` and ``TurbPy`` through the same path-setup
pattern used by the mature 5pt pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import glob
import os
import pickle
import traceback
import warnings

import numpy as np
import pandas as pd
import pywt
from scipy import constants

try:
    from .path_setup import ensure_project_paths
except ImportError:
    from path_setup import ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
import general_functions as func
import TurbPy as turb

try:
    from .backend import modwt, modwtmra
    from .shared_logic import (
        _has_columns,
        infer_frame_from_data,
        _background_polarity,
        _mag_from_frame,
        est_alignment_angles,
        fast_unit_vec,
        mag_of_ell_projections_and_angles,
        structure_functions_3D,
        vars_2_estimate,
        quants_2_estimate,
        save_flucs as save_flucs_shared,
    )
except Exception:
    from backend import modwt, modwtmra
    from shared_logic import (
        _has_columns,
        infer_frame_from_data,
        _background_polarity,
        _mag_from_frame,
        est_alignment_angles,
        fast_unit_vec,
        mag_of_ell_projections_and_angles,
        structure_functions_3D,
        vars_2_estimate,
        quants_2_estimate,
        save_flucs as save_flucs_shared,
    )

mu0 = constants.mu_0
m_p = constants.m_p


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _get_component_keys(df: pd.DataFrame, candidate_sets: Sequence[Sequence[str]]) -> List[str]:
    cols = set(df.columns)
    for keys in candidate_sets:
        if all(k in cols for k in keys):
            return list(keys)
    raise KeyError("Could not infer component keys from dataframe columns.")


def _to_frame(x: Union[pd.Series, pd.DataFrame], name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name if x.name is None else x.name)
    if isinstance(x, pd.DataFrame):
        return x.copy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        return pd.DataFrame({name: arr})
    raise TypeError("Expected pandas Series/DataFrame or 1D array-like.")


def _rolling_mean_centered(x: Union[pd.Series, pd.DataFrame], window: str = "1min", samples: int = 21):
    if isinstance(x.index, pd.DatetimeIndex):
        return x.rolling(window, center=True).mean()
    return x.rolling(samples, center=True, min_periods=1).mean()


def _normalize_requested_quants(ts_list: Optional[Union[str, Sequence[str]]]) -> set:
    requested = vars_2_estimate(ts_list=ts_list)
    if requested is None:
        quants = set()
    elif isinstance(requested, str):
        quants = {requested}
    else:
        quants = set(requested)
    if ts_list is None:
        return quants
    if isinstance(ts_list, str):
        quants.add(ts_list)
    else:
        quants.update(ts_list)
    return quants


def _make_vec_df(arr: np.ndarray, keys: Sequence[str], index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(arr, dtype=float), index=index, columns=list(keys))


def _store_component_family(
    variables: MutableMapping[str, Any],
    quants: set,
    arr: np.ndarray,
    canonical_prefix: str,
    comp_map: Mapping[str, int],
    family_aliases: Sequence[str] = (),
    canonical_component_alias_prefixes: Sequence[str] = (),
    legacy_component_output_prefixes: Sequence[str] = (),
    bare_legacy_output: bool = False,
) -> None:
    family_requested = any(alias in quants for alias in family_aliases)
    for comp, idx in comp_map.items():
        canonical_key = f"{canonical_prefix}{comp}"
        canonical_component_requested = canonical_key in quants or any(
            f"{prefix}{comp}" in quants for prefix in canonical_component_alias_prefixes
        )
        if family_requested or canonical_component_requested:
            variables[canonical_key] = arr[:, idx]
        if bare_legacy_output and comp in quants:
            variables[comp] = arr[:, idx]
        for prefix in legacy_component_output_prefixes:
            legacy_key = f"{prefix}{comp}"
            if legacy_key in quants:
                variables[legacy_key] = arr[:, idx]


def _estimate_pvi_from_flucts(vec: np.ndarray, window: int = 101) -> np.ndarray:
    amp2 = np.nansum(vec**2, axis=1)
    s = pd.Series(np.asarray(amp2, dtype=float))
    denom = np.sqrt(s.rolling(window, center=True, min_periods=max(3, window // 10)).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        pvi = np.sqrt(amp2) / denom.to_numpy()
    return pvi


@dataclass
class LevelResult:
    level: int
    scale_samples: float
    tau_equiv_samples: float
    tau_equiv_samples_int: int
    band_low_samples: float
    band_high_samples: float
    boundary_width_samples: int
    period_s: float
    frequency_hz: float
    valid_mask: np.ndarray
    n_valid: int
    valid_fraction: float
    is_effective: bool
    needed_index: pd.Index
    keep_turb_amp: Dict[str, Any]
    B_l: np.ndarray
    V_l: np.ndarray
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
    l_mag: np.ndarray
    l_ell: np.ndarray
    l_xi: np.ndarray
    l_lambda: np.ndarray
    VBangle: np.ndarray
    Phiangle: np.ndarray
    polarity: np.ndarray
    local_polarity: np.ndarray
    kinet_normal: np.ndarray
    align_angles_vb: Dict[str, Any]
    align_angles_zpm: Dict[str, Any]


# -----------------------------------------------------------------------------
# MODWT decomposition helpers
# -----------------------------------------------------------------------------


def _normalize_wavelet_name(wname: str) -> str:
    """Map common MODWT legacy names to PyWavelets names."""
    name = str(wname).strip()
    lower = name.lower()
    legacy = {
        "la8": "sym4",
        "la10": "sym5",
        "la12": "sym6",
        "la14": "sym7",
        "la16": "sym8",
        "la18": "sym9",
        "la20": "sym10",
    }
    return legacy.get(lower, name)


def _modwt_level_limits(n_samples: int, wname: str) -> Dict[str, int]:
    wavelet = pywt.Wavelet(_normalize_wavelet_name(str(wname)))
    filt_len = int(wavelet.dec_len)
    n_safe = max(2, int(n_samples))
    max_transform = max(1, int(np.floor(np.log2(n_safe))))
    if filt_len <= 1:
        max_unbiased = max_transform
    else:
        max_unbiased = int(np.floor(np.log2((float(n_samples) / float(filt_len - 1)) + 1.0)))
        max_unbiased = max(1, min(max_transform, max_unbiased))
    return {
        "max_transform": int(max_transform),
        "max_unbiased": int(max_unbiased),
        "filter_length": int(filt_len),
    }


def _recommended_level(n_samples: int, wname: str, level: Optional[int], level_mode: str = "recommended") -> int:
    if level is not None:
        return max(1, int(level))

    mode = str(level_mode).strip().lower()
    limits = _modwt_level_limits(n_samples, wname)
    max_transform = int(limits["max_transform"])
    max_unbiased = int(limits["max_unbiased"])

    if mode in ("recommended", "safe", "conservative", "modwt", "unbiased"):
        return max_unbiased
    if mode in ("legacy", "max", "aggressive", "all", "transform"):
        return max_transform

    raise ValueError(
        f"Unknown level_mode={level_mode!r}. Use 'recommended'/'unbiased' or 'legacy'/'transform'."
    )


def _dyadic_scale_metadata(wname: str, dt: float, n_levels: int) -> Dict[str, np.ndarray]:
    wavelet = pywt.Wavelet(_normalize_wavelet_name(str(wname)))
    levels = np.arange(1, n_levels + 1, dtype=int)
    band_low = 2.0 ** (levels - 1)
    band_high = 2.0 ** levels
    tau_equiv = np.sqrt(band_low * band_high)
    tau_int = np.maximum(1, np.rint(tau_equiv).astype(int))
    period = tau_equiv * float(dt)
    freq = 1.0 / np.where(period > 0.0, period, np.nan)
    support = (2.0 ** levels - 1.0) * float(wavelet.dec_len - 1) + 1.0
    boundary = np.ceil(0.5 * np.maximum(0.0, support - 1.0)).astype(int)
    return {
        "levels": levels,
        "scale_samples": tau_equiv.copy(),
        "tau_equiv_samples": tau_equiv,
        "tau_equiv_samples_int": tau_int,
        "band_low_samples": band_low,
        "band_high_samples": band_high,
        "support_samples": support,
        "boundary_width_samples": boundary,
        "frequency_hz": freq,
        "period_s": period,
        "wavelet_filter_length": int(wavelet.dec_len),
    }


def _valid_mask_from_boundary(n: int, boundary_width: int) -> np.ndarray:
    mask = np.ones(int(n), dtype=bool)
    bw = int(max(0, boundary_width))
    if bw == 0:
        return mask
    if 2 * bw >= int(n):
        mask[:] = False
        return mask
    mask[:bw] = False
    mask[-bw:] = False
    return mask


def _valid_masks_from_boundaries(n: int, boundary_widths: Sequence[int]) -> np.ndarray:
    return np.vstack([_valid_mask_from_boundary(n, int(bw)) for bw in boundary_widths])


def _mask_with_valid(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if out.ndim == 1:
        out[~valid_mask] = np.nan
    else:
        out[~valid_mask, ...] = np.nan
    return out


def _mask_with_valid_stacked(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    if out.ndim == 2:
        out[~mask] = np.nan
    elif out.ndim == 3:
        out[~mask, :] = np.nan
    else:
        raise ValueError("Expected stacked array with ndim 2 or 3.")
    return out


def _estimate_vec_magnitude_stacked(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(np.asarray(arr, dtype=float) ** 2, axis=-1))


def _nanmean_no_warn(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    count = np.sum(np.isfinite(arr), axis=axis)
    total = np.nansum(arr, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = total / count
    out = np.asarray(out, dtype=float)
    out[count == 0] = np.nan
    return out


def _fast_unit_vec_stacked(arr: np.ndarray) -> np.ndarray:
    mag = _estimate_vec_magnitude_stacked(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = arr / mag[..., None]
    return out


def _perp_vector_stacked(a: np.ndarray, b: np.ndarray, return_paral_comp: bool = False):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    bmag2 = np.nansum(b * b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = (np.nansum(a * b, axis=-1) / bmag2)[..., None] * b
    par = np.where(np.isfinite(proj), proj, 0.0)
    perp = a - par
    if return_paral_comp:
        return perp, par
    return perp


def _angle_between_vectors_stacked(a: np.ndarray, b: np.ndarray, restrict_2_90: bool = False) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    amag = _estimate_vec_magnitude_stacked(a)
    bmag = _estimate_vec_magnitude_stacked(b)
    denom = amag * bmag
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.nansum(a * b, axis=-1) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    if restrict_2_90:
        ang = np.where(ang > 90.0, 180.0 - ang, ang)
    return ang


def _mag_of_ell_projections_and_angles_stacked(
    l_vector: np.ndarray,
    B_l_vector: np.ndarray,
    db_perp_vector: np.ndarray,
    est_proj_ells: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B_hat = _fast_unit_vec_stacked(B_l_vector)
    db_perp_hat = _fast_unit_vec_stacked(db_perp_vector)

    if est_proj_ells:
        b_perp_hat = np.cross(B_hat, db_perp_hat, axis=-1)
        l_ell = np.abs(np.nansum(l_vector * B_hat, axis=-1))
        l_xi = np.abs(np.nansum(l_vector * db_perp_hat, axis=-1))
        l_lambda = np.abs(np.nansum(l_vector * b_perp_hat, axis=-1))
    else:
        shape = np.shape(l_vector)[:-1]
        l_ell = np.full(shape, np.nan)
        l_xi = np.full(shape, np.nan)
        l_lambda = np.full(shape, np.nan)

    l_perp = _perp_vector_stacked(l_vector, B_hat)
    VBangle = _angle_between_vectors_stacked(l_vector, B_hat, restrict_2_90=True)
    Phiangle = _angle_between_vectors_stacked(l_perp, db_perp_hat, restrict_2_90=True)
    return l_ell, l_xi, l_lambda, VBangle, Phiangle


def _background_polarity_stacked(
    B: pd.DataFrame,
    B_l: np.ndarray,
    needed_index: pd.Index,
    sc: Optional[str] = None,
    frame: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    frame_used = infer_frame_from_data(B, sc=sc, frame=frame)

    if frame_used == "RTN" and _has_columns(B, "Br"):
        bg_series = B["Br"]
        local_component = B_l[:, :, 0]
        sign_factor = -1.0 if sc in ("SOLO", "PSP") else 1.0
    elif frame_used == "GSE" and _has_columns(B, "Bx"):
        bg_series = B["Bx"]
        local_component = B_l[:, :, 0]
        sign_factor = 1.0
    elif _has_columns(B, "Bz"):
        bg_series = B["Bz"]
        local_component = B_l[:, :, 2]
        sign_factor = 1.0
    else:
        bg_series = B.iloc[:, 0]
        local_component = B_l[:, :, 0]
        sign_factor = 1.0

    bg_smooth = _rolling_mean_centered(bg_series.to_frame("bg"))
    polarity_1d = sign_factor * np.sign(func.newindex(bg_smooth, needed_index).values.ravel())
    polarity = np.broadcast_to(polarity_1d[None, :], local_component.shape).astype(float)
    local_polarity = sign_factor * np.sign(local_component)
    return polarity, local_polarity, frame_used


def _alignment_stats_stacked(xvec: np.ndarray, yvec: np.ndarray, est_sigma_c: bool = False) -> Dict[str, Any]:
    numer = np.sqrt(np.nansum(np.cross(xvec, yvec, axis=-1) ** 2, axis=-1))
    numer_cos = np.nansum(xvec * yvec, axis=-1)
    xmag = _estimate_vec_magnitude_stacked(xvec)
    ymag = _estimate_vec_magnitude_stacked(yvec)
    denom = xmag * ymag

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_ts = (xmag**2 - ymag**2) / (xmag**2 + ymag**2)
        if est_sigma_c:
            sigma_mean = _nanmean_no_warn(xmag**2 - ymag**2, axis=1) / _nanmean_no_warn(xmag**2 + ymag**2, axis=1)
        else:
            sigma_mean = (_nanmean_no_warn(xmag**2, axis=1) - _nanmean_no_warn(ymag**2, axis=1)) / (_nanmean_no_warn(xmag**2, axis=1) + _nanmean_no_warn(ymag**2, axis=1))
        reg = _nanmean_no_warn(numer / denom, axis=1)
        polar = _nanmean_no_warn(numer, axis=1) / _nanmean_no_warn(denom, axis=1)
    counts = np.sum(np.isfinite(numer), axis=1).astype(int)
    weighted = np.full(xvec.shape[0], np.nan)

    return {
        "sigma_ts": sigma_ts,
        "sigma_mean": sigma_mean,
        "sigma_median": np.full(xvec.shape[0], np.nan),
        "sins_num": numer,
        "cos_num": numer_cos,
        "sins_den": denom,
        "x_mag": xmag,
        "y_mag": ymag,
        "reg": reg,
        "polar": polar,
        "weighted": weighted,
        "counts": counts,
    }


def _modwt_component_details_and_background(x: np.ndarray, wname: str, n_levels: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=float)
    coeffs = modwt(arr, _normalize_wavelet_name(str(wname)), int(n_levels))
    mra = modwtmra(coeffs, _normalize_wavelet_name(str(wname)))
    details = np.asarray(mra[:-1], dtype=float)
    smooth = np.asarray(mra[-1], dtype=float)

    approx = np.empty_like(details)
    running = smooth.copy()
    for j in range(n_levels - 1, -1, -1):
        approx[j] = running
        running = running + details[j]
    return approx, details


def estimate_coeffs_background_flucs_MODWT(
    x: Union[np.ndarray, pd.DataFrame],
    wname: str,
    level: Optional[int] = None,
    level_mode: str = "recommended",
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Input x must be 1D or 2D with shape (N, C).")

    n_samples, n_comp = arr.shape
    limits = _modwt_level_limits(n_samples, wname)
    n_levels = _recommended_level(n_samples, wname, level, level_mode=level_mode)

    approx_all, detail_all = _modwt_component_details_and_background(arr, wname, n_levels)

    meta = _dyadic_scale_metadata(wname, dt=float(dt), n_levels=n_levels)
    meta.update({
        "J0": int(n_levels),
        "wname_used": _normalize_wavelet_name(str(wname)),
        "wname_requested": str(wname),
        "n_samples": int(n_samples),
        "n_components": int(n_comp),
        "level_mode_used": str(level_mode),
        "level_max_transform": int(limits["max_transform"]),
        "level_max_unbiased": int(limits["max_unbiased"]),
        "wavelet_filter_length": int(limits["filter_length"]),
        "background_definition": "A_j = S_J + sum_{k>j} D_k",
        "detail_definition": "D_j from MODWT multiresolution reconstruction",
    })
    return approx_all, detail_all, meta


def estimate_approxs(
    flucs: Mapping[str, np.ndarray],
    comp_cols: Sequence[str],
    backgrounds: Mapping[str, np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    comps = list(comp_cols)
    n_levels = int(np.asarray(flucs[comps[0]]).shape[0])
    approx_by_level: List[np.ndarray] = []
    detail_by_level: List[np.ndarray] = []

    for j in range(n_levels):
        approx_by_level.append(np.column_stack([np.asarray(backgrounds[c])[j] for c in comps]))
        detail_by_level.append(np.column_stack([np.asarray(flucs[c])[j] for c in comps]))
    return approx_by_level, detail_by_level


# -----------------------------------------------------------------------------
# Core MODWT local-scale analysis
# -----------------------------------------------------------------------------
def modwt_local_structure_function(
    B: pd.DataFrame,
    V: pd.DataFrame,
    V_sc_vel_removed: Optional[pd.DataFrame],
    Np: Union[pd.Series, pd.DataFrame],
    dt: Optional[float],
    wname: str,
    level: Optional[int] = None,
    level_mode: str = "recommended",
    estimate_alignment_angle: bool = False,
    return_mag_align_correl: bool = False,
    return_B_in_vel_units: bool = False,
    use_local_polarity: bool = True,
    est_proj_ells: bool = True,
    sc: Optional[str] = None,
    frame: Optional[str] = None,
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
) -> Dict[str, Any]:
    del return_mag_align_correl

    if V_sc_vel_removed is None:
        V_sc_vel_removed = V

    B = B.copy()
    interp_method = "time" if isinstance(B.index, pd.DatetimeIndex) else "linear"
    V = func.newindex(V.copy(), B.index).interpolate(method=interp_method)
    V_sc_vel_removed = func.newindex(V_sc_vel_removed.copy(), B.index).interpolate(method=interp_method)
    Np_df = _to_frame(func.newindex(_to_frame(Np, "np"), B.index), "np").interpolate(method=interp_method)

    dt = float(func.find_cadence(B)) if dt is None else float(dt)

    frame = infer_frame_from_data(B, sc=sc, frame=frame)
    b_keys = _get_component_keys(B, (("Br", "Bt", "Bn"), ("Bx", "By", "Bz")))
    v_keys = _get_component_keys(V, (("Vr", "Vt", "Vn"), ("Vx", "Vy", "Vz")))

    B_comp = B.loc[:, b_keys].copy()
    V_comp = V.loc[:, v_keys].copy()
    V_back = V_sc_vel_removed.loc[:, v_keys].copy()

    N_roll = _rolling_mean_centered(Np_df)
    N_roll_vals = np.clip(N_roll.to_numpy(dtype=float).ravel(), 1.0e-12, None)
    di_arr_full = (228.0 / np.sqrt(N_roll_vals)).astype(float)
    kinet_normal = (1.0e-15 / np.sqrt(mu0 * N_roll_vals * m_p)).astype(float)
    Va = B_comp.multiply(kinet_normal, axis=0).interpolate(method=interp_method)
    normal_flag = "B_in_vel_units" if return_B_in_vel_units else "B_in_nT_units"

    ApprB3, DetB3, meta = estimate_coeffs_background_flucs_MODWT(
        B_comp.to_numpy(dtype=float),
        wname,
        level=level,
        level_mode=level_mode,
        dt=dt,
    )
    target_level = int(meta["J0"])
    ApprV3, _, _ = estimate_coeffs_background_flucs_MODWT(
        V_back.to_numpy(dtype=float),
        wname,
        level=target_level,
        level_mode=level_mode,
        dt=dt,
    )
    _, DetVraw3, _ = estimate_coeffs_background_flucs_MODWT(
        V_comp.to_numpy(dtype=float),
        wname,
        level=target_level,
        level_mode=level_mode,
        dt=dt,
    )
    _, DetVa3, _ = estimate_coeffs_background_flucs_MODWT(
        Va.to_numpy(dtype=float),
        wname,
        level=target_level,
        level_mode=level_mode,
        dt=dt,
    )
    _, DetN3, _ = estimate_coeffs_background_flucs_MODWT(
        Np_df.to_numpy(dtype=float),
        wname,
        level=target_level,
        level_mode=level_mode,
        dt=dt,
    )

    n_levels = int(meta["J0"])
    n_samples = len(B.index)
    valid_masks = _valid_masks_from_boundaries(n_samples, meta["boundary_width_samples"])

    dB_perp_nT_raw, dB_parallel_nT_raw = _perp_vector_stacked(DetB3, ApprB3, return_paral_comp=True)
    dVa_perp_raw, dVa_parallel_raw = _perp_vector_stacked(DetVa3, ApprB3, return_paral_comp=True)

    tau_scale = np.asarray(meta["tau_equiv_samples"], dtype=float)[:, None, None] * dt
    l_vec_raw = ApprV3 * tau_scale
    l_ell_raw, l_xi_raw, l_lambda_raw, VBangle_raw, Phiangle_raw = _mag_of_ell_projections_and_angles_stacked(
        l_vec_raw,
        ApprB3,
        dB_perp_nT_raw,
        est_proj_ells=est_proj_ells,
    )
    di_stack = di_arr_full[None, :]
    l_ell_raw = l_ell_raw / di_stack
    l_xi_raw = l_xi_raw / di_stack
    l_lambda_raw = l_lambda_raw / di_stack
    l_mag_raw = _estimate_vec_magnitude_stacked(l_vec_raw) / di_stack

    polarity_raw, local_polarity_raw, _ = _background_polarity_stacked(B_comp, ApprB3, B.index, sc=sc, frame=frame)
    sign_back = local_polarity_raw if use_local_polarity else polarity_raw
    dZp_raw = DetVraw3 + sign_back[:, :, None] * DetVa3
    dZm_raw = DetVraw3 - sign_back[:, :, None] * DetVa3

    B_l = _mask_with_valid_stacked(ApprB3, valid_masks)
    V_l = _mask_with_valid_stacked(ApprV3, valid_masks)
    dB_nT = _mask_with_valid_stacked(DetB3, valid_masks)
    dV = _mask_with_valid_stacked(DetVraw3, valid_masks)
    dVa = _mask_with_valid_stacked(DetVa3, valid_masks)
    dN = _mask_with_valid_stacked(DetN3, valid_masks)
    dB_perp_nT = _mask_with_valid_stacked(dB_perp_nT_raw, valid_masks)
    dB_parallel_nT = _mask_with_valid_stacked(dB_parallel_nT_raw, valid_masks)
    dVa_perp = _mask_with_valid_stacked(dVa_perp_raw, valid_masks)
    dVa_parallel = _mask_with_valid_stacked(dVa_parallel_raw, valid_masks)
    dZp = _mask_with_valid_stacked(dZp_raw, valid_masks)
    dZm = _mask_with_valid_stacked(dZm_raw, valid_masks)
    l_mag = _mask_with_valid_stacked(l_mag_raw, valid_masks)
    l_ell = _mask_with_valid_stacked(l_ell_raw, valid_masks)
    l_xi = _mask_with_valid_stacked(l_xi_raw, valid_masks)
    l_lambda = _mask_with_valid_stacked(l_lambda_raw, valid_masks)
    VBangle = _mask_with_valid_stacked(VBangle_raw, valid_masks)
    Phiangle = _mask_with_valid_stacked(Phiangle_raw, valid_masks)
    polarity = _mask_with_valid_stacked(polarity_raw, valid_masks)
    local_polarity = _mask_with_valid_stacked(local_polarity_raw, valid_masks)
    kinet_level = _mask_with_valid_stacked(np.broadcast_to(kinet_normal[None, :], (n_levels, n_samples)), valid_masks)

    if return_B_in_vel_units:
        dB_out = dVa
        dB_perp_out = dVa_perp
        dB_parallel_out = dVa_parallel
    else:
        dB_out = dB_nT
        dB_perp_out = dB_perp_nT
        dB_parallel_out = dB_parallel_nT

    du_perp = _mask_with_valid_stacked(_perp_vector_stacked(DetVraw3, ApprB3), valid_masks)
    dzp_perp = _mask_with_valid_stacked(_perp_vector_stacked(dZp_raw, ApprB3), valid_masks)
    dzm_perp = _mask_with_valid_stacked(_perp_vector_stacked(dZm_raw, ApprB3), valid_masks)

    align_vb_all: Optional[Dict[str, Any]] = None
    align_zpm_all: Optional[Dict[str, Any]] = None
    if estimate_alignment_angle:
        align_vb_all = _alignment_stats_stacked(du_perp, dVa_perp, est_sigma_c=False)
        align_zpm_all = _alignment_stats_stacked(dzp_perp, dzm_perp, est_sigma_c=True)

    n_valid = np.sum(valid_masks, axis=1).astype(int)
    valid_fraction = n_valid.astype(float) / float(max(1, n_samples))
    effective_level_mask = (n_valid >= int(max(1, min_valid_count))) & (valid_fraction >= float(min_valid_fraction))
    first_bad = np.where(~effective_level_mask)[0]
    J_effective = int(first_bad[0]) if first_bad.size else int(n_levels)

    level_results: List[LevelResult] = []
    for j in range(n_levels):
        keep_turb_amp = {
            "dva_perp": dVa_perp[j],
            "du_perp": du_perp[j],
            "dzp_perp": dzp_perp[j],
            "dzm_perp": dzm_perp[j],
            "dB_nT": dB_nT[j],
            "dB_perp_amp_nT": func.estimate_vec_magnitude(dB_perp_nT[j]),
            "dB_parallel_amp_nT": func.estimate_vec_magnitude(dB_parallel_nT[j]),
            "B_l": B_l[j],
        }

        align_angles_vb: Dict[str, Any] = {}
        align_angles_zpm: Dict[str, Any] = {}
        if estimate_alignment_angle and align_vb_all is not None and align_zpm_all is not None:
            align_angles_vb = {
                "sig_r_ts": align_vb_all["sigma_ts"][j],
                "sig_r_mean": align_vb_all["sigma_mean"][j],
                "sig_r_median": align_vb_all["sigma_median"][j],
                "sins_ub_num": align_vb_all["sins_num"][j],
                "cos_ub_num": align_vb_all["cos_num"][j],
                "sins_ub_den": align_vb_all["sins_den"][j],
                "v_mag": align_vb_all["x_mag"][j],
                "va_mag": align_vb_all["y_mag"][j],
                "reg_angle": align_vb_all["reg"][j],
                "polar_inter_angle": align_vb_all["polar"][j],
                "weighted_angle": align_vb_all["weighted"][j],
                "counts": int(align_vb_all["counts"][j]),
            }
            align_angles_zpm = {
                "sig_c_ts": align_zpm_all["sigma_ts"][j],
                "sig_c_mean": align_zpm_all["sigma_mean"][j],
                "sig_c_median": align_zpm_all["sigma_median"][j],
                "sins_zp_num": align_zpm_all["sins_num"][j],
                "cos_zp_num": align_zpm_all["cos_num"][j],
                "sins_zp_den": align_zpm_all["sins_den"][j],
                "zp_mag": align_zpm_all["x_mag"][j],
                "zm_mag": align_zpm_all["y_mag"][j],
                "reg_angle": align_zpm_all["reg"][j],
                "polar_inter_angle": align_zpm_all["polar"][j],
                "weighted_angle": align_zpm_all["weighted"][j],
                "counts": int(align_zpm_all["counts"][j]),
            }

        level_results.append(
            LevelResult(
                level=int(meta["levels"][j]),
                scale_samples=float(meta["tau_equiv_samples"][j]),
                tau_equiv_samples=float(meta["tau_equiv_samples"][j]),
                tau_equiv_samples_int=int(meta["tau_equiv_samples_int"][j]),
                band_low_samples=float(meta["band_low_samples"][j]),
                band_high_samples=float(meta["band_high_samples"][j]),
                boundary_width_samples=int(meta["boundary_width_samples"][j]),
                period_s=float(meta["period_s"][j]),
                frequency_hz=float(meta["frequency_hz"][j]),
                valid_mask=valid_masks[j].copy(),
                n_valid=int(n_valid[j]),
                valid_fraction=float(valid_fraction[j]),
                is_effective=bool(effective_level_mask[j]),
                needed_index=B.index,
                keep_turb_amp=keep_turb_amp,
                B_l=B_l[j],
                V_l=V_l[j],
                dB=dB_out[j],
                dB_nT=dB_nT[j],
                dV=dV[j],
                dVa=dVa[j],
                dN=dN[j],
                dB_perp=dB_perp_out[j],
                dB_parallel=dB_parallel_out[j],
                dB_perp_nT=dB_perp_nT[j],
                dB_parallel_nT=dB_parallel_nT[j],
                dVa_perp=dVa_perp[j],
                dVa_parallel=dVa_parallel[j],
                dZp=dZp[j],
                dZm=dZm[j],
                l_mag=l_mag[j],
                l_ell=l_ell[j],
                l_xi=l_xi[j],
                l_lambda=l_lambda[j],
                VBangle=VBangle[j],
                Phiangle=Phiangle[j],
                polarity=polarity[j],
                local_polarity=local_polarity[j],
                kinet_normal=kinet_level[j],
                align_angles_vb=align_angles_vb,
                align_angles_zpm=align_angles_zpm,
            )
        )

    meta = dict(meta)
    meta["n_valid"] = n_valid
    meta["valid_fraction"] = valid_fraction
    meta["effective_level_mask"] = effective_level_mask
    meta["J_effective"] = J_effective

    return {
        "index": B.index,
        "dt": dt,
        "di_mean": float(np.nanmean(228.0 / np.sqrt(np.clip(Np_df.to_numpy(dtype=float), 1e-12, None)))),
        "Vsw_mean": float(np.nanmean(func.estimate_vec_magnitude(V_back.to_numpy(dtype=float)))),
        "frame": frame,
        "b_keys": b_keys,
        "v_keys": v_keys,
        "Np": Np_df,
        "B": B_comp,
        "V": V_comp,
        "V_sc_vel_removed": V_back,
        "Va": Va,
        "return_B_in_vel_units": bool(return_B_in_vel_units),
        "B_flag": normal_flag,
        "meta": meta,
        "levels": level_results,
        "effective_level_mask": effective_level_mask,
        "J_effective": J_effective,
    }


# -----------------------------------------------------------------------------
# MODWT quantity estimator, mirroring the 5-point quants_2_estimate logic
# -----------------------------------------------------------------------------


def modwt_quants_2_estimate(
    level_result: LevelResult,
    raw_context: Mapping[str, Any],
    ts_list: Optional[Union[str, Sequence[str]]] = None,
    av_hours: Optional[float] = None,
) -> Dict[str, Any]:
    if av_hours is None:
        av_hours = 1.0 / 120.0
    return quants_2_estimate(
        level_result.l_ell,
        level_result.l_lambda,
        level_result.l_xi,
        level_result.l_mag,
        level_result.V_l,
        level_result.B_l,
        level_result.local_polarity,
        level_result.dB,
        level_result.dB_perp,
        level_result.dB_parallel,
        level_result.dV,
        level_result.dZp,
        level_result.dZm,
        level_result.dN,
        raw_context['Np'],
        level_result.keep_turb_amp,
        level_result.kinet_normal,
        level_result.polarity,
        raw_context['B'].copy(),
        raw_context['V'].copy(),
        level_result.Phiangle,
        level_result.VBangle,
        level_result.align_angles_zpm,
        level_result.align_angles_vb,
        int(level_result.tau_equiv_samples_int),
        level_result.needed_index,
        float(raw_context['di_mean']),
        float(raw_context['Vsw_mean']),
        five_points_sfunc=False,
        av_hours=av_hours,
        ts_list=ts_list,
    )




def save_flucs(indices: np.ndarray, final_variables: Mapping[str, Any], ells: np.ndarray, ell_identifier: str):
    return save_flucs_shared(indices, final_variables, ells, ell_identifier)


def _ensure_qorder(qorder: Optional[Sequence[float]]) -> np.ndarray:
    if qorder is None:
        return np.array([2.0], dtype=float)
    q = np.asarray(qorder, dtype=float)
    if q.ndim != 1:
        raise ValueError("qorder must be 1D.")
    return q


def modwt_estimate_full(
    B: pd.DataFrame,
    V: pd.DataFrame,
    V_sc_vel_removed: Optional[pd.DataFrame],
    Np: Union[pd.Series, pd.DataFrame],
    dt: Optional[float],
    di: Optional[float],
    conditions: Mapping[str, Mapping[str, float]],
    qorder: Optional[Sequence[float]] = None,
    wname: str = "la8",
    level: Optional[int] = None,
    level_mode: str = "recommended",
    estimate_alignment_angle: bool = False,
    return_mag_align_correl: bool = False,
    return_coefs: bool = False,
    ts_list: Optional[Union[str, Sequence[str]]] = None,
    return_B_in_vel_units: bool = False,
    use_local_polarity: bool = True,
    est_proj_ells: bool = True,
    sc: Optional[str] = None,
    frame: Optional[str] = None,
    analysis_mode: str = "full",
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
    respect_effective_levels: bool = False,
) -> Dict[str, Any]:
    mode = str(analysis_mode).strip().lower()
    if mode not in ("full", "fast", "sfuncs_only", "core"):
        raise ValueError("analysis_mode must be one of {'full', 'fast', 'sfuncs_only', 'core'}.")

    if mode in ("fast", "sfuncs_only", "core") and return_coefs:
        warnings.warn("analysis_mode=%r ignores return_coefs=True and will not build per-point diagnostic tables." % analysis_mode)
        return_coefs = False

    results = modwt_local_structure_function(
        B=B,
        V=V,
        V_sc_vel_removed=V_sc_vel_removed,
        Np=Np,
        dt=dt,
        wname=wname,
        level=level,
        level_mode=level_mode,
        estimate_alignment_angle=estimate_alignment_angle,
        return_mag_align_correl=return_mag_align_correl,
        return_B_in_vel_units=return_B_in_vel_units,
        use_local_polarity=use_local_polarity,
        est_proj_ells=est_proj_ells,
        sc=sc,
        frame=frame,
        min_valid_fraction=min_valid_fraction,
        min_valid_count=min_valid_count,
    )

    q = _ensure_qorder(qorder)
    meta = results["meta"]
    n_levels = int(meta["J0"])
    if di is None:
        di = float(results["di_mean"])

    cond_perp = conditions["ell_perp"]
    cond_disp = conditions["Ell_perp"]
    cond_par = conditions["ell_par"]
    cond_par_rest = conditions.get("ell_par_rest", cond_par)

    def init_nan(shape):
        return np.full(shape, np.nan)

    sf_ell_perp_B = init_nan((n_levels, len(q)))
    sf_Ell_perp_B = init_nan((n_levels, len(q)))
    sf_ell_par_B = init_nan((n_levels, len(q)))
    sf_ell_par_rest_B = init_nan((n_levels, len(q)))
    sf_overall_B = init_nan((n_levels, len(q)))

    sf_ell_perp_V = init_nan((n_levels, len(q)))
    sf_Ell_perp_V = init_nan((n_levels, len(q)))
    sf_ell_par_V = init_nan((n_levels, len(q)))
    sf_ell_par_rest_V = init_nan((n_levels, len(q)))
    sf_overall_V = init_nan((n_levels, len(q)))

    sf_ell_perp_Zp = init_nan((n_levels, len(q)))
    sf_Ell_perp_Zp = init_nan((n_levels, len(q)))
    sf_ell_par_Zp = init_nan((n_levels, len(q)))
    sf_ell_par_rest_Zp = init_nan((n_levels, len(q)))
    sf_overall_Zp = init_nan((n_levels, len(q)))

    sf_ell_perp_Zm = init_nan((n_levels, len(q)))
    sf_Ell_perp_Zm = init_nan((n_levels, len(q)))
    sf_ell_par_Zm = init_nan((n_levels, len(q)))
    sf_ell_par_rest_Zm = init_nan((n_levels, len(q)))
    sf_overall_Zm = init_nan((n_levels, len(q)))

    counts_ell_perp = init_nan(n_levels)
    counts_Ell_perp = init_nan(n_levels)
    counts_ell_par = init_nan(n_levels)
    counts_ell_par_rest = init_nan(n_levels)
    counts_overall = init_nan(n_levels)

    sdk_ell_perp_B = init_nan(n_levels)
    sdk_Ell_perp_B = init_nan(n_levels)
    sdk_ell_par_B = init_nan(n_levels)
    sdk_ell_par_rest_B = init_nan(n_levels)
    sdk_overall_B = init_nan(n_levels)

    l_ell_perp = init_nan(n_levels)
    l_Ell_perp = init_nan(n_levels)
    l_ell_par = init_nan(n_levels)
    l_ell_par_rest = init_nan(n_levels)
    l_overall = init_nan(n_levels)

    all_thetas: Dict[str, np.ndarray] = {}
    all_phis: Dict[str, np.ndarray] = {}
    vars_by_level: Dict[str, Dict[str, Any]] = {}
    lambda_dict: Dict[str, Any] = {}
    xi_dict: Dict[str, Any] = {}
    ell_par_dict: Dict[str, Any] = {}
    ell_par_rest_dict: Dict[str, Any] = {}
    ell_all_dict: Dict[str, Any] = {}

    align_summary = {
        "VB": {"reg": [], "polar": [], "weighted": [], "sig_r_mean": [], "sig_r_median": [], "counts": []},
        "Zpm": {"reg": [], "polar": [], "weighted": [], "sig_c_mean": [], "sig_c_median": [], "counts": []},
    }

    for jj, lvl in enumerate(results["levels"]):
        all_thetas[str(jj)] = lvl.VBangle
        all_phis[str(jj)] = lvl.Phiangle

        if estimate_alignment_angle and lvl.align_angles_vb and lvl.align_angles_zpm:
            align_summary["VB"]["reg"].append(lvl.align_angles_vb["reg_angle"])
            align_summary["VB"]["polar"].append(lvl.align_angles_vb["polar_inter_angle"])
            align_summary["VB"]["weighted"].append(lvl.align_angles_vb["weighted_angle"])
            align_summary["VB"]["sig_r_mean"].append(lvl.align_angles_vb["sig_r_mean"])
            align_summary["VB"]["sig_r_median"].append(lvl.align_angles_vb["sig_r_median"])
            align_summary["VB"]["counts"].append(lvl.align_angles_vb["counts"])
            align_summary["Zpm"]["reg"].append(lvl.align_angles_zpm["reg_angle"])
            align_summary["Zpm"]["polar"].append(lvl.align_angles_zpm["polar_inter_angle"])
            align_summary["Zpm"]["weighted"].append(lvl.align_angles_zpm["weighted_angle"])
            align_summary["Zpm"]["sig_c_mean"].append(lvl.align_angles_zpm["sig_c_mean"])
            align_summary["Zpm"]["sig_c_median"].append(lvl.align_angles_zpm["sig_c_median"])
            align_summary["Zpm"]["counts"].append(lvl.align_angles_zpm["counts"])

        if respect_effective_levels and (not lvl.is_effective):
            continue

        final_variables = None
        if return_coefs:
            final_variables = modwt_quants_2_estimate(lvl, results, ts_list=ts_list)
            vars_by_level[str(jj)] = final_variables

        base_mask = lvl.valid_mask & np.isfinite(lvl.VBangle) & np.isfinite(lvl.Phiangle)
        idx_perp = np.where(base_mask & (lvl.VBangle > cond_perp["theta"]) & (lvl.Phiangle > cond_perp["phi"]))[0]
        idx_disp = np.where(base_mask & (lvl.VBangle > cond_disp["theta"]) & (lvl.Phiangle < cond_disp["phi"]))[0]
        idx_par = np.where(base_mask & (lvl.VBangle < cond_par["theta"]) & (lvl.Phiangle < cond_par["phi"]))[0]
        idx_par_rest = np.where(base_mask & (lvl.VBangle < cond_par_rest["theta"]) & (lvl.Phiangle < cond_par_rest["phi"]))[0]
        idx_all = np.where(base_mask & (lvl.VBangle > 0.0) & (lvl.Phiangle > 0.0))[0]

        if return_coefs and final_variables is not None:
            lambda_dict[str(jj)] = save_flucs(idx_perp, final_variables, lvl.l_lambda, "lambdas")
            xi_dict[str(jj)] = save_flucs(idx_disp, final_variables, lvl.l_xi, "xis")
            ell_par_dict[str(jj)] = save_flucs(idx_par, final_variables, lvl.l_ell, "ells")
            ell_par_rest_dict[str(jj)] = save_flucs(idx_par_rest, final_variables, lvl.l_ell, "ells_rest")
            ell_all_dict[str(jj)] = save_flucs(idx_all, final_variables, lvl.l_mag, "lambda")

        sf_ell_perp_B[jj, :], sdk_ell_perp_B[jj] = structure_functions_3D(idx_perp, q, lvl.dB)
        sf_Ell_perp_B[jj, :], sdk_Ell_perp_B[jj] = structure_functions_3D(idx_disp, q, lvl.dB)
        sf_ell_par_B[jj, :], sdk_ell_par_B[jj] = structure_functions_3D(idx_par, q, lvl.dB)
        sf_ell_par_rest_B[jj, :], sdk_ell_par_rest_B[jj] = structure_functions_3D(idx_par_rest, q, lvl.dB)
        sf_overall_B[jj, :], sdk_overall_B[jj] = structure_functions_3D(idx_all, q, lvl.dB)

        sf_ell_perp_V[jj, :], _ = structure_functions_3D(idx_perp, q, lvl.dV)
        sf_Ell_perp_V[jj, :], _ = structure_functions_3D(idx_disp, q, lvl.dV)
        sf_ell_par_V[jj, :], _ = structure_functions_3D(idx_par, q, lvl.dV)
        sf_ell_par_rest_V[jj, :], _ = structure_functions_3D(idx_par_rest, q, lvl.dV)
        sf_overall_V[jj, :], _ = structure_functions_3D(idx_all, q, lvl.dV)

        sf_ell_perp_Zp[jj, :], _ = structure_functions_3D(idx_perp, q, lvl.dZp)
        sf_Ell_perp_Zp[jj, :], _ = structure_functions_3D(idx_disp, q, lvl.dZp)
        sf_ell_par_Zp[jj, :], _ = structure_functions_3D(idx_par, q, lvl.dZp)
        sf_ell_par_rest_Zp[jj, :], _ = structure_functions_3D(idx_par_rest, q, lvl.dZp)
        sf_overall_Zp[jj, :], _ = structure_functions_3D(idx_all, q, lvl.dZp)

        sf_ell_perp_Zm[jj, :], _ = structure_functions_3D(idx_perp, q, lvl.dZm)
        sf_Ell_perp_Zm[jj, :], _ = structure_functions_3D(idx_disp, q, lvl.dZm)
        sf_ell_par_Zm[jj, :], _ = structure_functions_3D(idx_par, q, lvl.dZm)
        sf_ell_par_rest_Zm[jj, :], _ = structure_functions_3D(idx_par_rest, q, lvl.dZm)
        sf_overall_Zm[jj, :], _ = structure_functions_3D(idx_all, q, lvl.dZm)

        counts_ell_perp[jj] = idx_perp.size
        counts_Ell_perp[jj] = idx_disp.size
        counts_ell_par[jj] = idx_par.size
        counts_ell_par_rest[jj] = idx_par_rest.size
        counts_overall[jj] = idx_all.size

        l_ell_perp[jj] = np.nanmean(lvl.l_lambda[idx_perp]) if idx_perp.size else np.nan
        l_Ell_perp[jj] = np.nanmean(lvl.l_xi[idx_disp]) if idx_disp.size else np.nan
        l_ell_par[jj] = np.nanmean(lvl.l_ell[idx_par]) if idx_par.size else np.nan
        l_ell_par_rest[jj] = np.nanmean(lvl.l_ell[idx_par_rest]) if idx_par_rest.size else np.nan
        l_overall[jj] = np.nanmean(lvl.l_mag[idx_all]) if idx_all.size else np.nan

    flucts = None
    if return_coefs:
        def _flatten_saved_dict(saved: Mapping[str, Any]) -> pd.DataFrame:
            if not saved:
                return pd.DataFrame()
            rows = []
            for _, item in saved.items():
                rows.append(pd.DataFrame({k: pd.Series(v) for k, v in item.items()}))
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        flucts = {
            "ell_perp": pd.DataFrame(lambda_dict).T if lambda_dict else pd.DataFrame(),
            "Ell_perp": pd.DataFrame(xi_dict).T if xi_dict else pd.DataFrame(),
            "ell_par": pd.DataFrame(ell_par_dict).T if ell_par_dict else pd.DataFrame(),
            "ell_par_rest": pd.DataFrame(ell_par_rest_dict).T if ell_par_rest_dict else pd.DataFrame(),
            "ell_all": _flatten_saved_dict(ell_all_dict),
            "vars_by_level": vars_by_level,
            "tau_lags": meta["tau_equiv_samples_int"],
            "tau_equiv_samples": meta["tau_equiv_samples"],
            "l_di": l_overall,
            "Vsw": results["Vsw_mean"],
            "di": di,
            "dt": results["dt"],
            "B_flag": results["B_flag"],
            "meta": meta,
        }

    Sfunctions = {
        "B": {
            "ell_perp": sf_ell_perp_B.T,
            "Ell_perp": sf_Ell_perp_B.T,
            "ell_par": sf_ell_par_B.T,
            "ell_par_rest": sf_ell_par_rest_B.T,
            "ell_overall": sf_overall_B.T,
            "sdk_ell_perp": sdk_ell_perp_B,
            "sdk_Ell_perp": sdk_Ell_perp_B,
            "sdk_ell_par": sdk_ell_par_B,
            "sdk_ell_par_rest": sdk_ell_par_rest_B,
            "sdk_ell_overall": sdk_overall_B,
        },
        "V": {
            "ell_perp": sf_ell_perp_V.T,
            "Ell_perp": sf_Ell_perp_V.T,
            "ell_par": sf_ell_par_V.T,
            "ell_par_rest": sf_ell_par_rest_V.T,
            "ell_overall": sf_overall_V.T,
        },
        "Zp": {
            "ell_perp": sf_ell_perp_Zp.T,
            "Ell_perp": sf_Ell_perp_Zp.T,
            "ell_par": sf_ell_par_Zp.T,
            "ell_par_rest": sf_ell_par_rest_Zp.T,
            "ell_overall": sf_overall_Zp.T,
        },
        "Zm": {
            "ell_perp": sf_ell_perp_Zm.T,
            "Ell_perp": sf_Ell_perp_Zm.T,
            "ell_par": sf_ell_par_Zm.T,
            "ell_par_rest": sf_ell_par_rest_Zm.T,
            "ell_overall": sf_overall_Zm.T,
        },
        "counts_ell_perp": counts_ell_perp,
        "counts_Ell_perp": counts_Ell_perp,
        "counts_ell_par": counts_ell_par,
        "counts_ell_par_rest": counts_ell_par_rest,
        "counts_ell_overall": counts_overall,
        "l_ell_perp": l_ell_perp,
        "l_Ell_perp": l_Ell_perp,
        "l_ell_par": l_ell_par,
        "l_ell_par_rest": l_ell_par_rest,
        "l_overall": l_overall,
        "B_flag": results["B_flag"],
        "effective_level_mask": meta["effective_level_mask"],
        "J_effective": int(meta["J_effective"]),
    }

    overall_align_angles = None
    if estimate_alignment_angle:
        overall_align_angles = {
            "l_di": l_overall,
            "VB": align_summary["VB"],
            "Zpm": align_summary["Zpm"],
        }

    scales_df = pd.DataFrame(
        {
            "level": meta["levels"],
            "tau_equiv_samples": meta["tau_equiv_samples"],
            "tau_equiv_samples_int": meta["tau_equiv_samples_int"],
            "band_low_samples": meta["band_low_samples"],
            "band_high_samples": meta["band_high_samples"],
            "boundary_width_samples": meta["boundary_width_samples"],
            "support_samples": meta["support_samples"],
            "n_valid": meta["n_valid"],
            "valid_fraction": meta["valid_fraction"],
            "is_effective": meta["effective_level_mask"],
            "period_s": meta["period_s"],
            "frequency_hz": meta["frequency_hz"],
            "l_di": l_overall,
        }
    )

    return {
        "thetas": all_thetas,
        "phis": all_phis,
        "flucts": flucts,
        "l_di": l_overall,
        "ell_di": l_overall,
        "scales": scales_df,
        "Sfunctions": Sfunctions,
        "Sfuncs": Sfunctions,
        "PDFs": None,
        "overall_align_angles": overall_align_angles,
        "meta": meta,
        "raw": results,
        "J_effective": int(meta["J_effective"]),
        "effective_level_mask": meta["effective_level_mask"],
        "analysis_mode": mode,
        "fivept_bridge_path": None,
    }


# -----------------------------------------------------------------------------
# Notebook compatibility wrapper
# -----------------------------------------------------------------------------
def _coerce_level_vectors(
    flucs: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]],
    comp_order: Sequence[str] = ("R", "T", "N"),
) -> List[np.ndarray]:
    if isinstance(flucs, Mapping):
        comps = list(comp_order)
        n_levels = int(np.asarray(flucs[comps[0]]).shape[0])
        return [np.column_stack([np.asarray(flucs[c])[j] for c in comps]) for j in range(n_levels)]
    return [np.asarray(x, dtype=float) for x in flucs]


def estimate_3D_sfuncs_legacy(
    flucs: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]],
    detB: Sequence[np.ndarray],
    ApprB: Sequence[np.ndarray],
    ApprV: Sequence[np.ndarray],
    dt: float,
    Vsw: float,
    di: float,
    conditions: Mapping[str, Mapping[str, float]],
    qorder: Optional[Sequence[float]] = None,
    estimate_PDFS: bool = False,
    return_unit_vecs: bool = False,
    five_points_sfuncs: bool = False,
    return_coefs: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del estimate_PDFS, return_unit_vecs, five_points_sfuncs, Vsw, di  # kept only for API compatibility

    q = _ensure_qorder(qorder)
    dB_levels = _coerce_level_vectors(flucs)
    dV_levels = _coerce_level_vectors(detB)
    B_l_levels = [np.asarray(x, dtype=float) for x in ApprB]
    V_l_levels = [np.asarray(x, dtype=float) for x in ApprV]

    n_levels = len(dB_levels)
    freqs = 1.0 / (dt * (2.0 ** np.arange(n_levels, dtype=float)))

    thetas: Dict[str, np.ndarray] = {}
    phis: Dict[str, np.ndarray] = {}
    flucts_out: Optional[Dict[str, Any]] = None
    if return_coefs:
        flucts_out = {"levels": {}, "frequencies": freqs}

    sf_perp = np.full((n_levels, len(q)), np.nan)
    sf_Perp = np.full((n_levels, len(q)), np.nan)
    sf_par = np.full((n_levels, len(q)), np.nan)
    sf_ov = np.full((n_levels, len(q)), np.nan)

    for j in range(n_levels):
        dB = dB_levels[j]
        dV = dV_levels[j]
        B_l = B_l_levels[j]
        V_l = V_l_levels[j]

        dB_perp = func.perp_vector(dB, B_l)
        l_vec = V_l * (dt * (2.0 ** j))
        _, _, _, VBangle, Phiangle = mag_of_ell_projections_and_angles(l_vec, B_l, dB_perp, est_proj_ells=False)
        thetas[str(j)] = VBangle
        phis[str(j)] = Phiangle

        idx_perp = np.where((VBangle > conditions["ell_perp"]["theta"]) & (Phiangle > conditions["ell_perp"]["phi"]))[0]
        idx_disp = np.where((VBangle > conditions["Ell_perp"]["theta"]) & (Phiangle < conditions["Ell_perp"]["phi"]))[0]
        idx_par = np.where((VBangle < conditions["ell_par"]["theta"]) & (Phiangle < conditions["ell_par"]["phi"]))[0]
        idx_all = np.where((VBangle > 0.0) & (Phiangle > 0.0))[0]

        sf_perp[j, :], _ = structure_functions_3D(idx_perp, q, dB)
        sf_Perp[j, :], _ = structure_functions_3D(idx_disp, q, dB)
        sf_par[j, :], _ = structure_functions_3D(idx_par, q, dB)
        sf_ov[j, :], _ = structure_functions_3D(idx_all, q, dB)

        if return_coefs and flucts_out is not None:
            flucts_out["levels"][str(j)] = {
                "dB": dB,
                "dV": dV,
                "B_l": B_l,
                "V_l": V_l,
                "thetas": VBangle,
                "phis": Phiangle,
            }

    if len(q) == 1:
        sf_perp = sf_perp[:, 0]
        sf_Perp = sf_Perp[:, 0]
        sf_par = sf_par[:, 0]
        sf_ov = sf_ov[:, 0]

    return thetas, phis, flucts_out, freqs, sf_perp, sf_Perp, sf_par, sf_ov


def estimate_3D_sfuncs(*args, **kwargs):
    """Public 5pt-style entry point for the MODWT pipeline.

    This name must accept the same outer-call contract as the mature 5pt
    implementation because notebooks, wrappers, and joblib workers call it by
    name. The compact coefficient-only helper remains available as
    ``estimate_3D_sfuncs_legacy``.
    """
    return estimate_3D_sfuncs_same_format(*args, **kwargs)


def estimate_3D_sfuncs_same_format(
    B: pd.DataFrame,
    V: pd.DataFrame,
    V_sc_vel_removed: pd.DataFrame,
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
    return_B_in_vel_units: bool = False,
    turb_amp_analysis: bool = True,
    estimate_dzp_dzm: bool = False,
    also_return_db_nT: bool = False,
    use_local_polarity: bool = False,
    use_np_factor: bool = True,
    est_proj_ells: bool = True,
    sc: Optional[str] = None,
    frame: Optional[str] = None,
    wname: str = "la8",
    level: Optional[int] = None,
    level_mode: str = "recommended",
    analysis_mode: str = "full",
    min_valid_fraction: float = 0.10,
    min_valid_count: int = 32,
    respect_effective_levels: bool = False,
):
    del tau_values, five_points_sfuncs, extra_conditions, thetas_phis_step, turb_amp_analysis, estimate_dzp_dzm, also_return_db_nT, use_np_factor
    out = modwt_estimate_full(
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
        return_mag_align_correl=return_mag_align_correl,
        return_coefs=return_coefs,
        ts_list=ts_list,
        return_B_in_vel_units=return_B_in_vel_units,
        use_local_polarity=use_local_polarity,
        est_proj_ells=est_proj_ells,
        sc=sc,
        frame=frame,
        analysis_mode=analysis_mode,
        min_valid_fraction=min_valid_fraction,
        min_valid_count=min_valid_count,
        respect_effective_levels=respect_effective_levels,
    )

    if return_coefs and only_general:
        flucts = out["flucts"]
        if flucts is not None and "ell_all" in flucts and not flucts["ell_all"].empty:
            mask = (flucts["ell_all"]["thetas"] > theta_thresh_gen) & (flucts["ell_all"]["phis"] > phi_thresh_gen)
            flucts = dict(flucts)
            flucts["ell_all"] = flucts["ell_all"].loc[mask].reset_index(drop=True)
        out["flucts"] = flucts

    raw = out["raw"]
    last = raw["levels"][-1]
    return last.l_mag, last.l_lambda, last.l_xi, last.l_ell, last.VBangle, last.Phiangle, out["flucts"], out["l_di"], out["Sfunctions"], out["PDFs"], out["overall_align_angles"]



def _pick_existing_columns(df: pd.DataFrame, candidates: Sequence[Sequence[str]]) -> List[str]:
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return list(cols)
    return list(df.columns[: min(3, len(df.columns))])



def _get_B_dataframe(res: Mapping[str, Any], use_low_resol_data: bool = False) -> pd.DataFrame:
    mag = res['Mag']
    if use_low_resol_data:
        B_source = mag.get('B_resampled_part_res', mag['B_resampled'])
    else:
        B_source = mag['B_resampled']
    b_cols = _pick_existing_columns(B_source, [('Br', 'Bt', 'Bn'), ('Bx', 'By', 'Bz')])
    return B_source[b_cols]



def _get_V_dataframe(res: Mapping[str, Any]) -> pd.DataFrame:
    V_source = res['Par']['V_resampled']
    v_cols = _pick_existing_columns(V_source, [('Vr', 'Vt', 'Vn'), ('Vx', 'Vy', 'Vz')])
    return V_source[v_cols]



def _get_Np_dataframe(res: Mapping[str, Any]) -> pd.DataFrame:
    V_source = res['Par']['V_resampled']
    if 'np' in V_source.columns:
        return V_source[['np']]
    if 'Np' in V_source.columns:
        return V_source[['Np']].rename(columns={'Np': 'np'})
    raise KeyError("Could not find density column 'np' or 'Np' in V_resampled")



def build_output_names(consider_Vsc: bool, strict_thresh: int, return_flucs: bool, only_general: int,
                       extra_conditions: bool, theta_thresh_gen: float, phi_thresh_gen: float,
                       thetas_phis_step: int, wname: str) -> Tuple[str, Optional[str]]:
    strict_suffix = '5deg_' if strict_thresh == 1 else ('2deg_' if strict_thresh == 2 else '')
    conditions_suffix = 'extra_conditions_' if extra_conditions else ''
    sfuncs_suffix = '' if return_flucs else 'sfuncs_estimated'
    general_suffix = 'general_SF_' if only_general == 1 else ''
    vsc_suffix = 'Vsc_removed_' if consider_Vsc else ''
    wavelet_suffix = f'modwt_{wname}_'
    align_name = None
    if only_general == 1:
        fname = f"{general_suffix}{wavelet_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}theta_{theta_thresh_gen}_phi_{phi_thresh_gen}_v2.pkl"
        align_name = f"alignment_angles_{wavelet_suffix}{vsc_suffix}_v2.pkl"
    elif only_general == 2:
        fname = f"_all_bins_{general_suffix}{wavelet_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}_step_{str(thetas_phis_step)}_v2.pkl"
    else:
        fname = f"{general_suffix}{wavelet_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}_final_v2.pkl"
        align_name = f"alignment_angles_{wavelet_suffix}{vsc_suffix}_v2.pkl"
    return fname, align_name



def modwt_two_pt_wavelet_analysis(i: int,
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
                                  return_B_in_vel_units: bool = False,
                                  max_interval_dur: float = 240,
                                  estimate_dzp_dzm: bool = False,
                                  use_low_resol_data: bool = False,
                                  use_local_polarity: bool = False,
                                  dt_step: float = 0.25,
                                  wname: str = 'la8',
                                  level: Optional[int] = None,
                                  level_mode: str = 'recommended',
                                  analysis_mode: str = 'full',
                                  min_valid_fraction: float = 0.10,
                                  min_valid_count: int = 32,
                                  respect_effective_levels: bool = True,
                                  output_subdir: str = 'final_modwt'):
    del credentials, Estimate_5point, keep_wave_coeefs, max_hours, estimate_dzp_dzm, dt_step
    warnings.filterwarnings('ignore')
    try:
        func.progress_bar(i, len(fnames))
        res = pd.read_pickle(fnames[i])
        gen_name = fnames[i].replace('final.pkl', 'general.pkl')
        gen = pd.read_pickle(gen_name)
        dts = (gen['End_Time'] - gen['Start_Time']).total_seconds() / 3600.0
        if dts >= max_interval_dur:
            return None
        fname, align_name = build_output_names(consider_Vsc, strict_thresh, return_flucs, only_general,
                                               extra_conditions, theta_thresh_gen, phi_thresh_gen,
                                               thetas_phis_step, wname)
        outdir = str(Path(gen_name.replace('general.pkl', '')).joinpath(output_subdir))
        check_file = str(Path(outdir).joinpath(fname))
        if os.path.exists(check_file) and not overwrite_existing_files:
            print('Skipping existing file:', check_file)
            return check_file
        if overwrite_existing_files and os.path.exists(check_file):
            print('Overwriting', check_file)
        B = _get_B_dataframe(res, use_low_resol_data=use_low_resol_data)
        V = _get_V_dataframe(res)
        Np = _get_Np_dataframe(res)
        B = B[~B.index.duplicated()]
        V = V[~V.index.duplicated()]
        Np = Np[~Np.index.duplicated()]
        try:
            ephem = res.get('Ephem', {}) if isinstance(res, Mapping) else {}
            V = func.newindex(V, B.index)
            if consider_Vsc and isinstance(ephem, Mapping) and 'sc_vel_r' in ephem:
                V_sc = func.newindex(ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].interpolate(), B.index)
                V_sc_rem = V - V_sc.values
            else:
                V_sc_rem = V
        except Exception:
            V_sc_rem = V
        Np = func.newindex(Np, B.index)
        di = float(res['Par']['di_mean'])
        Vsw = float(res['Par']['Vsw_mean'])
        Vsw_norm = float(np.nanmean(np.linalg.norm(V_sc_rem.to_numpy(dtype=float), axis=1)))
        dt = func.find_cadence(B)
        out = estimate_3D_sfuncs_same_format(
            B,
            V,
            V_sc_rem,
            Np,
            dt,
            di,
            conditions,
            qorder,
            tau_values=None,
            five_points_sfuncs=False,
            estimate_alignment_angle=estimate_alignment_angle,
            return_mag_align_correl=return_mag_align_correl,
            return_coefs=return_flucs,
            only_general=only_general,
            theta_thresh_gen=theta_thresh_gen,
            phi_thresh_gen=phi_thresh_gen,
            extra_conditions=extra_conditions,
            ts_list=ts_list,
            thetas_phis_step=thetas_phis_step,
            return_B_in_vel_units=return_B_in_vel_units,
            use_local_polarity=use_local_polarity,
            sc=sc,
            wname=wname,
            level=level,
            level_mode=level_mode,
            analysis_mode=analysis_mode,
            min_valid_fraction=min_valid_fraction,
            min_valid_count=min_valid_count,
            respect_effective_levels=respect_effective_levels,
        )
        _, _, _, _, thetas, phis, flucts, ell_di, Sfunctions, PDFs, overall_align_angles = out
        keep_sfuncs_final = {'di': di, 'Vsw': Vsw, 'Vsw_norm': Vsw_norm, 'ell_di': ell_di, 'Sfuncs': Sfunctions, 'flucts': flucts, 'PDFs': PDFs}
        func.savepickle(keep_sfuncs_final, outdir, fname)
        if estimate_alignment_angle and align_name is not None:
            func.savepickle(overall_align_angles, outdir, align_name)
        return str(Path(outdir) / fname)
    except Exception:
        traceback.print_exc()
        return None


estimate_3D_sfuncs_modwt = estimate_3D_sfuncs_same_format
modwt_analysis = estimate_3D_sfuncs_same_format

__all__ = [name for name in globals() if not name.startswith('_')]
