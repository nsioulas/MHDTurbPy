
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import constants

try:
    from .path_setup import ensure_project_paths
except ImportError:
    from path_setup import ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
import general_functions as func
import TurbPy as turb

try:
    import astropy.units as u
except Exception:  # pragma: no cover
    u = None

mu0 = constants.mu_0
m_p = constants.m_p


def _has_columns(df, *cols):
    return all(col in df.columns for col in cols)


def infer_frame_from_data(B, sc=None, frame=None):
    if frame is not None:
        return frame
    if (sc == 'WIND') and _has_columns(B, 'Bx'):
        return 'GSE'
    if (sc in ('SOLO', 'PSP')) and _has_columns(B, 'Br'):
        return 'RTN'
    if _has_columns(B, 'Br', 'Bt', 'Bn'):
        return 'RTN'
    if _has_columns(B, 'Bx', 'By', 'Bz'):
        return 'GSE'
    return 'sc_frame'


def _background_polarity(B, B_l, needed_index, sc=None, frame=None):
    frame_used = infer_frame_from_data(B, sc=sc, frame=frame)
    if frame_used == 'RTN' and _has_columns(B, 'Br'):
        bg_series = B['Br']
        local_component = B_l[:, 0]
        sign_factor = -1.0 if sc in ('SOLO', 'PSP') else 1.0
    elif frame_used == 'GSE' and _has_columns(B, 'Bx'):
        bg_series = B['Bx']
        local_component = B_l[:, 0]
        sign_factor = 1.0
    elif _has_columns(B, 'Bz'):
        bg_series = B['Bz']
        local_component = B_l[:, 2]
        sign_factor = 1.0
    else:
        bg_series = B.iloc[:, 0]
        local_component = B_l[:, 0]
        sign_factor = 1.0

    bg_smooth = bg_series.to_frame('bg').rolling('30min', center=True).mean() if isinstance(B.index, pd.DatetimeIndex) else bg_series.to_frame('bg').rolling(241, center=True, min_periods=1).mean()
    polarity = sign_factor * np.sign(func.newindex(bg_smooth, needed_index).values.ravel())
    local_polarity = sign_factor * np.sign(local_component)
    return polarity, local_polarity, frame_used


def _mag_from_frame(B, frame):
    if (frame == 'RTN') and _has_columns(B, 'Br', 'Bt', 'Bn'):
        return pd.DataFrame(np.sqrt(B.Br ** 2 + B.Bt ** 2 + B.Bn ** 2), index=B.index)
    if (frame == 'GSE') and _has_columns(B, 'Bx', 'By', 'Bz'):
        return pd.DataFrame(np.sqrt(B.Bx ** 2 + B.By ** 2 + B.Bz ** 2), index=B.index)
    if _has_columns(B, 'Bx', 'By', 'Bz'):
        return pd.DataFrame(np.sqrt(B.Bx ** 2 + B.By ** 2 + B.Bz ** 2), index=B.index)
    if _has_columns(B, 'Br', 'Bt', 'Bn'):
        return pd.DataFrame(np.sqrt(B.Br ** 2 + B.Bt ** 2 + B.Bn ** 2), index=B.index)
    return pd.DataFrame(np.sqrt(np.nansum(B.values ** 2, axis=1)), index=B.index)


def est_alignment_angles(xvec, yvec, return_mag_align_correl=False, est_sigma_c=False):
    numer = np.sqrt(np.nansum(np.cross(xvec, yvec, axis=1) ** 2, axis=1))
    numer_cos = np.nansum(xvec * yvec, axis=1)
    xvec_mag = func.estimate_vec_magnitude(xvec)
    yvec_mag = func.estimate_vec_magnitude(yvec)
    sigma_ts = (xvec_mag ** 2 - yvec_mag ** 2) / (xvec_mag ** 2 + yvec_mag ** 2)
    if est_sigma_c:
        sigma_mean = np.nanmean(xvec_mag ** 2 - yvec_mag ** 2) / np.nanmean(xvec_mag ** 2 + yvec_mag ** 2)
    else:
        sigma_mean = (np.nanmean(xvec_mag ** 2) - np.nanmean(yvec_mag ** 2)) / (np.nanmean(xvec_mag ** 2) + np.nanmean(yvec_mag ** 2))
    sigma_median = np.nan
    denom = xvec_mag * yvec_mag
    numer[np.isinf(numer)] = np.nan
    denom[np.isinf(denom)] = np.nan
    counts = len(numer[numer > -1e10])
    reg_align_angle_sin = np.nanmean(numer / denom)
    polar_int_angle_sin = np.nanmean(numer) / np.nanmean(denom)
    weighted_sins = np.nan
    if return_mag_align_correl is False:
        sins = None
    return counts, sigma_ts, sigma_mean, sigma_median, numer, numer_cos, denom, xvec_mag, yvec_mag, reg_align_angle_sin, polar_int_angle_sin, weighted_sins


def fast_unit_vec(a):
    mag = func.estimate_vec_magnitude(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (a.T / mag).T


def mag_of_ell_projections_and_angles(l_vector, B_l_vector, db_perp_vector, est_proj_ells=True):
    try:
        B_l_vector = (B_l_vector.T / func.estimate_vec_magnitude(B_l_vector)).T
        db_perp_vector = (db_perp_vector.T / func.estimate_vec_magnitude(db_perp_vector)).T
        if est_proj_ells:
            b_perp_vector = np.cross(B_l_vector, db_perp_vector)
            l_ell = np.abs(func.dot_product(l_vector, B_l_vector))
            l_xi = np.abs(func.dot_product(l_vector, db_perp_vector))
            l_lambda = np.abs(func.dot_product(l_vector, b_perp_vector))
        else:
            l_ell, l_xi, l_lambda = np.nan, np.nan, np.nan
        l_perp = func.perp_vector(l_vector, B_l_vector)
        VBangle = func.angle_between_vectors(l_vector, B_l_vector, restrict_2_90=True)
        Phiangle = func.angle_between_vectors(l_perp, db_perp_vector, restrict_2_90=True)
    except Exception:
        traceback.print_exc()
        l_ell, l_xi, l_lambda = np.nan, np.nan, np.nan
        VBangle = np.full(len(l_vector), np.nan)
        Phiangle = np.full(len(l_vector), np.nan)
    return l_ell, l_xi, l_lambda, VBangle, Phiangle


def structure_functions_3D(indices, qorder, mat, max_std=12):
    result = np.zeros(len(qorder))
    ar = np.abs(mat.T[0][indices])
    at = np.abs(mat.T[1][indices])
    an = np.abs(mat.T[2][indices])
    std_r = np.nanstd(ar)
    std_t = np.nanstd(at)
    std_n = np.nanstd(an)
    index = (ar < max_std * std_r) & (at < max_std * std_t) & (an < max_std * std_n)
    ar = ar[index]
    at = at[index]
    an = an[index]
    dbtot = np.sqrt(ar ** 2 + at ** 2 + an ** 2)
    for i in range(len(qorder)):
        result[i] = np.nanmean((dbtot) ** qorder[i])
    sdk = result[3] / result[1] ** 2 if (len(result) > 3) and (result[1] != 0.0) else np.nan
    return list(result), sdk


def vars_2_estimate(ts_list=None):
    default_vars = ['R', 'T', 'N']
    return default_vars if ts_list is None else list(ts_list) + default_vars


def quants_2_estimate(l_ell, l_lambda, l_xi, l_mag, V_l, B_l, local_polarity, dB, dB_perp, dB_parallel, dV, dzp, dzm, dN,
                      Np, keep_turb_amp, kinet_normal, polarity, B, V, phis, thetas, align_angles_zpm, align_angles_vb,
                      tau_value, needed_index, di, Vsw, five_points_sfunc=True, av_hours=None, ts_list=None):
    from scipy import constants as _constants
    if av_hours is None:
        av_hours = 1 / 60

    def _normalize_requested_quants(ts_list):
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

    def _get_component_keys(df, candidate_sets):
        cols = set(df.columns)
        for keys in candidate_sets:
            if all(k in cols for k in keys):
                return list(keys)
        raise KeyError('Could not infer component keys from dataframe columns.')

    def _to_component_frame(df, keys):
        return df.loc[:, keys].copy()

    def _to_series_norm(df, keys):
        vals = df.loc[:, keys].to_numpy(dtype=float)
        return pd.Series(np.sqrt(np.sum(vals ** 2, axis=1)), index=df.index)

    def _ensure_dataframe(x, default_name):
        if isinstance(x, pd.Series):
            name = x.name if x.name is not None else default_name
            return x.to_frame(name=name)
        return x.copy()

    def _store_component_family(arr, canonical_prefix, family_aliases=(), canonical_component_alias_prefixes=(), legacy_component_output_prefixes=(), bare_legacy_output=False):
        family_requested = any(alias in quants for alias in family_aliases)
        for comp, idx in comp_map.items():
            canonical_key = f"{canonical_prefix}{comp}"
            canonical_component_requested = canonical_key in quants or any(f"{prefix}{comp}" in quants for prefix in canonical_component_alias_prefixes)
            if family_requested or canonical_component_requested:
                variables[canonical_key] = arr[:, idx]
            if bare_legacy_output and comp in quants:
                variables[comp] = arr[:, idx]
            for prefix in legacy_component_output_prefixes:
                legacy_key = f"{prefix}{comp}"
                if legacy_key in quants:
                    variables[legacy_key] = arr[:, idx]

    def _make_vec_df(arr, keys):
        return pd.DataFrame(arr, index=needed_index, columns=keys)

    def _estimate_vec_pvi(df, keys):
        return func.newindex(
            turb.estimate_PVI(
                df.copy(), [1], [tau_value], di, Vsw, hours=1, keys=keys, five_points_sfunc=five_points_sfunc,
                PVI_vec_or_mod='vec', use_taus=True, return_only_PVI=True, n_jobs=-1, input_flucts=True, dbs=df,
            ),
            needed_index,
        ).values.T[0]

    quants = _normalize_requested_quants(ts_list=ts_list)
    variables = {}
    comp_map = {'R': 0, 'T': 1, 'N': 2}
    b_keys = _get_component_keys(B, (('Br', 'Bt', 'Bn'), ('Bx', 'By', 'Bz')))
    v_keys = _get_component_keys(V, (('Vr', 'Vt', 'Vn'), ('Vx', 'Vy', 'Vz')))
    B_comp = _to_component_frame(B, b_keys)
    V_comp = _to_component_frame(V, v_keys)
    Np_df = _ensure_dataframe(Np, 'Np')

    _store_component_family(dB, canonical_prefix='dB_', family_aliases=('dB',), bare_legacy_output=True)
    _store_component_family(dV, canonical_prefix='dV_', family_aliases=('dV',), legacy_component_output_prefixes=('V_',))
    _store_component_family(dzp, canonical_prefix='zp_', family_aliases=('zp', 'dzp'), canonical_component_alias_prefixes=('dzp_',))
    _store_component_family(dzm, canonical_prefix='zm_', family_aliases=('zm', 'dzm'), canonical_component_alias_prefixes=('dzm_',))
    _store_component_family(B_l, canonical_prefix='B_l_', family_aliases=('B_l',))

    if 'puq_pol_heat_rate' in quants and u is not None:
        V_l_mag = np.sqrt(np.nansum(V_l ** 2, axis=1))
        dzp_longitud = np.nansum(dzp * (V_l.T / V_l_mag).T, axis=1) * (1e3 * u.m / u.s)
        dzm_longitud = np.nansum(dzm * (V_l.T / V_l_mag).T, axis=1) * (1e3 * u.m / u.s)
        zp_sq = np.sum(dzp ** 2, axis=1) * (1e3 * u.m / u.s) ** 2
        zm_sq = np.sum(dzm ** 2, axis=1) * (1e3 * u.m / u.s) ** 2
        ell = l_mag * di * 1e3 * u.m
        rho = (func.newindex(Np_df, needed_index).values.ravel() * (u.cm ** -3) * (_constants.m_p * u.kg)).to(u.kg / u.m ** 3)
        variables['e_plus'] = (-(((3 / 4) * dzm_longitud * zp_sq / ell) * rho).to(u.W / u.m ** 3).value)
        variables['e_minus'] = (-(((3 / 4) * dzp_longitud * zm_sq / ell) * rho).to(u.W / u.m ** 3).value)

    if 'db_perp_amp_nT' in quants and isinstance(keep_turb_amp, Mapping) and 'dB_perp_amp_nT' in keep_turb_amp:
        variables['db_perp_amp'] = keep_turb_amp['dB_perp_amp_nT']
    if 'db_par_amp_nT' in quants and isinstance(keep_turb_amp, Mapping) and 'dB_parallel_amp_nT' in keep_turb_amp:
        variables['db_par_amp'] = keep_turb_amp['dB_parallel_amp_nT']

    if 'PVI_vec_zp' in quants:
        dzp_df = _make_vec_df(dzp, ['Zpr', 'Zpt', 'Zpn'])
        variables['PVI_vec_zp'] = _estimate_vec_pvi(dzp_df, ['Zpr', 'Zpt', 'Zpn'])
    if 'PVI_vec_zm' in quants:
        dzm_df = _make_vec_df(dzm, ['Zmr', 'Zmt', 'Zmn'])
        variables['PVI_vec_zm'] = _estimate_vec_pvi(dzm_df, ['Zmr', 'Zmt', 'Zmn'])
    if 'PVI_vec' in quants:
        db_df = _make_vec_df(dB, b_keys)
        variables['PVI_vec'] = _estimate_vec_pvi(db_df, b_keys)
    if 'PVI_vec_V' in quants:
        dv_df = _make_vec_df(dV, v_keys)
        variables['PVI_vec_V'] = _estimate_vec_pvi(dv_df, v_keys)
    if 'PVI_Np' in quants:
        variables['PVI_Np'] = func.newindex(
            turb.estimate_PVI(Np_df.copy(), [1], [tau_value], di, Vsw, hours=1, keys=list(Np_df.columns), five_points_sfunc=five_points_sfunc,
                              PVI_vec_or_mod='mod', use_taus=True, return_only_PVI=True, n_jobs=-1),
            needed_index,
        ).values.T[0]

    for key, arr in [('l_ell', l_ell), ('l_lambda', l_lambda), ('l_xi', l_xi), ('polarity', polarity), ('local_polarity', local_polarity), ('db_index', needed_index), ('kinet_normal', kinet_normal), ('phis', phis), ('thetas', thetas)]:
        if key in quants:
            variables[key] = arr
    if 'N_p' in quants:
        dN_arr = np.asarray(dN)
        variables['N_p'] = dN_arr[:, 0] if dN_arr.ndim > 1 else dN_arr
    if 'Vsw' in quants:
        variables['Vsw'] = func.newindex(_to_series_norm(V_comp, v_keys), needed_index).values
    if 'Bmod' in quants:
        variables['Bmod'] = func.newindex(_to_series_norm(B_comp, b_keys), needed_index).values
    if 'VBangle_big' in quants:
        variables['VBangle_big'] = func.newindex(pd.DataFrame({'values': func.angle_between_vectors(B_comp.values, V_comp.values)}, index=B_comp.index), needed_index).values.T[0]
    if 'sig_c' in quants and isinstance(align_angles_zpm, Mapping) and 'sig_c_ts' in align_angles_zpm:
        variables['sig_c'] = align_angles_zpm['sig_c_ts']
    if 'sig_r' in quants and isinstance(align_angles_vb, Mapping) and 'sig_r_ts' in align_angles_vb:
        variables['sig_r'] = align_angles_vb['sig_r_ts']
    for key in ('sins_ub_num', 'cos_ub_num', 'sins_ub_den'):
        if key in quants and isinstance(align_angles_vb, Mapping) and key in align_angles_vb:
            variables[key] = align_angles_vb[key]
    for key in ('sins_zp_num', 'cos_zp_num', 'sins_zp_den', 'zp_mag', 'zm_mag'):
        if key in quants and isinstance(align_angles_zpm, Mapping) and key in align_angles_zpm:
            variables[key] = align_angles_zpm[key]
    if 'sins_zp' in quants and isinstance(align_angles_zpm, Mapping):
        variables['sins_zp'] = align_angles_zpm['sins_zp_num'] / align_angles_zpm['sins_zp_den']
    if 'sins_ub' in quants and isinstance(align_angles_vb, Mapping):
        variables['sins_ub'] = align_angles_vb['sins_ub_num'] / align_angles_vb['sins_ub_den']

    if 'compress_squire' in quants:
        variables['compress_squire'] = func.newindex(turb.compressibility_complex_squire(tau_value, B_comp.copy(), av_hours=av_hours), needed_index).values.T[0]
    if 'compress_squire_V' in quants:
        variables['compress_squire_V'] = func.newindex(turb.compressibility_complex_squire(tau_value, V_comp.copy(), keys=v_keys, av_hours=av_hours), needed_index).values.T[0]
    if 'compress_chen' in quants:
        variables['compress_chen'] = func.newindex(turb.compressibility_complex_chen(tau_value, B_comp.copy(), av_hours=av_hours), needed_index).values.T[0]
    if 'compress_chen_V' in quants:
        variables['compress_chen_V'] = func.newindex(turb.compressibility_complex_chen(tau_value, V_comp.copy(), keys=v_keys, av_hours=av_hours), needed_index).values.T[0]
    if 'compress_simple' in quants:
        variables['compress_simple'] = func.newindex(turb.calculate_compressibility(tau_value, B_comp.copy(), keys=b_keys, five_points_sfunc=five_points_sfunc), needed_index).values.T[0]
    if 'compress_simple_V' in quants:
        variables['compress_simple_V'] = func.newindex(turb.calculate_compressibility(tau_value, V_comp.copy(), keys=v_keys, five_points_sfunc=five_points_sfunc), needed_index).values.T[0]
    if 'variance' in quants:
        variables['variance'] = func.newindex(turb.variance_anisotropy_verdini(tau_value, B_comp.copy(), av_hours=av_hours), needed_index).values
    if 'norm_turb_amplitude' in quants:
        variables['norm_turb_amplitude'] = func.newindex(turb.norm_fluct_amplitude(tau_value, B_comp.copy(), av_hours=av_hours, denom_av_hours='4H'), needed_index).values.T[0]
    return variables


def save_flucs(indices, final_variables, ells, ell_identifier):
    var_keys = list(final_variables.keys())
    if len(indices) > 0:
        selected_points = {var_key: final_variables[var_key][indices] for var_key in var_keys}
        selected_points[ell_identifier] = ells[indices]
    else:
        selected_points = {var_key: [np.nan] for var_key in var_keys}
        selected_points[ell_identifier] = [np.nan]
    return selected_points
