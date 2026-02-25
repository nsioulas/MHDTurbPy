import numpy as np
import pandas as pd
import sys
from pathlib import Path

try:
    from .path_setup import ensure_project_paths
except ImportError:
    from path_setup import ensure_project_paths

_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
import gc
import time
from numba import jit, njit, prange, objmode
import os
from pathlib import Path
from glob import glob
from gc import collect
import traceback
import datetime

# Import TurbPy
import TurbPy as turb
import general_functions as func
import plasma_params as plasma

from scipy import constants
mu_0            = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu0             = constants.mu_0
m_p             = constants.m_p   # Proton mass [kg]
kb              = constants.k     # Boltzman's constant [J/K]
au_to_km        = 1.496e8
T_to_Gauss      = 1e4
km2m            = 1e3
nT2T            = 1e-9
cm2m            = 1e-2


def _coord_columns(coord_type='RTN'):
    """Return magnetic/velocity component names for a coordinate frame."""
    coord = str(coord_type).upper()
    if coord == 'RTN':
        return ["Br", "Bt", "Bn"], ["Vr", "Vt", "Vn"]
    return ["Bx", "By", "Bz"], ["Vx", "Vy", "Vz"]


def apply_rolling_mean(f_df, settings, coord_type='RTN'):
    """Apply rolling mean columns for magnetic and velocity vectors."""
    columns_b, columns_v = _coord_columns(coord_type)

    for cols in (columns_b, columns_v):
        f_df[[f"{col}_mean" for col in cols]] = (
            f_df[cols]
            .rolling(settings['rol_window'], center=True)
            .mean()
            .interpolate()
        )

    return f_df


def apply_rolling_mean_and_get_columns(f_df, settings):
    """Apply rolling means using preferred frame with fallback."""
    prefer_rtn = bool(settings.get('in_rtn', True))
    tries = ('RTN', 'XYZ') if prefer_rtn else ('XYZ', 'RTN')

    last_err = None
    for coord in tries:
        try:
            f_df = apply_rolling_mean(f_df, settings, coord)
            columns_b, columns_v = _coord_columns(coord)
            return f_df, columns_b, columns_v
        except KeyError as err:
            last_err = err

    raise KeyError(f"Could not find valid magnetic/velocity component columns. Last error: {last_err}")


def _rolling_sign(series: pd.Series, window='30min', negate=False):
    sign = np.sign(series.rolling(window, center=True).median())
    return -sign if negate else sign


def calculate_signB(f_df, settings):
    """Calculate sign(B_r) proxy used in Elsasser definitions."""
    sc = settings['sc']

    if sc in ['PSP', 'SOLO', 'HELIOS_A', 'HELIOS_B', 'ACE', 'Ulysses']:
        if 'Br' in f_df.columns:
            return _rolling_sign(f_df['Br'], negate=True)
        if sc == 'PSP' and 'Bz' in f_df.columns:
            return _rolling_sign(f_df['Bz'])
        if 'Bx' in f_df.columns:
            return _rolling_sign(f_df['Bx'])
        raise ValueError(f"{sc} requires Br/Bx (or Bz for PSP) to estimate signB.")

    if sc == 'WIND':
        # Keep WIND sign convention consistent with other missions:
        # use -sign(Br) in RTN, which is equivalent to +sign(Bx) in GSE
        # when Br ~= -Bx after the L1-approx conversion.
        if 'Br' in f_df.columns:
            return _rolling_sign(f_df['Br'], negate=True)
        if 'Bx' in f_df.columns:
            return _rolling_sign(f_df['Bx'])
        raise ValueError("WIND requires Br or Bx column to estimate signB.")

    raise ValueError("Required column is missing in DataFrame.")


def calculate_components(dv, dva, signB):
    """Calculate Zp and Zm components."""
    Zpr, Zmr = dv[0] + signB * dva[0], dv[0] - signB * dva[0]
    Zpt, Zmt = dv[1] + signB * dva[1], dv[1] - signB * dva[1]
    Zpn, Zmn = dv[2] + signB * dva[2], dv[2] - signB * dva[2]
    return np.array([Zpr, Zpt, Zpn]), np.array([Zmr, Zmt, Zmn])


def estimate_magnetic_field_psd(df_mag,
                                dtb,
                                settings,
                                diagnostics,
                                return_mag_df=True):
    """
    Estimate the Power Spectral Density (PSD) of the magnetic field, perform optional smoothing,
    and return a dictionary containing the PSD information and diagnostics.
    """

    def estimate_B_psd_and_smooth(df_mag, dtb, settings):
        psd_B_R, psd_B_T, psd_B_N, f_B, psd_B, f_B_mid, f_B_mean, psd_B_smooth, psd_Mod = (None,) * 9

        if settings['estimate_psd_b']:
            try:
                f_B, psd_B, psd_B_R, psd_B_T, psd_B_N, psd_Mod = turb.TracePSD(
                    df_mag.values.T[0],
                    df_mag.values.T[1],
                    df_mag.values.T[2],
                    dtb,
                    return_components=settings['est_PSD_components'],
                    return_mod=True
                )

                if settings['smooth_psd']:
                    f_B_mid, f_B_mean, psd_B_smooth = func.smoothing_function(f_B, psd_B, window=2, pad=1)

            except Exception:
                traceback.print_exc()

        return f_B, psd_B, psd_B_R, psd_B_T, psd_B_N, f_B_mid, f_B_mean, psd_B_smooth, psd_Mod

    f_B, psd_B, psd_B_R, psd_B_T, psd_B_N, f_B_mid, f_B_mean, psd_B_smooth, psd_Mod = estimate_B_psd_and_smooth(
        df_mag, dtb, settings
    )

    mag_dict = {
        "f_B_orig": f_B,
        "f_B_mid": f_B_mid,
        "f_B_mean": f_B_mean,
        "PSD_f_orig": psd_B,
        "PSD_f_orig_R": psd_B_R,
        "PSD_f_orig_T": psd_B_T,
        "PSD_f_orig_N": psd_B_N,
        "PSD_f_orig_Mod": psd_Mod,
        "PSD_f_smoothed": psd_B_smooth,
        "Fraction_missing": diagnostics['Mag']["Frac_miss"],
        "resolution": diagnostics['Mag']["resol"]
    }

    return mag_dict


def estimate_psd_dict(settings, sigs_df, dtb, estimate_means=False):

    def estimate_psds(sigs_df, component_keys, dtb, settings):
        values = [sigs_df[key].values for key in component_keys]
        return turb.TracePSD(
            *values,
            dtb,
            remove_mean=False,
            return_components=settings['est_PSD_components']
        )

    psd_zp_R = psd_zp_T = psd_zp_N = None
    psd_zm_R = psd_zm_T = psd_zm_N = None
    psd_bb_R = psd_bb_T = psd_bb_N = None
    psd_vv_R = psd_vv_T = psd_vv_N = None
    f_zp = psd_zp = f_zm = psd_zm = f_vv = psd_vv = f_bb = psd_bb = None

    if settings['estimate_psd_v']:
        components = {
            'zp': ['dZpr', 'dZpt', 'dZpn'],
            'zm': ['dZmr', 'dZmt', 'dZmn'],
            'vv': ['dv_r', 'dv_t', 'dv_n'],
            'bb': ['dva_r', 'dva_t', 'dva_n']
        }

        results = {key: estimate_psds(sigs_df, components[key], dtb, settings) for key in components}

        f_zp, psd_zp, psd_zp_R, psd_zp_T, psd_zp_N = results['zp']
        f_zm, psd_zm, psd_zm_R, psd_zm_T, psd_zm_N = results['zm']
        f_vv, psd_vv, psd_vv_R, psd_vv_T, psd_vv_N = results['vv']
        f_bb, psd_bb, psd_bb_R, psd_bb_T, psd_bb_N = results['bb']

    dict_psd = {
        "f_zpm": f_zp,
        "psd_v": psd_vv,
        "psd_b": psd_bb,
        "psd_b_R": psd_bb_R,
        "psd_b_T": psd_bb_T,
        "psd_b_N": psd_bb_N,
        "psd_zp": psd_zp,
        "psd_zm": psd_zm,
    } if settings['estimate_psd_v'] else {}

    addit_dict = {
        'median_sw_speed': np.nanmedian(sigs_df['Vsw']),
        'mean_sw_speed': np.nanmean(sigs_df['Vsw']),
        'std_sw_speed': np.nanstd(sigs_df['Vsw']),

        'median_alfv_speed': np.nanmedian(sigs_df['Va']),
        'mean_alfv_speed': np.nanmean(sigs_df['Va']),
        'std_alfv_speed': np.nanstd(sigs_df['Va']),

        'median_MA': np.nanmedian(np.abs(sigs_df['Ma'])),
        'mean_MA': np.nanmean(np.abs(sigs_df['Ma'])),
        'std_MA': np.nanstd(np.abs(sigs_df['Ma'])),

        'median_MA_r': np.nanmedian(np.abs(sigs_df['Ma_r'])),
        'mean_MA_r': np.nanmean(np.abs(sigs_df['Ma_r'])),
        'std_MA_r': np.nanstd(np.abs(sigs_df['Ma_r'])),

        'beta_mean': np.nanmean(sigs_df['beta']),
        'beta_std': np.nanstd(sigs_df['beta']),

        'sigma_r_mean': np.nanmean(sigs_df['sigma_r']),
        'sigma_r_median': np.nanmedian(sigs_df['sigma_r']),
        'sigma_r_std': np.nanstd(sigs_df['sigma_r']),

        'sigma_c_median': np.nanmedian(np.abs(sigs_df['sigma_c'])),
        'sigma_c_mean': np.nanmean(np.abs(sigs_df['sigma_c'])),
        'sigma_c_std': np.nanstd(np.abs(sigs_df['sigma_c'])),

        'Np_mean': np.nanmean(sigs_df['np']),
        'Np_std': np.nanstd(sigs_df['np']),
        'di_mean': np.nanmean(sigs_df['d_i']),
        'di_std': np.nanstd(sigs_df['d_i']),

        'VBangle_mean': np.nanmean(sigs_df['VB']),
        'VBangle_std': np.nanstd(sigs_df['VB']),
    } if estimate_means else {}

    return dict_psd, addit_dict


def calculate_diagnostics(
    df_mag,
    df_part,
    dist,
    settings,
    diagnostics
):

    f_df = df_mag.join(df_part)

    

    # Estimate fluctuations
    f_df, columns_b, columns_v = apply_rolling_mean_and_get_columns(f_df, settings)

    vx, vy, vz = f_df[columns_v].to_numpy().T
    bx, by, bz = f_df[columns_b].to_numpy().T

    vx_me, vy_me, vz_me = (
        f_df[columns_v]
        .rolling("1min", center=True, min_periods=1)
        .median()
        .to_numpy()
        .T
    )

    Bmod_med = (
        np.sqrt(f_df[columns_b].pow(2).sum(axis=1))
        .rolling("1min", center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # Spacecraft-velocity removal: ONLY for PSP.
    # If PSP and sc-vel columns are missing -> set components to 0 (no removal).
    # ------------------------------------------------------------------
    sc_name = str(settings.get('sc', '')).upper()
    #use_sc_removal = (sc_name in {"PSP", "SOLO"}) and ("Vr" in columns_v)
    use_sc_removal = (sc_name in {"PSP", "SOLO"}) and ("Vr" in columns_v)

    if use_sc_removal:
        n = len(f_df)
        sc_r = np.zeros(n, dtype=float)
        sc_t = np.zeros(n, dtype=float)
        sc_n = np.zeros(n, dtype=float)

        if {"sc_vel_r", "sc_vel_t", "sc_vel_n"}.issubset(f_df.columns):
            sc_r, sc_t, sc_n = (
                f_df[["sc_vel_r", "sc_vel_t", "sc_vel_n"]]
                .fillna(0.0)
                .to_numpy()
                .T
            )

        vx_sc_rem = vx - sc_r
        vy_sc_rem = vy - sc_t
        vz_sc_rem = vz - sc_n

    # Const to normalize mag field in vel units
    f_df['np_mean'] = f_df['np'].rolling('10min', center=True).mean().interpolate()

    if settings['sc'] in ['WIND']:
        kinet_normal = 1e-15 / np.sqrt(1.16 * mu0 * f_df['np_mean'].values * m_p)
    else:
        kinet_normal = 1e-15 / np.sqrt(mu0 * f_df['np_mean'].values * m_p)

    # Background speeds
    Va_ts_me = Bmod_med * kinet_normal
    V_ts_me = np.sqrt(vx_me**2 + vy_me**2 + vz_me**2)

    Va_ts = np.vstack([bx, by, bz]) * kinet_normal
    V_ts = np.vstack([vx, vy, vz])

    mean_columns_b = [f"{col}_mean" for col in columns_b]
    mean_columns_v = [f"{col}_mean" for col in columns_v]

    dva = Va_ts - f_df[mean_columns_b].to_numpy().T * kinet_normal
    dv = V_ts - f_df[mean_columns_v].to_numpy().T

    Bmag = np.sqrt(bx**2 + by**2 + bz**2)
    Vsw = np.sqrt(vx**2 + vy**2 + vz**2)

    Vth, Vth_mean, Vth_median, Vth_std = plasma.estimate_Vth(f_df.Vth.values)
    Vsw, Vsw_mean, Vsw_median, Vsw_std = plasma.estimate_Vsw(Vsw)
    Np, Np_mean, Np_median, Np_std = plasma.estimate_Np(f_df.np.values)
    di, di_mean, di_median, di_std = plasma.estimate_di(Np)

    from plasmapy.formulary import beta  # noqa: F401
    print(f_df.keys())
    beta, beta_mean, beta_median, beta_std = plasma.estimate_beta(Bmag, Np, f_df.Tp.values)

    rho_i, rho_i_mean, rho_i_median, rho_i_std = plasma.estimate_rho_i(di, beta)

    alfv_speed = np.sqrt(np.sum(Va_ts**2, axis=0))
    alfv_speed_back = np.sqrt(np.sum(Va_ts**2, axis=0))

    Ma_ts = Vsw / alfv_speed
    Ma_r_ts = Vsw / alfv_speed
    Ma_ts_me = V_ts_me / Va_ts_me

    # ------------------------------------------------------------------
    # VB angle: ONLY use sc-removal branch for PSP (and only when use_sc_removal is True)
    # ------------------------------------------------------------------
    if use_sc_removal:
        Vsw_sc_rem = np.sqrt(vx_sc_rem**2 + vy_sc_rem**2 + vz_sc_rem**2)
        vbang = np.arccos(
            (vx_sc_rem * bx + vy_sc_rem * by + vz_sc_rem * bz) / (Bmag * Vsw_sc_rem)
        ) * 180 / np.pi
    else:
        vbang = np.arccos(
            (vx * bx + vy * by + vz * bz) / (Bmag * Vsw)
        ) * 180 / np.pi

    VBangle_mean, VBangle_std = np.nanmean(vbang), np.nanstd(vbang)

    signB = calculate_signB(f_df, settings)
    dZp, dZm = calculate_components(dv, dva, signB)
    Zp, Zm = calculate_components(V_ts, Va_ts, signB)

    sigs_df = (
        pd.DataFrame({
            'DateTime': f_df.index.values,

            'dZpr': dZp[0], 'dZpt': dZp[1], 'dZpn': dZp[2],
            'dZmr': dZm[0], 'dZmt': dZm[1], 'dZmn': dZm[2],
            'dva_r': dva[0], 'dva_t': dva[1], 'dva_n': dva[2],
            'dv_r': dv[0], 'dv_t': dv[1], 'dv_n': dv[2],

            'Zpr': Zp[0], 'Zpt': Zp[1], 'Zpn': Zp[2],
            'Zmr': Zm[0], 'Zmt': Zm[1], 'Zmn': Zm[2],
            'va_r': Va_ts[0], 'va_t': Va_ts[1], 'va_n': Va_ts[2],
            'v_r': V_ts[0], 'v_t': V_ts[1], 'v_n': V_ts[2],

            'beta': beta, 'np': Np, 'Tp': f_df.Tp.values,
            'VB': vbang, 'd_i': di,
            'Ma_inst': Ma_ts, 'Ma': Ma_ts_me, 'rho_i': rho_i,
            'Vsw': Vsw, 'kin_norm': kinet_normal,
            'Ma_r': Ma_r_ts,
            'Va': alfv_speed
        })
        .set_index('DateTime')
    )

    rol_w = settings['rol_window']

    moments = pd.DataFrame({
        'num': (dva * dv).sum(axis=0),
        'den_v': (dv**2).sum(axis=0),
        'den_b': (dva**2).sum(axis=0),
    }, index=f_df.index).rolling(window=rol_w, center=True).mean()

    sigs_df['sigma_c'] = 2 * signB * moments['num'] / (moments['den_v'] + moments['den_b'])
    sigs_df['sigma_r'] = (moments['den_v'] - moments['den_b']) / (moments['den_v'] + moments['den_b'])

    del moments

    dict_psd, _ = estimate_psd_dict(
        settings,
        sigs_df,
        func.find_cadence(sigs_df)
    )

    part_dict = {
        'dict_psd': dict_psd,

        'mean_sw_speed  ': Vsw_mean,
        'median_sw_speed': Vsw_median,
        'std_sw_speed': Vsw_std,

        'median_alfv_speed': np.nanmedian(alfv_speed),
        'mean_alfv_speed': np.nanmean(alfv_speed),
        'std_alfv_speed': np.nanstd(alfv_speed),

        'median_MA': np.nanmedian(Ma_ts),
        'mean_MA': np.nanmean(Ma_ts),
        'std_MA': np.nanstd(Ma_ts),

        'median_MA_r': np.nanmedian(Ma_r_ts),
        'mean_MA_r': np.nanmean(Ma_r_ts),
        'std_MA_r': np.nanstd(Ma_r_ts),

        'beta_mean': beta_mean,
        'beta_std': beta_std,

        'sigma_r_mean': np.nanmean(sigs_df['sigma_r']),
        'sigma_r_median': np.nanmedian(sigs_df['sigma_r']),
        'sigma_r_std': np.nanstd(sigs_df['sigma_r']),

        'sigma_c_median': np.nanmedian(np.abs(sigs_df['sigma_c'])),
        'sigma_c_mean': np.nanmean(np.abs(sigs_df['sigma_c'])),
        'sigma_c_std': np.nanstd(np.abs(sigs_df['sigma_c'])),

        'Vth_mean': Vth_mean,
        'Vth_std': Vth_std,

        'Vsw_mean': Vsw_mean,
        'Vsw_std': Vsw_std,
        'Np_mean': Np_mean,
        'Np_std': Np_std,
        'di_mean': di_mean,
        'di_std': di_std,
        'rho_i_mean': rho_i_mean,
        'rho_i_std': rho_i_std,
        'VBangle_mean': VBangle_mean,
        'VBangle_std': VBangle_std
    }

    return part_dict, sigs_df


def general_dict_func(diagnostics,
                      df_mag,
                      df_par,
                      dist):
    """Make distance dataframe to begin at the same time as magnetic field timeseries."""
    try:
        r_psp = np.nanmean(func.use_dates_return_elements_of_df_inbetween(
            df_mag.index[0], df_mag.index[-1], dist['Dist_au']
        ))
    except Exception:
        try:
            r_psp = np.nanmean(func.use_dates_return_elements_of_df_inbetween(
                df_mag.index[0], df_mag.index[-1], df_par['Dist_au']
            ))
        except Exception:
            r_psp = 1

    if 'qtn_flag' not in diagnostics:
        diagnostics['qtn_flag'] = None

    if 'part_flag' not in diagnostics:
        diagnostics['part_flag'] = None

    general_dict = {
        "Start_Time": df_mag.index[0],
        "End_Time": df_mag.index[-1],
        "d": r_psp,
        "Fraction_missing_MAG": diagnostics['Mag']["Frac_miss"],
        "Fract_large_gaps": diagnostics['Mag']["Large_gaps"],
        "Resolution_MAG": diagnostics['Mag']["resol"],
        'part_flag': diagnostics['part_flag'],
        'qtn_flag': diagnostics['qtn_flag']
    }

    return general_dict
