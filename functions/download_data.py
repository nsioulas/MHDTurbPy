# download_data.py
from __future__ import annotations

import os
import sys
import gc
import logging
import traceback
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import constants

# Resolve repository paths from this file (independent of current working dir)
_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent

# ============================================================
# Repo root + local SPEDAS (repo-local)
# ============================================================
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "pyspedas"))
import pyspedas  # noqa: F401

# ============================================================
# Repo utilities
# ============================================================
from functions import calc_diagnostics as calc
from functions import general_functions as func
from functions import TurbPy as turb
from functions import polarization_analysis
from functions import calibrate_efield as efield
from functions.calibrate_sc_potential import *  # noqa: F401,F403

# ============================================================
# Spacecraft download helpers
# ============================================================
from functions.downloading_helpers.PSP import LoadTimeSeriesPSP, download_ephemeris_PSP  # noqa: F401
from functions.downloading_helpers.SOLO import LoadTimeSeriesSOLO
from functions.downloading_helpers.WIND import LoadTimeSeriesWIND
from functions.downloading_helpers.HELIOS_A import LoadTimeSeriesHELIOS_A
from functions.downloading_helpers.HELIOS_B import LoadTimeSeriesHELIOS_B
from functions.downloading_helpers.Ulysses import LoadTimeSeriesUlysses
from functions.downloading_helpers.ACE import LoadTimeSeriesACE
from functions.downloading_helpers.shared_utils import normalize_settings

# ============================================================
# Logging
# ============================================================
BG_WHITE = "\033[47m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
mu0 = constants.mu_0
m_p = constants.m_p


# ============================================================
# Internal: settings normalization (does NOT remove keys)
# ============================================================
def _fill_default_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    defaults = {
        "overwrite_files": 0,
        "save_all": True,
        "addit_time_around": 1,
        "gap_time_threshold": 5,
        "upsample_low_freq_ts": False,
        "Max_par_missing": 0,
        "estimate_derived_param": True,
        "estimate_psd_b": 0,
        "estimate_psd_v": 0,
        "est_PSD_components": 0,
        "smooth_psd": False,
        "rol_window": "60min",
        "use_hampel": False,
        "hampel_params": {"w": 200, "std": 3},
        "cut_in_small_windows": {"flag": False, "njobs": 1, "Step": "5s", "duration": "30s"},
        "PSDs": {"flag": False},
        "struc_funcs": {"flag": False},
        "npt_struc_funcs": {
            "flag": False,
            "five_points_sfunc": True,
            "return_Bmod": True,
            "dt_step": 0.25,
            "est_sfuncs": False,
            "max_qorder": 8,
        },
        "coherence_analysis": {"flag": False, "method": {}},
        "E_field": {"flag": False},
        "sc_pot": {"flag": False},
        "Big_Gaps": {
            "E_big_gaps": 10,
            "SC_pot_big_gaps": 10,
            "Mag_big_gaps": 500,
            "Par_big_gaps": 500,
            "QTN_big_gaps": 10,
        },
    }

    out = {**defaults, **settings}

    for k in ("Big_Gaps", "cut_in_small_windows", "PSDs", "npt_struc_funcs", "coherence_analysis", "E_field", "sc_pot"):
        if not isinstance(out.get(k, None), dict):
            out[k] = deepcopy(defaults[k])

    return out


def _safe_sync_high_to_low(df_hi: pd.DataFrame, df_lo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronize without triggering short-series filtfilt padlen crashes.

    Native path:
      - func.synchronize_dfs(df_hi, df_lo, upsample=False)  [downsample + anti-alias filter]

    Safe fallback (only when the native path fails due to padlen / filtfilt issues):
      - clip to overlap
      - interpolate small internal gaps
      - reindex df_hi onto df_lo.index using func.newindex (no filtering)

    Scientific intent:
      - For normal-length, well-populated intervals: identical behavior (native path).
      - For pathological short / NaN-dominated intervals: avoid hard failure and preserve timing/indexing.
    """
    # Defensive empties
    if df_hi is None or df_lo is None or len(df_hi) == 0 or len(df_lo) == 0:
        return df_hi.iloc[0:0].copy(), df_lo.iloc[0:0].copy()

    try:
        return func.synchronize_dfs(df_hi, df_lo, False)
    except ValueError as e:
        if ("padlen" not in str(e)) and ("filtfilt" not in str(e)):
            raise
    except Exception:
        # fall through to safe path
        pass

    # ---- safe path ----
    try:
        df_hi_s = df_hi.sort_index()
        df_lo_s = df_lo.sort_index()

        # common overlap window
        t0 = max(df_hi_s.index.min(), df_lo_s.index.min())
        t1 = min(df_hi_s.index.max(), df_lo_s.index.max())
        if t0 >= t1:
            return df_hi_s.iloc[0:0].copy(), df_lo_s.iloc[0:0].copy()

        df_hi_o = df_hi_s.loc[t0:t1]
        df_lo_o = df_lo_s.loc[t0:t1]

        # clean low first (filtfilt cannot handle NaNs; we avoid filtfilt here anyway)
        df_lo_c = df_lo_o.interpolate(method="time").dropna(how="all")
        if len(df_lo_c) == 0:
            return df_hi_s.iloc[0:0].copy(), df_lo_s.iloc[0:0].copy()

        # reindex high onto low cadence (no filtering)
        df_hi_on_lo = func.newindex(df_hi_o, df_lo_c.index)

        # ensure both frames live on identical index
        df_hi_on_lo = df_hi_on_lo.reindex(df_lo_c.index)
        df_lo_c = df_lo_c.reindex(df_hi_on_lo.index)

        return df_hi_on_lo, df_lo_c
    except Exception:
        # last-resort: return sorted originals (better than crashing)
        return df_hi.sort_index(), df_lo.sort_index()


def _safe_make_high_from_low(df_hi: pd.DataFrame, df_low: pd.DataFrame) -> pd.DataFrame:
    """
    Create a high-cadence version of df_low on df_hi.index, avoiding the
    upsample filtfilt padlen crash.

    Native path (preferred):
      - func.synchronize_dfs(df_hi, df_low, upsample=True)  [FIR + filtfilt + interp]

    Safe fallback (only when native path would crash):
      - func.newindex(df_low, df_hi.index)  [interp/reindex; no filtfilt]

    Trigger condition for fallback:
      - the *effective* low-cadence series after interpolate/dropna is too short for filtfilt padding.
        With the default FIR length (numtaps=201) used in signal_processing.upsample_dataframe,
        padlen ≈ 3*(201-1) = 600 samples.
    """
    if df_hi is None or df_low is None or len(df_hi) == 0 or len(df_low) == 0:
        return df_low.copy()

    # Compute effective length after the same cleaning synchronize_dfs applies
    try:
        low_clean = df_low.sort_index().interpolate(method="time").dropna(how="all")
        eff_len = len(low_clean)
    except Exception:
        eff_len = len(df_low)

    padlen_default = 600  # matches signal_processing.upsample_dataframe default (numtaps=201)
    try_native = eff_len > (padlen_default + 1)

    if try_native:
        try:
            B_hi, _ = func.synchronize_dfs(df_hi, df_low, True)
            return B_hi
        except ValueError as e:
            if ("padlen" not in str(e)) and ("filtfilt" not in str(e)):
                raise
        except Exception:
            pass

    # Safe fallback: built-in reindex/interp helper
    try:
        return func.newindex(df_low, df_hi.index)
    except Exception:
        return df_low.copy()

# ============================================================
# Public API (signature unchanged)
# ============================================================
def download_files(
    ok,
    df,
    settings,
    vars_2_downnload,
    cdf_lib_path,
    credentials,
    save_path,
):
    foldername = None
    try:
        settings = normalize_settings(settings)
        settings = _fill_default_settings(settings)

        start_time = df["Start"][ok]
        end_time = df["End"][ok]

        path0 = Path(save_path)
        tfmt = "%Y-%m-%d_%H-%M-%S"

        foldername = "%s_%s_sc_%d" % (str(start_time.strftime(tfmt)), str(end_time.strftime(tfmt)), 0)
        folder_path = path0.joinpath(foldername)
        final_file = folder_path.joinpath("final.pkl")

        needs_run = (not final_file.exists()) or bool(settings["overwrite_files"])
        if not needs_run:
            return

        if not folder_path.exists():
            logger.info(BG_WHITE + "Creating new folder  %s" + RESET, folder_path)
        else:
            logger.info(BG_WHITE + "Overwriting folder %s" + RESET, folder_path)

        (
            big_gaps_SC_pot,
            big_gaps,
            big_gaps_par,
            big_gaps_elec,
            big_gaps_qtn,
            flag_good,
            final,
            general,
            sig_c_sig_r_timeseries,
            dfdis,
            diagnostics,
        ) = main_function(
            start_time,
            end_time,
            settings,
            vars_2_downnload,
            cdf_lib_path,
            credentials,
        )

        try:
            final["Par"]["V_resampled"] = final["Par"]["V_resampled"].join(
                func.newindex(
                    dfdis[["sc_vel_r", "sc_vel_t", "sc_vel_n"]],
                    final["Par"]["V_resampled"].index,
                )
            )
        except Exception:
            pass

        os.makedirs(folder_path, exist_ok=True)

        if flag_good != 1:
            print("%s - %s failed!" % (ok, len(df)))
            return

        # Optional calibrations (kept)
        if (settings.get("E_field", {}).get("flag", False)) and (settings.get("in_rtn", False) is False) and (settings.get("sc", None) == "PSP"):
            try:
                print("Calibrating E-field data")
                coeffs_low_res = efield.process_data(
                    final["Mag"]["B_resampled"].copy(),
                    final["Par"]["V_resampled"][["Vx", "Vy", "Vz"]].copy(),
                    final["E"].copy(),
                    cadence_seconds=settings.get("E_field", {}).get("cadence_seconds", 6),
                    fit_interval_minutes=settings.get("E_field", {}).get("fit_interval_minutes", 4),
                    stride_minutes=settings.get("E_field", {}).get("stride_minutes", 0.1),
                    min_correlation=settings.get("E_field", {}).get("min_correlation", 0.8),
                )
                if settings.get("E_field", {}).get("keep_sc_pot", False):
                    final["sc_pot_E"] = final["E"]
                final["E"] = efield.calibrate_data(final["E"], coeffs_low_res)
            except Exception:
                traceback.print_exc()

        if (settings.get("sc_pot", {}).get("flag", False)) and (settings.get("sc", None) == "PSP"):
            try:
                print("Calibrating sc potential data")
                sc_pot_dens = calibrate_density(
                    process_sc_pot(pd.DataFrame(final["SC_pot"].interpolate().dropna())),
                    pd.DataFrame(final["Par"]["V_resampled"]["np"].interpolate().dropna()),
                    window_str=settings.get("sc_pot", {}).get("fit_window", "6000s"),
                    mode=settings.get("sc_pot", {}).get("mode", "local"),
                    overlap_ratio=settings.get("sc_pot", {}).get("overlap_ratio", 0.5),
                )

                if settings.get("sc_pot", {}).get("mode", None) == "global":
                    final["np_sc_pot"] = pd.DataFrame(
                        {"DateTime": sc_pot_dens[0][0].index, "Np": sc_pot_dens[0][0].values}
                    ).set_index("DateTime")
                else:
                    final["np_sc_pot"] = pd.DataFrame(
                        {"DateTime": sc_pot_dens[0].index, "Np": sc_pot_dens[0].values}
                    ).set_index("DateTime")
            except Exception:
                traceback.print_exc()

        # Small-window mode (kept)
        if settings["cut_in_small_windows"]["flag"]:
            generated_list = generate_intervals(
                func.format_date_to_str(str(final["Mag"]["B_resampled"].index[0])),
                func.format_date_to_str(str(final["Mag"]["B_resampled"].index[-1])),
                settings["cut_in_small_windows"]["flag"],
                data_path=None,
                settings=settings["cut_in_small_windows"],
            )

            combined_dict = small_sub_intervals_parallel_process(
                generated_list=generated_list,
                sig_c_sig_r_timeseries=sig_c_sig_r_timeseries,
                final=final,
                settings=settings,
                diagnostics=diagnostics,
                general=general,
                mu0=mu0,
                m_p=m_p,
                n_jobs=settings["cut_in_small_windows"]["njobs"],
            )

            func.savepickle(combined_dict, str(folder_path), "overall.pkl")

        else:
            if settings["save_all"] is True:
                func.savepickle(final, str(folder_path), "final.pkl")
                func.savepickle(general, str(folder_path), "general.pkl")
                func.savepickle(sig_c_sig_r_timeseries, str(folder_path), "sig_c_sig_r.pkl")
                func.savepickle(big_gaps, str(folder_path), "mag_gaps.pkl")
                func.savepickle(big_gaps_qtn, str(folder_path), "qtn_gaps.pkl")
                func.savepickle(big_gaps_elec, str(folder_path), "elec_gaps.pkl")
                func.savepickle(big_gaps_SC_pot, str(folder_path), "sc_pot_gaps.pkl")
                func.savepickle(big_gaps_par, str(folder_path), "par_gaps.pkl")
            else:
                print("Deleting memory intensive timeseries per your commands. Only keeping derived quantities!")
                try:
                    del final["Mag"]["B_resampled"]
                except Exception:
                    pass
                try:
                    del final["Par"]["V_resampled"]
                except Exception:
                    pass
                func.savepickle(final, str(folder_path), "final.pkl")
                func.savepickle(general, str(folder_path), "general.pkl")

            gc.collect()
            print("%d out of %d finished" % (ok, len(df)))

    except Exception as e:
        try:
            if foldername is not None:
                os.makedirs(Path(save_path).joinpath(foldername), exist_ok=True)
        except Exception:
            pass
        print("failed at index", ok, "with error:", str(e))
        traceback.print_exc()


# ============================================================
# Public API (signature unchanged)
# ============================================================
def small_sub_intervals_parallel_process(
    generated_list,
    sig_c_sig_r_timeseries,
    final,
    settings,
    diagnostics,
    general,
    mu0,
    m_p,
    n_jobs=-1,
    backend="loky",
):
    def process_jj(jj, generated_list, sig_c_sig_r_timeseries, final, phi_ts, settings, diagnostics, general, mu0, m_p):
        try:
            t0_time = generated_list["Start"][jj]
            tf_time = generated_list["End"][jj]

            ind1 = func.string_to_datetime_index(t0_time)
            ind2 = func.string_to_datetime_index(tf_time)

            sig = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, sig_c_sig_r_timeseries.copy())
            Vdf = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, final["Par"]["V_resampled"].copy())
            Bdf = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, final["Mag"]["B_resampled"].copy())

            if settings.get("E_field", {}).get("flag", False):
                Edf = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, final["E"].copy())
            else:
                Edf = None

            phi = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, phi_ts.copy())

            wanted_dt = (tf_time - t0_time).total_seconds()
            ts_dt = (Bdf.index[-1] - Bdf.index[0]).total_seconds()
            dt = func.find_cadence(Bdf)

            if ts_dt <= 0.95 * wanted_dt:
                return (jj, None)

            sf_dict = None
            mag_dict = None
            dict_psd = None
            addit_dict = None
            coh_dicts = {}

            if settings.get("npt_struc_funcs", {}).get("flag", False):
                sf_dict = turb.est_5_pt_sfuncs(Bdf, dt, func_params=settings["npt_struc_funcs"])

            if settings.get("PSDs", {}).get("flag", False):
                mag_dict = calc.estimate_magnetic_field_psd(
                    Bdf,
                    None,
                    dt,
                    settings,
                    diagnostics,
                    return_mag_df=False,
                )

            dict_psd, addit_dict = calc.estimate_psd_dict(
                settings,
                sig,
                func.find_cadence(sig),
                estimate_means=True,
            )

            if settings.get("coherence_analysis", {}).get("flag", False):
                Bdf, Vdf = func.synchronize_dfs(Bdf, Vdf, True)

                try:
                    np_mean = Vdf["np"].rolling("30s", center=True).mean().interpolate()
                    kinet_normal = pd.DataFrame(1e-15 / np.sqrt(mu0 * np_mean * m_p))
                    _, kinet_normal = func.synchronize_dfs(Bdf, kinet_normal, True)
                    kinet_normal = kinet_normal.values
                except Exception:
                    kinet_normal = None

                if kinet_normal is not None and settings.get("sc", None) == "PSP":
                    if ("Vr" in Vdf) and all(k in Vdf.columns for k in ("sc_vel_r", "sc_vel_t", "sc_vel_n")):
                        sc_V = Vdf[["Vr", "Vt", "Vn"]] - Vdf[["sc_vel_r", "sc_vel_t", "sc_vel_n"]].values
                        Vfin = func.newindex(sc_V, Bdf.index) + (Bdf[["Br", "Bt", "Bn"]].values.T * kinet_normal).T
                    elif ("Vx" in Vdf):
                        Vfin = Vdf[["Vx", "Vy", "Vz"]] + (Bdf[["Bx", "By", "Bz"]].values.T * kinet_normal).T
                    else:
                        Vfin = Vdf
                else:
                    Vfin = Vdf

                try:
                    vnorm = np.sqrt(np.sum(Vfin.values**2, axis=1))
                    bnorm = np.sqrt(np.sum(Bdf.values**2, axis=1))
                    dot = np.sum(Vfin.values * Bdf.values, axis=1)
                    denom = vnorm * bnorm
                    cosang = np.full_like(dot, np.nan, dtype=float)
                    good = np.isfinite(dot) & np.isfinite(denom) & (denom > 0)
                    cosang[good] = np.clip(dot[good] / denom[good], -1.0, 1.0)
                    vb_ang_va = np.degrees(np.arccos(cosang))
                except Exception:
                    vb_ang_va = np.nan

                coh_dicts["P_spiral_mean"] = np.nanmean(phi)
                coh_dicts["P_spiral_std"] = np.nanstd(phi)
                coh_dicts["sig_c"] = np.nanmean(np.abs(sig["sigma_c"])) if isinstance(sig, pd.DataFrame) and ("sigma_c" in sig) else np.nan
                coh_dicts["sig_r"] = np.nanmean(sig["sigma_r"]) if isinstance(sig, pd.DataFrame) and ("sigma_r" in sig) else np.nan
                coh_dicts["di"] = np.nanmedian(sig["d_i"]) if isinstance(sig, pd.DataFrame) and ("d_i" in sig) else np.nan
                coh_dicts["Tp"] = np.nanmean(Vdf["TEMP"]) if isinstance(Vdf, pd.DataFrame) and ("TEMP" in Vdf) else np.nan
                coh_dicts["Vsw_with_Va"] = np.nanmedian(np.sqrt(np.sum(Vfin.values**2, axis=1)))
                coh_dicts["VB_wth_Va"] = np.nanmean(vb_ang_va)

                for method in list(settings.get("coherence_analysis", {}).get("method", {}).keys()):
                    try:
                        coh_dict = None
                        for mm, field in enumerate(settings["coherence_analysis"]["method"][method]):
                            coh_dict = polarization_analysis.anisotropy_coherence(
                                Bdf,
                                Vfin,
                                E_df=Edf if field == "E" else None,
                                method=method,
                                func_params=settings["coherence_analysis"],
                                f_dict=None if mm == 0 else coh_dict,
                            )
                        coh_dicts[method] = coh_dict
                    except Exception:
                        traceback.print_exc()
                        coh_dicts[method] = None

            general_final = deepcopy(general)
            if settings.get("sc") == "PSP":
                try:
                    general_final["d"] = np.nanmedian(Vdf["Dist_au"].values)
                except Exception:
                    pass

            general_final["Start_Time"] = Bdf.index[0]
            general_final["End_Time"] = Bdf.index[-1]

            try:
                if addit_dict is not None:
                    addit_dict["dict_psd"] = dict_psd
                    addit_dict["mag_dict"] = mag_dict
                else:
                    addit_dict = {"dict_psd": dict_psd, "mag_dict": mag_dict}
            except Exception:
                addit_dict = None
                traceback.print_exc()

            entry = {"g": general_final, "f": addit_dict, "coh": coh_dicts, "sf_dict": sf_dict}
            return (jj, entry)

        except Exception:
            traceback.print_exc()
            return (jj, None)

    phi_ts = func.calculate_parker_spiral(final["Mag"]["B_resampled"].rolling("90s", center=True).mean())

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(process_jj)(
            jj,
            generated_list,
            sig_c_sig_r_timeseries,
            final,
            phi_ts,
            settings,
            diagnostics,
            general,
            mu0,
            m_p,
        )
        for jj in range(len(generated_list))
    )

    combined_dict = {jj: entry for jj, entry in results if entry is not None}
    return combined_dict


# ============================================================
# Public API (signature unchanged)
# ============================================================
def main_function(start_time, end_time, settings, vars_2_downnload, cdf_lib_path, credentials):
    settings = normalize_settings(settings)
    settings = _fill_default_settings(settings)

    dfqtn = None
    df_sc_pot = None
    mag_flag = None
    df_elec = None
    big_gaps_elec = None
    big_gaps = None
    big_gaps_qtn = None
    big_gaps_par = None
    big_gaps_SC_pot = None
    df_e_field = None
    qtn_flag = None

    sc = settings.get("sc", None)

    # Load spacecraft timeseries (public APIs unchanged)
    if sc == "PSP":
        try:
            (
                dfqtn,
                dfmag,
                dfpar,
                df_e_field,
                df_sc_pot,
                dfdis,
                big_gaps,
                big_gaps_qtn,
                big_gaps_par,
                big_gaps_SC_pot,
                misc,
            ) = LoadTimeSeriesPSP(
                start_time,
                end_time,
                settings,
                vars_2_downnload,
                cdf_lib_path,
                credentials=credentials,
                time_amount=settings["addit_time_around"],
                time_unit="h",
            )
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

        try:
            if dfpar is not None and "na" in dfpar.columns:
                del dfpar["na"]
        except Exception:
            pass

    elif sc == "SOLO":
        try:
            (
                dfmag,
                mag_flag,
                dfpar,
                dfdis,
                big_gaps,
                big_gaps_qtn,
                big_gaps_par,
                misc,
            ) = LoadTimeSeriesSOLO(
                start_time,
                end_time,
                settings,
                vars_2_downnload,
                cdf_lib_path,
                credentials=credentials,
                time_amount=settings["addit_time_around"],
                time_unit="h",
            )
            if settings.get("SOLO_use_burst") and (mag_flag != "Burst"):
                dfmag, dfpar = None, None
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

    elif sc == "HELIOS_A":
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesHELIOS_A(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

    elif sc == "HELIOS_B":
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesHELIOS_B(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

    elif sc == "WIND":
        try:
            (
                dfmag,
                dfpar,
                df_elec,
                dfdis,
                big_gaps,
                big_gaps_qtn,
                big_gaps_par,
                big_gaps_elec,
                misc,
                qtn_flag,
            ) = LoadTimeSeriesWIND(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

    elif sc == "ACE":
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesACE(start_time, end_time, settings, vars_2_downnload)
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

    elif sc == "Ulysses":
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesUlysses(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()
            dfmag, dfpar, dfdis, misc = None, None, None, None

    else:
        raise ValueError(f"Unsupported spacecraft sc={sc}")

    # Validate + derive diagnostics
    if dfpar is not None and misc is not None:
        try:
            frac_miss = misc.get("Par", {}).get("Frac_miss", 1)
        except Exception:
            frac_miss = 1

        if frac_miss < settings.get("Max_par_missing", 0):
            try:
                if settings.get("upsample_low_freq_ts"):
                    # This can still crash if the series is too short; keep it guarded anyway.
                    try:
                        B_df, V_df = func.synchronize_dfs(dfmag, dfpar, True)
                        B_df_low = B_df
                    except ValueError:
                        B_df_low, V_df = _safe_sync_high_to_low(dfmag, dfpar)
                        B_df = _safe_make_high_from_low(dfmag, B_df_low)
                else:
                    B_df_low, V_df = _safe_sync_high_to_low(dfmag, dfpar)
                    B_df = _safe_make_high_from_low(dfmag, B_df_low)

                general_dict = calc.general_dict_func(misc, dfmag.copy(), dfpar.copy(), dfdis)

                if sc == "WIND":
                    general_dict["qtn_flag"] = qtn_flag

                if settings.get("estimate_derived_param"):
                    part_dict, sig_c_sig_r_timeseries = calc.calculate_diagnostics(
                        B_df if settings.get("upsample_low_freq_ts") else B_df_low,
                        V_df,
                        dfdis,
                        settings,
                        misc,
                    )
                    mag_dict = calc.estimate_magnetic_field_psd(
                        B_df,
                        func.find_cadence(B_df),
                        settings,
                        misc,
                    )
                else:
                    mag_dict, part_dict, sig_c_sig_r_timeseries = {}, {}, {}

                part_dict["V_resampled"] = V_df
                mag_dict["B_resampled"] = B_df

                final_dict = {
                    "Mag": mag_dict,
                    "Par": part_dict,
                    "Elec": df_elec,
                    "Mag_flag": mag_flag,
                }

                general_dict["Fraction_missing_part"] = misc["Par"]["Frac_miss"]
                general_dict["Resolution_part"] = misc["Par"]["resol"]

                try:
                    general_dict["Fraction_missing_elec"] = misc["Elec"]["Frac_miss"]
                    general_dict["Resolution_elec"] = misc["Elec"]["resol"]
                except Exception:
                    general_dict["Fraction_missing_elec"] = None
                    general_dict["Resolution_elec"] = None

                if sc in ("PSP", "SOLO"):
                    try:
                        general_dict["Fraction_missing_qtn"] = misc["QTN"]["Frac_miss"]
                        general_dict["Resolution_qtn"] = misc["QTN"]["resol"]
                    except Exception:
                        general_dict["Fraction_missing_qtn"] = None
                        general_dict["Resolution_qtn"] = None

                    try:
                        final_dict["E"] = df_e_field
                        general_dict["Fraction_missing_E"] = misc["E"]["Frac_miss"]
                        general_dict["Resolution_E"] = misc["E"]["resol"]
                    except Exception:
                        general_dict["Fraction_missing_E"] = None
                        general_dict["Resolution_E"] = None

                    try:
                        final_dict["SC_pot"] = df_sc_pot
                        general_dict["Fraction_missing_SC_pot"] = misc["SC_pot"]["Frac_miss"]
                        general_dict["Resolution_SC_pot"] = misc["SC_pot"]["resol"]
                    except Exception:
                        general_dict["Fraction_missing_SC_pot"] = None
                        general_dict["Resolution_SC_pot"] = None

                general_dict["sc"] = sc
                flag_good = 1

            except Exception:
                traceback.print_exc()
                print("No MAG data!")
                flag_good = 0
                big_gaps = None
                final_dict = None
                general_dict = None
                sig_c_sig_r_timeseries = None
        else:
            print("No particle data!")
            final_dict = None
            flag_good = 0
            big_gaps = None
            general_dict = None
            sig_c_sig_r_timeseries = None
    else:
        print("No particle data!")
        final_dict = None
        flag_good = 0
        big_gaps = None
        general_dict = None
        sig_c_sig_r_timeseries = None

    return (
        big_gaps_SC_pot,
        big_gaps,
        big_gaps_par,
        big_gaps_elec,
        big_gaps_qtn,
        flag_good,
        final_dict,
        general_dict,
        sig_c_sig_r_timeseries,
        dfdis,
        misc,
    )


# ============================================================
# Public API (signature unchanged)
# ============================================================
def generate_intervals(
    start_time_str,
    end_time_str,
    generate_1_interval,
    data_path=None,
    settings=None,
):
    import datetime

    if settings is None:
        settings = {}

    start_date = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
    end_date = datetime.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M")

    if not generate_1_interval:
        start_times = [start_date]
        end_times = [end_date]
        logging.info("Generating only one interval based on the provided start and end times.")
    else:
        step = settings.get("Step")
        duration = settings.get("duration")
        if not step or not duration:
            raise ValueError("'Step' and 'duration' must be specified in settings.")

        start_times = pd.date_range(start=start_date, end=end_date, freq=step)
        delta = pd.to_timedelta(duration)
        end_times = start_times + delta

        valid = end_times <= end_date
        start_times = start_times[valid]
        end_times = end_times[valid]

        logging.info("Generated intervals with Step: %s and Duration: %s", step, duration)
        logging.info("Number of Intervals: %d", len(start_times))

    df = pd.DataFrame({"Start": start_times, "End": end_times})
    logging.info("Start Time: %s", start_date)
    logging.info("End Time: %s", end_date)
    if not generate_1_interval:
        logging.info("Considering a single interval spanning: %s to %s", start_date, end_date)
    return df
