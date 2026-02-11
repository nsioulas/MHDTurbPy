"""
PSP.py

Clean, pythonic rewrite of the PSP download helper while preserving:

- Public API (signatures unchanged)SCM
- Return ordering unchanged
- Output DataFrame contracts unchanged
- "misc" dict key structure unchanged
- QTN / ephemeris / MAG / SCAM logic unchanged in meaning
- Gap + diagnostics behavior unchanged in meaning
- Optional SCAM wheel-noise removal (now centralized & consistent)

This module is intentionally tidy and readable.
"""

from __future__ import annotations

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import constants

# ------------------------------------------------------------
# Local SPEDAS
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pyspedas"))
import pyspedas
from pytplot import get_data
from project_config import repo_data_file

# ------------------------------------------------------------
# Your repo utilities (must exist)
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(REPO_ROOT / "functions"))
import general_functions as func
import TurbPy as turb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(REPO_ROOT / "functions" / "downloading_helpers"))
# ------------------------------------------------------------
# Shared utilities (lightweight, single authority)
# ------------------------------------------------------------
from shared_utils import (
    diag_default,
    keep_diag_keys,
    ensure_datetime_index,
    sanitize_timeseries_df,
    clip_to_requested,
    resolve_mag_noise_settings,
    apply_optional_wheel_noise_removal,
    normalize_settings,
)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Constants (kept)
# ------------------------------------------------------------
mu0 = constants.mu_0
mu_0 = constants.mu_0
m_p = constants.m_p
kb = constants.k

au_to_km = 1.496e8
_MAG_HIGHRES_THRESHOLD = 230  # kept


# ============================================================
# 1) Settings defaults + validation (public behavior preserved)
# ============================================================
def init_psp_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Central PSP defaults initializer.
    Does NOT remove legacy keys, only fills missing ones.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    settings = normalize_settings(settings)

    if "Data_path" not in settings:
        raise KeyError("settings must include 'Data_path'")

    defaults = {
        "particle_mode": "9th_perih_cut",
        "part_resol": 900,
        "MAG_resol": 1,
        "use_local_data": False,
        "in_rtn": True,
        "must_have_qtn": False,
        "max_PSP_dist": None,
        "allow_max_SWEAP_distance": False,
        "max_SWEAP_distance": 0.25,
        "span_key": "spi_sf00",
        "apply_hampel": True,
        "hampel_params": {"w": 100, "std": 3},
        "Big_Gaps": {
            "Mag_big_gaps": 10,
            "Par_big_gaps": 10,
            "QTN_big_gaps": 10,
            "E_big_gaps": 10,
            "SC_pot_big_gaps": 10,
        },
        "Mag_SCAM_PSP": {
            "flag": False,
            "noise_flag": False,
            "noise_removal": {
                "window_size": 2048,
                "avg_length": 16,
                "power_threshold": 6.0,
                "freq_min": 10.0,
                "hampel_wind": 51,
                "hampel_thresh": 3.5,
            },
        },
        # Unified option (preferred); legacy keys still supported via resolve_mag_noise_settings()
        "Mag_SCM": {
            "use_SCM"   : False,
            "noise_flag": False,
            "noise_removal": {
                "window_size": 2048,
                "avg_length": 16,
                "power_threshold": 6.0,
                "freq_min": 10.0,
                "hampel_wind": 51,
                "hampel_thresh": 3.5,
            },
        },
    }

    out = {**defaults, **settings}

    # enforce dict sub-structures exist
    if not isinstance(out.get("Big_Gaps", None), dict):
        out["Big_Gaps"] = defaults["Big_Gaps"]
    if not isinstance(out.get("Mag_SCAM_PSP", None), dict):
        out["Mag_SCAM_PSP"] = defaults["Mag_SCAM_PSP"]
    if not isinstance(out["Mag_SCAM_PSP"].get("noise_removal", None), dict):
        out["Mag_SCAM_PSP"]["noise_removal"] = defaults["Mag_SCAM_PSP"]["noise_removal"]

    if not isinstance(out.get("Mag_SCM", None), dict):
        out["Mag_SCM"] = defaults["Mag_SCM"]
    if not isinstance(out["Mag_SCM"].get("noise_removal", None), dict):
        out["Mag_SCM"]["noise_removal"] = defaults["Mag_SCM"]["noise_removal"]

    return out


# ============================================================
# 2) Variable defaults (public API unchanged)
# ============================================================
def default_variables_to_download_PSP(vars_2_downnload: Dict[str, Any]):
    """
    Must preserve your default variable contracts.
    """
    varnames_MAG = ["B_RTN"] if vars_2_downnload.get("mag", None) is None else vars_2_downnload["mag"]
    varnames_QTN = (
        ["electron_density", "electron_core_temperature"]
        if vars_2_downnload.get("qtn", None) is None
        else vars_2_downnload["qtn"]
    )
    varnames_SPAN = (
        ["DENS", "VEL_RTN_SUN", "TEMP", "SUN_DIST", "SC_VEL_RTN_SUN"]
        if vars_2_downnload.get("span", None) is None
        else vars_2_downnload["span"]
    )
    varnames_SPC = (
        ["np_moment", "wp_moment", "vp_moment_RTN", "sc_pos_HCI", "carr_longitude", "general_flag"]
        if vars_2_downnload.get("spc", None) is None
        else vars_2_downnload["spc"]
    )
    varnames_SPAN_alpha = ["DENS"] if vars_2_downnload.get("span-a", None) is None else vars_2_downnload["span-a"]
    varnames_EPHEM = ["position", "velocity"] if vars_2_downnload.get("ephem", None) is None else vars_2_downnload["ephem"]

    varnames_E_field = ["psp_fld_l2_dfb_wf_dVdc_sc"] if vars_2_downnload.get("E_field", False) else None
    varnames_SC_pot = ["dfb_wf_vdc"] if vars_2_downnload.get("sc_pot", False) else None

    return (
        varnames_MAG,
        varnames_QTN,
        varnames_SPAN,
        varnames_SPC,
        varnames_SPAN_alpha,
        varnames_EPHEM,
        varnames_E_field,
        varnames_SC_pot,
    )


# ============================================================
# 3) Column mappings (public API unchanged)
# ============================================================
def map_col_names_PSP(instrument: str, varnames: List[str]) -> List[List[str]]:
    fields_MAG_cols = {
        "mag_RTN_4_Sa_per_Cyc": ["Br", "Bt", "Bn"],
        "mag_SC_4_Sa_per_Cyc": ["Bx", "By", "Bz"],
        "mag_rtn_4_per_cycle": ["Br", "Bt", "Bn"],
        "mag_sc_4_per_cycle": ["Bx", "By", "Bz"],
        "mag_RTN": ["Br", "Bt", "Bn"],
        "mag_SC": ["Bx", "By", "Bz"],
        "mag_rtn": ["Br", "Bt", "Bn"],
        "mag_sc": ["Bx", "By", "Bz"],
        "psp_fld_l2_dfb_wf_dVdc_sc": ["dvx", "dvy"],
        "dfb_wf_vdc": [
            "psp_fld_l2_dfb_wf_V1dc",
            "psp_fld_l2_dfb_wf_V2dc",
            "psp_fld_l2_dfb_wf_V3dc",
            "psp_fld_l2_dfb_wf_V4dc",
        ],
    }

    fields_QTN_cols = {
        "electron_density": ["ne_qtn"],
        "electron_core_temperature": ["Te_qtn"],
    }

    spc_cols = {
        "np_moment": ["np"],
        "wp_moment": ["Vth"],
        "vp_moment_RTN": ["Vr", "Vt", "Vn"],
        "vp_moment_SC": ["Vx", "Vy", "Vz"],
        "sc_pos_HCI": ["sc_x", "sc_y", "sc_z"],
        "sc_vel_HCI": ["sc_vel_x", "sc_vel_y", "sc_vel_z"],
        "carr_latitude": ["carr_lat"],
        "carr_longitude": ["carr_lon"],
        "general_flag": ["flag"],
    }

    span_cols = {
        "DENS": ["np"],
        "VEL_SC": ["Vx", "Vy", "Vz"],
        "VEL_RTN_SUN": ["Vr", "Vt", "Vn"],
        "TEMP": ["TEMP"],
        "SUN_DIST": ["Dist_au"],
        "SC_VEL_RTN_SUN": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
    }

    span_alpha_cols = {"DENS": ["na"]}

    ephem_cols = {
        "position": ["sc_pos_r", "sc_pos_t", "sc_pos_n"],
        "velocity": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
    }

    if instrument == "SPC":
        return [spc_cols[v] for v in varnames if v in spc_cols]
    if instrument == "FIELDS-MAG":
        return [fields_MAG_cols[v] for v in varnames if v in fields_MAG_cols]
    if instrument == "QTN":
        return [fields_QTN_cols[v] for v in varnames if v in fields_QTN_cols]
    if instrument == "SPAN":
        return [span_cols[v] for v in varnames if v in span_cols]
    if instrument == "SPAN-alpha":
        return [span_alpha_cols[v] for v in varnames if v in span_alpha_cols]
    if instrument == "EPHEMERIS":
        return [ephem_cols[v] for v in varnames if v in ephem_cols]
    return []


# ============================================================
# 4) Internal: tplot->DataFrame conversion
# ============================================================
def _tplot_to_df(tplot_var: str, columns: List[str]) -> pd.DataFrame:
    arr = get_data(tplot_var)
    if arr is None:
        return pd.DataFrame()

    times = getattr(arr, "times", None)
    y = getattr(arr, "y", None)
    if times is None or y is None:
        try:
            times, y = arr[0], arr[1]
        except Exception:
            return pd.DataFrame()

    try:
        df = pd.DataFrame(index=times, data=y)
        if df.shape[1] == len(columns):
            df.columns = columns
        else:
            if len(columns) >= df.shape[1]:
                df.columns = columns[: df.shape[1]]
            else:
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
        return ensure_datetime_index(df)
    except Exception:
        return pd.DataFrame()


# ============================================================
# 5) MAG (fluxgate) download
# ============================================================
def download_MAG_FIELD_PSP(t0, t1, credentials, varnames, settings):
    """
    Public helper used internally.
    """
    try:
        dfmag = pd.DataFrame()

        for v in varnames:
            if v == "B_RTN":
                datatype_private = "mag_RTN_4_Sa_per_Cyc" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_RTN"
                datatype_public = "mag_rtn_4_per_cycle" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_rtn"
            else:
                datatype_private = "mag_SC_4_Sa_per_Cyc" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_SC"
                datatype_public = "mag_sc_4_per_cycle" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_sc"

            used_datatype = None
            MAGdata = None

            try:
                username = credentials["psp"]["fields"]["username"]
                password = credentials["psp"]["fields"]["password"]
                MAGdata = pyspedas.psp.fields(
                    trange=[t0, t1],
                    datatype=datatype_private,
                    level="l2",
                    time_clip=True,
                    username=username,
                    password=password,
                    no_update=settings["use_local_data"],
                )
                if MAGdata and len(MAGdata) > 0:
                    used_datatype = datatype_private
            except Exception:
                MAGdata = None
                used_datatype = None

            if not MAGdata or len(MAGdata) == 0:
                MAGdata = pyspedas.psp.fields(
                    trange=[t0, t1],
                    datatype=datatype_public,
                    level="l2",
                    time_clip=True,
                    no_update=settings["use_local_data"],
                )
                if MAGdata and len(MAGdata) > 0:
                    used_datatype = datatype_public

            if not MAGdata or len(MAGdata) == 0 or used_datatype is None:
                continue

            cols_map = map_col_names_PSP("FIELDS-MAG", [used_datatype])
            cols = cols_map[0] if cols_map else (["Br", "Bt", "Bn"] if v == "B_RTN" else ["Bx", "By", "Bz"])

            df_part = _tplot_to_df(MAGdata[0], cols)
            dfmag = dfmag.join(df_part, how="outer") if len(dfmag) else df_part

        dfmag = sanitize_timeseries_df(dfmag)
        return dfmag

    except Exception:
        logger.exception("Error while retrieving PSP MAG data.")
        return None


def LoadSCAMFromSPEDAS_PSP(t0, t1, credentials, settings):
    """
    Public API used internally.
    """
    try:
        username = credentials["psp"]["fields"]["username"]
        password = credentials["psp"]["fields"]["password"]

        if settings.get("in_rtn", True):
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1],
                datatype="merged_scam_wf",
                varnames=["psp_fld_l3_merged_scam_wf_RTN"],
                level="l3",
                time_clip=1,
                downloadonly=False,
                username=username,
                password=password,
                no_update=settings["use_local_data"],
            )
            cols = ["Br", "Bt", "Bn"]
        else:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1],
                datatype="merged_scam_wf",
                varnames=["psp_fld_l3_merged_scam_wf_SC"],
                level="l3",
                time_clip=1,
                downloadonly=False,
                username=username,
                password=password,
                no_update=settings["use_local_data"],
            )
            cols = ["Bx", "By", "Bz"]

        if not scam_vars or len(scam_vars) == 0:
            return None

        arr = get_data(scam_vars[0])
        if arr is None:
            return None

        df = pd.DataFrame(index=arr.times, data=arr.y, columns=cols)
        return sanitize_timeseries_df(df)

    except Exception:
        traceback.print_exc()
        return None


def process_mag_field_data(t0, t1, settings, credentials, varnames_MAG, ind1, ind2):
    """
    Returns:
    (dfmag_clipped, big_gaps_mag, diagnostics_MAG)
    """
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        if settings["Mag_SCM"]["use_SCM"]:
            logger.info("PSP MAG: using SCAM")
            dfmag = LoadSCAMFromSPEDAS_PSP(t0, t1, credentials, settings)
        else:
            logger.info("PSP MAG: using fluxgate")
            dfmag = download_MAG_FIELD_PSP(t0, t1, credentials, varnames_MAG, settings)

        dfmag = sanitize_timeseries_df(dfmag)
        if dfmag is None:
            return None, None, diag_default()

        dfmag = clip_to_requested(dfmag, ind1, ind2, req_start, req_end, func_module=func)
        dfmag = sanitize_timeseries_df(dfmag)
        if dfmag is None:
            return None, None, diag_default()

        big_gaps = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], str(ind1), str(ind2))
        diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)
        diagnostics_MAG.setdefault("resampled_df", None)

        # ======================================================
        # OPTIONAL wheel-noise removal (unified config resolver)
        # Applied at resampled_df stage only (matches SOLO)
        # ======================================================
        try:
            noise_flag, noise_cfg = resolve_mag_noise_settings(settings)
            if noise_flag and isinstance(diagnostics_MAG.get("resampled_df", None), pd.DataFrame):
                dt = func.find_cadence(diagnostics_MAG["resampled_df"])
                logger.info("PSP MAG: wheel-noise removal enabled (resampled_df stage)")
                diagnostics_MAG["resampled_df"] = apply_optional_wheel_noise_removal(
                    resampled_df=diagnostics_MAG["resampled_df"],
                    cadence_seconds=dt,
                    remove_wheel_noise_func=turb.remove_wheel_noise,
                    noise_cfg=noise_cfg,
                    logger=logger,
                )
        except Exception:
            traceback.print_exc()

        return dfmag, big_gaps, diagnostics_MAG

    except Exception:
        logger.exception("PSP MAG processing failed.")
        return None, None, diag_default()


# ============================================================
# 6) SPC
# ============================================================
def download_SPC_PSP(t0, t1, credentials, varnames, settings):
    try:
        spcdata = None

        # private
        try:
            username = credentials["psp"]["sweap"]["username"]
            password = credentials["psp"]["sweap"]["password"]
            spcdata = pyspedas.psp.spc(
                trange=[t0, t1],
                datatype="l3i",
                level="L3",
                varnames=varnames,
                time_clip=True,
                username=username,
                password=password,
                no_update=settings["use_local_data"],
            )
        except Exception:
            spcdata = None

        # public fallback
        if not spcdata or len(spcdata) == 0:
            spcdata = pyspedas.psp.spc(
                trange=[t0, t1],
                datatype="l3i",
                level="l3",
                varnames=varnames,
                time_clip=True,
                no_update=settings["use_local_data"],
            )

        if not spcdata or len(spcdata) == 0:
            return None

        col_names = map_col_names_PSP("SPC", varnames)
        dfs = []
        for i, v in enumerate(spcdata):
            cols = col_names[i] if i < len(col_names) else [f"spc_{i}"]
            arr = get_data(v)
            if arr is None:
                continue
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        df = pd.concat(dfs, axis=1)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None

        if {"sc_x", "sc_y", "sc_z"}.issubset(df.columns):
            df["Dist_au"] = np.sqrt((df[["sc_x", "sc_y", "sc_z"]] ** 2).sum(axis=1)) / au_to_km
            df.drop(["sc_x", "sc_y", "sc_z"], axis=1, inplace=True)

        return df

    except Exception:
        logger.exception("PSP SPC download failed.")
        return None


def process_spc_data(t0, t1, credentials, varnames_SPC, settings, ind1=None, ind2=None):
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        dfspc = download_SPC_PSP(t0, t1, credentials, varnames_SPC, settings)
        dfspc = sanitize_timeseries_df(dfspc)
        if dfspc is None:
            return None, diag_default(), "No SPC", None

        dfspc = clip_to_requested(dfspc, ind1, ind2, req_start, req_end, func_module=func)
        dfspc = sanitize_timeseries_df(dfspc)
        if dfspc is None:
            return None, diag_default(), "No SPC", None

        if settings.get("apply_hampel", False):
            cols = ["Vr", "Vt", "Vn", "np", "Vth"] if "Vr" in dfspc.columns else ["Vx", "Vy", "Vz", "Fnp", "Vth"]
            ws = settings["hampel_params"]["w"]
            nn = settings["hampel_params"]["std"]
            for c in cols:
                if c not in dfspc.columns:
                    continue
                try:
                    out_idx = func.hampel(dfspc[c], window_size=ws, n=nn)
                    if isinstance(out_idx, tuple) and len(out_idx) == 2:
                        out_idx = out_idx[1]
                    dfspc.loc[dfspc.index[out_idx], c] = np.nan
                except Exception:
                    traceback.print_exc()


        # Tp estimation kept (best-effort)
        try:
            from astropy.constants import m_p as mp_ast, k_B
            from astropy import units as u

            dfspc["Tp"] = np.array(
                ((mp_ast * ((dfspc["Vth"].values * u.km / u.s).to(u.m / u.s) ** 2)) / (2 * k_B)).to(
                    u.eV, equivalencies=u.temperature_energy()
                )
            )
        except Exception:
            pass

        big_gaps_spc = func.find_big_gaps(dfspc, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
        diagnostics_SPC = func.resample_timeseries_estimate_gaps(dfspc, settings["part_resol"], large_gaps=10)
        diagnostics_SPC.setdefault("resampled_df", None)

        return dfspc, diagnostics_SPC, "SPC", big_gaps_spc

    except Exception:
        logger.exception("PSP SPC processing failed.")
        return None, diag_default(), "No SPC", None


# ============================================================
# 7) SPAN
# ============================================================
def download_SPAN_PSP(t0, t1, credentials, varnames, varnames_alpha, settings):
    try:
        span_key = settings.get("span_key", "spi_sf00").lower()
        use_local = settings.get("use_local_data", False)

        products = (
            [span_key, "spi_sf00_l3_mom" if span_key == "spi_sf00" else "spi_sf00"]
            if span_key in {"spi_sf00", "spi_sf00_l3_mom"}
            else ["spi_sf00", "spi_sf00_l3_mom"]
        )

        spandata = None
        for key in products:
            try:
                if key == "spi_sf00_l3_mom":
                    qvars = [f"psp_spi_{v}" for v in varnames]
                    spandata = pyspedas.psp.spi(
                        trange=[t0, t1],
                        datatype="spi_sf00_l3_mom",
                        level="l3",
                        varnames=qvars,
                        time_clip=True,
                        no_update=use_local,
                    )
                else:
                    user = credentials["psp"]["sweap"]["username"]
                    pwd = credentials["psp"]["sweap"]["password"]
                    spandata = pyspedas.psp.spi(
                        trange=[t0, t1],
                        datatype="spi_sf00",
                        level="L3",
                        varnames=varnames,
                        time_clip=True,
                        username=user,
                        password=pwd,
                        no_update=use_local,
                    )

                if spandata and len(spandata) > 0:
                    break
            except Exception:
                spandata = None

        if not spandata or len(spandata) == 0:
            return None

        col_names = map_col_names_PSP("SPAN", varnames)
        dfs = []
        for i, v in enumerate(spandata):
            cols = col_names[i] if i < len(col_names) else [f"span_{i}"]
            arr = get_data(v)
            if arr is None:
                continue
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        df = pd.concat(dfs, axis=1)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None

        if "Dist_au" in df.columns:
            df["Dist_au"] = df["Dist_au"] / au_to_km

        if "TEMP" in df.columns:
            df["Tp"] = df.pop("TEMP")
            df["Vth"] = 13.84112218 * np.sqrt(df["Tp"]) / np.sqrt(3)

        return df

    except Exception:
        traceback.print_exc()
        return None


def process_span_data(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1, ind2):
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        dfspan = download_SPAN_PSP(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings)
        dfspan = sanitize_timeseries_df(dfspan)
        if dfspan is None:
            return None, diag_default(), "No SPAN", None

        if settings.get("apply_hampel", False):
            cols = (["Vr", "Vt", "Vn"] if "Vr" in dfspan.columns else ["Vx", "Vy", "Vz"])
            cols += ["np", 'Tp']
            ws = settings["hampel_params"]["w"]
            nn = settings["hampel_params"]["std"]
            for c in cols:
                if c not in dfspan.columns:
                    continue
                try:
                    out_idx = func.hampel(dfspan[c], window_size=ws, n=nn)
                    if isinstance(out_idx, tuple) and len(out_idx) == 2:
                        out_idx = out_idx[1]
                    dfspan.loc[dfspan.index[out_idx], c] = np.nan
                except Exception:
                    traceback.print_exc()

        dfspan = clip_to_requested(dfspan, ind1, ind2, req_start, req_end, func_module=func)
        dfspan = sanitize_timeseries_df(dfspan)
        if dfspan is None:
            return None, diag_default(), "No SPAN", None

        big_gaps_span = func.find_big_gaps(dfspan, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
        diagnostics_SPAN = func.resample_timeseries_estimate_gaps(dfspan, settings["part_resol"], large_gaps=10)
        diagnostics_SPAN.setdefault("resampled_df", None)

        return dfspan, diagnostics_SPAN, "SPAN", big_gaps_span

    except Exception:
        logger.exception("PSP SPAN processing failed.")
        return None, diag_default(), "No SPAN", None


# ============================================================
# 8) QTN
# ============================================================
def download_QTN_PSP(t0, t1, credentials, varnames, settings):
    try:
        qtndata = None

        try:
            username = credentials["psp"]["fields"]["username"]
            password = credentials["psp"]["fields"]["password"]
            qtndata = pyspedas.psp.fields(
                trange=[t0, t1],
                datatype="sqtn_rfs_V1V2",
                level="l3",
                varnames=varnames,
                time_clip=True,
                username=username,
                password=password,
                no_update=settings["use_local_data"],
            )
            if qtndata == []:
                qtndata = pyspedas.psp.fields(
                    trange=[t0, t1],
                    datatype="rfs_lfr_qtn",
                    level="l2",
                    time_clip=True,
                    username=username,
                    password=password,
                    no_update=settings["use_local_data"],
                )
        except Exception:
            qtndata = None

        if not qtndata or len(qtndata) == 0:
            qtndata = pyspedas.psp.fields(
                trange=[t0, t1],
                datatype="sqtn_rfs_v1v2",
                level="l3",
                varnames=varnames,
                time_clip=True,
                no_update=settings["use_local_data"],
            )

        if not qtndata or len(qtndata) == 0:
            return None

        col_names = map_col_names_PSP("QTN", varnames)
        dfs = []
        for i, v in enumerate(qtndata):
            cols = col_names[i] if i < len(col_names) else [f"qtn_{i}"]
            arr = get_data(v)
            if arr is None:
                continue
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        dfqtn = pd.concat(dfs, axis=1)
        dfqtn = sanitize_timeseries_df(dfqtn)
        if dfqtn is None:
            return None

        if "ne_qtn" in dfqtn.columns:
            dfqtn["np_qtn"] = dfqtn["ne_qtn"] * 0.96

        return dfqtn

    except Exception:
        logger.exception("PSP QTN download failed.")
        return None


def process_qtn_data(t0, t1, credentials, varnames_QTN, ind1, ind2, settings):
    dfqtn = None

    # Orlando QTN pickle (optional)
    try:
        if settings.get("orlandos_QTN", None) is not None:
            logger.info("PSP QTN: trying Orlando pickle")
            dfqtn = pd.read_pickle(settings["orlandos_QTN"])
            dfqtn = sanitize_timeseries_df(dfqtn)
            if dfqtn is not None:
                df_between = dfqtn.loc[t0:t1]
                dfqtn = df_between if len(df_between) > 0 else None
    except Exception:
        dfqtn = None

    # Hardcoded legacy fallback pickles (kept)
    if dfqtn is None:
        try:
            dfqtn = pd.read_pickle(repo_data_file("psp_data", "PSP_QTN_Monc", "E22.pkl"))
            if "Te_qtn" in dfqtn.columns:
                del dfqtn["Te_qtn"]

            dfqtn2 = pd.read_pickle(repo_data_file("psp_data", "PSP_QTN_Monc", "E23.pkl"))
            if "Te_qtn" in dfqtn2.columns:
                del dfqtn2["Te_qtn"]

            dfqtn3 = pd.read_pickle(repo_data_file("psp_data", "PSP_QTN_Romeo", "save_pickled_dfs", "e24.pkl"))
            if "ne_qtn" in dfqtn3.columns:
                del dfqtn3["ne_qtn"]

            dfqtn = pd.concat([dfqtn, dfqtn2, dfqtn3])
            dfqtn = sanitize_timeseries_df(dfqtn)
            if dfqtn is not None:
                df_between = dfqtn.loc[t0:t1]
                dfqtn = df_between if len(df_between) > 0 else None
        except Exception:
            dfqtn = None

    # final fallback to SPEDAS
    if dfqtn is None:
        dfqtn = download_QTN_PSP(t0, t1, credentials, varnames_QTN, settings)

    try:
        dfqtn = sanitize_timeseries_df(dfqtn)
        if dfqtn is None:
            raise ValueError("No QTN found")

        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)
        dfqtn = clip_to_requested(dfqtn, ind1, ind2, req_start, req_end, func_module=func)
        dfqtn = sanitize_timeseries_df(dfqtn)
        if dfqtn is None:
            raise ValueError("QTN has no overlap with requested interval")

        big_gaps = func.find_big_gaps(dfqtn, settings["Big_Gaps"]["QTN_big_gaps"], str(ind1), str(ind2))
        diagnostics_QTN = func.resample_timeseries_estimate_gaps(dfqtn, settings["part_resol"], large_gaps=10)
        diagnostics_QTN.setdefault("resampled_df", None)

        return dfqtn, diagnostics_QTN, "QTN", big_gaps

    except Exception:
        diagnostics_QTN = {
            "Init_dt": np.nan,
            "resampled_df": None,
            "Frac_miss": None,
            "Large_gaps": None,
            "Tot_gaps": None,
            "resol": None,
        }
        return None, diagnostics_QTN, "No QTN", None


# ============================================================
# 9) Ephemeris (public API preserved)
# ============================================================
def download_ephemeris_PSP(t0, t1, credentials, varnames, settings=None):
    try:
        username = credentials["psp"]["fields"]["username"]
        password = credentials["psp"]["fields"]["password"]

        ephemdata = pyspedas.psp.fields(
            trange=[t0, t1],
            datatype="ephem_spp_rtn",
            level="l1",
            varnames=varnames,
            time_clip=True,
            username=username,
            password=password,
            no_update=settings["use_local_data"] if isinstance(settings, dict) and "use_local_data" in settings else False,
        )

        if not ephemdata or len(ephemdata) == 0:
            return None

        col_names = map_col_names_PSP("EPHEMERIS", varnames)
        dfs = []
        for i, v in enumerate(ephemdata):
            cols = col_names[i] if i < len(col_names) else [f"ephem_{i}"]
            arr = get_data(v)
            if arr is None:
                continue
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        df = pd.concat(dfs, axis=1)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None

        if {"sc_pos_r", "sc_pos_t", "sc_pos_n"}.issubset(df.columns):
            df["Dist_au"] = np.sqrt(np.sum(df[["sc_pos_r", "sc_pos_t", "sc_pos_n"]] ** 2, axis=1)) / au_to_km

        return df

    except Exception:
        logger.exception("PSP ephemeris could not be loaded.")
        return None


def process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1, ind2, settings):
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        dfephem = download_ephemeris_PSP(t0, t1, credentials, varnames_EPHEM, settings)
        dfephem = sanitize_timeseries_df(dfephem)
        if dfephem is None:
            return None

        dfephem = clip_to_requested(dfephem, ind1, ind2, req_start, req_end, func_module=func)
        dfephem = sanitize_timeseries_df(dfephem)
        return dfephem

    except Exception:
        return None


# ============================================================
# 10) Optional E-field + SC potential
# ============================================================
def download_efield(t0, t1, credentials, varnames, settings):
    try:
        if varnames is None:
            return None

        fields_vars = pyspedas.psp.fields(
            trange=[t0, t1],
            datatype="dfb_wf_dvdc",
            varnames=varnames,
            level="l2",
            time_clip=True,
            no_update=settings["use_local_data"],
        )

        if not fields_vars or len(fields_vars) == 0:
            return None

        cols_map = map_col_names_PSP("FIELDS-MAG", varnames)
        cols = cols_map[0] if (cols_map and len(cols_map) > 0) else ["dvx", "dvy"]

        df = _tplot_to_df(fields_vars[0], cols)
        return sanitize_timeseries_df(df)

    except Exception:
        logger.exception("PSP E-field download failed.")
        return None


def process_e_field_data(t0, t1, settings, credentials, varnames, ind1, ind2):
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        df = download_efield(t0, t1, credentials, varnames, settings)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None, None, diag_default()

        df = clip_to_requested(df, ind1, ind2, req_start, req_end, func_module=func)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None, None, diag_default()

        big_gaps = func.find_big_gaps(df, settings["Big_Gaps"]["E_big_gaps"], str(ind1), str(ind2))
        diagnostics = func.resample_timeseries_estimate_gaps(df, 1, large_gaps=10)
        diagnostics.setdefault("resampled_df", None)

        return df, big_gaps, diagnostics

    except Exception:
        logger.exception("PSP E-field processing failed.")
        return None, None, diag_default()


def sc_potential_derived_density(t0, t1, credentials, varnames, settings):
    try:
        if varnames is None:
            return None

        fields_vars = pyspedas.psp.fields(
            trange=[t0, t1],
            datatype="dfb_wf_vdc",
            level="l2",
            time_clip=True,
            no_update=settings["use_local_data"],
        )
        if not fields_vars or len(fields_vars) == 0:
            return None

        wanted_cols = map_col_names_PSP("FIELDS-MAG", ["dfb_wf_vdc"])
        wanted_cols = wanted_cols[0] if wanted_cols else [
            "psp_fld_l2_dfb_wf_V1dc",
            "psp_fld_l2_dfb_wf_V2dc",
            "psp_fld_l2_dfb_wf_V3dc",
            "psp_fld_l2_dfb_wf_V4dc",
        ]

        df_try = _tplot_to_df(fields_vars[0], wanted_cols)
        if isinstance(df_try, pd.DataFrame) and df_try.shape[1] == 4:
            return sanitize_timeseries_df(df_try)

        dfs = []
        for i, tv in enumerate(fields_vars[:4]):
            col = [wanted_cols[i]] if i < len(wanted_cols) else [f"Vdc_{i+1}"]
            dfs.append(_tplot_to_df(tv, col))

        if len(dfs) == 0:
            return None

        df_density = pd.concat(dfs, axis=1)
        return sanitize_timeseries_df(df_density)

    except Exception:
        logger.exception("PSP SC potential download failed.")
        return None


def process_sc_potential_data(t0, t1, settings, credentials, varnames, ind1, ind2):
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        df = sc_potential_derived_density(t0, t1, credentials, varnames, settings)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None, None, diag_default()

        df = clip_to_requested(df, ind1, ind2, req_start, req_end, func_module=func)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None, None, diag_default()

        big_gaps = func.find_big_gaps(df, settings["Big_Gaps"]["SC_pot_big_gaps"], str(ind1), str(ind2))
        diagnostics = func.resample_timeseries_estimate_gaps(df, 1, large_gaps=10)
        diagnostics.setdefault("resampled_df", None)

        return df, big_gaps, diagnostics

    except Exception:
        logger.exception("PSP SC potential processing failed.")
        return None, None, diag_default()


# ============================================================
# 11) Particle selection + QTN integration
# ============================================================
def create_particle_dataframe(
    PSP_distance_au,
    end_time,
    diagnostics_spc,
    diagnostics_span,
    dfqtn,
    dfqtn_flag,
    big_gaps_span,
    big_gaps_spc,
    settings,
):
    def integrate_qtn_data(source_df: pd.DataFrame, dfqtn_in: Optional[pd.DataFrame]):
        try:
            if dfqtn_in is None or not isinstance(dfqtn_in, pd.DataFrame) or len(dfqtn_in) == 0:
                if "np" in source_df.columns:
                    source_df["np_sweap"] = source_df["np"].copy()
                return source_df, "No_QTN"

            try:
                source_df, dfqtn_sync = func.synchronize_dfs(source_df, dfqtn_in, True)
            except Exception:
                dfqtn_sync = func.newindex(dfqtn_in, source_df.index)

            if "np_qtn" in dfqtn_sync.columns:
                source_df["np"] = dfqtn_sync["np_qtn"].values

            return source_df, "QTN"

        except Exception:
            if "np" in source_df.columns:
                source_df["np_sweap"] = source_df["np"].copy()
            return source_df, "No_QTN"

    mode = settings.get("particle_mode", "9th_perih_cut")

    if mode == "9th_perih_cut":
        use_spc = pd.Timestamp(end_time) < pd.Timestamp("2021-07-15")
        df_selected = diagnostics_spc.get("resampled_df", None) if use_spc else diagnostics_span.get("resampled_df", None)
        big_gaps = big_gaps_spc if use_spc else big_gaps_span
        part_flag = "spc" if use_spc else "span"

    elif settings.get("allow_max_SWEAP_distance", False):
        use_spc = (PSP_distance_au is not None) and (PSP_distance_au > settings.get("max_SWEAP_distance", 0.25))
        df_selected = diagnostics_spc.get("resampled_df", None) if use_spc else diagnostics_span.get("resampled_df", None)
        big_gaps = big_gaps_spc if use_spc else big_gaps_span
        part_flag = "spc" if use_spc else "span"

    elif mode == "spc":
        df_selected = diagnostics_spc.get("resampled_df", None)
        big_gaps = big_gaps_spc
        part_flag = "spc"

    elif mode == "span":
        df_selected = diagnostics_span.get("resampled_df", None)
        big_gaps = big_gaps_span
        part_flag = "span"

    else:
        raise ValueError(f"Unsupported particle mode: {mode}")

    if df_selected is None or not isinstance(df_selected, pd.DataFrame) or len(df_selected) == 0:
        return None, part_flag, "No_QTN", big_gaps

    try:
        df_selected = func.replace_negative_with_nan(df_selected)
    except Exception:
        pass

    df_selected, dfqtn_flag_out = integrate_qtn_data(df_selected, dfqtn)

    try:
        df_out = df_selected.interpolate().dropna()
    except Exception:
        df_out = df_selected

    return df_out, part_flag, dfqtn_flag_out, big_gaps


# ============================================================
# 12) MAIN entry point (PUBLIC API, signature + outputs unchanged)
# ============================================================
def LoadTimeSeriesPSP(
    start_time,
    end_time,
    settings,
    vars_2_downnload,
    cdf_lib_path,
    credentials=None,
    time_amount=2,
    time_unit="h",
):
    """
    Returns EXACTLY (11 items), unchanged order:

    (
      dfqtn_resampled,
      dfmag_resampled,
      dfpar_resampled,
      df_e_field_resampled,
      df_sc_pot_resampled,
      dfephem,
      big_gaps_mag,
      big_gaps_qtn,
      big_gaps_par,
      big_gaps_sc_pot,
      misc
    )
    """
    settings = init_psp_settings(settings)

    os.chdir(settings["Data_path"])
    Path("./psp_data").mkdir(exist_ok=True)

    # interval expansion (kept)
    t0i, t1i = func.ensure_time_format(start_time, end_time)
    t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
    t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

    # small padding for ephem/qtn clipping helper (kept)
    t0i_e = func.add_time_to_datetime_string(t0i, -2, "m")
    t1i_e = func.add_time_to_datetime_string(t1i, 2, "m")

    ind1 = func.string_to_datetime_index(t0i)
    ind2 = func.string_to_datetime_index(t1i)
    ind1_e = func.string_to_datetime_index(t0i_e)
    ind2_e = func.string_to_datetime_index(t1i_e)

    (
        varnames_MAG,
        varnames_QTN,
        varnames_SPAN,
        varnames_SPC,
        varnames_SPAN_alpha,
        varnames_EPHEM,
        varnames_E_field,
        varnames_SC_pot,
    ) = default_variables_to_download_PSP(vars_2_downnload)

    # --------------------------------------------------------
    # QTN
    # --------------------------------------------------------
    dfqtn, diagnostics_QTN, dfqtn_flag, dfqtn_big_gaps = process_qtn_data(
        t0, t1, credentials, varnames_QTN, ind1_e, ind2_e, settings
    )

    # --------------------------------------------------------
    # Ephemeris (distance threshold logic kept)
    # --------------------------------------------------------
    dfephem = process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1_e, ind2_e, settings)

    mean_dist = np.nan
    try:
        if dfephem is not None and "Dist_au" in dfephem.columns and len(dfephem) > 0:
            mean_dist = float(np.nanmean(dfephem["Dist_au"].values))
            mean_dist = round(mean_dist, 2)
    except Exception:
        mean_dist = np.nan

    max_dist = settings.get("max_PSP_dist", None)
    dist_threshold = True if max_dist is None else (mean_dist < float(max_dist))
    qtn_threshold = (dfqtn_flag == "QTN") or (settings["must_have_qtn"] is False)

    if not (dist_threshold and qtn_threshold):
        return None, None, None, None, None, None, None, None, None, None, None

    # --------------------------------------------------------
    # SC potential
    # --------------------------------------------------------
    if vars_2_downnload.get("sc_pot", False):
        df_SC_pot, big_gaps_SC_pot, diagnostics_SC_pot = process_sc_potential_data(
            t0, t1, settings, credentials, varnames_SC_pot, ind1, ind2
        )
    else:
        df_SC_pot, big_gaps_SC_pot, diagnostics_SC_pot = None, None, diag_default()

    # --------------------------------------------------------
    # E-field
    # --------------------------------------------------------
    if vars_2_downnload.get("E_field", False):
        df_e_field, big_gaps_e_field, diagnostics_e_field = process_e_field_data(
            t0, t1, settings, credentials, varnames_E_field, ind1, ind2
        )
    else:
        df_e_field, big_gaps_e_field, diagnostics_e_field = None, None, diag_default()

    # --------------------------------------------------------
    # MAG
    # --------------------------------------------------------
    dfmag, big_gaps_mag, diagnostics_MAG = process_mag_field_data(
        t0, t1, settings, credentials, varnames_MAG, ind1, ind2
    )

    # --------------------------------------------------------
    # SPAN / SPC (download depending on particle_mode, unchanged logic)
    # --------------------------------------------------------
    if settings["particle_mode"] in {"span", "9th_perih_cut"}:
        dfspan, diagnostics_SPAN, span_flag, big_gaps_span = process_span_data(
            t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1_e, ind2_e
        )
    else:
        dfspan, diagnostics_SPAN, span_flag, big_gaps_span = None, diag_default(), None, None

    if settings["particle_mode"] in {"spc", "9th_perih_cut"}:
        dfspc, diagnostics_SPC, spc_flag, big_gaps_spc = process_spc_data(
            t0, t1, credentials, varnames_SPC, settings, ind1_e, ind2_e
        )
    else:
        dfspc, diagnostics_SPC, spc_flag, big_gaps_spc = None, diag_default(), None, None

    # --------------------------------------------------------
    # Particle assembly (QTN integration + selection)
    # --------------------------------------------------------
    try:
        qtn_resampled = diagnostics_QTN.get("resampled_df", None)
        if qtn_resampled is not None and not isinstance(qtn_resampled, pd.DataFrame):
            qtn_resampled = pd.DataFrame(qtn_resampled)

        dfpar, part_flag, dfqtn_flag2, big_gaps_par = create_particle_dataframe(
            mean_dist,
            end_time,
            diagnostics_SPC,
            diagnostics_SPAN,
            qtn_resampled,
            dfqtn_flag,
            big_gaps_span,
            big_gaps_spc,
            settings,
        )

        if dfpar is None or len(dfpar) == 0:
            return None, None, None, None, None, None, None, None, None, None, None

        diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
        diagnostics_PAR.setdefault("resampled_df", None)

        # misc dict structure must be identical
        misc = {
            "SPC": keep_diag_keys(diagnostics_SPC),
            "SPAN": keep_diag_keys(diagnostics_SPAN),
            "QTN": keep_diag_keys(diagnostics_QTN),
            "Par": keep_diag_keys(diagnostics_PAR),
            "E": keep_diag_keys(diagnostics_e_field),
            "SC_pot": keep_diag_keys(diagnostics_SC_pot),
            "Mag": keep_diag_keys(diagnostics_MAG),
            "part_flag": part_flag,
            "qtn_flag": dfqtn_flag2,
        }

        # preserve np fallback behavior
        if dfqtn_flag2 == "No_QTN" and isinstance(diagnostics_PAR.get("resampled_df", None), pd.DataFrame):
            if "np_sweap" in diagnostics_PAR["resampled_df"].columns and "np" not in diagnostics_PAR["resampled_df"].columns:
                diagnostics_PAR["resampled_df"]["np"] = diagnostics_PAR["resampled_df"].pop("np_sweap")

        # return order must remain identical
        return (
            diagnostics_QTN.get("resampled_df", None),
            diagnostics_MAG.get("resampled_df", None),
            diagnostics_PAR.get("resampled_df", None),
            diagnostics_e_field.get("resampled_df", None),
            diagnostics_SC_pot.get("resampled_df", None),
            dfephem.interpolate() if isinstance(dfephem, pd.DataFrame) else dfephem,
            big_gaps_mag,
            dfqtn_big_gaps,
            big_gaps_par,
            big_gaps_SC_pot,
            misc,
        )

    except Exception:
        logger.exception("LoadTimeSeriesPSP failed in final assembly.")
        return None, None, None, None, None, None, None, None, None, None, None

