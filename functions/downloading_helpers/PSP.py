"""
PSP.py

Clean, pythonic rewrite of the PSP download helper while preserving:

- Public API (signatures unchanged)
- Return ordering unchanged
- Output DataFrame contracts unchanged
- "misc" dict key structure unchanged
- QTN / ephemeris / MAG / SCAM logic unchanged in meaning
- Gap + diagnostics behavior unchanged in meaning
- Optional SCAM wheel-noise removal (now centralized & consistent)

This module is intentionally tidy and readable.

NOTES (important for your issue):
- SWEAP/SPAN-i "SC" frame (VEL_SC) is NOT the same as "instrument" frame (VEL_INST).
  The SPAN-i moments product provides BOTH VEL_SC (spacecraft coordinates) and VEL_INST
  (instrument coordinates). If you request VEL_SC you DO NOT need to convert; if you
  request VEL_INST and want spacecraft components, you must rotate using the provided
  attitude/rotation quantities (e.g., QUAT_SC_TO_RTN / etc), which we do NOT do here.
- To avoid the downstream Pandas "Columns must be same length as key" failure caused by
  duplicated column names, we drop ephemeris sc_vel_r/t/n ONLY when particle data already
  contains sc_vel_r/t/n (SPAN case). SPC case still keeps ephemeris sc_vel_r/t/n.
"""

from __future__ import annotations

import os
import sys
import importlib.util
import logging
import traceback
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[1]

_PATH_SETUP = _REPO_ROOT / "functions" / "path_setup.py"
_spec = importlib.util.spec_from_file_location("mhdturbpy_path_setup", _PATH_SETUP)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load path setup from {_PATH_SETUP}")
_path_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_path_setup)
ensure_project_paths = _path_setup.ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True, include_anisotropy_toolbox=True)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import constants

# ------------------------------------------------------------
# Local SPEDAS
# ------------------------------------------------------------
import pyspedas  # type: ignore
from pytplot import get_data  # type: ignore

# ------------------------------------------------------------
# Your repo utilities (must exist)
# ------------------------------------------------------------
import general_functions as func  # type: ignore
import TurbPy as turb  # type: ignore


# ------------------------------------------------------------
# Shared utilities (single authority; already in your repo)
# ------------------------------------------------------------
from shared_utils import (  # type: ignore
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
# Constants
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
        "verbose": True,
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
            "use_SCM": False,
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

    # SPAN-i moments product variables (these are in the CDF for sf00_l3_mom):
    # DENS, VEL_RTN_SUN, VEL_SC, VEL_INST, TEMP, SUN_DIST, SC_VEL_RTN_SUN, ...
    varnames_SPAN = (
        ["DENS", "VEL_RTN_SUN", "TEMP", "SUN_DIST", "SC_VEL_RTN_SUN"]
        if vars_2_downnload.get("span", None) is None
        else vars_2_downnload["span"]
    )

    # SPC: keep legacy defaults, but we will robustly fall back if pyspedas returns different names
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

    # SPC: accept both moment and fit-style names (pyspedas + CDAWeb variants)
    spc_cols = {
        "np_moment": ["np"],
        "np_moment_gd": ["np"],
        "np_fit": ["np"],
        "np_fit_gd": ["np"],
        "wp_moment": ["Vth"],
        "wp_moment_gd": ["Vth"],
        "wp_fit": ["Vth"],
        "wp_fit_gd": ["Vth"],
        "vp_moment_RTN": ["Vr", "Vt", "Vn"],
        "vp_moment_RTN_gd": ["Vr", "Vt", "Vn"],
        "vp_fit_RTN": ["Vr", "Vt", "Vn"],
        "vp_fit_RTN_gd": ["Vr", "Vt", "Vn"],
        "vp_moment_SC": ["Vx", "Vy", "Vz"],
        "vp_moment_SC_gd": ["Vx", "Vy", "Vz"],
        "vp_fit_SC": ["Vx", "Vy", "Vz"],
        "vp_fit_SC_gd": ["Vx", "Vy", "Vz"],
        "sc_pos_HCI": ["sc_x", "sc_y", "sc_z"],
        "sc_vel_HCI": ["sc_vel_x", "sc_vel_y", "sc_vel_z"],
        "carr_latitude": ["carr_lat"],
        "carr_longitude": ["carr_lon"],
        "general_flag": ["flag"],
    }

    # SPAN-i moments (sf00_l3_mom)
    span_cols = {
        "DENS": ["np"],
        "VEL_RTN_SUN": ["Vr", "Vt", "Vn"],
        "VEL_SC": ["Vx", "Vy", "Vz"],          # spacecraft coordinates (SC frame)
        "VEL_INST": ["Vix", "Viy", "Viz"],     # instrument coordinates (NOT spacecraft)
        "VEL_INST_SUN": ["Vix_sun", "Viy_sun", "Viz_sun"],
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
    Private (team) first, public second.
    """
    verbose = bool(settings.get("verbose", True))
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
            used_auth = "public"
            MAGdata = None

            # --- private first (if creds exist) ---
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
                    used_auth = "private"
            except Exception:
                MAGdata = None
                used_datatype = None

            # --- public fallback ---
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
                    used_auth = "public"

            if not MAGdata or len(MAGdata) == 0 or used_datatype is None:
                if verbose:
                    print(f"[PSP/MAG] No data returned for {v} (private {datatype_private} -> public {datatype_public})")
                continue

            if verbose:
                print(f"[PSP/MAG] Using {used_auth} {used_datatype} for {v}")

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
    Private (team) first, public second.
    """
    verbose = bool(settings.get("verbose", True))
    try:
        user = None
        pwd = None
        try:
            user = credentials["psp"]["fields"]["username"]
            pwd = credentials["psp"]["fields"]["password"]
        except Exception:
            user = None
            pwd = None

        def _call(use_creds: bool):
            kwargs = dict(
                trange=[t0, t1],
                datatype="merged_scam_wf",
                level="l3",
                time_clip=1,
                downloadonly=False,
                no_update=settings["use_local_data"],
            )
            if settings.get("in_rtn", True):
                kwargs.update(varnames=["psp_fld_l3_merged_scam_wf_RTN"])
                cols = ["Br", "Bt", "Bn"]
            else:
                kwargs.update(varnames=["psp_fld_l3_merged_scam_wf_SC"])
                cols = ["Bx", "By", "Bz"]

            if use_creds and user and pwd:
                kwargs.update(username=user, password=pwd)
            return pyspedas.psp.fields(**kwargs), cols, ("private" if use_creds and user and pwd else "public")

        scam_vars, cols, auth = _call(use_creds=True)
        if not scam_vars or len(scam_vars) == 0:
            scam_vars, cols, auth = _call(use_creds=False)

        if not scam_vars or len(scam_vars) == 0:
            if verbose:
                print("[PSP/SCAM] No SCAM returned (private->public).")
            return None

        if verbose:
            frame = "RTN" if cols[0].startswith("B") and cols[0] == "Br" else "SC"
            print(f"[PSP/SCAM] Using {auth} merged_scam_wf ({frame})")

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

        # OPTIONAL wheel-noise removal (resampled_df stage only)
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
    """
    Private first (if SWEAP creds exist), then public.
    Robust against variable-name drift by allowing simple fallback sets.
    """
    verbose = bool(settings.get("verbose", True))
    try:
        spcdata = None
        used_auth = "public"

        user = None
        pwd = None
        try:
            user = credentials["psp"]["sweap"]["username"]
            pwd = credentials["psp"]["sweap"]["password"]
        except Exception:
            user = None
            pwd = None

        def _call(vnames, use_creds: bool):
            kwargs = dict(
                trange=[t0, t1],
                datatype="l3i",
                level="l3",
                varnames=vnames,
                time_clip=True,
                no_update=settings["use_local_data"],
            )
            if use_creds and user and pwd:
                kwargs.update(username=user, password=pwd)
            return pyspedas.psp.spc(**kwargs)

        # 1) private first (if possible)
        if user and pwd:
            try:
                spcdata = _call(varnames, use_creds=True)
                if spcdata and len(spcdata) > 0:
                    used_auth = "private"
            except Exception:
                spcdata = None

        # 2) public fallback
        if not spcdata or len(spcdata) == 0:
            try:
                spcdata = _call(varnames, use_creds=False)
                used_auth = "public"
            except Exception:
                spcdata = None

        # 3) minimal fallback (only if nothing returned)
        if (not spcdata or len(spcdata) == 0) and isinstance(varnames, list) and len(varnames) > 0:
            alt = []
            m = {
                "np_moment": "np_moment",
                "wp_moment": "wp_moment",
                "vp_moment_RTN": "vp_moment_RTN",
                "vp_moment_SC": "vp_moment_SC",
                "np_fit": "np_fit",
                "wp_fit": "wp_fit",
                "vp_fit_RTN": "vp_fit_RTN",
                "vp_fit_SC": "vp_fit_SC",
                "sc_pos_HCI": "sc_pos_HCI",
                "sc_vel_HCI": "sc_vel_HCI",
                "carr_longitude": "carr_longitude",
                "general_flag": "general_flag",
            }
            for v in varnames:
                vv = m.get(v, v)
                if vv not in alt:
                    alt.append(vv)
            try:
                spcdata = _call(alt, use_creds=False)
                used_auth = "public"
            except Exception:
                spcdata = None

        if not spcdata or len(spcdata) == 0:
            if verbose:
                print("[PSP/SPC] No SPC returned (private->public).")
            return None

        if verbose:
            print(f"[PSP/SPC] Using {used_auth} spc:l3i")
            print(f"[PSP/SPC] Requested vars: {varnames}")

        # Map columns by matching returned tplot variable names to requested keys
        wanted = {}
        for vn in (varnames or []):
            if not isinstance(vn, str):
                continue
            if vn in map_col_names_PSP("SPC", [vn])[0:1]:
                pass
        # Build mapping dict using known keys (both old and common current names)
        for key in [
            "np_moment", "wp_moment", "vp_moment_RTN", "vp_moment_SC",
            "np_fit", "wp_fit", "vp_fit_RTN", "vp_fit_SC",
            "sc_pos_HCI", "sc_vel_HCI", "carr_longitude", "general_flag"
        ]:
            cols = map_col_names_PSP("SPC", [key])
            if cols:
                wanted[key.lower()] = cols[0]

        dfs = []
        for tv in spcdata:
            arr = get_data(tv)
            if arr is None:
                continue
            tvl = str(tv).lower()
            cols = None
            for k, ccols in wanted.items():
                if tvl.endswith(k) or tvl.endswith(k.lower()) or tvl.endswith(k.upper()) or tvl.find(k) >= 0:
                    cols = ccols
                    break
            if cols is None:
                yy = np.asarray(arr.y) if hasattr(arr, "y") else None
                if yy is not None and yy.ndim == 2 and yy.shape[1] == 3:
                    cols = ["col0", "col1", "col2"]
                else:
                    cols = ["col0"]
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        df = pd.concat(dfs, axis=1)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None

        # If sc_pos_HCI present, compute Dist_au and drop raw pos columns (legacy behavior)
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
            cols = ["Vr", "Vt", "Vn", "np", "Vth"] if "Vr" in dfspc.columns else ["Vx", "Vy", "Vz", "np", "Vth"]
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

        # Tp estimation kept (best-effort): from Vth assuming most-probable thermal speed
        try:
            from astropy.constants import m_p as mp_ast, k_B
            from astropy import units as u

            if "Vth" in dfspc.columns:
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
# 7) SPAN (SWEAP/SPAN-i)
# ============================================================
def download_SPAN_PSP(t0, t1, credentials, varnames, varnames_alpha, settings):
    """
    IMPORTANT:
    - Must follow updated pyspedas PSP/SPI wrapper behavior.
    - Private (team) first, public second.
    - DO NOT prepend 'psp_spi_' to varnames; varnames must match CDF variables (e.g. DENS, VEL_RTN_SUN, ...).
    - SC frame is VEL_SC (spacecraft coordinates) and is different from VEL_INST (instrument).
    """
    verbose = bool(settings.get("verbose", True))
    try:
        span_key = str(settings.get("span_key", "spi_sf00")).strip().lower()
        use_local = settings.get("use_local_data", False)

        user = None
        pwd = None
        try:
            user = credentials["psp"]["sweap"]["username"]
            pwd = credentials["psp"]["sweap"]["password"]
        except Exception:
            user = None
            pwd = None

        # Decide datatypes to try.
        # Updated spi.py supports: sf00_l3_mom, sf0a_l3_mom, sf00_l3_mom_inst, sf0a_l3_mom_inst
        # Accept legacy span_key strings, but prioritize the moments products.
        def _dtype_list(key: str) -> List[str]:
            if "sf0a" in key or "alpha" in key:
                base = "sf0a"
            else:
                base = "sf00"
            if "inst" in key:
                return [f"{base}_l3_mom_inst", f"spi_{base}_l3_mom_inst"]
            if "l3_mom" in key or "mom" in key or "spi_sf00" in key or "spi_sf0a" in key:
                return [f"{base}_l3_mom", f"spi_{base}_l3_mom", f"spi_{base}"]
            return [f"{base}_l3_mom", f"spi_{base}_l3_mom", f"spi_{base}"]

        datatypes = _dtype_list(span_key)

        # Normalize requested variable names to the CDF convention (mostly uppercase for SPAN-i moments)
        vreq = []
        for v in (varnames or []):
            if isinstance(v, str):
                vreq.append(v.strip().upper())
        if len(vreq) == 0:
            return None

        def _call_spi(datatype: str, use_creds: bool):
            # Level case matters for the SWEAP team server: use "L3" when authenticated, "l3" otherwise.
            level = "L3" if (use_creds and user and pwd) else "l3"
            kwargs = dict(
                trange=[t0, t1],
                datatype=datatype,
                level=level,
                varnames=vreq,
                time_clip=True,
                no_update=use_local,
            )
            if use_creds and user and pwd:
                kwargs.update(username=user, password=pwd)
            return pyspedas.psp.spi(**kwargs)

        spandata = None
        used_dtype = None
        used_auth = "public"

        for dt in datatypes:
            # private first if possible
            if user and pwd:
                try:
                    spandata = _call_spi(dt, use_creds=True)
                    if spandata and len(spandata) > 0:
                        used_dtype = dt
                        used_auth = "private"
                        break
                except Exception:
                    spandata = None

            # public fallback
            try:
                spandata = _call_spi(dt, use_creds=False)
                if spandata and len(spandata) > 0:
                    used_dtype = dt
                    used_auth = "public"
                    break
            except Exception:
                spandata = None

        if not spandata or len(spandata) == 0:
            if verbose:
                print(f"[PSP/SPAN] No SPAN returned (private->public). Tried datatypes={datatypes}")
                print(f"[PSP/SPAN] Requested vars={vreq}")
            return None

        if verbose:
            print(f"[PSP/SPAN] Using {used_auth} spi:{used_dtype}")
            print(f"[PSP/SPAN] Requested vars: {vreq}")
            if "VEL_SC" in vreq:
                print("[PSP/SPAN] Frame note: VEL_SC is spacecraft coordinates (SC frame). No conversion needed.")
            if "VEL_INST" in vreq or "VEL_INST_SUN" in vreq:
                print("[PSP/SPAN] Frame note: VEL_INST* is instrument coordinates (NOT spacecraft). Conversion not performed here.")

        # Match returned tplot variable names to requested keys (robust; avoids order assumptions)
        cols_by_key = {}
        for key in vreq:
            cols = map_col_names_PSP("SPAN", [key])
            if cols:
                cols_by_key[key.upper()] = cols[0]

        dfs = []
        seen_keys = set()
        for tv in spandata:
            tvu = str(tv).upper()
            match = None
            for key in cols_by_key.keys():
                if tvu.endswith(key) or tvu.find(key) >= 0:
                    match = key
                    break
            if match is None:
                continue
            if match in seen_keys:
                # if pyspedas ever returns duplicates, keep first (prevents duplicate columns)
                continue
            seen_keys.add(match)
            arr = get_data(tv)
            if arr is None:
                continue
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols_by_key[match]))

        if len(dfs) == 0:
            return None

        df = pd.concat(dfs, axis=1)
        df = sanitize_timeseries_df(df)
        if df is None:
            return None

        if "Dist_au" in df.columns:
            df["Dist_au"] = df["Dist_au"] / au_to_km

        # SPAN-i TEMP is eV in the moments product; preserve your legacy conversions
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
            cols += ["np", "Tp"]
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
    """
    Private first (FIELDS creds) then public.
    """
    verbose = bool(settings.get("verbose", True))
    try:
        qtndata = None
        used_auth = "public"
        used_dtype = None

        user = None
        pwd = None
        try:
            user = credentials["psp"]["fields"]["username"]
            pwd = credentials["psp"]["fields"]["password"]
        except Exception:
            user = None
            pwd = None

        # private first
        if user and pwd:
            try:
                qtndata = pyspedas.psp.fields(
                    trange=[t0, t1],
                    datatype="sqtn_rfs_V1V2",
                    level="l3",
                    varnames=varnames,
                    time_clip=True,
                    username=user,
                    password=pwd,
                    no_update=settings["use_local_data"],
                )
                if qtndata == []:
                    qtndata = pyspedas.psp.fields(
                        trange=[t0, t1],
                        datatype="rfs_lfr_qtn",
                        level="l2",
                        time_clip=True,
                        username=user,
                        password=pwd,
                        no_update=settings["use_local_data"],
                    )
                if qtndata and len(qtndata) > 0:
                    used_auth = "private"
                    used_dtype = "sqtn_rfs_V1V2"
            except Exception:
                qtndata = None

        # public fallback
        if not qtndata or len(qtndata) == 0:
            qtndata = pyspedas.psp.fields(
                trange=[t0, t1],
                datatype="sqtn_rfs_v1v2",
                level="l3",
                varnames=varnames,
                time_clip=True,
                no_update=settings["use_local_data"],
            )
            if qtndata and len(qtndata) > 0:
                used_auth = "public"
                used_dtype = "sqtn_rfs_v1v2"

        if not qtndata or len(qtndata) == 0:
            if verbose:
                print("[PSP/QTN] No QTN returned (private->public).")
            return None

        if verbose:
            print(f"[PSP/QTN] Using {used_auth} {used_dtype}")

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
            dfqtn = pd.read_pickle(str(Path(settings["Data_path"]).joinpath("psp_data/PSP_QTN_Monc/E22.pkl")))
            if "Te_qtn" in dfqtn.columns:
                del dfqtn["Te_qtn"]

            dfqtn2 = pd.read_pickle(str(Path(settings["Data_path"]).joinpath("psp_data/PSP_QTN_Monc/E23.pkl")))
            if "Te_qtn" in dfqtn2.columns:
                del dfqtn2["Te_qtn"]

            dfqtn3 =  pd.read_pickle(str(Path(settings["Data_path"]).joinpath("psp_data/PSP_QTN_Romeo/save_pickled_dfs/e24.pkl")))
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
    """
    Ephemeris is FIELDS (needs FIELDS creds for private access).
    Private first then public.
    """
    verbose = bool(settings.get("verbose", True)) if isinstance(settings, dict) else True
    try:
        user = None
        pwd = None
        try:
            user = credentials["psp"]["fields"]["username"]
            pwd = credentials["psp"]["fields"]["password"]
        except Exception:
            user = None
            pwd = None

        def _call(use_creds: bool):
            kwargs = dict(
                trange=[t0, t1],
                datatype="ephem_spp_rtn",
                level="l1",
                varnames=varnames,
                time_clip=True,
                no_update=settings["use_local_data"] if isinstance(settings, dict) and "use_local_data" in settings else False,
            )
            if use_creds and user and pwd:
                kwargs.update(username=user, password=pwd)
            return pyspedas.psp.fields(**kwargs)

        ephemdata = None
        used_auth = "public"

        if user and pwd:
            try:
                ephemdata = _call(True)
                if ephemdata and len(ephemdata) > 0:
                    used_auth = "private"
            except Exception:
                ephemdata = None

        if not ephemdata or len(ephemdata) == 0:
            ephemdata = _call(False)
            used_auth = "public"

        if not ephemdata or len(ephemdata) == 0:
            if verbose:
                print("[PSP/EPHEM] No ephemeris returned (private->public).")
            return None

        if verbose:
            print(f"[PSP/EPHEM] Using {used_auth} ephem_spp_rtn:l1")
            print(f"[PSP/EPHEM] Requested vars: {varnames}")

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
    verbose = bool(settings.get("verbose", True))

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

    if verbose:
        print("=== PSP loader summary ===")
        print(f"Requested interval : {pd.Timestamp(t0i)} -> {pd.Timestamp(t1i)}")
        print(f"Expanded interval  : {pd.Timestamp(t0)} -> {pd.Timestamp(t1)}")
        print(f"Target cadences    : MAG_resol={settings.get('MAG_resol')} , part_resol={settings.get('part_resol')}")
        print(f"Frames             : MAG in_rtn={settings.get('in_rtn', True)} ; SPAN keys={settings.get('span_key','spi_sf00')}")
        print("Private-first      : FIELDS+SWEAP credentials used when provided; public fallback otherwise.")
        print("--------------------------")

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
        if verbose:
            print("=== PSP loader exit early ===")
            print(f"Distance check ok? {dist_threshold} (mean Dist_au={mean_dist}, max_PSP_dist={max_dist})")
            print(f"QTN check ok?      {qtn_threshold} (qtn_flag={dfqtn_flag}, must_have_qtn={settings.get('must_have_qtn')})")
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
            if verbose:
                print("[PSP/PAR] No particle dataframe assembled.")
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

        # ------------------------------------------------------------------
        # CRITICAL FIX for your new Pandas error:
        # If particle data already contains sc_vel_r/t/n (SPAN case),
        # drop ephemeris sc_vel_r/t/n to prevent duplicate column names downstream.
        # (SPC case typically does not include sc_vel_r/t/n, so ephem keeps them.)
        # ------------------------------------------------------------------
        dfephem_out = dfephem
        try:
            par_df = diagnostics_PAR.get("resampled_df", None)
            if isinstance(dfephem, pd.DataFrame) and isinstance(par_df, pd.DataFrame):
                dup = [c for c in ("sc_vel_r", "sc_vel_t", "sc_vel_n") if (c in dfephem.columns and c in par_df.columns)]
                if dup:
                    if verbose:
                        print(f"[PSP] Dropping ephemeris columns {dup} to prevent duplicate-column downstream failures.")
                    dfephem_out = dfephem.drop(columns=dup)
        except Exception:
            dfephem_out = dfephem

        if verbose:
            print("=== PSP loader done ===")
            print(f"part_flag={part_flag} qtn_flag={dfqtn_flag2} mean Dist_au={mean_dist}")
            if isinstance(diagnostics_MAG.get("resampled_df", None), pd.DataFrame):
                print(f"[PSP/MAG] resampled cols={list(diagnostics_MAG['resampled_df'].columns)}")
            if isinstance(diagnostics_PAR.get("resampled_df", None), pd.DataFrame):
                print(f"[PSP/PAR] resampled cols={list(diagnostics_PAR['resampled_df'].columns)}")
            if isinstance(dfephem_out, pd.DataFrame):
                print(f"[PSP/EPHEM] cols={list(dfephem_out.columns)}")
            print("------------------------")

        # return order must remain identical
        return (
            diagnostics_QTN.get("resampled_df", None),
            diagnostics_MAG.get("resampled_df", None),
            diagnostics_PAR.get("resampled_df", None),
            diagnostics_e_field.get("resampled_df", None),
            diagnostics_SC_pot.get("resampled_df", None),
            dfephem_out.interpolate() if isinstance(dfephem_out, pd.DataFrame) else dfephem_out,
            big_gaps_mag,
            dfqtn_big_gaps,
            big_gaps_par,
            big_gaps_SC_pot,
            misc,
        )

    except Exception:
        logger.exception("LoadTimeSeriesPSP failed in final assembly.")
        return None, None, None, None, None, None, None, None, None, None, None
