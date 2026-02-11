
"""
PSP.py  
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

# -----------------------------------------------------------------------------
# Local SPEDAS (keep repo layout compatibility)
# -----------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.getcwd(), "pyspedas"))
import pyspedas  # noqa: E402
from pytplot import get_data  # noqa: E402

# -----------------------------------------------------------------------------
# Your helper functions (repo-local)
# -----------------------------------------------------------------------------
sys.path.insert(1, os.path.join(os.getcwd(), "functions"))
import general_functions as func  # noqa: E402
import TurbPy as turb  # noqa: E402

# -----------------------------------------------------------------------------
# Logging (do NOT spam, do NOT print-storm)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Constants (kept)
# -----------------------------------------------------------------------------
mu0 = constants.mu_0
m_p = constants.m_p
au_to_km = 1.496e8


# =============================================================================
# 0) DataFrame hygiene (STRICT everywhere)
# =============================================================================
def _ensure_datetime_index(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Enforce:
      - DatetimeIndex (tz-naive)
      - sorted, monotonic
      - no duplicated timestamps
      - return DataFrame or None
    """
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        return None
    if len(df) == 0:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        # Robust: handle numeric epoch stored as object index
        idx = pd.Index(df.index)
        vals = pd.to_numeric(idx, errors="coerce")

        finite = vals[np.isfinite(vals)]
        if finite.size > 0 and (finite.size / max(1, len(vals))) > 0.8:
            med = float(np.nanmedian(finite[: min(200, finite.size)]))
            if med > 1e15:
                unit = "ns"
            elif med > 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")
            df.index = pd.DatetimeIndex(dt).tz_localize(None)
        else:
            df.index = pd.to_datetime(idx, errors="coerce")
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize(None)

    # drop NaT index rows
    if isinstance(df.index, pd.DatetimeIndex):
        mask = ~pd.isna(df.index)
        df = df.loc[mask]

    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="first")]
    df.index.name = "datetime"
    return df


def _clip_df_pandas(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.loc[(df.index >= start) & (df.index <= end)]


def _clip_to_requested(
    df: Optional[pd.DataFrame],
    ind1,
    ind2,
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Keep your legacy behavior if possible (func.use_dates_return_elements_of_df_inbetween),
    but provide a robust pandas fallback to avoid empty slices due to index typing issues.
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None

    try:
        out = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, df)
        if isinstance(out, pd.DataFrame) and len(out) > 0:
            return _ensure_datetime_index(out)
    except Exception:
        pass

    out = _clip_df_pandas(df, req_start, req_end)
    if len(out) == 0:
        return pd.DataFrame()
    return out


def _diag_default() -> Dict[str, Any]:
    return {
        "Init_dt": np.nan,
        "resampled_df": None,
        "Frac_miss": 100,
        "Large_gaps": 100,
        "Tot_gaps": 100,
        "resol": 100,
    }


def _keep_diag_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    keys_to_keep = ["Frac_miss", "Large_gaps", "Tot_gaps", "resol"]
    try:
        return func.filter_dict(d, keys_to_keep)
    except Exception:
        return {k: d.get(k, None) for k in keys_to_keep}


def _tplot_to_df(tplot_var: str, columns: List[str]) -> pd.DataFrame:
    """
    Convert pytplot variable -> DataFrame safely.
    """
    arr = get_data(tplot_var)
    if arr is None:
        return pd.DataFrame()

    times = getattr(arr, "times", None)
    y = getattr(arr, "y", None)

    if times is None or y is None:
        try:
            times = arr[0]
            y = arr[1]
        except Exception:
            return pd.DataFrame()

    try:
        df = pd.DataFrame(index=times, data=y)
        if df.shape[1] == len(columns):
            df.columns = columns
        else:
            df.columns = columns[: df.shape[1]]
        return _ensure_datetime_index(df) or pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# =============================================================================
# 1) Settings normalization (legacy flat OR new nested schema)
# =============================================================================
def init_psp_settings(user_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single authoritative initializer.

    Accepts BOTH:
      - legacy flat keys (your current workflow)
      - new nested schema:
            settings["paths"]["Data_path"]
            settings["resolution"]["MAG_resol"], ...
            settings["quality"]["must_have_qtn"], ...
            settings["PSP"][... PSP-only ...]

    Returns a FLAT dict with legacy keys preserved, so downstream code stays identical.
    """
    if not isinstance(user_settings, dict):
        raise TypeError("settings must be a dict")

    # 1) Pull nested common schema (if present)
    flat: Dict[str, Any] = {}

    paths = user_settings.get("paths", {})
    if isinstance(paths, dict):
        if "Data_path" in paths:
            flat["Data_path"] = paths["Data_path"]
        if "save_destination" in paths:
            flat["save_destination"] = paths["save_destination"]

    io = user_settings.get("io", {})
    if isinstance(io, dict):
        for k in ("use_local_data", "overwrite_files", "save_all", "addit_time_around", "gap_time_threshold"):
            if k in io:
                flat[k] = io[k]

    intervals = user_settings.get("intervals", {})
    if isinstance(intervals, dict):
        for k in ("start_date", "end_date", "duration", "Step", "multiple_intervals"):
            if k in intervals:
                flat[k] = intervals[k]

    resolution = user_settings.get("resolution", {})
    if isinstance(resolution, dict):
        for k in ("part_resol", "MAG_resol", "upsample_low_freq_ts"):
            if k in resolution:
                flat[k] = resolution[k]

    quality = user_settings.get("quality", {})
    if isinstance(quality, dict):
        for k in ("Max_par_missing", "must_have_qtn", "max_PSP_dist"):
            if k in quality:
                flat[k] = quality[k]

    gaps = user_settings.get("gaps", {})
    if isinstance(gaps, dict):
        if "Big_Gaps" in gaps:
            flat["Big_Gaps"] = gaps["Big_Gaps"]

    # analysis toggles: keep as-is if user provides them there
    analysis = user_settings.get("analysis", {})
    if isinstance(analysis, dict):
        for k in (
            "estimate_derived_param",
            "PSDs",
            "struc_funcs",
            "npt_struc_funcs",
            "coherence_analysis",
            "E_field",
            "sc_pot",
        ):
            if k in analysis:
                flat[k] = analysis[k]

    # 2) PSP-only namespace override (if present)
    psp_only = user_settings.get("PSP", {})
    if isinstance(psp_only, dict):
        flat.update(psp_only)

    # 3) Finally apply legacy user_settings flat keys (highest priority)
    merged = {**user_settings, **flat}

    # 4) Defaults (only fill missing)
    defaults = {
        "sc": "PSP",
        "in_rtn": True,
        "particle_mode": "9th_perih_cut",
        "part_resol": 900,
        "MAG_resol": 1,
        "use_local_data": False,
        "must_have_qtn": False,
        "Max_par_missing": 30,
        "max_PSP_dist": None,
        "allow_max_SWEAP_distance": False,
        "max_SWEAP_distance": 0.25,
        "upsample_low_freq_ts": False,
        "apply_hampel": False,
        "hampel_params": {"w": 200, "std": 3},
        "orlandos_QTN": None,
        "E_field": {"flag": False},
        "sc_pot": {"flag": False},
        "Mag_SCAM_PSP": {
            "flag": False,        # if True: use SCAM instead of fluxgate
            "noise_flag": False,  # if True: remove wheel noise after resampling
            "noise_removal": {
                "window_size": 2**15,
                "avg_length": 1,
                "power_threshold": 3.0,
                "freq_min": 10.0,
                "hampel_wind": 51,
                "hampel_thresh": 3.5,
            },
        },
        "Big_Gaps": {
            "E_big_gaps": 10,
            "SC_pot_big_gaps": 10,
            "Mag_big_gaps": 500,
            "Par_big_gaps": 500,
            "QTN_big_gaps": 10,
        },
    }

    out = {**defaults, **merged}

    # Ensure nested dicts exist
    if not isinstance(out.get("E_field", {}), dict):
        out["E_field"] = {"flag": bool(out.get("E_field"))}
    if not isinstance(out.get("sc_pot", {}), dict):
        out["sc_pot"] = {"flag": bool(out.get("sc_pot"))}
    if not isinstance(out.get("Mag_SCAM_PSP", {}), dict):
        out["Mag_SCAM_PSP"] = defaults["Mag_SCAM_PSP"]
    if not isinstance(out.get("Big_Gaps", {}), dict):
        out["Big_Gaps"] = defaults["Big_Gaps"]

    # Hard requirement
    if "Data_path" not in out:
        raise KeyError("settings must include Data_path (or paths['Data_path'])")

    return out


# =============================================================================
# 2) Variable defaults (PUBLIC API behavior preserved)
# =============================================================================
def default_variables_to_download_PSP(vars_2_downnload: Dict[str, Any]):
    if vars_2_downnload["mag"] is None:
        varnames_MAG = ["B_RTN"]
    else:
        varnames_MAG = vars_2_downnload["mag"]

    if vars_2_downnload["qtn"] is None:
        varnames_QTN = ["electron_density", "electron_core_temperature"]
    else:
        varnames_QTN = vars_2_downnload["qtn"]

    if vars_2_downnload["span"] is None:
        varnames_SPAN = ["DENS", "VEL_RTN_SUN", "TEMP", "SUN_DIST", "SC_VEL_RTN_SUN"]
    else:
        varnames_SPAN = vars_2_downnload["span"]

    if vars_2_downnload["spc"] is None:
        varnames_SPC = ["np_moment", "wp_moment", "vp_moment_RTN", "sc_pos_HCI", "carr_longitude", "general_flag"]
    else:
        varnames_SPC = vars_2_downnload["spc"]

    if vars_2_downnload["span-a"] is None:
        varnames_SPAN_alpha = ["DENS"]
    else:
        varnames_SPAN_alpha = vars_2_downnload["span-a"]

    if vars_2_downnload["ephem"] is None:
        varnames_EPHEM = ["position", "velocity"]
    else:
        varnames_EPHEM = vars_2_downnload["ephem"]

    if vars_2_downnload.get("E_field", False):
        varnames_E_field = ["psp_fld_l2_dfb_wf_dVdc_sc"]
    else:
        varnames_E_field = None

    if vars_2_downnload.get("sc_pot", False):
        varnames_SC_pot = ["dfb_wf_vdc"]
    else:
        varnames_SC_pot = None

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


# =============================================================================
# 3) Column mapping (PUBLIC API behavior preserved)
# =============================================================================
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


# =============================================================================
# 4) MAG: fluxgate OR SCAM (wheel-noise removal optional)
# =============================================================================
_MAG_HIGHRES_THRESHOLD = 230


def download_MAG_FIELD_PSP(t0: str, t1: str, credentials: Dict[str, Any], varnames: List[str], settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Fluxgate MAG download.
    """
    try:
        dfmag = pd.DataFrame()

        for varname in varnames:
            if varname == "B_RTN":
                datatype_private = "mag_RTN_4_Sa_per_Cyc" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_RTN"
                datatype_public = "mag_rtn_4_per_cycle" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_rtn"
            else:
                datatype_private = "mag_SC_4_Sa_per_Cyc" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_SC"
                datatype_public = "mag_sc_4_per_cycle" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_sc"

            MAGdata = None
            used_datatype = None

            # private
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
                if MAGdata and len(MAGdata):
                    used_datatype = datatype_private
            except Exception:
                MAGdata, used_datatype = None, None

            # public fallback
            if not MAGdata or len(MAGdata) == 0:
                MAGdata = pyspedas.psp.fields(
                    trange=[t0, t1],
                    datatype=datatype_public,
                    level="l2",
                    time_clip=True,
                    no_update=settings["use_local_data"],
                )
                if MAGdata and len(MAGdata):
                    used_datatype = datatype_public

            if not MAGdata or len(MAGdata) == 0 or used_datatype is None:
                continue

            cols = map_col_names_PSP("FIELDS-MAG", [used_datatype])
            cols = cols[0] if (cols and len(cols) > 0) else (["Br", "Bt", "Bn"] if varname == "B_RTN" else ["Bx", "By", "Bz"])

            part = _tplot_to_df(MAGdata[0], cols)
            if len(part) == 0:
                continue

            dfmag = dfmag.join(part, how="outer") if len(dfmag) else part

        dfmag = _ensure_datetime_index(dfmag)
        if dfmag is None:
            return None
        return dfmag

    except Exception:
        traceback.print_exc()
        return None


def LoadSCAMFromSPEDAS_PSP(t0: str, t1: str, credentials: Dict[str, Any], settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    SCAM (search-coil) merged waveform product.
    Keeps exact column conventions used elsewhere:
      - RTN -> Br,Bt,Bn
      - SC  -> Bx,By,Bz
    """
    try:
        username = credentials["psp"]["fields"]["username"]
        password = credentials["psp"]["fields"]["password"]

        if bool(settings.get("in_rtn", True)):
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
        return _ensure_datetime_index(df)

    except Exception:
        traceback.print_exc()
        return None


def _apply_scam_wheel_noise_removal(diagnostics_MAG: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """
    EXACT behavior requested by you:
    - apply ONLY if Mag_SCAM_PSP["noise_flag"] is True
    - operate on diagnostics_MAG["resampled_df"]
    - uses turb.remove_wheel_noise for EACH column
    """
    try:
        if not settings.get("Mag_SCAM_PSP", {}).get("noise_flag", False):
            return

        df_res = diagnostics_MAG.get("resampled_df", None)
        if not isinstance(df_res, pd.DataFrame) or len(df_res) == 0:
            return

        logging.info("Removing wheel noise from SCAM/merged MAG data")

        dt = func.find_cadence(df_res)
        fs = 1.0 / dt if (dt is not None and dt > 0) else None
        if fs is None:
            return

        nr = settings["Mag_SCAM_PSP"]["noise_removal"]
        keys = list(df_res.columns)

        for key in keys:
            try:
                cleaned = turb.remove_wheel_noise(
                    df_res[key].values,
                    fs,
                    window_size=nr["window_size"],
                    avg_length=nr["avg_length"],
                    power_threshold=nr["power_threshold"],
                    freq_min=nr["freq_min"],
                    hampel_wind=nr.get("hampel_wind", 51),
                    hampel_thresh=nr.get("hampel_thresh", 3.5),
                )
                df_res[key] = cleaned
            except Exception:
                traceback.print_exc()

        diagnostics_MAG["resampled_df"] = df_res

    except Exception:
        traceback.print_exc()


def process_mag_field_data(
    t0: str,
    t1: str,
    settings: Dict[str, Any],
    credentials: Dict[str, Any],
    varnames_MAG: List[str],
    ind1,
    ind2,
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Dict[str, Any]]:
    """
    Returns:
      (dfmag_clipped, big_gaps_mag, diagnostics_MAG)
    """
    req_start = pd.to_datetime(t0)
    req_end = pd.to_datetime(t1)

    try:
        if settings.get("Mag_SCAM_PSP", {}).get("flag", False):
            logging.info("PSP MAG: using SCAM merged waveform product")
            dfmag = LoadSCAMFromSPEDAS_PSP(t0, t1, credentials, settings)
            mag_source = "SCAM_MERGED"
        else:
            logging.info("PSP MAG: using fluxgate product")
            dfmag = download_MAG_FIELD_PSP(t0, t1, credentials, varnames_MAG, settings)
            mag_source = "FLUXGATE"

        dfmag = _ensure_datetime_index(dfmag)
        if dfmag is None or len(dfmag) == 0:
            return None, None, _diag_default()

        dfmag = _clip_to_requested(dfmag, ind1, ind2, req_start, req_end)
        if dfmag is None or len(dfmag) == 0:
            return None, None, _diag_default()

        big_gaps = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], str(ind1), str(ind2))

        diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)
        diagnostics_MAG.setdefault("resampled_df", None)

        # Apply optional noise removal (requested)
        # This is meaningful ONLY when SCAM/merged data are used.
        if mag_source in ("SCAM_MERGED",):
            _apply_scam_wheel_noise_removal(diagnostics_MAG, settings)

        return dfmag, big_gaps, diagnostics_MAG

    except Exception:
        traceback.print_exc()
        return None, None, _diag_default()


# =============================================================================
# 5) QTN
# =============================================================================
def download_QTN_PSP(t0: str, t1: str, credentials: Dict[str, Any], varnames: List[str], settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        qtndata = None

        # private
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

        if not qtndata or len(qtndata) == 0:
            return None

        col_names = map_col_names_PSP("QTN", varnames)
        dfs = []
        for i, data in enumerate(qtndata):
            cols = col_names[i] if i < len(col_names) else [f"qtn_{i}"]
            dfs.append(pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=cols))

        dfqtn = pd.concat(dfs, axis=1)
        dfqtn = _ensure_datetime_index(dfqtn)

        if dfqtn is None or len(dfqtn) == 0:
            return None

        if "ne_qtn" in dfqtn.columns:
            dfqtn["np_qtn"] = dfqtn["ne_qtn"] * 0.96

        return dfqtn

    except Exception:
        traceback.print_exc()
        return None


def process_qtn_data(
    t0: str,
    t1: str,
    credentials: Dict[str, Any],
    varnames_QTN: List[str],
    ind1,
    ind2,
    settings: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str, Optional[Any]]:
    """
    Returns:
      (dfqtn_clipped, diagnostics_QTN, dfqtn_flag, big_gaps_qtn)
    """
    req_start = pd.to_datetime(t0)
    req_end = pd.to_datetime(t1)

    dfqtn = None

    # 1) Orlando merged pickle (if provided)
    try:
        qpath = settings.get("orlandos_QTN", None)
        if qpath:
            df_try = pd.read_pickle(qpath)
            df_try = _ensure_datetime_index(df_try)
            if isinstance(df_try, pd.DataFrame) and len(df_try) > 0:
                df_between = df_try.loc[req_start:req_end]
                if len(df_between) > 0:
                    dfqtn = df_between
    except Exception:
        dfqtn = None

    # 2) Legacy fallback pickles (now resolved from one configured root path)
    if dfqtn is None:
        try:
            qtn_root = Path(settings.get("Data_path", Path.cwd())) / "psp_data"
            qtn_paths = settings.get(
                "legacy_qtn_paths",
                [
                    qtn_root / "PSP_QTN_Monc" / "E22.pkl",
                    qtn_root / "PSP_QTN_Monc" / "E23.pkl",
                    qtn_root / "PSP_QTN_Romeo" / "save_pickled_dfs" / "e24.pkl",
                ],
            )
            qtn_paths = [Path(p) for p in qtn_paths]

            dfqtn1 = pd.read_pickle(qtn_paths[0])
            if "Te_qtn" in dfqtn1.columns:
                del dfqtn1["Te_qtn"]

            dfqtn2 = pd.read_pickle(qtn_paths[1])
            if "Te_qtn" in dfqtn2.columns:
                del dfqtn2["Te_qtn"]

            dfqtn3 = pd.read_pickle(qtn_paths[2])
            if "ne_qtn" in dfqtn3.columns:
                del dfqtn3["ne_qtn"]

            dfqtn_all = pd.concat([dfqtn1, dfqtn2, dfqtn3])
            dfqtn_all = _ensure_datetime_index(dfqtn_all)
            if isinstance(dfqtn_all, pd.DataFrame) and len(dfqtn_all) > 0:
                df_between = dfqtn_all.loc[req_start:req_end]
                if len(df_between) > 0:
                    dfqtn = df_between
        except Exception:
            dfqtn = None

    # 3) SPEDAS fallback
    if dfqtn is None:
        dfqtn = download_QTN_PSP(t0, t1, credentials, varnames_QTN, settings)

    dfqtn = _ensure_datetime_index(dfqtn)
    if dfqtn is None or len(dfqtn) == 0:
        diagnostics_QTN = _diag_default()
        diagnostics_QTN.update({"Frac_miss": None, "Large_gaps": None, "Tot_gaps": None, "resol": None})
        return None, diagnostics_QTN, "No QTN", None

    dfqtn = _clip_to_requested(dfqtn, ind1, ind2, req_start, req_end)
    dfqtn = _ensure_datetime_index(dfqtn)

    if dfqtn is None or len(dfqtn) == 0:
        diagnostics_QTN = _diag_default()
        diagnostics_QTN.update({"Frac_miss": None, "Large_gaps": None, "Tot_gaps": None, "resol": None})
        return None, diagnostics_QTN, "No QTN", None

    big_gaps = func.find_big_gaps(dfqtn, settings["Big_Gaps"]["QTN_big_gaps"], str(ind1), str(ind2))
    diagnostics_QTN = func.resample_timeseries_estimate_gaps(dfqtn, settings["part_resol"], large_gaps=10)
    diagnostics_QTN.setdefault("resampled_df", None)

    return dfqtn, diagnostics_QTN, "QTN", big_gaps


# =============================================================================
# 6) SPC + SPAN (particles)
# =============================================================================
def download_SPC_PSP(t0: str, t1: str, credentials: Dict[str, Any], varnames: List[str], settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
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
        for i, data in enumerate(spcdata):
            arr = get_data(data)
            if arr is None:
                continue
            cols = col_names[i] if i < len(col_names) else [f"spc_{i}"]
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        dfspc = pd.concat(dfs, axis=1)
        dfspc = _ensure_datetime_index(dfspc)

        if dfspc is None or len(dfspc) == 0:
            return None

        # Compute Dist_au if possible (SC position in km)
        if {"sc_x", "sc_y", "sc_z"}.issubset(dfspc.columns):
            dfspc["Dist_au"] = np.sqrt((dfspc[["sc_x", "sc_y", "sc_z"]] ** 2).sum(axis=1)) / au_to_km
            dfspc.drop(["sc_x", "sc_y", "sc_z"], axis=1, inplace=True)

        return dfspc

    except Exception:
        traceback.print_exc()
        return None


def process_spc_data(
    t0: str,
    t1: str,
    credentials: Dict[str, Any],
    varnames_SPC: List[str],
    settings: Dict[str, Any],
    ind1,
    ind2,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str, Optional[Any]]:
    req_start = pd.to_datetime(t0)
    req_end = pd.to_datetime(t1)

    try:
        dfspc = download_SPC_PSP(t0, t1, credentials, varnames_SPC, settings)
        dfspc = _ensure_datetime_index(dfspc)
        if dfspc is None or len(dfspc) == 0:
            return None, _diag_default(), "No SPC", None

        dfspc = _clip_to_requested(dfspc, ind1, ind2, req_start, req_end)
        dfspc = _ensure_datetime_index(dfspc)
        if dfspc is None or len(dfspc) == 0:
            return None, _diag_default(), "No SPC", None

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

#         # Tp estimation kept (best-effort)
#         try:
#             from astropy.constants import m_p as mp_ast, k_B
#             from astropy import units as u

#             dfspc["Tp"] = np.array(
#                 ((mp_ast * ((dfspc["Vth"].values * u.km / u.s).to(u.m / u.s) ** 2)) / (2 * k_B)).to(
#                     u.eV, equivalencies=u.temperature_energy()
#                 )
#             )
#         except Exception:
#             pass

        big_gaps_spc = func.find_big_gaps(dfspc, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
        diagnostics_SPC = func.resample_timeseries_estimate_gaps(dfspc, settings["part_resol"], large_gaps=10)
        diagnostics_SPC.setdefault("resampled_df", None)

        return dfspc, diagnostics_SPC, "SPC", big_gaps_spc

    except Exception:
        traceback.print_exc()
        return None, _diag_default(), "No SPC", None


def download_SPAN_PSP(
    t0: str,
    t1: str,
    credentials: Dict[str, Any],
    varnames: List[str],
    varnames_alpha: List[str],
    settings: Dict[str, Any],
) -> Optional[pd.DataFrame]:
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
                if spandata and len(spandata):
                    break
            except Exception:
                spandata = None

        if not spandata or len(spandata) == 0:
            return None

        col_names = map_col_names_PSP("SPAN", varnames)
        dfs = []
        for i, d in enumerate(spandata):
            arr = get_data(d)
            if arr is None:
                continue
            cols = col_names[i] if i < len(col_names) else [f"span_{i}"]
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        dfspan = pd.concat(dfs, axis=1)
        dfspan = _ensure_datetime_index(dfspan)
        if dfspan is None or len(dfspan) == 0:
            return None

        if "Dist_au" in dfspan.columns:
            dfspan["Dist_au"] = dfspan["Dist_au"] / au_to_km

        if "TEMP" in dfspan.columns:
            dfspan["Tp"] = dfspan.pop("TEMP")
            dfspan["Vth"] = 13.84112218 * np.sqrt(dfspan["Tp"]) / np.sqrt(3)

        return dfspan

    except Exception:
        traceback.print_exc()
        return None


def process_span_data(
    t0: str,
    t1: str,
    credentials: Dict[str, Any],
    varnames_SPAN: List[str],
    varnames_SPAN_alpha: List[str],
    settings: Dict[str, Any],
    ind1,
    ind2,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str, Optional[Any]]:
    req_start = pd.to_datetime(t0)
    req_end = pd.to_datetime(t1)

    try:
        dfspan = download_SPAN_PSP(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings)
        dfspan = _ensure_datetime_index(dfspan)
        if dfspan is None or len(dfspan) == 0:
            return None, _diag_default(), "No SPAN", None

        dfspan = _clip_to_requested(dfspan, ind1, ind2, req_start, req_end)
        dfspan = _ensure_datetime_index(dfspan)
        if dfspan is None or len(dfspan) == 0:
            return None, _diag_default(), "No SPAN", None

        if settings.get("apply_hampel", False):
            cols = (["Vr", "Vt", "Vn"] if "Vr" in dfspan.columns else ["Vx", "Vy", "Vz"])
            cols += ["np", "Vth", "Tp"]
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

        big_gaps_span = func.find_big_gaps(dfspan, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
        diagnostics_SPAN = func.resample_timeseries_estimate_gaps(dfspan, settings["part_resol"], large_gaps=10)
        diagnostics_SPAN.setdefault("resampled_df", None)

        return dfspan, diagnostics_SPAN, "SPAN", big_gaps_span

    except Exception:
        traceback.print_exc()
        return None, _diag_default(), "No SPAN", None


# =============================================================================
# 7) Ephemeris (PUBLIC API function required by download_data.py)
# =============================================================================
def download_ephemeris_PSP(
    t0: str,
    t1: str,
    credentials: Dict[str, Any],
    varnames: List[str],
    settings: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    PUBLIC API. Keep name and signature unchanged.
    """
    try:
        username = credentials["psp"]["fields"]["username"]
        password = credentials["psp"]["fields"]["password"]

        use_local = False
        if isinstance(settings, dict):
            use_local = bool(settings.get("use_local_data", False))

        ephemdata = pyspedas.psp.fields(
            trange=[t0, t1],
            datatype="ephem_spp_rtn",
            level="l1",
            varnames=varnames,
            time_clip=True,
            username=username,
            password=password,
            no_update=use_local,
        )

        if not ephemdata or len(ephemdata) == 0:
            return None

        col_names = map_col_names_PSP("EPHEMERIS", varnames)
        dfs = []
        for i, data in enumerate(ephemdata):
            arr = get_data(data)
            if arr is None:
                continue
            cols = col_names[i] if i < len(col_names) else [f"ephem_{i}"]
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=cols))

        if len(dfs) == 0:
            return None

        dfephem = pd.concat(dfs, axis=1)
        dfephem = _ensure_datetime_index(dfephem)
        if dfephem is None or len(dfephem) == 0:
            return None

        if {"sc_pos_r", "sc_pos_t", "sc_pos_n"}.issubset(dfephem.columns):
            dfephem["Dist_au"] = np.sqrt(
                np.sum(dfephem[["sc_pos_r", "sc_pos_t", "sc_pos_n"]] ** 2, axis=1)
            ) / au_to_km

        return dfephem

    except Exception:
        traceback.print_exc()
        return None


def process_ephemeris(
    t0: str,
    t1: str,
    credentials: Dict[str, Any],
    varnames_EPHEM: List[str],
    ind1,
    ind2,
    settings: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    try:
        req_start = pd.to_datetime(t0)
        req_end = pd.to_datetime(t1)

        dfephem = download_ephemeris_PSP(t0, t1, credentials, varnames_EPHEM, settings=settings)
        dfephem = _ensure_datetime_index(dfephem)
        if dfephem is None or len(dfephem) == 0:
            return None

        dfephem = _clip_to_requested(dfephem, ind1, ind2, req_start, req_end)
        dfephem = _ensure_datetime_index(dfephem)
        if dfephem is None or len(dfephem) == 0:
            return None

        return dfephem

    except Exception:
        return None


# =============================================================================
# 8) E-field + SC potential (optional features)
# =============================================================================
def download_efield(t0: str, t1: str, credentials: Dict[str, Any], varnames: List[str], settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
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

        cols = map_col_names_PSP("FIELDS-MAG", varnames)
        cols = cols[0] if (cols and len(cols) > 0) else ["dvx", "dvy"]
        df = _tplot_to_df(fields_vars[0], cols)
        return _ensure_datetime_index(df)

    except Exception:
        traceback.print_exc()
        return None


def process_e_field_data(
    t0: str,
    t1: str,
    settings: Dict[str, Any],
    credentials: Dict[str, Any],
    varnames: Optional[List[str]],
    ind1,
    ind2,
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Dict[str, Any]]:
    req_start = pd.to_datetime(t0)
    req_end = pd.to_datetime(t1)

    try:
        df = download_efield(t0, t1, credentials, varnames, settings)
        df = _ensure_datetime_index(df)
        if df is None or len(df) == 0:
            return None, None, _diag_default()

        df = _clip_to_requested(df, ind1, ind2, req_start, req_end)
        df = _ensure_datetime_index(df)
        if df is None or len(df) == 0:
            return None, None, _diag_default()

        big_gaps = func.find_big_gaps(df, settings["Big_Gaps"]["E_big_gaps"], str(ind1), str(ind2))
        diagnostics = func.resample_timeseries_estimate_gaps(df, 1, large_gaps=10)
        diagnostics.setdefault("resampled_df", None)
        return df, big_gaps, diagnostics

    except Exception:
        traceback.print_exc()
        return None, None, _diag_default()


def sc_potential_derived_density(t0: str, t1: str, credentials: Dict[str, Any], varnames: List[str], settings: Dict[str, Any]) -> Optional[pd.DataFrame]:
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
        df_try = _ensure_datetime_index(df_try)
        if isinstance(df_try, pd.DataFrame) and df_try.shape[1] == 4:
            return df_try

        # fallback: attempt per-variable
        dfs = []
        for i, tv in enumerate(fields_vars[:4]):
            col = [wanted_cols[i]] if i < len(wanted_cols) else [f"Vdc_{i+1}"]
            dfs.append(_tplot_to_df(tv, col))

        if len(dfs) == 0:
            return None

        df = pd.concat(dfs, axis=1)
        return _ensure_datetime_index(df)

    except Exception:
        traceback.print_exc()
        return None


def process_sc_potential_data(
    t0: str,
    t1: str,
    settings: Dict[str, Any],
    credentials: Dict[str, Any],
    varnames: Optional[List[str]],
    ind1,
    ind2,
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Dict[str, Any]]:
    req_start = pd.to_datetime(t0)
    req_end = pd.to_datetime(t1)

    try:
        df = sc_potential_derived_density(t0, t1, credentials, varnames, settings)
        df = _ensure_datetime_index(df)
        if df is None or len(df) == 0:
            return None, None, _diag_default()

        df = _clip_to_requested(df, ind1, ind2, req_start, req_end)
        df = _ensure_datetime_index(df)
        if df is None or len(df) == 0:
            return None, None, _diag_default()

        big_gaps = func.find_big_gaps(df, settings["Big_Gaps"]["SC_pot_big_gaps"], str(ind1), str(ind2))
        diagnostics = func.resample_timeseries_estimate_gaps(df, 1, large_gaps=10)
        diagnostics.setdefault("resampled_df", None)
        return df, big_gaps, diagnostics

    except Exception:
        traceback.print_exc()
        return None, None, _diag_default()


# =============================================================================
# 9) Particle selection + QTN integration (behavior preserved)
# =============================================================================
def create_particle_dataframe(
    PSP_distance_au: float,
    end_time,
    diagnostics_spc: Dict[str, Any],
    diagnostics_span: Dict[str, Any],
    dfqtn_resampled: Optional[pd.DataFrame],
    dfqtn_flag: str,
    big_gaps_span,
    big_gaps_spc,
    settings: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], str, str, Optional[Any]]:
    """
    Returns:
      (dfpar_selected, part_flag, dfqtn_flag_out, big_gaps_par_reference)
    """

    def integrate_qtn_data(source_df: pd.DataFrame, dfqtn_in: Optional[pd.DataFrame]):
        try:
            if source_df is None or not isinstance(source_df, pd.DataFrame) or len(source_df) == 0:
                return source_df, "No_QTN"

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
            traceback.print_exc()
            if "np" in source_df.columns:
                source_df["np_sweap"] = source_df["np"].copy()
            return source_df, "No_QTN"

    mode = settings.get("particle_mode", "9th_perih_cut")

    if mode == "9th_perih_cut":
        use_spc = pd.Timestamp(end_time) < pd.Timestamp("2021-07-15")
        df_selected = diagnostics_spc.get("resampled_df", None) if use_spc else diagnostics_span.get("resampled_df", None)
        big_gaps_ref = big_gaps_spc if use_spc else big_gaps_span
        part_flag = "spc" if use_spc else "span"

    elif settings.get("allow_max_SWEAP_distance", False):
        use_spc = (PSP_distance_au is not None) and (PSP_distance_au > settings.get("max_SWEAP_distance", 0.25))
        df_selected = diagnostics_spc.get("resampled_df", None) if use_spc else diagnostics_span.get("resampled_df", None)
        big_gaps_ref = big_gaps_spc if use_spc else big_gaps_span
        part_flag = "spc" if use_spc else "span"

    elif mode == "spc":
        df_selected = diagnostics_spc.get("resampled_df", None)
        big_gaps_ref = big_gaps_spc
        part_flag = "spc"

    elif mode == "span":
        df_selected = diagnostics_span.get("resampled_df", None)
        big_gaps_ref = big_gaps_span
        part_flag = "span"

    else:
        raise ValueError(f"Unsupported particle mode: {mode}")

    if df_selected is None or not isinstance(df_selected, pd.DataFrame) or len(df_selected) == 0:
        return None, part_flag, "No_QTN", big_gaps_ref

    try:
        df_selected = func.replace_negative_with_nan(df_selected)
    except Exception:
        pass

    df_selected, dfqtn_flag_out = integrate_qtn_data(df_selected, dfqtn_resampled)

    try:
        out = df_selected.interpolate().dropna()
    except Exception:
        out = df_selected

    return out, part_flag, dfqtn_flag_out, big_gaps_ref


# =============================================================================
# 10) MAIN: LoadTimeSeriesPSP (PUBLIC API, return order MUST MATCH)
# =============================================================================
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
    PUBLIC API.
    Return order MUST remain identical for compatibility with download_data.py.
    """
    try:
        settings = init_psp_settings(settings)

        # Preserve old behavior
        os.chdir(settings["Data_path"])
        Path("./psp_data").mkdir(exist_ok=True)

        # Requested interval strings
        t0i, t1i = func.ensure_time_format(start_time, end_time)

        # Expanded range for downloads
        t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
        t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

        # Small pad for QTN/ephem matching
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

        # ---------------------------------------------------------
        # QTN (optional unless must_have_qtn=True)
        # ---------------------------------------------------------
        dfqtn, diagnostics_QTN, dfqtn_flag, big_gaps_qtn = process_qtn_data(
            t0, t1, credentials, varnames_QTN, ind1_e, ind2_e, settings
        )

        # ---------------------------------------------------------
        # Ephemeris / distance
        # ---------------------------------------------------------
        dfdis = process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1_e, ind2_e, settings)

        mean_dist = np.nan
        try:
            if isinstance(dfdis, pd.DataFrame) and "Dist_au" in dfdis.columns and len(dfdis) > 0:
                mean_dist = float(np.nanmean(dfdis["Dist_au"].values))
                mean_dist = round(mean_dist, 2)
        except Exception:
            mean_dist = np.nan

        # Preserve your threshold logic
        max_dist = settings.get("max_PSP_dist", None)
        dist_threshold = True if max_dist is None else (mean_dist < float(max_dist))
        qtn_threshold = (dfqtn_flag == "QTN") or (settings.get("must_have_qtn", False) is False)

        if not (dist_threshold and qtn_threshold):
            if (dist_threshold is False) and (qtn_threshold is False):
                logging.info("Discarded (no QTN and d=%.3f au)", mean_dist)
            elif dist_threshold is False:
                logging.info("Discarded (d=%.3f au)", mean_dist)
            elif qtn_threshold is False:
                logging.info("Discarded (no QTN)")
            return (None, None, None, None, None, None, None, None, None, None, None)

        # ---------------------------------------------------------
        # Optional SC potential
        # ---------------------------------------------------------
        if vars_2_downnload.get("sc_pot", False):
            df_sc_pot, big_gaps_sc_pot, diagnostics_sc_pot = process_sc_potential_data(
                t0, t1, settings, credentials, varnames_SC_pot, ind1, ind2
            )
        else:
            df_sc_pot, big_gaps_sc_pot, diagnostics_sc_pot = None, None, _diag_default()

        # ---------------------------------------------------------
        # Optional E-field
        # ---------------------------------------------------------
        if vars_2_downnload.get("E_field", False):
            df_e_field, big_gaps_e_field, diagnostics_e_field = process_e_field_data(
                t0, t1, settings, credentials, varnames_E_field, ind1, ind2
            )
        else:
            df_e_field, big_gaps_e_field, diagnostics_e_field = None, None, _diag_default()

        # ---------------------------------------------------------
        # MAG (required)
        # ---------------------------------------------------------
        dfmag, big_gaps_mag, diagnostics_MAG = process_mag_field_data(
            t0, t1, settings, credentials, varnames_MAG, ind1, ind2
        )
        if dfmag is None or len(dfmag) == 0:
            # must return 11 items
            return (None, None, None, None, None, dfdis, None, big_gaps_qtn, None, big_gaps_sc_pot, None)

        # ---------------------------------------------------------
        # SPAN / SPC (both downloaded depending on particle_mode)
        # ---------------------------------------------------------
        if settings.get("particle_mode", "9th_perih_cut") in {"span", "9th_perih_cut"}:
            dfspan, diagnostics_SPAN, _, big_gaps_span = process_span_data(
                t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1_e, ind2_e
            )
        else:
            dfspan, diagnostics_SPAN, big_gaps_span = None, _diag_default(), None

        if settings.get("particle_mode", "9th_perih_cut") in {"spc", "9th_perih_cut"}:
            dfspc, diagnostics_SPC, _, big_gaps_spc = process_spc_data(
                t0, t1, credentials, varnames_SPC, settings, ind1_e, ind2_e
            )
        else:
            dfspc, diagnostics_SPC, big_gaps_spc = None, _diag_default(), None

        # ---------------------------------------------------------
        # Select particle source + inject QTN density (unchanged)
        # ---------------------------------------------------------
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

        if dfpar is None or not isinstance(dfpar, pd.DataFrame) or len(dfpar) == 0:
            return (diagnostics_QTN.get("resampled_df", None), diagnostics_MAG.get("resampled_df", None),
                    None, diagnostics_e_field.get("resampled_df", None), diagnostics_sc_pot.get("resampled_df", None),
                    dfdis, big_gaps_mag, big_gaps_qtn, None, big_gaps_sc_pot, None)

        diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
        diagnostics_PAR.setdefault("resampled_df", None)

        # Keep misc structure identical
        misc = {
            "SPC": _keep_diag_keys(diagnostics_SPC),
            "SPAN": _keep_diag_keys(diagnostics_SPAN),
            "QTN": _keep_diag_keys(diagnostics_QTN),
            "Par": _keep_diag_keys(diagnostics_PAR),
            "E": _keep_diag_keys(diagnostics_e_field),
            "SC_pot": _keep_diag_keys(diagnostics_sc_pot),
            "Mag": _keep_diag_keys(diagnostics_MAG),
            "part_flag": part_flag,
            "qtn_flag": dfqtn_flag2,
        }

        # If no QTN was integrated, restore "np" key behavior
        if dfqtn_flag2 == "No_QTN":
            try:
                par_res = diagnostics_PAR.get("resampled_df", None)
                if isinstance(par_res, pd.DataFrame) and "np_sweap" in par_res.columns and "np" not in par_res.columns:
                    par_res["np"] = par_res.pop("np_sweap")
                    diagnostics_PAR["resampled_df"] = par_res
            except Exception:
                pass

        # ---------------------------------------------------------
        # RETURN ORDER (MUST MATCH EXACTLY)
        # ---------------------------------------------------------
        return (
            diagnostics_QTN.get("resampled_df", None),
            diagnostics_MAG.get("resampled_df", None),
            diagnostics_PAR.get("resampled_df", None),
            diagnostics_e_field.get("resampled_df", None),
            diagnostics_sc_pot.get("resampled_df", None),
            dfdis.interpolate() if isinstance(dfdis, pd.DataFrame) else dfdis,
            big_gaps_mag,
            big_gaps_qtn,
            big_gaps_par,
            big_gaps_sc_pot,
            misc,
        )

    except Exception:
        traceback.print_exc()
        return (None, None, None, None, None, None, None, None, None, None, None)





# # ============================================================
# # PSP.py  (REVISED, CLEAN, ROBUST, FIXED)
# #
# # - Preserves the public API + I/O formats you had.
# # - Centralizes defaults in init_psp_settings().
# # - FIXES numeric-time -> datetime bug (major).
# # - FIXES MAG mapping when falling back to public datatypes.
# # ============================================================

# from __future__ import annotations

# import os
# import sys
# import logging
# import traceback
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import numpy as np
# import pandas as pd

# from matplotlib import pyplot as plt
# import matplotlib.dates as mdates
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from scipy import constants

# # ============================================================
# # Terminal colors (kept)
# # ============================================================
# BG_WHITE = "\033[47m"
# RESET = "\033[0m"
# BG_RED = "\033[41m"
# BG_GREEN = "\033[42m"
# BG_YELLOW = "\033[43m"
# BG_BLUE = "\033[44m"
# BG_MAGENTA = "\033[45m"
# BG_CYAN = "\033[46m"

# # ============================================================
# # Logging (single authoritative config)
# # ============================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s: %(message)s",
#     datefmt="%d-%b-%y %H:%M:%S",
# )

# # ============================================================
# # Local PySPEDAS path
# # ============================================================
# _LOCAL_PYSPEDAS = "/pyspedas"
# if _LOCAL_PYSPEDAS not in sys.path and Path(_LOCAL_PYSPEDAS).exists():
#     sys.path.insert(0, _LOCAL_PYSPEDAS)

# import pyspedas
# from pyspedas.utilities import time_string
# from pytplot import get_data

# # ============================================================
# # Your manual functions (must exist)
# # ============================================================
# sys.path.insert(1, os.path.join(os.getcwd(), "functions"))
# import general_functions as func
# import TurbPy as turb

# # ============================================================
# # Constants (kept)
# # ============================================================
# au_to_km = 1.496e8
# rsun = 696340
# mu0 = constants.mu_0
# mu_0 = constants.mu_0
# m_p = constants.m_p
# k = constants.k
# au_to_rsun = 215.032
# T_to_Gauss = 1e4

# _MAG_HIGHRES_THRESHOLD = 230


# # ============================================================
# # Internal helpers
# # ============================================================
# def _diag_default() -> Dict[str, Any]:
#     return {
#         "Init_dt": np.nan,
#         "resampled_df": None,
#         "Frac_miss": 100,
#         "Large_gaps": 100,
#         "Tot_gaps": 100,
#         "resol": 100,
#     }


# def _is_numeric_index(idx) -> bool:
#     try:
#         return np.issubdtype(idx.dtype, np.number)
#     except Exception:
#         return False


# def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     FIXED:
#     - If index is numeric Unix seconds (SPEDAS), pd.to_datetime() is WRONG.
#     - Use time_string.time_datetime() instead.
#     """
#     if df is None or not isinstance(df, pd.DataFrame):
#         return df

#     if not isinstance(df.index, pd.DatetimeIndex):
#         idx = df.index

#         # --- SPEDAS numeric time arrays (Unix seconds) ---
#         if _is_numeric_index(idx):
#             try:
#                 dt = time_string.time_datetime(time=idx)
#                 df.index = pd.DatetimeIndex(dt)
#             except Exception:
#                 df.index = pd.to_datetime(idx, errors="coerce")

#         # --- already string-like / datetime-like ---
#         else:
#             df.index = pd.to_datetime(idx, errors="coerce")

#     df = df.sort_index()
#     df = df.loc[~df.index.duplicated(keep="first")]
#     df.index = df.index.tz_localize(None)
#     df.index.name = "datetime"
#     return df


# def _clip_df_pandas(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
#     df = _ensure_datetime_index(df)
#     if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
#         return pd.DataFrame()
#     return df.loc[(df.index >= start) & (df.index <= end)]


# def _clip_to_requested(df: Optional[pd.DataFrame], ind1, ind2, req_start: pd.Timestamp, req_end: pd.Timestamp) -> Optional[pd.DataFrame]:
#     """
#     Use your original slicer, with a correct pandas fallback.
#     """
#     if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
#         return None

#     # 1) your original helper
#     try:
#         out = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, df)
#         if isinstance(out, pd.DataFrame) and len(out) > 0:
#             return out
#     except Exception:
#         pass

#     # 2) fallback
#     out = _clip_df_pandas(df, req_start, req_end)
#     return out if len(out) > 0 else pd.DataFrame()


# def _tplot_to_df(tplot_var: str, columns: List[str]) -> pd.DataFrame:
#     """
#     Robust converter from pytplot variable -> DataFrame.
#     """
#     arr = get_data(tplot_var)
#     if arr is None:
#         return pd.DataFrame()

#     times = getattr(arr, "times", None)
#     y = getattr(arr, "y", None)

#     if times is None or y is None:
#         try:
#             times = arr[0]
#             y = arr[1]
#         except Exception:
#             return pd.DataFrame()

#     try:
#         df = pd.DataFrame(index=times, data=y)
#         if df.shape[1] == len(columns):
#             df.columns = columns
#         else:
#             if len(columns) == 1:
#                 df.columns = columns
#             else:
#                 df.columns = columns[: df.shape[1]]

#         df = _ensure_datetime_index(df)
#         return df
#     except Exception:
#         return pd.DataFrame()


# def init_psp_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Single authoritative defaults.
#     """
#     if not isinstance(settings, dict):
#         raise TypeError("settings must be a dict")
#     if "Data_path" not in settings:
#         raise KeyError("settings must include 'Data_path'")

#     defaults = {
#         "particle_mode": "9th_perih_cut",
#         "apply_hampel": True,
#         "hampel_params": {"w": 100, "std": 3},
#         "part_resol": 900,
#         "MAG_resol": 1,
#         "use_local_data": False,
#         "in_rtn": True,
#         "must_have_qtn": False,
#         "max_PSP_dist": None,
#         "allow_max_SWEAP_distance": False,
#         "max_SWEAP_distance": 0.25,
#         "span_key": "spi_sf00",
#         "Mag_SCAM_PSP": {
#             "flag": False,
#             "noise_flag": False,
#             "noise_removal": {
#                 "window_size": 2048,
#                 "avg_length": 16,
#                 "power_threshold": 6.0,
#                 "freq_min": 10.0,
#                 "hampel_wind": 51,
#                 "hampel_thresh": 3.5,
#             },
#         },
#     }

#     out = {**defaults, **settings}

#     if "Mag_SCAM_PSP" not in out or not isinstance(out["Mag_SCAM_PSP"], dict):
#         out["Mag_SCAM_PSP"] = defaults["Mag_SCAM_PSP"]

#     if "Big_Gaps" not in out:
#         out["Big_Gaps"] = {
#             "Mag_big_gaps": 10,
#             "Par_big_gaps": 10,
#             "QTN_big_gaps": 10,
#             "E_big_gaps": 10,
#             "SC_pot_big_gaps": 10,
#         }

#     return out


# # ============================================================
# # Public API: variable lists
# # ============================================================
# def default_variables_to_download_PSP(vars_2_downnload):
#     if vars_2_downnload["mag"] is None:
#         varnames_MAG = ["B_RTN"]
#     else:
#         varnames_MAG = vars_2_downnload["mag"]

#     if vars_2_downnload["qtn"] is None:
#         varnames_QTN = ["electron_density", "electron_core_temperature"]
#     else:
#         varnames_QTN = vars_2_downnload["qtn"]

#     if vars_2_downnload["span"] is None:
#         varnames_SPAN = ["DENS", "VEL_RTN_SUN", "TEMP", "SUN_DIST", "SC_VEL_RTN_SUN"]
#     else:
#         varnames_SPAN = vars_2_downnload["span"]

#     if vars_2_downnload["spc"] is None:
#         varnames_SPC = ["np_moment", "wp_moment", "vp_moment_RTN", "sc_pos_HCI", "carr_longitude", "general_flag"]
#     else:
#         varnames_SPC = vars_2_downnload["spc"]

#     if vars_2_downnload["span-a"] is None:
#         varnames_SPAN_alpha = ["DENS"]
#     else:
#         varnames_SPAN_alpha = vars_2_downnload["span-a"]

#     if vars_2_downnload["ephem"] is None:
#         varnames_EPHEM = ["position", "velocity"]
#     else:
#         varnames_EPHEM = vars_2_downnload["ephem"]

#     if vars_2_downnload.get("E_field", False):
#         varnames_E_field = ["psp_fld_l2_dfb_wf_dVdc_sc"]
#     else:
#         varnames_E_field = None

#     if vars_2_downnload.get("sc_pot", False):
#         varnames_SC_pot = ["dfb_wf_vdc"]
#     else:
#         varnames_SC_pot = None

#     return (
#         varnames_MAG,
#         varnames_QTN,
#         varnames_SPAN,
#         varnames_SPC,
#         varnames_SPAN_alpha,
#         varnames_EPHEM,
#         varnames_E_field,
#         varnames_SC_pot,
#     )


# # ============================================================
# # Public API: column mapping
# # ============================================================
# def map_col_names_PSP(instrument, varnames):
#     fields_MAG_cols = {
#         "mag_RTN_4_Sa_per_Cyc": ["Br", "Bt", "Bn"],
#         "mag_SC_4_Sa_per_Cyc": ["Bx", "By", "Bz"],
#         "mag_rtn_4_per_cycle": ["Br", "Bt", "Bn"],
#         "mag_sc_4_per_cycle": ["Bx", "By", "Bz"],
#         "mag_RTN": ["Br", "Bt", "Bn"],
#         "mag_SC": ["Bx", "By", "Bz"],
#         "mag_rtn": ["Br", "Bt", "Bn"],
#         "mag_sc": ["Bx", "By", "Bz"],
#         "psp_fld_l2_dfb_wf_dVdc_sc": ["dvx", "dvy"],
#         "dfb_wf_vdc": [
#             "psp_fld_l2_dfb_wf_V1dc",
#             "psp_fld_l2_dfb_wf_V2dc",
#             "psp_fld_l2_dfb_wf_V3dc",
#             "psp_fld_l2_dfb_wf_V4dc",
#         ],
#     }

#     fields_QTN_cols = {
#         "electron_density": ["ne_qtn"],
#         "electron_core_temperature": ["Te_qtn"],
#     }

#     spc_cols = {
#         "np_moment": ["np"],
#         "wp_moment": ["Vth"],
#         "vp_moment_RTN": ["Vr", "Vt", "Vn"],
#         "vp_moment_SC": ["Vx", "Vy", "Vz"],
#         "sc_pos_HCI": ["sc_x", "sc_y", "sc_z"],
#         "sc_vel_HCI": ["sc_vel_x", "sc_vel_y", "sc_vel_z"],
#         "carr_latitude": ["carr_lat"],
#         "carr_longitude": ["carr_lon"],
#         "general_flag": ["flag"],
#     }

#     span_cols = {
#         "DENS": ["np"],
#         "VEL_SC": ["Vx", "Vy", "Vz"],
#         "VEL_RTN_SUN": ["Vr", "Vt", "Vn"],
#         "TEMP": ["TEMP"],
#         "SUN_DIST": ["Dist_au"],
#         "SC_VEL_RTN_SUN": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
#     }

#     span_alpha_cols = {"DENS": ["na"]}

#     ephem_cols = {
#         "position": ["sc_pos_r", "sc_pos_t", "sc_pos_n"],
#         "velocity": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
#     }

#     if instrument == "SPC":
#         return [spc_cols[var] for var in varnames if var in spc_cols]
#     if instrument == "FIELDS-MAG":
#         return [fields_MAG_cols[var] for var in varnames if var in fields_MAG_cols]
#     if instrument == "QTN":
#         return [fields_QTN_cols[var] for var in varnames if var in fields_QTN_cols]
#     if instrument == "SPAN":
#         return [span_cols[var] for var in varnames if var in span_cols]
#     if instrument == "SPAN-alpha":
#         return [span_alpha_cols[var] for var in varnames if var in span_alpha_cols]
#     if instrument == "EPHEMERIS":
#         return [ephem_cols[var] for var in varnames if var in ephem_cols]
#     return []


# # ============================================================
# # MAG download (FIXED used_datatype tracking)
# # ============================================================
# def download_MAG_FIELD_PSP(t0, t1, credentials, varnames, settings):
#     try:
#         dfmag = pd.DataFrame()

#         for varname in varnames:
#             # Decide frame + resol
#             if varname == "B_RTN":
#                 datatype_private = "mag_RTN_4_Sa_per_Cyc" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_RTN"
#                 datatype_public = "mag_rtn_4_per_cycle" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_rtn"
#             else:
#                 datatype_private = "mag_SC_4_Sa_per_Cyc" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_SC"
#                 datatype_public = "mag_sc_4_per_cycle" if settings["MAG_resol"] > _MAG_HIGHRES_THRESHOLD else "mag_sc"

#             MAGdata = None
#             used_datatype = None

#             # private
#             try:
#                 username = credentials["psp"]["fields"]["username"]
#                 password = credentials["psp"]["fields"]["password"]
#                 MAGdata = pyspedas.psp.fields(
#                     trange=[t0, t1],
#                     datatype=datatype_private,
#                     level="l2",
#                     time_clip=True,
#                     username=username,
#                     password=password,
#                     no_update=settings["use_local_data"],
#                 )
#                 if MAGdata and len(MAGdata):
#                     used_datatype = datatype_private
#             except Exception:
#                 MAGdata = None
#                 used_datatype = None

#             # public fallback
#             if not MAGdata or len(MAGdata) == 0:
#                 MAGdata = pyspedas.psp.fields(
#                     trange=[t0, t1],
#                     datatype=datatype_public,
#                     level="l2",
#                     time_clip=True,
#                     no_update=settings["use_local_data"],
#                 )
#                 if MAGdata and len(MAGdata):
#                     used_datatype = datatype_public

#             if not MAGdata or len(MAGdata) == 0 or used_datatype is None:
#                 continue

#             col_names = map_col_names_PSP("FIELDS-MAG", [used_datatype])
#             cols = col_names[0] if col_names else (["Br", "Bt", "Bn"] if varname == "B_RTN" else ["Bx", "By", "Bz"])

#             df_part = _tplot_to_df(MAGdata[0], cols)
#             dfmag = dfmag.join(df_part, how="outer") if len(dfmag) else df_part

#         dfmag = _ensure_datetime_index(dfmag)
#         return dfmag.drop_duplicates().sort_index()

#     except Exception as e:
#         logging.exception("Error occurred while retrieving MAG data: %s", e)
#         return None


# def process_mag_field_data(t0, t1, settings, credentials, varnames_MAG, ind1, ind2):
#     try:
#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         if settings["Mag_SCAM_PSP"]["flag"]:
#             logging.info("Working on SCAM mag data")
#             dfmag = LoadSCAMFromSPEDAS_PSP(t0, t1, credentials, settings)
#         else:
#             logging.info("Working on fluxgate mag data")
#             dfmag = download_MAG_FIELD_PSP(t0, t1, credentials, varnames_MAG, settings)

#         if dfmag is None or len(dfmag) == 0:
#             return None, None, _diag_default()

#         dfmag = _clip_to_requested(dfmag, ind1, ind2, req_start, req_end)
#         if dfmag is None or len(dfmag) == 0:
#             return None, None, _diag_default()

#         big_gaps = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], str(ind1), str(ind2))
#         diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)
#         diagnostics_MAG.setdefault("resampled_df", None)

#         # Optional SCAM noise removal
#         if settings["Mag_SCAM_PSP"]["noise_flag"] and isinstance(diagnostics_MAG.get("resampled_df", None), pd.DataFrame):
#             logging.info("Removing wheel noise from SCAM data")
#             dt = func.find_cadence(diagnostics_MAG["resampled_df"])
#             fs = 1.0 / dt if (dt is not None and dt > 0) else None

#             if fs is not None:
#                 keys = list(diagnostics_MAG["resampled_df"].columns)
#                 nr = settings["Mag_SCAM_PSP"]["noise_removal"]

#                 for key in keys:
#                     try:
#                         cleaned = turb.remove_wheel_noise(
#                             diagnostics_MAG["resampled_df"][key].values,
#                             fs,
#                             window_size=nr["window_size"],
#                             avg_length=nr["avg_length"],
#                             power_threshold=nr["power_threshold"],
#                             freq_min=nr["freq_min"],
#                             hampel_wind=nr.get("hampel_wind", 51),
#                             hampel_thresh=nr.get("hampel_thresh", 3.5),
#                         )
#                         diagnostics_MAG["resampled_df"][key] = cleaned
#                     except Exception:
#                         traceback.print_exc()

#         return dfmag.drop_duplicates(), big_gaps, diagnostics_MAG

#     except Exception as e:
#         logging.exception("MAG processing failed: %s", e)
#         return None, None, _diag_default()


# # ============================================================
# # SPC (unchanged I/O)
# # ============================================================
# def download_SPC_PSP(t0, t1, credentials, varnames, settings):
#     try:
#         spcdata = None

#         # private
#         try:
#             username = credentials["psp"]["sweap"]["username"]
#             password = credentials["psp"]["sweap"]["password"]
#             spcdata = pyspedas.psp.spc(
#                 trange=[t0, t1],
#                 datatype="l3i",
#                 level="L3",
#                 varnames=varnames,
#                 time_clip=True,
#                 username=username,
#                 password=password,
#                 no_update=settings["use_local_data"],
#             )
#         except Exception:
#             spcdata = None

#         # public fallback
#         if not spcdata or len(spcdata) == 0:
#             spcdata = pyspedas.psp.spc(
#                 trange=[t0, t1],
#                 datatype="l3i",
#                 level="l3",
#                 varnames=varnames,
#                 time_clip=True,
#                 no_update=settings["use_local_data"],
#             )

#         if not spcdata or len(spcdata) == 0:
#             return None

#         col_names = map_col_names_PSP("SPC", varnames)
#         dfs = []
#         for i, data in enumerate(spcdata):
#             cols = col_names[i] if i < len(col_names) else [f"spc_{i}"]
#             dfs.append(pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=cols))

#         dfspc = pd.concat(dfs, axis=1)
#         dfspc = _ensure_datetime_index(dfspc)

#         if {"sc_x", "sc_y", "sc_z"}.issubset(dfspc.columns):
#             dfspc["Dist_au"] = np.sqrt((dfspc[["sc_x", "sc_y", "sc_z"]] ** 2).sum(axis=1)) / au_to_km
#             dfspc.drop(["sc_x", "sc_y", "sc_z"], axis=1, inplace=True)

#         return dfspc.drop_duplicates()

#     except Exception as e:
#         logging.exception("SPC download failed: %s", e)
#         return None


# def process_spc_data(t0, t1, credentials, varnames_SPC, settings, ind1=None, ind2=None):
#     try:
#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         dfspc = download_SPC_PSP(t0, t1, credentials, varnames_SPC, settings)
#         if dfspc is None or len(dfspc) == 0:
#             raise ValueError("No SPC data returned")

#         dfspc = _clip_to_requested(dfspc, ind1, ind2, req_start, req_end)
#         if dfspc is None or len(dfspc) == 0:
#             raise ValueError("SPC has no overlap with requested interval")

#         if settings["apply_hampel"]:
#             cols = ["Vr", "Vt", "Vn", "np", "Vth"] if "Vr" in dfspc.columns else ["Vx", "Vy", "Vz", "np", "Vth"]
#             ws = settings["hampel_params"]["w"]
#             nn = settings["hampel_params"]["std"]

#             for c in cols:
#                 if c not in dfspc.columns:
#                     continue
#                 try:
#                     out_idx = func.hampel(dfspc[c], window_size=ws, n=nn)
#                     if isinstance(out_idx, tuple) and len(out_idx) == 2:
#                         out_idx = out_idx[1]
#                     dfspc.loc[dfspc.index[out_idx], c] = np.nan
#                 except Exception:
#                     traceback.print_exc()

#         # Tp estimation kept (best-effort)
#         try:
#             from astropy.constants import m_p as mp_ast, k_B
#             from astropy import units as u

#             dfspc["Tp"] = np.array(
#                 ((mp_ast * ((dfspc["Vth"].values * u.km / u.s).to(u.m / u.s) ** 2)) / (2 * k_B)).to(
#                     u.eV, equivalencies=u.temperature_energy()
#                 )
#             )
#         except Exception:
#             pass

#         big_gaps_spc = func.find_big_gaps(dfspc, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
#         diagnostics_SPC = func.resample_timeseries_estimate_gaps(dfspc, settings["part_resol"], large_gaps=10)
#         diagnostics_SPC.setdefault("resampled_df", None)

#         return dfspc.drop_duplicates(), diagnostics_SPC, "SPC", big_gaps_spc

#     except Exception as e:
#         logging.exception("SPC processing failed: %s", e)
#         return None, _diag_default(), "No SPC", None


# # ============================================================
# # SPAN (kept API)
# # ============================================================
# def download_SPAN_PSP(t0, t1, credentials, varnames, varnames_alpha, settings):
#     try:
#         span_key = settings.get("span_key", "spi_sf00").lower()
#         use_local = settings.get("use_local_data", False)

#         products = (
#             [span_key, "spi_sf00_l3_mom" if span_key == "spi_sf00" else "spi_sf00"]
#             if span_key in {"spi_sf00", "spi_sf00_l3_mom"}
#             else ["spi_sf00", "spi_sf00_l3_mom"]
#         )

#         spandata = None

#         for key in products:
#             try:
#                 if key == "spi_sf00_l3_mom":
#                     qvars = [f"psp_spi_{v}" for v in varnames]
#                     spandata = pyspedas.psp.spi(
#                         trange=[t0, t1],
#                         datatype="spi_sf00_l3_mom",
#                         level="l3",
#                         varnames=qvars,
#                         time_clip=True,
#                         no_update=use_local,
#                     )
#                 else:
#                     user = credentials["psp"]["sweap"]["username"]
#                     pwd = credentials["psp"]["sweap"]["password"]
#                     spandata = pyspedas.psp.spi(
#                         trange=[t0, t1],
#                         datatype="spi_sf00",
#                         level="L3",
#                         varnames=varnames,
#                         time_clip=True,
#                         username=user,
#                         password=pwd,
#                         no_update=use_local,
#                     )
#                 if spandata and len(spandata):
#                     break
#             except Exception:
#                 spandata = None

#         if not spandata or len(spandata) == 0:
#             return None

#         col_names = map_col_names_PSP("SPAN", varnames)
#         dfs = []
#         for i, d in enumerate(spandata):
#             cols = col_names[i] if i < len(col_names) else [f"span_{i}"]
#             dfs.append(pd.DataFrame(index=get_data(d).times, data=get_data(d).y, columns=cols))

#         dfspan = pd.concat(dfs, axis=1)
#         dfspan = _ensure_datetime_index(dfspan)

#         if "Dist_au" in dfspan.columns:
#             dfspan["Dist_au"] = dfspan["Dist_au"] / au_to_km

#         if "TEMP" in dfspan.columns:
#             dfspan["Tp"] = dfspan.pop("TEMP")
#             dfspan["Vth"] = 13.84112218 * np.sqrt(dfspan["Tp"]) / np.sqrt(3)

#         return dfspan.drop_duplicates()

#     except Exception:
#         traceback.print_exc()
#         return None


# def process_span_data(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1, ind2):
#     try:
#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         dfspan = download_SPAN_PSP(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings)
#         if dfspan is None or len(dfspan) == 0:
#             return None, _diag_default(), "No SPAN", None

#         if settings.get("apply_hampel", False):
#             cols = (["Vr", "Vt", "Vn"] if "Vr" in dfspan.columns else ["Vx", "Vy", "Vz"])
#             cols += ["np", "Vth", "Tp"]

#             ws = settings["hampel_params"]["w"]
#             nn = settings["hampel_params"]["std"]

#             for c in cols:
#                 if c not in dfspan.columns:
#                     continue
#                 try:
#                     out_idx = func.hampel(dfspan[c], window_size=ws, n=nn)
#                     if isinstance(out_idx, tuple) and len(out_idx) == 2:
#                         out_idx = out_idx[1]
#                     dfspan.loc[dfspan.index[out_idx], c] = np.nan
#                 except Exception:
#                     traceback.print_exc()

#         dfspan = _clip_to_requested(dfspan, ind1, ind2, req_start, req_end)
#         if dfspan is None or len(dfspan) == 0:
#             return None, _diag_default(), "No SPAN", None

#         big_gaps_span = func.find_big_gaps(dfspan, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
#         diagnostics_SPAN = func.resample_timeseries_estimate_gaps(dfspan, settings["part_resol"], large_gaps=10)
#         diagnostics_SPAN.setdefault("resampled_df", None)

#         return dfspan.drop_duplicates(), diagnostics_SPAN, "SPAN", big_gaps_span

#     except Exception as e:
#         logging.exception("SPAN processing failed: %s", e)
#         return None, _diag_default(), "No SPAN", None


# # ============================================================
# # QTN
# # ============================================================
# def download_QTN_PSP(t0, t1, credentials, varnames, settings):
#     try:
#         qtndata = None

#         try:
#             username = credentials["psp"]["fields"]["username"]
#             password = credentials["psp"]["fields"]["password"]
#             qtndata = pyspedas.psp.fields(
#                 trange=[t0, t1],
#                 datatype="sqtn_rfs_V1V2",
#                 level="l3",
#                 varnames=varnames,
#                 time_clip=True,
#                 username=username,
#                 password=password,
#                 no_update=settings["use_local_data"],
#             )
#             if qtndata == []:
#                 qtndata = pyspedas.psp.fields(
#                     trange=[t0, t1],
#                     datatype="rfs_lfr_qtn",
#                     level="l2",
#                     time_clip=True,
#                     username=username,
#                     password=password,
#                     no_update=settings["use_local_data"],
#                 )
#         except Exception:
#             qtndata = None

#         if not qtndata or len(qtndata) == 0:
#             qtndata = pyspedas.psp.fields(
#                 trange=[t0, t1],
#                 datatype="sqtn_rfs_v1v2",
#                 level="l3",
#                 varnames=varnames,
#                 time_clip=True,
#                 no_update=settings["use_local_data"],
#             )

#         if not qtndata or len(qtndata) == 0:
#             return None

#         col_names = map_col_names_PSP("QTN", varnames)
#         dfs = []
#         for i, data in enumerate(qtndata):
#             cols = col_names[i] if i < len(col_names) else [f"qtn_{i}"]
#             dfs.append(pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=cols))

#         dfqtn = pd.concat(dfs, axis=1)
#         dfqtn = _ensure_datetime_index(dfqtn)

#         if "ne_qtn" in dfqtn.columns:
#             dfqtn["np_qtn"] = dfqtn["ne_qtn"] * 0.96

#         return dfqtn.drop_duplicates()

#     except Exception as e:
#         logging.exception("QTN download failed: %s", e)
#         return None


# def process_qtn_data(t0, t1, credentials, varnames_QTN, ind1, ind2, settings):
#     dfqtn = None

#     # Orlando QTN pickle (optional)
#     try:
#         if settings.get("orlandos_QTN", None) is not None:
#             logging.info("Attempting Orlando QTN pickle...")
#             dfqtn = pd.read_pickle(settings["orlandos_QTN"])
#             dfqtn = _ensure_datetime_index(dfqtn)
#             df_between = dfqtn.loc[t0:t1]
#             dfqtn = df_between if len(df_between) > 0 else None
#     except Exception:
#         dfqtn = None

#     # Hardcoded fallback pickles (kept)
#     if dfqtn is None:
#         try:
#             dfqtn = pd.read_pickle("/Users/turbulator/work/MHDTurbPy/psp_data/PSP_QTN_Monc/E22.pkl")
#             if "Te_qtn" in dfqtn.columns:
#                 del dfqtn["Te_qtn"]

#             dfqtn2 = pd.read_pickle("/Users/turbulator/work/MHDTurbPy/psp_data/PSP_QTN_Monc/E23.pkl")
#             if "Te_qtn" in dfqtn2.columns:
#                 del dfqtn2["Te_qtn"]

#             dfqtn3 = pd.read_pickle("/Users/turbulator/work/MHDTurbPy/psp_data/PSP_QTN_Romeo/save_pickled_dfs/e24.pkl")
#             if "ne_qtn" in dfqtn3.columns:
#                 del dfqtn3["ne_qtn"]

#             dfqtn = pd.concat([dfqtn, dfqtn2, dfqtn3])
#             dfqtn = _ensure_datetime_index(dfqtn)
#             df_between = dfqtn.loc[t0:t1]
#             dfqtn = df_between if len(df_between) > 0 else None
#         except Exception:
#             dfqtn = None

#     # final fallback to SPEDAS
#     if dfqtn is None:
#         dfqtn = download_QTN_PSP(t0, t1, credentials, varnames_QTN, settings)

#     try:
#         if dfqtn is None or len(dfqtn) == 0:
#             raise ValueError("No QTN data found")

#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         dfqtn = _clip_to_requested(dfqtn, ind1, ind2, req_start, req_end)
#         if dfqtn is None or len(dfqtn) == 0:
#             raise ValueError("QTN has no overlap with requested interval")

#         big_gaps = func.find_big_gaps(dfqtn, settings["Big_Gaps"]["QTN_big_gaps"], str(ind1), str(ind2))
#         diagnostics_QTN = func.resample_timeseries_estimate_gaps(dfqtn, settings["part_resol"], large_gaps=10)
#         diagnostics_QTN.setdefault("resampled_df", None)

#         return dfqtn.drop_duplicates(), diagnostics_QTN, "QTN", big_gaps

#     except Exception:
#         diagnostics_QTN = {
#             "Init_dt": np.nan,
#             "resampled_df": None,
#             "Frac_miss": None,
#             "Large_gaps": None,
#             "Tot_gaps": None,
#             "resol": None,
#         }
#         return None, diagnostics_QTN, "No QTN", None


# # ============================================================
# # EPHEMERIS
# # ============================================================
# def download_ephemeris_PSP(t0, t1, credentials, varnames, settings=None):
#     try:
#         username = credentials["psp"]["fields"]["username"]
#         password = credentials["psp"]["fields"]["password"]

#         ephemdata = pyspedas.psp.fields(
#             trange=[t0, t1],
#             datatype="ephem_spp_rtn",
#             level="l1",
#             varnames=varnames,
#             time_clip=True,
#             username=username,
#             password=password,
#             no_update=settings["use_local_data"] if isinstance(settings, dict) and "use_local_data" in settings else False,
#         )

#         if not ephemdata or len(ephemdata) == 0:
#             return None

#         col_names = map_col_names_PSP("EPHEMERIS", varnames)
#         dfs = []
#         for i, data in enumerate(ephemdata):
#             cols = col_names[i] if i < len(col_names) else [f"ephem_{i}"]
#             dfs.append(pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=cols))

#         dfephem = pd.concat(dfs, axis=1)
#         dfephem = _ensure_datetime_index(dfephem)

#         if {"sc_pos_r", "sc_pos_t", "sc_pos_n"}.issubset(dfephem.columns):
#             dfephem["Dist_au"] = np.sqrt(np.sum(dfephem[["sc_pos_r", "sc_pos_t", "sc_pos_n"]] ** 2, axis=1)) / au_to_km

#         return dfephem.drop_duplicates()

#     except Exception as e:
#         logging.exception("Ephemeris could not be loaded: %s", e)
#         return None


# def process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1, ind2, settings):
#     try:
#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         dfephem = download_ephemeris_PSP(t0, t1, credentials, varnames_EPHEM, settings)
#         if dfephem is None or len(dfephem) == 0:
#             return None

#         dfephem = _clip_to_requested(dfephem, ind1, ind2, req_start, req_end)
#         return dfephem if (dfephem is not None and len(dfephem) > 0) else None

#     except Exception:
#         return None


# # ============================================================
# # E-field
# # ============================================================
# def download_efield(t0, t1, credentials, varnames, settings):
#     try:
#         if varnames is None:
#             return None

#         fields_vars = pyspedas.psp.fields(
#             trange=[t0, t1],
#             datatype="dfb_wf_dvdc",
#             varnames=varnames,
#             level="l2",
#             time_clip=True,
#             no_update=settings["use_local_data"],
#         )

#         if not fields_vars or len(fields_vars) == 0:
#             return None

#         col_names = map_col_names_PSP("FIELDS-MAG", varnames)
#         cols = col_names[0] if (col_names and len(col_names) > 0) else ["dvx", "dvy"]

#         df_efield = _tplot_to_df(fields_vars[0], cols)
#         df_efield = _ensure_datetime_index(df_efield)

#         return df_efield.drop_duplicates()

#     except Exception as e:
#         logging.exception("E-field download failed: %s", e)
#         return None


# def process_e_field_data(t0, t1, settings, credentials, varnames, ind1, ind2):
#     try:
#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         df_efield = download_efield(t0, t1, credentials, varnames, settings)
#         if df_efield is None or len(df_efield) == 0:
#             return None, None, _diag_default()

#         df_efield = _clip_to_requested(df_efield, ind1, ind2, req_start, req_end)
#         if df_efield is None or len(df_efield) == 0:
#             return None, None, _diag_default()

#         big_gaps_e_field = func.find_big_gaps(df_efield, settings["Big_Gaps"]["E_big_gaps"], str(ind1), str(ind2))
#         diagnostics_e_field = func.resample_timeseries_estimate_gaps(df_efield, 1, large_gaps=10)
#         diagnostics_e_field.setdefault("resampled_df", None)

#         return df_efield.drop_duplicates(), big_gaps_e_field, diagnostics_e_field

#     except Exception as e:
#         logging.exception("E-field processing failed: %s", e)
#         return None, None, _diag_default()


# # ============================================================
# # SC potential
# # ============================================================
# def sc_potential_derived_density(t0, t1, credentials, varnames, settings):
#     try:
#         if varnames is None:
#             return None

#         fields_vars = pyspedas.psp.fields(
#             trange=[t0, t1],
#             datatype="dfb_wf_vdc",
#             level="l2",
#             time_clip=True,
#             no_update=settings["use_local_data"],
#         )
#         if not fields_vars or len(fields_vars) == 0:
#             return None

#         wanted_cols = map_col_names_PSP("FIELDS-MAG", ["dfb_wf_vdc"])
#         wanted_cols = wanted_cols[0] if wanted_cols else [
#             "psp_fld_l2_dfb_wf_V1dc",
#             "psp_fld_l2_dfb_wf_V2dc",
#             "psp_fld_l2_dfb_wf_V3dc",
#             "psp_fld_l2_dfb_wf_V4dc",
#         ]

#         df_try = _tplot_to_df(fields_vars[0], wanted_cols)
#         if isinstance(df_try, pd.DataFrame) and df_try.shape[1] == 4:
#             return df_try.drop_duplicates()

#         dfs = []
#         for i, tv in enumerate(fields_vars[:4]):
#             col = [wanted_cols[i]] if i < len(wanted_cols) else [f"Vdc_{i+1}"]
#             dfs.append(_tplot_to_df(tv, col))

#         if len(dfs) == 0:
#             return None

#         df_density = pd.concat(dfs, axis=1)
#         df_density = _ensure_datetime_index(df_density)

#         return df_density.drop_duplicates()

#     except Exception as e:
#         logging.exception("SC potential download failed: %s", e)
#         return None


# def process_sc_potential_data(t0, t1, settings, credentials, varnames, ind1, ind2):
#     try:
#         req_start = pd.to_datetime(t0)
#         req_end = pd.to_datetime(t1)

#         df_density = sc_potential_derived_density(t0, t1, credentials, varnames, settings)
#         if df_density is None or len(df_density) == 0:
#             return None, None, _diag_default()

#         df_density = _clip_to_requested(df_density, ind1, ind2, req_start, req_end)
#         if df_density is None or len(df_density) == 0:
#             return None, None, _diag_default()

#         big_gaps_density = func.find_big_gaps(df_density, settings["Big_Gaps"]["SC_pot_big_gaps"], str(ind1), str(ind2))
#         diagnostics_density = func.resample_timeseries_estimate_gaps(df_density, 1, large_gaps=10)
#         diagnostics_density.setdefault("resampled_df", None)

#         return df_density, big_gaps_density, diagnostics_density

#     except Exception:
#         traceback.print_exc()
#         return None, None, _diag_default()


# # ============================================================
# # Particle selection / integration
# # ============================================================
# def create_particle_dataframe(
#     PSP_distance_au,
#     end_time,
#     diagnostics_spc,
#     diagnostics_span,
#     dfqtn,
#     dfqtn_flag,
#     big_gaps_span,
#     big_gaps_spc,
#     settings,
# ):
#     def integrate_qtn_data(source_df, dfqtn_in):
#         try:
#             if source_df is None or not isinstance(source_df, pd.DataFrame) or len(source_df) == 0:
#                 return source_df, "No_QTN"

#             if dfqtn_in is None or not isinstance(dfqtn_in, pd.DataFrame) or len(dfqtn_in) == 0:
#                 if "np" in source_df.columns:
#                     source_df["np_sweap"] = source_df["np"].copy()
#                 return source_df, "No_QTN"

#             try:
#                 source_df, dfqtn_sync = func.synchronize_dfs(source_df, dfqtn_in, True)
#             except Exception:
#                 dfqtn_sync = func.newindex(dfqtn_in, source_df.index)

#             if "np_qtn" in dfqtn_sync.columns:
#                 source_df["np"] = dfqtn_sync["np_qtn"].values

#             return source_df, "QTN"

#         except Exception as e:
#             logging.exception("Failed QTN integration: %s", e)
#             if source_df is not None and "np" in source_df.columns:
#                 source_df["np_sweap"] = source_df["np"].copy()
#             return source_df, "No_QTN"

#     mode = settings.get("particle_mode", "9th_perih_cut")

#     if mode == "9th_perih_cut":
#         use_spc = pd.Timestamp(end_time) < pd.Timestamp("2021-07-15")
#         df_selected = diagnostics_spc.get("resampled_df", None) if use_spc else diagnostics_span.get("resampled_df", None)
#         big_gaps = big_gaps_spc if use_spc else big_gaps_span
#         part_flag = "spc" if use_spc else "span"

#     elif settings.get("allow_max_SWEAP_distance", False):
#         use_spc = (PSP_distance_au is not None) and (PSP_distance_au > settings.get("max_SWEAP_distance", 0.25))
#         df_selected = diagnostics_spc.get("resampled_df", None) if use_spc else diagnostics_span.get("resampled_df", None)
#         big_gaps = big_gaps_spc if use_spc else big_gaps_span
#         part_flag = "spc" if use_spc else "span"

#     elif mode == "spc":
#         df_selected = diagnostics_spc.get("resampled_df", None)
#         big_gaps = big_gaps_spc
#         part_flag = "spc"

#     elif mode == "span":
#         df_selected = diagnostics_span.get("resampled_df", None)
#         big_gaps = big_gaps_span
#         part_flag = "span"

#     else:
#         raise ValueError(f"Unsupported particle mode: {mode}")

#     if df_selected is None or not isinstance(df_selected, pd.DataFrame) or len(df_selected) == 0:
#         return None, part_flag, "No_QTN", big_gaps

#     try:
#         df_selected = func.replace_negative_with_nan(df_selected)
#     except Exception:
#         pass

#     df_selected, dfqtn_flag_out = integrate_qtn_data(df_selected, dfqtn)

#     try:
#         out = df_selected.interpolate().dropna()
#     except Exception:
#         out = df_selected

#     return out, part_flag, dfqtn_flag_out, big_gaps


# # ============================================================
# # LoadTimeSeriesPSP (PUBLIC API, SAME OUTPUT ORDER)
# # ============================================================
# def LoadTimeSeriesPSP(
#     start_time,
#     end_time,
#     settings,
#     vars_2_downnload,
#     cdf_lib_path,
#     credentials=None,
#     time_amount=2,
#     time_unit="h",
# ):
#     settings = init_psp_settings(settings)

#     os.chdir(settings["Data_path"])
#     Path("./psp_data").mkdir(exist_ok=True)

#     t0i, t1i = func.ensure_time_format(start_time, end_time)
#     t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
#     t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

#     t0i_e = func.add_time_to_datetime_string(t0i, -2, "m")
#     t1i_e = func.add_time_to_datetime_string(t1i, 2, "m")

#     ind1 = func.string_to_datetime_index(t0i)
#     ind2 = func.string_to_datetime_index(t1i)
#     ind1_e = func.string_to_datetime_index(t0i_e)
#     ind2_e = func.string_to_datetime_index(t1i_e)

#     (
#         varnames_MAG,
#         varnames_QTN,
#         varnames_SPAN,
#         varnames_SPC,
#         varnames_SPAN_alpha,
#         varnames_EPHEM,
#         varnames_E_field,
#         varnames_SC_pot,
#     ) = default_variables_to_download_PSP(vars_2_downnload)

#     # QTN
#     try:
#         dfqtn, diagnostics_QTN, dfqtn_flag, dfqtn_big_gaps = process_qtn_data(
#             t0, t1, credentials, varnames_QTN, ind1_e, ind2_e, settings
#         )
#     except Exception:
#         traceback.print_exc()
#         dfqtn, diagnostics_QTN, dfqtn_flag, dfqtn_big_gaps = None, _diag_default(), "No QTN", None

#     # Ephemeris
#     dfephem = process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1_e, ind2_e, settings)

#     mean_dist = np.nan
#     try:
#         if dfephem is not None and "Dist_au" in dfephem.columns and len(dfephem) > 0:
#             mean_dist = float(np.nanmean(dfephem["Dist_au"].values))
#             mean_dist = round(mean_dist, 2)
#     except Exception:
#         mean_dist = np.nan

#     max_dist = settings.get("max_PSP_dist", None)
#     dist_threshold = True if max_dist is None else (mean_dist < float(max_dist))
#     qtn_threshold = (dfqtn_flag == "QTN") or (settings["must_have_qtn"] is False)

#     if not (dist_threshold and qtn_threshold):
#         if (dist_threshold is False) and (qtn_threshold is False):
#             logging.info(BG_BLUE + "Discarded, No qtn and d=%s" + RESET, mean_dist)
#         elif dist_threshold is False:
#             logging.info(BG_BLUE + "Discarded, d=%s" + RESET, mean_dist)
#         elif qtn_threshold is False:
#             logging.info(BG_BLUE + "Discarded, no qtn dat." + RESET)
#         return None, None, None, None, None, None, None, None, None, None, None

#     logging.info("Passed Dist & QTN thresholds")

#     # SC potential
#     if vars_2_downnload.get("sc_pot", False):
#         df_SC_pot, big_gaps_SC_pot, diagnostics_SC_pot = process_sc_potential_data(
#             t0, t1, settings, credentials, varnames_SC_pot, ind1, ind2
#         )
#     else:
#         df_SC_pot, big_gaps_SC_pot, diagnostics_SC_pot = None, None, _diag_default()

#     # E-field
#     if vars_2_downnload.get("E_field", False):
#         df_e_field, big_gaps_e_field, diagnostics_e_field = process_e_field_data(
#             t0, t1, settings, credentials, varnames_E_field, ind1, ind2
#         )
#     else:
#         df_e_field, big_gaps_e_field, diagnostics_e_field = None, None, _diag_default()

#     # MAG
#     dfmag, big_gaps, diagnostics_MAG = process_mag_field_data(
#         t0, t1, settings, credentials, varnames_MAG, ind1, ind2
#     )

#     # SPAN
#     if settings["particle_mode"] in {"span", "9th_perih_cut"}:
#         dfspan, diagnostics_SPAN, span_flag, big_gaps_span = process_span_data(
#             t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1_e, ind2_e
#         )
#     else:
#         dfspan, diagnostics_SPAN, span_flag, big_gaps_span = None, _diag_default(), None, None

#     # SPC
#     if settings["particle_mode"] in {"spc", "9th_perih_cut"}:
#         dfspc, diagnostics_SPC, spc_flag, big_gaps_spc = process_spc_data(
#             t0, t1, credentials, varnames_SPC, settings, ind1_e, ind2_e
#         )
#     else:
#         dfspc, diagnostics_SPC, spc_flag, big_gaps_spc = None, _diag_default(), None, None

#     try:
#         qtn_resampled = diagnostics_QTN.get("resampled_df", None)
#         if qtn_resampled is not None and not isinstance(qtn_resampled, pd.DataFrame):
#             qtn_resampled = pd.DataFrame(qtn_resampled)

#         dfpar, part_flag, dfqtn_flag2, big_gaps_par = create_particle_dataframe(
#             mean_dist,
#             end_time,
#             diagnostics_SPC,
#             diagnostics_SPAN,
#             qtn_resampled,
#             dfqtn_flag,
#             big_gaps_span,
#             big_gaps_spc,
#             settings,
#         )

#         if dfpar is None or len(dfpar) == 0:
#             raise ValueError("Particle dataframe is empty after selection/integration")

#         diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
#         diagnostics_PAR.setdefault("resampled_df", None)

#         keys_to_keep = ["Frac_miss", "Large_gaps", "Tot_gaps", "resol"]
#         misc = {
#             "SPC": func.filter_dict(diagnostics_SPC, keys_to_keep),
#             "SPAN": func.filter_dict(diagnostics_SPAN, keys_to_keep),
#             "QTN": func.filter_dict(diagnostics_QTN, keys_to_keep),
#             "Par": func.filter_dict(diagnostics_PAR, keys_to_keep),
#             "E": func.filter_dict(diagnostics_e_field, keys_to_keep),
#             "SC_pot": func.filter_dict(diagnostics_SC_pot, keys_to_keep),
#             "Mag": func.filter_dict(diagnostics_MAG, keys_to_keep),
#             "part_flag": part_flag,
#             "qtn_flag": dfqtn_flag2,
#         }

#         if dfqtn_flag2 == "No_QTN" and isinstance(diagnostics_PAR.get("resampled_df", None), pd.DataFrame):
#             if "np_sweap" in diagnostics_PAR["resampled_df"].columns and "np" not in diagnostics_PAR["resampled_df"].columns:
#                 diagnostics_PAR["resampled_df"]["np"] = diagnostics_PAR["resampled_df"].pop("np_sweap")

#         return (
#             diagnostics_QTN.get("resampled_df", None),
#             diagnostics_MAG.get("resampled_df", None),
#             diagnostics_PAR.get("resampled_df", None),
#             diagnostics_e_field.get("resampled_df", None),
#             diagnostics_SC_pot.get("resampled_df", None),
#             dfephem.interpolate() if isinstance(dfephem, pd.DataFrame) else dfephem,
#             big_gaps,
#             dfqtn_big_gaps,
#             big_gaps_par,
#             big_gaps_SC_pot,
#             misc,
#         )

#     except Exception as e:
#         logging.exception("LoadTimeSeriesPSP failed in final assembly: %s", e)
#         return None, None, None, None, None, None, None, None, None, None, None


# # ============================================================
# # SCAM loader (PUBLIC API)
# # ============================================================
# def LoadSCAMFromSPEDAS_PSP(t0, t1, credentials, settings):
#     try:
#         username = credentials["psp"]["fields"]["username"]
#         password = credentials["psp"]["fields"]["password"]

#         if settings.get("in_rtn", True):
#             scam_vars = pyspedas.psp.fields(
#                 trange=[t0, t1],
#                 datatype="merged_scam_wf",
#                 varnames=["psp_fld_l3_merged_scam_wf_RTN"],
#                 level="l3",
#                 time_clip=1,
#                 downloadonly=False,
#                 username=username,
#                 password=password,
#                 no_update=settings["use_local_data"],
#             )
#             cols = ["Br", "Bt", "Bn"]
#         else:
#             scam_vars = pyspedas.psp.fields(
#                 trange=[t0, t1],
#                 datatype="merged_scam_wf",
#                 varnames=["psp_fld_l3_merged_scam_wf_SC"],
#                 level="l3",
#                 time_clip=1,
#                 downloadonly=False,
#                 username=username,
#                 password=password,
#                 no_update=settings["use_local_data"],
#             )
#             cols = ["Bx", "By", "Bz"]

#         if not scam_vars or len(scam_vars) == 0:
#             return None

#         data = get_data(scam_vars[0])
#         if data is None:
#             return None

#         dfscam = pd.DataFrame(index=data.times, data=data.y, columns=cols)
#         dfscam = _ensure_datetime_index(dfscam)

#         return dfscam.drop_duplicates()

#     except Exception:
#         traceback.print_exc()
#         return None
