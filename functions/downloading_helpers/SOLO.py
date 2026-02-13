from __future__ import annotations

import os
import sys
import importlib.util
import time
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
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pytz

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Local SPEDAS
# ============================================================
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

# ============================================================
# Your helper functions
# ============================================================
import general_functions as func
import TurbPy as turb


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


# ============================================================
# SPDF / CDAS API (for distance / ephemeris)
# ============================================================
from cdasws import CdasWs
cdas = CdasWs()


# ============================================================
# Column mappings (single authority)
# ============================================================
_FIELDS_MAG_COLS = {
    "rtn-normal": ["Br", "Bt", "Bn"],
    "srf-normal": ["Bx", "By", "Bz"],
    "rtn-burst":  ["Br", "Bt", "Bn"],
    "srf-burst":  ["Bx", "By", "Bz"],
}

_SWA_COLS = {
    "N": ["np"],
    "T": ["T"],
    "V_RTN": ["Vr", "Vt", "Vn"],
    "V_SRF": ["Vx", "Vy", "Vz"],
    "V_SOLO_RTN": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
}

_RPW_COLS = {
    "bia-density-10-seconds": ["ne_qtn"],
    "bia-density": ["ne_qtn"],
}

_EPHEM_COLS = {
    "position": ["sc_pos_r", "sc_pos_t", "sc_pos_n"],
    "velocity": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
}


# ============================================================
# One place for ALL defaults (no signature changes)
# ============================================================
def init_solo_settings(user_settings: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(user_settings, dict):
        raise TypeError("settings must be a dict")

    user_settings = normalize_settings(user_settings)

    if "Data_path" not in user_settings:
        raise KeyError("settings must include 'Data_path'")

    data_path = Path(user_settings["Data_path"])
    default_dist_cache = data_path / "solar_orbiter_data" / "solo_distance_helio1day_cache.pkl"

    defaults = {
        "use_hampel": False,
        "hampel_params": {"w": 200, "std": 3},

        "part_resol": 900,
        "MAG_resol": 1,
        "use_local_data": False,

        "must_have_qtn": False,
        "use_qtn_density": True,
        "RPW_n_tries": 2,
        "RPW_retry_sleep": 1.0,

        "SOLO_use_merged_MAG": False,
        "SOLO_merged_fs": 256,
        "in_rtn": 1,
        "SOLO_merged_product": None,

        "SOLO_dist_cache_path": str(default_dist_cache),
        "SOLO_dist_overwrite_cache": False,

        "Dist_au_min": None,
        "Dist_au_max": None,
        "SOLO_dist_path": None,

        "Big_Gaps": {
            "Mag_big_gaps": 500,
            "Par_big_gaps": 500,
            "QTN_big_gaps": 10,
            "E_big_gaps": 10,
            "SC_pot_big_gaps": 10,
        },

        # Unified noise option (preferred). Gate is applied only for merged SCM.
        "Mag_SCM": {
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
        # Legacy SOLO-only key (supported by resolve_mag_noise_settings)
        "Mag_SCM_SOLO": {
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

    out = {**defaults, **user_settings}

    if not isinstance(out.get("Big_Gaps", None), dict):
        out["Big_Gaps"] = defaults["Big_Gaps"]

    return out


# ============================================================
# Defaults for variable lists
# ============================================================
def default_variables_to_download_SOLO(vars_2_downnload: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    if vars_2_downnload.get("mag", None) is None:
        varnames_MAG = ["B_RTN"]
    else:
        varnames_MAG = vars_2_downnload["mag"]

    if vars_2_downnload.get("rpw", None) is None:
        varnames_RPW = ["bia-density-10-seconds"]
    else:
        varnames_RPW = vars_2_downnload["rpw"]

    if vars_2_downnload.get("swa", None) is None:
        varnames_SWA = ["N", "V_RTN", "T"]
    else:
        varnames_SWA = vars_2_downnload["swa"]

    if vars_2_downnload.get("ephem", None) is None:
        varnames_EPHEM = ["position", "velocity"]
    else:
        varnames_EPHEM = vars_2_downnload["ephem"]

    return varnames_MAG, varnames_SWA, varnames_EPHEM, varnames_RPW


def map_col_names_SOLO(instrument: str, varnames: List[str]) -> List[List[str]]:
    if instrument == "SWA":
        return [_SWA_COLS[var] for var in varnames if var in _SWA_COLS]
    if instrument == "RPW":
        return [_RPW_COLS[var] for var in varnames if var in _RPW_COLS]
    if instrument == "MAG":
        return [_FIELDS_MAG_COLS[var] for var in varnames if var in _FIELDS_MAG_COLS]
    if instrument == "EPHEMERIS":
        return [_EPHEM_COLS[var] for var in varnames if var in _EPHEM_COLS]
    return []


# ============================================================
# Time string helpers
# ============================================================
def _pyspedas_trange(t0: str, t1: str) -> List[str]:
    return [
        pd.to_datetime(t0).strftime("%Y-%m-%d/%H:%M:%S"),
        pd.to_datetime(t1).strftime("%Y-%m-%d/%H:%M:%S"),
    ]


# ============================================================
# Distance download + cache (SpaceData-safe)
# ============================================================
def _space_data_to_dataframe(data_obj: Any, variables: List[str]) -> pd.DataFrame:
    if data_obj is None:
        raise RuntimeError("CDAS returned None data object")

    # Determine time axis
    time_key = None
    for k in ("Epoch", "epoch", "Time", "time"):
        try:
            if k in data_obj:
                time_key = k
                break
        except Exception:
            continue

    if time_key is None:
        raise RuntimeError("Could not find a time key in CDAS SpaceData (expected 'Epoch')")

    t_raw = data_obj[time_key]

    # SpacePy Ticktock has .UTC
    if hasattr(t_raw, "UTC"):
        t = pd.to_datetime(t_raw.UTC).tz_localize(None)
    else:
        t = pd.to_datetime(np.asarray(t_raw)).tz_localize(None)

    cols = {}
    for v in variables:
        if v not in data_obj:
            cols[v] = np.full(len(t), np.nan)
            continue
        arr = np.asarray(data_obj[v])
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        cols[v] = arr

    df = pd.DataFrame(cols, index=t)
    df.index.name = "datetime"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def download_distance_SOLO_cdas(req_start: pd.Timestamp, req_end: pd.Timestamp, cdf_lib_path: str) -> pd.DataFrame:
    os.environ["CDF_LIB"] = str(cdf_lib_path)

    t0 = (req_start - pd.Timedelta("3d")).to_pydatetime().replace(tzinfo=pytz.UTC)
    t1 = (req_end + pd.Timedelta("3d")).to_pydatetime().replace(tzinfo=pytz.UTC)

    status, data = cdas.get_data(
        "SOLO_HELIO1DAY_POSITION",
        ["RAD_AU", "SE_LAT", "SE_LON", "HG_LAT", "HG_LON", "HGI_LAT", "HGI_LON"],
        t0,
        t1,
    )

    df = _space_data_to_dataframe(
        data_obj=data,
        variables=["RAD_AU", "SE_LAT", "SE_LON", "HG_LAT", "HG_LON", "HGI_LAT", "HGI_LON"],
    )

    df["Dist_au"] = df["RAD_AU"]
    return df


def load_or_download_distance_SOLO(
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    cdf_lib_path: str,
    cache_path: Optional[str],
    overwrite: bool = False,
) -> pd.DataFrame:
    if cache_path is None:
        return download_distance_SOLO_cdas(req_start, req_end, cdf_lib_path)

    cache_file = Path(str(cache_path))
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    df_cache = None
    if cache_file.exists() and (not overwrite):
        try:
            df_cache = pd.read_pickle(str(cache_file))
            if isinstance(df_cache, pd.DataFrame) and len(df_cache) > 0:
                df_cache = df_cache.sort_index()
                df_cache = df_cache[~df_cache.index.duplicated(keep="first")]
        except Exception:
            df_cache = None

    df_new = download_distance_SOLO_cdas(req_start, req_end, cdf_lib_path)
    df_new = df_new.sort_index()
    df_new = df_new[~df_new.index.duplicated(keep="first")]

    if df_cache is None or len(df_cache) == 0:
        df_out = df_new
    else:
        df_out = pd.concat([df_cache, df_new]).sort_index()
        df_out = df_out[~df_out.index.duplicated(keep="first")]

    try:
        df_out.to_pickle(str(cache_file))
    except Exception:
        pass

    return df_out


def _distance_window_reject(
    dfdis: Optional[pd.DataFrame],
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    dist_min: Optional[float],
    dist_max: Optional[float],
) -> bool:
    if (dist_min is None) and (dist_max is None):
        return False
    if dfdis is None or (not isinstance(dfdis, pd.DataFrame)) or len(dfdis) == 0:
        return True
    if "Dist_au" not in dfdis.columns:
        return True

    dfi = dfdis.loc[(dfdis.index >= req_start) & (dfdis.index <= req_end)]
    if len(dfi) == 0:
        return True

    x = pd.to_numeric(dfi["Dist_au"], errors="coerce").dropna()
    if len(x) == 0:
        return True

    xmin = float(np.nanmin(x.values))
    xmax = float(np.nanmax(x.values))

    if (dist_min is not None) and (xmin < float(dist_min)):
        return True
    if (dist_max is not None) and (xmax > float(dist_max)):
        return True
    return False


def _safe_distance_placeholder(req_start: pd.Timestamp, req_end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(req_start, req_end, freq="1H")
    df = pd.DataFrame(index=idx, data={"Dist_au": np.nan})
    df.index.name = "datetime"
    return df


# ============================================================
# MAG loader (SOAR merged optional + SPEDAS fallback)
# ============================================================
def download_MAG_SOLO(
    t0: str,
    t1: str,
    settings: Dict[str, Any],
    varnames: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:

    def _read_soar_merged(tt0: str, tt1: str) -> pd.DataFrame:
        try:
            import cdflib
            from sunpy.net import Fido
            import sunpy.net.attrs as a
            import sunpy_soar  # noqa: F401

            t0_dt = pd.to_datetime(tt0)
            t1_dt = pd.to_datetime(tt1)

            fs = int(settings.get("SOLO_merged_fs", 256))
            frame = "rtn" if settings.get("in_rtn", 1) else "srf"

            product = settings.get("SOLO_merged_product", None)
            if product is None:
                product = f"multi-mag-rpw-scm-merged-{frame}-{fs}"

            qr = Fido.search(a.Time(t0_dt, t1_dt), a.soar.Product(product))
            if len(qr) == 0:
                return pd.DataFrame()

            files = Fido.fetch(qr)
            if files is None or len(files) == 0:
                return pd.DataFrame()

            dfs = []
            for fp in files:
                try:
                    cdf = cdflib.CDF(str(fp))
                    info = cdf.cdf_info()
                    zvars = set(info.zVariables) if hasattr(info, "zVariables") else set(info.get("zVariables", []))

                    if "Epoch" not in zvars:
                        continue

                    if frame == "rtn":
                        want_var = "B_RTN"
                        cols = ["Br", "Bt", "Bn"]
                    else:
                        want_var = "B_SRF"
                        cols = ["Bx", "By", "Bz"]

                    if want_var not in zvars:
                        if "B_RTN" in zvars:
                            want_var = "B_RTN"
                            cols = ["Br", "Bt", "Bn"]
                        elif "B_SRF" in zvars:
                            want_var = "B_SRF"
                            cols = ["Bx", "By", "Bz"]
                        else:
                            continue

                    epoch_tt2000 = cdf.varget("Epoch")
                    B = cdf.varget(want_var)
                    if epoch_tt2000 is None or B is None:
                        continue

                    epoch_dt = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch_tt2000)).tz_localize(None)
                    df = pd.DataFrame(np.asarray(B), index=epoch_dt, columns=cols)
                    df = df.loc[(df.index >= t0_dt) & (df.index <= t1_dt)]
                    if len(df) > 0:
                        dfs.append(df)

                except Exception:
                    traceback.print_exc()

            if len(dfs) == 0:
                return pd.DataFrame()

            out = pd.concat(dfs).sort_index()
            out = out[~out.index.duplicated(keep="first")]

            out.attrs["MAG_source"] = "SOAR_MERGED_SCM"
            out.attrs["SCM_merged_loaded"] = True
            out.attrs["SOAR_merged_product"] = product
            return out

        except Exception:
            traceback.print_exc()
            return pd.DataFrame()

    def _retrieve_spedas(datatype: str, tt0: str, tt1: str) -> pd.DataFrame:
        magdata = pyspedas.solo.mag(
            trange=[tt0, tt1],
            datatype=datatype,
            level="l2",
            time_clip=True,
            no_update=settings["use_local_data"],
        )

        if magdata is None or len(magdata) == 0:
            return pd.DataFrame()

        col_names = map_col_names_SOLO("MAG", [datatype])
        arr = get_data(magdata[0])
        if arr is None:
            return pd.DataFrame()

        return pd.DataFrame(index=arr.times, data=arr.y, columns=col_names[0])

    try:
        if settings.get("SOLO_use_merged_MAG", False):
            merged_df = _read_soar_merged(t0, t1)
            merged_df = sanitize_timeseries_df(merged_df)
            if isinstance(merged_df, pd.DataFrame) and len(merged_df) > 0:
                logger.info("SOAR merged dataset loaded successfully.")
                # Merged MAG noise-removal (unified with PSP SCAM)
                noise_flag, noise_cfg = resolve_mag_noise_settings(settings)
                if noise_flag:
                    try:
                        dt = func.find_cadence(merged_df)
                        merged_df = apply_optional_wheel_noise_removal(
                            resampled_df=merged_df,
                            cadence_seconds=float(dt) if dt is not None else None,
                            remove_wheel_noise_func=turb.remove_wheel_noise,
                            noise_cfg=noise_cfg,
                            logger=logging.getLogger(__name__),
                        )
                    except Exception as e:
                        logging.warning(f"Merged MAG noise-removal skipped: {e}")

                return merged_df, "Burst"
            logger.info("SOAR merged dataset empty/unreadable -> fallback to SPEDAS.")

        dfmag = pd.DataFrame()
        mag_flag = None

        tt0, tt1 = _pyspedas_trange(t0, t1)

        for v in varnames:
            if v == "B_RTN":
                if settings["MAG_resol"] > 230:
                    datatype = "rtn-normal"
                    mag_flag = "Regular"
                    logger.info("Using normal-resol MAG (RTN).")
                else:
                    datatype = "rtn-burst"
            else:
                datatype = "srf-normal" if settings["MAG_resol"] > 230 else "srf-burst"

            df = _retrieve_spedas(datatype, tt0, tt1)
            dfmag = dfmag.join(df, how="outer")

        dfmag = sanitize_timeseries_df(dfmag)
        if dfmag is None:
            return pd.DataFrame(), "Regular"

        dfmag.attrs["MAG_source"] = "SPEDAS_L2"
        dfmag.attrs["SCM_merged_loaded"] = False
        dfmag.attrs["SOAR_merged_product"] = None

        int_dur = (pd.to_datetime(t1) - pd.to_datetime(t0)).total_seconds() / 3600.0
        deviation = (
            abs((dfmag.index[-1] - pd.to_datetime(t1)) / np.timedelta64(1, "h"))
            + abs((dfmag.index[0] - pd.to_datetime(t0)) / np.timedelta64(1, "h"))
        )

        if deviation >= 0.1 * int_dur:
            logger.info("Too little burst data -> fallback to normal.")
            dfmag = pd.DataFrame()

            for v in varnames:
                datatype = "rtn-normal" if v == "B_RTN" else "srf-normal"
                df = _retrieve_spedas(datatype, tt0, tt1)
                dfmag = dfmag.join(df, how="outer")

            dfmag = sanitize_timeseries_df(dfmag)
            if dfmag is None:
                return pd.DataFrame(), "Regular"

            dfmag.attrs["MAG_source"] = "SPEDAS_L2"
            dfmag.attrs["SCM_merged_loaded"] = False
            dfmag.attrs["SOAR_merged_product"] = None
            mag_flag = "Regular"
        else:
            if mag_flag != "Regular":
                mag_flag = "Burst"

        return dfmag, mag_flag

    except Exception as e:
        logger.exception(f"MAG download failed: {e}")
        return None, None


# ============================================================
# SWA loader
# ============================================================
def download_SWA_SOLO(
    t0: str,
    t1: str,
    settings: Dict[str, Any],
    varnames: List[str],
) -> Optional[pd.DataFrame]:
    try:
        tt0, tt1 = _pyspedas_trange(t0, t1)

        swadata = pyspedas.solo.swa(
            trange=[tt0, tt1],
            varnames=varnames,
            datatype="pas-grnd-mom",
            no_update=settings["use_local_data"],
        )
        if swadata is None or len(swadata) == 0:
            return None

        col_names = map_col_names_SOLO("SWA", varnames)
        dfs = []

        for i, data in enumerate(swadata):
            arr = get_data(data)
            if arr is None:
                continue
            dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=col_names[i]))

        if len(dfs) == 0:
            return None

        dfswa = dfs[0].join(dfs[1:]) if len(dfs) > 1 else dfs[0]
        dfswa = sanitize_timeseries_df(dfswa)
        if dfswa is None:
            return None

        if "T" in dfswa.columns:
            dfswa["Tp"] = dfswa.pop("T")
            dfswa["Vth"] = 13.84112218 * np.sqrt(dfswa["Tp"])

        if "np" in dfswa.columns:
            dfswa["np_qtn"] = dfswa["np"]
            dfswa["ne_qtn"] = dfswa["np"]

        return dfswa

    except Exception as e:
        logger.exception(f"SWA download failed: {e}")
        return None


# ============================================================
# RPW loader (QTN density)
# ============================================================
def download_RPW_SOLO(
    t0: str,
    t1: str,
    settings: Dict[str, Any],
    varnames: List[str],
) -> Optional[pd.DataFrame]:
    n_tries = int(settings.get("RPW_n_tries", 2))
    sleep_s = float(settings.get("RPW_retry_sleep", 1.0))

    for attempt in range(n_tries + 1):
        try:
            tt0, tt1 = _pyspedas_trange(t0, t1)

            varname_in = varnames[0] if isinstance(varnames, (list, tuple)) and len(varnames) > 0 else "bia-density-10-seconds"

            if varname_in == "bia-density-10-seconds":
                datatype = "bia-density-10-seconds"
                vnames = ["DENSITY"]
            else:
                datatype = "bia-density"
                vnames = ["DENSITY"]

            col_names = map_col_names_SOLO("RPW", [datatype])

            rpwdata = pyspedas.solo.rpw(
                trange=[tt0, tt1],
                level="l3",
                varnames=vnames,
                datatype=datatype,
                no_update=settings["use_local_data"],
            )
            if rpwdata is None or len(rpwdata) == 0:
                return None

            dfs = []
            for i, data in enumerate(rpwdata):
                arr = get_data(data)
                if arr is None:
                    continue
                dfs.append(pd.DataFrame(index=arr.times, data=arr.y, columns=col_names[i]))

            if len(dfs) == 0:
                return None

            dfrpw = dfs[0].join(dfs[1:]) if len(dfs) > 1 else dfs[0]
            dfrpw = sanitize_timeseries_df(dfrpw)
            if dfrpw is None:
                return None

            if "ne_qtn" in dfrpw.columns:
                dfrpw["np_qtn"] = dfrpw["ne_qtn"] * 0.96

            return dfrpw

        except Exception as e:
            if attempt < n_tries:
                logger.warning(f"RPW download failed (attempt {attempt+1}/{n_tries+1}): {e}")
                time.sleep(sleep_s)
                continue
            logger.exception(f"RPW download failed (final): {e}")
            return None


# ============================================================
# MAIN ENTRY POINT (signature + outputs unchanged)
# ============================================================
def LoadTimeSeriesSOLO(
    start_time,
    end_time,
    settings,
    vars_2_downnload,
    cdf_lib_path,
    credentials=None,
    time_amount=12,
    time_unit="H",
):
    def _reject(msg: Optional[str] = None):
        if msg:
            logger.info(msg)
        return (None, None, None, None, None, None, None, None)

    try:
        settings = init_solo_settings(settings)

        os.chdir(settings["Data_path"])
        solo_dir = _REPO_ROOT / "solar_orbiter_data"
        solo_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0i, t1i = func.ensure_time_format(start_time, end_time)
        except Exception:
            t0i = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
            t1i = pd.to_datetime(end_time).strftime("%Y-%m-%d %H:%M:%S")

        t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
        t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

        ind1 = func.string_to_datetime_index(t0i)
        ind2 = func.string_to_datetime_index(t1i)

        req_start = pd.to_datetime(t0i)
        req_end = pd.to_datetime(t1i)

        varnames_MAG, varnames_SWA, varnames_EPHEM, varnames_RPW = default_variables_to_download_SOLO(vars_2_downnload)

        # ==========================================================
        # DISTANCE FIRST
        # ==========================================================
        dfdis = None
        dist_min = settings.get("Dist_au_min", None)
        dist_max = settings.get("Dist_au_max", None)

        try:
            dfdis = load_or_download_distance_SOLO(
                req_start=req_start,
                req_end=req_end,
                cdf_lib_path=cdf_lib_path,
                cache_path=settings.get("SOLO_dist_cache_path", None),
                overwrite=bool(settings.get("SOLO_dist_overwrite_cache", False)),
            )
            dfdis = ensure_datetime_index(dfdis)
            if dfdis is not None:
                dfdis = dfdis.loc[(dfdis.index >= req_start) & (dfdis.index <= req_end)]

        except Exception as e:
            logger.warning(f"Distance handling failed (CDAS): {e}")
            dfdis = None

        if dfdis is None or (not isinstance(dfdis, pd.DataFrame)) or len(dfdis) == 0:
            try:
                local_path = settings.get("SOLO_dist_path", None)
                if local_path is not None and Path(str(local_path)).exists():
                    df_local = pd.read_pickle(str(local_path))
                    df_local = ensure_datetime_index(df_local)
                    if isinstance(df_local, pd.DataFrame) and len(df_local) > 0:
                        if "Dist_au" not in df_local.columns and "RAD_AU" in df_local.columns:
                            df_local["Dist_au"] = df_local["RAD_AU"]
                        dfdis = df_local.loc[(df_local.index >= req_start) & (df_local.index <= req_end)]
            except Exception:
                dfdis = None

        if dfdis is None or (not isinstance(dfdis, pd.DataFrame)) or len(dfdis) == 0:
            dfdis = _safe_distance_placeholder(req_start, req_end)

        if _distance_window_reject(dfdis, req_start, req_end, dist_min, dist_max):
            return _reject(f"Rejecting interval by distance window: Dist_au_min={dist_min}, Dist_au_max={dist_max}")

        # ==========================================================
        # RPW (OPTIONAL unless must_have_qtn=True)
        # ==========================================================
        dfrpw = None
        dfqtn_flag = "NO_QTN"
        big_gaps_qtn = None
        diagnostics_RPW = diag_default()

        try:
            dfrpw = download_RPW_SOLO(t0, t1, settings, varnames_RPW)
            if isinstance(dfrpw, pd.DataFrame) and len(dfrpw) > 0:
                dfrpw = clip_to_requested(dfrpw, ind1, ind2, req_start, req_end, func_module=func)

            if isinstance(dfrpw, pd.DataFrame) and len(dfrpw) > 0:
                big_gaps_qtn = func.find_big_gaps(dfrpw, settings["Big_Gaps"]["QTN_big_gaps"], ind1, ind2)
                diagnostics_RPW = func.resample_timeseries_estimate_gaps(dfrpw, settings["part_resol"], large_gaps=10)
                diagnostics_RPW.setdefault("resampled_df", None)
                dfqtn_flag = "QTN"
            else:
                dfrpw = None
                dfqtn_flag = "NO_QTN"

        except Exception as e:
            logger.warning(f"RPW failed (will continue if must_have_qtn=False): {e}")
            dfrpw = None
            dfqtn_flag = "NO_QTN"

        if (settings.get("must_have_qtn", False) is True) and (dfqtn_flag != "QTN"):
            return _reject("No QTN data and must_have_qtn=True -> rejecting interval")

        # ==========================================================
        # SWA (REQUIRED)
        # ==========================================================
        dfpar = None
        big_gaps_par = None
        diagnostics_PAR = diag_default()
        part_flag = None
        qtn_flag = "No_QTN"

        try:
            dfpar = download_SWA_SOLO(t0, t1, settings, varnames_SWA)
            if not (isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0):
                logger.info("No particle data!")
                return (None, None, None, None, None, big_gaps_qtn, None, None)

            dfpar = clip_to_requested(dfpar, ind1, ind2, req_start, req_end, func_module=func)
            if not (isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0):
                logger.info("SWA exists but no overlap with requested interval.")
                return (None, None, None, None, None, big_gaps_qtn, None, None)

            big_gaps_par = func.find_big_gaps(dfpar, settings["Big_Gaps"]["Par_big_gaps"], ind1, ind2)
            diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
            diagnostics_PAR.setdefault("resampled_df", None)

            part_flag = "SWA"

            use_qtn_density = bool(settings.get("use_qtn_density", True))
            if (dfrpw is not None) and use_qtn_density:
                try:
                    dfrpw = dfrpw[~dfrpw.index.duplicated(keep="first")]
                    dfpar = dfpar[~dfpar.index.duplicated(keep="first")]
                    dfrpw = func.newindex(dfrpw, dfpar.index)
                    if "np_qtn" in dfrpw.columns:
                        dfpar["np"] = dfrpw["np_qtn"]
                        qtn_flag = "QTN"
                except Exception:
                    traceback.print_exc()
                    qtn_flag = "No_QTN"
            elif (dfrpw is not None) and (not use_qtn_density):
                qtn_flag = "QTN_NOT_USED"

        except Exception:
            traceback.print_exc()
            return (None, None, None, None, None, big_gaps_qtn, None, None)

        # ==========================================================
        # MAG (REQUIRED)
        # ==========================================================
        dfmag = None
        mag_flag = None
        big_gaps = None
        diagnostics_MAG = diag_default()

        try:
            dfmag, mag_flag = download_MAG_SOLO(t0, t1, settings, varnames_MAG)
            if not (isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0):
                logger.info("No MAG data!")
                return (None, None, diagnostics_PAR.get("resampled_df", None), dfdis, None, big_gaps_qtn, big_gaps_par, None)

            mag_source = dfmag.attrs.get("MAG_source", "UNKNOWN")
            scm_merged_loaded = bool(dfmag.attrs.get("SCM_merged_loaded", False))
            soar_product = dfmag.attrs.get("SOAR_merged_product", None)
            logger.info(f"MAG_source={mag_source} | SCM_merged_loaded={scm_merged_loaded} | SOAR_product={soar_product}")

            dfmag = clip_to_requested(dfmag, ind1, ind2, req_start, req_end, func_module=func)
            if not (isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0):
                logger.info("MAG exists but no overlap with requested interval.")
                return (None, None, diagnostics_PAR.get("resampled_df", None), dfdis, None, big_gaps_qtn, big_gaps_par, None)

            big_gaps = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], ind1, ind2)
            diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)
            diagnostics_MAG.setdefault("resampled_df", None)

            # ======================================================
            # OPTIONAL: wheel-noise removal for merged SCM (SOLO)
            # Applied ONLY at resampled_df stage (matches PSP)
            # ======================================================
            try:
                scm_loaded = bool(dfmag.attrs.get("SCM_merged_loaded", False))
                noise_flag, noise_cfg = resolve_mag_noise_settings(settings)
                if scm_loaded and noise_flag and isinstance(diagnostics_MAG.get("resampled_df", None), pd.DataFrame):
                    dt = func.find_cadence(diagnostics_MAG["resampled_df"])
                    logger.info("SOLO MAG: wheel-noise removal enabled (merged SCM, resampled_df stage)")
                    diagnostics_MAG["resampled_df"] = apply_optional_wheel_noise_removal(
                        resampled_df=diagnostics_MAG["resampled_df"],
                        cadence_seconds=dt,
                        remove_wheel_noise_func=turb.remove_wheel_noise,
                        noise_cfg=noise_cfg,
                        logger=logger,
                    )
            except Exception:
                traceback.print_exc()

        except Exception:
            traceback.print_exc()
            return (None, None, diagnostics_PAR.get("resampled_df", None), dfdis, None, big_gaps_qtn, big_gaps_par, None)

        # ==========================================================
        # Attach Dist_au to PAR resampled_df (ALWAYS define column)
        # ==========================================================
        try:
            par_resampled = diagnostics_PAR.get("resampled_df", None)
            if isinstance(par_resampled, pd.DataFrame) and len(par_resampled) > 0:
                if isinstance(dfdis, pd.DataFrame) and len(dfdis) > 0 and ("Dist_au" in dfdis.columns):
                    dtmp = dfdis[["Dist_au"]].copy()
                    dtmp = dtmp.sort_index()
                    dtmp = dtmp[~dtmp.index.duplicated(keep="first")]
                    dtmp = func.newindex(dtmp, par_resampled.index)
                    par_resampled["Dist_au"] = dtmp["Dist_au"].to_numpy()
                else:
                    par_resampled["Dist_au"] = np.nan
        except Exception:
            try:
                if isinstance(diagnostics_PAR.get("resampled_df", None), pd.DataFrame):
                    diagnostics_PAR["resampled_df"]["Dist_au"] = np.nan
            except Exception:
                pass

        misc = {
            "Par": keep_diag_keys(diagnostics_PAR),
            "Mag": keep_diag_keys(diagnostics_MAG),
            "QTN": keep_diag_keys(diagnostics_RPW),

            "part_flag": part_flag,
            "qtn_flag": qtn_flag,

            "MAG_source": dfmag.attrs.get("MAG_source", "UNKNOWN") if dfmag is not None else "NONE",
            "SCM_merged_loaded": bool(dfmag.attrs.get("SCM_merged_loaded", False)) if dfmag is not None else False,
            "SOAR_merged_product": dfmag.attrs.get("SOAR_merged_product", None) if dfmag is not None else None,

            "use_qtn_density": bool(settings.get("use_qtn_density", True)),
            "qtn_loaded": (dfrpw is not None),

            "Dist_au_min": settings.get("Dist_au_min", None),
            "Dist_au_max": settings.get("Dist_au_max", None),
            "SOLO_dist_cache_path": settings.get("SOLO_dist_cache_path", None),
        }

        return (
            diagnostics_MAG["resampled_df"],
            mag_flag,
            diagnostics_PAR["resampled_df"],
            dfdis,
            big_gaps,
            big_gaps_qtn,
            big_gaps_par,
            misc,
        )

    except Exception:
        traceback.print_exc()
        return (None, None, None, None, None, None, None, None)
