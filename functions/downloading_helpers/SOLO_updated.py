import os
import sys
import time
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from dateutil import parser

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

# ============================================================
# Local SPEDAS
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pyspedas"))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data
from functions.project_config import merge_user_paths_into_settings

# ============================================================
# Your helper functions
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(REPO_ROOT / "functions"))
import general_functions as func

# ============================================================
# SPDF API (kept, not used in the core SOLO pipeline below)
# ============================================================
from cdasws import CdasWs
cdas = CdasWs()


# # ============================================================
# # Small diagnostics helpers (minimal + clean)
# # ============================================================
# def _median_cadence_seconds(df):
#     try:
#         if df is None or len(df) < 3:
#             return np.nan
#         dt = df.index.to_series().diff().median()
#         return float(dt.total_seconds())
#     except Exception:
#         return np.nan


def _df_info_line(name, df, req_start=None, req_end=None):
    if df is None:
        logging.info(f"{name:<18s}: None")
        return

    if not isinstance(df, pd.DataFrame):
        logging.info(f"{name:<18s}: not a DataFrame ({type(df)})")
        return

    if len(df) == 0:
        logging.info(f"{name:<18s}: EMPTY")
        return

    t0 = df.index[0]
    t1 = df.index[-1]
    dur_h = (t1 - t0).total_seconds() / 3600.0
    cad = func.find_cadence(df)

    msg = f"{name:<18s}: N={len(df):<10d} | {t0} -> {t1} | dur={dur_h:.3f} h | cad~{cad:.3f} s"

    if (req_start is not None) and (req_end is not None):
        req_h = (req_end - req_start).total_seconds() / 3600.0
        msg += f" | req={req_h:.3f} h"

    logging.info(msg)


# ============================================================
# Defaults
# ============================================================
def default_variables_to_download_SOLO(vars_2_downnload):

    if vars_2_downnload["mag"] is None:
        varnames_MAG = ["B_RTN"]
    else:
        varnames_MAG = vars_2_downnload["mag"]

    if vars_2_downnload["rpw"] is None:
        varnames_RPW = ["bia-density-10-seconds"]
    else:
        varnames_RPW = vars_2_downnload["rpw"]

    if vars_2_downnload["swa"] is None:
        varnames_SWA = ["N", "V_RTN", "T"]
    else:
        varnames_SWA = vars_2_downnload["swa"]

    if vars_2_downnload["ephem"] is None:
        varnames_EPHEM = ["position", "velocity"]
    else:
        varnames_EPHEM = vars_2_downnload["ephem"]

    return varnames_MAG, varnames_SWA, varnames_EPHEM, varnames_RPW


def map_col_names_SOLO(instrument, varnames):

    fields_MAG_cols = {
        "rtn-normal": ["Br", "Bt", "Bn"],
        "srf-normal": ["Bx", "By", "Bz"],
        "rtn-burst": ["Br", "Bt", "Bn"],
        "srf-burst": ["Bx", "By", "Bz"],
    }

    swa_cols = {
        "N": ["np"],
        "T": ["T"],
        "V_RTN": ["Vr", "Vt", "Vn"],
        "V_SRF": ["Vx", "Vy", "Vz"],
        "V_SOLO_RTN": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
    }

    rpw_cols = {
        "bia-density-10-seconds": ["ne_qtn"],
        "bia-density": ["ne_qtn"],
    }

    ephem_cols = {
        "position": ["sc_pos_r", "sc_pos_t", "sc_pos_n"],
        "velocity": ["sc_vel_r", "sc_vel_t", "sc_vel_n"],
    }

    if instrument == "SWA":
        return [swa_cols[var] for var in varnames if var in swa_cols]

    if instrument == "RPW":
        return [rpw_cols[var] for var in varnames if var in rpw_cols]

    if instrument == "MAG":
        return [fields_MAG_cols[var] for var in varnames if var in fields_MAG_cols]

    if instrument == "EPHEMERIS":
        return [ephem_cols[var] for var in varnames if var in ephem_cols]

    return []


# ============================================================
# MAG loader (SOAR merged optional + SPEDAS fallback)
# ============================================================
def download_MAG_SOLO(t0, t1, settings, varnames):
    """
    Returns:
        dfmag, mag_flag

    dfmag is always expected to look like:
        index = tz-naive datetime
        columns = ['Br','Bt','Bn'] (RTN) or ['Bx','By','Bz'] (SRF)

    CLEAN FLAGS:
        dfmag.attrs["MAG_source"] = "SOAR_MERGED_SCM" or "SPEDAS_L2"
        dfmag.attrs["SCM_merged_loaded"] = True/False
        dfmag.attrs["SOAR_merged_product"] = product string or None
    """

    def _safe_pyspedas_trange(tt0, tt1):
        return (
            pd.to_datetime(tt0).strftime("%Y-%m-%d/%H:%M:%S"),
            pd.to_datetime(tt1).strftime("%Y-%m-%d/%H:%M:%S"),
        )

    def _read_soar_merged(tt0, tt1):
        try:
            import cdflib
            from sunpy.net import Fido
            import sunpy.net.attrs as a
            import sunpy_soar  # noqa: F401

            t0_dt = pd.to_datetime(tt0)
            t1_dt = pd.to_datetime(tt1)

            fs = int(settings.get("SOLO_merged_fs", 256))
            frame = "rtn" if settings.get("in_rtn", 1) else "srf"
            product = settings.get(
                "SOLO_merged_product",
                f"multi-mag-rpw-scm-merged-{frame}-{fs}",
            )

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
                        # fallback
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

    def _retrieve_spedas(datatype, tt0, tt1):
        MAGdata = pyspedas.solo.mag(
            trange=[tt0, tt1],
            datatype=datatype,
            level="l2",
            time_clip=True,
            no_update=settings["use_local_data"],
        )
        col_names = map_col_names_SOLO("MAG", [datatype])
        arr = get_data(MAGdata[0])
        return pd.DataFrame(index=arr.times, data=arr.y, columns=col_names[0])

    try:
        # 1) SOAR merged first
        if settings.get("SOLO_use_merged_MAG", False):
            merged_df = _read_soar_merged(t0, t1)
            if isinstance(merged_df, pd.DataFrame) and len(merged_df) > 0:
                logging.info("SOAR merged dataset loaded successfully.")
                return merged_df, "Burst"
            logging.info("SOAR merged dataset empty/unreadable -> fallback to SPEDAS.")

        # 2) SPEDAS fallback (kept close to your original)
        dfmag = pd.DataFrame()
        mag_flag = None

        t0p, t1p = _safe_pyspedas_trange(t0, t1)

        for varname in varnames:
            if varname == "B_RTN":
                if settings["MAG_resol"] > 230:
                    datatype = "rtn-normal"
                    mag_flag = "Regular"
                    logging.info("Using normal-resol data!")
                else:
                    datatype = "rtn-burst"
            else:
                datatype = "srf-normal" if settings["MAG_resol"] > 230 else "srf-burst"

            df = _retrieve_spedas(datatype, t0p, t1p)
            dfmag = dfmag.join(df, how="outer")

        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)

        dfmag.attrs["MAG_source"] = "SPEDAS_L2"
        dfmag.attrs["SCM_merged_loaded"] = False
        dfmag.attrs["SOAR_merged_product"] = None

        if len(dfmag) == 0:
            return pd.DataFrame(), "Regular"

        # original burst sufficiency test (kept)
        int_dur = (pd.to_datetime(t1) - pd.to_datetime(t0)).total_seconds() / 3600.0
        deviation = (
            abs((dfmag.index[-1] - pd.to_datetime(t1)) / np.timedelta64(1, "h"))
            + abs((dfmag.index[0] - pd.to_datetime(t0)) / np.timedelta64(1, "h"))
        )

        if deviation >= 0.1 * int_dur:
            logging.info("Too little burst data -> fallback to normal.")
            dfmag = pd.DataFrame()
            for varname in varnames:
                datatype = "rtn-normal" if varname == "B_RTN" else "srf-normal"
                df = _retrieve_spedas(datatype, t0p, t1p)
                dfmag = dfmag.join(df, how="outer")

            dfmag.index = time_string.time_datetime(time=dfmag.index)
            dfmag.index = dfmag.index.tz_localize(None)
            dfmag.attrs["MAG_source"] = "SPEDAS_L2"
            dfmag.attrs["SCM_merged_loaded"] = False
            dfmag.attrs["SOAR_merged_product"] = None
            mag_flag = "Regular"
        else:
            if mag_flag != "Regular":
                mag_flag = "Burst"

        return dfmag, mag_flag

    except Exception as e:
        logging.exception(f"MAG download failed: {e}")
        return None, None


# ============================================================
# SWA (particles)
# ============================================================
def download_SWA_SOLO(t0, t1, settings, varnames):
    try:
        t0p = pd.to_datetime(t0).strftime("%Y-%m-%d/%H:%M:%S")
        t1p = pd.to_datetime(t1).strftime("%Y-%m-%d/%H:%M:%S")

        swadata = pyspedas.solo.swa(
            trange=[t0p, t1p],
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

        # keep your original format behavior
        if "T" in dfswa.columns:
            dfswa["Tp"] = dfswa.pop("T")
            dfswa["Vth"] = 13.84112218 * np.sqrt(dfswa["Tp"])

        dfswa.index = time_string.time_datetime(time=dfswa.index)
        dfswa.index = dfswa.index.tz_localize(None)
        dfswa.index.name = "datetime"

        if "np" in dfswa.columns:
            dfswa["np_qtn"] = dfswa["np"]
            dfswa["ne_qtn"] = dfswa["np"]

        return dfswa

    except Exception as e:
        logging.exception(f"SWA download failed: {e}")
        return None


# ============================================================
# RPW (QTN density)
# ============================================================
def download_RPW_SOLO(t0, t1, settings, varnames):

    # NOTE: network dropouts happen; do minimal retry
    n_tries = int(settings.get("RPW_n_tries", 2))
    sleep_s = float(settings.get("RPW_retry_sleep", 1.0))

    for attempt in range(n_tries + 1):
        try:
            t0p = pd.to_datetime(t0).strftime("%Y-%m-%d/%H:%M:%S")
            t1p = pd.to_datetime(t1).strftime("%Y-%m-%d/%H:%M:%S")

            varname_in = varnames[0] if isinstance(varnames, (list, tuple)) and len(varnames) > 0 else "bia-density-10-seconds"

            if varname_in == "bia-density-10-seconds":
                datatype = "bia-density-10-seconds"
                vnames = ["DENSITY"]
            else:
                datatype = "bia-density"
                vnames = ["DENSITY"]

            col_names = map_col_names_SOLO("RPW", [datatype])

            rpwdata = pyspedas.solo.rpw(
                trange=[t0p, t1p],
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

            dfrpw.index = time_string.time_datetime(time=dfrpw.index)
            dfrpw.index = dfrpw.index.tz_localize(None)
            dfrpw.index.name = "datetime"

            if "ne_qtn" in dfrpw.columns:
                dfrpw["np_qtn"] = dfrpw["ne_qtn"] * 0.96

            return dfrpw

        except Exception as e:
            if attempt < n_tries:
                logging.warning(f"RPW download failed (attempt {attempt+1}/{n_tries+1}): {e}")
                time.sleep(sleep_s)
                continue
            logging.exception(f"RPW download failed (final): {e}")
            return None


# ============================================================
# Distance (kept as-is, you mostly load from pickle anyway)
# ============================================================
def download_ephem_SOLO(t0, t1, cdf_lib_path):
    os.environ["CDF_LIB"] = cdf_lib_path

    time_range = [
        (pd.Timestamp(t0) - pd.Timedelta("3d")).to_pydatetime().replace(tzinfo=pytz.UTC),
        (pd.Timestamp(t1) + pd.Timedelta("3d")).to_pydatetime().replace(tzinfo=pytz.UTC),
    ]

    status, data = cdas.get_data(
        "SOLO_HELIO1DAY_POSITION",
        ["RAD_AU", "SE_LAT", "SE_LON", "HG_LAT", "HG_LON", "HGI_LAT", "HGI_LON"],
        time_range[0],
        time_range[1],
    )

    dfdis = data[["RAD_AU", "SE_LAT", "SE_LON", "HG_LAT", "HG_LON", "HGI_LAT", "HGI_LON"]].to_dataframe()
    dfdis.index.name = "datetime"
    dfdis["Dist_au"] = dfdis["RAD_AU"]
    return dfdis


# ============================================================
# MAIN ENTRY POINT (same signature + same outputs)
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

    import os
    import traceback
    import pandas as pd
    import logging
    from pathlib import Path
    import numpy as np

    settings = merge_user_paths_into_settings(settings)
    data_root = Path(settings["Data_path"]).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SPEDAS_DATA_DIR", str(data_root))
    (data_root / "solar_orbiter_data").mkdir(parents=True, exist_ok=True)

    default_settings = {
        "use_hampel": False,
        "part_resol": 900,
        "MAG_resol": 1,
        "use_local_data": False,

        # ============================================================
        # NEW (minimal): control whether to overwrite np with QTN density
        # ============================================================
        "use_qtn_density": True,
    }

    try:
        settings = {**default_settings, **settings}

        # allow "YYYY-mm-dd HH:MM" OR "...:SS"
        try:
            t0i, t1i = func.ensure_time_format(start_time, end_time)
        except Exception:
            t0i = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
            t1i = pd.to_datetime(end_time).strftime("%Y-%m-%d %H:%M:%S")

        # padded interval for loading
        t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
        t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

        # requested interval
        ind1 = func.string_to_datetime_index(t0i)
        ind2 = func.string_to_datetime_index(t1i)

        req_start = pd.to_datetime(t0i)
        req_end   = pd.to_datetime(t1i)

        varnames_MAG, varnames_SWA, varnames_EPHEM, varnames_RPW = default_variables_to_download_SOLO(vars_2_downnload)

        # ==========================================================
        # RPW (OPTIONAL unless must_have_qtn=True)
        # ==========================================================
        dfrpw = None
        dfqtn_flag = "NO_QTN"
        big_gaps_qtn = None
        diagnostics_RPW = {"Frac_miss": 100, "Large_gaps": 100, "Tot_gaps": 100, "resol": 100, "resampled_df": None}

        try:
            dfrpw = download_RPW_SOLO(t0, t1, settings, varnames_RPW)

            if (dfrpw is not None) and isinstance(dfrpw, pd.DataFrame) and len(dfrpw) > 0:
                dfrpw = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfrpw)
                big_gaps_qtn = func.find_big_gaps(dfrpw, settings["Big_Gaps"]["QTN_big_gaps"], ind1, ind2)
                diagnostics_RPW = func.resample_timeseries_estimate_gaps(dfrpw, settings["part_resol"], large_gaps=10)
                diagnostics_RPW.setdefault("resampled_df", None)
                dfqtn_flag = "QTN"
            else:
                dfrpw = None
                dfqtn_flag = "NO_QTN"

        except Exception as e:
            logging.warning(f"RPW failed (will continue if must_have_qtn=False): {e}")
            dfrpw = None
            dfqtn_flag = "NO_QTN"

        # if must_have_qtn=True, require that RPW was actually loaded
        if (settings.get("must_have_qtn", False) is True) and (dfqtn_flag != "QTN"):
            logging.info("No QTN data and must_have_qtn=True -> rejecting interval")
            return (None, None, None, None, None, None, None, None)

        # ==========================================================
        # SWA (REQUIRED)
        # ==========================================================
        dfpar = None
        big_gaps_par = None
        diagnostics_PAR = {"Frac_miss": 100, "Large_gaps": 100, "Tot_gaps": 100, "resol": 100, "resampled_df": None}
        part_flag = None
        qtn_flag = "No_QTN"

        try:
            dfpar = download_SWA_SOLO(t0, t1, settings, varnames_SWA)

            if not (isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0):
                logging.info("No particle data!")
                return (None, None, None, None, None, big_gaps_qtn, None, None)

            # clip SWA to requested interval
            dfpar = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)

            # ---- DIAGNOSTIC: explain short coverage (this answers your question cleanly)
            if isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0:
                sw0 = dfpar.index.min()
                sw1 = dfpar.index.max()
                if sw0 > req_start:
                    logging.info(f"SWA missing at interval START by {(sw0 - req_start).total_seconds()/3600.0:.3f} h")
                if sw1 < req_end:
                    logging.info(f"SWA missing at interval END   by {(req_end - sw1).total_seconds()/3600.0:.3f} h")

            big_gaps_par = func.find_big_gaps(dfpar, settings["Big_Gaps"]["Par_big_gaps"], ind1, ind2)
            diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
            diagnostics_PAR.setdefault("resampled_df", None)

            part_flag = "SWA"

            # ==========================================================
            # QTN overwrite SWITCH (minimal + clean)
            # ==========================================================
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
                # QTN exists but user requested NOT to use it
                qtn_flag = "QTN_NOT_USED"

        except Exception:
            traceback.print_exc()
            return (None, None, None, None, None, big_gaps_qtn, None, None)

        # ==========================================================
        # MAG (SOAR merged optional)
        # ==========================================================
        dfmag = None
        mag_flag = None
        big_gaps = None
        diagnostics_MAG = {"Frac_miss": 100, "Large_gaps": 100, "Tot_gaps": 100, "resol": 100, "resampled_df": None}

        try:
            dfmag, mag_flag = download_MAG_SOLO(t0, t1, settings, varnames_MAG)

            if not (isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0):
                logging.info("No MAG data!")
                return (None, None, diagnostics_PAR.get("resampled_df", None), None, None, big_gaps_qtn, big_gaps_par, None)

            mag_source = dfmag.attrs.get("MAG_source", "UNKNOWN")
            scm_merged_loaded = bool(dfmag.attrs.get("SCM_merged_loaded", False))
            soar_product = dfmag.attrs.get("SOAR_merged_product", None)
            logging.info(f"MAG_source={mag_source} | SCM_merged_loaded={scm_merged_loaded} | SOAR_product={soar_product}")

            dfmag = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
            big_gaps = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], ind1, ind2)
            diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)
            diagnostics_MAG.setdefault("resampled_df", None)

        except Exception:
            traceback.print_exc()
            return (None, None, diagnostics_PAR.get("resampled_df", None), None, None, big_gaps_qtn, big_gaps_par, None)

        # ==========================================================
        # DISTANCE (unchanged)
        # ==========================================================
        dfdis = None
        try:
            dfdis = pd.read_pickle(settings["SOLO_dist_path"])
            dfdis = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfdis)
            if diagnostics_PAR.get("resampled_df") is not None:
                dfdis = func.newindex(dfdis, diagnostics_PAR["resampled_df"].index)
                diagnostics_PAR["resampled_df"]["Dist_au"] = dfdis.values
        except Exception:
            pass

        # ==========================================================
        # MISC (preserve + add clear QTN usage info)
        # ==========================================================
        keys_to_keep = ["Frac_miss", "Large_gaps", "Tot_gaps", "resol"]

        mag_source = dfmag.attrs.get("MAG_source", "UNKNOWN") if dfmag is not None else "NONE"
        scm_merged_loaded = bool(dfmag.attrs.get("SCM_merged_loaded", False)) if dfmag is not None else False
        soar_product = dfmag.attrs.get("SOAR_merged_product", None) if dfmag is not None else None

        misc = {
            "Par": func.filter_dict(diagnostics_PAR, keys_to_keep),
            "Mag": func.filter_dict(diagnostics_MAG, keys_to_keep),
            "QTN": func.filter_dict(diagnostics_RPW, keys_to_keep),

            "part_flag": part_flag,
            "qtn_flag": qtn_flag,

            "MAG_source": mag_source,
            "SCM_merged_loaded": scm_merged_loaded,
            "SOAR_merged_product": soar_product,

            # NEW (minimal, harmless)
            "use_qtn_density": bool(settings.get("use_qtn_density", True)),
            "qtn_loaded": (dfrpw is not None),
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




# import numpy as np
# import pandas as pd
# import sys
# import scipy.io
# import os
# import sys
# from pathlib import Path
# import pickle
# from gc import collect
# from glob import glob
# from datetime import datetime
# import traceback
# from time import sleep
# import matplotlib.dates as mdates
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# from dateutil import parser

# import logging
# import traceback


# BG_WHITE = '\033[47m'
# RESET    = '\033[0m'  # Reset the color
# BG_RED = '\033[41m'
# BG_GREEN = '\033[42m'
# BG_YELLOW = '\033[43m'
# BG_BLUE = '\033[44m'
# BG_MAGENTA = '\033[45m'
# BG_CYAN = '\033[46m'

# import logging
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
#     datefmt='%Y-%m-%d %H:%M:%S',  # Format for the timestamp
# )

# # Setup basic configuration for logging
# logging.basicConfig(level=logging.INFO)




# # Make sure to use the local spedas
# REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'pyspedas'))
# import pyspedas
# from pyspedas.utilities import time_string
# from pytplot import get_data


# """ Import manual functions """
# REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(REPO_ROOT / 'functions'))
# import general_functions as func


# # Some constants
# from scipy import constants
# au_to_km        = 1.496e8  # Conversion factor
# rsun            = 696340   # Sun radius in units of  [km]
# mu0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
# mu_0            =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
# m_p             =  constants.m_p   # Proton mass [kg]
# k               = constants.k                  # Boltzman's constant     [j/K]
# au_to_rsun      = 215.032
# T_to_Gauss      = 1e4



# import pytz
# # SPDF API
# from cdasws import CdasWs
# cdas = CdasWs()




# def default_variables_to_download_SOLO(vars_2_downnload):
    
#     if vars_2_downnload['mag'] is None:
#         varnames_MAG          = ['B_RTN']
#     else:
#         varnames_MAG          = vars_2_downnload['mag']
        
#     if vars_2_downnload['rpw'] is None:
#         varnames_RPW          = ['bia-density-10-seconds']
#     else:
#         varnames_RPW          = vars_2_downnload['rpw']
        
#     if vars_2_downnload['swa'] is None:
#         varnames_SWA          = ['N', 'V_RTN', 'T']
#     else:
#         varnames_SWA          = vars_2_downnload['swa']        
    
#     if vars_2_downnload['ephem'] is None:
#         varnames_EPHEM         = ['position','velocity']  
#     else:
#         varnames_EPHEM          = vars_2_downnload['ephem']
#     return varnames_MAG,  varnames_SWA ,varnames_EPHEM, varnames_RPW


# def map_col_names_SOLO(instrument, varnames):
    
#     # Mapping between variable names and column names for FIELDS
#     fields_MAG_cols = {
#         'rtn-normal'                   : ['Br', 'Bt', 'Bn'],
#         'srf-normal'                   : ['Bx', 'By', 'Bz'],
#         'rtn-burst'                    : ['Br', 'Bt', 'Bn'],
#         'srf-burst'                    : ['Bx', 'By', 'Bz'],

#     }

#     # Mapping between variable names and column names for SPAN
#     swa_cols = {
#         'N'               : ['np'] ,
#         'T'               : ['T'] ,
#         'V_RTN'           : ['Vr','Vt','Vn'],
#         'V_SRF'           : ['Vx','Vy','Vz'],
#         'V_SOLO_RTN'      : ['sc_vel_r','sc_vel_t','sc_vel_n']

#     }
    
    
#     rpw_cols = {
#         'bia-density-10-seconds'      : ['ne_qtn'] ,
#         'bia-density'                 : ['ne_qtn'] ,
#     }

#     # Mapping between variable names and column names for EPHEMERIS
#     ephem_cols = {
#         'position'            : ['sc_pos_r','sc_pos_t','sc_pos_n'],
#         'velocity'            : ['sc_vel_r','sc_vel_t','sc_vel_n'],
#     }    
  
    
#     if instrument == 'SWA':
#         return [swa_cols[var] for var in varnames if var in swa_cols]
    
#     if instrument == 'RPW':
#         return [rpw_cols[var] for var in varnames if var in rpw_cols]
    
#     elif instrument == 'MAG':
#         return [fields_MAG_cols[var] for var in varnames if var in fields_MAG_cols]
#     elif instrument =='EPHEMERIS':
#          return [ephem_cols[var] for var in varnames if var in ephem_cols]
#     else:
#         return []
    
    







# from dateutil import parser
# import pandas as pd
# import numpy as np
# import logging
# import traceback


# def download_MAG_SOLO(t0, t1, settings, varnames):
#     """
#     MINIMAL REVISION:
#     - Adds CLEAN flags indicating whether SOAR merged SCM dataset was used
#       without changing return type or dataframe format.

#     Output:
#       (dfmag, mag_flag)

#     Flags are stored in:
#       dfmag.attrs["MAG_source"]          -> "SOAR_MERGED_SCM" or "SPEDAS_L2"
#       dfmag.attrs["SCM_merged_loaded"]   -> True/False
#       dfmag.attrs["SOAR_merged_product"] -> the product name if used
#     """

#     def _safe_time_strings_for_pyspedas(tt0, tt1):
#         t0p = pd.to_datetime(tt0).strftime("%Y-%m-%d/%H:%M:%S")
#         t1p = pd.to_datetime(tt1).strftime("%Y-%m-%d/%H:%M:%S")
#         return t0p, t1p

#     def _read_soar_merged(tt0, tt1):
#         """
#         Reads SOAR merged MAG+SCM dataset and returns DataFrame identical to MAG:
#           index: tz-naive datetime
#           columns: ['Br','Bt','Bn'] for RTN or ['Bx','By','Bz'] for SRF
#         """
#         try:
#             import cdflib
#             from sunpy.net import Fido
#             import sunpy.net.attrs as a
#             import sunpy_soar  # noqa: F401 (registers SOAR)

#             t0_dt = pd.to_datetime(tt0)
#             t1_dt = pd.to_datetime(tt1)

#             fs = int(settings.get("SOLO_merged_fs", 256))
#             frame = "rtn" if settings.get("in_rtn", 1) else "srf"
#             product = settings.get("SOLO_merged_product", f"multi-mag-rpw-scm-merged-{frame}-{fs}")

#             qr = Fido.search(a.Time(t0_dt, t1_dt), a.soar.Product(product))

#             if len(qr) == 0:
#                 return pd.DataFrame()

#             files = Fido.fetch(qr)

#             if (files is None) or (len(files) == 0):
#                 return pd.DataFrame()

#             dfs = []
#             for fp in files:
#                 try:
#                     cdf = cdflib.CDF(fp)
#                     info = cdf.cdf_info()

#                     if hasattr(info, "zVariables"):
#                         zvars = set(info.zVariables)
#                     elif isinstance(info, dict) and ("zVariables" in info):
#                         zvars = set(info["zVariables"])
#                     else:
#                         zvars = set()

#                     if "Epoch" not in zvars:
#                         continue

#                     if frame == "rtn":
#                         want_var = "B_RTN"
#                         cols = ["Br", "Bt", "Bn"]
#                     else:
#                         want_var = "B_SRF"
#                         cols = ["Bx", "By", "Bz"]

#                     # fallback if variable naming differs
#                     if want_var not in zvars:
#                         if "B_RTN" in zvars:
#                             want_var = "B_RTN"
#                             cols = ["Br", "Bt", "Bn"]
#                         elif "B_SRF" in zvars:
#                             want_var = "B_SRF"
#                             cols = ["Bx", "By", "Bz"]
#                         else:
#                             continue

#                     epoch_tt2000 = cdf.varget("Epoch")
#                     B = cdf.varget(want_var)

#                     if epoch_tt2000 is None or B is None:
#                         continue

#                     epoch_dt = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch_tt2000))
#                     df = pd.DataFrame(B, index=epoch_dt, columns=cols)

#                     df.index = df.index.tz_localize(None)
#                     df = df.loc[(df.index >= t0_dt) & (df.index <= t1_dt)]

#                     if len(df) > 0:
#                         dfs.append(df)

#                 except Exception:
#                     traceback.print_exc()
#                     continue

#             if len(dfs) == 0:
#                 return pd.DataFrame()

#             out = pd.concat(dfs).sort_index()
#             out = out[~out.index.duplicated(keep="first")]
#             out.attrs["MAG_source"] = "SOAR_MERGED_SCM"
#             out.attrs["SCM_merged_loaded"] = True
#             out.attrs["SOAR_merged_product"] = product
#             return out

#         except Exception:
#             traceback.print_exc()
#             return pd.DataFrame()

#     def retrieve_mag_data_spedas(datatype, tt0, tt1):
#         MAGdata = pyspedas.solo.mag(
#             trange=[tt0, tt1],
#             datatype=datatype,
#             level="l2",
#             time_clip=True,
#             no_update=settings["use_local_data"],
#         )
#         col_names = map_col_names_SOLO("MAG", [datatype])
#         df = pd.DataFrame(
#             index=get_data(MAGdata[0]).times,
#             data=get_data(MAGdata[0]).y,
#             columns=col_names[0],
#         )
#         return df

#     try:
#         # ==========================================================
#         # 1) Try SOAR merged dataset FIRST (if requested)
#         # ==========================================================
#         if settings.get("SOLO_use_merged_MAG", False):
#             merged_df = _read_soar_merged(t0, t1)
#             if isinstance(merged_df, pd.DataFrame) and (len(merged_df) > 0):
#                 logging.info("SOAR merged dataset loaded successfully.")
#                 return merged_df, "Burst"

#             logging.info("SOAR merged dataset empty/unreadable -> fallback to SPEDAS.")

#         # ==========================================================
#         # 2) Fallback: SPEDAS MAG logic (original behavior)
#         # ==========================================================
#         dfmag = pd.DataFrame()
#         mag_flag = None

#         t0p, t1p = _safe_time_strings_for_pyspedas(t0, t1)

#         for varname in varnames:
#             if varname == "B_RTN":
#                 if settings["MAG_resol"] > 230:
#                     datatype = "rtn-normal"
#                     mag_flag = "Regular"
#                     print("Using normal-resol data!")
#                 else:
#                     datatype = "rtn-burst"
#             else:
#                 datatype = "srf-normal" if settings["MAG_resol"] > 230 else "srf-burst"

#             df = retrieve_mag_data_spedas(datatype, t0p, t1p)
#             dfmag = dfmag.join(df, how="outer")

#         dfmag.index = time_string.time_datetime(time=dfmag.index)
#         dfmag.index = dfmag.index.tz_localize(None)

#         # set clean attrs
#         dfmag.attrs["MAG_source"] = "SPEDAS_L2"
#         dfmag.attrs["SCM_merged_loaded"] = False
#         dfmag.attrs["SOAR_merged_product"] = None

#         if len(dfmag) == 0:
#             return pd.DataFrame(), "Regular"

#         int_dur = (parser.parse(t1) - parser.parse(t0)).total_seconds() / 3600.0
#         deviation = (
#             abs((dfmag.index[-1] - parser.parse(t1)) / np.timedelta64(1, "h"))
#             + abs((dfmag.index[0] - parser.parse(t0)) / np.timedelta64(1, "h"))
#         )

#         if deviation >= 0.1 * int_dur:
#             print("Too little burst data!")
#             dfmag = pd.DataFrame()
#             for varname in varnames:
#                 datatype = "rtn-normal" if varname == "B_RTN" else "srf-normal"
#                 df = retrieve_mag_data_spedas(datatype, t0p, t1p)
#                 dfmag = dfmag.join(df, how="outer")

#             dfmag.index = time_string.time_datetime(time=dfmag.index)
#             dfmag.index = dfmag.index.tz_localize(None)

#             dfmag.attrs["MAG_source"] = "SPEDAS_L2"
#             dfmag.attrs["SCM_merged_loaded"] = False
#             dfmag.attrs["SOAR_merged_product"] = None
#             mag_flag = "Regular"
#         else:
#             if mag_flag != "Regular":
#                 print("Ok, We have enough burst mag data")
#                 mag_flag = "Burst"

#         return dfmag, mag_flag

#     except Exception as e:
#         logging.exception("MAG download failed: %s", e)
#         return None, None



# def download_ephem_SOLO(t0, t1, cdf_lib_path):
#     # Set environment
#     os.environ["CDF_LIB"] = cdf_lib_path

#     time = [
#         (pd.Timestamp(t0) - pd.Timedelta('3d')).to_pydatetime().replace(tzinfo=pytz.UTC),
#         (pd.Timestamp(t1) + pd.Timedelta('3d')).to_pydatetime().replace(tzinfo=pytz.UTC)
#     ]
#     status, data = cdas.get_data(
#         'SOLO_HELIO1DAY_POSITION', 
#         ['RAD_AU', 'SE_LAT', 'SE_LON', 'HG_LAT', 'HG_LON', 'HGI_LAT', 'HGI_LON'], 
#         time[0], time[1]
#     )

#     # Convert the xarray.Dataset subset to a DataFrame
#     dfdis = data[['RAD_AU', 'SE_LAT', 'SE_LON', 'HG_LAT', 'HG_LON', 'HGI_LAT', 'HGI_LON']].to_dataframe()
    
#     # Optionally, if the dataset has a coordinate 'Epoch' that should be used as the index:
#     dfdis.index.name = 'datetime'

#     # Add the additional column
#     dfdis['Dist_au'] = dfdis['RAD_AU']
    
#     return dfdis


# def download_SWA_SOLO(t0, t1, settings, varnames):
#     """
#     MINIMAL REVISION:
#     - Always returns a DataFrame or None (never (None, None)).
#     - Keeps output format identical if successful.
#     """
#     try:
#         # pyspedas expects YYYY-MM-DD/HH:MM:SS
#         t0p = pd.to_datetime(t0).strftime("%Y-%m-%d/%H:%M:%S")
#         t1p = pd.to_datetime(t1).strftime("%Y-%m-%d/%H:%M:%S")

#         swadata = pyspedas.solo.swa(
#             trange=[t0p, t1p],
#             varnames=varnames,
#             datatype="pas-grnd-mom",
#             no_update=settings["use_local_data"],
#         )

#         if (swadata is None) or (len(swadata) == 0):
#             return None

#         col_names = map_col_names_SOLO("SWA", varnames)

#         dfs = []
#         for i, data in enumerate(swadata):
#             arr = get_data(data)
#             if arr is None:
#                 continue
#             dfs.append(
#                 pd.DataFrame(
#                     index=arr.times,
#                     data=arr.y,
#                     columns=col_names[i],
#                 )
#             )

#         if len(dfs) == 0:
#             return None

#         dfswa = dfs[0].join(dfs[1:]) if len(dfs) > 1 else dfs[0]

#         # Rename Proton temperature [eV]
#         if "T" in dfswa.columns:
#             dfswa["Tp"] = dfswa.pop("T")
#             dfswa["Vth"] = 13.84112218 * np.sqrt(dfswa["Tp"])

#         # Fix datetime index
#         dfswa.index = time_string.time_datetime(time=dfswa.index)
#         dfswa.index = dfswa.index.tz_localize(None)
#         dfswa.index.name = "datetime"

#         # Keep same QTN placeholders as before
#         if "np" in dfswa.columns:
#             dfswa["np_qtn"] = dfswa["np"]
#             dfswa["ne_qtn"] = dfswa["np"]

#         return dfswa

#     except Exception as e:
#         logging.exception("SWA download failed: %s", e)
#         return None

    
    
    
# def download_RPW_SOLO(t0, t1, settings, varnames):
#     """
#     MINIMAL REVISION:
#     - Removes the WRONG call to pyspedas.solo.mag() that caused bogus SPDF paths.
#     - Always returns a DataFrame or None (never (None, None)).
#     - Keeps output format identical if successful.
#     """
#     try:
#         # pyspedas expects YYYY-MM-DD/HH:MM:SS
#         t0p = pd.to_datetime(t0).strftime("%Y-%m-%d/%H:%M:%S")
#         t1p = pd.to_datetime(t1).strftime("%Y-%m-%d/%H:%M:%S")

#         # Only one supported RPW density var at a time in your code
#         varname_in = varnames[0] if isinstance(varnames, (list, tuple)) and len(varnames) > 0 else "bia-density-10-seconds"

#         if varname_in == "bia-density-10-seconds":
#             datatype = "bia-density-10-seconds"
#             vnames = ["DENSITY"]
#         else:
#             datatype = "bia-density"
#             vnames = ["DENSITY"]

#         col_names = map_col_names_SOLO("RPW", [datatype])

#         rpwdata = pyspedas.solo.rpw(
#             trange=[t0p, t1p],
#             level="l3",
#             varnames=vnames,
#             datatype=datatype,
#             no_update=settings["use_local_data"],
#         )

#         if (rpwdata is None) or (len(rpwdata) == 0):
#             return None

#         dfs = []
#         for i, data in enumerate(rpwdata):
#             arr = get_data(data)
#             if arr is None:
#                 continue
#             dfs.append(
#                 pd.DataFrame(
#                     index=arr.times,
#                     data=arr.y,
#                     columns=col_names[i],
#                 )
#             )

#         if len(dfs) == 0:
#             return None

#         dfrpw = dfs[0].join(dfs[1:]) if len(dfs) > 1 else dfs[0]

#         # Fix datetime index
#         dfrpw.index = time_string.time_datetime(time=dfrpw.index)
#         dfrpw.index = dfrpw.index.tz_localize(None)
#         dfrpw.index.name = "datetime"

#         # alpha correction
#         if "ne_qtn" in dfrpw.columns:
#             dfrpw["np_qtn"] = dfrpw["ne_qtn"] * 0.96

#         return dfrpw

#     except Exception as e:
#         logging.exception("RPW download failed: %s", e)
#         return None


# def LoadTimeSeriesSOLO(start_time,
#                       end_time,
#                       settings,
#                       vars_2_downnload,
#                       cdf_lib_path,
#                       credentials     = None,
#                       time_amount     = 12,
#                       time_unit       = 'H'
#                      ):

#     import os
#     import traceback
#     import pandas as pd
#     import logging
#     from pathlib import Path

#     os.chdir(settings['Data_path'])

#     if not os.path.exists("./solar_orbiter_data"):
#         working_dir = os.getcwd()
#         os.makedirs(str(Path(working_dir).joinpath("solar_orbiter_data")), exist_ok=True)

#     default_settings = {
#         'use_hampel'    : False,
#         'part_resol'    : 900,
#         'MAG_resol'     : 1,
#         'use_local_data': False
#     }

#     try:
#         settings = {**default_settings, **settings}

#         # accept "YYYY-mm-dd HH:MM" or "...:SS"
#         try:
#             t0i, t1i = func.ensure_time_format(start_time, end_time)
#         except Exception:
#             t0i = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
#             t1i = pd.to_datetime(end_time).strftime("%Y-%m-%d %H:%M:%S")

#         # enforce padding
#         t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
#         t1 = func.add_time_to_datetime_string(t1i,  time_amount, time_unit)

#         ind1  = func.string_to_datetime_index(t0i)
#         ind2  = func.string_to_datetime_index(t1i)

#         varnames_MAG, varnames_SWA, varnames_EPHEM, varnames_RPW = default_variables_to_download_SOLO(vars_2_downnload)

#         # ==========================================================
#         # RPW (OPTIONAL unless must_have_qtn=True)
#         # ==========================================================
#         dfrpw = None
#         dfqtn_flag = 'NO_QTN'
#         big_gaps_qtn = None
#         diagnostics_RPW = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}

#         try:
#             dfrpw = download_RPW_SOLO(t0, t1, settings, varnames_RPW)

#             if (dfrpw is not None) and isinstance(dfrpw, pd.DataFrame) and len(dfrpw) > 0:
#                 dfrpw = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfrpw)
#                 big_gaps_qtn = func.find_big_gaps(dfrpw, settings['Big_Gaps']['QTN_big_gaps'], ind1, ind2)
#                 diagnostics_RPW = func.resample_timeseries_estimate_gaps(dfrpw, settings['part_resol'], large_gaps=10)
#                 dfqtn_flag = 'QTN'
#             else:
#                 dfrpw = None
#                 dfqtn_flag = 'NO_QTN'

#         except Exception as e:
#             logging.warning(f"RPW failed (will continue if must_have_qtn=False): {e}")
#             dfrpw = None
#             dfqtn_flag = 'NO_QTN'

#         if (settings.get("must_have_qtn", False) is True) and (dfqtn_flag != "QTN"):
#             print("No QTN data and must_have_qtn=True -> rejecting interval")
#             return (None, None, None, None, None, None, None, None)

#         # ==========================================================
#         # SWA (REQUIRED)
#         # ==========================================================
#         dfpar = None
#         big_gaps_par = None
#         diagnostics_PAR = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
#         part_flag = None
#         qtn_flag = "No_QTN"

#         try:
#             dfpar = download_SWA_SOLO(t0, t1, settings, varnames_SWA)

#             if not (isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0):
#                 print("No particle data!")
#                 return (None, None, None, None, None, big_gaps_qtn, None, None)

#             dfpar = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)

#             big_gaps_par = func.find_big_gaps(dfpar, settings['Big_Gaps']['Par_big_gaps'], ind1, ind2)
#             diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'], large_gaps=10)

#             part_flag = 'SWA'

#             # if QTN exists, overwrite np
#             if dfrpw is not None:
#                 try:
#                     dfrpw = dfrpw[~dfrpw.index.duplicated(keep='first')]
#                     dfpar = dfpar[~dfpar.index.duplicated(keep='first')]
#                     dfrpw = func.newindex(dfrpw, dfpar.index)
#                     if "np_qtn" in dfrpw.columns:
#                         dfpar["np"] = dfrpw["np_qtn"]
#                         qtn_flag = "QTN"
#                 except Exception:
#                     traceback.print_exc()
#                     qtn_flag = "No_QTN"

#         except Exception:
#             traceback.print_exc()
#             return (None, None, None, None, None, big_gaps_qtn, None, None)

#         # ==========================================================
#         # MAG (SOAR merged optional)
#         # ==========================================================
#         dfmag = None
#         mag_flag = None
#         big_gaps = None
#         diagnostics_MAG = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}

#         try:
#             dfmag, mag_flag = download_MAG_SOLO(t0, t1, settings, varnames_MAG)

#             if not (isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0):
#                 print("No MAG data!")
#                 return (None, None, diagnostics_PAR.get("resampled_df", None), None, None, big_gaps_qtn, big_gaps_par, None)

#             # CLEAN info about MAG source
#             mag_source = dfmag.attrs.get("MAG_source", "UNKNOWN")
#             scm_merged_loaded = bool(dfmag.attrs.get("SCM_merged_loaded", False))
#             soar_product = dfmag.attrs.get("SOAR_merged_product", None)

#             logging.info(f"MAG_source = {mag_source} | SCM_merged_loaded = {scm_merged_loaded} | SOAR_product = {soar_product}")

#             dfmag = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
#             big_gaps = func.find_big_gaps(dfmag, settings['Big_Gaps']['Mag_big_gaps'], ind1, ind2)
#             diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings['MAG_resol'], large_gaps=10)

#         except Exception:
#             traceback.print_exc()
#             return (None, None, diagnostics_PAR.get("resampled_df", None), None, None, big_gaps_qtn, big_gaps_par, None)

#         # ==========================================================
#         # DISTANCE (unchanged)
#         # ==========================================================
#         dfdis = None
#         try:
#             dfdis = pd.read_pickle(settings['SOLO_dist_path'])
#             dfdis = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfdis)
#             dfdis = func.newindex(dfdis, diagnostics_PAR["resampled_df"].index)
#             diagnostics_PAR["resampled_df"]['Dist_au'] = dfdis.values
#         except Exception:
#             pass

#         # ==========================================================
#         # MISC (add clean merged flag)
#         # ==========================================================
#         keys_to_keep = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']

#         mag_source = dfmag.attrs.get("MAG_source", "UNKNOWN") if dfmag is not None else "NONE"
#         scm_merged_loaded = bool(dfmag.attrs.get("SCM_merged_loaded", False)) if dfmag is not None else False
#         soar_product = dfmag.attrs.get("SOAR_merged_product", None) if dfmag is not None else None

#         misc = {
#             'Par'                : func.filter_dict(diagnostics_PAR,  keys_to_keep),
#             'Mag'                : func.filter_dict(diagnostics_MAG,  keys_to_keep),
#             'QTN'                : func.filter_dict(diagnostics_RPW,  keys_to_keep),
#             'part_flag'          : part_flag,
#             'qtn_flag'           : qtn_flag,
#             'MAG_source'         : mag_source,
#             'SCM_merged_loaded'  : scm_merged_loaded,
#             'SOAR_merged_product': soar_product,
#         }

#         return (
#             diagnostics_MAG["resampled_df"],
#             mag_flag,
#             diagnostics_PAR["resampled_df"],
#             dfdis,
#             big_gaps,
#             big_gaps_qtn,
#             big_gaps_par,
#             misc
#         )

#     except Exception:
#         traceback.print_exc()
#         return (None, None, None, None, None, None, None, None)
