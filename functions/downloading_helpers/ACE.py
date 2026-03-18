# functions/downloading_helpers/ACE.py
from __future__ import annotations

import logging
import os
import sys
import importlib.util
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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent with WIND: K per eV (1 eV = 11604.518... K)
_EV_TO_K = 11604.518121550082

# Consistent with WIND: Vth[km/s] = 0.128487 * sqrt(T[K])
_VTH_COEFF = 0.12848657328083132

# SWEPAM-L2 availability on CDAWeb ends here (as in your previous ACE file)
_ACE_SWEPAM_L2_END = pd.Timestamp("2024-07-09 23:59:59")

_AU_KM = 1.495978707e8


def _ephem_units(settings: Dict[str, Any]) -> str:
    u = str(settings.get("ephem_units", "km")).strip().lower()
    if u in ("au", "astronomicalunit", "astronomical_unit", "astronomical-units"):
        return "au"
    return "km"


def _format_horizons_step(seconds: float) -> str:
    sec = float(seconds)
    if not np.isfinite(sec) or sec <= 0:
        return "10m"
    sec = max(sec, 60.0)
    if sec < 3600.0:
        m = int(max(1, round(sec / 60.0)))
        return f"{m}m"
    h = int(max(1, round(sec / 3600.0)))
    return f"{h}h"


from functools import lru_cache


@lru_cache(maxsize=128)
def _horizons_base_df(target: str, start: str, stop: str, step: str) -> pd.DataFrame:
    """Cached Horizons call via sc_pos.horizons_sun_lonlat.get_lonlat_xyz_timeseries."""
    from sc_pos.horizons_sun_lonlat import get_lonlat_xyz_timeseries  # type: ignore

    tr = get_lonlat_xyz_timeseries(target=str(target), start=str(start), stop=str(stop), step=str(step), carrington=False)
    df = tr.df.copy()
    df = df.loc[~df.index.isna()].sort_index()
    df = df.loc[~df.index.duplicated(keep="first")]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None), name=df.index.name or "time_utc")
    return df


def _ephem_on_index(target: str, index: pd.Index) -> pd.DataFrame:
    """Interpolate Horizons ephemeris to the provided DatetimeIndex."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    idx = idx.sort_values()

    if len(idx) == 0:
        return pd.DataFrame(
            index=idx,
            columns=[
                "Dist_au",
                "Dist_km",
                "lon",
                "lat",
                "x_au",
                "y_au",
                "z_au",
                "x_km",
                "y_km",
                "z_km",
            ],
        )

    t_ns = idx.view("i8")
    dt_s = float(np.nanmedian(np.diff(t_ns))) / 1e9 if len(t_ns) >= 2 else 600.0
    step = _format_horizons_step(dt_s)

    pad = pd.Timedelta("2h") if step.endswith("h") else pd.Timedelta("20m")
    start = (idx[0] - pad).strftime("%Y-%m-%dT%H:%M:%S")
    stop = (idx[-1] + pad).strftime("%Y-%m-%dT%H:%M:%S")

    base = _horizons_base_df(target=str(target), start=start, stop=stop, step=step)

    need = ("hgs_r_au", "hgs_lon_deg", "hgs_lat_deg", "hee_x_au", "hee_y_au", "hee_z_au")
    missing = [k for k in need if k not in base.columns]
    if missing:
        raise RuntimeError(f"Horizons base dataframe is missing required columns: {missing}")

    eph = pd.DataFrame(index=base.index)
    eph["Dist_au"] = pd.to_numeric(base["hgs_r_au"], errors="coerce").astype(float)
    eph["lon"] = pd.to_numeric(base["hgs_lon_deg"], errors="coerce").astype(float)
    eph["lat"] = pd.to_numeric(base["hgs_lat_deg"], errors="coerce").astype(float)

    eph["x_au"] = pd.to_numeric(base["hee_x_au"], errors="coerce").astype(float)
    eph["y_au"] = pd.to_numeric(base["hee_y_au"], errors="coerce").astype(float)
    eph["z_au"] = pd.to_numeric(base["hee_z_au"], errors="coerce").astype(float)

    eph["Dist_km"] = eph["Dist_au"] * _AU_KM
    eph["x_km"] = eph["x_au"] * _AU_KM
    eph["y_km"] = eph["y_au"] * _AU_KM
    eph["z_km"] = eph["z_au"] * _AU_KM

    uidx = eph.index.union(idx)
    tmp = eph.reindex(uidx).sort_index()
    tmp = tmp.interpolate(method="time", limit_direction="both")
    out = tmp.loc[idx]
    out.index = idx
    return out


def _attach_ephem_to_par(dfpar: pd.DataFrame, target: str, settings: Dict[str, Any]):
    """Attach ephemeris columns to dfpar and return (dfpar_out, dfdis)."""
    dfdis = _ephem_on_index(target=str(target), index=dfpar.index)
    units = _ephem_units(settings)
    out = dfpar.copy()

    if units == "au":
        for c in ("Dist_au", "x_au", "y_au", "z_au"):
            out[c] = dfdis[c].to_numpy(dtype=float, copy=False)
    else:
        for c in ("Dist_km", "x_km", "y_km", "z_km"):
            out[c] = dfdis[c].to_numpy(dtype=float, copy=False)

    return out, dfdis



def LoadTimeSeriesACE(
    start_time,
    end_time,
    settings: Dict[str, Any],
    vars_2_downnload: Dict[str, Any],
    cdf_lib_path: Optional[str] = None,
    credentials: Optional[Any] = None,
    time_amount: float = 1.0,
    time_unit: str = "h",
):
    """
    Contract (MUST remain):
      returns (dfmag, dfpar, dfdis, big_gaps, misc)

    Thermals (consistent with WIND):
      Tp stored in eV
      Tp_K = _EV_TO_K * Tp
      Vth (km/s) = _VTH_COEFF * sqrt(Tp_K)
    """



    spedas_dir = settings['Data_path']
    spedas_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SPEDAS_DATA_DIR"] = str(spedas_dir)

    import general_functions as func  # use repo utilities

    verbose = bool(settings.get("verbose", True))

    t0_req = pd.Timestamp(start_time)
    t1_req = pd.Timestamp(end_time)

    # Expand window (legacy behavior)
    try:
        t0 = pd.Timestamp(func.add_time_to_datetime_string(str(t0_req), -float(time_amount), unit=time_unit))
        t1 = pd.Timestamp(func.add_time_to_datetime_string(str(t1_req), +float(time_amount), unit=time_unit))
    except Exception:
        t0, t1 = t0_req, t1_req

    mag_resol_ms = int(settings.get("MAG_resol", 16000))
    part_resol_ms = int(settings.get("part_resol", 64000))
    mag_req_s = mag_resol_ms / 1000.0
    par_req_s = part_resol_ms / 1000.0

    frame = str(settings.get("ace_frame", "GSE")).upper()

    # Ensure SPEDAS dir is inside your Data_path if provided
    data_path = settings.get("Data_path", None)
    if data_path:
        os.environ["SPEDAS_DATA_DIR"] = str(Path(data_path).expanduser().resolve())

    # Local pyspedas import (repo-local)
    import pyspedas  # type: ignore
    from pytplot import del_data, get_data, tplot_names  # type: ignore

    if verbose:
        print("=== ACE loader summary ===")
        print(f"Requested interval : {t0_req} -> {t1_req}")
        print(f"Expanded interval  : {t0} -> {t1}")
        print(f"Target cadences    : MAG={mag_req_s:.3f}s, PAR={par_req_s:.3f}s")
        print(f"Velocity frame key : ace_frame={frame} (only used if we must manufacture V-components)")

    # --------------------
    # MAG (ACE/MFI) via pyspedas
    # --------------------
    del_data("*")
    mag_dtype = str(vars_2_downnload.get("mag", {}).get("datatype", "h0")).lower()

    mag_source = f"pyspedas:ace.mfi:{mag_dtype}"
    try:
        pyspedas.projects.ace.mfi(trange=[str(t0), str(t1)], datatype=mag_dtype, time_clip=True)
    except Exception:
        del_data("*")
        mag_dtype = "h0"
        mag_source = "pyspedas:ace.mfi:h0"
        pyspedas.projects.ace.mfi(trange=[str(t0), str(t1)], datatype="h0", time_clip=True)

    names = list(tplot_names())
    dfmag = None

    bx_name = "BX_GSE" if "BX_GSE" in names else None
    by_name = "BY_GSE" if "BY_GSE" in names else None
    bz_name = "BZ_GSE" if "BZ_GSE" in names else None

    if bx_name and by_name and bz_name:
        tx, bx = get_data(bx_name)
        ty, by = get_data(by_name)
        tz, bz = get_data(bz_name)

        dfx = pd.DataFrame({"Bx": np.asarray(bx).reshape(-1)}, index=pd.to_datetime(tx, unit="s", errors="coerce"))
        dfy = pd.DataFrame({"By": np.asarray(by).reshape(-1)}, index=pd.to_datetime(ty, unit="s", errors="coerce"))
        dfz = pd.DataFrame({"Bz": np.asarray(bz).reshape(-1)}, index=pd.to_datetime(tz, unit="s", errors="coerce"))

        dfx.index = dfx.index.tz_localize(None)
        dfy.index = dfy.index.tz_localize(None)
        dfz.index = dfz.index.tz_localize(None)

        dfmag = pd.merge_asof(dfx.sort_index(), dfy.sort_index(), left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("2s"))
        dfmag = pd.merge_asof(dfmag.sort_index(), dfz.sort_index(), left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("2s"))
    else:
        for n in names:
            dat = get_data(n)
            if dat is None:
                continue
            tt, yy = dat[0], dat[1]
            yy = np.asarray(yy)
            if yy.ndim == 2 and yy.shape[1] >= 3:
                idx = pd.to_datetime(tt, unit="s", errors="coerce").tz_localize(None)
                dfmag = pd.DataFrame(yy[:, :3], index=idx, columns=["Bx", "By", "Bz"])
                break

    if dfmag is None or len(dfmag) == 0:
        raise RuntimeError("ACE MAG load failed (no MFI data).")

    dfmag = dfmag.loc[~dfmag.index.isna()].sort_index()
    dfmag = dfmag.loc[~dfmag.index.duplicated(keep="first")]
    dfmag.index.name = "datetime"

    for c in ("Bx", "By", "Bz"):
        if c in dfmag.columns:
            dfmag.loc[dfmag[c] < -1e30, c] = np.nan

    if verbose and not dfmag.empty:
        print("--- MAG ---")
        print(f"Source            : {mag_source}")
        print(f"Returned coverage : {dfmag.index[0]} -> {dfmag.index[-1]}")
        print(f"Columns           : {list(dfmag.columns)}")

    # --------------------
    # PAR: CDAWeb first (<= 2024-07-09), else pyspedas swe (h0 -> k0)
    # --------------------
    dfpar: Optional[pd.DataFrame] = None
    plasma_source = "none"
    manufactured_vec = False

    if t0_req <= _ACE_SWEPAM_L2_END:
        if verbose:
            print("--- PAR ---")
            print("Trying CDAWeb      : AC_H0_SWE (legacy SWEPAM-L2)")

        try:
            from cdasws import CdasWs  # type: ignore

            cdas = CdasWs()
            t0s = pd.Timestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ")
            t1s = pd.Timestamp(t1).strftime("%Y-%m-%dT%H:%M:%SZ")

            payload = None
            try:
                payload = cdas.get_data("AC_H0_SWE", ["Np", "Tpr", "Vp", "V_GSE"], t0s, t1s)
            except Exception:
                payload = cdas.get_data("AC_H0_SWE", ["Np", "Tpr", "Vp"], t0s, t1s)

            epoch_key = "Epoch" if isinstance(payload, dict) and "Epoch" in payload else ("epoch" if isinstance(payload, dict) and "epoch" in payload else None)
            if epoch_key is not None:
                idx = pd.to_datetime(payload[epoch_key], errors="coerce").tz_localize(None)
                dfpar = pd.DataFrame(index=pd.DatetimeIndex(idx))

                if "Np" in payload:
                    dfpar["np"] = np.asarray(payload["Np"]).reshape(-1)
                if "Tpr" in payload:
                    dfpar["Tp"] = np.asarray(payload["Tpr"]).reshape(-1)  # likely K here
                if "Vp" in payload:
                    dfpar["Vp"] = np.asarray(payload["Vp"]).reshape(-1)

                if "V_GSE" in payload:
                    v = np.asarray(payload["V_GSE"])
                    if v.ndim == 2 and v.shape[1] >= 3:
                        dfpar["Vx"] = v[:, 0]
                        dfpar["Vy"] = v[:, 1]
                        dfpar["Vz"] = v[:, 2]

                dfpar = dfpar.loc[~dfpar.index.isna()].sort_index()
                dfpar = dfpar.loc[~dfpar.index.duplicated(keep="first")]
                dfpar.index.name = "datetime"

                if len(dfpar):
                    plasma_source = "cdaweb:AC_H0_SWE"
        except Exception:
            dfpar = None

    if dfpar is None or len(dfpar) == 0:
        if verbose:
            print("--- PAR ---")
            if t0_req > _ACE_SWEPAM_L2_END:
                print(f"CDAWeb skipped     : requested start {t0_req} is after {_ACE_SWEPAM_L2_END}")
            else:
                print("CDAWeb failed       : falling back to pyspedas ace.swe")

        par_dtype_req = str(vars_2_downnload.get("par", {}).get("datatype", "h0")).lower()
        tried = []
        for dt in [par_dtype_req, "h0", "k0"]:
            if dt in tried:
                continue
            tried.append(dt)

            try:
                del_data("*")
                pyspedas.projects.ace.swe(trange=[str(t0), str(t1)], datatype=dt, time_clip=True)
                names = list(tplot_names())

                vvec_name = None
                for n in names:
                    ln = n.lower()
                    if ("v" in ln) and (("gse" in ln) or ("rtn" in ln)):
                        dat = get_data(n)
                        if dat is None:
                            continue
                        yy = np.asarray(dat[1])
                        if yy.ndim == 2 and yy.shape[1] >= 3:
                            vvec_name = n
                            break

                vp_name = None
                np_name = None
                tp_name = None
                for n in names:
                    ln = n.lower()
                    if vp_name is None and (ln == "vp" or ln.endswith("vp") or "flow_speed" in ln or ("speed" in ln and "mag" not in ln)):
                        vp_name = n
                    if np_name is None and (ln == "np" or ln.endswith("np") or "proton_density" in ln or ("dens" in ln and "mag" not in ln)):
                        np_name = n
                    if tp_name is None and (ln == "tpr" or "tpr" in ln or (("temp" in ln or ln == "tp") and "tplot" not in ln)):
                        tp_name = n

                if vp_name is None:
                    continue

                tv, vv = get_data(vp_name)
                idx = pd.to_datetime(tv, unit="s", errors="coerce").tz_localize(None)
                dfpar = pd.DataFrame(index=pd.DatetimeIndex(idx))
                dfpar["Vp"] = np.asarray(vv).reshape(-1)

                if np_name is not None:
                    tn, nn = get_data(np_name)
                    dfn = pd.DataFrame({"np": np.asarray(nn).reshape(-1)}, index=pd.to_datetime(tn, unit="s", errors="coerce").tz_localize(None)).sort_index()
                    dfpar = pd.merge_asof(dfpar.sort_index(), dfn, left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("2min"))

                if tp_name is not None:
                    tt, ttval = get_data(tp_name)
                    dft = pd.DataFrame({"Tp": np.asarray(ttval).reshape(-1)}, index=pd.to_datetime(tt, unit="s", errors="coerce").tz_localize(None)).sort_index()
                    dfpar = pd.merge_asof(dfpar.sort_index(), dft, left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("2min"))

                if vvec_name is not None:
                    tvec, vvec = get_data(vvec_name)
                    vvec = np.asarray(vvec)
                    dfv = pd.DataFrame(
                        vvec[:, :3],
                        index=pd.to_datetime(tvec, unit="s", errors="coerce").tz_localize(None),
                        columns=["Vx", "Vy", "Vz"] if "gse" in vvec_name.lower() else ["Vr", "Vt", "Vn"],
                    ).sort_index()
                    dfpar = pd.merge_asof(dfpar.sort_index(), dfv, left_index=True, right_index=True, direction="nearest", tolerance=pd.Timedelta("2min"))
                else:
                    comp = dfpar["Vp"].astype(float) / np.sqrt(3.0)
                    if frame == "GSE":
                        dfpar["Vx"] = comp
                        dfpar["Vy"] = comp
                        dfpar["Vz"] = comp
                    else:
                        dfpar["Vr"] = comp
                        dfpar["Vt"] = comp
                        dfpar["Vn"] = comp
                    manufactured_vec = True

                dfpar = dfpar.loc[~dfpar.index.isna()].sort_index()
                dfpar = dfpar.loc[~dfpar.index.duplicated(keep="first")]
                dfpar.index.name = "datetime"

                if len(dfpar):
                    plasma_source = f"pyspedas:ace.swe:{dt}"
                    break
            except Exception:
                dfpar = None
                continue

    if dfpar is None or len(dfpar) == 0:
        misc = {
            "Mag": {"Frac_miss": 0.0, "resol": mag_resol_ms / 1000.0},
            "Par": {"Frac_miss": 100.0, "resol": part_resol_ms / 1000.0},
            "ACE": {"mag_source": mag_source, "plasma_source": plasma_source, "vector_manufactured": False},
        }
        if verbose:
            print("--- PAR ---")
            print("No particle data returned.")
        return dfmag, None, None, None, misc

    for c in ("np", "Tp", "Vp", "Vx", "Vy", "Vz", "Vr", "Vt", "Vn"):
        if c in dfpar.columns:
            dfpar.loc[dfpar[c] < -1e30, c] = np.nan
    if "np" in dfpar.columns:
        dfpar.loc[dfpar["np"] < 0, "np"] = np.nan

    if "Vp" not in dfpar.columns:
        if all(k in dfpar.columns for k in ("Vx", "Vy", "Vz")):
            dfpar["Vp"] = np.sqrt(dfpar["Vx"] ** 2 + dfpar["Vy"] ** 2 + dfpar["Vz"] ** 2)
        elif all(k in dfpar.columns for k in ("Vr", "Vt", "Vn")):
            dfpar["Vp"] = np.sqrt(dfpar["Vr"] ** 2 + dfpar["Vt"] ** 2 + dfpar["Vn"] ** 2)

    # Temperature normalization consistent with WIND
    if "Tp" in dfpar.columns:
        tp_raw = pd.to_numeric(dfpar["Tp"], errors="coerce").to_numpy(dtype=float, copy=False)
        tp_med = np.nanmedian(tp_raw) if np.any(np.isfinite(tp_raw)) else np.nan

        if np.isfinite(tp_med) and tp_med > 500.0:
            tp_k = tp_raw
            tp_ev = tp_k / _EV_TO_K
        else:
            tp_ev = tp_raw
            tp_k = tp_ev * _EV_TO_K

        dfpar["Tp"] = tp_ev
        dfpar["Tp_K"] = tp_k
        dfpar["Vth"] = _VTH_COEFF * np.sqrt(np.clip(tp_k, 0.0, None))

        if "TEMP" not in dfpar.columns:
            dfpar["TEMP"] = dfpar["Tp"]

    if verbose and not dfpar.empty:
        vcols = [c for c in ("Vx", "Vy", "Vz", "Vr", "Vt", "Vn") if c in dfpar.columns]
        tcols = [c for c in ("Tp", "Tp_K", "Vth") if c in dfpar.columns]
        print("--- PAR ---")
        print(f"Source            : {plasma_source}")
        print(f"Returned coverage : {dfpar.index[0]} -> {dfpar.index[-1]}")
        print(f"Vector columns    : {vcols if vcols else 'none'}")
        if manufactured_vec:
            print("Vector note       : manufactured only because fallback had only Vp; used V*=Vp/sqrt(3)")
        if tcols:
            print("Thermals          : Tp[eV], Tp_K[K], Vth[km/s] computed (WIND-consistent)")
        print(f"Key columns       : {sorted(set(['np','Vp'] + vcols + tcols))}")

    # --------------------
    # Clip to requested window using your funcs
    # --------------------
    ind1 = func.string_to_datetime_index(t0_req)
    ind2 = func.string_to_datetime_index(t1_req)
    try:
        dfmag = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
    except Exception:
        dfmag = dfmag.loc[(dfmag.index >= t0_req) & (dfmag.index <= t1_req)]
    try:
        dfpar = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)
    except Exception:
        dfpar = dfpar.loc[(dfpar.index >= t0_req) & (dfpar.index <= t1_req)]

    # --------------------
    # Native cadence estimate on the SELECTED interval + notify if requested cadence is higher
    # --------------------
    native_mag_s = np.nan
    native_par_s = np.nan

    if isinstance(dfmag, pd.DataFrame) and len(dfmag.index) >= 3:
        dt_ns = np.diff(dfmag.index.values.astype("datetime64[ns]").astype("int64"))
        dt_ns = dt_ns[dt_ns > 0]
        if dt_ns.size:
            native_mag_s = float(np.nanmedian(dt_ns) / 1e9)

    if isinstance(dfpar, pd.DataFrame) and len(dfpar.index) >= 3:
        dt_ns = np.diff(dfpar.index.values.astype("datetime64[ns]").astype("int64"))
        dt_ns = dt_ns[dt_ns > 0]
        if dt_ns.size:
            native_par_s = float(np.nanmedian(dt_ns) / 1e9)

    if verbose:
        print("--- Cadence check (selected interval) ---")
        if np.isfinite(native_mag_s):
            print(f"MAG native cadence : ~{native_mag_s:.3f}s (median Δt over selected interval)")
            if mag_req_s < native_mag_s:
                print(f"MAG note           : requested cadence {mag_req_s:.3f}s is HIGHER than native")
            else:
                print(f"MAG note           : requested cadence {mag_req_s:.3f}s is <= native; output is downsampled/compatible.")
        else:
            print("MAG native cadence : could not estimate (too few samples)")

        if np.isfinite(native_par_s):
            print(f"PAR native cadence : ~{native_par_s:.3f}s (median Δt over selected interval)")
            if par_req_s < native_par_s:
                print(f"PAR note           : requested cadence {par_req_s:.3f}s is HIGHER than native")
            else:
                print(f"PAR note           : requested cadence {par_req_s:.3f}s is <= native; output is downsampled/compatible.")
        else:
            print("PAR native cadence : could not estimate (too few samples)")

    # --------------------
    # Resample / gap stats using your funcs
    # --------------------
    misc: Dict[str, Any] = {
        "ACE": {
            "mag_source": mag_source,
            "plasma_source": plasma_source,
            "vector_manufactured": manufactured_vec,
            "native_mag_cad_s": native_mag_s,
            "native_par_cad_s": native_par_s,
            "requested_mag_cad_s": mag_req_s,
            "requested_par_cad_s": par_req_s,
        }
    }
    big_gaps = None
    dfdis = None

    try:
        mag_res = func.resample_timeseries_estimate_gaps(dfmag, mag_resol_ms, settings.get("gap_time_threshold", 5))
        dfmag_r = mag_res.get("resampled_df", dfmag)
        misc["Mag"] = {k: mag_res.get(k, None) for k in ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")}
    except Exception:
        dfmag_r = dfmag
        misc["Mag"] = {"Frac_miss": 0.0, "Large_gaps": np.nan, "Tot_gaps": np.nan, "resol": mag_resol_ms / 1000.0}

    try:
        par_res = func.resample_timeseries_estimate_gaps(dfpar, part_resol_ms, settings.get("gap_time_threshold", 5))
        dfpar_r = par_res.get("resampled_df", dfpar)
        misc["Par"] = {k: par_res.get(k, None) for k in ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")}
    except Exception:
        dfpar_r = dfpar
        misc["Par"] = {
            "Frac_miss": float(dfpar_r.isna().any(axis=1).mean() * 100.0),
            "Large_gaps": np.nan,
            "Tot_gaps": np.nan,
            "resol": part_resol_ms / 1000.0,
        }

    if verbose:
        print("--- Diagnostics ---")
        print(f"Mag fraction missing: {misc.get('Mag', {}).get('Frac_miss', np.nan)}")
        print(f"Par fraction missing: {misc.get('Par', {}).get('Frac_miss', np.nan)}")
        print("=== ACE loader done ===")

    out_mag = dfmag_r[["Bx", "By", "Bz"]] if all(c in dfmag_r.columns for c in ("Bx", "By", "Bz")) else dfmag_r.copy()
    out_par = dfpar_r.copy()

    # Attach ephemeris aligned to particle cadence (automatic unless explicitly disabled)
    if isinstance(out_par, pd.DataFrame) and len(out_par) > 0 and bool(settings.get("Down_ephem", True)):
        try:
            out_par, dfdis = _attach_ephem_to_par(out_par, target="ACE", settings=settings)
        except Exception:
            if verbose:
                print("[ACE] Ephemeris fetch failed; returning dfdis=None and no ephem columns on dfpar.")
            dfdis = None

    return out_mag, out_par, dfdis, big_gaps, misc