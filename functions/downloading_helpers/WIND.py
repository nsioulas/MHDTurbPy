"""WIND mission download helpers.

This module provides the WIND-specific loaders consumed by ``download_data.py``.
The public API intentionally keeps the historical function names/signatures used by
the pipeline:

- ``LoadTimeSeriesWind_electrons``
- ``LoadTimeSeriesWind_particles``
- ``LoadHighResMagWind``
- ``LoadTimeSeriesWIND``

This revision adds an optional JPL Horizons ephemeris attachment (via the repo's
``sc_pos.horizons_sun_lonlat`` helper). The ephemeris is interpolated onto the
final particle cadence (after resampling / optional Hampel filtering) and returned
as ``dfdis`` with the same DatetimeIndex as ``dfpar``.

By default (``settings['ephem_units']`` absent), ephemeris Cartesian components are
attached in km:

- ``dfpar[['Dist_km','x_km','y_km','z_km']]``

If the user selects AU via ``settings['ephem_units'] = 'au'``, then:

- ``dfpar[['Dist_au','x_au','y_au','z_au']]``

The returned ``dfdis`` always contains both km and AU columns (plus lon/lat).
"""

from __future__ import annotations

import importlib.util
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from cdasws import CdasWs
from scipy import constants

# Ensure project paths so we can import sc_pos utilities from within downloading_helpers.
_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[1]
_PATH_SETUP = _REPO_ROOT / "functions" / "path_setup.py"
_spec = importlib.util.spec_from_file_location("mhdturbpy_path_setup", _PATH_SETUP)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load path setup from {_PATH_SETUP}")
_path_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_path_setup)
ensure_project_paths = _path_setup.ensure_project_paths
ensure_project_paths(
    start=Path(__file__).resolve(),
    include_downloading_helpers=True,
    include_anisotropy_toolbox=True,
)

import general_functions as func  # noqa: E402

# Do not share one global CdasWs session across parallel notebook threads.
# Create a fresh client per request instead.

# Kelvin per eV, used for eV -> K conversion.
_EV_TO_K = 1.0 / constants.physical_constants["Boltzmann constant in eV/K"][0]
_AU_KM = 1.495978707e8


def _to_utc_datetime(value: Any) -> pd.Timestamp:
    """Convert input to timezone-aware UTC timestamp."""
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _empty_diag() -> Dict[str, float]:
    return {"Frac_miss": 100.0, "Large_gaps": 100.0, "Tot_gaps": 100.0, "resol": 100.0}


def _subset_interval(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Subset using pipeline utility, with fallback for index-type mismatches."""
    try:
        return func.use_dates_return_elements_of_df_inbetween(start, end, df)
    except Exception:
        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        return func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(start), pd.to_numeric(end), tmp)


def _clean_fill_values(df: pd.DataFrame, cols: Tuple[str, ...], threshold: float = -1e30) -> pd.DataFrame:
    """Replace common CDAWeb fill values with NaN for selected columns."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out.loc[out[col] < threshold, col] = np.nan
    return out


def _pick_first_key(data: Any, *candidates: str):
    """Return the first present key from a CDAWeb/SpaceData payload."""
    for key in candidates:
        try:
            if key in data:
                return data[key]
        except Exception:
            pass
    for key in candidates:
        try:
            return data[key]
        except Exception:
            continue
    raise KeyError(f"None of the candidate keys were found: {candidates}")




def _cdas_get_data(dataset: str, variables, start_time, end_time, retries: int = 4, sleep_s: float = 2.0):
    """Thread-safe/retry-safe CDAWeb request using a fresh client per attempt."""
    last_exc = None
    for attempt in range(max(1, int(retries))):
        try:
            client = CdasWs()
            return client.get_data(dataset, list(variables), start_time, end_time)
        except Exception as exc:
            last_exc = exc
            if attempt >= int(retries) - 1:
                raise
            time.sleep(float(sleep_s) * (2 ** attempt))
    raise last_exc  # pragma: no cover


def _iter_time_chunks(start_time, end_time, max_days: float):
    """Yield contiguous [start, end] chunks spanning the requested interval."""
    t0 = pd.Timestamp(start_time)
    t1 = pd.Timestamp(end_time)
    if t1 <= t0:
        yield t0, t1
        return

    dt = pd.Timedelta(days=float(max_days))
    left = t0
    while left < t1:
        right = min(left + dt, t1)
        yield left, right
        left = right


def _concat_timeseries_chunks(frames):
    """Concatenate chunked time-series pieces safely."""
    valid = [df for df in frames if isinstance(df, pd.DataFrame) and len(df) > 0]
    if not valid:
        return pd.DataFrame()
    out = pd.concat(valid, axis=0).sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    return out


def describe_wind_source_selection(settings: Dict[str, Any]) -> Dict[str, str]:
    """Return dataset names selected by current WIND cadence settings."""
    mag_res = float(settings["MAG_resol"])
    part_res = float(settings["part_resol"])

    mag_source = "WI_H2_MFI"
    part_source = "WI_PM_3DP" if part_res <= 3 else "WI_PLSP_3DP"
    elec_source = "WI_H5_SWE" if settings.get("Down_electrons", False) else "disabled"
    coord_out = "RTN(L1-approx)" if bool(settings.get("in_rtn", True)) else "GSE"
    mag_note = "native cadence" if mag_res < 3 else "downloaded at native cadence then synchronized to particle cadence"

    return {"mag": mag_source, "par": part_source, "elec": elec_source, "coord_out": coord_out, "mag_note": mag_note}


def _approx_gse_to_l1_rtn(vec_gse: np.ndarray) -> np.ndarray:
    """Approximate GSE -> RTN transform for near-Earth/L1 spacecraft.

    Convention used across this repository (same as ACE helper):
    R ~ -X_GSE, T ~ Y_GSE, N ~ Z_GSE.
    """
    x = vec_gse[:, 0]
    y = vec_gse[:, 1]
    z = vec_gse[:, 2]
    return np.column_stack([-x, y, z])


def _ensure_nx3(vec: Any, n_expected: Optional[int] = None) -> np.ndarray:
    """Ensure vector-like input is shaped (N, 3)."""
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D vec3 array, got shape {arr.shape}")
    if arr.shape[1] == 3:
        out = arr
    elif arr.shape[0] == 3:
        out = arr.T
    else:
        raise ValueError(f"Could not interpret vec3 array shape {arr.shape}")

    if n_expected is not None and out.shape[0] != int(n_expected):
        raise ValueError(f"Vec3 length mismatch: expected {n_expected}, got {out.shape[0]}")
    return out


def _rename_vec_by_frame(df: pd.DataFrame, frame: str, prefix: str) -> pd.DataFrame:
    """Rename vector columns to either RTN (r,t,n) or XYZ (x,y,z)."""
    out = df.copy()
    frame = str(frame).upper()
    if prefix == "B":
        out = out.rename(
            columns={"c1": "Br", "c2": "Bt", "c3": "Bn"} if frame == "RTN" else {"c1": "Bx", "c2": "By", "c3": "Bz"}
        )
    elif prefix == "V":
        out = out.rename(
            columns={"c1": "Vr", "c2": "Vt", "c3": "Vn"} if frame == "RTN" else {"c1": "Vx", "c2": "Vy", "c3": "Vz"}
        )
    return out


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
    idx_orig = pd.DatetimeIndex(pd.to_datetime(index))
    if idx_orig.tz is not None:
        idx_orig = idx_orig.tz_convert(None)

    if len(idx_orig) == 0:
        return pd.DataFrame(
            index=idx_orig,
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

    idx_valid = pd.DatetimeIndex(idx_orig[~idx_orig.isna()])
    idx_support = idx_valid.sort_values().unique()

    t_ns = idx_support.view("i8")
    dt_s = float(np.nanmedian(np.diff(t_ns))) / 1e9 if len(t_ns) >= 2 else 600.0
    step = _format_horizons_step(dt_s)

    pad = pd.Timedelta("2h") if step.endswith("h") else pd.Timedelta("20m")
    start = (idx_support[0] - pad).strftime("%Y-%m-%dT%H:%M:%S")
    stop = (idx_support[-1] + pad).strftime("%Y-%m-%dT%H:%M:%S")

    base = _horizons_base_df(target=str(target), start=start, stop=stop, step=step)

    need = ("hgs_r_au", "hgs_lon_deg", "hgs_lat_deg", "hee_x_au", "hee_y_au", "hee_z_au")
    missing = [k for k in need if k not in base.columns]
    if missing:
        raise RuntimeError(f"Horizons base dataframe is missing required columns: {missing}")

    eph = pd.DataFrame(index=pd.DatetimeIndex(base.index))
    eph = eph.loc[~eph.index.isna()]
    eph = eph.loc[~eph.index.duplicated(keep="first")]

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

    support = eph.index.union(idx_support).sort_values().unique()
    tmp = eph.reindex(support).sort_index()
    tmp = tmp.interpolate(method="time", limit_direction="both")

    out_valid = tmp.reindex(idx_valid)
    out = pd.DataFrame(index=idx_orig, columns=out_valid.columns, dtype=float)
    out.loc[idx_valid, :] = out_valid.to_numpy()
    return out


def _attach_ephem_to_par(dfpar: pd.DataFrame, target: str, settings: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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



def LoadTimeSeriesWind_electrons(start_time, end_time, settings):
    """Load WIND electron moments from SWE H5 and return Te, Te_core in eV."""
    from astropy import units as u  # type: ignore

    t0 = _to_utc_datetime(start_time).to_pydatetime()
    t1 = _to_utc_datetime(end_time).to_pydatetime()

    frames = []
    for c0, c1 in _iter_time_chunks(t0, t1, max_days=14.0):
        status, data = _cdas_get_data("WI_H5_SWE", ["T_elec", "TcElec"], c0, c1)
        if not status:
            continue

        te_ev = u.Quantity(data["T_elec"], u.K).to(u.eV, equivalencies=u.temperature_energy()).value
        te_core_ev = u.Quantity(data["TcElec"], u.K).to(u.eV, equivalencies=u.temperature_energy()).value

        frames.append(pd.DataFrame(index=pd.to_datetime(data["Epoch"]), data={"Te": te_ev, "Te_core": te_core_ev}))

    df = _concat_timeseries_chunks(frames)
    if df.empty:
        raise RuntimeError("Failed to download WI_H5_SWE electron data.")
    return _clean_fill_values(df, ("Te", "Te_core"))



def LoadTimeSeriesWind_particles(start_time, end_time, settings):
    """Load WIND proton moments; source depends on requested ``part_resol``."""
    part_resol = float(settings["part_resol"])
    frames = []

    if part_resol <= 3:
        t0 = _to_utc_datetime(start_time).to_pydatetime().replace(tzinfo=pytz.UTC)
        t1 = _to_utc_datetime(end_time).to_pydatetime().replace(tzinfo=pytz.UTC)

        for c0, c1 in _iter_time_chunks(t0, t1, max_days=7.0):
            status, data = _cdas_get_data("WI_PM_3DP", ["P_DENS", "P_VELS", "P_TEMP", "TIME"], c0, c1)
            if not status:
                continue

            vec = _ensure_nx3(data["P_VELS"], n_expected=len(data["Epoch"]))
            if bool(settings.get("in_rtn", True)):
                vec = _approx_gse_to_l1_rtn(vec)
                frame = "RTN"
            else:
                frame = "GSE"

            df_chunk = pd.DataFrame(
                index=pd.to_datetime(data["Epoch"]),
                data={
                    "c1": vec[:, 0],
                    "c2": vec[:, 1],
                    "c3": vec[:, 2],
                    "np": data["P_DENS"],
                    "Tp": data["P_TEMP"],
                },
            )
            df_chunk = _rename_vec_by_frame(df_chunk, "RTN" if frame == "RTN" else "GSE", "V")
            df_chunk["Tp_K"] = _EV_TO_K * df_chunk["Tp"]
            df_chunk["Vth"] = 0.128487 * np.sqrt(df_chunk["Tp_K"].clip(lower=0))
            frames.append(df_chunk)

        dfpar = _concat_timeseries_chunks(frames)
        if dfpar.empty:
            raise RuntimeError("Failed to download WI_PM_3DP particle data.")
        qtn_flag = "No_QTN"
    else:
        for c0, c1 in _iter_time_chunks(pd.Timestamp(start_time), pd.Timestamp(end_time), max_days=14.0):
            status, data = _cdas_get_data(
                "WI_PLSP_3DP",
                ["MOM.P.DENSITY", "MOM.P.VELOCITY", "MOM.P.VTHERMAL", "TIME"],
                str(pd.Timestamp(c0)),
                str(pd.Timestamp(c1)),
            )
            if not status:
                continue

            vel = _pick_first_key(data, "MOM$P$VELOCITY", "MOM.P.VELOCITY")
            dens = _pick_first_key(data, "MOM$P$DENSITY", "MOM.P.DENSITY")
            vth = _pick_first_key(data, "MOM$P$VTHERMAL", "MOM.P.VTHERMAL")

            vec = _ensure_nx3(vel, n_expected=len(data["Epoch"]))
            if bool(settings.get("in_rtn", True)):
                vec = _approx_gse_to_l1_rtn(vec)
                frame = "RTN"
            else:
                frame = "GSE"

            df_chunk = pd.DataFrame(
                index=pd.to_datetime(data["Epoch"]),
                data={
                    "c1": vec[:, 0],
                    "c2": vec[:, 1],
                    "c3": vec[:, 2],
                    "np": dens,
                    "Vth": vth,
                },
            )
            df_chunk = _rename_vec_by_frame(df_chunk, "RTN" if frame == "RTN" else "GSE", "V")

            vth_mps = np.asarray(df_chunk["Vth"], dtype=float) * 1e3
            tp_k = (constants.m_p * (vth_mps ** 2)) / (2 * constants.k)
            df_chunk["Tp"] = tp_k / _EV_TO_K
            frames.append(df_chunk)

        dfpar = _concat_timeseries_chunks(frames)
        if dfpar.empty:
            raise RuntimeError("Failed to download WI_PLSP_3DP particle data.")
        qtn_flag = "No_QTN"

    vec_cols = ("Vr", "Vt", "Vn") if bool(settings.get("in_rtn", True)) else ("Vx", "Vy", "Vz")
    dfpar = _clean_fill_values(dfpar, vec_cols + ("np", "Tp", "Vth"))

    dfdis_placeholder = pd.DataFrame(index=dfpar.index, data={"Dist_au": np.nan, "lon": np.nan, "lat": np.nan, "RAD_AU": np.nan})
    return dfpar, dfdis_placeholder, qtn_flag



def LoadHighResMagWind(start_time, end_time, settings, verbose=True):
    """Load WIND magnetic field.

    Always download ``WI_H2_MFI``. When ``MAG_resol >= 3``, the final cadence
    matching to particles is handled later inside ``LoadTimeSeriesWIND`` via
    ``func.synchronize_dfs(..., upsample=False)``.
    """
    t0 = _to_utc_datetime(start_time).to_pydatetime().replace(tzinfo=pytz.UTC)
    t1 = _to_utc_datetime(end_time).to_pydatetime().replace(tzinfo=pytz.UTC)

    frames = []
    for c0, c1 in _iter_time_chunks(t0, t1, max_days=14.0):
        status, data = _cdas_get_data("WI_H2_MFI", ["BGSE", "BF1"], c0, c1)
        if not status:
            continue

        vec = _ensure_nx3(data["BGSE"], n_expected=len(data["Epoch"]))
        if bool(settings.get("in_rtn", True)):
            vec = _approx_gse_to_l1_rtn(vec)
            frame = "RTN"
        else:
            frame = "GSE"

        df_chunk = pd.DataFrame(
            index=pd.to_datetime(data["Epoch"]),
            data={
                "c1": vec[:, 0],
                "c2": vec[:, 1],
                "c3": vec[:, 2],
                "Btot": data["BF1"],
            },
        )
        df_chunk = _rename_vec_by_frame(df_chunk, "RTN" if frame == "RTN" else "GSE", "B")
        frames.append(df_chunk)

    dfmag = _concat_timeseries_chunks(frames)
    if dfmag.empty:
        raise RuntimeError("Failed to download WI_H2_MFI magnetic data.")

    bcols = ("Br", "Bt", "Bn") if bool(settings.get("in_rtn", True)) else ("Bx", "By", "Bz")
    dfmag = _clean_fill_values(dfmag, bcols + ("Btot",), threshold=-1e30)
    dfmag.loc[np.abs(dfmag["Btot"]) > 1e3, list(bcols) + ["Btot"]] = np.nan

    if verbose and not dfmag.empty:
        print("Done.")
        print(f"Input tstart = {t0}, tend = {t1}")
        print(f"Returned tstart = {dfmag.index[0]}, tend = {dfmag.index[-1]}")

    return dfmag


def LoadTimeSeriesWIND(
    start_time,
    end_time,
    settings,
    gap_time_threshold=10,
    time_amount=4,
    time_unit="h",
):
    """Load WIND MAG/particle/electron timeseries for the pipeline.

    Returns
    -------
    tuple
        ``(dfmag, dfpar, df_elec, dfdis, big_gaps_mag, big_gaps_qtn,
        big_gaps_par, big_gaps_elec, misc, qtn_flag)``
    """
    del gap_time_threshold  # kept in signature for backward compatibility

    t0i, t1i = func.ensure_time_format(start_time, end_time)
    t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
    t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

    ind1 = func.string_to_datetime_index(t0i)
    ind2 = func.string_to_datetime_index(t1i)

    big_gaps_qtn = None
    qtn_flag = "No_QTN"

    # --- Magnetic field ---
    try:
        dfmag_raw = LoadHighResMagWind(pd.Timestamp(t0), pd.Timestamp(t1), settings, verbose=True)
        dfmag = _subset_interval(dfmag_raw, ind1, ind2)
    except Exception:
        traceback.print_exc()
        dfmag = None

    # --- Electrons ---
    download_electrons = bool(settings.get("Down_electrons", False))
    if download_electrons:
        try:
            df_elec_raw = LoadTimeSeriesWind_electrons(pd.Timestamp(t0), pd.Timestamp(t1), settings)
            df_elec = _subset_interval(df_elec_raw, ind1, ind2)
            big_gaps_elec = func.find_big_gaps(df_elec, settings["Big_Gaps"]["E_big_gaps"], str(ind1), str(ind2))
            diagnostics_elec = func.resample_timeseries_estimate_gaps(df_elec, settings["part_resol"], large_gaps=10)
            print("Elec fraction missing", diagnostics_elec["Frac_miss"])
        except Exception:
            traceback.print_exc()
            df_elec = None
            big_gaps_elec = None
            diagnostics_elec = _empty_diag()
    else:
        df_elec = None
        big_gaps_elec = None
        diagnostics_elec = _empty_diag()

    big_gaps_mag = None
    diagnostics_mag = _empty_diag()

    # --- Particles ---
    try:
        dfpar_raw, _, qtn_flag = LoadTimeSeriesWind_particles(pd.Timestamp(t0), pd.Timestamp(t1), settings)
        dfpar = _subset_interval(dfpar_raw, ind1, ind2)

        if isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0 and float(settings.get("MAG_resol", 0)) >= 3:
            dfmag, dfpar = func.synchronize_dfs(dfmag, dfpar, upsample=False)

        big_gaps_par = func.find_big_gaps(dfpar, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
        diagnostics_par = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
        print("Par fraction missing", diagnostics_par["Frac_miss"])

        if isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0:
            big_gaps_mag = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], str(ind1), str(ind2))
            mag_target_resol = settings["part_resol"] if float(settings.get("MAG_resol", 0)) >= 3 else settings["MAG_resol"]
            diagnostics_mag = func.resample_timeseries_estimate_gaps(dfmag, mag_target_resol, large_gaps=10)
            print("Mag fraction missing", diagnostics_mag["Frac_miss"])
    except Exception:
        traceback.print_exc()
        dfpar = None
        big_gaps_par = None
        diagnostics_par = _empty_diag()
    # Optional despiking on resampled particle moments
    if settings.get("apply_hampel", False) and isinstance(diagnostics_par, dict) and "resampled_df" in diagnostics_par:
        try:
            print("Applying hampel filter to particle data!")
            dfpar_filt = diagnostics_par["resampled_df"].copy()
            list_2_hampel = ["Vr", "Vt", "Vn", "np", "Vth"] if bool(settings.get("in_rtn", True)) else ["Vx", "Vy", "Vz", "np", "Vth"]
            ws_hampel = settings.get("hampel_params", {}).get("w", 200)
            n_hampel = settings.get("hampel_params", {}).get("std", 3)

            for col in list_2_hampel:
                if col in dfpar_filt.columns:
                    outliers = func.hampel(dfpar_filt[col], window_size=ws_hampel, n=n_hampel)
                    dfpar_filt.loc[dfpar_filt.index[outliers], col] = np.nan

            dfpar = dfpar_filt
            print("Applied hampel filter to WIND particle columns:", list_2_hampel, "Window size", ws_hampel)
        except Exception:
            traceback.print_exc()

    # Attach ephemeris aligned to final particle cadence (automatic unless explicitly disabled)
    dfdis = None
    if isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0 and bool(settings.get("Down_ephem", True)):
        try:
            dfpar, dfdis = _attach_ephem_to_par(dfpar, target="WIND", settings=settings)
        except Exception:
            traceback.print_exc()
            dfdis = None

    keys_to_keep = ["Frac_miss", "Large_gaps", "Tot_gaps", "resol"]
    misc = {
        "Par": func.filter_dict(diagnostics_par, keys_to_keep),
        "Mag": func.filter_dict(diagnostics_mag, keys_to_keep),
        "Elec": func.filter_dict(diagnostics_elec, keys_to_keep),
    }

    # Use resampled MAG dataframe when available (matches pipeline expectation)
    if isinstance(diagnostics_mag, dict) and "resampled_df" in diagnostics_mag:
        dfmag_out = diagnostics_mag["resampled_df"]
    else:
        dfmag_out = dfmag

    return dfmag_out, dfpar, df_elec, dfdis, big_gaps_mag, big_gaps_qtn, big_gaps_par, big_gaps_elec, misc, qtn_flag
