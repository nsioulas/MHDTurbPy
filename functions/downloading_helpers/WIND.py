"""WIND mission download helpers.

This module provides the WIND-specific loaders consumed by ``download_data.py``.
The public API intentionally keeps the historical function names/signatures used by
the pipeline:

- ``LoadTimeSeriesWind_electrons``
- ``LoadTimeSeriesWind_particles``
- ``LoadHighResMagWind``
- ``LoadTimeSeriesWIND``
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
from cdasws import CdasWs
from scipy import constants

import general_functions as func

cdas = CdasWs()

# Kelvin per eV, used for eV -> K conversion.
_EV_TO_K = 1.0 / constants.physical_constants["Boltzmann constant in eV/K"][0]


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


def _build_wind_distance_df(index: pd.Index) -> pd.DataFrame:
    """WIND ephemeris placeholder used by legacy diagnostics pipeline."""
    n = len(index)
    return pd.DataFrame(
        index=index,
        data={
            "Dist_au": np.ones(n),
            "lon": np.ones(n),
            "lat": np.ones(n),
            "RAD_AU": np.ones(n),
        },
    )


def _clean_fill_values(df: pd.DataFrame, cols: Tuple[str, ...], threshold: float = -1e30) -> pd.DataFrame:
    """Replace common CDAWeb fill values with NaN for selected columns."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out.loc[out[col] < threshold, col] = np.nan
    return out






def _approx_gse_to_l1_rtn(vec_gse: np.ndarray) -> np.ndarray:
    """Approximate GSE -> RTN transform for near-Earth/L1 spacecraft.

    Convention used across this repository (same as ACE helper):
    R ~ -X_GSE, T ~ Y_GSE, N ~ Z_GSE.
    """
    x = vec_gse[:, 0]
    y = vec_gse[:, 1]
    z = vec_gse[:, 2]
    return np.column_stack([-x, y, z])




def _ensure_nx3(vec: Any, n_expected: int | None = None) -> np.ndarray:
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
        out = out.rename(columns={"c1": "Br", "c2": "Bt", "c3": "Bn"} if frame == "RTN" else {"c1": "Bx", "c2": "By", "c3": "Bz"})
    elif prefix == "V":
        out = out.rename(columns={"c1": "Vr", "c2": "Vt", "c3": "Vn"} if frame == "RTN" else {"c1": "Vx", "c2": "Vy", "c3": "Vz"})
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

def describe_wind_source_selection(settings: Dict[str, Any]) -> Dict[str, str]:
    """Return dataset names selected by current WIND cadence settings."""
    mag_res = float(settings["MAG_resol"])
    part_res = float(settings["part_resol"])

    if mag_res < 3:
        mag_source = "WI_H2_MFI"
    elif mag_res == 3:
        mag_source = "WI_H0_MFI"
    else:
        mag_source = "WI_PLSP_3DP"

    part_source = "WI_PM_3DP" if part_res <= 3 else "WI_PLSP_3DP"
    elec_source = "WI_H5_SWE" if settings.get("Down_electrons", False) else "disabled"

    return {"mag": mag_source, "par": part_source, "elec": elec_source, "coord_out": "RTN(L1-approx)" if bool(settings.get("in_rtn", True)) else "GSE"}


def LoadTimeSeriesWind_electrons(start_time, end_time, settings):
    """Load WIND electron moments from SWE H5 and return Te, Te_core in eV."""
    from astropy import units as u

    t0 = _to_utc_datetime(start_time).to_pydatetime()
    t1 = _to_utc_datetime(end_time).to_pydatetime()

    status, data = cdas.get_data("WI_H5_SWE", ["T_elec", "TcElec"], t0, t1)
    if not status:
        raise RuntimeError("Failed to download WI_H5_SWE electron data.")

    te_ev = u.Quantity(data["T_elec"], u.K).to(u.eV, equivalencies=u.temperature_energy()).value
    te_core_ev = u.Quantity(data["TcElec"], u.K).to(u.eV, equivalencies=u.temperature_energy()).value

    df = pd.DataFrame(index=pd.to_datetime(data["Epoch"]), data={"Te": te_ev, "Te_core": te_core_ev})
    return _clean_fill_values(df, ("Te", "Te_core"))


def LoadTimeSeriesWind_particles(start_time, end_time, settings):
    """Load WIND proton moments; source depends on requested ``part_resol``."""
    part_resol = float(settings["part_resol"])

    if part_resol <= 3:
        t0 = _to_utc_datetime(start_time).to_pydatetime().replace(tzinfo=pytz.UTC)
        t1 = _to_utc_datetime(end_time).to_pydatetime().replace(tzinfo=pytz.UTC)

        status, data = cdas.get_data("WI_PM_3DP", ["P_DENS", "P_VELS", "P_TEMP", "TIME"], t0, t1)
        if not status:
            raise RuntimeError("Failed to download WI_PM_3DP particle data.")

        vec = _ensure_nx3(data["P_VELS"], n_expected=len(data["Epoch"]))
        if bool(settings.get("in_rtn", True)):
            vec = _approx_gse_to_l1_rtn(vec)
            frame = "RTN"
        else:
            frame = "GSE"

        dfpar = pd.DataFrame(
            index=pd.to_datetime(data["Epoch"]),
            data={
                "c1": vec[:, 0],
                "c2": vec[:, 1],
                "c3": vec[:, 2],
                "np": data["P_DENS"],
                "Tp": data["P_TEMP"],
            },
        )
        dfpar = _rename_vec_by_frame(dfpar, "RTN" if frame == "RTN" else "GSE", "V")
        dfpar["Tp_K"] = _EV_TO_K * dfpar["Tp"]
        dfpar["Vth"] = 0.128487 * np.sqrt(dfpar["Tp_K"].clip(lower=0))
        qtn_flag = "No_QTN"
    else:
        status, data = cdas.get_data(
            "WI_PLSP_3DP",
            ["MOM.P.DENSITY", "MOM.P.VELOCITY", "MOM.P.VTHERMAL", "TIME"],
            str(pd.Timestamp(start_time)),
            str(pd.Timestamp(end_time)),
        )
        if not status:
            raise RuntimeError("Failed to download WI_PLSP_3DP particle data.")

        vel = _pick_first_key(data, "MOM$P$VELOCITY", "MOM.P.VELOCITY")
        dens = _pick_first_key(data, "MOM$P$DENSITY", "MOM.P.DENSITY")
        vth = _pick_first_key(data, "MOM$P$VTHERMAL", "MOM.P.VTHERMAL")

        vec = _ensure_nx3(vel, n_expected=len(data["Epoch"]))
        if bool(settings.get("in_rtn", True)):
            vec = _approx_gse_to_l1_rtn(vec)
            frame = "RTN"
        else:
            frame = "GSE"

        dfpar = pd.DataFrame(
            index=pd.to_datetime(data["Epoch"]),
            data={
                "c1": vec[:, 0],
                "c2": vec[:, 1],
                "c3": vec[:, 2],
                "np": dens,
                "Vth": vth,
            },
        )
        dfpar = _rename_vec_by_frame(dfpar, "RTN" if frame == "RTN" else "GSE", "V")

        # Legacy formula used in this pipeline branch: kT = (m_p * v_th^2) / 2
        vth_mps = np.asarray(dfpar["Vth"], dtype=float) * 1e3
        tp_k = (constants.m_p * (vth_mps**2)) / (2 * constants.k)
        dfpar["Tp"] = tp_k / _EV_TO_K
        qtn_flag = "No_QTN"

    vec_cols = ("Vr", "Vt", "Vn") if bool(settings.get("in_rtn", True)) else ("Vx", "Vy", "Vz")
    dfpar = _clean_fill_values(dfpar, vec_cols + ("np", "Tp", "Vth"))
    dfdis = _build_wind_distance_df(dfpar.index)
    return dfpar, dfdis, qtn_flag


def LoadHighResMagWind(start_time, end_time, settings, verbose=True):
    """Load WIND magnetic field; source depends on requested ``MAG_resol``."""
    mag_resol = float(settings["MAG_resol"])
    t0 = _to_utc_datetime(start_time).to_pydatetime().replace(tzinfo=pytz.UTC)
    t1 = _to_utc_datetime(end_time).to_pydatetime().replace(tzinfo=pytz.UTC)

    if mag_resol == 3:
        status, data = cdas.get_data("WI_H0_MFI", ["B3GSE", "B3F1"], t0, t1)
        if not status:
            raise RuntimeError("Failed to download WI_H0_MFI magnetic data.")

        vec = _ensure_nx3(data["B3GSE"], n_expected=len(data["Epoch3"]))
        if bool(settings.get("in_rtn", True)):
            vec = _approx_gse_to_l1_rtn(vec)
            frame = "RTN"
        else:
            frame = "GSE"

        dfmag = pd.DataFrame(
            index=pd.to_datetime(data["Epoch3"]),
            data={
                "c1": vec[:, 0],
                "c2": vec[:, 1],
                "c3": vec[:, 2],
                "Btot": data["B3F1"],
            },
        )
        dfmag = _rename_vec_by_frame(dfmag, "RTN" if frame == "RTN" else "GSE", "B")
    elif mag_resol < 3:
        status, data = cdas.get_data("WI_H2_MFI", ["BGSE", "BF1"], t0, t1)
        if not status:
            raise RuntimeError("Failed to download WI_H2_MFI magnetic data.")

        vec = _ensure_nx3(data["BGSE"], n_expected=len(data["Epoch"]))
        if bool(settings.get("in_rtn", True)):
            vec = _approx_gse_to_l1_rtn(vec)
            frame = "RTN"
        else:
            frame = "GSE"

        dfmag = pd.DataFrame(
            index=pd.to_datetime(data["Epoch"]),
            data={
                "c1": vec[:, 0],
                "c2": vec[:, 1],
                "c3": vec[:, 2],
                "Btot": data["BF1"],
            },
        )
        dfmag = _rename_vec_by_frame(dfmag, "RTN" if frame == "RTN" else "GSE", "B")
    else:
        status, data = cdas.get_data("WI_PLSP_3DP", ["MOM.P.MAGF"], str(start_time), str(end_time))
        if not status:
            raise RuntimeError("Failed to download WI_PLSP_3DP magnetic fallback data.")

        magf = _pick_first_key(data, "MOM$P$MAGF", "MOM.P.MAGF")

        vec = _ensure_nx3(magf, n_expected=len(data["Epoch"]))
        if bool(settings.get("in_rtn", True)):
            vec = _approx_gse_to_l1_rtn(vec)
            frame = "RTN"
        else:
            frame = "GSE"

        dfmag = pd.DataFrame(
            index=pd.to_datetime(data["Epoch"]),
            data={
                "c1": vec[:, 0],
                "c2": vec[:, 1],
                "c3": vec[:, 2],
            },
        ).interpolate()
        dfmag = _rename_vec_by_frame(dfmag, "RTN" if frame == "RTN" else "GSE", "B")

        bcols_tmp = ["Br", "Bt", "Bn"] if bool(settings.get("in_rtn", True)) else ["Bx", "By", "Bz"]
        dfmag["Btot"] = np.sqrt(dfmag[bcols_tmp[0]] ** 2 + dfmag[bcols_tmp[1]] ** 2 + dfmag[bcols_tmp[2]] ** 2)

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
        big_gaps_mag = func.find_big_gaps(dfmag, settings["Big_Gaps"]["Mag_big_gaps"], str(ind1), str(ind2))
        diagnostics_mag = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)
        print("Mag fraction missing", diagnostics_mag["Frac_miss"])
    except Exception:
        traceback.print_exc()
        dfmag = None
        big_gaps_mag = None
        diagnostics_mag = _empty_diag()

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

    # --- Particles ---
    try:
        dfpar_raw, dfdis_raw, qtn_flag = LoadTimeSeriesWind_particles(pd.Timestamp(t0), pd.Timestamp(t1), settings)
        dfpar = _subset_interval(dfpar_raw, ind1, ind2)
        dfdis = _subset_interval(dfdis_raw, ind1, ind2)

        big_gaps_par = func.find_big_gaps(dfpar, settings["Big_Gaps"]["Par_big_gaps"], str(ind1), str(ind2))
        diagnostics_par = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
        print("Par fraction missing", diagnostics_par["Frac_miss"])
    except Exception:
        traceback.print_exc()
        dfpar = None
        dfdis = None
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
