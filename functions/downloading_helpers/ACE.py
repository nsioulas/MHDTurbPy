from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- constants
K_PER_eV = 11604.518121550082      # Kelvin per eV
eV_PER_K = 1.0 / K_PER_eV
_VTH_COEFF = 13.84112218           # km/s * sqrt(eV) = sqrt(2 e / m_p)/1000

# ------------------------------------------------------------
# Local SPEDAS import (repo-local pattern)
# ------------------------------------------------------------
sys.path.insert(0, os.path.join(os.getcwd(), "pyspedas"))
import pyspedas  # type: ignore
from pytplot import get_data  # type: ignore


# ============================================================
# 0) Small utilities
# ============================================================
def _set_spedas_data_dir(settings: Dict[str, Any]) -> None:
    data_path = settings.get("Data_path", None)
    if data_path:
        os.environ["SPEDAS_DATA_DIR"] = str(Path(data_path).expanduser().resolve())


def _ensure_dt_index(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out.index = out.index.tz_localize(None)
    out = out.loc[~out.index.duplicated(keep="first")]
    out.index.name = "datetime"
    return out if len(out) else None


def _normalize_resolution_seconds(x: Any, default_s: float) -> float:
    """Normalize cadence input to seconds (accepting legacy ms values)."""
    try:
        v = float(x)
    except Exception:
        return float(default_s)
    return (v / 1000.0) if v > 1000.0 else v


def _get_times_y_attrs(varname: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    obj = get_data(varname)
    if obj is None:
        return None, None, {}
    if isinstance(obj, tuple) and len(obj) >= 2:
        return np.asarray(obj[0]), np.asarray(obj[1]), {}
    times = getattr(obj, "times", None)
    y = getattr(obj, "y", None)
    attrs = getattr(obj, "attrs", {}) or {}
    if times is None or y is None:
        return None, None, dict(attrs) if isinstance(attrs, dict) else {}
    return np.asarray(times), np.asarray(y), dict(attrs) if isinstance(attrs, dict) else {}


def _times_to_datetime(times: np.ndarray) -> pd.DatetimeIndex:
    if np.issubdtype(times.dtype, np.number):
        idx = pd.to_datetime(times, unit="s", errors="coerce")
    else:
        idx = pd.to_datetime(times, errors="coerce")
    return pd.DatetimeIndex(idx).tz_localize(None)


def _tplot_to_df(varname: str, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    times, y, attrs = _get_times_y_attrs(varname)
    if times is None or y is None:
        return pd.DataFrame(), attrs

    idx = _times_to_datetime(times)
    y = np.asarray(y)

    if y.ndim == 1:
        col0 = columns[0] if columns else "col0"
        df = pd.DataFrame(index=idx, data={col0: y})
    else:
        ncol = y.shape[1]
        cols = columns[:ncol] if (columns and len(columns) >= ncol) else [f"col_{i}" for i in range(ncol)]
        df = pd.DataFrame(index=idx, data=y[:, : len(cols)], columns=cols)

    out = _ensure_dt_index(df)
    return (out if out is not None else pd.DataFrame()), attrs


def _unit_str(attrs: Dict[str, Any]) -> str:
    u = attrs.get("units", "") if isinstance(attrs, dict) else ""
    return str(u).strip().lower()


def _pick_by_suffix(created: List[str], suffix: str) -> Optional[str]:
    hits = [v for v in created if v.endswith(suffix)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return sorted(hits, key=len)[0]
    return None


def _sanitize_numeric(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    df = _ensure_dt_index(df)
    if df is None:
        return None
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        x = out[c].to_numpy(dtype=float, copy=False)
        bad = ~np.isfinite(x) | (np.abs(x) > 1.0e30)
        if np.any(bad):
            out.loc[bad, c] = np.nan
    return _ensure_dt_index(out)


def _clean_nonpositive(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float, copy=False)
            out.loc[x <= 0.0, c] = np.nan
    return out


def _tp_to_ev(tp: pd.Series, unit_hint: str = "") -> pd.Series:
    """Convert proton temperature to eV using explicit units first, then fallback heuristics."""
    out = pd.to_numeric(tp, errors="coerce").astype(float)
    hint = (unit_hint or "").lower()

    if "ev" in hint:
        return out

    if ("k" in hint) or ("kelvin" in hint):
        return out * eV_PER_K

    # Ambiguous unit fallback: very large values are almost certainly Kelvin.
    finite = out[np.isfinite(out)]
    if len(finite) and float(np.nanmedian(finite)) > 1.0e3:
        return out * eV_PER_K

    return out


def _build_vth_from_tp_ev(tp_ev: pd.Series) -> pd.Series:
    tp = pd.to_numeric(tp_ev, errors="coerce").astype(float)
    tp[tp <= 0] = np.nan
    # IMPORTANT physics fix: no /sqrt(3) here for standard thermal speed definition used elsewhere.
    return _VTH_COEFF * np.sqrt(tp)


# ============================================================
# 1) MAG: one call to MFI, extract BGSEc -> Bx,By,Bz
# ============================================================
def _load_ace_mag_gse(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    settings: Dict[str, Any],
    vars_2_download: Dict[str, Any],
) -> pd.DataFrame:
    _set_spedas_data_dir(settings)
    no_update = bool(settings.get("use_local_data", False))

    dtype = str(vars_2_download.get("mag", {}).get("datatype", "h0")).lower()
    prefix = f"ace_mfi_{dtype}_"

    created = pyspedas.projects.ace.mfi(
        trange=[str(t0), str(t1)],
        datatype=dtype,
        prefix=prefix,
        varnames=["BGSEc"],
        time_clip=True,
        no_update=no_update,
    )
    created = list(created) if created is not None else []

    bgse = _pick_by_suffix(created, "BGSEc")
    if bgse is None:
        raise RuntimeError("ACE/MFI: BGSEc not returned by pyspedas.")

    dfB, _ = _tplot_to_df(bgse, ["Bx", "By", "Bz"])
    dfB = _sanitize_numeric(dfB)
    if dfB is None or len(dfB) == 0:
        raise RuntimeError("ACE/MFI: BGSEc converted to empty dataframe.")

    return dfB


# ============================================================
# 2) Plasma: one call to SWE (h0 then k0), map into contract
# ============================================================
def _load_ace_plasma_gse_from_swe(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    settings: Dict[str, Any],
    vars_2_download: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    _set_spedas_data_dir(settings)
    no_update = bool(settings.get("use_local_data", False))

    meta: Dict[str, Any] = {}

    for dtype in ("h0", "k0"):
        prefix = f"ace_swe_{dtype}_"
        try:
            created = pyspedas.projects.ace.swe(
                trange=[str(t0), str(t1)],
                datatype=dtype,
                prefix=prefix,
                varnames=[],
                get_support_data=False,
                time_clip=True,
                no_update=no_update,
            )
            created = list(created) if created is not None else []
        except Exception as e:
            meta[f"swe_{dtype}_error"] = str(e)
            continue

        v_gse = _pick_by_suffix(created, "V_GSE")
        vp = _pick_by_suffix(created, "Vp")
        npv = _pick_by_suffix(created, "Np")
        tpr = _pick_by_suffix(created, "Tpr")

        dfs: List[pd.DataFrame] = []
        unit_meta: Dict[str, str] = {}

        if v_gse is not None:
            dfV, v_attrs = _tplot_to_df(v_gse, ["Vx", "Vy", "Vz"])
            dfs.append(dfV)
            unit_meta["V_GSE"] = _unit_str(v_attrs)

        if vp is not None:
            dfVp, vp_attrs = _tplot_to_df(vp, ["Vp"])
            dfs.append(dfVp)
            unit_meta["Vp"] = _unit_str(vp_attrs)

        if npv is not None:
            dfNp, np_attrs = _tplot_to_df(npv, ["np"])
            dfs.append(dfNp)
            unit_meta["Np"] = _unit_str(np_attrs)

        if tpr is not None:
            dfTp, tp_attrs = _tplot_to_df(tpr, ["Tpr"])
            dfs.append(dfTp)
            unit_meta["Tpr"] = _unit_str(tp_attrs)

        if len(dfs) == 0:
            meta[f"swe_{dtype}_note"] = "No usable plasma variables created."
            continue

        df = dfs[0]
        for d in dfs[1:]:
            df = df.join(d, how="outer")

        df = _sanitize_numeric(df)
        if df is None or len(df) == 0:
            meta[f"swe_{dtype}_note"] = "Converted plasma dataframe empty."
            continue

        meta["plasma_source"] = f"SWE_{dtype}"
        meta["units_raw"] = dict(unit_meta)

        # Speed conversions
        if "Vp" in df.columns:
            u = unit_meta.get("Vp", "")
            if "m/s" in u:
                df["Vp"] = df["Vp"] / 1000.0
                meta["Vp_conversion"] = "m/s -> km/s"
            else:
                meta["Vp_conversion"] = "assume km/s"

        if all(c in df.columns for c in ("Vx", "Vy", "Vz")):
            u = unit_meta.get("V_GSE", "")
            if "m/s" in u:
                df[["Vx", "Vy", "Vz"]] = df[["Vx", "Vy", "Vz"]] / 1000.0
                meta["V_GSE_conversion"] = "m/s -> km/s"
            else:
                meta["V_GSE_conversion"] = "assume km/s"

        # Density conversions
        if "np" in df.columns:
            u = unit_meta.get("Np", "")
            if ("m-3" in u) or ("m^(-3)" in u) or ("m^-3" in u):
                df["np"] = df["np"] / 1.0e6
                meta["np_conversion"] = "m^-3 -> cm^-3"
            else:
                meta["np_conversion"] = "assume cm^-3"

        # Temperature conversion to eV
        if "Tpr" in df.columns:
            df["Tp"] = _tp_to_ev(df["Tpr"], unit_hint=unit_meta.get("Tpr", ""))
            src_u = unit_meta.get("Tpr", "")
            meta["Tp_conversion"] = f"Tpr({src_u or 'unknown'}) -> eV"
        else:
            df["Tp"] = np.nan

        # Enforce physical positivity
        df = _clean_nonpositive(df, ["np", "Vp", "Tp"])

        # Prefer measured vec3; if only Vp exists do not fabricate direction.
        if all(c not in df.columns for c in ("Vx", "Vy", "Vz")):
            if "Vp" in df.columns:
                df["Vx"] = np.nan
                df["Vy"] = np.nan
                df["Vz"] = np.nan
                meta["vector_manufactured"] = False
                meta["vector_note"] = "No vec3 available; kept Vx/Vy/Vz as NaN (no isotropy assumption)."
            else:
                df["Vx"] = np.nan
                df["Vy"] = np.nan
                df["Vz"] = np.nan
                df["Vp"] = np.nan

        if "Vp" not in df.columns:
            if all(c in df.columns for c in ("Vx", "Vy", "Vz")):
                df["Vp"] = np.sqrt(df["Vx"] ** 2 + df["Vy"] ** 2 + df["Vz"] ** 2)
            else:
                df["Vp"] = np.nan

        df["TEMP"] = df["Tp"]
        df["Vth"] = _build_vth_from_tp_ev(df["Tp"])
        df["Vth_kms"] = df["Vth"]

        df = _sanitize_numeric(df)
        if df is None or len(df) == 0:
            meta[f"swe_{dtype}_note"] = "Final plasma dataframe empty after cleaning."
            continue

        return df, meta

    raise RuntimeError(f"ACE/SWE: failed for both h0 and k0. Meta: {meta}")


# ============================================================
# 3) Public entry point (contract preserved)
# ============================================================
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
    """Returns (dfmag, dfpar, dfdis, big_gaps, misc) in GSE contract."""
    del cdf_lib_path, credentials  # preserved signature
    import general_functions as func

    t0_req = pd.Timestamp(start_time)
    t1_req = pd.Timestamp(end_time)

    try:
        t0 = pd.Timestamp(func.add_time_to_datetime_string(str(t0_req), -float(time_amount), unit=time_unit))
        t1 = pd.Timestamp(func.add_time_to_datetime_string(str(t1_req), +float(time_amount), unit=time_unit))
    except Exception:
        t0, t1 = t0_req, t1_req

    mag_resol_s = _normalize_resolution_seconds(settings.get("MAG_resol", 16), 16.0)
    par_resol_s = _normalize_resolution_seconds(settings.get("part_resol", 64), 64.0)

    dfmag = _load_ace_mag_gse(t0, t1, settings, vars_2_downnload)

    try:
        dfpar, plasma_meta = _load_ace_plasma_gse_from_swe(t0, t1, settings, vars_2_downnload)
    except Exception as e:
        plasma_meta = {"plasma_source": "SWE_failed", "error": str(e)}
        dfpar = None

    dfpar = _ensure_dt_index(dfpar)
    if dfpar is None or len(dfpar) == 0:
        misc = {
            "ACE": plasma_meta,
            "Mag": {"Frac_miss": 0.0, "resol": mag_resol_s},
            "Par": {"Frac_miss": 100.0, "resol": par_resol_s},
        }
        return dfmag, None, None, None, misc

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

    dfmag = _ensure_dt_index(dfmag)
    dfpar = _ensure_dt_index(dfpar)

    misc: Dict[str, Any] = {"ACE": plasma_meta}
    big_gaps = None
    dfdis = None

    try:
        mag_res = func.resample_timeseries_estimate_gaps(dfmag, mag_resol_s, large_gaps=10)
        dfmag_r = mag_res.get("resampled_df", None)
        misc["Mag"] = {k: mag_res.get(k, None) for k in ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")}
    except TypeError:
        mag_res = func.resample_timeseries_estimate_gaps(dfmag, int(mag_resol_s * 1000), settings.get("gap_time_threshold", 5))
        dfmag_r = mag_res.get("resampled_df", None)
        misc["Mag"] = {k: mag_res.get(k, None) for k in ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")}
    except Exception:
        dfmag_r = dfmag
        misc["Mag"] = {"Frac_miss": 0.0, "Large_gaps": np.nan, "Tot_gaps": np.nan, "resol": mag_resol_s}

    try:
        par_res = func.resample_timeseries_estimate_gaps(dfpar, par_resol_s, large_gaps=10)
        dfpar_r = par_res.get("resampled_df", None)
        misc["Par"] = {k: par_res.get(k, None) for k in ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")}
    except TypeError:
        par_res = func.resample_timeseries_estimate_gaps(dfpar, int(par_resol_s * 1000), settings.get("gap_time_threshold", 5))
        dfpar_r = par_res.get("resampled_df", None)
        misc["Par"] = {k: par_res.get(k, None) for k in ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")}
    except Exception:
        dfpar_r = dfpar
        miss = float(dfpar_r.isna().any(axis=1).mean() * 100.0) if isinstance(dfpar_r, pd.DataFrame) else 100.0
        misc["Par"] = {"Frac_miss": miss, "Large_gaps": np.nan, "Tot_gaps": np.nan, "resol": par_resol_s}

    dfmag_r = _ensure_dt_index(dfmag_r) if isinstance(dfmag_r, pd.DataFrame) else dfmag
    dfpar_r = _ensure_dt_index(dfpar_r) if isinstance(dfpar_r, pd.DataFrame) else dfpar

    out_mag = dfmag_r[["Bx", "By", "Bz"]] if isinstance(dfmag_r, pd.DataFrame) and all(c in dfmag_r.columns for c in ("Bx", "By", "Bz")) else dfmag_r

    out_par = dfpar_r.copy() if isinstance(dfpar_r, pd.DataFrame) else dfpar
    if isinstance(out_par, pd.DataFrame):
        # Recompute robustly after resampling to preserve physics consistency.
        out_par["Tp"] = _tp_to_ev(out_par.get("Tp", pd.Series(index=out_par.index, dtype=float)), unit_hint="eV")
        out_par["TEMP"] = out_par["Tp"]
        out_par["Vth"] = _build_vth_from_tp_ev(out_par["Tp"])
        out_par["Vth_kms"] = out_par["Vth"]

        for c in ("Vx", "Vy", "Vz", "Vp", "np", "Tp", "TEMP", "Vth", "Vth_kms"):
            if c not in out_par.columns:
                out_par[c] = np.nan

    return out_mag, out_par, dfdis, big_gaps, misc
