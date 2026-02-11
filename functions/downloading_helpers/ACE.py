# functions/downloading_helpers/ACE.py
from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

__all__ = [
    "LoadTimeSeriesACE",
    "main_function",
]


# ---------------------------------------------------------------------
# Small, robust helpers
# ---------------------------------------------------------------------
def _as_dtindex(x: Any) -> pd.DatetimeIndex:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.datetime64):
        return pd.to_datetime(x)
    # assume unix seconds
    return pd.to_datetime(x, unit="s", utc=True).tz_convert(None)


def _clean_fill(y: Any) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y[np.abs(y) > 1e30] = np.nan
    return y


def _get_xy(get_data_func, varname: str) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    pytplot.get_data(var) may return:
      - (x, y)
      - (x, y, v)
      - an object with attributes x,y
    """
    d = get_data_func(varname)
    if d is None:
        raise ValueError(f"pytplot.get_data returned None for {varname}")

    if isinstance(d, tuple) and len(d) >= 2:
        x, y = d[0], d[1]
        return _as_dtindex(x), _clean_fill(y)

    if hasattr(d, "x") and hasattr(d, "y"):
        return _as_dtindex(d.x), _clean_fill(d.y)

    raise TypeError(f"Unrecognized pytplot.get_data return type for {varname}: {type(d)}")


def _pick_vec3(
    tvars: Sequence[str],
    get_data_func,
    preferred: Sequence[str] = (),
) -> Optional[str]:
    tset = set(tvars)
    for nm in preferred:
        if nm in tset:
            try:
                _, y = _get_xy(get_data_func, nm)
            except Exception:
                continue
            if y.ndim == 2 and y.shape[1] == 3:
                return nm

    for nm in tvars:
        try:
            _, y = _get_xy(get_data_func, nm)
        except Exception:
            continue
        if y.ndim == 2 and y.shape[1] == 3:
            return nm

    return None


def _pick_scalar(
    tvars: Sequence[str],
    get_data_func,
    preferred: Sequence[str] = (),
) -> Optional[str]:
    tset = set(tvars)
    for nm in preferred:
        if nm in tset:
            try:
                _, y = _get_xy(get_data_func, nm)
            except Exception:
                continue
            if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
                return nm

    for nm in tvars:
        try:
            _, y = _get_xy(get_data_func, nm)
        except Exception:
            continue
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            return nm

    return None


def _approx_gse_to_l1_rtn(vec_gse: np.ndarray) -> np.ndarray:
    """
    L1 approx: R ~ -X_GSE, T ~ Y_GSE, N ~ Z_GSE.
    """
    x = vec_gse[:, 0]
    y = vec_gse[:, 1]
    z = vec_gse[:, 2]
    return np.column_stack([-x, y, z])


def _clip_df(df: Optional[pd.DataFrame], t0: pd.Timestamp, t1: pd.Timestamp) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    return df.loc[(df.index >= t0) & (df.index <= t1)].copy()


def _safe_ms(settings: Dict[str, Any], key: str, default_ms: int, floor_ms: int) -> int:
    """
    Prevent nonsense like resampling ACE to 1 ms.
    """
    try:
        v = int(settings.get(key, default_ms))
    except Exception:
        v = default_ms
    if v < floor_ms:
        v = default_ms
    return v




def _ensure_tp_eV(series: pd.Series, source_hint: str = "") -> pd.Series:
    """Normalize proton temperature to eV.

    Strategy:
    1) Use source hints when we know the unit convention.
    2) Fallback to magnitude heuristic only when unit is ambiguous.
    """
    out = series.astype(float).copy()
    finite = out[np.isfinite(out)]
    if len(finite) == 0:
        return out

    hint = str(source_hint).upper()
    k_B_ev_per_K = constants.physical_constants["Boltzmann constant in eV/K"][0]

    # Explicit unit hints
    if "EV" in hint:
        return out
    if "K" in hint or "KELVIN" in hint or "OMNI" in hint:
        return out * k_B_ev_per_K

    # Ambiguous temperature channels: infer from realistic scale.
    med = float(np.nanmedian(finite))
    if med > 1e3:
        out = out * k_B_ev_per_K

    return out


def _normalize_tp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the pipeline contract: temperature must be in column 'Tp'.
    - If TEMP exists -> rename to Tp
    - If multiple temperature-like columns exist -> prefer Tp, else TEMP, else Tpr/Tp/TEMP variants
    """
    if df is None or df.empty:
        return df

    cols = list(df.columns)
    if "Tp" in cols:
        return df

    if "TEMP" in cols:
        return df.rename(columns={"TEMP": "Tp"})

    # common variants
    for cand in ["Tpr", "tp", "temp", "temperature", "T_p", "Tproton", "proton_temperature"]:
        for c in cols:
            if c == cand:
                return df.rename(columns={c: "Tp"})
            if c.lower() == cand.lower():
                return df.rename(columns={c: "Tp"})

    return df


# ---------------------------------------------------------------------
# OMNI fallback (for recent years when ACE/SWEPAM vec3 is unavailable)
# ---------------------------------------------------------------------
def _load_omni_plasma_cdasws(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    in_rtn: bool,
    settings: Dict[str, Any],
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Pull near-Earth plasma from OMNI via CDAWeb (cdasws), but do not hardcode variable names.
    We:
      1) query variable metadata
      2) select candidates by dimension/description patterns
      3) download only selected variables
    """
    meta: Dict[str, Any] = {"source": "OMNI (CDAWeb)", "dataset": None, "picked": {}}

    try:
        from cdasws import CdasWs  # type: ignore
    except Exception as e:
        raise RuntimeError(f"cdasws import failed (needed for OMNI fallback): {e}")

    cdas = CdasWs()

    # Try common OMNI high-res dataset names (string choices are historically used in CDAWeb)
    dataset_candidates = [
        "OMNI_HRO_1MIN",
        "OMNI_HRO2_1MIN",
        "OMNI2_HRO_1MIN",
        "OMNI2_HRO2_1MIN",
    ]

    varinfo = None
    dataset = None
    last_err = None
    for ds in dataset_candidates:
        try:
            varinfo = cdas.get_variables(ds)
            dataset = ds
            break
        except Exception as e:
            last_err = e
            continue

    if dataset is None or varinfo is None:
        raise RuntimeError(f"Could not find an OMNI high-res dataset via CDAWeb. Last error: {last_err}")

    meta["dataset"] = dataset

    # varinfo is typically a dict keyed by variable name with metadata
    # Normalize into a list of (name, mdict)
    if isinstance(varinfo, dict) and "Variables" in varinfo:
        # some cdasws versions wrap results
        variables = varinfo.get("Variables", [])
        # variables may be list of dicts with 'Name'
        vmap = {}
        for v in variables:
            if isinstance(v, dict) and "Name" in v:
                vmap[v["Name"]] = v
        varinfo_map = vmap
    elif isinstance(varinfo, dict):
        varinfo_map = varinfo
    else:
        raise RuntimeError(f"Unexpected cdasws.get_variables return type: {type(varinfo)}")

    def _md(name: str) -> Dict[str, Any]:
        v = varinfo_map.get(name, {})
        return v if isinstance(v, dict) else {}

    # Heuristics:
    # - pick a vec3 velocity var if exists (DimSizes includes 3, or depends on a 3-vector)
    # - else pick 3 scalars for components
    # - pick density scalar
    # - pick proton temperature scalar
    vel_vec_candidates: List[str] = []
    dens_candidates: List[str] = []
    tp_candidates: List[str] = []
    scalar_candidates: List[str] = []

    for name, md in varinfo_map.items():
        if not isinstance(md, dict):
            continue
        desc = str(md.get("Description", "") or md.get("description", "")).lower()
        units = str(md.get("Units", "") or md.get("units", "")).lower()
        dims = md.get("DimSizes", md.get("dim_sizes", md.get("Dims", None)))

        if "velocity" in desc or "flow speed" in desc or re.search(r"\bv[xyz]\b", name.lower()):
            # try to identify vectors vs scalars by dims
            if isinstance(dims, (list, tuple)) and 3 in [int(x) for x in dims if str(x).isdigit()]:
                vel_vec_candidates.append(name)
            else:
                scalar_candidates.append(name)

        if "density" in desc and ("proton" in desc or "plasma" in desc or "ion" in desc):
            dens_candidates.append(name)
        elif name.lower() in {"np", "n_p", "proton_density"}:
            dens_candidates.append(name)

        if "temperature" in desc and ("proton" in desc or "plasma" in desc or "ion" in desc):
            tp_candidates.append(name)
        elif name.lower() in {"tp", "t_p", "tproton", "proton_temperature"}:
            tp_candidates.append(name)

    # Prefer a vec3 velocity variable; otherwise look for Vx/Vy/Vz scalars by name.
    vel_vec = None
    if vel_vec_candidates:
        # prefer names that clearly indicate GSE
        vel_vec_candidates_sorted = sorted(
            vel_vec_candidates,
            key=lambda s: (("gse" not in s.lower()), ("vec" not in s.lower()), len(s)),
        )
        vel_vec = vel_vec_candidates_sorted[0]

    def _pick_by_name_regex(names: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
        for pat in patterns:
            r = re.compile(pat, flags=re.IGNORECASE)
            for n in names:
                if r.search(n):
                    return n
        return None

    vx = _pick_by_name_regex(varinfo_map.keys(), [r"\bvx\b", r"vx_gse", r"v_gse_x", r"flow_vx"])
    vy = _pick_by_name_regex(varinfo_map.keys(), [r"\bvy\b", r"vy_gse", r"v_gse_y", r"flow_vy"])
    vz = _pick_by_name_regex(varinfo_map.keys(), [r"\bvz\b", r"vz_gse", r"v_gse_z", r"flow_vz"])

    dens = None
    if dens_candidates:
        dens = sorted(dens_candidates, key=lambda s: (("proton" not in _md(s).get("Description", "").lower()), len(s)))[0]

    tp = None
    if tp_candidates:
        tp = sorted(tp_candidates, key=lambda s: (("proton" not in _md(s).get("Description", "").lower()), len(s)))[0]

    # Build request var list
    req_vars: List[str] = []
    if vel_vec is not None:
        req_vars.append(vel_vec)
        meta["picked"]["vel_vec"] = vel_vec
    elif vx and vy and vz:
        req_vars.extend([vx, vy, vz])
        meta["picked"]["vx"] = vx
        meta["picked"]["vy"] = vy
        meta["picked"]["vz"] = vz
    else:
        raise RuntimeError("OMNI fallback: could not identify a velocity vector (vec3 or components).")

    if dens is not None:
        req_vars.append(dens)
        meta["picked"]["dens"] = dens
    else:
        raise RuntimeError("OMNI fallback: could not identify a proton/plasma density variable.")

    if tp is not None:
        req_vars.append(tp)
        meta["picked"]["tp"] = tp
    else:
        # not fatal for all pipelines, but your calc_diagnostics wants Tp
        raise RuntimeError("OMNI fallback: could not identify a proton/plasma temperature variable.")

    # Query data
    t0s = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    t1s = t1.strftime("%Y-%m-%dT%H:%M:%SZ")
    status, data = cdas.get_data(dataset, req_vars, t0s, t1s)
    if not status:
        raise RuntimeError("OMNI fallback: CDAWeb request failed.")

    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError("OMNI fallback: CDAWeb returned no data (empty payload).")

    # Find the time axis key
    time_key = None
    for k in data.keys():
        if str(k).lower() in {"epoch", "time", "datetime"}:
            time_key = k
            break
    if time_key is None:
        # common cdasws uses 'Epoch'
        for k in data.keys():
            if "epoch" in str(k).lower():
                time_key = k
                break
    if time_key is None:
        raise RuntimeError(f"OMNI fallback: could not identify time key in CDAWeb payload keys={list(data.keys())}")

    idx = pd.to_datetime(np.asarray(data[time_key]))

    # Velocity
    if vel_vec is not None:
        V = np.asarray(data[vel_vec], dtype=float)
        if V.ndim != 2 or V.shape[1] != 3:
            raise RuntimeError(f"OMNI fallback: velocity vec3 has unexpected shape {V.shape}")
        V_gse = V
    else:
        Vx = np.asarray(data[vx], dtype=float).ravel()
        Vy = np.asarray(data[vy], dtype=float).ravel()
        Vz = np.asarray(data[vz], dtype=float).ravel()
        V_gse = np.column_stack([Vx, Vy, Vz])

    # Density, temperature
    np_ = np.asarray(data[dens], dtype=float).ravel()
    Tp_ = np.asarray(data[tp], dtype=float).ravel()

    # Build DF in requested frame
    Tp_eV = _ensure_tp_eV(pd.Series(Tp_, index=idx), source_hint="OMNI")

    if in_rtn:
        V_out = _approx_gse_to_l1_rtn(V_gse)
        df = pd.DataFrame({"Vr": V_out[:, 0], "Vt": V_out[:, 1], "Vn": V_out[:, 2], "np": np_, "Tp": Tp_eV.values}, index=idx)
        meta["coord_in"] = "GSE"
        meta["coord_out"] = "RTN(L1-approx)"
    else:
        df = pd.DataFrame({"Vx": V_gse[:, 0], "Vy": V_gse[:, 1], "Vz": V_gse[:, 2], "np": np_, "Tp": Tp_eV.values}, index=idx)
        meta["coord_in"] = "GSE"
        meta["coord_out"] = "GSE"

    df = df.sort_index()
    df = _clip_df(df, t0, t1)

    return df, meta


# ---------------------------------------------------------------------
# Public API expected by the pipeline
# ---------------------------------------------------------------------
def LoadTimeSeriesACE(
    start_time,
    end_time,
    settings: Dict[str, Any],
    vars_2_downnload: Optional[Dict[str, Any]] = None,
    time_amount: int = 1,
    time_unit: str = "h",
):
    """
    ACE loader.

    MAG: ACE/MFI via pyspedas (BGSEc) -> (Bx,By,Bz) or (Br,Bt,Bn) (L1 approx RTN).
    PAR: Try ACE/SWEPAM via pyspedas (h0/h2). If not available (recent years), fallback to OMNI via CDAWeb (cdasws).

    Returns:
      dfmag, dfpar, dfdis, big_gaps, diagnostics
    """
    import general_functions as func

    from pytplot import get_data  # type: ignore
    import pyspedas  # type: ignore

    # ---------------------------
    # time handling
    # ---------------------------
    t0_req, t1_req = func.ensure_time_format(start_time, end_time)
    t0_pad = func.add_time_to_datetime_string(t0_req, -time_amount, time_unit)
    t1_pad = func.add_time_to_datetime_string(t1_req, +time_amount, time_unit)

    t0_req_dt = pd.to_datetime(t0_req)
    t1_req_dt = pd.to_datetime(t1_req)

    in_rtn = bool(settings.get("in_rtn", False))
    ace_frame = str(settings.get("ace_frame", "RTN" if in_rtn else "GSE")).upper()
    if ace_frame not in {"GSE", "RTN"}:
        ace_frame = "RTN" if in_rtn else "GSE"
    in_rtn = (ace_frame == "RTN")

    # ---------------------------
    # cache directory
    # ---------------------------
    data_root = settings.get("Data_path", ".")
    ace_cache = os.path.join(data_root, "ace_data")
    os.makedirs(ace_cache, exist_ok=True)
    os.environ["SPEDAS_DATA_DIR"] = ace_cache

    vars_2_downnload = vars_2_downnload or {}

    mag_req = vars_2_downnload.get("mag") or {}
    mfi_datatype = str(mag_req.get("datatype", "h0"))

    par_req = vars_2_downnload.get("par") or vars_2_downnload.get("swe") or {}
    swe_datatype = str(par_req.get("datatype", "h0"))

    # ---------------------------
    # Load MAG (ACE/MFI)
    # ---------------------------
    dfmag: Optional[pd.DataFrame] = None
    mag_meta: Dict[str, Any] = {"mfi_datatype": mfi_datatype, "mag_var": None, "coord_in": "GSE"}

    try:
        tvars_mfi = pyspedas.projects.ace.mfi(
            trange=[t0_pad, t1_pad],
            datatype=mfi_datatype,
            varnames=["BGSEc"],
            time_clip=True,
        ) or []

        mag_var = _pick_vec3(tvars_mfi, get_data, preferred=("BGSEc",))
        if mag_var is None:
            tvars_mfi = pyspedas.projects.ace.mfi(
                trange=[t0_pad, t1_pad],
                datatype=mfi_datatype,
                time_clip=True,
            ) or []
            mag_var = _pick_vec3(tvars_mfi, get_data, preferred=("BGSEc",))

        if mag_var is None:
            raise ValueError("ACE/MFI: no vec3 MAG variable found (expected BGSEc).")

        idx, B_gse = _get_xy(get_data, mag_var)
        if B_gse.ndim != 2 or B_gse.shape[1] != 3:
            raise ValueError(f"ACE/MFI: {mag_var} not vec3 (shape={B_gse.shape}).")

        if in_rtn:
            B_out = _approx_gse_to_l1_rtn(B_gse)
            dfmag = pd.DataFrame(B_out, index=idx, columns=["Br", "Bt", "Bn"])
        else:
            dfmag = pd.DataFrame(B_gse, index=idx, columns=["Bx", "By", "Bz"])

        dfmag = dfmag.sort_index()
        dfmag = _clip_df(dfmag, t0_req_dt, t1_req_dt)
        mag_meta["mag_var"] = mag_var

    except Exception as e:
        warnings.warn(f"ACE/MFI load failed: {e}")
        dfmag = None

    # ---------------------------
    # Load PAR (ACE/SWEPAM via pyspedas; fallback OMNI)
    # ---------------------------
    dfpar: Optional[pd.DataFrame] = None
    par_meta: Dict[str, Any] = {
        "attempted": [],
        "swe_datatype_requested": swe_datatype,
        "source": "ACE/SWEPAM",
        "coord_out": "RTN(L1-approx)" if in_rtn else "GSE",
        "vel_var": None,
        "dens_var": None,
        "temp_var": None,
    }

    def _try_load_swe(datatype_try: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        meta = dict(par_meta)
        meta["attempted"] = meta.get("attempted", []) + [datatype_try]
        meta["swe_datatype_used"] = datatype_try

        tvars_swe = pyspedas.projects.ace.swe(
            trange=[t0_pad, t1_pad],
            datatype=datatype_try,
            time_clip=True,
        ) or []

        if len(tvars_swe) == 0:
            raise ValueError("ACE/SWE: no tplot variables returned.")

        # Prefer native RTN if present, else GSE; older h0 typically provides vec3 in GSE.
        vel_pref_rtn = ("V_RTN", "VRTN", "Vrtn", "V_rtn")
        vel_pref_gse = ("V_GSE", "VGSE", "Vgse", "V_gse", "V")
        if in_rtn:
            vel_var = _pick_vec3(tvars_swe, get_data, preferred=vel_pref_rtn + vel_pref_gse)
        else:
            vel_var = _pick_vec3(tvars_swe, get_data, preferred=vel_pref_gse + vel_pref_rtn)

        if vel_var is None:
            raise ValueError("ACE/SWE: could not find any vec3 velocity variable (this is common for k0/k1).")

        vel_idx, V_vec = _get_xy(get_data, vel_var)
        if V_vec.ndim != 2 or V_vec.shape[1] != 3:
            raise ValueError(f"ACE/SWE: {vel_var} not vec3 (shape={V_vec.shape}).")

        dens_var = _pick_scalar(tvars_swe, get_data, preferred=("Np", "np", "DENS", "density", "proton_density"))
        if dens_var is None:
            raise ValueError("ACE/SWE: could not find a density scalar.")

        dens_idx, np_ = _get_xy(get_data, dens_var)
        np_ = np_.ravel()

        temp_var = _pick_scalar(tvars_swe, get_data, preferred=("Tpr", "Tp", "TEMP", "temperature", "T_p"))
        if temp_var is None:
            raise ValueError("ACE/SWE: could not find a temperature scalar (needed as Tp).")

        temp_idx, Tp_ = _get_xy(get_data, temp_var)
        Tp_ = Tp_.ravel()

        # Build velocity DF in its native frame first
        dfV = pd.DataFrame(V_vec, index=vel_idx, columns=["V1", "V2", "V3"]).sort_index()

        vel_name = vel_var.upper()
        coord_in = "UNKNOWN"
        if "RTN" in vel_name:
            coord_in = "RTN"
        elif "GSE" in vel_name:
            coord_in = "GSE"
        elif "GSM" in vel_name:
            coord_in = "GSM"
        meta["coord_in"] = coord_in

        if in_rtn:
            if coord_in == "RTN":
                dfV.columns = ["Vr", "Vt", "Vn"]
            elif coord_in == "GSE":
                V_rtn = _approx_gse_to_l1_rtn(dfV.to_numpy())
                dfV = pd.DataFrame(V_rtn, index=dfV.index, columns=["Vr", "Vt", "Vn"])
            else:
                dfV.columns = ["Vr", "Vt", "Vn"]
        else:
            if coord_in == "GSE":
                dfV.columns = ["Vx", "Vy", "Vz"]
            else:
                dfV.columns = ["Vx", "Vy", "Vz"]

        dfn = pd.DataFrame({"np": np_}, index=dens_idx).sort_index()
        dfT = pd.DataFrame({"Tp": _ensure_tp_eV(pd.Series(Tp_, index=temp_idx), source_hint=temp_var).values}, index=temp_idx).sort_index()

        tol_ms = _safe_ms(settings, "part_resol", default_ms=64000, floor_ms=1000)
        tol = pd.Timedelta(milliseconds=tol_ms)

        df = pd.merge_asof(
            dfV.reset_index().rename(columns={"index": "time"}),
            dfn.reset_index().rename(columns={"index": "time"}),
            on="time",
            direction="nearest",
            tolerance=tol,
        ).set_index("time")

        df = pd.merge_asof(
            df.reset_index().rename(columns={"time": "time"}),
            dfT.reset_index().rename(columns={"index": "time"}),
            on="time",
            direction="nearest",
            tolerance=tol,
        ).set_index("time")

        df = df.sort_index()
        df = _clip_df(df, t0_req_dt, t1_req_dt)

        meta["vel_var"] = vel_var
        meta["dens_var"] = dens_var
        meta["temp_var"] = temp_var

        return df, meta

    # Try SWEPAM h0/h2 first (works for older periods)
    try:
        try:
            dfpar, par_meta = _try_load_swe(swe_datatype)
        except Exception:
            if swe_datatype.lower() != "h2":
                dfpar, par_meta = _try_load_swe("h2")
            else:
                raise
        dfpar = _normalize_tp_column(dfpar)
        if "Tp" in dfpar.columns:
            dfpar["Tp"] = _ensure_tp_eV(dfpar["Tp"], source_hint=par_meta.get("temp_var", ""))
    except Exception as e_swe:
        # Fallback to OMNI for recent years where SWEPAM vec3 isn't available
        try:
            dfpar, omni_meta = _load_omni_plasma_cdasws(t0_req_dt, t1_req_dt, in_rtn, settings)
            dfpar = _normalize_tp_column(dfpar)
            if "Tp" in dfpar.columns:
                dfpar["Tp"] = _ensure_tp_eV(dfpar["Tp"], source_hint="OMNI")
            par_meta["source"] = "OMNI (CDAWeb) fallback"
            par_meta["omni"] = omni_meta
        except Exception as e_omni:
            warnings.warn(f"ACE/SWEPAM load failed: {e_swe}. OMNI fallback failed: {e_omni}")
            dfpar = None

    # ---------------------------
    # Resample + gap diagnostics (pipeline contract)
    # ---------------------------
    diagnostics: Dict[str, Any] = {}

    mag_resol_ms = _safe_ms(settings, "MAG_resol", default_ms=16000, floor_ms=1000)
    par_resol_ms = _safe_ms(settings, "part_resol", default_ms=64000, floor_ms=1000)

    def _resample_and_diag(df: Optional[pd.DataFrame], resol_ms: int, large_gaps: int) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"resampled_df": None, "Frac_miss": 100.0, "large_gaps": []}

        if hasattr(func, "resample_timeseries_estimate_gaps"):
            out = func.resample_timeseries_estimate_gaps(df, resol_ms, large_gaps=large_gaps)
            if isinstance(out, dict):
                out.setdefault("resampled_df", out.get("resampled_df", df))
                out.setdefault("Frac_miss", out.get("Frac_miss", np.nan))
                out.setdefault("large_gaps", out.get("large_gaps", []))
            return out

        rule = f"{int(resol_ms)}ms"
        dfr = df.resample(rule).median()
        frac = float(np.mean(~np.isfinite(dfr.to_numpy())) * 100.0)
        return {"resampled_df": dfr, "Frac_miss": frac, "large_gaps": []}

    diag_mag = _resample_and_diag(
        dfmag,
        mag_resol_ms,
        large_gaps=int(settings.get("Big_Gaps", {}).get("Mag_big_gaps", 10)),
    )
    if diag_mag.get("resampled_df") is not None:
        dfmag = diag_mag["resampled_df"]

    diag_par = _resample_and_diag(
        dfpar,
        par_resol_ms,
        large_gaps=int(settings.get("Big_Gaps", {}).get("Par_big_gaps", 10)),
    )
    if diag_par.get("resampled_df") is not None:
        dfpar = diag_par["resampled_df"]
        if dfpar is not None:
            dfpar = _normalize_tp_column(dfpar)
        if "Tp" in dfpar.columns:
            dfpar["Tp"] = _ensure_tp_eV(dfpar["Tp"], source_hint=par_meta.get("temp_var", ""))

    diagnostics["Mag"] = diag_mag
    diagnostics["Par"] = diag_par

    # ---------------------------
    # Align intersection after resampling
    # ---------------------------
    if dfmag is not None and not dfmag.empty and dfpar is not None and not dfpar.empty:
        tmin = max(dfmag.index.min(), dfpar.index.min())
        tmax = min(dfmag.index.max(), dfpar.index.max())
        dfmag = dfmag.loc[tmin:tmax]
        dfpar = dfpar.loc[tmin:tmax]

    # ---------------------------
    # Big gaps (optional)
    # ---------------------------
    big_gaps: Dict[str, Any] = {}
    try:
        bg_cfg = settings.get("Big_Gaps", {})
        if dfmag is not None and not dfmag.empty and hasattr(func, "find_big_gaps"):
            big_gaps["MAG"] = func.find_big_gaps(dfmag, bg_cfg.get("Mag_big_gaps", 10), str(t0_req_dt), str(t1_req_dt))
        else:
            big_gaps["MAG"] = []

        if dfpar is not None and not dfpar.empty and hasattr(func, "find_big_gaps"):
            big_gaps["PAR"] = func.find_big_gaps(dfpar, bg_cfg.get("Par_big_gaps", 10), str(t0_req_dt), str(t1_req_dt))
        else:
            big_gaps["PAR"] = []
    except Exception:
        big_gaps.setdefault("MAG", [])
        big_gaps.setdefault("PAR", [])

    # ---------------------------
    # Minimal distance DF
    # ---------------------------
    dfdis: Optional[pd.DataFrame] = None
    try:
        if dfmag is not None and not dfmag.empty:
            dfdis = pd.DataFrame({"Dist_au": np.full(len(dfmag), 1.0)}, index=dfmag.index)
        elif dfpar is not None and not dfpar.empty:
            dfdis = pd.DataFrame({"Dist_au": np.full(len(dfpar), 1.0)}, index=dfpar.index)
    except Exception:
        dfdis = None

    # ---------------------------
    # Metadata
    # ---------------------------
    diagnostics["meta"] = {
        "coord_frame_out": "RTN(L1-approx)" if in_rtn else "GSE",
        "MAG": mag_meta,
        "PAR": par_meta,
        "SPEDAS_DATA_DIR": ace_cache,
    }

    return dfmag, dfpar, dfdis, big_gaps, diagnostics


def main_function(
    start_time,
    end_time,
    settings: Dict[str, Any],
    vars_2_downnload: Optional[Dict[str, Any]] = None,
    time_amount: int = 1,
    time_unit: str = "h",
):
    return LoadTimeSeriesACE(
        start_time=start_time,
        end_time=end_time,
        settings=settings,
        vars_2_downnload=vars_2_downnload,
        time_amount=time_amount,
        time_unit=time_unit,
    )
