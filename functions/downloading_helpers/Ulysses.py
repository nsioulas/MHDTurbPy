"""
Ulysses.py

CDAWeb loader helpers for Ulysses (MAG + plasma + position) used by MHDTurbPy.

Non-negotiable constraints:
- Preserve the public function signatures and return contracts:
    LoadTimeSeriesUlysses_particles(start_time, end_time) -> (dfpar, dfdis)
    LoadHighResMagUlysses(start_time, end_time, verbose=True) -> (dfmag, dfmag1, infos)
    LoadMagUlysses(start_time, end_time, verbose=True) -> (dfmag, dfmag1, infos)
    LoadTimeSeriesUlysses(start_time, end_time, settings, gap_time_threshold=10, time_amount=4, time_unit="h")
        -> (dfmag_out, dfpar_out, dfdis_out, big_gaps_out, misc)

This file adds:
- Robust timezone handling (always return tz-naive UTC indices for compatibility with the existing pipeline).
- Duplicate-index handling (CDAWeb can return duplicates across chunk boundaries).
- Optional, explicit debug diagnostics gated by settings["debug_ulysses"].
- Fallback dataset chains (particles / MAG / position) if the primary dataset is missing or returns empty.
- Guaranteed dfpar["Dist_au"] column, populated from CDA position if available, otherwise via direct JPL Horizons API.
"""

from __future__ import annotations

import traceback
import io
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytz

from cdasws import CdasWs
from astropy.time import Time

import general_functions as func


# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------
cdas = CdasWs()

# Cache variable lists per CDAWeb dataset to avoid repeated metadata calls.
_CDAS_VAR_CACHE = {}

# Cache Horizons distance grids to avoid repeated network calls.
_HORIZONS_DIST_CACHE = {}


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _to_utc_dt(t):
    """
    Convert arbitrary timestamp-like input to tz-aware UTC python datetime.
    """
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime().replace(tzinfo=pytz.UTC)


def _mask_cda_fill(arr, thr=1e30):
    """
    Convert CDAWeb fill values (~ -1e30) to NaN.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.number):
        a = a.astype(float, copy=False)
        a[a <= -thr] = np.nan
    return a


def _force_index_tz_naive_utc(df):
    """
    Convert tz-aware DatetimeIndex -> tz-naive UTC (required by legacy pipeline code).
    """
    if df is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df


def _dedup_sort_index(df):
    """
    Ensure monotonic increasing, unique DatetimeIndex.
    """
    if df is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df


def _median_dt_seconds(df):
    if df is None or len(df) < 3:
        return np.nan
    try:
        dt = df.index.to_series().diff().dropna()
        if len(dt) == 0:
            return np.nan
        return float(dt.median().total_seconds())
    except Exception:
        return np.nan


def _debug_enabled(settings):
    try:
        return bool(settings.get("debug_ulysses", False))
    except Exception:
        return False


def _debug_print(settings, msg):
    if _debug_enabled(settings):
        print(msg)


def _debug_df(name, df, settings, cols_to_check=None):
    if not _debug_enabled(settings):
        return
    if df is None:
        print(f"[ULYSSES DEBUG] {name}: None")
        return

    ndup = int(df.index.duplicated().sum()) if isinstance(df.index, pd.DatetimeIndex) else -1
    tz = df.index.tz if isinstance(df.index, pd.DatetimeIndex) else None
    dtmed = _median_dt_seconds(df)
    print(
        f"[ULYSSES DEBUG] {name}: shape={df.shape}, idx=[{df.index.min()} -> {df.index.max()}], "
        f"tz={tz}, dup_idx={ndup}, dt_med_s={dtmed}"
    )
    if cols_to_check:
        missing = [c for c in cols_to_check if c not in df.columns]
        print(f"[ULYSSES DEBUG] {name}: missing_cols={missing}")

    maxrows = int(settings.get("ulysses_debug_maxrows", 2))
    try:
        print(f"[ULYSSES DEBUG] {name}.head({maxrows}):\n{df.head(maxrows)}")
        print(f"[ULYSSES DEBUG] {name}.tail({maxrows}):\n{df.tail(maxrows)}")
    except Exception:
        pass


def _debug_intersection(dfmag, dfpar, settings, tag):
    if not _debug_enabled(settings):
        return
    if dfmag is None or dfpar is None:
        print(f"[ULYSSES DEBUG] intersection {tag}: dfmag or dfpar is None")
        return
    try:
        inter = dfmag.index.intersection(dfpar.index)
        print(f"[ULYSSES DEBUG] intersection {tag}: |MAG ∩ PAR|={len(inter)} (MAG={len(dfmag)}, PAR={len(dfpar)})")
    except Exception as e:
        print(f"[ULYSSES DEBUG] intersection {tag}: failed: {e}")


def _get_cdas_varnames(dataset, settings=None):
    """
    Return list of variable names available in a CDAWeb dataset.
    Cached in-memory.
    """
    if dataset in _CDAS_VAR_CACHE:
        return _CDAS_VAR_CACHE[dataset]
    try:
        vars_meta = cdas.get_variables(dataset)
        names = [v.get("Name") for v in (vars_meta or []) if isinstance(v, dict) and "Name" in v]
        _CDAS_VAR_CACHE[dataset] = names
        return names
    except Exception as e:
        if settings is not None:
            _debug_print(settings, f"[ULYSSES DEBUG] get_variables failed for {dataset}: {e}")
        _CDAS_VAR_CACHE[dataset] = []
        return []


def _fetch_cdas_in_chunks(dataset, vars_list, t0_utc, t1_utc, chunk_hours, settings=None):
    """
    Fetch data in chunks to avoid CDAWeb timeouts / payload limits.
    Returns dict with "Epoch" and variables, or None if nothing returned.
    """
    t0 = pd.Timestamp(t0_utc).tz_convert("UTC")
    t1 = pd.Timestamp(t1_utc).tz_convert("UTC")

    out_epochs = []
    out_vars = {v: [] for v in vars_list}

    step = pd.Timedelta(hours=float(chunk_hours))
    cur = t0
    while cur < t1:
        nxt = min(cur + step, t1)
        try:
            status, data = cdas.get_data(
                dataset,
                vars_list,
                cur.to_pydatetime().replace(tzinfo=pytz.UTC),
                nxt.to_pydatetime().replace(tzinfo=pytz.UTC),
            )
        except Exception as e:
            if settings is not None:
                _debug_print(settings, f"[ULYSSES DEBUG] get_data failed for {dataset} [{cur} -> {nxt}]: {e}")
            data = None

        if data is not None and "Epoch" in data and data["Epoch"] is not None and len(data["Epoch"]) > 0:
            out_epochs.append(np.asarray(data["Epoch"]))
            for v in vars_list:
                if v in data and data[v] is not None:
                    out_vars[v].append(np.asarray(data[v]))
        cur = nxt

    if len(out_epochs) == 0:
        return None

    merged = {"Epoch": np.concatenate(out_epochs)}
    for v in vars_list:
        if len(out_vars[v]) > 0:
            try:
                merged[v] = np.concatenate(out_vars[v], axis=0)
            except Exception:
                merged[v] = np.concatenate([np.asarray(x) for x in out_vars[v]], axis=0)
        else:
            merged[v] = None
    return merged


def _try_fetch_with_fallback(datasets, vars_list, t0, t1, chunk_hours, settings, label):
    """
    Try datasets in order until we get non-empty data.
    """
    errs = {}
    for ds in datasets:
        try:
            avail = _get_cdas_varnames(ds, settings=settings)
            missing = [v for v in vars_list if v not in avail]
            if len(avail) > 0 and len(missing) > 0:
                errs[ds] = f"missing_vars={missing}"
                _debug_print(settings, f"[ULYSSES DEBUG] {label}: skip {ds} (missing {missing})")
                continue

            data = _fetch_cdas_in_chunks(ds, vars_list, t0, t1, chunk_hours, settings=settings)
            if data is None or "Epoch" not in data or data["Epoch"] is None or len(data["Epoch"]) == 0:
                errs[ds] = "empty"
                _debug_print(settings, f"[ULYSSES DEBUG] {label}: {ds} returned empty")
                continue

            _debug_print(settings, f"[ULYSSES DEBUG] {label}: using dataset={ds} (n={len(data['Epoch'])})")
            return ds, data

        except Exception as e:
            errs[ds] = repr(e)
            _debug_print(settings, f"[ULYSSES DEBUG] {label}: {ds} failed: {e}")
            continue

    raise RuntimeError(f"All {label} datasets failed or were empty: {errs}")


def _extract_vec3(arr, fill_to_nan=True):
    """
    Convert CDA arrays into (N,3) float array when possible.
    If only (N,) provided, returns (N,3) with [x, nan, nan].
    """
    a = np.asarray(arr)
    if fill_to_nan:
        a = _mask_cda_fill(a)
    if a.ndim == 1:
        out = np.full((len(a), 3), np.nan, dtype=float)
        out[:, 0] = a.astype(float, copy=False)
        return out
    if a.ndim == 2 and a.shape[1] >= 3:
        return a[:, :3].astype(float, copy=False)
    if a.ndim == 2 and a.shape[1] == 1:
        out = np.full((a.shape[0], 3), np.nan, dtype=float)
        out[:, 0] = a[:, 0].astype(float, copy=False)
        return out
    flat = a.reshape(-1).astype(float, copy=False)
    out = np.full((len(flat), 3), np.nan, dtype=float)
    out[:, 0] = flat
    return out


def _extract_scalar(arr, idx0=0):
    """
    Return 1D float array from CDA scalar-or-vector.
    If (N,M), choose column idx0 if present.
    """
    a = _mask_cda_fill(arr)
    if a.ndim == 1:
        return a.astype(float, copy=False)
    if a.ndim == 2 and a.shape[1] > idx0:
        return a[:, idx0].astype(float, copy=False)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0].astype(float, copy=False)
    return np.asarray(a).reshape(-1).astype(float, copy=False)


def _dropna_particles_lenient(df):
    """
    Keep rows that have at least Vr and density; tolerate missing tangential components.
    """
    if df is None or len(df) == 0:
        return df

    subset = []
    for c in ("Vr",):
        if c in df.columns:
            subset.append(c)
    if "np" in df.columns:
        subset.append("np")
    elif "Np" in df.columns:
        subset.append("Np")
    if len(subset) == 0:
        return df

    out = df.dropna(subset=subset)
    if len(out) == 0:
        return df
    return out


def _time_interp_series_to_index(s, idx):
    """
    Time-interpolate a Series onto DatetimeIndex `idx` (tz-naive UTC in this file).
    """
    idx = pd.DatetimeIndex(idx)
    if len(idx) == 0:
        return pd.Series(dtype="float64", index=idx, name="Dist_au")

    if s is None:
        return pd.Series(np.nan, index=idx, name="Dist_au", dtype="float64")

    s = pd.Series(s).copy()
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) == 0:
        return pd.Series(np.nan, index=idx, name="Dist_au", dtype="float64")

    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]

    all_idx = s.index.union(idx).sort_values()
    s2 = s.reindex(all_idx).interpolate(method="time", limit_direction="both")
    out = s2.reindex(idx)
    out.name = "Dist_au"
    return out.astype("float64")


def _download_sun_distance_au_horizons(
    target,
    idx,
    min_step_s=120,
    max_points=5000,
    timeout=60,
    cache=True,
):
    """
    Direct JPL Horizons API query for heliocentric distance in AU on DatetimeIndex `idx`.

    This intentionally bypasses sunpy/astroquery's Horizons path that can fail with
    "Unknown units specification" when OUT_UNITS=AU-D appears.
    """
    idx = pd.DatetimeIndex(idx)
    if len(idx) == 0:
        return pd.Series(dtype="float64", index=idx, name="Dist_au")

    idx_sorted = idx.sort_values()
    t0 = pd.Timestamp(idx_sorted[0])
    t1 = pd.Timestamp(idx_sorted[-1])

    total_s = max(1.0, (idx_sorted[-1] - idx_sorted[0]).total_seconds())
    step_s = int(np.ceil(total_s / max(1, (int(max_points) - 1))))
    step_s = max(int(min_step_s), step_s)

    cache_key = (str(target), str(t0.value), str(t1.value), int(step_s))
    if cache and cache_key in _HORIZONS_DIST_CACHE:
        s_grid = _HORIZONS_DIST_CACHE[cache_key]
        return _time_interp_series_to_index(s_grid, idx)

    base = "https://ssd.jpl.nasa.gov/api/horizons.api"
    params = {
        "format": "text",
        "MAKE_EPHEM": "'YES'",
        "TABLE_TYPE": "'VECTORS'",
        "COMMAND": f"'{str(target)}'",
        "CENTER": "'500@10'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
        "TP_TYPE": "'ABSOLUTE'",
        "VEC_LABELS": "'YES'",
        "VEC_TABLE": "'3'",
        "CSV_FORMAT": "'YES'",
        "VEC_CORR": "'NONE'",
        "VEC_DELTA_T": "'NO'",
        "OBJ_DATA": "'NO'",
        "START_TIME": f"'{t0.strftime('%Y-%m-%d %H:%M:%S')}'",
        "STOP_TIME": f"'{t1.strftime('%Y-%m-%d %H:%M:%S')}'",
        "STEP_SIZE": f"'{int(step_s)}s'",
        "OUT_UNITS": "'KM-S'",
    }

    url = base + "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "MHDTurbPy/1.0"})

    with urlopen(req, timeout=int(timeout)) as r:
        text = r.read().decode("utf-8", errors="replace")

    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError(text[:1200])

    pre, rest = text.split("$$SOE", 1)
    data_block, _ = rest.split("$$EOE", 1)

    header_line = None
    for line in reversed(pre.splitlines()):
        if "JDTDB" in line and "," in line:
            header_line = line
            break
    if header_line is None:
        raise ValueError("Could not locate Horizons CSV header line containing JDTDB.")

    cols = [c.strip() for c in header_line.split(",") if c.strip()]

    raw_lines = []
    for line in data_block.splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        raw_lines.append(s)

    if len(raw_lines) == 0:
        raise ValueError("Horizons response contained no data lines between $$SOE/$$EOE.")

    df = pd.read_csv(io.StringIO("\n".join(raw_lines)), header=None)
    if len(cols) == df.shape[1]:
        df.columns = cols
    else:
        df.columns = [f"c{i}" for i in range(df.shape[1])]

    if "JDTDB" not in df.columns:
        raise ValueError("Horizons parse failed: missing JDTDB column.")

    jd = df["JDTDB"].astype(float).to_numpy()
    t = pd.to_datetime(Time(jd, format="jd", scale="tdb").utc.datetime64)

    if "RG" in df.columns:
        rg_km = df["RG"].astype(float).to_numpy()
        dist_au = rg_km / 149597870.700
    else:
        need = ("X", "Y", "Z")
        if not all(c in df.columns for c in need):
            raise ValueError("Horizons parse failed: expected RG or X,Y,Z columns.")
        x = df["X"].astype(float).to_numpy()
        y = df["Y"].astype(float).to_numpy()
        z = df["Z"].astype(float).to_numpy()
        dist_au = np.sqrt(x * x + y * y + z * z) / 149597870.700

    s_grid = pd.Series(dist_au, index=pd.DatetimeIndex(t)).sort_index()
    s_grid = s_grid[~s_grid.index.duplicated(keep="first")]
    s_grid.name = "Dist_au"

    if cache:
        _HORIZONS_DIST_CACHE[cache_key] = s_grid

    return _time_interp_series_to_index(s_grid, idx)


# ---------------------------------------------------------------------
# Internal load functions with settings (fallback chains live here)
# ---------------------------------------------------------------------
def _load_ulysses_particles_with_settings(start_time, end_time, settings):
    """
    Particle + position load with dataset fallback.
    Returns (dfpar, dfdis).

    Guarantees: dfpar includes column Dist_au (AU), filled from CDA position if available,
    otherwise via Horizons.
    """
    default_settings = {
        "ulysses_particle_datasets": ["UY_M0_BAI", "UY_M1_BAI"],
        "ulysses_particle_vars": ["Velocity", "Density", "Temperature"],
        "ulysses_particle_chunk_hours": 24 * 30,

        "ulysses_pos_datasets": ["ULYSSES_HELIO1HR_POSITION", "ULYSSES_HELIO1DAY_POSITION"],
        "ulysses_pos_chunk_hours": 24 * 180,

        # "auto" -> CDA position if possible, else Horizons.
        # "cdas" -> CDA only.
        # "horizons" -> Horizons only.
        "ulysses_dist_mode": "auto",
        "ulysses_horizons_min_step_s": 120,
        "ulysses_horizons_max_points": 5000,
        "ulysses_horizons_timeout": 60,
        "ulysses_horizons_command": "-55",
    }
    settings = {**default_settings, **(settings or {})}

    t0 = _to_utc_dt(start_time)
    t1 = _to_utc_dt(end_time)

    # ---- plasma moments
    ds_par, data = _try_fetch_with_fallback(
        settings["ulysses_particle_datasets"],
        settings["ulysses_particle_vars"],
        t0,
        t1,
        settings["ulysses_particle_chunk_hours"],
        settings,
        label="particles",
    )

    idx = pd.to_datetime(data["Epoch"], utc=True)

    vel = _extract_vec3(data.get("Velocity"))
    dens = _extract_scalar(data.get("Density"), idx0=0)

    temp = data.get("Temperature")
    tp = _extract_scalar(temp, idx0=1)

    dfpar = pd.DataFrame(
        index=idx,
        data={
            "Vr": vel[:, 0],
            "Vt": vel[:, 1],
            "Vn": vel[:, 2],
            "np": dens,
            "Tp": tp,
        },
    )

    dfpar.loc[dfpar["Vr"] < -1e30, :] = np.nan
    dfpar["Vth"] = 0.128487 * np.sqrt(dfpar["Tp"])

    # Compatibility alias
    dfpar["Np"] = dfpar["np"]

    # Guaranteed column (filled below)
    dfpar["Dist_au"] = np.nan

    dfpar = _dedup_sort_index(dfpar)
    dfpar = _force_index_tz_naive_utc(dfpar)
    dfpar = _dedup_sort_index(dfpar)

    try:
        dfpar.attrs["cdas_dataset"] = ds_par
    except Exception:
        pass

    _debug_df("dfpar_loaded", dfpar, settings, cols_to_check=["Vr", "Vt", "Vn", "np", "Np", "Tp", "Vth", "Dist_au"])

    # ---- position (fallback)
    dfdis = None
    pos_errs = {}

    dist_mode = str(settings.get("ulysses_dist_mode", "auto")).lower().strip()

    # (A) CDA position path
    if dist_mode in ("auto", "cdas"):
        for ds_pos in settings["ulysses_pos_datasets"]:
            try:
                avail = _get_cdas_varnames(ds_pos, settings=settings)
                if len(avail) == 0:
                    pos_errs[ds_pos] = "no_var_metadata"
                    continue

                lat_var = None
                lon_var = None
                for base in ("HGI", "HG", "SE"):
                    lv = f"{base}_LAT"
                    lo = f"{base}_LON"
                    if lv in avail and lo in avail:
                        lat_var = lv
                        lon_var = lo
                        break

                if "RAD_AU" not in avail or lat_var is None or lon_var is None:
                    pos_errs[ds_pos] = f"missing RAD_AU or lon/lat (lat={lat_var}, lon={lon_var})"
                    continue

                vars_pos = ["RAD_AU", lat_var, lon_var]
                ds_used, data_pos = _try_fetch_with_fallback(
                    [ds_pos],
                    vars_pos,
                    t0,
                    t1,
                    settings["ulysses_pos_chunk_hours"],
                    settings,
                    label="position",
                )

                idxp = pd.to_datetime(data_pos["Epoch"], utc=True)
                rad = _extract_scalar(data_pos["RAD_AU"], idx0=0)
                lat = _extract_scalar(data_pos[lat_var], idx0=0)
                lon = _extract_scalar(data_pos[lon_var], idx0=0)

                dfdis_pos = pd.DataFrame(
                    index=idxp,
                    data={"Dist_au": rad, "lon": lon, "lat": lat, "RAD_AU": rad},
                )
                dfdis_pos = _dedup_sort_index(dfdis_pos)
                dfdis_pos = _force_index_tz_naive_utc(dfdis_pos)
                dfdis_pos = _dedup_sort_index(dfdis_pos)

                # Fill dfpar Dist_au by time interpolation onto particle cadence
                dist_al = _time_interp_series_to_index(dfdis_pos["Dist_au"], dfpar.index)
                dfpar.loc[:, "Dist_au"] = dist_al.to_numpy(copy=False)

                # Keep dfdis at dfpar cadence for downstream compatibility
                dfdis = dfdis_pos.reindex(pd.DatetimeIndex(dfpar.index), method="nearest")

                try:
                    dfdis.attrs["cdas_dataset"] = ds_used
                    dfdis.attrs["cdas_lonlat_frame"] = lat_var.split("_")[0]
                except Exception:
                    pass

                _debug_df("dfdis_loaded", dfdis, settings, cols_to_check=["Dist_au", "RAD_AU", "lat", "lon"])
                break

            except Exception as e:
                pos_errs[ds_pos] = repr(e)
                _debug_print(settings, f"[ULYSSES DEBUG] position: {ds_pos} failed: {e}")
                continue

    if dfdis is None:
        _debug_print(settings, f"[ULYSSES DEBUG] position: all CDA fallbacks failed: {pos_errs}")
        dfdis = pd.DataFrame(
            index=dfpar.index,
            data={"Dist_au": np.nan, "lon": np.nan, "lat": np.nan, "RAD_AU": np.nan},
        )

    dfdis = _dedup_sort_index(dfdis)
    dfdis = _force_index_tz_naive_utc(dfdis)
    dfdis = _dedup_sort_index(dfdis)

    # (B) Horizons fallback (fill Dist_au only if requested or needed)
    need_horiz = (dist_mode in ("auto", "horizons")) and (
        ("Dist_au" not in dfpar.columns) or (not np.isfinite(dfpar["Dist_au"].to_numpy()).any())
    )
    if need_horiz:
        try:
            dist_h = _download_sun_distance_au_horizons(
                target=str(settings.get("ulysses_horizons_command", "-55")),
                idx=dfpar.index,
                min_step_s=float(settings.get("ulysses_horizons_min_step_s", 120)),
                max_points=int(settings.get("ulysses_horizons_max_points", 5000)),
                timeout=int(settings.get("ulysses_horizons_timeout", 60)),
                cache=True,
            )
            dfpar.loc[:, "Dist_au"] = dist_h.to_numpy(copy=False)
            dfdis.loc[:, "Dist_au"] = dist_h.to_numpy(copy=False)
            dfdis.loc[:, "RAD_AU"] = dist_h.to_numpy(copy=False)
            _debug_print(settings, "[ULYSSES DEBUG] Dist_au filled from Horizons.")
        except Exception as e:
            _debug_print(settings, f"[ULYSSES DEBUG] Horizons Dist_au failed: {e}")

    if "Dist_au" not in dfpar.columns:
        dfpar["Dist_au"] = np.nan

    return dfpar, dfdis


def _load_ulysses_mag_with_settings(start_time, end_time, settings):
    """
    MAG load with dataset fallback.
    Returns (dfmag, dfmag1, infos).
    """
    default_settings = {
        "ulysses_mag_datasets": ["UY_1SEC_VHM", "UY_M1_VHM"],
        "ulysses_mag_vars": ["B_RTN", "B_MAG"],
        "ulysses_mag_chunk_hours": 24 * 3,
    }
    settings = {**default_settings, **(settings or {})}

    t0 = _to_utc_dt(start_time)
    t1 = _to_utc_dt(end_time)

    ds_mag, data = _try_fetch_with_fallback(
        settings["ulysses_mag_datasets"],
        settings["ulysses_mag_vars"],
        t0,
        t1,
        settings["ulysses_mag_chunk_hours"],
        settings,
        label="mag",
    )

    brtn = _extract_vec3(data.get("B_RTN"))
    bmag = _extract_scalar(data.get("B_MAG"), idx0=0)
    idx = pd.to_datetime(data["Epoch"], utc=True)

    dfmag = pd.DataFrame(
        index=idx,
        data={"Br": brtn[:, 0], "Bt": brtn[:, 1], "Bn": brtn[:, 2], "Btot": bmag},
    )

    dfmag.loc[np.abs(dfmag["Btot"]) > 1e3, ["Br", "Bt", "Bn", "Btot"]] = np.nan

    dfmag = _dedup_sort_index(dfmag)
    dfmag = _force_index_tz_naive_utc(dfmag)
    dfmag = _dedup_sort_index(dfmag)

    try:
        dfmag.attrs["cdas_dataset"] = ds_mag
    except Exception:
        pass

    try:
        dfmag1 = dfmag.resample("%ds" % (6)).mean()
    except Exception:
        dfmag1 = dfmag.copy()

    infos = {"resolution": 1}
    return dfmag, dfmag1, infos


# ---------------------------------------------------------------------
# Public API (SIGNATURES + RETURNS preserved)
# ---------------------------------------------------------------------
def LoadTimeSeriesUlysses_particles(start_time, end_time):
    """
    Load Ulysses Plasma Data.
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    Returns
    -------
    dfpar : DataFrame
        Columns: Vr, Vt, Vn, np, Tp, Vth, Np, Dist_au
    dfdis : DataFrame
        Columns: Dist_au, lon, lat, RAD_AU
    """
    return _load_ulysses_particles_with_settings(start_time, end_time, settings={})


def LoadHighResMagUlysses(start_time, end_time, verbose=True):
    """
    Load Ulysses MAG data (prefer 1-sec VHM) in RTN.
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """
    dfmag, dfmag1, infos = _load_ulysses_mag_with_settings(start_time, end_time, settings={})

    if verbose:
        try:
            print("Input tstart = %s, tend = %s" % (pd.Timestamp(start_time), pd.Timestamp(end_time)))
            print("Returned tstart = %s, tend = %s" % (dfmag.index[0], dfmag.index[-1]))
        except Exception:
            pass

    return dfmag, dfmag1, infos


def LoadMagUlysses(start_time, end_time, verbose=True):
    """
    Load Ulysses MAG data (kept for backward compatibility).
    """
    dfmag, dfmag1, infos = _load_ulysses_mag_with_settings(start_time, end_time, settings={})

    if verbose:
        try:
            print("Done.")
            print("Input tstart = %s, tend = %s" % (pd.Timestamp(start_time), pd.Timestamp(end_time)))
            print("Returned tstart = %s, tend = %s" % (dfmag.index[0], dfmag.index[-1]))
        except Exception:
            pass

    return dfmag, dfmag1, infos


def LoadTimeSeriesUlysses(
    start_time,
    end_time,
    settings,
    gap_time_threshold=10,
    time_amount=4,
    time_unit="h",
):
    """
    Load Time Series from Ulysses sc.
    SIGNATURE + RETURN CONTRACT preserved.
    """
    default_settings = {
        "apply_hampel": True,
        "hampel_params": {"w": 100, "std": 3},
        "part_resol": 3000,
        "MAG_resol": 1,
        "debug_ulysses": False,
        "ulysses_debug_maxrows": 2,
        "force_resampled_particles": False,

        "ulysses_particle_datasets": ["UY_M0_BAI", "UY_M1_BAI"],
        "ulysses_mag_datasets": ["UY_1SEC_VHM", "UY_M1_VHM"],
        "ulysses_pos_datasets": ["ULYSSES_HELIO1HR_POSITION", "ULYSSES_HELIO1DAY_POSITION"],

        # Dist_au fill behavior:
        "ulysses_dist_mode": "auto",
        "ulysses_horizons_min_step_s": 120,
        "ulysses_horizons_max_points": 5000,
        "ulysses_horizons_timeout": 60,
        "ulysses_horizons_command": "-55",
    }
    settings = {**default_settings, **(settings or {})}

    if _debug_enabled(settings):
        print("\n[ULYSSES DEBUG] ------------------------------")

    t0i, t1i = func.ensure_time_format(start_time, end_time)

    t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
    t1 = func.add_time_to_datetime_string(t1i, time_amount, time_unit)

    ind1 = func.string_to_datetime_index(t0i)
    ind2 = func.string_to_datetime_index(t1i)

    dfmag_out = None
    dfpar_out = None
    dfdis_out = None
    big_gaps_out = None

    diagnostics_MAG = {"Frac_miss": 100, "Large_gaps": 100, "Tot_gaps": 100, "resol": 100}
    diagnostics_PAR = {"Frac_miss": 100, "Large_gaps": 100, "Tot_gaps": 100, "resol": 100}

    # ---- MAG
    try:
        dfmag, dfmag1, infos = _load_ulysses_mag_with_settings(
            pd.Timestamp(t0),
            pd.Timestamp(t1),
            settings=settings,
        )

        _debug_df("dfmag_raw_loaded", dfmag, settings, cols_to_check=["Br", "Bt", "Bn", "Btot"])

        dfmag = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
        dfmag = _dedup_sort_index(dfmag)
        _debug_df("dfmag_cropped", dfmag, settings)

        big_gaps_out = func.find_big_gaps(dfmag, gap_time_threshold)

        diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings["MAG_resol"], large_gaps=10)

        if isinstance(diagnostics_MAG, dict) and "resampled_df" in diagnostics_MAG:
            dfmag_out = diagnostics_MAG["resampled_df"].interpolate(limit_direction="both").dropna()

        print("Mag fraction missing", diagnostics_MAG.get("Frac_miss", np.nan))
        _debug_df("dfmag_out", dfmag_out, settings)

    except Exception:
        traceback.print_exc()

    # ---- PAR + DIS
    try:
        dfpar, dfdis = _load_ulysses_particles_with_settings(
            pd.Timestamp(t0),
            pd.Timestamp(t1),
            settings=settings,
        )

        _debug_df(
            "dfpar_raw_loaded",
            dfpar,
            settings,
            cols_to_check=["Vr", "Vt", "Vn", "np", "Np", "Tp", "Vth", "Dist_au"],
        )
        _debug_df("dfdis_raw_loaded", dfdis, settings, cols_to_check=["Dist_au", "RAD_AU", "lat", "lon"])

        dfpar = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)
        dfdis = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfdis)

        dfpar = _dedup_sort_index(dfpar)
        dfdis = _dedup_sort_index(dfdis)

        _debug_df("dfpar_cropped", dfpar, settings)
        _debug_df("dfdis_cropped", dfdis, settings)

        diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings["part_resol"], large_gaps=10)
        print("Par fraction missing", diagnostics_PAR.get("Frac_miss", np.nan))

        dfpar_clean = dfpar.copy()

        if settings.get("force_resampled_particles", False) and isinstance(diagnostics_PAR, dict) and "resampled_df" in diagnostics_PAR:
            dfpar_out = diagnostics_PAR["resampled_df"].interpolate(limit_direction="both")
        else:
            dfpar_out = dfpar_clean.sort_index().interpolate(limit_direction="both")

        dfpar_out = _dropna_particles_lenient(dfpar_out)

        if dfpar_out is None or len(dfpar_out) == 0:
            print("[ULYSSES] particle dataframe empty after cleaning.")
            try:
                print(f"[ULYSSES] dfpar columns: {list(dfpar.columns)}")
                print(f"[ULYSSES] dfpar NaN fraction per col:\n{dfpar.isna().mean().sort_values(ascending=False).head(15)}")
            except Exception:
                pass

        dfdis_out = dfdis

        _debug_df("dfpar_out", dfpar_out, settings)
        _debug_intersection(dfmag_out, dfpar_out, settings, tag="MAG_out vs PAR_out")

        if isinstance(diagnostics_PAR, dict) and "resampled_df" in diagnostics_PAR:
            dfpar_rs = diagnostics_PAR["resampled_df"].interpolate(limit_direction="both")
            dfpar_rs = _dropna_particles_lenient(dfpar_rs)
            _debug_intersection(dfmag_out, dfpar_rs, settings, tag="MAG_out vs PAR_resampled")

    except Exception:
        traceback.print_exc()

    # ---- misc + guaranteed return
    try:
        keys_to_keep = ["Frac_miss", "Large_gaps", "Tot_gaps", "resol"]
        misc = {
            "Par": func.filter_dict(diagnostics_PAR, keys_to_keep) if isinstance(diagnostics_PAR, dict) else {},
            "Mag": func.filter_dict(diagnostics_MAG, keys_to_keep) if isinstance(diagnostics_MAG, dict) else {},
        }
        if _debug_enabled(settings):
            misc["Ulysses_debug"] = {
                "mag_len": None if dfmag_out is None else int(len(dfmag_out)),
                "par_len": None if dfpar_out is None else int(len(dfpar_out)),
                "mag_dt_med_s": None if dfmag_out is None else _median_dt_seconds(dfmag_out),
                "par_dt_med_s": None if dfpar_out is None else _median_dt_seconds(dfpar_out),
                "mag_par_intersection": None
                if (dfmag_out is None or dfpar_out is None)
                else int(len(dfmag_out.index.intersection(dfpar_out.index))),
                "mag_source": None if dfmag_out is None else getattr(dfmag_out, "attrs", {}).get("cdas_dataset", None),
                "par_source": None if dfpar_out is None else getattr(dfpar_out, "attrs", {}).get("cdas_dataset", None),
            }
    except Exception:
        traceback.print_exc()
        misc = {"Par": {}, "Mag": {}}

    return dfmag_out, dfpar_out, dfdis_out, big_gaps_out, misc
