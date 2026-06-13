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
final particle cadence (after resampling / optional Hampel filtering), converted
from heliocentric HEE into Earth-centered GSE for storage, and returned as
``dfdis`` with the same DatetimeIndex as ``dfpar``.

The particle dataframe always keeps ``Dist_au``, ``Dist_km``, and the legacy
``RAD_AU`` alias. Cartesian coordinates are attached in either km or AU according
to ``settings['ephem_units']``. The returned ``dfdis`` keeps both km and AU
Cartesian columns plus the raw HEE coordinates for provenance.
"""

import importlib.util
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from cdasws import CdasWs  # noqa: F401  (kept for downstream callers; in-loader use goes via _common.get_cdas_client)
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

import ephemeris as _ephem  # noqa: E402
import general_functions as func  # noqa: E402

# Do not share one global CdasWs session across parallel notebook threads.
# Create a fresh client per request instead.

# Plasma constants and fill-value cleanup live in the consolidated _common
# module — the single source of truth for every per-SC loader (2026-05-03).
# Top-level import (not relative) because downloading_helpers/ is added to
# sys.path directly by ``path_setup.ensure_project_paths``.
from _common import (  # noqa: E402
    EV_TO_K as _EV_TO_K,
    replace_fill_values as _clean_fill_values,
    get_cdas_client as _get_cdas_client,
)


def _wind_verbose(settings: Optional[Dict[str, Any]] = None) -> bool:
    if isinstance(settings, dict):
        return bool(settings.get("verbose_download", True))
    return True


def _wind_log(message: str, settings: Optional[Dict[str, Any]] = None) -> None:
    if _wind_verbose(settings):
        print(f"[WIND] {message}")


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




def _cdas_get_data(dataset: str, variables, start_time, end_time, retries: int = 4, sleep_s: float = 2.0, settings: Optional[Dict[str, Any]] = None):
    """CDAWeb request with retry-with-backoff on a memoised thread-local client.

    2026-05-03 — switched from a per-attempt fresh ``CdasWs()`` to a
    thread-local memoised client (``_common.get_cdas_client``) so the
    underlying ``requests.Session`` keeps its TCP/TLS keep-alive
    sockets warm across the many 14-day chunks the canonical 7-day
    interval splits into.  Thread safety is preserved because each
    thread retains its own client.
    """
    last_exc = None
    for attempt in range(max(1, int(retries))):
        try:
            _wind_log(
                f"Requesting {dataset} variables={list(variables)} chunk=({pd.Timestamp(start_time)} -> {pd.Timestamp(end_time)}) attempt={attempt + 1}/{max(1, int(retries))}",
                settings,
            )
            client = _get_cdas_client()
            status, data = client.get_data(dataset, list(variables), start_time, end_time)
            if status:
                try:
                    n_rows = len(data["Epoch"])
                except Exception:
                    n_rows = None
                _wind_log(f"Received {dataset} status=True rows={n_rows}", settings)
            else:
                _wind_log(f"Received {dataset} status=False for requested chunk.", settings)
            return status, data
        except Exception as exc:
            last_exc = exc
            _wind_log(f"Request failed for {dataset} on attempt {attempt + 1}: {exc}", settings)
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


def LoadTimeSeriesWind_electrons(start_time, end_time, settings):
    """Load WIND electron moments from SWE H5 and return Te, Te_core in eV."""
    from astropy import units as u  # type: ignore

    t0 = _to_utc_datetime(start_time).to_pydatetime()
    t1 = _to_utc_datetime(end_time).to_pydatetime()

    frames = []
    for c0, c1 in _iter_time_chunks(t0, t1, max_days=14.0):
        status, data = _cdas_get_data("WI_H5_SWE", ["T_elec", "TcElec"], c0, c1, settings=settings)
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
    _wind_log(f"Loading WIND particles for {pd.Timestamp(start_time)} -> {pd.Timestamp(end_time)} at requested cadence {part_resol}s", settings)

    if part_resol <= 3:
        t0 = _to_utc_datetime(start_time).to_pydatetime().replace(tzinfo=pytz.UTC)
        t1 = _to_utc_datetime(end_time).to_pydatetime().replace(tzinfo=pytz.UTC)

        for c0, c1 in _iter_time_chunks(t0, t1, max_days=7.0):
            status, data = _cdas_get_data("WI_PM_3DP", ["P_DENS", "P_VELS", "P_TEMP", "TIME"], c0, c1, settings=settings)
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
            df_chunk["Vth"] = 0.12848657328083132 * np.sqrt(df_chunk["Tp_K"].clip(lower=0))
            frames.append(df_chunk)

        dfpar = _concat_timeseries_chunks(frames)
        if dfpar.empty:
            raise RuntimeError("Failed to download WI_PM_3DP particle data.")
        _wind_log(f"Concatenated WI_PM_3DP particles: rows={len(dfpar)} span=({dfpar.index[0]} -> {dfpar.index[-1]})", settings)
        qtn_flag = "No_QTN"
    else:
        for c0, c1 in _iter_time_chunks(pd.Timestamp(start_time), pd.Timestamp(end_time), max_days=14.0):
            status, data = _cdas_get_data(
                "WI_PLSP_3DP",
                ["MOM.P.DENSITY", "MOM.P.VELOCITY", "MOM.P.VTHERMAL", "TIME"],
                str(pd.Timestamp(c0)),
                str(pd.Timestamp(c1)),
                settings=settings,
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
        _wind_log(f"Concatenated WI_PLSP_3DP particles: rows={len(dfpar)} span=({dfpar.index[0]} -> {dfpar.index[-1]})", settings)
        qtn_flag = "No_QTN"

    vec_cols = ("Vr", "Vt", "Vn") if bool(settings.get("in_rtn", True)) else ("Vx", "Vy", "Vz")
    dfpar = _clean_fill_values(dfpar, vec_cols + ("np", "Tp", "Vth"))

    # 2026-05-03 — placeholder removed.  Position attachment is the
    # caller's responsibility (LoadTimeSeriesWIND attaches via
    # ``_ephem.attach_to_particle_df``).  When the placeholder existed it
    # masqueraded as data with NaN columns; returning ``None`` instead
    # makes "no ephemeris" explicit.
    return dfpar, None, qtn_flag



def LoadHighResMagWind(start_time, end_time, settings, verbose=True):
    """Load WIND magnetic field.

    Always download ``WI_H2_MFI``. When ``MAG_resol >= 3``, the final cadence
    matching to particles is handled later inside ``LoadTimeSeriesWIND`` via
    ``func.synchronize_dfs(..., upsample=False)``.
    """
    t0 = _to_utc_datetime(start_time).to_pydatetime().replace(tzinfo=pytz.UTC)
    t1 = _to_utc_datetime(end_time).to_pydatetime().replace(tzinfo=pytz.UTC)

    _wind_log(f"Loading WIND magnetic field for {pd.Timestamp(t0)} -> {pd.Timestamp(t1)} at requested cadence {settings.get('MAG_resol')}s", settings)
    frames = []
    for c0, c1 in _iter_time_chunks(t0, t1, max_days=14.0):
        status, data = _cdas_get_data("WI_H2_MFI", ["BGSE", "BF1"], c0, c1, settings=settings)
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
    _wind_log(f"Concatenated WI_H2_MFI magnetic field: rows={len(dfmag)} span=({dfmag.index[0]} -> {dfmag.index[-1]})", settings)

    bcols = ("Br", "Bt", "Bn") if bool(settings.get("in_rtn", True)) else ("Bx", "By", "Bz")
    dfmag = _clean_fill_values(dfmag, bcols + ("Btot",), threshold=-1e30)
    dfmag.loc[np.abs(dfmag["Btot"]) > 1e3, list(bcols) + ["Btot"]] = np.nan

    if verbose and not dfmag.empty:
        print("Done.")
        print(f"Input tstart = {t0}, tend = {t1}")
        print(f"Returned tstart = {dfmag.index[0]}, tend = {dfmag.index[-1]}")

    return dfmag


# --------------------------------------------------------------------------- #
# Notebook-facing thin wrappers (parallel download + skip-if-cached)
#
# These mirror the signature style of ``ACE.load_ace_mag`` / ``load_ace_swepam``
# and ``IMAP.load_imap_mag`` so the multispacecraft sync notebook can treat WIND
# like any other L1 mission — one uniform ``load_<sc>_mag(start, end, ...)`` call,
# no hardcoded ``final.pkl`` path.  Two properties the raw loaders lack and the
# notebook needs:
#   1. Parallel download — the interval is split into 1-day chunks fetched
#      CONCURRENTLY (joblib threads) instead of one monolithic multi-day request.
#   2. Skip-if-cached — the merged result is written to ``<data_path>/<key>.pkl``
#      and reused on the next run (cdasws itself does not cache to disk), so a
#      re-run re-downloads nothing.  Pass ``force=True`` to ignore the cache.
# --------------------------------------------------------------------------- #
def _wind_default_settings(in_rtn: bool, verbose: bool = False) -> Dict[str, Any]:
    """Full settings dict the raw WIND loaders expect (GSE by default)."""
    return {
        "sc": "WIND", "in_rtn": bool(in_rtn), "Down_electrons": False,
        "MAG_resol": 1, "part_resol": 3, "upsample_low_freq_ts": False,
        "apply_hampel": False, "hampel_params": {"w": 200, "std": 3},
        "Max_par_missing": 100, "must_have_qtn": False,
        "verbose_download": bool(verbose),
        "Big_Gaps": {"E_big_gaps": 10, "SC_pot_big_gaps": 10, "Mag_big_gaps": 10,
                     "Par_big_gaps": 20, "QTN_big_gaps": 300},
    }


def _wind_daily_chunks(start_date, end_date):
    """Half-open 1-day [c0, c1) chunks spanning [start, end)."""
    edges = list(pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D"))
    if len(edges) < 2:                       # sub-day interval: single chunk
        return [(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if edges[-1] < pd.Timestamp(end_date):   # include the trailing partial day
        edges.append(pd.Timestamp(end_date))
    return list(zip(edges[:-1], edges[1:]))


def _wind_mag_chunk(c0, c1, settings):
    """Module-level (picklable) worker: one daily WI_H2_MFI chunk."""
    return LoadHighResMagWind(c0, c1, settings, verbose=False)


def _wind_par_chunk(c0, c1, settings):
    """Module-level (picklable) worker: one daily WI_PM_3DP chunk."""
    out = LoadTimeSeriesWind_particles(c0, c1, settings)
    return out[0] if isinstance(out, tuple) else out


def _wind_fetch_parallel(start_date, end_date, settings, worker, n_jobs, label, verbose):
    """Run ``worker(c0, c1, settings)`` over daily chunks and merge + dedup.

    Uses PROCESS-based parallelism (joblib ``loky``).  The NASA CDF library that
    cdasws reads through is NOT thread-safe: concurrent reads inside one process
    corrupt its global state and abort with severe ``NO_MORE_ACCESS`` /
    ``VAR_READ_ERROR`` errors.  Separate worker processes each own their CDF
    state, so the download is both parallel and crash-isolated (a worker that
    dies on a severe CDF error raises in the parent instead of killing the run).
    """
    from joblib import Parallel, delayed
    chunks = _wind_daily_chunks(start_date, end_date)
    n = max(1, min(int(n_jobs), len(chunks)))
    if verbose:
        print(f"[WIND] fetching {label} {start_date}..{end_date} in {len(chunks)} daily chunks, "
              f"n_jobs={n} ({'serial' if n == 1 or len(chunks) == 1 else 'loky processes'})", flush=True)
    if n == 1 or len(chunks) == 1:
        frames = [worker(c0, c1, settings) for c0, c1 in chunks]
    else:
        frames = Parallel(n_jobs=n, backend="loky")(
            delayed(worker)(c0, c1, settings) for c0, c1 in chunks
        )
    return _concat_timeseries_chunks([f for f in frames if isinstance(f, pd.DataFrame) and len(f)])


def load_wind_mag(
    start_date,
    end_date,
    *,
    data_path=None,
    out_cols=("Bx", "By", "Bz"),
    add_btot=True,
    btot_max=1e3,
    in_rtn=False,
    n_jobs=7,
    use_cache=True,
    force=False,
    settings=None,
    verbose=False,
):
    """Thin wrapper over :func:`LoadHighResMagWind` returning the GSE mag DataFrame.

    Mirrors :func:`ACE.load_ace_mag` / :func:`IMAP.load_imap_mag` so the
    multispacecraft sync notebook can treat WIND like any other L1 mission.
    WIND's native ``WI_H2_MFI`` cadence (~0.092 s) is returned untouched; the
    notebook downsamples it onto the master grid with the same
    ``func.synchronize_dfs`` machinery every other spacecraft uses.

    Daily chunks are fetched in PARALLEL (joblib threads) and the merged result
    is CACHED to ``<data_path>/<key>.pkl`` so re-runs skip the network entirely.

    Returns
    -------
    pandas.DataFrame
        Columns = ``[Bx, By, Bz(, Btot)]`` in GSE (nT), tz-naive UTC index.
    """
    s = dict(_wind_default_settings(in_rtn, verbose))
    if settings:
        s.update(settings)
    frame = "rtn" if bool(s.get("in_rtn", False)) else "gse"
    if data_path is None:
        data_path = Path.cwd() / "wind_data"
    data_path = Path(data_path).expanduser().resolve()
    key = (f"wind_mag_{pd.Timestamp(start_date):%Y%m%dT%H%M%S}_"
           f"{pd.Timestamp(end_date):%Y%m%dT%H%M%S}_{frame}.pkl")
    cache = data_path / key

    if use_cache and not force and cache.exists():
        dfmag = pd.read_pickle(cache)
        _wind_log(f"CACHED mag: {cache} ({len(dfmag)} rows) — skipping download.", s)
    else:
        dfmag = _wind_fetch_parallel(
            start_date, end_date, s, _wind_mag_chunk, n_jobs, "WI_H2_MFI", verbose,
        )
        if dfmag is None or dfmag.empty:
            raise RuntimeError("WIND MAG load returned no data.")
        if use_cache:
            data_path.mkdir(parents=True, exist_ok=True)
            dfmag.to_pickle(cache)
            _wind_log(f"cached mag -> {cache} ({len(dfmag)} rows)", s)

    bcols = ("Br", "Bt", "Bn") if frame == "rtn" else ("Bx", "By", "Bz")
    keep = [c for c in (out_cols if out_cols else bcols) if c in dfmag.columns]
    if len(keep) != 3:
        raise RuntimeError(f"WIND MAG DataFrame missing components; got columns {list(dfmag.columns)}")
    out = dfmag[keep].copy()
    if add_btot:
        if "Btot" in dfmag.columns:
            out["Btot"] = dfmag["Btot"].to_numpy(float)
        else:
            out["Btot"] = np.sqrt(np.nansum(out.to_numpy(float) ** 2, axis=1))
        if btot_max is not None and np.isfinite(btot_max):
            out.loc[np.abs(out["Btot"]) > float(btot_max), :] = np.nan
    return out


def load_wind_plasma(
    start_date,
    end_date,
    *,
    data_path=None,
    in_rtn=False,
    n_jobs=7,
    use_cache=True,
    force=False,
    settings=None,
    verbose=False,
):
    """Thin wrapper over :func:`LoadTimeSeriesWind_particles` returning WIND plasma.

    Same parallel-download + skip-if-cached behaviour as :func:`load_wind_mag`.
    WIND is the constellation's plasma provider (per CLAUDE.md sharing rule).

    Returns
    -------
    pandas.DataFrame
        WI_PM_3DP proton moments at native ~3 s cadence: ``Vx, Vy, Vz`` (GSE,
        km/s), ``np`` (cm^-3), ``Tp`` (eV), ``Tp_K`` (K), ``Vth`` (km/s);
        tz-naive UTC index.
    """
    s = dict(_wind_default_settings(in_rtn, verbose))
    if settings:
        s.update(settings)
    frame = "rtn" if bool(s.get("in_rtn", False)) else "gse"
    if data_path is None:
        data_path = Path.cwd() / "wind_data"
    data_path = Path(data_path).expanduser().resolve()
    key = (f"wind_plasma_{pd.Timestamp(start_date):%Y%m%dT%H%M%S}_"
           f"{pd.Timestamp(end_date):%Y%m%dT%H%M%S}_{frame}.pkl")
    cache = data_path / key

    if use_cache and not force and cache.exists():
        dfpar = pd.read_pickle(cache)
        _wind_log(f"CACHED plasma: {cache} ({len(dfpar)} rows) — skipping download.", s)
    else:
        dfpar = _wind_fetch_parallel(
            start_date, end_date, s, _wind_par_chunk, n_jobs, "WI_PM_3DP", verbose,
        )
        if dfpar is None or dfpar.empty:
            raise RuntimeError("WIND plasma load returned no data.")
        if use_cache:
            data_path.mkdir(parents=True, exist_ok=True)
            dfpar.to_pickle(cache)
            _wind_log(f"cached plasma -> {cache} ({len(dfpar)} rows)", s)
    return dfpar


def LoadTimeSeriesWIND(
    start_time,
    end_time,
    settings,
    gap_time_threshold=10,
    time_amount=4,
    time_unit="h",
):
    """Load WIND MAG/particle/electron timeseries for the pipeline.

    Cadence handling
    ----------------
    By default ``WI_H2_MFI`` is fetched at its native ~92 ms cadence and
    — when ``settings['MAG_resol'] >= 3`` — downsampled to the particle
    cadence via ``general_functions.synchronize_dfs`` (filtfilt
    anti-aliasing).  Set ``settings['mag_native_only'] = True`` to skip
    the downsample step and return the magnetic field DataFrame at its
    native MFI cadence; the magnetic and particle DataFrames will then
    have different indices and downstream consumers must reindex
    explicitly if a common timeline is required.  The downsample event,
    when it fires, is logged via ``_wind_log`` with the before/after
    sample counts so a silent precision loss is no longer possible
    (this used to happen with no notification — audit
    ``_phase0_inventory_2026-05-02.md`` Stage 1 "Inefficiencies", row 1).

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

    _wind_log(f"Entering LoadTimeSeriesWIND core loader for requested interval {t0i} -> {t1i} with padding {time_amount}{time_unit}", settings)

    # --- Magnetic field ---
    try:
        dfmag_raw = LoadHighResMagWind(pd.Timestamp(t0), pd.Timestamp(t1), settings, verbose=True)
        dfmag = _subset_interval(dfmag_raw, ind1, ind2)
    except Exception:
        traceback.print_exc()
        _wind_log("Magnetic-field stage failed.", settings)
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
            _wind_log("Electron stage failed.", settings)
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
        dfpar_raw, dfdis_placeholder_raw, qtn_flag = LoadTimeSeriesWind_particles(pd.Timestamp(t0), pd.Timestamp(t1), settings)
        dfpar = _subset_interval(dfpar_raw, ind1, ind2)
        # 2026-05-03 — ``LoadTimeSeriesWind_particles`` returns ``None`` for the
        # second tuple element after the placeholder removal in step 5.  Guard
        # so we don't pass ``None`` into ``_subset_interval`` (which would copy
        # a NoneType and crash).  ``dfdis`` is later populated by the SunPy
        # ephemeris attach below if ``Down_ephem`` is enabled.
        dfdis = (_subset_interval(dfdis_placeholder_raw, ind1, ind2)
                 if dfdis_placeholder_raw is not None else None)

        # Mag-cadence handling: ~92 ms native MFI → particle cadence (typically 3 s).
        # ``mag_native_only=True`` skips the downsample so the caller keeps the
        # high-rate MFI; logged-and-explicit either way (audit Stage 1 row 1
        # — "silent downsampling" is gone after 2026-05-03).
        mag_native_only = bool(settings.get("mag_native_only", False))
        do_downsample = (
            isinstance(dfmag, pd.DataFrame) and len(dfmag) > 0
            and float(settings.get("MAG_resol", 0)) >= 3
            and not mag_native_only
        )
        if do_downsample:
            n_before = len(dfmag)
            try:
                dfmag, dfpar = func.synchronize_dfs(dfmag, dfpar, upsample=False)
            except Exception:
                # Keep the interval alive for short / pathological spans by
                # falling back to direct time interpolation onto particle cadence.
                dfmag = func.newindex(dfmag, dfpar.index)
            n_after = len(dfmag)
            _wind_log(
                f"WIND mag downsampled from {n_before} samples (~92 ms native "
                f"WI_H2_MFI) to {n_after} samples (matched to particle cadence "
                f"{settings.get('part_resol')} s) via filtfilt anti-aliasing. "
                f"Set ``settings['mag_native_only'] = True`` to keep the "
                f"native-cadence mag instead.",
                settings,
            )
            if dfdis is not None:
                dfdis = dfdis.reindex(dfpar.index)
        elif (mag_native_only and isinstance(dfmag, pd.DataFrame)
              and len(dfmag) > 0
              and float(settings.get("MAG_resol", 0)) >= 3):
            _wind_log(
                f"mag_native_only=True: returning WIND mag at native cadence "
                f"({len(dfmag)} samples, ~92 ms MFI) without downsampling to "
                f"particle cadence. Mag and particle DataFrames have "
                f"different indices; downstream consumers must reindex "
                f"explicitly if a common timeline is required.",
                settings,
            )

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
        _wind_log("Particle stage failed.", settings)
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
            if isinstance(dfdis, pd.DataFrame):
                dfdis = dfdis.reindex(dfpar.index)
            print("Applied hampel filter to WIND particle columns:", list_2_hampel, "Window size", ws_hampel)
        except Exception:
            traceback.print_exc()

    # Attach ephemeris aligned to final particle cadence (automatic unless explicitly disabled).
    # 2026-05-03 — the legacy NaN placeholder fallback was removed: when ephemeris is disabled
    # or the SunPy fetch fails, ``dfdis`` is left as ``None`` so callers know real position
    # data is absent (and don't accidentally treat a NaN frame as valid coordinates).
    if isinstance(dfpar, pd.DataFrame) and len(dfpar) > 0 and bool(settings.get("Down_ephem", True)):
        try:
            dfpar, dfdis = _ephem.attach_to_particle_df(dfpar, target="WIND", settings=settings)
        except Exception:
            traceback.print_exc()
            print("WIND ephemeris attachment failed; dfdis set to None (no position data).")
            dfdis = None
    elif not isinstance(dfdis, pd.DataFrame):
        # Particle stage failed earlier or Down_ephem=False — leave dfdis explicit.
        dfdis = None

    keys_to_keep = ["Frac_miss", "Large_gaps", "Tot_gaps", "resol"]
    _wind_log(
        "Final WIND loader diagnostics: "
        + f"mag_rows={0 if not isinstance(dfmag, pd.DataFrame) else len(dfmag)}, "
        + f"par_rows={0 if not isinstance(dfpar, pd.DataFrame) else len(dfpar)}, "
        + f"elec_rows={0 if not isinstance(df_elec, pd.DataFrame) else len(df_elec)}",
        settings,
    )
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
