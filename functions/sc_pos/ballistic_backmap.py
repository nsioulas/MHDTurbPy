#!/usr/bin/env python3
"""General ballistic backmapping to source-surface maps from MHDTurbPy final.pkl.

This module supports both:
1) a Python API for notebook-style workflows (`run_backmapping`) and
2) a CLI for script/batch usage.

Ballistic backmapping maps in situ parcels to a source surface by assuming radial
propagation at observed Vsw and rigid solar rotation during transit. This differs
from PFSS tracing, which follows modeled magnetic field lines from source surface to
photosphere (not performed here).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions.sc_pos.horizons_sun_lonlat import ballistic_source_longitude, resolve_spacecraft_spkid

BR_CANDIDATES = ["Br", "B_R", "B_r", "BRTN_R"]
VR_CANDIDATES = ["Vr", "V_R", "Vx", "V_r", "V", "|V|"]
NP_CANDIDATES = ["Np", "np", "N_p", "n_p", "proton_density"]


def _to_utc_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    if out.empty:
        raise ValueError(f"{name} is empty after time parsing.")
    return out


def _find_col(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    lower_to_orig = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]
    return None


def _extract_frames(obj: object) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    def as_df(x: object) -> Optional[pd.DataFrame]:
        return x if isinstance(x, pd.DataFrame) else None

    if isinstance(obj, pd.DataFrame):
        df = _to_utc_index(obj, "final dataframe")
        return df, df, list(df.columns)

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported final.pkl object type: {type(obj)}")

    top_keys = list(obj.keys())
    mag_df = None
    par_df = None

    mag_obj = obj.get("Mag") or obj.get("mag")
    if isinstance(mag_obj, dict):
        mag_df = as_df(mag_obj.get("B_resampled"))
    elif isinstance(mag_obj, pd.DataFrame):
        mag_df = mag_obj

    par_obj = obj.get("Par") or obj.get("par") or obj.get("plasma")
    if isinstance(par_obj, dict):
        par_df = as_df(par_obj.get("V_resampled"))
    elif isinstance(par_obj, pd.DataFrame):
        par_df = par_obj

    if mag_df is None or par_df is None:
        dfs = {k: v for k, v in obj.items() if isinstance(v, pd.DataFrame)}
        if mag_df is None:
            for v in dfs.values():
                if _find_col(v.columns, BR_CANDIDATES):
                    mag_df = v
                    break
        if par_df is None:
            for v in dfs.values():
                if _find_col(v.columns, VR_CANDIDATES) and _find_col(v.columns, NP_CANDIDATES):
                    par_df = v
                    break

    if mag_df is None or par_df is None:
        raise ValueError(
            "Could not identify MAG and plasma dataframes in final.pkl. "
            f"Top-level keys: {top_keys}."
        )

    mag_df = _to_utc_index(mag_df, "MAG dataframe")
    par_df = _to_utc_index(par_df, "plasma dataframe")
    available = sorted(set(map(str, mag_df.columns)).union(map(str, par_df.columns)))
    return mag_df, par_df, available


def _interp_to_index(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return df.reindex(df.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)


def _size_from_metric(x: pd.Series, smin: float = 5.0, smax: float = 200.0) -> pd.Series:
    valid = x.replace([np.inf, -np.inf], np.nan)

    # If there are no finite values at all, fall back to a constant midpoint size
    if not np.isfinite(valid).any():
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)

    lo, hi = np.nanpercentile(valid, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)

    return smin + (smax - smin) * np.clip((valid - lo) / (hi - lo), 0, 1)


def _sunpy_time_str(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _normalize_horizons_step(cadence: str) -> str:
    c = str(cadence).strip()
    c = re.sub(r"\s+", "", c)
    c = c.replace("minutes", "min").replace("minute", "min")
    c = c.replace("hours", "h").replace("hour", "h")
    c = c.replace("days", "d").replace("day", "d")
    c = c.replace("min", "m")
    return c


def _query_ephem(times: pd.DatetimeIndex, cadence: str, cache_file: Optional[Path], target: str) -> pd.DataFrame:
    if cache_file and cache_file.exists():
        cached = pd.read_pickle(cache_file)
        if isinstance(cached, pd.DataFrame) and {"lon_carr", "lat", "r_au"}.issubset(cached.columns):
            cached = _to_utc_index(cached, "cached ephemeris")
            if cached.index.min() <= times.min() and cached.index.max() >= times.max():
                return cached

    try:
        from astropy import units as u
        from sunpy.coordinates.ephemeris import get_horizons_coord
        from sunpy.coordinates.frames import HeliographicCarrington
    except Exception as exc:
        raise RuntimeError(
            "sunpy/astropy is required for Horizons ephemeris queries. "
            "Install with: pip install sunpy astropy"
        ) from exc

    spkid = resolve_spacecraft_spkid(target)
    time_query = {
        "start": _sunpy_time_str(times.min()),
        "stop": _sunpy_time_str(times.max()),
        "step": _normalize_horizons_step(cadence),
    }
    query = get_horizons_coord(body=spkid, time=time_query, id_type="id")
    hgc = query.transform_to(HeliographicCarrington(obstime=query.obstime))
    ephem = pd.DataFrame(
        {
            "lon_carr": hgc.lon.to_value(u.deg),
            "lat": hgc.lat.to_value(u.deg),
            "r_au": hgc.radius.to_value(u.AU),
        },
        index=pd.to_datetime(hgc.obstime.datetime64, utc=True),
    ).sort_index()

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ephem.to_pickle(cache_file)
    return ephem


def plot_backmapping_maps(
    data: pd.DataFrame,
    outdir: str | Path,
    target_label: str,
    vmin: float = 300,
    vmax: float = 900,
    highlight_percentile: float = 99.0,
) -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = data.index.max().strftime("%Y-%m-%d %H:%M UTC")

    pol = np.sign(data["Br_large"])
    colors = np.where(pol > 0, "red", np.where(pol < 0, "blue", "grey"))

    fig1, ax1 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax1.scatter(data["phi_src"], data["lat_src"], c=colors, s=20, alpha=0.8, linewidths=0)
    ax1.set(xlim=(0, 360), ylim=(-35, 35), xlabel="Source longitude (deg)", ylabel="Source latitude (deg)")
    ax1.set_title(f"{target_label} MAG")
    f1 = out / "source_surface_polarity.png"
    fig1.savefig(f1, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    sc2 = ax2.scatter(
        data["phi_src"], data["lat_src"], c=data["Vr_large"], s=data["size"], cmap="viridis",
        vmin=vmin, vmax=vmax, alpha=0.8, linewidths=0,
    )
    ax2.set(xlim=(0, 360), ylim=(-35, 35), xlabel="Source longitude (deg)", ylabel="Source latitude (deg)")
    ax2.set_title(f"{target_label} SWA ({stamp})")
    fig2.colorbar(sc2, ax=ax2).set_label("km/s")
    f2 = out / "source_surface_speed.png"
    fig2.savefig(f2, dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    top_thr = np.nanpercentile(data["Pram"], highlight_percentile) if np.isfinite(data["Pram"]).any() else np.inf
    high = data["Pram"] >= top_thr
    ax3.scatter(data.loc[~high, "phi_src"], data.loc[~high, "lat_src"], c=colors[~high], s=data.loc[~high, "size"], alpha=0.8, linewidths=0)
    ax3.scatter(data.loc[high, "phi_src"], data.loc[high, "lat_src"], c=colors[high], s=data.loc[high, "size"], alpha=0.9, edgecolors="black", linewidths=0.5)
    ax3.set(xlim=(0, 360), ylim=(-35, 35), xlabel="Source longitude (deg)", ylabel="Source latitude (deg)")
    ax3.set_title(f"Ram pressure ({stamp})")
    f3 = out / "source_surface_ram_pressure.png"
    fig3.savefig(f3, dpi=180)
    plt.close(fig3)

    return {"polarity": f1, "speed": f2, "ram_pressure": f3}


def run_backmapping(
    final_pkl: str | Path,
    outdir: str | Path,
    target: str = "SOLO",
    target_label: Optional[str] = None,
    start: Optional[str] = None,
    stop: Optional[str] = None,
    cadence: str = "30min",
    smooth: str = "12h",
    r_ss_rsun: float = 2.5,
    omega_deg_per_day: float = 14.1844,
    vsw_fallback: float = 400.0,
    vmin: float = 300.0,
    vmax: float = 900.0,
    highlight_percentile: float = 99.0,
    cache_ephem: bool = False,
) -> dict[str, Any]:
    final_obj = pd.read_pickle(final_pkl)
    mag_df, par_df, available_cols = _extract_frames(final_obj)

    br_col = _find_col(mag_df.columns, BR_CANDIDATES)
    vr_col = _find_col(par_df.columns, VR_CANDIDATES)
    np_col = _find_col(par_df.columns, NP_CANDIDATES)
    if br_col is None or vr_col is None or np_col is None:
        raise ValueError(
            "Failed to find required columns. "
            f"Need Br from {BR_CANDIDATES}, Vr from {VR_CANDIDATES}, Np from {NP_CANDIDATES}. "
            f"Available columns: {available_cols}"
        )

    plasma = par_df[[vr_col, np_col]].rename(columns={vr_col: "Vr", np_col: "Np"})
    if start:
        plasma = plasma[plasma.index >= pd.to_datetime(start, utc=True)]
    if stop:
        plasma = plasma[plasma.index <= pd.to_datetime(stop, utc=True)]
    if plasma.empty:
        raise ValueError("No plasma samples in selected time range.")

    mag = mag_df[[br_col]].rename(columns={br_col: "Br"})
    data = plasma.join(_interp_to_index(mag, plasma.index), how="left")
    if cadence:
        data = data.resample(cadence).median()

    smooth_td = pd.Timedelta(smooth)
    for k in ["Br", "Vr", "Np"]:
        data[f"{k}_large"] = data[k].rolling(smooth_td, min_periods=3).median()

    cache_file = Path(outdir) / f"ephem_{target.replace(' ', '_')}.pkl" if cache_ephem else None
    ephem = _query_ephem(data.index, cadence, cache_file, target)
    data = data.join(_interp_to_index(ephem, data.index)[["lon_carr", "lat", "r_au"]])

    phi_src, tau_days, fallback_flag = ballistic_source_longitude(
        lon_carr_deg=data["lon_carr"],
        r_au=data["r_au"],
        vsw_kms=data["Vr_large"],
        r_ss_rsun=r_ss_rsun,
        omega_deg_per_day=omega_deg_per_day,
        vsw_fallback_kms=vsw_fallback,
    )
    data["phi_src"] = phi_src
    data["tau_days"] = tau_days
    data["vsw_fallback_used"] = fallback_flag
    data["lat_src"] = data["lat"]
    data["Br_r2"] = data["Br_large"] * data["r_au"] ** 2
    data["Np_r2"] = data["Np_large"] * data["r_au"] ** 2
    data["Pram"] = data["Np_large"] * data["Vr_large"] ** 2
    data["size"] = _size_from_metric(data["Pram"])

    files = plot_backmapping_maps(
        data=data,
        outdir=outdir,
        target_label=(target_label or target),
        vmin=vmin,
        vmax=vmax,
        highlight_percentile=highlight_percentile,
    )

    ts_file = Path(outdir) / "ballistic_backmap_timeseries.pkl"
    data.to_pickle(ts_file)
    files["timeseries"] = ts_file
    return {"data": data, "files": files}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproducible, spacecraft-agnostic ballistic source-surface backmapping")
    p.add_argument("--final_pkl", required=True, help="Path to MHDTurbPy final.pkl")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--target", default="SOLO", help="Horizons spacecraft name/alias/SPKID (e.g. SOLO, PSP, -144)")
    p.add_argument("--target_label", default=None, help="Optional display label for figure titles")
    p.add_argument("--start", default=None, help="UTC start time (optional)")
    p.add_argument("--stop", default=None, help="UTC stop time (optional)")
    p.add_argument("--cadence", default="30min", help="Resample cadence + ephemeris step, e.g. 30min")
    p.add_argument("--smooth", default="12h", help="Rolling window, e.g. 6h, 12h, 24h")
    p.add_argument("--r_ss_rsun", type=float, default=2.5, help="Source-surface radius in R_sun")
    p.add_argument("--omega_deg_per_day", type=float, default=14.1844, help="Rotation rate used in mapping")
    p.add_argument("--vsw_fallback", type=float, default=400.0, help="Fallback speed (km/s) for invalid Vr")
    p.add_argument("--vmin", type=float, default=300.0, help="Speed colormap minimum (km/s)")
    p.add_argument("--vmax", type=float, default=900.0, help="Speed colormap maximum (km/s)")
    p.add_argument("--highlight_percentile", type=float, default=99.0, help="Pram percentile for black-edge highlight")
    p.add_argument("--cache_ephem", action="store_true", help="Cache ephemeris in outdir")
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = run_backmapping(
        final_pkl=args.final_pkl,
        outdir=args.outdir,
        target=args.target,
        target_label=args.target_label,
        start=args.start,
        stop=args.stop,
        cadence=args.cadence,
        smooth=args.smooth,
        r_ss_rsun=args.r_ss_rsun,
        omega_deg_per_day=args.omega_deg_per_day,
        vsw_fallback=args.vsw_fallback,
        vmin=args.vmin,
        vmax=args.vmax,
        highlight_percentile=args.highlight_percentile,
        cache_ephem=args.cache_ephem,
    )
    print("Wrote:")
    for name, f in result["files"].items():
        print(f" - {name}: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
