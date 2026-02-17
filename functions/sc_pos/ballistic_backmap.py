#!/usr/bin/env python3
"""General ballistic backmapping to source-surface maps from MHDTurbPy final.pkl.

This script performs *ballistic backmapping* from in situ measurements to a source
surface (default 2.5 R_sun). In ballistic backmapping, each plasma parcel is assumed
to propagate radially at an observed bulk speed, and source Carrington longitude is
estimated by subtracting/adding the Sun's rigid rotation during transit.

This is different from PFSS field-line tracing:
- Ballistic mapping: maps (lon, lat) to a source surface using radial propagation
  and solar rotation only.
- PFSS tracing: starts from source-surface points and follows modeled magnetic
  field lines down to the photosphere; this optional extra step is NOT required for
  the source-surface plots produced here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions.sc_pos.horizons_sun_lonlat import resolve_spacecraft_spkid, ballistic_source_longitude

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


def _query_ephem(
    times: pd.DatetimeIndex,
    cadence: str,
    cache_file: Optional[Path],
    target: str,
) -> pd.DataFrame:
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
    query = get_horizons_coord(
        body=spkid,
        time={"start": times.min().isoformat(), "stop": times.max().isoformat(), "step": cadence},
        id_type="id",
    )
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


def _interp_to_index(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return df.reindex(df.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)


def _size_from_metric(x: pd.Series, smin: float = 5.0, smax: float = 200.0) -> pd.Series:
    valid = x.replace([np.inf, -np.inf], np.nan)
    lo, hi = np.nanpercentile(valid, [5, 95]) if np.isfinite(valid).any() else (0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)
    return smin + (smax - smin) * np.clip((valid - lo) / (hi - lo), 0, 1)


def run(args: argparse.Namespace) -> list[Path]:
    final_obj = pd.read_pickle(args.final_pkl)
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
    if args.start:
        plasma = plasma[plasma.index >= pd.to_datetime(args.start, utc=True)]
    if args.stop:
        plasma = plasma[plasma.index <= pd.to_datetime(args.stop, utc=True)]
    if plasma.empty:
        raise ValueError("No plasma samples in selected time range.")

    mag = mag_df[[br_col]].rename(columns={br_col: "Br"})
    data = plasma.join(_interp_to_index(mag, plasma.index), how="left")
    data = data.resample(args.cadence).median() if args.cadence else data

    smooth = pd.Timedelta(args.smooth)
    for k in ["Br", "Vr", "Np"]:
        data[f"{k}_large"] = data[k].rolling(smooth, min_periods=3).median()

    cache_file = Path(args.outdir) / f"ephem_{args.target.replace(' ', '_')}.pkl" if args.cache_ephem else None
    ephem = _query_ephem(data.index, args.cadence, cache_file, args.target)
    data = data.join(_interp_to_index(ephem, data.index)[["lon_carr", "lat", "r_au"]])

    phi_src, tau_days, fallback_flag = ballistic_source_longitude(
        lon_carr_deg=data["lon_carr"],
        r_au=data["r_au"],
        vsw_kms=data["Vr_large"],
        r_ss_rsun=args.r_ss_rsun,
        omega_deg_per_day=args.omega_deg_per_day,
        vsw_fallback_kms=args.vsw_fallback,
    )
    data["phi_src"] = phi_src
    data["tau_days"] = tau_days
    data["vsw_fallback_used"] = fallback_flag
    data["lat_src"] = data["lat"]

    data["Br_r2"] = data["Br_large"] * data["r_au"] ** 2
    data["Np_r2"] = data["Np_large"] * data["r_au"] ** 2
    data["Pram"] = data["Np_large"] * data["Vr_large"] ** 2
    data["size"] = _size_from_metric(data["Pram"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = data.index.max().strftime("%Y-%m-%d %H:%M UTC")
    target_label = args.target_label or args.target

    pol = np.sign(data["Br_large"])
    colors = np.where(pol > 0, "red", np.where(pol < 0, "blue", "grey"))

    fig1, ax1 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax1.scatter(data["phi_src"], data["lat_src"], c=colors, s=20, alpha=0.8, linewidths=0)
    ax1.set(xlim=(0, 360), ylim=(-35, 35), xlabel="Source longitude (deg)", ylabel="Source latitude (deg)")
    ax1.set_title(f"{target_label} MAG")
    f1 = outdir / "source_surface_polarity.png"
    fig1.savefig(f1, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    sc2 = ax2.scatter(
        data["phi_src"], data["lat_src"], c=data["Vr_large"], s=data["size"], cmap="viridis",
        vmin=args.vmin, vmax=args.vmax, alpha=0.8, linewidths=0,
    )
    ax2.set(xlim=(0, 360), ylim=(-35, 35), xlabel="Source longitude (deg)", ylabel="Source latitude (deg)")
    ax2.set_title(f"{target_label} SWA ({stamp})")
    fig2.colorbar(sc2, ax=ax2).set_label("km/s")
    f2 = outdir / "source_surface_speed.png"
    fig2.savefig(f2, dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    top_thr = np.nanpercentile(data["Pram"], args.highlight_percentile) if np.isfinite(data["Pram"]).any() else np.inf
    high = data["Pram"] >= top_thr
    ax3.scatter(data.loc[~high, "phi_src"], data.loc[~high, "lat_src"], c=colors[~high], s=data.loc[~high, "size"], alpha=0.8, linewidths=0)
    ax3.scatter(data.loc[high, "phi_src"], data.loc[high, "lat_src"], c=colors[high], s=data.loc[high, "size"], alpha=0.9, edgecolors="black", linewidths=0.5)
    ax3.set(xlim=(0, 360), ylim=(-35, 35), xlabel="Source longitude (deg)", ylabel="Source latitude (deg)")
    ax3.set_title(f"Ram pressure ({stamp})")
    f3 = outdir / "source_surface_ram_pressure.png"
    fig3.savefig(f3, dpi=180)
    plt.close(fig3)

    data.to_pickle(outdir / "ballistic_backmap_timeseries.pkl")
    return [f1, f2, f3]


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
    files = run(args)
    print("Wrote:")
    for f in files:
        print(f" - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
