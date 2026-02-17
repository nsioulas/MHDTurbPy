#!/usr/bin/env python3
"""
MHDTurbPy ballistic backmapping: source-surface longitude/latitude maps.

This module is designed to work with MHDTurbPy's *downloaded interval* pickles,
i.e. the exact structure you showed:

    fin         = pd.read_pickle(finnames[which_int])      # dict with keys like 'Mag','Par',...
    sig         = pd.read_pickle(signames[which_int])      # DataFrame with sigma_c, sigma_r, ...
    gen, gaps   = ...                                      # optional; not required for mapping

Core outputs:
- A timeseries DataFrame saved to: <outdir>/ballistic_backmap_timeseries.pkl
  containing (at minimum): phi_src, lat_src, r_au, Br, Vr, Np, Pram, plus any
  requested extra variables (e.g., sigma_c, sigma_r).
- A multi-panel PNG: <outdir>/source_surface_maps.png
  with one panel per requested variable.

Important:
- This is ballistic/Parker-style backmapping to a *source surface* (r_ss).
- PFSS is NOT performed here. PFSS would be an additional step if you want
  photospheric footpoints.

SunPy requirement:
- Newer SunPy requires HeliographicCarrington(observer=...) for HGS->HGC transforms.
  This module sets observer to Earth's HGS position at each obstime.

"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --------------------------------------------------------------------------------------
# Local MHDTurbPy imports (keep compatible with your notebook style)
# --------------------------------------------------------------------------------------
_FUNCTIONS_DIR = Path(__file__).resolve().parents[1]  # .../functions
if str(_FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_FUNCTIONS_DIR))

import general_functions as func  # noqa: E402

from sc_pos.horizons_sun_lonlat import (
    ballistic_source_longitude,
    resolve_spacecraft_spkid,
)  # noqa: E402


# --------------------------------------------------------------------------------------
# Column discovery (minimal guessing; fail loudly with available columns)
# --------------------------------------------------------------------------------------
BR_CANDIDATES = ("Br", "B_R", "B_r", "BRTN_R")
VR_CANDIDATES = ("Vr", "V_R", "Vx", "V_r", "V", "|V|", "Vsw")
NP_CANDIDATES = ("Np", "np", "N_p", "n_p", "proton_density")


@dataclass(frozen=True)
class IntervalInputs:
    fin: object
    gen: object
    sig: Optional[pd.DataFrame]
    mag_gaps: object
    qtn_gaps: object
    par_gaps: object
    sc_pot_gaps: object
    paths: Dict[str, Path]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _to_utc_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(df.index)}")
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    if out.empty:
        raise ValueError(f"{name} is empty after time parsing.")
    return out


def _find_col(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    cols = list(df.columns)
    for c in candidates:
        if c in df.columns:
            return c
    lower_to_orig = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_to_orig:
            return lower_to_orig[c.lower()]
    raise KeyError(f"Missing {label}. Candidates={list(candidates)}. Available={cols[:80]}...")


def _interp_to_index(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    # time interpolation on union index
    return df.reindex(df.index.union(idx)).sort_index().interpolate(method="time").reindex(idx)


def _normalize_horizons_step(cadence: str) -> str:
    c = str(cadence).strip()
    c = re.sub(r"\s+", "", c)
    c = c.replace("minutes", "min").replace("minute", "min")
    c = c.replace("hours", "h").replace("hour", "h")
    c = c.replace("days", "d").replace("day", "d")
    c = c.replace("min", "m")
    return c


def _sunpy_time_str(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _size_from_metric(x: pd.Series, smin: float = 12.0, smax: float = 220.0) -> pd.Series:
    x = pd.Series(x, index=x.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    if not np.isfinite(x).any():
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)
    lo, hi = np.nanpercentile(x, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)
    return smin + (smax - smin) * np.clip((x - lo) / (hi - lo), 0, 1)


def _should_lognorm(v: np.ndarray) -> bool:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    v = v[v > 0]
    if v.size == 0:
        return False
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0:
        return False
    return (vmax / vmin) > 10.0


# --------------------------------------------------------------------------------------
# Exact interval loading (your required structure)
# --------------------------------------------------------------------------------------
def load_interval_inputs(
    root_dir: str | Path,
    sc: str,
    which_int: int,
    load_path: Optional[str | Path] = None,
) -> IntervalInputs:
    """
    Load MHDTurbPy interval pickles exactly as in your notebook snippet,
    using func.load_files() (no hardcoded dates).

    Returns IntervalInputs with the loaded objects and the file paths used.
    """
    root_dir = Path(root_dir)
    if load_path is None:
        load_path = root_dir / "examples" / "downloaded_intervals" / sc
    else:
        load_path = Path(load_path)

    # Locate files (exactly your naming)
    finnames = func.load_files(load_path, "final.pkl")
    gennames = func.load_files(load_path, "general.pkl")
    signames = func.load_files(load_path, "sig_c_sig_r.pkl")
    maggaps = func.load_files(load_path, "mag_gaps.pkl")
    qtngaps = func.load_files(load_path, "qtn_gaps.pkl")
    pargaps = func.load_files(load_path, "par_gaps.pkl")
    sc_pot = func.load_files(load_path, "sc_pot_gaps.pkl")

    # Load (exactly your naming)
    fin = pd.read_pickle(finnames[which_int])
    gen = pd.read_pickle(gennames[which_int])
    sig = pd.read_pickle(signames[which_int]) if len(signames) else None
    mag_gaps = pd.read_pickle(maggaps[which_int]) if len(maggaps) else None
    qtn_gaps = pd.read_pickle(qtngaps[which_int]) if len(qtngaps) else None
    par_gaps = pd.read_pickle(pargaps[which_int]) if len(pargaps) else None
    sc_pot_gaps = pd.read_pickle(sc_pot[which_int]) if len(sc_pot) else None

    paths = {
        "final": Path(finnames[which_int]),
        "general": Path(gennames[which_int]),
        "sig": Path(signames[which_int]) if len(signames) else Path(),
        "mag_gaps": Path(maggaps[which_int]) if len(maggaps) else Path(),
        "qtn_gaps": Path(qtngaps[which_int]) if len(qtngaps) else Path(),
        "par_gaps": Path(pargaps[which_int]) if len(pargaps) else Path(),
        "sc_pot_gaps": Path(sc_pot[which_int]) if len(sc_pot) else Path(),
    }

    if isinstance(sig, pd.DataFrame):
        sig = _to_utc_index(sig, "sig dataframe")

    return IntervalInputs(
        fin=fin,
        gen=gen,
        sig=sig if isinstance(sig, pd.DataFrame) else None,
        mag_gaps=mag_gaps,
        qtn_gaps=qtn_gaps,
        par_gaps=par_gaps,
        sc_pot_gaps=sc_pot_gaps,
        paths=paths,
    )


# --------------------------------------------------------------------------------------
# Extract MAG/PAR frames from fin dict
# --------------------------------------------------------------------------------------
def _extract_mag_par_from_fin(fin: object) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    fin is typically a dict with keys: 'Mag','Par',... and values that are either
    DataFrames or dicts containing DataFrames.

    We keep this conservative:
    - MAG: first DataFrame that contains any BR_CANDIDATES
    - PAR: first DataFrame that contains any VR_CANDIDATES and any NP_CANDIDATES
    """
    if isinstance(fin, pd.DataFrame):
        df = _to_utc_index(fin, "fin dataframe")
        return df, df

    if not isinstance(fin, dict):
        raise TypeError(f"fin must be dict or DataFrame, got {type(fin)}")

    def _unwrap_df(x: object) -> List[pd.DataFrame]:
        out: List[pd.DataFrame] = []
        if isinstance(x, pd.DataFrame):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                if isinstance(v, pd.DataFrame):
                    out.append(v)
        return out

    candidates: List[pd.DataFrame] = []
    for k in fin.keys():
        candidates.extend(_unwrap_df(fin[k]))

    if not candidates:
        raise ValueError(f"fin dict contains no DataFrames. Keys={list(fin.keys())}")

    mag_df = None
    par_df = None
    for df in candidates:
        dfu = _to_utc_index(df, "candidate dataframe")
        try:
            _find_col(dfu, BR_CANDIDATES, "Br")
            mag_df = dfu
            break
        except KeyError:
            continue

    for df in candidates:
        dfu = _to_utc_index(df, "candidate dataframe")
        try:
            _find_col(dfu, VR_CANDIDATES, "Vr")
            _find_col(dfu, NP_CANDIDATES, "Np")
            par_df = dfu
            break
        except KeyError:
            continue

    if mag_df is None or par_df is None:
        msg = [
            "Could not identify MAG and/or PAR DataFrames in fin.",
            f"fin keys: {list(fin.keys())}",
            "Hint: print(fin['Mag'].keys()) etc to inspect the internal structure.",
        ]
        raise ValueError("\n".join(msg))

    return mag_df, par_df


# --------------------------------------------------------------------------------------
# Ephemeris: robust HGS->HGC with observer (fixes your ConvertError)
# --------------------------------------------------------------------------------------
def _query_ephem(
    times: pd.DatetimeIndex,
    cadence: str,
    cache_file: Optional[Path],
    target: str,
) -> pd.DataFrame:
    """
    Returns ephemeris DataFrame indexed by time with columns:
      lon_carr [deg], lat [deg], r_au [AU]

    Fix: HeliographicCarrington requires observer != None in newer SunPy.
    """
    times = pd.to_datetime(times, utc=True)
    if cache_file and cache_file.exists():
        cached = pd.read_pickle(cache_file)
        if isinstance(cached, pd.DataFrame) and {"lon_carr", "lat", "r_au"}.issubset(cached.columns):
            cached = _to_utc_index(cached, "cached ephemeris")
            if cached.index.min() <= times.min() and cached.index.max() >= times.max():
                return cached

    try:
        import astropy.units as u
        from sunpy.coordinates import get_horizons_coord
        from sunpy.coordinates.frames import HeliographicStonyhurst, HeliographicCarrington
        from sunpy.coordinates import get_body_heliographic_stonyhurst
    except Exception as exc:
        raise RuntimeError("Missing sunpy/astropy for ephemeris queries.") from exc

    spkid = resolve_spacecraft_spkid(target)
    time_query = {
        "start": _sunpy_time_str(times.min()),
        "stop": _sunpy_time_str(times.max()),
        "step": _normalize_horizons_step(cadence),
    }

    # Horizons query -> SkyCoord (typically HGS-like frame)
    coord = get_horizons_coord(body=spkid, time=time_query, id_type="id")

    # Ensure we are in HGS first
    hgs = coord.transform_to(HeliographicStonyhurst(obstime=coord.obstime))

    # Define Earth observer at same obstime (vectorized)
    earth_obs = get_body_heliographic_stonyhurst("earth", hgs.obstime)

    # Transform to HGC with observer explicitly set (this is the critical fix)
    hgc = hgs.transform_to(HeliographicCarrington(obstime=hgs.obstime, observer=earth_obs))

    def _dist_to_au(c):
        if hasattr(c, "radius"):
            return c.radius.to_value(u.AU)
        if hasattr(c, "distance"):
            return c.distance.to_value(u.AU)
        return c.spherical.distance.to_value(u.AU)

    ephem = pd.DataFrame(
        {
            "lon_carr": np.mod(hgc.lon.to_value(u.deg), 360.0),
            "lat": hgc.lat.to_value(u.deg),
            "r_au": _dist_to_au(hgc),
        },
        index=pd.to_datetime(hgc.obstime.datetime64, utc=True),
    ).sort_index()

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ephem.to_pickle(cache_file)
    return ephem


# --------------------------------------------------------------------------------------
# Main backmapping
# --------------------------------------------------------------------------------------
def run_backmapping_interval(
    root_dir: str | Path,
    sc: str,
    which_int: int,
    outdir: str | Path,
    *,
    target: Optional[str] = None,
    target_label: Optional[str] = None,
    cadence: str = "60min",
    smooth: str = "1h",
    r_ss_rsun: float = 2.5,
    omega_deg_per_day: float = 14.1844,
    vsw_fallback: float = 400.0,
    cache_ephem: bool = True,
    plot_vars: Optional[Sequence[str]] = None,
    size_by: str = "Pram",
) -> Dict[str, Any]:
    """
    End-to-end entry point that:
      1) Loads interval pickles using func.load_files() (no hardcoded dates)
      2) Builds a merged, smoothed dataset
      3) Queries JPL ephemeris and computes (phi_src, lat_src)
      4) Produces a multi-panel source-surface map figure

    plot_vars:
      - list of variables to color panels by (min..max).
      - variables can come from:
          (a) derived columns: 'Vr', 'Np', 'Pram', 'Br', 'Br_r2', 'Np_r2'
          (b) sig columns: e.g., 'sigma_c', 'sigma_r', etc.
      - if you want the smoothed version of a variable, request '<var>_large'.

    size_by:
      - metric controlling marker size (default 'Pram').
    """
    inp = load_interval_inputs(root_dir=root_dir, sc=sc, which_int=which_int)

    return run_backmapping_from_objects(
        fin=inp.fin,
        sig=inp.sig,
        outdir=outdir,
        target=(target or sc),
        target_label=(target_label or sc),
        cadence=cadence,
        smooth=smooth,
        r_ss_rsun=r_ss_rsun,
        omega_deg_per_day=omega_deg_per_day,
        vsw_fallback=vsw_fallback,
        cache_ephem=cache_ephem,
        plot_vars=plot_vars,
        size_by=size_by,
        meta_paths=inp.paths,
    )


def run_backmapping_from_objects(
    *,
    fin: object,
    sig: Optional[pd.DataFrame],
    outdir: str | Path,
    target: str,
    target_label: str,
    cadence: str,
    smooth: str,
    r_ss_rsun: float,
    omega_deg_per_day: float,
    vsw_fallback: float,
    cache_ephem: bool,
    plot_vars: Optional[Sequence[str]],
    size_by: str,
    meta_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mag, par = _extract_mag_par_from_fin(fin)

    br_col = _find_col(mag, BR_CANDIDATES, "Br")
    vr_col = _find_col(par, VR_CANDIDATES, "Vr")
    np_col = _find_col(par, NP_CANDIDATES, "Np")

    # Coarse cadence for "large-scale" maps
    mag_c = mag[[br_col]].rename(columns={br_col: "Br"}).resample(cadence).median()
    par_c = par[[vr_col, np_col]].rename(columns={vr_col: "Vr", np_col: "Np"}).resample(cadence).median()

    data = mag_c.join(par_c, how="inner")
    data = _to_utc_index(data, "merged data")

    # Merge sig if present (interpolate onto data index to avoid shape issues)
    if isinstance(sig, pd.DataFrame) and not sig.empty:
        sig_i = _interp_to_index(sig, data.index)
        # avoid overwriting existing names
        for c in sig_i.columns:
            if c in data.columns:
                data[f"sig_{c}"] = sig_i[c]
            else:
                data[c] = sig_i[c]

    # Rolling smoothing (median) on a time window
    smooth_td = pd.Timedelta(smooth)

    # Always smooth core fields
    smooth_base = ["Br", "Vr", "Np"]

    # Also smooth any requested plot vars (if they exist)
    req = list(plot_vars) if plot_vars else []
    for v in req:
        if v.endswith("_large"):
            base = v[:-6]
            if base not in smooth_base:
                smooth_base.append(base)
        else:
            if v not in ("polarity", "Br_sign") and v not in smooth_base:
                smooth_base.append(v)

    for k in smooth_base:
        if k in data.columns:
            data[f"{k}_large"] = data[k].rolling(smooth_td, min_periods=3).median()

    # Derived quantities that do not require ephemeris
    # Use large fields when available to match the slide logic
    data["Pram"] = data.get("Np_large", data["Np"]) * (data.get("Vr_large", data["Vr"]) ** 2)

# Ephemeris query
    cache_file = outdir / f"ephem_{target.replace(' ', '_')}.pkl" if cache_ephem else None
    ephem = _query_ephem(data.index, cadence, cache_file, target)
    ephem_i = _interp_to_index(ephem, data.index)[["lon_carr", "lat", "r_au"]]

    # Guard against column overlap (e.g., if upstream products already include r_au)
    overlap = [c for c in ("lon_carr", "lat", "r_au") if c in data.columns]
    if overlap:
        data = data.drop(columns=overlap)

    data = data.join(ephem_i)

    # Ballistic longitude to source surface
    phi_src, tau_days, fallback_flag = ballistic_source_longitude(
        lon_carr_deg=data["lon_carr"],
        r_au=data["r_au"],
        vsw_kms=data.get("Vr_large", data["Vr"]),
        r_ss_rsun=r_ss_rsun,
        omega_deg_per_day=omega_deg_per_day,
        vsw_fallback_kms=vsw_fallback,
    )
    data["phi_src"] = phi_src
    data["lat_src"] = data["lat"]
    data["tau_days"] = tau_days
    data["vsw_fallback_used"] = fallback_flag

    data["Br_r2"] = data.get("Br_large", data["Br"]) * (data["r_au"] ** 2)
    data["Np_r2"] = data.get("Np_large", data["Np"]) * (data["r_au"] ** 2)

    # marker sizing
    if size_by not in data.columns:
        raise KeyError(f"size_by={size_by!r} not in data columns. Available={list(data.columns)[:80]}...")
    data["marker_size"] = _size_from_metric(data[size_by])

    # Default plot vars if not specified
    if plot_vars is None:
        plot_vars = ["polarity", "Vr_large", "Pram"]

    # Make figure
    fig_path = plot_source_surface_panels(
        data=data,
        outdir=outdir,
        target_label=target_label,
        plot_vars=list(plot_vars),
        size_col="marker_size",
    )

    ts_file = outdir / "ballistic_backmap_timeseries.pkl"
    data.to_pickle(ts_file)

    out: Dict[str, Any] = {
        "data": data,
        "files": {
            "maps": fig_path,
            "timeseries": ts_file,
        },
    }
    if meta_paths:
        out["meta_paths"] = meta_paths
    return out


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_source_surface_panels(
    *,
    data: pd.DataFrame,
    outdir: Path,
    target_label: str,
    plot_vars: List[str],
    size_col: str = "marker_size",
) -> Path:
    """
    Create one panel per variable in plot_vars, all in source lon/lat.

    Special values:
      - 'polarity' or 'Br_sign': discrete sign(Br_large)
    For all other variables:
      - continuous colormap min..max from data[var] (or data[var_large] if requested explicitly)
      - if positive-only and spans > 1 decade => LogNorm
    """
    n = len(plot_vars)
    if n < 1:
        raise ValueError("plot_vars must contain at least one variable")

    stamp = data.index.max().strftime("%Y-%m-%d %H:%M UTC")
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.8 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, plot_vars):
        x = data["phi_src"].to_numpy()
        y = data["lat_src"].to_numpy()
        s = data[size_col].to_numpy()

        if var in ("polarity", "Br_sign"):
            br = data.get("Br_large", data["Br"])
            pol = np.sign(br.to_numpy(dtype=float))
            c = np.where(pol > 0, "red", np.where(pol < 0, "blue", "0.6"))
            ax.scatter(x, y, s=20, c=c, alpha=0.85, linewidths=0)
            ax.set_title(f"{target_label} MAG ({stamp})")
        else:
            if var not in data.columns:
                raise KeyError(f"Requested plot var {var!r} not found in data columns.")

            v = data[var].to_numpy(dtype=float)

            norm = None
            if _should_lognorm(v):
                vv = v[np.isfinite(v) & (v > 0)]
                norm = LogNorm(vmin=np.nanmin(vv), vmax=np.nanmax(vv))

            sc = ax.scatter(x, y, s=s, c=v, cmap="viridis", alpha=0.85, linewidths=0, norm=norm)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(var)

            ax.set_title(f"{target_label} {var} ({stamp})")

        ax.set_xlim(0, 360)
        ax.set_ylim(-35, 35)
        ax.set_xlabel("Source longitude (deg)")
        ax.set_ylabel("Source latitude (deg)")
        ax.grid(True, alpha=0.25)

    out_png = outdir / "source_surface_maps.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------------
# CLI (optional)
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MHDTurbPy ballistic backmapping for downloaded intervals.")
    p.add_argument("--root_dir", required=True, help="Repo root (MHDTurbPy root)")
    p.add_argument("--sc", required=True, help="Spacecraft label, e.g. SOLO")
    p.add_argument("--which_int", type=int, default=0, help="Interval index")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--cadence", default="60min")
    p.add_argument("--smooth", default="1h")
    p.add_argument("--r_ss_rsun", type=float, default=2.5)
    p.add_argument("--omega_deg_per_day", type=float, default=14.1844)
    p.add_argument("--vsw_fallback", type=float, default=400.0)
    p.add_argument("--no_cache_ephem", action="store_true")
    p.add_argument("--plot_vars", nargs="*", default=None, help="Variables to plot (e.g. polarity Vr_large sigma_c_large)")
    p.add_argument("--size_by", default="Pram", help="Column to size markers by (default Pram)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    res = run_backmapping_interval(
        root_dir=args.root_dir,
        sc=args.sc,
        which_int=args.which_int,
        outdir=args.outdir,
        cadence=args.cadence,
        smooth=args.smooth,
        r_ss_rsun=args.r_ss_rsun,
        omega_deg_per_day=args.omega_deg_per_day,
        vsw_fallback=args.vsw_fallback,
        cache_ephem=(not args.no_cache_ephem),
        plot_vars=args.plot_vars,
        size_by=args.size_by,
    )
    print("Wrote:")
    for k, v in res["files"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
