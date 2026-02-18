from __future__ import annotations

"""sc_pos.backmap.pipeline

Minimal, physics-first source-surface backmapping for MHDTurbPy interval pickles.

Public API
----------
- ``backmap_interval``: the single canonical entry point.
- ``run_backmapping_interval``: backward-compatibility shim (thin wrapper).

What this module guarantees
---------------------------
- Deterministic interval loading by ``which_int`` using ``general_functions.load_files``.
- One canonical cadence index.
- Explicit units for all physical columns via ``df.attrs['units']``.
- Strictly positive travel times for r_sc > r_ss.
- Circular-safe longitude handling (no 0/360 interpolation spikes).
- Transparent metadata: no silent fallbacks.

What this module does NOT pretend to do
--------------------------------------
- It does not claim ballistic mapping is valid in shear/interaction regions.
- It does not model latitudinal evolution.
- It does not infer non-radial flow corrections unless you implement them explicitly.

The intent is a minimal, auditable baseline suitable for publication-grade figures.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import astropy.units as u
    import astropy.constants as const
except Exception as e:  # pragma: no cover
    raise RuntimeError("MHDTurbPy backmapping requires astropy. Install with: pip install astropy") from e

# MHDTurbPy import style: notebooks commonly insert functions/ into sys.path.
_FUNCTIONS_DIR = Path(__file__).resolve().parents[2]
if str(_FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_FUNCTIONS_DIR))

import general_functions as func  # noqa: E402

from .circular import halfwidth_deg, circ_percentile_deg
from .ephemeris import get_ephemeris_hgc, interp_ephemeris_to_index
from .mapping import map_to_source_surface
from .plotting import (
    VAR_SPECS,
    merge_var_specs,
    marker_sizes_from_metric,
    plot_source_surface_2d,
    plot_source_surface_3d,
    plot_velocity_profile,
)
from .travel_time import TravelTimeModel, build_model
from .units import attach_units, default_input_units, q_from_df, to_value


# -----------------------------------------------------------------------------
# Strict column candidates (used only if explicit names are not supplied)
# -----------------------------------------------------------------------------
BR_CANDIDATES = ("Br", "B_R", "B_r", "BRTN_R")
VR_CANDIDATES = ("Vr", "V_R", "V_r", "Vsw", "|V|")
NP_CANDIDATES = ("Np", "np", "N_p", "n_p", "proton_density")


def _to_utc_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(df.index)}")
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    if out.empty:
        raise ValueError(f"{name} is empty after time parsing.")
    return out


def _extract_mag_par(fin: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically extract MAG and PAR DataFrames from MHDTurbPy interval pickle."""
    if isinstance(fin, pd.DataFrame):
        df = _to_utc_index(fin, "fin")
        return df, df

    if not isinstance(fin, dict):
        raise TypeError(f"fin must be dict or DataFrame, got {type(fin)}")

    if "Mag" in fin and "Par" in fin and isinstance(fin["Mag"], (pd.DataFrame, dict)) and isinstance(fin["Par"], (pd.DataFrame, dict)):
        mag = fin["Mag"]
        par = fin["Par"]

        if isinstance(mag, dict):
            mag_frames = [(k, v) for k, v in mag.items() if isinstance(v, pd.DataFrame)]
            if len(mag_frames) != 1:
                raise ValueError(
                    "fin['Mag'] must contain exactly one DataFrame. "
                    f"Found {len(mag_frames)} frames: {[k for k,_ in mag_frames]}. "
                    "Fix the interval pickle to store a single MAG DataFrame under fin['Mag']."
                )
            mag = mag_frames[0][1]

        if isinstance(par, dict):
            par_frames = [(k, v) for k, v in par.items() if isinstance(v, pd.DataFrame)]
            if len(par_frames) != 1:
                raise ValueError(
                    "fin['Par'] must contain exactly one DataFrame. "
                    f"Found {len(par_frames)} frames: {[k for k,_ in par_frames]}. "
                    "Fix the interval pickle to store a single PAR DataFrame under fin['Par']."
                )
            par = par_frames[0][1]

        if isinstance(mag, pd.DataFrame) and isinstance(par, pd.DataFrame):
            return _to_utc_index(mag, "Mag"), _to_utc_index(par, "Par")

    # Strict fallback: scan all nested frames and require uniqueness.
    frames: List[pd.DataFrame] = []
    for v in fin.values():
        if isinstance(v, pd.DataFrame):
            frames.append(v)
        elif isinstance(v, dict):
            frames.extend([vv for vv in v.values() if isinstance(vv, pd.DataFrame)])

    if not frames:
        raise ValueError(f"fin dict contains no DataFrames. Keys={list(fin.keys())}")

    mag_hits, par_hits = [], []
    for df in frames:
        d = _to_utc_index(df, "candidate")
        if any(c in d.columns for c in BR_CANDIDATES):
            mag_hits.append(d)
        if any(c in d.columns for c in VR_CANDIDATES) and any(c in d.columns for c in NP_CANDIDATES):
            par_hits.append(d)

    if len(mag_hits) != 1 or len(par_hits) != 1:
        raise ValueError(
            "Could not uniquely identify MAG/PAR frames in fin. "
            f"MAG hits={len(mag_hits)}, PAR hits={len(par_hits)}. "
            "Fix the interval pickle to include fin['Mag'] and fin['Par'], or pass a single DataFrame with Br/Vr/Np."
        )

    return mag_hits[0], par_hits[0]


def _select_unique_col(df: pd.DataFrame, candidates: Sequence[str], label: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Explicit {label} column {explicit!r} not in DataFrame.")
        return str(explicit)

    hits = [c for c in candidates if c in df.columns]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise KeyError(f"Missing {label}. Candidates={list(candidates)}. Available={list(df.columns)[:120]}...")
    raise KeyError(f"Ambiguous {label} column selection: matches={hits}. Pass an explicit column name.")


def load_interval_inputs(*, root_dir: Union[str, Path], sc: str, which_int: int) -> Dict[str, Any]:
    """Deterministic interval loading via MHDTurbPy's ``general_functions.load_files``."""
    root_dir = Path(root_dir)
    load_path = root_dir / "examples" / "downloaded_intervals" / str(sc)

    finnames = func.load_files(load_path, "final.pkl")
    gennames = func.load_files(load_path, "general.pkl")
    signames = func.load_files(load_path, "sig_c_sig_r.pkl")

    if which_int < 0 or which_int >= len(finnames):
        raise IndexError(f"which_int={which_int} out of range for {sc}. Available intervals: {len(finnames)}")

    fin = pd.read_pickle(finnames[which_int])
    gen = pd.read_pickle(gennames[which_int]) if len(gennames) else None
    sig = pd.read_pickle(signames[which_int]) if len(signames) else None
    if isinstance(sig, pd.DataFrame):
        sig = _to_utc_index(sig, "sig")

    return {
        "fin": fin,
        "gen": gen,
        "sig": sig if isinstance(sig, pd.DataFrame) else None,
        "paths": {
            "final": Path(finnames[which_int]),
            "general": Path(gennames[which_int]) if len(gennames) else None,
            "sig": Path(signames[which_int]) if len(signames) else None,
        },
    }


def _resample(df: pd.DataFrame, cadence: str, *, agg: Mapping[str, str]) -> pd.DataFrame:
    if df.columns.has_duplicates:
        dups = sorted({c for c in df.columns[df.columns.duplicated()].tolist()})
        raise ValueError(f"Duplicate column labels in input: {dups}")
    d = df.select_dtypes(include=[np.number]).copy()
    if d.empty:
        raise ValueError("No numeric columns available to resample.")

    # Build a stable aggregation map
    agg_map: Dict[str, str] = {}
    for c in d.columns:
        agg_map[c] = agg.get(c, "mean")

    out = d.resample(cadence).agg(agg_map)
    out = _to_utc_index(out, "resampled")
    return out


def build_cadence_dataframe(
    *,
    fin: object,
    sig: Optional[pd.DataFrame],
    cadence: str,
    plot_vars: Sequence[str],
    br_col: Optional[str] = None,
    vr_col: Optional[str] = None,
    np_col: Optional[str] = None,
    join: str = "inner",
    input_units: Optional[Mapping[str, u.Unit]] = None,
) -> pd.DataFrame:
    """Build the canonical cadence DataFrame used for mapping and plotting."""

    mag, par = _extract_mag_par(fin)

    br = _select_unique_col(mag, BR_CANDIDATES, "Br", br_col)
    vr = _select_unique_col(par, VR_CANDIDATES, "Vr", vr_col)
    np_ = _select_unique_col(par, NP_CANDIDATES, "Np", np_col)

    requested = [v for v in plot_vars if v not in ("polarity", "phi_src", "lat_src")]
    keep_mag = [br] + [c for c in requested if c in mag.columns and c != br]
    keep_par = [vr, np_] + [c for c in requested if c in par.columns and c not in (vr, np_)]

    # Resampling policy: background-flow quantities should be robust.
    agg_mag = {br: "mean"}
    agg_par = {vr: "median", np_: "median"}

    mag_c = _resample(mag[keep_mag], cadence, agg=agg_mag).rename(columns={br: "Br"})
    par_c = _resample(par[keep_par], cadence, agg=agg_par).rename(columns={vr: "Vr", np_: "Np"})

    if join not in ("inner", "outer"):
        raise ValueError("join must be 'inner' or 'outer'")

    data = mag_c.join(par_c, how=join)
    data = _to_utc_index(data, "cadence data")

    # Optionally join sigma diagnostics (dimensionless)
    if isinstance(sig, pd.DataFrame) and not sig.empty:
        sig_c = _resample(sig, cadence, agg={})
        data = data.join(sig_c, how=join)

    if len(data.columns) != len(set(data.columns)):
        dup = [c for c in data.columns if list(data.columns).count(c) > 1]
        raise ValueError(f"Duplicate columns after merge: {sorted(set(dup))}")

    # Attach the explicit unit contract
    iu = dict(default_input_units())
    if input_units is not None:
        iu.update({str(k): u.Unit(v) for k, v in dict(input_units).items()})

    attach_units(data, {"Br": iu["Br"], "Vr": iu["Vr"], "Np": iu["Np"]})

    # Dimensionless columns (if present)
    for c in ("sigma_c", "sigma_r"):
        if c in data.columns:
            attach_units(data, {c: u.one})

    return data


def _rolling_median(s: pd.Series, window: str) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).median()


def _rolling_mad(s: pd.Series, window: str) -> pd.Series:
    med = _rolling_median(s, window)
    mad = _rolling_median((s - med).abs(), window)
    return 1.4826 * mad


def _compute_background_speed(
    data: pd.DataFrame,
    *,
    vr_bg_window: str,
    vr_sigma_window: str,
    sigma_rel: float,
    sigma_abs: u.Quantity,
    v_min: u.Quantity,
    v_fallback: u.Quantity,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Compute Vr_bg and sigma_Vr on the canonical cadence index.

    Design goals (explicit)
    -----------------------
    - Vr_bg is a background speed by construction (rolling median).
    - Wave-scale contamination is suppressed by the windowing choice.
    - Invalid or non-physical Vr samples trigger a *transparent* fallback.
    - sigma_Vr combines MAD-scale variability with systematic components and is inflated
      on fallback samples (so downstream tau/phi uncertainties reflect degraded inputs).

    Returns
    -------
    Vr_bg : pd.Series
    sigma_Vr : pd.Series
    fallback_mask : np.ndarray[bool]
        True where raw Vr was invalid/non-physical and the fallback policy was used.
    """

    Vr_raw = pd.to_numeric(data["Vr"], errors="coerce")
    vmin_kms = float(u.Quantity(v_min).to_value(u.km / u.s))
    vfb_kms = float(u.Quantity(v_fallback).to_value(u.km / u.s))

    vr_arr = Vr_raw.to_numpy(dtype=float)
    fallback_mask = (~np.isfinite(vr_arr)) | (vr_arr <= 0.0)

    # Replace invalid samples before smoothing so Vr_bg remains defined.
    Vr_fill = Vr_raw.copy()
    Vr_fill[fallback_mask] = vfb_kms

    Vr_bg = _rolling_median(Vr_fill, vr_bg_window)

    # Enforce a physically meaningful floor.
    Vr_bg = Vr_bg.clip(lower=vmin_kms)

    # Variability estimate around the background.
    resid = Vr_fill - Vr_bg
    sig_mad = _rolling_mad(resid, vr_sigma_window)

    sys = float(max(0.0, sigma_rel)) * Vr_bg.abs()
    abs_term = float(u.Quantity(sigma_abs).to_value(u.km / u.s))

    sigma = np.sqrt(sig_mad.to_numpy(dtype=float) ** 2 + sys.to_numpy(dtype=float) ** 2 + abs_term ** 2)
    sigma = pd.Series(sigma, index=data.index)

    # Inflate uncertainty on fallback points (cannot trust instantaneous Vr).
    if np.any(fallback_mask):
        infl = 0.5 * Vr_bg.to_numpy(dtype=float) + abs_term
        sigma_vals = sigma.to_numpy(dtype=float)
        sigma_vals[fallback_mask] = np.maximum(sigma_vals[fallback_mask], infl[fallback_mask])
        sigma = pd.Series(sigma_vals, index=data.index)

    return Vr_bg, sigma, fallback_mask


def _derive_physical_diagnostics(data: pd.DataFrame, *, need: set[str]) -> None:
    """Add derived diagnostics requested by the plotting/output contract.

    Only computes what is explicitly needed. This keeps the pipeline auditable and avoids
    accidental hard dependencies on plasma columns that are irrelevant for a given run.
    """

    need = set(map(str, need))

    # Polarity (dimensionless): sign(Br). Treat exact zeros as missing.
    if "polarity" in need and "polarity" not in data.columns:
        br_val = pd.to_numeric(data["Br"], errors="coerce").to_numpy(dtype=float)
        pol = np.sign(br_val)
        pol[~np.isfinite(br_val)] = np.nan
        pol[pol == 0.0] = np.nan
        data["polarity"] = pol
        attach_units(data, {"polarity": u.one})

    # The remaining diagnostics require r_sc
    if "r_sc" not in data.columns:
        return

    r_sc = q_from_df(data, "r_sc").to(u.m)

    if ("Br_r2" in need) and ("Br_r2" not in data.columns):
        Br = q_from_df(data, "Br").to(u.T)
        Br_r2 = Br * r_sc ** 2  # [T m^2] == [Wb]
        data["Br_r2"] = to_value(Br_r2, u.Wb)
        attach_units(data, {"Br_r2": u.Wb})

    if ("P_ram" in need) and ("P_ram" not in data.columns):
        Vr = q_from_df(data, "Vr").to(u.m / u.s)
        Np = q_from_df(data, "Np").to(u.m ** -3)
        P_ram = const.m_p * Np * Vr ** 2
        data["P_ram"] = to_value(P_ram, u.Pa)
        attach_units(data, {"P_ram": u.Pa})

def _summary_text(meta: Dict[str, Any]) -> str:
    """Return a compact diagnostics string that is TeX-safe.

    Users often have ``text.usetex=True`` globally. Literal ``%`` or unicode
    symbols can crash LaTeX rendering. We therefore place percent values in
    math-mode and use ``$R_\odot$`` for solar radii.
    """

    def _tex_escape(s: str) -> str:
        # Minimal escape set for common user-facing strings.
        return (
            s.replace("\\", r"\\")
             .replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&")
             .replace("#", r"\#")
        )

    lines = []
    if "model" in meta:
        lines.append(f"model: {_tex_escape(str(meta['model']))}")
    if "r_ss_Rsun" in meta:
        lines.append(f"r_ss: {float(meta['r_ss_Rsun']):.2f} $R_\\odot$")
    if "omega_deg_per_day" in meta:
        lines.append(f"$\\Omega$: {float(meta['omega_deg_per_day']):.4g} deg/day")
    if "ephem_step" in meta:
        lines.append(f"ephem: {_tex_escape(str(meta['ephem_step']))}")
    if "frame3d" in meta:
        lines.append(f"frame: {_tex_escape(str(meta['frame3d']))}")

    # IMPORTANT: do NOT use ``\\%`` here.
    # Under TeX rendering, ``\\`` is a newline command; followed by ``%`` it
    # becomes a comment delimiter and can break parsing.
    # Use the literal TeX percent escape ``\%`` (single backslash) and avoid
    # wrapping these values in math-mode.
    if "fallback_fraction" in meta:
        lines.append(f"fallback: {100*float(meta['fallback_fraction']):.1f}\\%")
    if "masked_fraction" in meta:
        lines.append(f"masked: {100*float(meta['masked_fraction']):.1f}\\%")
    if "tau_median_hr" in meta:
        lines.append(f"$\\tau_{{\\rm med}}$: {float(meta['tau_median_hr']):.2f} h")
    if "sigma_phi_p84_deg" in meta:
        lines.append(f"$\\sigma_\\phi(84)$: {float(meta['sigma_phi_p84_deg']):.2f} deg")
    if "r_sc_median_Rsun" in meta:
        rs = float(meta["r_sc_median_Rsun"])
        lines.append(f"$r_{{\\rm sc,med}}$: {rs:.2f} $R_\\odot$")

    # Profile sanity flags (non-ballistic models must not silently degenerate)
    if bool(meta.get("profile_fallback_used", False)):
        lines.append("profile: FALLBACK")
    if meta.get("profile_degenerate", False):
        lines.append("profile: DEGENERATE")

    # Final TeX-safety guard: convert accidental \\% to \% (newline + comment is fatal).
    for ii, ln in enumerate(lines):
        if "\\\\%" in ln:
            lines[ii] = ln.replace("\\\\%", "\\%")

    return "\n".join(lines)


def _compute_3d_cartesian(
    data: pd.DataFrame,
    *,
    r_ss: u.Quantity,
    frame3d: str,
    observer: str = "earth",
) -> None:
    """Compute AU Cartesian coordinates for spacecraft and source-surface points."""

    try:
        from astropy.time import Time
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import get_body_heliographic_stonyhurst
        from sunpy.coordinates.frames import (
            HeliographicCarrington,
            HeliocentricEarthEcliptic,
            HeliocentricInertial,
        )
        import astropy.units as uu
    except Exception as e:
        raise RuntimeError("3D outputs require sunpy+astropy.") from e

    frame3d = str(frame3d).upper().strip()
    if frame3d not in {"HEE", "HCI"}:
        raise ValueError("frame3d must be 'HEE' or 'HCI'")

    t_index = pd.DatetimeIndex(data.index)
    obstime = Time(t_index.to_pydatetime())

    if str(observer).lower().strip() != "earth":
        raise ValueError("Only observer='earth' is supported in this minimal implementation.")

    earth_obs = get_body_heliographic_stonyhurst("earth", obstime)
    hgc = HeliographicCarrington(obstime=obstime, observer=earth_obs)

    lon = (pd.to_numeric(data["phi_src"], errors="coerce").to_numpy(dtype=float) * uu.deg)
    lat = (pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float) * uu.deg)

    rss_au = r_ss.to_value(u.AU)

    fp_ss = SkyCoord(lon=lon, lat=lat, radius=(rss_au * uu.AU), frame=hgc)

    if frame3d == "HEE":
        tf = HeliocentricEarthEcliptic(obstime=obstime)
    else:
        tf = HeliocentricInertial(obstime=obstime)

    ss = fp_ss.transform_to(tf)
    data["ss_x_au"] = ss.cartesian.x.to_value(uu.AU)
    data["ss_y_au"] = ss.cartesian.y.to_value(uu.AU)
    data["ss_z_au"] = ss.cartesian.z.to_value(uu.AU)

    # Spacecraft Cartesian coordinates in target frame (needed for link lines)
    if {"hee_x_au", "hee_y_au", "hee_z_au"}.issubset(data.columns):
        xsc = pd.to_numeric(data["hee_x_au"], errors="coerce").to_numpy(dtype=float)
        ysc = pd.to_numeric(data["hee_y_au"], errors="coerce").to_numpy(dtype=float)
        zsc = pd.to_numeric(data["hee_z_au"], errors="coerce").to_numpy(dtype=float)

        if frame3d == "HEE":
            data["sc_x_au"], data["sc_y_au"], data["sc_z_au"] = xsc, ysc, zsc
        else:
            rep = SkyCoord(
                x=xsc * uu.AU,
                y=ysc * uu.AU,
                z=zsc * uu.AU,
                frame=HeliocentricEarthEcliptic(obstime=obstime),
                representation_type="cartesian",
            )
            rep2 = rep.transform_to(HeliocentricInertial(obstime=obstime))
            data["sc_x_au"] = rep2.cartesian.x.to_value(uu.AU)
            data["sc_y_au"] = rep2.cartesian.y.to_value(uu.AU)
            data["sc_z_au"] = rep2.cartesian.z.to_value(uu.AU)
    else:
        raise ValueError("3D plotting requires HEE spacecraft Cartesian columns hee_x_au/hee_y_au/hee_z_au (set include_hee=True in ephemeris).")

    # Optional: Cartesian endpoints for longitude-interval uncertainty (p16/p84) if present.
    # This keeps the uncertainty visualization frame-consistent.
    if {"phi_src_p16", "phi_src_p84", "lat_src"}.issubset(data.columns):
        lon_lo = (pd.to_numeric(data["phi_src_p16"], errors="coerce").to_numpy(dtype=float) * uu.deg)
        lon_hi = (pd.to_numeric(data["phi_src_p84"], errors="coerce").to_numpy(dtype=float) * uu.deg)
        lat_u = (pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float) * uu.deg)

        fp_lo = SkyCoord(lon=lon_lo, lat=lat_u, radius=(rss_au * uu.AU), frame=hgc)
        fp_hi = SkyCoord(lon=lon_hi, lat=lat_u, radius=(rss_au * uu.AU), frame=hgc)

        lo_tf = fp_lo.transform_to(tf)
        hi_tf = fp_hi.transform_to(tf)

        data["ss_p16_x_au"] = lo_tf.cartesian.x.to_value(uu.AU)
        data["ss_p16_y_au"] = lo_tf.cartesian.y.to_value(uu.AU)
        data["ss_p16_z_au"] = lo_tf.cartesian.z.to_value(uu.AU)

        data["ss_p84_x_au"] = hi_tf.cartesian.x.to_value(uu.AU)
        data["ss_p84_y_au"] = hi_tf.cartesian.y.to_value(uu.AU)
        data["ss_p84_z_au"] = hi_tf.cartesian.z.to_value(uu.AU)






def backmap_interval(
    *,
    root_dir: Union[str, Path],
    sc: str,
    which_int: int,
    method: str = "ballistic_bg",
    cadence: str = "60min",
    r_ss: u.Quantity = 2.5 * u.R_sun,
    omega: u.Quantity = 14.1844 * u.deg / u.day,
    phi_sign: int = +1,
    ephem_step: str = "1h",
    ephem_observer: str = "earth",
    plot_vars: Sequence[str] = ("polarity", "Vr_bg", "P_ram", "sigma_c"),
    var_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    plot_percentiles: Tuple[float, float] = (2.0, 98.0),
    plot_3d: bool = False,
    plot_3d_var: Optional[str] = None,
    plot_3d_vars: Optional[Sequence[str]] = None,
    plot_3d_ncols: int = 2,
    plot_3d_camera: str = "iso",
    frame3d: str = "HEE",
    show_uncertainty: bool = True,
    show: bool = False,
    vr_bg_window: str = "6h",
    vr_sigma_window: str = "6h",
    sigma_rel: float = 0.15,
    sigma_abs: u.Quantity = 30 * u.km / u.s,
    v_min: u.Quantity = 50 * u.km / u.s,
    v_fallback: u.Quantity = 400 * u.km / u.s,
    input_units: Optional[Mapping[str, u.Unit]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    br_col: Optional[str] = None,
    vr_col: Optional[str] = None,
    np_col: Optional[str] = None,
    join: str = "inner",
    size_by: str = "P_ram",
    figsize_2d: Optional[Tuple[float, float]] = None,
    figsize_3d: Optional[Tuple[int, int]] = None,
    outdir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Backmap one MHDTurbPy interval to the source surface.

    Returns
    -------
    dict with keys: data, meta, files
    """

    root_dir = Path(root_dir)
    inp = load_interval_inputs(root_dir=root_dir, sc=sc, which_int=which_int)

    fin = inp["fin"]
    sig = inp.get("sig", None)

    # Output directory policy
    interval_dir = Path(inp["paths"]["final"]).parent
    method_tag = str(method).strip().lower()
    out_base = Path(outdir) if outdir is not None else (interval_dir / "back_mapping" / method_tag)
    out_base.mkdir(parents=True, exist_ok=True)

    # Build canonical cadence DataFrame
    data = build_cadence_dataframe(
        fin=fin,
        sig=sig,
        cadence=cadence,
        plot_vars=plot_vars,
        br_col=br_col,
        vr_col=vr_col,
        np_col=np_col,
        join=join,
        input_units=input_units,
    )

    # Ephemeris (Carrington; observer explicit)
    cache_file = out_base / "ephemeris_cache.pkl"
    eph = get_ephemeris_hgc(target=sc, times=data.index, step=ephem_step, observer=ephem_observer, cache_file=cache_file, include_hee=True)
    eph_i = interp_ephemeris_to_index(eph.df, data.index, circular_cols=("phi_sc_deg",))

    data["phi_sc"] = eph_i["phi_sc_deg"].to_numpy(dtype=float)
    data["lat_sc"] = eph_i["lat_sc_deg"].to_numpy(dtype=float)
    data["r_sc"] = eph_i["r_sc_au"].to_numpy(dtype=float)
    attach_units(data, {"phi_sc": u.deg, "lat_sc": u.deg, "r_sc": u.AU})

    for c in ("hee_x_au", "hee_y_au", "hee_z_au"):
        if c in eph_i.columns:
            data[c] = eph_i[c].to_numpy(dtype=float)

    # Background speed + uncertainty model
    Vr_bg, sigma_Vr, fb_vr = _compute_background_speed(
        data,
        vr_bg_window=vr_bg_window,
        vr_sigma_window=vr_sigma_window,
        sigma_rel=sigma_rel,
        sigma_abs=sigma_abs,
        v_min=v_min,
        v_fallback=v_fallback,
    )

    data["Vr_bg"] = Vr_bg.to_numpy(dtype=float)
    attach_units(data, {"Vr_bg": q_from_df(data, "Vr").unit})

    data["sigma_Vr"] = sigma_Vr.to_numpy(dtype=float)
    attach_units(data, {"sigma_Vr": q_from_df(data, "Vr").unit})

    # Travel time model
    model: TravelTimeModel
    if str(method_tag) in {"constant", "const"}:
        V0 = u.Quantity(model_kwargs.get("V0")) if model_kwargs else np.nan * (u.km / u.s)
        if not np.isfinite(V0.to_value(u.km / u.s)):
            V0 = np.nanmedian(data["Vr_bg"].to_numpy(dtype=float)) * (u.km / u.s)
        model = build_model("constant", model_kwargs={"V0": V0})
    else:
        model = build_model(method_tag, model_kwargs=model_kwargs)

    r_sc_q = q_from_df(data, "r_sc")
    V_bg_q = q_from_df(data, "Vr_bg")

    # Base evaluation (deterministic)
    tt = model.evaluate(r_sc=r_sc_q, V_bg=V_bg_q, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback)
    tau = tt.tau

    # --------------------------------------------------------------
    # Uncertainty propagation
    #
    # Default: treat Vr_bg uncertainty as Gaussian with sigma_Vr.
    # For the hybrid Parker profile family (paper-style heuristic),
    # allow a shape-uncertainty scan over rs to produce tau/phi percentiles.
    # --------------------------------------------------------------
    tau_samples_s = None  # shape (n_rs, n_time) in seconds
    phi_samples_deg = None  # shape (n_rs, n_time) in degrees

    use_rs_family = bool(
        str(method_tag) in {"parker_scaled", "hybrid_parker"}
        and model_kwargs
        and any(k in model_kwargs for k in ("rs_samples", "rs_min", "rs_max", "n_rs"))
    )

    if use_rs_family:
        # Parse rs samples (in R_sun unless already a Quantity)
        switch = str(model_kwargs.get("switch", "match"))
        n_grid = int(model_kwargs.get("n_grid", 8192))

        rs_mode = str(model_kwargs.get("rs_sample_mode", "rs_uniform")).strip().lower()
        rs_samples = model_kwargs.get("rs_samples", None)

        if rs_samples is None:
            rs_min = u.Quantity(model_kwargs.get("rs_min", 0.1 * u.R_sun)).to(u.R_sun)
            rs_max = u.Quantity(model_kwargs.get("rs_max", 3.0 * u.R_sun)).to(u.R_sun)
            n_rs = int(model_kwargs.get("n_rs", 20))

            if rs_mode == "tau_uniform":
                # Choose rs so that a representative travel time spans uniformly between
                # the ballistic and most-accelerating cases (heuristic but stable).
                rs_dense = np.linspace(rs_min.to_value(u.R_sun), rs_max.to_value(u.R_sun), num=max(200, 10 * n_rs)) * u.R_sun

                r_sc_rep = float(np.nanmedian(r_sc_q.to_value(u.R_sun))) * u.R_sun
                V_rep = float(np.nanmedian(V_bg_q.to_value(u.km / u.s))) * (u.km / u.s)

                tau_dense = np.full(len(rs_dense), np.nan, dtype=float)
                for j, rs_j in enumerate(rs_dense):
                    m_j = build_model("hybrid_parker", model_kwargs={"rs": rs_j, "n_grid": n_grid, "switch": switch})
                    tt_j = m_j.evaluate(r_sc=r_sc_rep, V_bg=V_rep, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback)
                    tau_dense[j] = float(np.atleast_1d(tt_j.tau.to_value(u.s))[0])

                # Target travel times: uniform between ballistic and max-accel (largest tau)
                tau_ball = float(((r_sc_rep - u.Quantity(r_ss)).to(u.m) / V_rep.to(u.m / u.s)).to_value(u.s))
                ok = np.isfinite(tau_dense)
                if ok.sum() < 5:
                    rs_samples = np.linspace(rs_min.to_value(u.R_sun), rs_max.to_value(u.R_sun), n_rs) * u.R_sun
                    rs_mode = "rs_uniform_fallback"
                else:
                    tau_ok = tau_dense[ok]
                    rs_ok = rs_dense.to_value(u.R_sun)[ok]

                    # Enforce monotonic interpolation (sort by tau)
                    srt = np.argsort(tau_ok)
                    tau_ok = tau_ok[srt]
                    rs_ok = rs_ok[srt]

                    tau_hi = float(np.nanmax(tau_ok))
                    tau_targets = np.linspace(min(tau_ball, tau_hi), tau_hi, n_rs)
                    rs_samples = np.interp(tau_targets, tau_ok, rs_ok) * u.R_sun
            else:
                rs_samples = np.linspace(rs_min.to_value(u.R_sun), rs_max.to_value(u.R_sun), n_rs) * u.R_sun
        else:
            if isinstance(rs_samples, u.Quantity):
                rs_samples = rs_samples.to(u.R_sun)
            else:
                rs_samples = np.asarray(rs_samples, dtype=float) * u.R_sun

# Evaluate tau (and mapped phi) for each rs profile
        tau_stack = []
        phi_stack = []
        phi_sc = data["phi_sc"].to_numpy(dtype=float)
        lat_sc = data["lat_sc"].to_numpy(dtype=float)

        for rs in rs_samples:
            m_rs = build_model("hybrid_parker", model_kwargs={"rs": rs, "n_grid": n_grid, "switch": switch})
            tt_rs = m_rs.evaluate(r_sc=r_sc_q, V_bg=V_bg_q, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback)
            tau_rs = tt_rs.tau

            tau_stack.append(tau_rs.to_value(u.s))

            # Map phi for each sample (lat is unchanged in our baseline)
            map_rs = map_to_source_surface(
                phi_sc_deg=phi_sc,
                lat_sc_deg=lat_sc,
                tau=tau_rs,
                omega=omega,
                phi_sign=phi_sign,
            )
            phi_stack.append(map_rs.phi_src_deg)

        tau_samples_s = np.stack(tau_stack, axis=0)  # seconds
        phi_samples_deg = np.stack(phi_stack, axis=0)

        # Central + percentiles
        tau = (np.nanmedian(tau_samples_s, axis=0) * u.s).to(u.s)
        tau_p16 = (np.nanpercentile(tau_samples_s, 16.0, axis=0) * u.s).to(u.s)
        tau_p84 = (np.nanpercentile(tau_samples_s, 84.0, axis=0) * u.s).to(u.s)

        # Phi percentiles on the circle
        phi_src = circ_percentile_deg(phi_samples_deg, 50.0, axis=0)
        phi_src_p16 = circ_percentile_deg(phi_samples_deg, 16.0, axis=0)
        phi_src_p84 = circ_percentile_deg(phi_samples_deg, 84.0, axis=0)

        sigma_tau = 0.5 * (tau_p84 - tau_p16)

        # Surface this explicitly for diagnostics/fig annotation downstream.
        tt.meta.setdefault("diagnostics", {})
        tt.meta["diagnostics"].update(
            {
                "rs_family_enabled": True,
                "rs_min_Rsun": float(np.nanmin(u.Quantity(rs_samples).to_value(u.R_sun))),
                "rs_max_Rsun": float(np.nanmax(u.Quantity(rs_samples).to_value(u.R_sun))),
                "rs_n": int(len(rs_samples)),
                "rs_switch": str(switch),
                "rs_sample_mode": str(rs_mode),
            }
        )

    else:
        # Uncertainty propagation: treat V_bg uncertainty as Gaussian with sigma_Vr
        V_sig_q = q_from_df(data, "sigma_Vr")

        V_p16 = (V_bg_q - V_sig_q).to(u.km / u.s)
        V_p84 = (V_bg_q + V_sig_q).to(u.km / u.s)

        # Floor to keep times positive and avoid hidden singularities
        V_p16 = np.maximum(V_p16.to_value(u.km / u.s), u.Quantity(v_min).to_value(u.km / u.s)) * (u.km / u.s)
        V_p84 = np.maximum(V_p84.to_value(u.km / u.s), u.Quantity(v_min).to_value(u.km / u.s)) * (u.km / u.s)

        tau_p16 = model.evaluate(r_sc=r_sc_q, V_bg=V_p84, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback).tau
        tau_p84 = model.evaluate(r_sc=r_sc_q, V_bg=V_p16, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback).tau

        sigma_tau = 0.5 * (tau_p84 - tau_p16)

    # Valid mask (recomputed after uncertainty logic)
    valid = (
        np.isfinite(tau.to_value(u.s))
        & np.isfinite(data["phi_sc"].to_numpy(dtype=float))
        & np.isfinite(data["lat_sc"].to_numpy(dtype=float))
    )

    fallback_fraction = float(np.nanmean(fb_vr[valid])) if valid.any() else 1.0
    masked_fraction = 1.0 - float(np.nanmean(valid)) if len(valid) else 1.0

    # Mapping
    phi_sc_arr = data["phi_sc"].to_numpy(dtype=float)
    lat_sc_arr = data["lat_sc"].to_numpy(dtype=float)

    # Central (median) mapping
    map0 = map_to_source_surface(
        phi_sc_deg=phi_sc_arr,
        lat_sc_deg=lat_sc_arr,
        tau=tau,
        omega=omega,
        phi_sign=phi_sign,
    )

    data["lat_src"] = map0.lat_src_deg

    if phi_samples_deg is not None:
        # Percentiles computed directly from the rs-family mapping (circular-safe).
        data["phi_src"] = phi_src
        data["phi_src_p16"] = phi_src_p16
        data["phi_src_p84"] = phi_src_p84
    else:
        map16 = map_to_source_surface(
            phi_sc_deg=phi_sc_arr,
            lat_sc_deg=lat_sc_arr,
            tau=tau_p16,
            omega=omega,
            phi_sign=phi_sign,
        )

        map84 = map_to_source_surface(
            phi_sc_deg=phi_sc_arr,
            lat_sc_deg=lat_sc_arr,
            tau=tau_p84,
            omega=omega,
            phi_sign=phi_sign,
        )

        data["phi_src"] = map0.phi_src_deg
        data["phi_src_p16"] = map16.phi_src_deg
        data["phi_src_p84"] = map84.phi_src_deg

    attach_units(data, {"phi_src": u.deg, "lat_src": u.deg, "phi_src_p16": u.deg, "phi_src_p84": u.deg})

    # Tau in hours for output clarity
    data["tau"] = tau.to_value(u.hour)
    data["tau_p16"] = tau_p16.to_value(u.hour)
    data["tau_p84"] = tau_p84.to_value(u.hour)
    data["sigma_tau"] = sigma_tau.to_value(u.hour)
    attach_units(data, {"tau": u.hour, "tau_p16": u.hour, "tau_p84": u.hour, "sigma_tau": u.hour})

    # Circular sigma_phi from the (p16,p84) interval
    data["sigma_phi"] = halfwidth_deg(data["phi_src_p16"].to_numpy(dtype=float), data["phi_src_p84"].to_numpy(dtype=float))
    attach_units(data, {"sigma_phi": u.deg})

    # Derived physical diagnostics (needs r_sc)
        # Derived diagnostics computed only when requested by the plotting/size contract
    need_diag: set[str] = set(plot_vars) | {str(size_by)}
    if plot_3d:
        vars3d = list(plot_3d_vars) if plot_3d_vars is not None else ([str(plot_3d_var)] if plot_3d_var is not None else list(plot_vars))
        need_diag |= set(map(str, vars3d))
    _derive_physical_diagnostics(data, need=need_diag)

    # Marker size for scatter maps
    if size_by in data.columns:
        data["marker_size"] = marker_sizes_from_metric(pd.to_numeric(data[size_by], errors="coerce"))
    else:
        data["marker_size"] = 30.0

    # Plotting
    specs = merge_var_specs(VAR_SPECS, var_specs)

    out_pkl = out_base / "backmap_timeseries.pkl"
    out_png = out_base / "source_surface_maps_2d.png"
    out_html = out_base / "source_surface_3d.html"
    out_meta = out_base / "meta.json"

    # Titles:
    #   - 2D should be clean (date/time only; no parameter soup).
    #   - 3D must explicitly include a radial distance in $R_\odot$.
    try:
        t0 = pd.to_datetime(data.index.min()).to_pydatetime()
        t1 = pd.to_datetime(data.index.max()).to_pydatetime()
        tspan = f"{t0:%Y-%m-%d %H:%M}-{t1:%H:%M} UTC"
    except Exception:
        tspan = ""

    title = f"{sc} | int {which_int}" + (f" | {tspan}" if tspan else "")

    r_sc_med_Rsun = (
        float(np.nanmedian(r_sc_q[valid].to_value(u.R_sun)))
        if valid.any()
        else float(np.nanmedian(r_sc_q.to_value(u.R_sun)))
    )
    title3d = (
        f"{title} | "
        f"$r_{{\\rm sc,med}}$={r_sc_med_Rsun:.1f} $R_\\odot$"
    )
    # ------------------------------------------------------------------
    # Preflight: strict plotting contract (fail fast, explicit errors)
    # ------------------------------------------------------------------
    def _require_columns(cols: Sequence[str], context: str) -> None:
        missing = [c for c in cols if c not in data.columns]
        if missing:
            avail = ", ".join(list(map(str, data.columns[:40]))) + (" ..." if len(data.columns) > 40 else "")
            raise ValueError(f"{context}: missing columns {missing}. Available columns: {avail}")

    # 2D variables must exist
    _require_columns(list(plot_vars), context="2D plotting")

    # 3D requires geometry + variables
    if plot_3d:
        if plot_3d_vars is not None:
            vars3d = list(plot_3d_vars)
        elif plot_3d_var is not None:
            # Accept a single variable name OR an iterable of names.
            if isinstance(plot_3d_var, (list, tuple)):
                vars3d = list(plot_3d_var)
            else:
                vars3d = [str(plot_3d_var)]
        else:
            vars3d = list(plot_vars)
        _require_columns(vars3d, context="3D plotting variables")
        if show_uncertainty:
            _require_columns(["phi_src_p16", "phi_src_p84"], context="3D uncertainty (longitude CI)")

    # ------------------------------------------------------------------
    # Velocity profile used for travel-time integration (mandatory 2D panel)
    # ------------------------------------------------------------------
    prof_png = None
    profile_panel = None
    diag_extra: Dict[str, Any] = {
        "profile_model": str(method_tag),
        "profile_ok": False,
        "profile_error": None,
        "profile_degenerate": None,
        "profile_fallback_used": False,
        "U_min_kms": None,
        "U_max_kms": None,
        "U_span_kms": None,
        "r_sc_profile_Rsun": None,
    }
    try:
        out_prof = out_base / "velocity_profile.png"

        if valid.any():
            r_sc_med = float(np.nanmedian(r_sc_q[valid].to_value(u.R_sun)))
            V_med = float(np.nanmedian(V_bg_q[valid].to_value(u.km / u.s)))
            V_sig_med = float(np.nanmedian(V_sig_q[valid].to_value(u.km / u.s)))
        else:
            r_sc_med = float(np.nanmedian(r_sc_q.to_value(u.R_sun)))
            V_med = float(np.nanmedian(V_bg_q.to_value(u.km / u.s)))
            V_sig_med = float(np.nanmedian(V_sig_q.to_value(u.km / u.s)))

        r_ss_R = float(u.Quantity(r_ss).to_value(u.R_sun))
        r_hi = max(r_ss_R * 1.02, r_sc_med)
        r_grid = np.geomspace(max(1.01, r_ss_R), r_hi, 420)

        v_min_profile = None
        if model_kwargs and ("v_min_profile" in model_kwargs):
            v_min_profile = u.Quantity(model_kwargs["v_min_profile"])

        U_med = model.speed_profile(
            r_grid=r_grid * u.R_sun,
            r_sc=r_sc_med * u.R_sun,
            V_bg=V_med * (u.km / u.s),
            r_ss=u.Quantity(r_ss),
            v_min=v_min_profile,
        )
        if U_med is not None:
            V_lo = max(float(u.Quantity(v_min).to_value(u.km / u.s)), V_med - V_sig_med)
            V_hi = V_med + V_sig_med
            U_lo = model.speed_profile(
                r_grid=r_grid * u.R_sun,
                r_sc=r_sc_med * u.R_sun,
                V_bg=V_lo * (u.km / u.s),
                r_ss=u.Quantity(r_ss),
                v_min=v_min_profile,
            )
            U_hi = model.speed_profile(
                r_grid=r_grid * u.R_sun,
                r_sc=r_sc_med * u.R_sun,
                V_bg=V_hi * (u.km / u.s),
                r_ss=u.Quantity(r_ss),
                v_min=v_min_profile,
            )

            profile_panel = {
                "r_grid_Rsun": r_grid,
                "U_med_kms": U_med.to_value(u.km / u.s),
                "U_lo_kms": (U_lo.to_value(u.km / u.s) if U_lo is not None else None),
                "U_hi_kms": (U_hi.to_value(u.km / u.s) if U_hi is not None else None),
                "r_ss_Rsun": float(r_ss_R),
                "r_sc_Rsun": float(r_sc_med),
            }
            # Diagnostics: detect degenerate/flat profiles (should not occur for accelerating models).
            try:
                uarr = np.asarray(profile_panel.get("U_med_kms", None), dtype=float)
                if uarr.size > 0 and np.isfinite(uarr).any():
                    umin = float(np.nanmin(uarr))
                    umax = float(np.nanmax(uarr))
                    span = float(umax - umin)
                else:
                    umin = umax = span = float('nan')
                diag_extra["profile_ok"] = True
                diag_extra["U_min_kms"] = umin
                diag_extra["U_max_kms"] = umax
                diag_extra["U_span_kms"] = span
                diag_extra["r_sc_profile_Rsun"] = float(r_sc_med)
                accel = str(method_tag).lower().strip() in {"parker_scaled", "exp_accel"}
                if accel and np.isfinite(span) and np.isfinite(V_med):
                    diag_extra["profile_degenerate"] = bool(span < max(1e-6, 1e-4 * float(V_med)))
                else:
                    diag_extra["profile_degenerate"] = False
            except Exception as _e_prof:
                diag_extra["profile_ok"] = False
                diag_extra["profile_error"] = f"profile_diagnostics_failed: {_e_prof}"


            # Keep a standalone diagnostic PNG as well.
            prof_title = r"$U(r)$ profile used for travel-time integration"
            prof_png, _ = plot_velocity_profile(
                out_png=out_prof,
                r_grid=r_grid,
                U_med=U_med.to_value(u.km / u.s),
                U_lo=(U_lo.to_value(u.km / u.s) if U_lo is not None else None),
                U_hi=(U_hi.to_value(u.km / u.s) if U_hi is not None else None),
                r_ss=r_ss_R,
                r_sc=r_sc_med,
                title=prof_title,
                show=show,
            )
    except Exception as _e_prof:
        prof_png = None
        profile_panel = None
        diag_extra["profile_ok"] = False
        diag_extra["profile_error"] = str(_e_prof)

    # Hard guard: the 2D figure must always contain a velocity-profile panel.
    if profile_panel is None:
        try:
            r_ss_R = float(u.Quantity(r_ss).to_value(u.R_sun))
            r_sc_med = float(r_sc_med_Rsun)
            r_hi = max(r_ss_R * 1.02, r_sc_med)
            r_grid = np.geomspace(max(1.01, r_ss_R), r_hi, 220)
            # fall back to the scalar background speed if present
            V_med = float(np.nanmedian(V_bg_q.to_value(u.km / u.s)))
            U = np.full_like(r_grid, V_med, dtype=float)
            profile_panel = {
                "r_grid_Rsun": r_grid,
                "U_med_kms": U,
                "U_lo_kms": None,
                "U_hi_kms": None,
                "r_ss_Rsun": float(r_ss_R),
                "r_sc_Rsun": float(r_sc_med),
            }
            diag_extra["profile_fallback_used"] = True
            diag_extra["profile_ok"] = True
            diag_extra["U_min_kms"] = float(np.nanmin(U)) if np.isfinite(U).any() else None
            diag_extra["U_max_kms"] = float(np.nanmax(U)) if np.isfinite(U).any() else None
            diag_extra["U_span_kms"] = 0.0
            diag_extra["r_sc_profile_Rsun"] = float(r_sc_med)
            accel = str(method_tag).lower().strip() in {"parker_scaled", "exp_accel"}
            diag_extra["profile_degenerate"] = bool(accel)
        except Exception:
            profile_panel = None

    # Summary box (TeX-safe; includes profile diagnostics)
    summary_box = _summary_text(
        {
            "model": str(method_tag),
            "r_ss_Rsun": float(u.Quantity(r_ss).to_value(u.R_sun)),
            "omega_deg_per_day": float(u.Quantity(omega).to_value(u.deg / u.day)),
            "ephem_step": str(ephem_step),
            "frame3d": str(frame3d),
            "fallback_fraction": float(fallback_fraction),
            "masked_fraction": float(masked_fraction),
            "r_sc_median_Rsun": float(r_sc_med_Rsun),
            "tau_median_hr": float(np.nanmedian(data.loc[valid, "tau"])) if valid.any() else np.nan,
            "sigma_phi_p84_deg": float(np.nanpercentile(data.loc[valid, "sigma_phi"], 84.0)) if valid.any() else np.nan,
            "profile_degenerate": diag_extra.get("profile_degenerate", None),
            "profile_fallback_used": bool(diag_extra.get("profile_fallback_used", False)),
        }
    )


    out_png_ret, fig2d = plot_source_surface_2d(
        data=data,
        out_png=out_png,
        plot_vars=list(plot_vars),
        var_specs=specs,
        percentiles=plot_percentiles,
        size_col="marker_size",
        show_uncertainty=bool(show_uncertainty),
        summary_box=summary_box,
        title=title,
        figsize=figsize_2d,
        show=show,
        profile_panel=profile_panel,
    )

    # 3D output (optional)
    fig3d = None
    if plot_3d:
        _compute_3d_cartesian(data, r_ss=u.Quantity(r_ss), frame3d=frame3d, observer=ephem_observer)
        # Geometry must exist after Cartesian conversion
        _require_columns(["ss_x_au", "ss_y_au", "ss_z_au"], context="3D geometry (source surface)")
        _require_columns(["sc_x_au", "sc_y_au", "sc_z_au"], context="3D geometry (spacecraft)")
        r_sun_au = const.R_sun.to_value(u.AU)
        r_ss_au = u.Quantity(r_ss).to_value(u.AU)
        w3d, h3d = 1700, 900
        if figsize_3d is not None:
            try:
                w3d, h3d = int(figsize_3d[0]), int(figsize_3d[1])
            except Exception:
                w3d, h3d = 1700, 900

        out_html_ret, fig3d = plot_source_surface_3d(
            data=data,
            out_html=out_html,
            var_specs=specs,
            plot_vars=list(plot_3d_vars)
            if plot_3d_vars is not None
            else ([str(plot_3d_var)] if plot_3d_var is not None else list(plot_vars)),
            ncols_vars=int(plot_3d_ncols),
            r_ss_au=float(r_ss_au),
            r_sun_au=float(r_sun_au),
            r_sc_med_rsun=float(r_sc_med_Rsun),
            frame3d=frame3d,
            percentiles=plot_percentiles,
            show_uncertainty_arcs=bool(show_uncertainty),
            # requested: links + RTN axes + camera sync across all sub-panels
            show_links=True,
            link_count=12,
            show_rtn_axes=True,
            rtn_axis_frac=0.22,
            sync_cameras=True,
            camera=str(plot_3d_camera),
            title=title3d,
            width=w3d,
            height=h3d,
            draw_panel_boxes=False,
            show=show,
        )

    # Build meta
    meta: Dict[str, Any] = {
        "sc": str(sc),
        "which_int": int(which_int),
        "cadence": str(cadence),
        "method": str(method_tag),
        "r_ss_Rsun": float(u.Quantity(r_ss).to_value(u.R_sun)),
        "omega_deg_per_day": float(u.Quantity(omega).to_value(u.deg / u.day)),
        "phi_sign": int(phi_sign),
        "ephemeris": dict(eph.meta),
        "plot": {
            "plot_vars": list(plot_vars),
            "plot_percentiles": list(plot_percentiles),
            "plot_3d": bool(plot_3d),
            "plot_3d_var": (str(plot_3d_var) if plot_3d_var is not None else None) if plot_3d else None,
            "plot_3d_vars": list(plot_3d_vars) if (plot_3d and plot_3d_vars is not None) else (list(plot_vars) if plot_3d else None),
            "plot_3d_ncols": int(plot_3d_ncols) if plot_3d else None,
            "frame3d": str(frame3d) if plot_3d else None,
            "show_uncertainty": bool(show_uncertainty),
        },
        "uncertainty_model": {
            "vr_bg_window": str(vr_bg_window),
            "vr_sigma_window": str(vr_sigma_window),
            "sigma_rel": float(sigma_rel),
            "sigma_abs_kms": float(u.Quantity(sigma_abs).to_value(u.km / u.s)),
            "v_min_kms": float(u.Quantity(v_min).to_value(u.km / u.s)),
            "v_fallback_kms": float(u.Quantity(v_fallback).to_value(u.km / u.s)),
        },
        "diagnostics": {
            "fallback_fraction": fallback_fraction,
            "masked_fraction": masked_fraction,
            "r_sc_median_Rsun": float(np.nanmedian(r_sc_q[valid].to_value(u.R_sun))) if valid.any() else float(np.nanmedian(r_sc_q.to_value(u.R_sun))),
            "r_sc_median_AU": float(np.nanmedian(r_sc_q[valid].to_value(u.AU))) if valid.any() else float(np.nanmedian(r_sc_q.to_value(u.AU))),
            "tau_median_hr": float(np.nanmedian(data.loc[valid, "tau"])) if valid.any() else np.nan,
            "tau_p16_hr": float(np.nanpercentile(data.loc[valid, "tau"], 16.0)) if valid.any() else np.nan,
            "tau_p84_hr": float(np.nanpercentile(data.loc[valid, "tau"], 84.0)) if valid.any() else np.nan,
            "sigma_phi_median_deg": float(np.nanmedian(data.loc[valid, "sigma_phi"])) if valid.any() else np.nan,
            "sigma_phi_p84_deg": float(np.nanpercentile(data.loc[valid, "sigma_phi"], 84.0)) if valid.any() else np.nan,
            "profile_ok": bool(diag_extra.get("profile_ok", False)),
            "profile_error": diag_extra.get("profile_error", None),
            "profile_degenerate": diag_extra.get("profile_degenerate", None),
            "profile_fallback_used": bool(diag_extra.get("profile_fallback_used", False)),
            "U_min_kms": diag_extra.get("U_min_kms", None),
            "U_max_kms": diag_extra.get("U_max_kms", None),
            "U_span_kms": diag_extra.get("U_span_kms", None),
            "r_sc_profile_Rsun": diag_extra.get("r_sc_profile_Rsun", None),
        },
        "model_meta": dict(tt.meta),
        "mapping_meta": dict(map0.meta),
        "units": dict(data.attrs.get("units", {})),
        "inputs": {
            "final_pkl": str(inp["paths"]["final"]),
            "sig_pkl": str(inp["paths"]["sig"]) if inp["paths"]["sig"] else None,
        },
    }

    # Storage policy
    pd.to_pickle(data, out_pkl)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)

    files = {
        "timeseries": str(out_pkl),
        "maps_2d": str(out_png),
        "maps_3d": str(out_html) if plot_3d else None,
        "velocity_profile": str(prof_png) if prof_png is not None else None,
        "meta": str(out_meta),
        "outdir": str(out_base),
        "ephemeris_cache": str(cache_file),
    }

    return {"data": data, "meta": meta, "files": files, "fig2d": fig2d, "fig3d": fig3d}


# -----------------------------------------------------------------------------
# Backward-compatibility shim
# -----------------------------------------------------------------------------

def run_backmapping_interval(
    *,
    root_dir: Union[str, Path],
    sc: str,
    which_int: int,
    method: str = "ballistic_bg",
    cadence: str = "60min",
    smooth: Optional[str] = None,  # deprecated
    plot_vars: Sequence[str] = ("polarity", "Vr_bg", "P_ram"),
    plot_3d: bool = False,
    # NOTE: plot_3d_var is historically a *single* variable name.
    # Some notebooks passed a tuple/list here; we accept that for backward
    # compatibility and route it to plot_3d_vars.
    plot_3d_var: Optional[Union[str, Sequence[str]]] = None,
    plot_3d_vars: Optional[Sequence[str]] = None,
    plot_3d_ncols: int = 2,
    r_ss_rsun: float = 2.5,
    omega_deg_per_day: float = 14.1844,
    phi_sign: int = +1,
    vsw_fallback_kms: float = 400.0,
    ephem_step: str = "1h",
    ephem_observer: str = "earth",
    join: str = "inner",
    show: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Legacy entry point retained as a thin wrapper.

    Notes
    -----
    - ``smooth`` is deprecated; use ``vr_bg_window`` in ``backmap_interval``.
    - ``vsw_fallback_kms`` maps to ``v_fallback``.
    """

    if smooth is not None:
        kwargs.setdefault("vr_bg_window", str(smooth))

    # Backward-compatibility: if the user provided a list/tuple via plot_3d_var,
    # treat it as plot_3d_vars.
    if plot_3d_vars is None and isinstance(plot_3d_var, (list, tuple)):
        plot_3d_vars = [str(v) for v in plot_3d_var]
        plot_3d_var = None

    return backmap_interval(
        root_dir=root_dir,
        sc=sc,
        which_int=which_int,
        method=method,
        cadence=cadence,
        r_ss=float(r_ss_rsun) * u.R_sun,
        omega=float(omega_deg_per_day) * u.deg / u.day,
        phi_sign=phi_sign,
        ephem_step=ephem_step,
        ephem_observer=ephem_observer,
        plot_vars=plot_vars,
        plot_3d=plot_3d,
        plot_3d_var=str(plot_3d_var) if isinstance(plot_3d_var, str) else None,
        plot_3d_vars=plot_3d_vars,
        join=join,
        v_fallback=float(vsw_fallback_kms) * u.km / u.s,
        show=show,
        **kwargs,
    )