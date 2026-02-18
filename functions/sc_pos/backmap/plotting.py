"""sc_pos.backmap.plotting

Publication-grade (but minimal) 2D and 3D visualizations for source-surface maps.

Design principles
-----------------
- Plot content is controlled by a single ``VAR_SPECS`` dictionary.
- Color limits are percentile-based by default (robust against outliers).
- Longitudes are treated as circular where needed.
- Uncertainty overlays are optional and intentionally decimated to avoid clutter.

Note
----
This module assumes the input DataFrame has unit metadata in ``df.attrs['units']``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import plotly.graph_objects as go
import plotly.io as pio

from .circular import delta_deg

# Optional (but preferred) for physically correct Carrington grid lines on spheres.
# If unavailable, the code falls back to simple frame-lon/lat gridlines.
try:
    import astropy.units as _u
    from astropy.coordinates import SkyCoord as _SkyCoord
    from sunpy.coordinates.frames import (
        HeliographicCarrington as _HGC,
        HeliocentricEarthEcliptic as _HEE,
        HeliocentricInertial as _HCI,
    )
    _HAS_SUNPY = True
except Exception:  # pragma: no cover
    _HAS_SUNPY = False



# --------------------------------------------------------------------------------------
# Variable specification dictionary (single source of truth; user-editable).
# --------------------------------------------------------------------------------------
VAR_SPECS: Dict[str, Dict[str, Any]] = {
    "polarity": {
        "mode": "discrete",
        "label": "Polarity (sign of $B_r$)",
        "values": (-1, 0, 1),
        "colors": ("tab:blue", "0.65", "tab:red"),
    },
    "Vr": {"mode": "scalar", "scale": "linear", "cmap": "viridis", "label": r"$V_r$"},
    "Vr_bg": {"mode": "scalar", "scale": "linear", "cmap": "viridis", "label": r"$V_{\rm bg}$"},
    "Np": {"mode": "scalar", "scale": "log", "cmap": "viridis", "label": r"$n_p$"},
    "Br": {"mode": "scalar", "scale": "linear", "cmap": "coolwarm", "label": r"$B_r$"},
    "Br_r2": {"mode": "scalar", "scale": "linear", "cmap": "coolwarm", "label": r"$B_r r^2$"},
    "P_ram": {"mode": "scalar", "scale": "log", "cmap": "viridis", "label": r"$P_{\rm ram}$"},
    "tau": {"mode": "scalar", "scale": "linear", "cmap": "magma", "label": r"$\tau$"},
    "sigma_tau": {"mode": "scalar", "scale": "linear", "cmap": "magma", "label": r"$\sigma_{\tau}$"},
    "sigma_phi": {"mode": "scalar", "scale": "linear", "cmap": "magma", "label": r"$\sigma_{\phi}$"},
    "r_sc": {"mode": "scalar", "scale": "linear", "cmap": "viridis", "label": r"$r_{\rm sc}$"},
    # common turbulence diagnostics
    "sigma_c": {"mode": "scalar", "scale": "linear", "cmap": "coolwarm", "label": r"$\sigma_c$", "vmin": -1.0, "vmax": 1.0},
    "sigma_r": {"mode": "scalar", "scale": "linear", "cmap": "coolwarm", "label": r"$\sigma_r$", "vmin": -1.0, "vmax": 1.0},
}


def merge_var_specs(default_specs: Dict[str, Dict[str, Any]], override: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in default_specs.items()}
    if override is None:
        return out
    if not isinstance(override, dict):
        raise TypeError(f"var_specs must be a dict, got {type(override)}")
    for k, v in override.items():
        if k in out:
            out[k].update(v)
        else:
            out[k] = dict(v)
    return out


def _unit_to_latex(unit_str: str) -> str:
    """Best-effort conversion of unit strings to LaTeX.

    Uses astropy (preferred) if available; otherwise falls back to a minimal
    string normalization.
    """
    us = str(unit_str).strip()
    if not us:
        return ""
    try:
        import astropy.units as u  # local import (pipeline requires astropy)
        s = u.Unit(us).to_string("latex_inline")
        # latex_inline returns something like '$\\mathrm{km\\,s^{-1}}$'
        return s.strip().strip("$")
    except Exception:
        # Minimal fallback: 'km / s' -> 'km\,s^{-1}'
        t = us.replace(" / ", "/").replace(" ", "")
        if "/" in t:
            num, den = t.split("/", 1)
            return f"\\mathrm{{{num}\\,{den}^{{-1}}}}"
        return f"\\mathrm{{{t}}}"


def _label_with_unit(data: pd.DataFrame, var: str, base_label: str) -> str:
    """Build a MathJax-friendly label with units, if known.

    If base_label is already in $...$, we keep it in math mode and append units
    inside math mode as '\\,[...]' for consistent rendering.
    """
    um = data.attrs.get("units", {})
    if var not in um:
        return str(base_label)

    u_ltx = _unit_to_latex(um[var])
    if not u_ltx:
        return str(base_label)

    bl = str(base_label)
    if bl.startswith("$") and bl.endswith("$") and len(bl) >= 2:
        inner = bl[1:-1]
        return f"$" + inner + f"\\,[{u_ltx}]$"
    # Non-math label: keep readable, but still MathJax-safe.
    return f"{bl} [{um[var]}]"

def _compute_scalar_limits(v: np.ndarray, *, spec: Dict[str, Any], percentiles: Tuple[float, float]) -> Tuple[float, float]:
    if spec.get("vmin", None) is not None and spec.get("vmax", None) is not None:
        return float(spec["vmin"]), float(spec["vmax"])

    vv = np.asarray(v, float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        raise ValueError("No finite values available to set colormap limits.")

    if spec.get("scale", "linear") == "log":
        vv = vv[vv > 0]
        if vv.size == 0:
            raise ValueError("Log scale requested but no positive values exist after masking.")

    lo, hi = np.nanpercentile(vv, list(percentiles))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vv))
        hi = float(np.nanmax(vv))
    return float(lo), float(hi)


def marker_sizes_from_metric(x: pd.Series, *, smin: float = 12.0, smax: float = 140.0, prc: Tuple[float, float] = (5.0, 95.0)) -> pd.Series:
    x = pd.Series(x, index=x.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    if not np.isfinite(x).any():
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)
    lo, hi = np.nanpercentile(x, list(prc))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(x), (smin + smax) / 2.0), index=x.index)
    return smin + (smax - smin) * np.clip((x - lo) / (hi - lo), 0, 1)


def plot_source_surface_2d(
    *,
    data: pd.DataFrame,
    out_png: Union[str, Path],
    plot_vars: List[str],
    var_specs: Dict[str, Dict[str, Any]],
    size_col: str = "marker_size",
    percentiles: Tuple[float, float] = (2.0, 98.0),
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    lat_lim: Optional[Tuple[float, float]] = None,
    show_uncertainty: bool = False,
    uncertainty_decimate: int = 4,
    uncertainty_alpha: float = 0.45,
    uncertainty_lw: float = 0.9,
    uncertainty_zorder: int = 3,
    uncertainty_ecolor: str = "k",
    uncertainty_capsize: float = 1.2,
    summary_box: Optional[str] = None,
    profile_panel: Optional[Dict[str, Any]] = None,
    show: bool = False,
) -> Tuple[Path, plt.Figure]:
    """2D (phi,lat) maps for a list of variables."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    for v in plot_vars:
        if v not in var_specs:
            raise KeyError(f"Missing VAR_SPECS entry for {v!r}.")
        if v != "polarity" and v not in data.columns:
            raise KeyError(f"Requested plot variable {v!r} not in DataFrame.")

    x = pd.to_numeric(data["phi_src"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float)

    s = data[size_col].to_numpy(dtype=float) if size_col in data.columns else np.full_like(x, 30.0)

    # Uncertainty on longitude (optional)
    xerr_l = xerr_r = None
    if show_uncertainty and {"phi_src_p16", "phi_src_p84"}.issubset(data.columns):
        lo = pd.to_numeric(data["phi_src_p16"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(data["phi_src_p84"], errors="coerce").to_numpy(dtype=float)
        xerr_l = np.abs(delta_deg(x, lo))
        xerr_r = np.abs(delta_deg(hi, x))

    n_maps = len(plot_vars)
    n = n_maps + (1 if profile_panel is not None else 0)
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(n / ncols))

    figsz = tuple(figsize) if (figsize is not None) else (6.4 * ncols, 4.3 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsz,
        constrained_layout=True,
        sharex=(profile_panel is None),
        sharey=(profile_panel is None),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, var in enumerate(plot_vars):
        ax = axes[i]
        spec = var_specs[var]

        if spec.get("mode") == "discrete" and var == "polarity":
            br = pd.to_numeric(data.get("Br", np.nan), errors="coerce").to_numpy(dtype=float)
            pol = np.sign(br)
            vals = spec["values"]
            cols = spec["colors"]
            cmap = {vals[j]: cols[j] for j in range(len(vals))}
            c = np.array([cmap.get(int(p), cols[1]) for p in pol], dtype=object)

            ax.scatter(x, y, s=np.minimum(s, 60.0), c=c, alpha=0.9, linewidths=0)

            handles = []
            labels = []
            for v, col in zip(vals, cols):
                handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=col, markersize=8))
                labels.append(str(v))
            ax.legend(handles, labels, title=spec.get("label", "Polarity"), loc="upper right", frameon=True)

            # Uncertainty overlay (longitude CI) should appear on *all* panels,
            # including discrete/categorical ones like polarity.
            if show_uncertainty and (xerr_l is not None) and (uncertainty_decimate is not None) and (int(uncertainty_decimate) > 0):
                step = max(1, int(uncertainty_decimate))
                ii = np.arange(len(x), dtype=int)[::step]
                ax.errorbar(
                    x[ii],
                    y[ii],
                    xerr=np.vstack([xerr_l[ii], xerr_r[ii]]),
                    fmt="none",
                    ecolor=str(uncertainty_ecolor),
                    elinewidth=float(uncertainty_lw),
                    alpha=float(uncertainty_alpha),
                    capsize=float(uncertainty_capsize),
                    zorder=int(uncertainty_zorder),
                )

        elif spec.get("mode") == "scalar":
            v = pd.to_numeric(data[var], errors="coerce").to_numpy(dtype=float)
            vmin, vmax = _compute_scalar_limits(v, spec=spec, percentiles=percentiles)

            cmap = spec.get("cmap", "viridis")
            if spec.get("scale", "linear") == "log":
                vv = v.copy()
                vv[~np.isfinite(vv) | (vv <= 0)] = np.nan
                norm = LogNorm(vmin=vmin, vmax=vmax)
                sc = ax.scatter(x, y, s=s, c=vv, cmap=cmap, alpha=0.9, linewidths=0, norm=norm)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
                sc = ax.scatter(x, y, s=s, c=v, cmap=cmap, alpha=0.9, linewidths=0, norm=norm)

            if show_uncertainty and (xerr_l is not None) and (uncertainty_decimate is not None) and (int(uncertainty_decimate) > 0):
                step = max(1, int(uncertainty_decimate))
                ii = np.arange(len(x), dtype=int)[::step]
                ax.errorbar(
                    x[ii],
                    y[ii],
                    xerr=np.vstack([xerr_l[ii], xerr_r[ii]]),
                    fmt="none",
                    ecolor=str(uncertainty_ecolor),
                    elinewidth=float(uncertainty_lw),
                    alpha=float(uncertainty_alpha),
                    capsize=float(uncertainty_capsize),
                    zorder=int(uncertainty_zorder),
                )

            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label(_label_with_unit(data, var, spec.get("label", var)))

        else:
            raise ValueError(f"Invalid VAR_SPECS mode for {var!r}: {spec.get('mode')!r}")

        ax.set_xlim(0, 360)
        if lat_lim is not None:
            ax.set_ylim(float(lat_lim[0]), float(lat_lim[1]))
        ax.set_xlabel(r"$\phi_{\rm src}$ (deg)")
        ax.set_ylabel(r"$\lambda_{\rm src}$ (deg)")
        ax.grid(True, alpha=0.25)

    # ------------------------------------------------------------------
    # Mandatory diagnostics panel: U(r) profile used for tau(r)
    # ------------------------------------------------------------------
    if profile_panel is not None:
        axp = axes[n_maps]

        r = np.asarray(profile_panel.get("r_grid_Rsun", []), float)
        U = np.asarray(profile_panel.get("U_med_kms", []), float)
        U_lo = profile_panel.get("U_lo_kms", None)
        U_hi = profile_panel.get("U_hi_kms", None)

        if r.size >= 2 and U.size == r.size:
            axp.plot(r, U, lw=2)
            if (U_lo is not None) and (U_hi is not None):
                lo = np.asarray(U_lo, float)
                hi = np.asarray(U_hi, float)
                if lo.size == r.size and hi.size == r.size:
                    axp.fill_between(r, lo, hi, alpha=0.22, linewidth=0)

            r_ss = profile_panel.get("r_ss_Rsun", None)
            r_sc = profile_panel.get("r_sc_Rsun", None)
            if r_ss is not None:
                axp.axvline(float(r_ss), lw=1.2, alpha=0.65)
            if r_sc is not None:
                axp.axvline(float(r_sc), lw=1.2, alpha=0.65, ls="--")

            axp.set_xscale("log")
            axp.set_xlabel(r"$r\,[R_\odot]$")
            axp.set_ylabel(r"$U(r)\,[\mathrm{km\,s^{-1}}]$")
            axp.grid(True, alpha=0.25)
        else:
            axp.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title)

    if summary_box:
        # Place a compact textbox in figure coordinates.
        fig.text(
            0.985,
            0.98,
            summary_box,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.3", alpha=0.92),
        )

    fig.savefig(out_png, dpi=240)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_png, fig


def plot_velocity_profile(
    *,
    out_png: Union[str, Path],
    r_grid: np.ndarray,
    U_med: np.ndarray,
    U_lo: Optional[np.ndarray] = None,
    U_hi: Optional[np.ndarray] = None,
    r_ss: Optional[float] = None,
    r_sc: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.4, 5.6),
    show: bool = False,
) -> Tuple[Path, plt.Figure]:
    """Diagnostic: assumed velocity and acceleration profile along r.

    Parameters
    ----------
    r_grid : array
        Radial grid in R_sun (monotonic increasing).
    U_* : array
        Speeds in km/s.

    Output
    ------
    PNG with two stacked panels:
    (top) U(r) with optional uncertainty band; (bottom) dU/dr.
    """

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    r = np.asarray(r_grid, float)
    U = np.asarray(U_med, float)
    if r.size != U.size or r.size < 4:
        raise ValueError("plot_velocity_profile requires r_grid and U_med with matching length >= 4")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    # Velocity
    ax0.plot(r, U, lw=2)
    if (U_lo is not None) and (U_hi is not None):
        lo = np.asarray(U_lo, float)
        hi = np.asarray(U_hi, float)
        if lo.shape == U.shape and hi.shape == U.shape:
            ax0.fill_between(r, lo, hi, alpha=0.25, linewidth=0)
    ax0.set_ylabel(r"$U(r)\,[\mathrm{km\,s^{-1}}]$")
    ax0.grid(True, alpha=0.25)

    # Acceleration proxy dU/dr
    dUdr = np.gradient(U, r)
    ax1.plot(r, dUdr, lw=2)
    ax1.set_ylabel(r"$dU/dr\,[\mathrm{km\,s^{-1}}\,R_\odot^{-1}]$")
    ax1.set_xlabel(r"$r\,[R_\odot]$")
    ax1.grid(True, alpha=0.25)

    # Log r is far more readable from 2.5 R_sun to ~100+ R_sun.
    ax1.set_xscale("log")

    if r_ss is not None:
        ax0.axvline(float(r_ss), lw=1.5, alpha=0.6)
        ax1.axvline(float(r_ss), lw=1.5, alpha=0.6)
    if r_sc is not None:
        ax0.axvline(float(r_sc), lw=1.5, alpha=0.6, ls="--")
        ax1.axvline(float(r_sc), lw=1.5, alpha=0.6, ls="--")

    if title:
        fig.suptitle(str(title))

    fig.savefig(out_png, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_png, fig


def _mpl_cmap_to_plotly(cmap_name: str, n: int = 256) -> list:
    cmap = plt.get_cmap(str(cmap_name))
    xs = np.linspace(0.0, 1.0, int(max(8, n)))
    cols = cmap(xs)
    out = []
    for x, (r, g, b, a) in zip(xs, cols):
        out.append([float(x), f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {float(a):.4f})"])
    return out


def _sphere_mesh3d(radius: float, n_lat: int = 42, n_lon: int = 84) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = float(radius)
    n_lat = int(max(8, n_lat))
    n_lon = int(max(16, n_lon))

    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    clat = np.cos(lat)[:, None]
    slat = np.sin(lat)[:, None]
    clon = np.cos(lon)[None, :]
    slon = np.sin(lon)[None, :]

    x = (r * clat * clon).ravel()
    y = (r * clat * slon).ravel()
    z = (r * slat * np.ones_like(clon)).ravel()

    def vid(i_lat: int, i_lon: int) -> int:
        return i_lat * n_lon + (i_lon % n_lon)

    ii, jj, kk = [], [], []
    for a in range(n_lat - 1):
        for b in range(n_lon):
            v00 = vid(a, b)
            v01 = vid(a, b + 1)
            v10 = vid(a + 1, b)
            v11 = vid(a + 1, b + 1)
            ii.append(v00); jj.append(v10); kk.append(v11)
            ii.append(v00); jj.append(v11); kk.append(v01)

    return x, y, z, np.asarray(ii, dtype=int), np.asarray(jj, dtype=int), np.asarray(kk, dtype=int)

def _median_obstime(data: pd.DataFrame):
    try:
        t = pd.to_datetime(data.index)
        if len(t) == 0:
            return None
        return t[len(t) // 2].to_pydatetime()
    except Exception:
        return None


def _target_frame(frame3d: str, obstime):
    f = str(frame3d).upper().strip()
    if not _HAS_SUNPY or obstime is None:
        return None
    if f == "HEE":
        return _HEE(obstime=obstime)
    if f == "HCI":
        return _HCI(obstime=obstime)
    return None  # GSE not supported here


def _grid_lines_carrington(
    *,
    radius_au: float,
    lat_step_deg: int,
    lon_step_deg: int,
    frame3d: str,
    obstime,
    observer: str = "earth",
    n: int = 240,
):
    """Return (x,y,z) arrays (AU) for Carrington lat/lon grid lines, transformed to frame3d.

    Longitude/latitude are Carrington (HGC). The grid is transformed to the requested plotting
    frame (HEE/HCI) at the given obstime. If SunPy is unavailable, callers must fall back to
    simple frame-lon/lat grids.
    """
    if (not _HAS_SUNPY) or (obstime is None):
        raise RuntimeError("SunPy not available or obstime missing")

    lat_step = int(max(5, abs(int(lat_step_deg))))
    lon_step = int(max(5, abs(int(lon_step_deg))))

    lats = np.arange(-90 + lat_step, 90, lat_step, dtype=float)
    lons = np.arange(0, 360, lon_step, dtype=float)

    tf = _target_frame(frame3d, obstime)
    if tf is None:
        raise RuntimeError(f"Unsupported frame3d for Carrington grids: {frame3d}")

    X, Y, Z = [], [], []

    # latitude circles
    ll = np.linspace(0.0, 360.0, n)
    for lat in lats:
        lon = ll
        c = _SkyCoord(
            lon=lon * _u.deg,
            lat=np.full_like(lon, lat, dtype=float) * _u.deg,
            radius=float(radius_au) * _u.AU,
            frame=_HGC(obstime=obstime, observer=observer),
        ).transform_to(tf)
        X.extend(c.cartesian.x.to_value(_u.AU).tolist() + [np.nan])
        Y.extend(c.cartesian.y.to_value(_u.AU).tolist() + [np.nan])
        Z.extend(c.cartesian.z.to_value(_u.AU).tolist() + [np.nan])

    # longitude meridians
    tt = np.linspace(-90.0, 90.0, n)
    for lon in lons:
        c = _SkyCoord(
            lon=np.full_like(tt, lon, dtype=float) * _u.deg,
            lat=tt * _u.deg,
            radius=float(radius_au) * _u.AU,
            frame=_HGC(obstime=obstime, observer=observer),
        ).transform_to(tf)
        X.extend(c.cartesian.x.to_value(_u.AU).tolist() + [np.nan])
        Y.extend(c.cartesian.y.to_value(_u.AU).tolist() + [np.nan])
        Z.extend(c.cartesian.z.to_value(_u.AU).tolist() + [np.nan])

    return np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)


def _grid_labels_carrington(
    *,
    radius_au: float,
    frame3d: str,
    obstime,
    observer: str = "earth",
    lon_labels_deg=(0, 90, 180, 270),
    lat_labels_deg=(-60, -30, 30, 60),
):
    """Sparse text anchors for Carrington lon/lat labels on the *source surface* sphere."""
    if (not _HAS_SUNPY) or (obstime is None):
        return None

    tf = _target_frame(frame3d, obstime)
    if tf is None:
        return None

    xs, ys, zs, texts = [], [], [], []

    # lon labels at equator
    for lon in lon_labels_deg:
        c = _SkyCoord(
            lon=float(lon) * _u.deg,
            lat=0.0 * _u.deg,
            radius=float(radius_au) * _u.AU,
            frame=_HGC(obstime=obstime, observer=observer),
        ).transform_to(tf)
        xs.append(c.cartesian.x.to_value(_u.AU))
        ys.append(c.cartesian.y.to_value(_u.AU))
        zs.append(c.cartesian.z.to_value(_u.AU))
        texts.append(f"{int(lon)}°")

    # lat labels on prime meridian (lon=0)
    for lat in lat_labels_deg:
        c = _SkyCoord(
            lon=0.0 * _u.deg,
            lat=float(lat) * _u.deg,
            radius=float(radius_au) * _u.AU,
            frame=_HGC(obstime=obstime, observer=observer),
        ).transform_to(tf)
        xs.append(c.cartesian.x.to_value(_u.AU))
        ys.append(c.cartesian.y.to_value(_u.AU))
        zs.append(c.cartesian.z.to_value(_u.AU))
        texts.append(f"{int(lat)}°")

    return np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float), texts


def plot_source_surface_3d(
    *,
    data: pd.DataFrame,
    out_html: Union[str, Path],
    var_specs: Dict[str, Dict[str, Any]],
    r_ss_au: float,
    r_sun_au: float,
    r_sc_med_rsun: Optional[float] = None,
    frame3d: str = "HEE",
    plot_vars: Optional[List[str]] = None,
    var: Optional[str] = None,
    percentiles: Tuple[float, float] = (2.0, 98.0),
    ncols_vars: int = 2,
    # visual scale: all panels are a Sun-centered zoom cube in AU
    sun_zoom_au: float = 0.06,
    width: int = 1700,
    height: int = 900,
    decimate: int = 1,
    # spacecraft rendering inside the zoom cube
    sc_project_to_shell: bool = True,
    sc_shell_frac: float = 0.98,
    # panel framing / clarity
    draw_panel_boxes: bool = True,
    show_cube_edges: bool = False,
    show_links: bool = True,
    link_count: int = 12,
    link_line_rgba: str = "rgba(0,0,0,0.22)",
    show_rtn_axes: bool = True,
    rtn_axis_frac: float = 0.22,
    # sphere latitude/longitude grid
    show_sphere_grid: bool = True,
    grid_lon_step_deg: int = 45,
    grid_lat_step_deg: int = 30,
    grid_opacity: float = 0.18,
    grid_width: int = 2,
    sphere_grid_frame: str = "carrington",
    sphere_grid_observer: str = "earth",
    show_grid_labels: bool = True,
    # uncertainty display
    show_uncertainty_arcs: bool = False,
    uncertainty_decimate: int = 10,
    # camera / interactivity
    sync_cameras: bool = True,
    camera: str = "iso",
    title: Optional[str] = None,
    show: bool = False,
    # legacy args kept for compatibility (ignored)
    plane_span_au: float = 1.2,
    camera_left: Optional[str] = None,
    camera_right: Optional[str] = None,
) -> Tuple[Path, go.Figure]:
    """3D source-surface backmapping with multi-variable subpanels.

    Key design choices (physics + visualization)
    -------------------------------------------
    1) Every subpanel is a Sun-centered *zoom cube* with identical axis ranges and aspect ratio.
       This prevents the Sun/source-surface spheres from becoming visually flattened ("pancake").
    2) The spacecraft trajectory is shown in every panel, but (optionally) projected to a shell
       inside the zoom cube so the Sun/source-surface remain readable.
    3) Source-surface points are colored by the chosen variable; scaling follows ``var_specs``.
       Polarity is rendered as a discrete categorical variable.
    4) Latitude/longitude grid lines are drawn directly on the spheres to improve spatial legibility.
    5) When exported to HTML, cameras can be synchronized across all subpanels.

    Required columns (upstream)
    ---------------------------
    - ss_x_au, ss_y_au, ss_z_au : mapped source-surface points (Cartesian, AU) in the plotting frame
    - sc_x_au, sc_y_au, sc_z_au : spacecraft position (Cartesian, AU) in the plotting frame

    Optional columns
    ----------------
    - phi_src, lat_src          : (Carrington) longitude/latitude used for 2D maps (shown in hover)
    - phi_src_p16/phi_src_p84    : longitude percentiles used to draw Carrington small-circle CI arcs
    """

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Parse variables list
    # -----------------------
    if plot_vars is None:
        if var is None:
            raise ValueError("Provide either plot_vars (list) or var (single variable).")
        plot_vars = [str(var)]
    else:
        plot_vars = [str(v) for v in list(plot_vars)]
        if len(plot_vars) == 0:
            raise ValueError("plot_vars cannot be empty.")
        if var is not None:
            raise ValueError("Provide only one of plot_vars or var, not both.")

    for v in plot_vars:
        if v not in var_specs:
            raise KeyError(f"Missing VAR_SPECS entry for {v!r}.")
        if v != "polarity" and v not in data.columns:
            raise KeyError(f"Requested 3D variable {v!r} not in DataFrame.")

    frame3d = str(frame3d).upper().strip()
    if frame3d not in {"HEE", "HCI"}:
        raise ValueError("frame3d must be 'HEE' or 'HCI'")

    if int(decimate) < 1:
        decimate = 1

    # -----------------------
    # Geometry
    # -----------------------
    req_ss = {"ss_x_au", "ss_y_au", "ss_z_au"}
    if not req_ss.issubset(data.columns):
        raise KeyError("3D plotting requires columns ss_x_au, ss_y_au, ss_z_au (computed upstream).")
    req_sc = {"sc_x_au", "sc_y_au", "sc_z_au"}
    have_sc = req_sc.issubset(data.columns)

    xs = pd.to_numeric(data["ss_x_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
    ys = pd.to_numeric(data["ss_y_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
    zs = pd.to_numeric(data["ss_z_au"], errors="coerce").to_numpy(dtype=float)[::decimate]

    if have_sc:
        xsc_raw = pd.to_numeric(data["sc_x_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
        ysc_raw = pd.to_numeric(data["sc_y_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
        zsc_raw = pd.to_numeric(data["sc_z_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
    else:
        xsc_raw = ysc_raw = zsc_raw = None

    # -----------------------
    # Zoom cube (fixes "pancake Sun")
    # -----------------------
    lim = float(max(float(sun_zoom_au), 1.25 * float(r_ss_au), 2.2 * float(r_sun_au)))
    lim = float(max(lim, 1.05 * float(r_ss_au)))
    shell_r = float(max(0.0, float(sc_shell_frac))) * lim

    xsc = ysc = zsc = None
    if have_sc and (xsc_raw is not None):
        if sc_project_to_shell:
            r = np.sqrt(xsc_raw * xsc_raw + ysc_raw * ysc_raw + zsc_raw * zsc_raw)
            ok = np.isfinite(r) & (r > 0)
            xsc = np.full_like(xsc_raw, np.nan, dtype=float)
            ysc = np.full_like(ysc_raw, np.nan, dtype=float)
            zsc = np.full_like(zsc_raw, np.nan, dtype=float)
            xsc[ok] = (xsc_raw[ok] / r[ok]) * shell_r
            ysc[ok] = (ysc_raw[ok] / r[ok]) * shell_r
            zsc[ok] = (zsc_raw[ok] / r[ok]) * shell_r
        else:
            xsc, ysc, zsc = xsc_raw.copy(), ysc_raw.copy(), zsc_raw.copy()

    # ecliptic plane patch inside the cube
    s2 = 0.55 * lim
    px = np.array([-s2, s2, s2, -s2], dtype=float)
    py = np.array([-s2, -s2, s2, s2], dtype=float)
    pz = np.zeros(4, dtype=float)

    # spheres
    sx, sy, sz, si, sj, sk = _sphere_mesh3d(float(r_sun_au), n_lat=48, n_lon=96)
    rx, ry, rz, ri, rj, rk = _sphere_mesh3d(float(r_ss_au), n_lat=48, n_lon=96)

    # -----------------------
    # Sphere grid lines (lat/lon)
    # -----------------------
    def _grid_lines(radius: float, *, lat_step: int, lon_step: int, n: int = 220):
        r = float(radius)
        lat_step = int(max(5, abs(int(lat_step))))
        lon_step = int(max(5, abs(int(lon_step))))

        lats = np.arange(-90 + lat_step, 90, lat_step, dtype=float)
        lons = np.arange(0, 360, lon_step, dtype=float)

        X, Y, Z = [], [], []
        ll = np.linspace(0.0, 2.0 * np.pi, n)
        for lat_deg in lats:
            lat = np.deg2rad(lat_deg)
            cl = float(np.cos(lat))
            sl = float(np.sin(lat))
            X.extend((r * cl * np.cos(ll)).tolist() + [np.nan])
            Y.extend((r * cl * np.sin(ll)).tolist() + [np.nan])
            Z.extend((np.full_like(ll, r * sl)).tolist() + [np.nan])

        tt = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n)
        for lon_deg in lons:
            lon = np.deg2rad(lon_deg)
            cl = np.cos(tt)
            sl = np.sin(tt)
            X.extend((r * cl * np.cos(lon)).tolist() + [np.nan])
            Y.extend((r * cl * np.sin(lon)).tolist() + [np.nan])
            Z.extend((r * sl).tolist() + [np.nan])

        return np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)

    theta_ring = np.linspace(0.0, 2.0 * np.pi, 360)

    def _equator_ring(radius: float):
        r = float(radius)
        return r * np.cos(theta_ring), r * np.sin(theta_ring), np.zeros_like(theta_ring)

    def _prime_meridian(radius: float):
        r = float(radius)
        lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 220)
        return r * np.cos(lat), np.zeros_like(lat), r * np.sin(lat)

    # -----------------------
    # -----------------------
    # Uncertainty arcs on SS sphere (Carrington small-circle, frame-consistent)
    # -----------------------
    segs = None
    if show_uncertainty_arcs and _HAS_SUNPY and {"phi_src_p16", "phi_src_p84", "lat_src"}.issubset(data.columns):
        try:
            from astropy.time import Time as _Time
            from sunpy.coordinates import get_body_heliographic_stonyhurst as _get_body_hgs
        except Exception:
            _Time = None
            _get_body_hgs = None

        if (_Time is not None) and (_get_body_hgs is not None):
            phi16 = pd.to_numeric(data["phi_src_p16"], errors="coerce").to_numpy(dtype=float)[::decimate]
            phi84 = pd.to_numeric(data["phi_src_p84"], errors="coerce").to_numpy(dtype=float)[::decimate]
            lat0 = pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float)[::decimate]

            ok = np.isfinite(phi16) & np.isfinite(phi84) & np.isfinite(lat0)
            idx_ok = np.where(ok)[0]
            if idx_ok.size:
                max_arcs = 120
                if idx_ok.size > max_arcs:
                    idx_ok = idx_ok[np.linspace(0, idx_ok.size - 1, max_arcs).astype(int)]

                nseg = max(6, int(uncertainty_decimate))
                ts = np.linspace(0.0, 1.0, nseg)

                def _wrap_360(x):
                    return np.mod(x, 360.0)

                def _delta_min(a, b):
                    return ((b - a + 180.0) % 360.0) - 180.0

                times_dec = pd.DatetimeIndex(data.index)[::decimate].to_pydatetime()

                xs_g, ys_g, zs_g = [], [], []
                for ii in idx_ok:
                    a = float(phi16[ii])
                    b = float(phi84[ii])
                    lat = float(lat0[ii])
                    dt = times_dec[ii]

                    d = _delta_min(a, b)
                    lons = _wrap_360(a + ts * d)

                    obstime = _Time(dt)
                    earth_obs = _get_body_hgs("earth", obstime)
                    hgc = _HGC(obstime=obstime, observer=earth_obs)

                    tf = _target_frame(frame3d, obstime)
                    if tf is None:
                        continue

                    fp = _SkyCoord(
                        lon=lons * _u.deg,
                        lat=lat * _u.deg,
                        radius=float(r_ss_au) * _u.AU,
                        frame=hgc,
                    ).transform_to(tf)

                    xs_g.extend(fp.cartesian.x.to_value(_u.AU).tolist() + [np.nan])
                    ys_g.extend(fp.cartesian.y.to_value(_u.AU).tolist() + [np.nan])
                    zs_g.extend(fp.cartesian.z.to_value(_u.AU).tolist() + [np.nan])

                if xs_g:
                    segs = (np.asarray(xs_g, float), np.asarray(ys_g, float), np.asarray(zs_g, float))

# -----------------------
    # Orientation axes (plotting-frame basis; avoids ambiguous 'RTN' labeling)
    # -----------------------
    rtn = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    axis_len = float(max(0.0, rtn_axis_frac)) * float(lim)


    # -----------------------
    # Subplot grid
    # -----------------------
    from plotly.subplots import make_subplots

    nvars = len(plot_vars)
    ncols_vars = int(max(1, ncols_vars))
    nrows = int(np.ceil(nvars / ncols_vars))
    total_cells = int(nrows * ncols_vars)

    specs = []
    for rr in range(nrows):
        row_specs = []
        for cc in range(ncols_vars):
            row_specs.append({"type": "scene"})
        specs.append(row_specs)

    fig = make_subplots(
        rows=nrows,
        cols=ncols_vars,
        specs=specs,
        horizontal_spacing=0.012,
        vertical_spacing=0.02,
    )

    fig.update_layout(
        template="plotly_white",
        width=int(width),
        height=int(height),
        margin=dict(l=0, r=0, t=55, b=0),
        title=dict(text=title or f"3D backmapping ({frame3d})", x=0.5, xanchor="center"),
        font=dict(family="Arial", size=14),
        legend=dict(groupclick="togglegroup"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # We intentionally avoid panel titles for long-term clarity.
    fig.layout.annotations = []

    # Always show the median spacecraft heliocentric distance on-figure (not hover-only).
    if r_sc_med_rsun is not None and np.isfinite(float(r_sc_med_rsun)):
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text=f"r_sc,med={float(r_sc_med_rsun):.1f} R_sun",
            showarrow=False, align="left",
            font=dict(size=14, color="rgba(0,0,0,0.75)"),
        )


    if any(v == "polarity" for v in plot_vars):
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers", marker=dict(size=7, color="#1f77b4"), name="polarity = -1", showlegend=True))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers", marker=dict(size=7, color="#7f7f7f"), name="polarity = 0", showlegend=True))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers", marker=dict(size=7, color="#d62728"), name="polarity = +1", showlegend=True))

    def _scene_axes(_ttl: str = ""):
        # Minimal 3D aesthetic: no ticks/background planes (presentation-first).
        return dict(
            title=dict(text=""),
            range=[-lim, lim],
            showspikes=False,
            showgrid=False,
            showbackground=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False,
        )

    def _cam(which: str):
        w = str(which).lower().strip()
        if w == "top":
            return dict(eye=dict(x=0.0, y=0.0, z=2.4))
        if w == "side":
            return dict(eye=dict(x=2.4, y=0.0, z=0.0))
        if w in {"ecliptic", "xy"}:
            return dict(eye=dict(x=2.0, y=1.6, z=0.25))
        return dict(eye=dict(x=1.65, y=1.35, z=0.95))

    def _add_cube_wireframe(row: int, col: int, *, lw: float = 2.0, colr: str = "rgba(0,0,0,0.28)"):
        # 12 edges of the bounding cube [-lim, lim]^3
        a = float(lim)
        pts = [
            (-a, -a, -a), (a, -a, -a), (a, a, -a), (-a, a, -a),
            (-a, -a, a),  (a, -a, a),  (a, a, a),  (-a, a, a),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i0, i1 in edges:
            x0, y0, z0 = pts[i0]
            x1, y1, z1 = pts[i1]
            fig.add_trace(
                go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode="lines",
                    line=dict(color=colr, width=lw),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    for k in range(total_cells):
        row = (k // ncols_vars) + 1
        col = (k % ncols_vars) + 1
        fig.update_scenes(
            dict(
                xaxis=_scene_axes("x [AU]"),
                yaxis=_scene_axes("y [AU]"),
                zaxis=_scene_axes("z [AU]"),
                aspectmode="cube",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            row=row,
            col=col,
        )
        fig.update_scenes(camera=_cam(camera), row=row, col=col)
        if show_cube_edges:
            _add_cube_wireframe(row, col)

    if draw_panel_boxes:
        def _scene_name(idx1: int) -> str:
            return "scene" if idx1 == 1 else f"scene{idx1}"
        for i in range(1, total_cells + 1):
            sname = _scene_name(i)
            dom = getattr(fig.layout, sname).domain
            x0, x1 = float(dom.x[0]), float(dom.x[1])
            y0, y1 = float(dom.y[0]), float(dom.y[1])
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="rgba(0,0,0,0.25)", width=1),
                fillcolor="rgba(0,0,0,0)",
                layer="below",
            )

    def _add_rtn_axes(*, row: int, col: int) -> None:
        if (not show_rtn_axes) or (rtn is None) or (axis_len <= 0):
            return
        R, T, N = rtn
        colr = "rgba(0,0,0,0.35)"
        lw = 5
        labx = f"{frame3d} +X"
        laby = f"{frame3d} +Y"
        labz = f"{frame3d} +Z"
        for nm, vec in [(labx, R), (laby, T), (labz, N)]:
            vx, vy, vz = (axis_len * vec).tolist()
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, vx], y=[0.0, vy], z=[0.0, vz],
                    mode="lines+text",
                    line=dict(width=lw, color=colr),
                    opacity=0.70,
                    text=["", nm],
                    textposition="top center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    def _add_links(*, row: int, col: int) -> None:
        if (not show_links) or (not have_sc) or (xsc is None):
            return
        n = len(xs)
        if n == 0:
            return
        m = int(max(1, link_count))
        idx = np.linspace(0, n - 1, min(m, n)).astype(int)

        Xl = np.empty(3 * len(idx), dtype=float)
        Yl = np.empty(3 * len(idx), dtype=float)
        Zl = np.empty(3 * len(idx), dtype=float)
        Xl[0::3] = xsc[idx]
        Yl[0::3] = ysc[idx]
        Zl[0::3] = zsc[idx]
        Xl[1::3] = xs[idx]
        Yl[1::3] = ys[idx]
        Zl[1::3] = zs[idx]
        Xl[2::3] = np.nan
        Yl[2::3] = np.nan
        Zl[2::3] = np.nan

        fig.add_trace(
            go.Scatter3d(
                x=Xl, y=Yl, z=Zl,
                mode="lines",
                line=dict(width=2, color=link_line_rgba),
                opacity=0.22,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    grid_sun = grid_ss = None
    if show_sphere_grid:
        obstime = _median_obstime(data)
        use_carr = (str(sphere_grid_frame).lower().strip() == "carrington") and _HAS_SUNPY and (obstime is not None) and (str(frame3d).upper().strip() in {"HEE","HCI"})
        if use_carr:
            grid_sun = _grid_lines_carrington(
                radius_au=float(r_sun_au),
                lat_step_deg=int(grid_lat_step_deg),
                lon_step_deg=int(grid_lon_step_deg),
                frame3d=str(frame3d),
                obstime=obstime,
                observer=str(sphere_grid_observer),
            )
            grid_ss = _grid_lines_carrington(
                radius_au=float(r_ss_au),
                lat_step_deg=int(grid_lat_step_deg),
                lon_step_deg=int(grid_lon_step_deg),
                frame3d=str(frame3d),
                obstime=obstime,
                observer=str(sphere_grid_observer),
            )
            grid_labels = _grid_labels_carrington(
                radius_au=float(r_ss_au),
                frame3d=str(frame3d),
                obstime=obstime,
                observer=str(sphere_grid_observer),
            ) if bool(show_grid_labels) else None
        else:
            grid_sun = _grid_lines(float(r_sun_au), lat_step=int(grid_lat_step_deg), lon_step=int(grid_lon_step_deg))
            grid_ss = _grid_lines(float(r_ss_au), lat_step=int(grid_lat_step_deg), lon_step=int(grid_lon_step_deg))
            grid_labels = None


    def _colorbar_pos(row: int, col: int):
        idx1 = (row - 1) * ncols_vars + col
        sname = "scene" if idx1 == 1 else f"scene{idx1}"
        dom = getattr(fig.layout, sname).domain
        x0, x1 = float(dom.x[0]), float(dom.x[1])
        y0, y1 = float(dom.y[0]), float(dom.y[1])
        # Place colorbars *inside* each panel near the right edge to avoid
        # giant outer margins (keeps the subplot grid compact).
        x = max(x0 + 0.01, min(0.985, x1 - 0.018))
        y = 0.5 * (y0 + y1)
        ln = 0.66 * (y1 - y0)
        return dict(x=x, y=y, len=ln)

    def _log10_label(lbl: str) -> str:
        s = str(lbl)
        if s.startswith("$") and s.endswith("$") and len(s) >= 2:
            inner = s[1:-1]
            return f"$\\log_{10}\\left({inner}\\right)$"
        return f"log10({s})"

    # local lon/lat for hover (plotting frame)
    rloc = np.sqrt(xs * xs + ys * ys + zs * zs)
    lon_loc = (np.degrees(np.arctan2(ys, xs)) + 360.0) % 360.0
    lat_loc = np.degrees(np.arcsin(np.where(rloc > 0, zs / rloc, np.nan)))

    try:
        t_hover = pd.to_datetime(data.index).astype("datetime64[ns]")[::decimate]
    except Exception:
        t_hover = None

    phi_car = pd.to_numeric(data["phi_src"], errors="coerce").to_numpy(dtype=float)[::decimate] if ("phi_src" in data.columns) else None
    lat_car = pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float)[::decimate] if ("lat_src" in data.columns) else None

    cols_cd = []
    if t_hover is not None:
        cols_cd.append(t_hover.astype("datetime64[ms]").astype(str))
    else:
        cols_cd.append(np.array([""] * len(xs), dtype=object))
    cols_cd.append(lon_loc)
    cols_cd.append(lat_loc)
    cols_cd.append(phi_car if phi_car is not None else np.full(len(xs), np.nan))
    cols_cd.append(lat_car if lat_car is not None else np.full(len(xs), np.nan))
    # radial distance in R_sun for hover (more intuitive than AU in the Sun-zoom cube)
    rloc_rsun = np.where(np.isfinite(rloc), rloc / float(r_sun_au), np.nan)
    cols_cd.append(rloc_rsun)
    custom_base = np.column_stack(cols_cd)

    base_hover = (
        "t=%{customdata[0]}<br>"
        "lon,lat (frame)=%{customdata[1]:.1f}°, %{customdata[2]:.1f}°<br>"
        "phi,lat (Carr)=%{customdata[3]:.1f}°, %{customdata[4]:.1f}°<br>"
        "r=%{customdata[5]:.2f} R⊙<br>"
    )

    for k, vname in enumerate(plot_vars):
        row = (k // ncols_vars) + 1
        col = (k % ncols_vars) + 1

        # ecliptic patch
        fig.add_trace(
            go.Mesh3d(
                x=px, y=py, z=pz,
                i=np.array([0, 0]), j=np.array([1, 2]), k=np.array([2, 3]),
                opacity=0.05,
                color="rgba(120,120,120,1.0)",
                flatshading=True,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # spheres
        fig.add_trace(
            go.Mesh3d(
                x=sx, y=sy, z=sz, i=si, j=sj, k=sk,
                opacity=0.16,
                color="rgb(180,180,180)",
                flatshading=True,
                lighting=dict(ambient=0.65, diffuse=0.55, specular=0.04, roughness=0.95, fresnel=0.02),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Mesh3d(
                x=rx, y=ry, z=rz, i=ri, j=rj, k=rk,
                opacity=0.07,
                color="rgb(130,130,130)",
                flatshading=True,
                lighting=dict(ambient=0.65, diffuse=0.55, specular=0.04, roughness=0.95, fresnel=0.02),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        if show_sphere_grid and (grid_sun is not None) and (grid_ss is not None):
            gx, gy, gz = grid_sun
            fig.add_trace(
                go.Scatter3d(
                    x=gx, y=gy, z=gz,
                    mode="lines",
                    line=dict(width=int(grid_width), color=f"rgba(0,0,0,{float(grid_opacity):.3f})"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            gx, gy, gz = grid_ss
            fig.add_trace(
                go.Scatter3d(
                    x=gx, y=gy, z=gz,
                    mode="lines",
                    line=dict(width=int(grid_width), color=f"rgba(0,0,0,{float(grid_opacity):.3f})", dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        # Sparse Carrington labels (on source surface only) if available
        if show_sphere_grid and (grid_labels is not None):
            lx, ly, lz, ltxt = grid_labels
            fig.add_trace(
                go.Scatter3d(
                    x=lx, y=ly, z=lz,
                    mode="text",
                    text=ltxt,
                    textfont=dict(size=11, color="rgba(0,0,0,0.55)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )


        ex, ey, ez = _equator_ring(float(r_sun_au))
        fig.add_trace(
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2, color="rgba(0,0,0,0.30)"), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )
        ex, ey, ez = _equator_ring(float(r_ss_au))
        fig.add_trace(
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2, color="rgba(0,0,0,0.24)", dash="dot"), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )
        mx, my, mz = _prime_meridian(float(r_ss_au))
        fig.add_trace(
            go.Scatter3d(x=mx, y=my, z=mz, mode="lines", line=dict(width=2, color="rgba(0,0,0,0.28)", dash="dash"), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )

        # spacecraft trajectory
        if have_sc and (xsc is not None) and (xsc_raw is not None):
            fig.add_trace(
                go.Scatter3d(
                    x=xsc, y=ysc, z=zsc,
                    customdata=np.column_stack([xsc_raw, ysc_raw, zsc_raw, np.sqrt(xsc_raw**2 + ysc_raw**2 + zsc_raw**2)/float(r_sun_au)]),
                    mode="lines",
                    line=dict(width=5, color="rgba(0,0,0,0.45)"),
                    opacity=0.55,
                    showlegend=False,
                    hovertemplate="SC (projected)<br>true x=%{customdata[0]:.3f} AU<br>true y=%{customdata[1]:.3f} AU<br>true z=%{customdata[2]:.3f} AU<br>r=%{customdata[3]:.2f} R⊙<extra></extra>",
                ),
                row=row,
                col=col,
            )

        _add_links(row=row, col=col)
        _add_rtn_axes(row=row, col=col)

        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers",
                marker=dict(size=5, color="rgba(0,0,0,0.6)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        if vname == "polarity":
            if "polarity" in data.columns:
                pol_ser = data["polarity"]
            elif "Br" in data.columns:
                pol_ser = np.sign(pd.to_numeric(data["Br"], errors="coerce"))
            else:
                pol_ser = pd.Series(np.nan, index=data.index)

            pol = pd.to_numeric(pol_ser, errors="coerce").to_numpy(dtype=float)[::decimate]
            colmap = np.where(pol > 0, "#d62728", np.where(pol < 0, "#1f77b4", "#7f7f7f"))

            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    customdata=np.column_stack([custom_base, pol]),
                    mode="markers",
                    marker=dict(size=4, color=colmap, opacity=0.92),
                    showlegend=False,
                    hovertemplate=base_hover + "polarity=%{customdata[6]:.0f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        else:
            spec = var_specs[vname]
            mode = str(spec.get("mode", "scalar")).lower().strip()
            if mode != "scalar":
                raise ValueError(f"3D scalar panel expected for {vname!r}, got mode={mode!r}.")

            vv = pd.to_numeric(data[vname], errors="coerce").to_numpy(dtype=float)[::decimate]
            vmin, vmax = _compute_scalar_limits(vv, spec=spec, percentiles=percentiles)

            scale = str(spec.get("scale", "linear")).lower().strip()
            cmap = spec.get("cmap", "viridis")
            colorscale = _mpl_cmap_to_plotly(cmap)

            vplot = vv.copy()
            cb = _colorbar_pos(row=row, col=col)
            cb_title = _label_with_unit(data, vname, spec.get("label", vname))

            if scale == "log":
                good = np.isfinite(vplot) & (vplot > 0)
                vplot[~good] = np.nan
                vplot = np.log10(vplot)
                # Limits in log-space
                vmin, vmax = _compute_scalar_limits(vplot, spec={"scale": "linear"}, percentiles=percentiles)
                cb_title = _log10_label(cb_title)

            marker = dict(
                size=4,
                color=vplot,
                colorscale=colorscale,
                cmin=float(vmin),
                cmax=float(vmax),
                opacity=0.94,
                colorbar=dict(
                    title=dict(text=cb_title, side="right"),
                    x=cb["x"],
                    y=cb["y"],
                    len=cb["len"],
                    thickness=10,
                    outlinewidth=0.0,
                    tickfont=dict(size=11),
                ),
            )

            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    customdata=np.column_stack([custom_base, vv]),
                    mode="markers",
                    marker=marker,
                    showlegend=False,
                    hovertemplate=base_hover + f"{vname}=%{{customdata[6]:.4g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

        if segs is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=segs[0], y=segs[1], z=segs[2],
                    mode="lines",
                    line=dict(width=3, color="rgba(0,0,0,0.30)"),
                    opacity=0.65,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    # -----------------------
    # Live camera sync (HTML post-script)
    # -----------------------
    sync_script = ""
    if sync_cameras and total_cells > 1:
        scene_names = ["scene"] + [f"scene{i}" for i in range(2, total_cells + 1)]
        scenes_js = ",".join([f"'{s}'" for s in scene_names])
        sync_script = f"""
var gd = document.getElementById('{{plot_id}}');
if(gd){{
  var _syncing = false;
  var _scenes = [{scenes_js}];
  gd.on('plotly_relayout', function(ev){{
    if(_syncing) return;
    if(!ev) return;
    var src = null;
    for (var k in ev) {{
      if (!k) continue;
      if (k.indexOf('.camera') > 0) {{ src = k.split('.')[0]; break; }}
    }}
    if(!src) return;
    var cam = ev[src + '.camera']
           || (gd.layout && gd.layout[src] && gd.layout[src].camera)
           || (gd._fullLayout && gd._fullLayout[src] && gd._fullLayout[src].camera);
    if(!cam) return;
    _syncing = true;
    var updates = {{}};
    for (var i=0; i<_scenes.length; i++) {{
      var s = _scenes[i];
      if (s === src) continue;
      updates[s + '.camera'] = cam;
    }}
    Plotly.relayout(gd, updates).then(function(){{ _syncing = false; }}).catch(function(){{ _syncing = false; }});
  }});
}}
"""

    # Standalone HTML should be self-contained (no CDN dependency) so that
    # it renders reliably when opened from disk.
    pio.write_html(
        fig,
        str(out_html),
        include_plotlyjs="inline",
        include_mathjax="cdn",
        full_html=True,
        post_script=sync_script if sync_script else None,
        config=dict(responsive=True, displaylogo=False),
        auto_open=False,
    )

    if show:
        try:
            from IPython.display import HTML, display  # type: ignore

            html = pio.to_html(
                fig,
                include_plotlyjs="inline",
                include_mathjax="cdn",
                full_html=False,
                post_script=sync_script if sync_script else None,
                config=dict(responsive=True, displaylogo=False),
                default_width="100%",
                default_height=f"{int(height)}px",
            )
            display(HTML(f"<div style='width:100%; margin:0; padding:0; overflow:hidden'>{html}</div>"))
        except Exception:
            fig.show()

    return out_html, fig