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
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

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
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite color limits.")
    if hi == lo:
        # Avoid zero-span colorscales (can hide the colorbar in Plotly).
        eps = 1e-12 if lo == 0.0 else 1e-6 * abs(lo)
        lo -= eps
        hi += eps
    return lo, hi


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
    clim: Optional[Tuple[float, float]] = None,
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

    if profile_panel is None:
        raise ValueError("plot_source_surface_2d requires profile_panel (mandatory U(r) panel).")

    # Traceability guard: the plotted U(r) must come from the executed model.
    sig_panel = None if profile_panel is None else profile_panel.get("executed_model_signature", None)
    sig_data = data.attrs.get("executed_model_signature", None)
    if (sig_panel is not None) and (sig_data is not None) and (str(sig_panel) != str(sig_data)):
        raise ValueError(
            "Traceability guard failed: profile_panel signature != data.attrs signature. "
            "This indicates compute/plot divergence."
        )

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
            cb.ax.set_facecolor("0.92")
            try:
                cb.outline.set_edgecolor("0.55")
            except Exception:
                pass
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
            # Optional: show the *time-varying* family U(r,t) used across the interval.
            # This makes it explicit whether we are using a single profile (shape+scale)
            # or a time-dependent scaling to V_bg(t) and r_sc(t).
            U_samp = profile_panel.get("U_samples_kms", None)
            t_hr = profile_panel.get("t_samples_hr", None)
            if U_samp is not None and t_hr is not None:
                try:
                    U_samp = np.asarray(U_samp, float)
                    t_hr = np.asarray(t_hr, float)
                    if U_samp.ndim == 2 and U_samp.shape[1] == r.size and U_samp.shape[0] >= 2:
                        # Keep the panel readable: cap number of profiles shown.
                        nshow = int(min(U_samp.shape[0], 14))
                        cmap = mpl.cm.get_cmap("viridis")
                        norm = Normalize(vmin=float(np.nanmin(t_hr[:nshow])), vmax=float(np.nanmax(t_hr[:nshow])))
                        for uu, tt in zip(U_samp[:nshow], t_hr[:nshow]):
                            if np.isfinite(uu).any() and np.isfinite(tt):
                                axp.plot(r, uu, lw=1.0, alpha=0.55, color=cmap(norm(float(tt))))
                        # Add a compact colorbar for time (hours since first shown sample).
                        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                        sm.set_array([])
                        cbt = plt.colorbar(sm, ax=axp, pad=0.015, fraction=0.045)
                        cbt.set_label("time [h]")
                except Exception:
                    pass

            # Median/profile envelope (representative diagnostic)
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

            # Also show a normalized profile to make weak acceleration visually obvious
            try:
                U_sc = float(U[-1])
                if np.isfinite(U_sc) and U_sc > 0.0:
                    axp2 = axp.twinx()
                    axp2.plot(r, U / U_sc, lw=1.2, alpha=0.6)
                    axp2.set_ylabel(r"$U/U_{\mathrm{sc}}$")
                    axp2.set_ylim(0.0, 1.05)
                    axp.text(0.02, 0.93, f"$U_{{\mathrm{{ss}}}}/U_{{\mathrm{{sc}}}}={U[0]/U_sc:.3f}$", transform=axp.transAxes)
            except Exception:
                pass
            axp.grid(True, alpha=0.25)
            # Annotate non-degeneracy diagnostics (if provided)
            try:
                umin = profile_panel.get("U_min_kms", None)
                umax = profile_panel.get("U_max_kms", None)
                span = profile_panel.get("U_span_kms", None)
                thr = profile_panel.get("U_span_thr_kms", None)
                deg = profile_panel.get("profile_degenerate", None)
                rea = profile_panel.get("degenerate_reason", None)
                if span is not None and thr is not None:
                    txt = "U_span={:.3g} km/s\nthr={:.3g} km/s\ndegenerate={}\n{}".format(
                        float(span), float(thr), str(bool(deg)), (str(rea) if rea is not None else "")
                    )
                    axp.text(
                        0.98,
                        0.02,
                        txt,
                        transform=axp.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", alpha=0.9),
                    )
            except Exception:
                pass

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



def plot_pfss_backmap_context_2d(
    *,
    data: pd.DataFrame,
    out_png: Union[str, Path],
    br2d: np.ndarray,
    which_br: str = "source_surface",
    neutral_lonlat: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    color_by: Optional[str] = None,
    percentiles: Tuple[float, float] = (2.0, 98.0),
    title: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> Path:
    """2D PFSS Br context map with backmapped footpoint track overlay.

    Geometry contract
    -----------------
    - br2d is assumed to be on a uniform Carrington lon/lat grid:
        lon in [0, 360), lat in [-90, 90], with array shape (nlat, nlon).
    - The backmapped coordinates are taken from:
        data['phi_src'] (deg, Carrington longitude) and data['lat_src'] (deg).
    - This function does not attempt to rotate between inertial/Carrington frames;
      it is intentionally strict and expects Carrington-consistent inputs.
    """
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    br = np.asarray(br2d, dtype=float)
    if br.ndim != 2:
        raise ValueError("br2d must be 2D (lat, lon)")
    nlat, nlon = br.shape

    lon_axis = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat_axis = np.linspace(-90.0, 90.0, nlat)

    # robust clim (or user-provided fixed clim)
    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
        # enforce symmetry for Br context unless the user explicitly breaks it
        mm = float(max(abs(vmin), abs(vmax)))
        if np.isfinite(mm) and mm > 0.0:
            vmin, vmax = -mm, +mm
        else:
            vmin, vmax = -1.0, +1.0
    else:
        lo_p, hi_p = float(percentiles[0]), float(percentiles[1])
        finite = np.isfinite(br)
        if finite.any():
            vmin = np.nanpercentile(br[finite], lo_p)
            vmax = np.nanpercentile(br[finite], hi_p)
            # symmetric about zero for Br context (helps polarity interpretation)
            m = float(max(abs(vmin), abs(vmax)))
            vmin, vmax = -m, +m
        else:
            vmin, vmax = -1.0, +1.0


    lon_fp = np.mod(pd.to_numeric(data["phi_src"], errors="coerce").to_numpy(dtype=float), 360.0)
    lat_fp = np.clip(pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float), -90.0, 90.0)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        br,
        origin="lower",
        extent=(0.0, 360.0, -90.0, 90.0),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )

    # neutral line overlay (optional)
    if neutral_lonlat is not None:
        try:
            nl_lon, nl_lat = neutral_lonlat
            nl_lon = np.mod(np.asarray(nl_lon, dtype=float), 360.0)
            nl_lat = np.clip(np.asarray(nl_lat, dtype=float), -90.0, 90.0)
            ax.plot(nl_lon, nl_lat, "-", linewidth=1.8, alpha=0.9)
        except Exception:
            pass

    # backmapped track (optionally colored)
    if (color_by is not None) and (str(color_by) in data.columns):
        c = pd.to_numeric(data[str(color_by)], errors="coerce").to_numpy(dtype=float)
        sc = ax.scatter(lon_fp, lat_fp, c=c, s=18.0, linewidths=0.0)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.ax.set_facecolor("0.92")
        try:
            cb.outline.set_edgecolor("0.55")
        except Exception:
            pass
        cb.set_label(str(color_by))
    else:
        ax.plot(lon_fp, lat_fp, "-", linewidth=1.5, alpha=0.9)
        ax.scatter(lon_fp[-1:], lat_fp[-1:], s=60.0)

    ax.set_xlabel("Carrington longitude [deg]")
    ax.set_ylabel("Heliographic latitude [deg]")
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)

    w = str(which_br).strip().lower()
    rlab = "photosphere" if w == "photosphere" else "source surface"
    if title is None:
        title = f"PFSS Br ({rlab}) + backmapped footpoints"
    ax.set_title(title)

    cb_pf = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb_pf.ax.set_facecolor("0.92")
    try:
        cb_pf.outline.set_edgecolor("0.55")
    except Exception:
        pass
    cb_pf.set_label(r"$B_r$ (arb.)")

    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return out_png


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

    try:
        if np.isfinite(U[[0,-1]]).all() and U[-1] != 0.0:
            ax0.text(0.02, 0.92, f'U_ss/U_sc={U[0]/U[-1]:.3f}', transform=ax0.transAxes)
    except Exception:
        pass

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

def _observer_coord(observer: str, obstime):
    """Return an observer coordinate suitable for HGC transforms.

    SunPy versions differ in whether `observer='earth'` (a string) is accepted.
    When possible, we construct an explicit observer coordinate.
    """
    if (not _HAS_SUNPY) or obstime is None:
        return observer

    obs = str(observer).strip().lower()
    try:
        from sunpy.coordinates import get_body_heliographic_stonyhurst
    except Exception:
        return observer

    if obs in {"earth", "sun"}:
        try:
            return get_body_heliographic_stonyhurst(obs, obstime)
        except Exception:
            return observer

    # Best effort for other named bodies
    try:
        return get_body_heliographic_stonyhurst(obs, obstime)
    except Exception:
        return observer


def _transform_lonlat_polyline_to_unit_xyz(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    radius_au: float,
    frame3d: str,
    obstime,
    observer: str = "earth",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a NaN-separated Carrington lon/lat polyline to unit xyz in frame3d.

    The input lon/lat are interpreted as Carrington (HGC). Disjoint segments should be
    separated by NaNs; those separators are preserved in the output so Plotly can break
    line segments correctly.
    """
    lon = np.asarray(lon_deg, dtype=float).reshape(-1)
    lat = np.asarray(lat_deg, dtype=float).reshape(-1)

    fin = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(fin):
        return np.full_like(lon, np.nan), np.full_like(lat, np.nan), np.full_like(lat, np.nan)

    f = str(frame3d).upper().strip()

    # HGC (or no SunPy): simple spherical conversion in the plot coordinate system.
    if (f not in {"HEE", "HCI"}) or (not _HAS_SUNPY) or (obstime is None):
        lonr = np.deg2rad(lon)
        latr = np.deg2rad(lat)
        cl = np.cos(latr)
        x = cl * np.cos(lonr)
        y = cl * np.sin(lonr)
        z = np.sin(latr)
        return x, y, z

    tf = _target_frame(f, obstime)
    if tf is None:
        lonr = np.deg2rad(lon)
        latr = np.deg2rad(lat)
        cl = np.cos(latr)
        x = cl * np.cos(lonr)
        y = cl * np.sin(lonr)
        z = np.sin(latr)
        return x, y, z

    obs_coord = _observer_coord(observer, obstime)

    X, Y, Z = [], [], []
    n = lon.size
    i = 0
    while i < n:
        if not fin[i]:
            i += 1
            continue
        j = i
        while j < n and fin[j]:
            j += 1

        seg_lon = lon[i:j]
        seg_lat = lat[i:j]
        try:
            cc = _SkyCoord(
                lon=seg_lon * _u.deg,
                lat=seg_lat * _u.deg,
                radius=float(radius_au) * _u.AU,
                frame=_HGC(obstime=obstime, observer=obs_coord),
            ).transform_to(tf)
            xau = cc.cartesian.x.to_value(_u.AU)
            yau = cc.cartesian.y.to_value(_u.AU)
            zau = cc.cartesian.z.to_value(_u.AU)
            rr = np.sqrt(xau * xau + yau * yau + zau * zau)
            rr = np.where(rr > 0.0, rr, np.nan)
            X.extend((xau / rr).tolist())
            Y.extend((yau / rr).tolist())
            Z.extend((zau / rr).tolist())
        except Exception:
            lonr = np.deg2rad(seg_lon)
            latr = np.deg2rad(seg_lat)
            cl = np.cos(latr)
            X.extend((cl * np.cos(lonr)).tolist())
            Y.extend((cl * np.sin(lonr)).tolist())
            Z.extend((np.sin(latr)).tolist())

        # Segment separator
        X.append(np.nan); Y.append(np.nan); Z.append(np.nan)
        i = j

    return np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)



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
            frame=_HGC(obstime=obstime, observer=_observer_coord(observer, obstime)),
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
            frame=_HGC(obstime=obstime, observer=_observer_coord(observer, obstime)),
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
            frame=_HGC(obstime=obstime, observer=_observer_coord(observer, obstime)),
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
            frame=_HGC(obstime=obstime, observer=_observer_coord(observer, obstime)),
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
    panel_px: int = 650,
    width: Optional[int] = None,
    height: Optional[int] = None,
    decimate: int = 1,
    # Optional: provide an independent spacecraft trajectory so the orbit line
    # remains visible even when cadence-grid samples are removed (e.g., gap masking).
    # Expected columns: sc_x_au, sc_y_au, sc_z_au (Cartesian AU in the plotting frame).
    sc_track: Optional[pd.DataFrame] = None,
    sc_track_decimate: Optional[int] = None,
    # radial scaling (visualization only)
    # - "zoom_au": legacy Sun-centered zoom cube in AU (may project SC to a shell)
    # - "linear":  true AU Cartesian coordinates (no projection)
    # - "log_r_over_Rsun": spherical-style radial compression with r_plot = log10(r/Rsun) + radial_log_offset
    radial_scale: str = "log_r_over_Rsun",
    radial_log_offset: float = 1.0,
    # spacecraft rendering inside the zoom cube
    sc_project_to_shell: bool = False,
    sc_shell_frac: float = 0.98,
    # panel framing / clarity
    draw_panel_boxes: bool = False,
    show_cube_edges: bool = False,
    show_links: bool = True,
    link_count: int = 12,
    link_line_rgba: str = "rgba(0,0,0,0.22)",
    show_rtn_axes: bool = True,
    rtn_axis_frac: float = 0.22,
    # ecliptic plane context (press-release friendly)
    show_ecliptic_circles: bool = True,
    ecliptic_circle_radii_au: Optional[Sequence[float]] = None,
    ecliptic_circle_max_au: Optional[float] = None,
    ecliptic_circle_count: int = 5,
    ecliptic_circle_spacing: str = "log",  # "linear" or "log" (log-spaced radii)
    ecliptic_circle_label_units: str = "Rsun",  # "Rsun", "AU", or "both"
    ecliptic_circle_rgba: str = "rgba(0,0,0,0.16)",
    ecliptic_circle_width: int = 2,
    show_ecliptic_axes: bool = True,
    # sphere latitude/longitude grid
    show_sphere_grid: bool = True,
    grid_lon_step_deg: int = 45,
    grid_lat_step_deg: int = 30,
    grid_opacity: float = 0.18,
    grid_width: int = 2,
    sphere_grid_frame: str = "carrington",
    sphere_grid_observer: str = "earth",
    show_grid_labels: bool = False,
    # uncertainty display
    show_uncertainty_arcs: bool = False,
    uncertainty_decimate: int = 10,
    # camera / interactivity
    sync_cameras: bool = True,
    camera: str = "iso",
    title: Optional[str] = None,
    show: bool = False,
    # Export control (used by movie renderer): if False, returns the figure without writing HTML
    write_html: bool = True,
    # Optional per-frame camera override (Plotly camera dict: {'eye': {...}, 'up': {...}, 'center': {...}})
    camera_dict: Optional[Dict[str, Any]] = None,
    # Optional per-panel colorbar marker values (raw variable values; one per plot variable).
    # Used by MP4 export to indicate the instantaneous value on each horizontal colorbar.
    cb_marker_values: Optional[Dict[str, float]] = None,
    # Optional highlight of the most recent sample (useful for movies).
    highlight_last_point: bool = False,
    highlight_size: int = 10,
    highlight_fill_rgba: str = 'rgba(255,255,255,0.92)',
    highlight_edge_rgba: str = 'rgba(0,0,0,0.95)',
    highlight_edge_width: int = 4,
    highlight_connector: bool = False,
    highlight_connector_rgba: str = 'rgba(0,0,0,0.55)',
    highlight_connector_width: int = 6,

# PFSS background (optional): Br map on either photosphere or source surface, in Carrington lon/lat.
# For physically meaningful overlays over long intervals, prefer pfss_date_mode='fixed' or 'interval_mid_day'
# upstream so the background is time-consistent.
pfss_br2d: Optional[np.ndarray] = None,
pfss_which_br: str = "source_surface",  # 'source_surface' or 'photosphere'
pfss_surface_stride: int = 2,
pfss_opacity: float = 0.35,
pfss_colorscale: Optional[Any] = None,  # Plotly colorscale; if None uses a diverging RdBu_r-like scale
pfss_clim: Optional[Tuple[float, float]] = None,  # if None uses symmetric robust percentiles
pfss_show_colorbar: bool = False,  # adds a single colorbar strip (first panel only)
pfss_show_in_all_panels: bool = True,
pfss_neutral_lonlat: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (lon_deg, lat_deg), typically Br=0 on source surface
pfss_neutral_rgba: str = "rgba(0,0,0,0.75)",
pfss_neutral_width: int = 3,
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
    if frame3d in {"CARRINGTON", "HGC"}:
        frame3d = "HGC"
    if frame3d not in {"HEE", "HCI", "HGC"}:
        raise ValueError("frame3d must be one of {'HEE','HCI','HGC'} (aliases: 'CARRINGTON' -> 'HGC')")

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

    # Optional separate spacecraft track (for continuous orbit rendering)
    track = None
    if sc_track is not None:
        try:
            track = sc_track.copy()
            if not isinstance(track.index, pd.DatetimeIndex):
                track.index = pd.to_datetime(track.index)
        except Exception:
            track = None
    have_track = (track is not None) and req_sc.issubset(track.columns)

    xs = pd.to_numeric(data["ss_x_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
    ys = pd.to_numeric(data["ss_y_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
    zs = pd.to_numeric(data["ss_z_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
    # Preserve physical (AU) coordinates for hover + sanity checks even when we re-scale for plotting.
    xs_phys = xs.copy()
    ys_phys = ys.copy()
    zs_phys = zs.copy()

    if have_sc:
        xsc_raw = pd.to_numeric(data["sc_x_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
        ysc_raw = pd.to_numeric(data["sc_y_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
        zsc_raw = pd.to_numeric(data["sc_z_au"], errors="coerce").to_numpy(dtype=float)[::decimate]
        xsc_phys = xsc_raw.copy()
        ysc_phys = ysc_raw.copy()
        zsc_phys = zsc_raw.copy()
    else:
        xsc_raw = ysc_raw = zsc_raw = None
        xsc_phys = ysc_phys = zsc_phys = None

    # Orbit track (may be denser and may include times where cadence-grid data were removed)
    xtrk_raw = ytrk_raw = ztrk_raw = None
    xtrk_phys = ytrk_phys = ztrk_phys = None
    dtrk = int(sc_track_decimate) if (sc_track_decimate is not None) else int(decimate)
    dtrk = max(1, dtrk)
    if have_track:
        xtrk_raw = pd.to_numeric(track["sc_x_au"], errors="coerce").to_numpy(dtype=float)[::dtrk]
        ytrk_raw = pd.to_numeric(track["sc_y_au"], errors="coerce").to_numpy(dtype=float)[::dtrk]
        ztrk_raw = pd.to_numeric(track["sc_z_au"], errors="coerce").to_numpy(dtype=float)[::dtrk]
        xtrk_phys = xtrk_raw.copy()
        ytrk_phys = ytrk_raw.copy()
        ztrk_phys = ztrk_raw.copy()

    # -----------------------
    # Radial scaling mode
    # -----------------------
    radial_mode = str(radial_scale).lower().strip()
    if radial_mode in {"zoom_cube", "zoom_au", "cube", "cube_au"}:
        radial_mode = "zoom_au"
    if radial_mode not in {"zoom_au", "linear", "log_r_over_rsun"}:
        raise ValueError('radial_scale must be one of {"zoom_au","linear","log_r_over_Rsun"}.')

    # Context extent: by default, show at least 1 AU (paper-style context), or the max SC radius.
    r_sc_max_au = float("nan")
    rr_list = []
    if have_sc and (xsc_phys is not None):
        rr_sc = np.sqrt(xsc_phys * xsc_phys + ysc_phys * ysc_phys + zsc_phys * zsc_phys)
        rr_list.append(rr_sc)
    if have_track and (xtrk_phys is not None):
        rr_trk = np.sqrt(xtrk_phys * xtrk_phys + ytrk_phys * ytrk_phys + ztrk_phys * ztrk_phys)
        rr_list.append(rr_trk)
    if rr_list:
        # NOTE: rr_list may be non-empty but still contain no finite samples.
        # Guard against np.concatenate([]) which raises ValueError.
        rr_parts = [r[np.isfinite(r)] for r in rr_list if (r is not None) and np.isfinite(r).any()]
        if rr_parts:
            rr_all = np.concatenate(rr_parts)
            if rr_all.size:
                r_sc_max_au = float(np.nanmax(rr_all))

    if ecliptic_circle_max_au is None:
        r_ctx_max_au = float(max(1.0, 1.05 * (r_sc_max_au if np.isfinite(r_sc_max_au) else float(r_ss_au))))
    else:
        r_ctx_max_au = float(max(0.0, float(ecliptic_circle_max_au)))

    # Helper: map physical AU radius -> plot radius
    def _r_au_to_plot(rr_au: float) -> float:
        rr = float(rr_au)
        if radial_mode in {"zoom_au", "linear"}:
            return rr
        off = float(radial_log_offset)
        if not np.isfinite(off) or off <= 0:
            raise ValueError("radial_log_offset must be finite and > 0 for log_r_over_Rsun mode.")
        return float(np.log10(rr / max(float(r_sun_au), 1e-30)) + off)

    def _xyz_to_plot(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        if radial_mode in {"zoom_au", "linear"}:
            return x, y, z
        r = np.sqrt(x * x + y * y + z * z)
        ok = np.isfinite(r) & (r > 0)
        outx = np.full_like(x, np.nan, dtype=float)
        outy = np.full_like(y, np.nan, dtype=float)
        outz = np.full_like(z, np.nan, dtype=float)
        r_plot = np.full_like(r, np.nan, dtype=float)
        r_plot[ok] = np.log10(r[ok] / max(float(r_sun_au), 1e-30)) + float(radial_log_offset)
        r_plot = np.maximum(r_plot, 1e-6)
        outx[ok] = x[ok] / r[ok] * r_plot[ok]
        outy[ok] = y[ok] / r[ok] * r_plot[ok]
        outz[ok] = z[ok] / r[ok] * r_plot[ok]
        return outx, outy, outz

    def _scale_xyz_arrays(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _xyz_to_plot(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))

    # Plot radii for Sun/SS spheres
    r_sun_plot = _r_au_to_plot(float(r_sun_au))
    r_ss_plot = _r_au_to_plot(float(r_ss_au))

    # Plot coordinates
    xs, ys, zs = _xyz_to_plot(xs_phys, ys_phys, zs_phys)

    # Project the mapped source-surface points onto the plotted SS sphere.
    # This removes any tiny radial drift introduced by numerical transforms and makes
    # the 3D backmapping look visually "attached" to the source surface.
    rr_ss = np.sqrt(xs * xs + ys * ys + zs * zs)
    ok_ss = np.isfinite(rr_ss) & (rr_ss > 0.0) & np.isfinite(float(r_ss_plot))
    if np.any(ok_ss):
        xs[ok_ss] = (xs[ok_ss] / rr_ss[ok_ss]) * float(r_ss_plot)
        ys[ok_ss] = (ys[ok_ss] / rr_ss[ok_ss]) * float(r_ss_plot)
        zs[ok_ss] = (zs[ok_ss] / rr_ss[ok_ss]) * float(r_ss_plot)

    xsc = ysc = zsc = None
    if have_sc and (xsc_phys is not None):
        xsc, ysc, zsc = _xyz_to_plot(xsc_phys, ysc_phys, zsc_phys)

    xtrk = ytrk = ztrk = None
    if have_track and (xtrk_phys is not None):
        xtrk, ytrk, ztrk = _xyz_to_plot(xtrk_phys, ytrk_phys, ztrk_phys)

    # Panel extent
    if radial_mode == "zoom_au":
        lim = float(max(float(sun_zoom_au), 1.25 * float(r_ss_au), 2.2 * float(r_sun_au)))
        lim = float(max(lim, 1.05 * float(r_ss_au)))
        shell_r = float(max(0.0, float(sc_shell_frac))) * lim
        if have_sc and (xsc_raw is not None) and sc_project_to_shell:
            r = np.sqrt(xsc_raw * xsc_raw + ysc_raw * ysc_raw + zsc_raw * zsc_raw)
            ok = np.isfinite(r) & (r > 0)
            xsc = np.full_like(xsc_raw, np.nan, dtype=float)
            ysc = np.full_like(ysc_raw, np.nan, dtype=float)
            zsc = np.full_like(zsc_raw, np.nan, dtype=float)
            xsc[ok] = (xsc_raw[ok] / r[ok]) * shell_r
            ysc[ok] = (ysc_raw[ok] / r[ok]) * shell_r
            zsc[ok] = (zsc_raw[ok] / r[ok]) * shell_r

        # Apply the same projection to the orbit track if present.
        if have_track and (xtrk_raw is not None) and (xtrk is not None) and sc_project_to_shell:
            r = np.sqrt(xtrk_raw * xtrk_raw + ytrk_raw * ytrk_raw + ztrk_raw * ztrk_raw)
            ok = np.isfinite(r) & (r > 0)
            xtrk = np.full_like(xtrk_raw, np.nan, dtype=float)
            ytrk = np.full_like(ytrk_raw, np.nan, dtype=float)
            ztrk = np.full_like(ztrk_raw, np.nan, dtype=float)
            xtrk[ok] = (xtrk_raw[ok] / r[ok]) * shell_r
            ytrk[ok] = (ytrk_raw[ok] / r[ok]) * shell_r
            ztrk[ok] = (ztrk_raw[ok] / r[ok]) * shell_r
    elif radial_mode == "linear":
        lim = float(1.05 * max(r_ctx_max_au, float(r_ss_au), float(r_sun_au)))
    else:
        lim = float(1.05 * max(_r_au_to_plot(r_ctx_max_au), r_ss_plot, r_sun_plot))

    # Ecliptic-plane context: concentric circles in the x–y plane (z=0).
    # NOTE: true log-log scaling is not meaningful in this Sun-centered Cartesian view (x and y change sign).
    # Instead, we optionally *log-space the ring radii* to improve visibility close to the Sun.
    spacing = str(ecliptic_circle_spacing).lower().strip()
    if spacing not in {"linear", "log"}:
        raise ValueError("ecliptic_circle_spacing must be 'linear' or 'log'")

    if ecliptic_circle_radii_au is None:
        anchors = [float(r_sun_au), float(r_ss_au)]
        try:
            n_extra = int(max(0, int(ecliptic_circle_count)))
        except Exception:
            n_extra = 5
        extra = []
        if n_extra > 0:
            rmax = float(r_ctx_max_au)
            if spacing == "log":
                lo = max(float(r_ss_au), 1.05 * float(r_sun_au), 1e-12)
                hi = max(lo * 1.01, rmax)
                extra = np.geomspace(lo, hi, n_extra)
            else:
                extra = np.linspace(float(r_ss_au), rmax, n_extra)
        base = anchors + [float(v) for v in extra]
        radii_au = sorted({float(v) for v in base if np.isfinite(v) and float(v) > 0.0})
    else:
        radii_au = sorted({float(v) for v in ecliptic_circle_radii_au if np.isfinite(float(v)) and float(v) > 0.0})

    radii_au = [r for r in radii_au if _r_au_to_plot(r) < 0.99 * float(lim)]

    # spheres (in plotting coordinates)
    sx, sy, sz, si, sj, sk = _sphere_mesh3d(float(r_sun_plot), n_lat=48, n_lon=96)
    rx, ry, rz, ri, rj, rk = _sphere_mesh3d(float(r_ss_plot), n_lat=48, n_lon=96)

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
                    if radial_mode == "log_r_over_rsun":
                        segs = _scale_xyz_arrays(segs[0], segs[1], segs[2])

# -----------------------
    # Orientation axes (plotting-frame basis; avoids ambiguous 'RTN' labeling)
    # -----------------------
    rtn = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    axis_len = float(max(0.0, rtn_axis_frac)) * float(lim)
    # Ensure orientation axes extend beyond the source-surface sphere in all radial modes
    # (important in log-radial visualization where r_ss_plot can exceed rtn_axis_frac*lim).
    axis_len = float(max(axis_len, 1.25 * float(r_ss_plot)))
    label_len = float(1.12 * axis_len)


    # -----------------------
    # Subplot grid
    # -----------------------
    from plotly.subplots import make_subplots

    nvars = len(plot_vars)
    ncols_vars = int(max(1, ncols_vars))
    nrows = int(np.ceil(nvars / ncols_vars))
    total_cells = int(nrows * ncols_vars)

    # Figure sizing: keep subplot domains close to square so spheres stay spherical.
    try:
        _pp = int(max(520, int(panel_px)))
    except Exception:
        _pp = 650
    if width is None or int(width) <= 0:
        width = int(_pp * ncols_vars)
    if height is None or int(height) <= 0:
        height = int(_pp * nrows + 90)

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

    # Radial scaling note (plotting coordinates can be non-physical in log mode).
    if radial_mode == "log_r_over_rsun":
        fig.add_annotation(
            x=0.99,
            y=0.99,
            xref="paper",
            yref="paper",
            text=f"radial scale: r_plot = log10(r/R⊙) + {float(radial_log_offset):.3g}",
            showarrow=False,
            align="right",
            font=dict(size=13, color="rgba(0,0,0,0.70)"),
        )

    # We intentionally avoid panel titles for long-term clarity.
    fig.layout.annotations = []


    # Reserve a thin strip above each 3D panel for a custom *horizontal* colorbar.
    # We draw the colorbar in paper coordinates inside this strip, so it never intrudes
    # into the data when the user zooms/rotates the 3D scene.
    _CB_STRIP_FRAC = 0.18 if (pfss_br2d is not None and bool(pfss_show_colorbar)) else 0.12  # fraction of each panel's domain height
    _panel_strip = {}
    for _i in range(1, total_cells + 1):
        _sname = "scene" if _i == 1 else f"scene{_i}"
        _dom = getattr(fig.layout, _sname).domain
        _x0, _x1 = float(_dom.x[0]), float(_dom.x[1])
        _y0, _y1 = float(_dom.y[0]), float(_dom.y[1])
        _h = max(1e-9, _y1 - _y0)
        _strip = float(_CB_STRIP_FRAC) * _h
        _y1_scene = _y1 - _strip
        getattr(fig.layout, _sname).domain = dict(x=[_x0, _x1], y=[_y0, _y1_scene])
        _row_i = (_i - 1) // ncols_vars + 1
        _col_i = (_i - 1) % ncols_vars + 1
        _panel_strip[(_row_i, _col_i)] = dict(x0=_x0, x1=_x1, y0_scene=_y0, y1_scene=_y1_scene, y0_strip=_y1_scene, y1_strip=_y1)

    # Always show the median spacecraft heliocentric distance on-figure (not hover-only).
    if r_sc_med_rsun is not None and np.isfinite(float(r_sc_med_rsun)):
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text=f"r_sc,med={float(r_sc_med_rsun):.1f} R_sun",
            showarrow=False, align="left",
            font=dict(size=14, color="rgba(0,0,0,0.75)"),
        )


    if any(v == "polarity" for v in plot_vars):
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers", marker=dict(size=6, symbol="square", color="#1f77b4"), name="polarity = -1", showlegend=True))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers", marker=dict(size=6, symbol="square", color="#7f7f7f"), name="polarity = 0", showlegend=True))
        fig.add_trace(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers", marker=dict(size=6, symbol="square", color="#d62728"), name="polarity = +1", showlegend=True))

    def _scene_axes(_ttl: str = ""):
        # Minimal 3D aesthetic: no ticks/background planes (presentation-first).
        # Axis titles are removed to avoid clutter (e.g., 'x [scaled]').
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
    # Axis titles are intentionally suppressed (presentation-first).
    _xlab, _ylab, _zlab = "", "", ""

    def _cam(which: str):
        w = str(which).lower().strip()
        if w == "top":
            return dict(eye=dict(x=0.0, y=0.0, z=1.95))
        if w == "side":
            return dict(eye=dict(x=1.95, y=0.0, z=0.0))
        if w in {"ecliptic", "xy"}:
            return dict(eye=dict(x=1.55, y=1.20, z=0.22))
        return dict(eye=dict(x=1.25, y=1.05, z=0.70))

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
                xaxis=_scene_axes(_xlab),
                yaxis=_scene_axes(_ylab),
                zaxis=_scene_axes(_zlab),
                aspectmode="cube",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            row=row,
            col=col,
        )
        fig.update_scenes(camera=_cam(camera), row=row, col=col)
        if camera_dict is not None:
            fig.update_scenes(camera=camera_dict, row=row, col=col)
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
        if (not show_rtn_axes) or (rtn is None) or (float(lim) <= 0):
            return

        # Keep axis directions physical (frame basis), but place axis labels using a
        # small perpendicular + out-of-ecliptic offset to avoid overlapping the
        # ecliptic distance-ring labels (which sit on +X, z=0).
        R, T, N = rtn
        colr = "rgba(0,0,0,0.35)"
        lw = 5

        labx = f"{frame3d} +X"
        laby = f"{frame3d} +Y"
        labz = f"{frame3d} +Z"

        lim_f = float(lim)

        # Push axes clearly beyond the SS sphere, but keep endpoints inside the scene range.
        axis_line_len = float(max(float(rtn_axis_frac) * lim_f, 1.85 * float(r_ss_plot)))
        axis_line_len = float(min(0.90 * lim_f, axis_line_len))
        label_base_len = float(min(0.965 * lim_f, 1.12 * axis_line_len))

        # Perpendicular offset magnitude for labels (kept modest so labels remain in-frame).
        off = 0.085 * lim_f

        def _norm(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return (v / n) if (n > 0) else v

        # "Clever" part: choose a perpendicular direction that moves labels away from
        # the ecliptic-ring label rail at y=0,z=0. Add +z for X/Y labels.
        perp_for = {
            labx: _norm(np.array([0.0, +1.0, +1.2])),
            laby: _norm(np.array([-1.0, 0.0, +1.2])),
            labz: _norm(np.array([+1.0, +1.0, 0.0])),
        }

        for nm, vec in [(labx, R), (laby, T), (labz, N)]:
            vec = _norm(np.asarray(vec, dtype=float))
            vx, vy, vz = (axis_line_len * vec).tolist()

            base = label_base_len * vec
            perp = perp_for.get(nm, _norm(np.array([1.0, 1.0, 1.0])))
            tx, ty, tz = (base + off * perp).tolist()

            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, vx], y=[0.0, vy], z=[0.0, vz],
                    mode="lines",
                    line=dict(width=lw, color=colr),
                    opacity=0.45,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[tx], y=[ty], z=[tz],
                    mode="text",
                    text=[nm],
                    textfont=dict(size=16, color=colr),
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
    grid_labels = None
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

    # If we are using a non-linear radial visualization, re-scale grid lines into plot space.
    if radial_mode == "log_r_over_rsun":
        if grid_sun is not None:
            grid_sun = _scale_xyz_arrays(grid_sun[0], grid_sun[1], grid_sun[2])
        if grid_ss is not None:
            grid_ss = _scale_xyz_arrays(grid_ss[0], grid_ss[1], grid_ss[2])
        if grid_labels is not None:
            lx, ly, lz, ltxt = grid_labels
            lx2, ly2, lz2 = _scale_xyz_arrays(lx, ly, lz)
            grid_labels = (lx2, ly2, lz2, ltxt)


    def _colorbar_pos(row: int, col: int):
        idx1 = (row - 1) * ncols_vars + col
        sname = "scene" if idx1 == 1 else f"scene{idx1}"
        dom = getattr(fig.layout, sname).domain
        x0, x1 = float(dom.x[0]), float(dom.x[1])
        y0, y1 = float(dom.y[0]), float(dom.y[1])
        # Place colorbars *inside* each panel near the right edge to avoid
        # giant outer margins (keeps the subplot grid compact).
        # We use xanchor='right' on the colorbar so the bar extends to the *left*.
        x = max(x0 + 0.02, min(0.995, x1 - 0.006))
        y = 0.5 * (y0 + y1)
        ln = 0.66 * (y1 - y0)
        return dict(x=x, y=y, len=ln)

    def _log10_label(lbl: str) -> str:
        s = str(lbl)
        if s.startswith("$") and s.endswith("$") and len(s) >= 2:
            inner = s[1:-1]
            return f"$\\log_{10}\\left({inner}\\right)$"
        return f"log10({s})"

    

    def _parse_rgb(_c: str) -> tuple[float, float, float]:
        s = str(_c).strip()
        if s.startswith("#") and len(s) in {7, 9}:
            r = int(s[1:3], 16)
            g = int(s[3:5], 16)
            b = int(s[5:7], 16)
            return float(r), float(g), float(b)
        if s.lower().startswith("rgb"):
            q = s[s.find("(") + 1 : s.find(")")]
            parts = [p.strip() for p in q.split(",") if p.strip()]
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
        return 128.0, 128.0, 128.0

    def _rgb_str(rgb: tuple[float, float, float]) -> str:
        r, g, b = rgb
        r = int(max(0, min(255, round(float(r)))))
        g = int(max(0, min(255, round(float(g)))))
        b = int(max(0, min(255, round(float(b)))))
        return f"rgb({r},{g},{b})"

    def _colorscale_color(colorscale: list, t: float) -> str:
        if not colorscale:
            return "rgb(128,128,128)"
        tt = float(max(0.0, min(1.0, t)))
        xs = [float(p[0]) for p in colorscale]
        cs = [p[1] for p in colorscale]
        if tt <= xs[0]:
            return str(cs[0])
        if tt >= xs[-1]:
            return str(cs[-1])
        j = 0
        for k in range(len(xs) - 1):
            if xs[k] <= tt <= xs[k + 1]:
                j = k
                break
        x0, x1 = float(xs[j]), float(xs[j + 1])
        c0 = _parse_rgb(cs[j])
        c1 = _parse_rgb(cs[j + 1])
        if (not np.isfinite(x1 - x0)) or (x1 == x0):
            return _rgb_str(c0)
        a = (tt - x0) / (x1 - x0)
        rr = c0[0] + a * (c1[0] - c0[0])
        gg = c0[1] + a * (c1[1] - c0[1])
        bb = c0[2] + a * (c1[2] - c0[2])
        return _rgb_str((rr, gg, bb))

    def _fmt_tick(v: float) -> str:
        try:
            return f"{float(v):.3g}"
        except Exception:
            return str(v)

    def _add_horizontal_colorbar(*, row: int, col: int, vmin: float, vmax: float, colorscale: list, title: str, nticks: int = 3, marker_value: Optional[float] = None, marker_label: str = '', slot: int = 0, nslots: int = 1) -> None:
        g = _panel_strip.get((int(row), int(col)), None)
        if g is None:
            return

        x0, x1 = float(g["x0"]), float(g["x1"])
        ys0, ys1 = float(g["y0_strip"]), float(g["y1_strip"])
        # Support stacking multiple colorbars in the reserved strip area (e.g., PFSS + variable).
        ns = int(max(1, int(nslots)))
        sl = int(max(0, min(ns - 1, int(slot))))
        hh_full = max(1e-9, ys1 - ys0)
        ys0 = ys0 + (sl / ns) * hh_full
        ys1 = ys0 + (1.0 / ns) * hh_full

        ww = max(1e-9, x1 - x0)
        hh = max(1e-9, ys1 - ys0)

        bx0 = x0 + 0.07 * ww
        bx1 = x1 - 0.03 * ww
        by0 = ys0 + 0.30 * hh
        by1 = ys0 + 0.60 * hh

        nseg = 48
        for i in range(nseg):
            t0 = i / nseg
            t1 = (i + 1) / nseg
            c = _colorscale_color(colorscale, 0.5 * (t0 + t1))
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=bx0 + t0 * (bx1 - bx0),
                x1=bx0 + t1 * (bx1 - bx0),
                y0=by0,
                y1=by1,
                line=dict(width=0),
                fillcolor=c,
                layer="above",
            )

        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=bx0,
            x1=bx1,
            y0=by0,
            y1=by1,
            line=dict(color="rgba(0,0,0,0.25)", width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

        # Optional instantaneous-value marker (used by MP4 export).
        if marker_value is not None and np.isfinite(float(marker_value)) and (float(vmax) != float(vmin)):
            tt = (float(marker_value) - float(vmin)) / (float(vmax) - float(vmin))
            tt = float(max(0.0, min(1.0, tt)))
            xm = bx0 + tt * (bx1 - bx0)
            fig.add_shape(
                type='line',
                xref='paper',
                yref='paper',
                x0=xm, x1=xm,
                y0=by0, y1=by1,
                line=dict(color='rgba(0,0,0,0.92)', width=2),
                layer='above',
            )
            if str(marker_label).strip():
                fig.add_annotation(
                    x=xm,
                    y=ys0 + 0.90 * hh,
                    xref='paper',
                    yref='paper',
                    text=str(marker_label),
                    showarrow=False,
                    xanchor='center',
                    yanchor='bottom',
                    font=dict(size=11, color='rgba(0,0,0,0.85)'),
                )

        fig.add_annotation(
            x=0.5 * (bx0 + bx1),
            y=ys0 + 0.78 * hh,
            xref="paper",
            yref="paper",
            text=str(title),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=12, color="rgba(0,0,0,0.85)"),
        )

        nt = int(max(2, nticks))
        tick_y = ys0 + 0.08 * hh
        for tt in np.linspace(0.0, 1.0, nt):
            xv = bx0 + tt * (bx1 - bx0)
            val = float(vmin) + tt * (float(vmax) - float(vmin))
            fig.add_annotation(
                x=xv,
                y=tick_y,
                xref="paper",
                yref="paper",
                text=_fmt_tick(val),
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=11, color="rgba(0,0,0,0.72)"),
            )

    # -----------------------
    # Optional PFSS background on a sphere (same geometry as the backmapped points).
    # This is purely a visualization overlay: the PFSS Br map is defined in Carrington lon/lat
    # and is rotated into the requested plotting frame at a representative obstime (median).
    # -----------------------
    pfss_surface = None  # dict with keys X,Y,Z,BR,cmin,cmax,colorscale,opacity,on
    pfss_neutral_xyz = None  # (x,y,z) arrays for a Br=0 neutral line polyline
    if pfss_br2d is not None:
        try:
            pfss_on = str(pfss_which_br).strip().lower()
            if pfss_on not in {"source_surface", "photosphere"}:
                pfss_on = "source_surface"
            stride_pf = int(max(1, int(pfss_surface_stride)))

            br_pf0 = np.asarray(pfss_br2d, dtype=float)
            br_pf = br_pf0[::stride_pf, ::stride_pf]
            nlat_pf, nlon_pf = br_pf.shape

            lon_pf = np.linspace(0.0, 360.0, int(nlon_pf), endpoint=False, dtype=float)
            lat_pf = np.linspace(-90.0, 90.0, int(nlat_pf), endpoint=True, dtype=float)

            lon_pf2 = np.concatenate([lon_pf, [360.0]])
            br_pf2 = np.concatenate([br_pf, br_pf[:, :1]], axis=1)

            lon_g, lat_g = np.meshgrid(lon_pf2, lat_pf)

            radius_au_pf = float(r_ss_au) if (pfss_on == "source_surface") else float(r_sun_au)
            radius_plot_pf = float(_r_au_to_plot(radius_au_pf))

            if pfss_clim is None:
                vv = br_pf2[np.isfinite(br_pf2)]
                if vv.size > 0:
                    lo, hi = np.nanpercentile(vv, [2.0, 98.0])
                    mm = float(max(abs(float(lo)), abs(float(hi))))
                    cmin_pf, cmax_pf = -mm, mm
                else:
                    cmin_pf, cmax_pf = -1.0, 1.0
            else:
                cmin_pf, cmax_pf = float(pfss_clim[0]), float(pfss_clim[1])

            if pfss_colorscale is None:
                # Ensure +Br -> red and -Br -> blue in Plotly (diverging convention).
                try:
                    from .pfss import _normalize_pfss_plotly_colorscale as _norm_pfss_scale  # noqa: WPS433
                    pfss_colorscale_use = _norm_pfss_scale("RdBu")
                except Exception:
                    # Fallback to the reversed alias if Plotly understands it; otherwise Plotly will use its default diverging scale.
                    pfss_colorscale_use = "RdBu_r"
            else:
                pfss_colorscale_use = pfss_colorscale

            lonr = np.deg2rad(lon_g)
            latr = np.deg2rad(lat_g)
            ux = np.cos(latr) * np.cos(lonr)
            uy = np.cos(latr) * np.sin(lonr)
            uz = np.sin(latr)

            obstime_pfss = _median_obstime(data)
            if (str(frame3d).upper().strip() in {"HEE", "HCI"}) and _HAS_SUNPY and (obstime_pfss is not None):
                tf = _target_frame(str(frame3d), obstime_pfss)
                if tf is not None:
                    try:
                        cc = _SkyCoord(
                            lon=lon_g.reshape(-1) * _u.deg,
                            lat=lat_g.reshape(-1) * _u.deg,
                            radius=(radius_au_pf * _u.AU),
                            frame=_HGC(obstime=obstime_pfss, observer=_observer_coord(str(sphere_grid_observer), obstime_pfss)),
                        ).transform_to(tf)
                        xau = cc.cartesian.x.to_value(_u.AU).reshape(lon_g.shape)
                        yau = cc.cartesian.y.to_value(_u.AU).reshape(lon_g.shape)
                        zau = cc.cartesian.z.to_value(_u.AU).reshape(lon_g.shape)
                        rr = np.sqrt(xau * xau + yau * yau + zau * zau)
                        rr = np.where(rr > 0.0, rr, np.nan)
                        ux = xau / rr
                        uy = yau / rr
                        uz = zau / rr
                    except Exception:
                        pass

            Xpf = radius_plot_pf * ux
            Ypf = radius_plot_pf * uy
            Zpf = radius_plot_pf * uz

            pfss_surface = dict(
                X=Xpf,
                Y=Ypf,
                Z=Zpf,
                BR=br_pf2,
                cmin=float(cmin_pf),
                cmax=float(cmax_pf),
                colorscale=pfss_colorscale_use,
                opacity=float(pfss_opacity),
                on=pfss_on,
            )

            if (pfss_neutral_lonlat is not None):
                nl_lon, nl_lat = pfss_neutral_lonlat
                nl_lon = np.asarray(nl_lon, dtype=float).reshape(-1)
                nl_lat = np.asarray(nl_lat, dtype=float).reshape(-1)

                # Neutral lines are stored as NaN-separated polylines; preserve NaNs to keep
                # contour segments disjoint in Plotly.
                if nl_lon.size and nl_lat.size:
                    nl_lon = np.mod(nl_lon, 360.0)
                    nl_lat = np.clip(nl_lat, -90.0, 90.0)

                    # Draw the neutral line at the *source surface* by default (HCS proxy),
                    # even if the PFSS texture is on the photosphere.
                    radius_au_nl = float(r_ss_au)
                    radius_plot_nl = float(_r_au_to_plot(radius_au_nl))

                    ux2, uy2, uz2 = _transform_lonlat_polyline_to_unit_xyz(
                        lon_deg=nl_lon,
                        lat_deg=nl_lat,
                        radius_au=radius_au_nl,
                        frame3d=str(frame3d),
                        obstime=obstime_pfss,
                        observer=str(sphere_grid_observer),
                    )

                    xnl = radius_plot_nl * ux2
                    ynl = radius_plot_nl * uy2
                    znl = radius_plot_nl * uz2
                    pfss_neutral_xyz = (xnl, ynl, znl)
        except Exception:
            pfss_surface = None
            pfss_neutral_xyz = None

    # local lon/lat for hover: ALWAYS computed from physical AU coordinates (not plot-scaled)
    rloc_phys = np.sqrt(xs_phys * xs_phys + ys_phys * ys_phys + zs_phys * zs_phys)
    lon_loc = (np.degrees(np.arctan2(ys_phys, xs_phys)) + 360.0) % 360.0
    lat_loc = np.degrees(np.arcsin(np.where(rloc_phys > 0, zs_phys / rloc_phys, np.nan)))

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
    rloc_rsun = np.where(np.isfinite(rloc_phys), rloc_phys / float(r_sun_au), np.nan)
    sc_hover_label = "SC (projected)" if (radial_mode == "zoom_au" and bool(sc_project_to_shell)) else "SC"
    cols_cd.append(rloc_rsun)
    custom_base = np.column_stack(cols_cd)

    base_hover = (
        "t=%{customdata[0]}<br>"
        "lon,lat (frame)=%{customdata[1]:.1f}°, %{customdata[2]:.1f}°<br>"
        "phi,lat (Carr)=%{customdata[3]:.1f}°, %{customdata[4]:.1f}°<br>"
        "r=%{customdata[5]:.2f} R⊙<br>"
    )

    # ------------------------------------------------------------------
    # Render backmapped points as *actual geometry on the sphere*.
    #
    # Plotly Scatter3d markers (especially "square") are billboarded (screen-aligned).
    # Large billboard markers make a curved spherical track look visually flat.
    #
    # Fix: approximate each point by a tiny square patch (two triangles) oriented by
    # the local tangent basis and re-normalized to sit exactly on the sphere.
    #
    # This helper is vectorized: no per-point Python loops.
    # ------------------------------------------------------------------
    def _sphere_patches_mesh_from_xyz(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        size_deg: float,
    ):
        """Build an aggregated Mesh3d patch set from xyz points on a sphere.

        Returns (xv, yv, zv, i, j, k, ok_idx). Each input point contributes 4 vertices
        and 2 triangles. ok_idx are indices into the input arrays for retained points.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        z = np.asarray(z, dtype=float).reshape(-1)

        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(ok):
            return (np.asarray([np.nan] * 4), np.asarray([np.nan] * 4), np.asarray([np.nan] * 4),
                    np.asarray([0, 0], int), np.asarray([1, 2], int), np.asarray([2, 3], int),
                    np.asarray([], int))

        ok_idx = np.where(ok)[0]
        x0 = x[ok]
        y0 = y[ok]
        z0 = z[ok]
        rr = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
        ok2 = np.isfinite(rr) & (rr > 0)
        if not np.any(ok2):
            return (np.asarray([np.nan] * 4), np.asarray([np.nan] * 4), np.asarray([np.nan] * 4),
                    np.asarray([0, 0], int), np.asarray([1, 2], int), np.asarray([2, 3], int),
                    np.asarray([], int))

        ok_idx = ok_idx[ok2]
        x0 = x0[ok2]
        y0 = y0[ok2]
        z0 = z0[ok2]
        rr = rr[ok2]

        u = np.column_stack([x0 / rr, y0 / rr, z0 / rr])

        # Tangent basis: e1 = zhat x u (fallback to yhat x u near poles).
        zhat = np.array([0.0, 0.0, 1.0], dtype=float)
        yhat = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = np.cross(zhat[None, :], u)
        n1 = np.linalg.norm(e1, axis=1)
        pole = n1 < 1e-10
        if np.any(pole):
            e1[pole] = np.cross(yhat[None, :], u[pole])
            n1[pole] = np.linalg.norm(e1[pole], axis=1)
        n1 = np.where(n1 > 0, n1, 1.0)
        e1 = e1 / n1[:, None]
        e2 = np.cross(u, e1)
        n2 = np.linalg.norm(e2, axis=1)
        n2 = np.where(n2 > 0, n2, 1.0)
        e2 = e2 / n2[:, None]

        # Square half-width in radians -> tangent-plane scale a = tan(hh).
        hh = np.deg2rad(float(max(0.2, size_deg))) * 0.5
        a = float(np.tan(hh))

        sx = np.array([-1.0, +1.0, +1.0, -1.0], dtype=float)
        sy = np.array([-1.0, -1.0, +1.0, +1.0], dtype=float)

        v = u[:, None, :] + a * (sx[None, :, None] * e1[:, None, :] + sy[None, :, None] * e2[:, None, :])
        vn = np.linalg.norm(v, axis=2)
        vn = np.where(vn > 0, vn, 1.0)
        v = v / vn[:, :, None]
        v = v * rr[:, None, None]
        v = v.reshape(-1, 3)

        m = int(rr.size)
        base = 4 * np.arange(m, dtype=int)
        i = np.concatenate([base + 0, base + 0])
        j = np.concatenate([base + 1, base + 2])
        k = np.concatenate([base + 2, base + 3])

        return (v[:, 0], v[:, 1], v[:, 2], i, j, k, ok_idx)

    for k, vname in enumerate(plot_vars):
        row = (k // ncols_vars) + 1
        col = (k % ncols_vars) + 1

        # Ecliptic context: circles in the x–y plane (z=0) with labeled radii.
        if show_ecliptic_circles:
            theta = np.linspace(0.0, 2.0 * np.pi, 361)
            for rr in radii_au:
                rr_plot = float(_r_au_to_plot(rr))
                cx = rr_plot * np.cos(theta)
                cy = rr_plot * np.sin(theta)
                cz = np.zeros_like(theta)
                fig.add_trace(
                    go.Scatter3d(
                        x=cx, y=cy, z=cz,
                        mode="lines",
                        line=dict(color=str(ecliptic_circle_rgba), width=int(ecliptic_circle_width)),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                # small label at +x for each ring
                rr_rsun = rr / float(r_sun_au)
                lab = f"{rr_rsun:.0f} R⊙"

                # Put distance labels on the "back side" of the view to reduce clutter:
                # choose the point opposite to the camera eye vector in the x–y plane.
                try:
                    _eye = (_cam(camera) or {}).get("eye", {}) or {}
                    _ex = float(_eye.get("x", 1.0))
                    _ey = float(_eye.get("y", 1.0))
                except Exception:
                    _ex, _ey = 1.0, 1.0

                if (not np.isfinite(_ex)) or (not np.isfinite(_ey)) or (abs(_ex) + abs(_ey) < 1e-12):
                    _ang = float(np.pi)
                else:
                    _ang = float(np.arctan2(_ey, _ex) + np.pi)

                _xl = 1.02 * rr_plot * np.cos(_ang)
                _yl = 1.02 * rr_plot * np.sin(_ang)

                fig.add_trace(
                    go.Scatter3d(
                        x=[_xl], y=[_yl], z=[0.0],
                        mode="text",
                        text=[lab],
                        textposition="middle center",
                        textfont=dict(size=11, color="rgba(0,0,0,0.55)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        # Optional: ecliptic axes (x and y) for orientation
        if show_ecliptic_axes:
            axy = 0.92 * float(lim)
            fig.add_trace(
                go.Scatter3d(
                    x=[-axy, axy], y=[0.0, 0.0], z=[0.0, 0.0],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.18)", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, 0.0], y=[-axy, axy], z=[0.0, 0.0],
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.18)", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # spheres (optionally replace one sphere with a PFSS Br texture surface)
        # Sun sphere
        if (pfss_surface is not None) and (bool(pfss_show_in_all_panels) or (k == 0)) and (str(pfss_surface.get("on", "")) == "photosphere"):
            fig.add_trace(
                go.Surface(
                    x=pfss_surface["X"],
                    y=pfss_surface["Y"],
                    z=pfss_surface["Z"],
                    surfacecolor=pfss_surface["BR"],
                    cmin=float(pfss_surface["cmin"]),
                    cmax=float(pfss_surface["cmax"]),
                    colorscale=pfss_surface["colorscale"],
                    opacity=float(pfss_surface["opacity"]),
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            # PFSS colorbar: draw as a horizontal strip (avoid a large global colorbar on the right).
            if bool(pfss_show_colorbar) and (k == 0):
                _pf_cs = pfss_surface.get('colorscale', []) if (pfss_surface is not None) else []
                if isinstance(_pf_cs, str):
                    try:
                        import plotly.colors as _pc  # noqa: WPS433
                        _pf_cs = _pc.get_colorscale(_pf_cs)
                    except Exception:
                        _pf_cs = [[0.0, 'rgb(0,0,255)'], [1.0, 'rgb(255,0,0)']]
                try:
                    _add_horizontal_colorbar(
                        row=row,
                        col=col,
                        vmin=float(pfss_surface['cmin']),
                        vmax=float(pfss_surface['cmax']),
                        colorscale=list(_pf_cs),
                        title='PFSS Br',
                        slot=0,
                        nslots=2,
                    )
                except Exception:
                    pass
        else:
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

        # Source-surface sphere
        if (pfss_surface is not None) and (bool(pfss_show_in_all_panels) or (k == 0)) and (str(pfss_surface.get("on", "")) == "source_surface"):
            fig.add_trace(
                go.Surface(
                    x=pfss_surface["X"],
                    y=pfss_surface["Y"],
                    z=pfss_surface["Z"],
                    surfacecolor=pfss_surface["BR"],
                    cmin=float(pfss_surface["cmin"]),
                    cmax=float(pfss_surface["cmax"]),
                    colorscale=pfss_surface["colorscale"],
                    opacity=float(pfss_surface["opacity"]),
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            # PFSS colorbar: draw as a horizontal strip (avoid a large global colorbar on the right).
            if bool(pfss_show_colorbar) and (k == 0):
                _pf_cs = pfss_surface.get('colorscale', []) if (pfss_surface is not None) else []
                if isinstance(_pf_cs, str):
                    try:
                        import plotly.colors as _pc  # noqa: WPS433
                        _pf_cs = _pc.get_colorscale(_pf_cs)
                    except Exception:
                        _pf_cs = [[0.0, 'rgb(0,0,255)'], [1.0, 'rgb(255,0,0)']]
                try:
                    _add_horizontal_colorbar(
                        row=row,
                        col=col,
                        vmin=float(pfss_surface['cmin']),
                        vmax=float(pfss_surface['cmax']),
                        colorscale=list(_pf_cs),
                        title='PFSS Br',
                        slot=0,
                        nslots=2,
                    )
                except Exception:
                    pass
            # Neutral line as an HCS proxy (source surface only)
            if (pfss_neutral_xyz is not None) and (k == 0 or bool(pfss_show_in_all_panels)):
                xnl, ynl, znl = pfss_neutral_xyz
                fig.add_trace(
                    go.Scatter3d(
                        x=xnl, y=ynl, z=znl,
                        mode="lines",
                        line=dict(color=str(pfss_neutral_rgba), width=int(pfss_neutral_width)),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
        else:
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


        ex, ey, ez = _equator_ring(float(r_sun_plot))
        fig.add_trace(
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2, color="rgba(0,0,0,0.30)"), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )
        ex, ey, ez = _equator_ring(float(r_ss_plot))
        fig.add_trace(
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2, color="rgba(0,0,0,0.24)", dash="dot"), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )
        mx, my, mz = _prime_meridian(float(r_ss_plot))
        fig.add_trace(
            go.Scatter3d(x=mx, y=my, z=mz, mode="lines", line=dict(width=2, color="rgba(0,0,0,0.28)", dash="dash"), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )

        # spacecraft trajectory (use independent track if provided)
        x_orb = xtrk if (have_track and (xtrk is not None) and (xtrk_raw is not None)) else xsc
        y_orb = ytrk if (have_track and (ytrk is not None) and (ytrk_raw is not None)) else ysc
        z_orb = ztrk if (have_track and (ztrk is not None) and (ztrk_raw is not None)) else zsc
        x_orb_raw = xtrk_raw if (have_track and (xtrk_raw is not None)) else xsc_raw
        y_orb_raw = ytrk_raw if (have_track and (ytrk_raw is not None)) else ysc_raw
        z_orb_raw = ztrk_raw if (have_track and (ztrk_raw is not None)) else zsc_raw

        if (x_orb is not None) and (x_orb_raw is not None):
            fig.add_trace(
                go.Scatter3d(
                    x=x_orb, y=y_orb, z=z_orb,
                    customdata=np.column_stack([x_orb_raw, y_orb_raw, z_orb_raw, np.sqrt(x_orb_raw**2 + y_orb_raw**2 + z_orb_raw**2)/float(r_sun_au)]),
                    mode="lines",
                    line=dict(width=5, color="rgba(0,0,0,0.45)"),
                    opacity=0.55,
                    showlegend=False,
                    hovertemplate=sc_hover_label + "<br>true x=%{customdata[0]:.3f} AU<br>true y=%{customdata[1]:.3f} AU<br>true z=%{customdata[2]:.3f} AU<br>r=%{customdata[3]:.2f} R⊙<extra></extra>",
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

        # ------------------------------------------------------------------
        # Geometry cue: draw the *continuous* mapped track as a faint line.
        # Plotly square markers are billboarded (camera-facing), which can make
        # a curved path on a sphere look visually "flat". A low-opacity line
        # anchored on the source-surface radius restores the curvature cue.
        # Draw the line *before* markers so markers stay on top.
        # ------------------------------------------------------------------
        try:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(width=3, color="rgba(0,0,0,0.22)"),
                    opacity=0.55,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        except Exception:
            pass

        # Marker size: adapt to point count for legibility.
        npts = int(xs.size) if hasattr(xs, "size") else len(xs)
        msize = 6 if npts <= 1500 else (5 if npts <= 4000 else 4)

        if vname == "polarity":
            if "polarity" in data.columns:
                pol_ser = data["polarity"]
            elif "Br" in data.columns:
                pol_ser = np.sign(pd.to_numeric(data["Br"], errors="coerce"))
            else:
                pol_ser = pd.Series(np.nan, index=data.index)

            pol = pd.to_numeric(pol_ser, errors="coerce").to_numpy(dtype=float)[::decimate]
            colmap = np.where(pol > 0, "#d62728", np.where(pol < 0, "#1f77b4", "#7f7f7f"))

            # Render as a sphere-attached patch mesh (fixes billboard "flat ring" artifact).
            try:
                size_deg = float(max(0.8, min(4.0, float(msize) * 0.28)))
                xv, yv, zv, ii, jj, kk, ok_idx = _sphere_patches_mesh_from_xyz(xs, ys, zs, size_deg=size_deg)
                if ok_idx.size > 0 and ok_idx.size <= 14000:
                    vc = np.repeat(colmap[ok_idx], 4).tolist()
                    cd = np.repeat(custom_base[ok_idx], 4, axis=0)
                    pol_rep = np.repeat(pol[ok_idx], 4)
                    cd = np.column_stack([cd, pol_rep])
                    fig.add_trace(
                        go.Mesh3d(
                            x=xv, y=yv, z=zv,
                            i=ii, j=jj, k=kk,
                            vertexcolor=vc,
                            opacity=0.94,
                            flatshading=True,
                            customdata=cd,
                            hovertemplate=base_hover + "polarity=%{customdata[6]:.0f}<extra></extra>",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    # Fallback: too many points for patches; keep markers but reduce billboard artifact.
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            customdata=np.column_stack([custom_base, pol]),
                            mode="markers",
                            marker=dict(size=max(2, msize - 2), symbol="circle", color=colmap, opacity=0.75, line=dict(color="rgba(0,0,0,0.25)", width=0.8)),
                            showlegend=False,
                            hovertemplate=base_hover + "polarity=%{customdata[6]:.0f}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
            except Exception:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        customdata=np.column_stack([custom_base, pol]),
                        mode="markers",
                        marker=dict(size=msize, symbol="circle", color=colmap, opacity=0.75),
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
                # For log-scaled panels, we keep vmin/vmax *physically meaningful* in the variable's
                # native units (spec['vmin'/'vmax'] if provided), then map the colorscale into log10-space.
                good = np.isfinite(vplot) & (vplot > 0)
                vplot[~good] = np.nan

                if (spec.get('vmin', None) is not None) and (spec.get('vmax', None) is not None):
                    vmin_raw = float(spec['vmin'])
                    vmax_raw = float(spec['vmax'])
                    if (not np.isfinite(vmin_raw)) or (not np.isfinite(vmax_raw)) or (vmin_raw <= 0) or (vmax_raw <= 0) or (vmax_raw <= vmin_raw):
                        # Fall back to data-driven limits if the user-provided limits are invalid for log scale.
                        vmin_raw, vmax_raw = _compute_scalar_limits(vv, spec=spec, percentiles=percentiles)
                    vmin = float(np.log10(vmin_raw))
                    vmax = float(np.log10(vmax_raw))
                else:
                    vmin_raw, vmax_raw = _compute_scalar_limits(vv, spec=spec, percentiles=percentiles)
                    vmin = float(np.log10(vmin_raw))
                    vmax = float(np.log10(vmax_raw))

                vplot = np.log10(vplot)
                cb_title = _log10_label(cb_title)

            marker = dict(
                size=msize,
                symbol="square",
                line=dict(color="rgba(0,0,0,0.35)", width=0.9),
                color=vplot,
                colorscale=colorscale,
                cmin=float(vmin),
                cmax=float(vmax),
                opacity=0.94,
                showscale=False,
            )
            # Optional instantaneous-value marker on the horizontal colorbar strip (MP4 export).
            mv_raw = None
            mv_plot = None
            mv_lab = ''
            if cb_marker_values is not None and (vname in cb_marker_values):
                try:
                    mv_raw = float(cb_marker_values.get(vname))
                except Exception:
                    mv_raw = None
            if mv_raw is not None and np.isfinite(float(mv_raw)):
                if scale == 'log':
                    if float(mv_raw) > 0:
                        mv_plot = float(np.log10(float(mv_raw)))
                        mv_lab = f'{float(mv_raw):.4g}'
                else:
                    mv_plot = float(mv_raw)
                    mv_lab = f'{float(mv_raw):.4g}'

            _add_horizontal_colorbar(
                row=row,
                col=col,
                vmin=float(vmin),
                vmax=float(vmax),
                colorscale=colorscale,
                title=cb_title,
                marker_value=mv_plot,
                marker_label=mv_lab,
                slot=(1 if (pfss_surface is not None and bool(pfss_show_colorbar) and (k == 0)) else 0),
                nslots=(2 if (pfss_surface is not None and bool(pfss_show_colorbar) and (k == 0)) else 1),
            )

            try:
                size_deg = float(max(0.8, min(4.0, float(msize) * 0.28)))
                xv, yv, zv, ii, jj, kk, ok_idx = _sphere_patches_mesh_from_xyz(xs, ys, zs, size_deg=size_deg)
                if ok_idx.size > 0 and ok_idx.size <= 14000:
                    vplot_ok = np.asarray(vplot, float)[ok_idx]
                    intens = np.repeat(vplot_ok, 4)
                    cd = np.repeat(custom_base[ok_idx], 4, axis=0)
                    vv_rep = np.repeat(vv[ok_idx], 4)
                    cd = np.column_stack([cd, vv_rep])
                    fig.add_trace(
                        go.Mesh3d(
                            x=xv, y=yv, z=zv,
                            i=ii, j=jj, k=kk,
                            intensity=intens,
                            colorscale=colorscale,
                            cmin=float(vmin),
                            cmax=float(vmax),
                            opacity=0.94,
                            flatshading=True,
                            showscale=False,
                            customdata=cd,
                            hovertemplate=base_hover + f"{vname}=%{{customdata[6]:.4g}}<extra></extra>",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    # Fallback: too many points for patches; reduce billboard artifact.
                    marker2 = marker.copy()
                    marker2["symbol"] = "circle"
                    marker2["size"] = max(2, int(msize) - 2)
                    marker2["opacity"] = 0.75
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            customdata=np.column_stack([custom_base, vv]),
                            mode="markers",
                            marker=marker2,
                            showlegend=False,
                            hovertemplate=base_hover + f"{vname}=%{{customdata[6]:.4g}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
            except Exception:
                marker2 = marker.copy()
                marker2["symbol"] = "circle"
                marker2["opacity"] = 0.75
                fig.add_trace(
                    go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        customdata=np.column_stack([custom_base, vv]),
                        mode="markers",
                        marker=marker2,
                        showlegend=False,
                        hovertemplate=base_hover + f"{vname}=%{{customdata[6]:.4g}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

        # Highlight the most recent cadence sample (movie-friendly).
        if bool(highlight_last_point):
            try:
                # Use the *last row of the input DataFrame* (not the decimated subset) so the
                # highlight always tracks the current time even when decimate>1.
                last = data.iloc[-1]
                xlp = float(pd.to_numeric(last.get('ss_x_au', np.nan), errors='coerce'))
                ylp = float(pd.to_numeric(last.get('ss_y_au', np.nan), errors='coerce'))
                zlp = float(pd.to_numeric(last.get('ss_z_au', np.nan), errors='coerce'))
                xsc_lp = float(pd.to_numeric(last.get('sc_x_au', np.nan), errors='coerce')) if have_sc else float('nan')
                ysc_lp = float(pd.to_numeric(last.get('sc_y_au', np.nan), errors='coerce')) if have_sc else float('nan')
                zsc_lp = float(pd.to_numeric(last.get('sc_z_au', np.nan), errors='coerce')) if have_sc else float('nan')

                xlp2, ylp2, zlp2 = _xyz_to_plot(np.array([xlp]), np.array([ylp]), np.array([zlp]))
                # Project onto plotted SS sphere (matches main point cloud).
                rr = float(np.sqrt(float(xlp2[0])**2 + float(ylp2[0])**2 + float(zlp2[0])**2))
                if np.isfinite(rr) and rr > 0 and np.isfinite(float(r_ss_plot)):
                    xlp2[0] = xlp2[0] / rr * float(r_ss_plot)
                    ylp2[0] = ylp2[0] / rr * float(r_ss_plot)
                    zlp2[0] = zlp2[0] / rr * float(r_ss_plot)

                # Highlight as a sphere-attached patch (avoids billboard square artifact).
                try:
                    size_deg_hl = float(max(1.2, min(7.0, float(highlight_size) * 0.24)))
                    xv, yv, zv, ii, jj, kk, ok_idx = _sphere_patches_mesh_from_xyz(xlp2, ylp2, zlp2, size_deg=size_deg_hl)
                    if ok_idx.size:
                        fig.add_trace(
                            go.Mesh3d(
                                x=xv, y=yv, z=zv,
                                i=ii, j=jj, k=kk,
                                vertexcolor=[str(highlight_fill_rgba)] * 4,
                                opacity=1.0,
                                flatshading=True,
                                hoverinfo='skip',
                                showlegend=False,
                            ),
                            row=row,
                            col=col,
                        )
                        # Outline (closed loop through the 4 vertices)
                        if int(highlight_edge_width) > 0:
                            xw = [float(xv[0]), float(xv[1]), float(xv[2]), float(xv[3]), float(xv[0])]
                            yw = [float(yv[0]), float(yv[1]), float(yv[2]), float(yv[3]), float(yv[0])]
                            zw = [float(zv[0]), float(zv[1]), float(zv[2]), float(zv[3]), float(zv[0])]
                            fig.add_trace(
                                go.Scatter3d(
                                    x=xw, y=yw, z=zw,
                                    mode='lines',
                                    line=dict(color=str(highlight_edge_rgba), width=int(highlight_edge_width)),
                                    opacity=1.0,
                                    showlegend=False,
                                    hoverinfo='skip',
                                ),
                                row=row,
                                col=col,
                            )
                    else:
                        raise RuntimeError("highlight patch build failed")
                except Exception:
                    fig.add_trace(
                        go.Scatter3d(
                            x=xlp2, y=ylp2, z=zlp2,
                            mode='markers',
                            marker=dict(
                                symbol='circle',
                                size=int(max(3, int(highlight_size) - 2)),
                                color=str(highlight_fill_rgba),
                                opacity=0.95,
                            ),
                            showlegend=False,
                            hoverinfo='skip',
                        ),
                        row=row,
                        col=col,
                    )

                if bool(highlight_connector) and have_sc and np.isfinite(xsc_lp) and np.isfinite(ysc_lp) and np.isfinite(zsc_lp):
                    xsc2, ysc2, zsc2 = _xyz_to_plot(np.array([xsc_lp]), np.array([ysc_lp]), np.array([zsc_lp]))
                    fig.add_trace(
                        go.Scatter3d(
                            x=[float(xsc2[0]), float(xlp2[0])],
                            y=[float(ysc2[0]), float(ylp2[0])],
                            z=[float(zsc2[0]), float(zlp2[0])],
                            mode='lines',
                            line=dict(color=str(highlight_connector_rgba), width=int(highlight_connector_width)),
                            opacity=0.85,
                            showlegend=False,
                            hoverinfo='skip',
                        ),
                        row=row,
                        col=col,
                    )
            except Exception:
                pass

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
    if bool(write_html):
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


def plot_carrington_diagnostics(
    *,
    data: pd.DataFrame,
    ephem_orbit: pd.DataFrame,
    omega: "u.Quantity",
    phi_sign: int,
    out_png: Union[str, Path],
    title: str = "",
    show: bool = False,
) -> Tuple[Path, "plt.Figure"]:
    """Diagnostic plot to prevent Carrington-longitude misinterpretations.

    Panels
    ------
    (1) Unwrapped Carrington longitudes vs time (spacecraft and mapped source).
        In a Sun-fixed rotating longitude, a spacecraft that is not rigidly
        co-rotating with the Sun generally drifts and can sweep ~360° on
        week-scale times (order a Carrington rotation), depending on its orbit.

    (2) Mapping shift sanity: Δφ = (φ_src − φ_sc) in (−180,180] compared to
        the expected φ_sign*Ω*τ (wrapped to (−180,180]).

    This figure is intended as a physics audit aid, not a publication figure.
    """
    import astropy.units as u

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a non-empty DataFrame")
    if not isinstance(ephem_orbit, pd.DataFrame) or ephem_orbit.empty:
        raise ValueError("ephem_orbit must be a non-empty DataFrame")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # --- helper: unwrap ---
    def _unwrap_deg(phi: np.ndarray) -> np.ndarray:
        return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(phi, dtype=float))))

    # --- drift fits ---
    def _rate_deg_day(t: pd.DatetimeIndex, phi_unwrapped: np.ndarray) -> float:
        tt = pd.to_datetime(pd.DatetimeIndex(t), utc=True)
        ts = tt.view("int64").astype(float) / 1e9
        m = np.isfinite(phi_unwrapped) & np.isfinite(ts)
        if int(np.sum(m)) < 3:
            return float("nan")
        b, a = np.polyfit(ts[m], np.asarray(phi_unwrapped, float)[m], 1)
        return float(b * 86400.0)

    # Ephemeris (orbit; pre-gap by construction upstream)
    t_orb = pd.to_datetime(pd.DatetimeIndex(ephem_orbit.index), utc=True)
    if "phi_sc_deg" not in ephem_orbit.columns:
        raise KeyError("ephem_orbit missing phi_sc_deg")
    phi_sc_u = _unwrap_deg(ephem_orbit["phi_sc_deg"].to_numpy(dtype=float))
    rate_sc = _rate_deg_day(t_orb, phi_sc_u)

    # Data cadence series (mapped)
    t_dat = pd.to_datetime(pd.DatetimeIndex(data.index), utc=True)
    phi_src = None
    phi_src_u = None
    rate_src = float("nan")
    if "phi_src" in data.columns:
        phi_src = data["phi_src"].to_numpy(dtype=float)
        phi_src_u = _unwrap_deg(phi_src)
        rate_src = _rate_deg_day(t_dat, phi_src_u)

    # Mapping shift sanity
    shift_block = None
    if {"phi_sc", "phi_src", "tau_s"}.issubset(data.columns):
        phi_sc_dat = data["phi_sc"].to_numpy(dtype=float)
        phi_src_dat = data["phi_src"].to_numpy(dtype=float)
        tau_s = data["tau_s"].to_numpy(dtype=float)
        m = np.isfinite(phi_sc_dat) & np.isfinite(phi_src_dat) & np.isfinite(tau_s) & (tau_s > 0)
        if int(np.sum(m)) >= 5:
            dphi = delta_deg(phi_src_dat[m], phi_sc_dat[m])  # (-180,180]
            omega_deg_s = float(u.Quantity(omega).to_value(u.deg / u.s))
            if "delta_phi_signed" in data.columns:
                expected = data["delta_phi_signed"].to_numpy(dtype=float)[m]
            else:
                expected = float(phi_sign) * omega_deg_s * tau_s[m]
            expected = ((expected + 180.0) % 360.0) - 180.0
            err = dphi - expected
            err = ((err + 180.0) % 360.0) - 180.0
            shift_block = dict(
                t=t_dat[m],
                dphi=dphi,
                expected=expected,
                err=err,
                err_med=float(np.nanmedian(err)),
                err_p16=float(np.nanpercentile(err, 16.0)),
                err_p84=float(np.nanpercentile(err, 84.0)),
            )

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 6.0), sharex=True)
    ax0, ax1 = axs

    ax0.plot(t_orb, phi_sc_u, lw=1.2, label=r"$\phi_{\rm sc}$ (Carrington, unwrapped)")
    if phi_src_u is not None:
        ax0.plot(t_dat, phi_src_u, lw=1.2, alpha=0.85, label=r"$\phi_{\rm src}$ (unwrapped)")

    ax0.set_ylabel("Longitude (deg; unwrapped)")
    if title:
        ax0.set_title(str(title))

    # Annotation block
    lines = []
    if np.isfinite(rate_sc):
        w = float("inf") if abs(rate_sc) < 1e-9 else 360.0 / abs(rate_sc)
        lines.append(f"sc drift: {rate_sc:+.3f} deg/day  (360°/{w:.1f} d)")
    if np.isfinite(rate_src):
        w = float("inf") if abs(rate_src) < 1e-9 else 360.0 / abs(rate_src)
        lines.append(f"src drift: {rate_src:+.3f} deg/day  (360°/{w:.1f} d)")
    if lines:
        ax0.text(
            0.01, 0.02,
            "\n".join(lines),
            transform=ax0.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.75"),
        )
    ax0.legend(loc="upper left", fontsize=9, frameon=False)

    if shift_block is not None:
        ax1.plot(shift_block["t"], shift_block["dphi"], lw=1.0, label=r"$\Delta\phi = \phi_{\rm src}-\phi_{\rm sc}$")
        ax1.plot(shift_block["t"], shift_block["expected"], lw=1.0, alpha=0.8, label=r"expected shift")
        ax1.plot(shift_block["t"], shift_block["err"], lw=0.8, alpha=0.6, label="error")
        ax1.set_ylabel("Angle (deg; wrapped to ±180)")
        ax1.legend(loc="upper left", fontsize=9, frameon=False)
        ax1.text(
            0.01, 0.02,
            f"err median={shift_block['err_med']:+.2f}°  (p16={shift_block['err_p16']:+.2f}°, p84={shift_block['err_p84']:+.2f}°)",
            transform=ax1.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.75"),
        )
    else:
        ax1.text(
            0.5, 0.5,
            "mapping-shift check unavailable (need phi_sc, phi_src, tau_s)",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax1.set_ylabel("Angle (deg)")
    ax1.set_xlabel("UTC")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_png, fig


def _apply_cube_aspect(fig: go.Figure, *, sun_zoom_au: float, n_scenes: int) -> None:
    """Force equal aspect so spheres render as spheres (avoid 'pancake' Sun)."""
    try:
        zoom = float(sun_zoom_au)
    except Exception:
        zoom = None  # type: ignore
    if zoom is None or not np.isfinite(zoom) or zoom <= 0:
        return
    rng = [-zoom, zoom]
    for k in range(1, int(max(1, n_scenes)) + 1):
        key = "scene" if k == 1 else f"scene{k}"
        if not hasattr(fig.layout, key):
            continue
        try:
            scene = getattr(fig.layout, key)
            scene.update(
                aspectmode="cube",
                xaxis=dict(range=rng),
                yaxis=dict(range=rng),
                zaxis=dict(range=rng),
            )
        except Exception:
            continue


