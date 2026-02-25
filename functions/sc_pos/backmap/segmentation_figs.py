from __future__ import annotations

"""sc_pos.backmap.segmentation_figs

Figure helpers for source-segmentation diagnostics.

These figures are intended to make the segmentation step auditable:
  (i) which physical diagnostics were used,
 (ii) whether those diagnostics exhibit state changes on the chosen window,
(iii) where the multivariate change score crosses the threshold, and
 (iv) which times are retained as stable segments.

The segmentation model uses rolling medians and rolling MAD-based constancy
proxies internally; the diagnostic plots emphasize the *physical time series*
(rolling median shown explicitly) together with the final score/segments.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _tex_sanitize(s: str) -> str:
    """Sanitize labels for Matplotlib usetex=True (pdfTeX).

    pdfTeX (LaTeX2e) does not accept many Unicode symbols by default. Replace
    common offenders with TeX macros or ASCII fallbacks.
    """
    if not isinstance(s, str):
        return str(s)
    # inequalities
    s = s.replace("\u2264", r"$\\leq$")
    s = s.replace("\u2265", r"$\\geq$")
    # greek
    s = s.replace("\u03bb", r"$\\lambda$")
    s = s.replace("\u03bc", r"$\\mu$")
    s = s.replace("\u03c3", r"$\\sigma$")
    # dashes / approx
    s = s.replace("\u2013", "-")
    s = s.replace("\u2014", "-")
    s = s.replace("\u2248", "~")
    return s


def _safe_mkdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _as_float_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype(float)


def plot_segmentation_score_timeseries(
    *,
    data: pd.DataFrame,
    plot_vars: Sequence[str],
    threshold: float,
    window: str,
    mode: str,
    metric: str,
    ridge_alpha: float,
    out_png: Path,
    out_pdf: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """Plot the physical diagnostics used + the final segmentation score."""

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    out_png = Path(out_png)
    _safe_mkdir(out_png)

    t = pd.to_datetime(data.index)
    if isinstance(t, pd.DatetimeIndex) and (t.tz is not None):
        t = t.tz_convert(None)

    # score and segments
    score = _as_float_series(data.get("source_score", pd.Series(index=t, dtype=float)))
    seg = pd.to_numeric(data.get("source_segment", pd.Series(-1, index=t)), errors="coerce").fillna(-1).astype(int)
    stable = (seg >= 0) & np.isfinite(score.to_numpy(dtype=float))

    # Determine a sensible min_periods for the rolling median overlay
    try:
        wsec = float(pd.Timedelta(str(window)).total_seconds())
        dt = (t[1:] - t[:-1]).total_seconds().to_numpy(dtype=float)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        dt0 = float(np.nanmedian(dt)) if dt.size else float("nan")
        nwin = int(max(3, round(wsec / dt0))) if (np.isfinite(dt0) and dt0 > 0) else 3
        mp = int(max(3, int(0.5 * nwin)))
    except Exception:
        mp = 3

    # Keep only vars that exist
    vars_used = [str(v) for v in plot_vars if str(v) in data.columns]

    # Layout: one panel per variable + score + segment band
    n_feat = len(vars_used)
    nrows = max(1, n_feat) + 2
    fig_h = max(5.0, 0.95 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14.0, fig_h), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    r = 0
    if n_feat == 0:
        ax = axes[r]
        ax.text(
            0.5,
            0.5,
            "No requested segmentation diagnostics exist in the DataFrame.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel("diagnostic")
        r += 1
    else:
        for v in vars_used:
            ax = axes[r]
            x = pd.to_numeric(data[v], errors="coerce")
            # Overlay rolling median used conceptually by the segmentation
            med = x.rolling(str(window), center=True, min_periods=int(mp)).median()

            ax.plot(t, x.to_numpy(dtype=float), lw=0.8, alpha=0.35)
            ax.plot(t, med.to_numpy(dtype=float), lw=1.6)
            ax.set_ylabel(str(v))
            ax.grid(True, alpha=0.25)
            r += 1

    ax = axes[r]
    ax.plot(t, score.to_numpy(dtype=float), lw=1.7)
    if np.isfinite(float(threshold)):
        ax.axhline(float(threshold), ls="--", lw=1.0)
    if stable.any():
        ax.fill_between(
            t,
            0.0,
            1.0,
            where=stable.to_numpy(dtype=bool),
            transform=ax.get_xaxis_transform(),
            alpha=0.10,
        )
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.25)

    title = f"Source segmentation (window={window}, mode={mode}, metric={metric}, ridge_alpha={float(ridge_alpha):.3g}, thr={float(threshold):.3g})"
    ax.set_title(title)
    r += 1

    ax = axes[r]
    seg_arr = seg.to_numpy(dtype=float)
    seg_arr[seg_arr < 0] = np.nan

    img = np.tile(seg_arr[None, :], (10, 1))
    imgm = np.ma.masked_invalid(img)

    t0 = pd.to_datetime(t[0]).to_pydatetime()
    t1 = pd.to_datetime(t[-1]).to_pydatetime()
    x0 = mdates.date2num(t0)
    x1 = mdates.date2num(t1)

    im = ax.imshow(
        imgm,
        aspect="auto",
        interpolation="nearest",
        extent=(x0, x1, 0.0, 1.0),
        origin="lower",
    )
    ax.set_yticks([])
    ax.set_ylabel("segment")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cb.set_label("segment id")

    ax.set_xlim(t[0], t[-1])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel("time (UTC)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        _safe_mkdir(out_pdf)
        fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def plot_segmentation_footpoints(
    *,
    data: pd.DataFrame,
    out_png: Path,
    out_pdf: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot source-surface footpoints colored by segment id."""

    if not {"phi_src", "lat_src", "source_segment"}.issubset(set(data.columns)):
        return None

    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    _safe_mkdir(out_png)

    phi = pd.to_numeric(data["phi_src"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float)
    seg = pd.to_numeric(data["source_segment"], errors="coerce").fillna(-1).to_numpy(dtype=int)

    ok = np.isfinite(phi) & np.isfinite(lat)
    if not np.any(ok):
        return None

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    stable = ok & (seg >= 0)
    unstable = ok & (seg < 0)

    if np.any(unstable):
        ax.scatter(phi[unstable], lat[unstable], s=8, alpha=0.35, label="unstable", marker=".")
    if np.any(stable):
        sc = ax.scatter(phi[stable], lat[stable], c=seg[stable], s=10, alpha=0.90)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("segment id")

    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel(r"$\phi_{SS}$ [deg]")
    ax.set_ylabel(r"$\lambda_{SS}$ [deg]")
    ax.set_title("Source-surface footpoints colored by stable segment")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        _safe_mkdir(out_pdf)
        fig.savefig(out_pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def plot_segmentation_schematic(*, out_pdf: Path, out_png: Optional[Path] = None) -> Path:
    """Write a static schematic of the segmentation pipeline."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    out_pdf = Path(out_pdf)
    _safe_mkdir(out_pdf)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.axis("off")

    def box(x, y, w, h, text):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, _tex_sanitize(text), ha="center", va="center", fontsize=10)
        return (x, y, w, h)

    b1 = box(0.05, 0.70, 0.25, 0.18, "Cadence data\n(physical diagnostics vs time)")
    b2 = box(0.37, 0.70, 0.25, 0.18, "Feature construction\nrolling median + MAD-based constancy\n(kind-aware scaling)")
    b3 = box(0.69, 0.70, 0.26, 0.18, "Robust standardization\n(median/MAD)\n+ optional GMM")
    b4 = box(0.18, 0.40, 0.30, 0.18, "Two-sided mean shift\nd(i) = mean_R(z) - mean_L(z)")
    b5 = box(0.56, 0.40, 0.30, 0.18, "Ridge Mahalanobis score\n$S = \\|d\\|_{(C+\\lambda I)^{-1}}$\n(score $\\leq$ thr $\\Rightarrow$ stable)")
    b6 = box(0.37, 0.12, 0.25, 0.18, "Stable segments\n(contiguous stable runs)\n(optionally split by regime)")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))

    arrow(0.30, 0.79, 0.37, 0.79)
    arrow(0.62, 0.79, 0.69, 0.79)
    arrow(0.82, 0.70, 0.33, 0.58)
    arrow(0.48, 0.49, 0.56, 0.49)
    arrow(0.71, 0.40, 0.49, 0.30)

    ax.text(
        0.5,
        0.02,
        "Segmentation is a regime-change detector in a multivariate physical feature space.\n"
        "Use it to avoid fitting travel-time parameters across state boundaries.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        out_png = Path(out_png)
        _safe_mkdir(out_png)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
