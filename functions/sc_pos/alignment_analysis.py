from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import astropy.units as u
import numpy as np
import pandas as pd

ASSUMPTIONS = [
    "Solar-wind advection direction in GSE is approximately anti-sunward along -X_GSE.",
    "Spacecraft sample the same plasma stream when pairwise cross-flow separation is small.",
    "Along-flow separation is converted to advection lag with a representative bulk speed Vsw.",
    "Orbit windows are treated as fixed-length, non-overlapping intervals for ranking.",
    "Only timestamps where all requested spacecraft are available are scored.",
]


@dataclass
class AlignmentResult:
    ranking: pd.DataFrame
    interval_data: dict[pd.Interval, dict[str, pd.DataFrame]]
    metadata: dict


def print_alignment_assumptions(vsw_kms: float, interval_hours: float) -> None:
    """Print physically-motivated assumptions used by the alignment metric."""
    print("\n[Alignment assumptions]")
    print(f"- Representative advection speed Vsw = {vsw_kms:.1f} km/s")
    print(f"- Window duration = {interval_hours:.2f} hours")
    for item in ASSUMPTIONS:
        print(f"- {item}")


def _fetch_gse_xyz(target: str, start: str, stop: str, step: str) -> pd.DataFrame:
    from sunpy.coordinates import get_horizons_coord
    from sunpy.coordinates.frames import GeocentricSolarEcliptic

    coord = get_horizons_coord(target, {"start": start, "stop": stop, "step": step})
    gse = coord.transform_to(GeocentricSolarEcliptic(obstime=coord.obstime))

    out = pd.DataFrame(
        {
            "x_km": gse.cartesian.x.to_value(u.km),
            "y_km": gse.cartesian.y.to_value(u.km),
            "z_km": gse.cartesian.z.to_value(u.km),
        },
        index=pd.to_datetime(gse.obstime.datetime64),
    ).sort_index()
    out.index.name = "time_utc"
    return out


def _flow_unit_vector_gse() -> np.ndarray:
    # Anti-sunward direction in GSE.
    return np.array([-1.0, 0.0, 0.0])


def _interval_edges(index: pd.DatetimeIndex, duration: pd.Timedelta) -> pd.IntervalIndex:
    start = index.min().floor("s")
    stop = index.max().ceil("s")
    if stop <= start:
        stop = start + duration
    edges = pd.date_range(start, stop + duration, freq=duration)
    return pd.IntervalIndex.from_breaks(edges, closed="left")


def _pairwise_metrics_for_snapshot(points_km: np.ndarray, vsw_kms: float) -> tuple[float, float, float]:
    e = _flow_unit_vector_gse()
    perp_vals = []
    lag_vals_hr = []
    sep_vals_re = []

    for i, j in combinations(range(points_km.shape[0]), 2):
        d = points_km[i] - points_km[j]
        d_par = abs(float(np.dot(d, e)))
        d_perp = float(np.linalg.norm(d - np.dot(d, e) * e))
        d_tot = float(np.linalg.norm(d))

        perp_vals.append(d_perp)
        lag_vals_hr.append((d_par / vsw_kms) / 3600.0)
        sep_vals_re.append(d_tot / 6378.137)

    return (
        float(np.sqrt(np.mean(np.square(perp_vals))) / 6378.137),
        float(np.sqrt(np.mean(np.square(lag_vals_hr)))),
        float(np.max(sep_vals_re)),
    )


def rank_aligned_intervals(
    targets: Iterable[str],
    start: str,
    stop: str,
    *,
    step: str = "30m",
    interval_hours: float = 6.0,
    top_n: int = 3,
    vsw_kms: float = 400.0,
    w_perp: float = 0.75,
    w_lag: float = 0.25,
    perp_ref_re: float = 40.0,
    lag_ref_hr: float = 1.0,
    verbose: bool = True,
) -> AlignmentResult:
    """
    Rank fixed-duration windows by a physically motivated co-streaming metric.

    score = w_perp*(RMS_perp / perp_ref) + w_lag*(RMS_lag / lag_ref)

    Lower score => better alignment for sampling the same solar wind stream.
    """
    targets = list(targets)
    if len(targets) < 2:
        raise ValueError("Provide at least 2 spacecraft.")

    if verbose:
        print_alignment_assumptions(vsw_kms=vsw_kms, interval_hours=interval_hours)

    sc_data = {t: _fetch_gse_xyz(t, start, stop, step) for t in targets}

    common_index = None
    for df in sc_data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common timestamps across spacecraft in the requested range.")

    duration = pd.to_timedelta(interval_hours, unit="h")
    bins = _interval_edges(common_index, duration)

    rows = []
    interval_data: dict[pd.Interval, dict[str, pd.DataFrame]] = {}

    for window in bins:
        t0, t1 = window.left, window.right
        mask = (common_index >= t0) & (common_index < t1)
        times = common_index[mask]
        if len(times) < 2:
            continue

        perp_rms, lag_rms, max_sep = [], [], []
        per_sc = {k: v.loc[times].copy() for k, v in sc_data.items()}

        for t in times:
            pts = np.vstack([per_sc[k].loc[t, ["x_km", "y_km", "z_km"]].values for k in targets])
            p, l, m = _pairwise_metrics_for_snapshot(pts, vsw_kms=vsw_kms)
            perp_rms.append(p)
            lag_rms.append(l)
            max_sep.append(m)

        rms_perp_re = float(np.median(perp_rms))
        rms_lag_hr = float(np.median(lag_rms))
        max_sep_re = float(np.median(max_sep))
        score = w_perp * (rms_perp_re / perp_ref_re) + w_lag * (rms_lag_hr / lag_ref_hr)

        rows.append(
            {
                "interval_start": t0,
                "interval_end": t1,
                "n_samples": len(times),
                "rms_perp_re": rms_perp_re,
                "rms_lag_hr": rms_lag_hr,
                "max_pair_sep_re": max_sep_re,
                "score": score,
            }
        )
        interval_data[window] = per_sc

    ranking = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    if top_n > 0:
        ranking = ranking.head(top_n).copy()

    return AlignmentResult(
        ranking=ranking,
        interval_data=interval_data,
        metadata={
            "targets": targets,
            "step": step,
            "interval_hours": interval_hours,
            "vsw_kms": vsw_kms,
            "weights": {"w_perp": w_perp, "w_lag": w_lag},
            "normalizations": {"perp_ref_re": perp_ref_re, "lag_ref_hr": lag_ref_hr},
        },
    )


def plot_ranked_alignment_intervals(result: AlignmentResult, marker_size: int = 4):
    """Create one 3D plot per top-ranked interval."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figs = []
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for _, row in result.ranking.iterrows():
        t0 = pd.Timestamp(row["interval_start"])
        t1 = pd.Timestamp(row["interval_end"])
        key = pd.Interval(t0, t1, closed="left")
        sc_map = result.interval_data.get(key)
        if sc_map is None:
            continue

        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])

        for i, (name, df) in enumerate(sc_map.items()):
            c = palette[i % len(palette)]
            x = df["x_km"].values / 6378.137
            y = df["y_km"].values / 6378.137
            z = df["z_km"].values / 6378.137

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    name=name,
                    line=dict(color=c, width=6),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[x[0], x[-1]],
                    y=[y[0], y[-1]],
                    z=[z[0], z[-1]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=marker_size, color=c, symbol="circle"),
                    hovertemplate="%{x:.1f}, %{y:.1f}, %{z:.1f} Re<extra></extra>",
                )
            )

        fig.update_layout(
            title=(
                f"Rank {int(row['rank'])}: {t0} to {t1} | "
                f"score={row['score']:.3f}, rms_perp={row['rms_perp_re']:.1f} Re, "
                f"rms_lag={row['rms_lag_hr']:.2f} h"
            ),
            scene=dict(
                xaxis_title="X_GSE [Re]",
                yaxis_title="Y_GSE [Re]",
                zaxis_title="Z_GSE [Re]",
                aspectmode="data",
            ),
            template="plotly_white",
            height=700,
            width=950,
        )
        figs.append(fig)

    return figs
