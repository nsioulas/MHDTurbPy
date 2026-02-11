from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from horizons_sun_lonlat import get_lonlat_xyz_timeseries


def build_figure(dfs: List[pd.DataFrame], names: List[str]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode="markers",
            name="Sun",
            marker=dict(size=5),
            hovertemplate="Sun<extra></extra>",
        )
    )

    for df, name in zip(dfs, names):
        ht = (
            "time=%{customdata[0]}<br>"
            "HGS lon=%{customdata[1]:.2f} deg<br>"
            "HGS lat=%{customdata[2]:.2f} deg<br>"
            "r=%{customdata[3]:.3f} AU<br>"
            "<extra>" + name + "</extra>"
        )
        custom = list(
            zip(
                df.index.astype(str),
                df["hgs_lon_deg"].to_numpy(),
                df["hgs_lat_deg"].to_numpy(),
                df["hgs_r_au"].to_numpy(),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=df["hee_x_au"],
                y=df["hee_y_au"],
                z=df["hee_z_au"],
                mode="lines",
                name=name,
                customdata=custom,
                hovertemplate=ht,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="HEE x [AU]",
            yaxis_title="HEE y [AU]",
            zaxis_title="HEE z [AU]",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Spacecraft orbits (HEE XYZ) with Sun-centered HGS lon/lat in hover",
    )
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot interactive heliocentric orbits; hover shows Sun-centered lon/lat.")
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--step", default="6h")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--outdir", default="out_orbits")
    p.add_argument("--html", default="out_orbits/orbits.html")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for t in args.targets:
        res = get_lonlat_xyz_timeseries(t, args.start, args.stop, args.step, carrington=False)
        dfs.append(res.df)

    fig = build_figure(dfs, args.targets)
    fig.write_html(args.html)
    print(f"Wrote: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
