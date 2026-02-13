from __future__ import annotations

import argparse
from pathlib import Path

import plotly.express as px
from plotly.subplots import make_subplots

import helpers
from horizons_sun_lonlat import get_repo_style_orbit_df



import argparse
from pathlib import Path

import plotly.express as px
from plotly.subplots import make_subplots

import helpers
from horizons_sun_lonlat import get_repo_style_orbit_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--step", default="6h")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--outdir", default="out_orbits")
    p.add_argument("--html", default="out_orbits/interactive_orbits.html")
    p.add_argument("--rss-rsun", type=float, default=2.5)

    # NEW: figure dimensions
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=900)

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04)

    colors = px.colors.qualitative.Plotly
    for i, tgt in enumerate(args.targets):
        res = get_repo_style_orbit_df(
            tgt,
            args.start,
            args.stop,
            args.step,
            rss_rsun=args.rss_rsun,
        )
        df = res.df
        df.to_csv(outdir / f"{tgt.replace(' ', '_')}_repo_orbit.csv")
        fig = helpers.add_sc(fig, df, tgt, colors[i % len(colors)])

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="Year", step="year", stepmode="backward"),
                    dict(count=1, label="Month", step="month", stepmode="backward"),
                    dict(count=7, label="Week", step="day", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        )
    )

    fig.update_yaxes(title_text="Radius [AU]", row=1, col=1)
    fig.update_yaxes(title_text="Latitude [deg]", row=2, col=1)
    fig.update_yaxes(title_text="Carrington lon [deg]", range=[0, 360], tick0=0, dtick=90, row=3, col=1)
    fig.update_yaxes(title_text="Mapped lon [deg]", range=[0, 360], tick0=0, dtick=90, row=4, col=1)

    fig.update_layout(
        title_text="Interactive Orbits (Carrington + ballistic mapping)",
        hovermode="x",
        width=args.width,
        height=args.height,
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(orientation="v"),
    )

    fig.write_html(args.html, auto_open=False)
    print(f"Wrote: {args.html}")
    print(f"Wrote CSVs in: {outdir}")


if __name__ == "__main__":
    main()


