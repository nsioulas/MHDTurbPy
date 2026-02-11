from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, CartesianRepresentation

from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates.frames import (
    HeliocentricEarthEcliptic,
    HeliocentricInertial,
    HeliographicCarrington,
)

try:
    from . import helpers
    from .horizons_sun_lonlat import get_repo_style_orbit_df, resolve_spacecraft_spkid
except Exception:
    import helpers
    from horizons_sun_lonlat import get_repo_style_orbit_df, resolve_spacecraft_spkid


def _get_xyz_timeseries(target_id: str, start: str, stop: str, step: str, frame: str) -> pd.DataFrame:
    coord0 = get_horizons_coord(target_id, {"start": start, "stop": stop, "step": step})

    if frame.upper() == "HEE":
        fr = HeliocentricEarthEcliptic(obstime=coord0.obstime)
    elif frame.upper() == "HCI":
        fr = HeliocentricInertial(obstime=coord0.obstime)
    else:
        raise ValueError("frame must be 'HEE' or 'HCI'")

    c = coord0.transform_to(fr)
    t = pd.to_datetime(c.obstime.datetime64)
    x = c.cartesian.x.to_value(u.AU)
    y = c.cartesian.y.to_value(u.AU)
    z = c.cartesian.z.to_value(u.AU)

    df = pd.DataFrame({"x_au": x, "y_au": y, "z_au": z}, index=t)
    df.index.name = "time_utc"
    return df


def build_timeseries_figure(targets: list[str], start: str, stop: str, step: str, rss_rsun: float, omega_deg_per_day: float, width: int, height: int):
    colors = px.colors.qualitative.Plotly
    DASH = {"ACE": "solid", "Wind": "dash", "IMAP": "dot", "SOLAR-1": "dashdot", "SWFO-L1": "dashdot"}
    SYM = {"ACE": "circle", "Wind": "square", "IMAP": "diamond", "SOLAR-1": "triangle-up", "SWFO-L1": "triangle-up"}

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045)

    for i, t in enumerate(targets):
        df = get_repo_style_orbit_df(t, start, stop, step, rss_rsun=rss_rsun, omega_deg_per_day=omega_deg_per_day).df
        fig = helpers.add_sc(fig, df, t, colors[i % len(colors)], line_dash=DASH.get(t, "solid"), marker_symbol=SYM.get(t, "circle"), show_by_default=True)

    fig.update_yaxes(title_text="R [AU]", row=1, col=1)
    fig.update_yaxes(title_text="Lat [deg]", row=2, col=1)
    fig.update_yaxes(title_text="Lon [deg]", range=[0, 360], tick0=0, dtick=90, row=3, col=1)
    fig.update_yaxes(title_text="Mapped lon [deg]", range=[0, 360], tick0=0, dtick=90, row=4, col=1)
    fig.update_layout(width=width, height=height, title="Interactive Orbits", hovermode="x unified", legend=dict(groupclick="togglegroup"), margin=dict(l=70, r=40, t=80, b=50))
    return fig


def _plane_patch_in_target_frame(obstime, span_au: float, target_frame: str, n: int = 35):
    grid = np.linspace(-span_au, span_au, n)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)

    hee = HeliocentricEarthEcliptic(obstime=obstime)
    rep = CartesianRepresentation(X.ravel() * u.AU, Y.ravel() * u.AU, Z.ravel() * u.AU)
    c = SkyCoord(rep, frame=hee)
    ct = c if target_frame.upper() == "HEE" else c.transform_to(HeliocentricInertial(obstime=obstime))

    Xt = ct.cartesian.x.to_value(u.AU).reshape(X.shape)
    Yt = ct.cartesian.y.to_value(u.AU).reshape(X.shape)
    Zt = ct.cartesian.z.to_value(u.AU).reshape(X.shape)
    return Xt, Yt, Zt


def _carrington_and_footpoints(spkid: str, start: str, stop: str, step: str, rss_rsun: float, omega_deg_per_day: float, vsw_kms: float):
    coord0 = get_horizons_coord(spkid, {"start": start, "stop": stop, "step": step})
    carr = coord0.transform_to(HeliographicCarrington(obstime=coord0.obstime, observer="earth"))

    Rsun = const.R_sun.to(u.km)
    rss = (rss_rsun * const.R_sun).to(u.km)
    omega = (omega_deg_per_day * u.deg) / (24 * 3600 * u.s)
    vsw = (vsw_kms * u.km / u.s)

    r_sc = carr.spherical.distance.to(u.km)
    lon_sc = carr.lon
    lat_sc = carr.lat
    delta = omega * (r_sc - rss) / vsw
    lon_ss = (lon_sc + delta).wrap_at(360 * u.deg)

    fp_ss = SkyCoord(lon=lon_ss, lat=lat_sc, radius=rss, frame=carr.frame)
    fp_sun = SkyCoord(lon=lon_ss, lat=lat_sc, radius=Rsun, frame=carr.frame)
    return pd.to_datetime(carr.obstime.datetime64), carr, fp_ss, fp_sun


def _to_xyz_in_frame(coord: SkyCoord, frame: str):
    fr = HeliocentricEarthEcliptic(obstime=coord.obstime) if frame.upper() == "HEE" else HeliocentricInertial(obstime=coord.obstime)
    c = coord.transform_to(fr)
    return c.cartesian.x.to_value(u.AU), c.cartesian.y.to_value(u.AU), c.cartesian.z.to_value(u.AU)


def build_3d_figure(targets: list[str], start: str, stop: str, step: str, frame3d: str = "HEE", rss_rsun: float = 2.5, omega_deg_per_day: float = 14.1844, vsw1_kms: float = 300.0, vsw2_kms: float | None = 700.0, width: int = 1800, height: int = 900, sun_zoom_au: float = 0.06, plane_span_au: float = 1.2, show_spokes: bool = True, spoke_count: int = 8, decimate: int = 1, verbose: bool = True):
    frame3d = frame3d.upper()
    if frame3d not in {"HEE", "HCI"}:
        raise ValueError("frame3d must be 'HEE' or 'HCI'")
    if decimate < 1:
        decimate = 1

    vsw_list = [v for v in (vsw1_kms, vsw2_kms) if v is not None]
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], column_widths=[0.66, 0.34], horizontal_spacing=0.02, subplot_titles=("AU-scale view", "Sun-zoom backmapping"))
    fig.update_layout(template="plotly_white", width=width, height=height, margin=dict(l=0, r=0, t=85, b=0), title=dict(text=f"3D orbits + ecliptic plane + ballistic backmapping (frame={frame3d})", x=0.5, xanchor="center"), legend=dict(groupclick="togglegroup"))

    earth_df = _get_xyz_timeseries("399", start, stop, step, frame=frame3d)
    obstime_ref = earth_df.index[len(earth_df) // 2].to_pydatetime()
    Xp, Yp, Zp = _plane_patch_in_target_frame(obstime=obstime_ref, span_au=plane_span_au, target_frame=frame3d)
    fig.add_trace(go.Surface(x=Xp, y=Yp, z=Zp, showscale=False, opacity=0.10, hoverinfo="skip", showlegend=False), row=1, col=1)

    Rsun_au = const.R_sun.to_value(u.AU)
    rss_au = (rss_rsun * const.R_sun).to_value(u.AU)
    theta = np.linspace(0, 2 * np.pi, 260)

    fig.add_trace(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="markers+text", name="Sun", marker=dict(size=6), text=["Sun"], textposition="top center"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=Rsun_au * np.cos(theta), y=Rsun_au * np.sin(theta), z=np.zeros_like(theta), mode="lines", line=dict(width=3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=rss_au * np.cos(theta), y=rss_au * np.sin(theta), z=np.zeros_like(theta), mode="lines", line=dict(width=2, dash="dot"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=earth_df["x_au"][::decimate], y=earth_df["y_au"][::decimate], z=earth_df["z_au"][::decimate], mode="lines", name="Earth (traj)", line=dict(width=3)), row=1, col=1)

    for i, name in enumerate(targets):
        spkid = resolve_spacecraft_spkid(name)
        col = colors[i % len(colors)]
        sc_df = _get_xyz_timeseries(spkid, start, stop, step, frame=frame3d).iloc[::decimate]
        fig.add_trace(go.Scatter3d(x=sc_df["x_au"], y=sc_df["y_au"], z=sc_df["z_au"], mode="lines", name=name, line=dict(width=4, color=col)), row=1, col=1)

        for j, vsw in enumerate(vsw_list):
            _, _, fp_ss, fp_sun = _carrington_and_footpoints(spkid, start, stop, step, rss_rsun, omega_deg_per_day, vsw)
            x_ss, y_ss, z_ss = _to_xyz_in_frame(fp_ss[::decimate], frame3d)
            x_su, y_su, z_su = _to_xyz_in_frame(fp_sun[::decimate], frame3d)
            fig.add_trace(go.Scatter3d(x=x_ss, y=y_ss, z=z_ss, mode="lines", line=dict(width=3, dash="solid" if j == 0 else "dash", color=col), showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter3d(x=x_su, y=y_su, z=z_su, mode="lines", line=dict(width=5, dash="dot" if j == 0 else "dashdot", color=col), showlegend=False), row=1, col=2)

    fig.update_layout(scene=dict(xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]", aspectmode="data"), scene2=dict(xaxis_title="x [AU]", yaxis_title="y [AU]", zaxis_title="z [AU]", aspectmode="cube", xaxis=dict(range=[-sun_zoom_au, sun_zoom_au]), yaxis=dict(range=[-sun_zoom_au, sun_zoom_au]), zaxis=dict(range=[-sun_zoom_au, sun_zoom_au])))
    return fig


def write_combined_html(fig_ts: go.Figure, fig_3d: go.Figure, out_html: str):
    html_ts = pio.to_html(fig_ts, include_plotlyjs="cdn", full_html=False)
    html_3d = pio.to_html(fig_3d, include_plotlyjs=False, full_html=False)
    page = f"""<html><head><meta charset='utf-8'></head><body><div style='max-width:100%;margin:0 auto;'>{html_ts}</div><hr style='margin:40px 0;'><div style='max-width:100%;margin:0 auto;'>{html_3d}</div></body></html>"""
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--step", default="6h")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--outdir", default="out_orbits")
    p.add_argument("--html", default="out_orbits/interactive_orbits_plus3d.html")
    p.add_argument("--rss-rsun", type=float, default=2.5)
    p.add_argument("--omega-deg-per-day", type=float, default=14.1844)
    p.add_argument("--width", type=int, default=1800)
    p.add_argument("--height", type=int, default=1100)
    p.add_argument("--width3d", type=int, default=1800)
    p.add_argument("--height3d", type=int, default=900)
    p.add_argument("--frame3d", choices=["HEE", "HCI"], default="HEE")
    p.add_argument("--vsw1-kms", type=float, default=300.0)
    p.add_argument("--vsw2-kms", type=float, default=700.0)
    p.add_argument("--sun-zoom-au", type=float, default=0.06)
    p.add_argument("--plane-span-au", type=float, default=1.2)
    p.add_argument("--no-spokes", action="store_true")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)

    fig_ts = build_timeseries_figure(args.targets, args.start, args.stop, args.step, args.rss_rsun, args.omega_deg_per_day, args.width, args.height)
    fig_3d = build_3d_figure(args.targets, args.start, args.stop, args.step, frame3d=args.frame3d, rss_rsun=args.rss_rsun, omega_deg_per_day=args.omega_deg_per_day, vsw1_kms=args.vsw1_kms, vsw2_kms=args.vsw2_kms, width=args.width3d, height=args.height3d, sun_zoom_au=args.sun_zoom_au, plane_span_au=args.plane_span_au, show_spokes=(not args.no_spokes))
    write_combined_html(fig_ts, fig_3d, args.html)
    print(f"Wrote: {args.html}")


if __name__ == "__main__":
    main()
