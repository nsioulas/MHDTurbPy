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

from sunpy.coordinates import get_horizons_coord, get_body_heliographic_stonyhurst
from sunpy.coordinates.frames import (
    GeocentricSolarEcliptic,
    HeliocentricEarthEcliptic,
    HeliocentricInertial,
    HeliographicStonyhurst,
    HeliographicCarrington,
)

import helpers
from horizons_sun_lonlat import get_repo_style_orbit_df, resolve_spacecraft_spkid


def _get_xyz_timeseries(target_id: str, start: str, stop: str, step: str, frame: str) -> pd.DataFrame:
    coord0 = get_horizons_coord(target_id, {"start": start, "stop": stop, "step": step})

    if frame.upper() == "HEE":
        fr = HeliocentricEarthEcliptic(obstime=coord0.obstime)
    elif frame.upper() == "HCI":
        fr = HeliocentricInertial(obstime=coord0.obstime)
    elif frame.upper() == "GSE":
        fr = GeocentricSolarEcliptic(obstime=coord0.obstime)
    else:
        raise ValueError("frame must be 'HEE', 'HCI' or 'GSE'")

    c = coord0.transform_to(fr)

    t = pd.to_datetime(c.obstime.datetime64)
    x = c.cartesian.x.to_value(u.AU)
    y = c.cartesian.y.to_value(u.AU)
    z = c.cartesian.z.to_value(u.AU)

    df = pd.DataFrame({"x_au": x, "y_au": y, "z_au": z}, index=t)
    df.index.name = "time_utc"
    return df


def build_timeseries_figure(
    targets: list[str],
    start: str,
    stop: str,
    step: str,
    rss_rsun: float,
    omega_deg_per_day: float,
    width: int,
    height: int,
):
    colors = px.colors.qualitative.Plotly
    DASH = {"ACE": "solid", "Wind": "dash", "IMAP": "dot", "SOLAR-1": "dashdot", "SWFO-L1": "dashdot"}
    SYM = {"ACE": "circle", "Wind": "square", "IMAP": "diamond", "SOLAR-1": "diamond-open", "SWFO-L1": "triangle-up"}

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=[
            "Heliocentric distance (AU)",
            "Heliographic latitude (Carrington; deg)",
            "Heliographic longitude (Carrington; deg) — breaks at 0/360",
            f"Ballistic-mapped Carrington longitude to r_ss={rss_rsun:.2f} R_sun (deg): 300 km/s (solid) and 700 km/s (dotted)",
        ],
    )

    for i, t in enumerate(targets):
        df = get_repo_style_orbit_df(
            t,
            start,
            stop,
            step,
            rss_rsun=rss_rsun,
            omega_deg_per_day=omega_deg_per_day,
        ).df

        fig = helpers.add_sc(
            fig,
            df,
            t,
            colors[i % len(colors)],
            line_dash=DASH.get(t, "solid"),
            marker_symbol=SYM.get(t, "circle"),
            show_by_default=True,
        )

    fig.update_yaxes(title_text="R [AU]", row=1, col=1)
    fig.update_yaxes(title_text="Lat [deg]", row=2, col=1)
    fig.update_yaxes(title_text="Lon [deg]", range=[0, 360], tick0=0, dtick=90, row=3, col=1)
    fig.update_yaxes(title_text="Mapped lon [deg]", range=[0, 360], tick0=0, dtick=90, row=4, col=1)

    fig.update_layout(
        width=width,
        height=height,
        title="Interactive Orbits",
        hovermode="x unified",
        legend=dict(groupclick="togglegroup"),
        margin=dict(l=70, r=40, t=80, b=50),
    )
    return fig


def _plane_patch_in_target_frame(
    obstime,
    span_au: float,
    target_frame: str,
    n: int = 35,
):
    grid = np.linspace(-span_au, span_au, n)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)

    rep = CartesianRepresentation(X.ravel() * u.AU, Y.ravel() * u.AU, Z.ravel() * u.AU)
    tf = target_frame.upper()

    if tf == "GSE":
        ct = SkyCoord(rep, frame=GeocentricSolarEcliptic(obstime=obstime))
    else:
        c_hee = SkyCoord(rep, frame=HeliocentricEarthEcliptic(obstime=obstime))
        if tf == "HEE":
            ct = c_hee
        elif tf == "HCI":
            ct = c_hee.transform_to(HeliocentricInertial(obstime=obstime))
        else:
            raise ValueError("target_frame must be 'HEE', 'HCI' or 'GSE'")

    Xt = ct.cartesian.x.to_value(u.AU).reshape(X.shape)
    Yt = ct.cartesian.y.to_value(u.AU).reshape(X.shape)
    Zt = ct.cartesian.z.to_value(u.AU).reshape(X.shape)
    return Xt, Yt, Zt


def _carrington_and_footpoints(
    spkid: str,
    start: str,
    stop: str,
    step: str,
    rss_rsun: float,
    omega_deg_per_day: float,
    vsw_kms: float,
):
    coord0 = get_horizons_coord(spkid, {"start": start, "stop": stop, "step": step})
    hgs = coord0.transform_to(HeliographicStonyhurst(obstime=coord0.obstime))

    earth_obs = get_body_heliographic_stonyhurst("earth", hgs.obstime)
    hgc_frame = HeliographicCarrington(obstime=hgs.obstime, observer=earth_obs)
    carr = hgs.transform_to(hgc_frame)

    Rsun = const.R_sun.to(u.km)
    rss = (rss_rsun * const.R_sun).to(u.km)

    omega = (omega_deg_per_day * u.deg) / (24 * 3600 * u.s)
    vsw = (vsw_kms * u.km / u.s)

    r_sc = carr.spherical.distance.to(u.km)
    lon_sc = carr.lon
    lat_sc = carr.lat

    delta = omega * (r_sc - rss) / vsw
    lon_ss = (lon_sc + delta).wrap_at(360 * u.deg)

    fp_ss = SkyCoord(lon=lon_ss, lat=lat_sc, radius=rss, frame=hgc_frame)
    fp_sun = SkyCoord(lon=lon_ss, lat=lat_sc, radius=Rsun, frame=hgc_frame)

    t = pd.to_datetime(carr.obstime.datetime64)

    return t, carr, fp_ss, fp_sun


def _to_xyz_in_frame(coord: SkyCoord, frame: str):
    if frame.upper() == "HEE":
        fr = HeliocentricEarthEcliptic(obstime=coord.obstime)
    elif frame.upper() == "HCI":
        fr = HeliocentricInertial(obstime=coord.obstime)
    elif frame.upper() == "GSE":
        fr = GeocentricSolarEcliptic(obstime=coord.obstime)
    else:
        raise ValueError("frame must be 'HEE', 'HCI' or 'GSE'")
    c = coord.transform_to(fr)
    return (
        c.cartesian.x.to_value(u.AU),
        c.cartesian.y.to_value(u.AU),
        c.cartesian.z.to_value(u.AU),
    )


def build_3d_figure(
    targets: list[str],
    start: str,
    stop: str,
    step: str,
    frame3d: str = "HEE",
    rss_rsun: float = 2.5,
    omega_deg_per_day: float = 14.1844,
    vsw_kms: tuple[float, float | None] = (300.0, 700.0),
    vsw1_kms: float | None = None,
    vsw2_kms: float | None = None,
    width: int = 1800,
    height: int = 900,
    sun_zoom_au: float = 0.06,
    plane_span_au: float = 1.2,
    show_spokes: bool = True,
    spoke_count: int = 8,
    decimate: int = 1,
    gse_axis_units: str = "Re",
    verbose: bool = True,
):
    """
    3D visualization (all units in AU).

    - HEE/HCI: two panels (AU-scale trajectories + Sun-zoom ballistic backmapping inset).
    - GSE: single geocentric panel only (no ballistic mapping to the Sun).
    """
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    frame3d = frame3d.upper()
    if frame3d not in {"HEE", "HCI", "GSE"}:
        raise ValueError("frame3d must be 'HEE', 'HCI' or 'GSE'")

    if decimate < 1:
        decimate = 1

    if (vsw1_kms is not None) or (vsw2_kms is not None):
        vsw_kms = (vsw1_kms if vsw1_kms is not None else 300.0, vsw2_kms)

    vsw_list = [float(v) for v in vsw_kms if v is not None]
    if len(vsw_list) == 0:
        raise ValueError("Provide at least one Vsw in vsw_kms")

    colors = px.colors.qualitative.Plotly
    dash_map = {
        "ACE": "solid",
        "WIND": "dash",
        "IMAP": "dot",
        "SOLAR-1": "dashdot",
        "SWFO-L1": "dashdot",
        "SWIFO-1": "dashdot",
        "DSCOVR": "longdash",
        "DISCOVER": "longdash",
        "ADITYA": "longdashdot",
        "ADITYA-L1": "longdashdot",
        "PSP": "solid",
    }
    sym_map = {
        "ACE": "circle",
        "WIND": "square",
        "IMAP": "diamond",
        "SOLAR-1": "diamond-open",
        "SWFO-L1": "diamond-open",
        "SWIFO-1": "diamond-open",
        "DSCOVR": "x",
        "DISCOVER": "x",
        "ADITYA": "cross",
        "ADITYA-L1": "cross",
        "PSP": "circle",
    }

    geocentric = frame3d == "GSE"
    gse_units = str(gse_axis_units).strip().upper()
    if geocentric and gse_units not in {"AU", "RE"}:
        raise ValueError("gse_axis_units must be 'AU' or 'Re'")

    au_to_re = (u.AU / const.R_earth).decompose().value
    pos_scale = au_to_re if (geocentric and gse_units == "RE") else 1.0
    coord_unit = "Re" if (geocentric and gse_units == "RE") else "AU"

    if verbose:
        print(f"[3D] Building 3D figure in frame3d={frame3d} (positions in AU).")
        if geocentric:
            print("[3D] GSE mode: single geocentric panel with axis/sun-wind guides (no Sun backmapping inset).")
        else:
            print("[3D] Left panel: spacecraft/Earth trajectories + Sun-centered ecliptic plane.")
            print(f"[3D] Right panel: Sun-zoom inset (±{sun_zoom_au:.3f} AU) with ballistic backmapping.")
            print(f"[3D] Backmapping: r_ss={rss_rsun:.2f} R_sun, Ω={omega_deg_per_day:.4f} deg/day, Vsw={vsw_list} km/s.")
            print("[3D] Assumptions: constant Vsw, rigid Ω, radial projection below r_ss.")

    if geocentric:
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
        title_text = f"3D geocentric trajectories (frame={frame3d})"
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            column_widths=[0.66, 0.34],
            horizontal_spacing=0.02,
            subplot_titles=("AU-scale view", "Sun-zoom backmapping"),
        )
        title_text = f"3D orbits + ecliptic plane + ballistic backmapping (frame={frame3d})"

    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=85, b=0),
        font=dict(family="Arial", size=14),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        legend=dict(groupclick="togglegroup"),
    )
    if not geocentric:
        fig.update_annotations(font_size=15)

    max_abs_extent = 0.0
    gse_extent_samples = []

    # --- ecliptic plane patch (Sun-centered for heliocentric frames; Earth-centered for GSE) ---
    earth_df = _get_xyz_timeseries("399", start, stop, step, frame=frame3d)
    obstime_ref = earth_df.index[len(earth_df) // 2].to_pydatetime()

    if verbose:
        print("[3D] Adding ecliptic plane patch in target frame.")

    Xp, Yp, Zp = _plane_patch_in_target_frame(obstime=obstime_ref, span_au=plane_span_au, target_frame=frame3d)
    if geocentric and pos_scale != 1.0:
        Xp, Yp, Zp = Xp * pos_scale, Yp * pos_scale, Zp * pos_scale
    fig.add_trace(
        go.Surface(
            x=Xp,
            y=Yp,
            z=Zp,
            showscale=False,
            opacity=0.10,
            hoverinfo="skip",
            name="Ecliptic plane",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Sun markers/rings (realistic AU units) ---
    Rsun_au = const.R_sun.to_value(u.AU)
    rss_au = (rss_rsun * const.R_sun).to_value(u.AU)

    if geocentric:
        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers+text",
                name="Earth",
                marker=dict(size=6, symbol="circle"),
                text=["Earth"],
                textposition="top center",
                hovertemplate="Earth at origin (GSE)<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers+text",
                name="Sun",
                marker=dict(size=6, symbol="circle"),
                text=["Sun"],
                textposition="top center",
                hovertemplate=f"Sun (R_sun={Rsun_au:.5f} AU)<extra></extra>",
            ),
            row=1,
            col=1,
        )

    theta = np.linspace(0, 2 * np.pi, 260)
    if not geocentric:
        # Sun ring in inset
        fig.add_trace(
            go.Scatter3d(
                x=Rsun_au * np.cos(theta),
                y=Rsun_au * np.sin(theta),
                z=np.zeros_like(theta),
                mode="lines",
                line=dict(width=3),
                opacity=0.75,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        # Source surface ring in inset
        fig.add_trace(
            go.Scatter3d(
                x=rss_au * np.cos(theta),
                y=rss_au * np.sin(theta),
                z=np.zeros_like(theta),
                mode="lines",
                line=dict(width=2, dash="dot"),
                opacity=0.40,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # --- Earth trajectory (heliocentric frames only) ---
    if not geocentric:
        if verbose:
            print("[3D] Adding Earth trajectory (AU-scale).")

        fig.add_trace(
            go.Scatter3d(
                x=earth_df["x_au"][::decimate],
                y=earth_df["y_au"][::decimate],
                z=earth_df["z_au"][::decimate],
                mode="lines",
                name="Earth (traj)",
                line=dict(width=3),
                opacity=0.55,
                hovertemplate="Earth trajectory<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[float(earth_df["x_au"].iloc[-1])],
                y=[float(earth_df["y_au"].iloc[-1])],
                z=[float(earth_df["z_au"].iloc[-1])],
                mode="markers+text",
                marker=dict(size=5),
                text=["Earth"],
                textposition="top center",
                showlegend=False,
                hovertemplate="Earth (marker shown; physical R_earth is tiny in AU)<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # --- spacecraft trajectories + backmapping traces (in inset) ---
    for i, name in enumerate(targets):
        spkid = resolve_spacecraft_spkid(name)
        col = colors[i % len(colors)]
        alias = str(name).strip().upper()
        dash = dash_map.get(alias, "solid")
        sym = sym_map.get(alias, "circle")

        if verbose:
            print(f"[3D] {name}: fetching ephemeris (SPKID={spkid}), building trajectory and footpoints...")

        sc_df = _get_xyz_timeseries(spkid, start, stop, step, frame=frame3d).iloc[::decimate]
        sc_xyz = sc_df[["x_au", "y_au", "z_au"]].to_numpy() * pos_scale
        max_abs_extent = max(max_abs_extent, float(np.nanmax(np.abs(sc_xyz))))
        if geocentric:
            gse_extent_samples.append(sc_xyz)

        # AU-scale trajectory
        fig.add_trace(
            go.Scatter3d(
                x=sc_df["x_au"] * pos_scale,
                y=sc_df["y_au"] * pos_scale,
                z=sc_df["z_au"] * pos_scale,
                mode="lines",
                name=name,
                line=dict(width=4, dash=dash, color=col),
                hovertemplate=f"{name}<br>x=%{{x:.3f}} {coord_unit}<br>y=%{{y:.3f}} {coord_unit}<br>z=%{{z:.3f}} {coord_unit}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # end marker label
        fig.add_trace(
            go.Scatter3d(
                x=[float(sc_df["x_au"].iloc[-1] * pos_scale)],
                y=[float(sc_df["y_au"].iloc[-1] * pos_scale)],
                z=[float(sc_df["z_au"].iloc[-1] * pos_scale)],
                mode="markers+text",
                marker=dict(size=5, color=col, symbol=sym),
                text=[name],
                textposition="top center",
                showlegend=False,
                hovertemplate=f"{name} (end)<extra></extra>",
            ),
            row=1,
            col=1,
        )

        if not geocentric:
            # backmapping: for each vsw
            for j, vsw in enumerate(vsw_list):
                _, _, fp_ss, fp_sun = _carrington_and_footpoints(
                    spkid=spkid,
                    start=start,
                    stop=stop,
                    step=step,
                    rss_rsun=rss_rsun,
                    omega_deg_per_day=omega_deg_per_day,
                    vsw_kms=vsw,
                )

                fp_ss_sel = fp_ss[::decimate]
                fp_sun_sel = fp_sun[::decimate]

                x_ss, y_ss, z_ss = _to_xyz_in_frame(fp_ss_sel, frame3d)
                x_su, y_su, z_su = _to_xyz_in_frame(fp_sun_sel, frame3d)

                # style: first Vsw = solid family, second Vsw = dashed family
                ss_dash = "solid" if j == 0 else "dash"
                su_dash = "dot" if j == 0 else "dashdot"
                alpha = 0.90 if j == 0 else 0.55

                # source-surface footpoints
                fig.add_trace(
                    go.Scatter3d(
                        x=x_ss,
                        y=y_ss,
                        z=z_ss,
                        mode="lines",
                        line=dict(width=3, dash=ss_dash, color=col),
                        opacity=alpha,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )
                # photospheric projection
                fig.add_trace(
                    go.Scatter3d(
                        x=x_su,
                        y=y_su,
                        z=z_su,
                        mode="lines",
                        line=dict(width=5, dash=su_dash, color=col),
                        opacity=min(0.95, alpha + 0.1),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )

                # spokes (visual guide only): a few lines from sc -> R_sun footpoint (for the primary Vsw only)
                if show_spokes and j == 0 and len(sc_df) > 3:
                    idx = np.linspace(0, len(sc_df) - 1, min(spoke_count, len(sc_df))).astype(int)
                    x_sc = (sc_df["x_au"].to_numpy() * pos_scale)[idx]
                    y_sc = (sc_df["y_au"].to_numpy() * pos_scale)[idx]
                    z_sc = (sc_df["z_au"].to_numpy() * pos_scale)[idx]

                    x_fp = np.array(x_su)[idx]
                    y_fp = np.array(y_su)[idx]
                    z_fp = np.array(z_su)[idx]

                    Xl = np.empty(3 * len(idx))
                    Yl = np.empty(3 * len(idx))
                    Zl = np.empty(3 * len(idx))
                    Xl[0::3], Yl[0::3], Zl[0::3] = x_sc, y_sc, z_sc
                    Xl[1::3], Yl[1::3], Zl[1::3] = x_fp, y_fp, z_fp
                    Xl[2::3], Yl[2::3], Zl[2::3] = np.nan, np.nan, np.nan

                    fig.add_trace(
                        go.Scatter3d(
                            x=Xl, y=Yl, z=Zl,
                            mode="lines",
                            line=dict(width=2, color=col),
                            opacity=0.18,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )

    # --- scene formatting: make it look like a real “orbital geometry” figure ---
    if geocentric:
        axis_title = f"[{coord_unit}] (GSE)"

        if gse_extent_samples:
            arr = np.vstack(gse_extent_samples)
            robust_extent = float(np.nanpercentile(np.abs(arr), 95.0))
        else:
            robust_extent = (plane_span_au * pos_scale) * 0.5

        lim = max(0.1, 1.25 * robust_extent)
        axis_len = 0.92 * lim

        # Explicit GSE axis guides (X:red, Y:green, Z:blue)
        for axis_name, vec, colr in [
            ("+X (Sunward)", (axis_len, 0.0, 0.0), "#d62728"),
            ("+Y (Dusk)", (0.0, axis_len, 0.0), "#2ca02c"),
            ("+Z (North)", (0.0, 0.0, axis_len), "#1f77b4"),
        ]:
            vx, vy, vz = vec
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, vx], y=[0.0, vy], z=[0.0, vz],
                    mode="lines+text",
                    line=dict(width=7, color=colr),
                    text=["", axis_name],
                    textposition="top center",
                    name=f"GSE {axis_name}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # Solar wind direction cue in GSE: approximately toward -X at Earth
        sw_start = 0.92 * lim
        sw_end = 0.15 * lim
        fig.add_trace(
            go.Scatter3d(
                x=[sw_start, sw_end], y=[0.0, 0.0], z=[0.0, 0.0],
                mode="lines+text",
                line=dict(width=8, color="#ff7f0e", dash="dash"),
                text=["", "Solar wind → Earth"],
                textposition="top center",
                name="Solar wind direction",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        # Indicate Sunward side on +X boundary without forcing zoom-out with full Sun trajectory
        fig.add_trace(
            go.Scatter3d(
                x=[0.96 * lim], y=[0.0], z=[0.0],
                mode="markers+text",
                marker=dict(size=5, color="#f2c14e", symbol="circle"),
                text=["Sunward"],
                textposition="top center",
                name="Sunward boundary",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=f"X {axis_title}",
                yaxis_title=f"Y {axis_title}",
                zaxis_title=f"Z {axis_title}",
                aspectmode="cube",
                xaxis=dict(range=[-lim, lim], showspikes=False, gridcolor="rgba(0,0,0,0.10)", zeroline=True, zerolinecolor="rgba(80,80,80,0.4)"),
                yaxis=dict(range=[-lim, lim], showspikes=False, gridcolor="rgba(0,0,0,0.10)", zeroline=True, zerolinecolor="rgba(80,80,80,0.4)"),
                zaxis=dict(range=[-lim, lim], showspikes=False, gridcolor="rgba(0,0,0,0.10)", zeroline=True, zerolinecolor="rgba(80,80,80,0.4)"),
            ),
        )
        fig.update_scenes(camera=dict(eye=dict(x=1.05, y=1.0, z=0.7)), row=1, col=1)
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title="x [AU]",
                yaxis_title="y [AU]",
                zaxis_title="z [AU]",
                aspectmode="data",
                xaxis=dict(showspikes=False, gridcolor="rgba(0,0,0,0.10)"),
                yaxis=dict(showspikes=False, gridcolor="rgba(0,0,0,0.10)"),
                zaxis=dict(showspikes=False, gridcolor="rgba(0,0,0,0.10)"),
            ),
            scene2=dict(
                xaxis_title="x [AU]",
                yaxis_title="y [AU]",
                zaxis_title="z [AU]",
                aspectmode="cube",
                xaxis=dict(range=[-sun_zoom_au, sun_zoom_au], showspikes=False),
                yaxis=dict(range=[-sun_zoom_au, sun_zoom_au], showspikes=False),
                zaxis=dict(range=[-sun_zoom_au, sun_zoom_au], showspikes=False),
            ),
        )
        # Cameras (stable defaults)
        fig.update_scenes(camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)), row=1, col=1)
        fig.update_scenes(camera=dict(eye=dict(x=1.75, y=1.20, z=0.75)), row=1, col=2)

    return fig



def write_combined_html(fig_ts: go.Figure, fig_3d: go.Figure, out_html: str):
    html_ts = pio.to_html(fig_ts, include_plotlyjs="cdn", full_html=False)
    html_3d = pio.to_html(fig_3d, include_plotlyjs=False, full_html=False)

    page = f"""
<html>
<head><meta charset="utf-8"></head>
<body>
<div style="max-width: 100%; margin: 0 auto;">{html_ts}</div>
<hr style="margin: 40px 0;">
<div style="max-width: 100%; margin: 0 auto;">{html_3d}</div>
</body>
</html>
"""
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

    p.add_argument("--frame3d", choices=["HEE", "HCI", "GSE"], default="HEE")

    p.add_argument("--vsw1-kms", type=float, default=300.0)
    p.add_argument("--vsw2-kms", type=float, default=700.0)

    p.add_argument("--sun-zoom-au", type=float, default=0.06)
    p.add_argument("--plane-span-au", type=float, default=1.2)

    p.add_argument("--no-spokes", action="store_true")
    p.add_argument("--gse-axis-units", choices=["AU", "Re", "RE", "au", "re"], default="Re")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)

    fig_ts = build_timeseries_figure(
        targets=args.targets,
        start=args.start,
        stop=args.stop,
        step=args.step,
        rss_rsun=args.rss_rsun,
        omega_deg_per_day=args.omega_deg_per_day,
        width=args.width,
        height=args.height,
    )

    vsw2 = args.vsw2_kms if args.vsw2_kms is not None else None

    fig_3d = build_3d_figure(
        targets=args.targets,
        start=args.start,
        stop=args.stop,
        step=args.step,
        frame3d=args.frame3d,
        rss_rsun=args.rss_rsun,
        omega_deg_per_day=args.omega_deg_per_day,
        vsw1_kms=args.vsw1_kms,
        vsw2_kms=vsw2,
        width=args.width3d,
        height=args.height3d,
        sun_zoom_au=args.sun_zoom_au,
        plane_span_au=args.plane_span_au,
        show_spokes=(not args.no_spokes),
        gse_axis_units=args.gse_axis_units,
    )

    write_combined_html(fig_ts, fig_3d, args.html)
    print(f"Wrote: {args.html}")


if __name__ == "__main__":
    main()
