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

try:
    from . import helpers
    from .horizons_sun_lonlat import get_repo_style_orbit_df, resolve_spacecraft_spkid
except ImportError:  # pragma: no cover
    import helpers
    from horizons_sun_lonlat import get_repo_style_orbit_df, resolve_spacecraft_spkid


AU_IN_RE = (1 * u.AU).to_value(u.Rearth)


def _get_xyz_timeseries(target_id, start, stop, step, frame):
    time_spec = {"start": start, "stop": stop, "step": step}

    tid = str(target_id).strip()
    id_type = "id" if tid.lstrip("+-").isdigit() else None

    coord0 = get_horizons_coord(tid, time_spec, id_type=id_type)

    fr_name = str(frame).upper()
    if fr_name == "HEE":
        fr = HeliocentricEarthEcliptic(obstime=coord0.obstime)
    elif fr_name == "HCI":
        fr = HeliocentricInertial(obstime=coord0.obstime)
    elif fr_name == "GSE":
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
    targets,
    start,
    stop,
    step,
    rss_rsun,
    omega_deg_per_day,
    width,
    height,
):
    colors = px.colors.qualitative.Plotly

    DASH = {
        "ACE": "solid",
        "WIND": "dash",
        "IMAP": "dot",
        "SOLAR-1": "dashdot",
        "SWFO-L1": "dashdot",
        "SWIFO-1": "dashdot",
        "DSCOVR": "longdash",
        "ADITYA": "longdashdot",
        "PSP": "solid",
        "SOLO": "solid",
        "SOHO": "solid",
    }
    SYM = {
        "ACE": "circle",
        "WIND": "square",
        "IMAP": "diamond",
        "SOLAR-1": "diamond-open",
        "SWFO-L1": "triangle-up",
        "SWIFO-1": "triangle-up",
        "DSCOVR": "x",
        "ADITYA": "cross",
        "PSP": "circle",
        "SOLO": "circle",
        "SOHO": "circle",
    }

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

        key = str(t).strip().upper()
        fig = helpers.add_sc(
            fig,
            df,
            t,
            colors[i % len(colors)],
            line_dash=DASH.get(key, "solid"),
            marker_symbol=SYM.get(key, "circle"),
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
    span_au,
    target_frame,
    n=35,
):
    grid = np.linspace(-span_au, span_au, n)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)

    rep = CartesianRepresentation(X.ravel() * u.AU, Y.ravel() * u.AU, Z.ravel() * u.AU)
    tf = str(target_frame).upper()

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
    spkid,
    start,
    stop,
    step,
    rss_rsun,
    omega_deg_per_day,
    vsw_kms,
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


def _to_xyz_in_frame(coord, frame):
    fr_name = str(frame).upper()
    if fr_name == "HEE":
        fr = HeliocentricEarthEcliptic(obstime=coord.obstime)
    elif fr_name == "HCI":
        fr = HeliocentricInertial(obstime=coord.obstime)
    elif fr_name == "GSE":
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
    targets,
    start,
    stop,
    step,
    frame3d="HEE",
    rss_rsun=2.5,
    omega_deg_per_day=14.1844,
    vsw_kms=(300.0, 700.0),
    vsw1_kms=None,
    vsw2_kms=None,
    width=1800,
    height=900,
    sun_zoom_au=0.06,
    plane_span_au=1.2,
    show_spokes=True,
    spoke_count=8,
    decimate=1,
    gse_axis_units="Re",
    verbose=True,
):
    frame3d = str(frame3d).upper()
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

    l1_dist = (0.01 * u.AU).to_value(u.AU) * pos_scale
    l2_dist = (0.01 * u.AU).to_value(u.AU) * pos_scale

    if verbose:
        print(f"[3D] Building 3D figure in frame3d={frame3d}.")
        if geocentric:
            print("[3D] GSE mode: single geocentric panel (no Sun backmapping inset).")
        else:
            print("[3D] Left panel: AU-scale trajectories + ecliptic plane.")
            print(f"[3D] Right panel: Sun-zoom inset (±{sun_zoom_au:.3f} AU) with ballistic backmapping.")
            print(f"[3D] Backmapping: r_ss={rss_rsun:.2f} R_sun, Ω={omega_deg_per_day:.4f} deg/day, Vsw={vsw_list} km/s.")

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

    gse_extent_samples = []

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
            opacity=(0.03 if geocentric else 0.10),
            hoverinfo="skip",
            name="Ecliptic plane",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

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

    text_positions = [
        "top center", "middle right", "middle left", "bottom center",
        "top right", "top left", "bottom right", "bottom left",
    ]

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
        if geocentric:
            gse_extent_samples.append(sc_xyz)

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

        fig.add_trace(
            go.Scatter3d(
                x=[float(sc_df["x_au"].iloc[-1] * pos_scale)],
                y=[float(sc_df["y_au"].iloc[-1] * pos_scale)],
                z=[float(sc_df["z_au"].iloc[-1] * pos_scale)],
                mode="markers+text",
                marker=dict(size=5, color=col, symbol=sym),
                text=[name],
                textposition=text_positions[i % len(text_positions)],
                showlegend=False,
                hovertemplate=f"{name} (end)<extra></extra>",
            ),
            row=1,
            col=1,
        )

        if not geocentric:
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

                ss_dash = "solid" if j == 0 else "dash"
                su_dash = "dot" if j == 0 else "dashdot"
                alpha = 0.90 if j == 0 else 0.55

                fig.add_trace(
                    go.Scatter3d(
                        x=x_ss, y=y_ss, z=z_ss,
                        mode="lines",
                        line=dict(width=3, dash=ss_dash, color=col),
                        opacity=alpha,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=x_su, y=y_su, z=z_su,
                        mode="lines",
                        line=dict(width=5, dash=su_dash, color=col),
                        opacity=min(0.95, alpha + 0.1),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=2,
                )

                if show_spokes and j == 0 and len(sc_df) > 3:
                    idx = np.linspace(0, len(sc_df) - 1, min(spoke_count, len(sc_df))).astype(int)
                    x_sc = (sc_df["x_au"].to_numpy() * pos_scale)[idx]
                    y_sc = (sc_df["y_au"].to_numpy() * pos_scale)[idx]
                    z_sc = (sc_df["z_au"].to_numpy() * pos_scale)[idx]

                    x_fp = np.asarray(x_su)[idx]
                    y_fp = np.asarray(y_su)[idx]
                    z_fp = np.asarray(z_su)[idx]

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

    if geocentric:
        axis_title = f"[{coord_unit}] (GSE)"

        if gse_extent_samples:
            arr = np.vstack(gse_extent_samples)
            robust_extent = float(np.nanpercentile(np.abs(arr), 95.0))
        else:
            robust_extent = (plane_span_au * pos_scale) * 0.5

        lim = max(0.1, 1.5 * robust_extent)
        axis_len = 0.92 * lim

        for axis_name, vec in [
            ("+X (Sunward)", (axis_len, 0.0, 0.0)),
            ("+Y (Dusk)", (0.0, axis_len, 0.0)),
            ("+Z (North)", (0.0, 0.0, axis_len)),
        ]:
            vx, vy, vz = vec
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, vx], y=[0.0, vy], z=[0.0, vz],
                    mode="lines+text",
                    line=dict(width=5, color="rgba(0,0,0,0.35)"),
                    opacity=0.65,
                    text=["", axis_name],
                    textposition="top center",
                    name=f"GSE {axis_name}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

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

        fig.add_trace(
            go.Scatter3d(
                x=[l1_dist, -l2_dist], y=[0.0, 0.0], z=[0.0, 0.0],
                mode="markers+text",
                marker=dict(size=4, color="black", symbol="circle"),
                text=["L1", "L2"],
                textposition="top center",
                name="Lagrange points",
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
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
                xaxis=dict(
                    range=[-lim, lim],
                    showspikes=False,
                    showbackground=False,
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    ticks="outside",
                    tickcolor="black",
                    tickfont=dict(color="black"),
                    title=dict(font=dict(color="black")),
                ),
                yaxis=dict(
                    range=[-lim, lim],
                    showspikes=False,
                    showbackground=False,
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    ticks="outside",
                    tickcolor="black",
                    tickfont=dict(color="black"),
                    title=dict(font=dict(color="black")),
                ),
                zaxis=dict(
                    range=[-lim, lim],
                    showspikes=False,
                    showbackground=False,
                    showline=True,
                    linecolor="black",
                    linewidth=2,
                    ticks="outside",
                    tickcolor="black",
                    tickfont=dict(color="black"),
                    title=dict(font=dict(color="black")),
                ),
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
        fig.update_scenes(camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)), row=1, col=1)
        fig.update_scenes(camera=dict(eye=dict(x=1.75, y=1.20, z=0.75)), row=1, col=2)

    return fig


def _pairwise_separations(xyz, flow_hat):
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    if n < 2:
        return np.array([]), np.array([]), np.array([])

    flow_hat = np.asarray(flow_hat, dtype=float)
    flow_hat = flow_hat / np.linalg.norm(flow_hat)

    dr = xyz[:, None, :] - xyz[None, :, :]
    dpar = np.tensordot(dr, flow_hat, axes=([2], [0]))
    dpar_abs = np.abs(dpar)

    dr_perp = dr - dpar[..., None] * flow_hat[None, None, :]
    dperp = np.linalg.norm(dr_perp, axis=2)

    iu = np.triu_indices(n, k=1)
    return dperp[iu], dpar_abs[iu], dpar[iu]


def _pair_prob_same_stream(dperp_au, dpar_abs_au, vsw_kms, lperp_au, tau0_h):
    tau_h = ((dpar_abs_au * u.AU) / (vsw_kms * u.km / u.s)).to_value(u.hour)
    exponent = - (dperp_au / lperp_au) ** 2 - (tau_h / tau0_h) ** 2
    p = np.exp(exponent)
    return np.clip(p, 1e-12, 1.0), tau_h


def _all_prob_same_stream_worst_pair(dperp_au, dpar_abs_au, vsw_kms, lperp_au, tau0_h):
    dperp_au = np.asarray(dperp_au, dtype=float)
    dpar_abs_au = np.asarray(dpar_abs_au, dtype=float)

    if dperp_au.size == 0:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    AU_KM = 149597870.7
    tau_h = (dpar_abs_au * AU_KM) / max(float(vsw_kms), 1e-6) / 3600.0

    lperp = max(float(lperp_au), 1e-12)
    tau0 = max(float(tau0_h), 1e-12)

    E = (dperp_au / lperp) ** 2 + (tau_h / tau0) ** 2
    k = int(np.nanargmax(E))
    Emax = float(E[k])

    p_all = float(np.clip(np.exp(-Emax), 1e-12, 1.0))
    return p_all, float(dperp_au[k]), float(dpar_abs_au[k]), float(tau_h[k]), Emax

def find_best_stream_aligned_intervals(
    targets,
    start,
    stop,
    step="1h",
    window_hours=12.0,
    top_n=3,
    frame="GSE",
    flow_dir_gse=(-1.0, 0.0, 0.0),
    vsw_kms=400.0,
    along_weight=0.20,
    perp_scale=0.35,
    lag_tolerance=0.5,
    vsw_rel_unc=0.15,
    decorrelation_rel_unc=0.25,
    min_coverage=0.75,
    verbose=True,
):
    import numpy as np
    import pandas as pd
    import astropy.units as u
    import astropy.constants as const

    if str(frame).upper() != "GSE":
        raise ValueError("This interval-selection metric is implemented for GSE only.")
    if float(window_hours) <= 0:
        raise ValueError("window_hours must be > 0")
    if int(top_n) < 1:
        raise ValueError("top_n must be >= 1")
    if float(perp_scale) <= 0:
        raise ValueError("perp_scale must be > 0")
    if float(lag_tolerance) <= 0:
        raise ValueError("lag_tolerance must be > 0")

    flow_hat = np.asarray(flow_dir_gse, dtype=float)
    nrm = np.linalg.norm(flow_hat)
    if nrm == 0:
        raise ValueError("flow_dir_gse must be non-zero")
    flow_hat = flow_hat / nrm

    tracks = {}
    for t in targets:
        spkid = resolve_spacecraft_spkid(t)
        tracks[t] = _get_xyz_timeseries(spkid, start, stop, step, frame="GSE")

    time_index = None
    for df in tracks.values():
        time_index = df.index if time_index is None else time_index.intersection(df.index)
    time_index = pd.DatetimeIndex(sorted(time_index))
    if len(time_index) == 0:
        raise RuntimeError("No overlapping timestamps were found across spacecraft.")

    AU_KM = (1.0 * u.AU).to_value(u.km)

    T = len(time_index)
    N = len(targets)
    xyz = np.empty((T, N, 3), dtype=float)
    for j, name in enumerate(targets):
        df = tracks[name].loc[time_index]
        xyz[:, j, 0] = df["x_au"].to_numpy(dtype=float)
        xyz[:, j, 1] = df["y_au"].to_numpy(dtype=float)
        xyz[:, j, 2] = df["z_au"].to_numpy(dtype=float)

    dr = xyz[:, :, None, :] - xyz[:, None, :, :]
    dpar = dr @ flow_hat
    dpar_abs = np.abs(dpar)
    dr_perp = dr - dpar[..., None] * flow_hat[None, None, :]
    dperp = np.linalg.norm(dr_perp, axis=3)

    iu = np.triu_indices(N, k=1)
    dpar_abs_pair = dpar_abs[:, iu[0], iu[1]]
    dperp_pair = dperp[:, iu[0], iu[1]]

    vsw_kms = float(vsw_kms)
    window_hours = float(window_hours)

    l_adv_au = ((vsw_kms * u.km / u.s) * (window_hours * u.hour)).to_value(u.AU)
    tau0_h = float(lag_tolerance) * window_hours
    lperp_au = float(perp_scale) * float(l_adv_au)

    tau_h_ref = (dpar_abs_pair * AU_KM) / max(vsw_kms, 1e-6) / 3600.0

    def _E_pair(vsw_kms_use, lperp_au_use, tau0_h_use):
        vsw_kms_use = max(float(vsw_kms_use), 1e-6)
        lperp_au_use = max(float(lperp_au_use), 1e-12)
        tau0_h_use = max(float(tau0_h_use), 1e-12)
        tau_h = tau_h_ref * (vsw_kms / vsw_kms_use)
        return (dperp_pair / lperp_au_use) ** 2 + (tau_h / tau0_h_use) ** 2

    E_ref = _E_pair(vsw_kms, lperp_au, tau0_h)
    Emax_ref = np.nanmax(E_ref, axis=1)
    p_all_ref = np.exp(-Emax_ref)

    P_pairs_ref = np.exp(-E_ref)
    pair_p_iqr = np.nanpercentile(P_pairs_ref, 75, axis=1) - np.nanpercentile(P_pairs_ref, 25, axis=1)

    worst_k = np.nanargmax(E_ref, axis=1)
    worst_dperp_au = dperp_pair[np.arange(T), worst_k]
    worst_dpar_abs_au = dpar_abs_pair[np.arange(T), worst_k]
    worst_tau_h = tau_h_ref[np.arange(T), worst_k]

    worst_dperp_re = worst_dperp_au * AU_IN_RE
    worst_dpar_re = worst_dpar_abs_au * AU_IN_RE

    lo_vsw = max(vsw_kms * (1.0 - float(vsw_rel_unc)), 1e-3)
    hi_vsw = vsw_kms * (1.0 + float(vsw_rel_unc))
    lo_lperp = max(lperp_au * (1.0 - float(decorrelation_rel_unc)), 1e-9)
    hi_lperp = lperp_au * (1.0 + float(decorrelation_rel_unc))
    lo_tau0 = max(tau0_h * (1.0 - float(decorrelation_rel_unc)), 1e-9)
    hi_tau0 = max(tau0_h * (1.0 + float(decorrelation_rel_unc)), 1e-9)

    E_lo = _E_pair(lo_vsw, lo_lperp, lo_tau0)
    E_hi = _E_pair(hi_vsw, hi_lperp, hi_tau0)
    Emax_lo = np.nanmax(E_lo, axis=1)
    Emax_hi = np.nanmax(E_hi, axis=1)

    metric_df = pd.DataFrame(
        {
            "Emax_ref": Emax_ref,
            "Emax_lo": Emax_lo,
            "Emax_hi": Emax_hi,
            "p_all_ref": p_all_ref,
            "pair_p_iqr": pair_p_iqr,
            "worst_dperp_re": worst_dperp_re,
            "worst_dpar_re": worst_dpar_re,
            "worst_tau_h": worst_tau_h,
        },
        index=time_index,
    )
    metric_df.index.name = "time"

    w = pd.Timedelta(hours=window_hours)
    t0 = metric_df.index[0]
    metric_df["_win_id"] = ((metric_df.index - t0) // w).astype(int)

    step_td = pd.to_timedelta(step)
    expected_per_window = max(int(np.floor(w / step_td)), 1)

    score_rows = []
    for wid, g in metric_df.groupby("_win_id"):
        if len(g) == 0:
            continue

        ws = t0 + wid * w
        we = ws + w

        coverage = float(len(g)) / float(expected_per_window)
        if coverage < float(min_coverage):
            continue

        Eref = g["Emax_ref"].to_numpy(dtype=float)
        Elo = g["Emax_lo"].to_numpy(dtype=float)
        Ehi = g["Emax_hi"].to_numpy(dtype=float)

        score_q84 = float(np.nanpercentile(Eref, 84))
        var_pen = float(np.nanmedian(g["pair_p_iqr"].to_numpy(dtype=float)))
        score = float(score_q84 + float(along_weight) * var_pen)

        score_lo = float(np.nanpercentile(np.minimum(Elo, Ehi), 84) + float(along_weight) * var_pen)
        score_hi = float(np.nanpercentile(np.maximum(Elo, Ehi), 84) + float(along_weight) * var_pen)

        pall = g["p_all_ref"].to_numpy(dtype=float)

        score_rows.append(
            {
                "window_start": ws,
                "window_end": we,
                "n_samples": int(len(g)),
                "coverage": coverage,
                "score_q84": score_q84,
                "pair_score_iqr_med": var_pen,
                "alignment_metric": score,
                "alignment_metric_p16": min(score_lo, score_hi),
                "alignment_metric_p84": max(score_lo, score_hi),
                "p_all_q16": float(np.nanpercentile(pall, 16)),
                "p_all_q50": float(np.nanpercentile(pall, 50)),
                "p_all_q84": float(np.nanpercentile(pall, 84)),
                "worst_dperp_re_q84": float(np.nanpercentile(g["worst_dperp_re"].to_numpy(dtype=float), 84)),
                "worst_tau_h_q84": float(np.nanpercentile(g["worst_tau_h"].to_numpy(dtype=float), 84)),
                "median_perp_au": float(np.nanmedian(dperp_pair[g.index.map(metric_df.index.get_loc)].ravel())),
                "median_par_au": float(np.nanmedian(dpar_abs_pair[g.index.map(metric_df.index.get_loc)].ravel())),
                "median_tau_h": float(np.nanpercentile((g["worst_tau_h"].to_numpy(dtype=float)), 50)),
                "vsw_kms": float(vsw_kms),
                "l_adv_au": float(l_adv_au),
                "lperp_au": float(lperp_au),
                "tau0_h": float(tau0_h),
            }
        )

    scores = pd.DataFrame(score_rows).sort_values("alignment_metric", ascending=True).reset_index(drop=True)
    if len(scores) == 0:
        raise RuntimeError("No valid windows found. Try reducing min_coverage or window_hours.")

    best = scores.head(int(top_n)).copy()
    metric_df = metric_df.drop(columns=["_win_id"])

    if verbose:
        print("[alignment v3] Worst-pair misalignment:")
        print("  Emax(t) = max_pairs [(d⊥/L⊥)^2 + (τ/τ0)^2],  p_all(t)=exp(-Emax).")
        print("  Window score = Q84(Emax) + w * median(IQR(pair p)). Lower is better.")
        print(f"  L_adv={l_adv_au:.4f} AU ({l_adv_au * AU_IN_RE:.1f} Re), "
              f"L_perp={lperp_au:.4f} AU ({lperp_au * AU_IN_RE:.1f} Re), tau0={tau0_h:.2f} h.")

    return {
        "scores": scores,
        "best": best,
        "tracks": tracks,
        "flow_hat": flow_hat,
        "metric_df": metric_df,
    }
    

def plot_stream_alignment_interval(
    tracks,
    targets,
    window_start,
    window_end,
    flow_hat,
    summary_row=None,
    width=1350,
    height=880,
    max_points_3d=350,
    max_pair_points=4500,
):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    AU_KM = 149597870.7

    colors = px.colors.qualitative.Plotly
    w0 = pd.Timestamp(window_start)
    w1 = pd.Timestamp(window_end)

    idx = None
    for t in targets:
        seg = tracks[t].loc[(tracks[t].index >= w0) & (tracks[t].index < w1)]
        idx = seg.index if idx is None else idx.intersection(seg.index)
    idx = pd.DatetimeIndex(sorted(idx))
    if len(idx) == 0:
        raise RuntimeError("No overlapping samples inside the requested interval.")

    T = len(idx)
    N = len(targets)

    xyz = np.empty((T, N, 3), dtype=float)
    for j, name in enumerate(targets):
        df = tracks[name].loc[idx]
        xyz[:, j, 0] = df["x_au"].to_numpy(dtype=float)
        xyz[:, j, 1] = df["y_au"].to_numpy(dtype=float)
        xyz[:, j, 2] = df["z_au"].to_numpy(dtype=float)

    flow_hat = np.asarray(flow_hat, dtype=float)
    nrm = np.linalg.norm(flow_hat)
    if nrm == 0:
        raise ValueError("flow_hat must be non-zero")
    flow_hat = flow_hat / nrm

    if summary_row is not None:
        vsw_kms = float(summary_row.get("vsw_kms", 400.0))
        lperp_au = float(summary_row.get("lperp_au", np.nan))
        tau0_h = float(summary_row.get("tau0_h", np.nan))
    else:
        vsw_kms = 400.0
        tau0_h = 0.5 * (w1 - w0).total_seconds() / 3600.0
        l_adv_au = (vsw_kms * tau0_h * 3600.0) / AU_KM
        lperp_au = 0.35 * l_adv_au

    if not np.isfinite(tau0_h) or tau0_h <= 0:
        tau0_h = 0.5 * (w1 - w0).total_seconds() / 3600.0
    if not np.isfinite(lperp_au) or lperp_au <= 0:
        l_adv_au = (vsw_kms * tau0_h * 3600.0) / AU_KM
        lperp_au = 0.35 * l_adv_au

    dr = xyz[:, :, None, :] - xyz[:, None, :, :]
    dpar = dr @ flow_hat
    dpar_abs = np.abs(dpar)
    dr_perp = dr - dpar[..., None] * flow_hat[None, None, :]
    dperp = np.linalg.norm(dr_perp, axis=3)

    iu = np.triu_indices(N, k=1)
    dpar_signed_pair = dpar[:, iu[0], iu[1]]
    dpar_abs_pair = dpar_abs[:, iu[0], iu[1]]
    dperp_pair = dperp[:, iu[0], iu[1]]

    tau_h = (dpar_abs_pair * AU_KM) / max(vsw_kms, 1e-6) / 3600.0
    E_pair = (dperp_pair / max(lperp_au, 1e-12)) ** 2 + (tau_h / max(tau0_h, 1e-12)) ** 2
    p_pair = np.exp(-E_pair)

    Emax = np.nanmax(E_pair, axis=1)
    worst_k = np.nanargmax(E_pair, axis=1)
    p_all = np.exp(-Emax)

    worst_dpar_signed_re = dpar_signed_pair[np.arange(T), worst_k] * AU_IN_RE
    worst_dperp_re = dperp_pair[np.arange(T), worst_k] * AU_IN_RE
    worst_tau_h = tau_h[np.arange(T), worst_k]

    flat_dpar_re = (dpar_signed_pair * AU_IN_RE).ravel()
    flat_dperp_re = (dperp_pair * AU_IN_RE).ravel()
    flat_E = E_pair.ravel()

    good = np.isfinite(flat_dpar_re) & np.isfinite(flat_dperp_re) & np.isfinite(flat_E)
    flat_dpar_re = flat_dpar_re[good]
    flat_dperp_re = flat_dperp_re[good]
    flat_E = flat_E[good]

    if flat_dpar_re.size > max_pair_points:
        take = np.linspace(0, flat_dpar_re.size - 1, max_pair_points).astype(int)
        flat_dpar_re = flat_dpar_re[take]
        flat_dperp_re = flat_dperp_re[take]
        flat_E = flat_E[take]

    x_lim = 10.0
    y_lim = 10.0
    if flat_dpar_re.size > 0:
        x_lim = max(x_lim, 1.15 * float(np.nanmax(np.abs(flat_dpar_re))))
        y_lim = max(y_lim, 1.15 * float(np.nanmax(flat_dperp_re)))

    stride_3d = max(1, int(np.ceil(T / max_points_3d)))
    idx_3d = idx[::stride_3d]

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "xy"}],
            [None, {"type": "xy", "secondary_y": True}],
            [None, {"type": "xy"}],
        ],
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.07,
        vertical_spacing=0.10,
    )

    arr = (xyz * AU_IN_RE).reshape(-1, 3)
    robust_extent = float(np.nanpercentile(np.abs(arr), 95.0))
    lim = max(15.0, 1.5 * robust_extent)
    axis_len = 0.92 * lim

    obstime_ref = idx[int(len(idx) // 2)].to_pydatetime()
    span_au = (lim / AU_IN_RE) * 1.05
    Xp, Yp, Zp = _plane_patch_in_target_frame(obstime=obstime_ref, span_au=span_au, target_frame="GSE")
    fig.add_trace(
        go.Surface(
            x=Xp * AU_IN_RE,
            y=Yp * AU_IN_RE,
            z=Zp * AU_IN_RE,
            showscale=False,
            opacity=0.05,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode="markers+text",
            marker=dict(size=6, symbol="circle"),
            text=["Earth"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1
    )

    l1_re = (0.01 * AU_IN_RE)
    fig.add_trace(
        go.Scatter3d(
            x=[l1_re, -l1_re], y=[0.0, 0.0], z=[0.0, 0.0],
            mode="markers+text",
            marker=dict(size=4, symbol="circle"),
            text=["L1", "L2"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1
    )

    for axis_name, vec in [
        ("+X (Sunward)", (axis_len, 0.0, 0.0)),
        ("+Y (Dusk)", (0.0, axis_len, 0.0)),
        ("+Z (North)", (0.0, 0.0, axis_len)),
    ]:
        vx, vy, vz = vec
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, vx], y=[0.0, vy], z=[0.0, vz],
                mode="lines+text",
                line=dict(width=5),
                opacity=0.6,
                text=["", axis_name],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1
        )

    text_positions = [
        "top center", "middle right", "middle left", "bottom center",
        "top right", "top left", "bottom right", "bottom left",
    ]

    for i, name in enumerate(targets):
        c = colors[i % len(colors)]
        seg3 = tracks[name].loc[idx_3d]
        xr = seg3["x_au"].to_numpy(dtype=float) * AU_IN_RE
        yr = seg3["y_au"].to_numpy(dtype=float) * AU_IN_RE
        zr = seg3["z_au"].to_numpy(dtype=float) * AU_IN_RE

        fig.add_trace(
            go.Scatter3d(
                x=xr, y=yr, z=zr,
                mode="lines",
                line=dict(width=6, color=c),
                hovertemplate=f"{name}<br>X=%{{x:.1f}} Re<br>Y=%{{y:.1f}} Re<br>Z=%{{z:.1f}} Re<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=[float(xr[-1])], y=[float(yr[-1])], z=[float(zr[-1])],
                mode="markers+text",
                marker=dict(size=6, color=c),
                text=[name],
                textposition=text_positions[i % len(text_positions)],
                hovertemplate=f"{name} (end)<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1
        )

    center = np.nanmean(arr, axis=0)
    ext = np.linalg.norm(arr - center, axis=1)
    arrow_len = max(float(np.nanpercentile(ext, 90)), 10.0)
    p0 = center
    p1 = center + (flow_hat * arrow_len)
    fig.add_trace(
        go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=8, dash="dash", color="black"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X [Re] (GSE)",
            yaxis_title="Y [Re] (GSE)",
            zaxis_title="Z [Re] (GSE)",
            aspectmode="cube",
            xaxis=dict(range=[-lim, lim], showspikes=False, showbackground=False),
            yaxis=dict(range=[-lim, lim], showspikes=False, showbackground=False),
            zaxis=dict(range=[-lim, lim], showspikes=False, showbackground=False),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=flat_dpar_re,
            y=flat_dperp_re,
            mode="markers",
            marker=dict(size=7, color="rgba(0,0,0,0.25)"),
            hovertemplate="d∥=%{x:.1f} Re<br>d⊥=%{y:.1f} Re<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=worst_dpar_signed_re,
            y=worst_dperp_re,
            mode="markers+lines",
            marker=dict(size=10),
            line=dict(width=2),
            hovertemplate="worst pair<br>d∥=%{x:.1f} Re<br>d⊥=%{y:.1f} Re<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[0.0, 0.0],
            y=[0.0, y_lim],
            mode="lines",
            line=dict(width=1, dash="dot", color="rgba(0,0,0,0.45)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="signed d∥ [Re]", range=[-x_lim, x_lim], zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="d⊥ [Re]", range=[0, y_lim], zeroline=False, row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=Emax,
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate="Emax=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=p_all,
            mode="lines",
            line=dict(dash="dot", width=2),
            hovertemplate="p_all=exp(-Emax)=%{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2, secondary_y=True
    )

    fig.update_yaxes(title_text="Emax(t)", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="p_all(t)", range=[0, 1.02], row=2, col=2, secondary_y=True)
    fig.update_xaxes(title_text="time", row=2, col=2)

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=worst_dperp_re,
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate="worst d⊥=%{y:.1f} Re<extra></extra>",
            showlegend=False,
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=worst_tau_h,
            mode="lines",
            line=dict(dash="dot", width=2),
            hovertemplate="worst τ=%{y:.2f} h<extra></extra>",
            showlegend=False,
        ),
        row=3, col=2
    )
    fig.update_yaxes(title_text="worst d⊥ [Re]  and  worst τ [h] (dotted)", row=3, col=2)
    fig.update_xaxes(title_text="time", row=3, col=2)

    title = (
        f"GSE interval: {w0} to {w1}"
        f"<br><sup>"
        f"Vsw={vsw_kms:.0f} km/s,  L⊥={lperp_au*AU_IN_RE:.1f} Re,  τ0={tau0_h:.2f} h,  "
        f"Q84(Emax)={np.nanpercentile(Emax,84):.3f},  median(p_all)={np.nanmedian(p_all):.3f}"
        f"</sup>"
    )

    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=95, b=10),
        showlegend=False,
    )

    fig.update_scenes(camera=dict(eye=dict(x=1.55, y=1.25, z=0.85)), row=1, col=1)
    return fig
    
def build_best_alignment_interval_figures(
    targets,
    start,
    stop,
    step="1h",
    window_hours=12.0,
    top_n=3,
    vsw_kms=400.0,
    along_weight=0.20,
    perp_scale=0.35,
    lag_tolerance=0.5,
    min_coverage=0.75,
    verbose=True,
):
    out = find_best_stream_aligned_intervals(
        targets=targets,
        start=start,
        stop=stop,
        step=step,
        window_hours=window_hours,
        top_n=top_n,
        frame="GSE",
        vsw_kms=vsw_kms,
        along_weight=along_weight,
        perp_scale=perp_scale,
        lag_tolerance=lag_tolerance,
        min_coverage=min_coverage,
        verbose=verbose,
    )

    figs = []
    for _, row in out["best"].iterrows():
        figs.append(
            plot_stream_alignment_interval(
                tracks=out["tracks"],
                targets=targets,
                window_start=row["window_start"],
                window_end=row["window_end"],
                flow_hat=out["flow_hat"],
                summary_row=row,
            )
        )

    return out["best"], figs, out["scores"]


def write_combined_html(fig_ts, fig_3d, out_html):
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