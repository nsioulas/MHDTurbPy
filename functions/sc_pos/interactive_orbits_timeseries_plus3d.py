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


AU_IN_RE = (1 * u.AU).to_value(u.Rearth)


def _get_xyz_timeseries(target_id, start, stop, step, frame):
    coord0 = get_horizons_coord(target_id, {"start": start, "stop": stop, "step": step})

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

        for axis_name, vec, colr in [
            ("+X (Sunward)", (axis_len, 0.0, 0.0), "rgba(0,0,0,0.6)"),
            ("+Y (Dusk)", (0.0, axis_len, 0.0), "rgba(0,0,0,0.6)"),
            ("+Z (North)", (0.0, 0.0, axis_len), "rgba(0,0,0,0.6)"),
        ]:
            vx, vy, vz = vec
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, vx], y=[0.0, vy], z=[0.0, vz],
                    mode="lines+text",
                    line=dict(width=5, color=colr), opacity=0.65,
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
                xaxis=dict(range=[-lim, lim], showspikes=False, showbackground=False),
                yaxis=dict(range=[-lim, lim], showspikes=False, showbackground=False),
                zaxis=dict(range=[-lim, lim], showspikes=False, showbackground=False),
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
    if dperp_au.size == 0:
        return 1.0, 0.0, 0.0, 0.0

    tau_h = ((dpar_abs_au * u.AU) / (vsw_kms * u.km / u.s)).to_value(u.hour)
    max_dperp = float(np.max(dperp_au))
    max_dpar = float(np.max(dpar_abs_au))
    max_tau = float(np.max(tau_h))

    exponent = - (max_dperp / lperp_au) ** 2 - (max_tau / tau0_h) ** 2
    p_all = float(np.clip(np.exp(exponent), 1e-12, 1.0))
    return p_all, max_dperp, max_dpar, max_tau


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

    l_adv_au = ((float(vsw_kms) * u.km / u.s) * (float(window_hours) * u.hour)).to_value(u.AU)
    tau0_h = float(lag_tolerance) * float(window_hours)
    l_perp_au = float(perp_scale) * float(l_adv_au)

    T = len(time_index)
    N = len(targets)
    xyz = np.empty((T, N, 3), dtype=float)
    for j, name in enumerate(targets):
        df = tracks[name].loc[time_index]
        xyz[:, j, 0] = df["x_au"].to_numpy(dtype=float)
        xyz[:, j, 1] = df["y_au"].to_numpy(dtype=float)
        xyz[:, j, 2] = df["z_au"].to_numpy(dtype=float)

    lo_vsw = max(float(vsw_kms) * (1.0 - float(vsw_rel_unc)), 1e-3)
    hi_vsw = float(vsw_kms) * (1.0 + float(vsw_rel_unc))
    lo_lperp = max(l_perp_au * (1.0 - float(decorrelation_rel_unc)), 1e-9)
    hi_lperp = l_perp_au * (1.0 + float(decorrelation_rel_unc))
    lo_tau0 = max(tau0_h * (1.0 - float(decorrelation_rel_unc)), 1e-9)
    hi_tau0 = tau0_h * (1.0 + float(decorrelation_rel_unc))

    rows = []
    for k, ts in enumerate(time_index):
        dperp, dpar_abs, dpar_signed = _pairwise_separations(xyz[k], flow_hat)
        if dperp.size == 0:
            continue

        p_pair, tau_h = _pair_prob_same_stream(dperp, dpar_abs, vsw_kms=float(vsw_kms), lperp_au=l_perp_au, tau0_h=tau0_h)
        p_all_ref, max_dperp, max_dpar, max_tau_h = _all_prob_same_stream_worst_pair(
            dperp, dpar_abs, vsw_kms=float(vsw_kms), lperp_au=l_perp_au, tau0_h=tau0_h
        )
        p_all_lo, _, _, _ = _all_prob_same_stream_worst_pair(dperp, dpar_abs, vsw_kms=lo_vsw, lperp_au=lo_lperp, tau0_h=lo_tau0)
        p_all_hi, _, _, _ = _all_prob_same_stream_worst_pair(dperp, dpar_abs, vsw_kms=hi_vsw, lperp_au=hi_lperp, tau0_h=hi_tau0)

        pair_iqr = float(np.percentile(p_pair, 75) - np.percentile(p_pair, 25))

        rows.append(
            {
                "time": ts,
                "median_perp_au": float(np.median(dperp)),
                "median_par_au": float(np.median(dpar_abs)),
                "median_tau_h": float(np.median(tau_h)),
                "median_pair_probability": float(np.median(p_pair)),
                "all_pair_p_ref": float(p_all_ref),
                "all_pair_p_ref_p16": float(min(p_all_lo, p_all_hi)),
                "all_pair_p_ref_p84": float(max(p_all_lo, p_all_hi)),
                "pairwise_prob_iqr": pair_iqr,
                "p75_pair_probability": float(np.percentile(p_pair, 75)),
                "max_perp_au": float(max_dperp),
                "max_par_au": float(max_dpar),
                "max_tau_h": float(max_tau_h),
            }
        )

    metric_df = pd.DataFrame(rows).set_index("time").sort_index()
    if len(metric_df) == 0:
        raise RuntimeError("No metric samples were produced.")

    w = pd.Timedelta(hours=float(window_hours))
    t0 = metric_df.index[0]
    win_id = ((metric_df.index - t0) // w).astype(int)
    metric_df["_win_id"] = win_id

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

        p_ref = g["all_pair_p_ref"].to_numpy()
        p_ref_p16 = g["all_pair_p_ref_p16"].to_numpy()
        p_ref_p84 = g["all_pair_p_ref_p84"].to_numpy()

        same_stream_prob = float(np.median(p_ref))
        pair_iqr_med = float(np.median(g["pairwise_prob_iqr"].to_numpy()))

        metric = float(np.percentile(p_ref, 84) + float(along_weight) * pair_iqr_med)
        j_p16 = float(np.percentile(p_ref_p16, 84) + float(along_weight) * pair_iqr_med)
        j_p84 = float(np.percentile(p_ref_p84, 84) + float(along_weight) * pair_iqr_med)

        score_rows.append(
            {
                "window_start": ws,
                "window_end": we,
                "n_samples": int(len(g)),
                "coverage": coverage,
                "median_perp_au": float(g["median_perp_au"].median()),
                "median_par_au": float(g["median_par_au"].median()),
                "median_tau_h": float(g["median_tau_h"].median()),
                "same_flow_score": float(1.0 - same_stream_prob),
                "same_flow_score_p16": float(1.0 - np.percentile(p_ref, 84)),
                "same_flow_score_p84": float(1.0 - np.percentile(p_ref, 16)),
                "same_stream_prob": same_stream_prob,
                "same_stream_prob_p16": float(np.percentile(p_ref_p16, 16)),
                "same_stream_prob_p84": float(np.percentile(p_ref_p84, 84)),
                "max_perp_au": float(g["max_perp_au"].median()),
                "max_par_au": float(g["max_par_au"].median()),
                "max_tau_h": float(g["max_tau_h"].median()),
                "adv_length_au": float(l_adv_au),
                "perp_length_au": float(l_perp_au),
                "tau0_h": float(tau0_h),
                "vsw_kms": float(vsw_kms),
                "vsw_rel_unc": float(vsw_rel_unc),
                "decorrelation_rel_unc": float(decorrelation_rel_unc),
                "alignment_metric": metric,
                "alignment_metric_p16": j_p16,
                "alignment_metric_p84": j_p84,
            }
        )

    scores = pd.DataFrame(score_rows).sort_values("alignment_metric", ascending=True).reset_index(drop=True)
    if len(scores) == 0:
        raise RuntimeError("No valid windows found. Try reducing min_coverage or window_hours.")

    best = scores.head(int(top_n)).copy()

    if verbose:
        print("[alignment] Stream-separation model (GSE):")
        print("  - Uses |d_parallel| for lags (fixes signed-max bug).")
        print(f"  - L_adv={l_adv_au:.4f} AU ({l_adv_au * AU_IN_RE:.1f} Re), "
              f"L_perp={l_perp_au:.4f} AU ({l_perp_au * AU_IN_RE:.1f} Re), tau0={tau0_h:.2f} h.")

    metric_df = metric_df.drop(columns=["_win_id"])

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
    width=1250,
    height=820,
):
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

    xyz_re = xyz * AU_IN_RE

    if summary_row is not None:
        vsw_kms = float(summary_row.get("vsw_kms", 400.0))
        lperp_au = float(summary_row.get("perp_length_au", np.nan))
        tau0_h = float(summary_row.get("tau0_h", np.nan))
        vsw_rel_unc = float(summary_row.get("vsw_rel_unc", 0.15))
        decor_rel_unc = float(summary_row.get("decorrelation_rel_unc", 0.25))
    else:
        vsw_kms = 400.0
        vsw_rel_unc = 0.15
        decor_rel_unc = 0.25
        tau0_h = 0.5 * (w1 - w0).total_seconds() / 3600.0
        lperp_au = 0.35 * ((vsw_kms * u.km / u.s) * (tau0_h * u.hour)).to_value(u.AU)

    if not np.isfinite(tau0_h) or tau0_h <= 0:
        tau0_h = 0.5 * (w1 - w0).total_seconds() / 3600.0
    if not np.isfinite(lperp_au) or lperp_au <= 0:
        lperp_au = 0.35 * ((vsw_kms * u.km / u.s) * (tau0_h * u.hour)).to_value(u.AU)

    lo_vsw = max(vsw_kms * (1.0 - vsw_rel_unc), 1e-3)
    hi_vsw = vsw_kms * (1.0 + vsw_rel_unc)
    lo_lperp = max(lperp_au * (1.0 - decor_rel_unc), 1e-9)
    hi_lperp = lperp_au * (1.0 + decor_rel_unc)
    lo_tau0 = max(tau0_h * (1.0 - decor_rel_unc), 1e-9)
    hi_tau0 = tau0_h * (1.0 + decor_rel_unc)

    max_dperp_re = np.empty(T, dtype=float)
    max_dpar_re = np.empty(T, dtype=float)
    max_tau_h = np.empty(T, dtype=float)
    p_all = np.empty(T, dtype=float)
    p16 = np.empty(T, dtype=float)
    p84 = np.empty(T, dtype=float)

    for k in range(T):
        dperp, dpar_abs, _ = _pairwise_separations(xyz[k], flow_hat)
        p0, mdp, mdpar, mtau = _all_prob_same_stream_worst_pair(dperp, dpar_abs, vsw_kms, lperp_au, tau0_h)
        plo, _, _, _ = _all_prob_same_stream_worst_pair(dperp, dpar_abs, lo_vsw, lo_lperp, lo_tau0)
        phi, _, _, _ = _all_prob_same_stream_worst_pair(dperp, dpar_abs, hi_vsw, hi_lperp, hi_tau0)

        max_dperp_re[k] = mdp * AU_IN_RE
        max_dpar_re[k] = mdpar * AU_IN_RE
        max_tau_h[k] = mtau
        p_all[k] = p0
        p16[k] = min(plo, phi)
        p84[k] = max(plo, phi)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
        subplot_titles=(
            "3D GSE trajectories (Re)",
            "Worst-pair cross-flow separation",
            "Worst-pair along-flow separation and lag",
            "All-spacecraft same-stream probability (worst-pair model)",
        ),
    )

    for i, name in enumerate(targets):
        c = colors[i % len(colors)]
        seg = tracks[name].loc[idx]
        xr = seg["x_au"].to_numpy(dtype=float) * AU_IN_RE
        yr = seg["y_au"].to_numpy(dtype=float) * AU_IN_RE
        zr = seg["z_au"].to_numpy(dtype=float) * AU_IN_RE
        fig.add_trace(
            go.Scatter3d(
                x=xr, y=yr, z=zr,
                mode="lines+markers",
                marker=dict(size=3, color=c),
                line=dict(width=5, color=c),
                name=name,
            ),
            row=1, col=1
        )

    center = xyz_re.reshape(-1, 3).mean(axis=0)
    extent = np.linalg.norm(xyz_re.reshape(-1, 3) - center, axis=1)
    arrow_len = max(float(np.nanmax(extent)), 10.0)
    flow_hat = np.asarray(flow_hat, dtype=float)
    flow_hat = flow_hat / np.linalg.norm(flow_hat)
    p0 = center
    p1 = center + flow_hat * arrow_len

    fig.add_trace(
        go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(color="black", width=8, dash="dash"),
            name="Assumed flow",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=idx, y=max_dperp_re, mode="lines+markers", showlegend=False),
        row=1, col=2
    )
    fig.update_yaxes(title_text="max d⊥ [Re]", row=1, col=2)

    fig.add_trace(
        go.Scatter(x=idx, y=max_dpar_re, mode="lines+markers", showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=idx, y=max_tau_h, mode="lines", line=dict(dash="dot"), showlegend=False),
        row=2, col=2
    )
    fig.update_yaxes(title_text="max |d∥| [Re] and max lag [h]", row=2, col=2)

    fig.add_trace(
        go.Scatter(x=idx, y=p84, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=idx, y=p16, mode="lines", fill="tonexty", line=dict(width=0), showlegend=False),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=idx, y=p_all, mode="lines+markers", showlegend=False),
        row=3, col=2
    )
    fig.update_yaxes(title_text="p_all(t)", range=[0, 1.02], row=3, col=2)

    title = f"GSE window: {w0} to {w1}"
    if summary_row is not None and "alignment_metric" in summary_row:
        title += (
            f"<br><sup>J={summary_row['alignment_metric']:.4f} "
            f"[{summary_row['alignment_metric_p16']:.4f}, {summary_row['alignment_metric_p84']:.4f}], "
            f"P(all same)={summary_row['same_stream_prob']:.3f} "
            f"[{summary_row['same_stream_prob_p16']:.3f}, {summary_row['same_stream_prob_p84']:.3f}]</sup>"
        )

    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        scene=dict(xaxis_title="X [Re]", yaxis_title="Y [Re]", zaxis_title="Z [Re]", aspectmode="data"),
        margin=dict(l=10, r=10, t=90, b=10),
    )

    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_xaxes(title_text="time", row=2, col=2)
    fig.update_xaxes(title_text="time", row=3, col=2)

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