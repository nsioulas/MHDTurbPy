import astropy.units as u
import numpy as np
import plotly.graph_objects as go
import astropy.constants as const

from .horizons_sun_lonlat import ballistic_source_longitude


def delta_long(r, rss, vsw, omega_deg_per_day=14.1844):
    sun_rot = (omega_deg_per_day * u.deg) / (24 * 3600 * u.s)
    return sun_rot * (r - rss) / vsw


def ballistic_map(carr_coord, rss, vsw, omega_deg_per_day=14.1844):
    r = carr_coord.spherical.distance
    ss_skycoord = carr_coord.lon + delta_long(r, rss, vsw, omega_deg_per_day=omega_deg_per_day)
    ss_lon = np.mod(ss_skycoord.to_value(u.deg), 360.0)
    return ss_lon






def ballistic_source_longitude_from_df(df, lon_col="Carr_lon", r_col="Radius", vsw_col="Vr", **kwargs):
    """Convenience wrapper for dataframe-based ballistic source-longitude mapping.

    Expects longitude in deg, radius in km or AU, and speed in km/s.
    If `r_col` max exceeds 10, values are treated as km and converted to AU.
    """
    lon = df[lon_col]
    r = df[r_col]
    r_au = r / 1.495978707e8 if float(np.nanmax(np.abs(r))) > 10 else r
    vsw = df[vsw_col]
    return ballistic_source_longitude(lon_carr_deg=lon, r_au=r_au, vsw_kms=vsw, **kwargs)

def wrap(arr, difference=340):
    arr = np.array(arr, dtype=float, copy=True)
    diff = np.diff(arr)
    idx = np.argwhere(np.abs(diff) > difference).ravel()
    arr[idx + 1] = np.nan
    return arr


def add_sc(
    fig,
    df,
    label,
    colour,
    line_dash="solid",
    marker_symbol="circle",
    show_by_default=True,
):
    is_visible = True if show_by_default else "legendonly"

    radii_rs = df["Radius"].values / const.R_sun.to(u.km).value
    text_list = [f"{value:.0f}" for value in radii_rs]

    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df["Radius"].values / u.au.to(u.km),
            name=label,
            text=text_list,
            showlegend=True,
            visible=is_visible,
            legendgroup=label,
            mode="lines+markers",
            hovertemplate="R: %{y:.4f} AU<br>%{text} Rs",
            line=dict(color=colour, width=2, dash=line_dash),
            marker=dict(color=colour, size=5, symbol=marker_symbol),
        ),
        row=1,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df["Carr_lat"].values,
            showlegend=False,
            visible=is_visible,
            legendgroup=label,
            mode="lines+markers",
            hovertemplate="%{y:.2f} deg<br>",
            line=dict(color=colour, width=2, dash=line_dash),
            marker=dict(color=colour, size=4, symbol=marker_symbol),
        ),
        row=2,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=wrap(df["Carr_lon"].values),
            showlegend=False,
            visible=is_visible,
            legendgroup=label,
            mode="lines+markers",
            hovertemplate="%{y:.1f} deg<br>",
            line=dict(color=colour, width=2, dash=line_dash),
            marker=dict(color=colour, size=4, symbol=marker_symbol),
        ),
        row=3,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=wrap(df["Mapped_300"].values.copy(), 180),
            showlegend=False,
            visible=is_visible,
            legendgroup=label,
            mode="lines",
            hovertemplate="%{y:.1f} deg<br>",
            line=dict(color=colour, width=2, dash=line_dash),
        ),
        row=4,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=wrap(df["Mapped_700"].values.copy(), 180),
            showlegend=False,
            visible=is_visible,
            legendgroup=label,
            mode="lines",
            hovertemplate="%{y:.1f} deg<br>",
            line=dict(color=colour, width=2, dash="dot" if line_dash == "solid" else line_dash),
        ),
        row=4,
        col=1,
    )

    return fig