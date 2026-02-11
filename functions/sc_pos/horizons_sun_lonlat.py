from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import requests


@dataclass(frozen=True)
class TrajResult:
    target: str
    spkid: str
    df: pd.DataFrame


HORIZONS_LOOKUP_URL = "https://ssd.jpl.nasa.gov/api/horizons_lookup.api"

KNOWN_SPKID: Dict[str, str] = {
    "ACE": "-92",
    "WIND": "-8",
    "IMAP": "-43",
    "SWFO-L1": "-231",
    "SOLAR-1": "-231",
    "DSCOVR": "-78",
    "SOHO": "-21",
    "PSP": "-96",
    "PARKER SOLAR PROBE": "-96",
    "SOLAR ORBITER": "-144",
    "SOLO": "-144",
}


def _require_sunpy():
    try:
        import sunpy  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sunpy\n"
            "Install with: pip install sunpy astropy numpy pandas"
        ) from e


def resolve_spacecraft_spkid(target: str) -> str:
    s = str(target).strip()

    if re.fullmatch(r"-?\d+", s):
        return s

    key = s.upper()
    if key in KNOWN_SPKID:
        return KNOWN_SPKID[key]

    r = requests.get(
        HORIZONS_LOOKUP_URL,
        params={"sstr": s, "group": "sct", "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()

    count = int(payload.get("count", 0))
    if count == 0:
        raise ValueError(f"No spacecraft match found for {s!r} via Horizons lookup.")

    results = payload.get("result", [])
    if not results:
        raise ValueError(f"Horizons lookup returned count={count} but empty result for {s!r}.")

    if len(results) == 1:
        return str(results[0]["spkid"])

    s_low = s.lower()
    for item in results:
        name = str(item.get("name", "")).strip().lower()
        if name == s_low:
            return str(item["spkid"])

    msg_lines = [f"Ambiguous spacecraft name {s!r}. Pick one SPKID and use that as --targets <ID>:"]
    for item in results[:30]:
        msg_lines.append(f"  {item.get('spkid')}  {item.get('name')}  {item.get('designation','')}")
    raise ValueError("\n".join(msg_lines))


def get_lonlat_xyz_timeseries(
    target: str,
    start: str,
    stop: str,
    step: str,
    carrington: bool = False,
) -> TrajResult:
    _require_sunpy()

    import astropy.units as u
    from sunpy.coordinates import get_horizons_coord
    from sunpy.coordinates.frames import (
        HeliographicCarrington,
        HeliographicStonyhurst,
        HeliocentricEarthEcliptic,
    )

    spkid = resolve_spacecraft_spkid(target)

    coord0 = get_horizons_coord(
        spkid,
        {"start": start, "stop": stop, "step": step},
    )

    coord_hgs = coord0.transform_to(HeliographicStonyhurst(obstime=coord0.obstime))
    coord_hee = coord0.transform_to(HeliocentricEarthEcliptic(obstime=coord0.obstime))

    def _dist_to_au(c):
        if hasattr(c, "radius"):
            return c.radius.to_value(u.AU)
        if hasattr(c, "distance"):
            return c.distance.to_value(u.AU)
        return c.spherical.distance.to_value(u.AU)

    out = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(coord_hgs.obstime.datetime64),
            "hgs_lon_deg": coord_hgs.lon.to_value(u.deg),
            "hgs_lat_deg": coord_hgs.lat.to_value(u.deg),
            "hgs_r_au": _dist_to_au(coord_hgs),
            "hee_lon_deg": coord_hee.lon.to_value(u.deg),
            "hee_lat_deg": coord_hee.lat.to_value(u.deg),
            "hee_r_au": _dist_to_au(coord_hee),
            "hee_x_au": coord_hee.cartesian.x.to_value(u.AU),
            "hee_y_au": coord_hee.cartesian.y.to_value(u.AU),
            "hee_z_au": coord_hee.cartesian.z.to_value(u.AU),
        }
    ).set_index("time_utc")

    if carrington:
        coord_hgc = coord_hgs.transform_to(HeliographicCarrington(obstime=coord_hgs.obstime))
        out["hgc_lon_deg"] = coord_hgc.lon.to_value(u.deg)
        out["hgc_lat_deg"] = coord_hgc.lat.to_value(u.deg)
        out["hgc_r_au"] = _dist_to_au(coord_hgc)

    return TrajResult(target=str(target), spkid=str(spkid), df=out)



def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--step", default="60m")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--coord", choices=["HGS", "HGS+HGC"], default="HGS")
    p.add_argument("--outdir", default="out")
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    want_carr = args.coord == "HGS+HGC"

    frames = []
    for t in args.targets:
        res = get_lonlat_xyz_timeseries(t, args.start, args.stop, args.step, carrington=want_carr)
        df = res.df.copy()
        df.insert(0, "target", res.target)
        df.insert(1, "spkid", res.spkid)
        frames.append(df)
        df.to_csv(outdir / f"{res.target.replace(' ', '_')}_lonlat_xyz.csv")

    combined = pd.concat(frames).reset_index().rename(columns={"index": "time_utc"}).set_index(["time_utc", "target"])
    if args.csv is not None:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.csv)

    print(f"Wrote per-target CSVs in: {outdir}")
    if args.csv is not None:
        print(f"Wrote combined CSV: {args.csv}")
    return 0


def get_repo_style_orbit_df(
    target: str,
    start: str,
    stop: str,
    step: str,
    rss_rsun: float = 2.5,
    vsw_low_kms: float = 300.0,
    vsw_high_kms: float = 700.0,
    omega_deg_per_day: float = 14.1844,
):
    _require_sunpy()

    import numpy as np
    import pandas as pd
    import astropy.units as u
    import astropy.constants as const
    from sunpy.coordinates import get_horizons_coord, get_body_heliographic_stonyhurst
    from sunpy.coordinates.frames import HeliographicStonyhurst, HeliographicCarrington

    try:
        from . import helpers as helpers_mod
    except Exception:
        import helpers as helpers_mod

    spkid = resolve_spacecraft_spkid(target)

    coord0 = get_horizons_coord(
        spkid,
        {"start": start, "stop": stop, "step": step},
    )

    hgs = coord0.transform_to(HeliographicStonyhurst(obstime=coord0.obstime))

    earth_obs = get_body_heliographic_stonyhurst("earth", hgs.obstime)
    carr = hgs.transform_to(HeliographicCarrington(obstime=hgs.obstime, observer=earth_obs))

    r_km = carr.spherical.distance.to_value(u.km)
    lon_deg = np.mod(carr.lon.to_value(u.deg), 360.0)
    lat_deg = carr.lat.to_value(u.deg)

    rss = (rss_rsun * const.R_sun).to(u.km)
    vsw_low = (vsw_low_kms * u.km / u.s)
    vsw_high = (vsw_high_kms * u.km / u.s)

    mapped_300 = helpers_mod.ballistic_map(carr, rss=rss, vsw=vsw_low, omega_deg_per_day=omega_deg_per_day)
    mapped_700 = helpers_mod.ballistic_map(carr, rss=rss, vsw=vsw_high, omega_deg_per_day=omega_deg_per_day)

    df = pd.DataFrame(
        {
            "Radius": r_km,
            "Carr_lat": lat_deg,
            "Carr_lon": lon_deg,
            "Mapped_300": mapped_300,
            "Mapped_700": mapped_700,
        },
        index=pd.to_datetime(carr.obstime.datetime64),
    )
    df.index.name = "time_utc"

    return TrajResult(target=str(target), spkid=str(spkid), df=df)


if __name__ == "__main__":
    raise SystemExit(main())
