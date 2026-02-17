from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import requests
import numpy as np


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
    "SWIFO-1": "-231",
    "SOLAR-1": "-231",
    "SOLAR 1": "-231",
    "DSCOVR": "-78",
    "DISCOVER": "-78",
    "DISCOVR": "-78",
    "ADITYA": "-156",
    "ADITYA-L1": "-156",
    "ADITYA L1": "-156",
    "SOHO": "-21",
    "PSP": "-96",
    "PARKER SOLAR PROBE": "-96",
    "SOLAR ORBITER": "-144",
    "SOLO": "-144",
}


def canonicalize_spacecraft_target(target: str) -> str:
    """Normalize common spacecraft aliases/typos to canonical mission labels."""
    s = str(target).strip()
    if re.fullmatch(r"-?\d+", s):
        return s

    cleaned = " ".join(s.upper().replace("_", " ").split())
    compact = cleaned.replace(" ", "").replace("-", "")

    alias_map = {
        "ACE": "ACE",
        "WIND": "WIND",
        "IMAP": "IMAP",
        "SWFOL1": "SWFO-L1",
        "SWIFOL1": "SWIFO-1",
        "SOLAR1": "SOLAR-1",
        "DSCOVR": "DSCOVR",
        "DISCOVR": "DSCOVR",
        "DISCOVER": "DSCOVR",
        "ADITYA": "ADITYA-L1",
        "ADITYAL1": "ADITYA-L1",
        "AIDTYA": "ADITYA-L1",
        "AIDTYAL1": "ADITYA-L1",
        "SOHO": "SOHO",
        "PSP": "PSP",
        "PARKERSOLARPROBE": "PSP",
        "SOLARORBITER": "SOLO",
        "SOLO": "SOLO",
    }
    return alias_map.get(compact, cleaned)


def validate_time_window(start: str, stop: str) -> tuple[str, str]:
    """Validate and normalize ISO-like start/stop strings for Horizons calls."""

    def _parse_one(label: str, value: str) -> pd.Timestamp:
        txt = str(value).strip()
        year_token = txt.split("-", 1)[0]
        if year_token.isdigit() and len(year_token) != 4:
            raise ValueError(
                f"{label} must begin with a 4-digit year (got {value!r}). "
                "Example: '2025-12-10T00:00:00'."
            )
        try:
            return pd.to_datetime(txt, utc=False)
        except Exception as exc:
            raise ValueError(
                f"Invalid {label} datetime {value!r}. Use ISO-like format, "
                "e.g. '2025-10-01T00:00:00'."
            ) from exc

    t_start = _parse_one("start", start)
    t_stop = _parse_one("stop", stop)
    if t_stop <= t_start:
        raise ValueError(f"stop must be later than start (start={start!r}, stop={stop!r}).")

    return t_start.isoformat(), t_stop.isoformat()


def _require_sunpy():
    try:
        import sunpy  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sunpy\n"
            "Install with: pip install sunpy astropy numpy pandas"
        ) from e


def resolve_spacecraft_spkid(target: str) -> str:
    s_raw = str(target).strip()

    if re.fullmatch(r"-?\d+", s_raw):
        return s_raw

    s = canonicalize_spacecraft_target(s_raw)
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
    s_raw_low = s_raw.lower()
    for item in results:
        name = str(item.get("name", "")).strip().lower()
        if name == s_low or name == s_raw_low:
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
    """
    Horizons trajectory timeseries in HGS (and optionally HGC), plus HEE Cartesian.

    FIX: if carrington=True, the HGS->HGC transform requires observer != None in newer SunPy.
    We set observer to Earth's HGS position at each obstime.
    """
    _require_sunpy()

    import astropy.units as u
    from sunpy.coordinates import get_horizons_coord
    from sunpy.coordinates import get_body_heliographic_stonyhurst
    from sunpy.coordinates.frames import (
        HeliographicCarrington,
        HeliographicStonyhurst,
        HeliocentricEarthEcliptic,
    )

    start, stop = validate_time_window(start, stop)
    canonical_target = canonicalize_spacecraft_target(target)
    spkid = resolve_spacecraft_spkid(canonical_target)

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
        earth_obs = get_body_heliographic_stonyhurst("earth", coord_hgs.obstime)
        coord_hgc = coord_hgs.transform_to(
            HeliographicCarrington(obstime=coord_hgs.obstime, observer=earth_obs)
        )
        out["hgc_lon_deg"] = np.mod(coord_hgc.lon.to_value(u.deg), 360.0)
        out["hgc_lat_deg"] = coord_hgc.lat.to_value(u.deg)
        out["hgc_r_au"] = _dist_to_au(coord_hgc)

    return TrajResult(target=str(canonical_target), spkid=str(spkid), df=out)


def ballistic_source_longitude(
    lon_carr_deg,
    r_au,
    vsw_kms,
    r_ss_rsun: float = 2.5,
    omega_deg_per_day: float = 14.1844,
    vsw_fallback_kms: float = 400.0,
):
    """Vectorized ballistic mapping from spacecraft Carrington longitude to source-surface longitude.

    Parameters are array-like and interpreted in degrees/AU/km/s.
    Returns `(phi_src_deg_wrapped, tau_days, fallback_mask)`.
    """
    lon = pd.Series(lon_carr_deg, copy=False, dtype=float)
    r = pd.Series(r_au, copy=False, dtype=float)
    vsw = pd.Series(vsw_kms, copy=False, dtype=float)

    fallback = (~np.isfinite(vsw)) | (vsw <= 0)
    vsw_eff = vsw.copy()
    vsw_eff[fallback] = float(vsw_fallback_kms)

    rsun_au = 0.00465047
    r_ss_au = float(r_ss_rsun) * rsun_au
    km_per_au = 1.495978707e8
    tau_days = (r - r_ss_au) * km_per_au / (vsw_eff * 86400.0)

    phi_src = np.mod(lon - float(omega_deg_per_day) * tau_days, 360.0)
    return phi_src, tau_days, fallback


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--step", default="60m")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--coord", choices=["HGS", "HGS+HGC"], default="HGS")
    p.add_argument("--outdir", default="out")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    carr = (args.coord == "HGS+HGC")
    for tgt in args.targets:
        tr = get_lonlat_xyz_timeseries(tgt, args.start, args.stop, args.step, carrington=carr)
        csv = outdir / f"{tr.target.replace(' ', '_')}_{tr.spkid}.csv"
        tr.df.to_csv(csv)
        print(f"Wrote {csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
