"""sc_pos.backmap.mapping

Backmapping from spacecraft Carrington coordinates to a source surface.

Frame and sign conventions (explicit)
------------------------------------
Inputs ``phi_sc`` and ``lat_sc`` are expected to be **heliographic Carrington**
coordinates of the spacecraft at the *measurement time* ``t``.

Let ``t0 = t - tau`` be the launch time at the source surface.
Assuming the plasma parcel propagates radially in an inertial frame,
its inertial longitude is approximately conserved along the trajectory.

Carrington longitude is a rotating-frame longitude. Under the above assumption,
	the Carrington longitude of the parcel at launch satisfies:

    phi_src(t0) = phi_sc(t) + Omega * tau(t)

where Omega is the Carrington rotation rate.

We therefore implement:

    phi_src = wrap_0_360( phi_sc + phi_sign * Omega * tau )

with ``phi_sign`` defaulting to +1. A negative sign is permitted only as a
*diagnostic* to catch convention mismatches; it is not recommended for
production.

Latitude handling
-----------------
This module implements the minimal, explicit baseline:

    lat_src = lat_sc

No latitudinal evolution is modeled.

Limitations
-----------
Ballistic mapping is physically indefensible when (for example) strong shear,
stream–stream interaction, or large non-radial flows dominate. We surface
these risks through metadata and masking in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import astropy.units as u
except Exception as e:  # pragma: no cover
    raise RuntimeError("MHDTurbPy backmapping requires astropy. Install with: pip install astropy") from e


from .circular import wrap_0_360


@dataclass(frozen=True)
class MappingResult:
    phi_src_deg: np.ndarray
    lat_src_deg: np.ndarray
    meta: Dict[str, Any]


def map_to_source_surface(
    *,
    phi_sc_deg: np.ndarray,
    lat_sc_deg: np.ndarray,
    tau: u.Quantity,
    omega: u.Quantity,
    delta_phi_deg: Optional[object] = None,
    phi_sign: int = +1,
    sign: Optional[int] = None,  # deprecated alias
) -> MappingResult:
    """Compute (phi_src, lat_src) for a given travel time tau.

    Parameters
    ----------
    phi_sc_deg, lat_sc_deg
        Spacecraft Carrington lon/lat at the measurement time (degrees).
    tau
        Travel time from source surface to spacecraft. Must be positive.
    omega
        Rotation rate. Must be convertible to angular speed (deg/s or rad/s).
    sign
        +1 implements the standard Carrington ballistic mapping.

    Returns
    -------
    MappingResult
    """

    s = int(phi_sign if sign is None else sign)
    if s not in (-1, +1):
        raise ValueError("phi_sign must be +1 or -1")

    tau_s = u.Quantity(tau).to(u.s).to_value(u.s)
    if np.nanmin(tau_s) <= 0:
        raise ValueError("tau must be strictly positive everywhere it is used.")

    omega_deg_s = u.Quantity(omega).to(u.deg / u.s).to_value(u.deg / u.s)

    phi = np.asarray(phi_sc_deg, dtype=float).reshape(-1)
    lat = np.asarray(lat_sc_deg, dtype=float).reshape(-1)

    if delta_phi_deg is None:
        dphi = omega_deg_s * tau_s
        dphi_def = 'Omega*tau'
    else:
        dq = u.Quantity(delta_phi_deg, u.deg).to_value(u.deg)
        dq = np.asarray(dq, dtype=float)
        if dq.size == 1:
            dphi = np.full_like(phi, float(dq.reshape(-1)[0]), dtype=float)
        else:
            if dq.shape[0] != phi.shape[0]:
                raise ValueError('delta_phi_deg must be scalar or have the same length as phi_sc_deg')
            dphi = dq.reshape(-1)
        dphi_def = 'provided'

    phi_src = wrap_0_360(phi + s * dphi)

    meta = {
        "frame": "HeliographicCarrington (observer explicit upstream)",
        "phi_src_def": "wrap_0_360(phi_sc + phi_sign*DeltaPhi)",
        "phi_sign": int(s),
        "omega": float(u.Quantity(omega).to_value(u.deg / u.day)),
        "delta_phi_def": str(dphi_def),
        "lat_src_def": "lat_sc (no latitudinal evolution modeled)",
    }
    return MappingResult(phi_src_deg=phi_src, lat_src_deg=lat.copy(), meta=meta)
