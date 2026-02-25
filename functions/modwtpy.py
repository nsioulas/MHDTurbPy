#!/usr/bin/env python3
"""
wavelet_3d_anis_pipeline_v5.py

This is the verified, benchmarked pipeline that addresses:

(1) Amplitude mismatch SF5 vs wavelet:
    - Not a missing coefficient; it is operator transfer-function mismatch.
    - Provides a *constant* overlay coefficient (preserves slopes) and tools to build
      spectrum-aware coefficients if desired.

(2) Higher-order moments + wavelet-derived kurtosis:
    - Computes M_p = <|δB|^p>_{bin} for any p.
    - Provides flatness F = M4 / M2^2 (kurtosis-like).

(3) New vs legacy wavelet pipelines:
    - NEW: δB_j := W_j, B_l := v^{(j)} (co-located cascade outputs).
    - LEGACY: δB_j := D_j (MRA reconstructed), B_l := A_j = S + Σ_{k>j} D_k (your Appr).
      Implemented with FFT kernels and a reconstruction sanity check.

(4) Benchmarks:
    - Times 5-point, new wavelet, and legacy wavelet pipelines.
    - Reports SF2 slopes and amplitude ratios vs 5-point (for the "all" bin).

Notes:
- Convolution uses modwtpy_fast.circular_convolve_d only if validated correct on (N,3).
  Otherwise it uses numpy roll accumulation (correct and fast for short L).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pywt

Array = np.ndarray


# =============================================================================
# Wavelet naming + scaled filters
# =============================================================================
@lru_cache(maxsize=64)
def normalize_wname_for_pywt(wname: str) -> str:
    w = str(wname).lower().strip()
    if w.startswith("la"):
        digits = "".join(ch for ch in w if ch.isdigit())
        if digits:
            L = int(digits)
            if L % 2 != 0:
                raise ValueError(f"Expected even LA length, got {wname}")
            return f"sym{L // 2}"
    if w.startswith("d"):
        digits = "".join(ch for ch in w if ch.isdigit())
        if digits:
            L = int(digits)
            if L % 2 == 0:
                return f"sym{L // 2}"
        return "db2"
    return w


@lru_cache(maxsize=64)
def dec_filters_scaled(wname: str) -> Tuple[Array, Array]:
    """
    MODWT cascade filters (same scaling as in your code):
      h_t = dec_hi / sqrt(2)
      g_t = dec_lo / sqrt(2)
    """
    wname_norm = normalize_wname_for_pywt(wname)
    wav = pywt.Wavelet(wname_norm)
    h_t = np.asarray(wav.dec_hi, dtype=float) / np.sqrt(2.0)
    g_t = np.asarray(wav.dec_lo, dtype=float) / np.sqrt(2.0)
    return h_t, g_t


def _as_2d(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("Expected 1D or 2D array")


# =============================================================================
# Circular dilated convolution backends
# =============================================================================
def _cconv_roll_minus(x: Array, f: Array, step: int) -> Array:
    """
    y[i,...] = Σ_t f[t] x[(i - t*step) mod N, ...]
    Implemented as y = Σ_t f[t] roll(x, t*step).
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    if x.ndim == 1:
        y = np.zeros_like(x)
        for t in range(f.size):
            y += f[t] * np.roll(x, t * step, axis=0)
        return y
    if x.ndim == 2:
        y = np.zeros_like(x)
        for t in range(f.size):
            y += f[t] * np.roll(x, t * step, axis=0)
        return y
    raise ValueError("x must be 1D or 2D (N,C)")


def _try_import_modwtpy_fast() -> Tuple[bool, Any]:
    try:
        import modwtpy_fast as mw  # type: ignore
        return True, mw
    except Exception:
        return False, None


_HAVE_MW, _MW = _try_import_modwtpy_fast()


def _mw_cconv_ok() -> bool:
    """
    Validate modwtpy_fast.circular_convolve_d for (N,3) arrays.
    """
    if not _HAVE_MW:
        return False
    rng = np.random.default_rng(0)
    N = 128
    x = rng.normal(size=(N, 3))
    f = rng.normal(size=8)
    j = 3
    step = 2 ** (j - 1)
    y_ref = _cconv_roll_minus(x, f, step)
    try:
        y_mw = _MW.circular_convolve_d(f, x, j)
    except Exception:
        return False
    err = float(np.max(np.abs(y_ref - y_mw)))
    return err < 1e-10


_MW_OK = _mw_cconv_ok()


def cconv_dilated_minus(x: Array, f: Array, j: int) -> Array:
    """
    y[i] = Σ_t f[t] x[(i - t*2^{j-1}) mod N]
    """
    step = 2 ** (j - 1)
    if _MW_OK:
        return _MW.circular_convolve_d(np.asarray(f, float), np.asarray(x, float), j)
    return _cconv_roll_minus(np.asarray(x, float), np.asarray(f, float), step)


# =============================================================================
# MODWT cascade
# =============================================================================
def modwt_cascade_multichannel(X: Array, wname: str, J: int) -> Tuple[Array, Array, Array]:
    """
    Returns:
      W:  (J,N,C) detail coefficients (bandpass)
      V:  (J,N,C) scaling outputs v^{(j)} (local mean at level j)
      vJ: (N,C)   final scaling output
    """
    if J < 1:
        raise ValueError("J must be >=1")
    h_t, g_t = dec_filters_scaled(wname)
    v = _as_2d(X)
    N, C = v.shape
    W = np.empty((J, N, C), dtype=float)
    V = np.empty((J, N, C), dtype=float)
    for j in range(1, J + 1):
        W[j - 1] = cconv_dilated_minus(v, h_t, j)
        v = cconv_dilated_minus(v, g_t, j)
        V[j - 1] = v
    return W, V, v


# =============================================================================
# Binning masks using cosine thresholds
# =============================================================================
@dataclass(frozen=True)
class AnisBins:
    theta_perp_deg: float = 80.0
    phi_perp_deg: float = 80.0
    theta_disp_deg: float = 80.0
    phi_disp_deg: float = 10.0
    theta_par_deg: float = 10.0
    phi_par_deg: float = 90.0


@lru_cache(maxsize=128)
def _cosd(x_deg: float) -> float:
    return float(np.cos(np.deg2rad(x_deg)))


def _safe_norm(v: Array, eps: float) -> Array:
    return np.maximum(np.linalg.norm(v, axis=-1), eps)


def compute_bin_masks_fast(
    B_l: Array,
    V_l: Array,
    dB: Array,
    bins: AnisBins,
    V_sc: Optional[Array] = None,
    eps: float = 1e-30,
    par_ignores_phi: bool = True,
) -> Dict[str, Array]:
    """
    Returns masks for {all, perp, disp, par}.
    """
    B_l = np.asarray(B_l, float)
    V_l = np.asarray(V_l, float)
    dB = np.asarray(dB, float)
    V_rel = V_l if V_sc is None else (V_l - np.asarray(V_sc, float))

    Vn = _safe_norm(V_rel, eps)
    Bn = _safe_norm(B_l, eps)
    cos_theta = np.sum(V_rel * B_l, axis=-1) / (Vn * Bn)

    b_hat = B_l / Bn[..., None]
    Vp = V_rel - np.sum(V_rel * b_hat, axis=-1, keepdims=True) * b_hat
    dBp = dB - np.sum(dB * b_hat, axis=-1, keepdims=True) * b_hat
    Vpn = _safe_norm(Vp, eps)
    dBpn = _safe_norm(dBp, eps)

    valid = np.isfinite(cos_theta) & (Vn > eps) & (Bn > eps) & (Vpn > eps) & (dBpn > eps)
    cos_phi = np.sum(Vp * dBp, axis=-1) / (Vpn * dBpn)
    valid = valid & np.isfinite(cos_phi)

    c_th_perp = _cosd(bins.theta_perp_deg)
    c_th_disp = _cosd(bins.theta_disp_deg)
    c_th_par = _cosd(bins.theta_par_deg)

    c_ph_perp = _cosd(bins.phi_perp_deg)
    c_ph_disp = _cosd(bins.phi_disp_deg)
    c_ph_par = _cosd(bins.phi_par_deg)

    m_all = valid
    m_perp = m_all & (cos_theta < c_th_perp) & (cos_phi < c_ph_perp)  # φ>φ0
    m_disp = m_all & (cos_theta < c_th_disp) & (cos_phi > c_ph_disp)  # φ<φ0
    if par_ignores_phi:
        m_par = m_all & (cos_theta > c_th_par)
    else:
        m_par = m_all & (cos_theta > c_th_par) & (cos_phi > c_ph_par)

    return {"all": m_all, "perp": m_perp, "disp": m_disp, "par": m_par}


# =============================================================================
# Band metrics for PSD conversion
# =============================================================================
@dataclass(frozen=True)
class WaveletScaleMap:
    tau: Array
    f_center: Array
    bandwidth: Array


def _dtft(filt: Array, omega: Array) -> Array:
    n = np.arange(filt.size, dtype=float)
    return np.sum(filt[None, :] * np.exp(-1j * omega[:, None] * n[None, :]), axis=1)


def modwt_band_metrics(wname: str, J: int, dt: float, n_omega: int = 8192) -> WaveletScaleMap:
    """
    Compute BW_j and f_center,j from cascade transfer H_j.
    """
    if J < 1:
        raise ValueError("J must be >=1")
    if dt <= 0:
        raise ValueError("dt must be >0")

    h_t, g_t = dec_filters_scaled(wname)
    omega = np.linspace(0.0, np.pi, int(n_omega), endpoint=True)
    domega = omega[1] - omega[0]
    f = omega / (2.0 * np.pi * dt)
    df = domega / (2.0 * np.pi * dt)

    def eval_scaled(filt: Array, scale: float) -> Array:
        om = np.mod(scale * omega, 2.0 * np.pi)
        return _dtft(filt, om)

    Hj = np.empty((J, omega.size), dtype=np.complex128)
    prodG = np.ones_like(omega, dtype=np.complex128)
    for j in range(1, J + 1):
        Hs = eval_scaled(h_t, 2.0 ** (j - 1))
        if j == 1:
            Hj[j - 1] = Hs
        else:
            Gs = eval_scaled(g_t, 2.0 ** (j - 2))
            prodG = prodG * Gs
            Hj[j - 1] = Hs * prodG

    P = np.abs(Hj) ** 2
    bandwidth = np.sum(P, axis=1) * df
    f_center = (np.sum(P * f[None, :], axis=1) * df) / np.maximum(bandwidth, 1e-300)
    tau = (2.0 ** np.arange(1, J + 1)) * dt
    return WaveletScaleMap(tau=tau, f_center=f_center, bandwidth=bandwidth)


# =============================================================================
# Five-point baseline operators
# =============================================================================
def five_point_increment_1d(x: Array, m: int) -> Array:
    x = np.asarray(x, float)
    return (np.roll(x, 2 * m) - 4 * np.roll(x, m) + 6 * x - 4 * np.roll(x, -m) + np.roll(x, -2 * m)) / 35.0


def five_point_mean_1d(x: Array, m: int) -> Array:
    x = np.asarray(x, float)
    return (np.roll(x, 2 * m) + 4 * np.roll(x, m) + 6 * x + 4 * np.roll(x, -m) + np.roll(x, -2 * m)) / 16.0


def five_point_increment_vec(B: Array, m: int) -> Array:
    B = np.asarray(B, float)
    out = np.empty_like(B)
    for c in range(B.shape[1]):
        out[:, c] = five_point_increment_1d(B[:, c], m)
    return out


def five_point_mean_vec(B: Array, m: int) -> Array:
    B = np.asarray(B, float)
    out = np.empty_like(B)
    for c in range(B.shape[1]):
        out[:, c] = five_point_mean_1d(B[:, c], m)
    return out


# =============================================================================
# Legacy MRA reconstruction (Det + Appr) via FFT
# =============================================================================
def _periodize_to_N(li: Array, N: int) -> Array:
    li = np.asarray(li, float)
    out = np.zeros(N, dtype=float)
    for i, val in enumerate(li):
        out[i % N] += val
    return out


@lru_cache(maxsize=64)
def _mra_kernels_periodized(wname: str, J: int, N: int) -> Tuple[Tuple[Array, ...], Array]:
    wname_norm = normalize_wname_for_pywt(wname)
    wav = pywt.Wavelet(wname_norm)
    h = np.asarray(wav.dec_hi, float)
    g = np.asarray(wav.dec_lo, float)

    def up_arrow(li: Array, j: int) -> Array:
        if j == 0:
            return np.array([1.0], dtype=float)
        li = np.asarray(li, float)
        step = 2 ** (j - 1)
        out = np.zeros(step * (li.size - 1) + 1, dtype=float)
        out[::step] = li
        return out

    h_kernels: List[Array] = []
    g_j_part = np.array([1.0], dtype=float)

    for j in range(J):
        g_j_up = up_arrow(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)

        h_j_up = up_arrow(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)

        h_j_t = h_j / (2.0 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2.0)

        h_kernels.append(_periodize_to_N(h_j_t, N))

    j = J - 1
    g_j_up = up_arrow(g, j + 1)
    g_J = np.convolve(g_j_part, g_j_up)
    g_J_t = g_J / (2.0 ** ((j + 1) / 2.0))
    g_kernel = _periodize_to_N(g_J_t, N)

    return tuple(h_kernels), g_kernel


@lru_cache(maxsize=64)
def _fft_kernels_for_plus(wname: str, J: int, N: int) -> Tuple[Tuple[Array, ...], Array]:
    h_kernels, g_kernel = _mra_kernels_periodized(wname, J, N)
    idx = (-np.arange(N)) % N  # kstd[k] = kplus[(-k) mod N]
    hk_fft: List[Array] = []
    for j in range(J):
        hk_fft.append(np.fft.rfft(h_kernels[j][idx]))
    g_fft = np.fft.rfft(g_kernel[idx])
    return tuple(hk_fft), g_fft


def _cconv_std_fft(X: Array, kstd_rfft: Array) -> Array:
    X = np.asarray(X, float)
    if X.ndim == 1:
        return np.fft.irfft(np.fft.rfft(X) * kstd_rfft, n=X.shape[0])
    if X.ndim == 2:
        FX = np.fft.rfft(X, axis=0)
        return np.fft.irfft(FX * kstd_rfft[:, None], n=X.shape[0], axis=0)
    raise ValueError("X must be 1D or 2D")


def legacy_mra_details_and_background(X: Array, wname: str, J: int) -> Tuple[Array, Array, Dict[str, float]]:
    X = _as_2d(X)
    N, C = X.shape

    W, _V, vJ = modwt_cascade_multichannel(X, wname=wname, J=J)
    hk_fft, g_fft = _fft_kernels_for_plus(wname, J, N)

    Det = np.empty_like(W)
    for j in range(J):
        Det[j] = _cconv_std_fft(W[j], hk_fft[j])

    S = _cconv_std_fft(vJ, g_fft)
    rec = S + np.sum(Det, axis=0)
    recon_maxabs = float(np.max(np.abs(X - rec)))

    Appr = np.empty_like(Det)
    tail = np.cumsum(Det[::-1], axis=0)[::-1]
    for j in range(J):
        Appr[j] = S if j == J - 1 else (S + tail[j + 1])

    return Appr, Det, {"recon_maxabs": recon_maxabs}


# =============================================================================
# Moments + flatness
# =============================================================================
def flatness_from_moments(M2: Array, M4: Array) -> Array:
    return M4 / np.maximum(M2 * M2, 1e-300)


def _moments_level(amp: Array, masks: Dict[str, Array], orders: Tuple[float, ...]) -> Tuple[Dict[str, Array], Dict[str, int]]:
    keys = ("all", "perp", "disp", "par")
    out = {k: np.full((len(orders),), np.nan, dtype=float) for k in keys}
    cnt = {k: 0 for k in keys}
    for k in keys:
        m = masks[k]
        cnt[k] = int(np.sum(m))
        if cnt[k] == 0:
            continue
        a = amp[m]
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        for kk, p in enumerate(orders):
            out[k][kk] = float(np.mean(a ** p))
    return out, cnt


@dataclass(frozen=True)
class AnisMomentResult:
    tau: Array
    f_center: Optional[Array]
    moments: Dict[str, Array]     # key -> (J,n_orders)
    counts: Dict[str, Array]      # key -> (J,)
    flatness: Optional[Dict[str, Array]]
    meta: Dict[str, Any]


def anisotropic_fivepoint(
    B: Array,
    V: Array,
    dt: float,
    J: int,
    orders: Sequence[float],
    bins: Optional[AnisBins] = None,
    V_sc: Optional[Array] = None,
    par_ignores_phi: bool = True,
    eps: float = 1e-30,
) -> AnisMomentResult:
    if bins is None:
        bins = AnisBins()
    B = _as_2d(B)
    V = _as_2d(V)
    if B.shape != V.shape or B.shape[1] != 3:
        raise ValueError("B and V must both be (N,3)")

    orders_t = tuple(float(p) for p in orders)
    keys = ("all", "perp", "disp", "par")
    M = {k: np.full((J, len(orders_t)), np.nan, dtype=float) for k in keys}
    C = {k: np.zeros(J, dtype=int) for k in keys}
    tau = (2.0 ** np.arange(1, J + 1)) * dt

    for j in range(J):
        m = int(2 ** (j + 1))
        Bl = five_point_mean_vec(B, m)
        Vl = five_point_mean_vec(V, m)
        dB = five_point_increment_vec(B, m)
        masks = compute_bin_masks_fast(Bl, Vl, dB, bins=bins, V_sc=V_sc, eps=eps, par_ignores_phi=par_ignores_phi)
        amp = np.linalg.norm(dB, axis=-1)
        out, cnt = _moments_level(amp, masks, orders_t)
        for k in keys:
            M[k][j, :] = out[k]
            C[k][j] = cnt[k]

    flat = None
    if 2.0 in orders_t and 4.0 in orders_t:
        i2 = orders_t.index(2.0)
        i4 = orders_t.index(4.0)
        flat = {k: flatness_from_moments(M[k][:, i2], M[k][:, i4]) for k in keys}

    return AnisMomentResult(tau=tau, f_center=None, moments=M, counts=C, flatness=flat, meta={"operator": "5pt"})


def anisotropic_wavelet_new(
    B: Array,
    V: Array,
    wname: str,
    dt: float,
    J: int,
    orders: Sequence[float],
    bins: Optional[AnisBins] = None,
    V_sc: Optional[Array] = None,
    trim_edges: bool = True,
    par_ignores_phi: bool = True,
    eps: float = 1e-30,
) -> AnisMomentResult:
    if bins is None:
        bins = AnisBins()
    B = _as_2d(B)
    V = _as_2d(V)
    if B.shape != V.shape or B.shape[1] != 3:
        raise ValueError("B and V must both be (N,3)")

    orders_t = tuple(float(p) for p in orders)
    keys = ("all", "perp", "disp", "par")
    M = {k: np.full((J, len(orders_t)), np.nan, dtype=float) for k in keys}
    C = {k: np.zeros(J, dtype=int) for k in keys}

    Wb, Vb, _ = modwt_cascade_multichannel(B, wname=wname, J=J)
    _, Vv, _ = modwt_cascade_multichannel(V, wname=wname, J=J)
    scale = modwt_band_metrics(wname=wname, J=J, dt=dt)

    h_t, _g_t = dec_filters_scaled(wname)
    L0 = int(h_t.size)

    for j in range(J):
        Bl = Vb[j]
        Vl = Vv[j]
        dB = Wb[j]
        masks = compute_bin_masks_fast(Bl, Vl, dB, bins=bins, V_sc=V_sc, eps=eps, par_ignores_phi=par_ignores_phi)

        if trim_edges:
            jj = j + 1
            Leff = int((2 ** jj - 1) * (L0 - 1))
            if Leff > 0 and 2 * Leff < Bl.shape[0]:
                edge = np.ones(Bl.shape[0], dtype=bool)
                edge[:Leff] = False
                edge[-Leff:] = False
                for k in keys:
                    masks[k] = masks[k] & edge

        amp = np.linalg.norm(dB, axis=-1)
        out, cnt = _moments_level(amp, masks, orders_t)
        for k in keys:
            M[k][j, :] = out[k]
            C[k][j] = cnt[k]

    flat = None
    if 2.0 in orders_t and 4.0 in orders_t:
        i2 = orders_t.index(2.0)
        i4 = orders_t.index(4.0)
        flat = {k: flatness_from_moments(M[k][:, i2], M[k][:, i4]) for k in keys}

    return AnisMomentResult(
        tau=scale.tau,
        f_center=scale.f_center,
        moments=M,
        counts=C,
        flatness=flat,
        meta={"operator": "wavelet_new", "mw_backend_used": bool(_MW_OK), "bandwidth": scale.bandwidth},
    )


def anisotropic_wavelet_legacy(
    B: Array,
    V: Array,
    wname: str,
    dt: float,
    J: int,
    orders: Sequence[float],
    bins: Optional[AnisBins] = None,
    V_sc: Optional[Array] = None,
    trim_edges: bool = True,
    par_ignores_phi: bool = True,
    eps: float = 1e-30,
) -> AnisMomentResult:
    if bins is None:
        bins = AnisBins()
    B = _as_2d(B)
    V = _as_2d(V)
    if B.shape != V.shape or B.shape[1] != 3:
        raise ValueError("B and V must both be (N,3)")

    orders_t = tuple(float(p) for p in orders)
    keys = ("all", "perp", "disp", "par")
    M = {k: np.full((J, len(orders_t)), np.nan, dtype=float) for k in keys}
    C = {k: np.zeros(J, dtype=int) for k in keys}

    ApprB, DetB, metaB = legacy_mra_details_and_background(B, wname=wname, J=J)
    ApprV, DetV, metaV = legacy_mra_details_and_background(V, wname=wname, J=J)
    scale = modwt_band_metrics(wname=wname, J=J, dt=dt)

    h_t, _g_t = dec_filters_scaled(wname)
    L0 = int(h_t.size)

    for j in range(J):
        Bl = ApprB[j]
        Vl = ApprV[j]
        dB = DetB[j]
        masks = compute_bin_masks_fast(Bl, Vl, dB, bins=bins, V_sc=V_sc, eps=eps, par_ignores_phi=par_ignores_phi)

        if trim_edges:
            jj = j + 1
            Leff = int((2 ** jj - 1) * (L0 - 1))
            if Leff > 0 and 2 * Leff < Bl.shape[0]:
                edge = np.ones(Bl.shape[0], dtype=bool)
                edge[:Leff] = False
                edge[-Leff:] = False
                for k in keys:
                    masks[k] = masks[k] & edge

        amp = np.linalg.norm(dB, axis=-1)
        out, cnt = _moments_level(amp, masks, orders_t)
        for k in keys:
            M[k][j, :] = out[k]
            C[k][j] = cnt[k]

    flat = None
    if 2.0 in orders_t and 4.0 in orders_t:
        i2 = orders_t.index(2.0)
        i4 = orders_t.index(4.0)
        flat = {k: flatness_from_moments(M[k][:, i2], M[k][:, i4]) for k in keys}

    return AnisMomentResult(
        tau=scale.tau,
        f_center=scale.f_center,
        moments=M,
        counts=C,
        flatness=flat,
        meta={
            "operator": "wavelet_legacy",
            "recon_B_maxabs": metaB["recon_maxabs"],
            "recon_V_maxabs": metaV["recon_maxabs"],
            "bandwidth": scale.bandwidth,
        },
    )


def anisotropic_wavelet_psd(res: AnisMomentResult, order_index: int = 0) -> Dict[str, Array]:
    bw = np.maximum(np.asarray(res.meta.get("bandwidth", np.ones_like(res.tau)), float), 1e-300)
    return {k: res.moments[k][:, order_index] / bw for k in res.moments}


# =============================================================================
# Constant amplitude overlay coefficient
# =============================================================================
def constant_match_coefficient(M_ref: Array, M_tgt: Array, j_fit: Optional[Sequence[int]] = None, p: float = 2.0) -> float:
    """
    c = ( median_{j in j_fit} (M_ref / M_tgt) )^{1/p}
    """
    M_ref = np.asarray(M_ref, float)
    M_tgt = np.asarray(M_tgt, float)
    if j_fit is None:
        idx = np.where(np.isfinite(M_ref) & np.isfinite(M_tgt) & (M_ref > 0) & (M_tgt > 0))[0]
    else:
        j_fit = np.asarray(list(j_fit), dtype=int)
        m = np.isfinite(M_ref[j_fit]) & np.isfinite(M_tgt[j_fit]) & (M_ref[j_fit] > 0) & (M_tgt[j_fit] > 0)
        idx = j_fit[m]
    if idx.size == 0:
        return float("nan")
    r = M_ref[idx] / M_tgt[idx]
    return float(np.nanmedian(r) ** (1.0 / p))


# =============================================================================
# Benchmarking
# =============================================================================
@dataclass(frozen=True)
class BenchResult:
    name: str
    seconds: float
    slope_SF2_all: float
    amp_ratio_all: float


def _fit_loglog_slope(x: Array, y: Array, j0: int = 2, j1: int = 8) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    idx = np.where(m)[0]
    idx = idx[(idx >= j0) & (idx < j1)]
    if idx.size < 3:
        return float("nan")
    xx = np.log10(x[idx])
    yy = np.log10(y[idx])
    A = np.vstack([xx, np.ones_like(xx)]).T
    a, _b = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(a)


def benchmark_against_5pt(N: int = 2 ** 14, J: int = 10, dt: float = 1.0, wname: str = "la8", repeats: int = 2) -> List[BenchResult]:
    import time

    rng = np.random.default_rng(0)
    B = rng.normal(size=(N, 3))
    V = rng.normal(size=(N, 3))
    bins = AnisBins()

    # warmup
    _ = anisotropic_fivepoint(B, V, dt=dt, J=J, orders=(2.0, 4.0), bins=bins)
    _ = anisotropic_wavelet_new(B, V, wname=wname, dt=dt, J=J, orders=(2.0, 4.0), bins=bins)
    _ = anisotropic_wavelet_legacy(B, V, wname=wname, dt=dt, J=J, orders=(2.0, 4.0), bins=bins)

    def timeit(fn):
        t0 = time.perf_counter()
        out = None
        for _ in range(repeats):
            out = fn()
        t1 = time.perf_counter()
        return (t1 - t0) / repeats, out

    t5, r5 = timeit(lambda: anisotropic_fivepoint(B, V, dt=dt, J=J, orders=(2.0, 4.0), bins=bins))
    tn, rn = timeit(lambda: anisotropic_wavelet_new(B, V, wname=wname, dt=dt, J=J, orders=(2.0, 4.0), bins=bins))
    tl, rl = timeit(lambda: anisotropic_wavelet_legacy(B, V, wname=wname, dt=dt, J=J, orders=(2.0, 4.0), bins=bins))

    M5 = r5.moments["all"][:, 0]
    s5 = _fit_loglog_slope(r5.tau, M5)
    sN = _fit_loglog_slope(rn.tau, rn.moments["all"][:, 0])
    sL = _fit_loglog_slope(rl.tau, rl.moments["all"][:, 0])

    j_fit = list(range(2, min(8, J)))
    rN = float(np.nanmedian(rn.moments["all"][j_fit, 0] / np.maximum(M5[j_fit], 1e-300)))
    rL = float(np.nanmedian(rl.moments["all"][j_fit, 0] / np.maximum(M5[j_fit], 1e-300)))

    return [
        BenchResult("5pt", t5, s5, 1.0),
        BenchResult("wavelet_new", tn, sN, rN),
        BenchResult("wavelet_legacy", tl, sL, rL),
    ]


# =============================================================================
# Tests (call manually or with pytest)
# =============================================================================
def test_conv_definition() -> None:
    rng = np.random.default_rng(0)
    N = 256
    x = rng.normal(size=(N, 3))
    f = rng.normal(size=8)
    j = 3
    step = 2 ** (j - 1)
    y = _cconv_roll_minus(x, f, step)
    y_ref = np.zeros_like(x)
    for i in range(N):
        acc = np.zeros(3)
        for t in range(f.size):
            acc += f[t] * x[(i - t * step) % N]
        y_ref[i] = acc
    assert float(np.max(np.abs(y - y_ref))) < 1e-12


def test_legacy_reconstruction_small() -> None:
    rng = np.random.default_rng(0)
    N = 4096
    x = rng.normal(size=(N, 3))
    _A, _D, meta = legacy_mra_details_and_background(x, wname="la8", J=6)
    assert meta["recon_maxabs"] < 1e-1


def test_benchmark_runs() -> None:
    benches = benchmark_against_5pt(N=2**12, J=8, dt=1.0, wname="la8", repeats=1)
    assert len(benches) == 3
    for b in benches:
        assert np.isfinite(b.seconds)


if __name__ == "__main__":
    benches = benchmark_against_5pt(N=2**14, J=10, dt=1.0, wname="la8", repeats=2)
    for b in benches:
        print(b)




# import numpy as np
# import pdb
# import pywt
# from scipy.ndimage import convolve1d
# from joblib import Parallel, delayed


# def upArrow_op(li, j):
#     if j == 0:
#         return [1]
#     N = len(li)
#     li_n = np.zeros(2**(j - 1) * (N - 1) + 1)
#     for i in range(N):
#         li_n[2**(j - 1) * i] = li[i]
#     return li_n


# def period_list(li, N):
#     n = len(li)
#     # append [0 0 ...]
#     n_app = N - np.mod(n, N)
#     li = list(li)
#     li = li + [0] * n_app
#     if len(li) < 2 * N:
#         return np.array(li)
#     else:
#         li = np.array(li)
#         li = np.reshape(li, [-1, N])
#         li = np.sum(li, axis=0)
#         return li


# def circular_convolve_mra(h_j_o, w_j):
#     ''' calculate the mra D_j'''
#     return convolve1d(w_j,
#                       np.flip(h_j_o),
#                       mode="wrap",
#                       origin=(len(h_j_o) - 1) // 2)


# def circular_convolve_d(h_t, v_j_1, j):
#     '''
#     jth level decomposition
#     h_t: \tilde{h} = h / sqrt(2)
#     v_j_1: v_{j-1}, the (j-1)th scale coefficients
#     return: w_j (or v_j)
#     '''
#     N = len(v_j_1)
#     w_j = np.zeros(N)
#     ker = np.zeros(len(h_t) * 2**(j - 1))

#     # make kernel
#     for i, h in enumerate(h_t):
#         ker[i * 2**(j - 1)] = h

#     w_j = convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)
#     return w_j


# def circular_convolve_s(h_t, g_t, w_j, v_j, j):
#     '''
#     (j-1)th level synthesis from w_j, w_j
#     see function circular_convolve_d
#     '''
#     N = len(v_j)

#     h_ker = np.zeros(len(h_t) * 2**(j - 1))
#     g_ker = np.zeros(len(g_t) * 2**(j - 1))

#     for i, (h, g) in enumerate(zip(h_t, g_t)):
#         h_ker[i * 2**(j - 1)] = h
#         g_ker[i * 2**(j - 1)] = g

#     v_j_1 = np.zeros(N)

#     v_j_1 = convolve1d(w_j,
#                        np.flip(h_ker),
#                        mode="wrap",
#                        origin=(len(h_ker) - 1) // 2)
#     v_j_1 += convolve1d(v_j,
#                         np.flip(g_ker),
#                         mode="wrap",
#                         origin=(len(g_ker) - 1) // 2)
#     return v_j_1


# # def modwt(x, filters, level):
# #     '''
# #     filters: 'db1', 'db2', 'haar', ...
# #     return: see matlab
# #     '''
# #     # filter
# #     wavelet = pywt.Wavelet(filters)
# #     h = wavelet.dec_hi
# #     g = wavelet.dec_lo
# #     h_t = np.array(h) / np.sqrt(2)
# #     g_t = np.array(g) / np.sqrt(2)
# #     wavecoeff = []
# #     v_j_1 = x
# #     for j in range(level):
# #         w = circular_convolve_d(h_t, v_j_1, j + 1)
# #         v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
# #         wavecoeff.append(w)
# #     wavecoeff.append(v_j_1)
# #     return np.vstack(wavecoeff)




# def modwt(x, filters, level):
#     wavelet = pywt.Wavelet(filters)
#     h = wavelet.dec_hi
#     g = wavelet.dec_lo
#     h_t = np.array(h) / np.sqrt(2)
#     g_t = np.array(g) / np.sqrt(2)

#     # Function to perform the convolution at each level
#     def convolve_level(j):
#         w = circular_convolve_d(h_t, v_j_1, j + 1)
#         v_j = circular_convolve_d(g_t, v_j_1, j + 1)
#         return w, v_j

#     v_j_1 = x
#     wavecoeff = []

#     # Parallel computation for each level
#     results = Parallel(n_jobs=-1)(delayed(convolve_level)(j) for j in range(level))
    
#     for w, v_j in results:
#         wavecoeff.append(w)
#         v_j_1 = v_j

#     wavecoeff.append(v_j_1)
#     return np.vstack(wavecoeff)


# def imodwt(w, filters):
#     ''' inverse modwt '''
#     # filter
#     wavelet = pywt.Wavelet(filters)
#     h = wavelet.dec_hi
#     g = wavelet.dec_lo
#     h_t = np.array(h) / np.sqrt(2)
#     g_t = np.array(g) / np.sqrt(2)
#     level = len(w) - 1
#     v_j = w[-1]
#     for jp in range(level):
#         j = level - jp - 1
#         v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
#     return v_j


# def modwtmra(w, filters):
#     ''' Multiresolution analysis based on MODWT'''
#     # filter
#     wavelet = pywt.Wavelet(filters)
#     h = wavelet.dec_hi
#     g = wavelet.dec_lo
#     # D
#     level, N = w.shape
#     level = level - 1
#     D = []
#     g_j_part = [1]
#     for j in range(level):
#         # g_j_part
#         g_j_up = upArrow_op(g, j)
#         g_j_part = np.convolve(g_j_part, g_j_up)
#         # h_j_o
#         h_j_up = upArrow_op(h, j + 1)
#         h_j = np.convolve(g_j_part, h_j_up)
#         h_j_t = h_j / (2**((j + 1) / 2.))
#         if j == 0: h_j_t = h / np.sqrt(2)
#         h_j_t_o = period_list(h_j_t, N)
#         D.append(circular_convolve_mra(h_j_t_o, w[j]))
#     # S
#     j = level - 1
#     g_j_up = upArrow_op(g, j + 1)
#     g_j = np.convolve(g_j_part, g_j_up)
#     g_j_t = g_j / (2**((j + 1) / 2.))
#     g_j_t_o = period_list(g_j_t, N)
#     S = circular_convolve_mra(g_j_t_o, w[-1])
#     D.append(S)
#     return np.vstack(D)



# if __name__ == '__main__':
#     s1 = np.arange(10)
#     ws = modwt(s1, 'db2', 3)
#     s1p = imodwt(ws, 'db2')
#     mra = modwtmra(ws, 'db2')