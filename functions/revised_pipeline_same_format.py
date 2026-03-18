import os
import sys
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import pywt
from scipy import constants

# ------------------------------------------------------------
# Optional Numba acceleration for geometry / conditional spectra
# ------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap


# ------------------------------------------------------------
# MHDTurbPy imports (unchanged style)
# ------------------------------------------------------------
sys.path.insert(1, os.path.join(os.getcwd(), "functions"))
import calc_diagnostics as calc  # noqa: F401
import TurbPy as turb  # noqa: F401
import general_functions as func
import Figures as figs  # noqa: F401
from SEA import SEA  # noqa: F401
import three_D_funcs as threeD  # noqa: F401
import download_data as download  # noqa: F401

# IMPORTANT: this should point to the patched backend
import modwtpy_fast_fixed as mw


# ------------------------------------------------------------
# Wavelet name normalization for PyWavelets
# ------------------------------------------------------------
@lru_cache(maxsize=64)
def _normalize_wname_for_modwtpy(wname):
    wname = str(wname).lower().strip()
    if wname.startswith("la"):
        L = int("".join(ch for ch in wname if ch.isdigit()))
        if L % 2 != 0:
            raise ValueError("Expected even LA length, got %s" % wname)
        return "sym%d" % (L // 2)
    if wname.startswith("d"):
        digits = "".join(ch for ch in wname if ch.isdigit())
        if digits:
            L = int(digits)
            if L % 2 == 0:
                return "sym%d" % (L // 2)
        return "db2"
    return wname


@lru_cache(maxsize=64)
def _wavelet_obj(wname_norm):
    return pywt.Wavelet(wname_norm)


@lru_cache(maxsize=64)
def _dec_filters(wname_norm):
    wavelet = _wavelet_obj(wname_norm)
    h_t = np.asarray(wavelet.dec_hi, dtype=float) / np.sqrt(2.0)
    g_t = np.asarray(wavelet.dec_lo, dtype=float) / np.sqrt(2.0)
    return h_t, g_t


@lru_cache(maxsize=64)
def _filter_length(wname_norm):
    return int(_wavelet_obj(wname_norm).dec_len)


# ------------------------------------------------------------
# Conservative and legacy MODWT level choices
# ------------------------------------------------------------
def recommended_modwt_level(N, wname):
    wname_norm = _normalize_wname_for_modwtpy(wname)
    filt_len = _filter_length(wname_norm)
    if N <= 1:
        return 1
    val = int(np.floor(np.log2(max(float(N) / float(max(filt_len - 1, 1)), 1.0))))
    return max(1, val)


def legacy_default_modwt_level(N):
    if N <= 2:
        return 1
    return max(1, int(np.floor(np.log2(N))) - 1)


# ------------------------------------------------------------
# Sequential MODWT cascade on (N,C)
# ------------------------------------------------------------
def _modwt_seq_multi(x_mat, wname, level):
    wname_norm = _normalize_wname_for_modwtpy(wname)
    h_t, g_t = _dec_filters(wname_norm)

    v = np.asarray(x_mat, dtype=float)
    if v.ndim == 1:
        v = v[:, None]

    coeffs = []
    for j in range(level):
        w = mw.circular_convolve_d(h_t, v, j + 1)
        v = mw.circular_convolve_d(g_t, v, j + 1)
        coeffs.append(w)

    coeffs.append(v)
    return np.stack(coeffs, axis=0)


# ------------------------------------------------------------
# Cached sparse MRA filters
# ------------------------------------------------------------
@lru_cache(maxsize=64)
def _mra_sparse_filters_periodized(wname_norm, level, N):
    wavelet = _wavelet_obj(wname_norm)
    h = np.asarray(wavelet.dec_hi, dtype=float)
    g = np.asarray(wavelet.dec_lo, dtype=float)

    h_idx_list = []
    h_val_list = []
    g_j_part = np.array([1.0], dtype=float)

    for j in range(level):
        g_j_up = mw.upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)

        h_j_up = mw.upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)

        h_j_t = h_j / (2.0 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2.0)

        h_j_t_o = mw.period_list(h_j_t, N)
        idx = np.nonzero(h_j_t_o != 0.0)[0].astype(np.int64)
        val = np.asarray(h_j_t_o[idx], dtype=float)
        h_idx_list.append(idx)
        h_val_list.append(val)

    j = level - 1
    g_j_up = mw.upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2.0 ** ((j + 1) / 2.0))
    g_j_t_o = mw.period_list(g_j_t, N)
    g_idx = np.nonzero(g_j_t_o != 0.0)[0].astype(np.int64)
    g_val = np.asarray(g_j_t_o[g_idx], dtype=float)

    return h_idx_list, h_val_list, g_idx, g_val


def _modwtmra_multi(W, wname):
    W = np.asarray(W, dtype=float)
    if W.ndim == 2:
        W = W[:, :, None]

    level_plus_1, N, _ = W.shape
    level = level_plus_1 - 1
    wname_norm = _normalize_wname_for_modwtpy(wname)

    h_idx_list, h_val_list, g_idx, g_val = _mra_sparse_filters_periodized(wname_norm, level, N)

    M = np.empty_like(W)
    for j in range(level):
        M[j] = mw.circular_convolve_mra_sparse(h_idx_list[j], h_val_list[j], W[j])
    M[-1] = mw.circular_convolve_mra_sparse(g_idx, g_val, W[-1])
    return M


# ------------------------------------------------------------
# Reconstruction-consistent details + approximations per level
# ------------------------------------------------------------
def estimate_coeffs_background_flucs_MODWT(x, wname, level=None, level_mode="recommended"):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x_mat = x[:, None]
    elif x.ndim == 2:
        x_mat = x
    else:
        raise ValueError("x must be 1D or 2D (N,C)")

    N, C = x_mat.shape
    rec_level = recommended_modwt_level(N, wname)
    leg_level = legacy_default_modwt_level(N)

    if level is None:
        if level_mode == "legacy":
            level = leg_level
        else:
            level = rec_level
    else:
        level = int(level)

    if level < 1:
        raise ValueError("level must be >= 1")
    if level > leg_level:
        raise ValueError("level=%d exceeds legacy_default_modwt_level=%d for N=%d" % (level, leg_level, N))
    if level > rec_level:
        warnings.warn(
            "Requested MODWT level %d exceeds conservative recommended level %d for N=%d and wavelet=%s"
            % (level, rec_level, N, _normalize_wname_for_modwtpy(wname)),
            RuntimeWarning,
        )

    W = _modwt_seq_multi(x_mat, wname, level)
    M = _modwtmra_multi(W, wname)

    Det = np.asarray(M[:-1], dtype=float)
    S = np.asarray(M[-1], dtype=float)
    J0 = Det.shape[0]

    Appr = np.zeros_like(Det)
    if J0 == 1:
        Appr[0] = S
    else:
        tail = np.cumsum(Det[::-1, :, :], axis=0)[::-1, :, :]
        for j in range(J0):
            Appr[j] = S if j == J0 - 1 else (S + tail[j + 1])

    x_rec = S + np.sum(Det, axis=0)
    recon_maxabs = float(np.nanmax(np.abs(x_mat - x_rec)))

    prefix = np.cumsum(Det, axis=0)
    per_level = np.nanmax(np.abs(prefix + Appr - x_mat[None, :, :]), axis=(1, 2))
    per_level_maxabs = float(np.nanmax(per_level))

    meta = {
        "N": int(N),
        "C": int(C),
        "J0": int(J0),
        "recon_maxabs": recon_maxabs,
        "per_level_identity_maxabs": per_level_maxabs,
        "per_level_identity": per_level,
        "wname_used": _normalize_wname_for_modwtpy(wname),
        "level_requested": int(level),
        "recommended_level": int(rec_level),
        "legacy_default_level": int(leg_level),
        "level_mode": str(level_mode),
    }

    if x.ndim == 1:
        return Appr[:, :, 0], Det[:, :, 0], meta
    return Appr, Det, meta


def estimate_approxs(db, b_cols, Bl):
    ApprB = {}
    detB = {}
    for b_col in b_cols:
        det = np.asarray(db[b_col], dtype=float)
        app = np.asarray(Bl[b_col], dtype=float)
        if det.shape != app.shape:
            raise ValueError("%s: det shape %s != app shape %s" % (b_col, det.shape, app.shape))
        ApprB[b_col] = app
        detB[b_col] = det
    return ApprB, detB


# ------------------------------------------------------------
# Robust local geometry
# ------------------------------------------------------------
def _project_perp(a, b, eps=1e-30):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    b2 = np.sum(b * b, axis=-1, keepdims=True)
    out = np.full_like(a, np.nan, dtype=float)
    valid = b2[..., 0] > eps
    if np.any(valid):
        coeff = np.sum(a[valid] * b[valid], axis=-1, keepdims=True) / b2[valid]
        out[valid] = a[valid] - coeff * b[valid]
    return out


def _folded_angle_deg(a, b, eps=1e-30):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    out = np.full(an.shape, np.nan, dtype=float)
    valid = (an > eps) & (bn > eps)
    if np.any(valid):
        cosang = np.sum(a[valid] * b[valid], axis=-1) / (an[valid] * bn[valid])
        cosang = np.clip(np.abs(cosang), 0.0, 1.0)
        out[valid] = np.degrees(np.arccos(cosang))
    return out


@njit(cache=True)
def _conditional_power_numba(flucts, theta, phi, theta_perp_min, phi_aligned_max, phi_cross_min):
    n = flucts.shape[0]
    out = np.zeros(4, dtype=np.float64)
    cnt = np.zeros(4, dtype=np.int64)
    for i in range(n):
        th = theta[i]
        ph = phi[i]
        if np.isfinite(th):
            val = flucts[i, 0] * flucts[i, 0] + flucts[i, 1] * flucts[i, 1] + flucts[i, 2] * flucts[i, 2]
            out[3] += val
            cnt[3] += 1
            if th < 10.0:
                out[2] += val
                cnt[2] += 1
            if np.isfinite(ph) and (th > theta_perp_min):
                if ph < phi_aligned_max:
                    out[1] += val
                    cnt[1] += 1
                if ph > phi_cross_min:
                    out[0] += val
                    cnt[0] += 1
    for k in range(4):
        if cnt[k] > 0:
            out[k] /= cnt[k]
        else:
            out[k] = np.nan
    return out, cnt


def local_structure_function(dB, B_l, V_l, tau, return_unit_vecs=False, five_points_sfunc=True):
    del tau, return_unit_vecs, five_points_sfunc
    dB_perp = _project_perp(dB, B_l)
    V_perp = _project_perp(V_l, B_l)
    VBangle = _folded_angle_deg(V_l, B_l)
    Phiangle = _folded_angle_deg(V_perp, dB_perp)
    return dB, VBangle, Phiangle


def PSD_anis_MODWT(coeffs, indices, iterration, dt):
    coeff = (2 ** (iterration + 1)) * dt
    if indices.size == 0:
        return np.nan
    vals = coeffs[indices]
    return coeff * float(np.nanmean(np.sum(vals * vals, axis=1)))


def estimate_3D_sfuncs(
    flucs,
    db,
    B_l,
    V_l,
    dt,
    Vsw,
    di,
    conditions,
    estimate_PDFS=False,
    return_unit_vecs=False,
    five_points_sfuncs=True,
    return_coefs=False,
):
    del Vsw, di, estimate_PDFS, return_unit_vecs, five_points_sfuncs, return_coefs

    sf_ell_perp_conds = conditions["ell_perp"]
    sf_Ell_perp_conds = conditions["Ell_perp"]

    tau_values = 2 ** np.arange(1, len(B_l["R"]) + 1)

    sf_ell_perp = np.full(len(tau_values), np.nan, dtype=float)
    sf_Ell_perp = np.full(len(tau_values), np.nan, dtype=float)
    sf_ell_par = np.full(len(tau_values), np.nan, dtype=float)
    sf_overall = np.full(len(tau_values), np.nan, dtype=float)
    thetas = {}
    phis = {}

    B_stack = np.stack((B_l["R"], B_l["T"], B_l["N"]), axis=2)
    V_stack = np.stack((V_l["R"], V_l["T"], V_l["N"]), axis=2)
    dB_stack = np.stack((db["R"], db["T"], db["N"]), axis=2)
    fl_stack = np.stack((flucs["R"], flucs["T"], flucs["N"]), axis=2)

    theta_perp_min = float(min(sf_ell_perp_conds["theta"], sf_Ell_perp_conds["theta"]))
    phi_aligned_max = float(sf_Ell_perp_conds["phi"])
    phi_cross_min = float(sf_ell_perp_conds["phi"])

    for jj, tau_value in enumerate(tau_values):
        try:
            Bls = np.asarray(B_stack[jj], dtype=float)
            Vls = np.asarray(V_stack[jj], dtype=float)
            dBs = np.asarray(dB_stack[jj], dtype=float)
            flucts = np.asarray(fl_stack[jj], dtype=float)

            dB, VBangle, Phiangle = local_structure_function(dBs, Bls, Vls, tau_value)
            thetas[str(jj)] = VBangle
            phis[str(jj)] = Phiangle

            coeff = (2 ** (jj + 1)) * dt
            if _HAVE_NUMBA:
                pows, _ = _conditional_power_numba(flucts, VBangle, Phiangle, theta_perp_min, phi_aligned_max, phi_cross_min)
                sf_ell_perp[jj] = coeff * pows[0]
                sf_Ell_perp[jj] = coeff * pows[1]
                sf_ell_par[jj] = coeff * pows[2]
                sf_overall[jj] = coeff * pows[3]
            else:
                energy = np.sum(flucts * flucts, axis=1)
                theta_valid = np.isfinite(VBangle)
                phi_valid = np.isfinite(Phiangle)
                overall_mask = theta_valid
                par_mask = theta_valid & (VBangle < 10.0)
                aligned_mask = theta_valid & phi_valid & (VBangle > theta_perp_min) & (Phiangle < phi_aligned_max)
                cross_mask = theta_valid & phi_valid & (VBangle > theta_perp_min) & (Phiangle > phi_cross_min)

                if np.any(cross_mask):
                    sf_ell_perp[jj] = coeff * float(np.nanmean(energy[cross_mask]))
                if np.any(aligned_mask):
                    sf_Ell_perp[jj] = coeff * float(np.nanmean(energy[aligned_mask]))
                if np.any(par_mask):
                    sf_ell_par[jj] = coeff * float(np.nanmean(energy[par_mask]))
                if np.any(overall_mask):
                    sf_overall[jj] = coeff * float(np.nanmean(energy[overall_mask]))
        except Exception:
            sf_overall[jj] = np.nan
            sf_ell_par[jj] = np.nan
            sf_ell_perp[jj] = np.nan
            sf_Ell_perp[jj] = np.nan

    nyquist_freq = 0.5 / dt
    frequencies = nyquist_freq / (2 ** (np.arange(1, len(sf_overall) + 1)))

    return thetas, phis, None, frequencies, sf_ell_perp.T, sf_Ell_perp.T, sf_ell_par.T, sf_overall.T
