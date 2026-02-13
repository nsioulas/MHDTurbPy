
# ==============================================================
#  FULL, COPY/PASTE PIPELINE (REVISED: NO in-place DF mutation)
#
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import traceback
from pathlib import Path

_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent

from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from scipy import constants
from scipy.signal import convolve

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import ssqueezepy

sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'pyspedas'))
sys.path.insert(0, str(_FUNCTIONS_DIR))
import TurbPy as turb
import general_functions as func


# ==============================================================
#  Wavelet / smoothing utilities
# ==============================================================

def morlet_sigma_from_freq(freq_hz, omega0=6.0):
    """
    Morlet equivalent time-width parameter sigma (in seconds) from frequency f [Hz].

    Uses:
        f = (ω0 + sqrt(2+ω0^2)) / (4π sigma)
    so
        sigma = (ω0 + sqrt(2+ω0^2)) / (4π f)

    This sigma is what Gerick+2017 calls the Gaussian envelope width parameter.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    const = (omega0 + np.sqrt(2.0 + omega0**2)) / (4.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = const / freq_hz
    sigma[~np.isfinite(sigma)] = np.nan
    return sigma


def estimate_cwt(
    signal,
    dt,
    nv=32,
    omega0=6,
    scale_type="log-piecewise",
    vectorized=True,
    l1_norm=False,
    min_freq=None,
    max_freq=None,
):
    """
    Returns:
      W: dict of wavelet coeffs if DataFrame input, else ndarray
      scales: sigma(t) [seconds] corresponding to each frequency
      freqs: pseudo-frequencies [Hz]
      psd_norm: 2*dt (same convention you were using)
      coi: None (placeholder)
    """
    fs = 1.0 / dt
    wavelet = ssqueezepy.Wavelet(("morlet", {"mu": omega0}))

    def _mask_band(W, scales, freqs):
        nyquist = fs / 2.0
        cutoff = 0.95 * nyquist
        mask = freqs < cutoff

        W = W[mask, :]
        scales = scales[mask]
        freqs = freqs[mask]

        if min_freq is not None and max_freq is not None:
            idx = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
            W = W[idx, :]
            scales = scales[idx]
            freqs = freqs[idx]

        return W, scales, freqs

    if isinstance(signal, pd.DataFrame):
        w_df = {}
        for col in signal.columns:
            W, scales = ssqueezepy.cwt(
                signal[col].values,
                wavelet=wavelet,
                scales=scale_type,
                l1_norm=l1_norm,
                fs=fs,
                nv=nv,
                vectorized=vectorized,
            )
            freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal[col]), fs)

            # Convert to the Gerick-style sigma (seconds)
            scales = morlet_sigma_from_freq(freqs, omega0=omega0)

            W, scales, freqs = _mask_band(W, scales, freqs)
            w_df[col] = W

        return w_df, scales, freqs, 2.0 * dt, None

    else:
        W, scales = ssqueezepy.cwt(
            signal,
            wavelet=wavelet,
            scales=scale_type,
            l1_norm=l1_norm,
            fs=fs,
            nv=nv,
            vectorized=vectorized,
        )

        freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal), fs)
        scales = morlet_sigma_from_freq(freqs, omega0=omega0)

        W, scales, freqs = _mask_band(W, scales, freqs)
        return W, scales, freqs, 2.0 * dt, None


def gaussian_kernel_from_sigma_samples(sigma_samples, num_efoldings=3):
    """
    Build the exact same normalized Gaussian kernel used by your original convolve(same).
    """
    if not np.isfinite(sigma_samples) or sigma_samples <= 0:
        return None

    half_width = int(np.ceil(num_efoldings * sigma_samples))
    t = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-(t**2) / (2.0 * sigma_samples**2))
    s = kernel.sum()
    if s <= 0:
        return None
    kernel /= s
    return kernel


def gaussian_smooth_1d_same(x, kernel):
    """
    Exact 'same' convolution with a precomputed kernel.
    """
    if kernel is None:
        return np.full_like(x, np.nan, dtype=float)
    return convolve(x, kernel, mode="same")


def gaussian_smooth_2d_same(X, kernel):
    """
    Smooth an (N, M) array along axis=0 with the same 1D kernel.
    Keeps exact behavior of doing convolve(col, kernel, mode='same').

    X: float array shape (N, M)
    """
    if kernel is None:
        return np.full_like(X, np.nan, dtype=float)

    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = convolve(X[:, j], kernel, mode="same")
    return out


def _safe_unit(arr_2d):
    nrm = np.linalg.norm(arr_2d, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = arr_2d / nrm
    out[~np.isfinite(out)] = np.nan
    return out


def compute_local_Va_sigma(
    B_vec_sm,
    np_sm=None,
    frame_flag=None,
    C_ALFVEN=21.82,
):
    """
    B_vec_sm: (N,3) smoothed B in RTN-like basis labeled [R,T,N]
    np_sm:    (N,) smoothed proton density
    """
    B_mag = np.linalg.norm(B_vec_sm, axis=1)

    if np_sm is None:
        Va = np.full_like(B_mag, np.nan, dtype=float)
    else:
        np_sm = np.asarray(np_sm, dtype=float)
        np_sm[np_sm <= 0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            Va = C_ALFVEN * B_mag / np.sqrt(np_sm)
        Va[~np.isfinite(Va)] = np.nan

    if frame_flag is None:
        frame_flag = "sc"

    if str(frame_flag).upper() == "RTN":
        Br_like = B_vec_sm[:, 0]
    else:
        Br_like = -B_vec_sm[:, 2]

    sigma = np.sign(Br_like)
    sigma[sigma == 0] = 1.0
    sigma[~np.isfinite(sigma)] = 1.0

    return Va, sigma


# ==============================================================
#  Polarization utilities
# ==============================================================

def calculate_polarization_spectra(Wx, Wy):
    PL = np.abs(Wx - 1j * Wy) ** 2
    PR = np.abs(Wx + 1j * Wy) ** 2
    return PL, PR


# ==============================================================
#  build_V_mod_TH (kept as-is; still used for OTHER saved metrics)
# ==============================================================

def build_V_mod_TH(
    B, V,
    axes=None,
    sc="PSP",
    window="30min",
    correct_sign=True,
    consider_Va=True,
    consider_Vsc=True,
    return_addit=False,
):
    import numpy as np
    import pandas as pd

    _C_ALFVEN = 21.82

    if axes is None:
        axes = ["r", "t", "n"] if "Br" in B.columns else ["x", "y", "z"]
    is_rtn = axes[0] == "r"
    base_cols = [f"V{a}" for a in axes]
    sc_vel_cols = [f"sc_vel_{a}" for a in ("r", "t", "n")]
    B_cols = [f"B{a}" for a in axes]

    n_p_raw = V["np"].astype(float)
    Np_mean = n_p_raw.rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)
    Np_mean[Np_mean <= 0] = np.nan

    B0_vec_df = B[B_cols].rolling(window, center=True, min_periods=1).median()
    B0_vec_arr = B0_vec_df.to_numpy(dtype=float)

    B_inst_mag = np.linalg.norm(B[B_cols].to_numpy(dtype=float), axis=1)
    B0_scalar_vals = (
        pd.Series(B_inst_mag, index=B.index)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    V_arr = V[base_cols].to_numpy(dtype=float)

    if consider_Vsc and is_rtn and sc.upper() == "PSP":
        if all(c in V.columns for c in sc_vel_cols):
            V_sc_arr = V[sc_vel_cols].to_numpy(dtype=float)
            V_arr = V_arr - V_sc_arr

    V_rel_df = pd.DataFrame(V_arr, index=V.index, columns=base_cols)

    B0_vec_mag = np.linalg.norm(B0_vec_arr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_par = B0_vec_arr / B0_vec_mag[:, None]
    e_par[~np.isfinite(e_par)] = 0.0

    ref = np.array([1.0, 0.0, 0.0])
    e_p1 = np.cross(e_par, ref)
    idx_deg = np.linalg.norm(e_p1, axis=1) < 1e-6
    if np.any(idx_deg):
        e_p1[idx_deg] = np.cross(e_par[idx_deg], np.array([0.0, 1.0, 0.0]))
    n1 = np.linalg.norm(e_p1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_p1 = e_p1 / n1[:, None]
    e_p1[~np.isfinite(e_p1)] = 0.0
    e_p2 = np.cross(e_par, e_p1)

    v_par = np.einsum("ij,ij->i", V_arr, e_par)
    v_p1 = np.einsum("ij,ij->i", V_arr, e_p1)
    v_p2 = np.einsum("ij,ij->i", V_arr, e_p2)

    Vrel_perp_mag = np.sqrt(v_p1**2 + v_p2**2)
    Vrel_mag = np.sqrt(v_par**2 + Vrel_perp_mag**2)

    Va_bg = _C_ALFVEN * B0_scalar_vals / np.sqrt(Np_mean)
    if not consider_Va:
        Va_bg = 0.0 * Va_bg

    Br_like = B0_vec_arr[:, 0] if is_rtn else -B0_vec_arr[:, 2]
    sigma_bg = np.sign(Br_like)
    sigma_bg[sigma_bg == 0] = 1.0
    if not correct_sign:
        sigma_bg = np.abs(sigma_bg)

    v_par_eff = v_par + sigma_bg * Va_bg
    V_mod_mag = np.sqrt(v_par_eff**2 + Vrel_perp_mag**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = v_par / Vrel_mag
    theta_VB = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    summary_df = pd.DataFrame(
        {
            "θ_VB": theta_VB,
            "|Va|": Va_bg,
            "|V_rel|": Vrel_mag,
            "|V_mod_TH|": V_mod_mag,
            "B0_scalar": B0_scalar_vals,
            "B0_vec_mag": B0_vec_mag,
            "Np_background": Np_mean,
        },
        index=V.index,
    )

    if return_addit:
        V_MTH_extra = pd.DataFrame(
            {
                "Vrel_par": v_par,
                "Vrel_perp": Vrel_perp_mag,
                "Va_bg": Va_bg,
                "sigma_bg": sigma_bg,
                "sigma_Va": sigma_bg * Va_bg,
            },
            index=V.index,
        )
        return summary_df, V_rel_df, V_MTH_extra

    return summary_df, V_rel_df


# ==============================================================
#  coherence_analysis (REVISED: no DF mutation on B/V inputs)
# ==============================================================

def coherence_analysis(
    B_vec_sm,
    V_vec_sm,
    df_w,
    method,
    func_params=None,
):
    if func_params is None:
        func_params = {}
    rotate_to_k = bool(func_params.get("rotate_to_k", False))

    b_hat = _safe_unit(B_vec_sm)
    v_hat = _safe_unit(V_vec_sm)

    dot_bv = np.einsum("ij,ij->i", b_hat, v_hat)
    VBangles = np.degrees(np.arccos(np.clip(dot_bv, -1.0, 1.0)))

    eps = float(func_params.get("basis_eps", 1e-10))

    ey = np.cross(b_hat, v_hat)
    ey_norm = np.linalg.norm(ey, axis=1, keepdims=True)
    bad = (~np.isfinite(ey_norm[:, 0])) | (ey_norm[:, 0] < eps)

    if np.any(bad):
        ref1 = np.array([1.0, 0.0, 0.0], dtype=float)
        ref2 = np.array([0.0, 1.0, 0.0], dtype=float)

        ey_f = np.cross(b_hat[bad], ref1)
        n_f = np.linalg.norm(ey_f, axis=1, keepdims=True)

        use_ref2 = (~np.isfinite(n_f[:, 0])) | (n_f[:, 0] < eps)
        if np.any(use_ref2):
            ey_f2 = np.cross(b_hat[bad][use_ref2], ref2)
            ey_f[use_ref2] = ey_f2
            n_f[use_ref2] = np.linalg.norm(ey_f[use_ref2], axis=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            ey_f = ey_f / n_f
        ey_f[~np.isfinite(ey_f)] = np.nan

        ey[bad] = ey_f
        ey_norm[bad] = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ey = ey / ey_norm
    ey[~np.isfinite(ey)] = np.nan

    ex = np.cross(ey, b_hat)
    ex_norm = np.linalg.norm(ex, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ex = ex / ex_norm
    ex[~np.isfinite(ex)] = np.nan

    # Project wavelet coefficients
    rtn = df_w[["R", "T", "N"]].to_numpy(dtype=np.complex128, copy=False)
    Wx = np.einsum("ij,ij->i", ex, rtn).astype(np.complex128)
    Wy = np.einsum("ij,ij->i", ey, rtn).astype(np.complex128)

    df_w["Wx"] = Wx
    df_w["Wy"] = Wy
    df_w["W0"] = np.einsum("ij,ij->i", b_hat, rtn).astype(np.complex128)

    phi = np.zeros(len(df_w), dtype=float)

    if rotate_to_k:
        Pxx = (Wx.real**2 + Wx.imag**2)
        Pyy = (Wy.real**2 + Wy.imag**2)
        Pxy = (Wx.real * Wy.real + Wx.imag * Wy.imag)

        phi = 0.5 * np.arctan2(2.0 * Pxy, (Pyy - Pxx))
        sinφ = np.sin(phi)
        cosφ = np.cos(phi)

        W_perp2 = Wy * cosφ + Wx * sinφ
        W_perp1 = -Wy * sinφ + Wx * cosφ

        df_w["Wy"] = W_perp2
        df_w["Wx"] = W_perp1

        e1 = ex * cosφ[:, None] - ey * sinφ[:, None]

        e1_norm = np.linalg.norm(e1, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = e1 / e1_norm
        e1[~np.isfinite(e1)] = np.nan

        e2 = np.cross(b_hat, e1)
        e2_norm = np.linalg.norm(e2, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            e2 = e2 / e2_norm
        e2[~np.isfinite(e2)] = np.nan
    else:
        e1 = ex
        e2 = ey

    # Velocity projections
    V_par_local = np.einsum("ij,ij->i", V_vec_sm, b_hat)

    V_perp_vec = V_vec_sm - V_par_local[:, None] * b_hat
    V_perp = np.linalg.norm(V_perp_vec, axis=1)

    V_p1_local = np.einsum("ij,ij->i", V_vec_sm, e1)
    V_kperp = np.einsum("ij,ij->i", V_vec_sm, e2)

    with np.errstate(divide="ignore", invalid="ignore"):
        cphi = np.abs(V_kperp) / V_perp
    cphi[~np.isfinite(cphi)] = np.nan

    PL, PR = calculate_polarization_spectra(df_w["Wx"].values, df_w["Wy"].values)

    helicity = PR - PL
    trace_PSD = PR + PL
    cross_term = 2.0 * np.imag(np.conj(df_w["Wy"]) * df_w["W0"])
    power_sum = (
        np.abs(df_w["Wx"]) ** 2
        + np.abs(df_w["Wy"]) ** 2
        + np.abs(df_w["W0"]) ** 2
    )

    df_w.drop(columns=["R", "T", "N"], inplace=True, errors="ignore")

    kinematics = {
        "phi": phi,
        "V_par_local": V_par_local,
        "V_perp": V_perp,
        "V_kperp": V_kperp,
        "cphi": cphi,
    }

    return df_w, VBangles, helicity, trace_PSD, cross_term, power_sum, kinematics


# ==============================================================
#  Structure functions
# ==============================================================

def est_sfuncs(df_w, df_mod, index_par, index_per, scale, dts, func_params=None):
    def compute_SF(db, m, tau):
        return np.nanmean(np.abs(db / np.sqrt(tau)) ** m)

    if func_params is None:
        func_params = {}

    types = ["ov", "par", "per", "mod"]
    sf_dict = {f"SF_{t}_{m}": [] for t in types for m in range(func_params["max_qorder"])}

    db_vec = np.sqrt(np.nansum(np.abs(df_w.values * np.conj(df_w.values)), axis=1))
    db_mods = np.sqrt(np.abs(df_mod * np.conj(df_mod)))

    if func_params.get("est_sfuncs", False):
        for t in types:
            db = {
                "par": db_vec[index_par] if index_par is not None else np.nan,
                "per": db_vec[index_per] if index_per is not None else np.nan,
                "ov": db_vec,
                "mod": db_mods,
            }.get(t)

            for m in range(func_params["max_qorder"]):
                sf_dict[f"SF_{t}_{m}"].append(compute_SF(db, m, scale))

    return sf_dict


# ==============================================================
#  PSD / flatness / compressibility helpers
# ==============================================================

def compute_norm_psd(df_ws_needed, rolling_params, dt, counts, window_size, step, psd_norm):
    if not isinstance(df_ws_needed, pd.DataFrame):
        if hasattr(df_ws_needed, "to_frame"):
            df_ws_needed = df_ws_needed.to_frame()
        else:
            df_ws_needed = pd.DataFrame(df_ws_needed)

    psd_sum = psd_norm * (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean().sum(axis=1)
    PSD = (psd_sum * (counts / window_size)).resample(step).mean()
    return PSD


def calculate_wavelet_flatness(df_ws, rolling_params, step):
    if not isinstance(df_ws, pd.DataFrame):
        df_ws = pd.DataFrame(df_ws)

    power = np.abs(df_ws) ** 2
    A2 = power.sum(axis=1)
    A = np.sqrt(A2)

    m2 = A.pow(2).rolling(**rolling_params).mean()
    m4 = A.pow(4).rolling(**rolling_params).mean()

    flatness = (m4 / m2.pow(2)).resample(step).mean()
    return flatness


def compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm):
    if not isinstance(df_ws_needed, pd.DataFrame):
        df_ws_needed = pd.DataFrame(df_ws_needed)

    power = (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean()
    psd_sum = psd_norm * power.sum(axis=1)

    if index_mask is not None:
        counts = index_mask.rolling(**rolling_params).sum().resample(step).mean()
    else:
        counts = np.nan

    PSD = psd_sum.resample(step).mean()
    return PSD, counts


def calculate_wavelet_flatness_from_power(power, rolling_params, step):
    sum_power = power.sum(axis=1)
    numerator = sum_power.pow(2).rolling(**rolling_params).mean()
    denominator = sum_power.rolling(**rolling_params).mean().pow(2)
    flatness = (numerator / denominator).resample(step).mean()
    return flatness


def compute_all_metrics(
    df_num,
    df_den,
    rolling_params,
    step,
    psd_norm,
    dt,
    compute_sdk=True,
    mask=None,
):
    if not isinstance(df_num, pd.DataFrame):
        df_num = pd.DataFrame(df_num)
    if df_den is not None and not isinstance(df_den, pd.DataFrame):
        df_den = pd.DataFrame(df_den)

    if mask is not None:
        df_num = df_num.where(mask)
        if df_den is not None:
            df_den = df_den.where(mask)

    power_num = (df_num * np.conj(df_num)).apply(np.real)
    rolling_power_num = power_num.rolling(**rolling_params).mean()
    PSD_num = psd_norm * rolling_power_num.sum(axis=1)
    PSD_num = PSD_num.resample(step).mean()

    if compute_sdk:
        SDK_num = calculate_wavelet_flatness_from_power(power_num, rolling_params, step)
    else:
        SDK_num = None

    compress = None
    if df_den is not None:
        power_den = (df_den * np.conj(df_den)).apply(np.real)
        ratio = power_num.sum(axis=1) / power_den.sum(axis=1)
        compress = ratio.rolling(**rolling_params).mean().resample(step).mean()

    return {"PSD": PSD_num, "SDK": SDK_num, "compress": compress}


def est_compress(df_mod, df_w, index_par, index_per, dt, psd_norm, func_params=None):
    if func_params is None:
        func_params = {}

    estimate_comp = func_params.get("estimate_comp", False)
    use_rolling_mean = func_params.get("use_rolling_mean", False)

    PSD_par = PSD_per = PSD_mod = PSD_mod_per = PSD_mod_par = np.nan
    PSD_W0 = PSD_W0_per = PSD_W0_par = np.nan
    PSD_Wxy = PSD_Wxy_per = PSD_Wxy_par = np.nan
    compress_mod = compress_mod_par = compress_mod_per = np.nan
    SDK_par = SDK_per = SDK_W0 = SDK_Wxy = SDK_mod_par = SDK_mod_per = SDK_mod = np.nan

    if estimate_comp:
        try:
            if use_rolling_mean:
                averaging_window = func_params.get("averaging_window", "60s")
                step = func_params.get("step", "10s")
                rolling_params = {"window": averaging_window, "center": True, "min_periods": 10}

                df_mod_df = pd.DataFrame({"Mod": df_mod}, index=df_w.index)

                out_w0 = compute_all_metrics(
                    df_num=df_w[["W0"]],
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=None,
                )
                PSD_W0 = out_w0["PSD"]
                SDK_W0 = out_w0["SDK"]

                out_wxy = compute_all_metrics(
                    df_num=df_w[["Wx", "Wy"]],
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=None,
                )
                PSD_Wxy = out_wxy["PSD"]
                SDK_Wxy = out_wxy["SDK"]

                out_mod = compute_all_metrics(
                    df_num=df_mod_df,
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=None,
                )
                PSD_mod = out_mod["PSD"]
                SDK_mod = out_mod["SDK"]
                compress_mod = out_mod["compress"]

                mask_par = pd.Series(index_par, index=df_w.index) if index_par is not None else None
                mask_per = pd.Series(index_per, index=df_w.index) if index_per is not None else None

                out_par = compute_all_metrics(
                    df_num=df_w,
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_par,
                )
                PSD_par = out_par["PSD"]
                SDK_par = out_par["SDK"]

                out_per = compute_all_metrics(
                    df_num=df_w,
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per,
                )
                PSD_per = out_per["PSD"]
                SDK_per = out_per["SDK"]

                out_mod_par = compute_all_metrics(
                    df_num=df_mod_df,
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_par,
                )
                PSD_mod_par = out_mod_par["PSD"]
                SDK_mod_par = out_mod_par["SDK"]
                compress_mod_par = out_mod_par["compress"]

                out_mod_per = compute_all_metrics(
                    df_num=df_mod_df,
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per,
                )
                PSD_mod_per = out_mod_per["PSD"]
                SDK_mod_per = out_mod_per["SDK"]
                compress_mod_per = out_mod_per["compress"]

                out_w0_par = compute_all_metrics(
                    df_num=df_w[["W0"]],
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_par,
                )
                PSD_W0_par = out_w0_par["PSD"]

                out_w0_per = compute_all_metrics(
                    df_num=df_w[["W0"]],
                    df_den=df_w,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per,
                )
                PSD_W0_per = out_w0_per["PSD"]

                out_wxy_par = compute_all_metrics(
                    df_num=df_w[["Wx", "Wy"]],
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_par,
                )
                PSD_Wxy_par = out_wxy_par["PSD"]

                out_wxy_per = compute_all_metrics(
                    df_num=df_w[["Wx", "Wy"]],
                    df_den=None,
                    rolling_params=rolling_params,
                    step=step,
                    psd_norm=psd_norm,
                    dt=dt,
                    compute_sdk=True,
                    mask=mask_per,
                )
                PSD_Wxy_per = out_wxy_per["PSD"]
            else:
                tot_par_power = (
                    np.real(df_w.iloc[index_par] * np.conj(df_w.iloc[index_par]))
                    if index_par is not None
                    else np.nan
                )
                mod_par_power = (
                    np.real(df_mod[index_par] * np.conj(df_mod[index_par]))
                    if index_par is not None
                    else np.nan
                )
                compress_mod_par = (
                    (np.nanmean(mod_par_power) / np.nanmean(tot_par_power.sum(axis=1)))
                    if index_par is not None
                    else np.nan
                )

                tot_per_power = (
                    np.real(df_w.iloc[index_per] * np.conj(df_w.iloc[index_per]))
                    if index_per is not None
                    else np.nan
                )
                mod_per_power = (
                    np.real(df_mod[index_per] * np.conj(df_mod[index_per]))
                    if index_per is not None
                    else np.nan
                )
                compress_mod_per = (
                    (np.nanmean(mod_per_power) / np.nanmean(tot_per_power.sum(axis=1)))
                    if index_per is not None
                    else np.nan
                )

                tot_all_power = np.real(df_w * np.conj(df_w))
                mod_all_power = np.real(df_mod * np.conj(df_mod))
                compress_mod = np.nanmean(mod_all_power) / np.nanmean(tot_all_power.sum(axis=1))
        except Exception:
            traceback.print_exc()
            compress_mod_par = compress_mod_per = compress_mod = np.nan

    return {
        "PSD_par": PSD_par,
        "PSD_per": PSD_per,
        "PSD_mod": PSD_mod,
        "PSD_mod_par": PSD_mod_par,
        "PSD_mod_per": PSD_mod_per,
        "PSD_Bpar": PSD_W0,
        "PSD_Bpar_par": PSD_W0_par,
        "PSD_Bpar_per": PSD_W0_per,
        "PSD_Bper": PSD_Wxy,
        "PSD_Bper_par": PSD_Wxy_par,
        "PSD_Bper_per": PSD_Wxy_per,
        "compress_mod": compress_mod,
        "compress_mod_par": compress_mod_par,
        "compress_mod_per": compress_mod_per,
        "SDK_par": SDK_par,
        "SDK_per": SDK_per,
        "SDK_Bpar": SDK_W0,
        "SDK_Bper": SDK_Wxy,
        "SDK_mod": SDK_mod,
        "SDK_mod_par": SDK_mod_par,
        "SDK_mod_per": SDK_mod_per,
    }


# ==============================================================
#  do_coh_analysis (keeps all original keys; k-keys already present)
# ==============================================================

def do_coh_analysis(
    k_df,
    df_w,
    S0,
    S3,
    S0_full,
    Syz,
    index_par,
    index_per,
    dt,
    scale,
    psd_norm,
    func_params=None,
):
    if func_params is None:
        func_params = {}

    if not func_params.get("estimate_coh_coeffs", True):
        return None

    use_rolling_mean = func_params.get("use_rolling_mean", False)
    estimate_KAW_psd = func_params.get("est_sig_yz_cond_moms", False)

    alpha_sigma = float(func_params.get("alpha_sigma", func_params.get("alpha", 3.0)))
    num_efoldings = int(func_params.get("num_efoldings", 3))

    if use_rolling_mean:
        if not isinstance(df_w.index, pd.DatetimeIndex):
            raise ValueError("df_w must have a DateTimeIndex when use_rolling_mean is True.")

        averaging_window = func_params["averaging_window"]
        step = func_params["step"]
        rolling_params = {"window": averaging_window, "center": True, "min_periods": 10}

        sigma_av_yz = np.nan

        # Here we smooth the coherence diagnostic at width alpha_sigma * (wavelet sigma)
        sigma_samples = (alpha_sigma * scale) / dt
        ker = gaussian_kernel_from_sigma_samples(sigma_samples, num_efoldings=num_efoldings)

        num_value = gaussian_smooth_1d_same(np.asarray(S3, dtype=float), ker)
        den_value = gaussian_smooth_1d_same(np.asarray(S0, dtype=float), ker)
        sigma = num_value / den_value

        index_coh = np.abs(sigma) > func_params["coh_th"]
        index_non_coh = ~index_coh

        sigma_av_xy = (pd.Series(sigma, index=df_w.index)).resample(step).mean()

        coh_counts     = (pd.Series(index_coh, index=df_w.index)).rolling(**rolling_params).sum()
        non_coh_counts = (pd.Series(~index_coh, index=df_w.index)).rolling(**rolling_params).sum()
        window_size    = coh_counts + non_coh_counts

        if estimate_KAW_psd:
            sigma_samples_yz = (1.0 * scale) / dt
            ker_yz = gaussian_kernel_from_sigma_samples(sigma_samples_yz, num_efoldings=num_efoldings)

            num_value_yz = gaussian_smooth_1d_same(np.asarray(Syz, dtype=float), ker_yz)
            den_value_yz = gaussian_smooth_1d_same(np.asarray(S0_full, dtype=float), ker_yz)
            sigma_yz_ind = num_value_yz / den_value_yz

            index_p = sigma_yz_ind > func_params["sigma_yz_thresh"]["pos"]
            index_n = sigma_yz_ind < func_params["sigma_yz_thresh"]["neg"]
            index_z = np.abs(sigma_yz_ind) < func_params["sigma_yz_thresh"]["zer"]
        else:
            index_p = index_n = index_z = None

        k_df_roll = k_df.rolling(**rolling_params).mean().resample(step).mean()

        if index_per is not None:
            k_perp_per = (
                pd.Series(k_df["k_perp"], index=k_df.index)
                .where(pd.Series(index_per, index=df_w.index))
                .rolling(**rolling_params)
                .mean()
                .resample(step)
                .mean()
            )
            dfdk_perp_per = (
                pd.Series(k_df["dfdk_perp"], index=k_df.index)
                .where(pd.Series(index_per, index=df_w.index))
                .rolling(**rolling_params)
                .mean()
                .resample(step)
                .mean()
            )
        else:
            k_perp_per = np.nan
            dfdk_perp_per = np.nan

        if index_par is not None:
            k_par_par = (
                pd.Series(k_df["k_par"], index=k_df.index)
                .where(pd.Series(index_par, index=df_w.index))
                .rolling(**rolling_params)
                .mean()
                .resample(step)
                .mean()
            )
            dfdk_par_par = (
                pd.Series(k_df["dfdk_par"], index=k_df.index)
                .where(pd.Series(index_par, index=df_w.index))
                .rolling(**rolling_params)
                .mean()
                .resample(step)
                .mean()
            )
        else:
            k_par_par = np.nan
            dfdk_par_par = np.nan

        sigma_xy = (
            pd.Series(S3, index=df_w.index).rolling(**rolling_params).mean()
            / pd.Series(S0, index=df_w.index).rolling(**rolling_params).mean()
        ).resample(step).mean()
        sigma_yz = (
            pd.Series(Syz, index=df_w.index).rolling(**rolling_params).mean()
            / pd.Series(S0_full, index=df_w.index).rolling(**rolling_params).mean()
        ).resample(step).mean()

        num_mean_per_yz = (
            pd.Series(Syz, index=df_w.index)
            .where(pd.Series(index_per, index=df_w.index))
            .rolling(**rolling_params)
            .mean()
        )
        den_mean_per_yz = (
            pd.Series(S0_full, index=df_w.index)
            .where(pd.Series(index_per, index=df_w.index))
            .rolling(**rolling_params)
            .mean()
        )
        sigma_yz_per = (num_mean_per_yz / den_mean_per_yz).resample(step).mean()

        num_mean_par_xy = (
            pd.Series(S3, index=df_w.index)
            .where(pd.Series(index_par, index=df_w.index))
            .rolling(**rolling_params)
            .mean()
        )
        den_mean_par_xy = (
            pd.Series(S0, index=df_w.index)
            .where(pd.Series(index_par, index=df_w.index))
            .rolling(**rolling_params)
            .mean()
        )
        sigma_xy_par = (num_mean_par_xy / den_mean_par_xy).resample(step).mean()

        num_mean_per_xy = (
            pd.Series(S3, index=df_w.index)
            .where(pd.Series(index_per, index=df_w.index))
            .rolling(**rolling_params)
            .mean()
        )
        den_mean_per_xy = (
            pd.Series(S0, index=df_w.index)
            .where(pd.Series(index_per, index=df_w.index))
            .rolling(**rolling_params)
            .mean()
        )
        sigma_xy_per = (num_mean_per_xy / den_mean_per_xy).resample(step).mean()

        index_mask = pd.Series(index_coh, index=df_w.index)
        df_ws_needed = df_w.where(index_mask)
        PSD_coh = compute_norm_psd(df_ws_needed, rolling_params, dt, coh_counts, window_size, step, psd_norm)

        index_mask = pd.Series(~index_coh, index=df_w.index)
        df_ws_needed = df_w.where(index_mask)
        PSD_non_coh = compute_norm_psd(df_ws_needed, rolling_params, dt, non_coh_counts, window_size, step, psd_norm)
        SDK_non_coh = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

        Trace_PSD = np.nansum([PSD_coh, PSD_non_coh], axis=0)
        SDK = calculate_wavelet_flatness(df_w, rolling_params, step)

        index_mask = pd.Series(~index_coh & index_per, index=df_w.index)
        df_ws_needed = df_w.where(index_mask)
        PSD_non_coh_per, non_coh_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
        SDK_non_coh_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

        index_mask = pd.Series(~index_coh & index_par, index=df_w.index)
        df_ws_needed = df_w.where(index_mask)
        PSD_non_coh_par, non_coh_par_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
        SDK_non_coh_par = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

        if estimate_KAW_psd:
            index_mask = pd.Series(index_p, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_p, sig_yz_p_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_p = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series(index_p & index_per, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_p_per, sig_yz_p_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_p_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series(index_n, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_n, sig_yz_n_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_n = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series(index_n & index_per, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_n_per, sig_yz_n_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_n_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series(index_n | index_p, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_pn, sig_yz_pn_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_pn = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series((index_n | index_p) & index_per, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_pn_per, sig_yz_pn_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_pn_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series(index_z, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_z, sig_yz_z_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_z = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

            index_mask = pd.Series(index_z & index_per, index=df_w.index)
            df_ws_needed = df_w.where(index_mask)
            PSD_sig_yz_z_per, sig_yz_z_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
            SDK_sig_yz_z_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
        else:
            PSD_sig_yz_n = PSD_sig_yz_p = PSD_sig_yz_z = PSD_sig_yz_pn = np.nan
            PSD_sig_yz_z_per = PSD_sig_yz_p_per = PSD_sig_yz_n_per = PSD_sig_yz_pn_per = np.nan
            SDK_sig_yz_n = SDK_sig_yz_p = SDK_sig_yz_z = SDK_sig_yz_pn = np.nan
            SDK_sig_yz_z_per = SDK_sig_yz_p_per = SDK_sig_yz_n_per = SDK_sig_yz_pn_per = np.nan
            sig_yz_p_counts = sig_yz_n_counts = sig_yz_z_counts = sig_yz_pn_counts = np.nan
            sig_yz_p_per_counts = sig_yz_n_per_counts = sig_yz_z_per_counts = sig_yz_pn_per_counts = np.nan

        return {
            "k": k_df_roll["k"].values,
            "k_perp": k_df_roll["k_perp"].values,
            "k_par": k_df_roll["k_par"].values,
            "dfdk_perp": k_df_roll["dfdk_perp"].values,
            "dfdk_par": k_df_roll["dfdk_par"].values,
            "k_perp_per": np.asarray(k_perp_per) if isinstance(k_perp_per, (pd.Series, pd.DataFrame)) else k_perp_per,
            "k_par_par": np.asarray(k_par_par) if isinstance(k_par_par, (pd.Series, pd.DataFrame)) else k_par_par,
            "dfdk_perp_per": np.asarray(dfdk_perp_per) if isinstance(dfdk_perp_per, (pd.Series, pd.DataFrame)) else dfdk_perp_per,
            "dfdk_par_par": np.asarray(dfdk_par_par) if isinstance(dfdk_par_par, (pd.Series, pd.DataFrame)) else dfdk_par_par,
            "sigma_xy_av": sigma_av_xy,
            "sigma_yz_av": sigma_av_yz,
            "sigma_xy": sigma_xy,
            "sigma_xy_par": sigma_xy_par,
            "sigma_xy_per": sigma_xy_per,
            "sigma_yz": sigma_yz,
            "sigma_yz_per": sigma_yz_per,
            "PSD_Trace": Trace_PSD,
            "PSD_coh": PSD_coh,
            "PSD_non_coh": PSD_non_coh,
            "SDK": SDK,
            "SDK_non_coh": SDK_non_coh,
            "SDK_non_coh_par": SDK_non_coh_par,
            "SDK_non_coh_per": SDK_non_coh_per,
            "SDK_sig_yz_n": SDK_sig_yz_n,
            "SDK_sig_yz_p": SDK_sig_yz_p,
            "SDK_sig_yz_z": SDK_sig_yz_z,
            "SDK_sig_yz_pn": SDK_sig_yz_pn,
            "SDK_sig_yz_z_per": SDK_sig_yz_z_per,
            "SDK_sig_yz_p_per": SDK_sig_yz_p_per,
            "SDK_sig_yz_n_per": SDK_sig_yz_n_per,
            "SDK_sig_yz_pn_per": SDK_sig_yz_pn_per,
            "PSD_non_coh_per": PSD_non_coh_per,
            "counts_non_coh_per": non_coh_per_counts,
            "PSD_non_coh_par": PSD_non_coh_par,
            "counts_non_coh_par": non_coh_par_counts,
            "PSD_sig_yz_n": PSD_sig_yz_n,
            "PSD_sig_yz_p": PSD_sig_yz_p,
            "PSD_sig_yz_z": PSD_sig_yz_z,
            "PSD_sig_yz_pn": PSD_sig_yz_pn,
            "PSD_sig_yz_z_per": PSD_sig_yz_z_per,
            "PSD_sig_yz_p_per": PSD_sig_yz_p_per,
            "PSD_sig_yz_n_per": PSD_sig_yz_n_per,
            "PSD_sig_yz_pn_per": PSD_sig_yz_pn_per,
            "counts_sig_yz_p": sig_yz_p_counts,
            "counts_sig_yz_n": sig_yz_n_counts,
            "counts_sig_yz_z": sig_yz_z_counts,
            "counts_sig_yz_pn": sig_yz_pn_counts,
            "counts_sig_yz_p_per": sig_yz_p_per_counts,
            "counts_sig_yz_n_per": sig_yz_n_per_counts,
            "counts_sig_yz_z_per": sig_yz_z_per_counts,
            "counts_sig_yz_pn_per": sig_yz_pn_per_counts,
            "counts_par": ((pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).sum()).resample(step).sum(),
            "counts_per": ((pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).sum()).resample(step).sum(),
            "counts_coh": coh_counts.resample(step).mean(),
            "counts_non_coh": non_coh_counts.resample(step).mean(),
            "counts": window_size.resample(step).mean(),
            "coh_thresh": func_params["coh_th"],
        }

    # Non-rolling path (kept)
    sigma_samples = (func_params["alpha"] * scale) / dt
    ker = gaussian_kernel_from_sigma_samples(sigma_samples, num_efoldings=num_efoldings)

    num_value = gaussian_smooth_1d_same(np.asarray(S3, dtype=float), ker)
    den_value = gaussian_smooth_1d_same(np.asarray(S0, dtype=float), ker)
    sigma = num_value / den_value

    sigma_xy = np.nanmean(S3) / np.nanmean(S0)
    sigma_xy_par = np.nanmean(S3[index_par]) / np.nanmean(S0[index_par]) if index_par is not None else np.nan
    sigma_xy_per = np.nanmean(S3[index_per]) / np.nanmean(S0[index_per]) if index_per is not None else np.nan

    sigma_yz = np.nanmean(Syz) / np.nanmean(S0) if Syz is not None else np.nan
    sigma_yz_par = np.nanmean(Syz[index_par]) / np.nanmean(S0[index_par]) if (Syz is not None and index_par is not None) else np.nan
    sigma_yz_per = np.nanmean(Syz[index_per]) / np.nanmean(S0[index_per]) if (Syz is not None and index_per is not None) else np.nan

    index_coh = np.abs(sigma) > func_params["coh_th"]
    index_non_coh = ~index_coh

    coherent_sum = np.nanmean(np.real(df_w.iloc[index_coh].values * np.conj(df_w.iloc[index_coh].values)), axis=0).sum()
    non_coherent_sum = np.nanmean(np.real(df_w.iloc[index_non_coh].values * np.conj(df_w.iloc[index_non_coh].values)), axis=0).sum()

    PSD_coh = np.sum(index_coh) / len(index_coh) * coherent_sum
    PSD_non_coh = np.sum(index_non_coh) / len(index_coh) * non_coherent_sum
    Trace_PSD = PSD_coh + PSD_non_coh

    k_mean = np.nanmean(k_df["k"].values) if "k" in k_df else np.nan
    kper_mean = np.nanmean(k_df["k_perp"].values) if "k_perp" in k_df else np.nan
    kpar_mean = np.nanmean(k_df["k_par"].values) if "k_par" in k_df else np.nan
    dfdkper_mean = np.nanmean(k_df["dfdk_perp"].values) if "dfdk_perp" in k_df else np.nan
    dfdkkpar_mean = np.nanmean(k_df["dfdk_par"].values) if "dfdk_par" in k_df else np.nan

    return {
        "k": k_mean,
        "k_perp": kper_mean,
        "k_par": kpar_mean,
        "dfdk_perp": dfdkper_mean,
        "dfdk_par": dfdkkpar_mean,
        "sigma_xy": sigma_xy,
        "sigma_xy_par": sigma_xy_par,
        "sigma_xy_per": sigma_xy_per,
        "sigma_yz": sigma_yz,
        "sigma_yz_par": sigma_yz_par,
        "sigma_yz_per": sigma_yz_per,
        "PSD_coh": PSD_coh,
        "PSD_non_coh": PSD_non_coh,
        "PSD_Trace": Trace_PSD,
        "counts_par": np.nansum(index_par) if index_par is not None else np.nan,
        "counts_per": np.nansum(index_per) if index_per is not None else np.nan,
        "counts_coh": np.nansum(index_coh),
        "counts_non_coh": np.nansum(index_non_coh),
        "counts_Trace": len(df_w.dropna()),
        "coh_thresh": func_params["coh_th"],
    }


# ==============================================================
#  Merge: coherence + compress + sfuncs
# ==============================================================

def return_desired_quants(
    df_w,
    df_mod,
    S0,
    S3,
    S0_full,
    Syz,
    VBangles,
    k_df,
    dt,
    scale,
    psd_norm,
    func_params=None,
):
    if func_params is None:
        func_params = {}

    if func_params.get("estimate_PSDs", False) or func_params.get("estimate_coh_coeffs", False):
        if func_params.get("use_rolling_mean", False):
            index_per = VBangles > func_params["per_thresh"]
            index_par = VBangles < func_params["par_thresh"]
        else:
            index_per = (np.where(VBangles > func_params["per_thresh"])[0]).astype(np.int64)
            index_par = (np.where(VBangles < func_params["par_thresh"])[0]).astype(np.int64)
    else:
        index_per = None
        index_par = None

    coh_res = do_coh_analysis(k_df, df_w, S0, S3, S0_full, Syz, index_par, index_per, dt, scale, psd_norm, func_params=func_params)
    sf_res = est_sfuncs(df_w, df_mod, index_par, index_per, scale, dt, func_params=func_params)
    comp_res = est_compress(df_mod, df_w, index_par, index_per, dt, psd_norm, func_params=func_params)

    return {**coh_res, **comp_res, **sf_res}


def define_W_df(B_index, R, T, N):
    return pd.DataFrame({"DateTime": B_index, "R": R, "T": T, "N": N}).set_index("DateTime")


# ==============================================================
#  anisotropy_coherence (REVISED: no shared DF mutation + no df.apply smoothing)
# ==============================================================

def anisotropy_coherence(
    B_df,
    V_df,
    mag_V_mod_TH,
    field_flag,
    E_df=None,
    Np_df=None,
    method="min_var",
    func_params=None,
    f_dict=None,
    frame_flag=None,
    mag_Va_ts=None,
    sigma_Va_ts=None,
):
    if func_params is None:
        func_params = {}

    # -----------------------------
    # Column normalization to R,T,N
    # -----------------------------
    if B_df.columns[0] == "Bx":
        B_df = B_df.rename(columns={"Bx": "R", "By": "T", "Bz": "N"})
        V_df = V_df.rename(columns={"Vx": "R", "Vy": "T", "Vz": "N"})

        if (method == "TN_only") and (E_df is not None):
            E_df["R"] = 0 * E_df["Ey"]
            E_df["T"] = E_df["Ex"]
            E_df["N"] = -E_df["Ey"]
            E_df.drop(columns=["Ex", "Ey"], inplace=True, errors="ignore")
        else:
            E_df = None if E_df is None else E_df.rename(columns={"Ex": "R", "Ey": "T", "Ez": "N"})
    else:
        B_df = B_df.rename(columns={"Br": "R", "Bt": "T", "Bn": "N"})
        V_df = V_df.rename(columns={"Vr": "R", "Vt": "T", "Vn": "N"})
        E_df = None if E_df is None else E_df.rename(columns={"Er": "R", "Et": "T", "En": "N"})

    dt_B, dt_V = func.find_cadence(B_df), func.find_cadence(V_df)
    dt_E = func.find_cadence(E_df) if E_df is not None else None

    if E_df is not None and dt_E != dt_B:
        try:
            E_df, B_df = func.synchronize_dfs(E_df, B_df, True)
        except Exception:
            E_df = func.newindex(E_df, B_df.index)

    if dt_V != dt_B:
        B_df, V_df = func.synchronize_dfs(B_df, V_df, True)

    dt = dt_E if dt_E is not None else dt_B

    omega0 = float(func_params.get("omega0", 6.0))

    # -----------------------------
    # Compute CWT once
    # -----------------------------
    Wvec, scales, freqs, psd_norm, coi = estimate_cwt(
        E_df if E_df is not None else B_df[["R", "T", "N"]],
        dt,
        nv=func_params["nv"],
        omega0=omega0,
        min_freq=func_params["min_freq"],
        max_freq=func_params["max_freq"],
    )

    # -----------------------------
    # Optional "mod" CWT
    # -----------------------------
    if func_params.get("est_mod", False):
        if Np_df is None:
            Wmod, _, _, _, _ = estimate_cwt(
                pd.DataFrame({"Mod": np.linalg.norm(B_df[["R", "T", "N"]].values, axis=1)}, index=B_df.index),
                dt,
                nv=func_params["nv"],
                omega0=omega0,
                min_freq=func_params["min_freq"],
                max_freq=func_params["max_freq"],
            )
        else:
            if isinstance(Np_df, pd.DataFrame) and "np" in Np_df.columns:
                mod_series = Np_df["np"].values
            elif isinstance(Np_df, pd.Series):
                mod_series = Np_df.values
            else:
                mod_series = np.asarray(Np_df).reshape(-1)

            Wmod, _, _, _, _ = estimate_cwt(
                pd.DataFrame({"Mod": mod_series}, index=B_df.index),
                dt,
                nv=func_params["nv"],
                omega0=omega0,
                min_freq=func_params["min_freq"],
                max_freq=func_params["max_freq"],
            )
    else:
        Wmod = None

    # -----------------------------
    # Immutable base arrays (NO mutation)
    # -----------------------------
    idx = B_df.index
    B_base = B_df[["R", "T", "N"]].to_numpy(dtype=float, copy=False)
    V_base = V_df[["R", "T", "N"]].to_numpy(dtype=float, copy=False)

    np_base = None
    if Np_df is not None:
        if isinstance(Np_df, pd.DataFrame) and "np" in Np_df.columns:
            np_base = Np_df["np"].to_numpy(dtype=float, copy=False)
        elif isinstance(Np_df, pd.Series):
            np_base = Np_df.to_numpy(dtype=float, copy=False)
        else:
            np_base = np.asarray(Np_df, dtype=float).reshape(-1)

    if np_base is None and "np" in V_df.columns:
        np_base = V_df["np"].to_numpy(dtype=float, copy=False)

    # -----------------------------
    # Per-scale worker (no DF mutation, no heavy argument pickling)
    # -----------------------------
    bg_alpha = float(func_params.get("alpha_bg", func_params.get("bg_alpha", 5.0)))
    bg_num_efoldings = int(func_params.get("bg_num_efoldings", func_params.get("num_efoldings", 3)))
    local_V_mode = str(func_params.get("local_V_mode", "same")).lower()

    V_floor = float(func_params.get("V_floor", 1e-6))
    cphi_floor = float(func_params.get("cphi_floor", 1e-3))

    cphi_mode = str(func_params.get("cphi_mode", "measured")).lower()
    use_Vkperp = bool(func_params.get("use_Vkperp", True)) and bool(func_params.get("rotate_to_k", False))

    kpar_mode = str(func_params.get("kpar_mode", "mth")).lower()

    def _one_scale(ii, scale_sigma, freq_hz):
        try:
            # Background smoothing width = alpha_bg * sigma_wavelet
            sigma_time = bg_alpha * float(scale_sigma)
            sigma_samples = sigma_time / dt

            ker = gaussian_kernel_from_sigma_samples(sigma_samples, num_efoldings=bg_num_efoldings)

            # Smooth B
            B_sm_arr = gaussian_smooth_2d_same(B_base, ker)

            # Smooth V
            if local_V_mode == "same":
                V_sm_arr = gaussian_smooth_2d_same(V_base, ker)
            elif local_V_mode == "none":
                V_sm_arr = V_base
            elif local_V_mode == "alpha":
                alpha_V = float(func_params.get("alpha_V", bg_alpha))
                sigma_samples_V = (alpha_V * float(scale_sigma)) / dt
                kerV = gaussian_kernel_from_sigma_samples(sigma_samples_V, num_efoldings=bg_num_efoldings)
                V_sm_arr = gaussian_smooth_2d_same(V_base, kerV)
            else:
                V_sm_arr = V_base

            # Smooth density if present
            np_sm = None
            if np_base is not None:
                np_sm = gaussian_smooth_1d_same(np_base, ker)

            Va_loc, sigma_loc = compute_local_Va_sigma(B_sm_arr, np_sm=np_sm, frame_flag=frame_flag)
            b_hat = _safe_unit(B_sm_arr)

            sigmaVa = sigma_loc * Va_loc
            V_eff_vec = V_sm_arr + (sigmaVa[:, None] * b_hat)
            Vmod_loc = np.linalg.norm(V_eff_vec, axis=1)
            Vmod_loc[~np.isfinite(Vmod_loc)] = np.nan

            # Build wavelet coefficient DF for this scale
            df_w = define_W_df(idx, Wvec["R"][ii], Wvec["T"][ii], Wvec["N"][ii])

            df_mod = Wmod["Mod"][ii] if Wmod is not None else np.nan

            if func_params.get("do_coherence_analysis", True):
                df_w, VBangles, S3, S0, Syz, S0_full, kin = coherence_analysis(
                    B_sm_arr,
                    V_sm_arr,
                    df_w,
                    method,
                    func_params=func_params,
                )
            else:
                dot = np.einsum("ij,ij->i", _safe_unit(B_sm_arr), _safe_unit(V_sm_arr))
                VBangles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
                S3 = S0 = Syz = S0_full = None
                kin = {
                    "phi": np.zeros(len(idx), dtype=float),
                    "V_par_local": np.einsum("ij,ij->i", V_sm_arr, b_hat),
                    "V_perp": np.sqrt(
                        np.maximum(
                            0.0,
                            np.linalg.norm(V_sm_arr, axis=1) ** 2
                            - np.einsum("ij,ij->i", V_sm_arr, b_hat) ** 2,
                        )
                    ),
                    "V_kperp": np.nan * np.ones(len(idx)),
                    "cphi": np.nan * np.ones(len(idx)),
                }

            VBangles = np.asarray(VBangles, dtype=float)
            VBangles[VBangles > 90] = 180 - VBangles[VBangles > 90]

            omega = 2.0 * np.pi * float(freq_hz)

            V_par_local = np.asarray(kin["V_par_local"], dtype=float)
            V_perp_mag = np.asarray(kin["V_perp"], dtype=float)
            V_kperp = np.asarray(kin["V_kperp"], dtype=float)
            cphi_meas = np.asarray(kin["cphi"], dtype=float)

            # Modified TH: k_par from omega / |V_par +/- Va|
            if kpar_mode in ("simple", "no_va", "nova", "none"):
                V_par_eff = V_par_local
            else:
                V_par_eff = V_par_local + sigmaVa

            V_par_eff_abs = np.abs(V_par_eff)
            V_par_eff_abs[V_par_eff_abs < V_floor] = np.nan

            # k_perp needs a projection factor
            if use_Vkperp and np.any(np.isfinite(V_kperp)):
                V_perp_proj = np.abs(V_kperp)
            else:
                if cphi_mode == "axisym":
                    cphi_eff = (2.0 / np.pi) * np.ones_like(V_perp_mag, dtype=float)
                elif cphi_mode == "unity":
                    cphi_eff = np.ones_like(V_perp_mag, dtype=float)
                else:
                    cphi_eff = np.abs(cphi_meas)
                    cphi_eff = np.nan_to_num(cphi_eff, nan=1.0)
                    cphi_eff = np.maximum(cphi_eff, cphi_floor)
                V_perp_proj = np.abs(V_perp_mag) * cphi_eff

            V_perp_proj = np.asarray(V_perp_proj, dtype=float)
            V_perp_proj[V_perp_proj < V_floor] = np.nan

            Vmod_abs = np.abs(Vmod_loc)
            Vmod_abs[Vmod_abs < V_floor] = np.nan

            k_par_ts = omega / V_par_eff_abs
            k_ts = omega / Vmod_abs
            k_perp_ts = omega / V_perp_proj

            dfdk_perp_ts = V_perp_proj / (2.0 * np.pi)
            dfdk_par_ts = V_par_eff_abs / (2.0 * np.pi)

            k_df_raw = pd.DataFrame(
                {
                    "k": k_ts,
                    "k_perp": k_perp_ts,
                    "k_par": k_par_ts,
                    "dfdk_perp": dfdk_perp_ts,
                    "dfdk_par": dfdk_par_ts,
                },
                index=idx,
            )

            est_quants = return_desired_quants(
                df_w,
                df_mod,
                S0,
                S3,
                S0_full,
                Syz,
                VBangles,
                k_df_raw,
                dt=dt,
                scale=float(scale_sigma),
                psd_norm=psd_norm,
                func_params=func_params,
            )

            if func_params.get("return_coeffs", False) is False:
                return est_quants, None, None, None, None, None

            return est_quants, VBangles, df_w.values.T, S3, S0, Syz

        except Exception:
            traceback.print_exc()
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # -----------------------------
    # Parallel execution
    # -----------------------------
    prefer = str(func_params.get("prefer_parallel", "processes")).lower()
    if prefer in ("threads", "thread", "threading"):
        results = Parallel(n_jobs=func_params["njobs"], prefer="threads")(
            delayed(_one_scale)(ii, scales[ii], freqs[ii]) for ii in tqdm(range(len(scales)), total=len(scales))
        )
    else:
        results = Parallel(n_jobs=func_params["njobs"])(
            delayed(_one_scale)(ii, scales[ii], freqs[ii]) for ii in tqdm(range(len(scales)), total=len(scales))
        )

    est_quants, VBangles, df_w_out, S3_out, S0_out, Syz_out = zip(*results)

    # -----------------------------
    # Pack outputs
    # -----------------------------
    if f_dict is None:
        if func_params.get("use_rolling_mean", False):
            f_dict = {field_flag: {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}}
            f_dict["freqs"] = freqs
            f_dict["scales"] = scales
            f_dict["Wave_coeffs"] = df_w_out
            f_dict["S3_ts"] = S3_out
            f_dict["S0_ts"] = S0_out
            f_dict["VB_ts"] = VBangles
            f_dict["flag"] = method
        else:
            f_dict = {field_flag: {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}}
    else:
        if func_params.get("use_rolling_mean", False):
            f_dict[field_flag] = {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}
        else:
            f_dict[field_flag] = {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}

    return f_dict


# ==============================================================
#  Keep original call style in run_coh_pipeline
# ==============================================================

class polarization_analysis:
    anisotropy_coherence = staticmethod(anisotropy_coherence)
    build_V_mod_TH = staticmethod(build_V_mod_TH)


# ==============================================================
#  run_coh_pipeline (kept compatible; passes Np_df into anisotropy)
# ==============================================================

def run_coh_pipeline(
    load_path,
    folder_name,
    func_params,
    overwrite_files=False,
    overwrite_by_date=False,
    overwrite_before="2026-01-09",
):
    import logging

    BG_WHITE = globals().get("BG_WHITE", "")
    RESET = globals().get("RESET", "")

    def should_overwrite(outfile):
        if not outfile.exists():
            return True
        if bool(overwrite_files):
            return True
        if bool(overwrite_by_date):
            cutoff = datetime.strptime(overwrite_before, "%Y-%m-%d")
            mtime = datetime.fromtimestamp(outfile.stat().st_mtime)
            return mtime < cutoff
        return False

    finnames = func.load_files(load_path, "final.pkl")
    gennames = func.load_files(load_path, "general.pkl")
    signames = func.load_files(load_path, "sig_c_sig_r.pkl")
    maggaps = func.load_files(load_path, "mag_gaps.pkl")
    qtngaps = func.load_files(load_path, "qtn_gaps.pkl")
    pargaps = func.load_files(load_path, "par_gaps.pkl")
    elecgaps = func.load_files(load_path, "elec_gaps.pkl")

    averaging_window = func_params["averaging_window"]
    step = func_params["step"]
    rolling_params = {"window": averaging_window, "center": True, "min_periods": 10}

    mu0 = constants.mu_0
    m_p = constants.m_p

    for jj in range(len(finnames)):
        try:
            func.progress_bar(jj, len(finnames))

            foldername = (
                f"coh_step_{func_params['step']}_window_{func_params['averaging_window']}_"
                f"method_{list(func_params['method'].keys())[0]}_per_ang_{func_params['per_thresh']}.pkl"
            )
            path0 = finnames[jj].replace("final.pkl", "")
            outdir = Path(path0) / folder_name
            outfile = outdir / foldername

            if should_overwrite(outfile):
                if not outdir.exists():
                    logging.info(BG_WHITE + "Creating new folder %s" + RESET, outdir)
                    outdir.mkdir(parents=True, exist_ok=True)
                else:
                    if outfile.exists():
                        logging.info(BG_WHITE + "Overwriting file %s" + RESET, outfile)
                    else:
                        logging.info(BG_WHITE + "Creating new file %s" + RESET, outfile)

                fin = pd.read_pickle(finnames[jj])
                sig = pd.read_pickle(signames[jj])
                mag_gaps = pd.read_pickle(maggaps[jj])
                qtn_gaps = pd.read_pickle(qtngaps[jj])
                par_gaps = pd.read_pickle(pargaps[jj])
                elec_gaps = pd.read_pickle(elecgaps[jj])

                if fin.get("E") is None:
                    n_index = fin["Mag"]["B_resampled"].index
                    E_df = None
                    B_df = fin["Mag"]["B_resampled"]
                else:
                    E_df = fin["E"]
                    n_index = E_df.index
                    B_df = func.newindex(fin["Mag"]["B_resampled"], n_index)
                    E_df, B_df = func.synchronize_dfs(E_df, B_df, True)

                V_df = fin["Par"]["V_resampled"]
                sig, V_df = func.synchronize_dfs(sig, V_df, True)

                if "Vr" in V_df:
                    frame_flag = "RTN"
                    B_df_labs = ["Br", "Bt", "Bn"]
                    V_df_labs = ["Vr", "Vt", "Vn"]
                    V_df[V_df_labs] = V_df[V_df_labs] - V_df[["sc_vel_r", "sc_vel_t", "sc_vel_n"]].values
                    Br_sign = np.sign(B_df["Br"].rolling("10min", center=True).mean())

                    coh_dicts = {}
                    mag_Va = None
                    V_MTH_extra = None
                    V_sc_rem = {"RTN": V_df[V_df_labs]}

                    Np_local = None
                    if "np" in V_df.columns:
                        Np_local = V_df[["np"]]

                else:
                    frame_flag = "sc"
                    B_df_labs = ["Bx", "By", "Bz"]
                    V_df_labs = ["Vx", "Vy", "Vz"]

                    Br_sign = -np.sign(B_df["Bz"].rolling("10min", center=True).mean())

                    V_df = fin["Par"]["V_resampled"][V_df_labs + ["np", "Vth"]].interpolate()
                    V_df, sig = func.synchronize_dfs(V_df, sig, True)
                    B_df, V_df = func.synchronize_dfs(B_df, V_df, True)
                    Dist = func.newindex(fin["Par"]["V_resampled"]["Dist_au"], n_index)

                    V_mod_TH, V_sc_rem, V_MTH_extra = polarization_analysis.build_V_mod_TH(
                        B_df.copy(),
                        V_df.copy(),
                        return_addit=True,
                    )

                    mag_Va = pd.DataFrame(V_mod_TH["|Va|"], index=V_mod_TH.index, columns=["|Va|"])
                    mag_V_mod_TH = pd.DataFrame(V_mod_TH["|V_mod_TH|"], index=V_mod_TH.index, columns=["|V_mod_TH|"])

                    del fin

                    coh_dicts = {}

                    np_mean = V_df["np"].rolling("30s", center=True).mean().interpolate()
                    kinet_normal = func.newindex(1e-15 / np.sqrt(mu0 * np_mean * m_p), B_df.index).values

                    Bmag = np.sqrt(
                        B_df[B_df_labs[0]] ** 2
                        + B_df[B_df_labs[1]] ** 2
                        + B_df[B_df_labs[2]] ** 2
                    )

                    rho_ci, _, _, _ = func.estimate_rho_i(V_df["Vth"], Bmag)

                    del np_mean, kinet_normal

                    B_df, sig = func.synchronize_dfs(B_df, sig, True)

                    coh_dicts["sig_r"] = (sig.sigma_r).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["d"] = (Dist).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["beta"] = (sig.beta).rolling(**rolling_params).mean().resample(step).mean()

                    coh_dicts["di"] = (sig.d_i).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["VB"] = (sig.VB).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["VB_std"] = (sig.VB).rolling(**rolling_params).std().resample(step).mean()
                    coh_dicts["Vsw"] = (sig.Vsw).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["Vsw_Va"] = (mag_V_mod_TH).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["rho_ci"] = ((sig.d_i) * np.sqrt(sig.beta)).rolling(**rolling_params).mean().resample(step).mean()

                    coh_dicts["Vrel_par"] = (V_MTH_extra["Vrel_par"]).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["Vrel_perp"] = (V_MTH_extra["Vrel_perp"]).rolling(**rolling_params).mean().resample(step).mean()
                    coh_dicts["sigma_Va"] = (V_MTH_extra["sigma_Va"]).rolling(**rolling_params).mean().resample(step).mean()

                    Np_local = V_df[["np"]] if "np" in V_df.columns else None

                num = sig["dva_r"] * sig["dv_r"] + sig["dva_t"] * sig["dv_t"] + sig["dva_n"] * sig["dv_n"]
                den_1 = sig["dva_r"] ** 2 + sig["dva_t"] ** 2 + sig["dva_n"] ** 2
                den_2 = sig["dv_r"] ** 2 + sig["dv_t"] ** 2 + sig["dv_n"] ** 2

                sig_num = (num.rolling(func_params["averaging_window"], center=True).mean()).resample(func_params["step"]).mean()
                sig_den1 = (den_1.rolling(func_params["averaging_window"], center=True).mean()).resample(func_params["step"]).mean()
                sig_den2 = (den_2.rolling(func_params["averaging_window"], center=True).mean()).resample(func_params["step"]).mean()

                coh_dicts["sig_c_v2"] = np.abs(2 * sig_num / (sig_den1 + sig_den2))

                del sig

                for method in list(func_params["method"].keys()):
                    try:
                        for mm, field in enumerate(func_params["method"][method]):
                            print("Working on:", field, ". Coh method:", method)

                            E_sw = globals().get("E_sw", None)

                            coh_dict = polarization_analysis.anisotropy_coherence(
                                B_df,
                                V_sc_rem[V_df_labs] if isinstance(V_sc_rem, dict) else V_sc_rem[V_df_labs],
                                mag_V_mod_TH if "mag_V_mod_TH" in locals() else pd.DataFrame(np.nan, index=B_df.index, columns=["|V_mod_TH|"]),
                                field_flag=field,
                                E_df=E_df.copy() if field == "E" else (E_sw.copy() if (field == "E_sw" and E_sw is not None) else None),
                                Np_df=Np_local,
                                method=method,
                                func_params=func_params,
                                f_dict=None if mm == 0 else coh_dict,
                                frame_flag=frame_flag,
                                mag_Va_ts=mag_Va if mag_Va is not None else None,
                                sigma_Va_ts=V_MTH_extra["sigma_Va"] if V_MTH_extra is not None else None,
                            )

                    except Exception:
                        traceback.print_exc()
                        coh_dict = None

                    coh_dicts[method] = coh_dict

                func.savepickle(coh_dicts, str(outdir), foldername)

            else:
                logging.info(
                    BG_WHITE + "File already exists; overwrite conditions not met: %s" + RESET,
                    outfile,
                )

        except Exception:
            traceback.print_exc()
            pass
















# # ==============================================================
# #  FULL, COPY/PASTE PIPELINE (REVISED: scale-consistent local background)
# #  - Keeps ALL original saved variables/keys
# #  - Only ADDS new keys that were already present in your version:
# #      k_perp, k_par, dfdk_perp, dfdk_par (+ conditional/rolled variants)
# #
# #  CORE CHANGE (per your request):
# #    * All kinematic ingredients used for k-mapping are computed CONSISTENTLY
# #      at EACH wavelet scale via a background window W(tau)=alpha_bg*tau.
# #    * NO "conditional" backgrounds: conditioning is ONLY applied in averaging/PSD stage.
# #    * Removes the logically unclosed "mth_exact" correction for k_perp by default.
# #    * Adds cphi_mode option: measured | axisym (2/pi) | unity
# #    * Uses vectors for V_eff and V_mod to avoid basis/scale mismatch.
# #
# #  Assumes your existing modules exist: TurbPy as turb, general_functions as func
# # ==============================================================

# import warnings
# warnings.filterwarnings("ignore")

# import os
# import sys
# import traceback
# from pathlib import Path
# from datetime import datetime

# import numpy as np
# import pandas as pd
# import scipy
# from scipy import constants

# import joblib
# from joblib import Parallel, delayed
# from tqdm import tqdm

# import ssqueezepy

# sys.path.insert(0, str(_FUNCTIONS_DIR))
# import TurbPy as turb
# import general_functions as func


# # ==============================================================
# #  Wavelet / smoothing utilities
# # ==============================================================

# def estimate_cwt(signal,
#                  dt,
#                  nv=32,
#                  omega0=6,
#                  scale_type="log-piecewise",
#                  vectorized=True,
#                  l1_norm=False,
#                  min_freq=None,
#                  max_freq=None):
#     fs = 1 / dt
#     wavelet = ssqueezepy.Wavelet(("morlet", {"mu": omega0}))

#     if isinstance(signal, pd.DataFrame):
#         w_df = {}
#         for col in signal.columns:
#             W, scales = ssqueezepy.cwt(signal[col].values,
#                                        wavelet=wavelet,
#                                        scales=scale_type,
#                                        l1_norm=l1_norm,
#                                        fs=fs,
#                                        nv=nv,
#                                        vectorized=vectorized)

#             freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal[col]), fs)
#             scales = (omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * omega0**2))

#             nyquist = fs / 2.0
#             cutoff = 0.95 * nyquist
#             mask = freqs < cutoff

#             W = W[mask, :]
#             scales = scales[mask]
#             freqs = freqs[mask]

#             if min_freq is not None:
#                 indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
#                 W = W[indices, :]
#                 scales = scales[indices]
#                 freqs = freqs[indices]

#             w_df[col] = W
#     else:
#         W, scales = ssqueezepy.cwt(signal,
#                                    wavelet=wavelet,
#                                    scales=scale_type,
#                                    l1_norm=l1_norm,
#                                    fs=fs,
#                                    nv=nv,
#                                    vectorized=vectorized)

#         freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(signal), fs)
#         scales = (omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * omega0**2))

#         W = W[5:, :]
#         scales = scales[5:]
#         freqs = freqs[5:]

#         if min_freq is not None:
#             indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
#             W = W[indices, :]
#             scales = scales[indices]
#             freqs = freqs[indices]

#         w_df = W

#     coi = None
#     return w_df, scales, freqs, 2 * dt, coi


# def local_gaussian_averaging(signal, dt, scale, alpha=1.0, num_efoldings=3):
#     sigma_time = alpha * scale
#     sigma_samples = sigma_time / dt

#     half_width = int(np.ceil(num_efoldings * sigma_samples))
#     t_samples = np.arange(-half_width, half_width + 1)
#     kernel = np.exp(-(t_samples**2) / (2 * sigma_samples**2))
#     kernel /= kernel.sum()

#     return scipy.signal.convolve(signal, kernel, mode="same")


# def unit_vectors(df, prefix="", sufix="_hat", vector_cols=None):
#     if vector_cols is None:
#         raise ValueError("`vector_cols` must be supplied.")
#     arr = df[vector_cols].to_numpy(float, copy=False)
#     nrm = np.linalg.norm(arr, axis=1, keepdims=True)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         arr = arr / nrm
#     arr[~np.isfinite(arr)] = np.nan
#     df[[f"{prefix}{c}{sufix}" for c in vector_cols]] = arr
#     return df


# # ==============================================================
# #  Polarization utilities
# # ==============================================================

# def calculate_polarization_spectra(Bx, By):
#     PL = np.abs(Bx - 1j * By)**2
#     PR = np.abs(Bx + 1j * By)**2
#     return PL, PR


# # ==============================================================
# #  build_V_mod_TH (kept as-is; still used for OTHER saved metrics)
# # ==============================================================

# def build_V_mod_TH(
#     B, V,
#     axes: list[str] | None = None,
#     sc: str            = "PSP",
#     window: str        = "30min",
#     correct_sign: bool = True,
#     consider_Va:  bool = True,
#     consider_Vsc: bool = True,
#     return_addit: bool = False
# ):
#     import numpy as np
#     import pandas as pd

#     _C_ALFVEN = 21.82

#     if axes is None:
#         axes = ["r","t","n"] if "Br" in B.columns else ["x","y","z"]
#     is_rtn        = axes[0] == "r"
#     base_cols     = [f"V{a}" for a in axes]
#     sc_vel_cols   = [f"sc_vel_{a}" for a in ("r","t","n")]
#     B_cols        = [f"B{a}" for a in axes]

#     n_p_raw = V["np"].astype(float)
#     Np_mean = n_p_raw.rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)
#     Np_mean[Np_mean <= 0] = np.nan

#     B0_vec_df  = B[B_cols].rolling(window, center=True, min_periods=1).median()
#     B0_vec_arr = B0_vec_df.to_numpy(dtype=float)

#     B_inst_mag = np.linalg.norm(B[B_cols].to_numpy(dtype=float), axis=1)
#     B0_scalar_vals = pd.Series(B_inst_mag, index=B.index).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)

#     V_arr = V[base_cols].to_numpy(dtype=float)

#     if consider_Vsc and is_rtn and sc.upper() == "PSP":
#         if all(c in V.columns for c in sc_vel_cols):
#             V_sc_arr = V[sc_vel_cols].to_numpy(dtype=float)
#             V_arr    = V_arr - V_sc_arr

#     V_rel_df = pd.DataFrame(V_arr, index=V.index, columns=base_cols)

#     B0_vec_mag = np.linalg.norm(B0_vec_arr, axis=1)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         e_par = B0_vec_arr / B0_vec_mag[:, None]
#     e_par[~np.isfinite(e_par)] = 0.0

#     ref = np.array([1.0, 0.0, 0.0])
#     e_p1 = np.cross(e_par, ref)
#     idx_deg = np.linalg.norm(e_p1, axis=1) < 1e-6
#     if np.any(idx_deg):
#         e_p1[idx_deg] = np.cross(e_par[idx_deg], np.array([0.0, 1.0, 0.0]))
#     n1 = np.linalg.norm(e_p1, axis=1)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         e_p1 = e_p1 / n1[:, None]
#     e_p1[~np.isfinite(e_p1)] = 0.0
#     e_p2 = np.cross(e_par, e_p1)

#     v_par = np.einsum("ij,ij->i", V_arr, e_par)
#     v_p1  = np.einsum("ij,ij->i", V_arr, e_p1)
#     v_p2  = np.einsum("ij,ij->i", V_arr, e_p2)

#     Vrel_perp_mag = np.sqrt(v_p1**2 + v_p2**2)
#     Vrel_mag      = np.sqrt(v_par**2 + Vrel_perp_mag**2)

#     Va_bg = _C_ALFVEN * B0_scalar_vals / np.sqrt(Np_mean)
#     if not consider_Va:
#         Va_bg = 0.0 * Va_bg

#     Br_like = B0_vec_arr[:, 0] if is_rtn else -B0_vec_arr[:, 2]
#     sigma_bg = np.sign(Br_like)
#     sigma_bg[sigma_bg == 0] = 1.0
#     if not correct_sign:
#         sigma_bg = np.abs(sigma_bg)

#     v_par_eff = v_par + sigma_bg * Va_bg
#     V_mod_mag = np.sqrt(v_par_eff**2 + Vrel_perp_mag**2)

#     with np.errstate(divide='ignore', invalid='ignore'):
#         cos_theta = v_par / Vrel_mag
#     theta_VB = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

#     summary_df = pd.DataFrame({
#         "θ_VB"         : theta_VB,
#         "|Va|"         : Va_bg,
#         "|V_rel|"      : Vrel_mag,
#         "|V_mod_TH|"   : V_mod_mag,
#         "B0_scalar"    : B0_scalar_vals,
#         "B0_vec_mag"   : B0_vec_mag,
#         "Np_background": Np_mean
#     }, index=V.index)

#     if return_addit:
#         V_MTH_extra = pd.DataFrame({
#             "Vrel_par" : v_par,
#             "Vrel_perp": Vrel_perp_mag,
#             "Va_bg"    : Va_bg,
#             "sigma_bg" : sigma_bg,
#             "sigma_Va" : sigma_bg * Va_bg
#         }, index=V.index)
#         return summary_df, V_rel_df, V_MTH_extra

#     return summary_df, V_rel_df


# # ==============================================================
# #  coherence_analysis (kept diagnostics; returns kinematics)
# # ==============================================================

# def coherence_analysis(B0_f_o,
#                        V0_f_o,
#                        df_w,
#                        method,
#                        func_params=None):
#     if func_params is None:
#         func_params = {}
#     rotate_to_k = bool(func_params.get("rotate_to_k", False))

#     # Build unit vectors (written into new *_hat columns)
#     B0_f_o = unit_vectors(B0_f_o, prefix="B_0_", vector_cols=["R", "T", "N"])
#     V0_f_o = unit_vectors(V0_f_o, prefix="V_0_", vector_cols=["R", "T", "N"])

#     # Unit directions
#     b_hat = B0_f_o[["B_0_R_hat", "B_0_T_hat", "B_0_N_hat"]].to_numpy(dtype=float, copy=False)
#     v_hat = V0_f_o[["V_0_R_hat", "V_0_T_hat", "V_0_N_hat"]].to_numpy(dtype=float, copy=False)

#     # Angle between unit vectors (robustly clipped)
#     dot_bv = np.einsum("ij,ij->i", b_hat, v_hat)
#     VBangles = np.degrees(np.arccos(np.clip(dot_bv, -1.0, 1.0)))

#     # Raw vectors (for later projections)
#     B_vec = B0_f_o[["R", "T", "N"]].to_numpy(dtype=float, copy=False)
#     V_vec = V0_f_o[["R", "T", "N"]].to_numpy(dtype=float, copy=False)

#     # Numerical controls
#     eps = float(func_params.get("basis_eps", 1e-10))

#     # ----------------------------------------------------------
#     # Build a robust, right-handed orthonormal triad (e1,e2,b_hat)
#     # with e2 ~ (b_hat x v_hat) / |b_hat x v_hat|
#     # and fall back if b_hat || v_hat.
#     # ----------------------------------------------------------
#     ey = np.cross(b_hat, v_hat)
#     ey_norm = np.linalg.norm(ey, axis=1, keepdims=True)

#     bad = (~np.isfinite(ey_norm[:, 0])) | (ey_norm[:, 0] < eps)

#     if np.any(bad):
#         ref1 = np.array([1.0, 0.0, 0.0], dtype=float)
#         ref2 = np.array([0.0, 1.0, 0.0], dtype=float)

#         ey_f = np.cross(b_hat[bad], ref1)
#         n_f = np.linalg.norm(ey_f, axis=1, keepdims=True)

#         use_ref2 = (~np.isfinite(n_f[:, 0])) | (n_f[:, 0] < eps)
#         if np.any(use_ref2):
#             ey_f2 = np.cross(b_hat[bad][use_ref2], ref2)
#             ey_f[use_ref2] = ey_f2
#             n_f[use_ref2] = np.linalg.norm(ey_f[use_ref2], axis=1, keepdims=True)

#         with np.errstate(divide="ignore", invalid="ignore"):
#             ey_f = ey_f / n_f
#         ey_f[~np.isfinite(ey_f)] = np.nan

#         ey[bad] = ey_f
#         ey_norm[bad] = 1.0

#     with np.errstate(divide="ignore", invalid="ignore"):
#         ey = ey / ey_norm
#     ey[~np.isfinite(ey)] = np.nan

#     ex = np.cross(ey, b_hat)
#     ex_norm = np.linalg.norm(ex, axis=1, keepdims=True)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         ex = ex / ex_norm
#     ex[~np.isfinite(ex)] = np.nan

#     # ----------------------------------------------------------
#     # Project wavelet coefficients onto the perpendicular basis
#     # ----------------------------------------------------------
#     rtn = df_w[["R", "T", "N"]].to_numpy(dtype=np.complex128, copy=False)

#     Wx = np.einsum("ij,ij->i", ex, rtn).astype(np.complex128)
#     Wy = np.einsum("ij,ij->i", ey, rtn).astype(np.complex128)

#     df_w["Wx"] = Wx
#     df_w["Wy"] = Wy
#     df_w["W0"] = np.einsum("ij,ij->i", b_hat, rtn).astype(np.complex128)

#     phi = np.zeros(len(df_w), dtype=float)

#     # ----------------------------------------------------------
#     # Optional rotate-to-k within the perpendicular plane:
#     # choose phi that diagonalizes the 2x2 real covariance of (Wx,Wy)
#     # ----------------------------------------------------------
#     if rotate_to_k:
#         Pxx = (Wx.real**2 + Wx.imag**2)
#         Pyy = (Wy.real**2 + Wy.imag**2)
#         Pxy = (Wx.real * Wy.real + Wx.imag * Wy.imag)

#         phi = 0.5 * np.arctan2(2.0 * Pxy, (Pyy - Pxx))
#         sinφ = np.sin(phi)
#         cosφ = np.cos(phi)

#         W_perp2 = Wy * cosφ + Wx * sinφ
#         W_perp1 = -Wy * sinφ + Wx * cosφ

#         df_w["Wy"] = W_perp2
#         df_w["Wx"] = W_perp1

#         # Rotate basis vectors the same way
#         e1 = ex * cosφ[:, None] - ey * sinφ[:, None]

#         # Re-normalize e1 and rebuild e2 = b_hat x e1 to guarantee orthonormality
#         e1_norm = np.linalg.norm(e1, axis=1, keepdims=True)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             e1 = e1 / e1_norm
#         e1[~np.isfinite(e1)] = np.nan

#         e2 = np.cross(b_hat, e1)
#         e2_norm = np.linalg.norm(e2, axis=1, keepdims=True)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             e2 = e2 / e2_norm
#         e2[~np.isfinite(e2)] = np.nan

#     else:
#         e1 = ex
#         e2 = ey

#     # ----------------------------------------------------------
#     # Velocity projections: make V_perp consistent with b_hat
#     # ----------------------------------------------------------
#     V_par_local = np.einsum("ij,ij->i", V_vec, b_hat)

#     V_perp_vec = V_vec - V_par_local[:, None] * b_hat
#     V_perp = np.linalg.norm(V_perp_vec, axis=1)

#     V_p1_local = np.einsum("ij,ij->i", V_vec, e1)
#     V_kperp = np.einsum("ij,ij->i", V_vec, e2)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         cphi = np.abs(V_kperp) / V_perp
#     cphi[~np.isfinite(cphi)] = np.nan

#     # ----------------------------------------------------------
#     # Polarization spectra and derived diagnostics (as before)
#     # ----------------------------------------------------------
#     PL, PR = calculate_polarization_spectra(df_w["Wx"].values, df_w["Wy"].values)

#     helicity = PR - PL
#     trace_PSD = PR + PL
#     cross_term = 2.0 * np.imag(np.conj(df_w["Wy"]) * df_w["W0"])
#     power_sum = (np.abs(df_w["Wx"])**2 +
#                  np.abs(df_w["Wy"])**2 +
#                  np.abs(df_w["W0"])**2)

#     # Drop raw components; keep projections only
#     df_w.drop(columns=["R", "T", "N"], inplace=True, errors="ignore")

#     kinematics = {
#         "phi": phi,
#         "V_par_local": V_par_local,
#         "V_perp": V_perp,
#         "V_kperp": V_kperp,
#         "cphi": cphi
#     }

#     return df_w, VBangles, helicity, trace_PSD, cross_term, power_sum, kinematics


# # ==============================================================
# #  Structure functions
# # ==============================================================

# def est_sfuncs(df_w,
#                df_mod,
#                index_par,
#                index_per,
#                scale,
#                dts,
#                func_params=None):
#     def compute_SF(db, m, tau):
#         return np.nanmean(np.abs(db / np.sqrt(tau)) ** m)

#     if func_params is None:
#         func_params = {}

#     types = ["ov", "par", "per", "mod"]
#     sf_dict = {f"SF_{t}_{m}": [] for t in types for m in range(func_params["max_qorder"])}

#     db_vec  = np.sqrt(np.nansum(np.abs(df_w.values * np.conj(df_w.values)), axis=1))
#     db_mods = np.sqrt(np.abs(df_mod * np.conj(df_mod)))

#     if func_params.get("est_sfuncs", False):
#         for t in types:
#             db = {
#                 "par": db_vec[index_par] if index_par is not None else np.nan,
#                 "per": db_vec[index_per] if index_per is not None else np.nan,
#                 "ov" : db_vec,
#                 "mod": db_mods
#             }.get(t)

#             for m in range(func_params["max_qorder"]):
#                 sf_dict[f"SF_{t}_{m}"].append(compute_SF(db, m, scale))

#     return sf_dict


# # ==============================================================
# #  PSD / flatness / compressibility helpers
# # ==============================================================

# def compute_norm_psd(df_ws_needed, rolling_params, dt, counts, window_size, step, psd_norm):
#     if not isinstance(df_ws_needed, pd.DataFrame):
#         if hasattr(df_ws_needed, "to_frame"):
#             df_ws_needed = df_ws_needed.to_frame()
#         else:
#             df_ws_needed = pd.DataFrame(df_ws_needed)

#     psd_sum = psd_norm * (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean().sum(axis=1)
#     PSD = (psd_sum * (counts / window_size)).resample(step).mean()
#     return PSD


# def calculate_wavelet_flatness(df_ws, rolling_params, step):
#     if not isinstance(df_ws, pd.DataFrame):
#         df_ws = pd.DataFrame(df_ws)

#     power = np.abs(df_ws)**2
#     A2 = power.sum(axis=1)
#     A = np.sqrt(A2)

#     m2 = A.pow(2).rolling(**rolling_params).mean()
#     m4 = A.pow(4).rolling(**rolling_params).mean()

#     flatness = (m4 / m2.pow(2)).resample(step).mean()
#     return flatness


# def compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm):
#     if not isinstance(df_ws_needed, pd.DataFrame):
#         df_ws_needed = pd.DataFrame(df_ws_needed)

#     power = (df_ws_needed * np.conj(df_ws_needed)).rolling(**rolling_params).mean()
#     psd_sum = psd_norm * power.sum(axis=1)

#     if index_mask is not None:
#         counts = index_mask.rolling(**rolling_params).sum().resample(step).mean()
#     else:
#         counts = np.nan

#     PSD = psd_sum.resample(step).mean()
#     return PSD, counts


# def calculate_wavelet_flatness_from_power(power, rolling_params, step):
#     sum_power = power.sum(axis=1)
#     numerator = sum_power.pow(2).rolling(**rolling_params).mean()
#     denominator = sum_power.rolling(**rolling_params).mean().pow(2)
#     flatness = (numerator / denominator).resample(step).mean()
#     return flatness


# def compute_all_metrics(df_num,
#                         df_den,
#                         rolling_params,
#                         step,
#                         psd_norm,
#                         dt,
#                         compute_sdk=True,
#                         mask=None):
#     if not isinstance(df_num, pd.DataFrame):
#         df_num = pd.DataFrame(df_num)
#     if df_den is not None and not isinstance(df_den, pd.DataFrame):
#         df_den = pd.DataFrame(df_den)

#     if mask is not None:
#         df_num = df_num.where(mask)
#         if df_den is not None:
#             df_den = df_den.where(mask)

#     power_num = (df_num * np.conj(df_num)).apply(np.real)
#     rolling_power_num = power_num.rolling(**rolling_params).mean()
#     PSD_num = psd_norm * rolling_power_num.sum(axis=1)
#     PSD_num = PSD_num.resample(step).mean()

#     if compute_sdk:
#         SDK_num = calculate_wavelet_flatness_from_power(power_num, rolling_params, step)
#     else:
#         SDK_num = None

#     compress = None
#     if df_den is not None:
#         power_den = (df_den * np.conj(df_den)).apply(np.real)
#         ratio = power_num.sum(axis=1) / power_den.sum(axis=1)
#         compress = ratio.rolling(**rolling_params).mean().resample(step).mean()

#     return {
#         "PSD": PSD_num,
#         "SDK": SDK_num,
#         "compress": compress
#     }


# def est_compress(df_mod,
#                  df_w,
#                  index_par,
#                  index_per,
#                  dt,
#                  psd_norm,
#                  func_params=None):
#     if func_params is None:
#         func_params = {}

#     estimate_comp = func_params.get("estimate_comp", False)
#     use_rolling_mean = func_params.get("use_rolling_mean", False)

#     PSD_par = PSD_per = PSD_mod = PSD_mod_per = PSD_mod_par = np.nan
#     PSD_W0 = PSD_W0_per = PSD_W0_par = np.nan
#     PSD_Wxy = PSD_Wxy_per = PSD_Wxy_par = np.nan
#     compress_mod = compress_mod_par = compress_mod_per = np.nan
#     SDK_par = SDK_per = SDK_W0 = SDK_Wxy = SDK_mod_par = SDK_mod_per = SDK_mod = np.nan

#     if estimate_comp:
#         try:
#             if use_rolling_mean:
#                 averaging_window = func_params.get("averaging_window", "60s")
#                 step = func_params.get("step", "10s")
#                 rolling_params = {"window": averaging_window, "center": True, "min_periods": 10}

#                 df_mod_df = pd.DataFrame({"Mod": df_mod}, index=df_w.index)

#                 out_w0 = compute_all_metrics(df_num=df_w[["W0"]],
#                                              df_den=df_w,
#                                              rolling_params=rolling_params,
#                                              step=step,
#                                              psd_norm=psd_norm,
#                                              dt=dt,
#                                              compute_sdk=True,
#                                              mask=None)
#                 PSD_W0 = out_w0["PSD"]
#                 SDK_W0 = out_w0["SDK"]

#                 out_wxy = compute_all_metrics(df_num=df_w[["Wx", "Wy"]],
#                                               df_den=None,
#                                               rolling_params=rolling_params,
#                                               step=step,
#                                               psd_norm=psd_norm,
#                                               dt=dt,
#                                               compute_sdk=True,
#                                               mask=None)
#                 PSD_Wxy = out_wxy["PSD"]
#                 SDK_Wxy = out_wxy["SDK"]

#                 out_mod = compute_all_metrics(df_num=df_mod_df,
#                                               df_den=df_w,
#                                               rolling_params=rolling_params,
#                                               step=step,
#                                               psd_norm=psd_norm,
#                                               dt=dt,
#                                               compute_sdk=True,
#                                               mask=None)
#                 PSD_mod = out_mod["PSD"]
#                 SDK_mod = out_mod["SDK"]
#                 compress_mod = out_mod["compress"]

#                 mask_par = pd.Series(index_par, index=df_w.index) if index_par is not None else None
#                 mask_per = pd.Series(index_per, index=df_w.index) if index_per is not None else None

#                 out_par = compute_all_metrics(df_num=df_w,
#                                               df_den=None,
#                                               rolling_params=rolling_params,
#                                               step=step,
#                                               psd_norm=psd_norm,
#                                               dt=dt,
#                                               compute_sdk=True,
#                                               mask=mask_par)
#                 PSD_par = out_par["PSD"]
#                 SDK_par = out_par["SDK"]

#                 out_per = compute_all_metrics(df_num=df_w,
#                                               df_den=None,
#                                               rolling_params=rolling_params,
#                                               step=step,
#                                               psd_norm=psd_norm,
#                                               dt=dt,
#                                               compute_sdk=True,
#                                               mask=mask_per)
#                 PSD_per = out_per["PSD"]
#                 SDK_per = out_per["SDK"]

#                 out_mod_par = compute_all_metrics(df_num=df_mod_df,
#                                                   df_den=df_w,
#                                                   rolling_params=rolling_params,
#                                                   step=step,
#                                                   psd_norm=psd_norm,
#                                                   dt=dt,
#                                                   compute_sdk=True,
#                                                   mask=mask_par)
#                 PSD_mod_par = out_mod_par["PSD"]
#                 SDK_mod_par = out_mod_par["SDK"]
#                 compress_mod_par = out_mod_par["compress"]

#                 out_mod_per = compute_all_metrics(df_num=df_mod_df,
#                                                   df_den=df_w,
#                                                   rolling_params=rolling_params,
#                                                   step=step,
#                                                   psd_norm=psd_norm,
#                                                   dt=dt,
#                                                   compute_sdk=True,
#                                                   mask=mask_per)
#                 PSD_mod_per = out_mod_per["PSD"]
#                 SDK_mod_per = out_mod_per["SDK"]
#                 compress_mod_per = out_mod_per["compress"]

#                 out_w0_par = compute_all_metrics(df_num=df_w[["W0"]],
#                                                  df_den=df_w,
#                                                  rolling_params=rolling_params,
#                                                  step=step,
#                                                  psd_norm=psd_norm,
#                                                  dt=dt,
#                                                  compute_sdk=True,
#                                                  mask=mask_par)
#                 PSD_W0_par = out_w0_par["PSD"]

#                 out_w0_per = compute_all_metrics(df_num=df_w[["W0"]],
#                                                  df_den=df_w,
#                                                  rolling_params=rolling_params,
#                                                  step=step,
#                                                  psd_norm=psd_norm,
#                                                  dt=dt,
#                                                  compute_sdk=True,
#                                                  mask=mask_per)
#                 PSD_W0_per = out_w0_per["PSD"]

#                 out_wxy_par = compute_all_metrics(df_num=df_w[["Wx", "Wy"]],
#                                                   df_den=None,
#                                                   rolling_params=rolling_params,
#                                                   step=step,
#                                                   psd_norm=psd_norm,
#                                                   dt=dt,
#                                                   compute_sdk=True,
#                                                   mask=mask_par)
#                 PSD_Wxy_par = out_wxy_par["PSD"]

#                 out_wxy_per = compute_all_metrics(df_num=df_w[["Wx", "Wy"]],
#                                                   df_den=None,
#                                                   rolling_params=rolling_params,
#                                                   step=step,
#                                                   psd_norm=psd_norm,
#                                                   dt=dt,
#                                                   compute_sdk=True,
#                                                   mask=mask_per)
#                 PSD_Wxy_per = out_wxy_per["PSD"]

#             else:
#                 tot_par_power = np.real(df_w.iloc[index_par] * np.conj(df_w.iloc[index_par])) if index_par is not None else np.nan
#                 mod_par_power = np.real(df_mod[index_par] * np.conj(df_mod[index_par])) if index_par is not None else np.nan
#                 compress_mod_par = (np.nanmean(mod_par_power) / np.nanmean(tot_par_power.sum(axis=1))) if index_par is not None else np.nan

#                 tot_per_power = np.real(df_w.iloc[index_per] * np.conj(df_w.iloc[index_per])) if index_per is not None else np.nan
#                 mod_per_power = np.real(df_mod[index_per] * np.conj(df_mod[index_per])) if index_per is not None else np.nan
#                 compress_mod_per = (np.nanmean(mod_per_power) / np.nanmean(tot_per_power.sum(axis=1))) if index_per is not None else np.nan

#                 tot_all_power = np.real(df_w * np.conj(df_w))
#                 mod_all_power = np.real(df_mod * np.conj(df_mod))
#                 compress_mod = (np.nanmean(mod_all_power) / np.nanmean(tot_all_power.sum(axis=1)))

#         except Exception:
#             traceback.print_exc()
#             compress_mod_par = compress_mod_per = compress_mod = np.nan

#     return {
#         "PSD_par": PSD_par,
#         "PSD_per": PSD_per,

#         "PSD_mod": PSD_mod,
#         "PSD_mod_par": PSD_mod_par,
#         "PSD_mod_per": PSD_mod_per,

#         "PSD_Bpar": PSD_W0,
#         "PSD_Bpar_par": PSD_W0_par,
#         "PSD_Bpar_per": PSD_W0_per,

#         "PSD_Bper": PSD_Wxy,
#         "PSD_Bper_par": PSD_Wxy_par,
#         "PSD_Bper_per": PSD_Wxy_per,

#         "compress_mod": compress_mod,
#         "compress_mod_par": compress_mod_par,
#         "compress_mod_per": compress_mod_per,

#         "SDK_par": SDK_par,
#         "SDK_per": SDK_per,

#         "SDK_Bpar": SDK_W0,
#         "SDK_Bper": SDK_Wxy,

#         "SDK_mod": SDK_mod,
#         "SDK_mod_par": SDK_mod_par,
#         "SDK_mod_per": SDK_mod_per
#     }


# # ==============================================================
# #  do_coh_analysis (keeps all original keys; k-keys already present)
# # ==============================================================

# def do_coh_analysis(k_df,
#                     df_w,
#                     S0,
#                     S3,
#                     S0_full,
#                     Syz,
#                     index_par,
#                     index_per,
#                     dt,
#                     scale,
#                     psd_norm,
#                     func_params=None):
#     if func_params is None:
#         func_params = {}

#     if not func_params.get("estimate_coh_coeffs", True):
#         return None

#     use_rolling_mean = func_params.get("use_rolling_mean", False)
#     estimate_KAW_psd = func_params.get("est_sig_yz_cond_moms", False)

#     if use_rolling_mean:
#         if not isinstance(df_w.index, pd.DatetimeIndex):
#             raise ValueError("df_w must have a DateTimeIndex when use_rolling_mean is True.")

#         averaging_window = func_params["averaging_window"]
#         step = func_params["step"]
#         rolling_params = {"window": averaging_window, "center": True, "min_periods": 10}

#         sigma_av_yz = np.nan

#         num_value = local_gaussian_averaging(S3, dt, scale,
#                                              alpha=func_params.get("alpha_sigma", 3),
#                                              num_efoldings=func_params.get("num_efoldings", 3))
#         den_value = local_gaussian_averaging(S0, dt, scale,
#                                              alpha=func_params.get("alpha_sigma", 3),
#                                              num_efoldings=func_params.get("num_efoldings", 3))
#         sigma = num_value / den_value

#         index_coh = np.abs(sigma) > func_params["coh_th"]
#         index_non_coh = ~index_coh

#         sigma_av_xy = (pd.Series(sigma, index=df_w.index)).resample(step).mean()

#         coh_counts = (pd.Series(index_coh, index=df_w.index)).rolling(**rolling_params).sum()
#         non_coh_counts = (pd.Series(~index_coh, index=df_w.index)).rolling(**rolling_params).sum()
#         window_size = coh_counts + non_coh_counts

#         if estimate_KAW_psd:
#             num_value_yz = local_gaussian_averaging(Syz, dt, scale,
#                                                     alpha=1,
#                                                     num_efoldings=func_params.get("num_efoldings", 3))
#             den_value_yz = local_gaussian_averaging(S0_full, dt, scale,
#                                                     alpha=1,
#                                                     num_efoldings=func_params.get("num_efoldings", 3))
#             sigma_yz_ind = num_value_yz / den_value_yz

#             index_p = sigma_yz_ind > func_params["sigma_yz_thresh"]["pos"]
#             index_n = sigma_yz_ind < func_params["sigma_yz_thresh"]["neg"]
#             index_z = np.abs(sigma_yz_ind) < func_params["sigma_yz_thresh"]["zer"]
#         else:
#             index_p = index_n = index_z = None

#         k_df_roll = k_df.rolling(**rolling_params).mean().resample(step).mean()

#         if index_per is not None:
#             k_perp_per = (pd.Series(k_df["k_perp"], index=k_df.index)
#                           .where(pd.Series(index_per, index=df_w.index))
#                           .rolling(**rolling_params).mean().resample(step).mean())
#             dfdk_perp_per = (pd.Series(k_df["dfdk_perp"], index=k_df.index)
#                              .where(pd.Series(index_per, index=df_w.index))
#                              .rolling(**rolling_params).mean().resample(step).mean())
#         else:
#             k_perp_per = np.nan
#             dfdk_perp_per = np.nan

#         if index_par is not None:
#             k_par_par = (pd.Series(k_df["k_par"], index=k_df.index)
#                          .where(pd.Series(index_par, index=df_w.index))
#                          .rolling(**rolling_params).mean().resample(step).mean())
#             dfdk_par_par = (pd.Series(k_df["dfdk_par"], index=k_df.index)
#                             .where(pd.Series(index_par, index=df_w.index))
#                             .rolling(**rolling_params).mean().resample(step).mean())
#         else:
#             k_par_par = np.nan
#             dfdk_par_par = np.nan

#         sigma_xy = (pd.Series(S3, index=df_w.index).rolling(**rolling_params).mean()
#                     / pd.Series(S0, index=df_w.index).rolling(**rolling_params).mean()).resample(step).mean()
#         sigma_yz = (pd.Series(Syz, index=df_w.index).rolling(**rolling_params).mean()
#                     / pd.Series(S0_full, index=df_w.index).rolling(**rolling_params).mean()).resample(step).mean()

#         num_mean_per_yz = pd.Series(Syz, index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
#         den_mean_per_yz = pd.Series(S0_full, index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
#         sigma_yz_per = (num_mean_per_yz / den_mean_per_yz).resample(step).mean()

#         num_mean_par_xy = pd.Series(S3, index=df_w.index).where(pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).mean()
#         den_mean_par_xy = pd.Series(S0, index=df_w.index).where(pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).mean()
#         sigma_xy_par = (num_mean_par_xy / den_mean_par_xy).resample(step).mean()

#         num_mean_per_xy = pd.Series(S3, index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
#         den_mean_per_xy = pd.Series(S0, index=df_w.index).where(pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).mean()
#         sigma_xy_per = (num_mean_per_xy / den_mean_per_xy).resample(step).mean()

#         index_mask = pd.Series(index_coh, index=df_w.index)
#         df_ws_needed = df_w.where(index_mask)
#         PSD_coh = compute_norm_psd(df_ws_needed, rolling_params, dt, coh_counts, window_size, step, psd_norm)

#         index_mask = pd.Series(~index_coh, index=df_w.index)
#         df_ws_needed = df_w.where(index_mask)
#         PSD_non_coh = compute_norm_psd(df_ws_needed, rolling_params, dt, non_coh_counts, window_size, step, psd_norm)
#         SDK_non_coh = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#         Trace_PSD = np.nansum([PSD_coh, PSD_non_coh], axis=0)
#         SDK = calculate_wavelet_flatness(df_w, rolling_params, step)

#         index_mask = pd.Series(~index_coh & index_per, index=df_w.index)
#         df_ws_needed = df_w.where(index_mask)
#         PSD_non_coh_per, non_coh_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#         SDK_non_coh_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#         index_mask = pd.Series(~index_coh & index_par, index=df_w.index)
#         df_ws_needed = df_w.where(index_mask)
#         PSD_non_coh_par, non_coh_par_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#         SDK_non_coh_par = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#         if estimate_KAW_psd:
#             index_mask = pd.Series(index_p, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_p, sig_yz_p_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_p = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series(index_p & index_per, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_p_per, sig_yz_p_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_p_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series(index_n, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_n, sig_yz_n_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_n = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series(index_n & index_per, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_n_per, sig_yz_n_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_n_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series(index_n | index_p, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_pn, sig_yz_pn_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_pn = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series((index_n | index_p) & index_per, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_pn_per, sig_yz_pn_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_pn_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series(index_z, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_z, sig_yz_z_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_z = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)

#             index_mask = pd.Series(index_z & index_per, index=df_w.index)
#             df_ws_needed = df_w.where(index_mask)
#             PSD_sig_yz_z_per, sig_yz_z_per_counts = compute_psd_with_counts(df_ws_needed, rolling_params, dt, index_mask, step, psd_norm)
#             SDK_sig_yz_z_per = calculate_wavelet_flatness(df_ws_needed, rolling_params, step)
#         else:
#             PSD_sig_yz_n = PSD_sig_yz_p = PSD_sig_yz_z = PSD_sig_yz_pn = np.nan
#             PSD_sig_yz_z_per = PSD_sig_yz_p_per = PSD_sig_yz_n_per = PSD_sig_yz_pn_per = np.nan
#             SDK_sig_yz_n = SDK_sig_yz_p = SDK_sig_yz_z = SDK_sig_yz_pn = np.nan
#             SDK_sig_yz_z_per = SDK_sig_yz_p_per = SDK_sig_yz_n_per = SDK_sig_yz_pn_per = np.nan
#             sig_yz_p_counts = sig_yz_n_counts = sig_yz_z_counts = sig_yz_pn_counts = np.nan
#             sig_yz_p_per_counts = sig_yz_n_per_counts = sig_yz_z_per_counts = sig_yz_pn_per_counts = np.nan

#         return {
#             "k": k_df_roll["k"].values,

#             "k_perp": k_df_roll["k_perp"].values,
#             "k_par":  k_df_roll["k_par"].values,
#             "dfdk_perp": k_df_roll["dfdk_perp"].values,
#             "dfdk_par":  k_df_roll["dfdk_par"].values,

#             "k_perp_per": np.asarray(k_perp_per) if isinstance(k_perp_per, (pd.Series, pd.DataFrame)) else k_perp_per,
#             "k_par_par":  np.asarray(k_par_par)  if isinstance(k_par_par,  (pd.Series, pd.DataFrame)) else k_par_par,
#             "dfdk_perp_per": np.asarray(dfdk_perp_per) if isinstance(dfdk_perp_per, (pd.Series, pd.DataFrame)) else dfdk_perp_per,
#             "dfdk_par_par":  np.asarray(dfdk_par_par)  if isinstance(dfdk_par_par,  (pd.Series, pd.DataFrame)) else dfdk_par_par,

#             "sigma_xy_av": sigma_av_xy,
#             "sigma_yz_av": sigma_av_yz,

#             "sigma_xy": sigma_xy,
#             "sigma_xy_par": sigma_xy_par,
#             "sigma_xy_per": sigma_xy_per,

#             "sigma_yz": sigma_yz,
#             "sigma_yz_per": sigma_yz_per,

#             "PSD_Trace": Trace_PSD,
#             "PSD_coh": PSD_coh,
#             "PSD_non_coh": PSD_non_coh,

#             "SDK": SDK,
#             "SDK_non_coh": SDK_non_coh,
#             "SDK_non_coh_par": SDK_non_coh_par,
#             "SDK_non_coh_per": SDK_non_coh_per,
#             "SDK_sig_yz_n": SDK_sig_yz_n,
#             "SDK_sig_yz_p": SDK_sig_yz_p,
#             "SDK_sig_yz_z": SDK_sig_yz_z,
#             "SDK_sig_yz_pn": SDK_sig_yz_pn,
#             "SDK_sig_yz_z_per": SDK_sig_yz_z_per,
#             "SDK_sig_yz_p_per": SDK_sig_yz_p_per,
#             "SDK_sig_yz_n_per": SDK_sig_yz_n_per,
#             "SDK_sig_yz_pn_per": SDK_sig_yz_pn_per,

#             "PSD_non_coh_per": PSD_non_coh_per,
#             "counts_non_coh_per": non_coh_per_counts,

#             "PSD_non_coh_par": PSD_non_coh_par,
#             "counts_non_coh_par": non_coh_par_counts,

#             "PSD_sig_yz_n": PSD_sig_yz_n,
#             "PSD_sig_yz_p": PSD_sig_yz_p,
#             "PSD_sig_yz_z": PSD_sig_yz_z,
#             "PSD_sig_yz_pn": PSD_sig_yz_pn,

#             "PSD_sig_yz_z_per": PSD_sig_yz_z_per,
#             "PSD_sig_yz_p_per": PSD_sig_yz_p_per,
#             "PSD_sig_yz_n_per": PSD_sig_yz_n_per,
#             "PSD_sig_yz_pn_per": PSD_sig_yz_pn_per,

#             "counts_sig_yz_p": sig_yz_p_counts,
#             "counts_sig_yz_n": sig_yz_n_counts,
#             "counts_sig_yz_z": sig_yz_z_counts,
#             "counts_sig_yz_pn": sig_yz_pn_counts,

#             "counts_sig_yz_p_per": sig_yz_p_per_counts,
#             "counts_sig_yz_n_per": sig_yz_n_per_counts,
#             "counts_sig_yz_z_per": sig_yz_z_per_counts,
#             "counts_sig_yz_pn_per": sig_yz_pn_per_counts,

#             "counts_par": ((pd.Series(index_par, index=df_w.index)).rolling(**rolling_params).sum()).resample(step).sum(),
#             "counts_per": ((pd.Series(index_per, index=df_w.index)).rolling(**rolling_params).sum()).resample(step).sum(),
#             "counts_coh": coh_counts,
#             "counts_non_coh": non_coh_counts,

#             "counts": window_size,
#             "coh_thresh": func_params["coh_th"]
#         }

#     num_value = local_gaussian_averaging(S3, dt, scale, alpha=func_params["alpha"], num_efoldings=func_params["num_efoldings"])
#     den_value = local_gaussian_averaging(S0, dt, scale, alpha=func_params["alpha"], num_efoldings=func_params["num_efoldings"])
#     sigma = num_value / den_value

#     sigma_xy = np.nanmean(S3) / np.nanmean(S0)
#     sigma_xy_par = np.nanmean(S3[index_par]) / np.nanmean(S0[index_par]) if index_par is not None else np.nan
#     sigma_xy_per = np.nanmean(S3[index_per]) / np.nanmean(S0[index_per]) if index_per is not None else np.nan

#     sigma_yz = np.nanmean(Syz) / np.nanmean(S0) if Syz is not None else np.nan
#     sigma_yz_par = np.nanmean(Syz[index_par]) / np.nanmean(S0[index_par]) if (Syz is not None and index_par is not None) else np.nan
#     sigma_yz_per = np.nanmean(Syz[index_per]) / np.nanmean(S0[index_per]) if (Syz is not None and index_per is not None) else np.nan

#     index_coh = np.abs(sigma) > func_params["coh_th"]
#     index_non_coh = ~index_coh

#     coherent_sum = np.nanmean(np.real(df_w.iloc[index_coh].values * np.conj(df_w.iloc[index_coh].values)), axis=0).sum()
#     non_coherent_sum = np.nanmean(np.real(df_w.iloc[index_non_coh].values * np.conj(df_w.iloc[index_non_coh].values)), axis=0).sum()

#     PSD_coh = np.sum(index_coh) / len(index_coh) * coherent_sum
#     PSD_non_coh = np.sum(index_non_coh) / len(index_coh) * non_coherent_sum
#     Trace_PSD = PSD_coh + PSD_non_coh

#     k_mean = np.nanmean(k_df["k"].values) if "k" in k_df else np.nan
#     kper_mean = np.nanmean(k_df["k_perp"].values) if "k_perp" in k_df else np.nan
#     kpar_mean = np.nanmean(k_df["k_par"].values) if "k_par" in k_df else np.nan
#     dfdkper_mean = np.nanmean(k_df["dfdk_perp"].values) if "dfdk_perp" in k_df else np.nan
#     dfdkkpar_mean = np.nanmean(k_df["dfdk_par"].values) if "dfdk_par" in k_df else np.nan

#     return {
#         "k": k_mean,
#         "k_perp": kper_mean,
#         "k_par": kpar_mean,
#         "dfdk_perp": dfdkper_mean,
#         "dfdk_par": dfdkkpar_mean,

#         "sigma_xy": sigma_xy,
#         "sigma_xy_par": sigma_xy_par,
#         "sigma_xy_per": sigma_xy_per,

#         "sigma_yz": sigma_yz,
#         "sigma_yz_par": sigma_yz_par,
#         "sigma_yz_per": sigma_yz_per,

#         "PSD_coh": PSD_coh,
#         "PSD_non_coh": PSD_non_coh,
#         "PSD_Trace": Trace_PSD,

#         "counts_par": np.nansum(index_par) if index_par is not None else np.nan,
#         "counts_per": np.nansum(index_per) if index_per is not None else np.nan,
#         "counts_coh": np.nansum(index_coh),
#         "counts_non_coh": np.nansum(index_non_coh),
#         "counts_Trace": len(df_w.dropna()),
#         "coh_thresh": func_params["coh_th"]
#     }


# # ==============================================================
# #  Merge: coherence + compress + sfuncs
# # ==============================================================

# def return_desired_quants(df_w,
#                           df_mod,
#                           S0,
#                           S3,
#                           S0_full,
#                           Syz,
#                           VBangles,
#                           k_df,
#                           dt,
#                           scale,
#                           psd_norm,
#                           func_params=None):
#     if func_params is None:
#         func_params = {}

#     if func_params.get("estimate_PSDs", False) or func_params.get("estimate_coh_coeffs", False):
#         if func_params.get("use_rolling_mean", False):
#             index_per = VBangles > func_params["per_thresh"]
#             index_par = VBangles < func_params["par_thresh"]
#         else:
#             index_per = (np.where(VBangles > func_params["per_thresh"])[0]).astype(np.int64)
#             index_par = (np.where(VBangles < func_params["par_thresh"])[0]).astype(np.int64)
#     else:
#         index_per = None
#         index_par = None

#     coh_res = do_coh_analysis(k_df, df_w, S0, S3, S0_full, Syz, index_par, index_per, dt, scale, psd_norm, func_params=func_params)
#     sf_res = est_sfuncs(df_w, df_mod, index_par, index_per, scale, dt, func_params=func_params)
#     comp_res = est_compress(df_mod, df_w, index_par, index_per, dt, psd_norm, func_params=func_params)

#     return {**coh_res, **comp_res, **sf_res}


# def define_W_df(B_index, R, T, N):
#     return pd.DataFrame({
#         "DateTime": B_index,
#         "R": R,
#         "T": T,
#         "N": N
#     }).set_index("DateTime")


# # ==============================================================
# #  NEW helpers: scale-consistent local background for Va and sigma
# # ==============================================================

# def _safe_unit(arr_2d):
#     nrm = np.linalg.norm(arr_2d, axis=1, keepdims=True)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         out = arr_2d / nrm
#     out[~np.isfinite(out)] = np.nan
#     return out


# def compute_local_Va_sigma(
#     B_sm: pd.DataFrame,
#     np_sm: np.ndarray | None,
#     frame_flag: str | None,
#     C_ALFVEN: float = 21.82
# ):
#     B_vec = B_sm[["R", "T", "N"]].to_numpy(dtype=float, copy=False)
#     B_mag = np.linalg.norm(B_vec, axis=1)

#     if np_sm is None:
#         Va = np.full_like(B_mag, np.nan, dtype=float)
#     else:
#         np_sm = np.asarray(np_sm, dtype=float)
#         np_sm[np_sm <= 0] = np.nan
#         with np.errstate(divide="ignore", invalid="ignore"):
#             Va = C_ALFVEN * B_mag / np.sqrt(np_sm)
#         Va[~np.isfinite(Va)] = np.nan

#     if frame_flag is None:
#         frame_flag = "sc"

#     if str(frame_flag).upper() == "RTN":
#         Br_like = B_sm["R"].to_numpy(dtype=float, copy=False)
#     else:
#         Br_like = -B_sm["N"].to_numpy(dtype=float, copy=False)

#     sigma = np.sign(Br_like)
#     sigma[sigma == 0] = 1.0
#     sigma[~np.isfinite(sigma)] = 1.0

#     return Va, sigma


# # ==============================================================
# #  anisotropy_coherence (REVISED: scale-consistent k mapping)
# # ==============================================================

# def anisotropy_coherence(
#     B_df,
#     V_df,
#     mag_V_mod_TH,
#     field_flag,
#     E_df=None,
#     Np_df=None,
#     method="min_var",
#     func_params=None,
#     f_dict=None,
#     frame_flag=None,
#     mag_Va_ts=None,
#     sigma_Va_ts=None
# ):
#     if func_params is None:
#         func_params = {}

#     def define_B_df(B_df):
#         B_df["RR"] = B_df.R * B_df.R
#         B_df["TT"] = B_df["T"] * B_df["T"]
#         B_df["NN"] = B_df.N * B_df.N
#         B_df["RT"] = B_df.R * B_df["T"]
#         B_df["RN"] = B_df.R * B_df.N
#         B_df["TN"] = B_df["T"] * B_df.N
#         return B_df

#     def parallel_oper(mag_V_mod_TH,
#                       ii,
#                       scale,
#                       dt,
#                       B_df,
#                       V_df,
#                       df_w,
#                       df_mod,
#                       frequencies,
#                       psd_norm,
#                       method="min_var",
#                       func_params=None,
#                       frame_flag=None,
#                       mag_Va_ts=None,
#                       sigma_Va_ts=None,
#                       Np_df=None):
#         try:
#             if func_params is None:
#                 func_params = {}

#             bg_alpha = float(func_params.get("alpha_bg", func_params.get("bg_alpha", 5.0)))
#             bg_num_efoldings = int(func_params.get("bg_num_efoldings", func_params.get("num_efoldings", 3)))
#             local_V_mode = str(func_params.get("local_V_mode", "same")).lower()

#             if method == "min_var":
#                 B_df = define_B_df(B_df)

#             B_sm = B_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale,
#                                                                    alpha=bg_alpha,
#                                                                    num_efoldings=bg_num_efoldings), axis=0)

#             if local_V_mode == "same":
#                 V_sm = V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale,
#                                                                        alpha=bg_alpha,
#                                                                        num_efoldings=bg_num_efoldings), axis=0)
#             elif local_V_mode == "none":
#                 V_sm = V_df
#             elif local_V_mode == "alpha":
#                 alpha_V = float(func_params.get("alpha_V", bg_alpha))
#                 V_sm = V_df.apply(lambda col: local_gaussian_averaging(col.values, dt, scale,
#                                                                        alpha=alpha_V,
#                                                                        num_efoldings=bg_num_efoldings), axis=0)
#             else:
#                 V_sm = V_df

#             np_sm = None
#             if Np_df is not None:
#                 if isinstance(Np_df, pd.DataFrame) and "np" in Np_df.columns:
#                     np_series = Np_df["np"]
#                 elif isinstance(Np_df, pd.Series):
#                     np_series = Np_df
#                 else:
#                     np_series = None
#                 if np_series is not None:
#                     np_sm = local_gaussian_averaging(np_series.to_numpy(dtype=float, copy=False),
#                                                      dt, scale, alpha=bg_alpha, num_efoldings=bg_num_efoldings)
#             if np_sm is None and "np" in V_df.columns:
#                 np_series = V_df["np"]
#                 np_sm = local_gaussian_averaging(np_series.to_numpy(dtype=float, copy=False),
#                                                  dt, scale, alpha=bg_alpha, num_efoldings=bg_num_efoldings)

#             Va_loc, sigma_loc = compute_local_Va_sigma(B_sm, np_sm, frame_flag)

#             B_vec = B_sm[["R", "T", "N"]].to_numpy(dtype=float, copy=False)
#             b_hat = _safe_unit(B_vec)

#             V_vec = V_sm[["R", "T", "N"]].to_numpy(dtype=float, copy=False)
#             sigmaVa = sigma_loc * Va_loc
#             V_eff_vec = V_vec + (sigmaVa[:, None] * b_hat)

#             Vmod_loc = np.linalg.norm(V_eff_vec, axis=1)
#             Vmod_loc[~np.isfinite(Vmod_loc)] = np.nan

#             if func_params.get("do_coherence_analysis", True):
#                 df_w, VBangles, S3, S0, Syz, S0_full, kin = coherence_analysis(
#                     B_sm,
#                     V_sm,
#                     df_w,
#                     method,
#                     func_params=func_params
#                 )
#             else:
#                 B_tmp = unit_vectors(B_sm.copy(), prefix="B_0_", vector_cols=["R", "T", "N"])
#                 V_tmp = unit_vectors(V_sm.copy(), prefix="V_0_", vector_cols=["R", "T", "N"])

#                 VBangles = np.degrees(np.arccos(np.einsum(
#                     "ij,ij->i",
#                     B_tmp[["B_0_R_hat", "B_0_T_hat", "B_0_N_hat"]].values,
#                     V_tmp[["V_0_R_hat", "V_0_T_hat", "V_0_N_hat"]].values
#                 )))

#                 S3 = S0 = Syz = S0_full = None
#                 kin = {
#                     "phi": np.zeros(len(V_sm), dtype=float),
#                     "V_par_local": np.einsum("ij,ij->i", V_vec, b_hat),
#                     "V_perp": np.sqrt(np.maximum(0.0, np.linalg.norm(V_vec, axis=1)**2 - np.einsum("ij,ij->i", V_vec, b_hat)**2)),
#                     "V_kperp": np.nan * np.ones(len(V_sm)),
#                     "cphi": np.nan * np.ones(len(V_sm))
#                 }

#             VBangles = np.asarray(VBangles, dtype=float)
#             VBangles[VBangles > 90] = 180 - VBangles[VBangles > 90]

#             f = float(frequencies)
#             omega = 2.0 * np.pi * f

#             V_par_local = np.asarray(kin["V_par_local"], dtype=float)
#             V_perp_mag  = np.asarray(kin["V_perp"], dtype=float)
#             V_kperp     = np.asarray(kin["V_kperp"], dtype=float)
#             cphi_meas   = np.asarray(kin["cphi"], dtype=float)

#             V_floor = float(func_params.get("V_floor", 1e-6))
#             cphi_floor = float(func_params.get("cphi_floor", 1e-3))
#             cphi_mode = str(func_params.get("cphi_mode", "measured")).lower()
#             use_Vkperp = bool(func_params.get("use_Vkperp", True)) and bool(func_params.get("rotate_to_k", False))

#             kpar_mode = str(func_params.get("kpar_mode", "mth")).lower()
#             if kpar_mode in ("simple", "no_va", "novA", "none"):
#                 V_par_eff = V_par_local
#             else:
#                 V_par_eff = V_par_local + sigmaVa

#             V_par_eff_abs = np.abs(V_par_eff)
#             V_par_eff_abs[V_par_eff_abs < V_floor] = np.nan

#             if use_Vkperp and np.any(np.isfinite(V_kperp)):
#                 V_perp_proj = np.abs(V_kperp)
#             else:
#                 if cphi_mode == "axisym":
#                     cphi_eff = (2.0 / np.pi) * np.ones_like(V_perp_mag, dtype=float)
#                 elif cphi_mode == "unity":
#                     cphi_eff = np.ones_like(V_perp_mag, dtype=float)
#                 else:
#                     cphi_eff = np.abs(cphi_meas)
#                     cphi_eff = np.nan_to_num(cphi_eff, nan=1.0)
#                     cphi_eff = np.maximum(cphi_eff, cphi_floor)
#                 V_perp_proj = np.abs(V_perp_mag) * cphi_eff

#             V_perp_proj = np.asarray(V_perp_proj, dtype=float)
#             V_perp_proj[V_perp_proj < V_floor] = np.nan

#             Vmod_abs = np.abs(Vmod_loc)
#             Vmod_abs[Vmod_abs < V_floor] = np.nan

#             k_par_ts = omega / V_par_eff_abs
#             k_ts     = omega / Vmod_abs

#             kper_mode = str(func_params.get("kper_mode", "simple")).lower()
#             if kper_mode == "mth_exact":
#                 warnings.warn(
#                     "kper_mode='mth_exact' is not logically closed for single-spacecraft mapping "
#                     "unless k_par is independently known. Falling back to simple k_perp mapping.",
#                     RuntimeWarning
#                 )

#             k_perp_ts = omega / V_perp_proj

#             dfdk_perp_ts = V_perp_proj / (2.0 * np.pi)
#             dfdk_par_ts  = V_par_eff_abs / (2.0 * np.pi)

#             k_df_raw = pd.DataFrame({
#                 "k": k_ts,
#                 "k_perp": k_perp_ts,
#                 "k_par": k_par_ts,
#                 "dfdk_perp": dfdk_perp_ts,
#                 "dfdk_par": dfdk_par_ts
#             }, index=V_sm.index)

#             est_quants = return_desired_quants(
#                 df_w,
#                 df_mod,
#                 S0,
#                 S3,
#                 S0_full,
#                 Syz,
#                 VBangles,
#                 k_df_raw,
#                 dt=dt,
#                 scale=scale,
#                 psd_norm=psd_norm,
#                 func_params=func_params
#             )

#             if func_params.get("return_coeffs", False) is False:
#                 return est_quants, None, None, None, None, None

#             return est_quants, VBangles, df_w.values.T, S3, S0, Syz

#         except Exception:
#             traceback.print_exc()
#             return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

#     if B_df.columns[0] == "Bx":
#         B_df = B_df.rename(columns={"Bx": "R", "By": "T", "Bz": "N"})
#         V_df = V_df.rename(columns={"Vx": "R", "Vy": "T", "Vz": "N"})

#         if (method == "TN_only") and (E_df is not None):
#             E_df["R"] = 0 * E_df["Ey"]
#             E_df["T"] = E_df["Ex"]
#             E_df["N"] = -E_df["Ey"]
#             E_df.drop(columns=["Ex", "Ey"], inplace=True, errors="ignore")
#         else:
#             E_df = None if E_df is None else E_df.rename(columns={"Ex": "R", "Ey": "T", "Ez": "N"})
#     else:
#         B_df = B_df.rename(columns={"Br": "R", "Bt": "T", "Bn": "N"})
#         V_df = V_df.rename(columns={"Vr": "R", "Vt": "T", "Vn": "N"})
#         E_df = None if E_df is None else E_df.rename(columns={"Er": "R", "Et": "T", "En": "N"})

#     dt_B, dt_V = func.find_cadence(B_df), func.find_cadence(V_df)
#     dt_E = func.find_cadence(E_df) if E_df is not None else None

#     if E_df is not None and dt_E != dt_B:
#         try:
#             E_df, B_df = func.synchronize_dfs(E_df, B_df, True)
#         except Exception:
#             E_df = func.newindex(E_df, B_df.index)

#     if dt_V != dt_B:
#         B_df, V_df = func.synchronize_dfs(B_df, V_df, True)

#     dt = dt_E if dt_E is not None else dt_B

#     Wvec, scales, freqs, psd_norm, coi = estimate_cwt(
#         E_df if E_df is not None else B_df,
#         dt,
#         nv=func_params["nv"],
#         min_freq=func_params["min_freq"],
#         max_freq=func_params["max_freq"]
#     )

#     if func_params.get("est_mod", False):
#         if Np_df is None:
#             Wmod, _, _, _, _ = estimate_cwt(
#                 pd.DataFrame({"Mod": np.linalg.norm(B_df.iloc[:, :3], axis=1)}, index=B_df.index),
#                 dt,
#                 nv=func_params["nv"],
#                 min_freq=func_params["min_freq"],
#                 max_freq=func_params["max_freq"]
#             )
#         else:
#             if isinstance(Np_df, pd.DataFrame) and "np" in Np_df.columns:
#                 mod_series = Np_df["np"].values
#             else:
#                 mod_series = np.asarray(Np_df.values).reshape(-1)
#             Wmod, _, _, _, _ = estimate_cwt(
#                 pd.DataFrame({"Mod": mod_series}, index=B_df.index),
#                 dt,
#                 nv=func_params["nv"],
#                 min_freq=func_params["min_freq"],
#                 max_freq=func_params["max_freq"]
#             )
#     else:
#         Wmod = None

#     results = Parallel(n_jobs=func_params["njobs"])(
#         delayed(parallel_oper)(
#             mag_V_mod_TH,
#             ii,
#             scale,
#             dt,
#             B_df.copy(),
#             V_df.copy(),
#             define_W_df(B_df.index, Wvec["R"][ii], Wvec["T"][ii], Wvec["N"][ii]),
#             Wmod["Mod"][ii] if Wmod is not None else np.nan,
#             freqs[ii],
#             psd_norm,
#             method=method,
#             func_params=func_params,
#             frame_flag=frame_flag,
#             mag_Va_ts=mag_Va_ts,
#             sigma_Va_ts=sigma_Va_ts,
#             Np_df=Np_df
#         ) for ii, scale in tqdm(enumerate(scales), total=len(scales))
#     )

#     est_quants, VBangles, df_w, S3, S0, Syz = zip(*results)

#     if f_dict is None:
#         if func_params.get("use_rolling_mean", False):
#             f_dict = {field_flag: {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}}
#             f_dict["freqs"] = freqs
#             f_dict["scales"] = scales
#             f_dict["Wave_coeffs"] = df_w
#             f_dict["S3_ts"] = S3
#             f_dict["S0_ts"] = S0
#             f_dict["VB_ts"] = VBangles
#             f_dict["flag"] = method
#         else:
#             f_dict = {field_flag: {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}}
#     else:
#         if func_params.get("use_rolling_mean", False):
#             f_dict[field_flag] = {key: np.array([q[key] for q in est_quants]) for key in est_quants[0].keys()}
#         else:
#             f_dict[field_flag] = {key: np.hstack(np.array([q[key] for q in est_quants])) for key in est_quants[0].keys()}

#     return f_dict


# # ==============================================================
# #  Keep original call style in run_coh_pipeline
# # ==============================================================

# class polarization_analysis:
#     anisotropy_coherence = staticmethod(anisotropy_coherence)
#     build_V_mod_TH = staticmethod(build_V_mod_TH)


# # ==============================================================
# #  run_coh_pipeline (kept compatible; now passes Np_df into anisotropy)
# # ==============================================================

# def run_coh_pipeline(
#     load_path: str,
#     folder_name: str,
#     func_params: dict,
#     overwrite_files: bool = False,
#     overwrite_by_date: bool = False,
#     overwrite_before: str = "2026-01-09",
# ):
#     import logging

#     BG_WHITE = globals().get("BG_WHITE", "")
#     RESET = globals().get("RESET", "")

#     def should_overwrite(outfile: Path) -> bool:
#         if not outfile.exists():
#             return True
#         if bool(overwrite_files):
#             return True
#         if bool(overwrite_by_date):
#             cutoff = datetime.strptime(overwrite_before, "%Y-%m-%d")
#             mtime = datetime.fromtimestamp(outfile.stat().st_mtime)
#             return mtime < cutoff
#         return False

#     finnames = func.load_files(load_path, "final.pkl")
#     gennames = func.load_files(load_path, "general.pkl")
#     signames = func.load_files(load_path, "sig_c_sig_r.pkl")
#     maggaps = func.load_files(load_path, "mag_gaps.pkl")
#     qtngaps = func.load_files(load_path, "qtn_gaps.pkl")
#     pargaps = func.load_files(load_path, "par_gaps.pkl")
#     elecgaps = func.load_files(load_path, "elec_gaps.pkl")

#     averaging_window = func_params["averaging_window"]
#     step = func_params["step"]
#     rolling_params = {"window": averaging_window, "center": True, "min_periods": 10}

#     mu0 = constants.mu_0
#     m_p = constants.m_p

#     for jj in range(len(finnames)):
#         try:
#             func.progress_bar(jj, len(finnames))

#             foldername = (
#                 f"coh_step_{func_params['step']}_window_{func_params['averaging_window']}_"
#                 f"method_{list(func_params['method'].keys())[0]}_per_ang_{func_params['per_thresh']}.pkl"
#             )
#             path0 = finnames[jj].replace("final.pkl", "")
#             outdir = Path(path0) / folder_name
#             outfile = outdir / foldername

#             if should_overwrite(outfile):
#                 if not outdir.exists():
#                     logging.info(BG_WHITE + "Creating new folder %s" + RESET, outdir)
#                     outdir.mkdir(parents=True, exist_ok=True)
#                 else:
#                     if outfile.exists():
#                         logging.info(BG_WHITE + "Overwriting file %s" + RESET, outfile)
#                     else:
#                         logging.info(BG_WHITE + "Creating new file %s" + RESET, outfile)

#                 fin = pd.read_pickle(finnames[jj])
#                 sig = pd.read_pickle(signames[jj])
#                 mag_gaps = pd.read_pickle(maggaps[jj])
#                 qtn_gaps = pd.read_pickle(qtngaps[jj])
#                 par_gaps = pd.read_pickle(pargaps[jj])
#                 elec_gaps = pd.read_pickle(elecgaps[jj])

#                 if fin.get("E") is None:
#                     n_index = fin["Mag"]["B_resampled"].index
#                     E_df = None
#                     B_df = fin["Mag"]["B_resampled"]
#                 else:
#                     E_df = fin["E"]
#                     n_index = E_df.index
#                     B_df = func.newindex(fin["Mag"]["B_resampled"], n_index)
#                     E_df, B_df = func.synchronize_dfs(E_df, B_df, True)

#                 V_df = fin["Par"]["V_resampled"]
#                 sig, V_df = func.synchronize_dfs(sig, V_df, True)

#                 if "Vr" in V_df:
#                     frame_flag = "RTN"
#                     B_df_labs = ["Br", "Bt", "Bn"]
#                     V_df_labs = ["Vr", "Vt", "Vn"]
#                     V_df[V_df_labs] = V_df[V_df_labs] - V_df[["sc_vel_r", "sc_vel_t", "sc_vel_n"]].values
#                     Br_sign = np.sign(B_df["Br"].rolling("10min", center=True).mean())

#                     coh_dicts = {}
#                     mag_Va = None
#                     V_MTH_extra = None
#                     V_sc_rem = {"RTN": V_df[V_df_labs]}

#                     Np_local = None
#                     if "np" in V_df.columns:
#                         Np_local = V_df[["np"]]

#                 else:
#                     frame_flag = "sc"
#                     B_df_labs = ["Bx", "By", "Bz"]
#                     V_df_labs = ["Vx", "Vy", "Vz"]

#                     Br_sign = -np.sign(B_df["Bz"].rolling("10min", center=True).mean())

#                     V_df = fin["Par"]["V_resampled"][V_df_labs + ["np", "Vth"]].interpolate()
#                     V_df, sig = func.synchronize_dfs(V_df, sig, True)
#                     B_df, V_df = func.synchronize_dfs(B_df, V_df, True)
#                     Dist = func.newindex(fin["Par"]["V_resampled"]["Dist_au"], n_index)

#                     V_mod_TH, V_sc_rem, V_MTH_extra = polarization_analysis.build_V_mod_TH(
#                         B_df.copy(),
#                         V_df.copy(),
#                         return_addit=True,
#                     )

#                     mag_Va = pd.DataFrame(V_mod_TH["|Va|"], index=V_mod_TH.index, columns=["|Va|"])
#                     mag_V_mod_TH = pd.DataFrame(V_mod_TH["|V_mod_TH|"], index=V_mod_TH.index, columns=["|V_mod_TH|"])

#                     del fin

#                     coh_dicts = {}

#                     np_mean = V_df["np"].rolling("30s", center=True).mean().interpolate()
#                     kinet_normal = func.newindex(1e-15 / np.sqrt(mu0 * np_mean * m_p), B_df.index).values

#                     Bmag = np.sqrt(
#                         B_df[B_df_labs[0]] ** 2
#                         + B_df[B_df_labs[1]] ** 2
#                         + B_df[B_df_labs[2]] ** 2
#                     )

#                     rho_ci, _, _, _ = func.estimate_rho_i(V_df["Vth"], Bmag)

#                     del np_mean, kinet_normal

#                     B_df, sig = func.synchronize_dfs(B_df, sig, True)

#                     coh_dicts["sig_r"] = (sig.sigma_r).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["d"] = (Dist).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["beta"] = (sig.beta).rolling(**rolling_params).mean().resample(step).mean()

#                     coh_dicts["di"] = (sig.d_i).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["VB"] = (sig.VB).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["VB_std"] = (sig.VB).rolling(**rolling_params).std().resample(step).mean()
#                     coh_dicts["Vsw"] = (sig.Vsw).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["Vsw_Va"] = (mag_V_mod_TH).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["rho_ci"] = ((sig.d_i) * np.sqrt(sig.beta)).rolling(**rolling_params).mean().resample(step).mean()

#                     coh_dicts["Vrel_par"] = (V_MTH_extra["Vrel_par"]).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["Vrel_perp"] = (V_MTH_extra["Vrel_perp"]).rolling(**rolling_params).mean().resample(step).mean()
#                     coh_dicts["sigma_Va"] = (V_MTH_extra["sigma_Va"]).rolling(**rolling_params).mean().resample(step).mean()

#                     Np_local = V_df[["np"]] if "np" in V_df.columns else None

#                 num = sig["dva_r"] * sig["dv_r"] + sig["dva_t"] * sig["dv_t"] + sig["dva_n"] * sig["dv_n"]
#                 den_1 = sig["dva_r"] ** 2 + sig["dva_t"] ** 2 + sig["dva_n"] ** 2
#                 den_2 = sig["dv_r"] ** 2 + sig["dv_t"] ** 2 + sig["dv_n"] ** 2

#                 sig_num = (num.rolling(func_params["averaging_window"], center=True).mean()).resample(func_params["step"]).mean()
#                 sig_den1 = (den_1.rolling(func_params["averaging_window"], center=True).mean()).resample(func_params["step"]).mean()
#                 sig_den2 = (den_2.rolling(func_params["averaging_window"], center=True).mean()).resample(func_params["step"]).mean()

#                 coh_dicts["sig_c_v2"] = np.abs(2 * sig_num / (sig_den1 + sig_den2))

#                 del sig

#                 for method in list(func_params["method"].keys()):
#                     try:
#                         for mm, field in enumerate(func_params["method"][method]):
#                             print("Working on:", field, ". Coh method:", method)

#                             E_sw = globals().get("E_sw", None)

#                             coh_dict = polarization_analysis.anisotropy_coherence(
#                                 B_df.copy(),
#                                 V_sc_rem[V_df_labs].copy() if isinstance(V_sc_rem, dict) else V_sc_rem[V_df_labs].copy(),
#                                 mag_V_mod_TH if "mag_V_mod_TH" in locals() else pd.DataFrame(np.nan, index=B_df.index, columns=["|V_mod_TH|"]),
#                                 field_flag=field,
#                                 E_df=E_df.copy() if field == "E" else (E_sw.copy() if (field == "E_sw" and E_sw is not None) else None),
#                                 Np_df=Np_local,
#                                 method=method,
#                                 func_params=func_params,
#                                 f_dict=None if mm == 0 else coh_dict,
#                                 frame_flag=frame_flag,
#                                 mag_Va_ts=mag_Va if mag_Va is not None else None,
#                                 sigma_Va_ts=V_MTH_extra["sigma_Va"] if V_MTH_extra is not None else None,
#                             )

#                     except Exception:
#                         traceback.print_exc()
#                         coh_dict = None

#                     coh_dicts[method] = coh_dict

#                 func.savepickle(coh_dicts, str(outdir), foldername)

#             else:
#                 import logging
#                 logging.info(
#                     BG_WHITE + "File already exists; overwrite conditions not met: %s" + RESET,
#                     outfile,
#                 )

#         except Exception:
#             traceback.print_exc()
#             pass

