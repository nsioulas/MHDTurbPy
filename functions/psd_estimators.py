from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import os
import sys

import numpy as np
import pycwt
import pywt
import ssqueezepy

sys.path.insert(1, os.path.join(os.getcwd(), "functions/modwt/wmtsa"))
import modwt


def _as_array(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(values)


@dataclass(frozen=True)
class FFTTracePSD:
    remove_mean: bool = False
    return_components: bool = False
    return_mod: bool = False

    def estimate(
        self, x: Any, y: Any, z: Any, dt: float
    ) -> Tuple[np.ndarray, ...]:
        x = _as_array(x)
        y = _as_array(y)
        z = _as_array(z)

        if self.remove_mean:
            x -= np.nanmean(x)
            y -= np.nanmean(y)
            z -= np.nanmean(z)

        n = len(x)
        xf = np.fft.rfft(x)
        yf = np.fft.rfft(y)
        zf = np.fft.rfft(z)

        p_x = 2 * (np.abs(xf) ** 2) / n * dt
        p_y = 2 * (np.abs(yf) ** 2) / n * dt
        p_z = 2 * (np.abs(zf) ** 2) / n * dt
        p_trace = p_x + p_y + p_z

        freqs = np.fft.rfftfreq(n, dt)

        if self.return_mod:
            mod = np.sqrt(x**2 + y**2 + z**2)
            p_mod = 2 * (np.abs(np.fft.rfft(mod)) ** 2) / n * dt
            return freqs, p_trace, p_x, p_y, p_z, p_mod

        if self.return_components:
            return freqs, p_trace, p_x, p_y, p_z

        return freqs, p_trace


@dataclass(frozen=True)
class MODWTTracePSD:
    wname: str = "sym8"

    def estimate(
        self, r: Any, t: Any, n: Any, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        wr, _ = modwt.modwt(r, wtf=self.wname, nlevels="conservative", boundary="reflection", RetainVJ=True)
        wt, _ = modwt.modwt(t, wtf=self.wname, nlevels="conservative", boundary="reflection", RetainVJ=True)
        wn, _ = modwt.modwt(n, wtf=self.wname, nlevels="conservative", boundary="reflection", RetainVJ=True)

        scale = 2 ** np.arange(1, np.shape(wr)[0] + 1)
        freqs = pywt.scale2frequency("coif6", scale) / dt

        psd_r = modwt.wspec(wr, dt)
        psd_t = modwt.wspec(wt, dt)
        psd_n = modwt.wspec(wn, dt)

        return freqs, 2 * (psd_r[0] + psd_t[0] + psd_n[0]), scale


@dataclass(frozen=True)
class HaarWaveletPSD:
    wavelet: str = "haar"

    def estimate(
        self, x: Any, y: Any, z: Any, dt: float
    ) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = pywt.wavedec(x, self.wavelet)
        y = pywt.wavedec(y, self.wavelet)
        z = pywt.wavedec(z, self.wavelet)

        px = np.array([np.nanmean(coeff**2) for coeff in x[1:]])
        py = np.array([np.nanmean(coeff**2) for coeff in y[1:]])
        pz = np.array([np.nanmean(coeff**2) for coeff in z[1:]])

        px = dt * (px[::-1]) / np.log2(2)
        py = dt * (py[::-1]) / np.log2(2)
        pz = dt * (pz[::-1]) / np.log2(2)
        p_trace = px + py + pz

        freqs = 2.0 ** (-np.arange(1, len(px) + 1)) / dt

        return x, y, z, freqs, p_trace, px, py, pz


@dataclass(frozen=True)
class PyCWTWaveletPSD:
    dj: float = 1 / 2
    mother_wave: str = "morlet"

    def estimate(
        self, x: Any, y: Any, z: Any, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mother_wave_dict = {
            "gaussian": pycwt.DOG(),
            "paul": pycwt.Paul(),
            "mexican_hat": pycwt.MexicanHat(),
        }

        mother_morlet = mother_wave_dict.get(self.mother_wave, pycwt.Morlet())

        db_x, _, freqs, _, _, _ = pycwt.cwt(x, dt, self.dj, wavelet=mother_morlet)
        db_y, _, freqs, _, _, _ = pycwt.cwt(y, dt, self.dj, wavelet=mother_morlet)
        db_z, _, freqs, _, _, _ = pycwt.cwt(z, dt, self.dj, wavelet=mother_morlet)

        psd = (
            np.nanmean(np.abs(db_x) ** 2, axis=1)
            + np.nanmean(np.abs(db_y) ** 2, axis=1)
            + np.nanmean(np.abs(db_z) ** 2, axis=1)
        ) * (2 * dt)

        scales = (1 / (freqs)) / dt

        return db_x, db_y, db_z, freqs, psd, scales


@dataclass(frozen=True)
class SSqueezepyWaveletPSD:
    nv: int = 16
    scales_type: Optional[str] = "log"
    wavelet: Optional[Any] = None
    wname: Optional[str] = None
    l1_norm: bool = False
    est_psd: bool = True
    est_mod: bool = False
    omega0: float = 6.0

    def estimate(
        self, x: Any, y: Any, z: Any, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        if self.wavelet is None:
            wavelet = ssqueezepy.Wavelet(("morlet", {"mu": 13.4}))
        else:
            wavelet = ssqueezepy.Wavelet((self.wname, {"mu": 13.4}))

        scales_type = self.scales_type or "log"
        fs = 1 / dt

        wx, scales = ssqueezepy.cwt(x, wavelet, scales_type, fs, l1_norm=self.l1_norm, nv=self.nv)
        wy, _ = ssqueezepy.cwt(y, wavelet, scales_type, fs, l1_norm=self.l1_norm, nv=self.nv)
        wz, _ = ssqueezepy.cwt(z, wavelet, scales_type, fs, l1_norm=self.l1_norm, nv=self.nv)

        if self.est_mod:
            wmod, _ = ssqueezepy.cwt(
                np.sqrt(x**2 + y**2 + z**2),
                wavelet,
                scales_type,
                fs,
                l1_norm=self.l1_norm,
                nv=self.nv,
            )
        else:
            wmod = None

        freqs = ssqueezepy.experimental.scale_to_freq(scales, wavelet, len(x), fs)

        scales = (self.omega0) / (2 * np.pi * freqs) * (1 + 1 / (2 * self.omega0**2)) * fs

        if self.est_psd:
            psd = (
                np.nanmean(np.abs(wx) ** 2, axis=1)
                + np.nanmean(np.abs(wy) ** 2, axis=1)
                + np.nanmean(np.abs(wz) ** 2, axis=1)
            ) * (2 * dt)

            psd_mod = (np.nanmean(np.abs(wmod) ** 2, axis=1)) * (2 * dt) if self.est_mod else None
        else:
            psd = None
            psd_mod = None

        return wx, wy, wz, wmod, freqs, psd, psd_mod, scales


PSD_ESTIMATORS: Dict[str, Any] = {
    "fft": FFTTracePSD,
    "modwt": MODWTTracePSD,
    "haar": HaarWaveletPSD,
    "pycwt": PyCWTWaveletPSD,
    "ssqueezepy": SSqueezepyWaveletPSD,
}


def get_psd_estimator(method: str, **kwargs: Any) -> Any:
    estimator = PSD_ESTIMATORS.get(method)
    if estimator is None:
        raise ValueError(f"Unknown PSD estimator method: {method}")
    return estimator(**kwargs)
