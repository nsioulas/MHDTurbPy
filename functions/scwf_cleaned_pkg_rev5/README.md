# scwf_anisotropy

This package implements a **self-consistent wavelet-frame (SCWF)** pipeline for
**conditional band-averaged spectra** and **scale-normalized wavelet moments** in a
local anisotropy frame.

It is designed to sit under `MHDTurbPy/functions/scwf_anisotropy` and to be called
through the notebook-facing batch driver or directly from `three_D_funcs.py`.

## What this revision changes

This revision does not change the estimator family. It changes the normalization,
unit bookkeeping, and interpretation so that the saved outputs match the equations
actually implemented by the code.

- The background and fluctuation estimates are treated explicitly as **centered,
  symmetric, zero-phase** multiscale estimates in the interior of the interval.
- Edge handling is an explicit **cone-of-influence** rule. Samples whose local
  wavelet support overlaps either interval edges or raw data gaps are excluded at
  that scale.
- The second-order spectral estimate is normalized by the **positive-frequency
  discrete response-energy integral of the exact FFT response used by the code**.
  The old bandwidth normalization is retained only as `LegacySpectraBandwidth`.
- The code now exports the magnetic families explicitly as:
  - `B_nT`: magnetic coefficient moments in nT-based units;
  - `B_vel`: magnetic coefficient moments in Alfv'en-speed-equivalent units;
  - `B`: backward-compatible alias that still follows `return_B_in_vel_units`.
- `Sfuncs` is preserved for compatibility, but it should be interpreted as a
  **scale-normalized wavelet-moment surrogate** rather than as a strict
  finite-difference structure function.
- The local anisotropy basis is used to define **conditional angle bins**. The code
  then averages the **full vector coefficient magnitude** inside each bin. It does
  **not** estimate basis-component spectra such as `P_parallel`, `P_xi`, or
  `P_lambda`.
- The `(xi, lambda)` orientation is treated as undefined when `|dB_perp|` is too
  small; those samples are excluded from direction-resolved bins instead of being
  assigned an artificial perpendicular basis.
- The exported metadata now records that `mw8` means **8 voices per octave** unless
  an explicit derivative-order token is present. The derivative order otherwise
  defaults to `4`.
- The saved metadata now distinguishes:
  - `WaveletMoments`: raw coefficient moments;
  - `ScaleNormalizedWaveletMoments`: `tau^{-q/2}`-normalized surrogates;
  - `Spectra` / `ConditionalBandAveragedPSD`: second-order conditional band-power
    estimates.

## Package layout

- `data_analysis.py`: lightweight public entry points
- `three_D_funcs.py`: core estimator and batch driver
- `notebooks/3D_LOGFILTERBANK_sfuncs_cleaned.ipynb`: minimal driver notebook
- `appendix_method.tex`: detailed method appendix
- `appendix_method.pdf`: compiled method appendix
- `revision_summary.pdf`: compact summary of the physics and implementation changes

## Call chain

`3D_LOGFILTERBANK_sfuncs_cleaned.ipynb`
→ `data_analysis.run_logscale_filterbank_analysis`
→ `three_D_funcs.run_filterbank_interval_analysis`
→ `three_D_funcs.estimate_wavelet_interval`
→ `three_D_funcs.estimate_local_wavelet_geometry`
→ `three_D_funcs.estimate_wavelet_backgrounds_and_fluctuations`

## Method summary

At each scale `s`, the pipeline defines a low-pass background and a band-pass
wavelet coefficient from the same multiresolution family,

`B_l(t,s) = (L_s * B)(t),   W_B(t,s) = (H_s * B)(t)`.

Because the frequency responses are real and even, and because the time-domain
padding is symmetric, these estimates are centered in time in the interval
interior.

The local basis is built self-consistently from those same quantities,

`e_l = B_l / |B_l|`,

`W_B,perp = W_B - (W_B · e_l) e_l`,

`e_xi = W_B,perp / |W_B,perp|`,

`e_lambda = e_l × e_xi`.

The effective local sampling vector is

`ell(t,s) = V_l(t,s) tau_s`,

where `tau_s` is the peak-frequency calibration of the implemented band-pass
response.

The second-order conditional spectral estimate is

`P_hat(s | C) = < |W_X(t,s)|^2 >_C / A_s`,

where `A_s` is the positive-frequency response-energy integral of the exact FFT
response used in the implementation, and `C` denotes one of the angle-conditioned
buckets.

Higher-order conditional moments are accumulated first as raw wavelet moments

`M_q(s | C) = < |W_X(t,s)|^q >_C`,

and are then converted to the saved scale-normalized surrogate

`M_q_norm(s | C) = M_q(s | C) / tau_s^(q/2)`.

This normalization restores the physical field dimensions `[X]^q`, but it does not
turn the quantity into an exact finite-difference structure function.

## Output conventions

The saved outputs retain the old external structure and add clearer aliases:

- `WaveletMoments`: raw conditioned wavelet-coefficient moments
- `Sfuncs` / `StructureFunctions` / `ScaleNormalizedWaveletMoments`: scale-normalized
  wavelet-moment surrogates with physical `[X]^q` units
- `Spectra` / `ConditionalBandAveragedPSD`: second-order conditional band-averaged
  PSD estimates of the **full vector coefficient magnitude**
- `LegacySpectraBandwidth`: older bandwidth-based normalization, kept only for
  comparison
- `flucts`: optional per-sample tables, including `polarity`, `local_polarity`, and
  `polarity_used` when requested
- `overall_align_angles`: per-level alignment summaries, including `sig_c_scale` /
  `sig_r_scale` and the preserved local-ratio summaries
- `meta`: scale grid, equivalent lags, bandwidths, response-energy integrals, and
  cone-of-influence widths

## Scientific interpretation

The pipeline is a **local-angle-conditioned reduced multiscale analysis**. It does
not reconstruct the full 3D spectral tensor. It does not compute exact finite
increments. The Gaussian background and even-DoG fluctuation are paired filters,
not an exact additive decomposition `X = X_l + W_X`.

Its defensible interpretation is narrower and cleaner: conditional multiscale
statistics as functions of local field geometry and local fluctuation direction,
with the second-order product interpreted as a conditional band-averaged PSD and
higher-order products interpreted as scale-normalized wavelet moments.
