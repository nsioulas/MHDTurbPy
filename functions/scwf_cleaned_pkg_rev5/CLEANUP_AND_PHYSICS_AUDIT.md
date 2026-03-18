# Cleanup and physics audit snapshot

## What was kept

The surviving notebook-facing API is:

- `run_filterbank_interval_analysis`
- `run_logscale_filterbank_analysis`
- `estimate_3D_sfuncs_same_format`
- `estimate_3D_sfuncs`
- `estimate_filterbank_backgrounds_and_fluctuations`

The implementation continues to use a single SCWF backend rather than separate
increment, MODWT, and CWT code paths.

## Physics-critical points fixed in this package revision

### 1. PSD normalization now matches the discretized estimator

The present package computes

`P_hat(s|C) = <|W_X|^2>_C / A_s`,

where `A_s` is the positive-frequency discrete response-energy integral of the exact
FFT response used by the transform. This preserves the correct PSD dimensions and
removes the earlier mismatch between the documented normalization and the actual
implemented grid.

### 2. The bandwidth diagnostic is no longer mislabeled

The half-width frequencies are now determined from half power in `|H|^2`, not from
half amplitude in `H`.

### 3. Magnetic outputs are explicit in both unit systems

The package exports `B_nT` and `B_vel` in the moment and spectral families. The
compatibility family `B` remains tied to `return_B_in_vel_units`.

### 4. The anisotropy bins are documented honestly

The local basis is used to define angle-conditioned subsets. The code then averages
the full vector coefficient magnitude in those subsets. This is not a projection of
power onto basis components.

### 5. The saved metadata now uses the real notebook path and clearer names

The execution chain points to `3D_LOGFILTERBANK_sfuncs_cleaned.ipynb`, and the saved
result now includes clearer aliases for the actual observables.

## What the code now claims carefully

- `WaveletMoments`: raw conditional wavelet-coefficient moments.
- `ScaleNormalizedWaveletMoments`: scale-normalized wavelet-moment surrogates.
- `Spectra` / `ConditionalBandAveragedPSD`: conditional band-averaged PSD estimates.
- `LegacySpectraBandwidth`: legacy comparison product only.

## Important remaining interpretation constraints

1. `tau_equiv_samples` remains a peak-frequency calibration, not an exact increment lag.
2. The local `(xi, lambda)` basis is undefined when `|W_B,perp| -> 0`; those samples
   are masked rather than repaired.
3. `Sfuncs` remains a compatibility name. Its correct interpretation is a
   scale-normalized wavelet-moment surrogate, not a strict finite-difference
   structure function.
4. The code does not reconstruct the full 3D spectral tensor from a single
   spacecraft time series.
