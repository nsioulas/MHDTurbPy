# Revision notes: rev5 physics-centered normalization cleanup

This revision keeps the SCWF estimator family but tightens its interpretation and
normalization.

## Core code changes

1. **Second-order spectra now use the exact implemented FFT response**
   - `response_energy_integral` is computed on the actual positive-frequency FFT grid
     used by the transform instead of being treated only as the continuum constant.
   - This preserves the correct PSD units while matching the discretized estimator.

2. **Half-power bandwidth now means half power**
   - The old code matched the half-amplitude points of the mother response and then
     called the result a half-power bandwidth.
   - The revised code finds the half-power points of `|H|^2` instead.

3. **Magnetic outputs are no longer unit-ambiguous**
   - `B_nT` and `B_vel` are exported explicitly in `WaveletMoments`, `Sfuncs`, and
     `Spectra`.
   - The compatibility family `B` still follows `return_B_in_vel_units`.

4. **Clear aliases are exported for the actual observables**
   - `ScaleNormalizedWaveletMoments` is an alias of `Sfuncs`.
   - `ConditionalBandAveragedPSD` and `ConditionalBandPower` are aliases of
     `Spectra`.

5. **Metadata now states what the anisotropy buckets really mean**
   - The local basis defines the angle bins.
   - The spectra and higher-order moments are computed from the **full vector
     coefficient magnitude** within those bins, not from basis-component projections.

6. **Wavelet-name parsing is documented explicitly**
   - `mw8` means 8 voices per octave unless an explicit derivative-order token is
     present.
   - The derivative order otherwise defaults to 4.

## What remains intentionally unchanged

- The estimator still uses a Gaussian low-pass plus an even derivative-of-Gaussian
  band-pass coefficient family.
- The code still returns `Sfuncs` for backward compatibility.
- The local basis still uses `B_l` and `W_B,perp`.
- The bucket names `ell_perp`, `Ell_perp`, `ell_par`, and `ell_overall` are retained
  for compatibility with downstream analysis code.

## Interpretation that is now defensible

- `WaveletMoments`: raw coefficient moments with units `[X]^q s^(q/2)`.
- `ScaleNormalizedWaveletMoments`: `tau^{-q/2}`-normalized surrogates with units
  `[X]^q`.
- `Spectra`: conditional band-averaged PSD estimates with units `[X]^2/Hz`.
- `ell_*` buckets: angle-conditioned sample subsets labeled by the projected
  separation used on the horizontal axis.

## Important limitation preserved deliberately

The code still does **not** estimate exact basis-component spectra such as
`P_parallel`, `P_xi`, or `P_lambda`. It estimates conditional band-power curves of the
full vector coefficient amplitude inside bins defined by the local anisotropy frame.
