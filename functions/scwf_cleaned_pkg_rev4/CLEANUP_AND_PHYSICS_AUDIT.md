# Cleanup and physics audit snapshot

## What was removed

The package previously exposed multiple public names that did not correspond to distinct algorithms in this archive. The cleaned revision removes the unused MODWT/two-point aliases from the public API and drops dead internal helpers that were not referenced anywhere in the execution path. The surviving notebook-facing compatibility surface is:

- `run_filterbank_interval_analysis`
- `run_logscale_filterbank_analysis`
- `estimate_3D_sfuncs_same_format`
- `estimate_3D_sfuncs`
- `estimate_filterbank_backgrounds_and_fluctuations`

Internally removed as dead or misleading in this archive:

- `_rolling_mean_centered`
- `conditional_moments_3D`
- `estimate_3D_sfuncs_logscale`
- `estimate_filterbank_backgrounds_and_fluctuations_same_format`
- legacy MODWT/two-point alias assignments in `three_D_funcs.py`

## Physics-critical corrections

### 1. Missing structure-function conversion implementation

The source called `_structure_function_from_wavelet_moments(...)`, but no such function existed. This was a hard inconsistency in the code path. The cleaned revision implements the intended conversion explicitly as

$$
M_q^{\mathrm{inc}}(s) = M_q^{\mathrm{wav}}(s) \, s^{-q/2},
$$

with the interpretation stated honestly: this removes the universal $L^2$ wavelet normalization factor, but it remains an increment-equivalent wavelet surrogate rather than a strict finite-difference structure function.

### 2. Velocity correction applied inconsistently

The previous implementation used the spacecraft-corrected velocity only for the Taylor-mapping background flow, while the velocity fluctuation $\delta V$ and the Elsasser fluctuations still came from the uncorrected velocity series. That mixed two different velocity definitions in the same local geometry. The cleaned revision uses the same corrected velocity time series consistently both for the local flow and for the velocity/Elsasser wavelet coefficients whenever that corrected input is provided.

### 3. Background plus fluctuation overclaim

The Gaussian low-pass background and the even-DoG band-pass coefficient are paired filters evaluated at the same scale, but they are not an exact additive decomposition of the original signal. The cleaned metadata and README now state this explicitly.

### 4. Fake execution metadata

The previous saved `analysis_chain` included a function that was not actually called on the runtime path. The cleaned metadata now records the real call sequence.

## What the code now claims more carefully

- `WaveletMoments` stores raw conditional wavelet-coefficient moments.
- `Sfuncs`/`StructureFunctions` stores increment-equivalent wavelet surrogates obtained by removing the universal $s^{q/2}$ factor.
- `Spectra` stores second-order raw wavelet moments divided by the exact response-energy integral of the implemented filter.
- The velocity used for local Taylor mapping and the velocity used for Elsasser construction are now the same field.

## Important remaining interpretation constraints

These were not silently changed, but they must be kept in mind during the deeper audit:

1. `tau_equiv_samples` is a peak-frequency calibration, not an exact finite-difference lag.
2. The local $(\xi, \lambda)$ basis becomes undefined when $|\delta B_\perp| 	o 0$; the code now masks those cases, but the geometry is still singular there by construction.
3. The saved magnetic family `B` still follows the user-selected unit system controlled by `return_B_in_vel_units`; this is convenient, but it means the physical units of the `B` family are not fixed unless the metadata is checked.
4. Exported samplewise `sig_c` and `sig_r` are local surrogate ratios, not the canonical scale-level ratios. The canonical per-scale quantities remain `sig_c_scale` and `sig_r_scale`.
