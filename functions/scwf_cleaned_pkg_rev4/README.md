# scwf_anisotropy

This package implements a **self-consistent wavelet-frame anisotropy** pipeline for local-field anisotropic spectra and higher-order conditioned moments.

It is designed to sit under `MHDTurbPy/functions/scwf_anisotropy` and to be called through the batch driver or directly from `three_D_funcs.py`.

## What this revision changes

This revision tightens the estimator rather than changing its scientific target.

- The background and fluctuation estimates are treated explicitly as **centered, symmetric, zero-phase** multiscale estimates in the interior of the interval.
- Edge handling is now an explicit **cone-of-influence** rule. Samples whose local wavelet support overlaps either interval edges or raw data gaps are excluded at that scale.
- Row-wise fluctuation exports are normalized to a strict **samplewise 1-D contract**, so level-wise scalars and malformed objects cannot crash the table writer.
- The pipeline now **separates raw wavelet moments from increment-equivalent moment surrogates**. The latter are obtained by removing the $\sqrt{s}$ factor implied by the $L^2$ wavelet normalization, but they remain wavelet-based surrogates rather than strict finite-difference structure functions.
- The reduced spectra are now normalized by the exact filter-energy integral $\int_0^\infty |H_s(f)|^2\,df$ instead of by an effective half-power bandwidth.
- When coefficient tables are saved, the magnetic fluctuation family is exported **by default** in the user-selected units:
  - `return_B_in_vel_units=True` (default): `dB_*` is saved in Alfv'en-speed units;
  - `return_B_in_vel_units=False`: `dB_*` is saved in nT.
- Explicit access to both forms is preserved through `dB_nT_*` and `dVa_*`.
- The $(\xi,\lambda)$ orientation is now treated as undefined when $|\delta B_\perp|$ is numerically negligible; those samples are excluded from direction-resolved bins instead of being assigned an artificial perpendicular basis.
- The Elsasser sign actually used by the code is now exported explicitly as `polarity_used`, so the distinction between `polarity`, `local_polarity`, and the chosen sign is visible in the saved tables.
- Alignment summaries now separate the canonical scale-level ratios from averages of pointwise ratios: `sig_c_scale` / `sig_r_scale` are the physically standard per-scale diagnostics, while `sig_c_local_mean`, `sig_c_local_median`, `sig_r_local_mean`, and `sig_r_local_median` retain the pointwise summaries.
- The parallel anisotropy bin no longer requires a finite azimuth `phi`. The backward-compatible `ell_par_rest` bucket is retained, but it is now a theta-only quasi-parallel alias rather than a phi-conditioned bin.

## Package layout

- `data_analysis.py`: lightweight public entry points
- `three_D_funcs.py`: core estimator and batch driver
- `notebooks/3D_LOGFILTERBANK_sfuncs.ipynb`: minimal driver notebook
- `APPENDIX_scwf_anisotropy_method.tex`: method appendix
- `APPENDIX_scwf_anisotropy_method.pdf`: compiled appendix

## Call chain

`3D_LOGFILTERBANK_sfuncs.ipynb`
→ `data_analysis.run_logscale_filterbank_analysis`
→ `three_D_funcs.run_filterbank_interval_analysis`
→ `three_D_funcs.estimate_wavelet_interval`
→ `three_D_funcs.estimate_local_wavelet_geometry`
→ `three_D_funcs.estimate_wavelet_backgrounds_and_fluctuations`

## Method summary

At each scale $s$, the pipeline defines a low-pass background and a band-pass fluctuation from the same multiresolution decomposition,

$B_\ell(t,s) = (\phi_s * B)(t), \qquad \delta B_s(t) = (\psi_s * B)(t).$

Because the frequency responses are real and even, and because the time-domain padding is symmetric, these estimates are **centered in time** in the interior of the interval. The coefficient at time $t$ is therefore interpreted as a local estimate built from a symmetric support about $t$.

The local basis is then built self-consistently from those same quantities,

$\hat e_\parallel(t,s) = B_\ell(t,s)/|B_\ell(t,s)|,$

$\delta B_{\perp,s}(t) = \delta B_s(t) - [\delta B_s(t)\cdot\hat e_\parallel(t,s)]\,\hat e_\parallel(t,s),$

$\hat e_\xi(t,s) = \delta B_{\perp,s}(t)/|\delta B_{\perp,s}(t)|,$

$\hat e_\lambda(t,s) = \hat e_\parallel(t,s) \times \hat e_\xi(t,s).$

The effective local sampling vector is

$\boldsymbol{\ell}(t,s) = \tau_s U_\ell(t,s),$

where $U_\ell$ is the local advecting flow used for Taylor mapping. When a spacecraft-velocity correction is supplied, the same corrected velocity is now used consistently both here and in the velocity/Elsasser fluctuations. Its projections onto the local basis define the geometry used in the anisotropy bins.

The Elsasser fluctuations use the explicit sign chosen by the user option `use_local_polarity`,

$\delta z^{\pm}(t,s) = \delta V(t,s) \pm p_B(t,s)\,\delta V_A(t,s),$

with $p_B(t,s)=\texttt{local\_polarity}(t,s)$ when `use_local_polarity=True` and $p_B(t,s)=\texttt{polarity}(t,s)$ otherwise.

The local reduced second-order statistics are accumulated conditionally in that frame,

$P_a(s|\Omega) = \langle |\delta B_s \cdot \hat e_a|^2 \mid \Omega(t,s) \rangle,$

with $a \in \{\parallel,\xi,\lambda\}$ and $\Omega$ the angle bin.

Higher-order conditioned moments are accumulated first as raw wavelet moments

$M_q(s|\Omega) = \langle |W_B(t,s)|^q \mid \Omega(t,s) \rangle,$

and then converted to the saved increment-equivalent surrogate by removing the universal $s^{q/2}$ factor of the $L^2$ normalization.

## Cone of influence

The estimator is evaluated on gap-filled inputs because the FFT backend cannot tolerate NaNs. The code therefore separates **transform evaluation** from **coefficient validity**.

At each scale, a coefficient is retained only if its centered local support lies fully inside the interval and does not overlap any raw missing sample. In practice this is implemented as a cone-of-influence half-width proportional to the scale, with raw gaps treated as additional local boundaries.

## Output conventions

The saved outputs retain the old external structure:

- `Sfuncs` / `StructureFunctions`: increment-equivalent wavelet-moment surrogates with physical $[X]^q$ units
- `WaveletMoments`: raw conditioned wavelet-coefficient moments
- `Spectra`: second-order raw wavelet moments divided by the exact response-energy integral
- `LegacySpectraBandwidth`: the previous bandwidth-based normalization, kept only for comparison
- `flucts`: optional per-sample tables, including `polarity`, `local_polarity`, and `polarity_used` when requested
- `overall_align_angles`: per-level alignment summaries, including `sig_c_scale` / `sig_r_scale` and the preserved local-ratio summaries
- `meta`: scale grid, equivalent lags, bandwidths, response-energy integrals, and cone-of-influence widths

When `flucts` is saved, the default magnetic fluctuation export is now always present through the `dB_*` columns, with the units controlled by `return_B_in_vel_units`.

## Scientific interpretation

The pipeline is a **local-angle-conditioned reduced multiscale analysis**. It does not uniquely reconstruct the full 3D spectral tensor. The Gaussian background and even-DoG fluctuation are paired filters, not an exact additive decomposition $X=B_\ell+\delta B$. Its proper interpretation is the one accessible to a single-spacecraft time series under local Taylor mapping: conditioned multiscale statistics as functions of local field geometry and local fluctuation direction.
