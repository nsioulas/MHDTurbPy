# Revision notes: rev4 follow-up audit

This revision addresses additional logic problems found after the previous cleanup pass.

## Fixed in `three_D_funcs.py`

1. **Undefined projection bases no longer collapse to zero**
   - `_perp_vector_stacked(...)` now propagates invalid bases as `NaN` instead of silently using zero parallel components.
   - This prevents fabricated perpendicular directions and contaminated projections when `|B_l|` is too small or non-finite.

2. **Unknown-frame polarity is no longer guessed from the first component**
   - `_background_polarity(...)` now uses:
     - `sign(Br)` for RTN,
     - `sign(Bx)` for GSE,
     - unity fallback for unknown frames.
   - This avoids an unjustified polarity correction when the radial-like axis is not identified.

3. **Sigma diagnostics are always constructed**
   - `sig_c` / `sig_r` no longer depend on the alignment-angle flag merely because they were computed inside the same helper.
   - The level objects now always carry the sigma-family summaries; the flag still controls whether the interval-level summary arrays are populated in the outward report.

4. **Raw contextual diagnostics now use aligned raw series**
   - The transform still uses gap-filled series internally, but saved contextual quantities such as `Vsw`, `Bmod`, and the legacy `external_raw__*` diagnostics are now evaluated on the aligned raw series instead of on the gap-filled transform inputs.

5. **Raw contextual diagnostics are now masked by the current level support**
   - Context outputs are masked with the level validity mask before export, so they cannot leak unsupported samples into the row-wise tables.

6. **`di_mean` and `Vsw_mean` are now derived from the raw aligned series**
   - These metadata summaries no longer depend on the gap-filled arrays used internally by the FFT backend.

7. **Polarity metadata is explicit**
   - The returned raw context now records `polarity_definition`.

8. **No-warning conditional averaging**
   - `_moments_from_precomputed_powers(...)` no longer emits `nanmean` warnings when the selected bucket contains only invalid values.
   - `_alignment_stats_by_level(...)` uses a safe median helper to avoid all-NaN median warnings.

## Documentation update

The appendix now states two additional facts explicitly:

- raw contextual diagnostics are evaluated on aligned raw series rather than on the gap-filled transform inputs;
- polarity correction falls back to unity for unknown frames instead of guessing from the first component.
