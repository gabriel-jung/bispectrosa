# Changelog

All notable changes to bispectrosa are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [0.1.0] - 2026-07-21

Initial version. Bispectral (third-order) audio features with a librosa-style API:

- `mel_bispectrogram(y, sr=sr)`: the main feature, 49 signed modal coefficients per
  frame (degree-12 Legendre basis; `basis="bands"` for localized mel-band
  coefficients), with `time_pool` for one vector per clip.
- `raw_bispectrum` / `full_bispectrum`: the explicit frame-averaged estimate, as flat
  valid triplets or the mirrored square matrix, band-limited by `kmin`/`kmax`.
- `project_bispectrum`, `reconstruct_bispectrum`, `modal_gram_matrix`: the modal
  estimator's building blocks, with bin-aligned reconstruction via `kmin`/`kmax`.
- `snr_bispectrum` (signal-to-noise form `B / sqrt(P1 P2 P3)`), `mel_bin_bispectrum`
  (mel binning on both frequency axes), `modal_shape_correlation` (shape correlation
  of coefficient vectors under the basis Gram metric).
- `stft`, `mel_spectrogram`, and display helpers (`plot_bispectrum`,
  `animate_bispectrum`, `symlog_norm`).
- Docs: `docs/theory.md` and the guided tour `examples/intro.ipynb`.
