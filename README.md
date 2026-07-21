<h1 align="center">bispectrosa</h1>

<p align="center">
  <strong>Bispectral features for audio.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/bispectrosa/"><img src="https://img.shields.io/pypi/v/bispectrosa?style=flat-square&color=7c3aed" alt="PyPI" /></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.14-7c3aed?style=flat-square" alt="Python 3.10-3.14" />
  <a href="https://github.com/gabriel-jung/bispectrosa/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/gabriel-jung/bispectrosa/ci.yml?style=flat-square&color=7c3aed" alt="CI" /></a>
  <a href="https://github.com/gabriel-jung/bispectrosa/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-7c3aed?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#quick-start">Quick start</a> &bull;
  <a href="#what-it-computes-bispectrum-modal-coefficients">What it computes</a> &bull;
  <a href="#standard-bispectrum-reconstruction">Reconstruction</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#documentation">Docs</a> &bull;
  <a href="#citing">Cite</a>
</p>

The bispectrum `B(f₁, f₂) = ⟨X(f₁) X(f₂) X*(f₁+f₂)⟩` is the third-order spectrum: the
correlation between the frequency components at `f₁`, `f₂`, and `f₁+f₂`, a part of the
signal the power spectrum cannot see. `bispectrosa` turns it into a practical audio
feature: one call, waveform in, a few dozen bispectral coefficients per frame out.

The API follows librosa conventions (waveform in, feature out, time as the last axis).

## Install

```bash
pip install "bispectrosa[audio]"      # audio front-end (pulls in librosa)
pip install "bispectrosa[audio,viz]"  # + plotting
```

The generic core imports with only `numpy` and `scipy`; `librosa` is needed only for the
audio layer.

## Quick start

```python
import numpy as np, bispectrosa as bs

y = np.random.default_rng(0).standard_normal(16000).astype(np.float32)

B = bs.mel_bispectrogram(y, sr=16000)   # (49, n_frames) at degree 12
v = bs.time_pool(B)                     # (49,) utterance-level vector
```

The signature follows librosa: `y` is positional, everything else is keyword-only
(`sr`, `degree`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`, `window`, `center`, ...).
Defaults are tuned for speech (`sr=16000`, `n_fft=400`, `hop_length=160`, `n_mels=80`),
not librosa's music defaults, so set them explicitly for other domains. Coefficients come
out time-last, so librosa post-processing composes directly, e.g.
`librosa.feature.delta(B)` for delta features.

To compute the STFT once and share it across features, pass `S=` (the complex STFT,
since the bispectrum needs phase) instead of `y`:

```python
X = bs.stft(y)                          # one transform
B = bs.mel_bispectrogram(S=X, sr=16000) # third-order feature
M = bs.mel_spectrogram(S=X, sr=16000)            # second-order companion
```

## What it computes: bispectrum modal coefficients

The power spectrum treats every frequency separately: it carries no information about
how components relate to one another. The bispectrum is the correlation between three
components at once, those at `f₁`, `f₂`, and `f₁+f₂`: it averages to zero when they
fluctuate independently and survives when they are correlated (nonlinearity). Harmonic
structure, timbre, and source nonlinearities live there, invisible to second-order
features.

Computed directly, the bispectrum is a large, noisy 2-D matrix per frame
(`raw_bispectrum`, kept as the reference estimator). The main feature
`mel_bispectrogram` uses a **modal estimator** instead, borrowed from cosmology, where
the same compression problem appears: expand the bispectrum on a small basis of smooth
functions built from mel-binned modes, and keep the coefficients `β`; 49 numbers per
frame at the default degree, signed-log compressed like a log-mel spectrogram. The full
derivation is in
[`docs/theory.md`](https://github.com/gabriel-jung/bispectrosa/blob/main/docs/theory.md).

## Standard bispectrum reconstruction

The coefficients determine a best least-squares fit of the bispectrum by the basis;
rebuilding it shows what the basis captures.

```python
import numpy as np, bispectrosa as bs
from bispectrosa import bispectrum

Xstft = bs.stft(y)
F = Xstft.shape[0]
degree = 7
pairs = bs.modal_index_pairs(degree)
modes = bs.legendre_modes(np.linspace(-1, 1, F), degree)

beta = bispectrum.project_bispectrum(Xstft, pairs, modes)
gram = bs.modal_gram_matrix(pairs, modes)
x, _, B = bispectrum.reconstruct_bispectrum(beta, pairs, modes, gram=gram, return_full=True)
# x is a normalized [-1, 1] axis: here x = -1 is 0 Hz and x = +1 is sr / 2
```

## Architecture

The generic math is separated from the audio choices, so the core (bispectrum.py) can be
reused for other domains:

| module | role |
|--------|------|
| `bispectrum` | generic core (`numpy`/`scipy` only): the full and modal estimators (`raw_bispectrum`, `ModalBispectrum`), Legendre modes, pair kernels, Gram matrix, reconstruction. Knows nothing about `sr`, Hz, or mel. |
| `filters` | audio layer: lays the 1-D modes on a frequency grid (Slaney mel, or custom triangular filterbanks) and builds the `ModalBispectrum` the core consumes. |
| `feature` | front door: `mel_bispectrogram`, `mel_spectrogram`, `mel_bin_bispectrum`, `stft`, `time_pool`. |
| `display` | bispectrum plotting. |

## Documentation

- [`docs/theory.md`](https://github.com/gabriel-jung/bispectrosa/blob/main/docs/theory.md):
  the bispectrum and the modal estimator, from definition to the fast form the code
  computes.
- API reference: the numpydoc docstrings (`help(bs.mel_bispectrogram)`, or any IDE hover).
- [`examples/intro.ipynb`](https://github.com/gabriel-jung/bispectrosa/blob/main/examples/intro.ipynb):
  the guided tour, from a controlled phase-coupled signal to real recordings: the
  feature, the raw grid, reconstruction, and cost.
- [`examples/animate_bispectrum.py`](https://github.com/gabriel-jung/bispectrosa/blob/main/examples/animate_bispectrum.py):
  the per-frame bispectrum as a movie.

## Development

```bash
uv sync --group dev     # install with dev dependencies
uv run pytest -q        # run the test suite
uv run ruff check .     # lint
uv run ruff format .    # format
```

Feature outputs are frozen bit-for-bit by `tests/test_regression.py`; see
[`CHANGELOG.md`](https://github.com/gabriel-jung/bispectrosa/blob/main/CHANGELOG.md)
for the release history.

## Citing

If you use `bispectrosa` in your research, please cite it; see
[`CITATION.cff`](https://github.com/gabriel-jung/bispectrosa/blob/main/CITATION.cff).

## License

MIT © 2026 Gabriel Jung. See
[`LICENSE`](https://github.com/gabriel-jung/bispectrosa/blob/main/LICENSE).
