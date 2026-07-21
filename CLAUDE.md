# bispectrosa

Public, pip-installable library of bispectral (third-order) audio features, librosa-style API.
Entry point: `bispectrosa.mel_bispectrogram(y, sr=sr)` (all parameters after `y` are keyword-only). See `README.md`.

## Core vs audio layer

- `bispectrum.py` is the generic core: `numpy`/`scipy` only, no `sr`/Hz/mel/filterbanks, and it
  **never imports librosa** (`tests/test_import_guard.py` enforces this). `ModalBispectrum` is the
  hand-off object.
- `filters.py` / `feature.py` / `display.py` own the STFT, mel/log warps, and librosa (lazy,
  optional `[audio]` extra). New domains get a module beside `feature.py`; the core is untouched.

## Numerical identity

Feature outputs are frozen in `tests/fixtures/research_ref.npz` and checked bit-for-bit by
`tests/test_regression.py`. **Never change a feature's output** except via a deliberate fixture
update (a green regression after an output change means you moved the fixtures too).

- Mel is Slaney `librosa.filters.mel` only, never HTK. `triangular_filterbank` is for custom band
  layouts, not a second mel path.
- Coefficients are degree-ordered: degree-`d` output is the `modal_pair_dim(d)`-length prefix of
  any higher degree. Tested; preserve it.

## Conventions

- Semver from 0.1.0. Public API is each module's `__all__`. Additive changes are fine; renames,
  removals, and output changes are breaking.
- Physics tests assert only robust properties (triplet detection, degree nesting, Gram PSD), never
  phase direction or reconstruction correlation (convention-dependent; show those in the example).

## Dev

```bash
uv sync --group dev
uv run pytest -q
```
