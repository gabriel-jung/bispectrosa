# Regression fixtures

`research_ref.npz` holds reference feature outputs frozen from `bispectrosa`
itself (regenerated 2026-07-17; the original lineage from the private research
`bispectrum` package ended there). `tests/test_regression.py` asserts
`bispectrosa` keeps reproducing them bit-for-bit (float32 floor), so any
accidental change to the feature numerics is caught immediately.

Contents (computed on a fixed seeded signal, also stored as `y`):

| key | source | shape |
|-----|--------|-------|
| `y` | the input waveform (16 kHz) | (16000,) |
| `mel_modal_pair_pooled` | `time_pool(mel_bispectrogram(y, sr=16000, degree=12))` | (49,) |
| `mel_modal_pair_frames` | `mel_bispectrogram(y, sr=16000, degree=12).T` | (101, 49) |
| `direct_f1`, `direct_B` | `raw_bispectrum(stft(y))`, bin axis in Hz (`k * sr / n_fft`) | (201,), (201, 201) |

**Regenerating** (only when an output change is deliberate): recompute the
table above with the current `bispectrosa` and overwrite this file. A green
regression run after an output change means the fixture moved too; see the
numerical-identity rules in `CLAUDE.md` and the change log.
