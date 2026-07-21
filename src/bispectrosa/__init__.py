"""bispectrosa: bispectral features for audio, modal estimation, librosa-style API.

Quick start
-----------
>>> import numpy as np, bispectrosa as bs
>>> y = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
>>> B = bs.mel_bispectrogram(y, sr=16000)      # (49, n_frames) at degree 12
>>> v = bs.time_pool(B)                      # (49,) utterance-level vector

Layers
------
- :mod:`bispectrosa.bispectrum`   generic core (numpy/scipy only): the full and modal
  estimators (``raw_bispectrum``, ``ModalBispectrum``), Legendre modes, pair kernels,
  Gram + reconstruction. No audio, no ``sr``.
- :mod:`bispectrosa.filters`   Slaney mel warp + the ``ModalBispectrum`` factory.
- :mod:`bispectrosa.feature`   librosa-style front door: ``mel_bispectrogram``,
  ``mel_spectrogram``, ``stft``, ``time_pool``.
- :mod:`bispectrosa.display`   bispectrum plotting.
"""

import importlib.metadata as _im

from .bispectrum import (
    ModalBispectrum,
    average_bispectrum_at_triplets,
    full_bispectrum,
    legendre_modes,
    modal_gram_matrix,
    modal_index_pairs,
    modal_pair_dim,
    modal_pair_kernel,
    modal_shape_correlation,
    project_bispectrum,
    raw_bispectrum,
    reconstruct_bispectrum,
    rescale_to_symmetric,
    snr_bispectrum,
    valid_frequency_pairs,
)

# Single-sourced from pyproject.toml via the installed package metadata.
try:
    __version__ = _im.version("bispectrosa")
except _im.PackageNotFoundError:  # source tree used without an install
    __version__ = "0.0.0"

# Names that live in optional (lazily imported) submodules, grouped by module;
# each set mirrors that module's ``__all__`` (asserted in tests/test_api.py).
_LAZY = {
    "feature": {
        "mel_bispectrogram",
        "mel_spectrogram",
        "mel_bin_bispectrum",
        "stft",
        "time_pool",
        "BISPECTROGRAM_EPS",
        "signed_log",
    },
    "filters": {
        "mel_legendre_modal_bispectrum",
        "mel_band_modal_bispectrum",
        "mel_legendre_modes",
        "mel_filterbank",
        "triangular_filterbank",
        "DEFAULT_SR",
        "DEFAULT_N_FFT",
        "DEFAULT_HOP_LENGTH",
        "DEFAULT_N_MELS",
        "DEFAULT_DEGREE",
    },
    "display": {
        "plot_bispectrum",
        "animate_bispectrum",
        "symlog_norm",
        "symlog_colorbar",
        "set_freq_scale",
    },
}

__all__ = [
    "__version__",
    # core
    "ModalBispectrum",
    "legendre_modes",
    "rescale_to_symmetric",
    "modal_index_pairs",
    "modal_pair_dim",
    "modal_pair_kernel",
    "valid_frequency_pairs",
    "average_bispectrum_at_triplets",
    "raw_bispectrum",
    "full_bispectrum",
    "snr_bispectrum",
    "modal_gram_matrix",
    "project_bispectrum",
    "reconstruct_bispectrum",
    "modal_shape_correlation",
    # audio layer (imported lazily below to keep core librosa-free)
    *sorted(set().union(*_LAZY.values())),
]


def __getattr__(name):
    """Lazily surface audio/viz-layer names so ``import bispectrosa`` needs no librosa."""
    import importlib

    for module, names in _LAZY.items():
        if name in names:
            obj = getattr(importlib.import_module(f".{module}", __name__), name)
            globals()[name] = obj  # cache: later accesses skip __getattr__
            return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include the lazily-exposed names in tab-completion / ``dir()``."""
    return sorted(set(__all__) | set(globals()))
