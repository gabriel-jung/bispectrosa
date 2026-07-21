"""Audio layer: filterbanks, the mel Legendre modes, and the ModalBispectrum factories.

Turns a sampling rate into band layouts and 1-D modes sampled on the frequency
bins.
"""

import numpy as np

from .bispectrum import ModalBispectrum, legendre_modes, modal_index_pairs

__all__ = [
    "mel_filterbank",
    "triangular_filterbank",
    "mel_legendre_modes",
    "mel_legendre_modal_bispectrum",
    "mel_band_modal_bispectrum",
    "DEFAULT_SR",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_N_MELS",
    "DEFAULT_DEGREE",
]

#: Package defaults, tuned for speech: 16 kHz, 25 ms analysis window with 10 ms hop,
#: 80 mel bands, degree-12 pair basis. Defined once here; ``feature`` imports them.
DEFAULT_SR = 16000
DEFAULT_N_FFT = 400
DEFAULT_HOP_LENGTH = 160
DEFAULT_N_MELS = 80
DEFAULT_DEGREE = 12


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised via optional extra
        raise ImportError(
            "This path needs librosa; install the audio extra: pip install 'bispectrosa[audio]'"
        ) from exc
    return librosa


# --------------------------------------------------------------------------- #
# Filterbanks
# --------------------------------------------------------------------------- #
def mel_filterbank(
    *,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Slaney mel filterbank: thin wrapper over ``librosa.filters.mel`` (``htk=False``).

    Parameters
    ----------
    sr : int
        Sampling rate in Hz.
    n_fft : int
        FFT length; the filters are sampled on the ``n_fft // 2 + 1`` bins.
    n_mels : int
        Number of mel bands.
    fmin, fmax : float
        Frequency range covered by the bank, in Hz; ``fmax=None`` means ``sr / 2``.
    dtype : np.dtype
        Output dtype (default float32).

    Returns
    -------
    np.ndarray, shape (n_mels, n_fft // 2 + 1), of ``dtype``
        Filter weights ``H[b, k]``.
    """
    librosa = _require_librosa()
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=dtype)


def triangular_filterbank(
    edges_hz, *, sr: int, n_fft: int, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Generic triangular filterbank.

    One Slaney-normalized triangle per band, placed on the edges you pass;
    mel-spaced edges (``librosa.mel_frequencies(n_mels + 2)``) give back the
    mel filterbank of :func:`mel_filterbank`, up to float rounding.

    Parameters
    ----------
    edges_hz : array-like, shape (M + 2,)
        Band edges in Hz, ascending; band ``b`` is the triangle
        ``[edges[b], edges[b + 1], edges[b + 2]]``.
    sr : int
        Sampling rate in Hz.
    n_fft : int
        FFT length; the filters are sampled on the ``n_fft // 2 + 1`` bins.
    dtype : np.dtype
        Output dtype (default float32).

    Returns
    -------
    np.ndarray, shape (M, n_fft // 2 + 1), of ``dtype``
        Filter weights, each triangle area-normalized to unit area in Hz.
    """
    edges = np.asarray(edges_hz, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 3:
        raise ValueError(f"edges_hz must be 1-D with at least 3 edges, got shape {edges.shape}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            "edges_hz must be strictly increasing; a repeated edge makes a "
            "zero-width triangle slope (0 / 0 -> NaN weights)"
        )
    M = len(edges) - 2
    fft_f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fdiff = np.diff(edges)
    ramps = edges[:, None] - fft_f[None, :]
    # band m rises along [edges[m], edges[m+1]] and falls along [edges[m+1], edges[m+2]]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    fb = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (edges[2 : M + 2] - edges[0:M])
    fb *= enorm[:, None]
    return fb.astype(dtype)


# --------------------------------------------------------------------------- #
# Modal bispectrum modes: Legendre on the band index, smeared through the filterbank
# --------------------------------------------------------------------------- #
def mel_legendre_modes(
    degree: int,
    *,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """The mel Legendre modes behind :func:`bispectrosa.feature.mel_bispectrogram`.

    ``tilde_q[p, k] = sum_b P_p(2 b / (n_mels - 1) - 1) H[b, k]`` with ``H`` the
    Slaney filterbank of :func:`mel_filterbank`: Legendre evaluated on the band
    index rescaled to ``[-1, 1]``, smeared back onto the bins.

    Parameters
    ----------
    degree : int
        Maximum mode order (rows ``0..degree``).
    sr, n_fft, n_mels, fmin, fmax
        Filterbank parameters, see :func:`mel_filterbank`.
    dtype : np.dtype
        Working and output dtype (default float32).

    Returns
    -------
    np.ndarray, shape (degree + 1, n_fft // 2 + 1), of ``dtype``
        Modes ``tilde_q[p, k]``.
    """
    mel = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=dtype)
    leg = legendre_modes(np.linspace(-1.0, 1.0, n_mels), degree).astype(dtype)
    return (leg @ mel).astype(dtype, copy=False)


# --------------------------------------------------------------------------- #
# ModalBispectrum factories: the objects handed to the core
# --------------------------------------------------------------------------- #
def mel_legendre_modal_bispectrum(
    degree: int = DEFAULT_DEGREE,
    *,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    dtype: np.dtype = np.float32,
) -> ModalBispectrum:
    """Modal estimator on the mel Legendre modes.

    Builds the modes with :func:`mel_legendre_modes`, takes every mode pair
    ``(p, r)`` with ``p + r <= degree``, and returns the
    :class:`~bispectrosa.bispectrum.ModalBispectrum` that estimates one
    ``beta`` coefficient per pair from an STFT.
    :func:`bispectrosa.feature.mel_bispectrogram` uses exactly this object.

    Parameters
    ----------
    degree : int
        Maximum total degree of the pair basis (``p + r <= degree``).
    sr, n_fft, n_mels, fmin, fmax
        Filterbank parameters, see :func:`mel_filterbank`.
    dtype : np.dtype
        Working precision of the basis (default float32).

    Returns
    -------
    ModalBispectrum
        With ``dim = floor((degree + 2)**2 / 4)`` pair coefficients and
        ``n_irfft = n_fft``.
    """
    modes = mel_legendre_modes(
        degree, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=dtype
    )
    return ModalBispectrum(modes=modes, pairs=modal_index_pairs(degree), n_irfft=n_fft, dtype=dtype)


def mel_band_modal_bispectrum(
    n_mels: int,
    *,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    fmin: float = 0.0,
    fmax: float | None = None,
    dtype: np.dtype = np.float32,
) -> ModalBispectrum:
    """Modal estimator on the mel bands themselves.

    The modes are the :func:`mel_filterbank` rows, one per band; every band
    pair ``(b1, b2)`` with ``0 <= b1 <= b2 < n_mels`` gets one ``beta``
    coefficient: the coupling whose legs fall in bands ``b1`` and ``b2``,
    with the third leg left unresolved (the estimator's constant third leg).

    A mid-size feature between the few dozen Legendre coefficients and the
    full grid: ``n_mels * (n_mels + 1) / 2`` coefficients, localized per
    band pair instead of smooth and global.

    Parameters
    ----------
    n_mels : int
        Number of mel bands; ``n_mels * (n_mels + 1) / 2`` coefficients.
        Keep it small (10-30) unless you want thousands of coefficients: when
        the bands get narrower than 2 STFT bins the basis degenerates (a
        warning fires) and Gram-based reconstruction turns ill-conditioned.
    sr, n_fft, fmin, fmax
        Filterbank parameters, see :func:`mel_filterbank`.
    dtype : np.dtype
        Working precision of the basis (default float32).

    Returns
    -------
    ModalBispectrum
        With ``n_mels * (n_mels + 1) / 2`` band-pair coefficients and
        ``n_irfft = n_fft``.
    """
    modes = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=dtype)
    narrow = int((np.count_nonzero(modes, axis=1) < 2).sum())
    if narrow:
        import warnings

        warnings.warn(
            f"{narrow} of {n_mels} mel bands span fewer than 2 STFT bins at n_fft={n_fft}: "
            "the band basis is nearly degenerate and Gram-based reconstruction will be "
            "ill-conditioned. Use fewer bands or a larger n_fft.",
            stacklevel=2,
        )
    pairs = [(p, r) for p in range(n_mels) for r in range(p, n_mels)]
    return ModalBispectrum(modes=modes, pairs=pairs, n_irfft=n_fft, dtype=dtype)
