"""Front door: librosa-style functional API for audio bispectrum features::

    import bispectrosa as bs
    B = bs.mel_bispectrogram(y, sr=16000)      # (49, n_frames) at degree 12

Organized as the pipeline runs: the analysis transform (:func:`stft`), the
signed-log compression it feeds (:func:`signed_log`), the features
(:func:`mel_spectrogram`, :func:`mel_bispectrogram`), and the time pooling
that turns a feature into an utterance vector (:func:`time_pool`).
"""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from .filters import (
    DEFAULT_DEGREE,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_SR,
    _require_librosa,
    mel_band_modal_bispectrum,
    mel_filterbank,
    mel_legendre_modal_bispectrum,
)

__all__ = [
    "stft",
    "signed_log",
    "BISPECTROGRAM_EPS",
    "mel_spectrogram",
    "mel_bin_bispectrum",
    "mel_bispectrogram",
    "time_pool",
]


def mel_bin_bispectrum(
    B: ArrayLike,
    *,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    kmin: int = 1,
    min_coverage: float = 0.02,
) -> np.ndarray:
    """Bin a square bispectrum onto mel bands on both frequency axes.

    Computes ``H B H^T`` with ``H`` the mel filterbank: the 2-D analogue of
    mel-binning a spectrum. Useful to compare a bispectrum with a
    mel-resolution representation (like the modal rebuild) at the resolution
    that representation lives on.

    Parameters
    ----------
    B : array-like, shape (n, n)
        Square bispectrum whose grid starts at absolute bin ``kmin`` (as
        returned by :func:`~bispectrosa.bispectrum.raw_bispectrum` with
        ``return_full=True``). NaN cells contribute nothing.
    sr, n_fft, n_mels, fmin, fmax
        Mel filterbank parameters, see
        :func:`~bispectrosa.filters.mel_filterbank`.
    kmin : int
        Absolute bin index of the grid's first row/column.
    min_coverage : float
        Mel cells whose share of valid bins falls below ``min_coverage``
        times the best-covered cell are returned as NaN (they lie outside
        the valid region).

    Returns
    -------
    np.ndarray, shape (n_mels, n_mels)
        The mel-binned bispectrum, float64, NaN where coverage is too low.

    Raises
    ------
    ValueError
        If ``B`` is not square or the grid does not fit the filterbank.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"need a square (n, n) bispectrum, got shape {B.shape}")
    H = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).astype(np.float64)
    F = H.shape[1]
    if kmin < 0 or kmin + B.shape[0] > F:
        raise ValueError(
            f"grid [kmin, kmin + n) = [{kmin}, {kmin + B.shape[0]}) does not fit the {F}-bin filterbank"
        )
    hi = kmin + B.shape[0]
    filled = np.zeros((F, F))
    filled[kmin:hi, kmin:hi] = np.where(np.isfinite(B), B, 0.0)
    support = np.zeros((F, F))
    support[kmin:hi, kmin:hi] = np.isfinite(B)
    weight = H @ support @ H.T
    out = H @ filled @ H.T
    out[weight <= min_coverage * weight.max()] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Analysis transform
# --------------------------------------------------------------------------- #
def stft(
    y: np.ndarray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    win_length: int | None = None,
    window: str | tuple | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Complex STFT, the shared analysing transform (wraps ``librosa.stft``).

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Waveform.
    n_fft : int
        Frame and FFT length in samples.
    hop_length : int
        Samples between frame centers.
    win_length : int, optional
        Analysis window length; ``None`` (default) uses ``n_fft``. A shorter
        window is zero-padded to ``n_fft`` before the FFT.
    window : str, tuple, or np.ndarray
        Window spec passed to ``librosa.stft`` (default ``"hann"``).
    center : bool
        Pad ``y`` by ``n_fft // 2`` on each side (see ``pad_mode``) so frame
        ``t`` is centered on sample ``t * hop_length``.
    pad_mode : str
        Padding used when ``center=True`` (default ``"constant"``, zeros).

    Returns
    -------
    np.ndarray, shape (n_fft // 2 + 1, n_frames), complex
        Precision follows ``y``: complex64 for float32 input, complex128 for
        float64.

    Notes
    -----
    No mean subtraction (detrending) is applied, staying faithful to
    ``librosa.stft``. The bispectrum convention of working with mean-free
    signals is instead enforced downstream: the estimators default to
    ``kmin=1`` (the zero-frequency mode never enters) and the mel modes
    carry no weight at the zero frequency anyway, so
    :func:`mel_bispectrogram` is insensitive to the signal mean. Including
    the mean is an explicit opt-in (``kmin=0``); see
    :func:`bispectrosa.bispectrum.project_bispectrum`.
    """
    librosa = _require_librosa()
    return librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )


def _stft_or_S(y, S, *, n_fft, hop_length, win_length, window, center, pad_mode):
    """Return the STFT a feature will run on: computed from ``y``, or ``S`` as given.

    Exactly one of ``y`` (waveform) or ``S`` (precomputed complex STFT) must
    be passed. A given ``S`` must have ``n_fft // 2 + 1`` frequency bins,
    otherwise the mel filters downstream would sit on the wrong bins.
    """
    if y is not None and S is not None:
        raise ValueError("pass either y (a waveform) or S (a precomputed complex STFT), not both")
    if y is None and S is None:
        raise ValueError("provide y (a waveform) or S (a precomputed complex STFT)")
    if S is None:
        return stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
    if S.shape[0] != n_fft // 2 + 1:
        raise ValueError(
            f"S has {S.shape[0]} frequency bins but n_fft={n_fft} implies "
            f"{n_fft // 2 + 1}; pass the n_fft that S was built with"
        )
    return S


# --------------------------------------------------------------------------- #
# Signed-log compression (the bispectrogram features' log warp)
# --------------------------------------------------------------------------- #
#: Default signed-log floor for the bispectrogram features (see :func:`signed_log`).
BISPECTROGRAM_EPS = 1e-15


def signed_log(x: ArrayLike, eps: float = BISPECTROGRAM_EPS) -> np.ndarray:
    """Compress ``x`` to ``sign(x) * log1p(|x| / eps)``.

    Parameters
    ----------
    x : array-like
        Values to compress.
    eps : float
        Compression floor.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x)
    with np.errstate(over="ignore"):
        compressed = np.log1p(np.abs(x) / eps)
    # float32 input can overflow the division (|x| > float32 max * eps) even
    # though the compressed value itself is representable; redo those entries
    # in float64. In-range entries keep their original (dtype-native) values.
    overflow = np.isinf(compressed) & np.isfinite(x)
    if overflow.any():
        # recompute only the flagged entries: overflow is typically a handful
        # of saturated values, not worth a full-array float64 pass
        exact = np.log1p(np.abs(x[overflow]).astype(np.float64) / eps)
        compressed[overflow] = exact.astype(compressed.dtype)
    return np.sign(x) * compressed


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def mel_spectrogram(
    y: np.ndarray | None = None,
    *,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    top_db: float | None = 80.0,
    S: np.ndarray | None = None,
    win_length: int | None = None,
    window: str | tuple | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Per-frame mel spectrogram in power dB (the standard log-mel feature).

    The second-order companion feature to :func:`mel_bispectrogram`: the mel
    projection uses the same :func:`~bispectrosa.filters.mel_filterbank` that
    builds the modal basis, so the two features share their bands exactly.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,), optional
        Waveform. Provide exactly one of ``y`` or ``S``.
    sr : int
        Sampling rate in Hz.
    n_fft, hop_length : int
        STFT frame length and hop, in samples.
    n_mels : int
        Number of mel bands.
    fmin, fmax : float
        Frequency range of the mel bank, in Hz; ``fmax=None`` means ``sr / 2``.
    top_db : float, optional
        Clip the output to the top ``top_db`` decibels below the clip's peak
        (``librosa.power_to_db``'s default 80). This couples every frame to
        the loudest one; pass ``None`` to keep the full dynamic range.
    S : np.ndarray, shape (n_fft // 2 + 1, n_frames), complex, optional
        Precomputed complex STFT (e.g. from :func:`stft`), reused instead of
        transforming ``y`` so one STFT can feed both this and
        :func:`mel_bispectrogram`. When given, the STFT parameters
        (``hop_length``, ``win_length``, ``window``, ``center``, ``pad_mode``)
        are ignored (``S`` already encodes them); ``n_fft`` must match ``S``.
    win_length, window, center, pad_mode
        STFT analysis parameters, see :func:`stft` (ignored when ``S`` is given).

    Returns
    -------
    np.ndarray, shape (n_mels, n_frames), float32
        Mel powers in dB (``10 log10``).
    """
    librosa = _require_librosa()
    S = _stft_or_S(
        y,
        S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = fb @ (np.abs(S) ** 2)
    return librosa.power_to_db(mel, top_db=top_db).astype(np.float32)


@lru_cache(maxsize=32)
def _mel_modal_bispectrum_cached(basis, degree, sr, n_fft, n_mels, fmin, fmax, dtype):
    # the estimator is deterministic in its arguments and dominates the
    # cost of short-clip calls; treated as immutable, never handed out mutably
    if basis == "legendre":
        return mel_legendre_modal_bispectrum(
            degree, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=dtype
        )
    if basis == "bands":
        return mel_band_modal_bispectrum(
            n_mels, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, dtype=dtype
        )
    raise ValueError(f"basis must be 'legendre' or 'bands', got {basis!r}")


def mel_bispectrogram(
    y: np.ndarray | None = None,
    *,
    sr: int = DEFAULT_SR,
    basis: str = "legendre",
    degree: int = DEFAULT_DEGREE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    fmin: float = 0.0,
    fmax: float | None = None,
    eps: float = BISPECTROGRAM_EPS,
    log: bool = True,
    dtype: np.dtype = np.float32,
    workers: int | None = None,
    S: np.ndarray | None = None,
    win_length: int | None = None,
    window: str | tuple | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Mel bispectrogram: modal pair coefficients ``beta`` per frame.

    The package's main feature: STFT, beta estimation on a mel mode family,
    signed-log compression. Pool over time with :func:`time_pool` for an
    utterance vector.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,), optional
        Waveform. Provide exactly one of ``y`` or ``S``.
    sr : int
        Sampling rate in Hz.
    basis : {"legendre", "bands"}
        Mode family behind the coefficients. ``"legendre"`` (default): the
        smooth mel Legendre pair basis
        (:func:`~bispectrosa.filters.mel_legendre_modal_bispectrum`), a few
        dozen global coefficients ordered by total degree, so a row prefix
        reads any lower degree. ``"bands"``: the mel bands themselves
        (:func:`~bispectrosa.filters.mel_band_modal_bispectrum`), one
        coefficient per band pair, localized instead of smooth; ``degree``
        is ignored, and ``n_mels`` sets the size (keep it small, 10-30).
    degree : int
        Maximum total degree of the Legendre pair basis (``p + r <= degree``;
        ``basis="legendre"`` only).
    n_fft, hop_length : int
        STFT frame length and hop, in samples.
    n_mels : int
        Number of mel bands behind the mode basis.
    fmin, fmax : float
        Frequency range of the mel bank behind the mode basis, in Hz;
        ``fmax=None`` means ``sr / 2``.
    eps : float
        Signed-log floor (see :func:`signed_log`; ignored when ``log=False``).
    log : bool
        Apply the signed-log compression (``False`` for raw coefficients).
    dtype : np.dtype
        Working precision of the estimator (default float32; pass
        ``np.float64`` to study the estimator's numerics).
    workers : int, optional
        Threads for the estimator's IRFFT and reduction, see
        :meth:`~bispectrosa.bispectrum.ModalBispectrum.estimate_beta` (bit-identical
        for any value; ``None`` auto-tunes, an integer sets the count).
    S : np.ndarray, shape (n_fft // 2 + 1, n_frames), complex, optional
        Precomputed complex STFT (e.g. from :func:`stft`), reused instead of
        transforming ``y`` so one STFT can feed both this and
        :func:`mel_spectrogram`. Must be built on the same ``n_fft``; the
        STFT parameters (``hop_length``, ``win_length``, ``window``,
        ``center``, ``pad_mode``) are then ignored.
    win_length, window, center, pad_mode
        STFT analysis parameters, see :func:`stft` (ignored when ``S`` is given).

    Returns
    -------
    np.ndarray, shape (n_coeffs, n_frames), of ``dtype``
        One ``beta`` column per frame. ``n_coeffs`` is
        ``floor((degree + 2)**2 / 4)`` for ``basis="legendre"`` (49 at the
        default ``degree=12``) and ``n_mels * (n_mels + 1) / 2`` for
        ``basis="bands"``.

    Notes
    -----
    For a fully custom mode family, build a
    :class:`~bispectrosa.bispectrum.ModalBispectrum` (e.g. on a
    :func:`~bispectrosa.filters.triangular_filterbank`) and run the pipeline
    yourself: ``signed_log(mb.estimate_beta(stft(y, n_fft=mb.n_irfft))).T``.
    """
    mb = _mel_modal_bispectrum_cached(basis, degree, sr, n_fft, n_mels, fmin, fmax, np.dtype(dtype))
    S = _stft_or_S(
        y,
        S,
        n_fft=mb.n_irfft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    beta = mb.estimate_beta(S, workers=workers)  # (n_frames, n_coeffs)
    if log:
        beta = signed_log(beta, eps=eps)
    return beta.T  # (n_coeffs, n_frames), the estimator's dtype throughout


# --------------------------------------------------------------------------- #
# Time pooling
# --------------------------------------------------------------------------- #
def time_pool(feat: np.ndarray, *, fn: Callable = np.mean) -> np.ndarray:
    """Pool a per-frame feature over time.

    Parameters
    ----------
    feat : np.ndarray, shape (n_coeffs, n_frames)
        Per-frame feature, time last (librosa convention).
    fn : callable
        Reducer applied along the time axis (default ``np.mean``).

    Returns
    -------
    np.ndarray, shape (n_coeffs,)
        Utterance-level vector.
    """
    return fn(feat, axis=1)
