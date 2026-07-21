"""Generic bispectrum core.

Everything here works on arrays of complex Fourier coefficients ``X``
of shape ``(F, T)``: ``F`` frequencies as rows, ``T`` independent
realizations as columns. For one realization, the bispectrum at a pair of
frequencies ``(k1, k2)`` is

    B[k1, k2] = Re( X[k1] X[k2] X*[k1 + k2] ),

defined for ``k1 + k2 < F`` (the sum frequency must exist). The real part
is not a convention of convenience: the modal estimator's time-domain triple
product of real filtered signals can only measure ``Re``, so both estimators
target the same real object.

Two estimators (math in ``docs/theory.md``):

- :func:`raw_bispectrum`: B evaluated at every valid triplet, flat by default
  (a square grid with ``return_full=True``). Explicit and exact, but large
  (``~F**2 / 2`` values) and statistically noisy (each value comes from a
  single frequency triplet); best for inspection and validation.
- :meth:`ModalBispectrum.estimate_beta`: the modal bispectrum estimator.
  Per realization it returns a few dozen coefficients
  ``beta_pr = <B, Q_pr>`` (up to a fixed constant, see
  :func:`project_bispectrum`), the projections of B onto smooth pair kernels;
  each coefficient pools many triplets, and the separable form (an IRFFT
  and a sample sum) enables much faster computation than the full version.

:func:`modal_gram_matrix` and :func:`reconstruct_bispectrum` turn ``beta`` back
into the 2-D picture (the kernels are not orthogonal: solve
``Gamma alpha = beta``).
"""

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "average_bispectrum_at_triplets",
    "valid_frequency_pairs",
    "raw_bispectrum",
    "full_bispectrum",
    "snr_bispectrum",
    "rescale_to_symmetric",
    "legendre_modes",
    "modal_index_pairs",
    "modal_pair_dim",
    "modal_pair_kernel",
    "ModalBispectrum",
    "project_bispectrum",
    "modal_gram_matrix",
    "reconstruct_bispectrum",
    "modal_shape_correlation",
]


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
#: Elements per gather batch in :func:`average_bispectrum_at_triplets`: each
#: batch materializes three ``(step, T)`` complex copies of ~32 MB, instead of
#: the full ``(n_triplets, T)`` gather (gigabytes for long inputs).
_TRIPLET_GATHER_ELEMENTS = 1 << 21


def average_bispectrum_at_triplets(
    X: np.ndarray, i1: np.ndarray, i2: np.ndarray, i3: np.ndarray
) -> np.ndarray:
    """Bispectrum values at explicit frequency triplets, averaged over realizations.

    ``out[n] = mean_t Re( X[i1[n]] X[i2[n]] X*[i3[n]] )``: the elementwise
    estimator behind :func:`raw_bispectrum` and :func:`project_bispectrum`
    (pass ``i3 = i1 + i2`` for the bispectrum's valid triplets). Gathers the
    triplets in batches; each triplet's mean over realizations is independent
    of the others, so batching is bit-identical.

    Parameters
    ----------
    X : np.ndarray, shape (F, T), complex
        Fourier coefficients, one column per realization; ``T >= 1``.
    i1, i2, i3 : np.ndarray of int, 1-D, same length
        Frequency indices of the three legs, all within ``[0, F)``.

    Returns
    -------
    np.ndarray, shape (n_triplets,), float64

    Raises
    ------
    ValueError
        If ``X`` has no realizations (the average is undefined) or the index
        arrays are not 1-D.
    """
    T = X.shape[1]
    if T == 0:
        raise ValueError("X has no realizations (T = 0): the average is undefined")
    if not (np.ndim(i1) == np.ndim(i2) == np.ndim(i3) == 1):
        raise ValueError("i1, i2, i3 must be 1-D index arrays")
    step = max(1, _TRIPLET_GATHER_ELEMENTS // T)
    out = np.empty(i1.size, dtype=np.float64)
    for s in range(0, i1.size, step):
        rows = slice(s, s + step)
        out[rows] = np.real(X[i1[rows]] * X[i2[rows]] * np.conj(X[i3[rows]])).mean(axis=1)
    return out


def _resolve_window(kmin: int, kmax: int | None, n_freq: int) -> tuple[int, int]:
    """Default and validate the shared ``[kmin, kmax]`` frequency window (inclusive).

    The one window convention of both estimators: ``kmax=None`` means the
    last frequency, and the bounds must satisfy
    ``0 <= kmin <= kmax <= n_freq - 1``.
    """
    if kmax is None:
        kmax = n_freq - 1
    if not 0 <= kmin <= kmax <= n_freq - 1:
        raise ValueError(
            f"need 0 <= kmin <= kmax <= n_freq - 1 = {n_freq - 1}, got kmin={kmin}, kmax={kmax}"
        )
    return int(kmin), int(kmax)


def valid_frequency_pairs(n_freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the frequency-index pairs ``(i1, i2)`` with ``i1 + i2 < n_freq``.

    Parameters
    ----------
    n_freq : int
        Number of positive frequencies; pairs whose sum frequency
        ``i1 + i2`` would fall outside are excluded.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Int index arrays covering the valid triplets.

    Notes
    -----
    Both orderings of each pair are included (every valid triplet, not just
    the ``i2 <= i1`` half). This is consistent across
    :func:`modal_gram_matrix` / :func:`project_bispectrum` because the pair kernels
    are symmetric; off-diagonal contributions simply count twice on both sides.
    """
    i1, i2 = np.meshgrid(np.arange(n_freq), np.arange(n_freq), indexing="ij")
    keep = i1 + i2 < n_freq
    return i1[keep], i2[keep]


# --------------------------------------------------------------------------- #
# Full bispectrum estimator
# --------------------------------------------------------------------------- #
def raw_bispectrum(
    X, *, kmin: int = 1, kmax: int | None = None, average: bool = True, return_full: bool = False
):
    """Bispectrum on the Fourier transform's own frequency grid.

    The explicit reference estimator: every valid triplet of the ``F``
    frequencies, with the third factor read at exactly the sum frequency
    ``k1 + k2``. By default the valid triplets come back as flat arrays,
    each exactly once (the ``k2 <= k1`` half, no NaN padding);
    ``return_full=True`` assembles the square matrix instead, with the
    redundant half filled by the symmetry ``B(f1, f2) = B(f2, f1)`` (via
    :func:`full_bispectrum`) and only the cells whose sum frequency leaves
    the window NaN.

    Parameters
    ----------
    X : np.ndarray, shape (F, T), complex
        Fourier coefficients, one column per realization (upcast to
        complex128; the triple products run in double precision).
    kmin, kmax : int, optional
        Frequency window of the analysis, inclusive; ``kmax=None`` (default)
        means ``F - 1``. All three legs are restricted to the window (the
        same convention as :class:`ModalBispectrum`): triplets whose sum
        frequency ``k1 + k2`` exceeds ``kmax`` are excluded, so no
        information outside the window enters the result. The default
        ``kmin=1`` starts at the first nonzero frequency: the bispectrum is
        defined for mean-free signals, so the zero-frequency mode (the mean)
        never enters unless you pass ``kmin=0`` explicitly.
    average : bool
        Average over the ``T`` realizations. ``False`` skips the mean and
        keeps the per-realization terms, realizations last.
    return_full : bool
        Return the square matrix instead of the flat triplet arrays.

    Returns
    -------
    (k1, k2, values)
        Default form: 1-D int arrays of absolute bin indices covering the
        valid triplets ``kmin <= k2 <= k1`` with ``k1 + k2 <= kmax``, and
        float64 ``values`` of shape ``(n_triplets,)``
        (``+ (T,)`` when ``average=False``).
    (k, k, B)
        With ``return_full=True``: ``k = arange(kmin, kmax + 1)`` twice, and
        ``B`` as ``(len(k), len(k))`` float64 (``+ (T,)`` when
        ``average=False``), symmetric in its first two axes, NaN only where
        ``k1 + k2 > kmax``.

    Raises
    ------
    ValueError
        If the crop violates ``0 <= kmin <= kmax <= F - 1``, or ``X`` has no
        realizations with ``average=True``.
    """
    X = np.asarray(X)
    if X.dtype != np.complex128:
        X = X.astype(np.complex128)
    F, T = X.shape
    kmin, kmax = _resolve_window(kmin, kmax, F)
    k = np.arange(kmin, kmax + 1)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    # not valid_frequency_pairs: that helper enumerates the full triangle
    # (both orderings, no window); the estimator wants the windowed k2 <= k1 half
    mask = (K2 <= K1) & (K1 + K2 <= kmax)
    i1 = K1[mask]
    i2 = K2[mask]
    i3 = i1 + i2
    if average:
        vals = average_bispectrum_at_triplets(X, i1, i2, i3)
    else:
        # same gather batching as average_bispectrum_at_triplets: the unbatched
        # expression holds several (n_triplets, T) complex temporaries at once
        vals = np.empty((i1.size, T), dtype=np.float64)
        step = max(1, _TRIPLET_GATHER_ELEMENTS // max(T, 1))
        for s in range(0, i1.size, step):
            rows = slice(s, s + step)
            vals[rows] = np.real(X[i1[rows]] * X[i2[rows]] * np.conj(X[i3[rows]]))
    if not return_full:
        return i1, i2, vals
    shape = (k.size, k.size) if average else (k.size, k.size, T)
    B = np.full(shape, np.nan)
    B[i1 - kmin, i2 - kmin] = vals
    return k, k, full_bispectrum(B)


def full_bispectrum(
    B: ArrayLike, k2: ArrayLike | None = None, values: ArrayLike | None = None
) -> np.ndarray:
    """Fill the redundant half of a wedge by the symmetry ``B(f1, f2) = B(f2, f1)``.

    Two call forms:

    ``full_bispectrum(B)``
        ``B`` is a square matrix with the redundant half NaN (the library's
        ``return_full=True`` squares are already filled; this form serves
        hand-built half-wedge matrices). Returns a copy where each NaN cell
        is filled from its transpose partner (first two axes swapped) when
        that partner is finite. Cells whose sum frequency leaves the window
        stay NaN: both partners are NaN there and the bispectrum is
        undefined in that region.
    ``full_bispectrum(k1, k2, values)``
        The flat triplet form returned by :func:`raw_bispectrum` (its default,
        non-``return_full`` output): ``k1``, ``k2`` are the 1-D triplet index
        arrays (``k1`` passed as ``B``) and ``values`` the bispectrum values
        at them. The frequency grid is inferred as
        ``k = arange(k2.min(), (k1 + k2).max() + 1)``, ``values`` is
        scattered into the ``k2 <= k1`` half of a wedge on that grid, and the
        result is mirror-filled exactly as in the matrix form. This
        reproduces ``raw_bispectrum(..., return_full=True)`` exactly for the
        same window, because ``k2.min() == kmin`` and
        ``(k1 + k2).max() == kmax`` whenever the wedge is non-empty.

    Parameters
    ----------
    B : array-like
        Matrix form: wedge matrix, shape ``(n, n)`` or ``(n, n, T)`` (stacked
        with realizations last). Triple form: ``k1``, the first triplet
        index array (pass ``k2`` and ``values`` too, both required together).
    k2 : array-like, optional
        Triple form only: the second triplet index array, same length as
        ``k1``. Its presence (non-``None``) selects the triple form.
    values : array-like, optional
        Triple form only: bispectrum values at the triplets, shape ``(n,)``
        or ``(n, T)``.

    Returns
    -------
    np.ndarray
        Matrix form: same shape as ``B``, float64, symmetric in its first
        two axes wherever either partner is defined.
    (k, k, B)
        Triple form: ``k = arange(k2.min(), (k1 + k2).max() + 1)`` twice, and
        ``B`` as ``(len(k), len(k))`` float64 (``+ (T,)`` for stacked
        ``values``), mirror-filled as above.

    Raises
    ------
    ValueError
        Matrix form: if ``B`` is not square in its first two axes or not
        2-D/3-D. Triple form: if ``k1`` is empty (the grid cannot be
        inferred) or the triplets are not in the estimators' ``k2 <= k1``
        ordering.
    """
    if k2 is not None:
        k1 = np.asarray(B)
        k2 = np.asarray(k2)
        values = np.asarray(values, dtype=np.float64)
        if k1.size == 0:
            raise ValueError("k1 is empty: the frequency grid cannot be inferred")
        if (k2 > k1).any():
            raise ValueError(
                "triplets must satisfy k2 <= k1 (the estimators' wedge ordering); "
                "swap the offending pairs before assembling"
            )
        kmin = int(k2.min())
        kmax = int((k1 + k2).max())
        k = np.arange(kmin, kmax + 1)
        shape = (k.size, k.size) if values.ndim == 1 else (k.size, k.size, values.shape[1])
        wedge = np.full(shape, np.nan)
        wedge[k1 - kmin, k2 - kmin] = values
        return k, k, full_bispectrum(wedge)

    B = np.array(B, dtype=np.float64)
    if B.ndim not in (2, 3) or B.shape[0] != B.shape[1]:
        raise ValueError(f"need a square (n, n) or stacked (n, n, T) array, got shape {B.shape}")
    out = B
    mirror = np.swapaxes(out, 0, 1)
    fill = np.isnan(out) & ~np.isnan(mirror)
    out[fill] = mirror[fill]
    return out


# --------------------------------------------------------------------------- #
# Modal bispectrum ingredients: mode family, pair index set, kernels.
# The core never builds a mode matrix itself: callers assemble one (row p =
# mode q_p sampled on the F frequencies) and hand it to ModalBispectrum.
# --------------------------------------------------------------------------- #
def rescale_to_symmetric(values: ArrayLike, lo: float, hi: float) -> np.ndarray:
    """Rescale ``values`` from ``[lo, hi]`` onto the symmetric interval ``[-1, 1]`` affinely.

    Use before :func:`legendre_modes`, which expects its argument in ``[-1, 1]``
    (the Legendre domain, not the unit interval ``[0, 1]``).

    Parameters
    ----------
    values : array-like
        Coordinates to rescale.
    lo, hi : float
        Endpoints mapped to ``-1`` and ``1``.

    Returns
    -------
    np.ndarray, float64

    Raises
    ------
    ValueError
        If ``lo >= hi`` (a degenerate interval would silently produce inf/NaN).
    """
    if not lo < hi:
        raise ValueError(f"need lo < hi, got lo={lo}, hi={hi}")
    return 2.0 * (np.asarray(values, dtype=np.float64) - lo) / (hi - lo) - 1.0


def legendre_modes(z: ArrayLike, degree: int) -> np.ndarray:
    """Legendre polynomials ``P_0 .. P_degree`` evaluated at ``z``.

    The generic 1-D mode family. Evaluated on an axis mapped to ``[-1, 1]``,
    the result is a ready ``modes`` matrix for :class:`ModalBispectrum`:
    ``legendre_modes(np.linspace(-1, 1, F), degree)`` gives modes linear in
    frequency (use :func:`rescale_to_symmetric` to map any other axis).
    The mel basis instead evaluates it on the mel band
    index and smears the result onto the frequencies through the mel
    filterbank (:func:`bispectrosa.filters.mel_legendre_modes`).

    Parameters
    ----------
    z : array-like
        Evaluation points, already in ``[-1, 1]`` (see :func:`rescale_to_symmetric`).
    degree : int
        Highest polynomial order.

    Returns
    -------
    np.ndarray, shape (degree + 1, \\*z.shape)
    """
    # deferred: scipy.special dominates the package's import time, and only
    # basis construction needs it
    from scipy.special import legendre

    z = np.asarray(z, dtype=np.float64)
    return np.stack([legendre(p)(z) for p in range(degree + 1)])


def modal_index_pairs(degree: int) -> list[tuple[int, int]]:
    """Enumerate the mode index pairs ``p <= r`` with ``p + r <= degree``, degree-ordered.

    Ordering key ``(p + r, p, r)`` nests the list: the first
    :func:`modal_pair_dim` ``(d)`` entries are exactly the degree-``d`` list.

    Parameters
    ----------
    degree : int
        Maximum total degree ``D``; pairs satisfy ``p + r <= D``.

    Returns
    -------
    list of (int, int)
    """
    pairs = [
        (p, r) for (p, r) in combinations_with_replacement(range(degree + 1), 2) if p + r <= degree
    ]
    return sorted(pairs, key=lambda pr: (pr[0] + pr[1], pr[0], pr[1]))


def modal_pair_dim(degree: int) -> int:
    """Number of pairs in :func:`modal_index_pairs`, ``floor((degree + 2)**2 / 4)``.

    Parameters
    ----------
    degree : int
        Maximum total degree ``D`` of the pair set.

    Returns
    -------
    int
        20 at D=7, 36 at D=10, 49 at D=12.
    """
    return ((degree + 2) ** 2) // 4


def modal_pair_kernel(
    p: int,
    r: int,
    modes_f1: np.ndarray,
    modes_f2: np.ndarray,
    modes_f3: np.ndarray,
    third_f1=1.0,
    third_f2=1.0,
    third_f3=1.0,
) -> np.ndarray:
    """Evaluate the pair kernel ``Q_pr = Sym[q_p(f1) q_r(f2) 1(f3)]`` at triplets.

    The mean over the six assignments of ``{q_p, q_r, 1}`` to the three legs
    ``(f1, f2, f3 = f1 + f2)`` of a valid triplet: the two modes of the pair
    land on two legs, the third leg is unweighted (constant ``1``).

    Parameters
    ----------
    p, r : int
        Mode orders.
    modes_f1, modes_f2, modes_f3 : np.ndarray
        The mode matrix gathered at each leg of the triplet list (e.g.
        ``modes[:, i1]``), indexed ``[order, triplet]``.
    third_f1, third_f2, third_f3 : scalar or np.ndarray, optional
        The third-leg weight gathered at each leg, default the constant
        ``1``. Pass :attr:`ModalBispectrum.third` gathered at the triplets
        to reproduce a frequency-windowed estimator exactly.

    Returns
    -------
    np.ndarray, one value per triplet
    """
    return (
        (modes_f1[p] * modes_f2[r] + modes_f1[r] * modes_f2[p]) * third_f3
        + (modes_f1[p] * modes_f3[r] + modes_f1[r] * modes_f3[p]) * third_f2
        + (modes_f2[p] * modes_f3[r] + modes_f2[r] * modes_f3[p]) * third_f1
    ) / 6.0


# --------------------------------------------------------------------------- #
# Modal estimator
# --------------------------------------------------------------------------- #
def _split_across_threads(fn, n_items: int, n_threads: int) -> None:
    """Split ``[0, n_items)`` into even contiguous ranges, run ``fn(lo, hi)`` on each in a thread.

    ``fn`` writes into shared preallocated output (nothing is returned).
    With one thread the pool is skipped entirely. ``list()`` drains the lazy
    map, which waits for completion and re-raises the first worker exception
    (already-running siblings finish first; outputs are discarded either way).
    """
    if n_threads == 1:
        fn(0, n_items)
        return
    bounds = np.linspace(0, n_items, n_threads + 1).astype(int)
    with ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(fn, bounds[:-1], bounds[1:]))


def _thread_count(workers: int | None, *, n_items: int, auto_limit: int) -> int:
    """The shared thread-count policy of :class:`ModalBispectrum`'s loops.

    ``workers=None`` auto-tunes: up to 8 depending on CPU count, further
    capped by ``auto_limit`` (the loop's own gate, e.g. one thread per so
    many columns). An explicit ``workers`` is honored as-is, bounded only by
    ``n_items``. Always at least 1.
    """
    if workers is not None:
        return max(1, min(workers, n_items))
    return max(1, min(8, os.cpu_count() or 1, auto_limit))


# Tuning knobs for ModalBispectrum. They set speed and peak memory only, never
# values: every threaded or blocked loop below is bit-identical for any
# setting. The numbers are conservative round figures measured on one 8-core
# laptop; they transfer because they encode orders of magnitude (a thread
# dispatch costs tens of microseconds, consumer memory buses saturate at a
# few streaming threads), not machine specifics. To tune for unusual
# hardware, pass ``workers=`` to the public calls rather than editing these.

#: :meth:`ModalBispectrum.apply_modes` (IRFFT stage): at most one thread per this many
#: columns. Dispatching a thread costs ~tens of microseconds, about the IRFFT
#: time of 64 columns; smaller shares spend more on bookkeeping than on work.
_MIN_COLS_PER_THREAD = 64

#: :meth:`ModalBispectrum.estimate_beta` (reduction stage): the triple product
#: streams the large ``z`` arrays with almost no arithmetic per byte, so its
#: limit is memory bandwidth, not CPU. A second thread only pays off past
#: ~300 columns of work (measured); the gate sits conservatively below that.
_REDUCE_COLS_PER_THREAD = 200

#: The memory bus saturates around 3-4 streaming threads on consumer
#: hardware; further reduction threads just contend for it. (The IRFFT stage
#: is compute-bound and carries no such cap.)
_MAX_REDUCE_THREADS = 4

#: :meth:`ModalBispectrum.estimate_beta` processes columns in blocks of this size,
#: capping the peak (degree + 1, n_irfft, block) apply_modes output at ~170 MB
#: (degree 12, n_irfft 400) instead of letting it grow with the input length.
_ESTIMATE_COLS_PER_BLOCK = 8192


class ModalBispectrum:
    """The modal bispectrum estimator: a mode matrix, its pair index set, and
    the separable estimation that runs on them.

    A mode is a 1-D weight profile ``q_p[k]`` over the ``n_freq``
    frequencies (one row of ``modes``). The class does not build modes; the
    caller supplies the matrix, and where the modes sit on the frequency
    axis is that builder's choice. :func:`legendre_modes` is the generic
    family; the mel builders are
    :func:`bispectrosa.filters.mel_legendre_modes` (the modes) and
    :func:`bispectrosa.filters.mel_legendre_modal_bispectrum` (a ready ``ModalBispectrum``).

    Each pair kernel puts the pair's two modes on two legs of the triplet
    and leaves the third leg unweighted: the third factor is the constant
    ``1``, restricted to the analysis window (:attr:`third`, the ``z``
    factor shared by every product in :meth:`estimate_beta`). No row of
    ``modes`` plays a special role.

    Parameters
    ----------
    modes : np.ndarray, shape (degree + 1, n_freq)
        Mode-weight matrix ``q_p[k]``: mode order ``p`` sampled on the
        ``n_freq`` positive frequencies. Stored as ``dtype``.
    pairs : sequence of (int, int)
        The ``(p, r)`` mode index set (typically :func:`modal_index_pairs`).
    n_irfft : int, optional
        IRFFT length, the number of output samples ``u`` of :meth:`apply_modes`.
        ``None`` (default) derives the even-length case
        ``2 * (modes.shape[1] - 1)``; pass it explicitly only for an
        odd-length transform (``2 * modes.shape[1] - 1``). Any other value
        raises: ``n_irfft // 2 + 1`` must equal ``modes.shape[1]``, else
        ``np.fft.irfft`` would silently crop or zero-pad the spectrum.
    kmin, kmax : int, optional
        Frequency window of the analysis, inclusive: mode columns outside
        ``[kmin, kmax]`` are zeroed, and the constant third leg
        (:attr:`third`) is ``1`` inside the window and ``0`` outside, so
        every leg of every triplet is restricted to the window (a
        band-limited bispectrum, the same convention as
        :func:`raw_bispectrum`; several windows can share one ``modes``
        matrix). The default ``kmin=1`` excludes the zero frequency: the
        bispectrum is defined for mean-free signals, so the mean never
        enters unless you pass ``kmin=0`` explicitly. When validating a
        windowed estimator against :func:`project_bispectrum` /
        :func:`modal_gram_matrix` / :func:`reconstruct_bispectrum`, pass
        ``self.modes`` and ``third=self.third`` so all sides use the same
        window.
    dtype : np.dtype
        Working precision of :meth:`apply_modes` / :meth:`estimate_beta`
        (default float32; pass ``np.float64`` to study the estimator's
        numerics).

    Attributes
    ----------
    third : np.ndarray, shape (n_freq,)
        The third-leg weight: ``1`` inside the analysis window, ``0``
        outside.

    Raises
    ------
    ValueError
        If ``modes`` is not 2-D, ``n_irfft`` disagrees with the number of
        mode frequencies, ``pairs`` indexes mode orders outside ``modes``'
        rows, or the window violates ``0 <= kmin <= kmax <= n_freq - 1``.
    """

    def __init__(
        self,
        modes: ArrayLike,
        pairs: Sequence[tuple[int, int]],
        n_irfft: int | None = None,
        dtype: np.dtype = np.float32,
        *,
        kmin: int = 1,
        kmax: int | None = None,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.modes = np.asarray(modes, dtype=self.dtype)
        if self.modes.ndim != 2:
            raise ValueError(f"modes must be 2-D (order, frequency), got shape {self.modes.shape}")
        self.pairs = [(int(p), int(r)) for p, r in pairs]
        n_freq = self.modes.shape[1]
        self.kmin, self.kmax = _resolve_window(kmin, kmax, n_freq)
        if self.kmin > 0 or self.kmax < n_freq - 1:
            self.modes = self.modes.copy()
            self.modes[:, : self.kmin] = 0
            self.modes[:, self.kmax + 1 :] = 0
        self.third = np.zeros(n_freq, dtype=self.dtype)
        self.third[self.kmin : self.kmax + 1] = 1
        self.n_irfft = int(n_irfft) if n_irfft is not None else 2 * (n_freq - 1)
        if self.n_irfft // 2 + 1 != self.modes.shape[1]:
            raise ValueError(
                f"n_irfft={self.n_irfft} implies {self.n_irfft // 2 + 1} "
                f"frequencies but modes has {self.modes.shape[1]}; np.fft.irfft would "
                "silently crop or zero-pad the spectrum"
            )
        n_orders = self.modes.shape[0]
        if any(p < 0 or r < 0 or p >= n_orders or r >= n_orders for p, r in self.pairs):
            raise ValueError(
                f"pairs index mode orders outside modes' {n_orders} rows: "
                f"{[pr for pr in self.pairs if max(pr) >= n_orders or min(pr) < 0][:5]}"
            )
        # modes plus the constant third leg as the last row, ready for the one
        # batched IRFFT of _estimate_beta_block
        self._modes_and_third = np.vstack([self.modes, self.third])

    @property
    def dim(self) -> int:
        """Number of pair coefficients (= ``len(self.pairs)``)."""
        return len(self.pairs)

    @property
    def degree(self) -> int:
        """Highest mode order (= ``modes.shape[0] - 1``)."""
        return self.modes.shape[0] - 1

    def apply_modes(self, X: np.ndarray, workers: int | None = None) -> np.ndarray:
        """Mode-filtered copies of the input, one per mode order.

        Weights the Fourier coefficients by each mode and inverse-transforms
        to ``n_irfft`` samples: ``z[p, u, t] = IRFFT_k(modes[p, k] X[k, t])``.

        Parameters
        ----------
        X : np.ndarray, shape (n_freq, T), complex
            Fourier coefficients on the frequencies the basis was built for.
        workers : int, optional
            Threads for the IRFFT, splitting the realizations (columns)
            between them; they are independent, so the result is
            bit-identical for any value. ``None`` (default) auto-tunes: up
            to 8 depending on CPU count, and no more than one thread per
            ``_MIN_COLS_PER_THREAD`` columns. An explicit integer is honored
            as-is (bounded by the column count); pass ``1`` to force
            single-threaded, e.g. when parallelizing over inputs.

        Returns
        -------
        np.ndarray, shape (degree + 1, n_irfft, T), of the basis ``dtype``
        """
        return self._irfft_filter(self.modes, X, workers)

    def _irfft_filter(self, weights: np.ndarray, X: np.ndarray, workers: int | None) -> np.ndarray:
        """IRFFT of ``weights[i, k] * X[k, t]`` rows, threaded over realizations."""
        if X.shape[0] != self.modes.shape[1]:
            raise ValueError(
                f"X has {X.shape[0]} frequencies but the basis was built for {self.modes.shape[1]}"
            )
        T = X.shape[1]
        # the sample axis must stay contiguous, matching np.fft.irfft's output
        # layout: numpy picks its summation blocking from it, so any other
        # layout shifts float32 reductions downstream
        z = np.empty((weights.shape[0], T, self.n_irfft), dtype=self.dtype)
        z = z.transpose(0, 2, 1)
        n_threads = _thread_count(workers, n_items=T, auto_limit=T // _MIN_COLS_PER_THREAD)

        def fill(lo: int, hi: int) -> None:
            weighted = weights[:, :, None] * X[None, :, lo:hi]
            z[:, :, lo:hi] = np.fft.irfft(weighted, n=self.n_irfft, axis=1)

        _split_across_threads(fill, T, n_threads)
        return z

    def estimate_beta(self, X: np.ndarray, workers: int | None = None) -> np.ndarray:
        """Estimate the per-realization pair coefficients ``beta``.

        The fast separable form: ``beta_pr[t] = sum_u z_p z_r z_c``, with
        ``z_p`` the :meth:`apply_modes` outputs and ``z_c`` the third-leg
        copy, the IRFFT of the window-limited coefficients themselves
        (:attr:`third` times ``X``). The sample sum keeps the triplets with
        ``k1 + k2 = k3 (mod n_irfft)``, so the ``O(F**2)`` bispectrum grid is
        never formed; see :func:`project_bispectrum` for how this relates to
        the exact projection (a fixed constant, plus aliased and
        zero-frequency corrections). The sum accumulates in float64 whatever the basis
        ``dtype``, so the result does not depend on the memory layout of the
        mode-filtered output.

        Parameters
        ----------
        X : np.ndarray, shape (n_freq, T), complex
        workers : int, optional
            Threads for the estimator's two threaded loops: the IRFFT splits
            the realizations between threads (see :meth:`apply_modes`) and the
            bandwidth-bound sample-sum reduction splits the pair coefficients.
            Bit-identical for any value. ``None`` (default) auto-tunes each,
            capping the reduction at a few threads where it saturates memory
            bandwidth; pass an integer to set the count (to tune for the
            machine, or ``1`` to force single-threaded, e.g. when parallelizing
            over inputs).

        Returns
        -------
        np.ndarray, shape (T, dim), of the basis ``dtype``
            Raw coefficients ``beta_pr[t]``, one row per realization (rows
            are written blockwise, hence realizations first; callers wanting
            coefficients-first transpose). Any compression or pooling over
            realizations is the caller's step.

        Notes
        -----
        Realizations are processed in fixed-size blocks, so peak memory does
        not grow with the input length (the full :meth:`apply_modes` output for a
        long input would be gigabytes). Blocking is bit-identical to a single
        pass: every ``beta`` row depends only on its own column of ``X``.
        """
        T = X.shape[1]
        beta = np.empty((T, len(self.pairs)), dtype=self.dtype)
        for lo in range(0, T, _ESTIMATE_COLS_PER_BLOCK):
            hi = min(T, lo + _ESTIMATE_COLS_PER_BLOCK)
            self._estimate_beta_block(X[:, lo:hi], beta[lo:hi], workers)
        return beta

    def _estimate_beta_block(self, X: np.ndarray, beta: np.ndarray, workers: int | None) -> None:
        """One :meth:`estimate_beta` block: fill ``beta`` (a view) from ``X``."""
        # one batched IRFFT of the modes plus the constant third leg (last row)
        z = self._irfft_filter(self._modes_and_third, X, workers)
        T = X.shape[1]
        n_pairs = len(self.pairs)

        def reduce(lo: int, hi: int) -> None:
            tmp = np.empty_like(z[0])
            for i in range(lo, hi):
                p, r = self.pairs[i]
                np.multiply(z[p], z[r], out=tmp)
                tmp *= z[-1]
                # float64 accumulation: numpy picks pairwise vs sequential
                # summation from the axis contiguity of z, which shifts float32
                # sums by up to ~1e-3 relative; accumulating in double pins beta
                # to the math (and keeps it bit-identical however the pairs split)
                beta[:, i] = tmp.sum(axis=0, dtype=np.float64)

        # bandwidth-bound loop over the pairs, split between threads: each thread
        # fills its own columns of beta, so the output is bit-identical for any
        # thread count. workers=None auto-tunes conservatively (the reduction
        # saturates memory bandwidth well before the IRFFT does, so gate on the
        # realization count and cap the thread count); an explicit workers is
        # honored as-is so it can be tuned per machine, bounded only by the
        # number of pairs.
        n_threads = _thread_count(
            workers,
            n_items=n_pairs,
            auto_limit=min(_MAX_REDUCE_THREADS, n_pairs, T // _REDUCE_COLS_PER_THREAD),
        )
        _split_across_threads(reduce, n_pairs, n_threads)


def project_bispectrum(
    X: ArrayLike,
    pairs: Sequence[tuple[int, int]],
    modes: np.ndarray,
    third: np.ndarray | None = None,
) -> np.ndarray:
    """Project the full bispectrum of ``X`` onto the modal pair kernels.

    ``beta_n = <B, Q_n>`` with ``B[i1, i2] = mean_t Re(X[i1] X[i2] X*[i1 + i2])``,
    summed over the valid triplets (:func:`valid_frequency_pairs`) with the kernel of
    :func:`modal_pair_kernel`. Validation-side only: together with
    :func:`modal_gram_matrix` it feeds :func:`reconstruct_bispectrum`. The
    fast path is :meth:`ModalBispectrum.estimate_beta`: its
    realization-averaged output equals ``6 / n_irfft**2`` times this
    projection, up to two corrections its sample sum introduces. Triplets
    aliased around Nyquist (``k1 + k2 + k3 = n_irfft``) enter alongside the
    valid ones, negligible for smooth modes and measured at ~1e-5 relative
    for band modes; and triplets with a zero-frequency leg are
    under-weighted (the zero frequency has no Hermitian mirror), which
    vanishes for mean-free realizations or a window excluding the zero
    frequency (the estimator's ``kmin=1`` default).

    Parameters
    ----------
    X : np.ndarray, shape (n_freq, T), complex
        Fourier coefficients.
    pairs : sequence of (int, int)
        Mode index set.
    modes : np.ndarray, shape (degree + 1, n_freq)
        Mode matrix on the frequencies; pass the same one to :func:`modal_gram_matrix` /
        :func:`reconstruct_bispectrum` so all three use the same kernels.
    third : np.ndarray, shape (n_freq,), optional
        Third-leg weight; default the constant ``1`` everywhere. To match a
        frequency-windowed estimator exactly, pass its
        :attr:`ModalBispectrum.third`.

    Returns
    -------
    np.ndarray, shape (len(pairs),), float64
        Computed in double precision regardless of the input dtypes
        (validation precision).
    """
    X = np.asarray(X, dtype=np.complex128)
    n_freq = X.shape[0]
    i1, i2 = valid_frequency_pairs(n_freq)
    i3 = i1 + i2
    B_avg = average_bispectrum_at_triplets(X, i1, i2, i3)
    modes_f1, modes_f2, modes_f3, t1, t2, t3 = _gather_kernel_legs(modes, third, i1, i2, i3)
    # measured: the per-pair elementwise passes beat a single einsum contraction
    # here (the (orders, orders, points) intermediate costs more than it saves)
    return np.array(
        [
            (modal_pair_kernel(p, r, modes_f1, modes_f2, modes_f3, t1, t2, t3) * B_avg).sum()
            for (p, r) in pairs
        ]
    )


def _gather_kernel_legs(modes: np.ndarray, third: np.ndarray | None, i1, i2, i3):
    """Kernel factors gathered at each triplet, in validation precision (float64).

    The one leg-gathering convention of the validation trio
    (:func:`project_bispectrum` / :func:`modal_gram_matrix` /
    :func:`reconstruct_bispectrum`): the mode columns on the three legs, and
    the third-leg weights (scalar 1s when ``third`` is unset).
    """
    modes = np.asarray(modes, dtype=np.float64)
    if third is None:
        t1 = t2 = t3 = 1.0
    else:
        third = np.asarray(third, dtype=np.float64)
        t1, t2, t3 = third[i1], third[i2], third[i3]
    return modes[:, i1], modes[:, i2], modes[:, i3], t1, t2, t3


def modal_gram_matrix(
    pairs: Sequence[tuple[int, int]], modes: np.ndarray, third: np.ndarray | None = None
) -> np.ndarray:
    """Gram matrix ``Gamma_mn = <Q_m, Q_n>`` of the modal pair kernels.

    The inner product runs over the valid triplets with the same
    kernel :func:`modal_pair_kernel` as :func:`project_bispectrum`, so the expansion
    weights ``alpha = Gamma^-1 beta`` give the least-squares fit
    ``B ~ sum_n alpha_n Q_n``.

    Parameters
    ----------
    pairs : sequence of (int, int)
        Mode index set.
    modes : np.ndarray, shape (degree + 1, n_freq)
        Mode matrix on the frequencies.
    third : np.ndarray, shape (n_freq,), optional
        Third-leg weight; default the constant ``1``. Use the same value as
        in :func:`project_bispectrum`.

    Returns
    -------
    np.ndarray, shape (len(pairs), len(pairs))

    Notes
    -----
    Computed in float64 regardless of the modes' dtype (validation precision;
    the Gram is ill-conditioned). Increasingly ill-conditioned as ``degree``
    rises: invert only at modest degree, preferably via ``solve`` / ``lstsq``.
    """
    i1, i2 = valid_frequency_pairs(np.shape(modes)[1])
    i3 = i1 + i2
    modes_f1, modes_f2, modes_f3, t1, t2, t3 = _gather_kernel_legs(modes, third, i1, i2, i3)
    Q_mat = np.zeros((len(pairs), i1.size))
    for n, (p, r) in enumerate(pairs):
        Q_mat[n] = modal_pair_kernel(p, r, modes_f1, modes_f2, modes_f3, t1, t2, t3)
    return Q_mat @ Q_mat.T


def _eval_pair_kernels(
    alpha: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    modes: np.ndarray,
    third: np.ndarray | None,
    idx1: np.ndarray,
    idx2: np.ndarray,
    idx3: np.ndarray,
) -> np.ndarray:
    """Weighted sum of the pair kernels at the given frequency-index triplets."""
    modes_f1, modes_f2, modes_f3, t1, t2, t3 = _gather_kernel_legs(modes, third, idx1, idx2, idx3)
    vals = np.zeros(idx1.size)
    for n, (p, r) in enumerate(pairs):
        vals += alpha[n] * modal_pair_kernel(p, r, modes_f1, modes_f2, modes_f3, t1, t2, t3)
    return vals


def reconstruct_bispectrum(
    beta: ArrayLike,
    pairs: Sequence[tuple[int, int]],
    modes: np.ndarray,
    gram: np.ndarray | None = None,
    n_grid: int = 64,
    third: np.ndarray | None = None,
    return_full: bool = False,
    kmin: int | None = None,
    kmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ``B(f1, f2)`` from pair coefficients.

    Two grid conventions. The default is a uniform ``n_grid``-point grid over
    the modes' full frequency range, in normalized coordinates. Passing
    ``kmin`` switches to the estimator's own bins: the reconstruction is
    evaluated exactly on the ``[kmin, kmax]`` window with the same wedge rule
    as :func:`raw_bispectrum`, so the output is cell-for-cell aligned with
    ``raw_bispectrum(X, kmin=kmin, kmax=kmax)`` (for comparisons,
    :func:`snr_bispectrum`, and shared frequency axes).

    Parameters
    ----------
    beta : array-like, shape (len(pairs),)
        Pair coefficients (from :func:`project_bispectrum`).
    pairs : sequence of (int, int)
        Mode index set.
    modes : np.ndarray, shape (degree + 1, n_freq)
        Mode matrix on the frequencies.
    gram : np.ndarray, optional
        Gram matrix from :func:`modal_gram_matrix`. The expansion weights solve
        ``gram @ alpha = beta`` as a minimum-norm least-squares problem
        (solved, never inverted: a Gram can be exactly singular, e.g. the
        raw-Legendre kernels carry one linear dependence through
        ``f3 = f1 + f2``). When omitted, ``alpha = beta``.
    n_grid : int
        Grid resolution per axis (normalized form only; ignored when ``kmin``
        is given).
    third : np.ndarray, shape (n_freq,), optional
        Third-leg weight; default the constant ``1``. Use the same value as
        in :func:`project_bispectrum` / :func:`modal_gram_matrix`.
    kmin, kmax : int, optional
        Absolute frequency-bin window (inclusive), same convention as
        :func:`raw_bispectrum`: ``kmax=None`` means the last mode bin.
        Passing ``kmin`` selects the bin form.

    Returns
    -------
    (x1, x2, values)
        Default form: the normalized ``[-1, 1]`` coordinates of each valid
        grid point (the ``f2 <= f1`` half whose sum frequency is in range)
        and the reconstructed value at each, as flat float64 arrays. In the
        bin form the coordinates are the absolute bin indices ``(k1, k2)``
        instead, exactly as :func:`raw_bispectrum` returns them.
    (x, x, B)
        With ``return_full=True``: the axis twice (``linspace(-1, 1,
        n_grid)``, or the bin grid ``arange(kmin, kmax + 1)``) and ``B`` as a
        square float64 matrix, symmetric in its axes (via
        :func:`full_bispectrum`), NaN only outside the sum-frequency range.
        The caller maps the axes to Hz for display.

    Raises
    ------
    ValueError
        If the bin window violates ``0 <= kmin <= kmax <= n_freq - 1``.
    """
    beta = np.asarray(beta, dtype=np.float64)
    if gram is None:
        alpha = beta
    else:
        alpha, *_ = np.linalg.lstsq(np.asarray(gram, dtype=np.float64), beta, rcond=None)
    n_freq = np.shape(modes)[1]
    if kmin is not None:
        kmin, kmax = _resolve_window(kmin, kmax, n_freq)
        k = np.arange(kmin, kmax + 1)
        K1, K2 = np.meshgrid(k, k, indexing="ij")
        mask = (K2 <= K1) & (K1 + K2 <= kmax)
        idx1 = K1[mask]
        idx2 = K2[mask]
        vals = _eval_pair_kernels(alpha, pairs, modes, third, idx1, idx2, idx1 + idx2)
        if not return_full:
            return idx1, idx2, vals
        B = np.full(K1.shape, np.nan)
        B[idx1 - kmin, idx2 - kmin] = vals
        return k, k, full_bispectrum(B)
    x = np.linspace(-1.0, 1.0, n_grid)
    # grid-point frequency indices, and the valid region in grid-index form:
    # exact integer arithmetic, so the boundary never falls to float error
    gidx = np.round(np.linspace(0.0, n_freq - 1, n_grid)).astype(int)
    J, K = np.meshgrid(np.arange(n_grid), np.arange(n_grid), indexing="ij")
    mask = (K <= J) & (J + K <= n_grid - 1)
    # evaluate only the valid region (each point depends only on its own legs)
    idx1 = gidx[J[mask]]
    idx2 = gidx[K[mask]]
    # rounding can push idx1 + idx2 one past the last frequency on the boundary
    idx3 = np.minimum(idx1 + idx2, n_freq - 1)
    vals = _eval_pair_kernels(alpha, pairs, modes, third, idx1, idx2, idx3)
    if not return_full:
        return x[J[mask]], x[K[mask]], vals
    B = np.full(J.shape, np.nan)
    B[mask] = vals
    return x, x, full_bispectrum(B)


def snr_bispectrum(
    B: ArrayLike, P: ArrayLike, *, kmin: int = 1, floor: float | None = None
) -> np.ndarray:
    """Signal-to-noise form of a square bispectrum: ``B / sqrt(P1 P2 P3)``.

    The estimate's variance scales with the power-spectrum product of the
    three legs, so dividing it out (inverse-variance weighting) removes raw
    loudness and leaves the coupling itself. This is the form to plot for
    real sounds and the right weighting for comparing bispectrum shapes.

    Parameters
    ----------
    B : array-like, shape (n, n)
        Square bispectrum whose grid starts at absolute bin ``kmin`` (as
        returned by :func:`raw_bispectrum` with ``average=True,
        return_full=True``); NaN cells stay NaN. Every finite cell's sum leg
        ``k1 + k2`` must have a power bin: the estimators' own squares
        satisfy this (their out-of-window cells are NaN), and anything else
        is rejected rather than silently misweighted.
    P : array-like, shape (F,)
        Power per absolute frequency bin (e.g. the frame-averaged STFT
        power).
    kmin : int
        Absolute bin index of the grid's first row/column.
    floor : float, optional
        Clip ``P`` below ``floor * P.max()`` before dividing. Real spectra
        span many decades; without a floor the low-power corner dominates.

    Returns
    -------
    np.ndarray
        Same shape as ``B``, float64.

    Raises
    ------
    ValueError
        If ``B`` is not square, the grid does not fit inside ``P``, or a
        finite cell's sum leg falls outside ``P``.
    """
    B = np.asarray(B, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"need a square (n, n) bispectrum, got shape {B.shape}")
    if kmin < 0 or kmin + B.shape[0] > P.size:
        raise ValueError(
            f"grid [kmin, kmin + n) = [{kmin}, {kmin + B.shape[0]}) does not fit in P of size {P.size}"
        )
    if floor is not None:
        P = np.maximum(P, floor * P.max())
    k = np.arange(kmin, kmin + B.shape[0])
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    out_of_range = K1 + K2 > P.size - 1
    if (out_of_range & np.isfinite(B)).any():
        raise ValueError(
            f"finite cells with sum leg k1 + k2 > {P.size - 1}: P is too short for this grid"
        )
    # the clamp only ever touches NaN cells now; it keeps the gather in range
    K3 = np.where(out_of_range, P.size - 1, K1 + K2)
    return B / np.sqrt(P[K1] * P[K2] * P[K3])


def modal_shape_correlation(betas: ArrayLike, gram: np.ndarray) -> np.ndarray:
    """Shape correlation of modal coefficient vectors under the basis Gram metric.

    The pair kernels overlap, so comparing raw ``beta`` vectors directly
    overweights their shared smooth components. The right inner product is
    ``beta_a^T Gamma^+ beta_b`` (with ``Gamma^+`` the Gram pseudo-inverse),
    which equals the inner product of the reconstructed bispectra over the
    valid domain without building them.

    Parameters
    ----------
    betas : array-like, shape (n_vectors, n_pairs)
        Modal coefficient vectors, one per row (from
        :func:`project_bispectrum` or :meth:`ModalBispectrum.estimate_beta`).
    gram : np.ndarray, shape (n_pairs, n_pairs)
        Gram matrix of the pair kernels (:func:`modal_gram_matrix`). Applied
        through a pseudo-inverse, so an exactly singular Gram is fine.

    Returns
    -------
    np.ndarray, shape (n_vectors, n_vectors)
        Symmetric matrix of cosine similarities under the Gram metric, with
        unit diagonal. A vector with zero norm under the metric (all-zero,
        or entirely inside the Gram's null space) has no defined shape: its
        row and column are NaN, and its diagonal entry too.

    Raises
    ------
    ValueError
        If ``betas`` is not 2-D.
    """
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim != 2:
        raise ValueError(f"need betas of shape (n_vectors, n_pairs), got {betas.shape}")
    S = betas @ np.linalg.pinv(np.asarray(gram, dtype=np.float64)) @ betas.T
    # the pinv metric is PSD only up to rounding: clamp before the sqrt, and
    # give zero-norm vectors an explicitly NaN row/column instead of 0/0 noise
    d = np.sqrt(np.maximum(np.diag(S), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = S / np.outer(d, d)
    C[d == 0.0, :] = np.nan
    C[:, d == 0.0] = np.nan
    return C
