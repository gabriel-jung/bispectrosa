"""Physics + consistency sanity checks.

These assert only properties that are robustly true for this estimator:
- the bispectrum detects a genuine frequency triplet (three-way coupling);
- the degree-ordered pair feature is *nested* (a low-degree feature is a prefix
  of a higher-degree one);
- structural invariants of the basis, projection, and reconstruction.

Phase-*direction* (coupled vs uncoupled) lives in the complex argument with
convention-dependent offsets and is demonstrated in the example notebook, not
asserted here.
"""

import numpy as np
import pytest

import bispectrosa as bs
from bispectrosa import bispectrum

SR = 16000
N_FFT = 400


def _tones(freqs, phases, n=SR, sr=SR, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    y = noise * rng.standard_normal(n)
    for f, ph in zip(freqs, phases, strict=True):
        y = y + np.sin(2 * np.pi * f * t + ph)
    return y.astype(np.float32)


def _bispectrum_at_bin(sig, b1, b2):
    """Complex bispectrum ``mean_t X[b1] X[b2] X*[b1+b2]`` at exact STFT bins."""
    X = bs.stft(sig, n_fft=N_FFT)
    return np.mean(X[b1] * X[b2] * np.conj(X[b1 + b2]))


def test_triplet_detection():
    # tones placed exactly on STFT bins (multiples of sr/n_fft = 40 Hz)
    b1, b2 = 30, 13  # 1200 Hz, 520 Hz -> sum 1720 Hz (bin 43)
    f1, f2 = b1 * 40, b2 * 40
    with_triplet = _tones([f1, f2, f1 + f2], [1.1, 0.4, 2.7], noise=0.1, seed=1)
    no_sum = _tones([f1, f2], [1.1, 0.4], noise=0.1, seed=1)
    noise = _tones([], [], noise=1.0, seed=7)

    b_triplet = abs(_bispectrum_at_bin(with_triplet, b1, b2))
    b_nosum = abs(_bispectrum_at_bin(no_sum, b1, b2))
    b_noise = abs(_bispectrum_at_bin(noise, b1, b2))
    # a real triplet lights the bin ~2-3 orders of magnitude over "no coupling"
    assert b_triplet > 20 * b_nosum
    assert b_triplet > 20 * b_noise


def test_degree_ordered_nesting():
    # mel_bispectrogram at degree d equals the first modal_pair_dim(d) rows at degree 12
    y = _tones([1200.0, 520.0, 1720.0], [1.1, 0.4, 1.5], noise=0.2, seed=2)
    full = bs.time_pool(bs.mel_bispectrogram(y, degree=12))
    for d in (7, 10):
        sub = bs.time_pool(bs.mel_bispectrogram(y, degree=d))
        np.testing.assert_allclose(full[: bs.modal_pair_dim(d)], sub, rtol=1e-4, atol=1e-4)


def test_deterministic():
    y = _tones([1200.0, 520.0], [0.0, 0.0], noise=0.3, seed=3)
    a = bs.mel_bispectrogram(y)
    b = bs.mel_bispectrogram(y)
    np.testing.assert_array_equal(a, b)


def test_precomputed_stft_matches_waveform():
    # one STFT feeds both features; the S= path must reproduce the y= path
    # bit-for-bit (default stft params match the features' defaults)
    y = _tones([1200.0, 520.0], [1.1, 0.4], noise=0.3, seed=7)
    X = bs.stft(y)
    np.testing.assert_array_equal(bs.mel_bispectrogram(y), bs.mel_bispectrogram(S=X))
    np.testing.assert_array_equal(bs.mel_spectrogram(y), bs.mel_spectrogram(S=X))


def test_y_or_S_contract():
    y = _tones([1200.0], [0.0], noise=0.1, seed=8)
    X = bs.stft(y)
    for call in (
        lambda: bs.mel_bispectrogram(y, S=X),  # both
        lambda: bs.mel_bispectrogram(),  # neither
        lambda: bs.mel_spectrogram(y, S=X),
        lambda: bs.mel_spectrogram(),
    ):
        with pytest.raises(ValueError):
            call()
    # an S built on the wrong n_fft is caught by the bin-count check
    with pytest.raises(ValueError):
        bs.mel_bispectrogram(S=bs.stft(y, n_fft=512))


def test_fmin_fmax_band_limits():
    # fmin/fmax reach the mel bank and change the feature; the default
    # (0, None) leaves it identical to omitting them
    y = _tones([1200.0, 520.0, 1720.0], [1.1, 0.4, 1.5], noise=0.2, seed=9)
    base = bs.mel_bispectrogram(y)
    np.testing.assert_array_equal(base, bs.mel_bispectrogram(y, fmin=0.0, fmax=None))
    assert not np.array_equal(base, bs.mel_bispectrogram(y, fmin=300.0, fmax=3400.0))


def test_workers_bit_identical():
    # both threaded stages chunk over an independent axis (the IRFFT over
    # frames, the beta reduction over pair columns), so any workers value must
    # give bit-identical output. 7 s is past the reduction's threading
    # threshold, exercising the multi-chunk path as well as the serial one.
    y = _tones([1200.0, 520.0], [1.1, 0.4], noise=0.3, seed=5, n=7 * SR)
    a = bs.mel_bispectrogram(y, workers=1)
    np.testing.assert_array_equal(a, bs.mel_bispectrogram(y, workers=4))
    # an explicit workers overrides the reduction's auto thread cap; more chunks
    # than pairs is bounded to the pair count, and output stays bit-identical
    np.testing.assert_array_equal(a, bs.mel_bispectrogram(y, workers=64))


def test_estimate_beta_layout_independent():
    # beta must not depend on the memory layout of the filter output: numpy
    # sums pairwise or sequentially depending on axis contiguity, so a float32
    # accumulation would shift with the IRFFT output layout. The float64
    # accumulation makes C-contiguous z reproduce estimate_beta bit-for-bit.
    y = _tones([1200.0, 520.0], [1.1, 0.4], noise=0.3, seed=3)
    X = bs.stft(y, n_fft=N_FFT)
    mb = bs.mel_legendre_modal_bispectrum(12)
    beta = mb.estimate_beta(X)
    z = np.ascontiguousarray(mb.apply_modes(X))  # transform axis no longer contiguous
    # the constant third leg's filtered copy: IRFFT of the window-limited frame
    zc = np.fft.irfft(mb.third[:, None] * X, n=N_FFT, axis=0).astype(mb.dtype)
    manual = np.empty_like(beta)
    for i, (p, r) in enumerate(mb.pairs):
        manual[:, i] = (z[p] * z[r] * zc).sum(axis=0, dtype=np.float64)
    np.testing.assert_array_equal(beta, manual)


def test_finite_and_shaped():
    y = _tones([1200.0, 520.0, 1720.0], [1.1, 0.4, 1.5], noise=0.2, seed=4)
    B = bs.mel_bispectrogram(y, degree=12)
    assert B.shape[0] == 49
    assert np.isfinite(B).all()


def test_dtype_option():
    # float64 opt-in returns float64 and agrees with the float32 default
    y = _tones([1200.0, 520.0, 1720.0], [1.1, 0.4, 1.5], noise=0.2, seed=6)
    b32 = bs.mel_bispectrogram(y)
    b64 = bs.mel_bispectrogram(y, dtype=np.float64)
    assert b32.dtype == np.float32
    assert b64.dtype == np.float64
    np.testing.assert_allclose(b64, b32, rtol=1e-3, atol=0.05)


def test_mel_bispectrogram_bands_basis():
    # basis="bands" runs the band-pair estimator through the same pipeline
    y = _tones([1200.0, 520.0, 1720.0], [1.1, 0.4, 1.5], noise=0.2, seed=10)
    n_mels = 12
    B = bs.mel_bispectrogram(y, basis="bands", n_mels=n_mels)
    assert B.shape[0] == n_mels * (n_mels + 1) // 2
    assert np.isfinite(B).all()
    # bit-identical to running the band estimator's pipeline by hand
    mb = bs.mel_band_modal_bispectrum(n_mels)
    manual = bs.signed_log(mb.estimate_beta(bs.stft(y))).T
    np.testing.assert_array_equal(B, manual)
    with pytest.raises(ValueError, match="basis"):
        bs.mel_bispectrogram(y, basis="nope")


def test_signed_log_float32_overflow():
    # |x| / eps can overflow float32 to inf even though the compressed value
    # is tiny; those entries are redone in float64, in-range ones untouched
    x = np.array([1.0, -1e30], dtype=np.float32)
    out = bs.signed_log(x, eps=1e-15)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0], np.log1p(1e15), rtol=1e-6)
    np.testing.assert_allclose(out[1], -np.log1p(1e45), rtol=1e-6)


def test_modal_bispectrum_validation():
    modes = bs.legendre_modes(np.linspace(-1.0, 1.0, 201), 4)
    pairs = bs.modal_index_pairs(4)
    mb = bispectrum.ModalBispectrum(modes=modes, pairs=pairs, n_irfft=400)
    assert mb.degree == 4
    assert mb.dim == len(pairs)
    with pytest.raises(ValueError, match="n_irfft"):
        bispectrum.ModalBispectrum(modes=modes, pairs=pairs, n_irfft=64)
    with pytest.raises(ValueError, match="pairs"):
        bispectrum.ModalBispectrum(modes=modes, pairs=[(0, 5)], n_irfft=400)


def test_triangular_filterbank_rejects_bad_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        bs.triangular_filterbank([0.0, 120.0, 120.0, 300.0], n_fft=400, sr=SR)
    with pytest.raises(ValueError, match="at least 3"):
        bs.triangular_filterbank([0.0, 100.0], n_fft=400, sr=SR)


def test_raw_bispectrum_native_grid_and_valid_region():
    # the grid is the STFT's own bins windowed to [kmin, kmax]; valid cells are
    # k2 <= k1 with the sum frequency also inside the window, all else NaN.
    # 40 Hz bins at SR = 16000, n_fft = 400: [1000, 6000] Hz = bins [25, 150].
    y = _tones([1200.0, 520.0], [0.0, 0.3], noise=0.2, n=4000)
    k, _k2, B = bs.raw_bispectrum(bs.stft(y), kmin=25, kmax=150, return_full=True)
    f1 = k * 40.0
    assert np.array_equal(f1, np.arange(25, 151) * 40.0)
    F1, F2 = np.meshgrid(f1, f1, indexing="ij")
    valid = F1 + F2 <= 6000.0
    assert np.isfinite(B[valid]).all()
    assert np.isnan(B[~valid]).all()


def test_raw_bispectrum_nyquist_cut():
    # cells whose sum frequency f1 + f2 sits above sr / 2 are NaN, never
    # fabricated from a clipped STFT bin
    y = _tones([1200.0], [0.0], n=4000)
    k, _k2, B = bs.raw_bispectrum(
        bs.stft(y), kmin=50, return_full=True
    )  # 2000 Hz floor, 40 Hz bins
    f1 = k * 40.0
    a = np.searchsorted(f1, 3600.0)
    assert np.isfinite(B[a, a])  # f1 + f2 = 7200 Hz, below Nyquist
    b = np.searchsorted(f1, 4800.0)
    assert np.isnan(B[b, b])  # in the f2 <= f1 half, but f1 + f2 = 9600 Hz > sr / 2


def test_signed_log_properties():
    x = np.array([-5.0, -1e-20, 0.0, 1e-20, 5.0])
    out = bs.signed_log(x, eps=1e-15)
    assert np.sign(out[0]) == -1 and np.sign(out[-1]) == +1
    assert out[2] == 0.0
    # monotone increasing
    assert np.all(np.diff(out) >= 0)


def test_modal_gram_matrix_symmetric_psd():
    modes = bs.legendre_modes(np.linspace(-1.0, 1.0, 201), degree=12)
    G = bs.modal_gram_matrix(bs.modal_index_pairs(12), modes)
    assert G.shape == (49, 49)
    np.testing.assert_allclose(G, G.T, rtol=1e-6, atol=1e-6)  # symmetric
    assert np.linalg.eigvalsh(G).min() > -1e-3  # PSD (Q Qᵀ)


def test_reconstruct_structure():
    # reconstruction runs, is NaN outside the valid region, finite inside
    y = _tones([1200.0, 520.0, 1720.0], [1.1, 0.4, 1.5], noise=0.2, seed=5)
    Xstft = bs.stft(y)
    F = Xstft.shape[0]
    degree = 7
    pairs = bs.modal_index_pairs(degree)
    modes = bs.legendre_modes(np.linspace(-1.0, 1.0, F), degree)
    beta = bispectrum.project_bispectrum(Xstft, pairs, modes)
    assert beta.shape == (len(pairs),)
    gram = bs.modal_gram_matrix(pairs, modes)
    x, _y, B = bs.reconstruct_bispectrum(beta, pairs, modes, gram=gram, n_grid=24, return_full=True)
    assert B.shape == (24, 24)
    # sum-frequency range: finite inside, NaN outside (both halves now filled)
    finite = np.isfinite(B)
    assert finite.any() and (~finite).any()
    assert np.array_equal(B, B.T, equal_nan=True)
    assert np.isnan(B[-1, -1])  # (f1 max, f2 max) is outside the sum range


@pytest.mark.parametrize("degree,dim", [(7, 20), (10, 36), (12, 49)])
def test_dim_formula(degree, dim):
    assert bs.modal_pair_dim(degree) == dim
    assert len(bs.modal_index_pairs(degree)) == dim
    assert bs.mel_bispectrogram(_tones([1200.0], [0.0]), degree=degree).shape[0] == dim


def test_raw_bispectrum_window_matches_full_range():
    # kmin/kmax band-limit all three legs: inside the window the values equal
    # the corresponding block of the full computation bit-for-bit, and cells
    # whose sum frequency leaves the window are NaN (no out-of-band leakage)
    y = _tones([1200.0, 520.0], [0.0, 0.3], noise=0.2, n=4000)
    X = bs.stft(y, n_fft=N_FFT)
    k1, k2, vals = bispectrum.raw_bispectrum(X)
    assert k1.min() == 1 and k2.min() == 1  # the zero-frequency mode is excluded by default
    kc1, kc2, valsc = bispectrum.raw_bispectrum(X, kmin=10, kmax=60)
    assert kc1.min() == 10 and (kc1 + kc2).max() <= 60
    # inside the window the values equal the full computation bit-for-bit
    sel = (k2 >= 10) & (k1 + k2 <= 60)
    np.testing.assert_array_equal(valsc, vals[sel])


def test_average_bispectrum_at_triplets_input_guards():
    X = np.zeros((8, 0), dtype=np.complex128)
    with pytest.raises(ValueError, match="no realizations"):
        bispectrum.average_bispectrum_at_triplets(X, np.array([0]), np.array([0]), np.array([0]))
    X = np.zeros((8, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match="1-D"):
        bispectrum.average_bispectrum_at_triplets(
            X, np.zeros((2, 2), int), np.zeros((2, 2), int), np.zeros((2, 2), int)
        )


def test_apply_modes_honors_explicit_workers():
    # explicit workers must not be silently capped away on short inputs
    mb = bs.mel_legendre_modal_bispectrum(4)
    X = (np.ones((201, 100)) + 1j * np.ones((201, 100))).astype(np.complex64)
    z1 = mb.apply_modes(X, workers=1)
    z4 = mb.apply_modes(X, workers=4)  # T=100 < 4 * _MIN_COLS_PER_THREAD
    assert np.array_equal(z1, z4)  # bit-identical whatever the thread count


def test_rescale_to_symmetric():
    out = bs.rescale_to_symmetric([0.0, 5.0, 10.0], 0.0, 10.0)
    assert np.allclose(out, [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="lo < hi"):
        bs.rescale_to_symmetric([1.0], 3.0, 3.0)


def test_modal_bispectrum_derives_n_irfft():
    modes = np.ones((3, 201), dtype=np.float32)
    mb = bispectrum.ModalBispectrum(modes=modes, pairs=[(0, 0)])
    assert mb.n_irfft == 400  # 2 * (201 - 1), the even-length default


def test_mel_band_modal_bispectrum_structure():
    mb = bs.mel_band_modal_bispectrum(12)
    # the modes ARE the filterbank rows, one per band (kmin=1 default is a
    # no-op here: the Slaney bank is already zero at the zero frequency)
    np.testing.assert_array_equal(mb.modes, bs.mel_filterbank(n_mels=12))
    assert mb.pairs == [(p, r) for p in range(12) for r in range(p, 12)]
    np.testing.assert_array_equal(mb.third[1:], 1.0)
    assert mb.third[0] == 0.0


def test_mel_band_modal_bispectrum_detects_triplet():
    # Fourier coefficients built directly: unit tones on bins (12, 18, 30) with
    # phase-coupled (p3 = p1 + p2) vs independent phases, over T realizations.
    rng = np.random.default_rng(0)
    F, T = N_FFT // 2 + 1, 800

    def coeffs(coupled):
        X = 0.05 * (rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T)))
        p1, p2 = rng.uniform(0, 2 * np.pi, (2, T))
        p3 = p1 + p2 if coupled else rng.uniform(0, 2 * np.pi, T)
        for k, p in ((12, p1), (18, p2), (30, p3)):
            X[k] += np.exp(1j * p)
        return X

    mb = bs.mel_band_modal_bispectrum(10)
    assert len(mb.pairs) == 10 * 11 // 2
    beta_c = mb.estimate_beta(coeffs(coupled=True)).mean(axis=0)
    beta_u = mb.estimate_beta(coeffs(coupled=False)).mean(axis=0)

    # the strongest coefficient pairs two of the three legs' bands
    H = bs.mel_filterbank(n_mels=10)
    b1, b2, b3 = (int(np.argmax(H[:, k])) for k in (12, 18, 30))
    leg_pairs = {tuple(sorted(p)) for p in [(b1, b2), (b1, b3), (b2, b3)]}
    top = tuple(mb.pairs[int(np.argmax(np.abs(beta_c)))])
    assert top in leg_pairs
    # and coupling clearly beats the phase-randomized control
    assert np.abs(beta_c).max() > 20 * np.abs(beta_u).max()


def test_modal_bispectrum_frequency_window():
    rng = np.random.default_rng(1)
    F, T = N_FFT // 2 + 1, 64
    X = (rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T))).astype(np.complex128)
    modes = bs.mel_legendre_modes(6)
    pairs = bispectrum.modal_index_pairs(6)

    # windowing == zeroing the coefficients outside the window: every leg
    # (the pair modes and the constant third leg) sees the same restriction
    windowed = bispectrum.ModalBispectrum(modes=modes, pairs=pairs, kmin=20, kmax=120)
    X_masked = X.copy()
    X_masked[:20] = 0
    X_masked[121:] = 0
    unwindowed = bispectrum.ModalBispectrum(modes=modes, pairs=pairs, kmin=0)
    np.testing.assert_array_equal(windowed.estimate_beta(X), unwindowed.estimate_beta(X_masked))

    # the default window only removes the zero frequency
    full = bispectrum.ModalBispectrum(modes=modes, pairs=pairs)
    assert full.kmin == 1 and full.third[0] == 0
    np.testing.assert_array_equal(full.modes[:, 1:], modes[:, 1:])

    # a triplet with a leg outside the window disappears from the estimate
    p1, p2 = rng.uniform(0, 2 * np.pi, (2, T))
    Xc = 0.01 * (rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T)))
    for k, p in ((12, p1), (18, p2), (30, p1 + p2)):
        Xc[k] += np.exp(1j * p)
    seen = np.abs(full.estimate_beta(Xc).mean(axis=0)).max()
    blind = np.abs(
        bispectrum.ModalBispectrum(modes=modes, pairs=pairs, kmin=40).estimate_beta(Xc).mean(axis=0)
    ).max()
    assert seen > 100 * blind

    with pytest.raises(ValueError, match="kmin"):
        bispectrum.ModalBispectrum(modes=modes, pairs=pairs, kmin=10, kmax=5)
