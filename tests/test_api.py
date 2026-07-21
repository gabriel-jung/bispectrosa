"""The lazily-exported surface must stay in sync with the submodules' ``__all__``.

``bispectrosa._LAZY`` is a hand-written map (it can't be derived without
importing the optional submodules eagerly); this pins it to the ground truth.
"""

import importlib

import numpy as np
import pytest

import bispectrosa as bs


def test_lazy_map_matches_submodule_all():
    for module, names in bs._LAZY.items():
        mod = importlib.import_module(f"bispectrosa.{module}")
        assert names == set(mod.__all__), f"_LAZY['{module}'] drifted from __all__"


def test_all_names_resolve():
    for name in bs.__all__:
        getattr(bs, name)


def test_core_all_reexported():
    from bispectrosa import bispectrum

    missing = set(bispectrum.__all__) - set(bs.__all__)
    assert not missing, f"core names absent from the package root: {sorted(missing)}"


def test_dir_covers_exports_and_globals():
    d = dir(bs)
    assert set(bs.__all__) <= set(d)  # lazy names included for tab completion
    assert "bispectrum" in d  # real module globals are not hidden by __dir__


def _rand_stft(F=64, T=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T))


def _half_nan_square(X, **kw):
    k1, k2, vals = bs.raw_bispectrum(X, **kw)
    # the k1 == kmax row never holds a valid pair once kmin >= 1 (its only
    # partner k2 would need to be <= 0), so k1.max() alone underscopes the
    # grid; (k1 + k2).max() recovers the true kmax whenever the mask is
    # non-empty (the pair k1 = kmax - kmin, k2 = kmin always sums to kmax)
    kmax = (k1 + k2).max()
    kmin = k1.min()
    n = kmax - kmin + 1
    shape = (n, n) if vals.ndim == 1 else (n, n, vals.shape[1])
    B = np.full(shape, np.nan)
    B[k1 - kmin, k2 - kmin] = vals
    return np.arange(kmin, kmax + 1), B


def test_full_bispectrum_is_symmetric_and_keeps_corner_nan():
    X = _rand_stft()
    k, B = _half_nan_square(X, kmax=40)
    full = bs.full_bispectrum(B)
    finite = np.isfinite(B)
    assert np.array_equal(full[finite], B[finite])  # original cells untouched
    assert np.array_equal(full, full.T, equal_nan=True)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    corner = K1 + K2 > 40
    assert np.isnan(full[corner]).all()
    assert np.isfinite(full[~corner]).all()


def test_full_bispectrum_stack_mirrors_per_realization():
    X = _rand_stft()
    _, Bs = _half_nan_square(X, kmax=40, average=False)
    fs = bs.full_bispectrum(Bs)
    assert fs.shape == Bs.shape
    for t in range(Bs.shape[2]):
        assert np.array_equal(fs[:, :, t], bs.full_bispectrum(Bs[:, :, t]), equal_nan=True)


def test_raw_bispectrum_return_full_matches_utility():
    X = _rand_stft()
    k, B = _half_nan_square(X, kmax=40)
    kf, _, Bf = bs.raw_bispectrum(X, kmax=40, return_full=True)
    assert np.array_equal(kf, k)
    assert np.array_equal(Bf, bs.full_bispectrum(B), equal_nan=True)


def test_full_bispectrum_on_reconstruction():
    from bispectrosa import bispectrum

    X = _rand_stft()
    F = X.shape[0]
    z = bispectrum.rescale_to_symmetric(np.arange(F), 0, F - 1)
    modes = bispectrum.legendre_modes(z, 2)
    pairs = bispectrum.modal_index_pairs(2)
    beta = bispectrum.project_bispectrum(X, pairs, modes)
    x1, x2, vals = bispectrum.reconstruct_bispectrum(beta, pairs, modes, n_grid=32)
    grid = np.linspace(-1.0, 1.0, 32)
    Brec = np.full((32, 32), np.nan)
    Brec[np.searchsorted(grid, x1), np.searchsorted(grid, x2)] = vals
    full = bs.full_bispectrum(Brec)
    assert np.array_equal(full, full.T, equal_nan=True)
    assert np.isfinite(full).sum() > np.isfinite(Brec).sum()

    _, _, Bfull = bispectrum.reconstruct_bispectrum(beta, pairs, modes, n_grid=32, return_full=True)
    assert np.array_equal(Bfull, full, equal_nan=True)


def test_full_bispectrum_rejects_non_square():
    with pytest.raises(ValueError):
        bs.full_bispectrum(np.zeros((3, 4)))


def test_full_bispectrum_triple_matches_return_full_cropped_window():
    X = _rand_stft()
    k1, k2, values = bs.raw_bispectrum(X, kmin=5, kmax=40)
    k, k2_out, B = bs.full_bispectrum(k1, k2, values)
    kf, kf2, Bf = bs.raw_bispectrum(X, kmin=5, kmax=40, return_full=True)
    assert np.array_equal(k, kf)
    assert np.array_equal(k2_out, kf2)
    assert np.array_equal(B, Bf, equal_nan=True)


def test_full_bispectrum_triple_matches_return_full_average_false_stack():
    X = _rand_stft()
    k1, k2, values = bs.raw_bispectrum(X, kmin=5, kmax=40, average=False)
    k, _, B = bs.full_bispectrum(k1, k2, values)
    kf, _, Bf = bs.raw_bispectrum(X, kmin=5, kmax=40, average=False, return_full=True)
    assert np.array_equal(k, kf)
    assert np.array_equal(B, Bf, equal_nan=True)


def test_full_bispectrum_triple_rejects_empty_input():
    empty_i = np.array([], dtype=int)
    empty_v = np.array([], dtype=np.float64)
    with pytest.raises(ValueError):
        bs.full_bispectrum(empty_i, empty_i, empty_v)


def test_modal_shape_correlation_properties():
    rng = np.random.default_rng(3)
    betas = rng.standard_normal((4, 10))
    gram = np.eye(10)
    C = bs.modal_shape_correlation(betas, gram)
    # identity Gram reduces to plain cosine similarity
    Vn = betas / np.linalg.norm(betas, axis=1, keepdims=True)
    np.testing.assert_allclose(C, Vn @ Vn.T, rtol=1e-12)
    assert np.allclose(np.diag(C), 1.0)
    np.testing.assert_allclose(C, C.T, rtol=1e-12)


def test_modal_shape_correlation_matches_rebuild_inner_product():
    from bispectrosa import bispectrum

    X = _rand_stft(F=48, T=6)
    z = bispectrum.rescale_to_symmetric(np.arange(48), 0, 47)
    modes = bispectrum.legendre_modes(z, 3)
    prs = bispectrum.modal_index_pairs(3)
    gram = bispectrum.modal_gram_matrix(prs, modes)
    betas = np.stack([bispectrum.project_bispectrum(X * s, prs, modes) for s in (1.0, 1.0 + 1j)])
    C = bs.modal_shape_correlation(betas, gram)
    # equals the cosine of the rebuilt maps over the valid region
    maps = []
    for b in betas:
        _, _, Br = bispectrum.reconstruct_bispectrum(
            b, prs, modes, gram=gram, n_grid=48, return_full=True
        )
        maps.append(Br[np.isfinite(Br)])
    v0, v1 = maps
    cos = (v0 @ v1) / np.sqrt((v0 @ v0) * (v1 @ v1))
    np.testing.assert_allclose(C[0, 1], cos, atol=2e-2)


def test_snr_bispectrum_matches_manual():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((64, 8)) + 1j * rng.standard_normal((64, 8))
    P = (np.abs(X) ** 2).mean(1)
    k, _, B = bs.raw_bispectrum(X, kmax=50, return_full=True)
    S = bs.snr_bispectrum(B, P, kmin=1)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    K3 = np.minimum(K1 + K2, P.size - 1)
    expect = B / np.sqrt(P[K1] * P[K2] * P[K3])
    m = np.isfinite(B)
    np.testing.assert_allclose(S[m], expect[m], rtol=1e-12)
    assert np.array_equal(np.isfinite(S), m)


def test_snr_bispectrum_floor():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((32, 4)) + 1j * rng.standard_normal((32, 4))
    P = (np.abs(X) ** 2).mean(1)
    P[5] = 1e-12  # a near-dead bin
    _, _, B = bs.raw_bispectrum(X, return_full=True)
    S = bs.snr_bispectrum(B, P, floor=1e-2)
    Pf = np.maximum(P, 1e-2 * P.max())
    k = np.arange(1, 32)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    K3 = np.minimum(K1 + K2, P.size - 1)
    m = np.isfinite(B)
    np.testing.assert_allclose(S[m], (B / np.sqrt(Pf[K1] * Pf[K2] * Pf[K3]))[m], rtol=1e-12)


def test_mel_bin_bispectrum_matches_manual():
    from bispectrosa import filters

    rng = np.random.default_rng(7)
    X = rng.standard_normal((201, 8)) + 1j * rng.standard_normal((201, 8))
    _, _, B = bs.raw_bispectrum(X, return_full=True)
    Bm = bs.mel_bin_bispectrum(B, sr=16000, n_fft=400, n_mels=40)
    assert Bm.shape == (40, 40)
    H = filters.mel_filterbank(sr=16000, n_fft=400, n_mels=40).astype(np.float64)
    Mf = np.zeros((201, 201))
    Mf[1:, 1:] = np.where(np.isfinite(B), B, 0.0)
    sup = np.zeros((201, 201))
    sup[1:, 1:] = np.isfinite(B)
    w = H @ sup @ H.T
    covered = w > 0.02 * w.max()
    manual = H @ Mf @ H.T
    np.testing.assert_allclose(Bm[covered], manual[covered], rtol=1e-10)
    assert np.isnan(Bm[~covered]).all()


def test_reconstruct_bispectrum_bin_form_aligns_with_raw_grid():
    from bispectrosa import bispectrum

    X = _rand_stft()
    F = X.shape[0]
    z = bispectrum.rescale_to_symmetric(np.arange(F), 0, F - 1)
    modes = bispectrum.legendre_modes(z, 2)
    pairs = bispectrum.modal_index_pairs(2)
    beta = bispectrum.project_bispectrum(X, pairs, modes)

    # flat bin form enumerates exactly raw_bispectrum's wedge, absolute bins
    k1r, k2r, _ = bs.raw_bispectrum(X, kmin=1)
    k1b, k2b, vals = bispectrum.reconstruct_bispectrum(beta, pairs, modes, kmin=1)
    assert np.array_equal(k1b, k1r) and np.array_equal(k2b, k2r)
    assert vals.shape == k1b.shape

    # full form: same NaN geometry as the raw square, mirror symmetry included
    kr, _, Braw = bs.raw_bispectrum(X, kmin=1, return_full=True)
    kb, _, Bbin = bispectrum.reconstruct_bispectrum(beta, pairs, modes, kmin=1, return_full=True)
    assert np.array_equal(kb, kr)
    assert np.array_equal(np.isnan(Bbin), np.isnan(Braw))
    assert np.array_equal(Bbin, Bbin.T, equal_nan=True)


def test_reconstruct_bispectrum_bin_form_matches_full_range_grid():
    from bispectrosa import bispectrum

    X = _rand_stft()
    F = X.shape[0]
    z = bispectrum.rescale_to_symmetric(np.arange(F), 0, F - 1)
    modes = bispectrum.legendre_modes(z, 2)
    pairs = bispectrum.modal_index_pairs(2)
    beta = bispectrum.project_bispectrum(X, pairs, modes)

    # with kmin=0, kmax=F-1 the bin grid is the n_grid=F normalized grid
    # (gidx is the identity there), so the two paths must agree exactly
    _, _, Bbin = bispectrum.reconstruct_bispectrum(beta, pairs, modes, kmin=0, return_full=True)
    _, _, Bgrid = bispectrum.reconstruct_bispectrum(beta, pairs, modes, n_grid=F, return_full=True)
    assert np.array_equal(Bbin, Bgrid, equal_nan=True)

    # a kmin >= 1 window is the same evaluation restricted to those bins
    _, _, Bwin = bispectrum.reconstruct_bispectrum(beta, pairs, modes, kmin=2, return_full=True)
    assert np.array_equal(Bwin, Bbin[2:, 2:], equal_nan=True)


def test_reconstruct_bispectrum_bin_form_rejects_bad_window():
    from bispectrosa import bispectrum

    F = 16
    z = bispectrum.rescale_to_symmetric(np.arange(F), 0, F - 1)
    modes = bispectrum.legendre_modes(z, 2)
    pairs = bispectrum.modal_index_pairs(2)
    beta = np.zeros(len(pairs))
    with pytest.raises(ValueError):
        bispectrum.reconstruct_bispectrum(beta, pairs, modes, kmin=5, kmax=3)
    with pytest.raises(ValueError):
        bispectrum.reconstruct_bispectrum(beta, pairs, modes, kmin=0, kmax=F)


def test_full_bispectrum_triple_rejects_swapped_ordering():
    with pytest.raises(ValueError, match="k2 <= k1"):
        bs.full_bispectrum(np.array([1, 1]), np.array([3, 4]), np.array([1.0, 2.0]))


def test_snr_bispectrum_rejects_finite_cells_past_power():
    # a hand-built square whose finite corner needs sum legs beyond P
    B = np.ones((5, 5))
    P = np.ones(6)
    with pytest.raises(ValueError, match="sum leg"):
        bs.snr_bispectrum(B, P, kmin=1)
    # the estimator's own square (out-of-window cells NaN) still passes
    X = _rand_stft(F=16)
    _, _, Bq = bs.raw_bispectrum(X, kmin=1, return_full=True)
    P = np.abs(X).mean(1) ** 2
    out = bs.snr_bispectrum(Bq, P, kmin=1)
    assert np.array_equal(np.isnan(out), np.isnan(Bq))


def test_modal_shape_correlation_zero_norm_rows_are_nan():
    betas = np.array([[1.0, 0.0], [0.0, 0.0]])
    C = bs.modal_shape_correlation(betas, np.eye(2))
    assert C[0, 0] == pytest.approx(1.0)
    assert np.isnan(C[1, :]).all() and np.isnan(C[:, 1]).all()


def test_mel_band_modal_bispectrum_warns_on_narrow_bands():
    pytest.importorskip("librosa")
    from bispectrosa import filters

    with pytest.warns(UserWarning, match="fewer than 2 STFT bins"):
        filters.mel_band_modal_bispectrum(80, sr=16000, n_fft=400)
