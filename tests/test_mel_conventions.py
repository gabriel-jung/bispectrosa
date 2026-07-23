"""htk/norm passthrough: the mel-axis convention knobs reach librosa verbatim."""

import numpy as np
import pytest

import bispectrosa as bs

librosa = pytest.importorskip("librosa")

# a non-default combination of the passthrough knobs; htk=True alone is a
# warp-only A/B
HTK_STYLE = {"htk": True, "norm": None, "fmin": 20.0}


def _tone_mix(seed=0, n=16000):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    y = 0.2 * rng.standard_normal(n)
    for f, ph in ((1200.0, 1.1), (520.0, 0.4), (1720.0, 1.5)):
        y = y + np.sin(2 * np.pi * f * t + ph)
    return y.astype(np.float32)


def test_mel_filterbank_htk_norm_passthrough():
    # htk/norm/fmin forward verbatim: bit-identical to calling librosa directly
    ours = bs.mel_filterbank(sr=16000, n_fft=512, n_mels=80, **HTK_STYLE)
    ref = librosa.filters.mel(
        sr=16000, n_fft=512, n_mels=80, fmin=20.0, fmax=None, htk=True, norm=None
    )
    np.testing.assert_array_equal(ours, ref)


def test_mel_filterbank_defaults_are_slaney():
    # omitting the new kwargs reproduces the Slaney bank bit-for-bit
    np.testing.assert_array_equal(
        bs.mel_filterbank(), librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    )


def test_mel_legendre_modes_htk_reaches_filterbank():
    slaney = bs.mel_legendre_modes(4)
    htk = bs.mel_legendre_modes(4, **HTK_STYLE)
    assert htk.shape == slaney.shape
    assert not np.array_equal(htk, slaney)


def test_mel_band_modal_bispectrum_htk_modes():
    # the band estimator's modes ARE the filterbank rows; kwargs must reach them
    mb = bs.mel_band_modal_bispectrum(12, **HTK_STYLE)
    np.testing.assert_array_equal(mb.modes, bs.mel_filterbank(n_mels=12, **HTK_STYLE))


def test_mel_bispectrogram_htk_reaches_basis():
    y = _tone_mix()
    b_slaney = bs.mel_bispectrogram(y)
    b_htk = bs.mel_bispectrogram(y, **HTK_STYLE)
    assert b_htk.shape == b_slaney.shape
    assert np.isfinite(b_htk).all()
    assert not np.array_equal(b_htk, b_slaney)


def test_mel_bispectrogram_explicit_defaults_identity():
    # spelling out the defaults hits the same cached estimator and output
    y = _tone_mix(seed=1)
    np.testing.assert_array_equal(
        bs.mel_bispectrogram(y), bs.mel_bispectrogram(y, htk=False, norm="slaney")
    )


def test_degree_nesting_holds_under_htk():
    # degree-ordered prefix property is warp-independent. Tolerance is loose
    # enough for cross-platform float32 accumulation differences, which the
    # signed-log surfaces as absolute differences of a few 1e-4.
    y = _tone_mix(seed=2)
    full = bs.time_pool(bs.mel_bispectrogram(y, degree=12, **HTK_STYLE))
    sub = bs.time_pool(bs.mel_bispectrogram(y, degree=7, **HTK_STYLE))
    np.testing.assert_allclose(full[: bs.modal_pair_dim(7)], sub, rtol=1e-3, atol=2e-3)


def test_mel_spectrogram_htk():
    y = _tone_mix(seed=3)
    s_slaney = bs.mel_spectrogram(y)
    s_htk = bs.mel_spectrogram(y, **HTK_STYLE)
    assert s_htk.shape == s_slaney.shape
    assert not np.array_equal(s_htk, s_slaney)


def test_mel_bin_bispectrum_htk():
    y = _tone_mix(seed=4)
    _k1, _k2, B = bs.raw_bispectrum(bs.stft(y), kmin=1, kmax=120, return_full=True)
    m_slaney = bs.mel_bin_bispectrum(B, n_mels=20)
    m_htk = bs.mel_bin_bispectrum(B, n_mels=20, **HTK_STYLE)
    assert m_htk.shape == m_slaney.shape
    assert not np.array_equal(np.nan_to_num(m_htk), np.nan_to_num(m_slaney))
