"""Smoke tests for the display layer (skipped when matplotlib is absent)."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

import bispectrosa as bs


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _stack(n=8, n_frames=4, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n, n_frames))
    B[0, -1, :] = np.nan  # a masked cell outside the valid region
    return B


def test_plot_bispectrum_draws_image():
    f = np.linspace(0.0, 4000.0, 8)
    ax = bs.plot_bispectrum(f, f, _stack()[:, :, 0], title="frame")
    assert len(ax.images) == 1


def test_plot_bispectrum_colorbar_toggle():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()[:, :, 0]
    ax = bs.plot_bispectrum(f, f, B, colorbar=False)
    assert len(ax.figure.axes) == 1  # panel only, no appended colorbar axis
    ax = bs.plot_bispectrum(f, f, B, colorbar=True)
    assert len(ax.figure.axes) == 2  # panel + colorbar


def test_symlog_colorbar_reusable_and_symlog_ticks():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()[:, :, 0]
    ax = bs.plot_bispectrum(f, f, B, norm=bs.symlog_norm(B, decades=3), colorbar=False)
    cbar = bs.symlog_colorbar(ax.images[0])  # ax defaults to im.axes
    assert cbar.ax in ax.figure.axes
    # the crowded zero region is a single centered tick, not two collided decades
    labels = [t.get_text() for t in cbar.ax.get_yticklabels()]
    assert any(r"\pm" in s for s in labels)


def test_symlog_colorbar_into_given_cax():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()[:, :, 0]
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    bs.plot_bispectrum(
        f, f, B, ax=ax, norm=bs.symlog_norm(B, decades=3), colorbar=False, aspect="auto"
    )
    cbar = bs.symlog_colorbar(ax.images[0], cax=cax)
    assert cbar.ax is cax  # drawn into the caller's axis, no divider axis created
    assert ax.images[0].axes.get_aspect() == "auto"


def test_plot_bispectrum_scientific_defaults():
    f = np.linspace(0.0, 4000.0, 8)
    ax = bs.plot_bispectrum(f, f, _stack()[:, :, 0], colorbar=False)
    # inward ticks on all four sides, with minor ticks
    assert ax.xaxis.get_tick_params()["direction"] == "in"
    assert ax.xaxis.majorTicks[0].tick2line.get_visible()  # top ticks on
    assert ax.yaxis.majorTicks[0].tick2line.get_visible()  # right ticks on
    assert ax.xaxis.get_minorticklocs().size > 0
    # math labels and a grey (not white) masked-region facecolor
    assert ax.get_xlabel() == "$f_1$" and ax.get_ylabel() == "$f_2$"
    assert ax.get_facecolor()[:3] != (1.0, 1.0, 1.0)


def test_symlog_colorbar_thins_labels_on_tall_range():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()[:, :, 0] * 1e3  # wide dynamic range -> many decades
    ax = bs.plot_bispectrum(f, f, B, norm=bs.symlog_norm(B, decades=6), colorbar=False)
    cbar = bs.symlog_colorbar(ax.images[0])
    nonblank = [t.get_text() for t in cbar.ax.get_yticklabels() if t.get_text()]
    # every decade gets a tick, but labels are thinned (fewer than the tick count)
    assert len(nonblank) <= len(cbar.get_ticks())


def test_plot_bispectrum_mirror_fills_symmetric_half():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()[:, :, 0]  # NaN at (0, -1), finite mirror at (-1, 0)

    shown = np.asarray(bs.plot_bispectrum(f, f, B).images[0].get_array(), float)
    assert shown[-1, 0] == B[-1, 0]  # displayed array is B.T
    assert np.isfinite(shown).all()  # the NaN cell was filled by symmetry

    shown = np.asarray(bs.plot_bispectrum(f, f, B, mirror=False).images[0].get_array(), float)
    assert np.isnan(shown[-1, 0])  # unmirrored view keeps the gap

    # input must not be modified in place
    assert np.isnan(B[0, -1])


def test_plot_bispectrum_mirror_skipped_on_mismatched_axes():
    B = _stack()[:, :, 0]  # square, NaN at (0, -1)
    f1 = np.linspace(0.0, 4000.0, 8)
    f2 = np.linspace(0.0, 2000.0, 8)  # same length, different axis
    shown = np.asarray(bs.plot_bispectrum(f1, f2, B).images[0].get_array(), float)
    assert np.isnan(shown[-1, 0])  # (f2, f1) is not this cell's mirror; stays blank


def test_plot_bispectrum_freq_scale_switches_renderer():
    f = np.linspace(40.0, 4000.0, 8)
    B = _stack()[:, :, 0]
    # linear stays imshow (the tested, non-breaking default)
    ax = bs.plot_bispectrum(f, f, B, freq_scale="linear", colorbar=False)
    assert len(ax.images) == 1 and len(ax.collections) == 0
    plt.close(ax.figure)
    # log / mel warp the axes and draw with pcolormesh (mappable in collections)
    for scale, xscale in [("log", "log"), ("mel", "function")]:
        ax = bs.plot_bispectrum(f, f, B, freq_scale=scale, colorbar=True)
        assert len(ax.images) == 0 and len(ax.collections) == 1
        assert ax.get_xscale() == xscale and ax.get_yscale() == xscale
        assert len(ax.figure.axes) == 2  # colorbar still drawn for a QuadMesh
        plt.close(ax.figure)


def test_plot_bispectrum_freq_scale_invalid():
    f = np.linspace(40.0, 4000.0, 8)
    with pytest.raises(ValueError, match="linear"):
        bs.plot_bispectrum(f, f, _stack()[:, :, 0], freq_scale="bogus")


def test_set_freq_scale_axis_and_validation():
    _fig, ax = plt.subplots()
    bs.set_freq_scale(ax, "mel", axis="x")
    assert ax.get_xscale() == "function" and ax.get_yscale() == "linear"
    bs.set_freq_scale(ax, "log", axis="y")
    assert ax.get_yscale() == "log"
    with pytest.raises(ValueError, match="scale"):
        bs.set_freq_scale(ax, "bogus")
    with pytest.raises(ValueError, match="axis"):
        bs.set_freq_scale(ax, "mel", axis="z")


def test_mel_warp_roundtrips_and_monotone():
    from bispectrosa.display import _hz_to_mel, _mel_to_hz

    hz = np.array([0.0, 200.0, 1000.0, 4000.0, 8000.0])
    np.testing.assert_allclose(_mel_to_hz(_hz_to_mel(hz)), hz, rtol=1e-9, atol=1e-6)
    assert np.all(np.diff(_hz_to_mel(hz)) > 0)  # a valid (monotone) axis warp
    librosa = pytest.importorskip("librosa")  # and it is the Slaney warp the mel bands use
    np.testing.assert_allclose(_hz_to_mel(hz), librosa.hz_to_mel(hz, htk=False), rtol=1e-6)


def test_animate_nonsymmetric_uses_stack_norm():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()
    B[1, 0, -1] = 50.0  # the stack max lives in the last frame
    _fig, ax = plt.subplots()
    anim = bs.animate_bispectrum(f, f, B, ax=ax, symmetric=False, mirror=False)
    im = ax.images[0]
    assert im.norm.vmax == 50.0  # scaled over the stack, not autoscaled to frame 0
    assert im.norm.vmin == np.nanmin(B)
    anim.to_jshtml()  # render; also silences matplotlib's deleted-unrendered warning


def test_symlog_norm_signed_and_positive():
    signed = np.array([-3e-4, 0.0, 5e-2, np.nan])
    norm = bs.symlog_norm(signed, decades=3.0)
    assert isinstance(norm, colors.SymLogNorm)
    assert norm.vmax == pytest.approx(0.1)  # power of ten above max |B|
    assert norm.vmin == pytest.approx(-0.1)

    positive = np.array([1e-4, 2e-2])
    norm = bs.symlog_norm(positive, decades=2.0)
    assert isinstance(norm, colors.LogNorm)
    assert norm.vmax == pytest.approx(0.1)
    assert norm.vmin == pytest.approx(1e-3)

    assert bs.symlog_norm(np.full(3, np.nan)).vmax == 1.0  # degenerate input

    touching_zero = np.array([0.0, 5e-2])  # a LogNorm could not render the zeros
    norm = bs.symlog_norm(touching_zero, decades=3.0)
    assert isinstance(norm, colors.SymLogNorm)
    assert norm.vmin == 0.0
    assert norm.vmax == pytest.approx(0.1)


def test_animate_bispectrum_renders_all_frames():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()
    anim = bs.animate_bispectrum(f, f, B, times=np.arange(B.shape[2]) * 0.05)
    assert isinstance(anim, FuncAnimation)
    anim.to_jshtml()  # renders every frame through the update path


def test_animate_rejects_2d_input():
    f = np.linspace(0.0, 4000.0, 8)
    with pytest.raises(ValueError, match="n_frames"):
        bs.animate_bispectrum(f, f, _stack()[:, :, 0])


def test_raw_bispectrum_stack_matches_average():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(8000).astype(np.float32)
    X = bs.stft(y)
    k1, k2, v_avg = bs.raw_bispectrum(X, kmax=75)
    _k1, _k2, v_stack = bs.raw_bispectrum(X, kmax=75, average=False)
    assert v_stack.ndim == 2 and v_stack.shape[0] == v_avg.size
    np.testing.assert_allclose(v_stack.mean(axis=1), v_avg)


def test_mel_warp_matches_librosa():
    # display.py hand-copies the Slaney warp constants so the viz extra never
    # needs librosa; this pins them to librosa's definition when it is around
    librosa = pytest.importorskip("librosa")
    from bispectrosa.display import _hz_to_mel, _mel_to_hz

    hz = np.linspace(0.0, 8000.0, 257)
    np.testing.assert_allclose(_hz_to_mel(hz), librosa.hz_to_mel(hz, htk=False), rtol=1e-12)
    mel = _hz_to_mel(hz)
    np.testing.assert_allclose(_mel_to_hz(mel), librosa.mel_to_hz(mel, htk=False), rtol=1e-12)


def test_animate_rejects_short_times():
    f = np.linspace(0.0, 4000.0, 8)
    B = _stack()
    with pytest.raises(ValueError, match="times"):
        bs.animate_bispectrum(f, f, B, times=np.arange(B.shape[2] - 1) * 0.05)
