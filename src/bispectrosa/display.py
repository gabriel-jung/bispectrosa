"""Plotting helpers for the 2-D bispectrum (matplotlib optional).

Install with the ``viz`` extra: ``pip install 'bispectrosa[viz]'``.

Organized as a plot is assembled: the color scaling (:func:`symlog_norm`,
:func:`symlog_colorbar`), the frequency-axis warps (:func:`set_freq_scale`),
the shared frame-drawing helpers, and the two renderers
(:func:`plot_bispectrum`, :func:`animate_bispectrum`).
"""

import numpy as np

__all__ = [
    "symlog_norm",
    "symlog_colorbar",
    "set_freq_scale",
    "plot_bispectrum",
    "animate_bispectrum",
]


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Plotting needs matplotlib; install: pip install 'bispectrosa[viz]'"
        ) from exc
    return plt


# --------------------------------------------------------------------------- #
# Color scaling
# --------------------------------------------------------------------------- #
def _symmetric_norm(B):
    """Linear color normalization centered on zero (bispectra are signed)."""
    from matplotlib import colors

    finite = B[np.isfinite(B)]
    vmax = np.abs(finite).max() if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0  # all-zero panel: keep a usable color scale
    return colors.Normalize(vmin=-vmax, vmax=vmax)


def symlog_norm(B: np.ndarray, decades: float = 3.0):
    """Log-scale color normalization sized to a bispectrum's dynamic range.

    Bispectrum values are signed and span many orders of magnitude, so a
    linear color scale shows only the few largest frequency triplets. This builds a
    symmetric-log normalization (log-scaled away from zero, linear through
    it) whose top is the power of ten just above ``max |B|`` and which spans
    ``decades`` decades below it. Strictly positive data get a plain log
    scale; data touching zero keep the symmetric-log form (its linear zone
    can render exact zeros, which a log scale cannot).
    Pass the result as ``norm=`` to :func:`plot_bispectrum` or
    :func:`animate_bispectrum`; for an animation, build it from the
    full frame stack so every frame shares one scale.

    Parameters
    ----------
    B : np.ndarray
        Bispectrum values (any shape); NaN entries are ignored.
    decades : float
        Dynamic range of the log scale, in decades below the maximum.

    Returns
    -------
    matplotlib.colors.SymLogNorm or matplotlib.colors.LogNorm
    """
    _require_mpl()
    from matplotlib import colors

    B = np.asarray(B, dtype=float)
    finite = B[np.isfinite(B)]
    peak = np.abs(finite).max() if finite.size else 0.0
    vmax = 10.0 ** np.ceil(np.log10(peak)) if peak > 0 else 1.0
    linthresh = vmax / 10.0**decades
    lo = finite.min() if finite.size else -1.0  # empty input keeps the signed form
    if lo > 0:
        return colors.LogNorm(vmin=linthresh, vmax=vmax)
    if lo == 0:
        return colors.SymLogNorm(linthresh=linthresh, linscale=0.01, vmin=0.0, vmax=vmax, base=10)
    return colors.SymLogNorm(linthresh=linthresh, linscale=0.01, vmin=-vmax, vmax=vmax, base=10)


def symlog_colorbar(im, ax=None, cax=None, label=None, size="5%", pad=0.1):
    """Colorbar with readable symmetric-log ticks, sized to the panel.

    The colorbar :func:`plot_bispectrum` and :func:`animate_bispectrum` draw,
    exposed for reuse when laying out a bispectrum by hand (e.g. a custom grid
    of panels, or a panel that already carries other axes). For a
    :class:`~matplotlib.colors.SymLogNorm` (as built by :func:`symlog_norm`),
    the crowded zero region is replaced by a single ``±10^k`` tick with decade
    ticks above it; any other norm keeps matplotlib's default ticks.

    With ``cax=None`` a divider-based colorbar axis is appended to the right of
    ``ax``, tracking the panel box exactly (the height of an aspect-equal plot).
    Pass an explicit ``cax`` (e.g. a gridspec cell spanning only the bispectrum
    row) to control the bar's height and position yourself, for instance when
    another axis shares the panel's column; this path is also compatible with
    ``layout="constrained"``, which the divider path is not.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        The mappable to draw the colorbar for, e.g. the return value of
        ``ax.imshow`` (or of :func:`plot_bispectrum` via its image).
    ax : matplotlib.axes.Axes, optional
        Panel the colorbar attaches to; defaults to ``im.axes``.
    cax : matplotlib.axes.Axes, optional
        Existing axis to draw the colorbar into. When given, ``size`` and
        ``pad`` are ignored (no divider axis is created).
    label : str, optional
        Colorbar label, e.g. ``"$B(f_1, f_2)$"``; placed horizontally above the
        bar (clear of the tick labels), as a vertical colorbar reads best.
    size : str
        Colorbar width as a percentage of the panel width (divider path only).
    pad : float
        Gap between the panel and the colorbar, in inches (divider path only).

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    _require_mpl()
    from matplotlib import colors

    if ax is None:
        ax = im.axes
    if cax is None:
        from mpl_toolkits import axes_grid1

        # a divider-based cax tracks the (aspect-equal) panel box exactly, so the
        # bar always has the height of the plot itself
        cax = axes_grid1.make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
        cbar = ax.figure.colorbar(im, cax=cax)
        ax.figure.sca(ax)  # append_axes leaves cax current; restore the panel
    else:
        cbar = cax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(direction="in")
    cbar.minorticks_off()  # drop the stray auto minor ticks around the symlog centre
    norm = im.norm
    if isinstance(norm, colors.SymLogNorm):
        # matplotlib crowds 0 and ±linthresh together; keep one central tick
        # labeled ±linthresh and decade ticks above it, as in the log case
        lt, vmax = norm.linthresh, norm.vmax
        decs = np.arange(np.floor(np.log10(lt)) + 1, np.floor(np.log10(vmax)) + 1).astype(int)
        pos = 10.0**decs
        cbar.set_ticks(np.concatenate([-pos[::-1], [0.0], pos]))
        lt_exp = np.log10(lt)
        center = (
            f"$\\pm 10^{{{int(round(lt_exp))}}}$"
            if abs(lt_exp - round(lt_exp)) < 1e-9
            else f"$\\pm${lt:.1e}"
        )
        # keep a tick at every decade, but label at most ~4 per side (always the
        # outermost) so a tall bar spanning many decades is not crowded
        n = len(decs)
        stride = max(1, int(np.ceil(n / 4)))
        keep = [(n - 1 - i) % stride == 0 for i in range(n)]
        pos_lbl = [f"$10^{{{d}}}$" if k else "" for d, k in zip(decs, keep, strict=True)]
        neg_lbl = [f"$-10^{{{d}}}$" if k else "" for d, k in zip(decs, keep, strict=True)][::-1]
        cbar.set_ticklabels(neg_lbl + [center] + pos_lbl)
    if label is not None:
        # a vertical bar reads best with the quantity on top (horizontal), clear
        # of the tick labels, rather than rotated along the side
        cbar.ax.set_title(label, pad=6)
    return cbar


# --------------------------------------------------------------------------- #
# Frequency-axis warps
# --------------------------------------------------------------------------- #
#: Frequency-axis warps accepted by ``set_freq_scale`` and ``plot_bispectrum``.
_FREQ_SCALES = ("linear", "log", "mel")

# Slaney mel warp (the same bands the modal basis uses), for the frequency axes.
_MEL_F_SP = 200.0 / 3.0
_MEL_LOG_HZ = 1000.0
_MEL_LOG_MEL = _MEL_LOG_HZ / _MEL_F_SP  # 15.0
_MEL_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=float)
    return np.where(
        hz >= _MEL_LOG_HZ,
        _MEL_LOG_MEL + np.log(np.maximum(hz, 1e-9) / _MEL_LOG_HZ) / _MEL_LOGSTEP,
        hz / _MEL_F_SP,
    )


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=float)
    return np.where(
        mel >= _MEL_LOG_MEL,
        _MEL_LOG_HZ * np.exp(_MEL_LOGSTEP * (mel - _MEL_LOG_MEL)),
        _MEL_F_SP * mel,
    )


def set_freq_scale(ax, scale: str = "linear", axis: str = "both") -> None:
    """Warp a frequency axis to ``"linear"``, ``"log"``, or ``"mel"`` (Slaney).

    The companion to :func:`plot_bispectrum`'s ``freq_scale``: apply the *same*
    warp to a paired axis (e.g. a power-spectrum strip sharing a bispectrum's
    frequency axis) so the two line up. ``"mel"`` uses the Slaney warp of the
    modal mel bands, giving low frequencies more room without ``"log"``'s blow-up
    of the sub-100 Hz region; ``"log"`` needs strictly positive axis limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    scale : {"linear", "log", "mel"}
    axis : {"both", "x", "y"}
        Which axis to warp.
    """
    if scale not in _FREQ_SCALES:
        raise ValueError(f"scale must be one of {_FREQ_SCALES}, got {scale!r}")
    if axis not in ("both", "x", "y"):
        raise ValueError(f"axis must be 'both', 'x', or 'y', got {axis!r}")
    for which in ("x", "y") if axis == "both" else (axis,):
        setter = ax.set_xscale if which == "x" else ax.set_yscale
        if scale == "mel":
            setter("function", functions=(_hz_to_mel, _mel_to_hz))
        else:
            setter(scale)


# --------------------------------------------------------------------------- #
# Frame drawing, shared by both renderers
# --------------------------------------------------------------------------- #
#: Out-of-domain (NaN) cells show as mid grey ("no data"), kept clearly off-white so it
#: never blends with a diverging colormap's zero. Shared by both frame renderers.
_MASK_FACECOLOR = "#C0C0C0"


def _mirror_symmetric_half(B, f1, f2, mirror):
    """Complete the symmetric half: fill NaN at ``(f1, f2)`` from ``(f2, f1)``.

    Exact, not cosmetic: ``B(f1, f2) = B(f2, f1)`` for the bispectrum. The
    estimators' ``return_full=True`` squares arrive already mirror-filled (this
    is then a no-op); a hand-built half-wedge square is completed here. Applies
    only with
    ``mirror`` set and a square grid on a shared axis (``B`` is returned
    unchanged otherwise); works on single frames and ``(f1, f2, t)`` stacks
    alike.
    """
    if not (mirror and B.shape[0] == B.shape[1] and np.array_equal(f1, f2)):
        return B
    swapped = np.swapaxes(B, 0, 1)
    return np.where(np.isnan(B), swapped, B)


def _style_axes(ax):
    """Inward major and minor ticks on all four sides (a clean scientific default)."""
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=4)
    ax.tick_params(which="minor", length=2)


def _panel_axes(ax):
    """Return the target axes, creating the standard panel figure when ``ax`` is None."""
    if ax is not None:
        return ax
    plt = _require_mpl()
    # constrained layout keeps the labels inside saved animations, where
    # savefig's bbox_inches="tight" cannot help
    _fig, ax = plt.subplots(figsize=(4.8, 4), layout="constrained")
    return ax


def _label_panel(ax):
    """The shared panel dressing: f1/f2 axis labels and the house tick style."""
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    _style_axes(ax)


def _imshow_frame(ax, f1, f2, B, cmap, norm, aspect="equal"):
    """Draw one ``B(f1, f2)`` frame; NaN cells (outside the valid region) show as grey."""
    ax.set_facecolor(_MASK_FACECOLOR)
    # imshow's extent is the outer pixel edges, while f1/f2 are cell centers:
    # pad by half a cell so each pixel sits centered on its frequency, matching
    # _pcolor_frame's shading="nearest" and any axis-sharing spectrum strip
    d1 = (f1[-1] - f1[0]) / max(len(f1) - 1, 1)
    d2 = (f2[-1] - f2[0]) / max(len(f2) - 1, 1)
    return ax.imshow(
        B.T,
        origin="lower",
        aspect=aspect,  # "equal" keeps the panel square (f1, f2 share units)
        cmap=cmap,
        norm=norm,
        extent=[f1[0] - d1 / 2, f1[-1] + d1 / 2, f2[0] - d2 / 2, f2[-1] + d2 / 2],
    )


def _pcolor_frame(ax, f1, f2, B, cmap, norm, freq_scale, aspect="equal"):
    """Draw one frame with ``pcolormesh`` and warp the axes (imshow can't follow a warp).

    Self-contained like :func:`_imshow_frame`: owns the mask facecolor, the frequency
    warp, the axis limits, and the panel aspect.
    """
    ax.set_facecolor(_MASK_FACECOLOR)
    im = ax.pcolormesh(
        f1, f2, np.ma.masked_invalid(B.T, copy=False), cmap=cmap, norm=norm, shading="nearest"
    )
    set_freq_scale(ax, freq_scale)
    ax.set_xlim(f1[0], f1[-1])
    ax.set_ylim(f2[0], f2[-1])
    if aspect == "equal":
        ax.set_box_aspect(1)  # data-equal is meaningless on a warp; keep the panel square
    elif aspect != "auto":
        ax.set_aspect(aspect)
    return im


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def plot_bispectrum(
    f1: np.ndarray,
    f2: np.ndarray,
    B: np.ndarray,
    ax=None,
    cmap: str = "magma",
    title: str | None = None,
    symmetric: bool = True,
    norm=None,
    mirror: bool = True,
    colorbar: bool = True,
    colorbar_label: str = "$B(f_1, f_2)$",
    aspect: str = "equal",
    freq_scale: str = "linear",
):
    """Heatmap of a 2-D bispectrum ``B(f1, f2)`` (NaN cells masked out).

    Parameters
    ----------
    f1, f2 : np.ndarray
        Axis coordinates (bin index, Hz, or normalized), e.g. from
        :func:`~bispectrosa.bispectrum.raw_bispectrum` (with
        ``return_full=True``) or :func:`~bispectrosa.bispectrum.reconstruct_bispectrum`.
    B : np.ndarray, shape (len(f1), len(f2))
        Bispectrum values; NaN marks cells outside the valid region.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created when omitted.
    cmap : str
        Matplotlib colormap name.
    title : str, optional
        Axes title.
    symmetric : bool
        Center the color scale on zero (bispectra are signed).
    norm : matplotlib.colors.Normalize, optional
        Explicit color normalization, e.g. from :func:`symlog_norm`;
        overrides ``symmetric``.
    mirror : bool
        Show the full symmetric bispectrum: NaN cells in the ``f2 > f1`` half
        are filled from their transpose partners via ``B(f1, f2) = B(f2, f1)``,
        exactly. Library squares (``return_full=True``) are already filled, so
        this only acts on hand-built half-wedge input. Needs a square ``B`` on
        a shared axis (skipped otherwise); ``False`` shows the input as is.
    colorbar : bool
        Draw the panel-height colorbar (:func:`symlog_colorbar`). Pass
        ``False`` to place your own, e.g. to control its padding or to add
        other axes (a power-spectrum strip) beside the panel first.
    colorbar_label : str
        Label for the built-in colorbar (ignored when ``colorbar=False``).
    aspect : str
        Image aspect. ``"equal"`` (default) keeps the panel square, since
        ``f1`` and ``f2`` share units; ``"auto"`` fills the axes box (useful in
        a grid where the panel should match a neighbouring axis's height). On a
        warped ``freq_scale`` (below), ``"equal"`` keeps a square *panel* (data
        units are no longer comparable).
    freq_scale : {"linear", "log", "mel"}
        Frequency-axis warp for both ``f1`` and ``f2`` (display only, no
        rebinning). ``"linear"`` (default) draws with ``imshow`` unchanged.
        ``"log"`` / ``"mel"`` give the low frequencies more room, where real
        bispectra concentrate, and are drawn with ``pcolormesh`` (so the mappable
        is in ``ax.collections``, not ``ax.images``). Match a paired
        power-spectrum axis with :func:`set_freq_scale`. ``"log"`` needs
        ``f1[0] > 0``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_mpl()
    if freq_scale not in _FREQ_SCALES:  # fail before drawing anything
        raise ValueError(f"freq_scale must be one of {_FREQ_SCALES}, got {freq_scale!r}")
    ax = _panel_axes(ax)
    B = _mirror_symmetric_half(B, f1, f2, mirror)
    if norm is None and symmetric:
        norm = _symmetric_norm(B)
    # imshow maps a uniform grid linearly (fast, and the animation's set_data path);
    # a warped axis needs pcolormesh, which places each cell at its coordinate
    if freq_scale == "linear":
        im = _imshow_frame(ax, f1, f2, B, cmap, norm, aspect=aspect)
    else:
        im = _pcolor_frame(ax, f1, f2, B, cmap, norm, freq_scale, aspect=aspect)
    _label_panel(ax)
    if title:
        ax.set_title(title)
    if colorbar:
        symlog_colorbar(im, ax, label=colorbar_label)
    return ax


def animate_bispectrum(
    f1: np.ndarray,
    f2: np.ndarray,
    B: np.ndarray,
    times: np.ndarray | None = None,
    ax=None,
    cmap: str = "magma",
    title: str | None = None,
    symmetric: bool = True,
    norm=None,
    mirror: bool = True,
    interval: int = 100,
):
    """Animate a time-varying bispectrum ``B(f1, f2, t)``, one frame per step.

    All frames share one color scale (computed over the full stack when
    ``norm`` is omitted) so brightness is comparable across time. Keep a
    reference to the returned animation while displaying it; write it out
    with ``anim.save("bisp.gif")`` or embed it with ``anim.to_jshtml()``.

    Parameters
    ----------
    f1, f2 : np.ndarray
        Axis coordinates (bin index, Hz, or normalized).
    B : np.ndarray, shape (len(f1), len(f2), n_frames)
        Bispectrum frames, time last (e.g. from
        :func:`~bispectrosa.bispectrum.raw_bispectrum` with
        ``average=False, return_full=True``); NaN marks cells outside the
        valid region.
    times : np.ndarray, optional
        Frame times in seconds, used in the per-frame title; frame indices
        are shown when omitted.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created when omitted.
    cmap : str
        Matplotlib colormap name.
    title : str, optional
        Title prefix, followed by the frame time or index.
    symmetric : bool
        Center the shared color scale on zero (bispectra are signed).
    norm : matplotlib.colors.Normalize, optional
        Explicit color normalization, e.g. from :func:`symlog_norm` on the
        full stack; overrides ``symmetric``.
    mirror : bool
        Show the full symmetric bispectrum, filling ``f2 > f1`` from the
        computed half via ``B(f1, f2) = B(f2, f1)`` (see
        :func:`plot_bispectrum`); needs square frames, skipped otherwise.
    interval : int
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    _require_mpl()
    from matplotlib.animation import FuncAnimation

    B = np.asarray(B)
    if B.ndim != 3:
        raise ValueError(f"expected B of shape (len(f1), len(f2), n_frames), got {B.shape}")
    n_frames = B.shape[2]
    if times is not None and len(times) != n_frames:
        # fail now rather than with an IndexError mid-save at the first missing frame
        raise ValueError(f"times has {len(times)} entries but B has {n_frames} frames")
    ax = _panel_axes(ax)
    B = _mirror_symmetric_half(B, f1, f2, mirror)
    if norm is None:
        if symmetric:
            norm = _symmetric_norm(B)
        else:
            # still fix one scale over the whole stack; without it imshow
            # autoscales to frame 0 and later frames clip or wash out
            finite = B[np.isfinite(B)]
            if finite.size:
                from matplotlib import colors

                norm = colors.Normalize(vmin=finite.min(), vmax=finite.max())

    def frame_title(i):
        stamp = f"t = {times[i]:.2f} s" if times is not None else f"frame {i}"
        return f"{title}, {stamp}" if title else stamp

    im = _imshow_frame(ax, f1, f2, B[:, :, 0], cmap, norm)
    _label_panel(ax)
    ax.set_title(frame_title(0))
    symlog_colorbar(im, ax)

    def update(i):
        im.set_data(B[:, :, i].T)
        ax.set_title(frame_title(i))
        return (im,)

    return FuncAnimation(ax.figure, update, frames=n_frames, interval=interval)
