"""Animate the time-varying bispectrum of a real recording (solo trumpet).

Run: ``uv run --group dev python examples/animate_bispectrum.py``
Writes ``examples/animate_bispectrum.gif`` and a single-frame PNG.
The clip is fetched by ``librosa.example`` on first run, then cached locally.
"""

from pathlib import Path

import numpy as np

import bispectrosa as bs

SR = 16000
HOP = 1024
FMAX = 3000.0  # the harmonic lattice lives in the low kHz; zoom in on it


def main():
    import librosa
    import matplotlib

    matplotlib.use("Agg")

    # ~5 s of solo trumpet: a strongly harmonic source, so its harmonics are
    # phase-coupled and the bispectrum shows a lattice at (i f0, j f0) that
    # moves with the melody
    y, _sr = librosa.load(librosa.example("trumpet"), sr=SR, mono=True)

    # per-frame full bispectrum stack on the native bins, time last
    X = bs.stft(y, hop_length=HOP)
    k, _, B = bs.raw_bispectrum(
        X, kmax=int(FMAX * bs.DEFAULT_N_FFT / SR), average=False, return_full=True
    )
    f1 = f2 = k * (SR / bs.DEFAULT_N_FFT)  # the cropped bin grid, in Hz
    times = np.arange(B.shape[2]) * HOP / SR

    # one shared symlog color scale for the frame plot and every gif frame;
    # 5 decades so the quieter notes stay visible next to the loudest one
    norm = bs.symlog_norm(B, decades=5)

    # single frame at the most strongly coupled moment
    k = int(np.argmax(np.nansum(np.abs(B), axis=(0, 1))))
    ax = bs.plot_bispectrum(
        f1,
        f2,
        B[:, :, k],
        cmap="RdBu_r",
        norm=norm,
        title=f"trumpet, t = {times[k]:.2f} s",
    )
    frame_png = Path(__file__).with_name("animate_frame.png")
    ax.figure.savefig(frame_png, dpi=110, bbox_inches="tight")
    print(f"wrote {frame_png}")

    anim = bs.animate_bispectrum(
        f1,
        f2,
        B,
        times=times,
        cmap="RdBu_r",
        norm=norm,
        title="trumpet",
        interval=80,
    )
    gif = Path(__file__).with_name("animate_bispectrum.gif")
    anim.save(gif, writer="pillow", fps=12)
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
