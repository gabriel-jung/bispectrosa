"""Real-speed bispectrum video of the trumpet clip, with sound (for the README).

Run: ``uv run --group dev python examples/animate_bispectrum_video.py``
Writes ``examples/animate_bispectrum.mp4``: the per-frame bispectrum animated at
one frame per STFT hop (real time) and muxed with the audio itself.
Needs the ``ffmpeg`` binary on PATH; the clip is fetched by ``librosa.example``
on first run, then cached locally.
"""

import subprocess
from pathlib import Path

import numpy as np

import bispectrosa as bs

SR = 16000
HOP = 1024
FMAX = 3000.0  # the harmonic lattice lives in the low kHz; zoom in on it


def main():
    import librosa
    import matplotlib
    import soundfile as sf

    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter

    # same stack as animate_bispectrum.py: ~5 s of solo trumpet, per-frame full
    # bispectrum on the native bins, one shared symlog color scale
    y, _sr = librosa.load(librosa.example("trumpet"), sr=SR, mono=True)
    X = bs.stft(y, hop_length=HOP)
    k, _, B = bs.raw_bispectrum(
        X, kmax=int(FMAX * bs.DEFAULT_N_FFT / SR), average=False, return_full=True
    )
    f1 = f2 = k * (SR / bs.DEFAULT_N_FFT)
    times = np.arange(B.shape[2]) * HOP / SR
    norm = bs.symlog_norm(B, decades=5)

    # real time: one animation frame per STFT hop, so video and audio line up
    fps = SR / HOP
    anim = bs.animate_bispectrum(
        f1,
        f2,
        B,
        times=times,
        cmap="RdBu_r",
        norm=norm,
        title=r"$B(f_1, f_2)$",
        interval=1000 / fps,
    )
    here = Path(__file__).parent
    silent = here / "animate_bispectrum_silent.mp4"
    writer = FFMpegWriter(fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    anim.save(silent, writer=writer, dpi=110)

    # mux the clip itself underneath (video timeline starts at t = 0, so does y)
    wav = here / "animate_bispectrum_audio.wav"
    sf.write(wav, y, SR)
    out = here / "animate_bispectrum.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-i",
            str(wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    silent.unlink()
    wav.unlink()
    print(f"wrote {out} ({B.shape[2]} frames at {fps:g} fps, {B.shape[2] / fps:.1f} s)")


if __name__ == "__main__":
    main()
