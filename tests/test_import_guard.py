"""The generic core must import and compute with numpy/scipy only: librosa must
never leak down into it. Run the core in a subprocess where importing librosa is
forcibly blocked.
"""

import subprocess
import sys
import textwrap


def test_core_works_without_librosa():
    code = textwrap.dedent(
        """
        import sys, importlib.abc, importlib.machinery

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "librosa" or name.startswith("librosa."):
                    raise ImportError("librosa is blocked for this test")
                return None

        sys.meta_path.insert(0, _Block())

        # core + modal math must work with no librosa importable
        import numpy as np
        from bispectrosa import bispectrum

        pairs = bispectrum.modal_index_pairs(12)
        assert len(pairs) == 49
        modes = bispectrum.legendre_modes(np.linspace(-1, 1, 201), 12)
        G = bispectrum.modal_gram_matrix(pairs, modes)
        assert G.shape == (49, 49)
        # ModalBispectrum beta estimation is pure numpy too
        rng = np.random.default_rng(0)
        Xstft = (rng.standard_normal((201, 20)) + 1j * rng.standard_normal((201, 20)))
        basis = bispectrum.ModalBispectrum(modes=modes.astype("float32"), pairs=pairs, n_irfft=400)
        beta = basis.estimate_beta(Xstft)
        assert beta.shape == (20, 49)

        assert "librosa" not in sys.modules
        print("OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
