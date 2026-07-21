# Theory: the modal bispectrum for audio

This page derives what `bispectrosa` computes.

## 1. Preprocessing: the STFT

Slice the waveform `y[n]` (sampling rate `sr`) into overlapping frames of length `N`
taken every `R` samples (defaults `N = 400`, `R = 160`: 25 ms windows, 10 ms hop at
16 kHz; the signal is zero-padded by `N/2` so frame `t` is centered on sample
`tR`), taper each frame with a Hann window, and take its DFT `X[k, t]`. The frame is real, so
only the `F = N/2 + 1` non-negative-frequency bins are kept:

$$
x_t[n] = y[tR + n - N/2], \qquad
w[n] = \tfrac{1}{2}\Big(1 - \cos\tfrac{2\pi n}{N}\Big),
\qquad n = 0, \dots, N-1,
$$

$$
X[k, t] = \sum_{n=0}^{N-1} w[n]\, x_t[n]\, e^{-i 2\pi k n / N},
\qquad k = 0, \dots, N/2 .
$$

Bin `k` sits at `f_k = k · sr / N` Hz (40 Hz spacing at the defaults). This complex
array is the **STFT** (`stft`), the single shared input to everything below.

Every later object is built frame by frame, so from here on we drop
the frame index `t` inside equations.

## 2. The second and third-order spectral moments

For a real, zero-mean signal, the **power spectrum** and the **bispectrum** are the
second- and third-order spectral moments: the ensemble average `⟨·⟩` of a product of
two, and of three, spectrum values:

$$
P(f) = \langle X(f)\, X^*(f)\rangle, \qquad
B(f_1, f_2) = \big\langle X(f_1)\, X(f_2)\, X^*(f_1+f_2)\big\rangle .
$$

In practice the ensemble average is estimated by averaging over the STFT frames in a
given time window, treating them as realizations of the same process.

The power spectrum treats every frequency separately: it carries no information about
how components relate to one another. The bispectrum is the correlation between three
components at once, those at `f₁`, `f₂`, and `f₁+f₂`: it averages to zero when they
fluctuate independently (any Gaussian signal, e.g., white noise)
and survives when they are correlated (nonlinearity).

## 3. Recovering the mel power spectrum

This section re-derives `librosa.feature.melspectrogram` in the present notation.

**Power.** Square the magnitude:

$$
S[k] = \lvert X[k] \rvert^2 .
$$

**The mel filterbank.** Pick the frequency range to cover, `[f_min, f_max]` (defaults
`0` to `sr/2`, i.e. all the bins), and `M` bands (default 80) whose centers are uniform
not in Hz but on the mel axis `m(f)`, a perceptual frequency scale (Slaney convention):
exactly linear below 1 kHz, `m(f) = 3f/200`, and exactly logarithmic above,
`m(f) = 15 + 27 ln(f/1000) / ln 6.4`. Place `M + 2` points uniformly between
`m(f_min)` and `m(f_max)`, map them back to Hz to get band edges `f̂_b`, and build one
triangle per band, sampled at the bin frequencies `f_k = k · sr/N` of section 1 (the
translation between bin index and Hz throughout):

$$
H[b, k] = \frac{2}{\hat f_{b+2} - \hat f_b}\;
\max\!\Bigg(0,\ \min\!\Big(\frac{f_k - \hat f_b}{\hat f_{b+1} - \hat f_b},\;
\frac{\hat f_{b+2} - f_k}{\hat f_{b+2} - \hat f_{b+1}}\Big)\Bigg).
$$

The prefactor is the Slaney area normalization: every triangle has unit area in Hz, so
wide high-frequency bands integrate the spectrum with the same total weight as narrow
low-frequency ones (`mel_filterbank`, `triangular_filterbank`).

**Mel pooling.** Pool the bin powers through the filterbank:

$$
S_{\mathrm{mel}}[b] = \sum_{k} H[b, k]\, S[k] .
$$

Stacked over frames, this is exactly librosa's mel spectrogram at `power=2`; in
decibels, `10 log₁₀ S_mel`, it is the standard **log-mel** feature (`mel_spectrogram`).


## 4. The bispectrum estimator

### The direct estimator

The estimator is the triple product on every pair whose third bin exists:

$$
B[k_1, k_2] =
\mathrm{Re}\big(X[k_1]\, X[k_2]\, X^*[k_1{+}k_2]\big),
\qquad k_1 + k_2 < F,
$$

where `F = N/2 + 1` is the number of STFT bins of section 1 (it keeps the third
component below the Nyquist frequency).

`raw_bispectrum` computes this on the transform's own bin grid, averaged over all
frames (or kept as a per-frame stack with `average=False`), over the half
`k₂ ≤ k₁` (the bispectrum is symmetric) with the third factor read at exactly the
sum frequency `k₁ + k₂`, so every sampled triplet is valid. An optional window
`kmin`/`kmax` restricts all three legs to `[kmin, kmax]` (triplets whose sum
frequency leaves the window are simply excluded; in the square `return_full=True`
form their cells are NaN), the same band-limit convention the modal estimator
applies through its mode support.

### Issues with this direct estimator

Two issues, the same two that motivate modal methods in cosmology.

- **Size.** At the defaults `F = 201`, there are ~`F²/2 ≈ 2·10⁴` valid bin pairs
  (~`10⁴` independent ones by symmetry). That is a large 2-D object per frame (to
  compare to the 80 numbers of the mel power spectrum), and computing it costs
  `O(F²)` per frame (measured, around 30× the mel power spectrogram cost at the
  package defaults; the gap grows quadratically with frequency resolution).
- **Variance.** Each cell is a single triple product per frame, so the raw matrix is
  extremely noisy; coupling only emerges after long frame averaging, cell by cell.

More efficient ways to extract the bispectrum information have been introduced to
study cosmological signals (see [Planck 2013 XXIV](https://arxiv.org/abs/1303.5084)
and references therein).

## 5. The modal estimator

We use the **modal** estimator of
[Fergusson, Liguori & Shellard 2010](https://arxiv.org/abs/0912.5516), which consists
of expanding the bispectrum on a well-chosen basis of simple functions:

$$
B[k_1, k_2] = \sum_{n, m} (\Gamma^{-1})_{nm}\, \beta_{m}\, Q_{n}[k_1, k_2] .
$$

`Γ` is the **Gram matrix** of the basis, `Γ_nm = ⟨Q_n, Q_m⟩` with
`⟨A, C⟩ = Σ_{k₁+k₂<F} A[k₁,k₂] C[k₁,k₂]` the sum of a product over the frequency domain:
it corrects for the mutual overlaps of the basis functions, which do not need to be
orthogonal.

The trick is then to use 2-D basis functions `Q_n` that are built separably from a family of
smooth 1-D modes `q_0, …, q_D` sampled on the `F` bins; the index `n`
runs over the pairs of mode orders as defined below.

### Pair kernels and coefficients

Each pair `n=(p, r)` with `p ≤ r` (the kernel is symmetric in `p, r`) defines its
kernel on the `(k₁, k₂)` plane by placing the pair's two modes on two of the
three legs `(k₁, k₂, k₁+k₂)`, leaving the remaining leg unweighted (a constant
factor `1`), and symmetrizing the assignment over the legs:

$$
Q_{pr} = \tfrac{1}{6}\big(
q_p[k_1] q_r[k_2] + q_r[k_1] q_p[k_2]
+ q_p[k_1] q_r[k_1+k_2] + q_r[k_1] q_p[k_1+k_2]
+ q_p[k_2] q_r[k_1+k_2] + q_r[k_2] q_p[k_1+k_2]
\big).
$$

### The fast separable form

Because the kernel is a sum of separable products, `β` is computed directly from the
STFT, without ever forming the `O(F²)` bispectrum grid.
Weighting the spectrum by one mode and transforming back to the time domain gives a
filtered copy of the frame,

$$
z_p[u] = \frac{1}{N} \sum_{k} q_p[k]\, X[k]\, e^{i 2\pi k u / N}
\qquad (\texttt{irfft}),
$$

with `u` the sample index inside the frame, as in section 1, and the sum running over
all `N` DFT bins, the upper half filled by Hermitian symmetry (`X[N−k] = X*[k]`,
`q_p[N−k] = q_p[k]`) so that `z_p` is real: this is exactly `np.fft.irfft`. The
unweighted third leg contributes the same transform with `q` replaced by the
analysis window's indicator (`1` on the kept bins, `0` outside; the default keeps
every bin except `k = 0`): `z_c` is the band-limited frame itself. The
coefficient is then a plain product of three filtered frames, summed over the samples:

$$
\beta_{pr} = \sum_{u=0}^{N-1} z_p[u]\, z_r[u]\, z_c[u] .
$$

Substituting the first equation into the second, each triplet of bins `(k₁, k₂, k₃)`
enters with the factor `Σ_u e^{i2π(k₁+k₂−k₃)u/N}`, which vanishes unless
`k₁ + k₂ ≡ k₃ (mod N)`: the sample sum picks out exactly those triplets and equals,
up to a fixed constant, the projection `⟨Q_pr, B⟩` of the bispectrum onto the kernel.
This form is the definition `ModalBispectrum.estimate_beta` computes. The shipped feature keeps one column per frame, like a
spectrogram (`mel_bispectrogram` returns `(n_coeffs, n_frames)`).

One can then reconstruct the bispectrum through the first equation of this section
(`reconstruct_bispectrum`, validated against `raw_bispectrum`), but it is really the
`β` coefficients we are interested in: they carry the bispectrum information in
compressed form.

### Truncation and nesting

Truncation is by total degree, `p + r ≤ D` (the smoothness prior: forbid two
high-order modes at once), giving

$$
N_{\text{coeffs}} = \Big\lfloor \tfrac{(D+2)^2}{4} \Big\rfloor
$$

coefficients (`modal_pair_dim`): 20 at `D = 7`, 49 at `D = 12`. The pair list is
**degree-ordered** (`modal_index_pairs`), so a low-degree feature is exactly the
length-`modal_pair_dim(d)` prefix of a higher-degree one: store at high `D`, read any
lower degree by slicing.


### Conditioning: signed-log

Raw coefficients are heavy-tailed (a triple product spans many orders of magnitude), so
each is passed through a **signed-log**:

$$
\mathrm{slog}(x) = \operatorname{sign}(x)\,\log\!\big(1 + |x|/\epsilon\big),
\qquad \epsilon = 10^{-15}.
$$

It is monotonic and invertible (no information lost, the same role log plays in
log-mel), and the sign is kept because the projection of a real bispectrum has a
meaningful sign (the direction of the coupling).

## 6. Choosing the modes: where mel enters

### The raw Legendre reference

Map each bin to its frequency rescaled to `[-1, 1]` and evaluate Legendre polynomials
there (`P_p`, playing the role the DCT cosines play in MFCCs):

$$
q_p[k] = P_p\big(2 f_k / f_{\max} - 1\big)
$$

(`legendre_modes`; here `f_max = sr/2`). Because `f₁ + f₂` is linear in Hz, a polynomial evaluated there is
still a polynomial in `(f₁, f₂)`: everything collapses cleanly, which makes this family
the validation reference. As a feature basis it is weak: resolution is uniform in Hz,
so much of the degree budget is spent on the sparse high frequencies.

### Mel-binned modes

The mel Legendre family pushes the mel filterbank of section 3 into the modes: evaluate
Legendre on the band index `b` and smear it back onto the bins,

$$
\tilde q_s[k] = \sum_{b} P_s\Big(\tfrac{2b}{M-1} - 1\Big)\, H[b, k],
\qquad s = 0, \dots, D
$$

(`mel_legendre_modes`, `mel_legendre_modal_bispectrum`). Every `β_pr` then places the
mel-binned modes of orders `p` and `r` on two of the three legs, the third
unweighted.

Compared to raw Legendre, the mel modes concentrate resolution below ~2 kHz (where
speech puts its coupling structure), give a much better conditioned Gram matrix, and
average bin-level noise inside each band before the expansion.

## References

The modal estimator follows the separable-basis bispectrum method developed for
cosmological non-Gaussianity; `bispectrosa` adapts its 2-index audio specialization.

- Fergusson, Liguori & Shellard, *General CMB and primordial bispectrum estimation:
  mode expansion, map-making and measures of f_NL*, Phys. Rev. D 82, 023502 (2010),
  [arXiv:0912.5516](https://arxiv.org/abs/0912.5516).
- Planck Collaboration, *Planck 2013 results. XXIV. Constraints on primordial
  non-Gaussianity*, A&A 571, A24 (2014),
  [arXiv:1303.5084](https://arxiv.org/abs/1303.5084).

To cite this software, see [`CITATION.cff`](../CITATION.cff).
