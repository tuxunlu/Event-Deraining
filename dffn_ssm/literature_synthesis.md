# Evidence Base: Dynamic Fourier Filtering × State Space Models for Event-Camera Deraining

*Synthesis of 5 upstream literature surveys + 5 adversarial novelty audits. Verified live this session: arXiv:2211.14453, arXiv:2505.05307, arXiv:2604.14724. WebSearch budget was exhausted (200/200) before this agent ran; everything else is inherited from upstream agents at their stated confidence, and confidence is propagated honestly in §7.*

---

## 1. Confirmed prior art table

| # | Title | Venue / Year | arXiv / URL | What it gives us |
|---|---|---|---|---|
| 1 | Analysis of Rain and Snow in Frequency Space — Barnum, Narasimhan, Kanade | IJCV 86(2-3):256–274, 2010 | [10.1007/s11263-008-0200-2](https://link.springer.com/article/10.1007/s11263-008-0200-2) | The only closed-form spectrum of rain; the oriented ridge-through-DC prior our whole design rests on |
| 2 | Vision and Rain — Garg, Nayar | IJCV 2007 | [10.1007/s11263-006-0028-6](https://link.springer.com/article/10.1007/s11263-006-0028-6) | Streak breadth `b` and length `l` as functions of drop size/depth — the parameters that set the spectral envelope |
| 3 | Rain rendering for evaluating robustness to bad weather — Tremblay et al. | IJCV 2021 | [arXiv:2009.03683](https://arxiv.org/abs/2009.03683) | Origin of the "KITTI at N mm/hr" convention; 50 and 150 are grid points of this simulator, not arbitrary |
| 4 | **PRE-Mamba: A 4D SSM for Ultra-High-Frequent Event Camera Deraining** — Ruan et al. | **ICCV 2025** | [arXiv:2505.05307](https://arxiv.org/abs/2505.05307) | **The benchmark holder**: EventRain-27K, the SR/NR/DA metric definitions, and the only published numbers on our exact task |
| 5 | EDmamba: Efficient Event Denoising with Spatiotemporal Decoupled SSMs — Ruan et al. | arXiv May 2025 (preprint) | [arXiv:2505.05391](https://arxiv.org/abs/2505.05391) | Our real efficiency rival: 88.98K params / 2.27 GFLOPs; proves axis-decoupled scanning beats joint scanning |
| 6 | EDformer: Transformer-Based Event Denoising Across Varied Noise Levels — Jiang et al. | ECCV 2024 | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03905.pdf) | Strongest non-SSM point-based baseline; shows transformers collapse as rain density rises |
| 7 | EPM / EDnCNN — Baldwin, Almatrafi, Asari, Hirakawa | CVPR 2020 | [arXiv:2003.08282](https://arxiv.org/abs/2003.08282) | Founding learned denoiser; the CNN baseline our DFFN must beat at 1/8500 the params |
| 8 | AEDNet — Fang, Wu, Li et al. | ACM MM 2022 | [10.1145/3503161.3548048](https://dl.acm.org/doi/10.1145/3503161.3548048) | Cautionary baseline: 45.87M params, NR collapses 0.732→0.547 as rain thickens |
| 9 | Distill Drops into Data (voxel event deraining) — Ruan et al. | ACM MobiCom 2024 | [10.1145/3636534.3694737](https://doi.org/10.1145/3636534.3694737) | The precedent for voxel/frame event deraining — and the reason PRE-Mamba excludes voxel methods from SR/NR/DA |
| 10 | **FourierMamba: Fourier Learning Integration with SSMs for Image Deraining** — Li, Liu, Fu, Xu, Zha | **ICML 2025** (PMLR v267) | [arXiv:2405.19450](https://arxiv.org/abs/2405.19450) | **Model (A)'s source**; owns "selective scan over ordered frequency bins, amplitude and phase separately" |
| 11 | FreqMamba — Zou, Hu, Feng | ACM MM 2024 | [arXiv:2404.09476](https://arxiv.org/abs/2404.09476) | Triple-branch spatial-Mamba + band-Mamba + Fourier; the "Mamba alone can't see global frequency degradation" argument |
| 12 | Image Deraining with Frequency-Enhanced SSM (DFSSM) — Yamashita, Ikehara | ACCV 2024 | [arXiv:2405.16470](https://arxiv.org/abs/2405.16470) | The "SSM stays spatial, FFT is an additive side branch" baseline; states rain = high-intensity components in specific directions |
| 13 | DeRainMamba — Zhu, Zeng, Yang, Luo, Zeng | IEEE SPL 2025 | [arXiv:2510.06746](https://arxiv.org/abs/2510.06746) | The dissent on "degradation is amplitude-only"; argues rain distorts phase too; anisotropic MDPConv |
| 14 | **SFHformer: When FFT Meets Transformer for Image Restoration** — Jiang, Zhang, Gao, Deng | ECCV 2024 | [PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06190.pdf) | **Closest published relative of the DFFN block**: depthwise conv *across* frequency bins + per-bin softmax dynamic conv, on **real/imag** not magnitude/phase |
| 15 | Fourmer: Efficient Global Modeling for Image Restoration — Zhou, Huang, Guo, Li | ICML 2023 | [PMLR v202](https://proceedings.mlr.press/v202/zhou23f/zhou23f.pdf) | The lightweight-Fourier-deraining ancestor; amplitude/phase swap experiment; pointwise-in-frequency is the cheap default |
| 16 | Fast Fourier Convolution (FFC) — Chi, Jiang, Mu | NeurIPS 2020 | [proceedings](https://proceedings.neurips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html) | Origin of local-spatial + global-spectral two-path design; concatenates real/imag over channels |
| 17 | Global Filter Networks (GFNet) — Rao, Zhao, Zhu, Lu, Zhou | NeurIPS 2021 / TPAMI 2023 | [arXiv:2107.00645](https://arxiv.org/abs/2107.00645) | The LTI reference filter that *can* notch and amplify; the resolution-dependence counterexample |
| 18 | AFNO: Adaptive Fourier Neural Operators — Guibas, Mardani et al. | ICLR 2022 | [arXiv:2111.13587](https://arxiv.org/abs/2111.13587) | Soft-thresholding/shrinkage on modes = the cheapest published **learned notch**; resolution-independent by weight sharing |
| 19 | Intriguing Findings of Frequency Selection for Deblurring (Res FFT-ReLU) — Mao et al. | AAAI 2023 | [10.1609/aaai.v37i2.25281](https://ojs.aaai.org/index.php/AAAI/article/view/25281) | A ReLU with learned threshold *in the spectrum* — a one-line, ablated, near-free notch mechanism |
| 20 | **Transform Once (T1): Efficient Operator Learning in Frequency Domain** — Poli, Massaroli, Berto, Park, Dao, Ré, Ermon | **NeurIPS 2022** | [arXiv:2211.14453](https://arxiv.org/abs/2211.14453) | **Kills the single-FFT-trunk claim**: one transform, N frequency-resident layers, variance-preserving init, 3×–10× speedup |
| 21 | FlashFFTConv — Fu, Kumbong, Nguyen, Ré | ICLR 2024 | [arXiv:2311.05908](https://arxiv.org/abs/2311.05908) | Fuses FFT→multiply→iFFT into one kernel (7.93× exact-FFT speedup) — the cheap alternative to a frequency-resident trunk |
| 22 | State-Free Inference of SSMs: Transfer Function Approach (RTF) — Parnichkun et al. | ICML 2024 | [arXiv:2405.06147](https://arxiv.org/abs/2405.06147) | `H(z)=h₀+C(zI−A)⁻¹B` in explicit rational form; SSM spectrum from one FFT of two short coefficient vectors |
| 23 | S4D: Parameterization and Initialization of Diagonal SSMs — Gu, Gupta, Goel, Ré | NeurIPS 2022 | [arXiv:2206.11893](https://arxiv.org/abs/2206.11893) | `K_l = Σ_n C_n λ_n^l B_n` — a diagonal SSM *is* a parallel bank of one-pole IIR filters |
| 24 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu, Dao | arXiv Dec 2023 | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | The cost model we must budget against: 3ED² projection params, O(BLDN) scan FLOPs |
| 25 | Transformers are SSMs (Mamba-2 / SSD) — Dao, Gu | ICML 2024 | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) | Chunkwise matmul form: 2–8× faster than S6, and chunked training ≡ frame-by-frame streaming with identical weights |
| 26 | Hydra: Bidirectional SSMs via Generalized Matrix Mixers — Hwang, Lahoti, Dao, Gu | NeurIPS 2024 | [arXiv:2407.09941](https://arxiv.org/abs/2407.09941) | The non-causal quasiseparable mixer needed for a frequency axis, on the Mamba-2 kernel |
| 27 | Were RNNs All We Needed? (minGRU/minLSTM) — Feng, Tung, Ahmed, Bengio, Hajimirsadeghi | arXiv Oct 2024 | [arXiv:2410.01201](https://arxiv.org/pdf/2410.01201) | The cheapest content-dependent global mixer (~2 MACs/channel/step) — the right floor for a 72K-param budget |
| 28 | Tuning Frequency Bias of State Space Models — Yu, Lyu, Lim, Mahoney, Erichson | ICLR 2025 | [arXiv:2410.02035](https://arxiv.org/abs/2410.02035) | SSMs have an innate low-frequency bias fixed at init that training does not remove — a direct risk to rain (high-freq) |
| 29 | **State Space Models for Event Cameras** — Zubić, Gehrig, Scaramuzza | **CVPR 2024** (Spotlight) | [arXiv:2402.15584](https://arxiv.org/abs/2402.15584) | Train-as-CNN / infer-as-RNN with state saved across windows; timescale rescaling for rate transfer; SSM transfer-function band-limiting |
| 30 | STREAM: Universal SSM for Sparse Geometric Data — Schöne et al. | arXiv Nov 2024 | [arXiv:2411.12603](https://arxiv.org/pdf/2411.12603) | `Δᵢ = (tᵢ−tᵢ₋₁)·softplus(δ)` — Δ set from **physical inter-event time**, explicitly contrasted with learned/input-dependent Δ |

**Runners-up cited throughout §2/§4** (not in the top-30 table but load-bearing): SFANet/WFCA [arXiv:2302.13598](https://arxiv.org/abs/2302.13598) (frequency-resolution mismatch); LoFormer [arXiv:2407.16993](https://arxiv.org/abs/2407.16993) (windowed spectral attention); DFFormer [arXiv:2303.03932](https://arxiv.org/abs/2303.03932) (dynamic filter as token mixer); AFF [arXiv:2307.14008](https://arxiv.org/abs/2307.14008); FDConv CVPR 2025 [arXiv:2503.18783](https://arxiv.org/abs/2503.18783); DKN [arXiv:1910.08373](https://arxiv.org/abs/1910.08373); DCNv4 [arXiv:2401.06197](https://arxiv.org/html/2401.06197v1); DDF [arXiv:2104.14107](https://arxiv.org/abs/2104.14107); KernelWarehouse [arXiv:2406.07879](https://arxiv.org/abs/2406.07879); CARAFE [arXiv:1905.02188](https://arxiv.org/abs/1905.02188); AHFNet WACV 2025 [arXiv:2412.01559](https://arxiv.org/abs/2412.01559); BPN [arXiv:1912.04421](https://arxiv.org/abs/1912.04421); Anti-Oversmoothing ICLR 2022 [arXiv:2203.05962](https://arxiv.org/abs/2203.05962); Ai & Ling anti-wrapping [arXiv:2211.15974](https://arxiv.org/abs/2211.15974); Meta-AF [arXiv:2204.11942](https://arxiv.org/abs/2204.11942); TF-GridNet [arXiv:2211.12433](https://arxiv.org/abs/2211.12433); HAMSA [arXiv:2604.14724](https://arxiv.org/abs/2604.14724); PAS-Mamba [arXiv:2601.14530](https://arxiv.org/abs/2601.14530); PRISMamba [arXiv:2602.04170](https://arxiv.org/abs/2602.04170); eFFT [IEEE TPAMI 46(12):9630-9647](https://ieeexplore.ieee.org/document/10582443/); RAKI [MRM 81(1):439-453](https://onlinelibrary.wiley.com/doi/10.1002/mrm.27420).

---

## 2. Settled technical facts (design constraints)

### 2.1 Rain's Fourier signature — what the spectrum actually looks like

**F1. Rain is an oriented ridge through DC, not a high-frequency shell.** Barnum et al. give the only closed form: a streak = circular Gaussian of breadth `b` motion-blurred over length `l` at orientation θ, with

```
G(u,v;b,l) = i·exp(−b²(u²+v²))·(1−exp(2πi·l·v))/(2πv)   for v ≠ 0
           = exp(−b²u²)·l                                 for v = 0
```

Full field = `α · Σ_θ ∫_{a=0.1mm}^{3mm} ∫_{z=0.5m}^{10m} G(u,v; b(a,z), l(a,z), θ) dz da`, with α set by rainfall rate and θ spread ≈ ±0.1 rad. [IJCV 2010](https://link.springer.com/article/10.1007/s11263-008-0200-2)

**Design consequences:**
- The envelope is the **isotropic Gaussian `exp(−b²(u²+v²))`**, so the cutoff is set by *drop breadth* (1–3 px), not by "rain is high-frequency". Barnum explicitly notes rain "covers such a broad part of the frequency space". **Any design that assumes rain lives in a thin high-frequency shell is contradicted by the physics.**
- Energy peaks on the line `v=0` through the origin → an elongated ridge/wedge through DC, oriented perpendicular to the spatial streak. (The perpendicularity is the upstream agent's reading of Eq. 5, not a sentence Barnum writes — cite the equation, not the paraphrase.)
- Along the streak-parallel frequency axis the magnitude falls as **1/|v| with ripples of period 1/l**.
- **The model is constant in temporal frequency ω** — rain is flat/white along time. So a single event time-window's 2D spectrum shows the ridge, but *no temporal separability is available within a window*. The cross-window discriminative signal is the **persistence of ridge orientation**, not a temporal spectrum. This is the precise physical justification for a temporal state, and the precise reason a temporal *FFT* would buy nothing.

**F2. Modern restatements agree qualitatively and disagree on phase.** DFSSM: "rain streaks produce high-intensity frequency components in specific directions" ([arXiv:2405.16470](https://arxiv.org/abs/2405.16470)). FourierMamba: degradation "mainly in amplitude components" ([arXiv:2405.19450](https://arxiv.org/abs/2405.19450)). DeRainMamba **dissents**: rain distorts *both* amplitude and phase, worst in high-frequency regions, and ignoring phase gives ghosting/edge blurring ([arXiv:2510.06746](https://arxiv.org/abs/2510.06746)). See §6.

**F3. This is unverified for event tensors.** Barnum's model is for motion-blurred *integrated intensity*. Polarity, per-pixel contrast thresholds and refractory period change what an event voxel's 2D spectrum looks like. **Plot the measured rain-only event spectrum in polar coordinates before betting an architecture on the wedge.**

### 2.2 What a diagonal SSM is, as a filter

**F4. A diagonal LTI SSM is an N-pole/N-zero rational (ARMA/IIR) filter.**
`H(z) = D + C(zI − A)⁻¹B`; with `A = diag(λ₁…λ_N)`:
`H(z) = D + Σ_n C_n B_n z⁻¹/(1 − λ_n z⁻¹)` — a parallel bank of one-pole IIR filters. Equivalently `K_l = Σ_n C_n λ_n^l B_n` (S4D, [arXiv:2206.11893](https://arxiv.org/abs/2206.11893)). Canonical rational form, Eq. 4 of RTF: `H(z) = h₀ + (b₁z⁻¹+…+b_nz⁻ⁿ)/(1+a₁z⁻¹+…+a_nz⁻ⁿ)`, params `2n+1` vs `n²+2n+1` dense ([arXiv:2405.06147](https://arxiv.org/abs/2405.06147)).

**F5. It can notch AND amplify — but only with the right order.** Poles near the unit circle give `|H|≫1` (boost); numerator zeros *on* the unit circle give `|H(e^{jω₀})|=0` (a true notch at a learnable frequency). **Caveat flagged by the adversarial audit and worth heeding:** with `N=1` and `D=0` there is one pole and **no finite zero** — a pure one-pole response that *cannot* null a bin. You need `N≥2` with complex λ, or a non-zero `D` skip, to place a transmission zero. Write "poles plus the D-skip give a rational response with zeros", not "a diagonal SSM can notch".

**F6. Selective (S6) SSMs have no single transfer function.** Mamba's input-dependent `A_bar, B, C` make the system linear-time-*varying*; the correct object is the 1-semiseparable / quasiseparable matrix view (Mamba-2, [arXiv:2405.21060](https://arxiv.org/abs/2405.21060); generalized to arbitrary diagonal A in [arXiv:2510.04944](https://arxiv.org/abs/2510.04944)). Per-position it is still a rational filter — the poles just move with content.

**F7. Diagonal SSMs have an innate low-frequency bias fixed at initialization that ordinary training does not change** ([Tuning Frequency Bias of SSMs, ICLR 2025](https://arxiv.org/abs/2410.02035); [Uncovering Spectral Bias in Diagonal SSMs, NeurIPS 2025, arXiv:2508.20441](https://arxiv.org/abs/2508.20441)). Fixes: scale the initialization, apply a Sobolev-norm gradient reweighting, or use S4D-DFouT (a diagonal init defined directly on the DFT domain). **Rain is high-frequency and anisotropic — this is a first-order risk, not a footnote.**

### 2.3 What a softmax-over-taps kernel can and cannot do

**F8. A convex-combination kernel is provably a non-expansive, unit-DC-gain smoother.** If `h_k ≥ 0` and `Σh_k = 1`, then for all ω, `|H(e^{jω})| = |Σ_k h_k e^{−jωk}| ≤ Σ_k h_k = 1 = H(e^{j0})`. Therefore: DC gain is pinned at exactly 1; no frequency can be amplified above DC; the filter is structurally a moving average; a notch can occur only by accidental phase cancellation at an isolated ω and cannot be placed independently of the passband. *(This is elementary DSP, reported as a derivation — no paper states it in this form. See §7.)*

**F9. The literature states the same fact from three directions:**
- CARAFE (positive framing): "the normalization step forces the sum of kernel values to 1… CARAFE does not perform any rescaling and change the mean values of the feature map" ([arXiv:1905.02188](https://arxiv.org/abs/1905.02188)).
- DKN (rejection): softmax "encourages the estimated kernel to have only a few non-zero elements, which is not appropriate for image filtering. The estimated kernels should be similar to high-pass filters, with kernel weights adding to 0" ([arXiv:1910.08373](https://arxiv.org/abs/1910.08373)).
- AHFNet Prop. 1 (structural dual, WACV 2025): any **non-negative** linear combination of high-pass filters remains high-pass — read contrapositively, a non-negative combination of shifted deltas stays in the non-negative unit-DC-gain cone ([arXiv:2412.01559](https://arxiv.org/abs/2412.01559)).
- Formal spectral theory: a row-stochastic mixing operator "inherently amounts to a low-pass filter" ([Anti-Oversmoothing, ICLR 2022, arXiv:2203.05962](https://arxiv.org/abs/2203.05962)); "MSAs are low-pass filters, but Convs are high-pass filters" ([arXiv:2202.06709](https://arxiv.org/abs/2202.06709)).

**F10. Quantified cost of the softmax, from controlled ablations:**

| Evidence | Number |
|---|---|
| DCNv4: adding softmax to ConvNeXt depthwise conv | **−21.4% top-1 at epoch 5**, never recovers ([arXiv:2401.06197](https://arxiv.org/html/2401.06197v1)) |
| Involution: attaching softmax/sigmoid to the kernel generator | **>1% top-1 drop** ([arXiv:2103.06255](https://arxiv.org/abs/2103.06255)) |
| ODConv: Sigmoid vs Softmax on the per-tap **spatial** attention (ResNet18) | 73.41% vs 73.23% ([arXiv:2209.07947](https://arxiv.org/abs/2209.07947)) |
| CondConv Table 3 | "the baseline's Sigmoid significantly outperforms Softmax" ([arXiv:1904.04971](https://arxiv.org/abs/1904.04971)) |
| SFHformer: replacing dynamic FDC with a **static** pointwise conv | −0.11 dB; removing local frequency conv FCPE −0.05 dB (Table 8) |
| DyConv temperature annealing τ: 30→1 over 10 epochs | 69.4% → **69.9%** ([arXiv:1912.03458](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Dynamic_Convolution_Attention_Over_Convolution_Kernels_CVPR_2020_paper.html)) |

**F11. Softmax is buying you stability, and the field has published four better trades.** Ranked drop-ins for DFFN's 9 taps:
1. **Mean-subtraction → sum-to-zero, signed taps** + residual (DKN, IJCV 2021) — the closest to "make it a high-pass".
2. **BN-style filter normalization** `D = α(Ď−μ(Ď))/δ(Ď) + β` (DDF, CVPR 2021) — bounded, zero-mean, signed, explicitly motivated by gradient stability.
3. **Signed L1-of-logits (KernelWarehouse CAF)** `α_ij = τβ_ij + (1−τ)z_ij/Σ_p|z_ip|`, which "enables both negative and positive attentions" (ICML 2024).
4. **Sigmoid gain, not sum-to-one** (CondConv, ODConv spatial, DCNv2 modulation).
5. **Softmax over kernel BASES, not taps** — SFHformer's FDC and DFFormer both do this; the effective kernel stays sign-unconstrained. *This is the single cheapest change that preserves DFFN's exact structure.*
6. **Temperature annealing** — a 2-line experiment; run it before any redesign.

**F12. Nobody has published what a K×K convolution over the *spectrum* means in the spatial domain — and it is an apodization, not a filter.** By the dual of the convolution theorem, convolving the spectrum with kernel `k` is exactly multiplying the image by `iDFT(k)`. For a 3×3 kernel over offsets `(du,dv)∈{−1,0,1}²`:

```
m(x,y) = Σ_{du,dv} w_{du,dv} · exp(j2π(du·x/H + dv·y/W))
```

— a linear combination of the nine lowest non-zero 2D complex exponentials, i.e. an image-wide, one-cycle-across-the-frame smooth complex gain. With softmax taps (`w≥0`, `Σw=1`) additionally `|m(x,y)| ≤ 1` with equality only at the spatial origin. The textbook identity confirms it: applying a **Hann window** in the signal domain is implemented *exactly* by convolving the DFT with `[1/4, 1/2, 1/4]`, and "any L-term Blackman-Harris window requires convolution of the critically sampled spectrum with a smoother of length 2L−1" ([J.O. Smith, *Spectral Audio Signal Processing*](https://www.dsprelated.com/freebooks/sasp/Spectrum_Analysis_Windows.html)). `[1/4,1/2,1/4]` is itself non-negative and sums to 1.

**⇒ DFFN's spectral 3×3 is provably learning a per-channel 2D generalized Hann-family window / vignetting applied to the image. It is spectral leakage control, not deraining.** This is unclaimed analysis and is the strongest theory contribution available (MRI has the analysis — SENSE/GRAPPA duality, RAKI [MRM 2019](https://onlinelibrary.wiley.com/doi/10.1002/mrm.27420) — vision does not).

### 2.4 FFT-based long convolution and what it means for a model already in the spectrum

**F13. All FFT-conv methods materialize a length-L kernel then compute `y = iFFT(FFT(K)⊙FFT(u))` in O(L log L).** S4 constructs K in O((L+N)log²(L+N)) via a Cauchy/DPLR trick; S4D reduces it to a Vandermonde matmul; Hyena emits K from an MLP over positional encodings; FlashFFTConv maps the FFT onto tensor cores via a Monarch decomposition and **fuses FFT→multiply→iFFT into one kernel so the spectrum never round-trips to HBM** (7.93× exact-FFT speedup, 4.4× end-to-end) ([arXiv:2311.05908](https://arxiv.org/abs/2311.05908)).

**F14. If you already hold `rfft2(x)`, an LTI SSM along that axis is ONE complex multiply per bin — zero extra transforms.** RTF makes this concrete: Eq. 15, `H_l(z^t) = FFT_l(b̃)_t / FFT_l(a)_t + h₀` — the DFT of the SSM kernel obtained by FFT-ing two short coefficient vectors of length n ≪ L and dividing. O(l log l) time, **O(l) space independent of state size**, 35% training speedup over S4 on LRA ([arXiv:2405.06147](https://arxiv.org/abs/2405.06147)).

**F15. FlashFFTConv's "partial" and "frequency-sparse" convolutions are implemented by simply not computing some frequency bins.** For a deraining model where rain occupies a known band, this is a published FLOP-reduction mechanism.

**F16. Hyena's long filters are input-INDEPENDENT** — only the gates are content-dependent ([arXiv:2302.10866](https://arxiv.org/abs/2302.10866)). If you want a content-dependent frequency response, the SSM route is the one, not Hyena.

### 2.5 Cost model — what you can afford at 72K params / 5.36 GFLOPs / 10.8 ms

**F17. Mamba's cost is dominated by projections, not by the scan.** Verbatim from [arXiv:2312.00752](https://arxiv.org/html/2312.00752v2): "most of the parameters (3ED²) are in the linear projections"; "the naive recurrent computation uses O(BLDN) FLOPs"; defaults E=2, N=16.

Derived (arithmetic on those quoted facts, not a citation):

| D (d_model) | Mamba block params (~6D²+…) | Proj MACs/token (3ED²) | Scan MACs/token (~2EDN) | Scan share |
|---|---|---|---|---|
| 32 | ~6.1K | 6,144 | 2,048 | ~25% |
| 64 | ~24.6K | 24,576 | 4,096 | ~14% |
| 128 | ~98K | 98,304 | 8,192 | ~8% |
| 256 | ~393K | 393,216 | 16,384 | ~4% |

**⇒ A D=64, E=2 Mamba block is already the entire DFFN parameter budget, and the projections — not the SSM — are what blow it. A bare N=8–16 diagonal scan is ~2DN ≈ 1024 MACs/token, the same order as DFFN's existing 18 MACs/channel/bin spectral conv, but with an infinite receptive field along the scanned axis. Strip the 6D² projection stack and keep the recurrence.**

**F18. Ranked cheapest content-dependent global 1-D mixers:** (1) minGRU/minLSTM — effectively N=1 selective diagonal SSM, ~2 MACs/channel/step, natively streaming ([arXiv:2410.01201](https://arxiv.org/pdf/2410.01201)); (2) diagonal selective scan with N∈[2,8]; (3) **Mamba-2/SSD — best FLOP-to-wallclock because it is matmuls, "2–8× faster" than the S6 scan** ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)); (4) GLA/FlashLinearAttention ([arXiv:2312.06635](https://arxiv.org/abs/2312.06635v6)); (5) LRU as the *content-independent ablation floor* ([arXiv:2303.06349](https://arxiv.org/pdf/2303.06349)).

**F19. THE LATENCY TRAP.** A selective scan is **memory-bandwidth bound, not FLOP bound**. At 260×346 the rfft2 half-spectrum is ~45K bins/channel; a bidirectional scan over amplitude *and* phase is ≥4 scans per block. Our own FourierMamba2D port already measures **153 ms vs DFFN's 10.8 ms**. Shrinking the generator head from `C·K²·2` to `C+2N` output channels cuts parameters and a depthwise conv's FLOPs — **both already negligible** — and does not touch sequential memory traffic. Unless you use SSD chunked matmul or a d_state=1 minGRU-class recurrence, **expect a large latency regression.** Budget this before committing.

### 2.6 Global vs local spectral operations — the empirical verdict

**F20. Local ≈ global alone; the win is having BOTH.** The only clean apples-to-apples ablation found is GLFNet's on ACDC (medical segmentation, [arXiv:2403.00396](https://arxiv.org/html/2403.00396v1)): global-only **92.88** Dice, local-only **92.82**, global+global 93.01, local+local 92.99, **global+local 93.12**. Corroborated by SFHformer's restoration ablation: hybrid spatial+frequency **41.85 dB** vs frequency-only 41.71 vs spatial-only 41.62.

**⇒ The literature's direct answer to DFFN limitation (1): do NOT make the filter generator global by brute force. Add a second, cheap global path (SSM sweep over the spectrum, or a GAP→MLP conditioning vector à la DFFormer) and KEEP the local 3×3 as the fine-grained path.** Corroborating warning from operator learning: FNO-style global spectral convolution is "prone to over-smoothing and may fail to capture local details"; adding local integral/differential kernels back cut relative L2 error by **34–72%** ([Neural Operators with Localized Integral and Differential Kernels, ICML 2024, arXiv:2402.16845](https://arxiv.org/abs/2402.16845)).

**F21. Full-spectrum weight tensors are resolution-fragile.** SFANet's "frequency resolution mismatch": a full-image FFT gives different bin spacing at train-crop size vs test-image size, so a learned per-bin weight tensor is inconsistent between train and inference ([arXiv:2302.13598](https://arxiv.org/abs/2302.13598)). GFNet's own remedy (interpolate the filter) concedes the point. This applies verbatim to event tensors and to variable time-window lengths.

### 2.7 Phase

**F22. Phase carries the structure** (Oppenheim & Lim, Proc. IEEE 69(5):529–541, 1981 — [PDF](https://dsp-group.mit.edu/wp-content/uploads/2024/11/ImportancePhaseSignals_1981.pdf)); **amplitude is easily disturbed by corruption while phase carries robust semantics** (APR, ICCV 2021, [arXiv:2108.08487](https://arxiv.org/abs/2108.08487)). So smoothing phase destroys exactly the component that carries structure.

**F23. Four pathologies, in increasing severity for DFFN:**
1. **Wrapping/branch cut.** atan2 confines phase to (−π,π] and is discontinuous at ±π; "ordinary loss functions for regression problems, including MSE, do not consider such [periodic] nature of phase" (Masuyama et al., ICASSP 2020, [arXiv:2002.05832](https://arxiv.org/abs/2002.05832)).
2. **A convex combination of angles is NOT the circular mean.** `Σ_k w_k φ_k` over nine wrapped phases straddling the cut returns ≈0 where the correct answer is ≈π. The correct operator is `arg(Σ_k w_k e^{jφ_k})`.
3. **Phase is shift-sensitive, magnitude is shift-invariant** — translation adds a linear ramp to phase.
4. **Near-zero magnitude makes phase undefined and its gradient unbounded** (`∂φ/∂(Re,Im) ~ 1/|F|`) — for a sparse event tensor that is *most* of the spectrum.

**F24. The elegant fix is free.** Applying the **same** convex kernel to the **complex** spectrum (real/imag jointly) automatically computes the magnitude-weighted circular mean of phase *and* the arithmetic mean of complex amplitudes. This is what FFC, Fourmer and SFHformer all do; SFHformer says so explicitly: "different from respectively extracting features from real and imaginary, we aggregate the real F_R and imaginary F_I in the channel dimension."

**F25. This fix does NOT address limitation (2).** A non-negative sum-to-one 3×3 kernel on the complex spectrum is still a pure smoother with DC gain 1. Fixing (3) leaves (2) exactly where it was.

**F26. If you keep magnitude/phase, use complex poles.** A complex pole `λ = r·e^{jθ}` multiplies magnitude by `r` and **adds** `θ` to phase — the correct, automatically wrap-safe group-delay semantics. S4D's complex `λ_n` already does this; Mamba-3's data-dependent RoPE on B/C is the modern selective version ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569), medium confidence on details).

### 2.8 Streaming state

**F27. Chunked training ≡ per-step recurrent inference, same weights.** Mamba-2/SSD's semiseparable block decomposition handles intra-chunk mixing with diagonal blocks and passes a low-rank state between chunks; train on whole clips in chunked matmul mode, deploy frame-by-frame with O(1) state, no retraining ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)). Zubić et al. do exactly this for events: "State-space models function as CNN during training and are converted to an efficient RNN at test time"; "By saving SSM states, each stage retains temporal information for the whole feature map" ([arXiv:2402.15584](https://arxiv.org/abs/2402.15584)).

**F28. You must TRAIN for streaming or the state is worse than useless.** Mamba-OTR: trained on fixed chunks, mp-mAP sliding-window 45.48 → streaming 43.35; an equivalent Transformer collapses to **0.04** ([arXiv:2507.16342](https://arxiv.org/html/2507.16342v1)).

**F29. Irregular timestamps have a settled treatment.** `Λ̄_k = exp(Λ·δ·Δ_k)` with `Δ_k = t_k − t_{k−1}` (Event-SSM, ICONS 2024, [arXiv:2404.18508](https://arxiv.org/abs/2404.18508)); STREAM Eq. 11 states it most explicitly and *explicitly contrasts it with Mamba-style input-dependent Δ*; a Nature Communications 2026 RRAM paper realizes the same decay physically with memristor short-term memory ([DOI 10.1038/s41467-025-68227-w](https://pmc.ncbi.nlm.nih.gov/articles/PMC12891684/)). **All of these are classification/detection — none is a restoration model.**

**F30. Explicit filters and recurrences are interconvertible.** Laughing Hyena Distillery extracts low-order SSMs from trained long convolutions via rational interpolation, giving O(1) per-token state at 10× Transformer throughput with no quality loss ([arXiv:2310.18780](https://arxiv.org/abs/2310.18780)); SpectraLDS gives the provable version with guarantees independent of state dimension ([arXiv:2505.17868](https://arxiv.org/abs/2505.17868)). **This is a concrete migration path: train the spectral filter you already have, then fit poles/zeros to it and get temporal state for free.**

### 2.9 Kernel-prediction head cost control (if you keep a per-bin kernel)

| Trick | Reduction | Source |
|---|---|---|
| Global basis + per-pixel coefficients | NK² → B channels/pixel; enables much larger kernels; >1 dB PSNR over per-pixel KPN | BPN, CVPR 2020 ([arXiv:1912.04421](https://arxiv.org/abs/1912.04421)) |
| Rank-1 separable (1-D × 1-D) | n² → 2n; enables 51×51 kernels | SepConv, ICCV 2017 |
| Decoupled spatial × channel | c·k² + σc²(1+k²) vs c³k²; **fewer params than a standard conv at σ=0.2,k=3,c=256**; ResNet50 +1.9% top-1 at ~half cost | DDF, CVPR 2021 ([arXiv:2104.14107](https://arxiv.org/abs/2104.14107)) |
| Channel-agnostic / grouped | K²·G instead of C·K²; RedNet-50 −39.5% params, −34.1% FLOPs | Involution ([arXiv:2103.06255](https://arxiv.org/abs/2103.06255)) |
| Fourier-disjoint weight groups | ResNet-50 **+3.6M** params vs CondConv +90M, DY-Conv +75.3M, ODConv +65.1M, KW +76.5M | FDConv, CVPR 2025 ([arXiv:2503.18783](https://arxiv.org/abs/2503.18783)) |
| Per-bin scalar gate in Fourier (skip the kernel) | one gain/bin instead of K² taps, O(N log N) not O(N·K²) | AFF, ICCV 2023 ([arXiv:2307.14008](https://arxiv.org/abs/2307.14008)) |
| Spectral descriptor instead of GAP for the generator | fixes "generator has no global context" at no head cost | FADConv 2025 ([arXiv:2504.03510](https://arxiv.org/abs/2504.03510)); ADFNet SEKG, AAAI 2023 ([arXiv:2211.12051](https://arxiv.org/abs/2211.12051)); IDF, ICCV 2025, ~0.04M params ([arXiv:2508.19649](https://arxiv.org/abs/2508.19649)) |

**F31. FDConv's diagnosis is our limitation (2) in kernel space:** attention-weighted sums of kernels give near-identical frequency responses — ODConv's learned weights have **cosine similarity >0.88**. A convex mixture of kernels is spectrally redundant.

---

## 3. Competitive landscape on the 50mm / 150mm event rain benchmark

**Benchmark identity.** The dataset is **EventRain-27K**, introduced *by PRE-Mamba itself* (ICCV 2025) — there is no separate dataset paper. Synthetic split (>7K samples) = KITTI + SPAC clean video → rain rendered with the Tremblay et al. simulator → events via Vid2E. Plus >7K self-recorded artificial (Prophesee EVK4) and >9K real-world samples. Windows are `dt = 0.1 s`; PRE-Mamba consumes **5 consecutive windows (0.5 s) per forward pass**. HF card (medium confidence): synthetic_KITTI 460×352, synthetic_SPAC 640×480, EVK4 1280×720; 301 GB, MIT. [HF](https://huggingface.co/datasets/Rshnn/EventRain-27K) · [code](https://github.com/softword-tt/PRE-Mamba)

**Metric definitions (verbatim from PRE-Mamba §5.1):** `DA = ½(SR + NR) = ½(PB/TB + PR/TR)`.
- **SR** = Signal Retention = PB/TB = recall on the background/signal class = TPR.
- **NR** = Noise Removal = PR/TR = recall on the rain class = TNR.
- **DA** = balanced accuracy at a *single operating point* (not an AUC).

> ⚠️ A live WebFetch of the arXiv abstract page this session had the summarizing model gloss "SR" as *"Spatial Reconstruction"*. **That is the fetcher's invention, not the paper's.** Use Signal Retention.

### 3.1 Accuracy, EventRain-27K synthetic split (SR / NR / DA)

| Method | 5 mm/h | 20 mm/h | **50 mm/h** | 80 mm/h | 125 mm/h | **150 mm/h** |
|---|---|---|---|---|---|---|
| TS | — | — | .883/.231/.557 | — | — | .872/.243/.557 |
| DWF | — | — | .755/.352/.553 | — | — | .782/.375/.578 |
| Knoise | — | — | .884/.241/.563 | — | — | .896/.214/.555 |
| Ynoise | — | — | .663/.481/.572 | — | — | .634/.487/.561 |
| RED | — | — | .833/.208/.520 | — | — | .789/.208/.499 |
| EDnCNN | .968/.905/.937 | — | .948/.888/.918 | — | — | .929/.843/.886 |
| AEDNet | — | .938/.876/.907 | .928/.732/.830 | — | — | .923/.547/.735 |
| EDformer | .981/.818/.899 | .962/.832/.897 | .924/.844/.884 | — | — | .839/.834/.836 |
| **PRE-Mamba** | .994/.914/.954 | .978/.915/.947 | **.955/.911/.933** | .940/.903/.922 | .918/.898/.908 | **.908/.895/.902** |
| DistillNet (voxel) | — | — | *excluded* | — | — | *excluded* |
| **— our (A) FourierMamba2D** | — | — | **.967/.906/.936** | — | — | **.921/.908/.915** |
| **— our (B) DFFN** | — | — | **.965/.868/.914** | — | — | **.916/.876/.896** |

**Verification status:** PRE-Mamba's *headline averages* (SR 0.95, NR 0.91, 0.4 s/M events, 0.26M params, EventRain-27K) were **verified live this session from the arXiv abstract page**. The **entire per-rain-rate table above is UNVERIFIED by me** — it comes from an upstream agent's read of arXiv HTML v2 Tables 1–2, reported at high confidence. Re-read those tables before publishing a joint table. **No second paper anywhere reports SR/NR/DA on EventRain-27K** — there is no leaderboard, no third-party reproduction, and no independent cross-check of 0.955/0.911/0.933.

### 3.2 Efficiency — ⚠️ INCOMMENSURABLE DENOMINATORS

| Method | Params | GFLOPs | Time | Denominator |
|---|---|---|---|---|
| EDnCNN | 614.55K | 234.51 | 20.1885 s | per 100K events (1.0× ref) |
| AEDNet | 45.87M | 4400.46 | 43.4250 s | per 100K events (0.46×) |
| DistillNet | 18.96M | 255.17 | 0.2029 s | per 100K events (99.50×) |
| EDformer | 49.80K | 8.41 | 2.4943 s | per 100K events (8.09×) |
| **PRE-Mamba** | **264.63K** | **6.23** | **0.0987 s** | per 100K events (204.54×) |
| **EDmamba** | **88.98K** | **2.27** | **0.0685 s** (RTX A6000) | per 100K events (294.72×) |
| **our (A) FourierMamba2D** | 23.6M | 260 | 153 ms | **per forward pass** |
| **our (B) DFFN** | 72K | 5.36 | 10.8 ms | **per forward pass** |

**These cannot go in one column.** Theirs is per 100K events; ours is per forward pass over a windowed tensor. Convert via (events/window × 5 windows) before any comparison, and state the conversion in the paper.

### 3.3 Three things a reviewer will attack first

1. **Split mismatch.** PRE-Mamba's table mixes KITTI *and* SPAC sources. If we train/test only on the KITTI half, our numbers are **not directly comparable**. Verify before publishing a joint table. Our SR at 50mm (.967/.965) exceeds every baseline including PRE-Mamba's .955 — this will be scrutinized.
2. **Per-event scoring from a dense model.** SR/NR/DA are per-event metrics. PRE-Mamba explicitly excludes DistillNet from the accuracy table because voxelization destroys per-event correspondence. **We are a frame/voxel model and inherit that objection.** We must state the events↔voxel mapping and show how per-event SR/NR/DA are recovered from a dense output.
3. **The wrong rival.** **EDmamba (88.98K / 2.27 GFLOPs), not PRE-Mamba, is our efficiency rival.** DFFN at 72K is 1.25× smaller but uses **2.4× more GFLOPs**. PRE-Mamba is the accuracy rival.

### 3.4 Where the headroom actually is

DFFN's gap to PRE-Mamba at 50 mm is **SR +0.010 (we win), NR −0.043 (we lose), DA −0.019**. At 150 mm: SR +0.008, NR −0.019, DA −0.006. **The gap is almost entirely NR** — deciding which events are noise, a discriminative and largely *local* decision. Global spectral context is not obviously the right medicine for NR. Model (A) already closes it (NR .906/.908) at 14× the latency. **Run the cheap ablation first: does an unconstrained/signed tap parameterization (DDF filter-norm, KW signed-L1, or Res FFT-ReLU thresholding) close the NR gap at zero latency cost?** That is the control experiment a reviewer will demand and it costs nothing.

---

## 4. Novelty verdicts

### 4.1 `ssm-as-filter` — replace the per-bin softmax 3×3 with a selective SSM scan over frequency bins → **ALREADY_DONE**

**Closest prior art:** [FourierMamba, ICML 2025 / arXiv:2405.19450](https://arxiv.org/abs/2405.19450). Verified against the primary PDF (2405.19450v2, Eq. 6, Fig. 3): "The amplitude spectrum and phase spectrum are then processed separately using the progressive frequency scanning method… A′(F_l)=FourScan(A(F_l)), P′(F_l)=FourScan(P(F_l))", with the operator spelled out as S6 with ZOH (`Ā,B̄ = exp(ΔA), ΔB; h_k = Āh_{k−1}+B̄x_k; y_k = Ch_k+Dx_k`) over a zigzag-ordered bin sequence. Also: [OSMamba](https://arxiv.org/abs/2411.15255) (parallel Amplitude-Mamba/Phase-Mamba on FFT tokens), [PAS-Mamba](https://arxiv.org/abs/2601.14530), [Adaptive IIR Filters, EMNLP 2023](https://arxiv.org/abs/2305.14952) (input-dependent order-2 IIR whose coefficients are determined from previous chunks — the framing), [S4D](https://arxiv.org/abs/2206.11893) + [RTF](https://arxiv.org/abs/2405.06147) (the pole-bank/rational-filter facts), [TF-Mamba, Interspeech 2025](https://arxiv.org/abs/2409.05034) and [SPMamba](https://arxiv.org/abs/2404.02063) (frequency-axis Mamba in audio), [BSRNN](https://arxiv.org/abs/2209.15174) (frequency-axis recurrence emitting a per-bin mask, 2022).

**Differentiator:** *None as an operator.* This is FourierMamba's FourScan verbatim — and FourierMamba is our own model (A), which means the claim reduces to "put model A's operator inside model B's block". The three stated advantages are all pre-owned: (a) global spectral receptive field = FourierMamba/FreqMamba's stated motivation; (b) rational/IIR expressivity = S4D + RTF, input-dependent version = EMNLP 2023; (c) `C+2N` head cost = Mamba's default parameterization plus arithmetic. **The only defensible framing is application-level: "first frequency-bin selective scan on event-camera data, presented as a budget-neutral drop-in for a K² spectral dynamic filter with a head-cost-matched ablation."** Never claim a new operator or a new expressivity argument.

### 4.2 `polar-scan` — order the spectral scan by angle, then radius, to make the rain wedge contiguous → **INCREMENTAL**

**Closest prior art:** [Vcamba, arXiv:2507.23601](https://arxiv.org/abs/2507.23601) — FSS "starts from the spectral center and spirals outward along increasing radii in a clockwise manner", bidirectional, feeding parallel S6 blocks; [PAS-Mamba's CFDS](https://arxiv.org/abs/2601.14530) — "Circular Frequency Domain Scanning to serialize features from low to high frequencies", respecting concentric k-space geometry; **Synergistic Mamba (ESWA 2026, [code](https://github.com/cookfu/Synergistic_Mamba))** — verified from released source `synmamba_arch.py`: `getFourierSort` builds `freq_grid = freq_grid_x**2 + freq_grid_y**2` and returns `torch.argsort(freq_grid_flat)`, i.e. Euclidean Concentric Scanning **is literally an argsort of rfft2 bins by squared radius with the angle discarded**; [DH-Mamba, arXiv:2501.08163](https://arxiv.org/abs/2501.08163) ("circular scanning is customized for spectrum unfolding"); [PRISMamba, arXiv:2602.04170](https://arxiv.org/abs/2602.04170) — the only angle-major SSM found ("short radial SSMs" across concentric rings), but in the **spatial** domain for rotation robustness.

**Differentiator:** *"No published spectral SSM scan is angle-major — FourierMamba zigzags, DH-Mamba/PAS-Mamba walk concentric rings, Synergistic Mamba's released code argsorts rfft2 bins by u²+v² with the angle discarded, and Vcamba spirals outward along increasing radii; in every case radius is the slow variable, so a fixed-orientation rain ridge through DC is fragmented into O(R) short runs, and ordering primarily by θ and only secondarily by r is the unique permutation under which an entire rain wedge is one contiguous run."* Defensible but **thin** — it is a loop-order transposition of a module published at least four times. It survives review only if shipped with an isolated ablation (identical block, identical parameter count, **only the index permutation changed**: raster vs zigzag vs radius-major vs angle-major) **plus** evidence that the selective gate's Δ actually tracks streak orientation.

### 4.3 `temporal-state` — carry the SSM hidden state across event time windows to condition the spectral filter generator → **INCREMENTAL**

**Closest prior art:** [Meta-AF, TASLP 2023 / arXiv:2204.11942](https://arxiv.org/abs/2204.11942) — the mechanism kill: GRU layers "maintain separate state ψ_k[τ] per frequency", weights "shared across all frequency bins", driving an overlap-save **frequency-domain** filter updated once per STFT frame at cost linear in parameter count. [State Space Models for Event Cameras, CVPR 2024](https://arxiv.org/abs/2402.15584) — "By saving SSM states, each stage retains temporal information for the whole feature map"; CNN-train / RNN-infer. [oSpatialNet, arXiv:2403.07675](https://arxiv.org/abs/2403.07675) — per-frequency-bin Mamba with state carried across frames, linear inference complexity, length extrapolation. [DeepFilterNet, arXiv:2305.08227](https://arxiv.org/abs/2305.08227) — recurrent-state-driven complex frequency-domain filter, RTF 0.19 on one CPU thread. [CETUS, arXiv:2509.13784](https://arxiv.org/abs/2509.13784) — "predefined time windows… introduce window latency", replaced with causal Mamba + variable-rate scheduling. [PRE-Mamba](https://arxiv.org/abs/2505.05307) — batches 5 windows, does **not** carry state.

**Differentiator:** *"The SSM hidden state carried across event time windows conditions the FILTER-GENERATION NETWORK of a per-frequency-bin dynamic 2-D kernel applied to the rFFT of an event tensor — i.e. the past of the stream enters as the generator's context rather than as extra scanned tokens — which no prior work does: Meta-AF/DeepFilterNet/oSpatialNet do it with GRUs/Mamba on audio spectra with no learned 2-D spectral kernel, Zubić et al. and CETUS carry event-SSM state with no Fourier stage at all, and PRE-Mamba does event deraining with an SSM but uses frequency only as a loss and batches windows instead of streaming."*

**Do not claim** "first temporal state for a frequency filter" or "first streaming event restoration" — both are false. **Do not write an unqualified O(1) marginal cost** — the per-window rfft2/irfft2 remains O(HW log HW); only the conditioning path is O(1).

**Strongest supporting argument you own:** PRE-Mamba's own window ablation — DA 0.8268 @274 ms (3 windows) → 0.9015 @394 ms (5) → **0.9330 @526 ms (8)**, still climbing. A carried state would in principle give the 8-window accuracy at the 3-window latency. That is a quantified, published motivation.

### 4.4 `single-fft-trunk` — one rfft2 in, N frequency-resident blocks, one irfft2 out → **ALREADY_DONE**

**Closest prior art:** **[Transform Once (T1), NeurIPS 2022, arXiv:2211.14453](https://arxiv.org/abs/2211.14453)** — verified live this session, abstract verbatim: *"Existing FDMs are based on complex-valued transforms… and layers that perform computation on the spectrum and input data separately. This design introduces considerable computational overhead: for each layer, a forward and inverse FT. Instead, this work introduces a blueprint for frequency domain learning through a single transform: transform once (T1)."* Authors: Poli, Massaroli, Berto, Park, Dao, Ré, Ermon. Reports **3×–10× speedups** that grow with resolution and model size, **>20% reduction in average predictive error**, and 5 hours instead of 32 on their large experiment — i.e. *both faster and more accurate*, the exact result we would claim. It also derives the variance-preserving init required to make a frequency-resident trunk trainable, and its best variant is a **U-Net living entirely in the spectrum**. Also: [Deep Residual Learning in the JPEG Transform Domain, ICCV 2019](https://arxiv.org/abs/1812.11690) (a whole ResNet resident in a transform domain); [TF-GridNet, TASLP 2023](https://arxiv.org/abs/2211.12433) and [FRCRN, ICASSP 2022](https://arxiv.org/abs/2206.07293) (the entire speech-enhancement field runs one STFT → N T-F-resident blocks → one iSTFT, and treats it as unremarkable background).

**Differentiator:** *None.* The claim is T1's literal contribution down to the motivating sentence. A writeup presenting the single-FFT trunk as a contribution will be desk-rejected on T1 alone. The only residue is positioning: no *image-restoration / deraining / event-camera* paper has ported it (an S2 bulk query for restoration-terms + "frequency domain" + "entire network"/"all layers"/"throughout the network" returned **total=0**). If pursued, frame as *"we port T1's transform-once blueprint to event deraining and show it composes with a spectral SSM"*, with T1 cited in the first method paragraph.

**Also note:** FlashFFTConv already recovers most of the same saving by fusing FFT→multiply→iFFT into one kernel with **zero architectural change**. A reviewer will ask why we did not just do that. And profile first: at 72K params / 5.36 GFLOPs / 10.8 ms the transforms may already be a small share of the budget.

### 4.5 `complex-circular` — filter (cos φ, sin φ) or (Re, Im) instead of wrapped phase → **ALREADY_DONE**

**Closest prior art:** [SFHformer, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06190.pdf) — the kill shot, because it is the *same architecture family doing the same operation with the opposite representation, deliberately*: "different from respectively extracting features from real and imaginary, we aggregate the real F_R and imaginary F_I in the channel dimension", then applies FCPE (a depthwise conv *across* the (u,v) grid) and FDC (per-bin softmax-mixed dynamic conv). [Ai & Ling, ICASSP 2023 / arXiv:2211.15974](https://arxiv.org/abs/2211.15974) — publishes the branch-cut argument as an explicit architecture: parallel pseudo-real/pseudo-imaginary heads recombined by a quadrant-corrected arctangent, with the true error stated as `e = min{|P̂−P|, 2π−|P̂−P|}`. [PHASEN, AAAI 2020](https://arxiv.org/abs/1911.04697) — phase stream normalized to unit amplitude per T-F bin, i.e. (cos φ, sin φ). [Complex ratio masking, TASLP 2016](https://dblp.org/rec/journals/taslp/WilliamsonWW16) — the canonical "estimate Re/Im, never regress an angle". [Lee et al. TGRS 1998 / Goldstein–Werner GRL 1998] — settled InSAR doctrine for ~30 years. [FFC](https://proceedings.neurips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html), [Fourmer](https://proceedings.mlr.press/v202/zhou23f/zhou23f.pdf), [GFNet](https://arxiv.org/abs/2107.00645), [AFNO](https://arxiv.org/abs/2111.13587) — all rectangular. [HAMSA, arXiv:2604.14724](https://arxiv.org/abs/2604.14724) (verified live) — SAGU is "magnitude-based gating for stable gradient flow in the frequency domain", i.e. a 2026 SSM+spectral paper already argues phase must not be naively operated on.

**Differentiator:** *None as a method.* Operating on the complex spectrum is the field default, and the (cos, sin)+atan2 variant is published with exactly the branch-cut motivation. The only residue is **diagnostic**: nobody has published a controlled ablation, inside one Fourier restoration/deraining model, of convex filtering of *wrapped phase* vs. the circular-mean/complex equivalent, quantifying the damage — searches for `"phase wrapping" + deraining` and `abs:"phase wrapping" AND cat:cs.CV` return zero restoration hits. **Ship it as a bug fix and a one-line ablation row, never as a contribution.**

---

## 5. Open gaps — the whitespace, as opportunities

Ranked by defensibility × payoff.

**G1. Nobody has analysed what a K×K convolution over the *spectrum* is.** The duality (convolution in frequency = multiplication in space) is in DSP textbooks and analysed in MRI (SENSE/GRAPPA, RAKI), but **no vision or restoration paper states that a small conv over the spectrum is a spatial window/apodization, nor examines what constraints on the taps imply.** Combined with the Hann identity `[1/4,1/2,1/4]`, this yields a clean theorem-flavoured result: *DFFN's softmax spectral 3×3 is provably a learned generalized-Hann apodization, hence unable to notch.* **This is publishable analysis at zero experimental cost and it converts your worst limitation into your best contribution.**

**G2. Nobody has applied a learned local convolution in the frequency domain to event-camera data.** The event deraining/denoising literature is entirely spatiotemporal or point-based (PRE-Mamba, EDmamba, EDformer, AEDNet, EDnCNN, Event-Based De-Snowing). The only event+frequency work is non-learned and analytic: eFFT (exact incremental 2D DFT), DDHF (per-pixel NDFT), FRIES (resonant time surface). **"Dynamic Fourier filter + SSM + event stream" is unoccupied as of July 2026.**

**G3. No frequency filter with temporal memory, anywhere in restoration.** FourierMamba / FreqMamba / DFSSM / DeRainMamba / OSMamba / PAS-Mamba all scan frequency bins *within a single frame*. RainMamba and DemMamba carry temporal SSM state but purely spatially. **A recurrent state carrying the estimated rain-ridge orientation across event windows is well-motivated by Barnum's result (rain is flat in temporal frequency but stable in orientation) and appears novel.** DTF (ECCV 2022, [arXiv:2211.08252](https://arxiv.org/abs/2211.08252)) is the closest — dynamic frequency filters applied along time by 1-D FFT — and is the baseline to beat.

**G4. No separate hypernetwork emitting full SSM parameters (Δ, A, B, C) per position on a 2-D or frequency grid.** Every confirmed selective SSM (Mamba, Mamba-2, S7, Mamba-3, Liquid-S4) uses plain **linear** projections. **Swapping DFFN's existing Conv1×1→DW3×3→Hardswish→Conv1×1 head's output space from "9 softmaxed taps" to "(Δ, B, C) per bin" is a drop-in: same generator, same cost, strictly larger reachable filter class.** Four differently-phrased searches found nothing.

**G5. No circular-statistics-aware phase filtering in image restoration.** Nobody uses `arg(Σ_k w_k e^{jφ_k})`, a von Mises formulation, or a wrapping-aware phase loss in a deraining/denoising network. **Caveat:** the argument is fully published in speech (Ai & Ling) and InSAR, so this is a *bug fix with a citation*, not a contribution. **Free bonus nobody has taken: the resultant length |r| = |Σ_k w_k e^{jφ_k}| is the circular concentration of the local phase neighbourhood — a physically meaningful rain-vs-shot-noise discriminator that atan2 throws away.** Feed it to the FFN or a gate; this is more likely to buy accuracy than the wrap fix itself.

**G6. No direct empirical comparison of a full-spectrum weight tensor vs a local frequency-neighbourhood convolution for deraining.** The only apples-to-apples global-vs-local ablation (GLFNet) is on medical segmentation. **A clean global-vs-local-vs-both ablation in deraining is an unclaimed, cheap experimental contribution.**

**G7. Nobody stays entirely in the frequency domain across blocks AND uses an SSM as the frequency-axis mixer with no round trip.** T1 owns the trunk; FourierMamba/FreqMamba/OSMamba/PAS-Mamba/SpectMamba/ASGMamba all transform back each block. **The composition is open — but must be framed as composition, not as the trunk.**

**G8. No delta-from-physical-inter-event-time in an event RESTORATION model.** Established in Event-SSM, STREAM, S7, and the RRAM CIM paper — all classification/detection. PRE-Mamba and EDmamba both use normalized time as a *coordinate feature* instead.

**G9. eFFT as a streaming Fourier front-end for events is completely untouched by the learning community.** An exact 2D DFT maintained *incrementally* per event, quadtree-stored butterflies, microsecond per-event latency, C++ released ([TPAMI 46(12):9630-9647](https://ieeexplore.ieee.org/document/10582443/), [DOI 10.5334/jors.642](https://doi.org/10.5334/jors.642)). **Replacing the per-window rfft2 with a continuously-maintained spectrum pairs naturally with a carried SSM state and is a genuinely novel systems contribution.** Caveat: CPU-only benchmarks so far.

**G10. Training-side idea nobody has ported to events:** FADformer's Frequency-domain Contrastive Regularization uses rain-streak patterns as **negative** samples — free for us, since the synthetic KITTI pipeline gives paired rainy/clean event tensors ([ECCV 2024](https://eccv.ecva.net/virtual/2024/poster/1791)). PRE-Mamba's `L_fft` is worth only ~+0.15 DA points (0.9000→0.9015), so there is room.

---

## 6. Contradictions and disagreements between sources

**C1. Amplitude-only vs amplitude-and-phase degradation.** Fourmer's amplitude/phase swap experiment concludes "the degradation primarily occurs in the amplitude component" (ICML 2023), and FourierMamba adopts it as its justification for Fourier deraining. **DeRainMamba (SPL 2025) explicitly contradicts this**, arguing rain distorts *both* spectra, worst in high-frequency regions where edges live, and that ignoring phase distortion produces ghosting, edge blurring and structural misalignment. DFMIR-Net (J. Supercomputing 2026) independently votes for "phase-preserving". **Unresolved. It is directly testable on our own data (swap amplitude/phase between rainy and clean event tensors) and the experiment is cheap. Do it — it determines whether the phase branch is worth fixing at all.**

**C2. Global vs local spectral operators.** GFNet/FNO/AFNO/DFFormer argue for full-spectrum global filters; SFANet argues global full-spectrum tensors are fundamentally fragile due to frequency-resolution mismatch; GLFNet's ablation says local ≈ global alone and only the *combination* wins; ICML 2024's localized neural operators say global spectral convolution over-smooths and loses local detail (34–72% error reduction from adding local kernels back). **Resolution: keep both paths. This is a genuine consensus once you read the ablations rather than the abstracts.**

**C3. Magnitude/phase vs real/imaginary.** The *highest-performing* frequency blocks (FFC 2020, Fourmer 2023, SFHformer 2024) deliberately use real/imag. The *SSM* frequency papers (FourierMamba, OSMamba, ExpoMamba, PAS-Mamba, Synergistic Mamba, DFFN) all split magnitude/phase. HAMSA gates on magnitude only and explicitly cites gradient stability. **The community treats this as a design axis that must be ablated (Deep Fourier Shifting provides both variants), not a settled choice. It is not settled — but the mathematical argument (F24) favours rectangular.**

**C4. Softmax normalization: stability aid or expressivity killer?** DyConv (CVPR 2020) defends it explicitly on optimization grounds — "the sum-to-one constraint further compresses the kernel space to a triangle… significantly simplifies the learning of π_k(x)". DCNv4 (CVPR 2024), Involution (CVPR 2021) and DKN (IJCV 2021) all remove it and quantify the gain. **Both are true.** The resolution the literature converged on is stability *without* convexity: DDF's filter normalization, KernelWarehouse's signed L1, S7/LRU's stable reparameterization.

**C5. Event-SSM's own reported numbers differ between sources.** arXiv gives SSC 88.4% / SHD 95.9% / DVS128 97.7%; S7's comparison table gives SSC 87.1% / SHD 95.5% / DVS128 97.7% @5.4M. Likely different revisions/augmentation. Do not quote either as canonical.

**C6. Two different papers are called "EventMamba."** [arXiv:2503.19721](https://arxiv.org/abs/2503.19721) (Ge et al., video reconstruction, voxel + Hilbert) and [arXiv:2405.06116](https://arxiv.org/abs/2405.06116) (Ren et al., point-based classification/regression). Do not conflate. Similarly, **"Frequency Dynamic Convolution" is a name collision**: FDConv (CVPR 2025, Fourier re-parameterization of *weights*) vs FDY-conv (INTERSPEECH 2022, per-frequency-bin kernels applied in the *spatial* domain over a spectrogram). Structurally, **FDY-conv is closer to DFFN than FDConv is.**

**C7. MamEVSR venue.** CVF OpenAccess lists it under CVPR 2025 with a CVPR 2025 poster page; one secondary source says ICCV 2025. Unresolved.

---

## 7. What could not be verified

**Verified live this session (3 items only):** arXiv:2211.14453 (T1 — title, full author list, abstract verbatim, the "forward and inverse FT per layer" sentence, 3×–10× speedup, >20% error reduction); arXiv:2505.05307 (PRE-Mamba — title, authors, ICCV 2025, 0.95 SR / 0.91 NR / 0.4 s per M events / 0.26M params / EventRain-27K); arXiv:2604.14724 (HAMSA — title, authors Patro & Agneeswaran, submitted 16 Apr 2026, abstract verbatim including SpectralPulseNet and SAGU).

**Everything else is inherited from upstream agents at their stated confidence.** The WebSearch budget was exhausted (200/200) before this agent's first call, so no independent search-based cross-checking was possible.

### Numbers I could not verify
1. **The entire PRE-Mamba per-rain-rate SR/NR/DA table (§3.1)** and **the entire efficiency table (§3.2)** — read by an upstream agent from arXiv HTML v2 Tables 1–2 at high confidence, not re-verified. **Re-read before publishing.**
2. **No independent reproduction of PRE-Mamba exists.** No leaderboard, no third party, no second paper reporting SR/NR/DA on EventRain-27K — not even EDmamba, by the same authors, released the same day.
3. **EventRain-27K resolutions** (synthetic_KITTI 460×352, synthetic_SPAC 640×480, EVK4 1280×720) and the `dt = 0.1 s` window come from the **HuggingFace card via a summarizing fetch**, not the paper. The paper never states the synthetic resolution.
4. **No published per-rain-rate sample counts** for EventRain-27K (how many of the ~7K synthetic samples are 50 mm vs 150 mm).
5. **Fourmer's deraining config** (~1.29M params / ~20.6 GFLOPs) — column alignment in the extracted table was uncertain. Do not quote.
6. **FFC's Fourier Unit internal kernel size** — the paper says only "Conv". Widely implemented as 1×1 in `pkumivision/FFC`. **Verify in code before asserting "FFC is pointwise in frequency."**
7. **FDConv reports no latency anywhere** (paper or README). KPN reports no FLOPs, only ~0.7 MP/s on a GTX 1080 Ti.
8. **FDY-conv's basis-attention normalization** (whether softmax with temperature) could not be verified — the ISCA PDF is image-only. Do not assert.
9. **EventZoom quantitative numbers** — CVF and IEEE returned 403; only abstract-level claims (≥40× temporal efficiency) confirmed.
10. **ICCV 2023 "Unsupervised Video Deraining with An Event Camera"** — PDF 403; PSNR/SSIM unconfirmed.
11. **CETUS and TURTLE results tables** — PDF text extraction failed; no numbers reported above for either.
12. **Synergistic Mamba (ESWA 2026)** — ScienceDirect 403. The ECS mechanism comes from the released GitHub source (which *is* strong primary evidence) plus search snippets; the paper text itself was not read.
13. **DFMIR-Net (J. Supercomputing 2026)** — Springer paywall; no numbers.
14. **DistillNet full text** — ACM DL 403. Architecture description comes from PRE-Mamba's related work.
15. **Guo & Delbruck TPAMI 2023 (DND21 origin)** — paywalled; only the dblp record verified.
16. **Lee et al. TGRS 1998 internal mechanism** (real/imag filtered separately then recombined by arctangent) — IEEE abstract publisher-restricted. Medium confidence; verify before citing that detail.

### Author lists / venues not confirmed
- **GLFNet** ([arXiv:2403.00396](https://arxiv.org/html/2403.00396v1)) — HTML did not expose the author list. **Its ACDC ablation is load-bearing for our global+local design decision; verify before citing.**
- **HADES** (arXiv:2603.22333) — authors reported as Shin/Kim/Park by one agent, "not resolved" by another. **Conflicting.**
- **Mamba-3** (arXiv:2603.15569) and **SurgicalMamba** (arXiv:2605.14889) — author lists unresolved.
- **"Deep Fourier Shifting"** (OpenReview 3gKsKFeuMA) — authors unconfirmed.
- **CosAE** (NeurIPS 2024) — first-author ordering not confirmed.
- **DTF** (ECCV 2022) — authors beyond Fuchen Long unverified.
- **SPANet** (arXiv:2503.23947), **FMDConv** (arXiv:2503.17530), **RainMamba** (arXiv:2407.21773), **EventMamba point-based** (arXiv:2405.06116) — partially verified.
- **Mamba** (arXiv:2312.00752) conference venue (COLM 2024?) and **FreqMamba** venue — unconfirmed by the surveying agents despite both being widely cited.
- **K-Space Transformer** (arXiv:2206.06947) BMVC 2022 venue — low confidence.

### Claims reported as derivations, NOT citations
- **F8** (`|H(e^{jω})| ≤ 1 = H(1)` for non-negative sum-to-one taps). Elementary triangle inequality. **No paper states it in this form** — softmax-normalized dynamic kernels are standard practice and their non-expansive/low-pass-only consequence appears unremarked. This is exactly why G1 is publishable, but do not cite a paper for it.
- **F12's explicit `m(x,y)` expression** — derived from the DFT duality; the Hann identity `[1/4,1/2,1/4]` *is* citable (J.O. Smith, SASP).
- **F17's per-D cost table** — arithmetic on quoted Mamba facts.
- **The "ridge is perpendicular to the spatial streak"** reading of Barnum Eq. 5 — the upstream agent's interpretation, not a sentence Barnum writes. **Cite the equation.**
- **F23.4** (unbounded phase gradient at near-zero magnitude) — standard numerics, unsourced engineering guidance.
- **PRE-Mamba's limitations** — the paper has **no limitations or future-work section** (checked twice). Any limitation attributed to it must be labelled as our own analysis.

### Leads seen only as titles — NOT confirmed, do not cite
"Three-domain joint deraining network" (SPIC 2025); "Learning Dual-Domain Multi-Scale Representations for Single Image Deraining" (arXiv:2503.12014); "Hybrid Mamba-Transformer with Frequency Enhancement for Single Image Deraining" (Springer 2025); "PRISM: Progressive Rain removal with Integrated State-space Modeling" (arXiv:2509.26413); "SD-Conv" (arXiv:2204.02227); "PAD-Net" (arXiv:2211.05528); "Asynchronous Event-Based Fourier Analysis" (IEEE TIP 2017, DOI 10.1109/TIP.2017.2661702); MFMamba (ScienceDirect S0952197625033408); FMambaIR (DOI 10.1109/TGRS.2025.3526927); DeLiVR (in-plane rain angle prediction — **relevant to the polar-scan motivation if it exists; verify separately**).

### Negative results whose scope must be stated honestly
Every "nobody has done X" in §5 is a **web-search / arXiv-API negative, not an exhaustive database sweep.** Specific limits: arXiv abstract search cannot see method details buried in full text; pre-arXiv journal work is poorly covered. Three literatures were **not searched at all** and plausibly contain prior art for our *motivations* (not our operators): (i) **phase unwrapping / InSAR / SAR** for circular-statistics filtering — G5 is almost certainly weaker than it looks there; (ii) **curvelet / shearlet / steerable-pyramid** deraining and **seismic f-k fan/dip filtering** and **remote-sensing directional Fourier destriping**, all of which partition the spectrum into angular wedges to remove oriented linear noise — this is prior art for the polar-scan *motivation*; (iii) **DCT-domain CNN** literature for the frequency-resident trunk. Check these before claiming novelty in §4.2 or §5-G5.

### Two name-level traps
**"FAMamba" is not a paper** — it is an aggregator-site umbrella term (emergentmind) for frequency-aware Mamba designs. Do not cite it. **Mamba-OTR's "8 ns per frame"** is almost certainly a table mis-parse — do not cite it.

---

## Recommended shape, given all of the above

*(Synthesis, not a citation. Ordered by expected value per unit of risk.)*

1. **Run three zero-architecture ablations first, in this order.** (a) Temperature annealing on the existing softmax (τ 30→1 over 10 epochs — DyConv's recipe, +0.5% top-1 there). (b) Replace softmax-over-taps with **softmax-over-bases** (SFHformer FDC style) or DDF filter-normalization — signed, sign-unconstrained, same cost. (c) Amplitude/phase swap on our own event tensors to settle C1. Each is a day's work and each could close the NR gap, which is where all our headroom is.
2. **Keep the local 3×3.** F20 says local ≈ global alone and the win is having both. Add the SSM as a *second, cheap global path* conditioning the generator — do not replace the local path.
3. **Use SSD/Mamba-2 chunked matmul or a d_state≤4 minGRU-class recurrence, not S6.** F17/F19: strip the 6D² projection stack, keep the recurrence. Set an explicit latency budget before writing code.
4. **Filter real/imag, not magnitude/phase.** Free, cheaper (no atan2/sin/cos in the inner loop), and it makes the convex kernel compute the correct magnitude-weighted circular mean. Watch the Hermitian-symmetry constraint on the u=0 and Nyquist columns — a 3×3 that mixes neighbouring bins on Re/Im will generically violate it and irfft2 will silently discard the violation.
5. **Initialize the spectral SSM in the Fourier domain (S4D-DFouT) or you will reproduce the low-pass behaviour you are escaping** (F7).
6. **Spend the novelty budget on G1 (the apodization theorem) + G3/G4 (temporal state conditioning the generator) + G2 (events).** Cite FourierMamba, T1, SFHformer, Ai & Ling and Synergistic Mamba's ECS in the first paragraph of the method. Do not claim a new operator.
7. **Before any polar-scan work, plot the orientation histogram of streaks in the 50mm and 150mm splits and the measured rain-only event spectrum in polar coordinates.** If orientation is near-constant (likely, for a fixed-rate rain-rendering pipeline), a *static* oriented prior captures the gain and the dynamic module is untestable; if the wedge is not visibly narrow in event data (F3), the motivation collapses entirely.