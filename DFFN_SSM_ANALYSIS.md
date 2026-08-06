# Dynamic Fourier Filtering × SSM/Mamba for Event Deraining

*Project scan, measured diagnosis, literature positioning, ranked proposals.*
*2026-07-28. Every number tagged **[measured]** was produced and verified in this session, on this
project's own code and the real paired data in `dataset/efft_results`. Scripts in §8.*

> **Note on parallel work.** A second Claude session was running in this workspace concurrently and
> authored `DFFN_SSM_PROPOSAL.md` plus several scratch scripts. This file is independent. Where the two
> overlap I re-ran the experiment myself (§4.3); where I could not verify a number I left it out. See §9.

---

## 0. Bottom line

**The fusion works — but not for the reason you'd expect, and that distinction should drive the
research plan.**

In a controlled pilot (same data, loss, seed, schedule), replacing DFFN's 3×3 frequency-domain dynamic
filter with a selective state-space scan gives **+1.7 DA points with 32 % fewer parameters and 1.2–1.5×
lower latency** [measured]. So far so good.

**Then the control arm ties it.** Giving DFFN's *existing* 3×3 filter generator a 72-dimensional
orientation × radius histogram of the spectrum — **no SSM anywhere** — scores 0.8362 against the scan's
0.8346, a gap of 0.0016 at one seed. Both beat the baseline's 0.8173 by ≈ +1.8. **The accuracy gain
comes from letting the filter generator see global spectral structure, not from the state space model.**
§2.4 explains why: rain's Fourier signature is a *global, per-frame orientation prior*, and a 3×3
window over frequency bins is structurally blind to it.

**The operator is also not novel — and the prior art is your own baseline.** An adversarial prior-art
sweep returned `ALREADY_DONE`: a selective scan with input-dependent (Δ, B, C) over an ordered sequence
of Fourier bins, applied separately to amplitude and phase, is **FourierMamba's FourScan**
(arXiv:2405.19450, Eq. 6 — that PDF is in your repo) — i.e. model A of your own deck. Also OSMamba
(arXiv:2411.15255), PAS-Mamba (arXiv:2601.14530), and in audio BSRNN / SPMamba.

**What survives is an efficiency claim.** The scan reaches that shared accuracy ceiling at 48.7 K
parameters versus 95.9 K for the global-context fix and 72.1 K for the baseline, with the lowest
latency and 2.3× less memory. FourierMamba spends 23.6 M params / 260 GFLOPs / 153 ms on the same
operator; this reaches it 484× smaller. That is real and publishable — as an efficiency result, not a
modelling one.

**The strongest result is a piece of theory, not an architecture.** By the dual of the convolution
theorem, a K×K convolution over the *spectrum* is exactly a multiplication of the *image* by
`iDFT(k)` — an apodization, not a filter. Under DFFN's softmax that mask satisfies `|m(x,y)| ≤ 1`
everywhere, so the frequency branch is a **vignette that can only attenuate**; even a Hann window needs
signed taps softmax cannot express. Verified to 10⁻¹⁴ (§2.6). The sweep found no vision or restoration
paper stating this — it costs no experiments and it explains the NR-specific weakness.

**And one result argues against reaching for an SSM at all:** in a controlled receptive-field study,
plain **dilated depthwise convolutions beat a 4-direction selective scan (+29.5 % vs +20.7 % error
reduction) at 6.4× the speed** [measured, independently reproduced]. The decisive untested experiment
is therefore a *dilated frequency-domain filter* — cheap, and it may dominate everything above.

---

## 1. Where the project stands

| Model | Params | GFLOPs | Latency | 50 mm SR/NR/DA | 150 mm SR/NR/DA |
|---|---|---|---|---|---|
| FourierMamba2D (adapted) | 23.59 M | 260.3 | 153 ms | .967 / .906 / .936 | .921 / .908 / .915 |
| **DFFN** | **72.1 K** | **5.36** | **10.8 ms** | .965 / .868 / .914 | .916 / .876 / .896 |
| DFFN-SEKG | 73.2 K | 5.36 | 10.9 ms | — | — |
| DFFN-FDConv | 48.3 K | 4.20 | 15.2 ms | — | — |

DFFN reaches ~98 % of FourierMamba's DA at 1/327 the parameters. **The gap is almost entirely NR**:
SR is level (.965 vs .967), NR is 3.8 points lower. *DFFN retains signal as well as FourierMamba; it
fails to remove rain as well.* §2 shows this follows from how the operator is parameterised.

Beyond the deck: `PointDFFNet` and `PointDFFNet_cufinufft` already exist — the "Into 3D" direction,
lifting the block to an `(x,y,t)` voxel grid with `rfftn`, plus NUFFT experiments.

**Benchmark provenance** (from the sweep): the 50/150 mm data is **EventRain-27K**, introduced by
PRE-Mamba (arXiv:2505.05307); `DA = ½(SR + NR)`, SR = recall on background events, NR = recall on rain
events.

---

## 2. Diagnosis — measured

### 2.1 The cost is in *predicting* the filter, not applying it

256×256, dim 32, 4 blocks, RTX A5000 [measured]:

| stage (one block) | ms | share | | analytic MACs / block | MMACs | share |
|---|---|---|---|---|---|---|
| **dynamic filter (unfold) ×2** | **0.631** | **24.3 %** | | **FGN 1×1 head (16→576)** | **304.3** | **45.6 %** |
| **softmax over 9 taps ×2** | **0.409** | **15.7 %** | | FFN 1×1 ×2 | 268.4 | 40.2 % |
| **FGN** | **0.393** | **15.1 %** | | FFN dw 3×3 | 37.7 | 5.6 % |
| FFN + SE | 0.359 | 13.8 % | | FGN 1×1 in | 33.8 | 5.1 % |
| LayerNorm | 0.276 | 10.6 % | | **filter mul-add ×2** | **19.0** | **2.8 %** |
| rfft2 + irfft2 | 0.309 | 11.9 % | | FGN dw 3×3 | 4.8 | 0.7 % |

1. **The K²=9 filter machinery is 55 % of block latency** (unfold + softmax + FGN). The single 1×1 conv
   emitting `C·K²·2 = 576` channels is **45.6 % of all MACs**; *applying* the filter is 2.8 %.
2. **Memory-bandwidth bound, not FLOP bound.** 5.35 GFLOPs should take ≈0.06 ms on this GPU; it takes
   11.8 ms — ~200× off roofline. `F.unfold` materialises `[B,32,9,256,129]` = 36.3 MiB, twice per block
   (279 MiB peak for *one* block at B=1). Batching does not amortise: B=1 → 11.8 ms, B=4 → 49.5 ms.
3. **FFT + iFFT are only 11.9 %.** Anything justified by "remove the FFT round trips" is chasing ~9 %
   — and §6.2 shows that redesign is a net loss.

### 2.2 The softmax makes the filter strictly smoothing

Softmax over the K² taps forces non-negative weights summing to 1, so the filtered magnitude at an
interior bin is confined to the **convex hull of its 3×3 spectral neighbourhood** [measured]:

- Best-case retained magnitude (neighbourhood min ÷ own value): median **0.36**; **0.21** on the
  top-1 % rain-energy bins. An isolated spectral peak cannot be pushed below its neighbours.
- Cancelling via the additive residual needs the anti-phase `φ+π`. That lies inside the neighbourhood's
  `[min, max]` for only **78.6 %** of bins — for **21.4 %**, no softmax setting can produce it.

This is a known trade-off, not an oversight: Bako et al. (KPCN, SIGGRAPH 2017) introduced
softmax-over-taps precisely to bound output inside the input's convex hull; Mildenhall et al. (Burst
Denoising with KPN, CVPR 2018) explicitly removed it so kernels could go negative. DFFN inherited the
constrained variant — and its measured weakness is NR, the metric for *removing* things.

The clean DSP statement: if `h_k ≥ 0` and `Σh_k = 1` then `|H(e^{jω})| ≤ Σ_k h_k = 1 = H(e^{j0})` for
all ω. **DC gain is pinned at exactly 1 and no frequency can be amplified above DC** — structurally a
moving average. Published ablations quantify the cost of this choice elsewhere: adding a softmax to
ConvNeXt's depthwise conv costs **−21.4 % top-1** and never recovers (DCNv4, arXiv:2401.06197);
Involution reports **>1 % top-1 drop** from softmax/sigmoid on the kernel generator (arXiv:2103.06255);
DKN rejects it outright — kernels "should be similar to high-pass filters, with weights adding to 0"
(arXiv:1910.08373).

### 2.3 Phase is filtered as if it were linear — a correctness bug

The block applies a softmax-weighted average to `torch.angle(...)`, wrapped on (−π, π]. Against the
correct circular mean, at the *most favourable* simplex point (uniform weights), real 150 mm data
[measured]:

| discrepancy | share of bins |
|---|---|
| > 0.5 rad | **70.3 %** |
| > 1.0 rad | **51.0 %** |
| > 2.0 rad | 24.2 % |

Mean error 1.22 rad (70°), median 1.03 rad, max π. Skewed weights make it worse. **Fix this regardless
of everything else here** — carry `(real, imag)`, or filter `(cos φ, sin φ)` and recombine via `atan2`.
Cost: nothing.

This is a *bug fix*, not a contribution — filtering the complex spectrum is the field default (FFC,
GFNet, AFNO). SFHformer (ECCV 2024) already applies a dynamic per-bin local convolution to the complex
spectrum held as real/imag channels.

### 2.4 Rain's spectral signature is real, directional, and global

Per-sample, n = 24 per rate [measured]:

| | 50 mm | 150 mm |
|---|---|---|
| rain energy inside its own top-5 % of bins | **37.0 %** | 33.5 % |
| *clean signal* energy inside its own top-5 % (reference) | 19.0 % | 19.0 % |
| top-3-of-12 orientation sectors | **45.4 %** (uniform 25 %) | 45.4 % |
| dominant orientation sector, per frame | sector 0, std **0.00** | sector 0, std **0.00** |

Rain is ~2× more spectrally concentrated than the scene, and its orientation is identical in every
frame. **This quantitatively validates the deck's premise that the frequency domain carries a strong
rain prior** — specifically an *orientation* prior, which is a **global, per-frame** property. A 3×3
window over frequency bins is structurally blind to it.

This matches the physics. Barnum et al. (IJCV 2010, DOI 10.1007/s11263-008-0200-2) give the only closed
form for rain in Fourier space: a streak's transform has an **isotropic Gaussian envelope
`exp(−b²(u²+v²))` set by drop breadth**, with energy peaking on a line through DC — an oriented ridge,
*not* a high-frequency shell. Barnum explicitly notes rain "covers such a broad part of the frequency
space", which is exactly what I measure (only 37 % of rain energy in its top 5 % of bins; the bulk
spread across mid frequencies). **Any design assuming rain lives in a narrow high-frequency band is
contradicted by both the physics and this data.**

### 2.5 Temporal structure: carry statistics, not signal

Consecutive frames, n = 40 [measured]:

| | 50 mm | 150 mm |
|---|---|---|
| rain-spectrum correlation, lag 1 | 0.481 | 0.633 |
| scene-spectrum correlation, lag 1 | 0.815 | 0.815 |
| **orientation-histogram correlation, lag 1** | **0.983** | **0.990** |

The obvious temporal hypothesis is **false**: rain's per-bin spectrum is *less* temporally correlated
than the scene's, so carrying the raw spectrum forward would hurt. But rain's *low-dimensional
statistics* are near-constant. **Temporal state should condition the filter generator, not the signal
path** — with exactly the capacity of a summary, which is what an SSM state `[d, N]` is (4.0 KiB at
d = 64, N = 16 [measured]).

Barnum's model predicts precisely this: it is **constant in temporal frequency** — rain is white along
time — so no temporal spectrum is available, and the only cross-window discriminative signal is the
*persistence of ridge orientation*. That is the physical justification for a temporal state, and the
reason a temporal FFT would buy nothing.

### 2.6 What DFFN's spectral 3×3 actually *is* — and this is the best result here

By the dual of the convolution theorem, convolving the spectrum with a kernel `k` is **exactly**
multiplying the image by `iDFT(k)`. For a 3×3 kernel over offsets `(du,dv) ∈ {−1,0,1}²`:

```
m(x,y) = Σ_{du,dv} w_{du,dv} · exp(2πi(du·x/H + dv·y/W))
```

— a 9-term trigonometric polynomial, i.e. a smooth, image-wide, one-cycle-across-the-frame complex
gain. **Verified numerically to 1.0×10⁻¹⁴** [measured, `apodization.py`].

So a K×K convolution over the spectrum is **an apodization — a window/vignette on the image — not a
frequency filter.** The textbook identity confirms it from the other side: applying a **Hann window**
is implemented exactly by convolving the spectrum with `(−¼, ½, −¼)`, reproduced here to 7.6×10⁻¹⁵.

Two consequences, both sharp:

1. **Under softmax, `|m(x,y)| ≤ 1` everywhere with equality only at the origin** — measured range
   `[0.003, 1.000]`, mean ≈ 0.36. DFFN's frequency branch can only *attenuate*, as a smooth vignette.
   It is doing **spectral leakage control, not deraining.**
2. **Even the Hann window needs signed taps** `(−¼, ½, −¼)` — which softmax structurally cannot
   represent. The operator cannot express the one classical window it most resembles.

Caveat, stated precisely: the identity is exact for taps held constant across the spectrum. DFFN's taps
vary per bin, so the true operator is frequency-index-varying — a *smoothly-varying* gain field rather
than a single global vignette. It remains far closer to "a multiplicative mask on the image" than to
"a filter of the spectrum". Measuring how fast a trained model's taps vary across bins would quantify
the gap, and is a cheap experiment.

**This also explains §4.3**: a per-pixel spatial gain predictor performs well precisely because a gain
field is what this block was computing all along — just by an expensive and constrained route.

**The literature sweep found no vision or restoration paper stating this.** MRI has the analogous
analysis (SENSE/GRAPPA duality, RAKI); vision does not. It is theory at zero experimental cost, and it
converts the project's worst limitation into its most defensible contribution.

---

## 3. Literature positioning

Six parallel surveys + five adversarial prior-art examinations. Verdicts:

| Idea | Verdict | Closest prior art |
|---|---|---|
| Selective scan over frequency bins **as the filter** | **ALREADY_DONE** | FourierMamba arXiv:2405.19450 (**= your model A**); OSMamba arXiv:2411.15255; PAS-Mamba arXiv:2601.14530; BSRNN / SPMamba (audio) |
| Filter complex spectrum instead of wrapped phase | **ALREADY_DONE** | SFHformer ECCV 2024; PHASEN AAAI 2020; complex ratio masking TASLP 2016 |
| Single-FFT trunk | **ALREADY_DONE** | T1 NeurIPS 2022; FCNN 2017; arXiv:2410.04342 |
| Polar / angle-major scan order | INCREMENTAL | no spectral SSM is angle-major — but §6.1 refutes the premise empirically |
| SSM state across time windows conditioning the generator | **INCREMENTAL** | Zubic et al. event-SSM; PRE-Mamba; Meta-AF / DeepFilterNet — the narrow conjunction is unoccupied |

Supporting theory the sweep confirmed: a diagonal SSM's kernel is a sum of damped complex exponentials
— a bank of one-pole IIR filters (S4D, NeurIPS 2022); the transfer-function view is formalised in RTF
(ICML 2024); input-dependent IIR filters with a tiny coefficient-emitting head are Lutati et al.
(EMNLP 2023). **Every element of the "SSM as a rational filter over the spectrum" argument is already
in print.**

### Two corrections to my own initial framing

1. **"A diagonal SSM can notch" was overstated.** With `H(z) = Σₙ CₙB̄ₙ/(z − λₙ) + D`, a single real
   pole with `D = 0` has no finite zero — a one-pole low-pass that cannot null a bin. Transmission
   zeros need `N ≥ 2` with complex λ, or a non-zero `D` skip. (The implementation here uses N = 16 and
   `D ≠ 0`, so it does have zeros — but state it as *"poles plus the D-skip give a rational response
   with zeros"*.)
2. **Filtering along the frequency *index* is not filtering the image.** By duality, exponential decay
   in the frequency variable is a Lorentzian convolution in space — a global halo. "Notching the
   spectrum" and "the spectral-axis IIR having a zero" are different things.

---

## 4. Measured results of the proposed block

`FSSNet` replaces the K² frequency filter with a bidirectional selective scan whose (Δ, B, C) come from
a tiny conv generator, and carries the spectrum as `(real, imag)` so §2.3 disappears.
**Generator head: `d + 2N = 96` output channels instead of `C·K²·2 = 576` — 6× narrower** [measured].

### 4.1 Cost

| model | params | B=1 | B=4 | peak mem (B=4) |
|---|---|---|---|---|
| DFFN baseline | 72.1 K | 11.77 ms | 49.54 ms | 1194 MiB |
| **FSSNet** | **48.7 K** | **9.77 ms** | **33.29 ms** | **515 MiB** |
| | −32 % | 1.21× faster | 1.49× faster | 2.3× less |

### 4.2 Accuracy — controlled pilot

Identical data (4 rain rates, 1556 frames), loss (BCE on the binary event mask), optimiser, seed,
10-epoch schedule. Validated on two **held-out** rain rates (20 mm, 80 mm). DA threshold-swept so each
model is scored at its own best operating point [measured]:

| arm | params | best mean DA | train wall | inference (B=1) |
|---|---|---|---|---|
| A. DFFN baseline (K²=9 unfold) | 72,074 | 0.8173 | 11.3 min | 11.77 ms |
| **B. FSSNet (frequency selective scan)** | **48,650** | 0.8346 | **9.3 min** | **9.77 ms** |
| C. FSSNet + polar scan order | 48,650 | 0.8359 | 14.1 min | ~10 ms |
| D. DFFN + global spectral context, **no SSM** | 95,946 | **0.8362** | 11.3 min | ~12 ms |

**This is the most important result in the document, and it is not the one I expected.**

All three interventions land at the *same* accuracy — B 0.8346, C 0.8359, D 0.8362, a spread of
**0.0016 at one seed, i.e. indistinguishable** — and all three beat the baseline by ≈ **+1.8 DA
points**. The common factor is not the state space model. It is that **the filter generator gains
access to global spectral structure**. A 72-dimensional orientation × radius histogram plus a two-layer
MLP (arm D) buys exactly as much accuracy as a bidirectional selective scan.

So the honest conclusion:

- **The accuracy gain is attributable to global spectral context, not to the SSM.** §2.4 predicted
  this (rain's signature is a global, per-frame orientation prior that a 3×3 window cannot see); §4.3
  corroborates it (positional long-range context is what helps).
- **The SSM's remaining justification is purely efficiency**: it reaches that accuracy with
  **48.7 K parameters vs arm D's 95.9 K** (and the baseline's 72.1 K), at 1.2–1.5× lower latency and
  2.3× less peak memory. That is a real and defensible win — but it is an efficiency claim, not a
  modelling one, and it must be written that way.

⚠️ **Caveats, plainly:** single seed; 10 epochs; 4 of 14 training rain rates; absolute DA (~0.836) is
well below the deck's ~0.91 because this is a short pilot with an untuned loss, **not a reproduction**.
The *comparison* is controlled; the *absolute numbers* are not competitive results. Differences among
B/C/D are inside noise; only the ~+1.8 gap to A is large enough to trust at one seed. C vs B (+0.0013)
independently confirms §6.1 — scan ordering does nothing.

### 4.3 What is actually doing the work — the uncomfortable part

FSSNet changes **three** things at once (scan instead of K²; complex instead of wrapped phase; narrower
head). A receptive-field study isolates the mechanism: all variants predict a per-pixel gain and differ
*only* in the predictor's receptive field [measured, re-run and reproduced independently]:

| predictor | params | ms | test NMSE | vs local |
|---|---|---|---|---|
| local 3×3 dw (what the FGN has) | 481 | 0.29 | 0.4607 | — |
| **dilated 3×3 dw (1/4/16)** | 801 | **0.31** | **0.3249** | **+29.5 %** |
| local + global average pool | 1,009 | 0.36 | 0.4401 | +4.5 % |
| local + 4-direction selective scan | 2,465 | 1.99 | 0.3653 | +20.7 % |

- **Long-range *positional* context is what matters** — a single pooled global number buys almost
  nothing (+4.5 %). Consistent with §2.4: the useful signal is *where* energy sits, not how much.
- **A selective scan is not the cheapest way to buy it.** Dilated depthwise convolution is better *and*
  6.4× faster.

Scoping matters: this study is in the **spatial** domain, whereas §4.2 shows that in the **frequency**
domain the scan *does* beat the K² filter. Both can be true. But the burden is now on showing the
frequency-domain scan beats a *dilated frequency-domain filter* — which nobody has tested. **That is
the single most valuable next experiment.**

---

## 5. Ranked proposals

**P0 — Write up the apodization result (§2.6).** *(zero experiments, highest defensibility)*
A K×K convolution over the spectrum is an image-domain window, not a filter; under softmax it is a
vignette that can only attenuate; even a Hann window needs signed taps softmax cannot express. Verified
to 10⁻¹⁴. No vision or restoration paper states this. It explains DFFN's NR-specific weakness, motivates
P2 and P3, and stands on its own.

**P1 — Fix the phase branch cut.** *(today, independent of everything else)*
Carry `(real, imag)`. Zero cost; removes a defect affecting 51 % of bins. A bug fix, not a contribution.

**P2 — Drop the softmax over taps.**
The convex-combination constraint is a plausible cause of the NR-specific gap (§2.2, §2.6); removing it
is a one-line change with published precedent (KPN, CVPR 2018). Cheapest possible test — run it
*before* any architecture work. Ranked drop-ins, cheapest first:
1. **Softmax over kernel *bases*, not taps** (SFHformer's FDC, DFFormer) — preserves DFFN's exact
   structure and stability while leaving the effective kernel sign-unconstrained. Start here.
2. **Mean-subtraction → sum-to-zero signed taps** + residual (DKN, IJCV 2021) — makes it a high-pass.
3. **BN-style filter normalisation** `α(Ď−μ)/σ + β` (DDF, CVPR 2021) — bounded, zero-mean, signed.
4. **Sigmoid gain instead of sum-to-one** (CondConv, ODConv, DCNv2 modulation).
5. **Temperature annealing** on the existing softmax — a two-line experiment worth running first.

**P3 — Give the filter generator global spectral context.** *(this is where the accuracy is)*
Arm D: a 12 × 6 orientation × radius histogram + 2-layer MLP, concatenated to the FGN input.
**+1.9 DA points, ~30 lines of code, no SSM, no new dependency.** This is the single highest
value-per-effort change in the document, and §2.4/§4.3 both explain why it works. Do this before P4.

**P4 — Replace the K² filter with a selective scan (FSSNet), positioned strictly as an efficiency
result.** Same accuracy as P3 (0.8346 vs 0.8362, inside noise) at **half the parameters** (48.7 K vs
95.9 K), lowest latency, 2.3× less memory. Frame as *"a 484×-smaller instantiation of FourierMamba's
frequency scan, for event streams, head-cost-matched against a K² spectral dynamic filter"* —
**never** as a new operator, and **never** as an accuracy contribution; the control arm ties it.
Mandatory ablations: dilated frequency-domain filter (§4.3), global-context-only (arm D), head-cost
table. A reviewer holding FourierMamba will challenge novelty immediately.

⚠️ **Risk to budget before committing:** diagonal SSMs carry an innate **low-frequency bias fixed at
initialisation that ordinary training does not remove** (ICLR 2025 arXiv:2410.02035; NeurIPS 2025
arXiv:2508.20441). Rain is broadband and anisotropic (§2.4), so this is first-order, not a footnote —
mitigations are init scaling, Sobolev-norm gradient reweighting, or a DFT-domain diagonal init
(S4D-DFouT). Separately, selective scans are memory-bandwidth bound; my 1.2–1.5× speedup holds at this
resolution and width but will not survive naive scaling — use Mamba-2/SSD chunked matmuls or a
minGRU-class `d_state=1` recurrence if it regresses.

**P5 — Temporal state conditioning the generator.** *(most novel remaining slot)*
§2.5 says how: carry the SSM state across event time windows and feed it to the *filter generator*, not
the signal path — the past enters as the generator's context, not as extra scanned tokens. Costs 4 KiB
and no extra FLOPs/frame, and enables genuine streaming inference. State it narrowly: **not** "first
streaming event restoration" (false), but that specific conjunction.

**P6 — Carry this into 3-D / PointDFFNet, where the payoff is larger.**
In `PointDFFNet` the FGN head is `16 → C·K³·2 = 1728` channels; **78 % of the model's 157.9 K parameters
sit in filter generators**, and it takes 57 ms for 40,960 events at B=2 on a *dense* 48×48×8 grid
regardless of sparsity [measured]. The scan substitution is **18× narrower** here (96 vs 1728) versus
6× in 2-D, and the 3-D spectrum is only 11,520 bins so the scan is cheap. Strongest engineering case in
the project, and it is the deck's own stated direction.

---

## 6. Refuted — do not re-try

### 6.1 Polar / angle-major scan ordering — refuted by measurement

Per-sample fragmentation of the top-5 % rain-energy bins [measured]:

| scan order | runs (50 mm) | mean run len | runs (150 mm) | mean run len |
|---|---|---|---|---|
| raster | 1131 | 1.46 | 1124 | 1.47 |
| zigzag (FourierMamba) | 1129 | 1.46 | 1121 | 1.47 |
| polar (angle, radius) | 1165 | **1.42** | 1167 | **1.42** |
| orientation mod π | 1161 | 1.42 | 1158 | 1.43 |

Polar is *slightly worse* than raster at both rates. Mean run length ≈1.4 under **every** ordering:
high-rain-energy bins are essentially isolated. **No 1-D ordering of this spectrum makes rain
contiguous.** The training pilot agrees (+0.0013 DA, noise). This also undercuts the usual
justification for FourierMamba's zigzag on event data — zigzag is no better than raster here.

### 6.2 Single-FFT trunk — refuted by measurement *and* already published

All blocks resident in the frequency domain (1×1 conv exact; dw 3×3 → learned per-bin mask;
squeeze-excite → gate off the DC bin, since spatial GAP *is* the DC coefficient):

| model | params | B=1 | B=4 | #FFTs |
|---|---|---|---|---|
| DFFN baseline | 72.1 K | 11.77 ms | 49.54 ms | 8 |
| FSSNet per-block FFT | 48.7 K | **9.77 ms** | 33.29 ms | 8 |
| FSSNet single-FFT trunk | **17.0 M** | 17.55 ms | 31.40 ms | 2 |

**Slower at B=1 and 236× larger** — a full-resolution per-bin mask costs 4.2 M params/block. The FFT
pair is 11.9 % of block time, so the ceiling was ~9 % anyway. Also `ALREADY_DONE` (T1, NeurIPS 2022).

---

## 7. Suggested experiment order

1. **P1 + P2 + P3** (a day or two). Phase fix, drop the softmax, add the global spectral descriptor.
   P3 alone already recovers the full measured gain. If dropping the softmax also moves NR, §2.2
   becomes the paper's core claim and the story simplifies enormously.
2. **Dilated frequency-domain filter**, identical budget, against arms B and D. Given §4.3 this may
   dominate both — it is cheap and it decides whether the SSM survives review. **Run before writing
   anything.**
3. **Repeat the pilot with ≥3 seeds.** B/C/D differ by 0.0016; nothing among them is currently
   distinguishable, and the ranking could reorder.
4. **Disentangle FSSNet's three changes** — scan vs complex representation vs narrow head — one at a
   time against arm D, to attribute the parameter saving correctly.
5. **Full-schedule run**, all 14 rain rates, deck protocol, to turn §4.2 into a competitive number.
6. **P5 streaming state**, then **P6 in 3-D**, where the head-cost argument is 18× rather than 6×.

Three further openings the sweep flagged that cost little:

- **Free discriminator you are currently discarding.** The resultant length `|Σ_k w_k e^{jφ_k}|` is the
  circular concentration of the local phase neighbourhood — a physically meaningful rain-vs-shot-noise
  cue that `atan2` throws away. Feed it to the FFN or a gate. More likely to buy accuracy than the
  wrap fix itself.
- **A clean global-vs-local-vs-both spectral ablation in deraining is unclaimed.** The only
  apples-to-apples study (GLFNet, arXiv:2403.00396) is medical segmentation, and it finds
  global-only ≈ local-only < **both**. Your arms A/B/D are three quarters of that experiment already.
- **Frequency-domain contrastive regularisation** using rain streaks as *negative* samples (FADformer,
  ECCV 2024) is free here — the synthetic pipeline already gives paired rainy/clean tensors. For
  reference, PRE-Mamba's `L_fft` is worth only ~+0.15 DA points, so there is room.

---

## 8. Reproducing

Everything below is copied into **`dffn_ssm/`** in this repo (the originals were in a job scratch
directory that gets cleaned up). Raw logs `train.log`, `where_ssm.log`, `train_results.json` are there too.

| Script | Produces |
|---|---|
| `profile_dffn.py` | §2.1 stage latency, MAC breakdown, unfold memory |
| `expressivity.py` | §2.2 suppression floor & anti-phase reachability, §2.3 phase-wrap error |
| `scan_order2.py` | §2.4 concentration/orientation, §6.1 fragmentation |
| `temporal.py` | §2.5 temporal correlations |
| `fss_model.py` | the proposed FSSNet block |
| `proto_fss.py`, `full_arch.py` | §4.1 cost, §6.2 trunk comparison |
| `train_compare.py`, `dffn_global.py` | §4.2 controlled pilot (4 arms) |
| `where_ssm_helps.py` | §4.3 receptive-field study *(authored by the parallel session; re-run here)* |
| `apodization.py` | §2.6 apodization identity + Hann cross-check |
| `literature_synthesis.md` | full 12-agent literature synthesis with per-claim URLs |

---

## 9. Provenance

A second Claude session ran concurrently in this workspace and authored `DFFN_SSM_PROPOSAL.md` and
several scratch scripts (`spectral_analysis.py`, `ablation_features.py`, `operator_bakeoff.py`,
`temporal_and_phase.py`, `scan_cost.py`, `why_it_works.py`, `where_ssm_helps.py`).

Everything tagged **[measured]** in *this* file was produced by scripts I wrote and ran, with two
exceptions, both stated: §4.3 uses the parallel session's `where_ssm_helps.py`, which **I re-ran
independently and whose direction I reproduce** (dilated +29.5 % vs scan +20.7 %, dilated 6.4× faster;
the parallel run reported +26.5 % / +19.0 % / 8.8× at a different epoch count and GPU). Claims that
appear in `DFFN_SSM_PROPOSAL.md` but not here — including its R² figures and its L40S latency table —
were **not** verified by me and should be re-checked before use.

The literature verdicts in §3 come from a 12-agent sweep I commissioned; each verdict is traceable to
`journal.jsonl` in the workflow transcript directory, with per-claim URLs.
