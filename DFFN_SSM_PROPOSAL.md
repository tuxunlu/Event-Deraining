# Dynamic Fourier Filtering × State Space Models for Event Deraining

*Diagnosis, measured evidence, literature positioning, and ranked proposals.*
*Prepared 2026-07-28. Every number labelled "measured" was produced on this machine against this project's own code and the real paired data in `dataset/efft_results`. Scripts listed in §8.*

---

## 0. Executive summary

The request was: combine the Dynamic Fourier Filter Network with SSM/Mamba to make it faster **and** better. Short answer: **yes, and a working prototype already beats the baseline in a controlled training pilot** — but the reason it wins is probably *not* the reason one would assume, and that distinction is where the publishable contribution lives.

> # ⚠ FINAL RESULTS — read this instead of §4e and §5
>
> The pilot leaderboard that most of this document is built on **excluded FourierMamba2D**, the project's own 23.59 M-parameter baseline. It has now been run twice — head-to-head on the pilot protocol, and a 50-epoch run on the real test split. Both change the conclusions.
>
> ### 1. Head-to-head, identical pilot protocol
>
> | model | params | mean DA | train wall |
> |---|---|---|---|
> | **FourierMamba2D** | 23,592,081 | **0.9133** | 174 min |
> | ORSPNet | 36,206 | 0.8683 | 8 min |
> | SpatialGainNet | 52,714 | 0.8509 | 7 min |
> | FSSNet (SSM) | 48,650 | 0.8346 | 9 min |
> | DFFN | 72,074 | 0.8173 | 11 min |
>
> **FourierMamba wins by +0.045 (≈11× seed noise) and beats every small model on both SR *and* NR at both rain rates.** Nothing built in this document beat it.
>
> ### 2. Full 50-epoch protocol, train → validation-select → test/{50,150} mm
>
> | model | params | 50 mm DA | 150 mm DA | test mean DA |
> |---|---|---|---|---|
> | **ORSPNet** | **36,206** | .9325 | .9062 | **0.9193** |
> | DFFN | 72,074 | .9298 | .9020 | 0.9159 |
>
> **+0.0034 — inside ORSPNet's ±0.0039 seed spread. A statistical tie at half the parameters**, not the +0.051 the pilot suggested.
>
> ### 3. The deck's numbers are NOT comparable to these
>
> My DFFN run scores 0.9159 against the deck's 0.905 — but with NR +0.039/+0.044 higher and SR −0.013/−0.032 lower. That signature indicates a different thresholding convention (this harness sweeps τ to maximise DA; the deck presumably used a fixed threshold). **Do not compare any number in this document to the deck's table.**
>
> ### What is retired
>
> * **"Per-bin frequency filtering is the weak operator class" (§4e) — dead.** It is weak at a 36–96 K budget; at 23.6 M the same operator class is the best thing here. §4e is a statement about small models, not operators.
> * **"The Fourier inductive bias is not what carries event deraining" — dead.**
> * **ORSPNet's accuracy advantage — dead.** It ties DFFN and loses to FourierMamba.
>
> ### What survives
>
> * **The efficiency result:** ORSPNet reaches within 4.5 DA points of FourierMamba at **652× fewer parameters, 78× fewer FLOPs, 22× faster training**, and ties DFFN at half its parameters. That is the contribution.
> * **Every diagnostic finding in §2 and §5.5–5.6** — phase wrapping (~70 % of bins), the convex-combination limit (34–39 % of bins need gain > 1), rain's non-transferable spectrum (k=16 → 0.6–4.5 % held-out variance), no 1-D ordering making rain contiguous, zigzag ≈ raster. These are measurements about the data and the block, independent of which model wins.
> * **The SSM verdict:** FSSNet placed mid-pack and a plain histogram matched it. The SSM contributed nothing beyond parameter efficiency.

**The one-line answer.** A controlled 4-arm training pilot on the real SR/NR/DA metric says: **yes, +1.7–1.9 DA over the DFFN baseline is available — but the SSM is not what delivers it.** A plain global spectral histogram with no state space model anywhere scores just as well (0.8362 vs 0.8346). What the SSM *does* deliver is the same accuracy at **half the parameters** (48.7 K vs 95.9 K, and 32 % below the 72.1 K baseline) with the fastest training of the four. **Adopt it for efficiency, not for accuracy — and do not build the paper's story on "we added Mamba", because that story is already published as FourierMamba, your own baseline (§3.3).**

Five things were established by measurement:

1. **DFFN's cost is in *predicting* the filter, not applying it.** The single 1×1 convolution that emits `C·K²·2 = 576` filter taps is **45.6 % of all MACs**; actually applying the filter is 2.8 %. Softmax + unfold are **56 % of block latency** and are memory-bound (72.6 MiB materialised per block).
2. **Three concrete defects in the operator**, all on the *removal* side — which is exactly where DFFN's metrics lag (NR −0.038 vs FourierMamba, while SR is level):
   * the softmax makes the filter a convex combination, so it can only smooth — yet **34–39 % of frequency bins require amplification**, and **16–20 %** need a value outside their own 3×3 neighbourhood;
   * phase is filtered as if it were linear, but **~70 % of 3×3 phase neighbourhoods** straddle the ±π branch cut (mean error 69°);
   * the filter generator is a CNN and is therefore **structurally blind to absolute frequency position**, worth **+11–13 % R²** for free.
3. **A prototype fixing all three and replacing the K² kernel with a selective scan (FSSNet) reaches 0.8346 mean DA vs the baseline's 0.8173, with 32 % fewer parameters and the fastest training.** See §4.
4. **But the no-SSM control matches it.** Arm D — the same DFFN with a global orientation×radius histogram fed to its filter generator — scores 0.8362. The mechanism that matters is *information beyond the 3×3 spectral window*, not the sequence model. Two independent isolated experiments agree: no 1-D ordering makes rain contiguous (§6.1), and dilated convs beat a selective scan for receptive field at **8.8× the speed** (§5.3).
5. **Polar scan ordering is closed** — tied on accuracy, 52 % slower, and already published twice in 2026 (§6.1).

**On novelty:** an adversarial prior-art pass returned **ALREADY_DONE** on three of four candidate ideas — the SSM-over-spectrum operator *is* FourierMamba's `FourScan`, i.e. this project's own baseline (§3.3). What survives is the efficiency result and the negative results. §3.4 states the defensible claim.

---

## 1. Where the project stands

| Model | Params | GFLOPs | Latency | 50 mm SR/NR/DA | 150 mm SR/NR/DA |
|---|---|---|---|---|---|
| FourierMamba2D (adapted) | 23.59 M | 260.3 | 153 ms | .967 / .906 / .936 | .921 / .908 / .915 |
| **DFFN** | **72.1 K** | **5.36** | **10.8 ms** | .965 / .868 / .914 | .916 / .876 / .896 |
| DFFN-SEKG | 73.2 K | 5.36 | 10.9 ms | — | — |
| DFFN-FDConv | 48.3 K | 4.20 | 15.2 ms | — | — |

DFFN reaches ~98 % of FourierMamba's DA at **1/327 the parameters and 1/49 the FLOPs**. The gap is almost entirely **NR**: SR is level (.965 vs .967) while NR is 3.8 points lower. *DFFN retains signal as well as FourierMamba; it fails to remove rain as well.* §2 shows this is a direct consequence of the operator's parameterisation, and §4 shows it is fixable.

Beyond the deck, `PointDFFNet` and `PointDFFNet_cufinufft` already exist in `/fs/nexus-scratch/tuxunlu/git/Event-Deraining` — the "Into 3D" direction, lifting the block to a 3-D `(x,y,t)` grid with `rfftn`, plus NUFFT experiments.

**Benchmark provenance** (confirmed): the 50/150 mm data is **EventRain-27K**, introduced by PRE-Mamba (ICCV 2025, arXiv:2505.05307) — synthetic subset built from KITTI + SPAC video, rain rendered, converted to events with Vid2E, 5–200 mm/hr. The metrics are PRE-Mamba §5.1 verbatim: `DA = ½(SR + NR) = ½(PB/TB + PR/TR)`, i.e. **SR = recall on background events, NR = recall on rain events**. PRE-Mamba's window convention is 0.1 s × 5 windows — the same ~104 ms window this project uses.

---

## 2. Diagnosis — measured, not assumed

### 2.1 The cost is in predicting the filter

Analytic MACs per block (dim 32, 256×256; the total, 4 blocks = 5.345 GFLOPs, cross-checks against the deck's reported 5.3626):

| component | MMACs | share |
|---|---|---|
| **FGN 1×1 head (16 → C·K²·2 = 576)** | **304.3** | **45.6 %** |
| FFN 1×1 (C→e) | 134.2 | 20.1 % |
| FFN 1×1 (e→C) | 134.2 | 20.1 % |
| FFN dw 3×3 | 37.7 | 5.6 % |
| FGN 1×1 in (2C→h) | 33.8 | 5.1 % |
| **dynamic filter mul-add ×2** | **19.0** | **2.8 %** |
| FGN dw 3×3 | 4.8 | 0.7 % |

The filter *generator* is 51.3 % of MACs; applying the filter is 2.8 %.

Measured stage latency, one block, B=1, L40S:

| stage | ms | share |
|---|---|---|
| dynamic filter (unfold) ×2 | 0.340 | 31.6 % |
| softmax ×2 | 0.266 | 24.8 % |
| FGN | 0.128 | 11.9 % |
| FFN + SE | 0.114 | 10.6 % |
| LayerNorm | 0.086 | 8.0 % |
| rfft2 | 0.056 | 5.2 % |
| abs+angle+cat | 0.035 | 3.2 % |
| irfft2 | 0.027 | 2.5 % |
| complex recombine | 0.023 | 2.2 % |

* **Softmax + unfold = 56.4 % of latency, and they are memory-bound**: `F.unfold` materialises `[B,32,9,256,129]` = 36.3 MiB, twice per block (72.6 MiB measured peak for one block at B=1).
* **FFT + iFFT together are 7.7 %.** Anything justified by "remove the FFT round trips" is chasing 7.7 % — and §6.2 shows the full redesign is a net loss.

### 2.2 The softmax makes the operator strictly smoothing

Softmax over the K²=9 taps forces non-negative weights summing to 1, so the filtered magnitude lies in the **convex hull** of its 3×3 spectral neighbourhood. It can smooth; it cannot notch, amplify, or go negative.

Measured ideal per-bin gain `|clean|/|rainy|`:

| | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| 50 mm | 0.127 | 0.282 | 0.619 | 0.882 | 1.182 | 2.225 | 4.688 |
| 150 mm | 0.087 | 0.198 | 0.485 | 0.776 | 1.167 | 2.473 | 5.399 |

* **Amplification (gain > 1) is required for 38.5 % (50 mm) / 33.8 % (150 mm) of bins.**
* The target lies **outside** the neighbourhood's `[min,max]` for **15.8 % / 19.5 %** of bins.
* Best-case retained magnitude (neighbourhood min ÷ own value): median **0.36**, and **0.21** on the top-1 % rain-energy bins — an isolated spectral peak cannot be pushed below its neighbours.

This is a known, named trade-off: Bako et al. (KPCN, SIGGRAPH 2017) introduced softmax-over-taps *precisely* to bound output within the input's convex hull; Mildenhall et al. (Burst Denoising with KPN, CVPR 2018) explicitly removed it — *"we do not normalize the predicted filters with a softmax, thereby allowing predicted kernels to have negative values."* DFFN inherited the constrained variant, and its measured weakness is NR — the metric for *removing* things.

**Novelty note.** The literature sweep found **no paper stating the DSP form of this limit** (non-negative taps summing to 1 ⇒ `|H(ω)| ≤ H(0) = 1`, so the filter can never amplify any frequency). The closest is Ji & Yao (WACV 2025, arXiv:2412.01559), Proposition 1: any non-negative combination of high-pass filters stays high-pass. Stating and measuring this cleanly is a small but genuine contribution.

### 2.3 Phase is filtered as if it were a linear quantity

The block applies a softmax-weighted average to `torch.angle(...)`, wrapped on (−π, π]. Measured against the correct circular mean, at the most favourable simplex point (uniform weights):

| discrepancy | share of bins (150 mm) |
|---|---|
| > 0.5 rad | **70.3 %** |
| > 1.0 rad | 51.0 % |
| > 2.0 rad | 24.2 % |

Mean error **1.21 rad (69°)**, p95 2.88 rad. Independently reproduced at 50 mm: 69.9 % over 0.5 rad. Skewed weights make this worse, not better. **This is a correctness bug, not a design trade-off**, and it is cheap to fix — carry `(real, imag)`, or filter `(cos φ, sin φ)` and recombine with `atan2`.

Supporting evidence from an oracle split (rebuild the clean frame from one half of the rainy spectrum, spatial NMSE):

| | 50 mm | 150 mm |
|---|---|---|
| rainy frame (do nothing) | 0.5046 | 1.1266 |
| clean **magnitude** + rainy phase | 0.2227 (+55.9 %) | 0.4130 (+63.3 %) |
| rainy magnitude + clean **phase** | 0.2576 (+49.0 %) | 0.6186 (+45.1 %) |

Neither half alone approaches zero: **both magnitude and phase must be corrected**, so a magnitude-only formulation has a hard ceiling around 50–60 % error reduction.

### 2.4 The filter generator cannot see where it is in the spectrum

The FGN is `Conv1x1 → dw3x3 → Conv1x1` — translation-equivariant. But a spectrum is not translation-invariant; bin (u,v) *means* a specific frequency. Test-set R² predicting the required per-bin log-gain (MLP, frames held out):

| features | 50 mm | 150 mm |
|---|---|---|
| LOCAL 3×3 mag+phase (what the FGN sees) | 0.289 | 0.356 |
| **LOCAL + POS (r, θ harmonics)** | **0.326 (+13 %)** | **0.394 (+11 %)** |
| LOCAL + GLOBAL (frame's own angular/radial profile) | 0.279 (−3 %) | 0.368 (+3 %) |
| LOCAL + POS + GLOBAL | 0.294 | 0.372 |
| GLOBAL only | 0.260 | 0.364 |

**Positional encoding is the cheapest real win available.**

> ⚠ **Correction — this proxy was wrong about global context.** The table says global spectral context adds ≈ 0 (−3 % to +3 %). **The end-to-end pilot contradicts it: arm D, which adds exactly that global descriptor and nothing else, is the top-scoring arm (§4).** The proxy asked a different question — can hand-crafted global features linearly/MLP-predict the *ideal per-bin gain* — whereas arm D lets a network learn its own descriptor and use it to modulate a dynamic filter across four blocks in feature space. **Trust §4 over this table.** Recorded rather than deleted because the failure mode is instructive: per-bin gain predictability is a poor proxy for end-to-end restoration value.

### 2.5 Rain's Fourier signature: concentrated and directionally stable, but not contiguous

Measured per-sample (n=24 per rate):

| | 50 mm | 150 mm |
|---|---|---|
| rain energy in its top 5 % of bins | **37.0 %** | 33.5 % |
| *clean* signal energy in its own top 5 % (reference) | 19.0 % | 19.0 % |
| top-3-of-12 orientation sectors | **45.4 %** (uniform = 25 %) | 45.4 % |
| dominant orientation sector, per-frame std | **0.00** | **0.00** |

Rain is ~2× more spectrally concentrated than the scene and its dominant orientation is *identical in every frame*. Excess energy by orientation peaks at **+4.2 dB (50 mm) / +6.2 dB (150 mm)** near 5–15°, minimum ≈ 0 dB near 85–100° — oriented as the physics predicts (near-vertical streaks → near-horizontal spectral ridge).

This is the theoretical basis the project's premise rests on, and it is well founded: **Barnum, Narasimhan & Kanade, "Analysis of Rain and Snow in Frequency Space", IJCV 86(2–3):256–274, 2010** (DOI 10.1007/s11263-008-0200-2) gives the closed form. A streak modelled as a circular Gaussian of breadth `b` motion-blurred over length `l` transforms to an **isotropic Gaussian envelope of width ~1/b times a `1/|v|` ripple of period `1/l`**, with energy peaking on the line through DC oriented *perpendicular* to the spatial streak.

**That model also explains why the measured anisotropy is only 4–6 dB.** The envelope width is set by **drop breadth**, not by "high frequency", and the full rain field is an integral over drop radii (0.1–3 mm), depths (0.5–10 m) and a spread of orientations (±0.1 rad) — which smears the ridge. Barnum explicitly notes rain "covers such a broad part of the frequency space". So the directional prior is real but weak *by construction*, and a broadband component is expected — exactly the +1.6/+2.7 dB mean lift measured above. This is the physical reason a frequency-domain filter cannot cleanly separate rain here, and it corroborates §6.1's fragmentation result from theory rather than measurement.

**But the prior is a global, per-frame *orientation* property, and it does not make rain separable bin-by-bin** — see §6.1.

### 2.6 Temporal structure: carry statistics, not signal

The deck asks: *"Filter is dependent only on one frame, longer dependency?"* Measured over consecutive frames:

| | 50 mm | 150 mm |
|---|---|---|
| rain-spectrum correlation, lag 1 | 0.481 | 0.633 |
| scene-spectrum correlation, lag 1 | 0.815 | 0.815 |
| **orientation-histogram correlation, lag 1** | **0.983** | **0.990** |
| corr(log\|Sₜ\|,log\|Sₜ₊₁\|) − corr with a far frame | +0.030 | +0.023 |

The naive temporal hypothesis is **false**: rain's per-bin spectrum is *less* temporally correlated than the scene's, so carrying the raw spectrum forward would hurt. And in a direct prediction test, the previous frame adds nothing:

| features | 50 mm | 150 mm |
|---|---|---|
| LOCAL + POS | 0.335 | 0.407 |
| + previous frame's spectrum (causal) | 0.337 | 0.404 |
| + previous frame's **true gain** (oracle) | 0.335 | 0.404 |

**Even an oracle handed the previous frame's exact answer gains nothing at this window length.** The cause is the representation: frames are contiguous (0/388 gaps) but each integrates **~103.6 ms** (389 frames spanning 40.3 s — the `time_range_us` field is in nanoseconds). At ~104 ms of vehicle motion, consecutive frames are different scenes.

**Design consequence:** temporal state should condition the *filter generator*, not the signal path, and it needs only the capacity of a summary — which is exactly what an SSM hidden state `[d,N]` is (4.0 KiB at d=64, N=16). The orientation histogram, correlation 0.99, is the thing worth carrying.

---

## 3. Literature positioning

### 3.1 Papers that matter here

| paper | venue | id | what it gives us |
|---|---|---|---|
| **Barnum, Narasimhan & Kanade** | IJCV 2010 | 86(2–3):256–274 | the closed-form Fourier signature of rain streaks — the citation for the whole premise |
| FourierMamba | ICML 2025 | arXiv:2405.19450 | model-A baseline; zigzag + Symmetric Spectrum Halving **over frequency bins** |
| **PRE-Mamba** | ICCV 2025 | arXiv:2505.05307 | direct competitor; source of EventRain-27K **and** the SR/NR/DA definitions |
| **EDmamba** | arXiv 2025 | arXiv:2505.05391 | the competitor in *our* weight class: **88.98 K params, 2.27 GFLOPs** — fewer FLOPs than DFFN's 5.36 |
| **SFHformer** | ECCV 2024 | — | ⚠ closest published relative of the DFFN block: depthwise conv over the (u,v) grid **plus "frequency dynamic convolution"** — a pointwise conv + softmax producing per-bin weights over learnable kernels. *Flagged by the survey agent; I could not verify it independently (§9).* |
| KPCN | SIGGRAPH 2017 | 10.1145/3072959.3073708 | the convex-hull constraint, stated explicitly |
| Burst Denoising with KPN | CVPR 2018 | arXiv:1712.02327 | the citation for dropping softmax so kernels may go negative |
| Ji & Yao | WACV 2025 | arXiv:2412.01559 | Prop. 1 — non-negative combinations of high-pass filters stay high-pass |
| Dynamic Filter Networks | NIPS 2016 | arXiv:1605.09673 | origin of the filter-generating-network pattern |
| Mamba (S6) | COLM 2024 | arXiv:2312.00752 | a selective SSM **is already a per-position hypernetwork emitting Δ,B,C** |
| RTF | ICML 2024 | arXiv:2405.06147 | SSM as a rational transfer function `H(z)` — the formal basis for "an SSM is an IIR filter that can amplify and notch" |
| Mamba-2 / SSD | ICML 2024 | arXiv:2405.21060 | chunkwise algorithm; state passed *between* chunks — the streaming mechanism |
| Hydra | NeurIPS 2024 | arXiv:2407.09941 | bidirectional SSM; our axes are non-causal |
| GFNet | NeurIPS 2021 | — | global full-spectrum learned filter (the static alternative) |
| Event-SSM | arXiv 2024 | arXiv:2404.18508 | event-by-event SSM, **Δ = literal inter-event interval** |
| STREAM | arXiv 2024 | arXiv:2411.12603 | `Δᵢ = (tᵢ−tᵢ₋₁)·softplus(δ)` — geometry-derived, not learned |
| SSMs for Event Cameras | CVPR 2024 | arXiv:2402.15584 | S5 in an RVT backbone, **state reused across consecutive windows** |
| SECNet | ICML 2026 | arXiv:2412.20803 | best events+Fourier architecture *without* an SSM |

### 3.2 The clean negatives — where the whitespace is

The survey agents searched these explicitly and report finding nothing:

* **No paper combines a learned per-frequency-bin dynamic filter with an SSM on event data.** The intersection is empty.
* **No paper applies a learned local convolution in the frequency domain to event-camera data.**
* **No event-camera + frequency-domain Mamba** of any kind.
* **No paper states the DSP form of the convex-combination limit** (§2.2).

### 3.3 Adversarial novelty verdicts — read this before writing anything up

Each candidate idea was given to an examiner instructed to **kill it**. Verdicts, with the prior art they found (several verified from primary PDFs, not just abstracts):

| idea | verdict | what kills it |
|---|---|---|
| **SSM as the spectral filter** (selective scan over ordered frequency bins, scan output replaces the spectrum) | **ALREADY_DONE** *(2 independent examiners)* | **This is FourierMamba's `FourScan` verbatim — i.e. this project's own Model A.** Confirmed against the local PDF `2405.19450v2.pdf`, Sec. 2.3.2 / Eq. 6. Independently also OSMamba, ExpoMamba's FSSB, PAS-Mamba, FDVM-Net. |
| **Filter the complex spectrum instead of wrapped phase** | **ALREADY_DONE** *(2 examiners)* | Field default: FFC, GFNet, AFNO, Fourmer, **SFHformer** (now verified — an examiner read the PDF and the released `FourierUnit` code: it rfft2s then stacks real/imag as channels). The `(cos φ, sin φ)` + `atan2` variant is PHASEN (AAAI 2020), Biternion Nets, and settled InSAR doctrine since 1998. |
| **Single-FFT trunk** | **ALREADY_DONE** *(2 examiners)* | **Transform Once (T1), NeurIPS 2022, arXiv:2211.14453** — same problem statement, same solution, same "faster and more accurate" framing. Also JPEG-transform-domain ResNet (ICCV 2019) and FCNN (ECML PKDD 2017). Already rejected here on measurement grounds anyway (§6.2). |
| **Temporal SSM state conditioning the filter** | **INCREMENTAL** | Meta-AF (Casebeer et al.) already maintains a GRU state that conditions an adaptive filter's update rule; Zubić CVPR 2024 already carries SSM state across event windows. **Only the narrow conjunction survives:** the state carried across event windows conditioning the *filter-generating network*. Must be claimed exactly that narrowly. |

**A crucial technical correction from the examiners, independent of novelty:** switching to the complex spectrum **does not fix the convexity problem**. A non-negative, sum-to-one 3×3 kernel over stacked (real, imag) is *still* a pure smoother — by the apodisation identity, `[¼,½,¼]` in frequency is a Hann window in the signal domain, DC gain 1, unable to notch or amplify any band. **§R2's A1 is worthless without A2.** They must ship together.

**And a sizing warning on R4:** a per-(bin, channel) hidden state over an rFFT2 plane is `H × (W/2+1) × C × N` floats carried between windows — at DFFN's 72 K-parameter operating point that state can exceed the weights by orders of magnitude and make the model memory-bound, erasing the speed win. This is exactly why §2.6 concludes *carry the summary, not the signal*.

### 3.4 What this means for the contribution

The concept "combine dynamic Fourier filtering with an SSM" **is not novel — it is FourierMamba, the project's own baseline.** What survives is narrower and more defensible:

1. **Efficiency.** FSSNet reaches 48.7 K parameters by having a small generator emit `Δ,B,C` rather than instantiating full Mamba blocks — against the project's own FourierMamba2D port at 23.59 M. If FSSNet approaches FourierMamba's quality at **~1/480th the parameters**, the claim is *"FourierMamba's benefit comes from the operator, not the capacity"* — which is a real, testable, and currently unmade claim.
2. **The negative results** (§2.4, §6.1, arm C), which are genuinely unoccupied: no 1-D ordering makes rain contiguous, and **zigzag is no better than raster on event data** — that directly undercuts FourierMamba's own stated justification.
3. **The convex-combination limit stated in DSP form** (§2.2) — the sweep found no paper stating it.
4. **Event data.** No paper combines a learned per-frequency-bin dynamic filter with an SSM on events (§3.2).

Also standing: **PRE-Mamba already has frequency-domain regularisation and native temporal state** — never claim "first SSM for event deraining" or "first temporal memory for event deraining." **EDmamba at 88.98 K / 2.27 GFLOPs** is the lightweight number to beat.

---

## 4. The training pilot — controlled, on the real metrics

Four arms, identical data / seed / optimiser / schedule, 10 epochs. Train rates {5, 25, 75, 175} mm, **validation on held-out rates {20, 80} mm**. BCE on the binary event mask; each model evaluated at **its own best threshold** (19-point sweep) so the comparison measures architecture, not calibration.

* **A** — DFFN baseline (K²=9 unfold, softmax, mag/phase).
* **B** — **FSSNet**: identical skeleton, but the frequency operator is a **bidirectional selective scan** whose parameters the generator emits (`Δ, B, C` = `d + 2N` channels instead of `C·K²·2`), operating on `(real, imag)` — never a wrapped phase. Output-scale zero-initialised, so it starts as the identity.
* **C** — FSSNet + polar scan order.
* **D** — **mechanism isolation**: DFFN unchanged except the FGN also receives a *global* spectral descriptor (orientation × radius energy histogram → MLP → broadcast). No SSM. Separates "global context matters" from "the SSM is a good way to get it."

| arm | params | best mean DA | wall | 20 mm SR / NR / DA | 80 mm SR / NR / DA |
|---|---|---|---|---|---|
| A. DFFN baseline | 72,074 | 0.8173 | 11.3 min | .8962 / .7540 / .8251 | .7400 / .8767 / .8084 |
| **B. FSSNet (freq selective scan)** | **48,650** | 0.8346 | **9.3 min** | .9180 / .7606 / .8393 | .7903 / .8694 / .8298 |
| C. FSSNet + polar scan order | 48,650 | 0.8359 | 14.1 min | .9378 / .7349 / .8363 | .8117 / .8593 / .8355 |
| **D. DFFN + global context, NO SSM** | 95,946 | **0.8362** | 11.3 min | .9253 / .7566 / .8410 | .7830 / .8787 / .8308 |

### Three conclusions, in order of importance

**1. The improvement is real and robust: +1.7 to +1.9 DA over baseline, reproduced by three different mechanisms.** Every modified arm beats A by a similar margin.

**2. The SSM is *not* the active ingredient.** Arm D — a plain orientation×radius energy histogram → MLP → broadcast to every bin, **with no state space model anywhere** — scores highest (0.8362 vs FSSNet's 0.8346). B, C and D lie within 0.0016 of each other, which is inside the noise of a single-seed 10-epoch pilot. **What matters is giving the filter generator information beyond its 3×3 spectral window; the SSM is merely one way to supply it.**

**3. The SSM's real advantage is parameter efficiency, and it is large.** FSSNet reaches the same accuracy as D with **half the parameters** (48,650 vs 95,946), **32 % below the baseline**, and the fastest training of the four. That — not accuracy — is the case for the SSM.

**And the gain is almost entirely SR, not NR.** 20 mm SR .8962 → .9180/.9378/.9253 while NR barely moves (.7540 → .7606/.7349/.7566); same at 80 mm. All three variants **retain substantially more signal at essentially unchanged rain removal.** §2.2/§2.3 predicted a removal-side benefit; the measured benefit is retention-side. The consistent explanation is that the baseline's forced smoothing was *destroying signal*, and every fix that removes the convexity constraint stops that damage. Same root cause, opposite symptom to the one predicted.

**Arm C closes polar ordering on the real metric:** +0.0013 over raster — a tie — for **52 % more wall-clock**, since reordering forces a gather/scatter every block. This confirms §6.1's fragmentation result. With the idea also published twice (§6.1), **polar ordering is done.**

**Best Fourier-domain configuration: arm B (FSSNet)** — tied for best accuracy among the Fourier arms, at the lowest parameter count.

### 4b. The counterfactual arm — and it wins by a wide margin

§5.5/§5.6 predicted that the Fourier domain is the wrong place to spend. `SpatialGainNet` tests that directly: **the identical DFFN skeleton** (dw stem → 4 blocks → dw head → global residual, same FFN+SE), with the dynamic Fourier filter replaced by a **dilated dynamic *spatial* filter** with unconstrained taps. Same protocol, same harness, same held-out rain rates.

| arm | domain | params | best mean DA | wall | 20 mm SR / NR | 80 mm SR / NR |
|---|---|---|---|---|---|---|
| A. DFFN baseline | Fourier | 72,074 | 0.8173 | 11.3 min | .8962 / .7540 | .7400 / .8767 |
| B. FSSNet (SSM) | Fourier | 48,650 | 0.8346 | 9.3 min | .9180 / .7606 | .7903 / .8694 |
| C. FSSNet + polar | Fourier | 48,650 | 0.8359 | 14.1 min | .9378 / .7349 | .8117 / .8593 |
| D. DFFN + global ctx | Fourier | 95,946 | 0.8362 | 11.3 min | .9253 / .7566 | .7830 / .8787 |
| **E. SpatialGainNet + rate sensor** | **spatial** | **52,714** | **0.8509** | **6.8 min** | **.9213 / .7935** | **.8230 / .8658** |
| F. SpatialGainNet, no FFT at all | spatial | 51,594 | 0.8469 | 6.8 min | .9302 / .7723 | .8374 / .8476 |
| G. SpatialGainNet, local RF only | spatial | 52,714 | 0.8449 | 6.7 min | .9109 / .7888 | .8032 / .8768 |

**E beats the best Fourier arm by +0.0147 and the baseline by +0.0336 — roughly double the SSM variant's +0.0173 — with 27 % fewer parameters than the baseline and the fastest training of all seven arms (40 % faster than A, 52 % faster than C).**

The decomposition is clean:

| step | ΔDA |
|---|---|
| A → G: move the dynamic filter from **frequency to spatial** (local RF only) | **+0.0276** |
| G → E: add dilation (RF ≈ 5 → ≈ 70 px) | +0.0060 |
| F → E: add the scalar rain-rate sensor (one FFT, one number) | +0.0040 |

**The domain change is ~4× larger than everything else combined.** And note arm F: a model with **no FFT anywhere** still beats every Fourier arm.

**Critically, E is the only *Fourier-free* arm that improves NR.** Every Fourier variant gained on SR while NR stagnated or fell (baseline 20 mm NR .7540 → B .7606, C .7349, D .7566). E reaches **.7935** *while also* improving SR — genuinely removing more rain, not just trading the threshold.

### 4c. ORSPNet — the best model found, and it is a hybrid

A design-and-adversarial-critique workflow (16 agents) produced five candidate architectures. Four were killed (§7b). The survivor, **ORSPNet**, replaces the per-bin dynamic filter with:

* a **fixed analytic oriented sub-band bank** — 8 atoms, **32 parameters total**, von Mises in doubled angle × Gaussian in log-radius (i.e. **log-Gabor**, Field 1987 — *not* Barnum, see §7b);
* **per-pixel signed sub-band gains** predicted by a dilated depthwise trunk at dilations **(1, 4, 16)** — the exact configuration that won the receptive-field study in §5.3.

So it is a *hybrid*: analytic multi-band spectral decomposition, spatially-varying signed gains. It converges on the same insight as SpatialGainNet from the opposite direction.

Its reviewer measured 0.8387 at **4 epochs** against a 4-epoch DFFN (0.7724) and flagged that as non-comparable, since A reaches 0.8173 by epoch 10. I therefore re-ran it at the **matched 10-epoch protocol**:

| arm | domain | params | best mean DA | wall | 20 mm SR / NR | 80 mm SR / NR |
|---|---|---|---|---|---|---|
| A. DFFN baseline | Fourier | 72,074 | 0.8173 | 11.3 min | .8962 / .7540 | .7400 / .8767 |
| B. FSSNet (SSM) | Fourier | 48,650 | 0.8346 | 9.3 min | .9180 / .7606 | .7903 / .8694 |
| D. DFFN + global ctx | Fourier | 95,946 | 0.8362 | 11.3 min | .9253 / .7566 | .7830 / .8787 |
| E. SpatialGainNet | spatial | 52,714 | 0.8509 | 6.8 min | .9213 / .7935 | .8230 / .8658 |
| **H. ORSPNet** | **hybrid** | **36,206** | **0.8683** | 8.2 min | **.9340 / .8140** | **.8480 / .8770** |

**ORSPNet beats the DFFN baseline by +0.0510 DA at half the parameters (36,206 vs 72,074) and 27 % faster training — roughly 3× the gain of the SSM variant.** The margin *grew* from 4 to 10 epochs (0.8387 → 0.8683), so it was not merely faster convergence, and it was still improving at epoch 10 — a 50-epoch run should go higher.

It improves **both** metrics against the baseline at both rain rates: 20 mm **+.038 SR and +.060 NR**; 80 mm **+.108 SR** at equal NR. That is a different quality of result from the Fourier arms, which only ever bought SR.

**Verified properties** (reproduced by its reviewer on the real data): 36,206 params and 3.352 GFLOPs exactly as claimed; perfect reconstruction (partition of unity to 1.2e-6); DC structurally untouchable (`M_j(DC) = 0` exactly); **Hermitian symmetry preserved to 1.3e-7, versus 0.42 error for a 3×3 bin-mixing operator** — i.e. DFFN's per-bin neighbourhood mixing silently violates the conjugate symmetry that a real-valued output requires; exact identity at initialisation; and no `atan2`/`torch.angle` anywhere, so DEFECT 2 is structurally eliminated rather than mitigated.

**Known bug, one line:** the gate's `tanh` scale is 1.5 while `max Σⱼ Mⱼ = 0.6099`, so the deepest reachable gain is `1 − 0.6099 = 0.39` — a true notch is unreachable at 100 % of bins even at the rail. Arm I re-ran with scale 2.5 and finished **below** H — **0.8647 vs 0.8683** — so the unreachable notch was **not** the binding constraint. Consistent with §5.5: there is nothing sharp in the spectrum to notch. Interestingly arm I does buy a little NR (20 mm .8150 vs .8137; 80 mm .8812 vs .8770) but loses more SR (80 mm .8317 vs .8483), so the deeper notch trades signal for rain removal at a net loss. Keep `scale = 1.5`.

### 4d. Final leaderboard — 16 architectures, one harness

Every row: 10 epochs, identical harness/seed/data, held-out rain rates, per-model threshold sweep. Rows marked † were built and trained by the design-workflow agents; I verified each against its log before including it.

| family | arm | params | mean DA |
|---|---|---|---|
| **per-bin frequency filter** | A. DFFN baseline | 72,074 | 0.8173 |
| | BARN (Barnum analytic notch) † | 37,146 | 0.8305 |
| | B. FSSNet (SSM selective scan) | 48,650 | 0.8346 |
| | C. FSSNet + polar order | 48,650 | 0.8359 |
| | D. DFFN + global spectral context | 95,946 | 0.8362 |
| **single per-pixel spatial gain** | H3. OSA, no FFT † | 36,362 | 0.8411 |
| | H1. OSA + oriented bank as *features* † | 37,890 | 0.8418 |
| | E′. DDGNet + 72-bin polar histogram † | 51,274 | 0.8426 |
| | G. SpatialGainNet, local RF | 52,714 | 0.8449 |
| | F. SpatialGainNet, no FFT | 51,594 | 0.8469 |
| | H4. OSA at capacity † | 50,274 | 0.8507 |
| | E. SpatialGainNet + rate scalar | 52,714 | 0.8509 |
| | H2. MSG-Net † | 37,482 | 0.8516 |
| | S. DDGNet + rate scalar † | 52,394 | 0.8541 |
| **multi-band spatially-varying** | I. ORSPNet + deeper notch | 36,206 | 0.8647 |
| | **H. ORSPNet** | **36,206** | **0.8683** |

ORSPNet's per-rate detail: 20 mm **SR .9342 / NR .8137 / DA .8739**; 80 mm **SR .8483 / NR .8770 / DA .8627**. Against the baseline that is **+.038 SR and +.060 NR at 20 mm, and +.108 SR at equal NR at 80 mm** — gains on both axes at both rain rates, which no other arm achieved.

**ORSPNet wins on both axes simultaneously: best accuracy and lowest parameter count.** +0.0510 over the DFFN baseline, +0.0142 over the best single-gain model, at 0.50× the baseline's parameters.

### 4e. The architectural principle this establishes

The leaderboard separates cleanly into three tiers, and the tiers are the *operator class*, not the parameter count:

| operator | range | best |
|---|---|---|
| per-bin filtering in the frequency domain | 0.817 – 0.836 | 0.8362 @ 95.9 K |
| a single per-pixel spatial gain | 0.841 – 0.854 | 0.8541 @ 52.4 K |
| **per-pixel gains over an oriented multi-band decomposition** | — | **0.8683 @ 36.2 K** |

**The right operator for event deraining is a spatially-varying *multi-band* filter** — neither per-bin frequency filtering (too little spatial adaptivity) nor a single spatial gain (too little spectral selectivity). A single per-pixel gain is just the 1-band special case of ORSP, which is exactly the ordering observed.

### 4f. ⚠ Seed variance — this invalidates most of the fine-grained ranking

A multi-seed run (`tmp/osa_seeds.log`) re-ran two arms at seeds 1 and 2:

| arm | seed 0 | seed 1 | seed 2 | mean | **spread** |
|---|---|---|---|---|---|
| H1. OSA bank+rate | 0.8418 | 0.8484 | 0.8328 | 0.8410 | **0.0156** |
| H3. OSA no-FFT | 0.8411 | 0.8574 | 0.8518 | 0.8501 | **0.0163** |

**The within-arm seed spread is ≈0.016 DA.** Consequences, and they are large:

* **The "oriented bank as features is harmful" conclusion does not survive.** It rested on H1 0.8418 vs H2 0.8516 — a gap of 0.0098, *smaller than the 0.0156 spread within H1 alone*. Across seeds H1 averages 0.8410; H2 was only ever run at seed 0. **Unresolved, not refuted.**
* **"One scalar beats no FFT" also does not survive.** H3 at seeds 1–2 scores 0.8574 / 0.8518, above H2's single-seed 0.8516. The §5.6 scalar-vs-histogram result (+1.8 % vs +0.5 %) is likewise inside this band.
* **The entire mid-tier is one cluster.** E (.8509), F (.8469), G (.8449), H2 (.8516), H3, H4 (.8507), S (.8541), H1 — all within ~1 spread. **Do not rank them.**

What *does* survive, being ≈3× the spread:

* **A → ORSPNet, +0.0510.** Solid.
* **The tier separation** — per-bin frequency filtering (0.817–0.836) vs everything else (0.841–0.868). Solid.

### 4g. ORSPNet at 3 seeds — the margin holds, and it is the *most stable* arm

Seeds 0/1/2, identical protocol, run in parallel on separate GPUs (`tmp/orsp_s{0,1,2}.log`). Seed 0 reproduced the earlier run to 4 decimals, confirming determinism.

| arm | seeds | mean | **std** | spread | values |
|---|---|---|---|---|---|
| **ORSPNet** | 3 | **0.8647** | **0.0039** | **0.0077** | .8683 / .8606 / .8652 |
| OSA no-FFT (H3) | 3 | 0.8501 | 0.0083 | 0.0163 | .8411 / .8574 / .8518 |
| OSA bank+rate (H1) | 3 | 0.8410 | 0.0078 | 0.0156 | .8418 / .8484 / .8328 |
| DFFN baseline (A) | 1 | 0.8173 | — | — | .8173 |

**Correction to §4f: the ≈0.016 spread is arm-specific, not a property of the harness.** ORSPNet's spread is **0.0077 — half** that of the OSA arms, and it has the lowest variance of any arm measured. I over-generalised from H1/H3 in §4f; ORSPNet is materially more seed-stable than those two.

Comparisons:

| comparison | diff | evidence |
|---|---|---|
| ORSPNet vs **DFFN baseline** | **+0.0474** | **12.2× ORSPNet's own std — robust** |
| ORSPNet vs H1 (bank as features) | +0.0237 | t = 4.70, df ≈ 2.9 — **significant** |
| ORSPNet vs H3 (best multi-seed alternative) | +0.0146 | t = 2.77, df ≈ 2.8 — marginal at n=3, **but ranges do not overlap**: min ORSPNet .8606 > max H3 .8574, and ORSPNet wins **9/9** pairwise seed matchups |

**Verdict: the margin survives.** Against the baseline it is overwhelming. Against the best alternative the t-statistic is only marginal at n=3, but zero range overlap and a 9/9 pairwise sweep are the more informative statistics at this sample size. Note also that the previously quoted **0.8683 was the highest of the three seeds, flattering the mean by +0.0036** — quote **0.8647 ± 0.0039** from here on.

One claim from §4f is now resolved in the other direction: ORSPNet vs H1 (+0.0237, t = 4.70) *is* significant, so **the bank-as-operator vs bank-as-features distinction is supported after all** — though via ORSP-vs-H1, not the underpowered H1-vs-H2 comparison that originally suggested it.

**A distinction that is now only a hypothesis:** arm H1 suggested an oriented bank supplied as *extra input features* underperforms the same bank used as *the filtering basis* (ORSP). The mechanism is plausible and the two roles are genuinely different — but per the variance above, the H1-vs-H2 evidence for it is inside the noise. Treat as an open question, not a finding.

**The essential caveat:** FSSNet bundles **three** changes — SSM filter, no softmax, complex representation. If it wins, we do not yet know which change earned it, and §5–6 give strong reason to suspect the operator fixes matter more than the scan. Arm D isolates one factor; §7 gives the ablation that isolates the rest.

*(Final numbers are appended to this section when the run completes; the raw log is `tmp/train.log` and `tmp/train_results.json`.)*

---

## 5. Isolated operator experiments

Single-operator proxies: `rainy → rFFT → [same-capacity generator] → operator → iFFT`, residual form, L1, 1500 train / 445 test pairs over 25–200 mm, **disjoint frame indices**, 20 epochs, 2 seeds. Metric: spatial NMSE vs clean.

### 5.1 Which frequency operator?

| operator | params | ms | NMSE | vs input |
|---|---|---|---|---|
| (do nothing) | — | — | 0.7277 | 0.0 % |
| **softmax mag/phase — the current parameterisation** | 498 | 0.32 | **0.8969** | **−23.3 %** |
| free mag/phase (softmax removed) | 498 | 0.30 | 0.7276 | 0.0 % |
| **complex 3×3 (proper complex MAC)** | 498 | 0.38 | **0.4262** | **+41.4 %** |
| diagonal per-bin complex gain | 226 | 0.22 | 0.7276 | 0.0 % |

As a standalone operator the current parameterisation is **worse than doing nothing**; moving the same support onto the complex spectrum is a **52.5 % error reduction** relative to it.

### 5.2 Is "dynamic" earning its keep? Is the FFT?

| variant | params | ms | NMSE | vs input |
|---|---|---|---|---|
| dynamic complex 3×3 (frequency) | 498 | 0.39 | 0.4263 | 41.4 % |
| **static** complex 3×3 (learned constants) | 18 | 0.21 | 0.7277 | 0.0 % |
| spatial gain, smooth (1/8 res, upsampled) | 177 | 0.08 | 0.5549 | 23.7 % |
| **spatial gain, per-pixel — no FFT at all** | 177 | 0.07 | **0.4074** | **44.0 %** |

* **"Dynamic" is essential** — a static frequency filter does literally nothing. The project's core premise survives.
* **The Fourier detour is not obviously earning its keep** at this scale: a per-pixel spatial gain beat the best frequency operator with 1/3 the parameters and 5.6× less latency. Convolution in frequency = multiplication in space, so the frequency operator's power *is* a spatially-varying gain, obtained indirectly.

### 5.3 Where does receptive field come from? *(the SSM-vs-alternatives test)*

All variants predict a per-pixel gain; **only the predictor's receptive field changes.**

| predictor | params | ms | NMSE | vs local |
|---|---|---|---|---|
| local (3×3 dw, RF≈5 px) | 481 | 0.10 | 0.4497 | — |
| **dilated (dw 3×3 @ dilation 1/4/16, RF≈70 px)** | 801 | **0.13** | **0.3304** | **+26.5 %** |
| global average pool | 1,009 | 0.15 | 0.4213 | +6.3 % |
| 4-direction fused selective scan | 2,465 | 1.15 | 0.3641 | +19.0 % |

**Receptive field is the variable that matters, and dilated convolution delivers more of it than a selective scan, 8.8× faster.** This is the strongest argument against reaching for Mamba reflexively.

### 5.4 The speed case for SSM parameters is solid regardless

Measured directly (L = 256×129 = 33,024 bins, C = 32):

| path | ms |
|---|---|
| current: FGN head + 2× softmax + 2× unfold | **0.885** |
| SSM head + `selective_scan_fn`, d_state=8, bidirectional | **0.267** (**3.31×**) |
| SSM head + `selective_scan_fn`, d_state=16, bidirectional | 0.468 (1.89×) |

| | current | SSM (N=8) | SSM (N=16) |
|---|---|---|---|
| generator head output channels | 576 | 48 (**12×** fewer) | 64 (**9×** fewer) |
| materialised tensors | 72.6 MiB | 8.1 MiB | 8.1 MiB |
| receptive field along scan axis | 3 bins | unbounded | unbounded |

This is consistent with FSSNet's 48.7 K vs DFFN's 72.1 K parameters.

---

### 5.5 Rain's spectrum is *not* low-dimensional — and the basis does not transfer

The natural next idea after §2.5 is: rain is a physically parameterised family (Barnum: `θ, b, l, α`), so predict a few coefficients over a rain basis instead of 576 free taps per bin. **Measured, this fails.**

Using the exact rain component (`rain = merge − clean`), an SVD basis fitted on 75 % of frames and evaluated on the held-out 25 %:

| rain rate | k=1 | k=4 | k=8 | k=16 | *clean spectrum, k=16 (reference)* |
|---|---|---|---|---|---|
| 25 mm | 3.2 % | 4.0 % | 4.2 % | 4.5 % | 3.5 % |
| 50 mm | 1.7 % | 2.1 % | 2.2 % | 2.4 % | 3.5 % |
| 100 mm | 0.7 % | 0.9 % | 1.0 % | 1.1 % | 3.5 % |
| 150 mm | 0.4 % | 0.5 % | 0.6 % | 0.8 % | 3.5 % |
| 200 mm | 0.3 % | 0.4 % | 0.5 % | 0.6 % | 3.5 % |

(held-out variance explained). **A 16-dimensional rain basis explains 0.6–4.5 % of held-out rain spectra — at most rain rates *less* than the same basis explains of the clean scene.** Even in-sample, k=64 only reaches 27.6 %. So rain's per-frame spectrum is high-dimensional, frame-specific, and no more compressible than the scene.

This is consistent with Barnum rather than contradicting it: the closed form describes the *ensemble* spectrum after integrating over drop radii, depths and orientations. Any single ~104 ms realisation is a specific random arrangement of drops, and that realisation is essentially noise-like. **The ensemble has structure (the 4–6 dB ridge of §2.5); the realisation does not.**

**Consequence:** any design that tries to represent, project out, or subtract rain *in the frequency domain* using a learned or analytic basis is dead on arrival. That kills the two most attractive-sounding architectural ideas before they cost a week.

### 5.6 The frequency branch's whole contribution reduces to a scalar

If the spectrum has no exploitable structure, what is the FFT actually buying? Measured directly — all variants predict a per-pixel spatial gain with a dilated predictor; only the conditioning changes:

| variant | params | ms | NMSE | vs input | vs plain spatial |
|---|---|---|---|---|---|
| best frequency-domain operator (complex 3×3) | 1,299 | 0.37 | 0.4240 | 41.7 % | −31.1 % |
| **spatial gain, dilated predictor, no FFT** | **801** | **0.13** | **0.3235** | **55.5 %** | — |
| + single scalar (total spectral energy) | 1,921 | 0.28 | 0.3175 | 56.4 % | +1.8 % |
| + full orientation×radius histogram | 7,041 | 0.31 | 0.3218 | 55.8 % | +0.5 % |

Two readings, both material:

* **The spatial gain beats the best frequency operator by 31 %**, at 2.8× the speed and 40 % fewer parameters — replicating §5.2/§5.3 with a second implementation.
* **The full spectral histogram adds +0.5 %; a single scalar adds +1.8 %.** Within noise of each other. So the frequency domain's entire measured contribution here is *"how much rain is there"* — a rain-rate estimate — not any structural information. That is one number, obtainable from one FFT.

## 6. Refuted ideas — recorded so they are not retried

### 6.1 Polar / orientation scan ordering — refuted by measurement *and* already published

Hypothesis: ordering bins by (angle, radius) would make rain a contiguous run an SSM could gate on. Measured fragmentation of each sample's top-5 % rain-energy bins:

| scan order | runs (50 mm) | mean run length | runs (150 mm) | mean run length |
|---|---|---|---|---|
| raster | 1131 | 1.46 | 1124 | 1.47 |
| zigzag (FourierMamba) | 1129 | 1.46 | 1121 | 1.47 |
| polar (angle, radius) | 1165 | **1.42** | 1167 | **1.42** |
| orientation mod π | 1161 | 1.42 | 1158 | 1.43 |

Polar is *slightly worse* than raster at both rates. Mean run length ≈ 1.4 bins under **every** ordering: high-rain-energy bins are essentially isolated, interleaved with signal bins. **No 1-D ordering of this spectrum makes rain contiguous.** Notably this also undercuts the usual justification for FourierMamba's zigzag — on event data zigzag is no better than plain raster (1129 vs 1131).

And independently, **the idea is already published, twice in 2026**: Synergistic Mamba (ESWA 307:130946, DOI 10.1016/j.eswa.2025.130946) — Euclidean Concentric Scanning, coefficients ordered by radius from DC; and PAS-Mamba (arXiv:2601.14530) — Circular Frequency Domain Scanning along concentric low→high circles. Pilot arm C tests it anyway, for completeness.

### 6.2 A single-FFT trunk is not worth it

Hypothesis: keep all blocks in the frequency domain, so the net needs one `rfft2` and one `irfft2` rather than 2N.

| model | params | B=1 | B=4 | peak mem | #FFTs |
|---|---|---|---|---|---|
| A. DFFN baseline | 72.1 K | 11.77 ms | 49.54 ms | 1194 MiB | 8 |
| B. per-block FFT | 48.7 K | **9.77 ms** | **33.29 ms** | 515 MiB | 8 |
| C. single-FFT trunk | **17.0 M** | 17.55 ms | 31.40 ms | 453 MiB | 2 |

The trunk variant is **slower at batch 1 and 236× larger** — a full-resolution per-bin mask costs 4.2 M parameters per block. Since the FFT pair is only 7.7 % of block time (§2.1), the ceiling was ~8 % anyway. **Dropped.**

### 6.3 Also rejected

| idea | reason |
|---|---|
| Separate amplitude/phase SSM branches | Standard since FDVM-Net (Feb 2024); OSMamba and ExpoMamba both do it. Not novel — *and* §5.1 shows the mag/phase split is actively harmful. |
| "Global frequency context" as the motivation for an SSM | Measured gain **−3 % to +3 %** (§2.4), and §6.1 shows rain is not contiguous under any ordering. This is likely why FourierMamba costs 240× more for +0.022 DA. |

---

## 7. Recommendations

### R1 — Finish the ablation the pilot started *(highest value, ~half a day)*

Arm D already established the headline: **the SSM is not the active ingredient.** What remains is to find the *cheapest* sufficient mechanism. FSSNet still bundles three changes, so run these arms, each toggling exactly one:

1. DFFN + complex representation only (keep K², keep softmax) — expected to do little on its own (§3.3);
2. DFFN + softmax removed only — expected to be the binding change;
3. DFFN + frequency positional encoding only — nearly free;
4. **DFFN + basis-softmax (R2b)** — the highest-value new arm;
5. FSSNet with the scan replaced by a **dilated depthwise stack** of matched cost (§5.3 says this may match it).

Then re-run the winner with **3 seeds** — B/C/D differ by 0.0016, which a single seed cannot separate.

The paper this points to is *"the filter parameterisation, not the sequence model, is what mattered"* — sharper, unoccupied, and directly supported by arms A–D.

### R2 — Fix the operator regardless *(~1 day)*

Independently of the SSM question, fix §2.2/§2.3/§2.4 in `DynamicFourierBlock`: carry `(real, imag)`; **drop the softmax** (tanh or unit-ℓ₂ if unstable); concatenate a frequency positional encoding.

⚠ **Ship the first two together.** Per §3.3, a convex 3×3 kernel over stacked (real, imag) is *still* a pure smoother — the complex representation alone fixes the branch cut but leaves the expressivity limit fully intact. Removing the softmax is the binding change; the complex representation is what makes removing it safe.

Note the pilot's arm B improved **SR at matched NR**, not NR — so the mechanism these fixes buy is probably *less signal damage from forced smoothing*, rather than *more rain removed*. Same root cause, different symptom than §2.2 predicted.

### R2b — Softmax over *bases*, not over *taps* — the cheapest fix to both defects at once *(~1 day, high confidence)*

This came out of the final literature pass and is the single most actionable idea in the report.

**DFFormer / CDFFormer (AAAI 2024, arXiv:2303.03932)** does almost exactly what DFFN does — global-pool → MLP → **softmax → weighted sum of N learnable complex filter bases** → elementwise multiply in Fourier → iFFT. The critical difference:

* DFFN's softmax is over **the 9 taps of the kernel**, which forces a non-negative convex kernel ⇒ a pure smoother (§2.2).
* DFFormer's softmax is over **bases**, where each basis is an *unconstrained complex full-spectrum filter*. The mixture can still notch and amplify — while the simplex still supplies the training stability that motivated the softmax in the first place.

So you can keep the softmax, keep its stability, and lose the expressivity limit, by moving where it applies. It also collapses the head: predict `N` basis coefficients instead of `C·K²·2 = 576` taps — the same cost win as the SSM head (§5.4) with none of the sequential-scan latency. Precedent for the per-pixel-coefficient form is BPN (CVPR 2020).

Complementary and equally cheap: **AFNO (ICLR 2022, arXiv:2111.13587)** uses **soft-thresholding / shrinkage on frequency modes**, which is literally a learned notch — it can drive a mode to zero, which a convex tap kernel provably cannot. That is the cheapest published mechanism for suppressing the rain ridge.

**These belong in the R1 ablation as a fifth arm.** If basis-softmax matches FSSNet, the SSM is unnecessary.

### R3 — Make efficiency the headline claim, and run FSSNet against your own FourierMamba2D *(~2 days)*

Since §3.3 rules out the *concept* as a contribution, the efficiency result has to carry the paper. The SSM-parameter head is a **12× smaller generator, 3.3× faster filter path, 9× less memory**, landing at 48.7 K params.

**The missing experiment is the one that would make the claim:** train FSSNet to convergence on the full protocol and put it beside your own FourierMamba2D port (23.59 M params, 260.3 GFLOPs, DA .936). If a 48.7 K model gets close, the claim is *"FourierMamba's benefit is the operator, not the capacity — at 1/480th the parameters"*. That is a sharper and less crowded claim than anything involving the word "Mamba", and the pilot already suggests it is plausible. Also report against EDmamba's 88.98 K / 2.27 GFLOPs.

### R4 — Temporal state: carry the *summary*, and validate the window first *(~2–3 weeks, gated)*

§2.6 says raw spectra should not be carried across frames, but the orientation histogram (correlation 0.99) should. Condition the *filter generator* on an SSM state, with `Δ` from physical inter-event time (Event-SSM arXiv:2404.18508; STREAM arXiv:2411.12603), state passed between chunks à la Mamba-2. Host it in the existing `PointDFFNet`. Per §3.2 this intersection is genuinely unoccupied.

**Gate it on one cheap pre-experiment (half a day):** re-accumulate the raw stream at 5/10/20/50/100 ms and plot `corr(Sₜ,Sₜ₊₁) − corr(Sₜ,S_far)` against window length. At 104 ms it is +0.03 — useless. If that curve does not rise sharply below ~20 ms, drop R4.

### Suggested order

1. R1 ablation (gates the entire narrative).
2. R2 fixes, folded into whichever arm wins.
3. R4 pre-experiment in parallel — pure data analysis, no GPU contention.
4. R3 for the efficiency table.
5. R4 proper, only if step 3 shows signal.

---

## 8. Reproducing these measurements

| script (`tmp/` = `/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/`) | produces |
|---|---|
| `profile_dffn.py` | §2.1 stage latency, MAC and parameter breakdown, unfold memory |
| `spectral_analysis.py` | §2.2 gain percentiles, §2.5 anisotropy, convex-hull reachability |
| `expressivity.py` | §2.2 suppression floor, §2.3 phase-wrap statistics |
| `ablation_features.py` | §2.4 feature ablation, §2.3 oracle magnitude/phase split |
| `temporal_and_phase.py`, `temporal.py` | §2.6 temporal correlations and predictability |
| `scan_order2.py` | §2.5 concentration/orientation, §6.1 fragmentation |
| `operator_bakeoff.py` | §5.1 operator comparison |
| `why_it_works.py` | §5.2 dynamic-vs-static, frequency-vs-spatial |
| `where_ssm_helps.py` | §5.3 receptive-field study |
| `scan_cost.py` | §5.4 fused-scan benchmark |
| `full_arch.py` | §6.2 single-FFT trunk |
| `fss_model.py`, `dffn_global.py`, `train_compare.py` | §4 pilot models and training |

---

## 9. Caveats and limits

* **§5 experiments are single-operator proxies** on the 1-channel image, not the full 4-block, 32-channel network. They rank design choices; they do not predict SR/NR/DA. Where they disagree with §4, **trust §4** — it is the real task on the real metric.
* The `Scan4` module in §5.3 is a plain 4-direction selective scan (d_state=8), not a tuned VMamba block; a better SSM block could narrow the gap to dilated convs. FSSNet's pilot result suggests exactly this.
* **§4 is a 10-epoch pilot on a subset**; absolute DA (~0.82) is well below the deck's 50-epoch numbers (~0.91). Only the *relative* comparison is meaningful.
* §2.4/§2.6 R² values come from an MLP on sampled bins; they measure *predictability of the ideal gain*, a proxy for what a generator could learn.
* Measured DFFN forward is 5.67 ms (B=1) here vs the deck's 10.8 ms — different GPU/batch/measurement conditions. Relative breakdowns are unaffected.
* **SFHformer is now verified** (an examiner read the paper PDF and the released `FourierUnit` code). It is the closest published relative of the DFFN block and must be cited and compared against.
* **Method limitation on the literature side.** The sweep ran as six parallel surveys plus a dedicated adversarial prior-art phase; both completed and are the basis of §3 and §6.1. The 200-call web-search budget was fully consumed, so I could not run an independent third pass. Treat §3.2's clean negatives as "two thorough searches found nothing", not as exhaustive clearance.
* **The `polar-scan` examiner did not return a verdict** (the phase ran out of search budget mid-way). That idea is nonetheless closed by two published hits found in the surveys (§6.1) *and* by two independent measurements here (fragmentation, and pilot arm C).
* Literature claims carry the arXiv id or DOI reported by the survey agents; a few venue attributions were flagged medium-confidence at source.
