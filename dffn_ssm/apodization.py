"""Verify: a K x K convolution over the SPECTRUM is a multiplication in the
image domain (an apodization / window), not a frequency filter.

Dual of the convolution theorem:  (K * X)[u,v]  <->  x[n,m] * iDFT(K)[n,m]

If true, DFFN's frequency-domain 3x3 dynamic conv is -- to the extent its taps
vary slowly across the spectrum -- a smooth global multiplicative gain field on
the image, which is a completely different operator from "filtering the spectrum".
"""
import torch
import numpy as np

torch.manual_seed(0)
H = W = 64

# ---------------------------------------------------------------- exact identity
x = torch.randn(H, W, dtype=torch.float64)
X = torch.fft.fft2(x)

k = torch.randn(3, 3, dtype=torch.float64)          # arbitrary 3x3 taps

# circular convolution of the spectrum with k (correlation convention matched below)
Xc = torch.zeros_like(X)
for i, du in enumerate((-1, 0, 1)):
    for j, dv in enumerate((-1, 0, 1)):
        Xc += k[i, j] * torch.roll(X, shifts=(du, dv), dims=(0, 1))

y = torch.fft.ifft2(Xc)

# predicted: multiply the image by m(n,m) = sum_{du,dv} k * exp(-2i pi (du n/H + dv m/W))
n = torch.arange(H, dtype=torch.float64).view(-1, 1)
m = torch.arange(W, dtype=torch.float64).view(1, -1)
mask = torch.zeros(H, W, dtype=torch.complex128)
for i, du in enumerate((-1, 0, 1)):
    for j, dv in enumerate((-1, 0, 1)):
        mask += k[i, j] * torch.exp(2j * np.pi * (du * n / H + dv * m / W))
y_pred = x * mask

err = (y - y_pred).abs().max().item()
print("== Exact identity check (fixed taps, circular spectral convolution)")
print(f"   max |spectral-conv result  -  image * iDFT(kernel)| = {err:.3e}")
print(f"   -> {'CONFIRMED' if err < 1e-10 else 'FAILED'}: a KxK spectral convolution IS")
print(f"      an image-domain multiplication by a 9-term trigonometric polynomial.\n")

# ------------------------------------------------- what softmax taps imply
print("== What DFFN's softmax constraint implies for that mask")
for trial in range(3):
    logits = torch.randn(9, dtype=torch.float64)
    w = torch.softmax(logits, 0).view(3, 3)
    mask = torch.zeros(H, W, dtype=torch.complex128)
    for i, du in enumerate((-1, 0, 1)):
        for j, dv in enumerate((-1, 0, 1)):
            mask += w[i, j] * torch.exp(2j * np.pi * (du * n / H + dv * m / W))
    a = mask.abs()
    print(f"   trial {trial}: |m| range [{a.min():.4f}, {a.max():.4f}]  "
          f"mean {a.mean():.4f}   |m(0,0)| = {a[0,0]:.6f}")
print("   -> with w >= 0 and sum(w) = 1, |m(x,y)| <= 1 everywhere, = 1 only at the origin.")
print("      The image is multiplied by a smooth vignette that can only ATTENUATE.\n")

# ------------------------------------------------- the Hann identity
print("== Textbook cross-check: Hann window <-> [1/4, 1/2, 1/4] spectral smoother")
L = 256
hann = torch.hann_window(L, periodic=True, dtype=torch.float64)
sig = torch.randn(L, dtype=torch.float64)
lhs = torch.fft.fft(sig * hann)
S = torch.fft.fft(sig)
rhs = 0.25 * torch.roll(S, 1) + 0.5 * S + 0.25 * torch.roll(S, -1)
# hann = 0.5 - 0.5 cos(2 pi n / L) -> taps (-1/4, 1/2, -1/4) with sign convention
rhs2 = -0.25 * torch.roll(S, 1) + 0.5 * S - 0.25 * torch.roll(S, -1)
print(f"   ||FFT(x*hann) - [1/4,1/2,1/4]*FFT(x)||inf  = {(lhs-rhs).abs().max():.3e}")
print(f"   ||FFT(x*hann) - [-1/4,1/2,-1/4]*FFT(x)||inf = {(lhs-rhs2).abs().max():.3e}")
print("   -> the Hann window is EXACTLY a 3-tap spectral convolution; note the")
print("      correct taps are SIGNED (-1/4, 1/2, -1/4), which softmax cannot represent.\n")

# --------------------------------- does DFFN's per-bin variation change this?
print("== DFFN's taps vary per bin, so the operator is frequency-index-varying.")
print("   The identity above is exact only for taps constant across the spectrum.")
print("   Measuring how fast a trained model's taps vary would say how close the")
print("   operator is to a pure apodization; with slowly-varying taps it is a")
print("   spatially-smooth, locally-defined gain field -- still a multiplicative")
print("   mask on the image, not a spectral filter.")
