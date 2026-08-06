"""Test, on REAL event-rain data, the two structural limits of DFFN's
frequency-domain dynamic filter:

  (1) phase is filtered by a *linear* convex combination although it is a
      circular quantity -> wrap error across the +-pi branch cut;
  (2) magnitude+phase are filtered independently by convex combinations, so
      the block cannot synthesise the anti-phase needed to CANCEL a frequency
      component through its additive residual.

Both are measured, not asserted.
"""
import sys, glob
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
ROOT = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/efft_results"


def load(split_glob, n=8):
    fs = sorted(glob.glob(split_glob))[:n]
    out = []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        out.append(torch.from_numpy(d["fft_complex"].astype(np.complex64)))
    return torch.stack(out).to(DEV)


rain = load(f"{ROOT}/merge_data/test/150mm/*.npz", 8)
clean = load(f"{ROOT}/raw_data/*.npz", 8)
print(f"loaded rain {tuple(rain.shape)}  clean {tuple(clean.shape)}\n")

# work with rfft2 of the spatial images, exactly as the model does
rain_img = torch.fft.ifft2(rain).real.unsqueeze(1)
clean_img = torch.fft.ifft2(clean).real.unsqueeze(1)
Zr = torch.fft.rfft2(rain_img, norm='ortho')
Zc = torch.fft.rfft2(clean_img, norm='ortho')
mag, pha = torch.abs(Zr), torch.angle(Zr)
B, C, Hf, Wf = Zr.shape
print(f"spectrum {Hf}x{Wf}\n")

# ---------------------------------------------------------------------------
# (1) phase wrap: linear (softmax) mean vs circular mean over each 3x3 nbhd
# ---------------------------------------------------------------------------
print("== (1) Phase filtering across the +-pi branch cut")
pad = F.pad(pha, (1, 1, 1, 1), mode='reflect')
patches = pad.unfold(2, 3, 1).unfold(3, 3, 1).reshape(B, C, Hf, Wf, 9)

w = torch.full((9,), 1 / 9, device=DEV)                       # uniform simplex point
linear_mean = (patches * w).sum(-1)                            # what DFFN computes
circ_mean = torch.atan2((torch.sin(patches) * w).sum(-1),
                        (torch.cos(patches) * w).sum(-1))      # what is correct

err = torch.abs(torch.atan2(torch.sin(linear_mean - circ_mean),
                            torch.cos(linear_mean - circ_mean)))
for thr in (0.5, 1.0, 2.0):
    frac = (err > thr).float().mean().item()
    print(f"   |linear-mean - circular-mean| > {thr:.1f} rad : {100*frac:5.2f}% of bins")
print(f"   mean error {err.mean():.4f} rad, median {err.median():.4f} rad, max {err.max():.4f} rad")
print("   (a uniform simplex point is the *best case*; skewed softmax weights are worse)\n")

# ---------------------------------------------------------------------------
# (2) can the block CANCEL a rain frequency?
#     The block computes  X + filtered_mag * exp(i * filtered_phase).
#     Cancelling bin k needs filtered_mag ~ |X_k| AND filtered_phase ~ phase_k + pi.
#     filtered_phase is a convex combination of the 9 neighbouring phases, so it
#     is confined to [min, max] of that neighbourhood. Measure reachability.
# ---------------------------------------------------------------------------
print("== (2) Reachability of the anti-phase needed for cancellation")
target = torch.atan2(torch.sin(pha + torch.pi), torch.cos(pha + torch.pi))   # phase_k + pi
pmin = patches.min(-1).values
pmax = patches.max(-1).values
reachable = ((target >= pmin) & (target <= pmax))
print(f"   anti-phase lies inside the neighbourhood's [min,max]: "
      f"{100*reachable.float().mean().item():5.2f}% of bins")
print(f"   -> for the other {100*(1-reachable.float().mean().item()):5.2f}% the block "
      f"CANNOT produce a cancelling phase at any softmax setting")

# where is rain energy concentrated, and is it reachable there?
rain_energy = (torch.abs(Zr) - torch.abs(Zc)).clamp(min=0)
k = max(1, int(0.01 * rain_energy.numel()))
thresh = rain_energy.flatten().topk(k).values.min()
hot = rain_energy >= thresh
print(f"   restricted to the top-1% rain-energy bins: "
      f"{100*reachable[hot].float().mean().item():5.2f}% reachable")

# ---------------------------------------------------------------------------
# (3) magnitude: convex combination cannot go below the neighbourhood minimum
# ---------------------------------------------------------------------------
print("\n== (3) Magnitude suppression floor (interior bins only, no zero-padding)")
mpad = F.pad(mag, (1, 1, 1, 1), mode='reflect')
mpatch = mpad.unfold(2, 3, 1).unfold(3, 3, 1).reshape(B, C, Hf, Wf, 9)
nb_min = mpatch.min(-1).values
# how much of a rain bin's magnitude survives even at the best convex setting?
surviving = (nb_min / mag.clamp(min=1e-8))
print(f"   best-case retained fraction of each bin's magnitude "
      f"(= neighbourhood min / own value):")
print(f"      median {surviving.median():.4f},  mean {surviving.mean():.4f}")
print(f"      on top-1% rain-energy bins: median {surviving[hot].median():.4f}")
print(f"   -> a convex-combination magnitude filter can drive a rain bin no lower")
print(f"      than its 3x3 neighbourhood minimum; isolated spectral peaks survive.")
