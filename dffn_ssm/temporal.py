"""The deck asks: "filter is dependent only on one frame, longer dependency?"

Whether temporal state HELPS depends on an empirical question: is rain's
spectral signature more stable across consecutive frames than the scene's?
If yes, a filter conditioned on the past can separate them better than one
that sees a single frame, because averaging suppresses the varying component.
"""
import sys, glob
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")

import numpy as np
import torch

DEV = "cuda"
ROOT = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/efft_results"


def load(pat, n):
    fs = sorted(glob.glob(pat))[:n]
    return torch.stack([torch.from_numpy(np.load(f, allow_pickle=True)["fft_complex"]
                                         .astype(np.complex64)) for f in fs]).to(DEV), fs


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


N = 40
clean, cf = load(f"{ROOT}/raw_data/*.npz", N)
print(f"consecutive frames, n={clean.shape[0]}")
print(f"  first files: {[f.split('/')[-1] for f in cf[:3]]}\n")

for mm in ("50mm", "150mm"):
    rain, _ = load(f"{ROOT}/merge_data/test/{mm}/*.npz", N)
    n = min(rain.shape[0], clean.shape[0])
    Zr = torch.fft.rfft2(torch.fft.ifft2(rain[:n]).real, norm='ortho').abs()
    Zc = torch.fft.rfft2(torch.fft.ifft2(clean[:n]).real, norm='ortho').abs()
    Er = (Zr - Zc).clamp(min=0)                 # rain-only spectral energy
    Es = Zc                                     # scene spectral energy

    print(f"== {mm}")
    for lag in (1, 2, 5, 10):
        cr = torch.stack([corr(Er[i], Er[i + lag]) for i in range(n - lag)]).mean()
        cs = torch.stack([corr(Es[i], Es[i + lag]) for i in range(n - lag)]).mean()
        print(f"   lag {lag:2d} frames:  rain-spectrum corr {cr:.4f}   "
              f"scene-spectrum corr {cs:.4f}   ratio {cr/cs:.3f}")

    # orientation stability: is the dominant rain orientation steady over time?
    Hf, Wf = Er.shape[-2:]
    fy = torch.fft.fftfreq(Hf, device=DEV).view(-1, 1).expand(Hf, Wf)
    fx = (torch.arange(Wf, device=DEV).float() / (2 * (Wf - 1))).view(1, -1).expand(Hf, Wf)
    angm = torch.remainder(torch.atan2(fy, fx), torch.pi)
    nb = 36
    sect = (angm / torch.pi * nb).long().clamp(0, nb - 1)
    hist = torch.stack([torch.stack([Er[i][sect == k].sum() for k in range(nb)])
                        for i in range(n)])
    hist = hist / hist.sum(1, keepdim=True)
    peak = hist.argmax(1)
    print(f"   dominant orientation bin per frame (of {nb}): "
          f"mean {peak.float().mean():.1f}, std {peak.float().std():.2f}")
    hc = torch.stack([corr(hist[i], hist[i + 1]) for i in range(n - 1)]).mean()
    print(f"   frame-to-frame correlation of the orientation histogram: {hc:.4f}")
    print()

print("If rain's spectrum is markedly more temporally stable than the scene's,")
print("a state carried across frames gives the filter generator evidence a")
print("single frame cannot provide.")
