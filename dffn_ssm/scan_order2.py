"""Per-sample version of the scan-order test (the averaged version could
smear per-frame directional structure). Also asks the prior question:
is rain actually CONCENTRATED in the event-frame spectrum at all?
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
                                         .astype(np.complex64)) for f in fs]).to(DEV)


N = 24
clean = load(f"{ROOT}/raw_data/*.npz", N)

for mm in ("50mm", "150mm"):
    rain = load(f"{ROOT}/merge_data/test/{mm}/*.npz", N)
    n = min(rain.shape[0], clean.shape[0])
    Zr = torch.fft.rfft2(torch.fft.ifft2(rain[:n]).real, norm='ortho')
    Zc = torch.fft.rfft2(torch.fft.ifft2(clean[:n]).real, norm='ortho')
    E = (Zr.abs() - Zc.abs()).clamp(min=0)                   # [n, Hf, Wf] per sample
    nS, Hf, Wf = E.shape

    fy = torch.fft.fftfreq(Hf, device=DEV).view(-1, 1).expand(Hf, Wf)
    fx = (torch.arange(Wf, device=DEV).float() / (2 * (Wf - 1))).view(1, -1).expand(Hf, Wf)
    rad = torch.sqrt(fy ** 2 + fx ** 2)
    ang = torch.atan2(fy, fx)
    ang_mod = torch.remainder(ang, torch.pi)

    print(f"\n================ {mm}  (per-sample, n={nS}) ================")

    # --- concentration: what share of rain energy sits in the top q% of bins?
    print("  concentration of rain energy (per sample, then averaged):")
    for q in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(q * Hf * Wf))
        share = E.flatten(1).topk(k, dim=1).values.sum(1) / E.flatten(1).sum(1)
        print(f"     top {100*q:5.1f}% of bins hold {100*share.mean():5.1f}% "
              f"(+-{100*share.std():.1f}) of rain energy")

    # same for the CLEAN signal, as the reference for 'is rain more concentrated?'
    Ec = Zc.abs()
    k = max(1, int(0.05 * Hf * Wf))
    sc = Ec.flatten(1).topk(k, dim=1).values.sum(1) / Ec.flatten(1).sum(1)
    print(f"     [reference] clean-signal energy in its own top 5%: {100*sc.mean():5.1f}%")

    # --- orientation concentration per sample
    nb = 12
    sect = (ang_mod / torch.pi * nb).long().clamp(0, nb - 1)
    sh = torch.stack([E[:, sect == kk].sum(1) for kk in range(nb)], dim=1)
    sh = sh / sh.sum(1, keepdim=True)
    top3 = sh.topk(3, dim=1).values.sum(1)
    peak = sh.argmax(1)
    print(f"  orientation: top-3/12 sectors hold {100*top3.mean():5.1f}% "
          f"(+-{100*top3.std():.1f}); uniform = 25.0%")
    print(f"     dominant sector per sample: {peak.tolist()[:12]} ... "
          f"(consistent across frames? {'yes' if peak.float().std() < 1.5 else 'no'})")

    # --- fragmentation per sample under each order
    def raster(): return torch.arange(Hf * Wf, device=DEV)

    def zigzag():
        i = torch.arange(Hf * Wf, device=DEV).view(Hf, Wf).clone()
        i[1::2] = i[1::2].flip(-1)
        return i.flatten()

    def polar(nbins=64):
        ab = ((ang + torch.pi) / (2 * torch.pi) * nbins).long().clamp(0, nbins - 1)
        return torch.argsort(ab.flatten().float() * 1e3 + rad.flatten() * 1e2)

    def orient(nbins=64):
        ab = (ang_mod / torch.pi * nbins).long().clamp(0, nbins - 1)
        return torch.argsort(ab.flatten().float() * 1e3 + rad.flatten() * 1e2)

    kk = max(1, int(0.05 * Hf * Wf))
    thr = E.flatten(1).topk(kk, dim=1).values[:, -1].view(-1, 1)
    mask = (E.flatten(1) >= thr).float()                     # [n, L]
    print("  fragmentation of each sample's top-5% rain bins:")
    for name, fn in [("raster", raster), ("zigzag (FourierMamba)", zigzag),
                     ("polar (angle,radius)", polar), ("orientation mod pi", orient)]:
        seq = mask[:, fn()]
        sw = (seq[:, 1:] != seq[:, :-1]).sum(1).float()
        runs = sw / 2 + 1
        mrl = seq.sum(1) / runs
        print(f"     {name:24s} runs {runs.mean():8.0f}   mean run length {mrl.mean():6.2f}")
