"""Inside one RainSubspaceBlock: which operation owns the 1.85 ms?

The stage profile put the three blocks at 86% of the trunk's 6.43 ms, and batch
scaling showed only ~1.1 ms of launch overhead available (per-item cost falls
6.42 -> 5.28 ms from batch 1 to 8). So the time is real work inside the blocks,
and the question is which work.

The parameter table says the FFN is 75% of a block's WEIGHTS. That says nothing
about time: the spectral path does one rfft2 of p=4 channels and one irfft2 of
r*p=32 channels at 256x256, which is a lot of bytes for very few parameters.

Also fixes the fp16 arm: .half() breaks the atom bank, whose coeffs() @ basis
matmul mixes a half parameter with an fp32 basis grid. autocast is the correct
mechanism -- it keeps the reductions in fp32.
"""
import sys

import torch

sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/evorsp")
import config as C

C.bootstrap()
from rsp_3d import ORSPNet3D

DEV, R = "cuda", 256
WARM, ITER = 20, 150


def tmin(fn, iters=ITER):
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    with torch.no_grad():
        for _ in range(WARM):
            fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            e0.record(); fn(); e1.record()
            torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1))
    return best


def main():
    m = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3, use_off=True,
                  out_chans=16).to(DEV).eval()
    on = torch.rand(1, 4, R, R, device=DEV).round()
    off = torch.rand(1, 4, R, R, device=DEV).round()
    with torch.no_grad():
        p = torch.cat([m.front(on), m.front(off)], 1)
        x = m.in_proj(p)
    blk = m.blocks[0]
    total = tmin(lambda: blk(x))
    print(f"\n  one block, total                {total:7.3f} ms\n")

    with torch.no_grad():
        xn = blk.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        P = torch.fft.rfft2(blk.to_probe(xn).float(), norm="ortho")
        Hf, Wf = P.shape[-2:]
        M = blk.bank(Hf, Wf, x.device)
        Sb = P.unsqueeze(1) * M.unsqueeze(0).unsqueeze(2)
        Y = torch.fft.irfft2(Sb.reshape(1, blk.r * blk.probe, Hf, Wf),
                             s=(R, R), norm="ortho").view(1, blk.r, blk.probe, R, R)
        e = Y.pow(2).mean(2)
        q = e / (e.sum(1, keepdim=True) + 1e-8)
        ent = -(q * (q + 1e-8).log()).sum(1, keepdim=True)
        feat = torch.cat([torch.log(e + 1e-6), q, ent, blk.ctx_proj(xn)], 1)
        Eg = (P.abs().pow(2).sum(1).unsqueeze(1) * M.unsqueeze(0)).sum((-1, -2))
        Eg = Eg / (Eg.sum(1, keepdim=True) + 1e-8)
        gam, bet = blk.film(Eg).chunk(2, 1)
        film = (gam.view(1, blk.r, 1, 1), bet.view(1, blk.r, 1, 1))
        g = blk.gate(feat, film)
        rain = (g.unsqueeze(2) * Y * blk.band_scale).sum(1)
        x2 = x - blk.lift(rain)
        xn2 = blk.norm2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    parts = [
        ("norm1 + permutes", lambda: blk.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)),
        ("to_probe 1x1 (32->4)", lambda: blk.to_probe(xn)),
        ("rfft2  (4 ch)", lambda: torch.fft.rfft2(blk.to_probe(xn).float(), norm="ortho")),
        ("bank masks (cached@eval)", lambda: blk.bank(Hf, Wf, x.device)),
        ("band multiply", lambda: P.unsqueeze(1) * M.unsqueeze(0).unsqueeze(2)),
        ("irfft2 (32 ch)  <-- ?", lambda: torch.fft.irfft2(
            Sb.reshape(1, blk.r * blk.probe, Hf, Wf), s=(R, R), norm="ortho")),
        ("gate evidence", lambda: torch.cat(
            [torch.log(Y.pow(2).mean(2) + 1e-6), q, ent, blk.ctx_proj(xn)], 1)),
        ("ObliqueGate", lambda: blk.gate(feat, film)),
        ("oblique subtract + lift", lambda: x - blk.lift(
            (g.unsqueeze(2) * Y * blk.band_scale).sum(1))),
        ("norm2 + permutes", lambda: blk.norm2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)),
        ("FFN  (75% of weights)", lambda: blk.ffn(xn2)),
    ]
    for name, fn in parts:
        t = tmin(fn)
        print(f"    {name:28s} {t:7.3f} ms  {100 * t / total:5.1f}%")

    print("\n  --- what would actually help ---")
    with torch.no_grad():
        print(f"    autocast bf16 whole trunk     "
              f"{tmin(lambda: _ac(m, on, off)):7.3f} ms")
    mc = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3, use_off=True,
                   out_chans=16).to(DEV).eval().to(memory_format=torch.channels_last)
    onc = on.to(memory_format=torch.channels_last)
    offc = off.to(memory_format=torch.channels_last)
    print(f"    channels_last                 {tmin(lambda: mc(onc, x_off=offc)):7.3f} ms")
    for pr in (4, 2):
        mm = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3, use_off=True,
                       out_chans=16, probe=pr).to(DEV).eval()
        print(f"    probe={pr} (irfft2 {8*pr} ch)      "
              f"{tmin(lambda: mm(on, x_off=off)):7.3f} ms")


def _ac(m, on, off):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(on, x_off=off)


if __name__ == "__main__":
    main()
