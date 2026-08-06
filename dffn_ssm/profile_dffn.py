"""Component-level latency / FLOP / param profile of DynamicFourierFilterNet.

Answers: where do the 5.36 GFLOPs and 10.8 ms actually go?
"""
import sys, time, json
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DynamicFourierFilterNet import DynamicFourierFilterNet, DynamicFourierBlock

DEV = "cuda"
H = W = 256
DIM = 32
NB = 4


def sync():
    torch.cuda.synchronize()


def bench(fn, n=50, warmup=15):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000.0  # ms


# ---------------------------------------------------------------- whole model
model = DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=DIM, num_blocks=NB).to(DEV).eval()
params = sum(p.numel() for p in model.parameters())
print(f"== DynamicFourierFilterNet  dim={DIM} blocks={NB}  params={params:,}")

results = {}
for B in (1, 4):
    x = torch.randn(B, 1, H, W, device=DEV)
    with torch.no_grad():
        ms = bench(lambda: model(x))
    print(f"   full forward  B={B}  {ms:8.3f} ms   ({ms/B:.3f} ms/frame)")
    results[f"full_b{B}"] = ms

# ------------------------------------------------------- per-module parameters
print("\n== Parameter breakdown (per block, dim=32, k=3)")
blk = model.blocks[0]
for name, mod in [("norm1", blk.norm1), ("fgn (filter generator)", blk.fgn),
                  ("norm2", blk.norm2), ("ffn", blk.ffn)]:
    n = sum(p.numel() for p in mod.parameters())
    print(f"   {name:24s} {n:8,}")
fgn_last = blk.fgn[3]
print(f"   -> fgn final 1x1 conv: {fgn_last.in_channels} -> {fgn_last.out_channels} "
      f"= {sum(p.numel() for p in fgn_last.parameters()):,} params "
      f"({100*sum(p.numel() for p in fgn_last.parameters())/params:.1f}% of whole net)")

# -------------------------------------------------- component latency in-block
print("\n== Component latency inside ONE DynamicFourierBlock (B=1, 256x256, dim=32)")
B = 1
xb = torch.randn(B, DIM, H, W, device=DEV)
k2 = 9

with torch.no_grad():
    # stage tensors
    xn = blk.norm1(xb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    fft_feat = torch.fft.rfft2(xn.float(), norm='ortho')
    mag = torch.abs(fft_feat) + 1e-8
    phase = torch.angle(fft_feat)
    fgn_in = torch.cat([mag, phase], dim=1)
    filters = blk.fgn(fgn_in)
    mf, pf = torch.chunk(filters, 2, dim=1)
    Hf, Wf = mag.shape[-2:]
    mfs = F.softmax(mf.view(B, DIM, k2, Hf, Wf), dim=2).view(B, -1, Hf, Wf)
    pfs = F.softmax(pf.view(B, DIM, k2, Hf, Wf), dim=2).view(B, -1, Hf, Wf)
    fm = blk.dynamic_filter(mag, mfs)
    fp = blk.dynamic_filter(phase, pfs)
    fc = torch.complex(fm * torch.cos(fp), fm * torch.sin(fp))

    stages = [
        ("LayerNorm",         lambda: blk.norm1(xb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)),
        ("rfft2",             lambda: torch.fft.rfft2(xn.float(), norm='ortho')),
        ("abs+angle+cat",     lambda: torch.cat([torch.abs(fft_feat) + 1e-8,
                                                 torch.angle(fft_feat)], dim=1)),
        ("FGN (filter gen)",  lambda: blk.fgn(fgn_in)),
        ("softmax x2",        lambda: (F.softmax(mf.view(B, DIM, k2, Hf, Wf), dim=2),
                                       F.softmax(pf.view(B, DIM, k2, Hf, Wf), dim=2))),
        ("dyn filter (unfold) x2", lambda: (blk.dynamic_filter(mag, mfs),
                                            blk.dynamic_filter(phase, pfs))),
        ("complex recombine", lambda: torch.complex(fm * torch.cos(fp), fm * torch.sin(fp))),
        ("irfft2",            lambda: torch.fft.irfft2(fc, s=(H, W), norm='ortho')),
        ("FFN + SE",          lambda: blk.ffn(xb)),
    ]
    total = 0.0
    rows = []
    for name, fn in stages:
        ms = bench(fn, n=60)
        total += ms
        rows.append((name, ms))
    for name, ms in rows:
        print(f"   {name:26s} {ms:8.4f} ms   {100*ms/total:5.1f}%")
    print(f"   {'SUM of stages':26s} {total:8.4f} ms")

# --------------------------------------------------------------- FLOP estimate
print("\n== Analytic MAC counts per block (B=1, 256x256 -> spectrum 256x129)")
Hf, Wf = H, W // 2 + 1
bins = Hf * Wf
fgn_hidden = max(8, int(DIM * 0.5))
macs = {}
macs["fgn 1x1 (2C->h)"] = (2 * DIM) * fgn_hidden * bins
macs["fgn dw 3x3 (h)"] = fgn_hidden * 9 * bins
macs["fgn 1x1 (h->C*k2*2)"] = fgn_hidden * (DIM * 9 * 2) * bins
macs["dyn filter mul-add x2"] = 2 * DIM * 9 * bins
ffn_hidden = 64
macs["ffn 1x1 (C->e)"] = DIM * ffn_hidden * H * W
macs["ffn dw 3x3"] = ffn_hidden * 9 * H * W
macs["ffn 1x1 (e->C)"] = ffn_hidden * DIM * H * W
tot = sum(macs.values())
for k, v in sorted(macs.items(), key=lambda kv: -kv[1]):
    print(f"   {k:26s} {v/1e6:9.1f} MMACs  {100*v/tot:5.1f}%")
print(f"   {'TOTAL / block':26s} {tot/1e6:9.1f} MMACs = {2*tot/1e9:.3f} GFLOPs")
print(f"   {'x4 blocks':26s} {4*tot/1e6:9.1f} MMACs = {2*4*tot/1e9:.3f} GFLOPs")

fgn_share = (macs["fgn 1x1 (2C->h)"] + macs["fgn dw 3x3 (h)"] + macs["fgn 1x1 (h->C*k2*2)"]) / tot
print(f"\n   -> filter-generation network is {100*fgn_share:.1f}% of per-block MACs")
print(f"   -> the single 1x1 head h->C*k^2*2 alone is "
      f"{100*macs['fgn 1x1 (h->C*k2*2)']/tot:.1f}%")

# ------------------------------------------- peak memory of the unfold pathway
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = blk.dynamic_filter(mag, mfs)
sync()
print(f"\n== Dynamic-filter unfold peak memory (B=1): "
      f"{torch.cuda.max_memory_allocated()/2**20:.1f} MiB "
      f"(materialises [B,{DIM},9,{Hf},{Wf}] = "
      f"{DIM*9*Hf*Wf*4/2**20:.1f} MiB per call, x2 for mag+phase)")

json.dump(results, open("/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/dffn_profile.json", "w"))
