"""Where do ORSP's 8 ms actually go, and what would make it much faster?

A 37K-parameter model spending 8 ms on a 256x256 grid is suspicious: that is
almost no arithmetic. The hypothesis is that the trunk is LAUNCH- and
MEMORY-bound rather than FLOP-bound -- many small kernels (two LayerNorm
permutes, an rfft2, an irfft2 of r*p channels, several 1x1s and depthwise 3x3s
per block), each costing a launch far larger than its work.

If so the useful optimisations are the ones that remove launches or bytes
(CUDA graphs, torch.compile fusion, channels_last, half precision) rather than
the ones that remove parameters (pruning, fewer blocks), and the FFN being 75%
of the WEIGHTS would be irrelevant to the TIME.

Measured here, all min-of-many CUDA-event timings so contention cannot inflate:
  1. per-stage breakdown       where the time sits
  2. batch scaling             flat cost => launch-bound; linear => compute-bound
  3. half precision            cheap win if memory-bound
  4. channels_last             layout win for convolutions
  5. torch.compile             fusion, fewer launches
"""
import sys
import time

import torch

sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/evorsp")
import config as C

C.bootstrap()
from rsp_3d import ORSPNet3D

DEV = "cuda"
R = 256
WARM, ITER = 20, 200


def tmin(fn, iters=ITER):
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    with torch.no_grad():
        for _ in range(WARM):
            fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1))
    return best


def build(**kw):
    return ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                     use_off=True, out_chans=16, **kw).to(DEV).eval()


def main():
    m = build()
    on = torch.rand(1, 4, R, R, device=DEV).round()
    off = torch.rand(1, 4, R, R, device=DEV).round()
    full = tmin(lambda: m(on, x_off=off))
    print(f"\n  full forward                    {full:7.2f} ms")

    # ---- 1. per-stage
    print("\n  --- per stage ---")
    p = None

    def _front():
        nonlocal p
        p = torch.cat([m.front(on), m.front(off)], 1)
    t_front = tmin(_front)
    _front()
    h = m.in_proj(p)
    t_in = tmin(lambda: m.in_proj(p))
    t_blk = []
    hh = h
    for i, blk in enumerate(m.blocks):
        t_blk.append(tmin(lambda blk=blk, hh=hh: blk(hh)))
        with torch.no_grad():
            hh = blk(hh)
    t_out = tmin(lambda: m.out_proj(hh))
    print(f"    TemporalFrontend (x2 pol)     {t_front:7.2f} ms  "
          f"{100*t_front/full:5.1f}%")
    print(f"    in_proj                       {t_in:7.2f} ms  {100*t_in/full:5.1f}%")
    for i, t in enumerate(t_blk):
        print(f"    block {i}                       {t:7.2f} ms  {100*t/full:5.1f}%")
    print(f"    out_proj                      {t_out:7.2f} ms  "
          f"{100*t_out/full:5.1f}%")
    acc = t_front + t_in + sum(t_blk) + t_out
    print(f"    (sum of stages {acc:.2f} ms vs full {full:.2f} ms -- the gap is "
          f"launch overhead the stages hide)")

    # ---- 2. batch scaling: flat => launch-bound
    print("\n  --- batch scaling (flat per-item cost => LAUNCH-bound) ---")
    for b in (1, 2, 4, 8):
        ob = on.expand(b, -1, -1, -1).contiguous()
        fb = off.expand(b, -1, -1, -1).contiguous()
        t = tmin(lambda: m(ob, x_off=fb), iters=60)
        print(f"    batch {b}   {t:7.2f} ms total   {t/b:6.2f} ms/item")

    # ---- 3/4/5. cheap wins
    print("\n  --- optimisations ---")
    onh, offh = on.half(), off.half()
    mh = build().half()
    print(f"    fp16                          {tmin(lambda: mh(onh, x_off=offh)):7.2f} ms")

    mc = build().to(memory_format=torch.channels_last)
    onc = on.to(memory_format=torch.channels_last)
    offc = off.to(memory_format=torch.channels_last)
    print(f"    channels_last                 {tmin(lambda: mc(onc, x_off=offc)):7.2f} ms")

    try:
        mk = torch.compile(build(), mode="max-autotune")
        t = tmin(lambda: mk(on, x_off=off), iters=60)
        print(f"    torch.compile max-autotune    {t:7.2f} ms   "
              f"({full/t:.2f}x)")
    except Exception as e:
        print(f"    torch.compile                 failed: {type(e).__name__}")

    # CUDA graph: the definitive launch-overhead test
    try:
        g = torch.cuda.CUDAGraph()
        si = torch.cuda.Stream()
        si.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(si), torch.no_grad():
            for _ in range(5):
                m(on, x_off=off)
        torch.cuda.current_stream().wait_stream(si)
        with torch.no_grad(), torch.cuda.graph(g):
            m(on, x_off=off)
        t = tmin(lambda: g.replay(), iters=200)
        print(f"    CUDA graph replay             {t:7.2f} ms   ({full/t:.2f}x)")
    except Exception as e:
        print(f"    CUDA graph                    failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
