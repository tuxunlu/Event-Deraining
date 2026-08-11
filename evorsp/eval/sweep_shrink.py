"""Latency Pareto for shrinking the trunk. No training -- shapes only.

The campaign's central finding is that accuracy spans ~0.013 event-DA across
28K to 23.6M parameters. If capacity genuinely does not matter here, the
exploitable direction is DOWN: shrink until accuracy actually moves.

Profiling says where the time is inside a block -- FFN 19.5%, ObliqueGate 17.4%,
the two norms 29.6%, the FFT path 14.6% -- and no single item dominates, so the
levers that matter are the ones touching EVERY item at once:

    dim         scales the FFN, both norms, ctx_proj, lift, in/out_proj
    num_blocks  scales everything in the 86% the blocks own
    probe       scales the irfft2 width (r*p channels) and the lift
    n_rad       scales the atom count r, so the irfft2 width and gate output

Every arm is measured compiled, since that is the deployment configuration, and
with min-of-many CUDA events -- median wall-clock timing has produced a
fixed-shape trunk "varying" 6.85-10.13 ms on an idle GPU in this campaign.
"""
import itertools
import sys

import torch

sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/evorsp")
import config as C

C.bootstrap()
from rsp_3d import ORSPNet3D

DEV, R = "cuda", 256
WARM, ITER = 15, 60


def tmin(fn, it=ITER):
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    with torch.no_grad():
        for _ in range(WARM):
            fn()
        torch.cuda.synchronize()
        for _ in range(it):
            e0.record(); fn(); e1.record()
            torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1))
    return best


def run(nb, dim, probe, n_rad, compile_=True):
    m = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=nb, use_off=True,
                  out_chans=16, n_extra=4, dim=dim, probe=probe,
                  n_rad=n_rad).to(DEV).eval()
    n = sum(p.numel() for p in m.parameters())
    if compile_:
        m = torch.compile(m)
    on = torch.rand(1, 4, R, R, device=DEV).round()
    off = torch.rand(1, 4, R, R, device=DEV).round()
    ex = torch.rand(1, 4, R, R, device=DEV)
    return n, tmin(lambda: m(on, x_off=off, x_extra=ex))


if __name__ == "__main__":
    print(f"\n  all arms torch.compile'd, min-of-{ITER} CUDA-event timings")
    print(f"\n  {'blocks':>6s} {'dim':>4s} {'probe':>6s} {'n_rad':>6s} "
          f"{'params':>9s} {'ms':>7s} {'vs base':>8s}")
    print("  " + "-" * 56)
    base = None
    grid = [(3, 32, 4, 2)]                                   # shipped config
    grid += [(2, 32, 4, 2), (3, 24, 4, 2), (3, 16, 4, 2),
             (3, 32, 2, 2), (3, 32, 4, 1),
             (2, 16, 4, 2), (2, 24, 2, 2), (2, 16, 2, 1)]
    for nb, dim, pr, nr in grid:
        try:
            n, t = run(nb, dim, pr, nr)
        except Exception as e:
            print(f"  {nb:>6d} {dim:>4d} {pr:>6d} {nr:>6d}   FAILED "
                  f"{type(e).__name__}")
            continue
        if base is None:
            base = t
        tag = "  <- shipped" if (nb, dim, pr, nr) == (3, 32, 4, 2) else ""
        print(f"  {nb:>6d} {dim:>4d} {pr:>6d} {nr:>6d} {n:>9,} {t:>7.2f} "
              f"{base / t:>7.2f}x{tag}")
    print("\n  Latency only. Any arm worth keeping must then be TRAINED --")
    print("  the campaign's spread is 0.013 event-DA, so a shrink that costs")
    print("  more than ~0.005 is not free even if it looks cheap here.")
