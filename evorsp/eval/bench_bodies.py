"""Matched latency for all four fair-harness bodies.

Accuracy has always been measured under the fairness protocol, but latency
never was -- only ORSP had numbers. This runs all three back to back on ONE
device, same input shape, same output width, same warmup and timing protocol,
so the comparison is apples to apples.

CONTENTION. If other jobs are on the GPU the absolute milliseconds are inflated.
Measured back to back on the same device they are inflated by roughly the same
factor, so the RATIOS survive; the absolutes should be re-taken on an idle node
before being quoted. The idle reference for ORSP-3T is 5.69 ms (latency_final).
"""
import sys
import time

import torch

sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/evorsp")
import config as C

C.bootstrap()
from bodies_e import FrontendBody

DEV = "cuda"
R = 256
WARM, ITER = 20, 300


def bench(kind, n_extra=0):
    m = FrontendBody(kind, T=4, t_out=16, dim=32, n_extra=n_extra).to(DEV).eval()
    on = torch.rand(1, 4, R, R, device=DEV).round()
    off = torch.rand(1, 4, R, R, device=DEV).round()
    kw = {"x_extra": torch.rand(1, n_extra, R, R, device=DEV)} if n_extra else {}
    # MINIMUM over many single-shot CUDA-event timings, not the mean. Under
    # contention the mean is dominated by whoever else is on the device -- the
    # earlier mean-based runs reported context making models FASTER, which is
    # impossible. The minimum recovers the uncontended cost whenever a clean
    # slot occurs, and converges from above as ITER grows.
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    with torch.no_grad():
        for _ in range(WARM):
            m(on, x_off=off, **kw)
        torch.cuda.synchronize()
        for _ in range(ITER):
            ev0.record()
            m(on, x_off=off, **kw)
            ev1.record()
            torch.cuda.synchronize()
            best = min(best, ev0.elapsed_time(ev1))
    return sum(p.numel() for p in m.parameters()), best


if __name__ == "__main__":
    print(f"\n  {'body':12s} {'params':>10s} {'no ctx':>10s} {'+ctx2':>10s} "
          f"{'ms / 10k params':>16s}")
    print("  " + "-" * 62)
    rows = []
    for k in ("orsp", "streaknet", "dffn", "fmamba"):
        p0, t0 = bench(k, 0)
        p2, t2 = bench(k, 4)
        rows.append((k, p0, t0, t2))
        print(f"  {k:12s} {p0:>10,} {t0:>9.2f}m {t2:>9.2f}m "
              f"{t0 / (p0 / 1e4):>15.3f}")
    base = rows[0][2]
    print(f"\n  relative to ORSP (no ctx): " + "  ".join(
        f"{k} {t / base:.2f}x" for k, _, t, _ in rows))
    print("  min-of-300 single-shot timings; a context arm FASTER than its\n  no-context arm would indicate the measurement is still contended.")
