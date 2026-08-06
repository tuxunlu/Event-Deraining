"""F4: GPU microbench of the per-event native-evidence head's moving parts.

Measured on an IDLE node (chained after FourierMamba exits), median of 7:
  (i)   scatter-add build of the native guide G [8,720,1280]
  (ii)  3x3x8 patch gather for N events                       -> [N,72]
  (iii) 17-channel bilinear gather (logit + 16 trunk feats)   -> [N,17]
  (iv)  head GEMMs [N,96] -> 48 -> 32 -> 1
at N = 1e5 (typical) and N = 3e5 (heavy rain worst case).

Pre-registered: KILL if total > 2.5 ms; comfort line 1.5 ms.
"""
import statistics
import time

import torch

DEV = "cuda"
torch.backends.cudnn.benchmark = True
NH, NW, RH, RW = 720, 1280, 256, 448


def timeit(fn, warm=50, reps=7, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    med = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        med.append((time.perf_counter() - t0) / iters * 1000)
    return statistics.median(med)


L = torch.randn(1, 17, RH, RW, device=DEV)                    # logit + feats
W1 = torch.randn(96, 48, device=DEV)
W2 = torch.randn(48, 32, device=DEV)
W3 = torch.randn(32, 1, device=DEV)

print(f"{'N':>8s} {'scatter':>8s} {'patch':>7s} {'bilin':>7s} {'gemm':>7s} "
      f"{'TOTAL':>7s}  verdict")
for N in (100_000, 300_000):
    x = torch.randint(0, NW, (N,), device=DEV)
    y = torch.randint(0, NH, (N,), device=DEV)
    ch = torch.randint(0, 8, (N,), device=DEV)
    gx = (x.float() + 0.5) / NW * 2 - 1
    gy = (y.float() + 0.5) / NH * 2 - 1
    grid = torch.stack([gx, gy], 1)[None, :, None, :]

    G = torch.zeros(8, NH, NW, dtype=torch.int16, device=DEV)
    flat_idx = ch * (NH * NW) + y * NW + x

    def scatter():
        G.zero_()
        G.view(-1).scatter_add_(0, flat_idx,
                                torch.ones(N, dtype=torch.int16, device=DEV))
    scatter()
    Gp = torch.nn.functional.pad(G.float()[None], (1, 1, 1, 1))[0]
    xs = torch.stack([x + dx for dx in (0, 1, 2) for _ in (0,)]* 1)  # noqa
    def patch():
        cols = []
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                cols.append(Gp[:, y + dy, x + dx])            # [8,N]
        return torch.log1p(torch.cat(cols, 0).T)              # [N,72]
    def bilin():
        return torch.nn.functional.grid_sample(
            L, grid, mode="bilinear", align_corners=False)[0, :, :, 0].T
    P = patch()
    B = bilin()
    z = torch.cat([P, B, torch.randn(N, 7, device=DEV)], 1)   # [N,96] w/ meta
    def gemm():
        h = torch.relu(z @ W1)
        h = torch.relu(h @ W2)
        return h @ W3

    t_s = timeit(scatter)
    t_p = timeit(patch)
    t_b = timeit(bilin)
    t_g = timeit(gemm)
    tot = t_s + t_p + t_b + t_g
    verdict = "KILL >2.5" if tot > 2.5 else ("PASS (comfort)" if tot <= 1.5
                                             else "PASS")
    print(f"{N:>8,d} {t_s:>8.3f} {t_p:>7.3f} {t_b:>7.3f} {t_g:>7.3f} "
          f"{tot:>7.3f}  {verdict}")
