"""Latency of BOTH operating points, broken down by stage.

  (A) trunk only        rate-INVARIANT: fixed-size grids, cost independent of
                        how many events the window holds.
  (B) trunk + per-event head   rate-DEPENDENT: the head touches every event.

Reported per stage so the expensive parts are visible rather than averaged
away. Idle-node protocol: batch 1, 100 warm-up, 7 repeats, median, reject if
spread > +/-0.15 ms (GPU stages).

Honesty note: the head's feature stages are unoptimised NumPy on CPU. They are
reported as measured, not as they could be after a GPU port.
"""
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D
from run_kitti_perevent import sample_at
from run_kitti_headv3 import HeadV2 as HeadV3, multiscale_patch
from fast_tensor import tensor_cols_fast

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
NW, NH, R, T16 = 460, 352, 256, 16
torch.backends.cudnn.benchmark = True

hck = torch.load(f"{TMP}/phv3.pt", map_location="cpu")
trunk = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=16, n_extra=4)
trunk.load_state_dict(hck["trunk"])
trunk = trunk.to(DEV).eval()
head = HeadV3(feat_dim=128).to(DEV)
head.load_state_dict(hck["head"])
head.eval()
feats, blk = {}, {}
trunk.out_proj.register_forward_pre_hook(
    lambda m, i: feats.__setitem__("f", i[0]))
for bi, b in enumerate(trunk.blocks):
    b.register_forward_hook(lambda m, i, o, bi=bi: blk.__setitem__(bi, o))


def gpu_median(fn, warm=100, reps=7, iters=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / iters * 1000)
    return statistics.median(out), max(out) - min(out)


def cpu_median(fn, warm=2, reps=5):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out), max(out) - min(out)


on4 = torch.rand(1, 4, R, R, device=DEV).round()
off4 = torch.rand(1, 4, R, R, device=DEV).round()
ex = torch.rand(1, 4, R, R, device=DEV).round()

with torch.no_grad():
    m, s = gpu_median(lambda: trunk(on4, x_off=off4, x_extra=ex))
    print(f"(A) TRUNK ONLY  (rate-invariant)")
    print(f"    trunk forward            {m:8.2f} ms   spread {s:.3f} "
          f"{'OK' if s < 0.30 else 'REJECT'}")
    print(f"    -> operating point A total {m:6.2f} ms at ANY event rate\n")
    trunk_ms = m

    print("(B) TRUNK + PER-EVENT HEAD  (rate-dependent), by stage:")
    print(f"    {'events':>9s} {'patch(CPU)':>11s} {'tensor(CPU)':>12s} "
          f"{'sample+MLP':>11s} {'head tot':>9s} {'A+B tot':>9s}")
    rng = np.random.default_rng(0)
    for N in (100_000, 300_000, 600_000):
        x = rng.integers(0, NW, N).astype(np.int64)
        y = rng.integers(0, NH, N).astype(np.int64)
        t = np.sort(rng.integers(0, 104_000_000, N))
        p = rng.integers(0, 2, N)
        tn = ((t - t.min()) / max(int(t.max() - t.min()), 1)).astype(np.float32)
        idx = np.arange(N)

        mp, _ = cpu_median(lambda: multiscale_patch(x, y, tn, p, idx))
        tc_ms, _ = cpu_median(
            lambda: tensor_cols_fast(x, y, t, idx, 5_000_000, [4, 16, 64],
                                     NW, NH, 1_000_000))
        patch = torch.from_numpy(multiscale_patch(x, y, tn, p, idx))[None].to(DEV)
        tcol = torch.from_numpy(
            tensor_cols_fast(x, y, t, idx, 5_000_000, [4, 16, 64], NW, NH,
                             1_000_000))[None].to(DEV)
        xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
        ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
        tns = torch.from_numpy(tn)[None].to(DEV)

        def gpu_part():
            lm = trunk(on4, x_off=off4, x_extra=ex)
            fm = torch.cat([feats["f"]] + [blk[i] for i in range(3)], 1)
            To = lm.shape[1]
            lv = sample_at(lm[:, None], xs, ys, tns)
            fv = sample_at(fm[:, :, None].expand(-1, -1, To, -1, -1), xs, ys, tns)
            return head(lv, fv, patch, tcol, tns[..., None])

        g, gs = gpu_median(gpu_part, warm=20, iters=5)
        head_only = g - trunk_ms
        print(f"    {N:>9,d} {mp:11.1f} {tc_ms:12.1f} {head_only:11.2f} "
              f"{mp + tc_ms + head_only:9.1f} {mp + tc_ms + g:9.1f}")

print("\n  CPU stages are unoptimised NumPy; a GPU port is untested.")
print("  PRE-Mamba reference on this node: 306 ms (50mm) / 409 ms (150mm).")
