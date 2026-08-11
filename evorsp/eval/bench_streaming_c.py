"""End-to-end STREAMING latency, including the stages a batch bench hides.

Everything measured so far was compute on pre-built inputs. A deployed system
must also: receive events, move them to the GPU, BUILD the occupancy planes
(done offline in our pack builder, so never benchmarked), run the model, and
return decisions. This measures that chain.

The dominant term is not compute. A 100 ms accumulation window means the first
event of a window waits 100 ms for its own decision, whatever the model costs.
That is reported separately from compute so the two are not conflated.
"""
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D
from run_kitti_perevent import sample_at
from run_kitti_headv3 import HeadV2 as HeadV3
from gpu_feats import patch_gpu, tensor_gpu

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
R, T16 = 256, 16
NW, NH = 1280, 720          # EVK4 native: the deployment sensor
WINDOW_MS = 100
torch.backends.cudnn.benchmark = True

hck = torch.load(f"{TMP}/phv3.pt", map_location="cpu")
trunk = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=16, n_extra=4)
trunk.load_state_dict(hck["trunk"])
trunk = trunk.to(DEV).eval()
head = HeadV3(feat_dim=128).to(DEV)
head.load_state_dict(hck["head"])
head.eval()

# torch.compile the TRUNK. Its input shape is fixed by construction -- always
# 4x256x256 planes regardless of event count -- so there is exactly one graph
# to compile and no recompilation at runtime. Measured 6.51 -> 4.31 ms in
# isolation; this checks what survives in the full chain, where H2D and plane
# building are untouched by compilation.
# The hooks stay attached: they read the same modules, so the per-event head
# still gets its feature taps.
import os as _os
if _os.environ.get("COMPILE", "1") == "1":
    trunk = torch.compile(trunk)
    print("  trunk: torch.compile ENABLED", flush=True)
else:
    print("  trunk: eager", flush=True)
_f, _b = {}, {}
trunk.out_proj.register_forward_pre_hook(lambda m, i: _f.__setitem__("f", i[0]))
for bi, blk in enumerate(trunk.blocks):
    blk.register_forward_hook(lambda m, i, o, bi=bi: _b.__setitem__(bi, o))


def med(fn, warm=20, reps=7, iters=10):
    """MIN over many single-shot CUDA-event timings.

    The original median-of-wall-clock version reported a fixed-shape trunk
    varying 6.85-10.13 ms across event counts on an idle GPU, and rated a
    compiled trunk slower than eager. perf_counter includes Python and launch
    jitter; the minimum of CUDA-event timings recovers the true device cost.
    """
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps * iters):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1))
    return best


def build_planes(xg, yg, tg, pg):
    """Scatter raw events into the T=16 ON/OFF occupancy planes, on GPU.
    Done offline by the pack builder, so never previously benchmarked."""
    sx = (xg * R) // NW
    sy = (yg * R) // NH
    t0 = tg[0]
    span = torch.clamp(tg[-1] - t0, min=1)
    tb = torch.clamp((tg - t0) * T16 // span, max=T16 - 1)
    flat = (tb * R + sy) * R + sx
    on = torch.zeros(T16 * R * R, device=DEV)
    off = torch.zeros(T16 * R * R, device=DEV)
    ison = pg == 1
    on.index_fill_(0, flat[ison], 1.0)
    off.index_fill_(0, flat[~ison], 1.0)
    on = on.view(T16, R, R).view(4, 4, R, R).amax(1)[None]
    off = off.view(T16, R, R).view(4, 4, R, R).amax(1)[None]
    return on, off, sx, sy, tb


print(f"streaming chain, EVK4 {NW}x{NH}, {WINDOW_MS} ms windows")
print(f"  {'events':>8s} {'H2D':>7s} {'planes':>8s} {'trunk':>7s} "
      f"{'head':>8s} {'D2H':>6s} {'compute A':>10s} {'compute B':>10s}")
rng = np.random.default_rng(0)
for N in (100_000, 300_000, 600_000, 900_000):
    x = np.sort(rng.integers(0, NW, N))
    y = rng.integers(0, NH, N)
    t = np.sort(rng.integers(0, WINDOW_MS * 1000, N))
    p = rng.integers(0, 2, N)
    xh = torch.from_numpy(x.astype(np.int64)).pin_memory()
    yh = torch.from_numpy(y.astype(np.int64)).pin_memory()
    th = torch.from_numpy(t.astype(np.int64)).pin_memory()
    ph = torch.from_numpy(p.astype(np.int64)).pin_memory()

    def h2d():
        return (xh.to(DEV, non_blocking=True), yh.to(DEV, non_blocking=True),
                th.to(DEV, non_blocking=True), ph.to(DEV, non_blocking=True))

    t_h2d = med(h2d)
    xg, yg, tg, pg = h2d()
    t_planes = med(lambda: build_planes(xg, yg, tg, pg))
    on, off, sx, sy, tb = build_planes(xg, yg, tg, pg)
    ex = torch.zeros(1, 4, R, R, device=DEV)
    t_trunk = med(lambda: trunk(on, x_off=off, x_extra=ex))

    tn = ((tg - tg[0]).float() / float(max(int(t[-1] - t[0]), 1)))

    def head_fwd():
        lm = trunk(on, x_off=off, x_extra=ex)
        fm = torch.cat([_f["f"]] + [_b[i] for i in range(3)], 1)
        lv = sample_at(lm[:, None], xg.float()[None], yg.float()[None], tn[None])
        fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                       xg.float()[None], yg.float()[None], tn[None])
        pv = patch_gpu(xg, yg, tn, pg, NW, NH)[None]
        tc = tensor_gpu(xg, yg, tg, 5_000, [4, 16, 64], NW, NH, 1_000)[None]
        return head(lv, fv, pv, tc, tn[None, :, None])

    t_head = med(head_fwd, warm=5, iters=3) - t_trunk
    dec = torch.zeros(N, dtype=torch.bool, device=DEV)
    t_d2h = med(lambda: dec.to("cpu", non_blocking=True))
    a = t_h2d + t_planes + t_trunk + t_d2h
    b = a + t_head
    print(f"  {N:>8,d} {t_h2d:7.2f} {t_planes:8.2f} {t_trunk:7.2f} "
          f"{t_head:8.2f} {t_d2h:6.2f} {a:10.2f} {b:10.2f}")

print(f"\n  compute A = trunk-only path, compute B = with per-event head (ms)")
print(f"  NOT included: the {WINDOW_MS} ms accumulation window itself, which")
print(f"  dominates end-to-end latency for any model at this window size.")
