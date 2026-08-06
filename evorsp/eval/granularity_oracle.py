"""The granularity ceiling: how much event-level DA does output granularity buy?

Motivated by reading PRE-Mamba's GridSample:
    scaled_coord = coord / np.array((1, 1, grid_size))
i.e. the voxel size divides ONLY the time axis. Their decision granularity is
therefore (native pixel) x (1/grid_size normalized-time bins) -- 20 bins at
test time (grid 0.05) -- and per-voxel predictions are broadcast to every event
in the voxel via return_inverse. It is not per-raw-event computation.

Ours is (256^2 pixel) x (1 time bin): one mask per window.

This script computes, for the KITTI test frames, the EXACT upper bound on
event-level DA achievable by any model that decides at a given granularity.
Per cell with b background and r rain events, the BA-optimal oracle keeps the
cell iff b/N_bg > r/N_rain (the same BA-weighted cut used in the OS-CFAR
falsifier). The resulting DA is the ceiling for that granularity.

Reported for the cross product of
    spatial  in {256^2 (ours), 460x352 (native)}
    temporal in {1 (ours), 4, 8, 16, 20 (theirs), 64}
so the 0.212 event-DA gap can be attributed to spatial vs temporal output
resolution before any model is built.
"""
import glob
import os

import numpy as np

S = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
SRC_W, SRC_H = 460, 352
SPATIAL = [("256^2 (ours)", 256, 256), ("native 460x352", SRC_W, SRC_H)]
TEMPORAL = [1, 4, 8, 16, 20, 64]
MAX_FRAMES = 80          # per intensity; DA is a frame-mean, 80 is plenty


def key(x, y, t):
    return (t.astype(np.int64) * (SRC_W * SRC_H)
            + y.astype(np.int64) * SRC_W + x.astype(np.int64))


acc = {}
for mm in ("50mm", "150mm"):
    files = sorted(glob.glob(f"{S}/merge_data/{mm}/*.npz"))
    step = max(1, len(files) // MAX_FRAMES)
    files = files[::step][:MAX_FRAMES]
    for n, f in enumerate(files):
        b = os.path.basename(f)
        with np.load(f) as d:
            x, y, t = d["x"], d["y"], d["t"]
        with np.load(f"{S}/raw_data/{b}") as d:
            rk = np.sort(key(d["x"], d["y"], d["t"]))
        rain = ~np.isin(key(x, y, t), rk)           # True = rain event
        n_bg = max(int((~rain).sum()), 1)
        n_rn = max(int(rain.sum()), 1)
        t0 = t.min()
        span = max(int(t.max() - t0), 1)
        tn = (t - t0) / span                        # normalized time in [0,1]
        for sname, W, H in SPATIAL:
            sx = np.clip((x.astype(np.int64) * W) // SRC_W, 0, W - 1)
            sy = np.clip((y.astype(np.int64) * H) // SRC_H, 0, H - 1)
            pix = sy * W + sx
            for T in TEMPORAL:
                tb = np.clip((tn * T).astype(np.int64), 0, T - 1)
                cell = pix * T + tb
                # per-cell background and rain counts
                order = np.argsort(cell, kind="stable")
                c = cell[order]
                r = rain[order]
                bounds = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
                cnt = np.diff(np.r_[bounds, len(c)])
                rain_cnt = np.add.reduceat(r.astype(np.int64), bounds)
                bg_cnt = cnt - rain_cnt
                # BA-optimal oracle: keep cell iff bg/n_bg > rain/n_rn
                keep = bg_cnt / n_bg > rain_cnt / n_rn
                sr = bg_cnt[keep].sum() / n_bg
                nr = rain_cnt[~keep].sum() / n_rn
                a = acc.setdefault((sname, T), [0.0, 0.0, 0.0, 0])
                a[0] += sr
                a[1] += nr
                a[2] += 0.5 * (sr + nr)
                a[3] += 1
        if n % 40 == 0:
            print(f"  {mm} {n}/{len(files)}", flush=True)

print("\n=== EVENT-LEVEL DA CEILING BY OUTPUT GRANULARITY (KITTI test) ===")
print("  ours achieves 0.7052 at (256^2, T=1); PRE-Mamba achieves 0.9172 "
      "at (native, T=20)\n")
hdr = "  " + "spatial".ljust(18) + "".join(f"T={T}".rjust(9) for T in TEMPORAL)
print(hdr)
for sname, _, _ in SPATIAL:
    row = "  " + sname.ljust(18)
    for T in TEMPORAL:
        a = acc[(sname, T)]
        row += f"{a[2] / a[3]:9.4f}"
    print(row)
print("\n  (cells = spatial cell x temporal bin; oracle = BA-optimal keep/drop "
      "per cell)")
for sname, _, _ in SPATIAL:
    for T in (1, 20):
        a = acc[(sname, T)]
        print(f"  {sname:18s} T={T:<3d} SR {a[0]/a[3]:.4f}  NR {a[1]/a[3]:.4f}"
              f"  DA {a[2]/a[3]:.4f}")
