"""Evaluate PRE-Mamba's saved per-event predictions under OUR pixel protocol.

Reverse bridge of kitti_event_eval.py, completing the 2x2 matrix:
  - their saved test preds (result/{mm}_{frame}.npy, argmax class per point,
    current scan first in point order = merge npz event order),
  - rasterized to the eFFT-style ON-occupancy output mask: a pixel is kept
    iff >= 1 ON event in it is kept (class 0 = background),
  - scored with run_kitti3d's exact metric: SR/NR/DA over lit pixels
    (lit = ON-union of the rainy input, GT = clean ON occupancy),
    per-intensity mean over frames, then mean of the two intensities.
No threshold to select: argmax is their protocol's own decision rule.
"""
import glob
import os

import numpy as np

S = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
PACK = "/fs/nexus-scratch/tuxunlu/kitti_t16/test"
RES = "/fs/nexus-scratch/tuxunlu/git/PRE-Mamba/exp/event_rain/SYTHETIC/result"
T_BUILD, R = 16, 256
SRC_W, SRC_H = 460, 352

per_mm = {}
for mm in ("50mm", "150mm"):
    das, srs, nrs = [], [], []
    n_miss = 0
    for f in sorted(glob.glob(f"{S}/merge_data/{mm}/*.npz")):
        b = os.path.basename(f)
        frame = b.replace(".npz", "")
        pp = f"{RES}/{mm}_{frame}.npy"
        pk = f"{PACK}/{mm}/{b}"
        if not (os.path.exists(pp) and os.path.exists(pk)):
            n_miss += 1
            continue
        with np.load(f) as d:
            x, y, p = d["x"], d["y"], d["p"]
        pred = np.load(pp)
        if len(pred) < len(x):
            n_miss += 1
            continue
        keep = pred[: len(x)] == 0                      # class 0 = background
        on = p == 1
        sx = np.clip((x[on & keep].astype(np.int64) * R) // SRC_W, 0, R - 1)
        sy = np.clip((y[on & keep].astype(np.int64) * R) // SRC_H, 0, R - 1)
        kept_px = np.zeros((R, R), bool)
        kept_px[sy, sx] = True
        with np.load(pk) as d:
            lit = np.unpackbits(d["on"])[: T_BUILD * R * R] \
                .reshape(T_BUILD, R, R).max(0) > 0
            gt = np.unpackbits(d["gt"])[: R * R].reshape(R, R) > 0
        real = gt & lit
        rain = lit & ~gt
        rs, ns = int(real.sum()), int(rain.sum())
        if rs == 0 or ns == 0:
            continue
        sr = int((kept_px & real).sum()) / rs
        nr = (ns - int((kept_px & rain).sum())) / ns
        srs.append(sr)
        nrs.append(nr)
        das.append(0.5 * (sr + nr))
    per_mm[mm] = (np.mean(srs), np.mean(nrs), np.mean(das), len(das), n_miss)
    print(f"{mm}: SR {np.mean(srs):.4f} NR {np.mean(nrs):.4f} "
          f"DA {np.mean(das):.4f}  ({len(das)} frames, {n_miss} missing)")

m50, m150 = per_mm["50mm"], per_mm["150mm"]
print("\n=== PRE-MAMBA UNDER OUR PIXEL PROTOCOL (test {50,150}mm) ===")
print(f"TEST MEAN DA {(m50[2] + m150[2]) / 2:.4f}   "
      f"(SR {(m50[0]+m150[0])/2:.4f}, NR {(m50[1]+m150[1])/2:.4f})")
