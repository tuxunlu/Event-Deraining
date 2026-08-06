"""Rebuild KITTI packs with EVENT-ACCOUNTING targets, subdivided in time.

target_oracle.py measured why EvORSP-3T scores 0.705 event-DA while its own
256^2 granularity permits 0.947: the training target is the problem, not the
architecture.

    A  OR of clean ON events   (our target)   event-DA 0.6981  <- model gets 0.7052
    B  OR of clean events, any polarity                0.8789
    C  count majority (clean > rain)                   0.9440
    D  BA-optimal per pixel (ceiling)                  0.9466   > PRE-Mamba 0.9172

Target A discards every OFF signal event (SR 0.52). This builder therefore
stores, per (time-bin, pixel) cell of the T=16 grid:

    on, off  : occupancy planes, unchanged
    gt_or    : >=1 background event of EITHER polarity          (rule B)
    gt_maj   : background events outnumber rain events          (rule C)

Both are packed bit-planes, so any T dividing 16 comes from max-pooling for
on/off; gt_maj must be recomputed per T from counts, so counts are kept too
(uint8, clipped at 255) to allow exact re-derivation at any T.

Background/rain is decided per merge event by exact (x, y, t) membership in the
clean raw_data stream -- the same rule the event-level evaluator uses.

Output: $EVORSP_WORK/kitti_t16e/{split}/{mm}/NNNN.npz
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob
import os
from multiprocessing import Pool

import numpy as np

S = f"{C.KITTI_SRC}"
OUT = f"{C.KITTI_PACK}"
T, R = 16, 256
SRC_W, SRC_H = 460, 352
VAL = {"20mm", "80mm"}
TEST = {"50mm", "150mm"}


def split_of(mm):
    return "val" if mm in VAL else "test" if mm in TEST else "train"


def key(x, y, t):
    return (t.astype(np.int64) * (SRC_W * SRC_H)
            + y.astype(np.int64) * SRC_W + x.astype(np.int64))


def build_one(args):
    mpath, gpath, dst = args
    if os.path.exists(dst):
        return 0
    try:
        with np.load(mpath) as d:
            mx, my, mt, mp = d["x"], d["y"], d["t"], d["p"]
        with np.load(gpath) as d:
            cx, cy, ct = d["x"], d["y"], d["t"]
    except Exception:
        return 0
    if len(mt) < 200:
        return 0

    rain = ~np.isin(key(mx, my, mt), np.sort(key(cx, cy, ct)))

    sx = np.clip((mx.astype(np.int64) * R) // SRC_W, 0, R - 1)
    sy = np.clip((my.astype(np.int64) * R) // SRC_H, 0, R - 1)
    t0 = mt.min()
    span = max(int(mt.max() - t0), 1)
    tb = np.clip(((mt - t0) * T) // span, 0, T - 1).astype(np.int64)
    cell = tb * (R * R) + sy * R + sx

    n_cell = T * R * R
    bg_cnt = np.bincount(cell[~rain], minlength=n_cell)
    rn_cnt = np.bincount(cell[rain], minlength=n_cell)

    on, off = mp == 1, mp != 1
    m_on = np.zeros(n_cell, bool)
    m_on[cell[on]] = True
    m_off = np.zeros(n_cell, bool)
    m_off[cell[off]] = True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst,
             on=np.packbits(m_on), off=np.packbits(m_off),
             gt_or=np.packbits(bg_cnt > 0),
             gt_maj=np.packbits(bg_cnt > rn_cnt),
             bg=np.minimum(bg_cnt, 255).astype(np.uint8),
             rn=np.minimum(rn_cnt, 255).astype(np.uint8))
    return 1


def main():
    jobs = []
    for mm in sorted(os.listdir(f"{S}/merge_data")):
        sp = split_of(mm)
        for p in sorted(glob.glob(f"{S}/merge_data/{mm}/*.npz")):
            g = f"{S}/raw_data/{os.path.basename(p)}"
            if os.path.exists(g):
                jobs.append((p, g, f"{OUT}/{sp}/{mm}/{os.path.basename(p)}"))
    print(f"{len(jobs)} frames", flush=True)
    with Pool(10) as pool:
        done = 0
        for i, r in enumerate(pool.imap_unordered(build_one, jobs, chunksize=8)):
            done += r
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"built {done} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
