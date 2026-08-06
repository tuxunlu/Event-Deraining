"""Build a temporally-resolved SPAC dataset: T=16 sub-window occupancy planes.

Why T=16 and not the T we actually train on: build once at the finest resolution
we can afford, then derive every coarser arm by OR-merging adjacent planes at
load time. That makes the T=1 baseline the EXACT union of the T=16 treatment's
planes -- a bit-identical control rather than a separately-built approximation.
A 104 ms window at T=16 gives 6.5 ms bins, which sits on the measured AUC peak
(0.7265 at tau=5 ms, still 0.7099 at 10 ms).

Semantics match the existing eFFT frontend exactly (verified by unit test this
session): occupancy is an idempotent OR, and the current pipeline keeps ON
events only. We store OFF planes too, because the same event pass gets them for
free and the 1-bit frontend is the other open lever -- but the primary
experiment uses ON only, so the temporal axis is the single variable.

Labels come from the verified exact-subset relation: clean is a subset of rainy
on (x,y,t,p), so rain = merge \\ gt with zero label noise.

Output per frame, packed to bits:
    on   uint8[T*256*256/8]   rainy stream, ON events, T sub-windows
    off  uint8[T*256*256/8]   rainy stream, OFF events
    gt   uint8[256*256/8]     clean stream, ON events, union over the window
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

S = "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC"
MERGE = f"{S}/SPAC-dataset-merge/events"
GT = f"{S}/SPAC-dataset-event/gt"
OUT = "/fs/nexus-scratch/tuxunlu/spac_t16"

T = 16
R = 256
SRC_W, SRC_H = 640, 480

# scene-disjoint, chosen before looking at any result
SPLIT = {
    "train": [f"t{i}" for i in range(1, 9)],
    "val":   ["a1", "a2", "a3", "a4"],
    "test":  ["b1", "b2", "b3", "b4"],
}


def scene(seq):
    return seq.split("_")[0]


def occupancy(x, y, tb, npl):
    """OR-accumulate into npl planes of R x R. Idempotent, matching eFFT."""
    m = np.zeros((npl, R * R), dtype=bool)
    idx = y * R + x
    m[tb, idx] = True
    return m


def build_one(args):
    mpath, gpath, dst = args
    if os.path.exists(dst):
        return 0
    with np.load(mpath) as d:
        mx, my, mt, mp = d["x"], d["y"], d["t"], d["p"]
    with np.load(gpath) as d:
        gx, gy, gp = d["x"], d["y"], d["p"]
    if len(mt) < 200:
        return 0

    # 640x480 -> 256x256, both axes scaled independently. The existing KITTI
    # pipeline does the same on a non-square 460x352 source, so this matches.
    sx = np.clip((mx * R) // SRC_W, 0, R - 1).astype(np.int32)
    sy = np.clip((my * R) // SRC_H, 0, R - 1).astype(np.int32)

    t0, t1 = mt.min(), mt.max()
    span = max(int(t1 - t0), 1)
    tb = np.clip(((mt - t0) * T) // span, 0, T - 1).astype(np.int32)

    on, off = mp == 1, mp != 1
    m_on = occupancy(sx[on], sy[on], tb[on], T)
    m_off = occupancy(sx[off], sy[off], tb[off], T)

    gsx = np.clip((gx * R) // SRC_W, 0, R - 1).astype(np.int32)
    gsy = np.clip((gy * R) // SRC_H, 0, R - 1).astype(np.int32)
    gon = gp == 1
    m_gt = occupancy(gsx[gon], gsy[gon], np.zeros(gon.sum(), np.int32), 1)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst,
             on=np.packbits(m_on.ravel()),
             off=np.packbits(m_off.ravel()),
             gt=np.packbits(m_gt.ravel()))
    return 1


def main():
    jobs = []
    for seq in sorted(os.listdir(MERGE)):
        sc = scene(seq)
        split = next((k for k, v in SPLIT.items() if sc in v), None)
        if split is None:
            continue
        gdir = f"{GT}/{sc}_GT"
        if not os.path.isdir(gdir):
            continue
        for p in sorted(glob.glob(f"{MERGE}/{seq}/*.npz")):
            g = f"{gdir}/{os.path.basename(p)}"
            if os.path.exists(g):
                jobs.append((p, g, f"{OUT}/{split}/{seq}/{os.path.basename(p)}"))

    print(f"{len(jobs)} frames to build -> {OUT}", flush=True)
    for k, v in SPLIT.items():
        n = sum(1 for j in jobs if f"/{k}/" in j[2])
        print(f"  {k:5s} {n:5d} frames  scenes {v}")

    with Pool(16) as pool:
        done = 0
        for r in pool.imap_unordered(build_one, jobs, chunksize=8):
            done += r
            if done % 400 == 0:
                print(f"  ...{done}", flush=True)
    print(f"done, {done} written")


if __name__ == "__main__":
    main()
