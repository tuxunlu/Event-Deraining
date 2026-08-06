"""Build the temporally-resolved KITTI dataset: T=16 ON/OFF/GT bit planes.

Mirror of spac_build.py with three differences:
  * source resolution 460x352 (matches efft metadata original_resolution);
  * pairing is by FILENAME across intensity folders (raw_data/NNNN.npz is the
    clean stream for merge_data/{mm}/NNNN.npz at every mm) -- verified: clean is
    an EXACT subset of rainy at all 18 intensities;
  * splits follow the established KITTI protocol -- BY INTENSITY, not by scene:
    train = 14 intensities, val = {20,80}mm, test = {50,150}mm. This matches the
    2D leaderboard (run_protocol.py) so numbers are directly comparable.

Output: /fs/nexus-scratch/tuxunlu/kitti_t16/{split}/{mm}/NNNN.npz
        on/off: uint8-packed [T=16,256,256] occupancy; gt: packed [256,256].
"""
import glob
import os
from multiprocessing import Pool

import numpy as np

S = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
OUT = "/fs/nexus-scratch/tuxunlu/kitti_t16"
T, R = 16, 256
SRC_W, SRC_H = 460, 352

VAL = {"20mm", "80mm"}
TEST = {"50mm", "150mm"}


def split_of(mm):
    return "val" if mm in VAL else "test" if mm in TEST else "train"


def occupancy(x, y, tb, npl):
    m = np.zeros((npl, R * R), dtype=bool)
    m[tb, y * R + x] = True
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

    sx = np.clip((mx * R) // SRC_W, 0, R - 1).astype(np.int32)
    sy = np.clip((my * R) // SRC_H, 0, R - 1).astype(np.int32)
    t0 = mt.min()
    span = max(int(mt.max() - t0), 1)
    tb = np.clip(((mt - t0) * T) // span, 0, T - 1).astype(np.int32)

    on, off = mp == 1, mp != 1
    m_on = occupancy(sx[on], sy[on], tb[on], T)
    m_off = occupancy(sx[off], sy[off], tb[off], T)

    gsx = np.clip((gx * R) // SRC_W, 0, R - 1).astype(np.int32)
    gsy = np.clip((gy * R) // SRC_H, 0, R - 1).astype(np.int32)
    gon = gp == 1
    m_gt = occupancy(gsx[gon], gsy[gon], np.zeros(int(gon.sum()), np.int32), 1)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, on=np.packbits(m_on.ravel()), off=np.packbits(m_off.ravel()),
             gt=np.packbits(m_gt.ravel()))
    return 1


def main():
    jobs = []
    for mm in sorted(os.listdir(f"{S}/merge_data")):
        sp = split_of(mm)
        for p in sorted(glob.glob(f"{S}/merge_data/{mm}/*.npz")):
            g = f"{S}/raw_data/{os.path.basename(p)}"
            if os.path.exists(g):
                jobs.append((p, g, f"{OUT}/{sp}/{mm}/{os.path.basename(p)}"))
    print(f"{len(jobs)} frames -> {OUT}", flush=True)
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s} {sum(1 for j in jobs if f'/{sp}/' in j[2]):5d}")

    with Pool(4) as pool:
        done = 0
        for r in pool.imap_unordered(build_one, jobs, chunksize=8):
            done += r
            if done and done % 800 == 0:
                print(f"  ...{done}", flush=True)
    print(f"done, {done} written")


if __name__ == "__main__":
    main()
