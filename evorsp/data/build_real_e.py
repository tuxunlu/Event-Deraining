"""Rebuild REAL EVK4 packs with event-accounting targets (T=16 cells).

The existing real packs (build_real.py) carry the SAME defect the KITTI packs
had: GT is `sig_on = (lab == 1) & on` -- an OR-union over the window, ON events
only. On KITTI that target's event-DA ceiling measured 0.6981 while the model
scored 0.7052 (saturated); fixing it to a count-majority, polarity-complete
target took the trained model from 0.7052 to 0.9183.

Label orientation (verified, not assumed): `rain_k` is a RECORDING INDEX, not an
intensity -- scene4/rain_1 averages 924K events/frame against rain_13's 242K --
so the rising label-1 fraction with k is not evidence about rain. The decisive
test is cross-frame persistence: scene structure recurs at the same pixels,
rain does not. Measured on scene4, events landing on a pixel active in the next
frame:  label 1 -> 0.54-0.55,  label 0 -> 0.18-0.24.
Therefore **label 1 = scene (background to keep), label 0 = rain**.

Stores per (time-bin, pixel) cell of a T=16 x 256 x 256 grid:
    on, off  : occupancy bit-planes (unchanged)
    bg, rn   : exact counts of scene / rain events (uint16, compressed)
so any T_out dividing 16 is derived by summing counts, and the event-level
metric is computable exactly from the packs with no raw-event pass.

Output: /fs/nexus-scratch/tuxunlu/real_t16e/{scene}/{rain_k}/NNNN.npz
"""
import glob
import os
from multiprocessing import Pool

import numpy as np

S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
OUT = "/fs/nexus-scratch/tuxunlu/real_t16e"
T, R = 16, 256
W, H = 1280, 720


def build_one(args):
    mpath, lpath, dst = args
    if os.path.exists(dst):
        return 0
    try:
        with np.load(mpath) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        lab = np.load(lpath)
    except Exception:
        return 0
    if len(lab) != len(x) or len(x) < 200:
        return 0

    lab = lab.astype(np.int64)
    sx = (x.astype(np.int64) * R) // W
    sy = (y.astype(np.int64) * R) // H
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T) // span, 0, T - 1).astype(np.int64)
    cell = tb * (R * R) + sy * R + sx
    n_cell = T * R * R

    is_bg = lab == 1                      # scene events (verified above)
    bg = np.bincount(cell[is_bg], minlength=n_cell)
    rn = np.bincount(cell[~is_bg], minlength=n_cell)

    on, off = p == 1, p != 1
    m_on = np.zeros(n_cell, bool)
    m_on[cell[on]] = True
    m_off = np.zeros(n_cell, bool)
    m_off[cell[off]] = True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(
        dst, on=np.packbits(m_on), off=np.packbits(m_off),
        bg=np.minimum(bg, 65535).astype(np.uint16),
        rn=np.minimum(rn, 65535).astype(np.uint16))
    return 1


def main():
    jobs = []
    for scene in sorted(os.listdir(S)):
        md = f"{S}/{scene}/merge_data"
        if not os.path.isdir(md):
            continue
        for lvl in sorted(os.listdir(md)):
            for m in sorted(glob.glob(f"{md}/{lvl}/*.npz")):
                base = os.path.basename(m)
                l = (f"{S}/{scene}/labels/labels_{lvl}/labels_{base}"
                     .replace(".npz", ".npy"))
                if os.path.exists(l):
                    jobs.append((m, l, f"{OUT}/{scene}/{lvl}/{base}"))
    print(f"{len(jobs)} labelled frames", flush=True)
    with Pool(10) as pool:
        done = 0
        for i, r in enumerate(pool.imap_unordered(build_one, jobs, chunksize=4)):
            done += r
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"built {done} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
