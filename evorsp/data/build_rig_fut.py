"""Dense future-persistence target for the rig cache -- the auxiliary task.

The question: does predicting future persistence carry information the true
rig labels do not? If so it belongs in the main training recipe, not only in
no-label adaptation.

DENSE, not the confident tails. Pseudo-labelling wanted high precision, so it
kept only events that persist in ALL K windows or NONE. An auxiliary task wants
the opposite -- a target defined for EVERY event, so every event contributes
gradient. Here that is simply: is this event's pixel lit in the NEXT window?
One extra file read per cached frame instead of four.

Writes {FUT}/{scene}/{rain_k}/NNNN.npz with `fut` aligned to the cache's `sel`.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = "/fs/nexus-scratch/tuxunlu/real_headv2"
SRC = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
FUT = "/fs/nexus-scratch/tuxunlu/real_fut"
NW, NH = 1280, 720


def one(cf):
    dst = cf.replace(CACHE, FUT)
    if os.path.exists(dst):
        return 0
    sc, mm = cf.split("/")[-3], cf.split("/")[-2]
    base = os.path.basename(cf)
    idx = int(base.split(".")[0])
    cur = f"{SRC}/{sc}/merge_data/{mm}/{idx:010d}.npz"
    nxt = f"{SRC}/{sc}/merge_data/{mm}/{idx+1:010d}.npz"
    if not (os.path.exists(cur) and os.path.exists(nxt)):
        return 0
    try:
        with np.load(cf) as d:
            sel = d["sel"].astype(np.int64)
        with np.load(cur) as d:
            x, y = d["x"], d["y"]
        occ = np.zeros((NH, NW), bool)
        with np.load(nxt) as d:
            occ[d["y"], d["x"]] = True
    except Exception:
        return 0
    if sel.max(initial=-1) >= len(x):
        return 0
    fut = occ[y[sel].astype(np.int64), x[sel].astype(np.int64)]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, fut=fut.astype(np.uint8))
    return 1


def main():
    files = sorted(glob.glob(f"{CACHE}/*/*/*.npz"))
    print(f"{len(files)} cached frames", flush=True)
    with Pool(12) as pool:
        n = 0
        for i, r in enumerate(pool.imap_unordered(one, files, chunksize=8)):
            n += r
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    print(f"wrote {n} -> {FUT}", flush=True)


if __name__ == "__main__":
    main()
