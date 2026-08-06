"""ITI-regularity columns for the real EVK4 head cache (8 cols, GPU, fast).

Separate from real_headv2 so the expensive structure-tensor cache is not
rebuilt. Keyed by the SAME `sel` indices, so the two load side by side.
Measured separability of persistent rain vs scene: AUC 0.864 (burstiness,
16 px tiles) -- the model currently has no access to this.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob, os, sys
import numpy as np, torch
from iti_feats import iti_gpu

S = f"{C.REAL_SRC}"
CACHE = f"{C.REAL_HEAD}"
OUT = f"{C.REAL_ITI}"
NW, NH = 1280, 720

files = sorted(glob.glob(f"{CACHE}/*/*/*.npz"))
print(f"{len(files)} cached frames", flush=True)
done = 0
for i, cf in enumerate(files):
    sc, lv, base = cf.split("/")[-3], cf.split("/")[-2], os.path.basename(cf)
    dst = f"{OUT}/{sc}/{lv}/{base}"
    if os.path.exists(dst):
        done += 1
        continue
    try:
        with np.load(cf) as d:
            sel = d["sel"].astype(np.int64)
        with np.load(f"{S}/{sc}/merge_data/{lv}/{base}") as d:
            x, y, t = d["x"], d["y"], d["t"]
    except Exception:
        continue
    f8 = iti_gpu(torch.from_numpy(x.astype(np.int64)).cuda(),
                 torch.from_numpy(y.astype(np.int64)).cuda(),
                 torch.from_numpy(t.astype(np.int64)).cuda(),
                 nw=NW, nh=NH)[torch.from_numpy(sel).cuda()]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, iti=f8.cpu().numpy().astype(np.float16))
    done += 1
    if i % 500 == 0:
        print(f"  {i}/{len(files)}  built {done}", flush=True)
print(f"built {done} -> {OUT}", flush=True)
