"""Recurrence / long-persistence cache for real EVK4 (12 cols per event).

Rolling buffer over the last 8 windows, so each frame is read once. Must walk
each sequence IN ORDER; the features for a frame use only STRICTLY EARLIER
windows, so there is no leakage from the future.

Separability on the failing population (persistent rain vs scene):
    pers_K8_s16   0.881      strip_yextent 0.815      strip_count 0.769
The fine/coarse persistence RATIO I expected to carry it measured 0.539 and is
retained only because it is free once the scales are computed.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob, os, sys
import numpy as np
from recur_feats import Recur, NCOL

S = f"{C.REAL_SRC}"
CACHE = f"{C.REAL_HEAD}"
OUT = f"{C.REAL_RECUR}"

seqs = sorted({(f.split("/")[-3], f.split("/")[-2])
               for f in glob.glob(f"{CACHE}/*/*/*.npz")})
print(f"{len(seqs)} sequences", flush=True)
done = 0
for si, (sc, lv) in enumerate(seqs):
    cfs = sorted(glob.glob(f"{CACHE}/{sc}/{lv}/*.npz"))
    st = Recur()
    for cf in cfs:                     # IN ORDER: the buffer is causal
        base = os.path.basename(cf)
        dst = f"{OUT}/{sc}/{lv}/{base}"
        try:
            with np.load(cf) as d:
                sel = d["sel"].astype(np.int64)
            with np.load(f"{S}/{sc}/merge_data/{lv}/{base}") as d:
                x, y = d["x"].astype(np.int64), d["y"].astype(np.int64)
        except Exception:
            continue
        if not os.path.exists(dst):
            f12 = st.features(x, y, sel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, recur=f12.astype(np.float16))
            done += 1
        st.push(x, y)                  # push AFTER computing
    if si % 5 == 0:
        print(f"  seq {si}/{len(seqs)}  built {done}", flush=True)
print(f"built {done} -> {OUT}", flush=True)
