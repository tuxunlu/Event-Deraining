"""Build T=16 ON/OFF/GT planes from the REAL EVK4 rain data (EventRain-27K real/).

Source: real/EVK4_artifical/scene{1..4}/merge_data/rain_{0..3}/NNNN.npz with
per-event labels labels/labels_rain_k/labels_NNNN.npy (verified visually:
label 1 = scene signal, label 0 = rain).

EVK4 is 1280x720, t in MICROseconds over ~100 ms windows. Mapping to the model
grid mirrors the KITTI build: x*256//1280, y*256//720, per-frame min/max t
binned into T=16.

GT convention matches KITTI exactly: the GT plane marks pixels with >=1 ON
signal event (the eFFT pipeline was ON-only), lit = ON-union of the rainy
stream, rain pixels = lit AND NOT gt.
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

S = f"{C.REAL_SRC}"
OUT = f"{C.WORK / 'real_t16'}"
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
        if len(lab) != len(x) or len(x) < 200:
            return 0
    except Exception:
        return 0                                  # partially-downloaded file

    sx = (x.astype(np.int64) * R) // W
    sy = (y.astype(np.int64) * R) // H
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T) // span, 0, T - 1).astype(np.int64)

    on, off = p == 1, p != 1
    m_on = np.zeros((T, R * R), bool)
    m_on[tb[on], sy[on] * R + sx[on]] = True
    m_off = np.zeros((T, R * R), bool)
    m_off[tb[off], sy[off] * R + sx[off]] = True
    sig_on = (lab == 1) & on
    m_gt = np.zeros(R * R, bool)
    m_gt[sy[sig_on] * R + sx[sig_on]] = True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, on=np.packbits(m_on.ravel()), off=np.packbits(m_off.ravel()),
             gt=np.packbits(m_gt))
    return 1


def main():
    jobs = []
    for scene in sorted(os.listdir(S)):
        for lvl in sorted(os.listdir(f"{S}/{scene}/merge_data")):
            for m in sorted(glob.glob(f"{S}/{scene}/merge_data/{lvl}/*.npz")):
                base = os.path.basename(m)
                l = f"{S}/{scene}/labels/labels_{lvl}/labels_{base}".replace(
                    ".npz", ".npy")
                if os.path.exists(l):
                    jobs.append((m, l, f"{OUT}/{scene}/{lvl}/{base}"))
    print(f"{len(jobs)} frames with labels available", flush=True)
    with Pool(4) as pool:
        done = sum(pool.imap_unordered(build_one, jobs, chunksize=8))
    print(f"built {done} new frames -> {OUT}")


if __name__ == "__main__":
    main()
