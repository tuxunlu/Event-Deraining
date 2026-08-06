"""Cache per-event oriented features + a mixed-cell-stratified sample.

Two things the plain head lacks, both aimed at the occlusion failure:

  ORIENTATION.  eigenpyramid.tensor_cols gives 19 structure-tensor columns per
  event (coherence, cos2t/sin2t, log-spread-per-mass, minor-axis residual at
  3 scales, plus cross-scale agreement gates). A car edge is coherent and
  oriented; a rain drop is an isotropic blob. The plain head's 3x3 count patch
  cannot express that. These columns already survived their own falsifier
  (0.7504 event-BA vs 0.6888 count-only).
  Retargeted to KITTI: module globals NW/NH -> 460x352 and TSCALES -> {2,8,32}
  (same relative coverage as {4,16,64} at 1280x720).

  SAMPLING.  Mixed-cell events are 4.4% of events, so uniform sampling gives the
  failure case ~4% of the gradient. We store a stratified sample -- up to half
  mixed-cell, remainder uniform -- together with 1/p importance weights and the
  FULL-frame class totals, so the training loss stays an unbiased estimate of
  the whole-frame balanced-accuracy objective despite the biased sample.

tensor_cols is a causal 1 ms time-sliced loop (slow), hence the cache.
Output: $EVORSP_WORK/kitti_headv2/{split}/{mm}/NNNN.npz
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

from fast_tensor import tensor_cols_fast

SRC = f"{C.KITTI_SRC}"
PACK = f"{C.KITTI_PACK}"
OUT = f"{C.KITTI_HEAD}"
NW, NH, R, T16 = 460, 352, 256, 16
N_SAMP = 24000
# KITTI timestamps are NANOseconds (span ~1.04e8 ns = 104 ms), unlike the
# EVK4 data eigenpyramid was written for (microseconds). Both constants are
# therefore scaled by 1000, giving the same physical 1 ms slice / 5 ms tau.
SLICE_NS = 1_000_000
TAU_NS = 5_000_000

# NOTE: eigenpyramid.py must NOT be imported -- it runs its entire falsifier
# sweep at module level (loads EVK4 data, fits logistic regressions). fast_tensor
# holds a parameterised copy, verified equal to it to 2e-12 (and without its
# hardcoded prev[4]/prev[16]/prev[64] scale keys).
SCALES = [4, 16, 64]


def key(x, y, t):
    return (t.astype(np.int64) * (NW * NH) + y.astype(np.int64) * NW
            + x.astype(np.int64))


def build_one(args):
    mpath, dst = args
    if os.path.exists(dst):
        return 0
    base = os.path.basename(mpath)
    try:
        with np.load(mpath) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        with np.load(f"{SRC}/raw_data/{base}") as d:
            clean = np.sort(key(d["x"], d["y"], d["t"]))
    except Exception:
        return 0
    n = len(x)
    if n < 500:
        return 0
    rain = ~np.isin(key(x, y, t), clean)
    n_bg, n_rn = int((~rain).sum()), int(rain.sum())
    if n_bg < 50 or n_rn < 50:
        return 0

    # mixed cells at OUR decision granularity
    sx = np.clip((x.astype(np.int64) * R) // NW, 0, R - 1)
    sy = np.clip((y.astype(np.int64) * R) // NH, 0, R - 1)
    t0, span = t.min(), max(int(t.max() - t.min()), 1)
    tn = ((t - t0) / span).astype(np.float32)
    tb = np.clip((tn * T16).astype(np.int64), 0, T16 - 1)
    cell = (tb * R + sy) * R + sx
    order = np.argsort(cell, kind="stable")
    c, r = cell[order], rain[order]
    b = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    cells, cnt = c[b], np.diff(np.r_[b, len(c)])
    rn_c = np.add.reduceat(r.astype(np.int64), b)
    mixed_cells = set(cells[(cnt - rn_c > 0) & (rn_c > 0)].tolist())
    is_mixed = np.fromiter((cc in mixed_cells for cc in cell), bool, n)

    # stratified sample: up to half mixed, remainder uniform over the rest
    rng = np.random.default_rng(abs(hash(base)) % (2 ** 31))
    idx_m = np.flatnonzero(is_mixed)
    idx_u = np.flatnonzero(~is_mixed)
    take_m = min(len(idx_m), N_SAMP // 2)
    take_u = min(len(idx_u), N_SAMP - take_m)
    sel_m = rng.choice(idx_m, take_m, replace=False) if take_m else \
        np.empty(0, np.int64)
    sel_u = rng.choice(idx_u, take_u, replace=False) if take_u else \
        np.empty(0, np.int64)
    sel = np.concatenate([sel_m, sel_u]).astype(np.int64)
    if len(sel) < 100:
        return 0
    # 1/p importance weights so the biased sample still estimates the full frame
    inv_p = np.concatenate([
        np.full(take_m, len(idx_m) / max(take_m, 1), np.float32),
        np.full(take_u, len(idx_u) / max(take_u, 1), np.float32)])

    tcols = tensor_cols_fast(x, y, t, sel, TAU_NS, SCALES, NW, NH,
                             SLICE_NS).astype(np.float16)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(
        dst, sel=sel.astype(np.int32), tcols=tcols,
        lab=(~rain[sel]).astype(np.int8),            # 1 = scene
        mixed=is_mixed[sel], inv_p=inv_p,
        n_bg=np.int64(n_bg), n_rn=np.int64(n_rn))
    return 1


def main():
    jobs = []
    for split in ("train", "val", "test"):
        for f in sorted(glob.glob(f"{PACK}/{split}/*/*.npz")):
            mm, base = f.split("/")[-2], os.path.basename(f)
            jobs.append((f"{SRC}/merge_data/{mm}/{base}",
                         f"{OUT}/{split}/{mm}/{base}"))
    print(f"{len(jobs)} frames", flush=True)
    with Pool(3) as pool:   # only 4 CPUs on this node, training is running
        done = 0
        for i, rres in enumerate(pool.imap_unordered(build_one, jobs,
                                                     chunksize=4)):
            done += rres
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  built {done}", flush=True)
    print(f"built {done} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
