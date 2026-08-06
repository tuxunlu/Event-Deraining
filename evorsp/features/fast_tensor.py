"""Vectorised drop-in for eigenpyramid.tensor_cols.

Identical mathematics, two mechanical changes:
  * np.add.at(arr, tid, v)  ->  arr += np.bincount(tid, weights=v, minlength=n)
    np.add.at is an unbuffered ufunc scatter and is 10-50x slower than bincount.
  * the six per-scale accumulators are held in one (6, ntile) array so the decay
    is a single multiply instead of six.

Equivalence against the original is asserted in __main__ (max |diff| < 1e-6).
Kept separate from eigenpyramid.py so the falsifier result that module produced
(0.7504 event-BA) remains reproducible from the code that produced it.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import numpy as np


def tensor_cols_fast(x, y, t, sel, tau, scales, nw, nh, slice_len):
    """19 structure-tensor columns for events at indices `sel`.

    Causal: an event reads the state accumulated by STRICTLY earlier slices.
    """
    order = np.argsort(t, kind="mergesort")
    inv = np.empty(len(t), np.int64)
    inv[order] = np.arange(len(t))
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    ts = t[order]

    t0, t1 = ts[0], ts[-1]
    nsl = max(int((t1 - t0) // slice_len) + 1, 1)
    sl = np.minimum(((ts - t0) // slice_len).astype(np.int64), nsl - 1)
    cuts = np.searchsorted(sl, np.arange(nsl + 1))
    decay = np.exp(-slice_len / tau)

    st, ntile = {}, {}
    for s in scales:
        n = (nh // s + 1) * (nw // s + 1)
        st[s] = np.zeros((6, n))                    # W, Sx, Sy, Sxx, Sxy, Syy
        ntile[s] = n
    feats = np.zeros((len(t), 19), np.float32)

    for i in range(nsl):
        a, b = cuts[i], cuts[i + 1]
        if a == b:
            for s in scales:
                st[s] *= decay
            continue
        xe, ye = xs[a:b], ys[a:b]
        col, prev = 0, {}
        for s in scales:
            A = st[s]
            tid = (ye.astype(np.int64) // s) * (nw // s + 1) \
                + (xe.astype(np.int64) // s)
            w = A[0][tid]
            ok = w > 1e-3
            wsafe = np.maximum(w, 1e-9)
            mx = np.where(ok, A[1][tid] / wsafe, xe)
            my = np.where(ok, A[2][tid] / wsafe, ye)
            cxx = np.maximum(A[3][tid] / wsafe - mx * mx, 0)
            cyy = np.maximum(A[5][tid] / wsafe - my * my, 0)
            cxy = A[4][tid] / wsafe - mx * my
            tr = cxx + cyy
            det = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4 * cxy ** 2, 0))
            l1, l2 = (tr + det) / 2, np.maximum((tr - det) / 2, 0)
            coh = np.where(tr > 1e-6, (l1 - l2) / np.maximum(tr, 1e-9), 0)
            ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
            c2t, s2t = np.cos(2 * ang), np.sin(2 * ang)
            spread = np.log1p(tr) - np.log1p(w)
            evx, evy = np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)
            res = np.abs((xe - mx) * evx + (ye - my) * evy) / (np.sqrt(l2) + 1.0)
            block = np.stack([coh, c2t, s2t, spread, np.log1p(res)], 1)
            feats[a:b, col:col + 5] = np.where(ok[:, None], block, 0)
            prev[s] = (coh, c2t, s2t, ok)
            col += 5

            A *= decay                              # decay, then insert
            n = ntile[s]
            A[0] += np.bincount(tid, minlength=n)
            A[1] += np.bincount(tid, weights=xe, minlength=n)
            A[2] += np.bincount(tid, weights=ye, minlength=n)
            A[3] += np.bincount(tid, weights=xe * xe, minlength=n)
            A[4] += np.bincount(tid, weights=xe * ye, minlength=n)
            A[5] += np.bincount(tid, weights=ye * ye, minlength=n)

        sa, sb, sc = scales
        for (p, q), j in zip(((sa, sb), (sb, sc)), (15, 16)):
            ca, c2a, s2a, oka = prev[p]
            cb, c2b, s2b, okb = prev[q]
            feats[a:b, j] = np.where(oka & okb, c2a * c2b + s2a * s2b, 0)
        feats[a:b, 17] = np.where(prev[sa][3] & prev[sb][3],
                                  prev[sa][0] / np.maximum(prev[sb][0], 1e-3), 0)
        feats[a:b, 18] = np.where(prev[sb][3] & prev[sc][3],
                                  prev[sb][0] / np.maximum(prev[sc][0], 1e-3), 0)
    return feats[inv][sel]


if __name__ == "__main__":
    import time

    NW, NH, SCALES, SLICE, TAU = 460, 352, [4, 16, 64], 1_000_000, 5_000_000
    # the reference hardcodes prev[4]/prev[16]/prev[64] in its cross-scale gates,
    # so it only runs at those scales; the fast version is parameterised.
    # eigenpyramid.py runs its whole falsifier sweep at module level, so it must
    # NOT be imported. Lift just the reference function out of the source text.
    src = open(f"{C.CKPT}/eigenpyramid.py").read()
    body = src[src.index("def tensor_cols"):src.index("def count_cols")]
    ns = {"np": np, "NW": NW, "NH": NH, "TSCALES": SCALES, "SLICE_US": SLICE}
    exec(body, ns)
    ref_fn = ns["tensor_cols"]

    rng = np.random.default_rng(0)
    n = 6000
    x = rng.integers(0, NW, n)
    y = rng.integers(0, NH, n)
    t = np.sort(rng.integers(0, 20_000_000, n))     # 20 ms -> 20 slices
    sel = np.arange(0, n, 7)

    t0 = time.time()
    ref = ref_fn(x, y, t, sel, TAU)
    t_ref = time.time() - t0
    t0 = time.time()
    got = tensor_cols_fast(x, y, t, sel, TAU, SCALES, NW, NH, SLICE)
    t_fast = time.time() - t0
    d = np.abs(ref - got).max()
    print(f"  original {t_ref:6.2f}s | fast {t_fast:6.2f}s | "
          f"speedup {t_ref / max(t_fast, 1e-9):5.1f}x")
    print(f"  max |diff| = {d:.3e}  -> {'MATCH' if d < 1e-6 else 'MISMATCH'}")
