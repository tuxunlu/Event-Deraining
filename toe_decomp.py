#!/usr/bin/env python3
"""
TOE decomposition: WHAT carries the sub-frame gain?

Per LIT pixel, occupancy bitmask w over K bins of the 104 ms window.
Four descriptors, each a strictly weaker (more invariant) view of w:

  ABS   = w                      absolute bin identity  -> can read a fixed clock
  CANON = rotate(w) to canonical shift-invariant form    -> shape, no clock phase
  SPAN  = last_bin - first_bin   temporal extent         -> intermittency only
  POP   = popcount(w)            number of occupied bins -> concentration only

plus controls:
  PHASE : per-frame random circular roll of t before binning (kills a clock
          locked to the window origin, preserves every relative timing)
  XSEQ  : train on {a1,a3,a4}, test on {b1,b2,b4}  (different scenes/renders)

BA(K=1) is the count-only structural zero for every column.
"""
import numpy as np

ROOT = '/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
S, W0, H0 = 256, 640, 480


def load(seq, idx):
    r = np.load(f'{ROOT}/SPAC-dataset-merge/events/{seq}_Rain/{idx:010d}.npz')
    g = np.load(f'{ROOT}/SPAC-dataset-event/gt/{seq}_GT/{idx:010d}.npz')
    key = lambda d: (((d['t'].astype(np.int64)) * 641 + d['x']) * 481 + d['y']) * 3 + (d['p'] > 0)
    isbg = np.isin(key(r), key(g))
    t = r['t'].astype(np.int64)
    o = np.argsort(t, kind='mergesort')
    return (r['x'][o].astype(np.int64), r['y'][o].astype(np.int64),
            t[o], r['p'][o].astype(np.int64), isbg[o])


POPC = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.int8)


def canon(w, K):
    """canonical (min over circular rotations) form of a K-bit word array"""
    best = w.copy()
    cur = w.copy()
    m = (1 << K) - 1
    for _ in range(K - 1):
        cur = ((cur >> 1) | ((cur & 1) << (K - 1))) & m
        best = np.minimum(best, cur)
    return best


def table(x, y, t, p, isbg, K, phase=None, strip=False):
    if strip:
        ut, inv, cts = np.unique(t, return_inverse=True, return_counts=True)
        k = (cts[inv] < 5) & (t != t.max())
        x, y, t, p, isbg = x[k], y[k], t[k], p[k], isbg[k]
    t0, t1 = t.min(), t.max()
    T = max(t1 - t0 + 1, 1)
    u = (t - t0)
    if phase is not None:
        u = (u + int(phase * T)) % T
    b = np.clip((u * K) // T, 0, K - 1).astype(np.int64)

    xs = np.clip(x * S // W0, 0, S - 1); ys = np.clip(y * S // H0, 0, S - 1)
    pid = xs * S + ys
    sgn = np.where(p > 0, 1, -1)
    merge_net = np.bincount(pid, weights=sgn, minlength=S * S)
    clean_net = np.bincount(pid[isbg], weights=sgn[isbg], minlength=S * S)
    cnt = np.bincount(pid, minlength=S * S)
    lit = merge_net > 0.5
    keep = (clean_net > 0.5)[lit].astype(np.int8)
    idx = np.flatnonzero(lit)

    if K == 1:
        w = np.zeros(len(idx), dtype=np.int64)
    else:
        occ = np.bincount(pid * K + b, minlength=S * S * K).reshape(S * S, K) > 0
        w = (occ[idx] * (1 << np.arange(K))).sum(1).astype(np.int64)
    return dict(n=cnt[idx], w=w, keep=keep)


def ba(tr, te, feat, ncap=8):
    ftr, fte = feat(tr), feat(te)
    mx = int(max(ftr.max(), fte.max())) + 1
    ctr = np.minimum(tr['n'], ncap) * mx + ftr
    cte = np.minimum(te['n'], ncap) * mx + fte
    M = int(max(ctr.max(), cte.max())) + 1
    wk = 1.0 / max(tr['keep'].sum(), 1); wd = 1.0 / max((1 - tr['keep']).sum(), 1)
    nk = np.bincount(ctr[tr['keep'] == 1], minlength=M) * wk
    nd = np.bincount(ctr[tr['keep'] == 0], minlength=M) * wd
    dec = (nk + 0.5 * wk) > (nd + 0.5 * wd)
    pr = dec[cte]; yy = te['keep'] == 1
    return 0.5 * (pr[yy].mean() + (~pr[~yy]).mean())


def cat(ds): return {k: np.concatenate([d[k] for d in ds]) for k in ('n', 'w', 'keep')}


F_ABS = lambda d: d['w']
F_POP = lambda d: POPC[d['w'] & 0xFFFF].astype(np.int64) + POPC[(d['w'] >> 16) & 0xFFFF]


def mkspan(K):
    def f(d):
        w = d['w']
        if K == 1: return np.zeros(len(w), dtype=np.int64)
        lo = np.zeros(len(w), dtype=np.int64); hi = np.zeros(len(w), dtype=np.int64)
        for i in range(K):
            bit = (w >> i) & 1
            hi = np.where(bit == 1, i, hi)
            lo = np.where((bit == 1) & (lo == 0) & (hi == i), i, lo)
        # recompute lo properly
        lo = np.full(len(w), K, dtype=np.int64)
        for i in range(K - 1, -1, -1):
            lo = np.where(((w >> i) & 1) == 1, i, lo)
        return np.clip(hi - lo, 0, K)
    return f


def mkcanon(K):
    return lambda d: canon(d['w'], K) if K > 1 else np.zeros(len(d['w']), dtype=np.int64)


if __name__ == '__main__':
    A = ['a1', 'a3', 'a4']; B = ['b1', 'b2', 'b4']
    frames = list(range(2, 22))
    raw = {}
    for s in A + B:
        for f in frames:
            try: raw[(s, f)] = load(s, f)
            except Exception: pass
    print(f'{len(raw)} frames loaded')
    rng = np.random.default_rng(0)
    phases = {k: rng.random() for k in raw}

    tr_k = [k for k in raw if k[1] % 2 == 0]
    te_k = [k for k in raw if k[1] % 2 == 1]
    xtr_k = [k for k in raw if k[0] in A]
    xte_k = [k for k in raw if k[0] in B]

    print('\nK   dt(ms) | ABS    CANON  SPAN   POP   | ABS+phase | ABS xseq | ABS strip')
    for K in [1, 2, 4, 8, 16]:
        get = lambda ks, **kw: cat([table(*raw[k], K, **kw) for k in ks])
        tr, te = get(tr_k), get(te_k)
        r = [ba(tr, te, F_ABS), ba(tr, te, mkcanon(K)), ba(tr, te, mkspan(K)), ba(tr, te, F_POP)]
        trp = cat([table(*raw[k], K, phase=phases[k]) for k in tr_k])
        tep = cat([table(*raw[k], K, phase=phases[k]) for k in te_k])
        rp = ba(trp, tep, F_ABS)
        rx = ba(get(xtr_k), get(xte_k), F_ABS)
        trs, tes = get(tr_k, strip=True), get(te_k, strip=True)
        rs = ba(trs, tes, F_ABS)
        print(f'{K:<3} {103.6/K:6.2f} | {r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} |'
              f'   {rp:.4f}  |  {rx:.4f}  |  {rs:.4f}')
