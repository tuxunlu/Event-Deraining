#!/usr/bin/env python3
"""
THE DECISIVE TEST.

The clock-invariant sub-frame descriptor (POP over K bins = how many sub-windows
this pixel fired in) is worth +0.086 balanced accuracy over per-pixel COUNT alone.
But ORSPNet does not classify from per-pixel count -- it classifies from SPATIAL
context (its only proven lever was receptive field).

So: condition the empirical-Bayes table on a spatial context descriptor that a
conv net trivially has, and ask whether the temporal descriptor adds anything ON
TOP of it.

Conditioner C = ( min(n,8), oct(net count in 3x3), oct(net count in 15x15) )
Compare  BA[C]  vs  BA[C x POP_K].   Delta is the sub-frame headroom in DA units.
Seed noise on our task is +-0.0039.
"""
import numpy as np
from scipy.ndimage import uniform_filter

ROOT = '/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
S, W0, H0 = 256, 640, 480
POPC = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.int64)


def load(seq, idx):
    r = np.load(f'{ROOT}/SPAC-dataset-merge/events/{seq}_Rain/{idx:010d}.npz')
    g = np.load(f'{ROOT}/SPAC-dataset-event/gt/{seq}_GT/{idx:010d}.npz')
    key = lambda d: (((d['t'].astype(np.int64)) * 641 + d['x']) * 481 + d['y']) * 3 + (d['p'] > 0)
    isbg = np.isin(key(r), key(g))
    t = r['t'].astype(np.int64); o = np.argsort(t, kind='mergesort')
    return (r['x'][o].astype(np.int64), r['y'][o].astype(np.int64),
            t[o], r['p'][o].astype(np.int64), isbg[o])


def table(x, y, t, p, isbg, K):
    t0 = t.min(); T = max(t.max() - t0 + 1, 1)
    b = np.clip(((t - t0) * K) // T, 0, K - 1).astype(np.int64)
    xs = np.clip(x * S // W0, 0, S - 1); ys = np.clip(y * S // H0, 0, S - 1)
    pid = xs * S + ys
    sgn = np.where(p > 0, 1, -1)
    merge = np.bincount(pid, weights=sgn, minlength=S * S)
    clean = np.bincount(pid[isbg], weights=sgn[isbg], minlength=S * S)
    cnt = np.bincount(pid, minlength=S * S)
    img = merge.reshape(S, S)
    n3 = uniform_filter(img, 3, mode='nearest').ravel() * 9
    n15 = uniform_filter(img, 15, mode='nearest').ravel() * 225
    lit = merge > 0.5
    idx = np.flatnonzero(lit)
    if K == 1:
        w = np.zeros(len(idx), dtype=np.int64)
    else:
        occ = np.bincount(pid * K + b, minlength=S * S * K).reshape(S * S, K) > 0
        w = (occ[idx] * (1 << np.arange(K))).sum(1).astype(np.int64)
    pop = POPC[w & 0xFFFF] + POPC[(w >> 16) & 0xFFFF]
    return dict(n=cnt[idx], pop=pop, n3=n3[idx], n15=n15[idx],
                keep=(clean > 0.5)[idx].astype(np.int8))


def oct_(v, edges):  # quantile bucket
    return np.digitize(v, edges)


def ba(tr, te, code_tr, code_te):
    M = int(max(code_tr.max(), code_te.max())) + 1
    wk = 1.0 / max(tr['keep'].sum(), 1); wd = 1.0 / max((1 - tr['keep']).sum(), 1)
    nk = np.bincount(code_tr[tr['keep'] == 1], minlength=M) * wk
    nd = np.bincount(code_tr[tr['keep'] == 0], minlength=M) * wd
    dec = (nk + 0.5 * wk) > (nd + 0.5 * wd)
    pr = dec[code_te]; yy = te['keep'] == 1
    sr = pr[yy].mean(); nr = (~pr[~yy]).mean()
    return 0.5 * (sr + nr), sr, nr


def cat(ds): return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


if __name__ == '__main__':
    seqs = ['a1', 'a3', 'a4', 'b1', 'b2', 'b4']
    frames = list(range(2, 22))
    raw = {}
    for s in seqs:
        for f in frames:
            try: raw[(s, f)] = load(s, f)
            except Exception: pass
    tr_k = [k for k in raw if k[1] % 2 == 0]; te_k = [k for k in raw if k[1] % 2 == 1]
    print(f'{len(raw)} frames | train {len(tr_k)} test {len(te_k)}')
    print('\nK   dt(ms)  BA[count]  BA[count+POP]  BA[spatial]  BA[spatial+POP]   DELTA')
    for K in [1, 2, 4, 8, 16, 32]:
        tr = cat([table(*raw[k], K) for k in tr_k])
        te = cat([table(*raw[k], K) for k in te_k])
        e3 = np.quantile(tr['n3'], np.linspace(0, 1, 9)[1:-1])
        e15 = np.quantile(tr['n15'], np.linspace(0, 1, 9)[1:-1])
        def C(d, withpop):
            c = np.minimum(d['n'], 8)
            c = c * 9 + oct_(d['n3'], e3)
            c = c * 9 + oct_(d['n15'], e15)
            return c * (K + 1) + d['pop'] if withpop else c
        def Cn(d, withpop):
            c = np.minimum(d['n'], 8)
            return c * (K + 1) + d['pop'] if withpop else c
        b_c, _, _ = ba(tr, te, Cn(tr, False), Cn(te, False))
        b_cp, _, _ = ba(tr, te, Cn(tr, True), Cn(te, True))
        b_s, sr, nr = ba(tr, te, C(tr, False), C(te, False))
        b_sp, sr2, nr2 = ba(tr, te, C(tr, True), C(te, True))
        print(f'{K:<3} {103.6/K:6.2f}   {b_c:.4f}     {b_cp:.4f}       {b_s:.4f}      '
              f'{b_sp:.4f}      {b_sp-b_s:+.4f}')
