#!/usr/bin/env python3
"""
Temporal Oracle Envelope (TOE)
==============================
Non-parametric, training-free upper bound on how much DA the SUB-FRAME temporal
axis can add, per LIT pixel, at any resolution Delta.

Two parts:
  PART A -- metric-structure audit.  Our DA defines
        gt_b    = clean_lit                       (KEEP class)
        rain_gt = merge_lit AND NOT clean_lit     (DROP class)
    so the DROP class is BY CONSTRUCTION the set of pixels with no clean signal.
    We measure the mixed-pixel fraction: KEEP pixels that also contain rain
    events.  Those are pixels where a rain *presence* detector is definitionally
    wrong.

  PART B -- TOE.  Per LIT pixel, form a K-bin binary occupancy word over the
    104 ms window (K = 1,2,4,...).  Fit the EMPIRICAL BAYES lookup table
    P(KEEP | n, word) on a train split of frames, evaluate BALANCED ACCURACY on
    a held-out split.  K=1 is exactly the count-only control (word is constant),
    so BA(K) - BA(1) is the sub-frame contribution in DA units, with a built-in
    structural zero.  Also run a within-window timestamp-shuffle control and a
    simultaneity-stripped variant (renderer-artefact removal).
"""
import numpy as np, sys, os

ROOT = '/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
S = 256
W0, H0 = 640, 480


def load(seq, idx):
    r = np.load(f'{ROOT}/SPAC-dataset-merge/events/{seq}_Rain/{idx:010d}.npz')
    g = np.load(f'{ROOT}/SPAC-dataset-event/gt/{seq}_GT/{idx:010d}.npz')

    def key(d):
        return (((d['t'].astype(np.int64)) * 641 + d['x']) * 481 + d['y']) * 3 + (d['p'] > 0)
    kg = key(g)
    isbg = np.isin(key(r), kg)
    rec = isbg.sum() / len(kg)
    t = r['t'].astype(np.int64)
    o = np.argsort(t, kind='mergesort')
    return (r['x'][o].astype(np.int64), r['y'][o].astype(np.int64), t[o],
            r['p'][o].astype(np.int64), isbg[o], rec)


def pixel_table(x, y, t, p, isbg, K, strip_sim=False, shuffle=False, rng=None):
    """Returns per-LIT-pixel: n (net-signed lit test), label(KEEP=1), occupancy word."""
    if strip_sim:
        # drop events whose exact timestamp is shared by >=5 events FOV-wide,
        # and the terminal clamp dump
        ut, inv, cts = np.unique(t, return_inverse=True, return_counts=True)
        keep = (cts[inv] < 5) & (t != t.max())
        x, y, t, p, isbg = x[keep], y[keep], t[keep], p[keep], isbg[keep]
    if shuffle:
        t = t[rng.permutation(len(t))]
        o = np.argsort(t, kind='mergesort')
        x, y, t, p, isbg = x[o], y[o], t[o], p[o], isbg[o]

    xs = np.clip(x * S // W0, 0, S - 1)
    ys = np.clip(y * S // H0, 0, S - 1)
    pid = xs * S + ys
    t0, t1 = t.min(), t.max()
    b = np.clip(((t - t0) * K) // max(t1 - t0 + 1, 1), 0, K - 1).astype(np.int64)

    # signed accumulation, as the eFFT/IFFT pipeline sees it
    sgn = np.where(p > 0, 1, -1)
    merge_net = np.bincount(pid, weights=sgn, minlength=S * S)
    clean_net = np.bincount(pid[isbg], weights=sgn[isbg], minlength=S * S)
    cnt_all = np.bincount(pid, minlength=S * S)
    cnt_rain = np.bincount(pid[~isbg], minlength=S * S)
    cnt_bg = np.bincount(pid[isbg], minlength=S * S)

    merge_lit = merge_net > 0.5
    clean_lit = clean_net > 0.5
    lit = merge_lit | clean_lit          # union == the pixels DA sums over
    keep = clean_lit                     # gt_b
    drop = merge_lit & ~clean_lit        # rain_gt

    # occupancy word per pixel
    word = np.zeros(S * S, dtype=np.int64)
    if K > 1:
        code = pid * K + b
        occ = np.bincount(code, minlength=S * S * K).reshape(S * S, K) > 0
        word = (occ * (1 << np.arange(K))).sum(1)

    idx = np.flatnonzero(lit & (merge_lit))   # only pixels visible to the model
    return dict(n=cnt_all[idx], word=word[idx], keep=keep[idx].astype(np.int8),
                nrain=cnt_rain[idx], nbg=cnt_bg[idx],
                n_keep=int(keep[lit & merge_lit].sum()),
                n_drop=int(drop[lit & merge_lit].sum()),
                n_lit_total=int(lit.sum()), n_dark=int(S * S - lit.sum()))


def bal_acc_bayes(tr, te, ncap=8):
    """Empirical-Bayes lookup on (min(n,ncap), word). Balanced accuracy on te."""
    def code(d):
        return np.minimum(d['n'], ncap) * (d['word'].max() + 1 if d['word'].max() > 0 else 1) + d['word']
    mx = max(tr['word'].max(), te['word'].max()) + 1
    ctr = np.minimum(tr['n'], ncap) * mx + tr['word']
    cte = np.minimum(te['n'], ncap) * mx + te['word']
    M = int(max(ctr.max(), cte.max())) + 1
    # class-balanced posterior (so the decision rule targets balanced accuracy)
    wk = 1.0 / max(tr['keep'].sum(), 1)
    wd = 1.0 / max((1 - tr['keep']).sum(), 1)
    nk = np.bincount(ctr[tr['keep'] == 1], minlength=M) * wk
    nd = np.bincount(ctr[tr['keep'] == 0], minlength=M) * wd
    a = 0.5  # Laplace
    dec = (nk + a * wk) > (nd + a * wd)
    pred = dec[cte]
    y = te['keep'] == 1
    sr = pred[y].mean() if y.sum() else np.nan
    nr = (~pred[~y]).mean() if (~y).sum() else np.nan
    return 0.5 * (sr + nr), sr, nr


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ('n', 'word', 'keep', 'nrain', 'nbg')}


if __name__ == '__main__':
    seqs = ['a1', 'a3', 'b1', 'b2', 'b4']
    frames = list(range(2, 22))
    Ks = [1, 2, 4, 8, 16, 32]
    rng = np.random.default_rng(0)

    raw = {}
    for s in seqs:
        for f in frames:
            try:
                raw[(s, f)] = load(s, f)
            except Exception:
                pass
    print(f'loaded {len(raw)} frames; gt recovery '
          f'{min(v[5] for v in raw.values()):.4f}-{max(v[5] for v in raw.values()):.4f}')

    # ---------- PART A : metric structure ----------
    tot_keep = tot_drop = tot_lit = tot_dark = 0
    mixed = keep_pix = 0
    drop_with_bg = drop_pix = 0
    for k, (x, y, t, p, isbg, rec) in raw.items():
        d = pixel_table(x, y, t, p, isbg, 1)
        tot_keep += d['n_keep']; tot_drop += d['n_drop']
        tot_lit += d['n_lit_total']; tot_dark += d['n_dark']
        m = d['keep'] == 1
        keep_pix += m.sum(); mixed += (d['nrain'][m] > 0).sum()
        md = d['keep'] == 0
        drop_pix += md.sum(); drop_with_bg += (d['nbg'][md] > 0).sum()
    print('\n=== PART A: metric structure (256x256, signed-count lit test) ===')
    print(f'  dark pixels                       : {tot_dark/(tot_dark+tot_lit):.3f}')
    print(f'  KEEP-class pixels (gt_b)          : {tot_keep}')
    print(f'  DROP-class pixels (rain_gt)       : {tot_drop}   '
          f'({tot_drop/(tot_keep+tot_drop):.3f} of lit)')
    print(f'  MIXED: KEEP pixels containing rain: {mixed/max(keep_pix,1):.4f}')
    print(f'  DROP pixels containing any bg evt : {drop_with_bg/max(drop_pix,1):.4f}')

    # ---------- PART B : TOE ----------
    print('\n=== PART B: Temporal Oracle Envelope, balanced acc on held-out frames ===')
    print('  K    dt(ms)   BA_raw   BA_stripped  BA_shuffled')
    tr_keys = [k for k in raw if k[1] % 2 == 0]
    te_keys = [k for k in raw if k[1] % 2 == 1]
    for K in Ks:
        row = []
        for mode in ('raw', 'strip', 'shuf'):
            tr = cat([pixel_table(*raw[k][:5], K,
                                  strip_sim=(mode == 'strip'),
                                  shuffle=(mode == 'shuf'), rng=rng) for k in tr_keys])
            te = cat([pixel_table(*raw[k][:5], K,
                                  strip_sim=(mode == 'strip'),
                                  shuffle=(mode == 'shuf'), rng=rng) for k in te_keys])
            ba, sr, nr = bal_acc_bayes(tr, te)
            row.append(ba)
        print(f'  {K:<4} {103.6/K:7.2f}  {row[0]:.4f}     {row[1]:.4f}      {row[2]:.4f}')
