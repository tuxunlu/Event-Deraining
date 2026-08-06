#!/usr/bin/env python3
"""
Probe: the eFFT accumulator is a binary OR over ON-events.
Measure how much per-pixel information a K-sub-window OR split restores.
Zero training. CPU only.
"""
import sys, glob, os
import numpy as np

ROOT = "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC"
S = 256
W, H = 640, 480


def load(p):
    d = np.load(p)
    out = (d['x'], d['y'], d['t'].astype(np.int64), d['p'])
    d.close()
    return out


def key(x, y, t, p):
    # exact event identity
    return (t.astype(np.int64) * 1000003 + x.astype(np.int64) * 1009 + y.astype(np.int64) * 7 + (p > 0).astype(np.int64))


def masks(xs, ys, ts, K, t0, t1):
    """K binary OR masks over sub-windows, returned as [K, S, S] uint8."""
    m = np.zeros((K, S * S), dtype=np.uint8)
    b = np.clip(((ts - t0) * K // max(t1 - t0, 1)).astype(np.int64), 0, K - 1)
    idx = ys * S + xs
    m[b, idx] = 1
    return m


def auc(score, lab):
    # lab: 1 = positive
    o = np.argsort(score, kind='mergesort')
    r = np.empty(len(score), dtype=np.float64)
    s = score[o]
    rk = np.arange(1, len(score) + 1, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rk[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    r[o] = rk
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def bal_acc_best_threshold(score, lab):
    """best achievable balanced accuracy by thresholding score (oracle threshold)."""
    vals = np.unique(score)
    best = 0.5
    for v in vals:
        pred = score >= v
        tpr = pred[lab == 1].mean() if (lab == 1).any() else 0
        tnr = (~pred[lab == 0]).mean() if (lab == 0).any() else 0
        best = max(best, 0.5 * (tpr + tnr))
    return best


def run_seq(seq, nframes=6):
    rain_dir = f"{ROOT}/SPAC-dataset-merge/events/{seq}_Rain"
    gt_dir = f"{ROOT}/SPAC-dataset-event/gt/{seq}_GT"
    rf = sorted(glob.glob(rain_dir + "/*.npz"))
    gf = sorted(glob.glob(gt_dir + "/*.npz"))
    n = min(len(rf), len(gf), nframes)
    res = []
    for i in range(n):
        x, y, t, p = load(rf[i])
        gx, gy, gt_, gp = load(gf[i])
        kr = key(x, y, t, p)
        kg = key(gx, gy, gt_, gp)
        isbg = np.isin(kr, kg)  # True = real background event
        rec = isbg.sum() / max(len(kg), 1)
        # ---- the representation actually sees ONLY p==1 events ----
        on = (p == 1)
        xo = (x[on] * S // W).astype(np.int32)
        yo = (y[on] * S // H).astype(np.int32)
        to = t[on]
        bgo = isbg[on]
        t0, t1 = to.min(), to.max()
        idx = yo.astype(np.int64) * S + xo
        # true per-pixel ON count, and per-pixel background-ON count
        cnt = np.bincount(idx, minlength=S * S)
        cnt_bg = np.bincount(idx[bgo], minlength=S * S)
        lit = cnt > 0
        # pixel label: 1 = pixel should be KEPT (has >=1 true background ON event)
        keep = (cnt_bg > 0)[lit].astype(np.int64)
        out = {'seq': seq, 'f': i, 'rec': rec, 'nlit': int(lit.sum()),
               'lit_frac': lit.mean(), 'keep_frac': keep.mean(),
               'rain_evt_frac': 1 - isbg.mean()}
        for K in (1, 2, 4, 8):
            m = masks(xo, yo, to, K, t0, t1)
            s = m.sum(0).astype(np.float64)[lit]
            out[f'auc_K{K}'] = auc(s, keep)
            out[f'da_K{K}'] = bal_acc_best_threshold(s, keep)
        c = cnt[lit].astype(np.float64)
        out['auc_truecount'] = auc(c, keep)
        out['da_truecount'] = bal_acc_best_threshold(np.log1p(c), keep)
        res.append(out)
    return res


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["a1", "b2", "a3", "b4"]
    allr = []
    for s in seqs:
        r = run_seq(s)
        allr += r
        for d in r:
            print(f"{d['seq']}[{d['f']}] gtrec={d['rec']:.4f} lit={d['lit_frac']:.3f} "
                  f"keepfrac={d['keep_frac']:.3f} rainevt={d['rain_evt_frac']:.3f} | "
                  f"AUC K1={d['auc_K1']:.4f} K2={d['auc_K2']:.4f} K4={d['auc_K4']:.4f} "
                  f"K8={d['auc_K8']:.4f} true={d['auc_truecount']:.4f} | "
                  f"DA K1={d['da_K1']:.4f} K2={d['da_K2']:.4f} K4={d['da_K4']:.4f} "
                  f"K8={d['da_K8']:.4f} true={d['da_truecount']:.4f}")
    import collections
    print("\n=== MEANS ===")
    for k in ['auc_K1', 'auc_K2', 'auc_K4', 'auc_K8', 'auc_truecount',
              'da_K1', 'da_K2', 'da_K4', 'da_K8', 'da_truecount', 'lit_frac', 'keep_frac']:
        print(f"  {k}: {np.nanmean([d[k] for d in allr]):.4f}")
