#!/usr/bin/env python3
"""
The decisive number: does the K=2 sub-window OR channel add anything the
CURRENT input (the K=1 binary ON-mask, spatially pooled) does not already have?

Baseline  A: 15x15 box-mean of the K=1 mask   (== what ORSPNet's gate can see today)
Candidate B: A  +  15x15 box-mean of the K=2 soft count (m_A + m_B in {0,1,2})

Report per-pixel AUC of each, and the conditional AUC of the K=2 pooled feature
within deciles of the baseline. Class-balanced logistic regression gives DeltaDA.
"""
import sys, glob
import numpy as np

ROOT = "/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC"
S, W, H = 256, 640, 480
R = 7  # 15x15 box


def load(p):
    d = np.load(p); o = (d['x'], d['y'], d['t'].astype(np.int64), d['p']); d.close(); return o


def key(x, y, t, p):
    return t.astype(np.int64) * 1000003 + x.astype(np.int64) * 1009 + y.astype(np.int64) * 7 + (p > 0).astype(np.int64)


def box(img, r):
    ii = np.pad(img.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    Hh, Ww = img.shape
    ys = np.arange(Hh); xs = np.arange(Ww)
    y0 = np.clip(ys - r, 0, Hh); y1 = np.clip(ys + r + 1, 0, Hh)
    x0 = np.clip(xs - r, 0, Ww); x1 = np.clip(xs + r + 1, 0, Ww)
    A = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    n = np.outer(y1 - y0, x1 - x0)
    return A / n


def auc(score, lab):
    o = np.argsort(score, kind='mergesort'); s = score[o]
    rk = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        rk[i:j + 1] = (i + j + 2) / 2.0; i = j + 1
    r = np.empty(len(s)); r[o] = rk
    n1 = lab.sum(); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def balacc_logreg(X, yv, Xte, yte, iters=400, lr=0.5):
    """class-balanced logistic regression, returns balanced accuracy on test"""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd; Xts = (Xte - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))]); Xts = np.hstack([Xts, np.ones((len(Xts), 1))])
    w = np.zeros(Xs.shape[1])
    wt = np.where(yv == 1, 0.5 / max(yv.mean(), 1e-8), 0.5 / max(1 - yv.mean(), 1e-8))
    for _ in range(iters):
        z = Xs @ w; pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = Xs.T @ (wt * (pr - yv)) / len(yv)
        w -= lr * g * 10
    z = Xts @ w
    best = 0.5
    for q in np.quantile(z, np.linspace(0.01, 0.99, 99)):
        pd = z >= q
        best = max(best, 0.5 * (pd[yte == 1].mean() + (~pd[yte == 0]).mean()))
    return best


def frame_feats(rf, gf):
    x, y, t, p = load(rf); gx, gy, gt_, gp = load(gf)
    isbg = np.isin(key(x, y, t, p), key(gx, gy, gt_, gp))
    on = (p == 1)
    xo = (x[on] * S // W).astype(np.int32); yo = (y[on] * S // H).astype(np.int32)
    to = t[on]; bgo = isbg[on]
    t0, t1 = to.min(), to.max(); tm = (t0 + t1) // 2
    idx = yo.astype(np.int64) * S + xo
    m1 = np.zeros(S * S, np.uint8); m1[idx] = 1
    mA = np.zeros(S * S, np.uint8); mA[idx[to <= tm]] = 1
    mB = np.zeros(S * S, np.uint8); mB[idx[to > tm]] = 1
    cbg = np.bincount(idx[bgo], minlength=S * S)
    m1 = m1.reshape(S, S); s2 = (mA + mB).reshape(S, S).astype(np.float64)
    lit = m1 > 0
    keep = (cbg > 0).reshape(S, S)[lit].astype(np.float64)
    f_base = np.stack([box(m1, R)[lit], box(m1, 2)[lit], m1[lit].astype(np.float64)], 1)
    f_new = np.stack([box(s2, R)[lit], box(s2, 2)[lit], s2[lit]], 1)
    return f_base, f_new, keep


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["a1", "b2", "a3", "b4"]
    per_seq = {}
    for sq in seqs:
        rf = sorted(glob.glob(f"{ROOT}/SPAC-dataset-merge/events/{sq}_Rain/*.npz"))[:8]
        gf = sorted(glob.glob(f"{ROOT}/SPAC-dataset-event/gt/{sq}_GT/*.npz"))[:8]
        B, N, K = [], [], []
        for a, b in zip(rf, gf):
            fb, fn, kp = frame_feats(a, b); B.append(fb); N.append(fn); K.append(kp)
        per_seq[sq] = (np.vstack(B), np.vstack(N), np.concatenate(K))
        print(f"{sq}: {len(per_seq[sq][2])} lit pixels")

    for held in seqs:
        tr = [s for s in seqs if s != held]
        Xb = np.vstack([per_seq[s][0] for s in tr]); Xn = np.vstack([per_seq[s][1] for s in tr])
        yv = np.concatenate([per_seq[s][2] for s in tr])
        Tb, Tn, yt = per_seq[held]
        a = balacc_logreg(Xb, yv, Tb, yt)
        c = balacc_logreg(np.hstack([Xb, Xn]), yv, np.hstack([Tb, Tn]), yt)
        print(f"held-out {held}:  DA_base(K1 only)={a:.4f}   DA_base+K2={c:.4f}   Delta={c-a:+.4f}")

    # conditional AUC of pooled K2 within deciles of pooled K1
    Xb = np.vstack([per_seq[s][0] for s in seqs]); Xn = np.vstack([per_seq[s][1] for s in seqs])
    yv = np.concatenate([per_seq[s][2] for s in seqs])
    print(f"\nmarginal AUC pooled-K1 = {auc(Xb[:,0], yv):.4f}")
    print(f"marginal AUC pooled-K2 = {auc(Xn[:,0], yv):.4f}")
    q = np.quantile(Xb[:, 0], np.linspace(0, 1, 11))
    ca = []
    for i in range(10):
        m = (Xb[:, 0] >= q[i]) & (Xb[:, 0] <= q[i + 1])
        if m.sum() > 500 and 0 < yv[m].mean() < 1:
            ca.append(auc(Xn[m, 0], yv[m]))
    ca = np.array(ca)
    print("conditional AUC of pooled-K2 within pooled-K1 deciles:",
          np.array2string(ca, precision=4))
    print(f"  mean |AUC-0.5| = {np.nanmean(np.abs(ca-0.5)):.4f}   max = {np.nanmax(np.abs(ca-0.5)):.4f}")
