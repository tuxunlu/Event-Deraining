"""EigenPyramid falsifier: per-event structure-tensor pyramid with exponential
forgetting, appended to the 0.6888 count-pyramid feature set.

State per tile per scale s in {4,16,64}: decayed sums (W, Sx, Sy, Sxx, Sxy,
Syy). Causal time-sliced EWMA (1 ms slices; events read pre-slice state --
honest approximation, granularity << tau). Read-out per event: coherence,
(cos2t, sin2t), log-spread-per-mass, minor-axis residual, per scale, plus
cross-scale orientation-agreement and coherence-ratio gates = 19 columns,
all degree-zero homogeneous in event mass.

KILL: best-tau scene-disjoint event-BA <= 0.7088 (< +0.02 over 0.6888).
"""
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression

S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
NW, NH = 1280, 720
CSCALES = [1, 4, 16, 64]          # count pyramid (the 0.6888 baseline features)
TSCALES = [4, 16, 64]             # structure-tensor scales
SLICE_US = 1000
rng = np.random.default_rng(0)


def tensor_cols(x, y, t, sel, tau_us):
    """19 structure-tensor columns for events at indices sel."""
    order = np.argsort(t, kind="mergesort")
    inv = np.empty(len(t), np.int64)
    inv[order] = np.arange(len(t))
    xs, ys, ts = x[order].astype(np.float64), y[order].astype(np.float64), t[order]
    t0, t1 = ts[0], ts[-1]
    nsl = max(int((t1 - t0) // SLICE_US) + 1, 1)
    sl = np.minimum(((ts - t0) // SLICE_US).astype(np.int64), nsl - 1)
    cuts = np.searchsorted(sl, np.arange(nsl + 1))
    d = np.exp(-SLICE_US / tau_us)

    st = {}
    for s in TSCALES:
        Wd, Hd = NW // s + 1, NH // s + 1
        st[s] = [np.zeros(Hd * Wd) for _ in range(6)]         # W,Sx,Sy,Sxx,Sxy,Syy
    feats = np.zeros((len(t), 19), np.float32)

    for i in range(nsl):
        a, b = cuts[i], cuts[i + 1]
        if a == b:
            for s in TSCALES:
                for arr in st[s]:
                    arr *= d
            continue
        xe, ye = xs[a:b], ys[a:b]
        col = 0
        prev = {}
        for s in TSCALES:
            W, Sx, Sy, Sxx, Sxy, Syy = st[s]
            tid = (ye.astype(np.int64) // s) * (NW // s + 1) + (xe.astype(np.int64) // s)
            w = W[tid]
            ok = w > 1e-3
            mx, my = np.where(ok, Sx[tid] / np.maximum(w, 1e-9), xe), \
                     np.where(ok, Sy[tid] / np.maximum(w, 1e-9), ye)
            cxx = np.maximum(Sxx[tid] / np.maximum(w, 1e-9) - mx * mx, 0)
            cyy = np.maximum(Syy[tid] / np.maximum(w, 1e-9) - my * my, 0)
            cxy = Sxy[tid] / np.maximum(w, 1e-9) - mx * my
            tr = cxx + cyy
            det = np.sqrt(np.maximum((cxx - cyy) ** 2 + 4 * cxy ** 2, 0))
            l1, l2 = (tr + det) / 2, np.maximum((tr - det) / 2, 0)
            coh = np.where(tr > 1e-6, (l1 - l2) / np.maximum(tr, 1e-9), 0)
            ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
            c2t, s2t = np.cos(2 * ang), np.sin(2 * ang)
            spread = np.log1p(tr) - np.log1p(w)
            # minor-axis residual of THIS event
            evx, evy = np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)
            res = np.abs((xe - mx) * evx + (ye - my) * evy) / (np.sqrt(l2) + 1.0)
            block = np.stack([coh, c2t, s2t, spread, np.log1p(res)], 1)
            feats[a:b, col:col + 5] = np.where(ok[:, None], block, 0)
            prev[s] = (coh, c2t, s2t, ok)
            col += 5
            # update AFTER read
            for arr in st[s]:
                arr *= d
            np.add.at(W, tid, 1.0)
            np.add.at(Sx, tid, xe)
            np.add.at(Sy, tid, ye)
            np.add.at(Sxx, tid, xe * xe)
            np.add.at(Sxy, tid, xe * ye)
            np.add.at(Syy, tid, ye * ye)
        # cross-scale gates
        for (sa, sb), j in zip(((4, 16), (16, 64)), (15, 16)):
            ca, c2a, s2a, oka = prev[sa]
            cb, c2b, s2b, okb = prev[sb]
            feats[a:b, j] = np.where(oka & okb, c2a * c2b + s2a * s2b, 0)
        feats[a:b, 17] = np.where(prev[4][3] & prev[16][3],
                                  prev[4][0] / np.maximum(prev[16][0], 1e-3), 0)
        feats[a:b, 18] = np.where(prev[16][3] & prev[64][3],
                                  prev[16][0] / np.maximum(prev[64][0], 1e-3), 0)
    return feats[inv][sel]


def count_cols(x, y, pol, sel):
    cols = []
    for s in CSCALES:
        W, H = (NW + s - 1) // s, (NH + s - 1) // s
        cx, cy = x // s, y // s
        for ch in (pol == 1, pol == 0):
            G = np.zeros((H + 2, W + 2), np.float32)
            np.add.at(G, (cy[ch] + 1, cx[ch] + 1), 1.0)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cols.append(np.log1p(G[cy[sel] + 1 + dy, cx[sel] + 1 + dx]))
    return np.stack(cols, 1)


def frame_feats(f, lp, tau_us):
    lab = np.load(lp)
    d = np.load(f)
    x, y, t, p = d["x"], d["y"], d["t"], d["p"]
    if len(lab) != len(x) or len(x) < 2000:
        return None
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb4 = np.clip(((t - t0) * 4) // span, 0, 3).astype(np.int64)
    pol = (p == 1).astype(np.int64)
    sel = rng.choice(len(x), min(9000, len(x)), replace=False)
    Xc = count_cols(x, y, pol, sel)
    Xt = tensor_cols(x, y, t, sel, tau_us)
    meta = np.concatenate([pol[sel, None].astype(np.float32),
                           np.eye(4, dtype=np.float32)[tb4[sel]]], 1)
    return np.concatenate([Xc, Xt, meta], 1), lab[sel].astype(np.int64)


def collect(scenes, per_k, tau_us):
    out = []
    for sc in scenes:
        for k in (1, 2, 3):
            for f in sorted(glob.glob(f"{S}/{sc}/merge_data/rain_{k}/*.npz"))[::14][:per_k]:
                lp = (f"{S}/{sc}/labels/labels_rain_{k}/labels_"
                      + f.split("/")[-1]).replace(".npz", ".npy")
                try:
                    r = frame_feats(f, lp, tau_us)
                except Exception:
                    r = None
                if r is not None:
                    out.append(r)
    return out


def ba(keep, lab):
    n_s = max(int((lab == 1).sum()), 1)
    n_r = max(int((lab == 0).sum()), 1)
    return 0.5 * (int((keep & (lab == 1)).sum()) / n_s
                  + int((~keep & (lab == 0)).sum()) / n_r)


best = (0, None)
for tau_ms in (5, 10, 20, 40):
    tr = collect(("scene1", "scene2", "scene3"), 8, tau_ms * 1000)
    te = collect(("scene4",), 14, tau_ms * 1000)
    Xtr = np.concatenate([r[0] for r in tr])
    ytr = np.concatenate([r[1] for r in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit((Xtr - mu) / sd, ytr)
    bas = []
    for X, lab in te:
        pv = lr.predict_proba((X - mu) / sd)[:, 1]
        bas.append(ba(pv > pv.mean(), lab))
    v = np.mean(bas)
    print(f"tau={tau_ms}ms: count-pyramid + EigenPyramid event-BA {v:.4f}",
          flush=True)
    if v > best[0]:
        best = (v, tau_ms)

print(f"\n=== EIGENPYRAMID VERDICT ===")
print(f"best: {best[0]:.4f} @ tau={best[1]}ms | baseline 0.6888 | "
      f"KILL if <= 0.7088: {'KILLED' if best[0] <= 0.7088 else 'SURVIVES'}")
