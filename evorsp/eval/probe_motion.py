"""Candidate features for the streaks we still miss, ranked by separability.

Reading the four comparison frames together gives a pattern the aggregate number
hides:

  real4 scene2/rain_5   DIAGONAL thin streaks   -> we remove them well
  real1 scene4/rain_13  thin streaks            -> mostly removed, faint remnants
  real2 scene1/rain_2   thick VERTICAL columns  -> substantial residue
  real3 scene3/rain_9   dense VERTICAL striping -> worst; heavy residue

Failure tracks VERTICAL + dense + spatially recurrent, not rain in general.

Physical hypothesis this suggests: a falling drop writes a trail on which y
GROWS WITH t. A static scene edge -- a pole, a window frame -- has no such
space-time slope, however vertical it looks. Our structure tensor is purely
SPATIAL (x, y), so it can see that a streak is vertical but not that it is
FALLING. That is a cue we have never given the model.

Measured here against the population that actually fails (persistent rain vs
scene), with the same >= 0.60 bar the ITI probe used, and with the already-known
ITI result as the reference to beat:
    burstiness (16 px)  AUC 0.864
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

import numpy as np
import torch


S = f"{C.REAL_SRC}"
DEV = "cuda"
NW, NH = 1280, 720
SEQS = [("scene1", "rain_2"), ("scene3", "rain_9"),
        ("scene4", "rain_13"), ("scene2", "rain_5")]
NFR = 8
SCALES = (4, 16)


def motion_cols(x, y, t, scales=SCALES):
    """Per-tile space-time statistics: corr(y,t), corr(x,t), |dy/dt|, aniso."""
    xg = torch.from_numpy(x.astype(np.float64)).to(DEV)
    yg = torch.from_numpy(y.astype(np.float64)).to(DEV)
    tg = torch.from_numpy(t.astype(np.float64)).to(DEV)
    tg = (tg - tg.min()) / max(float(tg.max() - tg.min()), 1.0)   # -> [0,1]
    n = len(x)
    out = torch.zeros(n, 4 * len(scales), device=DEV, dtype=torch.float64)
    for si, s in enumerate(scales):
        wt = NW // s + 1
        tile = ((yg.long() // s) * wt + (xg.long() // s))
        nt = wt * (NH // s + 1)

        def acc(v):
            o = torch.zeros(nt, device=DEV, dtype=torch.float64)
            o.index_add_(0, tile, v)
            return o

        one = torch.ones_like(xg)
        c = acc(one)
        sx, sy, st = acc(xg), acc(yg), acc(tg)
        sxx, syy, stt = acc(xg * xg), acc(yg * yg), acc(tg * tg)
        syt, sxt = acc(yg * tg), acc(xg * tg)
        cc = torch.clamp(c, min=1)
        mx, my, mt = sx / cc, sy / cc, st / cc
        vx = torch.clamp(sxx / cc - mx * mx, min=0)
        vy = torch.clamp(syy / cc - my * my, min=0)
        vt = torch.clamp(stt / cc - mt * mt, min=1e-12)
        cyt = syt / cc - my * mt
        cxt = sxt / cc - mx * mt
        ok = c >= 4
        r_yt = torch.where(ok, cyt / torch.sqrt(torch.clamp(vy * vt, min=1e-12)),
                           torch.zeros_like(vy))
        r_xt = torch.where(ok, cxt / torch.sqrt(torch.clamp(vx * vt, min=1e-12)),
                           torch.zeros_like(vx))
        slope = torch.where(ok, cyt / vt, torch.zeros_like(vy))     # dy/dt
        aniso = torch.where(ok, (vy - vx) / torch.clamp(vy + vx, min=1e-9),
                            torch.zeros_like(vy))                   # verticality
        k = si * 4
        out[:, k + 0] = r_yt[tile]
        out[:, k + 1] = r_xt[tile]
        out[:, k + 2] = torch.log1p(torch.abs(slope[tile]))
        out[:, k + 3] = aniso[tile]
    return out.float().cpu().numpy()


NAMES = [f"{n}_s{s}" for s in SCALES
         for n in ("corr_y_t", "corr_x_t", "log|dy/dt|", "vertical_aniso")]


def auc(pos, neg):
    if len(pos) < 100 or len(neg) < 100:
        return np.nan
    k = min(len(pos), len(neg), 20000)
    rp = np.random.default_rng(0).choice(pos, k, replace=False)
    rn = np.random.default_rng(1).choice(neg, k, replace=False)
    r = np.argsort(np.argsort(np.concatenate([rp, rn])))
    a = (r[:k].sum() - k * (k - 1) / 2) / (k * k)
    return max(a, 1 - a)


F = {n: {"pr": [], "sc": []} for n in NAMES}
nfr = 0
for sc_, lv in SEQS:
    files = sorted(glob.glob(f"{S}/{sc_}/merge_data/{lv}/*.npz"))
    step = max(1, len(files) // NFR)
    for f in files[::step][:NFR]:
        base = os.path.basename(f)
        idx = int(base.split(".")[0])
        lp = f"{S}/{sc_}/labels/labels_{lv}/labels_{base}".replace(".npz", ".npy")
        pvf = f"{S}/{sc_}/merge_data/{lv}/{idx-1:010d}.npz"
        if not (os.path.exists(lp) and os.path.exists(pvf) and idx > 0):
            continue
        with np.load(f) as d:
            x, y, t = d["x"], d["y"], d["t"]
        lab = np.load(lp).astype(np.int64)
        if len(lab) != len(x) or len(x) < 5000:
            continue
        occ = np.zeros((NH, NW), bool)
        with np.load(pvf) as d:
            occ[d["y"], d["x"]] = True
        pers = occ[y, x]
        is_sc = lab == 1
        M = motion_cols(x, y, t)
        for i, nm in enumerate(NAMES):
            F[nm]["pr"].append(M[(~is_sc) & pers, i])
            F[nm]["sc"].append(M[is_sc, i])
        nfr += 1

print(f"\n=== SPACE-TIME MOTION CUES vs the failing population ({nfr} frames) ===")
print(f"  {'feature':18s} {'persistent rain':>16s} {'scene':>10s} {'AUC':>7s}")
best = ("", 0.0)
for nm in NAMES:
    pr = np.concatenate(F[nm]["pr"])
    sc2 = np.concatenate(F[nm]["sc"])
    a = auc(pr, sc2)
    if np.isfinite(a) and a > best[1]:
        best = (nm, a)
    print(f"  {nm:18s} {np.median(pr):16.3f} {np.median(sc2):10.3f} {a:7.3f}")
print(f"\n  best motion cue: {best[0]} AUC {best[1]:.3f}")
print(f"  reference: ITI burstiness (16 px) AUC 0.864; bar to clear 0.60")
