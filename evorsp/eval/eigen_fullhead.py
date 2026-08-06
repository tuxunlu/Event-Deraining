"""Do the EigenPyramid columns improve the CLOCKED flagship head?
Full 110-dim head features + 19 structure-tensor columns (tau=5ms),
logreg, scenes 1-3 -> scene 4. If no gain: the trunk already encodes the
oriented statistic (expected -- it has a log-Gabor bank); the columns' value
is async-only. If gain: the flagship improves too."""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob, sys
import numpy as np, torch
sys.path.insert(0, ".")
from rsp_3d import ORSPNet3D
from sklearn.linear_model import LogisticRegression
from eigenpyramid import tensor_cols, count_cols   # reuse

DEV = "cuda"; T16, RW, RH = 16, 448, 256; NW, NH = 1280, 720
S = f"{C.REAL_SRC}"
TAU_US = 5000
rng = np.random.default_rng(0)

ck = torch.load("real_evorsp.pt", map_location="cpu")
m = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
m.load_state_dict(ck["state_dict"]); m = m.to(DEV).eval()
FEAT = {}
m.out_proj.register_forward_pre_hook(lambda mod, i: FEAT.__setitem__("f", i[0].detach()))

@torch.no_grad()
def frame(f, lp):
    lab = np.load(lp); d = np.load(f)
    x, y, t, p = d["x"], d["y"], d["t"], d["p"]
    if len(lab) != len(x) or len(x) < 2000: return None
    sx = (x.astype(np.int64) * RW) // NW; sy = (y.astype(np.int64) * RH) // NH
    t0 = t.min(); span = max(int(t.max() - t0), 1)
    tb16 = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
    tb4 = tb16 // 4; pol = (p == 1).astype(np.int64)
    on = np.zeros((T16, RH * RW), bool); off = np.zeros((T16, RH * RW), bool)
    sm = pol == 1
    on[tb16[sm], sy[sm] * RW + sx[sm]] = True
    off[tb16[~sm], sy[~sm] * RW + sx[~sm]] = True
    on4 = torch.from_numpy(on.reshape(T16, RH, RW).reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    off4 = torch.from_numpy(off.reshape(T16, RH, RW).reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    logit = m(on4, x_off=off4)[0, 0].cpu().numpy(); feat = FEAT["f"][0].cpu().numpy()
    u = (x + 0.5) * RW / NW - 0.5; v = (y + 0.5) * RH / NH - 0.5
    u0 = np.clip(np.floor(u).astype(np.int64), 0, RW - 2)
    v0 = np.clip(np.floor(v).astype(np.int64), 0, RH - 2)
    au = np.clip(u - u0, 0, 1); av = np.clip(v - v0, 0, 1)
    def bil(M):
        return ((M[..., v0, u0] * (1 - au) + M[..., v0, u0 + 1] * au) * (1 - av)
                + (M[..., v0 + 1, u0] * (1 - au) + M[..., v0 + 1, u0 + 1] * au) * av)
    G = np.zeros((8, NH, NW), np.uint8); np.add.at(G, (pol * 4 + tb4, y, x), 1)
    Gp = np.pad(G, ((0, 0), (1, 1), (1, 1)))
    patch = np.log1p(np.stack([Gp[:, y + dy, x + dx] for dy in range(3)
                               for dx in range(3)], 1).reshape(len(x), 72).astype(np.float32))
    sel = rng.choice(len(x), min(9000, len(x)), replace=False)
    meta = np.concatenate([pol[sel, None].astype(np.float32),
                           np.eye(4, dtype=np.float32)[tb4[sel]]], 1)
    Xhead = np.concatenate([bil(logit)[sel, None].astype(np.float32), patch[sel],
                            bil(feat).T[sel].astype(np.float32), meta], 1)
    Xt = tensor_cols(x, y, t, sel, TAU_US)
    return Xhead, Xt, lab[sel].astype(np.int64)

def collect(scenes, per_k):
    out = []
    for sc in scenes:
        for k in (1, 2, 3):
            for f in sorted(glob.glob(f"{S}/{sc}/merge_data/rain_{k}/*.npz"))[::14][:per_k]:
                lp = (f"{S}/{sc}/labels/labels_rain_{k}/labels_" + f.split("/")[-1]).replace(".npz", ".npy")
                try: r = frame(f, lp)
                except Exception: r = None
                if r is not None: out.append(r)
    return out

def ba(keep, lab):
    n_s = max(int((lab == 1).sum()), 1); n_r = max(int((lab == 0).sum()), 1)
    return 0.5 * (int((keep & (lab == 1)).sum()) / n_s
                  + int((~keep & (lab == 0)).sum()) / n_r)

tr = collect(("scene1", "scene2", "scene3"), 8); te = collect(("scene4",), 14)
print(f"train {len(tr)} test {len(te)} frames", flush=True)
for name, use_t in [("head only", False), ("head + EigenPyramid", True)]:
    Xtr = np.concatenate([np.concatenate([a, b], 1) if use_t else a for a, b, _ in tr])
    ytr = np.concatenate([l for _, _, l in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit((Xtr - mu) / sd, ytr)
    bas = []
    for a, b, l in te:
        X = np.concatenate([a, b], 1) if use_t else a
        pv = lr.predict_proba((X - mu) / sd)[:, 1]
        bas.append(ba(pv > pv.mean(), l))
    print(f"{name:24s} scene-disjoint event-BA {np.mean(bas):.4f}", flush=True)
