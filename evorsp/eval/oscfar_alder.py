"""EvOS-CFAR step 0 (the operating-point gate) + ALDER step 1 (trunk-free head).

EvOS-CFAR step 0 -- on held-out scene 4, decompose the operating-point headroom
for BOTH trunk p and head p:
    global self-prior tau  ->  per-FRAME oracle tau  ->  per-TILE oracle tau
    (squares 16/32/64/128 native px + full-height COLUMN strips 16/32/64)
  Per-tile oracle: exact best cut per tile under the frame's BA weights
  (BA decomposes over tiles, so independent per-tile choices are jointly
  optimal). PRE-REGISTERED KILL: (per-tile oracle - global self-prior) < 0.010
  event-BA => the whole operating-point niche (OS-CFAR, GMM, router,
  Carbone-Kay, Poisson-z, learned threshold) is dead.

ALDER step 1 -- logistic head refit WITHOUT trunk-derived dims (col 0 logit +
32 trunk feats), i.e. local-only evidence: 72-dim native patch + polarity +
4 time-bin one-hots = 77 dims. Variants:
    (ii) local-only float    (iii) 8-bit quantized weights+features
    (iv) quantized + counts read through a 1 Mb counting-hash (collision sim;
         head fitted unhashed, evaluated hashed -- approximation, stated)
  PRE-REGISTERED KILL: quantized+hashed scene-disjoint event-BA < 0.820.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from rsp_3d import ORSPNet3D

DEV = "cuda"
T16, RW, RH = 16, 448, 256
NW, NH = 1280, 720
S = f"{C.REAL_SRC}"
CACHE = f"{C.WORK / 'evhead_cache'}"
TMP = f"{C.CKPT}"
HASH_BITS = 20                       # 1M buckets

# ---------------- heads from the cache (CPU) ------------------------------
tr_f = sorted(glob.glob(f"{CACHE}/scene[123]_*.npz"))
Xtr = np.concatenate([np.load(f)["X"] for f in tr_f]).astype(np.float32)
ytr = np.concatenate([np.load(f)["lab"] for f in tr_f]).astype(np.float32)
LOCAL = list(range(1, 73)) + list(range(105, 110))           # patch + pol + bins
mu_f, sd_f = Xtr.mean(0), Xtr.std(0) + 1e-6
lr_full = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_full.fit((Xtr - mu_f) / sd_f, ytr)
Xl = Xtr[:, LOCAL]
mu_l, sd_l = Xl.mean(0), Xl.std(0) + 1e-6
lr_loc = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_loc.fit((Xl - mu_l) / sd_l, ytr)
# 8-bit quantization of the local head: weights and normalised features
w = lr_loc.coef_[0]
wq = np.round(w / (np.abs(w).max() / 127)) * (np.abs(w).max() / 127)
print(f"heads fitted on {len(ytr):,} rows "
      f"({len(tr_f)} train frames)", flush=True)

# ---------------- trunk ----------------------------------------------------
ck = torch.load(f"{TMP}/real_evorsp.pt", map_location="cpu")
m = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
m.load_state_dict(ck["state_dict"])
m = m.to(DEV).eval()
FEAT = {}
m.out_proj.register_forward_pre_hook(
    lambda mod, inp: FEAT.__setitem__("f", inp[0].detach()))


def ba(keep, lab):
    n_s = max(int((lab == 1).sum()), 1)
    n_r = max(int((lab == 0).sum()), 1)
    return 0.5 * (int((keep & (lab == 1)).sum()) / n_s
                  + int((~keep & (lab == 0)).sum()) / n_r)


def oracle_tiled(p, lab, tile_id):
    """Exact jointly-optimal per-tile thresholds under frame BA weights."""
    n_s = max(int((lab == 1).sum()), 1)
    n_r = max(int((lab == 0).sum()), 1)
    w_ev = np.where(lab == 1, 1.0 / n_s, -1.0 / n_r)
    order = np.argsort(tile_id, kind="mergesort")
    tid, pp, ww = tile_id[order], p[order], w_ev[order]
    cut = np.r_[0, np.nonzero(np.diff(tid))[0] + 1, len(tid)]
    total = 0.0
    for a, b in zip(cut[:-1], cut[1:]):
        po = np.argsort(pp[a:b])[::-1]                       # keep highest-p first
        cw = np.cumsum(ww[a:b][po])
        best = max(0.0, float(cw.max()) if b > a else 0.0)   # keep-none allowed
        total += best
    return 0.5 * (1.0 + total)                               # BA identity


rng = np.random.default_rng(0)
frames = []
for k in (1, 2, 3):
    frames += [(f, k) for f in
               sorted(glob.glob(f"{S}/scene4/merge_data/rain_{k}/*.npz"))[::6][:20]]

acc = {k: [] for k in
       ["glob_t", "frame_t", "tile_t", "col_t",
        "glob_h", "frame_h", "tile_h", "col_h",
        "full", "loc", "locq", "locqh"]}
tile_sizes = {"tile": 64, "col": 32}                          # best-of swept below
sweep = {sz: [] for sz in (16, 32, 64, 128)}
sweep_c = {sz: [] for sz in (16, 32, 64)}

with torch.no_grad():
    for f, k in frames:
        base = f.split("/")[-1]
        lab = np.load(f"{S}/scene4/labels/labels_rain_{k}/labels_{base}"
                      .replace(".npz", ".npy"))
        d = np.load(f)
        x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        if len(lab) != len(x) or len(x) < 2000:
            continue
        sx = (x.astype(np.int64) * RW) // NW
        sy = (y.astype(np.int64) * RH) // NH
        t0 = t.min()
        span = max(int(t.max() - t0), 1)
        tb16 = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
        tb4 = tb16 // 4
        pol = (p == 1).astype(np.int64)
        on = np.zeros((T16, RH * RW), bool)
        off = np.zeros((T16, RH * RW), bool)
        sm = pol == 1
        on[tb16[sm], sy[sm] * RW + sx[sm]] = True
        off[tb16[~sm], sy[~sm] * RW + sx[~sm]] = True
        on4 = torch.from_numpy(on.reshape(T16, RH, RW)
              .reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
        off4 = torch.from_numpy(off.reshape(T16, RH, RW)
               .reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
        logit = m(on4, x_off=off4)[0, 0].cpu().numpy()
        feat = FEAT["f"][0].cpu().numpy()
        u = (x + 0.5) * RW / NW - 0.5
        v = (y + 0.5) * RH / NH - 0.5
        u0 = np.clip(np.floor(u).astype(np.int64), 0, RW - 2)
        v0 = np.clip(np.floor(v).astype(np.int64), 0, RH - 2)
        au = np.clip(u - u0, 0, 1)
        av = np.clip(v - v0, 0, 1)
        def bil(M):
            return ((M[..., v0, u0] * (1 - au) + M[..., v0, u0 + 1] * au)
                    * (1 - av)
                    + (M[..., v0 + 1, u0] * (1 - au)
                       + M[..., v0 + 1, u0 + 1] * au) * av)
        p_tr = 1 / (1 + np.exp(-bil(logit)))
        f_bil = bil(feat).T

        G = np.zeros((8, NH, NW), np.uint8)
        np.add.at(G, (pol * 4 + tb4, y, x), 1)
        Gp = np.pad(G, ((0, 0), (1, 1), (1, 1)))
        patch = np.log1p(np.stack([Gp[:, y + dy, x + dx] for dy in range(3)
                                   for dx in range(3)], 1)
                         .reshape(len(x), 72).astype(np.float32))
        # hashed counts: rebuild G through a 1 Mb counting hash, then patches
        key = ((pol * 4 + tb4).astype(np.int64) * 6151
               + y.astype(np.int64) * 2654435761 + x.astype(np.int64) * 40503)
        Hb = np.zeros(1 << HASH_BITS, np.uint8)
        np.add.at(Hb, (key & ((1 << HASH_BITS) - 1)), 1)
        # per-event 3x3 hashed patch
        ph = np.zeros((len(x), 72), np.float32)
        c = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for ci in range(8):
                    kk = (ci * 6151
                          + (y + dy).clip(0, NH - 1).astype(np.int64) * 2654435761
                          + (x + dx).clip(0, NW - 1).astype(np.int64) * 40503)
                    ph[:, c] = Hb[kk & ((1 << HASH_BITS) - 1)]
                    c += 1
        ph = np.log1p(ph)

        meta = np.concatenate([pol[:, None].astype(np.float32),
                               np.eye(4, dtype=np.float32)[tb4]], 1)
        Xf = np.concatenate([bil(logit)[:, None].astype(np.float32),
                             patch, f_bil.astype(np.float32), meta], 1)
        p_full = lr_full.predict_proba((Xf - mu_f) / sd_f)[:, 1]
        Xloc = np.concatenate([patch, meta], 1)
        Zl = (Xloc - mu_l) / sd_l
        p_loc = lr_loc.predict_proba(Zl)[:, 1]
        # 8-bit: quantize normalized features to int8 grid and use wq
        Zq = np.clip(np.round(Zl * 32), -127, 127) / 32
        p_locq = 1 / (1 + np.exp(-(Zq @ wq + lr_loc.intercept_[0])))
        Xlh = np.concatenate([ph, meta], 1)
        Zh = np.clip(np.round(((Xlh - mu_l) / sd_l) * 32), -127, 127) / 32
        p_locqh = 1 / (1 + np.exp(-(Zh @ wq + lr_loc.intercept_[0])))

        for tag, pv in [("t", p_tr), ("h", p_full)]:
            acc[f"glob_{tag}"].append(ba(pv > pv.mean(), lab))
            ts = np.quantile(pv, np.linspace(0.02, 0.98, 49))
            acc[f"frame_{tag}"].append(max(ba(pv > tt, lab) for tt in ts))
            for sz in sweep:
                tid = (y // sz) * 4096 + (x // sz)
                sweep[sz].append((tag, oracle_tiled(pv, lab, tid)))
            for sz in sweep_c:
                sweep_c[sz].append((tag, oracle_tiled(pv, lab, x // sz)))
        for name, pv in [("full", p_full), ("loc", p_loc),
                         ("locq", p_locq), ("locqh", p_locqh)]:
            acc[name].append(ba(pv > pv.mean(), lab))

n = len(acc["glob_t"])
print(f"\n{n} scene-4 frames\n")
print("=== EvOS-CFAR STEP 0 ===")
for tag, lbl in [("t", "trunk p"), ("h", "head p")]:
    g = np.mean(acc[f"glob_{tag}"])
    fr = np.mean(acc[f"frame_{tag}"])
    best_sq = max([(np.mean([v for tg, v in sweep[sz] if tg == tag]), sz)
                   for sz in sweep])
    best_co = max([(np.mean([v for tg, v in sweep_c[sz] if tg == tag]), sz)
                   for sz in sweep_c])
    print(f"{lbl}: global self-prior {g:.4f} | frame-oracle {fr:.4f} | "
          f"tile-oracle {best_sq[0]:.4f} (sq{best_sq[1]}) | "
          f"col-oracle {best_co[0]:.4f} (w{best_co[1]})")
    gap = max(best_sq[0], best_co[0]) - g
    print(f"    GAP = {gap:+.4f}   "
          f"{'NICHE LIVE (>=0.010)' if gap >= 0.010 else 'NICHE DEAD (<0.010)'}")
print("\n=== ALDER STEP 1 ===")
print(f"full head (reference)          {np.mean(acc['full']):.4f}")
print(f"local-only (77-dim)            {np.mean(acc['loc']):.4f}")
print(f"local-only 8-bit               {np.mean(acc['locq']):.4f}")
print(f"local-only 8-bit + 1Mb hash    {np.mean(acc['locqh']):.4f}   "
      f"KILL if < 0.820: "
      f"{'PASS' if np.mean(acc['locqh']) >= 0.820 else 'KILL'}")
