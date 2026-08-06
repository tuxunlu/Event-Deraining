"""Verify the research's RANK-1 identity, and test its RANK-2 free fix.

RANK 1 claims event-DA decomposes per cell as
    DA = 0.5 + 0.5 * sum_c d_c * (bg_c/N_bg - rn_c/N_rn)
so the optimal label is y_c = 1[bg_c/N_bg > rn_c/N_rn] and the cost of a wrong
decision is exactly |bg_c/N_bg - rn_c/N_rn|. Derivation (checked by hand):
    SR = sum_c d_c bg_c / N_bg
    NR = sum_c (1-d_c) rn_c / N_rn = 1 - sum_c d_c rn_c / N_rn
    DA = 0.5(SR+NR) = 0.5 + 0.5 sum_c d_c (bg_c/N_bg - rn_c/N_rn)
Part A checks this numerically on real packed frames.

RANK 2 claims the deployment threshold should be the EVENT-level prior --
the COUNT-weighted mean predicted probability over lit cells -- not the
unweighted cell mean we have been using as the self-prior rule. Part B tests
three thresholding rules on the already-trained checkpoint, no retraining:
    (a) one global tau selected on val   (the protocol number, 0.9183)
    (b) per-frame unweighted self-prior  (the campaign's existing rule)
    (c) per-frame COUNT-weighted self-prior             (RANK 2)
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

from rsp_3d import ORSPNet3D

ROOT = f"{C.KITTI_PACK}"
TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R, T_OUT = 16, 256, 16

# ---------------- Part A: the identity ----------------
print("=== A. RANK-1 identity check on real frames ===")
rng = np.random.default_rng(0)
files = sorted(glob.glob(f"{ROOT}/test/*/*.npz"))[::97][:12]
worst = 0.0
for f in files:
    with np.load(f) as d:
        bg = d["bg"].reshape(T_BUILD, R, R).astype(np.float64)
        rn = d["rn"].reshape(T_BUILD, R, R).astype(np.float64)
    k = T_BUILD // T_OUT
    bg = bg.reshape(T_OUT, k, R, R).sum(1)
    rn = rn.reshape(T_OUT, k, R, R).sum(1)
    n_bg, n_rn = bg.sum(), rn.sum()
    if n_bg < 1 or n_rn < 1:
        continue
    d_rand = rng.random(bg.shape) > 0.5                    # arbitrary decisions
    sr = (bg * d_rand).sum() / n_bg
    nr = (rn * ~d_rand).sum() / n_rn
    da_direct = 0.5 * (sr + nr)
    da_identity = 0.5 + 0.5 * (d_rand * (bg / n_bg - rn / n_rn)).sum()
    worst = max(worst, abs(da_direct - da_identity))
print(f"  max |direct - identity| over {len(files)} frames, random decisions: "
      f"{worst:.2e}   -> {'IDENTITY HOLDS' if worst < 1e-9 else 'MISMATCH'}")

# ---------------- Part B: threshold rules ----------------
print("\n=== B. RANK-2 free threshold fix (no retraining) ===")
ck = torch.load(f"{TMP}/k3de_T4o16_maj.pt", map_location="cpu")
m = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3, use_off=True,
              out_chans=16)
m.load_state_dict(ck["state_dict"])
m = m.to(DEV).eval()
tau_global = float(ck["tau"])
print(f"  checkpoint tau (val-selected) = {tau_global:.2f}")

acc = {"(a) global val tau": [], "(b) self-prior, unweighted": [],
       "(c) self-prior, COUNT-weighted": []}
with torch.no_grad():
    for f in sorted(glob.glob(f"{ROOT}/test/*/*.npz")):
        with np.load(f) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            bg = torch.from_numpy(d["bg"].reshape(T_BUILD, R, R).astype(np.float32))
            rn = torch.from_numpy(d["rn"].reshape(T_BUILD, R, R).astype(np.float32))
        on4 = torch.from_numpy(on.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
        off4 = torch.from_numpy(off.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
        bg, rn = bg.to(DEV), rn.to(DEV)
        nb, nr = float(bg.sum()), float(rn.sum())
        if nb < 1 or nr < 1:
            continue
        p = torch.sigmoid(m(on4, x_off=off4))[0]           # [16,R,R]
        cnt = bg + rn
        lit = cnt > 0
        taus = {
            "(a) global val tau": tau_global,
            "(b) self-prior, unweighted": float(p[lit].mean()),
            "(c) self-prior, COUNT-weighted":
                float((p * cnt).sum() / cnt.sum().clamp(min=1e-6)),
        }
        for name, t in taus.items():
            keep = p > t
            sr = float((bg * keep).sum()) / nb
            nrr = float((rn * ~keep).sum()) / nr
            acc[name].append(0.5 * (sr + nrr))

for name, v in acc.items():
    print(f"  {name:32s} test event-DA {np.mean(v):.4f}  ({len(v)} frames)")
print("  reference: protocol number for this checkpoint = 0.9183")
