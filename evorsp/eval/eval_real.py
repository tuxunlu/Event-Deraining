"""Zero-shot evaluation of every KITTI-trained model on REAL EVK4 rain.

No retraining, no re-tuning: each model runs at the threshold selected by its
own KITTI protocol (the deployment condition). The oracle-best tau is reported
alongside as an upper bound, clearly marked -- it uses test labels and is NOT a
deployment number.

DA = 1/2(SR + NR) over lit pixels, the same construction as every other number
in this project. Real rain here is a hard domain shift: EVK4 sensor (not
simulation), 1280x720 optics, and full-height vertical streaks unlike the
short synthetic streaks either training set contains.
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


TMP = f"{C.CKPT}"
ROOT = f"{C.WORK / 'real_t16'}"
DEV = "cuda"
T, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)


def build(kind):
    if kind == "dffn":
        from model.DynamicFourierFilterNet import DynamicFourierFilterNet
        return DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32, num_blocks=4)
    if kind == "fmamba":
        from model.FourierMamba2D import FourierMamba2D
        return FourierMamba2D(in_chans=1, out_chans=1, dim=32, num_blocks=[2, 2, 2, 2])
    if kind == "orsp_dil":
        from rsp_model_v2 import ORSPNet
        return ORSPNet(dilations=(1, 8, 32, 64))
    if kind == "streaknet":
        from rsp_streak import StreakNet
        return StreakNet(K=127, use_strip=True, use_rate=True, use_darkmask=True)
    if kind == "evorsp3t":
        from rsp_3d import ORSPNet3D
        return ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
    raise ValueError(kind)


MODELS = [
    ("dffn",      "proto_dffn",          "dffn",      "DFFN (2D)"),
    ("fmamba",    "ckpt_fmamba",         "fmamba",    "FourierMamba2D (2D)*"),
    ("orsp_best", "proto_orsp_bal_dil",  "orsp_dil",  "ORSPNet+dil+bal (2D)"),
    ("streaknet", "proto_streaknet_bal", "streaknet", "StreakNet (2D)"),
    ("evorsp",    "k3d_T4b3off",         "evorsp3t",  "EvORSP-3T (temporal+OFF)"),
]

nets, meta = {}, {}
for k, f, b, lbl in MODELS:
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = build(b)
    m.load_state_dict(ck["state_dict"])
    nets[k] = m.to(DEV).eval()
    meta[k] = {"tau": float(ck.get("test_tau", ck.get("tau", 0.5))), "label": lbl}


@torch.no_grad()
def run():
    files = sorted(glob.glob(f"{ROOT}/*/rain_*/*.npz"))
    print(f"{len(files)} labeled real frames\n", flush=True)
    acc = {k: [np.zeros(len(TAUS)), np.zeros(len(TAUS)), 0] for k in nets}
    for i, f in enumerate(files):
        with np.load(f) as d:
            on = np.unpackbits(d["on"])[: T * R * R].reshape(T, R, R)
            off = np.unpackbits(d["off"])[: T * R * R].reshape(T, R, R)
            gt = np.unpackbits(d["gt"])[: R * R].reshape(R, R) > 0
        on4 = torch.from_numpy(on.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
        off4 = torch.from_numpy(off.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
        merge = on4.amax(1, keepdim=True)                 # the 2D models' input
        lit = merge[0, 0] > 0.5
        gt_t = torch.from_numpy(gt).to(DEV)
        real = gt_t & lit
        rain = lit & ~gt_t
        rs, ns = int(real.sum()), int(rain.sum())
        if rs < 50 or ns < 50:
            continue
        for k in nets:
            if k == "evorsp":
                p = torch.sigmoid(nets[k](on4, x_off=off4))[0, 0]
            else:
                p = torch.sigmoid(nets[k](merge))[0, 0]
            for j, t in enumerate(TAUS):
                pr = p > t
                acc[k][0][j] += float((pr & real).sum()) / rs
                acc[k][1][j] += (ns - float((pr & rain).sum())) / ns
            acc[k][2] += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(files)}", flush=True)

    print(f"\n{'model':<26s} {'@KITTI tau':>11s} {'oracle-best':>12s}  (n = frames)")
    print("-" * 60)
    for k in nets:
        sr, nr, n = acc[k]
        da = 0.5 * (sr + nr) / max(n, 1)
        jk = max(0, min(len(TAUS) - 1, int(np.round((meta[k]["tau"] - .05) / .05))))
        print(f"{meta[k]['label']:<26s} {da[jk]:>11.4f} {da.max():>12.4f}   ({n})")
    print("\n@KITTI tau = zero-shot deployment; oracle-best uses test labels "
          "(upper bound only).")


run()
