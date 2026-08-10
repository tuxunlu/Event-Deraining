"""Feature + pseudo-label cache for unlabelled wild rain (scene5).

Pseudo-labels use the rule calibrated in pseudo_calibrate2.py against rig
ground truth -- K=4 future windows, burstiness quantile 0.85:

    scene := lit in ALL 4 following windows                 precision 0.949
    rain  := lit in NONE of them AND burstiness > q(0.85)   precision 0.934
    else  := unlabelled, excluded from the loss             coverage  0.212

The future windows are the supervision the model cannot see at inference, which
is what makes this self-supervised rather than a confirmation loop.

Stores per frame the same tensors the rig cache holds, so the adapter can reuse
run_real_full's forward path unchanged, plus `pl` (pseudo-label) and `pm`
(confident mask).
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_feats import patch_gpu, tensor_gpu
from iti_feats import iti_gpu
from recur_feats import Recur

WILD = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_realworld"
OUT = "/fs/nexus-scratch/tuxunlu/wild_cache"
DEV = "cuda"
NW, NH = 1280, 720
R, T16 = 256, 16
K_FUT, Q_BURST = 4, 0.85
NSEL = 20000                  # events sampled per frame, as in the rig cache
STRIDE = 4                    # every 4th frame -- ~2.4k frames is ample
MIN_EV = 5000
MIN_PER_CLASS = 500    # skip frames too imbalanced to sample evenly
BURST_COL = 7


def occ_of(path):
    with np.load(path) as d:
        o = np.zeros((NH, NW), bool)
        o[d["y"], d["x"]] = True
    return o


def main():
    seqs = sorted(glob.glob(f"{WILD}/scene5/merge_data/rain_*"))
    rec = Recur(nw=NW, nh=NH)
    done = 0
    for sq in seqs:
        files = sorted(glob.glob(f"{sq}/*.npz"))
        if len(files) < K_FUT + 2:
            continue
        rec.reset()
        lv = os.path.basename(sq)
        for i, fp in enumerate(files):
            with np.load(fp) as d:
                x, y, t, p = d["x"], d["y"], d["t"], d["p"]
            xi, yi = x.astype(np.int64), y.astype(np.int64)
            want = (i % STRIDE == 0) and len(x) >= MIN_EV and i + K_FUT < len(files)
            dst = f"{OUT}/{lv}/{i:010d}.npz"
            if not want or os.path.exists(dst):
                rec.push(xi, yi)            # buffer stays causal and dense
                continue

            # ---- pseudo-labels from the FUTURE (never an input at inference)
            hits = np.zeros((len(x), K_FUT), bool)
            for k in range(1, K_FUT + 1):
                hits[:, k - 1] = occ_of(files[i + k])[yi, xi]
            tg = torch.from_numpy(t.astype(np.int64)).to(DEV)
            xg, yg = torch.from_numpy(xi).to(DEV), torch.from_numpy(yi).to(DEV)
            pg = torch.from_numpy(p.astype(np.int64)).to(DEV)
            burst = iti_gpu(xg, yg, tg, nw=NW, nh=NH)[:, BURST_COL]
            thr = torch.quantile(burst.float(), Q_BURST)
            ps = hits.all(1)
            pr = (~hits.any(1)) & (burst > thr).cpu().numpy()
            conf = ps | pr
            # BALANCED sampling, equal events per class. Wild persists far more
            # than the rig (mean future-hit 0.63 vs 0.42), so the same rule
            # yields up to 18:1 scene:rain here against 1.2:1 there. Sampling
            # uniformly from the confident set inherits that skew and the model
            # simply learns to keep everything.
            idx = np.arange(len(x))
            s_idx, r_idx = idx[ps], idx[pr & ~ps]
            per = min(len(s_idx), len(r_idx), NSEL // 2)
            if per < MIN_PER_CLASS:
                rec.push(xi, yi)
                continue
            rng = np.random.default_rng(i)
            sel = np.sort(np.concatenate([
                rng.choice(s_idx, per, replace=False),
                rng.choice(r_idx, per, replace=False)]))

            t0 = t.min()
            span = max(int(t.max() - t0), 1)
            tn = ((t - t0) / span).astype(np.float32)
            tns = torch.from_numpy(tn).to(DEV)
            pv = patch_gpu(xg, yg, tns, pg, NW, NH)[sel].cpu().numpy()
            tc = tensor_gpu(xg, yg, tg, 5_000, [4, 16, 64], NW, NH,
                            1_000)[sel].cpu().numpy()
            it = iti_gpu(xg, yg, tg, nw=NW, nh=NH)[sel].cpu().numpy()
            rc = rec.features(xi, yi, sel)
            rec.push(xi, yi)

            # ---- occupancy planes for the trunk
            sx = (xi * R) // NW
            sy = (yi * R) // NH
            tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
            on = np.zeros(T16 * R * R, bool)
            off = np.zeros(T16 * R * R, bool)
            s = p == 1
            cell = tb * (R * R) + sy * R + sx
            on[cell[s]] = True
            off[cell[~s]] = True

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(
                dst, on=np.packbits(on), off=np.packbits(off),
                sel=sel.astype(np.int32), x=x[sel].astype(np.int16),
                y=y[sel].astype(np.int16), tn=tn[sel],
                patch=pv.astype(np.float16), tcols=tc.astype(np.float16),
                iti=it.astype(np.float16), recur=rc.astype(np.float16),
                pl=ps[sel].astype(np.uint8), pm=conf[sel].astype(np.uint8))
            done += 1
            if done % 100 == 0:
                print(f"  {done} frames cached ({lv})", flush=True)
    print(f"cached {done} wild frames -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
