"""Can an INDEPENDENT signal rescue the pseudo-RAIN class?

pseudo_calibrate.py failed its gate in a specific, diagnosable way: the scene
side is reliable (precision 0.949 at K=4) but the rain side is not (0.688).
Non-persistence does not imply rain -- scene produces transient events too.

The fix is not to lower the bar. It is to sharpen the negative class with a
signal that is INDEPENDENT of persistence. ITI burstiness qualifies: it measures
timing regularity within a tile, was validated separately at AUC 0.864, and is
computed from inter-arrival times rather than from cross-frame occupancy.

    pseudo-RAIN  :=  (lit in NONE of the next K windows) AND (burstiness > q)
    pseudo-SCENE :=  (lit in ALL of the next K windows)          [unchanged]

NEW PRE-REGISTERED GATE, fixed before running:

    precision(rain) >= 0.85  AND  precision(scene) >= 0.85  AND
    coverage >= 0.10        (relaxed from 0.15 ONLY because requiring two
                             independent signals to agree must cost coverage;
                             the precision bar is NOT relaxed)

If no (K, q) passes, the conclusion is that this signal pair cannot produce
trainable pseudo-labels, and the adapter should not be built on it. Report and
stop -- do not try a third signal.
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iti_feats import iti_gpu

RIG = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
DEV = "cuda"
NW, NH = 1280, 720
KS = (2, 3, 4)
QS = (0.50, 0.70, 0.85)        # burstiness quantile, per frame
NFR = 50
MIN_EV = 2000
BURST_COL = 7                  # burstiness at the 16 px scale


def occ_of(path):
    with np.load(path) as d:
        o = np.zeros((NH, NW), bool)
        o[d["y"], d["x"]] = True
    return o


def main():
    seqs = [("scene4", f"rain_{k}") for k in (2, 6, 13)]
    kmax = max(KS)
    cell = {(k, q): dict(tp_s=0, n_s=0, tp_r=0, n_r=0, cov=0, tot=0)
            for k in KS for q in QS}
    nfr = 0
    for sc, lv in seqs:
        files = sorted(glob.glob(f"{RIG}/{sc}/merge_data/{lv}/*.npz"))
        step = max(1, len(files) // NFR)
        for i in range(0, len(files) - kmax - 1, step):
            with np.load(files[i]) as d:
                x, y, t = d["x"], d["y"], d["t"]
            if len(x) < MIN_EV:
                continue
            base = os.path.basename(files[i])
            lp = (f"{RIG}/{sc}/labels/labels_{lv}/labels_{base}"
                  .replace(".npz", ".npy"))
            if not os.path.exists(lp):
                continue
            lab = np.load(lp)
            if len(lab) != len(x):
                continue
            xi, yi = x.astype(np.int64), y.astype(np.int64)
            hits = np.zeros((len(x), kmax), bool)
            for k in range(1, kmax + 1):
                hits[:, k - 1] = occ_of(files[i + k])[yi, xi]
            burst = iti_gpu(torch.from_numpy(xi).to(DEV),
                            torch.from_numpy(yi).to(DEV),
                            torch.from_numpy(t.astype(np.int64)).to(DEV),
                            nw=NW, nh=NH).cpu().numpy()[:, BURST_COL]
            is_sc = lab == 1
            for k in KS:
                h = hits[:, :k]
                ps, base_r = h.all(1), ~h.any(1)
                for q in QS:
                    thr = np.quantile(burst, q)
                    pr = base_r & (burst > thr)
                    a = cell[(k, q)]
                    a["tp_s"] += int((ps & is_sc).sum()); a["n_s"] += int(ps.sum())
                    a["tp_r"] += int((pr & ~is_sc).sum()); a["n_r"] += int(pr.sum())
                    a["cov"] += int(ps.sum() + pr.sum()); a["tot"] += len(x)
            nfr += 1
        print(f"  scanned {sc}/{lv}", flush=True)

    print(f"\n=== persistence AND burstiness ({nfr} frames) ===")
    print(f"  {'K':>2s} {'q':>5s} {'prec(scene)':>12s} {'prec(rain)':>11s} "
          f"{'coverage':>9s}  gate")
    best = None
    for k in KS:
        for q in QS:
            a = cell[(k, q)]
            ps = a["tp_s"] / max(a["n_s"], 1)
            pr = a["tp_r"] / max(a["n_r"], 1)
            cov = a["cov"] / max(a["tot"], 1)
            ok = ps >= 0.85 and pr >= 0.85 and cov >= 0.10
            if ok and (best is None or cov > best[3]):
                best = (k, q, pr, cov)
            print(f"  {k:2d} {q:5.2f} {ps:12.3f} {pr:11.3f} {cov:9.3f}  "
                  f"{'PASS' if ok else 'fail'}")
    print("\n  gate: prec >= 0.85 both classes, coverage >= 0.10")
    if best is None:
        print("  RESULT: no (K,q) passes. STOP -- do not build the adapter,")
        print("  and do not reach for a third signal.")
    else:
        print(f"  RESULT: K={best[0]} q={best[1]:.2f} passes "
              f"(prec_rain {best[2]:.3f}, coverage {best[3]:.3f}).")


if __name__ == "__main__":
    main()
