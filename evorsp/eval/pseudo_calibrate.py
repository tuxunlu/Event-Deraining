"""Are future-persistence pseudo-labels good enough to train on? Measure first.

The plan is to adapt the rig-trained model to unlabelled wild data using
pseudo-labels derived from persistence. Before writing any training code, that
premise has to be tested where ground truth exists -- on the rig.

WHY *FUTURE* PERSISTENCE. The pseudo-label is computed from the K windows that
come AFTER the one being classified. The model never sees them at inference, so
this is not a confirmation loop: we are teaching it to predict future
persistence from present appearance, not re-teaching a cue it already has.
(Past persistence would be circular -- the trunk's context planes and the
recurrence columns already encode it.)

RULE. An event is pseudo-SCENE if its pixel is lit in ALL K following windows,
pseudo-RAIN if in NONE. Everything between is left unlabelled: pseudo-labelling
works by taking the confident tails, not by forcing a decision on every event.

PRE-REGISTERED GATE, fixed before running:

    precision(scene) >= 0.85  AND  precision(rain) >= 0.85  AND
    coverage >= 0.15                     (enough events to train on)

Below that the pseudo-labels are too noisy to be worth training against, and
the honest move is to report that and stop rather than to relax the bar.

Reported per K so the accuracy/coverage trade is visible rather than assumed.
Also run on WILD, where only coverage and class balance are computable -- if
wild coverage collapses, the method cannot be applied there whatever the rig
numbers say.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RIG = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
WILD = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_realworld"
NW, NH = 1280, 720
KS = (1, 2, 3, 4)              # how many future windows must agree
NFR = 60                       # frames sampled per sequence
MIN_EV = 2000


def occupancy(path):
    with np.load(path) as d:
        occ = np.zeros((NH, NW), bool)
        occ[d["y"], d["x"]] = True
    return occ


def future_hits(files, i, x, y, kmax):
    """[len(x), kmax] -- is this pixel lit in each of the next kmax windows?"""
    xi, yi = x.astype(np.int64), y.astype(np.int64)
    out = np.zeros((len(x), kmax), bool)
    for k in range(1, kmax + 1):
        j = i + k
        if j >= len(files):
            return None
        out[:, k - 1] = occupancy(files[j])[yi, xi]
    return out


def scan(root, seqs, labelled):
    kmax = max(KS)
    agg = {k: dict(tp_s=0, n_s=0, tp_r=0, n_r=0, cov=0, tot=0) for k in KS}
    nfr = 0
    for sc, lv in seqs:
        files = sorted(glob.glob(f"{root}/{sc}/merge_data/{lv}/*.npz"))
        if len(files) < kmax + 2:
            continue
        step = max(1, len(files) // NFR)
        for i in range(0, len(files) - kmax - 1, step):
            with np.load(files[i]) as d:
                x, y = d["x"], d["y"]
            if len(x) < MIN_EV:
                continue
            lab = None
            if labelled:
                base = os.path.basename(files[i])
                lp = (f"{root}/{sc}/labels/labels_{lv}/labels_{base}"
                      .replace(".npz", ".npy"))
                if not os.path.exists(lp):
                    continue
                lab = np.load(lp)
                if len(lab) != len(x):
                    continue
            fh = future_hits(files, i, x, y, kmax)
            if fh is None:
                continue
            for k in KS:
                h = fh[:, :k]
                ps = h.all(1)                       # pseudo-scene
                pr = ~h.any(1)                      # pseudo-rain
                a = agg[k]
                a["cov"] += int(ps.sum() + pr.sum())
                a["tot"] += len(x)
                if lab is not None:
                    is_sc = lab == 1
                    a["tp_s"] += int((ps & is_sc).sum()); a["n_s"] += int(ps.sum())
                    a["tp_r"] += int((pr & ~is_sc).sum()); a["n_r"] += int(pr.sum())
            nfr += 1
    return agg, nfr


def main():
    rig_seqs = [("scene4", f"rain_{k}") for k in (2, 6, 13)]
    print("=== RIG (ground truth available) ===", flush=True)
    agg, nfr = scan(RIG, rig_seqs, True)
    print(f"  {nfr} frames\n")
    print(f"  {'K':>2s} {'prec(scene)':>12s} {'prec(rain)':>11s} "
          f"{'coverage':>9s} {'pseudo-rain share':>18s}  gate")
    best = None
    for k in KS:
        a = agg[k]
        ps = a["tp_s"] / max(a["n_s"], 1)
        pr = a["tp_r"] / max(a["n_r"], 1)
        cov = a["cov"] / max(a["tot"], 1)
        share = a["n_r"] / max(a["n_s"] + a["n_r"], 1)
        ok = ps >= 0.85 and pr >= 0.85 and cov >= 0.15
        if ok and best is None:
            best = k
        print(f"  {k:2d} {ps:12.3f} {pr:11.3f} {cov:9.3f} {share:18.3f}  "
              f"{'PASS' if ok else 'fail'}")

    wild_seqs = [("scene5", f"rain_{k}") for k in (1, 2, 5, 10)]
    print("\n=== WILD (no ground truth -- coverage only) ===", flush=True)
    wagg, wnfr = scan(WILD, wild_seqs, False)
    print(f"  {wnfr} frames\n")
    print(f"  {'K':>2s} {'coverage':>9s} {'pseudo-rain share':>18s}")
    for k in KS:
        a = wagg[k]
        cov = a["cov"] / max(a["tot"], 1)
        print(f"  {k:2d} {cov:9.3f} {'n/a':>18s}")

    print(f"\n  gate: prec(scene) >= 0.85, prec(rain) >= 0.85, coverage >= 0.15")
    if best is None:
        print("  RESULT: no K passes. Pseudo-labels are too noisy to train on;")
        print("  do not build the adapter on this signal.")
    else:
        print(f"  RESULT: K={best} passes. Proceed with K={best}.")


if __name__ == "__main__":
    main()
