"""Pre-registered falsifier: do ITI statistics separate the ACTUAL failure?

The failure is not "rain vs scene" in general -- the model already scores 0.84
there. It is specifically PERSISTENT rain (the rig's continuous water columns,
0.412 of their pixels active in the previous frame) being confidently called
scene (mean p 0.692 against a 0.547 threshold).

So the question this probe must answer is narrow: on the frames where the model
fails, can any inter-arrival-time column separate PERSISTENT RAIN from SCENE?

KILL RULE, fixed before looking: if no single ITI column reaches AUC >= 0.60 on
persistent-rain vs scene, the feature lacks the signal and retraining is NOT
justified. For reference the existing per-event features already reach ~0.84 on
the easy population, so a column that only helps there adds nothing here.
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from iti_feats import iti_gpu

S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
DEV = "cuda"
NW, NH = 1280, 720
SEQS = [("scene1", "rain_2"), ("scene4", "rain_13")]
NFR = 14
NAMES = ["log_mean_dt_s4", "CV_s4", "LV_s4", "burst_s4",
         "log_mean_dt_s16", "CV_s16", "LV_s16", "burst_s16"]


def auc(pos, neg):
    """P(score(pos) > score(neg)) via rank statistic."""
    if len(pos) < 50 or len(neg) < 50:
        return np.nan
    k = min(len(pos), len(neg), 20000)
    rp = np.random.default_rng(0).choice(pos, k, replace=False)
    rn = np.random.default_rng(1).choice(neg, k, replace=False)
    allv = np.concatenate([rp, rn])
    r = np.argsort(np.argsort(allv))
    return (r[:k].sum() - k * (k - 1) / 2) / (k * k)


F = {n: {"pers_rain": [], "trans_rain": [], "scene": []} for n in NAMES}
n_fr = 0
for sc, lv in SEQS:
    files = sorted(glob.glob(f"{S}/{sc}/merge_data/{lv}/*.npz"))
    step = max(1, len(files) // NFR)
    for f in files[::step][:NFR]:
        base = os.path.basename(f)
        idx = int(base.split(".")[0])
        lp = f"{S}/{sc}/labels/labels_{lv}/labels_{base}".replace(".npz", ".npy")
        if not os.path.exists(lp) or idx == 0:
            continue
        with np.load(f) as d:
            x, y, t = d["x"], d["y"], d["t"]
        lab = np.load(lp).astype(np.int64)
        if len(lab) != len(x) or len(x) < 5000:
            continue
        pvf = f"{S}/{sc}/merge_data/{lv}/{idx-1:010d}.npz"
        if not os.path.exists(pvf):
            continue
        occ = np.zeros((NH, NW), bool)
        with np.load(pvf) as d:
            occ[d["y"], d["x"]] = True
        pers = occ[y, x]
        is_scene = lab == 1
        is_rain = ~is_scene
        feats = iti_gpu(torch.from_numpy(x.astype(np.int64)).to(DEV),
                        torch.from_numpy(y.astype(np.int64)).to(DEV),
                        torch.from_numpy(t.astype(np.int64)).to(DEV),
                        nw=NW, nh=NH).cpu().numpy()
        for i, nm in enumerate(NAMES):
            F[nm]["pers_rain"].append(feats[is_rain & pers, i])
            F[nm]["trans_rain"].append(feats[is_rain & ~pers, i])
            F[nm]["scene"].append(feats[is_scene, i])
        n_fr += 1

print(f"\n=== ITI SEPARABILITY ON THE FAILING POPULATION ({n_fr} frames) ===")
print(f"  {'column':16s} {'persistent rain':>16s} {'scene':>10s} "
      f"{'AUC vs scene':>13s} {'AUC transient':>14s}")
best = 0.0
for nm in NAMES:
    pr = np.concatenate(F[nm]["pers_rain"])
    tr = np.concatenate(F[nm]["trans_rain"])
    sc_ = np.concatenate(F[nm]["scene"])
    a = auc(pr, sc_)
    a2 = auc(tr, sc_)
    a = max(a, 1 - a) if np.isfinite(a) else a          # direction-agnostic
    a2 = max(a2, 1 - a2) if np.isfinite(a2) else a2
    best = max(best, a if np.isfinite(a) else 0)
    print(f"  {nm:16s} {np.median(pr):16.3f} {np.median(sc_):10.3f} "
          f"{a:13.3f} {a2:14.3f}")
print(f"\n  best AUC (persistent rain vs scene) = {best:.3f}")
print(f"  KILL RULE: proceed only if >= 0.60 -> "
      f"{'PROCEED' if best >= 0.60 else 'KILL, do not retrain'}")
