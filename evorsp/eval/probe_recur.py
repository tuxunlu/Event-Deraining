"""Pre-registered separability check for the recurrence / long-persistence cols.

Same bar as before: proceed only if some column reaches AUC >= 0.60 on
PERSISTENT RAIN vs SCENE. Reference to beat: ITI burstiness 0.864; the motion
cues died at 0.599.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from recur_feats import Recur, NAMES, NW, NH

S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
SEQS = [("scene1","rain_2"), ("scene3","rain_9"), ("scene4","rain_13"), ("scene2","rain_5")]

def auc(pos, neg):
    if len(pos) < 100 or len(neg) < 100: return np.nan
    k = min(len(pos), len(neg), 20000)
    rp = np.random.default_rng(0).choice(pos, k, replace=False)
    rn = np.random.default_rng(1).choice(neg, k, replace=False)
    r = np.argsort(np.argsort(np.concatenate([rp, rn])))
    a = (r[:k].sum() - k*(k-1)/2) / (k*k)
    return max(a, 1-a)

F = {n: {"pr": [], "sc": []} for n in NAMES}
nfr = 0
for sc_, lv in SEQS:
    files = sorted(glob.glob(f"{S}/{sc_}/merge_data/{lv}/*.npz"))[:60]
    st = Recur()
    prev_occ = None
    for f in files:
        base = os.path.basename(f)
        lp = f"{S}/{sc_}/labels/labels_{lv}/labels_{base}".replace(".npz",".npy")
        if not os.path.exists(lp):
            continue
        with np.load(f) as d:
            x, y = d["x"].astype(np.int64), d["y"].astype(np.int64)
        lab = np.load(lp).astype(np.int64)
        if len(lab) != len(x) or len(x) < 5000:
            continue
        sel = np.arange(len(x))
        feats = st.features(x, y, sel)
        if prev_occ is not None and len(st.strips) >= 4:
            pers1 = prev_occ[y, x]
            is_sc = lab == 1
            for i, nm in enumerate(NAMES):
                F[nm]["pr"].append(feats[(~is_sc) & pers1, i])
                F[nm]["sc"].append(feats[is_sc, i])
            nfr += 1
        st.push(x, y)
        prev_occ = np.zeros((NH, NW), bool); prev_occ[y, x] = True

print(f"\n=== RECURRENCE / LONG-PERSISTENCE vs the failing population ({nfr} frames) ===")
print(f"  {'feature':20s} {'persistent rain':>16s} {'scene':>10s} {'AUC':>7s}")
best = ("", 0.0)
for nm in NAMES:
    pr = np.concatenate(F[nm]["pr"]); s2 = np.concatenate(F[nm]["sc"])
    a = auc(pr, s2)
    if np.isfinite(a) and a > best[1]: best = (nm, a)
    print(f"  {nm:20s} {np.median(pr):16.3f} {np.median(s2):10.3f} {a:7.3f}")
print(f"\n  best: {best[0]} AUC {best[1]:.3f}   (bar 0.60 | ITI 0.864 | motion died at 0.599)")
