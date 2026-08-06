"""Why does EvORSP-3T score 0.705 when its own granularity permits 0.947?

granularity_oracle.py showed the BA-optimal per-(256^2 pixel) decision reaches
event-DA 0.9466 -- ABOVE PRE-Mamba's 0.9172. So output granularity is NOT the
binding constraint. The suspect is the TRAINING TARGET.

Our GT plane is an OR-union: gt[y,x] = 1 iff >=1 CLEAN ON event lands there.
A pixel holding 1 signal event and 100 rain events is therefore labelled keep,
and a perfect model trained on it keeps all 101 events. The BA-optimal oracle
would drop that pixel.

This script evaluates, at the event level, four per-pixel decision rules on the
same frames -- each is the ceiling of a model trained to predict that target:

  A  OR of clean ON events        <- our actual training target
  B  OR of clean events, any polarity  (fixes ON-only blindness)
  C  count majority (bg > rain)
  D  BA-optimal (bg/N_bg > rain/N_rain)   <- the ceiling from the first oracle

The spread between A and D says how much of the 0.24 shortfall is the target
definition rather than model capacity or granularity.
"""
import glob
import os

import numpy as np

S = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
SRC_W, SRC_H = 460, 352
R = 256
MAX_FRAMES = 80


def key(x, y, t):
    return (t.astype(np.int64) * (SRC_W * SRC_H)
            + y.astype(np.int64) * SRC_W + x.astype(np.int64))


rules = ["A OR clean-ON (our target)", "B OR clean any-pol", "C count majority",
         "D BA-optimal (ceiling)"]
acc = {r: [0.0, 0.0, 0.0, 0] for r in rules}

for mm in ("50mm", "150mm"):
    files = sorted(glob.glob(f"{S}/merge_data/{mm}/*.npz"))
    step = max(1, len(files) // MAX_FRAMES)
    files = files[::step][:MAX_FRAMES]
    for n, f in enumerate(files):
        b = os.path.basename(f)
        with np.load(f) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        with np.load(f"{S}/raw_data/{b}") as d:
            cx, cy, ct, cp = d["x"], d["y"], d["t"], d["p"]
        rain = ~np.isin(key(x, y, t), np.sort(key(cx, cy, ct)))
        n_bg = max(int((~rain).sum()), 1)
        n_rn = max(int(rain.sum()), 1)

        sx = np.clip((x.astype(np.int64) * R) // SRC_W, 0, R - 1)
        sy = np.clip((y.astype(np.int64) * R) // SRC_H, 0, R - 1)
        pix = sy * R + sx

        order = np.argsort(pix, kind="stable")
        pc = pix[order]
        rc = rain[order]
        bounds = np.flatnonzero(np.r_[True, pc[1:] != pc[:-1]])
        cells = pc[bounds]
        cnt = np.diff(np.r_[bounds, len(pc)])
        rain_cnt = np.add.reduceat(rc.astype(np.int64), bounds)
        bg_cnt = cnt - rain_cnt

        # clean-stream occupancy masks, mapped to the same grid
        def occ(mask):
            gx = np.clip((cx[mask].astype(np.int64) * R) // SRC_W, 0, R - 1)
            gy = np.clip((cy[mask].astype(np.int64) * R) // SRC_H, 0, R - 1)
            m = np.zeros(R * R, bool)
            m[gy * R + gx] = True
            return m

        occ_on = occ(cp == 1)
        occ_any = occ(np.ones(len(cx), bool))

        decisions = {
            rules[0]: occ_on[cells],
            rules[1]: occ_any[cells],
            rules[2]: bg_cnt > rain_cnt,
            rules[3]: bg_cnt / n_bg > rain_cnt / n_rn,
        }
        for r, keep in decisions.items():
            sr = bg_cnt[keep].sum() / n_bg
            nr = rain_cnt[~keep].sum() / n_rn
            a = acc[r]
            a[0] += sr
            a[1] += nr
            a[2] += 0.5 * (sr + nr)
            a[3] += 1
        if n % 40 == 0:
            print(f"  {mm} {n}/{len(files)}", flush=True)

print("\n=== EVENT-DA OF EACH PER-PIXEL TARGET (256^2, T=1, KITTI test) ===")
print("  our trained model achieves 0.7052 here; PRE-Mamba 0.9172\n")
for r in rules:
    a = acc[r]
    print(f"  {r:30s} SR {a[0]/a[3]:.4f}  NR {a[1]/a[3]:.4f}  "
          f"DA {a[2]/a[3]:.4f}")
