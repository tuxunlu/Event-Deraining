"""Do the real EVK4 GROUND-TRUTH labels themselves contain the rain columns?

The GT panel of cmp_real_allmodels visibly retains streak-like structure. If a
substantial share of the rig's water-column events carry label 1 (= scene),
then:
  * our model keeping them is agreement with the labels, not error;
  * PRE-Mamba removing them is a label MISMATCH that merely looks cleaner;
  * every real-data number here is partly measuring agreement with noisy labels.

Test, using only quantities defined independently of any model:

  1  Locate the columns geometrically -- x-strips whose event density is a
     strong outlier (>= 4x the frame median), which is what the rig produces.
  2  Report the LABEL COMPOSITION inside those strips vs outside.
  3  Ask whether label-1 events INSIDE the strips resemble label-1 events
     outside (real scene) or label-0 events inside (rain), using the
     inter-arrival statistics already validated at AUC 0.864 and the
     cross-frame persistence measure. If inside-scene looks like inside-rain
     rather than outside-scene, the labels are unreliable exactly there.
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
STRIP = 8                     # px width of the density strips


def main():
    agg = {k: [] for k in ("frac_in", "lab1_in", "lab1_out",
                           "cv_in_s", "cv_in_r", "cv_out_s",
                           "b_in_s", "b_in_r", "b_out_s",
                           "pers_in_s", "pers_in_r", "pers_out_s")}
    nfr = 0
    for sc, lv in SEQS:
        files = sorted(glob.glob(f"{S}/{sc}/merge_data/{lv}/*.npz"))
        step = max(1, len(files) // NFR)
        for f in files[::step][:NFR]:
            base = os.path.basename(f)
            idx = int(base.split(".")[0])
            lp = (f"{S}/{sc}/labels/labels_{lv}/labels_{base}"
                  .replace(".npz", ".npy"))
            pvf = f"{S}/{sc}/merge_data/{lv}/{idx-1:010d}.npz"
            if not (os.path.exists(lp) and os.path.exists(pvf) and idx > 0):
                continue
            with np.load(f) as d:
                x, y, t = d["x"], d["y"], d["t"]
            lab = np.load(lp).astype(np.int64)
            if len(lab) != len(x) or len(x) < 5000:
                continue
            occ = np.zeros((NH, NW), bool)
            with np.load(pvf) as d:
                occ[d["y"], d["x"]] = True
            pers = occ[y, x]

            # 1. geometric column detection: outlier x-strips
            nb = NW // STRIP
            xs = np.clip(x // STRIP, 0, nb - 1)
            dens = np.bincount(xs, minlength=nb).astype(float)
            thr = 4.0 * np.median(dens[dens > 0]) if (dens > 0).any() else np.inf
            hot = dens >= thr
            if not hot.any():
                continue
            inside = hot[xs]

            feats = iti_gpu(torch.from_numpy(x.astype(np.int64)).to(DEV),
                            torch.from_numpy(y.astype(np.int64)).to(DEV),
                            torch.from_numpy(t.astype(np.int64)).to(DEV),
                            nw=NW, nh=NH).cpu().numpy()
            cv, burst = feats[:, 5], feats[:, 7]      # 16 px scale
            sc_m, rn_m = lab == 1, lab != 1

            def g(a, m):
                return float(np.median(a[m])) if m.sum() > 50 else np.nan

            agg["frac_in"].append(inside.mean())
            agg["lab1_in"].append(sc_m[inside].mean() if inside.any() else np.nan)
            agg["lab1_out"].append(sc_m[~inside].mean())
            agg["cv_in_s"].append(g(cv, inside & sc_m))
            agg["cv_in_r"].append(g(cv, inside & rn_m))
            agg["cv_out_s"].append(g(cv, ~inside & sc_m))
            agg["b_in_s"].append(g(burst, inside & sc_m))
            agg["b_in_r"].append(g(burst, inside & rn_m))
            agg["b_out_s"].append(g(burst, ~inside & sc_m))
            agg["pers_in_s"].append(g(pers.astype(float), inside & sc_m))
            agg["pers_in_r"].append(g(pers.astype(float), inside & rn_m))
            agg["pers_out_s"].append(g(pers.astype(float), ~inside & sc_m))
            nfr += 1

    m = {k: float(np.nanmean(v)) for k, v in agg.items()}
    print(f"\n=== ARE THE RAIN COLUMNS LABELLED AS SCENE? ({nfr} frames) ===")
    print(f"  events inside high-density strips: {100*m['frac_in']:.1f}% of frame")
    print(f"\n  fraction labelled SCENE (label 1)")
    print(f"    inside the columns    {m['lab1_in']:.3f}")
    print(f"    outside               {m['lab1_out']:.3f}")
    print(f"\n  do the INSIDE-scene events look like scene, or like rain?")
    print(f"  {'':22s} {'CV':>8s} {'burstiness':>12s} {'persistence':>12s}")
    print(f"    inside,  label SCENE  {m['cv_in_s']:8.2f} {m['b_in_s']:12.3f} "
          f"{m['pers_in_s']:12.3f}")
    print(f"    inside,  label RAIN   {m['cv_in_r']:8.2f} {m['b_in_r']:12.3f} "
          f"{m['pers_in_r']:12.3f}")
    print(f"    outside, label SCENE  {m['cv_out_s']:8.2f} {m['b_out_s']:12.3f} "
          f"{m['pers_out_s']:12.3f}")
    print("\n  if inside-SCENE resembles inside-RAIN rather than outside-SCENE,")
    print("  the labels are unreliable exactly where we are being judged.")


if __name__ == "__main__":
    main()
