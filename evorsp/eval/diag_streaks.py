"""Why do dense vertical rain columns survive our model on real EVK4?

Observed: thick BLUE (OFF) vertical bands remain in our output on
scene1/rain_2 and scene4/rain_13, exactly where the rig's water columns are;
PRE-Mamba clears them. Six competing explanations, measured on the same frames:

  H1 POLARITY      the survivors are OFF -- is our rain-recall polarity-skewed?
  H2 PERSISTENCE   the rig's columns are CONTINUOUS STREAMS at fixed pixels, so
                   they are temporally PERSISTENT. Our whole discriminative cue
                   is "rain is brief, scene persists" -- a stationary stream
                   violates the assumption the model is built on.
  H3 DENSITY       are survivors the densest rain, i.e. saturated cells?
  H4 CONFIDENCE    did the model score them near tau (ambiguous) or confidently
                   wrong (misclassified)?
  H5 THRESHOLD     is the self-prior tau simply too low on these frames? Compare
                   with the per-frame oracle-optimal tau.
  H6 STAGE         does the TRUNK keep them and the head fail to fix, or does the
                   head itself introduce them?
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D
from run_kitti_perevent import sample_at
from run_real_perevent import HeadV2 as HeadR
from gpu_feats import patch_gpu, tensor_gpu

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
DEV = "cuda"
NW, NH, R, T16 = 1280, 720, 256, 16
SEQS = [("scene1", "rain_2"), ("scene4", "rain_13")]
NFR = 12

hck = torch.load(f"{TMP}/realph_theirs.pt", map_location="cpu")
trunk = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=1)
trunk.load_state_dict(hck["trunk"])
trunk = trunk.to(DEV).eval()
head = HeadR(feat_dim=128).to(DEV)
head.load_state_dict(hck["head"])
head.eval()
_f, _b = {}, {}
trunk.out_proj.register_forward_pre_hook(lambda m, i: _f.__setitem__("f", i[0]))
for bi, blk in enumerate(trunk.blocks):
    blk.register_forward_hook(lambda m, i, o, bi=bi: _b.__setitem__(bi, o))

acc = {k: [] for k in ("pol_rain", "pol_kept_rain", "pol_scene",
                       "pers_kept_rain", "pers_drop_rain", "pers_scene",
                       "dens_kept_rain", "dens_drop_rain",
                       "p_kept_rain", "p_drop_rain", "p_scene",
                       "tau_sp", "tau_opt", "da_sp", "da_opt",
                       "trunk_keeps_rain", "head_keeps_rain")}

with torch.no_grad():
    for sc, lv in SEQS:
        files = sorted(glob.glob(f"{S}/{sc}/merge_data/{lv}/*.npz"))
        step = max(1, len(files) // NFR)
        for f in files[::step][:NFR]:
            base = os.path.basename(f)
            lp = f"{S}/{sc}/labels/labels_{lv}/labels_{base}".replace(".npz", ".npy")
            if not os.path.exists(lp):
                continue
            with np.load(f) as d:
                x, y, t, p = d["x"], d["y"], d["t"], d["p"]
            lab = np.load(lp).astype(np.int64)
            if len(lab) != len(x) or len(x) < 5000:
                continue
            is_scene = lab == 1
            is_rain = ~is_scene
            if is_rain.sum() < 100 or is_scene.sum() < 100:
                continue

            # previous frame occupancy -> persistence of each event's pixel
            idx = int(base.split(".")[0])
            pv = f"{S}/{sc}/merge_data/{lv}/{max(idx-1,0):010d}.npz"
            occ = np.zeros((NH, NW), bool)
            if os.path.exists(pv) and idx > 0:
                with np.load(pv) as d:
                    occ[d["y"], d["x"]] = True
            pers = occ[y, x]

            # local density at native res (3x3 count)
            G = np.zeros((NH + 2, NW + 2), np.int32)
            np.add.at(G, (y + 1, x + 1), 1)
            dens = sum(G[y + 1 + dy, x + 1 + dx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1))

            sx = (x.astype(np.int64) * R) // NW
            sy = (y.astype(np.int64) * R) // NH
            t0 = t.min()
            span = max(int(t.max() - t0), 1)
            tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
            on = np.zeros((T16, R * R), bool)
            off = np.zeros((T16, R * R), bool)
            s = p == 1
            on[tb[s], sy[s] * R + sx[s]] = True
            off[tb[~s], sy[~s] * R + sx[~s]] = True
            on4 = torch.from_numpy(on.reshape(T16, R, R).reshape(4, 4, R, R)
                                   .max(1)).float()[None].to(DEV)
            off4 = torch.from_numpy(off.reshape(T16, R, R).reshape(4, 4, R, R)
                                    .max(1)).float()[None].to(DEV)

            lm = trunk(on4, x_off=off4)
            fm = torch.cat([_f["f"]] + [_b[i] for i in range(3)], 1)
            tn = ((t - t0) / span).astype(np.float32)
            xg = torch.from_numpy(x.astype(np.int64)).to(DEV)
            yg = torch.from_numpy(y.astype(np.int64)).to(DEV)
            tg = torch.from_numpy(t.astype(np.int64)).to(DEV)
            pg = torch.from_numpy(p.astype(np.int64)).to(DEV)
            tns = torch.from_numpy(tn)[None].to(DEV)
            xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
            ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
            lv_ = sample_at(lm[:, None], xs, ys, tns)
            fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                           xs, ys, tns)
            pv2 = patch_gpu(xg, yg, tns[0], pg, NW, NH)[None]
            tc = tensor_gpu(xg, yg, tg, 5_000, [4, 16, 64], NW, NH, 1_000)[None]
            ev = torch.sigmoid(head(lv_, fv, pv2, tc, tns[..., None]))[0, :, 0]
            pe = ev.cpu().numpy()
            tau_sp = float(pe.mean())
            keep = pe > tau_sp

            # trunk-alone decision, same self-prior rule
            pt = torch.sigmoid(lm)[0, 0].cpu().numpy()[sy, sx]
            keep_tr = pt > float(pt.mean())

            n_s, n_r = int(is_scene.sum()), int(is_rain.sum())
            kr = keep & is_rain                     # rain we WRONGLY kept
            dr = (~keep) & is_rain
            acc["pol_rain"].append((p[is_rain] != 1).mean())
            acc["pol_kept_rain"].append((p[kr] != 1).mean() if kr.any() else np.nan)
            acc["pol_scene"].append((p[is_scene] != 1).mean())
            acc["pers_kept_rain"].append(pers[kr].mean() if kr.any() else np.nan)
            acc["pers_drop_rain"].append(pers[dr].mean() if dr.any() else np.nan)
            acc["pers_scene"].append(pers[is_scene].mean())
            acc["dens_kept_rain"].append(np.median(dens[kr]) if kr.any() else np.nan)
            acc["dens_drop_rain"].append(np.median(dens[dr]) if dr.any() else np.nan)
            acc["p_kept_rain"].append(pe[kr].mean() if kr.any() else np.nan)
            acc["p_drop_rain"].append(pe[dr].mean() if dr.any() else np.nan)
            acc["p_scene"].append(pe[is_scene].mean())
            # oracle-optimal tau for this frame
            grid = np.quantile(pe, np.linspace(0.02, 0.98, 49))
            das = [0.5 * ((pe > g)[is_scene].mean() + (pe <= g)[is_rain].mean())
                   for g in grid]
            j = int(np.argmax(das))
            acc["tau_opt"].append(float(grid[j]))
            acc["tau_sp"].append(tau_sp)
            acc["da_opt"].append(das[j])
            acc["da_sp"].append(0.5 * (keep[is_scene].mean()
                                       + (~keep)[is_rain].mean()))
            acc["trunk_keeps_rain"].append(keep_tr[is_rain].mean())
            acc["head_keeps_rain"].append(keep[is_rain].mean())

m = {k: float(np.nanmean(v)) for k, v in acc.items()}
print(f"\n=== WHY RAIN STREAKS SURVIVE ({len(acc['tau_sp'])} frames, "
      f"scene1/rain_2 + scene4/rain_13) ===\n")
print("H1 POLARITY   fraction OFF")
print(f"   all rain events          {m['pol_rain']:.3f}")
print(f"   rain we WRONGLY KEPT     {m['pol_kept_rain']:.3f}   <- skewed OFF?")
print(f"   scene events             {m['pol_scene']:.3f}")
print("\nH2 PERSISTENCE   fraction whose pixel was active in the PREVIOUS frame")
print(f"   scene events             {m['pers_scene']:.3f}")
print(f"   rain we WRONGLY KEPT     {m['pers_kept_rain']:.3f}   <- looks like scene?")
print(f"   rain correctly dropped   {m['pers_drop_rain']:.3f}")
print("\nH3 DENSITY   median 3x3 native neighbour count")
print(f"   rain wrongly kept        {m['dens_kept_rain']:.1f}")
print(f"   rain correctly dropped   {m['dens_drop_rain']:.1f}")
print("\nH4 CONFIDENCE   mean predicted p (tau is the self-prior mean)")
print(f"   scene                    {m['p_scene']:.3f}")
print(f"   rain wrongly kept        {m['p_kept_rain']:.3f}")
print(f"   rain correctly dropped   {m['p_drop_rain']:.3f}")
print(f"   self-prior tau           {m['tau_sp']:.3f}")
print("\nH5 THRESHOLD")
print(f"   self-prior tau {m['tau_sp']:.3f} -> DA {m['da_sp']:.4f}")
print(f"   oracle tau     {m['tau_opt']:.3f} -> DA {m['da_opt']:.4f}"
      f"   (gap {m['da_opt']-m['da_sp']:+.4f})")
print("\nH6 STAGE   fraction of RAIN kept")
print(f"   trunk alone              {m['trunk_keeps_rain']:.3f}")
print(f"   trunk + head             {m['head_keeps_rain']:.3f}")
