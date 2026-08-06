"""Mixed-cell scene recall for trunk-only AND trunk+head, vs PRE-Mamba.

Same 98-frame sample, same cell partition (OUR granularity) as mixed_cell_diag.py,
so the numbers extend that table directly:
    ctx_c2 (trunk only)   0.4005
    exact  (BA-weighted)  0.4819
    PRE-Mamba             0.6903   <- target

Usage: python mixed_cell_diag2.py [head_checkpoint_tag]
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D
from run_kitti_perevent import Head, sample_at
from run_kitti_headv2 import HeadV2, multiscale_patch
import run_kitti_headv3 as HV3
from fast_tensor import tensor_cols_fast

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
SRC = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
PACK = "/fs/nexus-scratch/tuxunlu/kitti_t16e/test"
PM = "/fs/nexus-scratch/tuxunlu/git/PRE-Mamba/exp/event_rain/SYTHETIC/result"
DEV = "cuda"
NW, NH, R, T16 = 460, 352, 256, 16
HEAD_TAG = sys.argv[1] if len(sys.argv) > 1 else "peh_c2f4"

# trunk-only reference
ck = torch.load(f"{TMP}/ctx_f4o16_c2.pt", map_location="cpu")
trunk_only = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                       out_chans=16, n_extra=4)
trunk_only.load_state_dict(ck["state_dict"])
trunk_only = trunk_only.to(DEV).eval()

# trunk + per-event head
hck = torch.load(f"{TMP}/{HEAD_TAG}.pt", map_location="cpu")
FC1 = [hck["head"][k].shape[1] for k in hck["head"] if "fc1.weight" in k][0]
IS_V2 = FC1 > 200
IS_V3 = FC1 > 300          # v2 din=269 (32-d feats), v3 din=365 (128-d feats)
trunk_h = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                    out_chans=16, n_extra=4)
trunk_h.load_state_dict(hck["trunk"])
trunk_h = trunk_h.to(DEV).eval()
head = (HV3.HeadV2(feat_dim=128) if IS_V3 else
        (HeadV2() if IS_V2 else Head())).to(DEV)
head.load_state_dict(hck["head"])
head.eval()
feats = {}
trunk_h.out_proj.register_forward_pre_hook(
    lambda m, inp: feats.__setitem__("f", inp[0]))
blkout = {}
for _bi, _blk in enumerate(trunk_h.blocks):
    _blk.register_forward_hook(
        lambda m, i, o, bi=_bi: blkout.__setitem__(f"b{bi}", o))
print(f"loaded trunk-only + head [{HEAD_TAG}]", flush=True)


def key(x, y, t):
    return (t.astype(np.int64) * (NW * NH) + y.astype(np.int64) * NW
            + x.astype(np.int64))


def planes(path):
    with np.load(path) as d:
        on = np.unpackbits(d["on"])[: T16 * R * R].reshape(T16, R, R)
        off = np.unpackbits(d["off"])[: T16 * R * R].reshape(T16, R, R)
    return on, off


names = ["trunk only", f"trunk+head", "PRE-Mamba"]
tot = {n: [0.0, 0.0, 0.0, 0.0] for n in names}
n_frames = 0

with torch.no_grad():
    for mm in ("50mm", "150mm"):
        for f in sorted(glob.glob(f"{SRC}/merge_data/{mm}/*.npz"))[::8]:
            base = os.path.basename(f)
            pk, pp = f"{PACK}/{mm}/{base}", \
                f"{PM}/{mm}_{base.replace('.npz','')}.npy"
            if not (os.path.exists(pk) and os.path.exists(pp)):
                continue
            with np.load(f) as d:
                x, y, t, p = d["x"], d["y"], d["t"], d["p"]
            with np.load(f"{SRC}/raw_data/{base}") as d:
                clean = np.sort(key(d["x"], d["y"], d["t"]))
            rain = ~np.isin(key(x, y, t), clean)
            if rain.sum() < 50 or (~rain).sum() < 50:
                continue

            sx = np.clip((x.astype(np.int64) * R) // NW, 0, R - 1)
            sy = np.clip((y.astype(np.int64) * R) // NH, 0, R - 1)
            t0, span = t.min(), max(int(t.max() - t.min()), 1)
            tn = ((t - t0) / span).astype(np.float32)
            tb = np.clip((tn * T16).astype(np.int64), 0, T16 - 1)
            cell = (tb * R + sy) * R + sx

            on, off = planes(pk)
            on4 = torch.from_numpy(on.reshape(4, 4, R, R).max(1)
                                   ).float()[None].to(DEV)
            off4 = torch.from_numpy(off.reshape(4, 4, R, R).max(1)
                                    ).float()[None].to(DEV)
            lit_cell = torch.from_numpy((on | off).astype(bool)).to(DEV)
            idx = int(base.split(".")[0])
            ex = []
            for k in (1, 2):
                pv = f"{PACK}/{mm}/{max(idx - k, 0):010d}.npz"
                pon, poff = planes(pv if os.path.exists(pv) else pk)
                ex.append(pon.max(0)[None].astype(np.float32))
                ex.append(poff.max(0)[None].astype(np.float32))
            ex = torch.from_numpy(np.concatenate(ex, 0))[None].to(DEV)

            keeps = {}
            pr = torch.sigmoid(trunk_only(on4, x_off=off4, x_extra=ex))[0]
            tau = float(pr[lit_cell].mean())
            keeps["trunk only"] = (pr > tau).cpu().numpy()[tb, sy, sx]

            # trunk+head, per event
            logit_map = trunk_h(on4, x_off=off4, x_extra=ex)
            fmap = feats["f"]
            To = logit_map.shape[1]
            xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
            ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
            tns = torch.from_numpy(tn)[None].to(DEV)
            lv = sample_at(logit_map[:, None], xs, ys, tns)
            if IS_V3:
                fmap = torch.cat([fmap] + [blkout[f"b{i}"] for i in range(3)], 1)
            fv = sample_at(fmap[:, :, None].expand(-1, -1, To, -1, -1),
                           xs, ys, tns)
            tb4 = np.clip((tn * 4).astype(np.int64), 0, 3)
            G = np.zeros((8, NH + 2, NW + 2), np.uint8)
            np.add.at(G, ((p == 1).astype(np.int64) * 4 + tb4, y + 1, x + 1), 1)
            patch = np.log1p(np.stack([G[:, y + dy, x + dx]
                                       for dy in range(3) for dx in range(3)], 1)
                             .reshape(len(x), 72).astype(np.float32))
            pt = torch.from_numpy(patch)[None].to(DEV)
            if IS_V2:
                mp = HV3.multiscale_patch if IS_V3 else multiscale_patch
                pv2 = mp(x, y, tn, p, np.arange(len(x)))
                tc = tensor_cols_fast(x, y, t, np.arange(len(x)), 5_000_000,
                                      [4, 16, 64], NW, NH, 1_000_000)
                out = torch.sigmoid(head(
                    lv, fv, torch.from_numpy(pv2)[None].to(DEV),
                    torch.from_numpy(tc)[None].to(DEV),
                    tns[..., None]))[0, :, 0]
            else:
                out = torch.sigmoid(head(lv, fv, pt, tns[..., None]))[0, :, 0]
            tau_h = float(out.mean())
            keeps["trunk+head"] = (out > tau_h).cpu().numpy()

            pred = np.load(pp)
            keeps["PRE-Mamba"] = (pred[:len(x)] == 0) if len(pred) >= len(x) \
                else np.ones(len(x), bool)

            order = np.argsort(cell, kind="stable")
            c, r = cell[order], rain[order]
            b = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            cells, cnt = c[b], np.diff(np.r_[b, len(c)])
            rn_c = np.add.reduceat(r.astype(np.int64), b)
            bg_c = cnt - rn_c
            mixed = set(cells[(bg_c > 0) & (rn_c > 0)].tolist())
            if not mixed:
                continue
            in_mixed = np.array([cc in mixed for cc in cell])
            sel_s = in_mixed & ~rain           # scene events in mixed cells
            sel_r = in_mixed & rain            # rain  events in mixed cells
            if sel_s.sum() == 0 or sel_r.sum() == 0:
                continue
            for n in names:
                tot[n][0] += keeps[n][sel_s].sum()      # scene kept
                tot[n][1] += sel_s.sum()
                tot[n][2] += (~keeps[n][sel_r]).sum()   # rain dropped
                tot[n][3] += sel_r.sum()
            n_frames += 1

print(f"\n=== INSIDE MIXED CELLS ({n_frames} frames) ===")
print(f"  {'model':14s} {'scene kept':>11s} {'rain dropped':>13s} {'balanced':>10s}")
for n in names:
    ks, ds, kr, dr = tot[n]
    sr, nr = ks / max(ds, 1), kr / max(dr, 1)
    print(f"  {n:14s} {sr:11.4f} {nr:13.4f} {0.5*(sr+nr):10.4f}")
print("\n  scene-kept alone is one-sided (keep everything -> 1.0);")
print("  the balanced column is the fair comparison.")
