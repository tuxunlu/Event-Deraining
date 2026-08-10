"""Label-free validation on in-the-wild rain (EVK4_realworld / scene5).

There is no ground truth here and there cannot be: real rain admits no clean
stream to merge against, which is exactly why the artificial rig exists. So
"validate" has to mean something other than event-DA.

CALIBRATED PROXY, pre-registered before looking at any output.

On the LABELLED rig data we measured, per event, the fraction that persists to
the next window:

    scene events   0.583
    rain events    0.083

That is a 7x separation on a quantity computed from the raw stream alone -- no
model, no labels. So on wild data a working deraining model must reproduce it:
events it KEEPS should persist like scene, events it DROPS like rain.

  PASS  persistence(kept) >= 0.40  AND  persistence(dropped) <= 0.20
  FAIL  the two collapse together -- the model is splitting the stream on
        something other than the scene/rain distinction

This can be gamed by a model that keeps almost nothing (trivially raising the
kept-persistence), so keep-rate is reported alongside and a model dropping more
than 80% of events is disqualified regardless of separation.

Reported per model so the comparison is relative, not absolute: the old
trunk-only model is the reference the current one must beat.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsp_3d import ORSPNet3D
from run_kitti_perevent import sample_at
from run_real_full import HeadV2 as HeadFull
from gpu_feats import patch_gpu, tensor_gpu
from iti_feats import iti_gpu
from recur_feats import Recur

W = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_realworld"
TMP = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
NW, NH = 1280, 720
R, T16 = 256, 16
NFR = 40                       # frames per sequence
KEEP_FLOOR = 0.20              # below this kept-fraction the result is void


def load(tag):
    p = f"{TMP}/{tag}.pt"
    if not os.path.exists(p):
        return None
    b = torch.load(p, map_location="cpu", weights_only=False)
    trunk = ORSPNet3D(T=4, num_blocks=3, use_off=True,
                      dilations=(1, 8, 32, 64), out_chans=1).to(DEV).eval()
    trunk.load_state_dict(b["trunk"] if "trunk" in b else b["state_dict"])
    head = None
    if "head" in b:
        head = HeadFull(feat_dim=128).to(DEV).eval()
        head.load_state_dict(b["head"])
    f, bl = {}, {}
    trunk.out_proj.register_forward_pre_hook(lambda m, i: f.__setitem__("f", i[0]))
    for bi, blk in enumerate(trunk.blocks):
        blk.register_forward_hook(lambda m, i, o, bi=bi: bl.__setitem__(bi, o))
    return dict(tag=tag, trunk=trunk, head=head, f=f, bl=bl,
                tau=float(b.get("tau", 0.5)), recur=Recur(nw=NW, nh=NH))


@torch.no_grad()
def keep_mask(m, x, y, t, p):
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
    on4 = torch.from_numpy(on.reshape(T16, R, R).reshape(4, 4, R, R).max(1)
                           ).float()[None].to(DEV)
    off4 = torch.from_numpy(off.reshape(T16, R, R).reshape(4, 4, R, R).max(1)
                            ).float()[None].to(DEV)
    lm = m["trunk"](on4, x_off=off4)
    tn = ((t - t0) / span).astype(np.float32)
    tns = torch.from_numpy(tn)[None].to(DEV)
    xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
    ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)

    if m["head"] is None:                      # trunk-only reference
        pr = torch.sigmoid(sample_at(lm[:, None], xs, ys, tns))[0, :, 0]
        m["recur"].push(x.astype(np.int64), y.astype(np.int64))
        return (pr > m["tau"]).cpu().numpy()

    fm = torch.cat([m["f"]["f"]] + [m["bl"][i] for i in range(3)], 1)
    xg = torch.from_numpy(x.astype(np.int64)).to(DEV)
    yg = torch.from_numpy(y.astype(np.int64)).to(DEV)
    tg = torch.from_numpy(t.astype(np.int64)).to(DEV)
    pg = torch.from_numpy(p.astype(np.int64)).to(DEV)
    lv = sample_at(lm[:, None], xs, ys, tns)
    fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                   xs, ys, tns)
    pv = patch_gpu(xg, yg, tns[0], pg, NW, NH)[None]
    tc = tensor_gpu(xg, yg, tg, 5_000, [4, 16, 64], NW, NH, 1_000)
    it = iti_gpu(xg, yg, tg, nw=NW, nh=NH)
    xi, yi = x.astype(np.int64), y.astype(np.int64)
    rc = torch.from_numpy(m["recur"].features(xi, yi, np.arange(len(x)))).to(DEV)
    m["recur"].push(xi, yi)
    tc = torch.cat([tc, it, rc], 1)[None]
    pr = torch.sigmoid(m["head"](lv, fv, pv, tc, tns[..., None]))[0, :, 0]
    return (pr > m["tau"]).cpu().numpy()


def main():
    tags = [t for t in sys.argv[1:]] or ["real_evorsp", "realfull_ours"]
    ms = [q for q in (load(t) for t in tags) if q]
    if not ms:
        raise SystemExit("no checkpoints loaded")
    acc = {m["tag"]: dict(pk=0.0, nk=0.0, pd=0.0, nd=0.0) for m in ms}

    lvls = sorted(glob.glob(f"{W}/scene5/merge_data/rain_*"))[:6]
    for lv in lvls:
        files = sorted(glob.glob(f"{lv}/*.npz"))
        step = max(1, len(files) // NFR)
        for m in ms:
            m["recur"].reset()
        prev = None
        for i, fp in enumerate(files):
            with np.load(fp) as d:
                x, y, t, p = d["x"], d["y"], d["t"], d["p"]
            if i % step or len(x) < 500:
                for m in ms:                     # keep the causal buffer dense
                    m["recur"].push(x.astype(np.int64), y.astype(np.int64))
                prev = (x, y)
                continue
            # persistence: does this event's pixel light up in the NEXT window?
            nxt = files[i + 1] if i + 1 < len(files) else None
            if nxt is None:
                break
            with np.load(nxt) as d2:
                occ = np.zeros((NH, NW), bool)
                occ[d2["y"], d2["x"]] = True
            pers = occ[y.astype(np.int64), x.astype(np.int64)]
            for m in ms:
                k = keep_mask(m, x, y, t, p)
                a = acc[m["tag"]]
                a["pk"] += float(pers[k].sum()); a["nk"] += float(k.sum())
                a["pd"] += float(pers[~k].sum()); a["nd"] += float((~k).sum())
            prev = (x, y)
        print(f"  done {os.path.basename(lv)}", flush=True)

    print(f"\n{'model':18s} {'keep':>7s} {'persist(kept)':>14s} "
          f"{'persist(dropped)':>17s} {'gap':>7s}  verdict")
    print("  " + "-" * 72)
    for m in ms:
        a = acc[m["tag"]]
        kf = a["nk"] / max(a["nk"] + a["nd"], 1)
        pk = a["pk"] / max(a["nk"], 1)
        pd = a["pd"] / max(a["nd"], 1)
        if kf < KEEP_FLOOR:
            v = "VOID (keeps too little)"
        elif pk >= 0.40 and pd <= 0.20:
            v = "PASS"
        else:
            v = "FAIL"
        print(f"{m['tag']:18s} {kf:7.3f} {pk:14.3f} {pd:17.3f} "
              f"{pk - pd:7.3f}  {v}")
    print("\n  reference from LABELLED rig data: scene persists 0.583,")
    print("  rain 0.083. Kept should look like the former, dropped the latter.")


if __name__ == "__main__":
    main()
