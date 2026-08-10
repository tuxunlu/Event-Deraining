"""The metric the aggregate number hides: rain kept ON PERSISTENT PIXELS.

Your screenshots show vertical rain columns surviving on the background while
aggregate event-DA looks healthy. Those two facts are consistent, because the
residue lives in a small population that DA averages away: rain events landing
where the previous window was also lit. The rig's nozzles write continuous
water COLUMNS at fixed pixels, so they satisfy the "persistent => scene" rule
the trunk leans on, and DA barely moves when they survive.

Reports, on the real EVK4 test split, per model:

  rain kept | persistent   the failure being asked about. LOWER IS BETTER.
  scene kept | persistent  the guard: dropping the columns must not also drop
                           real persistent structure (poles, window frames).
  streak margin            scene_kept - rain_kept on that population; the
                           single number that says whether the model can tell
                           a nozzle from a building edge at all.
  event-DA                 aggregate, for continuity with the leaderboard.

Variants differ only in how many per-event columns the head sees:
    19  structure tensor only        (head-only)
    27  + inter-arrival regularity   (+ITI)
    39  + spatial recurrence         (+ITI +recur)
so a difference across rows is attributable to the features, not the trunk.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader

import run_real_full as RF
from rsp_3d import ORSPNet3D
from run_real_full import CacheSet, sample_at

DEV = "cuda"
TMP = os.path.dirname(os.path.abspath(__file__))
PATCH_D, FEAT_D = RF.PATCH_D, 128

# (checkpoint, split, n per-event columns the head was trained with)
MODELS = [
    ("realph_theirs", "theirs", 19),
    ("realiti_theirs", "theirs", 27),
    ("realfull_theirs", "theirs", 39),
    
    
    ("realfull_ours", "ours", 39),
]


class Head(nn.Module):
    """Same shape as each trainer's HeadV2, parameterised by column count."""

    def __init__(self, ncols):
        super().__init__()
        din = 1 + FEAT_D + PATCH_D + ncols + 1
        self.fc1 = nn.Linear(din, 64)
        self.fc2 = nn.Linear(64 + 1, 32)
        self.fc3 = nn.Linear(32 + 1, 1)

    def forward(self, l, feat, patch, tcols, tn):
        z = torch.cat([l, feat, patch, tcols, tn], -1)
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(torch.cat([h, l], -1)))
        return l + self.fc3(torch.cat([h, l], -1))


def persistence(ds, i, x, y):
    """True where this event's pixel was also lit in the PREVIOUS window.

    Independent of any model and of the labels -- it is a property of the raw
    stream, which is what makes it usable as a diagnostic.
    """
    cf = ds.files[i]
    sc, mm = cf.split("/")[-3], cf.split("/")[-2]
    idx = int(os.path.basename(cf).split(".")[0])
    if idx == 0:
        return np.zeros(len(x), bool)
    prev = f"{RF.SRC}/{sc}/merge_data/{mm}/{idx-1:010d}.npz"
    if not os.path.exists(prev):
        return np.zeros(len(x), bool)
    occ = np.zeros((720, 1280), bool)
    with RF._load_retry(prev) as d:
        occ[d["y"], d["x"]] = True
    return occ[y.astype(np.int64), x.astype(np.int64)]


def build(tag, ncols):
    p = f"{TMP}/{tag}.pt"
    if not os.path.exists(p):
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    trunk = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                      use_off=True, out_chans=1).to(DEV).eval()
    trunk.load_state_dict(blob["trunk"])
    head = Head(ncols).to(DEV).eval()
    head.load_state_dict(blob["head"])
    feats = {}
    trunk.out_proj.register_forward_pre_hook(
        lambda m, inp: feats.__setitem__("f", inp[0]))
    for bi, blk in enumerate(trunk.blocks):
        blk.register_forward_hook(
            lambda m, i, o, bi=bi: feats.__setitem__(f"b{bi}", o))
    return dict(trunk=trunk, head=head, feats=feats, tau=float(blob["tau"]),
                ncols=ncols, tag=tag,
                acc=dict(rk=0.0, rn=0.0, sk=0.0, sn=0.0, da=[]))


@torch.no_grad()
def evaluate_split(split, entries):
    """One pass over the test set; every model scored on the SAME batch.

    The previous frame is read from NFS once per frame rather than once per
    (frame, model) -- the naive version did that 3x over and was the dominant
    cost, not the forward passes.
    """
    ms = [m for m in entries if m]
    if not ms:
        return
    ds = CacheSet("test", split)
    ld = DataLoader(ds, batch_size=1, num_workers=4)
    for k, (on, off, ex, xs, ys, tn, patch, tcols, lab, inv_p,
            n_bg, n_rn, idx) in enumerate(ld):
        y0 = lab[0].numpy()
        pers = persistence(ds, int(idx[0]), xs[0].numpy(), ys[0].numpy())
        is_rain, is_scene = (y0 < 0.5) & pers, (y0 > 0.5) & pers
        w = inv_p[0].numpy()
        sc_m, rn_m = y0 > 0.5, y0 < 0.5
        xg, yg, tg = xs.to(DEV), ys.to(DEV), tn.to(DEV)
        pg, tcg = patch.to(DEV), tcols.to(DEV)
        for m in ms:
            lm = m["trunk"](on.to(DEV), x_off=off.to(DEV))
            f = m["feats"]
            fm = torch.cat([f["f"]] + [f[f"b{i}"] for i in
                                       range(len(m["trunk"].blocks))], 1)
            To = lm.shape[1]
            lv = sample_at(lm[:, None], xg, yg, tg)
            fv = sample_at(fm[:, :, None].expand(-1, -1, To, -1, -1),
                           xg, yg, tg)
            logit = m["head"](lv, fv, pg, tcg[..., :m["ncols"]],
                              tg[..., None])[..., 0]
            keep = (torch.sigmoid(logit) > m["tau"])[0].cpu().numpy()
            a = m["acc"]
            a["rk"] += float(keep[is_rain].sum()); a["rn"] += float(is_rain.sum())
            a["sk"] += float(keep[is_scene].sum()); a["sn"] += float(is_scene.sum())
            if w[sc_m].sum() > 0 and w[rn_m].sum() > 0:
                a["da"].append(0.5 * (
                    (keep[sc_m] * w[sc_m]).sum() / w[sc_m].sum()
                    + ((~keep[rn_m]) * w[rn_m]).sum() / w[rn_m].sum()))
        if k % 25 == 0:
            print(f"  {split} {k}/{len(ds)}", flush=True)


if __name__ == "__main__":
    rows = []
    for split in ("ours",):
        ent = [build(t, n) for t, sp, n in MODELS if sp == split]
        evaluate_split(split, ent)
        rows += [m for m in ent if m]
    print(f"\n{'model':20s} {'cols':>5s} {'rain kept':>10s} {'scene kept':>11s} "
          f"{'margin':>8s} {'event-DA':>9s}")
    print("  " + "-" * 68)
    for m in rows:
        a = m["acc"]
        rk, sk = a["rk"] / max(a["rn"], 1), a["sk"] / max(a["sn"], 1)
        da = float(np.mean(a["da"])) if a["da"] else float("nan")
        print(f"{m['tag']:20s} {m['ncols']:5d} {rk:10.3f} {sk:11.3f} "
              f"{sk-rk:8.3f} {da:9.4f}")
    print("\n  rain kept: LOWER is better -- it is the residue in the images.")
    print("  margin = scene_kept - rain_kept on persistent pixels; this is")
    print("  the population where a nozzle and a building edge look alike.")
