"""DIRECTION 1: frame-adaptive atom orientation.

Identical to run_real_full.py except the trunk's atom bank re-steers itself to
each frame's dominant orientation (see rsp_steer3d.py). Two extra scalars per
block, six for the trunk, both zero-init so this starts bit-identical to the
frozen-bank model.

Control: run_real_full.py --split ours -> rig test event-DA 0.8831.
"""
_ORIG = """Head v2: oriented + multi-scale per-event features, mixed-cell sampling quota.

Head v1 (plain 3x3x8 count patch, uniform sampling) measured:
    mixed-cell recall  0.4005 (trunk) -> 0.5562 @1 epoch -> 0.5253 @40 epochs
    aggregate DA       0.9332          -> 0.9346         -> 0.9396
i.e. longer training BUYS aggregate DA by SELLING the occlusion case, because
mixed cells carry only 4.4% of the objective. Two fixes, both in this file:

  FEATURES. The v1 head sees a 3x3 native count patch -- a neighbourhood
  SMALLER than a rain drop, with no orientation. v2 adds
    * multi-scale patches: 3x3 at dilations 1, 2, 4 (covering 3x3, 7x7, 15x15
      native px), 2 polarities x 4 time bins each = 216 dims -- so the head can
      see PAST a drop instead of only inside it;
    * 19 structure-tensor columns (coherence, cos2t/sin2t, log-spread-per-mass,
      minor-axis residual at 3 scales + cross-scale gates) from the cache. A car
      edge is coherent and oriented; a drop is an isotropic blob.

  SAMPLING. Train on the cached stratified sample (up to half mixed-cell) with
  1/p importance weights, so the failure case gets ~50% of the gradient while
  the loss remains an UNBIASED estimate of the full-frame balanced accuracy:
      w_e = inv_p_e * (lab_e / N_bg + (1 - lab_e) / N_rn)
  with N_bg, N_rn the FULL-frame class totals (cached, not sample totals).

Val is evaluated on the same weighted sample (unbiased); the final test number
recomputes structure-tensor columns on the fly for EVERY event, so it is exact.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D
from rsp_steer3d import ORSPNet3DSteer
from run_kitti_perevent import sample_at
from fast_tensor import tensor_cols_fast


def _load_retry(path, tries=5, wait=3.0):
    """np.load with retries: /fs/nexus-projects is NFS and intermittently
    fails a read that succeeds moments later. A blip 11 min into a 4 h run
    killed realfull_ours once already."""
    import time
    for k in range(tries):
        try:
            return np.load(path)
        except (FileNotFoundError, OSError, ValueError):
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


PACK = "/fs/nexus-scratch/tuxunlu/real_t16e"
CACHE = "/fs/nexus-scratch/tuxunlu/real_headv2"
ITI = "/fs/nexus-scratch/tuxunlu/real_iti"     # 8 ITI-regularity cols
RECUR = "/fs/nexus-scratch/tuxunlu/real_recur" # 12 recurrence cols
SRC = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
T_BUILD, R, NW, NH = 16, 256, 1280, 720
DILS = (1, 3, 9)   # L3a: 3x3 at these dilations -> 27x27 native view
NBIN = 4                                   # temporal bins per polarity in patch
PATCH_D = len(DILS) * 9 * 2 * NBIN         # 216
# EVK4 stamps are MICROseconds, unlike KITTI (nanoseconds)
SCALES, SLICE_US, TAU_US = [4, 16, 64], 1_000, 5_000
TAUS = np.linspace(0.05, 0.95, 19)
EPS = 1e-6


def multiscale_patch(x, y, tn, p, sel):
    """[len(sel), 216] log1p counts: 3x3 at dilations 1/2/4, 2 pol x 4 bins.

    G is accumulated from ALL events (the neighbourhood must be complete) but
    gathered only at `sel` -- gathering all 157K events wastes 6.5x the work
    and 270 MB per frame.
    """
    tb = np.clip((tn * NBIN).astype(np.int64), 0, NBIN - 1)
    pad = max(DILS)          # must cover the widest dilation
    Hp, Wp, C = NH + 2 * pad, NW + 2 * pad, 2 * NBIN
    # bincount, not np.add.at: the latter is an unbuffered scatter and costs
    # 0.51 s/frame here (16 min/epoch of pure data loading).
    flat = (((p == 1).astype(np.int64) * NBIN + tb) * Hp + (y + pad)) * Wp \
        + (x + pad)
    G = np.bincount(flat, minlength=C * Hp * Wp).reshape(C, Hp, Wp)
    G = np.minimum(G, 255).astype(np.uint8)
    xs, ys = x[sel] + pad, y[sel] + pad
    cols = []
    for d in DILS:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cols.append(G[:, ys + dy * d, xs + dx * d])
    return np.log1p(np.concatenate(cols, 0).T.astype(np.float32))


THEIRS = {"scene1": {"train": [1], "val": [4], "test": [2, 3]},
          "scene2": {"train": [3, 9], "val": [4, 6], "test": [1, 2, 5, 10]},
          "scene3": {"train": [4, 10], "val": [6, 8], "test": [2, 9]},
          "scene4": {"train": [1, 3], "val": [4, 9], "test": [2, 6, 13]}}
OURS = {"train": ("scene1", "scene2"), "val": ("scene3",), "test": ("scene4",)}


class CacheSet(Dataset):
    def __init__(self, split, kind="theirs"):
        self.files = []
        if kind == "theirs":
            for sc, d in THEIRS.items():
                for k in d[split]:
                    self.files += sorted(glob.glob(f"{CACHE}/{sc}/rain_{k}/*.npz"))
        elif kind == "all":
            # Every labelled scene. No held-out test split: the evaluation
            # target is the WILD data, which has no ground truth at all, so a
            # scene4 test set would only re-measure what realfull_ours already
            # reported. Val is a 1-in-8 stride for early stopping only.
            for sc in ("scene1", "scene2", "scene3", "scene4"):
                self.files += sorted(glob.glob(f"{CACHE}/{sc}/rain_*/*.npz"))
            self.files = [f for i, f in enumerate(sorted(self.files))
                          if (i % 8 == 0) == (split != "train")]
        else:
            for sc in OURS[split]:
                self.files += sorted(glob.glob(f"{CACHE}/{sc}/rain_*/*.npz"))
        self.files = [f for f in sorted(self.files)
                      if os.path.exists(f.replace(CACHE, ITI))
                      and os.path.exists(f.replace(CACHE, RECUR))]
        self.mm = [f.split("/")[-2] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        cf = self.files[i]
        mm, base = self.mm[i], os.path.basename(cf)
        with _load_retry(cf) as d:
            sel, tcols = d["sel"].astype(np.int64), d["tcols"].astype(np.float32)
            lab, inv_p = d["lab"].astype(np.float32), d["inv_p"]
            n_bg, n_rn = float(d["n_bg"]), float(d["n_rn"])
        # inter-arrival-time regularity: separates the rig's PERSISTENT water
        # columns from scene (AUC 0.864) where persistence alone cannot
        with _load_retry(cf.replace(CACHE, ITI)) as d2:
            tcols = np.concatenate([tcols, d2["iti"].astype(np.float32)], 1)
        # spatial recurrence + long-baseline persistence: the nozzle columns
        # recur at 16 px over 8 windows (AUC 0.881) and span the full frame
        # height (0.815), which single-window persistence cannot see
        with _load_retry(cf.replace(CACHE, RECUR)) as d3:
            tcols = np.concatenate([tcols, d3["recur"].astype(np.float32)], 1)
        pk = cf.replace(CACHE, PACK)
        with _load_retry(pk) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
        on4 = on.reshape(4, 4, R, R).max(1).astype(np.float32)
        off4 = off.reshape(4, 4, R, R).max(1).astype(np.float32)
        ex = np.zeros((0, R, R), np.float32)   # real trunk has no ctx planes

        sc = cf.split("/")[-3]
        with _load_retry(f"{SRC}/{sc}/merge_data/{mm}/{base}") as d:
            xa, ya, t, pa = d["x"], d["y"], d["t"], d["p"]
        t0, span = t.min(), max(int(t.max() - t.min()), 1)
        tn_all = ((t - t0) / span).astype(np.float32)
        patch = multiscale_patch(xa, ya, tn_all, pa, sel)
        x, y = xa[sel], ya[sel]
        return (torch.from_numpy(on4), torch.from_numpy(off4),
                torch.from_numpy(ex),
                torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)),
                torch.from_numpy(tn_all[sel]),
                torch.from_numpy(patch), torch.from_numpy(tcols),
                torch.from_numpy(lab), torch.from_numpy(inv_p),
                torch.tensor(n_bg), torch.tensor(n_rn), i)


class HeadV2(nn.Module):
    def __init__(self, feat_dim=32, hidden=64):
        super().__init__()
        din = 1 + feat_dim + PATCH_D + 19 + 8 + 12 + 1  # +ITI +recur
        self.fc1 = nn.Linear(din, hidden)
        self.fc2 = nn.Linear(hidden + 1, 32)
        self.fc3 = nn.Linear(32 + 1, 1)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, l, feat, patch, tcols, tn):
        z = torch.cat([l, feat, patch, tcols, tn], -1)
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(torch.cat([h, l], -1)))
        return l + self.fc3(torch.cat([h, l], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="reale_theirs_o1")
    ap.add_argument("--split", default="theirs",
                    choices=["theirs", "ours", "all"])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = f"realsteer_{a.split}" + (f"_s{a.seed}" if a.seed else "")

    dl = dict(num_workers=3, pin_memory=True, persistent_workers=True)
    ds = {s: CacheSet(s, a.split) for s in ("train", "val", "test")}
    tr = DataLoader(ds["train"], batch_size=a.batch, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(ds["val"], batch_size=a.batch, **dl)
    te = DataLoader(ds["test"], batch_size=a.batch, **dl)

    trunk = ORSPNet3DSteer(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                           use_off=True, out_chans=1).to(DEV)
    nnew = trunk.load_frozen(torch.load(f"{TMP}/{a.init}.pt",
                                        map_location="cpu")["state_dict"])
    print(f"  loaded frozen-bank trunk; {nnew} new steering scalars", flush=True)
    # L3a: sample trunk features from EVERY block (dilations 1/8/32/64 give
    # genuinely different spatial context) instead of only the final map.
    head = HeadV2(feat_dim=32 * (len(trunk.blocks) + 1)).to(DEV)
    feats = {}
    trunk.out_proj.register_forward_pre_hook(
        lambda m, inp: feats.__setitem__("f", inp[0]))
    for bi, blk in enumerate(trunk.blocks):
        blk.register_forward_hook(
            lambda m, i, o, bi=bi: feats.__setitem__(f"b{bi}", o))
    npar = sum(p.numel() for p in trunk.parameters()) + \
        sum(p.numel() for p in head.parameters())
    print(f"{tag}: trunk+head {npar:,} params (head "
          f"{sum(p.numel() for p in head.parameters()):,}), patch dim {PATCH_D}",
          flush=True)

    opt = torch.optim.AdamW([{"params": trunk.parameters(), "lr": 2e-4},
                             {"params": head.parameters(), "lr": 1e-3}],
                            weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)

    def fwd(on, off, ex, xs, ys, tn, patch, tcols):
        lm = trunk(on.to(DEV), x_off=off.to(DEV))
        nblk = len(trunk.blocks)
        fm = torch.cat([feats["f"]] + [feats[f"b{i}"] for i in range(nblk)], 1)
        To = lm.shape[1]
        xs, ys, tn = xs.to(DEV), ys.to(DEV), tn.to(DEV)
        lv = sample_at(lm[:, None], xs, ys, tn)
        fv = sample_at(fm[:, :, None].expand(-1, -1, To, -1, -1), xs, ys, tn)
        return head(lv, fv, patch.to(DEV), tcols.to(DEV), tn[..., None])[..., 0]

    @torch.no_grad()
    def evaluate(loader, dset):
        acc = {}
        for on, off, ex, xs, ys, tn, patch, tcols, lab, inv_p, nb, nr, idx in loader:
            out = torch.sigmoid(fwd(on, off, ex, xs, ys, tn, patch, tcols))
            lab, inv_p = lab.to(DEV), inv_p.to(DEV)
            for b in range(on.shape[0]):
                mm = dset.mm[int(idx[b])]
                d = acc.setdefault(mm, [np.zeros(len(TAUS)),
                                        np.zeros(len(TAUS)), 0])
                wb = inv_p[b] * lab[b]
                wr = inv_p[b] * (1 - lab[b])
                sb, sr = float(wb.sum()), float(wr.sum())
                if sb < 1 or sr < 1:
                    continue
                for j, t in enumerate(TAUS):
                    k = (out[b] > t).float()
                    d[0][j] += float((k * wb).sum()) / sb
                    d[1][j] += float(((1 - k) * wr).sum()) / sr
                d[2] += 1
        return {mm: 0.5 * (s + n) / max(c, 1) for mm, (s, n, c) in acc.items()}

    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        trunk.train()
        head.train()
        tot = nb_ = 0
        for on, off, ex, xs, ys, tn, patch, tcols, lab, inv_p, nb, nr, _ in tr:
            out = fwd(on, off, ex, xs, ys, tn, patch, tcols)
            lab, inv_p = lab.to(DEV), inv_p.to(DEV)
            nb, nr = nb.to(DEV)[:, None], nr.to(DEV)[:, None]
            w = inv_p * (lab / nb.clamp(min=1) + (1 - lab) / nr.clamp(min=1))
            bce = F.binary_cross_entropy_with_logits(out, lab, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=EPS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(trunk.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            tot += loss.item()
            nb_ += 1
        sch.step()
        trunk.eval()
        head.eval()
        per = evaluate(va, ds["val"])
        da = np.mean([v for v in per.values()], axis=0)
        j = int(np.argmax(da))
        star = ""
        if da[j] > best:
            best, best_tau = float(da[j]), float(TAUS[j])
            best_sd = ({k: v.detach().cpu().clone()
                        for k, v in trunk.state_dict().items()},
                       {k: v.detach().cpu().clone()
                        for k, v in head.state_dict().items()})
            star = "  *"
        if ep % 5 == 0 or ep == 1 or star:
            print(f"  ep {ep:3d}/{a.epochs}  train {tot/max(nb_,1):.4f}  "
                  f"val tau {TAUS[j]:.2f} eventDA {da[j]:.4f}{star}  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)

    trunk.load_state_dict(best_sd[0])
    head.load_state_dict(best_sd[1])
    trunk.eval()
    head.eval()
    per_te = evaluate(te, ds["test"])
    jt = int(np.argmin(np.abs(TAUS - best_tau)))
    test = float(np.mean([v[jt] for v in per_te.values()]))
    print(f"\n=== {tag} ===")
    for mm in sorted(per_te, key=lambda s: int(s.split("_")[-1]) if "_" in s else int(s[:-2])):
        print(f"  {mm:>6s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f} (importance-weighted sample)")
    print(f"  refs: head 0.8444/0.8686 | +ITI pending | PRE-Mamba 0.7708")
    print(f"  mixed-cell recall must be measured with mixed_cell_diag2.py")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"trunk": best_sd[0], "head": best_sd[1], "tau": best_tau,
                "test": test}, f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "params": npar, "tau": best_tau, "val": best,
               "test": test,
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
