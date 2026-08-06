"""EvORSP-3T/E+H: trunk + per-event head, trained END-TO-END.

Fixes the observed occlusion failure (a car edge deleted where a large rain drop
floods its cell). Measured scene-event recall inside mixed cells:
    ctx_c2 (count-majority)  0.4005
    exact  (BA-weighted)     0.4819      <- reweighting alone: partial
    PRE-Mamba                0.6903
The residual is LABEL AGGREGATION: the per-cell target discards the minority
events before the model ever sees them, so no loss or threshold can recover
them. The fix is to decide per EVENT, supervised per EVENT.

Architecture -- coarse grid emits a PARAMETER, not a decision:
    trunk  [256^2 x 16]  ->  per-cell logit + 32-d features
    each event samples both at its own (x, y, t) by trilinear interpolation
    plus a NATIVE-resolution 3x3 x (2 polarities x 4 time bins) count patch,
      i.e. 72 numbers describing the event's own neighbourhood at full sensor
      resolution and 8 temporal bins -- twice the trunk's T=4, which the oracle
      says is the more valuable axis (temporal beats spatial ~3x)
    -> MLP -> one logit per event, residual on the trunk logit

Loss: per-EVENT BCE with balanced-accuracy weights, w = 1/N_bg for scene events
and 1/N_rn for rain events, which is the exact per-event decomposition of the
metric (identity verified to 1.1e-16 in verify_lossmath.py).

Trained jointly: the bolt-on version (frozen trunk) is the configuration the
refinement literature reports as regressing below baseline.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
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

from rsp_3d import ORSPNet3D

PACK = f"{C.KITTI_PACK}"
SRC = f"{C.KITTI_SRC}"
TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R = 16, 256
NW, NH = 460, 352
TAUS = np.linspace(0.05, 0.95, 19)
EPS = 1e-6
N_SAMP = 20000            # events sampled per frame for the per-event loss


def key(x, y, t):
    return (t.astype(np.int64) * (NW * NH) + y.astype(np.int64) * NW
            + x.astype(np.int64))


class EventSet(Dataset):
    """Returns trunk input planes + a sample of raw events with labels."""

    def __init__(self, split, t_front, t_out, ctx, train):
        self.files = sorted(glob.glob(f"{PACK}/{split}/*/*.npz"))
        self.mm = [f.split("/")[-2] for f in self.files]
        self.tf, self.to, self.ctx, self.train = t_front, t_out, ctx, train
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.files)

    def _planes(self, path):
        with np.load(path) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
        return on, off

    def __getitem__(self, i):
        f = self.files[i]
        mm = self.mm[i]
        base = os.path.basename(f)
        on, off = self._planes(f)
        onf = on.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)
        offf = off.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)

        ex = []
        idx = int(base.split(".")[0])
        for k in range(1, self.ctx + 1):
            p = f"{PACK}/{f.split('/')[-3]}/{mm}/{max(idx - k, 0):010d}.npz"
            pon, poff = self._planes(p if os.path.exists(p) else f)
            ex.append(pon.max(0)[None].astype(np.float32))
            ex.append(poff.max(0)[None].astype(np.float32))
        ex = (np.concatenate(ex, 0) if ex else np.zeros((0, R, R), np.float32))

        # raw events + per-event labels
        with np.load(f"{SRC}/merge_data/{mm}/{base}") as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        with np.load(f"{SRC}/raw_data/{base}") as d:
            clean = np.sort(key(d["x"], d["y"], d["t"]))
        rain = ~np.isin(key(x, y, t), clean)

        n = len(x)
        if n > N_SAMP:
            sel = (self.rng if self.train else
                   np.random.default_rng(i)).choice(n, N_SAMP, replace=False)
        else:
            sel = np.arange(n)
        t0, span = t.min(), max(int(t.max() - t.min()), 1)
        tn = ((t[sel] - t0) / span).astype(np.float32)         # [0,1]
        # native 3x3 x (2 pol x 4 bins) count patch
        tb4 = np.clip((tn * 4).astype(np.int64), 0, 3)
        G = np.zeros((8, NH + 2, NW + 2), np.uint8)
        tb4_all = np.clip(((t - t0) / span * 4).astype(np.int64), 0, 3)
        np.add.at(G, ((p == 1).astype(np.int64) * 4 + tb4_all,
                      y + 1, x + 1), 1)
        xs, ys = x[sel], y[sel]
        patch = np.stack([G[:, ys + dy, xs + dx]
                          for dy in range(3) for dx in range(3)], 1)
        patch = np.log1p(patch.reshape(len(sel), 72).astype(np.float32))

        return (torch.from_numpy(onf).float(), torch.from_numpy(offf).float(),
                torch.from_numpy(ex).float(),
                torch.from_numpy(xs.astype(np.float32)),
                torch.from_numpy(ys.astype(np.float32)),
                torch.from_numpy(tn),
                torch.from_numpy(patch),
                torch.from_numpy((~rain[sel]).astype(np.float32)),   # 1 = scene
                i)


class Head(nn.Module):
    """Per-event MLP: trunk logit + trunk features + native patch -> logit."""

    def __init__(self, feat_dim=32, hidden=48):
        super().__init__()
        din = 1 + feat_dim + 72 + 1                      # +1 polarity-free time
        self.fc1 = nn.Linear(din, hidden)
        self.fc2 = nn.Linear(hidden + 1, 32)
        self.fc3 = nn.Linear(32 + 1, 1)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)                    # start == trunk

    def forward(self, l, feat, patch, tn):
        z = torch.cat([l, feat, patch, tn], -1)
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(torch.cat([h, l], -1)))
        return l + self.fc3(torch.cat([h, l], -1))       # residual on trunk


def sample_at(vol, xs, ys, tn):
    """Trilinear sample [B,C,T,H,W] at native event coords -> [B,N,C]."""
    B, C, T, H, W = vol.shape
    gx = (xs / (NW - 1)) * 2 - 1
    gy = (ys / (NH - 1)) * 2 - 1
    gt = tn * 2 - 1
    grid = torch.stack([gx, gy, gt], -1)[:, :, None, None, :]   # [B,N,1,1,3]
    out = F.grid_sample(vol, grid, mode="bilinear", align_corners=True)
    return out[:, :, :, 0, 0].permute(0, 2, 1)                  # [B,N,C]


def ba_weights(lab):
    """w = 1/N_bg for scene, 1/N_rn for rain (per frame) -- the exact metric."""
    n_bg = lab.sum(1, keepdim=True).clamp(min=1.0)
    n_rn = (1 - lab).sum(1, keepdim=True).clamp(min=1.0)
    return lab / n_bg + (1 - lab) / n_rn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2)
    ap.add_argument("--tfront", type=int, default=4)
    ap.add_argument("--tout", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="ctx_f4o16_c2")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = f"peh_c{a.ctx}f{a.tfront}" + (f"_s{a.seed}" if a.seed else "")
    n_extra = 2 * a.ctx

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    ds = {s: EventSet(s, a.tfront, a.tout, a.ctx, s == "train")
          for s in ("train", "val", "test")}
    tr = DataLoader(ds["train"], batch_size=a.batch, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(ds["val"], batch_size=a.batch, **dl)
    te = DataLoader(ds["test"], batch_size=a.batch, **dl)

    trunk = ORSPNet3D(T=a.tfront, dilations=(1, 8, 32, 64), num_blocks=3,
                      use_off=True, out_chans=a.tout, n_extra=n_extra).to(DEV)
    if a.init and os.path.exists(f"{TMP}/{a.init}.pt"):
        trunk.load_state_dict(torch.load(f"{TMP}/{a.init}.pt",
                                         map_location="cpu")["state_dict"])
        print(f"trunk warm-started from {a.init}", flush=True)
    head = Head().to(DEV)
    feats = {}
    trunk.out_proj.register_forward_pre_hook(
        lambda m, inp: feats.__setitem__("f", inp[0]))

    npar = sum(p.numel() for p in trunk.parameters()) + \
        sum(p.numel() for p in head.parameters())
    print(f"{tag}: trunk+head {npar:,} params "
          f"(head {sum(p.numel() for p in head.parameters()):,})", flush=True)

    opt = torch.optim.AdamW([{"params": trunk.parameters(), "lr": 2e-4},
                             {"params": head.parameters(), "lr": 1e-3}],
                            weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)

    def forward(on, off, ex, xs, ys, tn, patch):
        kw = {"x_extra": ex.to(DEV)} if n_extra else {}
        logit_map = trunk(on.to(DEV), x_off=off.to(DEV), **kw)   # [B,To,H,W]
        fmap = feats["f"]                                        # [B,32,H,W]
        B, To, H, W = logit_map.shape
        lv = sample_at(logit_map[:, None], xs.to(DEV), ys.to(DEV), tn.to(DEV))
        fv = sample_at(fmap[:, :, None].expand(-1, -1, To, -1, -1),
                       xs.to(DEV), ys.to(DEV), tn.to(DEV))
        return lv, fv, logit_map

    @torch.no_grad()
    def evaluate(loader, dset):
        acc = {}
        for on, off, ex, xs, ys, tn, patch, lab, idx in loader:
            lv, fv, _ = forward(on, off, ex, xs, ys, tn, patch)
            out = torch.sigmoid(head(lv, fv, patch.to(DEV),
                                     tn.to(DEV)[..., None]))[..., 0]
            lab = lab.to(DEV)
            for b in range(on.shape[0]):
                nb = float(lab[b].sum())
                nr = float((1 - lab[b]).sum())
                if nb < 10 or nr < 10:
                    continue
                mm = dset.mm[int(idx[b])]
                d = acc.setdefault(mm, [np.zeros(len(TAUS)),
                                        np.zeros(len(TAUS)), 0])
                for j, t in enumerate(TAUS):
                    k = out[b] > t
                    d[0][j] += float((k.float() * lab[b]).sum()) / nb
                    d[1][j] += float(((~k).float() * (1 - lab[b])).sum()) / nr
                d[2] += 1
        return {mm: 0.5 * (s + n) / max(c, 1) for mm, (s, n, c) in acc.items()}

    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        trunk.train()
        head.train()
        tot = nb = 0
        for on, off, ex, xs, ys, tn, patch, lab, _ in tr:
            lv, fv, _ = forward(on, off, ex, xs, ys, tn, patch)
            out = head(lv, fv, patch.to(DEV), tn.to(DEV)[..., None])[..., 0]
            lab = lab.to(DEV)
            w = ba_weights(lab)
            bce = F.binary_cross_entropy_with_logits(out, lab, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=EPS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(trunk.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
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
            print(f"  ep {ep:3d}/{a.epochs}  train {tot/max(nb,1):.4f}  "
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
    for mm in sorted(per_te, key=lambda s: int(s[:-2])):
        print(f"  {mm:>6s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f}  (val tau {best_tau:.2f})")
    print(f"  trunk-only reference: ctx_c2 0.9291 protocol / 0.9332 self-prior")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"trunk": best_sd[0], "head": best_sd[1], "tau": best_tau,
                "test": test}, f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "ctx": a.ctx, "tfront": a.tfront, "params": npar,
               "tau": best_tau, "val": best, "test": test,
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
