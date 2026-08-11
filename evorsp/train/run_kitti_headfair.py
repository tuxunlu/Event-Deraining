"""The per-event head on EVERY backbone, so 0.9576 can be attributed.

Our headline KITTI number uses a trunk PLUS a per-event head; every competitor
number is a backbone alone. Decomposed on our own stack the head is worth
+0.028 and context +0.007, so most of the gap between the fair-harness 0.9215
and the deployed 0.9576 is the head -- and the head is architecture-agnostic.
Until FourierMamba and StreakNet get one, 0.9576 is evidence about a SYSTEM,
not about an architecture.

FAIRNESS OF THE TAP. Each body is read at the feature maps it exposes at full
resolution -- every block output plus the input to its output projection -- and
those are projected to a COMMON 128 channels by a 1x1 conv, so the head itself
is byte-identical across bodies and only the backbone differs. FourierMamba is
a multi-scale U-Net whose interior runs at reduced resolution, so it offers one
full-resolution tap rather than several; that asymmetry is inherent to the
architecture and is reported rather than engineered away.

The patch and structure-tensor columns come from RAW EVENTS, not from the
trunk, so the existing cache is reused unchanged for every body.
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
from bodies_e import FrontendBody
from run_kitti_perevent import sample_at
from fast_tensor import tensor_cols_fast

PACK = f"{C.KITTI_PACK}"
CACHE = f"{C.KITTI_HEAD}"
SRC = f"{C.KITTI_SRC}"
TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R, NW, NH = 16, 256, 460, 352
DILS = (1, 3, 9)   # L3a: 3x3 at these dilations -> 27x27 native view
NBIN = 4                                   # temporal bins per polarity in patch
PATCH_D = len(DILS) * 9 * 2 * NBIN         # 216
SCALES, SLICE_NS, TAU_NS = [4, 16, 64], 1_000_000, 5_000_000
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


class CacheSet(Dataset):
    def __init__(self, split):
        self.files = sorted(glob.glob(f"{CACHE}/{split}/*/*.npz"))
        self.mm = [f.split("/")[-2] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        cf = self.files[i]
        mm, base = self.mm[i], os.path.basename(cf)
        with np.load(cf) as d:
            sel, tcols = d["sel"].astype(np.int64), d["tcols"].astype(np.float32)
            lab, inv_p = d["lab"].astype(np.float32), d["inv_p"]
            n_bg, n_rn = float(d["n_bg"]), float(d["n_rn"])
        pk = cf.replace(CACHE, PACK)
        with np.load(pk) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
        on4 = on.reshape(4, 4, R, R).max(1).astype(np.float32)
        off4 = off.reshape(4, 4, R, R).max(1).astype(np.float32)
        idx = int(base.split(".")[0])
        ex = []
        for k in (1, 2):
            pv = f"{os.path.dirname(pk)}/{max(idx - k, 0):010d}.npz"
            with np.load(pv if os.path.exists(pv) else pk) as d:
                pon = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
                poff = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            ex.append(pon.max(0)[None].astype(np.float32))
            ex.append(poff.max(0)[None].astype(np.float32))
        ex = np.concatenate(ex, 0)

        with np.load(f"{SRC}/merge_data/{mm}/{base}") as d:
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
        din = 1 + feat_dim + PATCH_D + 19 + 1
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
    ap.add_argument("--init", default="ctx_f4o16_c2")
    ap.add_argument("--ctx", type=int, default=2, choices=[0, 2],
                    help="0 drops the cache's context planes, so a "
                         "context-free fair checkpoint loads and the "
                         "HEAD effect is isolated by itself")
    ap.add_argument("--body", default="evorsp",
                    choices=["evorsp", "orsp", "dffn", "streaknet",
                             "fmamba"])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = f"headfair_{a.body}_c{a.ctx}" + (f"_s{a.seed}" if a.seed else "")

    dl = dict(num_workers=3, pin_memory=True, persistent_workers=True)
    ds = {s: CacheSet(s) for s in ("train", "val", "test")}
    tr = DataLoader(ds["train"], batch_size=a.batch, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(ds["val"], batch_size=a.batch, **dl)
    te = DataLoader(ds["test"], batch_size=a.batch, **dl)

    FEAT_D = 128          # common width, so the head is identical everywhere
    if a.body == "evorsp":
        trunk = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                          use_off=True, out_chans=16, n_extra=2 * a.ctx).to(DEV)
        trunk.load_state_dict(torch.load(f"{TMP}/{a.init}.pt",
                                         map_location="cpu")["state_dict"])
        taps = [trunk.out_proj] + list(trunk.blocks)
        raw_d = 32 * (len(trunk.blocks) + 1)
    else:
        trunk = FrontendBody(a.body, T=4, t_out=16, dim=32, n_extra=2 * a.ctx).to(DEV)
        sd = torch.load(f"{TMP}/{a.init}.pt", map_location="cpu")
        trunk.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        b = trunk.body
        if a.body == "fmamba":
            # U-Net interior runs at reduced resolution; the only full-resolution
            # tap is the input to the 1x1 that maps back to T_out. Reported, not
            # engineered around.
            taps = [trunk.head]
            raw_d = trunk.head.in_channels
        else:
            taps = [b.out_proj] + list(b.blocks)
            raw_d = 32 * (len(b.blocks) + 1)

    # project whatever the body exposes to a COMMON width
    proj = nn.Conv2d(raw_d, FEAT_D, 1).to(DEV)
    head = HeadV2(feat_dim=FEAT_D).to(DEV)
    feats = {}
    # tap 0 is a PRE-hook (input to the output projection); the rest are outputs
    taps[0].register_forward_pre_hook(
        lambda m, inp: feats.__setitem__(0, inp[0]))
    for bi, mod in enumerate(taps[1:]):
        mod.register_forward_hook(
            lambda m, i, o, bi=bi: feats.__setitem__(bi + 1, o))
    NTAP = len(taps)
    npar = sum(p.numel() for p in trunk.parameters()) + \
        sum(p.numel() for p in head.parameters())
    print(f"{tag}: trunk+head {npar:,} params (head "
          f"{sum(p.numel() for p in head.parameters()):,}), patch dim {PATCH_D}",
          flush=True)

    opt = torch.optim.AdamW([{"params": list(proj.parameters())},
                             {"params": trunk.parameters(), "lr": 2e-4},
                             {"params": head.parameters(), "lr": 1e-3}],
                            weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)

    def fwd(on, off, ex, xs, ys, tn, patch, tcols):
        kw = {"x_extra": ex.to(DEV)} if a.ctx else {}
        lm = trunk(on.to(DEV), x_off=off.to(DEV), **kw)
        fm = proj(torch.cat([feats[i] for i in range(NTAP)], 1))
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
    for mm in sorted(per_te, key=lambda s: int(s[:-2])):
        print(f"  {mm:>6s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f} (importance-weighted sample)")
    print(f"  refs: trunk 0.9332 | head v1 0.9396 | head v2 0.9575 | PRE-Mamba 0.9172")
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
