"""FAIRNESS at event level: every architecture, same input, same target.

Identical to run_kitti_e.py except the body is swappable (bodies_e.FrontendBody):
DFFN, ORSPNet+dil, StreakNet, FourierMamba2D all read the SAME 10-plane
temporal+polarity frontend, predict the SAME count-majority target, use the
SAME out_chans=T_out readout, and are scored with the SAME event-level metric
as EvORSP-3T/E and PRE-Mamba. Differences that survive are architecture.

What changed vs run_kitti3d.py -- supervision and readout, NOT architecture:

  1. TARGET.  Old GT was "pixel contains >=1 clean ON event" (an OR over the
     window, ON only). Measured event-DA ceiling of that target: 0.6981, and
     the trained model already reaches 0.7052 -- it is saturated. New target is
     "background events outnumber rain events in this cell" (rule C, ceiling
     0.9440 at T_out=1) computed from exact per-cell counts.
  2. POLARITY.  lit = ON or OFF (we now account for every event, not just ON);
     the old ON-only lit mask is what pinned SR at 0.52.
  3. TEMPORAL READOUT.  out_chans = T_out, so the head emits one decision per
     (time-bin, pixel) instead of one per pixel. This is PRE-Mamba's actual
     granularity mechanism -- its GridSample divides only the time axis
     (coord / (1,1,grid_size)) and broadcasts per-voxel labels back to events.
     Costs ~100 parameters; the frontend still sees T_front planes.

The metric here is EVENT-level DA computed exactly from per-cell background and
rain counts (no raw-event pass needed): SR = kept background / all background,
NR = dropped rain / all rain, per frame, frame-averaged -- identical accounting
to PRE-Mamba's SemSegTester. One global tau selected on val {20,80}mm, reported
on test {50,150}mm.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import argparse
import glob
import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
from bodies_e import FrontendBody

ROOT = f"{C.KITTI_PACK}"
TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)


class KittiESet(Dataset):
    def __init__(self, split, t_front, t_out, target, ctx=0):
        self.files = sorted(glob.glob(f"{ROOT}/{split}/*/*.npz"))
        self.mm = [f.split("/")[-2] for f in self.files]
        self.tf, self.to, self.target = t_front, t_out, target
        self.ctx = ctx
        assert T_BUILD % t_front == 0 and T_BUILD % t_out == 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            bg = d["bg"].reshape(T_BUILD, R, R).astype(np.float32)
            rn = d["rn"].reshape(T_BUILD, R, R).astype(np.float32)
        onf = on.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)
        offf = off.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)
        ex = []
        for k in range(1, self.ctx + 1):
            d0, b0 = os.path.split(self.files[i])
            j = max(int(b0.split(".")[0]) - k, 0)
            pv = f"{d0}/{j:010d}.npz"
            pv = pv if os.path.exists(pv) else self.files[i]
            with np.load(pv) as dp:
                pon = np.unpackbits(dp["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
                poff = np.unpackbits(dp["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            ex.append(pon.max(0)[None].astype(np.float32))
            ex.append(poff.max(0)[None].astype(np.float32))
        exa = (np.concatenate(ex, 0) if ex else np.zeros((0, R, R), np.float32))
        k = T_BUILD // self.to
        bgo = bg.reshape(self.to, k, R, R).sum(1)
        rno = rn.reshape(self.to, k, R, R).sum(1)
        lit = (bgo + rno) > 0
        tgt = (bgo > rno) if self.target == "maj" else (bgo > 0)
        return (torch.from_numpy(onf).float(), torch.from_numpy(offf).float(),
                torch.from_numpy(exa).float(),
                torch.from_numpy(tgt & lit).float(), torch.from_numpy(lit).float(),
                torch.from_numpy(bgo), torch.from_numpy(rno), i)


def lit_bce(logits, tgt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, ds):
    """Exact event-level DA at every tau, per intensity."""
    acc = {}
    for on, off, ex, tgt, lit, bg, rn, idx in loader:
        kw = {"x_extra": ex.to(DEV)} if ex.shape[1] else {}
        p = torch.sigmoid(model(on.to(DEV), x_off=off.to(DEV), **kw))
        bg, rn = bg.to(DEV), rn.to(DEV)
        for b in range(on.shape[0]):
            nb, nr = float(bg[b].sum()), float(rn[b].sum())
            if nb < 1 or nr < 1:
                continue
            mm = ds.mm[int(idx[b])]
            d = acc.setdefault(mm, [np.zeros(len(TAUS)), np.zeros(len(TAUS)), 0])
            for j, t in enumerate(TAUS):
                keep = p[b] > t
                d[0][j] += float((bg[b] * keep).sum()) / nb
                d[1][j] += float((rn[b] * ~keep).sum()) / nr
            d[2] += 1
    return {mm: 0.5 * (sr + nr) / max(n, 1) for mm, (sr, nr, n) in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True,
                    choices=["dffn", "orsp", "streaknet", "fmamba"])
    ap.add_argument("--tout", type=int, default=16)
    ap.add_argument("--tfront", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=0,
                    help="inter-window context planes, as in "
                         "run_kitti_ctx.py. Context is an INPUT "
                         "change, so every body gets it or none do.")
    ap.add_argument("--target", default="maj", choices=["maj", "or"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bf16", action="store_true",
                    help="bfloat16 autocast. FourierMamba's rfft2 uses the "
                         "default norm='backward', so its fp16 GRADIENTS "
                         "overflow: GradScaler then skips nearly every step "
                         "and the model never leaves initialisation (observed "
                         "as train=nan with event-DA pinned at 0.5273). bf16 "
                         "has fp32 exponent range and cannot overflow.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = f"fair_{a.body}_o{a.tout}" + (f"_c{a.ctx}" if a.ctx else "") + (f"_s{a.seed}" if a.seed else "")

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    tr_ds = KittiESet("train", a.tfront, a.tout, a.target, a.ctx)
    va_ds = KittiESet("val", a.tfront, a.tout, a.target, a.ctx)
    te_ds = KittiESet("test", a.tfront, a.tout, a.target, a.ctx)
    tr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(va_ds, batch_size=a.batch, **dl)
    te = DataLoader(te_ds, batch_size=a.batch, **dl)

    m = FrontendBody(a.body, T=a.tfront, t_out=a.tout,
                     n_extra=2 * a.ctx).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | train {len(tr_ds)} val {len(va_ds)} "
          f"test {len(te_ds)}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp and not a.bf16)
    _adt = torch.bfloat16 if a.bf16 else torch.float16
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        tot = nb = 0
        nskip = 0
        for on, off, ex, tgt, lit, bg, rn, _ in tr:
            kw = ({"x_extra": ex.to(DEV, non_blocking=True)}
                  if ex.shape[1] else {})
            with torch.cuda.amp.autocast(enabled=a.amp or a.bf16, dtype=_adt):
                out = m(on.to(DEV, non_blocking=True),
                        x_off=off.to(DEV, non_blocking=True), **kw)
                loss = lit_bce(out.float(), tgt.to(DEV, non_blocking=True),
                               lit.to(DEV, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nskip += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        sch.step()

        m.eval()
        per = evaluate(m, va, va_ds)
        da = np.mean([v for v in per.values()], axis=0)
        j = int(np.argmax(da))
        star = ""
        if da[j] > best:
            best, best_tau = float(da[j]), float(TAUS[j])
            best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            star = "  *"
        if ep % 5 == 0 or ep == 1 or star:
            print(f"  ep {ep:3d}/{a.epochs}  train {tot/max(nb,1):.4f}  "
                  f"val tau {TAUS[j]:.2f} eventDA {da[j]:.4f}{star}  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    per_te = evaluate(m, te, te_ds)
    jt = int(np.argmin(np.abs(TAUS - best_tau)))
    test = float(np.mean([v[jt] for v in per_te.values()]))
    print(f"\n=== {tag} ===")
    print(f"  params {npar:,}")
    for mm in sorted(per_te, key=lambda s: int(s[:-2])):
        print(f"  {mm:>6s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f}   (val tau {best_tau:.2f}, "
          f"val {best:.4f})")
    print("  reference: PRE-Mamba 0.9172 | EvORSP-3T/E 0.9196 "
          "(28,555p) | ceiling 0.9807 at T_out=16")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "tau": best_tau, "test": test},
               f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "body": a.body, "params": npar, "tout": a.tout,
               "seed": a.seed, "tau": best_tau, "val": best, "test": test,
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
