"""EvORSP-3T (and controls) on KITTI with per-event temporal data.

Protocol is the established KITTI one, replicated exactly so the result is
directly comparable to the 2D leaderboard (best: 36,782p / 7.11ms / 0.9248):
  * train on 14 intensities, select ONE global tau on val {20,80}mm,
    report mean of per-intensity DA on test {50,150}mm at that tau;
  * 50 epochs, batch 4, AdamW lr 5e-4 wd 5e-3, cosine T_max=50 eta_min=1e-6,
    grad clip 1.0 -- identical to run_protocol.py;
  * DA per intensity = mean over frames of 1/2(SR+NR), computed over LIT pixels.

The ONLY difference from the 2D pipeline is the input: T temporal sub-window
ON planes (+ OFF planes) instead of the collapsed 1-bit frame. The T=1 ON-only
case reproduces the 2D input exactly (OR-union), so the leaderboard is the
honest control.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D, DFFN3T

ROOT = "/fs/nexus-scratch/tuxunlu/kitti_t16"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)


class KittiSet(Dataset):
    def __init__(self, split, T):
        self.files = sorted(glob.glob(f"{ROOT}/{split}/*/*.npz"))
        self.mm = [f.split("/")[-2] for f in self.files]
        self.T = T
        assert T_BUILD % T == 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            gt = np.unpackbits(d["gt"])[: R * R].reshape(R, R)
        on = on.reshape(self.T, T_BUILD // self.T, R, R).max(1)
        off = off.reshape(self.T, T_BUILD // self.T, R, R).max(1)
        return (torch.from_numpy(on).float(), torch.from_numpy(off).float(),
                torch.from_numpy(gt).float().unsqueeze(0), i)


def lit_bce(logits, gt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, ds, use_off):
    """Per-intensity DA at every tau: {mm: [DA(tau_j)]}, protocol-style."""
    acc = {}
    for on, off, gt, idx in loader:
        on, gt = on.to(DEV), gt.to(DEV)
        p = torch.sigmoid(model(on, x_off=off.to(DEV)) if use_off else model(on))
        lit = on.amax(1, keepdim=True) > 0.5
        real = (gt > 0.5) & lit
        rain = lit & ~(gt > 0.5)
        for b in range(on.shape[0]):
            rs, ns = int(real[b].sum()), int(rain[b].sum())
            if rs == 0 or ns == 0:
                continue
            mm = ds.mm[int(idx[b])]
            d = acc.setdefault(mm, [np.zeros(len(TAUS)), np.zeros(len(TAUS)), 0])
            pv = p[b]
            for j, t in enumerate(TAUS):
                pr = pv > t
                d[0][j] += ((pr & real[b]).sum().item()) / rs
                d[1][j] += (ns - (pr & rain[b]).sum().item()) / ns
            d[2] += 1
    return {mm: 0.5 * (sr + nr) / max(n, 1) for mm, (sr, nr, n) in acc.items()}


def mean_da(per):
    return np.mean([v for v in per.values()], axis=0)     # [len(TAUS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evorsp", choices=["evorsp", "dffn3t"])
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dilations", default="1,8,32,64")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tag = (("k3d_" if args.model == "evorsp" else f"k3d_{args.model}_")
           + f"T{args.T}b{args.blocks}" + ("off" if args.off else "")
           + (f"_s{args.seed}" if args.seed else ""))

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    tr_ds = KittiSet("train", args.T)
    va_ds = KittiSet("val", args.T)
    te_ds = KittiSet("test", args.T)
    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(va_ds, batch_size=args.batch, **dl)
    te = DataLoader(te_ds, batch_size=args.batch, **dl)

    D = tuple(int(v) for v in args.dilations.split(","))
    if args.model == "dffn3t":
        assert args.off, "dffn3t always takes the full temporal+OFF input"
        m = DFFN3T(T=args.T, num_blocks=args.blocks).to(DEV)
    else:
        m = ORSPNet3D(T=args.T, dilations=D, num_blocks=args.blocks,
                      use_off=args.off, use_temporal=(args.T > 1)).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | train {len(tr_ds)} val {len(va_ds)} "
          f"test {len(te_ds)} | {len(tr)} steps/epoch", flush=True)

    if args.T > 1:
        m.eval()
        with torch.no_grad():
            x = torch.rand(2, args.T, R, R, device=DEV).round()
            xo = torch.rand(2, args.T, R, R, device=DEV).round()
            sh = args.T // 2
            a = m(x, x_off=xo) if args.off else m(x)
            b = (m(torch.roll(x, sh, 1), x_off=torch.roll(xo, sh, 1)) if args.off
                 else m(torch.roll(x, sh, 1)))
            print(f"  phase-roll invariance: max|d| = {(a-b).abs().max().item():.2e}",
                  flush=True)
        m.train()

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                     eta_min=1e-6)
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = nb = 0
        for on, off, gt, _ in tr:
            on, gt = on.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
            lit = (on.amax(1, keepdim=True) > 0.5).float()
            out = (m(on, x_off=off.to(DEV, non_blocking=True)) if args.off
                   else m(on))
            loss = lit_bce(out, (gt > 0.5).float(), lit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()

        m.eval()
        per = evaluate(m, va, va_ds, args.off)
        da = mean_da(per)
        j = int(np.argmax(da))
        star = ""
        if da[j] > best:
            best, best_tau = float(da[j]), float(TAUS[j])
            best_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            star = "  *"
        if ep % 5 == 0 or ep == 1 or star:
            print(f"  ep {ep:3d}/{args.epochs}  train {tot/max(nb,1):.4f}  "
                  f"val tau {TAUS[j]:.2f}  valDA {da[j]:.4f}{star}  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    per_te = evaluate(m, te, te_ds, args.off)
    jt = max(0, min(len(TAUS) - 1, int(np.round((best_tau - 0.05) / 0.05))))
    test = float(mean_da(per_te)[jt])
    print(f"\n=== KITTI-3D RESULT: {tag} ===")
    print(f"  params {npar:,}")
    for mm in sorted(per_te, key=lambda s: int(s[:-2])):
        print(f"  {mm:>6s} DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN DA {test:.4f}   (val tau {best_tau:.2f}, val {best:.4f})")
    print(f"  reference 2D leaderboard: orsp_bal_dil 0.9248 / orsp_lit_dil 0.9244 "
          f"@ 36,782p, 7.11ms")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "params": npar, "tag": tag,
                "tau": best_tau, "test": test}, f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "T": args.T, "blocks": args.blocks,
               "off": bool(args.off), "seed": args.seed, "params": npar,
               "tau": best_tau, "val": best, "test": test,
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
