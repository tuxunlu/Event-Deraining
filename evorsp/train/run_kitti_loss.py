"""Loss-function falsifier ladder for the event-weighted metric.

The trained model reaches 0.9196 event-DA against a 0.9807 ceiling at T_out=16.
The target is nearly optimal (count-majority ceils at 0.9440 vs BA-optimal
0.9466 at T=1), so the suspect is the LOSS, which is misaligned with the metric
in three ways:
  (1) every lit cell contributes equally, but the metric weights a cell by how
      many events it holds;
  (2) the hard majority label discards magnitude -- 51/49 and 100/0 cells get
      identical targets and identical gradient;
  (3) BCE optimizes log-likelihood, not balanced accuracy, and the metric's
      per-frame normalization by class totals appears nowhere.

Arms (identical trunk, data, schedule, seed -- ONLY the loss changes):
  bce     control, current: unweighted BCE on hard majority, lit-masked
  wbce    BCE weighted by cell event count (bg+rn)      -> fixes (1)
  soft    BCE against the soft ratio bg/(bg+rn), count-weighted -> fixes (1)+(2)
  softba  differentiable event-DA itself:
              DA_soft = 0.5*[ sum_c p_c*bg_c / N_bg + sum_c (1-p_c)*rn_c / N_rn ]
          minimizing 1 - DA_soft is EXACTLY the metric's expectation under a
          random decision with probability p_c                 -> fixes (1)+(3)
  hybrid  wbce + softba (BCE keeps probabilities calibrated for tau selection,
          softba supplies the metric-aligned gradient)

Pre-registered: an arm WINS only if it beats the control's 2-seed mean 0.9196
by more than the control's own 2-seed range, 0.0026. Anything smaller is noise.
"""
import argparse
import glob
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

ROOT = "/fs/nexus-scratch/tuxunlu/kitti_t16e"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)
EPS = 1e-6


class KittiLossSet(Dataset):
    def __init__(self, split, t_front, t_out):
        self.files = sorted(glob.glob(f"{ROOT}/{split}/*/*.npz"))
        self.mm = [f.split("/")[-2] for f in self.files]
        self.tf, self.to = t_front, t_out

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
        k = T_BUILD // self.to
        bgo = bg.reshape(self.to, k, R, R).sum(1)
        rno = rn.reshape(self.to, k, R, R).sum(1)
        return (torch.from_numpy(onf).float(), torch.from_numpy(offf).float(),
                torch.from_numpy(bgo), torch.from_numpy(rno), i)


def compute_loss(kind, logits, bg, rn):
    """logits/bg/rn: [B, T_out, H, W]."""
    lit = (bg + rn) > 0
    hard = (bg > rn).float()
    cnt = bg + rn

    if kind == "bce":
        bce = F.binary_cross_entropy_with_logits(logits, hard, reduction="none")
        return (bce * lit).sum() / lit.sum().clamp(min=1.0)

    if kind == "exact":
        # RANK 1: exact per-cell decomposition of event-DA (identity verified,
        # max error 1.1e-16). DA = 0.5 + 0.5*sum_c d_c*(bg_c/N_bg - rn_c/N_rn),
        # so the label is the sign of that slope and the cost is its magnitude.
        dims = tuple(range(1, logits.dim()))
        n_bg = bg.sum(dims, keepdim=True).clamp(min=EPS)
        n_rn = rn.sum(dims, keepdim=True).clamp(min=EPS)
        slope = bg / n_bg - rn / n_rn
        y = (slope > 0).float()
        w = slope.abs() * lit
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (bce * w).sum() / w.sum().clamp(min=EPS)

    if kind == "cwbce":
        # RANK 3: count-weighted AND class-weighted, keeping the hard label.
        dims = tuple(range(1, logits.dim()))
        n_bg = bg.sum(dims, keepdim=True).clamp(min=EPS)
        n_rn = rn.sum(dims, keepdim=True).clamp(min=EPS)
        pi = n_rn / (n_bg + n_rn)                 # event-level rain prior
        cls_w = torch.where(hard > 0.5, pi, 1.0 - pi)
        w = cnt * cls_w * lit
        bce = F.binary_cross_entropy_with_logits(logits, hard, reduction="none")
        return (bce * w).sum() / w.sum().clamp(min=EPS)

    if kind == "wbce":
        bce = F.binary_cross_entropy_with_logits(logits, hard, reduction="none")
        w = cnt * lit
        return (bce * w).sum() / w.sum().clamp(min=EPS)

    if kind == "soft":
        soft = bg / (cnt + EPS)
        bce = F.binary_cross_entropy_with_logits(logits, soft, reduction="none")
        w = cnt * lit
        return (bce * w).sum() / w.sum().clamp(min=EPS)

    if kind in ("softba", "hybrid"):
        p = torch.sigmoid(logits)
        dims = tuple(range(1, logits.dim()))                  # per-sample
        n_bg = bg.sum(dims).clamp(min=EPS)
        n_rn = rn.sum(dims).clamp(min=EPS)
        sr = (p * bg).sum(dims) / n_bg
        nr = ((1 - p) * rn).sum(dims).clamp(min=0) / n_rn
        ba = 0.5 * (sr + nr)
        loss_ba = (1.0 - ba).mean()
        if kind == "softba":
            return loss_ba
        bce = F.binary_cross_entropy_with_logits(logits, hard, reduction="none")
        w = cnt * lit
        return (bce * w).sum() / w.sum().clamp(min=EPS) + loss_ba

    if kind == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, hard, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * hard + (1 - p) * (1 - hard)
        foc = ((1 - pt) ** 2.0) * bce
        return (foc * lit).sum() / lit.sum().clamp(min=1.0)

    raise ValueError(kind)


@torch.no_grad()
def self_prior_test(model, loader):
    """Per-frame threshold = count-weighted mean p over lit cells (RANK 2).
    Measured +0.006 over a single global tau on the control checkpoint."""
    out = []
    for on, off, bg, rn, _ in loader:
        p = torch.sigmoid(model(on.to(DEV), x_off=off.to(DEV)))
        bg, rn = bg.to(DEV), rn.to(DEV)
        for b in range(on.shape[0]):
            nb, nr = float(bg[b].sum()), float(rn[b].sum())
            if nb < 1 or nr < 1:
                continue
            cnt = bg[b] + rn[b]
            tau = float((p[b] * cnt).sum() / cnt.sum().clamp(min=EPS))
            keep = p[b] > tau
            out.append(0.5 * (float((bg[b] * keep).sum()) / nb
                              + float((rn[b] * ~keep).sum()) / nr))
    return float(np.mean(out)) if out else 0.0


@torch.no_grad()
def evaluate(model, loader, ds):
    acc = {}
    for on, off, bg, rn, idx in loader:
        p = torch.sigmoid(model(on.to(DEV), x_off=off.to(DEV)))
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
    ap.add_argument("--loss", required=True,
                    choices=["bce", "exact", "cwbce", "wbce", "soft",
                             "softba", "hybrid", "focal"])
    ap.add_argument("--tout", type=int, default=16)
    ap.add_argument("--tfront", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = f"loss_{a.loss}_o{a.tout}" + (f"_s{a.seed}" if a.seed else "")

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    tr_ds = KittiLossSet("train", a.tfront, a.tout)
    va_ds = KittiLossSet("val", a.tfront, a.tout)
    te_ds = KittiLossSet("test", a.tfront, a.tout)
    tr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(va_ds, batch_size=a.batch, **dl)
    te = DataLoader(te_ds, batch_size=a.batch, **dl)

    m = ORSPNet3D(T=a.tfront, dilations=(1, 8, 32, 64), num_blocks=3,
                  use_off=True, out_chans=a.tout).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | loss={a.loss} | train {len(tr_ds)} "
          f"val {len(va_ds)} test {len(te_ds)}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        tot = nb = 0
        for on, off, bg, rn, _ in tr:
            out = m(on.to(DEV, non_blocking=True),
                    x_off=off.to(DEV, non_blocking=True))
            loss = compute_loss(a.loss, out, bg.to(DEV, non_blocking=True),
                                rn.to(DEV, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
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
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in m.state_dict().items()}
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
    sp = self_prior_test(m, te)
    win = test > 0.9196 + 0.0026
    print(f"\n=== {tag} ===")
    for mm in sorted(per_te, key=lambda s: int(s[:-2])):
        print(f"  {mm:>6s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f}  (val tau {best_tau:.2f}, "
          f"val {best:.4f})")
    print(f"  self-prior tau (per frame)  test event-DA {sp:.4f}   "
          f"(control with same rule: 0.9245)")
    print(f"  control (bce, 2 seeds) 0.9196 +/- 0.0013, range 0.0026")
    print(f"  verdict: {'WIN' if win else 'no better than control'}")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "tau": best_tau, "test": test},
               f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "loss": a.loss, "tout": a.tout, "params": npar,
               "seed": a.seed, "tau": best_tau, "val": best, "test": test,
               "test_selfprior": sp, "win": bool(win),
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
