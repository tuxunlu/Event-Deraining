"""EvORSP-3T/E on REAL EVK4: event-accounting supervision, both splits.

Same fix that took KITTI from 0.7052 to 0.9183 event-DA, applied to real data:
  target  = background events outnumber rain events in the cell  (was: >=1
            scene ON event anywhere in the window)
  lit     = ON or OFF present                                    (was: ON only)
  readout = out_chans = T_out, one decision per (time-bin, pixel)

Splits:
  --split theirs : PRE-Mamba's own within-scene split (their scene_split table)
                   -> directly comparable to PRE-Mamba's 0.7708 test event-DA
                   and to our old-target 0.7985 on the same sequences.
  --split ours   : the campaign's scene-disjoint split (train 1-2 / val 3 /
                   test 4) -> updates the real-data numbers in the README
                   (old target: EvORSP-3T 0.8170 vs 2D control 0.7649).

Metric is exact event-level DA from per-cell counts: SR = kept background /
all background, NR = dropped rain / all rain, per frame, frame-averaged --
the same accounting as PRE-Mamba's SemSegTester. tau selected on val, reported
on test.
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

ROOT = "/fs/nexus-scratch/tuxunlu/real_t16e"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)

THEIRS = {
    "scene1": {"train": [1], "val": [4], "test": [2, 3]},
    "scene2": {"train": [3, 9], "val": [4, 6], "test": [1, 2, 5, 10]},
    "scene3": {"train": [4, 10], "val": [6, 8], "test": [2, 9]},
    "scene4": {"train": [1, 3], "val": [4, 9], "test": [2, 6, 13]},
}
OURS = {"train": ("scene1", "scene2"), "val": ("scene3",), "test": ("scene4",)}


def files_of(split_kind, split):
    out = []
    if split_kind == "theirs":
        for sc, d in THEIRS.items():
            for k in d[split]:
                out += sorted(glob.glob(f"{ROOT}/{sc}/rain_{k}/*.npz"))
    else:
        for sc in OURS[split]:
            out += sorted(glob.glob(f"{ROOT}/{sc}/rain_*/*.npz"))
    return out


class RealESet(Dataset):
    def __init__(self, split_kind, split, t_front, t_out):
        self.files = files_of(split_kind, split)
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
        lit = (bgo + rno) > 0
        tgt = bgo > rno
        return (torch.from_numpy(onf).float(), torch.from_numpy(offf).float(),
                torch.from_numpy(tgt & lit).float(),
                torch.from_numpy(lit).float(),
                torch.from_numpy(bgo), torch.from_numpy(rno))


def lit_bce(logits, tgt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader):
    sr, nr, n = np.zeros(len(TAUS)), np.zeros(len(TAUS)), 0
    for on, off, tgt, lit, bg, rn in loader:
        p = torch.sigmoid(model(on.to(DEV), x_off=off.to(DEV)))
        bg, rn = bg.to(DEV), rn.to(DEV)
        for b in range(on.shape[0]):
            nb, nrr = float(bg[b].sum()), float(rn[b].sum())
            if nb < 50 or nrr < 50:
                continue
            for j, t in enumerate(TAUS):
                keep = p[b] > t
                sr[j] += float((bg[b] * keep).sum()) / nb
                nr[j] += float((rn[b] * ~keep).sum()) / nrr
            n += 1
    return 0.5 * (sr + nr) / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="theirs", choices=["theirs", "ours"])
    ap.add_argument("--tout", type=int, default=16)
    ap.add_argument("--tfront", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = (f"reale_{a.split}_o{a.tout}"
           + (f"_s{a.seed}" if a.seed else ""))

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    tr_ds = RealESet(a.split, "train", a.tfront, a.tout)
    va_ds = RealESet(a.split, "val", a.tfront, a.tout)
    te_ds = RealESet(a.split, "test", a.tfront, a.tout)
    tr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(va_ds, batch_size=a.batch, **dl)
    te = DataLoader(te_ds, batch_size=a.batch, **dl)

    m = ORSPNet3D(T=a.tfront, dilations=(1, 8, 32, 64), num_blocks=3,
                  use_off=True, out_chans=a.tout).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | train {len(tr_ds)} val {len(va_ds)} "
          f"test {len(te_ds)}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        tot = nb = 0
        for on, off, tgt, lit, bg, rn in tr:
            out = m(on.to(DEV, non_blocking=True),
                    x_off=off.to(DEV, non_blocking=True))
            loss = lit_bce(out, tgt.to(DEV, non_blocking=True),
                           lit.to(DEV, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()

        m.eval()
        da, _ = evaluate(m, va)
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
    da_te, n_te = evaluate(m, te)
    jt = int(np.argmin(np.abs(TAUS - best_tau)))
    test = float(da_te[jt])
    print(f"\n=== {tag} ===")
    print(f"  params {npar:,}")
    print(f"  TEST EVENT-DA {test:.4f}  ({n_te} frames, val tau {best_tau:.2f},"
          f" val {best:.4f})")
    print(f"  test best-tau {float(da_te.max()):.4f} (reference)")
    if a.split == "theirs":
        print("  reference: PRE-Mamba 0.7708 | our old ON-only target 0.7985")
    else:
        print("  reference: our old ON-only target -- EvORSP-3T 0.8170, "
              "2D control 0.7649")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "tau": best_tau, "test": test},
               f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "split": a.split, "tout": a.tout, "params": npar,
               "seed": a.seed, "tau": best_tau, "val": best, "test": test,
               "test_best": float(da_te.max()), "n_test": n_te},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
