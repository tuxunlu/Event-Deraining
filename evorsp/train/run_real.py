"""Train and evaluate on REAL EVK4 rain, scene-disjoint.

Split: train scenes 1-2, val scene 3, test scene 4 -- fully scene-disjoint,
fixed before any result. All four rain levels appear in every split. Protocol
mirrors the SPAC/KITTI harnesses: lit-masked BCE, 50 epochs, AdamW 5e-4/5e-3,
cosine, grad clip 1.0; threshold selected on val, reported on test.

Models:
  --model evorsp : EvORSP-3T (T=4 temporal + OFF, 3 blocks)
  --model orsp2d : the 2D control -- same body family, 4 blocks, anchor input
                   (ORSPNet+dil equivalent), so the comparison isolates the
                   input representation on real data exactly as it did on
                   KITTI and SPAC.
"""
import argparse
import glob
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

ROOT = "/fs/nexus-scratch/tuxunlu/real_t16"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)
SPLIT = {"train": ("scene1", "scene2"), "val": ("scene3",), "test": ("scene4",)}


class RealSet(Dataset):
    def __init__(self, split, T):
        self.files = []
        for sc in SPLIT[split]:
            self.files += sorted(glob.glob(f"{ROOT}/{sc}/rain_*/*.npz"))
        self.T = T

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
                torch.from_numpy(gt).float().unsqueeze(0))


def lit_bce(logits, gt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, use_off):
    sr = np.zeros(len(TAUS))
    nr = np.zeros(len(TAUS))
    n = 0
    for on, off, gt in loader:
        on, gt = on.to(DEV), gt.to(DEV)
        p = torch.sigmoid(model(on, x_off=off.to(DEV)) if use_off else
                          model(on.amax(1, keepdim=True)))
        lit = on.amax(1, keepdim=True) > 0.5
        real = (gt > 0.5) & lit
        rain = lit & ~(gt > 0.5)
        for b in range(on.shape[0]):
            rs, ns = int(real[b].sum()), int(rain[b].sum())
            if rs < 50 or ns < 50:
                continue
            pv = p[b]
            for j, t in enumerate(TAUS):
                pr = pv > t
                sr[j] += ((pr & real[b]).sum().item()) / rs
                nr[j] += (ns - (pr & rain[b]).sum().item()) / ns
            n += 1
    return 0.5 * (sr + nr) / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["evorsp", "orsp2d"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    use_off = args.model == "evorsp"
    tag = f"real_{args.model}" + (f"_s{args.seed}" if args.seed else "")
    T = 4 if use_off else 1

    dl = dict(num_workers=3, pin_memory=True, persistent_workers=True)
    tr = DataLoader(RealSet("train", T), batch_size=args.batch, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(RealSet("val", T), batch_size=args.batch, **dl)
    te = DataLoader(RealSet("test", T), batch_size=args.batch, **dl)

    D = (1, 8, 32, 64)
    if use_off:
        m = ORSPNet3D(T=4, dilations=D, num_blocks=3, use_off=True).to(DEV)
    else:
        m = ORSPNet3D(T=1, dilations=D, num_blocks=4, use_temporal=False).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | train {len(tr.dataset)} val {len(va.dataset)} "
          f"test {len(te.dataset)}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                     eta_min=1e-6)
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = nb = 0
        for on, off, gt in tr:
            on, gt = on.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
            lit = (on.amax(1, keepdim=True) > 0.5).float()
            out = (m(on, x_off=off.to(DEV, non_blocking=True)) if use_off
                   else m(on.amax(1, keepdim=True)))
            loss = lit_bce(out, (gt > 0.5).float(), lit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()

        m.eval()
        da, _ = evaluate(m, va, use_off)
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
    da_te, n_te = evaluate(m, te, use_off)
    jt = max(0, min(len(TAUS) - 1, int(np.round((best_tau - 0.05) / 0.05))))
    print(f"\n=== REAL RESULT: {tag} ===")
    print(f"  params {npar:,}")
    print(f"  TEST DA @ val tau {float(da_te[jt]):.4f}   ({n_te} frames, scene4)")
    print(f"  test best-tau     {float(da_te.max()):.4f}  (reference)")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "params": npar, "tag": tag,
                "tau": best_tau, "test": float(da_te[jt])}, f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "model": args.model, "params": npar, "seed": args.seed,
               "val": best, "tau": best_tau, "test": float(da_te[jt]),
               "test_best": float(da_te.max())},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
