"""Controlled SPAC protocol: does the temporal axis buy anything?

The only variable is T. Every arm shares the same data, the same splits, the
same body, the same loss, the same schedule. T=1 is the exact OR-union of the
T=16 planes, so the baseline is bit-identical to the current 2D input rather
than a separately-built approximation.

Splits are scene-disjoint and were fixed before any result was seen:
  train t1-t8 | val a1-a4 | test b1-b4

Protocol mirrors the KITTI one: train, select the threshold on val, report on
test. Loss is lit-masked BCE -- measured this session at 0.9244 test DA on
KITTI, tied with the balanced loss and the prior-threshold rule, and it leaves
the model calibrated.
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
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
from rsp_3d import ORSPNet3D

ROOT = f"{C.WORK / 'spac_t16'}"
TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)


class SpacSet(Dataset):
    def __init__(self, split, T):
        self.files = sorted(glob.glob(f"{ROOT}/{split}/*/*.npz"))
        self.T = T
        assert T_BUILD % T == 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            gt = np.unpackbits(d["gt"])[: R * R].reshape(R, R)
        # counts over the FULL T=16 build planes (permutation-invariant),
        # regardless of the training T
        cnt = np.stack([on.sum(0), off.sum(0)]).astype(np.float32) / T_BUILD
        # derive T by OR-merging adjacent sub-windows; T=1 is the exact union
        on = on.reshape(self.T, T_BUILD // self.T, R, R).max(1)
        off = off.reshape(self.T, T_BUILD // self.T, R, R).max(1)
        return (torch.from_numpy(on).float(), torch.from_numpy(off).float(),
                torch.from_numpy(cnt),
                torch.from_numpy(gt).float().unsqueeze(0))


def lit_bce(logits, gt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, use_off=False, use_counts=False):
    """DA = 1/2(SR + NR) over LIT pixels, swept over tau."""
    sr = np.zeros(len(TAUS))
    nr = np.zeros(len(TAUS))
    n = 0
    for on, off, cnt, gt in loader:
        on, gt = on.to(DEV), gt.to(DEV)
        kw = {}
        if use_off:
            kw["x_off"] = off.to(DEV)
        if use_counts:
            kw["x_cnt"] = cnt.to(DEV)
        p = torch.sigmoid(model(on, **kw))
        lit = on.amax(1, keepdim=True) > 0.5
        real = (gt > 0.5) & lit
        rain = lit & ~(gt > 0.5)
        for b in range(on.shape[0]):
            rs, ns = real[b].sum().item(), rain[b].sum().item()
            if rs == 0 or ns == 0:
                continue
            pv = p[b]
            for j, t in enumerate(TAUS):
                pr = pv > t
                sr[j] += ((pr & real[b]).sum().item()) / rs
                nr[j] += (ns - (pr & rain[b]).sum().item()) / ns
            n += 1
    da = 0.5 * (sr + nr) / max(n, 1)
    return da, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--n_t", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--swa", action="store_true",
                    help="average the last 10 epochs' weights; keep if better on val")
    ap.add_argument("--dilations", default="1,8,32,64")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = (f"T{args.T}" + (f"b{args.blocks}" if args.blocks != 4 else "")
           + ("off" if args.off else "") + ("cnt" if args.counts else "")
           + (f"_s{args.seed}" if args.seed else ""))

    dl = dict(num_workers=6, pin_memory=True, persistent_workers=True)
    tr = DataLoader(SpacSet("train", args.T), batch_size=args.batch, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(SpacSet("val", args.T), batch_size=args.batch, **dl)
    te = DataLoader(SpacSet("test", args.T), batch_size=args.batch, **dl)

    D = tuple(int(v) for v in args.dilations.split(","))
    m = ORSPNet3D(T=args.T, n_t=args.n_t, dilations=D, num_blocks=args.blocks,
                  use_off=args.off, use_counts=args.counts,
                  use_temporal=(args.T > 1)).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params  |  train {len(tr.dataset)} val {len(va.dataset)} "
          f"test {len(te.dataset)}  |  {len(tr)} steps/epoch", flush=True)

    # the frontend pools ENERGY over t, so it is invariant to a circular roll of
    # the sub-windows -- i.e. it structurally cannot read absolute clock phase,
    # which is the artefact that inflated every absolute-bin probe on this data.
    if args.T > 1:
        m.eval()
        with torch.no_grad():
            x = torch.rand(2, args.T, R, R, device=DEV).round()
            xo = torch.rand(2, args.T, R, R, device=DEV).round()
            xc = torch.rand(2, 2, R, R, device=DEV)
            sh = args.T // 2
            kw = {}
            if args.off:
                kw["x_off"] = xo
            if args.counts:
                kw["x_cnt"] = xc          # counts are per-window, roll-invariant
            kwr = dict(kw)
            if args.off:
                kwr["x_off"] = torch.roll(xo, sh, 1)
            a = m(x, **kw)
            b = m(torch.roll(x, sh, 1), **kwr)
            d = (a - b).abs().max().item()
        print(f"  phase-roll invariance check: max|delta| = {d:.2e} "
              f"({'PASS' if d < 1e-4 else 'FAIL'})", flush=True)
        m.train()

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = nb = 0
        for on, off, cnt, gt in tr:
            on, gt = on.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
            lit = (on.amax(1, keepdim=True) > 0.5).float()
            kw = {}
            if args.off:
                kw["x_off"] = off.to(DEV, non_blocking=True)
            if args.counts:
                kw["x_cnt"] = cnt.to(DEV, non_blocking=True)
            loss = lit_bce(m(on, **kw), (gt > 0.5).float(), lit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()

        # SWA rider: running average of the last 10 epochs' weights
        if args.swa and ep > args.epochs - 10:
            swa_n = getattr(main, "_swa_n", 0) + 1
            main._swa_n = swa_n
            sd = {k: v.detach().cpu().float() for k, v in m.state_dict().items()}
            if swa_n == 1:
                main._swa_sd = sd
            else:
                for k in sd:
                    main._swa_sd[k] += (sd[k] - main._swa_sd[k]) / swa_n

        m.eval()
        da, _ = evaluate(m, va, args.off, args.counts)
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

    swa_used = False
    if args.swa and getattr(main, "_swa_sd", None) is not None:
        ref = m.state_dict()
        swa_sd = {k: v.to(ref[k].dtype) for k, v in main._swa_sd.items()}
        m.load_state_dict(swa_sd)
        m.eval()
        da_swa, _ = evaluate(m, va, args.off, args.counts)
        j = int(np.argmax(da_swa))
        print(f"  SWA(last10) valDA {da_swa[j]:.4f} (tau {TAUS[j]:.2f})  "
              f"vs best single {best:.4f}", flush=True)
        if da_swa[j] > best:            # keep whichever wins on VAL only
            best, best_tau, best_sd = float(da_swa[j]), float(TAUS[j]), swa_sd
            swa_used = True

    m.load_state_dict(best_sd)
    m.eval()
    da_te, n_te = evaluate(m, te, args.off, args.counts)
    jt = int(np.round((best_tau - 0.05) / 0.05))
    jt = max(0, min(len(TAUS) - 1, jt))
    test_at_val_tau = float(da_te[jt])

    print(f"\n=== SPAC RESULT: {tag} ===")
    print(f"  params            {npar:,}")
    print(f"  val  best DA      {best:.4f}  (tau {best_tau:.2f})")
    print(f"  TEST DA @ val tau {test_at_val_tau:.4f}   ({n_te} frames)")
    print(f"  test best-tau DA  {da_te.max():.4f}  (optimistic, for reference)")
    print(f"  wall {(time.time()-t0)/60:.0f} min")

    torch.save({"state_dict": best_sd, "params": npar, "tag": tag,
                "val": best, "tau": best_tau, "test": test_at_val_tau},
               f"{TMP}/spac_{tag}.pt")
    json.dump({"tag": tag, "T": args.T, "params": npar, "val": best,
               "tau": best_tau, "test": test_at_val_tau,
               "test_best": float(da_te.max()), "blocks": args.blocks,
               "off": bool(args.off), "counts": bool(args.counts),
               "swa_used": swa_used, "seed": args.seed},
              open(f"{TMP}/spac_{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
