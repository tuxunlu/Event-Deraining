"""The fairness harness, on any of the three datasets -- not just KITTI.

run_kitti_fair.py established the campaign's central methodological claim: give
every architecture the SAME 10-plane temporal+polarity input, the SAME
count-majority target and the SAME T_out readout, and most of the reported
spread between "architectures" disappears. On KITTI the result was that both of
our headline components transfer to every backbone, and that FourierMamba beats
our trunk on matched input (0.9284 vs 0.9215) rather than losing to it.

That claim rested on ONE synthetic dataset. StreakNet, DFFN and FourierMamba
had never been run on SPAC or on real EVK4 at all, so "matched input changes the
ranking" was a single-dataset finding being quoted as a general one. This script
removes that limitation: same harness, same metric, same everything, with the
dataset chosen by --data.

WHAT DIFFERS BETWEEN THE THREE, and nothing else:

  kitti  packs at KITTI_PACK, pre-split train/val/test, groups are rain
         intensities ("20mm", "150mm", ...). val {20,80}, test {50,150}.
  spac   packs at SPAC_PACK, identical layout, groups are sequences
         ("t1_Rain_01"). NOTE the seed spread here reaches 0.0304 -- twelve
         times KITTI's +-0.0026 -- so a single-seed SPAC arm says nothing.
         Run at least three seeds before comparing anything.
  real   packs at REAL_PACK, laid out by scene rather than split, so the split
         is a CHOICE: --split theirs reproduces PRE-Mamba's own per-scene
         partition, --split ours holds out whole scenes (train 1-2, val 3,
         test 4) and is the harder, scene-disjoint number.

The metric is unchanged from run_kitti_fair.py: exact event-level DA from
per-cell background and rain counts, SR = kept background / all background,
NR = dropped rain / all rain, per frame, frame-averaged, one global tau
selected on val and reported on test. Identical accounting to PRE-Mamba's
SemSegTester.
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
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bodies_e import FrontendBody

TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)

# PRE-Mamba's own per-scene partition of the real rig data, and our
# scene-disjoint alternative. Copied verbatim from run_real_ctx.py so the two
# scripts cannot drift apart.
THEIRS = {"scene1": {"train": [1], "val": [4], "test": [2, 3]},
          "scene2": {"train": [3, 9], "val": [4, 6], "test": [1, 2, 5, 10]},
          "scene3": {"train": [4, 10], "val": [6, 8], "test": [2, 9]},
          "scene4": {"train": [1, 3], "val": [4, 9], "test": [2, 6, 13]}}
OURS = {"train": ("scene1", "scene2"), "val": ("scene3",), "test": ("scene4",)}


def _files_and_groups(data, split, real_split):
    """(file list, per-file group label) for one split of one dataset.

    The group label is what the event-DA is averaged over: rain intensity on
    KITTI, sequence on SPAC, scene/recording on real. Reporting per group is
    what keeps a dataset with one dominant condition from hiding a failure in
    the others.
    """
    if data == "real":
        root = f"{C.REAL_PACK}"
        files = []
        if real_split == "theirs":
            for sc, d in THEIRS.items():
                for k in d[split]:
                    files += glob.glob(f"{root}/{sc}/rain_{k}/*.npz")
        else:
            for sc in OURS[split]:
                files += glob.glob(f"{root}/{sc}/rain_*/*.npz")
        files = sorted(files)
        groups = [f.split("/")[-3] + "/" + f.split("/")[-2] for f in files]
    else:
        root = f"{C.KITTI_PACK}" if data == "kitti" else f"{C.SPAC_PACK}"
        files = sorted(glob.glob(f"{root}/{split}/*/*.npz"))
        groups = [f.split("/")[-2] for f in files]
    return files, groups


def _gkey(s):
    """Natural sort for group labels: '20mm' < '150mm', 't2_Rain_01' sanely."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]


class FairSet(Dataset):
    def __init__(self, data, split, t_front, t_out, target, ctx=0,
                 real_split="ours"):
        self.files, self.mm = _files_and_groups(data, split, real_split)
        self.tf, self.to, self.target, self.ctx = t_front, t_out, target, ctx
        assert T_BUILD % t_front == 0 and T_BUILD % t_out == 0
        assert self.files, f"no packs found for {data}/{split}"

    def __len__(self):
        return len(self.files)

    def _prev(self, path, k):
        d, b = os.path.split(path)
        j = max(int(b.split(".")[0]) - k, 0)
        p = f"{d}/{j:010d}.npz"
        return p if os.path.exists(p) else path

    def __getitem__(self, i):
        f = self.files[i]
        with np.load(f) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            bg = d["bg"].reshape(T_BUILD, R, R).astype(np.float32)
            rn = d["rn"].reshape(T_BUILD, R, R).astype(np.float32)
        onf = on.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)
        offf = off.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)

        ex = []
        for k in range(1, self.ctx + 1):
            with np.load(self._prev(f, k)) as dp:
                pon = np.unpackbits(dp["on"])[: T_BUILD * R * R] \
                    .reshape(T_BUILD, R, R)
                poff = np.unpackbits(dp["off"])[: T_BUILD * R * R] \
                    .reshape(T_BUILD, R, R)
            ex.append(pon.max(0)[None].astype(np.float32))
            ex.append(poff.max(0)[None].astype(np.float32))
        exa = np.concatenate(ex, 0) if ex else np.zeros((0, R, R), np.float32)

        k = T_BUILD // self.to
        bgo = bg.reshape(self.to, k, R, R).sum(1)
        rno = rn.reshape(self.to, k, R, R).sum(1)
        lit = (bgo + rno) > 0
        tgt = (bgo > rno) if self.target == "maj" else (bgo > 0)
        return (torch.from_numpy(onf).float(), torch.from_numpy(offf).float(),
                torch.from_numpy(exa).float(),
                torch.from_numpy(tgt & lit).float(),
                torch.from_numpy(lit).float(),
                torch.from_numpy(bgo), torch.from_numpy(rno), i)


def lit_bce(logits, tgt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, ds):
    """Exact event-level DA at every tau, per group."""
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
    ap.add_argument("--data", default="kitti", choices=["kitti", "spac", "real"])
    ap.add_argument("--split", default="ours", choices=["ours", "theirs"],
                    help="real only: 'theirs' is PRE-Mamba's per-scene "
                         "partition, 'ours' holds out whole scenes and is the "
                         "harder, scene-disjoint number")
    ap.add_argument("--body", required=True,
                    choices=["dffn", "orsp", "streaknet", "fmamba"])
    ap.add_argument("--tout", type=int, default=16)
    ap.add_argument("--tfront", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=0,
                    help="inter-window context planes. Context is an INPUT "
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
                         "has fp32 exponent range and cannot overflow. Always "
                         "pass this for --body fmamba.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dtag = a.data + (f"-{a.split}" if a.data == "real" else "")
    tag = (f"fair{'' if a.data == 'kitti' else '_' + dtag}_{a.body}_o{a.tout}"
           + (f"_c{a.ctx}" if a.ctx else "") + (f"_s{a.seed}" if a.seed else ""))

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    mk = lambda s: FairSet(a.data, s, a.tfront, a.tout, a.target, a.ctx, a.split)
    tr_ds, va_ds, te_ds = mk("train"), mk("val"), mk("test")
    tr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(va_ds, batch_size=a.batch, **dl)
    te = DataLoader(te_ds, batch_size=a.batch, **dl)

    m = FrontendBody(a.body, T=a.tfront, t_out=a.tout,
                     n_extra=2 * a.ctx).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | data {dtag} | train {len(tr_ds)} "
          f"val {len(va_ds)} test {len(te_ds)}", flush=True)
    if a.data == "spac" and a.seed == 0:
        print("  NOTE SPAC seed spread reaches 0.0304 (12x KITTI's). Run at "
              "least 3 seeds before comparing this to anything.", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp and not a.bf16)
    _adt = torch.bfloat16 if a.bf16 else torch.float16
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        tot = nb = nskip = 0
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
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in m.state_dict().items()}
            star = "  *"
        if ep % 5 == 0 or ep == 1 or star:
            sk = f" skip {nskip}" if nskip else ""
            print(f"  ep {ep:3d}/{a.epochs}  train {tot/max(nb,1):.4f}{sk}  "
                  f"val tau {TAUS[j]:.2f} eventDA {da[j]:.4f}{star}  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    per_te = evaluate(m, te, te_ds)
    jt = int(np.argmin(np.abs(TAUS - best_tau)))
    test = float(np.mean([v[jt] for v in per_te.values()]))
    print(f"\n=== {tag} ===")
    print(f"  params {npar:,}   data {dtag}")
    for mm in sorted(per_te, key=_gkey):
        print(f"  {mm:>16s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f}   (val tau {best_tau:.2f}, "
          f"val {best:.4f})")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "tau": best_tau, "test": test},
               f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "data": dtag, "body": a.body, "params": npar,
               "tout": a.tout, "ctx": a.ctx, "seed": a.seed, "tau": best_tau,
               "val": best, "test": test,
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
