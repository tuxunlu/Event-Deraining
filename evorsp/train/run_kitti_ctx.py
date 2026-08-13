"""Tier-1 input experiments: temporal resolution, inter-window context, counts.

Architecture is settled -- five models from 28K to 265K params all land in
0.9150-0.9218, so capacity is not the constraint. These three arms attack the
places where the INPUT demonstrably lacks information the target requires.

  --tfront 16   MATCHED TEMPORAL RESOLUTION. Every run so far fed T_front=4
                input planes while asking for T_out=16 output decisions: the
                model had to resolve 16 time bins from 4 bins of evidence.
                TemporalFrontend emits 1+n_t+1 channels regardless of T, so
                raising T costs only the 1-D FFT -- the body is unchanged.

  --ctx K       INTER-WINDOW CONTEXT. Our model sees ONE 100 ms window;
                PRE-Mamba gathers five (~500 ms) and its own ablation moves
                DA 0.8268 -> 0.9015 -> 0.9330 for 3 -> 5 -> 8 windows, a larger
                swing than either of its named modules. Our own physics probe
                says why this should work: rain is temporally brief, scene
                structure persistent. Adds 2K channels (ON-union, OFF-union of
                each preceding window). Frames are clamped at sequence start.

  --counts      COUNT PLANES. The input is 1-bit occupancy while the target is
                count-based (bg > rn). Adds log1p(events per cell) at T_front
                resolution, 1 channel. Label-free: the total count needs no
                ground truth, only the sum of the two class counts.

Everything else -- trunk, target, loss, schedule, metric -- matches
run_kitti_e.py exactly, so any difference is attributable to the input.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rsp_3d import ORSPNet3D
from rsp_guard3d import ORSPNet3DGuard
import density_aug

ROOT = f"{C.KITTI_PACK}"
TMP = f"{C.CKPT}"
DEV = "cuda"
T_BUILD, R = 16, 256
TAUS = np.linspace(0.05, 0.95, 19)
EPS = 1e-6


def _planes(path):
    with np.load(path) as d:
        on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
        off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
    return on, off


class KittiCtxSet(Dataset):
    """Windows, optionally augmented along the DENSITY axis.

    The entire val-test gap on this benchmark is intensity extrapolation: tau
    is picked on val {20,80} mm and reported on test {50,150} mm, 50 mm sits
    inside the trained range and scores at val level, and 150 mm -- heavier
    than anything seen in training -- drops 0.039. Only 17% of that is
    recoverable by ANY threshold rule, so the remainder is separability and
    has to be attacked during training.

    Augmentation applies to TRAIN ONLY. Passing it to val or test would move
    the thing being measured.
    """

    def __init__(self, split, t_front, t_out, ctx, counts, ctxres=1, aug="",
                 mix_p=0.5, drop_q=0.3, seed=0):
        self.files = sorted(glob.glob(f"{ROOT}/{split}/*/*.npz"))
        self.mm = [f.split("/")[-2] for f in self.files]
        self.tf, self.to, self.ctx, self.counts = t_front, t_out, ctx, counts
        assert T_BUILD % ctxres == 0, "ctxres must divide T_BUILD"
        self.ctxres = ctxres
        self.aug = set(a for a in aug.split("+") if a and a != "none")
        self.mix_p, self.drop_q = mix_p, drop_q
        self.rng = np.random.default_rng(seed + 9973)

    def _partner(self):
        j = int(self.rng.integers(len(self.files)))
        on2, off2 = _planes(self.files[j])
        with np.load(self.files[j]) as d:
            bg2 = d["bg"].reshape(T_BUILD, R, R).astype(np.float32)
            rn2 = d["rn"].reshape(T_BUILD, R, R).astype(np.float32)
        return j, on2, off2, bg2, rn2

    def _augment(self, on, off, bg, rn):
        return density_aug.augment(self.rng, on, off, bg, rn, self.aug,
                                   self.mix_p, self.drop_q, self._partner)

    def __len__(self):
        return len(self.files)

    def _prev(self, path, k):
        """Path of the k-th preceding window in the same sequence, clamped."""
        d, b = os.path.split(path)
        idx = max(int(b.split(".")[0]) - k, 0)
        p = f"{d}/{idx:010d}.npz"
        return p if os.path.exists(p) else path

    def __getitem__(self, i):
        f = self.files[i]
        on, off = _planes(f)
        with np.load(f) as d:
            bg = d["bg"].reshape(T_BUILD, R, R).astype(np.float32)
            rn = d["rn"].reshape(T_BUILD, R, R).astype(np.float32)
        mixed = None
        if self.aug:
            on, off, bg, rn, mixed = self._augment(on, off, bg, rn)
        onf = on.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)
        offf = off.reshape(self.tf, T_BUILD // self.tf, R, R).max(1)

        extra = []
        cr = self.ctxres
        for k in range(1, self.ctx + 1):
            pon, poff = _planes(self._prev(f, k))
            # keep the context consistent with whatever was done to the current
            # window: a superposed frame gets superposed history, a flipped
            # frame gets flipped history. Otherwise the context contradicts the
            # input and the model learns to distrust it.
            pon, poff = density_aug.augment_context(
                pon, poff, mixed,
                lambda j: _planes(self._prev(self.files[j], k)))
            if cr == 1:
                # the original: collapse the whole preceding window to ONE
                # occupancy plane per polarity. Cheap, and a severe compression
                # -- everything about WHEN a pixel fired inside that window is
                # discarded before the trunk ever sees it.
                extra.append(pon.max(0)[None].astype(np.float32))
                extra.append(poff.max(0)[None].astype(np.float32))
            else:
                # keep cr time slices instead, so the preceding window arrives
                # with the same temporal resolution the CURRENT window gets.
                extra.append(pon.reshape(cr, T_BUILD // cr, R, R)
                             .max(1).astype(np.float32))
                extra.append(poff.reshape(cr, T_BUILD // cr, R, R)
                             .max(1).astype(np.float32))
        if self.counts:
            cnt = (bg + rn).reshape(self.tf, T_BUILD // self.tf, R, R).sum(1)
            extra.append(np.log1p(cnt.sum(0))[None] / 4.0)      # 1 ch, ~[0,1]
        ex = (np.concatenate(extra, 0) if extra
              else np.zeros((0, R, R), np.float32))

        k = T_BUILD // self.to
        bgo = bg.reshape(self.to, k, R, R).sum(1)
        rno = rn.reshape(self.to, k, R, R).sum(1)
        lit = (bgo + rno) > 0
        tgt = bgo > rno
        return (torch.from_numpy(onf).float(), torch.from_numpy(offf).float(),
                torch.from_numpy(ex).float(),
                torch.from_numpy(tgt & lit).float(),
                torch.from_numpy(lit).float(),
                torch.from_numpy(bgo), torch.from_numpy(rno), i)


def lit_bce(logits, tgt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, ds, n_extra):
    acc = {}
    sp = []
    for on, off, ex, tgt, lit, bg, rn, idx in loader:
        kw = {"x_extra": ex.to(DEV)} if n_extra else {}
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
            cnt = bg[b] + rn[b]
            tau = float((p[b] * cnt).sum() / cnt.sum().clamp(min=EPS))
            keep = p[b] > tau
            sp.append(0.5 * (float((bg[b] * keep).sum()) / nb
                             + float((rn[b] * ~keep).sum()) / nr))
    per = {mm: 0.5 * (s + n) / max(c, 1) for mm, (s, n, c) in acc.items()}
    return per, (float(np.mean(sp)) if sp else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfront", type=int, default=4)
    ap.add_argument("--tout", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--ctxres", type=int, default=1,
                    help="time slices kept per CONTEXT window. 1 (default) is "
                         "the original max-pool-to-one-plane. --ctx 4 was null "
                         "(0.9287 vs 0.9291), but that added more COMPRESSED "
                         "windows; it never tested whether the compression "
                         "itself is what costs. PRE-Mamba consumes 5 windows at "
                         "full temporal resolution, so this is the one input "
                         "lever left untested.")
    ap.add_argument("--guard", default="",
                    help="alpha,bound for the retention-guaranteed trunk "
                         "(rsp_guard3d). logit = alpha*res + C*tanh(z/C), so a "
                         "fully persistent cell scores at least alpha-C no "
                         "matter what the weights are. Verified under attack at "
                         "exactly the floor while every alternative collapsed "
                         "to chance. This measures what the guarantee COSTS.")
    ap.add_argument("--dil", default="1,8,32,64",
                    help="gate dilation ladder. A 3x3 at dilation d reaches "
                         "+/-d px on the 256 grid, so 64 covers a quarter of "
                         "the frame and 128 covers all of it. Measured FREE: "
                         "same 28,719 params and 6.49-6.50 ms for every ladder, "
                         "so any accuracy gain here is pure profit.")
    ap.add_argument("--blocks", type=int, default=3,
                    help="depth is the ONLY lever that moves latency: "
                         "dim/probe/n_rad are free (4.23-4.27 ms across "
                         "12K-29K params), 3->2 blocks is 4.26->3.15 ms")
    ap.add_argument("--aug", default="",
                    help="TRAIN-ONLY density augmentation, '+'-joined: "
                         "mix (superpose another window -> heavier rain, "
                         "counts add so labels stay exact), drop (thin events "
                         "-> lighter rain), hflip (control). Phase 1 showed "
                         "only 17%% of the 150mm deficit is reachable by any "
                         "threshold, so the rest must be attacked here.")
    ap.add_argument("--mix-p", type=float, default=0.5)
    ap.add_argument("--drop-q", type=float, default=0.3)
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tag = (f"ctx_f{a.tfront}o{a.tout}_c{a.ctx}"
           + (f"r{a.ctxres}" if a.ctxres != 1 else "")
           + (f"_b{a.blocks}" if a.blocks != 3 else "")
           + ("" if a.dil == "1,8,32,64" else "_d" + a.dil.replace(',', '-'))
           + ("" if not a.guard else "_g" + a.guard.replace(",", "-"))
           + ("" if not a.aug else "_a" + a.aug.replace("+", "-"))
           + ("_cnt" if a.counts else "") + (f"_s{a.seed}" if a.seed else ""))
    n_extra = 2 * a.ctx * a.ctxres + (1 if a.counts else 0)

    dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    # augmentation on TRAIN ONLY -- val and test define the measurement
    ds = {s: KittiCtxSet(s, a.tfront, a.tout, a.ctx, a.counts, a.ctxres,
                         aug=(a.aug if s == "train" else ""),
                         mix_p=a.mix_p, drop_q=a.drop_q, seed=a.seed)
          for s in ("train", "val", "test")}
    tr = DataLoader(ds["train"], batch_size=a.batch, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(ds["val"], batch_size=a.batch, **dl)
    te = DataLoader(ds["test"], batch_size=a.batch, **dl)

    _dil = tuple(int(v) for v in a.dil.split(','))
    _cls = ORSPNet3D
    _gkw = {}
    if a.guard:
        _al, _bd = (float(v) for v in a.guard.split(","))
        _cls, _gkw = ORSPNet3DGuard, dict(alpha=_al, bound=_bd)
    m = _cls(T=a.tfront, dilations=_dil, num_blocks=a.blocks,
                  use_off=True, out_chans=a.tout, n_extra=n_extra, **_gkw).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | T_front {a.tfront} -> T_out {a.tout} | "
          f"ctx {a.ctx} windows x {a.ctxres} slices | counts {a.counts} | "
          f"n_extra {n_extra}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                     eta_min=1e-6)
    best, best_tau, best_sd = -1.0, 0.5, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        m.train()
        tot = nb = 0
        for on, off, ex, tgt, lit, bg, rn, _ in tr:
            kw = {"x_extra": ex.to(DEV, non_blocking=True)} if n_extra else {}
            out = m(on.to(DEV, non_blocking=True),
                    x_off=off.to(DEV, non_blocking=True), **kw)
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
        per, _ = evaluate(m, va, ds["val"], n_extra)
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
    per_te, sp = evaluate(m, te, ds["test"], n_extra)
    jt = int(np.argmin(np.abs(TAUS - best_tau)))
    test = float(np.mean([v[jt] for v in per_te.values()]))
    print(f"\n=== {tag} ===")
    print(f"  params {npar:,}")
    for mm in sorted(per_te, key=lambda s: int(s[:-2])):
        print(f"  {mm:>6s} event-DA @ val tau: {per_te[mm][jt]:.4f}")
    print(f"  TEST MEAN EVENT-DA {test:.4f}  (val tau {best_tau:.2f})")
    print(f"  self-prior tau      {sp:.4f}")
    print(f"  baseline f4o16 c0: 0.9196 protocol / 0.9245 self-prior; "
          f"ceiling 0.9807")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "tau": best_tau, "test": test},
               f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "tfront": a.tfront, "tout": a.tout, "ctx": a.ctx,
               "counts": a.counts, "params": npar, "seed": a.seed,
               "tau": best_tau, "val": best, "test": test,
               "test_selfprior": sp,
               "per_test": {k: float(v[jt]) for k, v in per_te.items()}},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
