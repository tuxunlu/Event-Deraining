"""Self-supervised adaptation to unlabelled wild rain.

Fine-tunes the rig-trained model on wild scene5 against pseudo-labels derived
from FUTURE persistence and burstiness -- calibrated in pseudo_calibrate2.py at
precision 0.949 (scene) / 0.934 (rain) against rig ground truth.

WHY THIS IS NOT CIRCULAR. The supervision comes from the K=4 windows AFTER the
one being classified. The model sees only the current window plus past context,
so it is learning to PREDICT future persistence from present appearance. Past
persistence would be circular -- the trunk's context planes and the recurrence
columns already carry it.

THREE THINGS ARE REPORTED EVERY EPOCH, and they answer different questions:

  wild pseudo-BA   on HELD-OUT wild sequences. Selection criterion. Measures
                   whether the learned mapping generalises, not whether it
                   memorised the training frames.
  rig event-DA     TRUE labels, rig test split. REGRESSION GUARD. Adaptation
                   that wins on wild by destroying rig performance has learned
                   a domain quirk, not deraining.
  wild keep-rate   the un-adapted model keeps 95.2% of wild events against
                   ~70% on rig -- it barely fires. Movement toward rig-like
                   keep-rate is independent evidence of domain alignment,
                   since nothing in the loss targets it.

Collapse guard: if keep-rate leaves [0.25, 0.95] the epoch is flagged. A model
that keeps nothing scores well on pseudo-rain and is useless.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_full as RF
from rsp_3d import ORSPNet3D
from run_real_full import CacheSet, HeadV2, sample_at

WILD_CACHE = "/fs/nexus-scratch/tuxunlu/wild_cache"
TMP = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
R, T16 = 256, 16
KEEP_LO, KEEP_HI = 0.25, 0.95


class WildSet(Dataset):
    """Held-out split is by SEQUENCE, not by frame: neighbouring frames share
    events, so a frame-level split would leak across train/val."""

    def __init__(self, split, val_levels=("rain_1", "rain_10", "rain_20")):
        seqs = sorted(glob.glob(f"{WILD_CACHE}/rain_*"))
        pick = [s for s in seqs
                if (os.path.basename(s) in val_levels) == (split == "val")]
        self.files = sorted(f for s in pick for f in glob.glob(f"{s}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as d:
            on = np.unpackbits(d["on"])[: T16 * R * R].reshape(T16, R, R)
            off = np.unpackbits(d["off"])[: T16 * R * R].reshape(T16, R, R)
            tc = np.concatenate([d["tcols"], d["iti"], d["recur"]],
                                1).astype(np.float32)
            out = (torch.from_numpy(on.reshape(4, 4, R, R).max(1)).float(),
                   torch.from_numpy(off.reshape(4, 4, R, R).max(1)).float(),
                   torch.from_numpy(d["x"].astype(np.float32)),
                   torch.from_numpy(d["y"].astype(np.float32)),
                   torch.from_numpy(d["tn"].astype(np.float32)),
                   torch.from_numpy(d["patch"].astype(np.float32)),
                   torch.from_numpy(tc),
                   torch.from_numpy(d["pl"].astype(np.float32)),
                   torch.from_numpy(d["pm"].astype(np.float32)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="realfull_ours")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--replay", type=float, default=1.0,
                    help="weight on the rig true-label replay term; 0 disables")
    a = ap.parse_args()

    blob = torch.load(f"{TMP}/{a.init}.pt", map_location="cpu",
                      weights_only=False)
    trunk = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                      use_off=True, out_chans=1).to(DEV)
    trunk.load_state_dict(blob["trunk"])
    head = HeadV2(feat_dim=128).to(DEV)
    head.load_state_dict(blob["head"])
    tau = float(blob["tau"])
    feats = {}
    trunk.out_proj.register_forward_pre_hook(
        lambda m, i: feats.__setitem__("f", i[0]))
    for bi, blk in enumerate(trunk.blocks):
        blk.register_forward_hook(
            lambda m, i, o, bi=bi: feats.__setitem__(bi, o))

    def fwd(on, off, xs, ys, tn, patch, tc):
        lm = trunk(on.to(DEV), x_off=off.to(DEV))
        fm = torch.cat([feats["f"]] + [feats[i] for i in range(3)], 1)
        xs, ys, tn = xs.to(DEV), ys.to(DEV), tn.to(DEV)
        lv = sample_at(lm[:, None], xs, ys, tn)
        fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                       xs, ys, tn)
        return head(lv, fv, patch.to(DEV), tc.to(DEV), tn[..., None])[..., 0]

    dl = dict(num_workers=4, pin_memory=True)
    tr = DataLoader(WildSet("train"), batch_size=1, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(WildSet("val"), batch_size=1, **dl)
    rig = DataLoader(CacheSet("test", "ours"), batch_size=1, **dl)
    # REPLAY. One epoch of naive adaptation moved wild pseudo-BA 0.625 -> 0.886
    # but dropped rig DA 0.883 -> 0.793: catastrophic forgetting. Interleaving
    # TRUE-label rig batches forces the model to satisfy both domains, and
    # also acts as the experiment that separates forgetting from genuine
    # conflict -- if rig DA still falls with replay on, the pseudo-label task
    # and real deraining are not the same objective.
    rig_tr = DataLoader(CacheSet("train", "ours"), batch_size=1, shuffle=True,
                        drop_last=True, **dl)
    print(f"wild train {len(tr.dataset)} / val {len(va.dataset)} frames; "
          f"rig guard {len(rig.dataset)} frames", flush=True)

    opt = torch.optim.AdamW([{"params": trunk.parameters(), "lr": a.lr_trunk},
                             {"params": head.parameters(), "lr": a.lr_head}],
                            weight_decay=5e-3)

    @torch.no_grad()
    def evaluate():
        trunk.eval(); head.eval()
        # held-out wild: balanced accuracy against pseudo-labels + keep rate
        c = np.zeros(4)                       # tp, np, tn, nn
        kk = kn = 0.0
        for on, off, xs, ys, tn, pt, tc, pl, pm in va:
            pr = torch.sigmoid(fwd(on, off, xs, ys, tn, pt, tc))
            keep = (pr > tau).float()
            kk += float(keep.sum()); kn += keep.numel()
            pl, pm = pl.to(DEV), pm.to(DEV)
            s, r = (pl > 0.5) & (pm > 0.5), (pl < 0.5) & (pm > 0.5)
            c[0] += float((keep * s).sum()); c[1] += float(s.sum())
            c[2] += float(((1 - keep) * r).sum()); c[3] += float(r.sum())
        ba = 0.5 * (c[0] / max(c[1], 1) + c[2] / max(c[3], 1))
        # rig guard: TRUE labels
        das = []
        for (on, off, ex, xs, ys, tn, pt, tc, lab, inv_p,
             nb, nr, idx) in rig:
            pr = torch.sigmoid(fwd(on, off, xs, ys, tn, pt, tc))
            keep = (pr > tau).cpu().numpy()[0]
            y0, w = lab[0].numpy(), inv_p[0].numpy()
            sm, rm = y0 > 0.5, y0 < 0.5
            if w[sm].sum() > 0 and w[rm].sum() > 0:
                das.append(0.5 * ((keep[sm] * w[sm]).sum() / w[sm].sum()
                                  + ((~keep[rm]) * w[rm]).sum() / w[rm].sum()))
        trunk.train(); head.train()
        return ba, float(np.mean(das)) if das else float("nan"), kk / max(kn, 1)

    ba0, rig0, kr0 = evaluate()
    print(f"  before      wild pseudo-BA {ba0:.4f} | rig DA {rig0:.4f} | "
          f"wild keep {kr0:.3f}", flush=True)

    best = -1.0
    rig_iter = iter(rig_tr)
    for ep in range(1, a.epochs + 1):
        tot = nb = 0
        for on, off, xs, ys, tn, pt, tc, pl, pm in tr:
            logit = fwd(on, off, xs, ys, tn, pt, tc)
            pl, pm = pl.to(DEV), pm.to(DEV)
            bce = F.binary_cross_entropy_with_logits(logit, pl,
                                                     reduction="none")
            loss = (bce * pm).sum() / pm.sum().clamp(min=1.0)
            if a.replay > 0:
                try:
                    rb = next(rig_iter)
                except StopIteration:
                    rig_iter = iter(rig_tr)
                    rb = next(rig_iter)
                (ron, roff, _rex, rxs, rys, rtn, rpt, rtc,
                 rlab, rinv, _nb, _nr, _i) = rb
                rlogit = fwd(ron, roff, rxs, rys, rtn, rpt, rtc)
                rw = rinv.to(DEV)
                rbce = F.binary_cross_entropy_with_logits(
                    rlogit, rlab.to(DEV), reduction="none")
                loss = loss + a.replay * (
                    (rbce * rw).sum() / rw.sum().clamp(min=1e-6))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(trunk.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        ba, rigda, kr = evaluate()
        flag = "" if KEEP_LO <= kr <= KEEP_HI else "  <-- COLLAPSE RISK"
        star = ""
        if ba > best and KEEP_LO <= kr <= KEEP_HI:
            best = ba
            torch.save({"trunk": trunk.state_dict(), "head": head.state_dict(),
                        "tau": tau, "wild_ba": ba, "rig_da": rigda,
                        "keep": kr}, f"{TMP}/wildadapt.pt")
            star = "  *"
        print(f"  ep {ep:2d}/{a.epochs}  loss {tot/max(nb,1):.4f} | "
              f"wild pseudo-BA {ba:.4f} | rig DA {rigda:.4f} "
              f"({rigda-rig0:+.4f}) | keep {kr:.3f}{star}{flag}", flush=True)

    print(f"\n  baseline: wild pseudo-BA {ba0:.4f}, rig DA {rig0:.4f}, "
          f"keep {kr0:.3f}")
    print("  adaptation is only a win if wild pseudo-BA rises while rig DA")
    print("  holds. A wild gain paid for with rig loss is a domain quirk.")


if __name__ == "__main__":
    main()
