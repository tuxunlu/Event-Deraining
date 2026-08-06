"""Train EvORSP-3T on PRE-Mamba's OWN EVK4 split, evaluate with THEIR metric.

Split = the scene_split dict from PRE-Mamba's EVK4MultiScansDataset (within-scene,
rain-level-disjoint). Training mirrors run_real.py exactly (lit-masked BCE,
AdamW 5e-4/5e-3, cosine, 50 epochs, best epoch by val pixel-DA). The reported
number is EVENT-level: per-frame SR/NR/DA over labeled events (their
SemSegTester formula), tau selected on their val split by event-DA, frame-mean
on their test split. DA is invariant to label orientation, so the headline
number is comparable regardless of the 0/1 naming question.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

ROOT = "/fs/nexus-scratch/tuxunlu/real_t16"
S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
DEV = "cuda"
T_BUILD, T, R = 16, 4, 256
W, H = 1280, 720
TAUS = np.linspace(0.05, 0.95, 19)

SCENE_SPLIT = {
    "scene1": {"train": [1], "val": [4], "test": [2, 3]},
    "scene2": {"train": [3, 9], "val": [4, 6], "test": [1, 2, 5, 10]},
    "scene3": {"train": [4, 10], "val": [6, 8], "test": [2, 9]},
    "scene4": {"train": [1, 3], "val": [4, 9], "test": [2, 6, 13]},
}


def files_of(split):
    out = []
    for sc, d in SCENE_SPLIT.items():
        for k in d[split]:
            out += sorted(glob.glob(f"{ROOT}/{sc}/rain_{k}/*.npz"))
    return out


class RealSetTheirs(Dataset):
    def __init__(self, split):
        self.files = files_of(split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with np.load(self.files[i]) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            gt = np.unpackbits(d["gt"])[: R * R].reshape(R, R)
        on = on.reshape(T, T_BUILD // T, R, R).max(1)
        off = off.reshape(T, T_BUILD // T, R, R).max(1)
        return (torch.from_numpy(on).float(), torch.from_numpy(off).float(),
                torch.from_numpy(gt).float().unsqueeze(0))


def lit_bce(logits, gt, lit):
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_px(model, loader):
    sr, nr, n = np.zeros(len(TAUS)), np.zeros(len(TAUS)), 0
    for on, off, gt in loader:
        on, gt = on.to(DEV), gt.to(DEV)
        p = torch.sigmoid(model(on, x_off=off.to(DEV)))
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


@torch.no_grad()
def event_stats(model, split):
    """Per-frame (rec1, rec0) at every tau; label 1/0 recalls, DA = mean."""
    rows = []
    for pk in files_of(split):
        parts = pk.split("/")
        sc, lvl, base = parts[-3], parts[-2], parts[-1]
        mp = f"{S}/{sc}/merge_data/{lvl}/{base}"
        lp = f"{S}/{sc}/labels/labels_{lvl}/labels_{base}".replace(".npz", ".npy")
        try:
            with np.load(mp) as d:
                x, y = d["x"], d["y"]
            lab = np.load(lp)
        except Exception:
            continue
        if len(lab) != len(x):
            continue
        with np.load(pk) as d:
            on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
            off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
        on = torch.from_numpy(on.reshape(T, T_BUILD // T, R, R).max(1)
                              ).float()[None].to(DEV)
        off = torch.from_numpy(off.reshape(T, T_BUILD // T, R, R).max(1)
                               ).float()[None].to(DEV)
        pmap = torch.sigmoid(model(on, x_off=off))[0, 0].cpu().numpy()
        sx = (x.astype(np.int64) * R) // W
        sy = (y.astype(np.int64) * R) // H
        pev = pmap[sy, sx]
        is1 = lab == 1
        n1, n0 = max(int(is1.sum()), 1), max(int((~is1).sum()), 1)
        row = []
        for tau in TAUS:
            keep = pev > tau
            row.append((int((keep & is1).sum()) / n1,
                        int((~keep & ~is1).sum()) / n0))
        rows.append(row)
    return np.array(rows)                                     # [n, 19, 2]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    tag = f"real_evorsp_theirsplit" + (f"_s{seed}" if seed else "")

    dl = dict(num_workers=3, pin_memory=True, persistent_workers=True)
    tr = DataLoader(RealSetTheirs("train"), batch_size=8, shuffle=True,
                    drop_last=True, **dl)
    va = DataLoader(RealSetTheirs("val"), batch_size=8, **dl)

    m = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                  use_off=True).to(DEV)
    npar = sum(q.numel() for q in m.parameters())
    print(f"{tag}: {npar:,} params | train {len(tr.dataset)} "
          f"val {len(va.dataset)} test {len(files_of('test'))}", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)
    best, best_sd = -1.0, None
    t0 = time.time()
    for ep in range(1, 51):
        m.train()
        tot = nb = 0
        for on, off, gt in tr:
            on, gt = on.to(DEV, non_blocking=True), gt.to(DEV, non_blocking=True)
            lit = (on.amax(1, keepdim=True) > 0.5).float()
            out = m(on, x_off=off.to(DEV, non_blocking=True))
            loss = lit_bce(out, (gt > 0.5).float(), lit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()
        m.eval()
        da, _ = evaluate_px(m, va)
        j = int(np.argmax(da))
        star = ""
        if da[j] > best:
            best = float(da[j])
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in m.state_dict().items()}
            star = "  *"
        if ep % 5 == 0 or ep == 1 or star:
            print(f"  ep {ep:3d}/50  train {tot/max(nb,1):.4f}  "
                  f"val pxDA {da[j]:.4f}{star}  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    ev_val = event_stats(m, "val")
    vda = ev_val.mean(0)                                       # [19,2]
    j = int(np.argmax(0.5 * (vda[:, 0] + vda[:, 1])))
    ev_te = event_stats(m, "test")
    a = ev_te[:, j, :]
    da_te = 0.5 * (a[:, 0] + a[:, 1]).mean()
    print(f"\n=== THEIR-SPLIT RESULT: {tag} ===")
    print(f"  params {npar:,} | best val pxDA {best:.4f}")
    print(f"  EVENT metric, their split: tau {TAUS[j]:.2f}  "
          f"rec(lab=1) {a[:,0].mean():.4f}  rec(lab=0) {a[:,1].mean():.4f}  "
          f"test event-DA {da_te:.4f}  ({len(a)} frames)")
    best_te = 0.5 * (ev_te[:, :, 0] + ev_te[:, :, 1]).mean(0)
    print(f"  test best-tau event-DA {float(best_te.max()):.4f} (reference)")
    print(f"  wall {(time.time()-t0)/60:.0f} min")
    torch.save({"state_dict": best_sd, "tau_event": float(TAUS[j]),
                "test_event_da": float(da_te)}, f"{TMP}/{tag}.pt")
    json.dump({"tag": tag, "params": npar, "seed": seed, "val_px": best,
               "tau_event": float(TAUS[j]), "test_event_da": float(da_te),
               "test_best": float(best_te.max())},
              open(f"{TMP}/{tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
