"""Controlled short-schedule pilot: DFFN baseline vs the proposed
frequency-selective-scan block, trained identically on the same data.

The data is BINARY event masks (GT 8.9% on, rainy 22.3% on, rainy is a
superset of GT), so the task is per-pixel classification: which of the
lit pixels are real signal and which are rain. Objective is BCE on logits,
identical for every model. Same seed, subset, optimiser and schedule.
"""
import sys, os, glob, time, json, argparse
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.DynamicFourierFilterNet import DynamicFourierFilterNet
from fss_model import FSSNet
from dffn_global import DFFNGlobal

ROOT = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/efft_results"
DEV = "cuda"


class RainSet(Dataset):
    def __init__(self, split, intensities, limit=None):
        self.raw = sorted(glob.glob(f"{ROOT}/raw_data/*.npz"))
        self.items = []
        for mm in intensities:
            fs = sorted(glob.glob(f"{ROOT}/merge_data/{split}/{mm}/*.npz"))
            n = min(len(fs), len(self.raw))
            if limit:
                n = min(n, limit)
            for i in range(n):
                self.items.append((fs[i], self.raw[i], mm))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _img(path):
        d = np.load(path, allow_pickle=True)
        z = torch.from_numpy(d["fft_complex"].astype(np.complex64))
        return torch.fft.ifft2(z).real.unsqueeze(0)

    def __getitem__(self, i):
        mp, rp, mm = self.items[i]
        return self._img(mp), self._img(rp), mm


def loss_fn(logits, gt):
    """Identical for every model: BCE on the binary event mask."""
    tgt = (gt > 0.5).float()
    return F.binary_cross_entropy_with_logits(logits, tgt)


TAUS = torch.linspace(0.05, 0.95, 19)


@torch.no_grad()
def evaluate(model, loader):
    """DA is threshold-sensitive, so sweep tau and report each model at its own
    best operating point — otherwise the comparison measures calibration, not
    architecture. Returns per-intensity metrics at the globally best tau."""
    model.eval()
    sr_sum = torch.zeros(len(TAUS), device=DEV)
    nr_sum = torch.zeros(len(TAUS), device=DEV)
    per = {}
    cnt = 0
    for merge, raw, mm in loader:
        merge, raw = merge.to(DEV), raw.to(DEV)
        pred = torch.sigmoid(model(merge))
        gt_b = (raw > 0.5).float()
        rain_gt = (merge > 0.5).float() * (1 - gt_b)
        for ti, t in enumerate(TAUS):
            pr = (pred > t.item()).float()
            sr = (pr * gt_b).sum((1, 2, 3)) / (gt_b.sum((1, 2, 3)) + 1e-8)
            nr = (rain_gt.sum((1, 2, 3)) - (pr * rain_gt).sum((1, 2, 3))) \
                / (rain_gt.sum((1, 2, 3)) + 1e-8)
            sr_sum[ti] += sr.sum(); nr_sum[ti] += nr.sum()
            for j, m in enumerate(mm):
                per.setdefault(m, {}).setdefault(ti, []).append(
                    (sr[j].item(), nr[j].item()))
        cnt += merge.shape[0]
    da = 0.5 * (sr_sum + nr_sum) / max(cnt, 1)
    bi = int(da.argmax().item())
    out = {}
    for m, d in per.items():
        arr = np.array(d[bi])
        s, n = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        out[m] = {"SR": s, "NR": n, "DA": 0.5 * (s + n)}
    out["_tau"] = float(TAUS[bi])
    return out


def run(name, build, tr, va, epochs, lr=5e-4):
    torch.manual_seed(0); np.random.seed(0)
    model = build().to(DEV)
    p = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    print(f"\n=== {name}  ({p:,} params) ===", flush=True)
    t0 = time.perf_counter()
    best = 0.0
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for merge, raw, _ in tr:
            merge, raw = merge.to(DEV, non_blocking=True), raw.to(DEV, non_blocking=True)
            loss = loss_fn(model(merge), raw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        m = evaluate(model, va)
        tau = m.pop("_tau")
        mean_da = float(np.mean([v["DA"] for v in m.values()]))
        best = max(best, mean_da)
        print(f"  ep {ep+1:2d}/{epochs}  train {tot/max(nb,1):.4f}  tau {tau:.2f}  " +
              "  ".join(f"{k}: SR {v['SR']:.3f} NR {v['NR']:.3f} DA {v['DA']:.3f}"
                        for k, v in sorted(m.items())) +
              f"   meanDA {mean_da:.4f}", flush=True)
    dt = time.perf_counter() - t0
    final = evaluate(model, va)
    tau = final.pop("_tau")
    print(f"  wall {dt/60:.1f} min   best meanDA {best:.4f}   final tau {tau:.2f}")
    return {"name": name, "params": p, "metrics": final, "best_meanDA": best,
            "tau": tau,
            "final_meanDA": float(np.mean([v["DA"] for v in final.values()]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=389)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    # span light -> heavy rain, identical for every model
    tr_int = ["5mm", "25mm", "75mm", "175mm"]
    va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
    print("train intensities:", tr_int)
    print("val   intensities:", va_int)

    tr = DataLoader(RainSet("train", tr_int, args.limit), batch_size=args.bs,
                    shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    va = DataLoader(RainSet("validation", va_int, 120), batch_size=args.bs,
                    shuffle=False, num_workers=2, pin_memory=True,
                    persistent_workers=True)
    print(f"train {len(tr.dataset)} frames, val {len(va.dataset)} frames, "
          f"{len(tr)} steps/epoch", flush=True)

    res = []
    res.append(run("A. DFFN baseline (K^2=9 unfold)",
                   lambda: DynamicFourierFilterNet(in_chans=1, out_chans=1,
                                                   dim=32, num_blocks=4),
                   tr, va, args.epochs))
    res.append(run("B. FSSNet (freq selective scan)",
                   lambda: FSSNet(dim=32, nb=4, d_state=16), tr, va, args.epochs))
    res.append(run("C. FSSNet + polar scan order",
                   lambda: FSSNet(dim=32, nb=4, d_state=16, order="polar"),
                   tr, va, args.epochs))
    # mechanism isolation: global spectral context WITHOUT any SSM
    res.append(run("D. DFFN + global spectral context",
                   lambda: DFFNGlobal(dim=32, num_blocks=4, ctx_dim=16),
                   tr, va, args.epochs))

    print("\n\n================ SUMMARY ================")
    for r in res:
        print(f"{r['name']:36s} {r['params']:8,} params   "
              f"final meanDA {r['final_meanDA']:.4f}   best {r['best_meanDA']:.4f}")
        for k, v in sorted(r["metrics"].items()):
            print(f"    {k:8s} SR {v['SR']:.4f}  NR {v['NR']:.4f}  DA {v['DA']:.4f}")
    json.dump(res, open("/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/train_results.json",
                        "w"), indent=2)
