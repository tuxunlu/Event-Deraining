"""Stage 5 — distillation from FourierMamba2D into ORSPNet.

Teacher: tmp/ckpt_fmamba.pt, 23,592,081 params. NOTE it was trained on the PILOT
split (train intensities {5,25,75,175}), so on the full 14-intensity train set it
is partly out of distribution. Rain rate is a smooth axis and 5/25/75/175 spans
it, so its logits should still be informative — but this is a cheap first test of
the distillation hypothesis, not the final teacher. If it works, retrain the
teacher on the full split before believing the number.

Teacher logits are cached ONCE to a fp16 memmap (5,446 x 256 x 256 = 714 MB), so
training costs one extra BCE term and ZERO extra forward passes. Inference cost
of the student is unchanged — that is the whole point.

Loss:  L = (1-alpha) * balanced_gt  +  alpha * kd
  balanced_gt : the confirmed win (align the loss with DA = 1/2(SR+NR))
  kd          : BCE(student_logits, sigmoid(teacher_logits)) on lit pixels
Precedents: SED arXiv:2606.14631 (554x compression, plain output-level KL,
weight 1.0, no temperature); LiteDenoiseNet arXiv:2605.03680 (9:1 teacher:GT).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_compare as TC
from run_exp import balanced_loss

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
CACHE = f"{TMP}/teacher_logits_train.f16"


class IndexedRainSet(Dataset):
    """RainSet that also returns the item index, so cached logits can be keyed."""

    def __init__(self, split, intensities):
        self.ds = TC.RainSet(split, intensities)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        m, r, mm = self.ds[i]
        return m, r, mm, i


def build_cache(ds, device):
    from model.FourierMamba2D import FourierMamba2D
    ck = torch.load(f"{TMP}/ckpt_fmamba.pt", map_location="cpu")
    t = FourierMamba2D(in_chans=1, out_chans=1, dim=32, num_blocks=[2, 2, 2, 2])
    t.load_state_dict(ck["state_dict"])
    t = t.to(device).eval()
    n = len(ds)
    mm = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float16,
                                   shape=(n, 256, 256))
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for merge, _, _, idx in dl:
            out = t(merge.to(device)).squeeze(1).half().cpu().numpy()
            mm[idx.numpy()] = out
    mm.flush()
    print(f"  cached {n} teacher logit maps in "
          f"{(time.perf_counter()-t0)/60:.1f} min -> {CACHE}", flush=True)
    del t
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"kd_a{args.alpha:.2f}".replace(".", "")

    ROOT = TC.ROOT
    tr_int = sorted(os.listdir(f"{ROOT}/merge_data/train"))
    va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
    tr_ds = IndexedRainSet("train", tr_int)
    va = DataLoader(TC.RainSet("validation", va_int), batch_size=4, shuffle=False,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    if not os.path.exists(CACHE):
        print("caching teacher logits (one-off) ...", flush=True)
        build_cache(tr_ds, TC.DEV)
    cache = np.load(CACHE, mmap_mode="r")
    assert cache.shape[0] == len(tr_ds), \
        f"cache {cache.shape[0]} != dataset {len(tr_ds)} — delete {CACHE} and rerun"

    tr = DataLoader(tr_ds, batch_size=4, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True, persistent_workers=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from rsp_model_v2 import ORSPNet
    m = ORSPNet().to(TC.DEV)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    print(f"[{tag}] student {npar:,} params  alpha={args.alpha}  "
          f"{len(tr)} steps/epoch", flush=True)

    best, hist = -1.0, []
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        m.train(); tot = nb = 0
        for merge, raw, _, idx in tr:
            merge = merge.to(TC.DEV, non_blocking=True)
            raw = raw.to(TC.DEV, non_blocking=True)
            tl = torch.from_numpy(np.asarray(cache[idx.numpy()], dtype=np.float32))
            tl = tl.unsqueeze(1).to(TC.DEV, non_blocking=True)

            out = m(merge)
            l_gt = balanced_loss(out, raw, merge)
            lit = (merge > 0.5).float()
            bce = F.binary_cross_entropy_with_logits(
                out, torch.sigmoid(tl), reduction="none")
            l_kd = ((bce * lit).sum() / lit.sum().clamp(min=1.0)
                    + 0.05 * (bce * (1 - lit)).sum() / (1 - lit).sum().clamp(min=1.0))
            loss = (1 - args.alpha) * l_gt + args.alpha * l_kd

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        vm = TC.evaluate(m, va); vt = vm.pop("_tau")
        vda = float(np.mean([v["DA"] for v in vm.values()]))
        best = max(best, vda); hist.append(vda)
        print(f"  ep {ep+1:2d}/{args.epochs}  loss {tot/nb:.4f}  tau {vt:.2f}  "
              f"valDA {vda:.4f}  [{(time.perf_counter()-t0)/60:.0f} min]", flush=True)

    json.dump({"tag": tag, "alpha": args.alpha, "params": npar, "best_valDA": best,
               "history": hist}, open(f"{TMP}/exp_{tag}.json", "w"), indent=2)
    print(f"\nEXPRESULT {tag}  params={npar}  best={best:.4f}  "
          f"(balanced-only baseline @ep10 = 0.9166)", flush=True)


if __name__ == "__main__":
    main()
