"""Triage the three teacher-free changes, 10 epochs on the FULL protocol split.

Comparison point: the completed ORSPNet protocol run scored val DA 0.9158 at
epoch 10 (proto_orsp.log), same seed, same data, same code path — so a 10-epoch
arm is directly comparable to it. Winners get promoted to the full 50 epochs.

Arms
  base       unchanged ORSPNet                              (control)
  nowd       no weight decay on 1-D params (band_scale, bank, biases, norms)
  balanced   loss = balanced accuracy surrogate over LIT pixels only
  gainsplit  gate = unbounded per-band constant + bounded spatial residual
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_compare as TC

TMP = f"{C.CKPT}"


def lit_bce(logits, raw, merge):
    """Plain BCE restricted to LIT pixels. ~78% of pixels are dark and cannot
    affect DA = 1/2(SR+NR), so gradient spent there has zero metric leverage."""
    tgt = (raw > 0.5).float()
    lit = (merge > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (bce * lit).sum() / lit.sum().clamp(min=1.0)


def balanced_loss(logits, raw, merge, w_dark=0.05):
    """Surrogate for DA = 1/2(SR + NR).

    SR is recall over GT-signal pixels and NR is recall over rain pixels, and BOTH
    are defined only on LIT pixels (merge > 0.5). Plain BCE instead averages over
    all 65,536 pixels, ~78% of which are dark and count toward neither metric, and
    it weights signal and rain by their (very unequal) frequencies. This weights
    the two classes equally and confines them to the pixels the metric sees.
    """
    tgt = (raw > 0.5).float()
    lit = (merge > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    sig = lit * tgt
    rain = lit * (1.0 - tgt)
    dark = 1.0 - lit
    ls = (bce * sig).sum() / sig.sum().clamp(min=1.0)
    lr = (bce * rain).sum() / rain.sum().clamp(min=1.0)
    ld = (bce * dark).sum() / dark.sum().clamp(min=1.0)
    return 0.5 * ls + 0.5 * lr + w_dark * ld


def param_groups(model, wd):
    """Standard practice: no weight decay on 1-D params. Here that specifically
    frees band_scale and the 32 analytic bank parameters, which decay was pulling
    toward zero — a candidate explanation for the measured rail saturation."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "band_scale" in n or "bank." in n
         or "band_const" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}], len(no_decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["base", "nowd", "balanced", "gainsplit", "dil", "dil_bal", "iso_bal", "litbce", "litbce_bal",
                             "unet", "unet_bal", "unet_small",
                             "strip", "rate", "mult", "streaknet"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=15,
                    help="strip length. dil reaches 1+2*64=129px, so K=127 matches it.")
    args = ap.parse_args()

    ROOT = TC.ROOT
    tr_int = sorted(os.listdir(f"{ROOT}/merge_data/train"))
    va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
    tr = DataLoader(TC.RainSet("train", tr_int), batch_size=4, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)
    va = DataLoader(TC.RainSet("validation", va_int), batch_size=4, shuffle=False,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.variant in ("strip", "rate", "mult", "streaknet"):
        from rsp_streak import StreakNet
        m = StreakNet(K=args.K,
                      use_strip=args.variant in ("strip", "streaknet"),
                      use_rate=args.variant in ("rate", "streaknet"),
                      use_darkmask=args.variant in ("mult", "streaknet")).to(TC.DEV)
    elif args.variant.startswith("unet"):
        from rsp_unet import ORSPUNet
        # unet_small trims block count to claw back the 2.2x latency cost
        kw = (dict(blocks=(1, 1, 2), dec_blocks=(1, 1), dims=(20, 28, 40))
              if args.variant == "unet_small" else {})
        m = ORSPUNet(**kw).to(TC.DEV)
    else:
        from rsp_model_v2 import ORSPNet
        # the gate's dilated dw convs are PARALLEL and summed, so per-block
        # receptive field is 1 + 2*max(dilation) = 33 px. (1,8,32,64) -> 129 px.
        dil = ((1, 8, 32, 64)
               if args.variant in ("dil", "dil_bal", "iso_bal", "litbce", "litbce_bal")
               else (1, 4, 16))
        m = ORSPNet(gain_split=(args.variant == "gainsplit"), dilations=dil,
                    isotropic=(args.variant == "iso_bal")).to(TC.DEV)
    npar = sum(p.numel() for p in m.parameters())

    if args.variant == "nowd":
        groups, n_nodecay = param_groups(m, 5e-3)
        opt = torch.optim.AdamW(groups, lr=5e-4)
        print(f"[{args.variant}] {n_nodecay} tensors excluded from weight decay", flush=True)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    print(f"[{args.variant}] {npar:,} params  (base is 36,206)  "
          f"{len(tr)} steps/epoch", flush=True)

    best, hist = -1.0, []
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        m.train(); tot = nb = 0
        for merge, raw, _ in tr:
            merge, raw = merge.to(TC.DEV, non_blocking=True), raw.to(TC.DEV, non_blocking=True)
            out = m(merge)
            loss = (lit_bce(out, raw, merge) if args.variant == "litbce" else
                    balanced_loss(out, raw, merge) if args.variant in ("balanced", "dil_bal", "iso_bal", "litbce_bal", "unet_bal", "unet_small",
                                     "strip", "rate", "mult", "streaknet")
                    else TC.loss_fn(out, raw))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        vm = TC.evaluate(m, va)
        vt = vm.pop("_tau")
        vda = float(np.mean([v["DA"] for v in vm.values()]))
        best = max(best, vda); hist.append(vda)
        print(f"  ep {ep+1:2d}/{args.epochs}  train {tot/nb:.4f}  tau {vt:.2f}  "
              f"valDA {vda:.4f}{'  *' if vda >= best else ''}  "
              f"[{(time.perf_counter()-t0)/60:.0f} min]", flush=True)

    json.dump({"variant": args.variant + (f"_K{args.K}" if args.K != 15 else ""), "params": npar, "best_valDA": best,
               "final_valDA": hist[-1], "history": hist,
               "wall_min": (time.perf_counter() - t0) / 60},
              open(f"{TMP}/exp_{args.variant}.json", "w"), indent=2)
    print(f"\nEXPRESULT {args.variant}{'_K'+str(args.K) if args.K!=15 else ''}  params={npar}  best={best:.4f}  "
          f"final={hist[-1]:.4f}   (ORSPNet baseline @ep10 = 0.9158)", flush=True)


if __name__ == "__main__":
    main()
