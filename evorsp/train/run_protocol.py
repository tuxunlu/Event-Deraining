"""Full 50-epoch protocol run, matching configs/config_dynamicfourierfilternet.yaml.

  train  : merge_data/train        (14 intensities x 389 = 5,446 frames)
  select : merge_data/validation   (20mm, 80mm)   <- model selection ONLY
  report : merge_data/test         (50mm, 150mm)  <- the deck's reported rates

Selecting on validation and reporting on test keeps the test set clean; the
pilot runs earlier in the session selected and reported on the same split, which
is fine for a relative ranking but not for a number anyone would publish.

Protocol from the config: 50 epochs, batch 4, AdamW lr 5e-4 wd 5e-3,
CosineAnnealingLR T_max=50 eta_min=1e-6, grad-clip 1.0, BCE on the binary mask.
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
from torch.utils.data import DataLoader

import train_compare as TC
from run_exp import balanced_loss, lit_bce

TMP = f"{C.CKPT}"


def make(name):
    """Inlined — train_save.py parses args at import time, so it cannot be imported."""
    if name == "dffn":
        from model.DynamicFourierFilterNet import DynamicFourierFilterNet
        return DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32, num_blocks=4)
    if name == "fssnet":
        from fss_model import FSSNet
        return FSSNet(dim=32, nb=4, d_state=16)
    if name == "spatial":
        from spatial_model import SpatialGainNet
        return SpatialGainNet(dim=32, num_blocks=4, use_rate=True)
    if name == "orsp":
        from rsp_model import ORSPNet
        return ORSPNet()
    if name.startswith("orsp_dil"):
        from rsp_model_v2 import ORSPNet as ORSPNetV2
        parts = name.split(":")
        d = tuple(int(v) for v in parts[1].split(","))
        nb = int(parts[2]) if len(parts) > 2 else 4
        return ORSPNetV2(dilations=d, num_blocks=nb)
    if name.startswith("streaknet"):
        from rsp_streak import StreakNet
        K = int(name.split(":")[1])
        return StreakNet(K=K, use_strip=True, use_rate=True, use_darkmask=True)
    if name == "fmamba":
        from model.FourierMamba2D import FourierMamba2D
        return FourierMamba2D(in_chans=1, out_chans=1, dim=32, num_blocks=[2, 2, 2, 2])
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)      # config value
    ap.add_argument("--micro_bs", type=int, default=0)   # 0 -> = batch
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="bce", choices=["bce", "balanced", "litbce"])
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--K", type=int, default=127, help="strip length for streaknet")
    ap.add_argument("--dilations", default="1,4,16",
                    help="ObliqueGate dilations; (1,8,32,64) gives RF 129px vs 33px")
    args = ap.parse_args()
    micro = args.micro_bs or args.batch
    accum = max(1, args.batch // micro)

    ROOT = TC.ROOT
    tr_int = sorted(os.listdir(f"{ROOT}/merge_data/train"))
    va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
    te_int = sorted(os.listdir(f"{ROOT}/merge_data/test"))
    print(f"train {tr_int}\nselect {va_int}\nreport {te_int}", flush=True)

    tr = DataLoader(TC.RainSet("train", tr_int), batch_size=micro, shuffle=True,
                    num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True)
    va = DataLoader(TC.RainSet("validation", va_int), batch_size=args.eval_bs,
                    shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)
    te = DataLoader(TC.RainSet("test", te_int), batch_size=args.eval_bs,
                    shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True)
    print(f"train {len(tr.dataset)} frames ({len(tr)} steps/epoch, micro {micro} "
          f"x accum {accum} = batch {args.batch})   val {len(va.dataset)}   "
          f"test {len(te.dataset)}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    mname = (f"streaknet:{args.K}" if args.model == "streaknet"
             else f"orsp_dil:{args.dilations}:{args.blocks}" if args.model == "orsp"
             and (args.dilations != "1,4,16" or args.blocks != 4) else args.model)
    m = make(mname).to(TC.DEV)
    npar = sum(p.numel() for p in m.parameters())
    print(f"{args.model}: {npar:,} params", flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state, best_ep = -1.0, None, -1
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        m.train()
        tot = nb = 0
        opt.zero_grad(set_to_none=True)
        for i, (merge, raw, _) in enumerate(tr):
            merge, raw = merge.to(TC.DEV, non_blocking=True), raw.to(TC.DEV, non_blocking=True)
            out = m(merge)
            loss = (balanced_loss(out, raw, merge) if args.loss == "balanced"
                    else lit_bce(out, raw, merge) if args.loss == "litbce"
                    else TC.loss_fn(out, raw))
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        sch.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            vm = TC.evaluate(m, va)
            vt = vm.pop("_tau")
            vda = float(np.mean([v["DA"] for v in vm.values()]))
            star = ""
            if vda > best_val:
                best_val, best_ep = vda, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                star = "  *"
            print(f"  ep {ep+1:3d}/{args.epochs}  train {tot/nb:.4f}  "
                  f"val tau {vt:.2f}  valDA {vda:.4f}{star}  "
                  f"[{(time.perf_counter()-t0)/60:.0f} min]", flush=True)
        else:
            print(f"  ep {ep+1:3d}/{args.epochs}  train {tot/nb:.4f}", flush=True)

    # ---- report on TEST using the validation-selected weights ----
    m.load_state_dict(best_state)
    tm = TC.evaluate(m, te)
    ttau = tm.pop("_tau")
    tda = float(np.mean([v["DA"] for v in tm.values()]))
    out = {"model": args.model, "params": npar, "best_val_epoch": best_ep,
           "best_valDA": best_val, "test_tau": ttau, "test": tm, "test_meanDA": tda,
           "wall_min": (time.perf_counter() - t0) / 60}
    tag = (args.model + (f"_b{args.blocks}" if args.blocks != 4 else "")
           + ("_bal" if args.loss == "balanced" else "_lit" if args.loss == "litbce" else "")) \
        + ("_dil" if args.dilations != "1,4,16" else "")
    torch.save({"state_dict": best_state, **out}, f"{TMP}/proto_{tag}.pt")
    json.dump(out, open(f"{TMP}/proto_{tag}.json", "w"), indent=2)

    print(f"\n=== PROTOCOL RESULT: {args.model} ===")
    print(f"  params {npar:,}   selected at epoch {best_ep} (valDA {best_val:.4f})")
    print(f"  TEST tau {ttau:.2f}")
    for k in sorted(tm):
        print(f"    {k:8s} SR {tm[k]['SR']:.4f}  NR {tm[k]['NR']:.4f}  DA {tm[k]['DA']:.4f}")
    print(f"  test mean DA {tda:.4f}   wall {out['wall_min']:.0f} min", flush=True)


if __name__ == "__main__":
    main()
