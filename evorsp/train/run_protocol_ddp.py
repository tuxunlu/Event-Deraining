"""FourierMamba2D, 50 epochs, DDP across all 4 GPUs.

Protocol identical to run_protocol.py (the ORSPNet / DFFN runs):
  train  merge_data/train (14 intensities, 5,446 frames)
  select merge_data/validation (20/80mm)
  report merge_data/test (50/150mm)
  50 epochs, AdamW lr 5e-4 wd 5e-3, CosineAnnealing T_max=50, clip 1.0, BCE.

Batch: 1 per GPU x 4 GPUs = effective batch 4, matching the config exactly
(a single GPU OOMs above micro-batch 2 on this model, so DDP is also what makes
the config batch reachable without gradient accumulation).

Evaluation runs on rank 0 only, on the full loader — no sharding, so the metric
is identical to the single-GPU runs and directly comparable.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import json, os, sys, time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import train_compare as TC

TMP = f"{C.CKPT}"
EPOCHS = int(os.environ.get("EPOCHS", 50))


def build():
    from model.FourierMamba2D import FourierMamba2D
    return FourierMamba2D(in_chans=1, out_chans=1, dim=32, num_blocks=[2, 2, 2, 2])


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    p0 = (rank == 0)

    ROOT = TC.ROOT
    tr_int = sorted(os.listdir(f"{ROOT}/merge_data/train"))
    va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
    te_int = sorted(os.listdir(f"{ROOT}/merge_data/test"))

    tr_ds = TC.RainSet("train", tr_int)
    samp = DistributedSampler(tr_ds, num_replicas=world, rank=rank, shuffle=True,
                              drop_last=True)
    tr = DataLoader(tr_ds, batch_size=1, sampler=samp, num_workers=2,
                    pin_memory=True, persistent_workers=True)

    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    net = build().to(dev)
    npar = sum(p.numel() for p in net.parameters())

    CKPT = f"{TMP}/fmamba_ckpt.pt"
    start_ep, ck = 0, None
    if os.environ.get("RESUME") == "1" and os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        net.load_state_dict(ck["model"])
        start_ep = ck["epoch"]
        if p0:
            print(f"RESUMED from epoch {start_ep} (best val {ck['best_val']:.4f} "
                  f"@ ep {ck['best_ep']})", flush=True)
    model = DDP(net, device_ids=[local])

    if p0:
        va = DataLoader(TC.RainSet("validation", va_int), batch_size=2, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)
        te = DataLoader(TC.RainSet("test", te_int), batch_size=2, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)
        print(f"FourierMamba2D  {npar:,} params   world={world}  "
              f"batch 1/gpu x {world} = effective {world}", flush=True)
        print(f"train {len(tr_ds)}   val {len(va.dataset)}   test {len(te.dataset)}   "
              f"{len(tr)} steps/epoch/rank", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_val, best_state, best_ep = -1.0, None, -1
    if ck is not None:
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        best_val, best_ep = ck["best_val"], ck["best_ep"]
        best_state = ck["best_state"]
    t0 = time.perf_counter()
    for ep in range(start_ep, EPOCHS):
        model.train(); samp.set_epoch(ep)
        tot = nb = 0
        for merge, raw, _ in tr:
            merge, raw = merge.to(dev, non_blocking=True), raw.to(dev, non_blocking=True)
            loss = TC.loss_fn(model(merge), raw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sch.step()

        if p0:
            vm = TC.evaluate(net, va)
            vt = vm.pop("_tau")
            vda = float(np.mean([v["DA"] for v in vm.values()]))
            star = ""
            if vda > best_val:
                best_val, best_ep = vda, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                star = "  *"
            print(f"  ep {ep+1:3d}/{EPOCHS}  train {tot/max(nb,1):.4f}  val tau {vt:.2f}  "
                  f"valDA {vda:.4f}{star}  [{(time.perf_counter()-t0)/60:.0f} min]", flush=True)
            torch.save({"epoch": ep + 1, "model": net.state_dict(),
                        "opt": opt.state_dict(), "sch": sch.state_dict(),
                        "best_val": best_val, "best_ep": best_ep,
                        "best_state": best_state}, CKPT + ".tmp")
            os.replace(CKPT + ".tmp", CKPT)
        dist.barrier()

    if p0:
        net.load_state_dict(best_state)
        tm = TC.evaluate(net, te)
        ttau = tm.pop("_tau")
        tda = float(np.mean([v["DA"] for v in tm.values()]))
        out = {"model": "fmamba_ddp", "params": npar, "best_val_epoch": best_ep,
               "best_valDA": best_val, "test_tau": ttau, "test": tm,
               "test_meanDA": tda, "wall_min": (time.perf_counter() - t0) / 60}
        torch.save({"state_dict": best_state, **out}, f"{TMP}/proto_fmamba_ddp.pt")
        json.dump(out, open(f"{TMP}/proto_fmamba_ddp.json", "w"), indent=2)
        print(f"\n=== PROTOCOL RESULT: FourierMamba2D (DDP) ===")
        print(f"  params {npar:,}   selected at epoch {best_ep} (valDA {best_val:.4f})")
        print(f"  TEST tau {ttau:.2f}")
        for k in sorted(tm):
            print(f"    {k:8s} SR {tm[k]['SR']:.4f}  NR {tm[k]['NR']:.4f}  DA {tm[k]['DA']:.4f}")
        print(f"  test mean DA {tda:.4f}   wall {out['wall_min']:.0f} min", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
