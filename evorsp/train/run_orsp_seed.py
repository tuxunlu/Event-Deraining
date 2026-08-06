"""ORSPNet at one seed, otherwise the IDENTICAL pilot protocol as arms A-I.

train_compare.run() hardcodes torch.manual_seed(0) BEFORE calling build(), so the
seed is applied inside build() -- that re-seeds both the weight init and the
global RNG the shuffling DataLoader draws from, giving init AND data-order
variation, which is what a seed sweep is supposed to measure.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import argparse, json, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_compare as TC
from rsp_model import ORSPNet

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--epochs", type=int, default=10)
args = ap.parse_args()

ROOT = TC.ROOT
tr_int = ["5mm", "25mm", "75mm", "175mm"]
va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))

tr = DataLoader(TC.RainSet("train", tr_int, 389), batch_size=8, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
va = DataLoader(TC.RainSet("validation", va_int, 120), batch_size=8, shuffle=False,
                num_workers=2, pin_memory=True, persistent_workers=True)
print(f"seed {args.seed}: train {len(tr.dataset)}  val {len(va.dataset)}  "
      f"{len(tr)} steps/epoch", flush=True)


def build():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    return ORSPNet()


r = TC.run(f"ORSPNet seed{args.seed}", build, tr, va, args.epochs)
out = f"{C.CKPT}/orsp_seed{args.seed}.json"
json.dump(r, open(out, "w"), indent=2)
print(f"\nSEEDRESULT seed={args.seed} params={r['params']} "
      f"best={r['best_meanDA']:.4f} final={r['final_meanDA']:.4f}", flush=True)
