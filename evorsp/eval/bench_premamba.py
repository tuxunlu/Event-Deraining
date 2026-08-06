"""Latency bench of PRE-Mamba (released SYTHETIC ckpt) under our protocol.

Protocol: batch 1, 100 warm-up, 7 repeats x 20 iters, median of repeats,
spread reported (reject if spread > +/-0.15 ms on an idle node -- rerun then).
Inputs: real test frames built by their own test pipeline (GridSample voxelize
+ Collect + collate), one median-size 50mm frame and one heavy 150mm frame.
Reports ms/frame and ms per million current-scan events (their paper metric
is 0.4 s/M events).
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import statistics
import sys
import time

import torch

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model

CFG = f"{C.PREMAMBA}/configs/event_rain/PRE_Mamba.py"
CKPT = (f"{C.PREMAMBA}/exp/event_rain/SYTHETIC/"
        "model/model_best.pth")
DEV = "cuda"

cfg = default_config_parser(CFG, None)
model = build_model(cfg.model).to(DEV).eval()
ck = torch.load(CKPT, map_location="cpu")
sd = {k.replace("module.", "", 1): v for k, v in ck["state_dict"].items()}
model.load_state_dict(sd, strict=True)
print(f"params: {sum(p.numel() for p in model.parameters()):,} | "
      f"ckpt epoch {ck.get('epoch')}")

ds = build_dataset(cfg.data.test)

# frame 194 of the 50mm block and 194 of the 150mm block (mid-sequence)
n50 = sum(1 for f in ds.data_list if "/50mm/" in f)
for idx in (194, n50 + 194):
    item = ds[idx]
    frags = item["fragment_list"]
    name = item["name"]
    n_ev = int((item["tn"].squeeze(1) == 0).sum())
    inputs = []
    for fr in frags:
        d = collate_fn([fr])
        d = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
             for k, v in d.items()}
        inputs.append(d)
    with torch.no_grad():
        for _ in range(100):
            for d in inputs:
                model(d)
        torch.cuda.synchronize()
        meds = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(20):
                for d in inputs:
                    model(d)
            torch.cuda.synchronize()
            meds.append((time.perf_counter() - t0) / 20 * 1000)
    med = statistics.median(meds)
    n_pts = sum(int(d["coord"].shape[0]) for d in inputs)
    print(f"{name}: {len(frags)} fragment(s), {n_pts:,} pts in, "
          f"{n_ev:,} current-scan events | median {med:.2f} ms "
          f"[{min(meds):.2f},{max(meds):.2f}] | "
          f"{med / n_ev * 1e6:.1f} ms/M events", flush=True)
