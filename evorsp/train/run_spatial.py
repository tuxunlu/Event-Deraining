"""Run SpatialGainNet through the IDENTICAL pilot protocol as arms A-D
(same RainSet, same loss, same optimiser/schedule, same threshold sweep)."""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import sys, json

import os
import numpy as np
from torch.utils.data import DataLoader

import train_compare as TC
from spatial_model import SpatialGainNet

EPOCHS = int(os.environ.get("EPOCHS", 10))
ROOT = TC.ROOT

tr_int = ["5mm", "25mm", "75mm", "175mm"]
va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
print("train intensities:", tr_int)
print("val   intensities:", va_int, flush=True)

tr = DataLoader(TC.RainSet("train", tr_int, 389), batch_size=8, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
va = DataLoader(TC.RainSet("validation", va_int, 120), batch_size=8, shuffle=False,
                num_workers=2, pin_memory=True, persistent_workers=True)
print(f"train {len(tr.dataset)} frames, val {len(va.dataset)} frames, "
      f"{len(tr)} steps/epoch", flush=True)

res = []
res.append(TC.run("E. SpatialGainNet (dilated, +rate sensor)",
                  lambda: SpatialGainNet(dim=32, num_blocks=4, use_rate=True),
                  tr, va, EPOCHS))
res.append(TC.run("F. SpatialGainNet (no FFT at all)",
                  lambda: SpatialGainNet(dim=32, num_blocks=4, use_rate=False),
                  tr, va, EPOCHS))
res.append(TC.run("G. SpatialGainNet local-only (dilation 1,1,1)",
                  lambda: SpatialGainNet(dim=32, num_blocks=4, use_rate=True,
                                         dilations=(1, 1, 1)),
                  tr, va, EPOCHS))

print("\n\n================ SUMMARY (spatial arms) ================")
for r in res:
    print(f"{r['name']:44s} {r['params']:8,} params   "
          f"final meanDA {r['final_meanDA']:.4f}   best {r['best_meanDA']:.4f}")
    for k, v in sorted(r["metrics"].items()):
        print(f"    {k:8s} SR {v['SR']:.4f}  NR {v['NR']:.4f}  DA {v['DA']:.4f}")
print("""
reference (identical protocol, run earlier):
  A. DFFN baseline                    72,074   best 0.8173
  B. FSSNet (freq selective scan)     48,650   best 0.8346
  C. FSSNet + polar order             48,650   best 0.8359
  D. DFFN + global spectral context   95,946   best 0.8362""")
json.dump(res, open(f"{C.CKPT}/spatial_results.json", "w"), indent=2)
