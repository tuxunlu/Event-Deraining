import sys, os, json
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
import torch
from torch.utils.data import DataLoader
import train_compare as TC
from osa_model import OSANet

ROOT = TC.ROOT
tr_int = ["5mm", "25mm", "75mm", "175mm"]
va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
tr = DataLoader(TC.RainSet("train", tr_int, 389), batch_size=8, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
va = DataLoader(TC.RainSet("validation", va_int, 120), batch_size=8, shuffle=False,
                num_workers=2, pin_memory=True, persistent_workers=True)
print(f"train {len(tr.dataset)} val {len(va.dataset)} {len(tr)} steps/ep", flush=True)

EP = 10
res = []
# H1: per-pixel oriented evidence + scalar rate  (the proposal)
res.append(TC.run("H1. OSA-Net (bank + rate)",
                  lambda: OSANet(dim=32, num_blocks=4), tr, va, EP))
# H2: scalar rate only -- isolates the BANK
res.append(TC.run("H2. OSA-Net rate only (no bank)",
                  lambda: OSANet(dim=32, num_blocks=4, use_bank=False), tr, va, EP))
# H3: no FFT anywhere -- isolates the whole front end
res.append(TC.run("H3. OSA-Net no FFT at all",
                  lambda: OSANet(dim=32, num_blocks=4, use_bank=False, use_rate=False),
                  tr, va, EP))
json.dump(res, open("/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/osa_results.json", "w"), indent=2)

print("\n============ SUMMARY ============")
for r in res:
    print(f"{r['name']:34s} {r['params']:>7,}p  best {r['best_meanDA']:.4f}")
    for k, v in sorted(r["metrics"].items()):
        print(f"    {k:8s} SR {v['SR']:.4f} NR {v['NR']:.4f} DA {v['DA']:.4f}")
print("""
reference (identical harness/protocol/seed, 10 epochs):
  A. DFFN baseline                    72,074   0.8173
  B. FSSNet (freq selective scan)     48,650   0.8346
  C. FSSNet + polar order             48,650   0.8359
  D. DFFN + global spectral context   95,946   0.8362
  G. SpatialGainNet local-only        52,714   0.8449
  F. SpatialGainNet no FFT            51,594   0.8469
  E. SpatialGainNet + rate scalar     52,714   0.8509
  F'. DDGNet no-descriptor            47,178   0.8510
  S. DDGNet + rate scalar             52,394   0.8541""")
