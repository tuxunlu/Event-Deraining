"""Seed sweep: the noise floor was never characterised. B/C/D differed by 0.0016
and that was ASSUMED to be single-seed noise. Measure it instead.

Runs the two extremes of the OSA-Net front-end ablation at seeds 1 and 2;
combined with the seed-0 runs in run_osa.py this gives 3 seeds of each.
"""
import sys, os, json
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
import numpy as np, torch
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

_orig = torch.manual_seed
SEED = [0]
def _patched(_):            # TC.run() calls manual_seed(0); redirect to our seed
    return _orig(SEED[0])
torch.manual_seed = _patched

res = []
for s in (1, 2):
    SEED[0] = s
    np.random.seed(s)
    res.append(TC.run(f"H1. OSA bank+rate  seed{s}",
                      lambda: OSANet(dim=32, num_blocks=4), tr, va, 10))
    res.append(TC.run(f"H3. OSA no-FFT     seed{s}",
                      lambda: OSANet(dim=32, num_blocks=4, use_bank=False,
                                     use_rate=False), tr, va, 10))
    json.dump(res, open("/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/osa_seeds.json", "w"),
              indent=2)

print("\n======== SEED SUMMARY ========")
for r in res:
    print(f"{r['name']:30s} {r['params']:>7,}p  best {r['best_meanDA']:.4f}")
