"""Run ORSPNet at the SAME 10-epoch protocol as arms A-G so it is comparable.
The technical critic measured 0.8387 at 4 EPOCHS against a 4-epoch DFFN (0.7724);
A at 10 epochs is 0.8173, so that margin is partly faster convergence.
Also runs the F3 fix (tanh scale >= 2) the critic identified: as shipped,
max sum_j M_j = 0.6099, so a true notch is unreachable at 100% of bins.
"""
import sys, json, os
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from torch.utils.data import DataLoader
import train_compare as TC
from rsp_model import ORSPNet

EPOCHS = int(os.environ.get("EPOCHS", 10))
ROOT = TC.ROOT
tr_int = ["5mm", "25mm", "75mm", "175mm"]
va_int = sorted(os.listdir(f"{ROOT}/merge_data/validation"))
tr = DataLoader(TC.RainSet("train", tr_int, 389), batch_size=8, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
va = DataLoader(TC.RainSet("validation", va_int, 120), batch_size=8, shuffle=False,
                num_workers=2, pin_memory=True, persistent_workers=True)
print(f"train {len(tr.dataset)}  val {len(va.dataset)}  {len(tr)} steps/epoch", flush=True)

def build(scale=None):
    def f():
        m = ORSPNet()
        if scale is not None:
            for mod in m.modules():
                if hasattr(mod, "scale") and isinstance(getattr(mod, "scale"), float):
                    mod.scale = scale
        return m
    return f

res = []
res.append(TC.run("H. ORSPNet (as reviewed)", build(), tr, va, EPOCHS))
res.append(TC.run("I. ORSPNet + F3 fix (tanh scale 2.5)", build(2.5), tr, va, EPOCHS))
print("\n================ SUMMARY (ORSP, 10 epochs) ================")
for r in res:
    print(f"{r['name']:38s} {r['params']:8,} params  best {r['best_meanDA']:.4f}")
    for k, v in sorted(r["metrics"].items()):
        print(f"    {k:8s} SR {v['SR']:.4f}  NR {v['NR']:.4f}  DA {v['DA']:.4f}")
print("""
matched-protocol reference (10 epochs, same harness):
  A. DFFN baseline            72,074  0.8173      E. SpatialGainNet   52,714  0.8509
  B. FSSNet (SSM)             48,650  0.8346      F. SpatialGain noFFT 51,594 0.8469
  D. DFFN + global context    95,946  0.8362      G. SpatialGain local 52,714 0.8449""")
json.dump(res, open("/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/orsp_results.json","w"), indent=2)
