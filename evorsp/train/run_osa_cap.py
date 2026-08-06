"""Capacity-matched arm: the 0.8449-0.8541 spatial models carry ~51-53K params;
OSA-Net at hidden=16 carries 37.9K. hidden=40 -> 50,274, matched to arm F
(51,594) and F' (47,178), so the comparison is capacity-fair."""
import sys, os, json
sys.path.insert(0,"/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0,"/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from torch.utils.data import DataLoader
import train_compare as TC
from osa_model import OSANet
ROOT=TC.ROOT
tr=DataLoader(TC.RainSet("train",["5mm","25mm","75mm","175mm"],389),batch_size=8,
              shuffle=True,num_workers=4,pin_memory=True,drop_last=True,persistent_workers=True)
va=DataLoader(TC.RainSet("validation",sorted(os.listdir(f"{ROOT}/merge_data/validation")),120),
              batch_size=8,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
res=[]
res.append(TC.run("H4. OSA-Net h=40 (bank+rate)",
                  lambda: OSANet(dim=32,num_blocks=4,hidden=40),tr,va,10))
res.append(TC.run("H5. OSA-Net h=40 no FFT",
                  lambda: OSANet(dim=32,num_blocks=4,hidden=40,use_bank=False,use_rate=False),tr,va,10))
json.dump(res,open("/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp/osa_cap.json","w"),indent=2)
for r in res: print(f"{r['name']:32s} {r['params']:>7,}p best {r['best_meanDA']:.4f}")
