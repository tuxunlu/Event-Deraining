"""Evaluate our KITTI EvORSP-3T under PRE-Mamba's event-level DA metric.

Protocol bridge for the head-to-head:
  - same test frames (merge_data/{50,150}mm, all 389 frames each),
  - same per-event labels as PRE-Mamba's loader: an event is background (0)
    iff its exact (x, y, t) appears in the clean raw_data stream, else rain (1),
  - same per-frame SR/NR/DA and frame-mean aggregation as their SemSegTester,
  - our model's per-pixel keep decision (protocol tau=0.70 from val {20,80})
    assigned to every event in that pixel (subset rule).
Neither model saw 50/150mm in training.
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

DEV = "cuda"
S = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
PACK = "/fs/nexus-scratch/tuxunlu/kitti_t16/test"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
T_BUILD, T, R = 16, 4, 256
SRC_W, SRC_H = 460, 352
TAUS = np.linspace(0.05, 0.95, 19)
SEEDS = ["k3d_T4b3off", "k3d_T4b3off_s1", "k3d_T4b3off_s2"]


def key(x, y, t):
    return (t.astype(np.int64) * (SRC_W * SRC_H)
            + y.astype(np.int64) * SRC_W + x.astype(np.int64))


def load_planes(path):
    with np.load(path) as d:
        on = np.unpackbits(d["on"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
        off = np.unpackbits(d["off"])[: T_BUILD * R * R].reshape(T_BUILD, R, R)
    on = on.reshape(T, T_BUILD // T, R, R).max(1)
    off = off.reshape(T, T_BUILD // T, R, R).max(1)
    return (torch.from_numpy(on).float()[None].to(DEV),
            torch.from_numpy(off).float()[None].to(DEV))


models = []
for tag in SEEDS:
    ck = torch.load(f"{TMP}/{tag}.pt", map_location="cpu")
    m = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
    m.load_state_dict(ck["state_dict"])
    models.append(m.to(DEV).eval())

def frame_stats(split, mm_list, desc):
    """Per-frame (SR, NR) at every tau for each seed: {tag: {mm: [n_frames, 19, 2]}}."""
    out = {tag: {} for tag in SEEDS}
    with torch.no_grad():
        for mm in mm_list:
            frames = sorted(glob.glob(f"{S}/merge_data/{mm}/*.npz"))
            per = {tag: [] for tag in SEEDS}
            for i, f in enumerate(frames):
                b = os.path.basename(f)
                pk = f"{PACK.replace('/test', '/' + split)}/{mm}/{b}"
                if not os.path.exists(pk):
                    continue
                with np.load(f) as d:
                    x, y, t = d["x"], d["y"], d["t"]
                with np.load(f"{S}/raw_data/{b}") as d:
                    rk = np.sort(key(d["x"], d["y"], d["t"]))
                lab_rain = ~np.isin(key(x, y, t), rk)          # 1 = rain
                sx = np.clip((x.astype(np.int64) * R) // SRC_W, 0, R - 1)
                sy = np.clip((y.astype(np.int64) * R) // SRC_H, 0, R - 1)
                n_bg = max(int((~lab_rain).sum()), 1)
                n_rn = max(int(lab_rain.sum()), 1)
                on, off = load_planes(pk)
                for tag, m in zip(SEEDS, models):
                    pmap = torch.sigmoid(m(on, x_off=off))[0, 0].cpu().numpy()
                    pev = pmap[sy, sx]
                    row = []
                    for tau in TAUS:
                        keep = pev > tau
                        sr = int((keep & ~lab_rain).sum()) / n_bg
                        nr = int((~keep & lab_rain).sum()) / n_rn
                        row.append((sr, nr))
                    per[tag].append(row)
                if i % 100 == 0:
                    print(f"{desc} {mm} {i}/{len(frames)}", flush=True)
            for tag in SEEDS:
                out[tag][mm] = np.array(per[tag])              # [n, 19, 2]
    return out


val = frame_stats("val", ("20mm", "80mm"), "val")
test = frame_stats("test", ("50mm", "150mm"), "test")

print("\n=== OUR EvORSP-3T UNDER PRE-MAMBA'S EVENT-LEVEL METRIC ===")
print("(tau selected on val {20,80}mm by event-DA, protocol-style)")
overall = []
for tag in SEEDS:
    vda = np.mean([val[tag][mm].mean(0) for mm in val[tag]], 0)  # [19,2]
    j = int(np.argmax(0.5 * (vda[:, 0] + vda[:, 1])))
    tau = TAUS[j]
    per = {}
    for mm in test[tag]:
        a = test[tag][mm][:, j, :]                              # [n,2]
        per[mm] = (a[:, 0].mean(), a[:, 1].mean(),
                   0.5 * (a[:, 0] + a[:, 1]).mean())
    m50, m150 = per["50mm"], per["150mm"]
    da = 0.5 * (m50[2] + m150[2])
    sr = 0.5 * (m50[0] + m150[0])
    nr = 0.5 * (m50[1] + m150[1])
    overall.append(da)
    print(f"{tag}: tau {tau:.2f} | 50mm DA {m50[2]:.4f} | 150mm DA {m150[2]:.4f}"
          f" | mean SR {sr:.4f} NR {nr:.4f} DA {da:.4f}")
o = np.array(overall)
print(f"3-seed event-DA mean {o.mean():.4f} +/- {o.std():.4f}")
