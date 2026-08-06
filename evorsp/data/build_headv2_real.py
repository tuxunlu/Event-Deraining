"""Head-v2 feature cache for REAL EVK4, built on GPU.

Same contents as build_headv2_cache.py (KITTI) but:
  * 1280x720 native, structure-tensor scales {4,16,64} -- the configuration the
    EigenPyramid falsifier was validated at (0.7504 event-BA vs 0.6888);
  * MICROsecond timestamps (EVK4's native unit), so slice = 1000 us and
    tau = 5000 us. KITTI needed these x1000 because its stamps are nanoseconds
    -- getting this wrong silently produces 1000x too many slices;
  * labels come from the dataset's per-event label files (label 1 = scene,
    established by cross-frame persistence: 0.54 vs 0.20 for rain);
  * features computed with gpu_feats.tensor_gpu (4 ms at 100K events against
    ~200 ms for the NumPy version), so this runs in minutes.

Output: /fs/nexus-scratch/tuxunlu/real_headv2/{scene}/{rain_k}/NNNN.npz
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from gpu_feats import tensor_gpu

S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
PACK = "/fs/nexus-scratch/tuxunlu/real_t16e"
OUT = "/fs/nexus-scratch/tuxunlu/real_headv2"
NW, NH, R, T16 = 1280, 720, 256, 16
SCALES, SLICE_US, TAU_US = [4, 16, 64], 1_000, 5_000
N_SAMP = 24000
DEV = "cuda"


def build_one(mpath, lpath, dst):
    if os.path.exists(dst):
        return 0
    try:
        with np.load(mpath) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        lab = np.load(lpath).astype(np.int64)
    except Exception:
        return 0
    n = len(x)
    if len(lab) != n or n < 500:
        return 0
    is_bg = lab == 1                                    # 1 = scene
    n_bg, n_rn = int(is_bg.sum()), int((~is_bg).sum())
    if n_bg < 50 or n_rn < 50:
        return 0

    sx = (x.astype(np.int64) * R) // NW
    sy = (y.astype(np.int64) * R) // NH
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
    cell = (tb * R + sy) * R + sx

    order = np.argsort(cell, kind="stable")
    c, r = cell[order], (~is_bg)[order]
    b = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    cells, cnt = c[b], np.diff(np.r_[b, len(c)])
    rn_c = np.add.reduceat(r.astype(np.int64), b)
    mixed_cells = set(cells[(cnt - rn_c > 0) & (rn_c > 0)].tolist())
    is_mixed = np.fromiter((cc in mixed_cells for cc in cell), bool, n)

    rng = np.random.default_rng(abs(hash(os.path.basename(mpath))) % (2 ** 31))
    idx_m, idx_u = np.flatnonzero(is_mixed), np.flatnonzero(~is_mixed)
    take_m = min(len(idx_m), N_SAMP // 2)
    take_u = min(len(idx_u), N_SAMP - take_m)
    sel = np.concatenate([
        rng.choice(idx_m, take_m, replace=False) if take_m else np.empty(0, np.int64),
        rng.choice(idx_u, take_u, replace=False) if take_u else np.empty(0, np.int64),
    ]).astype(np.int64)
    if len(sel) < 100:
        return 0
    inv_p = np.concatenate([
        np.full(take_m, len(idx_m) / max(take_m, 1), np.float32),
        np.full(take_u, len(idx_u) / max(take_u, 1), np.float32)])

    xg = torch.from_numpy(x.astype(np.int64)).to(DEV)
    yg = torch.from_numpy(y.astype(np.int64)).to(DEV)
    tg = torch.from_numpy(t.astype(np.int64)).to(DEV)
    tc = tensor_gpu(xg, yg, tg, TAU_US, SCALES, NW, NH, SLICE_US)
    tcols = tc[torch.from_numpy(sel).to(DEV)].cpu().numpy().astype(np.float16)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, sel=sel.astype(np.int32), tcols=tcols,
                        lab=is_bg[sel].astype(np.int8), mixed=is_mixed[sel],
                        inv_p=inv_p, n_bg=np.int64(n_bg), n_rn=np.int64(n_rn))
    return 1


def main():
    jobs = []
    for pk in sorted(glob.glob(f"{PACK}/*/*/*.npz")):
        sc, lv, base = pk.split("/")[-3], pk.split("/")[-2], os.path.basename(pk)
        jobs.append((f"{S}/{sc}/merge_data/{lv}/{base}",
                     f"{S}/{sc}/labels/labels_{lv}/labels_{base}".replace(".npz", ".npy"),
                     f"{OUT}/{sc}/{lv}/{base}"))
    print(f"{len(jobs)} frames", flush=True)
    done = 0
    for i, (m, l, d) in enumerate(jobs):
        done += build_one(m, l, d)
        if i % 300 == 0:
            print(f"  {i}/{len(jobs)}  built {done}", flush=True)
    print(f"built {done} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
