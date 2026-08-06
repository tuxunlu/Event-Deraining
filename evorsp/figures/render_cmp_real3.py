"""Real EVK4 comparison video: 4 held-out scenes, native 1280x720, per EVENT.

Fairness: both models shown were trained on PRE-Mamba's OWN within-scene split,
and every sequence rendered is a TEST sequence of that split for both. (Our
earlier real-data models used a different, scene-disjoint split, so they are
deliberately excluded here -- mixing splits would flatter one side.)

Panels (2x2): Input (rainy) | Ground truth (label 1 = scene)
              EvORSP-3T (28K params) | PRE-Mamba (265K params)

Label orientation was settled empirically, not assumed: `rain_k` is a recording
index, NOT an intensity (rain_1 averages 924K events/frame, rain_13 242K), so
the naive "label-1 fraction rises with k" reading is invalid. The decisive test
is cross-frame persistence -- scene structure recurs at the same pixels, rain
does not: label 1 persists at 0.54-0.55, label 0 at 0.18-0.24. Hence
label 1 = scene, label 0 = rain, and PRE-Mamba's predictions (which follow the
file convention, 0.8365 agreement) are rendered with keep = (pred == 1).
NOTE their config names classes ["background","rain"], i.e. inverted w.r.t.
these real files, so their printed SR/NR columns are swapped on real data
(DA is unaffected, being symmetric).
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob
import os
import sys

import cv2
import numpy as np
import torch

from rsp_3d import ORSPNet3D
from run_kitti_perevent import sample_at
from run_real_perevent import HeadV2 as HeadR
from gpu_feats import patch_gpu, tensor_gpu

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
S = f"{C.REAL_SRC}"
PM = f"{C.PM_REAL}"
DEV = "cuda"
NW, NH = 1280, 720
R, T16 = 256, 16
FPS, NPER = 8, 60
SEQS = [("scene1", "rain_2"), ("scene2", "rain_5"),
        ("scene3", "rain_9"), ("scene4", "rain_13")]

# fixed-target model: trained on PRE-Mamba's split with the
# count-majority, polarity-complete supervision (test event-DA 0.8066
# vs 0.7985 for the old ON-only OR target, vs PRE-Mamba 0.7708)
# BEST real model on PRE-Mamba's own split: trunk + per-event head v3,
# test event-DA 0.8444 (trunk-only 0.8066, PRE-Mamba 0.7708).
hck = torch.load(f"{TMP}/realph_theirs.pt", map_location="cpu")
net = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                out_chans=1)
net.load_state_dict(hck["trunk"])
net = net.to(DEV).eval()
head = HeadR(feat_dim=128).to(DEV)
head.load_state_dict(hck["head"])
head.eval()
_f, _b = {}, {}
net.out_proj.register_forward_pre_hook(lambda m, i: _f.__setitem__("f", i[0]))
for _bi, _blk in enumerate(net.blocks):
    _blk.register_forward_hook(lambda m, i, o, bi=_bi: _b.__setitem__(bi, o))
print("EvORSP-3T/E + per-event head (their split) loaded", flush=True)


def draw(x, y, p, keep=None):
    if keep is not None:
        x, y, p = x[keep], y[keep], p[keep]
    img = np.full((NH, NW, 3), 255, np.uint8)
    for pol, col in ((1, np.array([50, 50, 210], np.int32)),
                     (0, np.array([210, 110, 30], np.int32))):
        s = (p == 1) if pol == 1 else (p != 1)
        if not s.any():
            continue
        cnt = np.zeros((NH, NW), np.int32)
        np.add.at(cnt, (y[s], x[s]), 1)
        m = cnt > 0
        a = np.clip(0.55 + cnt[m] / 6.0, 0.55, 1.0)[:, None]
        img[m] = (img[m] * (1 - a) + col[None, :] * a).astype(np.uint8)
    return img


@torch.no_grad()
def ours_keep(x, y, t, p):
    sx = (x.astype(np.int64) * R) // NW
    sy = (y.astype(np.int64) * R) // NH
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
    on = np.zeros((T16, R * R), bool)
    off = np.zeros((T16, R * R), bool)
    s = p == 1
    on[tb[s], sy[s] * R + sx[s]] = True
    off[tb[~s], sy[~s] * R + sx[~s]] = True
    on4 = torch.from_numpy(on.reshape(T16, R, R).reshape(4, 4, R, R).max(1)
                           ).float()[None].to(DEV)
    off4 = torch.from_numpy(off.reshape(T16, R, R).reshape(4, 4, R, R).max(1)
                            ).float()[None].to(DEV)
    lm = net(on4, x_off=off4)
    fm = torch.cat([_f["f"]] + [_b[i] for i in range(3)], 1)
    tn = ((t - t0) / span).astype(np.float32)
    xg = torch.from_numpy(x.astype(np.int64)).to(DEV)
    yg = torch.from_numpy(y.astype(np.int64)).to(DEV)
    tg = torch.from_numpy(t.astype(np.int64)).to(DEV)
    pg = torch.from_numpy(p.astype(np.int64)).to(DEV)
    tns = torch.from_numpy(tn)[None].to(DEV)
    xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
    ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
    lv = sample_at(lm[:, None], xs, ys, tns)
    fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                   xs, ys, tns)
    # EVK4 stamps are MICROseconds: slice 1000 us, tau 5000 us
    pv = patch_gpu(xg, yg, tns[0], pg, NW, NH)[None]
    tc = tensor_gpu(xg, yg, tg, 5_000, [4, 16, 64], NW, NH, 1_000)[None]
    ev = torch.sigmoid(head(lv, fv, pv, tc, tns[..., None]))[0, :, 0]
    return (ev > float(ev.mean())).cpu().numpy()


COLS = ["Input (rainy, native 1280x720)", "Ground truth (labelled scene events)",
        "EvORSP-3T/E + per-event head  53,630 params  (0.8444)",
        "PRE-Mamba (ICCV'25)  264,632 params  (0.7708)"]
GAP, HDR, FTR = 8, 56, 36
Wc = GAP + 2 * (NW + GAP)
Hc = HDR + 2 * (NH + 24 + GAP) + FTR
raw = f"{TMP}/cmp_real3_raw.mp4"
vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
assert vw.isOpened()

for si, (sc, lv) in enumerate(SEQS):
    files = sorted(glob.glob(f"{S}/{sc}/merge_data/{lv}/*.npz"))
    step = max(1, len(files) // NPER)
    n_ok = 0
    for i, f in enumerate(files[::step][:NPER]):
        base = os.path.basename(f)
        lp = f"{S}/{sc}/labels/labels_{lv}/labels_{base}".replace(".npz", ".npy")
        pp = f"{PM}/{sc}/{lv}/{base.replace('.npz', '')}_pred.npy"
        if not (os.path.exists(lp) and os.path.exists(pp)):
            continue
        with np.load(f) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        lab = np.load(lp).astype(np.int64)
        pred = np.load(pp)
        if len(lab) != len(x) or len(x) < 200 or len(pred) < len(x):
            continue
        panels = [draw(x, y, p),
                  draw(x, y, p, lab == 1),
                  draw(x, y, p, ours_keep(x, y, t, p)),
                  draw(x, y, p, pred[:len(x)] == 1)]

        canvas = np.full((Hc, Wc, 3), 252, np.uint8)
        cv2.putText(canvas, f"REAL EVK4 rain rig -- {sc} / {lv}   "
                    f"(TEST sequence of PRE-Mamba's own split, held out for "
                    f"both models)   {si + 1}/{len(SEQS)}",
                    (GAP + 4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (30, 30, 30), 2, cv2.LINE_AA)
        for c, (name, panel) in enumerate(zip(COLS, panels)):
            r, cc = divmod(c, 2)
            x0 = GAP + cc * (NW + GAP)
            y0 = HDR + r * (NH + 24 + GAP)
            cv2.putText(canvas, name, (x0 + 2, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 90, 90), 1,
                        cv2.LINE_AA)
            canvas[y0 + 24:y0 + 24 + NH, x0:x0 + NW] = panel
        cv2.putText(canvas, "kept EVENTS at native sensor coordinates. "
                    "red = ON, blue = OFF, intensity = event count.  "
                    "label 1 = scene (verified by cross-frame persistence: "
                    "0.54 vs 0.20 for rain).",
                    (GAP + 4, Hc - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    (110, 110, 110), 1, cv2.LINE_AA)
        vw.write(canvas)
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"  {sc}/{lv} {n_ok}", flush=True)
    print(f"  {sc}/{lv}: {n_ok} frames", flush=True)
vw.release()
os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
          f"-crf 18 {OUT}/cmp_real_allmodels.mp4")
print(f"wrote {OUT}/cmp_real_allmodels.mp4  ({Wc}x{Hc})", flush=True)
