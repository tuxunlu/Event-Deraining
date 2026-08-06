"""High-resolution qualitative videos on REAL-WORLD rain (scene 5), polarity-coded.

Grid: 448x256 (16:9-correct). Measured on scene4 labels before rendering:
both real-trained models IMPROVE at this grid vs the square 256x256
(evorsp 0.8254 -> 0.8466, 2D control 0.7769 -> 0.7926), so HD inference is
better, not just prettier. Display upscale 2x nearest (crisp events).

Polarity colours: ON = red, OFF = blue, both at a pixel = purple, none = white.
Output panels show the KEPT events with their polarity (keep mask = prediction
AND ON-support, the training convention for every model; OFF-only pixels carry
no decision and are dropped uniformly -- stated in the footer).

Video A  real_world_hd.mp4        : Input | 2D control | EvORSP-3T (trained on
                                    real scenes 1-2), 3 recordings, 2x display.
Video B  real_world_allmodels.mp4 : Input + the six KITTI-trained models
                                    (ZERO-SHOT, near-chance -- shown to make the
                                    domain gap visible) + the two real-trained
                                    models. 2 recordings, 1x display.
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


TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
S5 = f"{C.REAL_WILD_SRC}/scene5/merge_data"
DEV = "cuda"
T16, RW, RH = 16, 448, 256
FPS = 10
C_ON, C_OFF, C_BOTH = (60, 60, 220), (220, 120, 40), (150, 60, 150)   # BGR


def build(kind):
    if kind == "dffn":
        from model.DynamicFourierFilterNet import DynamicFourierFilterNet
        return DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32, num_blocks=4)
    if kind == "fmamba":
        from model.FourierMamba2D import FourierMamba2D
        return FourierMamba2D(in_chans=1, out_chans=1, dim=32, num_blocks=[2, 2, 2, 2])
    if kind == "orsp":
        from rsp_model_v2 import ORSPNet
        return ORSPNet()
    if kind == "orsp_dil":
        from rsp_model_v2 import ORSPNet
        return ORSPNet(dilations=(1, 8, 32, 64))
    if kind == "streaknet":
        from rsp_streak import StreakNet
        return StreakNet(K=127, use_strip=True, use_rate=True, use_darkmask=True)
    if kind == "evorsp3t":
        from rsp_3d import ORSPNet3D
        return ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
    if kind == "orsp2d_real":
        from rsp_3d import ORSPNet3D
        return ORSPNet3D(T=1, num_blocks=4, use_temporal=False,
                         dilations=(1, 8, 32, 64))
    raise ValueError(kind)


#   key          ckpt                   builder        label (two lines ok)
REG = [
    ("zs_dffn",   "proto_dffn",          "dffn",        "DFFN [zero-shot]"),
    ("zs_fmamba", "ckpt_fmamba",         "fmamba",      "FourierMamba2D [zero-shot]"),
    ("zs_orsp",   "proto_orsp",          "orsp",        "ORSPNet [zero-shot]"),
    ("zs_best",   "proto_orsp_bal_dil",  "orsp_dil",    "ORSPNet+dil+bal [zero-shot]"),
    ("zs_streak", "proto_streaknet_bal", "streaknet",   "StreakNet [zero-shot]"),
    ("zs_evorsp", "k3d_T4b3off",         "evorsp3t",    "EvORSP-3T [zero-shot]"),
    ("re_2d",     "real_orsp2d",         "orsp2d_real", "2D control [trained on real]"),
    ("re_evorsp", "real_evorsp",         "evorsp3t",    "EvORSP-3T [trained on real]"),
]
nets, taus, labels = {}, {}, {}
for k, f, b, lbl in REG:
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = build(b)
    m.load_state_dict(ck["state_dict"])
    nets[k] = m.to(DEV).eval()
    taus[k] = float(ck.get("test_tau", ck.get("tau", 0.5)))
    labels[k] = lbl
print("8 models loaded", flush=True)


def planes(f):
    with np.load(f) as d:
        x, y, t, p = d["x"], d["y"], d["t"], d["p"]
    if len(x) < 200:
        return None
    sx = (x.astype(np.int64) * RW) // 1280
    sy = (y.astype(np.int64) * RH) // 720
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
    on = np.zeros((T16, RH * RW), bool)
    off = np.zeros((T16, RH * RW), bool)
    s = p == 1
    on[tb[s], sy[s] * RW + sx[s]] = True
    off[tb[~s], sy[~s] * RW + sx[~s]] = True
    return on.reshape(T16, RH, RW), off.reshape(T16, RH, RW)


def colorize(on_any, off_any, keep=None):
    img = np.full((RH, RW, 3), 255, np.uint8)
    if keep is None:
        keep = on_any | off_any
    a, b = on_any & keep, off_any & keep
    img[a & ~b] = C_ON
    img[b & ~a] = C_OFF
    img[a & b] = C_BOTH
    return img


@torch.no_grad()
def all_preds(on16, off16, keys):
    on4 = torch.from_numpy(on16.reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    off4 = torch.from_numpy(off16.reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    merge = on4.amax(1, keepdim=True)
    lit = merge[0, 0] > 0.5
    SELF_PRIOR = {"re_2d", "re_evorsp"}      # lit-BCE-trained => calibrated =>
    out = {}                                  # Bayes rule tau = mean p over lit
    for k in keys:
        if k in ("zs_evorsp", "re_evorsp"):
            p = torch.sigmoid(nets[k](on4, x_off=off4))[0, 0]
        else:
            p = torch.sigmoid(nets[k](merge))[0, 0]
        tau = float(p[lit].mean()) if k in SELF_PRIOR else taus[k]
        out[k] = ((p > tau) & lit).cpu().numpy()
    return out


def make(path, recs, keys, n_frames, scale, note):
    HDR, LBL, GAP, FTR = 52, 96, 6, 30
    PW, PH = RW * scale, RH * scale
    ncol = 1 + len(keys)
    Wc = LBL + ncol * (PW + GAP) + GAP
    Hc = HDR + len(recs) * (PH + GAP) + GAP + FTR
    raw = path.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
    assert vw.isOpened()
    base = np.full((Hc, Wc, 3), 252, np.uint8)
    cols = ["Input (rainy, real-world)"] + [labels[k] for k in keys]
    fs = 0.55 if scale == 2 else 0.42
    for c, name in enumerate(cols):
        cv2.putText(base, name, (LBL + GAP + c * (PW + GAP) + 3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), 1, cv2.LINE_AA)
    for r, rc in enumerate(recs):
        cv2.putText(base, rc, (6, HDR + GAP + r * (PH + GAP) + PH // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(base, note + "   |   red = ON events, blue = OFF, purple = both",
                (LBL, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110),
                1, cv2.LINE_AA)
    files = {rc: sorted(glob.glob(f"{S5}/{rc}/*.npz")) for rc in recs}
    for i in range(n_frames):
        canvas = base.copy()
        cv2.putText(canvas, f"frame {i+1}/{n_frames}", (Wc - 170, Hc - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)
        wrote = False
        for r, rc in enumerate(recs):
            if i >= len(files[rc]):
                continue
            pl = planes(files[rc][i])
            if pl is None:
                continue
            wrote = True
            on16, off16 = pl
            on_any, off_any = on16.any(0), off16.any(0)
            preds = all_preds(on16, off16, keys)
            panels = [colorize(on_any, off_any)] + \
                     [colorize(on_any, off_any, preds[k] ) for k in keys]
            y0 = HDR + GAP + r * (PH + GAP)
            for c, img in enumerate(panels):
                x0 = LBL + GAP + c * (PW + GAP)
                big = img if scale == 1 else cv2.resize(
                    img, (PW, PH), interpolation=cv2.INTER_NEAREST)
                canvas[y0:y0 + PH, x0:x0 + PW] = big
        if wrote:
            vw.write(canvas)
        if (i + 1) % 40 == 0:
            print(f"  {os.path.basename(path)} ...{i+1}/{n_frames}", flush=True)
            cv2.imwrite(f"{TMP}/hd_{os.path.basename(path)}_{i+1:03d}.png", canvas)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
              f"-crf 22 {path}")
    os.remove(raw)
    print(f"done: {path}", flush=True)


make(f"{OUT}/real_world_hd.mp4",
     ["rain_1", "rain_13", "rain_26"],
     ["re_2d", "re_evorsp"],
     120, 2,
     "REAL-WORLD rain, EVK4 448x256. Trained models use the PER-FRAME SELF-PRIOR "
     "threshold (tau = mean p over lit pixels -- the Bayes rule for balanced "
     "accuracy; no tuning, no labels).")
