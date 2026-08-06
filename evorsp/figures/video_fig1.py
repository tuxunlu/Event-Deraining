"""fig1 as a video: all 389 consecutive TEST frames, both rates, six models.

Layout mirrors fig1 exactly -- Input | Ground truth | DFFN | FourierMamba2D |
ORSPNet | ORSPNet+dil+bal | StreakNet | EvORSP-3T, one row per test rate.
Every output is rendered under the subset rule (pred AND input), each model at
its own protocol threshold. 10 fps -> ~39 s.

Written with cv2 (mp4v), then remuxed to H.264 with ffmpeg for compatibility.
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

import train_compare as TC


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
    raise ValueError(kind)


MODELS = [
    ("dffn",      "proto_dffn",          "dffn",      "DFFN (earlier)",           None, ""),
    ("fmamba",    "ckpt_fmamba",         "fmamba",    "FourierMamba2D (earlier)", None, "*"),
    ("orsp",      "proto_orsp",          "orsp",      "ORSPNet",                  None, ""),
    ("orsp_best", "proto_orsp_bal_dil",  "orsp_dil",  "ORSPNet+dil+bal",          None, ""),
    ("streaknet", "proto_streaknet_bal", "streaknet", "StreakNet K=127",          None, ""),
    ("evorsp",    "k3d_T4b3off",         "evorsp3t",  "EvORSP-3T (temporal+OFF)", None, ""),
]

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
T16 = f"{C.WORK / 'kitti_t16'}"
DEV = "cuda"
RATES = ["50mm", "150mm"]
FPS = 10
P = 256                     # panel size (native resolution)
HDR, LBL, GAP = 46, 88, 4   # header strip, left label strip, panel gap

nets, meta = {}, {}
for k, f, b, lbl, c, note in MODELS:
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = build(b)
    m.load_state_dict(ck["state_dict"])
    nets[k] = m.to(DEV).eval()
    meta[k] = {"tau": float(ck.get("test_tau", ck.get("tau", 0.5))),
               "label": lbl.replace("\n", " "), "note": note}
print("models loaded", flush=True)

files = {mm: sorted(glob.glob(f"{TC.ROOT}/merge_data/test/{mm}/*.npz")) for mm in RATES}
raws = sorted(glob.glob(f"{TC.ROOT}/raw_data/*.npz"))
N = min(len(raws), *(len(files[mm]) for mm in RATES))


def t16_planes(mm, basename):
    with np.load(f"{T16}/test/{mm}/{basename}") as d:
        on = np.unpackbits(d["on"])[:16 * P * P].reshape(16, P, P)
        off = np.unpackbits(d["off"])[:16 * P * P].reshape(16, P, P)
    on = torch.from_numpy(on.reshape(4, 4, P, P).max(1)).float()
    off = torch.from_numpy(off.reshape(4, 4, P, P).max(1)).float()
    return on.unsqueeze(0).to(DEV), off.unsqueeze(0).to(DEV)


@torch.no_grad()
def outputs(mm, i):
    f = files[mm][i]
    merge = TC.RainSet._img(f).unsqueeze(0).to(DEV)
    raw = TC.RainSet._img(raws[i]).unsqueeze(0).to(DEV)
    lit = (merge > 0.5).float()
    base = os.path.basename(f)
    panels = [lit[0, 0], (raw[0, 0] > 0.5).float()]
    for k, *_ in MODELS:
        if k == "evorsp":
            on, off = t16_planes(mm, base)
            pr = torch.sigmoid(nets[k](on, x_off=off))
        else:
            pr = torch.sigmoid(nets[k](merge))
        panels.append(((pr > meta[k]["tau"]).float() * lit)[0, 0])
    return [p.cpu().numpy() for p in panels]


COLS = ["Input (rainy)", "Ground truth"] + [meta[k]["label"] + meta[k]["note"]
                                            for k, *_ in MODELS]
W = LBL + len(COLS) * (P + GAP) + GAP
H = HDR + len(RATES) * (P + GAP) + GAP + 26
raw_path = f"{TMP}/fig1_video_raw.mp4"
vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
assert vw.isOpened(), "VideoWriter failed to open"

canvas0 = np.full((H, W, 3), 252, np.uint8)
for c, name in enumerate(COLS):                       # static header
    x = LBL + GAP + c * (P + GAP)
    for li, part in enumerate(name.split(" (")):
        txt = part if li == 0 else "(" + part
        cv2.putText(canvas0, txt, (x + 2, 18 + 16 * li),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
for r, mm in enumerate(RATES):
    y = HDR + GAP + r * (P + GAP)
    cv2.putText(canvas0, mm, (6, y + P // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (30, 30, 30), 2, cv2.LINE_AA)
cv2.putText(canvas0, "outputs restricted to input support (kept events are a "
            "subset of input events); each model at its protocol threshold",
            (LBL, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1,
            cv2.LINE_AA)

for i in range(N):
    canvas = canvas0.copy()
    cv2.putText(canvas, f"frame {i + 1}/{N}", (W - 150, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    for r, mm in enumerate(RATES):
        y = HDR + GAP + r * (P + GAP)
        for c, p in enumerate(outputs(mm, i)):
            x = LBL + GAP + c * (P + GAP)
            g = (255 - p * 255).astype(np.uint8)      # black events on white
            canvas[y:y + P, x:x + P] = g[:, :, None]
    vw.write(canvas)
    if (i + 1) % 50 == 0:
        print(f"  ...{i + 1}/{N}", flush=True)
        cv2.imwrite(f"{TMP}/video_still_{i+1:03d}.png", canvas)

vw.release()
os.system(f"ffmpeg -y -loglevel error -i {raw_path} -c:v libx264 -pix_fmt yuv420p "
          f"-crf 23 {OUT}/fig1_video.mp4")
print(f"done: {OUT}/fig1_video.mp4  ({N} frames, {N/FPS:.0f}s at {FPS} fps, "
      f"{W}x{H})")
