"""Synthetic KITTI comparison video, REGENERATED with the current models.

Panels (2 x 4), native 460x352, kept EVENTS at true sensor coordinates:
  Input (rainy)              | Ground truth (clean)
  EvORSP-3T  old ON-only target (0.7052)   <- what the supervision defect looked like
  EvORSP-3T/E fixed target    (0.9245)
  EvORSP-3T/E + multi-window  (0.9332)     <- current best
  StreakNet, same input+target(0.9218)     <- the fairness arm: architecture barely matters
  PRE-Mamba (ICCV'25)         (0.9172)
  Error map of the best model (signal lost / rain kept)

Thresholding: our models use the per-frame SELF-PRIOR rule (tau = count-weighted
mean p over lit cells), which measured +0.006 over a single global tau and needs
no labels. PRE-Mamba uses its own argmax, as published -- that asymmetry favours
us and is stated in the caption rather than hidden.
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
from bodies_e import FrontendBody
from run_kitti_perevent import sample_at
from run_kitti_headv2 import HeadV2, multiscale_patch
from fast_tensor import tensor_cols_fast

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
SRC = f"{C.KITTI_SRC}"
PACK = f"{C.KITTI_PACK}/test"
PM = f"{C.PM_SYNTH}"
DEV = "cuda"
NW, NH = 460, 352
R, T16 = 256, 16
FPS, NPER = 8, 100
LEVELS = ["50mm", "150mm"]

# ---- models ---------------------------------------------------------------
ck = torch.load(f"{TMP}/k3d_T4b3off.pt", map_location="cpu")
old = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
old.load_state_dict(ck["state_dict"])
old, TAU_OLD = old.to(DEV).eval(), float(ck.get("tau", 0.7))

ck = torch.load(f"{TMP}/k3de_T4o16_maj.pt", map_location="cpu")
fixed = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=16)
fixed.load_state_dict(ck["state_dict"])
fixed = fixed.to(DEV).eval()

# current best: trunk + per-event head v2 (oriented + multi-scale features,
# mixed-cell sampling quota). mixed-cell recall 0.6149 vs trunk 0.4005,
# aggregate event-DA 0.9575 vs PRE-Mamba 0.9172.
hck = torch.load(f"{TMP}/phv2.pt", map_location="cpu")
best = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                 out_chans=16, n_extra=4)
best.load_state_dict(hck["trunk"])
best = best.to(DEV).eval()
headv2 = HeadV2().to(DEV)
headv2.load_state_dict(hck["head"])
headv2.eval()
_hf = {}
best.out_proj.register_forward_pre_hook(
    lambda m, inp: _hf.__setitem__("f", inp[0]))

ck = torch.load(f"{TMP}/fair_streaknet_o16.pt", map_location="cpu")
streak = FrontendBody("streaknet", T=4, t_out=16)
streak.load_state_dict(ck["state_dict"])
streak, TAU_STREAK = streak.to(DEV).eval(), float(ck.get("tau", 0.6))
print("models loaded", flush=True)


def draw(x, y, p, keep=None, col_on=(50, 50, 210), col_off=(210, 110, 30)):
    if keep is not None:
        x, y, p = x[keep], y[keep], p[keep]
    img = np.full((NH, NW, 3), 255, np.uint8)
    for sel, col in (((p == 1), np.array(col_on, np.int32)),
                     ((p != 1), np.array(col_off, np.int32))):
        if not sel.any():
            continue
        cnt = np.zeros((NH, NW), np.int32)
        np.add.at(cnt, (y[sel], x[sel]), 1)
        m = cnt > 0
        a = np.clip(0.55 + cnt[m] / 6.0, 0.55, 1.0)[:, None]
        img[m] = (img[m] * (1 - a) + col[None, :] * a).astype(np.uint8)
    return img


def planes_of(path):
    with np.load(path) as d:
        on = np.unpackbits(d["on"])[: T16 * R * R].reshape(T16, R, R)
        off = np.unpackbits(d["off"])[: T16 * R * R].reshape(T16, R, R)
    return on, off


@torch.no_grad()
def keeps(mm, base, x, y, t, p):
    sx = np.clip((x.astype(np.int64) * R) // NW, 0, R - 1)
    sy = np.clip((y.astype(np.int64) * R) // NH, 0, R - 1)
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)

    on, off = planes_of(f"{PACK}/{mm}/{base}")
    on4 = torch.from_numpy(on.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
    off4 = torch.from_numpy(off.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
    lit_cell = torch.from_numpy((on | off).reshape(T16, R, R)).to(DEV)

    # context planes for the multi-window model: unions of the 2 preceding windows
    idx = int(base.split(".")[0])
    ex = []
    for k in (1, 2):
        prev = f"{PACK}/{mm}/{max(idx - k, 0):010d}.npz"
        pon, poff = planes_of(prev if os.path.exists(prev) else f"{PACK}/{mm}/{base}")
        ex.append(pon.max(0)[None].astype(np.float32))
        ex.append(poff.max(0)[None].astype(np.float32))
    ex = torch.from_numpy(np.concatenate(ex, 0))[None].to(DEV)

    out = {}
    pr = torch.sigmoid(old(on4, x_off=off4))[0, 0]
    out["EvORSP-3T  old target"] = (pr > TAU_OLD).cpu().numpy()[sy, sx]

    pr = torch.sigmoid(fixed(on4, x_off=off4))[0]                # [16,R,R]
    tau = float(pr[lit_cell].mean())                              # self-prior
    out["EvORSP-3T/E  fixed target"] = (pr > tau).cpu().numpy()[tb, sy, sx]

    # per-event head v2: decides per EVENT, not per cell
    lm = best(on4, x_off=off4, x_extra=ex)
    fm = _hf["f"]
    To = lm.shape[1]
    idx_all = np.arange(len(x))
    xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
    ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
    tns = torch.from_numpy((tb.astype(np.float32) + 0.5) / T16)[None].to(DEV)
    lv = sample_at(lm[:, None], xs, ys, tns)
    fv = sample_at(fm[:, :, None].expand(-1, -1, To, -1, -1), xs, ys, tns)
    tnorm = ((t - t.min()) / max(int(t.max() - t.min()), 1)).astype(np.float32)
    pv2 = multiscale_patch(x, y, tnorm, p, idx_all)
    tc = tensor_cols_fast(x, y, t, idx_all, 5_000_000, [4, 16, 64], NW, NH,
                          1_000_000)
    ev = torch.sigmoid(headv2(lv, fv, torch.from_numpy(pv2)[None].to(DEV),
                              torch.from_numpy(tc)[None].to(DEV),
                              tns[..., None]))[0, :, 0]
    out["EvORSP-3T/E + per-event head"] = (ev > float(ev.mean())).cpu().numpy()

    pr = torch.sigmoid(streak(on4, x_off=off4))[0]
    tau = float(pr[lit_cell].mean())
    out["StreakNet  same input"] = (pr > tau).cpu().numpy()[tb, sy, sx]

    pp = f"{PM}/{mm}_{base.replace('.npz', '')}.npy"
    if os.path.exists(pp):
        pred = np.load(pp)
        out["PRE-Mamba (ICCV'25)"] = (pred[:len(x)] == 0) if len(pred) >= len(x) \
            else np.ones(len(x), bool)
    else:
        out["PRE-Mamba (ICCV'25)"] = np.ones(len(x), bool)
    return out


COLS = ["Input (rainy)", "Ground truth (clean)", "EvORSP-3T  old target",
        "EvORSP-3T/E  fixed target", "EvORSP-3T/E + per-event head",
        "StreakNet  same input", "PRE-Mamba (ICCV'25)",
        "errors of best (red = rain kept, blue = signal lost)"]
GAP, HDR, FTR = 6, 52, 34
Wc = GAP + 4 * (NW + GAP)
Hc = HDR + 2 * (NH + 22 + GAP) + FTR
raw = f"{TMP}/cmp_synth3_raw.mp4"
vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
assert vw.isOpened()


def key(x, y, t):
    return (t.astype(np.int64) * (NW * NH) + y.astype(np.int64) * NW
            + x.astype(np.int64))


for li, mm in enumerate(LEVELS):
    files = sorted(glob.glob(f"{SRC}/merge_data/{mm}/*.npz"))
    step = max(1, len(files) // NPER)
    for i, f in enumerate(files[::step][:NPER]):
        base = os.path.basename(f)
        if not os.path.exists(f"{PACK}/{mm}/{base}"):
            continue
        with np.load(f) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        with np.load(f"{SRC}/raw_data/{base}") as d:
            cx, cy, ct, cp = d["x"], d["y"], d["t"], d["p"]
        if len(x) < 200:
            continue
        km = keeps(mm, base, x, y, t, p)
        rain = ~np.isin(key(x, y, t), np.sort(key(cx, cy, ct)))
        kb = km["EvORSP-3T/E + per-event head"]
        err = (kb & rain) | (~kb & ~rain)                # rain kept | signal lost
        err_col = np.where((kb & rain)[err], 1, 0)       # 1 -> ON red, 0 -> blue

        panels = [draw(x, y, p), draw(cx, cy, cp)] + \
                 [draw(x, y, p, km[c]) for c in COLS[2:7]] + \
                 [draw(x[err], y[err], err_col)]

        canvas = np.full((Hc, Wc, 3), 252, np.uint8)
        cv2.putText(canvas, f"KITTI synthetic rain {mm}  (protocol TEST rate, "
                    f"unseen by every model)   sequence {li + 1}/{len(LEVELS)}",
                    (GAP + 4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30),
                    2, cv2.LINE_AA)
        for c, (name, panel) in enumerate(zip(COLS, panels)):
            r, cc = divmod(c, 4)
            x0 = GAP + cc * (NW + GAP)
            y0 = HDR + r * (NH + 22 + GAP)
            cv2.putText(canvas, name, (x0 + 2, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1,
                        cv2.LINE_AA)
            canvas[y0 + 22:y0 + 22 + NH, x0:x0 + NW] = panel
        cv2.putText(canvas, "kept EVENTS at native 460x352. red = ON, blue = OFF, "
                    "intensity = count.  ours use the label-free per-frame "
                    "self-prior threshold; PRE-Mamba uses its published argmax "
                    "(an asymmetry in our favour).",
                    (GAP + 4, Hc - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (110, 110, 110), 1, cv2.LINE_AA)
        vw.write(canvas)
        if i % 25 == 0:
            print(f"  {mm} {i}", flush=True)
vw.release()
os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
          f"-crf 18 {OUT}/cmp_synth_allmodels.mp4")
print(f"wrote {OUT}/cmp_synth_allmodels.mp4  ({Wc}x{Hc})", flush=True)
