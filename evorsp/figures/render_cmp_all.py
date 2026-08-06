"""Every model, one canvas: the project's whole arc on the same test frames.

Panels (2 rows x 5), native 460x352, kept EVENTS at true sensor coordinates:

  Input (rainy)          Ground truth (clean)     DFFN                 72,074p
  ORSPNet + dil          StreakNet K=127          FourierMamba2D   23,592,081p
  EvORSP-3T (old target) EvORSP-3T/E trunk        + per-event head v3  54,289p
  PRE-Mamba (ICCV'25)  264,632p

Thresholds are each model's own published operating point, so this is what each
model actually does rather than a re-tuned version:
  * the four earlier models: their protocol tau from the 50-epoch leaderboard;
  * EvORSP-3T (old target): its protocol tau 0.70;
  * EvORSP-3T/E trunk and +head: per-frame SELF-PRIOR tau (label-free);
  * PRE-Mamba: its own published argmax.
The self-prior/argmax difference favours us and is stated in the footer.

WHAT TO LOOK FOR. The four earlier models render almost purely RED: they read
the ON-only eFFT pipeline, so their `lit` mask has no entry for pixels holding
only OFF events and every blue event is structurally discarded. That is the
supervision defect the campaign found, visible directly.
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
from rsp_3d import ORSPNet3D
from run_kitti_perevent import sample_at
from run_kitti_headv3 import HeadV2 as HeadV3, multiscale_patch
from gpu_feats import patch_gpu, tensor_gpu

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
SRC = f"{C.KITTI_SRC}"
PACK = f"{C.KITTI_PACK}/test"
PM = f"{C.PM_SYNTH}"
DEV = "cuda"
NW, NH, R, T16 = 460, 352, 256, 16
FPS, NPER = 8, 100
LEVELS = ["50mm", "150mm"]


def build(kind):
    if kind == "dffn":
        from model.DynamicFourierFilterNet import DynamicFourierFilterNet
        return DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32,
                                       num_blocks=4)
    if kind == "fmamba":
        from model.FourierMamba2D import FourierMamba2D
        return FourierMamba2D(in_chans=1, out_chans=1, dim=32,
                              num_blocks=[2, 2, 2, 2])
    if kind == "orsp_dil":
        from rsp_model_v2 import ORSPNet
        return ORSPNet(dilations=(1, 8, 32, 64))
    if kind == "streaknet":
        from rsp_streak import StreakNet
        return StreakNet(K=127, use_strip=True, use_rate=True,
                         use_darkmask=True)
    raise ValueError(kind)


PIX = [("DFFN  72,074p", "proto_dffn", "dffn"),
       ("ORSPNet + dil  36,782p", "proto_orsp_bal_dil", "orsp_dil"),
       ("StreakNet K=127  52,043p", "proto_streaknet_bal", "streaknet"),
       ("FourierMamba2D  23,592,081p", "proto_fmamba_ddp", "fmamba")]
nets, taus = {}, {}
for lbl, f, kind in PIX:
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = build(kind)
    m.load_state_dict(ck["state_dict"])
    nets[lbl] = m.to(DEV).eval()
    taus[lbl] = float(ck.get("test_tau", ck.get("tau", 0.5)))

ck = torch.load(f"{TMP}/k3d_T4b3off.pt", map_location="cpu")
old = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
old.load_state_dict(ck["state_dict"])
old, TAU_OLD = old.to(DEV).eval(), float(ck.get("tau", 0.7))

# standalone trunk-only operating point: the model that is actually DEPLOYED
# rate-invariantly (self-prior 0.9332). NOT phv3's trunk -- that one was
# fine-tuned end-to-end with the head, so it emits a parameter for the head
# rather than a calibrated standalone decision, and thresholding it alone
# misrepresents the trunk-only configuration.
sck = torch.load(f"{TMP}/ctx_f4o16_c2.pt", map_location="cpu")
solo = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                 out_chans=16, n_extra=4)
solo.load_state_dict(sck["state_dict"])
solo = solo.to(DEV).eval()

hck = torch.load(f"{TMP}/phv3.pt", map_location="cpu")
trunk = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=16, n_extra=4)
trunk.load_state_dict(hck["trunk"])
trunk = trunk.to(DEV).eval()
head = HeadV3(feat_dim=128).to(DEV)
head.load_state_dict(hck["head"])
head.eval()
_f, _b = {}, {}
trunk.out_proj.register_forward_pre_hook(lambda m, i: _f.__setitem__("f", i[0]))
for bi, blk in enumerate(trunk.blocks):
    blk.register_forward_hook(lambda m, i, o, bi=bi: _b.__setitem__(bi, o))
print("all models loaded", flush=True)


def draw(x, y, p, keep=None):
    if keep is not None:
        x, y, p = x[keep], y[keep], p[keep]
    img = np.full((NH, NW, 3), 255, np.uint8)
    for sel, col in (((p == 1), np.array([50, 50, 210], np.int32)),
                     ((p != 1), np.array([210, 110, 30], np.int32))):
        if not sel.any():
            continue
        cnt = np.zeros((NH, NW), np.int32)
        np.add.at(cnt, (y[sel], x[sel]), 1)
        m = cnt > 0
        a = np.clip(0.55 + cnt[m] / 6.0, 0.55, 1.0)[:, None]
        img[m] = (img[m] * (1 - a) + col[None, :] * a).astype(np.uint8)
    return img


def planes(path):
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
    tn = ((t - t0) / span).astype(np.float32)
    out = {}

    merge = TC.RainSet._img(f"{TC.ROOT}/merge_data/test/{mm}/{base}"
                            ).unsqueeze(0).to(DEV)
    lit2d = merge[0, 0] > 0.5
    for lbl in nets:
        pr = torch.sigmoid(nets[lbl](merge))[0, 0]
        out[lbl] = ((pr > taus[lbl]) & lit2d).cpu().numpy()[sy, sx]

    on, off = planes(f"{PACK}/{mm}/{base}")
    on4 = torch.from_numpy(on.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
    off4 = torch.from_numpy(off.reshape(4, 4, R, R).max(1)).float()[None].to(DEV)
    lit_cell = torch.from_numpy((on | off).astype(bool)).to(DEV)

    pr = torch.sigmoid(old(on4, x_off=off4))[0, 0]
    out["EvORSP-3T  old target  28,060p"] = (pr > TAU_OLD).cpu().numpy()[sy, sx]

    idx = int(base.split(".")[0])
    ex = []
    for k in (1, 2):
        pv = f"{PACK}/{mm}/{max(idx - k, 0):010d}.npz"
        pon, poff = planes(pv if os.path.exists(pv) else f"{PACK}/{mm}/{base}")
        ex.append(pon.max(0)[None].astype(np.float32))
        ex.append(poff.max(0)[None].astype(np.float32))
    ex = torch.from_numpy(np.concatenate(ex, 0))[None].to(DEV)

    prs = torch.sigmoid(solo(on4, x_off=off4, x_extra=ex))[0]
    tau = float(prs[lit_cell].mean())
    out["EvORSP-3T/E trunk  28,719p"] = (prs > tau).cpu().numpy()[tb, sy, sx]

    lm = trunk(on4, x_off=off4, x_extra=ex)

    fm = torch.cat([_f["f"]] + [_b[i] for i in range(3)], 1)
    xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
    ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
    tns = torch.from_numpy(tn)[None].to(DEV)
    lv = sample_at(lm[:, None], xs, ys, tns)
    fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                   xs, ys, tns)
    xg = torch.from_numpy(x.astype(np.int64)).to(DEV)
    yg = torch.from_numpy(y.astype(np.int64)).to(DEV)
    tg = torch.from_numpy(t).to(DEV)
    pg = torch.from_numpy(p.astype(np.int64)).to(DEV)
    pv2 = patch_gpu(xg, yg, tns[0], pg, NW, NH)[None]
    tc = tensor_gpu(xg, yg, tg, 5_000_000, [4, 16, 64], NW, NH,
                    1_000_000)[None]
    ev = torch.sigmoid(head(lv, fv, pv2, tc, tns[..., None]))[0, :, 0]
    out["+ per-event head v3  54,289p"] = (ev > float(ev.mean())).cpu().numpy()

    pp = f"{PM}/{mm}_{base.replace('.npz', '')}.npy"
    if os.path.exists(pp):
        pred = np.load(pp)
        out["PRE-Mamba (ICCV'25)  264,632p"] = (pred[:len(x)] == 0) \
            if len(pred) >= len(x) else np.ones(len(x), bool)
    else:
        out["PRE-Mamba (ICCV'25)  264,632p"] = np.ones(len(x), bool)
    return out


COLS = ["Input (rainy)", "Ground truth (clean)", "DFFN  72,074p",
        "ORSPNet + dil  36,782p", "StreakNet K=127  52,043p",
        "FourierMamba2D  23,592,081p", "EvORSP-3T  old target  28,060p",
        "EvORSP-3T/E trunk  28,719p", "+ per-event head v3  54,289p",
        "PRE-Mamba (ICCV'25)  264,632p"]
GAP, HDR, FTR = 6, 52, 34
Wc = GAP + 5 * (NW + GAP)
Hc = HDR + 2 * (NH + 22 + GAP) + FTR
raw = f"{TMP}/cmp_all2_raw.mp4"
vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
assert vw.isOpened()

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
            cx, cy, cp = d["x"], d["y"], d["p"]
        if len(x) < 200:
            continue
        km = keeps(mm, base, x, y, t, p)
        panels = [draw(x, y, p), draw(cx, cy, cp)] + \
                 [draw(x, y, p, km[c]) for c in COLS[2:]]

        canvas = np.full((Hc, Wc, 3), 252, np.uint8)
        cv2.putText(canvas, f"KITTI synthetic rain {mm}   (protocol TEST rate, "
                    f"unseen by every model)   sequence {li + 1}/{len(LEVELS)}",
                    (GAP + 4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (30, 30, 30), 2, cv2.LINE_AA)
        for c, (name, panel) in enumerate(zip(COLS, panels)):
            r, cc = divmod(c, 5)
            x0 = GAP + cc * (NW + GAP)
            y0 = HDR + r * (NH + 22 + GAP)
            cv2.putText(canvas, name, (x0 + 2, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 90, 90), 1,
                        cv2.LINE_AA)
            canvas[y0 + 22:y0 + 22 + NH, x0:x0 + NW] = panel
        cv2.putText(canvas, "kept EVENTS at native 460x352. red = ON, blue = OFF, "
                    "intensity = count.  The four earlier models read the ON-only "
                    "eFFT input, so OFF events are structurally discarded -- that "
                    "is why they render almost purely red.  Each model uses its "
                    "OWN operating point (earlier models: protocol tau; ours: "
                    "label-free self-prior; PRE-Mamba: published argmax).",
                    (GAP + 4, Hc - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (110, 110, 110), 1, cv2.LINE_AA)
        vw.write(canvas)
        if i % 25 == 0:
            print(f"  {mm} {i}", flush=True)
vw.release()
os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
          f"-crf 18 {OUT}/cmp_all_models.mp4")
print(f"wrote {OUT}/cmp_all_models.mp4  ({Wc}x{Hc})", flush=True)
