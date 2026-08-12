"""Tour videos: every real recording as a sequential chapter.

  real_world_tour.mp4 : all 26 scene-5 real-world recordings, ~8 s each.
  rig_scene4_tour.mp4 : all 14 scene-4 rig recordings (11 of them never
                        evaluated anywhere -- no labels exist for rain_4..14).

Layout per chapter: Input | 2D control | EvORSP-3T, one row, 448x256 grid
(aspect-correct) displayed 2x. Polarity colours (ON red / OFF blue / both
purple). Trained models at the per-frame SELF-PRIOR threshold.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob
import os
import re
import sys

import cv2
import numpy as np
import torch

from rsp_3d import ORSPNet3D

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
REAL = f"{C.DATA / 'real'}"
DEV = "cuda"
T16, RW, RH = 16, 448, 256
FPS, NPER = 10, 80
SCALE = 2
C_ON, C_OFF, C_BOTH = (60, 60, 220), (220, 120, 40), (150, 60, 150)

# The EvORSP panel used to render `real_evorsp` (test event-DA 0.8170). The
# current real model is `rctx_ours_f4o1_c2` (0.8466) -- same architecture plus
# two windows of inter-window context, so it needs 4 extra input planes that
# this renderer did not previously build. CTX is the number of preceding
# windows it expects; n_extra = 2 * CTX (one ON plane and one OFF plane each).
CTX = 2
models = {}
for tag, ckpt, kw in [
        ("re_2d", "real_orsp2d",
         dict(T=1, num_blocks=4, use_temporal=False)),
        ("re_evorsp", "rctx_ours_f4o1_c2",
         dict(T=4, num_blocks=3, use_off=True, out_chans=1, n_extra=2 * CTX))]:
    ck = torch.load(f"{TMP}/{ckpt}.pt", map_location="cpu")
    m = ORSPNet3D(dilations=(1, 8, 32, 64), **kw)
    m.load_state_dict(ck["state_dict"])
    models[tag] = m.to(DEV).eval()
print("models loaded", flush=True)


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


def ctx_planes(files, j):
    """The CTX preceding windows, as training built them.

    Indexing is into the FULL file list, not the strided render selection --
    the renderer subsamples frames, so consecutive rendered frames are not
    consecutive windows and taking the previous *rendered* frame would feed
    the model a gap it never saw in training. Training clamps at the start of
    a recording (max(j-k, 0)), so a frame's own planes stand in for its
    missing history; this matches that exactly.
    """
    ex = []
    for k in range(1, CTX + 1):
        pl = planes(files[max(j - k, 0)])
        if pl is None:                       # too few events -> fall back to self
            pl = planes(files[j])
        pon, poff = pl
        ex.append(pon.max(0)[None].astype(np.float32))
        ex.append(poff.max(0)[None].astype(np.float32))
    return torch.from_numpy(np.concatenate(ex, 0)).float()[None].to(DEV)


@torch.no_grad()
def frame_panels(files, j):
    pl = planes(files[j])
    if pl is None:
        return None
    on16, off16 = pl
    on_any, off_any = on16.any(0), off16.any(0)
    on4 = torch.from_numpy(on16.reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    off4 = torch.from_numpy(off16.reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    merge = on4.amax(1, keepdim=True)
    lit = merge[0, 0] > 0.5
    ex = ctx_planes(files, j)
    out = [colorize(on_any, off_any)]
    for k, m in models.items():
        p = torch.sigmoid(m(on4, x_off=off4, x_extra=ex)
                          if k == "re_evorsp" else m(merge))[0, 0]
        tau = float(p[lit].mean())                       # self-prior, per frame
        out.append(colorize(on_any, off_any, ((p > tau) & lit).cpu().numpy()))
    return out


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def tour(path, src, note):
    recs = sorted([os.path.basename(d) for d in glob.glob(f"{src}/rain_*")],
                  key=natkey)
    HDR, LBL, GAP, FTR = 52, 8, 6, 30
    PW, PH = RW * SCALE, RH * SCALE
    Wc = LBL + 3 * (PW + GAP) + GAP
    Hc = HDR + PH + GAP + FTR
    raw = path.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
    assert vw.isOpened()
    cols = ["Input (rainy)", "2D control [self-prior tau]",
            "EvORSP-3T +ctx2 [self-prior tau]"]
    for ri, rc in enumerate(recs):
        files = sorted(glob.glob(f"{src}/{rc}/*.npz"))
        step = max(1, len(files) // NPER)
        seg = list(range(0, len(files), step))[:NPER]
        for i, j in enumerate(seg):
            panels = frame_panels(files, j)
            if panels is None:
                continue
            canvas = np.full((Hc, Wc, 3), 252, np.uint8)
            cv2.putText(canvas, f"{rc}   ({ri + 1}/{len(recs)})", (LBL + 6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
            for c, name in enumerate(cols):
                cv2.putText(canvas, name, (LBL + GAP + c * (PW + GAP) + 3, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1,
                            cv2.LINE_AA)
            cv2.putText(canvas, note + "  |  red = ON, blue = OFF, purple = both",
                        (LBL + 6, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (110, 110, 110), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {i + 1}/{len(seg)}", (Wc - 170, Hc - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1,
                        cv2.LINE_AA)
            y0 = HDR + GAP
            for c, img in enumerate(panels):
                x0 = LBL + GAP + c * (PW + GAP)
                canvas[y0:y0 + PH, x0:x0 + PW] = cv2.resize(
                    img, (PW, PH), interpolation=cv2.INTER_NEAREST)
            vw.write(canvas)
        print(f"  {os.path.basename(path)}: {rc} done ({ri + 1}/{len(recs)})",
              flush=True)
        cv2.imwrite(f"{TMP}/tour_{os.path.basename(path)}_{rc}.png", canvas)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
              f"-crf 22 {path}")
    os.remove(raw)
    print(f"done: {path}", flush=True)


tour(f"{OUT}/real_world_tour.mp4", f"{REAL}/EVK4_realworld/scene5/merge_data",
     "ALL 26 real-world recordings (EVK4, unlabeled). Models trained on rig "
     "scenes 1-2 only.")
tour(f"{OUT}/rig_scene4_tour.mp4", f"{REAL}/EVK4_artifical/scene4/merge_data",
     "ALL 14 scene-4 rig recordings; rain_4..14 have no labels and were never "
     "evaluated anywhere. Held-out scene.")
