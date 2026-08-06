"""NATIVE-RESOLUTION visualization: render the kept EVENT STREAM at 1280x720.

Previous videos rendered the model's internal 448x256 grid -- the network's
input tensor, not its output. Deraining is verified subset selection, so the
true output is a subset of the ORIGINAL events, which keep native coordinates:
an event is kept iff its grid cell is kept, and is drawn at its true (x, y).

Rendering: white canvas, ON = red / OFF = blue, per-pixel intensity scales with
event count (1 -> light, >=4 -> full) so dense structure reads with tonal depth
like point-based visualizations. Decisions remain cell-granular (2.9x2.8 px at
native res) -- stated in the footer; the events inside kept cells are native.

Chapters: four real-world recordings spanning light -> storm.
Columns: Input | 2D control | EvORSP-3T (self-prior thresholds).
Canvas: 3 x 1280 native panels -> 3936x862 video.
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
OUT = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/figs"
S5 = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_realworld/scene5/merge_data"
DEV = "cuda"
T16, RW, RH = 16, 448, 256
NW, NH = 1280, 720
FPS, NPER = 10, 80
RECS = ["rain_1", "rain_13", "rain_20", "rain_26"]

models = {}
for tag, f, kw in [("2D control", "real_orsp2d",
                    dict(T=1, num_blocks=4, use_temporal=False)),
                   ("EvORSP-3T", "real_evorsp",
                    dict(T=4, num_blocks=3, use_off=True))]:
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = ORSPNet3D(dilations=(1, 8, 32, 64), **kw)
    m.load_state_dict(ck["state_dict"])
    models[tag] = m.to(DEV).eval()
print("models loaded", flush=True)


def draw_events(x, y, p, keep=None):
    """Native-res panel: count-shaded polarity scatter on white."""
    if keep is not None:
        x, y, p = x[keep], y[keep], p[keep]
    img = np.full((NH, NW, 3), 255, np.uint8)
    for pol, col in ((1, np.array([50, 50, 210], np.int32)),      # ON  red (BGR)
                     (-1, np.array([210, 110, 30], np.int32))):   # OFF blue
        s = p == pol
        if not s.any():
            continue
        cnt = np.zeros((NH, NW), np.int32)
        np.add.at(cnt, (y[s], x[s]), 1)
        m = cnt > 0
        a = np.clip(0.55 + cnt[m] / 6.0, 0.55, 1.0)[:, None]      # intensity
        img[m] = (img[m] * (1 - a) + col[None, :] * a).astype(np.uint8)
    return img


@torch.no_grad()
def keep_masks(x, y, t, p):
    sx = (x.astype(np.int64) * RW) // NW
    sy = (y.astype(np.int64) * RH) // NH
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
    on = np.zeros((T16, RH * RW), bool)
    off = np.zeros((T16, RH * RW), bool)
    s = p == 1
    on[tb[s], sy[s] * RW + sx[s]] = True
    off[tb[~s], sy[~s] * RW + sx[~s]] = True
    on4 = torch.from_numpy(on.reshape(T16, RH, RW)
                           .reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    off4 = torch.from_numpy(off.reshape(T16, RH, RW)
                            .reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
    merge = on4.amax(1, keepdim=True)
    lit = merge[0, 0] > 0.5
    out = {}
    for tag, m in models.items():
        pr = torch.sigmoid(m(on4, x_off=off4) if tag == "EvORSP-3T"
                           else m(merge))[0, 0]
        tau = float(pr[lit].mean())                               # self-prior
        cell_keep = ((pr > tau) & lit).cpu().numpy()
        out[tag] = cell_keep[sy, sx]                              # per-event
    return out


HDR, LBL, GAP, FTR = 56, 8, 6, 32
Wc = LBL + 3 * (NW + GAP) + GAP
Hc = HDR + NH + GAP + FTR
raw = f"{TMP}/native_raw.mp4"
vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
assert vw.isOpened()
cols = ["Input (rainy, native 1280x720)", "2D control [self-prior tau]",
        "EvORSP-3T [self-prior tau]"]

for ri, rc in enumerate(RECS):
    files = sorted(glob.glob(f"{S5}/{rc}/*.npz"))
    step = max(1, len(files) // NPER)
    for i, f in enumerate(files[::step][:NPER]):
        with np.load(f) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        if len(x) < 200:
            continue
        km = keep_masks(x, y, t, p)
        panels = [draw_events(x, y, p),
                  draw_events(x, y, p, km["2D control"]),
                  draw_events(x, y, p, km["EvORSP-3T"])]
        canvas = np.full((Hc, Wc, 3), 252, np.uint8)
        cv2.putText(canvas, f"{rc}   ({ri + 1}/{len(RECS)})", (LBL + 6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        for c, name in enumerate(cols):
            cv2.putText(canvas, name, (LBL + GAP + c * (NW + GAP) + 4, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1,
                        cv2.LINE_AA)
        cv2.putText(canvas, "kept EVENTS drawn at native sensor coordinates; "
                    "keep decision is cell-granular (448x256 grid). red = ON, "
                    "blue = OFF, intensity = event count.",
                    (LBL + 6, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {i + 1}", (Wc - 170, Hc - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1,
                    cv2.LINE_AA)
        y0 = HDR + GAP
        for c, img in enumerate(panels):
            x0 = LBL + GAP + c * (NW + GAP)
            canvas[y0:y0 + NH, x0:x0 + NW] = img
        vw.write(canvas)
    print(f"  {rc} done ({ri + 1}/{len(RECS)})", flush=True)
    cv2.imwrite(f"{TMP}/native_{rc}.png", canvas)

vw.release()
os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
          f"-crf 21 {OUT}/real_world_native.mp4")
os.remove(raw)
print(f"done: {OUT}/real_world_native.mp4", flush=True)
