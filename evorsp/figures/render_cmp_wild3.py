"""Real-WORLD (scene5, unlabelled) comparison: 4 recordings, native 1280x720.

No ground truth exists for these, so this is a deployment demo, not a
measurement -- nothing here should be quoted as a number. It answers the one
question the labelled scenes cannot: do the models still behave sensibly on
genuine outdoor storms they have never seen, from a rig they were not tuned on?

Panels (1x3): Input | EvORSP-3T (self-prior tau) | EvORSP-3T/E-style per-bin
PRE-Mamba is absent by necessity: scene5 has no entry in its split table, so no
predictions exist for it and running its released checkpoint would require the
architecture variant its public code cannot construct.

Thresholding: FIXED tau = 0.35 for both panels. The per-frame self-prior used
earlier is the better deployment rule, but it re-centres on every frame and
would absorb exactly the decision shift adaptation produced -- the un-adapted
model keeps 84% of wild events at this tau and the adapted one 57%, and
self-prior would erase that gap by construction. Fixed tau isolates the
adaptation as the only variable.
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
from run_real_full import HeadV2 as HeadR
from gpu_feats import patch_gpu, tensor_gpu
from iti_feats import iti_gpu
from recur_feats import Recur

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"
S5 = f"{C.REAL_WILD_SRC}/scene5/merge_data"
DEV = "cuda"
NW, NH = 1280, 720
R, T16 = 256, 16
FPS, NPER = 8, 55
RECS = ["rain_1", "rain_9", "rain_18", "rain_26"]

# BEST real models: trunk + per-event head v3, decisions made PER EVENT.
# scene-disjoint 0.8686 (trunk-only 0.8298); PRE-Mamba split 0.8444 (0.8066).
models, heads, hooks = {}, {}, {}
for tag, f in (("BEFORE adaptation  (rig-trained, 0.8831 on rig)", "realfull_ours"),
               ("AFTER self-supervised adaptation on wild", "wildadapt")):
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=1)
    m.load_state_dict(ck["trunk"])
    models[tag] = m.to(DEV).eval()
    h = HeadR(feat_dim=128).to(DEV)
    h.load_state_dict(ck["head"])
    h.eval()
    heads[tag] = h
    st = {}
    hooks[tag] = st
    m.out_proj.register_forward_pre_hook(
        lambda mod, i, st=st: st.__setitem__("f", i[0]))
    for bi, blk in enumerate(m.blocks):
        blk.register_forward_hook(
            lambda mod, i, o, st=st, bi=bi: st.__setitem__(bi, o))
print("before/after adaptation models loaded", flush=True)

# Both panels see IDENTICAL inputs, so any difference is the adaptation alone.
# Fixed tau rather than the per-frame self-prior: self-prior re-centres itself
# on every frame and would absorb exactly the shift adaptation produced,
# hiding the thing this video exists to show.
TAU = 0.35
RECUR = Recur(nw=NW, nh=NH)


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
def keeps(x, y, t, p):
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
    tn = ((t - t0) / span).astype(np.float32)
    xg = torch.from_numpy(x.astype(np.int64)).to(DEV)
    yg = torch.from_numpy(y.astype(np.int64)).to(DEV)
    tg = torch.from_numpy(t.astype(np.int64)).to(DEV)
    pg = torch.from_numpy(p.astype(np.int64)).to(DEV)
    tns = torch.from_numpy(tn)[None].to(DEV)
    xs = torch.from_numpy(x.astype(np.float32))[None].to(DEV)
    ys = torch.from_numpy(y.astype(np.float32))[None].to(DEV)
    # EVK4 stamps are MICROseconds: slice 1000 us, tau 5000 us
    pv = patch_gpu(xg, yg, tns[0], pg, NW, NH)[None]
    tc = tensor_gpu(xg, yg, tg, 5_000, [4, 16, 64], NW, NH, 1_000)
    it = iti_gpu(xg, yg, tg, nw=NW, nh=NH)
    xi, yi = x.astype(np.int64), y.astype(np.int64)
    rc_ = torch.from_numpy(RECUR.features(xi, yi, np.arange(len(x)))).to(DEV)
    RECUR.push(xi, yi)                     # features first, push after
    tc = torch.cat([tc, it, rc_], 1)[None]
    out = {}
    for tag, m in models.items():
        lm = m(on4, x_off=off4)
        st = hooks[tag]
        fm = torch.cat([st["f"]] + [st[i] for i in range(3)], 1)
        lv = sample_at(lm[:, None], xs, ys, tns)
        fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                       xs, ys, tns)
        ev = torch.sigmoid(heads[tag](lv, fv, pv, tc, tns[..., None]))[0, :, 0]
        out[tag] = (ev > TAU).cpu().numpy()
    return out


COLS = ["Input (rainy, native 1280x720)"] + list(models)
GAP, HDR, FTR = 8, 56, 36
Wc = GAP + 3 * (NW + GAP)
Hc = HDR + NH + 24 + GAP + FTR
raw = f"{TMP}/cmp_wild3_raw.mp4"
vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Wc, Hc))
assert vw.isOpened()

for ri, rc in enumerate(RECS):
    files = sorted(glob.glob(f"{S5}/{rc}/*.npz"))
    if not files:
        print(f"  {rc}: MISSING, skipped", flush=True)
        continue
    step = max(1, len(files) // NPER)
    n_ok = 0
    RECUR.reset()                     # history must not cross recordings
    # walk every frame so the recurrence buffer sees consecutive windows as
    # training did; render only every step-th
    for fi, f in enumerate(files):
        with np.load(f) as d:
            x, y, t, p = d["x"], d["y"], d["t"], d["p"]
        if fi % step or n_ok >= NPER or len(x) < 200:
            RECUR.push(x.astype(np.int64), y.astype(np.int64))
            continue
        km = keeps(x, y, t, p)
        panels = [draw(x, y, p)] + [draw(x, y, p, km[c]) for c in COLS[1:]]
        canvas = np.full((Hc, Wc, 3), 252, np.uint8)
        cv2.putText(canvas, f"REAL-WORLD storm (scene5 / {rc}, UNLABELLED -- "
                    f"qualitative only)   recording {ri + 1}/{len(RECS)}",
                    (GAP + 4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (30, 30, 30), 2, cv2.LINE_AA)
        for c, (name, panel) in enumerate(zip(COLS, panels)):
            x0 = GAP + c * (NW + GAP)
            cv2.putText(canvas, name, (x0 + 2, HDR + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (90, 90, 90), 1,
                        cv2.LINE_AA)
            canvas[HDR + 24:HDR + 24 + NH, x0:x0 + NW] = panel
        cv2.putText(canvas, "kept EVENTS at native sensor coordinates; "
                    "FIXED tau 0.35 for BOTH panels, so the only difference "
                    "is the adaptation (self-prior would re-centre per frame "
                    "and hide it). red = ON, blue = OFF. "
                    "No ground truth exists for scene5 -- do not quote numbers.",
                    (GAP + 4, Hc - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    (110, 110, 110), 1, cv2.LINE_AA)
        vw.write(canvas)
        n_ok += 1
    print(f"  {rc}: {n_ok} frames", flush=True)
vw.release()
os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -pix_fmt yuv420p "
          f"-crf 18 {OUT}/cmp_wild_realworld.mp4")
print(f"wrote {OUT}/cmp_wild_realworld.mp4  ({Wc}x{Hc})", flush=True)
