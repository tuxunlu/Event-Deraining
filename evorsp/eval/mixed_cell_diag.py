"""Quantify the observed failure: scene events lost inside dense-rain cells.

The visual failure (car edge deleted where a large rain drop overlaps it) is a
MIXED-CELL failure: a cell holding both classes gets one decision, and when the
drop supplies many more events than the edge, count-majority deletes the cell
and the edge with it.

This measures it directly, with the cell defined at OUR granularity so every
model is judged on the same partition:

    mixed cell        = cell (t-bin, y, x) holding >=1 scene AND >=1 rain event
    scene recall      = kept scene events / all scene events, restricted to
                        mixed cells, stratified by how rain-dominated the cell is
                        (rain:scene ratio)

Frame-averaged balanced accuracy hides this: mixed cells are only 2.3% of lit
cells. This metric is the one that would have caught it.

Models compared:
    ctx_c2   EvORSP-3T/E + multi-window, count-majority target   (current best)
    exact    the metric-exact BA-weighted loss -- normalizes by per-frame class
             totals, so a rare scene event outweighs an abundant rain event.
             If the diagnosis is right, this arm should already do better here.
    premamba its saved per-event predictions (native pixel x 20 time bins)
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
SRC = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/synthetic_KITTI/synthetic"
PACK = "/fs/nexus-scratch/tuxunlu/kitti_t16e/test"
PM = "/fs/nexus-scratch/tuxunlu/git/PRE-Mamba/exp/event_rain/SYTHETIC/result"
DEV = "cuda"
NW, NH, R, T16 = 460, 352, 256, 16
BANDS = [(1, 2), (2, 5), (5, 20), (20, 10 ** 9)]      # rain:scene ratio bands


def load(tag, n_extra=0):
    ck = torch.load(f"{TMP}/{tag}.pt", map_location="cpu")
    m = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64),
                  out_chans=16, n_extra=n_extra)
    m.load_state_dict(ck["state_dict"])
    return m.to(DEV).eval()


best = load("ctx_f4o16_c2", n_extra=4)
exact = load("loss_exact_o16")
print("models loaded", flush=True)


def key(x, y, t):
    return (t.astype(np.int64) * (NW * NH) + y.astype(np.int64) * NW
            + x.astype(np.int64))


acc = {k: {b: [0.0, 0.0] for b in BANDS} for k in
       ("ctx_c2", "exact", "premamba")}
overall = {k: [0.0, 0.0] for k in acc}
n_frames = 0

with torch.no_grad():
    for mm in ("50mm", "150mm"):
        files = sorted(glob.glob(f"{SRC}/merge_data/{mm}/*.npz"))[::8]
        for f in files:
            base = os.path.basename(f)
            pk = f"{PACK}/{mm}/{base}"
            pp = f"{PM}/{mm}_{base.replace('.npz', '')}.npy"
            if not (os.path.exists(pk) and os.path.exists(pp)):
                continue
            with np.load(f) as d:
                x, y, t = d["x"], d["y"], d["t"]
            with np.load(f"{SRC}/raw_data/{base}") as d:
                clean = np.sort(key(d["x"], d["y"], d["t"]))
            rain = ~np.isin(key(x, y, t), clean)
            if rain.sum() < 50 or (~rain).sum() < 50:
                continue

            sx = np.clip((x.astype(np.int64) * R) // NW, 0, R - 1)
            sy = np.clip((y.astype(np.int64) * R) // NH, 0, R - 1)
            t0 = t.min()
            span = max(int(t.max() - t0), 1)
            tb = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
            cell = (tb * R + sy) * R + sx

            with np.load(pk) as d:
                on = np.unpackbits(d["on"])[: T16 * R * R].reshape(T16, R, R)
                off = np.unpackbits(d["off"])[: T16 * R * R].reshape(T16, R, R)
            on4 = torch.from_numpy(on.reshape(4, 4, R, R).max(1)
                                   ).float()[None].to(DEV)
            off4 = torch.from_numpy(off.reshape(4, 4, R, R).max(1)
                                    ).float()[None].to(DEV)
            lit_cell = torch.from_numpy((on | off).astype(bool)).to(DEV)
            idx = int(base.split(".")[0])
            ex = []
            for k in (1, 2):
                pv = f"{PACK}/{mm}/{max(idx - k, 0):010d}.npz"
                with np.load(pv if os.path.exists(pv) else pk) as d:
                    pon = np.unpackbits(d["on"])[: T16 * R * R].reshape(T16, R, R)
                    poff = np.unpackbits(d["off"])[: T16 * R * R].reshape(T16, R, R)
                ex.append(pon.max(0)[None].astype(np.float32))
                ex.append(poff.max(0)[None].astype(np.float32))
            ex = torch.from_numpy(np.concatenate(ex, 0))[None].to(DEV)

            keeps = {}
            for name, mdl, kw in (("ctx_c2", best, {"x_extra": ex}),
                                  ("exact", exact, {})):
                pr = torch.sigmoid(mdl(on4, x_off=off4, **kw))[0]
                tau = float(pr[lit_cell].mean())
                keeps[name] = (pr > tau).cpu().numpy()[tb, sy, sx]
            pred = np.load(pp)
            keeps["premamba"] = (pred[:len(x)] == 0) if len(pred) >= len(x) \
                else np.ones(len(x), bool)

            # per-cell class counts at OUR granularity
            order = np.argsort(cell, kind="stable")
            c = cell[order]
            r = rain[order]
            b = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            cells = c[b]
            cnt = np.diff(np.r_[b, len(c)])
            rn_c = np.add.reduceat(r.astype(np.int64), b)
            bg_c = cnt - rn_c
            mixed = (bg_c > 0) & (rn_c > 0)
            if not mixed.any():
                continue
            ratio = rn_c[mixed] / np.maximum(bg_c[mixed], 1)
            cmap = {}
            for cc, rr in zip(cells[mixed], ratio):
                cmap[cc] = rr
            in_mixed = np.array([cc in cmap for cc in cell])
            sel = in_mixed & ~rain              # scene events inside mixed cells
            if sel.sum() == 0:
                continue
            rr_ev = np.array([cmap[cc] for cc in cell[sel]])
            for name in acc:
                kp = keeps[name][sel]
                overall[name][0] += kp.sum()
                overall[name][1] += len(kp)
                for lo, hi in BANDS:
                    m = (rr_ev >= lo) & (rr_ev < hi)
                    if m.any():
                        acc[name][(lo, hi)][0] += kp[m].sum()
                        acc[name][(lo, hi)][1] += m.sum()
            n_frames += 1

print(f"\n=== SCENE-EVENT RECALL INSIDE MIXED CELLS ({n_frames} frames) ===")
print("  (cells holding both classes; 'ratio' = rain events : scene events)\n")
hdr = "  " + "model".ljust(12) + "".join(
    f"{'x'.join(map(str, b)) if b[1] < 10**9 else '20+':>12s}" for b in BANDS)
print(hdr.replace("x", ":"))
for name in ("ctx_c2", "exact", "premamba"):
    row = "  " + name.ljust(12)
    for b in BANDS:
        k, n = acc[name][b]
        row += f"{(k / n if n else float('nan')):12.4f}"
    o = overall[name]
    row += f"   | all {o[0] / max(o[1], 1):.4f}"
    print(row)
print("\n  higher = more of the occluded structure survives")
