"""Threshold-free comparison on the population the streaks live in.

streak_eval.py judged each model at its OWN validation-selected tau (0.35-0.50).
That conflates two different things: how well a model SEPARATES persistent rain
from persistent scene, and how permissive its operating point happens to be. A
model tuned to keep more of everything looks worse on rain-kept without being
worse at the task.

This removes the confound. For every event on a persistent pixel it histograms
the model's probability by true class, then derives the whole ROC from those
histograms:

  AUC            threshold-free separability on this population. THE number.
                 0.5 = cannot tell a nozzle from a building edge at all.
  rain @ scene=X rain kept when the threshold is set so that X of persistent
                 SCENE events survive -- i.e. matched operating points, which
                 is the comparison streak_eval could not make.

Histograms rather than stored probabilities: exact to bin resolution, and flat
in memory over ~2300 frames x ~10^5 events.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_real_full as RF
from rsp_3d import ORSPNet3D
from run_real_full import CacheSet, sample_at
from streak_eval import Head, persistence

DEV = "cuda"
TMP = os.path.dirname(os.path.abspath(__file__))
NB = 2000                       # probability bins
TARGETS = (0.85, 0.90, 0.95)    # matched scene-kept levels

MODELS = [
    ("realph_theirs", "theirs", 19), ("realiti_theirs", "theirs", 27),
    ("realfull_theirs", "theirs", 39),
    ("realph_ours", "ours", 19), ("realiti_ours", "ours", 27),
    ("realfull_ours", "ours", 39),
]


def build(tag, ncols):
    p = f"{TMP}/{tag}.pt"
    if not os.path.exists(p):
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    trunk = ORSPNet3D(T=4, dilations=(1, 8, 32, 64), num_blocks=3,
                      use_off=True, out_chans=1).to(DEV).eval()
    trunk.load_state_dict(blob["trunk"])
    head = Head(ncols).to(DEV).eval()
    head.load_state_dict(blob["head"])
    feats = {}
    trunk.out_proj.register_forward_pre_hook(
        lambda m, inp: feats.__setitem__("f", inp[0]))
    for bi, blk in enumerate(trunk.blocks):
        blk.register_forward_hook(
            lambda m, i, o, bi=bi: feats.__setitem__(f"b{bi}", o))
    return dict(trunk=trunk, head=head, feats=feats, ncols=ncols, tag=tag,
                tau=float(blob["tau"]),
                hr=np.zeros(NB + 1), hs=np.zeros(NB + 1))


@torch.no_grad()
def run(split, ms):
    ms = [m for m in ms if m]
    if not ms:
        return
    ds = CacheSet("test", split)
    ld = DataLoader(ds, batch_size=1, num_workers=4)
    for k, (on, off, ex, xs, ys, tn, patch, tcols, lab, inv_p,
            n_bg, n_rn, idx) in enumerate(ld):
        y0 = lab[0].numpy()
        pers = persistence(ds, int(idx[0]), xs[0].numpy(), ys[0].numpy())
        m_rain, m_scene = (y0 < 0.5) & pers, (y0 > 0.5) & pers
        if not (m_rain.any() or m_scene.any()):
            continue
        xg, yg, tg = xs.to(DEV), ys.to(DEV), tn.to(DEV)
        pg, tcg = patch.to(DEV), tcols.to(DEV)
        for m in ms:
            lm = m["trunk"](on.to(DEV), x_off=off.to(DEV))
            f = m["feats"]
            fm = torch.cat([f["f"]] + [f[f"b{i}"] for i in
                                       range(len(m["trunk"].blocks))], 1)
            lv = sample_at(lm[:, None], xg, yg, tg)
            fv = sample_at(fm[:, :, None].expand(-1, -1, lm.shape[1], -1, -1),
                           xg, yg, tg)
            pr = torch.sigmoid(m["head"](lv, fv, pg, tcg[..., :m["ncols"]],
                                         tg[..., None])[..., 0])[0].cpu().numpy()
            b = np.clip((pr * NB).astype(np.int64), 0, NB)
            m["hr"] += np.bincount(b[m_rain], minlength=NB + 1)
            m["hs"] += np.bincount(b[m_scene], minlength=NB + 1)
        if k % 200 == 0:
            print(f"  {split} {k}/{len(ds)}", flush=True)


def roc(m):
    """-> (AUC, rain_kept at each matched scene-kept target).

    Sweep the threshold from high probability downwards; keeping means
    p > tau, so the survivors at bin i are the suffix sums.
    """
    rs = m["hr"][::-1].cumsum()[::-1] / max(m["hr"].sum(), 1)   # rain kept
    ss = m["hs"][::-1].cumsum()[::-1] / max(m["hs"].sum(), 1)   # scene kept
    order = np.argsort(rs)
    auc = float(np.trapz(ss[order], rs[order]))
    out = []
    for t in TARGETS:
        i = int(np.argmin(np.abs(ss - t)))
        out.append((rs[i], i / NB, ss[i]))
    return auc, out


if __name__ == "__main__":
    rows = []
    for split in ("theirs", "ours"):
        ent = [build(t, n) for t, sp, n in MODELS if sp == split]
        run(split, ent)
        rows += [m for m in ent if m]

    hdr = "  ".join(f"rain@{int(100*t)}" for t in TARGETS)
    print(f"\n{'model':18s} {'cols':>4s} {'AUC':>7s}   {hdr}")
    print("  " + "-" * 60)
    for m in rows:
        auc, pts = roc(m)
        cells = "   ".join(f"{r:.3f}@t{t:.2f}" for r, t, _ in pts)
        print(f"{m['tag']:18s} {m['ncols']:4d} {auc:7.4f}   {cells}")
    print("\n  AUC: separability of persistent rain vs persistent scene,")
    print("  independent of threshold. rain@X: rain kept when tau is set so")
    print("  that X% of persistent SCENE events survive -- matched operating")
    print("  points. LOWER rain@X is better at equal structure preservation.")
