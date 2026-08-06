"""fig1 / fig2 / fig4 with EVERY model across the whole project, now including
EvORSP-3T -- the six-model arc: DFFN, FourierMamba2D, ORSPNet, its best config,
StreakNet, and EvORSP-3T.

EvORSP-3T reads the temporal+OFF planes (kitti_t16) instead of the collapsed
eFFT frame. Verified before drawing: the two pipelines share the exact pixel
grid (anchor-vs-frame IoU 1.0000 on both test rates), so its panels are
pixel-aligned with the others. Each model runs at its own protocol-selected
threshold; FourierMamba2D keeps its earlier-protocol asterisk.
"""
import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
import train_compare as TC

TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
OUT = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/figs"
T16 = "/fs/nexus-scratch/tuxunlu/kitti_t16"
DEV = "cuda"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF, GRID = "#fcfcfb", "#e3e2dd"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
PURPLE = "#8a5fd0"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.titleweight": "bold",
})


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
    ("dffn",      "proto_dffn",          "dffn",      "DFFN\n(earlier)",           BLUE,   ""),
    ("fmamba",    "ckpt_fmamba",         "fmamba",    "FourierMamba2D\n(earlier)", "#b48bd9", "*"),
    ("orsp",      "proto_orsp",          "orsp",      "ORSPNet",                   ORANGE, ""),
    ("orsp_best", "proto_orsp_bal_dil",  "orsp_dil",  "ORSPNet\n+ dil + balanced", AQUA,   ""),
    ("streaknet", "proto_streaknet_bal", "streaknet", "StreakNet\nK=127",          YELLOW, ""),
    ("evorsp",    "k3d_T4b3off",         "evorsp3t",  "EvORSP-3T\n(temporal+OFF)", PURPLE, ""),
]

nets, meta = {}, {}
for k, f, b, lbl, c, note in MODELS:
    ck = torch.load(f"{TMP}/{f}.pt", map_location="cpu")
    m = build(b)
    m.load_state_dict(ck["state_dict"])
    nets[k] = m.to(DEV).eval()
    meta[k] = {"params": ck["params"],
               "tau": float(ck.get("test_tau", ck.get("tau", 0.5))),
               "da": float(ck.get("test_meanDA", ck.get("test", ck.get("meanDA", 0.0)))),
               "label": lbl, "color": c, "note": note}
    print(f"  {k:10s} {meta[k]['params']:>10,}p  tau {meta[k]['tau']:.2f}  "
          f"DA {meta[k]['da']:.4f} {note}", flush=True)

TEST = sorted(os.listdir(f"{TC.ROOT}/merge_data/test"),
              key=lambda s: int(s.replace("mm", "")))


def t16_planes(mm, basename):
    with np.load(f"{T16}/test/{mm}/{basename}") as d:
        on = np.unpackbits(d["on"])[:16 * 256 * 256].reshape(16, 256, 256)
        off = np.unpackbits(d["off"])[:16 * 256 * 256].reshape(16, 256, 256)
    on = torch.from_numpy(on.reshape(4, 4, 256, 256).max(1)).float()
    off = torch.from_numpy(off.reshape(4, 4, 256, 256).max(1)).float()
    return on.unsqueeze(0).to(DEV), off.unsqueeze(0).to(DEV)


@torch.no_grad()
def prob(k, merge, mm, basename):
    if k == "evorsp":
        on, off = t16_planes(mm, basename)
        return torch.sigmoid(nets[k](on, x_off=off))
    return torch.sigmoid(nets[k](merge))


@torch.no_grad()
def per_rate_metrics():
    out = {k: {} for k in nets}
    raws = sorted(glob.glob(f"{TC.ROOT}/raw_data/*.npz"))
    for mm in TEST:
        fs = sorted(glob.glob(f"{TC.ROOT}/merge_data/test/{mm}/*.npz"))
        acc = {k: [0.0, 0.0, 0] for k in nets}
        for i, f in enumerate(fs):
            merge = TC.RainSet._img(f).unsqueeze(0).to(DEV)
            raw = TC.RainSet._img(raws[i]).unsqueeze(0).to(DEV)
            gt = (raw[0, 0] > 0.5)
            lit = (merge[0, 0] > 0.5)
            real, rain = gt & lit, lit & ~gt
            rs, ns = int(real.sum()), int(rain.sum())
            if rs == 0 or ns == 0:
                continue
            base = os.path.basename(f)
            for k in nets:
                pr = prob(k, merge, mm, base)[0, 0] > meta[k]["tau"]
                acc[k][0] += float((pr & real).sum()) / rs
                acc[k][1] += (ns - float((pr & rain).sum())) / ns
                acc[k][2] += 1
        for k in nets:
            s, n, c = acc[k]
            out[k][mm] = {"SR": s / max(c, 1), "NR": n / max(c, 1),
                          "DA": 0.5 * (s + n) / max(c, 1)}
        print(f"  {mm}: " + "  ".join(f"{k} {out[k][mm]['DA']:.4f}" for k in nets),
              flush=True)
    return out


import json as _json
_MC = f"{TMP}/figs_met_cache.json"
if os.path.exists(_MC):
    MET = _json.load(open(_MC))
    print("metrics loaded from cache")
else:
    MET = per_rate_metrics()
    _json.dump(MET, open(_MC, "w"))

SAMPLES = []
raws = sorted(glob.glob(f"{TC.ROOT}/raw_data/*.npz"))
for mm in TEST:
    fs = sorted(glob.glob(f"{TC.ROOT}/merge_data/test/{mm}/*.npz"))
    i = 12
    SAMPLES.append((mm, os.path.basename(fs[i]),
                    TC.RainSet._img(fs[i]).unsqueeze(0).to(DEV),
                    TC.RainSet._img(raws[i]).unsqueeze(0).to(DEV)))
CROP = (slice(40, 200), slice(40, 220))


def predict(k, merge, mm, base):
    return (prob(k, merge, mm, base) > meta[k]["tau"]).float()


# ================================================================== FIG 1
cols = ["Input (rainy)", "Ground truth"] + [meta[k]["label"] for k, *_ in MODELS]
fig, ax = plt.subplots(len(SAMPLES), len(cols),
                       figsize=(1.9 * len(cols), 2.25 * len(SAMPLES)))
ax = np.atleast_2d(ax)
for r, (mm, base, merge, raw) in enumerate(SAMPLES):
    lit = (merge > 0.5).float()
    # deraining is verified subset selection: output events are a subset of
    # input events, so the rendered frame is pred AND input for EVERY model.
    # (Also hides the lit-masked models' unconstrained dark-pixel outputs,
    # which the metric never sees -- see README.)
    panels = [lit, (raw > 0.5).float()] + \
             [predict(k, merge, mm, base) * lit for k, *_ in MODELS]
    for c, img in enumerate(panels):
        a = ax[r, c]
        a.imshow(img[0, 0].cpu().numpy()[CROP], cmap="gray_r", vmin=0, vmax=1,
                 interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(True); s.set_color(GRID)
        if r == 0:
            a.set_title(cols[c], fontsize=8.2, color=INK, pad=6)
        if c == 0:
            a.set_ylabel(mm, fontsize=10, color=INK, fontweight="bold")
        if c >= 2 and r == len(SAMPLES) - 1:
            k = MODELS[c - 2][0]
            a.set_xlabel(f"DA {meta[k]['da']:.4f}{meta[k]['note']}\n"
                         f"{meta[k]['params']:,}p", fontsize=7.4, color=MUTED)
fig.suptitle("Derained event frames — protocol TEST rates, each model at its own "
             "selected threshold", fontsize=11, color=INK, y=0.995)
fig.text(.5, .008, "*FourierMamba2D's threshold is from the earlier bake-off "
         "protocol; its DA is not directly comparable.   All outputs are shown "
         "restricted to the input support (deraining is verified subset "
         "selection: kept events \u2286 input events).",
         ha="center", color=MUTED, fontsize=7.4)
fig.tight_layout(rect=[0, .028, 1, .965])
fig.savefig(f"{OUT}/fig1_qualitative.png", dpi=170)
plt.close(fig)
print("fig1_qualitative.png")

# ================================================================== FIG 2
err_cmap = ListedColormap([SURF, "#cfcec9", BLUE, RED])
fig, ax = plt.subplots(len(SAMPLES), len(MODELS),
                       figsize=(2.15 * len(MODELS), 2.35 * len(SAMPLES)))
ax = np.atleast_2d(ax)
for r, (mm, base, merge, raw) in enumerate(SAMPLES):
    gt = (raw > 0.5).float()
    lit = (merge > 0.5).float()
    for c, (k, *_) in enumerate(MODELS):
        pr = predict(k, merge, mm, base)
        code = torch.zeros_like(pr)
        code = torch.where((lit > 0) & (gt > 0) & (pr > 0), torch.ones_like(pr), code)
        code = torch.where((lit > 0) & (gt > 0) & (pr == 0), 2 * torch.ones_like(pr), code)
        code = torch.where((lit > 0) & (gt == 0) & (pr > 0), 3 * torch.ones_like(pr), code)
        a = ax[r, c]
        a.imshow(code[0, 0].cpu().numpy()[CROP], cmap=err_cmap, vmin=0, vmax=3,
                 interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(True); s.set_color(GRID)
        if r == 0:
            a.set_title(meta[k]["label"], fontsize=8.2, color=INK, pad=6)
        if c == 0:
            a.set_ylabel(mm, fontsize=10, color=INK, fontweight="bold")
        if r == len(SAMPLES) - 1:
            m = MET[k][mm]
            a.set_xlabel(f"SR {m['SR']:.3f}  NR {m['NR']:.3f}", fontsize=7.4,
                         color=MUTED)
fig.legend(handles=[Patch(facecolor="#cfcec9", label="signal kept (correct)"),
                    Patch(facecolor=BLUE, label="signal lost (hurts SR)"),
                    Patch(facecolor=RED, label="rain kept (hurts NR)")],
           loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, -.005))
fig.suptitle("Where each model makes its mistakes — protocol TEST rates",
             fontsize=11, color=INK, y=.995)
fig.tight_layout(rect=[0, .045, 1, .965])
fig.savefig(f"{OUT}/fig2_errors.png", dpi=170)
plt.close(fig)
print("fig2_errors.png")

# ================================================================== FIG 4
fig, axs = plt.subplots(1, len(TEST), figsize=(6.6 * len(TEST), 4.6), sharey=True)
axs = np.atleast_1d(axs)
mets = ["SR", "NR", "DA"]
for ai, mm in enumerate(TEST):
    a = axs[ai]
    xs = np.arange(len(mets))
    w = 0.135
    for mi, (k, *_) in enumerate(MODELS):
        vals = [MET[k][mm][t] for t in mets]
        off = (mi - (len(MODELS) - 1) / 2) * w
        a.bar(xs + off, vals, w, color=meta[k]["color"],
              label=meta[k]["label"].replace("\n", " ") + meta[k]["note"])
        for xi, v in enumerate(vals):
            a.text(xs[xi] + off, v + .006, f"{v:.3f}", ha="center", fontsize=6.2,
                   color=INK2, rotation=90)
    a.set_xticks(xs)
    a.set_xticklabels(["SR\n(signal recall)", "NR\n(rain recall)",
                       "DA\n= ½(SR+NR)"], fontsize=8.6)
    a.set_ylim(0.55, 1.08)
    a.set_title(f"{mm} rain", pad=10)
    if ai == 0:
        a.set_ylabel("recall")
axs[0].legend(fontsize=7.6, loc="lower left", ncol=2)
fig.suptitle("Per-rate signal and rain recall — all six models, protocol TEST "
             "rates, one evaluator, one pass", fontsize=11, color=INK, y=.995)
fig.text(.5, .01, "*FourierMamba2D ran under the earlier bake-off protocol; its "
         "bars are indicative, not comparable.", ha="center", color=MUTED,
         fontsize=7.6)
fig.tight_layout(rect=[0, .035, 1, .955])
fig.savefig(f"{OUT}/fig4_metrics.png", dpi=170)
plt.close(fig)
print("fig4_metrics.png")
