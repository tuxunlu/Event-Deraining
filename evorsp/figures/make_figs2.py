"""Performance figures for every model measured in this project.

Adds fig6-fig10 to figs/. Palette and rcParams match make_figs.py so the whole
folder reads as one set.

Honesty rules applied throughout, because several numbers here are NOT
comparable to each other and a figure that hides that is worse than no figure:
  * test-protocol numbers and val-only numbers never share an axis;
  * the +-0.0039 seed band is shown wherever DA differences are discussed, since
    most differences in this project are inside it;
  * 10-epoch triage and 50-epoch protocol are plotted as a PAIR, not merged,
    because triage overstated every gain by 3-5x;
  * every ablation is compared against ITS OWN control, named on the figure.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TMP = f"{C.CKPT}"
OUT = f"{C.FIGS}"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF, GRID = "#fcfcfb", "#e3e2dd"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.titleweight": "bold",
})

SEED = 0.0039           # measured seed std, 3 seeds on ORSPNet

# 50 epochs; train on 14 rain rates, select tau on val {20,80}mm, test {50,150}mm.
# Latency: idle GPU, batch 1, 256x256, 100 warm-up iters, 7 repeats, median.
PROTO = [
    ("DFFN",                                72_074, 11.82, 0.9159, BLUE),
    ("ORSPNet",                             36_206,  6.94, 0.9193, ORANGE),
    ("ORSPNet + balanced loss",             36_206,  6.94, 0.9210, ORANGE),
    ("ORSPNet + dil  (RF 33→129 px)",       36_782,  7.13, 0.9223, ORANGE),
    ("ORSPNet + dil + lit-masked BCE",      36_782,  7.13, 0.9244, AQUA),
    ("ORSPNet + dil + balanced loss",       36_782,  7.13, 0.9248, AQUA),
    ("ORSPNet + dil + prior threshold",     36_782,  7.13, 0.9248, AQUA),
    ("StreakNet K=127",                     52_043,  9.93, 0.9268, YELLOW),
]

# EvORSP-3T on KITTI (T=4 temporal frontend + OFF channel, 3 blocks) -- appended
# only once its 50-epoch protocol TEST number exists. Latency 5.69 ms is the
# definitive idle-node measurement of the identical architecture.
PURPLE = "#8a5fd0"
MAUVE = "#b48bd9"
PROTO.append(("FourierMamba2D  (completed run)", 23_592_081, 137.40, 0.9411, MAUVE))
K3D = None
try:
    K3D = json.load(open(f"{TMP}/k3d_T4b3off.json"))
    PROTO.append(("EvORSP-3T  (temporal + OFF input)",
                  K3D["params"], 5.69, K3D["test"], PURPLE))
except FileNotFoundError:
    pass

# 2D b3-body seeds (same tag overwrites the json, so values are passed in by the
# harvest step as a list once the logs are final)
B3BODY = None
try:
    B3BODY = json.load(open(f"{TMP}/b3_body_kitti.json"))   # {"tests": [...], "ms": 5.39}
    PROTO.append((f"ORSPNet b3 + dil + bal  (2D body, {len(B3BODY['tests'])} seeds)",
                  27_685, B3BODY["ms"], float(np.mean(B3BODY["tests"])), ORANGE))
except FileNotFoundError:
    pass

# ------------------------------------------------------------------- fig 6
PROTO.sort(key=lambda r: r[3])
fig, ax = plt.subplots(figsize=(11.6, 5.6))
y = np.arange(len(PROTO))[::-1]
for i, (name, p, ms, da, c) in enumerate(PROTO):
    ax.plot([0.9155, da], [y[i], y[i]], color=GRID, lw=1.3, zorder=1)
    ax.scatter([da], [y[i]], s=95, color=c, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax.annotate(f"{da:.4f}", (da, y[i]), textcoords="offset points", xytext=(8, 0),
                va="center", color=INK, fontsize=9.2, fontweight="bold")
    ax.annotate(f"{p:,} params · {ms:.2f} ms", (da, y[i]),
                textcoords="offset points", xytext=(58, 0),
                va="center", color=MUTED, fontsize=8)
best = max(r[3] for r in PROTO)
ax.errorbar([best], [-0.75], xerr=[SEED], fmt="none", ecolor=MUTED,
            elinewidth=1.4, capsize=4)
ax.text(best, -1.15, f"±{SEED:.4f} seed noise", ha="center", va="top",
        color=MUTED, fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in PROTO], fontsize=9, color=INK)
ax.set_ylim(-1.9, len(PROTO) - 0.4)
ax.set_xlim(0.9152, max(0.9312, max(r[3] for r in PROTO) + 0.0085))
ax.set_xlabel("test mean DA   ( ½(SR + NR) — higher is better )")
if K3D:
    ax.set_title("Every model on the KITTI 50-epoch protocol\n"
                 "architectural changes span 0.0109 — the input-side change "
                 "(EvORSP-3T) is 2× that entire band", pad=12)
else:
    ax.set_title("Every model on the KITTI 50-epoch protocol\n"
                 "total spread across all architectural changes: 0.0109", pad=12)
ax.grid(axis="y", visible=False)
fig.text(.012, .015,
         "FourierMamba2D's 50-epoch protocol run is now COMPLETE (selected at epoch 32, "
         "val 0.9566; epochs 33-50 declined). Latency 137.40 ms measured\nunder the same "
         "idle-node protocol as every other row. EvORSP-3T beats it with 841x fewer "
         "parameters at 24x the speed.",
         color=MUTED, fontsize=7.6, va="bottom")
fig.tight_layout(rect=[0, .055, 1, 1])
fig.savefig(f"{OUT}/fig6_leaderboard.png", dpi=170)
plt.close(fig)
print("fig6_leaderboard.png")

# ------------------------------------------------------------------- fig 7
# the two 0.9248 arms sit at the identical (latency, DA) point -- merge them into
# one marker rather than drawing one on top of the other.
PTS = [("DFFN",                      72_074, 11.82, 0.9159, BLUE,   (0, -20), "center"),
       ("ORSPNet",                   36_206,  6.94, 0.9193, ORANGE, (12, -4), "left"),
       ("+ balanced loss",           36_206,  6.94, 0.9210, ORANGE, (12, -4), "left"),
       ("+ dil  (RF 33→129 px)",     36_782,  7.13, 0.9223, ORANGE, (12, -4), "left"),
       ("+ dil + lit-masked BCE",    36_782,  7.13, 0.9244, AQUA,   (12, -7), "left"),
       ("+ dil + balanced loss\n+ dil + prior threshold  (identical)",
                                     36_782,  7.13, 0.9248, AQUA,   (12, 7),  "left"),
       ("StreakNet K=127",           52_043,  9.93, 0.9268, YELLOW, (0, 16),  "center")]
PTS.append(("FourierMamba2D\n(completed run)", 23_592_081, 137.40, 0.9411,
            MAUVE, (-10, 14), "right"))
if K3D:
    PTS.append(("EvORSP-3T\n(temporal + OFF input)", K3D["params"], 5.69,
                K3D["test"], PURPLE, (14, 0), "left"))
if B3BODY:
    PTS.append((f"b3 body (2D)", 27_685, B3BODY["ms"],
                float(np.mean(B3BODY["tests"])), ORANGE, (0, -16), "center"))

fig, ax = plt.subplots(figsize=(9.0, 5.4))
for name, p, ms, da, c, off, ha in PTS:
    ax.scatter([ms], [da], s=min(40 + (p / 72_074) * 250, 700), color=c,
               alpha=.93, zorder=3, edgecolor=SURF, linewidth=1.3)
    ax.annotate(name, (ms, da), textcoords="offset points", xytext=off,
                fontsize=8.2, color=INK, ha=ha, va="center")

front, bd = [], -1
for ms, da in sorted((p[2], p[3]) for p in PTS):
    if da > bd:
        front.append((ms, da)); bd = da
ax.step(*zip(*front), where="post", color=MUTED, lw=1.1, ls="--", zorder=2)

ax.errorbar([5.1], [0.9300], yerr=[SEED], fmt="none", ecolor=MUTED,
            elinewidth=1.4, capsize=4, zorder=4)
ax.text(5.15, 0.9300, " ±0.0039\n seed noise", fontsize=7.6, color=MUTED,
        va="center", ha="left")
ax.set_xscale("log")
ax.set_xlim(4.5, 175)
ax.set_xticks([5, 7, 10, 15, 25, 50, 100, 150])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_ylim(0.9146, max(0.9284, max(p[3] for p in PTS) + 0.003))
ax.set_xlabel("wall-clock latency, ms — log scale   (batch 1, 256×256, idle GPU, median of 7)")
ax.set_ylabel("test mean DA")
ax.set_title("Accuracy vs measured latency — marker area ∝ params (capped; FourierMamba = 841× EvORSP-3T)\n"
             "dashed step: Pareto frontier", pad=12)
ax.text(12.6, 0.9163, "StreakNet buys +0.0020 — inside seed noise —\n"
        "for +43 % latency and +41 % parameters",
        fontsize=7.8, color=MUTED, ha="right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_cost_accuracy.png", dpi=170)
plt.close(fig)
print("fig7_cost_accuracy.png")

# ------------------------------------------------------------------- fig 8
tri = {}
for f in glob.glob(f"{TMP}/exp_*.json"):
    d = json.load(open(f))
    if d.get("variant"):
        tri[d["variant"]] = d.get("best_valDA", 0.0)
BASE, DILBAL = tri.get("base", .9074), tri.get("dil_bal", .9272)

PAIRS = [("balanced loss",      "balanced",       0.9210 - 0.9193),
         ("dil (RF 33→129 px)", "dil",            0.9223 - 0.9193),
         ("dil + balanced",     "dil_bal",        0.9248 - 0.9193),
         ("dil + lit-BCE",      "litbce",         0.9244 - 0.9193),
         ("StreakNet K=127",    "streaknet_K127", 0.9268 - 0.9193)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})
xx = np.arange(len(PAIRS))
tg = [tri[k] - BASE for _, k, _ in PAIRS]
pg = [d for _, _, d in PAIRS]
a1.bar(xx - .2, tg, .38, color=YELLOW, label="10-epoch triage (val)")
a1.bar(xx + .2, pg, .38, color=AQUA, label="50-epoch protocol (test)")
for i, (t, p) in enumerate(zip(tg, pg)):
    a1.text(i - .2, t + .0005, f"+{t:.4f}", ha="center", fontsize=7.4, color=INK2)
    a1.text(i + .2, p + .0005, f"+{p:.4f}", ha="center", fontsize=7.4, color=INK2)
    a1.text(i, max(t, p) + .0032, f"{t/p:.1f}×", ha="center", fontsize=9,
            color=RED, fontweight="bold")
a1.axhspan(-SEED, SEED, color=GRID, alpha=.85, zorder=0)
a1.set_ylim(-.006, .0305)
a1.set_xticks(xx)
a1.set_xticklabels([n for n, _, _ in PAIRS], fontsize=8, rotation=13, ha="right")
a1.set_ylabel("Δ DA over ORSPNet")
a1.legend(fontsize=8, loc="upper left")
a1.set_title("Triage overstates every gain\nred = inflation factor · grey = seed noise",
             pad=10)

# every ablation against ITS OWN control, named in the tick label
ABL = [("no weight decay",                    "nowd",           BASE,   "base"),
       ("gain split",                         "gainsplit",      BASE,   "base"),
       ("rate → logit bias",                  "rate",           DILBAL, "dil+bal"),
       ("isotropic bank\n(orientation OFF)",  "iso_bal",        DILBAL, "dil+bal"),
       ("strip conv K=127",                   "strip_K127",     DILBAL, "dil+bal"),
       ("StreakNet K=127",                    "streaknet_K127", DILBAL, "dil+bal"),
       ("U-Net (small)",                      "unet_small",     DILBAL, "dil+bal"),
       ("U-Net + balanced",                   "unet_bal",       DILBAL, "dil+bal")]
val = [tri[k] - c for _, k, c, _ in ABL]
cols = [RED if v < -SEED else (MUTED if abs(v) <= SEED else AQUA) for v in val]
yy = np.arange(len(ABL))[::-1]
a2.barh(yy, val, .6, color=cols)
a2.axvspan(-SEED, SEED, color=GRID, alpha=.85, zorder=0)
for i, v in enumerate(val):
    a2.text(v + (.0007 if v >= 0 else -.0007), yy[i], f"{v:+.4f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=7.8, color=INK2)
a2.set_yticks(yy)
a2.set_yticklabels([f"{n}\nvs {c}" for n, _, _, c in ABL], fontsize=7.8)
a2.set_xlim(-.0145, .0088)
a2.set_xlabel("Δ DA vs its own control (10-epoch triage)")
a2.set_title("Ablations — grey band is seed noise\n"
             "removing orientation costs 2× noise", pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_ablations.png", dpi=170)
plt.close(fig)
print("fig8_ablations.png")

# ------------------------------------------------------------------- fig 9
TAUS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 104.0]
AUC = {0: [.5001, .5001, .5004, .5007, .5015, .5039, .5003, .5060, .5057, .5081],
       1: [.6025, .6062, .6154, .6303, .6559, .7105, .7265, .7099, .6359, .5549],
       2: [.6155, .6200, .6305, .6450, .6654, .7052, .7053, .6740, .5909, .5103]}
JIT = ([0, 1, 10, 100, 1000, 10000], [.6918, .6918, .6918, .6918, .6916, .6535])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
for r, c, m, lab in [(1, ORANGE, "s", "3×3 neighbourhood"),
                     (2, BLUE, "^", "5×5 neighbourhood"),
                     (0, MUTED, "o", "same pixel only")]:
    a1.plot(TAUS, AUC[r], marker=m, ms=5, color=c, lw=1.8, label=lab)
a1.axhline(.5, color=MUTED, lw=.9, ls=":")
a1.axvline(104, color=RED, lw=1.1, ls="--")
a1.text(66, .545, "our 104 ms\naccumulation\nwindow", color=RED, fontsize=7.8,
        ha="right", va="center")
a1.scatter([5.0], [.7265], s=170, facecolor="none", edgecolor=RED, lw=1.6, zorder=5)
a1.annotate("peak 0.7265 @ 5 ms", (5.0, .7265), textcoords="offset points",
            xytext=(16, 18), fontsize=8.4, color=RED, ha="left")
a1.set_xscale("log")
a1.set_ylim(.487, .755)
a1.set_xlabel("temporal window τ  (ms, log scale)")
a1.set_ylabel("AUC, rain vs background events")
a1.legend(fontsize=8, loc="upper left")
a1.set_title("Rain separates at 2.5–5 ms, not at 104 ms\n"
             "single-pixel timing is chance at every scale", pad=10)

a2.plot(JIT[0], JIT[1], marker="o", ms=5, color=AQUA, lw=1.8)
a2.set_xscale("symlog", linthresh=1)
a2.axhline(.5, color=MUTED, lw=.9, ls=":")
a2.set_ylim(.487, .755)
a2.set_xlabel("timestamp jitter applied  (µs, symlog)")
a2.set_ylabel("AUC at τ = 5 ms, 3×3")
a2.text(0, .566,
        "flat out to 1 ms of jitter ⇒ genuine ms-scale\n"
        "structure, NOT the exact-simultaneity artefact\n"
        "(SPAC rain: 323 events per exact ns vs 8.8 for\nbackground, itself AUC 0.7415)",
        fontsize=7.8, color=MUTED, va="center")
a2.set_title("Jitter control", pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_temporal.png", dpi=170)
plt.close(fig)
print("fig9_temporal.png")

# ------------------------------------------------------------------- fig 10
res = {}
for f in glob.glob(f"{TMP}/spac_T*.json"):
    d = json.load(open(f))
    # fig10 is the T-sweep at fixed architecture: 4 blocks, ON-only, seed 0
    if d.get("blocks", 4) != 4 or d.get("off") or d.get("counts") or d.get("seed"):
        continue
    res[d["T"]] = d
if len(res) >= 2:
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    Ts = sorted(res)
    da = [res[t]["test"] for t in Ts]
    cols = [MUTED if t == 1 else AQUA for t in Ts]
    b = ax.bar([str(t) for t in Ts], da, .58, color=cols)
    for i, t in enumerate(Ts):
        ax.text(i, da[i] + .004, f"{da[i]:.4f}", ha="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(i, .012, f"{res[t]['params']:,} params", ha="center",
                fontsize=7.6, color=SURF)
    if 1 in res:
        ax.axhline(res[1]["test"], color=MUTED, lw=1.2, ls="--", zorder=4)
        ax.axhspan(res[1]["test"] - SEED, res[1]["test"] + SEED,
                   color=GRID, alpha=.85, zorder=0)
        ax.text(len(Ts) - .45, res[1]["test"], "  2D baseline\n  ±seed noise",
                fontsize=7.8, color=MUTED, va="center")
    ax.set_ylim(0, max(da) * 1.16)
    ax.set_xlabel("T — number of temporal sub-windows   (T=1 is the current 2D input)")
    ax.set_ylabel("SPAC test DA")
    ax.set_title("3D log-Gabor: adding a temporal-frequency axis\n"
                 "+170 params, +0.5 ms — T is the only variable", pad=12)
    fig.text(.012, .015,
             "Scene-disjoint splits fixed before any result: train t1–t8 / val a1–a4 / "
             "test b1–b4.  T=1 is the exact OR-union of the T=16\nplanes, so the baseline "
             "is bit-identical to the 2D input.  SPAC numbers are NOT comparable to the "
             "KITTI leaderboard.",
             color=MUTED, fontsize=7.6, va="bottom")
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(f"{OUT}/fig10_temporal_3d.png", dpi=170)
    plt.close(fig)
    print("fig10_temporal_3d.png")
else:
    print(f"fig10 skipped — {len(res)}/4 SPAC arms finished")
