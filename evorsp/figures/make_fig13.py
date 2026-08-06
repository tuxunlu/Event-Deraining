"""fig13: the supervision result -- EvORSP-3T vs PRE-Mamba on the event metric.

Left: the diagnosis. Event-DA of each candidate per-pixel TARGET (measured
ceilings) with our old and new trained models placed against them. The old
model sits ON its target's ceiling -- it was saturated, not weak.
Right: cost. Same accuracy, 9.4x fewer params, ~55-100x faster.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/figs"
BLUE, MAUVE, GREY = "#3b6fb6", "#9b6b9e", "#b9b9b9"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                              gridspec_kw={"width_ratios": [1.5, 1]})

# ---- left: targets, ceilings, achieved -------------------------------------
names = ["A  ≥1 clean ON event\n(old target)", "B  ≥1 clean event\nany polarity",
         "C  background count\n> rain count", "D  BA-optimal\n(ceiling)"]
ceil = [0.6981, 0.8789, 0.9440, 0.9466]
xs = np.arange(4)
ax.bar(xs, ceil, 0.62, color=GREY, label="ceiling of that target (measured)")
for x, c in zip(xs, ceil):
    # nudge labels off bars A and C, where the achieved-value markers sit
    dx = -0.30 if x in (0, 2) else 0.0
    ha = "right" if dx else "center"
    ax.annotate(f"{c:.4f}", (x + dx, c), ha=ha, va="bottom", fontsize=10,
                color="0.35")

ax.plot([0], [0.7052], "o", ms=13, color=BLUE, zorder=5)
ax.annotate("EvORSP-3T trained on A\n0.7052 — AT its ceiling",
            (0, 0.7052), xytext=(0.15, 0.60), fontsize=10, color=BLUE,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
ax.plot([2], [0.9196], "o", ms=13, color=BLUE, zorder=5)
ax.annotate("retrained on C (+temporal readout)\n0.9196 ± 0.0013 test (2 seeds)",
            (2, 0.9196), xytext=(1.25, 0.985), fontsize=10, color=BLUE,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
ax.axhline(0.9172, color=MAUVE, lw=2, ls="--",
           label="PRE-Mamba 0.9172 (264,632 params, 306 ms)")

ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0.55, 1.02)
ax.set_ylabel("event-level DA, KITTI test {50,150}mm")
ax.set_title("The gap was the supervision target, not the architecture",
             fontsize=13)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---- right: cost at equal accuracy -----------------------------------------
labels = ["EvORSP-3T/E\n28,555 params", "PRE-Mamba\n264,632 params"]
lat = [5.74, 306.1]
bars = ax2.bar(labels, lat, color=[BLUE, MAUVE], width=0.5)
for r, v, da in zip(bars, lat, [0.9196, 0.9172]):
    ax2.annotate(f"{v:.1f} ms\nDA {da:.4f}", (r.get_x() + r.get_width() / 2, v),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_yscale("log")
ax2.set_ylim(3, 2000)
ax2.set_ylabel("latency per 100 ms window (ms, log)")
ax2.set_title("Same event-level accuracy, 53× faster", fontsize=13)
ax2.grid(axis="y", alpha=0.3)
ax2.text(0.5, 0.02, "idle-node protocol, spreads 0.07 and 1.8 ms;\n"
         "ours rate-invariant, theirs scales with event count",
         transform=ax2.transAxes, ha="center", va="bottom", fontsize=8.5,
         style="italic", color="0.35")

fig.suptitle("EvORSP-3T vs PRE-Mamba (ICCV'25) on PRE-Mamba's own per-event "
             "metric — same test frames", fontsize=14, y=1.00)
fig.tight_layout()
fig.savefig(f"{FIG}/fig13_premamba_2x2.png", dpi=170, bbox_inches="tight")
print("saved fig13")
