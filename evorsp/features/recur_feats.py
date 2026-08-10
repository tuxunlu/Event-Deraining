"""Spatial recurrence + long-baseline persistence, per event.

Two features aimed at the residue the ITI columns do not explain: the rig's
VERTICAL nozzle columns (worst in scene3/rain_9, scene1/rain_2), which survive
because they satisfy the "persistent => scene" rule the model relies on.

(3) MULTI-SCALE, MULTI-WINDOW PERSISTENCE.  Present persistence is binary and
    one window deep. Over K windows a NOZZLE and a BUILDING EDGE behave
    differently BY SCALE: the edge re-fires at the SAME PIXEL every window,
    while the column's drops scatter within the nozzle's x-range, so it recurs
    at 16 px but not at 1 px. The discriminator is therefore the RATIO of fine-
    to coarse-scale persistence, not persistence itself:
        scene       high at 1 px, high at 16 px  -> ratio ~ 1
        nozzle      low  at 1 px, high at 16 px  -> ratio << 1
        transient   low at both                  -> low, ratio uninformative

(2) COLUMN STATISTICS.  A nozzle writes a narrow x-strip that spans nearly the
    FULL frame height with roughly uniform density, and does so in every window.
    A vertical building edge occupies a limited y-range and is not uniform.
    Per 8 px x-strip: hotness vs the frame median, recurrence (the MINIMUM
    normalised count over the last K windows -- a nozzle is hot in all of them),
    y-extent, and y-uniformity.

12 columns. Computed with a rolling buffer so each frame is read once.
"""
import numpy as np

NW, NH = 1280, 720
SCALES = (1, 4, 16)
STRIP = 8
KWIN = 8
NCOL = 12
EPS = 1e-6


class Recur:
    """Rolling state over the last KWIN windows of one sequence."""

    def __init__(self, nw=NW, nh=NH, kwin=KWIN):
        self.nw, self.nh, self.kwin = nw, nh, kwin
        self.hist = {s: [] for s in SCALES}          # occupancy masks
        self.strips = []                             # normalised strip counts
        self.nb = nw // STRIP + 1

    def reset(self):
        """Clear history. MUST be called between sequences."""
        self.hist = {s: [] for s in SCALES}
        self.strips = []

    def _masks(self, x, y):
        out = {}
        for s in SCALES:
            wt = self.nw // s + 1
            m = np.zeros(wt * (self.nh // s + 1), bool)
            m[(y // s) * wt + (x // s)] = True
            out[s] = m
        return out

    def _stripcount(self, x):
        c = np.bincount(np.clip(x // STRIP, 0, self.nb - 1),
                        minlength=self.nb).astype(np.float64)
        med = np.median(c[c > 0]) if (c > 0).any() else 1.0
        return c / max(med, 1.0)

    def features(self, x, y, sel):
        """[len(sel), 12] using ONLY windows strictly before this one."""
        n = len(sel)
        f = np.zeros((n, NCOL), np.float32)
        xs, ys = x[sel], y[sel]

        # ---- (3) persistence over the last K windows, per scale
        pers = {}
        for si, s in enumerate(SCALES):
            wt = self.nw // s + 1
            tid = (ys // s) * wt + (xs // s)
            h = self.hist[s]
            if h:
                acc = np.zeros(n, np.float32)
                for m in h:
                    acc += m[tid]
                pers[s] = acc / len(h)
            else:
                pers[s] = np.zeros(n, np.float32)
            f[:, si] = pers[s]
        # fine/coarse ratio: the nozzle signature
        f[:, 3] = pers[1] / np.maximum(pers[16], EPS)
        f[:, 4] = pers[4] / np.maximum(pers[16], EPS)

        # ---- (2) column statistics for THIS window
        sb = np.clip(xs // STRIP, 0, self.nb - 1)
        cur = self._stripcount(x)
        f[:, 5] = np.log1p(cur[sb])                              # hotness

        if self.strips:
            hist = np.stack(self.strips, 0)                      # [K, nb]
            f[:, 6] = np.log1p(hist.min(0)[sb])                  # recurrence
            f[:, 7] = np.log1p(hist.mean(0)[sb])
            f[:, 8] = hist.std(0)[sb] / np.maximum(hist.mean(0)[sb], EPS)

        # y-extent and y-uniformity of each strip, this window
        allb = np.clip(x // STRIP, 0, self.nb - 1)
        ymin = np.full(self.nb, np.inf)
        ymax = np.full(self.nb, -np.inf)
        np.minimum.at(ymin, allb, y)
        np.maximum.at(ymax, allb, y)
        ext = np.where(np.isfinite(ymin) & np.isfinite(ymax),
                       (ymax - ymin) / self.nh, 0.0)
        f[:, 9] = ext[sb]
        # uniformity along y: entropy of an 8-bin y histogram per strip
        ybin = np.clip(y * 8 // self.nh, 0, 7).astype(np.int64)
        h2 = np.zeros((self.nb, 8))
        np.add.at(h2, (allb, ybin), 1.0)
        p = h2 / np.maximum(h2.sum(1, keepdims=True), 1.0)
        ent = -(p * np.log(np.maximum(p, EPS))).sum(1) / np.log(8)
        f[:, 10] = ent[sb]
        f[:, 11] = np.log1p(h2.sum(1))[sb]
        return f

    def push(self, x, y):
        for s in SCALES:
            self.hist[s].append(self._masks(x, y)[s])
            if len(self.hist[s]) > self.kwin:
                self.hist[s].pop(0)
        self.strips.append(self._stripcount(x))
        if len(self.strips) > self.kwin:
            self.strips.pop(0)


NAMES = ["pers_K8_s1", "pers_K8_s4", "pers_K8_s16", "ratio_s1_s16",
         "ratio_s4_s16", "strip_hot", "strip_recur_min", "strip_recur_mean",
         "strip_cv", "strip_yextent", "strip_yentropy", "strip_count"]
