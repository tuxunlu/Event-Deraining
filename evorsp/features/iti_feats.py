"""Inter-arrival-time regularity, per event, on GPU.

WHY. The measured failure on real EVK4: rain we wrongly keep is temporally
PERSISTENT (0.412 of its pixels were active in the previous frame, against 0.154
for rain we correctly drop and 0.511 for genuine scene). The rig pours
continuous water COLUMNS that occupy fixed pixels, so they satisfy the
"persistent => scene" rule the model relies on. Persistence cannot separate
them; the timing STATISTICS can.

A continuous stream is a stochastic emitter -- its events arrive at a pixel with
near-Poisson spacing. A scene edge fires in bursts tied to camera/object motion.
Four classical irregularity statistics separate those regimes:

    log mean dt   rate proxy
    CV            sigma/mu of inter-arrival times.   Poisson = 1
    LV            local variation, (3/(n-1)) * sum ((dt_i - dt_i+1) /
                  (dt_i + dt_i+1))^2. Poisson = 1, regular -> 0, bursty > 1.
                  Robust to slow RATE drift, which matters because rain rate
                  varies within a window -- CV is not.
    B             burstiness (sigma - mu)/(sigma + mu). Poisson = 0.

Computed per TILE rather than per pixel: at 1280x720 a 100 ms window averages
0.43 events/pixel, so single-pixel inter-arrival times are mostly undefined.
Two scales (4 and 16 px) give ~7 and ~110 events per tile.

Output: [N, 8] = 4 statistics x 2 scales, each broadcast to the events in its
tile. Undefined tiles (fewer than 3 events) emit zeros, flagged by CV = 0.
"""
import numpy as np
import torch

STATS = 4


def iti_gpu(x, y, t, scales=(4, 16), nw=1280, nh=720, dev="cuda"):
    """[N, 4*len(scales)] inter-arrival-time regularity columns."""
    n = x.shape[0]
    out = torch.zeros(n, STATS * len(scales), device=dev, dtype=torch.float64)
    td = t.double()
    for si, s in enumerate(scales):
        wt = nw // s + 1
        tile = (y // s) * wt + (x // s)
        ntile = wt * (nh // s + 1)

        # sort by (tile, time) so each tile is one contiguous, time-ordered run
        key = tile * (td.max() - td.min() + 2) + (td - td.min() + 1)
        order = torch.argsort(key)
        ts, tl = td[order], tile[order]

        # dt between successive events INSIDE a tile (invalid at tile borders)
        dt = ts[1:] - ts[:-1]
        same = tl[1:] == tl[:-1]
        dt = torch.where(same, dt, torch.zeros_like(dt))
        owner = tl[1:]                                    # tile each dt belongs to
        w = same.double()

        cnt = torch.zeros(ntile, device=dev, dtype=torch.float64)
        cnt.index_add_(0, owner, w)
        s1 = torch.zeros(ntile, device=dev, dtype=torch.float64)
        s1.index_add_(0, owner, dt * w)
        s2 = torch.zeros(ntile, device=dev, dtype=torch.float64)
        s2.index_add_(0, owner, dt * dt * w)

        # LV needs CONSECUTIVE dt pairs, both inside the same tile
        d1, d2 = dt[:-1], dt[1:]
        pair = same[:-1] & same[1:]
        denom = torch.clamp(d1 + d2, min=1e-9)
        lv_term = ((d1 - d2) / denom) ** 2 * pair.double()
        lcnt = torch.zeros(ntile, device=dev, dtype=torch.float64)
        lcnt.index_add_(0, owner[1:], pair.double())
        lsum = torch.zeros(ntile, device=dev, dtype=torch.float64)
        lsum.index_add_(0, owner[1:], lv_term)

        ok = cnt >= 3
        mu = s1 / torch.clamp(cnt, min=1)
        var = torch.clamp(s2 / torch.clamp(cnt, min=1) - mu * mu, min=0)
        sd = torch.sqrt(var)
        cv = torch.where(ok, sd / torch.clamp(mu, min=1e-9),
                         torch.zeros_like(mu))
        lv = torch.where(lcnt >= 2, 3.0 * lsum / torch.clamp(lcnt, min=1),
                         torch.zeros_like(mu))
        burst = torch.where(ok, (sd - mu) / torch.clamp(sd + mu, min=1e-9),
                            torch.zeros_like(mu))
        lmu = torch.where(ok, torch.log1p(mu), torch.zeros_like(mu))

        col = si * STATS
        out[:, col + 0] = lmu[tile]
        out[:, col + 1] = cv[tile]
        out[:, col + 2] = lv[tile]
        out[:, col + 3] = burst[tile]
    return out.float()


if __name__ == "__main__":
    # sanity: a REGULAR train, a POISSON train and a BURSTY train must land at
    # the textbook LV values (0, 1, >1) and burstiness (-1, 0, +1)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N = 4000
    for name, gen in (
        ("regular", lambda k: np.arange(k) * 1000.0),
        ("poisson", lambda k: np.cumsum(np.random.exponential(1000.0, k))),
        ("bursty", lambda k: np.cumsum(np.where(
            np.random.rand(k) < 0.8, np.random.exponential(50.0, k),
            np.random.exponential(5000.0, k)))),
    ):
        t = np.sort(gen(N)).astype(np.int64)
        x = np.zeros(N, np.int64)
        y = np.zeros(N, np.int64)
        f = iti_gpu(torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                    torch.from_numpy(t).to(dev), scales=(4,), dev=dev)
        cv, lv, b = float(f[0, 1]), float(f[0, 2]), float(f[0, 3])
        print(f"  {name:8s} CV {cv:6.3f}  LV {lv:6.3f}  B {b:+6.3f}")
    print("  expected: regular CV~0 LV~0 B~-1 | poisson CV~1 LV~1 B~0 | "
          "bursty CV>1 LV>1 B>0")
