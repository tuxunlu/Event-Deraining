"""Run a trained EvORSP trunk on your own events. No dataset conventions assumed.

    python evorsp/derain.py --events /path/to/data --out /path/to/derained

`--events` is either a directory of .npz files (one per time window) or a single
.npz holding a longer stream, which `--window` will slice. Each file needs the
arrays `x`, `y`, `t`, `p`; `t` may be in any unit and any origin, because every
window is normalised to its own span before binning. Sensor size is inferred
from the data unless you pass `--width/--height`.

Output is one .npz per window with the kept (scene) events, plus `keep`, the
boolean mask over the input events, so you can recover what was dropped.

If you also have per-event labels, pass `--labels <dir>` and it reports
event-level DA -- balanced accuracy over individual events, which is the metric
PRE-Mamba reports and the one to quote. Say `--labels-rain-is 0` if your files
mark rain with 0 (the real EVK4 convention) or `1` if the reverse. Guessing this
wrong silently inverts every number, so the script refuses to assume.

This runs the TRUNK: 28,924 parameters, 5.61 ms per window, and its cost does
not grow with event count. The per-event head adds ~0.02 event-DA at a
rate-dependent ~25 ms and needs a feature cache; see train/run_kitti_headv3.py.
"""
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()

import argparse
import glob
import json
import os

import numpy as np
import torch

from rsp_3d import ORSPNet3D

T_BUILD, R = 16, 256
EPS = 1e-6


def load_model(ckpt, device):
    """Resolve a checkpoint by name or path; its json sidecar defines the config."""
    have = sorted(os.path.basename(q)[:-3] for q in glob.glob(f"{C.CKPT}/*.pt"))
    if ckpt == "list":
        raise SystemExit("checkpoints in %s:\n  %s" % (C.CKPT, "\n  ".join(have)))
    p = ckpt if os.path.exists(ckpt) else f"{C.CKPT}/{ckpt}"
    if not p.endswith(".pt"):
        p += ".pt"
    j = p[:-3] + ".json"
    if not os.path.exists(p):
        raise SystemExit(f"no checkpoint at {p}\navailable:\n  "
                         + "\n  ".join(have))
    cfg = json.load(open(j)) if os.path.exists(j) else {}
    tf = cfg.get("tfront", 4)
    to = cfg.get("tout", 16)
    ctx = cfg.get("ctx", 0)
    counts = cfg.get("counts", False)
    n_extra = 2 * ctx + (1 if counts else 0)

    m = ORSPNet3D(T=tf, dilations=(1, 8, 32, 64), num_blocks=3, use_off=True,
                  out_chans=to, n_extra=n_extra).to(device).eval()
    blob = torch.load(p, map_location=device, weights_only=False)
    m.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    cfg.update(tfront=tf, tout=to, ctx=ctx, counts=counts, n_extra=n_extra,
               path=p)
    return m, cfg


def grid(x, y, t, p, w, h):
    """Events -> (on, off, cell index per event, time bin per event).

    Binning matches the training packs exactly: space rescaled to 256x256,
    time to 16 bins normalised within the window. Because time is normalised
    per window, the timestamp UNIT is irrelevant here -- which is the one trap
    that costs the most elsewhere in this codebase.
    """
    sx = np.clip((x.astype(np.int64) * R) // w, 0, R - 1)
    sy = np.clip((y.astype(np.int64) * R) // h, 0, R - 1)
    t0 = t.min()
    span = max(int(t.max() - t0), 1)
    tb = np.clip(((t.astype(np.int64) - t0) * T_BUILD) // span, 0, T_BUILD - 1)
    cell = tb * (R * R) + sy * R + sx

    on = np.zeros(T_BUILD * R * R, bool)
    off = np.zeros(T_BUILD * R * R, bool)
    is_on = p == 1
    on[cell[is_on]] = True
    off[cell[~is_on]] = True
    return (on.reshape(T_BUILD, R, R), off.reshape(T_BUILD, R, R),
            sx, sy, tb)


class Ctx:
    """Rolling ON/OFF unions of the preceding K windows, clamped at the start.

    Training read these from neighbouring pack files as `max(idx - k, 0)`, so
    the first frame's "previous" window is ITSELF, not zeros. Reproducing that
    clamp matters: getting it wrong costs ~0.001 event-DA on a 166-frame
    sequence and more on short ones. Push the current window first, then index
    backwards with the same clamp.
    """

    def __init__(self, k):
        self.k, self.buf = k, []

    def push(self, on, off):
        self.buf.append((on.max(0).astype(np.float32),
                         off.max(0).astype(np.float32)))
        if len(self.buf) > self.k + 1:
            self.buf.pop(0)

    def planes(self):
        out = []
        n = len(self.buf)
        for i in range(1, self.k + 1):
            on, off = self.buf[max(n - 1 - i, 0)]
            out += [on[None], off[None]]
        return out


@torch.no_grad()
def run_window(m, cfg, on, off, n_ev_cell, ctx, device):
    """-> per-cell keep probability [T_out, R, R] and the self-prior threshold."""
    tf, to = cfg["tfront"], cfg["tout"]
    k = T_BUILD // tf
    onf = on.reshape(tf, k, R, R).max(1).astype(np.float32)
    offf = off.reshape(tf, k, R, R).max(1).astype(np.float32)

    extra = ctx.planes() if cfg["ctx"] else []
    if cfg["counts"]:
        cnt = n_ev_cell.reshape(T_BUILD, R, R).sum(0)
        extra.append((np.log1p(cnt) / 4.0)[None].astype(np.float32))
    ex = np.concatenate(extra, 0) if extra else np.zeros((0, R, R), np.float32)

    kw = {"x_extra": torch.from_numpy(ex)[None].to(device)} if cfg["n_extra"] else {}
    prob = torch.sigmoid(m(torch.from_numpy(onf)[None].to(device),
                           x_off=torch.from_numpy(offf)[None].to(device),
                           **kw))[0].cpu().numpy()

    # Self-prior threshold: the count-weighted mean probability over this frame.
    # Worth ~+0.006 over a fixed tau and needs no labels, so it survives
    # deployment. Falls back to the trained tau on an empty frame.
    if cfg.get("fixed_tau") is not None:
        return prob, cfg["fixed_tau"]
    cnt_out = n_ev_cell.reshape(to, T_BUILD // to, R, R).sum(1)
    tot = cnt_out.sum()
    tau = float((prob * cnt_out).sum() / tot) if tot > 0 else cfg.get("tau", 0.5)
    return prob, tau


def windows(args):
    """Yield (name, x, y, t, p, labels_or_None) for each time window."""
    src = args.events
    files = ([src] if src.endswith(".npz")
             else sorted(glob.glob(f"{src}/**/*.npz", recursive=True)))
    if not files:
        raise SystemExit(f"no .npz under {src}")
    for f in files:
        with np.load(f) as d:
            missing = [k for k in ("x", "y", "t", "p") if k not in d]
            if missing:
                print(f"  skip {os.path.basename(f)}: missing {missing}")
                continue
            x, y, t, p = (d[k] for k in ("x", "y", "t", "p"))
        lab = None
        if args.labels:
            lp = os.path.join(args.labels, os.path.basename(f)[:-4] + ".npy")
            if not os.path.exists(lp):
                lp = os.path.join(args.labels,
                                  "labels_" + os.path.basename(f)[:-4] + ".npy")
            if os.path.exists(lp):
                lab = np.load(lp)
                if len(lab) != len(x):
                    lab = None

        rel = os.path.relpath(f, src if not src.endswith(".npz")
                              else os.path.dirname(src))
        if args.window and len(files) == 1:
            order = np.argsort(t)
            x, y, t, p = x[order], y[order], t[order], p[order]
            lab = lab[order] if lab is not None else None
            edges = np.arange(t.min(), t.max() + args.window, args.window)
            for i in range(len(edges) - 1):
                s = (t >= edges[i]) & (t < edges[i + 1])
                if s.sum() >= args.min_events:
                    yield (f"{i:010d}.npz", x[s], y[s], t[s], p[s],
                           lab[s] if lab is not None else None)
        elif len(x) >= args.min_events:
            yield rel, x, y, t, p, lab


def main():
    ap = argparse.ArgumentParser(
        description="Derain an event stream with a trained EvORSP trunk.")
    ap.add_argument("--events",
                    help="directory of .npz windows, or one .npz stream")
    ap.add_argument("--out", help="output directory (omit to only score)")
    ap.add_argument("--ckpt", default="ctx_f4o16_c2",
                    help="checkpoint name in evorsp/checkpoints, or a path")
    ap.add_argument("--width", type=int, help="sensor width (inferred if unset)")
    ap.add_argument("--height", type=int)
    ap.add_argument("--window", type=int,
                    help="slice a single-file stream into windows this long, "
                         "in the file's own time unit")
    ap.add_argument("--min-events", type=int, default=200)
    ap.add_argument("--tau-trained", action="store_true",
                    help="use the tau selected on the validation split, as "
                         "recorded in the checkpoint json")
    ap.add_argument("--labels", help="directory of per-event .npy labels")
    ap.add_argument("--labels-rain-is", type=int, choices=(0, 1),
                    help="label value meaning RAIN (real EVK4 uses 0)")
    ap.add_argument("--tau", type=float,
                    help="fixed decision threshold; default is the per-frame "
                         "self-prior, which needs no labels at deployment")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.ckpt == "list":
        load_model("list", "cpu")            # prints the list and exits
    if not a.events:
        ap.error("--events is required")
    if a.labels and a.labels_rain_is is None:
        raise SystemExit("--labels needs --labels-rain-is 0|1; guessing it "
                         "wrong inverts every reported number")

    m, cfg = load_model(a.ckpt, a.device)
    cfg["fixed_tau"] = a.tau if a.tau is not None else (
        cfg.get("tau") if a.tau_trained else None)
    print(f"{os.path.basename(cfg['path'])}: "
          f"T_front {cfg['tfront']} -> T_out {cfg['tout']}, ctx {cfg['ctx']}, "
          f"counts {cfg['counts']} | tau "
          f"{cfg['fixed_tau'] if cfg['fixed_tau'] is not None else 'self-prior'} | {sum(q.numel() for q in m.parameters()):,} "
          f"params | {a.device}")

    ctx = Ctx(cfg["ctx"])
    das, kept_frac, n = [], [], 0
    for name, x, y, t, p, lab in windows(a):
        w = a.width or int(x.max()) + 1
        h = a.height or int(y.max()) + 1
        on, off, sx, sy, tb = grid(x, y, t, p, w, h)

        n_cell = np.bincount(tb * (R * R) + sy * R + sx,
                             minlength=T_BUILD * R * R).astype(np.float32)
        ctx.push(on, off)
        prob, tau = run_window(m, cfg, on, off, n_cell, ctx, a.device)

        to = cfg["tout"]
        tb_out = (tb * to) // T_BUILD
        keep = prob[tb_out, sy, sx] > tau
        kept_frac.append(float(keep.mean()))

        if lab is not None:
            is_rain = lab == a.labels_rain_is
            nb, nr = (~is_rain).sum(), is_rain.sum()
            if nb > 0 and nr > 0:
                das.append(0.5 * (keep[~is_rain].mean() + (~keep[is_rain]).mean()))

        if a.out:
            dst = os.path.join(a.out, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, x=x[keep], y=y[keep], t=t[keep],
                                p=p[keep], keep=keep)
        n += 1
        if n % 200 == 0:
            print(f"  {n} windows", flush=True)

    print(f"\n{n} windows | kept {100*np.mean(kept_frac):.1f}% of events")
    if das:
        print(f"event-DA {np.mean(das):.4f} over {len(das)} labelled windows")
    elif a.labels:
        print("no label file matched a window -- check --labels")
    if a.out:
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
