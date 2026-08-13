"""Prophesee EVT3 (.raw) -> windowed .npz, the format the wild pipeline reads.

The Drive drop is raw IMX636 recordings: an ASCII header terminated by
`% end`, then a stream of little-endian uint16 words. Nothing on this cluster
can read it -- no Metavision SDK, no dv, no expelliarmus, no numba -- so this
is a pure-numpy decoder.

EVT3 word layout. Top 4 bits are the type, low 12 the payload:

    0x0  EVT_ADDR_Y     y[10:0]              set current row
    0x2  EVT_ADDR_X     x[10:0], pol=bit11   emit ONE event here
    0x3  VECT_BASE_X    x[10:0], pol=bit11   set run origin, emits nothing
    0x4  VECT_12        12-bit mask          emit a bit-run, advance base by 12
    0x5  VECT_8          8-bit mask          emit a bit-run, advance base by 8
    0x6  EVT_TIME_LOW   t[11:0]              low 12 bits of the microsecond clock
    0x8  EVT_TIME_HIGH  t[23:12]             high 12 bits
    0xA  EXT_TRIGGER                         ignored
    0x7 / 0xE / 0xF     CONTINUED / OTHERS   ignored

Everything is stateful -- a row word governs every event until the next row
word -- which is why naive decoders end up in a Python loop over a billion
words. This one is vectorised: the running row, polarity, timestamp and vector
origin are all forward-fills, computed with `np.maximum.accumulate` over the
positions where each field is set.

TWO THINGS THAT SILENTLY CORRUPT TIMESTAMPS, both handled:

  TIME_HIGH wraps every 2^12 ticks, i.e. every 16.7 s of recording. Left alone
  the clock jumps backwards mid-file. Wraps are detected as decreases in the
  TIME_HIGH sequence and accumulated.

  Chunking breaks the state. Files here are up to 4.4 GB (~2.2e9 words) and the
  decoded stream can exceed 1e9 events, so it cannot be done in one pass. Every
  piece of state -- row, polarity, base x, both clock halves, the wrap
  accumulator -- is carried across chunk boundaries explicitly.

Timestamps come out in MICROSECONDS, absolute from the start of the recording.
`derain.py` normalises per window, so the unit does not have to match KITTI's
nanoseconds.
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

Y, X, VBASE, V12, V8, TLOW, THIGH = 0x0, 0x2, 0x3, 0x4, 0x5, 0x6, 0x8


def read_header(path):
    """The ASCII preamble, and the byte offset where binary data starts."""
    meta, off = {}, 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line or not line.startswith(b"%"):
                # the first non-% line is already binary: rewind over it
                off = f.tell() - len(line)
                break
            off = f.tell()
            txt = line[1:].strip().decode("ascii", "replace")
            if txt == "end":
                break
            k, _, v = txt.partition(" ")
            meta[k] = v
    w = h = None
    if "geometry" in meta and "x" in meta["geometry"]:
        w, h = (int(v) for v in meta["geometry"].split("x"))
    return meta, off, w, h


def _ffill(mask, vals, init):
    """For each position, the most recent `vals` where `mask` held, else init."""
    n = mask.shape[0]
    idx = np.where(mask, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.empty(n, dtype=vals.dtype)
    seen = idx >= 0
    out[seen] = vals[idx[seen]]
    out[~seen] = init
    return out


class _State:
    """Everything that must survive a chunk boundary."""

    def __init__(self):
        self.y = 0
        self.pol = 0
        self.base_x = 0
        self.t_hi = 0          # already wrap-corrected
        self.t_lo = 0
        self.base_cumw = 0     # cumulative vector width at the last VECT_BASE_X


def _decode_chunk(w, st):
    """One chunk of uint16 words -> (x, y, t, p), advancing `st` in place."""
    typ = (w >> 12).astype(np.uint8)
    pay = (w & 0x0FFF).astype(np.int32)

    # ---- clock. TIME_HIGH wraps every 4096 ticks; accumulate the wraps.
    m_hi = typ == THIGH
    hi_vals = pay[m_hi]
    if hi_vals.size:
        # PHASE UNWRAP, not wrap-counting. TIME_HIGH is a 12-bit counter that
        # both wraps and jitters: the sensor emits it slightly out of order
        # around packet boundaries. Thresholding "is this a wrap?" fails at
        # both ends -- counting every decrease inflated a 57 s recording to
        # 20,961 s, and requiring a >2048 drop still left 3,899 s.
        #
        # Instead map each step into (-2048, +2048] and accumulate. A small
        # backward step stays a small backward step; a fall from 4090 to 5
        # becomes +11. This is exactly np.unwrap on a 4096-period counter, and
        # it needs no threshold at all. Sanity: a 57 s recording is 13,965
        # ticks of 4096 us, i.e. ~3.4 genuine wraps -- so any rule reporting
        # hundreds of them was wrong by construction.
        prev0 = st.t_hi & 0xFFF
        d = np.diff(np.r_[prev0, hi_vals].astype(np.int64))
        d = ((d + 2048) % 4096) - 2048
        hi_abs = st.t_hi + np.cumsum(d)
        full_hi = np.zeros_like(pay)
        full_hi[m_hi] = hi_abs
        t_hi = _ffill(m_hi, full_hi, st.t_hi)
        st.t_hi = int(hi_abs[-1])
    else:
        t_hi = np.full(w.shape[0], st.t_hi, np.int64)

    m_lo = typ == TLOW
    t_lo = _ffill(m_lo, pay, st.t_lo)
    if m_lo.any():
        st.t_lo = int(pay[m_lo][-1])
    t = (t_hi.astype(np.int64) << 12) | t_lo.astype(np.int64)

    # ---- running row
    m_y = typ == Y
    ys = _ffill(m_y, pay & 0x7FF, st.y)
    if m_y.any():
        st.y = int((pay & 0x7FF)[m_y][-1])

    # ---- vector origin and polarity. VECT_BASE_X sets both and emits nothing;
    # each following VECT_12/VECT_8 consumes 12 or 8 columns from that origin.
    m_b = typ == VBASE
    bx = _ffill(m_b, pay & 0x7FF, st.base_x)
    bpol = _ffill(m_b, (pay >> 11) & 1, st.pol)
    if m_b.any():
        st.base_x = int((pay & 0x7FF)[m_b][-1])
        st.pol = int(((pay >> 11) & 1)[m_b][-1])

    width = np.where(typ == V12, 12, np.where(typ == V8, 8, 0)).astype(np.int64)
    cumw = np.cumsum(width) + st.base_cumw
    base_cumw = _ffill(m_b, cumw - width, st.base_cumw)
    st.base_cumw = int(cumw[-1]) if w.shape[0] else st.base_cumw

    out = []
    # ---- singles
    m_x = typ == X
    if m_x.any():
        out.append((pay[m_x] & 0x7FF, ys[m_x], t[m_x], (pay[m_x] >> 11) & 1))

    # ---- vector runs: expand each mask's set bits into columns
    for tv, nbit in ((V12, 12), (V8, 8)):
        m_v = typ == tv
        if not m_v.any():
            continue
        masks = pay[m_v]
        bits = (masks[:, None] >> np.arange(nbit)[None, :]) & 1
        wi, bi = np.nonzero(bits)
        if wi.size == 0:
            continue
        start = (bx[m_v] + (cumw[m_v] - width[m_v]) - base_cumw[m_v])
        out.append((start[wi] + bi, ys[m_v][wi], t[m_v][wi], bpol[m_v][wi]))

    if not out:
        e = np.empty(0, np.int64)
        return e, e, e, e
    xs = np.concatenate([o[0] for o in out]).astype(np.int64)
    yy = np.concatenate([o[1] for o in out]).astype(np.int64)
    tt = np.concatenate([o[2] for o in out]).astype(np.int64)
    pp = np.concatenate([o[3] for o in out]).astype(np.int64)
    # a chunk mixes singles and runs, so restore temporal order
    o = np.argsort(tt, kind="stable")
    return xs[o], yy[o], tt[o], pp[o]


def iter_events(path, chunk_words=1 << 24):
    """Stream (x, y, t, p) in decode order. t is microseconds, p in {0,1}."""
    _, off, w, h = read_header(path)
    st = _State()
    with open(path, "rb") as f:
        f.seek(off)
        while True:
            buf = f.read(chunk_words * 2)
            if not buf:
                break
            if len(buf) & 1:            # never split a word
                f.seek(-1, os.SEEK_CUR)
                buf = buf[:-1]
            words = np.frombuffer(buf, dtype="<u2")
            x, y, t, p = _decode_chunk(words, st)
            if x.size:
                yield x, y, t, p, w, h


def decode_to_windows(raw, outdir, window_us=100_000, max_windows=0,
                      min_events=200):
    """Split a recording into fixed windows, one .npz each: x, y, t, p.

    Matches what the wild pipeline and the renderers already read, so this data
    can go straight into build_wild_cache.py / adapt_wild.py without a shim.
    """
    os.makedirs(outdir, exist_ok=True)
    buf, n_win, n_ev, t0 = [], 0, 0, None
    W = H = None

    def flush(parts, idx):
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        t = np.concatenate([p[2] for p in parts])
        p_ = np.concatenate([p[3] for p in parts])
        if x.size < min_events:
            return 0
        # t is stored RELATIVE to the window start, as uint32 microseconds.
        # Every consumer normalises per window (t0 = t.min(), span = ptp), so
        # this is semantically identical to absolute time -- but at ~19M ev/s
        # int64 timestamps alone would be 8 bytes on a million events per
        # window. uint32 holds 4295 s, far past any 100 ms window.
        t = (t - t[0]).astype(np.uint32)
        np.savez_compressed(f"{outdir}/{idx:010d}.npz",
                            x=x.astype(np.uint16), y=y.astype(np.uint16),
                            t=t, p=p_.astype(np.uint8),
                            t0_us=np.int64(int(parts[0][2][0])))
        return x.size

    for x, y, t, p, W, H in iter_events(raw):
        if t0 is None:
            t0 = int(t[0])
        while t.size:
            edge = t0 + (n_win + 1) * window_us
            cut = np.searchsorted(t, edge)
            if cut:
                buf.append((x[:cut], y[:cut], t[:cut], p[:cut]))
            if cut == t.size:
                break
            n_ev += flush(buf, n_win) if buf else 0
            buf = []
            x, y, t, p = x[cut:], y[cut:], t[cut:], p[cut:]
            # Jump straight to the window holding the next event. Advancing one
            # window at a time across a time gap would spin through hundreds of
            # thousands of empty indices.
            n_win = max(n_win + 1, int((t[0] - t0) // window_us)) if t.size \
                else n_win + 1
            if max_windows and n_win >= max_windows:
                return n_win, n_ev, W, H
    if buf:
        n_ev += flush(buf, n_win)
        n_win += 1
    return n_win, n_ev, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/fs/nexus-projects/DVS_Actions/rain",
                    help="root holding <split>/<sequence>/proph_*/events.raw")
    ap.add_argument("--out", default=f"{C.DATA}/real/EVK4_wildrain",
                    help="writes <out>/<split>/merge_data/<sequence>/*.npz")
    ap.add_argument("--window-us", type=int, default=100_000,
                    help="100 ms, matching every other dataset in the campaign")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = whole file")
    ap.add_argument("--limit", type=int, default=0, help="0 = every sequence")
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; keep only matching paths. "
                         "Lets several jobs split the set without stepping on "
                         "each other, since each writes its own directory.")
    a = ap.parse_args()

    raws = sorted(glob.glob(f"{a.src}/*/*/proph_*/events.raw"))
    if a.only:
        keys = [k for k in a.only.split(",") if k]
        raws = [r for r in raws if any(k in r for k in keys)]
    if a.limit:
        raws = raws[:a.limit]
    print(f"{len(raws)} recordings under {a.src}\n", flush=True)
    manifest = []
    for i, raw in enumerate(raws, 1):
        parts = raw.split("/")
        split, seq = parts[-4], parts[-3]
        out = f"{a.out}/{split}/merge_data/{seq}"
        # Resume on a COMPLETION MARKER, not on "the directory has npz in it".
        # A killed decode leaves a partial directory, and skipping those would
        # silently ship truncated sequences -- which is exactly what happened
        # when the scratch quota killed the first run mid-flight.
        done = f"{out}/.complete"
        if os.path.exists(done):
            print(f"[{i}/{len(raws)}] {split}/{seq}: complete, skipping",
                  flush=True)
            continue
        if os.path.isdir(out):
            stale = glob.glob(f"{out}/*.npz")
            if stale:
                print(f"[{i}/{len(raws)}] {split}/{seq}: {len(stale)} partial "
                      f"windows from an interrupted run, redoing", flush=True)
                for f in stale:
                    os.remove(f)
        gb = os.path.getsize(raw) / 1e9
        print(f"[{i}/{len(raws)}] {split}/{seq}  ({gb:.2f} GB) ...",
              end=" ", flush=True)
        nw, ne, W, H = decode_to_windows(raw, out, a.window_us, a.max_windows)
        print(f"{nw} windows, {ne:,} events, {W}x{H}", end=" ", flush=True)

        # CROSS-CHECK against the recording's own host timestamps. A decoder
        # bug that corrupts the clock shows up here and nowhere else: the event
        # count stays plausible and the frames look fine, only the timeline is
        # wrong. This exact check is what a bad wrap rule would have tripped --
        # sequence_000001 decoded to 20,961 s against a 57 s recording.
        span = None
        pk = os.path.join(os.path.dirname(raw), "event_packet_times.npy")
        if os.path.exists(pk) and not a.max_windows:
            tp = np.load(pk)
            span = float(tp.max() - tp.min()) / 1e6
            got = nw * a.window_us / 1e6
            drift = abs(got - span) / max(span, 1e-9)
            print(f"| {got:.1f}s vs {span:.1f}s metadata", end=" ", flush=True)
            if drift > 0.05:
                print(f"\n  REJECTED {split}/{seq}: decoded span differs from "
                      f"metadata by {100*drift:.0f}% -- not marking complete",
                      flush=True)
                continue
        print("OK", flush=True)
        rec = dict(split=split, sequence=seq, windows=nw, events=int(ne),
                   width=W, height=H, window_us=a.window_us, source=raw,
                   metadata_span_s=span)
        with open(done, "w") as f:                 # marker: this one finished
            json.dump(rec, f, indent=1)
        manifest.append(rec)
        # several jobs share this file, so rebuild it from the per-sequence
        # markers rather than from this process's own list
        marks = sorted(glob.glob(f"{a.out}/*/merge_data/*/.complete"))
        json.dump([json.load(open(m)) for m in marks],
                  open(f"{a.out}/manifest.json", "w"), indent=1)
    print(f"\nmanifest: {a.out}/manifest.json")


if __name__ == "__main__":
    main()
