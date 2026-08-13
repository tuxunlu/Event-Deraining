"""Re-encode PRE-Mamba's per-event label dumps from int64 to uint8.

PRE-Mamba's exp/event_rain/*/result holds one .npy per test window: a flat
array of per-event keep/drop labels. The values are binary -- unique == [0, 1]
-- but they are stored as int64, so every one-bit decision costs eight bytes.
2,897 files, 44.3 GB, for something that fits in 5.5 GB.

This is not a modelling choice, just NumPy's default integer dtype surviving
all the way to disk. Re-encoding is lossless and invisible to every consumer:
mixed_cell_diag2.py reads these as `pred[:len(x)] == 0`, which behaves
identically on uint8.

SAFETY. Label dumps are expensive to regenerate -- they are the output of
running someone else's model over the whole test set -- so this never trusts
the conversion:

  1. refuses any file whose values do not fit uint8, and leaves it untouched
  2. writes a temporary file beside the original
  3. reloads the temporary and asserts array_equal against the original
  4. only then atomically replaces, via os.replace

An interrupted run therefore leaves a mix of converted and unconverted files,
never a corrupt one, and re-running skips whatever is already uint8.
"""

import argparse
import glob
import os

import numpy as np


def convert(root, dry_run=False, limit=0):
    files = sorted(glob.glob(f"{root}/**/*.npy", recursive=True))
    if limit:
        files = files[:limit]
    before = after = 0
    done = skipped = refused = 0
    bad = []
    for i, f in enumerate(files, 1):
        sz = os.path.getsize(f)
        before += sz
        try:
            a = np.load(f)
        except Exception as e:
            refused += 1
            bad.append((f, f"unreadable: {type(e).__name__}"))
            after += sz
            continue

        if a.dtype == np.uint8:
            skipped += 1
            after += sz
            continue

        # only touch things that genuinely fit -- anything else is not what
        # this script was written for and is left exactly as found
        if a.size and (a.min() < 0 or a.max() > 255):
            refused += 1
            bad.append((f, f"values {a.min()}..{a.max()} do not fit uint8"))
            after += sz
            continue

        small = a.astype(np.uint8)
        if dry_run:
            after += small.nbytes
            done += 1
            continue

        tmp = f + ".u8tmp.npy"
        np.save(tmp, small)
        check = np.load(tmp)
        # verify against the ORIGINAL array, not against `small`
        if check.shape != a.shape or not np.array_equal(check.astype(a.dtype), a):
            os.remove(tmp)
            refused += 1
            bad.append((f, "round-trip mismatch"))
            after += sz
            continue
        os.replace(tmp, f)                      # atomic
        after += os.path.getsize(f)
        done += 1

        if i % 250 == 0:
            print(f"  {i}/{len(files)}  {before/1e9:.1f} GB -> "
                  f"{after/1e9:.1f} GB", flush=True)

    return dict(files=len(files), converted=done, already=skipped,
                refused=refused, before=before, after=after, bad=bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/fs/nexus-scratch/tuxunlu/git/PRE-Mamba/exp/event_rain")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print(f"{'DRY RUN: ' if a.dry_run else ''}scanning {a.root}", flush=True)
    r = convert(a.root, a.dry_run, a.limit)
    print(f"\n  files          {r['files']:,}")
    print(f"  converted      {r['converted']:,}")
    print(f"  already uint8  {r['already']:,}")
    print(f"  refused        {r['refused']:,}")
    print(f"  before         {r['before']/1e9:.2f} GB")
    print(f"  after          {r['after']/1e9:.2f} GB")
    print(f"  freed          {(r['before']-r['after'])/1e9:.2f} GB")
    for f, why in r["bad"][:20]:
        print(f"    REFUSED {why}: {f}")


if __name__ == "__main__":
    main()
