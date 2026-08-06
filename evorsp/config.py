"""Path configuration. Everything resolves from environment variables.

The campaign ran on a cluster with absolute paths baked into ~73 scripts. This
module replaces them so the code runs anywhere. Nothing here needs editing --
set the environment variables that apply to your machine, or accept the
defaults, which put everything under the repository.

    EVORSP_DATA   where raw datasets live      (default <repo>/data)
    EVORSP_WORK   where packs and caches go    (default <repo>/work)
    EVORSP_CKPT   trained weights              (default evorsp/checkpoints)

Per-dataset overrides, if your layout differs from the defaults:

    EVORSP_KITTI  EVORSP_REAL  EVORSP_SPAC  EVORSP_CUSTOM

Example:

    export EVORSP_DATA=/mnt/d/datasets
    export EVORSP_WORK=/mnt/d/evorsp_work
    export EVORSP_CUSTOM=/mnt/d/datasets/my_new_rain_set

Import from anywhere in the package:

    from evorsp.config import KITTI_SRC, WORK, CKPT
"""
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent


def _p(env, default):
    return Path(os.environ.get(env, str(default))).expanduser()


# ---- top level -------------------------------------------------------------
DATA = _p("EVORSP_DATA", REPO / "data")
WORK = _p("EVORSP_WORK", REPO / "work")
CKPT = _p("EVORSP_CKPT", PKG / "checkpoints")
FIGS = _p("EVORSP_FIGS", REPO / "figs")

# ---- raw dataset roots -----------------------------------------------------
# KITTI / SPAC ship a CLEAN stream, so labels come from exact (x,y,t) set
# subtraction. Real EVK4 ships PER-EVENT labels instead (1 = scene, 0 = rain).
KITTI_SRC = _p("EVORSP_KITTI", DATA / "synthetic_KITTI" / "synthetic")
REAL_SRC = _p("EVORSP_REAL", DATA / "real" / "EVK4_artifical")
REAL_WILD_SRC = _p("EVORSP_REAL_WILD", DATA / "real" / "EVK4_realworld")
SPAC_SRC = _p("EVORSP_SPAC", DATA / "synthetic_SPAC")
CUSTOM_SRC = _p("EVORSP_CUSTOM", DATA / "custom")

# ---- derived working directories -------------------------------------------
KITTI_PACK = WORK / "kitti_t16e"        # event-accounting packs (bg/rn counts)
REAL_PACK = WORK / "real_t16e"
SPAC_PACK = WORK / "spac_t16e"
CUSTOM_PACK = WORK / "custom_t16e"

KITTI_HEAD = WORK / "kitti_headv2"      # structure-tensor feature cache
REAL_HEAD = WORK / "real_headv2"
SPAC_HEAD = WORK / "spac_headv2"
CUSTOM_HEAD = WORK / "custom_headv2"

REAL_ITI = WORK / "real_iti"            # inter-arrival-time regularity
REAL_RECUR = WORK / "real_recur"        # recurrence / long persistence
CUSTOM_ITI = WORK / "custom_iti"
CUSTOM_RECUR = WORK / "custom_recur"

# PRE-Mamba predictions, if you cloned and ran it for comparison. The "SYTHETIC"
# spelling is theirs; do not fix it, it is the directory name their code writes.
PREMAMBA = _p("EVORSP_PREMAMBA", REPO.parent / "PRE-Mamba")
PM_SYNTH = PREMAMBA / "exp" / "event_rain" / "SYTHETIC" / "result"
PM_REAL = PREMAMBA / "exp" / "event_rain" / "REAL_OURS" / "result"

# Dead on the campaign machine: this tree disappeared mid-campaign, taking the
# 38 GB SPAC source with it. Scripts referencing it are kept as the record of
# what was run and will not execute until you point this somewhere real.
LEGACY_ED = _p("EVORSP_LEGACY", DATA / "Event-Deraining")

# ---- sensor geometry per dataset -------------------------------------------
# (width, height, timestamp unit). Getting the unit wrong is the single most
# expensive mistake in this codebase: the time-slice constants are scaled by it,
# and a 1000x error produces 100k time slices instead of 100 -- which presents
# as a hang, not as an error.
GEOM = {
    "kitti": (460, 352, "ns"),
    "real": (1280, 720, "us"),
    "spac": (640, 480, "ns"),
}


def geom(name):
    """(width, height, slice_len, tau) for a dataset key, in native units."""
    w, h, unit = GEOM[name.lower()]
    mul = 1_000 if unit == "ns" else 1        # 1 ms slice, 5 ms tau
    return w, h, 1_000 * mul, 5_000 * mul


SUBDIRS = ("model", "features", "data", "train", "eval", "figures")


def bootstrap():
    """Make the campaign's flat imports resolve from any subdirectory.

    The scripts were written side by side in one directory and import each
    other flatly (`from iti_feats import iti_gpu`). They now live in
    model/ features/ data/ train/ eval/ figures/, so every one of those has to
    be on sys.path for those imports to keep working. Replaces the
    sys.path.insert(<cluster tmp dir>) lines. Safe to call repeatedly.
    """
    for d in (REPO, PKG, *(PKG / s for s in SUBDIRS)):
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure(*paths):
    for q in paths:
        Path(q).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("EvORSP path configuration\n")
    for k in ("DATA", "WORK", "CKPT", "FIGS", "KITTI_SRC", "REAL_SRC",
              "SPAC_SRC", "CUSTOM_SRC", "KITTI_PACK", "REAL_PACK",
              "CUSTOM_PACK", "CUSTOM_HEAD"):
        v = globals()[k]
        mark = "ok " if Path(v).exists() else "   "
        print(f"  {mark}{k:14s} {v}")
    print("\n  'ok' = exists on this machine. Datasets and work dirs are")
    print("  created on demand by the builders; raw dataset roots are not.")
