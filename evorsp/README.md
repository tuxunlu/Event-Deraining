# EvORSP-3T/E — event-camera deraining

Code, checkpoints and diagnostics for the EvORSP campaign, including the
head-to-head against PRE-Mamba (ICCV 2025).

## The headline results

**KITTI, event-level DA** (test {50,150} mm, PRE-Mamba's own metric):

| model | params | latency | event-DA |
|---|---|---|---|
| EvORSP-3T/E + per-event head v3 | 54,289 | ~25 ms† | **0.9576** |
| trunk, tfront16 + ctx4 + counts | 28,924 | **5.61 ms** | 0.9362‡ |
| PRE-Mamba (ICCV'25) | 264,632 | 306 / 409 ms | 0.9172 |
| EvORSP-3T, **old ON-only target** | 28,060 | 5.73 ms | **0.7052** |

† rate-dependent, 100K events, GPU-ported features. ‡ per-frame self-prior τ.

**Real EVK4** (PRE-Mamba's split / our scene-disjoint split): trunk 0.8066 /
0.8298 → +context 0.8192 / 0.8466 → +per-event head **0.8444 / 0.8686**, against
PRE-Mamba's 0.7708.

## The finding that matters most

The 0.212 event-level gap that made a 265K-parameter point-based SSM look
necessary was **a ground-truth convention, not an architecture**. The original
target marked a pixel as signal only when a clean **ON** event landed in it,
inherited from the ON-only eFFT pipeline. That target's own event-DA ceiling is
**0.6981** — and the model already scored 0.7052, i.e. it was saturated. Fixing
the label alone, with the network untouched, recovered the entire gap.

Corollary, measured in `eval/`: five architectures spanning 28K–265K parameters
all land within 0.007 of each other once given the same input and supervision.
Architecture is not the lever; input representation and supervision are.

## Running it on your own machine and your own data

Nothing here has a path baked in any more. Every location resolves from
environment variables, with defaults under the repository, so the only setup is
telling it where your data lives.

    pip install -r evorsp/requirements.txt
    export EVORSP_DATA=/where/your/datasets/live     # optional
    export EVORSP_WORK=/where/caches/should/go       # optional, needs ~50 GB
    python evorsp/config.py                          # prints what resolved where

Then run the model on your events:

    python evorsp/derain.py --events /path/to/your/data --out /path/to/output

`--events` takes a directory of `.npz` windows, or one `.npz` holding a longer
stream together with `--window N` to slice it. Each file needs four equal-length
arrays:

| array | meaning |
|---|---|
| `x`, `y` | pixel coordinates, integer |
| `t` | timestamp — **any unit, any origin**; each window is normalised to its own span |
| `p` | polarity; `1` is treated as ON, anything else as OFF |

Sensor size is inferred from the data; pass `--width/--height` if your recording
never lights the last row or column. Output is one `.npz` per window holding the
kept events plus `keep`, the boolean mask over the input, so nothing is lost.

If your dataset has per-event labels, score it directly:

    python evorsp/derain.py --events <dir> --labels <dir> --labels-rain-is 0

`--labels-rain-is` is mandatory with `--labels` and has no default on purpose:
the two conventions are both common, real EVK4 uses **0 = rain**, and guessing
wrong silently inverts every number rather than failing.

**Which checkpoint.** `--ckpt` defaults to `ctx_f4o16_c2` (KITTI-trained,
28,719 params, event-DA 0.9332). Use `rctx_ours_f4o1_c2` for real footage and
`spac_f4o16_c2` for SPAC-like synthetic. `python evorsp/derain.py --ckpt list`
prints the rest. The threshold defaults to the per-frame self-prior, which needs
no labels at deployment; `--tau-trained` uses the value selected on validation.

Verified faithful: `rctx_ours_f4o1_c2` on real EVK4 `scene4/rain_13` scores
**0.8661** through this path, matching the training run's own recorded number to
four decimals.

**Training on your new dataset** needs labels, since this is supervised. Copy
`data/build_real_e.py` (per-event labels) or `data/kitti_build_e.py` (a separate
clean stream, labels by exact set subtraction), point it at `C.CUSTOM_SRC`,
then train with `train/run_kitti_ctx.py --tfront 4 --ctx 2`. The pack format is
the contract between the two; `build_real_e.py`'s docstring specifies it.

## Layout

    model/       trunk (rsp_3d) + the earlier bodies behind a shared frontend
    features/    per-event features: structure tensor, ITI regularity,
                 recurrence/long-persistence, and their GPU ports
    data/        pack builders (event-accounting targets) and feature caches
    train/       trainers for KITTI / real EVK4 / SPAC
    eval/        oracles, diagnostics, probes, latency benches
    figures/     figure and video renderers
    checkpoints/ trained weights (<1 MB each) and result json

`../figs/README.md` documents every figure, the protocols, and the traps.

## Reproducing

Order matters: build packs → build feature caches → train → evaluate.

    python data/kitti_build_e.py            # event-accounting targets
    python data/build_headv2_cache.py       # structure-tensor cache
    python train/run_kitti_headv3.py --epochs 30
    python eval/mixed_cell_diag2.py phv3    # the occlusion metric

## Things that will bite you

- **`features/eigenpyramid.py` runs its entire falsifier sweep at import.**
  Importing it stalls for minutes. `features/fast_tensor.py` holds a
  parameterised copy, verified equal to 2e-12, and fixes its hardcoded
  `prev[4]/prev[16]/prev[64]` scale keys. Import that one.
- **Timestamp units differ by dataset.** KITTI and SPAC are NANOseconds, real
  EVK4 is MICROseconds. The slice/tau constants must be scaled by 1000
  accordingly; getting it wrong produces 1000x too many time slices and looks
  like a hang, not an error.
- **Real EVK4 labels: 1 = scene, 0 = rain**, established by cross-frame
  persistence (0.54 vs 0.20), NOT by the naive "label-1 fraction rises with
  rain_k" reading — `rain_k` is a recording index, not an intensity. PRE-Mamba's
  own config names its classes in the opposite order, so its printed SR/NR
  columns are swapped on real data (DA is unaffected, being symmetric).
- **Pixel DA does not predict event DA.** Every number on the original
  leaderboard is a pixel-DA number and is internally consistent as such, but
  must not be read as per-event deraining quality.
- **`model/bodies_e.py` and `train/run_kitti_fair.py` need a tree that vanished
  mid-campaign**, taking the 38 GB SPAC source with it. They now resolve it
  through `C.LEGACY_ED` (`$EVORSP_LEGACY`); point that at a copy if you have
  one. Kept as the record of what was run — nothing else depends on them.

## Not in git

- `figs/*.mp4` — 2.1 GB of renders, several files above GitHub's 100 MB limit,
  no git-lfs configured. Reproduce with `figures/render_*.py`.
- FourierMamba checkpoints (103–409 MB). Every other checkpoint is under 1 MB.
- Datasets and packs (`/fs/nexus-scratch/tuxunlu/*_t16e`, `*_headv2`, ...).
