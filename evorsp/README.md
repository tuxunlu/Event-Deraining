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
- **`model/bodies_e.py` and `train/run_kitti_fair.py` import from
  `/fs/nexus-scratch/tuxunlu/git/Event-Deraining`, which no longer exists on
  disk** (it disappeared mid-campaign, taking the 38 GB SPAC source with it).
  Those two scripts are kept as the record of what was run; they will not
  execute until that path is restored.

## Not in git

- `figs/*.mp4` — 2.1 GB of renders, several files above GitHub's 100 MB limit,
  no git-lfs configured. Reproduce with `figures/render_*.py`.
- FourierMamba checkpoints (103–409 MB). Every other checkpoint is under 1 MB.
- Datasets and packs (`/fs/nexus-scratch/tuxunlu/*_t16e`, `*_headv2`, ...).
