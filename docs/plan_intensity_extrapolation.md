# Plan: close the intensity-extrapolation gap

## The diagnosis, measured rather than assumed

I previously said the trunk looked under-regularized. The data says otherwise.

**The val-test gap is constant across a 1200x capacity range.**

| model | params | val | test | gap |
|---|---|---|---|---|
| ORSP 2-block | 19,622 | 0.9481 | 0.9252 | 0.0229 |
| ORSP +ctx2 | 28,719 | 0.9503 | 0.9291 | 0.0212 |
| ORSP fair | 37,652 | 0.9450 | 0.9215 | 0.0235 |
| StreakNet +ctx2 | 53,077 | 0.9521 | 0.9295 | 0.0226 |
| DFFN +ctx2 | 73,108 | 0.9488 | 0.9258 | 0.0230 |
| FourierMamba +ctx2 | 23,603,572 | 0.9580 | 0.9379 | **0.0201** |

22 models, 19,622 to 23,600,048 parameters: gap mean **0.0225**, range 0.0155-0.0267.
SPAC shows the same ~0.022. The **largest** model has the **smallest** gap.

Overfitting-by-capacity would make the gap grow with parameters. It does not move
at all. That rules the hypothesis out.

**Where the gap actually lives.** KITTI selects tau on val {20, 80} mm and reports
on test {50, 150} mm. Split the test number by intensity:

| model | 50 mm (inside val range) | 150 mm (outside) | delta |
|---|---|---|---|
| ORSP +ctx2 | 0.9487 | 0.9096 | -0.0391 |
| ORSP c2r4 | 0.9507 | 0.9096 | -0.0410 |
| StreakNet +ctx2 | 0.9502 | 0.9088 | -0.0414 |
| DFFN +ctx2 | 0.9470 | 0.9046 | -0.0424 |
| FourierMamba +ctx2 | 0.9561 | 0.9196 | **-0.0364** |

50 mm sits inside the trained range and scores at val level. 150 mm is heavier than
anything the model has ever seen and drops ~0.041. **The entire val-test gap is
extrapolation to an unseen rain intensity.** It is a property of the benchmark
split, not of our architecture -- which is why nine consecutive architecture
experiments produced nothing.

## Hypothesis (pre-registered)

The 150 mm deficit is caused by never training on rain heavier than 80 mm.
Interventions that either **expose** the model to higher event density or make it
**invariant** to density will recover part of the -0.041.

**Falsifier:** if no arm moves 150 mm event-DA by more than the measured seed
range, the hypothesis is wrong and we report the diagnosis as the result.

## Metrics

- **Primary: 150 mm event-DA.** Control 0.9096. Better powered than overall test
  because the effect is concentrated there.
- **Secondary:** overall test event-DA, control 0.9291.
- **Guard:** 50 mm must not regress below 0.9487 - seed range. An arm that trades
  easy rain for hard rain has not solved anything.
- **Seed range on the 150 mm metric is UNMEASURED.** The +-0.0026 quoted
  throughout the campaign is for overall test. Phase 1 measures it before any
  arm is judged.

## Phases

### Phase 1 - establish the ruler, and the free win (no new training)
1. **Two extra seeds of the control** to get the seed range of the 150 mm metric
   specifically. Nothing below is interpretable without this. ~3 h.
2. **Rate-adaptive threshold.** Self-prior tau (per-frame mean over lit cells)
   already buys +0.004 overall (0.9291 -> 0.9332). Measure it per intensity, and
   fit tau as a function of window event count on val. Zero training, zero
   inference cost. Plausibly recovers a real slice of the 150 mm deficit, since
   the optimal operating point almost certainly moves with density.

### Phase 2 - density augmentation (4 arms, ~85 min each)
Implemented as `--aug` in `run_kitti_ctx.py`. All are training-time only, so
**inference cost is unchanged**.

- **2a `mix`** -- event superposition. With probability p, OR this window's ON/OFF
  planes with those of a randomly chosen other window, and ADD their bg/rn counts.
  Synthesises heavier rain from lighter. The target `bg > rn` and the lit mask
  recompute correctly from the summed counts, so the augmentation is
  label-preserving by construction. *Caveat to note in the writeup: OR-ing a
  binary occupancy grid is not the same as adding events, since two events in one
  cell still set one bit -- this is exactly what physically happens to an
  occupancy representation under overlap, but the counts and the planes are
  therefore not perfectly consistent.*
- **2b `drop`** -- event dropout, p=0.3. Simulates *lighter* rain. Tests whether
  what matters is density-invariance in general or high-density exposure
  specifically. A useful discriminator: if `drop` helps 150 mm as much as `mix`,
  the mechanism is invariance, not exposure.
- **2c `mix+drop`**
- **2d `hflip`** -- cheap control for "is any augmentation at all enough", and
  safe for rain physics in a way vertical flip is not.

### Phase 3 - density normalisation in the input (conditional on Phase 2)
If augmentation works, the same end may be reachable by construction: normalise
the trunk's inputs by window event density so intensity shift cannot reach the
weights. The campaign already owns degree-zero homogeneous features
(`eigenpyramid.tensor_cols`) built on exactly this principle.

### Phase 4 - confirm off KITTI
Best recipe on real EVK4, where intensity also varies and where context was worth
3x more. Real is the dataset whose ranking we actually lead.

### Track B (independent) - distil FourierMamba
Now that a stronger teacher is measured (0.9379 +ctx vs our 0.9297, and -0.0364 vs
our -0.0391 at 150 mm), distillation moves accuracy into our existing weights at
zero inference cost. Runs independently of Phases 1-4.

## What NOT to do

Tested and null, do not revisit: more blocks (costs latency *and* -0.0039), wider
dilation ladders (3 arms all below control), more context windows, higher context
temporal resolution (4 arms, all within 0.0010), auxiliary task, frame-adaptive
orientation, SSM in the gate slot (worse, 3x larger, 9x slower).

## Cost

Phase 1 ~3 h. Phase 2 four arms ~6 h. Phases 3-4 conditional. All arms are
~85 min on one GPU and cost nothing at inference; the only inference-time change
anywhere in this plan is the Phase 1 threshold rule, which is a scalar.
