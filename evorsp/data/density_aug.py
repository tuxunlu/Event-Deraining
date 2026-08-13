"""Train-only density augmentation, shared by the KITTI and real trainers.

Extracted verbatim from run_kitti_ctx.py once a second call site appeared. The
logic is subtle enough -- label-preserving superposition, occupancy-aware
thinning -- that two copies would diverge.

On KITTI this family was measured and came back null across all four arms
(mix -0.0018, drop -0.0004, mix+drop -0.0023, hflip -0.0017 on 150 mm, against
a 0.0015 sd control). That does not settle the question on real data: KITTI's
hard split is SYNTHETIC falling rain at an intensity never trained on, whereas
the real rig varies intensity through nozzle pressure and holds out a whole
scene, so the shift there is scene appearance as much as density.
"""

import numpy as np


def augment(rng, on, off, bg, rn, aug, mix_p=0.5, drop_q=0.3,
            draw_partner=None):
    """Augment raw T_BUILD-resolution planes and counts.

    `aug` is a set of {"mix", "drop", "hflip"}. `draw_partner()` must return
    (index, on2, off2, bg2, rn2) for the mix arm, or None to disable it.

    Returns (on, off, bg, rn, tag) where tag records what was done so the
    caller can treat the CONTEXT planes identically -- a superposed frame must
    get superposed history, or the context contradicts the input.
    """
    tag = None
    if "mix" in aug and draw_partner is not None and rng.random() < mix_p:
        # SUPERPOSITION -> heavier rain. Counts ADD, so the target bg>rn and
        # the lit mask recompute exactly: label-preserving by construction
        # rather than by assumption.
        #
        # Stated approximation: OR-ing a binary occupancy grid is not adding
        # events -- two events in one cell still set one bit. That is what
        # physically happens to an occupancy representation under overlap, but
        # planes and counts are not perfectly consistent afterwards.
        drawn = draw_partner()
        if drawn is not None:
            j, on2, off2, bg2, rn2 = drawn
            on, off = on | on2, off | off2
            bg, rn = bg + bg2, rn + rn2
            tag = j

    if "drop" in aug:
        # THINNING -> lighter rain. Counts thin binomially; an occupancy bit
        # survives with probability 1 - q^n, since a cell holding n events only
        # goes dark if all n are dropped. Using the total count for both
        # polarities is an approximation: the packs store per-CLASS counts
        # (background/rain), not per-polarity ones.
        n = bg + rn
        keep = rng.random(n.shape) < (1.0 - drop_q ** np.maximum(n, 1))
        keep &= n > 0
        on, off = on & keep, off & keep
        bg = rng.binomial(bg.astype(np.int64), 1.0 - drop_q).astype(np.float32)
        rn = rng.binomial(rn.astype(np.int64), 1.0 - drop_q).astype(np.float32)

    if "hflip" in aug and rng.random() < 0.5:
        on, off = on[:, :, ::-1], off[:, :, ::-1]
        bg, rn = bg[:, :, ::-1], rn[:, :, ::-1]
        tag = ("hflip", tag)
    return on, off, bg, rn, tag


def augment_context(pon, poff, tag, load_prev):
    """Apply to context planes whatever `augment` did to their window.

    `load_prev(j)` returns the same-offset context planes of partner window j.
    """
    if tag is None:
        return pon, poff
    flip = isinstance(tag, tuple)
    j = tag[1] if flip else tag
    if j is not None:
        p2on, p2off = load_prev(j)
        pon, poff = pon | p2on, poff | p2off
    if flip:
        pon, poff = pon[:, :, ::-1], poff[:, :, ::-1]
    return pon, poff
