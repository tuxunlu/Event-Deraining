"""A trunk whose retention of persistent content is PROVABLE, not asserted.

The adversarial test (adversarial_dc.py) refuted the claim I had been making.
The frontend's protected residual band is genuinely immovable -- attacked
directly it shifted 1.0x -- but the network as a whole still suppressed static
input by hundreds of times, and removing the protected frontend entirely did not
reduce that. The protection was real and completely undone downstream.

Confirmed run, see checkpoints/adversarial_dc_result.txt:

    band decomposition alone            1.0x     immovable, as claimed
    full shipped network              529.7x     protection undone
    same network, no protected band   363.3x     the band was not what helped
    plain CNN, no structure at all   1332.2x     reference point
    GUARDED (this module)               3.9x     min logit 2.000 == alpha - C

The full-network figure wanders ~400-700x across seeds -- it depends on how far
one short adversarial run happens to get. The ordering does not move, and the
guarded row is exact rather than stochastic: the floor is algebra, not a fit.

WHY A BYPASS ALONE DOES NOT FIX IT. Routing the residual to the output as
`z + res` gives no guarantee: z is unbounded, so the learnable branch can emit a
large negative value and cancel the bypass exactly. Any additive path can be
cancelled by an unconstrained partner.

WHAT ACTUALLY GIVES A GUARANTEE. Bound the learnable branch:

    logit = alpha * res + C * tanh(z / C)          alpha > 0, C fixed

The second term lies in (-C, +C) by construction, so

    logit  >  alpha * res - C

and therefore ANY cell whose persistent-band response exceeds C / alpha is
retained -- whatever the weights, whatever the training objective, whatever an
adversary does. The bound is a property of the algebra, not of the optimiser.

The cost is real and worth stating: the learnable correction is now saturating,
so the network can no longer express arbitrarily confident REMOVAL. It can still
express arbitrarily confident retention, which is the asymmetry deraining wants
-- keeping a rain event is a blemish, deleting the scene is a failure.

CALIBRATION MATTERS AND I GOT IT WRONG FIRST. The residual band is effectively
binary on persistent content -- measured 0.0 below the 90th percentile and
exactly 1.0 above it. With alpha=1, C=4 the floor sits at res > 4, which is
unreachable, so the guarantee never bound and the guarded model suppressed
static input MORE than the plain one (4540x). The condition is alpha > C:
at alpha=4, C=2 a fully lit persistent cell has logit > 2, i.e. keep-probability
> 0.88, no matter what the weights are, while non-persistent cells stay fully
learnable across +-2 logits.

alpha and C are fixed, not learned. A learnable alpha could anneal to zero and
the guarantee would evaporate, which is exactly the mistake the bypass-only
design makes.
"""
import torch
import torch.nn as nn

from rsp_3d import ORSPNet3D


class ORSPNet3DGuard(ORSPNet3D):
    """ORSPNet3D with a provable retention floor for persistent content."""

    def __init__(self, *a, alpha=4.0, bound=2.0, **kw):
        super().__init__(*a, **kw)
        assert self.use_temporal, "the guarantee needs the temporal frontend"
        self.alpha = float(alpha)          # fixed on purpose -- see module docstring
        self.bound = float(bound)

    def forward(self, x, x_off=None, x_cnt=None, x_extra=None):
        p = self.front(x)                                   # [B, 1+n_t+1, H, W]
        res = p[:, -1:]                                     # protected band, ON
        if self.use_off:
            p_off = self.front(x_off)
            # MAX, not mean: a cell is persistent if EITHER polarity
            # persists. Averaging halves the floor whenever one
            # polarity is quiet, which silently voids the guarantee.
            res = torch.maximum(res, p_off[:, -1:])
            p = torch.cat([p, p_off], 1)
        if self.use_counts and x_cnt is not None:
            p = torch.cat([p, x_cnt], 1)
        if getattr(self, "n_extra", 0) and x_extra is not None:
            p = torch.cat([p, x_extra], 1)

        z = self.out_proj(self._body(self.in_proj(p)))       # unbounded branch
        C = self.bound
        return self.alpha * res + C * torch.tanh(z / C)

    def _body(self, h):
        for blk in self.blocks:
            h = blk(h)
        return h

    def retention_floor(self):
        """Persistent-band response above which retention is guaranteed."""
        return self.bound / self.alpha
