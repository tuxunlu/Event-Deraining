"""RQ-B: is the "cannot delete the static scene" guarantee real, or overstated?

I have been claiming EvORSP "provably cannot delete static scene structure",
because DC is hard-protected -- temporally by A = A * (w > 0), spatially by
rho -> 0 killing every atom. This test attacks that claim directly: TRAIN the
model to do exactly the forbidden thing and see whether it can.

ADVERSARIAL OBJECTIVE. Input is a purely STATIC pattern -- the same spatial
image repeated in every time slice, so it is pure temporal DC. Target is zero.
A model that can drive its output to zero can delete the static world.

    fitted   -> the model CAN delete static structure; my claim is wrong
    floored  -> it cannot, and the guarantee is real

PRE-REGISTERED PREDICTION, written before running. I expect the FULL NETWORK to
largely fit this. The protection lives in the FRONTEND, which routes temporal DC
into a protected residual channel -- but in_proj, the blocks and out_proj that
follow are unconstrained and can suppress that channel afterwards. If so, the
honest claim is narrower than the one I have been making: the DECOMPOSITION
cannot attenuate DC, the NETWORK as a whole still can.

Two levels are therefore measured separately:

  NETWORK  can the whole trunk be trained to output zero on static input?
  FRONTEND can the TemporalFrontend itself be made to lose the static
           component? This is where the guarantee actually lives.

Arms: ours (protected frontend), the same trunk with use_temporal=False, and a
plain CNN with no frequency structure at all as the unconstrained reference.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D, TemporalFrontend
from rsp_guard3d import ORSPNet3DGuard

DEV = "cuda"
R, T = 256, 4
STEPS = 400
SEED = 0


def static_batch(b=2, density=0.06, gen=None):
    """A spatial pattern repeated across every time slice: pure temporal DC."""
    pat = (torch.rand(b, 1, R, R, generator=gen) < density).float()
    return pat.expand(b, T, R, R).contiguous().to(DEV)


class PlainCNN(nn.Module):
    """No frequency decomposition, no protected band. The unconstrained control."""

    def __init__(self, cin=T * 2, dim=32, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, out, 1))

    def forward(self, x, x_off=None):
        return self.net(torch.cat([x, x_off], 1))


def attack(model, tag, steps=STEPS, lr=3e-3):
    """Train the model to output zero on static input. Lower loss = more able.

    Also reports the response on the cells the guarantee is ABOUT -- those whose
    persistent band is lit. A global mean can fall a long way while the
    protected population is untouched, so the mean alone would misjudge this.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    first = last = None
    for i in range(steps):
        on = static_batch(gen=g)
        off = torch.zeros_like(on)
        out = model(on, x_off=off)
        loss = out.pow(2).mean()          # drive the response to zero
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    with torch.no_grad():
        on = static_batch(gen=torch.Generator().manual_seed(SEED + 1))
        out = model(on, x_off=torch.zeros_like(on))
        lit = on[:, :1] > 0.5
        prot = out[:, :1][lit]
    return first, last, float(prot.mean()), float(prot.min())


def attack_frontend(steps=STEPS, lr=3e-3):
    """Same attack against the TemporalFrontend alone.

    Scored on the RESIDUAL channel only -- the band the design claims is
    protected. Its 6 parameters are the entire attack surface.
    """
    fr = TemporalFrontend(n_t=3, T=T).to(DEV)
    opt = torch.optim.AdamW(fr.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEED)
    first = last = None
    for i in range(steps):
        on = static_batch(gen=g)
        p = fr(on)                        # [B, 1+n_t+1, R, R]
        res = p[:, -1]                    # the protected residual band
        loss = res.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def main():
    torch.manual_seed(SEED)
    kw = dict(T=T, dilations=(1, 8, 32, 64), num_blocks=3, use_off=True,
              out_chans=1)
    arms = [
        ("ours (protected frontend)", ORSPNet3D(**kw).to(DEV)),
        ("same trunk, no frontend", ORSPNet3D(**{**kw, "use_temporal": False}).to(DEV)),
        ("plain CNN (unconstrained)", PlainCNN().to(DEV)),
        ("GUARDED (bounded correction)", ORSPNet3DGuard(**kw).to(DEV)),
    ]
    print(f"\nADVERSARIAL: train each model to output ZERO on static input")
    print(f"  {STEPS} steps, {R}x{R}, pattern identical in all {T} slices\n")
    print(f"  {'arm':30s} {'suppressed':>11s} {'logit@lit':>11s} "
          f"{'min':>9s} {'keep-prob':>10s}")
    print("  " + "-" * 74)
    import math
    for tag, m in arms:
        a, b, pm, pmin = attack(m, tag)
        kp = 1 / (1 + math.exp(-pm))
        print(f"  {tag:30s} {a / max(b, 1e-30):10.1f}x {pm:11.3f} {pmin:9.3f} "
              f"{kp:10.3f}")

    a, b = attack_frontend()
    print(f"\n  {'FRONTEND residual band only':30s} {a:11.3e} {b:11.3e} "
          f"{a / max(b, 1e-30):11.1f}x")
    print("\n  'suppressed' = how many times smaller the response became.")
    print("  Large = the model learned to delete static content.")
    print("  ~1x  = it could not, whatever the optimiser tried.")


if __name__ == "__main__":
    main()
