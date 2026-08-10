"""Frame-adaptive atom orientation: the bank re-steers itself to each storm.

MOTIVATION, measured. Comparing the shipped checkpoint against its
initialisation, the atom orientations barely moved: 0/45/90/135 ->
6/3/37/37/90/101/143/140. The gradient reaching them is weak and second-order,
so initialisation is doing the work. Meanwhile theta is GLOBAL and FROZEN at
inference, while rain direction changes with wind -- scene1 is near-vertical
columns, scene2 is diagonal.

THE IDEA. The bank is *steerable* by construction, so rotating it is exact:
add a scalar to theta. And the dominant orientation of a frame has a closed
form -- the energy-weighted circular mean of the DOUBLED angle:

    theta_dom = 1/2 * atan2( sum e*sin2phi , sum e*cos2phi )

cos2phi and sin2phi are already columns of the bank's own basis grid, so this
costs one weighted sum. No hypernetwork, no second transform, no extra pass.

WHY IT IS NOT CIRCULAR. The estimate is computed from the raw spectrum, not by
projecting onto the masks, so it does not depend on the masks it steers.

    dtheta = a * theta_dom + b        a, b learned scalars per block

a and b are ZERO-INIT, so at step 0 this model is bit-identical to the frozen
bank and can only depart from it if that helps -- the same discipline the gate
head and lift bias already follow.

Two parameters per block. Six for the whole trunk.
"""
import math

import torch
import torch.nn as nn

from rsp_3d import ORSPNet3D
from rsp_model_v2 import RainSubspaceBlock


class SteerBlock(RainSubspaceBlock):
    """RainSubspaceBlock whose atom bank rotates with the frame."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.steer_a = nn.Parameter(torch.zeros(1))
        self.steer_b = nn.Parameter(torch.zeros(1))

    def _masks(self, Hf, Wf, device, dth):
        """Bank masks with every atom rotated by dth. [B, r, Hf, Wf]."""
        bank = self.bank
        basis = bank.grid(Hf, Wf, device, torch.float32)          # [5, L]
        kappa = bank.log_kappa.exp()                              # [r]
        inv2s2 = 0.5 / bank.log_sigma.exp().clamp_min(1e-3).pow(2)
        th2 = 2.0 * (bank.theta[None, :] + dth)                   # [B, r]
        wc = kappa[None, :] * torch.cos(th2)
        ws = kappa[None, :] * torch.sin(th2)
        B = th2.shape[0]
        c3 = (-inv2s2)[None, :].expand(B, -1)
        c4 = (2 * bank.mu * inv2s2)[None, :].expand(B, -1)
        c5 = (-kappa - bank.mu.pow(2) * inv2s2)[None, :].expand(B, -1)
        co = torch.stack([wc, ws, c3, c4, c5], dim=2)             # [B, r, 5]
        A = torch.exp(co @ basis).view(B, bank.r, Hf, Wf)
        return A / (1.0 + A.sum(1, keepdim=True))

    @staticmethod
    def _theta_dom(P, basis):
        """Energy-weighted circular mean of the doubled angle. No parameters.

        Uses the raw spectrum, so it is independent of the masks it will steer.
        DC and its immediate neighbourhood are excluded: orientation is
        meaningless there and the energy is huge.
        """
        e = P.abs().pow(2).sum(1).reshape(P.shape[0], -1)          # [B, L]
        c2, s2, lr = basis[0], basis[1], basis[3]
        w = (lr > math.log(0.02)).to(e.dtype)
        C = (e * (c2 * w)).sum(-1)
        S = (e * (s2 * w)).sum(-1)
        return 0.5 * torch.atan2(S, C)                             # [B]

    def forward(self, x):
        B, C_, H, W = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        P = torch.fft.rfft2(self.to_probe(xn).float(), norm="ortho")
        Hf, Wf = P.shape[-2:]
        basis = self.bank.grid(Hf, Wf, x.device, torch.float32)

        th_dom = self._theta_dom(P, basis)                          # [B]
        dth = (self.steer_a * th_dom + self.steer_b).unsqueeze(1)   # [B,1]
        M = self._masks(Hf, Wf, x.device, dth)                      # [B,r,Hf,Wf]

        Sb = P.unsqueeze(1) * M.unsqueeze(2)                        # [B,r,p,Hf,Wf]
        Y = torch.fft.irfft2(Sb.reshape(B, self.r * self.probe, Hf, Wf),
                             s=(H, W), norm="ortho")
        Y = Y.view(B, self.r, self.probe, H, W)

        e = Y.pow(2).mean(2)
        tot = e.sum(1, keepdim=True) + 1e-8
        q = e / tot
        ent = -(q * (q + 1e-8).log()).sum(1, keepdim=True)
        feat = torch.cat([torch.log(e + 1e-6), q, ent, self.ctx_proj(xn)], 1)

        # M is per-sample now, so no broadcast over a shared bank
        Eg = (P.abs().pow(2).sum(1).unsqueeze(1) * M).sum((-1, -2))
        Eg = Eg / (Eg.sum(1, keepdim=True) + 1e-8)
        gamma, beta = self.film(Eg).chunk(2, dim=1)
        film = (gamma.view(B, self.r, 1, 1), beta.view(B, self.r, 1, 1))

        g = self.gate(feat, film)
        rain = (g.unsqueeze(2) * Y * self.band_scale).sum(1)
        x = x - self.lift(rain)
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


class ORSPNet3DSteer(ORSPNet3D):
    """ORSPNet3D with steerable-per-frame blocks. Same weights plus 2 per block."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        n_theta = kw.get("n_theta", 4)
        n_rad = kw.get("n_rad", 2)
        probe = kw.get("probe", 4)
        blk_kw = {k: v for k, v in kw.items()
                  if k in ("gate_hidden", "ffn_expand", "se_ratio", "dilations")}
        dim = self.blocks[0].dim
        self.blocks = nn.ModuleList([
            SteerBlock(dim, n_theta, n_rad, probe, **blk_kw)
            for _ in range(len(self.blocks))])

    def load_frozen(self, sd):
        """Load a frozen-bank checkpoint; only steer_a / steer_b are new."""
        missing, unexpected = self.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.endswith(("steer_a", "steer_b"))]
        assert not bad, f"unexpected missing keys: {bad[:5]}"
        assert not unexpected, f"unexpected extra keys: {list(unexpected)[:5]}"
        return len(missing)
