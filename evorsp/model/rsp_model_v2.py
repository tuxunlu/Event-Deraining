"""ORSP-Net -- Oblique Rain-Subspace Projection.

Deraining as SEPARATION, not per-bin filtering.

Per block:
  1. a bank of r frequency-domain ATOMS, analytically parameterised in Barnum's
     (orientation, radial-envelope) family, partitions the rFFT grid into r rain
     hypotheses plus a protected residual band (partition of unity);
  2. a rank-p channel read-out is decomposed into those bands with ONE irfft2;
  3. a per-pixel SIGNED gain g_j(x) in (-s, s) obliquely projects each band out;
  4. the removed component is lifted back to full channel width.

Nothing ever computes a wrapped phase.  Nothing mixes neighbouring bins.  The
atom bank is an explicit analytic function of absolute polar frequency.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class SqueezeExcite(nn.Module):
    def __init__(self, c, ratio=0.25):
        super().__init__()
        red = max(8, int(c * ratio))
        self.fc1 = nn.Conv2d(c, red, 1)
        self.fc2 = nn.Conv2d(red, c, 1)

    def forward(self, x):
        s = self.fc2(F.relu(self.fc1(x.mean((2, 3), keepdim=True))))
        return x * F.hardsigmoid(s)


# ---------------------------------------------------------------------------
class SteerableAtomBank(nn.Module):
    """r analytic masks over the rfft2 grid + a protected residual band.

    atom_j(u,v) = exp(kappa_j * (cos(2(phi - theta_j)) - 1))          <- von Mises
                * exp(-(log rho - mu_j)^2 / (2 sigma_j^2))            <- log-radial

    The doubled angle makes the atom pi-periodic (correct for the orientation of
    a real signal) and wrap-safe: no angle is ever regressed or averaged, it
    enters only through cos/sin.  rho -> 0 kills every atom, so DC always lands
    in the residual band and can never be touched.

    Normalised into a partition of unity:
        M_j = A_j / (1 + sum_k A_k),   M_res = 1 / (1 + sum_k A_k)
        sum_j M_j + M_res == 1   (exact reconstruction, identity reachable)

    Parameters are 4 scalars per atom, defined on NORMALISED frequency, so the
    bank is resolution independent (SFANet's frequency-resolution-mismatch trap).
    """

    def __init__(self, n_theta=4, n_rad=2, isotropic=False):
        super().__init__()
        self.n_theta, self.n_rad, self.r = n_theta, n_rad, n_theta * n_rad
        self.isotropic = isotropic
        th = torch.arange(n_theta, dtype=torch.float32) * math.pi / n_theta
        mu = torch.linspace(-2.2, -1.0, n_rad)          # log of normalised radius
        self.theta = nn.Parameter(th.view(-1, 1).expand(n_theta, n_rad).reshape(-1).clone())
        self.mu = nn.Parameter(mu.view(1, -1).expand(n_theta, n_rad).reshape(-1).clone())
        self.log_kappa = nn.Parameter(torch.full((self.r,), math.log(2.0)))
        self.log_sigma = nn.Parameter(torch.full((self.r,), math.log(0.6)))
        self._grid = {}

    def grid(self, Hf, Wf, device, dtype):
        """The 5 fixed basis maps the atom exponent is LINEAR in.

        log A_j = kappa_j cos2t_j . c2  +  kappa_j sin2t_j . s2
                  - 1/(2 sig_j^2) . lr^2  +  mu_j/sig_j^2 . lr
                  - kappa_j - mu_j^2/(2 sig_j^2)
        so the whole bank is one [r,5] @ [5, Hf*Wf] matmul followed by one exp.
        """
        key = (Hf, Wf, str(device), str(dtype))
        if key not in self._grid:
            fy = torch.fft.fftfreq(Hf, device=device, dtype=dtype).view(-1, 1)
            fx = torch.linspace(0.0, 0.5, Wf, device=device, dtype=dtype).view(1, -1)
            fy, fx = fy.expand(Hf, Wf), fx.expand(Hf, Wf)
            rho = torch.sqrt(fy * fy + fx * fx).clamp_min(1e-6)
            c2 = (fx * fx - fy * fy) / (rho * rho)      # cos 2phi  (even in fy)
            s2 = (2.0 * fx * fy) / (rho * rho)          # sin 2phi  (odd  in fy)
            # Hermitian repair: on the fx = 0 and fx = Nyquist columns the rfft2
            # grid stores S[u,c] and S[-u,c] = conj(S[u,c]).  A real mask keeps
            # that identity only if it is EVEN in fy there, i.e. s2 must vanish.
            s2 = s2.clone()
            s2[:, 0] = 0.0
            s2[:, -1] = 0.0
            lr = torch.log(rho)
            self._grid[key] = torch.stack(
                [c2, s2, lr * lr, lr, torch.ones_like(lr)]).reshape(5, -1)
        return self._grid[key]

    def coeffs(self):
        kappa = self.log_kappa.exp()
        inv2s2 = 0.5 / self.log_sigma.exp().clamp_min(1e-3).pow(2)
        th2 = 2.0 * self.theta
        wc = kappa * torch.cos(th2)
        ws = kappa * torch.sin(th2)
        if self.isotropic:                                # ablation arm F
            wc, ws = torch.zeros_like(wc), torch.zeros_like(ws)
        return torch.stack([wc, ws, -inv2s2, 2 * self.mu * inv2s2,
                            -kappa - self.mu.pow(2) * inv2s2], dim=1)   # [r,5]

    def train(self, mode=True):
        self._mask = {}                     # atom params changed -> drop the cache
        return super().train(mode)

    def forward(self, Hf, Wf, device, dtype=torch.float32):
        key = (Hf, Wf, str(device))
        if not self.training and key in getattr(self, "_mask", {}):
            return self._mask[key]          # at inference the atoms are constants
        basis = self.grid(Hf, Wf, device, dtype)                        # [5, L]
        A = torch.exp(self.coeffs() @ basis).view(self.r, Hf, Wf)
        M = A / (1.0 + A.sum(0, keepdim=True))                          # [r,Hf,Wf]
        if not self.training:
            self._mask = getattr(self, "_mask", {})
            self._mask[key] = M
        return M


# ---------------------------------------------------------------------------
class ObliqueGate(nn.Module):
    """Per-pixel SIGNED band gain.  Zero-init -> exact identity at init.

    Trunk = 1x1 -> three PARALLEL dilated depthwise 3x3 (d = 1, 4, 16) summed
    -> Hardswish -> 1x1.  This is the receptive-field configuration that won the
    measured per-pixel-gain study (NMSE 0.3304 @ 801 params / 0.13 ms, beating a
    4-direction fused selective scan at 0.3641 / 2465 params / 1.15 ms).
    """

    def __init__(self, in_ch, r, hidden=16, dilations=(1, 4, 16), scale=1.5, gain_split=False):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.dw = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d, groups=hidden, bias=False)
            for d in dilations])
        self.act = nn.Hardswish()
        self.head = nn.Conv2d(hidden, r, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # GainSplit: the measured gate is 84% a per-band CONSTANT and 4/8 bands sit
        # pinned at the +-scale tanh rail. Give the constant its own UNBOUNDED
        # parameter so it never has to reach the rail, and leave tanh to carry only
        # the spatial residual. Zero-init keeps the exact identity at init.
        self.gain_split = gain_split
        self.band_const = nn.Parameter(torch.zeros(1, r, 1, 1)) if gain_split else None

    def forward(self, feat, film=None):
        h = self.proj(feat)
        h = sum(dw(h) for dw in self.dw)
        z = self.head(self.act(h))                                      # [B,r,H,W]
        if film is not None:
            gamma, beta = film
            z = z * (1.0 + gamma) + beta
        g = self.scale * torch.tanh(z)                                  # (-s, s)
        return g + self.band_const if self.gain_split else g


# ---------------------------------------------------------------------------
class RainSubspaceBlock(nn.Module):
    def __init__(self, dim=32, n_theta=4, n_rad=2, probe=4, gate_hidden=16,
                 ffn_expand=2.0, se_ratio=0.25, isotropic=False,
                 global_gate=False, positive_gate=False, temporal=False,
                 gain_split=False, dilations=(1, 4, 16)):
        super().__init__()
        self.dim, self.probe = dim, probe
        self.global_gate = global_gate
        self.temporal = temporal
        self.bank = SteerableAtomBank(n_theta, n_rad, isotropic=isotropic)
        r = self.r = self.bank.r

        self.norm1 = nn.LayerNorm(dim)
        self.to_probe = nn.Conv2d(dim, probe, 1, bias=False)   # commutes with the FFT
        self.band_scale = nn.Parameter(torch.ones(1, r, probe, 1, 1))
        self.ctx_proj = nn.Conv2d(dim, 8, 1, bias=False)
        # global band-energy profile -> FiLM on the gate logits (r floats in)
        self.film = nn.Sequential(nn.Linear(r, 16), nn.Hardswish(), nn.Linear(16, 2 * r))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.gate = ObliqueGate(2 * r + 1 + 8, r, gate_hidden,
                                dilations=dilations,
                                scale=1.0 if positive_gate else 1.5,
                                gain_split=gain_split)
        self.positive_gate = positive_gate
        # NB: exactly ONE of {gate head, lift} may be zero-initialised.  Zeroing
        # both makes the whole projection branch permanently dead (each one's
        # gradient is proportional to the other).  The gate head is the zeroed
        # one -- that gives an exact identity at init AND lets the gate receive
        # gradient on the very first step; only `lift` waits one step.
        self.lift = nn.Conv2d(probe, dim, 1, bias=True)
        nn.init.zeros_(self.lift.bias)

        self.norm2 = nn.LayerNorm(dim)
        hid = make_divisible(dim * ffn_expand)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),
            nn.Hardswish(), SqueezeExcite(hid, se_ratio),
            nn.Conv2d(hid, dim, 1, bias=True))
        self.register_buffer("ema", torch.zeros(1, r), persistent=False)
        self.ema_m = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, C, H, W = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # ---- 1. rank-p read-out, then ONE forward transform of p channels ----
        P = torch.fft.rfft2(self.to_probe(xn).float(), norm='ortho')     # [B,p,Hf,Wf]
        Hf, Wf = P.shape[-2:]
        M = self.bank(Hf, Wf, x.device)                                  # [r,Hf,Wf]

        # ---- 2. subspace analysis: ONE inverse transform of r*p channels -----
        Sb = P.unsqueeze(1) * M.unsqueeze(0).unsqueeze(2)                # [B,r,p,Hf,Wf]
        Y = torch.fft.irfft2(Sb.reshape(B, self.r * self.probe, Hf, Wf),
                             s=(H, W), norm='ortho')
        Y = Y.view(B, self.r, self.probe, H, W)

        # ---- 3. evidence for the gate ---------------------------------------
        e = Y.pow(2).mean(2)                                             # [B,r,H,W]
        tot = e.sum(1, keepdim=True) + 1e-8
        q = e / tot                                                      # band profile
        ent = -(q * (q + 1e-8).log()).sum(1, keepdim=True)               # peakiness
        feat = torch.cat([torch.log(e + 1e-6), q, ent, self.ctx_proj(xn)], 1)

        # global band-energy profile: which bands hold this frame's rain energy
        Eg = (P.abs().pow(2).sum(1).unsqueeze(1) * M.unsqueeze(0)).sum((-1, -2))
        Eg = Eg / (Eg.sum(1, keepdim=True) + 1e-8)                       # [B,r]
        if self.temporal:                                                # carry the
            m = torch.sigmoid(self.ema_m)                                # SUMMARY only
            if self.training and self.ema.shape[0] == B:
                Eg = (1 - m) * Eg + m * self.ema
            self.ema = Eg.detach()
        gamma, beta = self.film(Eg).chunk(2, dim=1)
        film = (gamma.view(B, self.r, 1, 1), beta.view(B, self.r, 1, 1))

        g = self.gate(feat, film)                                        # [B,r,H,W]
        if self.positive_gate:
            g = 0.5 * (g + 1.0)                                          # ablation H
        if self.global_gate:
            g = g.mean((-1, -2), keepdim=True).expand_as(g)              # ablation G

        # ---- 4. oblique projection + rank-p lift -----------------------------
        rain = (g.unsqueeze(2) * Y * self.band_scale).sum(1)             # [B,p,H,W]
        x = x - self.lift(rain)

        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


class ORSPNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, dim=32, num_blocks=4,
                 n_theta=4, n_rad=2, probe=4, **kw):
        super().__init__()
        if isinstance(num_blocks, (list, tuple)):
            num_blocks = int(sum(num_blocks))
        dim = make_divisible(dim)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, padding=1, groups=in_chans, bias=False),
            nn.Conv2d(in_chans, dim, 1, bias=True), nn.Hardswish())
        self.blocks = nn.ModuleList([
            RainSubspaceBlock(dim, n_theta, n_rad, probe, **kw) for _ in range(num_blocks)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, out_chans, 1, bias=True))

    def forward(self, x):
        f = self.in_proj(x)
        for b in self.blocks:
            f = b(f)
        return self.out_proj(f) + x
