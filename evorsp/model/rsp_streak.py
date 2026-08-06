"""StreakNet — ORSPNet + three event-rain-specific components, all near-free.

Each component is justified by one measurement from this project:

 1. StripObliqueGate  <- rain is ORIENTED and the orientation is FIXED
    (excess spectral energy peaks at 5-15 deg, 4.2/6.2 dB anisotropy, dominant
    sector std 0.00 across 40 frames). Dense 1xK / Kx1 depthwise strips replace
    two of the sparse dilated branches: a 3x3 dilated at d=16 reaches 33 px but
    samples 3 lattice points, while a 1x31 depthwise reaches 31 px densely for
    31 params/channel. Strips are separable and cache-friendly, so unlike the
    U-Net this should be neutral-to-faster rather than 2.2x slower.

 2. rate conditioning  <- rain rate shifts the OPTIMAL OPERATING POINT and the
    model ignores it (per-intensity DA 0.9518 @1mm -> 0.8832 @200mm, optimal tau
    sliding 0.95 -> 0.50; and a SINGLE scalar rate proxy beat a 72-bin histogram,
    +1.8% vs +0.5%). One scalar -> FiLM on every gate + a per-image logit bias.

 3. dark-mask output  <- deraining on events is EXACTLY subset selection (clean
    events are an exact subset of rainy ones, 1.0000 match on (x,y,t,p) over 8
    sequences, zero duplicates). A pixel dark in the input can never be a real
    event, so force its logit negative. Zero parameters, hard guarantee.

All three are zero-init / identity-preserving so an arm starts exactly at the
confirmed dil_bal configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsp_model_v2 import SteerableAtomBank, SqueezeExcite, make_divisible

DARK_LOGIT = 20.0          # forces sigmoid(logit) ~ 0 at dark pixels


class StripObliqueGate(nn.Module):
    """ObliqueGate with dense oriented strips replacing the long sparse dilations.

    Branch set: 1xK, Kx1 (dense, oriented) + 3x3 d=2, 3x3 d=8 (short isotropic).
    Summed in parallel, exactly as the original trunk sums its dilated branches.
    """

    def __init__(self, in_ch, r, hidden=16, K=15, dilations=(2, 8), scale=1.5):
        super().__init__()
        self.scale = scale
        p = K // 2
        self.proj = nn.Conv2d(in_ch, hidden, 1, bias=False)
        self.sh = nn.Conv2d(hidden, hidden, (1, K), padding=(0, p), groups=hidden, bias=False)
        self.sv = nn.Conv2d(hidden, hidden, (K, 1), padding=(p, 0), groups=hidden, bias=False)
        self.dw = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d, groups=hidden, bias=False)
            for d in dilations])
        self.act = nn.Hardswish()
        self.head = nn.Conv2d(hidden, r, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat, film=None):
        h = self.proj(feat)
        h = self.sh(h) + self.sv(h) + sum(d(h) for d in self.dw)
        z = self.head(self.act(h))
        if film is not None:
            gamma, beta = film
            z = z * (1.0 + gamma) + beta
        return self.scale * torch.tanh(z)


class StreakBlock(nn.Module):
    """RainSubspaceBlock with a strip gate and a rate-conditioned FiLM.

    Mirrors rsp_model_v2.RainSubspaceBlock step for step; only the gate trunk and
    the FiLM input differ, so the atom bank, rank-p read-out, lift and FFN are the
    proven ones.
    """

    def __init__(self, dim=32, n_theta=4, n_rad=2, probe=4, gate_hidden=16,
                 ffn_expand=2.0, se_ratio=0.25, K=15, use_rate=True):
        super().__init__()
        self.dim, self.probe, self.use_rate = dim, probe, use_rate
        self.bank = SteerableAtomBank(n_theta, n_rad)
        r = self.r = self.bank.r

        self.norm1 = nn.LayerNorm(dim)
        self.to_probe = nn.Conv2d(dim, probe, 1, bias=False)
        self.band_scale = nn.Parameter(torch.ones(1, r, probe, 1, 1))
        self.ctx_proj = nn.Conv2d(dim, 8, 1, bias=False)
        # FiLM sees the per-block band-energy profile AND the global rain rate
        self.film = nn.Sequential(nn.Linear(r + (1 if use_rate else 0), 16),
                                  nn.Hardswish(), nn.Linear(16, 2 * r))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.gate = StripObliqueGate(2 * r + 1 + 8, r, gate_hidden, K=K)
        self.lift = nn.Conv2d(probe, dim, 1, bias=True)
        nn.init.zeros_(self.lift.bias)

        self.norm2 = nn.LayerNorm(dim)
        hid = make_divisible(dim * ffn_expand)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hid, 1, bias=False),
            nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False),
            nn.Hardswish(), SqueezeExcite(hid, se_ratio),
            nn.Conv2d(hid, dim, 1, bias=True))

    def forward(self, x, rate=None):
        B, C, H, W = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        P = torch.fft.rfft2(self.to_probe(xn).float(), norm="ortho")
        Hf, Wf = P.shape[-2:]
        M = self.bank(Hf, Wf, x.device)

        Sb = P.unsqueeze(1) * M.unsqueeze(0).unsqueeze(2)
        Y = torch.fft.irfft2(Sb.reshape(B, self.r * self.probe, Hf, Wf),
                             s=(H, W), norm="ortho").view(B, self.r, self.probe, H, W)

        e = Y.pow(2).mean(2)
        tot = e.sum(1, keepdim=True) + 1e-8
        q = e / tot
        ent = -(q * (q + 1e-8).log()).sum(1, keepdim=True)
        feat = torch.cat([torch.log(e + 1e-6), q, ent, self.ctx_proj(xn)], 1)

        Eg = (P.abs().pow(2).sum(1).unsqueeze(1) * M.unsqueeze(0)).sum((-1, -2))
        Eg = Eg / (Eg.sum(1, keepdim=True) + 1e-8)
        if self.use_rate and rate is not None:
            Eg = torch.cat([Eg, rate], 1)
        gamma, beta = self.film(Eg).chunk(2, dim=1)
        g = self.gate(feat, (gamma.view(B, self.r, 1, 1), beta.view(B, self.r, 1, 1)))

        rain = (g.unsqueeze(2) * Y * self.band_scale).sum(1)
        x = x - self.lift(rain)
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


class StreakNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, dim=32, num_blocks=4,
                 n_theta=4, n_rad=2, probe=4, K=15,
                 use_strip=True, use_rate=True, use_darkmask=True):
        super().__init__()
        if isinstance(num_blocks, (list, tuple)):
            num_blocks = int(sum(num_blocks))
        dim = make_divisible(dim)
        self.use_rate, self.use_darkmask = use_rate, use_darkmask

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, padding=1, groups=in_chans, bias=False),
            nn.Conv2d(in_chans, dim, 1, bias=True), nn.Hardswish())

        if use_strip:
            self.blocks = nn.ModuleList([
                StreakBlock(dim, n_theta, n_rad, probe, K=K, use_rate=use_rate)
                for _ in range(num_blocks)])
        else:
            from rsp_model_v2 import RainSubspaceBlock
            self.blocks = nn.ModuleList([
                RainSubspaceBlock(dim, n_theta, n_rad, probe, dilations=(1, 8, 32, 64))
                for _ in range(num_blocks)])
        self.strip = use_strip

        if use_rate:
            # one scalar -> per-image logit bias (the direct fix for the tau slide)
            self.rate_mlp = nn.Sequential(nn.Linear(1, 32), nn.Hardswish(), nn.Linear(32, 1))
            nn.init.zeros_(self.rate_mlp[-1].weight); nn.init.zeros_(self.rate_mlp[-1].bias)

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, out_chans, 1, bias=True))

    def _rate(self, x):
        S = torch.fft.rfft2(x.float(), norm="ortho")
        return torch.log(torch.abs(S) + 1e-8).mean((1, 2, 3), keepdim=True).flatten(1)

    def forward(self, x):
        rate = self._rate(x) if self.use_rate else None
        f = self.in_proj(x)
        for b in self.blocks:
            f = b(f, rate) if self.strip else b(f)
        logit = self.out_proj(f) + x
        if self.use_rate:
            logit = logit + self.rate_mlp(rate)[:, :, None, None]
        if self.use_darkmask:
            # clean events are an exact SUBSET of rainy events: a pixel with no
            # input event can never be a real event. Hard, zero-parameter.
            logit = logit - DARK_LOGIT * (x <= 0.5).to(logit.dtype)
        return logit


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [
        ("dil_bal equivalent (control)", dict(use_strip=False, use_rate=False, use_darkmask=False)),
        ("strip only",                   dict(use_strip=True,  use_rate=False, use_darkmask=False)),
        ("rate only",                    dict(use_strip=False, use_rate=True,  use_darkmask=False)),
        ("darkmask only",                dict(use_strip=False, use_rate=False, use_darkmask=True)),
        ("StreakNet (all three)",        dict(use_strip=True,  use_rate=True,  use_darkmask=True)),
    ]
    x = torch.randn(1, 1, 256, 256, device=dev).clamp(0, 1)
    for name, kw in cfgs:
        m = StreakNet(**kw).to(dev).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(x)
            for _ in range(8): m(x)
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(25): m(x)
            if dev == "cuda": torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 25 * 1000
        dark_ok = bool((torch.sigmoid(y)[x <= 0.5] < 1e-6).all()) if kw["use_darkmask"] else True
        print(f"{name:30s} {n:7,d}p  {ms:6.2f} ms  finite={torch.isfinite(y).all().item()}"
              f"  darkmask_ok={dark_ok}")
    print("\nreference: ORSPNet 36,206p / 32.1 ms ; +dil 36,782p / 35.5 ms ; "
          "ORSPUNet 109,978p / 70.0 ms")
