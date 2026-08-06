"""Mechanism isolation: give DFFN's filter generator GLOBAL spectral context,
changing nothing else.

If the SSM's benefit really comes from seeing the whole spectrum rather than a
3x3 neighbourhood, then simply handing the existing 3x3 generator a global
descriptor should already recover part of the gain — with no SSM involved.
This separates "global context matters" from "the SSM is a good way to get it".

Descriptor = per-orientation-sector and per-radial-ring energy of the
magnitude spectrum (rain's signature is a stable orientation profile), passed
through a tiny MLP and broadcast to every frequency bin.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DynamicFourierFilterNet import (DynamicFilterLayer2D, SqueezeExcite,
                                           make_divisible)


class SpectralContext(nn.Module):
    """Orientation x radius energy histogram of the magnitude spectrum."""

    def __init__(self, dim, out_dim, n_ang=12, n_rad=6):
        super().__init__()
        self.n_ang, self.n_rad = n_ang, n_rad
        self.mlp = nn.Sequential(
            nn.Linear(n_ang * n_rad, 64), nn.Hardswish(), nn.Linear(64, out_dim))
        self._cache = {}

    def _bins(self, Hf, Wf, device):
        key = (Hf, Wf, str(device))
        if key not in self._cache:
            fy = torch.fft.fftfreq(Hf, device=device).view(-1, 1).expand(Hf, Wf)
            fx = (torch.arange(Wf, device=device).float()
                  / (2 * (Wf - 1))).view(1, -1).expand(Hf, Wf)
            angm = torch.remainder(torch.atan2(fy, fx), torch.pi)
            rad = torch.sqrt(fy ** 2 + fx ** 2)
            a = (angm / torch.pi * self.n_ang).long().clamp(0, self.n_ang - 1)
            r = (rad / (rad.max() + 1e-8) * self.n_rad).long().clamp(0, self.n_rad - 1)
            self._cache[key] = (a * self.n_rad + r).flatten()
        return self._cache[key]

    def forward(self, mag):
        B, C, Hf, Wf = mag.shape
        idx = self._bins(Hf, Wf, mag.device)
        e = mag.mean(1).flatten(1)                                  # [B, L]
        h = torch.zeros(B, self.n_ang * self.n_rad, device=mag.device)
        h.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), e)
        h = h / (h.sum(1, keepdim=True) + 1e-8)
        return self.mlp(h)                                          # [B, out_dim]


class DynamicFourierBlockGlobal(nn.Module):
    """DynamicFourierBlock with a global spectral descriptor fed to the FGN."""

    def __init__(self, dim, kernel_size=3, fgn_bottleneck_ratio=0.5,
                 ffn_expand_ratio=2.0, se_ratio=0.25, ctx_dim=16):
        super().__init__()
        self.dim = dim
        self.k2 = kernel_size ** 2
        fgn_hidden = max(8, int(dim * fgn_bottleneck_ratio))
        ffn_hidden = make_divisible(dim * ffn_expand_ratio, 8)

        self.norm1 = nn.LayerNorm(dim)
        self.ctx = SpectralContext(dim, ctx_dim)
        self.fgn = nn.Sequential(
            nn.Conv2d(dim * 2 + ctx_dim, fgn_hidden, 1, bias=False),
            nn.Conv2d(fgn_hidden, fgn_hidden, 3, padding=1, groups=fgn_hidden, bias=False),
            nn.Hardswish(),
            nn.Conv2d(fgn_hidden, dim * self.k2 * 2, 1),
        )
        self.dynamic_filter = DynamicFilterLayer2D(kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_hidden, 1, bias=False),
            nn.Conv2d(ffn_hidden, ffn_hidden, 3, padding=1, groups=ffn_hidden, bias=False),
            nn.Hardswish(), SqueezeExcite(ffn_hidden, se_ratio=se_ratio),
            nn.Conv2d(ffn_hidden, dim, 1, bias=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        fft_feat = torch.fft.rfft2(xn.float(), norm='ortho')
        mag = torch.abs(fft_feat) + 1e-8
        phase = torch.angle(fft_feat)
        Hf, Wf = mag.shape[-2:]

        g = self.ctx(mag).view(B, -1, 1, 1).expand(-1, -1, Hf, Wf)
        filters = self.fgn(torch.cat([mag, phase, g], dim=1))

        mf, pf = torch.chunk(filters, 2, dim=1)
        mf = F.softmax(mf.view(B, C, self.k2, Hf, Wf), dim=2).view(B, -1, Hf, Wf)
        pf = F.softmax(pf.view(B, C, self.k2, Hf, Wf), dim=2).view(B, -1, Hf, Wf)
        fm = self.dynamic_filter(mag, mf)
        fp = self.dynamic_filter(phase, pf)
        out = torch.fft.irfft2(torch.complex(fm * torch.cos(fp), fm * torch.sin(fp)),
                               s=(H, W), norm='ortho')
        x = x + out
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


class DFFNGlobal(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, dim=32, num_blocks=4, ctx_dim=16):
        super().__init__()
        if isinstance(num_blocks, (list, tuple)):
            num_blocks = int(sum(num_blocks))
        dim = make_divisible(dim, 8)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, padding=1, groups=in_chans, bias=False),
            nn.Conv2d(in_chans, dim, 1, bias=True), nn.Hardswish())
        self.blocks = nn.ModuleList(
            [DynamicFourierBlockGlobal(dim, ctx_dim=ctx_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, out_chans, 1, bias=True))

    def forward(self, x):
        f = self.in_proj(x)
        for b in self.blocks:
            f = b(f)
        return self.out_proj(f) + x
