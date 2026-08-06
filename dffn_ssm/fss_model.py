"""FSSNet — Dynamic Fourier filtering where the filter is a SELECTIVE STATE
SPACE SCAN over the spectrum instead of a K^2 local kernel.

Three changes vs DynamicFourierFilterNet:
  1. the filter-generation net emits SSM parameters (delta, B, C) — d + 2N
     channels — instead of C*k^2*2 explicit taps;
  2. the scan is an IIR filter over the *whole* ordered spectrum, so the
     receptive field is global and the response can attenuate or amplify
     (a softmax convex combination can only interpolate between neighbours);
  3. the spectrum is carried as (real, imag), never as a wrapped phase angle,
     so no branch-cut error.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


def polar_order(Hf, Wf, device, nbins=64):
    """Order bins by (angle, radius): rain occupies an angular wedge, so a
    rain-affected band becomes a contiguous run for the scan to gate on."""
    fy = torch.fft.fftfreq(Hf, device=device).view(-1, 1).expand(Hf, Wf)
    fx = (torch.arange(Wf, device=device).float() / (2 * (Wf - 1))).view(1, -1).expand(Hf, Wf)
    ang = torch.atan2(fy, fx)
    rad = torch.sqrt(fy ** 2 + fx ** 2)
    abin = ((ang + torch.pi) / (2 * torch.pi) * nbins).long().clamp(0, nbins - 1)
    key = abin.flatten().float() * 1e3 + rad.flatten() * 1e2
    order = torch.argsort(key)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=device)
    return order, inv


class FreqScan(nn.Module):
    def __init__(self, d, d_state=16):
        super().__init__()
        self.d, self.N = d, d_state
        h = max(8, d // 4)
        self.gen = nn.Sequential(
            nn.Conv2d(d, h, 1, bias=False),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False),
            nn.Hardswish(),
            nn.Conv2d(h, d + 2 * d_state, 1),
        )
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log_b = nn.Parameter(torch.log(A.clone()))
        self.D = nn.Parameter(torch.ones(d))
        self.D_b = nn.Parameter(torch.ones(d))
        self.dt_bias = nn.Parameter(torch.zeros(d))
        self.out_scale = nn.Parameter(torch.zeros(d))   # zero-init: starts as identity

    def forward(self, u, order=None, inv=None, state=None):
        B, d, Hf, Wf = u.shape
        p = self.gen(u).flatten(2)
        s = u.flatten(2)
        if order is not None:
            s, p = s[:, :, order], p[:, :, order]
        delta, Bp, Cp = torch.split(p, [self.d, self.N, self.N], dim=1)
        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_log_b.float())
        y = selective_scan_fn(s, delta, A, Bp, Cp, self.D.float(), z=None,
                              delta_bias=self.dt_bias.float(), delta_softplus=True)
        y = y + selective_scan_fn(s.flip(-1), delta.flip(-1), A_b, Bp.flip(-1),
                                  Cp.flip(-1), self.D_b.float(), z=None,
                                  delta_bias=self.dt_bias.float(),
                                  delta_softplus=True).flip(-1)
        if inv is not None:
            y = y[:, :, inv]
        return (y * self.out_scale.view(1, -1, 1)).view(B, d, Hf, Wf)


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        red = max(8, int(c * r))
        self.fc1 = nn.Conv2d(c, red, 1)
        self.fc2 = nn.Conv2d(red, c, 1)

    def forward(self, x):
        s = self.fc2(F.relu(self.fc1(x.mean((2, 3), keepdim=True))))
        return x * F.hardsigmoid(s)


class FSSBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.scan = FreqScan(2 * dim, d_state)
        h = make_divisible(dim * 2.0)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, h, 1, bias=False),
            nn.Conv2d(h, h, 3, padding=1, groups=h, bias=False),
            nn.Hardswish(), SqueezeExcite(h),
            nn.Conv2d(h, dim, 1, bias=True))

    def forward(self, x, order=None, inv=None):
        B, C, Hs, Ws = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Z = torch.fft.rfft2(xn.float(), norm='ortho')
        y = self.scan(torch.cat([Z.real, Z.imag], 1), order, inv)
        x = x + torch.fft.irfft2(torch.complex(y[:, :C], y[:, C:]),
                                 s=(Hs, Ws), norm='ortho')
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


class FSSNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, dim=32, nb=4, d_state=16, order=None):
        super().__init__()
        self.order_mode = order
        self._cache = {}
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, padding=1, groups=in_chans, bias=False),
            nn.Conv2d(in_chans, dim, 1), nn.Hardswish())
        self.blocks = nn.ModuleList([FSSBlock(dim, d_state) for _ in range(nb)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, out_chans, 1))

    def _order(self, Hf, Wf, device):
        if self.order_mode != "polar":
            return None, None
        key = (Hf, Wf, str(device))
        if key not in self._cache:
            self._cache[key] = polar_order(Hf, Wf, device)
        return self._cache[key]

    def forward(self, x):
        f = self.in_proj(x)
        Hf, Wf = x.shape[-2], x.shape[-1] // 2 + 1
        order, inv = self._order(Hf, Wf, x.device)
        for b in self.blocks:
            f = b(f, order, inv)
        return self.out_proj(f) + x
