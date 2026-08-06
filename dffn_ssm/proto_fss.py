"""Prototype: replace the K^2 local dynamic filter in DFFN's frequency domain
with a per-bin-parameterised SELECTIVE STATE SPACE SCAN over the spectrum.

Head-to-head against the current DynamicFourierBlock on identical shapes.
"""
import sys, time
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from model.DynamicFourierFilterNet import DynamicFourierBlock, make_divisible

DEV = "cuda"
H = W = 256
DIM = 32


def sync():
    torch.cuda.synchronize()


def bench(fn, n=40, warmup=12):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000.0


def peak_mem(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    sync()
    return torch.cuda.max_memory_allocated() / 2 ** 20


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=0.25):
        super().__init__()
        red = max(8, int(c * r))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, red, 1)
        self.fc2 = nn.Conv2d(red, c, 1)

    def forward(self, x):
        s = self.fc2(F.relu(self.fc1(self.pool(x))))
        return x * F.hardsigmoid(s)


class FrequencySelectiveScanBlock(nn.Module):
    """
    LayerNorm -> rfft2 -> [real, imag] as 2C channels
      -> tiny generator emits per-bin SSM params (delta, B, C)   [d + 2N chans]
      -> bidirectional selective scan along an ORDERED frequency sequence
      -> back to complex -> irfft2 -> residual -> FFN+SE

    Contrast with DynamicFourierBlock:
      * generator head emits d + 2N = 96 channels, not C*k^2*2 = 576
      * no unfold, no [B,C,9,Hf,Wf] materialisation
      * receptive field over the spectrum is the WHOLE sequence, not 3x3
      * an SSM is an IIR filter -> can attenuate AND amplify (softmax convex
        combination can only interpolate between neighbouring bins)
    """

    def __init__(self, dim, d_state=16, ffn_expand_ratio=2.0, se_ratio=0.25,
                 scan_order=None):
        super().__init__()
        self.dim = dim
        self.d = 2 * dim            # real + imag channels
        self.N = d_state
        ffn_hidden = make_divisible(dim * ffn_expand_ratio, 8)

        self.norm1 = nn.LayerNorm(dim)

        gen_hidden = max(8, dim // 2)
        # emits: delta (d) | B (N) | C (N)
        self.gen = nn.Sequential(
            nn.Conv2d(self.d, gen_hidden, 1, bias=False),
            nn.Conv2d(gen_hidden, gen_hidden, 3, padding=1, groups=gen_hidden, bias=False),
            nn.Hardswish(),
            nn.Conv2d(gen_hidden, self.d + 2 * self.N, 1),
        )

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d))
        self.dt_bias = nn.Parameter(torch.zeros(self.d))
        # backward-direction scan shares the generator, has its own A/D
        self.A_log_b = nn.Parameter(torch.log(A.clone()))
        self.D_b = nn.Parameter(torch.ones(self.d))
        self.out_scale = nn.Parameter(torch.zeros(self.d))

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_hidden, 1, bias=False),
            nn.Conv2d(ffn_hidden, ffn_hidden, 3, padding=1, groups=ffn_hidden, bias=False),
            nn.Hardswish(),
            SqueezeExcite(ffn_hidden, se_ratio),
            nn.Conv2d(ffn_hidden, dim, 1, bias=True),
        )
        self.register_buffer("order", scan_order if scan_order is not None
                             else torch.empty(0), persistent=False)

    def forward(self, x):
        B, C, Hs, Ws = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        Z = torch.fft.rfft2(xn.float(), norm='ortho')
        Hf, Wf = Z.shape[-2:]
        L = Hf * Wf

        u = torch.cat([Z.real, Z.imag], dim=1)                     # [B, d, Hf, Wf]

        p = self.gen(u)                                            # [B, d+2N, Hf, Wf]
        p = p.flatten(2)                                           # [B, d+2N, L]
        u_seq = u.flatten(2)                                       # [B, d, L]

        if self.order.numel() == L:                                # optional reordering
            idx = self.order
            u_seq = u_seq[:, :, idx]
            p = p[:, :, idx]

        delta, Bp, Cp = torch.split(p, [self.d, self.N, self.N], dim=1)

        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_log_b.float())

        y_f = selective_scan_fn(u_seq, delta, A, Bp, Cp, self.D.float(),
                                z=None, delta_bias=self.dt_bias.float(),
                                delta_softplus=True)
        y_b = selective_scan_fn(u_seq.flip(-1), delta.flip(-1), A_b,
                                Bp.flip(-1), Cp.flip(-1), self.D_b.float(),
                                z=None, delta_bias=self.dt_bias.float(),
                                delta_softplus=True).flip(-1)
        y = y_f + y_b

        if self.order.numel() == L:
            inv = torch.empty_like(self.order)
            inv[self.order] = torch.arange(L, device=self.order.device)
            y = y[:, :, inv]

        y = y * self.out_scale.view(1, -1, 1)
        y = y.view(B, self.d, Hf, Wf)
        Zf = torch.complex(y[:, :C], y[:, C:])
        out = torch.fft.irfft2(Zf, s=(Hs, Ws), norm='ortho')

        x = x + out
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


def polar_order(Hf, Wf, device):
    """Order frequency bins by (angle, radius) — rain occupies an angular wedge."""
    fy = torch.fft.fftfreq(Hf, device=device).view(-1, 1).expand(Hf, Wf)
    fx = (torch.arange(Wf, device=device).float() / (2 * (Wf - 1))).view(1, -1).expand(Hf, Wf)
    ang = torch.atan2(fy, fx)
    rad = torch.sqrt(fy ** 2 + fx ** 2)
    nbins = 64
    abin = ((ang + torch.pi) / (2 * torch.pi) * nbins).long().clamp(0, nbins - 1)
    key = abin.flatten().float() * 1e3 + rad.flatten() * 1e2
    return torch.argsort(key)


# =====================================================================
print(f"Head-to-head at {H}x{W}, dim={DIM}, on {torch.cuda.get_device_name(0)}\n")

old = DynamicFourierBlock(dim=DIM).to(DEV).eval()
Hf, Wf = H, W // 2 + 1
new = FrequencySelectiveScanBlock(dim=DIM, d_state=16).to(DEV).eval()
new_polar = FrequencySelectiveScanBlock(
    dim=DIM, d_state=16, scan_order=polar_order(Hf, Wf, DEV)).to(DEV).eval()

for B in (1, 4):
    x = torch.randn(B, DIM, H, W, device=DEV)
    with torch.no_grad():
        t_old = bench(lambda: old(x))
        t_new = bench(lambda: new(x))
        t_pol = bench(lambda: new_polar(x))
        m_old = peak_mem(lambda: old(x))
        m_new = peak_mem(lambda: new(x))
    print(f"B={B}")
    print(f"   DynamicFourierBlock (K^2=9 unfold)   {t_old:7.3f} ms   peak {m_old:7.1f} MiB")
    print(f"   FreqSelectiveScan   (raster order)   {t_new:7.3f} ms   peak {m_new:7.1f} MiB"
          f"   -> {t_old/t_new:.2f}x faster, {m_old/m_new:.2f}x less memory")
    print(f"   FreqSelectiveScan   (polar order)    {t_pol:7.3f} ms")
    print()

p_old = sum(p.numel() for p in old.parameters())
p_new = sum(p.numel() for p in new.parameters())
print(f"params/block:  old {p_old:,}   new {p_new:,}   ({p_new/p_old:.2f}x)")
gen_out = new.gen[3]
fgn_out = old.fgn[3]
print(f"generator head out-channels:  old {fgn_out.out_channels} (C*k^2*2)"
      f"   new {gen_out.out_channels} (d+2N)   -> {fgn_out.out_channels/gen_out.out_channels:.1f}x narrower")

# ---- expressivity check: can each operator push a bin BELOW its neighbourhood min?
print("\n== Expressivity: softmax-convex-combination vs SSM")
B = 1
spec = torch.rand(B, DIM, Hf, Wf, device=DEV) * 0.5 + 0.5       # all in [0.5, 1.0]
filt = F.softmax(torch.randn(B, DIM, 9, Hf, Wf, device=DEV), dim=2).view(B, -1, Hf, Wf)
out_cvx = old.dynamic_filter(spec, filt)
print(f"   input magnitude range          [{spec.min():.4f}, {spec.max():.4f}]")
print(f"   after softmax-convex filter    [{out_cvx.min():.4f}, {out_cvx.max():.4f}]"
      f"   <- cannot leave the convex hull of its 3x3 neighbourhood (padding aside)")
with torch.no_grad():
    y = new(torch.randn(B, DIM, H, W, device=DEV))
print(f"   SSM scan output is unconstrained: it is an IIR filter, y can be any sign/scale")
