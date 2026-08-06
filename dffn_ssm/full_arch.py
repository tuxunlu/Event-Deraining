"""End-to-end architecture comparison:

  A. DynamicFourierFilterNet            (current baseline)
  B. FSSNet-perblock   FFT per block, K^2 dynamic filter -> selective freq scan
  C. FSSNet-trunk      ONE rfft2 at input, ONE irfft2 at output; every block
                       stays resident in the frequency domain.

For C, each spatial op is replaced by its frequency-domain counterpart:
  1x1 conv          -> 1x1 conv          (exact: channel-linear commutes with FFT)
  depthwise 3x3     -> learned per-bin complex mask (a GFNet global filter;
                       strictly more expressive than a 3x3 dw conv)
  squeeze-excite    -> gate read off the DC bin (spatial GAP *is* the DC
                       coefficient, up to the ortho normalisation constant)
  pointwise nonlin  -> applied to the spectrum (NOT equivalent; this is the
                       AFNO/FNO-style design choice, stated plainly)
"""
import sys, time
sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from model.DynamicFourierFilterNet import DynamicFourierFilterNet, make_divisible

DEV = "cuda"
H = W = 256
DIM = 32
NB = 4


def bench(fn, n=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def peak_mem(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


# ---------------------------------------------------------------- shared parts
class FreqScan(nn.Module):
    """Per-bin-parameterised bidirectional selective scan over the spectrum."""

    def __init__(self, d, d_state=16, gen_hidden=None):
        super().__init__()
        self.d, self.N = d, d_state
        gen_hidden = gen_hidden or max(8, d // 4)
        self.gen = nn.Sequential(
            nn.Conv2d(d, gen_hidden, 1, bias=False),
            nn.Conv2d(gen_hidden, gen_hidden, 3, padding=1, groups=gen_hidden, bias=False),
            nn.Hardswish(),
            nn.Conv2d(gen_hidden, d + 2 * d_state, 1),
        )
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log_b = nn.Parameter(torch.log(A.clone()))
        self.D = nn.Parameter(torch.ones(d))
        self.D_b = nn.Parameter(torch.ones(d))
        self.dt_bias = nn.Parameter(torch.zeros(d))
        self.out_scale = nn.Parameter(torch.zeros(d))

    def forward(self, u, order=None, inv=None):
        B, d, Hf, Wf = u.shape
        p = self.gen(u).flatten(2)
        s = u.flatten(2)
        if order is not None:
            s, p = s[:, :, order], p[:, :, order]
        delta, Bp, Cp = torch.split(p, [self.d, self.N, self.N], dim=1)
        A, A_b = -torch.exp(self.A_log.float()), -torch.exp(self.A_log_b.float())
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
        self.fc1 = nn.Conv2d(c, red, 1); self.fc2 = nn.Conv2d(red, c, 1)

    def forward(self, x):
        s = self.fc2(F.relu(self.fc1(x.mean((2, 3), keepdim=True))))
        return x * F.hardsigmoid(s)


# --------------------------------------------------------------- B: per-block
class FSSBlockPerBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.scan = FreqScan(2 * dim, d_state)
        ffn_hidden = make_divisible(dim * 2.0, 8)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_hidden, 1, bias=False),
            nn.Conv2d(ffn_hidden, ffn_hidden, 3, padding=1, groups=ffn_hidden, bias=False),
            nn.Hardswish(), SqueezeExcite(ffn_hidden),
            nn.Conv2d(ffn_hidden, dim, 1, bias=True))

    def forward(self, x, order=None, inv=None):
        B, C, Hs, Ws = x.shape
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Z = torch.fft.rfft2(xn.float(), norm='ortho')
        y = self.scan(torch.cat([Z.real, Z.imag], 1), order, inv)
        out = torch.fft.irfft2(torch.complex(y[:, :C], y[:, C:]), s=(Hs, Ws), norm='ortho')
        x = x + out
        xn2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.ffn(xn2)


class FSSNetPerBlock(nn.Module):
    def __init__(self, dim=DIM, nb=NB, d_state=16):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, groups=1, bias=False),
            nn.Conv2d(1, dim, 1), nn.Hardswish())
        self.blocks = nn.ModuleList([FSSBlockPerBlock(dim, d_state) for _ in range(nb)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, 1, 1))

    def forward(self, x, order=None, inv=None):
        f = self.in_proj(x)
        for b in self.blocks:
            f = b(f, order, inv)
        return self.out_proj(f) + x


# -------------------------------------------------------------- C: FFT trunk
class SpectralBlock(nn.Module):
    """Lives entirely in the frequency domain: no FFT inside the block."""

    def __init__(self, dim, Hf, Wf, d_state=16):
        super().__init__()
        d = 2 * dim
        self.d = d
        self.norm1 = nn.GroupNorm(1, d)
        self.scan = FreqScan(d, d_state)
        self.norm2 = nn.GroupNorm(1, d)
        h = make_divisible(d * 2.0, 8)
        self.fc1 = nn.Conv2d(d, h, 1, bias=False)
        # depthwise 3x3 spatial conv  ->  per-bin learned mask (GFNet global filter)
        self.mask = nn.Parameter(torch.ones(1, h, Hf, Wf))
        self.act = nn.Hardswish()
        self.se = SqueezeExciteDC(h)
        self.fc2 = nn.Conv2d(h, d, 1, bias=True)

    def forward(self, u, order=None, inv=None):
        u = u + self.scan(self.norm1(u), order, inv)
        v = self.norm2(u)
        v = self.fc1(v) * self.mask
        v = self.fc2(self.se(self.act(v)))
        return u + v


class SqueezeExciteDC(nn.Module):
    """Spatial global-average-pool == the DC coefficient of the spectrum."""

    def __init__(self, c, r=0.25):
        super().__init__()
        red = max(8, int(c * r))
        self.fc1 = nn.Conv2d(c, red, 1); self.fc2 = nn.Conv2d(red, c, 1)

    def forward(self, u):
        dc = u[:, :, :1, :1]                       # no reduction over H,W needed
        s = self.fc2(F.relu(self.fc1(dc)))
        return u * F.hardsigmoid(s)


class FSSNetTrunk(nn.Module):
    def __init__(self, dim=DIM, nb=NB, d_state=16, Hs=H, Ws=W):
        super().__init__()
        self.Hs, self.Ws = Hs, Ws
        Hf, Wf = Hs, Ws // 2 + 1
        self.in_proj = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.Conv2d(1, dim, 1), nn.Hardswish())
        self.blocks = nn.ModuleList(
            [SpectralBlock(dim, Hf, Wf, d_state) for _ in range(nb)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, 1, 1))

    def forward(self, x, order=None, inv=None):
        f = self.in_proj(x)
        C = f.shape[1]
        Z = torch.fft.rfft2(f.float(), norm='ortho')       # <-- the ONLY forward FFT
        u = torch.cat([Z.real, Z.imag], 1)
        for b in self.blocks:
            u = b(u, order, inv)
        Zo = torch.complex(u[:, :C], u[:, C:])
        f = torch.fft.irfft2(Zo, s=(self.Hs, self.Ws), norm='ortho')  # <-- only inverse
        return self.out_proj(f) + x


# =====================================================================
print(f"{torch.cuda.get_device_name(0)}   {H}x{W}, dim={DIM}, blocks={NB}\n")
Hf, Wf = H, W // 2 + 1
L = Hf * Wf

models = {
    "A. DynamicFourierFilterNet (baseline)": DynamicFourierFilterNet(
        in_chans=1, out_chans=1, dim=DIM, num_blocks=NB).to(DEV).eval(),
    "B. FSSNet  per-block FFT":              FSSNetPerBlock().to(DEV).eval(),
    "C. FSSNet  single-FFT trunk":           FSSNetTrunk().to(DEV).eval(),
}

print(f"{'model':40s} {'params':>9s} {'B=1 ms':>9s} {'B=4 ms':>9s} {'peak MiB':>10s} {'#FFTs':>6s}")
print("-" * 92)
base = {}
for name, m in models.items():
    p = sum(q.numel() for q in m.parameters())
    with torch.no_grad():
        x1 = torch.randn(1, 1, H, W, device=DEV)
        x4 = torch.randn(4, 1, H, W, device=DEV)
        t1 = bench(lambda: m(x1)); t4 = bench(lambda: m(x4))
        mem = peak_mem(lambda: m(x4))
    nfft = 2 * NB if name.startswith(("A", "B")) else 2
    print(f"{name:40s} {p:9,} {t1:9.3f} {t4:9.3f} {mem:10.1f} {nfft:6d}")
    base[name] = (t1, t4)

a1, a4 = base["A. DynamicFourierFilterNet (baseline)"]
print()
for name in ("B. FSSNet  per-block FFT", "C. FSSNet  single-FFT trunk"):
    t1, t4 = base[name]
    print(f"   {name:38s} speedup vs A:  B=1 {a1/t1:.2f}x   B=4 {a4/t4:.2f}x")

# ------------------------------------------------- streaming / temporal state
print("\n== Temporal state across windows (idea: carry SSM state between frames)")
print("   selective_scan_fn supports return_last_state=True -> the hidden state")
print("   [B, d, N] can be carried across time windows.")
d, N = 2 * DIM, 16
u = torch.randn(1, d, L, device=DEV)
delta = torch.rand(1, d, L, device=DEV)
A = -torch.exp(torch.randn(d, N, device=DEV))
Bp = torch.randn(1, N, L, device=DEV); Cp = torch.randn(1, N, L, device=DEV)
y, last = selective_scan_fn(u, delta, A, Bp, Cp, torch.ones(d, device=DEV),
                            delta_softplus=True, return_last_state=True)
print(f"   carried state shape {tuple(last.shape)} = {last.numel()*4/1024:.1f} KiB per sample")
print(f"   -> temporal memory costs {last.numel()*4/1024:.1f} KiB and ZERO extra FLOPs/frame,")
print(f"      versus re-processing a multi-frame window (linear in window length).")
