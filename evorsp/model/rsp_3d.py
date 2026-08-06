"""ORSPNet-3D: a separable spatio-temporal log-Gabor bank.

THE IDEA. ORSPNet's atom is a product of two separable factors over the 2D
spectrum -- an orientation factor and a spatial-scale factor. Add a third,
over TEMPORAL frequency:

    Psi_j(wx, wy, wt) = [von Mises in orientation]        <- unchanged
                      * [log-Gaussian in spatial rho]    <- unchanged
                      * [log-Gaussian in |wt|]           <- NEW

WHY SEPARABILITY IS THE WHOLE TRICK. Because the atom factorises as
Psi = Psi_s(wx,wy) * Psi_t(wt), the temporal factor can be applied ONCE at the
front over the T axis and the spatial factors left exactly where they are,
inside the blocks. We never materialise a 3D [B, r, probe, T, H, W] tensor,
which at T=16 would be ~400 MB per sample and is what makes the naive 3D
extension unaffordable. Cost is a 1-D FFT over T plus a few extra input
channels.

WHAT THIS IS AND IS NOT. Pooling energy over t after the temporal filter is the
Adelson & Bergen (1985) motion-energy construction, not a linear 3D
convolution: the full linear operator would keep the t axis alive through the
spatial stage. This is an approximation, deliberately, and it is the one the
data supports -- see below.

WHY TEMPORAL *BANDWIDTH* AND NOT VELOCITY. The textbook move is
velocity-selective filtering: a pattern translating at velocity v concentrates
on the plane wx*vx + wy*vy + wt = 0. We measured that and it fails here --
contrast-maximisation mirror ratio for SPAC rain is 1.00-1.07 against 6.4-9.6
for genuine translation, i.e. a filter tuned to +v scores like one tuned to -v.
There is no coherent motion to select.

What the data DOES support is temporal-scale selectivity. Our own sweep of
local space-time density (3x3, exact per-event labels) reads as a bandpass:

    tau      250us    1ms    5ms    25ms   104ms
    AUC     0.6305  0.6654  0.7265  0.6359  0.5549

Rain is temporally brief; background is temporally persistent. In wt that is a
clean separation, with wt ~ 0 as the "persistent" band -- which is exactly the
role DC already plays in the spatial bank, so the same partition-of-unity
construction with a protected residual carries over unchanged.

The AUC is jitter-invariant out to 1 ms (0.6918 -> 0.6916), so it is genuine
ms-scale structure and not the timestamp-collision artefact that SPAC also has
(rain averages 323 events per exact nanosecond against background's 8.8).
"""
import math

import torch
import torch.nn as nn

from rsp_model_v2 import RainSubspaceBlock, make_divisible


class TemporalAtomBank(nn.Module):
    """n_t log-Gaussian atoms over |wt|, normalised into a partition of unity.

    Mirrors SteerableAtomBank exactly, one axis down:
      A_j(wt) = exp(-(log|wt| - nu_j)^2 / (2 lam_j^2))
      M_j = A_j / (1 + sum_k A_k),   M_res = 1 / (1 + sum_k A_k)
    so sum_j M_j + M_res == 1 and wt = 0 (the temporally PERSISTENT component)
    always lands in the protected residual, never in a band. Parameters are on
    NORMALISED temporal frequency, so the bank is independent of T.
    """

    def __init__(self, n_t=3, T=16):
        super().__init__()
        self.n_t = n_t
        # spread over the resolvable band: coarse -> fine (persistent -> brief)
        nu = torch.linspace(math.log(0.08), math.log(0.45), n_t)
        self.nu = nn.Parameter(nu)
        self.log_lam = nn.Parameter(torch.full((n_t,), math.log(0.55)))
        self._cache = {}

    def masks(self, T, device, dtype):
        key = (T, str(device), str(dtype))
        if key not in self._cache:
            self._cache[key] = torch.fft.rfftfreq(T, device=device, dtype=dtype).abs()
        w = self._cache[key]                                  # [Tf], w[0] = 0
        lw = torch.log(w.clamp_min(1e-6))                     # DC -> -13.8 -> A ~ 0
        lam = self.log_lam.exp().clamp(0.05, 3.0)
        A = torch.exp(-((lw[None, :] - self.nu[:, None]) ** 2) / (2 * lam[:, None] ** 2))
        A = A * (w[None, :] > 0)                              # hard-protect DC
        den = 1.0 + A.sum(0, keepdim=True)
        return A / den, 1.0 / den                             # [n_t,Tf], [1,Tf]


class TemporalFrontend(nn.Module):
    """[B,T,H,W] binary occupancy -> [B, 1 + n_t + 1, H, W] planes.

    channel 0        : OR over T -- BIT-IDENTICAL to the current 2D input, so the
                       T=1 baseline is an exact special case and the model always
                       has the old input available as an anchor.
    channels 1..n_t  : temporal-band amplitude, sqrt of energy pooled over t
    channel n_t+1    : residual (temporally persistent) band amplitude
    """

    def __init__(self, n_t=3, T=16):
        super().__init__()
        self.bank = TemporalAtomBank(n_t, T)
        self.n_t = n_t
        self.out_chans = 1 + n_t + 1

    def forward(self, x):                                     # x: [B,T,H,W]
        B, T, H, W = x.shape
        anchor = x.amax(1, keepdim=True)                      # OR over sub-windows
        if T == 1:                                            # baseline: no temporal axis
            z = x.new_zeros(B, self.n_t + 1, H, W)
            return torch.cat([anchor, z], 1)

        X = torch.fft.rfft(x.float(), dim=1)                  # [B,Tf,H,W]
        Mb, Mr = self.bank.masks(T, x.device, torch.float32)  # [n_t,Tf], [1,Tf]
        M = torch.cat([Mb, Mr], 0)                            # [n_t+1, Tf]

        Y = torch.fft.irfft(X.unsqueeze(1) * M[None, :, :, None, None],
                            n=T, dim=2)                       # [B,n_t+1,T,H,W]
        amp = Y.pow(2).mean(2).clamp_min(1e-12).sqrt()        # energy pool over t
        return torch.cat([anchor, amp.to(x.dtype)], 1)


class ORSPNet3D(nn.Module):
    """TemporalFrontend -> the unmodified ORSPNet body.

    The spatial bank, the ObliqueGate, the FFN and the subtraction are all
    untouched: this changes what enters the network, not how it computes.
    The global residual is taken against the anchor plane so the model still
    starts from 'predict the input'.
    """

    def __init__(self, T=16, n_t=3, dim=32, num_blocks=4, n_theta=4, n_rad=2,
                 probe=4, out_chans=1, use_temporal=True, use_off=False,
                 use_counts=False, n_extra=0, **kw):
        super().__init__()
        self.use_temporal, self.use_off = use_temporal, use_off
        self.use_counts = use_counts
        self.front = TemporalFrontend(n_t, T) if use_temporal else None
        base = self.front.out_chans if use_temporal else 1
        cin = ((base * 2 if use_off else base) + (2 if use_counts else 0)
               + n_extra)
        self.n_extra = n_extra
        dim = make_divisible(dim)
        self.in_proj = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, dim, 1, bias=True), nn.Hardswish())
        self.blocks = nn.ModuleList([
            RainSubspaceBlock(dim, n_theta, n_rad, probe, **kw)
            for _ in range(num_blocks)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, out_chans, 1, bias=True))

    def forward(self, x, x_off=None, x_cnt=None, x_extra=None):   # [B,T,H,W]
        if self.use_temporal:
            p = self.front(x)
            if self.use_off:
                # OFF events -- absent from the clean GT by construction, and
                # the trailing half of every streak. Same frontend, same bank.
                p = torch.cat([p, self.front(x_off)], 1)
        else:
            p = x.amax(1, keepdim=True)
            if self.use_off:                    # OFF-only ablation: T=1 + OFF
                p = torch.cat([p, x_off.amax(1, keepdim=True)], 1)
        if self.use_counts:
            # per-pixel bin-occupancy counts over the T=16 build planes,
            # normalised to [0,1]. Sum over bins is permutation-invariant, so
            # EXACTLY roll-invariant -- no phase audit needed. Measured probe:
            # c_on AUC 0.2833 (rain occupies FEWER bins -- persistence signal),
            # off/on ratio 0.6838.
            p = torch.cat([p, x_cnt], 1)
        if getattr(self, "n_extra", 0):
            # context / count planes, appended AFTER the anchor channel so the
            # global residual still starts from 'predict the input'
            p = torch.cat([p, x_extra], 1)
        anchor = p[:, :1]
        f = self.in_proj(p)
        for b in self.blocks:
            f = b(f)
        return self.out_proj(f) + anchor


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    D = (1, 8, 32, 64)
    for name, T, kw in [
        ("baseline  T=1  (2D)", 1, dict(use_temporal=False)),
        ("3D log-Gabor T=4",    4, dict(use_temporal=True, n_t=3)),
        ("3D log-Gabor T=8",    8, dict(use_temporal=True, n_t=3)),
        ("3D log-Gabor T=16",  16, dict(use_temporal=True, n_t=3)),
    ]:
        m = ORSPNet3D(T=T, dilations=D, **kw).to(dev).eval()
        n = sum(q.numel() for q in m.parameters())
        x = torch.rand(1, T, 256, 256, device=dev).round()
        with torch.no_grad():
            y = m(x)
            for _ in range(60): m(x)
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(40): m(x)
            if dev == "cuda": torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 40 * 1000
        print(f"{name:22s} {n:7,d}p  {ms:6.2f} ms  out {tuple(y.shape)} "
              f"finite={torch.isfinite(y).all().item()}")
    print("\nreference: ORSPNet+dil 36,782p / 7.13 ms (measured, idle GPU)")


class DFFN3T(nn.Module):
    """The fairness arm: DFFN's body behind the SAME temporal+OFF frontend.

    If the input -- not the architecture -- is the bottleneck, DFFN should jump
    just as EvORSP-3T did. Its own global residual would add the 10-channel
    frontend stack to a 1-channel logit, so the residual is taken against the
    anchor plane instead, exactly as ORSPNet3D does.
    """

    def __init__(self, T=4, n_t=3, dim=32, num_blocks=4, **kw):
        super().__init__()
        import sys
        sys.path.insert(0, "/fs/nexus-scratch/tuxunlu/git/Event-Deraining")
        from model.DynamicFourierFilterNet import DynamicFourierFilterNet
        self.front = TemporalFrontend(n_t, T)
        cin = self.front.out_chans * 2
        self.body = DynamicFourierFilterNet(in_chans=cin, out_chans=1,
                                            dim=dim, num_blocks=num_blocks)

    def forward(self, x, x_off=None):
        p = torch.cat([self.front(x), self.front(x_off)], 1)
        anchor = p[:, :1]
        f = self.body.in_proj(p)
        for b in self.body.blocks:
            f = b(f)
        return self.body.out_proj(f) + anchor
