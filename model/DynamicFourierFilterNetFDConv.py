import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_chans, se_ratio=0.25):
        super().__init__()
        reduced = max(8, int(in_chans * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_chans, reduced, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_chans, kernel_size=1)

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale)
        return x * scale


class DynamicFilterLayer2D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x, dynamic_filters):
        bsz, channels, height, width = x.shape
        x_unfolded = F.unfold(x, kernel_size=self.k, padding=self.pad)
        x_unfolded = x_unfolded.view(bsz, channels, self.k ** 2, height, width)
        filters = dynamic_filters.view(bsz, channels, self.k ** 2, height, width)
        out = torch.sum(x_unfolded * filters, dim=2)
        return out


class FrequencyBandModulator(nn.Module):
    """
    FDConv-inspired filter generator:
    1) Learn a fixed parameter budget in Fourier domain (per channel).
    2) Split it into disjoint frequency bands.
    3) Predict spatially varying per-band modulation maps (FBM).
    4) Reconstruct dynamic 3x3 kernels from band-wise modulated kernels.
    """

    def __init__(self, channels, kernel_size=3, num_bands=3, bottleneck_ratio=0.5):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("This FDConv variant currently assumes kernel_size=3.")
        self.channels = channels
        self.kernel_size = kernel_size
        self.k2 = kernel_size ** 2
        self.num_bands = num_bands

        hidden = max(8, int(channels * bottleneck_ratio))

        # Local content encoder for FBM coefficients.
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.Hardswish(),
        )
        self.mag_head = nn.Conv2d(hidden, channels * num_bands, kernel_size=1, bias=True)
        self.phase_head = nn.Conv2d(hidden, channels * num_bands, kernel_size=1, bias=True)

        # Fixed parameter budget learned in Fourier domain.
        self.mag_fourier_params = nn.Parameter(torch.randn(channels, kernel_size, kernel_size) * 0.02)
        self.phase_fourier_params = nn.Parameter(torch.randn(channels, kernel_size, kernel_size) * 0.02)

        band_masks = self._build_band_masks(kernel_size, num_bands)
        self.register_buffer("band_masks", band_masks, persistent=False)  # [G, K, K]

    @staticmethod
    def _build_band_masks(kernel_size, num_bands):
        # For 3x3 we have three natural wrapped-frequency radii: 0, 1, sqrt(2).
        if kernel_size == 3 and num_bands == 3:
            masks = torch.zeros(num_bands, kernel_size, kernel_size)
            for u in range(kernel_size):
                for v in range(kernel_size):
                    du = min(u, kernel_size - u)
                    dv = min(v, kernel_size - v)
                    r2 = du * du + dv * dv
                    if r2 == 0:
                        masks[0, u, v] = 1.0  # low
                    elif r2 == 1:
                        masks[1, u, v] = 1.0  # mid
                    else:
                        masks[2, u, v] = 1.0  # high
            return masks

        # Generic fallback: split by wrapped-frequency radius quantiles.
        masks = torch.zeros(num_bands, kernel_size, kernel_size)
        radii = []
        for u in range(kernel_size):
            for v in range(kernel_size):
                du = min(u, kernel_size - u)
                dv = min(v, kernel_size - v)
                radii.append((u, v, float((du * du + dv * dv) ** 0.5)))
        sorted_unique = sorted({r for _, _, r in radii})
        # Assign each unique radius to one band index.
        for u, v, r in radii:
            idx = int((sorted_unique.index(r) * num_bands) / max(1, len(sorted_unique)))
            idx = min(num_bands - 1, idx)
            masks[idx, u, v] = 1.0
        return masks

    def _fourier_bands_to_spatial_kernels(self, fourier_params):
        # fourier_params: [C, K, K]
        # band kernels in Fourier domain: [G, C, K, K]
        banded = self.band_masks.unsqueeze(1) * fourier_params.unsqueeze(0)
        # iFFT to spatial kernel bases (real-valued).
        spatial = torch.fft.ifft2(banded.to(torch.complex64), norm="ortho").real
        # [G, C, K^2]
        return spatial.view(self.num_bands, self.channels, self.k2)

    @staticmethod
    def _normalize_signed_kernel(weights, eps=1e-6):
        denom = weights.abs().sum(dim=2, keepdim=True).clamp_min(eps)
        return weights / denom

    def forward(self, fgn_in):
        # fgn_in: [N, 2C, H, W] in frequency-feature map coordinates.
        n, _, h, w = fgn_in.shape
        feat = self.encoder(fgn_in)

        mag_mod = torch.sigmoid(self.mag_head(feat)).view(n, self.num_bands, self.channels, h, w)
        phase_mod = torch.sigmoid(self.phase_head(feat)).view(n, self.num_bands, self.channels, h, w)

        mag_band_kernels = self._fourier_bands_to_spatial_kernels(self.mag_fourier_params)      # [G, C, K^2]
        phase_band_kernels = self._fourier_bands_to_spatial_kernels(self.phase_fourier_params)  # [G, C, K^2]

        # FBM: spatially vary each frequency band and reconstruct per-location kernels.
        mag_weights = torch.einsum("ngchw,gck->nckhw", mag_mod, mag_band_kernels)
        phase_weights = torch.einsum("ngchw,gck->nckhw", phase_mod, phase_band_kernels)

        mag_weights = self._normalize_signed_kernel(mag_weights).reshape(n, -1, h, w)
        phase_weights = self._normalize_signed_kernel(phase_weights).reshape(n, -1, h, w)
        return mag_weights, phase_weights


class DynamicFourierBlockFDConv(nn.Module):
    def __init__(self, dim, kernel_size=3, fgn_bottleneck_ratio=0.5, ffn_expand_ratio=2.0, se_ratio=0.25):
        super().__init__()
        self.dim = dim

        ffn_hidden = make_divisible(dim * ffn_expand_ratio, 8)
        self.norm1 = nn.LayerNorm(dim)

        self.fdconv = FrequencyBandModulator(
            channels=dim,
            kernel_size=kernel_size,
            num_bands=3,
            bottleneck_ratio=fgn_bottleneck_ratio,
        )
        self.dynamic_filter = DynamicFilterLayer2D(kernel_size=kernel_size)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_hidden, kernel_size=1, bias=False),
            nn.Conv2d(ffn_hidden, ffn_hidden, kernel_size=3, padding=1, groups=ffn_hidden, bias=False),
            nn.Hardswish(),
            SqueezeExcite(ffn_hidden, se_ratio=se_ratio),
            nn.Conv2d(ffn_hidden, dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        n, _, h, w = x.shape
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        fft_feat = torch.fft.rfft2(x_norm.float(), norm="ortho")
        mag = torch.abs(fft_feat) + 1e-8
        phase = torch.angle(fft_feat)

        fgn_in = torch.cat([mag, phase], dim=1)
        mag_filters, phase_filters = self.fdconv(fgn_in)

        filtered_mag = self.dynamic_filter(mag, mag_filters)
        filtered_phase = self.dynamic_filter(phase, phase_filters)

        filtered_complex = torch.complex(
            filtered_mag * torch.cos(filtered_phase),
            filtered_mag * torch.sin(filtered_phase),
        )
        fft_out = torch.fft.irfft2(filtered_complex, s=(h, w), norm="ortho")

        x = x + fft_out
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm2)
        return x


class DynamicFourierFilterNetFDConv(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        dim=32,
        num_blocks=4,
        width_mult=1.0,
        fgn_bottleneck_ratio=0.5,
        ffn_expand_ratio=2.0,
        se_ratio=0.25,
    ):
        super().__init__()
        if isinstance(num_blocks, (list, tuple)):
            if len(num_blocks) == 0:
                raise ValueError("num_blocks list/tuple cannot be empty.")
            num_blocks = int(sum(num_blocks))
        else:
            num_blocks = int(num_blocks)
        dim = make_divisible(dim * width_mult, 8)

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1, groups=in_chans, bias=False),
            nn.Conv2d(in_chans, dim, kernel_size=1, bias=True),
            nn.Hardswish(),
        )

        self.blocks = nn.ModuleList(
            [
                DynamicFourierBlockFDConv(
                    dim=dim,
                    kernel_size=3,
                    fgn_bottleneck_ratio=fgn_bottleneck_ratio,
                    ffn_expand_ratio=ffn_expand_ratio,
                    se_ratio=se_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, out_chans, kernel_size=1, bias=True),
        )

    def forward(self, x):
        identity = x
        feat = self.in_proj(x)
        for block in self.blocks:
            feat = block(feat)
        out = self.out_proj(feat)
        return out + identity
