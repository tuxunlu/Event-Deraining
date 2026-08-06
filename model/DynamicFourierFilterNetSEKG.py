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


class FilterGenerationNetworkSEKG(nn.Module):
    """
    Spatially Enhanced Kernel Generation:
    local feature branch + global pooled context gate before final 1x1 head.
    """

    def __init__(self, in_chans, hidden_chans, out_chans):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, bias=False),
            nn.Conv2d(
                hidden_chans,
                hidden_chans,
                kernel_size=3,
                padding=1,
                groups=hidden_chans,
                bias=False,
            ),
            nn.Hardswish(),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_chans, hidden_chans, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.head = nn.Conv2d(hidden_chans, out_chans, kernel_size=1)

    def forward(self, x):
        local_feat = self.local(x)
        context_gate = self.global_context(local_feat)
        enhanced_feat = local_feat * context_gate
        return self.head(enhanced_feat)


class DynamicFourierBlockSEKG(nn.Module):
    def __init__(self, dim, kernel_size=3, fgn_bottleneck_ratio=0.5, ffn_expand_ratio=2.0, se_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.k2 = kernel_size ** 2
        fgn_hidden = max(8, int(dim * fgn_bottleneck_ratio))
        ffn_hidden = make_divisible(dim * ffn_expand_ratio, 8)

        self.norm1 = nn.LayerNorm(dim)
        self.fgn = FilterGenerationNetworkSEKG(
            in_chans=dim * 2,
            hidden_chans=fgn_hidden,
            out_chans=dim * self.k2 * 2,
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
        bsz, channels, height, width = x.shape

        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm).permute(0, 3, 1, 2)

        fft_feat = torch.fft.rfft2(x_norm.float(), norm="ortho")
        mag = torch.abs(fft_feat) + 1e-8
        phase = torch.angle(fft_feat)

        fgn_in = torch.cat([mag, phase], dim=1)
        filters = self.fgn(fgn_in)

        mag_filters, phase_filters = torch.chunk(filters, 2, dim=1)
        height_f, width_f = mag.shape[-2:]
        mag_filters = F.softmax(
            mag_filters.view(bsz, channels, self.k2, height_f, width_f), dim=2
        ).view(bsz, -1, height_f, width_f)
        phase_filters = F.softmax(
            phase_filters.view(bsz, channels, self.k2, height_f, width_f), dim=2
        ).view(bsz, -1, height_f, width_f)

        filtered_mag = self.dynamic_filter(mag, mag_filters)
        filtered_phase = self.dynamic_filter(phase, phase_filters)

        filtered_complex = torch.complex(
            filtered_mag * torch.cos(filtered_phase),
            filtered_mag * torch.sin(filtered_phase),
        )
        fft_out = torch.fft.irfft2(filtered_complex, s=(height, width), norm="ortho")

        x = x + fft_out
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm2)
        return x


class DynamicFourierFilterNetSEKG(nn.Module):
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
                DynamicFourierBlockSEKG(
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
