import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from mamba_ssm import Mamba

def center_out_2d(H, W, device):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    cy, cx = H // 2, W // 2
    dist2 = (yy - cy)**2 + (xx - cx)**2
    return torch.argsort(dist2.flatten())

def build_3d_scans(N, H, W, device):
    spatial_order = center_out_2d(H, W, device)

    # Scan A: time-major, spatial center-out
    scanA = torch.cat([t*(H*W) + spatial_order for t in range(N)], dim=0)

    # Scan B: spatial-first, time zigzag
    scanB = []
    fwd = torch.arange(N, device=device)
    bwd = torch.arange(N-1, -1, -1, device=device)
    for i, sidx in enumerate(spatial_order):
        tids = fwd if i % 2 == 0 else bwd
        scanB.append(tids * (H*W) + sidx)
    scanB = torch.cat(scanB, dim=0)

    scanA_rev = torch.flip(scanA, dims=[0])
    scanB_rev = torch.flip(scanB, dims=[0])

    def inv_perm(p):
        inv = torch.empty_like(p)
        inv[p] = torch.arange(p.numel(), device=device)
        return inv

    perms = [scanA, scanB, scanA_rev, scanB_rev]
    invs = [inv_perm(p) for p in perms]
    return perms, invs


class FourScanBranch(nn.Module):
    def __init__(self, N, H, W, kernel_size=3):
        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.register_buffer("perms", None, persistent=False)
        self.register_buffer("inv_perms", None, persistent=False)

        self.dwconv = nn.Conv1d(4, 4, kernel_size, padding=kernel_size//2, groups=4)
        self.act = nn.SiLU()
        self.mamba = Mamba(d_model=4)
        self.norm = nn.LayerNorm(4)

    def _init_scans(self, x):
        perms, invs = build_3d_scans(self.N, self.H, self.W, x.device)
        self.perms = torch.stack(perms, dim=0)
        self.inv_perms = torch.stack(invs, dim=0)

    def forward(self, x):
        # x: (B, N, H, W)
        B, N, H, W = x.shape
        if self.perms is None:
            self._init_scans(x)

        L = N * H * W
        x_flat = x.view(B, L)

        # build 4 sequences
        seqs = torch.stack([x_flat[:, self.perms[i]] for i in range(4)], dim=-1)  # (B, L, 4)

        # DWConv→SiLU→Mamba→LN
        seqs = seqs.transpose(1, 2)          # (B, 4, L)
        seqs = self.dwconv(seqs)
        seqs = seqs.transpose(1, 2)          # (B, L, 4)
        seqs = self.act(seqs)
        seqs = self.mamba(seqs)
        seqs = self.norm(seqs)

        # unscan each channel and fuse
        recons = []
        for i in range(4):
            invp = self.inv_perms[i]         # [L]
            seq_i = seqs[:, :, i]            # (B, L)
            rec = torch.empty_like(seq_i)
            rec[:, invp] = seq_i
            recons.append(rec)

        fused = torch.stack(recons, dim=0).mean(dim=0)  # (B, L)
        return fused.view(B, N, H, W)


class FFTFourScanMambaBlock(nn.Module):
    def __init__(self, N, H, W):
        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.amp_branch = FourScanBranch(N, H, W)
        self.phase_branch = FourScanBranch(N, H, W)

    def forward(self, spatial_in: torch.Tensor, fft_in: torch.Tensor):
        """
        spatial_in: F_l, (B, N, H, W) real
        fft_in:     (B, N, H, W) complex, FFT of F_l
        returns: gated_spatial, new_fft
        """
        B, N, H, W = fft_in.shape

        # decompose fft
        amp   = torch.abs(fft_in)
        phase = torch.angle(fft_in)

        # process amplitude and phase via 4-scan mamba
        amp_p   = self.amp_branch(amp)
        phase_p = self.phase_branch(phase)

        # rebuild complex FFT
        real = amp_p * torch.cos(phase_p)
        imag = amp_p * torch.sin(phase_p)
        fft_rec = torch.complex(real, imag)

        # inverse fft to spatial
        spatial_rec = torch.fft.ifft2(fft_rec).real   # (B, N, H, W)

        # your gate: Ff = IFFT(...) ⊙ SiLU(Fl)
        gate = F.silu(spatial_in)
        spatial_out = spatial_rec * gate

        return spatial_out, fft_rec


class FFTFourScanMambaBackbone(nn.Module):
    def __init__(self, N, H, W, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FFTFourScanMambaBlock(N, H, W) for _ in range(depth)]
        )

    def forward(self, fft_init: torch.Tensor):
        """
        fft_init: (B, N, H, W) complex
        """
        # initial spatial feature F0
        spatial = torch.fft.ifft2(fft_init).real
        fft_cur = fft_init

        for blk in self.blocks:
            # block returns gated update
            spatial_update, fft_cur = blk(spatial, fft_cur)
            # residual between blocks
            spatial = spatial + spatial_update
            # keep fft in sync for next block
            fft_cur = torch.fft.fft2(spatial)

        return spatial


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, H, W = 1, 10, 128, 128

    # pretend we already have N 2D FFTs:
    # we can start from spatial and do fft2 to make a complex input
    spatial_input = torch.randn(B, N, H, W, device=device)
    fft_input = torch.fft.fft2(spatial_input)  # (B, N, H, W), complex

    model = FFTFourScanMambaBackbone(N, H, W, depth=4).to(device)
    out_spatial = model(fft_input)

    print("input fft shape:", fft_input.shape, fft_input.dtype)
    print("output spatial shape:", out_spatial.shape, out_spatial.dtype)