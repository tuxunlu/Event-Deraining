import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from mamba_ssm import Mamba


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)          # back to (B, C, H, W)
        return x


def zigzag_2d(H, W, device):
    order = []
    for s in range(H + W - 1):
        if s % 2 == 0:
            i = min(s, H - 1)
            j = s - i
            while i >= 0 and j < W:
                order.append(i * W + j)
                i -= 1
                j += 1
        else:
            j = min(s, W - 1)
            i = s - j
            while j >= 0 and i < H:
                order.append(i * W + j)
                i += 1
                j -= 1
    return torch.tensor(order, device=device)


def build_3d_scans(N, H, W, device):
    # same as before, just factored
    spatial_order = zigzag_2d(H, W, device)

    # time-major, spatial zigzag
    scanA = torch.cat([t * (H * W) + spatial_order for t in range(N)], dim=0)

    # spatial-first, time zigzag
    scanB = []
    fwd = torch.arange(N, device=device)
    bwd = torch.arange(N - 1, -1, -1, device=device)
    for i, sidx in enumerate(spatial_order):
        tids = fwd if i % 2 == 0 else bwd
        scanB.append(tids * (H * W) + sidx)
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
    """
    This now matches the FourierMamba-ish ordering:
        (4 scans) → 1x1 in-proj → depthwise 1x1 → SiLU → Mamba(d_model) → LN(d_model) → 1x1 out-proj → unscan
    and d_model is configurable per-instance.
    """
    def __init__(self, N, H, W, d_model: int):
        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.d_model = d_model
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4, but got {d_model}")
        self.d_group = d_model // 4  # e.g., if d_model=16, d_group=4
        self.register_buffer("perms", None, persistent=False)
        self.register_buffer("inv_perms", None, persistent=False)

        # 1. Input projection: (B, 4, L) -> (B, d_model, L)
        #    This is 4 independent 1D convs: (in=1, out=d_group)
        #    Scan 0 (ch 0) maps to channels 0..d_group-1
        #    Scan 1 (ch 1) maps to channels d_group..2*d_group-1
        #    ...etc.
        self.in_dwconv = nn.Conv1d(4, d_model, kernel_size=1, stride=1, padding=0, groups=4)

        self.act = nn.SiLU()
        # 2. Four independent Mamba blocks, one for each scan path
        #    Each Mamba block processes d_group channels
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=self.d_group) for _ in range(4)
        ])
        self.norm_blocks = nn.ModuleList([
            nn.LayerNorm(self.d_group) for _ in range(4)
        ])

        self.out_dwconv = nn.Conv1d(d_model, 4, kernel_size=1, stride=1, padding=0, groups=4)

    def _init_scans(self, x):
        perms, invs = build_3d_scans(self.N, self.H, self.W, x.device)
        self.perms = torch.stack(perms, dim=0)       # (4, L)
        self.inv_perms = torch.stack(invs, dim=0)    # (4, L)

    def forward(self, x):
        # x: (B, N, H, W), real
        B, N, H, W = x.shape
        if self.perms is None or self.perms.numel() == 0:
            self._init_scans(x)

        L = N * H * W
        x_flat = x.view(B, L)  # (B, L)

        # 4 differently scanned sequences → stack as channels
        seqs = torch.stack([x_flat[:, self.perms[i]] for i in range(4)], dim=-1)  # (B, L, 4)

        # to conv format
        seqs = seqs.transpose(1, 2)  # (B, 4, L)

        # 1. Project each 1-channel scan to d_group channels
        #    (B, 4, L) -> (B, d_model, L)
        seqs = self.in_dwconv(seqs)
        seqs = self.act(seqs)

        # Mamba expects (B, L, C)
        # (B, d_model, L) -> (B, L, d_model)
        seqs = seqs.transpose(1, 2)
        
        # 2. Split into 4 groups for independent processing
        #    This creates 4 tensors, each of shape (B, L, d_group)
        seq_groups = torch.split(seqs, self.d_group, dim=-1)

        # 3. Process each scan group with its own Mamba and Norm
        processed_groups = []
        for i in range(4):
            g = seq_groups[i]               # (B, L, d_group)
            g = self.mamba_blocks[i](g)     # (B, L, d_group)
            g = self.norm_blocks[i](g)      # (B, L, d_group)
            processed_groups.append(g)
        
        # 4. Concatenate back
        #    (B, L, d_model)
        seqs = torch.cat(processed_groups, dim=-1)

        # 5. Project each d_group-channel path back to a single 1-channel path
        #    Conv1d expects (B, C, L)
        #    (B, L, d_model) -> (B, d_model, L)
        seqs = seqs.transpose(1, 2)
        #    (B, d_model, L) -> (B, 4, L)
        seqs = self.out_dwconv(seqs)
        
        # --- End of Corrected General Fix ---

        # unscan for each of the 4 paths
        recons = []
        for i in range(4):
            # Now, seqs[:, i, :] is the (B, L) tensor for the i-th scan
            path_seq = seqs[:, i, :]  # (B, L) 
            recon = torch.empty_like(path_seq)
            recon[:, self.inv_perms[i]] = path_seq
            recons.append(recon)

        fused = torch.stack(recons, dim=0).mean(dim=0)  # (B, L)
        return fused.view(B, N, H, W)


class FFTFourScanMambaBlock(nn.Module):
    def __init__(self, N, H, W, d_model: int):
        super().__init__()
        self.amp_branch = FourScanBranch(N, H, W, d_model=d_model)
        self.phase_branch = FourScanBranch(N, H, W, d_model=d_model)
        self.spa_ln = ChannelLayerNorm2d(N)
        self.out_ln = ChannelLayerNorm2d(N)

    def forward(self, spatial_in: torch.Tensor, fft_in: torch.Tensor):
        # spatial_in: (B, N, H, W), real
        # fft_in:     (B, N, H, W), complex
        spatial_in = self.spa_ln(spatial_in)

        amplitude = torch.abs(fft_in)
        phase = torch.angle(fft_in)

        amplitude_rec = self.amp_branch(amplitude)
        phase_rec = self.phase_branch(phase)
        fft_rec = torch.polar(amplitude_rec, phase_rec)

        spatial_rec = torch.fft.ifft2(fft_rec).real

        gated = F.silu(spatial_in) * spatial_rec
        residual = self.out_ln(gated)
        return residual, fft_rec


class EventFFTMamba2D(nn.Module):
    """
    Now takes a list of d_models, one per block, e.g. [16, 32, 32, 64]
    to mimic the “progressive” channel growth that PRE-Mamba / vision SSM nets do.
    """
    def __init__(self, N, H, W, d_models):
        super().__init__()
        self.in_ln = ChannelLayerNorm2d(N)
        self.blocks = nn.ModuleList(
            [FFTFourScanMambaBlock(N, H, W, d_model=dm) for dm in d_models]
        )

    def forward(self, fft_init: torch.Tensor):
        spatial0 = torch.fft.ifft2(fft_init).real          # (B, N, H, W)
        x = self.in_ln(spatial0)

        fft_cur = fft_init
        L = len(self.blocks)
        mid = L // 2
        skips = []

        # encoder side
        for i in range(mid):
            dx, fft_cur = self.blocks[i](x, fft_cur)
            x = x + dx
            skips.append(x)

        # middle block if odd
        if L % 2 == 1:
            dx, fft_cur = self.blocks[mid](x, fft_cur)
            x = x + dx
            start_dec = mid + 1
        else:
            start_dec = mid

        # decoder side
        for i in range(start_dec, L):
            dx, fft_cur = self.blocks[i](x, fft_cur)
            pair = L - 1 - i
            if 0 <= pair < len(skips):
                x = x + dx + skips[pair]
            else:
                x = x + dx

        # final skip to original spatial
        x = x + spatial0
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, H, W = 1, 10, 128, 128

    spatial_input = torch.randn(B, N, H, W, device=device)
    fft_input = torch.fft.fft2(spatial_input)

    # pick per-block d_model like FourierMamba/PRE-Mamba stages
    model = EventFFTMamba(N, H, W, d_models=[4, 4, 4, 4, 4, 4, 4, 4]).to(device)
    out_spatial, out_fft = model(fft_input)

    print("input fft shape:", fft_input.shape, fft_input.dtype)
    print("output spatial shape:", out_spatial.shape, out_spatial.dtype)
