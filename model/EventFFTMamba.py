import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from mamba_ssm import Mamba

# -----------------------------------------------------------------------
# Helper Classes (Provided by User)
# -----------------------------------------------------------------------

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
    spatial_order = zigzag_2d(H, W, device)
    scanA = torch.cat([t * (H * W) + spatial_order for t in range(N)], dim=0)

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
    Acts as the core 'Mamba' processing unit for both Spatial and Frequency branches.
    """
    def __init__(self, N, H, W, d_model: int):
        super().__init__()
        self.N, self.H, self.W = N, H, W
        self.d_model = d_model
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4, but got {d_model}")
        self.d_group = d_model // 4
        self.register_buffer("perms", None, persistent=False)
        self.register_buffer("inv_perms", None, persistent=False)

        self.in_dwconv = nn.Conv1d(4, d_model, kernel_size=1, stride=1, padding=0, groups=4)
        self.act = nn.SiLU()
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=self.d_group) for _ in range(4)
        ])
        self.norm_blocks = nn.ModuleList([
            nn.LayerNorm(self.d_group) for _ in range(4)
        ])
        self.out_dwconv = nn.Conv1d(d_model, 4, kernel_size=1, stride=1, padding=0, groups=4)

    def _init_scans(self, x):
        perms, invs = build_3d_scans(self.N, self.H, self.W, x.device)
        self.perms = torch.stack(perms, dim=0)
        self.inv_perms = torch.stack(invs, dim=0)

    def forward(self, x):
        # x: (B, N, H, W), real
        B, N, H, W = x.shape
        if self.perms is None or self.perms.numel() == 0:
            self._init_scans(x)

        L = N * H * W
        x_flat = x.reshape(B, L)
        seqs = torch.stack([x_flat[:, self.perms[i]] for i in range(4)], dim=-1) # (B, L, 4)
        seqs = seqs.transpose(1, 2) # (B, 4, L)

        seqs = self.in_dwconv(seqs)
        seqs = self.act(seqs)
        seqs = seqs.transpose(1, 2) # (B, L, d_model)
        
        seq_groups = torch.split(seqs, self.d_group, dim=-1)
        processed_groups = []
        for i in range(4):
            g = seq_groups[i]
            g = self.mamba_blocks[i](g)
            g = self.norm_blocks[i](g)
            processed_groups.append(g)
        
        seqs = torch.cat(processed_groups, dim=-1)
        seqs = seqs.transpose(1, 2)
        seqs = self.out_dwconv(seqs)

        recons = []
        for i in range(4):
            path_seq = seqs[:, i, :]
            recon = torch.empty_like(path_seq)
            recon[:, self.inv_perms[i]] = path_seq
            recons.append(recon)

        fused = torch.stack(recons, dim=0).mean(dim=0)
        return fused.view(B, N, H, W)

# -----------------------------------------------------------------------
# Refactored Components to Match Diagram
# -----------------------------------------------------------------------

class FourierSpatialInteractionBlock(nn.Module):
    """
    Corresponds to the 'Fourier Spatial Interaction SSM' (FSIS) diagram.
    Contains parallel Spatial and Fourier branches with a UNet-style internal fusion.
    """
    def __init__(self, N, H, W, d_model: int):
        super().__init__()
        self.norm = ChannelLayerNorm2d(N)

        # --- Spatial Branch (Left side of diagram) ---
        # "Conv 1x1" -> "Spatial Mamba"
        self.spatial_conv = nn.Conv2d(N, N, kernel_size=1)
        self.spatial_mamba = FourScanBranch(N, H, W, d_model=d_model)

        # --- Fourier Branch (Right side of diagram) ---
        # "FFT" -> "Frequency Mamba" -> "IFFT"
        # We use two branches (Amp/Phase) to approximate complex Mamba
        self.freq_amp_mamba = FourScanBranch(N, H, W, d_model=d_model)
        self.freq_phase_mamba = FourScanBranch(N, H, W, d_model=d_model)

        # --- Fusion (Bottom of diagram) ---
        # "Concat & Conv 1x1"
        # Input to concat is (Original_Input + Sum_of_Branches)
        # Actually diagram shows: (Branch_Spa + Branch_Freq) merged, then Concat with Skip
        self.out_conv = nn.Conv2d(2 * N, N, kernel_size=1)

    def forward(self, x):
        # x: (B, N, H, W)
        identity = x
        
        # Shared LayerNorm (Top of diagram)
        x_norm = self.norm(x)

        # ===========================
        # 1. Spatial Branch
        # ===========================
        # Path A: Conv -> Mamba
        x_s = self.spatial_conv(x_norm)
        x_s_mamba = self.spatial_mamba(x_s)
        
        # Path B: SiLU (Gating)
        x_s_gate = F.silu(x_norm)
        
        # Element-wise Multiply
        spatial_out = x_s_mamba * x_s_gate

        # ===========================
        # 2. Fourier Branch
        # ===========================
        # FFT computed INSIDE the block
        x_fft = torch.fft.fft2(x_norm)
        
        # Frequency Mamba (Split Amplitude and Phase)
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        amp_out = self.freq_amp_mamba(amp)
        phase_out = self.freq_phase_mamba(phase)
        
        # IFFT
        fft_rec = torch.polar(amp_out, phase_out)
        x_f_mamba = torch.fft.ifft2(fft_rec).real
        
        # Path B: SiLU (Gating) - reusing the same gate as per standard gating practices
        # or we can recompute if strictly following separate lines. 
        # The diagram splits from LayerNorm separately for both.
        x_f_gate = F.silu(x_norm)
        
        # Element-wise Multiply
        fourier_out = x_f_mamba * x_f_gate

        # ===========================
        # 3. Fusion
        # ===========================
        # Sum branches
        branches_sum = spatial_out + fourier_out
        
        # Concat with original residual (identity) or normalized input?
        # Diagram: Arrow from top loops around to bottom "Concat".
        # Usually this is the residual connection.
        cat = torch.cat([identity, branches_sum], dim=1) # (B, 2N, H, W)
        
        # Conv 1x1
        out = self.out_conv(cat)
        
        return out

class EventFFTMamba(nn.Module):
    """
    Main architecture resembling the top part of the diagram.
    Input Projection -> Encoder -> Bottleneck -> Decoder -> Output Projection
    With Residual Connection from Input Image to Output.
    """
    def __init__(self, N, H, W, d_models):
        super().__init__()
        
        # Input Projection (e.g., expanding channels if needed, but here we keep N)
        self.input_proj = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=1),
            nn.SiLU()
        )

        # Build UNet-like structure
        # Assuming d_models list length is odd: Encoder...Bottleneck...Decoder
        L = len(d_models)
        mid = L // 2
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottleneck = None
        
        # Encoder
        for i in range(mid):
            self.encoder.append(
                FourierSpatialInteractionBlock(N, H, W, d_model=d_models[i])
            )
            
        # Bottleneck
        self.bottleneck = FourierSpatialInteractionBlock(N, H, W, d_model=d_models[mid])
        
        # Decoder
        for i in range(mid + 1, L):
            self.decoder.append(
                FourierSpatialInteractionBlock(N, H, W, d_model=d_models[i])
            )
            
        # Output Projection
        self.output_proj = nn.Conv2d(N, N, kernel_size=1)

    def forward(self, x):
        # x: (B, N, H, W)
        # 1. Global Residual connection
        x = torch.fft.ifft2(x).real          # (B, N, H, W)
        global_identity = x
        
        # 2. Input Projection
        x = self.input_proj(x)
        
        # 3. Encoder
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            
        # 4. Bottleneck
        x = self.bottleneck(x)
        
        # 5. Decoder
        for i, block in enumerate(self.decoder):
            # Retrieve skip connection (LIFO)
            skip_val = skips[-(i+1)]
            # Simple additive skip connection (common in Mamba/ResNets)
            x = x + skip_val 
            x = block(x)
            
        # 6. Output Projection
        x = self.output_proj(x)
        
        # 7. Global Residual Add
        out = x + global_identity
        
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # N is treated as channels in the Conv2d/LayerNorms, 
    # but as sequence length/time in build_3d_scans.
    B, N, H, W = 1, 10, 64, 64 # Reduced size for quick test

    # Input is now just spatial image (Degraded Image)
    spatial_input = torch.randn(B, N, H, W, device=device)

    # d_models for: 3 encoder, 1 bottleneck, 3 decoder
    d_models_config = [64, 128, 256, 512, 256, 128, 64]
    
    model = EventFFTMamba(N, H, W, d_models=d_models_config).to(device)
    
    # Calculate params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    out_spatial = model(spatial_input)

    print("Input shape:", spatial_input.shape)
    print("Output shape:", out_spatial.shape)
    
    # Check if dimensions preserved
    assert out_spatial.shape == spatial_input.shape
    print("Test passed: Dimensions match.")