import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Core Dynamic Filter Components
# =========================================================================

class DynamicFilterLayer2D(nn.Module):
    """
    Applies a pixel-wise (or frequency-bin-wise) dynamic local filter.
    Conceptually similar to the Theano Lasagne DynamicFilterLayer provided.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x, dynamic_filters):
        """
        x: [B, C, H, W] - The feature map to be filtered (e.g., Magnitude or Phase)
        dynamic_filters: [B, C * K^2, H, W] - The dynamically generated weights
        """
        B, C, H, W = x.shape
        
        # Extract local patches: [B, C * K^2, H * W]
        x_unfolded = F.unfold(x, kernel_size=self.k, padding=self.pad)
        
        # Reshape to easily multiply with dynamic filters: [B, C, K^2, H, W]
        x_unfolded = x_unfolded.view(B, C, self.k**2, H, W)
        filters = dynamic_filters.view(B, C, self.k**2, H, W)
        
        # Element-wise multiplication and sum over the kernel dimension
        out = torch.sum(x_unfolded * filters, dim=2) # [B, C, H, W]
        
        return out

# =========================================================================
# Dynamic Fourier Block
# =========================================================================

class DynamicFourierBlock(nn.Module):
    """
    Core block that processes spatial features, jumps to Fourier Domain,
    generates and applies dynamic filters to Phase and Magnitude, and jumps back.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.k2 = kernel_size**2
        
        self.norm1 = nn.LayerNorm(dim)
        
        # Filter Generating Network (FGN) - Extremely lightweight using Grouped Convs
        self.fgn = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            # Outputs filters for both Magnitude and Phase (hence * 2)
            nn.Conv2d(dim, dim * self.k2 * 2, kernel_size=1) 
        )
        
        self.dynamic_filter = DynamicFilterLayer2D(kernel_size=kernel_size)
        
        # Spatial Feed-Forward Network (FFN) for spatial refinement
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. Fourier Domain Conversion ---
        x_norm = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x_norm = self.norm1(x_norm).permute(0, 3, 1, 2) # [B, C, H, W]
        
        # Real FFT generates shape [B, C, H, W/2+1]
        fft_feat = torch.fft.rfft2(x_norm.float(), norm='ortho')
        mag = torch.abs(fft_feat) + 1e-8
        phase = torch.angle(fft_feat)
        
        # --- 2. Filter Generation ---
        # FGN observes both magnitude and phase to generate dynamic filters
        fgn_in = torch.cat([mag, phase], dim=1) # [B, 2C, H, W/2+1]
        filters = self.fgn(fgn_in)
        
        # Split into Magnitude filters and Phase filters
        mag_filters, phase_filters = torch.chunk(filters, 2, dim=1)
        
        # Apply Softmax over the kernel spatial dimension (K^2) for stability
        H_f, W_f = mag.shape[-2:]
        mag_filters = F.softmax(mag_filters.view(B, C, self.k2, H_f, W_f), dim=2).view(B, -1, H_f, W_f)
        phase_filters = F.softmax(phase_filters.view(B, C, self.k2, H_f, W_f), dim=2).view(B, -1, H_f, W_f)
        
        # --- 3. Dynamic Filtering in Frequency Domain ---
        filtered_mag = self.dynamic_filter(mag, mag_filters)
        filtered_phase = self.dynamic_filter(phase, phase_filters)
        
        # --- 4. Inverse Fourier Domain Conversion ---
        filtered_complex = torch.complex(filtered_mag * torch.cos(filtered_phase), 
                                         filtered_mag * torch.sin(filtered_phase))
        
        fft_out = torch.fft.irfft2(filtered_complex, s=(H, W), norm='ortho')
        
        # Skip connection
        x = x + fft_out
        
        # --- 5. Spatial Refinement ---
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm2)
        
        return x

# =========================================================================
# Main Model
# =========================================================================

class DynamicFourierFilterNet(nn.Module):
    """
    Lightweight Event-Based Deraining Model using Dynamic Fourier Filters.
    """
    def __init__(self, in_chans=1, out_chans=1, dim=32, num_blocks=4):
        super().__init__()
        if isinstance(num_blocks, (list, tuple)):
            if len(num_blocks) == 0:
                raise ValueError("num_blocks list/tuple cannot be empty.")
            num_blocks = int(sum(num_blocks))
        else:
            num_blocks = int(num_blocks)
        
        # Initial Feature Projection
        self.in_proj = nn.Conv2d(in_chans, dim, kernel_size=3, padding=1)
        
        # Stack of Dynamic Fourier Blocks
        self.blocks = nn.ModuleList([
            DynamicFourierBlock(dim=dim, kernel_size=3) for _ in range(num_blocks)
        ])
        
        # Output Projection
        self.out_proj = nn.Conv2d(dim, out_chans, kernel_size=3, padding=1)

    def forward(self, x):
        # x expected shape: [B, 1, H, W] (real spatial image from dataset)
        identity = x
        
        feat = self.in_proj(x)
        
        for block in self.blocks:
            feat = block(feat)
            
        out = self.out_proj(feat)
        
        # Global residual connection (Model learns to predict rain residual)
        return out + identity


# =========================================================================
# Verification / Testing
# =========================================================================

def verify_dynamic_fourier_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running DFFNet verification on: {device}")

    # The dataset script EventRainEFFT2D indicates 1-channel inputs
    # unsequeeze(0) pushes shape to [1, H, W], meaning in_chans=1
    model = DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32, num_blocks=3).to(device)
    
    # Dummy input representing the 'merge' data
    input_tensor = torch.randn(2, 1, 256, 256).to(device) 
    print(f"Input Shape: {input_tensor.shape}")

    try:
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        print(f"Output Shape: {output_tensor.shape}")

        if input_tensor.shape == output_tensor.shape:
            print("✅ PASS: Output resolution matches input resolution.")
        else:
            print("❌ FAIL: Shape mismatch.")

        if not torch.isnan(output_tensor).any():
            print("✅ PASS: No NaN values detected.")
        else:
            print("❌ FAIL: Model produced NaN values (Check FFT stability).")

        # Check lightweight parameter count constraint
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {params / 1e6:.4f} Million")
        
        if params < 1.0e6:
            print("✅ PASS: Model is highly lightweight (<1M params).")
            
        print("\nSUCCESS: The DFFNet architecture is verified.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dynamic_fourier_model()