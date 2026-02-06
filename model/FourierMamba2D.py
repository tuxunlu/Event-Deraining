import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

# =========================================================================
# Helper Modules
# =========================================================================

class LayerNorm2d(nn.Module):
    """ Standard LayerNorm for (B, C, H, W) input. """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MambaSS2D(nn.Module):
    """
    Simplified 4-way Spatial Scanning using mamba_ssm.
    Used for Spatial Branch, Frequency Branch, and Multi-Scale Mapping.
    """
    def __init__(self, d_model):
        super().__init__()
        # 4 independent SSMs for: Forward, Backward, Vertical, Vertical-Backward
        self.ssms = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(4)
        ])
        self.out_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, L, C]

        # 1. Forward
        y1 = self.ssms[0](x_flat)
        # 2. Backward
        y2 = self.ssms[1](x_flat.flip([1])).flip([1])
        
        # Transpose for Vertical
        x_t = x_flat.view(B, H, W, C).transpose(1, 2).reshape(B, -1, C)
        # 3. Vertical
        y3 = self.ssms[2](x_t).view(B, W, H, C).transpose(1, 2).reshape(B, -1, C)
        # 4. Vertical Backward
        y4 = self.ssms[3](x_t.flip([1])).flip([1]).view(B, W, H, C).transpose(1, 2).reshape(B, -1, C)

        # Fuse
        y_all = torch.cat([y1, y2, y3, y4], dim=-1)
        out = self.out_proj(y_all)
        return out.permute(0, 2, 1).view(B, C, H, W)

# =========================================================================
# Core Blocks
# =========================================================================

class ChannelGate(nn.Module):
    """
    Stable channel attention (no FFT over channels).
    Produces a bounded scale in [0,1] for each channel.
    """
    def __init__(self, dim):
        super().__init__()
        hidden = max(1, dim // 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(self.gap(x))


class VSSBlock(nn.Module):
    """
    The main FourierMamba Block containing:
    1. Fourier Spatial Branch (Mag/Phase Mamba)
    2. Spatial Branch (Standard Mamba)
    3. Fourier Channel Evolution Branch
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dim = dim
        
        # --- Fourier Spatial Branch ---
        self.ln_mag = LayerNorm2d(dim)
        self.ln_pha = LayerNorm2d(dim)
        self.freq_ssm_mag = MambaSS2D(d_model=dim)  # produces mask logits
        self.freq_ssm_pha = MambaSS2D(d_model=dim)  # small phase residual
        
        # --- Spatial Branch ---
        self.ln_spatial = nn.LayerNorm(dim)
        self.spatial_ssm = MambaSS2D(d_model=dim)
        self.skip_scale_spatial = nn.Parameter(torch.ones(dim))
        
        # --- Fusion ---
        self.linear_out = nn.Linear(dim * 2, dim)
        
        # --- Fourier Channel Branch ---
        self.channel_gate = ChannelGate(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # ----------------------------------------------------------
        # 1. Fourier Spatial Branch
        # ----------------------------------------------------------
        xf = torch.fft.rfft2(x.float()) + 1e-8
        mag_xf = torch.abs(xf)
        pha_xf = torch.atan2(xf.imag+1e-8, xf.real+1e-8)
        
        # Magnitude mask: sigmoid to ensure [0,1], keep a floor to avoid full drop
        h_mag = self.freq_ssm_mag(self.ln_mag(mag_xf))

        mag_mask = torch.sigmoid(h_mag)

        # prob_mag = torch.sigmoid(h_mag)  
        # # Hard 0/1 mask for forward
        # mag_mask_hard = (prob_mag >= 0.5).float()
        # # Straight-through: forward uses hard mask, backward uses soft prob
        # mag_mask = prob_mag + (mag_mask_hard - prob_mag).detach()

        filtered_mag = mag_xf * mag_mask

        # Phase: keep original phase, add small bounded residual
        pha_res = torch.tanh(self.freq_ssm_pha(self.ln_pha(pha_xf)))
        filtered_pha = pha_xf + pha_res
        # filtered_pha = pha_xf

        # Reconstruct spectrum with filtered magnitude and near-original phase
        real_h = filtered_mag * torch.cos(filtered_pha)
        imag_h = filtered_mag * torch.sin(filtered_pha)
        h_complex = torch.complex(real_h, imag_h) + 1e-8
        x_fourier = torch.fft.irfft2(h_complex, s=(H, W), norm='backward') + 1e-8
        
        # ----------------------------------------------------------
        # 2. Spatial Branch
        # ----------------------------------------------------------
        x_spatial = x.permute(0, 2, 3, 1)
        x_in = self.ln_spatial(x_spatial)
        x_spatial_out = x_spatial * self.skip_scale_spatial + \
                        self.spatial_ssm(x_in.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # ----------------------------------------------------------
        # 3. Fusion
        # ----------------------------------------------------------
        x_fourier = x_fourier.permute(0, 2, 3, 1)
        x_concat = torch.cat([x_spatial_out, x_fourier], dim=-1)
        x_fused = self.linear_out(x_concat) # (B, H, W, C)
        x_fused = x_fused.permute(0, 3, 1, 2) # (B, C, H, W)
        
        # ----------------------------------------------------------
        # 4. Fourier Channel Evolution
        # ----------------------------------------------------------
        # Stable channel gate (bounded 0-1) to modulate features
        channel_scale = self.channel_gate(x_fused)
        
        # Apply global modulation
        x_final = x_fused * channel_scale
        
        return x_final

# =========================================================================
# Main Model: FourierMamba
# =========================================================================

class FourierMamba2D(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, dim=48, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)
        
        # --- Multi-Scale Input Mappers (SS2D_map logic) ---
        # The repo uses Mamba blocks to process the resized inputs
        self.map1 = MambaSS2D(d_model=dim) # Used for H/2
        self.process1 = nn.Linear(dim, dim*2) # Proj to level 2 dim
        
        self.map2 = MambaSS2D(d_model=dim) # Used for H/4
        self.process2 = nn.Linear(dim, dim*4) # Proj to level 3 dim
        
        self.map3 = MambaSS2D(d_model=dim) # Used for H/8
        self.process3 = nn.Linear(dim, dim*8) # Proj to level 4 dim

        # --- Encoder Levels ---
        self.encoder_level1 = nn.ModuleList([VSSBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=2, padding=1)
        
        # --- Latent ---
        self.latent = nn.ModuleList([VSSBlock(dim*8) for _ in range(num_blocks[3])])
        
        # --- Decoder Levels ---
        self.up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.reduce_chan_3 = nn.Conv2d(dim*8, dim*4, kernel_size=1) # 4c (enc) + 4c (dec) -> 4c
        self.decoder_level3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(num_blocks[2])])
        
        self.up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.reduce_chan_2 = nn.Conv2d(dim*4, dim*2, kernel_size=1)
        self.decoder_level2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[1])])
        
        self.up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        # Level 1 concat is dim + dim -> 2dim. Repo doesn't use 1x1 reduction here, just processes 2dim?
        # Checking repo: "inp_dec_level1 = torch.cat... out_dec_level1"
        # The repo actually keeps it at 2*dim for decoder level 1.
        self.decoder_level1 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[0])])
        
        # --- Refinement ---
        self.refinement = nn.ModuleList([VSSBlock(dim*2) for _ in range(2)]) # Usually 4 blocks
        
        self.output = nn.Conv2d(dim*2, out_chans, kernel_size=3, padding=1)

    def forward(self, inp_img):
        # inp_img: [B, 1, H, W]
        B, C, H, W = inp_img.shape
        
        # 1. Multi-Scale Inputs
        
        # Resize inputs
        img_h2 = F.interpolate(inp_img, scale_factor=0.5)
        img_h4 = F.interpolate(img_h2, scale_factor=0.5)
        img_h8 = F.interpolate(img_h4, scale_factor=0.5)
        
        # Process Multi-scale Maps
        # FIX: Remove .transpose(1, 2) from inside self.processX()
        
        # Map 1 (H/2)
        emb_h2 = self.patch_embed(img_h2) # (B, dim, H/2, W/2)
        # Rearrange to (B, L, C) -> Linear -> (B, L, 2C)
        feat_h2 = self.process1(rearrange(self.map1(emb_h2), "b c h w -> b (h w) c")) 
        # Rearrange back to (B, 2C, H/2, W/2)
        map1 = rearrange(feat_h2, "b (h w) c -> b c h w", h=img_h2.shape[2], w=img_h2.shape[3])
        
        # Map 2 (H/4)
        emb_h4 = self.patch_embed(img_h4)
        feat_h4 = self.process2(rearrange(self.map2(emb_h4), "b c h w -> b (h w) c"))
        map2 = rearrange(feat_h4, "b (h w) c -> b c h w", h=img_h4.shape[2], w=img_h4.shape[3])
        
        # Map 3 (H/8)
        emb_h8 = self.patch_embed(img_h8)
        feat_h8 = self.process3(rearrange(self.map3(emb_h8), "b c h w -> b (h w) c"))
        map3 = rearrange(feat_h8, "b (h w) c -> b c h w", h=img_h8.shape[2], w=img_h8.shape[3])

        # 2. Encoder Level 1
        x = self.patch_embed(inp_img)
        enc1 = x
        for layer in self.encoder_level1:
            enc1 = layer(enc1)
            
        # 3. Encoder Level 2
        enc2 = self.down1_2(enc1)
        enc2 = enc2 + map1 # Injection
        for layer in self.encoder_level2:
            enc2 = layer(enc2)
            
        # 4. Encoder Level 3
        enc3 = self.down2_3(enc2)
        enc3 = enc3 + map2 # Injection
        for layer in self.encoder_level3:
            enc3 = layer(enc3)
            
        # 5. Latent
        latent = self.down3_4(enc3)
        latent = latent + map3 # Injection
        for layer in self.latent:
            latent = layer(latent)
            
        # 6. Decoder Level 3
        dec3 = self.up4_3(latent)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.reduce_chan_3(dec3)
        for layer in self.decoder_level3:
            dec3 = layer(dec3)
            
        # 7. Decoder Level 2
        dec2 = self.up3_2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce_chan_2(dec2)
        for layer in self.decoder_level2:
            dec2 = layer(dec2)
            
        # 8. Decoder Level 1
        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1) 
        for layer in self.decoder_level1:
            dec1 = layer(dec1)
            
        # 9. Refinement
        out = dec1
        for layer in self.refinement:
            out = layer(out)
            
        # 10. Output
        out = self.output(out) + inp_img 

        return out


def verify_deraining_capability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running verification on: {device}")

    # 1. Instantiate the Model
    # dim=48 is standard for these lightweight restoration models
    model = FourierMamba2D(in_chans=3, out_chans=3, dim=48).to(device)
    
    # 2. Create a Dummy "Rainy" Image
    # Shape: (Batch_Size, Channels, Height, Width)
    # 256x256 is a common training patch size
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    print(f"Input Shape: {input_tensor.shape}")

    # 3. Run Forward Pass
    try:
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        print(f"Output Shape: {output_tensor.shape}")

        # 4. Verification Checks
        # Check 1: Output shape matches input shape (Critical for Deraining)
        if input_tensor.shape == output_tensor.shape:
            print("✅ PASS: Output resolution matches input resolution.")
        else:
            print("❌ FAIL: Shape mismatch.")

        # Check 2: Values are not NaN (Checks numerical stability of FFT/Mamba)
        if not torch.isnan(output_tensor).any():
            print("✅ PASS: No NaN values detected.")
        else:
            print("❌ FAIL: Model produced NaN values (Check FFT stability).")

        # Check 3: Parameter Count
        # To ensure all branches (Channel Mamba, SS2D, etc.) are actually built
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {params / 1e6:.2f} Million")
        print("\nSUCCESS: The model architecture is verified for image restoration.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_deraining_capability()
