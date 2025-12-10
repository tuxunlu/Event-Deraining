import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

# =========================================================================
# Helper Modules (Unchanged)
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
    """ Simplified 4-way Spatial Scanning using mamba_ssm. """
    def __init__(self, d_model):
        super().__init__()
        self.ssms = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(4)
        ])
        self.out_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        y1 = self.ssms[0](x_flat)
        y2 = self.ssms[1](x_flat.flip([1])).flip([1])
        x_t = x_flat.view(B, H, W, C).transpose(1, 2).reshape(B, -1, C)
        y3 = self.ssms[2](x_t).view(B, W, H, C).transpose(1, 2).reshape(B, -1, C)
        y4 = self.ssms[3](x_t.flip([1])).flip([1]).view(B, W, H, C).transpose(1, 2).reshape(B, -1, C)
        y_all = torch.cat([y1, y2, y3, y4], dim=-1)
        out = self.out_proj(y_all)
        return out.permute(0, 2, 1).view(B, C, H, W)

class ChannelGate(nn.Module):
    """ Stable channel attention. """
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

# =========================================================================
# 3. IMPLEMENTATION: Cross-Domain Gated VSSBlock
# =========================================================================

class GatedVSSBlock(nn.Module):
    """
    Improved VSSBlock with Cross-Domain Gated Fusion.
    Instead of simple concatenation, the Frequency branch gates the Spatial branch
    and the Spatial branch gates the Frequency branch.
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dim = dim
        
        # --- Fourier Spatial Branch ---
        self.ln_mag = LayerNorm2d(dim)
        self.ln_pha = LayerNorm2d(dim)
        self.freq_ssm_mag = MambaSS2D(d_model=dim)
        self.freq_ssm_pha = MambaSS2D(d_model=dim)
        
        # --- Spatial Branch ---
        self.ln_spatial = nn.LayerNorm(dim)
        self.spatial_ssm = MambaSS2D(d_model=dim)
        self.skip_scale_spatial = nn.Parameter(torch.ones(dim))
        
        # --- Cross-Domain Gating Gates ---
        # G_s: Gate for Spatial (generated from Freq)
        self.gate_gen_s = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        # G_f: Gate for Freq (generated from Spatial)
        self.gate_gen_f = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        
        # Smooth fusion output
        self.linear_fuse = nn.Linear(dim, dim)
        
        # --- Fourier Channel Branch ---
        self.channel_gate = ChannelGate(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Fourier Branch
        xf = torch.fft.rfft2(x.float()) + 1e-8
        mag_xf = torch.abs(xf)
        pha_xf = torch.atan2(xf.imag+1e-8, xf.real+1e-8)
        
        # Mag Masking
        h_mag = self.freq_ssm_mag(self.ln_mag(mag_xf))
        prob_mag = torch.sigmoid(h_mag)
        mag_mask_hard = (prob_mag >= 0.5).float()
        mag_mask = prob_mag + (mag_mask_hard - prob_mag).detach()
        filtered_mag = mag_xf * mag_mask

        # Phase Residual
        pha_res = torch.tanh(self.freq_ssm_pha(self.ln_pha(pha_xf)))
        filtered_pha = pha_xf + pha_res
        
        # Reconstruct
        real_h = filtered_mag * torch.cos(filtered_pha)
        imag_h = filtered_mag * torch.sin(filtered_pha)
        h_complex = torch.complex(real_h, imag_h) + 1e-8
        x_fourier = torch.fft.irfft2(h_complex, s=(H, W), norm='backward') + 1e-8
        x_fourier_img = x_fourier # (B, C, H, W)
        
        # 2. Spatial Branch
        x_spatial = x.permute(0, 2, 3, 1) # BHW C
        x_in = self.ln_spatial(x_spatial)
        x_spatial_out = x_spatial * self.skip_scale_spatial + \
                        self.spatial_ssm(x_in.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_spatial_img = x_spatial_out.permute(0, 3, 1, 2) # (B, C, H, W)
        
        # 3. Cross-Domain Gated Fusion
        # Generate Gates
        # Gate for Spatial comes from Fourier features
        G_s = self.gate_gen_s(x_fourier_img) 
        # Gate for Fourier comes from Spatial features
        G_f = self.gate_gen_f(x_spatial_img)
        
        # Apply Gates (Bi-Directional Interaction)
        x_spatial_refined = x_spatial_img * G_s
        x_fourier_refined = x_fourier_img * G_f
        
        # Fuse (Additive + Linear)
        x_fused = x_spatial_refined + x_fourier_refined
        x_fused = x_fused.permute(0, 2, 3, 1) # B H W C
        x_fused = self.linear_fuse(x_fused)
        x_fused = x_fused.permute(0, 3, 1, 2) # B C H W
        
        # 4. Channel Evolution
        channel_scale = self.channel_gate(x_fused)
        x_final = x_fused * channel_scale
        
        return x_final

# =========================================================================
# 4. IMPLEMENTATION: Multi-Stage Progressive Components
# =========================================================================

class FourierMambaStage(nn.Module):
    """
    A single stage U-Net based on FourierMamba.
    Args:
        in_chans: Input channels (3 for Stage1, 6 for Stage2 usually)
    """
    def __init__(self, in_chans=3, out_chans=3, dim=48, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)
        
        # Multi-Scale Inputs
        self.map1 = MambaSS2D(d_model=dim)
        self.process1 = nn.Linear(dim, dim*2)
        self.map2 = MambaSS2D(d_model=dim)
        self.process2 = nn.Linear(dim, dim*4)
        self.map3 = MambaSS2D(d_model=dim)
        self.process3 = nn.Linear(dim, dim*8)

        # Encoder
        self.encoder_level1 = nn.ModuleList([GatedVSSBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        self.encoder_level2 = nn.ModuleList([GatedVSSBlock(dim*2) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1)
        self.encoder_level3 = nn.ModuleList([GatedVSSBlock(dim*4) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=2, padding=1)
        
        # Latent
        self.latent = nn.ModuleList([GatedVSSBlock(dim*8) for _ in range(num_blocks[3])])
        
        # Decoder
        self.up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.reduce_chan_3 = nn.Conv2d(dim*8, dim*4, kernel_size=1)
        self.decoder_level3 = nn.ModuleList([GatedVSSBlock(dim*4) for _ in range(num_blocks[2])])
        
        self.up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.reduce_chan_2 = nn.Conv2d(dim*4, dim*2, kernel_size=1)
        self.decoder_level2 = nn.ModuleList([GatedVSSBlock(dim*2) for _ in range(num_blocks[1])])
        
        self.up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.decoder_level1 = nn.ModuleList([GatedVSSBlock(dim*2) for _ in range(num_blocks[0])])
        
        # Refinement
        self.refinement = nn.ModuleList([GatedVSSBlock(dim*2) for _ in range(2)])
        
        self.output = nn.Conv2d(dim*2, out_chans, kernel_size=3, padding=1)

    def forward(self, inp_img):
        # Multi-scale map generation
        img_h2 = F.interpolate(inp_img, scale_factor=0.5)
        img_h4 = F.interpolate(img_h2, scale_factor=0.5)
        img_h8 = F.interpolate(img_h4, scale_factor=0.5)
        
        emb_h2 = self.patch_embed(img_h2)
        feat_h2 = self.process1(rearrange(self.map1(emb_h2), "b c h w -> b (h w) c")) 
        map1 = rearrange(feat_h2, "b (h w) c -> b c h w", h=img_h2.shape[2], w=img_h2.shape[3])
        
        emb_h4 = self.patch_embed(img_h4)
        feat_h4 = self.process2(rearrange(self.map2(emb_h4), "b c h w -> b (h w) c"))
        map2 = rearrange(feat_h4, "b (h w) c -> b c h w", h=img_h4.shape[2], w=img_h4.shape[3])
        
        emb_h8 = self.patch_embed(img_h8)
        feat_h8 = self.process3(rearrange(self.map3(emb_h8), "b c h w -> b (h w) c"))
        map3 = rearrange(feat_h8, "b (h w) c -> b c h w", h=img_h8.shape[2], w=img_h8.shape[3])

        # U-Net Pass
        x = self.patch_embed(inp_img)
        enc1 = x
        for layer in self.encoder_level1: enc1 = layer(enc1)
            
        enc2 = self.down1_2(enc1) + map1
        for layer in self.encoder_level2: enc2 = layer(enc2)
            
        enc3 = self.down2_3(enc2) + map2
        for layer in self.encoder_level3: enc3 = layer(enc3)
            
        latent = self.down3_4(enc3) + map3
        for layer in self.latent: latent = layer(latent)
            
        dec3 = self.up4_3(latent)
        dec3 = self.reduce_chan_3(torch.cat([dec3, enc3], dim=1))
        for layer in self.decoder_level3: dec3 = layer(dec3)
            
        dec2 = self.up3_2(dec3)
        dec2 = self.reduce_chan_2(torch.cat([dec2, enc2], dim=1))
        for layer in self.decoder_level2: dec2 = layer(dec2)
            
        dec1 = self.up2_1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        for layer in self.decoder_level1: dec1 = layer(dec1)
            
        feat_out = dec1
        for layer in self.refinement: feat_out = layer(feat_out)
            
        # Prediction
        out_img = self.output(feat_out) + inp_img[:, :3, :, :] # Residual on first 3 channels
        
        # Return refined features (for SAM) and Image
        return out_img, feat_out

class SAM(nn.Module):
    """
    Supervised Attention Module.
    Takes the refined features from Stage 1 and generates an attention mask
    to guide Stage 2.
    """
    def __init__(self, in_channels, kernel_size=3):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv2d(in_channels, 3, kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, features):
        # Generate Attention Mask from Stage 1 features
        mask = self.sigmoid(self.conv1(features))
        
        # Refine features for passing to Stage 2 (optional, or just use mask for loss)
        refined_feat = features * mask
        
        # Generate a residual image from these features for supervision (Auxiliary Loss)
        aux_img = self.conv_out(refined_feat)
        return aux_img, mask

class ProgressiveFourierMamba(nn.Module):
    """
    Two-Stage Cascaded Architecture.
    Stage 1: Coarse Restoration
    SAM: Attention Guidance
    Stage 2: Fine Restoration (Takes Input + Coarse Output)
    """
    def __init__(self, in_chans=3, out_chans=3, dim=48, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        # Stage 1: Standard Input
        # We pass the custom num_blocks configuration here
        self.stage1 = FourierMambaStage(
            in_chans=in_chans, 
            out_chans=out_chans, 
            dim=dim, 
            num_blocks=num_blocks
        )
        
        # SAM: Processes Stage 1 features (dim*2 because refinement output is dim*2)
        self.sam = SAM(in_channels=dim*2)
        
        # Stage 2: Takes Original Input (3) + Stage 1 Output (3) = 6 channels
        # We also pass the custom num_blocks configuration here
        self.stage2 = FourierMambaStage(
            in_chans=in_chans*2, 
            out_chans=out_chans, 
            dim=dim, 
            num_blocks=num_blocks
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        
        # --- Stage 1 ---
        x_coarse, feat_s1 = self.stage1(x)
        
        # --- SAM ---
        # aux_img can be used for loss: L_charbonnier(aux_img, gt)
        aux_img, attention_mask = self.sam(feat_s1)
        
        # --- Stage 2 ---
        # Concatenate original input and coarse prediction
        # (This is the Input Cascade strategy)
        stage2_input = torch.cat([x, x_coarse], dim=1) 
        
        x_final, _ = self.stage2(stage2_input)
        
        # During training, you would return [x_final, x_coarse, aux_img] to compute multi-stage loss
        if self.training:
            return x_final, x_coarse, aux_img
        else:
            return x_final

# =========================================================================
# Verification with Custom Params
# =========================================================================

def verify_progressive_architecture():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running verification on: {device}")

    # 1. Instantiate the Progressive Model with CUSTOM params
    # Example: deeper network (dim=64) with more blocks in the middle stages ([2, 4, 4, 6])
    custom_dim = 64
    custom_blocks = [2, 4, 4, 6]
    
    print(f"Initializing with dim={custom_dim} and num_blocks={custom_blocks}...")
    
    model = ProgressiveFourierMamba(
        in_chans=3, 
        out_chans=3, 
        dim=custom_dim, 
        num_blocks=custom_blocks
    ).to(device)
    
    # 2. Input
    input_tensor = torch.randn(1, 3, 128, 128).to(device)
    print(f"Input Shape: {input_tensor.shape}")

    # 3. Forward Pass
    try:
        model.eval() # Test inference mode
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output Shape: {output.shape}")

        if input_tensor.shape == output.shape:
            print("✅ PASS: Progressive Restoration Output shape matches.")
        else:
            print("❌ FAIL: Shape mismatch.")
            
        # Check Parameter Count 
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {params / 1e6:.2f} Million")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_progressive_architecture()