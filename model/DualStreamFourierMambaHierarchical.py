import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

# =========================================================================
# Helper Modules (Unchanged)
# =========================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MambaSS2D(nn.Module):
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
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.ln_mag = LayerNorm2d(dim)
        self.ln_pha = LayerNorm2d(dim)
        self.freq_ssm_mag = MambaSS2D(d_model=dim)
        self.freq_ssm_pha = MambaSS2D(d_model=dim)
        self.ln_spatial = nn.LayerNorm(dim)
        self.spatial_ssm = MambaSS2D(d_model=dim)
        self.skip_scale_spatial = nn.Parameter(torch.ones(dim))
        self.linear_out = nn.Linear(dim * 2, dim)
        self.channel_gate = ChannelGate(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Fourier Branch
        xf = torch.fft.rfft2(x.float()) + 1e-8
        mag_xf = torch.abs(xf)
        pha_xf = torch.atan2(xf.imag+1e-8, xf.real+1e-8)
        
        h_mag = self.freq_ssm_mag(self.ln_mag(mag_xf))
  
        mag_mask = torch.sigmoid(h_mag)
        filtered_mag = mag_xf * mag_mask

        # Phase: keep original phase, add small bounded residual
        pha_res = torch.tanh(self.freq_ssm_pha(self.ln_pha(pha_xf)))
        filtered_pha = pha_xf + pha_res


        real_h = filtered_mag * torch.cos(filtered_pha)
        imag_h = filtered_mag * torch.sin(filtered_pha)
        h_complex = torch.complex(real_h, imag_h) + 1e-8
        x_fourier = torch.fft.irfft2(h_complex, s=(H, W), norm='backward') + 1e-8
        
        # Spatial Branch
        x_spatial = x.permute(0, 2, 3, 1)
        x_in = self.ln_spatial(x_spatial)
        x_spatial_out = x_spatial * self.skip_scale_spatial + \
                        self.spatial_ssm(x_in.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # Fusion
        x_fourier = x_fourier.permute(0, 2, 3, 1)
        x_concat = torch.cat([x_spatial_out, x_fourier], dim=-1)
        x_fused = self.linear_out(x_concat).permute(0, 3, 1, 2)
        return x_fused * self.channel_gate(x_fused)

class MaskGuidedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mask_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, bg_feat, rain_feat):
        rain_mask = self.mask_pred(rain_feat)
        clean_confidence = 1.0 - rain_mask
        gated_bg = bg_feat * clean_confidence
        combined = torch.cat([gated_bg, rain_feat], dim=1)
        out = self.fusion(combined)
        return out, rain_mask

# =========================================================================
# DualStreamFourierMamba Model
# =========================================================================

class DualStreamFourierMambaHierarchical(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, dim=48, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        # Standard Patch Embed (Removed EventAdapter)
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)

        # --- Multi-Scale Input Mappers (Hierarchical Injection) ---
        # Level 2 Injection (Input H/2 -> Dim*2)
        self.map1 = MambaSS2D(d_model=dim) 
        self.process1 = nn.Linear(dim, dim*2)
        
        # Level 3 Injection (Input H/4 -> Dim*4)
        self.map2 = MambaSS2D(d_model=dim) 
        self.process2 = nn.Linear(dim, dim*4)
        
        # Latent Injection (Input H/8 -> Dim*8)
        self.map3 = MambaSS2D(d_model=dim) 
        self.process3 = nn.Linear(dim, dim*8)
        
        # --- Shared Encoder ---
        self.encoder_level1 = nn.ModuleList([VSSBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=2, padding=1)
        
        # Latent Bridge
        self.latent = nn.ModuleList([VSSBlock(dim*8) for _ in range(num_blocks[3])])
        
        # --- DUAL DECODER ---
        
        # Stream 1: Background Reconstruction
        self.bg_up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.bg_reduce3 = nn.Conv2d(dim*8, dim*4, kernel_size=1)
        self.bg_dec3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(num_blocks[2])])
        self.fusion3 = MaskGuidedFusion(dim*4)

        self.bg_up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.bg_reduce2 = nn.Conv2d(dim*4, dim*2, kernel_size=1)
        self.bg_dec2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[1])])
        self.fusion2 = MaskGuidedFusion(dim*2)

        self.bg_up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.bg_reduce1 = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.bg_dec1 = nn.ModuleList([VSSBlock(dim) for _ in range(num_blocks[0])])
        
        # Stream 2: Rain Residual
        self.rain_up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.rain_reduce3 = nn.Conv2d(dim*8, dim*4, kernel_size=1)
        self.rain_dec3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(1)])

        self.rain_up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.rain_reduce2 = nn.Conv2d(dim*4, dim*2, kernel_size=1)
        self.rain_dec2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(1)])

        self.rain_up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.rain_reduce1 = nn.Conv2d(dim*2, dim, kernel_size=1)
        self.rain_dec1 = nn.ModuleList([VSSBlock(dim) for _ in range(1)])

        # Heads
        self.out_bg = nn.Conv2d(dim, out_chans, kernel_size=3, padding=1)
        self.out_rain = nn.Conv2d(dim, out_chans, kernel_size=3, padding=1)

    def forward(self, inp_img):
        # inp_img: [B, C, H, W]
        B, C, H, W = inp_img.shape
        
        # --- 1. Generate Hierarchical Scales ---
        img_h2 = F.interpolate(inp_img, scale_factor=0.5)
        img_h4 = F.interpolate(img_h2, scale_factor=0.5)
        img_h8 = F.interpolate(img_h4, scale_factor=0.5)
        
        # --- 2. Process Multi-scale Injection Maps ---
        
        # Map 1 (Input for Level 2)
        emb_h2 = self.patch_embed(img_h2) # B, dim, H/2, W/2
        # MambaSS2D -> Flatten -> Project Dim -> Reshape
        feat_h2 = self.process1(rearrange(self.map1(emb_h2), "b c h w -> b (h w) c")) 
        map1 = rearrange(feat_h2, "b (h w) c -> b c h w", h=img_h2.shape[2], w=img_h2.shape[3])
        
        # Map 2 (Input for Level 3)
        emb_h4 = self.patch_embed(img_h4)
        feat_h4 = self.process2(rearrange(self.map2(emb_h4), "b c h w -> b (h w) c"))
        map2 = rearrange(feat_h4, "b (h w) c -> b c h w", h=img_h4.shape[2], w=img_h4.shape[3])
        
        # Map 3 (Input for Latent)
        emb_h8 = self.patch_embed(img_h8)
        feat_h8 = self.process3(rearrange(self.map3(emb_h8), "b c h w -> b (h w) c"))
        map3 = rearrange(feat_h8, "b (h w) c -> b c h w", h=img_h8.shape[2], w=img_h8.shape[3])

        # --- 3. Encoder (With Injection) ---
        
        # Level 1
        x = self.patch_embed(inp_img)
        enc1 = x
        for l in self.encoder_level1: enc1 = l(enc1)
        
        # Level 2
        enc2 = self.down1_2(enc1)
        enc2 = enc2 + map1 # <--- Hierarchical Injection
        for l in self.encoder_level2: enc2 = l(enc2)
        
        # Level 3
        enc3 = self.down2_3(enc2)
        enc3 = enc3 + map2 # <--- Hierarchical Injection
        for l in self.encoder_level3: enc3 = l(enc3)
        
        # Latent
        latent = self.down3_4(enc3)
        latent = latent + map3 # <--- Hierarchical Injection
        for l in self.latent: latent = l(latent)
        
        # --- 4. Decoder Stage 3 ---
        # Rain Stream
        r3 = self.rain_up4_3(latent)
        r3 = torch.cat([r3, enc3], dim=1)
        r3 = self.rain_reduce3(r3)
        for l in self.rain_dec3: r3 = l(r3)
        
        # Background Stream
        b3 = self.bg_up4_3(latent)
        b3 = torch.cat([b3, enc3], dim=1)
        b3 = self.bg_reduce3(b3)
        b3, mask3 = self.fusion3(b3, r3) # Fusion
        for l in self.bg_dec3: b3 = l(b3)

        # --- 5. Decoder Stage 2 ---
        # Rain Stream
        r2 = self.rain_up3_2(r3)
        r2 = torch.cat([r2, enc2], dim=1)
        r2 = self.rain_reduce2(r2)
        for l in self.rain_dec2: r2 = l(r2)
        
        # Background Stream
        b2 = self.bg_up3_2(b3)
        b2 = torch.cat([b2, enc2], dim=1)
        b2 = self.bg_reduce2(b2)
        b2, mask2 = self.fusion2(b2, r2) # Fusion
        for l in self.bg_dec2: b2 = l(b2)

        # --- 6. Decoder Stage 1 ---
        # Rain Stream
        r1 = self.rain_up2_1(r2)
        r1 = torch.cat([r1, enc1], dim=1) 
        r1 = self.rain_reduce1(r1)
        for l in self.rain_dec1: r1 = l(r1)
        
        # Background Stream
        b1 = self.bg_up2_1(b2)
        b1 = torch.cat([b1, enc1], dim=1) 
        b1 = self.bg_reduce1(b1)
        for l in self.bg_dec1: b1 = l(b1)

        # Outputs
        final_bg = self.out_bg(b1) + inp_img # Residual
        final_rain = self.out_rain(r1)

        return final_bg, final_rain, [mask3, mask2]

# =========================================================================
# Verification
# =========================================================================

def verify_hierarchical_model():
    print("🔹 Initializing Hierarchical Dual-Stream Mamba...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tiny model for verification
    model = DualStreamFourierMamba(
        in_chans=3, 
        out_chans=3, 
        dim=24, 
        num_blocks=[1, 1, 1, 1]
    ).to(device)
    
    B, C, H, W = 2, 3, 128, 128
    x = torch.randn(B, C, H, W).to(device)
    gt_bg = torch.randn(B, C, H, W).to(device)

    # 1. Forward
    bg_pred, rain_pred, masks = model(x)
    print(f"   ✅ Forward Pass Complete.")
    print(f"   ✅ Output Shapes: BG {bg_pred.shape}, Rain {rain_pred.shape}")
    
    # 2. Check Gradient Flow to Hierarchical Inputs
    # If grads flow to model.process3, then the H/8 input injection is working
    loss = nn.MSELoss()(bg_pred, gt_bg)
    loss.backward()
    
    if model.process3.weight.grad is not None:
        print(f"   ✅ Gradient flow verified in Deepest Hierarchical Injection (Level 3).")
    else:
        print(f"   ❌ Gradient flow FAILED for Level 3 Injection.")
        
    if model.process1.weight.grad is not None:
        print(f"   ✅ Gradient flow verified in Shallow Hierarchical Injection (Level 1).")
    
    print("🎉 Verification Successful.")

if __name__ == "__main__":
    verify_hierarchical_model()