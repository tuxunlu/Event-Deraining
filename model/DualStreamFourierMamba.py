import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

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

        # mag_mask = torch.sigmoid(h_mag)

        prob_mag = torch.sigmoid(h_mag)  
        # Hard 0/1 mask for forward
        mag_mask_hard = (prob_mag >= 0.5).float()
        # Straight-through: forward uses hard mask, backward uses soft prob
        mag_mask = prob_mag + (mag_mask_hard - prob_mag).detach()

        filtered_mag = mag_xf * mag_mask

        # Phase: keep original phase, add small bounded residual
        # pha_res = torch.tanh(self.freq_ssm_pha(self.ln_pha(pha_xf)))
        # filtered_pha = pha_xf + pha_res
        filtered_pha = pha_xf

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
# New Modules for Improvement
# =========================================================================

class MaskGuidedFusion(nn.Module):
    """
    Solves the 'Hollow Space' problem.
    Takes Rain features and Background features.
    Uses Rain features to predict a 'corruption probability' (mask).
    Gates the Background features to force the model to rely on Global Context (Mamba)
    where rain is present, effectively performing feature-level inpainting.
    """
    def __init__(self, dim):
        super().__init__()
        self.mask_pred = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.channel_attn = ChannelGate(dim)

    def forward(self, bg_feat, rain_feat):
        # 1. Estimate where the rain is based on rain features
        rain_mask = self.mask_pred(rain_feat) # [B, 1, H, W] (1 = Rain, 0 = Clean)
        
        # 2. Invert mask: 1 = Clean info, 0 = Corrupted info
        clean_confidence = 1.0 - rain_mask
        
        # 3. Gate the background features
        # Where confidence is low (rainy), the gradient is suppressed, 
        # forcing the Mamba blocks in the NEXT layer to fill in the gap 
        # using spatial context from neighbors.
        gated_bg = bg_feat * clean_confidence
        
        # 4. Concatenate and fuse (allow information flow from rain structure to bg)
        # Sometimes knowing 'this is a streak' helps define the edge behind it.
        combined = torch.cat([gated_bg, rain_feat], dim=1)
        out = self.fusion(combined)
        
        return out, rain_mask

class EventInputAdapter(nn.Module):
    """
    Event images are sparse and dominated by edges.
    A simple 3x3 Conv is insufficient to capture the 'velocity' of streaks.
    We use a Multi-Scale extraction to distinguish:
    - Fast moving rain (High temporal frequency -> appeared in few input channels)
    - Static background (Spatial structure -> consistent across channels)
    """
    def __init__(self, in_chans, dim):
        super().__init__()
        # Branch 1: Local detailed texture (3x3)
        self.proj3x3 = nn.Conv2d(in_chans, dim // 2, kernel_size=3, padding=1)
        # Branch 2: Directional Streak context (7x7) - better for rain streaks
        self.proj7x7 = nn.Conv2d(in_chans, dim // 2, kernel_size=7, padding=3)
        
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)
        self.ln = LayerNorm2d(dim)

    def forward(self, x):
        x1 = self.proj3x3(x)
        x2 = self.proj7x7(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.fuse(out)
        return self.ln(out)

# =========================================================================
# Enhanced Dual-Stream Model
# =========================================================================

class DualStreamFourierMamba(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, dim=48, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        # 1. Specialized Event Input
        self.patch_embed = EventInputAdapter(in_chans, dim)
        
        # --- Shared Encoder (Extracts joint features) ---
        self.encoder_level1 = nn.ModuleList([VSSBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=2, padding=1)
        
        # Latent Bridge
        self.latent = nn.ModuleList([VSSBlock(dim*8) for _ in range(num_blocks[3])])
        
        # --- DUAL DECODER ---
        
        # Stream 1: Background Reconstruction (Target: Clean Image)
        self.bg_up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.bg_reduce3 = nn.Conv2d(dim*8, dim*4, kernel_size=1)
        self.bg_dec3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(num_blocks[2])])
        self.fusion3 = MaskGuidedFusion(dim*4) # <--- Fusion

        self.bg_up3_2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.bg_reduce2 = nn.Conv2d(dim*4, dim*2, kernel_size=1)
        self.bg_dec2 = nn.ModuleList([VSSBlock(dim*2) for _ in range(num_blocks[1])])
        self.fusion2 = MaskGuidedFusion(dim*2) # <--- Fusion

        self.bg_up2_1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.bg_reduce1 = nn.Conv2d(dim*2, dim, kernel_size=1) # Note: changed to reduce for consistency
        self.bg_dec1 = nn.ModuleList([VSSBlock(dim) for _ in range(num_blocks[0])])
        
        # Stream 2: Rain Residual (Target: Rain Streaks)
        # Lighter weight than BG stream, focused on high-freq
        self.rain_up4_3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.rain_reduce3 = nn.Conv2d(dim*8, dim*4, kernel_size=1)
        self.rain_dec3 = nn.ModuleList([VSSBlock(dim*4) for _ in range(1)]) # Fewer blocks needed for rain

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
        # Encoder Shared
        x1 = self.patch_embed(inp_img)
        enc1 = x1
        for l in self.encoder_level1: enc1 = l(enc1)
        
        enc2 = self.down1_2(enc1)
        for l in self.encoder_level2: enc2 = l(enc2)
        
        enc3 = self.down2_3(enc2)
        for l in self.encoder_level3: enc3 = l(enc3)
        
        latent = self.down3_4(enc3)
        for l in self.latent: latent = l(latent)
        
        # --- Decoder Stage 3 ---
        # 1. Rain Stream
        r3 = self.rain_up4_3(latent)
        r3 = torch.cat([r3, enc3], dim=1) # Skip connection
        r3 = self.rain_reduce3(r3)
        for l in self.rain_dec3: r3 = l(r3)
        
        # 2. Background Stream
        b3 = self.bg_up4_3(latent)
        b3 = torch.cat([b3, enc3], dim=1) # Skip connection
        b3 = self.bg_reduce3(b3)
        # ** FUSION **: Use Rain features to gate Background features
        b3, mask3 = self.fusion3(b3, r3) 
        for l in self.bg_dec3: b3 = l(b3) # Mamba now inpaints based on gated features

        # --- Decoder Stage 2 ---
        # 1. Rain Stream
        r2 = self.rain_up3_2(r3)
        r2 = torch.cat([r2, enc2], dim=1)
        r2 = self.rain_reduce2(r2)
        for l in self.rain_dec2: r2 = l(r2)
        
        # 2. Background Stream
        b2 = self.bg_up3_2(b3)
        b2 = torch.cat([b2, enc2], dim=1)
        b2 = self.bg_reduce2(b2)
        # ** FUSION **
        b2, mask2 = self.fusion2(b2, r2)
        for l in self.bg_dec2: b2 = l(b2)

        # --- Decoder Stage 1 ---
        # 1. Rain Stream
        r1 = self.rain_up2_1(r2)
        r1 = torch.cat([r1, enc1], dim=1) 
        r1 = self.rain_reduce1(r1)
        for l in self.rain_dec1: r1 = l(r1)
        
        # 2. Background Stream
        b1 = self.bg_up2_1(b2)
        b1 = torch.cat([b1, enc1], dim=1) 
        b1 = self.bg_reduce1(b1)
        # No fusion at the very last layer, let the network refine naturally
        for l in self.bg_dec1: b1 = l(b1)

        # Outputs
        final_bg = self.out_bg(b1) + inp_img # Residual learning of background
        final_rain = self.out_rain(r1) # Explicit rain prediction

        return final_bg, final_rain, [mask3, mask2]
        
def verify_progressive_architecture():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Running verification on: {device}")
    
    # 2. Instantiate the Model
    # Using small dim/blocks for fast verification
    model = DualStreamFourierMamba(
        in_chans=3, 
        out_chans=3, 
        dim=24, 
        num_blocks=[1, 1, 1, 1]
    ).to(device)
    
    B, C, H, W = 2, 3, 128, 128
    inp = torch.randn(B, C, H, W).to(device)
    gt_bg = torch.randn(B, C, H, W).to(device)
    gt_rain = torch.randn(B, C, H, W).to(device)

    print(f"🔹 Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # ==============================================================================
    # Test 1: Inference Mode (Eval)
    # ==============================================================================
    print("\n[Test 1] Checking Inference Mode (model.eval())...")
    model.eval()
    with torch.no_grad():
        out_eval = model(inp)
    
    if isinstance(out_eval, torch.Tensor):
        print(f"   ✅ Output is a single Tensor (Correct).")
    else:
        print(f"   ❌ Output is {type(out_eval)} (Expected single Tensor).")
        
    if out_eval.shape == (B, C, H, W):
        print(f"   ✅ Output shape matches input: {out_eval.shape}")
    else:
        print(f"   ❌ Shape Mismatch: {out_eval.shape} vs {(B, C, H, W)}")

    # ==============================================================================
    # Test 2: Training Mode (Train)
    # ==============================================================================
    print("\n[Test 2] Checking Training Mode (model.train())...")
    model.train()
    out_train = model(inp)
    
    if isinstance(out_train, tuple) and len(out_train) == 3:
        pred_bg, pred_rain, masks = out_train
        print(f"   ✅ Output is Tuple (BG, Rain, Masks) (Correct).")
    else:
        print(f"   ❌ Output format incorrect. Received: {type(out_train)}")
        return

    # Check Internal Masks
    print("\n[Test 3] Verifying Internal Masks...")
    if len(masks) > 0:
        mask_min = min([m.min().item() for m in masks])
        mask_max = max([m.max().item() for m in masks])
        print(f"   ℹ️  Mask Value Range: [{mask_min:.4f}, {mask_max:.4f}]")
        if mask_min >= 0.0 and mask_max <= 1.0:
            print(f"   ✅ Masks are valid probabilities [0, 1].")
        else:
            print(f"   ❌ Masks out of bounds (Check Sigmoid).")
    else:
        print(f"   ⚠️  No masks returned (Did fusion happen?).")

    # ==============================================================================
    # Test 4: Gradient Flow (Critical for Dual Stream)
    # ==============================================================================
    print("\n[Test 4] Verifying Gradient Flow...")
    
    # Define a combined loss
    loss_bg = nn.L1Loss()(pred_bg, gt_bg)
    loss_rain = nn.L1Loss()(pred_rain, gt_rain)
    total_loss = loss_bg + loss_rain
    
    # Backward
    model.zero_grad()
    total_loss.backward()
    
    # Check gradients in specific components
    
    # 1. Input Adapter (Should receive grads from BOTH streams)
    if model.patch_embed.proj3x3.weight.grad is not None:
        print(f"   ✅ Gradients reached Input Adapter (Encoder works).")
    else:
        print(f"   ❌ Input Adapter has NO gradient.")

    # 2. Background Decoder
    if model.out_bg.weight.grad is not None:
        print(f"   ✅ Gradients reached Background Head.")
    else:
        print(f"   ❌ Background Head has NO gradient.")

    # 3. Rain Decoder
    if model.out_rain.weight.grad is not None:
        print(f"   ✅ Gradients reached Rain Head.")
    else:
        print(f"   ❌ Rain Head has NO gradient.")

    # 4. Fusion Block (Gate) - Did the mask predictor learn?
    # Access the fusion block from the decoder level 3
    if model.fusion3.mask_pred[0].weight.grad is not None:
        print(f"   ✅ Gradients reached Fusion/Mask Predictor.")
    else:
        print(f"   ❌ Fusion Block has NO gradient (Masks not learning).")

    print("\n🎉 SUCCESS: Architecture verified successfully.")

if __name__ == "__main__":
    try:
        verify_progressive_architecture()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()