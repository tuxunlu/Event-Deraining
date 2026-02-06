import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange


# =========================================================================
# Helper Modules
# =========================================================================

class LayerNorm2d(nn.Module):
    """
    Standard LayerNorm for (B, C, H, W) input.
    Normalizes over the channel dimension per spatial location.
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)        # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)        # (B, C, H, W)
        return x


class ChannelGate(nn.Module):
    """
    Simple channel attention gate producing scale in [0,1] per channel.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.gap(x))


# =========================================================================
# 4-way Spatial Mamba with Local Enhancement (MambaIR-style)
# =========================================================================

class MambaSS2D(nn.Module):
    """
    4-way Spatial Scanning using Mamba with:
      - Local depthwise convolution
      - Channel attention
      - Residual connection
    This is used for:
      - Fourier magnitude/phase modeling
      - Spatial branch
      - Multi-scale mapping branches
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_dwconv: bool = True,
        use_ca: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_dwconv = use_dwconv
        self.use_ca = use_ca

        # 4 directional Mambas: forward, backward, vertical, vertical-backward
        self.ssms = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])
        self.out_proj = nn.Linear(d_model * 4, d_model)

        if self.use_dwconv:
            self.ln = LayerNorm2d(d_model)
            self.dwconv = nn.Conv2d(
                d_model, d_model,
                kernel_size=3, stride=1, padding=1,
                groups=d_model
            )

        if self.use_ca:
            self.ca = ChannelGate(d_model, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        b, c, h, w = x.shape
        residual = x

        # Local enhancement
        if self.use_dwconv:
            x = self.dwconv(self.ln(x))     # (B, C, H, W)

        # Flatten spatial dims -> sequence
        x_flat = x.view(b, c, -1).permute(0, 2, 1).contiguous()   # (B, L, C), L = H*W

        # 1. Horizontal forward
        y1 = self.ssms[0](x_flat)

        # 2. Horizontal backward
        y2 = self.ssms[1](x_flat.flip([1])).flip([1])

        # 3. Vertical forward
        x_t = x_flat.view(b, h, w, c).transpose(1, 2).reshape(b, -1, c)  # (B, L, C) with (W * H, C)
        y3 = self.ssms[2](x_t)
        y3 = y3.view(b, w, h, c).transpose(1, 2).reshape(b, -1, c)       # back to (B, H*W, C)

        # 4. Vertical backward
        x_t_back = x_t.flip([1])
        y4 = self.ssms[3](x_t_back).flip([1])
        y4 = y4.view(b, w, h, c).transpose(1, 2).reshape(b, -1, c)

        # Aggregate
        y_all = torch.cat([y1, y2, y3, y4], dim=-1)   # (B, L, 4C)
        y_all = self.out_proj(y_all)                  # (B, L, C)
        y_all = y_all.permute(0, 2, 1).view(b, c, h, w)

        # Channel attention
        if self.use_ca:
            y_all = y_all * self.ca(y_all)

        # Residual
        return residual + y_all


# =========================================================================
# VSSBlock: Fourier + Spatial + Channel Evolution
# =========================================================================

class VSSBlock(nn.Module):
    """
    Visual State Space Block with dual Fourier+Spatial branches:
      1. Fourier Spatial Branch (Mag/Phase via MambaSS2D)
      2. Spatial Branch (MambaSS2D on RGB features)
      3. Channel-wise Fourier evolution via ChannelGate

    This version:
      - Uses a HARD 0/1 magnitude mask with a straight-through estimator
        so the forward uses discrete masks while gradients flow through the
        underlying sigmoid probability.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Fourier Spatial Branch
        self.ln_mag = LayerNorm2d(dim)
        self.ln_pha = LayerNorm2d(dim)
        self.freq_ssm_mag = MambaSS2D(d_model=dim)
        self.freq_ssm_pha = MambaSS2D(d_model=dim)

        # Spatial Branch (token-wise LayerNorm)
        self.ln_spatial = nn.LayerNorm(dim)
        self.spatial_ssm = MambaSS2D(d_model=dim)
        self.skip_scale_spatial = nn.Parameter(torch.ones(dim))

        # Fusion: concat spatial + Fourier -> linear(out_dim = dim)
        self.linear_out = nn.Linear(dim * 2, dim)

        # Channel evolution after fusion
        self.channel_gate = ChannelGate(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ----------------------------------------------------------
        # 1. Fourier Spatial Branch
        # ----------------------------------------------------------
        # rfft2: (B, C, H, W_fft)
        xf = torch.fft.rfft2(x.float(), norm="backward")
        mag_xf = torch.abs(xf) + 1e-8
        pha_xf = torch.atan2(xf.imag + 1e-8, xf.real + 1e-8)

        # Magnitude mask via MambaSS2D -> logits -> sigmoid -> hard mask (STE)
        mag_features = self.freq_ssm_mag(self.ln_mag(mag_xf))     # (B, C, H, W_fft)
        prob_mag = torch.sigmoid(mag_features)

        # Hard 0/1 mask + straight-through estimator
        mag_mask_hard = (prob_mag >= 0.5).float()
        mag_mask = prob_mag + (mag_mask_hard - prob_mag).detach()

        filtered_mag = mag_xf * mag_mask

        # Phase: small bounded residual (tanh)
        pha_features = self.freq_ssm_pha(self.ln_pha(pha_xf))
        pha_res = torch.tanh(pha_features)
        filtered_pha = pha_xf + pha_res

        # Reconstruct complex spectrum
        real_h = filtered_mag * torch.cos(filtered_pha)
        imag_h = filtered_mag * torch.sin(filtered_pha)
        h_complex = torch.complex(real_h, imag_h)

        # Back to spatial domain (same H, W as input)
        x_fourier = torch.fft.irfft2(h_complex, s=(h, w), norm="backward")

        # ----------------------------------------------------------
        # 2. Spatial Branch (RGB space)
        # ----------------------------------------------------------
        # (B, C, H, W) -> (B, H, W, C)
        x_spatial = x.permute(0, 2, 3, 1)
        # LayerNorm along channels
        x_in = self.ln_spatial(x_spatial)          # (B, H, W, C)

        # Back to channels-first for MambaSS2D
        x_in_chfirst = x_in.permute(0, 3, 1, 2)    # (B, C, H, W)
        spatial_out = self.spatial_ssm(x_in_chfirst)

        # Add scaled skip of original spatial tensor
        spatial_out = x_spatial * self.skip_scale_spatial + \
                      spatial_out.permute(0, 2, 3, 1)  # (B, H, W, C)

        # ----------------------------------------------------------
        # 3. Fusion
        # ----------------------------------------------------------
        x_fourier_hw_c = x_fourier.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_concat = torch.cat([spatial_out, x_fourier_hw_c], dim=-1)  # (B, H, W, 2C)

        x_fused = self.linear_out(x_concat)             # (B, H, W, C)
        x_fused = x_fused.permute(0, 3, 1, 2)           # (B, C, H, W)

        # ----------------------------------------------------------
        # 4. Fourier Channel Evolution (global modulation)
        # ----------------------------------------------------------
        channel_scale = self.channel_gate(x_fused)      # (B, C, 1, 1)
        x_final = x_fused * channel_scale

        return x_final


# =========================================================================
# Main Model: FourierMamba2D for Deraining
# =========================================================================

class FourierMamba2DTest(nn.Module):
    """
    U-shaped multi-scale deraining model with Fourier + MambaSS2D blocks.

    Key features:
      - Multi-scale Mamba maps injected into encoder stages.
      - VSSBlock at each encoder/decoder level + latent.
      - Two output heads:
          * clean_head: predicts background (derained) image
          * rain_head:  predicts rain layer
        We then enforce a shallow decomposition:
            pred_clean = clean_head(feat) + (input - pred_rain)
        so that approximately:
            pred_clean + pred_rain ≈ input
    """
    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 3,
        dim: int = 48,
        num_blocks=(2, 2, 2, 2),
    ):
        """
        Args:
            in_chans:  number of input channels (3 for RGB)
            out_chans: number of output channels (usually 3)
            dim:       base channel dimension
            num_blocks: 4-tuple for # of VSSBlocks at each depth:
                        (enc1, enc2, enc3, latent)
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)

        # Multi-Scale Input Mappers (Mamba-based)
        self.map1 = MambaSS2D(d_model=dim)       # Used at H/2
        self.process1 = nn.Linear(dim, dim * 2)

        self.map2 = MambaSS2D(d_model=dim)       # Used at H/4
        self.process2 = nn.Linear(dim, dim * 4)

        self.map3 = MambaSS2D(d_model=dim)       # Used at H/8
        self.process3 = nn.Linear(dim, dim * 8)

        # Encoder
        self.encoder_level1 = nn.ModuleList([VSSBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

        self.encoder_level2 = nn.ModuleList([VSSBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)

        self.encoder_level3 = nn.ModuleList([VSSBlock(dim * 4) for _ in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=2, padding=1)

        # Latent
        self.latent = nn.ModuleList([VSSBlock(dim * 8) for _ in range(num_blocks[3])])

        # Decoder
        self.up4_3 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.reduce_chan_3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)
        self.decoder_level3 = nn.ModuleList([VSSBlock(dim * 4) for _ in range(num_blocks[2])])

        self.up3_2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.reduce_chan_2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder_level2 = nn.ModuleList([VSSBlock(dim * 2) for _ in range(num_blocks[1])])

        self.up2_1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        # Level 1 concatenation is (dim + dim) = 2*dim
        self.decoder_level1 = nn.ModuleList([VSSBlock(dim * 2) for _ in range(num_blocks[0])])

        # Refinement
        self.refinement = nn.ModuleList([VSSBlock(dim * 2) for _ in range(2)])

        # Final dual heads for background and rain
        self.clean_head = nn.Conv2d(dim * 2, out_chans, kernel_size=3, padding=1)
        self.rain_head = nn.Conv2d(dim * 2, out_chans, kernel_size=3, padding=1)

    # ---------------------------------------------------------------------
    # Multi-scale helper
    # ---------------------------------------------------------------------
    def _compute_multi_scale_maps(self, inp_img: torch.Tensor):
        """
        Compute the three multi-scale Mamba maps that are injected into
        enc2, enc3, and latent levels.

        Returns:
            map1: (B, 2*dim, H/2, W/2)
            map2: (B, 4*dim, H/4, W/4)
            map3: (B, 8*dim, H/8, W/8)
        """
        b, c, h, w = inp_img.shape

        # Multi-scale images
        img_h2 = F.interpolate(inp_img, scale_factor=0.5, mode="bilinear", align_corners=False)
        img_h4 = F.interpolate(img_h2, scale_factor=0.5, mode="bilinear", align_corners=False)
        img_h8 = F.interpolate(img_h4, scale_factor=0.5, mode="bilinear", align_corners=False)

        # Map 1 (H/2): (dim -> 2*dim)
        emb_h2 = self.patch_embed(img_h2)                 # (B, dim, H/2, W/2)
        feat_h2 = self.map1(emb_h2)                       # (B, dim, H/2, W/2)
        feat_h2 = rearrange(feat_h2, "b c h w -> b (h w) c")
        feat_h2 = self.process1(feat_h2)                  # (B, (H/2*W/2), 2*dim)
        map1 = rearrange(feat_h2, "b (h w) c -> b c h w",
                         h=img_h2.shape[2], w=img_h2.shape[3])

        # Map 2 (H/4): (dim -> 4*dim)
        emb_h4 = self.patch_embed(img_h4)
        feat_h4 = self.map2(emb_h4)
        feat_h4 = rearrange(feat_h4, "b c h w -> b (h w) c")
        feat_h4 = self.process2(feat_h4)
        map2 = rearrange(feat_h4, "b (h w) c -> b c h w",
                         h=img_h4.shape[2], w=img_h4.shape[3])

        # Map 3 (H/8): (dim -> 8*dim)
        emb_h8 = self.patch_embed(img_h8)
        feat_h8 = self.map3(emb_h8)
        feat_h8 = rearrange(feat_h8, "b c h w -> b (h w) c")
        feat_h8 = self.process3(feat_h8)
        map3 = rearrange(feat_h8, "b (h w) c -> b c h w",
                         h=img_h8.shape[2], w=img_h8.shape[3])

        return map1, map2, map3

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, inp_img: torch.Tensor):
        """
        Args:
            inp_img: Rainy input image, shape (B, in_chans, H, W)

        Returns:
            pred_clean: Derained / background image, (B, out_chans, H, W)
            pred_rain:  Estimated rain layer,         (B, out_chans, H, W)
        """
        b, c, h, w = inp_img.shape

        # 1. Multi-scale Mamba maps
        map1, map2, map3 = self._compute_multi_scale_maps(inp_img)

        # 2. Encoder Level 1
        x = self.patch_embed(inp_img)        # (B, dim, H, W)
        enc1 = x
        for block in self.encoder_level1:
            enc1 = block(enc1)               # (B, dim, H, W)

        # 3. Encoder Level 2
        enc2 = self.down1_2(enc1)            # (B, 2*dim, H/2, W/2)
        enc2 = enc2 + map1                   # inject multi-scale map1
        for block in self.encoder_level2:
            enc2 = block(enc2)               # (B, 2*dim, H/2, W/2)

        # 4. Encoder Level 3
        enc3 = self.down2_3(enc2)            # (B, 4*dim, H/4, W/4)
        enc3 = enc3 + map2                   # inject multi-scale map2
        for block in self.encoder_level3:
            enc3 = block(enc3)               # (B, 4*dim, H/4, W/4)

        # 5. Latent
        latent = self.down3_4(enc3)          # (B, 8*dim, H/8, W/8)
        latent = latent + map3               # inject multi-scale map3
        for block in self.latent:
            latent = block(latent)           # (B, 8*dim, H/8, W/8)

        # 6. Decoder Level 3
        dec3 = self.up4_3(latent)            # (B, 4*dim, H/4, W/4)
        dec3 = torch.cat([dec3, enc3], dim=1)      # (B, 8*dim, H/4, W/4)
        dec3 = self.reduce_chan_3(dec3)            # (B, 4*dim, H/4, W/4)
        for block in self.decoder_level3:
            dec3 = block(dec3)                      # (B, 4*dim, H/4, W/4)

        # 7. Decoder Level 2
        dec2 = self.up3_2(dec3)                     # (B, 2*dim, H/2, W/2)
        dec2 = torch.cat([dec2, enc2], dim=1)       # (B, 4*dim, H/2, W/2)
        dec2 = self.reduce_chan_2(dec2)             # (B, 2*dim, H/2, W/2)
        for block in self.decoder_level2:
            dec2 = block(dec2)                      # (B, 2*dim, H/2, W/2)

        # 8. Decoder Level 1
        dec1 = self.up2_1(dec2)                     # (B, dim, H, W)
        dec1 = torch.cat([dec1, enc1], dim=1)       # (B, 2*dim, H, W)
        for block in self.decoder_level1:
            dec1 = block(dec1)                      # (B, 2*dim, H, W)

        # 9. Refinement
        feat = dec1
        for block in self.refinement:
            feat = block(feat)                      # (B, 2*dim, H, W)

        # 10. Dual-head outputs: clean & rain
        pred_rain_raw = self.rain_head(feat)        # (B, out_chans, H, W)
        pred_clean_base = self.clean_head(feat)     # (B, out_chans, H, W)

        # Decomposition constraint:
        #   input ≈ pred_clean + pred_rain
        #   -> pred_clean = pred_clean_base + (input - pred_rain_raw)
        pred_rain = pred_rain_raw
        pred_clean = pred_clean_base + (inp_img - pred_rain)

        return pred_clean, pred_rain


# =========================================================================
# Simple Sanity Check
# =========================================================================

def verify_deraining_capability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running verification on: {device}")

    # Instantiate model
    model = FourierMamba2D(in_chans=3, out_chans=3, dim=48).to(device)
    model.eval()

    # Dummy "rainy" image
    input_tensor = torch.randn(1, 3, 256, 256, device=device)
    print(f"Input Shape: {input_tensor.shape}")

    try:
        with torch.no_grad():
            pred_clean, pred_rain = model(input_tensor)

        print(f"Clean Output Shape: {pred_clean.shape}")
        print(f"Rain  Output Shape: {pred_rain.shape}")

        # Check 1: shape match
        if pred_clean.shape == input_tensor.shape and pred_rain.shape == input_tensor.shape:
            print("✅ PASS: Clean and rain outputs match input resolution.")
        else:
            print("❌ FAIL: Shape mismatch.")

        # Check 2: NaN detection
        if not (torch.isnan(pred_clean).any() or torch.isnan(pred_rain).any()):
            print("✅ PASS: No NaN values detected.")
        else:
            print("❌ FAIL: Model produced NaN values.")

        # Check 3: parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {params / 1e6:.2f} Million")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_deraining_capability()
