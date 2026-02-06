import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

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
        x_flat = x.view(B, C, -1).permute(0, 2, 1)

        y1 = self.ssms[0](x_flat)
        y2 = self.ssms[1](x_flat.flip([1])).flip([1])
        
        x_t = x_flat.view(B, H, W, C).transpose(1, 2).reshape(B, -1, C)
        y3 = self.ssms[2](x_t).view(B, W, H, C).transpose(1, 2).reshape(B, -1, C)
        y4 = self.ssms[3](x_t.flip([1])).flip([1]).view(B, W, H, C).transpose(1, 2).reshape(B, -1, C)

        y_all = torch.cat([y1, y2, y3, y4], dim=-1)
        out = self.out_proj(y_all)
        return out.permute(0, 2, 1).view(B, C, H, W)

class VSSBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.ln_mag = LayerNorm2d(dim)
        self.ln_pha = LayerNorm2d(dim)
        self.freq_ssm_mag = MambaSS2D(d_model=dim)
        self.freq_ssm_pha = MambaSS2D(d_model=dim)
        self.skip_scale_freq = nn.Parameter(torch.ones(dim, 1, 1))
        
        self.ln_spatial = nn.LayerNorm(dim)
        self.spatial_ssm = MambaSS2D(d_model=dim)
        self.skip_scale_spatial = nn.Parameter(torch.ones(dim))
        
        self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float()) + 1e-8
        
        mag_xf = torch.abs(xf)
        pha_xf = torch.angle(xf)
        
        h_mag = self.ln_mag(mag_xf)
        h_mag = mag_xf * self.skip_scale_freq + self.freq_ssm_mag(h_mag)
        
        h_pha = self.ln_pha(pha_xf)
        h_pha = pha_xf * self.skip_scale_freq + self.freq_ssm_pha(h_pha)
        
        real_h = h_mag * torch.cos(h_pha)
        imag_h = h_mag * torch.sin(h_pha)
        h_complex = torch.complex(real_h, imag_h) + 1e-8
        
        x_fourier = torch.fft.irfft2(h_complex, s=(H, W), norm='backward') + 1e-8
        
        x_spatial = x.permute(0, 2, 3, 1)
        x_in = self.ln_spatial(x_spatial)
        x_spatial_out = x_spatial * self.skip_scale_spatial + \
                        self.spatial_ssm(x_in.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        x_fourier = x_fourier.permute(0, 2, 3, 1)
        x_final = torch.cat([x_spatial_out, x_fourier], dim=-1)
        x_final = self.linear_out(x_final)
        
        return x_final.permute(0, 3, 1, 2)

class FourierMamba(nn.Module):
    def __init__(self, in_chans=3, dim=48, num_blocks=[2, 2]):
        super().__init__()
        
        # --- FIX: Add Input Projection (Patch Embed) ---
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)
        # -----------------------------------------------

        self.encoder_level1 = nn.ModuleList([
            VSSBlock(dim=dim) for _ in range(num_blocks[0])
        ])
        
        self.down1_2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(dim=dim*2) for _ in range(num_blocks[1])
        ])

    def forward(self, x):
        # x: [B, in_chans, H, W] -> e.g. [1, 1, 128, 128]
        
        # --- FIX: Project to dim ---
        x = self.patch_embed(x)  # Now x is [1, 48, 128, 128]
        # ---------------------------

        out_enc1 = x
        for layer in self.encoder_level1:
            out_enc1 = layer(out_enc1)
            
        inp_enc2 = self.down1_2(out_enc1)
        
        out_enc2 = inp_enc2
        for layer in self.encoder_level2:
            out_enc2 = layer(out_enc2)
            
        return out_enc2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Example input with 1 channel
    img = torch.randn(1, 1, 128, 128).to(device)
    
    # Initialize with in_chans=1
    model = FourierMamba(in_chans=1, dim=48, num_blocks=[4, 6, 6, 8]).to(device)
    
    out = model(img)
    print(f"Input: {img.shape}")
    print(f"Output: {out.shape}")