"""Every earlier architecture behind the SAME polarity-complete frontend.

The fairness question, now asked at event level with the corrected target: give
DFFN, ORSPNet, StreakNet and FourierMamba2D exactly the input EvORSP-3T/E gets
(T=4 temporal sub-windows x {ON, OFF} = 10 planes) and exactly the same
supervision (count-majority per cell) and readout (out_chans = T_out), then
score them all with PRE-Mamba's event-level metric. Whatever differences remain
are architecture, and nothing else.

Each body carries an internal global residual `out + x` that assumes
in_chans == out_chans, so feeding 10 planes and emitting T_out breaks it. Two
routes, both preserving each architecture's actual computation:
  - DFFN / ORSPNet / StreakNet expose in_proj -> blocks -> out_proj, so the
    body is run stage-wise and the residual is taken against the anchor plane
    instead (exactly what DFFN3T and ORSPNet3D already do).
  - FourierMamba2D is a multi-scale U-Net with no such decomposition, so it
    runs unmodified at in_chans = out_chans = 10 (its own residual intact) and
    a 1x1 conv maps 10 -> T_out afterwards.
"""

import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_d, _os.path.dirname(_d)]
import config as C
C.bootstrap()
import sys

import torch
import torch.nn as nn

from rsp_3d import TemporalFrontend


class FrontendBody(nn.Module):
    def __init__(self, kind, T=4, n_t=3, t_out=16, dim=32):
        super().__init__()
        self.kind = kind
        self.front = TemporalFrontend(n_t, T)
        cin = self.front.out_chans * 2                    # ON planes + OFF planes
        if kind == "dffn":
            from model.DynamicFourierFilterNet import DynamicFourierFilterNet
            self.body = DynamicFourierFilterNet(in_chans=cin, out_chans=t_out,
                                                dim=dim, num_blocks=4)
        elif kind == "orsp":
            from rsp_model_v2 import ORSPNet
            self.body = ORSPNet(in_chans=cin, out_chans=t_out, dim=dim,
                                num_blocks=4, dilations=(1, 8, 32, 64))
        elif kind == "streaknet":
            from rsp_streak import StreakNet
            self.body = StreakNet(in_chans=cin, out_chans=t_out, dim=dim,
                                  num_blocks=4, K=127, use_strip=True,
                                  use_rate=True, use_darkmask=True)
        elif kind == "fmamba":
            from model.FourierMamba2D import FourierMamba2D
            self.body = FourierMamba2D(in_chans=cin, out_chans=cin, dim=dim,
                                       num_blocks=[2, 2, 2, 2])
            self.head = nn.Conv2d(cin, t_out, 1)
        else:
            raise ValueError(kind)

    def forward(self, x, x_off=None):
        p = torch.cat([self.front(x), self.front(x_off)], 1)
        anchor = p[:, :1]
        if self.kind == "fmamba":
            return self.head(self.body(p)) + anchor
        b = self.body
        if self.kind == "streaknet":
            rate = b._rate(p) if b.use_rate else None
            f = b.in_proj(p)
            for blk in b.blocks:
                f = blk(f, rate) if b.strip else blk(f)
        else:
            f = b.in_proj(p)
            for blk in b.blocks:
                f = blk(f)
        return b.out_proj(f) + anchor


if __name__ == "__main__":
    dev = "cuda"
    x = torch.rand(1, 4, 128, 128, device=dev).round()
    o = torch.rand(1, 4, 128, 128, device=dev).round()
    for kind in ("dffn", "orsp", "streaknet", "fmamba"):
        try:
            m = FrontendBody(kind, t_out=16).to(dev).eval()
            with torch.no_grad():
                y = m(x, x_off=o)
            print(f"  {kind:10s} OK  out {tuple(y.shape)}  "
                  f"{sum(p.numel() for p in m.parameters()):,} params  "
                  f"finite={torch.isfinite(y).all().item()}")
        except Exception as e:
            print(f"  {kind:10s} FAIL {type(e).__name__}: {str(e)[:100]}")
