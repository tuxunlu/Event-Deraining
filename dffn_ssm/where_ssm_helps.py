"""
DECISION EXPERIMENT.

A per-pixel spatially varying gain beat every frequency-domain operator.
So the question that decides the architecture is no longer "how do we filter the
spectrum better" but: DOES PREDICTING THAT GAIN NEED GLOBAL CONTEXT?
If yes, an SSM is the cheap way to get it. If no, an SSM buys nothing.

All variants predict a per-pixel gain m(x,y) and output img*(1+m).
They differ ONLY in the receptive field of the predictor:
  local        3x3 depthwise            (RF ~ 5 px)     <- what the current FGN has
  dilated      3x3 dw, dilation 1/4/16  (RF ~ 70 px)    control: plain big RF
  globalpool   local + global avg pool   (RF = whole image, but 1 number)
  scan         local + 4-direction fused selective scan (RF = whole image, positional)
"""
import glob, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

ROOT = "/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/efft_results"
DEV = "cuda"; H = W = 256
EPOCHS = int(os.environ.get("EPOCHS", 20))
RATES = ["25mm", "50mm", "100mm", "150mm", "200mm"]


def load_img(p):
    with np.load(p, allow_pickle=True) as d:
        return np.fft.ifft2(d["fft_complex"].astype(np.complex64)).real.astype(np.float32)


raw_files = sorted(glob.glob(os.path.join(ROOT, "raw_data", "*.npz")))
clean = np.stack([load_img(f) for f in raw_files])
cache = {}
for rate in RATES:
    split = next((s for s in ("train", "validation", "test")
                  if os.path.isdir(os.path.join(ROOT, "merge_data", s, rate))), None)
    if split is None: continue
    rf = sorted(glob.glob(os.path.join(ROOT, "merge_data", split, rate, "*.npz")))
    for i in range(min(len(rf), len(clean))):
        cache[(rate, i)] = load_img(rf[i])
keys = sorted(cache)
idx_tr = [k for k in keys if k[1] < 300]
idx_te = [k for k in keys if k[1] >= 300]
print(f"train {len(idx_tr)}  test {len(idx_te)}")

C = 16


class Scan4(nn.Module):
    """4-direction fused selective scan over the spatial map."""
    def __init__(self, c):
        super().__init__()
        self.n = 8
        self.proj = nn.Conv2d(c, c + 2 * self.n, 1)
        self.A = nn.Parameter(-torch.rand(c, self.n))
        self.D = nn.Parameter(torch.ones(c))
        self.out = nn.Conv2d(c * 4, c, 1)

    def one(self, u, dt, Bm, Cm):
        return selective_scan_fn(u, dt, self.A, Bm, Cm, self.D, delta_softplus=True)

    def forward(self, x):
        B, c, h, w = x.shape
        p = self.proj(x)
        dt = p[:, :c].reshape(B, c, -1)
        Bm = p[:, c:c + self.n].reshape(B, 1, self.n, -1)
        Cm = p[:, c + self.n:].reshape(B, 1, self.n, -1)
        u = x.reshape(B, c, -1)
        ys = [self.one(u, dt, Bm, Cm),
              self.one(u.flip(-1), dt.flip(-1), Bm.flip(-1), Cm.flip(-1)).flip(-1)]
        xt = x.transpose(2, 3).reshape(B, c, -1)
        pt = p.transpose(2, 3)
        dtt = pt[:, :c].reshape(B, c, -1)
        Bt = pt[:, c:c + self.n].reshape(B, 1, self.n, -1)
        Ct = pt[:, c + self.n:].reshape(B, 1, self.n, -1)
        y3 = self.one(xt, dtt, Bt, Ct).reshape(B, c, w, h).transpose(2, 3).reshape(B, c, -1)
        y4 = self.one(xt.flip(-1), dtt.flip(-1), Bt.flip(-1), Ct.flip(-1)).flip(-1) \
                 .reshape(B, c, w, h).transpose(2, 3).reshape(B, c, -1)
        y = torch.cat([ys[0], ys[1], y3, y4], 1).reshape(B, 4 * c, h, w)
        return self.out(y)


class V(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.stem = nn.Sequential(nn.Conv2d(1, C, 1), nn.Hardswish())
        if kind == "local":
            self.body = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C), nn.Hardswish(),
                                      nn.Conv2d(C, C, 1), nn.Hardswish())
        elif kind == "dilated":
            self.body = nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, dilation=1, groups=C), nn.Hardswish(),
                nn.Conv2d(C, C, 3, padding=4, dilation=4, groups=C), nn.Hardswish(),
                nn.Conv2d(C, C, 3, padding=16, dilation=16, groups=C), nn.Hardswish(),
                nn.Conv2d(C, C, 1), nn.Hardswish())
        elif kind == "globalpool":
            self.body = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C), nn.Hardswish())
            self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(C, C, 1), nn.Hardswish())
            self.mix = nn.Sequential(nn.Conv2d(2 * C, C, 1), nn.Hardswish())
        elif kind == "scan":
            self.body = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C), nn.Hardswish())
            self.scan = Scan4(C)
            self.mix = nn.Sequential(nn.Conv2d(2 * C, C, 1), nn.Hardswish())
        self.head = nn.Conv2d(C, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, img):
        f = self.stem(img)
        b = self.body(f)
        if self.kind == "globalpool":
            b = self.mix(torch.cat([b, self.gp(f).expand_as(b)], 1))
        elif self.kind == "scan":
            b = self.mix(torch.cat([b, self.scan(f)], 1))
        return img * (1 + self.head(b))


def batches(idx, bs, sh=True):
    order = np.random.permutation(len(idx)) if sh else np.arange(len(idx))
    for s in range(0, len(order), bs):
        sel = [idx[j] for j in order[s:s + bs]]
        x = torch.tensor(np.stack([cache[k] for k in sel]), device=DEV).unsqueeze(1)
        y = torch.tensor(np.stack([clean[k[1]] for k in sel]), device=DEV).unsqueeze(1)
        yield x, y


with torch.no_grad():
    num = den = 0.0
    for x, y in batches(idx_te, 16, False):
        num += ((x - y) ** 2).sum().item(); den += (y ** 2).sum().item()
base = num / den

print(f"\n{'predictor receptive field':28s} {'params':>8s} {'ms':>7s} {'test NMSE':>11s} {'vs input':>9s}")
print(f"{'(do nothing)':28s} {'-':>8s} {'-':>7s} {base:11.4f} {'0.0%':>9s}")
res = {}
for kind in ["local", "dilated", "globalpool", "scan"]:
    torch.manual_seed(0)
    m = V(kind).to(DEV)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for _ in range(EPOCHS):
        for x, y in batches(idx_tr, 8):
            opt.zero_grad(); F.l1_loss(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    with torch.no_grad():
        num = den = 0.0
        for x, y in batches(idx_te, 16, False):
            p = m(x); num += ((p - y) ** 2).sum().item(); den += (y ** 2).sum().item()
        nm = num / den
        xb = torch.randn(1, 1, H, W, device=DEV)
        for _ in range(10): m(xb)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(30): m(xb)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 30 * 1000
    res[kind] = nm
    print(f"{kind:28s} {npar:8,d} {ms:7.2f} {nm:11.4f} {100*(1-nm/base):8.1f}%", flush=True)

print(f"\nrelative to purely LOCAL prediction:")
for k, v in res.items():
    if k != "local":
        print(f"   {k:12s} {100*(1-v/res['local']):+6.1f}% error reduction")
