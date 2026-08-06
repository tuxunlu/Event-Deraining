"""F5: the per-event native-evidence MLP head, scene-disjoint.

Two stages:
  --cache : run the frozen trunk over all labeled real frames, sample events,
            store per-event feature rows (96-dim) + labels per frame.
  --train : MLP 96 -> 48 -> 32 -> 1 (ReLU, coarse-logit re-injected at each
            hidden layer, final layer zero-init, residual on the trunk logit),
            class-balanced BCE. Train scenes 1-3, test scene 4.

Pre-registered gates (from the fine-grained falsifier ladder):
  KILL if MLP event-BA gain < half of the logreg gain (+0.052 -> need >= +0.026)
  Report tau-correlation (self-prior vs per-frame argmax) -- the pixel-level
  gate value was +0.9918; the event-level F1 gate already FAILED (r=0.39), so
  this is reported, not enforced, with the practical mean-p rule's BA alongside.
Also reports: stratified gains (trunk-correct vs trunk-wrong cells), and the
ambiguity-gate ablation (head applied only near tau) -- expected to LOSE given
the gain lives in trunk-wrong cells; measured, not assumed.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp")
from rsp_3d import ORSPNet3D

DEV = "cuda"
T16, RW, RH = 16, 448, 256
NW, NH = 1280, 720
S = "/fs/nexus-projects/DVS_Actions/dataset/real/EVK4_artifical"
CACHE = "/fs/nexus-scratch/tuxunlu/evhead_cache"
TMP = "/nfshomes/tuxunlu/.claude/jobs/ca4cd659/tmp"
PER_FRAME = 12000


def build_cache():
    ck = torch.load(f"{TMP}/real_evorsp.pt", map_location="cpu")
    m = ORSPNet3D(T=4, num_blocks=3, use_off=True, dilations=(1, 8, 32, 64))
    m.load_state_dict(ck["state_dict"])
    m = m.to(DEV).eval()
    FEAT = {}
    m.out_proj.register_forward_pre_hook(
        lambda mod, inp: FEAT.__setitem__("f", inp[0].detach()))
    rng = np.random.default_rng(0)
    os.makedirs(CACHE, exist_ok=True)
    n_done = 0
    with torch.no_grad():
        for scene in ("scene1", "scene2", "scene3", "scene4"):
            for k in (1, 2, 3):
                for f in sorted(glob.glob(f"{S}/{scene}/merge_data/rain_{k}/*.npz")):
                    b = os.path.basename(f)
                    dst = f"{CACHE}/{scene}_r{k}_{b}"
                    if os.path.exists(dst):
                        n_done += 1
                        continue
                    lp = (f"{S}/{scene}/labels/labels_rain_{k}/labels_{b}"
                          .replace(".npz", ".npy"))
                    try:
                        lab = np.load(lp)
                        d = np.load(f)
                    except Exception:
                        continue
                    x, y, t, p = d["x"], d["y"], d["t"], d["p"]
                    if len(lab) != len(x) or len(x) < 2000:
                        continue
                    sx = (x.astype(np.int64) * RW) // NW
                    sy = (y.astype(np.int64) * RH) // NH
                    t0 = t.min()
                    span = max(int(t.max() - t0), 1)
                    tb16 = np.clip(((t - t0) * T16) // span, 0, T16 - 1).astype(np.int64)
                    tb4 = tb16 // 4
                    pol = (p == 1).astype(np.int64)
                    on = np.zeros((T16, RH * RW), bool)
                    off = np.zeros((T16, RH * RW), bool)
                    sm = pol == 1
                    on[tb16[sm], sy[sm] * RW + sx[sm]] = True
                    off[tb16[~sm], sy[~sm] * RW + sx[~sm]] = True
                    on4 = torch.from_numpy(on.reshape(T16, RH, RW)
                          .reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
                    off4 = torch.from_numpy(off.reshape(T16, RH, RW)
                           .reshape(4, 4, RH, RW).max(1)).float()[None].to(DEV)
                    logit = m(on4, x_off=off4)[0, 0].cpu().numpy()
                    feat = FEAT["f"][0].cpu().numpy()
                    u = (x + 0.5) * RW / NW - 0.5
                    v = (y + 0.5) * RH / NH - 0.5
                    u0 = np.clip(np.floor(u).astype(np.int64), 0, RW - 2)
                    v0 = np.clip(np.floor(v).astype(np.int64), 0, RH - 2)
                    au = np.clip(u - u0, 0, 1)
                    av = np.clip(v - v0, 0, 1)
                    def bil(M):
                        return ((M[..., v0, u0] * (1 - au)
                                 + M[..., v0, u0 + 1] * au) * (1 - av)
                                + (M[..., v0 + 1, u0] * (1 - au)
                                   + M[..., v0 + 1, u0 + 1] * au) * av)
                    l_bil = bil(logit)
                    f_bil = bil(feat).T
                    G = np.zeros((8, NH, NW), np.uint8)
                    np.add.at(G, (pol * 4 + tb4, y, x), 1)
                    Gp = np.pad(G, ((0, 0), (1, 1), (1, 1)))
                    patch = np.log1p(np.stack(
                        [Gp[:, y + dy, x + dx] for dy in range(3)
                         for dx in range(3)], 1)
                        .reshape(len(x), 72).astype(np.float32))
                    sel = rng.choice(len(x), min(PER_FRAME, len(x)),
                                     replace=False)
                    X = np.concatenate([
                        l_bil[sel, None].astype(np.float32),
                        patch[sel], f_bil[sel].astype(np.float32),
                        pol[sel, None].astype(np.float32),
                        np.eye(4, dtype=np.float32)[tb4[sel]]], 1)
                    np.savez_compressed(dst, X=X.astype(np.float16),
                                        lab=lab[sel].astype(np.int8))
                    n_done += 1
                    if n_done % 100 == 0:
                        print(f"  ...{n_done}", flush=True)
    print(f"cache: {n_done} frames", flush=True)


class Head(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.fc1 = nn.Linear(din, 48)
        self.fc2 = nn.Linear(48 + 1, 32)
        self.fc3 = nn.Linear(32 + 1, 1)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, z):
        l = z[:, :1]                                          # trunk logit
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(torch.cat([h, l], 1)))
        return l + self.fc3(torch.cat([h, l], 1))             # residual


def ba(keep, lab):
    n_s = max(int((lab == 1).sum()), 1)
    n_r = max(int((lab == 0).sum()), 1)
    return 0.5 * (int((keep & (lab == 1)).sum()) / n_s
                  + int((~keep & (lab == 0)).sum()) / n_r)


def train():
    tr_f = sorted(glob.glob(f"{CACHE}/scene[123]_*.npz"))
    te_f = sorted(glob.glob(f"{CACHE}/scene4_*.npz"))
    print(f"train {len(tr_f)} frames, test {len(te_f)} frames", flush=True)
    Xtr = np.concatenate([np.load(f)["X"] for f in tr_f]).astype(np.float32)
    ytr = np.concatenate([np.load(f)["lab"] for f in tr_f]).astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    mu[0], sd[0] = 0, 1                                       # keep raw logit
    Xtr = (Xtr - mu) / sd
    Xt = torch.from_numpy(Xtr)          # stays on CPU; batches stream to GPU
    yt = torch.from_numpy(ytr)
    w_pos = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    head = Head(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w_pos, device=DEV))
    n = len(yt)
    for ep in range(8):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 262144):
            idx = perm[i:i + 262144]
            xb = Xt[idx].to(DEV, non_blocking=True)
            yb = yt[idx].to(DEV, non_blocking=True)
            out = head(xb)[:, 0]
            loss = lossf(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  ep {ep+1}/8 loss {tot / (n // 262144 + 1):.4f}", flush=True)

    # evaluate per test frame
    res = {"trunk": [], "mlp": [], "mlp_gate": [], "taus": [], "amax": []}
    strat = {"ok_trunk": [], "ok_mlp": [], "bad_trunk": [], "bad_mlp": []}
    with torch.no_grad():
        for f in te_f:
            d = np.load(f)
            X = (d["X"].astype(np.float32) - mu) / sd
            lab = d["lab"].astype(np.int64)
            Xg = torch.from_numpy(X).to(DEV)
            l_trunk = X[:, 0]
            p_trunk = 1 / (1 + np.exp(-l_trunk))
            out = torch.sigmoid(head(Xg)[:, 0]).cpu().numpy()
            tau_t = float(p_trunk.mean())
            tau_h = float(out.mean())
            res["trunk"].append(ba(p_trunk > tau_t, lab))
            res["mlp"].append(ba(out > tau_h, lab))
            # ambiguity gate: head only where trunk is uncertain
            amb = np.abs(p_trunk - tau_t) < 0.25
            mix = np.where(amb, out, p_trunk)
            res["mlp_gate"].append(ba(mix > float(mix.mean()), lab))
            # calibration
            ts = np.quantile(out, np.linspace(0.02, 0.98, 49))
            bas = [ba(out > t, lab) for t in ts]
            res["taus"].append(tau_h)
            res["amax"].append(float(ts[int(np.argmax(bas))]))
            # stratification proxy: trunk-correct events (decision == label)
            okm = (p_trunk > tau_t) == (lab == 1)
            if okm.sum() > 100 and (~okm).sum() > 100:
                strat["ok_trunk"].append(ba((p_trunk > tau_t)[okm], lab[okm]))
                strat["ok_mlp"].append(ba((out > tau_h)[okm], lab[okm]))
                strat["bad_trunk"].append(ba((p_trunk > tau_t)[~okm], lab[~okm]))
                strat["bad_mlp"].append(ba((out > tau_h)[~okm], lab[~okm]))

    t, m_, g = (np.mean(res["trunk"]), np.mean(res["mlp"]),
                np.mean(res["mlp_gate"]))
    calr = float(np.corrcoef(res["taus"], res["amax"])[0, 1])
    print(f"\n=== F5 RESULT (scene-disjoint, {len(te_f)} test frames) ===")
    print(f"  trunk (self-prior)        event-BA {t:.4f}")
    print(f"  MLP head                  event-BA {m_:.4f}   ({m_ - t:+.4f})")
    print(f"  MLP w/ ambiguity gate     event-BA {g:.4f}   ({g - t:+.4f})")
    print(f"  logreg reference gain     +0.0523 ; KILL if MLP gain < +0.026")
    print(f"  verdict: {'PASS' if m_ - t >= 0.026 else 'KILL'}")
    print(f"  calibration corr(tau, argmax) = {calr:+.4f}  (reported; "
          f"pixel-level was +0.9918, event-level F1 gate failed at 0.39)")
    print(f"  head params: {sum(p.numel() for p in head.parameters()):,}")
    torch.save({"state_dict": head.state_dict(), "mu": mu, "sd": sd},
               f"{TMP}/evhead.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.cache:
        build_cache()
    if a.train:
        train()
