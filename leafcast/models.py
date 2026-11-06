"""Forecasting models and the shared training loop.

Every model receives the raw history (times, N, a, mask, TTA spreads, days
monitored) and the target lead times. Outputs are ordered
[N_24, N_48, N_72, a_24, a_48, a_72].

Model families
--------------
direct  : y_hat = mu + sigma * f(x)                  (no trend anchoring)
trend   : y_hat = T + s * f(x)                       (trend-residual, TR)
gated   : y_hat = T + g(r) * s * f(x),  g in (0,1)   (reliability-gated TR)
gru     : y_hat = mu + sigma * GRU(sequence)         (sequence baseline)
where T is the Theil-Sen extrapolation of the history and r are observable
reliability indicators of the history (see features.summarize).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import summarize, trend_forecast


def mlp(i, o, h=128, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))


class Forecaster(nn.Module):
    def __init__(self, kind, n_feat, n_rel, n_img=0, n_out=6, use_tta=True, hidden=128):
        super().__init__()
        self.kind, self.n_img, self.use_tta = kind, n_img, use_tta
        for name, dim in (("f_mu", n_feat), ("f_sd", n_feat), ("r_mu", n_rel), ("r_sd", n_rel),
                          ("y_mu", n_out), ("y_sd", n_out), ("res_sd", n_out)):
            self.register_buffer(name, torch.zeros(dim) if "mu" in name else torch.ones(dim))
        if kind == "gru":
            self.gru = nn.GRU(6, 64, batch_first=True)
            self.head = mlp(64 + 1 + n_img, n_out, hidden)
        else:
            self.head = mlp(n_feat + n_img, n_out, hidden)
        if kind == "gated":
            self.gate = nn.Sequential(nn.Linear(n_rel, 32), nn.ReLU(), nn.Linear(32, n_out))

    def describe(self, b):
        sdN, sda = b["sdN"], b["sda"]
        if not self.use_tta:
            sdN, sda = torch.zeros_like(sdN), torch.zeros_like(sda)
        feats, params, rel = summarize(b["t"], b["N"], b["a"], b["mask"], sdN, sda, b["age"])
        T = trend_forecast(params, b["dt"])
        return feats, T, rel

    def forward(self, b, img=None):
        feats, T, rel = self.describe(b)
        f = (feats - self.f_mu) / self.f_sd
        gate = None
        if self.kind == "gru":
            x = torch.stack([(b["N"] - self.f_mu[0]) / self.f_sd[0],
                             (b["a"] * 20 - self.f_mu[7]) / self.f_sd[7],
                             b["t"] / 48.0, b["sdN"] if self.use_tta else 0 * b["sdN"],
                             (b["sda"] * 20) if self.use_tta else 0 * b["sda"],
                             b["mask"].float()], dim=2)
            _, h = self.gru(x)
            z = torch.cat([h[-1], f[:, -1:]] + ([img] if img is not None else []), 1)
            return self.y_mu + self.y_sd * self.head(z), T, gate
        z = torch.cat([f] + ([img] if img is not None else []), 1)
        out = self.head(z)
        if self.kind == "direct":
            return self.y_mu + self.y_sd * out, T, gate
        resid = self.res_sd * out
        if self.kind == "gated":
            gate = torch.sigmoid(self.gate((rel - self.r_mu) / self.r_sd))
            resid = gate * resid
        return T + resid, T, gate


def add_noise(b, sigma_N, a_ratio, gen):
    """Gaussian noise on the observed history; sigma_N has shape (B,)."""
    b = dict(b)
    eps1 = torch.randn(b["N"].shape, generator=gen, device=b["N"].device)
    eps2 = torch.randn(b["a"].shape, generator=gen, device=b["a"].device)
    s = sigma_N.unsqueeze(1)
    b["N"] = torch.where(b["mask"], (b["N"] + s * eps1).clamp(min=0), b["N"])
    b["a"] = torch.where(b["mask"], (b["a"] + s * a_ratio * eps2).clamp(0, 1), b["a"])
    return b


def subset(b, idx):
    return {k: v[idx] for k, v in b.items()}


@torch.no_grad()
def fit_normalisers(model, b):
    feats, T, rel = model.describe(b)
    model.f_mu.copy_(feats.mean(0)); model.f_sd.copy_(feats.std(0) + 1e-3)
    model.r_mu.copy_(rel.mean(0)); model.r_sd.copy_(rel.std(0) + 1e-3)
    model.y_mu.copy_(b["y"].mean(0)); model.y_sd.copy_(b["y"].std(0) + 1e-6)
    model.res_sd.copy_((b["y"] - T).std(0) + 1e-6)


def train(model, btr, bva, img_tr=None, img_va=None, seed=0, augment=False, a_ratio=0.0,
          noise_levels=(0.5, 1.0, 2.0), p_noise=0.5, lr=1e-3, epochs=300, patience=30,
          batch_size=128):
    torch.manual_seed(seed)
    gen = torch.Generator(device=btr["N"].device).manual_seed(seed)
    fit_normalisers(model, btr)
    scale = model.res_sd.clone()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = btr["N"].shape[0]
    levels = torch.tensor(noise_levels, device=btr["N"].device)
    best, best_state, wait, log = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=btr["N"].device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            b = subset(btr, idx)
            if augment:
                pick = torch.randint(0, len(levels), (len(idx),), generator=gen, device=idx.device)
                on = torch.rand(len(idx), generator=gen, device=idx.device) < p_noise
                b = add_noise(b, torch.where(on, levels[pick], torch.zeros_like(levels[pick])),
                              a_ratio, gen)
            pred, _, _ = model(b, None if img_tr is None else img_tr[idx])
            loss = F.huber_loss(pred / scale, b["y"] / scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred, _, _ = model(bva, img_va)
            v = float(F.huber_loss(pred / scale, bva["y"] / scale))
        log.append(v)
        if v < best - 1e-5:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_loss": best, "epochs_run": ep + 1, "best_epoch": int(np.argmin(log)) + 1}


@torch.no_grad()
def predict(model, b, img=None):
    model.eval()
    pred, T, gate = model(b, img)
    H = pred.shape[1] // 2
    pred = torch.cat([pred[:, :H].clamp(min=0), pred[:, H:].clamp(0, 1)], 1)
    return (pred.cpu().numpy(), T.cpu().numpy(),
            None if gate is None else gate.cpu().numpy())
