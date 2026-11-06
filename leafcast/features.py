"""Batched summaries of a measurement history (PyTorch).

All forecasters share the same description of the history window, computed
on the fly so that noise can be injected into the raw sequence during
training. Tensors have shape (B, L) with a boolean mask marking observed
frames; times are hours relative to the prediction origin (<= 0).
"""

import torch

EPS = 1e-6


def _masked(x, m, fill=0.0):
    return torch.where(m, x, torch.full_like(x, fill))


def theil_sen(t, y, m):
    """Theil-Sen slope and intercept (value at t = 0) for each row."""
    dt = t.unsqueeze(2) - t.unsqueeze(1)                     # t_i - t_j
    dy = y.unsqueeze(2) - y.unsqueeze(1)
    valid = (m.unsqueeze(2) & m.unsqueeze(1)) & (dt > 0.5)
    slopes = torch.where(valid, dy / dt.clamp(min=EPS), torch.full_like(dt, float("nan")))
    slope = torch.nanmedian(slopes.flatten(1), dim=1).values
    slope = torch.nan_to_num(slope, nan=0.0)
    resid = torch.where(m, y - slope.unsqueeze(1) * t, torch.full_like(y, float("nan")))
    level = torch.nan_to_num(torch.nanmedian(resid, dim=1).values, nan=0.0)
    return slope, level


def masked_mean(x, m):
    return _masked(x, m).sum(1) / m.sum(1).clamp(min=1)


def masked_std(x, m):
    mu = masked_mean(x, m)
    return torch.sqrt(masked_mean((x - mu.unsqueeze(1)) ** 2, m) + EPS)


def last_value(x, m):
    # sequences are right-aligned: the last column is the origin frame
    return x[:, -1]


def summarize(t, N, a, m, sdN, sda, age, a_scale=0.05):
    """Return (features, trend_params, reliability).

    trend_params: dict with Theil-Sen level and slope for N and a.
    features    : hand-crafted descriptors used by the feature-based models.
    reliability : observable indicators of how trustworthy the history is.
    """
    sN, lN = theil_sen(t, N, m)
    sa, la = theil_sen(t, a, m)
    n = m.sum(1).float()
    span = -torch.min(_masked(t, m, 0.0), dim=1).values
    resN = (N - (lN.unsqueeze(1) + sN.unsqueeze(1) * t)).abs()
    resa = (a - (la.unsqueeze(1) + sa.unsqueeze(1) * t)).abs()
    madN = torch.nan_to_num(torch.nanmedian(torch.where(m, resN, torch.nan), 1).values, nan=0.0)
    mada = torch.nan_to_num(torch.nanmedian(torch.where(m, resa, torch.nan), 1).values, nan=0.0)
    lastN, lasta = last_value(N, m), last_value(a, m)
    firstN = torch.gather(N, 1, (m.float().argmax(1, keepdim=True))).squeeze(1)
    firsta = torch.gather(a, 1, (m.float().argmax(1, keepdim=True))).squeeze(1)
    feats = torch.stack([
        lastN, masked_mean(N, m), masked_std(N, m), lN, sN * 24, madN, lastN - firstN,
        lasta / a_scale, masked_mean(a, m) / a_scale, masked_std(a, m) / a_scale,
        la / a_scale, sa * 24 / a_scale, mada / a_scale, (lasta - firsta) / a_scale,
        n, span, age,
    ], dim=1)
    rel = torch.stack([
        masked_mean(sdN, m), masked_mean(sda, m) / a_scale,
        madN, mada / a_scale, (lastN - lN).abs(), n, span,
    ], dim=1)
    return feats, {"N_slope": sN, "N_level": lN, "a_slope": sa, "a_level": la}, rel


def trend_forecast(params, horizons_dt):
    """Linear Theil-Sen extrapolation; horizons_dt has shape (B, H)."""
    N = params["N_level"].unsqueeze(1) + params["N_slope"].unsqueeze(1) * horizons_dt
    a = params["a_level"].unsqueeze(1) + params["a_slope"].unsqueeze(1) * horizons_dt
    return torch.cat([N.clamp(min=0), a.clamp(0, 1)], dim=1)   # (B, 2H): N..., a...


FEATURE_NAMES = ["N_last", "N_mean", "N_std", "N_level", "N_slope_day", "N_fit_mad",
                 "N_delta", "a_last", "a_mean", "a_std", "a_level", "a_slope_day",
                 "a_fit_mad", "a_delta", "n_frames", "span_h", "days_monitored"]
RELIABILITY_NAMES = ["N_tta_sd", "a_tta_sd", "N_fit_mad", "a_fit_mad", "N_last_minus_level",
                     "n_frames", "span_h"]
