"""Plant-level resampling statistics.

Forecast errors of samples from the same plant are correlated, so all
uncertainty statements resample plants (cluster bootstrap) rather than
individual forecasts.
"""

import numpy as np
from scipy import stats as st


def plant_groups(plants):
    uniq, inv = np.unique(np.asarray(plants), return_inverse=True)
    return uniq, inv


def cluster_bootstrap_mean(values, plants, n_boot=5000, seed=0):
    """Mean of ``values`` with a 95% percentile interval from resampling plants.
    ``values`` may be 1-D (samples) or 2-D (samples x outputs)."""
    v = np.asarray(values, float)
    uniq, inv = plant_groups(plants)
    k = len(uniq)
    sums = np.zeros((k,) + v.shape[1:])
    np.add.at(sums, inv, v)
    cnt = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    w = np.zeros((n_boot, k))
    for b in range(n_boot):
        w[b] = np.bincount(draws[b], minlength=k)
    num = w @ sums.reshape(k, -1)
    den = w @ cnt
    boot = (num / den[:, None]).reshape((n_boot,) + v.shape[1:])
    est = v.mean(0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return est, lo, hi, boot


def paired_test(err_a, err_b, plants, n_boot=10000, seed=0):
    """Compare two methods on the same samples.

    Returns the mean difference (a - b), its 95% cluster-bootstrap interval,
    a two-sided bootstrap p-value (null: mean difference = 0, obtained by
    centring the bootstrap distribution) and a Wilcoxon signed-rank p-value on
    plant-averaged differences.
    """
    d = np.asarray(err_a, float) - np.asarray(err_b, float)
    est, lo, hi, boot = cluster_bootstrap_mean(d, plants, n_boot, seed)
    centred = boot - est
    p_boot = (np.sum(np.abs(centred) >= abs(est)) + 1) / (n_boot + 1)
    uniq, inv = plant_groups(plants)
    per_plant = np.bincount(inv, weights=d) / np.bincount(inv)
    if np.allclose(per_plant, 0):
        p_w = 1.0
    else:
        p_w = float(st.wilcoxon(per_plant, zero_method="wilcox").pvalue)
    return {"diff": float(est), "ci_low": float(lo), "ci_high": float(hi),
            "p_bootstrap": float(p_boot), "p_wilcoxon": p_w,
            "plants_a_better": int(np.sum(per_plant < 0)), "n_plants": int(len(uniq))}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
