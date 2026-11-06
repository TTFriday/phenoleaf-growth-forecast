"""Evaluate all stored forecasts and run the paired comparisons.

Errors are first computed per seed; the per-sample error of a neural model is
the mean absolute error across its five seeds. Mean absolute errors are
reported with 95% plant-level bootstrap intervals, and the seed-to-seed
standard deviation is given separately. Comparisons are paired on identical
test samples; p-values from the plant-level bootstrap are Holm-adjusted within
each family of comparisons.

Outputs: outputs/results/metrics.csv, comparisons.csv, summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402
from leafcast.stats import cluster_bootstrap_mean, holm, paired_test  # noqa: E402

H = len(C.HORIZONS)
NOISE = (0.0, 0.5, 1.0, 2.0, 3.0)


def load(source, config, key="pred"):
    d = C.PRED_DIR / source
    files = sorted(d.glob(f"{config}_s*.npz"))
    files = [f for f in files if f.stem.rsplit("_s", 1)[0] == config]
    if not files:
        raise FileNotFoundError(f"{source}/{config}")
    return np.stack([np.load(f)[key] for f in files])          # (seeds, n, 6)


def noisy(source, config, sigma):
    if sigma == 0:
        return None
    z = []
    for f in sorted((C.PRED_DIR / source).glob(f"{config}_s*.npz")):
        if f.stem.rsplit("_s", 1)[0] != config:
            continue
        npz = np.load(f)
        z.append(np.stack([npz[k] for k in npz.files if k.startswith(f"noise_{sigma}_")]))
    return np.stack(z)                                          # (seeds, repeats, n_test, 6)


class Context:
    def __init__(self, source):
        self.source = source
        self.meta = pd.read_csv(C.PRED_DIR / source / "samples.csv")
        self.y = np.load(C.PRED_DIR / source / "targets.npy")
        self.test = np.where(self.meta.split == "test")[0]
        self.early = np.where((self.meta.split == "test") & self.meta.early)[0]

    def errors(self, config, subset="test", sigma=0.0):
        idx = self.test if subset == "test" else self.early
        if sigma == 0:
            P = load(self.source, config)[:, idx]
            E = np.abs(P - self.y[idx])                           # (seeds, n, 6)
        else:
            P = noisy(self.source, config, sigma)                 # test only
            pos = np.searchsorted(self.test, idx)
            E = np.abs(P[:, :, pos] - self.y[idx]).mean(1)
        return E


def summarise(ctx, config, subset="test", sigma=0.0):
    E = ctx.errors(config, subset, sigma)
    idx = ctx.test if subset == "test" else ctx.early
    plants = ctx.meta.plant.values[idx]
    e = E.mean(0)
    row = {"source": ctx.source, "model": config, "subset": subset, "sigma": sigma,
           "n_samples": len(idx), "n_plants": len(set(plants)), "n_seeds": E.shape[0]}
    cols = {}
    for j, h in enumerate(C.HORIZONS):
        cols[f"N_{h}"] = e[:, j]
        cols[f"a_{h}"] = e[:, H + j] * 100
    cols["N_mean"] = e[:, :H].mean(1)
    cols["a_mean"] = e[:, H:].mean(1) * 100
    for k, v in cols.items():
        est, lo, hi, _ = cluster_bootstrap_mean(v, plants, n_boot=2000)
        row[f"{k}"], row[f"{k}_lo"], row[f"{k}_hi"] = float(est), float(lo), float(hi)
    seedN = E[:, :, :H].mean((1, 2))
    seeda = E[:, :, H:].mean((1, 2)) * 100
    row["N_mean_seed_sd"] = float(seedN.std(ddof=1)) if len(seedN) > 1 else 0.0
    row["a_mean_seed_sd"] = float(seeda.std(ddof=1)) if len(seeda) > 1 else 0.0
    return row


def compare(ctx_a, cfg_a, ctx_b, cfg_b, family, label, subset="test", sigma_a=0.0, sigma_b=0.0):
    Ea = ctx_a.errors(cfg_a, subset, sigma_a).mean(0)
    Eb = ctx_b.errors(cfg_b, subset, sigma_b).mean(0)
    idx = ctx_a.test if subset == "test" else ctx_a.early
    plants = ctx_a.meta.plant.values[idx]
    out = []
    for target, sl, scale in (("N", slice(0, H), 1.0), ("a", slice(H, 2 * H), 100.0)):
        r = paired_test(Ea[:, sl].mean(1) * scale, Eb[:, sl].mean(1) * scale, plants)
        r.update({"family": family, "comparison": label, "target": target, "subset": subset,
                  "model_a": f"{ctx_a.source}/{cfg_a}", "model_b": f"{ctx_b.source}/{cfg_b}",
                  "sigma": sigma_a})
        out.append(r)
    return out


def main():
    A, B, I = Context("manual"), Context("measured"), Context("insample")
    base = ["persistence", "linear_trend", "damped_trend", "gradient_boosting"]
    neural_A = ["gru", "direct", "trend_residual", "gated", "direct+img_task",
                "direct+img_imagenet", "trend_residual+img_task", "trend_residual+img_imagenet"]
    neural_B = neural_A + ["gated+img_task", "gru_noaug", "direct_noaug",
                           "trend_residual_noaug", "gated_noaug", "gated_seqonly"]
    rows = []
    for subset in ("test", "early"):
        rows += [summarise(A, m, subset) for m in base + neural_A]
        rows += [summarise(B, m, subset) for m in base + neural_B]
        rows += [summarise(I, m, subset) for m in ["direct_noaug", "trend_residual_noaug",
                                                    "gru_noaug", "gated"]]
    robust = base + ["gru", "direct", "trend_residual", "gated", "gru_noaug", "direct_noaug",
                     "trend_residual_noaug", "gated_noaug", "gated_seqonly"]
    for s in NOISE[1:]:
        rows += [summarise(B, m, "test", s) for m in robust]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(C.RESULT_DIR / "metrics.csv", index=False)

    comps = []
    for subset in ("test", "early"):
        for ctx, models, fam in ((A, base[1:] + neural_A[:4], "A_vs_persistence"),
                                 (B, base[1:] + neural_B[:4], "B_vs_persistence")):
            for m in models:
                comps += compare(ctx, m, ctx, "persistence", fam, f"{m} vs persistence", subset)
        for ctx, tag in ((A, "A"), (B, "B")):
            for m in ("direct", "trend_residual"):
                for img in ("img_task", "img_imagenet"):
                    comps += compare(ctx, f"{m}+{img}", ctx, m, f"{tag}_visual_increment",
                                     f"{m}+{img} vs {m}", subset)
        comps += compare(B, "gated+img_task", B, "gated", "B_visual_increment",
                         "gated+img_task vs gated", subset)
        comps += compare(B, "gated", B, "trend_residual", "B_gating", "gated vs trend_residual", subset)
        comps += compare(B, "gated", B, "direct", "B_gating", "gated vs direct", subset)
        comps += compare(B, "gated", B, "gated_seqonly", "B_gating", "gated vs gated_seqonly", subset)
        comps += compare(B, "gated", B, "gated_noaug", "B_gating", "gated vs gated_noaug", subset)
        comps += compare(B, "trend_residual", B, "direct", "B_trend", "trend_residual vs direct", subset)
        for m in ("direct_noaug", "trend_residual_noaug", "gru_noaug", "gated"):
            comps += compare(I, m, B, m, "leakage", f"in-sample vs cross-fitted ({m})", subset)
        for m in ("gru", "direct", "trend_residual", "gated", "gradient_boosting"):
            comps += compare(B, m, A, m, "protocol_gap", f"{m}: measured vs manual", subset)
        for ctx, tag in ((A, "A"), (B, "B")):
            for m in ("direct", "trend_residual", "gated", "gradient_boosting"):
                comps += compare(ctx, "gru", ctx, m, f"{tag}_learned", f"gru vs {m}", subset)
        for m in ("gru", "direct", "trend_residual", "gated"):
            comps += compare(B, m, B, f"{m}_noaug", "B_noise_training",
                             f"{m}: noise training vs none", subset)
    for s in NOISE[1:]:
        for m in ("gru", "direct", "trend_residual", "gated"):
            comps += compare(B, m, B, f"{m}_noaug", "B_noise_training_noise",
                             f"{m}: noise training vs none (sigma={s})", "test", s, s)
    for s in NOISE[1:]:
        for m in ("trend_residual", "direct", "gru"):
            comps += compare(B, "gated", B, m, "B_gating_noise", f"gated vs {m} (sigma={s})",
                             "test", s, s)
        comps += compare(B, "gated", B, "gated_seqonly", "B_gating_noise",
                         f"gated vs gated_seqonly (sigma={s})", "test", s, s)
    comp = pd.DataFrame(comps)
    comp["p_holm"] = np.nan
    for (fam, sub, tgt), g in comp.groupby(["family", "subset", "target"]):
        comp.loc[g.index, "p_holm"] = holm(g.p_bootstrap.values)
    comp.to_csv(C.RESULT_DIR / "comparisons.csv", index=False)

    show = metrics[(metrics.sigma == 0) & (metrics.subset == "test")]
    pd.set_option("display.width", 200)
    print(show[["source", "model", "N_24", "N_48", "N_72", "N_mean", "N_mean_seed_sd",
                "a_mean", "a_mean_seed_sd"]].round(3).to_string(index=False))
    print(comp[comp.subset == "test"][["family", "comparison", "target", "diff", "ci_low",
                                         "ci_high", "p_bootstrap", "p_holm", "p_wilcoxon"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
