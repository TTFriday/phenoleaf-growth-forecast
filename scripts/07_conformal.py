"""Split-conformal prediction intervals for the forecasts.

Two calibrations are compared, both using the 37 calibration plants that were
never used for training or model selection:
  * standard  : interval = y_hat +/- q, q the conformal quantile of |y - y_hat|
  * normalized: interval = y_hat +/- q * s(x), where s(x) is a difficulty
    estimate (gradient boosting fitted on validation plants to predict
    |y - y_hat| from the history descriptors and reliability indicators) and
    q the conformal quantile of |y - y_hat| / s(x).
Coverage and mean width are evaluated on the test plants, overall and within
tertiles of history noisiness (Theil-Sen residual of the measured leaf
counts) to check conditional behaviour.

Output: outputs/results/conformal.csv, conformal_conditional.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402
from leafcast.data import build_arrays, load_tables, to_torch  # noqa: E402
from leafcast.features import summarize  # noqa: E402

ALPHAS = (0.10, 0.05)
H = len(C.HORIZONS)
MODELS = {"measured": ["gated", "trend_residual", "gru", "gradient_boosting"],
          "manual": ["trend_residual", "gradient_boosting"]}


def conformal_q(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[min(k, n) - 1]


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    samples, meas = load_tables()
    rows, cond = [], []
    for source, models in MODELS.items():
        arr, meta = build_arrays(samples, meas, source)
        b = to_torch(arr, np.arange(len(meta)), "cpu")
        feats, _, rel = summarize(b["t"], b["N"], b["a"], b["mask"], b["sdN"], b["sda"], b["age"])
        X = np.concatenate([feats.numpy(), rel.numpy()], 1)
        y = arr["y"]
        sp = meta.split.values
        va, ca, te = (np.where(sp == s)[0] for s in ("val", "calib", "test"))
        early = meta.early.values
        noisiness = rel.numpy()[:, 2]
        cuts = np.quantile(noisiness[te], [1 / 3, 2 / 3])
        tert = np.digitize(noisiness[te], cuts)
        for model in models:
            files = sorted((C.PRED_DIR / source).glob(f"{model}_s*.npz"))
            files = [f for f in files if f.stem.rsplit("_s", 1)[0] == model]
            for f in files:
                pred = np.load(f)["pred"]
                seed = int(f.stem.rsplit("_s", 1)[1])
                for j in range(2 * H):
                    target = "N" if j < H else "a"
                    h = C.HORIZONS[j % H]
                    res = np.abs(y[:, j] - pred[:, j])
                    Xj = np.concatenate([X, pred[:, [j]]], 1)
                    sm = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                                       min_samples_leaf=20, random_state=0)
                    sm.fit(Xj[va], res[va])
                    s = np.clip(sm.predict(Xj), np.quantile(res[va], 0.05) + 1e-6, None)
                    for alpha in ALPHAS:
                        q = conformal_q(res[ca], alpha)
                        qn = conformal_q(res[ca] / s[ca], alpha)
                        for kind, half in (("standard", np.full(len(y), q)), ("normalized", qn * s)):
                            lo = pred[:, j] - half
                            lo = np.clip(lo, 0, None)
                            hi = pred[:, j] + half
                            if target == "a":
                                hi = np.clip(hi, 0, 1)
                            cover = (y[:, j] >= lo) & (y[:, j] <= hi)
                            width = hi - lo
                            scale = 100 if target == "a" else 1
                            for subset, idx in (("test", te), ("early", te[early[te]])):
                                rows.append({"source": source, "model": model, "seed": seed,
                                             "target": target, "horizon": h, "method": kind,
                                             "nominal": 1 - alpha, "subset": subset, "coverage": cover[idx].mean(),
                                             "mean_width": width[idx].mean() * scale,
                                             "median_width": np.median(width[idx]) * scale})
                            for t in range(3):
                                idx = te[tert == t]
                                cond.append({"source": source, "model": model, "seed": seed,
                                             "target": target, "horizon": h, "method": kind,
                                             "nominal": 1 - alpha, "noise_tertile": t + 1, "coverage": cover[idx].mean(),
                                             "mean_width": width[idx].mean() * scale})
    df = pd.DataFrame(rows)
    dc = pd.DataFrame(cond)
    df.to_csv(C.RESULT_DIR / "conformal.csv", index=False)
    dc.to_csv(C.RESULT_DIR / "conformal_conditional.csv", index=False)
    g = df.groupby(["source", "model", "nominal", "subset", "target", "horizon", "method"])[
        ["coverage", "mean_width"]].mean().round(3)
    pd.set_option("display.width", 200)
    print(g.to_string())
    print(dc.groupby(["source", "model", "nominal", "target", "method", "noise_tertile"])[
        ["coverage", "mean_width"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
