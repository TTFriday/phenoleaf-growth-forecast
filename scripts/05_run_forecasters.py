"""Run all forecasters under both input protocols.

Protocol A ("manual") feeds annotation-derived leaf counts and areas;
Protocol B ("measured") feeds cross-fitted image-based measurements; the
"insample" source trains on in-sample measurements of the training plants
(leakage ablation). Every model is evaluated on the same samples and the
predictions for all partitions are stored for later analysis. Neural models
are trained with five seeds. For Protocol B, test-time robustness is probed by
adding Gaussian noise of fixed magnitude to the measured history.

Output: outputs/predictions/<source>/<config>_s<seed>.npz, run_log.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402
from leafcast.data import build_arrays, image_features, load_tables, to_torch  # noqa: E402
from leafcast.features import summarize, trend_forecast  # noqa: E402
from leafcast.models import Forecaster, add_noise, predict, train  # noqa: E402

DEVICE = torch.device("cuda")
H = len(C.HORIZONS)
TEST_NOISE = (0.0, 0.5, 1.0, 2.0, 3.0)
NOISE_REPEATS = 3

# name: (kind, image features, augment, use TTA spread)
NEURAL = {
    "manual": {
        "gru": ("gru", None, False, False),
        "direct": ("direct", None, False, False),
        "trend_residual": ("trend", None, False, False),
        "gated": ("gated", None, False, False),
        "direct+img_task": ("direct", "task", False, False),
        "direct+img_imagenet": ("direct", "imagenet", False, False),
        "trend_residual+img_task": ("trend", "task", False, False),
        "trend_residual+img_imagenet": ("trend", "imagenet", False, False),
    },
    "measured": {
        "gru": ("gru", None, True, True),
        "direct": ("direct", None, True, True),
        "trend_residual": ("trend", None, True, True),
        "gated": ("gated", None, True, True),
        "direct+img_task": ("direct", "task", True, True),
        "direct+img_imagenet": ("direct", "imagenet", True, True),
        "trend_residual+img_task": ("trend", "task", True, True),
        "trend_residual+img_imagenet": ("trend", "imagenet", True, True),
        "gated+img_task": ("gated", "task", True, True),
        "gru_noaug": ("gru", None, False, True),
        "direct_noaug": ("direct", None, False, True),
        "trend_residual_noaug": ("trend", None, False, True),
        "gated_noaug": ("gated", None, False, True),
        "gated_seqonly": ("gated", None, True, False),
    },
    "insample": {
        "direct_noaug": ("direct", None, False, True),
        "trend_residual_noaug": ("trend", None, False, True),
        "gru_noaug": ("gru", None, False, True),
        "gated": ("gated", None, True, True),
    },
}
ROBUSTNESS = {"gru", "direct", "trend_residual", "gated", "gru_noaug", "direct_noaug",
              "trend_residual_noaug", "gated_noaug", "gated_seqonly"}


def mae(p, y):
    return np.abs(p - y).mean(0)


# ---------------------------------------------------------------- baselines
def describe_np(arr, idx):
    b = to_torch(arr, idx, "cpu")
    feats, params, rel = summarize(b["t"], b["N"], b["a"], b["mask"], b["sdN"], b["sda"], b["age"])
    return b, feats.numpy(), params


def damped(params, dt, phi):
    def cum(x):
        return dt if phi >= 1.0 else (1 - phi ** dt) / (1 - phi)
    N = params["N_level"][:, None] + params["N_slope"][:, None] * cum(dt)
    a = params["a_level"][:, None] + params["a_slope"][:, None] * cum(dt)
    return N, a


def run_baselines(arr, splits, out_dir, noise_inputs=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(arr["y"])
    all_idx = np.arange(n)
    b, feats, params = describe_np(arr, all_idx)
    dt = arr["dt"]
    y = arr["y"]
    preds = {}
    last = np.concatenate([np.repeat(arr["N"][:, -1:], H, 1), np.repeat(arr["a"][:, -1:], H, 1)], 1)
    preds["persistence"] = last
    T = trend_forecast(params, torch.as_tensor(dt)).numpy()
    preds["linear_trend"] = T
    va = splits["val"]
    best = {}
    for tgt, sl in (("N", slice(0, H)), ("a", slice(H, 2 * H))):
        scores = {}
        for phi in (0.95, 0.97, 0.98, 0.99, 0.995, 1.0):
            Nd, ad = damped({k: v.numpy() for k, v in params.items()}, dt, phi)
            p = Nd if tgt == "N" else ad
            scores[phi] = float(np.abs(p[va] - y[va, sl]).mean())
        best[tgt] = min(scores, key=scores.get)
    Nd, _ = damped({k: v.numpy() for k, v in params.items()}, dt, best["N"])
    _, ad = damped({k: v.numpy() for k, v in params.items()}, dt, best["a"])
    preds["damped_trend"] = np.concatenate([np.clip(Nd, 0, None), np.clip(ad, 0, 1)], 1)
    # gradient boosting on the history descriptors and the trend forecast
    X = np.concatenate([feats, T], 1)
    tr = splits["train"]
    grid = [dict(max_iter=200, max_depth=3, learning_rate=0.05),
            dict(max_iter=400, max_depth=3, learning_rate=0.03, l2_regularization=1.0),
            dict(max_iter=300, max_depth=5, learning_rate=0.03, min_samples_leaf=40)]
    hg = np.zeros_like(y)
    hgb_models = []
    for j in range(2 * H):
        best_m, best_s = None, np.inf
        for g in grid:
            m = HistGradientBoostingRegressor(random_state=0, **g).fit(X[tr], y[tr, j])
            s = np.abs(m.predict(X[va]) - y[va, j]).mean()
            if s < best_s:
                best_m, best_s = m, s
        hg[:, j] = best_m.predict(X)
        hgb_models.append(best_m)
    hg[:, :H] = np.clip(hg[:, :H], 0, None)
    hg[:, H:] = np.clip(hg[:, H:], 0, 1)
    preds["gradient_boosting"] = hg
    noise = {}
    if noise_inputs is not None:
        te = splits["test"]
        for key, bn in noise_inputs.items():
            f2, p2, _ = summarize(bn["t"].cpu(), bn["N"].cpu(), bn["a"].cpu(), bn["mask"].cpu(),
                                  bn["sdN"].cpu(), bn["sda"].cpu(), bn["age"].cpu())
            T2 = trend_forecast(p2, bn["dt"].cpu()).numpy()
            last2 = np.concatenate([np.repeat(bn["N"][:, -1:].cpu().numpy(), H, 1),
                                    np.repeat(bn["a"][:, -1:].cpu().numpy(), H, 1)], 1)
            Nd2, _ = damped({k: v.numpy() for k, v in p2.items()}, dt[te], best["N"])
            _, ad2 = damped({k: v.numpy() for k, v in p2.items()}, dt[te], best["a"])
            X2 = np.concatenate([f2.numpy(), T2], 1)
            hg2 = np.stack([m.predict(X2) for m in hgb_models], 1)
            noise[key] = {"persistence": last2, "linear_trend": T2,
                          "damped_trend": np.concatenate([np.clip(Nd2, 0, None), np.clip(ad2, 0, 1)], 1),
                          "gradient_boosting": hg2}
    for name, p in preds.items():
        extra = {f"noise_{k}": v[name] for k, v in noise.items()}
        np.savez_compressed(out_dir / f"{name}_s0.npz", pred=p, **extra)
    return {"damped_phi": best,
            "val_mae": {k: mae(v[va], y[va]).round(4).tolist() for k, v in preds.items()}}


# ---------------------------------------------------------------- main
def main():
    C.ensure_dirs()
    t_start = time.time()
    samples, meas = load_tables()
    emb = {"task": np.load(C.MEAS_DIR / "task_embeddings.npy"),
           "imagenet": np.load(C.EMB_DIR / "imagenet_resnet18.npy")}
    tr_mask = meas.split == "train"
    a_ratio = float(np.mean(np.abs(meas.a_meas[tr_mask] - meas.a[tr_mask])) /
                    np.mean(np.abs(meas.N_meas[tr_mask] - meas.N[tr_mask])))
    log = {"a_noise_ratio": a_ratio, "runs": {}, "baselines": {}}
    print(f"a/N measurement-error ratio used for noise injection: {a_ratio:.5f}")

    for source, configs in NEURAL.items():
        arr, meta = build_arrays(samples, meas, source)
        # Leakage ablation: training inputs in-sample, evaluation inputs cross-fitted
        # measurements are identical for val/calib/test (same full model).
        splits = {s: np.where(meta.split.values == s)[0] for s in C.SPLITS}
        out_dir = C.PRED_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)
        meta.to_csv(out_dir / "samples.csv", index=False)
        np.save(out_dir / "targets.npy", arr["y"])
        allb = to_torch(arr, np.arange(len(meta)), DEVICE)
        te = splits["test"]
        bte = to_torch(arr, te, DEVICE)

        noise_inputs = None
        if source == "measured":
            noise_inputs = {}
            for sig in TEST_NOISE[1:]:
                for r in range(NOISE_REPEATS):
                    g = torch.Generator(device=DEVICE).manual_seed(1000 + int(sig * 10) * 10 + r)
                    noise_inputs[f"{sig}_{r}"] = add_noise(
                        bte, torch.full((len(te),), sig, device=DEVICE), a_ratio, g)
        if source != "insample":
            log["baselines"][source] = run_baselines(arr, splits, out_dir, noise_inputs)
            print(f"[{source}] baselines done")

        imgs = {k: torch.as_tensor(image_features(e, arr, splits["train"])[0], device=DEVICE)
                for k, e in emb.items()}
        btr, bva = to_torch(arr, splits["train"], DEVICE), to_torch(arr, splits["val"], DEVICE)
        for name, (kind, img_key, aug, use_tta) in configs.items():
            for seed in C.SEEDS:
                t0 = time.time()
                img_all = imgs[img_key] if img_key else None
                n_img = 0 if img_all is None else img_all.shape[1]
                model = Forecaster(kind, 17, 7, n_img=n_img, use_tta=use_tta).to(DEVICE)
                info = train(model, btr, bva,
                             None if img_all is None else img_all[splits["train"]],
                             None if img_all is None else img_all[splits["val"]],
                             seed=seed, augment=aug, a_ratio=a_ratio)
                pred, T, gate = predict(model, allb, img_all)
                extra = {}
                if gate is not None:
                    extra["gate"] = gate
                if noise_inputs is not None and name in ROBUSTNESS:
                    for key, bn in noise_inputs.items():
                        p, _, gt = predict(model, bn, None if img_all is None else img_all[te])
                        extra[f"noise_{key}"] = p
                        if gt is not None:
                            extra[f"gate_noise_{key}"] = gt
                np.savez_compressed(out_dir / f"{name}_s{seed}.npz", pred=pred, trend=T, **extra)
                te_mae = mae(pred[te], arr["y"][te])
                info.update({"test_N_mae": te_mae[:H].round(4).tolist(),
                             "test_a_mae": te_mae[H:].round(5).tolist(),
                             "seconds": round(time.time() - t0, 1)})
                log["runs"][f"{source}/{name}/s{seed}"] = info
                print(f"[{source}] {name:28s} s{seed}  N-MAE {np.mean(te_mae[:H]):.3f}  "
                      f"a-MAE {np.mean(te_mae[H:]) * 100:.3f}%  ep {info['epochs_run']:3d}  "
                      f"{info['seconds']:.0f}s", flush=True)
                json.dump(log, open(C.PRED_DIR / "run_log.json", "w"), indent=1)
    print(f"all runs finished in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
