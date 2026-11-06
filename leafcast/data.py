"""Assemble forecasting tensors from samples and per-frame measurements."""

import json

import numpy as np
import pandas as pd
import torch

from . import config as C

SOURCES = {
    # name: (column for N, column for a, use TTA spread)
    "manual": ("N", "a", False),           # Protocol A: annotation-derived history
    "measured": ("N_meas", "a_meas", True),  # Protocol B: cross-fitted measurements
    "insample": ("N_insample", "a_insample", True),  # leakage ablation
}


def load_tables():
    samples = json.load(open(C.SAMPLE_DIR / "samples.json"))
    meas = pd.read_csv(C.MEAS_DIR / "measurements.csv")
    meas["key"] = meas.plant + "|" + meas.filename
    return samples, meas


def build_arrays(samples, meas, source, L=C.MAX_SEQ_LEN):
    colN, cola, use_sd = SOURCES[source]
    lut = meas.set_index("key")
    vN, va = lut[colN].to_dict(), lut[cola].to_dict()
    sdN = lut["N_tta_sd"].to_dict() if use_sd else None
    sda = lut["a_tta_sd"].to_dict() if use_sd else None
    fidx = lut["frame_idx"].to_dict()
    n, H = len(samples), len(C.HORIZONS)
    out = {k: np.zeros((n, L), np.float32) for k in ("t", "N", "a", "sdN", "sda")}
    out["mask"] = np.zeros((n, L), bool)
    out["y"] = np.zeros((n, 2 * H), np.float32)
    out["dt"] = np.zeros((n, H), np.float32)
    out["age"] = np.zeros(n, np.float32)
    out["origin_frame_idx"] = np.zeros(n, np.int64)
    out["first_frame_idx"] = np.zeros(n, np.int64)
    for i, s in enumerate(samples):
        keys = [f"{s['plant']}|{f}" for f in s["history_frames"]][-L:]
        dts = s["history_dt_h"][-L:]
        k = len(keys)
        out["t"][i, L - k:] = dts
        out["N"][i, L - k:] = [vN[q] for q in keys]
        out["a"][i, L - k:] = [va[q] for q in keys]
        if use_sd:
            out["sdN"][i, L - k:] = [sdN[q] for q in keys]
            out["sda"][i, L - k:] = [sda[q] for q in keys]
        out["mask"][i, L - k:] = True
        out["age"][i] = s["days_since_first_frame"]
        out["origin_frame_idx"][i] = fidx[keys[-1]]
        out["first_frame_idx"][i] = fidx[keys[0]]
        for j, h in enumerate(C.HORIZONS):
            tg = s["targets"][str(h)]
            out["y"][i, j] = tg["N"]
            out["y"][i, H + j] = tg["a"]
            out["dt"][i, j] = tg["dt_h"]
    # left padding: copy the first observed value so that padded entries are harmless
    for key in ("t", "N", "a", "sdN", "sda"):
        first = out[key][np.arange(n), out["mask"].argmax(1)]
        out[key] = np.where(out["mask"], out[key], first[:, None])
    meta = pd.DataFrame({"sample_id": [s["sample_id"] for s in samples],
                         "plant": [s["plant"] for s in samples],
                         "genotype": [s["genotype"] for s in samples],
                         "split": [s["split"] for s in samples],
                         "early": [s["early"] for s in samples],
                         "origin_rank": [s["origin_rank"] for s in samples],
                         "age": out["age"]})
    return out, meta


def to_torch(arrs, idx, device):
    keys = ("t", "N", "a", "mask", "sdN", "sda", "age", "dt", "y")
    return {k: torch.as_tensor(arrs[k][idx], device=device) for k in keys}


def image_features(emb, arrs, idx_train, n_comp=16, seed=0):
    """PCA (fitted on training origins) of the origin-frame embedding and of
    the change between the first and the last history frame."""
    from sklearn.decomposition import PCA
    o = emb[arrs["origin_frame_idx"]]
    d = o - emb[arrs["first_frame_idx"]]
    pca_o = PCA(n_comp, random_state=seed).fit(o[idx_train])
    pca_d = PCA(n_comp, random_state=seed).fit(d[idx_train])
    z = np.concatenate([pca_o.transform(o), pca_d.transform(d)], 1).astype(np.float32)
    mu, sd = z[idx_train].mean(0), z[idx_train].std(0) + 1e-6
    return (z - mu) / sd, {"explained_origin": float(pca_o.explained_variance_ratio_.sum()),
                           "explained_change": float(pca_d.explained_variance_ratio_.sum())}
