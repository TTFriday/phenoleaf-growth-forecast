"""Train the image-based measurement models and measure every frame.

Leakage control (cross-fitting): the 190 training plants are divided into five
genotype-stratified folds. Frames of a training plant are measured by a model
that was trained on the other four folds only, so that the measurement error
seen by the forecasters during training resembles the error on unseen plants.
Validation, calibration and test frames are measured by a model trained on all
training plants. For the leakage ablation, the latter model is also applied to
the training frames ("in-sample" measurements).

Outputs:
  outputs/cache/images_160.npy, outputs/cache/frame_index.csv
  outputs/measurement/measurements.csv
  outputs/measurement/task_embeddings.npy      (512-d, cross-fitted)
  outputs/measurement/accuracy.json, training_log.json
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402
from leafcast.measure import predict, train_measurer  # noqa: E402

SIZE = 160
MAX_EPOCHS = 20
N_FOLDS = 5


def _load(path):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def cache_images(frames):
    path = C.CACHE_DIR / f"images_{SIZE}.npy"
    if path.exists():
        return np.load(path)
    paths = [str(C.RAW_DIR / p.split("|")[0] / p.split("|")[1] / f)
             for p, f in zip(frames.plant, frames.filename)]
    with Pool(12) as pool:
        imgs = pool.map(_load, paths, chunksize=64)
    arr = np.stack(imgs)
    np.save(path, arr)
    return arr


def fold_assignment(train_plants, genotype_of):
    folds = {}
    by_g = {}
    for p in sorted(train_plants):
        by_g.setdefault(genotype_of[p], []).append(p)
    k = 0
    for g in sorted(by_g):
        for p in by_g[g]:
            folds[p] = k % N_FOLDS
            k += 1
    return folds


def summarise(df, colN, cola):
    out = {}
    for s, d in df.groupby("split"):
        out[s] = {"frames": int(len(d)),
                  "N_mae": float(np.mean(np.abs(d[colN] - d.N))),
                  "N_rmse": float(np.sqrt(np.mean((d[colN] - d.N) ** 2))),
                  "N_within_1": float(np.mean(np.abs(d[colN] - d.N) <= 1.0)),
                  "a_mae": float(np.mean(np.abs(d[cola] - d.a))),
                  "a_rel_mae": float(np.mean(np.abs(d[cola] - d.a) / d.a)),
                  "N_r": float(np.corrcoef(d[colN], d.N)[0, 1]),
                  "a_r": float(np.corrcoef(d[cola], d.a)[0, 1])}
    return out


def main():
    C.ensure_dirs()
    device = torch.device("cuda")
    frames = pd.read_csv(C.PHENO_DIR / "frames.csv")
    frames = frames[frames.valid].reset_index(drop=True)
    split = json.load(open(C.SPLIT_PATH, encoding="utf-8"))["plants"]
    frames["split"] = frames.plant.map(split)
    frames[["plant", "filename", "genotype", "split", "time", "N", "a"]].to_csv(
        C.CACHE_DIR / "frame_index.csv", index_label="frame_idx")

    t0 = time.time()
    images = torch.from_numpy(cache_images(frames))
    print(f"image cache {tuple(images.shape)} ready in {time.time() - t0:.0f}s")

    N = frames.N.values.astype(np.float32)
    a = frames.a.values.astype(np.float32)
    idx_of = {s: np.where(frames.split.values == s)[0] for s in C.SPLITS}
    log = {}

    # Model trained on all training plants; the validation plants select the epoch.
    print("full model (all training plants)")
    full, hist = train_measurer(images, N, a, idx_of["train"], idx_of["val"],
                                MAX_EPOCHS, device, seed=0)
    best_epoch = int(min(hist, key=lambda r: r["val_N_mae"] / N[idx_of["train"]].std()
                         + r["val_a_mae"] / a[idx_of["train"]].std())["epoch"])
    log["full"] = {"history": hist, "best_epoch": best_epoch}
    print(f"  best epoch {best_epoch}")
    all_idx = np.arange(len(frames))
    pN, pa, sN, sa, emb = predict(full, images, all_idx, full.stats, device)
    frames["N_insample"], frames["a_insample"] = pN, pa
    frames["N_meas"], frames["a_meas"] = pN.copy(), pa.copy()
    frames["N_tta_sd"], frames["a_tta_sd"] = sN, sa
    embedding = emb.copy()
    torch.save(full.state_dict(), C.MEAS_DIR / "measurer_full.pt")

    # Cross-fitted measurements for the training plants.
    genotype_of = dict(zip(frames.plant, frames.genotype))
    folds = fold_assignment(set(frames.plant[frames.split == "train"]), genotype_of)
    frames["fold"] = frames.plant.map(folds).fillna(-1).astype(int)
    for k in range(N_FOLDS):
        tr = np.where((frames.split == "train") & (frames.fold != k))[0]
        te = np.where((frames.split == "train") & (frames.fold == k))[0]
        print(f"fold {k + 1}/{N_FOLDS}: train {len(tr)} frames, measure {len(te)} frames")
        m, _ = train_measurer(images, N, a, tr, None, best_epoch, device, seed=k + 1)
        pN, pa, sN, sa, emb = predict(m, images, te, m.stats, device)
        frames.loc[te, "N_meas"], frames.loc[te, "a_meas"] = pN, pa
        frames.loc[te, "N_tta_sd"], frames.loc[te, "a_tta_sd"] = sN, sa
        embedding[te] = emb
        torch.save(m.state_dict(), C.MEAS_DIR / f"measurer_fold{k}.pt")
        del m
        torch.cuda.empty_cache()

    cols = ["plant", "filename", "genotype", "split", "fold", "time", "N", "a",
            "N_meas", "a_meas", "N_tta_sd", "a_tta_sd", "N_insample", "a_insample"]
    frames[cols].to_csv(C.MEAS_DIR / "measurements.csv", index_label="frame_idx")
    np.save(C.MEAS_DIR / "task_embeddings.npy", embedding.astype(np.float32))
    acc = {"cross_fitted": summarise(frames, "N_meas", "a_meas"),
           "in_sample_full_model": summarise(frames, "N_insample", "a_insample"),
           "best_epoch": best_epoch, "image_size": SIZE, "n_folds": N_FOLDS}
    json.dump(acc, open(C.MEAS_DIR / "accuracy.json", "w"), indent=1)
    json.dump(log, open(C.MEAS_DIR / "training_log.json", "w"), indent=1)
    for kind, tab in acc.items():
        if isinstance(tab, dict):
            print(kind)
            for s, r in tab.items():
                print(f"  {s:6s} N-MAE {r['N_mae']:.3f}  a-MAE {r['a_mae']:.4f}  "
                      f"rel {r['a_rel_mae']:.3f}  r_N {r['N_r']:.3f}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
