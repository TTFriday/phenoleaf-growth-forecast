"""Construct rolling-origin forecasting samples.

For every plant, candidate prediction origins are visited chronologically. An
origin is accepted when (i) the preceding 48 h contain at least three frames
spanning at least 36 h, (ii) a frame exists within +/-3 h of each target time
(origin + 24/48/72 h) and (iii) it lies at least 24 h after the previously
accepted origin. The first accepted origin of each plant is flagged as the
"early" origin, which reproduces the single fixed-origin design of a
first-days forecast.

Inputs : outputs/phenotypes/frames.csv, splits/plant_split.json
Output : outputs/samples/samples.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402


def hours(a, b):
    return (a - b).total_seconds() / 3600.0


def build_plant_samples(df):
    t = [datetime.fromisoformat(s) for s in df.time]
    rows = df.reset_index(drop=True)
    out, last_origin = [], None
    for i in range(len(t)):
        hist = [k for k in range(i + 1) if hours(t[i], t[k]) <= C.HISTORY_H + 0.5]
        if len(hist) < C.MIN_HISTORY_FRAMES or hours(t[i], t[hist[0]]) < 36.0:
            continue
        if last_origin is not None and hours(t[i], last_origin) < C.ORIGIN_SPACING_H - 1.0:
            continue
        targets = {}
        for h in C.HORIZONS:
            best, best_d = None, None
            for j in range(i + 1, len(t)):
                d = abs(hours(t[j], t[i]) - h)
                if d <= C.TARGET_TOL_H and (best_d is None or d < best_d):
                    best, best_d = j, d
            if best is None:
                break
            targets[h] = best
        if len(targets) != len(C.HORIZONS):
            continue
        last_origin = t[i]
        out.append({
            "origin_frame": rows.filename[i],
            "origin_time": t[i].isoformat(),
            "days_since_first_frame": round(hours(t[i], t[0]) / 24.0, 3),
            "history_frames": [rows.filename[k] for k in hist],
            "history_dt_h": [round(hours(t[k], t[i]), 3) for k in hist],
            "targets": {str(h): {"frame": rows.filename[j],
                                 "dt_h": round(hours(t[j], t[i]), 3),
                                 "N": int(rows.N[j]), "a": float(rows.a[j])}
                        for h, j in targets.items()},
        })
    for k, s in enumerate(out):
        s["origin_rank"] = k
        s["early"] = k == 0
    return out


def main():
    C.ensure_dirs()
    frames = pd.read_csv(C.PHENO_DIR / "frames.csv")
    frames = frames[frames.valid].sort_values(["plant", "time"])
    split = json.load(open(C.SPLIT_PATH, encoding="utf-8"))["plants"]

    samples = []
    for plant, df in frames.groupby("plant", sort=True):
        for s in build_plant_samples(df):
            s.update({"plant": plant, "genotype": df.genotype.iloc[0], "split": split[plant]})
            samples.append(s)
    for k, s in enumerate(samples):
        s["sample_id"] = k

    json.dump(samples, open(C.SAMPLE_DIR / "samples.json", "w"), indent=1)
    tab = pd.DataFrame([{k: s[k] for k in ("plant", "split", "early", "origin_rank",
                                          "days_since_first_frame")} for s in samples])
    print(f"samples: {len(samples)} from {tab.plant.nunique()} plants")
    print(tab.groupby("split").agg(samples=("plant", "size"), plants=("plant", "nunique"),
                                   early=("early", "sum")))
    print("origins per plant: median", tab.groupby("plant").size().median(),
          "range", tab.groupby("plant").size().min(), "-", tab.groupby("plant").size().max())
    print("history frames: median",
          np.median([len(s["history_frames"]) for s in samples]))


if __name__ == "__main__":
    main()
