"""Paths and constants shared by all scripts.

The dataset location is read from the environment variable ``PHENOLEAF_ROOT``
(the folder that contains ``01_raw_dataset`` and ``02_leaf_labels``). If the
variable is not set, ``data/PhenoLeaf-TS_DS`` relative to the repository root
is used.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("PHENOLEAF_ROOT", REPO_ROOT / "data" / "PhenoLeaf-TS_DS"))
RAW_DIR = DATA_ROOT / "01_raw_dataset"
MASK_DIR = DATA_ROOT / "02_leaf_labels"
PALETTE_XLSX = DATA_ROOT / "leaves_unique_colours.xlsx"

OUT = REPO_ROOT / "outputs"
PHENO_DIR = OUT / "phenotypes"
SAMPLE_DIR = OUT / "samples"
SPLIT_PATH = REPO_ROOT / "splits" / "plant_split.json"
CACHE_DIR = OUT / "cache"
MEAS_DIR = OUT / "measurement"
EMB_DIR = OUT / "embeddings"
PRED_DIR = OUT / "predictions"
RESULT_DIR = OUT / "results"
FIG_DIR = OUT / "figures"

HORIZONS = (24, 48, 72)          # forecast horizons (h)
HISTORY_H = 48.0                 # history window length (h)
TARGET_TOL_H = 3.0               # tolerance when matching target frames (h)
ORIGIN_SPACING_H = 24.0          # minimum spacing between rolling origins (h)
MIN_HISTORY_FRAMES = 3
MAX_SEQ_LEN = 16

SPLITS = ("train", "val", "calib", "test")
SEEDS = (0, 1, 2, 3, 4)


def ensure_dirs():
    for d in (PHENO_DIR, SAMPLE_DIR, CACHE_DIR, MEAS_DIR, EMB_DIR, PRED_DIR,
              RESULT_DIR, FIG_DIR, SPLIT_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
