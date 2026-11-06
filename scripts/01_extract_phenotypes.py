"""Index the dataset and derive per-frame leaf count and projected area.

Output: outputs/phenotypes/frames.csv
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leafcast import config as C  # noqa: E402
from leafcast.io import list_frames, load_palette, mask_phenotypes  # noqa: E402

PALETTE = None


def _init():
    global PALETTE
    PALETTE = load_palette()


def _work(rec):
    out = dict(rec)
    out.update(mask_phenotypes(rec["mask_path"], PALETTE))
    return out


def main():
    C.ensure_dirs()
    frames = list_frames()
    print(f"frames with masks: {len(frames)}")
    with Pool(8, initializer=_init) as pool:
        rows = pool.map(_work, frames, chunksize=64)
    df = pd.DataFrame(rows).sort_values(["plant", "time"]).reset_index(drop=True)
    # Frames containing foreground colours outside the 31-colour palette cannot
    # be mapped to leaf identities and are excluded.
    bad = df["off_palette_frac"] > 0
    print(f"excluded frames (off-palette colours present): {int(bad.sum())}")
    df["valid"] = ~bad
    df.drop(columns=["rgb_path", "mask_path"]).to_csv(C.PHENO_DIR / "frames.csv", index=False)
    v = df[df.valid]
    print(f"valid frames: {len(v)}, plants: {v.plant.nunique()}, genotypes: {v.genotype.nunique()}")
    print(f"N range {v.N.min()}-{v.N.max()}, a range {v.a.min():.4f}-{v.a.max():.4f}")


if __name__ == "__main__":
    main()
