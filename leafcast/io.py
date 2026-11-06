"""Dataset indexing and mask-derived phenotypes."""

import re
from datetime import datetime

import numpy as np
from PIL import Image

from . import config as C

_TS = re.compile(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})-")


def parse_timestamp(filename):
    m = _TS.search(filename)
    if m is None:
        raise ValueError(f"cannot parse timestamp from {filename}")
    return datetime(*map(int, m.groups()))


def list_frames():
    """Return one record per RGB frame that has a matching label mask."""
    frames = []
    for gdir in sorted(p for p in C.RAW_DIR.iterdir() if p.is_dir()):
        for rdir in sorted(p for p in gdir.iterdir() if p.is_dir()):
            for img in sorted(rdir.glob("*.png")):
                mask = C.MASK_DIR / gdir.name / rdir.name / img.name
                if not mask.exists():
                    continue
                frames.append({
                    "genotype": gdir.name,
                    "plant": f"{gdir.name}|{rdir.name}",
                    "filename": img.name,
                    "rgb_path": str(img),
                    "mask_path": str(mask),
                    "time": parse_timestamp(img.name).isoformat(),
                })
    return frames


def load_palette():
    """31 leaf colours of the PhenoLeaf-TS annotation palette."""
    import openpyxl
    ws = openpyxl.load_workbook(C.PALETTE_XLSX, read_only=True).active
    colours = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[1] is None:
            continue
        rgb = tuple(int(v) for v in str(row[1]).strip("() ").split(","))
        colours.append(rgb)
    return np.array(colours, dtype=np.int64)


def mask_phenotypes(mask_path, palette, min_pixels=10):
    """Leaf count, projected area and fraction of off-palette foreground pixels.

    A leaf is counted when its palette colour covers at least ``min_pixels``
    pixels. Area is the union of all palette-coloured pixels, normalised by the
    image area.
    """
    m = np.asarray(Image.open(mask_path).convert("RGB")).astype(np.int64)
    code = (m[..., 0] << 16) | (m[..., 1] << 8) | m[..., 2]
    pal = (palette[:, 0] << 16) | (palette[:, 1] << 8) | palette[:, 2]
    fg = code != 0
    on_pal = np.isin(code, pal)
    ids, counts = np.unique(code[on_pal], return_counts=True)
    n_leaves = int(np.sum(counts >= min_pixels))
    area_px = int(on_pal.sum())
    off = int((fg & ~on_pal).sum())
    total = code.size
    return {
        "N": n_leaves,
        "area_px": area_px,
        "a": area_px / total,
        "off_palette_frac": off / max(int(fg.sum()), 1),
        "height": int(m.shape[0]),
        "width": int(m.shape[1]),
    }
