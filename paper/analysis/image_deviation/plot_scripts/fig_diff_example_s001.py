"""Pixel-diff variants for s001 base vs s001_v01.

Writes four standalone images so we can compare what suppresses the
Gemini regen noise best while preserving the metal tin (the real edit):

  1. pure_diff      — |base-foil|, no blur, no threshold (raw signal)
  2. blur_thresh    — σ=1.0 blur + threshold @ 10/255 (paper recipe)
  3. big_blur       — σ=3.0 blur + threshold @ 10/255 (smooths speckle further)
  4. cc_filtered    — paper recipe + drop connected components <0.05% of pixels
                       (keeps only the largest contiguous edit region)

All use max-over-channels (paper convention) and the same viridis
normalization as alignment_check.py (clip at 50/255).
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[4]
IMG_DIR = Path(os.environ.get("COMPOSET_DATA_DIR") or snapshot_download("CompoSET/CompoSET", repo_type="dataset")) / "images"
OUT_DIR = ROOT / "paper" / "analysis" / "image_deviation" / "figures"

THRESH = 10 / 255.0
CLIP   = 50 / 255.0
SIZE   = 512  # match compute_metrics.py


def load(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_viridis(scalar: np.ndarray) -> np.ndarray:
    norm = np.clip(scalar / CLIP, 0, 1)
    return (cm.viridis(norm)[:, :, :3] * 255).astype(np.uint8)


def save(arr: np.ndarray, name: str, pct: float):
    p = OUT_DIR / f"fig_diff_example_s001_v01_{name}.png"
    Image.fromarray(arr).save(p)
    print(f"  {name:14s}  {pct:6.2f}% pixels lit  -> {p.name}")


def main():
    base = load(IMG_DIR / "s001_base.png")
    foil = load(IMG_DIR / "s001_v01.png")
    h = min(base.shape[0], foil.shape[0]); w = min(base.shape[1], foil.shape[1])
    base = base[:h, :w]; foil = foil[:h, :w]
    per_pix = np.abs(base - foil).max(axis=2)  # paper convention

    # 1. pure pixel diff
    save(to_viridis(per_pix), "pure_diff", (per_pix > 0).mean() * 100)

    # 2. paper recipe
    blur1 = ndimage.gaussian_filter(per_pix, sigma=1.0)
    blur1_t = np.where(blur1 > THRESH, blur1, 0.0)
    save(to_viridis(blur1_t), "blur_thresh", (blur1_t > 0).mean() * 100)

    # 3. heavier blur
    blur3 = ndimage.gaussian_filter(per_pix, sigma=3.0)
    blur3_t = np.where(blur3 > THRESH, blur3, 0.0)
    save(to_viridis(blur3_t), "big_blur", (blur3_t > 0).mean() * 100)

    # 4. connected-component filter on the paper recipe
    mask = blur1 > THRESH
    lab, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        min_size = max(50, int(0.0005 * mask.size))  # 0.05% of pixels or 50 px
        keep = np.zeros_like(mask)
        for i, s in enumerate(sizes, start=1):
            if s >= min_size:
                keep |= (lab == i)
        cc = np.where(keep, blur1, 0.0)
    else:
        cc = blur1_t
    save(to_viridis(cc), "cc_filtered", (cc > 0).mean() * 100)


if __name__ == "__main__":
    main()
