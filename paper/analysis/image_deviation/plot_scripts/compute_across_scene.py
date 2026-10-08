"""Embed CompoSET base images and compute all pairwise (base-A vs base-B)
DINOv2 + CLIP image-image cosines. Output: across_scene_base_pairs.csv.

Companion to compute_metrics.py. Runs in ~30 s on GPU.
"""
from __future__ import annotations
import os
from huggingface_hub import snapshot_download
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

OUT = Path(__file__).resolve().parents[1]
HF_DIR = Path(os.environ.get("COMPOSET_DATA_DIR") or snapshot_download("CompoSET/CompoSET", repo_type="dataset"))
SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device={DEVICE}")


def load_scene_ids() -> list[str]:
    df = pd.read_parquet(HF_DIR / "variations.parquet")
    return sorted(df["scene_id"].unique().tolist())


def embed_all(scene_ids: list[str]):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dmdl = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cmdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()

    dino_feats, clip_feats = [], []
    for sid in tqdm(scene_ids, desc="embedding bases"):
        p = HF_DIR / "images" / f"{sid}_base.png"
        img = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        with torch.no_grad():
            di = dproc(images=[img], return_tensors="pt").to(DEVICE)
            d = dmdl(**di).last_hidden_state[:, 0]
            d = F.normalize(d, dim=-1).cpu().numpy()[0]
            ci = cproc(images=[img], return_tensors="pt").to(DEVICE)
            c = cmdl.get_image_features(**ci)
            if hasattr(c, "pooler_output"):
                c = c.pooler_output
            elif hasattr(c, "last_hidden_state"):
                c = c.last_hidden_state[:, 0]
            c = F.normalize(c, dim=-1).cpu().numpy()[0]
        dino_feats.append(d)
        clip_feats.append(c)
    return np.stack(dino_feats), np.stack(clip_feats)


def main():
    out_csv = OUT / "across_scene_base_pairs.csv"
    scene_ids = load_scene_ids()
    print(f"[load] {len(scene_ids)} scenes")
    D, C = embed_all(scene_ids)
    # Full NxN cosine matrices (already L2-normalized)
    dm = D @ D.T
    cm = C @ C.T
    rows = []
    n = len(scene_ids)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "scene_a": scene_ids[i],
                "scene_b": scene_ids[j],
                "dinov2_cos": float(dm[i, j]),
                "clip_image_cos": float(cm[i, j]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} n_pairs={len(df)}")
    print(df[["dinov2_cos", "clip_image_cos"]].describe())


if __name__ == "__main__":
    main()
