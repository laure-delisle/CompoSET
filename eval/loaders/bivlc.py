"""BiVLC — batched runner with streaming generator.

2933 instances × 2 images × 2 captions, Winoground-style 2x2 scoring.

Memory-bounded refactor (2026-04-26):
  - Pass 1: collect metadata only (caption pairs, type/subtype) — no PIL.
  - Pass 2: re-iterate the cached dataset and yield (image, negative_image)
    per row in stable order; encode_images consumes them in batches.
  - Image embedding indices align as (row0_pos, row0_neg, row1_pos, ...).
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


def run_bivlc(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    # Pass 1: metadata + texts only, no PIL.
    rows_meta = []
    texts = []
    ds = load_dataset("imirandam/BiVLC", split="test")
    for ex in ds:
        rows_meta.append({
            "caption_0": ex["caption"],
            "caption_1": ex["negative_caption"],
            "type": ex["type"],
            "subtype": ex["subtype"],
        })
        texts.append(ex["caption"])
        texts.append(ex["negative_caption"])

    print(f"BiVLC [{scorer.model_key}]: {len(rows_meta)} pairs → "
          f"{2 * len(rows_meta)} images, {len(texts)} texts (streaming)")

    def img_gen():
        # Re-iterate the cached dataset; yield (pos_img, neg_img) per row.
        ds2 = load_dataset("imirandam/BiVLC", split="test")
        for ex in ds2:
            yield ex["image"].convert("RGB")
            yield ex["negative_image"].convert("RGB")

    img_emb = scorer.encode_images(img_gen(), batch_size=img_bs)
    txt_emb = scorer.encode_texts(texts, batch_size=txt_bs)

    rows = []
    for k, m in enumerate(rows_meta):
        sim_00 = float((img_emb[2 * k]     @ txt_emb[2 * k]).item())
        sim_01 = float((img_emb[2 * k]     @ txt_emb[2 * k + 1]).item())
        sim_10 = float((img_emb[2 * k + 1] @ txt_emb[2 * k]).item())
        sim_11 = float((img_emb[2 * k + 1] @ txt_emb[2 * k + 1]).item())
        i2t = int(sim_00 > sim_01 and sim_11 > sim_10)
        t2i = int(sim_00 > sim_10 and sim_11 > sim_01)
        grp = int(i2t and t2i)
        rows.append({**m, "sim_00": sim_00, "sim_01": sim_01,
                     "sim_10": sim_10, "sim_11": sim_11,
                     "i2t_score": i2t, "t2i_score": t2i, "group_score": grp})
    return pd.DataFrame(rows)
