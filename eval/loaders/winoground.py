"""Winoground — batched runner (drop-in for compo_eval/winoground.py).

400 instances, 2x2 (2 images x 2 captions per instance). Winoground protocol:
  I2T: sim_00 > sim_01 AND sim_11 > sim_10
  T2I: sim_00 > sim_10 AND sim_11 > sim_01
  Group: I2T AND T2I
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


def run_winoground(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    ds = load_dataset("facebook/winoground", split="test")
    rows_meta = []
    images = []  # 2 per instance
    texts = []
    for ex in ds:
        rows_meta.append({
            "id": ex["id"],
            "caption_0": ex["caption_0"],
            "caption_1": ex["caption_1"],
            "tag": ex["tag"],
            "collapsed_tag": ex.get("collapsed_tag", ""),
        })
        images.append(ex["image_0"].convert("RGB"))
        images.append(ex["image_1"].convert("RGB"))
        texts.append(ex["caption_0"])
        texts.append(ex["caption_1"])

    print(f"Winoground [{scorer.model_key}]: encoding {len(images)} images "
          f"and {len(texts)} texts (batched)")
    img_emb = scorer.encode_images(images, batch_size=img_bs)
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
