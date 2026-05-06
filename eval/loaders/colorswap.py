"""ColorSwap — batched runner (drop-in for compo_eval/colorswap.py).

stanfordnlp/colorswap test split: 300 quadruplets, 2x2 schema.
Token must be present at ~/.cache/huggingface/token (set by huggingface_hub.login()).
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


COLORSWAP_HF = "stanfordnlp/colorswap"
DEFAULT_SPLIT = "test"


def run_colorswap(
    scorer: VLMScorer,
    split: str = DEFAULT_SPLIT,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    try:
        ds = load_dataset(COLORSWAP_HF, split=split)
    except Exception as e:
        print(f"[colorswap] WARN: load failed (auth?): {repr(e)[:150]}")
        return pd.DataFrame()

    rows_meta = []
    images = []
    texts = []
    for ex in ds:
        img_a = ex.get("image_1")
        img_b = ex.get("image_2")
        cap_a = ex.get("caption_1")
        cap_b = ex.get("caption_2")
        if not (img_a and img_b and cap_a and cap_b):
            continue
        rows_meta.append({
            "id": f"{split}__{ex.get('id')}",
            "split": split,
            "image_source": ex.get("image_source"),
            "caption_source": ex.get("caption_source"),
            "cap_a": cap_a,
            "cap_b": cap_b,
        })
        images.append(img_a.convert("RGB"))
        images.append(img_b.convert("RGB"))
        texts.append(cap_a)
        texts.append(cap_b)

    if not rows_meta:
        return pd.DataFrame()

    print(f"ColorSwap [{scorer.model_key}]: encoding {len(images)} images "
          f"and {len(texts)} texts (batched)")
    img_emb = scorer.encode_images(images, batch_size=img_bs)
    txt_emb = scorer.encode_texts(texts, batch_size=txt_bs)

    rows = []
    for k, m in enumerate(rows_meta):
        sim_aa = float((img_emb[2 * k]     @ txt_emb[2 * k]).item())
        sim_ab = float((img_emb[2 * k]     @ txt_emb[2 * k + 1]).item())
        sim_ba = float((img_emb[2 * k + 1] @ txt_emb[2 * k]).item())
        sim_bb = float((img_emb[2 * k + 1] @ txt_emb[2 * k + 1]).item())
        i2t = int(sim_aa > sim_ab and sim_bb > sim_ba)
        t2i = int(sim_aa > sim_ba and sim_bb > sim_ab)
        grp = int(i2t and t2i)
        rows.append({**m, "sim_aa": sim_aa, "sim_ab": sim_ab,
                     "sim_ba": sim_ba, "sim_bb": sim_bb,
                     "i2t_score": i2t, "t2i_score": t2i, "group_score": grp})
    return pd.DataFrame(rows)
