"""SugarCrepe++ — batched runner (drop-in for compo_eval/sugarcrepe_pp.py).

Each row: 1 image + 3 captions (positive, paraphrased positive, hard-negative).
Headline metric (SugarCrepe++ contribution): both positives must rank above
the negative — penalizes string-matching shortcuts.

i2t_pos1 = sim(img, pos1) > sim(img, neg)
i2t_pos2 = sim(img, pos2) > sim(img, neg)
i2t_score = i2t_pos1 AND i2t_pos2

T2I and Group are NaN (text-only foil).

Requires COCO val 2017 at $COCO_VAL2017_DIR (default
~/.cache/composet/coco_val2017/val2017/).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from ..models import VLMScorer


SCPP_HF = "Aman-J/SugarCrepe_pp"
SUBSETS = ["replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att"]

COCO_VAL2017_DIR = Path(
    os.environ.get(
        "COCO_VAL2017_DIR",
        os.path.expanduser("~/.cache/composet/coco_val2017/val2017"),
    )
)


def run_sugarcrepe_pp(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    if not COCO_VAL2017_DIR.exists():
        print(f"[sugarcrepe_pp] WARN: COCO val 2017 not found at "
              f"{COCO_VAL2017_DIR}.")
        return pd.DataFrame()

    # Pass 1: collect metadata + image paths only (no decoded PIL).
    items = []  # one entry per row
    texts = []  # 3 per row: pos1, pos2, neg
    for sub in SUBSETS:
        try:
            jpath = hf_hub_download(repo_id=SCPP_HF, filename=f"data/{sub}.json",
                                    repo_type="dataset")
        except Exception as e:
            print(f"[sugarcrepe_pp/{sub}] WARN: {repr(e)[:120]}")
            continue
        with open(jpath) as f:
            data = json.load(f)
        for ex in data:
            fname = ex.get("filename")
            pos1 = ex.get("caption")
            pos2 = ex.get("caption2")
            neg = ex.get("negative_caption")
            if not (fname and pos1 and pos2 and neg):
                continue
            img_path = COCO_VAL2017_DIR / fname
            if not img_path.exists():
                continue
            items.append({
                "id": f"{sub}__{ex.get('id')}",
                "subset": sub,
                "filename": fname,
                "img_path": img_path,
                "pos1": pos1, "pos2": pos2, "neg": neg,
            })
            texts.append(pos1)
            texts.append(pos2)
            texts.append(neg)

    if not items:
        return pd.DataFrame()

    print(f"SugarCrepe++ [{scorer.model_key}]: {len(items)} pairs, "
          f"{len(texts)} texts (streaming images, batch_size={img_bs})")

    # Pass 2: stream PIL via path generator — encode_images opens lazily.
    def img_gen():
        for it in items:
            yield Image.open(it["img_path"]).convert("RGB")

    img_emb = scorer.encode_images(img_gen(), batch_size=img_bs)
    txt_emb = scorer.encode_texts(texts, batch_size=txt_bs)

    rows = []
    for k, it in enumerate(items):
        s_pos1 = float((img_emb[k] @ txt_emb[3 * k]).item())
        s_pos2 = float((img_emb[k] @ txt_emb[3 * k + 1]).item())
        s_neg  = float((img_emb[k] @ txt_emb[3 * k + 2]).item())
        i2t_pos1 = int(s_pos1 > s_neg)
        i2t_pos2 = int(s_pos2 > s_neg)
        rows.append({
            "id": it["id"],
            "subset": it["subset"],
            "filename": it["filename"],
            "pos1_caption": it["pos1"],
            "pos2_caption": it["pos2"],
            "neg_caption": it["neg"],
            "sim_pos1": s_pos1,
            "sim_pos2": s_pos2,
            "sim_neg": s_neg,
            "i2t_pos1": i2t_pos1,
            "i2t_pos2": i2t_pos2,
            "i2t_score": int(i2t_pos1 and i2t_pos2),
            "t2i_score": float("nan"),
            "group_score": float("nan"),
        })
    return pd.DataFrame(rows)
