"""What'sUp — batched runner (drop-in for compo_eval/whatsup.py).

Pattern: Winoground-style 2x2 emitted as C(4,2) image pairs per 4-photo set.
Each pair has 2 images + 2 captions. We need:
  sim_aa = sim(img_a, cap_a)   sim_ab = sim(img_a, cap_b)
  sim_ba = sim(img_b, cap_a)   sim_bb = sim(img_b, cap_b)

Batched approach:
  1. Materialize all unique images and captions across both splits.
  2. Encode each set once.
  3. Per-pair lookup (img_a_idx, img_b_idx, cap_a_idx, cap_b_idx) → 4 sims.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


WHATSUP_HF = "ServiceNow/whatsup_all"
WHATSUP_SPLITS = ("Controlled_Images_A", "Controlled_Images_B")


def run_whatsup(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    pairs = []  # list of (split, img_a, img_b, cap_a, cap_b, row_a, row_b)
    images: dict[int, object] = {}  # id → PIL
    captions: dict[int, str] = {}    # id → str

    def img_id(im):
        for k, v in images.items():
            if v is im:
                return k
        new = len(images)
        images[new] = im
        return new

    def cap_id(s):
        for k, v in captions.items():
            if v == s:
                return k
        new = len(captions)
        captions[new] = s
        return new

    for split in WHATSUP_SPLITS:
        try:
            ds = load_dataset(WHATSUP_HF, split=split)
        except Exception as e:
            print(f"[whatsup/{split}] WARN: {repr(e)[:120]}")
            continue
        groups: dict[frozenset, list[dict]] = defaultdict(list)
        for i, ex in enumerate(ds):
            key = frozenset(ex["caption_options"])
            img = ex["image_options"][0] if isinstance(ex["image_options"], list) \
                  else ex["image_options"]
            groups[key].append({
                "row": i,
                "image": img.convert("RGB"),
                "captions": list(ex["caption_options"]),
            })
        for _, members in groups.items():
            if len(members) < 2:
                continue
            for a, b in combinations(members, 2):
                pairs.append({
                    "split": split,
                    "img_a_idx": img_id(a["image"]),
                    "img_b_idx": img_id(b["image"]),
                    "cap_a_idx": cap_id(a["captions"][0]),
                    "cap_b_idx": cap_id(b["captions"][0]),
                    "row_a": a["row"],
                    "row_b": b["row"],
                    "cap_a": a["captions"][0],
                    "cap_b": b["captions"][0],
                })

    if not pairs:
        return pd.DataFrame()

    print(f"What'sUp [{scorer.model_key}]: encoding "
          f"{len(images)} unique images and {len(captions)} unique captions (batched)")
    img_emb = scorer.encode_images(
        [images[i] for i in range(len(images))], batch_size=img_bs)
    txt_emb = scorer.encode_texts(
        [captions[i] for i in range(len(captions))], batch_size=txt_bs)

    rows = []
    for p in pairs:
        ia, ib = p["img_a_idx"], p["img_b_idx"]
        ca, cb = p["cap_a_idx"], p["cap_b_idx"]
        sim_aa = float((img_emb[ia] @ txt_emb[ca]).item())
        sim_ab = float((img_emb[ia] @ txt_emb[cb]).item())
        sim_ba = float((img_emb[ib] @ txt_emb[ca]).item())
        sim_bb = float((img_emb[ib] @ txt_emb[cb]).item())

        i2t = int(sim_aa > sim_ab and sim_bb > sim_ba)
        t2i = int(sim_aa > sim_ba and sim_bb > sim_ab)
        grp = int(i2t and t2i)
        rows.append({
            "id": f"{p['split']}__{p['row_a']}_{p['row_b']}",
            "split": p["split"],
            "cap_a": p["cap_a"],
            "cap_b": p["cap_b"],
            "sim_aa": sim_aa,
            "sim_ab": sim_ab,
            "sim_ba": sim_ba,
            "sim_bb": sim_bb,
            "i2t_score": i2t,
            "t2i_score": t2i,
            "group_score": grp,
        })
    return pd.DataFrame(rows)
