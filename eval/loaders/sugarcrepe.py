"""SugarCrepe — batched runner with streaming image generator.

Pattern: text-only foil. Each row = 1 image + 2 captions (positive, negative).
Score = I2T accuracy (sim(img, pos) > sim(img, neg)).

Memory-bounded approach (refactored 2026-04-24 after regression-test OOM):
  - Pass 1: iterate the 5 SugarCrepe subsets, collect metadata only
    (id, subset, pos, neg) — no PIL.
  - Pass 2: build a generator that re-iterates the same datasets and yields
    PIL images one at a time. Pass it to scorer.encode_images, which slices
    via itertools.islice and only holds one batch of PIL at a time.
  - The HF Datasets cache makes the second iteration cheap (no redownload).
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


SUBSETS = {
    "replace_obj": "HuggingFaceM4/SugarCrepe_replace_obj",
    "replace_att": "HuggingFaceM4/SugarCrepe_replace_att",
    "replace_rel": "HuggingFaceM4/SugarCrepe_replace_rel",
    "swap_obj":    "HuggingFaceM4/SugarCrepe_swap_obj",
    "swap_att":    "HuggingFaceM4/SugarCrepe_swap_att",
}


def _iter_rows():
    """Yield (subset_name, hf_name, row_idx, example) for valid rows only.

    Centralizes the filter so pass 1 (metadata) and pass 2 (PIL) iterate
    identically — the index alignment between meta entries and encoded
    image embeddings depends on this.
    """
    for sub, hf_name in SUBSETS.items():
        try:
            ds = load_dataset(hf_name, split="test")
        except Exception as e:
            print(f"[sugarcrepe/{sub}] WARN: {repr(e)[:120]}")
            continue
        for i, ex in enumerate(ds):
            labels = ex.get("tested_labels")
            if not labels or len(labels) < 2:
                continue
            yield sub, hf_name, i, ex


def run_sugarcrepe(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    # Pass 1: metadata only (no PIL).
    meta = []
    for sub, hf_name, i, ex in _iter_rows():
        labels = ex["tested_labels"]
        meta.append({
            "id": f"{sub}__{i}",
            "subset": sub,
            "pos": labels[0],
            "neg": labels[1],
        })
    if not meta:
        return pd.DataFrame()

    print(f"SugarCrepe [{scorer.model_key}]: {len(meta)} pairs "
          f"(streaming images, batch_size={img_bs})")

    # Pass 2: streaming PIL generator (cached dataset, fast re-iteration).
    def img_gen():
        for _, _, _, ex in _iter_rows():
            yield ex["image"].convert("RGB")

    img_emb = scorer.encode_images(img_gen(), batch_size=img_bs)

    flat_texts = []
    for r in meta:
        flat_texts.append(r["pos"])
        flat_texts.append(r["neg"])
    txt_emb = scorer.encode_texts(flat_texts, batch_size=txt_bs)

    rows = []
    for k, r in enumerate(meta):
        sim_pos = float((img_emb[k] @ txt_emb[2 * k]).item())
        sim_neg = float((img_emb[k] @ txt_emb[2 * k + 1]).item())
        rows.append({
            "id": r["id"],
            "subset": r["subset"],
            "pos_caption": r["pos"],
            "neg_caption": r["neg"],
            "sim_pos": sim_pos,
            "sim_neg": sim_neg,
            "i2t_score": int(sim_pos > sim_neg),
            "t2i_score": float("nan"),
            "group_score": float("nan"),
        })
    return pd.DataFrame(rows)
