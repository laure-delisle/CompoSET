"""NaturalBench — batched runner with streaming generator.

Each row: 2 images x 2 yes/no questions. Per question, identify the
"Yes" image (positive). T2I only — i2t_score and group_score are NaN.

Memory-bounded refactor (2026-04-26):
  - Pass 1: iterate the dataset, collect metadata only (row_idx, q_idx,
    pos_image_field, question text). No PIL held.
  - Pass 2: re-iterate the dataset with a generator that yields the
    (pos_img, neg_img) pair per question and lets encode_images stream
    them in batches.
  - Score lookup is by index into the encoded image embedding tensor,
    aligned with the generator's emission order.

Fixes silent SIGKILL on the previous version which built two parallel
PIL lists (the HF dataset retains a copy of every PIL it returned, plus
we held our own .convert("RGB") copies → ~24 GB just for PIL on the full
1900-row split, OOM-killed pre-allocation).
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


NATURALBENCH_HF = "BaiqiL/NaturalBench"


def _yn(s) -> str:
    """Normalize a yes/no answer to 'y' / 'n' / '?' (unknown)."""
    if s is None:
        return "?"
    c = str(s).strip().lower()[:1]
    if c == "y":
        return "y"
    if c == "n":
        return "n"
    return "?"


def _iter_questions():
    """Yield (row_idx, q_idx, question_text, question_type, pos_field, neg_field).

    pos_field/neg_field are the strings "Image_0" or "Image_1" — i.e. which
    field in the row holds the positive / negative image. We carry these as
    strings (not the PIL itself) so we can re-resolve them in pass 2 without
    holding decoded PIL across the whole dataset.
    """
    try:
        ds = load_dataset(NATURALBENCH_HF, split="train")
    except Exception as e:
        print(f"[naturalbench] WARN: load failed: {repr(e)[:160]}")
        return
    for i, ex in enumerate(ds):
        if not (ex.get("Image_0") and ex.get("Image_1")):
            continue
        for k in (0, 1):
            q = ex.get(f"Question_{k}")
            a0 = _yn(ex.get(f"Image_0_Question_{k}"))
            a1 = _yn(ex.get(f"Image_1_Question_{k}"))
            if q is None or a0 == "?" or a1 == "?":
                continue
            if a0 == "y" and a1 == "n":
                pos_field, neg_field = "Image_0", "Image_1"
            elif a1 == "y" and a0 == "n":
                pos_field, neg_field = "Image_1", "Image_0"
            else:
                continue
            yield (i, k, q, ex.get("Question_Type"),
                   ex.get("Index", i), pos_field, neg_field)


def run_naturalbench(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    # Pass 1: metadata only.
    meta = []
    for row_idx, q_idx, q, q_type, ex_index, pos_field, neg_field in _iter_questions():
        meta.append({
            "row": row_idx,
            "question_idx": q_idx,
            "question": q,
            "question_type": q_type,
            "id": f"{ex_index}__q{q_idx}",
            "pos_field": pos_field,
            "neg_field": neg_field,
        })
    if not meta:
        return pd.DataFrame()

    n_questions = len(meta)
    # Each question contributes 2 images (pos, neg) in stable order.
    print(f"NaturalBench [{scorer.model_key}]: {n_questions} questions → "
          f"{2 * n_questions} images, {n_questions} texts (streaming)")

    # Pass 2: image generator. Re-iterate dataset; for each question emit
    # (pos_img, neg_img) so the embedding indices align as
    # (q0_pos, q0_neg, q1_pos, q1_neg, ...).
    def img_gen():
        ds = load_dataset(NATURALBENCH_HF, split="train")
        # Build a lookup of (row_idx, q_idx) → (pos_field, neg_field) so we
        # only emit for the questions in `meta` (and in the same order).
        meta_iter = iter(meta)
        try:
            current = next(meta_iter)
        except StopIteration:
            return
        for i, ex in enumerate(ds):
            while current is not None and current["row"] == i:
                pf = current["pos_field"]
                nf = current["neg_field"]
                yield ex[pf].convert("RGB")
                yield ex[nf].convert("RGB")
                try:
                    current = next(meta_iter)
                except StopIteration:
                    current = None
            if current is None:
                return

    img_emb = scorer.encode_images(img_gen(), batch_size=img_bs)
    txt_emb = scorer.encode_texts([m["question"] for m in meta], batch_size=txt_bs)

    rows = []
    for k, m in enumerate(meta):
        s_pos = float((img_emb[2 * k] @ txt_emb[k]).item())
        s_neg = float((img_emb[2 * k + 1] @ txt_emb[k]).item())
        rows.append({
            "id": m["id"],
            "row": m["row"],
            "question_idx": m["question_idx"],
            "question": m["question"],
            "question_type": m["question_type"],
            "sim_pos": s_pos,
            "sim_neg": s_neg,
            "i2t_score": float("nan"),
            "t2i_score": int(s_pos > s_neg),
            "group_score": float("nan"),
        })
    return pd.DataFrame(rows)
