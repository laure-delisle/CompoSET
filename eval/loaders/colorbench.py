"""ColorBench Robustness — batched runner with streaming generator.

Pattern: image-only foil. One caption + base image + N color-edited foil
images per group. Score = T2I: sim(base, caption) > sim(foil, caption)?

Memory-bounded refactor (2026-04-26):
  - Pass 1: iterate the dataset, keep metadata only — caption, group key,
    pair indices, no PIL.
  - Pass 2: re-iterate the cached dataset and yield PIL images in the
    pair-emission order so encode_images can stream them in batches.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from datasets import load_dataset

from ..models import VLMScorer


COLORBENCH_HF = "umd-zhou-lab/ColorBench"
ROBUSTNESS_KEYS = ("Robustness", "Robust", "robustness")


def _is_robustness(ex) -> bool:
    return any(
        k in str(ex.get("type", "")) or k in str(ex.get("task", ""))
        for k in ROBUSTNESS_KEYS
    )


def run_colorbench(
    scorer: VLMScorer,
    img_bs: int = 32,
    txt_bs: int = 64,
) -> pd.DataFrame:
    try:
        ds = load_dataset(COLORBENCH_HF, split="test")
    except Exception as e:
        print(f"[colorbench] WARN: load failed: {repr(e)[:160]}")
        return pd.DataFrame()

    # Pass 1: gather Robustness rows' indices grouped by prompt.
    # We carry only row_idx + minimal metadata; no PIL.
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for i, ex in enumerate(ds):
        if not _is_robustness(ex):
            continue
        key = ex.get("prompt") or ex.get("question") or str(ex.get("id"))
        by_prompt[key].append({
            "row_idx": i,
            "id": ex.get("id"),
            "task": ex.get("task"),
            "type": ex.get("type"),
        })

    if not by_prompt:
        print("[colorbench] WARN: no Robustness rows found in test split.")
        return pd.DataFrame()

    # Build pairs (base, foil) using row indices. Assign each unique row a
    # slot in the image-embedding tensor; same caption gets one text slot.
    pairs = []  # (base_slot, foil_slot, cap_slot, base_id, foil_id, task, type, cap)
    row_to_slot: dict[int, int] = {}
    cap_to_slot: dict[str, int] = {}
    captions: list[str] = []

    for cap, members in by_prompt.items():
        if len(members) < 2:
            continue
        if cap not in cap_to_slot:
            cap_to_slot[cap] = len(captions)
            captions.append(cap)
        cap_slot = cap_to_slot[cap]
        base = members[0]
        if base["row_idx"] not in row_to_slot:
            row_to_slot[base["row_idx"]] = len(row_to_slot)
        for foil in members[1:]:
            if foil["row_idx"] not in row_to_slot:
                row_to_slot[foil["row_idx"]] = len(row_to_slot)
            pairs.append({
                "base_slot": row_to_slot[base["row_idx"]],
                "foil_slot": row_to_slot[foil["row_idx"]],
                "cap_slot": cap_slot,
                "base_id": base["id"],
                "foil_id": foil["id"],
                "task": base["task"],
                "type": base["type"],
                "cap": cap,
            })

    if not pairs:
        return pd.DataFrame()

    # Build the inverse map: emission order = sorted by slot, so we yield
    # PIL images in slot order. This lets encode_images return a tensor
    # indexed by slot directly.
    slot_to_row = sorted(row_to_slot.items(), key=lambda kv: kv[1])
    target_rows = {r for r, _ in slot_to_row}

    print(f"ColorBench [{scorer.model_key}]: {len(target_rows)} unique images, "
          f"{len(captions)} unique captions, {len(pairs)} pairs (streaming)")

    def img_gen():
        # Re-iterate the cached dataset; yield only target rows in slot order.
        # Two-pass: first pass collects rows-by-index, second yields by slot
        # order. Cheap because dataset is HF-cached after pass 1.
        wanted = {r: None for r in target_rows}
        ds2 = load_dataset(COLORBENCH_HF, split="test")
        for i, ex in enumerate(ds2):
            if i in wanted:
                wanted[i] = ex["image"]
        for row_idx, _slot in slot_to_row:
            im = wanted.get(row_idx)
            if im is None or not hasattr(im, "convert"):
                # Should not happen given the pass-1 filter, but yield a
                # blank to keep slot alignment if it does.
                from PIL import Image as _PILImage
                yield _PILImage.new("RGB", (224, 224), color=0)
            else:
                yield im.convert("RGB")

    img_emb = scorer.encode_images(img_gen(), batch_size=img_bs)
    txt_emb = scorer.encode_texts(captions, batch_size=txt_bs)

    rows = []
    for p in pairs:
        s_base = float((img_emb[p["base_slot"]] @ txt_emb[p["cap_slot"]]).item())
        s_foil = float((img_emb[p["foil_slot"]] @ txt_emb[p["cap_slot"]]).item())
        t2i = int(s_base > s_foil)
        rows.append({
            "id": f"{p['base_id']}_{p['foil_id']}",
            "task": p["task"],
            "type": p["type"],
            "caption": p["cap"],
            "sim_base": s_base,
            "sim_foil": s_foil,
            "i2t_score": float("nan"),
            "t2i_score": t2i,
            "group_score": float("nan"),
        })
    return pd.DataFrame(rows)
