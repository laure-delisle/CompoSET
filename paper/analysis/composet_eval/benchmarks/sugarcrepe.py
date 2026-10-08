"""SugarCrepe loader (Hsieh et al., 2023).

Text-only foils: each instance has one image and a list of candidate captions
(``tested_labels``); the index of the positive caption is ``true_label`` (it's
an offset into a global label table for the COCO captioning dataset, but
within-row, the convention is that ``tested_labels[0]`` is the positive
caption and ``tested_labels[1]`` is the hard-negative foil).

HF: split per category — replace_obj, replace_att, replace_rel, swap_obj,
swap_att (5 datasets in the HuggingFaceM4 namespace).
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


SUBSETS = {
    "replace_obj": "HuggingFaceM4/SugarCrepe_replace_obj",
    "replace_att": "HuggingFaceM4/SugarCrepe_replace_att",
    "replace_rel": "HuggingFaceM4/SugarCrepe_replace_rel",
    "swap_obj":    "HuggingFaceM4/SugarCrepe_swap_obj",
    "swap_att":    "HuggingFaceM4/SugarCrepe_swap_att",
}


class SugarCrepeLoader(BenchmarkLoader):
    name = "sugarcrepe"
    same_scene = "na"
    has_image_pair = False
    has_text_pair = True

    def __init__(self, subsets: Optional[list[str]] = None):
        self.subsets = subsets or list(SUBSETS.keys())

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        n = 0
        for sub in self.subsets:
            hf_name = SUBSETS.get(sub)
            if hf_name is None:
                continue
            try:
                ds = load_dataset(hf_name, split="test", streaming=True)
            except Exception as e:
                print(f"[sugarcrepe/{sub}] WARN: {repr(e)[:100]}")
                continue
            for i, ex in enumerate(ds):
                labels = ex.get("tested_labels")
                if not labels or len(labels) < 2:
                    continue
                pos, neg = labels[0], labels[1]
                yield BenchmarkPair(
                    benchmark=self.name,
                    pair_id=f"{sub}__{i}",
                    image_a=ex["image"].convert("RGB"),
                    caption_positive=pos,
                    caption_negative=neg,
                    same_scene=self.same_scene,
                    has_image_pair=False,
                    has_text_pair=True,
                    extras={"subset": sub, "row": i,
                            "true_label": ex.get("true_label")},
                )
                n += 1
                if limit is not None and n >= limit:
                    return
