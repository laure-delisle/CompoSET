"""BiVLC loader (Miranda et al., EMNLP 2024).

Approx-same-scene pairs (the negative image is a generative edit of the
original). 2,933 instances; each carries one (image, caption) ↔
(negative_image, negative_caption).

Iteration order matches HuggingFace ``imirandam/BiVLC`` test split, which is
the order the precursor ``compo_eval/bivlc.py`` consumed; required for the
per-row regression gate.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


class BiVLCLoader(BenchmarkLoader):
    name = "bivlc"
    same_scene = "approx"
    has_image_pair = True
    has_text_pair = True

    def __init__(self, hf_split: str = "test"):
        self.hf_split = hf_split

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        ds = load_dataset("imirandam/BiVLC", split=self.hf_split)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                return
            yield BenchmarkPair(
                benchmark=self.name,
                pair_id=str(i),
                image_a=ex["image"].convert("RGB"),
                image_b=ex["negative_image"].convert("RGB"),
                caption_positive=ex["caption"],
                caption_negative=ex["negative_caption"],
                same_scene=self.same_scene,
                has_image_pair=True,
                has_text_pair=True,
                extras={
                    "type": ex.get("type"),
                    "subtype": ex.get("subtype"),
                },
            )
