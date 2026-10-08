"""Winoground loader (Thrush et al., CVPR 2022).

Different-scene image+text pairs. 400 instances; each instance has two images
and two captions. ``image_a`` ↔ ``caption_positive`` (= caption_0),
``image_b`` ↔ ``caption_negative`` (= caption_1).

Iteration order matches HuggingFace ``facebook/winoground`` test split, which
is the same order the precursor ``compo_eval/winoground.py`` consumed; this is
required for the per-row regression gate.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


class WinogroundLoader(BenchmarkLoader):
    name = "winoground"
    same_scene = "no"
    has_image_pair = True
    has_text_pair = True

    def __init__(self, hf_split: str = "test"):
        self.hf_split = hf_split

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        ds = load_dataset("facebook/winoground", split=self.hf_split)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                return
            yield BenchmarkPair(
                benchmark=self.name,
                pair_id=str(ex["id"]),
                image_a=ex["image_0"].convert("RGB"),
                image_b=ex["image_1"].convert("RGB"),
                caption_positive=ex["caption_0"],
                caption_negative=ex["caption_1"],
                same_scene=self.same_scene,
                has_image_pair=True,
                has_text_pair=True,
                extras={
                    "tag": ex.get("tag"),
                    "collapsed_tag": ex.get("collapsed_tag"),
                    "secondary_tag": ex.get("secondary_tag"),
                    "num_main_preds": ex.get("num_main_preds"),
                },
            )
