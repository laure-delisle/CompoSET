"""ColorSwap loader (Burapacheep et al., 2024).

1,000 quadruplets: each has 2 images and 2 captions where colors are swapped
between two objects. We treat ``image_1`` ↔ ``caption_1`` as positive and
``image_2`` ↔ ``caption_2`` as negative — the standard Winoground-style 2x2.

HF: ``stanfordnlp/colorswap`` (gated; needs HF_TOKEN with access). Skip
cleanly with a warning if access is denied.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


COLORSWAP_HF = "stanfordnlp/colorswap"


class ColorSwapLoader(BenchmarkLoader):
    name = "colorswap"
    same_scene = "approx"
    has_image_pair = True
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        try:
            ds = load_dataset(COLORSWAP_HF)
        except Exception as e:
            print(f"[colorswap] WARN: load failed (gated?): {repr(e)[:120]}")
            return

        splits = list(ds.keys()) if hasattr(ds, "keys") else [None]
        n = 0
        for split in splits:
            sub = ds[split] if split is not None else ds
            for i, ex in enumerate(sub):
                img_a = ex.get("image_1") or ex.get("image_0")
                img_b = ex.get("image_2") or ex.get("image_1")
                cap_a = ex.get("caption_1") or ex.get("caption_0")
                cap_b = ex.get("caption_2") or ex.get("caption_1")
                if not (img_a and img_b and cap_a and cap_b):
                    continue
                yield BenchmarkPair(
                    benchmark=self.name,
                    pair_id=f"{split}__{i}",
                    image_a=img_a.convert("RGB") if hasattr(img_a, "convert") else None,
                    image_b=img_b.convert("RGB") if hasattr(img_b, "convert") else None,
                    caption_positive=cap_a,
                    caption_negative=cap_b,
                    same_scene=self.same_scene,
                    has_image_pair=True,
                    has_text_pair=True,
                    extras={"split": split, "row": i},
                )
                n += 1
                if limit is not None and n >= limit:
                    return
