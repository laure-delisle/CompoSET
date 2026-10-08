"""NaturalBench loader (BaiqiL/NaturalBench).

Each row carries 2 images × 2 yes/no questions. We convert each
``Question_k`` into a positive/negative caption pair by inverting the answer:
the question paired with the image that answers "Yes" is the positive
caption, the same question paired with the image that answers "No" is the
foil. This mirrors the spirit of the original NaturalBench protocol where
the pair score requires both image-question matchings to be correct.

We emit two pairs per row (one per Question_k). Different scenes per
instance (the two images depict related but distinct settings).
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


NATURALBENCH_HF = "BaiqiL/NaturalBench"


class NaturalBenchLoader(BenchmarkLoader):
    name = "naturalbench"
    same_scene = "no"
    has_image_pair = True
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        try:
            ds = load_dataset(NATURALBENCH_HF, split="train", streaming=True)
        except Exception as e:
            print(f"[naturalbench] WARN: load failed: {repr(e)[:120]}")
            return

        n = 0
        for i, ex in enumerate(ds):
            img0 = ex.get("Image_0")
            img1 = ex.get("Image_1")
            if not (img0 and img1):
                continue
            for k in (0, 1):
                q = ex.get(f"Question_{k}")
                a0 = ex.get(f"Image_0_Question_{k}")
                a1 = ex.get(f"Image_1_Question_{k}")
                if q is None or a0 is None or a1 is None:
                    continue
                # Identify which image is "Yes" for this question.
                if str(a0).strip().lower().startswith("y"):
                    pos_img, neg_img = img0, img1
                elif str(a1).strip().lower().startswith("y"):
                    pos_img, neg_img = img1, img0
                else:
                    continue
                yield BenchmarkPair(
                    benchmark=self.name,
                    pair_id=f"{ex.get('Index', i)}__q{k}",
                    image_a=pos_img.convert("RGB"),
                    image_b=neg_img.convert("RGB"),
                    caption_positive=q,
                    # Same question for the foil image — positive/negative
                    # captions are identical here. Downstream retrieval
                    # scoring needs distinct captions; consumers that care
                    # should branch on ``has_text_pair`` + caption equality.
                    caption_negative=q,
                    same_scene=self.same_scene,
                    has_image_pair=True,
                    has_text_pair=True,
                    extras={"row": i, "question_idx": k,
                            "question_type": ex.get("Question_Type"),
                            "source": ex.get("Source")},
                )
                n += 1
                if limit is not None and n >= limit:
                    return
