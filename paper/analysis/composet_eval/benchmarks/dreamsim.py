"""DreamSim / NIGHTS loader (Fu et al., 2023).

NIGHTS triplets: an anchor image and two foils; the human-judged "more
similar" foil is the positive. ~20K triplets. **No captions** — text fields
are None and ``has_text_pair=False``. Downstream retrieval scoring skips
these; image-deviation and object-count consume them.

HF candidates: ``nights`` (community), ``dreamsim/NIGHTS``, or the original
project bucket. We try a list and skip cleanly if nothing resolves.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


CANDIDATE_HF_NAMES = ["dreamsim/nights", "stephanie/nights", "nights"]


class DreamSimLoader(BenchmarkLoader):
    name = "dreamsim"
    same_scene = "yes"
    has_image_pair = True
    has_text_pair = False

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        ds = None
        for hf_name in CANDIDATE_HF_NAMES:
            try:
                ds = load_dataset(hf_name, streaming=True)
                print(f"[dreamsim] loaded from {hf_name}")
                break
            except Exception as e:
                print(f"[dreamsim] try {hf_name}: {repr(e)[:100]}")
        if ds is None:
            print("[dreamsim] WARN: no HF path resolved; loader is a stub.")
            return

        splits = list(ds.keys()) if hasattr(ds, "keys") else [None]
        n = 0
        for split in splits:
            sub = ds[split] if split is not None else ds
            for i, ex in enumerate(sub):
                anchor = ex.get("anchor") or ex.get("ref")
                pos = ex.get("positive") or ex.get("similar") or ex.get("img1")
                neg = ex.get("negative") or ex.get("dissimilar") or ex.get("img2")
                if not (anchor and pos):
                    continue
                # Emit two pairs per triplet: (anchor, pos) and (anchor, neg)
                for label, foil in (("positive", pos), ("negative", neg)):
                    if foil is None:
                        continue
                    yield BenchmarkPair(
                        benchmark=self.name,
                        pair_id=f"{split}__{i}__{label}",
                        image_a=anchor.convert("RGB") if hasattr(anchor, "convert") else None,
                        image_b=foil.convert("RGB") if hasattr(foil, "convert") else None,
                        caption_positive=None,
                        caption_negative=None,
                        same_scene=self.same_scene,
                        has_image_pair=True,
                        has_text_pair=False,
                        extras={"split": split, "row": i, "foil_label": label},
                    )
                    n += 1
                    if limit is not None and n >= limit:
                        return
