"""SPEC loader (Peng et al., 2024).

SPEC has 6 subsets (Absolute Size, Relative Size, Existence, Counting,
Absolute Position, Relative Position), each with ~500 instances. Each
instance has a query image plus N candidate captions (2-9). We emit
``(image, each_caption)`` style pairs as ``(caption_positive=correct,
caption_negative=foil)`` for every non-correct candidate.

HF path: as of writing, no canonical ``spec-benchmark`` namespace exists on
HF. We try a few likely names and skip cleanly with a warning if none load.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


CANDIDATE_HF_NAMES = [
    "wjpoom/SPEC",
    "spec-benchmark/SPEC",
    "PeixinPENG/SPEC",
]


class SPECLoader(BenchmarkLoader):
    name = "spec"
    same_scene = "yes"
    has_image_pair = False
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        from datasets import load_dataset
        ds = None
        for hf_name in CANDIDATE_HF_NAMES:
            try:
                ds = load_dataset(hf_name, streaming=True)
                print(f"[spec] loaded from {hf_name}")
                break
            except Exception as e:
                print(f"[spec] try {hf_name}: {repr(e)[:100]}")
        if ds is None:
            print("[spec] WARN: no HF path resolved; loader is a stub. "
                  "Update CANDIDATE_HF_NAMES once the canonical path is known.")
            return

        # Walk every split (the 6 subsets are usually splits or configs).
        n = 0
        splits = list(ds.keys()) if hasattr(ds, "keys") else [None]
        for split in splits:
            sub = ds[split] if split is not None else ds
            for i, ex in enumerate(sub):
                # Best-effort field detection — SPEC variants name fields
                # differently. Try common shapes.
                img = ex.get("image") or ex.get("query_image")
                positive = ex.get("caption") or ex.get("query_caption") or ex.get("positive_caption")
                candidates = ex.get("candidates") or ex.get("captions") or ex.get("text_candidates")
                if img is None or positive is None or not candidates:
                    continue
                for j, foil in enumerate(candidates):
                    if foil == positive:
                        continue
                    yield BenchmarkPair(
                        benchmark=self.name,
                        pair_id=f"{split}__{i}__c{j}",
                        image_a=img.convert("RGB") if hasattr(img, "convert") else None,
                        caption_positive=positive,
                        caption_negative=foil,
                        same_scene=self.same_scene,
                        has_image_pair=False,
                        has_text_pair=True,
                        extras={"split": split, "row": i, "candidate_idx": j},
                    )
                    n += 1
                    if limit is not None and n >= limit:
                        return
