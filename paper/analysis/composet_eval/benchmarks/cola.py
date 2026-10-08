"""Cola loader (Ray et al., 2023) — multi-object subset only.

Cola has two subsets: a CLEVR-templated synthetic subset (skipped) and a
multi-object real-image subset (210 instances). Each instance has two
images and two captions describing different attribute-object bindings.

HF candidates: search returned nothing under ``cola``; the dataset may be
hosted off-HF (project page on arongil.github.io). We try a couple of
guesses and skip cleanly with a warning if nothing resolves — the loader
is a scaffold pending a confirmed source.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


CANDIDATES = [
    "arijitray1993/COLA",
    "arijitray/cola",
    "ColaBench/cola",
]


class ColaLoader(BenchmarkLoader):
    name = "cola"
    same_scene = "no"
    has_image_pair = True
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        ds = None
        for hf_name in CANDIDATES:
            try:
                ds = load_dataset(hf_name)
                print(f"[cola] loaded from {hf_name}")
                break
            except Exception as e:
                print(f"[cola] try {hf_name}: {repr(e)[:100]}")
        if ds is None:
            print("[cola] WARN: no HF path resolved; loader is a stub.")
            return

        splits = list(ds.keys()) if hasattr(ds, "keys") else [None]
        n = 0
        for split in splits:
            sub = ds[split] if split is not None else ds
            for i, ex in enumerate(sub):
                img_a = ex.get("image_0") or ex.get("image_1") or ex.get("image")
                img_b = ex.get("image_1") or ex.get("image_2") or ex.get("negative_image")
                cap_a = ex.get("caption_0") or ex.get("caption_1") or ex.get("caption")
                cap_b = ex.get("caption_1") or ex.get("caption_2") or ex.get("negative_caption")
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
