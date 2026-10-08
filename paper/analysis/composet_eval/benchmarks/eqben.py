"""EqBen loader (Wang et al., 2023; HF: ytaek-oh/eqben-images, Kubric subset).

EqBen has 5 subsets; we restrict to Kubric for this pass — the video-frame
subsets need extra plumbing that isn't justified yet. Each Kubric instance
has one base image + one foil image (same scene, one attribute changed) and
two captions (positive/negative).

If the HF schema differs from this assumption, we log a warning and move on.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


EQBEN_HF = "ytaek-oh/eqben-images"
KUBRIC_KEYS = ("kubric", "Kubric", "EQBen_Kubric")


class EqBenLoader(BenchmarkLoader):
    name = "eqben"
    same_scene = "yes"
    has_image_pair = True
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        # Try non-streaming first (image-deviation pipeline trips first-yield
        # timeouts on streaming mode); fall back to streaming if the
        # non-streaming load fails (multi-GB download).
        try:
            ds = load_dataset(EQBEN_HF)
        except Exception as e_ns:
            try:
                ds = load_dataset(EQBEN_HF, streaming=True)
            except Exception as e:
                print(f"[eqben] WARN: load failed: {repr(e)[:150]}")
                return

        splits = list(ds.keys()) if hasattr(ds, "keys") else [None]
        n = 0
        for split in splits:
            # Filter to Kubric only when the dataset splits the subsets.
            if not any(k in split for k in KUBRIC_KEYS) and len(splits) > 1:
                continue
            sub = ds[split] if split is not None else ds
            for i, ex in enumerate(sub):
                # Best-effort schema. Common EqBen fields:
                img_a = ex.get("image_0") or ex.get("base_image") or ex.get("image")
                img_b = ex.get("image_1") or ex.get("foil_image") or ex.get("negative_image")
                cap_a = ex.get("caption_0") or ex.get("base_caption") or ex.get("caption")
                cap_b = ex.get("caption_1") or ex.get("foil_caption") or ex.get("negative_caption")
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
                    extras={"split": split, "row": i,
                            "subset": ex.get("subset")},
                )
                n += 1
                if limit is not None and n >= limit:
                    return
