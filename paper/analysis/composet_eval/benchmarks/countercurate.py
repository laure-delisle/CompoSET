"""CounterCurate loader (Zhang et al., 2024) — Positions + Counting subsets.

Each instance carries an original image, a counterfactual edit (negative
image), an original caption and a counterfactual caption. We restrict to
Positions and Counting since the Attributes subset effectively yields a
different scene.

HF path varies; we try a list and skip cleanly if nothing resolves.
"""

from __future__ import annotations

from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


CANDIDATES = [
    ("HanSolo9682/CounterCurate", None),
    ("xianjun-yang/CounterCurate", None),
]
KEEP_SUBSETS = {"position", "positions", "counting", "count"}


class CounterCurateLoader(BenchmarkLoader):
    name = "countercurate"
    same_scene = "approx"
    has_image_pair = True
    has_text_pair = True

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        ds = None
        for hf_name, split in CANDIDATES:
            try:
                ds = load_dataset(hf_name, split=split, streaming=True) if split \
                    else load_dataset(hf_name, streaming=True)
                print(f"[countercurate] loaded from {hf_name}")
                break
            except Exception as e:
                print(f"[countercurate] try {hf_name}: {repr(e)[:100]}")
        if ds is None:
            print("[countercurate] WARN: no HF path resolved; loader is a stub.")
            return

        splits = list(ds.keys()) if hasattr(ds, "keys") else [None]
        n = 0
        for split in splits:
            sub = ds[split] if split is not None else ds
            for i, ex in enumerate(sub):
                subset = str(ex.get("subset", "")).lower() or str(ex.get("type", "")).lower()
                if subset and not any(k in subset for k in KEEP_SUBSETS):
                    continue
                img_a = ex.get("image") or ex.get("orig_image")
                img_b = ex.get("counterfactual_image") or ex.get("negative_image")
                cap_a = ex.get("caption") or ex.get("orig_caption")
                cap_b = ex.get("counterfactual_caption") or ex.get("negative_caption")
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
                    extras={"split": split, "row": i, "subset": subset},
                )
                n += 1
                if limit is not None and n >= limit:
                    return
