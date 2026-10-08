"""ColorBench loader — Robustness slice (umd-zhou-lab/ColorBench).

ColorBench is mostly a VQA-style benchmark; the *Robustness* slice is the
one relevant to dual-encoder retrieval — a base image plus 9 color-edited
foil images, sharing a caption.

The HF dataset doesn't expose a clean "robustness" subset by name; we filter
on the ``task`` field for entries that name color robustness, and group rows
by the ``id``/anchor relation (best-effort). If the schema doesn't match
the robustness assumption, we log clearly and skip.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


COLORBENCH_HF = "umd-zhou-lab/ColorBench"
ROBUSTNESS_KEYS = ("Robustness", "Robust", "robustness")


class ColorBenchLoader(BenchmarkLoader):
    name = "colorbench"
    same_scene = "yes"
    has_image_pair = True
    has_text_pair = False

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        try:
            # Non-streaming: streaming=True trips first-yield timeouts in the
            # image-deviation pipeline; canonical compo_eval/colorbench.py also
            # uses non-streaming.
            ds = load_dataset(COLORBENCH_HF, split="test")
        except Exception as e:
            print(f"[colorbench] WARN: load failed: {repr(e)[:150]}")
            return

        # Filter to robustness rows. (Materialize because we need to group.)
        rows = [ex for ex in ds if any(k in str(ex.get("type", "")) or k in str(ex.get("task", ""))
                                       for k in ROBUSTNESS_KEYS)]
        if not rows:
            print("[colorbench] WARN: no Robustness rows found in test split.")
            return

        # Group by anchor — each anchor has 1 base + 9 foils. Heuristic:
        # rows that share the same ``prompt``/``question`` belong together.
        groups: dict[str, list[dict]] = defaultdict(list)
        for ex in rows:
            key = ex.get("prompt") or ex.get("question") or str(ex.get("id"))
            groups[key].append(ex)

        n = 0
        for key, group in groups.items():
            if len(group) < 2:
                continue
            base = group[0]
            for foil in group[1:]:
                pair_id = f"{base.get('id')}_{foil.get('id')}"
                yield BenchmarkPair(
                    benchmark=self.name,
                    pair_id=pair_id,
                    image_a=base["image"].convert("RGB") if hasattr(base.get("image"), "convert") else None,
                    image_b=foil["image"].convert("RGB") if hasattr(foil.get("image"), "convert") else None,
                    caption_positive=base.get("prompt"),
                    same_scene=self.same_scene,
                    has_image_pair=True,
                    has_text_pair=False,
                    extras={"task": base.get("task"),
                            "type": base.get("type"),
                            "filename_a": base.get("filename"),
                            "filename_b": foil.get("filename")},
                )
                n += 1
                if limit is not None and n >= limit:
                    return
