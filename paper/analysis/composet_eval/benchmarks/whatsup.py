"""What'sUp loader (Kamath et al., 2023; HF: ServiceNow/whatsup_all).

The HF release stores each photo as a row with one image and four candidate
captions ("X on/under/left/right of Y"); ``caption_options[0]`` is the
positive caption for that photo. Photos that share the same set of four
captions (frozen) belong to the same 4-photo *set* (one object pair, 4 spatial
arrangements) — we group on that and emit all 6 pairwise image pairs per set.

Pair schema choice (recorded here so it survives refactors):
    For each set of 4 photos, emit C(4,2) = 6 BenchmarkPair objects, each with
    image_a + image_b (same scene, different positions), caption_positive =
    image_a's correct caption, caption_negative = image_b's correct caption.
    same_scene="yes", has_image_pair=True, has_text_pair=True.

This is denser than (base + 3 foils) and matches the standard Winoground-style
2x2 retrieval consumer; the alternative would have been one row per
(base, each-of-3-foils) which loses image-image scoring.

Splits used: ``Controlled_Images_A`` and ``Controlled_Images_B`` (the spatial-
preposition splits). VG_Relation, VG_Attribution, COCO_Order, COCO_QA_*
mix dataset families and are out of scope for this loader.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterator, Optional

from datasets import load_dataset

from ..schema import BenchmarkPair
from .base import BenchmarkLoader


WHATSUP_HF = "ServiceNow/whatsup_all"
WHATSUP_SPLITS = ("Controlled_Images_A", "Controlled_Images_B")


class WhatsUpLoader(BenchmarkLoader):
    name = "whatsup"
    same_scene = "yes"
    has_image_pair = True
    has_text_pair = True

    def __init__(self, splits: tuple[str, ...] = WHATSUP_SPLITS):
        self.splits = splits

    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        n = 0
        for split in self.splits:
            try:
                ds = load_dataset(WHATSUP_HF, split=split)
            except Exception as e:
                print(f"[whatsup] WARN: skip split {split!r}: {e!r}")
                continue

            # Group rows by frozenset(caption_options); each group = 1 set.
            groups: dict[frozenset, list[dict]] = defaultdict(list)
            for i, ex in enumerate(ds):
                key = frozenset(ex["caption_options"])
                groups[key].append({
                    "split": split, "row": i,
                    "image": ex["image_options"],
                    "captions": list(ex["caption_options"]),
                })

            for key, rows in groups.items():
                if len(rows) < 2:
                    continue  # singleton set — no image pair possible
                for a, b in combinations(rows, 2):
                    pair_id = f"{split}__{a['row']}_{b['row']}"
                    yield BenchmarkPair(
                        benchmark=self.name,
                        pair_id=pair_id,
                        image_a=a["image"].convert("RGB"),
                        image_b=b["image"].convert("RGB"),
                        caption_positive=a["captions"][0],
                        caption_negative=b["captions"][0],
                        same_scene=self.same_scene,
                        has_image_pair=True,
                        has_text_pair=True,
                        extras={
                            "split": split,
                            "row_a": a["row"],
                            "row_b": b["row"],
                            "all_captions": a["captions"],
                        },
                    )
                    n += 1
                    if limit is not None and n >= limit:
                        return
