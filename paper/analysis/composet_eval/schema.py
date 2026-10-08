"""Unified pair-level schema for compositionality benchmarks.

All loaders yield BenchmarkPair objects; all consumers (retrieval scoring,
image-deviation, object-count, text-deviation) read this schema only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from PIL import Image


SameScene = Literal["yes", "no", "approx", "partial", "na"]


@dataclass
class BenchmarkPair:
    """One pair-level item from any benchmark.

    A pair has up to two image sides and up to two captions. Either side may be
    None: ``image_a`` and ``image_b`` both None means a text-only instance;
    ``image_b`` None means a single-image instance (image_a + two captions);
    ``caption_*`` both None means an image-only instance (e.g. NIGHTS triplets).

    Image fields can be set lazily — if ``image_a is None`` but
    ``image_a_path`` is set, consumers should open the path on demand.
    """

    benchmark: str
    pair_id: str

    image_a: Optional[Image.Image] = None
    image_b: Optional[Image.Image] = None
    image_a_path: Optional[str] = None
    image_b_path: Optional[str] = None

    caption_positive: Optional[str] = None
    caption_negative: Optional[str] = None

    scene_id: Optional[str] = None
    variation_id: Optional[str] = None
    edit_types: Optional[list[str]] = None
    verbosity_level: Optional[str] = None

    same_scene: SameScene = "na"
    has_image_pair: bool = False
    has_text_pair: bool = False
    extras: dict = field(default_factory=dict)

    def get_image_a(self) -> Optional[Image.Image]:
        if self.image_a is not None:
            return self.image_a
        if self.image_a_path is not None:
            return Image.open(self.image_a_path).convert("RGB")
        return None

    def get_image_b(self) -> Optional[Image.Image]:
        if self.image_b is not None:
            return self.image_b
        if self.image_b_path is not None:
            return Image.open(self.image_b_path).convert("RGB")
        return None
