"""Abstract base class for benchmark loaders."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from ..schema import BenchmarkPair, SameScene


CACHE_ROOT = Path(
    os.environ.get(
        "COMPOSET_BENCHMARKS_CACHE",
        str(Path(__file__).resolve().parents[2] / ".benchmarks_cache"),
    )
)


class BenchmarkLoader(ABC):
    """Subclasses must set ``name`` and the metadata defaults below."""

    name: str = ""
    same_scene: SameScene = "na"
    has_image_pair: bool = False
    has_text_pair: bool = False

    @abstractmethod
    def iter_pairs(
        self,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[BenchmarkPair]:
        """Yield BenchmarkPair objects. Must be deterministic given ``seed``."""

    def cache_dir(self) -> Path:
        d = CACHE_ROOT / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def images_dir(self) -> Path:
        d = self.cache_dir() / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d
