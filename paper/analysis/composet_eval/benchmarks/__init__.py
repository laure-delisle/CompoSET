"""Benchmark loaders. Each module exposes a single Loader class.

The ``LOADERS`` mapping is the single source of truth for "which benchmarks
exist" — used by the CLI driver, the smoke tests, and the inventory dump.
"""

from .base import BenchmarkLoader  # noqa: F401

# Tier 1 — ported from compo_eval/, regression-gated.
from .winoground import WinogroundLoader
from .bivlc import BiVLCLoader

# Tier 2 — same-scene benchmarks (load-bearing for image-deviation).
from .whatsup import WhatsUpLoader
from .spec import SPECLoader
from .eqben import EqBenLoader
from .colorbench import ColorBenchLoader
from .dreamsim import DreamSimLoader
from .countercurate import CounterCurateLoader

# Tier 3 — different-scene + text-only.
from .sugarcrepe import SugarCrepeLoader
from .colorswap import ColorSwapLoader
from .naturalbench import NaturalBenchLoader
from .mmvp import MMVPLoader
from .cola import ColaLoader


LOADERS: dict[str, type[BenchmarkLoader]] = {
    # Tier 1
    "winoground": WinogroundLoader,
    "bivlc": BiVLCLoader,
    # Tier 2
    "whatsup": WhatsUpLoader,
    "spec": SPECLoader,
    "eqben": EqBenLoader,
    "colorbench": ColorBenchLoader,
    "dreamsim": DreamSimLoader,
    "countercurate": CounterCurateLoader,
    # Tier 3
    "sugarcrepe": SugarCrepeLoader,
    "colorswap": ColorSwapLoader,
    "naturalbench": NaturalBenchLoader,
    "mmvp": MMVPLoader,
    "cola": ColaLoader,
}
