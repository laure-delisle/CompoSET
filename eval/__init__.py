"""CompoSET evaluation harness.

Public API:
    from eval import VLMScorer, run_composet, load_composet
"""
from .benchmark import TIERS, run_composet
from .loaders.composet import load_composet
from .models import VLMScorer

__all__ = ["TIERS", "VLMScorer", "load_composet", "run_composet"]
