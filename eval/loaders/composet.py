"""CompoSET loader — pulls the dataset from HuggingFace and returns a
DataFrame plus the local image directory.

The dataset lives at https://huggingface.co/datasets/CompoSET/CompoSET.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = os.environ.get("COMPOSET_HF_REPO", "CompoSET/CompoSET")


def load_composet(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Download (if needed) the CompoSET dataset from HF and return:

    Args:
        repo_id: HuggingFace dataset repo (default: env COMPOSET_HF_REPO or
            "CompoSET/CompoSET").
        cache_dir: Where to materialize the dataset. Default: HF default.
        revision: Optional git revision (branch / tag / commit) on the HF repo.

    Returns:
        (variations_df, image_root) tuple.
            * ``variations_df``: 1,776 rows, schema documented in the dataset
              card. Columns: id, scene_id, edit_type, image_base, image_var,
              captions (struct of {short, medium, long: {base, var}}).
            * ``image_root``: Path to the local checkout root. Image paths in
              ``image_base`` / ``image_var`` are relative to this directory.
    """
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
    )
    local = Path(local_dir)
    df = pd.read_parquet(local / "variations.parquet")
    return df, local


def load_scenes(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load per-scene metadata (scenes.parquet) from the same HF repo."""
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return pd.read_parquet(Path(local_dir) / "scenes.parquet")


def load_scene_captions(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load full per-scene corrected captions (scene_captions.parquet)."""
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return pd.read_parquet(Path(local_dir) / "scene_captions.parquet")
