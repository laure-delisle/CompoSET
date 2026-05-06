"""CompoSET 2x2 evaluation protocol.

For each variation x caption tier:
  given (base_image, var_image) and (base_caption, var_caption), compute four
  image-text similarities and derive Winoground-style I2T, T2I, Group scores.

Convention: base caption matches base image; var caption matches var image.
No partial credit. Chance levels: I2T=25%, T2I=25%, Group~16.67%.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .models import VLMScorer


TIERS = ("short", "medium", "long")


def run_composet(
    scorer: VLMScorer,
    variations: pd.DataFrame,
    image_root: str | Path,
    tiers: tuple[str, ...] = TIERS,
) -> pd.DataFrame:
    """Run the CompoSET 2x2 evaluation over all variations x tiers.

    Args:
        scorer: A VLMScorer instance.
        variations: DataFrame with columns id, scene_id, edit_type, image_base,
            image_var, captions (struct of {short, medium, long: {base, var}}).
        image_root: Directory containing image files referenced in image_base /
            image_var (the paths are relative to this root).
        tiers: Subset of (short, medium, long) to evaluate.

    Returns:
        DataFrame with one row per (variation x tier), columns:
            scene_id, variation_id, tier, edit_type,
            cap_base, cap_var,
            sim_base_cb, sim_base_cv, sim_var_cb, sim_var_cv,
            i2t_score (0/1), t2i_score (0/1), group_score (0/1).
    """
    image_root = Path(image_root)
    rows: list[dict] = []
    total = len(variations) * len(tiers)

    with tqdm(total=total, desc=f"CompoSET [{scorer.model_key}]") as pbar:
        for row in variations.itertuples(index=False):
            base_path = image_root / row.image_base
            var_path = image_root / row.image_var

            for tier in tiers:
                cap = row.captions[tier]
                cap_base, cap_var = cap["base"], cap["var"]

                s_base = scorer.sim(base_path, [cap_base, cap_var])
                s_var = scorer.sim(var_path, [cap_base, cap_var])

                sim_base_cb, sim_base_cv = s_base
                sim_var_cb, sim_var_cv = s_var

                i2t_score = int(sim_base_cb > sim_base_cv
                                and sim_var_cv > sim_var_cb)
                t2i_score = int(sim_base_cb > sim_var_cb
                                and sim_var_cv > sim_base_cv)
                group_score = int(i2t_score and t2i_score)

                rows.append({
                    "scene_id": row.scene_id,
                    "variation_id": row.id,
                    "tier": tier,
                    "edit_type": row.edit_type,
                    "cap_base": cap_base,
                    "cap_var": cap_var,
                    "sim_base_cb": sim_base_cb,
                    "sim_base_cv": sim_base_cv,
                    "sim_var_cb": sim_var_cb,
                    "sim_var_cv": sim_var_cv,
                    "i2t_score": i2t_score,
                    "t2i_score": t2i_score,
                    "group_score": group_score,
                })
                pbar.update(1)

    return pd.DataFrame(rows)
