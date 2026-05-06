"""Quantitative analysis for CompoSET results (hf_release schema).

All functions operate on the per-example DataFrame produced by
``benchmark.run_composet`` (or loaded from a saved CSV).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# -- Loading helpers --------------------------------------------------------

def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_all_results(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    return {
        p.stem: load_results(p)
        for p in sorted(results_dir.glob("*.csv"))
    }


# -- Single-model analysis --------------------------------------------------

SCORE_COLS = ["i2t_score", "t2i_score", "group_score"]
TIER_ORDER = ["short", "medium", "long"]


def accuracy_overall(df: pd.DataFrame) -> pd.Series:
    return df[SCORE_COLS].mean()


def accuracy_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("tier")[SCORE_COLS].mean()
    return out.reindex([t for t in TIER_ORDER if t in out.index])


def accuracy_by_edit_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("edit_type")[SCORE_COLS].mean()


def accuracy_by_tier_and_edit_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["tier", "edit_type"])[SCORE_COLS].mean()


def accuracy_by_scene(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("scene_id")[SCORE_COLS].mean()


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    overall = accuracy_overall(df).to_frame("overall").T
    by_tier = accuracy_by_tier(df)
    return pd.concat([overall, by_tier])


# -- Multi-model comparison -------------------------------------------------

def compare_models(
    results: dict[str, pd.DataFrame],
    score_col: str = "i2t_score",
) -> pd.DataFrame:
    parts = {m: df.groupby("tier")[score_col].mean() for m, df in results.items()}
    out = pd.DataFrame(parts)
    present = [t for t in TIER_ORDER if t in out.index]
    out = out.reindex(present)
    out.loc["overall"] = {m: df[score_col].mean() for m, df in results.items()}
    return out


def compare_models_by_edit_type(
    results: dict[str, pd.DataFrame],
    score_col: str = "i2t_score",
) -> pd.DataFrame:
    parts = {m: df.groupby("edit_type")[score_col].mean() for m, df in results.items()}
    return pd.DataFrame(parts)


# -- Failure analysis -------------------------------------------------------

def hard_examples(
    results: dict[str, pd.DataFrame],
    score_col: str = "i2t_score",
) -> pd.DataFrame:
    keys = ["variation_id", "tier"]
    fail_sets = []
    for df in results.values():
        failed = df.loc[df[score_col] < 1, keys]
        fail_sets.append(set(map(tuple, failed.values)))
    common = set.intersection(*fail_sets) if fail_sets else set()
    if not common:
        return pd.DataFrame(columns=keys)
    first_df = next(iter(results.values()))
    mask = first_df.apply(lambda r: (r["variation_id"], r["tier"]) in common, axis=1)
    return first_df.loc[mask, keys + ["edit_type", "cap_base", "cap_var"]].drop_duplicates()


# -- CLI --------------------------------------------------------------------

def print_summary(df: pd.DataFrame, model_name: str = "") -> None:
    header = f"  CompoSET -- {model_name}" if model_name else "  CompoSET"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(summary_table(df).to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print("By edit type:")
    print(accuracy_by_edit_type(df).to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m eval.analyze <results_dir>")
        sys.exit(1)

    all_results = load_all_results(sys.argv[1])
    for model, df in all_results.items():
        print_summary(df, model)

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  Model comparison (i2t_score)")
        print(f"{'=' * 60}")
        print(compare_models(all_results).to_string(float_format=lambda x: f"{x:.3f}"))
        print()
        print("By edit type:")
        print(compare_models_by_edit_type(all_results).to_string(float_format=lambda x: f"{x:.3f}"))
        print("=" * 60)
