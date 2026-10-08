"""Shared helpers for plot scripts. Reads CSVs only; no recomputation."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
RAW_CSV = OUT / "raw_pairs.csv"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

IMG_DEV_CSV = Path(__file__).resolve().parents[2] / "image_deviation" / "raw_pairs.csv"


def load_long():
    df = pd.read_csv(RAW_CSV, low_memory=False)
    for c in ("density_level", "scene_id", "variation_id", "edit_type"):
        df[c] = df[c].fillna("")
    return df


def pivot_wide(df, metrics):
    """Pivot long-format rows to one-row-per-pair with metric columns.

    pandas pivot_table drops index rows with NaN — fillna upstream in load_long.
    """
    keep = df[df["metric_name"].isin(metrics)].copy()
    for c in ("density_level", "scene_id", "variation_id", "edit_type"):
        keep[c] = keep[c].fillna("")
    keys = ["benchmark", "pair_id", "scene_id", "variation_id", "edit_type", "density_level"]
    wide = keep.pivot_table(index=keys, columns="metric_name", values="value",
                            aggfunc="first").reset_index()
    return wide


def save_both(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=150)
