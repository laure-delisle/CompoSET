"""Per-model boxplot of cosine margin for correct vs wrong predictions on
CompoSET. Surfaces models with margin-collapse failure modes.

Boxplot rather than bar — shows the full distribution shape (median, IQR,
whiskers, outliers) instead of just the mean. Colorblind-friendly Okabe-Ito
palette: blue for correct, vermillion for wrong.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_per_model_margin_compoSET.pdf"

# Okabe-Ito colorblind-safe palette.
COLOR_CORRECT = "#0072B2"   # blue
COLOR_WRONG   = "#D55E00"   # vermillion


def main():
    df = pd.read_csv(ROOT / "raw_pairs.csv", low_memory=False)
    cs = df[df["benchmark"] == "composet"].copy()

    # Sort models by I2T gap (correct mean − wrong mean), ascending top→bottom.
    rank = (cs[cs["direction"] == "I2T"]
            .groupby(["model", "decision_correct"])["margin"]
            .mean()
            .unstack()
            .rename(columns={0: "wrong", 1: "correct"}))
    rank["gap"] = rank["correct"] - rank["wrong"]
    models = list(rank.sort_values("gap", ascending=True).index)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)
    for ax, direction in zip(axes, ["I2T", "T2I"]):
        sub = cs[cs["direction"] == direction]
        if sub.empty:
            ax.set_axis_off()
            ax.set_title(f"{direction} (n/a)")
            continue

        # Per model, two boxes: wrong (decision_correct=0) and correct (=1).
        positions, widths = [], 0.36
        wrong_data, correct_data = [], []
        for i, m in enumerate(models):
            g = sub[sub["model"] == m]
            wrong_data.append(g.loc[g["decision_correct"] == 0, "margin"].values)
            correct_data.append(g.loc[g["decision_correct"] == 1, "margin"].values)
            positions.append(i)

        # Wrong (left, vermillion)
        bp_w = ax.boxplot(
            wrong_data, positions=[p - widths / 2 for p in positions],
            widths=widths * 0.9, vert=False,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=COLOR_WRONG, alpha=0.55, edgecolor=COLOR_WRONG),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=COLOR_WRONG),
            capprops=dict(color=COLOR_WRONG),
        )
        # Correct (right, blue)
        bp_c = ax.boxplot(
            correct_data, positions=[p + widths / 2 for p in positions],
            widths=widths * 0.9, vert=False,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=COLOR_CORRECT, alpha=0.55, edgecolor=COLOR_CORRECT),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=COLOR_CORRECT),
            capprops=dict(color=COLOR_CORRECT),
        )

        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(positions)
        ax.set_yticklabels(models, fontsize=7)
        ax.set_xlabel(f"cosine margin — {direction}")
        ax.set_title(f"CompoSET — margin distribution per model ({direction})", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    # Shared legend.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR_CORRECT, alpha=0.55, edgecolor=COLOR_CORRECT, label="correct"),
               Patch(facecolor=COLOR_WRONG,   alpha=0.55, edgecolor=COLOR_WRONG,   label="wrong")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
