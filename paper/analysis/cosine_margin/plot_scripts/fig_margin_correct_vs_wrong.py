"""Headline figure for contribution #5: overlaid margin distributions for
correct vs wrong predictions, faceted by benchmark, one panel per direction."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_margin_correct_vs_wrong.pdf"

# Pool models — one distribution per benchmark per correctness label.
BENCH_ORDER = ["composet", "winoground", "bivlc", "whatsup", "colorswap",
               "mmvp", "naturalbench", "colorbench", "sugarcrepe", "sugarcrepe_pp"]


def main():
    df = pd.read_csv(ROOT / "raw_pairs.csv")
    df = df.dropna(subset=["margin", "decision_correct"])
    df["decision_correct"] = df["decision_correct"].astype(int)

    benchmarks = [b for b in BENCH_ORDER if b in df["benchmark"].unique()]
    directions = ["I2T", "T2I"]

    fig, axes = plt.subplots(
        len(directions), len(benchmarks),
        figsize=(2.4 * len(benchmarks), 2.6 * len(directions)),
        sharex="row", sharey="row",
    )
    if len(directions) == 1:
        axes = np.array([axes])
    if len(benchmarks) == 1:
        axes = axes.reshape(len(directions), 1)

    for i, direction in enumerate(directions):
        for j, bm in enumerate(benchmarks):
            ax = axes[i, j]
            sub = df[(df["benchmark"] == bm) & (df["direction"] == direction)]
            if sub.empty:
                ax.set_axis_off()
                ax.set_title(f"{bm}\n{direction} (n/a)", fontsize=8)
                continue
            corr = sub.loc[sub["decision_correct"] == 1, "margin"].values
            wrng = sub.loc[sub["decision_correct"] == 0, "margin"].values
            lo = np.percentile(np.r_[corr, wrng], 1) if len(corr) + len(wrng) else -0.1
            hi = np.percentile(np.r_[corr, wrng], 99) if len(corr) + len(wrng) else 0.1
            bins = np.linspace(lo, hi, 60)
            if len(wrng):
                ax.hist(wrng, bins=bins, alpha=0.55, color="#d62728",
                        density=True, label=f"wrong (n={len(wrng)})")
            if len(corr):
                ax.hist(corr, bins=bins, alpha=0.55, color="#2ca02c",
                        density=True, label=f"correct (n={len(corr)})")
            ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
            ax.set_title(f"{bm} — {direction}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == len(directions) - 1:
                ax.set_xlabel("cosine margin", fontsize=8)
            if j == 0:
                ax.set_ylabel("density", fontsize=8)
            ax.legend(fontsize=6, loc="upper left", frameon=False)

    fig.suptitle("Cosine-margin distribution: correct vs wrong predictions "
                 "(pooled across 22 dual-encoders)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
