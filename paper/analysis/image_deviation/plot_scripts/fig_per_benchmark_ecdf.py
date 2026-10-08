"""ECDF of frac_pixels_affected by benchmark — compoSET as hero, priors muted.

Mirrors the text-deviation ECDF (text_deviation/plot_scripts/fig_per_benchmark_ecdf_medium.py)
so the two figures can be read side-by-side.

frac_pixels_affected uses the blur+threshold pipeline already shipped in raw_pairs.csv.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
RAW_CSV = OUT / "raw_pairs.csv"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_CSV, low_memory=False)


def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


HERO = "#C44E52"
WHATSUP = "#56B4E9"
priors = ["winoground", "bivlc", "mmvp", "colorswap"]
prior_colors = ["#0a3d62", "#1f6091", "#3d7eb5", "#7aa9ce"]

fig, ax = plt.subplots(figsize=(5.5, 4.0))

medians = []
for b, c in zip(priors, prior_colors):
    sub = df[df["benchmark"] == b]["frac_pixels_affected"].values
    if len(sub) == 0:
        continue
    x, y = ecdf(sub)
    ax.plot(x, y, color=c, lw=1.4, alpha=0.95, label=f"{b}  (n={len(sub)})")
    medians.append((b, np.median(sub), c))

ws = df[df["benchmark"] == "whatsup"]["frac_pixels_affected"].values
x, y = ecdf(ws)
ax.plot(x, y, color=WHATSUP, lw=2.2, ls="--", label=f"whatsup  (n={len(ws)})", zorder=8)
medians.append(("whatsup", np.median(ws), WHATSUP))

hero_sub = df[df["benchmark"] == "compoSET"]["frac_pixels_affected"].values
x, y = ecdf(hero_sub)
ax.plot(x, y, color=HERO, lw=3.0, label=f"compoSET  (n={len(hero_sub)})", zorder=10)
medians.append(("compoSET", np.median(hero_sub), HERO))

for b, m, c in medians:
    ax.axvline(m, color=c, lw=0.6, ls=":", alpha=0.4, ymax=0.5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Fraction of pixels affected", fontsize=15)
ax.set_ylabel("Cumulative fraction of pairs", fontsize=15)
ax.tick_params(axis="both", labelsize=13)
ax.set_title("Image-edit fraction across benchmarks",
             fontsize=16, loc="left", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5)
leg = ax.legend(loc="center right", bbox_to_anchor=(0.85, 0.5),
                fontsize=12, frameon=True,
                facecolor="white", edgecolor="#cccccc", framealpha=0.95)
leg.get_frame().set_linewidth(0.6)

ax.annotate("smaller image edits →\nmore controlled",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=12, color="#555", style="italic")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"fig_per_benchmark_ecdf.{ext}",
                bbox_inches="tight", dpi=150 if ext == "png" else None)
print("saved fig_per_benchmark_ecdf.{pdf,png}")
