"""ECDF of within-pair CLIP image cosine by benchmark — compoSET as hero.

Companion to fig_per_benchmark_ecdf.py (image-edit fraction) and the text-deviation
ECDF; intended for a side-by-side trio in §4. CompoSET pairs cluster near cos~1 by
construction (same scene, single edit); different-scene benchmarks spread broadly.
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
    sub = df[df["benchmark"] == b]["clip_image_cos"].dropna().values
    if len(sub) == 0:
        continue
    x, y = ecdf(sub)
    ax.plot(x, y, color=c, lw=1.4, alpha=0.95, label=f"{b}  (n={len(sub)})")
    medians.append((b, np.median(sub), c))

ws = df[df["benchmark"] == "whatsup"]["clip_image_cos"].dropna().values
x, y = ecdf(ws)
ax.plot(x, y, color=WHATSUP, lw=2.2, ls="--", label=f"whatsup  (n={len(ws)})", zorder=8)
medians.append(("whatsup", np.median(ws), WHATSUP))

hero_sub = df[df["benchmark"] == "compoSET"]["clip_image_cos"].dropna().values
x, y = ecdf(hero_sub)
ax.plot(x, y, color=HERO, lw=3.0, label=f"compoSET  (n={len(hero_sub)})", zorder=10)
medians.append(("compoSET", np.median(hero_sub), HERO))

for b, m, c in medians:
    ax.axvline(m, color=c, lw=0.6, ls=":", alpha=0.4, ymax=0.5)

ax.set_xlim(0.4, 1.0)
ax.set_ylim(0, 1)
ax.set_xlabel("Within-pair CLIP image cosine", fontsize=15)
ax.set_ylabel("Cumulative fraction of pairs", fontsize=15)
ax.tick_params(axis="both", labelsize=13)
ax.set_title("Within-pair image similarity across benchmarks",
             fontsize=16, loc="left", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5)
leg = ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5),
                fontsize=12, frameon=True,
                facecolor="white", edgecolor="#cccccc", framealpha=0.95)
leg.get_frame().set_linewidth(0.6)

ax.annotate("tighter same-scene pairs →\nmore controlled",
            xy=(0.45, 0.92), xycoords="axes fraction",
            fontsize=12, color="#555", style="italic")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"fig_per_benchmark_ecdf_clip_cos.{ext}",
                bbox_inches="tight", dpi=150 if ext == "png" else None)
print("saved fig_per_benchmark_ecdf_clip_cos.{pdf,png}")
