"""Hero + small-multiples scatter: text vs image deviation, compoSET=medium."""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from _common import load_long, pivot_wide, IMG_DEV_CSV, save_both

text = load_long()
text_wide = pivot_wide(text, ["frac_tokens_changed"])
cs_med = text_wide[(text_wide["benchmark"] == "compoSET") & (text_wide["density_level"] == "medium")]
priors = text_wide[text_wide["benchmark"] != "compoSET"]
text_join = pd.concat([cs_med, priors], ignore_index=True)
text_join["text_dev"] = text_join["frac_tokens_changed"]

img = pd.read_csv(IMG_DEV_CSV)[["benchmark", "pair_id", "frac_pixels_affected"]]
joined = text_join.merge(img, on=["benchmark", "pair_id"], how="inner")

HERO_COLOR = "#d62728"
PRIOR_COLOR = "#3a6ea5"
PRIOR_BMS = ["winoground", "bivlc", "whatsup", "sugarcrepe", "mmvp", "colorswap"]


def panel(ax, sub, color, title, hero=False):
    if len(sub) == 0:
        ax.text(0.5, 0.5, "no overlap with\nimage_deviation",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#aaa")
    else:
        ax.hexbin(sub["frac_pixels_affected"], sub["text_dev"],
                  gridsize=30 if hero else 16, cmap="Reds" if hero else "Blues",
                  mincnt=1, extent=(0, 1, 0, 1), linewidths=0)
        mx = sub["frac_pixels_affected"].median()
        my = sub["text_dev"].median()
        ax.axvline(mx, color=color, lw=0.8, alpha=0.6, ls="--")
        ax.axhline(my, color=color, lw=0.8, alpha=0.6, ls="--")
        ax.plot(mx, my, marker="o", ms=6, color=color,
                mec="white", mew=1.0, zorder=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}  (n={len(sub)})", fontsize=10, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 6, height_ratios=[2.2, 1], hspace=0.45, wspace=0.35)

ax_hero = fig.add_subplot(gs[0, :])
panel(ax_hero, joined[joined["benchmark"] == "compoSET"], HERO_COLOR,
      "compoSET-medium", hero=True)
ax_hero.set_xlabel("frac_pixels_affected (image deviation)", fontsize=11)
ax_hero.set_ylabel("frac_tokens_changed (text deviation)", fontsize=11)
ax_hero.set_title("compoSET-medium  (n={})".format(
    len(joined[joined["benchmark"] == "compoSET"])),
    fontsize=13, loc="left", color=HERO_COLOR, weight="bold")

for i, b in enumerate(PRIOR_BMS):
    ax = fig.add_subplot(gs[1, i])
    panel(ax, joined[joined["benchmark"] == b], PRIOR_COLOR, b, hero=False)
    if i == 0:
        ax.set_ylabel("text dev", fontsize=9)
    if i == len(PRIOR_BMS) // 2:
        ax.set_xlabel("image dev (pixels affected)", fontsize=9)

fig.suptitle("Text vs image deviation per pair — compoSET-medium (top) vs prior benchmarks (bottom)",
             fontsize=12, y=0.99, x=0.06, ha="left")

save_both(fig, "fig_main_text_vs_image_deviation_scatter_v2_medium")
print("saved fig_main_text_vs_image_deviation_scatter_v2_medium.{pdf,png}")
