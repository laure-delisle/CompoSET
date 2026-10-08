"""Headline figure: text-deviation vs image-deviation per pair.

Joins on (benchmark, pair_id) between this analysis and image_deviation/raw_pairs.csv.
For compoSET we restrict to density_level='short' (comparable to single-caption priors).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from _common import load_long, pivot_wide, IMG_DEV_CSV, save_both

text = load_long()
text_wide = pivot_wide(text, ["frac_tokens_changed", "mpnet_cos"])

# CompoSET: short only. Priors: keep all (density_level == "").
cs_short = text_wide[(text_wide["benchmark"] == "compoSET") & (text_wide["density_level"] == "short")]
priors = text_wide[text_wide["benchmark"] != "compoSET"]
text_join = pd.concat([cs_short, priors], ignore_index=True)
text_join["text_dev"] = text_join["frac_tokens_changed"]
text_join["text_cos_dist"] = 1 - text_join["mpnet_cos"]

img = pd.read_csv(IMG_DEV_CSV)[["benchmark", "pair_id", "frac_pixels_affected"]]
joined = text_join.merge(img, on=["benchmark", "pair_id"], how="inner")
print(f"joined rows per benchmark:\n{joined['benchmark'].value_counts()}")

# Two subplots: x = pixels, y = frac_tokens_changed (left); y = 1 - mpnet_cos (right)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
benches = sorted(joined["benchmark"].unique())
palette = {b: c for b, c in zip(benches, plt.cm.tab10.colors)}
for ax, ycol, ylab in [(axes[0], "text_dev", "frac_tokens_changed"),
                       (axes[1], "text_cos_dist", "1 - mpnet_cos")]:
    for b in benches:
        sub = joined[joined["benchmark"] == b]
        ax.scatter(sub["frac_pixels_affected"], sub[ycol], s=6, alpha=0.4,
                   label=f"{b} (n={len(sub)})", color=palette[b])
    ax.set_xlabel("frac_pixels_affected (image deviation)")
    ax.set_ylabel(ylab + " (text deviation)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
fig.suptitle("Text vs image deviation per pair (compoSET=short density)")
fig.tight_layout()
save_both(fig, "fig_main_text_vs_image_deviation_scatter")
print("saved fig_main_text_vs_image_deviation_scatter.{pdf,png}")
