"""Pairwise correlation of metrics on CompoSET short slice."""
import matplotlib.pyplot as plt
import numpy as np
from _common import load_long, pivot_wide, save_both

df = load_long()
metrics = ["token_levenshtein", "token_jaccard", "token_jaccard_dist",
           "frac_tokens_changed", "tokens_a_len", "tokens_b_len",
           "mpnet_cos", "mpnet_cos_dist", "minilm_cos", "clip_text_cos"]
wide = pivot_wide(df, metrics)
sub = wide[(wide["benchmark"] == "compoSET") & (wide["density_level"] == "short")]
corr = sub[metrics].corr()

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(metrics)))
ax.set_yticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.set_yticklabels(metrics)
for i in range(len(metrics)):
    for j in range(len(metrics)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="black")
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_title("CompoSET (short): metric correlations")
fig.tight_layout()
save_both(fig, "fig_metric_correlation_heatmap")
print("saved fig_metric_correlation_heatmap.{pdf,png}")
