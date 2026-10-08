"""Violin plots of CompoSET token + embedding metrics by caption verbosity tier."""
import matplotlib.pyplot as plt
from _common import load_long, save_both

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
})

df = load_long()
df = df[df["benchmark"] == "compoSET"]

METRIC_LABELS = {
    "token_levenshtein": "Token Levenshtein",
    "frac_tokens_changed": "Fraction tokens changed",
    "mpnet_cos": "MPNet cosine",
    "minilm_cos": "MiniLM cosine",
    "clip_text_cos": "CLIP text cosine",
}
metrics = list(METRIC_LABELS)
verbosity = ["short", "medium", "long"]

fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
for ax, m in zip(axes, metrics):
    data = [df[(df["metric_name"] == m) & (df["density_level"] == d)]["value"].values
            for d in verbosity]
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(verbosity, fontsize=12)
    ax.set_title(METRIC_LABELS[m], fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
fig.suptitle("CompoSET text-deviation metrics by caption verbosity",
             fontsize=15, fontweight="bold")
fig.tight_layout()
save_both(fig, "fig_per_density_violin")
print("saved fig_per_density_violin.{pdf,png}")
