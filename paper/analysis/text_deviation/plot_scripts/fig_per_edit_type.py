"""Per edit-type bar chart of mean text-deviation, faceted by caption verbosity."""
import matplotlib.pyplot as plt
from _common import load_long, save_both

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
})

df = load_long()
df = df[(df["benchmark"] == "compoSET") & (df["metric_name"] == "frac_tokens_changed")].copy()
df["edit_type"] = df["edit_type"].replace("modify_count", "modify_count")

agg = df.groupby(["edit_type", "density_level"])["value"].agg(["mean", "std", "count"]).reset_index()
edit_types = sorted(agg["edit_type"].unique())
verbosity = ["short", "medium", "long"]

fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
for ax, v in zip(axes, verbosity):
    sub = agg[agg["density_level"] == v].set_index("edit_type").reindex(edit_types)
    ax.bar(range(len(edit_types)), sub["mean"].values,
           yerr=sub["std"].values, color="#1f6091", alpha=0.85)
    ax.set_ylabel(f"{v}\nfraction changed", fontsize=15)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=13)
axes[-1].set_xticks(range(len(edit_types)))
axes[-1].set_xticklabels(edit_types, rotation=45, ha="right", fontsize=13)
fig.suptitle("Fraction of caption tokens changed, by edit type and verbosity",
             fontsize=18, fontweight="bold")
fig.tight_layout()
save_both(fig, "fig_per_edit_type_breakdown")
print("saved fig_per_edit_type_breakdown.{pdf,png}")
