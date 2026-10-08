"""Per-benchmark overlaid histogram of frac_tokens_changed.
CompoSET appears as 3 sub-curves (short / medium / long).
"""
import matplotlib.pyplot as plt
import numpy as np
from _common import load_long, save_both

df = load_long()
df = df[df["metric_name"] == "frac_tokens_changed"]

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 1, 41)

labels = []
# CompoSET 3 densities
for density in ["short", "medium", "long"]:
    sub = df[(df["benchmark"] == "compoSET") & (df["density_level"] == density)]
    ax.hist(sub["value"], bins=bins, density=True, alpha=0.45,
            label=f"compoSET-{density} (n={len(sub)})", histtype="stepfilled")
# Priors
for b in ["winoground", "bivlc", "whatsup", "sugarcrepe", "mmvp", "colorswap"]:
    sub = df[df["benchmark"] == b]
    ax.hist(sub["value"], bins=bins, density=True, alpha=0.55,
            label=f"{b} (n={len(sub)})", histtype="step", linewidth=2)

ax.set_xlabel("frac_tokens_changed")
ax.set_ylabel("density")
ax.set_title("Token-level edit fraction by benchmark")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
save_both(fig, "fig_per_benchmark_hist")
print("saved fig_per_benchmark_hist.{pdf,png}")
