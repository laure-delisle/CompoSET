"""ECDF of frac_tokens_changed by benchmark — compoSET=medium as hero, priors muted."""
import matplotlib.pyplot as plt
import numpy as np
from _common import load_long, save_both

df = load_long()
df = df[df["metric_name"] == "frac_tokens_changed"]


def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


HERO = "#C44E52"
WHATSUP = "#56B4E9"
priors = ["winoground", "bivlc", "sugarcrepe", "mmvp", "colorswap"]
prior_colors = ["#0a3d62", "#1f6091", "#3d7eb5", "#7aa9ce", "#b3cde0"]

fig, ax = plt.subplots(figsize=(5.5, 4.0))

series = []
for b, c in zip(priors, prior_colors):
    sub = df[df["benchmark"] == b]["value"].values
    if len(sub) == 0:
        continue
    x, y = ecdf(sub)
    ax.plot(x, y, color=c, lw=1.4, alpha=0.95, label=f"{b}  (n={len(sub)})")
    series.append((b, np.median(sub), c))

ws = df[df["benchmark"] == "whatsup"]["value"].values
x, y = ecdf(ws)
ax.plot(x, y, color=WHATSUP, lw=2.2, ls="--", label=f"whatsup  (n={len(ws)})", zorder=8)
series.append(("whatsup", np.median(ws), WHATSUP))

hero_sub = df[(df["benchmark"] == "compoSET") & (df["density_level"] == "medium")]["value"].values
x, y = ecdf(hero_sub)
ax.plot(x, y, color=HERO, lw=3.0, label=f"compoSET-medium  (n={len(hero_sub)})", zorder=10)
hero_med = np.median(hero_sub)

for b, m, c in series + [("compoSET-medium", hero_med, HERO)]:
    ax.axvline(m, color=c, lw=0.6, ls=":", alpha=0.4, ymax=0.5)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Fraction of tokens changed", fontsize=15)
ax.set_ylabel("Cumulative fraction of pairs", fontsize=15)
ax.tick_params(axis="both", labelsize=13)
ax.set_title("Caption-edit fraction across benchmarks",
             fontsize=16, loc="left", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5)
leg = ax.legend(loc="center right", fontsize=12, frameon=True,
                facecolor="white", edgecolor="#cccccc", framealpha=0.95)
leg.get_frame().set_linewidth(0.6)

ax.annotate("smaller text edits →\nmore controlled",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=12, color="#555", style="italic")

fig.tight_layout()
save_both(fig, "fig_per_benchmark_ecdf_medium")
print("saved fig_per_benchmark_ecdf_medium.{pdf,png}")
