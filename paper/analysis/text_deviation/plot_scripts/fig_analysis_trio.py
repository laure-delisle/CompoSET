"""Combined 1x3 ECDF figure for §4: image-edit fraction, text-edit fraction,
within-pair CLIP image cosine. Shared y-axis. CompoSET as hero.

Pulls image-deviation raw_pairs.csv directly and reuses load_long() for text.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import load_long

ROOT = Path(__file__).resolve().parents[2]
IMG_RAW = ROOT / "image_deviation/raw_pairs.csv"
ACROSS_COMPOSET = ROOT / "image_deviation/across_scene_base_pairs.csv"
OUT_DIR = ROOT / "text_deviation/figures"

# Match paper body font (NeurIPS template uses Times).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

HERO = "#C44E52"
WHATSUP = "#56B4E9"
PRIORS = ["winoground", "bivlc", "mmvp", "colorswap"]
PRIOR_LABELS = {
    "winoground": "Winoground",
    "bivlc": "BiVLC",
    "mmvp": "MMVP",
    "colorswap": "ColorSwap",
    "whatsup": "What's Up",
    "compoSET": "CompoSET",
}
PRIOR_COLORS = ["#0a3d62", "#1f6091", "#3d7eb5", "#7aa9ce"]


def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def panel(ax, df, value_col, hero_filter, hero_label,
          xlim, xlabel, title, hero_zorder=10):
    medians = []
    for b, c in zip(PRIORS, PRIOR_COLORS):
        sub = df[df["benchmark"] == b][value_col].dropna().values
        if len(sub) == 0:
            continue
        x, y = ecdf(sub)
        ax.plot(x, y, color=c, lw=1.4, alpha=0.95, label=PRIOR_LABELS[b])
        medians.append((b, np.median(sub), c))
    ws = df[df["benchmark"] == "whatsup"][value_col].dropna().values
    if len(ws):
        x, y = ecdf(ws)
        ax.plot(x, y, color=WHATSUP, lw=2.2, ls="--",
                label=PRIOR_LABELS["whatsup"], zorder=8)
        medians.append(("whatsup", np.median(ws), WHATSUP))
    hero_sub = hero_filter(df)[value_col].dropna().values
    x, y = ecdf(hero_sub)
    ax.plot(x, y, color=HERO, lw=3.0, label=hero_label, zorder=hero_zorder)
    medians.append((hero_label, np.median(hero_sub), HERO))
    for _, m, c in medians:
        ax.axvline(m, color=c, lw=0.6, ls=":", alpha=0.4, ymax=0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_title(title, fontsize=18, fontweight="bold", loc="center", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(axis="both", labelsize=13)


fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})

# Panel 1 — image-edit fraction
img_df = pd.read_csv(IMG_RAW, low_memory=False)
panel(axes[0], img_df, "frac_pixels_affected",
      hero_filter=lambda d: d[d["benchmark"] == "compoSET"],
      hero_label="CompoSET",
      xlim=(-0.01, 1.02),
      xlabel="Fraction of pixels affected",
      title="Localized visual edits")

# Panel 2 — text-edit fraction (medium tier)
txt_df = load_long()
txt_df = txt_df[txt_df["metric_name"] == "frac_tokens_changed"].rename(
    columns={"value": "frac_tokens_changed"})
panel(axes[1], txt_df, "frac_tokens_changed",
      hero_filter=lambda d: d[(d["benchmark"] == "compoSET")
                              & (d["density_level"] == "medium")],
      hero_label="CompoSET",
      xlim=(0, 1),
      xlabel="Fraction of tokens changed",
      title="Localized caption edits")

# Panel 3 — within-pair CLIP image cosine, with CompoSET across-scene baseline
panel(axes[2], img_df, "clip_image_cos",
      hero_filter=lambda d: d[d["benchmark"] == "compoSET"],
      hero_label="CompoSET",
      xlim=(0.0, 1.0),
      xlabel="CLIP image cosine",
      title="Tightly paired in latent space")

# CompoSET across-scene baseline: faint red fill between intra and across-scene
# ECDFs (behind blues), plus a dashed red line marking the across-scene curve.
across_df = pd.read_csv(ACROSS_COMPOSET)
intra_vals = img_df[img_df["benchmark"] == "compoSET"]["clip_image_cos"].dropna().values
across_vals = across_df["clip_image_cos"].dropna().values
y_grid = np.linspace(0.0, 1.0, 400)
x_intra_q = np.quantile(intra_vals, y_grid)
x_across_q = np.quantile(across_vals, y_grid)
axes[2].fill_betweenx(y_grid, x_across_q, x_intra_q,
                      color=HERO, alpha=0.13, lw=0, zorder=0.5)
acx, acy = ecdf(across_vals)
axes[2].plot(acx, acy, color=HERO, lw=1.6, ls="--", alpha=0.55, zorder=1,
             label="CompoSET\n(across-scene)")

axes[0].set_ylabel("Cumulative fraction of pairs", fontsize=16)
axes[0].set_ylim(0, 1)

# Single shared legend on the right, outside the axes. Pull the within-pair
# entries from panel 1 and append the CompoSET across-scene entry from panel 3.
handles, labels = axes[0].get_legend_handles_labels()
ix = {l: i for i, l in enumerate(labels)}
order = ["CompoSET", "What's Up", "Winoground", "BiVLC", "MMVP", "ColorSwap"]
handles = [handles[ix[k]] for k in order if k in ix]
labels = [k for k in order if k in ix]
h3, l3 = axes[2].get_legend_handles_labels()
for h, l in zip(h3, l3):
    if l == "CompoSET\n(across-scene)":
        handles.append(h)
        labels.append(l)
        break
fig.legend(handles, labels, loc="center left",
           bbox_to_anchor=(0.87, 0.5),
           fontsize=15, frameon=True,
           facecolor="white", edgecolor="#cccccc", framealpha=0.95,
           title="Benchmark", title_fontsize=16)

fig.tight_layout()
fig.subplots_adjust(right=0.86, wspace=0.12)
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig_analysis_trio.{ext}",
                bbox_inches="tight", dpi=150 if ext == "png" else None)
print("saved fig_analysis_trio.{pdf,png}")
