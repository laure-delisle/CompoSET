"""Generate all figures from raw_pairs.csv + diff_masks/*.npz.
Standalone; reads only from persisted artifacts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
DIFF = OUT / "diff_masks"

raw = pd.read_csv(OUT / "raw_pairs.csv")

# Order: keep the original three first (paper-headline figures expect this
# ordering), then any additional benchmarks present in raw_pairs.csv.
PRIMARY = ["compoSET", "winoground", "bivlc"]
_present = list(raw["benchmark"].dropna().unique())
EXTRA = [b for b in _present if b not in PRIMARY]
BENCHES = [b for b in PRIMARY if b in _present] + sorted(EXTRA)

# Stable colors: tab10 for new benchmarks, fixed colors for the original three.
import matplotlib.cm as cm
_tab10 = cm.get_cmap("tab10")
COLORS = {"compoSET": "#1f77b4", "winoground": "#ff7f0e", "bivlc": "#2ca02c"}
for i, b in enumerate(sorted(EXTRA)):
    # tab10 indices 0-2 already used; start from 3.
    COLORS.setdefault(b, _tab10((i + 3) % 10))

def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}")

# ---- 1. fraction_affected histogram (main) ----
# bins=100 with explicit range so density values are comparable across datasets.
fig, ax = plt.subplots(figsize=(6, 4))
for b in BENCHES:
    v = raw.loc[raw["benchmark"] == b, "frac_pixels_affected"].dropna()
    if len(v):
        ax.hist(v, bins=100, range=(0, 1), histtype="step", linewidth=2.0,
                label=f"{b} (n={len(v)}, μ={v.mean():.2f})",
                color=COLORS.get(b), density=True)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction of pixels affected by edit")
ax.set_ylabel("Density")
ax.legend()
ax.set_title("CompoSET foils change <10% of pixels; prior-benchmark foils change the whole image")
savefig(fig, "fig_main_fraction_affected")

# ---- 2. aggregate heatmap per benchmark ----
fig, axes = plt.subplots(1, len(BENCHES), figsize=(4 * len(BENCHES), 4))
for ax, b in zip(axes, BENCHES):
    rows = raw[raw["benchmark"] == b]
    heat = None
    count = 0
    for _, r in rows.iterrows():
        p = DIFF / f"{b}_{r['pair_id']}.npz"
        if not p.exists():
            continue
        m = np.load(p)["mask"].astype(np.float32)
        m = m / (m.max() + 1e-8)
        if heat is None:
            heat = np.zeros_like(m)
        heat += m
        count += 1
    if heat is not None and count > 0:
        heat /= count
        im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=heat.max())
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{b} (n={count})")
    ax.axis("off")
fig.suptitle("Aggregate edit-region per benchmark — CompoSET edits cluster near the image center; prior benchmarks fill the frame")
savefig(fig, "fig_heatmap_per_benchmark")

# ---- 3. CompoSET per-scene heatmap grid ----
cs = raw[raw["benchmark"] == "compoSET"]
scenes = sorted(cs["scene_id"].dropna().unique())
if scenes:
    n = len(scenes)
    cols = 10
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.1, rows_n * 1.1))
    axes = np.atleast_2d(axes)
    for i, s in enumerate(scenes):
        ax = axes[i // cols, i % cols]
        rows = cs[cs["scene_id"] == s]
        heat = None; cnt = 0
        for _, r in rows.iterrows():
            p = DIFF / f"compoSET_{r['pair_id']}.npz"
            if not p.exists(): continue
            m = np.load(p)["mask"].astype(np.float32)
            m = m / (m.max() + 1e-8)
            if heat is None: heat = np.zeros_like(m)
            heat += m; cnt += 1
        if heat is not None and cnt > 0:
            heat /= cnt
            ax.imshow(heat, cmap="viridis")
        ax.set_title(s, fontsize=6)
        ax.axis("off")
    # Hide extras
    for j in range(n, rows_n * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("Per-scene aggregate edit heatmap (CompoSET)")
    savefig(fig, "fig_heatmap_per_scene_compoSET")

# ---- 4. centroid scatter ----
fig, axes = plt.subplots(1, len(BENCHES), figsize=(4 * len(BENCHES), 4))
for ax, b in zip(axes, BENCHES):
    g = raw[raw["benchmark"] == b]
    ax.scatter(g["edit_centroid_x"], g["edit_centroid_y"], s=6, alpha=0.4, color=COLORS.get(b))
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.set_title(f"{b} edit centroid")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
savefig(fig, "fig_centroid_scatter")

# ---- 5. distance-to-center hist ----
# Standardized bins=100 over fixed [0, 0.6] range so densities are comparable
# across datasets (Pietro 2026-04-23). Prior benchmarks collapse to ~0
# (whole-image diff → centroid at image center); CompoSET edits land in a
# spatially-varied band but stay clustered near the image center, not at the
# periphery — informative supplementary detail.
fig, ax = plt.subplots(figsize=(7, 4))
for b in BENCHES:
    v = raw.loc[raw["benchmark"] == b, "distance_to_center"].dropna()
    if len(v):
        ax.hist(v, bins=100, range=(0, 0.6), histtype="step", linewidth=2.0,
                label=f"{b} (n={len(v)}, median={v.median():.3f})",
                color=COLORS.get(b), density=True)
ax.set_xlim(0, 0.6)
ax.set_xlabel("Distance from edit centroid to image center (normalized; 0 = center)")
ax.set_ylabel("Density")
ax.legend(loc="upper right")
ax.set_title("CompoSET edits are spatially varied but stay near the image center, not the periphery")
savefig(fig, "fig_distance_to_center_hist")

# ---- 6. ssim vs lpips scatter ----
fig, ax = plt.subplots(figsize=(6, 5))
for b in BENCHES:
    g = raw[raw["benchmark"] == b]
    ax.scatter(g["ssim"], g["lpips_alex"], s=6, alpha=0.4, label=b, color=COLORS.get(b))
ax.set_xlabel("SSIM (higher = more similar)")
ax.set_ylabel("LPIPS-Alex (lower = more similar)")
ax.legend()
ax.set_title("Structural vs perceptual similarity")
savefig(fig, "fig_ssim_vs_lpips_scatter")

# ---- 7. per-edit-type bar chart (CompoSET) ----
if len(cs):
    ORDER = [
        "modify_color","modify_pattern","modify_shape","modify_material","modify_transparency",
        "modify_spatial","modify_state","modify_presence","modify_count",
        "modify_action","modify_pose","modify_object",
        "swap_color","swap_material","swap_spatial","swap_role",
    ]
    cs2 = cs.copy()
    cs2["edit_type"] = cs2["edit_type"].replace("modify_count", "modify_count")
    means = cs2.groupby("edit_type")["frac_pixels_affected"].mean()
    stds = cs2.groupby("edit_type")["frac_pixels_affected"].std()
    ORDER = [t.replace("modify_count", "modify_count") for t in ORDER]
    labels = [t for t in ORDER if t in means.index]
    vals = [means[t] for t in labels]
    errs = [stds[t] for t in labels]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif"],
    })
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#1f6091" if t.startswith("modify_") else "#C44E52" for t in labels]
    ax.bar(range(len(labels)), vals, yerr=errs, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=13)
    ax.set_ylabel("Mean fraction of pixels affected", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title("CompoSET pixel-affected fraction by edit type",
                 fontsize=18, fontweight="bold", pad=10)
    fig.tight_layout()
    savefig(fig, "fig_per_edit_type_breakdown")

# ---- 8. DINOv2 image cosine histogram per benchmark ----
# bins=100 with explicit range so densities are comparable across datasets.
fig, ax = plt.subplots(figsize=(7, 4))
for b in BENCHES:
    v = raw.loc[raw["benchmark"] == b, "dinov2_cos"].dropna()
    if len(v):
        ax.hist(v, bins=100, range=(-0.2, 1.0), histtype="step", linewidth=2.0,
                label=f"{b} (n={len(v)}, μ={v.mean():.2f})",
                color=COLORS.get(b), density=True)
ax.set_xlim(-0.2, 1.0)
ax.set_xlabel("DINOv2 image–image cosine similarity (base vs foil)")
ax.set_ylabel("Density")
ax.legend(loc="upper left")
ax.set_title("CompoSET foils sit close to their bases in DINOv2 space; prior-benchmark foils do not")
savefig(fig, "fig_dinov2_cos_hist")

# ---- 9. CLIP image-image cosine histogram per benchmark ----
fig, ax = plt.subplots(figsize=(7, 4))
for b in BENCHES:
    v = raw.loc[raw["benchmark"] == b, "clip_image_cos"].dropna()
    if len(v):
        ax.hist(v, bins=100, range=(0.2, 1.0), histtype="step", linewidth=2.0,
                label=f"{b} (n={len(v)}, μ={v.mean():.2f})",
                color=COLORS.get(b), density=True)
ax.set_xlim(0.2, 1.0)
ax.set_xlabel("CLIP image–image cosine similarity (base vs foil)")
ax.set_ylabel("Density")
ax.legend(loc="upper left")
ax.set_title("CompoSET foils sit close to their bases in CLIP space; prior-benchmark foils do not")
savefig(fig, "fig_clip_image_cos_hist")

# ---- 10. within vs across image cosine, all three benchmarks ----
# Within = base/image_0 vs foil/image_1 (from raw_pairs.csv, per benchmark).
# Across for CompoSET = base A vs base B (across_scene_base_pairs.csv, n=3,160).
# Across for Winoground/BiVLC = image_0 of pair A vs image_0 of pair B
# (across_pair_prior.csv, n=19,900 per benchmark).
compo_across = OUT / "across_scene_base_pairs.csv"
prior_across = OUT / "across_pair_prior.csv"
if compo_across.exists() and prior_across.exists():
    ca = pd.read_csv(compo_across); ca["benchmark"] = "compoSET"
    pa = pd.read_csv(prior_across)
    across_all = pd.concat([ca[["benchmark", "dinov2_cos", "clip_image_cos"]],
                            pa[["benchmark", "dinov2_cos", "clip_image_cos"]]])
    # Only show benchmarks that have an across-pair distribution available.
    bench_with_across = [b for b in BENCHES if b in set(across_all["benchmark"].unique())]
    if not bench_with_across:
        bench_with_across = BENCHES
    n_b = len(bench_with_across)
    fig, axes = plt.subplots(2, n_b, figsize=(max(4, 4 * n_b), 6.5), sharey="row", squeeze=False)
    for row_idx, (metric, xlim, label) in enumerate([
        ("dinov2_cos",    (-0.2, 1.0), "DINOv2"),
        ("clip_image_cos", (0.2, 1.0), "CLIP"),
    ]):
        for col_idx, b in enumerate(bench_with_across):
            ax = axes[row_idx, col_idx]
            w = raw.loc[raw["benchmark"] == b, metric].dropna()
            a = across_all.loc[across_all["benchmark"] == b, metric].dropna()
            ax.hist(w, bins=100, range=xlim, histtype="step", linewidth=2.0,
                    color=COLORS.get(b), density=True,
                    label=f"within-pair (n={len(w)}, μ={w.mean():.2f})")
            ax.hist(a, bins=100, range=xlim, histtype="step", linewidth=2.0,
                    color="#555555", linestyle="--", density=True,
                    label=f"across-pair (n={len(a)}, μ={a.mean():.2f})")
            ax.set_xlim(xlim)
            if row_idx == 1:
                ax.set_xlabel(f"{label} image–image cosine")
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nDensity")
            ax.set_title(f"{b}  (gap = {w.mean() - a.mean():+.2f})", fontsize=10)
            ax.legend(loc="upper center", fontsize=8)
    fig.suptitle("CompoSET separates within- from across-pair cosine by ~0.87 on DINOv2;"
                 " prior benchmarks show no separation — their within-pair cos is already ≈ across-pair cos",
                 fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig_within_vs_across_scene_cos")
else:
    missing = [str(p.name) for p in [compo_across, prior_across] if not p.exists()]
    print(f"  skip fig_within_vs_across_scene_cos — missing {missing}; run compute_across_*.py")

print("done.")
