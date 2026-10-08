"""Per-scene difficulty — Variant 1: joint plot with marginal distributions.

Continuous coloring on scene success rate (no median quadrants). Marginal
KDEs on top (scene-level distribution: harder ↔ easier scenes) and right
(within-scene foil spread: uniform ↔ heterogeneous foils). Annotates the
two extreme scenes per axis to ground the spectrum.

Story: both axes carry continuous signal — scenes vary in mean difficulty
AND vary in foil-to-foil heterogeneity. Each is a real decomposable axis,
not a binary cut.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_per_scene_v2_jointplot_compoSET.pdf"


def per_scene_table(direction: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "raw_pairs.csv", low_memory=False)
    cs = (df[(df["benchmark"] == "composet") & (df["direction"] == direction)]
          .dropna(subset=["scene_id", "variation_id"]))
    per_foil = (cs.groupby(["scene_id", "variation_id"])["decision_correct"]
                  .mean().reset_index(name="foil_correct"))
    per_scene = (per_foil.groupby("scene_id")["foil_correct"]
                          .agg(scene_mean="mean", foil_std="std", n="count")
                          .reset_index())
    return per_scene


def main():
    # Two-panel figure (I2T, T2I) with joint plots.
    fig = plt.figure(figsize=(13, 6.2))
    outer = gridspec.GridSpec(1, 2, wspace=0.30)

    for col, direction in enumerate(["I2T", "T2I"]):
        ps = per_scene_table(direction)
        if ps.empty:
            continue

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[0, col],
            width_ratios=[5, 1], height_ratios=[1, 5],
            wspace=0.06, hspace=0.06,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])

        # Main scatter: continuous color on scene_mean (viridis).
        sc = ax_main.scatter(
            ps["scene_mean"], ps["foil_std"],
            c=ps["scene_mean"], cmap="viridis", vmin=0, vmax=0.4,
            s=44, alpha=0.92, edgecolor="white", linewidth=0.6,
        )
        ax_main.set_xlim(0, 0.4)
        ax_main.set_ylim(0, 0.4)
        ax_main.set_aspect("equal")
        ax_main.set_xlabel(f"scene success rate (mean over foils × 22 models) — {direction}")
        ax_main.set_ylabel("std of per-foil success rate within scene")
        ax_main.grid(alpha=0.25)

        # Annotate extremes — hardest scene, easiest scene, most-uniform, most-heterogeneous.
        easiest = ps.loc[ps["scene_mean"].idxmax()]
        hardest = ps.loc[ps["scene_mean"].idxmin()]
        most_unif = ps.loc[ps["foil_std"].idxmin()]
        most_het = ps.loc[ps["foil_std"].idxmax()]
        for label, row, dx, dy in [
            (f"easiest scene\n{easiest.scene_id}", easiest, -0.04, 0.04),
            (f"hardest scene\n{hardest.scene_id}", hardest,  0.05, 0.02),
            (f"most uniform\n{most_unif.scene_id}", most_unif, 0.05, -0.02),
            (f"most heterogeneous\n{most_het.scene_id}", most_het, -0.05, 0.04),
        ]:
            ax_main.annotate(
                label, xy=(row.scene_mean, row.foil_std),
                xytext=(row.scene_mean + dx, row.foil_std + dy),
                fontsize=7, ha="center", va="center", color="#222",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#888", lw=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5),
            )

        # Top marginal: KDE of scene_mean (harder ↔ easier scenes).
        x = ps["scene_mean"].dropna().values
        bw = 0.025
        xs = np.linspace(0, 0.4, 200)
        density = np.zeros_like(xs)
        for v in x:
            density += np.exp(-0.5 * ((xs - v) / bw) ** 2)
        density /= density.max() if density.max() > 0 else 1.0
        ax_top.fill_between(xs, 0, density, color="#4682B4", alpha=0.7)
        ax_top.set_xlim(0, 0.4)
        ax_top.set_ylim(0, 1.1)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for s in ("top", "right", "left"): ax_top.spines[s].set_visible(False)
        ax_top.annotate(
            "", xy=(0.38, 0.30), xytext=(0.02, 0.30),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1),
        )
        ax_top.text(0.04, 0.55, "harder", fontsize=8, color="#444")
        ax_top.text(0.32, 0.55, "easier", fontsize=8, color="#444")
        ax_top.set_title(f"CompoSET per-scene structure — {direction}",
                         fontsize=10, pad=4)

        # Right marginal: KDE of foil_std (uniform ↔ heterogeneous).
        y = ps["foil_std"].dropna().values
        ys = np.linspace(0, 0.4, 200)
        density_y = np.zeros_like(ys)
        for v in y:
            density_y += np.exp(-0.5 * ((ys - v) / bw) ** 2)
        density_y /= density_y.max() if density_y.max() > 0 else 1.0
        ax_right.fill_betweenx(ys, 0, density_y, color="#CD5C5C", alpha=0.7)
        ax_right.set_ylim(0, 0.4)
        ax_right.set_xlim(0, 1.1)
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        for s in ("top", "right", "bottom"): ax_right.spines[s].set_visible(False)
        ax_right.annotate(
            "", xy=(0.30, 0.38), xytext=(0.30, 0.02),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1),
        )
        ax_right.text(0.55, 0.04, "uniform", fontsize=8, color="#444", rotation=90)
        ax_right.text(0.55, 0.30, "heterogeneous", fontsize=8, color="#444", rotation=90)

    fig.suptitle(
        "CompoSET per-scene structure — both axes carry continuous signal: "
        "scene-level difficulty (x) and within-scene foil heterogeneity (y).",
        fontsize=10, y=0.98,
    )
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
