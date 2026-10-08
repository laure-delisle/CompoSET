"""Per-scene difficulty — Variant 3: variance decomposition.

Total variance in foil success rate = between-scene + within-scene.
This is exactly the question contribution #4 motivates: how much of the
total compositional difficulty comes from scene identity vs from foil
identity within a scene?

Top: two side-by-side histograms with double-headed-arrow spectra
labels — scene-level (harder ↔ easier) on the left, within-scene
foil-level (uniform ↔ heterogeneous) on the right.

Bottom: stacked horizontal bar showing the variance partition (fraction
of total variance from between-scene vs within-scene). One number for
the paper's Phase-C claim.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_per_scene_v4_variance_partition_compoSET.pdf"

C_SCENE = "#4682B4"   # steel blue
C_FOIL  = "#CD5C5C"   # indian red


def per_foil(direction: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "raw_pairs.csv", low_memory=False)
    cs = (df[(df["benchmark"] == "composet") & (df["direction"] == direction)]
          .dropna(subset=["scene_id", "variation_id"]))
    return (cs.groupby(["scene_id", "variation_id"])["decision_correct"]
              .mean().reset_index(name="foil_correct"))


def variance_partition(per_foil_df: pd.DataFrame) -> dict:
    grand_mean = per_foil_df["foil_correct"].mean()
    scene_means = per_foil_df.groupby("scene_id")["foil_correct"].mean()
    scene_counts = per_foil_df.groupby("scene_id").size()

    # Between-scene SS (weighted by per-scene n_foils).
    between_ss = float(((scene_means - grand_mean) ** 2 * scene_counts).sum())
    # Within-scene SS.
    within_ss = 0.0
    for sid, g in per_foil_df.groupby("scene_id"):
        within_ss += float(((g["foil_correct"] - scene_means[sid]) ** 2).sum())
    total_ss = between_ss + within_ss
    return {
        "grand_mean": grand_mean,
        "between_ss": between_ss,
        "within_ss": within_ss,
        "total_ss": total_ss,
        "between_frac": between_ss / total_ss if total_ss > 0 else 0.0,
        "within_frac":  within_ss  / total_ss if total_ss > 0 else 0.0,
    }


def main():
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 0.8, 0.6], hspace=0.55,
                           figure=fig)

    for col, direction in enumerate(["I2T", "T2I"]):
        pf = per_foil(direction)
        if pf.empty:
            continue

        scene_means = (pf.groupby("scene_id")["foil_correct"]
                         .agg(["mean", "std", "count"]).reset_index())

        # Top row: histogram of scene means (left) and foil-stds (right).
        ax_top = fig.add_subplot(gs[0, col])
        ax_top_twin = ax_top.twinx()

        # Scene means hist.
        bins = np.linspace(0, 0.4, 21)
        ax_top.hist(scene_means["mean"], bins=bins, color=C_SCENE, alpha=0.55,
                    edgecolor="white", label="scene mean", linewidth=0.5)
        ax_top.set_xlim(0, 0.4)
        ax_top.set_xlabel("scene success rate (mean over foils × 22 models)")
        ax_top.set_ylabel("# scenes  (scene-level)", color=C_SCENE)
        ax_top.tick_params(axis="y", colors=C_SCENE)

        # Foil-std hist on twin axis.
        ax_top_twin.hist(scene_means["std"], bins=bins, color=C_FOIL,
                         alpha=0.55, edgecolor="white",
                         label="within-scene foil std", linewidth=0.5)
        ax_top_twin.set_ylabel("# scenes  (within-scene heterogeneity)", color=C_FOIL)
        ax_top_twin.tick_params(axis="y", colors=C_FOIL)

        ax_top.set_title(f"CompoSET — {direction}", fontsize=11)
        ax_top.grid(alpha=0.2, axis="x")

        # Custom legend manually.
        from matplotlib.patches import Patch
        leg_handles = [
            Patch(facecolor=C_SCENE, alpha=0.55, label="scene success rate (axis: harder ↔ easier)"),
            Patch(facecolor=C_FOIL,  alpha=0.55, label="within-scene foil std (axis: uniform ↔ heterogeneous)"),
        ]
        ax_top.legend(handles=leg_handles, fontsize=8, loc="upper right",
                      framealpha=0.92)

        # Middle row: pair of double-headed-arrow spectra.
        ax_arrow = fig.add_subplot(gs[1, col])
        ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
        ax_arrow.set_axis_off()
        # Scene spectrum (top half).
        ax_arrow.annotate("", xy=(0.95, 0.78), xytext=(0.05, 0.78),
                          arrowprops=dict(arrowstyle="<->", color=C_SCENE, lw=2))
        ax_arrow.text(0.07, 0.92, "harder scenes", fontsize=9, color=C_SCENE)
        ax_arrow.text(0.83, 0.92, "easier scenes", fontsize=9, color=C_SCENE,
                      ha="right")
        # Foil spectrum (bottom half).
        ax_arrow.annotate("", xy=(0.95, 0.20), xytext=(0.05, 0.20),
                          arrowprops=dict(arrowstyle="<->", color=C_FOIL, lw=2))
        ax_arrow.text(0.07, 0.34, "uniform foils per scene", fontsize=9, color=C_FOIL)
        ax_arrow.text(0.83, 0.34, "heterogeneous foils per scene",
                      fontsize=9, color=C_FOIL, ha="right")

        # Bottom row: variance partition stacked bar.
        vp = variance_partition(pf)
        ax_bar = fig.add_subplot(gs[2, col])
        bw = vp["between_frac"]
        ww = vp["within_frac"]
        ax_bar.barh([0], [bw], color=C_SCENE, alpha=0.85,
                    label=f"between-scene ({bw*100:.1f}%)")
        ax_bar.barh([0], [ww], left=[bw], color=C_FOIL, alpha=0.85,
                    label=f"within-scene  ({ww*100:.1f}%)")
        ax_bar.set_xlim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_bar.set_xticklabels([f"{int(t*100)}%" for t in [0, 0.25, 0.5, 0.75, 1.0]])
        ax_bar.set_xlabel("variance partition of foil success rate")
        ax_bar.legend(fontsize=8, loc="lower center",
                      bbox_to_anchor=(0.5, -1.6), ncol=2, framealpha=0.92)
        # Labels inside bar.
        if bw > 0.06:
            ax_bar.text(bw / 2, 0, f"{bw*100:.1f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        if ww > 0.06:
            ax_bar.text(bw + ww / 2, 0, f"{ww*100:.1f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        for s in ("top", "right", "left"): ax_bar.spines[s].set_visible(False)

    fig.suptitle(
        "Decomposing CompoSET total difficulty: scene identity vs foil identity",
        fontsize=11, y=0.995,
    )
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
