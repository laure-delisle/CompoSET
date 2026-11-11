"""Per-scene structure plot for CompoSET — decomposes total difficulty into
scene-level and within-scene foil-level components.

Per (scene, foil): mean correct rate across 22 models = "foil success rate".
Per scene:
    x = mean foil success rate over all foils in the scene
        (= overall scene success rate)
    y = std of foil success rates within the scene
        (= within-scene heterogeneity)

Median splits on both axes color-code the scatter into four populated
groups. The labels are descriptive (lower/upper half of the 80-scene
distribution) — they say nothing about whether a scene is absolutely hard
or easy. The point of the figure is structural: there is real variance on
BOTH axes, so having ~22 foils per base lets us decompose difficulty into
a scene-level component and a within-scene foil-level component (which a
single-foil benchmark cannot).

Companion plot: per-edit-type bar chart of mean correct rate.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_SCENE = ROOT / "figures" / "fig_per_scene_difficulty_compoSET.pdf"
OUT_EDIT  = ROOT / "figures" / "fig_per_edit_type_correct_rate_compoSET.pdf"

# Okabe-Ito colorblind-safe palette.
C_NEITHER   = "#999999"
C_SCENE     = "#0072B2"  # blue
C_EDIT      = "#E69F00"  # orange-yellow
C_BOTH      = "#CC79A7"  # reddish purple

EDIT_ORDER = [
    "modify_color","modify_pattern","modify_shape","modify_material","modify_transparency",
    "modify_spatial","modify_state","modify_presence","modify_count",
    "modify_action","modify_pose","modify_object",
    "swap_color","swap_material","swap_spatial","swap_role",
]


def main():
    df = pd.read_csv(ROOT / "raw_pairs.csv", low_memory=False)
    cs = df[df["benchmark"] == "composet"].copy()
    cs = cs.dropna(subset=["scene_id", "variation_id"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, direction in zip(axes, ["I2T", "T2I"]):
        sub = cs[cs["direction"] == direction]
        if sub.empty:
            ax.set_axis_off(); ax.set_title(f"{direction} (n/a)"); continue

        # Per (scene, foil): mean correct rate across all 22 models.
        per_foil = (sub.groupby(["scene_id", "variation_id"])["decision_correct"]
                       .mean().reset_index(name="foil_correct"))

        # Per scene: mean foil_correct (x) and std of foil_correct within the scene (y).
        per_scene = (per_foil.groupby("scene_id")["foil_correct"]
                              .agg(scene_mean="mean", foil_std="std", n="count")
                              .reset_index())

        # Median splits — descriptive bins of the 80-scene population, not
        # absolute difficulty thresholds.
        x_cut = per_scene["scene_mean"].median()
        y_cut = per_scene["foil_std"].median()
        lower_x = per_scene["scene_mean"] < x_cut
        upper_y = per_scene["foil_std"]   > y_cut
        q_ll = lower_x  & ~upper_y   # lower x, lower y
        q_lh = lower_x  &  upper_y   # lower x, upper y
        q_hl = ~lower_x & ~upper_y   # upper x, lower y
        q_hh = ~lower_x &  upper_y   # upper x, upper y

        for mask, color, label in [
            (q_ll, C_NEITHER, f"lower success, lower spread ({q_ll.sum()})"),
            (q_lh, C_BOTH,    f"lower success, upper spread ({q_lh.sum()})"),
            (q_hl, C_SCENE,   f"upper success, lower spread ({q_hl.sum()})"),
            (q_hh, C_EDIT,    f"upper success, upper spread ({q_hh.sum()})"),
        ]:
            ax.scatter(per_scene.loc[mask, "scene_mean"],
                       per_scene.loc[mask, "foil_std"],
                       s=28, c=color, alpha=0.85, edgecolor="white",
                       linewidth=0.5, label=label)

        ax.axvline(x_cut, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.axhline(y_cut, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.set_xlabel(f"scene success rate (mean over foils × 22 models) — {direction}")
        if direction == "I2T":
            ax.set_ylabel("std of per-foil success rate within scene")
        # Equal x/y limits so a single visual scale reads across both axes.
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 0.4)
        ax.set_aspect("equal")
        ax.set_title(f"CompoSET per-scene difficulty ({direction}) — n_scenes={len(per_scene)}",
                     fontsize=10)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.grid(alpha=0.25)

        # Annotate the median-split cutoffs.
        anno_kw = dict(
            fontsize=7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888888", linewidth=0.6, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.6,
                            shrinkA=2, shrinkB=2),
        )
        # Vertical dashed → x_cut. Callout placed in upper-right empty region.
        ax.annotate(
            f"median scene\nsuccess rate\n(x = {x_cut:.2f})",
            xy=(x_cut, 0.34), xytext=(0.32, 0.34),
            **anno_kw,
        )
        # Horizontal dashed → y_cut. Callout placed at right side.
        ax.annotate(
            f"median within-scene\nstd of foil success\n(y = {y_cut:.2f})",
            xy=(0.36, y_cut), xytext=(0.32, 0.07),
            **anno_kw,
        )

    fig.suptitle("Decomposing total difficulty into scene-level (x) and within-scene-foil-level (y) components — "
                 "having ~22 foils per base makes both axes measurable",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_SCENE, bbox_inches="tight")
    fig.savefig(OUT_SCENE.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_SCENE}")

    # Companion: per-edit-type correct rate bar chart (mean across models × foils × scenes).
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, direction in zip(axes2, ["I2T", "T2I"]):
        sub = cs[cs["direction"] == direction]
        if sub.empty: ax.set_axis_off(); continue
        means = (sub.groupby("edit_type")["decision_correct"]
                    .agg(["mean", "count"]).reset_index())
        means["edit_type"] = pd.Categorical(means["edit_type"],
                                            categories=EDIT_ORDER, ordered=True)
        means = means.sort_values("edit_type")
        # Color: modify_* = blue, swap_* = reddish purple
        colors = [C_SCENE if t.startswith("modify_") else C_BOTH
                  for t in means["edit_type"].astype(str)]
        ax.bar(range(len(means)), means["mean"], color=colors, alpha=0.85)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means["edit_type"].astype(str),
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"mean correct rate — {direction}")
        ax.set_ylim(0, max(0.6, means["mean"].max() * 1.15))
        ax.axhline(means["mean"].mean(), color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_title(f"CompoSET — per-edit-type success rate ({direction})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C_SCENE, alpha=0.85, label="modify_*"),
               Patch(facecolor=C_BOTH,  alpha=0.85, label="swap_*")]
    fig2.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
                bbox_to_anchor=(0.5, -0.02))
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    fig2.savefig(OUT_EDIT, bbox_inches="tight")
    fig2.savefig(OUT_EDIT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_EDIT}")


if __name__ == "__main__":
    main()
