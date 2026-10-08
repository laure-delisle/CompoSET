"""Per-scene difficulty — Variant 2: scenes sorted along a 1D difficulty axis,
each scene shown as a vertical strip of its ~22 foil success rates.

x = scene rank, sorted by mean foil success rate (left = harder, right = easier)
y = success rate of each individual foil (mean over 22 models)
strip width within a scene shows within-scene foil heterogeneity
points colored by edit_type taxonomy group

Story: scenes lie on a continuous difficulty spectrum (left→right gradient),
and the within-scene vertical spread shows that some scenes have nearly
identical foils while others have very different per-foil difficulties.
The dominant edit-type colors at high vs low foil-success are visible too.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_per_scene_v3_stripsorted_compoSET.pdf"

# Edit-type taxonomy groups → group color (Okabe-Ito-ish).
GROUP_COLOR = {
    "Visual":         "#7B68EE",   # color, pattern, shape, material, transparency
    "Action/Pose":    "#CD5C5C",   # action, pose
    "Object":         "#3CB371",   # object
    "State/Quantity": "#DAA520",   # state, presence, cardinality
    "Spatial":        "#20B2AA",   # spatial
    "Swaps":          "#9370DB",   # swap_*
}
EDIT_TO_GROUP = {
    "modify_color": "Visual", "modify_pattern": "Visual",
    "modify_shape": "Visual", "modify_material": "Visual",
    "modify_transparency": "Visual",
    "modify_action": "Action/Pose", "modify_pose": "Action/Pose",
    "modify_object": "Object",
    "modify_state": "State/Quantity", "modify_presence": "State/Quantity",
    "modify_cardinality": "State/Quantity",
    "modify_spatial": "Spatial",
    "swap_color": "Swaps", "swap_material": "Swaps",
    "swap_spatial": "Swaps", "swap_role": "Swaps",
}


def main():
    df = pd.read_csv(ROOT / "raw_pairs.csv", low_memory=False)
    cs = (df[df["benchmark"] == "composet"]
          .dropna(subset=["scene_id", "variation_id", "edit_type"]))
    cs["group"] = cs["edit_type"].map(EDIT_TO_GROUP)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), sharex=True)

    for ax, direction in zip(axes, ["I2T", "T2I"]):
        sub = cs[cs["direction"] == direction]
        if sub.empty:
            ax.set_axis_off()
            continue

        # Per (scene, foil): mean correct rate across 22 models.
        per_foil = (sub.groupby(["scene_id", "variation_id", "edit_type", "group"])
                       ["decision_correct"].mean()
                       .reset_index(name="foil_correct"))
        # Sort scenes by mean foil_correct (ascending: hardest left).
        scene_order = (per_foil.groupby("scene_id")["foil_correct"]
                                .mean().sort_values().index.tolist())
        scene_to_x = {s: i for i, s in enumerate(scene_order)}
        per_foil["x"] = per_foil["scene_id"].map(scene_to_x)
        # Jitter horizontally within each scene's column.
        rng = np.random.default_rng(0)
        per_foil["xj"] = per_foil["x"] + rng.uniform(-0.30, 0.30, size=len(per_foil))

        # Plot each foil dot, colored by taxonomy group.
        for grp, color in GROUP_COLOR.items():
            sel = per_foil[per_foil["group"] == grp]
            ax.scatter(sel["xj"], sel["foil_correct"], s=10, c=color,
                       alpha=0.55, edgecolor="none", label=grp)

        # Per-scene mean line (sorted, monotonically rising).
        scene_means = per_foil.groupby("x")["foil_correct"].mean().sort_index()
        ax.plot(scene_means.index, scene_means.values, color="#111",
                lw=1.3, alpha=0.85, label="scene mean")

        ax.set_ylabel(f"foil success rate — {direction}")
        ax.set_ylim(0, max(0.7, per_foil["foil_correct"].max() * 1.05))
        ax.set_xlim(-0.8, len(scene_order) - 0.2)
        ax.grid(axis="y", alpha=0.25)
        if direction == "T2I":
            ax.set_xticks(range(0, len(scene_order), 5))
            ax.set_xticklabels([scene_order[i] for i in range(0, len(scene_order), 5)],
                               rotation=70, fontsize=7)
            ax.set_xlabel("scene rank — sorted by mean foil success rate "
                          "(left: harder scenes ←  → easier scenes)")
        else:
            ax.set_xticks([])

        # Hardest/easiest annotation arrows.
        ax.annotate("", xy=(len(scene_order) - 1.0, ax.get_ylim()[1] * 0.92),
                    xytext=(0.0, ax.get_ylim()[1] * 0.92),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
        ax.text(1.5, ax.get_ylim()[1] * 0.95, "harder scenes",
                fontsize=8, color="#444")
        ax.text(len(scene_order) - 14, ax.get_ylim()[1] * 0.95, "easier scenes",
                fontsize=8, color="#444")
        ax.set_title(f"CompoSET — {direction}: each column is one scene "
                     f"(n={len(scene_order)}); each dot is one foil "
                     "(mean correct rate over 22 models, color = edit-type group)",
                     fontsize=9)

    # Single legend at top.
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=len(by_label), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
