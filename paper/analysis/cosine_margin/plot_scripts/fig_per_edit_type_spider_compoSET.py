"""Spidergram (radar) of per-edit-type correct rate per model on CompoSET.

One axis per edit type (16 axes, ordered by the canonical taxonomy
grouping). One line per model, colored by model family. Family mean
overlaid as a bold line. Two figures: I2T and T2I.

Family colors are Okabe-Ito-friendly. Background shading marks the
five edit-type clusters from the taxonomy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

ROOT = Path(__file__).resolve().parents[1]
OUT_MOD_I2T = ROOT / "figures" / "fig_per_edit_type_spider_modify_I2T_compoSET.pdf"
OUT_MOD_T2I = ROOT / "figures" / "fig_per_edit_type_spider_modify_T2I_compoSET.pdf"
OUT_SWAP_I2T = ROOT / "figures" / "fig_per_edit_type_spider_swap_I2T_compoSET.pdf"
OUT_SWAP_T2I = ROOT / "figures" / "fig_per_edit_type_spider_swap_T2I_compoSET.pdf"

# Edit-type axis order — modify_* edits, grouped by taxonomy cluster.
MODIFY_GROUPS = [
    ("Visual",          ["modify_color", "modify_pattern", "modify_shape",
                         "modify_material", "modify_transparency"]),
    ("Action/Pose",     ["modify_action", "modify_pose"]),
    ("Object",          ["modify_object"]),
    ("State/Quantity",  ["modify_state", "modify_presence", "modify_count"]),
    ("Spatial",         ["modify_spatial"]),
]
# Swap_* edits — separate spider, since swaps are systematically harder than
# modifies and pooling on one chart compresses the modify_* signal.
SWAP_GROUPS = [
    ("Swaps",           ["swap_color", "swap_material", "swap_spatial", "swap_role"]),
]

# Group accent colors — used for the outer labeled arc, not full-pie shading.
GROUP_COLOR = {
    "Visual":         "#7B68EE",  # medium slate blue
    "Action/Pose":    "#CD5C5C",  # indian red
    "Object":         "#3CB371",  # medium sea green
    "State/Quantity": "#DAA520",  # goldenrod
    "Spatial":        "#20B2AA",  # light sea green
    "Swaps":          "#9370DB",  # medium purple
}

# Model families. Order matters for legend.
FAMILIES = [
    ("OpenAI CLIP",         ["clip-b32", "clip-l14", "clip-b16-openai"],
                            "#000000"),
    ("OpenCLIP (LAION)",    ["clip-b16-laion2b", "clip-b16-laion400m",
                             "clip-h14-laion2b", "clip-l14-laion2b"],
                            "#56B4E9"),
    ("DFN (Apple)",         ["dfn-b16", "dfn-h14"],
                            "#009E73"),
    ("MetaCLIP2",           ["metaclip2-h14"],
                            "#999933"),
    ("EVA02",               ["eva02-b16", "eva02-l14"],
                            "#CC79A7"),
    ("SigLIP",              ["siglip-b16", "siglip-l16-384", "siglip-so400m-384"],
                            "#D55E00"),
    ("SigLIP2",             ["siglip2-b16", "siglip2-l16-384", "siglip2-so400m-378"],
                            "#0072B2"),
    ("Compositionality-trained",
                            ["negclip-b32", "clove-b32", "dreamlip-b16-merged30m",
                             "jina-embeddings-v4"],
                            "#E69F00"),
]


def model_to_family():
    m = {}
    for fam, models, color in FAMILIES:
        for k in models:
            m[k] = (fam, color)
    return m


def render(direction: str, out_path: Path, edit_groups, family_kind: str):
    """family_kind: 'modify' or 'swap' — controls title + radial scale."""
    edit_order = [t for _, ts in edit_groups for t in ts]
    df = pd.read_csv(ROOT / "raw_pairs.csv", low_memory=False)
    cs = df[(df["benchmark"] == "composet") & (df["direction"] == direction)].copy()
    if cs.empty:
        print(f"skip {direction} — no rows")
        return

    # Per (model, edit_type): mean correct rate.
    grid = (cs.groupby(["model", "edit_type"])["decision_correct"]
              .mean().unstack("edit_type"))
    grid = grid.reindex(columns=edit_order)  # fix axis order

    # Verify model coverage; warn if any expected model missing.
    expected = [m for _, ms, _ in FAMILIES for m in ms]
    missing = [m for m in expected if m not in grid.index]
    if missing:
        print(f"  warn: {len(missing)} expected model(s) missing in raw_pairs: {missing}")

    fam_map = model_to_family()
    n_axes = len(edit_order)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    # Compute a sensible radial limit from the data; tick at 0.1, 0.2, ...
    r_max = float(grid.values.max(initial=0.0))
    r_lim = max(0.1, np.ceil(r_max * 10) / 10)
    if r_lim < 0.45:
        r_lim = max(r_lim, 0.45)

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("N")  # 0° at top
    ax.set_theta_direction(-1)       # clockwise

    # Per-model lines (translucent).
    for model, row in grid.iterrows():
        fam, color = fam_map.get(model, (None, "#999999"))
        vals = row.fillna(0).values
        vals_closed = np.concatenate([vals, vals[:1]])
        ax.plot(angles_closed, vals_closed, color=color, alpha=0.30,
                linewidth=1.0, zorder=2)

    # Family-mean lines (bold).
    for fam_name, models, color in FAMILIES:
        present = [m for m in models if m in grid.index]
        if not present:
            continue
        fam_mean = grid.loc[present].mean(axis=0).fillna(0).values
        fam_mean_closed = np.concatenate([fam_mean, fam_mean[:1]])
        ax.plot(angles_closed, fam_mean_closed, color=color, alpha=1.0,
                linewidth=2.4, zorder=3, label=fam_name)
        # Marker at each axis point.
        ax.scatter(angles, fam_mean, color=color, s=18, zorder=4,
                   edgecolor="white", linewidth=0.5)

    # Axis labels (edit types).
    ax.set_xticks(angles)
    ax.set_xticklabels(edit_order, fontsize=9)
    # Manual rotation so labels read along the spoke direction.
    for label, angle in zip(ax.get_xticklabels(), angles):
        rot = np.degrees(angle)
        if 90 < rot < 270:
            label.set_rotation(rot - 180)
            label.set_horizontalalignment("right")
        else:
            label.set_rotation(rot)
            label.set_horizontalalignment("left")

    # Radial ticks.
    ax.set_ylim(0, r_lim)
    rticks = np.arange(0.1, r_lim + 1e-6, 0.1)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t:.1f}" for t in rticks], fontsize=8, color="#555555")
    ax.set_rlabel_position(0)

    # Outer arcs labeling the taxonomy groups, sitting just outside the data.
    # Use ax.bar in polar to draw a thin colored band, then a centered text
    # label at a slightly larger radius.
    arc_inner = r_lim * 1.04
    arc_thick = r_lim * 0.04
    label_radius = r_lim * 1.15
    step = 2 * np.pi / n_axes
    cursor = 0
    for group_name, members in edit_groups:
        if not members:
            continue
        # Angular span: first axis − step/2 to last axis + step/2.
        first_idx = cursor
        last_idx = cursor + len(members) - 1
        center_idx = (first_idx + last_idx) / 2.0
        center_theta = center_idx * step
        width_theta = len(members) * step
        ax.bar(
            x=center_theta, height=arc_thick, width=width_theta * 0.95,
            bottom=arc_inner, color=GROUP_COLOR[group_name], alpha=0.85,
            edgecolor="white", linewidth=1.0, zorder=5,
        )
        # Tangential label, rotated to follow the arc.
        rot_deg = -np.degrees(center_theta)  # negate because clockwise
        if 90 < (np.degrees(center_theta) % 360) < 270:
            rot_deg = rot_deg + 180  # flip when on bottom half so text is upright
        ax.text(
            center_theta, label_radius, group_name,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=GROUP_COLOR[group_name],
            rotation=rot_deg, rotation_mode="anchor", zorder=6,
        )
        cursor += len(members)
    # Push the polar axis a bit further out so the arcs and labels fit.
    ax.set_rmax(r_lim * 1.25)

    kind_label = "modify_*" if family_kind == "modify" else "swap_*"
    ax.set_title(
        f"CompoSET — per-edit-type success rate ({kind_label}, {direction})\n"
        "Faint lines: 22 individual models. Bold lines: family means. "
        "Outer colored arcs: taxonomy groups.",
        fontsize=10, pad=24,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=8,
              title="Model family", title_fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    render("I2T", OUT_MOD_I2T,  MODIFY_GROUPS, "modify")
    render("T2I", OUT_MOD_T2I,  MODIFY_GROUPS, "modify")
    render("I2T", OUT_SWAP_I2T, SWAP_GROUPS,   "swap")
    render("T2I", OUT_SWAP_T2I, SWAP_GROUPS,   "swap")


if __name__ == "__main__":
    main()
