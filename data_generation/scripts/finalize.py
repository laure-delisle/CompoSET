#!/usr/bin/env python3
"""Export CompoSET to HuggingFace release layout.

Reads internal `data/` directory + `scene_settings.csv`, filters out
discarded variations, renumbers remaining variations contiguously, copies
and renames images, and writes the release under `hf_release/`:

  hf_release/
    variations.parquet   — 1776 rows, one per live variation
    scenes.parquet       — 80 rows, one per scene
    images/
      s001_base.png, s001_v01.png, s001_v02.png, ...

Also writes provenance mapping to `paper_stats/id_remap.json`.

Usage:
    python pipeline/export_hf.py
    python pipeline/export_hf.py --out hf_release
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCENE_SETTINGS = ROOT / "scene_settings.csv"
PAPER_STATS = ROOT / "paper_stats"


_SUBJECTS_MAP = {"0": "0", "1": "1", "2": "2", "several": "3+"}


# Canonical edit-type ordering: modify_* grouped by semantic axis (visual,
# spatial, state/quantity, action/pose, object), then swap_* grouped similarly.
EDIT_TYPE_ORDER: tuple[str, ...] = (
    # modify_* — visual attributes
    "modify_color", "modify_pattern", "modify_shape",
    "modify_material", "modify_transparency",
    # modify_* — spatial
    "modify_spatial",
    # modify_* — state / quantity
    "modify_state", "modify_presence", "modify_cardinality",
    # modify_* — action / pose
    "modify_action", "modify_pose",
    # modify_* — object substitution
    "modify_object",
    # swap_* — visual attributes
    "swap_color", "swap_material",
    # swap_* — spatial
    "swap_spatial",
    # swap_* — role
    "swap_role",
)
_EDIT_TYPE_RANK = {t: i for i, t in enumerate(EDIT_TYPE_ORDER)}


def sort_edit_types(types) -> list[str]:
    """Sort a collection of edit_types by canonical order."""
    return sorted(set(types), key=lambda t: _EDIT_TYPE_RANK.get(t, 999))


def read_scene_settings() -> dict[str, dict]:
    """Map scene_id -> {setting_category, setting}."""
    out = {}
    with open(SCENE_SETTINGS) as f:
        for row in csv.DictReader(f):
            sid = f"s{int(row['id']):03d}"
            out[sid] = {
                "setting_category": row["indoor_outdoor"],
                "setting": row["setting"],
            }
    return out


def read_corrected_caption(scene: str) -> str:
    """Parse s0NN_caption_corrected.txt and return a single-paragraph caption.

    The file is structured as:
        base_caption: "..."
        detail_sentences:
        surface: "..."
        objects: "..."
        background: "..."
        (etc.)

    Strips the label prefixes and concatenates all quoted sentences with
    single spaces. Order: base_caption first, then detail_sentences in
    file order.
    """
    p = DATA / scene / f"s{scene}_caption_corrected.txt"
    if not p.exists():
        return ""
    text = p.read_text()
    m_base = re.search(r'base_caption:\s*"(.+?)"', text, re.DOTALL)
    parts = []
    if m_base:
        parts.append(re.sub(r"\s+", " ", m_base.group(1)).strip())
    for mm in re.finditer(r'^\s*(\w+):\s*"(.+?)"', text, re.MULTILINE | re.DOTALL):
        if mm.group(1) == "base_caption":
            continue
        parts.append(re.sub(r"\s+", " ", mm.group(2)).strip())
    return " ".join(parts)


def read_scene_params(scene: str) -> dict:
    """Parse subjects + framing from s0NN_scene_params.txt."""
    p = DATA / scene / f"s{scene}_scene_params.txt"
    text = p.read_text() if p.exists() else ""
    m_sub = re.search(r"subjects:\s*(\S+)", text)
    m_fr = re.search(r"framing:\s*(\S+)", text)
    subj = m_sub.group(1).strip() if m_sub else "0"
    return {
        "n_subjects": _SUBJECTS_MAP.get(subj, "0"),
        "framing": m_fr.group(1).strip() if m_fr else "close-up",
    }


def load_qc(scene: str) -> dict:
    p = DATA / scene / "qc_decisions.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_variations(scene: str) -> list[dict]:
    p = DATA / scene / f"s{scene}_caption_variations.json"
    return json.loads(p.read_text())


def build_scene(scene: str, settings: dict) -> tuple[dict, list[dict], dict[str, str]]:
    """Process one scene. Returns (scene_row, variation_rows, id_remap)."""
    sid = f"s{scene}"
    params = read_scene_params(scene)
    qc = load_qc(scene)
    vars_ = load_variations(scene)

    live = [v for v in vars_ if qc.get(v["id"], {}).get("status") != "discarded"]
    # Preserve original order (already sorted by var number in the JSON)
    id_remap: dict[str, str] = {}
    var_rows: list[dict] = []
    types: list[str] = []

    for i, v in enumerate(live, start=1):
        new_vid = f"{sid}_v{i:02d}"
        id_remap[v["id"]] = new_vid
        et = (v.get("edit_types_v2") or v.get("edit_types") or ["?"])[0]
        types.append(et)
        caps = v.get("captions", {})
        var_rows.append({
            "id": new_vid,
            "scene_id": sid,
            "edit_type": et,
            "image_base": f"images/{sid}_base.png",
            "image_var": f"images/{new_vid}.png",
            "captions": {
                "short":  {"base": caps["short"]["base"],  "var": caps["short"]["var"]},
                "medium": {"base": caps["medium"]["base"], "var": caps["medium"]["var"]},
                "long":   {"base": caps["long"]["base"],   "var": caps["long"]["var"]},
            },
        })

    scene_row = {
        "scene_id": sid,
        "image_base": f"images/{sid}_base.png",
        "setting": settings.get("setting", ""),
        "setting_category": settings.get("setting_category", ""),
        "n_subjects": params["n_subjects"],
        "framing": params["framing"],
        "n_variations": len(live),
        "edit_types_present": sort_edit_types(types),
    }
    return scene_row, var_rows, id_remap


def copy_images(scene: str, id_remap: dict[str, str], out_dir: Path) -> tuple[int, int]:
    """Copy base + live var images to out_dir/images/ with renumbered names.
    Returns (n_copied, n_missing).
    """
    src_dir = DATA / scene / "images"
    dst_dir = out_dir / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    sid = f"s{scene}"

    # Base image
    base_src = src_dir / f"{sid}_base.png"
    if base_src.exists():
        shutil.copy2(base_src, dst_dir / f"{sid}_base.png")
        copied += 1
    else:
        missing += 1
        print(f"  MISSING base: {base_src}", file=sys.stderr)

    # Var images
    for old_id, new_id in id_remap.items():
        src = src_dir / f"{old_id}.png"
        dst = dst_dir / f"{new_id}.png"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            print(f"  MISSING var: {src}", file=sys.stderr)
    return copied, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hf_release",
                    help="Output directory (default: hf_release)")
    ap.add_argument("--skip-images", action="store_true",
                    help="Skip image copy (for fast schema testing)")
    args = ap.parse_args()

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to: {out_dir}", file=sys.stderr)

    settings = read_scene_settings()
    scene_rows: list[dict] = []
    caption_rows: list[dict] = []
    var_rows: list[dict] = []
    remap_all: dict[str, dict[str, str]] = {}
    total_copied = total_missing = 0

    scene_ids = sorted(
        d.name for d in DATA.iterdir()
        if d.is_dir() and d.name.isdigit()
    )

    for scene in scene_ids:
        s_settings = settings.get(f"s{scene}", {})
        scene_row, v_rows, id_remap = build_scene(scene, s_settings)
        scene_rows.append(scene_row)
        var_rows.extend(v_rows)
        remap_all[f"s{scene}"] = id_remap
        caption_rows.append({
            "scene_id": f"s{scene}",
            "base_caption": read_corrected_caption(scene),
        })
        if not args.skip_images:
            c, m = copy_images(scene, id_remap, out_dir)
            total_copied += c
            total_missing += m

    # variations.parquet — captions as nested struct
    var_df = pd.DataFrame(var_rows)
    var_table = pa.Table.from_pandas(var_df, preserve_index=False)
    pq.write_table(var_table, out_dir / "variations.parquet")

    # scenes.parquet
    scene_df = pd.DataFrame(scene_rows)
    scene_table = pa.Table.from_pandas(scene_df, preserve_index=False)
    pq.write_table(scene_table, out_dir / "scenes.parquet")

    # scene_captions.parquet — full corrected caption (one paragraph per scene)
    cap_df = pd.DataFrame(caption_rows)
    cap_table = pa.Table.from_pandas(cap_df, preserve_index=False)
    pq.write_table(cap_table, out_dir / "scene_captions.parquet")

    # Provenance: id remap for paper reproducibility
    PAPER_STATS.mkdir(exist_ok=True)
    (PAPER_STATS / "id_remap.json").write_text(
        json.dumps(remap_all, indent=2)
    )

    # Summary
    type_counts = Counter(r["edit_type"] for r in var_rows)
    print(f"\nExported:", file=sys.stderr)
    print(f"  scenes.parquet:         {len(scene_rows)} rows", file=sys.stderr)
    print(f"  scene_captions.parquet: {len(caption_rows)} rows", file=sys.stderr)
    print(f"  variations.parquet:     {len(var_rows)} rows", file=sys.stderr)
    if not args.skip_images:
        print(f"  images/:            {total_copied} copied, {total_missing} missing",
              file=sys.stderr)
    print(f"  id_remap.json:      {sum(len(m) for m in remap_all.values())} mappings",
          file=sys.stderr)
    print(f"\nEdit type distribution (live only):", file=sys.stderr)
    for et, c in type_counts.most_common():
        print(f"  {et:<25} {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
