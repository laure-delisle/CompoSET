#!/usr/bin/env python3
"""Flask server for CompoSET QC interface."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


DATA_DIR: Path = Path()  # set by CLI


# ── Naming helpers ──────────────────────────────────────────────────────────

def scene_prefix(scene_id: str) -> str:
    """Return the sXXX prefix for a scene folder name (e.g. '005' -> 's005')."""
    return f"s{scene_id}"


def caption_base_path(scene_dir: Path, sid: str) -> Path:
    return scene_dir / f"{sid}_caption_base.txt"


def caption_corrected_path(scene_dir: Path, sid: str) -> Path:
    return scene_dir / f"{sid}_caption_corrected.txt"


def variations_path(scene_dir: Path, sid: str) -> Path:
    """Return L0-only path if it exists, otherwise fall back to full variations."""
    l0 = scene_dir / f"{sid}_caption_variations_l0.json"
    if l0.exists():
        return l0
    return scene_dir / f"{sid}_caption_variations.json"


def base_image_url(scene_id: str, sid: str) -> str | None:
    scene_dir = DATA_DIR / scene_id
    img = scene_dir / "images" / f"{sid}_base.png"
    if img.exists():
        return f"/api/image/{scene_id}/images/{sid}_base.png"
    # Fallback for old naming
    for ext in ("png", "jpeg", "jpg"):
        p = scene_dir / f"2_base_image.{ext}"
        if p.exists():
            return f"/api/image/{scene_id}/2_base_image.{ext}"
    return None


# ── Caption parsing ──────────────────────────────────────────────────────────

def parse_caption_file(path: Path) -> dict:
    """Parse a caption file into {base_caption: str, detail_sentences: {key: value}}."""
    text = path.read_text()
    m = re.search(r'base_caption:\s*"(.+?)"', text, re.DOTALL)
    base_caption = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    details = {}
    for m in re.finditer(r'^\s*(\w+):\s*"(.+?)"', text, re.MULTILINE | re.DOTALL):
        key = m.group(1)
        if key == "base_caption":
            continue
        details[key] = re.sub(r"\s+", " ", m.group(2)).strip()

    return {"base_caption": base_caption, "detail_sentences": details}


def serialize_caption(data: dict) -> str:
    """Serialize caption dict back to the file format."""
    lines = [f'base_caption: "{data["base_caption"]}"']
    lines.append("detail_sentences:")
    for key, val in data["detail_sentences"].items():
        lines.append(f'  {key}: "{val}"')
    return "\n".join(lines) + "\n"


# ── Scene discovery ──────────────────────────────────────────────────────────

def get_scene_ids() -> list[str]:
    """Return sorted list of scene IDs from data directory."""
    scenes = []
    for d in DATA_DIR.iterdir():
        if not d.is_dir() or not d.name.isdigit():
            continue
        sid = scene_prefix(d.name)
        # Accept either new or old naming
        if (caption_base_path(d, sid).exists() or
                (d / "1_base_caption.txt").exists()):
            scenes.append(d.name)
    return sorted(scenes)


# ── Routes: pages ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ── Routes: API ──────────────────────────────────────────────────────────────

@app.route("/api/scenes")
def api_scenes():
    scenes = []
    for scene_id in get_scene_ids():
        scene_dir = DATA_DIR / scene_id
        sid = scene_prefix(scene_id)
        qc_path = scene_dir / "qc_decisions.json"
        qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}

        vp = variations_path(scene_dir, sid)
        n_vars = len(json.loads(vp.read_text())) if vp.exists() else 0

        base_reviewed = qc.get("__base_caption__", {}).get("status") == "reviewed"
        l1_reviewed = qc.get("__l1__", {}).get("status") == "reviewed"

        scenes.append({
            "id": scene_id,
            "base_reviewed": base_reviewed,
            "l1_reviewed": l1_reviewed,
            "n_variations": n_vars,
            "n_reviewed": sum(1 for v in qc.values()
                              if isinstance(v, dict) and v.get("status") in
                              ("approved", "edited", "discarded")
                              and not v.get("__base_caption__")),
        })
    return jsonify(scenes)


@app.route("/api/scene/<scene_id>/base")
def api_scene_base(scene_id):
    scene_dir = DATA_DIR / scene_id
    sid = scene_prefix(scene_id)

    base_path = caption_base_path(scene_dir, sid)
    corrected_path = caption_corrected_path(scene_dir, sid)

    if not base_path.exists():
        return jsonify({"error": "Scene not found"}), 404

    base = parse_caption_file(base_path)
    corrected = parse_caption_file(corrected_path) if corrected_path.exists() else base

    return jsonify({
        "scene_id": scene_id,
        "base_caption": base,
        "corrected_caption": corrected,
        "base_image": base_image_url(scene_id, sid),
    })


@app.route("/api/scene/<scene_id>/base", methods=["POST"])
def api_save_base(scene_id):
    scene_dir = DATA_DIR / scene_id
    sid = scene_prefix(scene_id)
    corrected = caption_corrected_path(scene_dir, sid)

    data = request.json
    # Remove detail sentences that were emptied during QC
    data["detail_sentences"] = {
        k: v for k, v in data["detail_sentences"].items() if v.strip()
    }
    corrected.write_text(serialize_caption(data))

    # Update QC state
    qc_path = scene_dir / "qc_decisions.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    qc["__base_caption__"] = {
        "status": "reviewed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")

    return jsonify({"ok": True})


@app.route("/api/scene/<scene_id>/variations")
def api_variations(scene_id):
    scene_dir = DATA_DIR / scene_id
    sid = scene_prefix(scene_id)
    vp = variations_path(scene_dir, sid)

    if not vp.exists():
        return jsonify({"error": "No variations"}), 404

    variations = json.loads(vp.read_text())
    qc_path = scene_dir / "qc_decisions.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}

    result = []
    for v in variations:
        vid = v["id"]
        img_path = scene_dir / "images" / f"{vid}.png"
        img_url = f"/api/image/{scene_id}/images/{vid}.png" if img_path.exists() else None

        qc_status = qc.get(vid, {})

        effective_types = v.get("edit_types_v2") or v.get("edit_types", [])
        is_spatial = any(
            t in ("modify_spatial", "swap_spatial", "spatial_rearrangement")
            for t in effective_types
        )
        result.append({
            "id": vid,
            "edit_types": effective_types,
            "edit_types_original": v.get("edit_types", []),
            "is_spatial": is_spatial,
            "edits": v.get("edits", []),
            "positive": v["captions"]["0"]["positive"],
            "negative": v["captions"]["0"]["negative"],
            "image": img_url,
            "qc_status": qc_status.get("status"),
        })

    return jsonify({
        "scene_id": scene_id,
        "base_image": base_image_url(scene_id, sid),
        "variations": result,
    })


def propagate_l0_to_edits_and_levels(v, old_pos, old_neg, new_pos, new_neg):
    """When L0 captions are edited, propagate changes to edits metadata and L1/L2.

    Strategy:
    - Compare old vs new L0 positive/negative to find what changed
    - Update edits[].from/to accordingly
    - Find-replace the old words with new words in L1/L2 captions
    """
    edits = v.get("edits", [])
    if not edits:
        return

    # Propagate negative changes -> edits[].to and L1/L2 negative
    if old_neg != new_neg:
        for edit in edits:
            old_to = edit.get("to", "")
            # Find old_to in old_neg and figure out what replaced it in new_neg
            if old_to and old_to in old_neg:
                # Replace old_to in old_neg with a placeholder, then find what's
                # in the same position in new_neg. Simpler: use the diff between
                # old_neg and new_neg word-by-word.
                new_to = _extract_replacement(old_neg, new_neg, old_to)
                if new_to is not None:
                    edit["to"] = new_to
                    # Propagate to L1/L2 negative
                    for level in ("1", "2"):
                        if level in v["captions"]:
                            v["captions"][level]["negative"] = (
                                v["captions"][level]["negative"].replace(old_to, new_to)
                            )

    # Propagate positive changes -> edits[].from and L1/L2 positive
    if old_pos != new_pos:
        for edit in edits:
            old_from = edit.get("from", "")
            if old_from and old_from in old_pos:
                new_from = _extract_replacement(old_pos, new_pos, old_from)
                if new_from is not None:
                    edit["from"] = new_from
                    for level in ("1", "2"):
                        if level in v["captions"]:
                            v["captions"][level]["positive"] = (
                                v["captions"][level]["positive"].replace(old_from, new_from)
                            )


def _extract_replacement(old_text, new_text, target):
    """Given old_text containing target and new_text where target was replaced,
    figure out what target was replaced with.

    Approach: remove the non-target parts (prefix/suffix) that are shared
    between old and new, and what remains in new_text is the replacement.
    """
    idx = old_text.find(target)
    if idx == -1:
        return None

    prefix = old_text[:idx]
    suffix = old_text[idx + len(target):]

    # The new text should share the same prefix and suffix
    if new_text.startswith(prefix) and new_text.endswith(suffix):
        # Extract what's between prefix and suffix in new_text
        start = len(prefix)
        end = len(new_text) - len(suffix) if suffix else len(new_text)
        if start <= end:
            return new_text[start:end]

    return None


@app.route("/api/variation/<scene_id>/<var_id>/save", methods=["POST"])
def api_save_variation(scene_id, var_id):
    scene_dir = DATA_DIR / scene_id
    sid = scene_prefix(scene_id)
    vp = variations_path(scene_dir, sid)
    data = request.json

    variations = json.loads(vp.read_text())
    for v in variations:
        if v["id"] == var_id:
            old_pos = v["captions"]["0"]["positive"]
            old_neg = v["captions"]["0"]["negative"]

            # Propagate L0 edits to edits metadata and L1/L2 before overwriting
            if data["positive"] != old_pos or data["negative"] != old_neg:
                propagate_l0_to_edits_and_levels(
                    v, old_pos, old_neg, data["positive"], data["negative"]
                )

            v["captions"]["0"]["positive"] = data["positive"]
            v["captions"]["0"]["negative"] = data["negative"]
            break
    else:
        return jsonify({"error": "Variation not found"}), 404

    vp.write_text(json.dumps(variations, indent=2) + "\n")

    # If we edited the L0 file, also propagate to the full variations file
    full_vp = scene_dir / f"{sid}_caption_variations.json"
    if full_vp.exists() and full_vp != vp:
        full_vars = json.loads(full_vp.read_text())
        for fv in full_vars:
            if fv["id"] == var_id:
                if data["positive"] != old_pos or data["negative"] != old_neg:
                    propagate_l0_to_edits_and_levels(
                        fv, old_pos, old_neg, data["positive"], data["negative"]
                    )
                fv["captions"]["0"]["positive"] = data["positive"]
                fv["captions"]["0"]["negative"] = data["negative"]
                break
        full_vp.write_text(json.dumps(full_vars, indent=2) + "\n")

    # Update QC state
    qc_path = scene_dir / "qc_decisions.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}

    entry = {"timestamp": datetime.now(timezone.utc).isoformat()}
    if data["positive"] != old_pos or data["negative"] != old_neg:
        entry["status"] = "edited"
        entry["positive_original"] = old_pos
        entry["negative_original"] = old_neg
        entry["positive_edited"] = data["positive"]
        entry["negative_edited"] = data["negative"]
    else:
        entry["status"] = "approved"

    qc[var_id] = entry
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")

    return jsonify({"ok": True})




@app.route("/api/variation/<scene_id>/<var_id>/discard", methods=["POST"])
def api_discard_variation(scene_id, var_id):
    scene_dir = DATA_DIR / scene_id
    qc_path = scene_dir / "qc_decisions.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}

    qc[var_id] = {
        "status": "discarded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")

    return jsonify({"ok": True})


















@app.route("/api/scene/<scene_id>/verbosities")
def api_verbosities(scene_id):
    """Return per-variation short/medium/long captions for read-only review.
    Skips discarded variations. L0 shown as reference.
    """
    scene_dir = DATA_DIR / scene_id
    sid = scene_prefix(scene_id)

    full_vp = scene_dir / f"{sid}_caption_variations.json"
    if not full_vp.exists():
        return jsonify({"error": "No variations file"}), 404

    full_variations = json.loads(full_vp.read_text())
    full_by_id = {v["id"]: v for v in full_variations}

    l0_vp = scene_dir / f"{sid}_caption_variations_l0.json"
    l0_variations = json.loads(l0_vp.read_text()) if l0_vp.exists() else full_variations

    qc_path = scene_dir / "qc_decisions.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}

    result = []
    for v in l0_variations:
        vid = v["id"]
        if qc.get(vid, {}).get("status") == "discarded":
            continue
        full_v = full_by_id.get(vid)
        if not full_v:
            continue
        caps = full_v.get("captions", {})
        if not all(t in caps for t in ("short", "medium", "long")):
            continue

        img_path = scene_dir / "images" / f"{vid}.png"
        img_url = f"/api/image/{scene_id}/images/{vid}.png" if img_path.exists() else None

        etypes = full_v.get("edit_types_v2") or full_v.get("edit_types") or []
        result.append({
            "id": vid,
            "image": img_url,
            "edit_type": etypes[0] if etypes else "",
            "l0": {
                "positive": v["captions"]["0"]["positive"],
                "negative": v["captions"]["0"]["negative"],
            },
            "short": {
                "base": caps["short"].get("base", ""),
                "var": caps["short"].get("var", ""),
            },
            "medium": {
                "base": caps["medium"].get("base", ""),
                "var": caps["medium"].get("var", ""),
            },
            "long": {
                "base": caps["long"].get("base", ""),
                "var": caps["long"].get("var", ""),
            },
        })

    return jsonify({
        "scene_id": scene_id,
        "base_image": base_image_url(scene_id, sid),
        "variations": result,
    })


@app.route("/api/image/<scene_id>/<path:subpath>")
def api_image(scene_id, subpath):
    scene_dir = DATA_DIR / scene_id
    return send_from_directory(scene_dir, subpath)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CompoSET QC Server")
    parser.add_argument("--data-dir", default="data",
                        help="Path to data directory (default: data)")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir).resolve()
    print(f"Serving data from: {DATA_DIR}")
    print(f"Found scenes: {get_scene_ids()}")

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
