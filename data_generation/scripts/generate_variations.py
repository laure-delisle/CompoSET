#!/usr/bin/env python3
"""Generate L0 variation captions and images for a CompoSET scene.

Runs steps 3-5 of the pipeline (after QC of base caption):
  3   Generate L0 variation captions via Claude
  4   Light L0-only verification via Claude
  5   Generate variation images via Gemini

After this script, the user reviews L0 + images in the QC interface.
Then run build_levels.py to reconstruct L1/L2 from the final L0.

Requires generate_base.py to have been run first, and the corrected caption
to have been reviewed via the QC interface.

Usage:
    python generate_variations.py 5
    python generate_variations.py --from 11

Output (in data/005/):
    s005_caption_variations_raw.txt    (raw Claude output)
    s005_caption_variations_l0.json    (L0-only variations for QC)
    s005_verification_l0_fixes.json    (audit: L0 fixes from verify step)
    s005_variation_metadata.json       (audit: image generation results)
    images/s005_v01.png ... s005_vNN.png
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import anthropic
from google import genai
from google.genai import types
from PIL import Image

from common import (
    PROMPTS_DIR, DATA_DIR, CLAUDE_MODEL, GEMINI_IMAGE_MODEL,
    require_key, scene_id, scene_dir, scene_files,
    parse_caption_file, step_header, strip_json_fences,
    find_scenes_with_base,
)

def load_variation_prompt(scene_num: int | None = None) -> str:
    """Load the canonical variation prompt."""
    return (PROMPTS_DIR / "caption_3_generate_variations.txt").read_text()


# ── Variation parsing (from pipeline/04_reconstruct_levels.py) ──────────────

def _get_field(text: str, key: str) -> str | None:
    m = re.search(rf'^[ \t]*{re.escape(key)}:\s*(.+)$', text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _get_quoted(text: str, key: str) -> str | None:
    m = re.search(rf'^[ \t]*{re.escape(key)}:\s*"(.*?)"', text,
                  re.MULTILINE | re.DOTALL)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()
    return None


def parse_minimal_variations(text: str) -> list[dict]:
    blocks = re.split(r'(?=^id:\s)', text, flags=re.MULTILINE)
    variations = []
    for block in blocks:
        block = block.strip()
        if not block or not block.startswith('id:'):
            continue
        try:
            variations.append(_parse_minimal_block(block))
        except Exception as e:
            print(f"Warning: could not parse block — {e}\n{block[:120]}",
                  file=sys.stderr)
    return variations


def _parse_minimal_block(block: str) -> dict:
    variation_id = _get_field(block, 'id')
    edit_type = _get_field(block, 'edit_type')
    source = _get_field(block, 'source')
    clause = _get_quoted(block, 'clause') or _get_field(block, 'clause')

    if not variation_id or not edit_type or not source:
        raise ValueError("Missing id, edit_type, or source")

    edits_m = re.search(r'^edits:\s*\n(.*?)(?=\n(?:clause:|level_0:)|\Z)',
                        block, re.DOTALL | re.MULTILINE)
    edits_block = edits_m.group(1) if edits_m else ''

    edit = {}
    for field in ('target_object', 'attribute', 'from', 'to'):
        m = re.search(rf'^[ \t]*{re.escape(field)}:\s*"?([^"\n]+)"?$',
                      edits_block, re.MULTILINE)
        if m:
            edit[field] = m.group(1).strip().strip('"')

    level0_m = re.search(r'^level_0:\s*\n(.*)', block, re.DOTALL | re.MULTILINE)
    level0_block = level0_m.group(1) if level0_m else ''
    pos0 = _get_quoted(level0_block, 'positive') or ''
    neg0 = _get_quoted(level0_block, 'negative') or ''

    return {
        'id': variation_id,
        'edit_type': edit_type,
        'source': source,
        'clause': clause,
        'edit': edit,
        'level_0': {'positive': pos0, 'negative': neg0},
    }



# ── Image generation helpers (from pipeline/05_gen_variation_images.py) ─────

def _augment_pattern_for_image(caption: str) -> str:
    return re.sub(r'\bstriped\b', 'wide-striped', caption)


def _presence_captions(var: dict) -> tuple[str, str]:
    source = var.get('source_sentence', '')
    if not source:
        return var['captions']['0']['positive'], var['captions']['0']['negative']

    clean = re.sub(r'^There (?:are|is)\s+', '', source).rstrip('.')
    edit = var['edits'][0] if var.get('edits') else {}
    target = edit.get('target_object', '')
    is_removal = edit.get('to') == 'absent'

    if is_removal and target:
        neg = re.sub(rf'\b(a|an)\s+({re.escape(target)})', r'NO \2',
                     clean, count=1, flags=re.IGNORECASE)
    else:
        return var['captions']['0']['positive'], var['captions']['0']['negative']

    return clean, neg


def _build_variation_prompt(template, base_caption, variation_caption,
                            edit_types=None, var=None):
    suffix = ""
    if edit_types and var and 'modify_presence' in edit_types:
        base_caption, variation_caption = _presence_captions(var)
        edit = var['edits'][0] if var.get('edits') else {}
        target = edit.get('target_object', '')
        if edit.get('to') == 'absent' and target:
            suffix = f"\n\nEDIT: remove the {target} from the image."
        elif edit.get('from') == 'absent' and target:
            suffix = f"\n\nEDIT: add a {target} to the image."
    if edit_types and 'modify_pattern' in edit_types:
        variation_caption = _augment_pattern_for_image(variation_caption)
    return template.format(base_caption, variation_caption) + suffix


CANONICAL_SIZE = (1408, 768)

def _generate_variation_image(client, model, image, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(10)
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    img = Image.open(BytesIO(part.inline_data.data))
                    if img.size != CANONICAL_SIZE:
                        print(f"  attempt {attempt + 1}: wrong size {img.size}, retrying")
                        continue
                    return img, "ok"
            print(f"  attempt {attempt + 1}: no image returned")
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(2 ** (attempt + 1))
    return None, "failed"


# ── Process worker (top-level for pickling) ─────────────────────────────────

def _worker_process(api_key, base_image_path, prompt, out_path, model):
    """Run in a separate process: create own client, generate one image."""
    client = genai.Client(api_key=api_key)
    base_image = Image.open(base_image_path).convert("RGB")
    if base_image.size != CANONICAL_SIZE:
        return f"wrong_base_size:{base_image.size}"
    var_image, status = _generate_variation_image(client, model, base_image, prompt)
    if var_image is not None:
        if var_image.size != CANONICAL_SIZE:
            return f"wrong_var_size:{var_image.size}"
        var_image.save(out_path)
    return status


# ── Step implementations ────────────────────────────────────────────────────

def step_3(scene_num: int) -> list[dict]:
    """Generate L0 variation captions via Claude, parse into L0 JSON."""
    step_header("3", "Generate L0 variation captions (Claude)")
    files = scene_files(scene_num)
    sid = scene_id(scene_num)
    sd = scene_dir(scene_num)
    raw_out = sd / f"{sid}_caption_variations_raw.txt"
    l0_out = files["variations_l0"]

    # Always regenerate — delete stale L0 and raw files
    if l0_out.exists():
        l0_out.unlink()
        print(f"  Removed stale {l0_out.name}", file=sys.stderr)
    if raw_out.exists():
        raw_out.unlink()
        print(f"  Removed stale {raw_out.name}", file=sys.stderr)

    # Clear stale QC decisions for variations (keep __base_caption__)
    qc_path = sd / "qc_decisions.json"
    if qc_path.exists():
        with open(qc_path) as f:
            qc = json.load(f)
        qc_clean = {k: v for k, v in qc.items() if k == "__base_caption__"}
        with open(qc_path, "w") as f:
            json.dump(qc_clean, f, indent=2)
        removed = len(qc) - len(qc_clean)
        if removed:
            print(f"  Cleared {removed} stale QC decision(s)", file=sys.stderr)

    # Generate raw text if needed
    if raw_out.exists():
        print(f"  [resume] {raw_out.name} exists, reusing", file=sys.stderr)
        raw_text = raw_out.read_text()
    else:
        prompt_template = load_variation_prompt(scene_num)
        corrected = files["caption_corrected"].read_text().strip()

        full_prompt = (
            f"{prompt_template}\n\n---\nINPUT\n\n"
            f"Scene ID: {sid}\n\n{corrected}"
        )

        require_key("ANTHROPIC_API_KEY")
        print(f"  Calling {CLAUDE_MODEL}...", file=sys.stderr)
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=8192,
            messages=[{"role": "user", "content": full_prompt}],
        )
        raw_text = response.content[0].text
        raw_out.write_text(raw_text)
        print(f"  Saved: {raw_out.name} ({len(raw_text)} chars)", file=sys.stderr)

    # Parse into L0 JSON
    parsed_list = parse_minimal_variations(raw_text)
    print(f"  Parsed {len(parsed_list)} variations", file=sys.stderr)

    base_caption, detail_sentences = parse_caption_file(files["caption_corrected"])

    variations = []
    for parsed in parsed_list:
        source = parsed.get("source", "")
        source_sentence = (base_caption if source == "base_caption"
                           else detail_sentences.get(source, ""))
        variations.append({
            "id": parsed["id"],
            "image": "",
            "edit_types": [parsed["edit_type"]],
            "edits": [parsed["edit"]] if parsed.get("edit") else [],
            "source": source,
            "clause": parsed.get("clause", ""),
            "source_sentence": source_sentence,
            "captions": {
                "0": parsed["level_0"],
            },
        })

    with open(l0_out, "w") as f:
        json.dump(variations, f, indent=2)
    print(f"  Saved: {l0_out.name} ({len(variations)} variations)", file=sys.stderr)
    return variations


def step_4(scene_num: int, variations: list[dict]) -> list[dict]:
    """Light L0-only verification via Claude."""
    step_header("4", "Verify L0 captions (Claude)")
    files = scene_files(scene_num)
    l0_out = files["variations_l0"]

    sd = scene_dir(scene_num)
    sid = scene_id(scene_num)
    verification_raw = sd / f"{sid}_verification_l0_raw.txt"
    verification_fixes = sd / f"{sid}_verification_l0_fixes.json"

    # Always re-verify (variations were just regenerated)
    if verification_fixes.exists():
        verification_fixes.unlink()
    if verification_raw.exists():
        verification_raw.unlink()

    verify_prompt_path = PROMPTS_DIR / "caption_4_verify_variations.txt"
    verify_prompt = verify_prompt_path.read_text()
    corrected_text = files["caption_corrected"].read_text().strip()

    # Build a lightweight payload with just L0 + edits
    l0_payload = []
    for v in variations:
        l0_payload.append({
            "id": v["id"],
            "edit_types": v["edit_types"],
            "edits": v["edits"],
            "source_sentence": v.get("source_sentence", ""),
            "level_0": v["captions"]["0"],
        })

    vars_json = json.dumps(l0_payload, indent=2)
    full_prompt = f"{corrected_text}\n\n---\nVARIATIONS:\n{vars_json}"

    require_key("ANTHROPIC_API_KEY")
    print(f"  Calling {CLAUDE_MODEL} for L0 verification...", file=sys.stderr)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4096,
        system=verify_prompt,
        messages=[{"role": "user", "content": full_prompt}],
    )
    raw_json = response.content[0].text

    # Save raw response
    verification_raw.write_text(raw_json)

    # Extract JSON array
    cleaned = strip_json_fences(raw_json)
    start = cleaned.find('[')
    if start != -1:
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    cleaned = cleaned[start:i + 1]
                    break
    fixes = json.loads(cleaned)

    with open(verification_fixes, "w") as f:
        json.dump(fixes, f, indent=2)

    if not fixes:
        print(f"  No issues found", file=sys.stderr)
    else:
        print(f"  Found {len(fixes)} issue(s) to fix:", file=sys.stderr)
        var_by_id = {v["id"]: v for v in variations}

        for fix in fixes:
            vid = fix["id"]
            print(f"    {vid}: {fix['issue']} — {fix['reason']}", file=sys.stderr)

            if vid not in var_by_id:
                print(f"    WARNING: variation {vid} not found, skipping",
                      file=sys.stderr)
                continue

            v = var_by_id[vid]
            f_data = fix.get("fixes")
            if not f_data:
                continue

            if f_data.get("level_0_positive"):
                v["captions"]["0"]["positive"] = f_data["level_0_positive"]
            if f_data.get("level_0_negative"):
                v["captions"]["0"]["negative"] = f_data["level_0_negative"]

            if f_data.get("from") is not None and v.get("edits"):
                v["edits"][0]["from"] = f_data["from"]
                v["edits"][0]["to"] = f_data["to"]

    # Save updated L0
    with open(l0_out, "w") as f:
        json.dump(variations, f, indent=2)
    print(f"  Saved: {l0_out.name} ({len(variations)} variations)", file=sys.stderr)
    return variations


def step_5(scene_num: int, workers: int = 16):
    """Generate variation images via Gemini (parallel)."""
    step_header("5", f"Generate variation images (Gemini, {workers} workers)")
    files = scene_files(scene_num)
    sid = scene_id(scene_num)
    base_image_path = files["base_image"]

    with open(files["variations_l0"]) as f:
        variations = json.load(f)

    prompt_template = (PROMPTS_DIR / "image_2_apply_variation.txt").read_text()
    prompt_template = prompt_template.replace("[base image]", "").rstrip()

    require_key("GEMINI_API_KEY")

    img_dir = scene_dir(scene_num) / "images"
    img_dir.mkdir(exist_ok=True)

    # Always regenerate all images (variations were just regenerated)
    meta_path = scene_dir(scene_num) / f"{sid}_variation_metadata.json"
    existing = {}

    print(f"  Model   : {GEMINI_IMAGE_MODEL}", file=sys.stderr)
    print(f"  Variants: {len(variations)}", file=sys.stderr)

    # Build work items, skipping already-done
    to_generate = []  # (index, var, out_path, prompt)
    skipped = {}      # var_id -> metadata entry (reused from previous run)

    for i, var in enumerate(variations):
        var_id = var["id"]
        level0 = var["captions"]["0"]
        base_caption = level0["positive"]
        variation_caption = level0["negative"]
        out_path = img_dir / f"{var_id}.png"

        if var_id in existing and existing[var_id]["status"] == "ok" and out_path.exists():
            prev = existing[var_id]
            if (prev.get("base_caption") == base_caption and
                    prev.get("variation_caption") == variation_caption):
                skipped[var_id] = prev
                continue

        prompt = _build_variation_prompt(prompt_template, base_caption,
                                         variation_caption,
                                         edit_types=var.get("edit_types"), var=var)
        to_generate.append((i, var, out_path, prompt))

    if skipped:
        print(f"  Skipping {len(skipped)} unchanged variations", file=sys.stderr)

    print(f"  Generating {len(to_generate)} images with {workers} workers...",
          file=sys.stderr)

    # Run in parallel with separate processes
    results = {}
    completed_count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for item in to_generate:
            i, var, out_path, prompt = item
            var_id = var["id"]
            future = pool.submit(
                _worker_process,
                os.environ["GEMINI_API_KEY"],
                str(base_image_path),
                prompt,
                str(out_path),
                GEMINI_IMAGE_MODEL,
            )
            futures[future] = (var_id, var)

        for future in as_completed(futures):
            var_id, var = futures[future]
            status = future.result()
            level0 = var["captions"]["0"]
            out_path = img_dir / f"{var_id}.png"

            entry = {
                "id": var_id,
                "edit_types": var["edit_types"],
                "base_caption": level0["positive"],
                "variation_caption": level0["negative"],
                "image": str(out_path) if status == "ok" else None,
                "status": status,
            }
            results[var_id] = entry

            completed_count += 1
            icon = "ok" if status == "ok" else "FAIL"
            print(f"  [{completed_count}/{len(to_generate)}] {var_id}  {icon}",
                  flush=True)

    # Assemble metadata in original variation order
    metadata = []
    for var in variations:
        var_id = var["id"]
        if var_id in skipped:
            metadata.append(skipped[var_id])
        elif var_id in results:
            metadata.append(results[var_id])

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    ok = sum(1 for m in metadata if m["status"] == "ok")
    print(f"\n  Done! {ok}/{len(metadata)} succeeded", file=sys.stderr)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate variation captions and images for a CompoSET scene.",
    )
    parser.add_argument("scene", type=int, nargs="?",
                        help="Scene number (e.g. 5). Generates just this scene")
    parser.add_argument("--from", type=int, default=None, dest="from_scene",
                        help="Generate all available base scenes from this number")
    parser.add_argument("--count", "-n", type=int, default=None,
                        help="Max number of scenes to generate (default: all available)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers for image generation (default: 16)")
    args = parser.parse_args()

    # Determine which scenes to process
    if args.scene is not None:
        scene_nums = [args.scene]
    elif args.from_scene is not None:
        scene_nums = find_scenes_with_base(args.from_scene)
    else:
        parser.error("Provide a scene number or --from")

    if args.count is not None:
        scene_nums = scene_nums[:args.count]

    if not scene_nums:
        print("No base scenes available to process.", file=sys.stderr)
        raise SystemExit(0)

    print(f"Will generate variations for {len(scene_nums)} scene(s): "
          f"{', '.join(str(n) for n in scene_nums)}", file=sys.stderr)

    for scene_num in scene_nums:
        sid = scene_id(scene_num)
        sd = scene_dir(scene_num)
        files = scene_files(scene_num)

        if not files["caption_corrected"].exists():
            print(f"Skipping scene {scene_num}: no corrected caption", file=sys.stderr)
            continue

        print(f"\nScene: {sid} (#{scene_num})", file=sys.stderr)
        print(f"Dir  : {sd}", file=sys.stderr)

        variations = step_3(scene_num)
        variations = step_4(scene_num, variations)
        step_5(scene_num, workers=args.workers)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  VARIATION GENERATION COMPLETE: {sd}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    print(f"\n  Done. Review L0 captions + images in the QC interface.",
          file=sys.stderr)
    print(f"  Then run build_levels.py to generate L1/L2 captions.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
