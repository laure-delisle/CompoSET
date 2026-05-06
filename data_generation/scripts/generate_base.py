#!/usr/bin/env python3
"""Generate base caption and image for a CompoSET scene.

Runs steps 0-2 of the pipeline:
  0  Sample scene parameters from scene_settings.csv
  1  Generate base caption via Claude
  1b Generate base image via Gemini
  2  Verify/correct caption against generated image via Gemini

Usage:
    python generate_base.py 5
    python generate_base.py 5 --seed 42
    python generate_base.py --resume              # auto-detect next scene
    python generate_base.py --resume-from 11      # start from scene 11

Output (in data/005/):
    s005_scene_params.txt
    s005_caption_base.txt
    s005_caption_corrected.txt
    s005_verification_raw.txt
    s005_image_metadata.json
    images/s005_base.png
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import os
import random
import re
import sys
from io import BytesIO
from pathlib import Path

import anthropic
from google import genai
from google.genai import types
from PIL import Image

from common import (
    PROMPTS_DIR, CLAUDE_MODEL, GEMINI_IMAGE_MODEL,
    require_key, get_scene_setting, scene_id, scene_dir, scene_files,
    parse_caption_file, parse_scene_parameters, step_header,
    find_last_scene,
)

CANONICAL_SIZE = (1408, 768)

# ── Prompt helpers ──────────────────────────────────────────────────────────

SUBJECTS_OPTIONS = [0, 1, 2, "several"]
FRAMING_OPTIONS = ["close-up", "medium shot", "full shot"]


def load_prompt(path: Path) -> str:
    text = path.read_text()
    text = re.sub(r"\s+\(e\.g\.[^)]*\)", "", text)
    return text


def fill_prompt(template: str, setting: str, subjects, framing: str) -> str:
    return (template
            .replace("{setting}", setting)
            .replace("{subjects}", str(subjects))
            .replace("{framing}", framing))


def build_image_prompt(base_caption: str, details: dict[str, str],
                       framing: str) -> str:
    template = (PROMPTS_DIR / "image_1_generate_base.txt").read_text()
    parts = [base_caption] + list(details.values())
    body = " ".join(parts)
    return template.replace("{base caption and details}", body).replace("{framing}", framing)


# ── Step implementations ────────────────────────────────────────────────────

def step_0(scene_num: int, seed: int, resume: bool) -> dict:
    """Sample scene parameters."""
    step_header("0", "Sample scene parameters")
    files = scene_files(scene_num)
    out = files["scene_params"]

    if resume and out.exists():
        print(f"  [resume] {out.name} exists, skipping", file=sys.stderr)
        return parse_scene_parameters(out)

    row = get_scene_setting(scene_num)
    random.seed(seed + scene_num)
    subjects = random.choice(SUBJECTS_OPTIONS)
    framing = random.choice(FRAMING_OPTIONS)

    params = {
        "setting": row["setting"],
        "tier": row["tier"],
        "indoor_outdoor": row["indoor_outdoor"],
        "subjects": str(subjects),
        "framing": framing,
    }

    out.write_text(
        f"setting:  {params['setting']}\n"
        f"subjects: {params['subjects']}\n"
        f"framing:  {params['framing']}\n"
    )
    for k, v in params.items():
        print(f"  {k}: {v}", file=sys.stderr)
    return params


def recent_captions_block(scene_num: int, window: int = 5) -> str:
    """Build a 'RECENT SCENES — avoid reusing' block from the last N scenes'
    corrected-or-base captions. Claude reads raw captions and extracts the
    palette/objects/garments itself, avoiding a brittle hardcoded vocab."""
    from common import DATA_DIR
    recent = []
    for n in range(scene_num - 1, 0, -1):
        if len(recent) >= window:
            break
        sdir = DATA_DIR / f"{n:03d}"
        sid = f"s{n:03d}"
        cap = sdir / f"{sid}_caption_corrected.txt"
        if not cap.exists():
            cap = sdir / f"{sid}_caption_base.txt"
        if cap.exists():
            recent.append(f"scene {n:03d}:\n{cap.read_text().strip()}")
    if not recent:
        return ""
    body = "\n\n".join(reversed(recent))
    return (
        "\n---\nRECENT SCENES — avoid reusing the same garments, colors, "
        "objects, actions, patterns, or overall composition as in the last "
        f"{len(recent)} scenes below. Deliberately pick different choices "
        "across every dimension (not just one). This is the most important "
        "source of diversity in the benchmark.\n\n" + body + "\n"
    )


def step_1(scene_num: int, params: dict, resume: bool) -> str:
    """Generate base caption via Claude."""
    step_header("1", "Generate base caption (Claude)")
    files = scene_files(scene_num)
    out = files["caption_base"]

    if resume and out.exists():
        print(f"  [resume] {out.name} exists, skipping", file=sys.stderr)
        return out.read_text()

    require_key("ANTHROPIC_API_KEY")
    prompt_template = load_prompt(PROMPTS_DIR / "caption_1_generate_base.txt")
    prompt = fill_prompt(prompt_template, params["setting"], params["subjects"],
                         params["framing"])
    prompt += recent_captions_block(scene_num)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    caption_text = response.content[0].text

    out.write_text(caption_text)
    print(f"  Saved: {out.name}", file=sys.stderr)
    print(f"  Preview: {caption_text[:120]}...", file=sys.stderr)
    return caption_text


def step_1b(scene_num: int, params: dict, seed: int, resume: bool) -> Path:
    """Generate base image via Gemini."""
    step_header("1b", "Generate base image (Gemini)")
    files = scene_files(scene_num)
    out_image = files["base_image"]

    if resume and out_image.exists():
        print(f"  [resume] {out_image.name} exists, skipping", file=sys.stderr)
        return out_image

    base_caption, details = parse_caption_file(files["caption_base"])

    random.seed(seed + scene_num)
    image_prompt = build_image_prompt(base_caption, details, params["framing"])

    require_key("GEMINI_API_KEY")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"  Generating image with {GEMINI_IMAGE_MODEL}...", file=sys.stderr)
    MAX_RETRIES = 5
    image = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[image_prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        candidate = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                candidate = Image.open(BytesIO(part.inline_data.data))
                break
        if candidate is None:
            print(f"  attempt {attempt}: no image returned", file=sys.stderr)
            continue
        if candidate.size != CANONICAL_SIZE:
            print(f"  attempt {attempt}: wrong size {candidate.size}, retrying",
                  file=sys.stderr)
            continue
        image = candidate
        print(f"  attempt {attempt}: ok {image.size}", file=sys.stderr)
        break

    if image is None:
        print(f"  ERROR: failed after {MAX_RETRIES} attempts", file=sys.stderr)
        raise SystemExit(2)

    image.save(out_image)
    print(f"  Saved: {out_image.name}", file=sys.stderr)

    metadata = {
        "model": GEMINI_IMAGE_MODEL,
        "prompt": image_prompt,
        "image": str(out_image),
    }
    sid = scene_id(scene_num)
    meta_path = scene_dir(scene_num) / f"{sid}_image_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return out_image


def step_2(scene_num: int, resume: bool) -> str:
    """Verify caption against base image via Gemini."""
    step_header("2", "Verify caption vs image (Gemini)")
    files = scene_files(scene_num)
    out = files["caption_corrected"]

    if resume and out.exists():
        print(f"  [resume] {out.name} exists, skipping", file=sys.stderr)
        return out.read_text()

    image_path = files["base_image"]
    caption_text = files["caption_base"].read_text().strip()

    require_key("GEMINI_API_KEY")
    system_prompt = load_prompt(PROMPTS_DIR / "caption_2_edit_from_image.txt")

    import PIL.Image
    image = PIL.Image.open(image_path)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"  Verifying against {image_path.name}...", file=sys.stderr)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, caption_text],
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    raw_response = response.text

    # Save raw for auditing
    sd = scene_dir(scene_num)
    sid = scene_id(scene_num)
    (sd / f"{sid}_verification_raw.txt").write_text(raw_response)

    # Extract corrected caption
    m = re.search(r"<<<CORRECTED>>>\s*\n(.*?)\n\s*<<<END>>>", raw_response, re.DOTALL)
    if m:
        corrected = m.group(1).strip()
    else:
        print("  WARNING: <<<CORRECTED>>> block not found, using original caption",
              file=sys.stderr)
        corrected = caption_text

    out.write_text(corrected)
    print(f"  Saved: {out.name}", file=sys.stderr)
    return corrected


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate base caption and image for a CompoSET scene.",
    )
    parser.add_argument("scene", type=int, nargs="?",
                        help="Scene number (e.g. 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Auto-detect next scene to generate")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Start from this scene number")
    parser.add_argument("--count", "-n", type=int, default=1,
                        help="Number of scenes to generate (default: 1)")
    args = parser.parse_args()

    if args.resume_from is not None:
        start = args.resume_from
    elif args.resume:
        last = find_last_scene()
        start = (last + 1) if last else 1
        print(f"Resuming from scene {start}", file=sys.stderr)
    elif args.scene is not None:
        start = args.scene
    else:
        parser.error("Provide a scene number, --resume, or --resume-from")

    do_resume = args.resume or args.resume_from is not None

    for scene_num in range(start, start + args.count):
        sid = scene_id(scene_num)
        sd = scene_dir(scene_num)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "images").mkdir(exist_ok=True)

        print(f"\nScene: {sid} (#{scene_num})", file=sys.stderr)
        print(f"Dir  : {sd}", file=sys.stderr)

        params = step_0(scene_num, args.seed, do_resume)
        step_1(scene_num, params, do_resume)
        step_1b(scene_num, params, args.seed, do_resume)
        step_2(scene_num, do_resume)

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  BASE GENERATION COMPLETE: {sd}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    if args.count > 1:
        print(f"\n  Generated {args.count} scenes ({start}-{start + args.count - 1})",
              file=sys.stderr)
    print(f"  Review base images and corrected captions, then run", file=sys.stderr)
    print(f"  generate_variations.py", file=sys.stderr)


if __name__ == "__main__":
    main()
