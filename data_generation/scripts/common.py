"""Shared helpers for the CompoSET pipeline."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
DATA_DIR = ROOT_DIR / "data"
SETTINGS_CSV = ROOT_DIR / "scene_settings.csv"

GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
CLAUDE_MODEL = "claude-opus-4-6"

# ── API keys ────────────────────────────────────────────────────────────────

def require_key(name: str):
    """Verify the named API key environment variable is set."""
    if not os.environ.get(name):
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        raise SystemExit(1)


# ── Scene settings ──────────────────────────────────────────────────────────

def get_scene_setting(scene_num: int) -> dict:
    """Look up scene_settings.csv row by scene number (1-indexed)."""
    rows = list(csv.DictReader(SETTINGS_CSV.open()))
    for row in rows:
        if int(row["id"]) == scene_num:
            return row
    raise ValueError(f"No row with id={scene_num} in {SETTINGS_CSV}")


# ── Naming conventions ──────────────────────────────────────────────────────

def scene_id(scene_num: int) -> str:
    """Format scene number as 3-digit ID: 5 -> 's005'."""
    return f"s{scene_num:03d}"


def scene_dir(scene_num: int) -> Path:
    """Return data directory for a scene: data/005/."""
    return DATA_DIR / f"{scene_num:03d}"


def scene_files(scene_num: int) -> dict[str, Path]:
    """Return a dict of all standard file paths for a scene."""
    sid = scene_id(scene_num)
    d = scene_dir(scene_num)
    return {
        "scene_params":         d / f"{sid}_scene_params.txt",
        "caption_base":         d / f"{sid}_caption_base.txt",
        "caption_corrected":    d / f"{sid}_caption_corrected.txt",
        "variations_l0":        d / f"{sid}_caption_variations_l0.json",
        "variations":           d / f"{sid}_caption_variations.json",
        "qc_decisions":         d / "qc_decisions.json",
        "base_image":           d / "images" / f"{sid}_base.png",
    }


def var_image_path(scene_num: int, var_num: int) -> Path:
    """Return image path for a variation: data/005/images/s005_v03.png."""
    sid = scene_id(scene_num)
    return scene_dir(scene_num) / "images" / f"{sid}_v{var_num:02d}.png"


# ── Caption parsing ─────────────────────────────────────────────────────────

def parse_caption_file(path: Path) -> tuple[str, dict[str, str]]:
    """Parse a caption file into (base_caption, detail_sentences)."""
    text = path.read_text()
    m = re.search(r'base_caption:\s*"(.+?)"', text, re.DOTALL)
    base_caption = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    details = {}
    for m in re.finditer(r'^\s*(\w+):\s*"(.+?)"', text, re.MULTILINE | re.DOTALL):
        key = m.group(1)
        if key == "base_caption":
            continue
        details[key] = re.sub(r"\s+", " ", m.group(2)).strip()

    return base_caption, details


def serialize_caption(base_caption: str, details: dict[str, str]) -> str:
    """Serialize caption back to the standard text format."""
    lines = [f'base_caption: "{base_caption}"', "detail_sentences:"]
    for key, val in details.items():
        lines.append(f'  {key}: "{val}"')
    return "\n".join(lines) + "\n"


# ── Scene parameters ────────────────────────────────────────────────────────

def parse_scene_parameters(path: Path) -> dict:
    """Parse scene parameters file into a dict."""
    params = {}
    for line in path.read_text().strip().splitlines():
        key, _, val = line.partition(":")
        params[key.strip()] = val.strip()
    return params


# ── Resume detection ────────────────────────────────────────────────────────

def find_last_scene() -> int | None:
    """Find the highest scene number in data/ that has a base image."""
    if not DATA_DIR.exists():
        return None
    nums = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit():
            n = int(d.name)
            files = scene_files(n)
            if files["base_image"].exists():
                nums.append(n)
    return max(nums) if nums else None


def find_last_scene_with_variations() -> int | None:
    """Find the highest scene number that has a variations JSON."""
    if not DATA_DIR.exists():
        return None
    nums = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit():
            n = int(d.name)
            files = scene_files(n)
            if files["variations"].exists():
                nums.append(n)
    return max(nums) if nums else None


def find_scenes_with_base(from_num: int = 1) -> list[int]:
    """Find all scene numbers >= from_num that have a corrected caption."""
    if not DATA_DIR.exists():
        return []
    nums = []
    for d in DATA_DIR.iterdir():
        if d.is_dir() and d.name.isdigit():
            n = int(d.name)
            if n >= from_num:
                files = scene_files(n)
                if files["caption_corrected"].exists():
                    nums.append(n)
    return sorted(nums)


# ── Misc ────────────────────────────────────────────────────────────────────

def strip_json_fences(text: str) -> str:
    """Remove markdown code fences from JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text


def step_header(step: str, desc: str):
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  STEP {step}: {desc}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
