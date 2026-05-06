#!/usr/bin/env python3
"""Build short / medium / long caption verbosities for all variations.

Generates three benchmark-aligned verbosity levels per variation via Claude:
  - short  (~4-8 words)   — ARO / Winoground / ColorSwap tier
  - medium (~8-12 words)  — COCO / SugarCrepe / BiVLC tier
  - long   (~20-40 words) — Flickr30k / SugarCrepe++ tier

Preserves captions."0" (L0, image-generation prompt) as historical.
Deletes captions."0.5" and captions."1" after new keys are written (Option A).
Discarded variations (status == "discarded" at L0 level) are never touched.
Halts if any variation has regen_requested — handle those first.

Usage:
    python pipeline/build_verbosities.py --dry-run                 # plan only
    python pipeline/build_verbosities.py --test s001_v21,s002_v12  # print JSON, no writes
    python pipeline/build_verbosities.py --scene 5                 # single scene
    python pipeline/build_verbosities.py                           # all scenes
    python pipeline/build_verbosities.py --from 11 --to 80 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import anthropic

from common import (
    CLAUDE_MODEL, DATA_DIR, PROMPTS_DIR,
    require_key, scene_files, strip_json_fences,
)


PROMPT_PATH = PROMPTS_DIR / "caption_7_build_verbosities.txt"


def _load_qc(scene_num: int) -> dict:
    qcp = scene_files(scene_num)["qc_decisions"]
    if not qcp.exists():
        return {}
    return json.loads(qcp.read_text())


def _discarded_ids(qc: dict) -> set[str]:
    return {
        vid for vid, entry in qc.items()
        if not vid.startswith("__")
        and isinstance(entry, dict)
        and entry.get("status") == "discarded"
    }


def _regen_requested_ids(qc: dict) -> list[str]:
    return [
        vid for vid, entry in qc.items()
        if not vid.startswith("__")
        and isinstance(entry, dict)
        and entry.get("regen_requested")
    ]


def classify_scene(scene_num: int) -> dict | None:
    """Plan the verbosity-build work for a scene.

    Returns None if required input files are missing.
    """
    files = scene_files(scene_num)
    if not files["variations_l0"].exists():
        return None
    if not files["caption_corrected"].exists():
        return None

    l0_vars = json.loads(files["variations_l0"].read_text())
    qc = _load_qc(scene_num)
    discarded = _discarded_ids(qc)
    regen = _regen_requested_ids(qc)

    targets = [v["id"] for v in l0_vars if v["id"] not in discarded]

    return {
        "scene": scene_num,
        "targets": targets,
        "discarded": sorted(discarded),
        "regen_requested": regen,
        "total": len(l0_vars),
    }


def _build_payload(l0_vars: list, target_ids: set[str]) -> list[dict]:
    out = []
    for v in l0_vars:
        if v["id"] not in target_ids:
            continue
        # Prefer edit_types_v2 when present (updated taxonomy)
        etypes = v.get("edit_types_v2") or v.get("edit_types", [])
        out.append({
            "id": v["id"],
            "edit_types": etypes,
            "edits": v.get("edits", []),
            "source_sentence": v.get("source_sentence", ""),
            "l0_positive": v["captions"]["0"]["positive"],
            "l0_negative": v["captions"]["0"]["negative"],
        })
    return out


def _call_claude(corrected: str, payload: list[dict],
                 client: anthropic.Anthropic) -> list[dict]:
    prompt_template = PROMPT_PATH.read_text()
    user_prompt = (
        f"CORRECTED CAPTION:\n{corrected}\n\n"
        f"---\nVARIATIONS:\n{json.dumps(payload, indent=2)}"
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=16384,
        system=prompt_template,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = resp.content[0].text
    cleaned = strip_json_fences(raw)
    start = cleaned.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    cleaned = cleaned[start:i + 1]
                    break
    return json.loads(cleaned)


def _apply_levels(variation: dict, lv: dict, only_tier: str | None = None) -> dict:
    """Write short/medium/long keys, delete legacy 0.5/1 keys. Keep 0 intact.

    If only_tier is given (one of 'short','medium','long'), only that tier is overwritten;
    other tiers are preserved as-is from the existing variation.
    """
    caps = dict(variation.get("captions", {}))
    if "0" not in caps:
        raise ValueError(f"{variation['id']} has no captions.0 — refusing to proceed")
    caps.pop("0.5", None)
    caps.pop("1", None)
    tiers = ("short", "medium", "long") if only_tier is None else (only_tier,)
    for t in tiers:
        caps[t] = {"base": lv[f"{t}_base"], "var": lv[f"{t}_var"]}
    nv = dict(variation)
    nv["captions"] = caps
    return nv


def _clear_verbosity_qc(qc: dict, target_ids: list[str]) -> None:
    """Clear l1_* and l05_* QC keys for regenerated variations.
    Keep base L0 QC state (status, timestamp, positive_original, etc.).
    Also drop scene-level __l1__ marker since L1 is gone/new.
    """
    for vid in target_ids:
        entry = qc.get(vid)
        if not isinstance(entry, dict):
            continue
        for k in list(entry.keys()):
            if k.startswith("l1_") or k.startswith("l05_"):
                del entry[k]
    qc.pop("__l1__", None)
    qc.pop("__l05__", None)


def run_scene(plan: dict, client: anthropic.Anthropic, only_tier: str | None = None) -> dict:
    n = plan["scene"]
    files = scene_files(n)
    target_ids = plan["targets"]
    if not target_ids:
        return {"scene": n, "regenerated": 0, "skipped_discarded": len(plan["discarded"])}

    l0_vars = json.loads(files["variations_l0"].read_text())
    corrected = files["caption_corrected"].read_text().strip()
    payload = _build_payload(l0_vars, set(target_ids))
    levels = _call_claude(corrected, payload, client)
    lv_by_id = {x["id"]: x for x in levels}

    # Build output variations list (preserving discarded entries untouched from existing file)
    existing = {}
    if files["variations"].exists():
        try:
            for v in json.loads(files["variations"].read_text()):
                existing[v["id"]] = v
        except Exception:
            pass

    out_vars = []
    missing = []
    for v in l0_vars:
        vid = v["id"]
        if vid in set(plan["discarded"]):
            # Preserve existing entry if present, else write from L0 untouched
            out_vars.append(existing.get(vid, dict(v)))
            continue
        if vid not in lv_by_id:
            missing.append(vid)
            # Use existing (unchanged) or a minimal shell
            out_vars.append(existing.get(vid, dict(v)))
            continue
        base_v = existing.get(vid, dict(v))
        # Clean any stale fields we don't want propagating
        base_v = dict(base_v)
        base_v.pop("source", None)
        base_v.pop("clause", None)
        # Ensure captions.0 is present (L0 historical)
        caps = dict(base_v.get("captions", {}))
        if "0" not in caps:
            # Bring L0 from the L0 file
            caps["0"] = dict(v["captions"]["0"])
        base_v["captions"] = caps
        out_vars.append(_apply_levels(base_v, lv_by_id[vid], only_tier=only_tier))

    files["variations"].write_text(json.dumps(out_vars, indent=2))

    # Clear verbosity-related QC state for regenerated variations
    qc = _load_qc(n)
    if qc:
        _clear_verbosity_qc(qc, target_ids)
        files["qc_decisions"].write_text(json.dumps(qc, indent=2))

    return {
        "scene": n,
        "regenerated": len(lv_by_id),
        "skipped_discarded": len(plan["discarded"]),
        "missing_from_claude": missing,
    }


def _print_test(plan: dict, client: anthropic.Anthropic, test_ids: set[str]):
    """Run the prompt on a subset of IDs (in any scene) and print JSON."""
    n = plan["scene"]
    files = scene_files(n)
    l0_vars = json.loads(files["variations_l0"].read_text())
    ids_in_scene = [v["id"] for v in l0_vars if v["id"] in test_ids]
    if not ids_in_scene:
        return None
    corrected = files["caption_corrected"].read_text().strip()
    payload = _build_payload(l0_vars, set(ids_in_scene))
    levels = _call_claude(corrected, payload, client)
    return levels


def main():
    parser = argparse.ArgumentParser(
        description="Build short/medium/long caption verbosities for variations.",
    )
    parser.add_argument("--scene", type=int, default=None)
    parser.add_argument("--from", dest="from_scene", type=int, default=1)
    parser.add_argument("--to", dest="to_scene", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only, no API calls or writes.")
    parser.add_argument("--test", type=str, default=None,
                        help="Comma-separated variation IDs (e.g. s001_v21,s002_v12). "
                             "Prints JSON to stdout without writing. For prompt validation.")
    parser.add_argument("--allow-regen-requested", action="store_true",
                        help="By default halts if any variation has regen_requested — use to override.")
    parser.add_argument("--only-tier", choices=["short", "medium", "long"], default=None,
                        help="Only overwrite the given tier in captions; preserve other tiers as-is.")
    args = parser.parse_args()

    # Select scenes
    if args.scene is not None:
        scene_nums = [args.scene]
    elif args.test:
        test_ids = set(x.strip() for x in args.test.split(",") if x.strip())
        scene_nums = sorted(set(int(vid[1:4]) for vid in test_ids if vid.startswith("s")))
    else:
        scene_nums = sorted(
            int(d.name) for d in DATA_DIR.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        scene_nums = [n for n in scene_nums
                      if n >= args.from_scene
                      and (args.to_scene is None or n <= args.to_scene)]

    plans = []
    for n in scene_nums:
        p = classify_scene(n)
        if p is None:
            continue
        plans.append(p)

    # Halt on regen_requested
    all_regen = [(p["scene"], vid) for p in plans for vid in p["regen_requested"]]
    if all_regen and not args.allow_regen_requested:
        print("ERROR: the following variations have regen_requested — handle first "
              "or pass --allow-regen-requested:", file=sys.stderr)
        for scene, vid in all_regen:
            print(f"  s{scene:03d} {vid}", file=sys.stderr)
        sys.exit(2)

    # Summary
    total_targets = sum(len(p["targets"]) for p in plans)
    total_discarded = sum(len(p["discarded"]) for p in plans)
    print(f"Scenes: {len(plans)}  variations: {total_targets + total_discarded} "
          f"(generate: {total_targets}, skip-discarded: {total_discarded})",
          file=sys.stderr)

    # Test mode
    if args.test:
        require_key("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic()
        test_ids = set(x.strip() for x in args.test.split(",") if x.strip())
        for p in plans:
            levels = _print_test(p, client, test_ids)
            if levels:
                print(json.dumps(levels, indent=2))
        return

    # Dry-run
    if args.dry_run:
        for p in plans:
            print(f"  s{p['scene']:03d}: generate={len(p['targets'])} "
                  f"discarded={len(p['discarded'])}",
                  file=sys.stderr)
        return

    # Full run
    require_key("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_scene, p, client, args.only_tier): p for p in plans if p["targets"]}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                missing = res.get("missing_from_claude") or []
                extra = f" MISSING={missing}" if missing else ""
                print(f"  s{p['scene']:03d}: regen={res['regenerated']} "
                      f"disc={res['skipped_discarded']}{extra}", file=sys.stderr)
            except Exception as e:
                errors.append((p["scene"], str(e)))
                print(f"  s{p['scene']:03d}: ERROR: {e}", file=sys.stderr)

    print(f"\nDone. {len(results)} ok, {len(errors)} failed.", file=sys.stderr)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
