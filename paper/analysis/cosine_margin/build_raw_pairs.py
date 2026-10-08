"""Build raw_pairs.csv for cosine-margin analysis.

Reads per-(model, benchmark) CSVs written by `python -m eval.run` (default: results/,
override with COMPOSET_RESULTS_DIR)
and emits long-format margin rows: (model, benchmark, pair_id, scene_id,
variation_id, edit_type, tier, direction, margin, decision_correct).

Margin sign convention (canonical):
  I2T margin = sim(image_a, caption_a) - sim(image_a, caption_b)
  T2I margin = sim(image_a, caption_a) - sim(image_b, caption_a)

Idempotent on (model, benchmark, pair_id, direction): re-runs append only new rows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(os.environ.get("COMPOSET_RESULTS_DIR", Path(__file__).resolve().parents[3] / "results"))
OUT_DIR = Path(__file__).resolve().parents[0]
OUT_CSV = OUT_DIR / "raw_pairs.csv"

PRIOR_BENCHMARKS = [
    "winoground", "bivlc", "naturalbench", "colorbench",
    "whatsup", "colorswap", "mmvp", "sugarcrepe", "sugarcrepe_pp",
]

ALL_MODELS_EXPECTED = [
    "clip-b16-laion2b", "clip-b16-laion400m", "clip-b16-openai", "clip-b32",
    "clip-h14-laion2b", "clip-l14", "clip-l14-laion2b", "clove-b32",
    "dfn-b16", "dfn-h14", "dreamlip-b16-merged30m", "eva02-b16", "eva02-l14",
    "jina-embeddings-v4", "metaclip2-h14", "negclip-b32",
    "siglip2-b16", "siglip2-l16-384", "siglip2-so400m-378",
    "siglip-b16", "siglip-l16-384", "siglip-so400m-384",
]


def _bin(x):
    """Coerce score column to 0/1; NaN if missing."""
    if pd.isna(x):
        return np.nan
    try:
        return int(x)
    except Exception:
        return np.nan


def process_composet(model: str) -> pd.DataFrame:
    p = RESULTS_DIR / "composet" / f"{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # cb = caption_base, cv = caption_var. base image / var image columns:
    # sim_base_cb = sim(image_base, caption_base) -- positive
    # sim_base_cv = sim(image_base, caption_var)
    # sim_var_cb  = sim(image_var,  caption_base)
    # sim_var_cv  = sim(image_var,  caption_var) -- positive on var-image side
    # Canonical positive (a) = base, negative (b) = var.
    rows = []
    for _, r in df.iterrows():
        pid = f"{r['scene_id']}|{r['variation_id']}|{r['tier']}"
        i2t_corr = _bin(r.get("i2t_score"))
        t2i_corr = _bin(r.get("t2i_score"))
        # I2T: sim_base_cb - sim_base_cv
        rows.append({
            "model": model, "benchmark": "composet", "pair_id": pid,
            "scene_id": r["scene_id"], "variation_id": r["variation_id"],
            "edit_type": r["edit_type"], "tier": r["tier"],
            "direction": "I2T",
            "margin": float(r["sim_base_cb"]) - float(r["sim_base_cv"]),
            "decision_correct": i2t_corr,
        })
        # T2I: sim_base_cb - sim_var_cb
        rows.append({
            "model": model, "benchmark": "composet", "pair_id": pid,
            "scene_id": r["scene_id"], "variation_id": r["variation_id"],
            "edit_type": r["edit_type"], "tier": r["tier"],
            "direction": "T2I",
            "margin": float(r["sim_base_cb"]) - float(r["sim_var_cb"]),
            "decision_correct": t2i_corr,
        })
    return pd.DataFrame(rows)


def process_winoground_like(model: str, benchmark: str) -> pd.DataFrame:
    """Schema with sim_00/01/10/11 (winoground, bivlc) or sim_aa/ab/ba/bb
    (whatsup, colorswap, mmvp). Both are 4-cell symmetric."""
    p = RESULTS_DIR / f"{benchmark}_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    cols = set(df.columns)
    if {"sim_00", "sim_01", "sim_10", "sim_11"}.issubset(cols):
        s_aa, s_ab, s_ba, s_bb = "sim_00", "sim_01", "sim_10", "sim_11"
    elif {"sim_aa", "sim_ab", "sim_ba", "sim_bb"}.issubset(cols):
        s_aa, s_ab, s_ba, s_bb = "sim_aa", "sim_ab", "sim_ba", "sim_bb"
    else:
        return pd.DataFrame()
    rows = []
    for idx, r in df.iterrows():
        pid = str(r.get("id", idx))
        i2t_corr = _bin(r.get("i2t_score"))
        t2i_corr = _bin(r.get("t2i_score"))
        rows.append({
            "model": model, "benchmark": benchmark, "pair_id": pid,
            "scene_id": np.nan, "variation_id": np.nan,
            "edit_type": np.nan, "tier": np.nan,
            "direction": "I2T",
            "margin": float(r[s_aa]) - float(r[s_ab]),
            "decision_correct": i2t_corr,
        })
        rows.append({
            "model": model, "benchmark": benchmark, "pair_id": pid,
            "scene_id": np.nan, "variation_id": np.nan,
            "edit_type": np.nan, "tier": np.nan,
            "direction": "T2I",
            "margin": float(r[s_aa]) - float(r[s_ba]),
            "decision_correct": t2i_corr,
        })
    return pd.DataFrame(rows)


def process_naturalbench(model: str) -> pd.DataFrame:
    """sim_pos / sim_neg only — asymmetric (T2I-only). i2t is blank."""
    p = RESULTS_DIR / f"naturalbench_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        pid = str(r["id"])
        t2i_corr = _bin(r.get("t2i_score"))
        # Only T2I direction: prompt = the caption being asked, sim_pos vs sim_neg
        # represent two captions for the same image. T2I-style margin per spec.
        rows.append({
            "model": model, "benchmark": "naturalbench", "pair_id": pid,
            "scene_id": np.nan, "variation_id": np.nan,
            "edit_type": np.nan, "tier": np.nan,
            "direction": "T2I",
            "margin": float(r["sim_pos"]) - float(r["sim_neg"]),
            "decision_correct": t2i_corr,
        })
    return pd.DataFrame(rows)


def process_colorbench(model: str) -> pd.DataFrame:
    """sim_base / sim_foil — image-side foils, single caption. I2T-margin
    well-defined (one image, two captions? no — one caption, two images for
    Robustness). Use i2t_score / t2i_score columns as ground truth and emit
    a single direction tagged conservatively as T2I (image foils, fixed text)."""
    p = RESULTS_DIR / f"colorbench_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        pid = str(r["id"])
        # Use the available score; ColorBench has both i2t and t2i for some rows
        # but the asymmetric sim layout fits T2I (one prompt, two images).
        i2t_corr = _bin(r.get("i2t_score"))
        t2i_corr = _bin(r.get("t2i_score"))
        # Pick whichever is non-nan as the decision tag, prefer T2I (matches sim shape)
        if not pd.isna(t2i_corr):
            corr, direction = t2i_corr, "T2I"
        elif not pd.isna(i2t_corr):
            corr, direction = i2t_corr, "I2T"
        else:
            continue
        rows.append({
            "model": model, "benchmark": "colorbench", "pair_id": pid,
            "scene_id": np.nan, "variation_id": np.nan,
            "edit_type": np.nan, "tier": np.nan,
            "direction": direction,
            "margin": float(r["sim_base"]) - float(r["sim_foil"]),
            "decision_correct": corr,
        })
    return pd.DataFrame(rows)


def process_sugarcrepe(model: str) -> pd.DataFrame:
    """Text-only foils — I2T direction only. sim_pos = sim(image, pos_cap),
    sim_neg = sim(image, neg_cap)."""
    p = RESULTS_DIR / f"sugarcrepe_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        pid = str(r["id"])
        i2t_corr = _bin(r.get("i2t_score"))
        if pd.isna(i2t_corr):
            continue
        rows.append({
            "model": model, "benchmark": "sugarcrepe", "pair_id": pid,
            "scene_id": np.nan, "variation_id": np.nan,
            "edit_type": np.nan, "tier": np.nan,
            "direction": "I2T",
            "margin": float(r["sim_pos"]) - float(r["sim_neg"]),
            "decision_correct": i2t_corr,
        })
    return pd.DataFrame(rows)


def process_sugarcrepe_pp(model: str) -> pd.DataFrame:
    """SugarCrepe++: sim_pos1, sim_pos2, sim_neg + filename (image swap to
    flip pos/neg roles is not in the file — text-foil + paraphrase). I2T-only
    margin = min(sim_pos1, sim_pos2) - sim_neg (worst-case margin) per pair."""
    p = RESULTS_DIR / f"sugarcrepe_pp_{model}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        pid = str(r["id"])
        i2t_corr = _bin(r.get("i2t_score"))
        if pd.isna(i2t_corr):
            continue
        m1 = float(r["sim_pos1"]) - float(r["sim_neg"])
        m2 = float(r["sim_pos2"]) - float(r["sim_neg"])
        # Use mean margin over the two positive paraphrases.
        margin = 0.5 * (m1 + m2)
        rows.append({
            "model": model, "benchmark": "sugarcrepe_pp", "pair_id": pid,
            "scene_id": np.nan, "variation_id": np.nan,
            "edit_type": np.nan, "tier": np.nan,
            "direction": "I2T",
            "margin": margin,
            "decision_correct": i2t_corr,
        })
    return pd.DataFrame(rows)


HANDLERS = {
    "composet": process_composet,
    "winoground": lambda m: process_winoground_like(m, "winoground"),
    "bivlc": lambda m: process_winoground_like(m, "bivlc"),
    "whatsup": lambda m: process_winoground_like(m, "whatsup"),
    "colorswap": lambda m: process_winoground_like(m, "colorswap"),
    "mmvp": lambda m: process_winoground_like(m, "mmvp"),
    "naturalbench": process_naturalbench,
    "colorbench": process_colorbench,
    "sugarcrepe": process_sugarcrepe,
    "sugarcrepe_pp": process_sugarcrepe_pp,
}


def main():
    # Discover models present
    composet_models = sorted(p.stem for p in (RESULTS_DIR / "composet").glob("*.csv"))
    print(f"composet models: {len(composet_models)}", file=sys.stderr)

    # Load existing for idempotency
    existing_keys = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        existing_keys = set(zip(prev.model, prev.benchmark, prev.pair_id, prev.direction))
        print(f"existing rows: {len(prev)}", file=sys.stderr)

    all_dfs = []
    benchmarks = ["composet"] + PRIOR_BENCHMARKS
    for model in composet_models:
        for bm in benchmarks:
            handler = HANDLERS[bm]
            df = handler(model)
            if df.empty:
                continue
            # idempotency filter
            if existing_keys:
                mask = [
                    (m, b, str(p), d) not in existing_keys
                    for m, b, p, d in zip(df.model, df.benchmark, df.pair_id, df.direction)
                ]
                df = df[mask]
            if not df.empty:
                all_dfs.append(df)
        print(f"  done {model}", file=sys.stderr)

    if not all_dfs:
        print("nothing new to write", file=sys.stderr)
        return
    new_rows = pd.concat(all_dfs, ignore_index=True)
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        out = pd.concat([prev, new_rows], ignore_index=True)
    else:
        out = new_rows
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} rows ({len(new_rows)} new) to {OUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
