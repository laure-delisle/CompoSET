"""Text-deviation metrics for CompoSET + 6 prior benchmarks.

Direct-load fallback (cache was capped at limit=20 from a smoke run; rebuilding
the full cache would also re-save images for ~thousands of pairs, wasting disk
and time. We only need captions for text-deviation, so we load them directly).

Outputs (long-format CSV) written to
this folder (paper/analysis/text_deviation/).

Idempotent: dedupes on (benchmark, pair_id, density_level, metric_name).
"""

from __future__ import annotations

from huggingface_hub import snapshot_download
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "paper" / "analysis" / "text_deviation"
OUT.mkdir(parents=True, exist_ok=True)
RAW_CSV = OUT / "raw_pairs.csv"

PARQUET = Path(os.environ.get("COMPOSET_DATA_DIR") or snapshot_download("CompoSET/CompoSET", repo_type="dataset")) / "variations.parquet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={DEVICE}")

# ----------------------------------------------------------------------------
# Load existing rows for idempotency
# ----------------------------------------------------------------------------
if RAW_CSV.exists():
    existing = pd.read_csv(RAW_CSV)
    seen = set(zip(existing["benchmark"], existing["pair_id"],
                   existing["density_level"].fillna(""),
                   existing["metric_name"]))
    print(f"[init] loaded {len(existing):,} existing rows; will dedupe.")
else:
    existing = pd.DataFrame()
    seen = set()


def already(benchmark, pair_id, density, metric):
    return (benchmark, pair_id, density or "", metric) in seen


# ----------------------------------------------------------------------------
# Collect (benchmark, pair_id, scene_id, variation_id, edit_type, density_level,
#          caption_a, caption_b) tuples from all 7 sources.
# ----------------------------------------------------------------------------
def collect_pairs():
    pairs = []  # list of dicts

    # --- CompoSET: read parquet directly, 3 densities per row ---
    print("[load] compoSET (parquet)")
    df = pd.read_parquet(PARQUET)
    for _, row in df.iterrows():
        pid = row["id"]  # e.g. s001_v01
        sid = row["scene_id"]
        vid = pid.split("_", 1)[1]
        et = row["edit_type"]
        caps = row["captions"]
        for density in ("short", "medium", "long"):
            base = caps[density]["base"]
            var = caps[density]["var"]
            if base == var:
                continue
            pairs.append({
                "benchmark": "compoSET",
                "pair_id": pid,
                "scene_id": sid,
                "variation_id": vid,
                "edit_type": et,
                "density_level": density,
                "caption_a": base,
                "caption_b": var,
            })

    # --- Prior benchmarks: load via composet_eval loaders (in-process) ---
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for composet_eval
    from composet_eval.benchmarks import LOADERS

    # Per-benchmark pair_id transform to match image_deviation's scheme
    def transform_pid(bm, raw):
        if bm == "winoground":
            return f"wg_{raw}"
        if bm == "bivlc":
            return f"bv_{raw}"
        return raw  # whatsup, sugarcrepe, mmvp, colorswap match raw

    benches = ["winoground", "bivlc", "whatsup", "sugarcrepe", "mmvp", "colorswap"]
    for bm in benches:
        print(f"[load] {bm}")
        cls = LOADERS[bm]
        n = 0
        skipped = 0
        for p in cls().iter_pairs():
            if not p.has_text_pair:
                continue
            cap_a = p.caption_positive
            cap_b = p.caption_negative
            if not cap_a or not cap_b or cap_a == cap_b:
                skipped += 1
                continue
            pid = transform_pid(bm, p.pair_id)
            pairs.append({
                "benchmark": bm,
                "pair_id": pid,
                "scene_id": "",
                "variation_id": "",
                "edit_type": "",
                "density_level": "",
                "caption_a": cap_a,
                "caption_b": cap_b,
            })
            n += 1
        print(f"   -> {n:,} pairs (skipped {skipped} with empty/equal captions)")

    return pairs


t0 = time.time()
all_pairs = collect_pairs()
print(f"[load] total pairs (incl. compoSET density expansion): {len(all_pairs):,}")
print(f"[load] elapsed {time.time()-t0:.1f}s")

# Per-benchmark counts (count unique pair_ids per benchmark, ignoring density)
df_pairs = pd.DataFrame(all_pairs)
n_by_bm = df_pairs.groupby("benchmark")["pair_id"].nunique().to_dict()
print(f"[load] unique pair_id per benchmark: {n_by_bm}")

# ----------------------------------------------------------------------------
# Tokenization (CLIP BPE) for token-level metrics
# ----------------------------------------------------------------------------
import open_clip
from open_clip import get_tokenizer

print("[init] loading CLIP ViT-B-32 (openai)")
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.eval().to(DEVICE)
clip_tokenizer = get_tokenizer("ViT-B-32")

# Use simple_tokenizer.encode for variable-length token sequences
from open_clip.tokenizer import SimpleTokenizer  # type: ignore
_simple = SimpleTokenizer()


def tokenize_clip(text: str) -> list[int]:
    """Return BPE token IDs (no padding)."""
    return _simple.encode(text)


import editdistance


def to_unicode_str(seq: list[int]) -> str:
    # CLIP vocab ~49k, well below 0x10FFFF; safe to map id -> chr(id+1).
    return "".join(chr(i + 1) for i in seq)


# ----------------------------------------------------------------------------
# Sentence-transformers + CLIP text encoder
# ----------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

print("[init] loading sentence-transformers all-mpnet-base-v2")
mpnet = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
print("[init] loading sentence-transformers all-MiniLM-L6-v2")
minilm = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


def st_cosine(model, caps_a: list[str], caps_b: list[str], batch_size=128) -> np.ndarray:
    emb_a = model.encode(caps_a, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=True, convert_to_numpy=True)
    emb_b = model.encode(caps_b, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=True, convert_to_numpy=True)
    return (emb_a * emb_b).sum(-1)


def clip_text_cosine(caps_a: list[str], caps_b: list[str], batch_size=256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for caps in (caps_a, caps_b):
            embs = []
            for i in tqdm(range(0, len(caps), batch_size), desc="clip_text"):
                batch = caps[i:i + batch_size]
                tok = clip_tokenizer(batch).to(DEVICE)
                e = clip_model.encode_text(tok)
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu().numpy())
            out.append(np.concatenate(embs, axis=0))
    emb_a, emb_b = out
    return (emb_a * emb_b).sum(-1)


# ----------------------------------------------------------------------------
# Compute all metrics in batch
# ----------------------------------------------------------------------------
caps_a = [p["caption_a"] for p in all_pairs]
caps_b = [p["caption_b"] for p in all_pairs]

print("[token] tokenizing + Levenshtein/Jaccard")
tokens_a = [tokenize_clip(c) for c in tqdm(caps_a)]
tokens_b = [tokenize_clip(c) for c in tqdm(caps_b)]

token_levs = []
token_jacs = []
frac_changed = []
len_a = []
len_b = []
for ta, tb in zip(tokens_a, tokens_b):
    sa, sb = to_unicode_str(ta), to_unicode_str(tb)
    lev = editdistance.eval(sa, sb)
    set_a, set_b = set(ta), set(tb)
    inter = len(set_a & set_b)
    union = len(set_a | set_b) or 1
    jac = inter / union
    denom = max(len(ta), len(tb), 1)
    token_levs.append(lev)
    token_jacs.append(jac)
    frac_changed.append(lev / denom)
    len_a.append(len(ta))
    len_b.append(len(tb))

token_levs = np.array(token_levs, dtype=float)
token_jacs = np.array(token_jacs, dtype=float)
frac_changed = np.array(frac_changed, dtype=float)
len_a = np.array(len_a, dtype=float)
len_b = np.array(len_b, dtype=float)

print("[embed] mpnet")
mpnet_cos = st_cosine(mpnet, caps_a, caps_b)
print("[embed] minilm")
minilm_cos = st_cosine(minilm, caps_a, caps_b)
print("[embed] clip text")
clip_cos = clip_text_cosine(caps_a, caps_b)

# ----------------------------------------------------------------------------
# Build long-format rows
# ----------------------------------------------------------------------------
metric_arrays = {
    "token_levenshtein": token_levs,
    "token_jaccard": token_jacs,
    "token_jaccard_dist": 1 - token_jacs,
    "frac_tokens_changed": frac_changed,
    "tokens_a_len": len_a,
    "tokens_b_len": len_b,
    "mpnet_cos": mpnet_cos,
    "mpnet_cos_dist": 1 - mpnet_cos,
    "minilm_cos": minilm_cos,
    "minilm_cos_dist": 1 - minilm_cos,
    "clip_text_cos": clip_cos,
    "clip_text_cos_dist": 1 - clip_cos,
}

print("[write] building long-format rows")
new_rows = []
for idx, p in enumerate(all_pairs):
    for mname, arr in metric_arrays.items():
        if already(p["benchmark"], p["pair_id"], p["density_level"], mname):
            continue
        new_rows.append({
            "benchmark": p["benchmark"],
            "pair_id": p["pair_id"],
            "scene_id": p["scene_id"],
            "variation_id": p["variation_id"],
            "edit_type": p["edit_type"],
            "density_level": p["density_level"],
            "metric_name": mname,
            "value": float(arr[idx]),
        })

print(f"[write] {len(new_rows):,} new rows to append")
new_df = pd.DataFrame(new_rows)
if not existing.empty:
    out_df = pd.concat([existing, new_df], ignore_index=True)
    # final dedupe just in case
    out_df = out_df.drop_duplicates(
        subset=["benchmark", "pair_id", "density_level", "metric_name"], keep="last")
else:
    out_df = new_df
out_df.to_csv(RAW_CSV, index=False)
print(f"[write] {RAW_CSV} -> {len(out_df):,} rows")

# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def summarize(df, group_cols):
    g = df.groupby(group_cols + ["metric_name"])["value"]
    out = g.agg(["mean", "median", "std",
                 lambda x: x.quantile(0.25),
                 lambda x: x.quantile(0.75),
                 "count"]).reset_index()
    out.columns = group_cols + ["metric_name", "mean", "median", "std", "p25", "p75", "n_pairs"]
    return out


print("[summary] per benchmark")
sb = summarize(out_df, ["benchmark", "density_level"])
sb.to_csv(OUT / "summary_per_benchmark.csv", index=False)

cs = out_df[out_df["benchmark"] == "compoSET"].copy()
print("[summary] per edit_type (compoSET)")
summarize(cs, ["edit_type", "density_level"]).to_csv(OUT / "summary_per_edit_type.csv", index=False)
print("[summary] per scene (compoSET)")
summarize(cs, ["scene_id", "density_level"]).to_csv(OUT / "summary_per_scene.csv", index=False)
print("[summary] per density (compoSET)")
summarize(cs, ["density_level"]).to_csv(OUT / "summary_per_density.csv", index=False)

print(f"[done] wallclock {time.time()-t0:.1f}s")
print("PER-BENCHMARK pair counts:")
print(df_pairs.groupby("benchmark")["pair_id"].nunique())
