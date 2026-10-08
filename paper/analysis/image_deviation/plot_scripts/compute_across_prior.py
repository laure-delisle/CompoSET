"""Compute across-pair DINOv2 + CLIP cosine for prior compositionality benchmarks.

For each benchmark, embed the first image (image_a) of every pair, then
compute all pairwise cosines between images from *different* pairs. Output:
across_pair_prior.csv with columns (benchmark, pair_a, pair_b, dinov2_cos,
clip_image_cos). Idempotent on (benchmark, pair_a, pair_b).

Now full-split (no subsample) and supports any benchmark in
composet_eval.benchmarks.LOADERS with has_image_pair=True. To avoid an O(N^2)
explosion on large benchmarks (e.g. BiVLC ~3k -> ~4.5M rows; full Winoground
~400 -> ~80k), we cap the per-benchmark embedding count via
ACROSS_MAX_IMAGES (default 1000, env override).
"""
from __future__ import annotations
from pathlib import Path
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Make composet_eval importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path(__file__).resolve().parents[1]
SIZE = 512
SEED = 42
N = 200  # legacy subsample for the original Winoground/BiVLC pre-existing rows
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACROSS_MAX_IMAGES = int(os.environ.get("ACROSS_MAX_IMAGES", "1000"))
print(f"[init] device={DEVICE} max_images_per_benchmark={ACROSS_MAX_IMAGES}")


def to_rgb_resized(im) -> Image.Image:
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    return im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)


def load_winoground_first_images():
    from datasets import load_dataset
    ds = load_dataset("facebook/winoground", trust_remote_code=True, split="test")
    idxs = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(idxs)
    idxs = sorted(idxs[:N])
    images, ids = [], []
    for i in idxs:
        row = ds[i]
        images.append(to_rgb_resized(row["image_0"]))
        ids.append(f"wg_{row.get('id', i)}")
    return ids, images


def load_bivlc_first_images():
    from datasets import load_dataset
    ds = load_dataset("imirandam/BiVLC", split="test")
    idxs = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(idxs)
    idxs = sorted(idxs[:N])
    ex = ds[idxs[0]]
    img_key = None
    for cand_a, _ in [("image", "negative_image"), ("image", "neg_image"),
                      ("caption_image", "negative_caption_image")]:
        if cand_a in ex:
            img_key = cand_a
            break
    if img_key is None:
        raise RuntimeError(f"unknown BiVLC schema, keys={list(ex.keys())}")
    images, ids = [], []
    for i in idxs:
        row = ds[i]
        images.append(to_rgb_resized(row[img_key]))
        ids.append(f"bv_{row.get('id', i)}")
    return ids, images


def embed_all(images: list[Image.Image]):
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dmdl = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cmdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()

    dino, clip = [], []
    for img in tqdm(images, desc="embedding"):
        with torch.no_grad():
            di = dproc(images=[img], return_tensors="pt").to(DEVICE)
            d = dmdl(**di).last_hidden_state[:, 0]
            d = F.normalize(d, dim=-1).cpu().numpy()[0]
            ci = cproc(images=[img], return_tensors="pt").to(DEVICE)
            c = cmdl.get_image_features(**ci)
            if hasattr(c, "pooler_output"):
                c = c.pooler_output
            elif hasattr(c, "last_hidden_state"):
                c = c.last_hidden_state[:, 0]
            c = F.normalize(c, dim=-1).cpu().numpy()[0]
        dino.append(d)
        clip.append(c)
    return np.stack(dino), np.stack(clip)


def across_pair_rows(ids: list[str], D: np.ndarray, C: np.ndarray, bench: str):
    dm = D @ D.T
    cm = C @ C.T
    rows = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "benchmark": bench,
                "pair_a": ids[i],
                "pair_b": ids[j],
                "dinov2_cos": float(dm[i, j]),
                "clip_image_cos": float(cm[i, j]),
            })
    return rows


def _timeout_iter(gen, per_yield_timeout_s: float, label: str):
    """Per-yield timeout wrapper (mirrors compute_metrics.py helper)."""
    import queue, threading
    q: "queue.Queue" = queue.Queue(maxsize=4)
    SENTINEL = object()
    EXC = object()

    def _run():
        try:
            for item in gen:
                q.put(item)
            q.put(SENTINEL)
        except Exception as e:
            q.put((EXC, e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while True:
        try:
            x = q.get(timeout=per_yield_timeout_s)
        except queue.Empty:
            print(f"[{label}] per-yield timeout ({per_yield_timeout_s}s); abandoning")
            return
        if x is SENTINEL:
            return
        if isinstance(x, tuple) and len(x) == 2 and x[0] is EXC:
            print(f"[{label}] generator raised: {repr(x[1])[:200]}")
            return
        yield x


def load_loader_first_images(bench_name: str, max_images: int,
                             per_yield_timeout_s: float = 180.0):
    """Walk a unified composet_eval loader; yield (id, image) for image_a.

    Capped at `max_images` to keep across-pair O(N^2) rows tractable.
    Deterministic in dataset order (no shuffle). Per-yield timeout protects
    against loaders that stall before producing the first pair (ColorBench,
    EqBen).
    """
    from composet_eval.benchmarks import LOADERS
    cls = LOADERS.get(bench_name)
    if cls is None:
        print(f"[{bench_name}] not in LOADERS; skipping", flush=True)
        return [], []
    try:
        loader = cls()
    except Exception as e:
        print(f"[{bench_name}] loader init failed: {repr(e)[:160]}", flush=True)
        return [], []
    ids, images = [], []
    raw = loader.iter_pairs(limit=max_images, seed=SEED)
    for idx, pair in enumerate(_timeout_iter(raw, per_yield_timeout_s, bench_name)):
        img = pair.get_image_a()
        if img is None:
            continue
        try:
            images.append(to_rgb_resized(img))
        except Exception as e:
            print(f"[{bench_name}] image resize fail row {idx}: {repr(e)[:80]}", flush=True)
            continue
        pid = (pair.pair_id or f"row_{idx}").replace("/", "_").replace(" ", "_")
        ids.append(pid)
        if len(ids) >= max_images:
            break
    return ids, images


# Map of benchmarks to handle. compoSET handled separately by
# compute_across_scene.py (different semantics: across *base* images per scene).
PRIOR_BENCHES = [
    "winoground", "bivlc",
    "cola", "colorbench", "colorswap", "countercurate",
    "dreamsim", "eqben", "mmvp", "naturalbench", "whatsup",
]


def load_existing():
    out_csv = OUT / "across_pair_prior.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        return df, set(zip(df["benchmark"], df["pair_a"], df["pair_b"]))
    return pd.DataFrame(), set()


def main():
    existing_df, done = load_existing()
    print(f"[init] existing across-pair rows: {len(existing_df)}")
    all_new_rows = []
    benches_to_run = sys.argv[1:] if len(sys.argv) > 1 else PRIOR_BENCHES
    for bench in benches_to_run:
        # Skip if we already have entries for this benchmark — idempotency.
        if not existing_df.empty and (existing_df["benchmark"] == bench).any():
            print(f"[{bench}] already has {(existing_df['benchmark']==bench).sum()} "
                  f"rows; skipping (delete csv rows to recompute)")
            continue
        print(f"[{bench}] loading images (cap={ACROSS_MAX_IMAGES})…")
        ids, imgs = load_loader_first_images(bench, ACROSS_MAX_IMAGES)
        if len(imgs) < 2:
            print(f"[{bench}] only {len(imgs)} images — cannot compute across pairs")
            continue
        print(f"[{bench}] {len(ids)} images loaded; embedding…")
        D, C = embed_all(imgs)
        rows = across_pair_rows(ids, D, C, bench)
        print(f"[{bench}] {len(rows)} across-pair rows")
        all_new_rows.extend(rows)
    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        merged = pd.concat([existing_df, new_df], ignore_index=True) \
            if not existing_df.empty else new_df
        merged = merged.drop_duplicates(subset=["benchmark", "pair_a", "pair_b"], keep="first")
        out_csv = OUT / "across_pair_prior.csv"
        merged.to_csv(out_csv, index=False)
        print(f"[done] wrote {out_csv} n_total={len(merged)}")
        print(merged.groupby("benchmark")[["dinov2_cos", "clip_image_cos"]]
              .agg(["mean", "median", "std"]))
    else:
        print("[done] no new rows added")


if __name__ == "__main__":
    main()
