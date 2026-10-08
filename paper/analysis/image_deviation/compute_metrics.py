"""Image-deviation analysis for CompoSET paper.

Computes per-pair image-deviation metrics for CompoSET, Winoground, BiVLC,
and 9 additional prior compositionality benchmarks (Tier 2/3 same-scene and
different-scene pairs). Outputs to
this folder (paper/analysis/image_deviation/).

Idempotent on raw_pairs.csv: dedupes on (benchmark, pair_id).
"""
from __future__ import annotations
from huggingface_hub import snapshot_download
import os, sys, time, io, json, warnings, random
from pathlib import Path
from typing import Optional, Tuple
warnings.filterwarnings("ignore")

# Make the composet_eval benchmark loaders importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim_fn

OUT = Path(__file__).resolve().parents[0]
HF_DIR = Path(os.environ.get("COMPOSET_DATA_DIR") or snapshot_download("CompoSET/CompoSET", repo_type="dataset"))
RAW_CSV = OUT / "raw_pairs.csv"
DIFF_DIR = OUT / "diff_masks"
OUT.mkdir(parents=True, exist_ok=True)
DIFF_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device={DEVICE}")
SIZE = 512
THRESH = 10.0 / 255.0
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------------------------
# Models (loaded lazily)
# ------------------------------------------------------------------
_models = {}

def get_lpips(net="alex"):
    key = f"lpips_{net}"
    if key not in _models:
        import lpips
        m = lpips.LPIPS(net=net, verbose=False).to(DEVICE).eval()
        _models[key] = m
    return _models[key]

def get_dists():
    if "dists" not in _models:
        from DISTS_pytorch import DISTS
        import DISTS_pytorch as _dp
        m = DISTS(load_weights=False)
        w_path = os.path.join(os.path.dirname(_dp.__file__), "weights.pt")
        w = torch.load(w_path, map_location="cpu")
        m.alpha.data = w["alpha"]
        m.beta.data = w["beta"]
        m = m.to(DEVICE).eval()
        _models["dists"] = m
    return _models["dists"]

def get_msssim():
    if "msssim" not in _models:
        from pytorch_msssim import ms_ssim as _ms
        _models["msssim"] = _ms
    return _models["msssim"]

def get_dinov2():
    if "dinov2" not in _models:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        mdl = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
        _models["dinov2"] = (proc, mdl)
    return _models["dinov2"]

def get_clip():
    if "clip" not in _models:
        from transformers import CLIPProcessor, CLIPModel
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        mdl = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        _models["clip"] = (proc, mdl)
    return _models["clip"]

# ------------------------------------------------------------------
# Image loading
# ------------------------------------------------------------------
def load_image(img: Image.Image | Path | str) -> np.ndarray:
    if isinstance(img, (str, Path)):
        im = Image.open(img).convert("RGB")
    else:
        im = img.convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0  # H,W,3 in [0,1]

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    # arr: H,W,3 in [0,1] -> 1,3,H,W
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return t

# ------------------------------------------------------------------
# Per-pair metric computation
# ------------------------------------------------------------------
@torch.no_grad()
def compute_pair_metrics(a_np: np.ndarray, b_np: np.ndarray) -> dict:
    """a_np, b_np: H,W,3 float32 in [0,1]. Returns dict of metrics."""
    out = {}
    diff = np.abs(a_np - b_np)  # H,W,3
    # pixel-space
    out["pixel_mae"] = float(diff.mean())
    out["pixel_l2"] = float(np.sqrt((diff**2).mean()))
    out["pixel_linf"] = float(diff.max())
    per_pix = diff.max(axis=2)  # H,W max across channels
    out["frac_pixels_above_thresh"] = float((per_pix > THRESH).mean())

    # SSIM / MS-SSIM (grayscale SSIM via skimage on RGB average — use channel_axis)
    a_u8 = (a_np * 255).astype(np.uint8)
    b_u8 = (b_np * 255).astype(np.uint8)
    out["ssim"] = float(ssim_fn(a_u8, b_u8, channel_axis=2, data_range=255))

    a_t = to_tensor(a_np)
    b_t = to_tensor(b_np)

    try:
        ms = get_msssim()(a_t, b_t, data_range=1.0, size_average=True)
        out["ms_ssim"] = float(ms.item())
    except Exception as e:
        out["ms_ssim"] = float("nan")

    # LPIPS expects [-1,1]
    a_lp = a_t * 2 - 1
    b_lp = b_t * 2 - 1
    out["lpips_alex"] = float(get_lpips("alex")(a_lp, b_lp).item())
    out["lpips_vgg"] = float(get_lpips("vgg")(a_lp, b_lp).item())
    # DISTS expects [0,1]
    out["dists"] = float(get_dists()(a_t, b_t).item())

    # DINOv2 cosine
    proc, mdl = get_dinov2()
    imgs = [Image.fromarray((a_np * 255).astype(np.uint8)),
            Image.fromarray((b_np * 255).astype(np.uint8))]
    inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
    feats = mdl(**inp).last_hidden_state[:, 0]  # CLS token
    feats = F.normalize(feats, dim=-1)
    out["dinov2_cos"] = float((feats[0] * feats[1]).sum().item())

    # CLIP image-image cosine
    cproc, cmdl = get_clip()
    cinp = cproc(images=imgs, return_tensors="pt").to(DEVICE)
    cfeats = cmdl.get_image_features(**cinp)
    if hasattr(cfeats, "pooler_output"):
        cfeats = cfeats.pooler_output
    elif hasattr(cfeats, "last_hidden_state"):
        cfeats = cfeats.last_hidden_state[:, 0]
    cfeats = F.normalize(cfeats, dim=-1)
    out["clip_image_cos"] = float((cfeats[0] * cfeats[1]).sum().item())

    # FFT — magnitude diff energy in 3 radial bands
    gray_a = a_np.mean(axis=2)
    gray_b = b_np.mean(axis=2)
    Fa = np.fft.fftshift(np.fft.fft2(gray_a))
    Fb = np.fft.fftshift(np.fft.fft2(gray_b))
    mag_diff = np.abs(np.abs(Fa) - np.abs(Fb))
    h, w = gray_a.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()
    low = r < rmax / 3
    mid = (r >= rmax / 3) & (r < 2 * rmax / 3)
    high = r >= 2 * rmax / 3
    total = mag_diff.sum() + 1e-12
    out["fft_low"] = float(mag_diff[low].sum() / total)
    out["fft_mid"] = float(mag_diff[mid].sum() / total)
    out["fft_high"] = float(mag_diff[high].sum() / total)

    # Diff mask: blur + threshold
    blurred = ndimage.gaussian_filter(per_pix, sigma=1.0)
    mask = (blurred > THRESH).astype(np.uint8)
    out["frac_pixels_affected"] = float(mask.mean())
    if mask.sum() > 0:
        lab, n = ndimage.label(mask)
        out["n_components"] = int(n)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        out["largest_component_area_norm"] = float(sizes.max() / mask.size) if len(sizes) else 0.0
        ys, xs = np.where(mask > 0)
        cx_n = xs.mean() / mask.shape[1]
        cy_n = ys.mean() / mask.shape[0]
        out["edit_centroid_x"] = float(cx_n)
        out["edit_centroid_y"] = float(cy_n)
        out["distance_to_center"] = float(np.sqrt((cx_n - 0.5)**2 + (cy_n - 0.5)**2))
        out["bbox_x0"] = float(xs.min() / mask.shape[1])
        out["bbox_y0"] = float(ys.min() / mask.shape[0])
        out["bbox_x1"] = float(xs.max() / mask.shape[1])
        out["bbox_y1"] = float(ys.max() / mask.shape[0])
    else:
        out["n_components"] = 0
        out["largest_component_area_norm"] = 0.0
        out["edit_centroid_x"] = float("nan")
        out["edit_centroid_y"] = float("nan")
        out["distance_to_center"] = float("nan")
        out["bbox_x0"] = out["bbox_y0"] = out["bbox_x1"] = out["bbox_y1"] = float("nan")

    # Downsampled diff mask (128x128 float) for npz
    mask_small = ndimage.zoom(blurred, 128 / blurred.shape[0], order=1).astype(np.float32)
    return out, mask_small


# ------------------------------------------------------------------
# Pair generators
# ------------------------------------------------------------------
def iter_compoSET():
    df = pd.read_parquet(HF_DIR / "variations.parquet")
    for _, row in df.iterrows():
        pair_id = row["id"]  # e.g. s001_v01
        a = HF_DIR / row["image_base"]
        b = HF_DIR / row["image_var"]
        yield {
            "benchmark": "compoSET",
            "pair_id": pair_id,
            "scene_id": row["scene_id"],
            "variation_id": pair_id.split("_", 1)[1],
            "edit_type": row["edit_type"],
            "image_a_path": str(a),
            "image_b_path": str(b),
            "a_loader": lambda p=a: load_image(p),
            "b_loader": lambda p=b: load_image(p),
        }

def iter_winoground(n=None, seed=42):
    """Full Winoground test split (400 pairs) by default; pass n to subsample.

    Note: when n is None we iterate the full split in dataset order so the
    pair_ids stay stable across runs (idempotency on (benchmark, pair_id)).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/winoground", trust_remote_code=True, split="test")
    except Exception as e:
        print(f"[winoground] FAILED to load: {e}")
        return
    if n is None:
        idxs = list(range(len(ds)))
    else:
        idxs = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(idxs)
        idxs = sorted(idxs[:n])
    print(f"[winoground] yielding {len(idxs)} pairs (full split={n is None})")
    for i in idxs:
        row = ds[i]
        pair_id = f"wg_{row.get('id', i)}"
        a = row["image_0"]
        b = row["image_1"]
        yield {
            "benchmark": "winoground",
            "pair_id": pair_id,
            "scene_id": np.nan,
            "variation_id": np.nan,
            "edit_type": np.nan,
            "image_a_path": f"winoground/{pair_id}_0",
            "image_b_path": f"winoground/{pair_id}_1",
            "a_loader": lambda im=a: load_image(im),
            "b_loader": lambda im=b: load_image(im),
        }

def iter_bivlc(n=None, seed=42):
    """Full BiVLC test split (~2,933 pairs) by default; pass n to subsample."""
    try:
        from datasets import load_dataset
        ds = load_dataset("imirandam/BiVLC", split="test")
    except Exception as e:
        print(f"[bivlc] FAILED to load: {e}")
        return
    if n is None:
        idxs = list(range(len(ds)))
    else:
        idxs = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(idxs)
        idxs = sorted(idxs[:n])
    # Inspect schema
    ex = ds[idxs[0]]
    img_key = None
    neg_key = None
    for cand_a, cand_b in [("image","negative_image"), ("image","neg_image"), ("caption_image","negative_caption_image")]:
        if cand_a in ex and cand_b in ex:
            img_key, neg_key = cand_a, cand_b
            break
    if img_key is None:
        print(f"[bivlc] unknown schema, keys={list(ex.keys())}")
        return
    print(f"[bivlc] using keys {img_key!r} / {neg_key!r}; yielding {len(idxs)} pairs (full split={n is None})")
    for i in idxs:
        row = ds[i]
        pair_id = f"bv_{row.get('id', i)}"
        a = row[img_key]
        b = row[neg_key]
        yield {
            "benchmark": "bivlc",
            "pair_id": pair_id,
            "scene_id": np.nan,
            "variation_id": np.nan,
            "edit_type": np.nan,
            "image_a_path": f"bivlc/{pair_id}_a",
            "image_b_path": f"bivlc/{pair_id}_b",
            "a_loader": lambda im=a: load_image(im),
            "b_loader": lambda im=b: load_image(im),
        }


# ------------------------------------------------------------------
# Generic wrapper for the unified BenchmarkLoader interface
# ------------------------------------------------------------------
def _pair_to_dict(bench_name: str, pair, idx: int) -> Optional[dict]:
    """Convert a BenchmarkPair into the iterator dict shape.

    Returns None if either side is missing an image (image-deviation needs
    both sides). Pair_id falls back to the row index if the loader didn't
    set one.
    """
    pid = pair.pair_id or f"row_{idx}"
    # sanitize for filenames
    pid_safe = pid.replace("/", "_").replace(" ", "_")
    img_a = pair.get_image_a()
    img_b = pair.get_image_b()
    if img_a is None or img_b is None:
        return None
    edit_type = np.nan
    if pair.edit_types:
        edit_type = "|".join(pair.edit_types) if isinstance(pair.edit_types, list) else str(pair.edit_types)
    return {
        "benchmark": bench_name,
        "pair_id": pid_safe,
        "scene_id": pair.scene_id if pair.scene_id is not None else np.nan,
        "variation_id": pair.variation_id if pair.variation_id is not None else np.nan,
        "edit_type": edit_type,
        "image_a_path": pair.image_a_path or f"{bench_name}/{pid_safe}_a",
        "image_b_path": pair.image_b_path or f"{bench_name}/{pid_safe}_b",
        "a_loader": lambda im=img_a: load_image(im),
        "b_loader": lambda im=img_b: load_image(im),
    }


def _timeout_iter(gen, per_yield_timeout_s: float, label: str):
    """Wrap a generator so each `next()` is capped at `per_yield_timeout_s`.

    Runs the generator in a background thread, reads via a queue. If the
    generator stalls (e.g. HF streaming filter scanning thousands of rows
    before the first match), we abandon it after the timeout — caller stops
    iterating. Generator thread is daemonized so it dies with the process.
    """
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
            print(f"[{label}] per-yield timeout ({per_yield_timeout_s}s); abandoning generator")
            return
        if x is SENTINEL:
            return
        if isinstance(x, tuple) and len(x) == 2 and x[0] is EXC:
            print(f"[{label}] generator raised: {repr(x[1])[:200]}")
            return
        yield x


def iter_loader(bench_name: str, loader_cls, max_pairs: Optional[int] = None):
    """Yield dict items by walking a BenchmarkLoader's iter_pairs().

    Logs total count emitted. Catches per-pair failures and continues.
    If the loader itself errors, logs and returns nothing.
    """
    try:
        loader = loader_cls()
    except Exception as e:
        print(f"[{bench_name}] FAILED to instantiate loader: {repr(e)[:200]}")
        return
    n_yielded = 0
    n_dropped = 0
    try:
        for idx, pair in enumerate(loader.iter_pairs(limit=max_pairs, seed=SEED)):
            try:
                item = _pair_to_dict(bench_name, pair, idx)
            except Exception as e:
                print(f"[{bench_name}] drop row {idx}: {repr(e)[:120]}")
                n_dropped += 1
                continue
            if item is None:
                n_dropped += 1
                continue
            n_yielded += 1
            yield item
    except Exception as e:
        print(f"[{bench_name}] loader iteration aborted at n_yielded={n_yielded}: {repr(e)[:200]}")
    print(f"[{bench_name}] yielded {n_yielded} pairs (dropped {n_dropped})")


def _make_iter(bench_name: str, per_yield_timeout_s: float = 300.0):
    """Look up a benchmark loader and return a timeout-wrapped iterator.

    `per_yield_timeout_s` caps how long we wait between successive items.
    Loaders that stall (e.g. ColorBench streaming filter that fetches
    thousands of remote images before the first matching robustness row)
    are abandoned and the run continues with the next benchmark.
    """
    from composet_eval.benchmarks import LOADERS
    cls = LOADERS.get(bench_name)
    if cls is None:
        print(f"[{bench_name}] no loader registered; skipping")
        return iter(())
    inner = iter_loader(bench_name, cls)
    return _timeout_iter(inner, per_yield_timeout_s, bench_name)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
COLUMNS = [
    "benchmark","pair_id","scene_id","variation_id","edit_type",
    "image_a_path","image_b_path",
    "pixel_mae","pixel_l2","pixel_linf","frac_pixels_above_thresh",
    "ssim","ms_ssim","lpips_alex","lpips_vgg","dists",
    "dinov2_cos","clip_image_cos",
    "fft_low","fft_mid","fft_high",
    "frac_pixels_affected","n_components","largest_component_area_norm",
    "edit_centroid_x","edit_centroid_y","distance_to_center",
    "bbox_x0","bbox_y0","bbox_x1","bbox_y1",
]

def load_existing():
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV)
        return df, set(zip(df["benchmark"], df["pair_id"]))
    return pd.DataFrame(columns=COLUMNS), set()


def run(benchmarks):
    existing_df, done = load_existing()
    print(f"[start] existing rows: {len(existing_df)}; done keys: {len(done)}")

    new_rows = []
    # buffer flush size
    FLUSH_EVERY = 50

    # Streaming-friendly safety cap to keep one runaway loader (e.g. NIGHTS
    # at ~20k triplets) from dominating the run. Override with env var.
    MAX_PER_BENCH = int(os.environ.get("IMAGE_DEV_MAX_PER_BENCH", "20000"))
    # Per-benchmark wallclock cap. Some HF loaders (colorbench) iterate a
    # streaming dataset that fetches every image over the network before
    # yielding the first matching pair — we cap so a slow loader doesn't
    # block the whole run. Override with env var.
    PER_BENCH_TIMEOUT_S = int(os.environ.get("IMAGE_DEV_PER_BENCH_TIMEOUT_S", "900"))

    for bench_name, gen in benchmarks:
        t0 = time.time()
        print(f"\n[bench] {bench_name} starting at {time.strftime('%H:%M:%S')}")
        n_seen = 0
        first_yield_t = None
        # Stream rather than materializing — some loaders stream from HF.
        for item in tqdm(gen, desc=bench_name):
            n_seen += 1
            if first_yield_t is None:
                first_yield_t = time.time()
            if MAX_PER_BENCH and n_seen > MAX_PER_BENCH:
                print(f"[{bench_name}] reached MAX_PER_BENCH={MAX_PER_BENCH}, stopping early")
                break
            if PER_BENCH_TIMEOUT_S and (time.time() - t0) > PER_BENCH_TIMEOUT_S:
                print(f"[{bench_name}] PER_BENCH_TIMEOUT_S={PER_BENCH_TIMEOUT_S}s exceeded, stopping early at n_seen={n_seen}")
                break
            key = (item["benchmark"], item["pair_id"])
            if key in done:
                continue
            try:
                a_np = item["a_loader"]()
                b_np = item["b_loader"]()
                metrics, mask_small = compute_pair_metrics(a_np, b_np)
            except Exception as e:
                print(f"[err] {key}: {e}")
                continue
            row = {c: item.get(c, np.nan) for c in ["benchmark","pair_id","scene_id","variation_id","edit_type","image_a_path","image_b_path"]}
            row.update(metrics)
            new_rows.append(row)
            done.add(key)
            # save mask
            np.savez_compressed(DIFF_DIR / f"{item['benchmark']}_{item['pair_id']}.npz", mask=mask_small)
            if len(new_rows) >= FLUSH_EVERY:
                _flush(existing_df, new_rows)
                existing_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows = []
        dt = time.time() - t0
        print(f"[bench] {bench_name} done in {dt/60:.2f} min")

    if new_rows:
        _flush(existing_df, new_rows)

def _flush(existing_df, new_rows):
    merged = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["benchmark","pair_id"], keep="first")
    merged.to_csv(RAW_CSV, index=False)
    print(f"[flush] {len(merged)} rows -> {RAW_CSV}")


# Benchmarks driven through the unified composet_eval loaders. Spec is
# excluded — its loader sets has_image_pair=False (text-only foils).
NEW_BENCHES = [
    "cola",
    "colorbench",
    "colorswap",
    "countercurate",
    "dreamsim",
    "eqben",
    "mmvp",
    "naturalbench",
    "whatsup",
]

DEFAULT_BENCHES = ["compoSET", "winoground", "bivlc"] + NEW_BENCHES


if __name__ == "__main__":
    t_start = time.time()
    benches = []
    args = set(sys.argv[1:])
    selected = args if args else set(DEFAULT_BENCHES)
    if "compoSET" in selected:
        benches.append(("compoSET", iter_compoSET()))
    if "winoground" in selected:
        # Full split (n=None). Existing 200-row entries in raw_pairs.csv are
        # idempotency-deduped by (benchmark, pair_id) so no recomputation.
        benches.append(("winoground", iter_winoground(n=None, seed=42)))
    if "bivlc" in selected:
        benches.append(("bivlc", iter_bivlc(n=None, seed=42)))
    for nb in NEW_BENCHES:
        if nb in selected:
            benches.append((nb, _make_iter(nb)))
    run(benches)
    print(f"\n[total] wallclock {(time.time()-t_start)/60:.2f} min")
