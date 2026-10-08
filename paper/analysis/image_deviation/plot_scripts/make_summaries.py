"""Build summary_per_benchmark.csv, summary_per_edit_type.csv, summary_per_scene.csv
from raw_pairs.csv. Standalone; re-readable from CSVs only.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(OUT / "raw_pairs.csv")

METRIC_COLS = [
    "pixel_mae","pixel_l2","pixel_linf","frac_pixels_above_thresh",
    "ssim","ms_ssim","lpips_alex","lpips_vgg","dists",
    "dinov2_cos","clip_image_cos",
    "fft_low","fft_mid","fft_high",
    "frac_pixels_affected","n_components","largest_component_area_norm",
    "edit_centroid_x","edit_centroid_y","distance_to_center",
]

def _agg(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_pairs"] = len(g)
        for c in METRIC_COLS:
            if c in g:
                v = g[c].dropna()
                row[f"{c}_mean"] = float(v.mean()) if len(v) else np.nan
                row[f"{c}_median"] = float(v.median()) if len(v) else np.nan
                row[f"{c}_std"] = float(v.std()) if len(v) else np.nan
                row[f"{c}_p25"] = float(v.quantile(0.25)) if len(v) else np.nan
                row[f"{c}_p75"] = float(v.quantile(0.75)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# per benchmark
pb = _agg(raw, ["benchmark"])
pb["seed"] = 42
pb.to_csv(OUT / "summary_per_benchmark.csv", index=False)
print(f"summary_per_benchmark: {len(pb)} rows")

# per edit type (CompoSET only)
cs = raw[raw["benchmark"] == "compoSET"]
if len(cs):
    pe = _agg(cs, ["edit_type"])
    pe.to_csv(OUT / "summary_per_edit_type.csv", index=False)
    print(f"summary_per_edit_type: {len(pe)} rows")
    ps = _agg(cs, ["scene_id"])
    ps.to_csv(OUT / "summary_per_scene.csv", index=False)
    print(f"summary_per_scene: {len(ps)} rows")
