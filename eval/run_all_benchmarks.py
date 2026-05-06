"""Run a model lineup against CompoSET and a suite of prior compositionality
benchmarks (Winoground, BiVLC, SugarCrepe, etc.).

Usage:
    python -m eval.run_all_benchmarks --model clip-b32 siglip-b16
    python -m eval.run_all_benchmarks --benchmarks winoground bivlc composet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark import run_composet
from .loaders.bivlc import run_bivlc
from .loaders.colorbench import run_colorbench
from .loaders.colorswap import run_colorswap
from .loaders.composet import load_composet
from .loaders.mmvp import run_mmvp
from .loaders.naturalbench import run_naturalbench
from .loaders.sugarcrepe import run_sugarcrepe
from .loaders.sugarcrepe_pp import run_sugarcrepe_pp
from .loaders.whatsup import run_whatsup
from .loaders.winoground import run_winoground
from .models import MODELS, VLMScorer


BENCH_DEFAULT = [
    "winoground", "bivlc", "composet",
    "sugarcrepe", "whatsup", "naturalbench", "colorbench",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None,
                        help="Model key(s) to run. Defaults to all.")
    parser.add_argument("--output", default="results/",
                        help="Output directory root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmarks", nargs="+", default=BENCH_DEFAULT,
                        help=f"One of: {BENCH_DEFAULT + ['mmvp', 'colorswap', 'sugarcrepe_pp']}")
    parser.add_argument("--composet-parquet", default=None,
                        help="Local path to variations.parquet. "
                             "If omitted, the dataset is pulled from HF.")
    parser.add_argument("--composet-image-root", default=None,
                        help="Local image root. Required when --composet-parquet "
                             "is set.")
    parser.add_argument("--hf-repo", default=None,
                        help="Override the CompoSET HF repo (default "
                             "CompoSET/CompoSET).")
    args = parser.parse_args()

    model_keys = args.model or list(MODELS.keys())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "composet").mkdir(exist_ok=True)

    composet_df = None
    composet_image_root = None

    for key in model_keys:
        scorer = None

        for bench in args.benchmarks:
            csv_path = out / "composet" / f"{key}.csv" if bench == "composet" \
                else out / f"{bench}_{key}.csv"

            if csv_path.exists():
                print(f"Skipping {bench}/{key} — {csv_path} already exists")
                continue

            if scorer is None:
                scorer = VLMScorer(key, device=args.device)

            if bench == "winoground":
                df = run_winoground(scorer)
            elif bench == "bivlc":
                df = run_bivlc(scorer)
            elif bench == "composet":
                if composet_df is None:
                    if args.composet_parquet:
                        if not args.composet_image_root:
                            parser.error(
                                "--composet-image-root is required when "
                                "--composet-parquet is set.")
                        composet_df = pd.read_parquet(args.composet_parquet)
                        composet_image_root = Path(args.composet_image_root)
                    else:
                        composet_df, composet_image_root = load_composet(
                            repo_id=args.hf_repo or "CompoSET/CompoSET",
                        )
                df = run_composet(scorer, composet_df, composet_image_root)
            elif bench == "sugarcrepe":
                df = run_sugarcrepe(scorer)
            elif bench == "whatsup":
                df = run_whatsup(scorer)
            elif bench == "naturalbench":
                df = run_naturalbench(scorer)
            elif bench == "colorbench":
                df = run_colorbench(scorer)
            elif bench == "mmvp":
                df = run_mmvp(scorer)
            elif bench == "colorswap":
                df = run_colorswap(scorer)
            elif bench == "sugarcrepe_pp":
                df = run_sugarcrepe_pp(scorer)
            else:
                raise ValueError(f"Unknown benchmark: {bench}")

            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} rows -> {csv_path}")

            scores = df[["i2t_score", "t2i_score", "group_score"]].mean() * 100
            print(f"  {bench} {key}: I2T={scores.i2t_score:.1f}%  "
                  f"T2I={scores.t2i_score:.1f}%  Group={scores.group_score:.1f}%")

        if scorer is not None:
            del scorer


if __name__ == "__main__":
    main()
