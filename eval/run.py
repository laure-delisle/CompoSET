"""CLI entry point: run CompoSET benchmark and save results.

Usage:
    # Pull dataset from HuggingFace (default):
    python -m eval.run --models clip-b32 siglip-b16 --output results/

    # Or use a local checkout:
    python -m eval.run --models clip-b32 \\
        --variations /path/to/variations.parquet \\
        --image-root /path/to/release_root \\
        --output results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import print_summary
from .benchmark import run_composet
from .loaders.composet import load_composet
from .models import MODELS, VLMScorer
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CompoSET benchmark.")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        help=f"Model keys to evaluate. Available: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--variations", type=str, default=None,
        help="Optional local path to variations.parquet. "
             "If omitted, the dataset is pulled from the HF Hub.",
    )
    parser.add_argument(
        "--image-root", type=str, default=None,
        help="Optional local root containing images/. Required when --variations "
             "is set. If both are omitted, image_root is the HF cache dir.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="Override the HF dataset repo (default CompoSET/CompoSET, or "
             "$COMPOSET_HF_REPO).",
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Directory to save per-model CSV results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    args = parser.parse_args()

    if args.variations:
        variations = pd.read_parquet(args.variations)
        if not args.image_root:
            parser.error("--image-root is required when --variations is set.")
        image_root = Path(args.image_root)
    else:
        variations, image_root = load_composet(
            repo_id=args.hf_repo or "CompoSET/CompoSET",
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        print(f"\n>>> Loading {model_key} ...")
        scorer = VLMScorer(model_key, device=args.device)

        df = run_composet(scorer, variations, image_root)

        out_path = output_dir / f"{model_key}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows -> {out_path}")

        print_summary(df, model_key)

        del scorer


if __name__ == "__main__":
    main()
