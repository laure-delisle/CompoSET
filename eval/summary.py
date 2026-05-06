"""Aggregate per-benchmark per-model CSVs into one summary table.

Scans ``results/`` for all per-(benchmark, model) CSVs and computes, for each
score column (i2t_score, t2i_score, group_score), the mean accuracy plus a
95% Wilson confidence interval. Columns where the benchmark doesn't apply
(NaN-filled in the CSV) are reported as empty.

Usage:
    python -m eval.summary
    python -m eval.summary --results results/ --out summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


SCORE_COLS = ("i2t_score", "t2i_score", "group_score")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n binary trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


_KNOWN_BENCHMARKS = (
    # Multi-word benchmark names must come first so the longest prefix matches.
    "sugarcrepe_pp",
    "winoground", "bivlc", "sugarcrepe", "whatsup",
    "naturalbench", "colorbench", "mmvp", "colorswap",
    # composet appears flat in generative/frontier dirs (composet_<model>.csv);
    # cosine runner uses composet/<model>.csv subdir (handled separately).
    "composet",
)


def parse_filename(path: Path) -> tuple[str, str] | None:
    """Map a CSV path to (benchmark, model_key).

    - ``results/composet/<model>.csv`` -> ("composet", <model>)
    - ``results/<bench>_<model>.csv`` -> (<bench>, <model>)

    Benchmark names containing underscores (e.g. ``sugarcrepe_pp``) are
    matched explicitly so the prefix split doesn't truncate them.
    """
    if path.parent.name == "composet":
        return ("composet", path.stem)
    stem = path.stem
    for bench in _KNOWN_BENCHMARKS:
        prefix = bench + "_"
        if stem.startswith(prefix):
            return (bench, stem[len(prefix):])
    return None


def summarize_one(df: pd.DataFrame) -> dict:
    """Compute mean + Wilson CI per score column for a single CSV.

    Rows with NaN in the score column are excluded from that column's n.
    """
    out = {"n_rows": len(df)}
    for col in SCORE_COLS:
        if col not in df.columns:
            out[f"{col}_n"] = 0
            out[f"{col}"] = float("nan")
            out[f"{col}_ci_lo"] = float("nan")
            out[f"{col}_ci_hi"] = float("nan")
            continue
        s = df[col].dropna()
        n = len(s)
        if n == 0:
            mean = float("nan")
            lo = hi = float("nan")
        else:
            k = int(s.sum())
            mean = k / n
            lo, hi = wilson_ci(k, n)
        out[f"{col}_n"] = n
        out[f"{col}"] = mean
        out[f"{col}_ci_lo"] = lo
        out[f"{col}_ci_hi"] = hi
    return out


def summarize_all(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.rglob("*.csv")):
        if path.name == "summary.csv" or path.parent.name == "_gate":
            continue
        parsed = parse_filename(path)
        if parsed is None:
            continue
        bench, model = parsed
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[skip] {path}: {e!r}")
            continue
        row = {"model": model, "benchmark": bench, **summarize_one(df)}
        rows.append(row)
    return pd.DataFrame(rows)


def pretty_print(summary: pd.DataFrame) -> None:
    """Print a scoreboard: one row per (model, benchmark) with score+CI."""
    if summary.empty:
        print("(no CSVs found)")
        return
    def fmt(m, lo, hi):
        if pd.isna(m):
            return "     -      "
        return f"{100*m:5.1f} [{100*lo:4.1f},{100*hi:5.1f}]"
    rows = []
    for _, r in summary.sort_values(["benchmark", "model"]).iterrows():
        rows.append({
            "benchmark": r["benchmark"],
            "model": r["model"],
            "n": int(r["group_score_n"] or r["i2t_score_n"] or r["t2i_score_n"] or 0),
            "I2T": fmt(r["i2t_score"], r["i2t_score_ci_lo"], r["i2t_score_ci_hi"]),
            "T2I": fmt(r["t2i_score"], r["t2i_score_ci_lo"], r["t2i_score_ci_hi"]),
            "Group": fmt(r["group_score"], r["group_score_ci_lo"], r["group_score_ci_hi"]),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/")
    ap.add_argument("--out", default=None,
                    help="Optional output CSV path (default: <results>/summary.csv)")
    ap.add_argument("--no-print", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results)
    summary = summarize_all(results_dir)
    out_path = Path(args.out) if args.out else results_dir / "summary.csv"
    summary.to_csv(out_path, index=False)
    if not args.no_print:
        pretty_print(summary)
    print(f"\n[{len(summary)} rows] -> {out_path}")


if __name__ == "__main__":
    main()
