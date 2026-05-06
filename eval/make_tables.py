#!/usr/bin/env python3
"""Generate LaTeX results tables from CompoSET evaluation results.

Usage:
    python make_tables.py results/composet/ -o results/composet/tables.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze import load_all_results, SCORE_COLS, TIER_ORDER

SCORE_LABELS = {
    "i2t_score": r"I$\to$T",
    "t2i_score": r"T$\to$I",
    "group_score": "Group",
}

# Friendly model names for display
MODEL_LABELS = {
    "clip-b32": r"CLIP ViT-B/32",
    "clip-l14": r"CLIP ViT-L/14",
    "clip-b16-openai": r"CLIP ViT-B/16 (OpenAI)",
    "clip-b16-laion400m": r"CLIP ViT-B/16 (LAION-400M)",
    "clip-b16-laion2b": r"CLIP ViT-B/16 (LAION-2B)",
    "siglip-b16": r"SigLIP ViT-B/16",
    "siglip2-b16": r"SigLIP2 ViT-B/16",
}

TIER_LABELS = {"short": "Short (NP)", "medium": "Medium", "long": "Long"}

# Preferred column order
MODEL_ORDER = [
    "clip-b32", "clip-b16-openai", "clip-b16-laion400m", "clip-b16-laion2b",
    "clip-l14", "siglip-b16", "siglip2-b16",
]


def _pct(x: float) -> str:
    """Format as percentage with one decimal."""
    return f"{100 * x:.1f}"


def _bold_max(series: pd.Series) -> list[str]:
    """Return formatted strings with the max value bolded."""
    mx = series.max()
    return [
        rf"\textbf{{{_pct(v)}}}" if v == mx else _pct(v)
        for v in series
    ]


def _order_models(models: list[str]) -> list[str]:
    """Sort models into preferred display order."""
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: rank.get(m, 999))


def table_overall(results: dict[str, pd.DataFrame]) -> str:
    """Table 1: Overall accuracy (all models x 3 scores x 2 levels + overall)."""
    models = _order_models(list(results.keys()))
    n_models = len(models)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CompoSET overall accuracy (\%). Best per row in \textbf{bold}.}",
        r"\label{tab:overall}",
        r"\small",
        r"\begin{tabular}{ll" + "c" * n_models + "}",
        r"\toprule",
    ]

    # Header row
    header = r"Score & Level"
    for m in models:
        header += rf" & {MODEL_LABELS.get(m, m)}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for sc in SCORE_COLS:
        sc_label = SCORE_LABELS[sc]
        # Overall
        vals = pd.Series({m: results[m][sc].mean() for m in models})
        fmt = _bold_max(vals)
        row = rf"{sc_label} & Overall"
        for f in fmt:
            row += rf" & {f}"
        lines.append(row + r" \\")

        # Per tier
        present_tiers = [t for t in TIER_ORDER if t in results[models[0]]["tier"].unique()]
        for tier in present_tiers:
            vals = pd.Series({m: results[m][results[m]["tier"] == tier][sc].mean() for m in models})
            fmt = _bold_max(vals)
            row = rf" & {TIER_LABELS.get(tier, tier)}"
            for f in fmt:
                row += rf" & {f}"
            lines.append(row + r" \\")

        if sc != SCORE_COLS[-1]:
            lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_by_edit_type(results: dict[str, pd.DataFrame], score_col: str = "group_score") -> str:
    """Table 2: Group accuracy by edit type (all models)."""
    models = _order_models(list(results.keys()))
    n_models = len(models)
    sc_label = SCORE_LABELS[score_col]

    # Collect per-edit-type accuracies
    parts = {}
    for m in models:
        parts[m] = results[m].groupby("edit_type")[score_col].mean()
    combined = pd.DataFrame(parts)
    combined = combined.sort_index()

    # Count variations per edit type (use first model)
    df0 = results[models[0]]
    n_tiers = df0["tier"].nunique()
    counts = (df0.groupby("edit_type")["tier"].count() / n_tiers).astype(int)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{CompoSET {sc_label} accuracy (\%) by edit type. Best per row in \textbf{{bold}}.}}",
        r"\label{tab:edit_type}",
        r"\small",
        r"\begin{tabular}{lr" + "c" * n_models + "}",
        r"\toprule",
    ]

    header = r"Edit type & $n$"
    for m in models:
        header += rf" & {MODEL_LABELS.get(m, m)}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for et in combined.index:
        vals = combined.loc[et]
        fmt = _bold_max(vals)
        # Clean up edit type name for display
        display_et = et.replace("_", r"\_")
        n = counts.get(et, "")
        row = rf"{display_et} & {n}"
        for f in fmt:
            row += rf" & {f}"
        lines.append(row + r" \\")

    # Overall row
    lines.append(r"\addlinespace")
    vals = pd.Series({m: results[m][score_col].mean() for m in models})
    fmt = _bold_max(vals)
    row = r"\textit{Overall} & "
    total_n = len(results[models[0]]) // n_tiers
    row += str(total_n)
    for f in fmt:
        row += rf" & {f}"
    lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_document(results: dict[str, pd.DataFrame]) -> str:
    """Assemble full LaTeX document."""
    n_scenes = results[next(iter(results))]["scene_id"].nunique()
    n_vars = results[next(iter(results))]["variation_id"].nunique()

    preamble = rf"""\documentclass{{article}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\geometry{{margin=1in, landscape}}

\begin{{document}}

\section*{{CompoSET Benchmark Results}}
{n_scenes} scenes, {n_vars} variations (after QC), {len(results)} models.

"""

    body = table_overall(results) + "\n\n" + table_by_edit_type(results)

    postamble = "\n\n\\end{document}\n"

    return preamble + body + postamble


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from CompoSET results.")
    parser.add_argument("results_dir", type=str, help="Directory with per-model CSVs")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output .tex file (default: <results_dir>/tables.tex)")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print("No CSV files found.")
        return

    doc = make_document(results)

    out_path = Path(args.output) if args.output else Path(args.results_dir) / "tables.tex"
    out_path.write_text(doc)
    print(f"Wrote {out_path} ({len(results)} models)")


if __name__ == "__main__":
    main()
