"""Build summary CSVs from raw_pairs.csv.

Outputs:
  summary_per_model_benchmark.csv  one row per (model, benchmark, direction, correct)
  summary_per_scene.csv            CompoSET only — per (model, scene_id, direction)
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_pairs.csv"


def main():
    df = pd.read_csv(RAW)
    df = df.dropna(subset=["margin", "decision_correct"])
    df["decision_correct"] = df["decision_correct"].astype(int)

    # 1) per (model, benchmark, direction, correct)
    grp = df.groupby(["model", "benchmark", "direction", "decision_correct"])["margin"]
    summary = grp.agg(
        n="count", mean="mean", median="median", std="std",
        p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75),
    ).reset_index().rename(columns={"decision_correct": "correct"})
    summary.to_csv(ROOT / "summary_per_model_benchmark.csv", index=False)
    print(f"wrote summary_per_model_benchmark.csv ({len(summary)} rows)")

    # 2) per scene (CompoSET)
    cs = df[df["benchmark"] == "composet"].copy()
    cs = cs.dropna(subset=["scene_id"])
    scene_grp = cs.groupby(["model", "scene_id", "direction"])
    sc = scene_grp["margin"].agg(
        n_foils="count", mean_margin="mean", std_margin="std",
        range=lambda s: float(s.max() - s.min()),
    ).reset_index()
    # scene_hard / edit_hard thresholds: per (model, direction) quantiles
    parts = []
    for (model, direction), g in sc.groupby(["model", "direction"]):
        g = g.copy()
        m_p25 = g["mean_margin"].quantile(0.25)
        s_p75 = g["std_margin"].quantile(0.75)
        g["scene_hard"] = (g["mean_margin"] < m_p25).astype(int)
        g["edit_hard"] = (g["std_margin"] > s_p75).astype(int)
        parts.append(g)
    sc_out = pd.concat(parts, ignore_index=True)
    sc_out.to_csv(ROOT / "summary_per_scene.csv", index=False)
    print(f"wrote summary_per_scene.csv ({len(sc_out)} rows)")


if __name__ == "__main__":
    main()
