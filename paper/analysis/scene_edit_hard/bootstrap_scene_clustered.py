#!/usr/bin/env python3
"""Scene-clustered bootstrap CIs for CompoSET scores.

Resamples the 80 scenes with replacement (B=10,000, seed=42) and recomputes
the tier-pooled mean score per replicate. A scene drawn k times contributes
its rows k times; implemented via multinomial draw counts against precomputed
per-scene sums/counts, so each replicate is a weighted mean.

Outputs bootstrap_scene_clustered.csv next to the per-model result CSVs.
"""

from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.environ.get("COMPOSET_RESULTS_DIR", str(Path(__file__).resolve().parents[3] / "results")), "composet")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootstrap_scene_clustered.csv")
B = 10_000
SEED = 42
GROUP_CHANCE = 1.0 / 6.0   # 16.667 %
DIR_CHANCE = 1.0 / 4.0     # 25 % for i2t / t2i

# Model that also gets per-direction bootstraps
DIRECTION_MODEL = "siglip2-so400m-378"


def load_models():
    models = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "bootstrap_scene_clustered":
            continue
        models[name] = pd.read_csv(path)
    return models


def per_scene_stats(df, metric, scene_index):
    """Return (sums, counts) arrays aligned to scene_index order."""
    g = df.groupby("scene_id")[metric].agg(["sum", "count"])
    g = g.reindex(scene_index)
    if g["sum"].isna().any():
        missing = g.index[g["sum"].isna()].tolist()
        raise ValueError(f"missing scenes for metric {metric}: {missing}")
    return g["sum"].to_numpy(), g["count"].to_numpy()


def bootstrap_ci(sums, counts, draws):
    """draws: (B, n_scenes) multinomial counts. Returns (lo, hi) percentile CI."""
    rep = (draws @ sums) / (draws @ counts)
    return np.percentile(rep, 2.5), np.percentile(rep, 97.5)


def main():
    models = load_models()
    assert len(models) == 22, f"expected 22 models, found {len(models)}"

    # Common scene index (same order for every model -> paired bootstrap)
    scene_index = sorted(models[next(iter(models))]["scene_id"].unique())
    n_scenes = len(scene_index)
    assert n_scenes == 80, f"expected 80 scenes, found {n_scenes}"

    # One shared set of multinomial draws (seed 42) reused for every model/metric
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(n_scenes, np.full(n_scenes, 1.0 / n_scenes), size=B)

    rows = []

    def run(name, df, metric, chance):
        point = df[metric].mean()
        sums, counts = per_scene_stats(df, metric, scene_index)
        lo, hi = bootstrap_ci(sums, counts, draws)
        rows.append(dict(
            model=name, metric=metric, point=point, ci_lo=lo, ci_hi=hi,
            excludes_chance_below=bool(hi < chance),   # CI entirely below chance
            excludes_chance_above=bool(lo > chance),   # CI entirely above chance
        ))
        return point, lo, hi

    for name, df in models.items():
        run(name, df, "group_score", GROUP_CHANCE)

    df_dir = models[DIRECTION_MODEL]
    for metric in ("i2t_score", "t2i_score"):
        run(DIRECTION_MODEL, df_dir, metric, DIR_CHANCE)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    # Console report
    print(f"B={B}, seed={SEED}, {n_scenes} scenes, paired draws across models\n")
    print(f"{'model':30s} {'metric':12s} {'point%':>7s} {'lo%':>7s} {'hi%':>7s} "
          f"{'halfw%':>7s} {'hi<chance':>9s} {'lo>chance':>9s}")
    for r in rows:
        chance = DIR_CHANCE if r["metric"] != "group_score" else GROUP_CHANCE
        print(f"{r['model']:30s} {r['metric']:12s} {100*r['point']:7.2f} "
              f"{100*r['ci_lo']:7.2f} {100*r['ci_hi']:7.2f} "
              f"{100*(r['ci_hi']-r['ci_lo'])/2:7.2f} "
              f"{str(r['excludes_chance_below']):>9s} "
              f"{str(r['excludes_chance_above']):>9s}")

    grp = [r for r in rows if r["metric"] == "group_score"]
    n_below = sum(r["excludes_chance_below"] for r in grp)
    n_above = sum(r["excludes_chance_above"] for r in grp)
    worst = max(grp, key=lambda r: r["ci_hi"])
    print(f"\ngroup_score: {n_below}/22 CIs entirely below 16.667%; "
          f"{n_above}/22 entirely above")
    print(f"max upper bound: {worst['model']} hi={100*worst['ci_hi']:.2f}%")

    # Naive Wald CI comparison (row-level iid) for each model
    print("\nWald (row-iid) vs clustered, group_score:")
    for name, df in models.items():
        p = df["group_score"].mean()
        n = len(df)
        se = np.sqrt(p * (1 - p) / n)
        wlo, whi = p - 1.96 * se, p + 1.96 * se
        r = next(r for r in rows if r["model"] == name and r["metric"] == "group_score")
        wald_excl = whi < GROUP_CHANCE
        clus_excl = r["excludes_chance_below"]
        flag = "  <-- Wald excluded chance, clustered does NOT" \
            if wald_excl and not clus_excl else ""
        print(f"{name:30s} Wald [{100*wlo:5.2f},{100*whi:5.2f}] "
              f"clustered [{100*r['ci_lo']:5.2f},{100*r['ci_hi']:5.2f}]{flag}")


if __name__ == "__main__":
    main()
