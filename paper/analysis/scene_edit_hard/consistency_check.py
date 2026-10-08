#!/usr/bin/env python3
"""R1-W2 quick check: are scene-hard / edit-hard assignments consistent across models?

Per (model, scene): mean and std of I2T margin (sim_base_cb - sim_base_cv) across foils.
Paper def (G.3): scene-hard = mean margin < 25th pctile (per model); edit-hard = std > 75th pctile.
Cross-model consistency:
  (a) avg pairwise Spearman rho of per-scene mean margins (continuous, threshold-free)
  (b) avg pairwise Jaccard of scene-hard sets (bottom-quartile)
  (c) same for edit-hard (std top-quartile)
  (d) within-family vs across-family agreement
"""
from pathlib import Path
import glob, os, itertools
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RES = os.path.join(os.environ.get("COMPOSET_RESULTS_DIR", str(Path(__file__).resolve().parents[3] / "results")), "composet")
files = sorted(glob.glob(f"{RES}/*.csv"))
files = [f for f in files if not os.path.basename(f).startswith("bootstrap")]  # skip analysis outputs
print(f"{len(files)} model files")

FAMILY = {}
for f in files:
    m = os.path.basename(f)[:-4]
    if m.startswith("clip"): fam = "clip"
    elif m.startswith("siglip2"): fam = "siglip2"
    elif m.startswith("siglip"): fam = "siglip"
    elif m.startswith(("dfn", "eva02", "metaclip")): fam = "curation"
    elif m.startswith(("negclip", "clove")): fam = "hardneg"
    elif m.startswith("dreamlip"): fam = "longcap"
    else: fam = "other"
    FAMILY[m] = fam

per_model = {}
for f in files:
    m = os.path.basename(f)[:-4]
    df = pd.read_csv(f)
    df["margin"] = df["sim_base_cb"] - df["sim_base_cv"]
    g = df.groupby("scene_id")["margin"].agg(["mean", "std"])
    per_model[m] = g

models = sorted(per_model)
scenes = per_model[models[0]].index
mean_mat = pd.DataFrame({m: per_model[m]["mean"] for m in models}).loc[scenes]
std_mat = pd.DataFrame({m: per_model[m]["std"] for m in models}).loc[scenes]
print(f"{len(scenes)} scenes x {len(models)} models")

# (a) pairwise Spearman on mean margins
rhos, pairs_fam = [], []
for a, b in itertools.combinations(models, 2):
    rho = spearmanr(mean_mat[a], mean_mat[b]).statistic
    rhos.append(rho)
    pairs_fam.append((FAMILY[a] == FAMILY[b], rho))
rhos = np.array(rhos)
print(f"\n(a) Scene mean-margin Spearman: mean {rhos.mean():.3f}  median {np.median(rhos):.3f}  min {rhos.min():.3f}  max {rhos.max():.3f}")
same = [r for s, r in pairs_fam if s]; diff = [r for s, r in pairs_fam if not s]
print(f"    same-family mean rho {np.mean(same):.3f} (n={len(same)}) | cross-family {np.mean(diff):.3f} (n={len(diff)})")

# (b)(c) quartile-set Jaccard
def jac(sets):
    js = [len(sa & sb) / len(sa | sb) for sa, sb in itertools.combinations(sets, 2)]
    return np.mean(js), np.median(js)

sh_sets = [set(mean_mat[m][mean_mat[m] <= mean_mat[m].quantile(.25)].index) for m in models]
eh_sets = [set(std_mat[m][std_mat[m] >= std_mat[m].quantile(.75)].index) for m in models]
jm, jmed = jac(sh_sets)
print(f"\n(b) Scene-hard (bottom-quartile mean) Jaccard: mean {jm:.3f} median {jmed:.3f}  [chance ~{0.25/(2-0.25)*100:.0f}% = 0.143]")
jm2, jmed2 = jac(eh_sets)
print(f"(c) Edit-hard (top-quartile std) Jaccard:     mean {jm2:.3f} median {jmed2:.3f}  [chance 0.143]")

# consensus: how many scenes are scene-hard for >= half the models
sh_count = pd.Series(0, index=scenes)
for s in sh_sets:
    sh_count[list(s)] += 1
print(f"\n(d) Scenes scene-hard for >=11/22 models: {(sh_count >= 11).sum()} / {len(scenes)}")
print(f"    Scenes scene-hard for >=17/22 models: {(sh_count >= 17).sum()}")
print(f"    Scenes scene-hard for ==0 models:     {(sh_count == 0).sum()}")

# (e) std-margin spearman too (edit-hard continuous)
rhos_std = np.array([spearmanr(std_mat[a], std_mat[b]).statistic for a, b in itertools.combinations(models, 2)])
print(f"\n(e) Scene margin-STD Spearman: mean {rhos_std.mean():.3f}  median {np.median(rhos_std):.3f}")

# variance decomposition of mean margins: scene vs model
long = mean_mat.stack().rename("v").reset_index()
long.columns = ["scene", "model", "v"]
gm = long.v.mean()
ss_scene = ((long.groupby("scene").v.mean() - gm) ** 2 * len(models)).sum()
ss_model = ((long.groupby("model").v.mean() - gm) ** 2 * len(scenes)).sum()
ss_tot = ((long.v - gm) ** 2).sum()
print(f"\n(f) Variance of per-scene mean margin: scene {ss_scene/ss_tot:.1%}  model {ss_model/ss_tot:.1%}  residual {1-(ss_scene+ss_model)/ss_tot:.1%}")
