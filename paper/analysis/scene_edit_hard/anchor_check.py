from pathlib import Path
import glob, os
import pandas as pd
import numpy as np

RES = os.path.join(os.environ.get("COMPOSET_RESULTS_DIR", str(Path(__file__).resolve().parents[3] / "results")), "composet")
files = sorted(glob.glob(f"{RES}/*.csv"))
files = [f for f in files if not os.path.basename(f).startswith("bootstrap")]  # skip analysis outputs

both_frac, sh_n, eh_n, total = 0, 0, 0, 0
rows = []
for f in files:
    m = os.path.basename(f)[:-4]
    df = pd.read_csv(f)
    df["margin"] = df["sim_base_cb"] - df["sim_base_cv"]
    g = df.groupby("scene_id")["margin"].agg(["mean", "std"])
    sh = g["mean"] <= g["mean"].quantile(.25)
    eh = g["std"] >= g["std"].quantile(.75)
    rows.append({"model": m, "sh": sh.mean(), "eh": eh.mean(), "both": (sh & eh).mean()})
    sh_n += sh.sum(); eh_n += eh.sum(); both_frac += (sh & eh).sum(); total += len(g)

r = pd.DataFrame(rows)
print(f"model-scene combos: {total}")
print(f"scene-hard: {sh_n/total:.1%}  edit-hard: {eh_n/total:.1%}  BOTH: {both_frac/total:.2%}")
print(f"(independence would predict {0.25*0.25:.2%} overlap)")

# between vs within scene variance of per-foil success rate (group_score), pooled across models
allg = []
for f in files:
    df = pd.read_csv(f)
    df["model"] = os.path.basename(f)[:-4]
    allg.append(df[["model", "scene_id", "variation_id", "group_score"]])
A = pd.concat(allg)
# per (model, variation): success rate is 0/1 per row; per-foil success rate across models:
foil = A.groupby(["scene_id", "variation_id"]).group_score.mean()
scene_means = foil.groupby("scene_id").mean()
gm = foil.mean()
n_per_scene = foil.groupby("scene_id").size()
ss_between = (n_per_scene * (scene_means - gm) ** 2).sum()
ss_total = ((foil - gm) ** 2).sum()
print(f"\nper-foil success-rate variance: between-scene {ss_between/ss_total:.1%}  within-scene {1 - ss_between/ss_total:.1%}")
