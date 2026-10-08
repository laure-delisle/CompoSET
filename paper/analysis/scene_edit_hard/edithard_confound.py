from pathlib import Path
import glob, os, itertools
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

RES = os.path.join(os.environ.get("COMPOSET_RESULTS_DIR", str(Path(__file__).resolve().parents[3] / "results")), "composet")
files = sorted(glob.glob(f"{RES}/*.csv"))
files = [f for f in files if not os.path.basename(f).startswith("bootstrap")]  # skip analysis outputs
A = []
for f in files:
    df = pd.read_csv(f)
    df["model"] = os.path.basename(f)[:-4]
    df["margin"] = df["sim_base_cb"] - df["sim_base_cv"]
    A.append(df[["model","scene_id","variation_id","tier","edit_type","margin"]])
A = pd.concat(A, ignore_index=True)

# 1) How much of margin variance does edit_type explain overall vs scene vs interaction?
gm = A.margin.mean()
ss_tot = ((A.margin-gm)**2).sum()
et = A.groupby("edit_type").margin.transform("mean")
sc = A.groupby("scene_id").margin.transform("mean")
both = A.groupby(["scene_id","edit_type"]).margin.transform("mean")
r2_et = 1-((A.margin-et)**2).sum()/ss_tot
r2_sc = 1-((A.margin-sc)**2).sum()/ss_tot
r2_add = None
# additive model scene + edit_type (one pass of means):
resid = A.margin - et
sc_r = resid.groupby(A.scene_id).transform("mean")
r2_additive = 1-((resid-sc_r)**2).sum()/ss_tot
r2_cell = 1-((A.margin-both)**2).sum()/ss_tot
print(f"R2 of margin: edit_type {r2_et:.3f} | scene {r2_sc:.3f} | scene+edit_type {r2_additive:.3f} | scene x edit_type cell {r2_cell:.3f}")
print(f"  -> interaction beyond additive: {r2_cell - r2_additive:.3f}")

# 2) edit-hard ranking before vs after edit-type residualization (per model, then avg)
rhos, flips = [], []
for m, dfm in A.groupby("model"):
    raw_std = dfm.groupby("scene_id").margin.std()
    dfm = dfm.copy()
    dfm["resid"] = dfm.margin - dfm.groupby("edit_type").margin.transform("mean")
    res_std = dfm.groupby("scene_id").resid.std()
    rhos.append(spearmanr(raw_std, res_std).statistic)
    top_raw = set(raw_std[raw_std >= raw_std.quantile(.75)].index)
    top_res = set(res_std[res_std >= res_std.quantile(.75)].index)
    flips.append(len(top_raw & top_res)/len(top_raw))
print(f"\nedit-hard scene ranking, raw vs edit-type-residualized: mean Spearman {np.mean(rhos):.3f}")
print(f"top-quartile edit-hard overlap after residualization: {np.mean(flips):.1%}")

# 3) does containing swap_role / pose / action predict edit-hard?
scene_has = A.groupby("scene_id").edit_type.agg(lambda s: set(s))
hard_types = {"swap_role","modify_pose","modify_action"}
raw_std_all = A.groupby(["model","scene_id"]).margin.std().groupby("scene_id").mean()
has_hard = scene_has.apply(lambda s: len(s & hard_types) > 0)
print(f"\nscenes containing swap_role/pose/action: {has_hard.sum()}/80")
from scipy.stats import mannwhitneyu
u = mannwhitneyu(raw_std_all[has_hard], raw_std_all[~has_hard])
print(f"margin-std with vs without hard types: {raw_std_all[has_hard].mean():.4f} vs {raw_std_all[~has_hard].mean():.4f} (MWU p={u.pvalue:.3g})")

# 4) per-edit-type mean margin spread (context)
tm = A.groupby("edit_type").margin.mean().sort_values()
print("\nper-edit-type mean margin (lowest/highest 4):")
print(tm.head(4).to_string()); print(tm.tail(4).to_string())
