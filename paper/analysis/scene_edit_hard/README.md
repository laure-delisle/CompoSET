# Scene-hard / edit-hard and scene-clustered bootstrap

Reads the per-model CompoSET result CSVs in `results/composet/` (or `$COMPOSET_RESULTS_DIR/composet/`).
The margin is the I2T cosine margin `sim_base_cb − sim_base_cv`, pooled across caption tiers.
Per (model, scene): scene-hard means the per-scene mean margin is in the model's bottom quartile;
edit-hard means the per-scene margin standard deviation is in the model's top quartile.

| Script | Paper result |
|---|---|
| `bootstrap_scene_clustered.py` | Scene-clustered bootstrap CIs on Group score (App. C) |
| `anchor_check.py` | Overlap of scene-hard and edit-hard labels; within- vs. between-scene variance |
| `consistency_check.py` | Cross-model consistency of scene difficulty; scene / model / interaction variance split |
| `edithard_confound.py` | Edit-type confound checks for the edit-hard ranking |

```
python paper/analysis/scene_edit_hard/bootstrap_scene_clustered.py   # writes bootstrap_scene_clustered.csv here
python paper/analysis/scene_edit_hard/anchor_check.py
```
