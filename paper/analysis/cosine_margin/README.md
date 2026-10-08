# Cosine margin

Per-pair cosine margins from the per-model result CSVs:

- I2T margin = sim(image_a, caption_a) − sim(image_a, caption_b)
- T2I margin = sim(image_a, caption_a) − sim(image_b, caption_a)

The scripts compare margins of correct and wrong decisions, per model and benchmark, and per CompoSET scene.

```
python paper/analysis/cosine_margin/build_raw_pairs.py
python paper/analysis/cosine_margin/plot_scripts/make_summaries.py
python paper/analysis/cosine_margin/plot_scripts/fig_margin_correct_vs_wrong.py   # App. G.3 figure
```

Needs model results in `results/` (or `COMPOSET_RESULTS_DIR`).
