# Image deviation

How much of the image changes between paired images, for CompoSET and prior image-pair benchmarks.

Each pair is resized to 512×512. The scripts compute pixel metrics (MAE, L2, L∞, fraction of pixels changed),
SSIM and MS-SSIM, perceptual metrics (LPIPS, DISTS), and embedding cosines (DINOv2, CLIP ViT-B/32).
The headline metric is the fraction of pixels changed: per-pixel channel maximum after a σ=1 Gaussian blur, threshold 10/255.
Prior benchmarks are subsampled to 200 pairs with seed 42. SugarCrepe is excluded because it has no image pairs.

```
python paper/analysis/image_deviation/compute_metrics.py                  # all benchmarks; or name some, e.g. compoSET winoground
python paper/analysis/image_deviation/plot_scripts/make_summaries.py      # per-benchmark / edit-type / scene summaries
python paper/analysis/image_deviation/plot_scripts/make_figures.py        # figures, incl. the per-edit-type breakdown
python paper/analysis/image_deviation/plot_scripts/compute_across_scene.py   # CompoSET across-scene CLIP cosine
python paper/analysis/image_deviation/plot_scripts/compute_across_prior.py   # prior-benchmark across-pair CLIP cosine
python paper/analysis/image_deviation/plot_scripts/threshold_sensitivity.py  # diff-mask threshold ablation
```

`compute_metrics.py` is idempotent: it appends only pairs missing from `raw_pairs.csv`.
