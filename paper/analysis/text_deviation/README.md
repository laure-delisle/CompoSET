# Text deviation

How much the caption changes between paired captions, for CompoSET (short, medium, long) and prior benchmarks
(Winoground, BiVLC, What's Up, SugarCrepe, MMVP, ColorSwap).

Metrics per pair: token-level Levenshtein distance, Jaccard, and fraction of tokens changed on CLIP ViT-B/32 BPE tokens;
sentence-embedding cosines (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`, CLIP text encoder).

```
python paper/analysis/text_deviation/compute_metrics.py
python paper/analysis/text_deviation/plot_scripts/fig_per_edit_type.py    # per-edit-type breakdown
python paper/analysis/text_deviation/plot_scripts/fig_analysis_trio.py    # Fig. 2 (needs image_deviation outputs too)
```

Pair ids match `image_deviation/`, so the two analyses can be joined.
