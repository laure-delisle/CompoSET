# CompoSET

> **CompoSET — Compositional Single-Edit Testbed.** A vision-language compositionality benchmark where each base / foil image pair differs by exactly one localized edit. 80 scenes, ~1,776 variations, 16 edit types, three caption verbosity tiers (`short` / `medium` / `long`). Submitted to NeurIPS 2026 Datasets & Benchmarks.

| | |
|---|---|
| 🤗 Dataset | https://huggingface.co/datasets/CompoSET/CompoSET |
| 📄 Paper   | (under review) |
| 📜 License | MIT (this code) · CC-BY-4.0 (the dataset on Hugging Face) |

---

## Repository layout

```
.
├── eval/                  Winoground-style I2T / T2I / Group scoring harness.
│                          Loaders for CompoSET (HF Hub) and 9 prior
│                          compositionality benchmarks.
├── data_generation/       Scripts + prompts to reproduce the dataset from
│                          scratch — see data_generation/PIPELINE.md.
├── qc_app/                Flask UI used during dataset construction to
│                          accept / discard generated variation pairs.
├── taxonomy.md            16-category edit taxonomy (12 modify_* + 4 swap_*).
├── pyproject.toml         Installable package metadata.
└── LICENSE / CITATION.bib
```

## Install

```bash
pip install -e .
# Optional extras
pip install -e ".[generation]"   # for data_generation/ (anthropic, google-genai)
pip install -e ".[qc]"           # for qc_app/ (Flask)
```

## Quickstart — score a model on CompoSET

```bash
python -m eval.run --models clip-b32 siglip-b16 --output results/
```

Defaults: pulls the dataset from `CompoSET/CompoSET` on Hugging Face,
runs the Winoground-style 2×2 protocol for each variation × verbosity tier,
and saves per-model CSVs under `results/`.

To use a local checkout instead:

```bash
python -m eval.run --models clip-b32 \
    --variations /path/to/variations.parquet \
    --image-root /path/to/release_root
```

## Quickstart — run the full prior-benchmark suite

```bash
python -m eval.run_all_benchmarks --model clip-b32 siglip-b16 \
    --benchmarks composet winoground bivlc sugarcrepe whatsup naturalbench colorbench
```

Each prior benchmark is pulled from its canonical Hugging Face repo. The
output CSVs share a common schema (one row per pair × tier where applicable;
columns include `i2t_score`, `t2i_score`, `group_score`).

## Aggregating results

```bash
python -m eval.summary --results results/        # per-model × per-benchmark table
python -m eval.make_tables --results results/    # paper-ready tables
```

## Reproducing the dataset

The full generation pipeline is documented in
[`data_generation/PIPELINE.md`](data_generation/PIPELINE.md). At a high
level: scene caption → base image → auto-QC → variation L0 captions →
foil images → caption verbosity tiers → manual QC → finalize.

```bash
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
python data_generation/scripts/generate_base.py 1
python data_generation/scripts/generate_variations.py 1
python data_generation/scripts/build_verbosities.py --scene 1
# then qc_app/server.py for manual review, finally:
python data_generation/scripts/finalize.py
```

## Custom-weight models (eval)

A few of the registered eval models (`negclip-b32`, `dreamlip-b16-merged30m`,
`clove-b32`) require pretrained weights distributed by their original
authors. Place them under `$COMPOSET_WEIGHTS_DIR` (default
`~/.cache/composet/weights/`) — the in-file comments in
[`eval/models.py`](eval/models.py) point at the source for each one.

## Citation

See [`CITATION.bib`](CITATION.bib).

## License

MIT for the code in this repository. The released dataset on Hugging Face
is CC-BY-4.0.
