# CompoSET data-generation pipeline

This directory contains the scripts and prompts used to produce the CompoSET benchmark from scratch (80 scenes, ~1,776 variations, three caption density tiers, base + foil images per variation). Reproduction is end-to-end: starting from `scene_settings.csv`, the pipeline below builds everything that ships in the released dataset.

## Quick reference

```
data_generation/
├── PIPELINE.md            # this file
├── scene_settings.csv     # scene-level specs (id, indoor/outdoor, setting)
├── prompts/               # text prompts used at each stage
└── scripts/               # Python drivers
```

## Inputs

- `scene_settings.csv` — one row per scene, with the setting (e.g. `roadside diner`) and an `indoor_outdoor` tag.
- `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` — set as environment variables.

## Outputs (per scene `s00N`, written under `data/00N/`)

```
data/00N/
├── s00N_scene_params.txt
├── s00N_caption_base.txt
├── s00N_caption_corrected.txt
├── s00N_caption_variations.json    # short / medium / long captions per variation
├── s00N_caption_variations_l0.json # base/var captions used for image generation
├── images/
│   ├── s00N_base.png
│   └── s00N_v01.png … s00N_vXX.png
├── s00N_image_metadata.json
├── s00N_variation_metadata.json
├── s00N_verification_raw.txt
└── s00N_verification_l0_raw.txt
```

After the final step (`finalize.py`), discarded variations are removed and the remaining variations are renumbered contiguously and packaged into the HuggingFace release format.

## The seven steps

The pipeline runs in three driver scripts. Each script bundles tightly-coupled steps that share IO (e.g. caption + base image generation are bundled because the caption is a prerequisite for the image, and re-running the caption normally implies re-running the image).

### 1. Generate the scene's base caption (Claude)

- Script: [`scripts/generate_base.py`](scripts/generate_base.py) — `step_1`
- Prompt: [`prompts/caption_1_generate_base.txt`](prompts/caption_1_generate_base.txt)
- Inputs: scene row from `scene_settings.csv`.
- Output: `data/00N/s00N_caption_base.txt` — a structured caption with a `base_caption` line and several detail sentences (surface, objects, background, action…). The caption is rich in swappable attributes by design, so step 3 has many candidate edits to pick from.

### 2. Generate the base image (Gemini)

- Script: [`scripts/generate_base.py`](scripts/generate_base.py) — `step_1b`
- Prompt: [`prompts/image_1_generate_base.txt`](prompts/image_1_generate_base.txt)
- Inputs: the structured base caption from step 1.
- Output: `data/00N/images/s00N_base.png`. The image is sampled at 1408×768 from `gemini-2.5-flash-image` (a.k.a. *nanobanana*) with up to five retries to enforce canonical size.

### 3. Auto-QC the caption against the image (Gemini)

- Script: [`scripts/generate_base.py`](scripts/generate_base.py) — `step_2`
- Prompt: [`prompts/caption_2_edit_from_image.txt`](prompts/caption_2_edit_from_image.txt)
- Inputs: base image + base caption.
- Output: `data/00N/s00N_caption_corrected.txt` — the caption rewritten to match what the model actually drew. Discrepancies between the prompt and the rendered image are caption-side concessions (the image is the ground truth from this point on).

### 4. Plan and generate variation L0 captions (Claude)

- Script: [`scripts/generate_variations.py`](scripts/generate_variations.py) — `step_3`
- Prompt: [`prompts/caption_3_generate_variations.txt`](prompts/caption_3_generate_variations.txt)
- Inputs: the corrected scene caption from step 3.
- Output: `data/00N/s00N_caption_variations_l0.json` — one variation per row, each with an `edit_type` from the 16-category taxonomy, the (target object, attribute, from→to) edit spec, and a `(positive, negative)` caption pair (short noun-phrase form) that drives image generation. A light verification pass (prompt: [`caption_4_verify_variations.txt`](prompts/caption_4_verify_variations.txt)) catches obvious caption-level errors before image generation.

### 5. Generate the foil images (Gemini, single-edit)

- Script: [`scripts/generate_variations.py`](scripts/generate_variations.py) — `step_5`
- Prompt: [`prompts/image_2_apply_variation.txt`](prompts/image_2_apply_variation.txt)
- Inputs: base image + each variation's `(positive, negative)` L0 captions.
- Output: `data/00N/images/s00N_v01.png` … one per variation. The prompt explicitly instructs the editor to make a single localized change to the base image, holding the rest of the scene constant. This is the load-bearing prompt behind contribution #1 (*surgical single-edit foils*) and is reusable on its own.

### 6. Generate the three density tiers (Claude)

- Script: [`scripts/build_densities.py`](scripts/build_densities.py)
- Prompt: [`prompts/caption_7_build_densities.txt`](prompts/caption_7_build_densities.txt)
- Inputs: the corrected scene caption + the L0 variation file.
- Output: writes the `short`, `medium`, and `long` caption tiers into `data/00N/s00N_caption_variations.json`. The three tiers are aligned to established benchmark caption styles:

  | tier | example | reference benchmarks |
  |---|---|---|
  | `short`  | `"a red metal tin"`                                                         | ARO / Winoground / ColorSwap |
  | `medium` | `"A red metal tin on the right side of the cart."`                          | COCO / SugarCrepe / BiVLC |
  | `long`   | full scene sentence ending with the edit, capital + period                  | Flickr30k / SugarCrepe++ |

### 7. Manual QC (web app)

- Tool: [`qc_app/`](../qc_app/) — Flask UI for reviewing each `(base, variation)` image pair against its captions. Reviewers can submit (accept) or discard variations. Discarded variations carry forward as failures in finalization.

### 8. Finalize: drop discards, renumber, emit the HF release

- Script: [`scripts/finalize.py`](scripts/finalize.py)
- Inputs: the live (post-QC) per-scene data.
- Output: `hf_release/`
  - `variations.parquet` — one row per live variation, with the `(short, medium, long)` × `(base, var)` caption struct.
  - `scenes.parquet` — per-scene metadata.
  - `scene_captions.parquet` — full corrected caption per scene.
  - `images/` — flattened, contiguous filenames (`s00N_base.png`, `s00N_v01.png`…).
- A side-effect `paper_stats/id_remap.json` records the old→new variation-ID map for provenance.

After finalization, the `hf_release/` folder is the unit that gets uploaded to HuggingFace.

## Models used

| step | model |
|---|---|
| 1, 4, 6 | `claude-opus-4-6` (Anthropic) |
| 2, 5    | `gemini-2.5-flash-image` (Google) |
| 3       | `gemini-2.5-flash` (Google; vision-language QC) |

Model IDs are pinned in [`scripts/common.py`](scripts/common.py).
