# S4: Synthetic generation (multi-symptom / co-occurrence)

## Purpose

S3 co-occurrence symptom combinations + S1 expanded descriptions → LLM (OpenAI Batch) generates first-person posts that reflect **multiple symptoms together** (clinical vs colloquial styles) → flat CSV → multi-label vectors from `symptom_dictionary.json` → stratified train/val for MLC.

## Inputs

- **`input/batch_results_generating_description_{target}.json`** (S1) — per-symptom text used to build prompts.
- **`input/batch_results_*.csv`** (S3) — column `symptoms` (list of symptom names per row); must match the combination file used when building the batch.
- **`input/symptom_dictionary.json`** — symptom name ↔ class index for label vectors.
- **`OPENAI_API_KEY`** in `synthetic-data-generation/.env`.

## Steps

1. **`batch_api_for_generating_synthetic_data_multi.py`** — For each co-occurrence row, inject descriptions into the template; write Batch JSONL (`batch_inputs_sdg_multi_depression.jsonl` by default), run Batch, save raw `batch_results_sdg_multi_depression.jsonl`.
2. **`batch_api_for_generating_synthetic_data_multi_formatting.py`** — Parse JSONL → rows with `post`, `type`, `symptom` (same combination CSV as generation); `parsed_results.json` in `output/`.
3. **`making_training_data_for_mlc_multi.py`** — Read formatted synthetic CSV, add multi-hot `label`, save merged table, split → `synth_train_*_multi.csv` / `synth_val_*_multi.csv`.

## Outputs

- **`output/batch_sdg_df_merged_added_label_{N}.csv`** — Full data + `label` before split.
- **`output/synth_train_{N}_multi.csv`, `output/synth_val_{N}_multi.csv`** — Train/validation splits.

## Run

```bash
python batch_api_for_generating_synthetic_data_multi.py --target depression
python batch_api_for_generating_synthetic_data_multi_formatting.py
python making_training_data_for_mlc_multi.py
```

Align `--comb_data` (generation), the formatting script’s combination path (`--load_dir` points at the same co-occurrence CSV), and `--synthetic_data` in the MLC step with the actual formatted CSV name (including `{N}`).
