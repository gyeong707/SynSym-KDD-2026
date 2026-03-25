# S2: Synthetic generation (single symptom)

## Purpose

S1 expanded descriptions → LLM (OpenAI Batch) generates first-person synthetic posts per symptom/sub-concept → tabular CSV → one-hot labels (`symptom_dictionary.json`) → stratified train/val for MLC.

## Inputs

- `input/batch_results_generating_description_{target}.json` (S1), `input/symptom_dictionary.json`, `OPENAI_API_KEY` in `synthetic-data-generation/.env`.

## Steps

1. **`batch_api_for_generating_synthetic_data_single.py`** — Batch request JSONL in, completion JSONL out.
2. **`batch_api_for_generating_synthetic_data_single_formatting.py`** — Parse JSONL → row CSV (`post`, `symptom`, `description`, …); optional `batch_results_parsed.json`.
3. **`making_training_data_for_mlc_single.py`** — Add `label`, save merged CSV, split → `synth_train_*` / `synth_val_*`.

## Outputs

- **`batch_sdg_df_merged_added_label_{N}.csv`** — Full data + one-hot `label` before split.
- **`synth_train_{N}_{tag}.csv`, `synth_val_{N}_{tag}.csv`** — Training splits (default tag `single`).

## Run

```bash
python batch_api_for_generating_synthetic_data_single.py --target depression
python batch_api_for_generating_synthetic_data_single_formatting.py
python making_training_data_for_mlc_single.py
```
