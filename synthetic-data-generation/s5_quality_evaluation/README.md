# S5: Quality evaluation (LLM scoring)

## Purpose

Take merged synthetic data from **S2** (single-symptom) or **S4** (multi-symptom), run an **OpenAI Batch** job where a model acts as a clinician and scores how clearly each post expresses the target symptom(s) on a **0–5** scale (plus short rationales). Parse completions into CSV for analysis or filtering.

## Inputs

- **`input/batch_sdg_df_merged_added_label_{N}.csv`** (or your merged file with the same columns the script expects):
  - **Single path**: expects `post`, `symptom`, `description`, `type`, … — the generator builds a `message` field (`symptom` + `description` when `description` is a string).
  - **Multi path**: expects `post` and `symptom` (symptom block passed into the prompt as `{description}`).
- **`OPENAI_API_KEY`** in `synthetic-data-generation/.env`.

## Steps

### Single-symptom evaluation

1. **`batch_api_for_generating_synthetic_data_evaluation_single.py`** — One Batch line per row; JSON with `score` and `reason` per post–symptom pair. Writes batch input JSONL to `--input_dir` / `--input_data`, saves raw completions to `--output_dir` / `--output_data` (default model `gpt-4o`, `temperature` 0 in code).
2. **`batch_api_for_generating_synthetic_data_evaluation_single_formatting.py`** — Joins JSONL with the same merged CSV → columns `post`, `type`, `symptom`, `description`, `score`, `reason`; writes `parsed_results.json` and `batch_sdg_df_evaluation_single_{N}.csv`.

### Multi-symptom evaluation

1. **`batch_api_for_generating_synthetic_data_evaluation_multi.py`** — One Batch line per row; model returns `post` and `symptom_scores` (per-symptom `score` / `reason`).
2. **`batch_api_for_generating_synthetic_data_evaluation_multi_formatting.py`** — Parses JSONL → `post`, `score` (list), `reason` (list); saves `batch_sdg_df_evaluation_multi_{N}.csv` and `parsed_results.json`.

## Outputs

- **`output/batch_sdg_df_evaluation_single_{N}.csv`** / **`output/batch_sdg_df_evaluation_multi_{N}.csv`** — Tabular scores aligned to inputs.

## Run

**Single**

```bash
python batch_api_for_generating_synthetic_data_evaluation_single.py --target depression
python batch_api_for_generating_synthetic_data_evaluation_single_formatting.py
```

**Multi**

```bash
python batch_api_for_generating_synthetic_data_evaluation_multi.py --target depression
python batch_api_for_generating_synthetic_data_evaluation_multi_formatting.py
```

Use `--load_dir` / `--load_data` for the merged CSV, and keep **`--output_data` from the Batch step** consistent with **`--input_data` in the formatting step** (rename or pass flags so the formatter reads the JSONL you actually produced).
