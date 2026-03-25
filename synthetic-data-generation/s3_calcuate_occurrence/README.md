# S3: Co-occurrence (symptom combinations)

## Purpose

Inject prior clinical / DSM-5-style background knowledge into the prompt and use an LLM (OpenAI Batch) to generate **plausible co-occurring depressive symptom sets** (2–5 symptoms per combination, with a severity label). Parse and normalize the responses into a table for downstream synthetic data or training pipelines.

## Inputs

- `input/symptom_dictionary.json` — symptom name ↔ index mapping (must align with the dictionary used in the generation script).
- `OPENAI_API_KEY` in `synthetic-data-generation/.env`.

## Steps

1. **`batch_api_for_generating_co_occurrence.py`** — Build Batch JSONL, submit the Batch job, save `batch_results.jsonl` when complete.
2. **`batch_api_for_generating_co_occurrence_formatting.py`** — Parse JSONL responses → `parsed_results.json` → CSV with symptom-name lists (`batch_results_{N}.csv`).

## Outputs

- **`output/batch_results_{N}.csv`** — Column `symptoms` (list stored as string).

## Run

```bash
python batch_api_for_generating_co_occurrence.py
python batch_api_for_generating_co_occurrence_formatting.py
```

Adjust `--iteration`, `--model`, `--temperature`, and I/O paths via each script’s CLI arguments as needed.
