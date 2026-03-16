# Financial Reports OCR Ingestion Pipeline

A three-layer (Bronze / Silver / Gold) idempotent PDF ingestion pipeline for financial reports.

## What it does

1. **Bronze — Ingest** registers PDF files in Supabase with a stable, deterministic ID (UUID5 of the absolute file path). Running it twice on the same file is a no-op.
2. **Silver — OCR** renders each PDF page to an image and runs it through [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) to extract raw text. Results are stored per-document.
3. **Gold — Formatting** takes the OCR text and runs a sequence of LLM prompts (Kimi K2.5 via Moonshot API) to produce structured JSON outputs, one per step (e.g. model name extraction, table extraction).

Pipeline state is tracked in Supabase PostgreSQL so every stage is resumable and idempotent.

---

## Architecture

```
ingestion_pipeline_reports/
├── main.py               CLI entry point (click)
├── config.py             Constants: model IDs, active steps, paths
├── pipeline/
│   ├── tracker.py        Supabase DB helpers
│   ├── ingest.py         Bronze layer — register PDFs
│   ├── ocr.py            Silver layer — GLM-OCR extraction
│   └── formatting.py     Gold layer — Kimi K2.5 structured output
└── prompts/
    ├── extract_model_name.txt
    └── extract_table.txt
```

### Database schema (Supabase PostgreSQL)

| Table | Layer | Description |
|-------|-------|-------------|
| `bronze_mapping` | Bronze | Append-only registry of ingested PDFs (no deletes enforced via trigger) |
| `pipeline` | Silver | Per-document processing state (`last_ocr`, `last_formatting`, `error`) |
| `ocr_results` | Silver | Full OCR text per document |
| `formatting` | Gold | Structured JSON output per document per step |

---

## CLI

```
uv run python main.py PDF_DIR [OPTIONS]
```

| Argument / Option | Description |
|---|---|
| `PDF_DIR` | Directory containing `.pdf` files to ingest |
| `--parse-all` | Force re-run of OCR and formatting on all documents |
| `--parse-date YYYY-MM-DD` | Re-run only documents ingested on or after this date |

### Examples

```bash
# First run — ingest and process all PDFs in ./reports/
uv run python main.py ./reports/

# Re-process everything
uv run python main.py ./reports/ --parse-all

# Re-process documents added since June 2024
uv run python main.py ./reports/ --parse-date 2024-06-01
```

---

## Adding new formatting steps

1. Add a prompt file: `prompts/<step_name>.txt`
   - Use `{ocr_text}` as the placeholder for OCR content.
   - The prompt must instruct the model to return a JSON object.
2. Register the step in `config.py`:
   ```python
   ACTIVE_STEPS = ["extract_model_name", "extract_table", "<step_name>"]
   ```
3. Results appear in the `formatting` table under `step_name = "<step_name>"`.

---

## Running tests

```bash
uv run pytest tests/ -v
```

48 tests cover all layers: tracker, ingest, OCR, formatting, and CLI.
