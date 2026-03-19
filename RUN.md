# Setup & Run Guide

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager
- A [Supabase](https://supabase.com) project (already provisioned at `hhkpfsympcpdmewbhhif`)
- A [Moonshot](https://platform.moonshot.cn) account for the Kimi K2.5 API
- ~5 GB disk space for the GLM-OCR model weights (downloaded on first run from Hugging Face)

---

## 1. Clone and install

```bash
git clone <repo-url>
cd ingestion_pipeline_reports

# Install all dependencies (creates .venv automatically)
uv sync
```

---

## 2. Configure environment variables

Create a `.env` file at the project root (it is gitignored):

```bash
cp .env.example .env   # if example exists, otherwise create manually
```

`.env` contents:

```dotenv
SUPABASE_URL=https://hhkpfsympcpdmewbhhif.supabase.co
SUPABASE_SERVICE_KEY=<your-supabase-service-role-key>
MOONSHOT_API_KEY=<your-moonshot-api-key>
GOOGLE_API_KEY=<your-google-api-key>
SCOUT_SCORE_THRESHOLD=0.6
```

### Where to find each key

| Variable | Where to get it |
|---|---|
| `SUPABASE_URL` | Supabase dashboard → Project Settings → API → Project URL |
| `SUPABASE_SERVICE_KEY` | Supabase dashboard → Project Settings → API → `service_role` key (keep this secret) |
| `MOONSHOT_API_KEY` | [platform.moonshot.cn](https://platform.moonshot.cn) → API Keys |
| `GOOGLE_API_KEY` | Google AI Studio / Google AI API credentials for Gemini |
| `SCOUT_SCORE_THRESHOLD` | Optional. Global page-shortlisting threshold for Scout, default `0.6` |

> **Never commit `.env` to version control.**

---

## 3. Database schema

The schema is already applied to the Supabase project. If you need to re-apply it (e.g. for a fresh project), run the migration SQL from the plan or use the Supabase dashboard SQL editor:

```sql
-- Apply schema_changes.sql for the page-score scout table
```

---

## 4. First run

Place your PDF reports in a directory, e.g. `./reports/`:

```bash
uv run python main.py ./reports/
```

On first run:
- GLM-OCR model weights (~3–5 GB) are downloaded from Hugging Face to `~/.cache/huggingface/`. This happens once.
- All PDFs are registered in `bronze_mapping`.
- OCR runs page-by-page for each document.
- Scout scores each OCR-successful page.
- Formatting runs against only the shortlisted pages per step.

---

## 5. Re-running / idempotency

The pipeline is fully idempotent:

```bash
# Safe to run twice — already-processed docs are skipped
uv run python main.py ./reports/

# Force re-run OCR and formatting on everything
uv run python main.py ./reports/ --parse-all

# Re-run only docs ingested on or after a date
uv run python main.py ./reports/ --parse-date 2024-06-01
```

---

## 6. Run tests

```bash
uv run pytest tests/ -v
```

Tests use mocks for Supabase and Kimi — no live API calls needed.

---

## 7. Hardware notes

- **OCR** runs on CPU by default. For large batches, a GPU is strongly recommended. Set `CUDA_VISIBLE_DEVICES` appropriately if multiple GPUs are available.
- **Memory**: GLM-OCR requires ~4 GB RAM at 150 DPI per page. Reduce DPI in `pipeline/ocr.py` (`dpi = 150`) if memory is constrained.
- **Kimi API rate limits**: The formatting step calls the API once per step per document. Add `time.sleep` between calls in `pipeline/formatting.py` if you hit rate limits.
