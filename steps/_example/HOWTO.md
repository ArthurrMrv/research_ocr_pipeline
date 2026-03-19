# How to Add a New Formatting Step

Each formatting step lives in its own folder under `steps/`. Adding a step requires **no code changes** — just copy this folder, fill in the three files, and register the step name.

---

## Step-by-step

### 1. Copy this folder

```bash
cp -r steps/_example steps/your_step_name
```

Use lowercase with underscores: `extract_risk_factors`, `summarize_executive`, etc.

### 2. Edit `config.json`

Choose a provider and model:

```json
{
  "definition": "Describe what page-level evidence indicates this extraction step is relevant",
  "provider": "moonshot",
  "model": "kimi-k2.5",
  "temperature": 0,
  "max_tokens": 2048
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `definition` | yes | Short page-level relevance description used by Scout |
| `provider` | yes | One of: `moonshot`, `openai`, `anthropic` |
| `model` | yes | Model ID accepted by that provider |
| `temperature` | no | Defaults to `0` (deterministic) |
| `max_tokens` | no | Defaults to `2048` |

**Required env vars per provider:**

| Provider | Env var |
|----------|---------|
| `moonshot` | `MOONSHOT_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |

### 3. Edit `prompt.txt`

Write a prompt that instructs the model to return a **JSON object**. Use `{ocr_text}` as the placeholder for the extracted text:

```
You are a financial document analyst. Given the OCR text below, extract ...

Return your answer as a JSON object with the following keys: ...

OCR TEXT:
{ocr_text}
```

**Tips:**
- Be explicit that the response must be JSON only.
- Describe the exact keys and value types expected.
- For Anthropic models, the system prompt already says "respond with JSON only", but repeating it in the user prompt helps.

### 4. Edit `schema.json`

Define the expected output shape using [JSON Schema draft-7](https://json-schema.org/draft-07/json-schema-validation.html):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["your_key"],
  "properties": {
    "your_key": { "type": "string" }
  }
}
```

The pipeline validates the LLM response against this schema. On failure it retries once, then soft-fails (logs error to `pipeline.error` and moves on).

### 5. Register the step

Add your step name to `ACTIVE_STEPS` in `config.py`:

```python
ACTIVE_STEPS = ["extract_model_name", "extract_table", "your_step_name"]
```

That's it. The next pipeline run will execute your step for all documents, and Scout will use `definition` when deciding which pages are relevant.

---

## What happens at runtime

1. `pipeline/formatting.py` reads `steps/your_step_name/config.json`
2. Looks up the provider in the registry (`pipeline/providers/registry.py`)
3. Loads `prompt.txt` and fills in `{ocr_text}`
4. Calls the provider's API
5. Validates the response against `schema.json`
6. Upserts the result to the `formatting` table: `(doc_id, step_name, model, content)`
