from __future__ import annotations

import json
from datetime import datetime, timezone

import jsonschema
from supabase import Client

from config import ACTIVE_STEPS, SCOUT_PAGE_PADDING, STEPS_DIR
from pipeline.page_utils import extract_pages, get_total_pages
from pipeline.providers.registry import get_provider
from pipeline.tracker import (
    append_error,
    formatting_upsert,
    get_bronze_row,
    get_ocr_chunks,
    get_scout_results,
    pipeline_get,
    pipeline_update,
)


def load_step(
    step_name: str, *, company_name: str | None = None
) -> tuple[str | None, dict, dict]:
    """
    Load prompt text, JSON schema, and config for a step folder.
    Returns (prompt_text, schema_dict, config_dict).

    If config has "per_company": true and a company_name is given,
    looks for a company-specific prompt in prompts/{company_name}.txt
    (case-insensitive). Returns prompt_text=None if per_company is true
    but no matching prompt file is found.
    """
    step_dir = STEPS_DIR / step_name
    schema = json.loads((step_dir / "schema.json").read_text(encoding="utf-8"))
    config = json.loads((step_dir / "config.json").read_text(encoding="utf-8"))

    is_per_company = config.get("per_company", False)

    if is_per_company and company_name:
        prompts_dir = step_dir / "prompts"
        prompt_text = _find_company_prompt(prompts_dir, company_name)
        if prompt_text is None:
            # Try default fallback prompt.txt
            fallback = step_dir / "prompt.txt"
            prompt_text = fallback.read_text(encoding="utf-8") if fallback.exists() else None
    elif is_per_company and not company_name:
        # per_company step but no company name — use fallback
        fallback = step_dir / "prompt.txt"
        prompt_text = fallback.read_text(encoding="utf-8") if fallback.exists() else None
    else:
        prompt_text = (step_dir / "prompt.txt").read_text(encoding="utf-8")

    return prompt_text, schema, config


def _find_company_prompt(prompts_dir, company_name: str) -> str | None:
    """Case-insensitive lookup for prompts/{company_name}.txt."""
    if not prompts_dir.exists():
        return None
    target = company_name.lower()
    for path in prompts_dir.iterdir():
        if path.stem.lower() == target and path.suffix == ".txt":
            return path.read_text(encoding="utf-8")
    return None


def validate_output(result: dict, schema: dict, step_name: str) -> None:
    """
    Validate result against the step's JSON Schema.
    Raises jsonschema.ValidationError on failure.
    """
    jsonschema.validate(instance=result, schema=schema)


def run_step(
    step_name: str, ocr_text: str, *, company_name: str | None = None
) -> dict | None:
    """
    Load step config, dispatch to provider, validate output against schema.
    Retries once on validation failure. Returns None on second failure (soft-fail).
    Returns None if prompt_text is None (missing company-specific prompt).
    """
    prompt_text, schema, config = load_step(step_name, company_name=company_name)
    if prompt_text is None:
        return None
    provider_name = config["provider"]
    provider = get_provider(provider_name, config)

    for attempt in range(2):
        result = provider.call(prompt_text, ocr_text)
        try:
            validate_output(result, schema, step_name)
            return result
        except jsonschema.ValidationError:
            if attempt == 1:
                return None
            # retry once

    return None  # unreachable, satisfies type checker


def run_formatting(doc_id: str, supa_client: Client) -> None:
    """
    Run all ACTIVE_STEPS for doc_id. Each step loads its own provider from config.json.
    Skips if all steps already completed. Soft-fails on per-step schema errors.
    Fetches company_name from bronze_mapping and passes to steps.
    Multi-chunk OCR: concatenates all chunks sorted by page range.
    """
    pipeline_row = pipeline_get(supa_client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    already_done = (
        pipeline_row.get("last_formatting") is not None
        and pipeline_row.get("formatting_nb", 0) == len(ACTIVE_STEPS)
    )
    if already_done:
        return

    # Fetch OCR chunks and concatenate
    ocr_chunks = get_ocr_chunks(supa_client, doc_id)
    if not ocr_chunks:
        raise ValueError(f"No OCR results for doc_id={doc_id}; run OCR first")
    ocr_text = "\n\n".join(chunk["content"] for chunk in ocr_chunks)

    # Fetch company_name from bronze_mapping
    bronze_row = get_bronze_row(supa_client, doc_id)
    company_name = bronze_row.get("company_name") if bronze_row else None

    # Fetch scout results (empty dict means scout hasn't run — fall back to full text)
    scout_results = get_scout_results(supa_client, doc_id)
    total_pages = get_total_pages(ocr_text)

    completed_steps = 0
    for step_name in ACTIVE_STEPS:
        if scout_results:
            if step_name not in scout_results:
                append_error(
                    supa_client,
                    doc_id,
                    f"Scout has no page range for step [{step_name}]; skipping",
                )
                continue
            scout_range = scout_results[step_name]
            start = max(1, scout_range["start_page"] - SCOUT_PAGE_PADDING)
            end = min(total_pages or scout_range["end_page"], scout_range["end_page"] + SCOUT_PAGE_PADDING)
            step_ocr_text = extract_pages(ocr_text, start, end)
        else:
            step_ocr_text = ocr_text

        try:
            result = run_step(step_name, step_ocr_text, company_name=company_name)
        except Exception as exc:
            append_error(supa_client, doc_id, f"Formatting error [{step_name}]: {exc}")
            continue

        if result is None:
            append_error(
                supa_client,
                doc_id,
                f"Formatting step [{step_name}]: output failed schema validation after retry",
            )
            continue

        _, _, config = load_step(step_name, company_name=company_name)
        formatting_upsert(
            supa_client,
            {
                "doc_id": doc_id,
                "step_name": step_name,
                "formatting_model": config["model"],
                "content": result,
            },
        )
        completed_steps += 1

    pipeline_update(
        supa_client,
        doc_id,
        {
            "last_formatting": datetime.now(timezone.utc).isoformat(),
            "formatting_nb": completed_steps,
        },
    )
