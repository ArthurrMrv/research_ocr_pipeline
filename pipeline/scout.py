from __future__ import annotations

from datetime import datetime, timezone

import jsonschema
from supabase import Client

from config import ACTIVE_STEPS
from pipeline.formatting import load_step, load_step_config, validate_output
from pipeline.page_utils import is_empty_page_content
from pipeline.providers.registry import get_provider
from pipeline.tracker import (
    append_error,
    delete_scout_page_scores,
    get_ocr_chunks,
    get_scout_page_scores,
    pipeline_get,
    pipeline_update,
    scout_page_scores_bulk_upsert,
)

SCOUT_STEP_NAME = "_scout_page"


def _get_step_definitions() -> dict[str, str]:
    """Load scout relevance definitions from each active step config."""
    definitions: dict[str, str] = {}
    for step_name in ACTIVE_STEPS:
        config = load_step_config(step_name)
        definition = str(config.get("definition") or "").strip()
        if not definition:
            raise ValueError(f"Missing 'definition' in config for step {step_name}")
        definitions[step_name] = definition
    return definitions


def _render_scout_prompt(prompt_text: str, step_definitions_map: dict[str, str]) -> str:
    step_names = ", ".join(f'"{step_name}"' for step_name in ACTIVE_STEPS)
    step_definitions = "\n".join(
        f'- "{step_name}": {step_definitions_map[step_name]}'
        for step_name in ACTIVE_STEPS
    )
    return (
        prompt_text
        .replace("{step_names}", step_names)
        .replace("{step_definitions}", step_definitions)
    )


def _validate_scout_scores(result: dict, schema: dict) -> None:
    """Validate score shape and require exact coverage for active steps."""
    validate_output(result, schema, SCOUT_STEP_NAME)

    expected_steps = set(ACTIVE_STEPS)
    actual_steps = set(result)
    if actual_steps != expected_steps:
        missing = sorted(expected_steps - actual_steps)
        extra = sorted(actual_steps - expected_steps)
        parts = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if extra:
            parts.append(f"unexpected keys: {extra}")
        raise ValueError("; ".join(parts))


def _score_page(provider, prompt_text: str, schema: dict, ocr_text: str) -> dict | None:
    """Call the scout model for one page, retrying once on schema/key mismatch."""
    for attempt in range(2):
        result = provider.call(prompt_text, ocr_text, schema=schema)
        try:
            _validate_scout_scores(result, schema)
            return result
        except (jsonschema.ValidationError, ValueError):
            if attempt == 1:
                return None
    return None


def run_scout(
    doc_id: str,
    client: Client,
    *,
    force: bool = False,
) -> str:
    """
    Run the scout step for doc_id: score each OCR-successful page for each active step.
    Stores per-page scores in scout_page_scores.
    Skips if already scouted unless force=True.
    Returns status: "skipped", "done", or "error".
    """
    pipeline_row = pipeline_get(client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    ocr_chunks = get_ocr_chunks(client, doc_id)
    if not ocr_chunks:
        raise ValueError(f"No OCR results for doc_id={doc_id}; run OCR first")

    scannable_pages = [chunk for chunk in ocr_chunks if not is_empty_page_content(chunk["content"])]
    if pipeline_row.get("last_scout") is not None and not force:
        existing_scores = get_scout_page_scores(client, doc_id)
        expected_count = len(scannable_pages) * len(ACTIVE_STEPS)
        if len(existing_scores) == expected_count:
            return "skipped"

    prompt_text, schema, config = load_step(SCOUT_STEP_NAME)
    if prompt_text is None:
        raise ValueError(f"Missing prompt for step {SCOUT_STEP_NAME}")
    step_definitions = _get_step_definitions()
    rendered_prompt = _render_scout_prompt(prompt_text, step_definitions)
    provider = get_provider(config["provider"], config)

    delete_scout_page_scores(client, doc_id)

    try:
        pending_rows: list[dict] = []
        for chunk in scannable_pages:
            result = _score_page(provider, rendered_prompt, schema, chunk["content"])
            if result is None:
                append_error(client, doc_id, "Scout: output failed schema validation after retry")
                return "error"

            page_number = chunk["page_number"]
            for step_name, score in result.items():
                pending_rows.append({
                    "doc_id": doc_id,
                    "step_name": step_name,
                    "page_number": page_number,
                    "score": score,
                    "scout_model": config["model"],
                })

        scout_page_scores_bulk_upsert(client, pending_rows)
    except Exception as exc:
        append_error(client, doc_id, f"Scout error: {exc}")
        raise

    pipeline_update(
        client,
        doc_id,
        {"last_scout": datetime.now(timezone.utc).isoformat()},
    )
    return "done"
