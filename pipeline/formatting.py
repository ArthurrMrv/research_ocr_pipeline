from __future__ import annotations

import json
from datetime import datetime, timezone

import jsonschema
from supabase import Client

from config import ACTIVE_STEPS, STEPS_DIR
from pipeline.providers.registry import get_provider
from pipeline.tracker import append_error, formatting_upsert, pipeline_get, pipeline_update


def load_step(step_name: str) -> tuple[str, dict, dict]:
    """
    Load prompt text, JSON schema, and config for a step folder.
    Returns (prompt_text, schema_dict, config_dict).
    """
    step_dir = STEPS_DIR / step_name
    prompt_text = (step_dir / "prompt.txt").read_text(encoding="utf-8")
    schema = json.loads((step_dir / "schema.json").read_text(encoding="utf-8"))
    config = json.loads((step_dir / "config.json").read_text(encoding="utf-8"))
    return prompt_text, schema, config


def validate_output(result: dict, schema: dict, step_name: str) -> None:
    """
    Validate result against the step's JSON Schema.
    Raises jsonschema.ValidationError on failure.
    """
    jsonschema.validate(instance=result, schema=schema)


def run_step(step_name: str, ocr_text: str) -> dict | None:
    """
    Load step config, dispatch to provider, validate output against schema.
    Retries once on validation failure. Returns None on second failure (soft-fail).
    """
    prompt_text, schema, config = load_step(step_name)
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

    ocr_row = (
        supa_client.table("ocr_results").select("content").eq("doc_id", doc_id).execute()
    )
    if not ocr_row.data:
        raise ValueError(f"No OCR results for doc_id={doc_id}; run OCR first")
    ocr_text = ocr_row.data[0]["content"]

    completed_steps = 0
    for step_name in ACTIVE_STEPS:
        try:
            result = run_step(step_name, ocr_text)
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

        _, _, config = load_step(step_name)
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
