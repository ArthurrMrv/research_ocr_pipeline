from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import jsonschema
from supabase import Client

from config import ACTIVE_STEPS, SCOUT_SCORE_THRESHOLD, STEPS_DIR
from pipeline import debug_logger

_FORMATTING_ERROR_RE = re.compile(
    r"^Formatting error \[(?P<step>[^\]]+)\](?: \(attempt \d+\))?: (?P<reason>.+)$",
    re.DOTALL,
)
_FORMATTING_STEP_RE = re.compile(
    r"^Formatting step \[(?P<step>[^\]]+)\]: (?P<reason>.+)$",
    re.DOTALL,
)
_SCOUT_SHORTLIST_RE = re.compile(
    r"^Scout shortlisted no pages for step \[(?P<step>[^\]]+)\] at threshold .+$",
    re.DOTALL,
)
from pipeline.providers.registry import get_provider
from pipeline.providers.base import NonJSONResponseError
from pipeline.step_errors import MissingPromptError
from pipeline.tracker import (
    append_error,
    delete_formatting_results,
    formatting_upsert,
    get_bronze_row,
    get_formatting_results,
    get_ocr_chunks,
    get_scout_page_scores,
    increment_formatting_attempts,
    pipeline_get,
    pipeline_update,
)


def load_step(
    step_name: str, *, institution: str | None = None
) -> tuple[str | None, dict, dict]:
    """
    Load prompt text, JSON schema, and config for a step folder.
    Returns (prompt_text, schema_dict, config_dict).

    If config has "per_company": true and a institution is given,
    looks for a company-specific prompt in prompts/{institution}.txt
    (case-insensitive). Returns prompt_text=None if per_company is true
    but no matching prompt file is found.
    """
    step_dir = STEPS_DIR / step_name
    schema = json.loads((step_dir / "schema.json").read_text(encoding="utf-8"))
    config = load_step_config(step_name)

    is_per_company = config.get("per_company", False)

    if is_per_company and institution:
        prompts_dir = step_dir / "prompts"
        prompt_text = _find_company_prompt(prompts_dir, institution)
        if prompt_text is None:
            # Try default fallback prompt.txt
            fallback = step_dir / "prompt.txt"
            prompt_text = fallback.read_text(encoding="utf-8") if fallback.exists() else None
    elif is_per_company and not institution:
        # per_company step but no company name — use fallback
        fallback = step_dir / "prompt.txt"
        prompt_text = fallback.read_text(encoding="utf-8") if fallback.exists() else None
    else:
        prompt_text = (step_dir / "prompt.txt").read_text(encoding="utf-8")

    return prompt_text, schema, config


def load_step_config(step_name: str) -> dict:
    """Load config.json for a step."""
    step_dir = STEPS_DIR / step_name
    return json.loads((step_dir / "config.json").read_text(encoding="utf-8"))


def _find_company_prompt(prompts_dir, institution: str) -> str | None:
    """Case-insensitive lookup for prompts/{institution}.txt."""
    if not prompts_dir.exists():
        return None
    target = institution.lower()
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
    step_name: str, ocr_text: str, *, institution: str | None = None
) -> tuple[dict | None, str]:
    """
    Load step config, dispatch to provider, validate output against schema.
    Retries once on validation failure. Returns (None, model) on second failure (soft-fail).
    Returns (result, model) on success.
    Raises MissingPromptError if prompt_text is None.
    """
    prompt_text, schema, config = load_step(step_name, institution=institution)
    if prompt_text is None:
        raise MissingPromptError(step_name, institution)
    provider_name = config["provider"]
    provider = get_provider(provider_name, config)
    model = config["model"]

    for attempt in range(2):
        debug_logger.print_step_start(step_name)
        result = provider.call(prompt_text, ocr_text)
        try:
            validate_output(result, schema, step_name)
            return result, model
        except jsonschema.ValidationError:
            if attempt == 1:
                return None, model
            # retry once

    return None, model  # unreachable, satisfies type checker


def _append_formatting_error(
    client: Client,
    doc_id: str,
    *,
    attempt: int,
    step_name: str,
    reason: str,
    raw_output: str | None = None,
) -> None:
    """Append a structured formatting error entry to pipeline.error."""
    context = {
        "stage": "formatting",
        "attempt": attempt,
        "step": step_name,
        "reason": reason,
    }
    if raw_output is not None:
        context["raw_output"] = raw_output
    append_error(
        client,
        doc_id,
        f"Formatting error [{step_name}] (attempt {attempt}): {reason}",
        context=context,
    )


def _get_formatting_attempt_history(pipeline_row: dict | None) -> list[dict]:
    """Return grouped formatting failures from pipeline.error."""
    if not pipeline_row:
        return []

    grouped: dict[int, list[dict]] = {}
    legacy_failures: list[dict] = []
    for entry in pipeline_row.get("error") or []:
        if entry.get("stage") != "formatting":
            parsed = _parse_legacy_formatting_failure(entry.get("message", ""))
            if parsed is not None:
                legacy_failures.append(parsed)
            continue
        attempt = entry.get("attempt")
        step = entry.get("step")
        reason = entry.get("reason")
        if not isinstance(attempt, int) or not isinstance(step, str) or not isinstance(reason, str):
            continue
        failure = {"step": step, "reason": reason}
        raw_output = entry.get("raw_output")
        if isinstance(raw_output, str):
            failure["raw_output"] = raw_output
        grouped.setdefault(attempt, []).append(failure)

    if not grouped and legacy_failures:
        return _group_legacy_formatting_failures(legacy_failures)

    return [
        {"attempt": attempt, "failures": grouped[attempt]}
        for attempt in sorted(grouped)
    ]


def _parse_legacy_formatting_failure(message: str) -> dict | None:
    """Parse older formatting error strings that lack structured attempt metadata."""
    for pattern in (_FORMATTING_ERROR_RE, _FORMATTING_STEP_RE):
        match = pattern.match(message)
        if match:
            return {
                "step": match.group("step"),
                "reason": match.group("reason"),
            }

    match = _SCOUT_SHORTLIST_RE.match(message)
    if match:
        return {
            "step": match.group("step"),
            "reason": "no scout pages above threshold",
        }

    return None


def _group_legacy_formatting_failures(failures: list[dict]) -> list[dict]:
    """Infer attempt groupings for older formatting errors using step order resets."""
    if not failures:
        return []

    step_order = {step_name: idx for idx, step_name in enumerate(ACTIVE_STEPS)}
    grouped: dict[int, list[dict]] = {}
    attempt = 1
    last_order = -1

    for failure in failures:
        order = step_order.get(failure["step"], len(step_order))
        if grouped and order <= last_order:
            attempt += 1
        grouped.setdefault(attempt, []).append(failure)
        last_order = order

    return [
        {"attempt": attempt_no, "failures": grouped[attempt_no]}
        for attempt_no in sorted(grouped)
    ]


def _has_valid_formatting_results(
    supa_client: Client,
    *,
    institution: str | None,
    existing_results: dict[str, dict] | None = None,
) -> bool:
    """Return True when formatting rows exist for all active steps and match current schemas."""
    results = existing_results or {}
    for step_name in ACTIVE_STEPS:
        row = results.get(step_name)
        if row is None:
            return False
        _, schema, _ = load_step(step_name, institution=institution)
        try:
            validate_output(row.get("content"), schema, step_name)
        except (jsonschema.ValidationError, ValueError):
            return False
    return True


def _sorted_page_numbers(rows: list[dict]) -> list[int]:
    """Return sorted page numbers, skipping rows without a page_number."""
    return sorted(
        row["page_number"]
        for row in rows
        if isinstance(row.get("page_number"), int)
    )


def run_formatting(
    doc_id: str,
    supa_client: Client,
    *,
    force: bool = False,
) -> dict:
    """
    Run all ACTIVE_STEPS for doc_id. Each step loads its own provider from config.json.
    Skips if all steps already completed unless force=True.
    Soft-fails on per-step schema errors.
    Fetches institution from bronze_mapping and passes to steps.
    Multi-chunk OCR: concatenates all chunks sorted by page range.
    Returns dict with keys: status ("skipped"|"done"), completed_steps, failed_steps.
    """
    pipeline_row = pipeline_get(supa_client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    # Fetch institution from bronze_mapping early because validity checks may depend on company prompts/schemas.
    bronze_row = get_bronze_row(supa_client, doc_id)
    institution = bronze_row.get("institution") if bronze_row else None
    existing_formatting = get_formatting_results(supa_client, doc_id)

    has_valid_results = _has_valid_formatting_results(
        supa_client,
        institution=institution,
        existing_results=existing_formatting,
    )
    if has_valid_results and not force:
        return {"status": "skipped", "completed_steps": 0, "failed_steps": 0, "failed_details": []}
    attempts = increment_formatting_attempts(supa_client, doc_id)

    # Fetch OCR chunks and concatenate
    ocr_chunks = get_ocr_chunks(supa_client, doc_id)
    if not ocr_chunks:
        raise ValueError(f"No OCR results for doc_id={doc_id}; run OCR first")
    if existing_formatting:
        delete_formatting_results(supa_client, doc_id)
    ocr_text = "\n\n".join(chunk["content"] for chunk in ocr_chunks)
    ocr_by_page = {chunk.get("page_number"): chunk["content"] for chunk in ocr_chunks}
    all_pages_given = _sorted_page_numbers(ocr_chunks)

    scout_has_run = pipeline_row.get("last_scout") is not None
    scout_scores = get_scout_page_scores(supa_client, doc_id) if scout_has_run else []
    scout_scores_by_step: dict[str, list[dict]] = {}
    for row in scout_scores:
        scout_scores_by_step.setdefault(row["step_name"], []).append(row)

    completed_steps = 0
    failed_steps = 0
    failed_details: list[dict] = []
    for step_name in ACTIVE_STEPS:
        if scout_has_run:
            shortlisted_pages = [
                (row["page_number"], ocr_by_page.get(row["page_number"]))
                for row in scout_scores_by_step.get(step_name, [])
                if row["score"] >= SCOUT_SCORE_THRESHOLD
            ]
            shortlisted_pages = [
                (page_number, content)
                for page_number, content in shortlisted_pages
                if content is not None
            ]
            shortlisted_pages.sort(key=lambda item: item[0])
            if not shortlisted_pages:
                reason = "no scout pages above threshold"
                _append_formatting_error(
                    supa_client,
                    doc_id,
                    attempt=attempts,
                    step_name=step_name,
                    reason=reason,
                )
                failed_details.append({"step": step_name, "reason": reason})
                failed_steps += 1
                continue
            step_ocr_text = "\n\n".join(content for _, content in shortlisted_pages)
            pages_given = [page_number for page_number, _ in shortlisted_pages]
        else:
            step_ocr_text = ocr_text
            pages_given = all_pages_given

        try:
            result, model_name = run_step(step_name, step_ocr_text, institution=institution)
        except MissingPromptError:
            reason = f"no prompt for institution '{institution}'"
            _append_formatting_error(
                supa_client,
                doc_id,
                attempt=attempts,
                step_name=step_name,
                reason=reason,
            )
            failed_details.append({"step": step_name, "reason": reason})
            failed_steps += 1
            continue
        except NonJSONResponseError as exc:
            reason = f"{exc.provider_name} returned non-JSON"
            raw_output = exc.raw_response
            _append_formatting_error(
                supa_client,
                doc_id,
                attempt=attempts,
                step_name=step_name,
                reason=reason,
                raw_output=raw_output,
            )
            failure = {"step": step_name, "reason": reason}
            if debug_logger.is_enabled():
                failure["raw_output"] = raw_output
            failed_details.append(failure)
            failed_steps += 1
            continue
        except Exception as exc:
            reason = f"provider error: {exc}"
            _append_formatting_error(
                supa_client,
                doc_id,
                attempt=attempts,
                step_name=step_name,
                reason=reason,
            )
            failed_details.append({"step": step_name, "reason": reason})
            failed_steps += 1
            continue

        if result is None:
            reason = "schema validation failed after retry"
            _append_formatting_error(
                supa_client,
                doc_id,
                attempt=attempts,
                step_name=step_name,
                reason=reason,
            )
            failed_details.append({"step": step_name, "reason": reason})
            failed_steps += 1
            continue

        formatting_upsert(
            supa_client,
            {
                "doc_id": doc_id,
                "step_name": step_name,
                "formatting_model": config["model"],
                "pages_given": pages_given,
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
    return {
        "status": "done",
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "failed_details": failed_details,
    }
