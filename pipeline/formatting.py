from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone

import jsonschema
from supabase import Client

from config import ACTIVE_STEPS, SCOUT_FALLBACK_TOP_N, SCOUT_SCORE_THRESHOLD, STEPS_DIR
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


def _resolve_prompt(step_dir, provider: str, filename: str = "default.txt") -> str | None:
    """Resolve a prompt file using provider-specific override with fallback.

    Lookup order: prompts/{provider}.txt -> prompts/default.txt -> legacy prompt.txt
    For verify prompts, pass filename="default.txt" and use the verify_prompts/ subdirectory.
    """
    prompts_dir = step_dir / "prompts" if filename == "default.txt" else step_dir
    provider_path = prompts_dir / f"{provider}.txt"
    if provider_path.exists():
        return provider_path.read_text(encoding="utf-8")
    default_path = prompts_dir / "default.txt"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")
    # Legacy fallback: prompt.txt in step root
    legacy = step_dir / "prompt.txt"
    return legacy.read_text(encoding="utf-8") if legacy.exists() else None


def load_step(
    step_name: str, *, institution: str | None = None
) -> tuple[str | None, dict, dict]:
    """
    Load prompt text, JSON schema, and config for a step folder.
    Returns (prompt_text, schema_dict, config_dict).

    Prompt lookup order:
      1. prompts/{provider}.txt  (provider-specific)
      2. prompts/default.txt     (default prompt)
      3. prompt.txt              (legacy fallback)

    If config has "per_company": true and an institution is given,
    looks for company-specific prompts first:
      prompts/{institution}_{provider}.txt -> prompts/{institution}.txt
    then falls back to the standard provider lookup.
    Returns prompt_text=None if per_company is true but no matching prompt is found.
    """
    step_dir = STEPS_DIR / step_name
    schema = json.loads((step_dir / "schema.json").read_text(encoding="utf-8"))
    config = load_step_config(step_name)
    provider = config.get("provider", "")

    is_per_company = config.get("per_company", False)

    if is_per_company and institution:
        prompts_dir = step_dir / "prompts"
        # Try company+provider, then company-only
        prompt_text = _find_company_prompt(prompts_dir, institution, provider=provider)
        if prompt_text is None:
            prompt_text = _find_company_prompt(prompts_dir, institution)
        if prompt_text is None:
            prompt_text = _resolve_prompt(step_dir, provider)
    elif is_per_company and not institution:
        prompt_text = _resolve_prompt(step_dir, provider)
    else:
        prompt_text = _resolve_prompt(step_dir, provider)

    return prompt_text, schema, config


def load_step_config(step_name: str) -> dict:
    """Load config.json for a step."""
    step_dir = STEPS_DIR / step_name
    return json.loads((step_dir / "config.json").read_text(encoding="utf-8"))


def _find_company_prompt(
    prompts_dir, institution: str, *, provider: str | None = None
) -> str | None:
    """Case-insensitive lookup for company prompts.

    If provider is given, looks for {institution}_{provider}.txt first.
    Otherwise looks for {institution}.txt.
    """
    if not prompts_dir.exists():
        return None
    target = f"{institution}_{provider}".lower() if provider else institution.lower()
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
    step_name: str,
    ocr_text: str,
    *,
    institution: str | None = None,
    extra_context: str | None = None,
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
    prompt_text = prompt_text.replace("{methodology_context}", extra_context or "")
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


def _most_common(values: list[str]) -> str:
    """Return the most common non-empty string, or empty string if none."""
    filtered = [v for v in values if v]
    if not filtered:
        return ""
    return Counter(filtered).most_common(1)[0][0]


def _union_lists(lists: list[list[str]]) -> list[str]:
    """Union multiple string lists preserving first-seen order, case-insensitive dedup."""
    seen: set[str] = set()
    result: list[str] = []
    for lst in lists:
        for item in lst:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def _merge_drafts(drafts: list[dict]) -> dict:
    """Merge N draft extractions by unioning list fields and picking most common scalars."""
    return {
        "model_name": _most_common([d.get("model_name", "") for d in drafts]),
        "notes_model": _most_common([d.get("notes_model", "") for d in drafts]),
        "variables": _union_lists([d.get("variables", []) for d in drafts]),
        "variables_important": _union_lists(
            [d.get("variables_important", []) for d in drafts]
        ),
        "assumptions": _union_lists([d.get("assumptions", []) for d in drafts]),
    }


def _merge_assumption_drafts(drafts: list[dict]) -> dict:
    """Merge N assumption draft extractions by deduplicating and averaging."""
    all_assumptions: list[dict] = []
    seen: set[str] = set()
    for d in drafts:
        for a in d.get("assumptions", []):
            key = a.get("assumption", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                all_assumptions.append(a)

    all_techniques: list[dict] = []
    seen_techniques: set[str] = set()
    for d in drafts:
        for t in d.get("techniques_used", []):
            name = t.get("technique_name", "").strip().lower()
            if name and name not in seen_techniques:
                seen_techniques.add(name)
                all_techniques.append(t)

    soph_values = [d.get("sophistication_index", 1) for d in drafts]
    avg_soph = int(round(sum(soph_values) / len(drafts))) if drafts else 1

    return {
        "assumptions": all_assumptions,
        "techniques_used": all_techniques,
        "sophistication_index": avg_soph,
        "sophistication_explanation": _most_common(
            [d.get("sophistication_explanation", "") for d in drafts]
        ),
    }


_MERGE_FUNCTIONS: dict[str, object] = {
    "extract_model_assumptions": _merge_assumption_drafts,
}


def _build_methodology_context(result: dict | None) -> str:
    """Format methodology step result into a context string for dependent steps."""
    if not result:
        return ""
    parts = ["METHODOLOGY CONTEXT (from prior extraction step):"]
    summary = result.get("steps_summary", "")
    if summary:
        parts.append(f"Summary: {summary}")
    sub_models = result.get("sub_models", [])
    if sub_models:
        parts.append(f"Sub-models used: {', '.join(sub_models)}")
    return "\n".join(parts) + "\n"


def _build_assumptions_context(dep_results: dict[str, dict]) -> str:
    """Format methodology + inputs results into context for the assumptions step."""
    parts = ["PRIOR EXTRACTION CONTEXT:"]

    meth = dep_results.get("extract_model_methodology")
    if meth:
        summary = meth.get("steps_summary", "")
        if summary:
            parts.append(f"\nMethodology Summary: {summary}")
        sub_models = meth.get("sub_models", [])
        if sub_models:
            parts.append(f"Sub-models: {', '.join(sub_models)}")
        structural = meth.get("assumptions", [])
        if structural:
            parts.append("\nStructural assumptions (from methodology step):")
            for a in structural:
                parts.append(f"  - {a}")

    inp = dep_results.get("extract_model_inputs")
    if inp:
        variables = inp.get("variables", [])
        if variables:
            parts.append(f"\nVariables/inputs identified: {', '.join(variables)}")
        value_assumptions = inp.get("assumptions", [])
        if value_assumptions:
            parts.append("\nValue-type assumptions (from inputs step):")
            for a in value_assumptions:
                parts.append(f"  - {a}")

    if len(parts) <= 1:
        return ""
    return "\n".join(parts) + "\n"


def _build_context(step_name: str, dep_results: dict[str, dict]) -> str:
    """Build context string for a step from its dependency results."""
    if step_name == "extract_model_assumptions":
        return _build_assumptions_context(dep_results)
    # Default: methodology context (backward compatible for extract_model_inputs)
    first_result = next(iter(dep_results.values()), None)
    return _build_methodology_context(first_result)


def _load_verify_prompt(step_name: str, provider: str = "") -> str:
    """Load verify prompt from verify_prompts/ subfolder.

    Lookup order: verify_prompts/{provider}.txt -> verify_prompts/default.txt
                  -> legacy verify_prompt.txt
    """
    step_dir = STEPS_DIR / step_name
    verify_dir = step_dir / "verify_prompts"
    if verify_dir.exists():
        provider_path = verify_dir / f"{provider}.txt"
        if provider_path.exists():
            return provider_path.read_text(encoding="utf-8")
        default_path = verify_dir / "default.txt"
        if default_path.exists():
            return default_path.read_text(encoding="utf-8")
    # Legacy fallback
    legacy = step_dir / "verify_prompt.txt"
    if legacy.exists():
        return legacy.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No verify prompt found for step '{step_name}'")


def _run_step_multipass(
    step_name: str,
    ocr_text: str,
    *,
    institution: str | None = None,
    extra_context: str | None = None,
) -> tuple[dict | None, str]:
    """
    Multi-pass extraction: N cheap drafts -> merge -> expensive verification.
    Falls back to single run_step() if all drafts fail.
    """
    prompt_text, schema, config = load_step(step_name, institution=institution)
    if prompt_text is None:
        raise MissingPromptError(step_name, institution)
    prompt_text = prompt_text.replace("{methodology_context}", extra_context or "")
    provider_name = config["provider"]
    draft_model = config["draft_model"]
    draft_runs = config.get("draft_runs", 3)
    pro_model = config["model"]

    # Step 1: N draft passes with cheap model
    flash_config = {**config, "model": draft_model}
    flash_provider = get_provider(provider_name, flash_config)
    drafts: list[dict] = []
    for _ in range(draft_runs):
        try:
            debug_logger.print_step_start(f"{step_name} [draft]")
            draft = flash_provider.call(prompt_text, ocr_text)
            drafts.append(draft)
        except Exception:
            continue

    if not drafts:
        return run_step(step_name, ocr_text, institution=institution, extra_context=extra_context)

    # Step 2: Merge drafts
    merge_fn = _MERGE_FUNCTIONS.get(step_name, _merge_drafts)
    merged = merge_fn(drafts)

    # Step 3: Verify with expensive model
    verify_prompt = _load_verify_prompt(step_name, provider=provider_name)
    verify_prompt = verify_prompt.replace("{methodology_context}", extra_context or "")
    pro_provider = get_provider(provider_name, config)
    filled_verify = verify_prompt.replace(
        "{draft_result}", json.dumps(merged, indent=2)
    )

    for attempt in range(2):
        debug_logger.print_step_start(f"{step_name} [verify]")
        result = pro_provider.call(filled_verify, ocr_text)
        try:
            validate_output(result, schema, step_name)
            return result, pro_model
        except jsonschema.ValidationError:
            if attempt == 1:
                return None, pro_model

    return None, pro_model


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
    # Do NOT delete existing results upfront — upsert per-step overwrites only what succeeds.
    # Failed steps keep their previous result rather than being erased to nothing.
    ocr_text = "\n\n".join(chunk["content"] for chunk in ocr_chunks)
    ocr_by_page = {chunk.get("page_number"): chunk["content"] for chunk in ocr_chunks}
    all_pages_given = _sorted_page_numbers(ocr_chunks)

    scout_has_run = pipeline_row.get("last_scout") is not None
    scout_scores = get_scout_page_scores(supa_client, doc_id) if scout_has_run else []
    scout_scores_by_step: dict[str, list[dict]] = {}
    for row in scout_scores:
        scout_scores_by_step.setdefault(row["step_name"], []).append(row)

    all_above_threshold_pages: set[int] = set()
    if scout_has_run:
        for scores_list in scout_scores_by_step.values():
            for row in scores_list:
                if row["score"] >= SCOUT_SCORE_THRESHOLD:
                    all_above_threshold_pages.add(row["page_number"])

    completed_steps = 0
    failed_steps = 0
    failed_details: list[dict] = []
    step_results: dict[str, dict] = {}
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
                step_scores_list = scout_scores_by_step.get(step_name, [])
                sorted_by_score = sorted(step_scores_list, key=lambda r: r["score"], reverse=True)
                top_n_pages = {row["page_number"] for row in sorted_by_score[:SCOUT_FALLBACK_TOP_N]}
                fallback_page_nums = top_n_pages | all_above_threshold_pages
                shortlisted_pages = [
                    (pn, ocr_by_page.get(pn))
                    for pn in sorted(fallback_page_nums)
                    if ocr_by_page.get(pn) is not None
                ]
                if not shortlisted_pages:
                    reason = "no scout scores available"
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
            step_config = load_step_config(step_name)
            # Build context from prior step(s) if configured
            extra_context = None
            depends_on = step_config.get("depends_on")
            if depends_on:
                if isinstance(depends_on, str):
                    depends_on = [depends_on]
                dep_results = {
                    dep: step_results[dep]
                    for dep in depends_on
                    if dep in step_results
                }
                if dep_results:
                    extra_context = _build_context(step_name, dep_results)
            if step_config.get("multi_pass"):
                result, model_name = _run_step_multipass(
                    step_name, step_ocr_text, institution=institution, extra_context=extra_context
                )
            else:
                result, model_name = run_step(
                    step_name, step_ocr_text, institution=institution, extra_context=extra_context
                )
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

        step_results[step_name] = result
        formatting_upsert(
            supa_client,
            {
                "doc_id": doc_id,
                "step_name": step_name,
                "formatting_model": model_name,
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
