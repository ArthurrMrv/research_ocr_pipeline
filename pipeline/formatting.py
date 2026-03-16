from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from openai import OpenAI
from supabase import Client

from config import ACTIVE_STEPS, FORMATTING_BASE_URL, FORMATTING_MODEL, PROMPTS_DIR
from pipeline.tracker import append_error, formatting_upsert, pipeline_get, pipeline_update


def get_kimi_client() -> OpenAI:
    """Create OpenAI-compatible client pointed at Moonshot/Kimi API."""
    api_key = os.environ["MOONSHOT_API_KEY"]
    return OpenAI(api_key=api_key, base_url=FORMATTING_BASE_URL)


def load_prompt(step_name: str) -> str:
    """Read prompt template from prompts/{step_name}.txt."""
    prompt_path = PROMPTS_DIR / f"{step_name}.txt"
    return prompt_path.read_text(encoding="utf-8")


def call_llm(kimi_client: OpenAI, prompt: str, ocr_text: str) -> dict:
    """
    Send prompt + ocr_text to Kimi K2.5 and parse JSON from response.
    Raises ValueError if response cannot be parsed as JSON.
    """
    filled_prompt = prompt.replace("{ocr_text}", ocr_text)
    response = kimi_client.chat.completions.create(
        model=FORMATTING_MODEL,
        messages=[{"role": "user", "content": filled_prompt}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON response: {raw[:200]}") from exc


def run_formatting(
    doc_id: str,
    supa_client: Client,
    kimi_client: OpenAI,
) -> None:
    """
    Run all ACTIVE_STEPS formatting for doc_id.
    Skips if all steps already completed.
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

    # Fetch OCR text
    ocr_row = (
        supa_client.table("ocr_results").select("content").eq("doc_id", doc_id).execute()
    )
    if not ocr_row.data:
        raise ValueError(f"No OCR results for doc_id={doc_id}; run OCR first")
    ocr_text = ocr_row.data[0]["content"]

    completed_steps = 0
    for step_name in ACTIVE_STEPS:
        try:
            prompt = load_prompt(step_name)
            result = call_llm(kimi_client, prompt, ocr_text)
            formatting_upsert(
                supa_client,
                {
                    "doc_id": doc_id,
                    "step_name": step_name,
                    "formatting_model": FORMATTING_MODEL,
                    "content": result,
                },
            )
            completed_steps += 1
        except Exception as exc:
            append_error(supa_client, doc_id, f"Formatting error [{step_name}]: {exc}")
            raise

    pipeline_update(
        supa_client,
        doc_id,
        {
            "last_formatting": datetime.now(timezone.utc).isoformat(),
            "formatting_nb": completed_steps,
        },
    )
