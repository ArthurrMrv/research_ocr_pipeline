from __future__ import annotations

from datetime import datetime, timezone

from supabase import Client

from config import ACTIVE_STEPS
from pipeline.formatting import load_step, run_step
from pipeline.tracker import (
    append_error,
    get_ocr_chunks,
    pipeline_get,
    pipeline_update,
    scout_upsert,
)


def run_scout(
    doc_id: str,
    client: Client,
    *,
    force: bool = False,
) -> str:
    """
    Run the scout step for doc_id: identify relevant page ranges per active step.
    Stores raw page ranges in scout_results table.
    Skips if already scouted unless force=True.
    Returns status: "skipped", "done", or "error".
    """
    pipeline_row = pipeline_get(client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    already_done = pipeline_row.get("last_scout") is not None
    if already_done and not force:
        return "skipped"

    ocr_chunks = get_ocr_chunks(client, doc_id)
    if not ocr_chunks:
        raise ValueError(f"No OCR results for doc_id={doc_id}; run OCR first")
    ocr_text = "\n\n".join(chunk["content"] for chunk in ocr_chunks)

    try:
        result = run_step("_scout", ocr_text)
    except Exception as exc:
        append_error(client, doc_id, f"Scout error: {exc}")
        raise

    if result is None:
        append_error(client, doc_id, "Scout: output failed schema validation after retry")
        return "error"

    _, _, config = load_step("_scout")
    scout_model = config["model"]

    for step_name, page_range in result.items():
        if step_name not in ACTIVE_STEPS:
            continue
        scout_upsert(
            client,
            {
                "doc_id": doc_id,
                "step_name": step_name,
                "start_page": page_range["start_page"],
                "end_page": page_range["end_page"],
                "scout_model": scout_model,
            },
        )

    pipeline_update(
        client,
        doc_id,
        {"last_scout": datetime.now(timezone.utc).isoformat()},
    )
    return "done"
