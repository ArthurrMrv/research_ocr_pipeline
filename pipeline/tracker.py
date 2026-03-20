import os
from datetime import datetime, timezone

from supabase import Client, create_client


def get_supabase_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def bronze_insert(
    client: Client,
    doc_id: str,
    file_path: str,
    doc_name: str,
    *,
    institution: str | None = None,
    report_date: str | None = None,
) -> None:
    """Insert a new bronze record. Raises on conflict (duplicate doc_id)."""
    row: dict = {"doc_id": doc_id, "file_path": file_path, "doc_name": doc_name}
    if institution is not None:
        row["institution"] = institution
    if report_date is not None:
        row["report_date"] = report_date
    client.table("bronze_mapping").insert(row).execute()


def bronze_update_path(client: Client, doc_id: str, file_path: str) -> None:
    """Update the stored absolute file path for an existing bronze record."""
    client.table("bronze_mapping").update({"file_path": file_path}).eq("doc_id", doc_id).execute()


def pipeline_insert(client: Client, doc_id: str) -> None:
    """Create an empty pipeline state row."""
    client.table("pipeline").insert({"doc_id": doc_id}).execute()


def pipeline_get(client: Client, doc_id: str) -> dict | None:
    result = (
        client.table("pipeline").select("*").eq("doc_id", doc_id).execute()
    )
    rows = result.data
    return rows[0] if rows else None


def pipeline_update(client: Client, doc_id: str, fields: dict) -> None:
    """Update pipeline state fields. Does not allow deletion."""
    client.table("pipeline").update(fields).eq("doc_id", doc_id).execute()


def silver_upsert(client: Client, table: str, row: dict, *, on_conflict: str | None = None) -> None:
    """Generic upsert for silver-layer tables."""
    query = client.table(table).upsert(row, on_conflict=on_conflict)
    query.execute()


def silver_bulk_upsert(client: Client, table: str, rows: list[dict], *, on_conflict: str | None = None) -> None:
    """Bulk upsert multiple rows into a silver-layer table in a single request."""
    if not rows:
        return
    client.table(table).upsert(rows, on_conflict=on_conflict).execute()


def formatting_upsert(client: Client, row: dict) -> None:
    """Upsert a formatting result keyed by (doc_id, step_name)."""
    client.table("formatting").upsert(row, on_conflict="doc_id,step_name").execute()


def delete_formatting_results(client: Client, doc_id: str) -> None:
    """Delete all formatting rows for a doc_id before a fresh formatting run."""
    client.table("formatting").delete().eq("doc_id", doc_id).execute()


def get_formatting_results(client: Client, doc_id: str) -> dict:
    """Return formatting results as {step_name: row_dict} for a doc_id."""
    result = (
        client.table("formatting")
        .select("*")
        .eq("doc_id", doc_id)
        .execute()
    )
    rows = result.data or []
    return {row["step_name"]: row for row in rows}


def append_error(client: Client, doc_id: str, error_msg: str, *, context: dict | None = None) -> None:
    """Append an error message to the JSONB error array in pipeline."""
    row = pipeline_get(client, doc_id)
    if row is None:
        return
    current_errors = row.get("error") or []
    error_entry = {"message": error_msg, "ts": datetime.now(timezone.utc).isoformat()}
    if context:
        error_entry.update(context)
    new_errors = list(current_errors) + [error_entry]
    pipeline_update(client, doc_id, {"error": new_errors})


def get_all_doc_ids(client: Client) -> list[str]:
    """Return all doc_ids registered in bronze_mapping."""
    result = client.table("bronze_mapping").select("doc_id").execute()
    return [row["doc_id"] for row in result.data]


def get_all_bronze_rows(client: Client) -> list[dict]:
    """Return all bronze_mapping rows in a single query."""
    result = client.table("bronze_mapping").select("*").execute()
    return result.data or []


def get_bronze_row(client: Client, doc_id: str) -> dict | None:
    """Return the bronze_mapping row for a doc_id, or None if not found."""
    result = (
        client.table("bronze_mapping").select("*").eq("doc_id", doc_id).execute()
    )
    rows = result.data
    return rows[0] if rows else None


def get_ocr_chunks(client: Client, doc_id: str) -> list[dict]:
    """Return all ocr_results rows for doc_id, sorted by page_number."""
    result = (
        client.table("ocr_results")
        .select("*")
        .eq("doc_id", doc_id)
        .execute()
    )
    rows = result.data or []
    return sorted(rows, key=lambda r: r.get("page_number") or 0)


def get_ocr_pages_for_range(
    client: Client, doc_id: str, start: int, end: int
) -> list[dict]:
    """Return ocr_results rows where page_number BETWEEN start AND end, sorted."""
    result = (
        client.table("ocr_results")
        .select("*")
        .eq("doc_id", doc_id)
        .gte("page_number", start)
        .lte("page_number", end)
        .execute()
    )
    rows = result.data or []
    return sorted(rows, key=lambda r: r.get("page_number", 0))


def delete_ocr_rows(client: Client, doc_id: str) -> None:
    """Delete all ocr_results rows for a doc_id (cleanup before re-OCR)."""
    client.table("ocr_results").delete().eq("doc_id", doc_id).execute()


def scout_page_score_upsert(client: Client, row: dict) -> None:
    """Upsert a scout page score keyed by (doc_id, step_name, page_number)."""
    client.table("scout_page_scores").upsert(
        row,
        on_conflict="doc_id,step_name,page_number",
    ).execute()


def scout_page_scores_bulk_upsert(client: Client, rows: list[dict]) -> None:
    """Bulk upsert scout page scores in a single request."""
    if not rows:
        return
    client.table("scout_page_scores").upsert(
        rows,
        on_conflict="doc_id,step_name,page_number",
    ).execute()


def delete_scout_page_scores(client: Client, doc_id: str) -> None:
    """Delete all scout page scores for a doc_id before a fresh scout run."""
    client.table("scout_page_scores").delete().eq("doc_id", doc_id).execute()


def increment_formatting_attempts(client: Client, doc_id: str) -> int:
    """Increment formatting_attempts counter and return the new value."""
    row = pipeline_get(client, doc_id)
    if row is None:
        return 0
    new_count = (row.get("formatting_attempts") or 0) + 1
    pipeline_update(client, doc_id, {"formatting_attempts": new_count})
    return new_count


def get_scout_page_scores(client: Client, doc_id: str, *, step_name: str | None = None) -> list[dict]:
    """Return scout page score rows for a doc_id, optionally filtered by step_name."""
    query = (
        client.table("scout_page_scores")
        .select("*")
        .eq("doc_id", doc_id)
    )
    if step_name is not None:
        query = query.eq("step_name", step_name)
    result = query.execute()
    rows = result.data or []
    return sorted(rows, key=lambda r: (r.get("step_name") or "", r.get("page_number") or 0))
