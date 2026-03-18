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


def silver_upsert(client: Client, table: str, row: dict) -> None:
    """Generic upsert for silver-layer tables."""
    client.table(table).upsert(row).execute()


def formatting_upsert(client: Client, row: dict) -> None:
    """Upsert a formatting result keyed by (doc_id, step_name)."""
    client.table("formatting").upsert(row).execute()


def append_error(client: Client, doc_id: str, error_msg: str) -> None:
    """Append an error message to the JSONB error array in pipeline."""
    row = pipeline_get(client, doc_id)
    if row is None:
        return
    current_errors = row.get("error") or []
    new_errors = list(current_errors) + [
        {"message": error_msg, "ts": datetime.now(timezone.utc).isoformat()}
    ]
    pipeline_update(client, doc_id, {"error": new_errors})


def get_all_doc_ids(client: Client) -> list[str]:
    """Return all doc_ids registered in bronze_mapping."""
    result = client.table("bronze_mapping").select("doc_id").execute()
    return [row["doc_id"] for row in result.data]


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


def scout_upsert(client: Client, row: dict) -> None:
    """Upsert a scout result keyed by (doc_id, step_name)."""
    client.table("scout_results").upsert(row).execute()


def get_scout_results(client: Client, doc_id: str) -> dict:
    """Return scout results as {step_name: row_dict} for a doc_id."""
    result = (
        client.table("scout_results")
        .select("*")
        .eq("doc_id", doc_id)
        .execute()
    )
    rows = result.data or []
    return {row["step_name"]: row for row in rows}
