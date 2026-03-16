import json
import os
from datetime import datetime, timezone

from supabase import Client, create_client


def get_supabase_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def bronze_insert(client: Client, doc_id: str, file_path: str, doc_name: str) -> None:
    """Insert a new bronze record. Raises on conflict (duplicate doc_id)."""
    client.table("bronze_mapping").insert(
        {"doc_id": doc_id, "file_path": file_path, "doc_name": doc_name}
    ).execute()


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
    pipeline_update(client, doc_id, {"error": json.dumps(new_errors)})


def get_all_doc_ids(client: Client) -> list[str]:
    """Return all doc_ids registered in bronze_mapping."""
    result = client.table("bronze_mapping").select("doc_id").execute()
    return [row["doc_id"] for row in result.data]
