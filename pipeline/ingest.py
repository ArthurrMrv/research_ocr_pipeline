import os
import uuid

from supabase import Client

from pipeline.tracker import bronze_insert, get_all_doc_ids, pipeline_insert

UUID_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


def make_doc_id(path: str) -> str:
    """Deterministic doc_id from absolute file path using UUID5."""
    abs_path = os.path.abspath(path)
    return str(uuid.uuid5(UUID_NS, abs_path))


def ingest(pdf_paths: list[str], client: Client) -> list[str]:
    """
    Register new PDFs in bronze_mapping. Skips already-registered paths.
    Returns list of newly added doc_ids.
    """
    existing_ids = set(get_all_doc_ids(client))
    new_ids = []

    for path in pdf_paths:
        doc_id = make_doc_id(path)
        if doc_id in existing_ids:
            continue

        doc_name = os.path.basename(path)
        bronze_insert(client, doc_id, os.path.abspath(path), doc_name)
        pipeline_insert(client, doc_id)
        new_ids.append(doc_id)

    return new_ids
