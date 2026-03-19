import os
import uuid

from supabase import Client

from pipeline.filename_parser import parse_filename
from pipeline.tracker import bronze_insert, bronze_update_path, get_all_doc_ids, get_bronze_row, pipeline_insert

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
        abs_path = os.path.abspath(path)
        doc_id = make_doc_id(abs_path)
        if doc_id in existing_ids:
            existing_row = get_bronze_row(client, doc_id)
            if existing_row and existing_row.get("file_path") != abs_path:
                bronze_update_path(client, doc_id, abs_path)
            continue

        doc_name = os.path.basename(path)
        institution, report_date = parse_filename(doc_name)
        bronze_insert(
            client,
            doc_id,
            abs_path,
            doc_name,
            institution=institution,
            report_date=report_date,
        )
        pipeline_insert(client, doc_id)
        new_ids.append(doc_id)

    return new_ids
