from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import fitz  # pymupdf
from PIL import Image
from supabase import Client

from config import MAX_PAGES_PER_BATCH, OCR_MODEL_ID, OCR_PROVIDER
from pipeline.ocr_providers.registry import get_ocr_provider
from pipeline.tracker import (
    append_error,
    delete_ocr_rows,
    pipeline_get,
    pipeline_update,
    silver_upsert,
)

if TYPE_CHECKING:
    pass


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    """Render each PDF page to a PIL Image at 150 DPI."""
    doc = fitz.open(pdf_path)
    images = []
    dpi = 150
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def make_page_range(start: int, end: int) -> str:
    """Return a page range string like '1-75' or '76-150'."""
    return f"{start}-{end}"


def _after_date(row: dict, since: str | None) -> bool:
    """Return True if row['added'] >= since (ISO date string). Returns False when since is None."""
    if since is None:
        return False
    added = row.get("added")
    if added is None:
        return False
    added_dt = datetime.fromisoformat(str(added).replace("Z", "+00:00"))
    since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    return added_dt >= since_dt


def run_ocr(
    doc_id: str,
    client: Client,
    *,
    force: bool = False,
    since: str | None = None,
) -> None:
    """
    Run OCR for doc_id and store result in ocr_results.
    Chunks pages into batches of MAX_PAGES_PER_BATCH.
    Each chunk is stored as a separate row with a 'pages' column.
    Skips if already OCR'd unless force=True or since date matches.
    """
    pipeline_row = pipeline_get(client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    already_done = pipeline_row.get("last_ocr") is not None
    if already_done and not force and not _after_date(pipeline_row, since):
        return

    # Fetch file path from bronze_mapping
    bronze_row = (
        client.table("bronze_mapping")
        .select("file_path")
        .eq("doc_id", doc_id)
        .execute()
    )
    if not bronze_row.data:
        raise ValueError(f"No bronze_mapping row for doc_id={doc_id}")
    file_path = bronze_row.data[0]["file_path"]

    try:
        provider = get_ocr_provider(OCR_PROVIDER)
        images = pdf_to_images(file_path)
        total_pages = len(images)

        # Delete existing OCR rows before writing new chunks
        delete_ocr_rows(client, doc_id)

        # Process in batches
        for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
            batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
            batch_images = images[batch_start:batch_end]

            chunk_text = provider.ocr_pages(batch_images, page_offset=batch_start)

            page_range = make_page_range(batch_start + 1, batch_end)
            silver_upsert(
                client,
                "ocr_results",
                {
                    "doc_id": doc_id,
                    "pages": page_range,
                    "ocr_model": OCR_MODEL_ID,
                    "content": chunk_text,
                },
            )

        pipeline_update(
            client,
            doc_id,
            {"last_ocr": datetime.now(timezone.utc).isoformat()},
        )
    except Exception as exc:
        append_error(client, doc_id, f"OCR error: {exc}")
        raise
