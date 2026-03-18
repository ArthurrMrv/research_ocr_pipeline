from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone

import fitz  # pymupdf
from PIL import Image
from supabase import Client

from config import MAX_PAGES_PER_BATCH, OCR_MODEL_ID, OCR_PROVIDER, ZAI_MAX_PAGES
from pipeline import debug_logger
from pipeline.ocr_providers.registry import get_ocr_provider
from pipeline.tracker import (
    append_error,
    delete_ocr_rows,
    get_ocr_chunks,
    pipeline_get,
    pipeline_update,
    silver_upsert,
)


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


_LEADING_IMAGE_RE = re.compile(r"^(!\[.*?\]\(.*?\))\s*", re.DOTALL)
_PAGE_MARKER_RE = re.compile(r"^---\s*Page\s+\d+\s*---\s*$", re.MULTILINE)


def _is_empty_page_content(content: str) -> bool:
    """Return True if a stored OCR row contains no real text (only a page marker)."""
    return not _PAGE_MARKER_RE.sub("", content).strip()


def _add_page_marker(content: str, page_num: int) -> str:
    """Prepend '--- Page N ---' respecting any leading markdown image."""
    marker = f"--- Page {page_num} ---"
    m = _LEADING_IMAGE_RE.match(content)
    if m:
        return f"{m.group(1)}\n{marker}\n{content[m.end():]}"
    return f"{marker}\n{content}"


def extract_sub_pdf_bytes(pdf_path: str, start: int, end: int) -> bytes:
    """Extract 0-indexed pages [start, end) from PDF and return as bytes."""
    src = fitz.open(pdf_path)
    dst = fitz.open()
    dst.insert_pdf(src, from_page=start, to_page=end - 1)
    data = dst.tobytes()
    dst.close()
    src.close()
    return data


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


def _reocr_pages(
    doc_id: str,
    client: Client,
    page_nums: list[int],
    progress_callback: Callable[[int, int], None] | None,
) -> dict:
    """Re-OCR specific pages of an already-processed document and upsert results."""
    bronze_row = (
        client.table("bronze_mapping").select("file_path").eq("doc_id", doc_id).execute()
    )
    file_path = bronze_row.data[0]["file_path"]
    provider = get_ocr_provider(OCR_PROVIDER)

    newly_empty: list[int] = []
    for i, page_num in enumerate(page_nums):
        chunk = extract_sub_pdf_bytes(file_path, page_num - 1, page_num)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(chunk)
            tmp.flush()
            try:
                pages = provider.ocr_document(tmp.name)
                raw_content = pages[0] if pages else ""
            except NotImplementedError:
                continue

        if not raw_content.strip():
            newly_empty.append(page_num)
        debug_logger.print_ocr_page(page_num, empty=not raw_content.strip())
        content = _add_page_marker(raw_content, page_num)
        silver_upsert(
            client,
            "ocr_results",
            {
                "doc_id": doc_id,
                "page_number": page_num,
                "ocr_model": OCR_MODEL_ID,
                "content": content,
            },
        )
        if progress_callback:
            progress_callback(i + 1, len(page_nums))

    return {"status": "done", "empty_pages": len(newly_empty)}


def run_ocr(
    doc_id: str,
    client: Client,
    *,
    force: bool = False,
    since: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Run OCR for doc_id and store result in ocr_results.
    Chunks pages into batches of MAX_PAGES_PER_BATCH.
    Each chunk is stored as a separate row with a 'pages' column.
    Skips if already OCR'd unless force=True or since date matches.
    If already done but has empty pages, re-OCRs only those pages.
    Returns dict with 'status' ("skipped"|"done") and 'empty_pages' count.
    """
    pipeline_row = pipeline_get(client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    already_done = pipeline_row.get("last_ocr") is not None
    if already_done and not force and not _after_date(pipeline_row, since):
        existing_rows = get_ocr_chunks(client, doc_id)
        empty_page_nums = [
            r["page_number"] for r in existing_rows if _is_empty_page_content(r["content"])
        ]
        if not empty_page_nums:
            return {"status": "skipped", "empty_pages": 0}
        return _reocr_pages(doc_id, client, empty_page_nums, progress_callback)

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
        delete_ocr_rows(client, doc_id)

        empty_pages: list[int] = []

        try:
            # Fast path: provider can OCR the PDF natively
            with fitz.open(file_path) as doc:
                total_pages = len(doc)

            if total_pages <= ZAI_MAX_PAGES:
                pages = provider.ocr_document(file_path)
                for i, raw_content in enumerate(pages):
                    page_num = i + 1
                    if not raw_content.strip():
                        empty_pages.append(page_num)
                    debug_logger.print_ocr_page(page_num, empty=not raw_content.strip())
                    content = _add_page_marker(raw_content, page_num)
                    silver_upsert(
                        client,
                        "ocr_results",
                        {
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "ocr_model": OCR_MODEL_ID,
                            "content": content,
                        },
                    )
                if progress_callback is not None:
                    progress_callback(1, 1)
            else:
                total_batches = math.ceil(total_pages / ZAI_MAX_PAGES)
                for batch_idx in range(total_batches):
                    batch_start = batch_idx * ZAI_MAX_PAGES
                    batch_end = min(batch_start + ZAI_MAX_PAGES, total_pages)
                    chunk_bytes = extract_sub_pdf_bytes(file_path, batch_start, batch_end)
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                        tmp.write(chunk_bytes)
                        tmp.flush()
                        pages = provider.ocr_document(tmp.name)
                    for i, raw_content in enumerate(pages):
                        page_num = batch_start + i + 1
                        if not raw_content.strip():
                            empty_pages.append(page_num)
                        debug_logger.print_ocr_page(page_num, empty=not raw_content.strip())
                        content = _add_page_marker(raw_content, page_num)
                        silver_upsert(
                            client,
                            "ocr_results",
                            {
                                "doc_id": doc_id,
                                "page_number": page_num,
                                "ocr_model": OCR_MODEL_ID,
                                "content": content,
                            },
                        )
                    if progress_callback is not None:
                        progress_callback(batch_idx + 1, total_batches)

            if empty_pages:
                append_error(
                    client,
                    doc_id,
                    f"OCR produced {len(empty_pages)} empty page(s): {empty_pages}",
                )
        except NotImplementedError:
            # Slow path: rasterise and send page images
            images = pdf_to_images(file_path)
            total_pages = len(images)
            for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
                batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
                chunk_text = provider.ocr_pages(images[batch_start:batch_end], page_offset=batch_start)
                silver_upsert(
                    client,
                    "ocr_results",
                    {
                        "doc_id": doc_id,
                        "ocr_model": OCR_MODEL_ID,
                        "content": chunk_text,
                    },
                )
                if progress_callback is not None:
                    batch_index = batch_start // MAX_PAGES_PER_BATCH
                    total_batches = (total_pages + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH
                    progress_callback(batch_index + 1, total_batches)

        pipeline_update(
            client,
            doc_id,
            {"last_ocr": datetime.now(timezone.utc).isoformat()},
        )
        return {"status": "done", "empty_pages": len(empty_pages)}
    except Exception as exc:
        append_error(client, doc_id, f"OCR error: {exc}")
        raise
