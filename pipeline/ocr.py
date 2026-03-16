from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import fitz  # pymupdf
from PIL import Image
from supabase import Client
from transformers import AutoModelForCausalLM, AutoProcessor

from config import OCR_MODEL_ID
from pipeline.tracker import append_error, pipeline_get, pipeline_update, silver_upsert

if TYPE_CHECKING:
    pass

# Module-level singletons — loaded once per process
_ocr_model = None
_ocr_processor = None


def load_ocr_model() -> tuple:
    """Load GLM-OCR model and processor (cached as module-level singleton)."""
    global _ocr_model, _ocr_processor
    if _ocr_model is None:
        _ocr_processor = AutoProcessor.from_pretrained(OCR_MODEL_ID, trust_remote_code=True)
        _ocr_model = AutoModelForCausalLM.from_pretrained(
            OCR_MODEL_ID, trust_remote_code=True
        )
        _ocr_model.eval()
    return _ocr_model, _ocr_processor


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


def ocr_image(image: Image.Image, model, processor) -> str:
    """Run GLM-OCR on a single PIL image and return extracted text."""
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


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
    Skips if already OCR'd unless force=True or since date matches.
    """
    pipeline_row = pipeline_get(client, doc_id)
    if pipeline_row is None:
        raise ValueError(f"No pipeline row for doc_id={doc_id}")

    already_done = pipeline_row.get("last_ocr") is not None
    if already_done and not force and not _after_date(pipeline_row, since):
        return

    # Fetch file path from bronze_mapping
    from supabase import Client as _Client  # noqa: F401
    bronze_row = (
        client.table("bronze_mapping").select("file_path").eq("doc_id", doc_id).execute()
    )
    if not bronze_row.data:
        raise ValueError(f"No bronze_mapping row for doc_id={doc_id}")
    file_path = bronze_row.data[0]["file_path"]

    try:
        model, processor = load_ocr_model()
        images = pdf_to_images(file_path)
        page_texts = []
        for i, img in enumerate(images, start=1):
            page_text = ocr_image(img, model, processor)
            page_texts.append(f"--- Page {i} ---\n{page_text}")
        full_text = "\n\n".join(page_texts)

        silver_upsert(
            client,
            "ocr_results",
            {"doc_id": doc_id, "ocr_model": OCR_MODEL_ID, "content": full_text},
        )
        pipeline_update(
            client,
            doc_id,
            {"last_ocr": datetime.now(timezone.utc).isoformat()},
        )
    except Exception as exc:
        append_error(client, doc_id, f"OCR error: {exc}")
        raise
