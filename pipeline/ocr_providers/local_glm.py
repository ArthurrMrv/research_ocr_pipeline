from __future__ import annotations

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from config import OCR_MODEL_ID
from pipeline.ocr_providers.base import OCRProvider

# Module-level singletons — loaded once per process
_ocr_model = None
_ocr_processor = None


def _load_model() -> tuple:
    """Load GLM-OCR model and processor (cached as module-level singleton)."""
    global _ocr_model, _ocr_processor
    if _ocr_model is None:
        _ocr_processor = AutoProcessor.from_pretrained(
            OCR_MODEL_ID, trust_remote_code=True
        )
        _ocr_model = AutoModelForCausalLM.from_pretrained(
            OCR_MODEL_ID, trust_remote_code=True
        )
        _ocr_model.eval()
    return _ocr_model, _ocr_processor


def _ocr_single_image(image: Image.Image, model, processor) -> str:
    """Run GLM-OCR on a single PIL image and return extracted text."""
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


class LocalGLMProvider(OCRProvider):
    """OCR provider using the local HuggingFace GLM-OCR model."""

    def ocr_pages(self, images: list[Image.Image], *, page_offset: int = 0) -> str:
        model, processor = _load_model()
        page_texts = []
        for i, img in enumerate(images, start=page_offset + 1):
            page_text = _ocr_single_image(img, model, processor)
            page_texts.append(f"--- Page {i} ---\n{page_text}")
        return "\n\n".join(page_texts)
