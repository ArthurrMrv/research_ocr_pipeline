from __future__ import annotations

import io
import os

from PIL import Image

from pipeline.ocr_providers.base import OCRProvider


class ZAIProvider(OCRProvider):
    """OCR provider using ZAI SDK's layout_parsing API."""

    def __init__(self) -> None:
        api_key = os.environ.get("ZAI_API_KEY")
        if not api_key:
            raise ValueError("ZAI_API_KEY environment variable is required for ZAI OCR provider")
        from zai import ZaiClient  # type: ignore[import-untyped]

        self._client = ZaiClient(api_key=api_key)

    def ocr_pages(self, images: list[Image.Image], *, page_offset: int = 0) -> str:
        page_texts = []
        for i, img in enumerate(images, start=page_offset + 1):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            result = self._client.layout_parsing.create(
                model="glm-ocr", file=buf
            )
            text = result.content if hasattr(result, "content") else str(result)
            page_texts.append(f"--- Page {i} ---\n{text.strip()}")
        return "\n\n".join(page_texts)
