from pipeline.ocr_providers.base import OCRProvider


def get_ocr_provider(provider_name: str) -> OCRProvider:
    """Instantiate and return the OCR provider for the given name."""
    if provider_name == "local":
        from pipeline.ocr_providers.local_glm import LocalGLMProvider

        return LocalGLMProvider()
    elif provider_name == "zai":
        from pipeline.ocr_providers.zai_provider import ZAIProvider

        return ZAIProvider()
    else:
        raise ValueError(
            f"Unknown OCR provider '{provider_name}'. Available: ['local', 'zai']"
        )
