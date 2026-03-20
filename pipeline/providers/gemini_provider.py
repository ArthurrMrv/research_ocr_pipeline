from pipeline.providers.base import OpenAICompatibleProvider

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini provider via OpenAI-compatible API."""

    _provider_label = "Gemini"
    _base_url = GEMINI_BASE_URL
    _env_var = "GOOGLE_API_KEY"
