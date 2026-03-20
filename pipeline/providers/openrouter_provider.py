from pipeline.providers.base import OpenAICompatibleProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider via OpenAI-compatible API."""

    _provider_label = "OpenRouter"
    _base_url = OPENROUTER_BASE_URL
    _env_var = "OPENROUTER_API_KEY"
