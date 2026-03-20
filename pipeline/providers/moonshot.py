from pipeline.providers.base import OpenAICompatibleProvider

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot / Kimi provider via OpenAI-compatible API."""

    _provider_label = "Moonshot"
    _base_url = MOONSHOT_BASE_URL
    _env_var = "MOONSHOT_API_KEY"
