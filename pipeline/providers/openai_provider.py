from pipeline.providers.base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Standard OpenAI provider."""

    _provider_label = "OpenAI"
    _env_var = "OPENAI_API_KEY"
