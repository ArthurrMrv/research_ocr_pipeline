from pipeline.providers.base import OpenAICompatibleProvider

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeProvider(OpenAICompatibleProvider):
    """Alibaba DashScope provider via OpenAI-compatible API."""

    _provider_label = "DashScope"
    _base_url = DASHSCOPE_BASE_URL
    _env_var = "DASHSCOPE_API_KEY"
