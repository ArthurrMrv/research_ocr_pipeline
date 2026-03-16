from pipeline.providers.anthropic_provider import AnthropicProvider
from pipeline.providers.base import LLMProvider
from pipeline.providers.moonshot import MoonshotProvider
from pipeline.providers.openai_provider import OpenAIProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "moonshot": MoonshotProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(provider_name: str, step_config: dict) -> LLMProvider:
    """Instantiate and return the provider for the given name."""
    if provider_name not in PROVIDERS:
        available = list(PROVIDERS)
        raise ValueError(
            f"Unknown provider '{provider_name}'. Available: {available}"
        )
    return PROVIDERS[provider_name](step_config)
