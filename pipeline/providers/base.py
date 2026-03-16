from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, step_config: dict) -> None:
        self.model: str = step_config["model"]
        self.temperature: float = step_config.get("temperature", 0)
        self.max_tokens: int = step_config.get("max_tokens", 2048)

    @abstractmethod
    def call(self, prompt: str, ocr_text: str) -> dict:
        """
        Fill {ocr_text} placeholder in prompt, call the API,
        parse and return the response as a JSON dict.

        Raises:
            ValueError: if the API response cannot be parsed as JSON.
        """
