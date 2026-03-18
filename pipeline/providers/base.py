from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


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

    def _call_with_retry(self, fn: Callable[[], _T], *, retries: int = 3) -> _T:
        """Call fn(), retrying on rate-limit (429) or transient server (5xx) errors."""
        for attempt in range(retries):
            try:
                return fn()
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                status = getattr(exc, "status_code", None)
                if status is not None and (status == 429 or status >= 500):
                    time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s
                else:
                    raise
        # unreachable, satisfies type checker
        raise RuntimeError("unreachable")
