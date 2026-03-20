from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from openai import OpenAI
from pipeline import debug_logger

_T = TypeVar("_T")


class NonJSONResponseError(ValueError):
    """Raised when a provider returns text that cannot be parsed as JSON."""

    def __init__(self, provider_name: str, raw_response: str) -> None:
        self.provider_name = provider_name
        self.raw_response = raw_response
        super().__init__(f"{provider_name} returned non-JSON: {raw_response[:200]}")


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

    def _non_json_error(self, provider_name: str, raw: str) -> NonJSONResponseError:
        """Build a structured non-JSON error that keeps the full raw response."""
        return NonJSONResponseError(provider_name, raw)


class OpenAICompatibleProvider(LLMProvider):
    """Shared base for providers that use the OpenAI SDK with a custom base_url."""

    _provider_label: str  # e.g. "Moonshot", "Gemini"
    _base_url: str | None = None  # None means default OpenAI endpoint
    _env_var: str  # e.g. "MOONSHOT_API_KEY"

    def __init__(self, step_config: dict) -> None:
        super().__init__(step_config)
        api_key = os.environ[self._env_var]
        kwargs: dict = {"api_key": api_key}
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def call(self, prompt: str, ocr_text: str) -> dict:
        filled = prompt.replace("{ocr_text}", ocr_text)
        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": filled}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )
        raw = response.choices[0].message.content
        try:
            result = json.loads(raw)
            debug_logger.print_llm_response(f"{self._provider_label} / {self.model}", raw, result)
            return result
        except json.JSONDecodeError as exc:
            debug_logger.print_llm_response(f"{self._provider_label} / {self.model}", raw)
            raise self._non_json_error(self._provider_label, raw) from exc
