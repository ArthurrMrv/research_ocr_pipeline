import json
import os

import anthropic

from pipeline.providers.base import LLMProvider

JSON_SYSTEM_PROMPT = (
    "You are a data extraction assistant. Always respond with valid JSON only. "
    "Do not include any explanation, markdown formatting, or text outside the JSON object."
)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the native Anthropic SDK."""

    def __init__(self, step_config: dict) -> None:
        super().__init__(step_config)
        api_key = os.environ["ANTHROPIC_API_KEY"]
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(self, prompt: str, ocr_text: str) -> dict:
        filled = prompt.replace("{ocr_text}", ocr_text)
        response = self._call_with_retry(
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=JSON_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": filled}],
            )
        )
        raw = response.content[0].text
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Anthropic returned non-JSON: {raw[:200]}") from exc
