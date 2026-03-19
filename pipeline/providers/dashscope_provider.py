from __future__ import annotations

import json
import os

from openai import OpenAI

from pipeline import debug_logger
from pipeline.providers.base import LLMProvider

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeProvider(LLMProvider):
    """Alibaba DashScope provider via OpenAI-compatible API."""

    def __init__(self, step_config: dict) -> None:
        super().__init__(step_config)
        api_key = os.environ["DASHSCOPE_API_KEY"]
        self._client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)

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
            debug_logger.print_llm_response(f"DashScope / {self.model}", raw, result)
            return result
        except json.JSONDecodeError as exc:
            debug_logger.print_llm_response(f"DashScope / {self.model}", raw)
            raise self._non_json_error("DashScope", raw) from exc
