import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.providers.base import LLMProvider
from pipeline.providers.registry import PROVIDERS, get_provider


class TestRegistry:
    def test_all_providers_registered(self):
        assert "moonshot" in PROVIDERS
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS

    def test_get_provider_returns_instance(self):
        config = {"provider": "moonshot", "model": "kimi-k2.5"}
        with patch("pipeline.providers.moonshot.OpenAI"):
            provider = get_provider("moonshot", config)
        assert isinstance(provider, LLMProvider)

    def test_get_provider_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider", {"model": "x"})


class TestMoonshotProvider:
    def _make_provider(self):
        from pipeline.providers.moonshot import MoonshotProvider
        with patch("pipeline.providers.moonshot.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider = MoonshotProvider({"model": "kimi-k2.5", "temperature": 0, "max_tokens": 1024})
            provider._client = mock_openai.return_value
        return provider

    def test_call_returns_parsed_json(self):
        provider = self._make_provider()
        provider._client.chat.completions.create.return_value.choices[0].message.content = (
            '{"model_name": "Alpha"}'
        )
        result = provider.call("Extract: {ocr_text}", "some text")
        assert result == {"model_name": "Alpha"}

    def test_call_fills_ocr_text_placeholder(self):
        provider = self._make_provider()
        provider._client.chat.completions.create.return_value.choices[0].message.content = "{}"
        provider.call("Prompt: {ocr_text}", "my ocr content")
        call_args = provider._client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert "my ocr content" in messages[0]["content"]

    def test_call_raises_on_invalid_json(self):
        provider = self._make_provider()
        provider._client.chat.completions.create.return_value.choices[0].message.content = (
            "not json"
        )
        with pytest.raises(ValueError, match="non-JSON"):
            provider.call("prompt", "ocr text")

    def test_uses_moonshot_base_url(self, monkeypatch):
        from pipeline.providers import moonshot as m
        with patch("pipeline.providers.moonshot.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from pipeline.providers.moonshot import MoonshotProvider
            MoonshotProvider({"model": "kimi-k2.5"})
            _, kwargs = mock_openai.call_args
            assert "api.moonshot.ai" in kwargs["base_url"]


class TestOpenAIProvider:
    def _make_provider(self, monkeypatch=None):
        if monkeypatch:
            monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        from pipeline.providers.openai_provider import OpenAIProvider
        with (
            patch("pipeline.providers.openai_provider.OpenAI") as mock_openai,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"}),
        ):
            mock_openai.return_value = MagicMock()
            provider = OpenAIProvider({"model": "gpt-4o", "temperature": 0, "max_tokens": 1024})
            provider._client = mock_openai.return_value
        return provider

    def test_call_returns_parsed_json(self):
        provider = self._make_provider()
        provider._client.chat.completions.create.return_value.choices[0].message.content = (
            '{"tables": []}'
        )
        result = provider.call("Extract tables: {ocr_text}", "text")
        assert result == {"tables": []}  # noqa: E501

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from pipeline.providers.openai_provider import OpenAIProvider
        with pytest.raises(KeyError):
            OpenAIProvider({"model": "gpt-4o"})


class TestAnthropicProvider:
    def _make_provider(self):
        from pipeline.providers.anthropic_provider import AnthropicProvider
        with (
            patch("pipeline.providers.anthropic_provider.anthropic") as mock_lib,
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-anthropic-key"}),
        ):
            mock_lib.Anthropic.return_value = MagicMock()
            provider = AnthropicProvider(
                {"model": "claude-sonnet-4-6", "temperature": 0, "max_tokens": 1024}
            )
            provider._client = mock_lib.Anthropic.return_value
        return provider

    def test_call_returns_parsed_json(self):
        provider = self._make_provider()
        provider._client.messages.create.return_value.content = [
            MagicMock(text='{"model_name": "Beta"}')
        ]
        result = provider.call("Extract: {ocr_text}", "text")
        assert result == {"model_name": "Beta"}

    def test_call_raises_on_invalid_json(self):
        provider = self._make_provider()
        provider._client.messages.create.return_value.content = [
            MagicMock(text="not json")
        ]
        with pytest.raises(ValueError, match="non-JSON"):
            provider.call("prompt", "ocr text")

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from pipeline.providers.anthropic_provider import AnthropicProvider
        with pytest.raises(KeyError):
            AnthropicProvider({"model": "claude-sonnet-4-6"})
