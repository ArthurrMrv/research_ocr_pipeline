import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.providers.base import LLMProvider, NonJSONResponseError
from pipeline.providers.registry import PROVIDERS, get_provider


class TestRegistry:
    def test_all_providers_registered(self):
        assert "moonshot" in PROVIDERS
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS
        assert "gemini" in PROVIDERS
        assert "dashscope" in PROVIDERS
        assert "openrouter" in PROVIDERS

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
        with pytest.raises(NonJSONResponseError, match="non-JSON"):
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
        with pytest.raises(NonJSONResponseError, match="non-JSON"):
            provider.call("prompt", "ocr text")

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from pipeline.providers.anthropic_provider import AnthropicProvider
        with pytest.raises(KeyError):
            AnthropicProvider({"model": "claude-sonnet-4-6"})


def _make_openai_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


_STEP_CONFIG = {"model": "test-model", "temperature": 0, "max_tokens": 512}


class TestGeminiProvider:
    def test_init_requires_google_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from pipeline.providers.gemini_provider import GeminiProvider
        with patch("pipeline.providers.gemini_provider.OpenAI"):
            with pytest.raises(KeyError):
                GeminiProvider(_STEP_CONFIG)

    def test_call_returns_parsed_json(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        payload = {"result": "gemini answer"}
        mock_response = _make_openai_response(json.dumps(payload))

        from pipeline.providers.gemini_provider import GeminiProvider
        with patch("pipeline.providers.gemini_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = GeminiProvider(_STEP_CONFIG)
            result = provider.call("Extract: {ocr_text}", "some ocr")

        assert result == payload
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"

    def test_call_raises_on_non_json(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        mock_response = _make_openai_response("not json at all")

        from pipeline.providers.gemini_provider import GeminiProvider
        with patch("pipeline.providers.gemini_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = GeminiProvider(_STEP_CONFIG)
            with pytest.raises(NonJSONResponseError) as excinfo:
                provider.call("prompt {ocr_text}", "text")

        assert excinfo.value.raw_response == "not json at all"

    def test_call_preserves_full_non_json_raw_response(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        raw = "{\n  \"tables\": [\n    1,\n    2\n  ]\n}\ntrailing text"
        mock_response = _make_openai_response(raw)

        from pipeline.providers.gemini_provider import GeminiProvider
        with (
            patch("pipeline.providers.gemini_provider.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = GeminiProvider(_STEP_CONFIG)
            with pytest.raises(NonJSONResponseError) as excinfo:
                provider.call("prompt {ocr_text}", "text")

        assert excinfo.value.raw_response == raw

    def test_call_truncates_non_json_exception_message(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        raw = "x" * 500
        mock_response = _make_openai_response(raw)

        from pipeline.providers.gemini_provider import GeminiProvider
        with (
            patch("pipeline.providers.gemini_provider.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = GeminiProvider(_STEP_CONFIG)
            with pytest.raises(NonJSONResponseError) as excinfo:
                provider.call("prompt {ocr_text}", "text")

        assert ("x" * 200) in str(excinfo.value)
        assert ("x" * 201) not in str(excinfo.value)

    def test_uses_gemini_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        from pipeline.providers.gemini_provider import GeminiProvider
        with patch("pipeline.providers.gemini_provider.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            GeminiProvider(_STEP_CONFIG)

        _, kwargs = mock_openai_cls.call_args
        assert "generativelanguage.googleapis.com" in kwargs["base_url"]


class TestOpenRouterProvider:
    def test_init_requires_openrouter_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from pipeline.providers.openrouter_provider import OpenRouterProvider
        with patch("pipeline.providers.openrouter_provider.OpenAI"):
            with pytest.raises(KeyError):
                OpenRouterProvider(_STEP_CONFIG)

    def test_call_returns_parsed_json(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
        payload = {"result": "openrouter answer"}
        mock_response = _make_openai_response(json.dumps(payload))

        from pipeline.providers.openrouter_provider import OpenRouterProvider
        with patch("pipeline.providers.openrouter_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = OpenRouterProvider(_STEP_CONFIG)
            result = provider.call("Extract: {ocr_text}", "some ocr")

        assert result == payload
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"

    def test_call_raises_on_non_json(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
        mock_response = _make_openai_response("not json")

        from pipeline.providers.openrouter_provider import OpenRouterProvider
        with patch("pipeline.providers.openrouter_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = OpenRouterProvider(_STEP_CONFIG)
            with pytest.raises(ValueError, match="non-JSON"):
                provider.call("prompt {ocr_text}", "text")

    def test_uses_openrouter_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
        from pipeline.providers.openrouter_provider import OpenRouterProvider
        with patch("pipeline.providers.openrouter_provider.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            OpenRouterProvider(_STEP_CONFIG)

        _, kwargs = mock_openai_cls.call_args
        assert "openrouter.ai" in kwargs["base_url"]


class TestDashScopeProvider:
    def test_init_requires_dashscope_api_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        from pipeline.providers.dashscope_provider import DashScopeProvider
        with patch("pipeline.providers.dashscope_provider.OpenAI"):
            with pytest.raises(KeyError):
                DashScopeProvider(_STEP_CONFIG)

    def test_call_returns_parsed_json(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
        payload = {"result": "dashscope answer"}
        mock_response = _make_openai_response(json.dumps(payload))

        from pipeline.providers.dashscope_provider import DashScopeProvider
        with patch("pipeline.providers.dashscope_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = DashScopeProvider(_STEP_CONFIG)
            result = provider.call("Extract: {ocr_text}", "ocr text")

        assert result == payload

    def test_call_raises_on_non_json(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
        mock_response = _make_openai_response("not json")

        from pipeline.providers.dashscope_provider import DashScopeProvider
        with patch("pipeline.providers.dashscope_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            provider = DashScopeProvider(_STEP_CONFIG)
            with pytest.raises(ValueError, match="non-JSON"):
                provider.call("prompt {ocr_text}", "text")

    def test_uses_dashscope_base_url(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
        from pipeline.providers.dashscope_provider import DashScopeProvider
        with patch("pipeline.providers.dashscope_provider.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            DashScopeProvider(_STEP_CONFIG)

        _, kwargs = mock_openai_cls.call_args
        assert "dashscope.aliyuncs.com" in kwargs["base_url"]
