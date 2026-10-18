import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.core.llm_client import DeepSeekClient, IncompleteResponseError


def test_summary_rejects_length_even_when_content_is_valid_json(monkeypatch):
    def respond(request):
        return httpx.Response(200, json={"choices": [{
            "message": {"content": '{"current_task":"partial"}'}, "finish_reason": "length",
        }]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    client = DeepSeekClient("test-key", "https://example.test", "deepseek-v4-flash")
    with pytest.raises(RuntimeError, match="Incomplete LLM response"):
        asyncio.run(client.chat(messages=[], require_complete=True))


def test_incomplete_response_preserves_diagnostics_without_reasoning_text(monkeypatch):
    usage = {"completion_tokens": 6000, "completion_tokens_details": {"reasoning_tokens": 5900}}

    def respond(request):
        return httpx.Response(200, json={"usage": usage, "choices": [{
            "message": {"content": "", "reasoning_content": "private reasoning"}, "finish_reason": "length"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    client = DeepSeekClient("test-key", "https://example.test", "deepseek-v4-flash")
    with pytest.raises(IncompleteResponseError) as error:
        asyncio.run(client.chat(messages=[], require_complete=True, max_tokens=6000))
    assert error.value.finish_reason == "length"
    assert error.value.usage == usage and error.value.max_tokens == 6000
    assert "private reasoning" not in str(error.value) and "test-key" not in str(error.value)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("override,expected", [(None, "disabled"), (True, "enabled")])
def test_agent_thinking_applies_to_both_transports(monkeypatch, stream, override, expected):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if stream:
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"OK."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK."}, "finish_reason": "stop"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    settings = Settings(_env_file=None)
    assert settings.deepseek_thinking is False
    client = DeepSeekClient("test-key", "https://example.test", "deepseek-v4-pro",
                            default_thinking=settings.deepseek_thinking)

    async def call():
        kwargs = {"messages": [{"role": "user", "content": "test"}], "thinking": override}
        if stream:
            return "".join([part async for part in client.chat_stream(**kwargs)])
        return await client.chat(**kwargs)

    assert asyncio.run(call()) == "OK."
    assert payloads[0]["thinking"] == {"type": expected}


@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_only_budget_exhaustion_is_explicit(monkeypatch, stream):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if stream:
            return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning_content":"reasoning"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "reasoning_content": "reasoning"}, "finish_reason": "length"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    client = DeepSeekClient("test-key", "https://example.test", "deepseek-v4-pro")

    async def call():
        kwargs = {"messages": [{"role": "user", "content": "test"}], "continue_on_length": True}
        if stream:
            return [part async for part in client.chat_stream(**kwargs)]
        return await client.chat(**kwargs)

    with pytest.raises(RuntimeError, match="budget exhausted before final content"):
        asyncio.run(call())
    assert len(payloads) == 1
