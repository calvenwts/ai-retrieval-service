"""Unit tests for the provider adapter layer."""

import pytest

from app.providers import ChatRequest, ChatResponse, _resolve_provider


def test_resolve_anthropic():
    assert _resolve_provider("claude-sonnet-4-6") == "anthropic"
    assert _resolve_provider("claude-haiku-4-5-20251001") == "anthropic"


def test_resolve_openai():
    assert _resolve_provider("gpt-4o") == "openai"
    assert _resolve_provider("gpt-4o-mini") == "openai"


def test_chat_request_defaults():
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.model == "claude-sonnet-4-6"
    assert req.max_tokens == 1024
    assert req.stream is False


def test_chat_response_model():
    resp = ChatResponse(text="hello", model="claude-sonnet-4-6", input_tokens=10, output_tokens=5)
    assert resp.text == "hello"
    assert resp.input_tokens == 10
