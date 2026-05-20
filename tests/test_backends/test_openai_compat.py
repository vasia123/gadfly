"""OpenAICompatBackend unit tests.

We monkeypatch urllib.request.urlopen so no network traffic is generated
and the tests are deterministic. The fixture captures the outgoing
Request and provides a configurable fake response, letting each test
assert one shape (payload, parse path, error path) in isolation.
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from typing import Any

import pytest

from gadfly.backends.openai_compat import OpenAICompatBackend
from gadfly.prompts import SUBMIT_VERDICT_JSON_SCHEMA


class _FakeResponse:
    """Minimal context-manager response for urllib.request.urlopen."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _install_urlopen(monkeypatch, *, body: dict[str, Any] | None = None,
                    raise_exc: Exception | None = None) -> dict[str, Any]:
    """Replace urllib.request.urlopen and return a dict that captures
    the outgoing Request for assertions."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        if raise_exc is not None:
            raise raise_exc
        payload = json.dumps(body or {}).encode("utf-8")
        return _FakeResponse(payload)

    import urllib.request as ur
    monkeypatch.setattr(ur, "urlopen", fake_urlopen)
    return captured


def _ok_response(verdict_args: dict[str, Any]) -> dict[str, Any]:
    """Build a Chat-Completions-shaped response with a forced tool-call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "submit_verdict",
                        "arguments": json.dumps(verdict_args),
                    },
                }],
            },
        }],
    }


def _evaluate(backend: OpenAICompatBackend, **overrides: Any):
    kwargs = dict(
        system_prompt="SYS",
        user_message="USR",
        model="test-model",
        tool_name="submit_verdict",
        tool_description="desc",
        tool_parameters=SUBMIT_VERDICT_JSON_SCHEMA,
        timeout_s=10.0,
    )
    kwargs.update(overrides)
    return asyncio.run(backend.evaluate(**kwargs))


def test_request_payload_shape(monkeypatch):
    cap = _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(base_url="https://api.example.com/v1", api_key="sk-xxx")
    _evaluate(backend)

    assert cap["url"] == "https://api.example.com/v1/chat/completions"
    assert cap["method"] == "POST"
    assert cap["headers"]["Content-type"] == "application/json"
    assert cap["headers"]["Authorization"] == "Bearer sk-xxx"

    body = cap["body"]
    assert body["model"] == "test-model"
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_verdict"},
    }
    assert body["temperature"] == 0
    assert len(body["tools"]) == 1
    fn = body["tools"][0]["function"]
    assert fn["name"] == "submit_verdict"
    assert fn["parameters"] == SUBMIT_VERDICT_JSON_SCHEMA


def test_parses_tool_call_arguments(monkeypatch):
    _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": False, "reason": "stub", "suggestion": "implement"}
    ))
    backend = OpenAICompatBackend(base_url="https://example.com/v1", api_key="k")
    res = _evaluate(backend)
    assert res.error is None
    assert res.verdict_args == {
        "professional": False, "reason": "stub", "suggestion": "implement",
    }
    assert res.latency_ms is not None and res.latency_ms >= 0


def test_accepts_dict_arguments(monkeypatch):
    """Some servers return arguments as parsed dict, not JSON string."""
    payload = {
        "choices": [{"message": {"tool_calls": [{
            "function": {
                "name": "submit_verdict",
                "arguments": {"professional": True, "reason": "", "suggestion": ""},
            },
        }]}}],
    }
    _install_urlopen(monkeypatch, body=payload)
    backend = OpenAICompatBackend(base_url="https://example.com/v1", api_key="k")
    res = _evaluate(backend)
    assert res.error is None
    assert res.verdict_args["professional"] is True


def test_handles_missing_tool_call(monkeypatch):
    payload = {"choices": [{"message": {"role": "assistant", "content": "I refuse"}}]}
    _install_urlopen(monkeypatch, body=payload)
    backend = OpenAICompatBackend(base_url="https://example.com/v1", api_key="k")
    res = _evaluate(backend)
    assert res.verdict_args is None
    assert "did not call" in (res.error or "")


def test_handles_malformed_arguments(monkeypatch):
    payload = {"choices": [{"message": {"tool_calls": [{
        "function": {"name": "submit_verdict", "arguments": "not json {"}
    }]}}]}
    _install_urlopen(monkeypatch, body=payload)
    backend = OpenAICompatBackend(base_url="https://example.com/v1", api_key="k")
    res = _evaluate(backend)
    assert res.verdict_args is None
    assert "not JSON" in (res.error or "")


def test_handles_http_error(monkeypatch):
    exc = urllib.error.HTTPError(
        url="https://example.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"bad key"}'),
    )
    _install_urlopen(monkeypatch, raise_exc=exc)
    backend = OpenAICompatBackend(base_url="https://example.com/v1", api_key="bad")
    res = _evaluate(backend)
    assert res.verdict_args is None
    assert res.error and res.error.startswith("HTTP 401")


def test_handles_url_error(monkeypatch):
    exc = urllib.error.URLError("Connection refused")
    _install_urlopen(monkeypatch, raise_exc=exc)
    backend = OpenAICompatBackend(base_url="https://example.com/v1", api_key="k")
    res = _evaluate(backend)
    assert res.verdict_args is None
    assert res.error and "URL error" in res.error


def test_strips_trailing_slash_on_base_url(monkeypatch):
    cap = _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(base_url="https://example.com/v1/", api_key="k")
    _evaluate(backend)
    assert cap["url"] == "https://example.com/v1/chat/completions"


def test_appends_v1_when_base_is_root(monkeypatch):
    cap = _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(base_url="https://example.com", api_key="k")
    _evaluate(backend)
    assert cap["url"] == "https://example.com/v1/chat/completions"


def test_accepts_full_chat_completions_url(monkeypatch):
    cap = _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(
        base_url="https://example.com/v1/chat/completions", api_key="k"
    )
    _evaluate(backend)
    assert cap["url"] == "https://example.com/v1/chat/completions"


def test_empty_api_key_omits_authorization_header(monkeypatch):
    cap = _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(base_url="http://localhost:8000/v1", api_key="")
    _evaluate(backend)
    assert "Authorization" not in cap["headers"]


def test_extra_headers_included(monkeypatch):
    cap = _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(
        base_url="https://example.com/v1", api_key="k",
        extra_headers={"anthropic-version": "2023-06-01", "x-custom": "hello"},
    )
    _evaluate(backend)
    # urllib lower-cases header keys via Request.header_items
    headers_lower = {k.lower(): v for k, v in cap["headers"].items()}
    assert headers_lower["anthropic-version"] == "2023-06-01"
    assert headers_lower["x-custom"] == "hello"


def test_empty_base_url_raises(monkeypatch):
    _install_urlopen(monkeypatch, body=_ok_response(
        {"professional": True, "reason": "", "suggestion": ""}
    ))
    backend = OpenAICompatBackend(base_url="", api_key="k")
    res = _evaluate(backend)
    # Empty base_url is detected when we try to build the endpoint;
    # surfaces as transport error in BackendResult, not raised.
    assert res.verdict_args is None
    assert res.error and "base_url" in res.error
