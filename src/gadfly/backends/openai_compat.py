"""OpenAI-compatible HTTP backend.

Talks Chat Completions wire protocol. Works against anything that
exposes `/v1/chat/completions` with tool-calling: OpenAI, Anthropic's
OpenAI-compat endpoint, OpenRouter, vLLM, llama.cpp server, Ollama,
Groq, Together, DeepInfra, and any local stack.

stdlib only (`urllib.request` + `json`) — same no-deps policy as
viewer.py. The request shape is fully static; httpx/openai-sdk would
be overkill.

The forced tool-choice (`tool_choice={"type":"function","function":
{"name":<tool_name>}}`) is the wire-protocol equivalent of the
Claude-SDK backend's `allowed_tools=[...]` plus the prompt's "call
exactly once" instruction.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import BackendResult


@dataclass
class OpenAICompatBackend:
    base_url: str
    api_key: str = ""  # empty → omit Authorization header (local endpoints)
    extra_headers: dict[str, str] | None = None
    # Best-effort: each backend keeps its own URL normalization so callers
    # can pass `https://api.openai.com`, `…/v1`, or `…/v1/` interchangeably.
    name: str = field(default="openai_compat", init=False)

    def _endpoint(self) -> str:
        base = (self.base_url or "").rstrip("/")
        if not base:
            raise ValueError("OpenAICompatBackend: base_url is required")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        # If the user passed a root URL, assume /v1/chat/completions —
        # the de-facto OpenAI path. Self-hosted servers that diverge
        # should pass the full /v1 segment explicitly.
        return base + "/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            for k, v in self.extra_headers.items():
                h[k] = v
        return h

    def _build_body(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_parameters,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name},
            },
            "temperature": 0,
        }

    def _post_sync(self, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(), data=data, headers=self._headers(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))

    async def evaluate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
        timeout_s: float,
    ) -> BackendResult:
        body = self._build_body(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters,
        )
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._post_sync, body, timeout_s),
                timeout=timeout_s + 5,  # outer guard against executor hang
            )
        except asyncio.TimeoutError:
            return BackendResult(
                verdict_args=None,
                error=f"timeout after {timeout_s}s",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                detail = ""
            return BackendResult(
                verdict_args=None,
                error=f"HTTP {exc.code}: {detail}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except urllib.error.URLError as exc:
            return BackendResult(
                verdict_args=None,
                error=f"URL error: {exc.reason!r}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BackendResult(
                verdict_args=None,
                error=f"transport error: {exc!r}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        dt = (time.monotonic() - t0) * 1000
        # Parse tool-call result. Strict path:
        # data.choices[0].message.tool_calls[0].function.arguments (JSON string)
        try:
            choices = data.get("choices") or []
            msg = (choices[0] if choices else {}).get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return BackendResult(
                    verdict_args=None,
                    error=f"model did not call {tool_name}",
                    latency_ms=dt,
                )
            fn = (tool_calls[0] or {}).get("function") or {}
            args_raw = fn.get("arguments")
            if not isinstance(args_raw, str):
                # Some servers return already-parsed dict; accept both.
                if isinstance(args_raw, dict):
                    return BackendResult(
                        verdict_args=args_raw, error=None, latency_ms=dt
                    )
                return BackendResult(
                    verdict_args=None,
                    error="tool_call.arguments missing or wrong type",
                    latency_ms=dt,
                )
            try:
                verdict_args = json.loads(args_raw)
            except json.JSONDecodeError as exc:
                return BackendResult(
                    verdict_args=None,
                    error=f"tool_call.arguments not JSON: {exc!s}",
                    latency_ms=dt,
                )
            if not isinstance(verdict_args, dict):
                return BackendResult(
                    verdict_args=None,
                    error="tool_call.arguments not an object",
                    latency_ms=dt,
                )
            return BackendResult(
                verdict_args=verdict_args, error=None, latency_ms=dt
            )
        except Exception as exc:
            return BackendResult(
                verdict_args=None,
                error=f"response parse error: {exc!r}",
                latency_ms=dt,
            )
