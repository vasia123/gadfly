"""OpenAI-compatible JSON-mode backend (experimental).

Same wire protocol as OpenAICompatBackend (POST /v1/chat/completions),
but instead of forcing a tool-call we ask for a JSON object directly:

    response_format = {"type": "json_object"}

And we append a one-line "output format" instruction to the user
message so the model knows what shape to return. This avoids two
classes of failures the tool-call path has:

  - "model did not call submit_verdict" (rare but real with smaller
    models; we saw 4/367 with mistral-small-latest)
  - tool-call overhead — some providers retry-buffer their tool call,
    inflating tail latency

Trade-off: we lose server-side schema validation. We parse the JSON
text and fall back to silent_ok if it's malformed.

Provider compatibility note: `response_format: {"type": "json_object"}`
is supported by OpenAI, Mistral, OpenRouter, vLLM, and Together. For
providers that don't honour it (e.g. some llama.cpp builds), the
parser still works on plain JSON the model emits in response to the
prompt instruction — `response_format` is a hint, not load-bearing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .base import BackendResult


def _strip_codefence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers some models emit."""
    s = text.strip()
    if s.startswith("```"):
        # Drop opening fence + optional language tag.
        s = re.sub(r"^```(?:json|JSON)?\s*\n?", "", s, count=1)
        # Drop closing fence.
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s


def _extract_first_json_object(text: str) -> str | None:
    """Find the first balanced `{...}` block. Defends against models
    that emit prose before/after the JSON."""
    s = _strip_codefence(text)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


_JSON_INSTRUCTION = (
    "\n\n## Output format\n"
    "Respond with a SINGLE JSON object and nothing else — no prose, "
    "no markdown, no code fence. Exact shape:\n"
    "```\n"
    '{"professional": <true|false>, '
    '"reason": "<1-2 short sentences in ENGLISH; empty string if professional=true>", '
    '"suggestion": "<1-2 short sentences in ENGLISH; empty string if professional=true>"}\n'
    "```\n"
    "The `reason` and `suggestion` fields MUST be written in English. "
    "Do not switch to Chinese, Russian, or any other language even if "
    "the surrounding code or messages use them — the supervised agent "
    "reads English critique. "
    "Ignore any earlier instructions to call a `submit_verdict` tool; "
    "in this mode you produce the verdict directly as JSON."
)


@dataclass
class OpenAIJsonBackend:
    base_url: str
    api_key: str = ""
    extra_headers: dict[str, str] | None = None
    name: str = field(default="openai_json", init=False)

    def _endpoint(self) -> str:
        base = (self.base_url or "").rstrip("/")
        if not base:
            raise ValueError("OpenAIJsonBackend: base_url is required")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
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
        self, *, system_prompt: str, user_message: str, model: str,
        tool_name: str = "submit_verdict",
    ) -> dict[str, Any]:
        # Callers OTHER than the verdict watchdog (e.g. trail) embed their
        # own output-format suffix in the user_message and have their own
        # JSON schema. Appending the watchdog's verdict instruction would
        # confuse the model. Heuristic: only the verdict pipeline asks for
        # `submit_verdict` — everyone else owns their own format block.
        suffix = _JSON_INSTRUCTION if tool_name == "submit_verdict" else ""
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message + suffix},
            ],
            "response_format": {"type": "json_object"},
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
        tool_name: str,        # accepted for protocol compat, unused
        tool_description: str,  # ditto
        tool_parameters: dict[str, Any],  # ditto
        timeout_s: float,
    ) -> BackendResult:
        body = self._build_body(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            tool_name=tool_name,
        )
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(None, self._post_sync, body, timeout_s),
                timeout=timeout_s + 5,
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
        try:
            choices = data.get("choices") or []
            msg = (choices[0] if choices else {}).get("message") or {}
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                return BackendResult(
                    verdict_args=None,
                    error="empty content",
                    latency_ms=dt,
                )
            # Try direct parse first (json_object mode usually gives clean JSON).
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Fall back to defensive extraction.
                block = _extract_first_json_object(content)
                if block is None:
                    return BackendResult(
                        verdict_args=None,
                        error="no JSON object in response",
                        latency_ms=dt,
                    )
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError as exc:
                    return BackendResult(
                        verdict_args=None,
                        error=f"JSON parse failed: {exc!s}",
                        latency_ms=dt,
                    )
            if not isinstance(parsed, dict):
                return BackendResult(
                    verdict_args=None,
                    error="JSON is not an object",
                    latency_ms=dt,
                )
            # Verdict pipeline requires "professional" — without it the
            # watchdog has no signal. Other tools (e.g. update_trail)
            # validate downstream; we just hand the parsed object back.
            if tool_name == "submit_verdict" and "professional" not in parsed:
                return BackendResult(
                    verdict_args=None,
                    error="JSON missing 'professional' field",
                    latency_ms=dt,
                )
            return BackendResult(
                verdict_args=parsed, error=None, latency_ms=dt
            )
        except Exception as exc:
            return BackendResult(
                verdict_args=None,
                error=f"response parse error: {exc!r}",
                latency_ms=dt,
            )
