"""Backend protocol.

A Backend submits ONE forced tool-call to a model and returns the
parsed arguments. It does not assemble prompts (watchdog.py does that),
it does not interpret the verdict semantics (verdict.py does that).
Backends only own the wire protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendResult:
    """Outcome of one model call.

    On success: verdict_args is the dict the model passed to the forced
    tool-call (`{"professional": bool, "reason": str, "suggestion": str}`),
    error is None.

    On any failure (HTTP error, timeout, model didn't call the tool,
    malformed arguments, etc.): verdict_args is None and error carries a
    short human-readable explanation that the watchdog logs and maps to
    silent_ok in the hook.
    """

    verdict_args: dict[str, Any] | None
    error: str | None
    latency_ms: float | None = None


class Backend(Protocol):
    """One model call, structured-output via forced tool-call."""

    name: str

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
    ) -> BackendResult: ...
