"""Verdict dataclass and conversion to Claude Code hook output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Verdict:
    professional: bool
    reason: str
    suggestion: str

    @classmethod
    def from_tool_input(cls, tool_input: dict[str, Any]) -> Verdict:
        """Build from the dict Haiku passed to submit_verdict.

        We trust the API to validate against input_schema, but be defensive
        about missing/None fields anyway — Haiku occasionally omits empty
        strings even when they're 'required'.
        """
        return cls(
            professional=bool(tool_input.get("professional", True)),
            reason=str(tool_input.get("reason") or ""),
            suggestion=str(tool_input.get("suggestion") or ""),
        )

    @classmethod
    def silent_ok(cls) -> Verdict:
        return cls(professional=True, reason="", suggestion="")

    def to_hook_output(self) -> dict[str, Any] | None:
        """Build the Claude Code PostToolUse hook stdout JSON.

        Returns None when there's nothing to say (professional=true) — the
        hook should then exit 0 with no output, so Claude Code does not
        inject anything into the agent's context.
        """
        if self.professional:
            return None
        msg = f"Watchdog (gadfly): {self.reason}".rstrip()
        if self.suggestion:
            msg += f" Suggestion: {self.suggestion}"
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": msg,
            }
        }

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "professional": self.professional,
            "reason": self.reason,
            "suggestion": self.suggestion,
        }
