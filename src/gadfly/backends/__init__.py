"""Backend abstraction: one model call → structured verdict.

The watchdog assembles prompts; the backend submits a forced tool-call
and returns the parsed arguments. Two backends ship today:

  - ClaudeSDKBackend     — claude-agent-sdk + user's Claude Code CLI
                           subscription (default).
  - OpenAICompatBackend  — stdlib HTTP against any /v1/chat/completions
                           endpoint (OpenAI, Anthropic OAI-compat,
                           OpenRouter, vLLM, llama.cpp, Ollama, …).
"""

from __future__ import annotations

from typing import Any

from .base import Backend, BackendResult
from .claude_sdk import ClaudeSDKBackend
from .openai_compat import OpenAICompatBackend


def select_backend(name: str, **kwargs: Any) -> Backend:
    """Factory by short name. CLI / config layer entry point."""
    if name == "claude_sdk":
        return ClaudeSDKBackend(**kwargs)
    if name == "openai_compat":
        return OpenAICompatBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r}")


__all__ = [
    "Backend",
    "BackendResult",
    "ClaudeSDKBackend",
    "OpenAICompatBackend",
    "select_backend",
]
