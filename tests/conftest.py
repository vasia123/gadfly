"""Shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_dir = tmp_path / "log"
    monkeypatch.setenv("GADFLY_LOG_DIR", str(log_dir))
    # Re-import log module so it picks up the new env var. The module reads
    # the env at import time, so we patch its module-level constant directly.
    from gadfly import log as log_mod

    monkeypatch.setattr(log_mod, "LOG_ROOT", log_dir)
    return log_dir


@pytest.fixture(autouse=True)
def _clear_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GADFLY_DISABLE", raising=False)
