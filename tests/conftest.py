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
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GADFLY_DISABLE", raising=False)
    # Disable journal in unit tests by default — it would otherwise call
    # the real Claude Code CLI via the SDK on every hook invocation.
    # Tests that exercise journal flow set GADFLY_JOURNAL=1 explicitly
    # (and inject a mocked run_query in journal.update_for_action).
    monkeypatch.setenv("GADFLY_JOURNAL", "0")
    # Phase-2 default flipped to "1" in production. In unit tests we
    # neutralise both flags so behaviour is unambiguous: tests that
    # exercise journal flow opt in explicitly.
    monkeypatch.setenv("GADFLY_JOURNAL_VERDICT", "0")
    # Phase-B (historian priors) default-on in production. Off in
    # unit tests so prompts don't get unexpected priors blocks.
    monkeypatch.setenv("GADFLY_HISTORIAN_PRIORS", "0")
    # Trail (longitudinal breadcrumb path) is default-on in production.
    # Off in unit tests so hook-level tests can assert single-record
    # audit logs; tests that exercise the trail flow set GADFLY_TRAIL=1
    # explicitly (and mock run_query the way test_trail.py does).
    monkeypatch.setenv("GADFLY_TRAIL", "0")
    monkeypatch.setenv("GADFLY_TRAIL_FEEDBACK", "0")
    # SHADOW mode swallows additionalContext in production. In unit
    # tests we EXPLICITLY validate that flag verdicts surface — so we
    # must neutralise the env var the .env file leaked in via
    # hook._load_env_file_once().
    monkeypatch.setenv("GADFLY_SHADOW", "0")
