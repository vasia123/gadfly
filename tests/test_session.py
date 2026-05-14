import json
from pathlib import Path

from gadfly import session


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_load_returns_empty_when_no_path():
    ctx = session.load(None)
    assert ctx.last_user_request is None
    assert ctx.last_assistant_plan is None
    assert ctx.recent_actions == []


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    ctx = session.load(str(tmp_path / "nope.jsonl"))
    assert ctx.last_user_request is None


def test_load_extracts_user_assistant_and_actions(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"message": {"role": "user", "content": "fix the auth bug"}},
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll edit auth.py"},
                        {"type": "tool_use", "name": "Edit", "input": {"file_path": "auth.py"}},
                    ],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "ok"}],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "now running tests"},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
                    ],
                }
            },
        ],
    )
    ctx = session.load(str(p))
    assert ctx.last_user_request == "fix the auth bug"
    assert ctx.last_assistant_plan == "now running tests"
    # Both tool_uses present, in chronological order.
    assert len(ctx.recent_actions) == 2
    assert ctx.recent_actions[0].startswith("Edit(")
    assert ctx.recent_actions[1].startswith("Bash:")


def test_load_skips_pure_tool_result_user_messages(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"message": {"role": "user", "content": "implement feature X"}},
            {"message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "done"}],
                }
            },
        ],
    )
    ctx = session.load(str(p))
    # The pure tool_result user-message must not become last_user_request.
    assert ctx.last_user_request == "implement feature X"


def test_recent_actions_include_edit_diff_so_haiku_sees_prior_context(tmp_path: Path):
    """Regression: Haiku used to flag the *current* edit because the symbol
    it referenced was introduced one Edit earlier in the same series, and
    the prior Edit was summarized as just `Edit(viewer.py)` without content.
    Now the diff is in the summary."""
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"message": {"role": "user", "content": "wire in the toggle"}},
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "step 1"},
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "file_path": "viewer.py",
                                "old_string": "let currentSession = null;",
                                "new_string": "let currentSession = null;\nlet openDetails = new Set();",
                            },
                        },
                    ],
                }
            },
        ],
    )
    ctx = session.load(str(p))
    assert len(ctx.recent_actions) == 1
    action = ctx.recent_actions[0]
    assert action.startswith("Edit(viewer.py)")
    # The new content must appear so Haiku can see openDetails was defined.
    assert "openDetails = new Set" in action


def test_recent_actions_include_bash_command(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv sync --extra dev"},
                        }
                    ],
                }
            }
        ],
    )
    ctx = session.load(str(p))
    assert ctx.recent_actions == ["Bash: uv sync --extra dev"]


def test_load_tolerates_malformed_lines(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        "\n".join(
            [
                "not json at all",
                json.dumps({"message": {"role": "user", "content": "hi"}}),
                "{broken json",
            ]
        )
    )
    ctx = session.load(str(p))
    assert ctx.last_user_request == "hi"
