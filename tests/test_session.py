import json
from pathlib import Path

from gadfly import session


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_load_returns_empty_when_no_path():
    ctx = session.load(None)
    assert ctx.recent_user_requests == []
    assert ctx.last_user_request is None  # convenience accessor stays
    assert ctx.last_assistant_plan is None
    assert ctx.recent_actions == []


def test_load_returns_empty_when_file_missing(tmp_path: Path):
    ctx = session.load(str(tmp_path / "nope.jsonl"))
    assert ctx.recent_user_requests == []
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
    assert ctx.recent_user_requests == ["fix the auth bug"]
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
    # The pure tool_result user-message must not be collected.
    assert ctx.recent_user_requests == ["implement feature X"]


def test_recent_user_requests_collects_multiple_oldest_first(tmp_path: Path):
    """The latest user message alone often misrepresents the goal — it's a
    clarification. Watchdog needs the trail to understand direction.
    """
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"message": {"role": "user", "content": "build a benchmark plan"}},
            {"message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
            {"message": {"role": "user", "content": [{"type": "tool_result", "content": "..."}]}},
            {"message": {"role": "user", "content": "focus on regression detection first"}},
            {"message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
            {"message": {"role": "user", "content": "actually use criterion not bench harness"}},
        ],
    )
    ctx = session.load(str(p))
    assert ctx.recent_user_requests == [
        "build a benchmark plan",
        "focus on regression detection first",
        "actually use criterion not bench harness",
    ]
    # tool_result user-messages stay filtered.
    assert all("tool_result" not in r for r in ctx.recent_user_requests)


def test_recent_user_requests_filters_service_tags(tmp_path: Path):
    """Regression: Claude Code injects /compact, /goal, command stdout and
    Stop-hook system-reminders as user-role messages. They were leaking
    into the watchdog prompt as if the user had typed them, so Haiku graded
    the agent against /compact instead of the real goal."""
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"message": {"role": "user", "content": "investigate marlin regression"}},
            {"message": {"role": "user", "content": "<command-name>/compact</command-name>\n<command-message>compact</command-message>"}},
            {"message": {"role": "user", "content": "<local-command-stdout>Compacted</local-command-stdout>"}},
            {"message": {"role": "user", "content": "<command-name>/goal</command-name>"}},
            {"message": {"role": "user", "content": "<system-reminder>\nA session-scoped Stop hook is now active …\n</system-reminder>"}},
            {"message": {"role": "user", "content": "now also profile EXL3 kernels"}},
        ],
    )
    ctx = session.load(str(p))
    assert ctx.recent_user_requests == [
        "investigate marlin regression",
        "now also profile EXL3 kernels",
    ]


def test_recent_user_requests_caps_at_max(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"message": {"role": "user", "content": f"msg-{i}"}}
            for i in range(10)
        ],
    )
    ctx = session.load(str(p))
    assert len(ctx.recent_user_requests) == session.MAX_USER_REQUESTS
    # Most recent are kept, in chronological order.
    assert ctx.recent_user_requests == [
        f"msg-{i}" for i in range(10 - session.MAX_USER_REQUESTS, 10)
    ]


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
    assert ctx.recent_user_requests == ["hi"]




# --- Per-file snapshot + trajectory ----------------------------------------


def _edit(file_path: str, new: str = "x", old: str = "") -> dict:
    return {
        "type": "tool_use",
        "name": "Edit",
        "input": {"file_path": file_path, "old_string": old, "new_string": new},
    }


def _asst(*tools: dict) -> dict:
    return {"message": {"role": "assistant", "content": list(tools)}}


def test_file_touch_trajectory_records_tool_and_path(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        _asst(_edit("a.py")),
        _asst(_edit("b.py")),
        _asst({"type": "tool_use", "name": "Bash",
               "input": {"command": "ls"}}),
        _asst(_edit("a.py")),
    ])
    ctx = session.load(str(p), distill=False)
    # Bash filtered out — only file-touching tools recorded.
    paths = [t[2] for t in ctx.file_touch_trajectory]
    assert paths == ["a.py", "b.py", "a.py"]
    # Action_index is the global tool_use index, not just file-touches.
    assert ctx.file_touch_trajectory[0][0] == 1
    assert ctx.file_touch_trajectory[1][0] == 2
    assert ctx.file_touch_trajectory[2][0] == 4
    # Tool names preserved.
    assert all(t[1] == "Edit" for t in ctx.file_touch_trajectory)


def test_file_touch_trajectory_caps_at_last_n(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [_asst(_edit(f"f{i}.py")) for i in range(20)])
    ctx = session.load(str(p), distill=False)
    assert len(ctx.file_touch_trajectory) == session.MAX_TRAJECTORY
    # The LAST 10 — files f10..f19.
    paths = [t[2] for t in ctx.file_touch_trajectory]
    assert paths[0] == "f10.py"
    assert paths[-1] == "f19.py"


def test_per_file_snapshot_reads_disk(tmp_path: Path):
    """Snapshot is the real on-disk content — that's what defeats
    edit-window blindness without any regex / heuristic."""
    proj = tmp_path / "proj"
    proj.mkdir()
    target = proj / "util.rs"
    target.write_text("pub fn helper(x: u32) -> u32 { x * 2 }\n")
    tpath = tmp_path / "t.jsonl"
    _write_jsonl(tpath, [_asst(_edit(str(target), new="content"))])

    out = session.extract_per_file_snapshots(
        [_asst(_edit(str(target)))],  # synthetic entries
        cwd=str(proj),
    )
    assert str(target) in out
    assert "pub fn helper" in out[str(target)]


def test_per_file_snapshot_picks_recently_touched_files(tmp_path: Path):
    proj = tmp_path / "p"
    proj.mkdir()
    for fname, body in [
        ("a.py", "def a(): pass\n"),
        ("b.py", "def b(): pass\n"),
        ("c.py", "def c(): pass\n"),
        ("d.py", "def d(): pass\n"),
        ("e.py", "def e(): pass\n"),
    ]:
        (proj / fname).write_text(body)
    entries = [_asst(_edit(str(proj / f))) for f in ("a.py", "b.py", "c.py", "d.py", "e.py")]
    out = session.extract_per_file_snapshots(entries, cwd=str(proj))
    # Capped at MAX_SNAPSHOT_FILES (3). Most recent wins.
    assert len(out) == session.MAX_SNAPSHOT_FILES
    paths = set(out.keys())
    assert str(proj / "e.py") in paths
    assert str(proj / "d.py") in paths
    assert str(proj / "c.py") in paths


def test_per_file_snapshot_truncates_huge_file(tmp_path: Path):
    """Files past the byte budget show head + truncation marker + tail.
    Marker is explicit — Haiku sees `[…truncated middle…]` and knows
    bytes were dropped (and which slice of the file is missing). This
    is a size-budget decision, NOT a content heuristic."""
    proj = tmp_path / "p"
    proj.mkdir()
    huge = proj / "big.py"
    body = "HEAD_MARKER\n" + ("x" * 40_000) + "\nTAIL_MARKER\n"
    huge.write_text(body)
    entries = [_asst(_edit(str(huge)))]
    out = session.extract_per_file_snapshots(entries, cwd=str(proj))
    snap = out[str(huge)]
    assert "HEAD_MARKER" in snap
    assert "TAIL_MARKER" in snap
    assert "truncated middle" in snap
    assert "bytes" in snap


def test_per_file_snapshot_refuses_paths_outside_cwd(tmp_path: Path):
    """Belt-and-braces against leaking arbitrary filesystem into prompts.
    Edit with file_path=/etc/passwd MUST NOT be read."""
    proj = tmp_path / "p"
    proj.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret content")
    entries = [_asst(_edit(str(outside)))]
    out = session.extract_per_file_snapshots(entries, cwd=str(proj))
    # outside isn't under cwd — refused.
    assert out == {}


def test_per_file_snapshot_swallows_read_errors(tmp_path: Path):
    """Missing file → no snapshot, no exception."""
    proj = tmp_path / "p"
    proj.mkdir()
    missing = proj / "does_not_exist.py"
    entries = [_asst(_edit(str(missing)))]
    out = session.extract_per_file_snapshots(entries, cwd=str(proj))
    assert out == {}


def test_per_file_snapshot_skips_oversize_files(tmp_path: Path):
    """Files >5MB on disk are skipped (avoid blowing up prompt
    composition / memory). Verifies the safety cap, not a heuristic
    about content."""
    proj = tmp_path / "p"
    proj.mkdir()
    big = proj / "binary.bin"
    big.write_bytes(b"x" * (6 * 1024 * 1024))  # 6MB
    entries = [_asst(_edit(str(big)))]
    out = session.extract_per_file_snapshots(entries, cwd=str(proj))
    assert out == {}


def test_load_populates_snapshots_and_trajectory_from_real_files(tmp_path: Path):
    """End-to-end: write actual file, fake a transcript that edits it,
    confirm ctx carries both trajectory and snapshot."""
    proj = tmp_path / "p"
    proj.mkdir()
    f = proj / "x.py"
    f.write_text("def hello(): return 1\n")
    tpath = tmp_path / "t.jsonl"
    _write_jsonl(tpath, [_asst(_edit(str(f)))])

    ctx = session.load(str(tpath), distill=False)
    ctx.cwd = str(proj)  # session.load() doesn't get cwd from transcript;
    # the hook normally sets it from the payload. In test we mimic by
    # re-running snapshot extraction:
    ctx.per_file_snapshots = session.extract_per_file_snapshots(
        [_asst(_edit(str(f)))], cwd=str(proj),
    )
    assert ctx.file_touch_trajectory == [(1, "Edit", str(f))]
    assert "def hello" in ctx.per_file_snapshots[str(f)]


# --- Active plan extraction -----------------------------------------------


def test_extract_active_plan_finds_approved_plan():
    entries = [
        {"message": {"role": "user", "content": "fix the bug"}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "looking into it"},
        ]}},
        {"message": {"role": "user", "content": (
            "## Approved Plan:\n\n"
            "Step 1: Read the file\n"
            "Step 2: Apply the patch\n"
            "Step 3: Add a test\n"
        )}},
    ]
    plan = session.extract_active_plan(entries)
    assert plan is not None
    assert "Step 1: Read the file" in plan
    assert "Step 3: Add a test" in plan


def test_extract_active_plan_takes_latest_when_multiple():
    entries = [
        {"message": {"role": "user", "content": "## Approved Plan:\nFirst plan"}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "doing first"},
        ]}},
        {"message": {"role": "user", "content": "## Approved Plan:\nSecond plan"}},
    ]
    plan = session.extract_active_plan(entries)
    assert plan == "Second plan"


def test_extract_active_plan_returns_none_when_absent():
    entries = [
        {"message": {"role": "user", "content": "fix the bug"}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "done"},
        ]}},
    ]
    assert session.extract_active_plan(entries) is None


def test_extract_active_plan_stops_at_system_reminder():
    """Plan body ends when the system-reminder closing block starts —
    we don't want to pull the EXITED PLAN MODE notice into the body."""
    entries = [
        {"message": {"role": "user", "content": (
            "## Approved Plan:\n"
            "PLAN_BODY_LINE_1\n"
            "PLAN_BODY_LINE_2\n"
            "<system-reminder>## Exited Plan Mode\nblah blah"
        )}},
    ]
    plan = session.extract_active_plan(entries)
    assert plan is not None
    assert "PLAN_BODY_LINE_1" in plan
    assert "PLAN_BODY_LINE_2" in plan
    # System-reminder content NOT included.
    assert "Exited Plan Mode" not in plan
    assert "blah blah" not in plan


def test_extract_active_plan_caps_at_8kb():
    body = "X" * 20000
    entries = [{"message": {"role": "user", "content": f"## Approved Plan:\n{body}"}}]
    plan = session.extract_active_plan(entries, max_bytes=8000)
    assert len(plan) <= 8000 + 100  # leeway for truncation marker
    assert "truncated" in plan


# --- Recent dialogue pairs + Read in trajectory ----------------------------


def test_recent_dialogue_pairs_extracts_last_n(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    entries = []
    for i in range(10):
        entries.append({"message": {"role": "assistant",
                                    "content": [{"type": "text",
                                                 "text": f"proposal {i}"}]}})
        entries.append({"message": {"role": "user",
                                    "content": f"reply {i}"}})
    _write_jsonl(p, entries)
    ctx = session.load(str(p), distill=False)
    assert len(ctx.recent_dialogue_pairs) == session.MAX_DIALOGUE_PAIRS
    # The LAST 5 (5..9).
    assert ctx.recent_dialogue_pairs[0] == ("proposal 5", "reply 5")
    assert ctx.recent_dialogue_pairs[-1] == ("proposal 9", "reply 9")


def test_recent_dialogue_pairs_empty_when_no_pairs(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        {"message": {"role": "assistant",
                     "content": [{"type": "text", "text": "hi"}]}},
        # No user reply → not a pair.
    ])
    ctx = session.load(str(p), distill=False)
    assert ctx.recent_dialogue_pairs == []


def test_read_action_enters_trajectory(tmp_path: Path):
    """Read enters trajectory so the watchdog sees the verification
    step before judging a downstream Edit."""
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        _asst({"type": "tool_use", "name": "Read",
               "input": {"file_path": "backend/handler.go"}}),
        _asst({"type": "tool_use", "name": "Read",
               "input": {"file_path": "backend/types.go"}}),
        _asst(_edit("frontend/ui.ts")),
    ])
    ctx = session.load(str(p), distill=False)
    paths = [(t[1], t[2]) for t in ctx.file_touch_trajectory]
    assert ("Read", "backend/handler.go") in paths
    assert ("Read", "backend/types.go") in paths
    assert ("Edit", "frontend/ui.ts") in paths


def test_read_targets_become_snapshot_candidates(tmp_path: Path):
    """When the agent reads upstream contracts then edits downstream,
    the upstream files become snapshot candidates so Haiku can verify
    the contract directly."""
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "handler.go").write_text("func ServeUpdate(...) {}\n")
    (proj / "types.go").write_text("type UpdateReq struct{}\n")
    (proj / "ui.ts").write_text("// frontend\n")
    entries = [
        _asst({"type": "tool_use", "name": "Read",
               "input": {"file_path": str(proj / "handler.go")}}),
        _asst({"type": "tool_use", "name": "Read",
               "input": {"file_path": str(proj / "types.go")}}),
        _asst(_edit(str(proj / "ui.ts"))),
    ]
    out = session.extract_per_file_snapshots(entries, cwd=str(proj))
    paths = set(out.keys())
    # All three relevant files appear in the snapshot pool — Read'd
    # contracts AND the downstream Edit target.
    assert str(proj / "ui.ts") in paths  # most recent (Edit)
    assert str(proj / "types.go") in paths
    assert str(proj / "handler.go") in paths
    assert "func ServeUpdate" in out[str(proj / "handler.go")]
    assert "type UpdateReq" in out[str(proj / "types.go")]


# --- Snapshot: bigger target cap + edit-region-centered window -----------


def test_target_snapshot_gets_larger_cap_than_reference(tmp_path: Path):
    """The file the current Edit targets gets MAX_TARGET_SNAPSHOT_BYTES;
    other files (recently touched but not the current target) get the
    smaller MAX_SNAPSHOT_BYTES."""
    proj = tmp_path / "p"
    proj.mkdir()
    # Target file: 50KB — well under 60KB target cap, well over 25KB
    # reference cap.
    target = proj / "target.py"
    target.write_text("HEADER\n" + "x" * 50_000 + "\nFOOTER\n")
    # Reference file: same size — would head+tail under reference cap.
    ref = proj / "ref.py"
    ref.write_text("REFHEAD\n" + "x" * 50_000 + "\nREFTAIL\n")
    entries = [
        _asst(_edit(str(ref))),
        _asst(_edit(str(target))),  # target is the LATEST
    ]
    out = session.extract_per_file_snapshots(
        entries, cwd=str(proj),
        target_file=str(target),
    )
    target_snap = out[str(target)]
    ref_snap = out[str(ref)]
    # Target fits whole — has both HEADER and FOOTER without truncation.
    assert "HEADER" in target_snap
    assert "FOOTER" in target_snap
    assert "truncated" not in target_snap
    # Reference is over its 25KB cap — head+tail with marker.
    assert "REFHEAD" in ref_snap
    assert "REFTAIL" in ref_snap
    assert "truncated" in ref_snap


def test_target_snapshot_edit_centered_when_over_budget(tmp_path: Path):
    """When the target file exceeds MAX_TARGET_SNAPSHOT_BYTES, the
    snapshot is sliced around `target_anchor` (the Edit's old_string)
    so the agent's edit area is visible — including any nearby newly
    added definitions. Regression for the prompts.py case where
    _shorten_path was added in the middle of an 80KB file and dropped
    by head+tail truncation."""
    proj = tmp_path / "p"
    proj.mkdir()
    target = proj / "big.py"
    # File over 60KB total: head + middle (with FRESHLY_ADDED) + tail.
    head_pad = "head_pad_line\n" * 1000  # ~14KB
    middle_pad = "middle_pad_line\n" * 2000  # ~32KB
    tail_pad = "tail_pad_line\n" * 1000  # ~14KB
    body = (
        head_pad
        + "FRESHLY_ADDED_FUNCTION\n"
        + "EDIT_ANCHOR_HERE\n"
        + middle_pad
        + tail_pad
    )
    target.write_text(body)
    assert len(body) > session.MAX_TARGET_SNAPSHOT_BYTES, (
        f"fixture needs to exceed cap: {len(body)} <= {session.MAX_TARGET_SNAPSHOT_BYTES}"
    )
    entries = [_asst(_edit(str(target), old="EDIT_ANCHOR_HERE", new="x"))]
    out = session.extract_per_file_snapshots(
        entries,
        cwd=str(proj),
        target_file=str(target),
        target_anchor="EDIT_ANCHOR_HERE",
    )
    snap = out[str(target)]
    # The edit region AND the freshly-added function (right next to it)
    # must both be visible. Without the centered slice they would be
    # in the middle band and dropped.
    assert "EDIT_ANCHOR_HERE" in snap
    assert "FRESHLY_ADDED_FUNCTION" in snap
    # And the truncation marker indicates head/tail were cut, not the
    # edit area.
    assert "before the edit window" in snap or "after the edit window" in snap


def test_reference_snapshot_keeps_head_tail_when_over_budget(tmp_path: Path):
    """Reference (non-target) files keep the old head+tail behavior —
    no anchor available, no edit-centered slicing."""
    proj = tmp_path / "p"
    proj.mkdir()
    ref = proj / "ref.py"
    body = "HEAD\n" + ("x" * 30_000) + "\nTAIL\n"
    ref.write_text(body)
    entries = [_asst(_edit(str(ref)))]
    out = session.extract_per_file_snapshots(
        entries, cwd=str(proj),
        # No target_file → ref.py is just a reference snapshot.
    )
    snap = out[str(ref)]
    assert "HEAD" in snap and "TAIL" in snap
    assert "truncated" in snap


# --- Recent bash actions trajectory ----------------------------------------


def _bash_tool_use(tid: str, cmd: str) -> dict:
    return {"type": "tool_use", "name": "Bash", "id": tid,
            "input": {"command": cmd}}


def _user_tool_result(tid: str, payload) -> dict:
    return {"message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": payload}
    ]}}


def test_recent_bash_actions_pairs_tool_use_with_result(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        _asst(_bash_tool_use("t1", "curl -X POST http://host/api")),
        _user_tool_result("t1", [{"type": "text",
                                   "text": '{"stdout":"{\\"error\\":\\"400\\"}", "exit_code":0, "stderr":""}'}]),
        _asst(_bash_tool_use("t2", "grep xgrammar /tmp/log")),
        _user_tool_result("t2", [{"type": "text",
                                   "text": '{"stdout":"", "exit_code":1, "stderr":""}'}]),
    ])
    ctx = session.load(str(p), distill=False)
    assert len(ctx.recent_bash_actions) == 2
    idx0, cmd0, ec0, out0, err0 = ctx.recent_bash_actions[0]
    idx1, cmd1, ec1, out1, err1 = ctx.recent_bash_actions[1]
    assert idx0 == 1 and "curl" in cmd0
    assert ec0 == 0 and "400" in out0
    assert idx1 == 2 and "grep" in cmd1
    assert ec1 == 1 and out1 == ""


def test_recent_bash_actions_caps_at_max_n(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    entries = []
    for i in range(15):
        tid = f"t{i}"
        entries.append(_asst(_bash_tool_use(tid, f"echo {i}")))
        entries.append(_user_tool_result(tid, [{"type": "text",
                                                 "text": f'{{"stdout":"{i}", "exit_code":0, "stderr":""}}'}]))
    _write_jsonl(p, entries)
    ctx = session.load(str(p), distill=False)
    assert len(ctx.recent_bash_actions) == session.MAX_BASH_ACTIONS
    # Last N, so first kept is i=15-MAX_BASH_ACTIONS.
    first_kept = 15 - session.MAX_BASH_ACTIONS
    assert f"echo {first_kept}" in ctx.recent_bash_actions[0][1]
    assert "echo 14" in ctx.recent_bash_actions[-1][1]


def test_recent_bash_actions_truncates_long_output(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    huge = "X" * 20_000
    _write_jsonl(p, [
        _asst(_bash_tool_use("t1", "verbose-cmd")),
        _user_tool_result("t1", [{"type": "text",
                                   "text": '{"stdout":"' + huge + '", "exit_code":0, "stderr":""}'}]),
    ])
    ctx = session.load(str(p), distill=False)
    _, _, _, stdout, _ = ctx.recent_bash_actions[0]
    # Head+tail with marker. Bounded by total budget.
    assert len(stdout) <= session.MAX_BASH_OUTPUT_TOTAL + 100
    assert "truncated" in stdout


def test_recent_bash_actions_handles_dict_result(tmp_path: Path):
    """Some transcript versions deliver tool_result.content as a raw
    dict {stdout, stderr, exit_code} rather than a text-block list."""
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        _asst(_bash_tool_use("t1", "ls")),
        _user_tool_result("t1", {"stdout": "a\nb", "exit_code": 0,
                                  "stderr": ""}),
    ])
    ctx = session.load(str(p), distill=False)
    _, _, ec, out, _ = ctx.recent_bash_actions[0]
    assert ec == 0
    assert "a\nb" in out


def test_recent_bash_actions_empty_when_no_bash(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [_asst(_edit("a.py"))])
    ctx = session.load(str(p), distill=False)
    assert ctx.recent_bash_actions == []


def test_recent_bash_actions_orphan_call_has_empty_output(tmp_path: Path):
    """Some bash calls in a stuck/aborted run have no matching
    tool_result. They still appear in the trajectory but with empty
    output and exit_code=None."""
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        _asst(_bash_tool_use("orphan", "cat huge_file")),
        # No matching tool_result.
    ])
    ctx = session.load(str(p), distill=False)
    assert len(ctx.recent_bash_actions) == 1
    _, _, ec, out, err = ctx.recent_bash_actions[0]
    assert ec is None
    assert out == ""
    assert err == ""
