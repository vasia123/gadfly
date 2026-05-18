from gadfly import prompts


def test_build_user_message_includes_all_sections():
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "auth.py", "old_string": "pass", "new_string": "return None  # TODO"},
        tool_response={"success": True},
        recent_user_requests=["implement validate_token properly"],
        last_assistant_plan="I will edit auth.py",
        recent_actions=["Read(auth.py)", "Bash: pytest -q"],
    )
    assert "User messages in this conversation" in msg
    assert "implement validate_token properly" in msg
    assert "Agent's reasoning immediately before this action" in msg
    assert "Recent prior actions" in msg
    assert "Read(auth.py)" in msg
    assert "tool_name\nEdit" in msg
    assert "auth.py" in msg
    assert "TODO" in msg


def test_build_user_message_handles_missing_context():
    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "git commit --no-verify -m fix"},
        tool_response={"exit_code": 0, "stdout": ""},
        recent_user_requests=[],
        last_assistant_plan=None,
        recent_actions=[],
    )
    assert "(unavailable)" in msg
    assert "--no-verify" in msg


def test_submit_verdict_schema_is_well_formed():
    schema = prompts.SUBMIT_VERDICT_INPUT_SCHEMA
    assert set(schema) == {"professional", "reason", "suggestion"}
    assert schema["professional"] is bool
    assert schema["reason"] is str
    assert schema["suggestion"] is str


def test_submit_verdict_description_mentions_default():
    # The description is the only prompt-engineering knob Haiku sees for the
    # tool itself; it must remind it that "true" is the default.
    assert "default true" in prompts.SUBMIT_VERDICT_DESCRIPTION.lower() or \
        "default" in prompts.SUBMIT_VERDICT_DESCRIPTION.lower()


def test_system_prompt_treats_reasoning_as_part_of_evaluation():
    """Regression for the mini_decoder_layer_forward / proxy false-negative:
    Haiku looked at the innocuous `grep` tool call and ignored the
    rationalization paragraph that came right before it. The prompt must
    spell out that the reasoning itself is in scope."""
    p = prompts.SYSTEM_PROMPT.lower()
    assert "reasoning" in p
    # Concrete rationalization patterns we want Haiku to recognise.
    assert "proxy" in p
    assert "harder than" in p
    assert "rationalization" in p
    # And the marker for the agent.
    assert "rationalization:" in p


def test_user_message_marks_assistant_plan_as_part_of_evaluation():
    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "grep foo bar"},
        tool_response={"stdout": ""},
        recent_user_requests=["implement Tier 2 bench"],
        last_assistant_plan="I'll use mini_decoder as a proxy because the real one is harder than needed.",
        recent_actions=[],
    )
    # The plan section must explicitly tell Haiku the plan is evaluable.
    assert "PART OF what you are evaluating" in msg
    # And the task statement must mention reasoning, not just the action.
    assert "reasoning" in msg.lower()


def test_journal_system_prompt_includes_load_bearing_sections():
    """Phase-2 prompt extends the legacy one with four new sections that
    every journal-aware verdict must respect. If any marker disappears,
    the calibration we wired up to fix the cuda-layernorm echo-chamber
    regresses."""
    p = prompts.SYSTEM_PROMPT_JOURNAL.lower()
    # All legacy markers still present.
    assert "rationalization:" in p
    assert "symptom fix:" in p
    # Journal-reader section.
    assert "session journal" in p
    assert "workstreams" in p
    assert "flag_history" in p
    # Repetition rule — the load-bearing fix for echo-chamber.
    assert "repetition rule" in p
    assert "agent_pushed_back" in p
    assert "stay silent" in p
    # Rationalization cross-check via journal notes.
    assert "cross-check" in p
    assert "notes" in p
    # Drift is informational.
    assert "drift" in p
    assert "do not flag drift" in p


def test_build_user_message_journal_mode_renders_journal_block():
    from gadfly.journal import Journal, Workstream, FlagEvent, Drift

    j = Journal(
        root_goal="fix cuda-layernorm regression and ship",
        workstreams=[
            Workstream(
                id="cuda-layernorm",
                title="cuda-layernorm 4.3us regression",
                status="in-progress",
                origin="action #34",
                notes="capture-state dispatch landed in normalization.rs:92",
                watchdog_flags=5,
                flag_history=[
                    FlagEvent(
                        action_index=40,
                        reason="symptom fix, not root cause",
                        marker="symptom",
                        agent_pushed_back=True,
                        pushback="edits did land, watchdog reads stale ctx",
                    ),
                ],
                last_touched=88,
            )
        ],
        drift=Drift(initial_workstream_ids=["marlin", "tiered-bench"], observations="cuda-layernorm emerged unplanned"),
        action_index=88,
    )
    msg = prompts.build_user_message(
        tool_name="Write",
        tool_input={"file_path": "MEMORY.md", "content": "summary"},
        tool_response={"success": True},
        recent_user_requests=["IGNORED in journal mode"],
        last_assistant_plan="Updating memory after fix landed",
        recent_actions=["IGNORED in journal mode"],
        journal=j,
    )
    assert "Session journal" in msg
    assert "cuda-layernorm 4.3us regression" in msg
    assert "in-progress" in msg
    assert "agent pushed back" in msg
    assert "REPETITION RULE" in msg
    assert "RATIONALIZATION cross-check" in msg
    # Journal mode drops legacy blocks.
    assert "User messages in this conversation" not in msg
    assert "Recent prior actions" not in msg


def test_build_user_message_renders_verdict_patterns_block():
    """E2: when verdict_patterns are provided in journal-mode, the prompt
    surfaces them so Haiku can self-calibrate against past flag values."""
    from gadfly.journal import Journal, Workstream
    from gadfly.project_state import VerdictPattern

    j = Journal(
        root_goal="x",
        workstreams=[Workstream(id="w", title="t", status="open")],
    )
    patterns = {
        "p1": VerdictPattern(
            fingerprint="symptom-composable",
            marker="symptom",
            sample_reason="Symptom fix: composable does not exist yet",
            total_flags=5,
            value_score=-4,  # strongly negative → false-positive class
        ),
        "p2": VerdictPattern(
            fingerprint="rationalization-no-evidence",
            marker="rationalization",
            sample_reason="Rationalization: claimed without evidence",
            total_flags=2,
            value_score=2,
        ),
    }
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "x.vue", "old_string": "a", "new_string": "b"},
        tool_response={"success": True},
        recent_user_requests=[],
        last_assistant_plan="refactor",
        recent_actions=[],
        journal=j,
        verdict_patterns=patterns,
    )
    assert "Verdict patterns" in msg
    assert "composable" in msg
    assert "score=-4" in msg or "-4" in msg
    assert "score=+2" in msg or "+2" in msg


def test_system_prompt_includes_reflection_mode():
    """Reflection-mode rule: serious flags (Symptom fix / Rationalization)
    use Reflexion-style provocation rather than imperative correction.
    The agent gets a chance to verbalise its own reasoning, which is a
    far stronger learning signal than 'do X instead of Y'."""
    p = prompts.SYSTEM_PROMPT.lower()
    assert "reflection mode" in p
    assert "reflexion" in p  # cite the source
    # Required structure of the reflection prompt:
    assert "why did you" in p
    assert "step-by-step" in p
    assert "execute" in p
    # Must explicitly distinguish trivial vs reflection mode
    assert "trivial fix mode" in p


def test_system_prompt_journal_inherits_reflection_mode():
    """SYSTEM_PROMPT_JOURNAL extends SYSTEM_PROMPT — reflection mode rules
    must also apply there. Otherwise Phase 2 verdicts regress to
    imperative-only suggestions."""
    p = prompts.SYSTEM_PROMPT_JOURNAL.lower()
    assert "reflection mode" in p
    assert "why did you" in p


def test_journal_system_prompt_includes_track_record_rule():
    """E2 prompt addition: tells Haiku how to read value_score."""
    p = prompts.SYSTEM_PROMPT_JOURNAL.lower()
    assert "value_score" in p
    assert "track record" in p or "over-calibrated" in p
    assert "do not flag" in p or "stay silent" in p


def test_build_user_message_falls_back_to_legacy_when_journal_empty():
    from gadfly.journal import empty_journal

    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response={"stdout": ""},
        recent_user_requests=["do X"],
        last_assistant_plan=None,
        recent_actions=[],
        journal=empty_journal(),
    )
    # Falls through because the journal has no workstreams.
    assert "User messages in this conversation" in msg


def test_system_prompt_includes_root_cause_discipline():
    """Regression: the rubric must spell out symptom-vs-cause explicitly,
    not just mention it in passing — otherwise Haiku won't reliably flag
    `except: pass` and `if x is None: x = default` patches as unprofessional."""
    p = prompts.SYSTEM_PROMPT.lower()
    # Top-level concept appears.
    assert "root cause" in p
    assert "symptom" in p
    # Reflective stop-and-think instruction is in.
    assert "pause" in p or "stop" in p or "ask yourself" in p
    # Several of the concrete red-flag patterns we want Haiku to recognise.
    assert "if x is none" in p or "x = default" in p
    assert "except: pass" in p or "except exception" in p
    assert "sleep" in p
    assert "type: ignore" in p or "noqa" in p
    # The output convention so the agent can spot this class of feedback.
    assert 'symptom fix:' in p or "symptom fix" in p


# --- Per-file edit history + explicit redirect rule ------------------------


def test_journal_system_prompt_includes_explicit_redirect_rule():
    """New rule must mention the key markers so a future prompt rewrite
    that drops the multilingual semantic clause fails this test."""
    p = prompts.SYSTEM_PROMPT_JOURNAL
    assert "EXPLICIT USER REDIRECT" in p
    assert "first action" in p.lower()
    # Multilingual examples — load-bearing for the "semantic, not lexical"
    # rule. If someone deletes them, fail loudly.
    assert "глянь" in p or "посмотри" in p
    assert "regarde" in p


def test_build_user_message_journal_mode_includes_latest_user_msg():
    from gadfly.journal import Journal, Workstream, Drift
    j = Journal(
        root_goal="historian work",
        workstreams=[Workstream(id="w1", title="finish historian",
                                status="in-progress", last_touched=10)],
        drift=Drift(),
        action_index=10,
    )
    msg = prompts.build_user_message(
        tool_name="Bash", tool_input={"command": "ls"},
        tool_response={"exit_code": 0},
        recent_user_requests=["глянь на vllm-rust сессию"],
        last_assistant_plan=None, recent_actions=[],
        journal=j,
        latest_user_message_verbatim="глянь на vllm-rust сессию",
    )
    assert "Most recent user message" in msg
    assert "глянь на vllm-rust сессию" in msg


def test_build_user_message_journal_mode_includes_per_file_snapshot():
    from gadfly.journal import Journal, Workstream, Drift
    j = Journal(
        root_goal="historian work",
        workstreams=[Workstream(id="w1", title="finish historian",
                                status="in-progress", last_touched=10)],
        drift=Drift(),
        action_index=10,
    )
    snapshots = {
        "src/a.py": "def helper():\n    return 42\n\ndef caller():\n    return helper()\n",
    }
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "src/a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["edit a.py"],
        last_assistant_plan=None, recent_actions=[],
        journal=j,
        per_file_snapshots=snapshots,
    )
    assert "src/a.py" in msg
    assert "def helper" in msg
    assert "Current on-disk snapshots" in msg
    # Verbatim source must be rendered inside a code fence so Haiku
    # reads it as content, not as instructions.
    assert "```" in msg


def test_build_user_message_legacy_mode_includes_per_file_snapshot():
    snapshots = {"src/a.py": "def helper(): pass\n"}
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "src/a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["edit a.py"],
        last_assistant_plan=None, recent_actions=["Edit(src/a.py)"],
        per_file_snapshots=snapshots,
    )
    assert "Current on-disk snapshots" in msg
    assert "def helper" in msg


def test_build_user_message_renders_file_touch_trajectory():
    trajectory = [
        (1, "Edit", "src/a.py"),
        (2, "Write", "src/b.py"),
        (7, "Edit", "src/a.py"),
    ]
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "src/a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["edit a.py"],
        last_assistant_plan=None, recent_actions=["Edit(src/a.py)"],
        file_touch_trajectory=trajectory,
    )
    assert "File-touch trajectory" in msg
    assert "#1 Edit" in msg and "src/a.py" in msg
    assert "#7 Edit" in msg


def test_build_user_message_without_snapshots_or_redirect_emits_no_blocks():
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "src/a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["edit a.py"],
        last_assistant_plan=None, recent_actions=[],
    )
    assert "Current on-disk snapshots" not in msg
    assert "File-touch trajectory" not in msg
    assert "Most recent user message" not in msg


# --- Active plan + journal trimming ---------------------------------------


def test_system_prompt_has_test_deletion_non_rule():
    p = prompts.SYSTEM_PROMPT
    assert "Deleting tests whose unit-under-test" in p
    assert "absent from its target module" in p


def test_system_prompt_has_active_plan_non_rule():
    p = prompts.SYSTEM_PROMPT
    assert "Active plan" in p
    assert "ExitPlanMode" in p


def test_journal_extensions_reference_active_plan_block():
    p = prompts.SYSTEM_PROMPT_JOURNAL
    assert "ACTIVE PLAN" in p
    assert "authoritative" in p.lower()


def test_build_user_message_renders_active_plan_legacy():
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["do the plan"],
        last_assistant_plan=None, recent_actions=[],
        active_plan="Step 1: refactor X\nStep 2: delete tests of removed Y\n",
    )
    assert "Active plan (approved by the user via ExitPlanMode)" in msg
    assert "Step 1: refactor X" in msg
    assert "delete tests of removed Y" in msg


def test_build_user_message_renders_active_plan_journal_mode():
    from gadfly.journal import Journal, Workstream, Drift
    j = Journal(
        root_goal="execute the plan",
        workstreams=[Workstream(id="w1", title="run the plan",
                                status="in-progress", last_touched=5)],
        drift=Drift(),
        action_index=5,
    )
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["proceeding"],
        last_assistant_plan=None, recent_actions=[],
        journal=j,
        active_plan="Step 1: do X\nStep 2: do Y\n",
    )
    assert "Active plan" in msg
    assert "Step 1: do X" in msg


def test_render_journal_trims_done_workstreams_to_oneliners():
    from gadfly.journal import Journal, Workstream, Drift
    j = Journal(
        root_goal="multi-phase work",
        workstreams=[
            Workstream(id="w-old", title="old long-ago workstream",
                       status="done", notes="A" * 500, last_touched=1),
            Workstream(id="w-now", title="current workstream",
                       status="in-progress", notes="B" * 500, last_touched=10),
        ],
        drift=Drift(),
        action_index=10,
    )
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["go"],
        last_assistant_plan=None, recent_actions=[],
        journal=j,
    )
    # Active workstream: full rendering including notes.
    assert "current workstream" in msg
    assert "B" * 200 in msg  # notes present
    # Done workstream: id + title only.
    assert "w-old" in msg
    assert "old long-ago workstream" in msg
    # Done workstream notes (long "AAAA…" block) NOT rendered.
    assert "A" * 200 not in msg


def test_render_journal_caps_total_at_12():
    from gadfly.journal import Journal, Workstream, Drift
    workstreams = [
        Workstream(id=f"w{i}", title=f"workstream {i}", status="in-progress",
                   last_touched=i)
        for i in range(25)
    ]
    j = Journal(root_goal="big project", workstreams=workstreams,
                drift=Drift(), action_index=25)
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "a.py", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["go"],
        last_assistant_plan=None, recent_actions=[],
        journal=j,
    )
    # Only the 12 most-recent (by last_touched DESC: w24..w13) appear.
    assert "workstream 24" in msg
    assert "workstream 13" in msg
    # w12 and below — dropped by the cap.
    assert "workstream 12" not in msg
    assert "workstream 0" not in msg


# --- Round 4: dialogue + bash output + Read ---------------------------


def test_journal_extensions_includes_multi_turn_authorization_rule():
    p = prompts.SYSTEM_PROMPT_JOURNAL
    assert "MULTI-TURN AUTHORIZATION" in p
    # Block describes the structural mechanism — no hardcoded phrases.
    assert "Recent dialogue" in p


def test_build_user_message_renders_recent_dialogue_journal_mode():
    from gadfly.journal import Journal, Workstream, Drift
    j = Journal(
        root_goal="install runtime",
        workstreams=[Workstream(id="w1", title="node-runtime",
                                status="in-progress", last_touched=5)],
        drift=Drift(),
        action_index=5,
    )
    pairs = [
        ("Поставлю Volta, она держит pinned-node на проекте", "погоди"),
        ("ок, тогда подумаю об альтернативах", "что предлагаешь?"),
        ("Использую nvm-prefix, Volta — открытый вопрос", "ставим"),
    ]
    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "volta install node@22"},
        tool_response={"stdout": "installed"},
        recent_user_requests=["ставим"],
        last_assistant_plan="Ставлю Volta, env будет изменён",
        recent_actions=[],
        journal=j,
        recent_dialogue_pairs=pairs,
    )
    assert "Recent dialogue" in msg
    # Earliest pair surfaces the original Volta proposal Haiku needs.
    assert "Поставлю Volta" in msg
    assert "ставим" in msg


def test_build_user_message_renders_recent_dialogue_legacy_mode():
    pairs = [("propose A", "no, do B"), ("ok will do B", "go")]
    msg = prompts.build_user_message(
        tool_name="Bash",
        tool_input={"command": "do_b"},
        tool_response={"exit_code": 0},
        recent_user_requests=["go"],
        last_assistant_plan="executing B",
        recent_actions=[],
        recent_dialogue_pairs=pairs,
    )
    assert "Recent dialogue" in msg
    assert "propose A" in msg
    assert "no, do B" in msg


def test_recent_dialogue_per_side_truncation():
    long_asst = "A" * 5000
    long_user = "U" * 1000
    block = prompts._render_recent_dialogue([(long_asst, long_user)])
    assert "Asst: " in block
    # Assistant content clipped past 800c.
    assert long_asst not in block
    # User content clipped past 200c.
    assert long_user not in block


def test_format_tool_response_head_tail_for_large_output():
    """The Prettier-postprocess case: large stdout where success
    signal lives at the end (chained `&& ls`). Head+tail preserves
    both ends."""
    long_stdout = "ERROR " * 1000 + "FAIL_MIDDLE_MARKER" * 500 + "SUCCESS_TAIL_MARKER"
    resp = {"stdout": long_stdout, "exit_code": 1}
    out = prompts._format_tool_response(resp)
    assert "ERROR" in out  # head preserved
    assert "SUCCESS_TAIL_MARKER" in out  # tail preserved
    assert "truncated" in out  # marker present
    # Middle filler dropped under budget pressure.
    assert "FAIL_MIDDLE_MARKER" * 500 not in out


def test_format_tool_response_short_input_unchanged():
    short = {"stdout": "ok", "exit_code": 0}
    out = prompts._format_tool_response(short)
    assert "ok" in out
    assert "truncated" not in out


def test_head_tail_truncate_below_budget_returns_as_is():
    assert prompts._head_tail_truncate("hello", total=100, head=50, tail=50) == "hello"


def test_head_tail_truncate_over_budget_splits():
    body = "HEAD_PART_" + "x" * 10000 + "_TAIL_PART"
    out = prompts._head_tail_truncate(body, total=200, head=100, tail=100)
    assert "HEAD_PART_" in out
    assert "_TAIL_PART" in out
    assert "truncated" in out


def test_snapshot_render_marks_current_edit_target():
    """v-calc regression: when multiple file snapshots are present and
    they happen to share function names (separate scopes), Haiku must
    not confuse them. The CURRENT EDIT TARGET marker steers Haiku to
    the right file."""
    snapshots = {
        "frontend/components/admin/CoatingsImport.vue": "function downloadDictionary() {}\n",
        "frontend/components/admin/MaterialsImport.vue": "function downloadDictionary() {}\n",
    }
    block = prompts._render_per_file_snapshots(
        snapshots,
        current_target="frontend/components/admin/MaterialsImport.vue",
    )
    # The target file gets the marker; the sibling does not.
    target_line = [l for l in block.splitlines() if "MaterialsImport.vue" in l and l.startswith("###")][0]
    sibling_line = [l for l in block.splitlines() if "CoatingsImport.vue" in l and l.startswith("###")][0]
    assert "CURRENT EDIT TARGET" in target_line
    assert "CURRENT EDIT TARGET" not in sibling_line


def test_snapshot_render_no_target_marker_when_unspecified():
    snapshots = {"a.py": "def x(): pass\n"}
    block = prompts._render_per_file_snapshots(snapshots)
    assert "CURRENT EDIT TARGET" not in block


def test_build_user_message_threads_current_target_through():
    snapshots = {
        "a.vue": "function shared() {}\n",
        "b.vue": "function shared() {}\n",
    }
    msg = prompts.build_user_message(
        tool_name="Edit",
        tool_input={"file_path": "a.vue", "old_string": "x", "new_string": "y"},
        tool_response={"success": True},
        recent_user_requests=["edit a"],
        last_assistant_plan=None, recent_actions=[],
        per_file_snapshots=snapshots,
    )
    a_heading = [l for l in msg.splitlines() if l.startswith("### a.vue")][0]
    b_heading = [l for l in msg.splitlines() if l.startswith("### b.vue")][0]
    assert "CURRENT EDIT TARGET" in a_heading
    assert "CURRENT EDIT TARGET" not in b_heading


def test_snapshot_render_uses_relative_paths_when_cwd_given():
    """Abs paths bloat tokens and read like noise. When cwd is set,
    snapshot heading + trajectory bullets render relative paths."""
    snapshots = {
        "/home/vasis/work/v-calc/frontend/components/MaterialsImport.vue":
            "function downloadDictionary() {}\n",
    }
    block = prompts._render_per_file_snapshots(
        snapshots,
        current_target="/home/vasis/work/v-calc/frontend/components/MaterialsImport.vue",
        cwd="/home/vasis/work/v-calc",
    )
    # Relative path shown, absolute prefix gone.
    assert "frontend/components/MaterialsImport.vue" in block
    assert "/home/vasis/work/v-calc/frontend" not in block


def test_trajectory_render_uses_relative_paths_when_cwd_given():
    traj = [(7, "Edit", "/home/u/proj/src/a.py")]
    block = prompts._render_file_touch_trajectory(traj, cwd="/home/u/proj")
    assert "src/a.py" in block
    assert "/home/u/proj/src/a.py" not in block


def test_snapshot_render_falls_back_to_absolute_without_cwd():
    snapshots = {"/abs/path/x.py": "content\n"}
    block = prompts._render_per_file_snapshots(snapshots)
    # No cwd → unchanged absolute path.
    assert "/abs/path/x.py" in block


def test_snapshot_render_keeps_path_when_outside_cwd():
    """If a snapshotted file lives outside cwd (rare, but defensive),
    fall back to the absolute path rather than emitting nonsense."""
    snapshots = {"/tmp/outside.py": "content\n"}
    block = prompts._render_per_file_snapshots(snapshots, cwd="/home/u/proj")
    assert "/tmp/outside.py" in block
